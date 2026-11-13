# src/avatar/engine.py
"""
SoulX-FlashHead 封装引擎
支持多会话、流式喂音频、打断
"""

from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Generator
import numpy as np
import torch
from loguru import logger

# # ============ 根据你实际路径修改 ============
FLASHHEAD_ROOT = Path("third_party").resolve()   # 改成你的实际路径
sys.path.insert(0, str(FLASHHEAD_ROOT))
# # ==========================================

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    get_infer_params,
)
from .state import SessionState
from .rocm_utils import get_device, synchronize, empty_cache


class FlashHeadEngine:
    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        model_type: str = "lite",
        device: str = "cuda",
    ):
        self.ckpt_dir = ckpt_dir
        self.wav2vec_dir = wav2vec_dir
        self.model_type = model_type
        self.device = get_device(device)

        logger.info(f"Loading FlashHead [{model_type}] ...")
        self.pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir,
        )
        self.infer_params = get_infer_params()
        self.sample_rate = self.infer_params["sample_rate"]  # 16000
        self.tgt_fps = self.infer_params["tgt_fps"]  # 25
        self.frame_num = self.infer_params["frame_num"]  # 33
        self.motion_frames_num = self.infer_params["motion_frames_num"]  # get_pipeline 时已写入
        self.slice_len = self.frame_num - self.motion_frames_num  # 例如 28

        self.slice_audio_len = self.slice_len * self.sample_rate // self.tgt_fps
        self.cached_audio_duration = self.infer_params["cached_audio_duration"]  # 8
        self.cached_audio_length = self.sample_rate * self.cached_audio_duration

        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps  # 200
        self.audio_start_idx = self.audio_end_idx - self.frame_num  # 167

        logger.info(
            f"FlashHead stream params: frame_num={self.frame_num}, "
            f"motion_frames_num={self.motion_frames_num}, slice_len={self.slice_len}, "
            f"slice_audio_len={self.slice_audio_len}, cached={self.cached_audio_length}"
        )

        # 多会话状态
        self.sessions: Dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        cond_image_path: str,
        seed: int = 42,
        use_face_crop: bool = True,
    ) -> SessionState:
        """创建新会话，初始化参考图与 motion 状态"""
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, recreating.")
            self.close_session(session_id)

        logger.info(f"Creating session {session_id} with image: {cond_image_path}")
        get_base_data(
            self.pipeline,
            cond_image_path_or_dir=cond_image_path,
            base_seed=seed,
            use_face_crop=use_face_crop,
        )

        state = SessionState(
            session_id=session_id,
            cond_image_path=cond_image_path,
            seed=seed,
        )
        # 官方：先填满 0
        state.audio_dq = deque([0.0] * self.cached_audio_length, maxlen=self.cached_audio_length)
        state.pending_samples = 0
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            empty_cache()
            logger.info(f"Session {session_id} closed.")

    def interrupt(self, session_id: str):
        """打断当前生成"""
        state = self.sessions.get(session_id)
        if state:
            state.interrupt()
            logger.info(f"Session {session_id} interrupted.")
    # ------------------------------------------------------------------
    # 核心推理接口
    # ------------------------------------------------------------------

    def feed_audio(
            self,
            session_id: str,
            audio_chunk: np.ndarray,
            sample_rate: int = 16000,
    ) -> Optional[tuple]:
        """
        返回 (frames, consumed_audio) 或 None
        - frames: (T,H,W,3) uint8
        - consumed_audio: 本次推理对应的 float32 音频（长度 = slice_audio_len）
        """
        state = self.sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id} not found")

        if state.interrupted:
            return None

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio_chunk.size == 0:
            return None

        state.audio_dq.extend(audio_chunk.tolist())
        state.pending_samples += len(audio_chunk)
        state.update_active()

        if state.pending_samples < self.slice_audio_len:
            return None

        state.pending_samples -= self.slice_audio_len

        # 本次 slice 对应的音频：环形缓存的最后 slice_audio_len 个点
        # 官方是整段 dq 送去 embedding，但保存预览时用的是当前 slice 的原始波形
        dq_list = list(state.audio_dq)
        consumed_audio = np.array(dq_list[-self.slice_audio_len:], dtype=np.float32)

        state.is_generating = True
        try:
            audio_array = np.array(state.audio_dq, dtype=np.float32)
            audio_embedding = get_audio_embedding(
                self.pipeline,
                audio_array,
                self.audio_start_idx,
                self.audio_end_idx,
            )
            video = run_pipeline(self.pipeline, audio_embedding)
            video = video[self.motion_frames_num:]

            if isinstance(video, torch.Tensor):
                frames = video.cpu()  # 保持 tensor，方便官方 save
            else:
                frames = torch.from_numpy(np.asarray(video))

            return frames, consumed_audio

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
        finally:
            state.is_generating = False

    def feed_audio_stream(self, session_id, audio_generator, sample_rate=16000):
        state = self.sessions.get(session_id)
        if state is None:
            raise ValueError(...)

        state.interrupted = False
        for chunk in audio_generator:
            if state.interrupted:
                break
            result = self.feed_audio(session_id, chunk, sample_rate)
            if result is not None:
                yield result  # (frames, consumed_audio)


# 简单测试
if __name__ == "__main__":
    engine = FlashHeadEngine(
        ckpt_dir="/home/coki/humanAI/dev1/model/SoulX-FlashHead-1_3B",
        wav2vec_dir="/home/coki/humanAI/dev1/model/wav2vec2-base-960h",
        model_type="lite",
    )
    print("Engine loaded successfully.")