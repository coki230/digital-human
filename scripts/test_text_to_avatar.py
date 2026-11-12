# scripts/test_text_to_avatar.py
"""
完整本地测试：文本输入 → 数字人视频
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import time
import numpy as np
import cv2
from loguru import logger

# ====================== 项目根目录 ======================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ====================== 导入模块 ======================
from src.audio.pipeline import AudioPipeline, AudioConfig
from src.avatar.engine import FlashHeadEngine
from src.llm.ollama_client import OllamaClient
from src.session.manager import SessionManager
from src.llm.streaming_utils import StreamingSentenceBuffer

# ====================== 配置区（请按实际修改） ======================
CONFIG = {
    # Ollama
    "ollama_model": "qwen3.8:latest",          # 改成你本地的模型
    "ollama_base_url": "http://localhost:11434",

    # FlashHead
    "flashhead_ckpt": "/home/coki/humanAI/dev1/model/SoulX-FlashHead-1_3B",
    "wav2vec_dir": "/home/coki/humanAI/dev1/model/wav2vec2-base-960h",
    "model_type": "lite",

    # 参考图
    "avatar_image": "assets/tx2.jpeg",   # 改成你的图片路径

    # CosyVoice
    "tts_model_dir": "/home/coki/humanAI/dev1/model/CosyVoice-300M-SFT",

    # 输出
    "output_dir": "outputs/test_videos",
    "fps": 25,
}


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """简单重采样（线性插值，够用）"""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def save_video(frames_list: list[np.ndarray], output_path: str, fps: int = 25):
    """把帧列表保存为 mp4"""
    if not frames_list:
        logger.warning("No frames to save.")
        return

    # 合并所有帧
    all_frames = np.concatenate(frames_list, axis=0)  # (T, H, W, 3)
    height, width = all_frames.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in all_frames:
        # OpenCV 使用 BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    logger.info(f"Video saved to: {output_path}  ({len(all_frames)} frames)")


def main():
    logger.info("========== 数字人本地测试启动 ==========")

    # 1. 初始化各组件
    logger.info("Initializing components...")

    audio_cfg = AudioConfig(
        tts_model_dir=CONFIG["tts_model_dir"],
        asr_model_size="base",          # 本次测试用不到 ASR
        asr_language="zh",
    )
    audio_pipe = AudioPipeline(audio_cfg)

    avatar_engine = FlashHeadEngine(
        ckpt_dir=CONFIG["flashhead_ckpt"],
        wav2vec_dir=CONFIG["wav2vec_dir"],
        model_type=CONFIG["model_type"],
    )

    llm = OllamaClient(
        base_url=CONFIG["ollama_base_url"],
        model=CONFIG["ollama_model"],
        temperature=0.7,
        system_prompt="你是一个友好、自然、口语化的数字人助手。回答简洁，尽量用短句，适合语音播报。",
    )

    manager = SessionManager(
        audio_pipeline=audio_pipe,
        avatar_engine=avatar_engine,
        llm_client=llm,
    )

    # 2. 创建会话
    session_id = "test_session_001"
    manager.create_session(
        session_id=session_id,
        avatar_image_path=CONFIG["avatar_image"],
        seed=42,
    )
    logger.info(f"Session created with avatar: {CONFIG['avatar_image']}")

    # 3. 准备输出目录
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("输入文本后回车，数字人会生成对应视频（输入 q 退出）")
    print("="*60 + "\n")

    try:
        user_input = "你好我是coki，很高兴认识你"
    except Exception as e:
        print("11111", e)
        print("\n退出")

    logger.info(f"User: {user_input}")
    start_time = time.time()

    # 收集所有生成的帧
    all_frames = []
    sentence_count = 0

    # 使用流式分句版 chat
    try:
        for frames in manager.chat(session_id, user_input):
            if frames is not None and len(frames) > 0:
                all_frames.append(frames)
                sentence_count += 1
                logger.info(f"Received frames from sentence #{sentence_count}, shape={frames.shape}")
    except Exception as e:
        logger.exception(f"Generation failed: {e}")


    elapsed = time.time() - start_time
    logger.info(f"Total generation time: {elapsed:.2f}s")

    # 保存视频
    if all_frames:
        timestamp = time.strftime("%H%M%S")
        output_path = str(output_dir / f"{timestamp}.mp4")
        save_video(all_frames, output_path, fps=CONFIG["fps"])
        print(f"→ 视频已保存: {output_path}\n")
    else:
        print("→ 没有生成有效视频帧\n")


    # 清理
    manager.close_session(session_id)
    logger.info("Test finished.")


if __name__ == "__main__":
    main()