# scripts/test_text_to_avatar.py
"""
完整本地测试：文本输入 → LLM → TTS → FlashHead → 带音频的 mp4
对齐官方 gradio_app_streaming 的保存方式（视频帧 + wav 混流）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

# ====================== 项目根目录 ======================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ====================== 导入模块 ======================
from src.audio.pipeline import AudioPipeline, AudioConfig
from src.avatar.engine import FlashHeadEngine
from src.avatar.video_utils import save_audio_wav, save_video_with_audio
from src.llm.ollama_client import OllamaClient
from src.session.manager import SessionManager

# ====================== 配置区 ======================
CONFIG = {
    # Ollama
    "ollama_model": "qwen3.8:latest",
    "ollama_base_url": "http://localhost:11434",

    # FlashHead
    "flashhead_ckpt": "/home/coki/humanAI/dev1/model/SoulX-FlashHead-1_3B",
    "wav2vec_dir": "/home/coki/humanAI/dev1/model/wav2vec2-base-960h",
    "model_type": "lite",

    # 参考图
    "avatar_image": "assets/tx2.jpeg",

    # CosyVoice
    "tts_model_dir": "/home/coki/humanAI/dev1/model/CosyVoice-300M-SFT",

    # 输出
    "output_dir": "outputs/test_videos",
}


def main():
    logger.info("========== 数字人本地测试启动 ==========")

    # 1. 初始化组件
    logger.info("Initializing components...")

    audio_cfg = AudioConfig(
        tts_model_dir=CONFIG["tts_model_dir"],
        asr_model_size="base",
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
        system_prompt=(
            "你是一个友好、自然、口语化的数字人助手。"
            "回答简洁，尽量用短句，适合语音播报。"
        ),
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

    # 3. 输出目录
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 测试输入（可改成 input() 做交互）
    user_input = "你好我是coki，很高兴认识你"
    # user_input = input("[你] ").strip()
    # if not user_input or user_input.lower() in ("q", "quit", "exit"):
    #     manager.close_session(session_id)
    #     return

    logger.info(f"User: {user_input}")
    start_time = time.time()

    # 5. 跑完整链路（chat 内部会写 ctx.last_frames / last_tts_audio）
    frame_count = 0
    try:
        for frames in manager.chat(session_id, user_input):
            if frames is not None:
                frame_count += 1
                shape = getattr(frames, "shape", None)
                logger.info(f"Received chunk #{frame_count}, shape={shape}")
    except Exception as e:
        logger.exception(f"Generation failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Total generation time: {elapsed:.2f}s")

    # 6. 保存：官方方式（帧 list + wav → ffmpeg 混流）
    ctx = manager.contexts.get(session_id)
    frames_list = getattr(ctx, "last_frames", None) or []
    audio_16k = getattr(ctx, "last_tts_audio", None)
    audio_sr = getattr(ctx, "last_tts_sr", 16000) or 16000
    fps = getattr(avatar_engine, "tgt_fps", 25)

    if frames_list and audio_16k is not None and len(audio_16k) > 0:
        ts = time.strftime("%Y%m%d-%H%M%S")
        wav_path = str(output_dir / f"tts_{ts}.wav")
        mp4_path = str(output_dir / f"res_{ts}.mp4")

        save_audio_wav(np.asarray(audio_16k, dtype=np.float32), wav_path, sample_rate=audio_sr)
        save_video_with_audio(frames_list, mp4_path, wav_path, fps=fps)

        print(f"→ 音频: {wav_path}")
        print(f"→ 视频: {mp4_path}")
        logger.info(f"Saved with audio: {mp4_path}")
    elif frames_list:
        logger.warning("有视频帧但没有 TTS 音频，跳过混流（只会静音）")
        print("→ 有画面无音频：请检查 SessionManager 是否写入 ctx.last_tts_audio")
    else:
        print("→ 没有生成有效视频帧")

    # 7. 清理
    manager.close_session(session_id)
    logger.info("Test finished.")


if __name__ == "__main__":
    main()