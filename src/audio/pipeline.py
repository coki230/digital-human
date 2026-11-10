# src/audio/pipeline.py
"""
统一音频管道：VAD → ASR → (外部LLM) → TTS
专为实时数字人设计，支持流式输入/输出与打断。
"""
from __future__ import annotations

# ===== 必须放在所有 import 之前 =====
import os
os.environ["CT2_USE_MKL"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ===================================

import numpy as np
import threading
from typing import Optional, Callable, Generator
from dataclasses import dataclass

from src.audio.vad import VAD
from src.audio.asr import ASR
from src.audio.tts import TTS


@dataclass
class AudioConfig:
    # 采样率
    sample_rate: int = 16000

    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 400
    vad_device: str = "cpu"

    # ASR
    asr_model_size: str = "base"  # tiny / base / small / medium
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: Optional[str] = "zh"

    # TTS
    tts_model_dir: str = "/home/coki/humanAI/dev1/model/CosyVoice-300M-SFT"
    tts_device: str = "cuda"
    tts_fp16: bool = True
    tts_prompt_speech: Optional[str] = None  # 零样本音色路径
    tts_prompt_text: str = "希望你以后能够做的比我还好呦。"
    tts_speaker: str = "中文女"

    # 管道行为
    speech_pad_ms: int = 300  # 语音前后保留的静音 padding
    max_speech_duration_s: float = 30.0  # 单次说话最长时长（防异常）


class AudioPipeline:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.cfg = config or AudioConfig()
        self.sample_rate = self.cfg.sample_rate

        logger.info("Initializing AudioPipeline...")

        # 2. ASR
        logger.info("Initializing ASR...")
        self.asr = ASR(
            model_size=self.cfg.asr_model_size,
            device=self.cfg.asr_device,
            # compute_type=self.cfg.asr_compute_type,
            language=self.cfg.asr_language,
        )

        logger.info("Initializing VAD...")
        # 1. VAD
        self.vad = VAD(
            threshold=self.cfg.vad_threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.cfg.vad_min_speech_ms,
            min_silence_duration_ms=self.cfg.vad_min_silence_ms,
            device=self.cfg.vad_device,
        )


        # 3. TTS
        logger.info("Initializing TTS...")
        self.tts = TTS(
            model_dir=self.cfg.tts_model_dir,
            # device=self.cfg.tts_device,
            # fp16=self.cfg.tts_fp16,
        )

        # 运行时状态
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking: bool = False
        self._interrupt_flag = threading.Event()
        self._lock = threading.Lock()

        logger.info("AudioPipeline ready.")

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self):
        """重置所有状态（新会话 / 打断后调用）"""
        with self._lock:
            self.vad.reset()
            self._speech_buffer.clear()
            self._is_speaking = False
            self._interrupt_flag.clear()
        logger.debug("AudioPipeline reset.")

    def interrupt(self):
        """立即打断当前 TTS 播放与生成"""
        self._interrupt_flag.set()
        logger.info("AudioPipeline interrupted.")

    def is_interrupted(self) -> bool:
        return self._interrupt_flag.is_set()

    # ------------------------------------------------------------------
    # 输入端：麦克风流 → 文本
    # ------------------------------------------------------------------

    def process_audio_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """
        处理单个音频块（来自麦克风）。
        返回：完整一句话的识别结果（结束时才返回），否则返回 None。

        典型用法：
            while True:
                chunk = mic.read()
                text = pipeline.process_audio_chunk(chunk)
                if text:
                    # 送入 LLM
        """
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        is_speech = self.vad.is_speech(chunk)

        with self._lock:
            if is_speech:
                if not self._is_speaking:
                    self._is_speaking = True
                    logger.debug("Speech started.")
                self._speech_buffer.append(chunk)
            else:
                if self._is_speaking:
                    # 检测到静音，认为一句话结束
                    self._is_speaking = False
                    if self._speech_buffer:
                        full_audio = np.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()

                        # 长度保护
                        duration = len(full_audio) / self.sample_rate
                        if duration > self.cfg.max_speech_duration_s:
                            logger.warning(f"Speech too long ({duration:.1f}s), truncated.")
                            full_audio = full_audio[:int(self.cfg.max_speech_duration_s * self.sample_rate)]

                        # ASR
                        try:
                            text = self.asr.transcribe(full_audio, sample_rate=self.sample_rate)
                            text = text.strip()
                            if text:
                                logger.info(f"ASR Result: {text}")
                                return text
                        except Exception as e:
                            logger.error(f"ASR failed: {e}")
                    return None
        return None

    def listen_and_transcribe(
            self,
            audio_stream: Generator[np.ndarray, None, None],
            on_partial: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        更高层的接口：持续监听音频流，yield 完整句子。
        """
        for chunk in audio_stream:
            if self.is_interrupted():
                break
            text = self.process_audio_chunk(chunk)
            if text:
                yield text

    # ------------------------------------------------------------------
    # 输出端：文本 → 流式音频
    # ------------------------------------------------------------------

    def text_to_speech_stream(
            self,
            text: str,
            prompt_speech_path: Optional[str] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        流式 TTS，yield 音频 chunk (float32)。
        支持中途 interrupt()。
        """
        self._interrupt_flag.clear()

        try:
            for chunk in self.tts.text_2_audio(
                    text=text,
                    speaker=self.cfg.tts_speaker,
            ):
                if self.is_interrupted():
                    logger.info("TTS stream interrupted.")
                    break
                yield chunk
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise

    def text_to_speech(self, text: str, prompt_speech_path: Optional[str] = None) -> np.ndarray:
        """非流式完整合成（调试用）"""
        chunks = list(self.text_to_speech_stream(text, prompt_speech_path))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @property
    def tts_sample_rate(self) -> int:
        return self.tts.get_sample_rate()

    def get_status(self) -> dict:
        return {
            "is_speaking": self._is_speaking,
            "buffer_chunks": len(self._speech_buffer),
            "interrupted": self.is_interrupted(),
        }


# ------------------------------------------------------------------
# 快速测试
# ------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.add(lambda msg: print(msg, end=""))

    config = AudioConfig(
        asr_model_size="base",
        asr_language="zh",
        tts_model_dir="/home/coki/humanAI/dev1/model/CosyVoice-300M-SFT",  # 改成你的实际路径
    )
    pipeline = AudioPipeline(config)

    # 测试 TTS 流式
    print("\n=== TTS Stream Test ===")
    text = "你好，我是你的数字人助手，很高兴为你服务。"
    for i, chunk in enumerate(pipeline.text_to_speech_stream(text)):
        print(f"chunk {i}: shape={chunk.shape}, max={chunk.max():.3f}")
        if i >= 5:  # 只打印前几块
            break

    print("\nAudioPipeline test done.")