# src/audio/asr.py
"""
ASR using faster-whisper.
支持中英，ROCm 可跑（通过 ctranslate2）。
"""

from __future__ import annotations
from typing import Optional, Generator
import numpy as np
from faster_whisper import WhisperModel


class ASR:
    def __init__(
        self,
        model_size: str = "base",          # tiny / base / small / medium / large-v3
        device: str = "cpu",              # ROCm 下用 "cuda"（HIP 兼容）
        compute_type: str = "int8",     # 显存紧可用 "int8"
        language: Optional[str] = "zh",    # None 自动检测，"zh" 强制中文
        beam_size: int = 5,
    ):
        self.language = language
        self.beam_size = beam_size

        print(f"[ASR] Loading faster-whisper {model_size} on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            # download_root="./models/whisper",  # 可自定义缓存路径
        )
        print("[ASR] Ready.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        完整音频转写
        audio: float32, mono, 16kHz
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,                 # 内置 VAD，进一步过滤静音
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        text = "".join([seg.text for seg in segments]).strip()
        return text

    def transcribe_stream(
        self,
        audio_generator: Generator[np.ndarray, None, None],
    ) -> Generator[str, None, None]:
        """
        简易流式接口（实际生产中建议用更成熟的流式方案）
        这里采用“累积 + 定期解码”策略
        """
        buffer = []
        for chunk in audio_generator:
            buffer.append(chunk)
            # 简单策略：每累计约 1.5~2 秒解码一次
            if sum(len(c) for c in buffer) > 16000 * 1.6:
                audio = np.concatenate(buffer)
                text = self.transcribe(audio)
                if text:
                    yield text
                buffer = []          # 或保留最后一点做 overlap

        # 最后一段
        if buffer:
            audio = np.concatenate(buffer)
            text = self.transcribe(audio)
            if text:
                yield text


# 测试
if __name__ == "__main__":
    import soundfile as sf

    asr = ASR(model_size="base", device="cpu", language="zh")
    audio, sr = sf.read("test.wav")
    print(asr.transcribe(audio))