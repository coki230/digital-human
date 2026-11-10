# src/audio/vad.py
"""
Voice Activity Detection using Silero VAD.
轻量、准确，CPU/ROCm 都友好。
"""

from __future__ import annotations
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps, VADIterator
from typing import Optional, Tuple


class VAD:
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        device: str = "cpu",          # Silero 推荐先跑 CPU，延迟更稳
    ):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.device = torch.device(device)

        # 加载 Silero VAD 模型
        self.model = load_silero_vad(
            onnx=False  # 如果你想用 onnx 加速，可以改为 True
        )
        self.model.to(self.device)
        self.model.eval()

        # 初始化流式迭代器
        self.vad_iterator = VADIterator(
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

    def reset(self):
        """重置流式状态（打断或新会话时调用）"""
        self.vad_iterator.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        判断当前音频块是否包含语音
        audio_chunk: float32, shape (n,), 16kHz, 范围 [-1, 1]
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Silero 需要 torch tensor
        tensor = torch.from_numpy(audio_chunk).to(self.device)
        speech_dict = self.vad_iterator(tensor, return_seconds=False)
        return speech_dict is not None

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        return_seconds: bool = False,
    ) -> list:
        """离线完整音频获取语音时间戳（调试用）"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        tensor = torch.from_numpy(audio)
        return get_speech_timestamps(
            tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=return_seconds,
        )


# 简单测试
if __name__ == "__main__":
    import soundfile as sf

    vad = VAD(threshold=0.5)
    print("VAD 初始化成功！")