# src/audio/tts.py
"""
TTS using CosyVoice (支持流式输出，中文效果优秀)。
需要先安装 CosyVoice 官方仓库。
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Generator, Optional, Union
import numpy as np


# ========== 根据你实际安装路径修改 ==========
COSYVOICE_ROOT = Path("/path/to/CosyVoice")          # 改成你的 CosyVoice 路径
sys.path.append(str(COSYVOICE_ROOT))
sys.path.append(str(COSYVOICE_ROOT / "third_party/Matcha-TTS"))
# ==========================================

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav


class TTS:
    def __init__(
        self,
        model_dir: str = "pretrained_models/CosyVoice2-0.5B",  # 或 CosyVoice-300M
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.device = device
        self.sample_rate = 22050          # CosyVoice 默认输出采样率

        print(f"[TTS] Loading CosyVoice from {model_dir}...")
        # CosyVoice2 推荐（更新、效果更好）
        self.model = CosyVoice2(
            model_dir,
            load_jit=False,
            load_trt=False,
            fp16=fp16,
        )
        print("[TTS] Ready.")

    def synthesize(
        self,
        text: str,
        prompt_speech_path: Optional[str] = None,
        prompt_text: str = "希望你以后能够做的比我还好呦。",
        speaker: str = "中文女",          # 零样本时可忽略
    ) -> np.ndarray:
        """
        非流式完整合成，返回 float32 numpy 数组 (n,)
        """
        if prompt_speech_path and os.path.exists(prompt_speech_path):
            # 零样本克隆
            prompt_speech = load_wav(prompt_speech_path, 16000)
            for result in self.model.inference_zero_shot(
                text, prompt_text, prompt_speech, stream=False
            ):
                audio = result["tts_speech"].numpy().flatten()
                return audio
        else:
            # 预训练 speaker
            for result in self.model.inference_sft(text, speaker, stream=False):
                audio = result["tts_speech"].numpy().flatten()
                return audio

        return np.array([], dtype=np.float32)

    def synthesize_stream(
        self,
        text: str,
        prompt_speech_path: Optional[str] = None,
        prompt_text: str = "希望你以后能够做的比我还好呦。",
        speaker: str = "中文女",
    ) -> Generator[np.ndarray, None, None]:
        """
        流式合成，yield 音频 chunk (float32)
        """
        if prompt_speech_path and os.path.exists(prompt_speech_path):
            prompt_speech = load_wav(prompt_speech_path, 16000)
            generator = self.model.inference_zero_shot(
                text, prompt_text, prompt_speech, stream=True
            )
        else:
            generator = self.model.inference_sft(text, speaker, stream=True)

        for result in generator:
            chunk = result["tts_speech"].numpy().flatten()
            yield chunk

    def get_sample_rate(self) -> int:
        return self.sample_rate


# 简单测试
if __name__ == "__main__":
    tts = TTS(model_dir="pretrained_models/CosyVoice2-0.5B")

    text = "你好，我是你的数字人助手，很高兴为你服务。"
    audio = tts.synthesize(text)
    print(f"Generated audio shape: {audio.shape}, sample_rate: {tts.get_sample_rate()}")

    # 流式测试
    # for chunk in tts.synthesize_stream(text):
    #     print(chunk.shape)