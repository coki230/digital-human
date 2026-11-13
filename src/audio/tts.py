# src/audio/tts.py
"""
TTS using CosyVoice (支持流式输出，中文效果优秀)。
需要先安装 CosyVoice 官方仓库。
"""
import os
import sys
from pathlib import Path
from typing import Generator, Optional
import numpy as np
import torchaudio

# ========== 根据你实际安装路径修改 ==========
COSYVOICE_ROOT = Path("/home/coki/humanAI/dev1/CosyVoice")          # 改成你的 CosyVoice 路径
sys.path.append(str(COSYVOICE_ROOT))
sys.path.append(str(COSYVOICE_ROOT / "third_party/Matcha-TTS"))
# ==========================================

from cosyvoice.cli.cosyvoice import AutoModel
import sounddevice as sd

class TTS:
    _instance = None

    def __new__(cls, model_dir='/home/coki/humanAI/dev1/model/CosyVoice-300M-SFT'):
        # 单例模式：确保整个生命周期只加载一次模型，避免重复占用显存/内存
        if cls._instance is None:
            cls._instance = super(TTS, cls).__new__(cls)
            cls._instance.initialize(model_dir)
        return cls._instance

    def initialize(self, model_dir):
        print(f"正在加载 CosyVoice 模型，路径: {model_dir} ...")
        # 检查路径是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型路径不存在: {model_dir}")

        # 初始化模型
        self.cosyvoice = AutoModel(model_dir=model_dir)
        self.sample_rate = self.cosyvoice.sample_rate
        print("CosyVoice 模型加载完成！采样率:", self.sample_rate)

    def text_2_audio(self, text, speaker='中文女', stream=False):
        resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
        """生成语音数据"""
        if stream:
            # 如果是流式，返回迭代器
            generator = self.cosyvoice.inference_sft(text, speaker, stream=True)
            for result in generator:
                speech_tensor = result.get('tts_speech', None)
                if speech_tensor is None:
                    continue

                # 避开太短的 chunk 造成的重采样吞字
                if speech_tensor.shape[-1] < 160:
                    # 如果太短，先直接转 numpy 或者跳过缓存到下一个
                    pass

                speech_tensor = resampler(speech_tensor)
                audio_data = speech_tensor.detach().cpu().numpy().squeeze()
                if audio_data.size > 0:
                    yield audio_data
            return None
        else:
            # 非流式，直接拿第一个结果
            result = next(self.cosyvoice.inference_sft(text, speaker, stream=False))
            speech_tensor = resampler(result['tts_speech'])
            audio_data = speech_tensor.detach().cpu().numpy().squeeze()
            return audio_data

    def play_audio(self, audio_data):
        """播放音频"""
        sd.play(audio_data, self.sample_rate)
        sd.wait()

    def get_sample_rate(self) -> int:
        return self.sample_rate



# === 使用示例 ===
if __name__ == '__main__':

    # === 全局单例实例化 ===
    # 在模块导入时或程序入口处初始化一次即可全局调用
    tts_engine = TTS()
    text_input = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
    # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    audio = tts_engine.text_2_audio(text_input, speaker='中文女', stream=False)
    tts_engine.play_audio(audio)