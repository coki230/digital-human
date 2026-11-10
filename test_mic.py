import sounddevice as sd
import soundfile as sf

duration = 3
sample_rate = 16000   # 换成 16kHz 试试
channels = 1
filename = "test.wav"

print("🎤 正在使用系统默认音频通道录音，请说话...")

# 不指定 device=5，让系统音频架构自动路由和重采样
audio_data = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=channels,
    dtype='int16'
)

sd.wait()
sf.write(filename, audio_data, sample_rate)
print(f"✅ 录音结束！已成功保存到: {filename}")