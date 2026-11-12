# src/avatar/video_utils.py
import os
import wave
import subprocess
import numpy as np
import imageio
import torch
from loguru import logger


def write_frames_to_mp4(frames_list, video_path, fps):
    """仅视频轨。frames_list 元素为 cpu tensor 或 np.ndarray，形状 (T,H,W,3)"""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    with imageio.get_writer(
        video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            if isinstance(frames, torch.Tensor):
                frames_np = frames.numpy().astype(np.uint8)
            else:
                frames_np = np.asarray(frames, dtype=np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i])
    return video_path


def save_audio_wav(audio_array: np.ndarray, wav_path: str, sample_rate: int = 16000):
    """float32 [-1,1] → wav"""
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return wav_path


def save_video_with_audio(frames_list, video_path, audio_path, fps):
    """官方同款：先写无声 mp4，再混音频"""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    write_frames_to_mp4(frames_list, temp_path, fps)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    logger.info(f"Saved video with audio: {video_path}")
    return video_path