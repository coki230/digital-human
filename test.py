
from datetime import datetime
import os
import time

import librosa
from loguru import logger
import numpy as np

import torch
import torch.distributed as dist
import subprocess

import imageio

from third_party.flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, \
    run_pipeline


def save_video(frames_list, video_path, audio_path, fps):
    temp_video_path = video_path.replace('.mp4', '_tmp.mp4')
    with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                            fps=fps, codec='h264', ffmpeg_params=['-bf', '0']) as writer:
        for frames in frames_list:
            frames = frames.numpy().astype(np.uint8)
            for i in range(frames.shape[0]):
                frame = frames[i, :, :, :]
                writer.append_data(frame)

    # merge video and audio
    cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path,
           '-y']
    subprocess.run(cmd)
    os.remove(temp_video_path)

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))


ckpt_dir = "/home/coki/humanAI/dev1/model/SoulX-FlashHead-1_3B"
wav2vec_dir = "/home/coki/humanAI/dev1/model/wav2vec2-base-960h"
model_type = "lite"
cond_image_dir = None
cond_image = "/home/coki/humanAI/dev1/hhtx.jpeg"
base_seed = 42
use_face_crop = False
audio_path = "/home/coki/humanAI/dev1/test.wav"
save_file = None

pipeline = get_pipeline(world_size=world_size, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir,
                        model_type=model_type)
get_base_data(pipeline,
              cond_image_path_or_dir=cond_image_dir if cond_image_dir is not None else cond_image,
              base_seed=base_seed, use_face_crop=use_face_crop)
infer_params = get_infer_params()

sample_rate = infer_params['sample_rate']
tgt_fps = infer_params['tgt_fps']
cached_audio_duration = infer_params['cached_audio_duration']
frame_num = infer_params['frame_num']
motion_frames_num = infer_params['motion_frames_num']
slice_len = frame_num - motion_frames_num

human_speech_array_all, _ = librosa.load(audio_path, sr=infer_params['sample_rate'], mono=True)
human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

if rank == 0:
    logger.info("Data preparation done. Start to generate video...")

generated_list = []

# pad audio with silence to avoid truncating the last chunk
remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
if remainder > 0:
    pad_length = human_speech_array_slice_len - remainder
    human_speech_array_all = np.concatenate(
        [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)])

# encode audio together
audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)

# split audio embedding into chunks
# for Pro model: 33, 28, 28, 28, ...; For Lite model: 33, 24, 24, 24, ...
audio_embedding_chunks_list = [audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous() for i in
                               range((audio_embedding_all.shape[1] - frame_num) // slice_len)]

for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
    torch.cuda.synchronize()
    start_time = time.time()

    # inference
    video = run_pipeline(pipeline, audio_embedding_chunk)

    if chunk_idx != 0:
        video = video[motion_frames_num:]

    torch.cuda.synchronize()
    end_time = time.time()
    if rank == 0:
        logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.3f}s")

    generated_list.append(video.cpu())

    if rank == 0:
        if save_file is None:
            output_dir = 'sample_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S-%f")[:-3]
            filename = f"res_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            save_file = filepath

        save_video(generated_list, save_file, audio_path, fps=tgt_fps)
        logger.info(f"Saving generated video to {save_file}")
        logger.info("Finished.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()