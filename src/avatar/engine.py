from third_party.flash_head.inference import get_pipeline
import numpy as np

class FlashHeadEngine:
    def __init__(self, ckpt_dir, wav2vec_dir, model_type="lite"):
        self.pipeline = get_pipeline(...)
        self.state = {}          # session_id → state

    def create_session(self, session_id, cond_image_path, seed=42):
        # 调用 get_base_data，初始化 motion cache
        ...

    def feed_audio(self, session_id, audio_chunk: np.ndarray):
        # 更新 audio deque → get_audio_embedding → run_pipeline
        # 返回 frames (numpy)
        ...

    def interrupt(self, session_id):
        # 清空队列 / 重置部分状态
        ...

    def close_session(self, session_id):
        ...