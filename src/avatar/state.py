# src/avatar/state.py
from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class SessionState:
    session_id: str
    cond_image_path: str
    seed: int = 42
    created_at: float = field(default_factory=time.time)

    # 由 engine 在 create_session 时初始化
    audio_dq: deque = None          # maxlen = cached_audio_length
    pending_samples: int = 0        # 距下一次推理还差多少采样

    is_generating: bool = False
    interrupted: bool = False
    last_active: float = field(default_factory=time.time)

    def update_active(self):
        self.last_active = time.time()

    def interrupt(self):
        self.interrupted = True
        self.is_generating = False
        if self.audio_dq is not None:
            self.audio_dq.clear()
            # 恢复成全 0 的缓存（与官方初始一致）
            self.audio_dq.extend([0.0] * self.audio_dq.maxlen)
        self.pending_samples = 0