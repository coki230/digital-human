# src/session/manager.py
"""
会话管理器：串联 AudioPipeline + LLM + FlashHeadEngine
"""

from __future__ import annotations
from typing import Optional, Generator, Callable
import numpy as np
from loguru import logger

from src.audio.pipeline import AudioPipeline, AudioConfig
from src.avatar.engine import FlashHeadEngine
from src.llm.streaming_utils import StreamingSentenceBuffer
from src.session.context import SessionContext


class SessionManager:
    def __init__(
        self,
        audio_pipeline: AudioPipeline,
        avatar_engine: FlashHeadEngine,
        llm_client,                          # 你自己的 LLM 封装（Ollama / vLLM 等）
    ):
        self.audio = audio_pipeline
        self.avatar = avatar_engine
        self.llm = llm_client

        self.contexts: dict[str, SessionContext] = {}

    def create_session(
        self,
        session_id: str,
        avatar_image_path: str,
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ):
        """创建完整会话（音频状态 + 数字人状态 + 对话上下文）"""
        # 1. Avatar
        self.avatar.create_session(
            session_id=session_id,
            cond_image_path=avatar_image_path,
            seed=seed,
        )

        # 2. 对话上下文
        ctx = SessionContext(
            session_id=session_id,
            avatar_image=avatar_image_path,
        )
        if system_prompt:
            ctx.system_prompt = system_prompt
        self.contexts[session_id] = ctx

        # 3. 音频重置
        self.audio.reset()

        logger.info(f"SessionManager: session {session_id} created.")
        return ctx

    def close_session(self, session_id: str):
        self.avatar.close_session(session_id)
        self.contexts.pop(session_id, None)
        self.audio.reset()
        logger.info(f"SessionManager: session {session_id} closed.")

    def interrupt(self, session_id: str):
        """用户打断"""
        self.audio.interrupt()
        self.avatar.interrupt(session_id)
        logger.info(f"Session {session_id} interrupted by user.")

    def process_user_audio(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
    ) -> Optional[str]:
        """
        处理用户音频块，如果检测到完整一句话则返回文本
        """
        return self.audio.process_audio_chunk(audio_chunk)

    def chat(
            self,
            session_id: str,
            user_text: str,
    ) -> Generator[np.ndarray, None, None]:
        """
        流式分句版对话（低首响延迟）
        """
        ctx = self.contexts.get(session_id)
        if ctx is None:
            raise ValueError(f"Session {session_id} not found")

        ctx.add_user(user_text)
        messages = ctx.get_messages()

        full_response_parts = []
        buffer = StreamingSentenceBuffer(min_sentence_length=4, max_buffer_length=70)

        try:
            # LLM 流式
            for token in self.llm.chat_stream(messages):
                # 支持外部打断
                if self.audio.is_interrupted():
                    logger.info("Chat interrupted during LLM generation.")
                    break

                for sentence in buffer.feed(token):
                    full_response_parts.append(sentence)
                    logger.info(f"[Emit Sentence] {sentence}")

                    # 立刻 TTS + 驱动数字人
                    yield from self._speak_sentence(session_id, sentence)

            # 处理最后剩余文本
            last_sentence = buffer.flush()
            if last_sentence and not self.audio.is_interrupted():
                full_response_parts.append(last_sentence)
                logger.info(f"[Emit Final] {last_sentence}")
                yield from self._speak_sentence(session_id, last_sentence)

        finally:
            # 记录完整回复
            full_text = "".join(full_response_parts).strip()
            if full_text:
                ctx.add_assistant(full_text)

    def _speak_sentence(self, session_id: str, sentence: str) -> Generator[np.ndarray, None, None]:
        """把单句送去 TTS 并驱动 Avatar"""

        def audio_gen():
            for chunk in self.audio.text_to_speech_stream(sentence):
                if self.audio.is_interrupted():
                    break
                yield chunk

        for frames in self.avatar.feed_audio_stream(session_id, audio_gen()):
            if self.audio.is_interrupted():
                break
            yield frames


# 使用示例（伪代码）
if __name__ == "__main__":
    # 初始化各组件
    audio_pipe = AudioPipeline(AudioConfig())
    avatar_engine = FlashHeadEngine(
        ckpt_dir="models/SoulX-FlashHead-1_3B",
        wav2vec_dir="models/wav2vec2-base-960h",
        model_type="lite",
    )
    # llm_client = YourLLMClient(...)

    # manager = SessionManager(audio_pipe, avatar_engine, llm_client)
    # manager.create_session("user_001", "assets/portraits/girl.png")