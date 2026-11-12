# src/llm/streaming_utils.py
"""
流式分句工具：把 LLM 增量 token 实时切成完整句子
"""

from __future__ import annotations
import re
from typing import Generator, Optional, Callable


# 中英文常见句子结束符
SENTENCE_ENDINGS = re.compile(r'([。！？.!?；;\n])')


class StreamingSentenceBuffer:
    """
    把流式文本缓存起来，遇到句子结束符就吐出完整句子。
    """

    def __init__(
        self,
        min_sentence_length: int = 4,          # 太短的不单独成句（避免“好的。”这种过于碎片）
        max_buffer_length: int = 80,           # 强制切分的最大长度，防止一直等不到句号
    ):
        self.min_sentence_length = min_sentence_length
        self.max_buffer_length = max_buffer_length
        self.buffer = ""

    def feed(self, text_chunk: str) -> list[str]:
        """
        喂入增量文本，返回当前可以输出的完整句子列表
        """
        if not text_chunk:
            return []

        self.buffer += text_chunk
        sentences = []

        while True:
            match = SENTENCE_ENDINGS.search(self.buffer)
            if not match:
                # 没有结束符，检查是否过长需要强制切
                if len(self.buffer) >= self.max_buffer_length:
                    # 尝试在逗号、空格处切
                    force_split_pos = self._find_force_split_pos(self.buffer)
                    sentence = self.buffer[:force_split_pos].strip()
                    self.buffer = self.buffer[force_split_pos:].lstrip()
                    if len(sentence) >= self.min_sentence_length:
                        sentences.append(sentence)
                    continue
                break

            end_pos = match.end()
            sentence = self.buffer[:end_pos].strip()
            self.buffer = self.buffer[end_pos:].lstrip()

            if len(sentence) >= self.min_sentence_length:
                sentences.append(sentence)
            # 太短的句子继续累积到下一句

        return sentences

    def flush(self) -> Optional[str]:
        """把剩余内容全部吐出（最后调用）"""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining and len(remaining) >= 2:
            return remaining
        return None

    def _find_force_split_pos(self, text: str) -> int:
        """强制切分时，优先在逗号、顿号、空格处切"""
        for sep in ["，", ",", "、", " ", " "]:
            pos = text.rfind(sep, 0, self.max_buffer_length)
            if pos > self.min_sentence_length:
                return pos + 1
        return self.max_buffer_length

    def reset(self):
        self.buffer = ""


def stream_sentences(
    text_stream: Generator[str, None, None],
    min_sentence_length: int = 4,
    max_buffer_length: int = 80,
) -> Generator[str, None, None]:
    """
    高阶工具：直接把 token 流变成句子流
    """
    buffer = StreamingSentenceBuffer(
        min_sentence_length=min_sentence_length,
        max_buffer_length=max_buffer_length,
    )

    for chunk in text_stream:
        for sent in buffer.feed(chunk):
            yield sent

    # 最后剩余内容
    last = buffer.flush()
    if last:
        yield last