# src/session/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import time


@dataclass
class DialogTurn:
    role: str               # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionContext:
    session_id: str
    avatar_image: str
    history: List[DialogTurn] = field(default_factory=list)
    system_prompt: str = "你是一个友好、自然的数字人助手。"

    def add_user(self, text: str):
        self.history.append(DialogTurn(role="user", content=text))

    def add_assistant(self, text: str):
        self.history.append(DialogTurn(role="assistant", content=text))

    def get_messages(self) -> list[dict]:
        """转成 LLM 常用的 messages 格式"""
        msgs = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-12:]:  # 保留最近几轮
            msgs.append({"role": turn.role, "content": turn.content})
        return msgs

    def clear(self):
        self.history.clear()