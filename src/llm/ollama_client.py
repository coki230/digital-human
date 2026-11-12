# src/llm/ollama_client.py
"""
Ollama LLM Client
支持普通对话 + 流式输出，方便后续做「边生成边TTS」
"""

from __future__ import annotations
import json
from typing import List, Dict, Optional, Generator, Union
import requests
from loguru import logger


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.8:latest",          # 按你实际拉取的模型改
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.default_system_prompt = system_prompt or "你是一个友好、自然、口语化的数字人助手，回答简洁有温度。"
        self.timeout = timeout

        # 检查服务是否可用
        self._check_service()

    def _check_service(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info(f"Ollama service online. Available models: {models}")
                if self.model not in models and not any(self.model in m for m in models):
                    logger.warning(f"Model '{self.model}' not found in Ollama. Please pull it first.")
            else:
                logger.warning("Ollama service responded abnormally.")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Error: {e}")
            raise ConnectionError(f"Ollama service unavailable: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        普通非流式对话
        messages 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        final_messages = self._build_messages(messages, system_prompt)

        payload = {
            "model": self.model,
            "messages": final_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式对话，yield 增量文本（适合边生成边送 TTS）
        """
        final_messages = self._build_messages(messages, system_prompt)

        payload = {
            "model": self.model,
            "messages": final_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    if data.get("done", False):
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"Ollama stream chat failed: {e}")
            raise

    def chat_simple(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """最简单的单轮接口"""
        messages = [{"role": "user", "content": user_input}]
        return self.chat(messages, system_prompt=system_prompt)

    def _build_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """组装最终 messages，自动补 system"""
        sys_prompt = system_prompt or self.default_system_prompt
        final = [{"role": "system", "content": sys_prompt}]

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in ("user", "assistant") and content:
                final.append({"role": role, "content": content})
        return final


# ------------------------------------------------------------------
# 测试
# ------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger
    logger.add(lambda msg: print(msg, end=""))

    client = OllamaClient(
        temperature=0.7,
    )

    print("=== 非流式测试 ===")
    reply = client.chat_simple("你好，请用一句话介绍你自己")
    print(f"回复: {reply}\n")

    print("=== 流式测试 ===")
    messages = [{"role": "user", "content": "讲一个很短的关于机器人的故事"}]
    for chunk in client.chat_stream(messages):
        print(chunk, end="", flush=True)
    print("\n")