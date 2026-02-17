from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import request


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaClient:
    model: str
    url: str = "http://localhost:11434/api/chat"
    timeout_seconds: int = 120

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": 0.2},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Не удалось обратиться к Ollama: {exc}") from exc

        content = _extract_content(data)
        if not content:
            raise LLMError("Ollama вернул пустой ответ.")
        return content


def _extract_content(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""

