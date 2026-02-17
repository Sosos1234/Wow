from __future__ import annotations

import base64
from dataclasses import dataclass, field
import time
from typing import Any

from assistant.llm import LLMError, OllamaClient


def _clip_text(value: str, max_chars: int) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass(slots=True)
class ScreenObserver:
    enabled: bool = False
    active: bool = True
    vision_model: str = "llava:7b"
    ollama_url: str = "http://localhost:11434/api/chat"
    min_refresh_seconds: float = 2.0
    timeout_seconds: int = 120
    available: bool = field(init=False, default=False)
    error_message: str = field(init=False, default="")
    _mss_module: Any | None = field(init=False, default=None, repr=False)
    _mss_tools: Any | None = field(init=False, default=None, repr=False)
    _client: OllamaClient | None = field(init=False, default=None, repr=False)
    _last_summary: str = field(init=False, default="", repr=False)
    _last_update_ts: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            self.active = False
            return
        self._init_backend()
        if not self.available:
            self.active = False

    def _init_backend(self) -> None:
        try:
            import mss  # type: ignore
            from mss import tools  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.error_message = (
                "Модуль захвата экрана недоступен. "
                "Установите extra-зависимость screen: pip install .[screen]. "
                f"Техническая причина: {exc}"
            )
            self.available = False
            return

        self._mss_module = mss
        self._mss_tools = tools
        self._client = OllamaClient(
            model=self.vision_model,
            url=self.ollama_url,
            timeout_seconds=self.timeout_seconds,
        )
        self.available = True

    def set_active(self, value: bool) -> bool:
        if not self.enabled or not self.available:
            self.active = False
            return False
        self.active = value
        return self.active

    def status_text(self) -> str:
        if not self.enabled:
            return "Видение экрана выключено. Включите через --screen-vision."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        mode = "включено" if self.active else "выключено"
        return (
            f"Видение экрана {mode}. "
            f"Модель: {self.vision_model}. "
            f"Минимальный интервал обновления: {self.min_refresh_seconds:.1f}с."
        )

    def describe_screen(self, user_query: str, force_refresh: bool = False) -> str:
        if not self.enabled:
            return "Видение экрана выключено."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        if not self.active:
            return "Видение экрана сейчас выключено (режим /screen off)."

        now = time.time()
        if (
            not force_refresh
            and self._last_summary
            and now - self._last_update_ts < self.min_refresh_seconds
        ):
            return self._last_summary

        try:
            image_b64 = self._capture_screen_png_base64()
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось сделать скриншот: {exc}"

        prompt = self._build_vision_prompt(user_query=user_query)
        try:
            summary = self._analyze_with_vision(prompt=prompt, image_b64=image_b64)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось проанализировать экран: {exc}"

        summary = _clip_text(summary, max_chars=1500)
        self._last_summary = summary
        self._last_update_ts = now
        return summary

    def build_agent_context(self, user_query: str) -> str | None:
        if not self.enabled or not self.available or not self.active:
            return None

        summary = self.describe_screen(user_query=user_query, force_refresh=False)
        if summary.startswith("Не удалось"):
            return f"Контекст экрана недоступен: {summary}"
        return (
            "КОНТЕКСТ С ЭКРАНА (может быть неточным):\n"
            f"{summary}\n"
            "Используй этот контекст только если он релевантен запросу пользователя."
        )

    def _capture_screen_png_base64(self) -> str:
        if self._mss_module is None or self._mss_tools is None:
            raise RuntimeError("Библиотека mss не инициализирована.")

        with self._mss_module.mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            screenshot = sct.grab(monitor)
            png_bytes = self._mss_tools.to_png(screenshot.rgb, screenshot.size)
        return base64.b64encode(png_bytes).decode("ascii")

    def _analyze_with_vision(self, prompt: str, image_b64: str) -> str:
        if self._client is None:
            raise RuntimeError("Vision-клиент не инициализирован.")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ]
        try:
            return self._client.chat_with_model(
                messages=messages,
                model=self.vision_model,
                temperature=0.1,
            )
        except LLMError as exc:
            raise RuntimeError(str(exc)) from exc

    def _build_vision_prompt(self, user_query: str) -> str:
        safe_query = _clip_text(user_query, max_chars=300)
        return (
            "Ты модуль компьютерного зрения для настольного ассистента.\n"
            "Опиши, что сейчас видно на экране, кратко и по делу.\n"
            "Верни:\n"
            "1) Что открыто (окна/приложения),\n"
            "2) Какие ключевые элементы/текст заметны,\n"
            "3) Что делает пользователь (если можно понять),\n"
            "4) Что важно именно для текущего запроса.\n"
            "Если текст плохо различим — так и напиши.\n"
            f"Текущий запрос пользователя: {safe_query}"
        )

