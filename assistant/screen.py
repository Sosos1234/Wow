from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import io
import re
import time
from typing import Any

from assistant.llm import LLMError, OllamaClient


def _clip_text(value: str, max_chars: int) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _sample_bytes(data: bytes, sample_size: int = 2048) -> bytes:
    if not data:
        return b""
    if len(data) <= sample_size:
        return data
    step = max(1, len(data) // sample_size)
    return data[::step][:sample_size]


def _sample_change_ratio_percent(prev_sample: bytes, curr_sample: bytes) -> float:
    if not prev_sample or not curr_sample:
        return 100.0
    n = min(len(prev_sample), len(curr_sample))
    if n == 0:
        return 100.0
    diff = sum(1 for i in range(n) if prev_sample[i] != curr_sample[i])
    return (diff / n) * 100.0


def detect_screen_alerts(summary: str, ocr_text: str = "") -> list[str]:
    text = f"{summary}\n{ocr_text}".lower()
    rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("traceback", "exception", "fatal"), "На экране видна ошибка выполнения (exception/traceback)."),
        (("error", "ошибка", "failed", "failure"), "На экране есть сообщение об ошибке."),
        (("warning", "предупреждение"), "На экране есть предупреждение."),
        (("denied", "forbidden", "permission", "доступ запрещен"), "Похоже на проблему с правами доступа."),
        (("not found", "404"), "Похоже, что ресурс не найден (404/not found)."),
        (("timeout", "timed out", "превышено время", "не отвечает"), "Похоже на таймаут или зависание."),
        (("refused", "connection reset", "disconnect", "соединение прервано"), "Похоже на сетевую проблему."),
    )

    alerts: list[str] = []
    for tokens, message in rules:
        if any(token in text for token in tokens):
            alerts.append(message)

    if re.search(r"\b(4\d\d|5\d\d)\b", text):
        alerts.append("На экране есть HTTP-код ошибки (4xx/5xx).")

    unique: list[str] = []
    seen: set[str] = set()
    for alert in alerts:
        key = alert.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(alert)
    return unique


@dataclass(slots=True)
class ScreenObserver:
    enabled: bool = False
    active: bool = True
    vision_model: str = "llava:7b"
    ollama_url: str = "http://localhost:11434/api/chat"
    min_refresh_seconds: float = 2.0
    enable_ocr: bool = False
    ocr_engine: str = "auto"
    ocr_language: str = "rus+eng"
    auto_react: bool = False
    change_threshold_percent: float = 8.0
    timeout_seconds: int = 120
    available: bool = field(init=False, default=False)
    error_message: str = field(init=False, default="")
    _mss_module: Any | None = field(init=False, default=None, repr=False)
    _mss_tools: Any | None = field(init=False, default=None, repr=False)
    _client: OllamaClient | None = field(init=False, default=None, repr=False)
    _pytesseract: Any | None = field(init=False, default=None, repr=False)
    _pil_image: Any | None = field(init=False, default=None, repr=False)
    _last_summary: str = field(init=False, default="", repr=False)
    _last_summary_digest: str = field(init=False, default="", repr=False)
    _last_ocr_text: str = field(init=False, default="", repr=False)
    _last_ocr_digest: str = field(init=False, default="", repr=False)
    _last_ocr_ts: float = field(init=False, default=0.0, repr=False)
    _last_frame_png_bytes: bytes = field(init=False, default=b"", repr=False)
    _last_frame_b64: str = field(init=False, default="", repr=False)
    _last_frame_digest: str = field(init=False, default="", repr=False)
    _last_frame_sample: bytes = field(init=False, default=b"", repr=False)
    _last_frame_ts: float = field(init=False, default=0.0, repr=False)
    _last_change_ratio_percent: float = field(init=False, default=0.0, repr=False)
    _last_change_significant: bool = field(init=False, default=False, repr=False)
    _last_change_message: str = field(
        init=False,
        default="Изменения экрана пока не измерялись.",
        repr=False,
    )
    _last_alerts: list[str] = field(init=False, default_factory=list, repr=False)
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
        self._init_ocr_backend()
        self.available = True

    def _init_ocr_backend(self) -> None:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            self._pytesseract = None
            self._pil_image = None
            return

        self._pytesseract = pytesseract
        self._pil_image = Image

    def set_active(self, value: bool) -> bool:
        if not self.enabled or not self.available:
            self.active = False
            return False
        self.active = value
        return self.active

    def set_ocr_enabled(self, value: bool) -> bool:
        if not self.enabled or not self.available:
            self.enable_ocr = False
            return False
        self.enable_ocr = value
        return self.enable_ocr

    def set_auto_react(self, value: bool) -> bool:
        if not self.enabled or not self.available:
            self.auto_react = False
            return False
        self.auto_react = value
        return self.auto_react

    def status_text(self) -> str:
        if not self.enabled:
            return "Видение экрана выключено. Включите через --screen-vision."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        mode = "включено" if self.active else "выключено"
        ocr_mode = "включен" if self.enable_ocr else "выключен"
        react_mode = "включен" if self.auto_react else "выключен"
        engine = self._effective_ocr_engine()
        return (
            f"Видение экрана {mode}. "
            f"Модель: {self.vision_model}. "
            f"Минимальный интервал обновления: {self.min_refresh_seconds:.1f}с. "
            f"OCR: {ocr_mode} (engine={engine}, lang={self.ocr_language}). "
            f"Порог изменений: {self.change_threshold_percent:.1f}%. "
            f"Авто-реакция: {react_mode}."
        )

    def describe_screen(self, user_query: str, force_refresh: bool = False) -> str:
        if not self.enabled:
            return "Видение экрана выключено."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        if not self.active:
            return "Видение экрана сейчас выключено (режим /screen off)."

        try:
            _, image_b64, frame_digest = self._get_frame(force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось сделать скриншот: {exc}"

        if (
            not force_refresh
            and self._last_summary
            and frame_digest == self._last_summary_digest
        ):
            return self._last_summary

        prompt = self._build_vision_prompt(user_query=user_query)
        try:
            summary = self._analyze_with_vision(prompt=prompt, image_b64=image_b64)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось проанализировать экран: {exc}"

        summary = _clip_text(summary, max_chars=1500)
        self._last_summary = summary
        self._last_summary_digest = frame_digest
        self._last_alerts = detect_screen_alerts(summary=summary, ocr_text=self._last_ocr_text)
        self._last_update_ts = time.time()
        return summary

    def extract_screen_text(self, force_refresh: bool = False) -> str:
        if not self.enabled:
            return "Видение экрана выключено."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        if not self.active:
            return "Видение экрана сейчас выключено (режим /screen off)."
        if not self.enable_ocr:
            return "OCR выключен. Включите через --screen-ocr или /screen ocr on."

        try:
            png_bytes, image_b64, frame_digest = self._get_frame(force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось сделать скриншот: {exc}"

        if (
            not force_refresh
            and self._last_ocr_text
            and frame_digest == self._last_ocr_digest
        ):
            return self._last_ocr_text

        engine = self._effective_ocr_engine()
        try:
            if engine == "tesseract":
                text = self._ocr_with_tesseract(png_bytes)
            else:
                text = self._ocr_with_vision(image_b64)
        except Exception as exc:  # noqa: BLE001
            if self.ocr_engine == "auto" and engine == "tesseract":
                try:
                    text = self._ocr_with_vision(image_b64)
                except Exception as fallback_exc:  # noqa: BLE001
                    return f"Не удалось распознать текст (OCR): {fallback_exc}"
            else:
                return f"Не удалось распознать текст (OCR): {exc}"

        text = _clip_text(text, max_chars=3500).strip()
        if not text:
            text = "Текст на экране не распознан."

        self._last_ocr_text = text
        self._last_ocr_digest = frame_digest
        self._last_ocr_ts = time.time()
        self._last_alerts = detect_screen_alerts(summary=self._last_summary, ocr_text=text)
        return text

    def get_change_report(self, force_refresh: bool = False) -> str:
        if not self.enabled:
            return "Видение экрана выключено."
        if not self.available:
            return f"Видение экрана недоступно: {self.error_message}"
        if not self.active:
            return "Видение экрана сейчас выключено (режим /screen off)."

        try:
            self._get_frame(force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось обновить кадр экрана: {exc}"
        return self._last_change_message

    def build_agent_context(self, user_query: str) -> str | None:
        if not self.enabled or not self.available or not self.active:
            return None

        summary = self.describe_screen(user_query=user_query, force_refresh=False)
        if summary.startswith("Не удалось"):
            return f"Контекст экрана недоступен: {summary}"

        lines = [
            "КОНТЕКСТ С ЭКРАНА (может быть неточным):",
            summary,
        ]

        if self._last_change_message:
            lines.append(f"Изменения экрана: {self._last_change_message}")

        ocr_text = ""
        if self.enable_ocr:
            ocr_text = self.extract_screen_text(force_refresh=False)
            if not ocr_text.startswith(("Не удалось", "Видение экрана", "OCR выключен")):
                lines.append(f"OCR (видимый текст): {_clip_text(ocr_text, max_chars=900)}")

        alerts = detect_screen_alerts(summary=summary, ocr_text=ocr_text)
        self._last_alerts = alerts
        if self.auto_react and alerts:
            lines.append("Потенциальные проблемы на экране:")
            for alert in alerts:
                lines.append(f"- {alert}")
            lines.append(
                "Если это уместно, проактивно предложи пользователю конкретные шаги решения."
            )

        lines.append("Используй контекст с экрана только если он релевантен запросу пользователя.")
        return "\n".join(lines)

    def _capture_screen_frame(self) -> tuple[bytes, str, str, bytes]:
        if self._mss_module is None or self._mss_tools is None:
            raise RuntimeError("Библиотека mss не инициализирована.")

        with self._mss_module.mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            screenshot = sct.grab(monitor)
            png_bytes = self._mss_tools.to_png(screenshot.rgb, screenshot.size)
        frame_digest = hashlib.sha256(png_bytes).hexdigest()
        frame_sample = _sample_bytes(png_bytes, sample_size=2048)
        frame_b64 = base64.b64encode(png_bytes).decode("ascii")
        return png_bytes, frame_b64, frame_digest, frame_sample

    def _get_frame(self, force_refresh: bool) -> tuple[bytes, str, str]:
        now = time.time()
        if (
            not force_refresh
            and self._last_frame_png_bytes
            and now - self._last_frame_ts < self.min_refresh_seconds
        ):
            return self._last_frame_png_bytes, self._last_frame_b64, self._last_frame_digest

        prev_digest = self._last_frame_digest
        prev_sample = self._last_frame_sample

        png_bytes, frame_b64, frame_digest, frame_sample = self._capture_screen_frame()
        self._update_change_tracking(
            prev_digest=prev_digest,
            prev_sample=prev_sample,
            curr_digest=frame_digest,
            curr_sample=frame_sample,
        )

        self._last_frame_png_bytes = png_bytes
        self._last_frame_b64 = frame_b64
        self._last_frame_digest = frame_digest
        self._last_frame_sample = frame_sample
        self._last_frame_ts = now
        return png_bytes, frame_b64, frame_digest

    def _update_change_tracking(
        self,
        prev_digest: str,
        prev_sample: bytes,
        curr_digest: str,
        curr_sample: bytes,
    ) -> None:
        if not prev_digest:
            self._last_change_ratio_percent = 0.0
            self._last_change_significant = False
            self._last_change_message = "Первый кадр захвачен, сравнить пока не с чем."
            return

        if prev_digest == curr_digest:
            self._last_change_ratio_percent = 0.0
            self._last_change_significant = False
            self._last_change_message = "Экран почти не изменился (0.0%)."
            return

        ratio = _sample_change_ratio_percent(prev_sample=prev_sample, curr_sample=curr_sample)
        significant = ratio >= self.change_threshold_percent
        self._last_change_ratio_percent = ratio
        self._last_change_significant = significant
        if significant:
            self._last_change_message = (
                f"Существенное изменение экрана: {ratio:.1f}% "
                f"(порог {self.change_threshold_percent:.1f}%)."
            )
        else:
            self._last_change_message = (
                f"Небольшое изменение экрана: {ratio:.1f}% "
                f"(порог {self.change_threshold_percent:.1f}%)."
            )

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

    def _ocr_with_vision(self, image_b64: str) -> str:
        prompt = (
            "Сделай OCR по скриншоту. "
            "Верни только распознанный текст без объяснений. "
            "Если текст почти не читается, верни: [text unreadable]"
        )
        return self._analyze_with_vision(prompt=prompt, image_b64=image_b64)

    def _ocr_with_tesseract(self, png_bytes: bytes) -> str:
        if self._pytesseract is None or self._pil_image is None:
            raise RuntimeError("pytesseract/Pillow недоступны.")
        image = self._pil_image.open(io.BytesIO(png_bytes))
        return str(self._pytesseract.image_to_string(image, lang=self.ocr_language))

    def _effective_ocr_engine(self) -> str:
        engine = self.ocr_engine.strip().lower()
        if engine not in {"auto", "tesseract", "vision"}:
            engine = "auto"
        if engine == "auto":
            if self._pytesseract is not None and self._pil_image is not None:
                return "tesseract"
            return "vision"
        return engine

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

