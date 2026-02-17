from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_COMMANDS = {"exit", "quit", "выход", "выйти", "стоп"}


def is_exit_command(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return normalized in EXIT_COMMANDS


@dataclass(slots=True)
class VoiceInput:
    enabled: bool = False
    language: str = "ru-RU"
    timeout: float | None = 5.0
    phrase_time_limit: float | None = 20.0
    available: bool = field(init=False, default=False)
    error_message: str = field(init=False, default="")
    _sr: Any | None = field(init=False, default=None, repr=False)
    _recognizer: Any | None = field(init=False, default=None, repr=False)
    _microphone: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import speech_recognition as sr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.error_message = (
                "Голосовой ввод недоступен. Установите пакеты "
                "SpeechRecognition и PyAudio. "
                f"Техническая причина: {exc}"
            )
            return

        try:
            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._microphone = sr.Microphone()
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.4)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Не удалось инициализировать микрофон: {exc}"
            self.available = False

    def listen_once(self) -> str | None:
        if not self.enabled or not self.available:
            return None

        sr = self._sr
        recognizer = self._recognizer
        microphone = self._microphone
        if sr is None or recognizer is None or microphone is None:
            return None

        try:
            print("Слушаю... Говорите.")
            with microphone as source:
                audio = recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
            text = recognizer.recognize_google(audio, language=self.language)
            return text.strip()
        except sr.WaitTimeoutError:
            print("Не услышал речь вовремя.")
        except sr.UnknownValueError:
            print("Не удалось распознать речь, повторите.")
        except sr.RequestError as exc:
            print(f"Ошибка сервиса распознавания: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка голосового ввода: {exc}")
        return None


@dataclass(slots=True)
class VoiceOutput:
    enabled: bool = False
    rate: int = 185
    volume: float = 1.0
    available: bool = field(init=False, default=False)
    error_message: str = field(init=False, default="")
    _engine: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.error_message = (
                "Голосовой вывод недоступен. Установите пакет pyttsx3. "
                f"Техническая причина: {exc}"
            )
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._engine = engine
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"Не удалось инициализировать TTS: {exc}"
            self.available = False

    def speak(self, text: str) -> None:
        if not self.enabled or not self.available:
            return
        if not text.strip():
            return
        if self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка голосового вывода: {exc}")

