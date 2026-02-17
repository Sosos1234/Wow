from __future__ import annotations

import argparse
from pathlib import Path

from assistant.agent import AgentConfig, DesktopAssistantAgent
from assistant.screen import ScreenObserver
from assistant.voice import VoiceInput, VoiceOutput, is_exit_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локальный AI-ассистент, который умеет выполнять действия на ПК."
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Название модели Ollama, например llama3.1:8b",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="URL Ollama API",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Рабочая директория для действий (файлы/shell).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Максимум циклов планирования действий на одно сообщение.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Автоматически подтверждать действия без вопроса.",
    )
    parser.add_argument(
        "--allow-outside-workspace",
        action="store_true",
        help="Разрешить доступ к путям вне рабочей директории.",
    )
    parser.add_argument(
        "--memory-file",
        default=".assistant_memory.json",
        help="Файл для долговременной памяти между запусками.",
    )
    parser.add_argument(
        "--memory-recent-turns",
        type=int,
        default=8,
        help="Сколько последних реплик из памяти добавлять в контекст.",
    )
    parser.add_argument(
        "--memory-recent-facts",
        type=int,
        default=20,
        help="Сколько фактов о пользователе добавлять в контекст.",
    )
    parser.add_argument(
        "--memory-relevant-items",
        type=int,
        default=6,
        help="Сколько релевантных воспоминаний подмешивать по текущему запросу.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Очистить память при старте.",
    )
    parser.add_argument(
        "--conversation-window",
        type=int,
        default=24,
        help="Сколько последних сообщений держать в активном диалоговом окне.",
    )
    parser.add_argument(
        "--screen-vision",
        action="store_true",
        help="Включить анализ текущего экрана через vision-модель.",
    )
    parser.add_argument(
        "--screen-model",
        default="llava:7b",
        help="Модель Ollama для анализа экрана (например llava:7b).",
    )
    parser.add_argument(
        "--screen-refresh-seconds",
        type=float,
        default=2.0,
        help="Минимальный интервал обновления анализа экрана.",
    )
    parser.add_argument(
        "--voice-input",
        action="store_true",
        help="Включить голосовой ввод с микрофона.",
    )
    parser.add_argument(
        "--voice-output",
        action="store_true",
        help="Включить озвучивание ответов ассистента.",
    )
    parser.add_argument(
        "--voice-language",
        default="ru-RU",
        help="Язык распознавания речи (например ru-RU, en-US).",
    )
    parser.add_argument(
        "--voice-timeout",
        type=float,
        default=5.0,
        help="Сколько секунд ждать начала речи.",
    )
    parser.add_argument(
        "--voice-phrase-time-limit",
        type=float,
        default=20.0,
        help="Максимальная длительность одной голосовой фразы.",
    )
    return parser.parse_args()


def _read_user_text(voice_input: VoiceInput) -> str:
    if not voice_input.enabled or not voice_input.available:
        return input("\nВы: ").strip()

    typed = input("\nВы (Enter = голос, текст = клавиатура): ").strip()
    if typed:
        return typed

    spoken = voice_input.listen_once()
    if spoken:
        print(f"Вы (голос): {spoken}")
        return spoken
    return ""


def _handle_screen_command(screen_observer: ScreenObserver, user_text: str) -> str:
    stripped = user_text.strip()
    lowered = stripped.lower()
    if lowered in {"/screen", "/screen status"}:
        return screen_observer.status_text()
    if lowered == "/screen on":
        if screen_observer.set_active(True):
            return "Видение экрана включено."
        return f"Не удалось включить видение экрана: {screen_observer.status_text()}"
    if lowered == "/screen off":
        if screen_observer.set_active(False):
            return "Видение экрана выключено."
        return "Видение экрана уже выключено."
    if lowered == "/screen now":
        description = screen_observer.describe_screen(user_query="Что сейчас на экране?", force_refresh=True)
        return f"Текущее наблюдение экрана:\n{description}"
    return (
        "Неизвестная команда экрана. Используйте: "
        "/screen, /screen status, /screen on, /screen off, /screen now"
    )


def main() -> None:
    args = parse_args()
    config = AgentConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        max_steps=args.max_steps,
        auto_approve=args.auto_approve,
        allow_outside_workspace=args.allow_outside_workspace,
        workspace=Path(args.workspace),
        memory_file=Path(args.memory_file),
        memory_recent_turns=max(1, args.memory_recent_turns),
        memory_recent_facts=max(1, args.memory_recent_facts),
        memory_relevant_items=max(1, args.memory_relevant_items),
        conversation_messages_limit=max(4, args.conversation_window),
    )
    agent = DesktopAssistantAgent(config=config)
    voice_input = VoiceInput(
        enabled=args.voice_input,
        language=args.voice_language,
        timeout=args.voice_timeout,
        phrase_time_limit=args.voice_phrase_time_limit,
    )
    voice_output = VoiceOutput(enabled=args.voice_output)
    screen_observer = ScreenObserver(
        enabled=args.screen_vision,
        active=args.screen_vision,
        vision_model=args.screen_model,
        ollama_url=args.ollama_url,
        min_refresh_seconds=max(0.2, args.screen_refresh_seconds),
    )

    print("AI-ассистент запущен.")
    print("Введите 'exit', 'quit' или 'выход' для завершения.")
    print(
        "Команды: /memory, /memory clear, "
        "/remember <факт>, /style <предпочтение>, "
        "/screen ..."
    )

    if args.reset_memory:
        agent.clear_memory()
        print("Память очищена при старте.")

    if args.voice_input:
        if voice_input.available:
            print("Голосовой ввод включен.")
        else:
            print(f"Голосовой ввод выключен: {voice_input.error_message}")
    if args.voice_output:
        if voice_output.available:
            print("Голосовой вывод включен.")
        else:
            print(f"Голосовой вывод выключен: {voice_output.error_message}")
    if args.screen_vision:
        print(screen_observer.status_text())

    while True:
        try:
            user_text = _read_user_text(voice_input)
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение.")
            break

        if not user_text:
            continue
        if is_exit_command(user_text):
            print("Пока!")
            break
        if user_text.strip().lower() == "/memory":
            summary = agent.get_memory_summary()
            print(summary)
            voice_output.speak(summary)
            continue
        if user_text.strip().lower() == "/memory clear":
            agent.clear_memory()
            message = "Память очищена."
            print(message)
            voice_output.speak(message)
            continue
        if user_text.strip().lower().startswith("/remember "):
            fact = user_text.strip()[len("/remember ") :].strip()
            if fact:
                agent.add_manual_fact(fact)
                message = "Факт сохранен в долгую память."
            else:
                message = "После /remember нужен текст факта."
            print(message)
            voice_output.speak(message)
            continue
        if user_text.strip().lower().startswith("/style "):
            style = user_text.strip()[len("/style ") :].strip()
            if style:
                agent.add_style_preference(style)
                message = "Предпочтение стиля сохранено."
            else:
                message = "После /style нужен текст предпочтения."
            print(message)
            voice_output.speak(message)
            continue
        if user_text.strip().lower().startswith("/screen"):
            message = _handle_screen_command(screen_observer, user_text)
            print(message)
            voice_output.speak(message)
            continue

        screen_context = screen_observer.build_agent_context(user_text)
        answer = agent.handle(user_text, extra_context=screen_context)
        print(f"Ассистент: {answer}")
        voice_output.speak(answer)


if __name__ == "__main__":
    main()

