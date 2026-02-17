from __future__ import annotations

import argparse
from pathlib import Path

from assistant.agent import AgentConfig, DesktopAssistantAgent


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        max_steps=args.max_steps,
        auto_approve=args.auto_approve,
        allow_outside_workspace=args.allow_outside_workspace,
        workspace=Path(args.workspace),
    )
    agent = DesktopAssistantAgent(config=config)

    print("AI-ассистент запущен.")
    print("Введите 'exit' или 'quit' для выхода.")

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Пока!")
            break

        answer = agent.handle(user_text)
        print(f"Ассистент: {answer}")


if __name__ == "__main__":
    main()

