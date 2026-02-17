from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assistant.actions import ActionExecutor
from assistant.llm import LLMError, OllamaClient
from assistant.protocol import parse_assistant_message


@dataclass(slots=True)
class AgentConfig:
    model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/chat"
    max_steps: int = 4
    auto_approve: bool = False
    allow_outside_workspace: bool = False
    workspace: Path = Path.cwd()


class DesktopAssistantAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(model=config.model, url=config.ollama_url)
        self.executor = ActionExecutor(
            workspace=config.workspace,
            auto_approve=config.auto_approve,
            allow_outside_workspace=config.allow_outside_workspace,
        )
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

    def _build_system_prompt(self) -> str:
        actions_description = self.executor.list_actions_for_prompt()
        return (
            "Ты — ассистент, который помогает пользователю и при необходимости вызывает действия на ПК.\n"
            "Отвечай СТРОГО в JSON-формате без markdown:\n"
            '{"reply":"текст для пользователя","actions":[{"name":"action_name","args":{}}]}\n'
            "Если действия не нужны, верни пустой массив actions.\n"
            "Не выдумывай действия, используй только доступные:\n"
            f"{actions_description}\n"
            "Если действие не удалось, объясни причину в reply."
        )

    def handle(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        last_reply = ""

        for _ in range(self.config.max_steps):
            try:
                raw = self.client.chat(self.messages)
            except LLMError as exc:
                return f"Ошибка LLM: {exc}"

            self.messages.append({"role": "assistant", "content": raw})
            parsed = parse_assistant_message(raw)
            last_reply = parsed.reply

            if not parsed.actions:
                return parsed.reply

            feedback_lines = []
            for action in parsed.actions:
                result = self.executor.execute(action)
                feedback_lines.append(result.as_feedback(action.name))

            feedback_text = "\n".join(feedback_lines)
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Результаты выполнения действий:\n"
                        f"{feedback_text}\n"
                        "Теперь сформируй финальный ответ для пользователя "
                        "или запланируй следующий шаг."
                    ),
                }
            )

        return (
            last_reply
            if last_reply
            else "Достигнут лимит шагов. Увеличьте max_steps или уточните задачу."
        )

