from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assistant.actions import ActionExecutor
from assistant.llm import LLMError, OllamaClient
from assistant.memory import MemoryStore
from assistant.protocol import parse_assistant_message


@dataclass(slots=True)
class AgentConfig:
    model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/chat"
    max_steps: int = 4
    auto_approve: bool = False
    allow_outside_workspace: bool = False
    workspace: Path = Path.cwd()
    memory_file: Path = Path(".assistant_memory.json")
    memory_recent_turns: int = 8
    memory_recent_facts: int = 20
    memory_max_turns: int = 200
    memory_max_facts: int = 100
    memory_relevant_items: int = 6
    conversation_messages_limit: int = 24


class DesktopAssistantAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OllamaClient(model=config.model, url=config.ollama_url)
        memory_path = config.memory_file
        if not memory_path.is_absolute():
            memory_path = (config.workspace / memory_path).resolve()
        self.memory = MemoryStore(
            path=memory_path,
            max_facts=config.memory_max_facts,
            max_turns=config.memory_max_turns,
        )
        self.executor = ActionExecutor(
            workspace=config.workspace,
            auto_approve=config.auto_approve,
            allow_outside_workspace=config.allow_outside_workspace,
        )
        self.system_prompt = self._build_system_prompt()
        self.messages: list[dict[str, str]] = []

    def _build_system_prompt(self) -> str:
        actions_description = self.executor.list_actions_for_prompt()
        return (
            "Ты — ассистент, который помогает пользователю и при необходимости вызывает действия на ПК.\n"
            "Отвечай СТРОГО в JSON-формате без markdown:\n"
            '{"reply":"текст для пользователя","actions":[{"name":"action_name","args":{}}]}\n'
            "Если действия не нужны, верни пустой массив actions.\n"
            "Учитывай сохраненную память о пользователе и предыдущих диалогах, "
            "но не придумывай факты, которых в памяти нет.\n"
            "Следуй предпочтениям стиля пользователя из памяти, если они там есть.\n"
            "Отвечай естественно, доброжелательно и по-человечески.\n"
            "Не выдумывай действия, используй только доступные:\n"
            f"{actions_description}\n"
            "Если действие не удалось, объясни причину в reply."
        )

    def handle(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self._trim_messages()
        last_reply = ""

        for _ in range(self.config.max_steps):
            try:
                raw = self.client.chat(self._build_chat_messages(query_text=user_text))
            except LLMError as exc:
                return f"Ошибка LLM: {exc}"

            self.messages.append({"role": "assistant", "content": raw})
            self._trim_messages()
            parsed = parse_assistant_message(raw)
            last_reply = parsed.reply

            if not parsed.actions:
                self.memory.add_turn(user_text, parsed.reply)
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
            self._trim_messages()

        final_reply = (
            last_reply
            if last_reply
            else "Достигнут лимит шагов. Увеличьте max_steps или уточните задачу."
        )
        self.memory.add_turn(user_text, final_reply)
        return final_reply

    def _build_chat_messages(self, query_text: str) -> list[dict[str, str]]:
        memory_context = self.memory.build_context(
            recent_turns=self.config.memory_recent_turns,
            recent_facts=self.config.memory_recent_facts,
            query_text=query_text,
            relevant_items=self.config.memory_relevant_items,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": memory_context},
            *self.messages,
        ]

    def get_memory_summary(self) -> str:
        return self.memory.human_readable(
            recent_turns=self.config.memory_recent_turns,
            recent_facts=self.config.memory_recent_facts,
        )

    def clear_memory(self) -> None:
        self.memory.clear()

    def add_manual_fact(self, text: str) -> None:
        self.memory.add_manual_fact(text)

    def add_style_preference(self, text: str) -> None:
        self.memory.add_style_preference(text)

    def _trim_messages(self) -> None:
        limit = max(4, self.config.conversation_messages_limit)
        if len(self.messages) <= limit:
            return
        self.messages = self.messages[-limit:]

