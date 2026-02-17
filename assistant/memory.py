from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compact_text(value: str, max_len: int) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_facts(user_text: str) -> list[str]:
    text = " ".join(user_text.strip().split())
    if not text:
        return []

    patterns = (
        r"\bменя зовут\s+([А-ЯA-ZЁ][\w\- ]{1,40})",
        r"\bмое имя\s+([А-ЯA-ZЁ][\w\- ]{1,40})",
        r"\bмне\s+(\d{1,3})\s*лет",
        r"\bя живу в\s+([^,.!?]{2,60})",
        r"\bмне нравится\s+([^.!?]{2,100})",
        r"\bя работаю\s+([^.!?]{2,100})",
        r"\bi am\s+([^.!?]{2,100})",
        r"\bmy name is\s+([A-Z][A-Za-z\- ]{1,40})",
        r"\bi live in\s+([^,.!?]{2,60})",
        r"\bi like\s+([^.!?]{2,100})",
    )

    facts: list[str] = []
    lowered = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _compact_text(match.group(1), max_len=120)
        if "мне " in pattern and "лет" in pattern:
            facts.append(f"Возраст пользователя: {value}")
        elif "меня зовут" in pattern or "мое имя" in pattern or "my name is" in pattern:
            facts.append(f"Имя пользователя: {value}")
        elif "живу в" in pattern or "live in" in pattern:
            facts.append(f"Пользователь живет в: {value}")
        elif "нравится" in pattern or "i like" in pattern:
            facts.append(f"Пользователю нравится: {value}")
        elif "работаю" in pattern or "i am" in pattern:
            facts.append(f"О себе: {value}")

    manual_match = re.search(r"\bзапомни(?:,| что)?\s+(.+)$", text, flags=re.IGNORECASE)
    if manual_match:
        facts.append(f"Явно сохраненный факт: {_compact_text(manual_match.group(1), max_len=160)}")

    if "remember that " in lowered:
        tail = lowered.split("remember that ", maxsplit=1)[1]
        facts.append(f"Явно сохраненный факт: {_compact_text(tail, max_len=160)}")

    unique: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique


@dataclass(slots=True)
class MemoryStore:
    path: Path
    max_facts: int = 100
    max_turns: int = 200
    _data: dict[str, object] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()
        self._data: dict[str, object] = {
            "facts": [],
            "turns": [],
            "updated_at": _utc_now(),
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(parsed, dict):
            return
        facts = parsed.get("facts", [])
        turns = parsed.get("turns", [])
        updated_at = parsed.get("updated_at", _utc_now())
        if not isinstance(facts, list) or not isinstance(turns, list):
            return
        self._data = {
            "facts": [str(x) for x in facts if isinstance(x, str)],
            "turns": [x for x in turns if isinstance(x, dict)],
            "updated_at": str(updated_at),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._data = {"facts": [], "turns": [], "updated_at": _utc_now()}
        self.save()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        facts = self._data["facts"]
        turns = self._data["turns"]
        assert isinstance(facts, list)
        assert isinstance(turns, list)

        for fact in _extract_facts(user_text):
            self._append_unique_fact(fact)

        turn_record = {
            "ts": _utc_now(),
            "user": _compact_text(user_text, max_len=500),
            "assistant": _compact_text(assistant_text, max_len=500),
        }
        turns.append(turn_record)
        if len(turns) > self.max_turns:
            del turns[: len(turns) - self.max_turns]

        facts = self._data["facts"]
        assert isinstance(facts, list)
        if len(facts) > self.max_facts:
            del facts[: len(facts) - self.max_facts]

        self._data["updated_at"] = _utc_now()
        self.save()

    def _append_unique_fact(self, value: str) -> None:
        facts = self._data["facts"]
        assert isinstance(facts, list)

        lowered_existing = {str(item).lower() for item in facts if isinstance(item, str)}
        if value.lower() in lowered_existing:
            return
        facts.append(value)

    def build_context(
        self,
        recent_turns: int = 8,
        recent_facts: int = 20,
    ) -> str:
        facts = self._data["facts"]
        turns = self._data["turns"]
        assert isinstance(facts, list)
        assert isinstance(turns, list)

        selected_facts = [x for x in facts if isinstance(x, str)][-recent_facts:]
        selected_turns = [x for x in turns if isinstance(x, dict)][-recent_turns:]

        lines: list[str] = [
            "ПАМЯТЬ АССИСТЕНТА (используй только как контекст, не выдумывай):"
        ]

        if selected_facts:
            lines.append("Факты о пользователе:")
            for fact in selected_facts:
                lines.append(f"- {fact}")
        else:
            lines.append("Факты о пользователе: (пока нет)")

        if selected_turns:
            lines.append("Недавний диалог:")
            for item in selected_turns:
                user_text = str(item.get("user", ""))
                assistant_text = str(item.get("assistant", ""))
                lines.append(f"- Пользователь: {user_text}")
                lines.append(f"  Ассистент: {assistant_text}")
        else:
            lines.append("Недавний диалог: (пока нет)")

        return "\n".join(lines)

    def human_readable(self, recent_turns: int = 8, recent_facts: int = 20) -> str:
        return self.build_context(recent_turns=recent_turns, recent_facts=recent_facts)

