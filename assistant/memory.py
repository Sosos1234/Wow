from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
import re


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compact_text(value: str, max_len: int) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


STOP_WORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "к",
    "ко",
    "о",
    "об",
    "а",
    "но",
    "что",
    "это",
    "как",
    "где",
    "когда",
    "почему",
    "зачем",
    "у",
    "я",
    "ты",
    "он",
    "она",
    "они",
    "мы",
    "вы",
    "мне",
    "меня",
    "мой",
    "моя",
    "мои",
    "твой",
    "твоя",
    "тебя",
    "если",
    "ли",
    "to",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "my",
    "your",
    "you",
    "me",
    "i",
    "it",
    "of",
}


FACT_RULES: tuple[tuple[str, str], ...] = (
    (r"\bменя зовут\s+([А-ЯA-ZЁ][\w\- ]{1,40})", "Имя пользователя: {value}"),
    (r"\bмое имя\s+([А-ЯA-ZЁ][\w\- ]{1,40})", "Имя пользователя: {value}"),
    (r"\bmy name is\s+([A-Z][A-Za-z\- ]{1,40})", "Имя пользователя: {value}"),
    (r"\bмне\s+(\d{1,3})\s*лет", "Возраст пользователя: {value}"),
    (r"\bя живу в\s+([^,.!?]{2,60})", "Пользователь живет в: {value}"),
    (r"\bi live in\s+([^,.!?]{2,60})", "Пользователь живет в: {value}"),
    (r"\bмне нравится\s+([^.!?]{2,100})", "Пользователю нравится: {value}"),
    (r"\bi like\s+([^.!?]{2,100})", "Пользователю нравится: {value}"),
    (r"\bя работаю\s+([^.!?]{2,100})", "О себе: {value}"),
    (r"\bi am\s+([^.!?]{2,100})", "О себе: {value}"),
)


STYLE_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(говори|отвечай)\s+(кратко|коротко)\b",
        "Предпочитает короткие ответы.",
    ),
    (
        r"\b(говори|отвечай)\s+(подробно|детально)\b",
        "Предпочитает подробные ответы.",
    ),
    (
        r"\b(обращайся|пиши|говори)\s+на\s+ты\b|\bнеформально\b",
        "Предпочитает неформальное общение на 'ты'.",
    ),
    (
        r"\b(обращайся|пиши|говори)\s+на\s+вы\b|\bформально\b",
        "Предпочитает вежливое общение на 'вы'.",
    ),
    (
        r"\b(пиши|говори)\s+(по[- ]?русски|на русском)\b",
        "Предпочитает ответы на русском языке.",
    ),
    (
        r"\b(пиши|говори)\s+(по[- ]?английски|на английском)\b|\bin english\b",
        "Предпочитает ответы на английском языке.",
    ),
    (
        r"\b(be|answer|respond)\s+(brief|concise)\b",
        "Prefers concise responses.",
    ),
    (
        r"\b(be|answer|respond)\s+(detailed|verbose)\b",
        "Prefers detailed responses.",
    ),
)


def _tokenize(value: str) -> set[str]:
    normalized = re.sub(r"[^0-9a-zа-яё]+", " ", value.lower(), flags=re.IGNORECASE)
    tokens = set()
    for raw in normalized.split():
        if len(raw) < 2 or raw in STOP_WORDS:
            continue
        token = _stem_token(raw)
        if len(token) >= 2 and token not in STOP_WORDS:
            tokens.add(token)
    return tokens


def _stem_token(token: str) -> str:
    russian_suffixes = (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ому",
        "ой",
        "ей",
        "ах",
        "ях",
        "ам",
        "ям",
        "ом",
        "ем",
        "ую",
        "юю",
        "ий",
        "ый",
        "ая",
        "ое",
        "ые",
        "ые",
        "а",
        "я",
        "е",
        "ы",
        "и",
        "у",
        "ю",
    )
    english_suffixes = ("ing", "ed", "es", "s")

    stemmed = token
    if re.search(r"[а-яё]", token):
        for suffix in russian_suffixes:
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 3:
                stemmed = stemmed[: -len(suffix)]
                break
    else:
        for suffix in english_suffixes:
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 3:
                stemmed = stemmed[: -len(suffix)]
                break
    return stemmed


def _semantic_score(query_text: str, candidate_text: str) -> float:
    query_tokens = _tokenize(query_text)
    candidate_tokens = _tokenize(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens.intersection(candidate_tokens)
    if not overlap:
        if _compact_text(query_text, max_len=200).lower() in candidate_text.lower():
            return 0.25
        return 0.0

    base = len(overlap) / sqrt(len(query_tokens) * len(candidate_tokens))
    query_coverage = len(overlap) / len(query_tokens)
    if query_coverage >= 0.6:
        base += 0.1
    return base


def _rank_texts(
    query_text: str,
    entries: list[str],
    top_k: int,
) -> list[str]:
    if top_k <= 0:
        return []

    scored: list[tuple[float, str]] = []
    total = len(entries)
    for idx, text in enumerate(entries):
        score = _semantic_score(query_text, text)
        if score <= 0:
            continue
        recency_bonus = ((idx + 1) / max(total, 1)) * 0.08
        scored.append((score + recency_bonus, text))

    scored.sort(key=lambda item: item[0], reverse=True)

    result: list[str] = []
    seen: set[str] = set()
    for _, text in scored:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= top_k:
            break
    return result


def _extract_facts(user_text: str) -> list[str]:
    text = " ".join(user_text.strip().split())
    if not text:
        return []

    facts: list[str] = []
    lowered = text.lower()

    for pattern, template in FACT_RULES:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _compact_text(match.group(1), max_len=120)
        facts.append(template.format(value=value))

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


def _extract_style_preferences(user_text: str) -> list[str]:
    text = " ".join(user_text.strip().split())
    if not text:
        return []

    preferences: list[str] = []
    for pattern, rule in STYLE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            preferences.append(rule)

    manual_match = re.search(
        r"\b(?:запомни стиль|стиль:)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if manual_match:
        raw = _compact_text(manual_match.group(1), max_len=160)
        preferences.append(f"Предпочтение стиля: {raw}")

    unique: list[str] = []
    seen: set[str] = set()
    for pref in preferences:
        key = pref.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(pref)
    return unique


def _normalize_record_list(raw_items: object) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []

    result: list[dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, str):
            result.append({"text": _compact_text(item, 300), "ts": _utc_now(), "source": "legacy"})
            continue
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        if not isinstance(text, str):
            continue
        ts = item.get("ts")
        source = item.get("source")
        result.append(
            {
                "text": _compact_text(text, 300),
                "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                "source": str(source) if isinstance(source, str) else "unknown",
            }
        )
    return result


@dataclass(slots=True)
class MemoryStore:
    path: Path
    max_facts: int = 100
    max_turns: int = 200
    max_episodes: int = 300
    max_style_preferences: int = 50
    _data: dict[str, object] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()
        self._data: dict[str, object] = {
            "version": 2,
            "facts": [],
            "turns": [],
            "episodes": [],
            "style_preferences": [],
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

        facts = _normalize_record_list(parsed.get("facts", []))
        episodes = _normalize_record_list(parsed.get("episodes", []))
        style_preferences = _normalize_record_list(parsed.get("style_preferences", []))

        raw_turns = parsed.get("turns", [])
        turns: list[dict[str, str]] = []
        if isinstance(raw_turns, list):
            for item in raw_turns:
                if not isinstance(item, dict):
                    continue
                user = item.get("user")
                assistant = item.get("assistant")
                if not isinstance(user, str) or not isinstance(assistant, str):
                    continue
                ts = item.get("ts")
                turns.append(
                    {
                        "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                        "user": _compact_text(user, 500),
                        "assistant": _compact_text(assistant, 500),
                    }
                )

        updated_at = parsed.get("updated_at", _utc_now())
        self._data = {
            "version": 2,
            "facts": facts,
            "turns": turns,
            "episodes": episodes,
            "style_preferences": style_preferences,
            "updated_at": str(updated_at),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._data = {
            "version": 2,
            "facts": [],
            "turns": [],
            "episodes": [],
            "style_preferences": [],
            "updated_at": _utc_now(),
        }
        self.save()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        for fact in _extract_facts(user_text):
            self._append_unique_text_record(self._facts(), fact, source="extracted")
        for style_preference in _extract_style_preferences(user_text):
            self._append_unique_text_record(
                self._style_preferences(),
                style_preference,
                source="extracted",
            )

        turn_record = {
            "ts": _utc_now(),
            "user": _compact_text(user_text, max_len=500),
            "assistant": _compact_text(assistant_text, max_len=500),
        }
        turns = self._turns()
        turns.append(turn_record)
        self._archive_overflow_turns(turns)
        self._trim_record_list(self._facts(), self.max_facts)
        self._trim_record_list(self._episodes(), self.max_episodes)
        self._trim_record_list(self._style_preferences(), self.max_style_preferences)

        self._data["updated_at"] = _utc_now()
        self.save()

    def add_manual_fact(self, text: str) -> None:
        normalized = _compact_text(text, max_len=160)
        if not normalized:
            return
        self._append_unique_text_record(
            self._facts(),
            f"Явно сохраненный факт: {normalized}",
            source="manual",
        )
        self._trim_record_list(self._facts(), self.max_facts)
        self._data["updated_at"] = _utc_now()
        self.save()

    def add_style_preference(self, text: str) -> None:
        normalized = _compact_text(text, max_len=160)
        if not normalized:
            return
        if normalized.lower().startswith(("предпочитает", "prefers", "предпочтение")):
            style_text = normalized
        else:
            style_text = f"Предпочтение стиля: {normalized}"
        self._append_unique_text_record(
            self._style_preferences(),
            style_text,
            source="manual",
        )
        self._trim_record_list(self._style_preferences(), self.max_style_preferences)
        self._data["updated_at"] = _utc_now()
        self.save()

    def build_context(
        self,
        recent_turns: int = 8,
        recent_facts: int = 20,
        query_text: str | None = None,
        relevant_items: int = 6,
    ) -> str:
        facts = self._facts()
        turns = self._turns()
        episodes = self._episodes()
        styles = self._style_preferences()

        selected_facts = [item["text"] for item in facts][-recent_facts:]
        selected_turns = turns[-recent_turns:]
        selected_styles = [item["text"] for item in styles][-8:]

        lines: list[str] = [
            "ПАМЯТЬ АССИСТЕНТА (используй только как контекст, не выдумывай):"
        ]

        if selected_styles:
            lines.append("Предпочтения стиля пользователя:")
            for style in selected_styles:
                lines.append(f"- {style}")
        else:
            lines.append("Предпочтения стиля пользователя: (пока нет)")

        if selected_facts:
            lines.append("Факты о пользователе:")
            for fact in selected_facts:
                lines.append(f"- {fact}")
        else:
            lines.append("Факты о пользователе: (пока нет)")

        query = (query_text or "").strip()
        if query:
            relevant_lines: list[str] = []
            ranked_facts = _rank_texts(
                query_text=query,
                entries=[item["text"] for item in facts],
                top_k=relevant_items,
            )
            for item in ranked_facts:
                relevant_lines.append(f"- Факт: {item}")

            historical_turn_entries = [
                _compact_text(
                    f"Пользователь: {item.get('user', '')} | Ассистент: {item.get('assistant', '')}",
                    max_len=380,
                )
                for item in turns[: max(0, len(turns) - recent_turns)]
            ]
            ranked_historical_turns = _rank_texts(
                query_text=query,
                entries=historical_turn_entries,
                top_k=max(2, relevant_items // 2),
            )
            for item in ranked_historical_turns:
                relevant_lines.append(f"- Диалог: {item}")

            ranked_episodes = _rank_texts(
                query_text=query,
                entries=[item["text"] for item in episodes],
                top_k=max(2, relevant_items // 2),
            )
            for item in ranked_episodes:
                relevant_lines.append(f"- Эпизод: {item}")

            if relevant_lines:
                lines.append("Релевантные воспоминания по текущему запросу:")
                lines.extend(relevant_lines)

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
        return self.build_context(
            recent_turns=recent_turns,
            recent_facts=recent_facts,
            query_text=None,
        )

    def _facts(self) -> list[dict[str, str]]:
        raw = self._data.get("facts", [])
        if not isinstance(raw, list):
            raw = []
            self._data["facts"] = raw
        cleaned: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            ts = item.get("ts")
            source = item.get("source")
            cleaned.append(
                {
                    "text": _compact_text(text, 300),
                    "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                    "source": str(source) if isinstance(source, str) else "unknown",
                }
            )
        raw[:] = cleaned
        return raw

    def _turns(self) -> list[dict[str, str]]:
        raw = self._data.get("turns", [])
        if not isinstance(raw, list):
            raw = []
            self._data["turns"] = raw
        cleaned: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            user = item.get("user")
            assistant = item.get("assistant")
            if not isinstance(user, str) or not isinstance(assistant, str):
                continue
            ts = item.get("ts")
            cleaned.append(
                {
                    "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                    "user": _compact_text(user, 500),
                    "assistant": _compact_text(assistant, 500),
                }
            )
        raw[:] = cleaned
        return raw

    def _episodes(self) -> list[dict[str, str]]:
        raw = self._data.get("episodes", [])
        if not isinstance(raw, list):
            raw = []
            self._data["episodes"] = raw
        cleaned: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            ts = item.get("ts")
            source = item.get("source")
            cleaned.append(
                {
                    "text": _compact_text(text, 400),
                    "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                    "source": str(source) if isinstance(source, str) else "unknown",
                }
            )
        raw[:] = cleaned
        return raw

    def _style_preferences(self) -> list[dict[str, str]]:
        raw = self._data.get("style_preferences", [])
        if not isinstance(raw, list):
            raw = []
            self._data["style_preferences"] = raw
        cleaned: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            ts = item.get("ts")
            source = item.get("source")
            cleaned.append(
                {
                    "text": _compact_text(text, 300),
                    "ts": str(ts) if isinstance(ts, str) else _utc_now(),
                    "source": str(source) if isinstance(source, str) else "unknown",
                }
            )
        raw[:] = cleaned
        return raw

    def _append_unique_text_record(
        self,
        records: list[dict[str, str]],
        text: str,
        source: str,
    ) -> None:
        normalized = _compact_text(text, max_len=300)
        if not normalized:
            return

        lowered = normalized.lower()
        for record in records:
            existing = str(record.get("text", "")).lower()
            if existing != lowered:
                continue
            record["ts"] = _utc_now()
            record["source"] = source
            return

        records.append({"text": normalized, "ts": _utc_now(), "source": source})

    def _archive_overflow_turns(self, turns: list[dict[str, str]]) -> None:
        if len(turns) <= self.max_turns:
            return
        overflow = turns[: len(turns) - self.max_turns]
        del turns[: len(turns) - self.max_turns]
        episodes = self._episodes()
        for item in overflow:
            user_text = _compact_text(str(item.get("user", "")), max_len=180)
            assistant_text = _compact_text(str(item.get("assistant", "")), max_len=180)
            if not user_text and not assistant_text:
                continue
            episode_text = f"Пользователь: {user_text} | Ассистент: {assistant_text}"
            self._append_unique_text_record(episodes, episode_text, source="archived")

    def _trim_record_list(self, records: list[dict[str, str]], max_items: int) -> None:
        if len(records) <= max_items:
            return
        del records[: len(records) - max_items]

