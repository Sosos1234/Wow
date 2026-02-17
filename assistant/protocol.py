from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(slots=True)
class PlannedAction:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedAssistantMessage:
    reply: str
    actions: list[PlannedAction]
    raw: str


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def try_parse_json(text: str) -> dict[str, Any] | None:
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_assistant_message(text: str) -> ParsedAssistantMessage:
    parsed = try_parse_json(text)
    if not parsed:
        return ParsedAssistantMessage(reply=text.strip(), actions=[], raw=text)

    reply = parsed.get("reply")
    if not isinstance(reply, str):
        reply = ""

    actions_data = parsed.get("actions", [])
    actions: list[PlannedAction] = []

    if isinstance(actions_data, list):
        for item in actions_data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            args = item.get("args", {})
            if isinstance(name, str) and isinstance(args, dict):
                actions.append(PlannedAction(name=name, args=args))

    if not reply:
        reply = "Готово."

    return ParsedAssistantMessage(reply=reply, actions=actions, raw=text)

