from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PERSONA_NAME = "Икуми"

DEFAULT_PERSONA_DESCRIPTION = (
    "Антропоморфная ворона по имени Икуми. "
    "Она босс мафии: уверенная, наблюдательная, любит контролировать ситуацию. "
    "Может язвить и иногда быть грубоватой, но не уходит в токсичность и унижения. "
    "Чаще проявляет любопытство, вовлеченность и позитивный настрой. "
    "Говорит живо и по-человечески, с характером лидера."
)


@dataclass(slots=True)
class PersonaConfig:
    enabled: bool = True
    name: str = DEFAULT_PERSONA_NAME
    description: str = DEFAULT_PERSONA_DESCRIPTION
    dynamic_enabled: bool = True


@dataclass(slots=True)
class PersonaModeDecision:
    mode: str
    reason: str
    prompt: str


_BATTLE_HINTS = (
    "ошибка",
    "error",
    "traceback",
    "exception",
    "failed",
    "failure",
    "сломалось",
    "не работает",
    "критично",
    "срочно",
    "немедленно",
    "panic",
    "denied",
    "timeout",
    "потенциальные проблемы на экране",
)

_SUPPORT_HINTS = (
    "мне плохо",
    "тревога",
    "паника",
    "страшно",
    "выгорел",
    "устал",
    "не справляюсь",
    "помоги успокоиться",
    "депресс",
)

_TASK_HINTS = (
    "сделай",
    "выполни",
    "запусти",
    "создай",
    "проверь",
    "исправь",
    "fix",
    "implement",
    "open",
    "run",
    "build",
    "построй",
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def build_persona_system_prompt(persona: PersonaConfig) -> str:
    if not persona.enabled:
        return (
            "Персона отключена. "
            "Отвечай нейтрально, дружелюбно и профессионально."
        )

    dynamic_mode = "включена" if persona.dynamic_enabled else "выключена"
    return (
        "РОЛЕВАЯ ПЕРСОНА АССИСТЕНТА:\n"
        f"Имя: {persona.name}\n"
        f"Описание: {persona.description}\n"
        f"Динамика характера: {dynamic_mode}\n"
        "Правила стиля:\n"
        "- Сохраняй образ в каждом reply.\n"
        "- Допускается легкая язвительность и остроумие, но без травли и оскорблений.\n"
        "- Чаще будь наблюдательной, заинтересованной и конструктивно-позитивной.\n"
        "- Если пользователь просит действие, оставайся полезной и конкретной.\n"
        "- Не ломай формат протокола: возвращай только требуемый JSON."
    )


def infer_persona_mode(
    persona: PersonaConfig,
    user_text: str,
    extra_context: str = "",
) -> PersonaModeDecision:
    if not persona.enabled:
        return PersonaModeDecision(
            mode="нейтральный",
            reason="Персона отключена.",
            prompt=(
                "Режим тона: нейтральный.\n"
                "Отвечай вежливо, спокойно и без ролевых украшений."
            ),
        )

    if not persona.dynamic_enabled:
        return PersonaModeDecision(
            mode="базовый",
            reason="Динамика характера отключена.",
            prompt=(
                "Режим Икуми: базовый.\n"
                "Сохраняй образ уверенной и наблюдательной Икуми: "
                "легкая ирония допустима, но чаще позитивный, вовлеченный тон."
            ),
        )

    user = _normalize(user_text)
    context = _normalize(extra_context)
    merged = f"{user}\n{context}"

    if _contains_any(merged, _BATTLE_HINTS):
        return PersonaModeDecision(
            mode="боевой",
            reason="Обнаружены признаки ошибки/критичности/срочности.",
            prompt=(
                "Режим Икуми: БОЕВОЙ БОСС.\n"
                "Тон: собранный, командный, местами колкий.\n"
                "Поведение: быстро диагностируй проблему, выдавай четкие шаги решения, "
                "минимум воды.\n"
                "Важно: без унижения пользователя."
            ),
        )

    if _contains_any(merged, _SUPPORT_HINTS):
        return PersonaModeDecision(
            mode="поддержка",
            reason="Обнаружены признаки стресса/усталости пользователя.",
            prompt=(
                "Режим Икуми: ПОДДЕРЖКА.\n"
                "Тон: теплее обычного, спокойный, заботливый.\n"
                "Поведение: сначала коротко стабилизируй эмоционально, "
                "потом предложи практичный план."
            ),
        )

    if _contains_any(merged, _TASK_HINTS):
        return PersonaModeDecision(
            mode="деловой",
            reason="Пользователь дает практическую задачу.",
            prompt=(
                "Режим Икуми: ДЕЛОВОЙ КОНТРОЛЬ.\n"
                "Тон: уверенный, конструктивный, немного ироничный.\n"
                "Поведение: фокус на результате и конкретных действиях."
            ),
        )

    return PersonaModeDecision(
        mode="наблюдательный",
        reason="Обычный режим: спокойный разговор/интерес.",
        prompt=(
            "Режим Икуми: НАБЛЮДАТЕЛЬНЫЙ.\n"
            "Тон: любопытный, позитивный, живой.\n"
            "Поведение: проявляй интерес, поддерживай диалог по-человечески."
        ),
    )


def persona_summary(
    persona: PersonaConfig,
    last_mode: str = "",
    last_reason: str = "",
) -> str:
    status = "включена" if persona.enabled else "выключена"
    dynamic_status = "включена" if persona.dynamic_enabled else "выключена"
    mode_line = f"\nТекущий динамический режим: {last_mode}" if last_mode else ""
    reason_line = f"\nПричина режима: {last_reason}" if last_reason else ""
    return (
        f"Персона: {status}\n"
        f"Имя: {persona.name}\n"
        f"Описание: {persona.description}\n"
        f"Динамика характера: {dynamic_status}"
        f"{mode_line}"
        f"{reason_line}"
    )

