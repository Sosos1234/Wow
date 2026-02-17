import unittest

from assistant.persona import (
    PersonaConfig,
    build_persona_system_prompt,
    infer_persona_mode,
    persona_summary,
)


class PersonaTests(unittest.TestCase):
    def test_persona_prompt_enabled(self) -> None:
        persona = PersonaConfig(
            enabled=True,
            name="Икуми",
            description="Антропоморфная ворона, босс мафии.",
        )
        prompt = build_persona_system_prompt(persona)
        self.assertIn("Икуми", prompt)
        self.assertIn("РОЛЕВАЯ ПЕРСОНА", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("Динамика характера", prompt)

    def test_persona_prompt_disabled(self) -> None:
        persona = PersonaConfig(enabled=False, name="x", description="y")
        prompt = build_persona_system_prompt(persona)
        self.assertIn("Персона отключена", prompt)

    def test_persona_summary(self) -> None:
        persona = PersonaConfig(
            enabled=True,
            name="Икуми",
            description="Любопытная и позитивная.",
        )
        summary = persona_summary(
            persona,
            last_mode="наблюдательный",
            last_reason="Обычный режим",
        )
        self.assertIn("Икуми", summary)
        self.assertIn("включена", summary)
        self.assertIn("наблюдательный", summary)

    def test_infer_battle_mode(self) -> None:
        persona = PersonaConfig(enabled=True, name="Икуми", description="x", dynamic_enabled=True)
        decision = infer_persona_mode(
            persona=persona,
            user_text="Срочно, у меня ошибка и traceback в приложении",
            extra_context="",
        )
        self.assertEqual(decision.mode, "боевой")
        self.assertIn("БОЕВОЙ", decision.prompt)

    def test_infer_support_mode(self) -> None:
        persona = PersonaConfig(enabled=True, name="Икуми", description="x", dynamic_enabled=True)
        decision = infer_persona_mode(
            persona=persona,
            user_text="Мне плохо, паника, я не справляюсь",
            extra_context="",
        )
        self.assertEqual(decision.mode, "поддержка")
        self.assertIn("ПОДДЕРЖКА", decision.prompt)

    def test_infer_static_mode_when_dynamic_disabled(self) -> None:
        persona = PersonaConfig(enabled=True, name="Икуми", description="x", dynamic_enabled=False)
        decision = infer_persona_mode(
            persona=persona,
            user_text="Срочно исправь ошибку",
            extra_context="",
        )
        self.assertEqual(decision.mode, "базовый")


if __name__ == "__main__":
    unittest.main()

