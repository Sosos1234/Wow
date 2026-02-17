import unittest

from assistant.persona import (
    PersonaConfig,
    build_persona_system_prompt,
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
        summary = persona_summary(persona)
        self.assertIn("Икуми", summary)
        self.assertIn("включена", summary)


if __name__ == "__main__":
    unittest.main()

