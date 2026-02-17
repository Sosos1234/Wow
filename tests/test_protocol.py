import unittest

from assistant.protocol import parse_assistant_message


class ProtocolParsingTests(unittest.TestCase):
    def test_parse_valid_json(self) -> None:
        raw = '{"reply":"Привет","actions":[{"name":"list_directory","args":{"path":"."}}]}'
        parsed = parse_assistant_message(raw)
        self.assertEqual(parsed.reply, "Привет")
        self.assertEqual(len(parsed.actions), 1)
        self.assertEqual(parsed.actions[0].name, "list_directory")
        self.assertEqual(parsed.actions[0].args, {"path": "."})

    def test_parse_json_in_code_fence(self) -> None:
        raw = """```json
{"reply":"Готово","actions":[]}
```"""
        parsed = parse_assistant_message(raw)
        self.assertEqual(parsed.reply, "Готово")
        self.assertEqual(parsed.actions, [])

    def test_parse_fallback_plain_text(self) -> None:
        raw = "Обычный текст без JSON"
        parsed = parse_assistant_message(raw)
        self.assertEqual(parsed.reply, raw)
        self.assertEqual(parsed.actions, [])


if __name__ == "__main__":
    unittest.main()

