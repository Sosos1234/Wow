from pathlib import Path
import tempfile
import unittest

from assistant.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_memory_persists_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            store = MemoryStore(path=memory_file, max_facts=20, max_turns=20)

            store.add_turn(
                "Меня зовут Антон, мне нравится Python",
                "Приятно познакомиться, Антон!",
            )
            snapshot = store.build_context(recent_turns=5, recent_facts=5)
            self.assertIn("Имя пользователя: Антон", snapshot)
            self.assertIn("Пользователю нравится: Python", snapshot)

            reloaded = MemoryStore(path=memory_file, max_facts=20, max_turns=20)
            snapshot_2 = reloaded.build_context(recent_turns=5, recent_facts=5)
            self.assertIn("Имя пользователя: Антон", snapshot_2)
            self.assertIn("Пользователь: Меня зовут Антон", snapshot_2)

    def test_memory_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            store = MemoryStore(path=memory_file)
            store.add_turn("Запомни что я люблю кофе", "Запомнил")
            store.clear()

            snapshot = store.build_context(recent_turns=5, recent_facts=5)
            self.assertIn("Факты о пользователе: (пока нет)", snapshot)
            self.assertIn("Недавний диалог: (пока нет)", snapshot)

    def test_semantic_retrieval_by_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            store = MemoryStore(path=memory_file, max_turns=2, max_facts=20)

            store.add_turn(
                "Меня зовут Лена, мне нравится йога и пилатес",
                "Класс!",
            )
            store.add_turn("Сегодня была тренировка по бегу", "Отлично")
            store.add_turn("Обсудим музыку", "Да")

            context = store.build_context(
                recent_turns=2,
                recent_facts=5,
                query_text="Что по йоге?",
                relevant_items=5,
            )
            self.assertIn("Релевантные воспоминания по текущему запросу:", context)
            self.assertIn("йога", context.lower())

    def test_manual_fact_and_style_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            store = MemoryStore(path=memory_file)

            store.add_manual_fact("У меня аллергия на арахис")
            store.add_style_preference("Отвечай кратко и дружелюбно")

            context = store.build_context(recent_turns=5, recent_facts=10)
            self.assertIn("аллергия", context.lower())
            self.assertIn("дружелюбно", context.lower())

    def test_extract_style_preferences_from_dialog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "memory.json"
            store = MemoryStore(path=memory_file)

            store.add_turn("Отвечай кратко и пиши на русском", "Хорошо")
            context = store.build_context(recent_turns=5, recent_facts=10)
            self.assertIn("короткие ответы", context.lower())
            self.assertIn("на русском", context.lower())


if __name__ == "__main__":
    unittest.main()

