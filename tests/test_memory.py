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


if __name__ == "__main__":
    unittest.main()

