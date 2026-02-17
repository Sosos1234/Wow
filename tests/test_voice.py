import unittest

from assistant.voice import is_exit_command


class VoiceHelpersTests(unittest.TestCase):
    def test_exit_command_english(self) -> None:
        self.assertTrue(is_exit_command("exit"))
        self.assertTrue(is_exit_command(" quit "))

    def test_exit_command_russian(self) -> None:
        self.assertTrue(is_exit_command("выход"))
        self.assertTrue(is_exit_command("  выйти  "))
        self.assertTrue(is_exit_command("стоп"))

    def test_non_exit_command(self) -> None:
        self.assertFalse(is_exit_command("открой браузер"))
        self.assertFalse(is_exit_command("ex it"))


if __name__ == "__main__":
    unittest.main()

