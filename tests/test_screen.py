import unittest

from assistant.screen import ScreenObserver


class ScreenObserverTests(unittest.TestCase):
    def test_disabled_status(self) -> None:
        observer = ScreenObserver(enabled=False)
        self.assertIn("выключено", observer.status_text().lower())
        self.assertIsNone(observer.build_agent_context("что видно?"))

    def test_set_active_fails_when_unavailable(self) -> None:
        observer = ScreenObserver(enabled=False)
        self.assertFalse(observer.set_active(True))
        self.assertFalse(observer.active)

    def test_describe_screen_when_disabled(self) -> None:
        observer = ScreenObserver(enabled=False)
        text = observer.describe_screen(user_query="что видно?", force_refresh=True)
        self.assertIn("выключено", text.lower())


if __name__ == "__main__":
    unittest.main()

