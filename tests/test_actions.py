from pathlib import Path
import tempfile
import unittest

from assistant.actions import ActionExecutor
from assistant.protocol import PlannedAction


class ActionExecutorTests(unittest.TestCase):
    def test_write_and_read_file_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            executor = ActionExecutor(workspace=workspace, auto_approve=True)

            write_result = executor.execute(
                PlannedAction(
                    name="write_file",
                    args={"path": "notes/a.txt", "content": "hello"},
                )
            )
            self.assertTrue(write_result.success)

            read_result = executor.execute(
                PlannedAction(name="read_file", args={"path": "notes/a.txt"})
            )
            self.assertTrue(read_result.success)
            self.assertIn("hello", read_result.output)

    def test_block_outside_workspace_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            executor = ActionExecutor(workspace=workspace, auto_approve=True)

            result = executor.execute(
                PlannedAction(name="read_file", args={"path": "/etc/hosts"})
            )
            self.assertFalse(result.success)
            self.assertIn("вне рабочей директории", result.output.lower())


if __name__ == "__main__":
    unittest.main()

