from __future__ import annotations

from dataclasses import dataclass
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Callable
from urllib.parse import urlparse
import webbrowser

from assistant.protocol import PlannedAction


MAX_OUTPUT_CHARS = 4000
DANGEROUS_PATTERNS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    ":(){:|:&};:",
)


@dataclass(slots=True)
class ActionResult:
    success: bool
    output: str

    def as_feedback(self, action_name: str) -> str:
        status = "OK" if self.success else "ERROR"
        return f"[{status}] {action_name}: {self.output}"


class ActionExecutor:
    def __init__(
        self,
        workspace: Path,
        auto_approve: bool = False,
        allow_outside_workspace: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.auto_approve = auto_approve
        self.allow_outside_workspace = allow_outside_workspace
        self._handlers: dict[str, Callable[[dict[str, Any]], ActionResult]] = {
            "run_shell": self._run_shell,
            "open_url": self._open_url,
            "list_directory": self._list_directory,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "launch_application": self._launch_application,
            "press_hotkey": self._press_hotkey,
            "type_text": self._type_text,
        }
        self._docs: dict[str, str] = {
            "run_shell": "Выполнить shell-команду в рабочей папке. args: {command: str, timeout: int?}",
            "open_url": "Открыть ссылку в браузере. args: {url: str}",
            "list_directory": "Показать содержимое каталога. args: {path: str?}",
            "read_file": "Прочитать файл. args: {path: str, max_chars: int?}",
            "write_file": "Записать файл. args: {path: str, content: str, append: bool?}",
            "launch_application": "Запустить приложение. args: {app: str, args: [str]?}",
            "press_hotkey": "Нажать сочетание клавиш (нужен pyautogui). args: {keys: [str]}",
            "type_text": "Напечатать текст (нужен pyautogui). args: {text: str}",
        }

    def list_actions_for_prompt(self) -> str:
        return json.dumps(self._docs, ensure_ascii=False, indent=2)

    def execute(self, action: PlannedAction) -> ActionResult:
        handler = self._handlers.get(action.name)
        if handler is None:
            return ActionResult(False, f"Неизвестное действие: {action.name}")

        if not self._approve(action):
            return ActionResult(False, "Действие отклонено пользователем.")

        try:
            return handler(action.args)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"Ошибка выполнения: {exc}")

    def _approve(self, action: PlannedAction) -> bool:
        if self.auto_approve:
            return True

        payload = json.dumps(action.args, ensure_ascii=False)
        try:
            answer = input(
                f"\nПодтвердить действие {action.name} с аргументами {payload}? [y/N]: "
            ).strip().lower()
        except EOFError:
            return False
        return answer in {"y", "yes", "д", "да"}

    def _run_shell(self, args: dict[str, Any]) -> ActionResult:
        command = args.get("command")
        timeout = args.get("timeout", 20)

        if not isinstance(command, str) or not command.strip():
            return ActionResult(False, "Нужен аргумент command (непустая строка).")
        if not isinstance(timeout, int) or timeout <= 0:
            timeout = 20

        lowered = command.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in lowered:
                return ActionResult(False, "Команда заблокирована как потенциально опасная.")

        completed = subprocess.run(
            command,
            cwd=str(self.workspace),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        lines = [part for part in [stdout, stderr] if part]
        output = "\n".join(lines) if lines else "(пустой вывод)"
        output = _truncate(output)
        if completed.returncode == 0:
            return ActionResult(True, output)
        return ActionResult(False, f"Код выхода {completed.returncode}. {output}")

    def _open_url(self, args: dict[str, Any]) -> ActionResult:
        url = args.get("url")
        if not isinstance(url, str):
            return ActionResult(False, "Нужен аргумент url (строка).")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ActionResult(False, "Поддерживаются только http/https URL.")

        opened = webbrowser.open(url, new=2)
        if opened:
            return ActionResult(True, f"Открыл ссылку: {url}")
        return ActionResult(False, f"Не удалось открыть ссылку: {url}")

    def _list_directory(self, args: dict[str, Any]) -> ActionResult:
        path = args.get("path", ".")
        if not isinstance(path, str):
            return ActionResult(False, "Аргумент path должен быть строкой.")
        resolved = self._resolve_path(path, expect_exists=True)
        if not resolved.is_dir():
            return ActionResult(False, f"Это не каталог: {resolved}")

        entries = []
        for entry in sorted(resolved.iterdir(), key=lambda item: item.name.lower()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{suffix}")
        return ActionResult(True, "\n".join(entries) if entries else "(пусто)")

    def _read_file(self, args: dict[str, Any]) -> ActionResult:
        path = args.get("path")
        max_chars = args.get("max_chars", MAX_OUTPUT_CHARS)
        if not isinstance(path, str):
            return ActionResult(False, "Нужен аргумент path (строка).")
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = MAX_OUTPUT_CHARS

        resolved = self._resolve_path(path, expect_exists=True)
        if not resolved.is_file():
            return ActionResult(False, f"Это не файл: {resolved}")

        content = resolved.read_text(encoding="utf-8", errors="replace")
        return ActionResult(True, _truncate(content, max_chars=max_chars))

    def _write_file(self, args: dict[str, Any]) -> ActionResult:
        path = args.get("path")
        content = args.get("content", "")
        append = args.get("append", False)
        if not isinstance(path, str):
            return ActionResult(False, "Нужен аргумент path (строка).")
        if not isinstance(content, str):
            return ActionResult(False, "Аргумент content должен быть строкой.")
        if not isinstance(append, bool):
            append = False

        resolved = self._resolve_path(path, expect_exists=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with resolved.open(mode, encoding="utf-8") as fh:
            fh.write(content)
        return ActionResult(True, f"Записал файл: {resolved}")

    def _launch_application(self, args: dict[str, Any]) -> ActionResult:
        app = args.get("app")
        app_args = args.get("args", [])
        if not isinstance(app, str) or not app:
            return ActionResult(False, "Нужен аргумент app (строка).")
        if not isinstance(app_args, list) or not all(isinstance(x, str) for x in app_args):
            return ActionResult(False, "Аргумент args должен быть списком строк.")

        system = platform.system().lower()
        if system == "windows":
            subprocess.Popen(["cmd", "/c", "start", "", app, *app_args], cwd=str(self.workspace))
        elif system == "darwin":
            subprocess.Popen(["open", "-a", app, *app_args], cwd=str(self.workspace))
        else:
            subprocess.Popen([app, *app_args], cwd=str(self.workspace))
        return ActionResult(True, f"Запустил приложение: {app}")

    def _press_hotkey(self, args: dict[str, Any]) -> ActionResult:
        keys = args.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(x, str) for x in keys):
            return ActionResult(False, "Нужен args.keys как непустой список строк.")

        try:
            import pyautogui  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"pyautogui недоступен: {exc}")

        pyautogui.hotkey(*keys)
        return ActionResult(True, f"Нажал горячие клавиши: {' + '.join(keys)}")

    def _type_text(self, args: dict[str, Any]) -> ActionResult:
        text = args.get("text")
        if not isinstance(text, str):
            return ActionResult(False, "Нужен аргумент text (строка).")

        try:
            import pyautogui  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"pyautogui недоступен: {exc}")

        pyautogui.write(text)
        return ActionResult(True, "Текст напечатан.")

    def _resolve_path(self, raw_path: str, *, expect_exists: bool) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (self.workspace / path).resolve()
        else:
            path = path.resolve()

        if not self.allow_outside_workspace:
            try:
                path.relative_to(self.workspace)
            except ValueError as exc:
                raise ValueError(
                    "Доступ к путям вне рабочей директории запрещен. "
                    "Разрешите через параметр allow_outside_workspace."
                ) from exc

        if expect_exists and not path.exists():
            raise ValueError(f"Путь не существует: {path}")
        return path


def _truncate(value: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[output truncated]..."

