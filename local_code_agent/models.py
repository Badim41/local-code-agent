from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpt_project_prompter import DEFAULT_MODEL

DEFAULT_IGNORE_FOLDERS: tuple[str, ...] = (
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "htmlcov",
    ".agents",
    ".deepsearch",
    ".get_gpt_project",
    ".replacer_backups",
)

DEFAULT_IGNORE_FILES: tuple[str, ...] = (
    "secret.py",
    "get_structure.py",
    "replacer.py",
    ".env",
    "prompt_for_project.txt",
)

MAX_READ_FILES_TOTAL: int = 30
MAX_READ_IMAGES_TOTAL: int = 10
MAX_IMAGE_FILE_BYTES: int = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Конфигурация одного зарегистрированного на ПК проекта."""

    id: str
    name: str
    path: Path
    ignore_folders: tuple[str, ...] = DEFAULT_IGNORE_FOLDERS
    ignore_files: tuple[str, ...] = DEFAULT_IGNORE_FILES
    model: str = DEFAULT_MODEL
    detail_level: int = 3
    max_structure_chars: int = 60_000
    max_prompt_chars: int = 500_000
    use_git: bool = True
    allow_create_new_files: bool = True
    default_powershell_timeout: int = 120
    max_powershell_timeout: int = 900

    def to_dict(self) -> dict[str, Any]:
        """Возвращает JSON-представление проекта для API и projects.json."""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path.as_posix(),
            "exists": self.path.is_dir(),
            "ignore_folders": list(self.ignore_folders),
            "ignore_files": list(self.ignore_files),
            "model": self.model,
            "detail_level": self.detail_level,
            "max_structure_chars": self.max_structure_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "use_git": self.use_git,
            "allow_create_new_files": self.allow_create_new_files,
            "default_powershell_timeout": self.default_powershell_timeout,
            "max_powershell_timeout": self.max_powershell_timeout,
        }

    def to_storage_dict(self) -> dict[str, Any]:
        """Возвращает словарь для сохранения в projects.json без вычисляемого поля exists."""
        data = self.to_dict()
        data.pop("exists", None)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProjectConfig:
        """Создаёт ProjectConfig из JSON-словаря."""
        pid = str(raw.get("id", "")).strip()
        name = str(raw.get("name", "")).strip() or pid
        path_raw = str(raw.get("path", "")).strip()
        if not pid or not path_raw:
            raise ValueError("Проект должен содержать непустые поля 'id' и 'path'.")

        ignore_folders_raw = raw.get("ignore_folders")
        ignore_files_raw = raw.get("ignore_files")
        return cls(
            id=pid,
            name=name,
            path=Path(path_raw).expanduser().resolve(),
            ignore_folders=(
                tuple(str(x) for x in ignore_folders_raw)
                if isinstance(ignore_folders_raw, list)
                else DEFAULT_IGNORE_FOLDERS
            ),
            ignore_files=(
                tuple(str(x) for x in ignore_files_raw)
                if isinstance(ignore_files_raw, list)
                else DEFAULT_IGNORE_FILES
            ),
            model=str(raw.get("model") or DEFAULT_MODEL),
            detail_level=int(raw.get("detail_level", 3)),
            max_structure_chars=int(raw.get("max_structure_chars", 60_000)),
            max_prompt_chars=int(raw.get("max_prompt_chars", 500_000)),
            use_git=bool(raw.get("use_git", True)),
            allow_create_new_files=bool(raw.get("allow_create_new_files", True)),
            default_powershell_timeout=int(raw.get("default_powershell_timeout", 120)),
            max_powershell_timeout=int(raw.get("max_powershell_timeout", 900)),
        )


@dataclass(frozen=True, slots=True)
class PendingShellCommand:
    """Команда (PowerShell / Python), ожидающая подтверждения пользователя в чате."""

    id: str
    project_id: str
    script: str
    timeout_used: int
    created_at: float
    steps: tuple[dict[str, str], ...] = ()
    read_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Возвращает JSON-описание отложенной команды для UI чата."""
        steps_list = (
            list(self.steps)
            if self.steps
            else [{"kind": "powershell", "code": self.script}]
        )
        return {
            "id": self.id,
            "project_id": self.project_id,
            "script": self.script,
            "steps": steps_list,
            "read_files": list(self.read_files),
            "timeout_used": self.timeout_used,
        }
