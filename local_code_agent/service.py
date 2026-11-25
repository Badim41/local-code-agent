from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Protocol

from convert_gpt_answer.replacer import (
    EditsReport,
    ParseError,
    Settings,
    ShellRunner,
    ShellStatus,
    apply_edits,
    list_backups,
    parse_response,
    restore_backup,
    run_shell,
)
from gpt_project_prompter import (
    DEFAULT_MODEL,
    get_gpt_prompt,
    get_project_structure,
)
from network_tools import NetworkToolsAPI

from .config import ServerConfig
from .feedback import (
    build_edits_failed_feedback,
    build_parse_error_feedback,
    build_shell_feedback,
    build_shell_rejected_feedback,
    truncate_log,
)
from .models import PendingShellCommand, ProjectConfig
from .workspace_state import diff_snapshots, fingerprint_snapshot, snapshot_workspace

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_STRUCTURE_PROMPT: str = (
    "Выведи все файлы проекта для агента (чтобы он имел полный доступ ко всему нужному)"
)


class ServiceError(Exception):
    """Доменная ошибка сервиса с HTTP-кодом статуса."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class StructureProviderProtocol(Protocol):
    """Абстракция получения кодовой базы (удобно подменять в тестах без сети)."""

    def fetch_codebase(
        self,
        *,
        project: ProjectConfig,
        structure_prompt: str,
        api_key: str | None,
        model: str,
        mode: str,
        detail_level: int,
        max_structure_chars: int,
        max_prompt_chars: int,
        include_edit_instructions: bool,
    ) -> str: ...


class GptPrompterStructureProvider:
    """Реализация поверх gpt_project_prompter (get_gpt_prompt / get_project_structure)."""

    def fetch_codebase(
        self,
        *,
        project: ProjectConfig,
        structure_prompt: str,
        api_key: str | None,
        model: str,
        mode: str,
        detail_level: int,
        max_structure_chars: int,
        max_prompt_chars: int,
        include_edit_instructions: bool,
    ) -> str:
        """Возвращает полный промпт с кодом (gpt_prompt) или только структуру (structure_only)."""
        if mode == "structure_only":
            structure = get_project_structure(
                path=project.path,
                ignore_folders=project.ignore_folders,
                ignore_file_list=project.ignore_files,
                detail_level=detail_level,
                max_structure_chars=max_structure_chars,
                max_prompt_chars=max_prompt_chars,
                use_git=project.use_git,
            )
            return f"# Структура проекта {project.path.name}\n{structure}"

        if not api_key:
            raise ServiceError(
                "Не передан API-ключ NetworkToolsAPI (передайте api_key, заголовок "
                "X-Network-Tools-Key или задайте LCA_NETWORK_TOOLS_API_KEY).",
                status_code=400,
            )

        network_api = NetworkToolsAPI(api_key)
        return get_gpt_prompt(
            network_tools=network_api,
            path=project.path,
            ignore_folders=project.ignore_folders,
            task=structure_prompt,
            model=model,
            print_file_list=False,
            ignore_file_list=project.ignore_files,
            detail_level=detail_level,
            max_structure_chars=max_structure_chars,
            max_prompt_chars=max_prompt_chars,
            use_git=project.use_git,
            include_edit_instructions=include_edit_instructions,
        )


@dataclass(frozen=True, slots=True)
class _StructureCacheEntry:
    fingerprint: str
    codebase_text: str


class AgentService:
    """Бизнес-логика локального агента: проекты, кэш структуры, правки, PowerShell и бэкапы."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        structure_provider: StructureProviderProtocol | None = None,
        shell_runner: ShellRunner | None = None,
    ) -> None:
        self._config = config
        self._structure_provider: StructureProviderProtocol = (
            structure_provider or GptPrompterStructureProvider()
        )
        self._shell_runner = shell_runner
        self._store_lock = threading.Lock()
        self._project_locks: dict[str, threading.Lock] = {}
        self._structure_cache: dict[tuple[object, ...], _StructureCacheEntry] = {}
        self._pending_shells: dict[str, PendingShellCommand] = {}
        self._projects: dict[str, ProjectConfig] = {}
        self._load_projects()

    def list_projects(self) -> list[dict[str, Any]]:
        """Возвращает отсортированный список зарегистрированных проектов."""
        with self._store_lock:
            ordered = sorted(self._projects.values(), key=lambda p: p.name.lower())
            return [p.to_dict() for p in ordered]

    def upsert_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Добавляет или обновляет проект в projects.json."""
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ServiceError("Поле 'path' обязательно.", status_code=400)
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.is_dir():
            raise ServiceError(f"Директория не существует: {resolved}", status_code=400)

        name = str(payload.get("name", "")).strip() or resolved.name
        pid = str(payload.get("id", "")).strip() or _slugify(name)
        normalized_payload = dict(payload)
        normalized_payload["id"] = pid
        normalized_payload["name"] = name
        normalized_payload["path"] = str(resolved)

        try:
            project = ProjectConfig.from_dict(normalized_payload)
        except (ValueError, TypeError) as exc:
            raise ServiceError(str(exc), status_code=400) from exc

        with self._store_lock:
            self._projects[project.id] = project
            self._save_projects_locked()
            self._invalidate_project_cache_locked(project.id)
        return project.to_dict()

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """Удаляет проект из списка projects.json (файлы на диске не трогаются)."""
        with self._store_lock:
            removed = self._projects.pop(project_id, None)
            if removed is None:
                raise ServiceError(f"Проект '{project_id}' не найден.", status_code=404)
            self._save_projects_locked()
            self._invalidate_project_cache_locked(project_id)
        return {"status": "ok", "deleted_id": project_id}

    def get_structure(
        self,
        *,
        project_id: str,
        structure_prompt: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        mode: str = "gpt_prompt",
        detail_level: int | None = None,
        max_structure_chars: int | None = None,
        max_prompt_chars: int | None = None,
        force_refresh: bool = False,
        include_edit_instructions: bool = False,
    ) -> dict[str, Any]:
        """Возвращает актуальную кодовую базу, переиспользуя кэш если файлы не менялись."""
        project = self._get_project_or_raise(project_id)
        if mode not in ("gpt_prompt", "structure_only"):
            raise ServiceError("Параметр 'mode' должен быть 'gpt_prompt' или 'structure_only'.", 400)

        eff_prompt = (structure_prompt or DEFAULT_STRUCTURE_PROMPT).strip()
        eff_model = (model or project.model or DEFAULT_MODEL).strip()
        eff_detail = project.detail_level if detail_level is None else int(detail_level)
        eff_struct_chars = (
            project.max_structure_chars if max_structure_chars is None else int(max_structure_chars)
        )
        eff_prompt_chars = (
            project.max_prompt_chars if max_prompt_chars is None else int(max_prompt_chars)
        )
        eff_key = api_key or self._config.default_api_key

        lock = self._get_project_lock(project.id)
        with lock:
            snap = snapshot_workspace(
                project.path,
                ignore_folders=project.ignore_folders,
                ignore_files=project.ignore_files,
            )
            fp = fingerprint_snapshot(snap)
            cache_key = (
                project.id,
                mode,
                eff_prompt,
                eff_model,
                eff_detail,
                eff_struct_chars,
                eff_prompt_chars,
                include_edit_instructions,
            )

            cached = self._structure_cache.get(cache_key)
            if not force_refresh and cached is not None and cached.fingerprint == fp:
                return {
                    "status": "ok",
                    "project_id": project.id,
                    "mode": mode,
                    "model": eff_model,
                    "codebase_text": cached.codebase_text,
                    "chars_count": len(cached.codebase_text),
                    "cache_hit": True,
                    "fingerprint": fp,
                }

            codebase = self._structure_provider.fetch_codebase(
                project=project,
                structure_prompt=eff_prompt,
                api_key=eff_key,
                model=eff_model,
                mode=mode,
                detail_level=eff_detail,
                max_structure_chars=eff_struct_chars,
                max_prompt_chars=eff_prompt_chars,
                include_edit_instructions=include_edit_instructions,
            )
            self._structure_cache[cache_key] = _StructureCacheEntry(
                fingerprint=fp,
                codebase_text=codebase,
            )
            return {
                "status": "ok",
                "project_id": project.id,
                "mode": mode,
                "model": eff_model,
                "codebase_text": codebase,
                "chars_count": len(codebase),
                "cache_hit": False,
                "fingerprint": fp,
            }

    def apply_llm_response(
        self,
        *,
        project_id: str,
        llm_text: str,
        auto_powershell: bool = False,
        powershell_timeout: int | None = None,
        max_shell_output_chars: int | None = None,
        refresh_structure_on_change: bool = True,
        structure_prompt: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        mode: str = "gpt_prompt",
    ) -> dict[str, Any]:
        """Применяет правки из ответа LLM и либо запускает PowerShell, либо ждёт подтверждения."""
        project = self._get_project_or_raise(project_id)
        max_out = max_shell_output_chars or self._config.max_shell_output_chars
        settings = _build_replacer_settings(project)

        lock = self._get_project_lock(project.id)
        with lock:
            try:
                parsed = parse_response(llm_text)
            except ParseError as exc:
                return {
                    "status": "patches_failed",
                    "project_id": project.id,
                    "files_changed": False,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": True,
                    "failed_blocks": [],
                    "rolled_back_blocks": [],
                    "errors": [{"block": None, "status": "PARSE_ERROR", "path": None, "message": str(exc)}],
                    "feedback_for_llm": build_parse_error_feedback(str(exc)),
                }

            if not parsed.blocks and parsed.script is None:
                return {
                    "status": "noop",
                    "project_id": project.id,
                    "files_changed": False,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": False,
                    "changes": [],
                }

            edits: EditsReport = apply_edits(parsed, project.path, settings)
            if not edits.success:
                failed_blocks = sorted({e.block for e in edits.errors if e.block is not None})
                rolled_back = [b.index for b in parsed.blocks if b.index not in failed_blocks]
                return {
                    "status": "patches_failed",
                    "project_id": project.id,
                    "files_changed": False,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": True,
                    "edits": edits.to_dict(),
                    "failed_blocks": failed_blocks,
                    "rolled_back_blocks": rolled_back,
                    "feedback_for_llm": build_edits_failed_feedback(
                        edits, failed_blocks, rolled_back
                    ),
                }

            edits_changed = bool(edits.changes)
            if edits_changed:
                self._invalidate_project_cache(project.id)

            # Только правки без PowerShell
            if parsed.script is None:
                res: dict[str, Any] = {
                    "status": "success",
                    "project_id": project.id,
                    "files_changed": edits_changed,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": False,
                    "edits": edits.to_dict(),
                    "changes": [c.to_dict() for c in edits.changes],
                    "backup_id": edits.backup_id,
                }
                if edits_changed and refresh_structure_on_change:
                    self._attach_refreshed_structure(
                        res, project, structure_prompt, api_key, model, mode
                    )
                return res

            eff_timeout = _resolve_effective_timeout(
                powershell_timeout, parsed.timeout_directive, settings
            )

            # PowerShell требует подтверждения пользователя в чате
            if not auto_powershell:
                pending = self._register_pending_shell(project.id, parsed.script, eff_timeout)
                res = {
                    "status": "awaiting_shell_confirmation",
                    "project_id": project.id,
                    "files_changed": edits_changed,
                    "requires_shell_confirmation": True,
                    "needs_llm_followup": False,
                    "edits": edits.to_dict(),
                    "changes": [c.to_dict() for c in edits.changes],
                    "backup_id": edits.backup_id,
                    "pending_shell": pending.to_dict(),
                }
                if edits_changed and refresh_structure_on_change:
                    self._attach_refreshed_structure(
                        res, project, structure_prompt, api_key, model, mode
                    )
                return res

            # Автоматический запуск PowerShell
            shell_report, shell_changed = self._run_shell_with_diff(
                project, parsed.script, eff_timeout
            )
            files_changed = edits_changed or bool(shell_changed)
            if shell_changed:
                self._invalidate_project_cache(project.id)

            trimmed_shell = _trim_shell_dict(shell_report.to_dict(), max_out)
            feedback = build_shell_feedback(
                script=parsed.script,
                shell_report=shell_report,
                edits=edits,
                shell_changed_files=shell_changed,
                max_output_chars=max_out,
            )
            res = {
                "status": "shell_completed",
                "project_id": project.id,
                "files_changed": files_changed,
                "requires_shell_confirmation": False,
                "needs_llm_followup": True,
                "edits": edits.to_dict(),
                "changes": [c.to_dict() for c in edits.changes],
                "backup_id": edits.backup_id,
                "shell": trimmed_shell,
                "shell_changed_files": shell_changed,
                "feedback_for_llm": feedback,
            }
            if files_changed and refresh_structure_on_change:
                self._attach_refreshed_structure(
                    res, project, structure_prompt, api_key, model, mode
                )
            return res

    def execute_pending_shell(
        self,
        *,
        project_id: str,
        command_id: str,
        approved: bool,
        timeout: int | None = None,
        max_shell_output_chars: int | None = None,
        refresh_structure_on_change: bool = True,
        structure_prompt: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        mode: str = "gpt_prompt",
    ) -> dict[str, Any]:
        """Выполняет или отклоняет отложенную PowerShell-команду после клика в чате."""
        project = self._get_project_or_raise(project_id)
        pending = self._pop_pending_shell(command_id, project_id)
        max_out = max_shell_output_chars or self._config.max_shell_output_chars

        if not approved:
            return {
                "status": "shell_rejected",
                "project_id": project.id,
                "command_id": pending.id,
                "files_changed": False,
                "needs_llm_followup": True,
                "feedback_for_llm": build_shell_rejected_feedback(pending.script),
            }

        settings = _build_replacer_settings(project)
        eff_timeout = _resolve_effective_timeout(timeout, pending.timeout_used, settings)

        lock = self._get_project_lock(project.id)
        with lock:
            shell_report, shell_changed = self._run_shell_with_diff(
                project, pending.script, eff_timeout
            )
            files_changed = bool(shell_changed)
            if files_changed:
                self._invalidate_project_cache(project.id)

            res: dict[str, Any] = {
                "status": "shell_completed",
                "project_id": project.id,
                "command_id": pending.id,
                "files_changed": files_changed,
                "needs_llm_followup": True,
                "shell": _trim_shell_dict(shell_report.to_dict(), max_out),
                "shell_changed_files": shell_changed,
                "feedback_for_llm": build_shell_feedback(
                    script=pending.script,
                    shell_report=shell_report,
                    shell_changed_files=shell_changed,
                    max_output_chars=max_out,
                ),
            }
            if files_changed and refresh_structure_on_change:
                self._attach_refreshed_structure(
                    res, project, structure_prompt, api_key, model, mode
                )
            return res

    def list_project_backups(self, project_id: str) -> dict[str, Any]:
        """Возвращает список zip-бэкапов replacer для проекта."""
        project = self._get_project_or_raise(project_id)
        backups = list_backups(project.path)
        return {
            "status": "ok",
            "project_id": project.id,
            "backups": [b.to_dict() for b in backups],
        }

    def undo_project(
        self,
        *,
        project_id: str,
        backup_id: str | None = None,
        refresh_structure: bool = True,
        structure_prompt: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        mode: str = "gpt_prompt",
    ) -> dict[str, Any]:
        """Откатывает последний или указанный бэкап проекта."""
        project = self._get_project_or_raise(project_id)
        lock = self._get_project_lock(project.id)
        with lock:
            report = restore_backup(project.path, backup_id=backup_id)
            if report.success:
                self._invalidate_project_cache(project.id)
            res: dict[str, Any] = {
                "status": "ok" if report.success else "error",
                "project_id": project.id,
                "files_changed": report.success,
                "restore": report.to_dict(),
            }
            if report.success and refresh_structure:
                self._attach_refreshed_structure(
                    res, project, structure_prompt, api_key, model, mode
                )
            return res

    def _run_shell_with_diff(
        self,
        project: ProjectConfig,
        script: str,
        timeout: int,
    ) -> tuple[Any, list[str]]:
        snap_before = snapshot_workspace(
            project.path,
            ignore_folders=project.ignore_folders,
            ignore_files=project.ignore_files,
        )
        shell_report = run_shell(script, project.path, timeout, runner=self._shell_runner)
        snap_after = snapshot_workspace(
            project.path,
            ignore_folders=project.ignore_folders,
            ignore_files=project.ignore_files,
        )
        created, modified, deleted = diff_snapshots(snap_before, snap_after)
        changed = sorted(set(created) | set(modified) | set(deleted))
        return shell_report, changed

    def _attach_refreshed_structure(
        self,
        response: dict[str, Any],
        project: ProjectConfig,
        structure_prompt: str | None,
        api_key: str | None,
        model: str | None,
        mode: str,
    ) -> None:
        eff_key = api_key or self._config.default_api_key
        if mode == "gpt_prompt" and not eff_key:
            return
        try:
            eff_prompt = (structure_prompt or DEFAULT_STRUCTURE_PROMPT).strip()
            eff_model = (model or project.model or DEFAULT_MODEL).strip()
            codebase = self._structure_provider.fetch_codebase(
                project=project,
                structure_prompt=eff_prompt,
                api_key=eff_key,
                model=eff_model,
                mode=mode,
                detail_level=project.detail_level,
                max_structure_chars=project.max_structure_chars,
                max_prompt_chars=project.max_prompt_chars,
                include_edit_instructions=False,
            )
            snap = snapshot_workspace(
                project.path,
                ignore_folders=project.ignore_folders,
                ignore_files=project.ignore_files,
            )
            fp = fingerprint_snapshot(snap)
            cache_key = (
                project.id,
                mode,
                eff_prompt,
                eff_model,
                project.detail_level,
                project.max_structure_chars,
                project.max_prompt_chars,
                False,
            )
            self._structure_cache[cache_key] = _StructureCacheEntry(
                fingerprint=fp,
                codebase_text=codebase,
            )
            response["updated_codebase"] = codebase
            response["updated_codebase_chars"] = len(codebase)
        except Exception as exc:
            logger.warning("Не удалось обновить структуру проекта %s: %s", project.id, exc)
            response["structure_refresh_error"] = str(exc)

    def _register_pending_shell(
        self,
        project_id: str,
        script: str,
        timeout_used: int,
    ) -> PendingShellCommand:
        now = time.monotonic()
        with self._store_lock:
            self._prune_expired_pending_locked(now)
            cmd_id = f"sh_{secrets.token_hex(8)}"
            pending = PendingShellCommand(
                id=cmd_id,
                project_id=project_id,
                script=script,
                timeout_used=timeout_used,
                created_at=now,
            )
            self._pending_shells[cmd_id] = pending
            return pending

    def _pop_pending_shell(self, command_id: str, project_id: str) -> PendingShellCommand:
        now = time.monotonic()
        with self._store_lock:
            self._prune_expired_pending_locked(now)
            pending = self._pending_shells.get(command_id)
            if pending is None or pending.project_id != project_id:
                raise ServiceError(
                    f"Команда '{command_id}' не найдена или истёк срок её ожидания.",
                    status_code=404,
                )
            return self._pending_shells.pop(command_id)

    def _prune_expired_pending_locked(self, now: float) -> None:
        ttl = self._config.pending_shell_ttl_seconds
        expired = [cid for cid, item in self._pending_shells.items() if (now - item.created_at) > ttl]
        for cid in expired:
            self._pending_shells.pop(cid, None)

    def _get_project_or_raise(self, project_id: str) -> ProjectConfig:
        with self._store_lock:
            project = self._projects.get(project_id)
        if project is None:
            raise ServiceError(f"Проект '{project_id}' не найден.", status_code=404)
        if not project.path.is_dir():
            raise ServiceError(f"Папка проекта недоступна: {project.path}", status_code=400)
        return project

    def _get_project_lock(self, project_id: str) -> threading.Lock:
        with self._store_lock:
            lock = self._project_locks.get(project_id)
            if lock is None:
                lock = threading.Lock()
                self._project_locks[project_id] = lock
            return lock

    def _invalidate_project_cache(self, project_id: str) -> None:
        with self._store_lock:
            self._invalidate_project_cache_locked(project_id)

    def _invalidate_project_cache_locked(self, project_id: str) -> None:
        stale = [k for k in self._structure_cache if k and k[0] == project_id]
        for k in stale:
            self._structure_cache.pop(k, None)

    def _load_projects(self) -> None:
        pfile = self._config.projects_file
        if not pfile.is_file():
            return
        try:
            loaded: object = json.loads(pfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Не удалось прочитать %s: %s", pfile, exc)
            return
        if not isinstance(loaded, list):
            return
        for item in loaded:
            if isinstance(item, dict):
                try:
                    proj = ProjectConfig.from_dict(item)
                    self._projects[proj.id] = proj
                except ValueError as exc:
                    logger.warning("Пропущен некорректный проект в %s: %s", pfile, exc)

    def _save_projects_locked(self) -> None:
        pfile = self._config.projects_file
        pfile.parent.mkdir(parents=True, exist_ok=True)
        serialized = [p.to_storage_dict() for p in sorted(self._projects.values(), key=lambda x: x.id)]
        pfile.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_replacer_settings(project: ProjectConfig) -> Settings:
    return Settings(
        allow_create_new_files=project.allow_create_new_files,
        allow_powershell=False,
        default_powershell_timeout=project.default_powershell_timeout,
        max_powershell_timeout=project.max_powershell_timeout,
        ignore_folders=frozenset(project.ignore_folders),
        ignore_files=frozenset(project.ignore_files),
    )


def _resolve_effective_timeout(
    explicit: int | None,
    directive: int | None,
    settings: Settings,
) -> int:
    requested = next(
        v for v in (explicit, directive, settings.default_powershell_timeout) if v is not None
    )
    return max(1, min(int(requested), settings.max_powershell_timeout))


def _trim_shell_dict(data: dict[str, Any], max_chars: int) -> dict[str, Any]:
    out = dict(data)
    if isinstance(out.get("log"), str):
        out["log"] = truncate_log(out["log"], max_chars)
    return out


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or f"proj-{secrets.token_hex(3)}"
