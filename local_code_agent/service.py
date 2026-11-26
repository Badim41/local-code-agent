from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
import io
import json
import logging
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Literal, Protocol

from convert_gpt_answer.replacer import (
    EditsReport,
    ExecutionStep,
    ParseError,
    Settings,
    ShellRunner,
    apply_edits,
    list_backups,
    parse_response,
    restore_backup,
    run_execution_steps,
)
from gpt_project_prompter import (
    DEFAULT_MODEL,
    get_gpt_prompt,
    get_project_structure,
)
from gpt_project_prompter_core.constants import MAX_TEXT_READ_BYTES
from gpt_project_prompter_core.media_probe import probe_image_dimensions
from gpt_project_prompter_core.reader import SafeFileReader
from network_tools import NetworkToolsAPI

from .config import ServerConfig
from .feedback import (
    build_edits_failed_feedback,
    build_parse_error_feedback,
    build_read_files_feedback,
    build_shell_feedback,
    build_shell_rejected_feedback,
    truncate_log,
)
from .models import (
    MAX_IMAGE_FILE_BYTES,
    MAX_READ_FILES_TOTAL,
    MAX_READ_IMAGES_TOTAL,
    PendingShellCommand,
    ProjectConfig,
)
from .workspace_state import diff_snapshots, fingerprint_snapshot, snapshot_workspace

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_STRUCTURE_PROMPT: str = (
    "Выведи все файлы проекта для агента (чтобы он имел полный доступ ко всему нужному)"
)
_DIRECT_IMAGE_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_CONVERTIBLE_IMAGE_EXTS: frozenset[str] = frozenset({".bmp", ".gif"})
_ALWAYS_FORBIDDEN_DIRS: frozenset[str] = frozenset(
    {".git", ".get_gpt_project", ".replacer_backups"}
)
_ALWAYS_FORBIDDEN_FILES: frozenset[str] = frozenset({".env"})


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
        extra_files: Sequence[str] = (),
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
        extra_files: Sequence[str] = (),
    ) -> str:
        """Возвращает полный промпт с кодом (gpt_prompt) или структуру + extra_files."""
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
            base = f"# Структура проекта {project.path.name}\n{structure}"
            if extra_files:
                files_block = SafeFileReader().read_files_content(
                    root_path=project.path,
                    relative_paths=extra_files,
                )
                if files_block:
                    combined = f"{base}\n\n# Файлы\n{files_block}"
                    return combined[:max_prompt_chars]
            return base

        if not api_key:
            raise ServiceError(
                "Не передан API-ключ NetworkToolsAPI (передайте api_key, заголовок "
                "X-Network-Tools-Key или задайте LCA_NETWORK_TOOLS_API_KEY).",
                status_code=400,
            )

        network_api = NetworkToolsAPI(api_key)
        kwargs: dict[str, Any] = {
            "network_tools": network_api,
            "path": project.path,
            "ignore_folders": project.ignore_folders,
            "task": structure_prompt,
            "model": model,
            "print_file_list": False,
            "ignore_file_list": project.ignore_files,
            "detail_level": detail_level,
            "max_structure_chars": max_structure_chars,
            "max_prompt_chars": max_prompt_chars,
            "use_git": project.use_git,
            "include_edit_instructions": include_edit_instructions,
        }
        if extra_files:
            kwargs["extra_files"] = tuple(extra_files)
        return get_gpt_prompt(**kwargs)


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
        python_runner: ShellRunner | None = None,
    ) -> None:
        self._config = config
        self._structure_provider: StructureProviderProtocol = (
            structure_provider or GptPrompterStructureProvider()
        )
        self._shell_runner = shell_runner
        self._python_runner = python_runner
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
        extra_files: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Возвращает актуальную кодовую базу и прикреплённые фото, переиспользуя кэш."""
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
            text_extra, attached_images, _ = _partition_extra_files(
                project, extra_files or (), ignore_missing=True
            )
            pinned_all = list(dict.fromkeys([*text_extra, *(img["path"] for img in attached_images)]))

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
                tuple(text_extra),
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
                    "pinned_files": pinned_all,
                    "attached_images": attached_images,
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
                extra_files=tuple(text_extra),
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
                "pinned_files": pinned_all,
                "attached_images": attached_images,
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
        extra_files: Sequence[str] | None = None,
        inspect_only: bool = False,
    ) -> dict[str, Any]:
        """Применяет правки, READ_FILES и PowerShell/Python (или делает сухую проверку)."""
        project = self._get_project_or_raise(project_id)
        max_out = max_shell_output_chars or self._config.max_shell_output_chars
        settings = _build_replacer_settings(project)
        existing_extra = list(extra_files or ())

        lock = self._get_project_lock(project.id)
        with lock:
            try:
                parsed = parse_response(llm_text)
            except ParseError as exc:
                if inspect_only:
                    return {
                        "status": "actions_pending",
                        "project_id": project.id,
                        "has_actions": True,
                        "pending_actions": {
                            "edits_count": 0,
                            "read_files": [],
                            "has_powershell": False,
                            "has_python": False,
                            "steps_count": 0,
                            "parse_error": str(exc),
                        },
                    }
                return {
                    "status": "patches_failed",
                    "project_id": project.id,
                    "files_changed": False,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": True,
                    "failed_blocks": [],
                    "rolled_back_blocks": [],
                    "errors": [
                        {"block": None, "status": "PARSE_ERROR", "path": None, "message": str(exc)}
                    ],
                    "feedback_for_llm": build_parse_error_feedback(str(exc)),
                }

            steps = parsed.effective_execution_steps()
            has_blocks = bool(parsed.blocks)
            has_steps = bool(steps)
            has_reads = bool(parsed.read_files)

            if not has_blocks and not has_steps and not has_reads:
                return {
                    "status": "noop",
                    "project_id": project.id,
                    "has_actions": False,
                    "files_changed": False,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": False,
                    "changes": [],
                }

            if inspect_only:
                return {
                    "status": "actions_pending",
                    "project_id": project.id,
                    "has_actions": True,
                    "pending_actions": {
                        "edits_count": len(parsed.blocks),
                        "read_files": list(parsed.read_files),
                        "has_powershell": any(s.kind == "powershell" for s in steps),
                        "has_python": any(s.kind == "python" for s in steps),
                        "steps_count": len(steps),
                        "parse_error": None,
                    },
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

            # Случай без скриптов (только правки и/или READ_FILES)
            if not has_steps:
                if not has_reads:
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
                        text_pinned, attached_images, _ = _partition_extra_files(
                            project, existing_extra, ignore_missing=True
                        )
                        self._attach_refreshed_structure(
                            res,
                            project,
                            structure_prompt,
                            api_key,
                            model,
                            mode,
                            extra_files=text_pinned,
                        )
                        if attached_images:
                            res["attached_images"] = attached_images
                    return res

                merged_pinned, text_pinned, attached_images, read_report = _merge_read_files(
                    project=project,
                    existing_extra=existing_extra,
                    requested_reads=parsed.read_files,
                )
                feedback = build_read_files_feedback(
                    read_files_report=read_report,
                    edits=edits if edits_changed else None,
                )
                res = {
                    "status": "files_read",
                    "project_id": project.id,
                    "files_changed": edits_changed,
                    "requires_shell_confirmation": False,
                    "needs_llm_followup": True,
                    "edits": edits.to_dict(),
                    "changes": [c.to_dict() for c in edits.changes],
                    "backup_id": edits.backup_id,
                    "read_files_report": read_report,
                    "pinned_files": merged_pinned,
                    "attached_images": attached_images,
                    "feedback_for_llm": feedback,
                }
                if refresh_structure_on_change and (edits_changed or bool(read_report["added_text"])):
                    self._attach_refreshed_structure(
                        res,
                        project,
                        structure_prompt,
                        api_key,
                        model,
                        mode,
                        extra_files=text_pinned,
                    )
                return res

            eff_timeout = _resolve_effective_timeout(
                powershell_timeout, parsed.timeout_directive, settings
            )
            steps_dicts = tuple(s.to_dict() for s in steps)
            combined_preview = _format_steps_preview(steps)

            # Скрипт требует подтверждения пользователя в чате
            if not auto_powershell:
                pending = self._register_pending_shell(
                    project.id,
                    combined_preview,
                    eff_timeout,
                    steps=steps_dicts,
                    read_files=parsed.read_files,
                )
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
                    text_pinned, attached_images, _ = _partition_extra_files(
                        project, existing_extra, ignore_missing=True
                    )
                    self._attach_refreshed_structure(
                        res,
                        project,
                        structure_prompt,
                        api_key,
                        model,
                        mode,
                        extra_files=text_pinned,
                    )
                    if attached_images:
                        res["attached_images"] = attached_images
                return res

            # Автоматический запуск скриптов (PowerShell и/или временный Python)
            shell_report, shell_changed = self._run_steps_with_diff(project, steps, eff_timeout)
            files_changed = edits_changed or bool(shell_changed)
            if shell_changed:
                self._invalidate_project_cache(project.id)

            merged_pinned, text_pinned, attached_images, read_report = _merge_read_files(
                project=project,
                existing_extra=existing_extra,
                requested_reads=parsed.read_files,
            )

            trimmed_shell = _trim_shell_dict(shell_report.to_dict(), max_out)
            feedback = build_shell_feedback(
                script=combined_preview,
                shell_report=shell_report,
                edits=edits,
                shell_changed_files=shell_changed,
                max_output_chars=max_out,
                steps=steps_dicts,
                read_files_report=read_report if has_reads else None,
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
                "steps": list(steps_dicts),
                "feedback_for_llm": feedback,
            }
            if has_reads or merged_pinned:
                res["pinned_files"] = merged_pinned
                res["attached_images"] = attached_images
            if has_reads:
                res["read_files_report"] = read_report

            if refresh_structure_on_change and (
                files_changed or (has_reads and bool(read_report["added_text"]))
            ):
                self._attach_refreshed_structure(
                    res,
                    project,
                    structure_prompt,
                    api_key,
                    model,
                    mode,
                    extra_files=text_pinned,
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
        extra_files: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Выполняет или отклоняет отложенную команду (PowerShell/Python) после клика в чате."""
        project = self._get_project_or_raise(project_id)
        pending = self._pop_pending_shell(command_id, project_id)
        max_out = max_shell_output_chars or self._config.max_shell_output_chars
        existing_extra = list(extra_files or ())
        steps_objs = _steps_from_pending(pending)
        steps_dicts = tuple(s.to_dict() for s in steps_objs)
        has_reads = bool(pending.read_files)

        lock = self._get_project_lock(project.id)
        with lock:
            if not approved:
                merged_pinned, text_pinned, attached_images, read_report = _merge_read_files(
                    project=project,
                    existing_extra=existing_extra,
                    requested_reads=pending.read_files,
                )
                res: dict[str, Any] = {
                    "status": "shell_rejected",
                    "project_id": project.id,
                    "command_id": pending.id,
                    "files_changed": False,
                    "needs_llm_followup": True,
                    "steps": list(steps_dicts),
                    "feedback_for_llm": build_shell_rejected_feedback(
                        pending.script,
                        steps=steps_dicts,
                        read_files_report=read_report if has_reads else None,
                    ),
                }
                if has_reads or merged_pinned:
                    res["pinned_files"] = merged_pinned
                    res["attached_images"] = attached_images
                if has_reads:
                    res["read_files_report"] = read_report
                    if refresh_structure_on_change and bool(read_report["added_text"]):
                        self._attach_refreshed_structure(
                            res,
                            project,
                            structure_prompt,
                            api_key,
                            model,
                            mode,
                            extra_files=text_pinned,
                        )
                return res

            settings = _build_replacer_settings(project)
            eff_timeout = _resolve_effective_timeout(timeout, pending.timeout_used, settings)

            shell_report, shell_changed = self._run_steps_with_diff(
                project, steps_objs, eff_timeout
            )
            files_changed = bool(shell_changed)
            if files_changed:
                self._invalidate_project_cache(project.id)

            merged_pinned, text_pinned, attached_images, read_report = _merge_read_files(
                project=project,
                existing_extra=existing_extra,
                requested_reads=pending.read_files,
            )

            res = {
                "status": "shell_completed",
                "project_id": project.id,
                "command_id": pending.id,
                "files_changed": files_changed,
                "needs_llm_followup": True,
                "shell": _trim_shell_dict(shell_report.to_dict(), max_out),
                "shell_changed_files": shell_changed,
                "steps": list(steps_dicts),
                "feedback_for_llm": build_shell_feedback(
                    script=pending.script,
                    shell_report=shell_report,
                    shell_changed_files=shell_changed,
                    max_output_chars=max_out,
                    steps=steps_dicts,
                    read_files_report=read_report if has_reads else None,
                ),
            }
            if has_reads or merged_pinned:
                res["pinned_files"] = merged_pinned
                res["attached_images"] = attached_images
            if has_reads:
                res["read_files_report"] = read_report

            if refresh_structure_on_change and (
                files_changed or (has_reads and bool(read_report["added_text"]))
            ):
                self._attach_refreshed_structure(
                    res,
                    project,
                    structure_prompt,
                    api_key,
                    model,
                    mode,
                    extra_files=text_pinned,
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
        extra_files: Sequence[str] | None = None,
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
                text_pinned, attached_images, _ = _partition_extra_files(
                    project, extra_files or (), ignore_missing=True
                )
                self._attach_refreshed_structure(
                    res,
                    project,
                    structure_prompt,
                    api_key,
                    model,
                    mode,
                    extra_files=text_pinned,
                )
                if attached_images:
                    res["attached_images"] = attached_images
            return res

    def _run_steps_with_diff(
        self,
        project: ProjectConfig,
        steps: Sequence[ExecutionStep],
        timeout: int,
    ) -> tuple[Any, list[str]]:
        snap_before = snapshot_workspace(
            project.path,
            ignore_folders=project.ignore_folders,
            ignore_files=project.ignore_files,
        )
        shell_report = run_execution_steps(
            steps,
            project.path,
            timeout,
            shell_runner=self._shell_runner,
            python_runner=self._python_runner,
        )
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
        *,
        extra_files: Sequence[str] = (),
    ) -> None:
        eff_key = api_key or self._config.default_api_key
        if mode == "gpt_prompt" and not eff_key:
            return
        try:
            eff_prompt = (structure_prompt or DEFAULT_STRUCTURE_PROMPT).strip()
            eff_model = (model or project.model or DEFAULT_MODEL).strip()
            extra_tuple = tuple(extra_files)
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
                extra_files=extra_tuple,
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
                extra_tuple,
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
        *,
        steps: tuple[dict[str, str], ...] = (),
        read_files: tuple[str, ...] = (),
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
                steps=steps,
                read_files=read_files,
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


def _format_steps_preview(steps: Sequence[ExecutionStep]) -> str:
    if len(steps) == 1:
        return steps[0].code
    parts: list[str] = []
    for idx, step in enumerate(steps, start=1):
        label = "Python (временный .py)" if step.kind == "python" else "PowerShell"
        parts.append(f"# === Шаг {idx}: {label} ===\n{step.code.strip()}")
    return "\n\n".join(parts)


def _steps_from_pending(pending: PendingShellCommand) -> tuple[ExecutionStep, ...]:
    if pending.steps:
        parsed_steps: list[ExecutionStep] = []
        for item in pending.steps:
            raw_kind = str(item.get("kind", "powershell"))
            kind: Literal["powershell", "python"] = "python" if raw_kind == "python" else "powershell"
            code = str(item.get("code", ""))
            if code.strip():
                parsed_steps.append(ExecutionStep(kind=kind, code=code))
        if parsed_steps:
            return tuple(parsed_steps)
    return (ExecutionStep(kind="powershell", code=pending.script),)


def _validate_project_file_path(
    raw_path: str,
    project: ProjectConfig,
) -> tuple[str | None, Path | None, str | None]:
    cleaned = raw_path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or "\x00" in cleaned or ":" in cleaned or cleaned.startswith("/"):
        return None, None, "небезопасный или некорректный путь"

    raw_parts = [p for p in cleaned.split("/") if p and p != "."]
    if not raw_parts or any(p == ".." for p in raw_parts):
        return None, None, "выход за пределы корня проекта запрещён"

    blocked_dirs = _ALWAYS_FORBIDDEN_DIRS | frozenset(project.ignore_folders)
    blocked_files_lower = {
        name.lower() for name in (*_ALWAYS_FORBIDDEN_FILES, *project.ignore_files)
    }
    if any(seg in blocked_dirs for seg in raw_parts[:-1]) or raw_parts[-1].lower() in blocked_files_lower:
        return None, None, "доступ к файлу запрещён настройками безопасности / ignore"

    resolved_root = project.path.resolve()
    try:
        target = (resolved_root / Path(*raw_parts)).resolve()
    except OSError as exc:
        return None, None, f"ошибка разрешения пути: {exc}"

    if not target.is_relative_to(resolved_root):
        return None, None, "выход за пределы корня проекта (включая симлинки)"
    if not target.is_file():
        return None, None, "файл не найден"

    canonical = target.relative_to(resolved_root).as_posix()
    c_parts = canonical.split("/")
    if any(seg in blocked_dirs for seg in c_parts[:-1]) or c_parts[-1].lower() in blocked_files_lower:
        return None, None, "доступ к файлу запрещён настройками безопасности / ignore"

    return canonical, target, None


def _read_image_attachment(
    rel_path: str,
    target: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    ext = target.suffix.lower()
    try:
        size_bytes = target.stat().st_size
    except OSError as exc:
        return None, f"ошибка чтения метаданных: {exc}"

    if size_bytes > MAX_IMAGE_FILE_BYTES:
        return (
            None,
            f"изображение превышает лимит ({size_bytes} > {MAX_IMAGE_FILE_BYTES} байт)",
        )

    if ext in _DIRECT_IMAGE_MIMES:
        try:
            raw = target.read_bytes()
        except OSError as exc:
            return None, f"ошибка чтения изображения: {exc}"
        dims = probe_image_dimensions(target)
        return (
            {
                "path": rel_path,
                "mime_type": _DIRECT_IMAGE_MIMES[ext],
                "base64": base64.b64encode(raw).decode("ascii"),
                "size_bytes": len(raw),
                "width": dims[0] if dims else None,
                "height": dims[1] if dims else None,
            },
            None,
        )

    if ext in _CONVERTIBLE_IMAGE_EXTS:
        try:
            from PIL import Image
        except ImportError:
            return (
                None,
                f"формат {ext} не поддерживается без Pillow (используйте .png, .jpg, .jpeg, .webp)",
            )
        try:
            with Image.open(target) as img:
                img.seek(0)
                converted = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                buf = io.BytesIO()
                converted.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                width, height = converted.size
        except Exception as exc:
            return None, f"ошибка конвертации {ext} в PNG: {exc}"
        if len(png_bytes) > MAX_IMAGE_FILE_BYTES:
            return None, "сконвертированное PNG-изображение превышает лимит размера"
        return (
            {
                "path": rel_path,
                "mime_type": "image/png",
                "base64": base64.b64encode(png_bytes).decode("ascii"),
                "size_bytes": len(png_bytes),
                "width": width,
                "height": height,
            },
            None,
        )

    return None, f"неподдерживаемый формат изображения: {ext}"


def _classify_and_read_file(
    rel_path: str,
    target: Path,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Возвращает ('text'|'image'|'error', image_dict_or_none, error_msg_or_none)."""
    ext = target.suffix.lower()
    if ext in _DIRECT_IMAGE_MIMES or ext in _CONVERTIBLE_IMAGE_EXTS:
        img_obj, err = _read_image_attachment(rel_path, target)
        if err is not None:
            return "error", None, err
        return "image", img_obj, None

    try:
        size_bytes = target.stat().st_size
    except OSError as exc:
        return "error", None, f"ошибка чтения: {exc}"

    if size_bytes > MAX_TEXT_READ_BYTES:
        return (
            "error",
            None,
            f"файл слишком большой ({size_bytes} > {MAX_TEXT_READ_BYTES} байт); "
            "проинспектируй нужный фрагмент через python_run",
        )

    try:
        with target.open("rb") as fp:
            head = fp.read(1024)
            if b"\x00" in head:
                return "error", None, "бинарный файл (не текст и не поддерживаемое изображение)"
            raw = head + fp.read()
        raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "error", None, "ошибка декодирования UTF-8 (файл не в текстовом формате)"
    except OSError as exc:
        return "error", None, f"ошибка чтения: {exc}"

    return "text", None, None


def _partition_extra_files(
    project: ProjectConfig,
    raw_paths: Sequence[str],
    *,
    ignore_missing: bool = False,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, str]]]:
    """Разделяет пути на валидные текстовые файлы, изображения и ошибки."""
    text_paths: list[str] = []
    images: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_text: set[str] = set()
    seen_img: set[str] = set()

    for raw in raw_paths:
        canonical, target, val_err = _validate_project_file_path(raw, project)
        if val_err is not None or canonical is None or target is None:
            if not (ignore_missing and val_err == "файл не найден"):
                errors.append({"path": raw, "error": val_err or "некорректный путь"})
            continue

        if canonical in seen_text or canonical in seen_img:
            continue

        kind, img_obj, read_err = _classify_and_read_file(canonical, target)
        if kind == "text":
            if len(text_paths) >= MAX_READ_FILES_TOTAL:
                errors.append(
                    {
                        "path": canonical,
                        "error": f"достигнут лимит закреплённых текстовых файлов ({MAX_READ_FILES_TOTAL})",
                    }
                )
                continue
            seen_text.add(canonical)
            text_paths.append(canonical)
        elif kind == "image" and img_obj is not None:
            if len(images) >= MAX_READ_IMAGES_TOTAL:
                errors.append(
                    {
                        "path": canonical,
                        "error": f"достигнут лимит прикреплённых изображений ({MAX_READ_IMAGES_TOTAL})",
                    }
                )
                continue
            seen_img.add(canonical)
            images.append(img_obj)
        else:
            errors.append({"path": canonical, "error": read_err or "не удалось прочитать файл"})

    return text_paths, images, errors


def _merge_read_files(
    *,
    project: ProjectConfig,
    existing_extra: Sequence[str],
    requested_reads: Sequence[str],
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    """Объединяет ранее закреплённые файлы хода с новыми READ_FILES и строит отчёт."""
    prev_text, prev_images, _ = _partition_extra_files(
        project, existing_extra, ignore_missing=True
    )
    prev_text_set = set(prev_text)
    prev_img_by_path = {img["path"]: img for img in prev_images}

    new_text: list[str] = []
    new_images_meta: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    text_out = list(prev_text)
    images_out = list(prev_images)
    seen_text = set(prev_text_set)
    seen_img = set(prev_img_by_path.keys())

    for raw in requested_reads:
        canonical, target, val_err = _validate_project_file_path(raw, project)
        if val_err is not None or canonical is None or target is None:
            errors.append({"path": raw, "error": val_err or "некорректный путь"})
            continue

        kind, img_obj, read_err = _classify_and_read_file(canonical, target)
        if kind == "text":
            if canonical not in seen_text:
                if len(text_out) >= MAX_READ_FILES_TOTAL:
                    errors.append(
                        {
                            "path": canonical,
                            "error": f"достигнут лимит закреплённых файлов ({MAX_READ_FILES_TOTAL})",
                        }
                    )
                    continue
                seen_text.add(canonical)
                text_out.append(canonical)
            if canonical not in new_text:
                new_text.append(canonical)
        elif kind == "image" and img_obj is not None:
            if canonical not in seen_img:
                if len(images_out) >= MAX_READ_IMAGES_TOTAL:
                    errors.append(
                        {
                            "path": canonical,
                            "error": f"достигнут лимит прикреплённых изображений ({MAX_READ_IMAGES_TOTAL})",
                        }
                    )
                    continue
                seen_img.add(canonical)
                images_out.append(img_obj)
            else:
                images_out = [img_obj if i["path"] == canonical else i for i in images_out]
            if not any(item["path"] == canonical for item in new_images_meta):
                new_images_meta.append(
                    {
                        "path": canonical,
                        "width": img_obj.get("width"),
                        "height": img_obj.get("height"),
                        "mime_type": img_obj.get("mime_type"),
                    }
                )
        else:
            errors.append({"path": canonical, "error": read_err or "не удалось прочитать файл"})

    merged_pinned = list(dict.fromkeys([*text_out, *(img["path"] for img in images_out)]))
    report = {
        "added_text": new_text,
        "added_images": new_images_meta,
        "errors": errors,
    }
    return merged_pinned, text_out, images_out, report
