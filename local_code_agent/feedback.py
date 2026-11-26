from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from convert_gpt_answer.replacer import (
    EditsReport,
    ShellReport,
    ShellStatus,
)

from .config import DEFAULT_MAX_SHELL_OUTPUT_CHARS

FEEDBACK_MARKER: str = "[LOCAL_CODE_AGENT_FEEDBACK]"


def truncate_log(text: str, max_chars: int = DEFAULT_MAX_SHELL_OUTPUT_CHARS) -> str:
    """Сокращает длинный вывод PowerShell, сохраняя 30% начала и 70% хвоста (стектрейс/ошибки)."""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 120:
        return text[:max_chars] + "..."
    head_len = max(40, int(max_chars * 0.30))
    tail_len = max(40, max_chars - head_len)
    omitted = len(text) - head_len - tail_len
    return (
        f"{text[:head_len]}"
        f"\n\n... [вывод PowerShell сокращён: пропущено {omitted} симв. из {len(text)}] ...\n\n"
        f"{text[-tail_len:]}"
    )


def build_parse_error_feedback(error_message: str) -> str:
    """Формирует сообщение для LLM при синтаксической ошибке формата блоков правок."""
    return (
        f"{FEEDBACK_MARKER}\n"
        f"[PARSE_ERROR] Ответ не удалось разобрать, правки НЕ применены:\n"
        f"{error_message}\n\n"
        f"Исправь формат блоков правок и повтори ответ."
    )


def build_edits_failed_feedback(
    edits: EditsReport,
    failed_blocks: Sequence[int],
    rolled_back_blocks: Sequence[int],
) -> str:
    """Формирует детальный отчёт для LLM о непринятых и отменённых блоках правок."""
    lines: list[str] = [
        FEEDBACK_MARKER,
        "[TRANSACTION ABORTED] Правки НЕ применены (транзакция отменена полностью):",
    ]
    if failed_blocks:
        lines.append(
            "Не принятые (ошибочные) блоки: " + ", ".join(f"#{n}" for n in failed_blocks)
        )
    if rolled_back_blocks:
        lines.append(
            "Отменённые из-за атомарного отката блоки (НЕ применены, нужно повторить): "
            + ", ".join(f"#{n}" for n in rolled_back_blocks)
        )

    lines.append("")
    for err in edits.errors:
        lines.append(f"- [{err.status.value}] {err.message}")

    if rolled_back_blocks:
        lines.append(
            "\nВАЖНО: На диск ничего не записано. Исправь ошибки и пришли полный набор правок заново."
        )
    else:
        lines.append("\nИсправь указанные ошибки с точным совпадением строк и повтори правки.")
    return "\n".join(lines)


def _append_edits_summary(lines: list[str], edits: EditsReport | None) -> None:
    if edits is not None and edits.changes:
        mod = [c.path for c in edits.changes if c.kind == "modified"]
        crt = [c.path for c in edits.changes if c.kind == "created"]
        lines.append(
            f"[EDITS APPLIED] Изменено файлов: {len(mod)} ({', '.join(mod) or 'нет'}); "
            f"создано: {len(crt)} ({', '.join(crt) or 'нет'})."
        )


def _append_read_files_section(
    lines: list[str],
    read_files_report: dict[str, Any] | None,
) -> None:
    if not read_files_report:
        return
    added_text = read_files_report.get("added_text") or []
    added_images = read_files_report.get("added_images") or []
    errors = read_files_report.get("errors") or []
    if not added_text and not added_images and not errors:
        return

    lines.append("[READ_FILES] Результат чтения запрошенных файлов:")
    if added_text:
        lines.append(
            '- Текстовые файлы (доступны полностью в разделе "# Файлы" на весь текущий ход агента): '
            + ", ".join(str(p) for p in added_text)
        )
    if added_images:
        img_parts: list[str] = []
        for item in added_images:
            if isinstance(item, dict):
                path = str(item.get("path", ""))
                w, h = item.get("width"), item.get("height")
                img_parts.append(f"{path} ({w}x{h})" if w and h else path)
            else:
                img_parts.append(str(item))
        lines.append(
            "- Изображения (прикреплены к запросу на весь текущий ход агента): "
            + ", ".join(img_parts)
        )
    if errors:
        lines.append("- Не удалось прочитать:")
        for err in errors:
            if isinstance(err, dict):
                lines.append(f"  - {err.get('path', '?')}: {err.get('error', 'ошибка')}")
            else:
                lines.append(f"  - {err}")


def _runner_tag(steps: Sequence[dict[str, str]]) -> tuple[str, str]:
    kinds = {str(s.get("kind", "powershell")) for s in steps if isinstance(s, dict)}
    if kinds == {"python"}:
        return "PYTHON", "python"
    if len(kinds) > 1:
        return "SCRIPT", ""
    return "POWERSHELL", "powershell"


def build_read_files_feedback(
    *,
    read_files_report: dict[str, Any],
    edits: EditsReport | None = None,
) -> str:
    """Формирует компактный отчёт для LLM о прочитанных через READ_FILES файлах и фото."""
    lines: list[str] = [FEEDBACK_MARKER]
    _append_edits_summary(lines, edits)
    _append_read_files_section(lines, read_files_report)
    lines.append(
        "\nЗапрошенные файлы уже подгружены в контекст этого запроса. Продолжай выполнение задачи."
    )
    return "\n".join(lines)


def build_shell_feedback(
    *,
    script: str,
    shell_report: ShellReport,
    edits: EditsReport | None = None,
    shell_changed_files: Sequence[str] = (),
    max_output_chars: int = DEFAULT_MAX_SHELL_OUTPUT_CHARS,
    steps: Sequence[dict[str, str]] = (),
    read_files_report: dict[str, Any] | None = None,
) -> str:
    """Формирует отчёт для LLM с результатом выполнения PowerShell/Python (с сокращением лога)."""
    lines: list[str] = [FEEDBACK_MARKER]
    _append_edits_summary(lines, edits)

    runner_label, fence_lang = _runner_tag(steps)
    status_map = {
        ShellStatus.OK: "УСПЕХ",
        ShellStatus.FAILED: "ОШИБКА",
        ShellStatus.TIMEOUT: "ТАЙМАУТ",
        ShellStatus.EXECUTABLE_NOT_FOUND: f"{runner_label} НЕ НАЙДЕН",
    }
    label = status_map.get(shell_report.status, str(shell_report.status.value))
    lines.append(
        f"[{runner_label} {label}] exit_code={shell_report.exit_code}, "
        f"duration={shell_report.duration_seconds:.2f}s"
    )
    if shell_changed_files:
        lines.append("Файлы, изменённые командой: " + ", ".join(shell_changed_files))

    _append_read_files_section(lines, read_files_report)

    cmd_preview = truncate_log(script.strip(), 1500)
    fence = "`" * 3
    lines.append(f"\nКоманда:\n{fence}{fence_lang}\n{cmd_preview}\n{fence}")

    log_body = shell_report.log.strip()
    if log_body:
        trimmed = truncate_log(log_body, max_output_chars)
        lines.append(f"\nВывод:\n{fence}\n{trimmed}\n{fence}")
    else:
        lines.append("\nВывод: (пусто)")

    return "\n".join(lines)


def build_shell_rejected_feedback(
    script: str,
    *,
    steps: Sequence[dict[str, str]] = (),
    read_files_report: dict[str, Any] | None = None,
) -> str:
    """Формирует уведомление для LLM, если пользователь отклонил запуск скрипта в чате."""
    runner_label, fence_lang = _runner_tag(steps)
    cmd_preview = truncate_log(script.strip(), 1500)
    fence = "`" * 3
    lines: list[str] = [
        FEEDBACK_MARKER,
        f"[{runner_label} REJECTED] Пользователь отклонил выполнение следующей команды:",
        f"{fence}{fence_lang}\n{cmd_preview}\n{fence}",
    ]
    _append_read_files_section(lines, read_files_report)
    lines.append(
        "\nНе повторяй эту команду без явной просьбы пользователя и скорректируй дальнейшие действия."
    )
    return "\n".join(lines)
