from __future__ import annotations

from collections.abc import Sequence

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


def build_shell_feedback(
    *,
    script: str,
    shell_report: ShellReport,
    edits: EditsReport | None = None,
    shell_changed_files: Sequence[str] = (),
    max_output_chars: int = DEFAULT_MAX_SHELL_OUTPUT_CHARS,
) -> str:
    """Формирует отчёт для LLM с результатом выполнения PowerShell (с сокращением лога)."""
    lines: list[str] = [FEEDBACK_MARKER]

    if edits is not None and edits.changes:
        mod = [c.path for c in edits.changes if c.kind == "modified"]
        crt = [c.path for c in edits.changes if c.kind == "created"]
        lines.append(
            f"[EDITS APPLIED] Изменено файлов: {len(mod)} ({', '.join(mod) or 'нет'}); "
            f"создано: {len(crt)} ({', '.join(crt) or 'нет'})."
        )

    status_map = {
        ShellStatus.OK: "УСПЕХ",
        ShellStatus.FAILED: "ОШИБКА",
        ShellStatus.TIMEOUT: "ТАЙМАУТ",
        ShellStatus.EXECUTABLE_NOT_FOUND: "POWERSHELL НЕ НАЙДЕН",
    }
    label = status_map.get(shell_report.status, str(shell_report.status.value))
    lines.append(
        f"[POWERSHELL {label}] exit_code={shell_report.exit_code}, "
        f"duration={shell_report.duration_seconds:.2f}s"
    )
    if shell_changed_files:
        lines.append("Файлы, изменённые командой: " + ", ".join(shell_changed_files))

    cmd_preview = truncate_log(script.strip(), 1500)
    fence = "`" * 3
    lines.append(f"\nКоманда:\n{fence}powershell\n{cmd_preview}\n{fence}")

    log_body = shell_report.log.strip()
    if log_body:
        trimmed = truncate_log(log_body, max_output_chars)
        lines.append(f"\nВывод:\n{fence}\n{trimmed}\n{fence}")
    else:
        lines.append("\nВывод: (пусто)")

    return "\n".join(lines)


def build_shell_rejected_feedback(script: str) -> str:
    """Формирует уведомление для LLM, если пользователь отклонил запуск PowerShell в чате."""
    cmd_preview = truncate_log(script.strip(), 1500)
    fence = "`" * 3
    return (
        f"{FEEDBACK_MARKER}\n"
        f"[POWERSHELL REJECTED] Пользователь отклонил выполнение следующей команды:\n"
        f"{fence}powershell\n{cmd_preview}\n{fence}\n\n"
        f"Не повторяй эту команду без явной просьбы пользователя и скорректируй дальнейшие действия."
    )
