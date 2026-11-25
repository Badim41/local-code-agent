from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path

from .models import DEFAULT_IGNORE_FILES, DEFAULT_IGNORE_FOLDERS


def snapshot_workspace(
    root: Path,
    *,
    ignore_folders: Iterable[str] = (),
    ignore_files: Iterable[str] = (),
) -> dict[str, tuple[int, int]]:
    """Быстрый снимок (rel_path -> (size, mtime_ns)) для детекции изменений на диске."""
    ignored_dirs = frozenset(DEFAULT_IGNORE_FOLDERS) | frozenset(ignore_folders)
    ignored_files_lower = frozenset(
        name.lower() for name in (*DEFAULT_IGNORE_FILES, *ignore_files)
    )
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        return {}

    result: dict[str, tuple[int, int]] = {}
    stack: list[Path] = [resolved_root]

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if name not in ignored_dirs and not name.startswith("."):
                            stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        if name.startswith(".") or name.lower() in ignored_files_lower:
                            continue
                        try:
                            st = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        rel = os.path.relpath(entry.path, resolved_root).replace("\\", "/")
                        result[rel] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue

    return result


def fingerprint_snapshot(snapshot: dict[str, tuple[int, int]]) -> str:
    """Вычисляет детерминированный SHA-256 отпечаток состояния файлов проекта."""
    hasher = hashlib.sha256()
    for rel_path in sorted(snapshot.keys()):
        size, mtime_ns = snapshot[rel_path]
        hasher.update(f"{rel_path}|{size}|{mtime_ns}\n".encode("utf-8"))
    return hasher.hexdigest()


def diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Возвращает (created, modified, deleted) списки относительных путей."""
    created = sorted(p for p in after if p not in before)
    deleted = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and after[p] != before[p])
    return created, modified, deleted
