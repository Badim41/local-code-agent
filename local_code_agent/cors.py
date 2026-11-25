from __future__ import annotations

from collections.abc import Sequence

from .config import normalize_origin


class CorsPolicy:
    """Проверяет CORS Origin и формирует заголовки (включая Private Network Access)."""

    ALLOWED_METHODS: str = "GET, POST, DELETE, OPTIONS"
    ALLOWED_HEADERS: str = "Content-Type, X-Agent-Token, X-Network-Tools-Key"

    def __init__(self, allowed_origins: Sequence[str]) -> None:
        self._allowed: frozenset[str] = frozenset(
            normalize_origin(o) for o in allowed_origins if o and o.strip()
        )

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Возвращает отсортированный кортеж разрешённых доменов."""
        return tuple(sorted(self._allowed))

    def is_origin_allowed(self, origin: str | None) -> bool:
        """Разрешает запросы без Origin (curl/сервер) или со строго доверенным Origin."""
        if not origin:
            return True
        return normalize_origin(origin) in self._allowed

    def build_headers(
        self,
        origin: str | None,
        *,
        is_preflight: bool = False,
        private_network_requested: bool = False,
    ) -> dict[str, str]:
        """Формирует заголовки CORS для ответа при разрешённом Origin."""
        if not origin:
            return {}
        norm = normalize_origin(origin)
        if norm not in self._allowed:
            return {}

        headers: dict[str, str] = {
            "Access-Control-Allow-Origin": norm,
            "Vary": "Origin",
        }
        if is_preflight:
            headers["Access-Control-Allow-Methods"] = self.ALLOWED_METHODS
            headers["Access-Control-Allow-Headers"] = self.ALLOWED_HEADERS
            headers["Access-Control-Max-Age"] = "600"
            # Обязательно для запросов с https:// на http://127.0.0.1 в Chromium (PNA/LNA)
            headers["Access-Control-Allow-Private-Network"] = "true"
        elif private_network_requested:
            headers["Access-Control-Allow-Private-Network"] = "true"
        return headers
