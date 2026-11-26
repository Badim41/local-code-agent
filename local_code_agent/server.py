from __future__ import annotations

import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from gpt_project_prompter import DEFAULT_MODEL, get_edit_instructions

from .config import ServerConfig
from .cors import CorsPolicy
from .service import AgentService, ServiceError

logger = logging.getLogger(__name__)


class LocalAgentHTTPServer(ThreadingHTTPServer):
    """Многопоточный HTTP-сервер с привязанными config, CORS и AgentService."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        config: ServerConfig,
        service: AgentService,
        cors: CorsPolicy,
    ) -> None:
        self.config = config
        self.service = service
        self.cors = cors
        super().__init__(server_address, LocalAgentRequestHandler)


class LocalAgentRequestHandler(BaseHTTPRequestHandler):
    """Обработчик JSON API для управления кодовой базой, правками и PowerShell."""

    protocol_version = "HTTP/1.1"
    server: LocalAgentHTTPServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        if not self.server.cors.is_origin_allowed(origin):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Origin запрещён политикой CORS."})
            return
        headers = self.server.cors.build_headers(
            origin,
            is_preflight=True,
            private_network_requested=True,
        )
        self.send_response(HTTPStatus.NO_CONTENT)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._handle_request("GET")

    def do_POST(self) -> None:
        self._handle_request("POST")

    def do_DELETE(self) -> None:
        self._handle_request("DELETE")

    def _handle_request(self, method: str) -> None:
        origin = self.headers.get("Origin")
        if not self.server.cors.is_origin_allowed(origin):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Origin запрещён политикой CORS."})
            return

        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if method == "GET" and route == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "local_code_agent",
                    "version": "1.0.0",
                    "auth_required": self.server.config.require_auth,
                    "default_model": DEFAULT_MODEL,
                    "default_edit_instructions": get_edit_instructions(),
                },
            )
            return

        if not self._check_auth():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "Неверный или отсутствующий заголовок X-Agent-Token."},
            )
            return

        try:
            status_code, payload = self._dispatch(method, route, query)
            self._send_json(status_code, payload)
        except ServiceError as exc:
            self._send_json(exc.status_code, {"status": "error", "error": exc.message})
        except Exception as exc:
            logger.exception("Внутренняя ошибка обработчика %s %s", method, route)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": f"Внутренняя ошибка сервера: {exc}"},
            )

    def _dispatch(
        self,
        method: str,
        route: str,
        query: dict[str, list[str]],
    ) -> tuple[int, dict[str, Any]]:
        svc = self.server.service

        if method == "GET" and route == "/api/projects":
            return HTTPStatus.OK, {"status": "ok", "projects": svc.list_projects()}

        if method == "POST" and route == "/api/projects":
            body = self._read_json_body()
            proj = svc.upsert_project(body)
            return HTTPStatus.CREATED, {"status": "ok", "project": proj}

        if method == "DELETE" and route.startswith("/api/projects/"):
            project_id = unquote(route[len("/api/projects/") :]).strip()
            if not project_id:
                raise ServiceError("Не указан project_id.", HTTPStatus.BAD_REQUEST)
            return HTTPStatus.OK, svc.delete_project(project_id)

        if method == "POST" and route == "/api/structure":
            body = self._read_json_body()
            api_key = self._extract_api_key(body)
            res = svc.get_structure(
                project_id=_require_str(body, "project_id"),
                structure_prompt=_optional_str(body, "structure_prompt"),
                api_key=api_key,
                model=_optional_str(body, "model"),
                mode=str(body.get("mode") or "gpt_prompt"),
                detail_level=_optional_int(body, "detail_level"),
                max_structure_chars=_optional_int(body, "max_structure_chars"),
                max_prompt_chars=_optional_int(body, "max_prompt_chars"),
                force_refresh=bool(body.get("force_refresh", False)),
                include_edit_instructions=bool(body.get("include_edit_instructions", False)),
                extra_files=_optional_str_list(body, "extra_files"),
            )
            return HTTPStatus.OK, res

        if method == "POST" and route == "/api/apply":
            body = self._read_json_body()
            api_key = self._extract_api_key(body)
            res = svc.apply_llm_response(
                project_id=_require_str(body, "project_id"),
                llm_text=_require_str(body, "llm_text", allow_empty=True),
                auto_powershell=bool(body.get("auto_powershell", False)),
                powershell_timeout=_optional_int(body, "powershell_timeout"),
                max_shell_output_chars=_optional_int(body, "max_shell_output_chars"),
                refresh_structure_on_change=bool(body.get("refresh_structure_on_change", True)),
                structure_prompt=_optional_str(body, "structure_prompt"),
                api_key=api_key,
                model=_optional_str(body, "model"),
                mode=str(body.get("mode") or "gpt_prompt"),
                extra_files=_optional_str_list(body, "extra_files"),
                inspect_only=bool(body.get("inspect_only", False)),
            )
            return HTTPStatus.OK, res

        if method == "POST" and route == "/api/shell/execute":
            body = self._read_json_body()
            api_key = self._extract_api_key(body)
            res = svc.execute_pending_shell(
                project_id=_require_str(body, "project_id"),
                command_id=_require_str(body, "command_id"),
                approved=bool(body.get("approved", False)),
                timeout=_optional_int(body, "timeout"),
                max_shell_output_chars=_optional_int(body, "max_shell_output_chars"),
                refresh_structure_on_change=bool(body.get("refresh_structure_on_change", True)),
                structure_prompt=_optional_str(body, "structure_prompt"),
                api_key=api_key,
                model=_optional_str(body, "model"),
                mode=str(body.get("mode") or "gpt_prompt"),
                extra_files=_optional_str_list(body, "extra_files"),
            )
            return HTTPStatus.OK, res

        if method == "GET" and route == "/api/backups":
            pids = query.get("project_id")
            if not pids or not pids[0].strip():
                raise ServiceError("Укажите query-параметр project_id.", HTTPStatus.BAD_REQUEST)
            return HTTPStatus.OK, svc.list_project_backups(pids[0].strip())

        if method == "POST" and route == "/api/undo":
            body = self._read_json_body()
            api_key = self._extract_api_key(body)
            res = svc.undo_project(
                project_id=_require_str(body, "project_id"),
                backup_id=_optional_str(body, "backup_id"),
                refresh_structure=bool(body.get("refresh_structure", True)),
                structure_prompt=_optional_str(body, "structure_prompt"),
                api_key=api_key,
                model=_optional_str(body, "model"),
                mode=str(body.get("mode") or "gpt_prompt"),
                extra_files=_optional_str_list(body, "extra_files"),
            )
            return HTTPStatus.OK, res

        raise ServiceError(f"Маршрут не найден: {method} {route}", HTTPStatus.NOT_FOUND)

    def _check_auth(self) -> bool:
        if not self.server.config.require_auth:
            return True
        expected = self.server.config.agent_token
        provided = self.headers.get("X-Agent-Token", "")
        if not expected or not provided:
            return False
        return hmac.compare_digest(provided, expected)

    def _extract_api_key(self, body: dict[str, Any]) -> str | None:
        header_key = self.headers.get("X-Network-Tools-Key", "").strip()
        if header_key:
            return header_key
        raw = body.get("api_key")
        return str(raw).strip() if isinstance(raw, str) and raw.strip() else None

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ServiceError("Некорректный заголовок Content-Length.", 400) from exc
        if length < 0 or length > self.server.config.max_body_bytes:
            raise ServiceError("Превышен допустимый размер тела запроса.", 413)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceError(f"Некорректный JSON в теле запроса: {exc}", 400) from exc
        if not isinstance(parsed, dict):
            raise ServiceError("Тело запроса должно быть JSON-объектом.", 400)
        return parsed

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        origin = self.headers.get("Origin")
        cors_headers = self.server.cors.build_headers(
            origin,
            private_network_requested=bool(
                self.headers.get("Access-Control-Request-Private-Network")
            ),
        )
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.send_header("Cache-Control", "no-store")
        for key, value in cors_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw_bytes)


def create_http_server(
    config: ServerConfig,
    *,
    service: AgentService | None = None,
) -> LocalAgentHTTPServer:
    """Создаёт экземпляр LocalAgentHTTPServer, готовый к serve_forever()."""
    active_service = service or AgentService(config)
    cors = CorsPolicy(config.allowed_origins)
    return LocalAgentHTTPServer(
        (config.host, config.port),
        config=config,
        service=active_service,
        cors=cors,
    )


def _require_str(body: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    val = body.get(key)
    if not isinstance(val, str):
        raise ServiceError(f"Поле '{key}' должно быть строкой.", 400)
    if not allow_empty and not val.strip():
        raise ServiceError(f"Поле '{key}' не может быть пустым.", 400)
    return val if allow_empty else val.strip()


def _optional_str(body: dict[str, Any], key: str) -> str | None:
    val = body.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ServiceError(f"Поле '{key}' должно быть строкой.", 400)
    return val


def _optional_int(body: dict[str, Any], key: str) -> int | None:
    val = body.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise ServiceError(f"Поле '{key}' должно быть целым числом.", 400)
    return val


def _optional_str_list(body: dict[str, Any], key: str) -> list[str] | None:
    val = body.get(key)
    if val is None:
        return None
    if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
        raise ServiceError(f"Поле '{key}' должно быть списком строк.", 400)
    return [item for item in val if item.strip()]
