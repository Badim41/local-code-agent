from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
from typing import Any

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8091

# Разрешённые домены CORS по умолчанию (можно расширить или заменить через env/config/CLI)
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "https://sert.yellowfire.ru:444",
    "https://ru.yellowfire.ru",
    "https://yellowfire.ru",
)

DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_SHELL_OUTPUT_CHARS: int = 12_000
DEFAULT_PENDING_SHELL_TTL_SECONDS: int = 1800  # 30 минут


def normalize_origin(origin: str) -> str:
    """Нормализует Origin (обрезает пробелы и завершающий слэш)."""
    return origin.strip().rstrip("/")


def _parse_csv_origins(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [normalize_origin(part) for part in raw.split(",") if part.strip()]


def build_allowed_origins(
    *,
    override_origins: Sequence[str] | None = None,
    extra_origins: Iterable[str] = (),
) -> tuple[str, ...]:
    """Собирает итоговый список разрешённых CORS Origin без дублей."""
    base = (
        [normalize_origin(o) for o in override_origins if o.strip()]
        if override_origins is not None
        else [normalize_origin(o) for o in DEFAULT_ALLOWED_ORIGINS]
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for item in [*base, *(normalize_origin(o) for o in extra_origins if o.strip())]:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Параметры запуска и безопасности локального HTTP-сервера."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    agent_token: str = ""
    require_auth: bool = True
    projects_file: Path = Path("projects.json")
    config_file: Path = Path("agent_config.json")
    default_api_key: str | None = None
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_shell_output_chars: int = DEFAULT_MAX_SHELL_OUTPUT_CHARS
    pending_shell_ttl_seconds: int = DEFAULT_PENDING_SHELL_TTL_SECONDS


def load_server_config(
    *,
    config_file: Path | None = None,
    projects_file: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    cli_extra_origins: Sequence[str] = (),
    no_auth: bool = False,
) -> ServerConfig:
    """Загружает конфигурацию из agent_config.json + переменных окружения + CLI-аргументов."""
    cfg_path = (config_file or Path(os.environ.get("LCA_CONFIG_FILE", "agent_config.json"))).resolve()
    proj_path = (projects_file or Path(os.environ.get("LCA_PROJECTS_FILE", "projects.json"))).resolve()

    raw_cfg = _load_or_init_config_file(cfg_path)

    resolved_host = host or os.environ.get("LCA_HOST") or str(raw_cfg.get("host") or DEFAULT_HOST)
    resolved_port = int(port or os.environ.get("LCA_PORT") or raw_cfg.get("port") or DEFAULT_PORT)

    # Приоритет CORS: LCA_ALLOWED_ORIGINS (полная замена) -> agent_config.allowed_origins -> DEFAULT_ALLOWED_ORIGINS
    env_override = _parse_csv_origins(os.environ.get("LCA_ALLOWED_ORIGINS"))
    cfg_override_raw = raw_cfg.get("allowed_origins")
    override_origins: Sequence[str] | None = None
    if env_override:
        override_origins = env_override
    elif isinstance(cfg_override_raw, list) and cfg_override_raw:
        override_origins = [str(item) for item in cfg_override_raw]

    cfg_extra = raw_cfg.get("extra_allowed_origins")
    extra_list: list[str] = []
    if isinstance(cfg_extra, list):
        extra_list.extend(str(item) for item in cfg_extra)
    extra_list.extend(_parse_csv_origins(os.environ.get("LCA_EXTRA_ALLOWED_ORIGINS")))
    extra_list.extend(cli_extra_origins)

    allowed_origins = build_allowed_origins(
        override_origins=override_origins,
        extra_origins=extra_list,
    )

    token = os.environ.get("LCA_AGENT_TOKEN") or str(raw_cfg.get("agent_token") or "")
    require_auth = False if no_auth else bool(raw_cfg.get("require_auth", True))
    if os.environ.get("LCA_REQUIRE_AUTH", "").strip().lower() in ("0", "false", "no"):
        require_auth = False

    default_api_key = (
        os.environ.get("LCA_NETWORK_TOOLS_API_KEY")
        or (str(raw_cfg["network_tools_api_key"]) if raw_cfg.get("network_tools_api_key") else None)
    )

    return ServerConfig(
        host=resolved_host,
        port=resolved_port,
        allowed_origins=allowed_origins,
        agent_token=token,
        require_auth=require_auth,
        projects_file=proj_path,
        config_file=cfg_path,
        default_api_key=default_api_key,
        max_shell_output_chars=int(
            raw_cfg.get("max_shell_output_chars", DEFAULT_MAX_SHELL_OUTPUT_CHARS)
        ),
    )


def _load_or_init_config_file(cfg_path: Path) -> dict[str, Any]:
    if cfg_path.is_file():
        try:
            loaded: object = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (OSError, json.JSONDecodeError):
            return {}

    initial: dict[str, Any] = {
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "agent_token": f"lca_{secrets.token_urlsafe(24)}",
        "require_auth": True,
        "allowed_origins": None,
        "extra_allowed_origins": [],
        "network_tools_api_key": None,
        "max_shell_output_chars": DEFAULT_MAX_SHELL_OUTPUT_CHARS,
    }
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return initial
