from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

from .config import load_server_config
from .server import create_http_server


def build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер запуска локального HTTP-сервера."""
    parser = argparse.ArgumentParser(
        prog="local-code-agent",
        description="Локальный HTTP-сервер (127.0.0.1:8091) для работы веб-чата с кодовой базой.",
    )
    parser.add_argument("--host", default=None, help="Хост прослушивания (по умолчанию 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Порт (по умолчанию 8091).")
    parser.add_argument("--config", type=Path, default=None, help="Путь к agent_config.json.")
    parser.add_argument("--projects", type=Path, default=None, help="Путь к projects.json.")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Добавить дополнительный CORS Origin (можно указывать несколько раз).",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Отключить проверку заголовка X-Agent-Token (не рекомендуется).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить подробное логирование DEBUG.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Точка входа для запуска HTTP-сервера local_code_agent."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_server_config(
        config_file=args.config,
        projects_file=args.projects,
        host=args.host,
        port=args.port,
        cli_extra_origins=args.allow_origin,
        no_auth=args.no_auth,
    )
    server = create_http_server(config)
    addr_host, addr_port = server.server_address[:2]

    print("=" * 64)
    print(f"local_code_agent запущен: http://{addr_host}:{addr_port}")
    print(f"Конфиг:   {config.config_file}")
    print(f"Проекты:  {config.projects_file}")
    print(f"CORS:     {', '.join(config.allowed_origins)}")
    if config.require_auth:
        print(f"X-Agent-Token: {config.agent_token}")
    else:
        print("ВНИМАНИЕ: Проверка X-Agent-Token ОТКЛЮЧЕНА (--no-auth).")
    print("=" * 64)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка local_code_agent...")
    finally:
        server.server_close()
