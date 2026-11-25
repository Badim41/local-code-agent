# local_code_agent

Локальный HTTP-сервер на `http://127.0.0.1:8091`, который подключает локальные проекты на вашем ПК к веб-чату с ИИ:
- строит и кэширует актуальную кодовую базу через [`gpt_project_prompter`](https://github.com/Badim41/gpt_project_prompter) (обновляется автоматически только при изменении файлов на диске);
- атомарно применяет правки (`CREATE` и `Search/Replace`) и управляет zip-бэкапами/откатом через [`convert_gpt_answer.replacer`](https://github.com/Badim41/convert_gpt_answer);
- поддерживает **двухфазное подтверждение PowerShell** (или авто-запуск) с обрезкой длинного вывода и готовым диагностическим фидбеком для ИИ-агента.

## Установка

Требуется Python 3.10+.

```bash
git clone https://github.com/badim41/local-code-agent.git
cd local-code-agent
pip install -e .
```

Для запуска тестов:
```bash
pip install -e ".[dev]"
pytest
```

## Быстрый старт

1. Скопируйте пример списка проектов (или добавляйте проекты позже через веб-интерфейс / `POST /api/projects`):
   ```bash
   cp projects.example.json projects.json
   ```
   Отредактируйте `projects.json`, указав реальные пути к вашим проектам.

2. Запустите сервер:
   ```bash
   python run_server.py
   ```
   (или `local-code-agent` после `pip install -e .`).

3. При первом запуске автоматически создастся `agent_config.json` со случайным `agent_token`, а в консоль выведется:
   ```text
   local_code_agent запущен: http://127.0.0.1:8091
   CORS:     https://sert.yellowfire.ru:444, https://ru.yellowfire.ru, https://yellowfire.ru
   X-Agent-Token: lca_...
   ```
   Скопируйте значение `X-Agent-Token` в настройки «Подключить кодовую базу» в веб-чате.

## Настройка CORS и безопасности

По умолчанию сервер разрешает запросы **только** со следующих доменов (`DEFAULT_ALLOWED_ORIGINS` в `local_code_agent/config.py`):
- `https://sert.yellowfire.ru:444`
- `https://ru.yellowfire.ru`
- `https://yellowfire.ru`

Все остальные `Origin` отклоняются с кодом `403 Forbidden`. В preflight-ответах (`OPTIONS`) автоматически добавляется `Access-Control-Allow-Private-Network: true` для работы `https://` → `http://127.0.0.1:8091` в Chrome / Edge / Firefox.

### Как расширить или изменить список CORS

1. **Через переменные окружения (высший приоритет):**
   - Добавить домены к списку по умолчанию:
     ```bash
     set LCA_EXTRA_ALLOWED_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
     ```
   - Полностью заменить список по умолчанию:
     ```bash
     set LCA_ALLOWED_ORIGINS=https://my-chat.example.com,http://localhost:3000
     ```
2. **Через `agent_config.json`:**
   - `"extra_allowed_origins": ["http://localhost:5000"]` — добавляет к базовому списку.
   - `"allowed_origins": ["https://my-chat.example.com"]` — заменяет базовый список (если `null` — используется `DEFAULT_ALLOWED_ORIGINS`).
3. **Через аргумент CLI:**
   ```bash
   python run_server.py --allow-origin http://localhost:5000 --allow-origin http://127.0.0.1:5000
   ```

### Основные переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `LCA_HOST` | Хост прослушивания | `127.0.0.1` |
| `LCA_PORT` | Порт сервера | `8091` |
| `LCA_AGENT_TOKEN` | Секретный токен `X-Agent-Token` | из `agent_config.json` |
| `LCA_ALLOWED_ORIGINS` | Полная замена списка CORS (через запятую) | `DEFAULT_ALLOWED_ORIGINS` |
| `LCA_EXTRA_ALLOWED_ORIGINS` | Дополнительные CORS-домены (через запятую) | пусто |
| `LCA_NETWORK_TOOLS_API_KEY` | Запасной ключ `NetworkToolsAPI`, если клиент не передал свой | `None` |
| `LCA_CONFIG_FILE` | Путь к `agent_config.json` | `./agent_config.json` |
| `LCA_PROJECTS_FILE` | Путь к `projects.json` | `./projects.json` |

## HTTP API

Все эндпоинты, кроме `GET /api/health`, требуют заголовок:
```http
X-Agent-Token: <ваш_токен>
```
Ключ для `NetworkToolsAPI` можно передавать в теле запроса (`"api_key"`) или в заголовке `X-Network-Tools-Key`.

### `GET /api/health`
Проверка доступности локального сервера. Возвращает версию, `auth_required`, `default_model` (`gemini-4-hydrogen`) и `default_edit_instructions`.

### `GET /api/projects`
Возвращает список зарегистрированных проектов с флагом `exists`.

### `POST /api/projects`
Добавляет или обновляет проект:
```json
{
  "name": "My App",
  "path": "C:/Projects/my-app",
  "ignore_folders": ["node_modules", ".venv"],
  "ignore_files": ["secret.py", ".env"],
  "model": "gemini-4-hydrogen"
}
```

### `DELETE /api/projects/{project_id}`
Удаляет проект из `projects.json` (файлы на диске не удаляются).

### `POST /api/structure`
Возвращает актуальную кодовую базу проекта. Если файлы на диске не изменились с прошлого вызова и `force_refresh: false`, отдаёт мгновенный кэш без повторного запроса к LLM.

```json
{
  "project_id": "my-app",
  "structure_prompt": "Выведи все файлы проекта для агента",
  "api_key": "ВАШ_КЛЮЧ",
  "mode": "gpt_prompt",
  "force_refresh": false
}
```
Режимы `mode`:
- `"gpt_prompt"` — умный отбор и полный текст файлов через `get_gpt_prompt`;
- `"structure_only"` — только индекс/структура через `get_project_structure` (работает без `api_key`).

### `POST /api/apply`
Разбирает ответ нейросети, атомарно применяет блоки правок и обрабатывает PowerShell-блок:

```json
{
  "project_id": "my-app",
  "llm_text": "...",
  "auto_powershell": false,
  "max_shell_output_chars": 12000,
  "refresh_structure_on_change": true,
  "structure_prompt": "...",
  "api_key": "ВАШ_КЛЮЧ"
}
```

Возможные значения `status` в ответе:
- `"noop"` — в ответе нет ни правок, ни PowerShell (обычный текст модели).
- `"success"` — правки применены, PowerShell не запрашивался (`updated_codebase` прикладывается при `files_changed: true`).
- `"patches_failed"` — транзакция отменена; в ответе `failed_blocks`, `rolled_back_blocks`, `edits.errors` и готовый `feedback_for_llm` для отправки обратно нейросети (`needs_llm_followup: true`).
- `"awaiting_shell_confirmation"` — правки применены (если были), но PowerShell ждёт подтверждения в чате; в ответе `pending_shell: { id, project_id, script, timeout_used }`.
- `"shell_completed"` — при `auto_powershell: true` скрипт выполнен сразу; в ответе `shell`, `shell_changed_files`, `files_changed`, `feedback_for_llm` (`needs_llm_followup: true`) и `updated_codebase` (только если файлы изменились).

### `POST /api/shell/execute`
Выполняет или отклоняет команду из `pending_shell`:
```json
{
  "project_id": "my-app",
  "command_id": "sh_...",
  "approved": true,
  "max_shell_output_chars": 12000,
  "refresh_structure_on_change": true,
  "api_key": "ВАШ_КЛЮЧ"
}
```
При `approved: false` возвращает `status: "shell_rejected"` и `feedback_for_llm` с уведомлением об отказе пользователя.

### `GET /api/backups?project_id=my-app`
Возвращает список zip-бэкапов из `.replacer_backups`.

### `POST /api/undo`
Откатывает последний (или указанный `backup_id`) бэкап и сбрасывает кэш структуры:
```json
{
  "project_id": "my-app",
  "backup_id": null,
  "refresh_structure": true,
  "api_key": "ВАШ_КЛЮЧ"
}
```
