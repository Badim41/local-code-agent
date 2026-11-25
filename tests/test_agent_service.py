from __future__ import annotations

from pathlib import Path

from convert_gpt_answer.replacer import ShellReport, ShellStatus

from local_code_agent.config import ServerConfig
from local_code_agent.models import ProjectConfig
from local_code_agent.service import AgentService

_OPEN = "<" * 7
_MID = "=" * 7
_CLOSE = ">" * 7
_FENCE = "`" * 3


class _FakeStructureProvider:
    def __init__(self) -> None:
        self.calls: int = 0

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
        _ = (structure_prompt, api_key, model, mode, detail_level, max_structure_chars, max_prompt_chars, include_edit_instructions)
        self.calls += 1
        files = sorted(p.name for p in project.path.glob("*.py"))
        return f"CODEBASE#{self.calls}:{','.join(files)}"


class _FakeShellRunner:
    def __init__(self, *, log: str = "ok\n", exit_code: int = 0, create_file: str | None = None) -> None:
        self.log = log
        self.exit_code = exit_code
        self.create_file = create_file
        self.ran_scripts: list[str] = []

    def run(self, script: str, cwd: Path, timeout: int) -> ShellReport:
        self.ran_scripts.append(script)
        if self.create_file:
            (cwd / self.create_file).write_text("generated = True\n", encoding="utf-8")
        status = ShellStatus.OK if self.exit_code == 0 else ShellStatus.FAILED
        return ShellReport(
            status=status,
            exit_code=self.exit_code,
            duration_seconds=0.12,
            timeout_used=timeout,
            log=self.log,
        )


def _make_service(
    tmp_path: Path,
    *,
    shell_runner: _FakeShellRunner | None = None,
) -> tuple[AgentService, _FakeStructureProvider, Path]:
    project_dir = tmp_path / "demo_proj"
    project_dir.mkdir()
    (project_dir / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    cfg = ServerConfig(
        agent_token="test-token",
        projects_file=tmp_path / "projects.json",
        config_file=tmp_path / "agent_config.json",
        default_api_key="fake-key",
    )
    provider = _FakeStructureProvider()
    service = AgentService(cfg, structure_provider=provider, shell_runner=shell_runner)
    service.upsert_project({"id": "demo", "name": "Demo", "path": str(project_dir)})
    return service, provider, project_dir


def test_structure_cache_refreshes_only_on_file_changes(tmp_path: Path) -> None:
    service, provider, project_dir = _make_service(tmp_path)

    r1 = service.get_structure(project_id="demo")
    r2 = service.get_structure(project_id="demo")
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    assert provider.calls == 1

    # Изменяем файл -> отпечаток меняется -> вызывается get_structure заново
    (project_dir / "extra.py").write_text("x = 1\n", encoding="utf-8")
    r3 = service.get_structure(project_id="demo")
    assert r3["cache_hit"] is False
    assert provider.calls == 2
    assert "extra.py" in r3["codebase_text"]


def test_apply_edits_and_undo_refresh_structure(tmp_path: Path) -> None:
    service, provider, project_dir = _make_service(tmp_path)

    llm_reply = (
        f"{_FENCE}markdown\n"
        f"{_OPEN} app.py\n"
        f"    return 'hello'\n"
        f"{_MID}\n"
        f"    return 'world'\n"
        f"{_CLOSE}\n"
        f"{_FENCE}\n"
    )
    res = service.apply_llm_response(project_id="demo", llm_text=llm_reply)
    assert res["status"] == "success"
    assert res["files_changed"] is True
    assert res["needs_llm_followup"] is False
    assert "updated_codebase" in res
    assert provider.calls == 1
    assert "return 'world'" in (project_dir / "app.py").read_text(encoding="utf-8")

    # Откат возвращает исходное содержимое
    undo_res = service.undo_project(project_id="demo")
    assert undo_res["status"] == "ok"
    assert "return 'hello'" in (project_dir / "app.py").read_text(encoding="utf-8")
    assert provider.calls == 2


def test_apply_failed_reports_failed_and_rolled_back_blocks(tmp_path: Path) -> None:
    service, _, project_dir = _make_service(tmp_path)

    llm_reply = (
        f"{_FENCE}markdown\n"
        f"{_OPEN} app.py\n"
        f"    return 'hello'\n"
        f"{_MID}\n"
        f"    return 'changed'\n"
        f"{_CLOSE}\n\n"
        f"{_OPEN} app.py\n"
        f"    non_existent_line()\n"
        f"{_MID}\n"
        f"    fixed()\n"
        f"{_CLOSE}\n"
        f"{_FENCE}\n"
    )
    res = service.apply_llm_response(project_id="demo", llm_text=llm_reply)
    assert res["status"] == "patches_failed"
    assert res["files_changed"] is False
    assert res["needs_llm_followup"] is True
    assert res["failed_blocks"] == [2]
    assert res["rolled_back_blocks"] == [1]
    assert "#2" in res["feedback_for_llm"]
    assert "#1" in res["feedback_for_llm"]
    # Файл на диске не изменился благодаря атомарности
    assert "return 'hello'" in (project_dir / "app.py").read_text(encoding="utf-8")


def test_two_phase_powershell_confirmation_and_log_truncation(tmp_path: Path) -> None:
    huge_log = "START\n" + ("noise line\n" * 500) + "FINAL_ASSERTION_ERROR"
    runner = _FakeShellRunner(log=huge_log, exit_code=1, create_file=None)
    service, provider, _ = _make_service(tmp_path, shell_runner=runner)

    llm_reply = f"{_FENCE}powershell\npytest -q\n{_FENCE}\n"
    res1 = service.apply_llm_response(
        project_id="demo",
        llm_text=llm_reply,
        auto_powershell=False,
    )
    assert res1["status"] == "awaiting_shell_confirmation"
    assert res1["requires_shell_confirmation"] is True
    assert runner.ran_scripts == []

    cmd_id = res1["pending_shell"]["id"]
    res2 = service.execute_pending_shell(
        project_id="demo",
        command_id=cmd_id,
        approved=True,
        max_shell_output_chars=400,
    )
    assert res2["status"] == "shell_completed"
    assert res2["files_changed"] is False
    assert res2["needs_llm_followup"] is True
    assert provider.calls == 0
    assert runner.ran_scripts == ["pytest -q"]
    feedback = res2["feedback_for_llm"]
    assert "START" in feedback
    assert "FINAL_ASSERTION_ERROR" in feedback
    assert len(feedback) < len(huge_log)


def test_auto_powershell_refreshes_structure_when_files_changed(tmp_path: Path) -> None:
    runner = _FakeShellRunner(log="generated ok\n", exit_code=0, create_file="generated.py")
    service, provider, project_dir = _make_service(tmp_path, shell_runner=runner)

    llm_reply = f"{_FENCE}powershell\npython gen.py\n{_FENCE}\n"
    res = service.apply_llm_response(
        project_id="demo",
        llm_text=llm_reply,
        auto_powershell=True,
    )
    assert res["status"] == "shell_completed"
    assert res["files_changed"] is True
    assert res["needs_llm_followup"] is True
    assert runner.ran_scripts == ["python gen.py"]
    assert (project_dir / "generated.py").is_file()
    assert provider.calls == 1
    assert "generated.py" in res["updated_codebase"]
    assert "generated ok" in res["feedback_for_llm"]


def test_pending_shell_rejected_does_not_run_command(tmp_path: Path) -> None:
    runner = _FakeShellRunner(log="should not run\n", exit_code=0)
    service, provider, _ = _make_service(tmp_path, shell_runner=runner)

    llm_reply = f"{_FENCE}powershell\nRemove-Item app.py\n{_FENCE}\n"
    res1 = service.apply_llm_response(
        project_id="demo",
        llm_text=llm_reply,
        auto_powershell=False,
    )
    assert res1["status"] == "awaiting_shell_confirmation"
    cmd_id = res1["pending_shell"]["id"]

    res2 = service.execute_pending_shell(
        project_id="demo",
        command_id=cmd_id,
        approved=False,
    )
    assert runner.ran_scripts == []
    assert res2["files_changed"] is False
    assert provider.calls == 0
    assert res2["feedback_for_llm"]


def test_mixed_edits_and_pending_powershell_applies_edits_first(tmp_path: Path) -> None:
    runner = _FakeShellRunner(log="1 passed\n", exit_code=0)
    service, provider, project_dir = _make_service(tmp_path, shell_runner=runner)

    llm_reply = (
        f"{_FENCE}markdown\n"
        f"{_OPEN} app.py\n"
        f"    return 'hello'\n"
        f"{_MID}\n"
        f"    return 'patched'\n"
        f"{_CLOSE}\n"
        f"{_FENCE}\n\n"
        f"{_FENCE}powershell\npytest\n{_FENCE}\n"
    )
    res1 = service.apply_llm_response(
        project_id="demo",
        llm_text=llm_reply,
        auto_powershell=False,
    )
    assert res1["status"] == "awaiting_shell_confirmation"
    assert res1["files_changed"] is True
    assert res1["requires_shell_confirmation"] is True
    assert "return 'patched'" in (project_dir / "app.py").read_text(encoding="utf-8")
    assert runner.ran_scripts == []
    assert provider.calls == 1

    cmd_id = res1["pending_shell"]["id"]
    res2 = service.execute_pending_shell(
        project_id="demo",
        command_id=cmd_id,
        approved=True,
    )
    assert res2["status"] == "shell_completed"
    assert res2["files_changed"] is False
    assert res2["needs_llm_followup"] is True
    assert runner.ran_scripts == ["pytest"]
    assert "1 passed" in res2["feedback_for_llm"]
    assert provider.calls == 1


def test_create_new_file_and_undo_deletes_it(tmp_path: Path) -> None:
    service, provider, project_dir = _make_service(tmp_path)

    llm_reply = (
        f"{_FENCE}markdown\n"
        f"{_OPEN} CREATE: new_mod.py\n"
        f"ANSWER = 42\n"
        f"{_CLOSE}\n"
        f"{_FENCE}\n"
    )
    res = service.apply_llm_response(project_id="demo", llm_text=llm_reply)
    assert res["status"] == "success"
    assert res["files_changed"] is True
    assert (project_dir / "new_mod.py").read_text(encoding="utf-8").strip() == "ANSWER = 42"
    assert provider.calls == 1
    assert "new_mod.py" in res["updated_codebase"]

    undo_res = service.undo_project(project_id="demo")
    assert undo_res["status"] == "ok"
    assert not (project_dir / "new_mod.py").exists()
    assert provider.calls == 2


def test_plain_text_response_is_noop(tmp_path: Path) -> None:
    service, provider, _ = _make_service(tmp_path)

    res = service.apply_llm_response(
        project_id="demo",
        llm_text="Готово, никаких правок в коде не требуется.",
    )
    assert res["status"] == "noop"
    assert res["files_changed"] is False
    assert res["needs_llm_followup"] is False
    assert provider.calls == 0
