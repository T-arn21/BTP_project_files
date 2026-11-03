"""Secure bootstrap tests use local Git repositories and a fake secret only."""
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
import src.bootstrap as bootstrap

BRANCH = "dev/unified-ecg-pipeline"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.STDOUT, text=True).strip()


@pytest.fixture
def local_remote(tmp_path):
    remote, producer = tmp_path / "remote.git", tmp_path / "producer"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", BRANCH, str(producer)], check=True, capture_output=True)
    (producer / "value.txt").write_text("first")
    git(producer, "add", "value.txt")
    git(producer, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial")
    git(producer, "remote", "add", "origin", str(remote))
    git(producer, "push", "-u", "origin", BRANCH)
    return remote, producer


def test_clone_update_secret_not_persisted(local_remote, tmp_path, monkeypatch, capsys):
    remote, producer = local_remote
    checkout = tmp_path / "checkout"
    sentinel = "fake_secret_that_must_never_be_written"
    original_git = bootstrap._git
    helpers = []
    def checked_git(arguments, **kwargs):
        env = kwargs.get("env", {})
        if "BTP_GITHUB_TOKEN" in env:
            helper = Path(env["GIT_ASKPASS"])
            helpers.append(helper)
            assert sentinel not in helper.read_text()
            assert all(sentinel not in arg for arg in arguments)
        return original_git(arguments, **kwargs)
    monkeypatch.setattr(bootstrap, "_git", checked_git)
    first = bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: sentinel)
    assert first["commit"] == git(producer, "rev-parse", "HEAD")
    (producer / "value.txt").write_text("second")
    git(producer, "add", "value.txt")
    git(producer, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "update")
    git(producer, "push", "origin", BRANCH)
    second = bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: sentinel)
    assert second["commit"] != first["commit"] and (checkout / "value.txt").read_text() == "second"
    assert all(not path.exists() for path in helpers)
    assert sentinel not in (checkout / ".git/config").read_text()
    assert sentinel not in capsys.readouterr().out
    assert not git(checkout, "status", "--porcelain")


def test_dirty_and_divergent_checkout_preserved(local_remote, tmp_path):
    remote, _ = local_remote
    checkout = tmp_path / "checkout"
    bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: "fake")
    original = git(checkout, "rev-parse", "HEAD")
    note = checkout / "untracked-note.txt"
    note.write_text("preserve untracked content")
    with pytest.raises(RuntimeError, match="Uncommitted"):
        bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: pytest.fail("dirty tree"))
    assert note.read_text() == "preserve untracked content"
    note.unlink()
    (checkout / "value.txt").write_text("uncommitted edits")
    with pytest.raises(RuntimeError, match="Uncommitted"):
        bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: pytest.fail("secret retrieval not needed for dirty tree"))
    assert (checkout / "value.txt").read_text() == "uncommitted edits"
    git(checkout, "add", "value.txt")
    git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "local only")
    local = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="Fast-forward safety"):
        bootstrap.sync_repository(checkout, repository_url=str(remote), token_provider=lambda: "fake")
    assert git(checkout, "rev-parse", "HEAD") == local != original


def test_kaggle_secret_errors_are_redacted(monkeypatch):
    class Client:
        def get_secret(self, name):
            assert name == "GITHUB_TOKEN"
            raise ValueError("fake_secret_provider_detail")
    monkeypatch.setitem(sys.modules, "kaggle_secrets", SimpleNamespace(UserSecretsClient=Client))
    with pytest.raises(RuntimeError) as error:
        bootstrap.kaggle_token()
    assert "fake_secret_provider_detail" not in str(error.value)
    assert error.value.__suppress_context__
    with pytest.raises(ValueError, match="Secrets"):
        bootstrap.execute_notebook({"GITHUB_TOKEN": "fake_secret"})


def test_git_failure_does_not_expose_auth_output(monkeypatch):
    class Process:
        returncode = 1
        def communicate(self, **kwargs):
            return "fake_secret", "password=fake_secret"
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *a, **kw: Process())
    with pytest.raises(RuntimeError) as error:
        bootstrap._git(["fetch"])
    assert "fake_secret" not in str(error.value)


def test_launcher_summary_success_and_bootstrap_failure(monkeypatch, tmp_path, capsys):
    root = Path(__file__).parents[1]
    settings = {"OUTPUT_ROOT": str(tmp_path), "CHECKOUT_DIR": str(root), "GIT_BRANCH": BRANCH, "GIT_TIMEOUT_SECONDS": 20}
    monkeypatch.setattr(bootstrap, "sync_repository", lambda *a, **kw: {"checkout": str(root), "branch": BRANCH, "commit": "a" * 40})
    # Avoid clearing the test suite's module namespace; select_import_root has
    # its own separate fresh-interpreter verification below.
    monkeypatch.setattr(bootstrap, "select_import_root", lambda root: str(Path(root) / "src/__init__.py"))
    def launch(checkout, config, session):
        data = json.loads(Path(config).read_text())
        assert "GITHUB_TOKEN" not in data and data["SOURCE_REVISION"] == "a" * 40
        (Path(session) / "final_summary.json").write_text(json.dumps({"status": "complete", "checks": {"repository_sync": "passed"},
            "completed": {"superclass": {"output_dir": "test_only"}}, "errors": {}, "session_dir": str(session)}))
        return 0
    monkeypatch.setattr(bootstrap, "launch_pipeline", launch)
    result = bootstrap.execute_notebook(settings)
    assert result["status"] == "complete" and "superclass" in result["completed"]
    def fail(*a, **kw):
        raise RuntimeError("Uncommitted changes; checkout untouched")
    monkeypatch.setattr(bootstrap, "sync_repository", fail)
    result = bootstrap.execute_notebook(settings)
    assert result["status"] == "failed" and Path(result["session_dir"], "final_summary.json").is_file()
    assert "EXECUTION SUMMARY" in capsys.readouterr().out


def test_source_precedence_in_fresh_interpreter(tmp_path):
    root = Path(__file__).parents[1]
    stale = tmp_path / "stale"
    (stale / "src").mkdir(parents=True)
    (stale / "src/__init__.py").write_text("STALE=True")
    script = f"""
import sys
sys.path.insert(0, {str(stale)!r})
import src
assert src.STALE
sys.path.insert(0, {str(root)!r})
# Load bootstrap explicitly because src is deliberately the stale package.
import importlib.util
spec=importlib.util.spec_from_file_location('bootstrap_test', {str(root / 'src/bootstrap.py')!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
path=module.select_import_root({str(root)!r})
import src
assert not hasattr(src, 'STALE')
print(path)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert str(root / "src/__init__.py") in result.stdout


def test_atomic_checkpoint_preserves_previous_on_interruption(tmp_path, monkeypatch):
    from src.training import atomic_checkpoint
    path = tmp_path / "last.pth"
    atomic_checkpoint({"epoch": 1}, path)
    original = path.read_bytes()
    def fail(state, destination):
        Path(destination).write_bytes(b"partial interrupted write")
        raise OSError("simulated interruption")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="interruption"):
        atomic_checkpoint({"epoch": 2}, path)
    assert path.read_bytes() == original


def test_controller_failure_stage_and_fresh_worker_environment(tmp_path, monkeypatch):
    import src.unattended as pipeline
    settings = dict(SESSION_DIR=str(tmp_path), CHECKOUT_DIR=str(Path(__file__).parents[1]), DIAGNOSTIC_MODE=False,
                    WHEEL_DIR="", ALLOW_SOURCE_BUILD=True, INSTALL_BASE=True, MAX_JOBS=2)
    def fail(*a, **kw):
        raise RuntimeError("incompatible CUDA binary")
    monkeypatch.setattr(pipeline, "setup_dependencies", fail)
    assert pipeline.controller(settings, tmp_path / "config.json") == 1
    report = json.loads((tmp_path / "final_summary.json").read_text())
    assert report["stage"] == "dependency setup and CUDA verification" and report["status"] == "failed"
    monkeypatch.setattr(pipeline, "setup_dependencies", lambda *a, **kw: {"installation": "test_double"})
    monkeypatch.setenv("GITHUB_TOKEN", "fake_secret")
    monkeypatch.setenv("BTP_GITHUB_TOKEN", "fake_secret")
    calls = []
    def launch(command, log_path, env):
        calls.append(command)
        assert "--worker" in command
        assert "GITHUB_TOKEN" not in env and "BTP_GITHUB_TOKEN" not in env
        assert env["PYTHONPATH"] == settings["CHECKOUT_DIR"]
        complete = json.loads((tmp_path / "final_summary.json").read_text())
        complete["status"] = "complete"
        (tmp_path / "final_summary.json").write_text(json.dumps(complete))
    monkeypatch.setattr(pipeline, "run_logged", launch)
    assert pipeline.controller(settings, tmp_path / "config.json") == 0
    assert len(calls) == 1
    monkeypatch.setattr(pipeline, "run_logged", lambda *a, **kw: None)
    assert pipeline.controller(settings, tmp_path / "config.json") == 1
