"""Self-contained standard-library bootstrap, embedded verbatim in the notebook.

The GitHub secret exists only in memory and Git child environments. It never
appears in URLs, argv, configuration, askpass file contents or returned results.
"""
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone
from uuid import uuid4


def _git(arguments, *, cwd=None, env=None, timeout=600, label="Git"):
    command = ["git", "-c", "credential.helper=", "-c", "core.hooksPath=" + os.devnull,
               "-c", "core.fsmonitor=false", *arguments]
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, errors="replace")
    start = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=min(30, timeout))
                break
            except subprocess.TimeoutExpired:
                print(f"{label} still running ({time.monotonic() - start:.0f}s)", flush=True)
                if time.monotonic() - start >= timeout:
                    raise RuntimeError(f"{label} timed out; authentication details suppressed") from None
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise
    if process.returncode:
        # Never forward authentication-related subprocess output or exception repr.
        raise RuntimeError(f"{label} failed (exit {process.returncode}). Check Internet, repository read permission, "
                           "branch and Kaggle GITHUB_TOKEN secret. Git output suppressed for credential safety.") from None
    return stdout.strip()


def kaggle_token():
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty secret")
        return value.strip()
    except Exception:
        raise RuntimeError("Cannot retrieve Kaggle secret GITHUB_TOKEN. Add it in Kaggle Secrets and grant this notebook access.") from None


def sync_repository(checkout, *, branch="dev/unified-ecg-pipeline", token_provider=None,
                    repository_url="https://github.com/T-arn21/BTP_project_files.git", timeout=600):
    """Clone/update clean checkouts by fast-forward only; never stash/reset/clean."""
    checkout = Path(checkout).resolve()
    parsed = urllib.parse.urlsplit(repository_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Credential-bearing or parameterized Git URLs are forbidden")
    if parsed.scheme and parsed.scheme != "https" and not Path(repository_url).exists():
        raise ValueError("Only HTTPS repositories or explicit local test repositories are supported")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", branch) or ".." in branch:
        raise ValueError("Invalid branch name")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GIT_TRACE", "GIT_CONFIG_")) and k not in {"GIT_CURL_VERBOSE", "GITHUB_TOKEN", "BTP_GITHUB_TOKEN"}}
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    exists = checkout.exists() and any(checkout.iterdir())
    if exists:
        top = _git(["rev-parse", "--show-toplevel"], cwd=checkout, env=env, label="Repository safety check")
        if Path(top).resolve() != checkout:
            raise RuntimeError("Checkout path is not its own Git repository; refusing to overwrite existing files")
        if _git(["status", "--porcelain", "--untracked-files=all"], cwd=checkout, env=env):
            raise RuntimeError("Uncommitted tracked/untracked local changes detected; checkout left untouched. Save them before unattended synchronization.")
        remote = _git(["remote", "get-url", "--all", "origin"], cwd=checkout, env=env)
        if remote != repository_url:
            raise RuntimeError("Existing origin differs from the configured repository; refusing synchronization")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    token = (token_provider or kaggle_token)()
    if not isinstance(token, str) or not token:
        raise RuntimeError("A nonempty GitHub credential is required")
    helper = None
    auth_env = None
    try:
        # File contains only environment lookup, never the secret itself.
        with tempfile.NamedTemporaryFile("w", suffix="_btp_askpass.py", delete=False, encoding="utf-8") as stream:
            helper = Path(stream.name)
            stream.write("#!/usr/bin/env python3\nimport os, sys\n"
                         "print('x-access-token' if 'username' in ' '.join(sys.argv[1:]).lower() "
                         "else os.environ['BTP_GITHUB_TOKEN'])\n")
        helper.chmod(0o700)
        auth_env = dict(env, GIT_ASKPASS=str(helper), BTP_GITHUB_TOKEN=token)
        if not exists:
            _git(["clone", "--single-branch", "--branch", branch, repository_url, str(checkout)],
                 env=auth_env, timeout=timeout, label="Git clone")
        else:
            _git(["fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
                 cwd=checkout, env=auth_env, timeout=timeout, label="Git fetch")
            if _git(["status", "--porcelain", "--untracked-files=all"], cwd=checkout, env=env):
                raise RuntimeError("Local changes appeared during fetch; refusing to update the checkout")
            # switch never forces files; Git refuses conflicting local state.
            _git(["switch", branch], cwd=checkout, env=env, label="Branch selection")
            _git(["merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"], cwd=checkout, env=env,
                 label="Fast-forward safety check (local-only/divergent commits are preserved)")
            _git(["merge", "--ff-only", f"origin/{branch}"], cwd=checkout, env=env, label="Fast-forward update")
        head = _git(["rev-parse", "HEAD"], cwd=checkout, env=env)
        remote_head = _git(["rev-parse", f"origin/{branch}"], cwd=checkout, env=env)
        if head != remote_head or not re.fullmatch(r"[a-f0-9]{40,64}", head):
            raise RuntimeError("Synchronized checkout does not match the fetched remote branch")
        return {"checkout": str(checkout), "branch": branch, "commit": head}
    finally:
        if auth_env is not None:
            auth_env.pop("BTP_GITHUB_TOKEN", None)
        token = None
        if helper is not None:
            helper.unlink(missing_ok=True)


def select_import_root(checkout):
    checkout = Path(checkout).resolve()
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path.insert(0, str(checkout))
    importlib.invalidate_caches()
    module = importlib.import_module("src")
    if Path(module.__file__).resolve().parent != checkout / "src":
        raise RuntimeError("Python resolved src outside the synchronized repository")
    return str(Path(module.__file__).resolve())


def launch_pipeline(checkout, config_path, session_dir):
    """Fresh controller/worker interpreters avoid kernel restarts and stale binaries."""
    checkout = Path(checkout).resolve()
    env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "BTP_GITHUB_TOKEN"}}
    env["PYTHONPATH"] = str(checkout)
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    log = Path(session_dir) / "launcher.log"
    # No secret is available to the installer, worker or its logs.
    command = [sys.executable, "-u", "-m", "src.unattended", "--config", str(config_path)]
    process = subprocess.Popen(command, cwd=checkout, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace",
                               start_new_session=os.name != "nt")
    import queue
    import threading
    lines = queue.Queue()
    def read():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=read, daemon=True).start()
    try:
        with log.open("a", encoding="utf-8", buffering=1) as stream:
            while True:
                try:
                    line = lines.get(timeout=30)
                except queue.Empty:
                    line = "Pipeline process is still running; see current stage in pipeline_status.json\n"
                if line is None:
                    break
                print(line, end="", flush=True)
                stream.write(line)
        return process.wait()
    except BaseException:
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        raise


def execute_notebook(settings):
    """Entry point for the notebook: always print a concise safe execution summary."""
    if any("TOKEN" in key.upper() or "SECRET" in key.upper() for key in settings):
        raise ValueError("Secrets must not be passed in execution settings; use Kaggle Secrets only")
    out = Path(settings["OUTPUT_ROOT"]).resolve()
    session = out / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=True)
    config = dict(settings, SESSION_DIR=str(session))
    summary_path = session / "final_summary.json"
    summary = {"status": "failed", "stage": "GitHub synchronization", "completed": {},
               "checks": {}, "errors": {}, "session_dir": str(session)}
    try:
        synchronized = sync_repository(settings["CHECKOUT_DIR"], branch=settings["GIT_BRANCH"],
                                       timeout=settings["GIT_TIMEOUT_SECONDS"])
        config["SOURCE_REVISION"] = synchronized["commit"]
        config["IMPORT_ROOT"] = select_import_root(synchronized["checkout"])
        print(f"Synchronized {synchronized['branch']} @ {synchronized['commit'][:12]}; imports: {config['IMPORT_ROOT']}", flush=True)
        path = session / "execution_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        summary["checks"]["repository_sync"] = "passed"
        summary["stage"] = "dependency setup / pipeline worker"
        code = launch_pipeline(synchronized["checkout"], path, session)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if code != 0:
            summary["status"] = "failed"
            summary["errors"].setdefault("pipeline", "Pipeline process failed; inspect stage status and logs")
        return summary
    except BaseException as exc:
        # All synchronization exceptions are credential-free by construction.
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["errors"][summary.get("stage", "launcher")] = str(exc) or type(exc).__name__
        return summary
    finally:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("EXECUTION SUMMARY:", json.dumps({"status": summary["status"], "checks": summary.get("checks", {}),
              "completed_experiments": list(summary.get("completed", {})), "errors": summary.get("errors", {}),
              "results": str(session)}, indent=2), flush=True)
