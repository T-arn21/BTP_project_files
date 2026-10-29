"""Standard-library-only setup, so dependencies can be installed before src imports.

The successful user-reported build is preserved, not presented as universal
wheel compatibility. No install runs on import and Torch is never requested.
"""
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from time import monotonic

VERSIONS = {"causal-conv1d": "1.7.0", "mamba-ssm": "2.3.2.post1", "wfdb": "4.3.1"}


def preserve_wheels(destination, roots=()):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("PIP_CACHE_DIR", str(Path.home() / ".cache/pip"))) / "wheels"
    for root in [cache, *map(Path, roots)]:
        for prefix in ("mamba_ssm-2.3.2.post1", "causal_conv1d-1.7.0"):
            for wheel in root.rglob(prefix + "*.whl"):
                target = destination / wheel.name
                if wheel.resolve() != target.resolve():
                    shutil.copy2(wheel, target)
    names = sorted(p.name for p in destination.glob("*.whl"))
    print("Preserved wheel artifacts:", names or "none found; retain your supplied wheels or source-build outputs", flush=True)
    return names


@contextmanager
def stage_log(directory, stage):
    """Persist stage and errors even before scientific libraries are importable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    def status(value):
        (directory / "pipeline_status.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log
        def write(self, message):
            self.stream.write(message)
            self.log.write(message)
            self.flush()
        def flush(self):
            self.stream.flush()
            self.log.flush()
    with (directory / "pipeline.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            status(dict(status="running", stage=stage))
            print(f"START {stage}", flush=True)
            stopped = threading.Event()
            def heartbeat():
                while not stopped.wait(30):
                    print(f"Still running stage '{stage}' ({monotonic() - start:.0f}s); inspect the latest batch/cache message above", flush=True)
            monitor = threading.Thread(target=heartbeat, daemon=True)
            monitor.start()
            try:
                yield
            except Exception as exc:
                traceback.print_exc()
                status(dict(status="failed", stage=stage, error=str(exc), seconds=monotonic() - start))
                raise
            else:
                print(f"DONE {stage}: {monotonic() - start:.2f}s", flush=True)
                status(dict(status="stage_complete", stage=stage, seconds=monotonic() - start))
            finally:
                stopped.set()
                monitor.join(timeout=1)


def run_logged(command, log_path, env=None):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(map(str, command)), "Log:", path, flush=True)
    start = monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", env=env)
    lines = queue.Queue()
    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=read_lines, daemon=True).start()
    with path.open("a", encoding="utf-8", buffering=1) as log:
        while True:
            try:
                line = lines.get(timeout=15)
            except queue.Empty:
                line = f"Still running ({monotonic() - start:.0f}s): {command[-1]}\n"
            if line is None:
                break
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
        if code:
            raise RuntimeError(f"Command failed with exit code {code}; inspect {path}")


def probe_mamba():
    """Fresh interpreter: verify the exact user-supplied CUDA backward probe."""
    script = """
import json, torch
from mamba_ssm import Mamba
assert torch.cuda.is_available(), 'Kaggle CUDA GPU is required'
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 100, 64, device='cuda', requires_grad=True)
y = model(x)
loss = y.square().mean()
loss.backward()
assert y.shape == (2,100,64) and torch.isfinite(loss)
assert x.grad is not None and torch.isfinite(x.grad).all()
torch.cuda.synchronize()
print(json.dumps({'probe': 'cuda_forward_backward_passed', 'torch': torch.__version__,
 'cuda': torch.version.cuda, 'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
 'gpu': torch.cuda.get_device_name(0)}))
"""
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Mamba CUDA probe timed out after 120s"


def setup_dependencies(repo_dir, output_root, *, wheel_dir="", allow_source_build=False,
                       install_base=True, max_jobs=2, require_gpu=True):
    """Prefer working install, then supplied wheels, then explicit source build.

    A fresh subprocess probes compiled extensions to avoid stale failed imports.
    Setup must precede importing third-party packages in the final notebook.
    """
    out = Path(output_root)
    logs = out / "setup_logs"
    logs.mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip"]
    base = ("numpy", "pandas", "scipy", "sklearn", "wfdb", "matplotlib")
    missing = [name for name in (base + ("einops", "transformers") if require_gpu else base)
               if importlib.util.find_spec(name) is None]
    if not require_gpu and importlib.util.find_spec("torch") is None:
        raise RuntimeError("CPU provenance audit requires preinstalled Torch for weights_only checkpoint loading; no CUDA extensions are required")
    if missing:
        if not install_base:
            raise RuntimeError(f"Missing base dependencies {missing}; enable INSTALL_BASE")
        if require_gpu:
            args = ["-r", str(Path(repo_dir) / "requirements-kaggle.txt")]
        else:
            specs = {"numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<3", "scipy": "scipy>=1.13,<2",
                     "sklearn": "scikit-learn>=1.5,<2", "wfdb": "wfdb==4.3.1", "matplotlib": "matplotlib>=3.8,<4"}
            args = [specs[name] for name in missing]
        run_logged(pip + ["install", *args], logs / "base.log")
    else:
        print("Base dependencies already present; skipping installation", flush=True)
    if not require_gpu:
        print("CPU provenance-only mode: CUDA/Mamba probes, wheel installation and source builds skipped", flush=True)
        return {"installation": "cpu_provenance_only", "cuda": "not_required", "mamba_extensions": "not_required"}
    works, details = probe_mamba()
    (logs / "cuda_probe_before.txt").write_text(details, encoding="utf-8")
    if works:
        print("Installed Mamba CUDA forward/backward works; skipping installation\n" + details, flush=True)
        return {"installation": "already_working", "probe": details,
                "preserved_wheels": preserve_wheels(out / "mamba_wheels")}
    print("CUDA probe not ready:", details[-3000:], flush=True)
    gpu = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'Select Kaggle GPU; keep its CUDA Torch installation'"],
                         capture_output=True, text=True)
    if gpu.returncode:
        raise RuntimeError(gpu.stdout + gpu.stderr)
    wheel_roots = [Path(wheel_dir)] if wheel_dir else sorted({p.parent for prefix in ("mamba_ssm", "causal_conv1d")
                    for p in Path("/kaggle/input").rglob(prefix + "*.whl")})
    if wheel_dir and not Path(wheel_dir).is_dir():
        raise FileNotFoundError(f"Wheel directory missing: {wheel_dir}")
    specs = [f"{name}=={VERSIONS[name]}" for name in ("causal-conv1d", "mamba-ssm")]
    if wheel_roots:
        try:
            args = [arg for root in wheel_roots for arg in ("--find-links", str(root))]
            run_logged(pip + ["install", "--no-deps", "--force-reinstall", "--no-index", *args, *specs], logs / "wheels.log")
            works, details = probe_mamba()
            if works:
                (logs / "cuda_probe_after.txt").write_text(details, encoding="utf-8")
                return {"installation": "supplied_wheels", "probe": details,
                        "preserved_wheels": preserve_wheels(out / "mamba_wheels", wheel_roots)}
        except RuntimeError as exc:
            print(f"Supplied wheel installation failed: {exc}", flush=True)
    if not allow_source_build:
        raise RuntimeError("No working compatible CUDA wheels. Set ALLOW_SOURCE_BUILD=True in the initial configuration "
                           "to explicitly compile, or attach matching wheels. See setup_logs.")
    destination = out / "mamba_wheels"
    destination.mkdir(parents=True, exist_ok=True)
    build_missing = [name for name in ("ninja", "packaging") if importlib.util.find_spec(name) is None]
    if build_missing:
        if not install_base:
            raise RuntimeError(f"Source build requires {build_missing}; enable INSTALL_BASE or install them first")
        run_logged(pip + ["install", "packaging>=24", "ninja>=1.11"], logs / "build_tools.log")
    env = os.environ.copy()
    env.update(MAMBA_KEEP_CUDA_BUILD="TRUE", MAX_JOBS=str(max_jobs))
    # Always build to a wheel directory so expensive source builds survive as artifacts.
    for spec in specs:
        run_logged(pip + ["wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
                          "--no-binary", "mamba-ssm,causal-conv1d", "-v", "--wheel-dir", str(destination), spec],
                   logs / (spec.split("==")[0] + "_build.log"), env=env)
    run_logged(pip + ["install", "--no-deps", "--force-reinstall", "--no-index", "--find-links", str(destination), *specs], logs / "source_install.log")
    works, details = probe_mamba()
    (logs / "cuda_probe_after.txt").write_text(details, encoding="utf-8")
    if not works:
        raise RuntimeError("CUDA verification failed after source build; see setup_logs/cuda_probe_after.txt\n" + details[-3000:])
    return {"installation": "source_built_saved_wheels", "probe": details}
