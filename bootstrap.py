"""One-command Nimo Lite installer/launcher using only the Python standard library."""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import venv
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/nadeemmhdm/nimo-lite.git"
ZIP_URL = "https://github.com/nadeemmhdm/nimo-lite/archive/refs/heads/main.zip"
MAX_ZIP = 128 * 1024 * 1024
BOOTSTRAP_URL = "https://raw.githubusercontent.com/nadeemmhdm/nimo-lite/main/bootstrap.py"


def _is_free_threaded_python() -> bool:
    """Return True for CPython builds compiled without the GIL (for example 3.13t)."""
    flag = sysconfig.get_config_var("Py_GIL_DISABLED")
    if str(flag).strip().lower() in {"1", "true", "yes"}:
        return True
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if callable(gil_enabled):
        try:
            return not gil_enabled()
        except Exception:
            return False
    return False


def _probe_python(command: list[str]) -> dict | None:
    """Inspect an interpreter without importing any third-party dependencies."""
    probe = (
        "import json,sys,sysconfig; "
        "flag=sysconfig.get_config_var('Py_GIL_DISABLED'); "
        "free_threaded=str(flag).strip().lower() in ('1','true','yes'); "
        "print(json.dumps({'executable':sys.executable,'free_threaded':free_threaded,"
        "'version':list(sys.version_info[:3])}))"
    )
    try:
        result = subprocess.run(
            [*map(str, command), "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            info = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(info, dict) and info.get("executable"):
            return info
    return None


def _find_standard_python() -> Path | None:
    """Find a regular CPython interpreter with a version supported by Nimo."""
    candidates: list[list[str]] = []
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher:
            minors = list(dict.fromkeys([sys.version_info.minor, 12, 13, 11, 10]))
            candidates.extend([[launcher, f"-3.{minor}"] for minor in minors if minor >= 10])

    names = [
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "python3.12",
        "python3.13",
        "python3.11",
        "python3.10",
        "python3",
        "python",
    ]
    for name in names:
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])

    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        signature = tuple(command)
        if signature in seen:
            continue
        seen.add(signature)
        info = _probe_python(command)
        if not info or info.get("free_threaded"):
            continue
        try:
            version = tuple(int(part) for part in info.get("version", ()))
        except (TypeError, ValueError):
            continue
        if version >= (3, 10):
            return Path(str(info["executable"]))
    return None


def _relaunch_if_free_threaded(arguments: list[str]) -> None:
    """Re-run bootstrap under regular CPython before creating an incompatible venv."""
    if not _is_free_threaded_python():
        return
    interpreter = _find_standard_python()
    if interpreter is None:
        if os.name == "nt":
            raise SystemExit(
                "Nimo Lite cannot install under a free-threaded Python build (for example Python 3.13t). "
                "NumPy, Pillow and PyTorch need regular CPython wheels. Install standard CPython 3.12 "
                "(do not choose the experimental free-threaded build) from https://www.python.org/downloads/windows/, "
                "then run:\n"
                f'py -3.12 -c "import urllib.request; exec(urllib.request.urlopen(\'{BOOTSTRAP_URL}\').read())"'
            )
        raise SystemExit(
            "Nimo Lite cannot install under a free-threaded Python build. Install regular CPython 3.12 or newer "
            "and run bootstrap again with that interpreter."
        )

    print(f"[Nimo] Free-threaded Python detected; restarting bootstrap with regular CPython: {interpreter}")
    script_file = globals().get("__file__")
    arguments = list(arguments)
    if script_file and Path(script_file).is_file():
        command = [str(interpreter), str(Path(script_file).resolve()), *arguments]
    else:
        code = f"import urllib.request; exec(urllib.request.urlopen({BOOTSTRAP_URL!r}, timeout=30).read())"
        command = [str(interpreter), "-c", code, *arguments]
    raise SystemExit(subprocess.call(command))


def run(cmd, cwd=None):
    print("[Nimo]", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def safe_extract(raw, target):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        total = 0
        for info in archive.infolist():
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe repository archive")
            total += info.file_size
            if total > MAX_ZIP:
                raise RuntimeError("Repository archive exceeds safety limit")
        archive.extractall(target)


def obtain_source(target):
    git = shutil.which("git")
    if (target / ".git").exists() and git:
        try:
            remote = subprocess.check_output([git, "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
            dirty = subprocess.check_output([git, "-C", str(target), "status", "--porcelain"], text=True).strip()
            if remote.rstrip("/") in {REPO_URL.rstrip("/"), "https://github.com/nadeemmhdm/nimo-lite"} and not dirty:
                run([git, "-C", target, "pull", "--ff-only", "origin", "main"])
                return
        except Exception:
            pass
    if target.exists() and any(target.iterdir()):
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    if git:
        run([git, "clone", "--depth", "1", REPO_URL, target])
        return
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "NimoLite-Bootstrap/0.2.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(MAX_ZIP + 1)
    if len(raw) > MAX_ZIP:
        raise RuntimeError("Repository download too large")
    temp = target.parent / "nimo-download"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    safe_extract(raw, temp)
    roots = [p for p in temp.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("Unexpected repository archive")
    roots[0].rename(target)
    shutil.rmtree(temp, ignore_errors=True)


def main(argv=None):
    if sys.version_info < (3, 10):
        raise SystemExit("Nimo Lite requires Python 3.10 or newer. Install a supported Python version, then run this command again.")
    parser = argparse.ArgumentParser(description="Install/update and launch Nimo Lite")
    parser.add_argument("--home", default=os.environ.get("NIMO_HOME", str(Path.home() / ".nimo-lite")))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--auto-update", action="store_true", help="Opt in to release-based update checks at launch")
    parser.add_argument("--no-run", action="store_true", help="Install/update only")
    args = parser.parse_args(argv)
    _relaunch_if_free_threaded(list(sys.argv[1:] if argv is None else argv))

    root = Path(args.home).expanduser().resolve()
    source = root / "app"
    env = root / "venv"
    data = root / "data"
    obtain_source(source)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if env.exists():
        env_info = _probe_python([str(python)])
        try:
            env_version = tuple(int(part) for part in env_info.get("version", ())) if env_info else ()
        except (TypeError, ValueError):
            env_version = ()
        if not env_info or env_info.get("free_threaded") or env_version < (3, 10):
            print("[Nimo] Existing virtual environment is incomplete or uses unsupported Python; rebuilding it")
            if env.is_dir():
                shutil.rmtree(env)
            else:
                env.unlink()
    if not env.exists():
        print("[Nimo] Creating isolated Python environment")
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    # Require wheels for native dependencies: do not attempt slow, toolchain-dependent source builds.
    run([
        python, "-m", "pip", "install", "--upgrade",
        "--only-binary=numpy,pillow,psutil,torch",
        "-e", f"{source}[open-data]",
    ])
    data.mkdir(parents=True, exist_ok=True)
    run([python, "-m", "nimo.maintenance"])
    if args.no_run:
        print(f"[Nimo] Installed in {root}")
        return
    print(f"[Nimo] Starting local assistant at http://127.0.0.1:{args.port}")
    os.chdir(source)
    command = [str(python), "-m", "nimo.launcher", "--data", str(data), "--port", str(args.port)]
    if args.auto_update:
        command.append("--auto-update")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
