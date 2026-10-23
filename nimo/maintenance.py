"""Local installation doctor and explicit dependency maintenance commands."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CORE_PACKAGES = (
    "fastapi", "uvicorn", "python-multipart", "Pillow", "psutil",
    "numpy", "httpx", "pypdf", "torch",
)


def _run(args, *, capture=False):
    kwargs = {"text": True}
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(args, check=False, **kwargs)
    if result.returncode:
        detail = (result.stderr or "").strip() if capture else ""
        raise RuntimeError(detail or f"Command failed with exit code {result.returncode}")
    return (result.stdout or "").strip() if capture else ""


def doctor():
    packages = {}
    missing = []
    for name in CORE_PACKAGES:
        lookup = "python-multipart" if name == "python-multipart" else name
        try:
            packages[name] = version(lookup)
        except PackageNotFoundError:
            packages[name] = None
            missing.append(name)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pip": bool(shutil.which("pip") or True),
        "packages": packages,
        "missing": missing,
        "healthy": not missing,
    }


def outdated():
    """Explicitly contacts the configured pip index to inspect available updates."""
    raw = _run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"], capture=True)
    rows = json.loads(raw or "[]")
    wanted = {name.casefold().replace("_", "-") for name in CORE_PACKAGES}
    return [row for row in rows if str(row.get("name", "")).casefold().replace("_", "-") in wanted]


def update_dependencies(source=None):
    """Upgrade Nimo and its declared direct dependencies inside the current environment."""
    source = Path(source or Path(__file__).resolve().parents[1]).resolve()
    pyproject = source / "pyproject.toml"
    if not pyproject.exists():
        raise ValueError("Could not find the Nimo source tree")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "-e", str(source)])
    return doctor()


def main():
    parser = argparse.ArgumentParser(description="Check or update the local Nimo Lite installation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--outdated", action="store_true", help="Check the package index for outdated Nimo dependencies")
    group.add_argument("--update", action="store_true", help="Upgrade Nimo's declared dependencies in this environment")
    parser.add_argument("--source", help="Optional Nimo source directory for --update")
    args = parser.parse_args()
    if args.update:
        result = update_dependencies(args.source)
    elif args.outdated:
        result = {"outdated": outdated()}
    else:
        result = doctor()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
