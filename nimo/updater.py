"""Release-based updater for Nimo Lite. Update checks are explicit/opt-in."""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

REPO = "nadeemmhdm/nimo-lite"
API = "https://api.github.com/repos/" + REPO + "/releases/latest"
MAX_UPDATE = 128 * 1024 * 1024
ALLOWED_HOSTS = {"api.github.com", "github.com", "codeload.github.com", "objects.githubusercontent.com"}


def current_version():
    try:
        return version("nimo-lite")
    except PackageNotFoundError:
        return "0.2.0"


def _semver(value):
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(value))
    return tuple(map(int, match.groups())) if match else (0, 0, 0)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = (urlsplit(newurl).hostname or "").casefold()
        if host not in ALLOWED_HOSTS:
            raise ValueError("Update redirect left the trusted GitHub hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _request(url):
    host = (urlsplit(url).hostname or "").casefold()
    if host not in ALLOWED_HOSTS:
        raise ValueError("Untrusted update host")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NimoLite-Updater/0.2.0", "Accept": "application/vnd.github+json"},
    )
    opener = urllib.request.build_opener(_SafeRedirect())
    with opener.open(req, timeout=15) as response:
        data = response.read(MAX_UPDATE + 1)
    if len(data) > MAX_UPDATE:
        raise ValueError("Update response exceeds safety limit")
    return data


def check_for_update():
    payload = json.loads(_request(API))
    latest = str(payload.get("tag_name") or "")
    current = current_version()
    return {
        "current": current,
        "latest": latest,
        "available": _semver(latest) > _semver(current),
        "published_at": payload.get("published_at"),
        "html_url": payload.get("html_url"),
        "zipball_url": payload.get("zipball_url"),
    }


def _safe_extract(raw, destination):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        total = 0
        infos = archive.infolist()
        if len(infos) > 10000:
            raise ValueError("Update archive has too many files")
        for info in infos:
            path = Path(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe update archive path")
            total += info.file_size
            if total > MAX_UPDATE:
                raise ValueError("Expanded update exceeds safety limit")
        archive.extractall(destination)
    roots = [p for p in Path(destination).iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError("Unexpected update archive layout")
    return roots[0]


def _validate_source_tree(root, expected_tag):
    pyproject = root / "pyproject.toml"
    if not pyproject.exists() or pyproject.stat().st_size > 256 * 1024:
        raise ValueError("Update archive is not a Nimo source tree")
    text = pyproject.read_text("utf-8")
    if not re.search(r'(?m)^name\s*=\s*["\']nimo-lite["\']\s*$', text):
        raise ValueError("Update package identity mismatch")
    version_match = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']\s*$', text)
    if not version_match:
        raise ValueError("Update package version is missing")
    if _semver(version_match.group(1)) != _semver(expected_tag):
        raise ValueError("Release tag and package version do not match")


def install_update():
    info = check_for_update()
    if not info["available"]:
        return {**info, "installed": False, "message": "Already up to date"}
    url = info.get("zipball_url")
    if not url:
        raise ValueError("Release does not provide a source archive")
    raw = _request(url)
    with tempfile.TemporaryDirectory(prefix="nimo-update-") as tmp:
        root = _safe_extract(raw, tmp)
        _validate_source_tree(root, info["latest"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", str(root)])
    return {**info, "installed": True, "message": "Update installed; restart Nimo to use it"}


def main():
    parser = argparse.ArgumentParser(description="Check/install official Nimo Lite releases")
    parser.add_argument("--check", action="store_true", help="Check only; do not install")
    args = parser.parse_args()
    print(json.dumps(check_for_update() if args.check else install_update(), indent=2))


if __name__ == "__main__":
    main()
