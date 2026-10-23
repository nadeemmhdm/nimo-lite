"""Local input boundaries and advisory content review."""
import re
import secrets
from pathlib import Path

MAX_UPLOAD = 64 * 1024 * 1024
MAX_TEXT = 32768


def review(text):
    flags=[]
    if re.search(r'ignore\s+(all\s+)?(previous|system)|system\s*prompt|exfiltrat|disable\s+(security|safety)',text,re.I):
        flags.append('Possible instruction injection; requires human review')
    if re.search(r'-----BEGIN .*PRIVATE KEY|\b(?:sk-|ghp_)[A-Za-z0-9]{16,}|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',text):
        flags.append('Possible sensitive data; remove or explicitly approve authorized use')
    return flags


def safe_id(value):
    if not re.fullmatch(r'[a-f0-9]{32}',value):raise ValueError('Invalid identifier')
    return value


def confined(root,relative):
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Path leaves local data directory')
    return path


def safe_archive_member(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or any(part in ('','.','~') for part in path.parts):
        raise ValueError('Unsafe archive path')
    return path


def token(root):
    path=Path(root)/'session.token'
    if not path.exists():
        path.write_text(secrets.token_urlsafe(32));path.chmod(0o600)
    return path.read_text().strip()
