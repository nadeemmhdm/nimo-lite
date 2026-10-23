"""Local, privacy-filtered user memory for Nimo Lite.

Memory is deliberately conservative. It stores only simple, low-risk preferences and
project/workflow facts inferred from explicit user statements. Secrets, contact data,
precise locations, and sensitive personal categories are never auto-saved.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .store import now, uid


SENSITIVE = re.compile(
    r"(?:password|passcode|api[ _-]?key|secret|token|private key|seed phrase|otp|cvv|card number|"
    r"bank account|account number|aadhaar|aadhar|pan number|upi pin|"
    r"\bsk-[A-Za-z0-9_-]+|\bghp_[A-Za-z0-9]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\+?\d[\d\s().-]{8,}\d|"
    r"diagnos|disease|medication|mental health|therapy|pregnan|sexual|orientation|"
    r"religion|muslim|christian|hindu|political|party|vote|criminal|union member|"
    r"latitude|longitude|street address|home address|exact address|house number)",
    re.I,
)

PATTERNS = (
    ("preferred_name", re.compile(r"\b(?:my name is|call me|you can call me)\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I)),
    ("preferred_name", re.compile(r"\b(?:ente peru|ente name)\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I)),
    ("preference", re.compile(r"\bI (?:prefer|like|love)\s+(.{2,160})", re.I)),
    ("preference", re.compile(r"\b(?:enikku|enike)\s+(.{2,120}?)\s+(?:ishtam|prefer)\b", re.I)),
    ("tool", re.compile(r"\bI (?:use|usually use|work with)\s+(.{2,160})", re.I)),
    ("tool", re.compile(r"\b(?:njan|jane)\s+(.{2,140}?)\s+(?:use cheyyunnu|use cyunnu|use akunnu|use aanu)\b", re.I)),
    ("project", re.compile(r"\b(?:my project is|I am working on|I'm working on|I am building|I'm building)\s+(.{2,180})", re.I)),
    ("project", re.compile(r"\b(?:ente project|project name)\s*(?:is|aanu|ane|:)\s*(.{2,180})", re.I)),
    ("goal", re.compile(r"\b(?:my goal is|I want to learn|I want to build)\s+(.{2,180})", re.I)),
    ("goal", re.compile(r"\b(?:enikku|enike)\s+(.{2,160}?)\s+(?:padikkanam|build cheyyanam|build cyanam|indakkanam)\b", re.I)),
)


def _clean(value: str) -> str:
    value = " ".join(value.strip().split())
    value = re.split(r"[\n\r]", value, maxsplit=1)[0]
    return value.strip(" .,!?:;")[:200]


def _safe(text: str) -> bool:
    if not text or SENSITIVE.search(text):
        return False
    if len(text) > 240:
        return False
    return True


def extract_memories(text: str) -> list[dict]:
    """Extract conservative memory candidates from one explicit user message."""
    if not isinstance(text, str) or not text.strip() or SENSITIVE.search(text):
        return []
    output = []
    seen = set()
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            value = _clean(match.group(1))
            if not _safe(value):
                continue
            key = (kind, value.casefold())
            if key in seen:
                continue
            seen.add(key)
            output.append({"kind": kind, "value": value})
            if len(output) >= 5:
                return output
    return output


@dataclass
class LocalMemory:
    store: object

    def __post_init__(self):
        with self.store.db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories(
                  id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  value TEXT NOT NULL,
                  fingerprint TEXT NOT NULL UNIQUE,
                  source TEXT NOT NULL,
                  created TEXT NOT NULL,
                  modified TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                """
            )

    def remember_from_message(self, text: str) -> list[dict]:
        saved = []
        for item in extract_memories(text):
            fingerprint = hashlib.sha256((item["kind"] + "\0" + item["value"].casefold()).encode()).hexdigest()
            with self.store.db() as db:
                row = db.execute("SELECT * FROM memories WHERE fingerprint=?", (fingerprint,)).fetchone()
                if row:
                    db.execute("UPDATE memories SET modified=? WHERE id=?", (now(), row["id"]))
                    saved.append(dict(row))
                    continue
                ident = uid()
                db.execute(
                    "INSERT INTO memories VALUES(?,?,?,?,?,?,?)",
                    (ident, item["kind"], item["value"], fingerprint, "auto:user-message", now(), now()),
                )
                saved.append({"id": ident, **item, "source": "auto:user-message"})
        return saved

    def list(self) -> list[dict]:
        with self.store.db() as db:
            return [dict(r) for r in db.execute("SELECT id,kind,value,source,created,modified FROM memories ORDER BY modified DESC")]

    def relevant(self, query: str, limit: int = 6) -> list[dict]:
        memories = self.list()
        words = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for item in memories:
            value_words = set(re.findall(r"\w+", item["value"].casefold()))
            score = len(words & value_words)
            if item["kind"] in {"preferred_name", "preference", "project", "goal"}:
                score += 1
            scored.append((score, item))
        scored.sort(key=lambda x: (x[0], x[1]["modified"]), reverse=True)
        return [item for score, item in scored[: max(1, min(limit, 12))] if score > 0]

    def delete(self, ident: str) -> bool:
        from .security import safe_id

        safe_id(ident)
        with self.store.db() as db:
            changed = db.execute("DELETE FROM memories WHERE id=?", (ident,)).rowcount
        return bool(changed)

    def clear(self) -> int:
        with self.store.db() as db:
            count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            db.execute("DELETE FROM memories")
        return int(count)
