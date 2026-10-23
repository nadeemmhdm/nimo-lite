"""Local, privacy-filtered user memory for Nimo Lite.

Memory is deliberately conservative. It stores only simple, low-risk preferences and
project/workflow facts inferred from explicit user statements. Secrets, contact data,
precise locations, and sensitive personal categories are never auto-saved.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
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


def _query_terms(text: str) -> list[str]:
    return [value for value in re.findall(r"\w+", text.casefold(), flags=re.UNICODE) if len(value) > 1][:24]


def _fts_query(text: str) -> str:
    return " OR ".join('"' + value.replace('"', '""') + '"' for value in _query_terms(text))


@dataclass
class LocalMemory:
    store: object

    def __post_init__(self):
        self._fts = False
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
                CREATE INDEX IF NOT EXISTS idx_memories_modified ON memories(modified DESC);
                """
            )
            try:
                db.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                      id UNINDEXED,
                      kind UNINDEXED,
                      value,
                      tokenize='unicode61'
                    );
                    CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                      INSERT INTO memories_fts(id,kind,value) VALUES(new.id,new.kind,new.value);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                      DELETE FROM memories_fts WHERE id=old.id;
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF value,kind ON memories BEGIN
                      DELETE FROM memories_fts WHERE id=old.id;
                      INSERT INTO memories_fts(id,kind,value) VALUES(new.id,new.kind,new.value);
                    END;
                    """
                )
                source_count = int(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
                index_count = int(db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0])
                if source_count != index_count:
                    db.execute("DELETE FROM memories_fts")
                    db.execute("INSERT INTO memories_fts(id,kind,value) SELECT id,kind,value FROM memories")
                self._fts = True
            except sqlite3.OperationalError:
                self._fts = False

    def remember_from_message(self, text: str) -> list[dict]:
        saved = []
        for item in extract_memories(text):
            fingerprint = hashlib.sha256(
                (item["kind"] + "\0" + item["value"].casefold()).encode()
            ).hexdigest()
            with self.store.db() as db:
                row = db.execute(
                    "SELECT * FROM memories WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
                if row:
                    db.execute(
                        "UPDATE memories SET modified=? WHERE id=?", (now(), row["id"])
                    )
                    saved.append(dict(row))
                    continue
                ident = uid()
                db.execute(
                    "INSERT INTO memories VALUES(?,?,?,?,?,?,?)",
                    (
                        ident,
                        item["kind"],
                        item["value"],
                        fingerprint,
                        "auto:user-message",
                        now(),
                        now(),
                    ),
                )
                saved.append({"id": ident, **item, "source": "auto:user-message"})
        return saved

    def list(self) -> list[dict]:
        with self.store.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,kind,value,source,created,modified FROM memories ORDER BY modified DESC"
                )
            ]

    def _fallback_relevant(self, query: str, limit: int) -> list[dict]:
        # Bounded fallback preserves the old preference/project weighting without a
        # full-table scan when FTS5 is unavailable or finds no lexical match.
        with self.store.db() as db:
            memories = [
                dict(row)
                for row in db.execute(
                    "SELECT id,kind,value,source,created,modified FROM memories ORDER BY modified DESC LIMIT 64"
                )
            ]
        words = set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))
        scored = []
        for item in memories:
            value_words = set(re.findall(r"\w+", item["value"].casefold(), flags=re.UNICODE))
            score = len(words & value_words)
            if item["kind"] in {"preferred_name", "preference", "project", "goal"}:
                score += 1
            scored.append((score, item))
        scored.sort(key=lambda value: (value[0], value[1]["modified"]), reverse=True)
        return [item for score, item in scored[:limit] if score > 0]

    def relevant(self, query: str, limit: int = 6) -> list[dict]:
        limit = max(1, min(int(limit), 12))
        match = _fts_query(query)
        if self._fts and match:
            try:
                with self.store.db() as db:
                    rows = [
                        dict(row)
                        for row in db.execute(
                            """
                            SELECT m.id,m.kind,m.value,m.source,m.created,m.modified,
                                   -bm25(memories_fts) AS relevance
                            FROM memories_fts
                            JOIN memories AS m ON m.id=memories_fts.id
                            WHERE memories_fts MATCH ?
                            ORDER BY bm25(memories_fts), m.modified DESC
                            LIMIT ?
                            """,
                            (match, limit),
                        )
                    ]
                if rows:
                    return rows
            except sqlite3.OperationalError:
                self._fts = False
        return self._fallback_relevant(query, limit)

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
