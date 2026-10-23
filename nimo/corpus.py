"""Scalable local text/corpus dataset layer for Nimo pretraining and instruction tuning."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

from .store import now, uid

MAX_ITEM_TEXT = 2 * 1024 * 1024
MAX_IMPORT = 64 * 1024 * 1024
MAX_ARCHIVE_FILES = 2000
TEXT_EXTENSIONS = {".txt",".md",".markdown",".py",".js",".ts",".html",".css",".sql",".sh",".bash",".json",".jsonl"}
CODE_EXTENSIONS = {".py",".js",".ts",".html",".css",".sql",".sh",".bash",".json"}


def detect_language(text: str) -> str:
    mal = sum(1 for c in text if "\u0d00" <= c <= "\u0d7f")
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    if mal and mal >= max(3, latin // 2):
        return "ml"
    lower = text.casefold()
    manglish_hints = (" entha"," ninte"," ente "," aanu"," alle"," venam"," cheyy"," patt"," engane"," onnu")
    if latin > 10 and any(x in lower for x in manglish_hints):
        return "manglish"
    return "en" if latin else "und"


def simhash64(text: str) -> int:
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if not tokens:
        tokens = [text[:256]]
    vec = [0] * 64
    for token in tokens[:20000]:
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vec[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, score in enumerate(vec):
        if score >= 0: out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def split_for_checksum(checksum: str) -> str:
    # Stable 90/5/5 split; no later movement means test rows do not leak into training.
    value = int(checksum[:8], 16) % 100
    return "train" if value < 90 else "validation" if value < 95 else "test"


class CorpusStore:
    def __init__(self, store):
        self.store = store
        for name in ("raw","cleaned","tokenizer","pretrain","instruction","validation","test","chats"):
            (store.root/name).mkdir(parents=True, exist_ok=True)
        with store.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              text TEXT NOT NULL,
              checksum TEXT NOT NULL UNIQUE,
              simhash TEXT NOT NULL,
              source TEXT NOT NULL,
              license TEXT NOT NULL,
              language TEXT NOT NULL,
              category TEXT NOT NULL,
              status TEXT NOT NULL,
              split TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1,
              metadata TEXT NOT NULL,
              created TEXT NOT NULL,
              modified TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_kind_status ON documents(kind,status,split);
            CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);
            """)

    def count(self):
        with self.store.db() as db:
            return int(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def _limit(self):
        return int(os.environ.get("NIMO_MAX_DOCUMENTS","250000"))

    def add(self,text,source,license,kind="raw",language="auto",category="",status="pending",metadata=None):
        if kind not in {"raw","knowledge","code","math","instruction","conversation"}:
            raise ValueError("Unsupported document kind")
        if status not in {"pending","approved","rejected"}:
            raise ValueError("Invalid review status")
        if not all(isinstance(x,str) for x in (text,source,license,kind,language,category)):
            raise ValueError("Document fields must be text")
        text=text.replace("\r\n","\n").replace("\r","\n")
        if not text.strip() or not source.strip() or not license.strip():
            raise ValueError("Text, source and license are required")
        if len(text.encode("utf-8")) > MAX_ITEM_TEXT:
            raise ValueError("Single document exceeds 2 MiB")
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest()
        sh=simhash64(text)
        if language=="auto":language=detect_language(text)
        with self.store.db() as db:
            exact=db.execute("SELECT id FROM documents WHERE checksum=?",(checksum,)).fetchone()
            if exact:return {"id":exact["id"],"duplicate":"exact"}
            if self.count()>=self._limit():raise ValueError("Configured document limit reached")
            # Compare recent approved/pending documents of same kind for conservative near duplicates.
            rows=db.execute("SELECT id,simhash FROM documents WHERE kind=? ORDER BY created DESC LIMIT 10000",(kind,)).fetchall()
            for row in rows:
                if hamming(sh,int(row["simhash"],16))<=2:
                    return {"id":row["id"],"duplicate":"near"}
            ident=uid();split=split_for_checksum(checksum)
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ident,kind,text,checksum,f"{sh:016x}",source.strip(),license.strip(),language,category[:120],
                 status,split,1,json.dumps(metadata or {},ensure_ascii=False),now(),now()))
        return {"id":ident,"duplicate":False,"split":split,"language":language}

    def update(self,ident,**changes):
        allowed={"status","language","category","split","metadata"}
        if set(changes)-allowed:raise ValueError("Unsupported document update")
        with self.store.db() as db:
            row=db.execute("SELECT * FROM documents WHERE id=?",(ident,)).fetchone()
            if not row:raise ValueError("Document not found")
            status=changes.get("status",row["status"])
            split=changes.get("split",row["split"])
            if status not in {"pending","approved","rejected"}:raise ValueError("Invalid review status")
            if split not in {"train","validation","test"}:raise ValueError("Invalid split")
            metadata=changes.get("metadata",json.loads(row["metadata"]))
            db.execute("""UPDATE documents SET status=?,language=?,category=?,split=?,metadata=?,version=version+1,modified=? WHERE id=?""",
                (status,changes.get("language",row["language"]),changes.get("category",row["category"])[:120],
                 split,json.dumps(metadata,ensure_ascii=False),now(),ident))
        return {"id":ident,"status":status,"split":split}

    def delete(self,ident):
        with self.store.db() as db:
            if not db.execute("DELETE FROM documents WHERE id=?",(ident,)).rowcount:raise ValueError("Document not found")
        return ident

    def list(self,kind=None,status=None,split=None,limit=500,offset=0):
        clauses=[];params=[]
        for col,val in (("kind",kind),("status",status),("split",split)):
            if val:clauses.append(col+"=?");params.append(val)
        where=(" WHERE "+" AND ".join(clauses)) if clauses else ""
        sql="SELECT * FROM documents"+where+" ORDER BY created DESC LIMIT ? OFFSET ?";params += [min(int(limit),2000),max(int(offset),0)]
        with self.store.db() as db:rows=db.execute(sql,params).fetchall()
        return [dict(r,metadata=json.loads(r["metadata"])) for r in rows]

    def training_rows(self,kinds,split="train"):
        if isinstance(kinds,str):kinds=[kinds]
        if not kinds:return []
        marks=",".join("?"*len(kinds))
        with self.store.db() as db:
            rows=db.execute(f"SELECT * FROM documents WHERE status='approved' AND split=? AND kind IN ({marks}) ORDER BY checksum",
                            [split,*kinds]).fetchall()
        return [dict(r,metadata=json.loads(r["metadata"])) for r in rows]

    def stats(self):
        with self.store.db() as db:
            total=db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            approved=db.execute("SELECT COUNT(*) FROM documents WHERE status='approved'").fetchone()[0]
            by_kind={r[0]:r[1] for r in db.execute("SELECT kind,COUNT(*) FROM documents GROUP BY kind")}
            by_split={r[0]:r[1] for r in db.execute("SELECT split,COUNT(*) FROM documents WHERE status='approved' GROUP BY split")}
            chars=db.execute("SELECT COALESCE(SUM(LENGTH(text)),0) FROM documents WHERE status='approved'").fetchone()[0]
        return {"documents":total,"approved":approved,"by_kind":by_kind,"by_split":by_split,"characters":chars}

    def import_bytes(self,filename,raw,source,license,status="pending"):
        if len(raw)>MAX_IMPORT:raise ValueError("Import exceeds 64 MiB")
        suffix=Path(filename).suffix.casefold()
        if suffix==".zip":return self._import_zip(raw,source,license,status)
        return self._import_one(filename,raw,source,license,status)

    def _import_one(self,filename,raw,source,license,status):
        suffix=Path(filename).suffix.casefold()
        if suffix not in TEXT_EXTENSIONS:raise ValueError("Unsupported dataset file type")
        try:text=raw.decode("utf-8")
        except UnicodeDecodeError as exc:raise ValueError("Datasets must be UTF-8") from exc
        results=[]
        if suffix==".jsonl":
            for index,line in enumerate(text.splitlines()):
                if not line.strip():continue
                if index>100000:raise ValueError("JSONL row limit exceeded")
                results.extend(self._json_item(json.loads(line),f"{source}#{index+1}",license,status))
        elif suffix==".json":
            payload=json.loads(text)
            items=payload if isinstance(payload,list) else [payload]
            if len(items)>100000:raise ValueError("JSON item limit exceeded")
            for index,item in enumerate(items):results.extend(self._json_item(item,f"{source}#{index+1}",license,status))
        else:
            kind="code" if suffix in CODE_EXTENSIONS else "raw"
            results.append(self.add(text,source,license,kind=kind,status=status,metadata={"filename":Path(filename).name}))
        return results

    def _json_item(self,item,source,license,status):
        if not isinstance(item,dict):raise ValueError("JSON dataset entries must be objects")
        if isinstance(item.get("messages"),list):
            messages=[]
            for m in item["messages"][:100]:
                if isinstance(m,dict) and m.get("role") in {"system","user","assistant","tool"} and isinstance(m.get("content"),str):
                    messages.append({"role":m["role"],"content":m["content"]})
            if not messages:raise ValueError("Conversation has no valid messages")
            return [self.add(json.dumps(messages,ensure_ascii=False),source,item.get("license",license),kind="conversation",
                             language=item.get("language","auto"),category=item.get("category","chat"),status=status,
                             metadata={"messages":messages})]
        if isinstance(item.get("question"),str) and isinstance(item.get("answer"),str):
            messages=[{"role":"user","content":item["question"]},{"role":"assistant","content":item["answer"]}]
            return [self.add(json.dumps(messages,ensure_ascii=False),source,item.get("license",license),kind="instruction",
                             language=item.get("language","auto"),category=item.get("category","instruction"),status=status,
                             metadata={"messages":messages})]
        text=item.get("text") or item.get("content")
        if not isinstance(text,str):raise ValueError("JSON item needs text/content, question+answer, or messages")
        return [self.add(text,source,item.get("license",license),kind=item.get("kind","raw"),
                         language=item.get("language","auto"),category=item.get("category",""),status=status,
                         metadata={"original":{k:v for k,v in item.items() if k not in {"text","content"}}})]

    def _import_zip(self,raw,source,license,status):
        results=[];total=0
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos=archive.infolist()
            if len(infos)>MAX_ARCHIVE_FILES:raise ValueError("Archive has too many files")
            for info in infos:
                if info.is_dir():continue
                path=Path(info.filename)
                if path.is_absolute() or ".." in path.parts:raise ValueError("Archive traversal path blocked")
                if info.file_size>MAX_ITEM_TEXT:raise ValueError("Archive member exceeds 2 MiB")
                total+=info.file_size
                if total>MAX_IMPORT:raise ValueError("Expanded archive exceeds 64 MiB")
                if path.suffix.casefold() not in TEXT_EXTENSIONS:continue
                results.extend(self._import_one(path.name,archive.read(info),f"{source}:{path.as_posix()}",license,status))
        return results
