"""Package independently trained original public Nimo fixtures, never personal workspace data."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from nimo.corpus import CorpusStore
from nimo.store import Store

SOURCES=[
    Path("nimo/resources/assistant-basics.json"),
    Path("nimo/resources/general-basics.json"),
    Path("nimo/resources/world-history-basics.json"),
]


def _native_eval_rows(root):
    corpus=CorpusStore(Store(root));rows=[]
    for doc in corpus.list(status="approved",split="test",limit=2000):
        if doc.get("kind") not in {"instruction","conversation"}:continue
        messages=(doc.get("metadata") or {}).get("messages")
        if not messages:
            try:messages=json.loads(doc.get("text", ""))
            except Exception:messages=[]
        question=next((m.get("content") for m in messages if isinstance(m,dict) and m.get("role")=="user" and isinstance(m.get("content"),str)),None)
        answer=next((m.get("content") for m in messages if isinstance(m,dict) and m.get("role")=="assistant" and isinstance(m.get("content"),str)),None)
        if isinstance(question,str) and isinstance(answer,str):
            rows.append({"id":doc["id"],"kind":"text","content":{"question":question,"answer":answer,"category":doc.get("category","qa")},
                         "checksum":doc["checksum"],"source":doc["source"],"license":doc["license"]})
    return rows


def package(root,report):
    root=Path(root);source_bytes={path.name:path.read_bytes() for path in SOURCES}
    combined_hash=hashlib.sha256(b"\0".join(source_bytes[name] for name in sorted(source_bytes))).hexdigest()
    evaluation=json.dumps({"samples":Store(root).samples("text"),"native_instruction":_native_eval_rows(root)},ensure_ascii=False).encode()
    entries=[]
    for row in report["models"]:
        if not row["metrics"].get("accepted"):continue
        raw=(root/row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row["sha256"]:raise ValueError("Trained checkpoint changed")
        entries.append({"task":row["task"],"file":f"nimo-{row['task']}.pt","sha256":row["sha256"],"metrics":row["metrics"]})
    manifest={"format":"nimo-model-bundle-v1","version":"0.2.0","models":entries,"dataset_sha256":combined_hash,
              "dataset_files":{name:hashlib.sha256(raw).hexdigest() for name,raw in source_bytes.items()},
              "evaluation_sha256":hashlib.sha256(evaluation).hexdigest()}
    out=Path("artifacts/nimo-lite-starter-v0.2.0.zip");out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(manifest,indent=2))
        for entry in entries:
            row=next(r for r in report["models"] if r["task"]==entry["task"]);z.write(root/row["path"],entry["file"])
        z.writestr("evaluation.json",evaluation)
        card=Path("docs/MODEL_CARD.md") if Path("docs/MODEL_CARD.md").exists() else Path("docs/STARTER_MODEL_CARD.md")
        z.writestr("MODEL_CARD.md",card.read_text())
        for name,raw in source_bytes.items():z.writestr(name,raw)
        z.writestr("LICENSE.txt","Original authored starter examples and experimental weights: CC0-1.0. Code: MIT. No capability warranty.\n")
    print(f"Bundle: {out}; accepted tasks: {[e['task'] for e in entries]}",flush=True);return out


if __name__=="__main__":package("artifacts/starter-training",json.loads(Path("docs/starter-training-results.json").read_text()))
