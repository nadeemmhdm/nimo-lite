"""Explicit local installation of bounded Nimo model bundles. Never fetches URLs."""
from __future__ import annotations
import argparse,hashlib,json,shutil,tempfile,zipfile
from pathlib import Path
from .models import load_model
from .store import Store,now,uid
from .tokenizer import NimoTokenizer
MAX_BUNDLE=256*1024*1024;MAX_MEMBERS=32

def _json(raw,what):
    if len(raw)>16*1024*1024:raise ValueError(f"{what} is too large")
    try:return json.loads(raw)
    except (ValueError,UnicodeError) as exc:raise ValueError(f"Invalid {what}") from exc

def _validate_public_row(row):
    if not isinstance(row,dict):raise ValueError("Invalid public evaluation example")
    c=row.get("content",{})
    if not isinstance(c.get("question"),str) or not isinstance(c.get("answer"),str) or len(c["question"]+c["answer"])>32768:raise ValueError("Invalid public evaluation example")
    return row

def install(bundle,root):
    bundle=Path(bundle)
    if not bundle.exists() or bundle.stat().st_size>MAX_BUNDLE:raise ValueError("Model bundle exceeds safe size")
    store=Store(root);validated=[]
    with zipfile.ZipFile(bundle) as z,tempfile.TemporaryDirectory() as tmp:
        infos=z.infolist()
        if len(infos)>MAX_MEMBERS or sum(i.file_size for i in infos)>MAX_BUNDLE:raise ValueError("Expanded bundle exceeds safe limits")
        names=[i.filename for i in infos]
        if len(names)!=len(set(names)):raise ValueError("Duplicate archive members")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):raise ValueError("Unsafe archive member")
        manifest=_json(z.read("manifest.json"),"bundle manifest")
        if manifest.get("format")!="nimo-model-bundle-v1" or not 1<=len(manifest.get("models",[]))<=3:raise ValueError("Unsupported model bundle manifest")
        evaluation_raw=z.read("evaluation.json")
        if hashlib.sha256(evaluation_raw).hexdigest()!=manifest.get("evaluation_sha256"):raise ValueError("Evaluation checksum mismatch")
        evaluation=_json(evaluation_raw,"evaluation data");legacy_public={r["id"]:r for r in evaluation.get("samples",[]) if isinstance(r,dict) and isinstance(r.get("id"),str)};native_public=[_validate_public_row(r) for r in evaluation.get("native_instruction",[])]
        for entry in manifest["models"]:
            name=entry.get("file");task=entry.get("task")
            if name not in ("nimo-text.pt","nimo-intent.pt","nimo-image.pt") or name!=f"nimo-{task}.pt":raise ValueError("Unsupported model filename")
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=entry.get("sha256"):raise ValueError("Checkpoint checksum mismatch")
            path=Path(tmp)/name;path.write_bytes(raw);model,ck=load_model(path)
            if ck.get("task")!=task:raise ValueError("Checkpoint task mismatch")
            native=task=="text" and isinstance(ck.get("tokenizer"),dict)
            if native:
                tok=NimoTokenizer.from_artifact(ck["tokenizer"])
                if ck.get("tokenizer_sha256")!=tok.artifact()["sha256"]:raise ValueError("Tokenizer checksum mismatch")
                rows=native_public
            else:
                classes=ck.get("classes",[])
                if task!="text" and len(classes)!=model.config.get("classes"):raise ValueError("Checkpoint labels mismatch")
                rows=[]
                for ident in ck.get("split",{}).get("test",[]):
                    if ident not in legacy_public:raise ValueError("Bundle evaluation sample missing")
                    row=_validate_public_row(legacy_public[ident]);c=row["content"];digest=hashlib.sha256((c["question"].strip()+"\0"+c["answer"].strip()).encode()).hexdigest()
                    if digest!=row.get("checksum"):raise ValueError("Evaluation sample checksum mismatch")
                    rows.append(row)
            if not rows:raise ValueError("Bundle needs held-out evaluation samples")
            metrics=entry.get("metrics",{})
            if metrics.get("accepted") is not True:raise ValueError("Only accepted models can be installed")
            validated.append((entry,path,json.dumps(rows,ensure_ascii=False).encode(),native))
        with store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE status IN ('running','queued','paused','stopping')").fetchone():raise ValueError("Stop training before installing models")
            installed=[]
            for entry,path,evaluation_bytes,native in validated:
                ident=uid();relative="models/"+ident+".pt";shutil.copyfile(path,store.root/relative);(store.root/"models"/f"{ident}.eval.json").write_bytes(evaluation_bytes)
                metrics=dict(entry["metrics"],bundled_evaluation_sha256=hashlib.sha256(evaluation_bytes).hexdigest(),provenance="Original bundled Nimo public fixtures; no personal data",native_tokenizer=native)
                db.execute("UPDATE models SET active=0 WHERE task=?",(entry["task"],));db.execute("INSERT INTO models VALUES(?,?,?,?,?,?)",(ident,entry["task"],relative,json.dumps(metrics),1,now()));store.event(db,"install_model",detail=entry["task"]+": "+entry["sha256"]);installed.append({"id":ident,"task":entry["task"],"native_tokenizer":native})
    return installed

def main():
    parser=argparse.ArgumentParser(description="Install a trusted local Nimo model bundle; no network access");parser.add_argument("bundle");parser.add_argument("--data",default="data");args=parser.parse_args();print(json.dumps({"installed":install(args.bundle,args.data)},indent=2))
if __name__=="__main__":main()
