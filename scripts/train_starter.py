"""Train reproducible Nimo demonstration models only from bundled original CC0 data.

This script demonstrates the native pipeline: corpus -> tokenizer -> base pretraining ->
instruction tuning. It is intentionally a small educational run, not a claim that the
bundled examples are sufficient for a generally capable assistant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from nimo.corpus import CorpusStore
from nimo.hardware import accelerator, hardware
from nimo.lm_training import run_job as run_lm_job, start_job as start_lm_job, train_local_tokenizer
from nimo.nlp import assistant_reply
from nimo.models import load_model
from nimo.store import Store
from nimo.training import run_job as run_legacy_job, start_job as start_legacy_job


SOURCES = [
    Path("nimo/resources/assistant-basics.json"),
    Path("nimo/resources/general-basics.json"),
    Path("nimo/resources/world-history-basics.json"),
]


def _load_sources():
    rows=[]; hashes={}
    for path in SOURCES:
        raw=path.read_bytes(); hashes[path.name]=hashlib.sha256(raw).hexdigest(); rows.extend(json.loads(raw))
    return rows,hashes


def _model_row(store,ident):
    with store.db() as db: row=dict(db.execute("SELECT * FROM models WHERE id=?",(ident,)).fetchone())
    row["metrics"]=json.loads(row["metrics"])
    row["sha256"]=hashlib.sha256((store.root/row["path"]).read_bytes()).hexdigest()
    return row


def main():
    parser=argparse.ArgumentParser(description="Train the small bundled Nimo demonstration models")
    parser.add_argument("--root",default="artifacts/starter-training")
    parser.add_argument("--intent-epochs",type=int,default=60)
    parser.add_argument("--pretrain-epochs",type=int,default=3)
    parser.add_argument("--instruction-epochs",type=int,default=8)
    parser.add_argument("--vocab-size",type=int,default=2048)
    parser.add_argument("--profile",choices=("tiny","small","medium"),default="tiny")
    parser.add_argument("--context",type=int,choices=(256,512,1024,2048),default=512)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if (root/"nimo.sqlite").exists(): raise ValueError("Choose a fresh output directory for an independent experiment")
    store=Store(root); corpus=CorpusStore(store)
    data,source_hashes=_load_sources()

    for item in data:
        imported=store.add_text(**item); store.update(imported["id"],{"status":"approved"})
        question=item["question"]; answer=item["answer"]
        corpus.add(
            kind="instruction",
            text=f"User: {question}\nAssistant: {answer}",
            source=item["source"], license=item["license"], category=item.get("category","qa"), status="approved",
            metadata={"question":question,"answer":answer,"messages":[{"role":"user","content":question},{"role":"assistant","content":answer}]},
        )
        corpus.add(kind="raw",text=f"{question}\n{answer}",source=item["source"],license=item["license"],
                   category=item.get("category","qa"),status="approved")

    tokenizer=train_local_tokenizer(store,args.vocab_size)
    print(f"Tokenizer: {tokenizer['vocab_size']} tokens",flush=True)
    start=time.monotonic(); runs=[]

    intent=start_legacy_job(store,"intent",args.intent_epochs,context=128,profile="micro")
    print(f"Training optional intent router: job {intent}",flush=True); run_legacy_job(root,intent)
    runs.append(_model_row(store,intent))

    pretrain=start_lm_job(store,"pretrain",args.pretrain_epochs,None,args.profile,args.context,
                          3e-4,None,None,0.03,200,"auto",None)
    print(f"Pretraining native base model: job {pretrain}",flush=True); run_lm_job(root,pretrain)
    pretrain_row=_model_row(store,pretrain)

    instruction=start_lm_job(store,"instruction",args.instruction_epochs,None,args.profile,args.context,
                             1e-4,None,None,0.03,200,"auto",pretrain)
    print(f"Instruction tuning from base {pretrain}: job {instruction}",flush=True); run_lm_job(root,instruction)
    text_row=_model_row(store,instruction); runs.append(text_row)

    probes={}
    intent_model,intent_ck=load_model(root/runs[0]["path"])
    probes["intent_router"]=[dict(question=q,**assistant_reply(intent_model,intent_ck,q)) for q in ["Hello","Good morning","Who are you?"]]
    text_model,text_ck=load_model(root/text_row["path"])
    tok_artifact=text_ck.get("tokenizer")
    probes["generative_model"]={
        "architecture":text_ck.get("architecture"),
        "tokenizer_vocab":tok_artifact.get("vocab_size") if tok_artifact else None,
        "note":"Use the application chat runtime for structured multilingual generation probes.",
    }

    report={
        "version":"0.2.0",
        "dataset":" + ".join(path.stem for path in SOURCES),
        "examples":len(data),"license":"CC0-1.0","source_sha256":source_hashes,
        "tokenizer":tokenizer,"profile":args.profile,"context":args.context,"hardware":hardware(root),
        "accelerator":accelerator(),"torch":torch.__version__,"elapsed_s":time.monotonic()-start,
        "pretrain_model":pretrain_row,"models":runs,"probes":probes,
        "limitations":"Tiny original demonstration corpus only. It verifies the pipeline, not general conversational capability.",
    }
    Path("docs/starter-training-results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    from package_starter import package
    package(root,report)


if __name__=="__main__": main()
