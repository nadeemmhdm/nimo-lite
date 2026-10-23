"""Native Nimo language-model tokenizer, pretraining and instruction-tuning engine."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .corpus import CorpusStore
from .hardware import accelerator, hardware
from .models import build_text_model, load_model
from .store import Store, now, uid
from .security import safe_id
from .tokenizer import NimoTokenizer, train_tokenizer


def tokenizer_path(store):
    return store.root/"tokenizer"/"tokenizer.json"


def tokenizer_status(store):
    path=tokenizer_path(store)
    if not path.exists():return {"ready":False}
    tok=NimoTokenizer.load(path)
    return {"ready":True,"path":"tokenizer/tokenizer.json","vocab_size":tok.vocab_size,"metadata":tok.metadata,
            "sha256":tok.artifact()["sha256"]}


def train_local_tokenizer(store,vocab_size=8000):
    corpus=CorpusStore(store)
    texts=[r["text"] for r in corpus.list(status="approved",limit=200000)]
    for row in store.samples("text"):
        if row["status"]=="approved":
            c=row["content"];texts.extend([c["question"],c["answer"]])
    if not texts:raise ValueError("Approve text/corpus data before training a tokenizer")
    tok=train_tokenizer(texts,vocab_size=vocab_size)
    tok.save(tokenizer_path(store))
    return tokenizer_status(store)


def _dataset_hash(rows):
    h=hashlib.sha256()
    for row in rows:
        h.update(str(row.get("checksum","")).encode());h.update(b"\0")
        h.update(str(row.get("version",1)).encode());h.update(b"\n")
    return h.hexdigest()


def _pretrain_sequences(rows,tok,context):
    sequences=[]
    for row in rows:
        ids=tok.encode(row["text"],add_bos=True,add_eos=True)
        stride=context
        for i in range(0,max(1,len(ids)-1),stride):
            chunk=ids[i:i+context+1]
            if len(chunk)>=3:sequences.append((chunk[:-1],chunk[1:]))
    return sequences


def _sft_pair(tok,messages,context):
    ids=[tok.special_id("BOS")];labels=[-100]
    for message in messages:
        role=str(message.get("role","")).lower()
        role_name={"system":"SYSTEM","user":"USER","assistant":"ASSISTANT","tool":"TOOL"}.get(role)
        if not role_name:continue
        role_id=tok.special_id(role_name);content=tok.encode(str(message.get("content","")))
        segment=[role_id]+content+[tok.special_id("EOS")];ids.extend(segment)
        if role=="assistant":labels.extend([-100]+content+[tok.special_id("EOS")])
        else:labels.extend([-100]*len(segment))
    ids=ids[:context+1];labels=labels[:context+1]
    if len(ids)<2:return None
    x=ids[:-1];y=labels[1:]
    if not any(v!=-100 for v in y):return None
    return x,y


def _instruction_rows(store,split):
    from .datasets import split_samples
    rows=[r for r in store.samples("text") if r["status"]=="approved" and not r["content"].get("category","").startswith("intent:")]
    grouped=split_samples(rows,"text");out=[]
    for row in grouped.get(split,[]):
        c=row["content"];out.append({"checksum":row["checksum"],"version":row["version"],
            "messages":[{"role":"user","content":c["question"]},{"role":"assistant","content":c["answer"]}]})
    corpus=CorpusStore(store)
    for row in corpus.training_rows(["instruction","conversation"],split):
        messages=row["metadata"].get("messages")
        if not messages:
            try:messages=json.loads(row["text"])
            except Exception:messages=None
        if messages:out.append({"checksum":row["checksum"],"version":row["version"],"messages":messages})
    return out


def _build_sequences(store,task,tok,context,split):
    corpus=CorpusStore(store)
    if task=="pretrain":
        rows=corpus.training_rows(["raw","knowledge","code","math"],split)
        return _pretrain_sequences(rows,tok,context),rows
    rows=_instruction_rows(store,split)
    seq=[pair for row in rows if (pair:=_sft_pair(tok,row["messages"],context))]
    return seq,rows


def validate_lm_data(store,task,context=1024):
    if task not in {"pretrain","instruction"}:raise ValueError("Choose pretrain or instruction")
    tokinfo=tokenizer_status(store);errors=[];warnings=[]
    if not tokinfo["ready"]:errors.append("Train a local tokenizer first")
    counts={}
    if tokinfo["ready"]:
        tok=NimoTokenizer.load(tokenizer_path(store))
        for split in ("train","validation","test"):
            seq,_=_build_sequences(store,task,tok,context,split);counts[split]=len(seq)
            if not seq:errors.append(f"No {split} sequences for {task}")
        if counts.get("train",0)<20:warnings.append("Very small training set; useful language ability requires far more data")
    return {"valid":not errors,"errors":errors,"warnings":warnings,"sequences":counts,"tokenizer":tokinfo}


def _hardware_defaults(store):
    hw=hardware(store.root);acc=accelerator();ram=hw["ram_gb"]
    profile="tiny" if ram<12 else "small" if ram<24 else "medium";context=512 if ram<8 else 1024 if ram<24 else 2048
    if acc["device"]=="cpu" and profile=="medium":profile="small"
    return {"profile":profile,"context":context,"batch_size":1 if ram<12 else 2 if ram<24 else 4,
            "gradient_accumulation":8 if ram<12 else 4,"device":acc["device"]}


def estimate_memory(profile,vocab_size,context,batch_size=1):
    model=build_text_model(profile,context,vocab_size);params=sum(p.numel() for p in model.parameters())
    train_bytes=params*16 + int(batch_size)*int(context)*model.config["width"]*model.config["layers"]*10
    infer_bytes=params*4 + int(context)*model.config["width"]*model.config["layers"]*8
    return {"parameters":params,"training_gb":round(train_bytes/2**30,2),"inference_gb":round(infer_bytes/2**30,2)}


def start_job(store,task,epochs=1,resume=None,profile=None,context=None,learning_rate=3e-4,
              batch_size=None,gradient_accumulation=None,warmup_ratio=0.03,checkpoint_every=200,
              precision="auto",base_model=None):
    if task not in {"pretrain","instruction"}:raise ValueError("Choose pretrain or instruction")
    defaults=_hardware_defaults(store);profile=profile or defaults["profile"];context=int(context or defaults["context"])
    batch_size=int(batch_size or defaults["batch_size"]);gradient_accumulation=int(gradient_accumulation or defaults["gradient_accumulation"])
    if not 1<=int(epochs)<=100:raise ValueError("Epochs must be 1-100")
    if context not in (256,512,1024,2048,4096):raise ValueError("Unsupported context length")
    if not 1<=batch_size<=64 or not 1<=gradient_accumulation<=128:raise ValueError("Unsafe batch/accumulation setting")
    report=validate_lm_data(store,task,context)
    if not report["valid"]:raise ValueError("; ".join(report["errors"]))
    tok=NimoTokenizer.load(tokenizer_path(store));estimate=estimate_memory(profile,tok.vocab_size,context,batch_size);hw=hardware(store.root)
    if hw["available_gb"]<1.5:raise ValueError("At least 1.5 GiB free RAM is required")
    if estimate["training_gb"]>max(2.0,hw["ram_gb"]*1.5):raise ValueError("Selected model is too large for detected RAM; choose a smaller profile/context")
    if resume:
        safe_id(resume);path=store.root/"models"/f"{resume}.pt"
        if not path.exists():raise ValueError("Resume checkpoint missing")
    if base_model:
        safe_id(base_model);path=store.root/"models"/f"{base_model}.pt"
        if not path.exists():raise ValueError("Base model checkpoint missing")
    elif task=="instruction" and not resume:
        with store.db() as db:active=db.execute("SELECT id FROM models WHERE task='text' AND active=1 ORDER BY created DESC LIMIT 1").fetchone()
        base_model=active["id"] if active else None
    ident=uid();cfg={"epochs":int(epochs),"profile":profile,"context":context,"learning_rate":float(learning_rate),
         "batch_size":batch_size,"gradient_accumulation":gradient_accumulation,"warmup_ratio":float(warmup_ratio),
         "checkpoint_every":max(1,int(checkpoint_every)),"precision":precision,"resume":resume,"base_model":base_model,"seed":42,
         "tokenizer_sha256":tok.artifact()["sha256"],"estimate":estimate}
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running','paused','stopping')").fetchone():raise ValueError("Only one training job may run")
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",(ident,task,"queued","run",json.dumps(cfg),"{}",now(),now()))
    return ident


def _pad_batch(pairs,pad):
    xs=[torch.tensor(p[0],dtype=torch.long) for p in pairs];ys=[torch.tensor(p[1],dtype=torch.long) for p in pairs]
    return nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=pad),nn.utils.rnn.pad_sequence(ys,batch_first=True,padding_value=-100)


def _precision(device,mode):
    if mode=="fp32":return torch.float32
    if device=="cuda":
        if mode in {"auto","bf16"} and torch.cuda.is_bf16_supported():return torch.bfloat16
        if mode in {"auto","fp16"}:return torch.float16
    if device=="mps" and mode in {"auto","fp16"}:return torch.float16
    return torch.float32


def _autocast(device,dtype):return nullcontext() if dtype==torch.float32 else torch.autocast(device_type=device,dtype=dtype)


def evaluate(model,sequences,device,batch_size,pad,precision):
    if not sequences:raise ValueError("No evaluation sequences")
    model.eval();loss_sum=tokens=correct=0
    with torch.inference_mode():
        for i in range(0,len(sequences),batch_size):
            x,y=_pad_batch(sequences[i:i+batch_size],pad);x=x.to(device);y=y.to(device)
            with _autocast(device,precision):logits=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction="sum")
            mask=y!=-100;loss_sum+=float(loss);tokens+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
    avg=loss_sum/max(tokens,1)
    return {"loss":avg,"perplexity":math.exp(min(20,avg)),"next_token_accuracy":correct/max(tokens,1),"tokens":tokens}


F=torch.nn.functional


def run_job(root,ident):
    store=Store(root);CorpusStore(store)
    with store.db() as db:job=dict(db.execute("SELECT * FROM jobs WHERE id=?",(ident,)).fetchone())
    if not job:raise ValueError("Training job not found")
    task=job["task"];cfg=json.loads(job["config"]);tok=NimoTokenizer.load(tokenizer_path(store))
    if tok.artifact()["sha256"]!=cfg["tokenizer_sha256"]:raise ValueError("Tokenizer changed after job creation")
    base_checkpoint=None;base_model_obj=None
    if cfg.get("base_model") and not cfg.get("resume"):
        base_checkpoint=cfg["base_model"];base_model_obj,base_ck=load_model(store.root/"models"/f"{base_checkpoint}.pt")
        if base_ck.get("task")!="text" or base_ck.get("tokenizer_sha256")!=cfg["tokenizer_sha256"]:raise ValueError("Instruction base model/tokenizer mismatch")
        arch=base_ck.get("architecture",{});cfg["context"]=int(arch.get("context",cfg["context"]));cfg["profile"]=base_ck.get("config",{}).get("profile",cfg["profile"])
    train_rows,raw_train=_build_sequences(store,task,tok,cfg["context"],"train");val_rows,raw_val=_build_sequences(store,task,tok,cfg["context"],"validation");test_rows,raw_test=_build_sequences(store,task,tok,cfg["context"],"test")
    dataset_hashes={"train":_dataset_hash(raw_train),"validation":_dataset_hash(raw_val),"test":_dataset_hash(raw_test)}
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    acc=accelerator();device=acc["device"]
    if device=="cuda" and acc.get("free_vram_gb",0)<1:device="cpu"
    dtype=_precision(device,cfg["precision"]);model=base_model_obj or build_text_model(cfg["profile"],cfg["context"],tok.vocab_size,dropout=0.0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],betas=(0.9,0.95),weight_decay=0.1)
    total_updates=max(1,math.ceil(len(train_rows)/cfg["batch_size"]/cfg["gradient_accumulation"])*cfg["epochs"]);warmup=max(1,int(total_updates*cfg["warmup_ratio"]))
    def lr_lambda(step):
        if step<warmup:return (step+1)/warmup
        progress=(step-warmup)/max(1,total_updates-warmup);return max(0.1,0.5*(1+math.cos(math.pi*min(1.0,progress))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda);start_epoch=0;global_step=0;best_loss=float("inf");path=store.root/"models"/f"{ident}.pt"
    if cfg.get("resume"):
        model,ck=load_model(store.root/"models"/f"{cfg['resume']}.pt")
        if ck["task"]!="text" or ck.get("tokenizer_sha256")!=cfg["tokenizer_sha256"]:raise ValueError("Resume model/tokenizer mismatch")
        if ck.get("dataset_hashes")!=dataset_hashes:raise ValueError("Resume requires unchanged train/validation/test datasets")
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],betas=(0.9,0.95),weight_decay=0.1);optimizer.load_state_dict(ck["optimizer"]);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
        if ck.get("scheduler"):scheduler.load_state_dict(ck["scheduler"])
        start_epoch=int(ck.get("epoch",0));global_step=int(ck.get("step",0));best_loss=float(ck.get("best_validation_loss",float("inf")))
        if ck.get("rng") is not None:torch.set_rng_state(ck["rng"])
    model.to(device);started=time.monotonic();history=[];tokens_seen=0
    def set_status(status,**extra):
        progress={"epoch":extra.pop("epoch",start_epoch),"epochs":cfg["epochs"],"step":global_step,"updates":total_updates,"device":device,"precision":str(dtype).replace("torch.",""),"tokens_seen":tokens_seen,"tokens_per_second":round(tokens_seen/max(time.monotonic()-started,1e-6),1),"history":history[-120:],"resources":hardware(root),**extra}
        with store.db() as db:db.execute("UPDATE jobs SET status=?,progress=?,modified=? WHERE id=?",(status,json.dumps(progress),now(),ident))
    def save(epoch,metrics=None,best=False):
        payload={"task":"text","training_task":task,"architecture":model.config,"tokenizer":tok.artifact(),"tokenizer_sha256":cfg["tokenizer_sha256"],"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng":torch.get_rng_state(),"step":global_step,"epoch":epoch,"dataset_hashes":dataset_hashes,"config":cfg,"metrics":metrics or {},"best_validation_loss":best_loss,"metadata":{"pretrained":task=="pretrain" or bool(base_checkpoint),"random_initialization":not bool(base_checkpoint),"base_model":base_checkpoint,"best":bool(best),"created":now()}}
        tmp=path.with_suffix(".tmp");torch.save(payload,tmp);os.replace(tmp,path)
    def command(epoch):
        while True:
            with store.db() as db:cmd=db.execute("SELECT command FROM jobs WHERE id=?",(ident,)).fetchone()[0]
            if cmd=="save":
                save(epoch)
                with store.db() as db:db.execute("UPDATE jobs SET command='run' WHERE id=?",(ident,))
                return True
            if cmd=="stop":save(epoch);set_status("stopped",checkpoint=ident);return False
            if cmd=="pause":save(epoch);set_status("paused",checkpoint=ident);time.sleep(.25);continue
            return True
    baseline=evaluate(model,val_rows,device,cfg["batch_size"],tok.special_id("PAD"),dtype);set_status("running",baseline=baseline,parameters=sum(p.numel() for p in model.parameters()));model.train();optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,cfg["epochs"]):
        order=list(range(len(train_rows)));random.Random(cfg["seed"]+epoch).shuffle(order);accum=0;epoch_loss=0.0;epoch_batches=0
        for batch_index in range(0,len(order),cfg["batch_size"]):
            if not command(epoch):return
            pairs=[train_rows[j] for j in order[batch_index:batch_index+cfg["batch_size"]]];x,y=_pad_batch(pairs,tok.special_id("PAD"));x=x.to(device);y=y.to(device)
            with _autocast(device,dtype):logits=model(x);raw_loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100);loss=raw_loss/cfg["gradient_accumulation"]
            loss.backward();accum+=1;epoch_loss+=float(raw_loss.detach());epoch_batches+=1;tokens_seen+=int((y!=-100).sum())
            if accum>=cfg["gradient_accumulation"] or batch_index+cfg["batch_size"]>=len(order):
                nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step();global_step+=1;accum=0
                if global_step%10==0:history.append({"step":global_step,"train_loss":round(epoch_loss/max(epoch_batches,1),5),"lr":optimizer.param_groups[0]["lr"]})
                if global_step%cfg["checkpoint_every"]==0:save(epoch)
                set_status("running",epoch=epoch+1,loss=float(raw_loss.detach()),lr=optimizer.param_groups[0]["lr"])
        val=evaluate(model,val_rows,device,cfg["batch_size"],tok.special_id("PAD"),dtype);history.append({"step":global_step,"validation_loss":round(val["loss"],5)})
        if val["loss"]<best_loss:best_loss=val["loss"];save(epoch+1,{"validation":val},best=True)
        else:save(epoch+1,{"validation":val})
        model.train()
    test=evaluate(model,test_rows,device,cfg["batch_size"],tok.special_id("PAD"),dtype);val=evaluate(model,val_rows,device,cfg["batch_size"],tok.special_id("PAD"),dtype);accepted=val["loss"]<baseline["loss"]
    metrics={"validation":val,"test":test,"baseline_validation":baseline,"accepted":accepted,"parameter_count":sum(p.numel() for p in model.parameters()),"tokenizer_vocab":tok.vocab_size,"training_task":task,"dataset_hashes":dataset_hashes,"history":history,"base_model":base_checkpoint}
    save(cfg["epochs"],metrics,best=val["loss"]<=best_loss)
    with store.db() as db:
        if accepted:db.execute("UPDATE models SET active=0 WHERE task='text'")
        db.execute("INSERT OR REPLACE INTO models VALUES(?,?,?,?,?,?)",(ident,"text","models/"+ident+".pt",json.dumps(metrics),int(accepted),now()))
    set_status("completed",epoch=cfg["epochs"],metrics=metrics,checkpoint=ident)


def main():
    import sys
    root,ident=sys.argv[1:3]
    try:run_job(root,ident)
    except Exception as exc:
        store=Store(root)
        with store.db() as db:db.execute("UPDATE jobs SET status=?,progress=?,modified=? WHERE id=?",("failed",json.dumps({"error":str(exc)}),now(),ident))
        raise

if __name__=="__main__":main()
