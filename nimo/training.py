"""One child process per run; SQLite commands; atomic safe-loadable checkpoints."""
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from PIL import Image
from .hardware import hardware, accelerator
from .models import ByteTokenizer, NimoLM, NimoVision, load_model
from .security import confined, safe_id
from .store import Store, uid, now


def image_tensor(path):
    with Image.open(path) as im:
        arr = np.array(im.convert('RGB').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1)


def text_tensor(row, context):
    c = row['content']; tok = ByteTokenizer()
    ids = ([tok.bos]+tok.encode('User: '+c['question']+'\nNimo: '+c['answer'])+[tok.eos])[:context+1]
    return torch.tensor(ids[:-1]),torch.tensor(ids[1:])


def batch(rows, task, store, classes, context, device):
    if task=='image':
        x = torch.stack([image_tensor(confined(store.root,r['content']['processed'])) for r in rows])
        y = torch.tensor([classes.index(r['content']['labels'][0]) for r in rows])
    else:
        pairs = [text_tensor(r,context) for r in rows]
        x = nn.utils.rnn.pad_sequence([p[0] for p in pairs],batch_first=True,padding_value=256)
        y = nn.utils.rnn.pad_sequence([p[1] for p in pairs],batch_first=True,padding_value=-100)
    return x.to(device),y.to(device)


def evaluate(model, rows, task, store, classes, device='cpu', batch_size=2):
    model.eval(); loss_sum,count,correct = 0.,0,0
    matrix = np.zeros((len(classes),len(classes)),dtype=np.int64) if task=='image' else None
    with torch.inference_mode():
        for i in range(0,len(rows),batch_size):
            x,y = batch(rows[i:i+batch_size],task,store,classes,getattr(model,'context',128),device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='sum')
            loss_sum += float(loss); count += int((y!=-100).sum())
            pred = logits.argmax(-1); correct += int(((pred==y)&(y!=-100)).sum())
            if matrix is not None:
                for a,b in zip(y.cpu().tolist(),pred.cpu().tolist()): matrix[a,b] += 1
    if not count: raise ValueError('No evaluation tokens/samples')
    result = dict(loss=loss_sum/count,accuracy=correct/count,count=count)
    if matrix is not None:
        p = np.diag(matrix)/np.maximum(matrix.sum(0),1); r = np.diag(matrix)/np.maximum(matrix.sum(1),1)
        result.update(precision_macro=float(p.mean()),recall_macro=float(r.mean()),
            f1_macro=float((2*p*r/np.maximum(p+r,1e-9)).mean()),confusion_matrix=matrix.tolist())
    else:
        result.update(perplexity=math.exp(min(20,result['loss'])),note='Byte prediction metrics do not establish reasoning or conversation ability')
    return result


def start_job(store, task, epochs=5, resume=None):
    if task not in ('text','image') or not isinstance(epochs,int) or not 1<=epochs<=100:
        raise ValueError('Use text or image, 1–100 epochs')
    report = store.validate(task)
    if not report['valid']: raise ValueError('; '.join(report['errors']))
    hw = hardware(store.root)
    if hw['available_gb']<1.5 or hw['disk_free_gb']<.2:
        raise ValueError('Training needs 1.5 GiB available RAM and 0.2 GiB free storage')
    if len(store.samples())>5000: raise ValueError('Prototype limit: 5000 samples')
    if resume:
        safe_id(resume)
        if not (store.root/'models'/f'{resume}.pt').exists(): raise ValueError('Resume checkpoint missing')
    ident = uid()
    config = dict(epochs=epochs,batch_size=hw['batch_size'],seed=42,learning_rate=.003,
        context=128,threads=hw['threads'],resume=resume,split=report['splits'])
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running','paused','stopping')").fetchone():
            raise ValueError('Only one training job may run')
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',
            (ident,task,'queued','run',json.dumps(config),'{}',now(),now()))
    return ident


def run_job(root, ident):
    store = Store(root); safe_id(ident)
    with store.db() as db: job = dict(db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone())
    config = json.loads(job['config']); task = job['task']
    torch.set_num_threads(config['threads'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if hardware(root)['available_gb']<1: raise ValueError('Insufficient free RAM')
    device = accelerator()['device']
    if device=='cuda' and accelerator()['free_vram_gb']<1: device='cpu'
    rows = {r['id']:r for r in store.samples(task)}
    splits = {name:[rows[i] for i in ids] for name,ids in config['split'].items()}
    classes = sorted({r['content']['labels'][0] for r in splits['train']}) if task=='image' else []
    model = NimoLM() if task=='text' else NimoVision(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    start_epoch,start_offset = 0,0
    snapshot = [[r['id'],r['version'],r['checksum']] for group in splits.values() for r in group]
    if config.get('resume'):
        model,ck = load_model(store.root/'models'/f"{config['resume']}.pt")
        if ck['task']!=task or ck['classes']!=classes or ck['snapshot']!=snapshot:
            raise ValueError('Resume requires unchanged dataset, labels, and splits')
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch,start_offset = ck['epoch'],ck.get('offset',0)
        torch.set_rng_state(ck['rng'])
        if start_epoch>=config['epochs']: raise ValueError('Resume total epochs must exceed checkpoint epoch')
    model.to(device)
    for state in optimizer.state.values():
        for key,val in state.items():
            if isinstance(val,torch.Tensor): state[key]=val.to(device)
    bs = config['batch_size']; started=time.monotonic()
    progress = dict(epoch=start_epoch,epochs=config['epochs'],batch=0,device=device,loss=None)
    path=store.root/'models'/f'{ident}.pt'
    def status(state, **kwargs):
        progress.update(kwargs)
        progress.update(duration_s=round(time.monotonic()-started,2),resources=hardware(root))
        with store.db() as db:
            db.execute('UPDATE jobs SET status=?,progress=?,modified=? WHERE id=?',(state,json.dumps(progress),now(),ident))
    def save(epoch,offset=0):
        payload=dict(task=task,architecture=model.config,model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,offset=offset,classes=classes,config=config,rng=torch.get_rng_state(),snapshot=snapshot,
            split=config['split'],metrics=progress.get('validation',{}),created=now())
        tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); os.replace(tmp,path)
    def control(epoch,offset):
        with store.db() as db: command=db.execute('SELECT command FROM jobs WHERE id=?',(ident,)).fetchone()[0]
        if command in ('pause','save','stop'): save(epoch,offset)
        if command=='save':
            with store.db() as db: db.execute("UPDATE jobs SET command='run' WHERE id=?",(ident,))
        if command=='stop':
            status('stopped',checkpoint=ident); return False
        if command=='pause':
            status('paused',checkpoint=ident)
            while True:
                time.sleep(.25)
                with store.db() as db: command=db.execute('SELECT command FROM jobs WHERE id=?',(ident,)).fetchone()[0]
                if command=='stop': status('stopped',checkpoint=ident); return False
                if command=='run': status('running'); break
        if hardware(root)['available_gb']<.5:
            save(epoch,offset); raise ValueError('Low RAM: checkpoint saved; stopped')
        return True
    status('running',parameters=sum(p.numel() for p in model.parameters()))
    baseline=evaluate(model,splits['validation'],task,store,classes,device,bs)
    with store.db() as db: active=db.execute('SELECT * FROM models WHERE active=1 AND task=?',(task,)).fetchone()
    previous=None
    if active:
        old,ck=load_model(confined(store.root,active['path']))
        if set(ck['split']['train']) & set(config['split']['validation']+config['split']['test']):
            raise ValueError('Prior model trained on current held-out samples; create clean evaluation data')
        if ck['classes']==classes: previous=evaluate(old,splits['validation'],task,store,classes,'cpu',bs)
        del old
    for epoch in range(start_epoch,config['epochs']):
        order=list(splits['train']); random.Random(42+epoch).shuffle(order)
        offset=start_offset if epoch==start_epoch else 0
        model.train()
        for i in range(offset,len(order),bs):
            if not control(epoch,i): return
            x,y=batch(order[i:i+bs],task,store,classes,128,device)
            optimizer.zero_grad(set_to_none=True); logits=model(x)
            loss=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            status('running',epoch=epoch+1,batch=i//bs+1,batches=math.ceil(len(order)/bs),loss=float(loss.detach()))
        val=evaluate(model,splits['validation'],task,store,classes,device,bs)
        progress['validation']=val; save(epoch+1)
    test=evaluate(model,splits['test'],task,store,classes,device,bs)
    val=progress['validation']; reference=previous or baseline
    accepted=val['loss']<reference['loss'] and (task!='image' or val['accuracy']>=max(.5,reference['accuracy']))
    metrics=dict(validation=val,test=test,baseline_validation=baseline,previous_validation=previous,accepted=accepted,
        acceptance='Validation loss improves; image accuracy >= max(0.5, reference). Test does not select versions.',
        duration_s=time.monotonic()-started,split=config['split'],dataset_versions={r['id']:r['version'] for r in rows.values()},seed=42,pretrained=False)
    with store.db() as db:
        if accepted: db.execute('UPDATE models SET active=0 WHERE task=?',(task,))
        db.execute('INSERT INTO models VALUES(?,?,?,?,?,?)',(ident,task,'models/'+ident+'.pt',json.dumps(metrics),int(accepted),now()))
    status('completed',metrics=metrics,checkpoint=ident)


def main():
    root,ident=sys.argv[1:3]
    try: run_job(root,ident)
    except Exception as exc:
        store=Store(root)
        with store.db() as db:
            db.execute('UPDATE jobs SET status=?,progress=?,modified=? WHERE id=?',('failed',json.dumps({'error':str(exc)}),now(),ident))
        raise

if __name__=='__main__': main()
