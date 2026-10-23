"""Train only bundled original public examples, never personal local datasets.
Produces reproducible metadata and an explicit model bundle of accepted models.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from nimo.models import MODEL_PROFILES, load_model
from nimo.nlp import assistant_reply
from nimo.store import Store
from nimo.training import start_job,run_job
from nimo.hardware import hardware,accelerator


def _load_sources():
    paths=[Path('nimo/resources/assistant-basics.json'),Path('nimo/resources/general-basics.json')]
    rows=[]
    hashes={}
    for path in paths:
        raw=path.read_bytes()
        hashes[path.name]=hashlib.sha256(raw).hexdigest()
        rows.extend(json.loads(raw))
    return rows,hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='artifacts/starter-training')
    parser.add_argument('--text-epochs',type=int,default=80)
    parser.add_argument('--intent-epochs',type=int,default=60)
    parser.add_argument('--profile',choices=sorted(MODEL_PROFILES),default='lite')
    parser.add_argument('--context',type=int,choices=(128,256,512),default=256)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if (root/'nimo.sqlite').exists():raise ValueError('Choose a fresh output directory for an independent experiment')
    store=Store(root)
    data,source_hashes=_load_sources()
    for item in data:
        imported=store.add_text(**item)
        store.update(imported['id'],dict(status='approved'))
    runs=[];start=time.monotonic()
    for task,epochs in [('intent',args.intent_epochs),('text',args.text_epochs)]:
        ident=start_job(store,task,epochs,context=args.context,profile=args.profile)
        print(f'Training {task}: {epochs} epochs, job {ident}',flush=True)
        run_job(root,ident)
        with store.db() as db:
            row=dict(db.execute('SELECT * FROM models WHERE id=?',(ident,)).fetchone())
        row['metrics']=json.loads(row['metrics'])
        row['sha256']=hashlib.sha256((root/row['path']).read_bytes()).hexdigest()
        runs.append(row)
        print(json.dumps(dict(task=task,validation=row['metrics']['validation'],test=row['metrics']['test'],accepted=row['metrics']['accepted'])),flush=True)
    probes={}
    for row in runs:
        model,ck=load_model(root/row['path'])
        questions=[
            'Hello','Good morning','Who are you?','What is machine learning?',
            'What is 25 percent of 200?','What is an operating system?',
            'How should I debug code?','What should an AI do when it is uncertain?',
            'What should an AI do when a question asks for today\'s information?'
        ]
        if row['task']=='intent':
            probes['intent']=[dict(question=q,**assistant_reply(model,ck,q)) for q in questions]
        else:
            probes['text']=[dict(question=q,answer=model.generate('User: '+q+'\nNimo: ',limit=192,temperature=0.2)) for q in questions]
    report=dict(version='0.2.0',dataset='assistant-basics-v1 + general-basics-v1',examples=len(data),license='CC0-1.0',
        source_sha256=source_hashes,profile=args.profile,context=args.context,hardware=hardware(root),accelerator=accelerator(),
        torch=torch.__version__,elapsed_s=time.monotonic()-start,models=runs,probes=probes,
        limitations='Small original educational corpus. Internal holdouts are not an external capability benchmark. Current facts still require web/tool verification.')
    Path('docs/starter-training-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    from package_starter import package
    package(root,report)

if __name__=='__main__':main()
