"""Explicit local installation of a bounded Nimo model bundle. Never fetches URLs."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from .models import load_model
from .store import Store,uid,now

MAX_BUNDLE=30*1024*1024


def install(bundle,root):
    bundle=Path(bundle)
    if bundle.stat().st_size>MAX_BUNDLE: raise ValueError('Model bundle exceeds 30 MiB')
    store=Store(root)
    validated=[]
    with zipfile.ZipFile(bundle) as z,tempfile.TemporaryDirectory() as tmp:
        if sum(i.file_size for i in z.infolist())>MAX_BUNDLE: raise ValueError('Expanded bundle exceeds 30 MiB')
        if len(z.namelist())!=len(set(z.namelist())): raise ValueError('Duplicate archive members')
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('format')!='nimo-model-bundle-v1' or not 1<=len(manifest.get('models',[]))<=3:
            raise ValueError('Unsupported model bundle manifest')
        evaluation_raw=z.read('evaluation.json')
        if hashlib.sha256(evaluation_raw).hexdigest()!=manifest.get('evaluation_sha256'):
            raise ValueError('Evaluation checksum mismatch')
        public={r['id']:r for r in json.loads(evaluation_raw)['samples']}
        for entry in manifest['models']:
            name=entry['file']; task=entry['task']
            if name not in ('nimo-text.pt','nimo-intent.pt','nimo-image.pt') or name!=f'nimo-{task}.pt':
                raise ValueError('Unsupported model filename')
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']: raise ValueError('Checkpoint checksum mismatch')
            path=Path(tmp)/name;path.write_bytes(raw)
            model,ck=load_model(path)
            if ck['task']!=task or len(ck['classes'])!=(0 if task=='text' else model.config['classes']):
                raise ValueError('Checkpoint task or labels mismatch')
            metrics=entry['metrics']
            if metrics.get('accepted') is not True: raise ValueError('Only accepted models can be installed')
            rows=[]
            for ident in ck['split']['test']:
                row=public[ident]
                c=row['content']
                if not isinstance(c.get('question'),str) or not isinstance(c.get('answer'),str) or len(c['question']+c['answer'])>32768:
                    raise ValueError('Invalid public evaluation example')
                digest=hashlib.sha256((c['question'].strip()+'\0'+c['answer'].strip()).encode()).hexdigest()
                if digest!=row['checksum']:raise ValueError('Evaluation sample checksum mismatch')
                rows.append(row)
            if not rows:raise ValueError('Bundle needs held-out evaluation samples')
            validated.append((entry,path,json.dumps(rows,ensure_ascii=False).encode()))
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE status IN ('running','queued','paused','stopping')").fetchone():
                raise ValueError('Stop training before installing models')
            installed=[]
            for entry,path,evaluation in validated:
                ident=uid();relative='models/'+ident+'.pt'
                shutil.copyfile(path,store.root/relative)
                (store.root/'models'/f'{ident}.eval.json').write_bytes(evaluation)
                metrics=dict(entry['metrics'],bundled_evaluation_sha256=hashlib.sha256(evaluation).hexdigest(),
                             provenance='Original public assistant-basics-v1; no personal data')
                db.execute('UPDATE models SET active=0 WHERE task=?',(entry['task'],))
                db.execute('INSERT INTO models VALUES(?,?,?,?,?,?)',(ident,entry['task'],relative,json.dumps(metrics),1,now()))
                store.event(db,'install_model',detail=entry['task']+': '+entry['sha256'])
                installed.append(dict(id=ident,task=entry['task']))
    return installed


def main():
    parser=argparse.ArgumentParser(description='Install a trusted local Nimo model bundle; no network access')
    parser.add_argument('bundle');parser.add_argument('--data',default='data')
    args=parser.parse_args()
    print(json.dumps(dict(installed=install(args.bundle,args.data)),indent=2))

if __name__=='__main__':main()
