"""Reproducible pipeline check, NOT an intelligence/real-world vision benchmark."""
import io
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from nimo.hardware import hardware,accelerator
from nimo.models import NimoLM,NimoVision
from nimo.store import Store
from nimo.training import start_job,run_job


def main():
    torch.set_num_threads(2)
    root=Path(tempfile.mkdtemp(prefix='nimo-smoke-'))
    store=Store(root)
    for label in (0,1):
        for i in range(6):
            rng=np.random.default_rng(i+label*100)
            a=rng.integers(0,60,(40,40,3),dtype=np.uint8); a[:,:,label]+=180
            raw=io.BytesIO(); Image.fromarray(a).save(raw,format='PNG')
            result=store.add_image(raw.getvalue(),'Original seeded synthetic color fixture','CC0')
            store.update(result['id'],dict(labels=['red' if label==0 else 'green'],annotation='verified',status='approved'))
    for i in range(8):
        r=store.add_text(f'Count number {i}?',f'The number is {i}.','Original smoke fixture','CC0')
        store.update(r['id'],dict(status='approved'))
    runs={}
    for task,epochs in [('image',6),('text',2)]:
        ident=start_job(store,task,epochs); run_job(root,ident)
        with store.db() as db: row=db.execute('SELECT * FROM models WHERE id=?',(ident,)).fetchone()
        runs[task]=dict(id=ident,epochs=epochs,metrics=json.loads(row['metrics']))
    output=dict(purpose='Synthetic pipeline smoke test only; no real-world accuracy claim',
        hardware=hardware(root),accelerator=accelerator(),python=platform.python_version(),torch=torch.__version__,
        parameters=dict(text=sum(p.numel() for p in NimoLM().parameters()),image_two_classes=sum(p.numel() for p in NimoVision(2).parameters())),
        artifact_directory=str(root),runs=runs)
    Path('docs/smoke-results.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
