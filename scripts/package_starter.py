"""Package only independently trained original public fixtures, not personal workspace data."""
import hashlib
import json
import zipfile
from pathlib import Path
from nimo.store import Store


def package(root,report):
    root=Path(root)
    source=Path('nimo/resources/assistant-basics.json')
    # Public evaluation samples remain separate from a recipient's training approval workflow.
    public_rows=Store(root).samples('text')
    evaluation=json.dumps(dict(samples=public_rows),ensure_ascii=False).encode()
    entries=[]
    for row in report['models']:
        if not row['metrics']['accepted']:continue
        raw=(root/row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Trained checkpoint changed')
        entries.append(dict(task=row['task'],file=f"nimo-{row['task']}.pt",sha256=row['sha256'],metrics=row['metrics']))
    manifest=dict(format='nimo-model-bundle-v1',version='0.2.0',models=entries,
        dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        evaluation_sha256=hashlib.sha256(evaluation).hexdigest())
    out=Path('artifacts/nimo-lite-starter-v0.2.0.zip')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
        for entry in entries:
            row=next(r for r in report['models'] if r['task']==entry['task']);z.write(root/row['path'],entry['file'])
        z.writestr('evaluation.json',evaluation)
        z.writestr('MODEL_CARD.md',Path('docs/STARTER_MODEL_CARD.md').read_text())
        z.writestr('assistant-basics.json',source.read_bytes())
        z.writestr('LICENSE.txt','Original authored starter examples and these experimental weights: CC0-1.0. Code: MIT. No capability warranty.\n')
    print(f'Bundle: {out}; accepted tasks: {[e["task"] for e in entries]}',flush=True)
    return out

if __name__=='__main__':
    package('artifacts/starter-training',json.loads(Path('docs/starter-training-results.json').read_text()))
