import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .hardware import hardware
from .security import MAX_UPLOAD, confined, safe_id, token, review
from .store import Store, uid, now

STATIC=Path(__file__).parent/'static'

class TextExample(BaseModel):
    question:str=Field(max_length=16384)
    answer:str=Field(max_length=16384)
    source:str=Field(min_length=1,max_length=1000)
    license:str=Field(min_length=1,max_length=500)
    category:str='qa'

class Knowledge(BaseModel):
    text:str=Field(min_length=1,max_length=32768)
    source:str=Field(min_length=1,max_length=1000)

class Train(BaseModel):
    task:str
    epochs:int=Field(default=5,ge=1,le=100)
    resume:str|None=None
    context:int=Field(default=128,ge=128,le=256)

class Chat(BaseModel):
    text:str=Field(min_length=1,max_length=4096)
    mode:str='knowledge'


def create_app(root=None):
    store=Store(root or os.environ.get('NIMO_DATA','data'))
    secret=token(store.root)
    processes={}; inference_lock=threading.Lock(); dataset_lock=threading.RLock()
    @asynccontextmanager
    async def lifespan(app):
        # Only one app instance may use a data directory (documented).
        with store.db() as db:
            db.execute("UPDATE jobs SET status='interrupted',command='stop' WHERE status IN ('running','queued','paused','stopping')")
        yield
        for proc in processes.values():
            if proc.poll() is None:
                with store.db() as db: db.execute("UPDATE jobs SET command='stop' WHERE status IN ('running','queued','paused')")
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.terminate()
    app=FastAPI(title='Nimo Lite',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.store=store; app.state.online=False

    @app.middleware('http')
    async def boundary(request, call_next):
        host=request.headers.get('host','').split(':')[0]
        if host not in ('localhost','127.0.0.1','testserver'):
            return JSONResponse({'detail':'Loopback access only; DNS rebinding blocked'},400)
        origin=request.headers.get('origin')
        if origin and (urlsplit(origin).netloc!=request.headers.get('host') or urlsplit(origin).scheme!='http'):
            return JSONResponse({'detail':'Cross-origin request denied'},403)
        if request.method not in ('GET','HEAD'):
            if not secrets.compare_digest(request.headers.get('x-nimo-token',''),secret):
                return JSONResponse({'detail':'Local session token required'},403)
            try: length=int(request.headers.get('content-length','0'))
            except ValueError: return JSONResponse({'detail':'Invalid content length'},400)
            if length>MAX_UPLOAD+65536:
                return JSONResponse({'detail':'Request exceeds 12 MiB limit'},413)
            # ASGI byte count also enforces the limit for chunked requests.
            received=0; old_receive=request._receive
            async def limited_receive():
                nonlocal received
                message=await old_receive()
                received+=len(message.get('body',b''))
                if received>MAX_UPLOAD+65536: raise HTTPException(413,'Request exceeds 12 MiB')
                return message
            request._receive=limited_receive
        try:
            response=await call_next(request)
        except ValueError as exc:
            response=JSONResponse({'detail':str(exc)},400)
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Cache-Control']='no-store'
        return response

    @app.exception_handler(ValueError)
    async def value_error(request,exc): return JSONResponse({'detail':str(exc)},400)

    def mutable():
        with store.db() as db:
            if db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running','paused','stopping')").fetchone():
                raise ValueError('Stop training before changing datasets; training uses an immutable snapshot')

    def model_row(ident=None,task=None):
        with store.db() as db:
            row=db.execute('SELECT * FROM models WHERE id=?',(safe_id(ident),)).fetchone() if ident else db.execute('SELECT * FROM models WHERE task=? AND active=1',(task,)).fetchone()
        if not row: raise ValueError('No matching model. Train and evaluate a model first')
        return dict(row)

    @app.get('/')
    def index(): return FileResponse(STATIC/'index.html')
    app.mount('/static',StaticFiles(directory=STATIC),name='static')

    @app.get('/api/bootstrap')
    def bootstrap(): return dict(token=secret,name='Nimo Lite',version='0.2.0',offline=not app.state.online)

    @app.get('/api/status')
    def status():
        for ident,proc in list(processes.items()):
            if proc.poll() is not None:
                with store.db() as db:
                    db.execute("UPDATE jobs SET status='failed',progress=? WHERE id=? AND status IN ('running','queued','paused')",
                               (json.dumps({'error':'Training worker exited unexpectedly; resume last saved checkpoint'}),ident))
                del processes[ident]
        with store.db() as db:
            jobs=[dict(r,config=json.loads(r['config']),progress=json.loads(r['progress'])) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 20')]
            models=[dict(r,metrics=json.loads(r['metrics'])) for r in db.execute('SELECT * FROM models ORDER BY created DESC')]
            size=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        return dict(hardware=hardware(store.root),jobs=jobs,models=models,samples=size,
                    dataset_bytes=sum(p.stat().st_size for p in (store.root/'originals').glob('*')),online=app.state.online)

    @app.get('/api/samples')
    def samples(): return store.samples()

    @app.post('/api/examples')
    def example(item:TextExample):
        with dataset_lock:
            mutable(); return store.add_text(**item.model_dump())

    @app.post('/api/import')
    async def import_json(file:UploadFile=File(...)):
        raw=await file.read(MAX_UPLOAD+1)
        if len(raw)>MAX_UPLOAD: raise ValueError('Dataset too large')
        try: items=json.loads(raw)
        except (ValueError,UnicodeError) as exc: raise ValueError('Expected UTF-8 JSON array') from exc
        if not isinstance(items,list) or len(items)>5000: raise ValueError('Expected up to 5000 examples')
        # Validate entire schema before importing; no archive extraction, pickle, or remote URLs.
        parsed=[TextExample.model_validate(x) for x in items]
        with dataset_lock:
            mutable(); results=[store.add_text(**x.model_dump()) for x in parsed]
        return dict(imported=len(results),results=results)

    @app.get('/api/starter')
    def starter_info():
        items=json.loads((Path(__file__).parent/'resources'/'assistant-basics.json').read_text(encoding='utf-8'))
        return dict(name='Assistant basics v1',examples=len(items),license='CC0-1.0',
                    source='Original Nimo Lite contributor examples; no scraped or private data',
                    note='Small educational dataset; does not establish general assistant capability')

    @app.post('/api/starter')
    def load_starter():
        items=json.loads((Path(__file__).parent/'resources'/'assistant-basics.json').read_text(encoding='utf-8'))
        with dataset_lock:
            mutable()
            results=[store.add_text(**item) for item in items]
        return dict(added=sum(not r['duplicate'] for r in results),duplicates=sum(r['duplicate'] for r in results),
                    note='Examples are pending review. No weights changed.')

    @app.post('/api/review')
    def batch_review(body:dict):
        ids=body.get('ids')
        if not isinstance(ids,list) or len(ids)>5000 or not ids or any(not isinstance(x,str) for x in ids):
            raise ValueError('Select 1–5000 sample IDs')
        if body.get('reviewed') is not True: raise ValueError('Confirm that you reviewed the selected examples')
        if body.get('status') not in ('approved','rejected'): raise ValueError('Choose approved or rejected')
        for ident in ids: safe_id(ident)
        changed=[]; errors=[]
        with dataset_lock:
            mutable()
            for ident in dict.fromkeys(ids):
                try:
                    store.update(ident,dict(status=body['status']))
                    changed.append(ident)
                except ValueError as exc:
                    errors.append(dict(id=ident,error=str(exc)))
        return dict(changed=len(changed),errors=errors,note='Flagged examples still require individual review')

    @app.post('/api/images')
    async def image(file:UploadFile=File(...),source:str=Form(...),license:str=Form(...)):
        raw=await file.read(MAX_UPLOAD+1)
        with dataset_lock:
            mutable(); return store.add_image(raw,source,license)

    @app.get('/api/images/{ident}')
    def preview(ident:str):
        safe_id(ident)
        row=next((r for r in store.samples('image') if r['id']==ident),None)
        if not row: raise HTTPException(404,'Image not found')
        return FileResponse(confined(store.root,row['content']['processed']),media_type='image/png')

    @app.patch('/api/samples/{ident}')
    def update(ident:str,changes:dict):
        with dataset_lock:
            mutable(); return store.update(ident,changes)

    @app.delete('/api/samples/{ident}')
    def delete(ident:str):
        with dataset_lock:
            mutable(); store.delete(ident); return dict(deleted=ident)

    @app.post('/api/categories')
    def category(body:dict):
        old,new=body.get('old'),body.get('new')
        if not isinstance(old,str) or not isinstance(new,str) or len(new)>100: raise ValueError('Supply old and new category; empty new deletes category')
        changed=0
        with dataset_lock:
            mutable()
            for row in store.samples('image'):
                c=row['content']
                if old in c['labels'] or any(b['label']==old for b in c['boxes']):
                    labels=[new if x==old else x for x in c['labels'] if x!=old or new]
                    boxes=[dict(b,label=new if b['label']==old else b['label']) for b in c['boxes'] if b['label']!=old or new]
                    store.update(row['id'],dict(labels=labels,boxes=boxes,status='pending'))
                    changed+=1
        return dict(changed=changed,approval='Affected annotations returned to pending review')

    @app.get('/api/validate/{task}')
    def validate(task:str): return store.validate(task)

    @app.post('/api/train')
    def train(body:Train):
        from .training import start_job
        with dataset_lock:
            ident=start_job(store,body.task,body.epochs,body.resume,body.context)
            log=open(store.root/'models'/f'{ident}.log','w')
            try:
                proc=subprocess.Popen([sys.executable,'-m','nimo.training',str(store.root),ident],stdout=log,stderr=log,
                    cwd=str(Path(__file__).parent.parent),env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
                processes[ident]=proc
            except OSError as exc:
                with store.db() as db: db.execute("UPDATE jobs SET status='failed' WHERE id=?",(ident,))
                raise ValueError('Cannot start training worker') from exc
            finally: log.close()
        return dict(id=ident)

    @app.post('/api/jobs/{ident}/{command}')
    def command(ident:str,command:str):
        safe_id(ident)
        if command not in ('pause','run','stop','save'): raise ValueError('Unknown job command')
        with store.db() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
            if not row or row['status'] not in ('queued','running','paused'): raise ValueError('Job is not running; use checkpoint resume for stopped jobs')
            if command=='save' and row['status']=='paused': raise ValueError('Paused checkpoint is already saved')
            db.execute('UPDATE jobs SET command=?,modified=? WHERE id=?',(command,now(),ident))
        return dict(command=command,note='Applied at next batch boundary')

    @app.post('/api/models/{ident}/activate')
    def activate(ident:str):
        row=model_row(ident)
        if not json.loads(row['metrics'])['accepted']: raise ValueError('Model failed acceptance criteria; cannot activate')
        with store.db() as db:
            db.execute('UPDATE models SET active=0 WHERE task=?',(row['task'],))
            db.execute('UPDATE models SET active=1 WHERE id=?',(ident,))
        return dict(active=ident)

    @app.post('/api/models/{ident}/evaluate')
    def reevaluate(ident:str):
        from .models import load_model
        from .training import evaluate
        with inference_lock:
            row=model_row(ident); model,ck=load_model(confined(store.root,row['path']))
            recorded=json.loads(row['metrics'])
            if recorded.get('bundled_evaluation_sha256'):
                import hashlib
                raw=(store.root/'models'/f'{ident}.eval.json').read_bytes()
                if hashlib.sha256(raw).hexdigest()!=recorded['bundled_evaluation_sha256']:
                    raise ValueError('Bundled evaluation file changed; reinstall the trusted model bundle')
                result=evaluate(model,json.loads(raw),row['task'],store,ck['classes'],objective=ck['config'].get('objective','full_sequence_v1'))
                return dict(split='bundled test',metrics=result,note='Immutable public evaluation samples; no training examples imported')
            current={r['id']:r for r in store.samples('text' if row['task']=='intent' else row['task'])}
            for sid,version,checksum in ck['snapshot']:
                if sid not in current or current[sid]['version']!=version or current[sid]['checksum']!=checksum:
                    raise ValueError('Dataset changed; historical metrics remain available but current reevaluation is blocked')
            result=evaluate(model,[current[i] for i in ck['split']['test']],row['task'],store,ck['classes'],objective=ck['config'].get('objective','full_sequence_v1'))
        return dict(split='test',metrics=result)

    @app.get('/api/models/{ident}/export')
    def model_export(ident:str):
        row=model_row(ident)
        return FileResponse(confined(store.root,row['path']),filename=f'nimo-{ident}.pt')

    @app.post('/api/classify/{ident}')
    def classify(ident:str):
        from .models import load_model
        from .training import image_tensor
        import torch
        safe_id(ident)
        sample=next((r for r in store.samples('image') if r['id']==ident),None)
        if not sample: raise ValueError('Image missing')
        with inference_lock:
            row=model_row(task='image'); model,ck=load_model(confined(store.root,row['path']))
            with torch.inference_mode(): scores=model(image_tensor(confined(store.root,sample['content']['processed'])).unsqueeze(0)).softmax(-1)[0].tolist()
        return dict(suggestions=sorted([dict(label=c,score=s,verified=False) for c,s in zip(ck['classes'],scores)],key=lambda x:x['score'],reverse=True),
                    model=row['id'],note='Uncalibrated local classifier suggestions; explicit human verification required')

    @app.post('/api/chat')
    def chat(body:Chat):
        started=time.monotonic()
        if body.mode=='knowledge':
            refs=store.retrieve(body.text)
            answer='\n\n'.join(r['text'] for r in refs) if refs else 'No matching local knowledge. Add a sourced note or train an experimental model.'
            return dict(answer=answer,mode='knowledge retrieval; weights unchanged',sources=refs,latency_s=time.monotonic()-started)
        if body.mode=='assistant':
            refs=store.retrieve(body.text)
            if refs:
                return dict(answer='\n\n'.join(r['text'] for r in refs),mode='Local knowledge retrieval; weights unchanged',
                            sources=refs,latency_s=time.monotonic()-started)
            try:
                row=model_row(task='intent')
            except ValueError:
                return dict(answer="I'm Nimo Lite, your local assistant workbench. Add sourced notes or train the starter intent model to enable basic assistant requests.",
                            mode='Built-in onboarding response; no model inference',sources=[],latency_s=time.monotonic()-started)
            from .models import load_model
            from .nlp import assistant_reply
            with inference_lock:
                model,ck=load_model(confined(store.root,row['path']))
                result=assistant_reply(model,ck,body.text)
            return dict(result,sources=[],model=row['id'],latency_s=time.monotonic()-started)
        if body.mode!='model': raise ValueError('Choose assistant, knowledge or model')
        from .models import load_model
        with inference_lock:
            row=model_row(task='text'); model,ck=load_model(confined(store.root,row['path']))
            answer=model.generate('User: '+body.text+'\nNimo: ')
        return dict(answer=answer or '[Model produced no text]',mode='Experimental original language model; output may be nonsensical',sources=[],latency_s=time.monotonic()-started)

    @app.post('/api/feedback')
    def feedback(body:dict):
        allowed=('Correct','Incorrect','Partially correct','Unhelpful','Unsafe','Needs a better explanation')
        if body.get('rating') not in allowed or len(json.dumps(body))>32768: raise ValueError('Invalid feedback')
        ident=uid()
        with store.db() as db:
            db.execute('INSERT INTO feedback VALUES(?,?,?)',(ident,json.dumps(dict(body,flags=review(json.dumps(body)))),now()))
        return dict(id=ident,note='Feedback saved locally. Add the correction as a pending training example and approve separately; weights unchanged.')

    @app.get('/api/knowledge')
    def knowledge():
        with store.db() as db: return [dict(r) for r in db.execute('SELECT * FROM knowledge ORDER BY created DESC')]
    @app.post('/api/knowledge')
    def remember(body:Knowledge): return dict(id=store.add_knowledge(body.text,body.source),mode='Knowledge saved; weights unchanged')
    @app.delete('/api/knowledge/{ident}')
    def forget(ident:str):
        with store.db() as db: db.execute('DELETE FROM knowledge WHERE id=?',(safe_id(ident),))
        return dict(deleted=ident)

    @app.post('/api/network')
    def network(body:dict):
        if not isinstance(body.get('online'),bool): raise ValueError('Supply online boolean')
        app.state.online=body['online']; return dict(online=app.state.online)

    @app.post('/api/search')
    def search(body:Chat):
        if not app.state.online: raise ValueError('Offline mode: enable online search first; query will be sent to Wikipedia')
        # Fixed HTTPS origin, no user-controlled URL, no redirects, no scripts or page instructions executed.
        try:
            with httpx.Client(timeout=10,follow_redirects=False,headers={'User-Agent':'NimoLite/0.1 (local educational prototype)'}) as client:
                response=client.get('https://en.wikipedia.org/w/api.php',params={'action':'query','list':'search','srsearch':body.text,'srlimit':5,'format':'json','srprop':''})
                response.raise_for_status(); payload=response.json()
        except (httpx.HTTPError,ValueError) as exc: raise ValueError('Search unavailable; local features remain available') from exc
        results=[dict(title=r['title'],url=f"https://en.wikipedia.org/?curid={int(r['pageid'])}",retrieved=now(),trust='untrusted; verify claims independently') for r in payload.get('query',{}).get('search',[])]
        return dict(results=results,note='Search metadata only. No automatic memory or training import; single source, no cross-source verification.')

    @app.get('/api/export')
    def export():
        rows=store.samples(); output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json',json.dumps(dict(format='nimo-dataset-v1',created=now(),samples=rows),ensure_ascii=False,indent=2))
            z.writestr('text.json',json.dumps([dict(r['content'],source=r['source'],license=r['license']) for r in rows if r['kind']=='text'],ensure_ascii=False))
            categories=sorted({b['label'] for r in rows if r['kind']=='image' for b in r['content']['boxes']})
            images=[]; annotations=[]
            for i,r in enumerate([r for r in rows if r['kind']=='image'],1):
                c=r['content']
                with __import__('PIL.Image',fromlist=['Image']).open(confined(store.root,c['processed'])) as im: w,h=im.size
                images.append(dict(id=i,file_name=c['processed'],width=w,height=h))
                for b in c['boxes']:
                    annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=categories.index(b['label'])+1,
                        bbox=[b['x']*w,b['y']*h,b['w']*w,b['h']*h],area=b['w']*w*b['h']*h,iscrowd=0))
                for k in ('original','processed'): z.write(confined(store.root,c[k]),c[k])
            z.writestr('coco.json',json.dumps(dict(images=images,annotations=annotations,categories=[dict(id=i+1,name=c) for i,c in enumerate(categories)])))
        return Response(output.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename=nimo-dataset.zip'})

    @app.get('/api/audit')
    def audit():
        with store.db() as db: return [dict(r) for r in db.execute('SELECT * FROM events ORDER BY id DESC LIMIT 200')]

    @app.post('/api/video')
    async def video(file:UploadFile=File(...),source:str=Form(...),license:str=Form(...)):
        from .video import extract
        raw=await file.read(MAX_UPLOAD+1)
        if len(raw)>MAX_UPLOAD: raise ValueError('Video exceeds 12 MiB prototype limit')
        with dataset_lock:
            mutable(); return extract(store,raw,source,license)
    return app


def main():
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser(description='Nimo Lite local training center')
    parser.add_argument('--data',default='data'); parser.add_argument('--port',type=int,default=8080)
    args=parser.parse_args()
    # Intentionally no host override: no remote multiuser authentication implemented.
    uvicorn.run(create_app(args.data),host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__': main()
