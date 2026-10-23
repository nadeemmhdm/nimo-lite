import hashlib
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from .security import MAX_UPLOAD, MAX_TEXT, review, confined, safe_id

Image.MAX_IMAGE_PIXELS = 16_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ('originals', 'processed', 'models', 'exports', 'video'):
            (self.root/name).mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY,kind TEXT NOT NULL,content TEXT NOT NULL,
              checksum TEXT NOT NULL UNIQUE,status TEXT NOT NULL,source TEXT NOT NULL,license TEXT NOT NULL,
              created TEXT NOT NULL,modified TEXT NOT NULL,version INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at TEXT NOT NULL,action TEXT NOT NULL,
              sample TEXT,detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS knowledge(id TEXT PRIMARY KEY,text TEXT NOT NULL,source TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,content TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,task TEXT NOT NULL,status TEXT NOT NULL,command TEXT NOT NULL,
              config TEXT NOT NULL,progress TEXT NOT NULL,created TEXT NOT NULL,modified TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,task TEXT NOT NULL,path TEXT NOT NULL,metrics TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 0,created TEXT NOT NULL);
            ''')

    def db(self):
        db = sqlite3.connect(self.root/'nimo.sqlite', timeout=20)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        return db

    def event(self, db, action, sample=None, detail=''):
        db.execute('INSERT INTO events(at,action,sample,detail) VALUES(?,?,?,?)', (now(), action, sample, detail))

    def samples(self, kind=None):
        with self.db() as db:
            rows = db.execute('SELECT * FROM samples' + (' WHERE kind=?' if kind else '') + ' ORDER BY created',
                              (kind,) if kind else ()).fetchall()
        return [dict(r, content=json.loads(r['content'])) for r in rows]

    def add_text(self, question, answer, source, license, category='qa'):
        if not all(isinstance(x, str) for x in (question, answer, source, license, category)):
            raise ValueError('Text fields must be strings')
        if not question.strip() or not answer.strip() or not source.strip() or not license.strip():
            raise ValueError('Question, answer, source, and usage/license declaration are required')
        if len(question+answer) > MAX_TEXT:
            raise ValueError('Example exceeds 32 KiB text limit')
        content = dict(question=question.strip(), answer=answer.strip(), category=category,
                       flags=review(question+'\n'+answer))
        digest = hashlib.sha256((content['question']+'\0'+content['answer']).encode()).hexdigest()
        return self._insert('text', content, digest, source, license)

    def _insert(self, kind, content, digest, source, license):
        ident = uid()
        with self.db() as db:
            existing = db.execute('SELECT id FROM samples WHERE checksum=?', (digest,)).fetchone()
            if existing:
                return dict(id=existing['id'], duplicate=True)
            if db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]>=5000:
                raise ValueError('Prototype limit: 5000 samples; export or delete older samples first')
            db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,1)',
                       (ident, kind, json.dumps(content), digest, 'pending', source, license, now(), now()))
            self.event(db, 'import', ident, kind)
        return dict(id=ident, duplicate=False)

    def add_image(self, raw, source, license):
        if not source.strip() or not license.strip():
            raise ValueError('Source and usage/license declaration required')
        if len(raw) > MAX_UPLOAD:
            raise ValueError('Image exceeds 12 MiB')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Only JPEG, PNG, WebP supported')
                if image.width*image.height > 16_000_000:
                    raise ValueError('Image exceeds 16 megapixels')
                image.load()
                image = ImageOps.exif_transpose(image).convert('RGB')
                digest = hashlib.sha256(image.size.__repr__().encode()+image.tobytes()).hexdigest()
                thumb = image.resize((8,8)).convert('L')
                vals = list(thumb.get_flattened_data() if hasattr(thumb,'get_flattened_data') else thumb.getdata())
                mean = sum(vals)/64
                perceptual = ''.join('1' if p >= mean else '0' for p in vals)
                content = dict(labels=[], boxes=[], notes='', annotation='unlabeled', phash=perceptual,
                               width=image.width, height=image.height, flags=[])
                # Content-addressed filenames; supplied names are never used as paths.
                content['original'] = 'originals/'+digest+'.bin'
                content['processed'] = 'processed/'+digest+'.png'
                result = self._insert('image', content, digest, source, license)
                if not result['duplicate']:
                    confined(self.root, content['original']).write_bytes(raw)
                    image.thumbnail((1024,1024))
                    image.save(confined(self.root, content['processed']), 'PNG')
                return result
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Corrupt, unsupported, or oversized image') from exc

    def update(self, ident, changes):
        safe_id(ident)
        with self.db() as db:
            row = db.execute('SELECT * FROM samples WHERE id=?', (ident,)).fetchone()
            if not row:
                raise ValueError('Sample not found')
            content = json.loads(row['content'])
            if row['kind'] == 'image':
                if 'labels' in changes:
                    labels = changes['labels']
                    if not isinstance(labels, list) or len(labels)>20 or any(not isinstance(x,str) or not x.strip() or len(x)>100 for x in labels):
                        raise ValueError('Use up to 20 nonempty labels (100 characters each)')
                    content['labels'] = sorted(set(x.strip() for x in labels))
                if 'boxes' in changes:
                    boxes = changes['boxes']
                    if not isinstance(boxes, list) or len(boxes)>200:
                        raise ValueError('At most 200 boxes')
                    for box in boxes:
                        coords = [box.get(k) for k in ('x','y','w','h')]
                        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1 for v in coords):
                            raise ValueError('Boxes use normalized x,y,w,h in [0,1]')
                        x,y,w,h = coords
                        if w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001 or not isinstance(box.get('label'),str) or not box['label'].strip():
                            raise ValueError('Invalid box bounds or label')
                    content['boxes'] = boxes
                if 'annotation' in changes:
                    if changes['annotation'] not in ('verified','unlabeled','ambiguous','skipped'):
                        raise ValueError('Unknown annotation state')
                    content['annotation'] = changes['annotation']
                if 'notes' in changes:
                    content['notes'] = str(changes['notes'])[:4096]
            status = changes.get('status', 'pending')
            if status not in ('pending','approved','rejected'):
                raise ValueError('Invalid approval state')
            flags = review(json.dumps(content, ensure_ascii=False))
            if status == 'approved' and flags and changes.get('acknowledge_flags') is not True:
                raise ValueError('Review flags and explicitly acknowledge authorized use before approval')
            if status == 'approved' and row['kind']=='image' and (not content['labels'] or content['annotation']!='verified'):
                raise ValueError('Verify labels before approval')
            db.execute('UPDATE samples SET content=?,status=?,modified=?,version=version+1 WHERE id=?',
                       (json.dumps(content), status, now(), ident))
            self.event(db, 'review', ident, status)
        return dict(id=ident, status=status)

    def delete(self, ident):
        safe_id(ident)
        with self.db() as db:
            row = db.execute('SELECT * FROM samples WHERE id=?', (ident,)).fetchone()
            if not row:
                raise ValueError('Sample not found')
            if row['kind']=='image':
                content = json.loads(row['content'])
                for key in ('original','processed'):
                    confined(self.root, content[key]).unlink(missing_ok=True)
            db.execute('DELETE FROM samples WHERE id=?', (ident,))
            self.event(db, 'delete', ident)

    def validate(self, task):
        from .datasets import split_samples
        if task not in ('text','image','intent'):
            raise ValueError('Choose text, image, or intent training')
        rows = [r for r in self.samples('text' if task=='intent' else task) if r['status']=='approved']
        if task=='intent':
            rows = [r for r in rows if r['content'].get('category','').startswith('intent:')]
        errors, warnings = [], []
        if task=='image':
            for r in rows:
                c = r['content']
                if len(c['labels']) != 1 or c['annotation']!='verified':
                    errors.append(f"{r['id']}: classifier requires one verified label; boxes are annotation/export only")
                if not confined(self.root, c['processed']).exists():
                    errors.append(f"{r['id']}: missing image")
            counts = {}
            for r in rows:
                for label in r['content']['labels']:
                    counts[label] = counts.get(label,0)+1
            if len(counts)<2:
                errors.append('Classification needs at least two categories')
            for i,a in enumerate(rows):
                for b in rows[:i]:
                    if (int(a['content']['phash'],2)^int(b['content']['phash'],2)).bit_count()<=4 and a['content']['labels']!=b['content']['labels']:
                        errors.append('Near-duplicate images have conflicting labels: '+a['id']+' / '+b['id'])
            if counts and min(counts.values())<10:
                warnings.append('Some categories have fewer than 10 samples; results will be unreliable')
            if counts and max(counts.values()) > 3*min(counts.values()):
                warnings.append('Class imbalance exceeds 3:1')
        else:
            questions = {}
            for r in rows:
                c = r['content']
                key = c['question'].casefold().strip()
                if key in questions and questions[key] != c['answer']:
                    errors.append('Contradictory answers for question: '+c['question'][:100])
                questions[key] = c['answer']
        if task=='intent':
            from .nlp import RESPONSES, intent_label
            labels = {intent_label(r) for r in rows}
            if len(labels)<2: errors.append('Intent training needs at least two intent categories')
            if labels-set(RESPONSES): errors.append('Supported assistant intents: '+', '.join(sorted(RESPONSES)))
        splits = split_samples(rows, task)
        if task=='intent':
            for label in {r['content']['category'] for r in rows}:
                if any(not any(r['content']['category']==label for r in group) for group in splits.values()):
                    errors.append('Each split needs independent examples of '+label)
        for name, group in splits.items():
            if not group:
                errors.append(f'Empty {name} split: add diverse independently sourced samples (at least 3 groups per class)')
        if task=='image' and rows:
            for label in {x for r in rows for x in r['content']['labels']}:
                if any(not any(label in r['content']['labels'] for r in group) for group in splits.values()):
                    errors.append('Every split needs independent examples of category '+label)
        return dict(valid=not errors, errors=sorted(set(errors)), warnings=warnings, samples=len(rows),
                    splits={k:[r['id'] for r in v] for k,v in splits.items()})

    def add_knowledge(self, text, source):
        if not isinstance(text,str) or not isinstance(source,str) or not text.strip() or not source.strip() or len(text)>MAX_TEXT:
            raise ValueError('Knowledge needs text (up to 32 KiB) and source')
        ident = uid()
        with self.db() as db:
            db.execute('INSERT INTO knowledge VALUES(?,?,?,?)', (ident, text, source, now()))
        return ident

    def retrieve(self, query):
        import re
        words = set(re.findall(r'\w+', query.casefold())) - {'a','an','the','is','are','was','were','what','who','how','do','does','can','could','please','you','your','my','i','me','to','of','in','and','it'}
        with self.db() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM knowledge')]
        scored = [(len(words & set(re.findall(r'\w+',r['text'].casefold()))),r) for r in rows]
        return [dict(r, trust='untrusted reference, never executable instructions') for score,r in sorted(scored,key=lambda x:x[0],reverse=True)[:5] if score]
