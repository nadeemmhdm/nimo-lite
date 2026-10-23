"""Deterministic leakage groups: normalized question or transitive image aHash distance <= 4.
Perceptual hashing is conservative and imperfect; audit splits before serious experiments.
"""
import hashlib
import re


def split_samples(rows, task):
    rows = sorted(rows, key=lambda r:r['checksum'])
    parent = list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i,a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if task=='image':
                pa,pb = a['content']['phash'],b['content']['phash']
                same = (int(pa,2)^int(pb,2)).bit_count()<=4
            else:
                norm = lambda r: re.sub(r'\W+','',r['content']['question'].casefold())
                same = norm(a)==norm(b)
            if same:
                parent[find(i)] = find(j)
    groups = {}
    for i,r in enumerate(rows):
        groups.setdefault(find(i),[]).append(r)
    buckets = {}
    for group in groups.values():
        labels = {r['content']['labels'][0] for r in group if r['content'].get('labels')}
        key = tuple(sorted(labels)) if task=='image' else ('text',)
        buckets.setdefault(key,[]).append(group)
    result = dict(train=[], validation=[], test=[])
    for key,groups in sorted(buckets.items()):
        groups.sort(key=lambda g:hashlib.sha256(g[0]['checksum'].encode()).hexdigest())
        for i,group in enumerate(groups):
            target = 'test' if i==0 else 'validation' if i==1 else 'train'
            result[target].extend(group)
    return result
