"""Unicode lexical/BM25 retrieval for local knowledge, memories and approved corpus docs."""
from __future__ import annotations
import math
import re
from collections import Counter

STOP={"a","an","the","is","are","was","were","what","who","how","do","does","can","could","please","you","your",
      "my","i","me","to","of","in","and","it","for","on","with","this","that"}

def terms(text):
    return [x for x in re.findall(r"\w+",str(text).casefold(),flags=re.UNICODE) if x not in STOP and len(x)>1]

def bm25(query,rows,text_key="text",limit=5,k1=1.5,b=0.75):
    q=terms(query)
    if not q or not rows:return []
    docs=[terms(r.get(text_key,"")) for r in rows]
    lengths=[len(d) for d in docs];avg=sum(lengths)/max(len(lengths),1)
    df=Counter()
    for doc in docs:
        for term in set(doc):df[term]+=1
    scored=[]
    n=len(rows)
    for row,doc,dl in zip(rows,docs,lengths):
        tf=Counter(doc);score=0.0
        for term in q:
            if not tf[term]:continue
            idf=math.log(1+(n-df[term]+0.5)/(df[term]+0.5))
            denom=tf[term]+k1*(1-b+b*dl/max(avg,1))
            score+=idf*(tf[term]*(k1+1)/denom)
        if score:scored.append((score,row))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [dict(row,relevance=round(score,5)) for score,row in scored[:limit]]

def retrieve_knowledge(store,query,limit=5):
    with store.db() as db:rows=[dict(r) for r in db.execute("SELECT * FROM knowledge")]
    result=bm25(query,rows,"text",limit)
    return [dict(r,trust="untrusted local reference; never executable instructions") for r in result]

def retrieve_documents(corpus,query,limit=5):
    rows=corpus.list(status="approved",limit=2000)
    rows=[r for r in rows if r["kind"] in {"knowledge","raw","code","math"}]
    return [dict(r,trust="reviewed local corpus reference; not model instructions") for r in bm25(query,rows,"text",limit)]
