"""Original Nimo model architectures.

No pretrained weights, external model hub, or downloaded tokenizer is required. Legacy
v0.1/v0.2 checkpoints remain loadable; new language-model runs use the modern NimoLMV2.
"""
from __future__ import annotations

import codecs
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(min(4, torch.get_num_threads()))


class ByteTokenizer:
    """Legacy byte tokenizer retained only for old checkpoints."""
    vocab_size = 259
    pad, bos, eos = 256, 257, 258
    def encode(self, text): return list(text.encode("utf-8"))
    def decode(self, ids): return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class NimoLM(nn.Module):
    """Legacy v1 Transformer, preserved for safe checkpoint compatibility."""
    def __init__(self, width=64, layers=2, context=128):
        super().__init__()
        self.config = dict(width=width,layers=layers,context=context)
        self.context=context
        self.token=nn.Embedding(259,width);self.position=nn.Embedding(context,width)
        block=nn.TransformerEncoderLayer(width,4,width*3,dropout=0.0,batch_first=True,activation="gelu")
        self.blocks=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,259,bias=False)
    def forward(self,tokens):
        length=tokens.shape[1]
        x=self.token(tokens)+self.position(torch.arange(length,device=tokens.device))
        mask=torch.triu(torch.ones(length,length,device=tokens.device,dtype=torch.bool),1)
        return self.head(self.norm(self.blocks(x,mask=mask)))
    @torch.inference_mode()
    def generate(self,prompt,limit=96,**_):
        tok=ByteTokenizer();ids=[tok.bos]+tok.encode(prompt);out=[];self.eval()
        for _ in range(min(limit,256)):
            x=torch.tensor([ids[-self.context:]],device=next(self.parameters()).device)
            predicted=int(self(x)[0,-1].argmax())
            if predicted in (tok.eos,tok.pad,tok.bos):break
            ids.append(predicted);out.append(predicted)
        return tok.decode(out)


class _LegacyAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.head_dim=width//heads
        self.qkv=nn.Linear(width,width*3,bias=False);self.out=nn.Linear(width,width,bias=False)
    def forward(self,x,cache=None,use_cache=False):
        b,t,w=x.shape
        qkv=self.qkv(x).view(b,t,3,self.heads,self.head_dim)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        if cache is not None:
            pk,pv=cache;k=torch.cat((pk,k),dim=2);v=torch.cat((pv,v),dim=2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=cache is None and t>1)
        y=y.transpose(1,2).contiguous().view(b,t,w)
        return self.out(y),(k,v) if use_cache else None


class _LegacyBlock(nn.Module):
    def __init__(self,width,heads,ff):
        super().__init__();self.norm1=nn.LayerNorm(width);self.attn=_LegacyAttention(width,heads)
        self.norm2=nn.LayerNorm(width);self.mlp=nn.Sequential(nn.Linear(width,ff,bias=False),nn.GELU(),nn.Linear(ff,width,bias=False))
    def forward(self,x,cache=None,use_cache=False):
        y,new=self.attn(self.norm1(x),cache,use_cache);x=x+y;x=x+self.mlp(self.norm2(x));return x,new


class LegacyNimoLMV2(nn.Module):
    """The transitional architecture used by earlier v0.2 language checkpoints."""
    def __init__(self,width=64,layers=2,heads=4,ff=192,context=128,version=2):
        super().__init__();self.context=context
        self.config=dict(version=2,width=width,layers=layers,heads=heads,ff=ff,context=context)
        self.token=nn.Embedding(259,width);self.position=nn.Embedding(context,width)
        self.blocks=nn.ModuleList([_LegacyBlock(width,heads,ff) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,259,bias=False);self.head.weight=self.token.weight
    def forward(self,tokens,cache=None,use_cache=False):
        length=tokens.shape[1];past=cache[0][0].shape[2] if cache else 0
        if past+length>self.context:raise ValueError("Input exceeds model context")
        pos=torch.arange(past,past+length,device=tokens.device);x=self.token(tokens)+self.position(pos)
        next_cache=[] if use_cache else None
        for i,block in enumerate(self.blocks):
            x,saved=block(x,cache[i] if cache else None,use_cache)
            if use_cache:next_cache.append(saved)
        logits=self.head(self.norm(x));return (logits,next_cache) if use_cache else logits
    @torch.inference_mode()
    def generate(self,prompt,limit=128,temperature=0.8,top_k=40,**_):
        tok=ByteTokenizer();ids=[tok.bos]+tok.encode(prompt)[-(self.context//2):];out=[];cache=None
        device=next(self.parameters()).device
        logits,cache=self(torch.tensor([ids],device=device),use_cache=True)
        for _ in range(min(limit,self.context-len(ids))):
            scores=logits[0,-1].float()
            if temperature<=0:pred=int(scores.argmax())
            else:
                scores/=max(temperature,1e-5)
                if top_k:
                    cut=torch.topk(scores,min(top_k,scores.numel())).values[-1]
                    scores=scores.masked_fill(scores<cut,float("-inf"))
                pred=int(torch.multinomial(scores.softmax(-1),1))
            if pred in (tok.eos,tok.pad,tok.bos):break
            out.append(pred);logits,cache=self(torch.tensor([[pred]],device=device),cache=cache,use_cache=True)
        return tok.decode(out)


@dataclass(frozen=True)
class ModelProfile:
    width: int
    layers: int
    heads: int
    ff: int
    context: int


MODEL_PROFILES = {
    "tiny": dict(width=288,layers=8,heads=6,ff=1152,context=1024),
    "small": dict(width=384,layers=12,heads=6,ff=1536,context=1024),
    "medium": dict(width=512,layers=20,heads=8,ff=2048,context=2048),
    "micro": dict(width=64,layers=2,heads=4,ff=192,context=128),
    "lite": dict(width=384,layers=12,heads=6,ff=1536,context=1024),
    "standard": dict(width=384,layers=12,heads=6,ff=1536,context=1024),
}


class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x):
        dtype=x.dtype
        y=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return (y*self.weight.float()).to(dtype)


def _rotate_half(x):
    x1=x[...,::2];x2=x[...,1::2]
    return torch.stack((-x2,x1),dim=-1).flatten(-2)


def _rope(q,k,positions,base=10000.0):
    dim=q.shape[-1]
    inv=1.0/(base**(torch.arange(0,dim,2,device=q.device,dtype=torch.float32)/dim))
    freq=torch.outer(positions.float(),inv)
    emb=torch.repeat_interleave(freq,2,dim=-1)
    cos=emb.cos()[None,None,:,:].to(q.dtype);sin=emb.sin()[None,None,:,:].to(q.dtype)
    return q*cos+_rotate_half(q)*sin,k*cos+_rotate_half(k)*sin


class CausalSelfAttention(nn.Module):
    def __init__(self,width,heads,dropout=0.0,rope_base=10000.0):
        super().__init__()
        if width%heads:raise ValueError("width must be divisible by heads")
        self.heads=heads;self.head_dim=width//heads;self.dropout=float(dropout);self.rope_base=rope_base
        self.qkv=nn.Linear(width,width*3,bias=False);self.out=nn.Linear(width,width,bias=False)
    def forward(self,x,positions,cache=None,use_cache=False):
        b,t,w=x.shape
        qkv=self.qkv(x).view(b,t,3,self.heads,self.head_dim)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        q,k=_rope(q,k,positions,self.rope_base)
        if cache is not None:
            pk,pv=cache;k=torch.cat((pk,k),dim=2);v=torch.cat((pv,v),dim=2)
        drop=self.dropout if self.training else 0.0
        causal=cache is None and t>1
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=drop,is_causal=causal)
        y=y.transpose(1,2).contiguous().view(b,t,w)
        return self.out(y),(k,v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self,width,hidden,dropout=0.0):
        super().__init__();self.gate=nn.Linear(width,hidden,bias=False);self.up=nn.Linear(width,hidden,bias=False)
        self.down=nn.Linear(hidden,width,bias=False);self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.down(F.silu(self.gate(x))*self.up(x)))


class DecoderBlock(nn.Module):
    def __init__(self,width,heads,ff,dropout=0.0,rope_base=10000.0):
        super().__init__();self.norm1=RMSNorm(width);self.attn=CausalSelfAttention(width,heads,dropout,rope_base)
        self.norm2=RMSNorm(width);self.ff=SwiGLU(width,ff,dropout)
    def forward(self,x,positions,cache=None,use_cache=False):
        y,new=self.attn(self.norm1(x),positions,cache,use_cache)
        x=x+y;x=x+self.ff(self.norm2(x));return x,new


class NimoLMV2(nn.Module):
    """Modern decoder-only causal Transformer trained from random initialization."""
    def __init__(self,vocab_size,width=384,layers=12,heads=6,ff=1536,context=1024,
                 dropout=0.0,rope_base=10000.0,version=3,family="nimo_lm_v2"):
        super().__init__()
        if not 264<=vocab_size<=65536:raise ValueError("vocab_size outside safe bounds")
        if width%heads:raise ValueError("width must be divisible by heads")
        self.context=int(context);self.vocab_size=int(vocab_size)
        self.config=dict(version=3,family=family,vocab_size=vocab_size,width=width,layers=layers,heads=heads,
                         ff=ff,context=context,dropout=float(dropout),rope_base=float(rope_base))
        self.token=nn.Embedding(vocab_size,width)
        self.blocks=nn.ModuleList([DecoderBlock(width,heads,ff,dropout,rope_base) for _ in range(layers)])
        self.norm=RMSNorm(width);self.head=nn.Linear(width,vocab_size,bias=False);self.head.weight=self.token.weight
        self.apply(self._init_weights)
        scale=1/math.sqrt(2*layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight,mean=0.0,std=0.02*scale)
            nn.init.normal_(block.ff.down.weight,mean=0.0,std=0.02*scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,tokens,cache=None,use_cache=False):
        if tokens.ndim!=2:raise ValueError("tokens must have shape [batch, sequence]")
        t=tokens.shape[1];past=cache[0][0].shape[2] if cache else 0
        if past+t>self.context:raise ValueError("Input exceeds model context")
        positions=torch.arange(past,past+t,device=tokens.device)
        x=self.token(tokens);next_cache=[] if use_cache else None
        for i,block in enumerate(self.blocks):
            x,saved=block(x,positions,cache[i] if cache else None,use_cache)
            if use_cache:next_cache.append(saved)
        logits=self.head(self.norm(x))
        return (logits,next_cache) if use_cache else logits

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _sample(scores,history,temperature,top_k,top_p,repetition_penalty,frequency_penalty,presence_penalty,generator):
        scores=scores.float().clone()
        if history:
            counts={}
            for token in history:counts[token]=counts.get(token,0)+1
            for token,count in counts.items():
                if 0<=token<scores.numel():
                    if repetition_penalty and repetition_penalty!=1.0:
                        scores[token]=scores[token]/repetition_penalty if scores[token]>=0 else scores[token]*repetition_penalty
                    scores[token]-=frequency_penalty*count
                    scores[token]-=presence_penalty
        if temperature is None or temperature<=0:return int(scores.argmax())
        scores/=max(float(temperature),1e-5)
        if top_k:
            k=min(int(top_k),scores.numel());cut=torch.topk(scores,k).values[-1]
            scores=scores.masked_fill(scores<cut,float("-inf"))
        probs=scores.softmax(-1)
        if top_p and 0<float(top_p)<1:
            sp,si=probs.sort(descending=True);cum=sp.cumsum(-1)
            remove=cum-sp>float(top_p);sp=sp.masked_fill(remove,0)
            denom=sp.sum()
            if denom>0:sp=sp/denom
            pick=torch.multinomial(sp,1,generator=generator);return int(si[pick])
        return int(torch.multinomial(probs,1,generator=generator))

    @torch.inference_mode()
    def generate_ids(self,prompt_ids,tokenizer,limit=256,temperature=0.7,top_k=40,top_p=0.9,
                     repetition_penalty=1.1,frequency_penalty=0.0,presence_penalty=0.0,
                     stop_sequences=None,seed=None,deterministic=False):
        self.eval();device=next(self.parameters()).device
        max_prompt=max(1,self.context-1)
        ids=list(map(int,prompt_ids[-max_prompt:]))
        if not ids:ids=[tokenizer.special_id("BOS")]
        x=torch.tensor([ids],dtype=torch.long,device=device)
        logits,cache=self(x,use_cache=True);produced=[]
        generator=None
        if seed is not None:
            generator=torch.Generator(device=device);generator.manual_seed(int(seed))
        if deterministic:temperature=0.0
        stop_ids=[tokenizer.encode(s) for s in (stop_sequences or []) if s]
        eos=tokenizer.special_id("EOS");pad=tokenizer.special_id("PAD")
        for _ in range(min(int(limit),max(0,self.context-len(ids)))):
            pred=self._sample(logits[0,-1],produced[-256:],temperature,top_k,top_p,repetition_penalty,
                              float(frequency_penalty),float(presence_penalty),generator)
            if pred in (eos,pad):break
            produced.append(pred)
            stopped=False
            for stop in stop_ids:
                if stop and produced[-len(stop):]==stop:
                    del produced[-len(stop):];stopped=True;break
            if stopped:break
            yield pred
            logits,cache=self(torch.tensor([[pred]],device=device),cache=cache,use_cache=True)

    @torch.inference_mode()
    def generate(self,prompt_ids,tokenizer,**kwargs):
        return tokenizer.decode(list(self.generate_ids(prompt_ids,tokenizer,**kwargs)))

    @torch.inference_mode()
    def stream_generate(self,prompt_ids,tokenizer,**kwargs):
        decoder=codecs.getincrementaldecoder("utf-8")("replace")
        for token in self.generate_ids(prompt_ids,tokenizer,**kwargs):
            piece=tokenizer.tokens[token] if token>=8 and token<len(tokenizer.tokens) else b""
            text=decoder.decode(piece,final=False)
            if text:yield text
        tail=decoder.decode(b"",final=True)
        if tail:yield tail


def build_text_model(profile="small",context=None,vocab_size=8000,dropout=0.0):
    if profile not in MODEL_PROFILES:raise ValueError("Unknown text model profile")
    if profile=="micro" and (context is None or int(context)<=512) and int(vocab_size)==8000:
        return LegacyNimoLMV2(width=64,layers=2,heads=4,ff=192,context=int(context or 128))
    cfg=dict(MODEL_PROFILES[profile])
    if context is not None:cfg["context"]=int(context)
    if cfg["context"] not in (256,512,1024,2048,4096):raise ValueError("Unsupported context size")
    return NimoLMV2(vocab_size=int(vocab_size),dropout=float(dropout),**cfg)


class NimoVision(nn.Module):
    def __init__(self,classes):
        super().__init__();self.config=dict(classes=classes)
        self.features=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,classes))
    def forward(self,x):return self.head(self.features(x))


class NimoIntent(nn.Module):
    def __init__(self,classes,features=2048):
        super().__init__();self.config=dict(classes=classes,features=features);self.head=nn.Linear(features,classes)
    def forward(self,x):return self.head(x)


def _validate_modern(a):
    width=a.get("width",0);layers=a.get("layers",0);heads=a.get("heads",0);ff=a.get("ff",0);context=a.get("context",0)
    vocab=a.get("vocab_size",0)
    if not (264<=vocab<=65536 and 128<=width<=1024 and 1<=layers<=32 and 1<=heads<=16 and width%heads==0
            and width<=ff<=width*8 and 128<=context<=8192):
        raise ValueError("Text architecture exceeds local safety bounds")


def load_model(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(checkpoint,dict) or "task" not in checkpoint or "architecture" not in checkpoint:
        raise ValueError("Invalid checkpoint")
    task=checkpoint["task"];a=checkpoint["architecture"]
    if task=="text":
        version=a.get("version",1)
        if version==1:
            if not (16<=a.get("width",0)<=256 and a.get("width",0)%4==0 and 1<=a.get("layers",0)<=6 and 32<=a.get("context",0)<=512):
                raise ValueError("Text architecture exceeds local safety bounds")
            model=NimoLM(**a)
        elif version==2:
            if not (32<=a.get("width",0)<=640 and 1<=a.get("layers",0)<=12 and 1<=a.get("heads",0)<=16
                    and a.get("width",0)%a.get("heads",1)==0 and 32<=a.get("context",0)<=512):
                raise ValueError("Text architecture exceeds local safety bounds")
            model=LegacyNimoLMV2(**a)
        elif version==3 and a.get("family")=="nimo_lm_v2":
            _validate_modern(a);model=NimoLMV2(**a)
        else:raise ValueError("Unsupported text architecture version")
    elif task=="image":
        if not 2<=a.get("classes",0)<=1000:raise ValueError("Classifier architecture exceeds local safety bounds")
        model=NimoVision(**a)
    elif task=="intent":
        if not 2<=a.get("classes",0)<=1000 or a.get("features")!=2048:raise ValueError("Unsupported intent architecture")
        model=NimoIntent(**a)
    else:raise ValueError("Unsupported checkpoint task")
    if "model" not in checkpoint:raise ValueError("Invalid checkpoint")
    model.load_state_dict(checkpoint["model"],strict=True);model.eval();return model,checkpoint


def quantize_for_inference(model,mode="auto",device="cpu"):
    """Quantize Nimo's own trained model; never downloads alternate weights."""
    mode=(mode or "auto").lower()
    if mode=="auto":
        mode="int8" if device=="cpu" else "bf16" if device=="cuda" and torch.cuda.is_bf16_supported() else "fp16" if device!="cpu" else "fp32"
    if mode=="fp32":return model.float(),mode
    if mode=="fp16":
        if device=="cpu":return model.float(),"fp32"
        return model.half(),mode
    if mode=="bf16":
        if device=="cpu" and not hasattr(torch,"bfloat16"):return model.float(),"fp32"
        return model.to(dtype=torch.bfloat16),mode
    if mode=="int8":
        if device!="cpu":return model, "fp32"
        try:
            q=torch.ao.quantization.quantize_dynamic(model,{nn.Linear},dtype=torch.qint8)
            return q,mode
        except Exception:
            return model.float(),"fp32"
    raise ValueError("Quantization must be auto, fp32, fp16, bf16, or int8")
