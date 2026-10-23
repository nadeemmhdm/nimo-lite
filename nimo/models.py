"""Original locally trainable model architectures. No pretrained weights or tokenizer downloads."""
from __future__ import annotations

import codecs

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(min(2, torch.get_num_threads()))


class ByteTokenizer:
    vocab_size = 259
    pad, bos, eos = 256, 257, 258

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class NimoLM(nn.Module):
    """Legacy v1 model kept byte-for-byte compatible with v0.1/v0.2 checkpoints."""

    def __init__(self, width=64, layers=2, context=128):
        super().__init__()
        self.config = dict(width=width, layers=layers, context=context)
        self.context = context
        self.token = nn.Embedding(259, width)
        self.position = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(
            width, 4, width * 3, dropout=0.0, batch_first=True, activation="gelu"
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 259, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), 1)
        return self.head(self.norm(self.blocks(x, mask=mask)))

    @torch.inference_mode()
    def generate(self, prompt, limit=96, **_):
        tok = ByteTokenizer()
        ids = [tok.bos] + tok.encode(prompt)
        output = []
        self.eval()
        for _ in range(min(limit, 256)):
            x = torch.tensor([ids[-self.context :]], device=next(self.parameters()).device)
            predicted = int(self(x)[0, -1].argmax())
            if predicted in (tok.eos, tok.pad, tok.bos):
                break
            ids.append(predicted)
            output.append(predicted)
        return tok.decode(output)


MODEL_PROFILES = {
    "micro": dict(width=64, layers=2, heads=4, ff=192),
    "lite": dict(width=256, layers=6, heads=8, ff=768),
    "standard": dict(width=512, layers=10, heads=8, ff=2048),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if cache is not None:
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        causal = cache is None and length > 1
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(y), (k, v) if use_cache else None


class NimoBlock(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, ff, bias=False),
            nn.GELU(),
            nn.Linear(ff, width, bias=False),
        )

    def forward(self, x, cache=None, use_cache=False):
        y, new_cache = self.attn(self.norm1(x), cache=cache, use_cache=use_cache)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class NimoLMV2(nn.Module):
    """Causal LM with SDPA and KV-cache decoding, trained entirely from local data."""

    def __init__(self, width=64, layers=2, heads=4, ff=192, context=128, version=2):
        super().__init__()
        if version != 2:
            raise ValueError("Unsupported NimoLMV2 architecture version")
        self.config = dict(
            version=2, width=width, layers=layers, heads=heads, ff=ff, context=context
        )
        self.context = context
        self.token = nn.Embedding(ByteTokenizer.vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([NimoBlock(width, heads, ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, cache=None, use_cache=False):
        length = tokens.shape[1]
        past = cache[0][0].shape[2] if cache else 0
        if past + length > self.context:
            raise ValueError("Input exceeds model context")
        pos = torch.arange(past, past + length, device=tokens.device)
        x = self.token(tokens) + self.position(pos)
        next_cache = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            layer_cache = cache[index] if cache else None
            x, saved = block(x, cache=layer_cache, use_cache=use_cache)
            if use_cache:
                next_cache.append(saved)
        logits = F.linear(self.norm(x), self.token.weight)
        return (logits, next_cache) if use_cache else logits

    @torch.inference_mode()
    def generate_tokens(
        self,
        prompt,
        limit=128,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.08,
    ):
        tok = ByteTokenizer()
        prompt_budget = max(1, self.context // 2)
        prompt_ids = tok.encode(prompt)[-prompt_budget:]
        ids = [tok.bos] + prompt_ids
        device = next(self.parameters()).device
        self.eval()
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, cache = self(x, use_cache=True)
        max_new = min(int(limit), max(0, self.context - len(ids)))
        produced = []
        for _ in range(max_new):
            scores = logits[0, -1].float()
            if repetition_penalty and repetition_penalty != 1.0:
                for token_id in set(produced[-64:]):
                    if 0 <= token_id < scores.numel():
                        scores[token_id] = (
                            scores[token_id] / repetition_penalty
                            if scores[token_id] >= 0
                            else scores[token_id] * repetition_penalty
                        )
            if temperature is None or temperature <= 0:
                predicted = int(scores.argmax())
            else:
                scores = scores / max(float(temperature), 1e-5)
                if top_k:
                    k = min(int(top_k), scores.numel())
                    cutoff = torch.topk(scores, k).values[-1]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                probs = scores.softmax(-1)
                if top_p and 0 < top_p < 1:
                    sorted_probs, sorted_idx = probs.sort(descending=True)
                    cumulative = sorted_probs.cumsum(-1)
                    remove = cumulative - sorted_probs > float(top_p)
                    sorted_probs = sorted_probs.masked_fill(remove, 0)
                    denom = sorted_probs.sum()
                    if denom > 0:
                        sorted_probs = sorted_probs / denom
                    pick = torch.multinomial(sorted_probs, 1)
                    predicted = int(sorted_idx[pick])
                else:
                    predicted = int(torch.multinomial(probs, 1))
            if predicted in (tok.eos, tok.pad, tok.bos):
                break
            produced.append(predicted)
            yield predicted
            step = torch.tensor([[predicted]], dtype=torch.long, device=device)
            logits, cache = self(step, cache=cache, use_cache=True)

    @torch.inference_mode()
    def generate(self, prompt, limit=128, **sampling):
        tok = ByteTokenizer()
        return tok.decode(list(self.generate_tokens(prompt, limit=limit, **sampling)))

    @torch.inference_mode()
    def stream_generate(self, prompt, limit=128, **sampling):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for token_id in self.generate_tokens(prompt, limit=limit, **sampling):
            piece = decoder.decode(bytes([token_id]), final=False)
            if piece:
                yield piece
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def build_text_model(profile="micro", context=128):
    if profile not in MODEL_PROFILES:
        raise ValueError("Unknown text model profile")
    if context not in (128, 256, 512):
        raise ValueError("Context must be 128, 256 or 512 bytes")
    return NimoLMV2(context=context, **MODEL_PROFILES[profile])


class NimoVision(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.config = dict(classes=classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.head(self.features(x))


class NimoIntent(nn.Module):
    """Original randomly initialized linear classifier over bounded hashed Unicode features."""

    def __init__(self, classes, features=2048):
        super().__init__()
        self.config = dict(classes=classes, features=features)
        self.head = nn.Linear(features, classes)

    def forward(self, x):
        return self.head(x)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    task = checkpoint["task"]
    architecture = checkpoint["architecture"]
    if task == "text":
        version = architecture.get("version", 1)
        if version == 1:
            if not (
                16 <= architecture.get("width", 0) <= 256
                and architecture["width"] % 4 == 0
                and 1 <= architecture.get("layers", 0) <= 6
                and 32 <= architecture.get("context", 0) <= 512
            ):
                raise ValueError("Text architecture exceeds local safety bounds")
            model = NimoLM(**architecture)
        elif version == 2:
            width = architecture.get("width", 0)
            layers = architecture.get("layers", 0)
            heads = architecture.get("heads", 0)
            ff = architecture.get("ff", 0)
            context = architecture.get("context", 0)
            if not (
                32 <= width <= 640
                and 1 <= layers <= 12
                and 1 <= heads <= 16
                and width % heads == 0
                and width <= ff <= width * 6
                and 32 <= context <= 512
            ):
                raise ValueError("Text architecture exceeds local safety bounds")
            model = NimoLMV2(**architecture)
        else:
            raise ValueError("Unsupported text architecture version")
    elif task == "image":
        if not 2 <= architecture.get("classes", 0) <= 1000:
            raise ValueError("Classifier architecture exceeds local safety bounds")
        model = NimoVision(**architecture)
    elif task == "intent":
        if not 2 <= architecture.get("classes", 0) <= 1000:
            raise ValueError("Classifier architecture exceeds local safety bounds")
        if architecture.get("features") != 2048:
            raise ValueError("Unsupported intent feature size")
        model = NimoIntent(**architecture)
    else:
        raise ValueError("Unsupported checkpoint task")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
