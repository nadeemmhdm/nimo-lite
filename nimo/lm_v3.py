"""Modern decoder-only Nimo language model implemented from local PyTorch primitives.

The architecture is initialized from random weights. It does not load pretrained
Transformer weights, tokenizers, or remote model code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F


MODEL_PRESETS = {
    # Parameter counts depend on tokenizer vocabulary. With an 8k vocabulary these
    # land roughly in the requested size classes.
    "tiny": dict(dim=320, layers=8, heads=5, ff_dim=864, context=1024),
    "small": dict(dim=384, layers=12, heads=6, ff_dim=1536, context=1024),
    "medium": dict(dim=640, layers=16, heads=10, ff_dim=2048, context=2048),
}


@dataclass
class NimoConfig:
    vocab_size: int
    dim: int = 384
    layers: int = 8
    heads: int = 6
    ff_dim: int = 1536
    context: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    version: int = 3
    architecture: str = "nimo-decoder-rms-rope-swiglu"

    def validate(self):
        if self.version != 3:
            raise ValueError("Unsupported Nimo decoder version")
        if not 265 <= self.vocab_size <= 32768:
            raise ValueError("vocab_size is outside safe bounds")
        if not 128 <= self.dim <= 1024 or self.dim % self.heads:
            raise ValueError("dim must be 128..1024 and divisible by heads")
        if not 1 <= self.layers <= 24 or not 1 <= self.heads <= 16:
            raise ValueError("layers/heads exceed safe local bounds")
        if not self.dim <= self.ff_dim <= self.dim * 6:
            raise ValueError("ff_dim exceeds safe bounds")
        if not 128 <= self.context <= 8192:
            raise ValueError("context must be 128..8192")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be 0..0.5")
        return self

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * scale * self.weight


def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, positions, theta=10000.0):
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    cos = emb.cos().to(q.dtype)[None, None, :, :]
    sin = emb.sin().to(q.dtype)[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalAttention(nn.Module):
    def __init__(self, config: NimoConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.theta = config.rope_theta
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        past = cache[0].shape[2] if cache is not None else 0
        positions = torch.arange(past, past + length, device=x.device)
        q, k = apply_rope(q, k, positions, self.theta)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        # Full-sequence training uses causal SDPA. Cached inference normally passes
        # one new token and all keys in the cache are valid history for that token.
        causal = cache is None and length > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(y), (k, v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, config: NimoConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.up = nn.Linear(config.dim, config.ff_dim, bias=False)
        self.down = nn.Linear(config.ff_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: NimoConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLU(config)

    def forward(self, x, cache=None, use_cache=False):
        a, saved = self.attn(self.attn_norm(x), cache=cache, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, saved


class NimoDecoderLM(nn.Module):
    def __init__(self, config: NimoConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = NimoConfig(**config)
        self.cfg = config.validate()
        self.config = self.cfg.to_dict()
        self.context = self.cfg.context
        self.token = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = RMSNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init_weights)
        # Standard residual scaling helps deeper randomly initialized decoders.
        scale = 1.0 / math.sqrt(2.0 * self.cfg.layers)
        for block in self.blocks:
            block.attn.out.weight.data.mul_(scale)
            block.ffn.down.weight.data.mul_(scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, cache=None, use_cache=False):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        past = cache[0][0].shape[2] if cache else 0
        if past + tokens.shape[1] > self.context:
            raise ValueError("Input exceeds model context")
        x = self.dropout(self.token(tokens))
        next_cache = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            layer_cache = cache[index] if cache else None
            x, saved = block(x, cache=layer_cache, use_cache=use_cache)
            if use_cache:
                next_cache.append(saved)
        logits = self.head(self.norm(x))
        return (logits, next_cache) if use_cache else logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_nimo_model(vocab_size: int, preset: str = "small", **overrides):
    if preset not in MODEL_PRESETS:
        raise ValueError("Unknown Nimo model preset")
    values = {**MODEL_PRESETS[preset], **overrides, "vocab_size": int(vocab_size)}
    return NimoDecoderLM(NimoConfig(**values))


def sample_next_token(
    logits,
    generated: list[int],
    *,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    generator=None,
):
    scores = logits.float().clone()
    if generated:
        counts = {}
        for token in generated:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            if repetition_penalty and repetition_penalty != 1.0:
                scores[token] = scores[token] / repetition_penalty if scores[token] >= 0 else scores[token] * repetition_penalty
            scores[token] -= float(frequency_penalty) * count
            scores[token] -= float(presence_penalty)
    if temperature is None or temperature <= 0:
        return int(scores.argmax())
    scores.div_(max(float(temperature), 1e-5))
    if top_k:
        k = min(max(1, int(top_k)), scores.numel())
        cutoff = torch.topk(scores, k).values[-1]
        scores.masked_fill_(scores < cutoff, float("-inf"))
    probs = scores.softmax(-1)
    if top_p and 0 < top_p < 1:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(-1)
        remove = cumulative - sorted_probs > float(top_p)
        sorted_probs.masked_fill_(remove, 0)
        total = sorted_probs.sum()
        if total <= 0:
            return int(scores.argmax())
        sorted_probs.div_(total)
        pick = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_idx[pick])
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def generate_ids(
    model: NimoDecoderLM,
    input_ids: list[int],
    *,
    eos_id: int,
    max_new_tokens: int = 256,
    stop_sequences: list[list[int]] | None = None,
    seed: int | None = None,
    **sampling,
):
    if not input_ids:
        raise ValueError("Generation requires at least one input token")
    device = next(model.parameters()).device
    budget = max(0, model.context - len(input_ids))
    limit = min(int(max_new_tokens), budget)
    model.eval()
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, cache = model(x, use_cache=True)
    generated: list[int] = []
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(seed))
    stops = stop_sequences or []
    for _ in range(limit):
        token = sample_next_token(logits[0, -1], generated, generator=generator, **sampling)
        if token == eos_id:
            break
        generated.append(token)
        yield token
        if any(len(stop) <= len(generated) and generated[-len(stop):] == stop for stop in stops if stop):
            break
        step = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, cache = model(step, cache=cache, use_cache=True)
