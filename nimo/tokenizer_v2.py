"""Nimo's own trainable Unicode BPE tokenizer with byte fallback.

No pretrained vocabulary or tokenizer files are downloaded. Training starts from the
characters in local text plus deterministic UTF-8 byte fallback tokens, then learns
merge rules from the supplied corpus. Whitespace and code formatting are preserved.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|tool|>", "<|memory|>", "<|unk|>",
)
BYTE_TOKENS = tuple(f"<|byte:{i:02x}|>" for i in range(256))
TOKENIZER_FORMAT = "nimo-unicode-bpe-v1"


@dataclass(frozen=True)
class TokenizerInfo:
    format: str
    vocab_size: int
    merges: int
    sha256: str


class NimoTokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], *, version: str = TOKENIZER_FORMAT):
        self.version = version
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.merges = [tuple(x) for x in merges]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special = {name: self.vocab[name] for name in SPECIAL_TOKENS}
        self.byte_ids = {i: self.vocab[BYTE_TOKENS[i]] for i in range(256)}
        self.pad = self.special["<|pad|>"]
        self.bos = self.special["<|bos|>"]
        self.eos = self.special["<|eos|>"]
        self.system = self.special["<|system|>"]
        self.user = self.special["<|user|>"]
        self.assistant = self.special["<|assistant|>"]
        self.tool = self.special["<|tool|>"]
        self.memory = self.special["<|memory|>"]
        self.unk = self.special["<|unk|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8000, min_frequency: int = 2):
        if not 512 <= int(vocab_size) <= 32768:
            raise ValueError("Vocabulary size must be between 512 and 32768")
        if not 1 <= int(min_frequency) <= 1000:
            raise ValueError("min_frequency must be between 1 and 1000")
        corpus = [str(text) for text in texts if isinstance(text, str) and text]
        if not corpus:
            raise ValueError("Tokenizer training needs non-empty local text")

        base_tokens = list(SPECIAL_TOKENS) + list(BYTE_TOKENS)
        chars = sorted({char for text in corpus for char in text})
        vocab_tokens = base_tokens + [char for char in chars if char not in base_tokens]
        if len(vocab_tokens) > vocab_size:
            char_counts = Counter(char for text in corpus for char in text)
            room = max(0, vocab_size - len(base_tokens))
            chars = [char for char, _ in char_counts.most_common(room)]
            vocab_tokens = base_tokens + chars
        known_chars = set(vocab_tokens)

        sequences: list[list[str]] = []
        for text in corpus:
            seq: list[str] = []
            for char in text:
                if char in known_chars:
                    seq.append(char)
                else:
                    seq.extend(BYTE_TOKENS[b] for b in char.encode("utf-8"))
            if seq:
                sequences.append(seq)

        merges: list[tuple[str, str]] = []
        existing = set(vocab_tokens)
        while len(vocab_tokens) < vocab_size:
            counts: Counter[tuple[str, str]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < min_frequency:
                break
            merged = pair[0] + pair[1]
            if merged not in existing:
                vocab_tokens.append(merged)
                existing.add(merged)
            merges.append(pair)
            new_sequences = []
            a, b = pair
            for seq in sequences:
                out = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                new_sequences.append(out)
            sequences = new_sequences
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        return cls(vocab, merges)

    def _base_sequence(self, text: str) -> list[str]:
        output: list[str] = []
        for char in text:
            if char in self.vocab:
                output.append(char)
            else:
                output.extend(BYTE_TOKENS[b] for b in char.encode("utf-8"))
        return output

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        seq = self._base_sequence(text)
        for a, b in self.merges:
            if len(seq) < 2:
                break
            merged = a + b
            out = []
            i = 0
            changed = False
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                    changed = True
                else:
                    out.append(seq[i])
                    i += 1
            if changed:
                seq = out
        ids = [self.vocab.get(token, self.unk) for token in seq]
        if bos:
            ids.insert(0, self.bos)
        if eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        special_ids = set(self.special.values())
        output: list[str] = []
        pending = bytearray()

        def flush_bytes():
            if pending:
                output.append(bytes(pending).decode("utf-8", errors="replace"))
                pending.clear()

        for raw in ids:
            idx = int(raw)
            if skip_special and idx in special_ids:
                flush_bytes()
                continue
            token = self.id_to_token.get(idx, "")
            if token.startswith("<|byte:") and token.endswith("|>"):
                try:
                    pending.append(int(token[7:9], 16))
                except ValueError:
                    flush_bytes()
                continue
            flush_bytes()
            if not (skip_special and token in SPECIAL_TOKENS):
                output.append(token)
        flush_bytes()
        return "".join(output)

    def encode_chat(self, messages: list[dict], *, add_assistant_prompt: bool = True) -> tuple[list[int], list[int]]:
        """Return token IDs and assistant-only SFT mask."""
        role_token = {
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "tool": self.tool,
            "memory": self.memory,
        }
        ids = [self.bos]
        mask = [0]
        for message in messages:
            role = str(message.get("role", "user")).lower()
            if role not in role_token:
                raise ValueError(f"Unsupported chat role: {role}")
            content_ids = self.encode(str(message.get("content", "")))
            ids.append(role_token[role]); mask.append(0)
            ids.extend(content_ids); mask.extend([1 if role == "assistant" else 0] * len(content_ids))
            ids.append(self.eos); mask.append(1 if role == "assistant" else 0)
        if add_assistant_prompt:
            ids.append(self.assistant); mask.append(0)
        return ids, mask

    def to_dict(self) -> dict:
        return {
            "format": self.version,
            "special_tokens": list(SPECIAL_TOKENS),
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path) -> TokenizerInfo:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return TokenizerInfo(self.version, self.vocab_size, len(self.merges), hashlib.sha256(raw).hexdigest())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Tokenizer file is unexpectedly large")
        data = json.loads(path.read_text("utf-8"))
        if data.get("format") != TOKENIZER_FORMAT:
            raise ValueError("Unsupported Nimo tokenizer format")
        vocab = data.get("vocab")
        merges = data.get("merges")
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise ValueError("Invalid tokenizer file")
        if not 265 <= len(vocab) <= 32768:
            raise ValueError("Tokenizer vocabulary exceeds safe bounds")
        if any(not isinstance(k, str) or not isinstance(v, int) for k, v in vocab.items()):
            raise ValueError("Invalid tokenizer vocabulary")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("Tokenizer IDs must be contiguous")
        for required in SPECIAL_TOKENS + BYTE_TOKENS:
            if required not in vocab:
                raise ValueError("Tokenizer is missing required fallback/special tokens")
        clean_merges = []
        for pair in merges:
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
                raise ValueError("Invalid tokenizer merge table")
            clean_merges.append((pair[0], pair[1]))
        return cls(vocab, clean_merges, version=data["format"])


def tokenizer_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
