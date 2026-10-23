# Tokenizer Guide

Nimo does not download a pretrained tokenizer.

`NimoTokenizer` is trained from approved local text. It guarantees lossless UTF-8 representation through byte fallback and learns frequent multi-byte pieces to reduce token count for common language/code patterns.

## Special tokens

`PAD`, `BOS`, `EOS`, `SYSTEM`, `USER`, `ASSISTANT`, `TOOL`, `MEMORY`.

Their order is fixed and validated when an artifact loads.

## Vocabulary size

- 8k: recommended Tiny/Small starting point
- 16k: useful for larger/more multilingual corpora
- maximum currently accepted by training UI: 32k

A larger vocabulary increases the tied embedding matrix and model RAM.

## Multilingual behavior

The trainer is script-agnostic UTF-8. Malayalam, English, Manglish, source code, punctuation, and math can coexist in the same artifact. The full byte alphabet guarantees unseen words/scripts remain encodable.

## Artifact

Default: `data/tokenizer/tokenizer.json`.

The artifact contains format/version, method, special tokens, byte pieces, metadata, and a SHA-256 checksum. Native model checkpoints also embed the tokenizer artifact and its hash, preventing accidental model/tokenizer mismatch.
