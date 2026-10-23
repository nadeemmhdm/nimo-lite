# NimoLM v2 Model Card

## Model

NimoLM v2 is an original decoder-only causal Transformer implemented locally in Nimo Lite and initialized from random weights. No pretrained LLM or tokenizer weights are required.

Architecture features: RMSNorm, RoPE, causal SDPA attention, SwiGLU, tied embeddings, configurable dropout, KV-cache inference.

## Profiles

Approximate counts with an 8k tokenizer:
- Tiny: ~13M parameters, width 288, 8 blocks, 6 heads, FF 1152
- Small: ~32M, width 384, 12 blocks, 6 heads, FF 1536
- Medium: ~88M, width 512, 20 blocks, 8 heads, FF 2048

Context defaults to 1024/1024/2048 respectively and is configurable.

## Tokenizer

Locally trained Nimo byte-subword tokenizer with complete byte fallback and role/special tokens. Designed for English, Malayalam, Manglish, code, math, and general UTF-8.

## Training

Base pretraining uses next-token prediction. Instruction tuning masks loss on system/user/tool content and optimizes assistant responses. Checkpoints record data hashes and exact tokenizer.

## Data

Nimo Lite ships only small original/CC0 demonstration examples. Users are responsible for supplying sufficiently large, correctly licensed or owned datasets. The project does not bundle a copyrighted web-scale corpus.

## Evaluation

Native checkpoints report validation/test loss, perplexity, next-token accuracy, training history, parameter count, and tokenizer size. Conversational continuity, multilingual, coding, math, uncertainty, memory retrieval, and source-grounding should be evaluated separately on held-out probes.

## Intended use

Local learning/research, personal assistants trained on authorized data, experimentation with small language models, and inspectable local memory/RAG.

## Limitations

A 13M–88M randomly initialized model trained on small data is not comparable to frontier assistants. It may hallucinate, generate malformed code, fail math, mishandle multilingual nuance, or give unsafe/inaccurate high-stakes advice. Current facts require source verification. Training loss alone does not establish reasoning or intelligence.

## Privacy and safety

Core inference/training is local. Memory is inspectable/deletable. Optional web retrieval is explicit and untrusted content is not executed. NimoLM has no shell execution tool.
