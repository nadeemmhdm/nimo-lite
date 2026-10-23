# Nimo Lite local inference performance

Nimo measures performance instead of displaying hard-coded speed claims. The benchmark uses Nimo's own architecture with local randomly initialized weights and downloads no pretrained model. Runtime/kernel speed is therefore measurable independently of language quality; model quality must still be evaluated on a trained checkpoint.

## Benchmark command

```bash
python -m nimo.benchmark --profile small --context 1024 --prompt-tokens 128 --new-tokens 24 --repeats 2 --output benchmark-results.json
```

Use the same machine, model profile, context, precision, prompt size, output length and thread configuration when comparing changes. Add `--quantization auto` to measure the locally selected inference representation. The JSON result records platform, PyTorch version, model parameters, load time, prompt size, tokenizer throughput, first-token latency, prefill throughput, decode throughput, incremental-prefix reuse, current/peak resident memory and the quantization decision.

## Recorded baseline before the inference rewrite

The baseline was captured in GitHub Actions before performance-critical inference code was changed.

- Platform: Linux Azure GitHub Actions runner
- CPU: 4 logical cores available to the job
- Device: CPU
- PyTorch: 2.14.1+cpu
- Nimo profile: Small
- Parameters: 31,393,152
- Context capacity: 1024 tokens
- Prompt: 128 tokens
- Generated decode workload: 24 tokens
- PyTorch threads: 2
- Model construction/load: 0.3421 s
- Cached first-token stage: 0.07437 s
- Cached prefill: 1,721.94 tokens/s
- Existing tuple-cache decode: 75.11 tokens/s
- Uncached full-sequence decode: 13.46 tokens/s
- Peak RSS: 417.68 MiB

These figures describe that CI runner only. Nimo must not present them as expected speed on another computer.

## Measured result after optimization

The optimized implementation was measured again on the same GitHub Actions workload and hardware class. Small run-to-run variation is expected on shared CI infrastructure, so these values are evidence for this run rather than a universal speed promise.

| Metric | Baseline | Optimized FP32 | Optimized Auto |
| --- | ---: | ---: | ---: |
| Model first-token stage | 74.37 ms | 62.07 ms | 26.09 ms |
| Prompt prefill | 1,721.94 tok/s | 2,062.40 tok/s | 4,924.93 tok/s |
| Cached decode | 75.11 tok/s | 75.38 tok/s | 199.89 tok/s |
| Uncached full-sequence decode | 13.46 tok/s | 15.16 tok/s | 38.54 tok/s |
| Model load / representation selection | 0.342 s | 0.357 s | 0.670 s |
| KV cache | previous growing tuples | 36.0 MiB preallocated | 36.0 MiB preallocated |
| Peak process RSS | 417.68 MiB | 427.75 MiB | 549.05 MiB |
| Steady RSS immediately after final model load | not recorded in baseline | 360.54 MiB | 330.79 MiB |
| Steady RSS after benchmark workload | not recorded in baseline | 428.04 MiB | 370.11 MiB |

On this run, FP32 reduced the measured model first-token stage by about 16.5% and improved prefill by about 19.8%; decode throughput was essentially unchanged because the previous implementation already had a functional tuple KV cache. The important FP32 change is that decode now writes into fixed K/V storage instead of repeatedly growing K/V tensors, which bounds cache allocation behavior and enables safe prefix reuse across turns.

CPU Auto selected INT8 only after its local speed and quality probe passed. Relative to the pre-change baseline on this run, Auto reduced the measured model first-token stage by about 64.9%, increased prefill throughput by about 186.0%, and increased cached decode throughput by about 166.1%. The probe measured a 2.76x kernel speedup, preserved the same next-token decision, and measured normalized logit MAE of about 0.0546, below the configured 0.08 guard.

Auto has a deliberate one-time startup tradeoff: while evaluating FP32 versus INT8 it temporarily holds both representations, so peak RSS and model-selection time are higher. After the rejected representation is released and the allocator is trimmed where supported, this run measured 330.79 MiB steady RSS after load, about 8.3% below the optimized FP32 run. Peak RSS is reported rather than hidden.

The incremental prompt benchmark also proved that an unchanged 128-token prefix was reused and only the appended 8-token suffix was processed. On this run the suffix processed at about 467.54 tok/s in FP32 and 1,139.09 tok/s in Auto/INT8. This optimization applies only when the prompt prefix is byte-for-byte/token-for-token compatible; edits, changed memory/RAG context and new chats safely invalidate or truncate the cached prefix.

The tokenizer microbenchmark in the same run measured roughly 794k encode tokens/s and 7.02M decode tokens/s in the FP32 process. These numbers are included to expose tokenizer cost, not as a cross-machine guarantee.

## Optimized inference design

The current fast path keeps training unchanged and adds an inference-specific batch-1 engine:

1. The prompt is prefetched once.
2. Each Transformer layer owns a preallocated K/V buffer sized to the active model context.
3. New tokens write K/V directly into the next cache position instead of concatenating the complete cache on every decode step.
4. Single-token decode uses PyTorch scaled-dot-product attention without constructing a square causal mask.
5. RoPE positions start at the current cache position, preserving cached/uncached numerical equivalence.
6. The exact unchanged token prefix can be reused on the next turn. Changed history truncates to the exact common prefix; a new chat resets inference prefix state.
7. The model object and tokenizer artifact are reused rather than reconstructed for every message.
8. Generation runs under `torch.inference_mode()` with the model in evaluation mode.
9. Sampling applies penalties only to tokens that have appeared and applies nucleus filtering to top-k candidates when top-k is enabled.
10. Streaming yields the first generated token before computing the following decode step.

## Retrieval and memory

Local RAG and long-term memory use SQLite FTS5 indexes where the local SQLite build supports FTS5. Inserts, updates and deletes are synchronized through database triggers. A portable bounded/BM25 fallback remains available for SQLite builds without FTS5. Normal retrieval is capped to a small top-k set before prompt injection.

## CPU threads

Inference threads are separate from training intent. Nimo accepts the environment override:

```bash
NIMO_INFERENCE_THREADS=4 nimo-run
```

Without an override, the inference engine runs a small one-time local microbenchmark across safe candidates such as 1, 2, 4 and the bounded physical-core count, then keeps the fastest candidate for the process. More threads are not assumed to be faster.

## Quantization

Available inference choices remain `Auto`, `FP32`, `BF16`, `FP16` and `INT8` where supported. Explicit choices are respected. On CPU, Auto no longer assumes INT8 is faster: it compares FP32 with a dynamic-INT8 candidate using a tiny local Nimo probe. INT8 is selected only when it clears a minimum measured speed gain and a next-token/logit quality guard; otherwise Auto keeps FP32. This decision occurs once while loading the model, not on every chat message.

The probe is a safety/performance guard, not a substitute for full task-quality evaluation of a trained quantized checkpoint.

## Warmup and acceleration

A tiny invisible warmup is performed after the final inference representation is selected. It can be disabled with `NIMO_WARMUP=0`. CUDA uses the compatible PyTorch SDPA backend and FP16/BF16 choices exposed by the runtime. Apple MPS uses supported PyTorch operations and falls back safely when acceleration is unavailable. Core inference remains fully local.

`torch.compile` is intentionally not enabled automatically. Compilation can increase startup latency for small local models and the current optimized path has not demonstrated a repeatable measured benefit from enabling it. It should only become an automatic choice after a same-hardware benchmark proves a net benefit for the relevant backend and workload.

## Context and generation

Long prompts increase prefill latency and K/V memory. Nimo's existing Fast/Balanced/Deep reasoning modes bound output and evidence budgets, while the prompt builder preserves the system instruction and current user message under context pressure. Use Fast for latency-sensitive chat, Balanced for normal use, and a larger-context model/profile only when the extra context is needed.

Generation stops on EOS/PAD, configured stop sequences, the output token limit, or client cancellation. The streaming generator yields between decode steps so the UI Stop action can cancel promptly.

## Developer performance panel

The existing Performance Monitor contains an optional collapsed inference developer panel. It shows measured values from the latest completed local generation rather than synthetic indicators: active device, dtype/quantization, parameter count, context, prompt/processed/reused tokens, model first-token latency, decode tokens/s, K/V cache memory, process RAM, inference threads, VRAM where available and the Auto quantization decision. Normal chat remains uncluttered.

## Correctness regressions

Performance changes are covered by tests for:

- fast prefill logits versus the normal full forward path;
- one-token cached decode versus full uncached forward;
- deterministic cached generation versus uncached greedy generation;
- exact prefix reuse and edit invalidation;
- stable preallocated K/V storage across decode steps;
- tokenizer artifact reuse;
- separate model cache sessions;
- no model reload between repeated requests for the same active model;
- model reload after an explicit model-cache clear/switch boundary;
- new-chat cache reset;
- indexed RAG and memory retrieval;
- explicit and Auto quantization loading.

## Interpreting results

First-token latency and decode throughput are separate. A fast decoder can still feel slow if tokenization, retrieval or prompt prefill dominates. The developer panel labels its first-token metric as model-stage latency for that reason. Benchmark the complete local application on the target hardware and use measured values rather than transferring CI numbers to another machine.
