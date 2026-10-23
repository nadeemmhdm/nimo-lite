# Training Guide

## Recommended pipeline

1. Import or create correctly licensed data.
2. Review and approve it.
3. Train the Nimo tokenizer.
4. Pretrain a base NimoLM.
5. Instruction tune the base/chat model.
6. Evaluate held-out validation/test data.
7. Activate an accepted checkpoint.

The bundled starter examples are only demonstration material. A useful general language model needs substantially more diverse data and compute.

## Tokenizer

Use 8,000 tokens for Tiny/Small experiments. 16,000 can help larger corpora/Medium models but increases embedding parameters. Tokenizer training uses only approved local corpus and reviewed text examples.

## Pretraining

Choose **Pretrain base language model**. Raw, knowledge, code, and math documents in the stable train split become contiguous next-token sequences. Validation/test documents are never selected for optimization.

Typical starting settings:
- Tiny, 512–1024 context on 6–10 GB RAM
- Small, 1024 context on 10–24 GB RAM
- Medium, 2048 context on 24+ GB RAM / capable GPU

Start with conservative batch size and use gradient accumulation.

## Instruction tuning

Choose **Instruction tune chat model** after base pretraining. Structured examples use system/user/assistant/tool roles. Loss is masked for non-assistant tokens so Nimo learns to produce assistant replies rather than echoing user/system text.

Multi-turn conversations are supported. Keep facts that should remain editable in memory/RAG instead of repeatedly baking personal information into weights.

## Checkpoints

Native checkpoints include architecture, tokenizer artifact/hash, weights, optimizer, scheduler, RNG state, step, epoch, dataset hashes, configuration, validation metrics, and metadata. Resume requires the same tokenizer and data hashes.

## Metrics

Watch training loss, validation loss, perplexity, next-token accuracy, tokens/second, and held-out task probes. Lower loss is necessary but not sufficient evidence of conversational quality.

## Precision

FP32 works everywhere. CUDA may use BF16/FP16; MPS may use FP16 where supported. Inference can request INT8 dynamic quantization on CPU. Quantization applies only to Nimo's own checkpoint.
