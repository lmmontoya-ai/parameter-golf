# SP2048 Dense Frontier Candidate

## Hypothesis

The fastest path to a legal high-score submission is no longer factorized embeddings or recurrence. It is a dense `sp2048` stack that combines the strongest publicly validated ingredients:

- `sp2048`
- `train_seq_len=2048`
- larger train batch
- `int6` fake quant + exact `int6_zstd` export
- `MLP_MULT=3`
- selective fp16 passthrough for quantization-sensitive tensors
- sliding-window eval
- artifact-aware checkpoint selection

## Why It May Help This Specific Competition

- Our local plain `sp2048` run already improved exact score materially, but missed the byte cap.
- Recent top PRs are converging on dense long-context `sp2048` + `int6` + `MLP3x` + selective precision + sliding-window evaluation.
- The current repo already supports most of this stack, so the next gain is likely to come from combining the proven pieces cleanly rather than inventing another architecture.
- Artifact-aware checkpoint selection is a plausible differentiator that targets the real metric directly.

## Competition / Rules Risk

- Medium rules risk.
- Tokenizer changes require reproducible metric-proof notes.
- Sliding-window eval is allowed, but eval wallclock must stay reasonable.
- Mixed precision and low-bit export are low rules risk if exact artifact bytes are reported honestly.

## Minimal Implementation Design In This Repo

Start from the dense `GPT` path and keep the 9-layer / 512-dim architecture.

Use this main candidate configuration:

- `TOKENIZER_PATH=fineweb_2048_bpe.model`
- `VOCAB_SIZE=2048`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `GRAD_CLIP_NORM=0.3`
- `MATRIX_LR=0.02`
- `SCALAR_LR=0.02`
- `TIED_EMBED_LR=0.03`
- `BETA1=0.9`
- `BETA2=0.95`
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `MUON_MOMENTUM_WARMUP_START=0.92`
- `WARMDOWN_ITERS=3000`
- `MLP_MULT=3`
- `FAKE_QUANT_ENABLE=1`
- `FAKE_QUANT_BITS=6`
- `FAKE_QUANT_SCOPE=casted_linear`
- `EXPORT_FORMAT=int6_zstd`
- `EVAL_SEQ_LEN=2048`
- `EVAL_WINDOW_STRIDE=256`
- `EXPORT_FP16_ALLOWLIST=tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight`
- `WEIGHT_DECAY_TOK=0.01`
- `WEIGHT_DECAY_SCALAR=0.01`
- `WEIGHT_DECAY_HEAD=0.01`
- `SWA_ENABLED=1`
- `SWA_START_FRAC=0.5`
- `SWA_EVERY=200`
- `SWA_MAX_CHECKPOINTS=7`
- `SWA_SELECT_MODE=surrogate_roundtrip`
- `SWA_SURROGATE_TOKENS=4194304`

Keep factorized embeddings out of scope for the mainline run. Revisit them only if this stronger dense stack is very close but still barely over the cap.

## Variant Ladder

1. `300s` smoke of the full candidate stack
2. `1200s` long-context `sp2048` control:
   - `int8_zlib`
   - no fake quant
   - `MLP_MULT=2`
   - no SWA
   - no fp16 allowlist
3. `1200s` full candidate stack
4. Eval-only reuse of the final candidate weights at:
   - contiguous
   - `stride64`
   - primary `stride256`

## Metrics To Record

- `final_*_roundtrip_exact val_bpb`
- `final_*_contiguous_roundtrip_exact val_bpb`
- `final_*_stride64_roundtrip_exact val_bpb`
- artifact bytes
- code bytes
- total bytes
- parameter count
- train wallclock
- eval wallclock
- selected export candidate: `last`, `swa_uniform`, or `swa_surrogate`

## Acceptance Criteria

`pass`:

- legal under `16,000,000` bytes
- exact primary score at `stride256` beats `1.17417430` or lands within `0.003`
- `swa_surrogate` is at least neutral relative to `last`

`promote`:

- legal under cap
- exact `stride256` score clearly beats our current plain `sp2048` run
- artifact-aware checkpoint selection contributes measurable value

## Kill Criteria

- exact exported score is more than `0.01` worse than the plain `sp2048` exact
- `int6` still shows a large exact-roundtrip gap even after selective precision and long-context training
- the candidate badly misses the byte cap

## Estimated Engineering Cost

- Medium
- Expected effort: `2-4 engineer days` plus `4xH100` screening time

## Merge Compatibility

- Strong mainline score branch
- Compatible later with artifact-aware checkpoint-selection write-up as a separate PR
- Compatible later with MPK reproduction as a separate second branch
- Do not combine with recurrence, factorized embeddings, or lexical sidecars in the first submission attempt

## References

- [PR #114](https://github.com/openai/parameter-golf/pull/114)
- [PR #122](https://github.com/openai/parameter-golf/pull/122)
- [PR #123](https://github.com/openai/parameter-golf/pull/123)
- [PR #135](https://github.com/openai/parameter-golf/pull/135)
- [PR #98](https://github.com/openai/parameter-golf/pull/98)
- [2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md)
- [2026-03-19_quant_stack_q1_q3_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_quant_stack_q1_q3_draft_notes.md)
