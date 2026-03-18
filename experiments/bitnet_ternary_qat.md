# BitNet-Style Ternary QAT

## Hypothesis

Progressive ternary quantization-aware training on the large 2D transformer matrices can preserve most of the baseline quality while substantially reducing compressed model bytes. This is a direct fit for Parameter Golf because it targets artifact size instead of only training speed.

## Why It May Help This Specific Competition

- The baseline model is currently small enough in compute but tight in artifact bytes.
- Ternary-friendly training can move a large fraction of the model closer to its final export format.
- The competition allows compressed artifacts so long as evaluation is self-contained and reproducible.

## Competition / Rules Risk

- Low rules risk if the student model is fully self-contained and evaluated without any external dependency.
- Medium implementation risk because training throughput may regress even if artifact bytes improve.
- Do not start with native ternary kernels. That is not required for leaderboard relevance in this repo.

## Minimal Implementation Design In This Repo

This experiment starts with **ternary export and fake-quant training**, not native low-bit compute kernels.

Proposed configuration knobs:

- `TERNARY_EXPORT_ENABLE=1`
- `TERNARY_QAT_ENABLE=0|1`
- `TERNARY_QAT_START_FRAC=0.85`
- `TERNARY_SCOPE=transformer_matrices`

Eligible tensors in v1:

- all large 2D matrices inside transformer blocks
- exclude:
  - token embeddings
  - untied output head if present
  - all vectors/scalars/control tensors

Export quantizer definition:

- for each eligible row `w_row`
- compute `alpha = mean(abs(w_row)) + eps`
- quantize with:
  - `q = clamp(round(w_row / alpha), -1, 1)`
- dequantized row is `alpha * q`
- pack ternary codes into 2-bit storage:
  - `00 -> -1`
  - `01 -> 0`
  - `10 -> +1`
  - `11` unused
- store per-row `alpha` in fp16
- compress packed payload with zlib after serialization

QAT definition:

- keep fp32 master weights
- during QAT-enabled forward passes, replace eligible matrices with fake-quantized ternary versions using the rule above
- backward uses a straight-through estimator on the master weights
- start QAT only in the final `15%` of training for v1
- keep optimizer state on the master weights

V1 explicitly avoids:

- native ternary kernels
- ternary embeddings
- ternary control tensors
- ternary-only attention kernels

## Variant Ladder

1. Export-only ternary:
   - train baseline normally
   - export eligible matrices in ternary format only
2. Late-stage QAT:
   - enable fake ternary forward for eligible matrices in final `15%` of training
3. Broader QAT coverage:
   - start QAT at final `30%`
   - optionally include untied head if the model is no longer tied and only if V2 is stable

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- ternary-export exact `val_bpb` if a separate exact log line is added
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- byte reduction vs baseline
- train throughput delta vs baseline

## Acceptance Criteria

`pass`:

- Export-only variant reduces compressed model bytes by at least `20%`
- final quality loss is no worse than `+0.010` `val_bpb`

`promote`:

- Late-stage QAT reduces compressed model bytes by at least `30%`
- final quality loss is no worse than `+0.005` `val_bpb`, or the run beats baseline outright
- training slowdown is no worse than `12%`

Byte targets:

- v1 export-only target: compressed model `<= 12MB`
- v2 late-stage QAT target: compressed model `<= 11MB`
- stretch target for later work: compressed model `<= 10MB`

## Kill Criteria

- export-only ternary loses more than `0.020` `val_bpb`
- late-stage QAT slows training by more than `12%` and does not recover quality
- implementation pressure forces native low-bit kernel work before the export/QAT path is understood

## Estimated Engineering Cost

- Medium to high
- Expected effort: `3-5 engineer days`

## Merge Compatibility

- Excellent candidate to combine with `recurrent_depth`
- Excellent candidate to combine with `masa_weight_sharing`
- Compatible with `data_selection`
- Compatible with `attention_residuals`
- Avoid combining with `distillation` until the ternary path is stable enough to attribute gains

## References

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527)
- [BitNet v2](https://arxiv.org/abs/2504.18415)
- [Official BitNet inference framework](https://github.com/microsoft/BitNet)
