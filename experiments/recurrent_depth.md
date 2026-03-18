# Recurrent Depth

## Hypothesis

Sharing the heavy transformer blocks across logical depth can preserve or improve compression quality per byte by trading extra compute for fewer stored parameters. This is one of the cleanest architecture bets for a byte-capped competition.

## Why It May Help This Specific Competition

- Parameter Golf rewards effective depth bought with compute instead of bytes.
- The baseline has no cross-layer weight sharing.
- Recurrent depth directly attacks the dominant cost center: repeated dense matrices.

## Competition / Rules Risk

- Low to medium rules risk.
- No tokenizer or dataset changes are required.
- Eval-time compute must stay within the challenge limit, so v1 should not rely on extra recurrence at evaluation beyond what is used in training.

## Minimal Implementation Design In This Repo

First implementation target: **fixed recurrence with shared heavy blocks and unshared lightweight step adapters**.

Keep these baseline pieces unchanged in v1:

- tokenizer
- dataset
- export path
- overall effective logical depth of `9`
- evaluation procedure

Proposed configuration knobs:

- `RECURRENT_ENABLE=1`
- `NUM_SHARED_BLOCKS=3`
- `RECURRENT_STEPS=3`
- `RECURRENT_EVAL_STEPS=3`

V1 architecture:

- Replace `9` unique transformer blocks with `3` physical blocks reused `3` times each, for effective depth `9`.
- Logical schedule:
  - layers `0,1,2,0,1,2,0,1,2`
- Keep these weights shared across recurrence:
  - `c_q`
  - `c_k`
  - `c_v`
  - attention output projection
  - MLP input projection
  - MLP output projection
  - internal RMSNorm modules
- Move these lightweight parameters to **unshared logical-step adapters**:
  - `q_gain`
  - `attn_scale`
  - `mlp_scale`
  - `resid_mix`
- Keep skip weights defined at logical depth positions, not physical block positions.

Train/eval policy:

- `RECURRENT_STEPS=3` and `RECURRENT_EVAL_STEPS=3` in v1
- no eval-only extra loops
- no adaptive halting

Optimizer policy:

- physical shared matrices stay in the matrix optimizer path
- step-adapter tensors stay in the scalar optimizer path
- token embedding / head policy remains baseline

Implementation simplicity threshold:

- no custom recurrent kernel
- no learned halting
- no step-dependent tokenizer or data changes

## Variant Ladder

1. `3` shared blocks x `3` recurrence, fixed train/eval loops
2. `2` shared blocks x `4` recurrence plus one final unique readout block
3. `1` shared block x `8` recurrence plus one final unique readout block

Do not move past variant 1 if the quality drop is already too large.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- parameter count reduction vs baseline
- recurrent compile overhead notes

## Acceptance Criteria

`pass`:

- compressed model bytes reduced by at least `20%`
- final `val_bpb` no worse than `+0.005` vs baseline
- total artifact bytes remain under `16,000,000`

`promote`:

- compressed model bytes reduced by at least `25%`
- final `val_bpb` no worse than `+0.003` vs baseline or better than baseline outright
- no more than `15%` training slowdown

## Kill Criteria

- variant 1 loses more than `0.010` `val_bpb`
- recurrence increases train wallclock by more than `15%` while byte savings stay below `20%`
- implementation requires major optimizer redesign beyond the step-adapter split

## Estimated Engineering Cost

- Medium to high
- Expected effort: `2-4 engineer days`

## Merge Compatibility

- Strong candidate to combine with `attention_residuals`
- Strong candidate to combine with `bitnet_ternary_qat`
- Potentially compatible with `masa_weight_sharing`, but only after both work in isolation
- Competes directly with `looped_llms` in the first round and should not be automatically merged with it

## References

- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [Recurrent depth reference repository](https://github.com/seal-rg/recurrent-pretraining)
