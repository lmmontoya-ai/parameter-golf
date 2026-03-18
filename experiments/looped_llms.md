# Looped LMs

## Hypothesis

A small shared block stack looped multiple times, with explicit loop-step conditioning and eventually adaptive depth, can outperform fixed recurrent depth by encouraging iterative latent computation without proportional parameter growth.

## Why It May Help This Specific Competition

- Like recurrent depth, it buys effective depth with compute instead of bytes.
- Unlike fixed recurrence, it gives a path toward learned depth allocation and test-time compute scaling.
- It is a plausible route to better reasoning-like latent computation under a tiny artifact budget.

## Competition / Rules Risk

- Medium rules risk.
- Fixed-loop versions are straightforward.
- Adaptive-depth or eval-only extra loops are riskier because evaluation-time compute is constrained by the challenge FAQ.
- This experiment should start as `non-record` unless the fixed-loop version is clearly within the evaluation budget.

## Minimal Implementation Design In This Repo

This is a **second-stage shared-depth branch**, not the first shared-depth implementation. Implement it only after `recurrent_depth` exists or is clearly outperformed conceptually.

Proposed configuration knobs:

- `LOOP_ENABLE=1`
- `NUM_SHARED_BLOCKS=2`
- `LOOP_STEPS_TRAIN=4`
- `LOOP_STEPS_EVAL=4`
- `LOOP_ADAPTIVE=0|1`

V1 architecture:

- Use `2` shared physical blocks looped `4` times for effective depth `8`.
- Keep token embedding, final norm, and output head outside the loop.
- Add learned loop-step embeddings `loop_embed[step]` of shape `[LOOP_STEPS_TRAIN, model_dim]`.
- At each loop, add the corresponding loop-step embedding to the hidden state before entering the shared stack.
- Keep unshared step adapters per loop and per physical block for:
  - `q_gain`
  - `attn_scale`
  - `mlp_scale`
  - `resid_mix`
- Keep tokenizer, data, optimizer split, and export path unchanged in v1.

V2 adaptive-depth extension:

- enable `LOOP_ADAPTIVE=1`
- add one halting head `Linear(model_dim, 1)` on the loop state
- allow loop counts in `[2, 6]`
- train with entropy regularization that discourages immediate collapse to always-min or always-max loops

Train/eval policy:

- V1: `LOOP_STEPS_TRAIN == LOOP_STEPS_EVAL == 4`
- V2: training max `6`, eval max `6`
- no eval-only extra loops for leaderboard-oriented runs until wallclock is explicitly measured

Implementation simplicity threshold:

- no separate reasoning tokens
- no external chain-of-thought data
- no custom loop kernel
- no test-time adaptation in this experiment

## Variant Ladder

1. Fixed loops: `2` shared blocks x `4` loops, learned loop-step embeddings
2. Fixed loops: `3` shared blocks x `3` loops, learned loop-step embeddings
3. Adaptive loops: min `2`, max `6`, entropy-regularized halting

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- loop-count distribution when adaptive mode is enabled
- compile stability notes

## Acceptance Criteria

`pass`:

- fixed-loop V1 or V2 reduces compressed model bytes by at least `25%`
- final `val_bpb` no worse than `+0.006` vs baseline
- eval wallclock remains within a clearly documented budget

`promote`:

- fixed-loop variant reduces compressed model bytes by at least `25%`
- final `val_bpb` no worse than `+0.003` vs baseline or better than baseline outright
- eval wallclock increase stays below `15%`

Adaptive-loop variants are `non-record` by default until evaluation wallclock is explicitly measured and deemed safe.

## Kill Criteria

- fixed-loop variants are clearly worse than the simpler `recurrent_depth` experiment on both bytes and quality
- loop conditioning creates compile instability that materially slows iteration
- adaptive loops collapse immediately to always-min or always-max depth and show no measurable benefit

## Estimated Engineering Cost

- High
- Expected effort: `4-6 engineer days`

## Merge Compatibility

- Candidate to combine later with `attention_residuals`
- Candidate to combine later with `bitnet_ternary_qat`
- Do not combine with `recurrent_depth` in the first round; treat them as competing shared-depth families
- Avoid combining with `distillation` until the fixed-loop architecture is stable

## References

- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)
- [Ouro project page](https://ouro-llm.github.io/)
