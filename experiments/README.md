# Experiment Program

This directory is the planning area for isolated experiment designs. These docs are intentionally written before implementation so the repo can compare ideas cleanly against the current baseline and avoid premature combination of unvalidated changes.

## Program Principles

- Experiment 0 is baseline reproduction on `8xH100`. No architecture experiment should be treated as meaningful until the stack reproduces the current baseline within the agreed tolerance.
- The first round is intentionally isolated and attributable. Each experiment is evaluated on the fixed baseline stack unless the doc explicitly says otherwise.
- Data, distillation, architecture, and export/compression are separate axes. Do not collapse them into one mixed branch.
- `Recurrent Depth` and `Looped LMs` are related but not the same:
  - `Recurrent Depth` is the minimal shared-depth retrofit on the current stack.
  - `Looped LMs` is the broader adaptive-loop family that builds on shared-depth ideas.
- `BitNet` in this repo should start as a ternary/QAT/export experiment, not as a native ternary-kernel rewrite.
- `Distillation` is rules-sensitive and defaults to `non-record-first`.

## Experiment Order

1. Baseline reproduction
2. Isolated low-coupling experiments on the fixed baseline:
   - attention residuals
   - MASA weight sharing
   - data selection
   - BitNet/QAT export path
3. Isolated shared-depth family experiments:
   - recurrent depth
   - looped LMs
4. Rules-risk incubators:
   - distillation
5. Combination phase only after isolated winners are known

## Combination Rules

- Never combine two unvalidated ideas at once.
- Combine only experiments that individually beat the baseline under the same evaluation setup.
- Treat `recurrent_depth` and `looped_llms` as competing families first, not automatic merge candidates.
- Treat `data_selection` as a separate axis and combine it only after an architecture winner exists.
- Treat `distillation` as optional and non-blocking for leaderboard-oriented architecture work.
- Do not start a combination branch until:
  - `00_baseline_reproduction_8xh100.md` has a pass condition and at least one successful reproduction run.
  - at least two isolated experiments have clear positive results and compatible mechanisms.

## Standard Experiment Template

Every experiment doc in this directory should use the same sections:

1. Hypothesis
2. Why it may help this specific competition
3. Competition / rules risk
4. Minimal implementation design in this repo
5. Variant ladder
6. Metrics to record
7. Acceptance criteria
8. Kill criteria
9. Estimated engineering cost
10. Merge compatibility

The design should be decision-complete. The implementer should not need to choose the experiment scope, first variant, success threshold, or merge gate.

## Common Metrics To Record

Every serious run should record all of the following:

- `final_int8_zlib_roundtrip_exact val_bpb`
- raw pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- reproducibility notes

Recommended additional logging:

- train step at wallclock stop
- validation cadence used
- train shards count
- tokenizer path / tokenizer family
- parameter count
- any eval-time compute multiplier beyond the training-time model definition

Current development caveat:

- while active experiment code lives outside `train_gpt.py`, the runner's logged `code bytes` undercounts true submission size
- treat that number as development-only until the exact runnable snapshot is frozen into `records/`

## Promotion Policy

Use these promotion labels when a doc refers to whether an experiment should advance:

- `pass`: good enough to justify another isolated run or the next planned variant
- `promote`: good enough to become a combination candidate
- `non-record`: interesting, but not ready for leaderboard-oriented development
- `kill`: stop spending time on this branch unless a new premise changes

Unless a doc states otherwise, promotion means:

- the experiment beats the baseline on `final_int8_zlib_roundtrip_exact val_bpb` or delivers a material byte reduction with acceptable quality loss
- the artifact remains under `16,000,000` bytes
- the result is repeatable enough that a second run is unlikely to reverse the conclusion

## Backlog

These ideas should be tracked, but not promoted to full design docs in the first round:

- TTT-E2E / test-time training
- Paired Head Attention
- Partial Key Offset
- Partitioned Hyperconnections
- Standalone tokenizer redesign beyond the combined tokenizer+embedding branch
- Compression-only export improvements beyond the baseline int8 path

Reason for backlog status:

- they are either less direct fits for Parameter Golf,
- overlap with higher-priority experiments,
- or introduce more evaluation/rules complexity than the first round should absorb.

## First-Round Docs

- [00_baseline_reproduction_8xh100.md](./00_baseline_reproduction_8xh100.md)
- [attention_residuals.md](./attention_residuals.md)
- [recurrent_depth.md](./recurrent_depth.md)
- [looped_llms.md](./looped_llms.md)
- [bitnet_ternary_qat.md](./bitnet_ternary_qat.md)
- [data_selection.md](./data_selection.md)
- [distillation.md](./distillation.md)
- [masa_weight_sharing.md](./masa_weight_sharing.md)

## Expansion Docs

These are second-wave branches that became interesting after the first recurrent and infrastructure passes:

- [compression_aware_dense_transformer.md](./compression_aware_dense_transformer.md)
- [larger_tokenizer_factorized_embeddings.md](./larger_tokenizer_factorized_embeddings.md)
- [early_layer_recurrence_width.md](./early_layer_recurrence_width.md)
- [low_rank_ffn_factorization.md](./low_rank_ffn_factorization.md)
- [hybrid_attention_recurrent_layers.md](./hybrid_attention_recurrent_layers.md)
- [multi_token_prediction.md](./multi_token_prediction.md)
- [eval_time_adaptation_cache.md](./eval_time_adaptation_cache.md)
- [frontier_decomposition_pr89.md](./frontier_decomposition_pr89.md)
- [sp2048_dense_frontier_candidate.md](./sp2048_dense_frontier_candidate.md)
