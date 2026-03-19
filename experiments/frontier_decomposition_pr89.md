# Frontier Decomposition PR89

## Hypothesis

The right way to learn from the current dense frontier is to decompose it into attributable pieces instead of copying a full stack blindly. If we isolate optimizer/schedule, sliding-window evaluation, quantization-aware training/export, and SWA separately, we can identify which parts are robust in this repo and where an orthogonal extension can still add value.

## Why It May Help This Specific Competition

- Recent strong PRs are converging on dense, compression-aware, eval-aware systems rather than large backbone changes.
- The biggest visible gain in the current frontier may come from evaluation and export behavior rather than raw checkpoint quality alone.
- A decomposed reproduction is a stronger research artifact than a monolithic recipe because it shows which parts actually move `final_*_roundtrip_exact val_bpb`.
- Once the non-tokenizer stack is understood, tokenizer plus factorized tied embeddings is the cleanest orthogonal gap to attack.

## Competition / Rules Risk

- Low to medium rules risk.
- Sliding-window eval is allowed but must still fit under the challenge’s evaluation-time constraints.
- Tokenizer changes require explicit metric-correctness proof and will receive extra scrutiny.
- Quantization-aware export changes must stay honest about artifact bytes and the exact final roundtrip metric.

## Minimal Implementation Design In This Repo

Keep the current dense baseline architecture as the fixed control:

- `sp1024`
- contiguous eval
- current dense baseline depth/width
- current tied-embedding setup
- no SWA
- no fake quant
- vanilla Muon
- `MLP_MULT=2`

Then stage the work in four isolated phases:

1. Optimizer / schedule:
   - add `MUON_VARIANT=muon|normuon`
   - isolate schedule retune from NorMuon itself

2. Eval:
   - reuse the new `EVAL_SEQ_LEN` / `EVAL_WINDOW_STRIDE` sliding-window path
   - measure pure eval gain from the same frozen checkpoint

3. Quantization-aware training / export:
   - add `FAKE_QUANT_ENABLE`, `FAKE_QUANT_BITS`, `EXPORT_FORMAT`, and `EXPORT_FP16_ALLOWLIST`
   - support `int6_zstd` alongside the existing `int8_zlib` path
   - keep fake quant scoped to `CastedLinear`

4. SWA:
   - add `SWA_ENABLED`, `SWA_START_FRAC`, `SWA_EVERY`, `SWA_MAX_CHECKPOINTS`, `SWA_SELECT_MODE`
   - average late checkpoints in fp32 and evaluate both `last` and `swa`

Only after those pieces are understood:

- rebuild `sp2048` / `sp4096`
- reuse `FACTOR_EMBED_ENABLE` / `FACTOR_EMBED_DIM`
- test tokenizer plus factorized tied embeddings against the best non-tokenizer pre-SWA stack

## Variant Ladder

1. `C0` dense control
2. `A1` schedule retune only
3. `A2` schedule retune + NorMuon
4. `B*` sliding-window eval on the frozen `A2` checkpoint
5. `C1/C2/C3` quant-aware export stack on top of `A2`
6. `D1` SWA on the best contiguous pre-SWA stack
7. `T1-T5` tokenizer + factorized embedding branch
8. `Z1/Z2` combine the best tokenizer branch with SWA and the best sliding-window setting

## Metrics To Record

- `final_*_roundtrip_exact val_bpb`
- contiguous post-export `val_bpb`
- sliding-window post-export `val_bpb`
- eval wallclock
- compressed model bytes
- code bytes
- total artifact bytes
- parameter count
- `tokens_per_byte` when tokenizer changes
- exact run command and hardware lane

## Acceptance Criteria

`pass`:

- `A2` beats `C0` by at least `0.005`
- best sliding-window variant beats contiguous by at least `0.010` with acceptable eval cost
- best quant-aware variant improves contiguous post-export score by at least `0.005` over `A2`, or halves the contiguous quant gap
- SWA improves the contiguous post-export score by at least `0.002`
- tokenizer/factor branch improves contiguous post-export score by at least `0.004` over the best `sp1024` pre-SWA stack

`promote`:

- combined winner improves the internal `sp1024` decomposed stack by at least `0.008`
- or clearly shifts the quality/byte frontier enough to justify `8xH100` confirmation

## Observed Results

- [2026-03-19_frontier_decomposition_a1_a2_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_frontier_decomposition_a1_a2_draft_notes.md):
  - `A1` passed
  - `A2` NorMuon was negative and was killed
- [2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md):
  - `stride64` sliding-window eval improved the promoted dense control to `1.18718097`
- [2026-03-19_quant_stack_q1_q3_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_quant_stack_q1_q3_draft_notes.md):
  - `Q1` (`int6_zstd + STE`) was a clear negative on the promoted metric
  - exact promoted score: `1.23889609`
  - delta vs control: `+0.05171512 val_bpb`
  - `Q2/Q3/S1/S2` were not promoted

## Kill Criteria

- schedule retune helps but NorMuon does not
- sliding-window gain is too small relative to eval cost
- quant-aware branch degrades post-export quality or burns too many bytes
- SWA gain is negligible
- tokenizer path fails metric-proof or loses the tokenizer gain back in bits/token or embedding bytes

## Estimated Engineering Cost

- High
- Expected effort: `4-7 engineer days` plus `4xH100` screening time

## Merge Compatibility

- Strong mainline dense branch
- Natural predecessor to `larger_tokenizer_factorized_embeddings`
- Natural predecessor to more principled mixed-precision export work
- Should stay separate from recurrence, val-only, and distillation while results are being attributed

## References

- [PR #89](https://github.com/openai/parameter-golf/pull/89)
- [compression_aware_dense_transformer.md](./compression_aware_dense_transformer.md)
- [larger_tokenizer_factorized_embeddings.md](./larger_tokenizer_factorized_embeddings.md)
- [eval_time_adaptation_cache.md](./eval_time_adaptation_cache.md)
