# Compression-Aware Dense Transformer

## Hypothesis

The highest-EV dense branch is to optimize directly for the exported artifact instead of the bf16 checkpoint. If the model is trained to reduce the post-export roundtrip gap, a plain dense Transformer may recover more leaderboard score than a flashier architecture change.

## Why It May Help This Specific Competition

- The repo is scored on `final_int8_zlib_roundtrip_exact val_bpb`, not raw checkpoint quality.
- The official baseline already shows a measurable gap between pre-export and post-export score.
- Dense models keep the training stack simple, stable, and easy to compare against the public baseline.
- A dense branch can be merged later with tokenizer, low-rank, or MTP work.

## Competition / Rules Risk

- Low rules risk.
- No tokenizer, dataset, or evaluation-protocol change is required.
- Medium implementation risk because export-aware training can overfit to the current exporter in brittle ways.

## Minimal Implementation Design In This Repo

Start from the current dense baseline path, not the recurrent path.

Keep fixed:

- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- tokenizer
- train/val split
- current export format
- Muon + Adam split

Add three tightly scoped mechanisms:

1. Learning-rate retune:
   - expose a dense-baseline retune track before changing the loss
   - treat this as the control for the rest of the branch

2. Export-aware cooldown:
   - in the final `10-15%` of training, periodically evaluate a cheap local export/deexport surrogate
   - optimize the main CE loss plus a small penalty on the difference between live weights and their exported surrogate

3. Mild outlier / compression penalty:
   - add a weak penalty that discourages export-sensitive weight outliers on selected large matrices
   - first target:
     - attention output projections
     - MLP output projections
     - embedding / head path only if later evidence says they dominate the roundtrip gap

Recommended config surface:

- `EXPORT_AWARE_ENABLE=0|1`
- `EXPORT_AWARE_COOLDOWN_FRAC=0.10`
- `EXPORT_AWARE_SURROGATE=int8_rowwise`
- `EXPORT_AWARE_PENALTY=0.01`
- `OUTLIER_PENALTY=0.0-1e-4`
- `EXPORT_SENSITIVE_SCOPE=attn_out,mlp_out`
- `EXPORT_MIXED_PRECISION_ALLOWLIST=...`

V1 simplifications:

- do not change the final export format yet
- do not add a second optimizer
- do not make the surrogate exactly identical to the final exporter if that would make training unstable
- keep all changes inside the existing dense path

## Variant Ladder

1. Dense LR retune only
2. LR retune + export-aware cooldown
3. Add mild outlier penalty on selected tensors
4. Add limited mixed-precision export allowlist for the most sensitive tensors if byte cost stays acceptable

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- roundtrip gap: pre-export minus post-export `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- number of tensors in export-sensitive allowlist

## Acceptance Criteria

`pass`:

- improve dense-baseline final exact `val_bpb` by at least `0.002`
- or reduce the roundtrip gap by at least `25%`
- with no more than `5%` training slowdown

`promote`:

- improve dense-baseline final exact `val_bpb` by at least `0.003`
- keep artifact bytes within `16,000,000`
- demonstrate that the gain survives the final exporter, not just the bf16 checkpoint

## Kill Criteria

- no measurable reduction in roundtrip gap after the cooldown variant
- apparent gain exists only before export
- mixed-precision allowlist buys too little quality for too many bytes

## Estimated Engineering Cost

- Medium
- Expected effort: `2-4 engineer days`

## Merge Compatibility

- Excellent candidate to combine later with `multi_token_prediction`
- Good candidate to combine later with `larger_tokenizer_factorized_embeddings`
- Good candidate to combine later with `bitnet_ternary_qat`
- Keep isolated from recurrent-depth work until the dense branch is clearly positive

## References

- local baseline and roundtrip behavior in `records/track_10min_16mb/2026-03-17_NaiveBaseline`
- local dense proxy reproduction in `docs/research/2026-03-18_4xh100_baseline_proxy.md`
