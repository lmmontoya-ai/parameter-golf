# Low-Rank FFN Or Projection Factorization

## Hypothesis

Low-rank FFN parameterization can free a large number of bytes with less quality loss than an equally aggressive change to attention. The saved parameters can then be reinvested into width, tokenizer, or export headroom.

## Why It May Help This Specific Competition

- FFNs are a major parameter sink.
- Low-rank structure is directly useful for artifact-size reduction.
- FFN factorization is simpler and more targeted than rewriting the whole model to be low-rank everywhere.

## Competition / Rules Risk

- Low rules risk.
- Medium implementation risk because training low-rank matrices from scratch can be sensitive to initialization and rank choice.

## Minimal Implementation Design In This Repo

Start with FFNs only.

For each FFN projection:

- `W_up = A_up @ B_up`
- `W_down = A_down @ B_down`

Keep:

- attention dense in v1
- tokenizer fixed
- export path fixed
- optimizer split unchanged

Recommended config surface:

- `LOW_RANK_FFN_ENABLE=1`
- `LOW_RANK_FFN_RATIO=0.5|0.375|0.25`
- `LOW_RANK_SCOPE=mlp_only`
- `LOW_RANK_WIDTH_REINVEST=0|1`

V1 rollout:

1. factorize FFNs only, no width reinvestment
2. if quality loss is modest, increase width while staying near the same byte budget

Initialization choice:

- initialize factor pairs so the product matches the baseline variance scale
- do not use SVD warm starts in v1

## Variant Ladder

1. FFN rank ratio `0.5`, no width reinvestment
2. FFN rank ratio `0.375`, no width reinvestment
3. best ratio + width reinvestment under the same byte budget

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- FFN parameter reduction
- width reinvestment if used

## Acceptance Criteria

`pass`:

- reduce artifact bytes by at least `10%`
- with no worse than `+0.008 val_bpb` quality loss

`promote`:

- match or beat the current best recurrent quality-per-byte frontier
- or improve final exact `val_bpb` at roughly equal total bytes after width reinvestment

## Kill Criteria

- FFN-only factorization loses more than `0.010 val_bpb`
- width reinvestment cannot recover the quality loss
- implementation spills into all-matrix low-rank pretraining before FFN-only behavior is understood

## Estimated Engineering Cost

- Medium
- Expected effort: `2-4 engineer days`

## Merge Compatibility

- Strong candidate to combine later with `compression_aware_dense_transformer`
- Strong candidate to combine later with `early_layer_recurrence_width`
- Strong candidate to combine later with `larger_tokenizer_factorized_embeddings`
- Avoid combining first with `bitnet_ternary_qat` until the FFN-only tradeoff is known

## References

- recent low-rank FFN and low-rank pretraining literature motivating the branch
