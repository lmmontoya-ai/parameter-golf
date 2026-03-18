# MASA-Style Weight Sharing

## Hypothesis

Attention projection matrices contain cross-layer redundancy that can be exploited through shared matrix atoms. If each layer's attention weights are represented as linear combinations of a small shared dictionary, the model can reduce attention parameters substantially without needing a full architecture rewrite.

## Why It May Help This Specific Competition

- This is a direct parameter-reduction strategy, not just a training-speed idea.
- Attention matrices are a large fraction of the baseline's dense weight budget.
- Weight sharing should compose naturally with export compression and shared-depth work.

## Competition / Rules Risk

- Low rules risk.
- No tokenizer or evaluation changes are required.
- Medium implementation risk because it changes the parameterization of core attention layers.

## Minimal Implementation Design In This Repo

Scope v1 to attention projections only.

Keep these baseline pieces unchanged:

- MLP weights
- tokenizer
- dataset
- export path
- skip-weight structure

Proposed configuration knobs:

- `MASA_ENABLE=1`
- `MASA_NUM_ATOMS=3`
- `MASA_SCOPE=all_qkvo`

V1 parameterization:

- For each projection family `Q`, `K`, `V`, and `O`, replace the per-layer weight tensors with:
  - `3` shared full-size atom matrices
  - per-layer scalar coefficients for the `3` atoms
- Realized weight for a given layer and projection family:
  - `W_l = c_l0 * A_0 + c_l1 * A_1 + c_l2 * A_2`

This gives a `66.7%` reduction in attention projection matrices across layers when compared to storing a unique full matrix for each layer.

Implementation decisions:

- use separate atom banks for `Q`, `K`, `V`, and `O`
- initialize atoms from the baseline-style weight init
- initialize per-layer coefficients so that one atom dominates initially and the starting behavior is close to a normal layer-specific weight
- train atoms with the matrix optimizer path
- train coefficients with the scalar optimizer path

Implementation simplicity threshold:

- no distillation requirement
- no low-rank factorization inside MASA v1
- no MLP sharing in the first pass
- no custom kernel work

## Variant Ladder

1. `MASA_NUM_ATOMS=3`, all `Q/K/V/O`
2. `MASA_NUM_ATOMS=4`, all `Q/K/V/O`
3. split dictionaries by stack half:
   - one atom bank for layers `0-3`
   - one atom bank for layers `4-8`

Only move past variant 1 if quality loss is clearly the limiting factor.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- parameter count reduction in attention projections
- coefficient sparsity / concentration notes

## Acceptance Criteria

`pass`:

- compressed model bytes reduced by at least `15%`
- final `val_bpb` no worse than `+0.005` vs baseline

`promote`:

- compressed model bytes reduced by at least `20%`
- final `val_bpb` no worse than `+0.003` vs baseline or better than baseline outright
- no more than `10%` training slowdown

## Kill Criteria

- variant 1 loses more than `0.010` `val_bpb`
- atom-based sharing becomes effectively dense again because variant 2 or 3 is required just to recover the baseline
- implementation pressure expands into a broader architecture rewrite

## Estimated Engineering Cost

- Medium
- Expected effort: `2-4 engineer days`

## Merge Compatibility

- Strong candidate to combine with `bitnet_ternary_qat`
- Strong candidate to combine with `recurrent_depth`
- Compatible later with `data_selection`
- Do not combine with `attention_residuals` until each is independently positive

## References

- [Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning](https://arxiv.org/abs/2508.04581)
