# Multi-Token Prediction

## Hypothesis

Multi-token prediction can improve sample efficiency enough to matter under the `10` minute training budget, while leaving the final exported artifact unchanged because the auxiliary heads are stripped before export.

## Why It May Help This Specific Competition

- This is a training-only add-on with potentially low artifact cost.
- If it helps, it composes with both dense and recurrent branches.
- It is a clean way to spend training compute more efficiently without changing the main model definition at inference time.

## Competition / Rules Risk

- Low rules risk if auxiliary heads are removed before export and the artifact accounting is explicit.
- Medium empirical risk because smaller models may benefit less than large-model reports suggest.

## Minimal Implementation Design In This Repo

Start as a training-only add-on on top of an existing architecture winner.

Design:

- add `K` auxiliary heads that predict tokens `t+2 ... t+K+1`
- share the trunk
- compute weighted CE losses during training
- strip auxiliary heads before final export and final roundtrip evaluation

Recommended config surface:

- `MTP_ENABLE=1`
- `MTP_K=2|4`
- `MTP_LOSS_WEIGHT=0.3`
- `MTP_STRIP_AT_EXPORT=1`

V1 scope:

- apply first to the dense baseline or the current best recurrent branch
- use a single shared hidden state, not a separate speculative decoder
- keep export code explicit about excluding auxiliary heads

## Variant Ladder

1. `MTP_K=2` on the dense baseline
2. `MTP_K=2` on the best recurrent branch
3. `MTP_K=4` only if `K=2` is clearly positive

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- training-only parameter overhead before stripping
- stripped export bytes after removing MTP heads

## Acceptance Criteria

`pass`:

- improve final exact `val_bpb` by at least `0.002`
- with no more than `5%` train-step slowdown
- and no export artifact penalty after stripping heads

`promote`:

- improve final exact `val_bpb` by at least `0.003`
- on either the dense or best recurrent branch
- with clear stripped-artifact accounting

## Kill Criteria

- no measurable gain after `K=2`
- auxiliary heads increase training cost too much for the observed improvement
- stripping heads leaves ambiguous or error-prone export accounting

## Estimated Engineering Cost

- Low to medium
- Expected effort: `1-3 engineer days`

## Merge Compatibility

- Excellent candidate to combine later with `compression_aware_dense_transformer`
- Excellent candidate to combine later with `larger_tokenizer_factorized_embeddings`
- Good candidate to combine later with the best recurrent branch
- Keep isolated from eval-time adaptation work

## References

- multi-token prediction literature and adjacent benchmark reports
