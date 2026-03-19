# Early-Layer Recurrence Then Width

## Hypothesis

Recur only early or middle layers, keep later layers unique, and spend the saved parameters on width. This should preserve more language-model quality than looping the whole stack while still exploiting recurrence for byte efficiency.

## Why It May Help This Specific Competition

- Recent recurrent results in this repo show that shared-depth ideas can get close to the dense baseline.
- Late layers are more likely to benefit from remaining unique.
- If recurrence savings are concentrated early, the recovered parameter budget can be reinvested into width where it most directly improves perplexity.

## Competition / Rules Risk

- Low rules risk.
- Medium implementation risk because this is a new recurrent architecture family, not just a retune of the current latent recurrent branch.

## Minimal Implementation Design In This Repo

This is a separate branch from the current `RecurrentGPT`.

Architecture shape for v1:

- unique input stem
- shared early block group repeated `R` times
- unique upper stack
- learned recurrence scale on the repeated group output

Concrete first design:

- `EARLY_RECURRENCE_ENABLE=1`
- `EARLY_SHARED_BLOCKS=2`
- `EARLY_REPEATS=3`
- `UNIQUE_UPPER_BLOCKS=5`
- `RECURRENCE_SCALE_ENABLE=1`
- width target chosen so total params stay at or below the current best recurrent `w576` run

Implementation choices:

- keep current attention and MLP internals
- do not use full latent-state reinitialization each loop
- repeat only the lower stack over the same hidden state
- use a learned scalar or vector recurrence scale after each repeat
- keep the output stack unique and non-recurrent

V1 explicitly avoids:

- looping the entire stack
- adaptive halting
- eval-only extra repeats
- external memory or cache tricks

## Variant Ladder

1. `2` shared early blocks repeated `3` times, `5` unique upper blocks
2. same topology with width increased to spend the saved bytes
3. shift recurrence to the middle stack if the early-only version underperforms

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- parameter count
- repeat-group vs unique-stack parameter split

## Acceptance Criteria

`pass`:

- beat the current narrower recurrent branch
- or match the current best recurrent quality with at least `10%` fewer bytes

`promote`:

- improve on the current best recurrent final exact `val_bpb` by at least `0.005`
- keep total artifact bytes under the cap with `>300KB` headroom
- remain stable through a full `4xH100 -> 1200s` proxy run

## Kill Criteria

- early-only recurrence is consistently worse than the current full recurrent branch at equal bytes
- the unique upper stack has to become so large that recurrence savings disappear
- recurrence scales add complexity without helping quality

## Estimated Engineering Cost

- Medium to high
- Expected effort: `3-5 engineer days`

## Merge Compatibility

- Competes directly with `recurrent_depth` at first
- Good later merge candidate with `bitnet_ternary_qat`
- Good later merge candidate with `low_rank_ffn_factorization`
- Do not combine with `compression_aware_dense_transformer` until the architecture itself is positive

## References

- local recurrent branch results in `docs/research/2026-03-18_recurrent44_deeper223_real_train_proxy_draft_notes.md`
- local widened recurrent branch results in `docs/research/2026-03-18_recurrent44_deeper223_w576_real_train_proxy_draft_notes.md`
