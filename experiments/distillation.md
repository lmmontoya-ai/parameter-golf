# Distillation

## Hypothesis

Soft targets can improve the student model's compression quality under the same artifact budget, especially when the student is very small. However, distillation is unusually sensitive to the challenge's "external compute" spirit and should be handled conservatively.

## Why It May Help This Specific Competition

- Small students often benefit more from dark knowledge than large models do.
- Distillation can improve quality without increasing final artifact bytes if the teacher is only a training-time aid.
- It may become especially valuable for shared-depth or ternary-constrained students.

## Competition / Rules Risk

- Medium to high risk.
- Distillation is not explicitly forbidden by the repo docs, but it may raise concerns under the challenge's external-compute clause.
- Default policy for this doc: **non-record-first**.
- No distilled experiment should be framed as leaderboard-ready until the teacher setup is clearly within the spirit of the challenge.

Risk labels:

- `low-risk`: online self-distillation from an EMA teacher produced inside the same run
- `medium-risk`: offline self-distillation from a longer run of the same architecture
- `high-risk`: external teacher from a different architecture or a much larger model

All variants must obey:

- student-only final artifact
- no eval-time teacher dependence
- reproducible student training path

## Minimal Implementation Design In This Repo

### V1: Online EMA Self-Distillation

Proposed configuration knobs:

- `DISTILL_ENABLE=1`
- `DISTILL_MODE=ema`
- `DISTILL_EVERY=4`
- `DISTILL_WEIGHT=0.05`
- `DISTILL_TEMPERATURE=2.0`
- `DISTILL_START_FRAC=0.10`
- `DISTILL_EMA_DECAY=0.999`

Implementation:

- maintain an EMA copy of the student weights
- every `4` optimizer steps, run one extra no-grad teacher forward on the same batch
- add KL divergence between student logits and teacher logits
- use distillation only after the first `10%` of training
- keep token loss as the primary objective

This is the only variant that can later be reconsidered for leaderboard suitability.

### V2: Offline Same-Architecture Teacher

- teacher is a stronger checkpoint of the same model family
- teacher logits or checkpoints are produced offline
- this is `non-record` by default

### V3: External Teacher

- teacher is a different or larger model family
- this is `non-record` and high risk by default

## Variant Ladder

1. V1 online EMA self-distillation
2. V2 offline same-architecture teacher
3. V3 external teacher

Do not start with V2 or V3.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- extra teacher compute overhead
- teacher type and provenance notes

## Acceptance Criteria

V1 `pass`:

- improves final `val_bpb` by at least `0.002`
- student artifact remains fully self-contained
- train wallclock overhead stays below `20%`

V1 `promote`:

- improves final `val_bpb` by at least `0.003`
- result remains reproducible without any eval-time teacher
- rules risk is judged acceptable after explicit review

V2 and V3 remain `non-record` unless the challenge guidance is clarified.

## Kill Criteria

- V1 adds more than `20%` train wallclock overhead without a measurable gain
- attribution becomes ambiguous because the teacher setup is too detached from the student run
- any branch requires teacher state at evaluation time

## Estimated Engineering Cost

- V1: Medium
- V2/V3: Medium to high
- Expected effort: `2-5 engineer days`

## Merge Compatibility

- Do not treat distillation as part of the first leaderboard-oriented architecture wave
- Consider combining with:
  - `recurrent_depth`
  - `looped_llms`
  - `bitnet_ternary_qat`
  only after those branches are stable in isolation
- Avoid combining with `data_selection` until the data effect is independently understood

## References

- Local challenge FAQ on external compute in `README.md`
- External benchmark precedent from `slowrun`, where distillation helped in an unlimited-compute setting
