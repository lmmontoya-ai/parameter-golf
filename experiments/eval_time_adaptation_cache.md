# Eval-Time Adaptation Or Cache Methods

## Hypothesis

Evaluation-time adaptation or cache-based methods can improve perplexity without retraining the base model, and this challenge explicitly permits aggressive evaluation strategies so long as evaluation still finishes under the hardware/time cap.

## Why It May Help This Specific Competition

- The rules allow aggressive evaluation methods.
- This branch can improve quality after the training recipe is already near its limit.
- It is a differentiated path for a second PR or a non-record branch if the mainline model work plateaus.

## Competition / Rules Risk

- Medium rules risk.
- The repo allows aggressive evaluation, but this branch will receive more scrutiny because it changes the evaluation procedure.
- High operational risk because the evaluation budget must still fit under `10` minutes on `8xH100`.

## Minimal Implementation Design In This Repo

Treat this as a second-submission branch, not the first mainline branch.

V1 split:

1. Cache-style adaptation:
   - maintain a bounded local cache or fast weight state during validation
   - no parameter updates

2. Dynamic evaluation:
   - perform tiny updates on a very small parameter subset during eval only
   - first scope:
     - norm gains
     - scalar controls
     - adapter vectors

Recommended config surface:

- `EVAL_ADAPT_ENABLE=1`
- `EVAL_ADAPT_MODE=cache|dynamic`
- `EVAL_ADAPT_SCOPE=norms|scalars`
- `EVAL_ADAPT_LR=...`
- `EVAL_ADAPT_STEPS=1|2`
- `EVAL_ADAPT_BUDGET_SECONDS=...`

Hard constraints:

- eval must remain self-contained
- no external teacher or service
- final wallclock must remain within the competition cap on the real hardware target

V1 deliberately avoids:

- full-model gradient updates at eval
- adaptation schemes that require long warm-start prefixes
- any method that cannot be cleanly disabled for baseline comparison

## Variant Ladder

1. Cache-only sidecar, no gradient updates
2. Dynamic eval on scalar/norm subset only
3. Combined cache + tiny dynamic eval only if each is independently positive

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- eval-only improvement over the same frozen model without adaptation
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- eval-time compute multiplier
- adaptation scope and parameter count

## Acceptance Criteria

`pass`:

- improve final exact `val_bpb` by at least `0.003`
- keep eval wallclock safely under the real cap
- make the evaluation procedure reproducible and deterministic

`promote`:

- improve final exact `val_bpb` by at least `0.005`
- while staying under the real eval budget on `8xH100`
- and without contaminating the base training comparison

`non-record` default:

- until the method is proven robust under the real eval cap, keep it in non-record framing

## Kill Criteria

- eval time is too high
- gains disappear under exact roundtrip evaluation
- method is too brittle to reproduce cleanly

## Estimated Engineering Cost

- Medium to high
- Expected effort: `2-5 engineer days`

## Merge Compatibility

- Apply only after a base dense or recurrent winner exists
- Good candidate as a second submission on top of the strongest frozen model
- Do not combine early with tokenizer or architecture changes

## References

- dynamic evaluation and cache-based adaptation literature
