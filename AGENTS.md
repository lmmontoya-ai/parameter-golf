# AGENTS.md

This repository is for OpenAI's Parameter Golf challenge. Future work in this repo should be judged against the competition contract first, not just local code quality or raw validation loss.

## Objective

Optimize the best submission artifact that:

- fits under `16,000,000` total bytes
- is reproducible on `8xH100`
- satisfies the 10-minute track constraints
- minimizes tokenizer-agnostic `val_bpb` on the fixed FineWeb validation split

The primary score to optimize in this codebase is the post-export metric:

- `final_int8_zlib_roundtrip_exact val_bpb`

Do not treat raw `val_loss` or pre-quantization `val_bpb` as the final objective.

## Hard Competition Constraints

- Artifact size is `code bytes + compressed model bytes`.
- The size cap is decimal `16,000,000` bytes, not `16 MiB`.
- No external downloads, network calls, or training-data access are allowed during evaluation.
- The submission artifact must be self-contained and reproducible.
- Leaderboard submissions must reproducibly run under the challenge's `10 minute on 8xH100` requirement.
- Evaluation-time methods are also constrained by the challenge FAQ. Do not assume unlimited test-time compute for a leaderboard run.
- If tokenizer or dataset logic changes, the submission must prove that `val_bpb` is still computed correctly.

## What The Baseline Script Actually Measures

`train_gpt.py` uses `MAX_WALLCLOCK_SECONDS` to cap the training loop, but the script's internal timer is not full end-to-end runtime:

- compile warmup is excluded from the measured training timer
- validation time is excluded from the measured training timer
- post-training serialization, quantization, reload, and final validation happen after the timed loop

This is a local implementation detail, not permission to ignore the competition's evaluation-time limit.

## Scoring And Evaluation Rules

- The fixed validation target is the full `fineweb_val_*` split.
- The challenge score is tokenizer-agnostic `val_bpb`, not plain token loss.
- In this repo, the competition-relevant printed line is `final_int8_zlib_roundtrip_exact`.
- Every serious experiment should record:
  - `final_int8_zlib_roundtrip_exact val_bpb`
  - total artifact bytes
  - code bytes
  - compressed model bytes
  - wallclock details

If an experiment improves raw model quality but gets worse after quantized roundtrip or exceeds the byte budget, it is not a winning result.

## Submission Process

Record-track submissions should be made as a pull request that only adds a new folder under the appropriate `records/` track.

Current tracks in this repo:

- `records/track_10min_16mb/`
- `records/track_non_record_16mb/`

Each submission folder should include:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- any additional dependencies needed to run the submission from inside that record folder

Broken or non-runnable submission scripts should be treated as invalid.

## New SOTA Requirements

For a new leaderboard record:

- beat the current SOTA by at least `0.005` nats
- provide enough logs to support the required statistical confidence (`p < 0.01`) when applicable
- reproduce under the track's `8xH100` time limit

The README notes that the `0.005`-nat evidence requirement is waived for systems-only speedups that do not change the ML.

## Repo Working Norms

- Treat `train_gpt.py` and `train_gpt_mlx.py` as launch points and baselines, not as the only place for competitive ideas.
- Keep the strongest experimental or record-specific code in the relevant `records/` folder.
- Preserve deterministic, easy-to-audit evaluation behavior unless there is a strong reason to change it.
- When changing anything that affects the tokenizer, dataset, or metric calculation, document the correctness argument explicitly.
- Always reason about compression and export format together with model design.

## Useful Local References

- `README.md`: challenge rules, FAQ, and submission requirements
- `docs/BASELINE.md`: detailed explanation of the current baseline script and record
- `data/README.md`: dataset and tokenizer export workflow

When in doubt, optimize for:

1. lower `final_int8_zlib_roundtrip_exact val_bpb`
2. reproducibility
3. byte-budget safety margin
4. faithful compliance with the challenge rules
