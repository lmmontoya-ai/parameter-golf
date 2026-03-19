# 2026-03-18: Recurrent Depth Param And Eval-Step Sweep Draft Notes

## Status

These notes are provisional. They capture a follow-up sweep on the recurrent-depth branch after the first `1/2/3/2` run and should be treated as draft research notes, not final conclusions.

## Purpose

Test whether the recurrent branch improves if we do both of the following at once:

- increase stored parameter count
- increase `RECURRENT_EVAL_STEPS` from `3` to `4`

Reference notes:

- baseline proxy: `docs/research/2026-03-18_4xh100_baseline_proxy.md`
- first recurrent run: `docs/research/2026-03-18_recurrent_depth_draft_notes.md`

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- machine family: same Nebraska host used for the baseline proxy and first recurrent run
- image base: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- extra system dependency required: `build-essential`
- compiler env required for `torch.compile`: `CC=gcc CXX=g++`

Common setup:

- dataset cache populated with `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
- tokenizer artifacts fetched with `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0`
- remote tests passed before the sweep:
  - `python -m unittest tests.test_attention_residuals tests.test_recurrent_depth`

## Sweep

All runs used:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RECURRENT_ENABLE=1 \
RECURRENT_BACKPROP_STEPS=3 \
RECURRENT_STATE_INIT=like_init \
RECURRENT_INPUT_INJECTION=linear_concat \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

### Run A: Wider `640d`, same `1/2/3/2`, `eval_steps=4`

Run-specific env:

```bash
RUN_ID=vast_4xh100_recurrent_v1_w640_eval4 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
RECURRENT_PRELUDE_LAYERS=1 \
RECURRENT_CORE_LAYERS=2 \
RECURRENT_STEPS=3 \
RECURRENT_CODA_LAYERS=2 \
RECURRENT_EVAL_STEPS=4
```

Observed key lines:

- `model_params:15817010`
- `step:200/20000 val_loss:3.2393 val_bpb:1.9185 train_time:22068ms step_avg:110.34ms`
- `step:400/20000 val_loss:3.5109 val_bpb:2.0793 train_time:44321ms step_avg:110.80ms`
- `step:600/20000 val_loss:3.7169 val_bpb:2.2014 train_time:66547ms step_avg:110.91ms`

Outcome:

- manually terminated early after it was clearly outside the viable range on both quality and throughput

Raw local log:

- `logs/vast_4xh100_recurrent_v1_w640_eval4.txt`

### Run B: Wider `576d`, same `1/2/3/2`, `eval_steps=4`

Run-specific env:

```bash
RUN_ID=vast_4xh100_recurrent_v1_w576_eval4 \
MODEL_DIM=576 \
NUM_HEADS=9 \
NUM_KV_HEADS=3 \
RECURRENT_PRELUDE_LAYERS=1 \
RECURRENT_CORE_LAYERS=2 \
RECURRENT_STEPS=3 \
RECURRENT_CODA_LAYERS=2 \
RECURRENT_EVAL_STEPS=4
```

Observed key lines:

- `model_params:12318381`
- `step:200/20000 val_loss:3.4198 val_bpb:2.0254 train_time:18923ms step_avg:94.61ms`
- `step:400/20000 val_loss:3.7244 val_bpb:2.2058 train_time:37956ms step_avg:94.89ms`

Outcome:

- manually terminated early after it showed the same failure pattern as the `640d` run

Raw local log:

- `logs/vast_4xh100_recurrent_v1_w576_eval4.txt`

### Run C: More unique blocks, same `512d`, `2/1/3/4`, `eval_steps=4`

Run-specific env:

```bash
RUN_ID=vast_4xh100_recurrent_v1_structural_eval4 \
RECURRENT_PRELUDE_LAYERS=2 \
RECURRENT_CORE_LAYERS=1 \
RECURRENT_STEPS=3 \
RECURRENT_CODA_LAYERS=4 \
RECURRENT_EVAL_STEPS=4
```

Observed key lines:

- `model_params:13900856`
- `step:200/20000 val_loss:3.0415 val_bpb:1.8014 train_time:16329ms step_avg:81.65ms`
- `step:400/20000 val_loss:3.2513 val_bpb:1.9256 train_time:32761ms step_avg:81.90ms`
- `step:600/20000 val_loss:3.5287 val_bpb:2.0899 train_time:49204ms step_avg:82.01ms`
- `step:800/20000 val_loss:3.7930 val_bpb:2.2464 train_time:65602ms step_avg:82.00ms`

Outcome:

- manually terminated early after it confirmed the same negative pattern even without changing width or head geometry

Raw local log:

- `logs/vast_4xh100_recurrent_v1_structural_eval4.txt`

## Comparison To Prior Baselines

Reference points:

- baseline proxy:
  - `step:200 val_bpb:1.6776`
  - `step:400 val_bpb:1.5253`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.22577632`
- first recurrent run (`1/2/3/2`, `eval_steps=3`):
  - `model_params:10228776`
  - `step:200 val_bpb:1.6657`
  - `step:400 val_bpb:1.5416`
  - `step_avg:81.54ms`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.29895208`

Observed sweep deltas:

- every `eval_steps=4` screening run was much worse than the earlier recurrent `eval_steps=3` run by step `200`
- the structural `13.9M` run was the least bad of the three, but it was still worse than the earlier recurrent run by about `+0.1357 val_bpb` at step `200`
- widening the model made throughput slower without helping quality:
  - `576d`: about `94.6ms`
  - `640d`: about `110.3ms`
  - earlier recurrent reference: about `81.5ms`
- increasing stored parameters by adding unique blocks preserved throughput, but quality still deteriorated badly once `eval_steps=4` was used

## Interpretation

- The common failure mode appears to be the move from `RECURRENT_EVAL_STEPS=3` to `4`, not just the added parameters.
- The `640d` and `576d` runs show that naive width increases are not stable under the current optimizer schedule.
- The structural `2/1/3/4` run is the most informative result:
  - parameters increased from `10.23M` to `13.90M`
  - runtime stayed near the earlier recurrent run
  - quality still collapsed relative to the earlier recurrent reference

Inference from those results:

- testing extra eval recurrence before the recurrent family is otherwise stabilized is premature
- the recurrent branch should keep `RECURRENT_EVAL_STEPS=3` for now
- if we want more parameters, the next test should likely hold eval recurrence fixed and study parameter increases separately

## Recommendation

Do not promote any of these `eval_steps=4` variants.

Recommended follow-up order:

- keep `RECURRENT_EVAL_STEPS=3` as the default recurrent setting for now
- if we still want a larger recurrent model, test that axis separately from eval-step scaling
- only revisit `RECURRENT_EVAL_STEPS=4` after a stronger recurrent training recipe exists

## Submission-Contract Notes

- these were coherent research runs on the fixed challenge dataset and validation split
- they were screening runs, not full frozen artifacts
- current `code bytes` accounting still undercounts modular development code under `research/`
