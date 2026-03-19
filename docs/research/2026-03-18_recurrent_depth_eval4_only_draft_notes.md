# 2026-03-18: Recurrent Depth Eval-Step-Only Draft Notes

## Status

These notes are provisional. They capture the isolated `RECURRENT_EVAL_STEPS=4` test on the original recurrent-depth configuration.

## Purpose

Isolate the effect of increasing eval recurrence without changing the trained recurrent architecture or stored parameter count.

Reference notes:

- baseline proxy: `docs/research/2026-03-18_4xh100_baseline_proxy.md`
- first recurrent run: `docs/research/2026-03-18_recurrent_depth_draft_notes.md`

## Key Training Logic

The current implementation handles recurrent steps like this:

- training mode uses `recurrent_config.steps`
- eval mode uses `recurrent_config.eval_steps`
- if `RECURRENT_BACKPROP_STEPS < RECURRENT_STEPS`, the prefix unrolls in training run under `torch.no_grad()`

Relevant code:

- `train_gpt.py:936-950`

That means this run keeps training at `RECURRENT_STEPS=3` and changes only validation/final-eval recurrence to `4`.

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- machine family: same Nebraska host family used for prior proxy runs
- image base: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- extra system dependency required: `build-essential`
- compiler env required for `torch.compile`: `CC=gcc CXX=g++`

## Run

Command:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_recurrent_v1_eval4_only \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
RECURRENT_ENABLE=1 \
RECURRENT_PRELUDE_LAYERS=1 \
RECURRENT_CORE_LAYERS=2 \
RECURRENT_STEPS=3 \
RECURRENT_BACKPROP_STEPS=3 \
RECURRENT_CODA_LAYERS=2 \
RECURRENT_EVAL_STEPS=4 \
RECURRENT_STATE_INIT=like_init \
RECURRENT_INPUT_INJECTION=linear_concat \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Observed key lines:

- `model_params:10228776`
- `step:200/20000 val_loss:3.5312 val_bpb:2.0914 train_time:16194ms step_avg:80.97ms`
- `step:400/20000 val_loss:3.9125 val_bpb:2.3172 train_time:32501ms step_avg:81.25ms`
- `step:600/20000 val_loss:4.2156 val_bpb:2.4967 train_time:48856ms step_avg:81.43ms`

Outcome:

- manually terminated early after the validation curve clearly diverged from the earlier recurrent reference

Raw local log:

- `logs/vast_4xh100_recurrent_v1_eval4_only.txt`

## Comparison To Earlier Recurrent Reference

Earlier recurrent reference (`RECURRENT_EVAL_STEPS=3`):

- `model_params:10228776`
- `step:200 val_bpb:1.6657`
- `step:400 val_bpb:1.5416`
- `step_avg:81.54ms`
- `final_int8_zlib_roundtrip_exact val_bpb:1.29895208`

Observed deltas:

- parameter count is unchanged
- throughput is effectively unchanged
- validation quality is dramatically worse:
  - step `200`: `2.0914` vs `1.6657`
  - step `400`: `2.3172` vs `1.5416`

Inference:

- increasing eval recurrence from `3` to `4` is strongly negative in the current recurrent branch, even when training recurrence and stored parameters are held fixed

## Recommendation

Keep `RECURRENT_EVAL_STEPS=3` as the default for this branch until the recurrent training recipe changes materially.

## Submission-Contract Notes

- this was a coherent screening run on the fixed challenge dataset and validation split
- this is a research note only, not a frozen submission artifact
