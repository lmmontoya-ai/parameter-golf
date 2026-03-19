# 2026-03-18: Recurrent Depth Train4 Eval4 Draft Notes

## Status

These notes are provisional. They capture the first recurrent-depth run that was trained and evaluated coherently with `4` recurrent steps.

## Purpose

Test whether the earlier negative `RECURRENT_EVAL_STEPS=4` result was caused by eval-only extrapolation rather than by four-step recurrence itself.

Reference notes:

- baseline proxy: `docs/research/2026-03-18_4xh100_baseline_proxy.md`
- first recurrent run: `docs/research/2026-03-18_recurrent_depth_draft_notes.md`
- eval-only negative control: `docs/research/2026-03-18_recurrent_depth_eval4_only_draft_notes.md`

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
RUN_ID=vast_4xh100_recurrent_v1_train4_eval4 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
NUM_LAYERS=11 \
RECURRENT_ENABLE=1 \
RECURRENT_PRELUDE_LAYERS=1 \
RECURRENT_CORE_LAYERS=2 \
RECURRENT_STEPS=4 \
RECURRENT_BACKPROP_STEPS=4 \
RECURRENT_CODA_LAYERS=2 \
RECURRENT_EVAL_STEPS=4 \
RECURRENT_STATE_INIT=like_init \
RECURRENT_INPUT_INJECTION=linear_concat \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Why `NUM_LAYERS=11`:

- the recurrent implementation validates `RECURRENT_PRELUDE_LAYERS + RECURRENT_CORE_LAYERS * RECURRENT_STEPS + RECURRENT_CODA_LAYERS == NUM_LAYERS`
- this lets us keep the same stored block layout as the original recurrent run while increasing effective depth from `9` to `11`

Observed key lines:

- `recurrent:enabled=1 prelude_layers=1 core_layers=2 steps=4 backprop_steps=4 coda_layers=2 eval_steps=4 state_init=like_init input_injection=linear_concat implementation=recurrent_depth_v1`
- `model_params:10228776`
- `step:200/20000 val_loss:2.8195 val_bpb:1.6699 train_time:19671ms step_avg:98.36ms`
- `step:400/20000 val_loss:2.6144 val_bpb:1.5484 train_time:39524ms step_avg:98.81ms`
- `step:1000/20000 val_loss:2.4252 val_bpb:1.4363 train_time:99099ms step_avg:99.10ms`
- `step:12091/20000 val_loss:2.1642 val_bpb:1.2817 train_time:1200070ms step_avg:99.25ms`
- `stopping_early: wallclock_cap train_time:1200070ms step:12091/20000`
- `peak memory allocated: 12615 MiB reserved: 12874 MiB`
- `Serialized model int8+zlib: 9449592 bytes`
- `Total submission size int8+zlib: 9509885 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.19048655 val_bpb:1.29732952`
- `final_int8_zlib_roundtrip eval_time:3358ms`

Raw local log:

- `logs/vast_4xh100_recurrent_v1_train4_eval4.txt`

## Comparison To Earlier Recurrent Run

Earlier recurrent reference (`steps=3`, `eval_steps=3`):

- `model_params:10228776`
- `step_avg:81.54ms`
- `step:14717/20000 val_bpb:1.2779`
- `Total submission size int8+zlib: 9514287 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb:1.29895208`

Observed deltas:

- stored parameter count is unchanged
- effective training/eval depth is higher
- training step time is slower by about `21.7%`:
  - `99.25ms` vs `81.54ms`
- fewer optimizer steps fit into the same `1200s` proxy window:
  - `12091` vs `14717`
- pre-quant stop metric is slightly worse:
  - `1.2817` vs `1.2779`
- final exact roundtrip metric is slightly better:
  - `1.29732952` vs `1.29895208`
  - improvement: about `-0.00162 val_bpb`
- artifact size is slightly smaller:
  - `9509885` vs `9514287`
  - delta: `-4402` bytes

## Comparison To Eval-Only Negative Control

Eval-only `RECURRENT_EVAL_STEPS=4` negative control:

- same stored params: `10228776`
- training still used `RECURRENT_STEPS=3`
- step `200 val_bpb:1.6699` in this run vs `2.0914` in the eval-only run
- step `400 val_bpb:1.5484` in this run vs `2.3172` in the eval-only run

Interpretation:

- the earlier failure was not “4 recurrent steps are impossible”
- the earlier failure was specifically “forcing a fourth eval step on a model trained for three steps is harmful”

## Interpretation

- This is the first coherent four-step recurrent result.
- Training the model for four recurrent updates restores stability and gives a small improvement over the earlier three-step recurrent run on the final roundtrip metric.
- The gain is real but modest, and it comes with a clear compute tradeoff:
  - slower steps
  - fewer optimizer updates in the same wallclock budget

## Recommendation

Keep this as a valid recurrent variant, but do not treat it as a breakout win.

Practical takeaway:

- `eval_steps > train_steps` is a bad idea for the current v1 recurrent setup
- `train_steps = eval_steps = 4` is coherent and slightly better than `3/3`
- future recurrent work should compare `3/3` and `4/4` as legitimate alternatives, not mix them

## Submission-Contract Notes

- this run used the fixed challenge dataset and validation split
- the artifact remained well under the `16,000,000`-byte cap after int8+zlib export
- current `code bytes` accounting still undercounts modular development code under `research/`
- this is a research note only, not a frozen submission artifact
