# 2026-03-18: Recurrent Depth Draft Notes

## Status

These notes are provisional. They capture the first rented-GPU recurrent-depth screening run and should be treated as draft research notes, not final conclusions.

## Purpose

Check whether the first recurrent-depth implementation is viable on the validated `4xH100 -> 1200s` proxy setup, using the reproduced baseline proxy run as the control.

Baseline control reference:

- `docs/research/2026-03-18_4xh100_baseline_proxy.md`

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- image base: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- extra system dependency required: `build-essential`
- compiler env required for `torch.compile`: `CC=gcc CXX=g++`

Dataset and tokenizer setup:

- dataset cache populated with `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
- tokenizer artifacts fetched with `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0`

## Run

Command:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_recurrent_v1_p1_c2_s3_b3_c2 \
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
RECURRENT_EVAL_STEPS=3 \
RECURRENT_STATE_INIT=like_init \
RECURRENT_INPUT_INJECTION=linear_concat \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Observed key lines:

- `recurrent:enabled=1 prelude_layers=1 core_layers=2 steps=3 backprop_steps=3 coda_layers=2 eval_steps=3 state_init=like_init input_injection=linear_concat implementation=recurrent_depth_v1`
- `model_params:10228776`
- `world_size:4 grad_accum_steps:2`
- `step:200/20000 val_loss:2.8125 val_bpb:1.6657 train_time:16156ms step_avg:80.78ms`
- `step:1000/20000 val_loss:2.4203 val_bpb:1.4334 train_time:81528ms step_avg:81.45ms`
- `step:5000/20000 val_loss:2.2573 val_bpb:1.3369 train_time:407650ms step_avg:81.53ms`
- `step:10000/20000 val_loss:2.2204 val_bpb:1.3151 train_time:815339ms step_avg:81.53ms`
- `step:14717/20000 val_loss:2.1577 val_bpb:1.2779 train_time:1200027ms step_avg:81.54ms`
- `stopping_early: wallclock_cap train_time:1200027ms step:14717/20000`
- `peak memory allocated: 10425 MiB reserved: 10572 MiB`
- `Serialized model int8+zlib: 9453994 bytes`
- `Total submission size int8+zlib: 9514287 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.19322617 val_bpb:1.29895208`
- `final_int8_zlib_roundtrip eval_time:2746ms`

Raw local log:

- `logs/vast_4xh100_recurrent_v1_p1_c2_s3_b3_c2.txt`

## Comparison To Baseline Proxy

Baseline proxy reference points from `docs/research/2026-03-18_4xh100_baseline_proxy.md`:

- `model_params:17059912`
- `step_avg:84.45ms`
- `step:14210/20000 val_loss:2.0568 val_bpb:1.2182`
- `Total submission size int8+zlib: 15873947 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.06967197 val_bpb:1.22577632`

Observed deltas:

- parameters dropped by `6,831,136` or about `40.04%`
- total int8+zlib artifact size dropped by `6,359,660` bytes or about `40.06%`
- training `step_avg` improved by about `3.45%`:
  - `81.54ms` vs `84.45ms`
- stop-step pre-quant `val_bpb` was worse than baseline by about `+0.0597`:
  - `1.2779` vs `1.2182`
- final exact roundtrip `val_bpb` was worse than baseline by about `+0.0732`:
  - `1.29895208` vs `1.22577632`

Memory impact:

- baseline proxy peak allocated: `10257 MiB`
- recurrent run peak allocated: `10425 MiB`
- observed increase: `+168 MiB`

## Interpretation

- This is the first architecture experiment that clearly preserves baseline-class throughput while delivering a large byte win.
- The macro-architecture behaves operationally as intended:
  - compile stayed stable
  - runtime stayed within the baseline envelope
  - memory stayed close to baseline
  - parameter and compressed-byte reductions were both about `40%`
- The quality hit is still too large for promotion under the current pass gate:
  - the experiment target was no worse than `+0.010 val_bpb`
  - the observed final gap was about `+0.0732`

What looks encouraging:

- the early curve was competitive:
  - at step `200`, recurrent depth was slightly better than the baseline proxy
  - throughput never regressed relative to baseline
- the branch appears structurally healthy, unlike the failed Attention Residuals v1 run

Likely takeaways:

- the recurrent retrofit is a strong **bytes-first** mechanism
- this v1 shape probably needs more help on optimization or effective capacity before it becomes leaderboard-competitive
- likely next levers are:
  - eval-only extra recurrence after the fixed-step model is stable
  - data-quality improvements
  - distillation or weight-sharing combinations
  - later low-bit / BitNet-style export experiments

## Recommendation

Do not promote this exact config as a standalone winner, but do keep the recurrent branch active.

Recommended next steps:

- keep this run as the baseline research note for recurrent depth v1
- treat recurrent depth as a serious combination candidate because the byte and throughput results are strong
- before merging ideas, run at least one focused follow-up on the recurrent family itself:
  - `RECURRENT_EVAL_STEPS=4` or `5` as an inference-only scaling check
  - or a conservative optimization/data improvement on the same architecture

## Submission-Contract Notes

- this run was coherent with the current training/eval contract:
  - fixed challenge dataset
  - fixed validation split
  - post-quant roundtrip metric recorded
- the artifact remained well under the `16,000,000`-byte cap after int8+zlib export
- current `code bytes` accounting still undercounts modular development code under `research/`
- this is a research note only, not a frozen submission artifact
