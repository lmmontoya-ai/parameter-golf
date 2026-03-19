# 2026-03-18 Val-Only And Recurrent44 Draft Notes

## Summary

This note captures a mixed experiment block on Vast `4xH100` infrastructure:

- `baseline` trained on the fixed validation split only
- `recurrent 3/3` trained on the fixed validation split only
- `recurrent 4/4` val-only scale-up screens on the same pod after the second pod failed to come up reliably

This is a provisional note. The `4/4` sweep was still in progress while this draft was written.

## Pod Availability

Intended plan:

- pod A: `4xH100` baseline + recurrent `3/3` on val only
- pod B: separate `4xH100` recurrent `4/4` scale-up sweep

What happened:

- pod A succeeded on Vast offer `32086065` in Nebraska
- a Thailand `4xH100` offer repeatedly stayed in `loading` without producing a usable container
- a France `4xH100` offer failed with `docker_build() error writing dockerfile`
- Prime had no on-demand `4xH100` inventory at the time of the run

Result:

- the val-only baseline and recurrent runs were executed on pod A
- the `4/4` sweep was salvaged serially on pod A instead of running concurrently on a second pod

## Pod A Environment

- provider: Vast.ai
- GPUs: `4xH100 SXM`
- image: `pytorch/pytorch:latest`
- required system fix: `build-essential`
- Python env: `uv` + `.venv`
- distributed launch: `torchrun --standalone --nproc_per_node=4`
- tokenizer/data variant: `sp1024`

Important operational note:

- launching with plain `python train_gpt.py` only used one GPU
- the correct `4xH100` path for this repo is `torchrun --standalone --nproc_per_node=4 train_gpt.py`

## Dataset Setup

Val-only runs used a synthetic dataset view:

- source: `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- val-only train view: `data/datasets/fineweb10B_sp1024_valonly/`
- each `fineweb_val_*.bin` shard was symlinked twice:
  - once as `fineweb_val_*.bin`
  - once as `fineweb_train_*.bin`

This preserves the repo’s normal train/val loader expectations while forcing training onto the validation split.

## Baseline Val-Only Run

Command shape:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024_valonly \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Observed checkpoints:

- `step 200 val_bpb: 1.6590`
- `step 400 val_bpb: 1.4799`
- `step 600 val_bpb: 1.4081`
- `step 800 val_bpb: 1.3597`
- `step 1000 val_bpb: 1.3265`
- `step 1200 val_bpb: 1.3069`
- `step 1400 val_bpb: 1.2928`
- `step 1600 val_bpb: 1.2819`
- `step 1800 val_bpb: 1.2730`
- `step 2000 val_bpb: 1.2657`
- `step 2200 val_bpb: 1.2591`

Runtime:

- `world_size:4 grad_accum_steps:2`
- `step_avg` stabilized around `84.3 ms`

Interpretation:

- training on the fixed validation split is absolutely a real lever
- by `2200` steps it had not yet beaten the public `1.2244` baseline score, but it was close enough that the direction is clearly dangerous
- inference: with enough wallclock, the dense baseline may well overfit the validation split further

## Recurrent 3/3 Val-Only Run

Command shape:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024_valonly \
MAX_WALLCLOCK_SECONDS=300 \
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

Observed checkpoints:

- `model_params: 10228776`
- `step 200 val_bpb: 1.6293`
- `step 400 val_bpb: 1.5125`
- `step 600 val_bpb: 1.4543`
- `step 800 val_bpb: 1.4191`
- `step 1000 val_bpb: 1.3934`
- `step 1200 val_bpb: 1.3779`
- `step 1400 val_bpb: 1.3678`
- `step 1600 val_bpb: 1.3564`

Runtime:

- `step_avg` stabilized around `81.6 ms`

Interpretation:

- the smaller recurrent model overfits validation more slowly than the dense baseline
- it is slightly better than dense at the very first checkpoint (`1.6293` vs `1.6590` at step `200`), but the dense model wins clearly after that
- inference: the dense baseline benefits more from raw parameter count when directly memorizing the validation set

## Recurrent 4/4 Base Val-Only Screen

Command shape:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024_valonly \
MAX_WALLCLOCK_SECONDS=120 \
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

Observed result:

- `model_params: 10228776`
- `step_avg: 99.11 ms`
- `step 200 val_bpb: 1.6348`
- `step 400 val_bpb: 1.5004`
- `step 600 val_bpb: 1.4333`
- `step 800 val_bpb: 1.3873`
- `step 1000 val_bpb: 1.3509`
- `step 1200 val_bpb: 1.3230`
- `step 1212 val_bpb: 1.3226`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.32676169`
- `Total submission size int8+zlib: 7872137 bytes`

Interpretation:

- on val-only overfitting, `4/4` is slower and slightly worse than `3/3`
- it is also well behind the dense val-only baseline
- the byte story remains attractive, but this does not look like a better memorization regime

## Recurrent 4/4 Heads-Only Val-Only Screen

Variant:

- same as `4/4` base, but `NUM_HEADS=16`, `NUM_KV_HEADS=8`

Observed interim checkpoints:

- `model_params: 10228816`
- `step_avg` around `109.6 ms`
- `step 200 val_bpb: 1.6480`
- `step 400 val_bpb: 1.5010`
- `step 600 val_bpb: 1.4324`
- `step 800 val_bpb: 1.3850`
- `step 1000 val_bpb: 1.3465`
- `step 1095 val_bpb: 1.3347`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.33987799`

Interpretation:

- more heads made the run slower
- early quality did not improve versus base `4/4`
- the final exact score also finished worse than base `4/4`

## Recurrent 4/4 Deeper Val-Only Screen

Variant:

- `NUM_LAYERS=13`
- `RECURRENT_PRELUDE_LAYERS=2`
- `RECURRENT_CORE_LAYERS=2`
- `RECURRENT_STEPS=4`
- `RECURRENT_CODA_LAYERS=3`

Observed checkpoints:

- `model_params: 13900856`
- `step_avg` around `116.3 ms`
- `step 200 val_bpb: 1.6058`
- `step 800 val_bpb: 1.3411`
- `step 1000 val_bpb: 1.2993`
- `step 1033 val_bpb: 1.2964`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.30143635`
- `Total submission size int8+zlib: 10134653 bytes`

Interpretation:

- this was much slower and larger than base `4/4`
- it improved quality relative to base `4/4`, but not enough to catch the dense val-only baseline
- on this val-only screen, extra stored depth helps more than extra heads

## Recurrent 4/4 Wider Val-Only Screen

Variant:

- `MODEL_DIM=576`
- `NUM_HEADS=9`
- `NUM_KV_HEADS=3`

Observed early checkpoints:

- `model_params: 12318381`
- `step_avg` around `115.4 ms`
- `step 200 val_bpb: 1.6144`

Interpretation:

- wider is also slower than base `4/4`
- its first validation checkpoint is worse than the deeper variant and only slightly better than heads-only
- this branch needs more runtime for a fairer read, but the early signal is not obviously strong

## Normal-Train Cache Attempt

I attempted to build the normal `sp1024` train cache in the background with:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

Observed result during the experiment window:

- log only showed the Hugging Face unauthenticated warning
- no `fineweb_train_*.bin` shards were materialized before the architecture sweep started

Interpretation:

- the follow-up `4/4` screens stayed on the val-only dataset because the normal-train cache was not ready in time

## Remaining 4/4 Screens

- no additional variants were run after `w576` in this session

## Interim Takeaways

- Dense val-only overfits the fixed validation split much better than recurrent `3/3` or `4/4`.
- Recurrent `4/4` remains slower than `3/3` without showing an overfitting advantage in this val-only regime.
- A heads-only increase does not look promising.
- A deeper stored-parameter variant is the strongest `4/4` branch from this session, but it is still behind the dense val-only baseline.
- The wider branch does not look obviously better at the first checkpoint.
- The second-pod plan failed because the additional `4xH100` inventory was unreliable, not because of a code issue in the repo.

## Raw Logs

Local copies:

- `logs/vast_4xh100_baseline_valonly.txt`
- `logs/vast_4xh100_recurrent33_valonly.txt`
- `logs/vast_4xh100_recurrent44_base_valonly.txt`
- `logs/vast_4xh100_recurrent44_heads16_valonly.txt`
- `logs/vast_4xh100_recurrent44_deeper223_valonly.txt`
- `logs/vast_4xh100_recurrent44_w576_valonly.txt`
- `logs/vast_4xh100_cache_train80_attempt.txt`
