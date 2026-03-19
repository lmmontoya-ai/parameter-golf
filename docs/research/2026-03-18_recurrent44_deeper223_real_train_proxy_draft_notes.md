# 2026-03-18 Recurrent44 Deeper223 Real-Train Proxy Draft Notes

## Summary

This note captures the first clean real-train run of the strongest recurrent `4/4` variant found so far:

- recurrent depth enabled
- deeper `2/2/4/3` structure
- real `sp1024` train split with `80` train shards
- validated `4xH100 -> 1200s` proxy for the `8xH100 -> 600s` track

The run completed cleanly and materially improved on the earlier recurrent variants, but it still did not catch the dense baseline proxy.

## Why This Run

Earlier findings established:

- `8xH100` inventory was not available when this run was launched on **March 18, 2026**
- the repo's `4xH100 -> 1200s` setup is a coherent proxy for the `8xH100 -> 600s` baseline track
- the `2/2/4/3` recurrent layout was the best `4/4` recurrent variant in the earlier val-only screens

So this run answers the next serious question: how does that architecture behave on the real train split under the actual proxy regime?

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 SXM`
- location: Nebraska, US
- image: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- required system dependency: `build-essential`

Important environment note:

- the pod needed a C compiler for `torch.compile` and Triton
- `build-essential` plus `CC=gcc CXX=g++` remained necessary here, just as in the earlier baseline proxy reproduction

## Data Gate

This was the first real-train run executed after the deterministic downloader changes.

Preparation sequence:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --verify-only
```

Observed result:

- `_download_state.json` reported `ready: true`
- requested set was `84` artifacts total:
  - `80` train shards
  - `1` val shard
  - `2` tokenizer artifacts
  - `1` manifest
- `missing_total: 0`

This is the first run in this project where the rented-pod data path was explicitly gated before training started.

## Command

```bash
CC=gcc \
CXX=g++ \
OMP_NUM_THREADS=1 \
RUN_ID=deeper223_train_4xh100_proxy \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=13 \
RECURRENT_ENABLE=1 \
RECURRENT_PRELUDE_LAYERS=2 \
RECURRENT_CORE_LAYERS=2 \
RECURRENT_STEPS=4 \
RECURRENT_BACKPROP_STEPS=4 \
RECURRENT_CODA_LAYERS=3 \
RECURRENT_EVAL_STEPS=4 \
RECURRENT_STATE_INIT=like_init \
RECURRENT_INPUT_INJECTION=linear_concat \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Model And Runtime

- `recurrent:enabled=1 prelude_layers=2 core_layers=2 steps=4 backprop_steps=4 coda_layers=3 eval_steps=4`
- `model_params: 13900856`
- `world_size:4 grad_accum_steps:2`
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `step_avg: 116.38 ms`
- `stopping_early: wallclock_cap train_time:1200037ms step:10311/20000`
- `peak memory allocated: 14727 MiB reserved: 14978 MiB`

## Key Training Checkpoints

- `step 200 val_bpb: 1.6631`
- `step 1000 val_bpb: 1.4125`
- `step 2000 val_bpb: 1.3557`
- `step 4000 val_bpb: 1.3167`
- `step 6000 val_bpb: 1.3015`
- `step 8000 val_bpb: 1.2890`
- `step 9200 val_bpb: 1.2824`
- `step 9400 val_bpb: 1.2762`
- `step 9600 val_bpb: 1.2698`
- `step 9800 val_bpb: 1.2643`
- `step 10000 val_bpb: 1.2581`
- `step 10200 val_bpb: 1.2525`
- `step 10311 val_bpb: 1.2503`

The curve improved steadily throughout the full training window. There was no sign of the late-run degradation seen in some earlier recurrent sweeps.

## Final Export Metrics

- `Serialized model: 54580348 bytes`
- `Code size: 60293 bytes`
- `Total submission size: 54640641 bytes`
- `Serialized model int8+zlib: 12850812 bytes`
- `Total submission size int8+zlib: 12911105 bytes`
- `final_int8_zlib_roundtrip_exact val_loss: 2.13637296`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.26528040`
- `final_int8_zlib_roundtrip eval_time: 3904ms`

Development caveat:

- the logged `Code size` still undercounts true submission bytes because active code now lives outside `train_gpt.py`
- treat the byte numbers here as research-comparison numbers, not final submission accounting

## Comparison

### Versus The Validated Dense `4xH100` Proxy Baseline

Dense baseline reference from `2026-03-18_4xh100_baseline_proxy.md`:

- `model_params: 17059912`
- `step_avg: 84.45 ms`
- `Total submission size int8+zlib: 15873947 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.22577632`

Observed deltas for `recurrent44 deeper223`:

- parameter count: `-3169056` params, about `-18.6%`
- step time: `+31.93 ms`, about `+37.8%`
- compressed artifact size: `-2962842` bytes, about `-18.7%`
- final exact quality: `+0.03950408 val_bpb` worse

### Versus The Earlier `recurrent 4/4 deeper223` Val-Only Screen

Earlier val-only screen from `2026-03-18_val_only_and_recurrent44_draft_notes.md`:

- `final_int8_zlib_roundtrip_exact val_bpb: 1.30143635`
- `Total submission size int8+zlib: 10134653 bytes`

Observed difference here:

- real-train quality is much better than the val-only screen
- the real-train run reached `1.26528040`, improving by about `0.0362 val_bpb`

Inference:

- this architecture is real, not a val-only artifact
- the recurrent branch can improve substantially under the actual train split

## Interpretation

- This is the strongest recurrent result so far on a principled mainline run.
- The architecture is stable, reproducible, and compatible with the new deterministic data-prep gate.
- The quality curve is good enough that recurrent depth remains worth pursuing.
- The current `2/2/4/3` configuration still misses the dense baseline by about `0.0395 val_bpb`, so it is not yet submission-ready as a standalone replacement.
- The byte savings are real but smaller than the original recurrent `1/2/3/2` branch because this deeper variant stores more unique layers.
- The late-run improvement suggests there is still room to tune this family rather than abandon it.

Best current reading:

- recurrent depth is now a credible mainline research branch
- `2/2/4/3` is the best recurrent candidate we have tested
- next steps should focus on improving its quality without losing too much of the byte advantage

## Raw Artifacts

- local pipeline log: `logs/vast_4xh100_recurrent44_deeper223_real_train_pipeline.txt`
- local train log: `logs/vast_4xh100_recurrent44_deeper223_real_train.txt`
- local runner log copy: `logs/deeper223_train_4xh100_proxy.txt`
- local download gate state: `logs/deeper223_train_4xh100_proxy_download_state.json`
