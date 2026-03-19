# 2026-03-18 Recurrent44 Deeper223 W576 Real-Train Proxy Draft Notes

## Summary

This note captures the next real-train recurrent-depth proxy run after the baseline `deeper223` result:

- same recurrent `2/2/4/3` structure
- same real `sp1024` train split with `80` train shards
- same validated `4xH100 -> 1200s` proxy
- widened model:
  - `MODEL_DIM=576`
  - `NUM_HEADS=9`
  - `NUM_KV_HEADS=3`

This is the strongest recurrent result so far.

## Why This Run

The previous real-train `deeper223` run established that recurrent depth was viable, but it still finished meaningfully behind the dense baseline while leaving substantial size headroom.

That result suggested one high-value next move:

- keep the recurrent structure fixed
- spend more of the artifact budget on width
- test whether quality improves enough to justify moving closer to the cap

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 SXM`
- location: Nebraska, US
- image: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- required system dependency: `build-essential`

As with the earlier runs, `build-essential` plus `CC=gcc CXX=g++` remained necessary for the compiled path.

## Data Gate

Preparation again used the deterministic downloader gate:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --verify-only
```

Observed result:

- `_download_state.json` reported `ready: true`
- requested set was `84` artifacts total
- `missing_total: 0`

So this run is directly comparable to the earlier real-train recurrent note without caveats about cache reliability.

## Command

```bash
CC=gcc \
CXX=g++ \
OMP_NUM_THREADS=1 \
RUN_ID=recurrent44_deeper223_w576_train_4xh100_proxy \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=576 \
NUM_HEADS=9 \
NUM_KV_HEADS=3 \
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

- `model_params: 16744383`
- `attention_mode:gqa num_heads:9 num_kv_heads:3`
- `step_avg: 134.33 ms`
- `stopping_early: wallclock_cap train_time:1200081ms step:8934/20000`
- `peak memory allocated: 16098 MiB reserved: 16184 MiB`

## Key Training Checkpoints

- `step 200 val_bpb: 1.6777`
- `step 1000 val_bpb: 1.4046`
- `step 2000 val_bpb: 1.3437`
- `step 4000 val_bpb: 1.3043`
- `step 6000 val_bpb: 1.2876`
- `step 7000 val_bpb: 1.2801`
- `step 8000 val_bpb: 1.2666`
- `step 8200 val_bpb: 1.2605`
- `step 8400 val_bpb: 1.2542`
- `step 8600 val_bpb: 1.2472`
- `step 8800 val_bpb: 1.2417`
- `step 8934 val_bpb: 1.2388`

The important curve shape:

- early checkpoints were only slightly better or mixed relative to the narrower run
- later checkpoints pulled away clearly
- the width increase paid back in the second half of training

## Final Export Metrics

- `Serialized model: 65823356 bytes`
- `Code size: 60293 bytes`
- `Total submission size: 65883649 bytes`
- `Serialized model int8+zlib: 15432272 bytes`
- `Total submission size int8+zlib: 15492565 bytes`
- `final_int8_zlib_roundtrip_exact val_loss: 2.11218404`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.25095436`
- `final_int8_zlib_roundtrip eval_time: 4433ms`

Development caveat:

- code-size logging still undercounts the true submission code footprint while active code lives outside `train_gpt.py`
- treat the byte numbers here as research-comparison numbers, not frozen submission accounting

## Comparison

### Versus The Narrower Real-Train `deeper223` Recurrent Run

Reference from `2026-03-18_recurrent44_deeper223_real_train_proxy_draft_notes.md`:

- `model_params: 13900856`
- `step_avg: 116.38 ms`
- `Total submission size int8+zlib: 12911105 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.26528040`

Observed deltas for `w576`:

- parameter count: `+2843527`, about `+20.5%`
- step time: `+17.95 ms`, about `+15.4%`
- compressed artifact size: `+2581460` bytes, about `+20.0%`
- final exact quality: `-0.01432604 val_bpb` better

Interpretation:

- this width increase was absolutely worth it
- the gain in quality is large enough to justify the slower throughput
- the artifact is now much closer to the cap

### Versus The Dense `4xH100` Proxy Baseline

Dense baseline reference from `2026-03-18_4xh100_baseline_proxy.md`:

- `model_params: 17059912`
- `step_avg: 84.45 ms`
- `Total submission size int8+zlib: 15873947 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.22577632`

Observed deltas for `w576`:

- parameter count: `-315529`, about `-1.8%`
- step time: `+49.88 ms`, about `+59.1%`
- compressed artifact size: `-381382` bytes
- final exact quality: `+0.02517804 val_bpb` worse

Interpretation:

- this recurrent model is now much closer to the dense baseline than the narrower recurrent branch was
- the remaining quality gap to the dense proxy is only about `0.0252 val_bpb`
- it stays safely under the `16,000,000`-byte cap

Additional size note:

- headroom to the hard cap is `507435` bytes

## Interpretation

- This is the best recurrent result in the project so far.
- The recurrent branch is no longer just “interesting because it is smaller.”
- It now competes on quality while staying under the cap.
- The main tradeoff is clear:
  - better quality
  - less byte headroom
  - slower training throughput
- Even after the slowdown, the final exact score improved enough that this run is strictly more important than the narrower recurrent variant.

Current read:

- width was the right next lever
- the recurrent branch should remain a mainline contender
- further recurrent scaling must now be very disciplined because the byte headroom is small

## Raw Artifacts

- local pipeline log: `logs/vast_4xh100_recurrent44_deeper223_w576_real_train_pipeline.txt`
- local train log: `logs/vast_4xh100_recurrent44_deeper223_w576_real_train.txt`
- local runner log copy: `logs/recurrent44_deeper223_w576_train_4xh100_proxy.txt`
- local download gate state: `logs/recurrent44_deeper223_w576_train_4xh100_proxy_download_state.json`
