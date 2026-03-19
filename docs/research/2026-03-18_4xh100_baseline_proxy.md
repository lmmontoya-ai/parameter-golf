# 2026-03-18: `4xH100` Baseline Proxy Reproduction

## Purpose

Check whether a `4xH100` run can serve as a coherent proxy for the `8xH100 / 10 minute` baseline track and verify that the current local stack behaves consistently with the submission rules.

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- image base: `pytorch/pytorch:latest`
- Python env: `uv` virtualenv at `.venv`
- torch inside venv: `2.10.0+cu128`
- CUDA available in venv: `True`
- extra system dependency required: `build-essential`

Important environment finding:

- the first run failed before training because `torch.compile` and Triton could not find a C compiler
- installing `build-essential` and setting `CC=gcc CXX=g++` fixed the issue

## Command

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_proxy_baseline_sp1024_ccfix \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

The raw local log is at:

- `logs/vast_4xh100_proxy_baseline_sp1024_ccfix.txt`

## Key Results

- `train_loader:dataset:fineweb10B_sp1024 train_shards:80`
- `val_loader:shards pattern=/root/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632`
- `attnres:enabled=0 block_layers=3 implementation=block_attnres_v1`
- `model_params:17059912`
- `world_size:4 grad_accum_steps:2`
- `stopping_early: wallclock_cap train_time:1200028ms step:14210/20000`
- `peak memory allocated: 10257 MiB reserved: 10710 MiB`
- `step_avg:84.45ms`
- pre-quant stop metric: `val_loss:2.0568 val_bpb:1.2182`
- `Serialized model int8+zlib: 15823476 bytes`
- `Total submission size int8+zlib: 15873947 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.06967197 val_bpb:1.22577632`
- `final_int8_zlib_roundtrip eval_time:2718ms`

## Comparison To Published `8xH100` Baseline

Published baseline from `records/track_10min_16mb/2026-03-17_NaiveBaseline`:

- `step_avg:43.54ms`
- `stopping_early: ... step:13780/20000`
- `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`
- `Total submission size int8+zlib: 15863489 bytes`

Observed deltas for the `4xH100` proxy run:

- step time is about `1.94x` slower than the `8xH100` run
- stop step is `14210`, close to the expected scaled neighborhood
- final exact roundtrip score is `+0.00141062 val_bpb` worse than the published `8xH100` result
- total artifact size is `+10458` bytes larger than the published baseline run

## Interpretation

- The `4xH100` proxy behaves coherently as a scaled version of the `8xH100` baseline.
- The fixed global batch behavior is consistent with the script design:
  - `8xH100` uses `grad_accum_steps:1`
  - `4xH100` uses `grad_accum_steps:2`
- The competition-relevant metric is confirmed to be the post-quant roundtrip exact score, not raw training loss.
- The script timer is confirmed to cap the measured training loop only:
  - compile warmup happened before timed training
  - final quantized roundtrip evaluation happened after the timed stop
- Evaluation time on this run was tiny relative to the budget, so eval itself is not the limiting factor here.

## Submission-Contract Notes

- This run used the full cached challenge dataset and the fixed validation split, which is required for meaningful comparisons.
- The artifact remained under the `16,000,000`-byte cap after int8+zlib export.
- Current `code bytes` logging is still a development-only approximation because active code now lives outside `train_gpt.py`.
- This is a useful research reproduction, not a frozen submission snapshot.

## Conclusion

The baseline stack is working coherently on rented H100 hardware, and the `4xH100 -> 1200s` proxy is good enough for early experiment screening.

Next recommended use:

- use this run as the baseline reference for `4xH100` A/B experiment work
- still verify promising branches on real `8xH100` hardware before treating them as record-quality evidence
