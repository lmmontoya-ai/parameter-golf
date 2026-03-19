# 2026-03-18: Attention Residuals Draft Notes

## Status

These notes are provisional. They capture the first rented-GPU AttnRes screening runs and should be treated as draft research notes, not final conclusions.

## Purpose

Check whether the first Attention Residuals implementation is viable on the validated `4xH100 -> 1200s` proxy setup, using the reproduced baseline proxy run as the control.

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

- dataset cache populated with `python data/cached_challenge_fineweb.py --variant sp1024`
- tokenizer artifacts had to be fetched explicitly with:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0
```

Important environment finding:

- the first broad cache job materialized the dataset shards but did not finish the tokenizer step promptly
- rerunning the cache script with `--train-shards 0` fetched the tokenizer artifacts cleanly

## Runs

### Run A: `ATTNRES_BLOCK_LAYERS=3`

Command:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_attnres_block3 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
ATTNRES_ENABLE=1 \
ATTNRES_BLOCK_LAYERS=3 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Observed key lines:

- `attnres:enabled=1 block_layers=3 implementation=block_attnres_v1`
- `model_params:17069128`
- `world_size:4 grad_accum_steps:2`
- `step:200/20000 val_loss:5.1434 val_bpb:3.0462 train_time:35637ms step_avg:178.18ms`
- `step:400/20000 val_loss:5.4580 val_bpb:3.2326 train_time:71324ms step_avg:178.31ms`
- `step:600/20000 val_loss:5.5219 val_bpb:3.2704 train_time:106982ms step_avg:178.30ms`

Outcome:

- this run was manually terminated after `step 650`
- it was already clearly outside the experiment pass gate on both throughput and validation quality
- no final export/eval metric was collected because the run was intentionally stopped early

Raw local log:

- `logs/vast_4xh100_attnres_block3.txt`

### Run B: `ATTNRES_BLOCK_LAYERS=9` short screen

This was an out-of-plan runtime/quality triage variant, not one of the main planned variants in `experiments/attention_residuals.md`.

Command:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_attnres_block9_short \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=75 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
ATTNRES_ENABLE=1 \
ATTNRES_BLOCK_LAYERS=9 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Observed key lines:

- `attnres:enabled=1 block_layers=9 implementation=block_attnres_v1`
- `model_params:17069128`
- `world_size:4 grad_accum_steps:2`
- `step:200/20000 val_loss:3.5071 val_bpb:2.0771 train_time:33303ms step_avg:166.51ms`
- `step:400/20000 val_loss:3.6149 val_bpb:2.1409 train_time:66668ms step_avg:166.67ms`
- `step:451/20000 val_loss:3.5906 val_bpb:2.1266 train_time:75157ms step_avg:166.65ms`
- `stopping_early: wallclock_cap train_time:75157ms step:451/20000`
- `peak memory allocated: 15392 MiB reserved: 15788 MiB`
- `Serialized model int8+zlib: 9566188 bytes`
- `Total submission size int8+zlib: 9616573 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:3.63334381 val_bpb:2.15187086`

Raw local log:

- `logs/vast_4xh100_attnres_block9_short.txt`

## Comparison To Baseline Proxy

Baseline proxy reference points from `docs/research/2026-03-18_4xh100_baseline_proxy.md`:

- `step:200/20000 val_loss:2.8325 val_bpb:1.6776 train_time:16753ms step_avg:83.77ms`
- `step:400/20000 val_loss:2.5754 val_bpb:1.5253 train_time:33626ms step_avg:84.07ms`
- `step:600/20000 val_loss:2.4523 val_bpb:1.4524 train_time:50545ms step_avg:84.24ms`
- `final_int8_zlib_roundtrip_exact val_loss:2.06967197 val_bpb:1.22577632`

Observed deltas:

- `block_layers=3` was about `2.12x` slower than baseline on `step_avg`
- `block_layers=9` was about `1.98x` slower than baseline on `step_avg`
- `block_layers=3` was worse than baseline by about `+1.3686 val_bpb` at step `200`
- `block_layers=9` was worse than baseline by about `+0.3995 val_bpb` at step `200`
- `block_layers=9` remained far worse than baseline after short-run quantized roundtrip:
  - `2.15187086` vs `1.22577632`

Memory impact:

- baseline proxy peak allocated: `10257 MiB`
- `block_layers=9` peak allocated: `15392 MiB`
- observed increase: about `+5.0 GiB`

Parameter delta:

- baseline proxy params: `17059912`
- AttnRes v1 params: `17069128`
- observed increase: `+9216` parameters

## Interpretation

- The first AttnRes v1 implementation is not promotion-worthy in its current form.
- The failure is not only a throughput problem; the early validation curve is also much worse than baseline.
- Coarser grouping (`block_layers=9`) is less bad than `block_layers=3`, which suggests the implementation cost is sensitive to how aggressively the model is forced to route through the AttnRes path.
- Even the better screening variant still misses the pass gate by a wide margin:
  - slowdown is far above the allowed `+10%`
  - quality is far worse than the allowed `0.003 val_bpb` loss threshold

Probable causes worth checking before any rewrite:

- zero-initialized pseudo-query may start too close to uniform averaging over stale states
- injecting AttnRes before both attention and MLP may be too aggressive for this shallow baseline
- the current formulation triggers extra Inductor softmax lowering work and noticeably higher memory use

## Recommendation

Do not promote this implementation to the merge phase.

If Attention Residuals stay on the roadmap, the next attempt should be smaller and more conservative:

- keep the current draft as the negative baseline for AttnRes v1
- try a less invasive variant before any full rerun:
  - bias initialization toward the current partial state
  - inject at one site only instead of both attention and MLP
  - or move the mechanism to a deeper/shared-depth model where the extra routing has a clearer use case

## Submission-Contract Notes

- these runs were coherent with the current training/eval contract:
  - fixed challenge dataset
  - fixed validation split
  - post-quant roundtrip metric available when the run was allowed to finish
- current `code bytes` accounting still undercounts modular development code under `research/`
- these are research notes only, not frozen submission artifacts
