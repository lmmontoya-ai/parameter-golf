# 2026-03-19: Frontier Decomposition `A1/A2` Draft Notes

## Purpose

Record the first optimizer/schedule decomposition runs on the pinned `4xH100 -> 1200s` mainline lane:

- `A1`: schedule-only retune on vanilla Muon
- `A2`: `A1` plus `MUON_VARIANT=normuon`

These runs are intended to isolate:

- `A1 - C0` = schedule gain
- `A2 - A1` = NorMuon gain

## Lane

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- runtime policy: `uv` environment with pinned `torch==2.10`
- dataset gate: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --verify-only`

## Control

Use the completed remote dense control note:

- [2026-03-19_dense_sw0_control_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_dense_sw0_control_draft_notes.md)

Key control score:

- `final_int8_zlib_roundtrip_exact val_bpb: 1.23111699`

## Runs

### A1: schedule-only retune

- command:
```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_a1_sched_v4 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
EVAL_SEQ_LEN=1024 \
EVAL_WINDOW_STRIDE=0 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
/root/parameter-golf/.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```
- raw log:
  - [vast_4xh100_a1_sched_v4.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_a1_sched_v4.txt)
- key results:
  - `eval_window:seq_len=1024 stride=0 mode=contiguous total_windows=60568 scored_tokens=62021632`
  - `tokenizer_tokens_per_byte:0.410521`
  - `model_params:17059912`
  - `stopping_early: wallclock_cap train_time:1200128ms step:10340/20000`
  - `step_avg:116.07ms`
  - `Serialized model int8_zlib: 15837788 bytes`
  - `Code size: 91713 bytes`
  - `Total submission size int8_zlib: 15929501 bytes`
  - `final_int8_zlib_roundtrip_exact val_loss:2.06242022 val_bpb:1.22148143`
- delta vs `C0`:
  - `1.22148143 - 1.23111699 = -0.00963556 val_bpb`

### A2: schedule-only retune + NorMuon

- command:
```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_a2_normuon_v1 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
EVAL_SEQ_LEN=1024 \
EVAL_WINDOW_STRIDE=0 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_VARIANT=normuon \
NORMUON_BETA2=0.95 \
/root/parameter-golf/.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```
- raw log:
  - [vast_4xh100_a2_normuon_v1.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_a2_normuon_v1.txt)
- status:
  - killed early at step `6000` to free the `4xH100` lane for Phase B after three matched checkpoints all underperformed `A1`
- matched checkpoints:
  - step `2000`: `1.3293` vs `A1 1.3287` (`+0.0006`)
  - step `4000`: `1.2789` vs `A1 1.2780` (`+0.0009`)
  - step `6000`: `1.2620` vs `A1 1.2608` (`+0.0012`)
- throughput:
  - about `85.0 ms/step`, essentially identical to early `A1`

## Interpretation

- `A1` is a real positive result on the clean contiguous lane.
- The schedule-only retune improved the exact exported score by about `0.0096 val_bpb` versus `C0`.
- The artifact stayed under the `16,000,000`-byte cap, though with less headroom than the control.
- The cost of the improvement was slower wallclock efficiency:
  - `C0`: about `104.54 ms/step`
  - `A1`: about `116.07 ms/step`
- That tradeoff is still acceptable for Phase A because the score gain exceeded the planned `0.005` pass gate before NorMuon was even added.
- `A2` did not show evidence of a NorMuon win.
- At every matched checkpoint through step `6000`, `A2` was slightly worse than `A1` while running at essentially the same step time.
- That pattern is strong enough to treat NorMuon as negative for this lane version and kill it early rather than spend the full `1200s` budget.

## Next Step

- keep `A1` as the winning Phase A checkpoint
- run the pure sliding-window eval matrix from the frozen `A1` checkpoint
- only return to optimizer variants later if another branch changes the training/export regime enough to justify re-testing NorMuon
