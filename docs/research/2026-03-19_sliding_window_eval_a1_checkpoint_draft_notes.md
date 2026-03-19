# 2026-03-19: Sliding-Window Eval From Frozen `A1` Checkpoint Draft Notes

## Purpose

Record Phase `B` of the frontier decomposition on the pinned `4xH100 -> 1200s` lane:

- freeze the winning Phase `A` checkpoint from `A1`
- re-evaluate that same model with different validation window settings
- isolate the pure eval-time gain from overlapping sliding windows

This is a checkpoint-reuse matrix, not a retraining matrix.

## Lane

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- runtime policy: `uv` environment with pinned `torch==2.10`
- frozen checkpoint:
  - `/root/parameter-golf/checkpoints/vast_4xh100_a1_sched_v4_final_model.pt`
- local raw logs:
  - [vast_4xh100_b0_a1_contig.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_b0_a1_contig.txt)
  - [vast_4xh100_b1_a1_sw256.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_b1_a1_sw256.txt)
  - [vast_4xh100_b2_a1_sw128.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_b2_a1_sw128.txt)
  - [vast_4xh100_b3_a1_sw64.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_b3_a1_sw64.txt)
  - [vast_4xh100_b4_a1_seq2048_sw64_v2.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_b4_a1_seq2048_sw64_v2.txt)

## Eval-Only Command Pattern

All runs used the same saved checkpoint and changed only the eval window settings:

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=<run_id> \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=0 \
WARMUP_STEPS=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=0 \
VAL_LOSS_EVERY=1 \
INIT_MODEL_PATH=/root/parameter-golf/checkpoints/vast_4xh100_a1_sched_v4_final_model.pt \
EVAL_SEQ_LEN=<seq_len> \
EVAL_WINDOW_STRIDE=<stride> \
/root/parameter-golf/.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Results

All runs kept:

- `tokenizer_tokens_per_byte: 0.410521`
- `model_params: 17059912`
- `Serialized model int8_zlib: 15837788 bytes`
- `Total submission size int8_zlib: 15929989 bytes`

### B0: contiguous control re-eval

- `eval_window:seq_len=1024 stride=0 mode=contiguous total_windows=60568 scored_tokens=62021632`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.22150721`
- `final_int8_zlib_roundtrip eval_time: 3266ms`

This reproduces the saved `A1` checkpoint closely enough to treat the reuse path as valid.

### B1: stride `256`

- `eval_window:seq_len=1024 stride=256 mode=sliding total_windows=242270 scored_tokens=62021845`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.18813150`
- delta vs `B0`: `-0.03337571`
- `final_int8_zlib_roundtrip eval_time: 13124ms`

### B2: stride `128`

- `eval_window:seq_len=1024 stride=128 mode=sliding total_windows=484539 scored_tokens=62021845`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.18739780`
- delta vs `B0`: `-0.03410941`
- `final_int8_zlib_roundtrip eval_time: 25959ms`

### B3: stride `64`

- `eval_window:seq_len=1024 stride=64 mode=sliding total_windows=969077 scored_tokens=62021845`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.18718097`
- delta vs `B0`: `-0.03432624`
- delta vs `B2`: `-0.00021683`
- `final_int8_zlib_roundtrip eval_time: 52052ms`

### B4: exploratory `seq_len=2048`, stride `64`

- `eval_window:seq_len=2048 stride=64 mode=sliding total_windows=969061 scored_tokens=62021845`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.30505442`
- delta vs `B0`: `+0.08354721`
- `final_int8_zlib_roundtrip eval_time: 115325ms`

## Interpretation

- Sliding-window eval is a major real gain on this lane.
- The gain is already large at stride `256` and improves a bit more at `128` and `64`.
- The curve is flattening by `64`; the improvement from `128 -> 64` is only about `0.00022 val_bpb`.
- `B3` is the best exact score in this matrix:
  - `1.22150721 -> 1.18718097`
  - about `0.0343 val_bpb` better than contiguous eval on the exact same frozen checkpoint
- Eval cost scales up materially:
  - contiguous: about `3.3s`
  - stride `256`: about `13.1s`
  - stride `128`: about `26.0s`
  - stride `64`: about `52.1s`
- Even `stride=64` remains operationally attractive on the `4xH100` lane because the total final eval wallclock is still well below the planned `2x` cap relative to the full training budget.
- Simply extending context length to `2048` without a matching RoPE/NTK change is strongly negative here.

## Decision

- promote `stride=64` as the default eval setting for the next mainline dense comparisons
- keep `stride=128` in mind as the cheaper near-tie if later branches become more eval-expensive
- do not use `EVAL_SEQ_LEN=2048` in this lane without an explicit context-scaling change

## Next Step

- combine the `A1` schedule-only training stack with `stride=64` as the default mainline eval mode
- continue the decomposition with Phase `C` or the tokenizer/factor branch using this eval setting as the promoted baseline
