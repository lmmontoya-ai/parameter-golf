# 2026-03-19: Dense `sp1024` Sliding-Window Control `A0` Draft Notes

## Purpose

Record the first completed run in the mainline score lane on the `4xH100 -> 1200s` proxy. This is the control for the sliding-window evaluation matrix:

- `EVAL_SEQ_LEN=1024`
- `EVAL_WINDOW_STRIDE=0`

## Hardware And Environment

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- pod image: `pytorch/pytorch:latest`
- remote Python: `3.10.13`
- remote torch: `2.2.1`
- remote workspace: `/root/parameter-golf`

Environment compatibility fixes required on the pod:

- `dist.init_process_group(..., device_id=...)` fallback for older torch
- version-tolerant `torch.backends.cuda` backend toggles
- `np.uint16` shard loading widened to `int32`
- local RMSNorm fallback for missing `torch.nn.functional.rms_norm`
- manual GQA K/V repetition for older `scaled_dot_product_attention`
- `libcuda.so` symlink for Inductor
- static masked loss reduction for sliding-window validation

## Command

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_a0_dense_sw0_v9 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
EVAL_SEQ_LEN=1024 \
EVAL_WINDOW_STRIDE=0 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Raw log copied locally:

- [logs/vast_4xh100_a0_dense_sw0_v9.log](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_a0_dense_sw0_v9.log)

## Key Results

- `train_loader:dataset:fineweb10B_sp1024 train_shards:80`
- `val_loader:shards pattern=/root/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021845`
- `eval_window:seq_len=1024 stride=0 mode=contiguous total_windows=60568 scored_tokens=62021632`
- `tokenizer_tokens_per_byte:0.410521`
- `model_params:17059912`
- `world_size:4 grad_accum_steps:2`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:1200.000`
- `step_avg:104.54ms`
- `stopping_early: wallclock_cap train_time:1200006ms step:11479/20000`
- `peak memory allocated: 11602 MiB reserved: 12004 MiB`
- `Serialized model int8+zlib: 15822210 bytes`
- `Code size: 76354 bytes`
- `Total submission size int8+zlib: 15898564 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.07868947 val_bpb:1.23111699`
- `final_int8_zlib_roundtrip eval_time:4428ms`

## Interpretation

- The pod is now operational enough to run the mainline lane end-to-end after the compatibility patches.
- This control establishes the correct baseline for the sliding-window eval matrix in the current remote environment.
- Compared with the earlier local `4xH100` proxy baseline, this remote torch-2.2.1 control is a bit worse on the final exact metric, but it is stable and reproducible enough to serve as the search control.
- The evaluator path did not yet test overlapping windows; this run only confirms the contiguous baseline with the new mask-compatible eval implementation.

## Submission-Contract Notes

- The artifact stayed under the `16,000,000`-byte cap after int8+zlib export.
- The run used the fixed `sp1024` dataset and the full validation split.
- The local `code bytes` accounting is still a development approximation because active code now lives outside `train_gpt.py`.

## Next Step

- run `A1` with `EVAL_WINDOW_STRIDE=256`
- if that is clearly not better, continue the matrix with `A2=128` and `A3=64`
- if a sliding-window variant beats the control by the planned gate, promote that setting as the default eval mode for later tokenizer/factorized-embedding comparisons
