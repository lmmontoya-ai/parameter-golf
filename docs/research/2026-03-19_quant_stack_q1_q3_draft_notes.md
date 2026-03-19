# 2026-03-19: Quant Stack `Q1` Draft Notes

## Purpose

Record the first faithful dense continuation after promoting the `A1 + stride64` control:

- `Q0`: `300s` smoke for `Q1`
- `Q1`: full `1200s` run with:
  - `FAKE_QUANT_ENABLE=1`
  - `FAKE_QUANT_BITS=6`
  - `FAKE_QUANT_SCOPE=casted_linear`
  - `EXPORT_FORMAT=int6_zstd`

This note closes the branch if `Q1` fails the promoted `stride64` gate against the current control:

- control exact: `1.18718097 val_bpb`

## Lane

- provider: Vast.ai
- hardware: `4xH100 80GB HBM3`
- offer used: `32086075`
- location: `Nebraska, US`
- pod image: `pytorch/pytorch:latest`
- remote environment:
  - `build-essential`
  - `uv`
  - `torch 2.10.0+cu128`
  - `zstandard 0.25.0`
- deterministic dataset gate:
  - `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
  - `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --verify-only`

## Commands

### Q0 smoke

Same as `Q1`, except:

- `RUN_ID=vast_4xh100_q0_q1_smoke_v1`
- `MAX_WALLCLOCK_SECONDS=300`

### Q1 full run

```bash
CC=gcc \
CXX=g++ \
NCCL_IB_DISABLE=1 \
RUN_ID=vast_4xh100_q1_int6_v1 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
EVAL_SEQ_LEN=1024 \
EVAL_WINDOW_STRIDE=64 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
FAKE_QUANT_ENABLE=1 \
FAKE_QUANT_BITS=6 \
FAKE_QUANT_SCOPE=casted_linear \
EXPORT_FORMAT=int6_zstd \
/root/parameter-golf/.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```

### Eval-only reuse

The final model is written to `/root/parameter-golf/final_model.pt`, not under `checkpoints/`.

Both eval-only runs used:

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
INIT_MODEL_PATH=/root/parameter-golf/final_model.pt \
EVAL_SEQ_LEN=1024 \
EVAL_WINDOW_STRIDE=<0_or_64> \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
FAKE_QUANT_ENABLE=1 \
FAKE_QUANT_BITS=6 \
FAKE_QUANT_SCOPE=casted_linear \
EXPORT_FORMAT=int6_zstd \
/root/parameter-golf/.venv/bin/torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Raw Logs

- [vast_4xh100_q0_q1_smoke_v1.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_q0_q1_smoke_v1.txt)
- [vast_4xh100_q1_int6_v1.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_q1_int6_v1.txt)
- [vast_4xh100_q1_int6_v1_reevals.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_4xh100_q1_int6_v1_reevals.txt)

## Results

### Q0 smoke

- stop: `step 3494` at `300019ms`
- throughput: `85.87 ms/step`
- promoted eval mode during training:
  - `eval_window:seq_len=1024 stride=64 mode=sliding`
- bytes:
  - `Serialized model int6_zstd: 10582096`
  - `Code size: 96997`
  - `Total submission size int6_zstd: 10679093`
- exact post-export score:
  - `final_int6_zstd_roundtrip_exact val_bpb: 1.24768623`

Interpretation:

- `Q0` passed the smoke gate operationally.
- Fake quant, `int6_zstd`, and the export path all ran end to end on the real `4xH100` lane.

### Q1 full run

- stop: `step 13971` at `1200087ms`
- throughput: `85.90 ms/step`
- in-training late validation proxy:
  - best observed: `step 13971 val_bpb 1.1826`
- bytes:
  - `Serialized model int6_zstd: 11488190`
  - `Code size: 96997`
  - `Total submission size int6_zstd: 11585187`
- final exact post-export score from the training run:
  - `final_int6_zstd_roundtrip_exact val_bpb: 1.23894552`

### Eval-only reuse from `final_model.pt`

#### Contiguous diagnostic

- `eval_window:seq_len=1024 stride=0 mode=contiguous`
- exact post-export score:
  - `final_int6_zstd_roundtrip_exact val_bpb: 1.27641977`

#### Promoted `stride64` re-eval

- `eval_window:seq_len=1024 stride=64 mode=sliding`
- exact post-export score:
  - `final_int6_zstd_roundtrip_exact val_bpb: 1.23889609`

This matches the full training-run final closely enough to trust the reuse path.

## Comparison To Current Control

Promoted control:

- [2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md](/Users/lumontoya/research/openai/parameter-golf/docs/research/2026-03-19_sliding_window_eval_a1_checkpoint_draft_notes.md)
- `A1 + stride64` exact: `1.18718097`

`Q1` exact promoted score:

- `1.23889609`

Delta:

- `1.23889609 - 1.18718097 = +0.05171512 val_bpb`

This fails the `Q*` pass gate by a wide margin.

## Interpretation

- `Q1` is a clean negative on the promoted metric.
- The training-time sliding-window proxy improved steadily and briefly reached the high `1.18x` range late in the run.
- The exact `int6_zstd` roundtrip score remained much worse than that proxy:
  - `1.1826` in-training proxy at the wallclock stop
  - `1.2389` exact after the real export roundtrip
- That gap is the main finding from this branch:
  - fake-quantized `int6_zstd` can look much better during training than it actually is after exact export/de-export
- The branch is not close to competitive with the promoted dense control, even though it stays comfortably under the byte cap.

## Decision

- kill `Q1`
- do not spend more `4xH100` time on:
  - `Q2` fp16 passthrough for `tok_emb.weight`
  - `Q3` `MLP_MULT=3`
  - `S1` uniform SWA
  - `S2` surrogate-roundtrip SWA selection

Reason:

- the base `Q1` stack is too far behind the current mainline control for the remaining planned additives to plausibly recover the gap.

## Next Step

- keep `A1 + stride64` as the current dense mainline control
- treat this run as evidence that `int6_zstd + STE` alone is not enough in this repo
- if this frontier route is revisited, it should use a materially different quantization/export recipe rather than simply stacking `Q2/Q3/S1/S2` on top of this failed base
