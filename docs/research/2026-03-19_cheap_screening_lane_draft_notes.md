# 2026-03-19 Cheap Screening Lane Draft Notes

## Context

This is the cheap-GPU screening lane for the dense / compression-oriented branch family. The run was executed on a single Vast.ai `1x RTX 4090` worker (`ssh1.vast.ai`, instance `33146548`) after the deterministic dataset gate passed for `fineweb10B_sp1024`.

Why this lane:
- low cost for early rejection
- fast enough to rank branches before spending `4xH100` time
- internally fair because every comparison used the same local lane settings

Common lane settings:

```bash
CC=gcc CXX=g++ \
MAX_WALLCLOCK_SECONDS=300 \
TRAIN_BATCH_TOKENS=262144 \
VAL_BATCH_SIZE=262144 \
WARMDOWN_ITERS=300
```

## Baseline Control

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
env PYTHONUNBUFFERED=1 CC=gcc CXX=g++ \
  MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 \
  python train_gpt.py
```

Key results:
- `model_params: 17059912`
- `step_avg: 484.60ms`
- `step:620/20000 val_loss:2.4718 val_bpb:1.4639`
- `Total submission size int8+zlib: 10804417 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.46937823`

Interpretation:
- This is the control for the cheap lane.
- It is substantially worse than the validated `4xH100` proxy baseline, but it establishes the lane’s own reference.

## Dense LR / Warmdown Retune

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
env PYTHONUNBUFFERED=1 CC=gcc CXX=g++ \
  MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 \
  WARMDOWN_ITERS=300 EMBED_LR=0.4 TIED_EMBED_LR=0.04 MATRIX_LR=0.03 SCALAR_LR=0.03 \
  python train_gpt.py
```

Key results:
- `step_avg: 481.66ms`
- `step:623/20000 val_loss:2.4240 val_bpb:1.4356`
- `Total submission size int8+zlib: 12260347 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.43684922`

Interpretation:
- This is the best result in the cheap lane.
- It improved the final exact score by about `0.0325 val_bpb` over the cheap-lane control.
- This keeps the dense branch alive for more serious follow-up on the main proxy lane.

## Dense Retune + fp16 Tied-Embedding Export

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
env PYTHONUNBUFFERED=1 CC=gcc CXX=g++ \
  MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 \
  WARMDOWN_ITERS=300 EMBED_LR=0.4 TIED_EMBED_LR=0.04 MATRIX_LR=0.03 SCALAR_LR=0.03 \
  INT8_PASSTHROUGH_FP16_NAME_PATTERNS=tok_emb.weight \
  python train_gpt.py
```

Key results:
- `step_avg: 480.33ms`
- `step:625/20000 val_loss:2.4237 val_bpb:1.4354`
- `Total submission size int8+zlib: 12586533 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.43643814`

Interpretation:
- This was only a tiny improvement over the dense retune.
- The byte cost went up by about `326 KB`.
- On this lane, fp16 tied-embedding export is not yet justified as a first-order win.

## Low-Rank FFN / Projection Factorization

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
env PYTHONUNBUFFERED=1 CC=gcc CXX=g++ \
  MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 \
  WARMDOWN_ITERS=300 EMBED_LR=0.4 TIED_EMBED_LR=0.04 MATRIX_LR=0.03 SCALAR_LR=0.03 \
  LOW_RANK_FFN_ENABLE=1 LOW_RANK_FFN_RATIO=0.5 \
  python train_gpt.py
```

Key results:
- `model_params: 14700616`
- `step_avg: 469.16ms`
- `step:640/20000 val_loss:2.4551 val_bpb:1.4540`
- `Total submission size int8+zlib: 10773355 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.45557399`

Interpretation:
- Low-rank FFN is a reject in this lane.
- It is smaller and a bit faster, but it loses too much quality to the dense retune.

## Multi-Token Prediction

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
env PYTHONUNBUFFERED=1 CC=gcc CXX=g++ \
  MAX_WALLCLOCK_SECONDS=300 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 \
  WARMDOWN_ITERS=300 EMBED_LR=0.4 TIED_EMBED_LR=0.04 MATRIX_LR=0.03 SCALAR_LR=0.03 \
  MTP_ENABLE=1 MTP_K=2 MTP_LOSS_WEIGHT=0.3 \
  python train_gpt.py
```

Key results:
- early loss was much worse than dense
- `step_avg: 482.11ms`
- `step:623/20000 val_loss:2.6575 val_bpb:1.5739`
- `Total submission size int8+zlib: 12301850 bytes`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.57527900`

Interpretation:
- This is a clear reject on the cheap lane.
- MTP hurts optimization enough that it is not worth more time here.

## Tokenizer Integration Sanity Check

Command:

```bash
cd /root/parameter-golf
. .venv/bin/activate
python data/cached_challenge_fineweb.py --variant sp4096 --train-shards 0 --verify-only
```

Result:
- failed immediately with:
  - `ValueError: dataset fineweb10B_sp4096 not found in datasets/manifest.json`

Interpretation:
- `SP-4096` is not ready in the current repo manifest.
- Tokenizer work should remain a separate future branch until the data/manifest plumbing exists.

## Summary

What looks promising:
- dense LR / warmdown retune
- possibly a later dense-side compression-aware follow-up on the main proxy lane

What is weak or blocked:
- fp16 tied-embedding export on this lane is only a marginal gain
- low-rank FFN is a reject here
- MTP is a reject here
- `SP-4096` is not available in the current manifest

Raw logs:
- [dense base](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/dense_base_300s.log)
- [dense retune](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/dense_retune_300s.log)
- [dense fp16 embed](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/dense_fp16embed_300s.log)
- [low-rank FFN](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/low_rank_ffn_300s.log)
- [MTP k=2](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/mtp_k2_300s.log)
- [SP-4096 verify-only](/Users/lumontoya/research/openai/parameter-golf/logs/cheap_screen_2026-03-19/sp4096_verify_only.log)
