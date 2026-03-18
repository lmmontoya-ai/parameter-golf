# Experiment 0: Baseline Reproduction On `8xH100`

## Hypothesis

The current repository baseline can be reproduced on a single-node `8xH100` machine closely enough that later experiments can attribute differences to model changes rather than infrastructure drift.

## Why It May Help This Specific Competition

- It validates the exact dataset, tokenizer, distributed launch, wallclock behavior, and export path used by this repo.
- It establishes the expected score, byte budget, and throughput envelope for all later experiments.
- It prevents architecture branches from hiding NCCL, data, or environment regressions.

## Competition / Rules Risk

- Low risk.
- This is a prerequisite, not a speculative research branch.
- The reference values come from the existing record folder in the repo.

## Minimal Implementation Design In This Repo

Use the existing baseline code path with no architectural changes.

Reference configuration from `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`:

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=hf_verify_sp1024_8gpu \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Required inputs:

- published `fineweb10B_sp1024` dataset
- published `fineweb_1024_bpe.model` tokenizer
- single node with `8xH100`
- local fast disk, not network-mounted training shards

Reference values from the baseline record:

- `model_params:17059912`
- `world_size:8 grad_accum_steps:1`
- `step:13780/20000 val_loss:2.0606 val_bpb:1.2172`
- `stopping_early: wallclock_cap train_time:600038ms step:13780/20000`
- `Serialized model int8+zlib: 15815847 bytes`
- `Total submission size int8+zlib: 15863489 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.07269931 val_bpb:1.22436570`

Expected train-loader lines:

- `train_loader:dataset:fineweb10B_sp1024`
- `val_loader:shards pattern=.../fineweb_val_*.bin`

Pass tolerances for reproduction:

- final exact `val_bpb` within `+-0.0050` of `1.22436570`
- total artifact bytes within `+-250,000` of `15,863,489`
- training stop step within `+-600` steps of `13,780`
- measured train time between `595,000ms` and `605,000ms`
- step average between `40ms` and `47ms`
- no unexplained missing log lines from the expected baseline outputs above

If the run is outside tolerance, use this failure triage in order:

1. Data mismatch:
   - wrong dataset root
   - wrong shard count
   - missing full validation split
2. Tokenizer mismatch:
   - wrong SentencePiece model
   - wrong vocab size
3. Infrastructure mismatch:
   - missing `8xH100`
   - slow storage
   - NCCL topology / IB behavior
4. Software drift:
   - different PyTorch / CUDA behavior
   - compile instability
5. Hidden code drift:
   - local edits in `train_gpt.py`

Do not proceed to architecture experiments until at least one run passes and the run logs are archived in a local experiment folder.

## Variant Ladder

1. Single exact baseline run with the reference command
2. One repeat run to measure noise
3. Only if needed: environment debugging runs that preserve the exact baseline model and data

No architectural or optimizer changes belong in this experiment.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- wallclock stop step
- train wallclock
- eval wallclock
- peak memory
- software versions from the log header

## Acceptance Criteria

- One run passes the tolerance gate above.
- A second run is either within the same tolerance or close enough that the remaining spread is clearly normal run-to-run noise.
- The reproduction logs are stored and can be cited by later experiment docs.

## Kill Criteria

- If three well-configured runs on valid `8xH100` hardware all miss tolerance and no concrete infrastructure bug is identified, stop architecture work and debug the stack first.

## Estimated Engineering Cost

- Low
- Expected effort: `0.5-1 engineer day`

## Merge Compatibility

- This is a prerequisite for every other experiment.
- No combination work is allowed before this experiment passes.
