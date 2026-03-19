# 2026-03-19: Mainline Lane `4xH100` Operational Blockers Draft Notes

## Purpose

Record the first remote-lane execution attempt for the mainline score stream and the portability issues encountered while bringing the `4xH100 -> 1200s` baseline control up on Vast.ai.

## Context

- lane: mainline score lane
- target: dense `sp1024` baseline with sliding-window eval controls
- hardware: Vast.ai `4xH100 80GB HBM3`
- pod image: `pytorch/pytorch:latest`
- remote torch: `2.2.1`
- remote workspace: `/root/parameter-golf`

## What Worked

- deterministic dataset prep succeeded on the pod:
  - `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
  - `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --verify-only`
- the cached dataset reported `ready: true`
- the A0 dense control was launched repeatedly with the expected `4xH100` proxy command shape
- the pod shows active `train_gpt.py` processes and H100 memory use while the latest launch is live

## Compatibility Issues Found

The remote torch build is older than the local development stack, so the first control launches exposed several portability issues:

1. `dist.init_process_group(..., device_id=...)` is unsupported on this torch build.
2. `torch.backends.cuda.enable_cudnn_sdp` is not importable here.
3. `torch.from_numpy(np.uint16)` is unsupported here, so shard loading needs a wider integer dtype.
4. `torch.nn.functional.rms_norm` is missing here, so the model needs a fallback RMSNorm implementation.
5. `F.scaled_dot_product_attention(..., enable_gqa=...)` is unsupported here, so GQA needs explicit K/V repetition.
6. The pod was missing `libcuda.so`, even though `libcuda.so.1` existed; a symlink fixed that compile-time lookup.
7. Boolean indexing in the sliding-window validation loss caused a `torch.compile` dynamic-shape failure; the loss reduction was rewritten to a masked sum / denominator.

## Status Of The Live Run

- current live run ID: `vast_4xh100_a0_dense_sw0_v9`
- state: running after the above compatibility fixes
- observed log state:
  - configuration and tokenizer metrics print correctly
  - GPU memory is allocated on all four H100s
  - no completed training step / validation metric has been observed yet

## Interpretation

The mainline lane is not blocked by the experiment design itself. The blockers were environmental and compatibility-related, and they were solvable without changing the research direction.

The most important takeaway is that the repo now needs a more explicit compatibility target for the remote torch build if we want the same code path to run cleanly on both local and rented hardware. The current fixes should be retained until we know whether the live `v9` launch reaches the first training step.

## Next Step

- keep the live `v9` run under observation until it yields the first meaningful training or validation signal
- if it fails again, record the new blocker before changing any experiment knobs
- once one dense control completes, use it as the baseline for the sliding-window A-matrix comparisons
