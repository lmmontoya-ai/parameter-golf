# Attention Residuals

## Hypothesis

Block attention residuals can improve depth-wise information routing with a very small byte overhead, giving the model more selective access to earlier representations than fixed additive residuals. This should be especially useful once the model explores deeper or shared-depth regimes, but is still cheap enough to test on the current baseline stack.

## Why It May Help This Specific Competition

- The baseline is shallow and byte-constrained, so a small number of routing parameters is attractive.
- Attention residuals add almost no artifact pressure compared with adding full new layers.
- The mechanism is compatible with recurrent/shared-depth designs, so even a modest isolated gain could compound later.

## Competition / Rules Risk

- Low rules risk.
- No tokenizer, data, or evaluation changes are required.
- Byte overhead must stay small enough that the baseline margin under `16,000,000` bytes is not consumed by implementation bloat.

## Minimal Implementation Design In This Repo

First implementation target: **Block AttnRes**, not full AttnRes.

Keep these baseline pieces unchanged in v1:

- tokenizer
- dataset
- validation path
- export path
- optimizer split

Proposed configuration knobs:

- `ATTNRES_ENABLE=1`
- `ATTNRES_BLOCK_LAYERS=3`

V1 design:

- Partition the current 9-layer model into `3` macroblocks of `3` transformer layers each.
- Maintain a list of completed macroblock states.
- Add a lightweight AttnRes module consisting of:
  - one `RMSNorm`
  - one learned pseudo-query vector of shape `[model_dim]`
- For a candidate hidden state `h`, compute softmax attention over:
  - completed macroblock states
  - the current partial macroblock state
- Use the resulting weighted sum as the hidden state that enters the next operation site.

Injection policy in v1:

- apply AttnRes before each block's attention sublayer using the current partial block state
- apply AttnRes again before each block's MLP sublayer using the updated partial block state
- count block boundaries in transformer blocks, not in separate attention and MLP sublayers

Implementation boundaries:

- keep the current `Block` attention and MLP modules intact
- store block summaries in `GPT.forward`
- do not redesign the export format
- do not change skip weights or tie-embedding behavior in this experiment
- every run must log the exact AttnRes config in the run log for later comparison

Implementation simplicity threshold:

- no extra KV cache format work
- no full-attention-over-depth implementation
- no custom kernels
- no more than one lightweight AttnRes site per logical injection point

## First Benchmark Pair

The first benchmark pair after CPU smoke tests is a matched `1xH100` proxy A/B comparison with identical data, tokenizer, and wallclock settings.

Baseline proxy command:

```bash
RUN_ID=attnres_v1_proxy_baseline_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_LOG_EVERY=25 \
VAL_LOSS_EVERY=100 \
ATTNRES_ENABLE=0 \
ATTNRES_BLOCK_LAYERS=3 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

AttnRes proxy command:

```bash
RUN_ID=attnres_v1_proxy_block3_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_LOG_EVERY=25 \
VAL_LOSS_EVERY=100 \
ATTNRES_ENABLE=1 \
ATTNRES_BLOCK_LAYERS=3 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Required run-log line for both commands:

- `attnres:enabled=0 block_layers=3 implementation=block_attnres_v1` for the baseline
- `attnres:enabled=1 block_layers=3 implementation=block_attnres_v1` for the AttnRes run

## Variant Ladder

1. `ATTNRES_ENABLE=1`, `ATTNRES_BLOCK_LAYERS=3`
2. `ATTNRES_ENABLE=1`, `ATTNRES_BLOCK_LAYERS=2`
3. `ATTNRES_ENABLE=1`, `ATTNRES_BLOCK_LAYERS=1`

Only move to the next variant if the previous one is not clearly dead.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- parameter count delta vs baseline
- notes on compile stability

Current development caveat:

- while the code lives in `research/`, the current `train_gpt.py` code-size logging undercounts true submission bytes because it only measures the runner file
- treat logged `code bytes` as development-only until a runnable snapshot is frozen into `records/`

## Acceptance Criteria

`pass`:

- both commands in the first benchmark pair finish successfully on the same machine class
- no NaNs, no compile disablement, and no missing `attnres:` config line
- AttnRes proxy slowdown is no worse than `+10%` in `step_avg`
- AttnRes proxy pre-quant `val_bpb` is no worse than baseline by more than `0.003`
- total artifact bytes increase by no more than `50,000`

`promote`:

- the proxy pair passes all `pass` gates above
- improves `final_int8_zlib_roundtrip_exact val_bpb` by at least `0.003`
- total artifact bytes remain under `16,000,000`
- no more than `10%` training slowdown relative to reproduced baseline
- survives at least one full `8xH100` or equivalent verified proxy run with the same AttnRes settings

## Kill Criteria

- the first proxy pair exceeds `+10%` slowdown
- the first proxy pair loses more than `0.003` pre-quant `val_bpb` against baseline
- any variant adds more than `50,000` total bytes without a measurable quality gain
- after the first two variants, no run beats baseline by at least `0.001` on the matched evaluation setup

## Estimated Engineering Cost

- Medium
- Expected effort: `1-2 engineer days`

## Merge Compatibility

- Good candidate to combine with `recurrent_depth`
- Good candidate to combine with `looped_llms`
- Neutral with `data_selection`
- Compatible with `bitnet_ternary_qat`
- Do not combine with `masa_weight_sharing` until each has an isolated positive result

## References

- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [Official Attention Residuals repository](https://github.com/MoonshotAI/Attention-Residuals)
