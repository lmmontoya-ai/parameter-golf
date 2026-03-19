# Hybrid Attention Plus Recurrent Or Linear Layers

## Hypothesis

A hybrid backbone with a small number of full-attention layers and more cheap recurrent or linear-mixing layers can beat pure recurrence or pure dense Transformers on the quality-per-byte frontier.

## Why It May Help This Specific Competition

- Full attention everywhere may be overpaying for capacity in a small-byte regime.
- Pure recurrence or pure SSM swaps are riskier and less aligned with the repo's current strengths.
- A hybrid can preserve the strongest benefits of attention while reducing parameter and compute pressure elsewhere.

## Competition / Rules Risk

- Low rules risk.
- High implementation risk because this is the largest architecture change in this set.

## Minimal Implementation Design In This Repo

Do not attempt a pure Mamba swap in v1.

Use a simple hybrid stack:

- attention blocks at fixed sparse positions
- lightweight recurrent or linear-mixing blocks elsewhere

Concrete first design:

- `HYBRID_BACKBONE_ENABLE=1`
- `HYBRID_ATTN_LAYOUT=every_third`
- `HYBRID_MIXER=griffin_like`
- `HYBRID_LOCAL_ATTN_WINDOW=128`

V1 recurrent/linear block requirements:

- causal
- parameter-cheap
- no custom kernels required
- compatible with `torch.compile`

Good enough first proxy block:

- gated channel mixing
- depthwise causal conv or cheap recurrent update
- residual + norm structure matching the current repo style

V1 deliberately avoids:

- full state-space model rewrite
- custom CUDA kernels
- attention deletion across the whole stack

## Variant Ladder

1. attention every third block, lightweight mixer elsewhere
2. attention only in lower and upper thirds, mixer in the middle
3. same topology with width retune if variant 1 is stable

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- attention-layer count
- mixer-layer count

## Acceptance Criteria

`pass`:

- deliver a meaningful speed or byte advantage over the dense baseline
- while keeping final exact quality within `+0.010 val_bpb`

`promote`:

- improve the best recurrent branch or the best dense branch on quality-per-byte
- remain stable through a full proxy run without custom-kernel dependence

## Kill Criteria

- hybrid mixer blocks destabilize training
- quality is clearly worse than dense and recurrent baselines at matched bytes
- implementation burden starts resembling a new framework rather than a repo experiment

## Estimated Engineering Cost

- High
- Expected effort: `4-7 engineer days`

## Merge Compatibility

- Competes with `recurrent_depth` and `early_layer_recurrence_width`
- Could later combine with `bitnet_ternary_qat`
- Do not combine early with tokenizer changes or eval-time adaptation

## References

- hybrid Transformer/recurrent backbone literature, including Griffin-, HGRN2-, and DeltaNet-style results
