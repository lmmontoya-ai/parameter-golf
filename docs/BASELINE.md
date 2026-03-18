# Baseline Training Script

This document explains the baseline `train_gpt.py` used by the initial leaderboard submission in this repository.

It has two goals:

1. Explain what the script is optimizing for in competition terms.
2. Explain, in practical detail, how the baseline model is built, trained, evaluated, and packaged.

## What The Baseline Is Trying To Do

The challenge is not simply "train the smallest model" or "get the lowest validation loss."

The actual objective is:

- train within roughly 10 minutes on `8xH100`
- produce a self-contained artifact under `16,000,000` bytes
- minimize tokenizer-agnostic `val_bpb` on the fixed FineWeb validation split

That means the baseline is solving a joint optimization problem over:

- model quality
- training speed
- model compressibility
- code size

The script is therefore designed to be:

- simple enough to reproduce
- fast enough to fit the wallclock limit
- strong enough to reach a competitive first score
- compressible enough that the final quantized artifact lands under the 16 MB cap

## Exact Baseline Submission

The current baseline record in `records/track_10min_16mb/2026-03-17_NaiveBaseline/` uses the repository snapshot of `train_gpt.py` plus the published `fineweb10B_sp1024` dataset and tokenizer.

Recorded configuration:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TIE_EMBEDDINGS=1`
- `TIED_EMBED_LR=0.05`
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=1024`
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=200`
- `torchrun --standalone --nproc_per_node=8 train_gpt.py`

Recorded outcome:

- parameter count: `17,059,912`
- timed stop: `13,780` training steps
- total train tokens seen: `7,224,688,640`
- final score: `final_int8_zlib_roundtrip_exact val_bpb: 1.22436570`
- compressed model size: `15,815,847` bytes
- counted code size: `47,642` bytes
- total artifact size: `15,863,489` bytes

The most important baseline lesson is that the model has about 17.1 million parameters, which is more than the raw byte budget would suggest, but it still fits because the final submission is an `int8 + zlib` artifact rather than a bf16/fp32 checkpoint.

## High-Level Design

The baseline is a compact transformer language model with a few deliberate efficiency choices:

- tied input/output embeddings
- grouped-query attention (`8` query heads, `4` KV heads)
- `relu^2` MLPs
- RMSNorm instead of LayerNorm
- no learned positional embeddings, using RoPE instead
- no dropout
- learned residual control parameters
- symmetric skip-style structure across the stack
- post-training quantization to int8, followed by zlib compression

Nothing here is exotic by modern standards. The baseline wins its initial score mostly by combining:

- a reasonable small-model architecture
- aggressive speed-oriented implementation choices
- a compression-aware export path

## Script Structure

`train_gpt.py` breaks cleanly into six parts:

1. hyperparameters
2. Muon optimizer
3. tokenizer-aware validation metric
4. post-training quantization
5. data loading
6. model definition and training loop

The rest of this document follows that structure.

## 1. Hyperparameters And Run Shape

The `Hyperparameters` class reads everything from environment variables.

Important defaults:

- dataset path: `./data/datasets/fineweb10B_sp1024`
- tokenizer path: `./data/tokenizers/fineweb_1024_bpe.model`
- validation batch size: `524,288` tokens
- iterations: `20,000`
- warmup steps: `20`
- warmdown iters: `1,200`
- global train batch: `524,288` tokens
- sequence length: `1,024`
- wallclock cap: `600` seconds
- seed: `1337`

Baseline model defaults:

- vocab size: `1,024`
- layers: `9`
- model width: `512`
- query heads: `8`
- KV heads: `4`
- MLP multiplier: `2`
- tied embeddings: enabled
- RoPE base: `10,000`
- logit softcap: `30.0`

Optimizer defaults:

- token embedding LR: `0.6`
- untied head LR: `0.008`
- tied embedding LR: `0.05`
- matrix LR: `0.04`
- scalar LR: `0.04`
- Muon momentum: `0.95`
- Adam betas: `(0.9, 0.95)`

The baseline record overrides only a small subset of these because the defaults already describe the intended baseline.

## 2. Distributed Setup And Effective Batch

The script is built around the idea that the reference training budget is `8` GPUs.

It computes:

- `world_size` from `torchrun`
- `grad_accum_steps = 8 // world_size`

That means the script preserves the same effective global batch as long as `WORLD_SIZE` divides `8`.

Examples:

- on `8` GPUs: `grad_accum_steps = 1`
- on `4` GPUs: `grad_accum_steps = 2`
- on `1` GPU: `grad_accum_steps = 8`

This is an important baseline design choice. It allows local or cheap-GPU debugging without changing the effective training batch, only the runtime.

The script also:

- requires CUDA
- initializes NCCL when distributed
- enables TF32
- forces Flash Attention style SDP backend selection
- disables the slower alternative SDP backends

This is a speed-first implementation.

## 3. Data Pipeline

### Shard Format

Training and validation data live in binary `.bin` shards. Each shard has:

- a fixed 256-int32 header
- a token payload stored as little-endian `uint16`

The loader validates:

- magic number
- version
- expected token count
- exact file size

This is intentionally minimal. There is no PyTorch `DataLoader`, no worker pool, and no random document sampling.

### TokenStream

`TokenStream` reads shards sequentially and wraps around forever.

Properties of this design:

- deterministic
- simple
- cheap in Python overhead
- no random-access complexity

The training loop always consumes a contiguous stream of tokens.

### DistributedTokenLoader

For each batch:

1. the shared stream emits one contiguous chunk for all ranks
2. each rank takes a disjoint slice
3. the slice is shifted by one token to form `(x, y)`

The extra `+1` token per rank is what allows:

- `x = tokens[:-1]`
- `y = tokens[1:]`

with no cross-rank dependency.

### What One Baseline Step Means

With baseline settings:

- global tokens per step: `524,288`
- sequence length: `1,024`
- sequences per global step: `512`

On `8` GPUs:

- tokens per GPU per step: `65,536`
- sequences per GPU per step: `64`

This is a large, throughput-oriented batch.

## 4. Validation Metric: Why The Competition Uses `val_bpb`

This competition allows custom tokenizers, so raw token cross-entropy is not enough for fair comparison.

The script therefore computes two validation metrics:

- `val_loss`: token-level cross-entropy in natural log units
- `val_bpb`: bits per byte

The formula implemented by the script is:

- `bits_per_token = val_loss / ln(2)`
- `tokens_per_byte = total_tokens / total_bytes`
- `val_bpb = bits_per_token * tokens_per_byte`

The byte count is reconstructed from the SentencePiece tokenizer by computing, for every token:

- how many UTF-8 bytes its piece contributes
- whether the token implies a leading space
- whether the previous token is a boundary token

This matters because a tokenizer with fewer tokens per document should not get a free win. The metric tries to measure compression of the underlying text, not compression of a particular token stream.

Validation always uses the full fixed `fineweb_val_*` split. The loader truncates the concatenated validation stream so that it fits an integer number of training sequences.

## 5. The Model

The model class is `GPT`, but the architecture is not a plain GPT-2 clone.

### 5.1 Token Embedding

The model begins with:

- a token embedding table of shape `1024 x 512`
- RMS normalization applied immediately after lookup

There are no positional embeddings. Position information is injected later via RoPE inside attention.

### 5.2 Layer Layout

The model uses `9` blocks total.

Internally it splits them as:

- encoder side: `4` blocks
- decoder side: `5` blocks

This is not an encoder-decoder model in the seq2seq sense. It is still a causal language model. The "encoder" and "decoder" names only describe how the stack is organized around skip connections.

The first half of the stack stores intermediate activations. The second half consumes them in reverse order through learned skip weights. This gives the model a shallow U-Net-like structure while staying fully autoregressive.

### 5.3 Attention

Each block contains a `CausalSelfAttention` module with:

- model dim `512`
- `8` query heads
- `4` KV heads
- head dim `64`

Because `num_heads != num_kv_heads`, the baseline uses grouped-query attention. This reduces K/V parameter count and attention-state size relative to full multi-head attention.

Attention path details:

- separate linear layers for `Q`, `K`, and `V`
- no biases
- RMS normalization applied to `Q` and `K` before attention
- rotary position embedding applied to `Q` and `K`
- learned per-head `q_gain`
- PyTorch `scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
- output projection back to model dim

The output projection is zero-initialized. That keeps early training stable by making each residual branch start near inactive.

### 5.4 MLP

The MLP uses:

- hidden size `2 x 512 = 1024`
- first linear projection
- `relu`
- square the activated values
- second linear projection back to `512`

This is the `relu^2` MLP variant inherited from modded-nanogpt style baselines.

As with attention output projection, the MLP projection back to the residual stream is zero-initialized.

### 5.5 Block-Level Residual Controls

Each block has three learned control mechanisms:

- `attn_scale`: per-channel multiplier on attention output
- `mlp_scale`: per-channel multiplier on MLP output
- `resid_mix`: per-channel two-way mixture between current hidden state `x` and original embedding stream `x0`

`resid_mix` is especially important. Every block can choose, per channel, how much to use:

- the running hidden state
- the original post-embedding representation

So the model has a learned direct path from the input embedding stream into every block.

### 5.6 Cross-Stack Skip Connections

In addition to `resid_mix`, the full model has `skip_weights`, one per matched encoder/decoder pair.

With `9` layers, the model has:

- `4` encoder blocks
- `5` decoder blocks
- `4` learned skip weight vectors

Mechanism:

1. run through the first `4` blocks, saving activations
2. in the last `5` blocks, add back those saved activations in reverse order
3. scale each added skip by a learned per-channel vector

This is another way the baseline trades a small number of control parameters for better optimization behavior.

### 5.7 Output Head And Loss

At the end:

- apply final RMSNorm
- flatten to `(batch * seq, dim)`
- project to vocab logits

Because `TIE_EMBEDDINGS=1`, the output projection reuses the token embedding matrix. There is no separate `lm_head` parameter tensor.

Before cross-entropy, logits are soft-capped:

- `logits = softcap * tanh(logits / softcap)`
- baseline softcap: `30.0`

This keeps logits bounded, which can improve stability in small fast-training setups.

The training loss is standard mean cross-entropy over next-token prediction.

## 6. Parameter Count Breakdown

For the leaderboard baseline, the parameter count is exactly `17,059,912`.

One useful way to understand the architecture is by where those parameters go:

| Component | Parameters |
| --- | ---: |
| token embedding (`1024 x 512`) | 524,288 |
| all attention weights + `q_gain` across 9 blocks | 7,077,960 |
| all MLP weights across 9 blocks | 9,437,184 |
| per-block scales and `resid_mix` across 9 blocks | 18,432 |
| cross-stack `skip_weights` | 2,048 |
| total | 17,059,912 |

Two points matter:

- most parameters live in dense matrix weights
- dense matrices compress well under the baseline export path

That is why a 17.1M-parameter model can still fit under the final byte cap.

## 7. Precision Strategy

The precision policy is deliberate.

The model is first moved to device and cast to bf16, but then:

- every `CastedLinear` module is restored to fp32 storage
- low-dimensional and control parameters are also restored to fp32

During forward passes, `CastedLinear` casts weights to the activation dtype on the fly for the actual matrix multiply.

The result is:

- bf16 compute throughput
- better optimizer/state quality for important weights and control tensors

This is a pragmatic compromise between speed and training stability.

## 8. Optimizer Split

The baseline does not use one optimizer for everything. It splits parameters into groups with different update rules.

### Token Embedding

The token embedding uses Adam.

If embeddings are tied, it uses `TIED_EMBED_LR`; otherwise it uses `EMBED_LR`.

For the baseline:

- tied embeddings are enabled
- token embedding LR is `0.05`

### Transformer Matrix Parameters

Matrix-shaped parameters inside transformer blocks use Muon.

These are the large 2D weights that dominate the model.

Muon:

- keeps a momentum buffer
- optionally uses Nesterov-style update
- orthogonalizes matrix gradients with a Newton-Schulz iteration
- applies the update directly without Adam moments

The intuition is that Muon is specialized for large matrix updates and can work very well in the fast-training regime used by modded-nanogpt-style systems.

### Scalar And Control Parameters

Vectors, scalars, and explicitly named control tensors use Adam with `SCALAR_LR`.

This covers things like:

- `attn_scale`
- `mlp_scale`
- `resid_mix`
- `q_gain`
- `skip_weights`

That split is sensible because these parameters are small and semantically different from the big dense matrices.

### Untied LM Head

If embeddings were not tied, the untied head would get its own Adam optimizer with `HEAD_LR`.

In the baseline record this path is unused because embeddings are tied.

## 9. Learning Rate Schedule

The schedule is intentionally simple.

There is no cosine decay or elaborate schedule. Instead, the script computes a scalar multiplier `scale` and applies it to every optimizer group's base learning rate.

Two regimes exist:

- if there is no wallclock cap, the last `warmdown_iters` steps decay linearly toward zero
- if there is a wallclock cap, the script estimates average step time and starts decaying once the remaining wallclock is less than the expected warmdown duration

This means the decay is adaptive to real throughput, not just nominal step count.

Muon momentum is also warmed up:

- start at `MUON_MOMENTUM_WARMUP_START`
- linearly interpolate to `MUON_MOMENTUM`
- baseline default: `0.85 -> 0.95` over `500` steps

## 10. Compile Warmup

Before timed training begins, the script runs `warmup_steps` full training iterations.

Why:

- compile the model graph
- compile the Muon backend routine
- prime forward, backward, and optimizer code paths
- avoid charging one-time compilation cost to the measured training budget

Crucially, after warmup the script restores:

- the initial model weights
- the initial optimizer states
- the training data stream

So measured training still starts from the true initialization and the true start of the token stream. Warmup affects only compilation and cache state, not the actual optimization trajectory.

This is one of the most important "competition engineering" details in the baseline.

## 11. Main Training Loop

The main loop is straightforward and intentionally low-overhead.

Per step:

1. optionally run validation
2. compute LR multiplier from elapsed time
3. zero grads
4. run `grad_accum_steps` microbatches
5. backward each microbatch with bf16 autocast
6. average training loss for logging
7. update Muon momentum
8. apply current learning rates
9. optional gradient clipping
10. optimizer step
11. log
12. check wallclock cap

Notable properties:

- no dropout
- no activation checkpointing
- no EMA model
- no custom fused kernels beyond what PyTorch gives through compile and SDP
- no stochastic data sampling

This is a baseline optimized for clean throughput and reproducibility, not maximal sophistication.

## 12. Why The Final Score Is Not The Same As The Pre-Quant Score

At the end of training, the script does not stop at evaluating the raw model.

Instead it:

1. saves the raw PyTorch state dict
2. quantizes the state dict to an export format
3. compresses that export with zlib
4. reloads the compressed artifact
5. dequantizes it back into tensors
6. evaluates the round-tripped model

The competition-relevant line is:

- `final_int8_zlib_roundtrip_exact`

This is the right score to optimize against because it is the quality of the compressed artifact that actually fits under the byte budget.

For the baseline record:

- pre-quant stop score: `val_bpb = 1.2172`
- post-quant roundtrip score: `val_bpb = 1.22436570`

So quantization costs about `0.0072` bpb in the recorded baseline.

That is large enough that future experiments must track quantized score directly, not only raw model quality.

## 13. Quantization Format

The export path is simple but carefully chosen.

Large floating-point tensors:

- 2D tensors use per-row int8 quantization
- non-2D tensors use per-tensor int8 quantization

Small float tensors:

- kept as passthrough tensors
- downcast to fp16 for storage when possible

Non-float tensors:

- stored exactly

Important quantization details:

- 2D tensors clip at percentile `99.99984`
- per-row scales are stored as fp16
- some named control tensors are intentionally kept in float form

The baseline is therefore not "all-int8." It is a mixed export scheme that preserves precision where small control tensors matter more than their byte cost.

This is a good baseline tradeoff.

## 14. Why The Baseline Fits Under 16 MB

The artifact budget is:

- compressed model bytes
- plus code bytes

For the baseline record:

- compressed model: `15,815,847`
- code: `47,642`
- total: `15,863,489`

So the submission clears the budget by only about `136,511` bytes.

This is not a large margin.

Implication:

- any architectural improvement that increases model size must either compress equally well or improve quality enough to justify reducing something else

In practice, this means size accounting is a first-class concern, not an afterthought.

## 15. What The Baseline Is Really Doing Well

The baseline's strength is not any one trick. It is the combination of several disciplined choices:

- a small but still reasonably expressive transformer
- tied embeddings to avoid paying twice for vocab parameters
- GQA to reduce attention cost and parameter count
- learned residual controls for easier optimization
- a fast, deterministic streaming data pipeline
- optimizer specialization by parameter type
- compilation warmup outside the measured window
- compression-aware final evaluation

This is why it is a strong starting point even though it looks simple.

## 16. Baseline Weaknesses And Obvious Research Surface

The baseline is intentionally naive in several ways.

It does not explore:

- alternative tokenizers
- recurrence or weight tying across layers
- stronger parameter sharing schemes
- quantization-aware training
- test-time compute
- more aggressive low-rank structure
- smarter export formats than the current int8 clean format
- better use of the 10-minute budget when scaling beyond the simple fixed design

So it is best understood as:

- a reproducible floor
- a reference for score, size, and runtime
- a template for future record submissions

not as a locally optimal architecture.

## 17. What We Should Preserve When We Start Modifying It

As we iterate, a few baseline properties are worth preserving unless we have a good reason not to:

- optimize against `final_int8_zlib_roundtrip_exact`, not raw `val_loss`
- keep byte accounting visible on every serious run
- preserve deterministic and simple data loading
- maintain a clean story for reproducibility
- separate local experimentation from final `8xH100` verification
- treat quantization and compression as part of the model design, not post-processing

## Summary

The baseline is a 9-layer, 512-dim, tied-embedding, GQA causal transformer trained on the published `sp1024` FineWeb export, with Muon on large matrices, Adam on embeddings and control parameters, Flash Attention-style kernels, a deterministic token stream, and an int8+zlib export path.

Its defining idea is simple:

- train a reasonably large dense model very fast
- then rely on a quantization/compression-aware submission path to make that model fit the artifact budget

That is the starting point we need to beat.
