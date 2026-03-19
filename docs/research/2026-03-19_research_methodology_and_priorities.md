# 2026-03-19 Research Methodology And Priorities

## Why This Repo Strategy Matters

This repo is not only trying to win a benchmark. It is also trying to demonstrate the qualities the challenge is explicitly screening for:

- the ability to optimize the true objective under hard constraints
- research taste about which ideas deserve more time
- clean experimental execution and honest interpretation
- willingness to try unusual ideas without turning the whole program into exploit chasing

The dual objective is therefore:

1. maximize `final_int8_zlib_roundtrip_exact val_bpb`
2. demonstrate research taste, rigor, and creativity under constraints

The public framing around the challenge makes it clear that leaderboard score matters, but “finding a loophole” by itself is not the strongest hiring signal. The stronger signal is a sequence of good decisions: build a faithful baseline, identify the real bottlenecks, test unusual but defensible ideas, and update quickly when the evidence is negative.

## Operating Research Methodology

The operating method for this repo should be:

1. Ground every branch against the true competition metric.
   The metric to optimize is `final_int8_zlib_roundtrip_exact val_bpb`, not raw checkpoint loss and not pre-export `val_bpb`.

2. Use the validated `4xH100 -> 1200s` setup for screening.
   The local reproduction in [2026-03-18_4xh100_baseline_proxy.md](./2026-03-18_4xh100_baseline_proxy.md) is close enough to the public `8xH100 / 600s` baseline to support A/B work.

3. Gate rented-pod runs with deterministic dataset verification.
   Do not start training until the dataset prep flow succeeds and `_download_state.json` reports `ready: true`.

4. Isolate one main variable per experiment.
   Avoid mixed branches until an isolated branch has a clean positive result on the same evaluation setup.

5. Write one dated note for every meaningful run.
   Put the exact command, hardware, train/eval wallclock, `final_int8_zlib_roundtrip_exact val_bpb`, artifact bytes, and interpretation in `docs/research/`.

6. Treat `8xH100` as confirmation hardware, not broad-search hardware.
   Search on the `4xH100` proxy first, then escalate only the branches that clearly deserve real-track validation.

Repo workflow:

- `experiments/` = planned hypotheses, gates, and variant ladders
- `docs/research/` = observed results, strategy conclusions, and environment findings
- `records/` = frozen runnable artifacts

Standard evidence bundle for every serious run:

- exact command
- hardware and provider
- train wallclock and eval wallclock
- `final_int8_zlib_roundtrip_exact val_bpb`
- compressed model bytes, code bytes, and total artifact bytes
- short interpretation of what changed and what should happen next

## What The Current Evidence Says

### Repo-local evidence

- The dense `4xH100` proxy baseline is coherent and strong:
  - `final_int8_zlib_roundtrip_exact val_bpb: 1.22577632`
  - `Total submission size int8+zlib: 15873947 bytes`
  - source: [2026-03-18_4xh100_baseline_proxy.md](./2026-03-18_4xh100_baseline_proxy.md)

- The best current recurrent real-train result is materially better than the earlier recurrent branch, but still behind dense:
  - `final_int8_zlib_roundtrip_exact val_bpb: 1.25095436`
  - `Total submission size int8+zlib: 15492565 bytes`
  - source: [2026-03-18_recurrent44_deeper223_w576_real_train_proxy_draft_notes.md](./2026-03-18_recurrent44_deeper223_w576_real_train_proxy_draft_notes.md)

- Attention Residuals were negative on both runtime and quality:
  - slower by about `2x`
  - far worse early validation
  - source: [2026-03-18_attention_residuals_draft_notes.md](./2026-03-18_attention_residuals_draft_notes.md)

- For the current recurrent recipe, `eval_steps > train_steps` is strongly negative:
  - source: [2026-03-18_recurrent_depth_eval4_only_draft_notes.md](./2026-03-18_recurrent_depth_eval4_only_draft_notes.md)

- Val-only training is real and can move the metric, but it should be handled as a separate rules-facing track:
  - dense val-only reached `1.2591` by step `2200`
  - source: [2026-03-18_val_only_and_recurrent44_draft_notes.md](./2026-03-18_val_only_and_recurrent44_draft_notes.md)

### Upstream PR evidence

The strongest legitimate upstream PR trends currently appear to be:

- compression-aware dense tuning
- `SP-4096`
- sliding-window eval
- selective mixed-precision export
- recurrence as active but trailing

Representative PRs:

- [#53](https://github.com/openai/parameter-golf/pull/53): `SP-4096` plus stride-64 sliding-window evaluation
- [#60](https://github.com/openai/parameter-golf/pull/60): dense baseline-family improvements through initialization, context/eval scaling, and schedule tuning
- [#39](https://github.com/openai/parameter-golf/pull/39): mixed-precision export with deeper dense model
- [#42](https://github.com/openai/parameter-golf/pull/42): fp16 tied embedding plus warmdown/LR retune
- [#33](https://github.com/openai/parameter-golf/pull/33): compression-aware fixed-step research
- [#31](https://github.com/openai/parameter-golf/pull/31): depth-recurrent model with width spend-back

Current read:

- dense, export-aware work is the most credible mainline
- tokenizer and eval improvements are real frontier branches, not just theory
- recurrence is a valid distinctive architecture direction, but it is not currently the best score path

## Ranked Research Directions

| Rank | Direction | Why it matters | Current evidence | Risk | Score upside | Hiring-signal value | Next experiment doc |
|---|---|---|---|---|---|---|---|
| 1 | Compression-aware dense Transformer | Directly optimizes the real exported artifact and keeps the stack stable | Strong local dense baseline; PRs [#33](https://github.com/openai/parameter-golf/pull/33), [#39](https://github.com/openai/parameter-golf/pull/39), [#42](https://github.com/openai/parameter-golf/pull/42), [#60](https://github.com/openai/parameter-golf/pull/60) | Low-medium | High | High | [compression_aware_dense_transformer.md](../../experiments/compression_aware_dense_transformer.md) |
| 2 | Larger tokenizer plus factorized/adaptive tied embeddings | Improves the BPB equation directly while trying to avoid giving the gain back in bytes | PR [#53](https://github.com/openai/parameter-golf/pull/53) and [#37](https://github.com/openai/parameter-golf/pull/37) make this look very real | Medium | Very high | High | [larger_tokenizer_factorized_embeddings.md](../../experiments/larger_tokenizer_factorized_embeddings.md) |
| 3 | Sliding-window / eval-time context improvements | Current PR frontier suggests eval is a first-class optimization surface | PR [#53](https://github.com/openai/parameter-golf/pull/53) and [#50](https://github.com/openai/parameter-golf/pull/50) | Medium | High | Medium-high | [eval_time_adaptation_cache.md](../../experiments/eval_time_adaptation_cache.md) |
| 4 | Early-layer recurrence / layer sharing then spend saved bytes on width | Best distinctive architecture branch we have so far; credible parameter-efficiency story | Local recurrent `w576` result is meaningful but still behind dense; PR [#31](https://github.com/openai/parameter-golf/pull/31) supports the family | Medium | Medium-high | Very high | [early_layer_recurrence_width.md](../../experiments/early_layer_recurrence_width.md) |
| 5 | Low-rank FFNs / projection factorization | Attractive way to trade structure for bytes, then reinvest in width or tokenizer | No local result yet, but strong conceptual fit | Medium | Medium-high | High | [low_rank_ffn_factorization.md](../../experiments/low_rank_ffn_factorization.md) |
| 6 | Hybrid attention + recurrent/linear layers | Could offer a more interesting backbone than plain recurrence, but higher rewrite cost | Mostly paper-driven at this stage; no local evidence yet | High | Medium | High | [hybrid_attention_recurrent_layers.md](../../experiments/hybrid_attention_recurrent_layers.md) |
| 7 | Multi-token prediction | Cheap training-only add-on that could improve sample efficiency without export cost | Interesting literature fit, but smaller expected effect here | Medium | Medium | Medium | [multi_token_prediction.md](../../experiments/multi_token_prediction.md) |
| 8 | Eval-time adaptation / TTT-E2E style work | Distinctive second-submission branch if eval-time adaptation can fit the budget | Rules allow aggressive eval, but this is harder to make robust and budget-safe | High | Medium | High | [eval_time_adaptation_cache.md](../../experiments/eval_time_adaptation_cache.md) |

Deprioritized for now:

- attention residuals
- full-stack looped recurrence as the first architecture bet
- val-only as a mainline branch
- distillation as a first submission path

## Concrete Research Program

### Now

1. Compression-aware dense baseline
   First experiment:
   - dense LR / warmdown / fp16-tied-embedding retune on the current strong dense baseline

2. Sliding-window eval on the dense baseline
   First experiment:
   - stride-64 sliding-window evaluation on the current dense baseline and on the first positive dense retune if it appears

3. Prepare tokenizer branch design with correctness and accounting guardrails
   First experiment:
   - metric-correct `SP-4096` integration only, without architecture changes in the first pass

### Next

- implement `SP-4096` plus cheaper embedding/output parameterization
- if the dense branch is positive, combine dense compression-aware tuning with sliding-window eval
- keep early-layer recurrence alive as the parallel architecture branch

### Later

- low-rank FFN
- hybrid recurrent/attention backbone
- multi-token prediction as an add-on

### Separate Track

- val-only
- TTT-E2E / stronger eval-time adaptation
- distillation

This separation matters:

- `Now` is for the most likely path to a strong score
- `Next` is for high-upside branches that should follow a positive dense result
- `Later` is for structurally interesting work that should not distract from the mainline yet
- `Separate Track` is for rules-sensitive or more speculative directions that may still be worth exploring, but should not steer the mainline

## Decision Rules

- Prefer branches that improve the real exported score, not just raw loss.
- Prefer branches with clean positive evidence over speculative sweeps.
- Keep one mainline branch and one distinctive side branch active at a time.
- If a branch is rules-sensitive, label it clearly and keep it separate.
- Kill branches quickly when the measured premise fails.

Current default operating stance:

- mainline score branch: dense compression-aware work plus tokenizer/eval improvements
- distinctive research branch: early-layer recurrence plus width
- separate side tracks: val-only, stronger eval-time adaptation, and distillation
