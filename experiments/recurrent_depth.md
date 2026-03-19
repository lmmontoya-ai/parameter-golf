# Recurrent Depth

## Hypothesis

A paper-shaped latent recurrent model should buy much better quality per byte than the current fixed-depth baseline by replacing repeated transformer blocks with a small shared recurrent core that can be unrolled multiple times.

## Why It May Help This Specific Competition

- Parameter Golf rewards effective depth bought with compute instead of stored weights.
- The baseline spends most of its bytes on repeated dense transformer blocks.
- A recurrent core is a cleaner byte-saving mechanism than the previous "share layers by logical index" idea because it preserves the paper's latent-state recurrence structure.

## Competition / Rules Risk

- Low to medium rules risk.
- No tokenizer, dataset, or export changes are required in v1.
- Eval-time compute must stay bounded, so v1 uses fixed train/eval recurrence counts and does not rely on extra test-time loops.

## Minimal Implementation Design In This Repo

First implementation target: **RecurrentGPT v1**.

Keep these baseline pieces unchanged in v1:

- tokenizer
- dataset
- validation path
- export path
- tied embeddings
- Muon + Adam optimizer split
- current attention and `relu^2` MLP internals

Replace the current recurrent-depth plan with this macro-architecture:

- unique **prelude** blocks
- shared **core** blocks reused across recurrent steps
- unique **coda** blocks
- one shared input-reinjection adapter per recurrent step

Default configuration:

- `RECURRENT_ENABLE=1`
- `RECURRENT_PRELUDE_LAYERS=1`
- `RECURRENT_CORE_LAYERS=2`
- `RECURRENT_STEPS=3`
- `RECURRENT_BACKPROP_STEPS=3`
- `RECURRENT_CODA_LAYERS=2`
- `RECURRENT_EVAL_STEPS=3`
- `RECURRENT_STATE_INIT=like_init`
- `RECURRENT_INPUT_INJECTION=linear_concat`

Effective depth constraint:

- `RECURRENT_PRELUDE_LAYERS + RECURRENT_CORE_LAYERS * RECURRENT_STEPS + RECURRENT_CODA_LAYERS == NUM_LAYERS`

V1 forward path:

1. embed tokens and apply the current embedding RMSNorm preprocessing
2. run unique prelude blocks to produce `input_embeds`
3. initialize a separate latent state `x`
4. for each recurrent step:
   - optionally run prefix steps under `torch.no_grad()` when `RECURRENT_BACKPROP_STEPS < RECURRENT_STEPS`
   - apply one shared adapter to `concat(x, input_embeds)`
   - run the shared core blocks
5. run unique coda blocks
6. apply final norm and LM head exactly as in the baseline

State init and input injection in v1:

- `RECURRENT_STATE_INIT=like_init` uses a truncated normal with `tied_embed_init_std`
- `RECURRENT_INPUT_INJECTION=linear_concat` uses one shared bias-free `CastedLinear(2 * model_dim, model_dim)`

What v1 explicitly does **not** do:

- no encoder/decoder skip path
- no logical-depth adapters
- no qk bias
- no gated SiLU MLP
- no Takase init
- no randomized recurrence sampling
- no adaptive exits or KV-cache-per-step inference logic

## Variant Ladder

1. `prelude=1`, `core=2`, `steps=3`, `coda=2`
2. same shape, but `RECURRENT_BACKPROP_STEPS=2`
3. same train shape, then test `RECURRENT_EVAL_STEPS=4` or `5` only if variant 1 is already stable

Do not move to eval-only extra steps if the fixed-step train/eval model is not already competitive.

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- parameter count reduction vs baseline
- compile stability notes

Current development caveat:

- while the code lives in `research/`, the current `train_gpt.py` code-size logging undercounts true submission bytes because it only measures the runner file
- treat logged `code bytes` as development-only until a runnable snapshot is frozen into `records/`

## Acceptance Criteria

`pass`:

- parameter count drops by about `35%` or more
- compressed model bytes drop by at least `25%`
- `final_int8_zlib_roundtrip_exact val_bpb` is no worse than `+0.010` vs the validated `4xH100` baseline proxy
- training `step_avg` is no worse than `+10%`
- compile stays enabled and stable

`promote`:

- compressed model bytes drop by `25%+`
- `final_int8_zlib_roundtrip_exact val_bpb` is no worse than `+0.005` vs baseline or better
- the same config survives a full `4xH100 -> 1200s` proxy run cleanly
- only after that, test `RECURRENT_EVAL_STEPS=4` or `5` as a separate v1.1 inference-scaling branch

## Kill Criteria

- byte savings under `20%`
- quality loss worse than `+0.010 val_bpb`
- compile/runtime instability comparable to the failed Attention Residuals branch
- major optimizer redesign becomes necessary

## Estimated Engineering Cost

- Medium to high
- Expected effort: `2-4 engineer days`

## Merge Compatibility

- Strong candidate to combine with `bitnet_ternary_qat`
- Potentially compatible with `masa_weight_sharing`
- Leave `attention_residuals` separate until recurrent depth works in isolation
- Competes directly with `looped_llms` in the first round and should not be auto-merged with it

## References

- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [ar5iv paper HTML](https://ar5iv.labs.arxiv.org/html/2502.05171v2)
- [seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)
- [`recpre/raven_modeling_minimal.py`](https://raw.githubusercontent.com/seal-rg/recurrent-pretraining/main/recpre/raven_modeling_minimal.py)

Observed draft note:

- `docs/research/2026-03-18_recurrent_depth_draft_notes.md`
- `docs/research/2026-03-18_recurrent_depth_param_eval_sweep_draft_notes.md`
- `docs/research/2026-03-18_recurrent_depth_eval4_only_draft_notes.md`
- `docs/research/2026-03-18_recurrent_depth_train4_eval4_draft_notes.md`
- `docs/research/2026-03-18_val_only_and_recurrent44_draft_notes.md`
- `docs/research/2026-03-18_recurrent44_deeper223_real_train_proxy_draft_notes.md`
- `docs/research/2026-03-18_recurrent44_deeper223_w576_real_train_proxy_draft_notes.md`
