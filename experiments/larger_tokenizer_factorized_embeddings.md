# Larger Tokenizer Plus Factorized Embeddings

## Hypothesis

A larger tokenizer can improve the leaderboard metric directly by lowering `tokens_per_byte`, and factorized or adaptive tied embeddings can prevent the vocab increase from donating all of that gain back in artifact bytes.

## Why It May Help This Specific Competition

- `val_bpb = bits_per_token * tokens_per_byte`, so tokenizer quality matters directly.
- The current baseline uses a small `1024`-vocab tokenizer.
- Vocabulary growth mostly taxes embeddings and output projection, which can be parameterized more efficiently than a naive full matrix.

## Competition / Rules Risk

- Medium to high rules risk.
- Tokenizer changes are allowed, but maintainers will scrutinize `val_bpb` accounting and reproducibility.
- This branch must produce an auditable metric proof and deterministic shard/tokenizer artifacts.

## Minimal Implementation Design In This Repo

Split the branch into two controlled stages.

Stage A: larger tokenizer with simple factorized tied embeddings.

- rebuild tokenizer and shards with the existing `data/` workflow
- first sweep `VOCAB_SIZE in {2048, 4096}`
- replace the embedding/head path with a factorized tied parameterization:
  - token lookup table `V x E`
  - projection `E x D`
  - output reuses the same factors in tied form

Stage B: adaptive tied embeddings only if Stage A is promising.

- keep tied output semantics
- allow a higher-capacity slice for frequent tokens and a compressed slice for the tail
- avoid a full adaptive-softmax rewrite in v1

Recommended config surface:

- `TOKENIZER_EXPERIMENT=sp2048|sp4096`
- `FACTOR_EMBED_ENABLE=1`
- `FACTOR_EMBED_DIM=192|256`
- `ADAPTIVE_EMBED_ENABLE=0|1`
- `ADAPTIVE_TAIL_RATIO=0.5`

Required implementation notes:

- document tokenizer bytes, vocab, and training corpus exactly
- log both `tokens_per_byte` and final `val_bpb`
- keep the evaluation split fixed at the document level
- prove that byte accounting is comparable across tokenizer variants

## Variant Ladder

1. `sp2048` + factorized tied embeddings
2. `sp4096` + factorized tied embeddings
3. `sp4096` + adaptive tied embeddings if variant 2 is byte-limited

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- pre-export `val_bpb`
- `tokens_per_byte`
- compressed model bytes
- tokenizer artifact bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- metric-proof notes for tokenizer correctness

## Acceptance Criteria

`pass`:

- improve final exact `val_bpb` by at least `0.004`
- keep total artifact bytes under `16,000,000`
- produce a reproducible tokenizer/data manifest

`promote`:

- improve final exact `val_bpb` by at least `0.006`
- keep artifact bytes within cap without hand-wavy accounting
- metric proof is documented clearly enough for external review

## Kill Criteria

- tokenizer change improves `tokens_per_byte` but loses too much in bits-per-token
- embedding/output bytes erase most of the tokenizer win
- metric accounting becomes contentious or hard to audit

## Estimated Engineering Cost

- Medium to high
- Expected effort: `3-6 engineer days`

## Merge Compatibility

- Strong candidate to combine later with `compression_aware_dense_transformer`
- Strong candidate to combine later with `multi_token_prediction`
- Compatible later with `low_rank_ffn_factorization`
- Do not combine with another tokenizer-changing branch until this one is isolated

## References

- local dataset and tokenizer workflow in `data/README.md`
- local data experiment rules in `experiments/data_selection.md`
