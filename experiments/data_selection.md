# Data Selection

## Hypothesis

Training-data choice can materially improve compression quality without changing the architecture. In benchmark systems adjacent to this repo, dataset quality changes have often produced larger gains than many architecture tweaks.

## Why It May Help This Specific Competition

- The baseline currently uses one published training export and does not test alternative corpora or mixture quality.
- Data changes can improve model quality without increasing artifact bytes.
- This competition allows dataset and tokenizer changes, but they must preserve correct `val_bpb` accounting and reproducibility.

## Competition / Rules Risk

- Track A is low risk:
  - change the training corpus while keeping the tokenizer and evaluation path fixed
- Track B is high scrutiny:
  - change tokenizer and dataset together
  - requires explicit proof that `val_bpb` remains correct

Rules notes from this repo's docs:

- dataset/tokenizer changes are allowed
- if tokenizer or dataset logic changes, the submission must prove `val_bpb` is correctly calculated
- evaluation must remain self-contained and reproducible

## Minimal Implementation Design In This Repo

Treat data as two separate tracks.

### Track A: Same Tokenizer, Stable Evaluation

Keep these fixed:

- `TOKENIZER_PATH=.../fineweb_1024_bpe.model`
- `VOCAB_SIZE=1024`
- fixed validation split
- current metric logic

Allowed changes:

- different training corpus
- different corpus mixture
- different shard order
- different quality filtering

Recommended first-pass candidate ladder:

1. Same corpus, different shard ordering or documented ordering policy
2. Same tokenizer, alternative high-quality corpus retokenized into the same shard format
3. Same tokenizer, curated mixture emphasizing web + code + math + high-quality text

Track A goal:

- isolate data quality from tokenizer effects
- keep comparisons smooth and easier to trust

### Track B: Tokenizer + Dataset Co-Design

Use the existing `data/` workflow to rebuild tokenizers and shards from published docs cache.

Required process:

- rebuild from an auditable docs cache with manifest
- export train and validation shards in the repo's binary format
- verify tokenizer byte accounting against the current `val_bpb` logic

Recommended first-pass ladder:

1. same documents, vocab sweep `{768, 1024, 1536}`
2. same documents, tokenizer family change only if the byte-accounting proof is clear
3. new corpus + new tokenizer only after 1 and 2 are understood

Implementation defaults:

- do not start with tokenizer changes
- run Track A before Track B
- keep early experiments to corpus changes under the same tokenizer

## Variant Ladder

1. Track A1: shard ordering / data-order policy
2. Track A2: same-tokenizer alternative corpus
3. Track A3: same-tokenizer curated mixture
4. Track B1: same-docs tokenizer vocab sweep
5. Track B2: new corpus + new tokenizer

## Metrics To Record

- `final_int8_zlib_roundtrip_exact val_bpb`
- raw pre-quant `val_bpb`
- compressed model bytes
- code bytes
- total artifact bytes
- train wallclock
- eval wallclock
- peak memory
- dataset manifest / docs hash notes
- tokenizer notes when Track B is used

## Acceptance Criteria

Track A `pass`:

- improves final `val_bpb` by at least `0.002`
- no architecture change is required to see the gain

Track A `promote`:

- improves final `val_bpb` by at least `0.003`
- comparisons are made under the same tokenizer and evaluation path

Track B `pass`:

- improves final `val_bpb` by at least `0.003`
- byte accounting proof is documented
- tokenizer export and validation logic are reproducible

Track B `promote`:

- improves final `val_bpb` by at least `0.005`
- proof of `val_bpb` correctness is included and auditable

## Kill Criteria

- any tokenizer-changing branch that cannot prove metric correctness
- data changes that look positive only because validation comparability was broken
- same-tokenizer corpus changes that fail to beat baseline after the first three candidate variants

## Estimated Engineering Cost

- Track A: Low to medium
- Track B: Medium to high
- Expected effort: `1-4 engineer days` depending on whether tokenizer rebuild is needed

## Merge Compatibility

- Data is a separate axis and should only be combined after one architecture/export winner exists
- Compatible later with `attention_residuals`, `recurrent_depth`, `looped_llms`, `bitnet_ternary_qat`, and `masa_weight_sharing`
- Do not combine with `distillation` until the data effect is isolated

## References

- Local challenge rules in `README.md`
- Local dataset workflow in `data/README.md`
- External benchmark motivation from the `nanochat` experiment log, where dataset change was one of the largest measured wins
