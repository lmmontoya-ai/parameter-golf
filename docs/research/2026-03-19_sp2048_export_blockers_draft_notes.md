# 2026-03-19: `sp2048` Export Blockers Draft Notes

## Purpose

Record the first serious attempt to activate the `sp2048` tokenizer branch on the mainline dense lane, including the operational blockers and the export methodology change needed to make the branch iterable.

This is not yet a model-result note. It is an environment and dataset-prep note for the tokenizer lane.

## Lane

- intended training lane: `4xH100 -> 1200s`
- tokenizer/export prep lane: separate large-disk Vast instance
- repo policy: `experiments/` for plans, `docs/research/` for observed findings, `records/` for frozen artifacts

## Initial Blocker

The first `sp2048` export attempt was launched on the same Nebraska `4xH100` host that had worked for the dense mainline:

- provider: Vast.ai
- instance: `33166364`
- hardware: `4xH100 SXM`
- disk: `50 GB`

That host was wrong for tokenizer export.

Observed failure:

- the public `docs_selected.jsonl` cache for the selected challenge corpus is about `48.17 GB`
- the host had only about `32 GB` free at runtime
- `data/download_hf_docs_and_tokenize.py` warned that the target Hugging Face cache location did not have enough free disk

This is not a model result. It is a storage-sizing requirement for tokenizer work.

Local raw log snapshot:

- [vast_export_sp2048_v1.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_export_sp2048_v1.txt)

## Corpus Identity

On the large-disk export host, the downloaded sidecar for the fixed selected-docs corpus reports:

- `num_docs: 15368808`
- `num_val_docs: 50000`
- `selection_seed: 1337`
- `docs_sha256: 84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7`
- `docs_bytes: 48166275520`
- `snapshot_kind: partial_docs_cache_from_50B_export`
- `source_export_root: /root/exports/fineweb_50Bsub100B_50keval_v0`

Local sidecar snapshot:

- [sp2048_docs_selected.source_manifest.json](/Users/lumontoya/research/openai/parameter-golf/logs/sp2048_docs_selected.source_manifest.json)

This confirms the scale of the branch:

- the selected-docs text file is about `48.17 GB`
- tokenizer work must be provisioned as a separate data-prep lane, not casually attached to a small-disk training pod

## Export Host Iteration

Two cheaper large-disk export hosts were tried first and rejected:

1. California `2xRTX 5090`, instance `33166810`
2. United Kingdom `2xRTX 4090`, instance `33166930`

Both stayed in `loading` and their Vast logs reported:

- `Error response from daemon: No such container: C.<instance_id>`

Those were infrastructure failures, not experiment outcomes.

The first healthy export host was:

- provider: Vast.ai
- instance: `33166985`
- hardware: `1xH100 NVL`
- CPU: `80` cores
- RAM: about `645 GB`
- disk: `200 GB`
- SSH: `ssh -p 4521 root@20.62.104.255`

This host is being used only for tokenizer and dataset prep, not for the actual score-lane training comparison.

## Methodology Change

The first full-corpus `sp2048` tokenizer-training attempt on the export host was allowed to proceed long enough to establish the operational cost:

- same fixed selected-docs corpus
- same deterministic tokenizer config
- no architecture changes
- same output root convention under `/root/parameter-golf/tokenizer_builds/sp2048_20260319`

Observed behavior:

- tokenizer training advanced through at least `11,000,000` loaded lines before entering the long `Normalizing sentences...` phase
- this was too slow for first-pass iteration on the tokenizer branch

Decision:

- keep the fixed docs corpus and full dataset export target
- cap tokenizer training to `5,000,000` docs for the first-pass `sp2048` branch using:
  - `--tokenizer-train-docs 5000000`

Reasoning:

- this keeps the tokenizer branch deterministic and tied to the same fixed selected-docs corpus
- it reduces tokenizer-training latency enough to make the branch operationally searchable
- it does not change the later plan to export the full dataset for training/eval once the tokenizer model is built

## Current In-Progress Export

Current live export command:

```bash
MATCHED_FINEWEB_TOKENIZER_THREADS=32 \
MATCHED_FINEWEB_SP_BATCH_SIZE=4096 \
./.venv/bin/python data/download_hf_docs_and_tokenize.py \
  --output-root /root/parameter-golf/tokenizer_builds/sp2048_20260319 \
  --tokenizer-config /root/parameter-golf/tmp_sp2048_tokenizer_specs.json \
  --tokenizer-train-docs 5000000
```

Current live log:

- remote: `/root/parameter-golf/logs/vast_export_sp2048_v2_5m.txt`
- local snapshot: [vast_export_sp2048_v2_5m.txt](/Users/lumontoya/research/openai/parameter-golf/logs/vast_export_sp2048_v2_5m.txt)

Local note status:

- tokenizer export is still running
- the SentencePiece artifacts now exist:
  - `fineweb_2048_bpe.model`
  - `fineweb_2048_bpe.vocab`
- the exporter has moved on to:
  - `Exporting dataset: fineweb10B_sp2048`
- no `sp2048` model score is being claimed yet

## Interpretation

- The tokenizer branch is real, but it is materially more operationally expensive than dense schedule/eval ablations.
- `sp2048` work should be treated as a two-stage pipeline:
  1. large-disk export host for tokenizer and dataset prep
  2. separate `4xH100 -> 1200s` training lane for the comparable model run
- The first training pod failure was a disk-sizing mistake, not evidence against `sp2048`.
- For first-pass iteration, capped tokenizer training is the practical choice.

## Next Step

- let the capped `5M` tokenizer export finish
- record tokenizer artifact names, bytes, dataset path, and `tokens_per_byte`
- launch a fresh `4xH100` training pod
- run the first `sp2048` dense smoke, then the full `1200s` comparison if the smoke is sane
