from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "cached_challenge_fineweb.py"
SPEC = importlib.util.spec_from_file_location("cached_challenge_fineweb", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load spec for {MODULE_PATH}")
cached_challenge_fineweb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cached_challenge_fineweb
SPEC.loader.exec_module(cached_challenge_fineweb)


def fake_manifest() -> dict:
    return {
        "datasets": [
            {
                "name": "fineweb10B_sp1024",
                "tokenizer_name": "fineweb_1024_bpe",
                "stats": {
                    "files_train": 3,
                    "files_val": 1,
                },
            }
        ],
        "tokenizers": [
            {
                "name": "fineweb_1024_bpe",
                "model_path": "tokenizers/fineweb_1024_bpe.model",
            }
        ],
    }


class CachedChallengeFinewebTest(unittest.TestCase):
    def test_build_download_plan_matches_requested_shards(self):
        dataset_entry, tokenizer_entry, artifacts = cached_challenge_fineweb.build_download_plan(
            manifest=fake_manifest(),
            dataset_dir="fineweb10B_sp1024",
            train_shards=2,
            skip_manifest_download=False,
            with_docs=False,
        )

        self.assertEqual(dataset_entry["name"], "fineweb10B_sp1024")
        self.assertEqual(tokenizer_entry["name"], "fineweb_1024_bpe")
        self.assertEqual(sum(1 for item in artifacts if item.kind == "manifest"), 1)
        self.assertEqual(sum(1 for item in artifacts if item.kind == "val"), 1)
        self.assertEqual(sum(1 for item in artifacts if item.kind == "train"), 2)
        self.assertEqual(sum(1 for item in artifacts if item.kind == "tokenizer"), 1)

    def test_summarize_artifacts_counts_ready_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            tokenizer_dir = root / "tokenizers"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(cached_challenge_fineweb, "ROOT", root), \
                mock.patch.object(cached_challenge_fineweb, "DATASETS_DIR", dataset_dir), \
                mock.patch.object(cached_challenge_fineweb, "TOKENIZERS_DIR", tokenizer_dir):
                _dataset_entry, _tokenizer_entry, artifacts = cached_challenge_fineweb.build_download_plan(
                    manifest=fake_manifest(),
                    dataset_dir="fineweb10B_sp1024",
                    train_shards=2,
                    skip_manifest_download=False,
                    with_docs=False,
                )
                artifacts[0].destination.parent.mkdir(parents=True, exist_ok=True)
                artifacts[0].destination.write_text("manifest", encoding="utf-8")
                artifacts[1].destination.parent.mkdir(parents=True, exist_ok=True)
                artifacts[1].destination.write_bytes(b"val")
                summary = cached_challenge_fineweb.summarize_artifacts(artifacts)

            self.assertEqual(summary["expected_total"], 5)
            self.assertEqual(summary["ready_total"], 2)
            self.assertEqual(summary["missing_total"], 3)
            self.assertEqual(summary["by_kind"]["train"]["expected"], 2)
            self.assertEqual(summary["by_kind"]["val"]["ready"], 1)

    def test_write_ready_state_persists_machine_readable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(cached_challenge_fineweb, "DATASETS_DIR", dataset_dir):
                state = {
                    "ready": True,
                    "dataset_dir": "fineweb10B_sp1024",
                    "requested_train_shards": 2,
                }
                path = cached_challenge_fineweb.write_ready_state("fineweb10B_sp1024", state)

            self.assertEqual(path.name, cached_challenge_fineweb.READY_STATE_FILENAME)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(loaded["ready"])
            self.assertEqual(loaded["requested_train_shards"], 2)


if __name__ == "__main__":
    unittest.main()
