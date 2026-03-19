from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"
READY_STATE_FILENAME = "_download_state.json"


@dataclass(frozen=True)
class DownloadArtifact:
    relative_path: str
    destination: Path
    kind: str

def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def get(relative_path: str, *, force: bool = False) -> Path:
    destination = local_path_for_remote(relative_path)
    if force and (destination.exists() or destination.is_symlink()):
        destination.unlink()
    if destination.exists():
        return destination
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    return destination


def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")


def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def dataset_local_dir(dataset_dir: str) -> Path:
    return DATASETS_DIR / dataset_dir


def ready_state_path(dataset_dir: str) -> Path:
    return dataset_local_dir(dataset_dir) / READY_STATE_FILENAME


def build_download_plan(
    *,
    manifest: dict,
    dataset_dir: str,
    train_shards: int,
    skip_manifest_download: bool,
    with_docs: bool,
) -> tuple[dict, dict, list[DownloadArtifact]]:
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in {REMOTE_ROOT_PREFIX}/manifest.json")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{dataset_dir} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json")

    artifacts: list[DownloadArtifact] = []
    if not skip_manifest_download:
        artifacts.append(
            DownloadArtifact(
                relative_path=f"{REMOTE_ROOT_PREFIX}/manifest.json",
                destination=manifest_path(),
                kind="manifest",
            )
        )
    if with_docs:
        artifacts.extend(
            [
                DownloadArtifact(
                    relative_path=f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl",
                    destination=local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl"),
                    kind="docs",
                ),
                DownloadArtifact(
                    relative_path=f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json",
                    destination=local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json"),
                    kind="docs",
                ),
            ]
        )

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    for i in range(val_shards):
        relative_path = f"{dataset_prefix}/fineweb_val_{i:06d}.bin"
        artifacts.append(
            DownloadArtifact(
                relative_path=relative_path,
                destination=local_path_for_remote(relative_path),
                kind="val",
            )
        )
    for i in range(train_shards):
        relative_path = f"{dataset_prefix}/fineweb_train_{i:06d}.bin"
        artifacts.append(
            DownloadArtifact(
                relative_path=relative_path,
                destination=local_path_for_remote(relative_path),
                kind="train",
            )
        )

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        relative_path = f"{REMOTE_ROOT_PREFIX}/{artifact_path}"
        artifacts.append(
            DownloadArtifact(
                relative_path=relative_path,
                destination=local_path_for_remote(relative_path),
                kind="tokenizer",
            )
        )
    return dataset_entry, tokenizer_entry, artifacts


def summarize_artifacts(artifacts: list[DownloadArtifact]) -> dict:
    by_kind: dict[str, dict[str, int]] = {}
    missing_paths: list[str] = []
    existing_total = 0
    for artifact in artifacts:
        stats = by_kind.setdefault(artifact.kind, {"expected": 0, "ready": 0})
        stats["expected"] += 1
        if artifact.destination.is_file():
            stats["ready"] += 1
            existing_total += 1
        else:
            missing_paths.append(artifact.relative_path)
    return {
        "expected_total": len(artifacts),
        "ready_total": existing_total,
        "missing_total": len(missing_paths),
        "missing_paths": missing_paths,
        "by_kind": by_kind,
    }


def build_ready_state(
    *,
    variant: str,
    dataset_dir: str,
    train_shards: int,
    dataset_entry: dict,
    tokenizer_entry: dict,
    with_docs: bool,
    summary: dict,
) -> dict:
    return {
        "ready": summary["missing_total"] == 0,
        "repo_id": REPO_ID,
        "remote_root_prefix": REMOTE_ROOT_PREFIX,
        "variant": variant,
        "dataset_dir": dataset_dir,
        "requested_train_shards": train_shards,
        "expected_train_shards": summary["by_kind"].get("train", {}).get("expected", 0),
        "ready_train_shards": summary["by_kind"].get("train", {}).get("ready", 0),
        "expected_val_shards": summary["by_kind"].get("val", {}).get("expected", 0),
        "ready_val_shards": summary["by_kind"].get("val", {}).get("ready", 0),
        "tokenizer_name": tokenizer_entry.get("name"),
        "tokenizer_artifacts": artifact_paths_for_tokenizer(tokenizer_entry),
        "with_docs": with_docs,
        "docs_ready": summary["by_kind"].get("docs", {}).get("ready", 0),
        "expected_total": summary["expected_total"],
        "ready_total": summary["ready_total"],
        "missing_total": summary["missing_total"],
        "missing_paths": summary["missing_paths"],
        "dataset_stats": dataset_entry.get("stats") or {},
    }


def write_ready_state(dataset_dir: str, state: dict) -> Path:
    path = ready_state_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def log(message: str) -> None:
    print(message, flush=True)


def print_plan(*, variant: str, dataset_dir: str, train_shards: int, tokenizer_entry: dict, summary: dict) -> None:
    log(
        f"data_cache:variant={variant} repo_id={REPO_ID} remote_root={REMOTE_ROOT_PREFIX} "
        f"dataset_dir={dataset_dir} requested_train_shards={train_shards} tokenizer={tokenizer_entry.get('name')}"
    )
    log(
        "data_cache:expected "
        f"manifest={summary['by_kind'].get('manifest', {}).get('expected', 0)} "
        f"val={summary['by_kind'].get('val', {}).get('expected', 0)} "
        f"train={summary['by_kind'].get('train', {}).get('expected', 0)} "
        f"tokenizer={summary['by_kind'].get('tokenizer', {}).get('expected', 0)} "
        f"docs={summary['by_kind'].get('docs', {}).get('expected', 0)} "
        f"total={summary['expected_total']}"
    )
    log(
        "data_cache:local_before "
        f"ready_total={summary['ready_total']} missing_total={summary['missing_total']} "
        f"ready_train={summary['by_kind'].get('train', {}).get('ready', 0)} "
        f"ready_val={summary['by_kind'].get('val', {}).get('ready', 0)} "
        f"ready_tokenizer={summary['by_kind'].get('tokenizer', {}).get('ready', 0)}"
    )


def download_artifacts(artifacts: list[DownloadArtifact], *, force: bool, retries: int) -> int:
    downloaded = 0
    total = len(artifacts)
    for index, artifact in enumerate(artifacts, start=1):
        if artifact.destination.is_file() and not force:
            log(f"data_cache:[{index}/{total}] ready kind={artifact.kind} path={artifact.relative_path}")
            continue

        for attempt in range(1, retries + 1):
            try:
                get(artifact.relative_path, force=force)
                downloaded += 1
                log(f"data_cache:[{index}/{total}] downloaded kind={artifact.kind} path={artifact.relative_path}")
                break
            except Exception as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"failed to download {artifact.relative_path} after {retries} attempts"
                    ) from exc
                delay_seconds = min(2 ** (attempt - 1), 30)
                print(
                    f"data_cache:[{index}/{total}] retry {attempt}/{retries} "
                    f"kind={artifact.kind} path={artifact.relative_path} error={exc!r} "
                    f"sleep={delay_seconds}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_seconds)
    return downloaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download challenge FineWeb shards from Hugging Face")
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer family to download, for example sp1024, sp4096, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only verify that the requested local artifacts already exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download requested artifacts even if the local destination already exists.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum download attempts per missing artifact. Defaults to 4.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    if args.retries < 1:
        raise ValueError("retries must be at least 1")

    manifest = load_manifest(skip_manifest_download=args.skip_manifest)
    dataset_entry, tokenizer_entry, artifacts = build_download_plan(
        manifest=manifest,
        dataset_dir=dataset_dir,
        train_shards=train_shards,
        skip_manifest_download=args.skip_manifest,
        with_docs=args.with_docs,
    )
    summary_before = summarize_artifacts(artifacts)
    print_plan(
        variant=args.variant,
        dataset_dir=dataset_dir,
        train_shards=train_shards,
        tokenizer_entry=tokenizer_entry,
        summary=summary_before,
    )

    downloaded = 0
    if args.verify_only:
        if summary_before["missing_total"] != 0:
            missing_preview = ", ".join(summary_before["missing_paths"][:3])
            raise FileNotFoundError(
                "verify_only failed because requested artifacts are missing: "
                f"{missing_preview}"
                + (" ..." if summary_before["missing_total"] > 3 else "")
            )
    else:
        downloaded = download_artifacts(artifacts, force=args.force, retries=args.retries)

    summary_after = summarize_artifacts(artifacts)
    state = build_ready_state(
        variant=args.variant,
        dataset_dir=dataset_dir,
        train_shards=train_shards,
        dataset_entry=dataset_entry,
        tokenizer_entry=tokenizer_entry,
        with_docs=args.with_docs,
        summary=summary_after,
    )
    state_path = write_ready_state(dataset_dir, state)
    log(
        "data_cache:local_after "
        f"downloaded={downloaded} ready_total={summary_after['ready_total']} "
        f"missing_total={summary_after['missing_total']} ready_file={state_path}"
    )
    if summary_after["missing_total"] != 0:
        raise RuntimeError(
            "download completed without a full ready dataset; missing artifacts remain: "
            + ", ".join(summary_after["missing_paths"][:3])
            + (" ..." if summary_after["missing_total"] > 3 else "")
        )


if __name__ == "__main__":
    main()
