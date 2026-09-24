from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_FILES = (
    Path("README.md"),
    Path("configs/balanced_dataset.yaml"),
    Path("configs/user_tower.balanced.yaml"),
    Path("notebooks/user_tower_training.ipynb"),
    Path("notebooks/reranker_training.ipynb"),
    Path("docs/logs/2026-09-22_balanced_retrieval_reranker_results.md"),
    Path("data/processed/balanced_u5_i2_v1/dataset_manifest.json"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(repo: Path, destination: Path, source: Path) -> None:
    absolute = repo / source
    if not absolute.is_file():
        raise FileNotFoundError(f"Required checkpoint input is missing: {absolute}")
    target = destination / source
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(absolute, target)


def _copy_tree(repo: Path, destination: Path, source: Path) -> None:
    absolute = repo / source
    if not absolute.is_dir():
        raise FileNotFoundError(f"Required checkpoint directory is missing: {absolute}")
    shutil.copytree(
        absolute,
        destination / source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _environment() -> dict[str, object]:
    details: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        details["torch"] = torch.__version__
        details["cuda_available"] = torch.cuda.is_available()
        details["cuda_version"] = torch.version.cuda
        details["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        details["torch"] = None
    return details


def create_checkpoint(
    repo: Path,
    destination: Path,
    *,
    files: Iterable[Path] = DEFAULT_FILES,
) -> Path:
    """Create an immutable, checksum-addressed two-stage experiment bundle."""
    repo = repo.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Checkpoint destination already exists and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True)

    try:
        for source in files:
            _copy_file(repo, destination, source)
        for source in (
            Path("data/artifacts/user_tower_balanced_v1"),
            Path("data/artifacts/reranker_v2"),
            Path("src/datn"),
        ):
            _copy_tree(repo, destination, source)

        metadata = destination / "checkpoint_metadata"
        metadata.mkdir()
        git_state = {
            "commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "status_porcelain": (_git(repo, "status", "--short") or "").splitlines(),
        }
        (metadata / "git_state.json").write_text(
            json.dumps(git_state, indent=2), encoding="utf-8"
        )
        (metadata / "environment.json").write_text(
            json.dumps(_environment(), indent=2), encoding="utf-8"
        )

        manifest_files = []
        for path in sorted(p for p in destination.rglob("*") if p.is_file()):
            relative = path.relative_to(destination).as_posix()
            manifest_files.append(
                {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "balanced_u5_i2_retrieval_reranker_v2",
            "immutable": True,
            "files": manifest_files,
        }
        (destination / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint the complete two-stage experiment")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    output = create_checkpoint(args.repo, args.destination)
    print(output)


if __name__ == "__main__":
    main()
