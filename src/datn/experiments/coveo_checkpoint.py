from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from datn.data.coveo import sha256


def _copy(repo: Path, destination: Path, relative: Path) -> None:
    source = repo / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        # Candidate tables can be regenerated from the frozen retrieval checkpoint
        # and may be tens of GB. Models, metrics and histories are the checkpoint.
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "candidates")
        shutil.copytree(source, target, ignore=ignore)
    else:
        shutil.copy2(source, target)


def checkpoint_coveo_run(repo: Path, destination: Path) -> Path:
    """Freeze code, notebooks, manifests, both models and all experiment metrics."""
    repo, destination = repo.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Checkpoint is immutable and already exists: {destination}")
    destination.mkdir(parents=True)
    required = (
        Path("configs/coveo.yaml"), Path("notebooks/coveo"), Path("docs/COVEO_NOTEBOOK_GUIDE.md"),
        Path("src/datn/data/coveo.py"), Path("src/datn/features/coveo.py"),
        Path("src/datn/recommenders/coveo"), Path("data/processed/coveo_v1/dataset_manifest.json"),
        Path("data/processed/coveo_embeddings_v1/embedding_manifest.json"),
        Path("data/artifacts/coveo_retrieval_v1"), Path("data/artifacts/coveo_reranker_v1"),
    )
    try:
        for relative in required:
            _copy(repo, destination, relative)
        def git(*args: str) -> str | None:
            try:
                return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        metadata = {
            "schema_version": 1,
            "experiment": "coveo_action_aware_two_tower_and_residual_reranker_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable": True,
            "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                    "status": (git("status", "--short") or "").splitlines()},
            "environment": {"python": sys.version, "platform": platform.platform()},
        }
        files = []
        for path in sorted(p for p in destination.rglob("*") if p.is_file()):
            files.append({"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
        metadata["files"] = files
        (destination / "checkpoint_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return destination
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
