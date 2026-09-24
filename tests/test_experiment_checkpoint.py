from __future__ import annotations

import json
from pathlib import Path

import pytest

from datn.experiments.checkpoint import create_checkpoint, sha256


def test_create_checkpoint_is_immutable_and_hashes_every_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    required_files = (Path("notebooks/train.ipynb"), Path("results/metrics.json"))
    for relative in required_files:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")

    for relative in (
        Path("data/artifacts/user_tower_balanced_v1"),
        Path("data/artifacts/reranker_v2"),
        Path("src/datn"),
    ):
        path = repo / relative / "artifact.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.as_posix().encode())

    destination = tmp_path / "checkpoint"
    create_checkpoint(repo, destination, files=required_files)
    manifest = json.loads((destination / "checkpoint_manifest.json").read_text())

    listed = {entry["path"]: entry for entry in manifest["files"]}
    copied = destination / "notebooks/train.ipynb"
    assert listed["notebooks/train.ipynb"]["sha256"] == sha256(copied)
    assert listed["data/artifacts/reranker_v2/artifact.bin"]["size_bytes"] > 0

    with pytest.raises(FileExistsError):
        create_checkpoint(repo, destination, files=required_files)
