from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl


REQUIRED_FILES = ("browsing_train.csv", "search_train.csv", "sku_to_content.csv")
SAMPLE_URLS = {
    "browsing_train.csv": "https://raw.githubusercontent.com/coveooss/SIGIR-ecom-data-challenge/main/start/browsing_train_sample.csv",
    "search_train.csv": "https://raw.githubusercontent.com/coveooss/SIGIR-ecom-data-challenge/main/start/search_train_sample.csv",
    "sku_to_content.csv": "https://raw.githubusercontent.com/coveooss/SIGIR-ecom-data-challenge/main/start/sku_to_content_sample.csv",
}


@dataclass(frozen=True)
class CoveoPrepareConfig:
    dataset_version: str = "coveo_v1"
    train_ratio: float = 0.80
    valid_ratio: float = 0.10
    min_session_events: int = 2
    max_sessions: int | None = None
    include_search_clicks: bool = True
    seed: int = 20260922


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_required(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not root.exists():
        return found
    for name in REQUIRED_FILES:
        matches = list(root.rglob(name))
        if matches:
            found[name] = matches[0]
    return found


def locate_coveo_files(*roots: Path) -> dict[str, Path]:
    """Return the first complete Coveo source without copying mounted data."""
    for root in roots:
        found = _find_required(root.resolve())
        if len(found) == len(REQUIRED_FILES):
            return found
    searched = ", ".join(str(path) for path in roots)
    raise FileNotFoundError(f"No complete Coveo source found under: {searched}")


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
        archive.extractall(destination)


def acquire_coveo(
    destination: Path,
    *,
    kaggle_root: Path = Path("/kaggle/input"),
    zip_path: Path | None = None,
    authorized_url: str | None = None,
    sample: bool = False,
    terms_accepted: bool | None = None,
) -> dict[str, object]:
    """Acquire Coveo without bypassing its registration or licence gate.

    Priority: an already mounted Kaggle dataset, an explicitly supplied local ZIP,
    an explicitly supplied authorized URL, then the small official GitHub sample.
    The full dataset requires the user to accept Coveo's terms first.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    accepted = terms_accepted if terms_accepted is not None else os.getenv("COVEO_TERMS_ACCEPTED") == "1"
    source = ""

    existing = _find_required(destination)
    mounted = _find_required(kaggle_root)
    if len(existing) == len(REQUIRED_FILES):
        source = f"existing:{destination}"
    elif len(mounted) == len(REQUIRED_FILES):
        for name, path in mounted.items():
            shutil.copy2(path, destination / name)
        source = f"kaggle:{kaggle_root}"
    elif zip_path is not None:
        if not accepted:
            raise PermissionError("Set COVEO_TERMS_ACCEPTED=1 after accepting the Coveo terms.")
        _safe_extract(zip_path.resolve(), destination)
        source = f"zip:{zip_path.resolve()}"
    elif authorized_url:
        if not accepted:
            raise PermissionError("Set COVEO_TERMS_ACCEPTED=1 after accepting the Coveo terms.")
        downloaded = destination / "coveo_download.zip"
        urllib.request.urlretrieve(authorized_url, downloaded)
        _safe_extract(downloaded, destination)
        downloaded.unlink()
        source = "authorized_url"
    elif sample:
        for name, url in SAMPLE_URLS.items():
            urllib.request.urlretrieve(url, destination / name)
        source = "official_github_sample"
    else:
        raise FileNotFoundError(
            "No Coveo source found. Mount a Kaggle dataset, provide COVEO_ZIP_PATH, "
            "or run the official sample profile. Full access requires Coveo registration."
        )

    located = _find_required(destination)
    if len(located) != len(REQUIRED_FILES):
        missing = sorted(set(REQUIRED_FILES) - set(located))
        raise FileNotFoundError(f"Missing required Coveo files after acquisition: {missing}")
    for name, source_path in located.items():
        target = destination / name
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)

    report = {
        "source": source,
        "sample": sample,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {"bytes": (destination / name).stat().st_size, "sha256": sha256(destination / name)}
            for name in REQUIRED_FILES
        },
    }
    (destination / "acquisition_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _explode_search_clicks(search_csv: Path, destination: Path, batch_size: int = 100_000) -> int:
    total = 0
    part_dir = destination.with_suffix("")
    part_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    part = 0
    with search_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for rank, sku in enumerate(_parse_list(row.get("clicked_skus_hash"))):
                rows.append({"session_id": row["session_id_hash"], "item_id": sku,
                             "action": "search_click", "timestamp": int(row["server_timestamp_epoch_ms"]),
                             "source": "search", "source_rank": rank})
            if len(rows) >= batch_size:
                pl.DataFrame(rows).write_parquet(part_dir / f"part-{part:05d}.parquet")
                total += len(rows); rows = []; part += 1
    if rows:
        pl.DataFrame(rows).write_parquet(part_dir / f"part-{part:05d}.parquet")
        total += len(rows)
    if total == 0:
        pl.DataFrame(schema={"session_id": pl.String, "item_id": pl.String, "action": pl.String,
                             "timestamp": pl.Int64, "source": pl.String, "source_rank": pl.Int64}).write_parquet(
            part_dir / "part-00000.parquet"
        )
    return total


explode_search_clicks = _explode_search_clicks


def _files_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    return [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(paths)
    ]


def prepare_coveo_dataset(raw_dir: Path, output_dir: Path, config: CoveoPrepareConfig) -> dict[str, object]:
    """Create immutable, session-chronological Parquet splits out-of-core."""
    raw_dir, output_dir = raw_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Dataset version already exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True)
    work = output_dir / "_work"
    work.mkdir()
    try:
        search_glob = None
        search_count = 0
        if config.include_search_clicks:
            search_count = _explode_search_clicks(raw_dir / "search_train.csv", work / "search_clicks.parquet")
            search_glob = (work / "search_clicks" / "*.parquet").as_posix()

        con = duckdb.connect(str(work / "etl.duckdb"))
        browse = (raw_dir / "browsing_train.csv").as_posix().replace("'", "''")
        con.execute(f"""
            CREATE TABLE events AS
            SELECT session_id_hash::VARCHAR AS session_id,
                   product_sku_hash::VARCHAR AS item_id,
                   product_action::VARCHAR AS action,
                   server_timestamp_epoch_ms::BIGINT AS timestamp,
                   'browse'::VARCHAR AS source,
                   0::BIGINT AS source_rank
            FROM read_csv_auto('{browse}', header=true, all_varchar=true)
            WHERE event_type='event_product'
              AND product_sku_hash IS NOT NULL
              AND product_action IN ('detail','add','purchase')
        """)
        if search_glob:
            con.execute(f"INSERT INTO events SELECT * FROM read_parquet('{search_glob}')")
        con.execute("""
            DELETE FROM events WHERE session_id IS NULL OR item_id IS NULL;
            CREATE TABLE dedup AS
            SELECT * EXCLUDE(rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY session_id,item_id,action,timestamp,source ORDER BY source_rank
                ) rn FROM events
            ) WHERE rn=1;
            CREATE TABLE eligible AS
            SELECT session_id, min(timestamp) session_start, max(timestamp) session_end, count(*) n_events
            FROM dedup GROUP BY session_id HAVING count(*) >= ?;
        """, [config.min_session_events])
        if config.max_sessions is not None:
            con.execute("""
                CREATE TABLE sampled AS SELECT * FROM eligible
                ORDER BY hash(session_id, ?) LIMIT ?
            """, [config.seed, config.max_sessions])
            eligible_table = "sampled"
        else:
            eligible_table = "eligible"
        con.execute(f"""
            CREATE TABLE session_splits AS
            SELECT *, CASE
                WHEN pr < {config.train_ratio} THEN 'train'
                WHEN pr < {config.train_ratio + config.valid_ratio} THEN 'valid'
                ELSE 'test' END AS split
            FROM (
                SELECT *, percent_rank() OVER (ORDER BY session_end, session_id) pr
                FROM {eligible_table}
            );
        """)
        for split in ("train", "valid", "test"):
            out = (output_dir / f"{split}.parquet").as_posix().replace("'", "''")
            con.execute(f"""
                COPY (
                    SELECT e.session_id AS user_id, e.item_id, e.action,
                           CASE e.action WHEN 'purchase' THEN 4.0 WHEN 'add' THEN 2.0
                                WHEN 'search_click' THEN 1.2 WHEN 'detail' THEN 1.0 ELSE 0.5 END AS action_weight,
                           e.timestamp, e.source, e.source_rank,
                           row_number() OVER (PARTITION BY e.session_id ORDER BY e.timestamp,e.source_rank,e.item_id) AS event_order,
                           1::UTINYINT AS is_positive
                    FROM dedup e JOIN session_splits s ON e.session_id=s.session_id
                    WHERE s.split='{split}' ORDER BY e.session_id,e.timestamp,e.source_rank
                ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """)
        catalog = (raw_dir / "sku_to_content.csv").as_posix().replace("'", "''")
        items_out = (output_dir / "items.parquet").as_posix().replace("'", "''")
        con.execute(f"""
            COPY (
                SELECT product_sku_hash::VARCHAR item_id,
                       category_hash::VARCHAR category_path,
                       try_cast(price_bucket AS DOUBLE) price_bucket,
                       description_vector IS NOT NULL AS has_text,
                       image_vector IS NOT NULL AS has_image
                FROM read_csv_auto('{catalog}', header=true, all_varchar=true)
                WHERE product_sku_hash IS NOT NULL
            ) TO '{items_out}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        counts = {
            split: con.execute(f"SELECT count(*), count(DISTINCT user_id), count(DISTINCT item_id) FROM read_parquet('{(output_dir / f'{split}.parquet').as_posix()}')").fetchone()
            for split in ("train", "valid", "test")
        }
        cutoffs = con.execute("SELECT split,min(session_end),max(session_end),count(*) FROM session_splits GROUP BY split ORDER BY min(session_end)").fetchall()
        con.close()
        shutil.rmtree(work)

        manifest = {
            "schema_version": 1,
            "dataset": "Coveo SIGIR eCommerce 2021",
            "dataset_version": config.dataset_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_protocol": "global chronological split by session_end; last event is target at evaluation",
            "config": asdict(config),
            "search_click_rows": search_count,
            "counts": {k: {"events": v[0], "sessions": v[1], "items": v[2]} for k, v in counts.items()},
            "split_time_ranges": [{"split": r[0], "min_ts": r[1], "max_ts": r[2], "sessions": r[3]} for r in cutoffs],
            "raw_files": _files_manifest((raw_dir / n for n in REQUIRED_FILES), raw_dir),
            "processed_files": _files_manifest(output_dir.glob("*.parquet"), output_dir),
        }
        (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
