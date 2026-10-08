from __future__ import annotations

import gzip
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import duckdb
import polars as pl

from .resources import MemoryPlan, make_memory_plan, memory_pressure
from .schema import INTERACTION_SCHEMA, ITEM_SCHEMA, interaction_record, item_record

LOGGER = logging.getLogger(__name__)


@dataclass
class ConversionStats:
    input_rows: int = 0
    output_rows: int = 0
    malformed_rows: int = 0
    missing_id_rows: int = 0
    peak_batch_rows: int = 0


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _records(path: Path) -> Iterator[tuple[int, dict | None]]:
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                yield line_number, value if isinstance(value, dict) else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield line_number, None


def convert_jsonl(
    source: Path,
    destination: Path,
    transform: Callable[[dict], dict],
    schema: dict[str, pl.DataType],
    plan: MemoryPlan,
    compression: str = "zstd",
    overwrite: bool = False,
    keep_parts: bool = False,
) -> ConversionStats:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats, batch = ConversionStats(), []
    batch_limit = plan.batch_rows
    parts_parent = destination.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.stem}-parts-", dir=parts_parent))
    try:
        def flush() -> None:
            nonlocal batch_limit
            if not batch:
                return
            part = temp_dir / f"part-{stats.output_rows:012d}.parquet"
            pl.DataFrame(batch, schema=schema, strict=False).write_parquet(part, compression=compression, statistics=True)
            stats.output_rows += len(batch)
            stats.peak_batch_rows = max(stats.peak_batch_rows, len(batch))
            batch.clear()
            if memory_pressure(plan):
                batch_limit = max(1_000, batch_limit // 2)

        for _, raw in _records(source):
            stats.input_rows += 1
            if raw is None:
                stats.malformed_rows += 1
                continue
            record = transform(raw)
            if not record.get("item_id"):
                stats.missing_id_rows += 1
                continue
            batch.append(record)
            if len(batch) >= batch_limit:
                flush()
        flush()
        parts = sorted(temp_dir.glob("*.parquet"))
        if not parts:
            raise ValueError(f"No valid records found in {source}")
        pl.scan_parquet(temp_dir / "*.parquet").sink_parquet(
            destination, compression=compression, maintain_order=True, mkdir=True
        )
        return stats
    finally:
        if not keep_parts:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_report(interactions: Path, items: Path, report: Path, plan: MemoryPlan, stats: dict[str, ConversionStats]) -> None:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{max(64, int(plan.budget_bytes / 1024**2))}MB'")
    con.execute("SET threads=2")
    i, m = _sql_path(interactions), _sql_path(items)
    summary = con.execute(f"""
        WITH ix AS (SELECT * FROM read_parquet('{i}')),
             it AS (SELECT * FROM read_parquet('{m}'))
        SELECT
          (SELECT count(*) FROM ix), (SELECT count(DISTINCT user_id) FROM ix),
          (SELECT count(DISTINCT item_id) FROM ix), (SELECT count(*) FROM it),
          (SELECT count(DISTINCT item_id) FROM it),
          (SELECT count(*) FROM ix LEFT JOIN it USING(item_id) WHERE it.item_id IS NULL),
          (SELECT count(*) FROM ix WHERE user_id IS NULL),
          (SELECT count(*) FROM it WHERE title IS NULL),
          (SELECT count(*) FROM it WHERE image_url IS NULL)
    """).fetchone()
    ratings = con.execute(f"SELECT rating, count(*) n FROM read_parquet('{i}') GROUP BY rating ORDER BY rating").fetchall()
    duplicates = con.execute(f"SELECT count(*) - count(DISTINCT (user_id, item_id, timestamp)) FROM read_parquet('{i}')").fetchone()[0]
    report.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase 1 data report

## Resource envelope

- Installed RAM: {plan.total_bytes / 1024**3:.2f} GiB
- Available RAM when the run started: {plan.available_bytes / 1024**3:.2f} GiB
- Pipeline memory ceiling: {plan.budget_gb:.2f} GiB
- Initial JSON batch: {plan.batch_rows:,} rows

The ceiling is calculated from live available memory and an OS reserve; installed RAM is not treated as usable RAM.

## Dataset summary

| Measure | Value |
|---|---:|
| Interaction rows | {summary[0]:,} |
| Distinct users | {summary[1]:,} |
| Distinct interaction items | {summary[2]:,} |
| Metadata rows | {summary[3]:,} |
| Distinct metadata items | {summary[4]:,} |
| Interactions without matching metadata | {summary[5]:,} |
| Interactions missing user ID | {summary[6]:,} |
| Items missing title | {summary[7]:,} |
| Items missing image URL | {summary[8]:,} |
| Duplicate interaction keys | {duplicates:,} |

## Rating distribution

| Rating | Rows |
|---:|---:|
{chr(10).join(f'| {rating} | {count:,} |' for rating, count in ratings)}

## Ingestion quality

| Input | Read | Written | Malformed | Missing item ID |
|---|---:|---:|---:|---:|
| Reviews | {stats['interactions'].input_rows:,} | {stats['interactions'].output_rows:,} | {stats['interactions'].malformed_rows:,} | {stats['interactions'].missing_id_rows:,} |
| Metadata | {stats['items'].input_rows:,} | {stats['items'].output_rows:,} | {stats['items'].malformed_rows:,} | {stats['items'].missing_id_rows:,} |
"""
    report.write_text(content, encoding="utf-8")
    con.close()


def run(config: dict, *, overwrite: bool | None = None) -> tuple[MemoryPlan, dict[str, ConversionStats]]:
    paths, resources, settings = config["paths"], config.get("resources", {}), config.get("pipeline", {})
    plan = make_memory_plan(resources)
    overwrite = settings.get("overwrite", False) if overwrite is None else overwrite
    common = dict(plan=plan, compression=resources.get("parquet_compression", "zstd"), overwrite=overwrite,
                  keep_parts=settings.get("keep_temporary_parts", False))
    stats = {
        "interactions": convert_jsonl(Path(paths["reviews"]), Path(paths["interactions_output"]), interaction_record,
                                      INTERACTION_SCHEMA, **common),
        "items": convert_jsonl(Path(paths["metadata"]), Path(paths["items_output"]), item_record, ITEM_SCHEMA, **common),
    }
    build_report(Path(paths["interactions_output"]), Path(paths["items_output"]), Path(paths["report"]), plan, stats)
    return plan, stats
