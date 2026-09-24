from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import polars as pl

from .coveo import REQUIRED_FILES, explode_search_clicks, sha256


DAY_MS = 86_400_000


@dataclass(frozen=True)
class DenseSliceConfig:
    dataset_version: str = "coveo_dense_v1"
    window_days_min: int = 21
    window_days_max: int = 28
    target_events: int = 1_200_000
    target_events_min: int = 1_000_000
    target_events_max: int = 1_500_000
    min_session_events: int = 3
    max_session_events: int = 50
    min_item_events: int = 5
    require_high_intent: bool = False
    include_search_clicks: bool = True
    train_time_ratio: float = 0.80
    valid_time_ratio: float = 0.10
    max_kcore_iterations: int = 50

    def validate(self) -> None:
        if self.window_days_min < 1 or self.window_days_max < self.window_days_min:
            raise ValueError("Invalid temporal window range")
        if self.min_session_events < 2 or self.max_session_events < self.min_session_events:
            raise ValueError("Invalid session length bounds")
        if self.min_item_events < 1:
            raise ValueError("min_item_events must be positive")
        if self.train_time_ratio + self.valid_time_ratio >= 1:
            raise ValueError("train_time_ratio + valid_time_ratio must be < 1")


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _choose_window(daily_rows: list[tuple], config: DenseSliceConfig) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_node: dict[tuple[str, int], dict[date, tuple[int, int]]] = defaultdict(dict)
    all_dates: set[date] = set()
    for day, node, depth, events, high_intent_events in daily_rows:
        by_node[(node, int(depth))][day] = (int(events), int(high_intent_events))
        all_dates.add(day)
    if not all_dates:
        raise ValueError("No eligible events found after session sanitization")
    first, last = min(all_dates), max(all_dates)
    diagnostics: list[dict[str, object]] = []
    for (node, depth), values in by_node.items():
        for days in range(config.window_days_min, config.window_days_max + 1):
            start = first
            while start + timedelta(days=days - 1) <= last:
                events = 0; intent = 0
                for offset in range(days):
                    current_events, current_intent = values.get(start + timedelta(days=offset), (0, 0))
                    events += current_events; intent += current_intent
                if events:
                    # Target size dominates. Intent and a coherent depth-2 node are
                    # small tie-breakers, never a reason to mix unrelated roots.
                    size_error = abs(math.log(max(events, 1) / config.target_events))
                    intent_rate = intent / events
                    outside = 0.0 if config.target_events_min <= events <= config.target_events_max else 0.35
                    score = size_error + outside - 0.08 * intent_rate - 0.02 * (depth - 1)
                    diagnostics.append({"category_node": node, "category_depth": depth,
                        "window_start": start, "window_end_exclusive": start + timedelta(days=days),
                        "window_days": days, "events_before_kcore": events,
                        "high_intent_event_rate": intent_rate, "selection_score": score})
                start += timedelta(days=1)
    if not diagnostics:
        raise ValueError("Could not form any contiguous temporal-window candidate")
    diagnostics.sort(key=lambda row: (row["selection_score"], -row["events_before_kcore"]))
    return diagnostics[0], diagnostics


def _iterate_kcore(con: duckdb.DuckDBPyConnection, table: str, config: DenseSliceConfig) -> list[dict[str, int]]:
    iterations: list[dict[str, int]] = []
    for iteration in range(1, config.max_kcore_iterations + 1):
        before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        con.execute(f"""
            DELETE FROM {table} WHERE item_id IN (
                SELECT item_id FROM {table} GROUP BY item_id HAVING count(*) < ?
            )
        """, [config.min_item_events])
        after_items = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        con.execute(f"""
            DELETE FROM {table} WHERE session_id IN (
                SELECT session_id FROM {table} GROUP BY session_id
                HAVING count(*) < ? OR count(*) > ?
            )
        """, [config.min_session_events, config.max_session_events])
        after_sessions = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        iterations.append({"iteration": iteration, "events_before": before,
                           "removed_by_item": before - after_items,
                           "removed_by_session": after_items - after_sessions,
                           "events_after": after_sessions})
        if after_sessions == before:
            return iterations
        if after_sessions == 0:
            raise ValueError("k-core removed every event; relax constraints or choose another window")
    raise RuntimeError("k-core did not converge within max_kcore_iterations")


def _checksum_rows(paths: list[Path], root: Path) -> list[dict[str, object]]:
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(paths)]


def build_dense_coveo_slice(
    source_files: dict[str, Path], output_dir: Path, config: DenseSliceConfig
) -> dict[str, object]:
    """Build a category-coherent dense temporal subgraph without random sampling."""
    config.validate()
    missing = sorted(set(REQUIRED_FILES) - set(source_files))
    if missing:
        raise FileNotFoundError(f"Missing Coveo source files: {missing}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Dense dataset version already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    work = output_dir / "_work"; work.mkdir()
    try:
        search_glob = None; search_rows = 0
        if config.include_search_clicks:
            search_rows = explode_search_clicks(source_files["search_train.csv"], work / "search_clicks.parquet")
            search_glob = _sql_path(work / "search_clicks" / "*.parquet")
        con = duckdb.connect(str(work / "dense.duckdb"))
        catalog_path = _sql_path(source_files["sku_to_content.csv"])
        browse_path = _sql_path(source_files["browsing_train.csv"])
        con.execute(f"""
            CREATE TABLE catalog AS
            SELECT product_sku_hash::VARCHAR item_id,
                   category_hash::VARCHAR category_path,
                   split_part(category_hash, '/', 1)::VARCHAR category_root,
                   CASE WHEN strpos(category_hash, '/') > 0
                        THEN split_part(category_hash, '/', 1) || '/' || split_part(category_hash, '/', 2)
                        ELSE category_hash END::VARCHAR category_focus,
                   try_cast(price_bucket AS DOUBLE) price_bucket,
                   description_vector IS NOT NULL AND description_vector <> '' has_text,
                   image_vector IS NOT NULL AND image_vector <> '' has_image
            FROM read_csv_auto('{catalog_path}', header=true, all_varchar=true)
            WHERE product_sku_hash IS NOT NULL AND category_hash IS NOT NULL;

            CREATE TABLE base_events AS
            SELECT session_id_hash::VARCHAR session_id, product_sku_hash::VARCHAR item_id,
                   product_action::VARCHAR AS event_action, server_timestamp_epoch_ms::BIGINT AS event_ts,
                   'browse'::VARCHAR AS event_source, 0::BIGINT AS source_rank
            FROM read_csv_auto('{browse_path}', header=true, all_varchar=true)
            WHERE event_type='event_product' AND product_sku_hash IS NOT NULL
              AND product_action IN ('detail','add','purchase');
        """)
        if search_glob:
            con.execute(f"""INSERT INTO base_events
                SELECT session_id,item_id,action AS event_action,timestamp AS event_ts,source AS event_source,source_rank
                FROM read_parquet('{search_glob}')""")
        con.execute("""
            CREATE TABLE dedup_events AS
            SELECT * EXCLUDE(rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY session_id,item_id,event_action,event_ts,event_source ORDER BY source_rank
                ) rn FROM base_events WHERE session_id IS NOT NULL AND item_id IS NOT NULL
            ) WHERE rn=1;

            CREATE TABLE sanitized_sessions AS
            SELECT session_id FROM dedup_events GROUP BY session_id
            HAVING count(*) BETWEEN ? AND ?
               AND (? = false OR count(*) FILTER (WHERE event_action IN ('add','purchase')) > 0);

            CREATE TABLE eligible_events AS
            SELECT e.*, c.category_root, c.category_focus
            FROM dedup_events e JOIN sanitized_sessions s USING(session_id)
            JOIN catalog c USING(item_id);
        """, [config.min_session_events, config.max_session_events, config.require_high_intent])
        daily_rows = con.execute("""
            SELECT day, category_node, category_depth, count(*) events,
                   count(*) FILTER (WHERE event_action IN ('add','purchase')) high_intent_events
            FROM (
                SELECT CAST(to_timestamp(event_ts / 1000.0) AS DATE) day, category_root category_node,
                       1 category_depth, event_action FROM eligible_events
                UNION ALL
                SELECT CAST(to_timestamp(event_ts / 1000.0) AS DATE) day, category_focus category_node,
                       2 category_depth, event_action FROM eligible_events
                WHERE category_focus <> category_root
            ) GROUP BY day,category_node,category_depth
        """).fetchall()
        chosen, diagnostics = _choose_window(daily_rows, config)
        pl.DataFrame(diagnostics).write_parquet(output_dir / "window_diagnostics.parquet")
        start_ms = int(datetime.combine(chosen["window_start"], datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.combine(chosen["window_end_exclusive"], datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        category_column = "category_root" if chosen["category_depth"] == 1 else "category_focus"
        con.execute(f"""
            CREATE TABLE core_events AS
            SELECT session_id,item_id,event_action,event_ts,event_source,source_rank
            FROM eligible_events
            WHERE event_ts >= ? AND event_ts < ? AND {category_column} = ?;
            DELETE FROM core_events WHERE session_id IN (
                SELECT session_id FROM core_events GROUP BY session_id
                HAVING count(*) < ? OR count(*) > ?
                   OR (? = true AND count(*) FILTER (WHERE event_action IN ('add','purchase')) = 0)
            );
        """, [start_ms, end_ms, chosen["category_node"], config.min_session_events,
              config.max_session_events, config.require_high_intent])
        kcore_iterations = _iterate_kcore(con, "core_events", config)

        train_cutoff = start_ms + int((end_ms - start_ms) * config.train_time_ratio)
        valid_cutoff = train_cutoff + int((end_ms - start_ms) * config.valid_time_ratio)
        con.execute("""
            CREATE TABLE session_ranges AS
            SELECT session_id,min(event_ts) session_start,max(event_ts) session_end
            FROM core_events GROUP BY session_id;
            DELETE FROM core_events WHERE session_id IN (
                SELECT session_id FROM session_ranges
                WHERE (session_start < ? AND session_end >= ?)
                   OR (session_start < ? AND session_end >= ?)
            );
        """, [train_cutoff, train_cutoff, valid_cutoff, valid_cutoff])
        boundary_kcore_iterations = _iterate_kcore(con, "core_events", config)
        con.execute("""
            CREATE TABLE final_session_ranges AS
            SELECT session_id,min(event_ts) session_start,max(event_ts) session_end
            FROM core_events GROUP BY session_id;
        """)
        for split, lower, upper in (
            ("train", start_ms, train_cutoff), ("valid", train_cutoff, valid_cutoff),
            ("test", valid_cutoff, end_ms),
        ):
            destination = _sql_path(output_dir / f"{split}.parquet")
            con.execute(f"""
                COPY (
                    SELECT e.session_id user_id,e.item_id,e.event_action AS action,
                           CASE e.event_action WHEN 'purchase' THEN 4.0 WHEN 'add' THEN 2.0
                                WHEN 'search_click' THEN 1.2 ELSE 1.0 END action_weight,
                           e.event_ts AS timestamp,e.event_source AS source,e.source_rank,
                           row_number() OVER (PARTITION BY e.session_id ORDER BY e.event_ts,e.source_rank,e.item_id) event_order,
                           1::UTINYINT is_positive
                    FROM core_events e JOIN final_session_ranges s USING(session_id)
                    WHERE s.session_start >= {lower} AND s.session_end < {upper}
                    ORDER BY e.session_id,e.event_ts,e.source_rank,e.item_id
                ) TO '{destination}' (FORMAT PARQUET,COMPRESSION ZSTD);
            """)
        items_path = _sql_path(output_dir / "items.parquet")
        con.execute(f"""
            COPY (
                SELECT c.item_id,c.category_path,c.price_bucket,c.has_text,c.has_image,
                       x.event_count,x.session_count
                FROM catalog c JOIN (
                    SELECT item_id,count(*) event_count,count(DISTINCT session_id) session_count
                    FROM core_events GROUP BY item_id
                ) x USING(item_id) ORDER BY x.event_count DESC,c.item_id
            ) TO '{items_path}' (FORMAT PARQUET,COMPRESSION ZSTD);
        """)
        counts = {split: con.execute(
            f"SELECT count(*),count(DISTINCT user_id),count(DISTINCT item_id),min(timestamp),max(timestamp) FROM read_parquet('{_sql_path(output_dir / f'{split}.parquet')}')"
        ).fetchone() for split in ("train", "valid", "test")}
        graph = con.execute("""
            SELECT count(*) events,count(DISTINCT session_id) sessions,count(DISTINCT item_id) items,
                   avg(session_len),avg(item_degree),
                   count(DISTINCT session_id) FILTER (WHERE session_id IN (
                       SELECT session_id FROM core_events WHERE event_action IN ('add','purchase')
                   )) high_intent_sessions
            FROM core_events,
                 (SELECT avg(n) session_len FROM (SELECT count(*) n FROM core_events GROUP BY session_id)),
                 (SELECT avg(n) item_degree FROM (SELECT count(*) n FROM core_events GROUP BY item_id))
        """).fetchone()
        con.close(); shutil.rmtree(work)
        processed_paths = list(output_dir.glob("*.parquet"))
        manifest = {
            "schema_version": 1, "dataset": "Coveo SIGIR eCommerce 2021",
            "dataset_version": config.dataset_version, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_strategy": "category-coherent dense contiguous temporal slice + iterative bipartite k-core",
            "random_or_hash_sampling": False, "config": asdict(config),
            "selected_window": {**chosen, "window_start": str(chosen["window_start"]),
                                "window_end_exclusive": str(chosen["window_end_exclusive"])},
            "absolute_split_cutoffs_ms": {"train_end": train_cutoff, "valid_end": valid_cutoff},
            "search_click_rows_parsed": search_rows,
            "kcore_iterations": kcore_iterations,
            "post_boundary_kcore_iterations": boundary_kcore_iterations,
            "graph": {"events": graph[0], "sessions": graph[1], "items": graph[2],
                      "mean_session_len": graph[3], "mean_item_degree": graph[4],
                      "high_intent_sessions": graph[5],
                      "target_band_met": config.target_events_min <= graph[0] <= config.target_events_max},
            "splits": {name: {"events": row[0], "sessions": row[1], "items": row[2],
                              "min_timestamp": row[3], "max_timestamp": row[4]} for name, row in counts.items()},
            "source_files": [{"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for name, path in sorted(source_files.items())],
            "processed_files": _checksum_rows(processed_paths, output_dir),
        }
        (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
