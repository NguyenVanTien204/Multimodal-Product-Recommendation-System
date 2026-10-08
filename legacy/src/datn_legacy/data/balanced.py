from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BalancedDatasetResult:
    output_dir: Path
    manifest: dict[str, Any]


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_list(paths: list[Path]) -> str:
    return "[" + ", ".join(f"'{_sql_path(path)}'" for path in paths) + "]"


def _copy_parquet(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    con.execute(
        f"COPY ({query}) TO '{_sql_path(destination)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def _validate_settings(settings: dict[str, Any]) -> None:
    if int(settings["min_user_positive_degree"]) < 3:
        raise ValueError("min_user_positive_degree must be >= 3 for train/valid/test")
    if int(settings["min_item_positive_degree"]) < 1:
        raise ValueError("min_item_positive_degree must be >= 1")
    if float(settings["strong_negative_max_rating"]) >= float(settings["positive_rating"]):
        raise ValueError("strong_negative_max_rating must be lower than positive_rating")


def prepare_balanced_dataset(config: dict[str, Any]) -> BalancedDatasetResult:
    """Build a versioned positive-aware recommendation benchmark.

    The k-core is computed on positive user-item edges only. All retained events are
    then reattached, but the two most recent positive events define validation and
    test targets. Events before the validation target become train data; non-positive
    events between validation and test are preserved separately as test context.

    The destination is intentionally immutable: callers must choose a new dataset
    version instead of overwriting a previous experiment.
    """

    paths = config["paths"]
    settings = {
        "name": "balanced_u5_i2_v1",
        "seed": 20260813,
        "min_user_positive_degree": 5,
        "min_item_positive_degree": 2,
        "positive_rating": 4.0,
        "strong_negative_max_rating": 2.0,
        "neutral_rating": 3.0,
        "verified_only": True,
        **config.get("dataset", {}),
    }
    _validate_settings(settings)

    interaction_paths = [Path(path) for path in paths["interactions"]]
    items_path = Path(paths["items"])
    output_dir = Path(paths["output_dir"])
    for path in [*interaction_paths, items_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite versioned dataset {output_dir}; choose a new output_dir"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-building-", dir=output_dir.parent))
    spill_dir = temp_dir / ".duckdb_temp"
    spill_dir.mkdir()

    resources = config.get("resources", {})
    memory_limit = str(resources.get("memory_limit", "2GB"))
    threads = int(resources.get("threads", 2))
    positive_rating = float(settings["positive_rating"])
    strong_negative = float(settings["strong_negative_max_rating"])
    min_user = int(settings["min_user_positive_degree"])
    min_item = int(settings["min_item_positive_degree"])
    verified_filter = "AND coalesce(e.verified_purchase, false)" if settings["verified_only"] else ""

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        con.execute(f"SET temp_directory='{_sql_path(spill_dir)}'")

        sources = _parquet_list(interaction_paths)
        con.execute(
            f"""
            CREATE TABLE events AS
            SELECT * EXCLUDE (_dedup_rank),
                   CASE WHEN rating >= {positive_rating} THEN 1 ELSE 0 END AS positive_label
            FROM (
                SELECT e.*,
                       row_number() OVER (
                           PARTITION BY e.user_id, e.item_id
                           ORDER BY e.timestamp DESC, e.rating DESC
                       ) AS _dedup_rank
                FROM read_parquet({sources}, union_by_name=true) e
                INNER JOIN read_parquet('{_sql_path(items_path)}') i USING (item_id)
                WHERE e.user_id IS NOT NULL AND e.item_id IS NOT NULL
                {verified_filter}
            )
            WHERE _dedup_rank = 1
            """
        )
        con.execute(
            f"""
            CREATE TABLE positive_core AS
            SELECT user_id, item_id, timestamp
            FROM events
            WHERE positive_label = 1
            """
        )

        iterations = 0
        while True:
            before = con.execute("SELECT count(*) FROM positive_core").fetchone()[0]
            con.execute(
                f"""
                CREATE OR REPLACE TABLE positive_core_next AS
                WITH eligible_users AS (
                    SELECT user_id FROM positive_core GROUP BY user_id HAVING count(*) >= {min_user}
                ), eligible_items AS (
                    SELECT item_id FROM positive_core GROUP BY item_id HAVING count(*) >= {min_item}
                )
                SELECT c.*
                FROM positive_core c
                INNER JOIN eligible_users USING (user_id)
                INNER JOIN eligible_items USING (item_id)
                """
            )
            after = con.execute("SELECT count(*) FROM positive_core_next").fetchone()[0]
            con.execute("CREATE OR REPLACE TABLE positive_core AS SELECT * FROM positive_core_next")
            iterations += 1
            if after == before:
                break
            if iterations >= 100:
                raise RuntimeError("Positive k-core did not converge after 100 iterations")

        if con.execute("SELECT count(*) FROM positive_core").fetchone()[0] == 0:
            raise ValueError("The configured positive k-core is empty")

        con.execute(
            """
            CREATE TABLE retained_events AS
            WITH users AS (SELECT DISTINCT user_id FROM positive_core),
                 items AS (SELECT DISTINCT item_id FROM positive_core)
            SELECT e.*,
                   row_number() OVER (
                       PARTITION BY e.user_id ORDER BY e.timestamp, e.item_id
                   ) AS event_order
            FROM events e
            INNER JOIN users USING (user_id)
            INNER JOIN items USING (item_id)
            """
        )
        con.execute(
            """
            CREATE TABLE positive_ranked AS
            SELECT user_id, item_id, timestamp, event_order,
                   row_number() OVER (
                       PARTITION BY user_id ORDER BY event_order DESC
                   ) AS positive_reverse_rank
            FROM retained_events
            WHERE positive_label = 1
            """
        )
        con.execute(
            """
            CREATE TABLE targets AS
            SELECT user_id,
                   max(CASE WHEN positive_reverse_rank = 2 THEN event_order END) AS valid_order,
                   max(CASE WHEN positive_reverse_rank = 1 THEN event_order END) AS test_order
            FROM positive_ranked
            GROUP BY user_id
            """
        )

        base_columns = "r.* EXCLUDE (event_order, positive_label)"
        _copy_parquet(
            con,
            f"""SELECT {base_columns} FROM retained_events r INNER JOIN targets t USING(user_id)
                 WHERE r.event_order < t.valid_order ORDER BY r.user_id, r.timestamp, r.item_id""",
            temp_dir / "train.parquet",
        )
        _copy_parquet(
            con,
            f"""SELECT {base_columns} FROM retained_events r INNER JOIN targets t USING(user_id)
                 WHERE r.event_order = t.valid_order ORDER BY r.user_id""",
            temp_dir / "valid.parquet",
        )
        _copy_parquet(
            con,
            f"""SELECT {base_columns} FROM retained_events r INNER JOIN targets t USING(user_id)
                 WHERE r.event_order = t.test_order ORDER BY r.user_id""",
            temp_dir / "test.parquet",
        )
        _copy_parquet(
            con,
            f"""SELECT {base_columns} FROM retained_events r INNER JOIN targets t USING(user_id)
                 WHERE r.event_order > t.valid_order AND r.event_order < t.test_order
                 ORDER BY r.user_id, r.timestamp, r.item_id""",
            temp_dir / "test_context.parquet",
        )
        _copy_parquet(
            con,
            f"""SELECT {base_columns} FROM retained_events r INNER JOIN targets t USING(user_id)
                 WHERE r.event_order < t.valid_order AND r.rating <= {strong_negative}
                 ORDER BY r.user_id, r.timestamp, r.item_id""",
            temp_dir / "train_strong_negatives.parquet",
        )
        _copy_parquet(
            con,
            f"""
            SELECT i.* FROM read_parquet('{_sql_path(items_path)}') i
            INNER JOIN (SELECT DISTINCT item_id FROM positive_core) c USING(item_id)
            ORDER BY i.item_id
            """,
            temp_dir / "items.parquet",
        )

        split_stats: dict[str, dict[str, int]] = {}
        for split in ("train", "valid", "test", "test_context", "train_strong_negatives"):
            file_path = temp_dir / f"{split}.parquet"
            row = con.execute(
                f"""
                SELECT count(*) AS rows, count(DISTINCT user_id) AS users,
                       count(DISTINCT item_id) AS items,
                       count(*) FILTER (WHERE rating >= {positive_rating}) AS positive_rows
                FROM read_parquet('{_sql_path(file_path)}')
                """
            ).fetchone()
            split_stats[split] = {
                "rows": int(row[0]),
                "users": int(row[1]),
                "items": int(row[2]),
                "positive_rows": int(row[3]),
            }

        checks = {
            "users_without_one_valid_and_test": int(
                con.execute(
                    """
                    SELECT count(*) FROM targets
                    WHERE valid_order IS NULL OR test_order IS NULL
                    """
                ).fetchone()[0]
            ),
            "non_positive_validation_targets": int(
                con.execute(
                    f"SELECT count(*) FROM read_parquet('{_sql_path(temp_dir / 'valid.parquet')}') "
                    f"WHERE rating < {positive_rating}"
                ).fetchone()[0]
            ),
            "non_positive_test_targets": int(
                con.execute(
                    f"SELECT count(*) FROM read_parquet('{_sql_path(temp_dir / 'test.parquet')}') "
                    f"WHERE rating < {positive_rating}"
                ).fetchone()[0]
            ),
            "temporal_violations": int(
                con.execute(
                    f"""
                    WITH tr AS (
                        SELECT user_id, max(timestamp) ts FROM read_parquet('{_sql_path(temp_dir / 'train.parquet')}') GROUP BY user_id
                    ), va AS (
                        SELECT user_id, min(timestamp) ts FROM read_parquet('{_sql_path(temp_dir / 'valid.parquet')}') GROUP BY user_id
                    ), te AS (
                        SELECT user_id, min(timestamp) ts FROM read_parquet('{_sql_path(temp_dir / 'test.parquet')}') GROUP BY user_id
                    )
                    SELECT count(*) FROM tr JOIN va USING(user_id) JOIN te USING(user_id)
                    WHERE tr.ts > va.ts OR va.ts > te.ts
                    """
                ).fetchone()[0]
            ),
            "duplicate_user_item": int(
                con.execute(
                    "SELECT count(*) - count(DISTINCT (user_id, item_id)) FROM retained_events"
                ).fetchone()[0]
            ),
        }
        if any(checks.values()):
            raise RuntimeError(f"Balanced dataset validation failed: {checks}")

        degree_stats = con.execute(
            """
            SELECT min(degree), median(degree), avg(degree), max(degree)
            FROM (SELECT item_id, count(*) degree FROM positive_core GROUP BY item_id)
            """
        ).fetchone()
        core_stats = con.execute(
            """
            SELECT count(*) AS n_rows,
                   count(DISTINCT user_id) AS n_users,
                   count(DISTINCT item_id) AS n_items
            FROM positive_core
            """
        ).fetchone()

        shutil.rmtree(spill_dir, ignore_errors=True)
        output_files = [
            temp_dir / name
            for name in (
                "items.parquet",
                "train.parquet",
                "valid.parquet",
                "test.parquet",
                "test_context.parquet",
                "train_strong_negatives.parquet",
            )
        ]
        manifest = {
            "name": settings["name"],
            "strategy": "positive iterative k-core; positive-target chronological leave-last-out",
            "seed": str(settings["seed"]),
            "settings": settings,
            "kcore_iterations": iterations,
            "positive_core": {
                "rows": int(core_stats[0]),
                "users": int(core_stats[1]),
                "items": int(core_stats[2]),
                "item_degree_min": int(degree_stats[0]),
                "item_degree_median": float(degree_stats[1]),
                "item_degree_mean": float(degree_stats[2]),
                "item_degree_max": int(degree_stats[3]),
            },
            "splits": split_stats,
            "checks": checks,
            "input_sha256": {
                str(path): _sha256(path) for path in [*interaction_paths, items_path]
            },
            "output_sha256": {path.name: _sha256(path) for path in output_files},
        }
        (temp_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        con.close()

    temp_dir.replace(output_dir)
    LOGGER.info("Balanced dataset written to %s", output_dir)
    return BalancedDatasetResult(output_dir=output_dir, manifest=manifest)
