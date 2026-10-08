"""Shop catalog for the H&M dataset: what the marketplace sells, derived from the thesis dataset.

Pure functions (polars in, polars out) so they can be unit-tested without Postgres/Qdrant; the CLI in
`datn.catalog.cli` does the I/O.

Facts about the source worth knowing:
* H&M transaction prices are normalised (median ~0.025), not money. `price_vnd` is therefore a documented
  convention -- `price_raw * factor`, factor chosen so the catalog median is `median_target_vnd` -- and every
  product is flagged `price_estimated`.
* `items.parquet` has no `index_group_name`; the audience is derived from `section_name`.
* `product_id` = `items.parquet` row + 1 = the recommender's item index (checked against the tower checkpoint).
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Collection

import numpy as np
import polars as pl

# (slug, Vietnamese name). The shop's navigation; H&M product_group_name is folded into these.
CATEGORIES: list[tuple[str, str]] = [
    ("ao", "Áo"),
    ("quan-chan-vay", "Quần & Chân váy"),
    ("dam-jumpsuit", "Đầm & Jumpsuit"),
    ("do-lot", "Đồ lót"),
    ("do-ngu", "Đồ ngủ"),
    ("do-boi", "Đồ bơi"),
    ("tat-quan-tat", "Tất & Quần tất"),
    ("giay-dep", "Giày dép"),
    ("phu-kien", "Phụ kiện"),
    ("khac", "Khác"),
]
_GROUP_TO_SLUG = {
    "Garment Upper body": "ao",
    "Garment Lower body": "quan-chan-vay",
    "Garment Full body": "dam-jumpsuit",
    "Underwear": "do-lot",
    "Underwear/nightwear": "do-lot",
    "Nightwear": "do-ngu",
    "Swimwear": "do-boi",
    "Socks & Tights": "tat-quan-tat",
    "Shoes": "giay-dep",
    "Accessories": "phu-kien",
    "Bags": "phu-kien",
}

AUDIENCES = ("women", "men", "divided", "kids", "baby", "other")
# Order matters: "Girls Underwear & Basics" / "Boys ..." are kids, not a generic catch-all.
_AUDIENCE_PREFIXES: list[tuple[str, tuple[str, ...]]] = [
    ("baby", ("Baby",)),
    ("kids", ("Kids", "Young", "Girls", "Boys")),
    ("men", ("Men", "Mens", "Denim Men")),
    ("divided", ("Divided", "EQ Divided")),
    ("women", ("Womens", "Ladies", "Mama", "Contemporary", "H&M+")),
]


def audience_of(section: str | None) -> str:
    s = (section or "").strip()
    for audience, prefixes in _AUDIENCE_PREFIXES:
        if s.startswith(prefixes):
            return audience
    return "other"


def category_slug_of(product_group: str | None) -> str:
    return _GROUP_TO_SLUG.get((product_group or "").strip(), "khac")


def raw_prices(items: pl.DataFrame, daily_counts: pl.DataFrame, as_of: dt.date) -> pl.DataFrame:
    """Mean transaction price per item before `as_of`; items without sales get their product type's median, then the global one."""
    before = daily_counts.filter(pl.col("t_dat") < pl.lit(as_of))
    sold = (
        before.group_by("article_id")
        .agg(pl.col("price_sum").sum().alias("psum"), pl.col("n").sum().cast(pl.Int64).alias("n_all"))
        .with_columns((pl.col("psum") / pl.col("n_all")).alias("price_raw"))
        .select("article_id", "price_raw", "n_all")
    )
    sold28 = (
        before.filter(pl.col("t_dat") >= pl.lit(as_of - dt.timedelta(days=28)))
        .group_by("article_id")
        .agg(pl.col("n").sum().cast(pl.Int64).alias("sold_28d"))
    )
    out = (
        items.select("item_id", "product_type_name")
        .join(sold, left_on="item_id", right_on="article_id", how="left")
        .join(sold28, left_on="item_id", right_on="article_id", how="left")
        .with_columns(pl.col("n_all").fill_null(0), pl.col("sold_28d").fill_null(0))
    )
    by_type = (
        out.filter(pl.col("price_raw").is_not_null())
        .group_by("product_type_name")
        .agg(pl.col("price_raw").median().alias("type_median"))
    )
    global_median = float(out["price_raw"].drop_nulls().median())
    out = out.join(by_type, on="product_type_name", how="left", maintain_order="left").with_columns(
        pl.coalesce("price_raw", "type_median", pl.lit(global_median)).alias("price_final"),
        pl.col("price_raw").is_null().alias("price_imputed"),
    )
    return out.select("item_id", pl.col("price_final").alias("price_raw"), "price_imputed", "n_all", "sold_28d")


def build_shop_catalog(
    items: pl.DataFrame,
    daily_counts: pl.DataFrame,
    *,
    as_of: dt.date,
    image_ids: Collection[str],
    median_target_vnd: int = 350_000,
    min_price_vnd: int = 20_000,
    seed: int = 20261006,
) -> tuple[pl.DataFrame, dict]:
    """One row per catalog item, in `items` order (product_id = row + 1).

    `image_ids`: article_ids whose photo exists on disk; only those are `active` (shown in the shop).
    """
    n = items.height
    prices = raw_prices(items, daily_counts, as_of)
    if prices["item_id"].to_list() != items["item_id"].to_list():
        raise AssertionError("price table lost the catalog order")
    sold = prices.filter(~pl.col("price_imputed"))["price_raw"]
    factor = median_target_vnd / float(sold.median())
    price_vnd = ((prices["price_raw"] * factor / 1000.0).round(0) * 1000.0).clip(lower_bound=float(min_price_vnd))

    rng = np.random.default_rng(seed)
    stock = rng.integers(15, 121, size=n)
    image_set = set(image_ids)
    f = items.to_dict(as_series=False)

    def col(name: str) -> list[str]:
        return [v or "" for v in f[name]]

    names = [f"{p} – {c}" if c else p for p, c in zip(col("prod_name"), col("colour_group_name"))]
    ids = f["item_id"]
    active = [i in image_set for i in ids]
    attributes = [
        json.dumps(
            {"product_type": pt, "product_group": pg, "colour": c, "appearance": ap, "section": s, "department": d, "sold_28d": int(s28)},
            ensure_ascii=False,
        )
        for pt, pg, c, ap, s, d, s28 in zip(
            col("product_type_name"), col("product_group_name"), col("colour_group_name"), col("graphical_appearance_name"),
            col("section_name"), col("department_name"), prices["sold_28d"].to_list(),
        )
    ]
    cat = pl.DataFrame(
        {
            "product_id": list(range(1, n + 1)),
            "sku": ids,
            "name": names,
            "description": col("detail_desc"),
            "price_vnd": price_vnd.to_list(),
            "price_raw": prices["price_raw"].to_list(),
            "price_estimated": [True] * n,
            "stock": stock.tolist(),
            "category_slug": [category_slug_of(g) for g in col("product_group_name")],
            "audience": [audience_of(s) for s in col("section_name")],
            "active": active,
            "image_url": [f"/api/static/hm/{i[:3]}/{i}.jpg" if a else None for i, a in zip(ids, active)],
            "attributes": attributes,
            "sold_28d": prices["sold_28d"].to_list(),
            "n_all": prices["n_all"].to_list(),
        }
    )

    act = cat.filter(pl.col("active"))
    manifest = {
        "as_of": as_of.isoformat(),
        "items": n,
        "active": act.height,
        "price_factor": factor,
        "median_target_vnd": median_target_vnd,
        "price_raw_median_sold": float(sold.median()),
        "price_imputed": int(prices["price_imputed"].sum()),
        "price_vnd_active_median": float(act["price_vnd"].median()) if act.height else None,
        "by_category": {k: int(v) for k, v in sorted(act["category_slug"].value_counts().iter_rows())},
        "by_audience": {k: int(v) for k, v in sorted(act["audience"].value_counts().iter_rows())},
        "never_sold_active": int((act["n_all"] == 0).sum()),
        "selling_28d_active": int((act["sold_28d"] > 0).sum()),
    }
    return cat, manifest
