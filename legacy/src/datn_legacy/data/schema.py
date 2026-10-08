from __future__ import annotations

import json
from typing import Any

import polars as pl

INTERACTION_SCHEMA = {
    "user_id": pl.String, "item_id": pl.String, "rating": pl.Float64,
    "timestamp": pl.Int64, "verified_purchase": pl.Boolean,
    "review_title": pl.String, "review_text": pl.String, "helpful_vote": pl.Int64,
}

ITEM_SCHEMA = {
    "item_id": pl.String, "title": pl.String, "description": pl.String,
    "features": pl.String, "category": pl.String, "brand": pl.String,
    "price": pl.Float64, "image_url": pl.String,
}

INTERACTION_COLUMNS = {
    "user_id": None,
    "item_id": None,
    "rating": None,
    "timestamp": None,
    "verified_purchase": None,
    "review_title": None,
    "review_text": None,
    "helpful_vote": None,
}

ITEM_COLUMNS = {
    "item_id": None,
    "title": None,
    "description": None,
    "features": None,
    "category": None,
    "brand": None,
    "price": None,
    "image_url": None,
}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if v is not None).strip() or None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = str(value).strip()
    return value or None


def interaction_record(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": _text(raw.get("user_id")),
        "item_id": _text(raw.get("parent_asin") or raw.get("asin")),
        "rating": raw.get("rating"),
        "timestamp": raw.get("timestamp"),
        "verified_purchase": raw.get("verified_purchase"),
        "review_title": _text(raw.get("title")),
        "review_text": _text(raw.get("text")),
        "helpful_vote": raw.get("helpful_vote"),
    }


def _first_image(raw: dict[str, Any]) -> str | None:
    images = raw.get("images")
    if not isinstance(images, list) or not images:
        return None
    first = images[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return _text(first.get("large") or first.get("hi_res") or first.get("thumb"))
    return None


def item_record(raw: dict[str, Any]) -> dict[str, Any]:
    categories = raw.get("categories")
    category = categories[-1] if isinstance(categories, list) and categories else raw.get("main_category")
    return {
        "item_id": _text(raw.get("parent_asin") or raw.get("asin")),
        "title": _text(raw.get("title")),
        "description": _text(raw.get("description")),
        "features": _text(raw.get("features")),
        "category": _text(category),
        "brand": _text(raw.get("store") or raw.get("brand")),
        "price": raw.get("price"),
        "image_url": _first_image(raw),
    }
