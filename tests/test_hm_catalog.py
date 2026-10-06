"""datn.catalog.hm (shop catalog builder) and datn.retrieval.hm_indexer.hm_payload."""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from datn.catalog.hm import AUDIENCES, CATEGORIES, audience_of, build_shop_catalog, category_slug_of

AS_OF = dt.date(2020, 9, 16)


@pytest.mark.parametrize(
    "section,expected",
    [
        ("Womens Everyday Collection", "women"), ("Ladies H&M Sport", "women"), ("Mama", "women"), ("Contemporary Street", "women"),
        ("H&M+", "women"), ("Men Underwear", "men"), ("Mens Outerwear", "men"), ("Denim Men", "men"),
        ("Divided Collection", "divided"), ("EQ Divided", "divided"), ("Kids Girl", "kids"), ("Young Boy", "kids"),
        ("Girls Underwear & Basics", "kids"), ("Boys Underwear & Basics", "kids"), ("Baby Girl", "baby"),
        ("Special Collections", "other"), ("Collaborations", "other"), ("", "other"), (None, "other"),
    ],
)
def test_audience_of(section, expected):
    assert audience_of(section) == expected
    assert audience_of(section) in AUDIENCES


def test_every_h_and_m_product_group_has_a_category():
    slugs = {s for s, _ in CATEGORIES}
    for group in ["Garment Upper body", "Garment Lower body", "Garment Full body", "Accessories", "Underwear", "Shoes", "Swimwear",
                  "Socks & Tights", "Nightwear", "Underwear/nightwear", "Bags"]:
        assert category_slug_of(group) in slugs and category_slug_of(group) != "khac"
    for rare in ["Unknown", "Cosmetic", "Furniture", None, ""]:
        assert category_slug_of(rare) == "khac"


def _fixture():
    items = pl.DataFrame(
        {
            "item_id": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "prod_name": ["Strap top", "Skinny jeans", "Mystery", "Teddy"],
            "product_type_name": ["Vest top", "Trousers", "Vest top", "Toy"],
            "product_group_name": ["Garment Upper body", "Garment Lower body", "Garment Upper body", "Fun"],
            "graphical_appearance_name": ["Solid", "Denim", "Solid", "Solid"],
            "colour_group_name": ["Black", "Blue", None, "Brown"],
            "department_name": ["Jersey Basic", "Trouser", "Jersey Basic", "Toys"],
            "section_name": ["Womens Everyday Basics", "Men Denim", "Womens Trend", "Baby Boy"],
            "detail_desc": ["Jersey top.", None, "x", "Soft toy."],
        }
    )
    counts = pl.DataFrame(
        {
            "article_id": ["0000000001", "0000000001", "0000000002", "0000000004", "0000000001"],
            "t_dat": [dt.date(2020, 9, 1), dt.date(2020, 8, 1), dt.date(2020, 9, 10), dt.date(2020, 9, 20), dt.date(2020, 9, 20)],
            "n": [3, 1, 2, 9, 9],
            "price_sum": [0.09, 0.02, 0.10, 9.0, 9.0],
        }
    ).with_columns(pl.col("n").cast(pl.UInt32))
    return items, counts


def test_build_shop_catalog():
    items, counts = _fixture()
    cat, m = build_shop_catalog(items, counts, as_of=AS_OF, image_ids={"0000000001", "0000000002", "0000000003"}, median_target_vnd=350_000)
    assert cat["product_id"].to_list() == [1, 2, 3, 4] and cat["sku"].to_list() == items["item_id"].to_list()
    assert cat["active"].to_list() == [True, True, True, False]
    assert cat["name"].to_list()[0] == "Strap top – Black" and cat["name"].to_list()[2] == "Mystery"   # no colour -> bare name
    assert cat["description"].to_list()[1] == ""
    # sales on/after the shop clock must not leak in: item 4 only sold on 2020-09-20 -> never sold, price imputed
    assert cat["n_all"].to_list() == [4, 2, 0, 0]
    assert cat["sold_28d"].to_list() == [3, 2, 0, 0]          # item 1: only the 09-01 sale is inside the 28 days (08-01 is not)
    assert m["price_imputed"] == 2
    # median of the *sold* items' raw prices maps to the target price (rounded to 1000)
    sold_raw = sorted(cat.filter(pl.col("n_all") > 0)["price_raw"].to_list())
    assert abs(m["price_raw_median_sold"] - sum(sold_raw) / 2) < 1e-12
    assert 0 < cat["price_vnd"].min() and all(p % 1000 == 0 for p in cat["price_vnd"].to_list())
    assert cat["price_estimated"].all()
    assert cat["image_url"].to_list()[0] == "/api/static/hm/000/0000000001.jpg" and cat["image_url"].to_list()[3] is None
    assert cat["category_slug"].to_list() == ["ao", "quan-chan-vay", "ao", "khac"]
    assert cat["audience"].to_list() == ["women", "men", "women", "baby"]
    assert m["active"] == 3 and m["items"] == 4


def test_build_is_deterministic():
    items, counts = _fixture()
    a, _ = build_shop_catalog(items, counts, as_of=AS_OF, image_ids={"0000000001"})
    b, _ = build_shop_catalog(items, counts, as_of=AS_OF, image_ids={"0000000001"})
    assert a.equals(b)


def test_hm_payload():
    pytest.importorskip("qdrant_client")
    from datn.retrieval import schema as S
    from datn.retrieval.hm_indexer import hm_payload

    row = {
        "product_id": 7, "item_id": "0000000007", "name": "Strap top – Black", "category_id": 2, "category_slug": "ao", "price_vnd": 349000.0,
        "image_url": "/api/static/hm/000/0000000007.jpg", "description": "Jersey top.", "audience": "women",
        "attributes": {"product_type": "Vest top", "colour": "Black", "section": "Womens Everyday Basics", "sold_28d": 12},
    }
    p = hm_payload(row)
    assert p[S.P_PRODUCT_ID] == 7 and p[S.P_ITEM_ID] == "0000000007" and p[S.P_PRICE_ESTIMATED] is True
    assert p[S.P_COLOUR] == "Black" and p[S.P_PRODUCT_TYPE] == "Vest top" and p[S.P_SOLD_28D] == 12 and p[S.P_AUDIENCE] == "women"
    assert S.P_BRAND not in p and S.P_AVG_RATING not in p            # no brand in H&M; rating comes with the review step
    assert S.P_PRODUCT_GROUP not in p                                # missing facets are dropped, not stored as null


def test_review_payload_and_stats():
    pytest.importorskip("qdrant_client")
    from datn.retrieval import schema as S
    from datn.retrieval.hm_indexer import product_review_stats, review_payload

    p = review_payload({"product_id": 3, "item_id": "0000000003", "rating": 4, "helpful_vote": None, "verified_purchase": 1,
                        "review_title": "", "review_text": "Nice fit", "source_item_id": "B00X", "match_score": 0.71234567})
    assert p[S.R_IS_MOCK] is True and p["source_item_id"] == "B00X" and p[S.R_HELPFUL] == 0 and p["match_score"] == 0.7123
    assert S.R_TITLE not in p
    meta = pl.DataFrame({"product_id": [1, 1, 2], "rating": [5.0, 4.0, 2.0]})
    assert product_review_stats(meta) == {1: (2, 4.5), 2: (1, 2.0)}
