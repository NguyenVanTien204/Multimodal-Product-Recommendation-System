from __future__ import annotations

import numpy as np
import pytest

from datn.catalog.mock_reviews import (
    assign_reviews,
    colour_conflict,
    document_frequency,
    has_external_reference,
    make_brand_set,
    mentions_brand,
    tokens,
    top_matches,
    truncate_renorm,
)


def test_truncate_renorm_matches_matryoshka_cut():
    x = np.random.default_rng(0).normal(size=(5, 1024)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = truncate_renorm(x, 512)
    assert y.shape == (5, 512) and np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-6)
    cos = (y * (x[:, :512] / np.linalg.norm(x[:, :512], axis=1, keepdims=True))).sum(1)
    assert np.allclose(cos, 1, atol=1e-6)


def test_brand_screening():
    docs = [tokens(t) for t in ["love this nike shirt", "love it", "love the fit", "so soft", "nice levis jeans"] + ["ok", "if you like it"] * 48]
    df = document_frequency(docs)
    counts = {"Nike": 400, "Levi's": 300, "Love": 90, "Calvin Klein": 200, "if you": 60, "Guess": 80, "ab": 90, "Tiny Brand": 3, "party": 70}
    brands = make_brand_set(counts, df, min_products=25, n_common=5)
    assert {"nike", "levi's", "calvin klein"} <= brands
    assert "if you" not in brands        # only very common review words
    assert "guess" not in brands and "party" not in brands        # everyday words on the stoplist
    assert "tiny brand" not in brands    # not an established brand (3 products)
    assert "ab" not in brands            # too short
    assert mentions_brand(tokens("These Calvin Klein briefs run small"), brands) == "calvin klein"
    assert mentions_brand(tokens("great fit and soft"), brands) is None


@pytest.mark.parametrize("text,bad", [("see https://x.com/a", True), ("email me a@b.co", True), ("bought on Amazon Prime", True),
                                      ("the seller was slow", True), ("fits perfectly", False)])
def test_external_reference(text, bad):
    assert has_external_reference(text) is bad


@pytest.mark.parametrize(
    "review,colour,conflict",
    [("love the black color", "Black", False), ("love the black color", "Dark Blue", True), ("navy and soft", "Dark Blue", False),
     ("so comfy", "Red", False), ("grey hoodie", "Dark Grey", False), ("gray hoodie", "Light Grey", False),
     ("red or pink?", "Light Pink", False), ("white tee", None, True)],
)
def test_colour_conflict(review, colour, conflict):
    assert colour_conflict(tokens(review), colour) is conflict


def test_top_matches_is_exact():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(30, 16)).astype("float32"), rng.normal(size=(100, 16)).astype("float32")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    idx, sim = top_matches(a, b, k=4, chunk=7)
    brute = np.argsort(-(a @ b.T), axis=1)[:, :4]
    assert (idx == brute).all() and (np.diff(sim, axis=1) <= 1e-7).all()


def _rv(i, asin, text):
    return {"review_id": i, "source_item_id": asin, "text": text, "rating": 5.0}


def test_assign_reviews_rules():
    cands = {"A": [(1, 0.9), (2, 0.8), (3, 0.3)], "B": [(2, 0.4)], "C": []}
    colours = {1: "Black", 2: "Red", 3: "Black"}
    reviews = [
        _rv(1, "A", "great fit"), _rv(2, "A", "great fit too"), _rv(3, "A", "see www.shop.com"),
        _rv(4, "A", "loved the red one"),             # only product 2 (Red) is colour-compatible
        _rv(5, "A", "Nike quality"),                   # brand mention
        _rv(6, "B", "nice"),                           # similarity 0.4 < floor
        _rv(7, "C", "nice"),                           # no candidates
    ]
    out, rej = assign_reviews(reviews, cands, colours, min_sim=0.5, min_chars=0, cap_per_item=2, brand_set={"nike"})
    by_id = {a["review_id"]: a for a in out}
    assert set(by_id) == {1, 2, 4}
    assert by_id[4]["product_id"] == 2
    assert {by_id[1]["product_id"], by_id[2]["product_id"]} <= {1, 2}
    assert rej == {"external_reference": 1, "brand_mention": 1, "no_similar_product": 2}
    # one review -> one product, and the cap is respected
    assert len({a["review_id"] for a in out}) == len(out)
    out2, rej2 = assign_reviews([_rv(i, "A", "fine") for i in range(10)], {"A": [(1, 0.9)]}, {1: None}, min_sim=0.5, min_chars=0, cap_per_item=3)
    assert len(out2) == 3 and rej2 == {"capacity": 7}


def test_assign_is_deterministic_and_balances_load():
    cands = {"A": [(1, 0.9), (2, 0.85), (3, 0.8)]}
    reviews = [_rv(i, "A", "fine fit") for i in range(9)]
    a, _ = assign_reviews(reviews, cands, {1: None, 2: None, 3: None}, min_sim=0.5, min_chars=0, cap_per_item=6)
    b, _ = assign_reviews(reviews, cands, {1: None, 2: None, 3: None}, min_sim=0.5, min_chars=0, cap_per_item=6)
    assert a == b
    counts = [sum(x["product_id"] == p for x in a) for p in (1, 2, 3)]
    assert counts == [3, 3, 3]


@pytest.mark.parametrize(
    "title,expected",
    [("Levi's Boys' Big 550 Relaxed-fit Jeans", {"kids"}), ("Milumia Women's Bishop Sleeve Sweater", {"women", "divided"}),
     ("Mens Cotton Polo Shirt", {"men", "divided"}), ("Baby Boy Romper", {"baby"}), ("Unisex Beanie", None), ("", None), (None, None),
     ("Men Women Water Sports Shoes", {"men", "women", "divided"})],
)
def test_allowed_audiences(title, expected):
    from datn.catalog.mock_reviews import allowed_audiences

    assert allowed_audiences(title) == expected


def test_clean_review_text_and_short_reviews_are_dropped():
    from datn.catalog.mock_reviews import clean_review_text

    raw = "[[VIDEOID:e1e6520e7428]] This dress is 5&#39;8&quot; friendly<br />  and soft "
    assert clean_review_text(raw) == "This dress is 5'8\" friendly and soft"
    assert clean_review_text(None) == ""
    out, rej = assign_reviews([_rv(1, "A", "Nice."), _rv(2, "A", "A longer, perfectly fine review")], {"A": [(1, 0.9)]}, {1: None}, min_sim=0.5)
    assert [a["review_id"] for a in out] == [2] and rej == {"too_short": 1}
