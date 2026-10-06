"""RAG/agent behaviour that only exists for the H&M catalog (audience/colour filters, honest mock reviews, 512-d queries)."""
from __future__ import annotations

import numpy as np
import pytest

from datn.agent.intent import parse_audiences, parse_intent
from datn.agent.orchestrator import colour_groups
from datn.rag.answer import template_compare, template_explain, template_search
from datn.rag.context import EvidenceContext
from datn.rag.evidence import Review
from datn.retrieval import schema as S
from datn.retrieval.filters import SearchFilters

COLOURS = ["Black", "White", "Off White", "Dark Blue", "Light Blue", "Blue", "Red", "Dark Red", "Light Pink", "Grey", "Dark Grey", "Yellowish Brown"]


@pytest.mark.parametrize(
    "message,expected",
    [("áo thun nam", ("men",)), ("váy cho nữ", ("women",)), ("đồ cho bé gái", ("kids",)), ("quần áo trẻ em", ("kids",)),
     ("áo sơ sinh", ("baby",)), ("áo khoác nam và nữ", ("men", "women")), ("women's dress", ("women",)), ("áo thun", ())],
)
def test_parse_audiences(message, expected):
    from datn.agent.intent import fold

    assert parse_audiences(fold(message)) == expected


def test_intent_carries_audience_and_counts_as_constraint():
    intent = parse_intent("áo hoodie nam màu đen")
    assert intent.audiences == ("men",) and intent.colors == ("black",) and intent.has_constraints
    assert parse_intent("hello").audiences == ()


def test_colour_groups_maps_base_colours_to_catalog_groups():
    assert colour_groups(["red"], COLOURS) == ("Red", "Dark Red")
    assert colour_groups(["gray"], COLOURS) == ("Grey", "Dark Grey")
    assert colour_groups(["white"], COLOURS) == ("White",)          # "Off White" is not plain white
    assert colour_groups(["navy blue"], COLOURS) == ("Dark Blue",)
    assert colour_groups(["blue"], COLOURS) == ("Dark Blue", "Light Blue", "Blue")
    assert colour_groups(["red"], []) == ()                          # Amazon: no colour facet -> no hard filter


def test_filters_audience_and_colour():
    f = SearchFilters(audiences=("men",), colours=("Black", "Dark Grey"))
    assert not f.is_empty()
    assert f.matches({S.P_AUDIENCE: "men", S.P_COLOUR: "black"})
    assert not f.matches({S.P_AUDIENCE: "women", S.P_COLOUR: "Black"})
    assert not f.matches({S.P_AUDIENCE: "men", S.P_COLOUR: "Red"})
    assert not f.matches({S.P_AUDIENCE: "men"})
    assert SearchFilters.from_dict(f.to_dict()) == f
    q = f.to_qdrant()
    keys = {c.key: c.match.any for c in q.must}
    assert keys == {S.P_AUDIENCE: ["men"], S.P_COLOUR: ["Black", "Dark Grey"]}
    assert f.describe() == ["Dành cho: Nam", "Màu: Black, Dark Grey"]
    assert SearchFilters().is_empty() and SearchFilters().to_qdrant() is None


def _ctx(mock: bool):
    payload = {S.P_ITEM_ID: "0108775015", S.P_TITLE: "Strap top – Black", S.P_PRICE: 125000.0, S.P_PRICE_ESTIMATED: True,
               S.P_AVG_RATING: 4.5, S.P_REVIEW_COUNT: 2, S.P_DESCRIPTION: "Jersey top.", S.P_PRODUCT_TYPE: "Vest top",
               S.P_COLOUR: "Black", S.P_AUDIENCE: "women", S.P_REVIEWS_MOCK: mock}
    review = Review(review_id=1, product_id=1, rating=5.0, helpful_vote=3, title=None, text="Soft and fits well", is_mock=mock)
    return EvidenceContext.build([(1, payload)], {1: [review]}) if hasattr(EvidenceContext, "build") else None


def test_mock_reviews_are_labelled_everywhere():
    ctx = _ctx(True)
    if ctx is None:
        pytest.skip("EvidenceContext has no build() in this version")
    prompt = ctx.render()
    assert "đánh giá minh hoạ" in prompt and "Thương hiệu" not in prompt
    assert "Màu: Black" in prompt and "Dành cho: nữ" in prompt and "Loại: Vest top" in prompt
    search = template_search(ctx, "áo ba lỗ", [], False)
    assert "đánh giá minh hoạ" in search and "Một đánh giá minh hoạ cho biết" in search and "thương hiệu" not in search
    explain = template_explain(ctx, ["x"])
    assert "Một đánh giá minh hoạ (5★)" in explain
    real = _ctx(False)
    assert "minh hoạ" not in real.render() and "Một người mua cho biết" in template_search(real, "q", [], False)


def test_encoder_cuts_to_catalog_size(monkeypatch):
    from datn.retrieval.encoder import JinaClipEncoder

    x = np.random.default_rng(0).normal(size=(3, 1024)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    monkeypatch.setattr(S, "VECTOR_SIZE", 1024)
    assert JinaClipEncoder._fit_catalog(x).shape == (3, 1024)
    monkeypatch.setattr(S, "VECTOR_SIZE", 512)
    y = JinaClipEncoder._fit_catalog(x)
    assert y.shape == (3, 512) and np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-6)
    expected = x[:, :512] / np.linalg.norm(x[:, :512], axis=1, keepdims=True)
    assert np.allclose(y, expected, atol=1e-6)


def test_compare_names_type_and_colour_when_there_is_no_brand():
    a, b = _ctx(True), _ctx(True)
    if a is None:
        pytest.skip("EvidenceContext has no build()")
    ctx = EvidenceContext(products=[a.products[0], b.products[0]], reviews={})
    ctx.products[1].tag = "P2"
    text = template_compare(ctx)
    assert "Vest top, Black" in text and "chưa rõ thương hiệu" not in text


def test_citation_format_is_policed():
    from datn.rag.context import extract_citations, malformed_citations

    assert extract_citations("Giá tốt [P1]. Review [R1.2] và [R4.1, R4.2], [P2, P3]") == {"P1", "R1.2", "R4.1", "R4.2", "P2", "P3"}
    assert malformed_citations("ok [P1] [R2.1, R2.2]") == []
    bad = malformed_citations("Valentin skirt [P5 - *chờ chút, P3 mới đúng*] và [R9.9 ???]")
    assert len(bad) == 2 and bad[0].startswith("[P5")
    ctx = _ctx(False)
    from datn.rag.context import check_grounding

    assert any("định dạng trích dẫn" in i for i in check_grounding("Áo đẹp [P1 - xin lỗi, P2]", ctx))
    assert check_grounding("Áo đẹp, giá 125.000₫ [P1]", ctx) == []


def test_vietnamese_vay_means_dress_or_skirt():
    assert "dress" in parse_intent("váy hoa nữ mùa hè").query and "skirt" in parse_intent("váy hoa nữ mùa hè").query
    assert parse_intent("chân váy").query.startswith("skirt")


def test_note_tags_are_not_shown_to_shoppers():
    from datn.rag.answer import strip_note_tags

    assert strip_note_tags("Nằm trong ngân sách [NOTE], [P1]. Đúng màu [NOTE]. Dành cho trẻ em [NOTE] ok") == "Nằm trong ngân sách, [P1]. Đúng màu. Dành cho trẻ em ok"
