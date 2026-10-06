"""Pure-Python tests for the chatbot building blocks (no torch / Qdrant / LLM needed)."""

from __future__ import annotations

import json
import time

import pytest

from datn.agent.intent import detect_lang, fold, parse_intent, parse_price
from datn.agent.orchestrator import _is_stricter
from datn.agent.session import SessionStore, ShownProduct
from datn.rag.answer import AnswerGenerator, template_compare, template_search
from datn.rag.context import EvidenceContext, check_grounding, extract_citations, format_vnd
from datn.rag.evidence import Review
from datn.rag.llm import OpenAICompatLLM, parse_json_object
from datn.retrieval import schema as S
from datn.retrieval.filters import Hit, SearchFilters
from datn.retrieval.search import rrf_fuse

BRANDS = {"nike": "Nike", "adidas": "Adidas"}


# ---- text normalisation ------------------------------------------------------------
def test_fold_keeps_length_and_strips_diacritics():
    text = "Giày thể thao đen"
    assert len(fold(text)) == len(text)
    assert fold(text) == "giay the thao den"


def test_detect_lang():
    assert detect_lang("giày chạy bộ") == "vi"
    assert detect_lang("black shoes under 100") == "en"
    assert detect_lang("tim ao so mi") == "vi"  # no diacritics, Vietnamese words


# ---- price parsing -----------------------------------------------------------------
@pytest.mark.parametrize(
    "text,lang,lo,hi",
    [
        ("dưới 500k", "vi", None, 500_000),
        ("giá dưới 1 triệu", "vi", None, 1_000_000),
        ("dưới 1.5tr", "vi", None, 1_500_000),
        ("từ 300k đến 600k", "vi", 300_000, 600_000),
        ("khoảng 2 triệu", "vi", 1_500_000, 2_500_000),
        ("trên 200k", "vi", 200_000, None),
        ("under 100", "en", None, 100 * S.USD_TO_VND),
        ("over $80", "en", 80 * S.USD_TO_VND, None),
        ("dưới 500", "vi", None, 500_000),  # bare number in Vietnamese == thousands of dong
    ],
)
def test_parse_price(text, lang, lo, hi):
    spec = parse_price(fold(text), lang)
    if lo is None:
        assert spec.min_price is None
    else:
        assert spec.min_price == pytest.approx(lo)
    if hi is None:
        assert spec.max_price is None
    else:
        assert spec.max_price == pytest.approx(hi)


def test_parse_price_relative_modes():
    assert parse_price(fold("rẻ hơn"), "vi").mode == "cheaper"
    assert parse_price(fold("cao cấp hơn"), "vi").mode == "pricier"
    assert parse_price(fold("rẻ hơn dưới 300k"), "vi").mode is None  # explicit number wins


# ---- intent routing ----------------------------------------------------------------
def test_search_extracts_constraints_and_glosses_query():
    i = parse_intent("giày chạy bộ nam màu đen dưới 500k", known_brands=BRANDS)
    assert i.action == "search"
    assert i.max_price == 500_000
    assert i.colors == ("black",)
    assert "running shoes" in i.query and "black" in i.query
    assert "500" not in i.query


def test_english_query_and_brand():
    i = parse_intent("Nike shoes over $80 rating 4 stars", known_brands=BRANDS)
    assert i.action == "search" and i.brands == ("Nike",)
    assert i.min_price == pytest.approx(80 * S.USD_TO_VND)
    assert i.min_rating == 4.0


@pytest.mark.parametrize(
    "msg,action",
    [
        ("xin chào", "greet"),
        ("bạn làm được gì", "help"),
        ("gợi ý cho tôi", "recommend"),
        ("xóa bộ lọc", "reset"),
    ],
)
def test_no_results_needed_actions(msg, action):
    assert parse_intent(msg).action == action


@pytest.mark.parametrize(
    "msg,action",
    [
        ("rẻ hơn", "refine"),
        ("màu đỏ", "refine"),
        ("thương hiệu Nike", "refine"),
        ("cái khác đi", "refine"),
        ("tại sao gợi ý sản phẩm 2", "explain"),
        ("review của #2 thế nào", "explain"),
        ("so sánh 1 và 3", "compare"),
        ("giống cái đầu tiên nhưng dưới 1 triệu", "similar"),
        ("đồng hồ nam", "search"),  # a new product noun starts a new search even with results on screen
    ],
)
def test_followup_actions_require_previous_results(msg, action):
    assert parse_intent(msg, has_results=True, known_brands=BRANDS).action == action


def test_refine_without_results_is_not_refine():
    assert parse_intent("rẻ hơn", has_results=False).action != "refine"


def test_ordinals():
    assert parse_intent("so sánh 1 và 3", has_results=True).ordinals == (1, 3)
    assert parse_intent("tại sao sản phẩm thứ hai", has_results=True).ordinals == (2,)
    assert parse_intent("why is #2 good", has_results=True).ordinals == (2,)


def test_image_only_message_is_search():
    assert parse_intent("", has_image=True).action == "search"
    assert parse_intent("tìm cái giống thế này", has_image=True).action in {"search", "similar"}


# ---- filters ------------------------------------------------------------------------
def test_filters_in_memory_match_and_describe():
    f = SearchFilters(min_price=100, max_price=500, brands=("Nike",), min_rating=4)
    good = {S.P_PRICE: 300, S.P_BRAND: "nike", S.P_AVG_RATING: 4.5, S.P_PRODUCT_ID: 1}
    assert f.matches(good)
    assert not f.matches({**good, S.P_PRICE: 600})
    assert not f.matches({**good, S.P_BRAND: "Puma"})
    assert not f.matches({**good, S.P_AVG_RATING: 3.0})
    assert not f.matches({**good, S.P_PRICE: None})  # unknown price can't satisfy a price bound
    assert SearchFilters(exclude_product_ids=(1,)).matches(good) is False
    assert any("Giá" in c for c in f.describe())


def test_filters_roundtrip_and_qdrant_translation():
    f = SearchFilters(max_price=1e6, brands=("A",), exclude_product_ids=(5, 6))
    assert SearchFilters.from_dict(f.to_dict()) == f
    q = f.to_qdrant()
    assert q is not None and len(q.must) == 2 and len(q.must_not) == 1
    assert SearchFilters().to_qdrant() is None and SearchFilters().is_empty()


def test_pool_reuse_only_for_stricter_filters():
    old = SearchFilters(max_price=1_000_000)
    assert _is_stricter(SearchFilters(max_price=500_000), old)
    assert not _is_stricter(SearchFilters(max_price=2_000_000), old)
    assert not _is_stricter(SearchFilters(), old)  # filter removed -> pool is too narrow
    assert _is_stricter(SearchFilters(brands=("Nike",)), SearchFilters())  # adding a filter is fine


# ---- fusion --------------------------------------------------------------------------
def _hit(pid, score=0.5):
    return Hit(pid, {S.P_PRODUCT_ID: pid}, score)


def test_rrf_prefers_items_ranked_well_by_several_signals():
    fused = rrf_fuse(
        {"text": [_hit(1), _hit(2), _hit(3)], "image": [_hit(3), _hit(2), _hit(9)]},
        {"text": 1.0, "image": 1.0},
    )
    assert [h.product_id for h in fused[:2]] == [2, 3] or fused[0].product_id in {2, 3}
    top = fused[0]
    assert set(top.ranks) == {"text", "image"}


def test_rrf_weights_matter():
    a = rrf_fuse({"text": [_hit(1)], "image": [_hit(2)]}, {"text": 1.0, "image": 0.1})
    assert a[0].product_id == 1


# ---- sessions ------------------------------------------------------------------------
def test_session_store_ttl_and_lru():
    store = SessionStore(ttl_s=0.05, max_sessions=2)
    s = store.get_or_create(None)
    assert store.get(s.session_id) is s
    time.sleep(0.07)
    assert store.get(s.session_id) is None
    store2 = SessionStore(ttl_s=60, max_sessions=2)
    ids = [store2.get_or_create(None).session_id for _ in range(3)]
    assert len(store2) == 2 and store2.get(ids[0]) is None


def test_session_resolves_ordinals_and_engagement():
    store = SessionStore()
    s = store.get_or_create(None)
    s.last_results = [ShownProduct(10, {}), ShownProduct(20, {})]
    assert [p.product_id for p in s.resolve([2, 5])] == [20]
    s.engage("A")
    s.engage("A")
    s.engage("B")
    assert [(e.sku, e.kind) for e in s.events] == [("A", "click"), ("B", "click")]


# ---- grounding -------------------------------------------------------------------------
def _ctx():
    payload = {
        S.P_ITEM_ID: "B0X", S.P_TITLE: "Trail shoe", S.P_BRAND: "Acme", S.P_CATEGORY: "Shoes",
        S.P_PRICE: 1_250_000.0, S.P_PRICE_ESTIMATED: False, S.P_AVG_RATING: 4.4, S.P_REVIEW_COUNT: 12,
        S.P_DESCRIPTION: "Light shoe", S.P_FEATURES: "Mesh upper",
    }
    payload2 = {**payload, S.P_ITEM_ID: "B0Y", S.P_TITLE: "Road shoe", S.P_PRICE: 900_000.0, S.P_PRICE_ESTIMATED: True}
    rev = Review(1, 1, 5.0, 3, "Great", "Very comfortable on long runs", 0.7)
    return EvidenceContext.build([(1, payload), (2, payload2)], {1: [rev]}, user_thresholds=[1_500_000])


def test_context_tags_and_render():
    ctx = _ctx()
    assert ctx.valid_tags() == {"P1", "P2", "R1.1"}
    text = ctx.render()
    assert "[P1]" in text and "1.250.000₫" in text and "giá tham khảo" in text and "[R1.1]" in text
    assert "chưa có nhận xét" in text  # product 2 has no reviews and says so explicitly


def test_grounding_accepts_supported_answer():
    ctx = _ctx()
    ans = "Trail shoe giá 1.250.000₫ [P1], một người mua khen êm [R1.1]. Rẻ hơn 350.000₫ nếu chọn [P2]."
    assert check_grounding(ans, ctx) == []
    assert extract_citations(ans) == {"P1", "R1.1", "P2"}


def test_grounding_rejects_invented_citation_and_price():
    ctx = _ctx()
    issues = check_grounding("Giá chỉ 999.000₫ [P1] và [R9.9]", ctx)
    assert any("trích dẫn" in i for i in issues)
    assert any("số tiền" in i for i in issues)


def test_grounding_understands_short_units():
    ctx = _ctx()
    assert check_grounding("khoảng 1,25 triệu [P1]", ctx) == []
    assert check_grounding("khoảng 900k [P2]", ctx) == []
    assert check_grounding("khoảng 800k [P2]", ctx) != []


def test_format_vnd():
    assert format_vnd(1234567) == "1.234.567₫"


# ---- answer generation -------------------------------------------------------------------
class _ScriptedLLM:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    @property
    def enabled(self):
        return True

    async def complete(self, system, user, **kw):
        self.calls += 1
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_generator_uses_llm_when_grounded():
    ctx = _ctx()
    llm = _ScriptedLLM(["Nên chọn [P1] (1.250.000₫), người mua nói êm [R1.1]."])
    out = await AnswerGenerator(llm).generate("search", "giày", ctx, "FALLBACK")
    assert out.source == "llm" and llm.calls == 1


@pytest.mark.asyncio
async def test_generator_retries_then_falls_back_on_hallucination():
    ctx = _ctx()
    llm = _ScriptedLLM(["Giá 100.000₫ [P1]", "Vẫn sai 5.000.000₫ [P1]"])
    out = await AnswerGenerator(llm).generate("search", "giày", ctx, "FALLBACK")
    assert out.source == "template" and out.text == "FALLBACK" and llm.calls == 2
    assert out.warnings


@pytest.mark.asyncio
async def test_generator_without_llm_returns_template():
    ctx = _ctx()
    out = await AnswerGenerator(OpenAICompatLLM()).generate("search", "giày", ctx, "FALLBACK")
    assert out.source == "template" and out.text == "FALLBACK"


def test_templates_are_self_grounded():
    ctx = _ctx()
    for text in (template_search(ctx, "giày", ["Giá ≤ 1.500.000₫"], True), template_compare(ctx)):
        assert check_grounding(text, ctx) == [], text


def test_json_extraction():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"action": "search"} ok') == {"action": "search"}
    assert parse_json_object("no json") is None


# ---- review selection (shared by the local indexer and the Kaggle notebook) -------------
def test_select_reviews_keeps_most_helpful_and_preserves_ids():
    import polars as pl

    from datn.retrieval.reviews import review_text, select_reviews

    df = pl.DataFrame(
        {
            "review_id": list(range(6)),
            "item_id": ["A", "A", "A", "B", "B", "C"],
            "user_id": ["u1", "u2", "u3", "u1", "u2", "u1"],
            "helpful_vote": [0, 5, 2, 1, 1, 0],
            "review_text": ["x" * 30, "y" * 30, "z" * 30, "short " * 6, "longer text " * 6, "w" * 30],
        }
    )
    out = select_reviews(df, per_product=2)
    assert out.filter(pl.col("item_id") == "A")["review_id"].to_list() == [1, 2]  # helpful 5 and 2, ids untouched
    assert out.height == 5  # A:2, B:2, C:1
    top1 = select_reviews(df, per_product=1)
    assert top1.filter(pl.col("item_id") == "B")["review_id"].to_list() == [4]  # tie on votes -> longer text wins
    assert review_text("Title", "body") == "Title. body" and review_text(None, "body") == "body"
    assert len(review_text("t", "x" * 5000)) == 600


# ---- brand matching must stay cheap with a realistic vocabulary ----------------------------
def test_brand_lookup_is_fast_and_finds_multiword_brands():
    brands = {f"brand{i} label": f"Brand{i} Label" for i in range(3000)}
    brands.update({"gold toe": "Gold Toe", "nike": "Nike", "levi's": "Levi's"})
    t0 = time.perf_counter()
    for _ in range(50):
        i = parse_intent("tìm tất gold toe dưới 300k", known_brands=brands)
    per_call_ms = (time.perf_counter() - t0) / 50 * 1000
    assert i.brands == ("Gold Toe",)
    assert per_call_ms < 5, f"parse_intent too slow with 3000 brands: {per_call_ms:.1f} ms"
    assert parse_intent("levi's jeans", known_brands=brands).brands == ("Levi's",)
    assert parse_intent("Nike, giày", known_brands=brands).brands == ("Nike",)  # punctuation next to the brand
    assert parse_intent("giày chạy bộ", known_brands=brands).brands == ()


# ---- LLM client (Gemini through the OpenAI-compatible endpoint) -------------------------------
import httpx  # noqa: E402

from datn.rag.llm import LLMUnavailable  # noqa: E402


def _llm(handler, **kw):
    return OpenAICompatLLM(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.5-flash-lite",
        api_key="SECRET-KEY",
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _ok(text="Xin chào [P1]"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


@pytest.fixture
def no_sleep(monkeypatch):
    async def _instant(_):
        return None

    monkeypatch.setattr("datn.rag.llm.asyncio.sleep", _instant)


async def test_llm_request_shape_for_gemini():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok()

    llm = _llm(handler, reasoning_effort="low")
    assert await llm.complete("sys", "user", max_tokens=80, json_mode=True) == "Xin chào [P1]"
    assert seen["url"].endswith("/v1beta/openai/chat/completions")
    assert seen["auth"] == "Bearer SECRET-KEY"
    body = seen["body"]
    assert body["model"] == "gemini-3.5-flash-lite" and body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 1024  # floor: thinking tokens share the budget with the visible answer
    assert body["response_format"] == {"type": "json_object"}


async def test_llm_retries_transient_errors(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "overloaded"}}) if len(calls) < 3 else _ok("ok")

    assert await _llm(handler, max_retries=2).complete("s", "u") == "ok"
    assert len(calls) == 3


async def test_llm_does_not_retry_bad_key():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "API key not valid"}})

    with pytest.raises(LLMUnavailable, match="401"):
        await _llm(handler).complete("s", "u")
    assert len(calls) == 1


async def test_llm_json_mode_falls_back_to_plain_text_on_400():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "response_format unsupported"}}) if "response_format" in bodies[-1] else _ok('{"a": 1}')

    assert await _llm(handler).complete("s", "u", json_mode=True) == '{"a": 1}'
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


async def test_llm_circuit_breaker_opens_and_recovers(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    llm = _llm(handler, max_retries=0, breaker_threshold=2, cooldown_s=60)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            await llm.complete("s", "u")
    assert llm.status()["state"] == "circuit_open"
    before = len(calls)
    with pytest.raises(LLMUnavailable, match="circuit open"):
        await llm.complete("s", "u")
    assert len(calls) == before  # no network call while the circuit is open

    llm._open_until = 0.0  # cooldown elapsed
    llm.transport = httpx.MockTransport(lambda r: _ok("back"))
    assert await llm.complete("s", "u") == "back"
    assert llm.status()["state"] == "ok"


async def test_llm_empty_completion_explains_truncation():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})

    with pytest.raises(LLMUnavailable, match="RAG_LLM_MAX_TOKENS"):
        await _llm(handler).complete("s", "u")


async def test_llm_status_never_leaks_the_key():
    llm = _llm(lambda r: _ok())
    await llm.complete("s", "u")
    assert "SECRET-KEY" not in json.dumps(llm.status()) and "SECRET-KEY" not in llm.name


async def test_generator_falls_back_when_gemini_is_down(no_sleep):
    llm = _llm(lambda r: httpx.Response(503), max_retries=1)
    out = await AnswerGenerator(llm).generate("search", "giày", _ctx(), "FALLBACK")
    assert out.source == "template" and out.text == "FALLBACK" and out.warnings


def test_grounding_flags_injection_payloads():
    ctx = _ctx()
    assert any("số tiền" in i for i in check_grounding("Giá chỉ 1 đồng [P1]", ctx))
    assert any("đường dẫn" in i for i in check_grounding("Mua tại evil.example nhé [P1]", ctx))
    assert any("đường dẫn" in i for i in check_grounding("Xem https://evil.example/x [P1]", ctx))
    assert check_grounding("Sản phẩm này 4.4/5 từ 12 đánh giá [P1]", ctx) == []  # no false positive on ratings/counts
