"""Interaction memory: profile maths, feedback intents, orchestrator wiring (fakes, no Qdrant/torch/LLM), simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from datn.agent.intent import parse_intent
from datn.agent.orchestrator import ChatAgent, ChatRequest
from datn.agent.preferences import (
    NEGATIVE_WEIGHT,
    PreferenceEvent,
    PreferenceProfile,
    merge_history,
    neighbour_penalty,
)
from datn.agent.recommender_client import PersonalRanking, RecommenderClient
from datn.agent.session import SessionStore
from datn.rag.llm import OpenAICompatLLM
from datn.retrieval import schema as S
from datn.retrieval.filters import Hit

DAY = 86400.0
NOW = 1_800_000_000.0


def ev(sku, kind, days_ago=0.0, event_id=None):
    kwargs = {"event_id": event_id} if event_id else {}
    return PreferenceEvent(sku=sku, kind=kind, ts=NOW - days_ago * DAY, **kwargs)


# ---- profile --------------------------------------------------------------------------
def test_weights_and_thresholds():
    p = PreferenceProfile.from_events([ev("A", "like"), ev("B", "view"), ev("C", "click"), ev("D", "dislike")], now=NOW)
    assert p.positives() == ["A", "C"]  # a single view (0.1) is too weak to count, a click (0.3) is enough
    assert p.negatives() == ["D"]


def test_implicit_signals_accumulate():
    p = PreferenceProfile.from_events([ev("A", "view"), ev("A", "view"), ev("A", "view")], now=NOW)
    assert p.positives() == ["A"]


def test_old_events_decay():
    fresh = PreferenceProfile.from_events([ev("A", "like", 0)], now=NOW)
    old = PreferenceProfile.from_events([ev("A", "like", 90)], now=NOW)  # three half-lives: 0.7 / 8
    assert fresh.positives() == ["A"] and old.positives() == []


def test_latest_explicit_verdict_wins():
    changed_mind = PreferenceProfile.from_events([ev("A", "dislike", 5), ev("A", "like", 1)], now=NOW)
    assert changed_mind.positives() == ["A"] and changed_mind.negatives() == []
    regretted = PreferenceProfile.from_events([ev("A", "like", 5), ev("A", "dislike", 1)], now=NOW)
    assert regretted.negatives() == ["A"] and regretted.positives() == []
    bought_after_dislike = PreferenceProfile.from_events([ev("A", "dislike", 5), ev("A", "purchase", 1)], now=NOW)
    assert bought_after_dislike.positives() == ["A"]


def test_same_event_relayed_twice_counts_once():
    once = PreferenceProfile.from_events([ev("A", "click", event_id="e1")], now=NOW)
    twice = PreferenceProfile.from_events([ev("A", "click", event_id="e1"), ev("A", "click", event_id="e1")], now=NOW)
    assert twice.scores == once.scores
    assert PreferenceProfile.from_events([ev("A", "click"), ev("A", "click")], now=NOW).scores["A"] > once.scores["A"]  # distinct ids add up


def test_positives_are_oldest_first_and_negatives_newest_first():
    p = PreferenceProfile.from_events(
        [ev("B", "like", 1), ev("A", "like", 3), ev("X", "dislike", 4), ev("Y", "dislike", 2), ev("Z", "dislike", 3)], now=NOW
    )
    assert p.positives() == ["A", "B"]
    assert p.negatives() == ["Y", "Z", "X"]
    assert p.negatives(limit=1) == ["Y"] and p.positives(limit=1) == ["B"]


def test_unknown_kinds_and_empty_skus_are_ignored():
    p = PreferenceProfile.from_events([ev("A", "shrug"), PreferenceEvent(sku="", kind="like")], now=NOW)
    assert p.scores == {}


def test_event_roundtrip():
    e = ev("A", "like")
    assert PreferenceEvent.from_dict(e.to_dict()) == e


def test_merge_history_orders_dedupes_and_drops_negatives():
    assert merge_history(["S1", "S2"], ["S2", "S3"], ["S1"], limit=10) == ["S2", "S3"]
    assert merge_history(["S1", "S2"], ["S1"], [], limit=10) == ["S2", "S1"]  # latest occurrence wins
    assert merge_history(["S1", "S2", "S3"], [], [], limit=2) == ["S2", "S3"]


def test_neighbour_penalty_uses_strongest_neighbourhood():
    near = {7: 1, 8: 30}
    far = {7: 20}
    assert neighbour_penalty([near, far], 7) == pytest.approx(NEGATIVE_WEIGHT / 61)
    assert neighbour_penalty([near], 8) == pytest.approx(NEGATIVE_WEIGHT / 90)
    assert neighbour_penalty([near], 99) == 0.0 and neighbour_penalty([], 7) == 0.0


# ---- intents -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "msg,kind",
    [
        ("không thích sản phẩm 2", "dislike"),
        ("mình ko thích cái 3", "dislike"),
        ("cái 4 xấu quá", "dislike"),
        ("I don't like #3", "dislike"),
        ("not a fan of the first one", "dislike"),
        ("thích cái 1", "like"),
        ("mình thích sản phẩm số 2", "like"),
        ("I love #2", "like"),
        ("không thích cái áo số 2", "dislike"),  # a product noun does not turn it into a new search
    ],
)
def test_feedback_intents(msg, kind):
    i = parse_intent(msg, has_results=True)
    assert i.action == "feedback" and i.feedback == kind


def test_feedback_needs_a_reference_to_a_shown_result():
    assert parse_intent("không thích màu đỏ", has_results=True).action != "feedback"  # catalogue preference, not item feedback
    assert parse_intent("thích cái 1", has_results=False).action != "feedback"
    assert parse_intent("tôi có thể thích gì", has_results=True).action != "feedback"


def test_existing_routing_beats_feedback():
    assert parse_intent("giống cái đầu tiên", has_results=True).action == "similar"
    liked_similar = parse_intent("thích cái 1, tìm cái giống", has_results=True)
    assert liked_similar.action == "similar" and liked_similar.feedback == "like"
    assert parse_intent("so sánh 1 và 3", has_results=True).feedback is None
    assert parse_intent("tại sao sản phẩm 2", has_results=True).action == "explain"


# ---- session -----------------------------------------------------------------------------
def test_session_records_events_and_skips_back_to_back_duplicates():
    s = SessionStore().get_or_create(None)
    s.engage("A")
    s.engage("A")
    s.engage("B")
    s.record("B", "dislike")
    assert [(e.sku, e.kind) for e in s.events] == [("A", "click"), ("B", "click"), ("B", "dislike")]
    assert [e.sku for e in s.pending_events] == ["A", "B", "B"]
    assert s.record("", "like") is None


# ---- orchestrator wiring (fakes) -----------------------------------------------------------
def _payload(i):
    return {
        S.P_PRODUCT_ID: i, S.P_ITEM_ID: f"S{i}", S.P_TITLE: f"Item {i}", S.P_BRAND: "Acme", S.P_CATEGORY: "Tops",
        S.P_PRICE: 100_000.0 + i, S.P_PRICE_ESTIMATED: False, S.P_AVG_RATING: 4.0, S.P_REVIEW_COUNT: 0,
        S.P_DESCRIPTION: "d", S.P_FEATURES: "f",
    }


class FakeSearcher:
    """10 products. Lexical scores are nearly tied so a soft penalty visibly reorders; 3 and 4 are the look-alikes of 2."""

    def __init__(self):
        self.payloads = {i: _payload(i) for i in range(1, 11)}
        self.neighbours = {2: [3, 4]}
        self.similar_calls = []

    def _hits(self, filters, k):
        return [Hit(i, p, 1.0 - 0.0001 * i) for i, p in self.payloads.items() if filters is None or filters.matches(p, i)][:k]

    def lexical_search(self, text, filters=None, k=10):
        return self._hits(filters, k)

    def popular(self, filters=None, k=10):
        return self._hits(filters, k)

    def products_by_sku(self, skus):
        return {p[S.P_ITEM_ID]: Hit(i, p) for i, p in self.payloads.items() if p[S.P_ITEM_ID] in set(skus)}

    def similar(self, ids, filters=None, k=10, pool=100):
        self.similar_calls.append(list(ids))
        return [Hit(i, self.payloads[i]) for i in self.neighbours.get(ids[0], [])][:k]

    def get_products(self, ids):
        return {int(i): self.payloads[int(i)] for i in ids if int(i) in self.payloads}


class FakeReviews:
    def for_products(self, ids, qvec, n):
        return {}

    def critical(self, pid, n):
        return []


class FakeRecommender(RecommenderClient):
    def __init__(self):
        super().__init__(base_url="http://fake", max_k=50)
        self.histories = []

    async def recommend(self, history_skus, k=50):
        self.histories.append(list(history_skus))
        skus = [f"S{i}" for i in range(1, 11)]
        return PersonalRanking(skus=skus, scores=[1.0 - 0.01 * i for i in range(10)], source="reranker", model_version="fake")


@pytest.fixture
def agent():
    searcher, rec = FakeSearcher(), FakeRecommender()
    a = ChatAgent(searcher, None, FakeReviews(), rec, OpenAICompatLLM())
    a.fake_searcher, a.fake_rec = searcher, rec
    return a


def _ids(resp):
    return [p.product_id for p in resp.products]


async def _start(agent):
    first = await agent.handle(ChatRequest(message="áo thun", k=5))
    assert _ids(first) == [1, 2, 3, 4, 5] and first.events == []
    return first.session_id


async def test_dislike_records_event_excludes_item_and_penalises_lookalikes(agent):
    sid = await _start(agent)
    resp = await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid, k=5))
    assert resp.action == "feedback" and resp.reply.startswith("Đã ghi nhận")
    assert [(e["sku"], e["kind"]) for e in resp.events] == [("S2", "dislike")]
    assert 2 not in _ids(resp)
    assert 3 not in _ids(resp) and 4 not in _ids(resp)  # look-alikes sink below the equally-scored alternatives
    assert _ids(resp) == [1, 5, 6, 7, 8]
    assert resp.meta["preferences"]["disliked"] == 1 and resp.meta["preferences"]["penalised"] == 2
    assert resp.meta["feedback"] == {"kind": "dislike", "product_ids": [2]}


async def test_dislike_carries_over_to_recommendations(agent):
    sid = await _start(agent)
    await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid))
    rec = await agent.handle(ChatRequest(message="gợi ý cho tôi", session_id=sid, k=5))
    assert rec.action == "recommend" and _ids(rec) == [1, 5, 6, 7, 8]
    assert agent.fake_rec.histories[-1] == []  # a dislike never reaches the tower history


async def test_like_feeds_the_tower_history_and_is_not_shown_again(agent):
    sid = await _start(agent)
    resp = await agent.handle(ChatRequest(message="thích cái 1", session_id=sid))
    assert resp.action == "feedback" and [(e["sku"], e["kind"]) for e in resp.events] == [("S1", "like")]
    rec = await agent.handle(ChatRequest(message="gợi ý cho tôi", session_id=sid, k=5))
    assert agent.fake_rec.histories[-1] == ["S1"]
    assert 1 not in _ids(rec)  # recommend hides what is already in the history
    assert rec.events == []  # events are only returned on the turn they were recorded


async def test_ui_feedback_action_with_explicit_product_id(agent):
    sid = await _start(agent)
    resp = await agent.handle(ChatRequest(action={"type": "feedback", "kind": "dislike", "product_id": 5}, session_id=sid, k=5))
    assert [(e["sku"], e["kind"]) for e in resp.events] == [("S5", "dislike")] and 5 not in _ids(resp)


async def test_durable_events_from_the_shop_apply_to_a_fresh_session(agent):
    req = ChatRequest(message="áo thun", k=5, events=[PreferenceEvent("S1", "dislike"), PreferenceEvent("S6", "like")])
    resp = await agent.handle(req)
    assert 1 not in _ids(resp)
    assert resp.events == []  # nothing new was said this turn
    assert agent.fake_rec.histories[-1] == ["S6"]


async def test_dislike_in_session_and_same_event_from_shop_counts_once(agent):
    sid = await _start(agent)
    first = await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid))
    echoed = PreferenceEvent.from_dict(first.events[0])
    resp = await agent.handle(ChatRequest(message="áo thun", session_id=sid, events=[echoed], k=5))
    assert resp.meta["preferences"]["disliked"] == 1


async def test_ambiguous_reference_is_not_guessed(agent):
    sid = await _start(agent)
    resp = await agent.handle(ChatRequest(message="không thích sản phẩm 9", session_id=sid))
    assert resp.events == [] and "sản phẩm nào" in resp.reply


async def test_liked_item_next_to_a_disliked_one_is_not_demoted(agent):
    sid = await _start(agent)
    await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid))  # look-alikes 3 and 4 get penalised
    await agent.handle(ChatRequest(action={"type": "feedback", "kind": "like", "product_id": 3}, session_id=sid))
    resp = await agent.handle(ChatRequest(message="áo thun", session_id=sid, k=5))
    assert 3 in _ids(resp) and 4 not in _ids(resp) and 2 not in _ids(resp)


async def test_forget_drops_session_feedback_and_lifts_the_exclusion(agent):
    sid = await _start(agent)
    await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid, k=5))
    resp = await agent.handle(ChatRequest(action={"type": "forget", "product_id": 2}, session_id=sid))
    assert resp.action == "forget" and resp.meta["forgot"] == [2]
    again = await agent.handle(ChatRequest(message="áo thun", session_id=sid, k=5))
    assert _ids(again) == [1, 2, 3, 4, 5] and "preferences" not in again.meta  # item 2 and its look-alikes are back


async def test_forget_all_clears_every_event(agent):
    sid = await _start(agent)
    await agent.handle(ChatRequest(message="không thích sản phẩm 2", session_id=sid))
    await agent.handle(ChatRequest(message="thích cái 1", session_id=sid))
    resp = await agent.handle(ChatRequest(action={"type": "forget"}, session_id=sid))
    assert resp.meta["forgot"] == "all"
    rec = await agent.handle(ChatRequest(message="gợi ý cho tôi", session_id=sid, k=5))
    assert agent.fake_rec.histories[-1] == []  # the like is gone from the tower history
    assert _ids(rec) == [1, 2, 3, 4, 5]  # and the disliked item 2 (and its look-alikes) are back


async def test_like_while_asking_for_similar_is_recorded_as_like(agent):
    sid = await _start(agent)
    resp = await agent.handle(ChatRequest(message="thích cái 1, tìm cái giống", session_id=sid, k=5))
    assert resp.action == "similar" and [(e["sku"], e["kind"]) for e in resp.events] == [("S1", "like")]


# ---- simulation engine ------------------------------------------------------------------------
from datn.recommenders.hm.feedback_sim import (  # noqa: E402
    SimUser,
    paired_bootstrap,
    rerank_with_penalty,
    simulate,
)

N_ITEMS = 100


def _line_rank(histories, excludes):
    """Toy recommender on a number line: closest items to the mean of the history first."""
    out = []
    for hist, ex in zip(histories, excludes):
        centre = float(np.mean(hist)) if hist else 50.0
        out.append([i for i in sorted(range(1, N_ITEMS + 1), key=lambda i: (abs(i - centre), i)) if i not in ex])
    return out


def _line_neighbours(item):
    near = sorted((i for i in range(1, N_ITEMS + 1) if i != item), key=lambda i: (abs(i - item), i))[:30]
    return {i: r for r, i in enumerate(near, start=1)}


def _users(n=20):
    # the user's taste sits above the history (items 52..71 vs. a history at 50): liked items pull the ranking towards it
    return [SimUser(context=(50,), truth=frozenset(range(52, 72))) for _ in range(n)]


def test_rerank_with_penalty_sinks_neighbours_only():
    ranked = list(range(1, 21))
    out = rerank_with_penalty(ranked, [{1: 1, 2: 2}], k=5)
    assert out == [3, 4, 5, 6, 7]
    assert rerank_with_penalty(ranked, [], k=5) == [1, 2, 3, 4, 5]


def test_simulation_round_one_is_identical_across_arms_and_feedback_helps():
    res = simulate(_users(), _line_rank, _line_neighbours, rounds=4, k=6, p_like=1.0, p_dislike=0.5, seed=1)
    first = {arm: res.recall[arm][:, 0] for arm in res.recall}
    assert all(np.array_equal(first["none"], v) for v in first.values())  # before any feedback every arm shows the same list
    assert res.recall["like"][:, -1].mean() > res.recall["none"][:, -1].mean()
    assert (res.recall["none"][:, -1] >= res.recall["none"][:, 0]).all()  # cumulative
    assert res.shown_dislikes["none"][:, -1].sum() > 0


def test_simulation_is_deterministic_and_paired():
    a = simulate(_users(5), _line_rank, _line_neighbours, rounds=3, k=6, seed=7)
    b = simulate(_users(5), _line_rank, _line_neighbours, rounds=3, k=6, seed=7)
    assert all(np.array_equal(a.recall[arm], b.recall[arm]) for arm in a.recall)


def test_paired_bootstrap_ci_brackets_the_mean_and_handles_empty():
    a, b = np.zeros(200), np.full(200, 0.1) + np.random.default_rng(0).normal(0, 0.05, 200)
    mean, lo, hi = paired_bootstrap(a, b)
    assert lo <= mean <= hi and lo > 0
    assert paired_bootstrap(np.array([]), np.array([])) == (0.0, 0.0, 0.0)
