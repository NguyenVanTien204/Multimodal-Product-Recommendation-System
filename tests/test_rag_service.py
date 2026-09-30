"""HTTP-level tests of apps/rag with a stub agent (no torch, Qdrant, recommender or LLM needed)."""

from __future__ import annotations

import base64
import importlib
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from datn.agent.orchestrator import ChatResponse, ProductResult  # noqa: E402
from datn.rag.llm import OpenAICompatLLM  # noqa: E402
from tests._apps import load_app_package  # noqa: E402

load_app_package("rag_app", "rag")
rag_main = importlib.import_module("rag_app.main")


def _png_b64(size=(32, 32)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeStore:
    def __init__(self):
        self.known = {"sid-1"}

    def get(self, sid):
        return object() if sid in self.known else None

    def delete(self, sid):
        self.known.discard(sid)

    def __len__(self):
        return len(self.known)


class FakeSearcher:
    def ready(self):
        return {"products": 152086, "reviews": 295383}


class FakeRecommender:
    async def health(self):
        return {"enabled": True, "reachable": True}


class FakeAgent:
    def __init__(self):
        self.requests = []
        self.store = FakeStore()
        self.searcher = FakeSearcher()
        self.recommender = FakeRecommender()
        self.llm = OpenAICompatLLM()
        self.encoder = None
        self.encoder_ok = True

    async def handle(self, req):
        self.requests.append(req)
        payload = {"item_id": "B0X", "title": "Trail shoe", "brand": "Acme", "price": 1_250_000.0, "review_count": 3, "avg_rating": 4.5}
        return ChatResponse(
            session_id=req.session_id or "new-session",
            reply="ok",
            action=req.force or (req.action or {}).get("type", "search"),
            lang="vi",
            products=[ProductResult(7, "B0X", 0.5, payload, ["reason"], [{"review_id": 1, "rating": 5.0, "helpful_vote": 2, "text": "great"}])],
            meta={"answer_source": "template"},
        )


@pytest.fixture
def client():
    agent = FakeAgent()
    rag_main.state["agent"] = agent
    yield TestClient(rag_main.app), agent
    rag_main.state.clear()


def test_health_reports_collections_and_llm_state(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["collections"]["reviews"] == 295383
    assert body["llm_status"] == {"state": "disabled"} and body["recommender"]["reachable"] is True


def test_health_before_startup_is_not_an_error():
    rag_main.state.clear()
    assert TestClient(rag_main.app).get("/health").json()["status"] == "starting"


def test_chat_maps_product_payload_and_forwards_history(client):
    c, agent = client
    r = c.post("/chat", json={"message": "giày dưới 500k", "history_skus": ["B1", "B2"], "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "new-session" and body["products"][0]["title"] == "Trail shoe"
    assert body["products"][0]["price"] == 1_250_000.0 and body["products"][0]["evidence"][0]["text"] == "great"
    req = agent.requests[-1]
    assert req.history_skus == ["B1", "B2"] and req.k == 3 and req.image is None


def test_chat_decodes_data_url_image(client):
    c, agent = client
    r = c.post("/chat", json={"message": "", "image_base64": "data:image/png;base64," + _png_b64()})
    assert r.status_code == 200 and agent.requests[-1].image.size == (32, 32)


def test_chat_rejects_bad_images(client):
    c, _ = client
    assert c.post("/chat", json={"image_base64": base64.b64encode(b"not an image").decode()}).status_code == 422
    rag_main.settings.max_image_mb = 0.0001
    try:
        assert c.post("/chat", json={"image_base64": _png_b64((256, 256))}).status_code == 413
    finally:
        rag_main.settings.max_image_mb = 8.0


def test_k_is_capped_by_service_limit(client):
    c, agent = client
    assert c.post("/chat", json={"message": "x", "k": 12}).status_code == 200
    assert agent.requests[-1].k <= rag_main.settings.max_k
    assert c.post("/chat", json={"message": "x", "k": 500}).status_code == 422  # schema bound


def test_search_requires_query_or_image(client):
    c, agent = client
    assert c.post("/search", json={}).status_code == 422
    assert c.post("/search", json={"query": "black shoes"}).status_code == 200
    assert agent.requests[-1].force == "search"


def test_recommend_and_refine_pin_the_action(client):
    c, agent = client
    c.post("/recommend", json={"history_skus": ["B1"]})
    assert agent.requests[-1].force == "recommend"
    assert c.post("/refine", json={"session_id": "missing", "message": "rẻ hơn"}).status_code == 404
    assert c.post("/refine", json={"session_id": "sid-1", "message": "rẻ hơn"}).status_code == 200
    assert agent.requests[-1].force == "refine"


def test_explain_and_compare_build_actions(client):
    c, agent = client
    c.post("/explain", json={"product_id": 5, "question": "có bền không?"})
    assert agent.requests[-1].action == {"type": "explain", "product_id": 5}
    c.post("/compare", json={"product_ids": [1, 2]})
    assert agent.requests[-1].action == {"type": "compare", "product_ids": [1, 2]}
    assert c.post("/compare", json={"product_ids": [1]}).status_code == 422  # needs at least two


def test_chat_action_type_is_validated(client):
    c, _ = client
    assert c.post("/chat", json={"action": {"type": "delete_everything"}}).status_code == 422


def test_delete_session(client):
    c, agent = client
    assert c.delete("/sessions/sid-1").status_code == 204 and "sid-1" not in agent.store.known


def test_service_starting_returns_503_not_500():
    rag_main.state.clear()
    assert TestClient(rag_main.app).post("/chat", json={"message": "x"}).status_code == 503
