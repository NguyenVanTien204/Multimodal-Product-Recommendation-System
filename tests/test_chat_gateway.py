"""Tests of the marketplace `/chat` gateway (apps/backend/app/chat) with SQLite and a fake RAG service."""

from __future__ import annotations

import importlib
import time

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("passlib")
pytest.importorskip("jose")
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from tests._apps import load_app_package  # noqa: E402

load_app_package("backend_app", "backend")
main = importlib.import_module("backend_app.main")
database = importlib.import_module("backend_app.core.database")
config = importlib.import_module("backend_app.core.config")
chat_router = importlib.import_module("backend_app.chat.router")
catalog_models = importlib.import_module("backend_app.catalog.models")
for module in ("auth", "cart", "orders", "preferences"):  # register every table on Base.metadata
    importlib.import_module(f"backend_app.{module}.models")
auth_models = importlib.import_module("backend_app.auth.models")
pref_models = importlib.import_module("backend_app.preferences.models")
security = importlib.import_module("backend_app.core.security")


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_async_client(response=None, error=None, sink=None):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            if sink is not None:
                sink.append((url, json))
            if error:
                raise error
            return response

    return _Client


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    database.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[database.get_db] = override_db
    with Session() as db:
        cat = catalog_models.Category(name="Giày", slug="giay")
        db.add(cat)
        db.flush()
        for pid, sku, name, active in ((1, "B0A", "Trail shoe", True), (2, "B0B", "Road shoe", True), (3, "B0C", "Hidden shoe", False)):
            db.add(catalog_models.Product(id=pid, sku=sku, name=name, description="d", price=500000, stock_quantity=5, category_id=cat.id, is_active=active))
        db.commit()
    monkeypatch.setattr(config.settings, "datn_rag_url", "http://rag:8200")
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def rag_payload(ids):
    return {
        "session_id": "s-1", "reply": "Đây là gợi ý [P1]", "action": "search", "lang": "vi",
        "products": [
            {"product_id": i, "score": 0.5, "brand": "Acme", "price_estimated": i == 2, "avg_rating": 4.5, "review_count": 3,
             "reasons": ["khớp mô tả"], "evidence": [{"tag": "R1.1", "review_id": 9, "rating": 5.0, "helpful_vote": 2, "title": "t", "text": "great", "relevance": 0.7}]}
            for i in ids
        ],
        "filter_chips": ["Giá ≤ 500.000₫"], "suggestions": ["Rẻ hơn"], "citations": ["P1"], "meta": {"answer_source": "llm"}, "warnings": [],
    }


def test_hydrates_products_from_postgres_and_drops_inactive(env, monkeypatch):
    sent = []
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, rag_payload([1, 2, 3, 999])), sink=sent))
    r = env.post("/chat", json={"message": "giày trail"})
    assert r.status_code == 200
    body = r.json()
    assert [p["product"]["id"] for p in body["products"]] == [1, 2]  # 3 is inactive, 999 does not exist
    assert body["products"][0]["product"]["name"] == "Trail shoe" and body["products"][1]["price_estimated"] is True
    assert body["products"][0]["evidence"][0]["text"] == "great"
    assert body["session_id"] == "s-1" and body["meta"]["answer_source"] == "llm" and body["filter_chips"] == ["Giá ≤ 500.000₫"]
    url, payload = sent[0]
    assert url == "http://rag:8200/chat" and payload["message"] == "giày trail" and payload["history_skus"] == []


def test_forwards_action_and_filters(env, monkeypatch):
    sent = []
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, rag_payload([1])), sink=sent))
    env.post("/chat", json={"action": {"type": "explain", "product_id": 1}, "filters": {"max_price": 100000}})
    payload = sent[0][1]
    assert payload["action"] == {"type": "explain", "product_id": 1, "product_ids": []}  # RAG resolves product_id when the list is empty
    assert payload["filters"] == {"max_price": 100000.0, "brands": [], "audiences": [], "colours": []}


def test_rag_down_falls_back_to_keyword_search(env, monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(error=ConnectionError("refused")))
    r = env.post("/chat", json={"message": "trail shoe"})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["answer_source"] == "keyword_fallback" and "rag_unavailable" in body["warnings"]
    assert [p["product"]["name"] for p in body["products"]] == ["Trail shoe"]


def test_rag_error_status_also_falls_back(env, monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(500)))
    assert env.post("/chat", json={"message": "shoe"}).json()["meta"]["answer_source"] == "keyword_fallback"


def test_validation_errors_from_rag_are_passed_through(env, monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(422, {"detail": "image_base64 is not a readable image"})))
    r = env.post("/chat", json={"message": "x", "image_base64": "AAAA"})
    assert r.status_code == 422 and "readable image" in r.json()["detail"]


def test_empty_request_is_rejected(env):
    assert env.post("/chat", json={}).status_code == 422


def test_unconfigured_rag_uses_fallback(env, monkeypatch):
    monkeypatch.setattr(config.settings, "datn_rag_url", None)
    r = env.post("/chat", json={"message": "road shoe"})
    assert r.status_code == 200 and r.json()["meta"]["answer_source"] == "keyword_fallback"


def test_chat_health_when_unconfigured(env, monkeypatch):
    monkeypatch.setattr(config.settings, "datn_rag_url", None)
    assert env.get("/chat/health").json() == {"enabled": False}


# ---- taste memory ---------------------------------------------------------------------------
@pytest.fixture
def member(env):
    """(client, auth headers, db session) for a registered user; the db session is the one the app uses."""
    gen = main.app.dependency_overrides[database.get_db]()
    db = next(gen)
    user = auth_models.User(email="a@b.vn", password_hash="x", full_name="A")
    db.add(user)
    db.commit()
    yield env, {"Authorization": f"Bearer {security.create_access_token(user.id)}"}, db, user
    gen.close()


def _events(db, user):
    rows = db.query(pref_models.PreferenceEvent).filter_by(user_id=user.id).all()
    return sorted((e.product_id, e.kind, e.event_id) for e in rows)


def test_chat_sends_stored_events_and_persists_new_ones(member, monkeypatch):
    client, headers, db, user = member
    db.add(pref_models.PreferenceEvent(user_id=user.id, product_id=1, event_id="old", kind="like", ts=time.time() - 60, source="web"))
    db.commit()
    reply = {
        **rag_payload([1]),
        "events": [
            {"event_id": "e1", "sku": "B0B", "kind": "dislike", "ts": time.time(), "source": "chat"},
            {"event_id": "e2", "sku": "NOT-IN-SHOP", "kind": "like", "ts": time.time(), "source": "chat"},
        ],
    }
    sent = []
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, reply), sink=sent))

    assert client.post("/chat", json={"message": "giày"}, headers=headers).status_code == 200
    assert [(e["sku"], e["kind"], e["event_id"]) for e in sent[0][1]["events"]] == [("B0A", "like", "old")]
    assert _events(db, user) == [(1, "like", "old"), (2, "dislike", "e1")]  # unknown SKU dropped

    client.post("/chat", json={"message": "giày"}, headers=headers)  # the same relayed event is stored once
    assert _events(db, user) == [(1, "like", "old"), (2, "dislike", "e1")]
    assert {e["event_id"] for e in sent[1][1]["events"]} == {"old", "e1"}


def test_anonymous_chat_sends_no_events_and_stores_nothing(env, monkeypatch):
    sent = []
    reply = {**rag_payload([1]), "events": [{"event_id": "e1", "sku": "B0B", "kind": "dislike", "ts": time.time(), "source": "chat"}]}
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, reply), sink=sent))
    assert env.post("/chat", json={"message": "giày"}).status_code == 200
    assert sent[0][1]["events"] == []


def test_feedback_action_is_forwarded(env, monkeypatch):
    sent = []
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, rag_payload([1])), sink=sent))
    env.post("/chat", json={"action": {"type": "feedback", "kind": "dislike", "product_id": 2}})
    assert sent[0][1]["action"] == {"type": "feedback", "kind": "dislike", "product_id": 2, "product_ids": []}
    assert env.post("/chat", json={"action": {"type": "feedback", "kind": "meh", "product_id": 2}}).status_code == 422


def test_forget_action_deletes_stored_rows_before_forwarding(member, monkeypatch):
    client, headers, db, user = member
    for pid, eid in ((1, "a"), (2, "b")):
        db.add(pref_models.PreferenceEvent(user_id=user.id, product_id=pid, event_id=eid, kind="dislike", ts=time.time(), source="chat"))
    db.commit()
    sent = []
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", fake_async_client(FakeResponse(200, rag_payload([1])), sink=sent))

    client.post("/chat", json={"action": {"type": "forget", "product_id": 1}}, headers=headers)
    assert _events(db, user) == [(2, "dislike", "b")]
    assert [e["event_id"] for e in sent[0][1]["events"]] == ["b"]  # the forgotten row is not sent back to RAG

    client.post("/chat", json={"action": {"type": "forget"}}, headers=headers)
    assert _events(db, user) == [] and sent[1][1]["events"] == []


def test_preferences_endpoints_show_and_forget_what_is_remembered(member):
    client, headers, db, user = member
    assert client.post("/me/preferences/events", json={"product_id": 1, "kind": "like"}, headers=headers).status_code == 201
    assert client.post("/me/preferences/events", json={"product_id": 2, "kind": "dislike"}, headers=headers).status_code == 201
    body = client.get("/me/preferences", headers=headers).json()
    assert [p["id"] for p in body["liked"]] == [1] and [p["id"] for p in body["disliked"]] == [2] and body["event_count"] == 2

    client.post("/me/preferences/events", json={"product_id": 1, "kind": "dislike"}, headers=headers)  # newest verdict wins
    body = client.get("/me/preferences", headers=headers).json()
    assert body["liked"] == [] and sorted(p["id"] for p in body["disliked"]) == [1, 2]

    assert client.delete("/me/preferences/1", headers=headers).status_code == 204
    assert [p["id"] for p in client.get("/me/preferences", headers=headers).json()["disliked"]] == [2]
    assert client.delete("/me/preferences", headers=headers).status_code == 204
    assert _events(db, user) == []


def test_preferences_validation_and_auth(member):
    client, headers, _, _ = member
    assert client.post("/me/preferences/events", json={"product_id": 1, "kind": "purchase"}, headers=headers).status_code == 422  # orders feed the model, not this endpoint
    assert client.post("/me/preferences/events", json={"product_id": 3, "kind": "like"}, headers=headers).status_code == 404  # inactive product
    assert client.post("/me/preferences/events", json={"product_id": 999, "kind": "like"}, headers=headers).status_code == 404
    assert client.get("/me/preferences").status_code in (401, 403)
