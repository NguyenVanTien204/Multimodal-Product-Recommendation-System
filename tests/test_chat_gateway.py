"""Tests of the marketplace `/chat` gateway (apps/backend/app/chat) with SQLite and a fake RAG service."""

from __future__ import annotations

import importlib

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
for module in ("auth", "cart", "orders"):  # register every table on Base.metadata
    importlib.import_module(f"backend_app.{module}.models")


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
    assert payload["filters"] == {"max_price": 100000.0, "brands": []}


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
