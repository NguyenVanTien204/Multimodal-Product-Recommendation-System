from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import statistics
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..rag.answer import Answer, AnswerGenerator, template_compare, template_explain, template_search
from ..rag.context import EvidenceContext
from ..rag.evidence import Review, ReviewRetriever
from ..rag.llm import LLM, LLMUnavailable, parse_json_object
from ..retrieval import schema as S
from ..retrieval.encoder import JinaClipEncoder
from ..retrieval.filters import Hit, SearchFilters
from ..retrieval.search import RRF_K, HybridSearcher
from .intent import Intent, detect_lang, parse_intent
from .recommender_client import PersonalRanking, RecommenderClient
from .session import Preferences, Session, SessionStore, ShownProduct

log = logging.getLogger(__name__)

# Per-request stage timings (ms), surfaced as `meta.timings_ms` so slow stages are visible in production.
_TIMINGS: ContextVar["dict[str, float] | None"] = ContextVar("rag_timings", default=None)


def _mark(name: str, t0: float) -> None:
    timings = _TIMINGS.get()
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000


POOL = 60  # candidates kept per vector search (and reused by refinements)
PERSONAL_WEIGHT = 0.6  # weight of the model's rank vs. the semantic rank in the fused score
MAX_HISTORY = 30

HELP_VI = (
    "Mình là trợ lý mua sắm thời trang. Mình có thể:\n"
    "• Tìm sản phẩm theo mô tả hoặc **ảnh bạn tải lên** (vd: “giày chạy bộ nam màu đen dưới 500k”)\n"
    "• Lọc/tinh chỉnh ngay trong hội thoại: “rẻ hơn”, “màu đỏ”, “thương hiệu Nike”, “cái khác”\n"
    "• Gợi ý cá nhân hoá theo lịch sử mua/xem của bạn (“gợi ý cho tôi”)\n"
    "• Giải thích vì sao gợi ý một sản phẩm và dẫn chứng từ đánh giá người mua (“tại sao sản phẩm 2?”)\n"
    "• So sánh sản phẩm (“so sánh 1 và 3”), tìm sản phẩm tương tự (“giống cái đầu tiên”)"
)
HELP_EN = (
    "I'm a fashion shopping assistant. I can search by text or an uploaded photo, refine results (“cheaper”, “red”, “Nike”), "
    "recommend based on your history, explain a pick using buyer reviews (“why #2?”), compare products (“compare 1 and 3”) "
    "and find similar items (“like the first one”)."
)


@dataclass
class ChatRequest:
    message: str = ""
    session_id: str | None = None
    image: Any | None = None  # PIL.Image
    history_skus: list[str] = field(default_factory=list)  # from the shop: cart/orders, oldest first
    action: dict[str, Any] | None = None  # UI button, e.g. {"type": "explain", "product_id": 12}
    filters: SearchFilters | None = None  # explicit UI filters (merged with parsed ones)
    force: str | None = None  # API endpoints pin the action (search|refine|recommend|...) instead of inferring it
    k: int = 5


@dataclass
class ProductResult:
    product_id: int
    sku: str
    score: float
    payload: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    session_id: str
    reply: str
    action: str
    lang: str
    products: list[ProductResult] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    filter_chips: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _digest(image: Any | None) -> str | None:
    if image is None:
        return None
    return hashlib.md5(image.convert("RGB").resize((32, 32)).tobytes()).hexdigest()


def _is_stricter(new: SearchFilters, old: SearchFilters) -> bool:
    """True when every product allowed by `new` was already allowed by `old`, i.e.
    a pool retrieved under `old` still contains everything `new` could return."""
    if old.min_price is not None and (new.min_price is None or new.min_price < old.min_price):
        return False
    if old.max_price is not None and (new.max_price is None or new.max_price > old.max_price):
        return False
    if old.brands and not (new.brands and {b.lower() for b in new.brands} <= {b.lower() for b in old.brands}):
        return False
    if old.category_slugs and not (new.category_slugs and set(new.category_slugs) <= set(old.category_slugs)):
        return False
    if old.min_rating is not None and (new.min_rating is None or new.min_rating < old.min_rating):
        return False
    return True


class ChatAgent:
    """Conversational multimodal recommender.

    Ties together: intent parsing -> session preferences -> hybrid multimodal
    retrieval (text and/or image query over Jina CLIP v2 vectors in Qdrant, with
    payload filters) -> optional personalization from the checkpointed User Tower
    + Reranker service -> review evidence -> grounded answer (LLM, verified) or
    deterministic template. Every stage degrades independently: no LLM -> template
    text; no recommender -> content-based similarity; no encoder -> keyword search.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        encoder: JinaClipEncoder | None,
        reviews: ReviewRetriever,
        recommender: RecommenderClient,
        llm: LLM,
        store: SessionStore | None = None,
    ) -> None:
        self.searcher = searcher
        self.encoder = encoder
        self.reviews = reviews
        self.recommender = recommender
        self.llm = llm
        self.generator = AnswerGenerator(llm)
        self.store = store or SessionStore()
        self.known_brands: dict[str, str] = {}
        self.encoder_ok = encoder is not None

    # ---- startup -----------------------------------------------------------------
    def warm_up(self) -> None:
        """Load brand vocabulary and (optionally) the encoder. Safe to call repeatedly."""
        brands = self.searcher.top_brands()
        from .intent import fold

        self.known_brands = {fold(b): b for b in brands if len(b) >= 3}
        if self.encoder is not None and not self.encoder.loaded:
            try:
                self.encoder.load()
                self.encoder_ok = True
            except Exception as exc:  # noqa: BLE001
                log.error("encoder failed to load, falling back to keyword search: %s", exc)
                self.encoder_ok = False

    # ---- public entry point --------------------------------------------------------
    async def handle(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        token = _TIMINGS.set({})
        session = self.store.get_or_create(req.session_id)
        lang = detect_lang(req.message, default=session.turns and detect_lang(session.turns[0]["content"]) or "vi") if req.message else "vi"

        t_intent = time.perf_counter()
        intent = self._intent_from_request(req, session, lang)
        if intent.action == "unknown":
            intent = await self._llm_intent(req.message, intent, session)

        _mark("intent_ms", t_intent)
        session.add_turn("user", req.message or f"[{intent.action}]")
        handler = {
            "search": self._do_search,
            "refine": self._do_search,
            "recommend": self._do_recommend,
            "similar": self._do_similar,
            "explain": self._do_explain,
            "compare": self._do_compare,
        }.get(intent.action)

        if intent.action == "reset":
            session.prefs = Preferences()
            session.candidate_pool, session.pool_key = [], None
            resp = self._plain(session, "reset", intent.lang, "Đã xóa các bộ lọc. Bạn muốn tìm gì tiếp theo?" if intent.lang == "vi" else "Filters cleared. What next?")
        elif handler is None:
            text = HELP_EN if intent.lang == "en" else HELP_VI
            if intent.action == "greet":
                text = ("Xin chào! " if intent.lang == "vi" else "Hello! ") + text
            if intent.action == "unknown":
                text = ("Mình chưa hiểu rõ ý bạn. " if intent.lang == "vi" else "I didn't quite get that. ") + text
            resp = self._plain(session, intent.action, intent.lang, text)
        else:
            try:
                t_handler = time.perf_counter()
                resp = await handler(session, intent, req)
                _mark("handler_ms", t_handler)
            except Exception:  # noqa: BLE001 - never leak a stack trace into chat
                log.exception("handler %s failed", intent.action)
                resp = self._plain(
                    session,
                    intent.action,
                    intent.lang,
                    "Xin lỗi, hệ thống gặp sự cố khi xử lý yêu cầu này. Bạn thử lại hoặc diễn đạt khác nhé." if intent.lang == "vi" else "Sorry, something went wrong. Please try again.",
                )
                resp.warnings.append("internal_error")

        session.add_turn("assistant", resp.reply)
        resp.meta["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        resp.meta["timings_ms"] = {k: round(v) for k, v in (_TIMINGS.get() or {}).items()}
        _TIMINGS.reset(token)
        resp.meta.setdefault("llm", self.llm.name if self.llm.enabled else "disabled")
        resp.meta["encoder"] = "jina-clip-v2" if self.encoder_ok else "unavailable(keyword-fallback)"
        return resp

    # ---- intent resolution ---------------------------------------------------------
    def _intent_from_request(self, req: ChatRequest, session: Session, lang: str) -> Intent:
        act = req.action or {}
        if act.get("type") in {"explain", "compare", "similar", "recommend"}:
            ids = [int(i) for i in act.get("product_ids") or ([act["product_id"]] if act.get("product_id") else [])]
            intent = Intent(action=act["type"], lang=lang, query=req.message, display_query=req.message)
            intent.ordinals = tuple(
                i + 1 for pid in ids for i, p in enumerate(session.last_results) if p.product_id == pid
            )
            intent.explicit_product_ids = tuple(ids)
            return intent
        intent = parse_intent(
            req.message,
            has_image=req.image is not None,
            has_results=bool(session.last_results),
            known_brands=self.known_brands,
            default_lang=lang,
        )
        if req.force:
            intent.action = req.force
        return intent

    async def _llm_intent(self, message: str, fallback: Intent, session: Session) -> Intent:
        if not self.llm.enabled or not message.strip():
            return fallback
        system = (
            "Phân loại ý định của khách mua hàng. Trả về JSON: "
            '{"action": "search|recommend|greet|help|none", "query": "<mô tả sản phẩm cần tìm hoặc rỗng>"}. Không giải thích.'
        )
        try:
            obj = parse_json_object(await self.llm.complete(system, message, temperature=0, max_tokens=80, json_mode=True))
        except LLMUnavailable:
            return fallback
        if not obj or obj.get("action") not in {"search", "recommend", "greet", "help"}:
            return fallback
        fallback.action = obj["action"]
        if obj["action"] == "search" and obj.get("query"):
            fallback.query = fallback.display_query = str(obj["query"])[:200]
        return fallback

    # ---- timed helpers -----------------------------------------------------------------
    async def _personal(self, history: list[str]) -> PersonalRanking | None:
        t = time.perf_counter()
        result = await self.recommender.recommend(history, k=self.recommender.max_k)
        _mark("personalization_ms", t)
        return result

    async def _timed_thread(self, name: str, fn, *args):
        t = time.perf_counter()
        result = await asyncio.to_thread(fn, *args)
        _mark(name, t)
        return result

    async def _generate(self, *args, **kwargs) -> Answer:
        t = time.perf_counter()
        result = await self.generator.generate(*args, **kwargs)
        _mark("answer_ms", t)
        return result

    # ---- helpers ---------------------------------------------------------------------
    def _plain(self, session: Session, action: str, lang: str, text: str) -> ChatResponse:
        return ChatResponse(
            session_id=session.session_id,
            reply=text,
            action=action,
            lang=lang,
            filters=session.prefs.filters.to_dict(),
            filter_chips=session.prefs.summary()["filter_chips"],
            suggestions=self._suggestions(session, lang),
        )

    def _suggestions(self, session: Session, lang: str) -> list[str]:
        if not session.last_results:
            return (
                ["Giày chạy bộ nam màu đen dưới 500k", "Áo sơ mi trắng công sở", "Túi xách nữ da thật", "Gợi ý cho tôi"]
                if lang == "vi"
                else ["Black casual shoes under $100", "White office shirt", "Women's leather handbag", "Recommend for me"]
            )
        n = len(session.last_results)
        base = ["Rẻ hơn", "Cái khác", "Tại sao sản phẩm 1?"]
        if n >= 2:
            base.append("So sánh 1 và 2")
        return base if lang == "vi" else ["Cheaper", "Show others", "Why #1?"] + (["Compare 1 and 2"] if n >= 2 else [])

    def _filters_from(self, intent: Intent, req: ChatRequest, base: SearchFilters | None = None) -> SearchFilters:
        f = base or SearchFilters()
        updates: dict[str, Any] = {}
        if intent.min_price is not None:
            updates["min_price"] = intent.min_price
        if intent.max_price is not None:
            updates["max_price"] = intent.max_price
        if intent.brands:
            updates["brands"] = tuple(intent.brands)
        if intent.min_rating is not None:
            updates["min_rating"] = intent.min_rating
        f = f.with_(**updates)
        if req.filters:
            explicit = {k: v for k, v in req.filters.to_dict().items() if v not in (None, [], ())}
            f = SearchFilters.from_dict({**f.to_dict(), **explicit})
        return f

    def _history(self, session: Session, req: ChatRequest) -> list[str]:
        merged: list[str] = []
        for sku in [*req.history_skus, *session.focus_history_skus]:
            if sku in merged:
                merged.remove(sku)
            merged.append(sku)
        return merged[-MAX_HISTORY:]

    def _relative_price(self, session: Session, intent: Intent) -> float | None:
        pool = session.resolve(list(intent.ordinals)) or session.last_results
        prices = [p.payload.get(S.P_PRICE) for p in pool if p.payload.get(S.P_PRICE)]
        return statistics.median(prices) if prices else None

    async def _query_vectors(self, text: str | None, image: Any | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.encoder_ok or self.encoder is None:
            return None, None
        tv = iv = None
        t_enc = time.perf_counter()
        try:
            if text:
                tv = (await asyncio.to_thread(self.encoder.encode_query, [text]))[0]
            if image is not None:
                iv = (await asyncio.to_thread(self.encoder.encode_images, [image]))[0]
        except Exception as exc:  # noqa: BLE001
            log.error("encoding failed: %s", exc)
            self.encoder_ok = False
        _mark("encode_ms", t_enc)
        return tv, iv

    def _evidence_dicts(self, revs: list[Review], tags: list[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "tag": tags[i] if tags else None,
                "review_id": r.review_id,
                "rating": r.rating,
                "helpful_vote": r.helpful_vote,
                "title": r.title,
                "text": r.snippet(360),
                "relevance": round(r.score, 3),
            }
            for i, r in enumerate(revs)
        ]

    def _to_result(self, hit: Hit, session_reasons: list[str] | None = None) -> ProductResult:
        return ProductResult(
            product_id=hit.product_id,
            sku=hit.sku,
            score=round(hit.score, 5),
            payload=hit.payload,
            reasons=list(session_reasons or []),
            signals={**{k: round(v, 4) for k, v in hit.scores.items()}, **{f"rank[{k}]": v for k, v in hit.ranks.items()}},
        )

    def _finish(
        self,
        session: Session,
        intent: Intent,
        hits: list[Hit],
        results: list[ProductResult],
        answer: Answer,
        ctx: EvidenceContext,
        meta: dict[str, Any],
    ) -> ChatResponse:
        session.last_results = [ShownProduct(r.product_id, r.payload, r.reasons) for r in results]
        session.seen_product_ids = (session.seen_product_ids + [r.product_id for r in results])[-200:]
        # tag <-> product mapping for the UI ([P1] == results[0])
        return ChatResponse(
            session_id=session.session_id,
            reply=answer.text,
            action=intent.action,
            lang=intent.lang,
            products=results,
            filters=session.prefs.filters.to_dict(),
            filter_chips=session.prefs.summary()["filter_chips"],
            suggestions=self._suggestions(session, intent.lang),
            citations=answer.citations,
            meta={"answer_source": answer.source, **meta},
            warnings=answer.warnings,
        )

    # ---- search / refine -----------------------------------------------------------
    async def _do_search(self, session: Session, intent: Intent, req: ChatRequest) -> ChatResponse:
        prefs = session.prefs
        refining = intent.action == "refine"

        if refining:
            filters = self._filters_from(intent, req, prefs.filters)
            if intent.price_mode:
                ref = self._relative_price(session, intent)
                if ref:
                    if intent.price_mode == "cheaper":
                        filters = filters.with_(max_price=ref * 0.8, min_price=filters.min_price if (filters.min_price or 0) < ref * 0.8 else None)
                    else:
                        filters = filters.with_(min_price=ref * 1.2, max_price=None)
            colors = {c.lower() for c in intent.colors}
            if intent.colors:
                for old in prefs.colors:  # a new colour replaces the previous one instead of piling up
                    prefs.query = re.sub(rf"\b{re.escape(old)}\b", " ", prefs.query, flags=re.IGNORECASE)
                prefs.query = " ".join(prefs.query.split())
                prefs.colors = intent.colors
            # leftover words after removing constraints extend the semantic query ("màu đỏ vải cotton")
            extra = " ".join(w for w in intent.query.split() if w.lower() not in colors)
            if extra:
                prefs.query = f"{prefs.query} {extra}".strip()
                prefs.display_query = f"{prefs.display_query} {extra}".strip()
            for c in prefs.colors:
                if c.lower() not in prefs.query.lower():
                    prefs.query = f"{c} {prefs.query}".strip()
        else:
            filters = self._filters_from(intent, req)
            prefs.query, prefs.display_query, prefs.colors = intent.query, intent.display_query, intent.colors
        if intent.clear_filters:
            filters = SearchFilters()
        exclude = tuple(session.seen_product_ids) if intent.exclude_shown else filters.exclude_product_ids
        filters = filters.with_(exclude_product_ids=tuple(exclude))
        prefs.filters = filters

        text = prefs.query.strip() or None
        image = req.image
        pool_key = (text, _digest(image))
        meta: dict[str, Any] = {"retrieval": "hybrid"}

        # Reuse the candidate pool when only stricter filters were added (no new vector query).
        pool_hits: list[Hit] | None = None
        if (
            refining
            and session.candidate_pool
            and session.pool_key == pool_key
            and _is_stricter(filters.without_exclusions(), session.pool_filters)
        ):
            kept = [
                Hit(p.product_id, p.payload, score=p.payload.get("_score", 0.0))
                for p in session.candidate_pool
                if filters.matches(p.payload, p.product_id)
            ]
            if len(kept) >= req.k:
                pool_hits = kept
                meta["retrieval"] = "cached_pool_refine"

        text_vec = img_vec = None
        if pool_hits is None:
            text_vec, img_vec = await self._query_vectors(text, image)
            t_ret = time.perf_counter()
            if text_vec is None and img_vec is None:
                if text:
                    pool_hits = await asyncio.to_thread(self.searcher.lexical_search, text, filters, POOL)
                    meta["retrieval"] = "lexical_fallback"
                else:
                    pool_hits = await asyncio.to_thread(self.searcher.popular, filters, POOL)
                    meta["retrieval"] = "popularity"
            else:
                pool_hits = await asyncio.to_thread(self.searcher.search, text_vec, img_vec, filters, POOL, 100)
                meta["query_modalities"] = [m for m, v in (("text", text_vec), ("image", img_vec)) if v is not None]
            session.candidate_pool = [ShownProduct(h.product_id, {**h.payload, "_score": h.score}) for h in pool_hits]
            session.pool_key, session.pool_filters = pool_key, filters.without_exclusions()
            _mark("retrieval_ms", t_ret)

        # Personalization: fuse the semantic rank with the checkpointed model's rank.
        history = self._history(session, req)
        ranking: PersonalRanking | None = None
        if history:
            ranking = await self._personal(history)
        personalized = False
        if ranking and ranking.source == "reranker":
            rank_of = ranking.rank_of()
            for h in pool_hits:
                r = rank_of.get(h.sku)
                if r is not None:
                    h.score += PERSONAL_WEIGHT / (RRF_K + r)
                    h.ranks["personal"] = r
                    personalized = True
            pool_hits = sorted(pool_hits, key=lambda h: h.score, reverse=True)
            meta["personalization"] = {"model": ranking.model_version, "hits_boosted": sum("personal" in h.ranks for h in pool_hits)}
        elif history:
            meta["personalization"] = "unavailable"

        top = pool_hits[: req.k]
        return await self._answer_hits(session, intent, req, top, text_vec, prefs, personalized, meta, task="search")

    async def _answer_hits(
        self,
        session: Session,
        intent: Intent,
        req: ChatRequest,
        top: list[Hit],
        text_vec: np.ndarray | None,
        prefs: Preferences,
        personalized: bool,
        meta: dict[str, Any],
        task: str,
    ) -> ChatResponse:
        # evidence: review snippets scoped to the returned products, ranked against the query
        qvec = text_vec
        if qvec is None and self.encoder_ok and prefs.query:
            qvec, _ = await self._query_vectors(prefs.query, None)
        per = await self._timed_thread("evidence_ms", self.reviews.for_products, [h.product_id for h in top], qvec, 2)

        ctx = EvidenceContext.build(
            [(h.product_id, h.payload) for h in top],
            per,
            extras=[f"Bộ lọc đang áp dụng: {', '.join(prefs.summary()['filter_chips'])}"] if prefs.summary()["filter_chips"] else [],
            user_thresholds=[v for v in (prefs.filters.min_price, prefs.filters.max_price) if v],
        )
        results = []
        for i, h in enumerate(top):
            r = self._to_result(h, self._reasons(h, prefs, intent, req))
            revs = per.get(h.product_id, [])
            r.evidence = self._evidence_dicts(revs, [t for t, _ in ctx.reviews.get(f"P{i + 1}", [])])
            results.append(r)

        constraints = prefs.summary()["filter_chips"]
        display = prefs.display_query or ("ảnh bạn gửi" if req.image is not None else "")
        fallback = template_search(ctx, display or "sản phẩm", constraints, personalized, intent.lang)
        answer = await self._generate(
            task, req.message or display, ctx, fallback, intent.lang
        )
        if not top:
            answer = Answer(fallback, "template")
        meta["personalized"] = personalized
        return self._finish(session, intent, top, results, answer, ctx, meta)

    def _reasons(self, h: Hit, prefs: Preferences, intent: Intent, req: ChatRequest) -> list[str]:
        reasons: list[str] = []
        s = h.scores
        text_sim = s.get("text_query->text")
        if text_sim is not None and prefs.display_query:
            reasons.append(f"Mô tả sản phẩm khớp với “{prefs.display_query}” (độ tương đồng {text_sim:.2f})")
        if s.get("text_query->image") is not None and h.ranks.get("text_query->image", 99) <= 20:
            reasons.append("Hình ảnh sản phẩm khớp với mô tả của bạn (tìm kiếm chéo văn bản → ảnh)")
        if s.get("image_query->image") is not None:
            reasons.append(f"Ngoại hình giống ảnh bạn gửi (độ tương đồng thị giác {s['image_query->image']:.2f})")
        f = prefs.filters
        if f.max_price is not None and h.payload.get(S.P_PRICE) is not None:
            reasons.append(f"Nằm trong ngân sách ≤ {int(f.max_price):,}₫".replace(",", "."))
        if f.brands and h.payload.get(S.P_BRAND):
            reasons.append(f"Thương hiệu {h.payload[S.P_BRAND]} như bạn yêu cầu")
        if "personal" in h.ranks:
            reasons.append(f"Nằm trong top {h.ranks['personal']} gợi ý cá nhân hoá từ lịch sử của bạn (User Tower + Reranker)")
        avg, n = h.payload.get(S.P_AVG_RATING), h.payload.get(S.P_REVIEW_COUNT, 0)
        if avg is not None and n:
            reasons.append(f"Được đánh giá {avg:.1f}/5 từ {n} người mua")
        for key in ("similar->image", "similar->text"):
            if key in s:
                reasons.append("Có nội dung/hình ảnh gần với sản phẩm bạn quan tâm")
                break
        return reasons

    # ---- recommend -------------------------------------------------------------------
    async def _do_recommend(self, session: Session, intent: Intent, req: ChatRequest) -> ChatResponse:
        prefs = session.prefs
        filters = self._filters_from(intent, req)
        prefs.filters, prefs.query, prefs.display_query = filters, "", "gợi ý dành cho bạn"
        history = self._history(session, req)
        meta: dict[str, Any] = {"retrieval": "personal"}
        hits: list[Hit] = []
        ranking = await self._personal(history)
        if ranking and ranking.skus:
            by_sku = await asyncio.to_thread(self.searcher.products_by_sku, ranking.skus)
            seen = set(history)
            for rank, (sku, score) in enumerate(zip(ranking.skus, ranking.scores), start=1):
                hit = by_sku.get(sku)
                if hit is None or sku in seen or not filters.matches(hit.payload, hit.product_id):
                    continue
                hit.score = 1.0 / (RRF_K + rank)
                hit.scores["model"] = score
                hit.ranks["personal"] = rank
                hits.append(hit)
            meta["personalization"] = {"model": ranking.model_version, "source": ranking.source}
            if ranking.source == "popularity_fallback":
                meta["cold_start"] = True
        if not hits and history:
            ids = [p.product_id for p in (await asyncio.to_thread(self.searcher.products_by_sku, history)).values()]
            hits = await asyncio.to_thread(self.searcher.similar, ids, filters, POOL) if ids else []
            meta["retrieval"] = "content_similarity_fallback"
        if not hits:
            hits = await asyncio.to_thread(self.searcher.popular, filters, POOL)
            meta["retrieval"] = "popularity"
        top = hits[: req.k]
        personalized = any("personal" in h.ranks for h in top)
        return await self._answer_hits(session, intent, req, top, None, prefs, personalized, meta, task="search")

    # ---- similar ---------------------------------------------------------------------
    def _targets(self, session: Session, intent: Intent, want: int | None = None) -> list[ShownProduct]:
        explicit = intent.explicit_product_ids
        if explicit:
            found = {p.product_id: p for p in session.last_results}
            return [found.get(pid) or ShownProduct(pid, {}) for pid in explicit]
        picked = session.resolve(list(intent.ordinals))
        if intent.refers_last and session.last_results:
            picked.append(session.last_results[-1])
        if not picked and session.last_results:
            picked = session.last_results[: want or 1]
        return picked

    async def _do_similar(self, session: Session, intent: Intent, req: ChatRequest) -> ChatResponse:
        targets = self._targets(session, intent)
        if not targets:
            return self._plain(session, "similar", intent.lang, "Bạn muốn tìm sản phẩm giống cái nào? Hãy chọn một sản phẩm (ví dụ “giống cái đầu tiên”).")
        prefs = session.prefs
        filters = self._filters_from(intent, req)
        prefs.filters = filters
        for t in targets:
            session.engage(str(t.payload.get(S.P_ITEM_ID, "")))
        ids = [t.product_id for t in targets]
        hits = await asyncio.to_thread(self.searcher.similar, ids, filters, POOL)
        prefs.query, prefs.display_query = "", f"giống “{targets[0].payload.get(S.P_TITLE, '')[:50]}”"
        return await self._answer_hits(session, intent, req, hits[: req.k], None, prefs, False, {"retrieval": "item_to_item"}, task="search")

    # ---- explain ---------------------------------------------------------------------
    async def _payloads(self, targets: list[ShownProduct]) -> list[ShownProduct]:
        missing = [t.product_id for t in targets if not t.payload]
        if missing:
            fetched = await asyncio.to_thread(self.searcher.get_products, missing)
            for t in targets:
                if not t.payload:
                    t.payload = fetched.get(t.product_id, {})
        return [t for t in targets if t.payload]

    async def _do_explain(self, session: Session, intent: Intent, req: ChatRequest) -> ChatResponse:
        targets = await self._payloads(self._targets(session, intent, 1)[:1])
        if not targets:
            return self._plain(session, "explain", intent.lang, "Bạn muốn hỏi về sản phẩm nào? Hãy chọn một sản phẩm trong danh sách (ví dụ “tại sao sản phẩm 2?”).")
        target = targets[0]
        session.engage(str(target.payload.get(S.P_ITEM_ID, "")))
        question = req.message or "Vì sao nên chọn sản phẩm này?"
        qvec, _ = await self._query_vectors(question, None)
        per = await self._timed_thread("evidence_ms", self.reviews.for_products, [target.product_id], qvec, 4)
        ctx = EvidenceContext.build(
            [(target.product_id, target.payload)],
            per,
            extras=[f"[WHY] {r}" for r in target.reasons],
            user_thresholds=[v for v in (session.prefs.filters.min_price, session.prefs.filters.max_price) if v],
        )
        fallback = template_explain(ctx, target.reasons, intent.lang)
        answer = await self._generate("explain", question, ctx, fallback, intent.lang)
        result = ProductResult(target.product_id, str(target.payload.get(S.P_ITEM_ID, "")), 0.0, target.payload, target.reasons)
        result.evidence = self._evidence_dicts(per.get(target.product_id, []), [t for t, _ in ctx.reviews.get("P1", [])])
        resp = ChatResponse(
            session_id=session.session_id,
            reply=answer.text,
            action="explain",
            lang=intent.lang,
            products=[result],
            filters=session.prefs.filters.to_dict(),
            filter_chips=session.prefs.summary()["filter_chips"],
            suggestions=self._suggestions(session, intent.lang),
            citations=answer.citations,
            meta={"answer_source": answer.source, "evidence_count": len(result.evidence)},
            warnings=answer.warnings,
        )
        return resp

    # ---- compare ---------------------------------------------------------------------
    async def _do_compare(self, session: Session, intent: Intent, req: ChatRequest) -> ChatResponse:
        targets = await self._payloads(self._targets(session, intent, 2))
        if len(targets) < 2:
            return self._plain(session, "compare", intent.lang, "Mình cần ít nhất hai sản phẩm để so sánh. Hãy nói ví dụ “so sánh 1 và 2”.")
        targets = targets[:4]
        for t in targets:
            session.engage(str(t.payload.get(S.P_ITEM_ID, "")))
        question = req.message or "So sánh các sản phẩm này"
        qvec, _ = await self._query_vectors(question, None)
        ids = [t.product_id for t in targets]
        per = await self._timed_thread("evidence_ms", self.reviews.for_products, ids, qvec, 2)
        crit = {pid: await asyncio.to_thread(self.reviews.critical, pid, 1) for pid in ids}
        # fold the critical reviews into the context so [R#.#] tags exist for them too
        merged = {pid: list(per.get(pid, [])) + [r for r in crit[pid] if all(r.review_id != x.review_id for x in per.get(pid, []))] for pid in ids}
        ctx = EvidenceContext.build([(t.product_id, t.payload) for t in targets], merged)
        crit_tagged = {
            f"P{i + 1}": [(tag, r) for tag, r in ctx.reviews[f"P{i + 1}"] if any(r.review_id == c.review_id for c in crit[pid])]
            for i, pid in enumerate(ids)
        }
        fallback = template_compare(ctx, crit_tagged)
        answer = await self._generate("compare", question, ctx, fallback, intent.lang)
        results = []
        for i, t in enumerate(targets):
            r = ProductResult(t.product_id, str(t.payload.get(S.P_ITEM_ID, "")), 0.0, t.payload, t.reasons)
            r.evidence = self._evidence_dicts(merged[t.product_id], [tag for tag, _ in ctx.reviews[f"P{i + 1}"]])
            results.append(r)
        return ChatResponse(
            session_id=session.session_id,
            reply=answer.text,
            action="compare",
            lang=intent.lang,
            products=results,
            filters=session.prefs.filters.to_dict(),
            filter_chips=session.prefs.summary()["filter_chips"],
            suggestions=self._suggestions(session, intent.lang),
            citations=answer.citations,
            meta={"answer_source": answer.source},
            warnings=answer.warnings,
        )
