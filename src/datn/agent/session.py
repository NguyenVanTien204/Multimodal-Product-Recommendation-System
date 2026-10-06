from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..retrieval.filters import SearchFilters
from .preferences import PreferenceEvent

MAX_TURNS = 20
MAX_EVENTS = 200


@dataclass
class ShownProduct:
    product_id: int
    payload: dict[str, Any]
    reasons: list[str] = field(default_factory=list)


@dataclass
class Preferences:
    """What the user has asked for so far. Refinements edit this object and the
    pipeline is simply re-run; nothing is retrained."""

    query: str = ""  # semantic text of the current search (English glossary terms appended)
    display_query: str = ""  # the user's own words, for messages
    filters: SearchFilters = field(default_factory=SearchFilters)
    colors: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.display_query or self.query,
            "colors": list(self.colors),
            "filters": self.filters.to_dict(),
            "filter_chips": self.filters.describe() + ([] if self.filters.colours else [f"Màu: {c}" for c in self.colors]),
        }


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: list[dict[str, str]] = field(default_factory=list)
    prefs: Preferences = field(default_factory=Preferences)
    last_results: list[ShownProduct] = field(default_factory=list)  # 1-based ordinals refer to this list
    candidate_pool: list[ShownProduct] = field(default_factory=list)  # wider set from the last vector search
    pool_key: tuple | None = None  # (semantic query, image digest) the pool was built for
    pool_filters: SearchFilters = field(default_factory=SearchFilters)
    seen_product_ids: list[int] = field(default_factory=list)
    events: list[PreferenceEvent] = field(default_factory=list)  # feedback recorded in this session, oldest first
    pending_events: list[PreferenceEvent] = field(default_factory=list)  # recorded during the current turn (returned to the shop DB)
    sku_product_ids: dict[str, int] = field(default_factory=dict)  # sku -> product id, resolved lazily for negatives
    neighbour_cache: dict[int, dict[int, int]] = field(default_factory=dict)  # disliked product id -> {neighbour id: rank}

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        del self.turns[:-MAX_TURNS]
        self.updated_at = time.time()

    def resolve(self, ordinals: list[int]) -> list[ShownProduct]:
        out: list[ShownProduct] = []
        for n in ordinals:
            if 1 <= n <= len(self.last_results):
                out.append(self.last_results[n - 1])
        return out

    def record(self, sku: str, kind: str) -> PreferenceEvent | None:
        """Remember one interaction. Repeating the same kind on the same item back-to-back is ignored."""
        if not sku:
            return None
        if self.events and self.events[-1].sku == sku and self.events[-1].kind == kind:
            return None
        event = PreferenceEvent(sku=sku, kind=kind)
        self.events.append(event)
        self.pending_events.append(event)
        del self.events[:-MAX_EVENTS]
        return event

    def engage(self, sku: str) -> None:
        """Implicit interest: the user asked about / compared / looked for items like this one."""
        self.record(sku, "click")


class SessionStore:
    """In-memory TTL/LRU store keyed by session id.

    API handlers stay stateless: all conversation state is addressed by
    `session_id`, so swapping this class for a Redis-backed one (same three
    methods) changes nothing else.
    """

    def __init__(self, ttl_s: float = 3600.0, max_sessions: int = 2000) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._data: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            self._evict()
            if session_id and session_id in self._data:
                self._data.move_to_end(session_id)
                return self._data[session_id]
            sid = session_id or uuid.uuid4().hex
            session = Session(session_id=sid)
            self._data[sid] = session
            self._evict()  # enforce the size cap right after inserting, not on the next call
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._evict()
            return self._data.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _evict(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._data.items() if now - v.updated_at > self.ttl_s]:
            del self._data[sid]
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
