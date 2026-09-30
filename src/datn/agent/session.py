from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..retrieval.filters import SearchFilters

MAX_TURNS = 20


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
            "filter_chips": self.filters.describe() + [f"Màu: {c}" for c in self.colors],
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
    focus_history_skus: list[str] = field(default_factory=list)  # items the user engaged with this session, oldest first

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

    def engage(self, sku: str) -> None:
        if sku and (not self.focus_history_skus or self.focus_history_skus[-1] != sku):
            self.focus_history_skus.append(sku)
            del self.focus_history_skus[:-20]


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
