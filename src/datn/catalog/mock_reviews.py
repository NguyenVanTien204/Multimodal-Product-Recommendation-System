"""Mock reviews for the H&M shop: Amazon reviews re-attached to similar H&M products.

H&M has no review text, so the RAG layer would have nothing to cite. Instead of inventing text, real Amazon
reviews are borrowed and attached to the H&M product that is closest to the Amazon product they were written
about (Jina CLIP v2 text-text cosine, both Matryoshka-cut to 512-d). Every borrowed review is flagged `is_mock`
and keeps its provenance (`source_item_id`, `source_review_id`, `match_score`), so it can be removed or replaced
by real reviews later without touching anything else.

Guards against obviously wrong borrows: a similarity floor, a one-review-one-product assignment (no duplicate
text across products), a per-product cap, and rejection of reviews that name a (foreign) brand, contain a link /
marketplace words, or name a colour the target product does not have.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9&']+")
_VIDEO_ID = re.compile(r"\[\[VIDEOID:[^\]]*\]\]")
_HTML_TAG = re.compile(r"<[^>]{1,40}>")
_EXTERNAL = re.compile(r"https?://|www\.|\S+@\S+\.\S+|\b(?:amazon|prime|walmart|ebay|etsy|seller|vendor)\b", re.I)

# Basic colour words as reviewers write them -> the H&M colour_group_name tokens that satisfy them.
COLOUR_WORDS: dict[str, set[str]] = {
    "black": {"black"}, "white": {"white", "off"}, "red": {"red"}, "blue": {"blue", "turquoise"}, "navy": {"blue"},
    "green": {"green", "khaki"}, "khaki": {"khaki", "green"}, "pink": {"pink"}, "grey": {"grey"}, "gray": {"grey"},
    "brown": {"brown", "beige"}, "beige": {"beige"}, "tan": {"beige", "brown"}, "yellow": {"yellow", "gold"},
    "purple": {"lilac", "purple"}, "orange": {"orange"}, "gold": {"gold", "yellow"}, "silver": {"silver", "grey"},
}


def clean_review_text(text: str | None) -> str:
    """Scrape artefacts out of an Amazon review: [[VIDEOID:...]] markers, HTML entities (&quot;) and tags (<br />)."""
    t = html.unescape(_VIDEO_ID.sub(" ", text or ""))
    return " ".join(_HTML_TAG.sub(" ", t).split())


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def truncate_renorm(x: np.ndarray, dim: int = 512) -> np.ndarray:
    """Matryoshka cut: first `dim` coordinates, then unit length again (same as encoding with truncate_dim)."""
    out = np.asarray(x[..., :dim], dtype=np.float32)
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-8)


# Brand names that are also everyday words in reviews ("I guess", "a good match"): screening them would reject
# reviews for nothing. Found by looking at the top triggers of the screen on the real corpus.
AMBIGUOUS_BRANDS = frozenset(
    {"guess", "match", "forever", "alternative", "hue", "head", "camel", "twisted", "impact", "gap", "lee", "citizen",
     "vibrant", "obviously", "often", "party", "simple", "travel", "start", "idea", "unique", "basic", "generic", "unknown", "unbranded"}
)


def make_brand_set(
    brand_counts: Mapping[str, int],
    doc_freq: Mapping[str, float],
    *,
    min_products: int = 25,
    n_common: int = 300,
    ambiguous: Iterable[str] = AMBIGUOUS_BRANDS,
) -> set[str]:
    """Lower-cased brand names worth screening for: the established brands (>= `min_products` products in the
    source catalog). The raw brand field is full of seller names and SEO phrases ('if you', 'which is', 'party');
    brands made only of the `n_common` most frequent review words and a stoplist of everyday words are dropped too."""
    common = set(sorted(doc_freq, key=lambda t: -doc_freq[t])[:n_common])
    skip = {a.lower() for a in ambiguous}
    out = set()
    for brand, count in brand_counts.items():
        b = (brand or "").strip().lower()
        if count < min_products or len(b) < 3 or b in skip:
            continue
        toks = tokens(b)
        if not toks or all(t in common for t in toks):
            continue
        out.add(b)
    return out


def document_frequency(token_lists: Iterable[Sequence[str]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    n = 0
    for toks in token_lists:
        n += 1
        counts.update(set(toks))
    return {t: c / max(n, 1) for t, c in counts.items()}


def mentions_brand(toks: Sequence[str], brands: set[str], max_n: int = 3) -> str | None:
    """First brand found as a 1..max_n-gram of the review, else None (O(words), not O(brands))."""
    for size in range(min(max_n, len(toks)), 0, -1):
        for i in range(len(toks) - size + 1):
            gram = " ".join(toks[i : i + size])
            if gram in brands:
                return gram
    return None


def has_external_reference(text: str) -> bool:
    return bool(_EXTERNAL.search(text or ""))


def colour_conflict(toks: Sequence[str], item_colour: str | None) -> bool:
    """True if the review names basic colours and none of them is (part of) the product's colour."""
    named = {w for w in toks if w in COLOUR_WORDS}
    if not named:
        return False
    have = set(tokens(item_colour or ""))
    return not any(COLOUR_WORDS[w] & have or w in have for w in named)


# Which H&M audiences may host a review whose Amazon product title names these audiences. "divided" is H&M's
# young-adult line (women's and men's pieces), so it is acceptable for adult cues but never for kids/baby.
_AUDIENCE_HOSTS = {"women": {"women", "divided"}, "men": {"men", "divided"}, "kids": {"kids"}, "baby": {"baby"}}


def allowed_audiences(amazon_title: str | None) -> set[str] | None:
    """H&M audiences compatible with the audience named in an Amazon title ("Women's ...", "Boys' ..."), or None
    when the title names none (then every audience is allowed)."""
    from ..agent.intent import fold, parse_audiences

    named = parse_audiences(fold(amazon_title or ""))
    if not named:
        return None
    hosts: set[str] = set()
    for a in named:
        hosts |= _AUDIENCE_HOSTS.get(a, set())
    return hosts or None


def top_matches(source: np.ndarray, target: np.ndarray, k: int = 5, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each source row, the k most similar target rows (cosine; rows are unit length). Returns (indices, sims), best first."""
    n = source.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        block = source[s : s + chunk] @ target.T
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[s : s + chunk] = np.take_along_axis(part, order, axis=1)
        sim[s : s + chunk] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def assign_reviews(
    reviews: Sequence[Mapping],
    candidates: Mapping[str, Sequence[tuple[int, float]]],
    item_colour: Mapping[int, str | None],
    *,
    min_sim: float,
    cap_per_item: int = 6,
    brand_set: set[str] | None = None,
    min_chars: int = 20,
    seed: int = 20261006,
) -> tuple[list[dict], dict[str, int]]:
    """One review -> at most one H&M product.

    `reviews`: dicts with review_id, source_item_id (Amazon ASIN), text (+ whatever else should be carried over).
    `candidates[asin]`: [(hm_product_id, similarity)] best first. A review goes to the least-loaded eligible
    candidate (ties: higher similarity), which spreads reviews across products instead of piling them on one.
    Returns (assignments, rejection counters).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(reviews))
    load: Counter[int] = Counter()
    rejected: Counter[str] = Counter()
    out: list[dict] = []
    brands = brand_set or set()
    for i in order:
        r = reviews[int(i)]
        text = r["text"]
        toks = tokens(text)
        if len(text) < min_chars:
            rejected["too_short"] += 1
            continue
        if has_external_reference(text):
            rejected["external_reference"] += 1
            continue
        if brands and mentions_brand(toks, brands):
            rejected["brand_mention"] += 1
            continue
        options = [(pid, s) for pid, s in candidates.get(r["source_item_id"], ()) if s >= min_sim]
        if not options:
            rejected["no_similar_product"] += 1
            continue
        options = [(pid, s) for pid, s in options if not colour_conflict(toks, item_colour.get(pid))]
        if not options:
            rejected["colour_conflict"] += 1
            continue
        options = [(pid, s) for pid, s in options if load[pid] < cap_per_item]
        if not options:
            rejected["capacity"] += 1
            continue
        pid, sim = min(options, key=lambda o: (load[o[0]], -o[1]))
        load[pid] += 1
        out.append({**r, "product_id": pid, "match_score": float(sim)})
    out.sort(key=lambda a: (a["product_id"], a["review_id"]))
    return out, dict(rejected)
