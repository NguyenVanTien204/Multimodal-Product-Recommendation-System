from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping

from ..retrieval.schema import USD_TO_VND

# ------------------------------------------------------------------------------
# Text normalisation. `fold` is 1:1 with the NFC-normalised input (same length),
# so match spans found on the folded text can be cut out of the original string
# to keep the user's diacritics for the multilingual encoder.
# ------------------------------------------------------------------------------


def _base(ch: str) -> str:
    if ch in "đĐ":
        return "d"
    decomposed = unicodedata.normalize("NFD", ch)
    return decomposed[0] if decomposed else ch


def fold(text: str) -> str:
    return "".join(_base(c) for c in unicodedata.normalize("NFC", text).lower())


_VI_MARKS = re.compile(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", re.IGNORECASE)
_VI_WORDS = {"toi", "minh", "cho", "tim", "muon", "can", "giay", "ao", "quan", "vay", "tui", "duoi", "tren", "khoang", "gia", "re", "hon",
             "mau", "cai", "nay", "kia", "nao", "xin", "chao", "goi", "y", "san", "pham", "so", "sanh", "giup", "voi", "va", "khong", "co"}


def detect_lang(text: str, default: str = "vi") -> str:
    if _VI_MARKS.search(text):
        return "vi"
    words = set(re.findall(r"[a-z]+", fold(text)))
    if words & _VI_WORDS:
        return "vi"
    return "en" if words else default


# ------------------------------------------------------------------------------
# Vietnamese -> English product glossary (the catalog is English Amazon data).
# Longest phrase wins; keys are folded.
# ------------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "ao thun": "t-shirt", "ao phong": "t-shirt", "ao so mi": "dress shirt", "ao khoac": "jacket", "ao len": "sweater",
    "ao hoodie": "hoodie", "ao ni": "hoodie", "ao ba lo": "tank top", "ao vest": "blazer", "ao polo": "polo shirt", "ao dai": "long sleeve top",
    "ao ngưc": "bra", "ao nguc": "bra", "ao": "top shirt", "quan jean": "jeans", "quan bo": "jeans", "quan short": "shorts", "quan dui": "shorts",
    "quan tay": "dress pants", "quan": "pants", "vay": "dress skirt", "dam": "dress", "giay the thao": "sneakers", "giay chay bo": "running shoes",
    "giay cao got": "high heels", "giay luoi": "loafers", "giay tay": "dress shoes", "giay boot": "boots", "giay": "shoes", "dep": "sandals slippers",
    "sandal": "sandals", "bot": "boots", "tui xach": "handbag", "tui deo cheo": "crossbody bag", "tui": "bag", "balo": "backpack", "vi da": "wallet",
    "vi": "wallet", "dong ho": "watch", "nhan": "ring", "day chuyen": "necklace", "vong co": "necklace", "vong tay": "bracelet", "lac tay": "bracelet",
    "bong tai": "earrings", "hoa tai": "earrings", "mu": "hat", "non": "hat", "kinh mat": "sunglasses", "kinh": "glasses", "that lung": "belt",
    "day nit": "belt", "tat": "socks", "vo": "socks", "khan": "scarf", "gang tay": "gloves", "do boi": "swimsuit", "bikini": "bikini",
    "ao croptop": "cropped top", "ao cardigan": "cardigan", "ao bodysuit": "bodysuit", "ao khoac long": "coat", "ao khoac gio": "windbreaker jacket",
    "quan legging": "leggings", "quan au": "trousers", "quan jogger": "joggers", "quan ni": "sweatpants", "quan tat": "tights", "chan vay": "skirt",
    "dam suong": "shift dress", "dam om": "bodycon dress", "jumpsuit": "jumpsuit", "do bo": "set outfit", "bo do": "set outfit", "ao bra": "bralette",
    "quan lot": "briefs underwear", "ao len cardigan": "cardigan", "dep xo ngon": "flip flops", "giay bup be": "flats", "giay de bet": "flats",
    "do lot": "underwear", "do ngu": "pajamas", "kinh ram": "sunglasses", "mu luoi trai": "baseball cap", "khan quang co": "scarf", "khan choang": "scarf", "do the thao": "activewear", "do tap": "workout clothes",
}
MODIFIERS: dict[str, str] = {
    "tre em": "kids", "be gai": "girls", "be trai": "boys", "nam": "men's", "nu": "women's", "cong so": "office work", "di lam": "work office",
    "di tiec": "party", "mua dong": "winter", "mua he": "summer", "qua tang": "gift", "da that": "genuine leather", "da": "leather",
    "chong nuoc": "waterproof", "thoang mat": "breathable", "thoai mai": "comfortable", "em chan": "comfortable cushioned", "sang trong": "elegant luxury",
    "the thao": "sports", "chay bo": "running", "vintage": "vintage", "casual": "casual", "thoi trang": "fashion",
}
COLORS: dict[str, str] = {
    "den": "black", "trang": "white", "do": "red", "xanh duong": "blue", "xanh da troi": "blue", "xanh la": "green", "xanh navy": "navy blue",
    "xanh": "blue", "vang": "yellow", "hong": "pink", "tim": "purple", "cam": "orange", "nau": "brown", "xam": "gray", "kem": "beige",
    "be": "beige", "bac": "silver", "black": "black", "white": "white", "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "pink": "pink", "purple": "purple", "orange": "orange", "brown": "brown", "gray": "gray", "grey": "gray", "beige": "beige",
    "silver": "silver", "gold": "gold", "navy": "navy blue",
}
_EN_NOUNS = {
    "shirt", "shirts", "t-shirt", "tee", "jacket", "coat", "sweater", "hoodie", "jeans", "pants", "shorts", "dress", "skirt", "shoes", "shoe",
    "sneakers", "sneaker", "boots", "boot", "sandals", "sandal", "heels", "loafers", "bag", "bags", "handbag", "backpack", "wallet", "watch",
    "watches", "ring", "necklace", "bracelet", "earrings", "hat", "cap", "sunglasses", "belt", "socks", "scarf", "gloves", "swimsuit", "bra",
    "underwear", "pajamas", "blouse", "top", "tops", "leggings", "vest", "suit", "tie", "jewelry", "slippers", "flats", "clothes", "outfit",
}

_FILLERS = re.compile(
    r"\b(?:tim|kiem|cho toi|cho minh|giup toi|giup minh|toi muon|minh muon|toi can|minh can|toi dang tim|minh dang tim|muon mua|can mua|xem|hay|giup|"
    r"vui long|ban oi|shop oi|nhe|nha|di|a|show me|find me|i want|i need|i'm looking for|looking for|please|can you|could you|search for|show)\b"
)

# ------------------------------------------------------------------------------
# Price parsing
# ------------------------------------------------------------------------------
_NUM = r"\d+(?:[.,]\d+)*"
_UNIT = r"(?:k|nghin|ngan|tr|trieu|m|usd|\$|do|dollars?|dong|d|vnd|vnđ)"
_AMOUNT = rf"(?P<n>{_NUM})\s*(?P<u>{_UNIT})?\b"


def _to_number(raw: str) -> float:
    # "1.500.000" / "1,500,000" thousands separators vs "1.5" / "1,5" decimals
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", raw):
        return float(re.sub(r"[.,]", "", raw))
    return float(raw.replace(",", "."))


def _amount_vnd(raw: str, unit: str | None, lang: str, dollar_prefix: bool = False) -> float:
    value = _to_number(raw)
    unit = (unit or "").lower()
    if dollar_prefix or unit in {"usd", "$", "do", "dollar", "dollars"}:
        return value * USD_TO_VND
    if unit in {"k", "nghin", "ngan"}:
        return value * 1_000
    if unit in {"tr", "trieu", "m"}:
        return value * 1_000_000
    if unit in {"dong", "d", "vnd", "vnđ"}:
        return value
    # Unit-less: an English message is almost certainly dollars ("under 100"),
    # a Vietnamese one almost certainly thousands of dong ("dưới 500").
    if lang == "en":
        return value * USD_TO_VND
    return value * 1_000 if value < 10_000 else value


@dataclass
class PriceSpec:
    min_price: float | None = None
    max_price: float | None = None
    mode: str | None = None  # "cheaper" | "pricier" | None (relative to what was shown)
    spans: list[tuple[int, int]] = field(default_factory=list)


_P = r"(?P<d>\$)?\s*" + _AMOUNT


def parse_price(folded: str, lang: str) -> PriceSpec:
    spec = PriceSpec()

    def grab(pattern: str) -> re.Match | None:
        return re.search(pattern, folded)

    rng = grab(rf"(?:tu|from|between)\s*(?P<d1>\$)?\s*(?P<n1>{_NUM})\s*(?P<u1>{_UNIT})?\s*(?:den|toi|-|to|and)\s*(?P<d2>\$)?\s*(?P<n2>{_NUM})\s*(?P<u2>{_UNIT})?\b") or grab(
        rf"(?P<d1>\$)?(?P<n1>{_NUM})\s*(?P<u1>{_UNIT})\s*(?:-|den|to)\s*(?P<d2>\$)?(?P<n2>{_NUM})\s*(?P<u2>{_UNIT})\b"
    )
    if rng:
        u1 = rng.group("u1") or rng.group("u2")
        lo = _amount_vnd(rng.group("n1"), u1, lang, bool(rng.group("d1")))
        hi = _amount_vnd(rng.group("n2"), rng.group("u2") or u1, lang, bool(rng.group("d2")))
        spec.min_price, spec.max_price = sorted((lo, hi))
        spec.spans.append(rng.span())
        return spec

    m = grab(rf"(?:duoi|under|below|less than|toi da|khong qua|khong vuot qua|max|re hon|<=?|nho hon|it hon)\s*{_P}")
    if m and m.group("n"):
        spec.max_price = _amount_vnd(m.group("n"), m.group("u"), lang, bool(m.group("d")))
        spec.spans.append(m.span())
    m = grab(rf"(?:tren|over|above|more than|toi thieu|it nhat|min|>=?|lon hon|cao hon)\s*{_P}")
    if m and m.group("n"):
        spec.min_price = _amount_vnd(m.group("n"), m.group("u"), lang, bool(m.group("d")))
        spec.spans.append(m.span())
    m = grab(rf"(?:khoang|tam|around|about|approximately)\s*{_P}")
    if m and m.group("n") and spec.min_price is None and spec.max_price is None:
        center = _amount_vnd(m.group("n"), m.group("u"), lang, bool(m.group("d")))
        spec.min_price, spec.max_price = center * 0.75, center * 1.25
        spec.spans.append(m.span())
    if spec.min_price is None and spec.max_price is None:
        m = grab(rf"\$\s*(?P<n>{_NUM})|(?P<n2>{_NUM})\s*(?:usd|dollars?)\b")
        # a bare "$50" with no direction keyword is treated as an upper bound
        if m:
            n = m.group("n") or m.group("n2")
            spec.max_price = _to_number(n) * USD_TO_VND
            spec.spans.append(m.span())
    if re.search(r"\b(?:re hon|cheaper|gia thap hon|binh dan hon|tiet kiem hon|it tien hon)\b", folded) and spec.max_price is None:
        spec.mode = "cheaper"
    elif re.search(r"\b(?:dat hon|cao cap hon|sang hon|pricier|more expensive|premium|xin hon)\b", folded) and spec.min_price is None:
        spec.mode = "pricier"
    return spec


# ------------------------------------------------------------------------------
# Intent
# ------------------------------------------------------------------------------
@dataclass
class Intent:
    action: str = "search"  # search | refine | recommend | similar | explain | compare | greet | help | reset | unknown
    query: str = ""  # semantic text for the encoder (EN gloss + original words)
    display_query: str = ""  # cleaned user words
    min_price: float | None = None
    max_price: float | None = None
    price_mode: str | None = None
    brands: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    audiences: tuple[str, ...] = ()  # H&M: women | men | divided | kids | baby
    min_rating: float | None = None
    ordinals: tuple[int, ...] = ()
    refers_last: bool = False  # "cái cuối", "the last one"
    clear_filters: bool = False
    exclude_shown: bool = False
    explicit_product_ids: tuple[int, ...] = ()  # set by UI buttons, bypasses ordinal parsing
    feedback: str | None = None  # "like" | "dislike" about the referenced result(s)
    has_product_noun: bool = False
    lang: str = "vi"

    @property
    def has_constraints(self) -> bool:
        return any(
            v is not None for v in (self.min_price, self.max_price, self.price_mode, self.min_rating)
        ) or bool(self.brands or self.colors or self.audiences)


_ORD_WORDS = {
    "dau tien": 1, "thu nhat": 1, "first": 1, "thu hai": 2, "second": 2, "thu ba": 3, "third": 3, "thu tu": 4, "fourth": 4, "thu nam": 5, "fifth": 5,
}
_GREET = re.compile(r"^\s*(?:xin chao|chao|hello|hi|hey|alo|good (?:morning|afternoon|evening))\b[\s!.,]*(?:ban|shop|bot|ai|assistant)?\W*$")
_HELP = re.compile(r"\b(?:ban lam duoc gi|ban giup duoc gi|huong dan|help|what can you do|cach su dung|ban la ai|who are you)\b")
_RESET = re.compile(r"\b(?:bo loc|xoa loc|bo het loc|tim lai|bat dau lai|lam lai|reset|clear filters?|start over|new search)\b")
_COMPARE = re.compile(r"\b(?:so sanh|compare|khac nhau|khac gi|nen chon|chon cai nao|cai nao tot hon|cai nao hon|which is better|versus|vs)\b")
_EXPLAIN = re.compile(
    r"\b(?:tai sao|vi sao|giai thich|why|review|danh gia|nhan xet|co tot khong|tot khong|ben khong|chi tiet|thong tin|chat lieu|how good|is it good|"
    r"ly do|explain|nguoi mua|feedback|co nen mua)\b"
)
_SIMILAR = re.compile(r"\b(?:giong|tuong tu|similar|kieu nay|mau nay|nhu cai|like this|more like|nhung cai giong)\b")
_RECOMMEND = re.compile(
    r"\b(?:goi y|de xuat|recommend|danh cho toi|danh cho minh|hop voi toi|toi co the thich|co the toi se thich|for me|suggest|"
    r"san pham hot|ban chay|pho bien|trending|popular)\b"
)
_DISLIKE = re.compile(
    r"\b(?:(?:khong|ko|chang|dung|chua)\s+(?:thich|ung|hop|muon|quan tam|dep)|(?:xau|te|chan)\s+qua|bo qua|loai bo|bo cai|"
    r"dislike|(?:don'?t|do not|didn'?t)\s+like|not (?:a fan|interested)|hate|skip|no thanks|nope)\b"
)
_LIKE = re.compile(
    r"\b(?:thich|ung y|ung qua|dep qua|dep do|xinh qua|yeu thich|like|love|nice|beautiful|great|perfect|lay cai|chon cai|quan tam)\b"
)
_MORE = re.compile(r"\b(?:cai khac|mau khac|san pham khac|khac di|xem them|them nua|more options|something else|others?|another)\b")
_RATING = re.compile(r"(?:danh gia|rating|stars?)\s*(?:tren|tu|>=?|cao hon|at least|over)?\s*(\d(?:[.,]\d)?)\b|\b(\d(?:[.,]\d)?)\s*(?:sao|stars?)\b(?:\s*(?:tro len|\+|and up))?")
_HIGH_RATED = re.compile(r"\b(?:danh gia cao|highly rated|best rated|top rated|nhieu sao)\b")
_BRAND_EXPLICIT = re.compile(r"\b(?:thuong hieu|hang|brand)\s+([a-z0-9&'.\- ]{2,30})")

_REFINE_WORDS = re.compile(r"\b(?:re hon|dat hon|cao cap hon|gia thap hon|cheaper|more expensive|premium|mau sac|mau|color|colour|rating|stars?)\b")

_BRAND_STOP = {
    "men", "women", "kids", "mens", "womens", "the", "classic", "sport", "sports", "style", "fashion", "new", "best", "gold", "silver", "black",
    "white", "generic", "unknown", "brand", "collection", "basic", "casual", "original", "vintage", "premium", "luxury", "designer", "home",
    "boys", "girls", "baby", "unisex", "usa", "one", "love", "star", "sky", "blue", "red", "pink", "green", "life", "city", "urban",
}


def _ordinals(folded: str) -> tuple[list[int], bool]:
    found: list[int] = []
    for m in re.finditer(r"#\s*(\d{1,2})", folded):
        found.append(int(m.group(1)))
    for m in re.finditer(r"\b(?:san pham|sp|mon|cai|mau|item|product|number|so|thu|option|lua chon)\s*(?:so|thu|#)?\s*(\d{1,2})\b", folded):
        found.append(int(m.group(1)))
    for m in re.finditer(r"\b(\d{1,2})\s*(?:va|voi|and|vs|&|,)\s*(\d{1,2})\b", folded):
        found += [int(m.group(1)), int(m.group(2))]
    for word, n in _ORD_WORDS.items():
        if re.search(rf"\b(?:cai|san pham|mon|mau|the)?\s*{word}\b", folded):
            found.append(n)
    seen: list[int] = []
    for n in found:
        if 1 <= n <= 50 and n not in seen:
            seen.append(n)
    last = bool(re.search(r"\b(?:cai cuoi|cuoi cung|the last|last one)\b", folded))
    return seen, last


def _replace_glossary(folded: str) -> tuple[list[str], bool]:
    terms: list[str] = []
    has_noun = False
    work = f" {folded} "
    for table, is_noun in ((GLOSSARY, True), (MODIFIERS, False)):
        for key in sorted(table, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", work):
                terms.append(table[key])
                work = re.sub(rf"(?<![a-z]){re.escape(key)}(?![a-z])", " ", work)
                has_noun = has_noun or is_noun
    if set(re.findall(r"[a-z\-]+", folded)) & _EN_NOUNS:
        has_noun = True
    return terms, has_noun


def _find_brands(folded: str, known: Mapping[str, str]) -> tuple[str, ...]:
    out: list[str] = []
    m = _BRAND_EXPLICIT.search(folded)
    if m:
        cand = m.group(1).strip()
        words = cand.split()
        for size in range(min(3, len(words)), 0, -1):
            key = " ".join(words[:size])
            if key in known:
                out.append(known[key])
                break
    if not out:
        # O(words) n-gram lookup. (A regex per known brand cost ~230 ms per message with ~3,000 brands.)
        words = [w.strip(".,;:!?()\"'") or w for w in folded.split()]
        for size in range(min(4, len(words)), 0, -1):  # prefer the longest brand phrase
            for i in range(len(words) - size + 1):
                key = " ".join(words[i : i + size])
                if len(key) >= 3 and key not in _BRAND_STOP and key in known:
                    out.append(known[key])
                    break
            if out:
                break
    return tuple(out)


def _strip(original: str, folded: str, spans: list[tuple[int, int]]) -> str:
    keep = [True] * len(original)
    for a, b in spans:
        for i in range(a, min(b, len(keep))):
            keep[i] = False
    return "".join(c for c, k in zip(original, keep) if k)


_AUD_BABY = re.compile(r"\b(?:em be|so sinh|baby|babies|infant|newborn|toddler)\b")
_AUD_KIDS = re.compile(r"\b(?:tre em|tre con|be gai|be trai|con trai nho|kids?|children|child|boys?|girls?)\b")
_AUD_MEN = re.compile(r"\b(?:nam gioi|dan ong|dang nam|nam|men|mens|man|male|guys?|gentlemen)\b")
_AUD_WOMEN = re.compile(r"\b(?:nu gioi|phu nu|con gai|nu|women|womens|woman|ladies|lady|female)\b")


def parse_audiences(folded: str) -> tuple[str, ...]:
    """Who the clothes are for, from the folded message: ('men',), ('women',), ('kids',), ('baby',) -- or both
    genders when both are named. Kids/baby win over gender words ('be gai' is a girl child, not a woman)."""
    if _AUD_BABY.search(folded):
        return ("baby",)
    if _AUD_KIDS.search(folded):
        return ("kids",)
    out: list[str] = []
    if _AUD_MEN.search(folded):
        out.append("men")
    if _AUD_WOMEN.search(folded):
        out.append("women")
    return tuple(out)


def parse_intent(
    message: str,
    *,
    has_image: bool = False,
    has_results: bool = False,
    known_brands: Mapping[str, str] | None = None,
    default_lang: str = "vi",
) -> Intent:
    """Rule-based NLU for the shop assistant (Vietnamese + English).

    Deterministic on purpose: the conversational state machine (refine vs new
    search, ordinal references, price constraints) must behave identically with
    or without an LLM, and is unit-tested. The LLM is only consulted for the
    residual `unknown` cases and to phrase answers.
    """
    text = unicodedata.normalize("NFC", message).strip()
    f = fold(text)
    lang = detect_lang(text, default_lang)
    intent = Intent(lang=lang)

    if _GREET.match(f):
        intent.action = "greet"
        return intent
    if _HELP.search(f):
        intent.action = "help"
        return intent

    ords, last = _ordinals(f)
    intent.ordinals, intent.refers_last = tuple(ords), last

    price = parse_price(f, lang)
    intent.min_price, intent.max_price, intent.price_mode = price.min_price, price.max_price, price.mode

    spans = list(price.spans)
    rm = _RATING.search(f)
    if _HIGH_RATED.search(f):
        intent.min_rating = 4.0
    elif rm:
        val = rm.group(1) or rm.group(2)
        try:
            rating = float(val.replace(",", "."))
            if 1 <= rating <= 5:
                intent.min_rating = rating
                spans.append(rm.span())
        except ValueError:
            pass

    for key in sorted(COLORS, key=len, reverse=True):
        if re.search(rf"(?:\bmau\s+)?(?<![a-z]){re.escape(key)}(?![a-z])", f) and (key not in {"do", "be", "tim", "cam", "kem", "bac", "vang"} or re.search(rf"\bmau\s+{key}\b|\b{key}\s+(?:color|colour)\b", f) or lang == "en"):
            color = COLORS[key]
            if color not in intent.colors:
                intent.colors += (color,)
            break

    intent.audiences = parse_audiences(f)

    if known_brands:
        intent.brands = _find_brands(f, known_brands)

    terms, has_noun = _replace_glossary(f)
    intent.has_product_noun = has_noun

    intent.clear_filters = bool(_RESET.search(f))
    intent.exclude_shown = bool(_MORE.search(f))

    # Feedback needs an explicit reference to a shown result ("không thích sản phẩm 2", "thích cái cuối"); without
    # one the same words are a catalogue preference ("không thích màu đỏ") and keep their normal routing.
    if has_results and (ords or last):
        if _DISLIKE.search(f):
            intent.feedback = "dislike"
        elif _LIKE.search(f):
            intent.feedback = "like"

    # ---- action routing ------------------------------------------------------
    if intent.clear_filters and not has_noun:
        intent.action = "reset"
    elif _COMPARE.search(f) and (len(ords) >= 2 or has_results):
        intent.action = "compare"
    elif _EXPLAIN.search(f) and (ords or last or (has_results and not has_noun)):
        intent.action = "explain"
    elif _SIMILAR.search(f) and (ords or last or has_results) and not has_noun:
        intent.action = "similar"
    elif intent.feedback:
        intent.action = "feedback"
    elif has_image and not has_noun and not intent.has_constraints:
        intent.action = "search"
    elif _RECOMMEND.search(f) and not has_noun:
        intent.action = "recommend"
    elif has_results and not has_noun and (intent.has_constraints or intent.exclude_shown):
        intent.action = "refine"
    elif has_noun or has_image or len(f.split()) >= 2:
        intent.action = "search"
    else:
        intent.action = "unknown"

    # ---- semantic query --------------------------------------------------------
    color_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\bmau\s+\w+(?:\s+(?:duong|la|da troi|navy))?", f):
        color_spans.append(m.span())
    cleaned = _strip(text, f, spans + color_spans)
    cf = fold(cleaned)
    for m in reversed(list(_FILLERS.finditer(cf))):
        cleaned = cleaned[: m.start()] + " " + cleaned[m.end() :]
        cf = cf[: m.start()] + " " + cf[m.end() :]
    for rx in (_HIGH_RATED, _RECOMMEND, _MORE, _SIMILAR, _EXPLAIN, _COMPARE, _RESET):
        for m in reversed(list(rx.finditer(cf))):
            cleaned = cleaned[: m.start()] + " " + cleaned[m.end() :]
            cf = cf[: m.start()] + " " + cf[m.end() :]
    cf = fold(cleaned)
    for rx in (_BRAND_EXPLICIT, _REFINE_WORDS):
        for m in reversed(list(rx.finditer(cf))):
            cleaned = cleaned[: m.start()] + " " + cleaned[m.end() :]
            cf = cf[: m.start()] + " " + cf[m.end() :]
    for name in intent.brands:
        cleaned = re.sub(re.escape(name), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:duoi|tren|khoang|under|over|below|above|around|voi|va|nhung|cua|nhu|the nao)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[#\d]+", " ", cleaned) if intent.action in {"similar", "explain", "compare"} else cleaned
    cleaned = re.sub(r"[?!,.;:]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    intent.display_query = cleaned

    gloss = " ".join(dict.fromkeys(" ".join(terms).split()))
    if lang == "en":
        intent.query = cleaned
    else:
        intent.query = f"{gloss}. {cleaned}".strip(". ").strip() if gloss else cleaned
    missing = [c for c in intent.colors if c not in intent.query.lower()]
    if missing:
        intent.query = f"{' '.join(missing)} {intent.query}".strip()
    return intent
