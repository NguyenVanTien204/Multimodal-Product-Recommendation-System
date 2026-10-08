"""Generate the standalone Kaggle notebook that does the GPU-heavy part of the RAG chatbot.

    python hm/scripts/build_rag_notebooks.py          # writes hm/notebooks/kaggle_rag_reviews_and_eval.ipynb
    python hm/scripts/build_rag_notebooks.py --check  # also executes the definition-only cells as a smoke test

The notebook needs exactly ONE Kaggle Dataset (the 9 data files listed below) and no
access to the repository: every function it uses is copied verbatim from `hm/src/datn/`
at build time (via `ast`), so the notebook can never drift from the code that the
chatbot service runs.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "hm" / "src" / "datn"
OUT = ROOT / "hm" / "notebooks" / "kaggle_rag_reviews_and_eval.ipynb"

DATA_FILES = [
    "items.parquet",
    "train.parquet",
    "valid.parquet",
    "test.parquet",
    "candidate_interactions.parquet",
    "image_embeddings.npy",
    "image_embedding_metadata.parquet",
    "text_embeddings.npy",
    "text_embedding_metadata.parquet",
]


# ---------------------------------------------------------------------------------------
# Embedding real source into cells
# ---------------------------------------------------------------------------------------
def extract(rel_path: str, names: list[str]) -> str:
    """Verbatim source of the named top-level definitions/assignments, in file order."""
    path = SRC / rel_path
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    wanted, found, chunks = set(names), set(), []
    for node in tree.body:
        node_names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_names = [node.name]
        elif isinstance(node, ast.Assign):
            node_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            node_names = [node.target.id]
        hit = wanted.intersection(node_names)
        if not hit:
            continue
        found |= hit
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        # keep a comment line that sits directly above the node
        while first > 1 and lines[first - 2].lstrip().startswith("#"):
            first -= 1
        chunks.append("\n".join(lines[first - 1 : node.end_lineno]))
    missing = wanted - found
    if missing:
        raise KeyError(f"{rel_path}: not found {sorted(missing)}")
    return "\n\n\n".join(chunks)


def embedded(rel_path: str, names: list[str], note: str) -> str:
    return f"# {note}\n# Sao chép nguyên văn từ hm/src/datn/{rel_path} khi sinh notebook (hm/scripts/build_rag_notebooks.py).\n\n" + extract(rel_path, names)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, embedded_src: bool = False) -> dict:
    cell = {
        "cell_type": "code",
        "metadata": {"tags": ["embedded"]} if embedded_src else {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }
    return cell


# ---------------------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------------------
FILES_TABLE = "\n".join(f"| `{f}` |" for f in DATA_FILES)

CELLS: list[dict] = []
add = CELLS.append

# ---- 0. Intro ------------------------------------------------------------------------
add(md(rf"""
# Chatbot RAG — embed review, đo độ trễ và đánh giá truy xuất (Kaggle)

Notebook **độc lập**: chỉ cần **1 Kaggle Dataset** và không cần mã nguồn của repo. Mọi hàm dùng ở đây được sao chép nguyên văn từ `hm/src/datn/` lúc sinh notebook (các cell gắn thẻ `embedded`).

## Mục tiêu
1. **Embed review** bằng Jina CLIP v2 → `review_embeddings.npy` + `reviews_meta.parquet`.
2. **Đo độ trễ** encode truy vấn trên GPU và CPU (chẩn đoán ~20 giây/truy vấn trên CPU ở máy local).
3. **Đánh giá truy xuất** bằng tìm kiếm chính xác (exact) trên GPU.

## Chuẩn bị: một Dataset duy nhất
Tạo Kaggle Dataset (ví dụ `datn-rag-inputs`) chứa đủ 9 file dưới đây (đặt ở bất kỳ thư mục con nào), rồi *Add Input* vào notebook. Bật **GPU** và **Internet** (để tải model từ Hugging Face).

| File |
|---|
{FILES_TABLE}

## Kết quả (thư mục `/kaggle/working/rag_outputs/`)
`review_embeddings.npy`, `reviews_meta.parquet`, `manifest.json`, `results.json`. Tải về rồi ở local chạy:
```
datn-retrieval import-reviews --embeddings review_embeddings.npy --meta reviews_meta.parquet
```
"""))

# ---- 1. Setup ------------------------------------------------------------------------
add(md("## 1. Cài đặt môi trường"))
add(md(
    "Ghim `transformers==4.46.3` — bản đã kiểm chứng ở local (vector tính lại khớp vector catalog, cosine 0,9999). "
    "Không dùng `transformers 5.x`: notebook embedding text cũ phải vá lỗi buffer RoPE bị ghi đè bằng dữ liệu rác."
))
add(code(r"""
!pip install -q "transformers==4.46.3" "tokenizers>=0.20,<0.21" einops timm polars pyarrow tqdm
"""))
add(md("Vá nhỏ cho Pillow mới (thiếu `PIL._typing._Ink`), cùng lỗi đã gặp ở các notebook embedding trước."))
add(code(r"""
import sys, types, typing
try:
    import PIL._typing
    if not hasattr(PIL._typing, "_Ink"):
        PIL._typing._Ink = typing.Any
except Exception:
    mod = types.ModuleType("PIL._typing"); mod._Ink = typing.Any; sys.modules["PIL._typing"] = mod
print("PIL patch ok")
"""))

add(md("## 2. Cấu hình"))
add(code(r"""
import os, json, time, hashlib, platform, gc, re, random, subprocess, threading, logging, unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

SEED = 20260930
PER_PRODUCT = int(os.getenv("RAG_PER_PRODUCT", 4))   # review được embed cho mỗi sản phẩm (khớp cấu hình local)
BATCH = int(os.getenv("RAG_BATCH", 128))             # giảm còn 64 nếu hết VRAM
CHUNK = 8192                                         # số review mỗi checkpoint
EVAL_N = int(os.getenv("RAG_EVAL_N", 1000))          # số truy vấn review->sản phẩm
RUN_CPU_BENCH = os.getenv("RAG_CPU_BENCH", "1") == "1"

ON_KAGGLE = Path("/kaggle/input").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("./kaggle_working")
CKPT, OUT, DATA = WORK / "ckpt", WORK / "rag_outputs", WORK / "data"
for d in (CKPT, OUT, DATA):
    d.mkdir(parents=True, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
"""))

add(md("## 3. Tìm dữ liệu (một nguồn duy nhất)"))
add(md(
    "Notebook tìm thư mục Dataset chứa `items.parquet`, rồi yêu cầu **tất cả** file còn lại nằm trong đúng Dataset đó "
    "(đặt `RAG_INPUT_DIR` để chỉ định thủ công). Nếu thiếu file nào, ô này báo rõ tên file và dừng."
))
add(md("**3.1 Danh sách file bắt buộc**"))
add(code(rf"""
REQUIRED = {DATA_FILES!r}
# Tên dự phòng: Kaggle đôi khi thêm hậu tố " (1)" khi tải file trùng tên lên.
ALIASES = {{"image_embeddings.npy": "image_embeddings (1).npy", "image_embedding_metadata.parquet": "image_embedding_metadata (1).parquet",
           "text_embeddings.npy": "text_embeddings (1).npy", "text_embedding_metadata.parquet": "text_embedding_metadata (1).parquet"}}

def files_under(root: Path) -> dict:
    found = {{}}
    for name in REQUIRED:
        p = next(iter(root.rglob(name)), None) or next(iter(root.rglob(ALIASES.get(name, name))), None)
        if p is not None:
            found[name] = p
    return found
"""))
add(md(
    "**3.2 Xác định thư mục Dataset**\n"
    "Bắt đầu từ thư mục chứa `items.parquet`, đi ngược lên và dừng ở thư mục thấp nhất chứa đủ 9 file. "
    "Không bao giờ lên tới `/kaggle/input`, nên các file rải ở nhiều Dataset khác nhau sẽ bị từ chối."
))
add(code(r"""
def locate_dataset():
    if os.getenv("RAG_INPUT_DIR"):
        root = Path(os.environ["RAG_INPUT_DIR"]); return root, files_under(root)
    stop = Path("/kaggle/input")
    for base in ("/kaggle/input", "data", "../data"):
        b = Path(base)
        hit = next(iter(b.rglob("items.parquet")), None) if b.exists() else None
        if hit is None:
            continue
        best = None
        for anc in [hit.parent, *hit.parents]:
            if anc == stop or not (anc == b or b in anc.parents):
                break
            found = files_under(anc)
            best = (anc, found)
            if len(found) == len(REQUIRED):
                break
        if best:
            return best
    raise FileNotFoundError("Không thấy items.parquet. Hãy Add Input Dataset chứa 9 file dữ liệu (xem ô hướng dẫn đầu notebook).")

DATASET_ROOT, FOUND = locate_dataset()
missing = [n for n in REQUIRED if n not in FOUND]
if missing:
    raise FileNotFoundError(f"Dataset {DATASET_ROOT} thiếu: {missing}. Cả 9 file phải nằm trong CÙNG MỘT Dataset.")
print("Dataset:", DATASET_ROOT)
"""))
add(md("**3.3 Liên kết file vào thư mục làm việc** (để code nhúng dùng đường dẫn cố định `DATA/<tên file>`)"))
add(code(r"""
for name, src in FOUND.items():
    link = DATA / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src.resolve())
for name in REQUIRED:
    print(f"  {name:36s} {(DATA / name).stat().st_size / 2**20:9.1f} MiB")
"""))

add(md("## 4. Kiểm tra môi trường"))
add(code(r"""
import transformers
print("python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__)
assert transformers.__version__.startswith("4.4"), "Cần transformers 4.4x (xem ô cài đặt); bản 5.x làm hỏng buffer RoPE của Jina."
assert torch.cuda.is_available(), "Cần bật GPU (Settings -> Accelerator)."
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB")
"""))

# ---- 5. Embedded source --------------------------------------------------------------
add(md(
    "## 5. Mã nguồn nhúng từ `hm/src/datn/`\n"
    "Các cell dưới đây là bản sao nguyên văn của code mà dịch vụ chatbot dùng, nên số liệu đo ở đây áp dụng đúng cho hệ thống thật."
))
add(md("**5.1 Nạp và làm sạch review** (`retrieval/reviews.py`) — cùng logic với `datn-retrieval index-reviews`, nên `review_id` khớp nhau."))
add(code(embedded(
    "retrieval/reviews.py",
    ["REVIEW_FILES", "MIN_REVIEW_CHARS", "MAX_REVIEW_CHARS", "MAX_REVIEWS_PER_PRODUCT", "_COLUMNS", "catalog_product_ids", "load_reviews"],
    "Nạp review",
), embedded_src=True))
add(md("**5.2 Chọn review tiêu biểu và dựng chuỗi để embed** (`retrieval/reviews.py`)"))
add(code(embedded("retrieval/reviews.py", ["review_stats", "select_reviews", "review_text"], "Chọn review"), embedded_src=True))

add(md("**5.3 Bộ mã hoá truy vấn** (`retrieval/encoder.py`) — Jina CLIP v2, tải lười và an toàn luồng. Cell này là một lớp thư viện nguyên vẹn nên khá dài; có thể thu gọn."))
add(code(embedded("retrieval/encoder.py", ["log", "MODEL_ID", "_l2_normalize", "JinaClipEncoder"], "Bộ mã hoá Jina CLIP v2"), embedded_src=True))

add(md("**5.4 Hằng số tên vector/payload** — tối giản từ `retrieval/schema.py`, chỉ giữ các tên mà `Hit` và trọng số RRF tham chiếu."))
add(code(
    "class S:\n"
    "    VECTOR_TEXT = \"text\"\n"
    "    VECTOR_IMAGE = \"image\"\n"
    "    P_ITEM_ID = \"item_id\"\n"
    "    P_TITLE = \"title\"\n",
    embedded_src=True,
))
add(md("**5.5 Kết quả truy xuất và hợp nhất xếp hạng** (`retrieval/filters.py`, `retrieval/search.py`) — `Hit` và weighted RRF đúng như dịch vụ thật."))
add(code(embedded("retrieval/filters.py", ["Hit"], "Kết quả truy xuất"), embedded_src=True))
add(code(embedded("retrieval/search.py", ["RRF_K", "DEFAULT_WEIGHTS", "rrf_fuse"], "Hợp nhất xếp hạng (weighted RRF)"), embedded_src=True))

add(md("**5.6 Glossary tiếng Việt → tiếng Anh** (`agent/intent.py`) — dùng để kiểm tra mở rộng truy vấn ở mục 11."))
add(code(embedded("agent/intent.py", ["_base", "fold", "_EN_NOUNS"], "Chuẩn hoá tiếng Việt"), embedded_src=True))
add(code(embedded("agent/intent.py", ["GLOSSARY", "MODIFIERS"], "Từ điển sản phẩm / bổ nghĩa (VI -> EN)"), embedded_src=True))
add(code(
    embedded("agent/intent.py", ["_replace_glossary"], "Tìm cụm từ trong glossary")
    + "\n\n\ndef expand_query(q: str) -> str:\n"
    "    # Giống nhánh tạo truy vấn của parse_intent(): '<từ khoá tiếng Anh>. <câu gốc>'\n"
    "    terms, _ = _replace_glossary(fold(q))\n"
    "    gloss = \" \".join(dict.fromkeys(\" \".join(terms).split()))\n"
    "    return f\"{gloss}. {q}\" if gloss else q\n",
    embedded_src=True,
))

# ---- 6. Model ------------------------------------------------------------------------
add(md("## 6. Nạp Jina CLIP v2"))
add(code(r"""
t0 = time.time()
enc = JinaClipEncoder(device="cuda").load()
print(f"Đã nạp model trên {enc.device} sau {time.time() - t0:.0f}s")
"""))

add(md("## 7. Nạp catalog và vector đã có"))
add(code(r"""
items = pl.read_parquet(DATA / "items.parquet")
text_meta = pl.read_parquet(DATA / "text_embedding_metadata.parquet")
img_meta = pl.read_parquet(DATA / "image_embedding_metadata.parquet")
text_emb = np.load(DATA / "text_embeddings.npy", mmap_mode="r")
img_emb = np.load(DATA / "image_embeddings.npy", mmap_mode="r")
assert len(items) == len(text_meta) == len(img_meta) == text_emb.shape[0] == img_emb.shape[0], "Số dòng items/embedding không khớp"
print(f"{len(items):,} sản phẩm | vector text {text_emb.shape} | vector ảnh {img_emb.shape}")
"""))

add(md(
    "### 7.1 Tự kiểm trước khi tốn thời gian embed\n"
    "Mã hoá lại 16 sản phẩm bằng đúng template của notebook embedding text rồi so với vector đã lưu. "
    "Nếu cosine < 0,99 nghĩa là môi trường làm hỏng model → **dừng ngay**, tránh tạo 300k vector sai."
))
add(code(r"""
def prepare_text(r):   # đúng template của legacy/notebooks/kaggle_text_embeddings.ipynb
    parts = []
    for key, label in (("title", "Title"), ("category", "Category"), ("brand", "Brand"), ("features", "Features"), ("description", "Description")):
        if r[key]:
            parts.append(f"{label}: {r[key]}")
    return " | ".join(parts).strip() or "unknown"

row_of = {sku: i for i, sku in enumerate(text_meta["item_id"].to_list())}
probe = items.sample(16, seed=SEED).to_dicts()
fresh = enc.encode_passages([prepare_text(r) for r in probe])
stored = [np.asarray(text_emb[row_of[r["item_id"]]], dtype=np.float32) for r in probe]
cos = [float(f @ (s / np.linalg.norm(s))) for f, s in zip(fresh, stored)]
ALIGN = {"min_cosine": min(cos), "mean_cosine": float(np.mean(cos)), "n": len(cos)}
print("Căn chỉnh với vector catalog:", ALIGN)
assert ALIGN["min_cosine"] > 0.99, "Vector tính lại KHÔNG khớp vector catalog -> môi trường sai, không embed review."
"""))

# ---- 8. Reviews ----------------------------------------------------------------------
add(md("## 8. Chọn và embed review"))
add(md("### 8.1 Chọn review"))
add(code(r"""
sku_to_pid = catalog_product_ids(DATA / "items.parquet")      # id sản phẩm trong shop = số dòng trong items.parquet + 1
all_reviews = load_reviews(DATA, set(sku_to_pid))
reviews = select_reviews(all_reviews, PER_PRODUCT)
n = reviews.height
print(f"review hợp lệ (đã khử trùng lặp): {all_reviews.height:,} | được chọn: {n:,} | sản phẩm có review: {reviews['item_id'].n_unique():,}")
print("Mặc định PER_PRODUCT=4 phải cho 514,744 -> 295,383")
"""))
add(code(r"""
texts = [review_text(t, x) for t, x in zip(reviews["review_title"].to_list(), reviews["review_text"].to_list())]
lens = np.array([len(t) for t in texts])
print("độ dài ký tự min/mean/max:", lens.min(), int(lens.mean()), lens.max())
print("ví dụ:", texts[0][:200])
"""))

add(md("### 8.2 Embed (có checkpoint theo chunk — nếu phiên bị ngắt, chạy lại ô này sẽ tiếp tục)"))
add(code(r"""
t0 = time.time(); done_new = 0
for ci, start in enumerate(tqdm(range(0, n, CHUNK), desc="chunks")):
    f = CKPT / f"chunk_{ci:05d}.npy"
    if f.exists():
        continue
    emb = enc.encode_passages(texts[start:start + CHUNK], batch_size=BATCH)
    np.save(f, emb.astype(np.float16))
    done_new += len(emb)
    if ci % 5 == 0:
        print(f"chunk {ci}: {done_new / (time.time() - t0):.0f} review/s")
ENCODE_SECONDS = time.time() - t0
print(f"Xong: {done_new:,} review mới trong {ENCODE_SECONDS / 60:.1f} phút")
"""))

add(md("### 8.3 Ghép file đầu ra"))
add(code(r"""
chunks = sorted(CKPT.glob("chunk_*.npy"))
assert len(chunks) == (n + CHUNK - 1) // CHUNK, "Thiếu chunk - chạy lại ô embed"
emb_path = OUT / "review_embeddings.npy"
final = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, 1024))
pos = 0
for f in chunks:
    a = np.load(f); final[pos:pos + len(a)] = a; pos += len(a)
final.flush(); assert pos == n
sample = np.asarray(final[:2000], dtype=np.float32)
assert np.isfinite(sample).all() and abs(np.linalg.norm(sample, axis=1).mean() - 1) < 0.01, "vector không chuẩn hoá / có NaN"
del final
print("đã ghi", emb_path, f"({emb_path.stat().st_size / 2**20:.0f} MiB)")
"""))
add(code(r"""
meta = reviews.select(
    "review_id", "item_id", "rating", "helpful_vote", "verified_purchase", "review_title", "review_text",
).with_columns(
    pl.col("item_id").replace_strict(sku_to_pid, return_dtype=pl.Int64).alias("product_id"),
    pl.col("review_text").str.slice(0, 600),
).select("review_id", "product_id", "item_id", "rating", "helpful_vote", "verified_purchase", "review_title", "review_text")
meta.write_parquet(OUT / "reviews_meta.parquet")
print(meta.head(3))
"""))
add(code(r"""
def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()

MANIFEST = {
    "model": "jinaai/jina-clip-v2", "task": "default (document side), same as the catalog text vectors",
    "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
    "per_product": PER_PRODUCT, "n_reviews": n, "n_reviews_before_selection": all_reviews.height,
    "dtype": "float16", "shape": [n, 1024], "encode_seconds_this_session": ENCODE_SECONDS, "batch_size": BATCH,
    "alignment_check": ALIGN,
    "sha256": {"review_embeddings.npy": sha256(emb_path), "reviews_meta.parquet": sha256(OUT / "reviews_meta.parquet")},
}
(OUT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
print(json.dumps(MANIFEST, indent=2))
"""))

# ---- 9. GPU matrices -----------------------------------------------------------------
add(md(
    "## 9. Tìm kiếm chính xác trên GPU\n"
    "Thay Qdrant bằng tích vô hướng chính xác, cùng công thức RRF và trọng số như dịch vụ thật ⇒ số liệu là **cận trên** của chất lượng vector (không có sai số ANN)."
))
add(md("### 9.1 Ma trận sản phẩm (hàng *i* ⇔ sản phẩm id *i+1*, sắp lại theo thứ tự `items.parquet`)"))
add(code(r"""
dev = "cuda"

def to_items_order(emb, meta_df):
    row = {sku: i for i, sku in enumerate(meta_df["item_id"].to_list())}
    idx = np.array([row[s] for s in items["item_id"].to_list()])
    m = torch.from_numpy(np.asarray(emb, dtype=np.float32)[idx]).to(dev)
    return torch.nn.functional.normalize(m, dim=1).half()

M_TEXT = to_items_order(text_emb, text_meta)
M_IMG = to_items_order(img_emb, img_meta)
TITLES = items["title"].fill_null("").to_list()
print("ma trận:", tuple(M_TEXT.shape), tuple(M_IMG.shape), f"| VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
"""))
add(md("### 9.2 Hàm tìm kiếm và metric"))
add(code(r"""
def topk(vec: np.ndarray, M: torch.Tensor, k: int = 100):
    scores = M @ torch.from_numpy(vec).to(dev).half()
    s, i = torch.topk(scores, k)
    return [Hit(int(j) + 1, {}, float(x)) for j, x in zip(i.tolist(), s.tolist())]

def fused(text_vec=None, image_vec=None, k=50, w_img_for_text=None):
    rank, w = {}, {}
    if text_vec is not None:
        for using, M in ((S.VECTOR_TEXT, M_TEXT), (S.VECTOR_IMAGE, M_IMG)):
            name = f"text_query->{using}"; rank[name] = topk(text_vec, M); w[name] = DEFAULT_WEIGHTS["text_query"][using]
        if w_img_for_text is not None:
            w[f"text_query->{S.VECTOR_IMAGE}"] = w_img_for_text
    if image_vec is not None:
        for using, M in ((S.VECTOR_IMAGE, M_IMG), (S.VECTOR_TEXT, M_TEXT)):
            name = f"image_query->{using}"; rank[name] = topk(image_vec, M); w[name] = DEFAULT_WEIGHTS["image_query"][using]
    return rrf_fuse(rank, w)[:k]

def rank_of(hits, pid):
    return next((i for i, h in enumerate(hits, 1) if h.product_id == pid), None)

def metrics(ranks, ks=(1, 5, 10, 50)):
    n_ = len(ranks)
    out = {f"HR@{k}": round(sum(r is not None and r <= k for r in ranks) / n_, 4) for k in ks}
    out["MRR@50"] = round(sum(1 / r for r in ranks if r is not None and r <= 50) / n_, 4)
    return out
"""))

# ---- 10. Latency ---------------------------------------------------------------------
add(md(
    "## 10. Độ trễ encode truy vấn\n"
    "GPU trước, rồi CPU (fp32 / bf16) để xác định nguyên nhân độ trễ ~20 giây/truy vấn đã đo trên laptop."
))
add(code(r"""
QUERIES = ["giày chạy bộ nam màu đen", "áo sơ mi trắng công sở", "black casual shoes under 100", "túi xách nữ da thật", "waterproof hiking backpack",
           "đồng hồ nam dây da", "summer floral dress", "nhẫn bạc nữ", "warm winter jacket", "kính râm phi công"] * 5

def stats(times):
    a = np.array(times) * 1000
    return {"n": len(a), "mean_ms": round(float(a.mean()), 1), "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1)}

def time_gpu(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return ts

LAT = {"gpu": torch.cuda.get_device_name(0)}
for q in QUERIES[:3]:
    enc.encode_query([q])            # warm-up
it = iter(QUERIES)
LAT["gpu_encode_text_query_single"] = stats(time_gpu(lambda: enc.encode_query([next(it)]), len(QUERIES) - 1))
print(LAT["gpu_encode_text_query_single"])
"""))
add(code(r"""
LAT["gpu_encode_text_query_batch32_total_ms"] = round(time_gpu(lambda: enc.encode_query(QUERIES[:32]), 3)[-1] * 1000, 1)

from PIL import Image
imgs = [Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)) for _ in range(10)]
enc.encode_images(imgs[:2])          # warm-up
ii = iter(imgs)
LAT["gpu_encode_image_single"] = stats(time_gpu(lambda: enc.encode_images([next(ii)]), len(imgs)))
print({k: v for k, v in LAT.items() if k != "gpu"})
"""))
add(code(r"""
qv = enc.encode_query(["black shoes"])[0]
LAT["gpu_exact_search_fused_text_query"] = stats(time_gpu(lambda: fused(text_vec=qv, k=10), 50))
print(LAT["gpu_exact_search_fused_text_query"])
"""))
add(md("### 10.1 CPU — thông tin máy"))
add(code(r"""
cpu_info = {}
if RUN_CPU_BENCH:
    try:
        cpu_info["lscpu_model"] = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    cpu_info.update({"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "mkldnn": torch.backends.mkldnn.is_available()})
    a = torch.randn(2048, 2048); torch.mm(a, a)
    t = time.perf_counter(); [torch.mm(a, a) for _ in range(5)]
    cpu_info["matmul_2048_fp32_gflops"] = round(2 * 2048**3 / ((time.perf_counter() - t) / 5) / 1e9, 1)
    print(cpu_info)
else:
    print("bỏ qua benchmark CPU (RAG_CPU_BENCH=0)")
"""))
add(md("### 10.2 CPU — encode truy vấn (nạp bản thứ hai trên CPU, cần ~4 GB RAM; đây là cấu hình mà container `rag` mặc định dùng)"))
add(code(r"""
if RUN_CPU_BENCH:
    enc_cpu = JinaClipEncoder(device="cpu").load()
    def bench(label, n_q=6):
        enc_cpu.encode_query(["warm up"])
        ts = []
        for q in QUERIES[:n_q]:
            t = time.perf_counter(); enc_cpu.encode_query([q]); ts.append(time.perf_counter() - t)
        cpu_info[label] = stats(ts); print(label, cpu_info[label])
    bench("cpu_fp32_single_query")
    for th in (1, 4, 8):
        if th <= (os.cpu_count() or 1):
            torch.set_num_threads(th); bench(f"cpu_fp32_threads_{th}", n_q=4)
    torch.set_num_threads(os.cpu_count() or 1)
"""))
add(code(r"""
if RUN_CPU_BENCH:
    try:
        enc_cpu._model.to(torch.bfloat16); bench("cpu_bf16_single_query", n_q=4)
    except Exception as exc:
        cpu_info["cpu_bf16_error"] = str(exc)[:200]; print("bf16 lỗi:", exc)
    LAT["cpu"] = cpu_info
    del enc_cpu; gc.collect()
"""))

# ---- 11. Evaluation ------------------------------------------------------------------
add(md(
    "## 11. Đánh giá truy xuất\n"
    "- **review → sản phẩm**: dùng nội dung một review làm truy vấn, kiểm tra có tìm lại đúng sản phẩm được review không (mô phỏng người dùng mô tả món đồ bằng ngôn ngữ tự nhiên).\n"
    "- **căn chỉnh chéo**: vector ảnh của sản phẩm → tìm vector text của chính nó và ngược lại.\n"
    "- **truy vấn tiếng Việt**: precision@10 theo từ khoá tiêu đề, so câu thô với câu đã mở rộng glossary."
))
add(md("### 11.1 review → sản phẩm: chỉ-text và chỉ-ảnh (chéo)"))
add(code(r"""
rng = np.random.default_rng(SEED)
cand = [i for i, t in enumerate(reviews["review_text"].to_list()) if 60 <= len(t) <= 400]
pick = rng.choice(cand, size=min(EVAL_N, len(cand)), replace=False)
q_texts = [reviews["review_text"][int(i)] for i in pick]
targets = [sku_to_pid[reviews["item_id"][int(i)]] for i in pick]
qvecs = enc.encode_query(q_texts)

half = len(pick) // 2          # nửa đầu = dev (chọn trọng số), nửa sau = test (báo cáo)
ranks = {"text_only": [], "image_only": []}
for v, tgt in zip(qvecs, targets):
    ranks["text_only"].append(rank_of(topk(v, M_TEXT, 50), tgt))
    ranks["image_only"].append(rank_of(topk(v, M_IMG, 50), tgt))
print("text_only (test):", metrics(ranks["text_only"][half:]))
print("image_only (test):", metrics(ranks["image_only"][half:]))
"""))
add(md("### 11.2 Hợp nhất RRF — chọn trọng số ảnh trên dev, báo cáo trên test"))
add(code(r"""
W_GRID = [0.0, 0.4, 0.8, 1.2]
fused_ranks = {w: [rank_of(fused(text_vec=v, k=50, w_img_for_text=w), tgt) for v, tgt in zip(qvecs, targets)] for w in W_GRID}
dev_mrr = {w: metrics(r[:half])["MRR@50"] for w, r in fused_ranks.items()}
best_w = max(dev_mrr, key=dev_mrr.get)
default_w = DEFAULT_WEIGHTS["text_query"][S.VECTOR_IMAGE]
print("MRR@50 trên dev theo trọng số ảnh:", dev_mrr, "-> chọn", best_w, "| mặc định của dịch vụ:", default_w)
"""))
add(code(r"""
REVIEW2ITEM = {
    "n_queries": len(pick), "n_dev": half, "n_test": len(pick) - half,
    "chosen_image_weight_on_dev": best_w, "service_default_image_weight": default_w,
    "text_only_test": metrics(ranks["text_only"][half:]),
    "image_only_cross_modal_test": metrics(ranks["image_only"][half:]),
    "fused_default_weight_test": metrics(fused_ranks[default_w][half:]) if default_w in fused_ranks else None,
    "fused_best_dev_weight_test": metrics(fused_ranks[best_w][half:]),
    "fused_by_weight_test": {str(w): metrics(r[half:]) for w, r in fused_ranks.items()},
}
print(json.dumps(REVIEW2ITEM, indent=2))
"""))
add(md("### 11.3 Căn chỉnh chéo ảnh ↔ text (2.000 sản phẩm ngẫu nhiên, xếp hạng trên toàn catalog)"))
add(code(r"""
ids = rng.choice(len(items), size=min(2000, len(items)), replace=False)

def self_ranks(src, dst):
    out = []
    for i in ids:
        s = dst @ src[int(i)]
        out.append(int((s > s[int(i)]).sum().item()) + 1)
    return np.array(out)

CROSS = {}
for name, (src, dst) in {"image_to_text": (M_IMG, M_TEXT), "text_to_image": (M_TEXT, M_IMG)}.items():
    r = self_ranks(src, dst)
    CROSS[name] = {"HR@1": float((r <= 1).mean()), "HR@10": float((r <= 10).mean()), "HR@50": float((r <= 50).mean()),
                   "median_rank": float(np.median(r)), "n": len(r)}
print(json.dumps(CROSS, indent=2))
"""))
add(md("### 11.4 Truy vấn tiếng Việt: câu thô và câu đã mở rộng glossary"))
add(md("Mỗi truy vấn kèm một regex mô tả loại sản phẩm đúng; điểm = tỉ lệ tiêu đề trong top-10 khớp regex."))
add(code(r"""
VI = [("giày chạy bộ nam", r"running|sneaker|shoe|trainer"), ("áo sơ mi trắng công sở", r"shirt|blouse|button"),
      ("túi xách nữ da thật", r"bag|handbag|purse|tote|satchel|clutch"), ("đồng hồ nam", r"watch"), ("nhẫn bạc", r"\bring"),
      ("dây chuyền vàng", r"necklace|pendant|chain"), ("váy đầm dự tiệc", r"dress|gown"),
      ("áo khoác mùa đông", r"jacket|coat|parka|hoodie|sweater"), ("quần jean nữ", r"jean|denim"), ("kính râm", r"sunglass|glasses|eyewear"),
      ("tất cotton", r"sock"), ("mũ lưỡi trai", r"cap|hat|beanie"), ("balo đi học", r"backpack|bag"),
      ("dép sandal nữ", r"sandal|slipper|flip|slide"), ("áo thun nam", r"t-shirt|tee|shirt"), ("đồ bơi nữ", r"swim|bikini|bathing"),
      ("thắt lưng da", r"belt"), ("găng tay", r"glove|mitten"), ("khăn quàng cổ", r"scarf|scarves|shawl|wrap"),
      ("đồ ngủ", r"pajama|pyjama|sleep|nightgown|robe"), ("bông tai", r"earring|stud"), ("vòng tay", r"bracelet|bangle|cuff"),
      ("quần short thể thao", r"short"), ("giày cao gót", r"heel|pump|stiletto|wedge")]

def p_at_10(query, pattern):
    top = fused(text_vec=enc.encode_query([query])[0], k=10)
    return float(np.mean([bool(re.search(pattern, TITLES[h.product_id - 1], re.I)) for h in top]))

rows = [{"query": q, "expanded": expand_query(q), "p10_raw_vi": p_at_10(q, pat), "p10_expanded": p_at_10(expand_query(q), pat)} for q, pat in VI]
df_vi = pl.DataFrame(rows)
print(df_vi.select("query", "expanded", "p10_raw_vi", "p10_expanded"))
"""))
add(code(r"""
VIET = {
    "n_queries": len(rows),
    "mean_p10_raw_vietnamese": float(df_vi["p10_raw_vi"].mean()),
    "mean_p10_glossary_expanded": float(df_vi["p10_expanded"].mean()),
    "per_query": rows,
}
print({k: v for k, v in VIET.items() if k != "per_query"})
"""))

# ---- 12. Output ----------------------------------------------------------------------
add(md("## 12. Ghi kết quả"))
add(code(r"""
RESULTS = {
    "seed": SEED, "encoder_device": "cuda", "alignment_check": ALIGN, "latency": LAT,
    "review_to_product": REVIEW2ITEM, "cross_modal_alignment": CROSS, "vietnamese_queries": VIET,
    "notes": [
        "exact GPU search (no ANN error): an upper bound for the Qdrant HNSW numbers",
        "review->product uses a review of the product as the query: a proxy for natural-language search, not human relevance judgements",
        "filter compliance is enforced inside Qdrant and is evaluated locally with datn-eval-retrieval",
    ],
}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
"""))
add(code(r"""
print("Các file trong", OUT)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 2**20:8.1f} MiB")
print("\nTải về 4 file trên rồi chạy: datn-retrieval import-reviews --embeddings review_embeddings.npy --meta reviews_meta.parquet")
"""))


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def smoke_check() -> None:
    """Execute the definition-only (`embedded`) cells and check basic behaviour."""
    ns: dict = {}
    pre = "import os, json, time, hashlib, platform, gc, re, random, subprocess, threading, logging, unicodedata\n" \
          "from pathlib import Path\nfrom dataclasses import dataclass, field\nfrom typing import Any, Sequence\n" \
          "import numpy as np\nimport polars as pl\n"
    exec(pre, ns)
    for cell in CELLS:
        if cell["cell_type"] == "code" and "embedded" in cell["metadata"].get("tags", []):
            exec("".join(cell["source"]), ns)
    assert ns["expand_query"]("giày chạy bộ nam").startswith("running shoes")
    hits = [ns["Hit"](i, {}, 0.5) for i in (1, 2, 3)]
    fused = ns["rrf_fuse"]({"a": hits, "b": hits[::-1]}, {"a": 1.0, "b": 1.0})
    assert {h.product_id for h in fused} == {1, 2, 3}
    print("smoke check ok: embedded cells execute and behave")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    code_cells = sum(c["cell_type"] == "code" for c in CELLS)
    longest = max(len(c["source"]) for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {OUT} ({len(CELLS)} cells: {code_cells} code, {len(CELLS) - code_cells} markdown; longest code cell {longest} lines)")
    if args.check:
        smoke_check()


if __name__ == "__main__":
    main()
