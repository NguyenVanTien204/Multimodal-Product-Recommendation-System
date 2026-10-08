"""Latency / throughput benchmark for the running RAG service (hm/apps/rag).

    python hm/scripts/benchmark_rag.py --url http://127.0.0.1:8200 --label gpu --reps 10

Runs sequential scenarios (each rep uses a fresh session unless it is a follow-up
turn), reports end-to-end p50/p95 plus the per-stage `meta.timings_ms` the service
returns, a small concurrency test, and quality-of-response sanity counters.
Results are written to data/artifacts/rag_eval/benchmark_<label>.json.
Pure standard library: safe to run on any machine, but the SERVICE it measures is heavy.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TEXT_QUERIES = [
    "giày chạy bộ nam màu đen", "áo sơ mi trắng công sở", "túi xách nữ da thật", "đồng hồ nam dây da", "váy đầm dự tiệc",
    "black casual shoes for men", "waterproof hiking backpack", "gold necklace for women", "warm winter jacket", "kính râm phi công",
]
FILTERED_QUERIES = [
    "giày chạy bộ nam dưới 800k", "áo thun nam từ 200k đến 500k", "black leather wallet under 40", "túi xách nữ dưới 1 triệu",
    "running shoes rating 4 stars", "đồng hồ nam dưới 2 triệu",
]
HISTORY = ["B0BYX9MVHB", "B07B9WS7GN", "B07CFWNW2V"]  # real catalog SKUs (socks, blouse, water shoes)


def post(url: str, path: str, body: dict, timeout: float = 300) -> tuple[dict, float]:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data, (time.perf_counter() - t0) * 1000


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=30) as resp:
        return json.load(resp)


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(name: str, totals: list[float], stages: list[dict], responses: list[dict]) -> dict:
    stage_keys = sorted({k for s in stages for k in s})
    out = {
        "n": len(totals),
        "mean_ms": round(statistics.mean(totals), 1),
        "p50_ms": round(pct(totals, 0.5), 1),
        "p95_ms": round(pct(totals, 0.95), 1),
        "max_ms": round(max(totals), 1),
        "stages_mean_ms": {k: round(statistics.mean(s.get(k, 0) for s in stages), 1) for k in stage_keys},
        "avg_products": round(statistics.mean(len(r.get("products", [])) for r in responses), 2),
        "share_with_evidence": round(
            statistics.mean(any(p.get("evidence") for p in r.get("products", [])) for r in responses if r.get("products")) if any(r.get("products") for r in responses) else 0, 3
        ),
        "answer_sources": {src: sum(r.get("meta", {}).get("answer_source") == src for r in responses) for src in {r.get("meta", {}).get("answer_source") for r in responses}},
        "warnings": sorted({w for r in responses for w in r.get("warnings", [])}),
    }
    print(f"{name:28s} n={out['n']:3d} p50={out['p50_ms']:8.1f} p95={out['p95_ms']:8.1f} ms | stages {out['stages_mean_ms']}")
    return out


def run_scenario(name: str, reps: int, warmup: int, fn) -> dict:
    totals, stages, responses = [], [], []
    for i in range(reps + warmup):
        resp, ms = fn(i)
        if i >= warmup:
            totals.append(ms)
            stages.append(resp.get("meta", {}).get("timings_ms", {}))
            responses.append(resp)
    return summarize(name, totals, stages, responses)


def nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return {"nvidia_smi": out}
    except Exception:  # noqa: BLE001
        return {}


def sample_image_b64(url: str) -> str:
    """A real product photo if reachable, else a synthetic image (still exercises the encoder)."""
    try:
        r, _ = post(url, "/search", {"query": "leather handbag", "k": 1})
        img_url = r["products"][0]["image_url"]
        req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read()
        return base64.b64encode(raw).decode()
    except Exception:  # noqa: BLE001
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (512, 512), (120, 60, 30)).save(buf, "JPEG")
        return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    url, reps, warmup = args.url.rstrip("/"), args.reps, args.warmup

    health = get(url, "/health")
    print("health:", json.dumps(health)[:300])
    report: dict = {"label": args.label, "reps": reps, "health": health, "gpu_before": nvidia_smi()}

    img_b64 = sample_image_b64(url)
    scenarios: dict[str, dict] = {}

    scenarios["text_search"] = run_scenario("text_search", reps, warmup, lambda i: post(url, "/chat", {"message": TEXT_QUERIES[i % len(TEXT_QUERIES)]}))
    scenarios["text_search_filtered"] = run_scenario("text_search_filtered", reps, warmup, lambda i: post(url, "/chat", {"message": FILTERED_QUERIES[i % len(FILTERED_QUERIES)]}))
    scenarios["image_search"] = run_scenario("image_search", reps, warmup, lambda i: post(url, "/chat", {"message": "", "image_base64": img_b64}))

    def multi_turn(kind: str):
        def step(i: int):
            first, _ = post(url, "/chat", {"message": TEXT_QUERIES[i % 5]})  # setup turn, not timed
            sid = first["session_id"]
            pid = first["products"][0]["product_id"] if first["products"] else 1
            pid2 = first["products"][1]["product_id"] if len(first["products"]) > 1 else 2
            if kind == "refine_price":
                return post(url, "/chat", {"message": "rẻ hơn", "session_id": sid})
            if kind == "refine_colour":
                return post(url, "/chat", {"message": "màu trắng", "session_id": sid})
            if kind == "explain":
                return post(url, "/chat", {"message": "tại sao sản phẩm 1", "session_id": sid})
            if kind == "compare":
                return post(url, "/chat", {"message": "so sánh 1 và 2", "session_id": sid})
            return post(url, "/chat", {"message": "", "session_id": sid, "action": {"type": "similar", "product_id": pid}})
        return step

    for kind in ("refine_price", "refine_colour", "explain", "compare", "similar"):
        scenarios[kind] = run_scenario(kind, reps, warmup, multi_turn(kind))
    scenarios["recommend_with_history"] = run_scenario("recommend_with_history", reps, warmup, lambda i: post(url, "/chat", {"message": "gợi ý cho tôi", "history_skus": HISTORY}))
    scenarios["search_with_history"] = run_scenario("search_with_history", reps, warmup, lambda i: post(url, "/chat", {"message": TEXT_QUERIES[i % len(TEXT_QUERIES)], "history_skus": HISTORY}))
    report["scenarios"] = scenarios

    # concurrency: N parallel text searches, measure wall time and per-request latency
    n = max(args.concurrency * 3, 8)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: post(url, "/chat", {"message": TEXT_QUERIES[i % len(TEXT_QUERIES)]})[1], range(n)))
    wall = time.perf_counter() - t0
    report["concurrency"] = {
        "workers": args.concurrency, "requests": n, "wall_s": round(wall, 2), "throughput_rps": round(n / wall, 2),
        "p50_ms": round(pct(results, 0.5), 1), "p95_ms": round(pct(results, 0.95), 1),
    }
    print("concurrency:", report["concurrency"])
    report["gpu_after"] = nvidia_smi()
    report["health_after"] = get(url, "/health")

    out = ROOT / "data" / "artifacts" / "rag_eval" / f"benchmark_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
