"""Thăm dò "luật phục vụ" của tower H&M ở lúc SUY LUẬN (không train lại).

Điểm = điểm tower + pop_weight·log(1 + số lần bán 7 ngày trước cutoff) + cold_bonus·[item cold]; loại item không bán trong `active_days` ngày.
Nạp phần định nghĩa (mô hình, hàm đánh giá, thống kê bán hàng) từ hm/notebooks/02_hm_retrieval.ipynb, KHÔNG chạy huấn luyện.
Thống kê bán hàng dùng item_daily_counts.parquet nếu có trong --data (hoặc HM_GLOBAL_STATS), nếu không dùng mẫu 50k khách.

Chạy:  python hm/scripts/probe_serving_rule.py --ckpt hm/checkpoints/best_retrieval.pt [--data data/hm] [--windows valid test]
Cần GPU nhỏ (RTX 3050 4GB đủ), ~1–2 phút. Kết quả đã dẫn trong hm/docs/03 (mục E6, E8).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="best_retrieval.pt hoặc best_retrieval_refit.pt")
parser.add_argument("--data", default="data/hm")
parser.add_argument("--windows", nargs="+", default=["valid", "test"], choices=["valid", "test"])
parser.add_argument("--tower-is-refit", action="store_true", help="ckpt là bản refit (cold theo degree của chính ckpt)")
args = parser.parse_args()

os.environ.setdefault("HM_DATA_DIR", str(Path(args.data).resolve()))
os.environ.setdefault("HM_OUTPUT_DIR", str(ROOT / "data" / "artifacts" / "hm" / "probe_out"))

nb = json.loads((ROOT / "hm/notebooks/02_hm_retrieval.ipynb").read_text(encoding="utf8"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
exec(compile(code[: code.index('epochs = int(os.getenv("HM_EPOCHS"')], "nb02_prefix", "exec"), globals())   # noqa: S102

ck = torch.load(args.ckpt, map_location=device, weights_only=False)
model = Tower().to(device)
model.load_state_dict(ck["state"], strict=False)   # bản xuất gọn không có buffer text/image -> giữ buffer từ embedding đã nạp
model.eval()
degree = np.asarray(ck["degree"])
model.set_degree(degree, 0)
print("checkpoint:", args.ckpt, "| best_epoch", ck.get("best_epoch"), "| serving trong ckpt:", ck.get("serving"))

@torch.no_grad()
def rank(context, targets, window, active, bonus, pop_w, topk=1000):
    users = [u for u in targets if context.get(u)]
    serving = make_serving(window, active, bonus, pop_w)
    table = model.item_table()
    out = {}
    for s in range(0, len(users), 256):
        chunk = users[s:s + 256]
        top = torch.topk(score_matrix(model, context, chunk, table, serving), topk, 1).indices.cpu().numpy()
        out.update({u: r.tolist() for u, r in zip(chunk, top)})
    return out

contexts = {"valid": ({u: train.get(u, []) + selection_window.get(u, []) for u in valid}, valid),
            "test": ({u: train.get(u, []) + selection_window.get(u, []) + valid.get(u, []) for u in test}, test)}
cold = lambda _, i: degree[i] == 0
GRID = [(0, 0, 0), (7, 0, 0), (14, 0, 0), (28, 0, 0), (56, 0, 0),            # chỉ lọc item đang bán
        (14, 0.1, 0), (28, 0.1, 0),                                           # + cộng điểm item cold
        (14, 0, 0.02), (14, 0, 0.05), (14, 0, 0.1),                           # + trọng số bán chạy
        (14, 0.1, 0.05), (14, 0.1, 0.1), (14, 0.1, 0.2), (0, 0.1, 0.1), (28, 0.1, 0.1), (0, 0.2, 0.1)]
for window in args.windows:
    ctx, tg = contexts[window]
    print(f"\n== {window} | thống kê bán hàng: {STATS_SOURCE}")
    print(f"{'active/bonus/pop_w':<20}{'Hit@12':>8}{'Hit@50':>8}{'Hit@100':>9}{'Rec@100':>9}{'cold H@100':>12}")
    for active, bonus, pop_w in GRID:
        r = rank(ctx, tg, window, active, bonus, pop_w)
        a = full_ranking_metrics(r, tg, ks=(12, 50, 100))
        c = full_ranking_metrics(r, tg, ks=(12, 50, 100), target_filter=cold)
        print(f"{f'{active}d/{bonus}/{pop_w}':<20}{a['HitRate@12']:>8.4f}{a['HitRate@50']:>8.4f}{a['HitRate@100']:>9.4f}{a['Recall@100']:>9.4f}{c['HitRate@100']:>12.4f}")
