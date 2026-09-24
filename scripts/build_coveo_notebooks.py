from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "coveo"


def notebook(title: str, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f"# {title}\n\nPipeline Coveo v1 — mọi output được version hóa và không ghi đè."), *cells])
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return nb


COMMON = """from pathlib import Path
import os, json, yaml

def find_repo():
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        if (root / 'pyproject.toml').exists(): return root
    raise FileNotFoundError('Không tìm thấy pyproject.toml')

REPO = find_repo()
os.chdir(REPO)
cfg = yaml.safe_load((REPO / 'configs/coveo.yaml').read_text(encoding='utf-8'))
PROFILE = os.getenv('COVEO_PROFILE', cfg['project']['profile'])
print('repo=', REPO, 'profile=', PROFILE)"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_get_data.ipynb": notebook("00 — Lấy dữ liệu Coveo", [
            nbf.v4.new_markdown_cell("""Dữ liệu đầy đủ yêu cầu đăng ký và chấp nhận điều khoản tại repository chính thức Coveo. Notebook không vượt qua bước này. Nguồn được ưu tiên: Kaggle mounted dataset → ZIP đã tải hợp lệ → URL tạm được cấp quyền → sample GitHub chính thức."""),
            nbf.v4.new_code_cell("%pip install -q -e \".[coveo]\""), nbf.v4.new_code_cell(COMMON),
            nbf.v4.new_code_cell("""from datn.data.coveo import acquire_coveo
raw_dir = REPO / cfg['paths']['raw_dir']
zip_value = os.getenv('COVEO_ZIP_PATH')
report = acquire_coveo(
    raw_dir,
    zip_path=Path(zip_value) if zip_value else None,
    authorized_url=os.getenv('COVEO_DATA_URL'),
    sample=os.getenv('COVEO_USE_SAMPLE', '0') == '1',
)
display(report)"""),
            nbf.v4.new_markdown_cell("Để smoke test: đặt `COVEO_USE_SAMPLE=1`. Với full data: tải ZIP sau khi chấp nhận điều khoản, đặt `COVEO_TERMS_ACCEPTED=1` và `COVEO_ZIP_PATH=/path/file.zip`.")]),
        "01_prepare_sessions.ipynb": notebook("01 — ETL session và chronological split", [
            nbf.v4.new_code_cell("%pip install -q -e \".[coveo]\""), nbf.v4.new_code_cell(COMMON),
            nbf.v4.new_code_cell("""from datn.data.coveo import CoveoPrepareConfig, prepare_coveo_dataset
data_cfg = cfg['data']
params = CoveoPrepareConfig(
    dataset_version=Path(cfg['paths']['processed_dir']).name,
    train_ratio=data_cfg['train_ratio'], valid_ratio=data_cfg['valid_ratio'],
    min_session_events=data_cfg['min_session_events'],
    max_sessions=data_cfg[f'max_sessions_{PROFILE}'],
    include_search_clicks=data_cfg['include_search_clicks'], seed=cfg['project']['seed'])
manifest = prepare_coveo_dataset(REPO / cfg['paths']['raw_dir'], REPO / cfg['paths']['processed_dir'], params)
display(manifest['counts'], manifest['split_time_ranges'])"""),
            nbf.v4.new_markdown_cell("Split theo thời gian kết thúc session toàn cục. Validation/test là session tương lai; event cuối mỗi session là ground truth, phần trước là context.")]),
        "02_prepare_embeddings.ipynb": notebook("02 — Chuẩn hóa text/image embeddings", [
            nbf.v4.new_code_cell("%pip install -q -e \".[coveo]\""), nbf.v4.new_code_cell(COMMON),
            nbf.v4.new_code_cell("""from datn.features.coveo import build_coveo_embeddings
processed = REPO / cfg['paths']['processed_dir']
report = build_coveo_embeddings(REPO / cfg['paths']['raw_dir'] / 'sku_to_content.csv',
    processed / 'items.parquet', REPO / cfg['paths']['embeddings_dir'])
display(report)"""),
            nbf.v4.new_code_cell("""import numpy as np
for name in ('text', 'image'):
    x = np.load(REPO / cfg['paths']['embeddings_dir'] / f'{name}_embeddings.npy')
    norms = np.linalg.norm(x[1:], axis=1); present = norms > 0
    print(name, x.shape, 'coverage=', present.mean(), 'mean norm=', norms[present].mean() if present.any() else 0)""")]),
        "03_train_retrieval.ipynb": notebook("03 — Train action-aware multimodal two-tower", [
            nbf.v4.new_code_cell("%pip install -q -e \".[coveo]\""), nbf.v4.new_code_cell(COMMON),
            nbf.v4.new_code_cell("""from datn.recommenders.coveo.pipeline import RetrievalTrainConfig, train_retrieval
r = cfg['retrieval']
params = RetrievalTrainConfig(max_seq_len=r['max_seq_len'], d_model=r['d_model'], n_heads=r['n_heads'],
    n_layers=r['n_layers'], dropout=r['dropout'], batch_size=r['batch_size'], epochs=r[f'epochs_{PROFILE}'],
    lr=r['lr'], weight_decay=r['weight_decay'], purchase_alpha=r['purchase_alpha'], temperature=r['temperature'],
    patience=r['patience'], seed=cfg['project']['seed'], device=r['device'])
metrics = train_retrieval(REPO / cfg['paths']['processed_dir'], REPO / cfg['paths']['embeddings_dir'],
    REPO / cfg['paths']['retrieval_dir'], params)
display(metrics)"""),
            nbf.v4.new_markdown_cell("Checkpoint tốt nhất được chọn duy nhất bằng `validation HitRate@1000`; test chỉ chạy sau khi khóa checkpoint.")]),
        "04_train_reranker.ipynb": notebook("04 — Candidate features và residual listwise reranker", [
            nbf.v4.new_code_cell("%pip install -q -e \".[coveo]\""), nbf.v4.new_code_cell(COMMON),
            nbf.v4.new_code_cell("""from datn.recommenders.coveo.pipeline import RerankerTrainConfig, generate_candidates, train_reranker
rr = cfg['reranker']
params = RerankerTrainConfig(candidate_k=rr['candidate_k'], retrieval_k=rr['retrieval_k'], popularity_k=rr['popularity_k'],
    max_sessions_per_split=rr[f'max_sessions_per_split_{PROFILE}'], batch_size=rr['batch_size'], epochs=rr[f'epochs_{PROFILE}'],
    lr=rr['lr'], weight_decay=rr['weight_decay'], patience=rr['patience'], seed=cfg['project']['seed'], device=rr['device'])
processed = REPO / cfg['paths']['processed_dir']; embeddings = REPO / cfg['paths']['embeddings_dir']
retrieval_ckpt = REPO / cfg['paths']['retrieval_dir'] / 'best_retrieval.pt'
reranker_dir = REPO / cfg['paths']['reranker_dir']; candidate_dir = reranker_dir / 'candidates'
reports = {}
for split in ('train', 'valid', 'test'):
    reports[split] = generate_candidates(split, processed, embeddings, retrieval_ckpt, candidate_dir / f'{split}.parquet', params)
display(reports)"""),
            nbf.v4.new_code_cell("""metrics = train_reranker(candidate_dir / 'train.parquet', candidate_dir / 'valid.parquet', candidate_dir / 'test.parquet', reranker_dir, params)
display(metrics)"""),
            nbf.v4.new_code_cell("""# Chạy sau khi toàn bộ artifact tồn tại; đổi RUN_ID nếu cần chạy một thí nghiệm mới.
from datetime import datetime, timezone
from datn.experiments.coveo_checkpoint import checkpoint_coveo_run
RUN_ID = os.getenv('COVEO_RUN_ID', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
bundle = checkpoint_coveo_run(REPO, REPO / 'checkpoints' / f'coveo_{RUN_ID}')
print('immutable checkpoint:', bundle)"""),
            nbf.v4.new_markdown_cell("Báo cáo đồng thời `candidate_recall`, `ConditionalHR@K` và HR end-to-end. Không force-add target vào candidates.")]),
    }
    for name, nb in notebooks.items():
        nbf.write(nb, OUT / name)


if __name__ == "__main__":
    main()
