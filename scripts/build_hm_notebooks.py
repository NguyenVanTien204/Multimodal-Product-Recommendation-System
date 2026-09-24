"""Generate the standalone, Kaggle-portable H&M notebook sequence."""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "hm"


def nb(title: str, *cells: str):
    result = nbf.v4.new_notebook()
    result.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nStandalone H&M pipeline. Outputs are versioned under `/kaggle/working/hm_v1`.")]
    for cell in cells:
        result.cells.append(nbf.v4.new_markdown_cell(cell[3:]) if cell.startswith("MD:") else nbf.v4.new_code_cell(cell))
    result.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
    return result


def merge(title: str, sections: list[tuple[str, nbf.NotebookNode]]) -> nbf.NotebookNode:
    """Combine cohesive stages while retaining visible section boundaries."""
    result = nbf.v4.new_notebook()
    result.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nStandalone H&M Kaggle pipeline.")]
    for heading, book in sections:
        result.cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        result.cells.extend(book.cells[1:])
    result.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
    return result


COMMON = r'''from pathlib import Path
import os, json, random, hashlib
import numpy as np
import polars as pl

PROFILE = os.getenv("HM_PROFILE", "quick").lower()
assert PROFILE in {"quick", "full"}
WORK = Path(os.getenv("HM_WORK_DIR", "/kaggle/working/hm_v1"))
WORK.mkdir(parents=True, exist_ok=True)
cfg = json.loads((WORK / "run_config.json").read_text())
SEED = cfg["seed"]
random.seed(SEED); np.random.seed(SEED)
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
"00_get_data.ipynb": nb("00 — Discover H&M data and lock run configuration", r'''# Kaggle: attach the competition data as an Input, then run this cell.
from pathlib import Path
import os, json, hashlib

def find_file(name: str) -> Path:
    roots = [Path("/kaggle/input"), Path.cwd()]
    for root in roots:
        if root.exists():
            matches = list(root.rglob(name))
            if matches: return matches[0]
    raise FileNotFoundError(f"Cannot find {name}. Attach the H&M competition dataset as a Kaggle Input.")

files = {name: find_file(name) for name in ("transactions_train.csv", "articles.csv", "customers.csv")}
image_dirs = [p / "images" for p in files["articles.csv"].parents if (p / "images").exists()]
profile = os.getenv("HM_PROFILE", "quick").lower()
assert profile in {"quick", "full"}
work = Path(os.getenv("HM_WORK_DIR", "/kaggle/working/hm_v1")); work.mkdir(parents=True, exist_ok=True)
config = {
    "dataset": "H&M Personalized Fashion Recommendations", "version": "hm_v1", "profile": profile,
    "seed": 20260922, "customers": 50_000 if profile == "quick" else 300_000,
    "min_train_events": 3, "max_events_per_customer": 100,
    "transactions": str(files["transactions_train.csv"]), "articles": str(files["articles.csv"]),
    "customers_file": str(files["customers.csv"]), "images": str(image_dirs[0]) if image_dirs else None,
}
(work / "run_config.json").write_text(json.dumps(config, indent=2))
for name, path in files.items(): print(f"{name}: {path} ({path.stat().st_size / 1e9:.2f} GB)")
print("images:", config["images"], "\nworking directory:", work)''',
"MD:Do not use `kaggle competitions download` inside this notebook. Mounting the competition data is more reliable, avoids credentials, and works with Internet disabled."),

"01_prepare_dataset.ipynb": nb("01 — Smart sample, 7-day split, and Parquet ETL", COMMON, r'''from datetime import timedelta
tx = cfg["transactions"]; articles_csv = cfg["articles"]
schema = pl.scan_csv(tx, schema_overrides={"article_id": pl.String, "customer_id": pl.String}, try_parse_dates=True)
max_date = schema.select(pl.col("t_dat").max()).collect().item()
test_start = max_date - timedelta(days=6); valid_start = test_start - timedelta(days=7); rerank_train_start = valid_start - timedelta(days=7)
print({"max_date": str(max_date), "rerank_train_start": str(rerank_train_start), "valid_start": str(valid_start), "test_start": str(test_start)})

# Customer eligibility and strata use only train-period purchases. Hash order is
# deterministic and avoids selecting users because of future labels.
history = (schema.filter(pl.col("t_dat") < pl.lit(rerank_train_start))
    .group_by("customer_id").agg(pl.len().alias("n_train"))
    .filter(pl.col("n_train") >= cfg["min_train_events"])
    .with_columns((pl.col("n_train").log10().floor().clip(0, 4).cast(pl.Int8)).alias("activity_bin"))
    .with_columns(pl.struct(["customer_id"]).hash(seed=SEED).alias("sample_hash")))
counts = history.group_by("activity_bin").len().collect().sort("activity_bin")
total = counts["len"].sum()
quotas = {row["activity_bin"]: max(1, round(cfg["customers"] * row["len"] / total)) for row in counts.to_dicts()}
selected = pl.concat([history.filter(pl.col("activity_bin") == b).sort("sample_hash").head(q) for b, q in quotas.items()]).select("customer_id")
selected_path = WORK / "selected_customers.parquet"; selected.collect().write_parquet(selected_path)
print("selected users:", sum(quotas.values()), "bins:", quotas)''', r'''# Keep all future interactions but cap only old training history per selected user.
events = (schema.join(pl.scan_parquet(selected_path), on="customer_id", how="inner")
    .with_columns(pl.col("t_dat").cast(pl.Date), pl.col("article_id").cast(pl.String))
    .with_columns(pl.when(pl.col("t_dat") < pl.lit(rerank_train_start)).then(pl.lit("train"))
        .when(pl.col("t_dat") < pl.lit(valid_start)).then(pl.lit("rerank_train"))
        .when(pl.col("t_dat") < pl.lit(test_start)).then(pl.lit("valid")).otherwise(pl.lit("test")).alias("split"))
    .with_columns(pl.when(pl.col("split") == "train").then(pl.lit(0)).otherwise(pl.lit(1)).alias("future"))
    .with_columns(pl.col("t_dat").rank("ordinal", descending=True).over("customer_id", "split").alias("recent_rank"))
    .filter((pl.col("split") != "train") | (pl.col("recent_rank") <= cfg["max_events_per_customer"]))
    .select("customer_id", "article_id", "t_dat", "price", "sales_channel_id", "split")
    .sort(["customer_id", "t_dat", "article_id"]))
for split in ("train", "rerank_train", "valid", "test"):
    path = WORK / f"{split}.parquet"
    events.filter(pl.col("split") == split).collect(streaming=True).write_parquet(path, compression="zstd")

# Product metadata has no interaction labels, so it is safe to retain all rows.
items = (pl.scan_csv(articles_csv, schema_overrides={"article_id": pl.String})
    .select("article_id", "prod_name", "product_type_name", "product_group_name", "graphical_appearance_name", "colour_group_name", "department_name", "section_name", "detail_desc")
    .rename({"article_id": "item_id"}).collect(streaming=True))
items.write_parquet(WORK / "items.parquet", compression="zstd")
counts = {s: pl.scan_parquet(WORK / f"{s}.parquet").select(pl.len().alias("events"), pl.col("customer_id").n_unique().alias("users"), pl.col("article_id").n_unique().alias("items")).collect().to_dicts()[0] for s in ("train", "rerank_train", "valid", "test")}
manifest = {"dataset": cfg["dataset"], "version": cfg["version"], "profile": cfg["profile"], "seed": SEED, "windows": {"retrieval_train_before": str(rerank_train_start), "reranker_train": [str(rerank_train_start), str(valid_start - timedelta(days=1))], "valid": [str(valid_start), str(test_start - timedelta(days=1))], "test": [str(test_start), str(max_date)]}, "sampling": {"customers": cfg["customers"], "min_train_events": cfg["min_train_events"], "max_old_train_events": cfg["max_events_per_customer"], "selection_uses": "before reranker-train window only"}, "counts": counts}
(WORK / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)); display(manifest)''',
"MD:Validation and test are global seven-day windows. Users with no target purchase in a window are retained in the events but excluded by the metric for that window."),

"02_embeddings.ipynb": nb("02 — Persistent Jina CLIP v2 embeddings", COMMON, "%pip install -q --upgrade --force-reinstall --no-cache-dir \"numpy==2.0.2\" \"scipy==1.14.1\" \"scikit-learn==1.5.2\" \"transformers==4.51.3\" \"tokenizers<0.22\" \"huggingface_hub<1.0\" \"einops==0.8.1\" \"timm==1.0.19\" \"pillow==11.3.0\"", r'''# Kaggle Internet must be enabled for this first run. The Hugging Face cache is
# persisted in WORK so retries/restarts reuse the downloaded Jina model.
import torch
from transformers import AutoModel

MODEL_ID = "jinaai/jina-clip-v2"; DIM = 512
os.environ["HF_HOME"] = str(WORK / "hf_cache")
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="float32").float().eval()
if torch.cuda.is_available(): model = model.to(device="cuda", dtype=torch.float32)
print("loaded", MODEL_ID, "on", next(model.parameters()).device, "with output dim", DIM)''', r'''items_path = WORK / "items.parquet"
if not items_path.is_file():
    raise FileNotFoundError(f"Missing {items_path}; run 00_hm_data_preparation.ipynb first.")
items = pl.read_parquet(str(items_path)).fill_null("")
texts = (items["prod_name"] + ". " + items["product_type_name"] + ". " + items["product_group_name"] + ". " + items["colour_group_name"] + ". " + items["department_name"] + ". " + items["detail_desc"]).to_list()
text_vectors = []
with torch.inference_mode():
    for start in range(0, len(texts), 64):
        # Article metadata is corpus content; this Jina checkpoint supports only
        # retrieval.query or None. None keeps text aligned with image vectors.
        vector = model.encode_text(texts[start:start + 64], task=None, truncate_dim=DIM)
        text_vectors.append(np.asarray(vector, dtype="float32"))
text_vectors = np.concatenate(text_vectors); text_vectors /= np.maximum(np.linalg.norm(text_vectors, axis=1, keepdims=True), 1e-12)
np.save(WORK / "text_embeddings.npy", text_vectors)
print("saved", WORK / "text_embeddings.npy", text_vectors.shape)''', r'''# Encode the H&M image paths in batches. A zero row means that this article has
# no source image, never that a random/untrained image vector was substituted.
image_vectors = np.zeros((items.height, DIM), dtype="float32")
paths = [(i, Path(cfg["images"]) / article[:3] / f"{article}.jpg") for i, article in enumerate(items["item_id"].to_list())]
paths = [(i, path) for i, path in paths if path.exists()]
limit = 60_000 if PROFILE == "quick" else None
if os.getenv("HM_MAX_IMAGES"): limit = int(os.environ["HM_MAX_IMAGES"])
paths = paths[:limit] if limit else paths
with torch.inference_mode():
    for start in range(0, len(paths), 32):
        batch = paths[start:start + 32]
        vector = np.asarray(model.encode_image([str(path) for _, path in batch], truncate_dim=DIM), dtype="float32")
        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        image_vectors[[i for i, _ in batch]] = vector
        if start % 3200 == 0: print(f"encoded {min(start + len(batch), len(paths)):,}/{len(paths):,} images")
np.save(WORK / "image_embeddings.npy", image_vectors)
manifest = {"model": MODEL_ID, "truncate_dim": DIM, "text_file": "text_embeddings.npy", "image_file": "image_embeddings.npy", "items": items.height, "image_coverage": float((np.linalg.norm(image_vectors, axis=1) > 0).mean()), "hf_cache": str(WORK / "hf_cache")}
(WORK / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2)); display(manifest)''',
"MD:Jina CLIP v2 produces aligned text/image vectors. This notebook owns downloading, encoding and persistent artifacts; later notebooks only read its `.npy` files and manifest."),

"03_train_retrieval.ipynb": nb("03 — Train recency-aware multimodal retrieval", COMMON, r'''import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

items = pl.read_parquet(WORK / "items.parquet"); item_ids = items["item_id"].to_list(); vocab = {x: i + 1 for i, x in enumerate(item_ids)}
text = np.vstack([np.zeros((1, np.load(WORK / "text_embeddings.npy").shape[1]), "float32"), np.load(WORK / "text_embeddings.npy")])
image = np.vstack([np.zeros((1, np.load(WORK / "image_embeddings.npy").shape[1]), "float32"), np.load(WORK / "image_embeddings.npy")])
def grouped(path):
    f = pl.read_parquet(path).with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx")).filter(pl.col("idx") > 0).sort(["customer_id", "t_dat"])
    return {r["customer_id"]: r["idx"] for r in f.group_by("customer_id").agg(pl.col("idx")).to_dicts()}
train, rr_train, valid, test = (grouped(WORK / f"{x}.parquet") for x in ("train", "rerank_train", "valid", "test"))
freq = np.ones(len(item_ids) + 1, dtype="float64")
for seq in train.values(): freq[np.asarray(seq)] += 1
prob = freq ** .75; prob /= prob.sum()
MAXLEN, D, NEG = 30, 128, 128
class PairData(Dataset):
    def __init__(self, histories): self.rows=[(s[:j],s[j]) for s in histories.values() for j in range(1,len(s))]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        h,y=self.rows[i]; return torch.tensor(([0]*MAXLEN+h[-MAXLEN:])[-MAXLEN:]), torch.tensor(y)
class Tower(nn.Module):
    def __init__(self):
        super().__init__(); self.id=nn.Embedding(len(item_ids)+1,D,padding_idx=0); self.t=nn.Linear(text.shape[1],D,bias=False); self.v=nn.Linear(image.shape[1],D,bias=False); self.gate=nn.Parameter(torch.tensor(0.0)); self.register_buffer("text",torch.tensor(text)); self.register_buffer("image",torch.tensor(image))
    def table(self): return nn.functional.normalize(self.id.weight+self.t(self.text)+torch.sigmoid(self.gate)*self.v(self.image),dim=1)
    def query(self,h):
        table=self.table(); x=table[h]; mask=h.ne(0); weights=torch.arange(1,h.shape[1]+1,device=h.device)[None]*mask; return nn.functional.normalize((x*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1),dim=1)
device="cuda" if torch.cuda.is_available() else "cpu"; model=Tower().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
loader=DataLoader(PairData(train),batch_size=512,shuffle=True,num_workers=2,pin_memory=device=="cuda")
print("train pairs",len(loader.dataset),"catalog",len(item_ids),"device",device)''', r'''def score_metrics(context, targets, k=12):
    model.eval(); aps=[]; recalls=[]
    for user, truth in targets.items():
        if not truth or user not in context: continue
        h=context[user][-MAXLEN:]; batch=torch.tensor([[0]*(MAXLEN-len(h))+h],device=device)
        scores=(model.query(batch)@model.table().T).squeeze(); scores[0]=-torch.inf
        ranked=torch.topk(scores,k).indices.cpu().tolist(); relevant=set(truth); hits=[int(x in relevant) for x in ranked]
        aps.append(sum(sum(hits[:i+1])/(i+1)*v for i,v in enumerate(hits))/min(k,len(relevant))); recalls.append(sum(hits)/len(relevant))
    return {"MAP@12":float(np.mean(aps) if aps else 0),"Recall@12":float(np.mean(recalls) if recalls else 0),"users":len(aps)}
epochs=3 if PROFILE=="quick" else 12; best=-1; history=[]
for epoch in range(1,epochs+1):
    model.train(); losses=[]
    for hist,target in loader:
        hist,target=hist.to(device),target.to(device); neg=torch.tensor(np.random.choice(len(prob),size=(len(target),NEG),p=prob),device=device)
        candidates=torch.cat([target[:,None],neg],1); logits=(model.query(hist)[:,None,:]*model.table()[candidates]).sum(-1)/.07
        loss=nn.functional.cross_entropy(logits,torch.zeros(len(target),dtype=torch.long,device=device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(loss.item())
    metrics=score_metrics(train,rr_train); history.append({"epoch":epoch,"loss":float(np.mean(losses)),**metrics}); print(history[-1])
    if metrics["MAP@12"]>best: best=metrics["MAP@12"]; torch.save({"state":model.state_dict(),"item_ids":item_ids,"d_model":D,"valid":metrics},WORK/"best_retrieval.pt")
state=torch.load(WORK/"best_retrieval.pt",map_location=device); model.load_state_dict(state["state"])
validation_context={u:train.get(u,[])+rr_train.get(u,[]) for u in valid}; test_context={u:train.get(u,[])+rr_train.get(u,[])+valid.get(u,[]) for u in test}
result={"selection_split":"reranker_train", "best_train_window":state["valid"], "valid":score_metrics(validation_context,valid), "test":score_metrics(test_context,test)}
(WORK/"retrieval_history.json").write_text(json.dumps(history,indent=2)); (WORK/"retrieval_metrics.json").write_text(json.dumps(result,indent=2)); display(result)''',
"MD:The retrieval checkpoint is selected on the pre-validation (`rerank_train`) window only. Validation and test remain sealed until checkpoint selection is complete."),

"04_train_reranker.ipynb": nb("04 — Candidate union and residual listwise reranker", COMMON, r'''import torch
from torch import nn
items = pl.read_parquet(WORK / "items.parquet"); item_ids = items["item_id"].to_list(); vocab={x:i+1 for i,x in enumerate(item_ids)}
text_raw=np.load(WORK/"text_embeddings.npy"); image_raw=np.load(WORK/"image_embeddings.npy")
text=np.vstack([np.zeros((1,text_raw.shape[1]),"float32"),text_raw]); image=np.vstack([np.zeros((1,image_raw.shape[1]),"float32"),image_raw])
def grouped(path):
 f=pl.read_parquet(path).with_columns(pl.col("article_id").replace_strict(vocab,default=0).alias("idx")).filter(pl.col("idx")>0).sort(["customer_id","t_dat"])
 return {r["customer_id"]:r["idx"] for r in f.group_by("customer_id").agg(pl.col("idx")).to_dicts()}
train,rr_train,valid,test=(grouped(WORK/f"{x}.parquet") for x in ("train","rerank_train","valid","test"))
class Tower(nn.Module):
 def __init__(self):
  super().__init__(); self.id=nn.Embedding(len(item_ids)+1,128,padding_idx=0); self.t=nn.Linear(text.shape[1],128,bias=False); self.v=nn.Linear(image.shape[1],128,bias=False); self.gate=nn.Parameter(torch.tensor(0.)); self.register_buffer("text",torch.tensor(text)); self.register_buffer("image",torch.tensor(image))
 def table(self): return nn.functional.normalize(self.id.weight+self.t(self.text)+torch.sigmoid(self.gate)*self.v(self.image),dim=1)
 def query(self,h):
  table=self.table(); x=table[h]; mask=h.ne(0); w=torch.arange(1,h.shape[1]+1,device=h.device)[None]*mask; return nn.functional.normalize((x*w[:,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1),dim=1)
device="cuda" if torch.cuda.is_available() else "cpu"; model=Tower().to(device); model.load_state_dict(torch.load(WORK/"best_retrieval.pt",map_location=device)["state"]); model.eval()
freq=np.ones(len(item_ids)+1); [freq.__setitem__(np.asarray(s),freq[np.asarray(s)]+1) for s in train.values()]; popular=np.argsort(-freq)[1:101]
MAXLEN=30; RETRIEVAL_K=300; CONTENT_K=100; CANDIDATE_K=500
print("catalog",len(item_ids),"device",device)''', r'''@torch.no_grad()
def candidates(context, targets):
    groups=[]; table=model.table()
    for user, truth in targets.items():
        h=context.get(user, [])[-MAXLEN:]
        if not h or not truth: continue
        batch=torch.tensor([[0]*(MAXLEN-len(h))+h],device=device)
        scores=(model.query(batch)@table.T).squeeze(); scores[0]=-torch.inf
        top=torch.topk(scores,RETRIEVAL_K).indices.cpu().numpy(); last=h[-1]
        content=np.argsort(-(text[last]@text.T))[:CONTENT_K]
        union=[]
        for source in (top,popular,content):
            for idx in source:
                idx=int(idx)
                if idx and idx not in union: union.append(idx)
        union=union[:CANDIDATE_K]; labels=np.array([int(i in set(truth)) for i in union],dtype="float32")
        if not len(union): continue
        f=np.column_stack([scores[union].detach().cpu().numpy(), np.log1p(freq[union]), text[last]@text[union].T, image[last]@image[union].T, np.arange(len(union),0,-1)/len(union)]).astype("float32")
        groups.append((f,labels))
    recall=float(np.mean([y.max() for _,y in groups])) if groups else 0.
    return groups,recall
rerank_context=train
valid_context={u:train.get(u,[])+rr_train.get(u,[]) for u in valid}
test_context={u:train.get(u,[])+rr_train.get(u,[])+valid.get(u,[]) for u in test}
train_groups,train_recall=candidates(rerank_context,rr_train); valid_groups,valid_recall=candidates(valid_context,valid); test_groups,test_recall=candidates(test_context,test)
# Fit scaling on the reranker training window only, then apply it unchanged.
mean=np.concatenate([x for x,_ in train_groups]).mean(0); std=np.concatenate([x for x,_ in train_groups]).std(0).clip(1e-6)
train_groups=[((x-mean)/std,y) for x,y in train_groups]; valid_groups=[((x-mean)/std,y) for x,y in valid_groups]; test_groups=[((x-mean)/std,y) for x,y in test_groups]
print({"train_groups":len(train_groups),"valid_groups":len(valid_groups),"test_groups":len(test_groups),"candidate_recall":{"train":train_recall,"valid":valid_recall,"test":test_recall}})''', r'''class Ranker(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(5,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,1))
 def forward(self,x): return self.net(x).squeeze(-1)
ranker=Ranker().to(device); opt=torch.optim.AdamW(ranker.parameters(),lr=1e-3,weight_decay=1e-4)
def evaluate(groups, recall):
 ranker.eval(); aps=[]; recalls=[]
 with torch.no_grad():
  for x,y in groups:
   order=torch.argsort(ranker(torch.tensor(x,device=device)),descending=True).cpu().numpy()[:12]; hit=y[order]; total=y.sum()
   if total <= 0: aps.append(0.); recalls.append(0.); continue
   aps.append(sum(hit[:i+1].sum()/(i+1)*v for i,v in enumerate(hit))/min(12,total)); recalls.append(hit.sum()/total)
 return {"candidate_recall":recall,"MAP@12":float(np.mean(aps) if aps else 0),"Recall@12":float(np.mean(recalls) if recalls else 0),"eligible_users":len(aps)}
epochs=4 if PROFILE=="quick" else 15; best=-1; history=[]
for epoch in range(1,epochs+1):
 ranker.train(); random.shuffle(train_groups); losses=[]
 for x,y in train_groups:
  if not y.max(): continue
  score=ranker(torch.tensor(x,device=device)); labels=torch.tensor(y,device=device)
  # Multi-positive listwise target: every purchased article receives probability mass.
  loss=-(torch.log_softmax(score,0)*labels/labels.sum()).sum(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
 metrics=evaluate(valid_groups,valid_recall); history.append({"epoch":epoch,"loss":float(np.mean(losses) if losses else 0),**metrics}); print(history[-1])
 if metrics["MAP@12"]>best: best=metrics["MAP@12"]; torch.save({"state":ranker.state_dict(),"valid":metrics,"features":["retrieval_score","log_popularity","text_last_similarity","image_last_similarity","union_position"]},WORK/"best_reranker.pt")
ranker.load_state_dict(torch.load(WORK/"best_reranker.pt",map_location=device)["state"])
result={"selection_split":"valid","best_valid":torch.load(WORK/"best_reranker.pt",map_location=device)["valid"],"test":evaluate(test_groups,test_recall)}
(WORK/"reranker_history.json").write_text(json.dumps(history,indent=2)); (WORK/"reranker_metrics.json").write_text(json.dumps(result,indent=2)); display(result)''',
"MD:Candidates are the deduplicated union of retrieval, train-only popularity, and last-item content similarity. Ground-truth articles are never inserted; users whose target is absent contribute to candidate recall but not conditional reranker MAP."),
    }
    stages = notebooks
    notebooks = {
        "00_hm_data_preparation.ipynb": merge("00 — H&M data preparation", [
            ("Discover the mounted Kaggle dataset and lock configuration", stages["00_get_data.ipynb"]),
            ("Smart sample, chronological windows, and Parquet ETL", stages["01_prepare_dataset.ipynb"]),
        ]),
        "01_hm_jina_embeddings.ipynb": merge("01 — H&M Jina CLIP embeddings", [
            ("Download Jina CLIP v2 and persist text/image embeddings", stages["02_embeddings.ipynb"]),
        ]),
        "02_hm_retrieval.ipynb": merge("02 — H&M retrieval", [
            ("Train and evaluate the retrieval model", stages["03_train_retrieval.ipynb"]),
        ]),
        "03_hm_reranking_evaluation.ipynb": merge("03 — H&M reranking and final evaluation", [
            ("Candidate union and residual listwise reranker", stages["04_train_reranker.ipynb"]),
        ]),
    }
    obsolete = set(stages) | {"01_hm_embeddings_retrieval.ipynb", "02_hm_reranking_evaluation.ipynb"}
    for legacy in obsolete:
        (OUT / legacy).unlink(missing_ok=True)
    for name, book in notebooks.items(): nbf.write(book, OUT / name)


if __name__ == "__main__": main()
