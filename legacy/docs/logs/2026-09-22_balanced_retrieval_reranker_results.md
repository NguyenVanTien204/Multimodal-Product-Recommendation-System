# Balanced retrieval + reranker experiment (2026-09-22)

## Decision

Use the versioned `balanced_u5_i2_v1` dataset and the practical four-source
candidate recipe for the next thesis baseline. The recipe has a hard upper bound
of 2,000 candidates before de-duplication:

- User Tower: 1,000
- popularity: 300
- recency-weighted raw CLIP centroid: 400
- last-item raw CLIP similarity: 300

The 2,500-candidate variant has higher recall, but the smaller recipe is the
better latency/quality trade-off for a realistic two-stage system.

## Dataset

The source splits were rebuilt from verified-purchase interactions. Duplicated
user-item events were collapsed to the latest event, positives were defined as
rating >= 4, an iterative positive 5-user/2-item core was applied, and the last
two positive events per user were held out chronologically.

| Statistic | Value |
|---|---:|
| Users | 21,690 |
| Catalog items | 32,557 |
| Positive interactions | 198,200 |
| Positive train interactions | 154,820 |
| Mean positive train history | 7.14 |
| Test targets absent from positive train | 5.27% |
| Strong negative train events (rating <= 2) | 12,886 |

The immutable dataset and checksum manifest are under
`data/processed/balanced_u5_i2_v1/`.

## User Tower

Training used sampled softmax with 512 negatives, a 50/50 uniform-popularity
mixture, log-Q correction, image+text content, and validation HR@1000 early
stopping. The best checkpoint was epoch 4; training stopped at epoch 9.

| Metric | Validation | Test |
|---|---:|---:|
| HR@10 | 2.960% | 3.038% |
| HR@50 | 6.556% | 6.556% |
| HR@100 | 9.359% | 9.142% |
| HR@500 | 20.858% | 19.889% |
| HR@1000 | 29.036% | 27.321% |

## Candidate recall

Candidate recipes were defined on validation. Test remained excluded from model
training and checkpoint selection.

| Candidate recipe | Max size | Validation recall | Test recall |
|---|---:|---:|---:|
| User Tower | 1,000 | 29.036% | 27.321% |
| User Tower + popularity | 1,500 | 31.738% | 29.899% |
| Practical four-source union | 2,000 | 36.035% | 34.444% |
| Recall-oriented four-source union | 2,500 | 37.995% | 36.501% |

The practical union recovers 11.72% of completely cold test targets (target
degree zero in positive train); the User Tower and popularity branches recover
none of them. This is the main reason to keep content retrieval as an explicit
candidate source rather than expecting a collaborative tower to solve cold
start by itself.

## Residual listwise reranker (v2)

The reranker was trained only when the validation target was naturally present
in the candidate set; targets were never injected. It uses sampled-listwise
softmax over one positive and 128 negatives, with 75% hard negatives mined
across all four candidate sources. Its score is a learned residual over the
standardized User Tower score, so epoch zero exactly reproduces retrieval.

Epoch and residual blend were selected on validation NDCG with an HR@100 safety
constraint. The locked checkpoint is epoch 5 with blend 0.75. Candidate
generation and evaluation stream chunks to remain runnable on a 16 GB machine.

| Metric | Previous pipeline | Raw balanced retrieval | Residual reranker v2 | v2 vs previous |
|---|---:|---:|---:|---:|
| Candidate recall | 23.218% | 27.321% @1000 | 34.444% | +11.227 pp |
| HR@10 | 3.213% | 3.038% | 3.518% | +0.305 pp |
| NDCG@10 | — | 2.106% | 2.489% | — |
| HR@50 | 6.425% | 6.556% | 6.962% | +0.537 pp |
| NDCG@50 | — | 2.863% | 3.223% | — |
| HR@100 | 8.881% | 9.142% | 9.521% | +0.640 pp |
| NDCG@100 | — | 3.282% | 3.636% | — |

Unlike v1, the residual/listwise model improves HR and NDCG at all reported
cutoffs relative to the raw User Tower. Test was evaluated only after epoch 5
and blend 0.75 had been locked from validation.

## Reproduction

```powershell
datn-balanced-data --config configs/balanced_dataset.yaml
datn-user-tower --config configs/user_tower.balanced.yaml train
datn-user-tower --config configs/user_tower.balanced.yaml evaluate --split test --mode full `
  --output data/artifacts/user_tower_balanced_v1/test_metrics.json
```

Then run `notebooks/reranker_training.ipynb` using the frozen User Tower
checkpoint. The notebook saves a complete checkpoint, scaler, run config,
training history and validation/test metrics under `data/artifacts/reranker_v2/`.

After both stages finish, create an immutable experiment bundle:

```powershell
datn-checkpoint --destination data/checkpoints/balanced_two_stage_v2
```
