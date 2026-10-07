# Drift-Modulated Federated Learning for Heart Disease

ECE 4424 / CS 4824 course project (Fall 2026): Bennett Vance, Nikolay Rozenov, JJ Feeney.

We give each hospital a personal model trained with a drift penalty
`(mu/2) * ||v - w_global||^2`, and study how changing `mu` per hospital helps
small hospitals whose patients differ from the rest. Data: FLamby Fed-Heart-Disease (4 hospitals).

## Layout

| Path | What | Owner |
|---|---|---|
| `fl/model.py` | Shared network (`Model`), `get_weights` / `set_weights` | Bennett |
| `fl/server.py` | `aggregate` (FedAvg), `run_rounds` | Bennett |
| `fl/client.py` | `local_update`, personal models with mu | Nikolay |
| `fl/data.py` | `load_center`, preprocessing | JJ |
| `eval/` | Metrics (AUROC, sensitivity, calibration), figures | Bennett |
| `data/` | Downloaded FLamby data (not committed) | — |
| `results/` | Results CSVs | all |
| `LOG.md` | Problems and failed attempts, as they happen | all |

## Agreed interfaces

```python
load_center(center, seed) -> (train, val, test)    # each is (X [n,13] float, y [n] float 0/1)
Model()                                            # forward(x) -> P(disease), shape [n]
local_update(model, global_weights, train_data, mu, epochs) -> state_dict
aggregate(weights_list, sample_counts) -> state_dict
```

Results CSV columns: `method, center, seed, auroc, sensitivity, ece`

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests          # server tests, no data needed
python -m fl.server             # end-to-end FedAvg (needs data.py + client.py)
```
