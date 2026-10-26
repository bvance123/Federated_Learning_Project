
"""Run local-only, pooled, and FedAvg baselines.

Owner: JJ
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, recall_score

from fl.baselines import (
    load_all_centers,
    run_local_only,
    run_pooled,
)
from fl.client import local_update
from fl.data import CENTERS
from fl.model import Model
from fl.server import run_rounds


RESULTS_DIR = Path("results")
SEED = 0
EPOCHS = 50
LR = 0.01
BATCH_SIZE = 32
FEDAVG_ROUNDS = 20
LOCAL_EPOCHS = 1


def evaluate(model, test_data):
    """Evaluate a binary classifier on held-out data."""
    X, y = test_data

    model.eval()
    with torch.no_grad():
        probs = model(X).reshape(-1).cpu().numpy()

    y_true = y.reshape(-1).cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    # AUROC is undefined when the test labels contain only one class.
    auroc = (
        roc_auc_score(y_true, probs)
        if len(np.unique(y_true)) == 2
        else float("nan")
    )

    sensitivity = (
        recall_score(y_true, y_pred, zero_division=0)
        if np.any(y_true == 1)
        else float("nan")
    )

    return {
        "auroc": float(auroc),
        "sensitivity": float(sensitivity),
        "n_test": len(y_true),
    }


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_metrics.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "seed", "center",
                "auroc", "sensitivity", "n_test",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics to {output_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = load_all_centers(seed=SEED)
    rows = []

    # 1. Local-only
    local_models = run_local_only(
        data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE
    )

    for center, model in local_models.items():
        metrics = evaluate(model, data[center]["test"])
        rows.append({
            "method": "local_only",
            "seed": SEED,
            "center": center,
            **metrics,
        })

    # 2. Centralized pooled
    pooled_model = run_pooled(
        data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE
    )

    for center in CENTERS:
        metrics = evaluate(pooled_model, data[center]["test"])
        rows.append({
            "method": "pooled",
            "seed": SEED,
            "center": center,
            **metrics,
        })

    # 3. Standard FedAvg
    clients = [
        {
            "name": f"center_{center}",
            "train": data[center]["train"],
        }
        for center in CENTERS
    ]

    torch.manual_seed(SEED)
    global_model = Model()

    history = run_rounds(
        global_model=global_model,
        clients=clients,
        local_update=local_update,
        n_rounds=FEDAVG_ROUNDS,
        mu=0.0,
        epochs=LOCAL_EPOCHS,
    )

    for center in CENTERS:
        metrics = evaluate(global_model, data[center]["test"])
        rows.append({
            "method": "fedavg",
            "seed": SEED,
            "center": center,
            **metrics,
        })

    save_results(rows)


if __name__ == "__main__":
    main()
