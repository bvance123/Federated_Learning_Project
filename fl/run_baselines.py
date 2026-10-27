
"""Run seeded local-only, pooled, and FedAvg baselines.

Owner: JJ
"""

from pathlib import Path

import numpy as np
import torch

from eval.metrics import append_rows, evaluate
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
SEEDS = range(5)
EPOCHS = 50
LR = 0.01
BATCH_SIZE = 32
FEDAVG_ROUNDS = EPOCHS
LOCAL_EPOCHS = 1


def predict(model, test_data):
    """Return labels and probabilities for one hospital's held-out data."""
    X, y = test_data

    model.eval()
    with torch.no_grad():
        probs = model(X).reshape(-1).cpu().numpy()

    y_true = y.reshape(-1).cpu().numpy().astype(int)
    return y_true, probs


def collect_metrics(method, model, center, seed, test_data):
    y_true, y_prob = predict(model, test_data)
    return evaluate(method, center, seed, y_true, y_prob)


def main():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = load_all_centers(seed=seed)

        local_models = run_local_only(
            data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=seed
        )
        for center, model in local_models.items():
            rows.append(collect_metrics(
                "local_only", model, center, seed, data[center]["test"]
            ))

        pooled_model = run_pooled(
            data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=seed
        )
        for center in CENTERS:
            rows.append(collect_metrics(
                "pooled", pooled_model, center, seed, data[center]["test"]
            ))

        clients = [
            {"name": f"center_{center}", "train": data[center]["train"]}
            for center in CENTERS
        ]
        torch.manual_seed(seed)
        global_model = Model()
        run_rounds(
            global_model=global_model,
            clients=clients,
            local_update=local_update,
            n_rounds=FEDAVG_ROUNDS,
            mu=0.0,
            epochs=LOCAL_EPOCHS,
            verbose=False,
        )

        for center in CENTERS:
            rows.append(collect_metrics(
                "fedavg", global_model, center, seed, data[center]["test"]
            ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_metrics.csv"
    append_rows(rows, output_path)
    print(f"Appended metrics to {output_path}")


if __name__ == "__main__":
    main()
