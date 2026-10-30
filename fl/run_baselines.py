
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
SEEDS = range(30)
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


def print_dataset_summary(data, seed):
    """Print patient counts, disease rates, and feature counts per center."""

    center_names = {
        0: "Cleveland",
        1: "Hungary",
        2: "Switzerland",
        3: "Long Beach VA",
    }

    print("\n" + "=" * 85)
    print(f"DATASET SUMMARY (seed={seed})")
    print("=" * 85)

    total_original_train = 0
    total_train = 0
    total_val = 0
    total_test = 0

    def describe_split(split_data):
        X, y = split_data
        y = y.reshape(-1)

        n = len(y)
        positives = int((y == 1).sum().item())
        negatives = int((y == 0).sum().item())
        positive_rate = 100 * positives / n if n else float("nan")

        return {
            "n": n,
            "positive": positives,
            "negative": negatives,
            "rate": positive_rate,
            "features": X.shape[1] if X.ndim > 1 else 1,
        }

    for center in CENTERS:
        center_data = data[center]

        train_stats = describe_split(center_data["train"])
        val_stats = describe_split(center_data["val"])
        test_stats = describe_split(center_data["test"])

        # Training + validation reconstructs the partition
        # originally supplied to the train/validation splitting code.
        original_train_n = train_stats["n"] + val_stats["n"]

        total_original_train += original_train_n
        total_train += train_stats["n"]
        total_val += val_stats["n"]
        total_test += test_stats["n"]

        name = center_names.get(center, f"Center {center}")

        print(f"\nCenter {center}: {name}")
        print(f"  Input features: {train_stats['features']}")
        print(
            f"  Original FLamby training partition: "
            f"{original_train_n} patients"
        )

        # for split_name, stats in [
        #     ("Train (used for fitting)", train_stats),
        #     ("Validation", val_stats),
        #     ("Held-out test", test_stats),
        # ]:
            # print(
            #     f"  {split_name:25s}: "
            #     f"N={stats['n']:4d} | "
            #     f"Positive={stats['positive']:4d} | "
            #     f"Negative={stats['negative']:4d} | "
            #     f"Disease rate={stats['rate']:6.2f}%"
            # )

        # Overall statistics for this center across all splits.
        all_splits = [
            center_data["train"],
            center_data["val"],
            center_data["test"],
        ]

        all_y = torch.cat(
            [split[1].reshape(-1) for split in all_splits]
        )

        total_n = len(all_y)
        total_positive = int((all_y == 1).sum().item())
        total_negative = int((all_y == 0).sum().item())
        disease_rate = (
            100.0 * total_positive / total_n if total_n else float("nan")
        )

        print("  Overall center statistics:")
        print(f"    Total patients: {total_n}")
        print(f"    Positive cases: {total_positive}")
        print(f"    Negative cases: {total_negative}")
        print(f"    Disease rate:   {disease_rate:.2f}%")

    print("\n" + "-" * 85)
    print("TOTALS ACROSS ALL CENTERS")
    print("-" * 85)
    print(f"Original FLamby training partitions: {total_original_train}")
    print(f"Final training patients:             {total_train}")
    print(f"Validation patients:                 {total_val}")
    print(f"Held-out test patients:              {total_test}")
    print(
        f"Total unique patients represented:   "
        f"{total_train + total_val + total_test}"
    )
    print("=" * 85 + "\n")



def main():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = load_all_centers(seed=seed)

        #print data only once
        if seed == 0:
            print_dataset_summary(data, seed)

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
