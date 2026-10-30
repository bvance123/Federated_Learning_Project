import csv
import torch

from fl.data import load_center
from fl.personal import run_personal
from fl.client import local_update, predict
from fl.model import Model, get_weights
from eval.metrics import auroc, evaluate, append_rows
from fl.baselines import run_local_only

#Define simulation variables
SEEDS = [0, 1, 2]
NAMES = {0: "Cleveland", 1: "Hungary", 2: "Switzerland", 3: "Long Beach VA"}
N_ROUNDS = 20
EPOCHS = 1


#Iterate through different seeds so that results don't depend 
#on a singular seed
for seed in SEEDS:
    torch.manual_seed(seed)

    data = {}
    clients = []
    for center in [0, 1, 2, 3]:
        train, val, test = load_center(center, seed=seed)
        data[center] = {"train": train, "val": val, "test": test}
        clients.append({"name": NAMES[center], "train": train, "val": val})

    # local-only models, JJ's code
    #Train local hospitals without mu adjustment
    local_models = run_local_only(data, seed=seed)

    #Record the results of training without mu
    local_val_auc = {}
    for center in [0, 1, 2, 3]:
        X_val, y_val = data[center]["val"]
        probs = predict(get_weights(local_models[center]), X_val)
        local_val_auc[NAMES[center]] = auroc(y_val, probs)
        

    #Train personal hospitals with the mu rule
    global_w, v, mu, history = run_personal(clients, local_val_auc, N_ROUNDS, EPOCHS)

        # 4. Run each personal hospital on its own  test data and record
        #how it does after training
    rows = []
    for center in [0, 1, 2, 3]:
        name = NAMES[center]
        X_test, y_test = data[center]["test"]
        probs = predict(v[name], X_test)
        row = evaluate("personal_adaptive", center, seed, y_test, probs, mu=mu[name])
        rows.append(row)
    append_rows(rows, "results.csv")

    # 5. save the mu history
    with open(f"mu_history_seed{seed}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)