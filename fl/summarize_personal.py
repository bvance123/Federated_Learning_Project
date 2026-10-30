import pandas as pd

df = pd.read_csv("results.csv")
cols = ["auroc", "sensitivity", "ece", "brier"]

print("Average over seeds, per hospital:")
print(df.groupby(["center", "method"])[cols].mean().round(3))

print()
print("Average over seeds and hospitals, per method:")
print(df.groupby("method")[cols].mean().round(3))

import glob

for path in sorted(glob.glob("mu_history_seed*.csv")):
    h = pd.read_csv(path)
    print()
    print(f"Mu by round ({path}):")
    print(h.pivot(index="round", columns="hospital", values="mu"))