import math
import torch
from fl.personal import run_personal


def make_hospital(name, n_train=60, n_val=40, seed=0, val_one_class=False):
    g = torch.Generator().manual_seed(seed)

    def make(n):
        X = torch.randn(n, 13, generator=g)
        y = ((X[:, 0] + 2 * torch.randn(n, generator=g)) > 0).float()
        # force both classes so AUROC is not NaN
        y[0] = 0.0
        y[1] = 1.0
        return X, y

    train = make(n_train)
    val = make(n_val)
    if val_one_class:
        val = (val[0], torch.zeros(n_val))
    return {"name": name, "train": train, "val": val}


def make_clients(val_one_class=False):
    return [
        make_hospital("A", seed=1, val_one_class=val_one_class),
        make_hospital("B", seed=2),
        make_hospital("C", seed=3),
    ]


def final_mu(history, name):
    rows = [r for r in history if r["hospital"] == name]
    return rows[-1]["mu"]


N_ROUNDS = 10
WARMUP = 5
EPOCHS = 1


def test_a_global_never_better_mu_shrinks():
    clients = make_clients()
    # 2.0 is impossible to beat, so the rule always halves mu
    local = {"A": 2.0, "B": 2.0, "C": 2.0}
    _, _, mu, history = run_personal(clients, local, N_ROUNDS, EPOCHS, warmup=WARMUP)
    for name in local:
        assert math.isclose(mu[name], 0.01), f"A: {name} mu={mu[name]}, expected 0.01"
    print("Test A passed: mu hits the 0.01 floor")


def test_b_global_always_better_mu_grows():
    clients = make_clients()
    # 0.0 is always beaten (AUROC >= 0), so the rule always doubles mu
    local = {"A": 0.0, "B": 0.0, "C": 0.0}
    _, _, mu, history = run_personal(clients, local, N_ROUNDS, EPOCHS, warmup=WARMUP)
    for name in local:
        assert math.isclose(mu[name], 1.0), f"B: {name} mu={mu[name]}, expected 1.0"
    print("Test B passed: mu hits the 1.0 ceiling")


def test_c_warmup_holds_mu():
    clients = make_clients()
    local = {"A": 0.0, "B": 0.0, "C": 0.0}
    _, _, _, history = run_personal(
        clients, local, N_ROUNDS, EPOCHS, mu_start=0.1, warmup=WARMUP
    )
    for row in history:
        if row["round"] < WARMUP:
            assert row["mu"] == 0.1, f"C: round {row['round']} mu={row['mu']}"
    print("Test C passed: mu unchanged during warm-up")


def test_d_history_complete():
    clients = make_clients()
    local = {"A": 0.5, "B": 0.5, "C": 0.5}
    _, _, _, history = run_personal(clients, local, N_ROUNDS, EPOCHS, warmup=WARMUP)
    assert len(history) == N_ROUNDS * len(clients), f"D: got {len(history)} rows"
    for r in range(N_ROUNDS):
        names = sorted(row["hospital"] for row in history if row["round"] == r)
        assert names == ["A", "B", "C"], f"D: round {r} has {names}"
    print("Test D passed: one row per hospital per round")


def test_e_nan_skips_update():
    # hospital A has a one-class validation set, so its global AUROC is NaN
    clients = make_clients(val_one_class=True)
    local = {"A": 0.0, "B": 0.0, "C": 0.0}
    _, _, mu, _ = run_personal(
        clients, local, N_ROUNDS, EPOCHS, mu_start=0.1, warmup=WARMUP
    )
    assert mu["A"] == 0.1, f"E: A mu={mu['A']}, expected 0.1 (NaN skipped)"
    assert math.isclose(mu["B"], 1.0), f"E: B mu={mu['B']}, expected 1.0"
    print("Test E passed: NaN AUROC leaves mu alone")


def test_f_personal_models_differ_from_global():
    clients = make_clients()
    local = {"A": 0.5, "B": 0.5, "C": 0.5}
    _, _, _, history = run_personal(clients, local, N_ROUNDS, EPOCHS, warmup=WARMUP)
    last = [r for r in history if r["round"] == N_ROUNDS - 1]
    for row in last:
        assert row["drift"] > 0, f"F: {row['hospital']} drift is {row['drift']}"
    print("Test F passed: personal models are not copies of global")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_a_global_never_better_mu_shrinks()
    test_b_global_always_better_mu_grows()
    test_c_warmup_holds_mu()
    test_d_history_complete()
    test_e_nan_skips_update()
    test_f_personal_models_differ_from_global()
    print("All personal.py tests passed")