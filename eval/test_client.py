# eval/test_client.py
import torch
from fl.model import Model, get_weights
from fl.client import local_update


def dist(a, b):
    """L2 distance between two state_dicts."""
    total = 0.0
    for k in a:
        total += ((a[k] - b[k]) ** 2).sum().item()
    return total ** 0.5


def make_fake_hospital(n=200, seed=0):
    """Fake patients: 13 features, disease driven mostly by feature 0 plus noise."""
    torch.manual_seed(seed)
    X = torch.randn(n, 13)
    y = ((X[:, 0] + 0.5 * torch.randn(n)) > 0).float()
    return X, y


def run_tests():
    X, y = make_fake_hospital()
    loss_fn = torch.nn.BCELoss()

    torch.manual_seed(1)
    g = get_weights(Model())          # fake "global" weights (random start)

    # ---- Test 1: training at mu = 0 should reduce the loss ----
    m = Model()
    m.load_state_dict(g)
    before = loss_fn(m(X), y).item()

    torch.manual_seed(2)
    w0 = local_update(Model(), g, (X, y), mu=0, epochs=50)
    m.load_state_dict(w0)
    after = loss_fn(m(X), y).item()

    print(f"Test 1  loss before: {before:.4f}  after: {after:.4f}")
    assert after < before, "FAIL: loss did not decrease at mu = 0"

    # ---- Test 2: input global weights must not be modified ----
    g_copy = {k: v.clone() for k, v in g.items()}   # copy BEFORE training
    w0 = local_update(Model(), g, (X, y), mu=0, epochs=50)
    assert dist(g, g_copy) == 0.0, "FAIL: global weights were changed by local_update"
    print("Test 2  global weights untouched: OK")
    # ---- Test 3: bigger mu should mean less drift ----
    print("Test 3  drift from global vs mu:")
    drifts = []
    for mu in [0, 0.1, 1, 10, 50]:
        torch.manual_seed(2)          # same shuffling each run, so only mu differs
        w = local_update(Model(), g, (X, y), mu=mu, epochs=50)
        d = dist(w, g)
        drifts.append(d)
        print(f"        mu = {mu:<5}  drift = {d:.4f}")

    for i in range(len(drifts) - 1):
        assert drifts[i] > drifts[i + 1], "FAIL: drift did not shrink as mu increased"
    assert drifts[-1] < 0.2 * drifts[0], "FAIL: large mu did not keep weights near global"

    # ---- Test 4: mu = 0 vs a tiny mu should be nearly identical ----
    torch.manual_seed(2)
    w_a = local_update(Model(), g, (X, y), mu=0, epochs=50)
    torch.manual_seed(2)
    w_b = local_update(Model(), g, (X, y), mu=1e-9, epochs=50)
    gap = dist(w_a, w_b)
    print(f"Test 4  mu=0 vs mu=1e-9 difference: {gap:.2e}")
    assert gap < 1e-3, "FAIL: tiny mu should behave like mu = 0"


        # ---- Test 5: init_weights ----
    torch.manual_seed(5)
    v = get_weights(Model())                 # a different starting point than g
    assert dist(v, g) > 1e-3, "FAIL: v should differ from g"

    # 5a: init_weights=g behaves exactly like not passing it
    torch.manual_seed(2)
    w_none = local_update(Model(), g, (X, y), mu=1, epochs=20)
    torch.manual_seed(2)
    w_g = local_update(Model(), g, (X, y), mu=1, epochs=20, init_weights=g)
    gap = dist(w_none, w_g)
    print(f"Test 5a init_weights=g vs none, difference: {gap:.2e}")
    assert gap < 1e-6, "FAIL: init_weights=g should match the default"

    # 5b: with zero epochs, the result is exactly where we told it to start
    w_v = local_update(Model(), g, (X, y), mu=1, epochs=0, init_weights=v)
    print(f"Test 5b epochs=0: dist to v = {dist(w_v, v):.2e}, dist to g = {dist(w_v, g):.4f}")
    assert dist(w_v, v) < 1e-6, "FAIL: should start from init_weights"
    assert dist(w_v, g) > 1e-3, "FAIL: should not have started from global_weights"

    # 5c: the leash pulls toward g, not toward where we started (v)
    torch.manual_seed(2)
    w_pull = local_update(Model(), g, (X, y), mu=50, epochs=50, init_weights=v)
    d_before = dist(v, g)
    d_after = dist(w_pull, g)
    print(f"Test 5c start at v, mu=50: dist to g before = {d_before:.4f}, after = {d_after:.4f}")
    assert d_after < 0.2 * d_before, "FAIL: leash should pull toward the global weights"

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()