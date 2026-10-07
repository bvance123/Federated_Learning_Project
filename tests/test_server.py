"""Tests for server.py using fake client updates (no real data or client code needed).

Run from the repo root:  python -m pytest tests   (or)   python tests/test_server.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.model import Model, get_weights, set_weights  # noqa: E402
from fl.server import aggregate, drift, run_rounds  # noqa: E402


def _const_weights(model, value):
    return {k: torch.full_like(v, value) for k, v in get_weights(model).items()}


def test_model_shapes():
    for hidden in (0, 16):
        m = Model(hidden=hidden)
        p = m(torch.randn(5, 13))
        assert p.shape == (5,)
        assert ((p > 0) & (p < 1)).all()


def test_aggregate_is_weighted_by_samples():
    m = Model()
    w = aggregate([_const_weights(m, 1.0), _const_weights(m, 4.0)], [300, 100])
    # (300 * 1 + 100 * 4) / 400 = 1.75
    for v in w.values():
        assert torch.allclose(v, torch.full_like(v, 1.75))


def test_aggregate_identical_updates_unchanged():
    m = Model()
    w0 = get_weights(m)
    w = aggregate([w0, w0, w0], [10, 50, 200])
    for k in w0:
        assert torch.allclose(w[k], w0[k])


def test_run_rounds_with_fake_clients():
    m = Model()
    set_weights(m, _const_weights(m, 0.0))

    def fake_update(model, global_weights, train_data, mu, epochs):
        # each fake client pushes every weight up by its own offset
        offset = float(train_data[1][0])
        return {k: v + offset for k, v in global_weights.items()}

    clients = [
        {"name": "a", "train": (torch.zeros(30, 13), torch.full((30,), 1.0))},
        {"name": "b", "train": (torch.zeros(10, 13), torch.full((10,), 3.0))},
    ]
    hist = run_rounds(m, clients, fake_update, n_rounds=2, mu=0.0, epochs=1, verbose=False)

    # each round adds (30*1 + 10*3)/40 = 1.5, so after 2 rounds every weight is 3.0
    for v in get_weights(m).values():
        assert torch.allclose(v, torch.full_like(v, 3.0))
    assert len(hist) == 2
    assert abs(hist[0]["drift"]["a"] - drift(
        {k: v + 1.0 for k, v in _const_weights(m, 0.0).items()}, _const_weights(m, 0.0))) < 1e-5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
