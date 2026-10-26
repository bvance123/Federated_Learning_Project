from fl.data import load_center

for center in [0, 1, 2, 3]:
    train, val, test = load_center(center, seed=42)

    print(f"Center {center}")
    for name, (X, y) in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        print(
            f"  {name}: X={tuple(X.shape)}, y={tuple(y.shape)}, "
            f"disease_rate={y.mean().item():.1%}"
        )