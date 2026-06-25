"""
test_handoff.py — verifies the handoff contract 
Run: python test_handoff.py   (expects handoff/ to exist)
"""
import json
import numpy as np
import joblib
from pathlib import Path

OUT = Path("handoff")
EXPECTED_FEATURES = 30
WINDOW_SIZE = 26

def main():
    splits = dict(np.load(OUT / "splits.npz"))
    windows = dict(np.load(OUT / "windows.npz"))
    scaler = joblib.load(OUT / "scaler.pkl")
    meta = json.loads((OUT / "metadata.json").read_text())
    schema = json.loads((OUT / "schema.json").read_text())

    checks = []
    def check(name, cond):
        checks.append((name, bool(cond)))

    # 1. no NaNs anywhere
    check("no NaN in splits", not any(np.isnan(a).any() for a in splits.values()))
    check("no NaN in windows", not any(np.isnan(a).any() for a in windows.values()))

    # 2. consistent feature dimension
    check("splits feature dim == 30", all(a.shape[1] == EXPECTED_FEATURES for a in splits.values()))
    check("windows feature dim == 30", all(a.shape[2] == EXPECTED_FEATURES for a in windows.values()))
    check("window length == 26", all(a.shape[1] == WINDOW_SIZE for a in windows.values()))

    # 3. dtype preserved
    check("splits are float32", all(a.dtype == np.float32 for a in splits.values()))
    check("windows are float32", all(a.dtype == np.float32 for a in windows.values()))

    # 4. anti-leakage: TRAIN standardized (~mean0/std1), VAL/TEST drift away
    tr = splits["train"]
    check("train mean ~ 0", abs(tr.mean()) < 0.05)
    check("train std ~ 1", abs(tr.std() - 1.0) < 0.05)
    val_mean = abs(splits["val"].mean())
    check("val NOT re-standardized (mean drifts from 0)", val_mean > 0.01)

    # 5. feature_order matches scaler width
    check("feature_order length == scaler", len(schema["feature_order"]) == scaler.n_features_in_)

    # 6. a 32-sample batch can be produced
    check("train has >=32 windows", windows["train"].shape[0] >= 32)
    batch = windows["train"][:32]
    check("batch shape (32,26,30)", batch.shape == (32, WINDOW_SIZE, EXPECTED_FEATURES))

    # report
    npass = sum(c for _, c in checks)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(checks)} passed")
    assert npass == len(checks), "HANDOFF CONTRACT FAILED"
    print("Handoff contract OK — safe to send to Eleni.")

if __name__ == "__main__":
    main()
