#!/usr/bin/env python3
"""Where does a net's remaining loss sit? Buckets the holdout by variant, phase, |static|,
material imbalance and pawn count; prints each bucket's share of total loss and its gain."""
import sys, numpy as np, torch
from train_eval_net import EvalNet, load, IN_SCALE, OUT_SCALE
data, ck_path = sys.argv[1], sys.argv[2]
K = 531.9
_, arr = load(data)
ck = torch.load(ck_path, map_location="cpu")
m = EvalNet(ck["n_features"], ck["hidden"], ck.get("hidden2", ck["hidden"])); m.load_state_dict(ck["state_dict"]); m.qat = True; m.eval()
x = torch.from_numpy(np.asarray(arr["x"]).astype(np.float32)) / IN_SCALE
with torch.no_grad():
    out = (m(x) * OUT_SCALE).clamp(-ck.get("cap", 500), ck.get("cap", 500)).numpy()
st = arr["static"].astype(np.float32); te = arr["teacher"].astype(np.float32); wdl = arr["wdl"].astype(np.float32) / 2
src = arr["source"]; lam = np.where(src == 0, 0.7, 0.5)
tgt = lam / (1 + np.exp(-te / K)) + (1 - lam) * wdl
sig = lambda v: 1 / (1 + np.exp(-v / K))
l = (sig(st + out) - tgt) ** 2; l0 = (sig(st) - tgt) ** 2
X = np.asarray(arr["x"]).astype(np.int32)
# feature index 25 = game.material_score/4, 29/30 = white/black pawn counts (v5 layout)
keys = {
    "variant": arr["variant"].astype(int),
    "phase": np.digitize(arr["phase"], [4, 8, 12, 16, 20]),
    "|static|": np.digitize(np.abs(st), [50, 150, 300, 600, 1200]),
    "|material|": np.digitize(np.abs(X[:, 25]) * 4, [50, 150, 300, 600, 1200]),
    "pawns": np.digitize(X[:, 29] + X[:, 30], [4, 8, 12, 16, 24]),
    "|resid|": np.digitize(np.abs(te - st - out), [50, 100, 200, 400, 800]),
}
tot = l.sum()
print(f"overall gain {100*(1-l.sum()/l0.sum()):.2f}%  n={len(l):,}")
for name, k in keys.items():
    print(f"\n== {name}")
    for v in np.unique(k):
        s = k == v
        print(f"  {v:4d}: share {100*l[s].sum()/tot:5.1f}%  n {s.sum():7,d}  mean loss {l[s].mean():.4f}  gain {100*(1-l[s].sum()/l0[s].sum()):6.2f}%")
