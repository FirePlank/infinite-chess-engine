#!/usr/bin/env python3
"""
Train the Stage-A hybrid eval net: a tiny MLP over HCE scalars that predicts a
residual on top of the frozen static eval.

    loss = MSE( sigmoid((static + net(x)) / K),  lam*sigmoid(teacher/K) + (1-lam)*wdl )

Data comes from `export_eval_features` (AEVDAT01 records). Validation is split
by GAME, not position, since plies within a game are correlated.

    python nnue/train_eval_net.py --data nnue/eval_net_data.bin --epochs 30
"""

import argparse
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MAGIC = b"AEVDAT01"
HEADER = struct.Struct("<8sIIQIQ")  # magic, version, n_features, schema, record_size, count

# Quantization contract shared with src/eval_net/inference.rs and export_eval_net.py.
IN_SCALE = 64.0          # integer feature / IN_SCALE = float input
L1_WMAX = 1.0            # weights quantized as round(w * 127)
L23_WMAX = 127.0 / 64.0  # weights quantized as round(w * 64)
OUT_SCALE = 400.0        # cp per unit of net output


def read_header(path):
    with open(path, "rb") as f:
        magic, version, n_feat, schema, rec_size, count = HEADER.unpack(f.read(HEADER.size))
    assert magic == MAGIC, f"bad magic {magic!r}"
    return dict(version=version, n_features=n_feat, schema=schema, record_size=rec_size, count=count)


def record_dtype(n_feat, rec_size):
    fields = [
        ("x", np.int16, (n_feat,)),
        ("static", np.int16),
        ("teacher", np.int16),
        ("wdl", np.uint8),
        ("stm", np.uint8),
        ("variant", np.uint8),
        ("source", np.uint8),
        ("phase", np.uint8),
        ("pad", np.uint8),
        ("game", np.uint32),
        ("ply", np.uint16),
        ("pad2", np.uint16),
    ]
    dt = np.dtype(fields)
    if dt.itemsize + 2 + 2 * 320 == rec_size:
        # `--sparse` export: the dense fields plus a count and 320 sparse indices.
        dt = np.dtype(fields + [("n_sp", np.uint16), ("sp", np.uint16, (320,))])
    assert dt.itemsize == rec_size, (dt.itemsize, rec_size)
    return dt


def load(path, max_records=0, stride=1):
    h = read_header(path)
    dt = record_dtype(h["n_features"], h["record_size"])
    count = h["count"]
    if max_records:
        count = min(count, max_records)
    arr = np.memmap(path, dtype=dt, mode="r", offset=HEADER.size, shape=(count,))
    if stride > 1:
        arr = arr[::stride]
    return h, arr


def fake_quant(t, scale, fn=torch.round):
    """Straight-through fake quantization onto the integer grid `1/scale`."""
    return t + (fn(t * scale) / scale - t).detach()


class EvalNet(nn.Module):
    """Quantization-aware once `qat` is set: weights snap to the i8 grids the
    exporter uses and activations to the 0..127 CReLU grid, with the floor that
    `acc >> shift` applies, so the integer net reproduces the float one.
    `n_out=2` gives an (mg, eg) pair tapered by the record's phase (screening only)."""

    def __init__(self, n_in, h1=32, h2=32, n_out=1):
        super().__init__()
        self.l1 = nn.Linear(n_in, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, n_out)
        nn.init.zeros_(self.l3.weight)
        nn.init.zeros_(self.l3.bias)
        self.qat = False
        self.n_out = n_out

    def head(self, y, phase):
        if self.n_out == 1:
            return y.squeeze(-1)
        return (y[:, 0] * phase + y[:, 1] * (24.0 - phase)) / 24.0

    def forward(self, x, phase=None):
        if not self.qat:
            h = torch.clamp(self.l1(x), 0.0, 1.0)
            h = torch.clamp(self.l2(h), 0.0, 1.0)
            return self.head(self.l3(h), phase)
        w1 = fake_quant(self.l1.weight, 127.0)
        b1 = fake_quant(self.l1.bias, 127.0 * 64.0)
        w2 = fake_quant(self.l2.weight, 64.0)
        b2 = fake_quant(self.l2.bias, 127.0 * 64.0)
        w3 = fake_quant(self.l3.weight, 64.0)
        b3 = fake_quant(self.l3.bias, 64.0 * 127.0)
        h = torch.clamp(fake_quant(nn.functional.linear(x, w1, b1), 127.0, torch.floor), 0.0, 1.0)
        h = torch.clamp(fake_quant(nn.functional.linear(h, w2, b2), 127.0, torch.floor), 0.0, 1.0)
        return self.head(nn.functional.linear(h, w3, b3), phase)

    def clamp_weights(self):
        with torch.no_grad():
            self.l1.weight.clamp_(-L1_WMAX, L1_WMAX)
            self.l2.weight.clamp_(-L23_WMAX, L23_WMAX)
            self.l3.weight.clamp_(-L23_WMAX, L23_WMAX)


PERSPECTIVE = False
# Set for v4 exports, where king-to-cloud distance is computed correctly.
KEEP_CLOUD = False

# v2 layout: 14 term rows as (White, Black) pairs, rows 0 (net material) and 7
# (complexity delta) are single White-ahead values; then globals; then two 39-wide
# per-side blocks (White at 51, Black at 90).
_SINGLE_ROWS = (0, 7)
_PAIR_COLS = [(2 * r, 2 * r + 1) for r in range(14) if r not in _SINGLE_ROWS]
_PAIR_COLS += [(31, 32), (33, 34), (35, 36), (37, 38)]
_PAIR_COLS += [(51 + i, 90 + i) for i in range(39)]
_NEGATE_COLS = [0, 14, 29]


def to_perspective(x, stm):
    """Re-encode rows as (side to move, opponent): a position and its colour mirror
    then give identical inputs. `stm` is the record's turn (1 White, 2 Black)."""
    x = x.copy()
    blk = stm == 2
    for a, b in _PAIR_COLS:
        xa = x[blk, a].copy()
        x[blk, a] = x[blk, b]
        x[blk, b] = xa
    for c in _NEGATE_COLS:
        x[blk, c] = -x[blk, c]
    x[:, 28] = 1
    # King-to-cloud distance mixed doubled and single units in the v2 export and is not
    # mirror-equivariant; dropped until the Rust side computes it properly.
    if not KEEP_CLOUD:
        x[:, 78] = 0
        x[:, 117] = 0
    return x


def stm_sign(stm):
    """+1 when White is to move: a perspective net's output is side-to-move relative."""
    return np.where(stm == 2, -1.0, 1.0).astype(np.float32)


def to_tensors(arr, mask, device, chunk=1_000_000, x_mult=1):
    """Moves the masked records to `device` chunk by chunk: fancy-indexing the
    whole memmap at once materializes a 2GB+ host copy that small boxes lack."""
    parts = {k: [] for k in ("x", "static", "teacher", "wdl", "source", "variant", "phase")}
    for i in range(0, len(arr), chunk):
        m = mask[i : i + chunk]
        if not m.any():
            continue
        a = arr[i : i + chunk][m]
        xa = np.ascontiguousarray(a["x"])
        sign = stm_sign(np.asarray(a["stm"])) if PERSPECTIVE else None
        if PERSPECTIVE:
            xa = to_perspective(xa, np.asarray(a["stm"]))
        if x_mult != 1:
            xa = (xa.astype(np.int32) * x_mult).clip(-32767, 32767).astype(np.int16)
        parts["x"].append(torch.from_numpy(xa).to(device))
        st, te, wd = a["static"].astype(np.float32), a["teacher"].astype(np.float32), a["wdl"].astype(np.float32) / 2.0
        if sign is not None:
            # Side-to-move relative labels: the loss is unchanged by negating everything.
            st, te, wd = st * sign, te * sign, np.where(sign < 0, 1.0 - wd, wd)
        parts["static"].append(torch.from_numpy(st).to(device))
        parts["teacher"].append(torch.from_numpy(te).to(device))
        parts["wdl"].append(torch.from_numpy(wd).to(device))
        parts["source"].append(torch.from_numpy(a["source"].astype(np.int64)).to(device))
        parts["variant"].append(torch.from_numpy(a["variant"].astype(np.int64)).to(device))
        parts["phase"].append(torch.from_numpy(a["phase"].astype(np.float32)).to(device))
        del a
    return tuple(torch.cat(parts[k]) for k in ("x", "static", "teacher", "wdl", "source", "variant", "phase"))


def targets(static, teacher, wdl, source, lam_by_source, k):
    lam = lam_by_source[source]
    return lam * torch.sigmoid(teacher / k) + (1.0 - lam) * wdl


def batch_loss(model, x, static, tgt, k, cap, weight=None, phase=None):
    out = (model(x.float() / IN_SCALE, phase) * OUT_SCALE).clamp(-cap, cap)
    p = torch.sigmoid((static + out) / k)
    l = (p - tgt) ** 2
    if weight is not None:
        l = l * weight
    return l.mean(), out


def evaluate_split(model, data, lam_by_source, k, cap, batch=65536):
    x, static, teacher, wdl, source, variant, phase = data
    model.eval()
    n = x.shape[0]
    loss_sum = 0.0
    base_sum = 0.0
    big = 0
    per_variant = {}
    per_source = {}
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            tgt = targets(static[sl], teacher[sl], wdl[sl], source[sl], lam_by_source, k)
            raw = model(x[sl].float() / IN_SCALE, phase[sl]) * OUT_SCALE
            out = raw.clamp(-cap, cap)
            p = torch.sigmoid((static[sl] + out) / k)
            p0 = torch.sigmoid(static[sl] / k)
            l = (p - tgt) ** 2
            l0 = (p0 - tgt) ** 2
            loss_sum += l.sum().item()
            base_sum += l0.sum().item()
            big += (raw.abs() >= cap).sum().item()
            for key, store in ((variant[sl], per_variant), (source[sl], per_source)):
                for v in torch.unique(key).tolist():
                    m = key == v
                    a = store.setdefault(v, [0.0, 0.0, 0])
                    a[0] += l[m].sum().item()
                    a[1] += l0[m].sum().item()
                    a[2] += int(m.sum().item())
    model.train()
    return loss_sum / n, base_sum / n, big / n, per_variant, per_source


def eval_only(args):
    header, arr = load(args.data, args.max_records, args.stride)
    dev = torch.device(args.device)
    mask = np.ones(len(arr), dtype=bool)
    lam_by_source = torch.tensor([args.lambda_texel, args.lambda_sprt], device=dev)
    for ck_path in args.eval_only:
        ck = torch.load(ck_path, map_location="cpu")
        if ck["n_features"] > header["n_features"]:
            print(f"{ck_path}: needs {ck['n_features']} features, data has {header['n_features']}, skipped")
            continue
        global PERSPECTIVE
        PERSPECTIVE = bool(ck.get("perspective", False))
        global KEEP_CLOUD
        KEEP_CLOUD = bool(ck.get("keep_cloud", False))
        data = to_tensors(arr, mask, dev, x_mult=int(ck.get("x_mult", 1)))
        # Schemas only ever append, so an older net reads the leading columns.
        data = (data[0][:, : ck["n_features"]].contiguous(),) + data[1:]
        model = EvalNet(ck["n_features"], ck["hidden"], ck.get("hidden2", ck["hidden"])).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.qat = True
        cap = float(ck.get("cap", args.cap))
        loss, base, big, _, _ = evaluate_split(model, data, lam_by_source, args.k, cap)
        print(f"{ck_path}: loss {loss:.6f} baseline {base:.6f} gain {100 * (1 - loss / base):5.2f}%  n={len(arr):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nnue/eval_net_data.bin")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=32, help="layer-1 width")
    ap.add_argument("--hidden2", type=int, default=32, help="layer-2 width")
    ap.add_argument("--k", type=float, default=531.9, help="logistic scale, texel DEFAULT_K_SCALE")
    ap.add_argument("--lambda-texel", type=float, default=0.7)
    ap.add_argument("--lambda-sprt", type=float, default=0.5)
    ap.add_argument("--lambda-src2", type=float, default=1.0, help="teacher weight for source 2 (perturbed positions)")
    ap.add_argument("--val-frac", type=float, default=0.05, help="fraction of GAMES held out")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth record")
    ap.add_argument("--x-mult", type=int, default=1, help="integer input pre-scale (Stage B uses 32)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qat-from", type=int, default=4, help="epoch from which fake quantization is on")
    ap.add_argument("--cap", type=float, default=250.0, help="residual cap applied in the loss (must match RESIDUAL_CAP)")
    ap.add_argument("--texel-weight", type=float, default=1.0, help="loss weight of fixed-depth (source 0) records")
    ap.add_argument("--phase-split", action="store_true", help="(mg, eg) output pair tapered by phase; screening only")
    ap.add_argument("--max-resid", type=float, default=0.0, help="drop training records with |teacher-static| above this (0 = keep all)")
    ap.add_argument("--out", default="nnue/checkpoints/eval_net.pt")
    ap.add_argument("--init", default=None, help="warm-start weights from this checkpoint")
    ap.add_argument("--n-cols", type=int, default=0, help="train on only the leading N feature columns")
    ap.add_argument("--perspective", action="store_true", help="(side to move, opponent) encoding")
    ap.add_argument("--keep-cloud", action="store_true", help="data has the fixed king-to-cloud distance (v4)")
    ap.add_argument("--distill", default=None, help="teacher checkpoint whose outputs are blended into targets")
    ap.add_argument("--distill-alpha", type=float, default=0.8, help="weight of the teacher in the target")
    ap.add_argument("--eval-only", nargs="*", default=None, metavar="CKPT",
                    help="score these checkpoints on --data (whole file as test set) and exit")
    args = ap.parse_args()
    global PERSPECTIVE, KEEP_CLOUD
    PERSPECTIVE = args.perspective
    KEEP_CLOUD = args.keep_cloud
    if args.eval_only is not None:
        return eval_only(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    header, arr = load(args.data, args.max_records, args.stride)
    n_feat = args.n_cols or header["n_features"]
    print(f"records={len(arr):,} features={n_feat} schema={header['schema']:#x} device={args.device}")

    game = np.asarray(arr["game"])
    # Hash the game id so the split is stable across re-exports that keep ids.
    h = (game.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    val_mask = (h % 1000) < int(args.val_frac * 1000)
    print(f"train={int((~val_mask).sum()):,} val={int(val_mask.sum()):,}")

    dev = torch.device(args.device)
    try:
        train = to_tensors(arr, ~val_mask, dev, x_mult=args.x_mult)
        val = to_tensors(arr, val_mask, dev, x_mult=args.x_mult)
        train = (train[0][:, :n_feat].contiguous(),) + train[1:]
        val = (val[0][:, :n_feat].contiguous(),) + val[1:]
    except RuntimeError as e:  # out of GPU memory: fall back to CPU tensors
        print("GPU load failed, using CPU:", e)
        dev = torch.device("cpu")
        train = to_tensors(arr, ~val_mask, dev, x_mult=args.x_mult)
        val = to_tensors(arr, val_mask, dev, x_mult=args.x_mult)

    lam_by_source = torch.tensor([args.lambda_texel, args.lambda_sprt, args.lambda_src2], device=dev)
    model = EvalNet(n_feat, args.hidden, args.hidden2, 2 if args.phase_split else 1).to(dev)
    schema_out = header["schema"]
    if args.init:
        init_ck = torch.load(args.init, map_location=dev)
        if init_ck["n_features"] == n_feat:
            schema_out = init_ck["schema"]
        sd = init_ck["state_dict"]
        old_in = sd["l1.weight"].shape[1]
        if old_in < n_feat:
            # Net surgery: layouts only append, so the old net's columns lead and
            # the new inputs start at zero weight (the net begins exactly as before).
            w = torch.zeros(sd["l1.weight"].shape[0], n_feat, device=dev)
            w[:, :old_in] = sd["l1.weight"]
            sd["l1.weight"] = w
        model.load_state_dict(sd)
        print(f"warm-started from {args.init} ({old_in} -> {n_feat} inputs)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = train[0].shape[0]
    steps_per_epoch = math.ceil(n_train / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )

    v_loss, v_base, v_big, _, _ = evaluate_split(model, val, lam_by_source, args.k, args.cap)
    print(f"epoch 0  val {v_loss:.6f}  baseline {v_base:.6f}  (zero-residual gain 0.00%)")
    best = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    x, static, teacher, wdl, source, _, phase = train
    if args.max_resid > 0:
        keep = (teacher - static).abs() <= args.max_resid
        x, static, teacher, wdl, source, phase = (
            x[keep], static[keep], teacher[keep], wdl[keep], source[keep], phase[keep]
        )
        n_train = x.shape[0]
        print(f"max-resid filter keeps {n_train:,} training records")
    src_weight = torch.tensor([args.texel_weight, 1.0, 1.0], device=dev)
    teacher_net = None
    if args.distill:
        tck = torch.load(args.distill, map_location=dev)
        teacher_net = EvalNet(tck["n_features"], tck["hidden"], tck.get("hidden2", tck["hidden"])).to(dev)
        teacher_net.load_state_dict(tck["state_dict"])
        teacher_net.qat = True
        teacher_net.eval()
        t_cap = float(tck.get("cap", args.cap))
        print(f"distilling from {args.distill} (alpha {args.distill_alpha})")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch == args.qat_from:
            model.qat = True
            best = float("inf")  # only quantization-aware checkpoints are exportable
        perm = torch.randperm(n_train, device=dev)
        run = 0.0
        for i in range(0, n_train, args.batch):
            idx = perm[i : i + args.batch]
            tgt = targets(static[idx], teacher[idx], wdl[idx], source[idx], lam_by_source, args.k)
            if teacher_net is not None:
                with torch.no_grad():
                    t_out = (teacher_net(x[idx].float() / IN_SCALE, phase[idx]) * OUT_SCALE).clamp(-t_cap, t_cap)
                    t_p = torch.sigmoid((static[idx] + t_out) / args.k)
                tgt = args.distill_alpha * t_p + (1.0 - args.distill_alpha) * tgt
            w = src_weight[source[idx]] if args.texel_weight != 1.0 else None
            loss, _ = batch_loss(model, x[idx], static[idx], tgt, args.k, args.cap, w, phase[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clamp_weights()
            run += loss.item() * idx.shape[0]
        v_loss, v_base, v_big, per_variant, per_source = evaluate_split(
            model, val, lam_by_source, args.k, args.cap
        )
        gain = 100.0 * (1.0 - v_loss / v_base)
        print(
            f"epoch {epoch:2d}  train {run / n_train:.6f}  val {v_loss:.6f}  baseline {v_base:.6f}"
            f"  gain {gain:5.2f}%  |out|>=cap: {100 * v_big:.2f}%  ({time.time() - t0:.0f}s)"
        )
        if v_loss < best:
            best = v_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_features": n_feat,
                    "hidden": args.hidden,
                    "hidden2": args.hidden2,
                    "schema": schema_out,
                    "in_scale": IN_SCALE,
                    "out_scale": OUT_SCALE,
                    "k": args.k,
                    "cap": args.cap,
                    "perspective": args.perspective,
                    "keep_cloud": args.keep_cloud,
                    "x_mult": args.x_mult,
                    "val_loss": v_loss,
                    "val_baseline": v_base,
                },
                args.out,
            )

    print("\nper-source val loss (net / baseline / n):")
    for s, (a, b, n) in sorted(per_source.items()):
        print(f"  source {s}: {a / n:.6f} / {b / n:.6f}  gain {100 * (1 - a / b):5.2f}%  n={n:,}")
    print("per-variant val loss (net / baseline / n):")
    for v, (a, b, n) in sorted(per_variant.items(), key=lambda kv: -kv[1][2]):
        print(f"  variant {v:3d}: {a / n:.6f} / {b / n:.6f}  gain {100 * (1 - a / b):5.2f}%  n={n:,}")
    print(f"\nbest val {best:.6f} saved to {args.out}")


if __name__ == "__main__":
    main()
