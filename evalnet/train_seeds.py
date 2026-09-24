#!/usr/bin/env python3
"""Train several seeds of the eval net at once, on one copy of the data.

The net is small enough that one run leaves the GPU mostly idle, so the seeds are
stacked into batched weight tensors and stepped together, each with its own
initialisation (as `train_eval_net.py --seed s` would draw it) and its own shuffle.
AdamW acts per element, so the optimisation is the same as separate runs. Each seed
is written as an ordinary checkpoint.

    python evalnet/train_seeds.py --data mixrel.bin --seeds 1,2,3,4,5,6 --epochs 120 \\
        --perspective --out base_s{s}.pt
    python evalnet/train_seeds.py --data rel.bin --seeds 1,2,3,4,5,6 --init base_s{s}.pt \\
        --epochs 20 --lr 2e-4 --qat-from 1 --val-frac 0.1 ... --out net_s{s}.pt
"""
import argparse
import math
import time

import numpy as np
import torch

import train_eval_net as tev
from train_eval_net import (IN_SCALE, L1_WMAX, L23_WMAX, OUT_SCALE, EvalNet, bare_king_mop_up,
                            fake_quant, load, targets, to_tensors)


class Stack(torch.nn.Module):
    """N independent nets as batched weights: layer tensors are [N, out, in]."""

    def __init__(self, nets):
        super().__init__()
        cat = lambda f: torch.nn.Parameter(torch.stack([f(n).detach().clone() for n in nets]))
        self.w1, self.b1 = cat(lambda n: n.l1.weight), cat(lambda n: n.l1.bias)
        self.w2, self.b2 = cat(lambda n: n.l2.weight), cat(lambda n: n.l2.bias)
        self.w3, self.b3 = cat(lambda n: n.l3.weight), cat(lambda n: n.l3.bias)
        self.qat = False

    @staticmethod
    def lin(x, w, b):
        return torch.baddbmm(b.unsqueeze(1), x, w.transpose(1, 2))

    def forward(self, x):  # x: [N, B, n_in] -> [N, B]
        if not self.qat:
            h = torch.clamp(self.lin(x, self.w1, self.b1), 0.0, 1.0)
            h = torch.clamp(self.lin(h, self.w2, self.b2), 0.0, 1.0)
            return self.lin(h, self.w3, self.b3).squeeze(-1)
        w1, b1 = fake_quant(self.w1, 127.0), fake_quant(self.b1, 127.0 * 64.0)
        w2, b2 = fake_quant(self.w2, 64.0), fake_quant(self.b2, 127.0 * 64.0)
        w3, b3 = fake_quant(self.w3, 64.0), fake_quant(self.b3, 64.0 * 127.0)
        h = torch.clamp(fake_quant(self.lin(x, w1, b1), 127.0, torch.floor), 0.0, 1.0)
        h = torch.clamp(fake_quant(self.lin(h, w2, b2), 127.0, torch.floor), 0.0, 1.0)
        return self.lin(h, w3, b3).squeeze(-1)

    def clamp_weights(self):
        with torch.no_grad():
            self.w1.clamp_(-L1_WMAX, L1_WMAX)
            self.w2.clamp_(-L23_WMAX, L23_WMAX)
            self.w3.clamp_(-L23_WMAX, L23_WMAX)

    def state_dict_of(self, i):
        return {"l1.weight": self.w1[i].detach().clone(), "l1.bias": self.b1[i].detach().clone(),
                "l2.weight": self.w2[i].detach().clone(), "l2.bias": self.b2[i].detach().clone(),
                "l3.weight": self.w3[i].detach().clone(), "l3.bias": self.b3[i].detach().clone()}


def val_losses(model, data, lam, k, cap, n, batch=65536):
    """Per-seed validation loss and the shared zero-residual baseline."""
    x, static, teacher, wdl, source, _, _ = data
    loss = torch.zeros(n, device=x.device, dtype=torch.float64)
    base = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            sl = slice(i, i + batch)
            tgt = targets(static[sl], teacher[sl], wdl[sl], source[sl], lam, k)
            xb = (x[sl].float() / IN_SCALE).unsqueeze(0).expand(n, -1, -1)
            out = (model(xb) * OUT_SCALE).clamp(-cap, cap)
            p = torch.sigmoid((static[sl] + out) / k)
            loss += ((p - tgt) ** 2).sum(1).double()
            base += ((torch.sigmoid(static[sl] / k) - tgt) ** 2).sum().item()
    m = x.shape[0]
    return (loss / m).tolist(), base / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="1,2,3", help="comma-separated seeds, trained together")
    ap.add_argument("--out", required=True, help="checkpoint path with {s} for the seed")
    ap.add_argument("--init", default=None, help="per-seed warm-start checkpoint path with {s}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--k", type=float, default=531.9)
    ap.add_argument("--cap", type=float, default=500.0)
    ap.add_argument("--lambda-texel", type=float, default=0.7)
    ap.add_argument("--lambda-sprt", type=float, default=0.5)
    ap.add_argument("--lambda-src2", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--qat-from", type=int, default=4)
    ap.add_argument("--n-cols", type=int, default=0)
    ap.add_argument("--perspective", action="store_true")
    ap.add_argument("--layout", default="")
    ap.add_argument("--keep-mop-up", action="store_true")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    n = len(seeds)
    tev.PERSPECTIVE = args.perspective
    tev.LAYOUT = tev.parse_layout(args.layout)

    header, arr = load(args.data, args.max_records)
    n_feat = args.n_cols or header["n_features"]
    game = np.asarray(arr["game"])
    h = (game.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    val_mask = (h % 1000) < int(args.val_frac * 1000)
    keep = np.ones(len(arr), dtype=bool) if args.keep_mop_up else ~bare_king_mop_up(arr)
    dev = torch.device(args.device)
    train = to_tensors(arr, ~val_mask & keep, dev)
    val = to_tensors(arr, val_mask & keep, dev)
    train = (train[0][:, :n_feat].contiguous(),) + train[1:]
    val = (val[0][:, :n_feat].contiguous(),) + val[1:]
    print(f"records={len(arr):,} train={train[0].shape[0]:,} val={val[0].shape[0]:,} seeds={seeds} device={dev}", flush=True)

    nets, schemas = [], []
    for s in seeds:
        torch.manual_seed(s)
        net = EvalNet(n_feat, args.hidden, args.hidden2)
        schema = header["schema"]
        if args.init:
            ck = torch.load(args.init.format(s=s), map_location="cpu")
            net.load_state_dict(ck["state_dict"])
            if ck["n_features"] == n_feat:
                schema = ck["schema"]
        nets.append(net)
        schemas.append(schema)
    model = Stack(nets).to(dev)
    lam = torch.tensor([args.lambda_texel, args.lambda_sprt, args.lambda_src2], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x, static, teacher, wdl, source, _, _ = train
    m = x.shape[0]
    steps = math.ceil(m / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    gens = [torch.Generator(device=dev).manual_seed(s) for s in seeds]
    best = [float("inf")] * n

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch == args.qat_from:
            model.qat = True
            best = [float("inf")] * n  # only quantization-aware checkpoints are exportable
        perms = torch.stack([torch.randperm(m, device=dev, generator=g) for g in gens])
        for i in range(0, m, args.batch):
            idx = perms[:, i : i + args.batch]  # [N, B]
            xb = x[idx].float() / IN_SCALE
            tgt = targets(static[idx], teacher[idx], wdl[idx], source[idx], lam, args.k)
            out = (model(xb) * OUT_SCALE).clamp(-args.cap, args.cap)
            p = torch.sigmoid((static[idx] + out) / args.k)
            # Summing per-seed means keeps each seed's gradient exactly its own.
            loss = ((p - tgt) ** 2).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clamp_weights()
        losses, base = val_losses(model, val, lam, args.k, args.cap, n)
        for j, s in enumerate(seeds):
            if losses[j] < best[j]:
                best[j] = losses[j]
                torch.save({"state_dict": model.state_dict_of(j), "n_features": n_feat, "hidden": args.hidden,
                            "hidden2": args.hidden2, "schema": schemas[j], "in_scale": IN_SCALE,
                            "out_scale": OUT_SCALE, "k": args.k, "cap": args.cap,
                            "perspective": args.perspective, "layout": list(tev.LAYOUT) if tev.LAYOUT else None,
                            "x_mult": 1,
                            "val_loss": losses[j], "val_baseline": base}, args.out.format(s=s))
        gains = " ".join(f"{100 * (1 - l / base):5.2f}" for l in losses)
        print(f"epoch {epoch:3d}  val gain % per seed: {gains}  ({time.time() - t0:.0f}s)", flush=True)
    print("best val per seed: " + " ".join(f"s{s}={b:.6f}" for s, b in zip(seeds, best)))


if __name__ == "__main__":
    main()
