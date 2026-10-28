#!/usr/bin/env python3
"""Trains the board NNUE (src/eval_net/nnue.rs) on `export_eval_features --nnue-out` records
and writes its AINNUE01 blob.

The net predicts the whole side-to-move score (no HCE base); the engine blends it with the
generic evaluator. Each data set is a prefix with .rec/.sp/.meta files; a `prefix@w` weight
repeats it. Validation is split by game, and it reports the blend the engine plays, scored
against the same targets as the Stage-A trainer:

    target = lam * sigmoid(teacher / K) + (1 - lam) * wdl        (side to move)

Integer contract (with the Stage-A head of export_eval_net.py after the accumulators):
    acc_v = ft_b + sum(ft_w[f]) over view v's features;  a_v = clamp(acc_v, 0, 64)
    head inputs = (Stage-A inputs, a_us, a_them), each read as x_int / 64
"""

import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn

from train_eval_net import EvalNet, fake_quant, IN_SCALE, OUT_SCALE

QA = 64.0
FT_WMAX = 1.5  # |ft weight| <= 96 in i16: 300 features cannot overflow an i16 lane
S1 = S2 = 6


def rec_dtype(dense):
    return np.dtype([
        ("x", "<i2", (dense,)), ("engine", "<i2"), ("teacher", "<i2"), ("wdl", "u1"),
        ("vid", "u1"), ("source", "u1"), ("pad", "u1"), ("n0", "<u2"), ("n1", "<u2"),
        ("gid", "<u4"), ("ply", "<u2"), ("nn", "<i2"),
    ])


def load_sets(specs, device, max_records=0, sample=1.0, seed=1):
    """Each set memory-mapped; a random `sample` of its records (whole games stay
    together) and their sparse features are copied out in chunks."""
    meta, parts = None, []
    rng = np.random.default_rng(seed)
    for si, spec in enumerate(specs):
        prefix, _, w = spec.partition("@")
        m = json.load(open(prefix + ".meta"))
        assert meta is None or m == meta, f"{prefix}: layout differs from the first set"
        meta = m
        rec_all = np.memmap(prefix + ".rec", dtype=rec_dtype(m["dense"]), mode="r")
        sp_all = np.memmap(prefix + ".sp", dtype="<u2", mode="r")
        if max_records:
            rec_all = rec_all[:max_records]
        lens_all = rec_all["n0"].astype(np.int64) + rec_all["n1"].astype(np.int64)
        off_all = np.concatenate([[0], np.cumsum(lens_all)[:-1]])
        assert off_all[-1] + lens_all[-1] <= len(sp_all), f"{prefix}: sparse stream too short"
        if sample < 1.0:
            gid = np.asarray(rec_all["gid"]).astype(np.uint64)
            keep_game = rng.random(int(gid.max()) + 1) < sample
            sel = np.nonzero(keep_game[gid])[0]
        else:
            sel = np.arange(len(rec_all))
        small = [(n, rec_all.dtype.fields[n][0]) for n in rec_all.dtype.names if n != "x"]
        rec = np.empty(len(sel), dtype=small)
        x = np.empty((len(sel), m["dense"]), dtype=np.int16)
        sp_parts = []
        for i in range(0, len(sel), 1_000_000):
            s_ = sel[i : i + 1_000_000]
            chunk = rec_all[s_]
            x[i : i + len(s_)] = chunk["x"]
            for n, _ in small:
                rec[n][i : i + len(s_)] = chunk[n]
            del chunk
            ln, st = lens_all[s_], off_all[s_]
            idx = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(int(ln.sum()))
            sp_parts.append(np.asarray(sp_all[idx]))
        sp = np.concatenate(sp_parts) if sp_parts else np.zeros(0, np.uint16)
        lens = rec["n0"].astype(np.int64) + rec["n1"].astype(np.int64)
        off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        del rec_all, sp_all
        parts.append((rec, x, sp, off, int(w or 1), si))
        print(f"{prefix}: {len(rec):,} records x{int(w or 1)}, {len(sp) / max(len(rec), 1):.1f} features each")
    return meta, parts


def to_device(parts, device, x_on_cpu=False):
    """One flat set on the device; repeated sets are indexed more than once instead of copied."""
    xs, eng, tea, wdl, src, n0, n1, off, sps, gid, nn_rs, vid = [], [], [], [], [], [], [], [], [], [], [], []
    base, index = 0, []
    first = 0
    for rec, x, sp, o, w, si in parts:
        xs.append(torch.from_numpy(x))
        eng.append(torch.from_numpy(rec["engine"].astype(np.float32)))
        tea.append(torch.from_numpy(rec["teacher"].astype(np.float32)))
        wdl.append(torch.from_numpy(rec["wdl"].astype(np.float32) / 2.0))
        src.append(torch.from_numpy(rec["source"].astype(np.int64)))
        n0.append(torch.from_numpy(rec["n0"].astype(np.int32)))
        n1.append(torch.from_numpy(rec["n1"].astype(np.int32)))
        off.append(torch.from_numpy(o + base))
        sps.append(torch.from_numpy(sp.astype(np.int16)))
        gid.append(torch.from_numpy(rec["gid"].astype(np.int64) + (si << 32)))
        nn_rs.append(torch.from_numpy(rec["nn"].astype(np.int32)))
        vid.append(torch.from_numpy(rec["vid"].astype(np.int64)))
        index.append((first, len(rec), w))
        first += len(rec)
        base += len(sp)
    cat = lambda t: torch.cat(t).to(device)
    d = dict(x=(xs[0] if len(xs) == 1 else torch.cat(xs)) if x_on_cpu else cat(xs), engine=cat(eng), teacher=cat(tea), wdl=cat(wdl), source=cat(src),
             n0=cat(n0), n1=cat(n1), off=cat(off), sp=cat(sps), gid=cat(gid), nn=cat(nn_rs), vid=cat(vid))
    return d, index


class Nnue(nn.Module):
    def __init__(self, num_sparse, acc, dense, h1, h2):
        super().__init__()
        self.acc = acc
        self.ft = nn.Parameter(torch.empty(num_sparse, acc).uniform_(-0.05, 0.05))
        self.ft_b = nn.Parameter(torch.full((acc,), 0.3))
        self.head = EvalNet(dense + 2 * acc, h1, h2)
        self.qat = False

    def set_qat(self, on):
        self.qat = on
        self.head.qat = on

    def forward(self, x, flat, bag_off, batch):
        w, b = self.ft, self.ft_b
        if self.qat:
            w, b = fake_quant(w, QA), fake_quant(b, QA)
        a = nn.functional.embedding_bag(flat, w, bag_off, mode="sum") + b
        a = torch.clamp(a, 0.0, 1.0).view(batch, 2 * self.acc)
        return self.head(torch.cat([x, a], dim=1))

    def clamp_weights(self):
        with torch.no_grad():
            self.ft.clamp_(-FT_WMAX, FT_WMAX)
            self.ft_b.clamp_(-FT_WMAX, FT_WMAX)
        self.head.clamp_weights()


def gather(d, ids):
    n0, n1, start = d["n0"][ids].long(), d["n1"][ids].long(), d["off"][ids]
    lens = n0 + n1
    total = int(lens.sum())
    seg = torch.repeat_interleave(start - (torch.cumsum(lens, 0) - lens), lens)
    flat = d["sp"][seg + torch.arange(total, device=ids.device)].long()
    sizes = torch.stack([n0, n1], 1).reshape(-1)
    bag_off = torch.cumsum(sizes, 0) - sizes
    x = d["x"][ids.to(d["x"].device)].to(ids.device, non_blocking=True)
    return x.float() / IN_SCALE, flat, bag_off


def targets(d, ids, lam, k):
    l = lam[d["source"][ids].clamp(max=len(lam) - 1)]
    return l * torch.sigmoid(d["teacher"][ids] / k) + (1.0 - l) * d["wdl"][ids]


def evaluate(model, d, ids, lam, k, batch=32768):
    """Mean losses of the engine's current eval, the NNUE alone, blends at 25/50/75%, and
    the engine scaled by 0.75/0.5: blending also shrinks the eval, which alone can help."""
    model.eval()
    sums = np.zeros(7)
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            b = ids[i : i + batch]
            x, flat, bag = gather(d, b)
            nn_cp = model(x, flat, bag, len(b)) * OUT_SCALE
            t = targets(d, b, lam, k)
            e = d["engine"][b]
            outs = [e, nn_cp] + [(e * (100 - s) + nn_cp * s) / 100 for s in (25, 50, 75)] + [e * 0.75, e * 0.5]
            for j, o in enumerate(outs):
                sums[j] += float(((torch.sigmoid(o / k) - t) ** 2).sum())
    model.train()
    return sums / len(ids)


def export(model, meta, path):
    ft = np.clip(np.round(model.ft.detach().cpu().numpy() * QA), -32767, 32767).astype("<i2")
    fb = np.clip(np.round(model.ft_b.detach().cpu().numpy() * QA), -32767, 32767).astype("<i2")
    h = model.head
    q8 = lambda w, s: np.clip(np.round(w * s), -127, 127).astype(np.int8)
    w1, b1 = h.l1.weight.detach().cpu().numpy(), h.l1.bias.detach().cpu().numpy()
    w2, b2 = h.l2.weight.detach().cpu().numpy(), h.l2.bias.detach().cpu().numpy()
    w3, b3 = h.l3.weight.detach().cpu().numpy()[0], float(h.l3.bias.detach().cpu().numpy()[0])
    net = {
        "ft_w": ft, "ft_b": fb,
        "l1_w": q8(w1, 127.0), "l1_b": np.round(b1 * 127.0 * 64.0).astype("<i4"),
        "l2_w": q8(w2, 64.0), "l2_b": np.round(b2 * 127.0 * 64.0).astype("<i4"),
        "l3_w": q8(w3, 64.0), "l3_b": int(round(b3 * 64.0 * 127.0)),
        "out_scale": np.float32(OUT_SCALE / (64.0 * 127.0)),
    }
    n_in, h1, h2 = w1.shape[1], w1.shape[0], w2.shape[0]
    with open(path, "wb") as f:
        f.write(b"AINNUE01")
        f.write(struct.pack("<III", meta["num_sparse"], meta["acc"], meta["dense"]))
        f.write(ft.tobytes(order="C"))
        f.write(fb.tobytes())
        f.write(b"AEVNET01")
        f.write(struct.pack("<IIIIII", 2, n_in, h1, h2, S1, S2))
        f.write(struct.pack("<Q", meta["schema"]))
        f.write(struct.pack("<f", float(net["out_scale"])))
        for k in ("l1_w", "l1_b", "l2_w", "l2_b", "l3_w"):
            f.write(net[k].tobytes(order="C"))
        f.write(struct.pack("<i", net["l3_b"]))
    return net


def read_blob(path):
    """An AINNUE01 blob back into export()'s integer arrays."""
    b = open(path, "rb").read()
    assert b[:8] == b"AINNUE01"
    ns, acc, dense = struct.unpack_from("<III", b, 8)
    o = 20
    arr = lambda dt, n: np.frombuffer(b, dtype=dt, count=n, offset=o)
    net = {"ft_w": arr("<i2", ns * acc).reshape(ns, acc)}
    o += 2 * ns * acc
    net["ft_b"] = arr("<i2", acc)
    o += 2 * acc + 8
    _, n_in, h1, h2, _, _ = struct.unpack_from("<IIIIII", b, o)
    o += 24 + 8
    net["out_scale"] = np.float32(struct.unpack_from("<f", b, o)[0])
    o += 4
    for k, dt, n in (("l1_w", np.int8, h1 * n_in), ("l1_b", "<i4", h1), ("l2_w", np.int8, h2 * h1),
                     ("l2_b", "<i4", h2), ("l3_w", np.int8, h2)):
        net[k] = arr(dt, n)
        o += net[k].nbytes
    net["l1_w"] = net["l1_w"].reshape(h1, n_in)
    net["l2_w"] = net["l2_w"].reshape(h2, h1)
    net["l3_b"] = struct.unpack_from("<i", b, o)[0]
    return net, acc


def int_forward(net, d, ids, acc):
    """The engine's integer forward pass, on the device."""
    x = d["x"][ids.to(d["x"].device)].to(ids.device).long()
    _, flat, bag = gather(d, ids)
    ft = torch.from_numpy(net["ft_w"].astype(np.int64)).to(x.device)
    fb = torch.from_numpy(net["ft_b"].astype(np.int64)).to(x.device)
    a = nn.functional.embedding_bag(flat, ft.double(), bag, mode="sum").round().long() + fb
    a = a.clamp(0, int(QA)).view(len(ids), 2 * acc)
    xin = torch.cat([x, a], 1).cpu().numpy()
    t = lambda k: np.asarray(net[k]).astype(np.int64)
    h1 = np.clip((xin @ t("l1_w").T + t("l1_b")) >> S1, 0, 127)
    h2 = np.clip((h1 @ t("l2_w").T + t("l2_b")) >> S2, 0, 127)
    raw = h2 @ t("l3_w") + net["l3_b"]
    return (raw.astype(np.float32) * net["out_scale"]).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="export prefixes, optionally prefix@weight")
    ap.add_argument("--out", required=True, help="checkpoint (.pt); the blob goes next to it as .bin")
    ap.add_argument("--h1", type=int, default=32)
    ap.add_argument("--acc", type=int, default=0, help="accumulator width (default: the export's; the engine's ACC must match)")
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--qat-epochs", type=int, default=3, help="last epochs trained quantization-aware")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=float, default=532.0)
    ap.add_argument("--lambda-texel", type=float, default=0.7)
    ap.add_argument("--lambda-sprt", type=float, default=0.5, help="validation target (the referee)")
    ap.add_argument("--train-lambda-sprt", type=float, default=None,
                    help="training target for SPRT records (default: --lambda-sprt); 1.0 = eval only")
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--val-mod", type=int, default=20, help="games with id %% val-mod == 0 validate")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of games kept from each set")
    ap.add_argument("--x-on-cpu", action="store_true", help="keep the dense inputs in host memory (bigger sets)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-blob", help="only compare this blob's integer pass with the records' embedded output")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    meta, parts = load_sets(args.sets, dev, args.max_records, args.sample, args.seed)
    d, index = to_device(parts, dev, args.x_on_cpu)
    del parts
    if args.acc and args.acc != meta["acc"]:
        # The schema seals ACC in (src/eval_net/nnue.rs::schema_hash).
        meta = dict(meta, schema=meta["schema"] ^ (meta["acc"] << 8) ^ (args.acc << 8), acc=args.acc)
    lam = torch.tensor([args.lambda_texel, args.lambda_sprt, 1.0], device=dev)
    tl_sprt = args.lambda_sprt if args.train_lambda_sprt is None else args.train_lambda_sprt
    lam_train = torch.tensor([args.lambda_texel, tl_sprt, 1.0], device=dev)
    if args.check_blob:
        net, acc = read_blob(args.check_blob)
        ids = torch.arange(len(d["nn"]), device=dev)
        diff = int_forward(net, d, ids, acc) - d["nn"].cpu().numpy()
        print(f"{args.check_blob}: {int(np.sum(diff != 0))} of {len(ids)} records differ from the engine "
              f"(max |diff| {int(np.abs(diff).max())})")
        return

    is_val = (d["gid"] * 2654435761 % 2**32) % args.val_mod == 0
    train_ids, val_parts = [], []
    for first, n, w in index:
        ids = torch.arange(first, first + n, device=dev)
        v = is_val[first : first + n]
        val_parts.append(ids[v])
        train_ids += [ids[~v]] * w
    train_ids, val_ids = torch.cat(train_ids), torch.cat(val_parts)
    print(f"train {len(train_ids):,} (with repeats), validation {len(val_ids):,}")

    model = Nnue(meta["num_sparse"], meta["acc"], meta["dense"], args.h1, args.h2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train_ids) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    names = ("engine", "nnue", "blend25", "blend50", "blend75", "engine*0.75", "engine*0.5")
    best = None
    for ep in range(1, args.epochs + 1):
        model.set_qat(ep > args.epochs - args.qat_epochs)
        t0, tl, nb = time.time(), 0.0, 0
        perm = train_ids[torch.randperm(len(train_ids), device=dev)]
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            x, flat, bag = gather(d, b)
            out = model(x, flat, bag, len(b)) * OUT_SCALE
            loss = ((torch.sigmoid(out / args.k) - targets(d, b, lam_train, args.k)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clamp_weights()
            tl += float(loss.detach())
            nb += 1
        v = evaluate(model, d, val_ids, lam, args.k)
        gains = " ".join(f"{n} {100 * (v[0] - v[j]) / v[0]:+.2f}%" for j, n in enumerate(names) if j)
        print(f"epoch {ep} train {tl / nb:.6f} val engine {v[0]:.6f} | {gains} | qat {model.qat} | {time.time() - t0:.0f}s", flush=True)
        if model.qat and (best is None or v[3] < best):
            best = v[3]
            torch.save({"state_dict": model.state_dict(), "meta": meta, "h1": args.h1, "h2": args.h2,
                        "args": vars(args)}, args.out)

    model.load_state_dict(torch.load(args.out)["state_dict"])
    model.set_qat(True)
    blob = os.path.splitext(args.out)[0] + ".bin"
    net = export(model, meta, blob)
    ids = val_ids[:200000]
    i_out = int_forward(net, d, ids, meta["acc"])
    model.eval()
    with torch.no_grad():
        x, flat, bag = gather(d, ids)
        f_out = (model(x, flat, bag, len(ids)) * OUT_SCALE).cpu().numpy()
    diff = np.abs(i_out - f_out)
    print(f"wrote {blob}: int vs float mean |diff| {diff.mean():.2f} cp, max {diff.max():.1f}; "
          f"nnue mean {i_out.mean():+.1f}, std {i_out.std():.1f}")


if __name__ == "__main__":
    main()
