#!/usr/bin/env python3
"""
Quantize a trained Stage-A eval net into the AEVNET01 blob embedded by
src/eval_net/weights.rs, and check the integer forward pass against the float
model on real records so the Rust inference cannot silently disagree.

Integer contract (mirrors src/eval_net/inference.rs):
    x_int  = feature as exported (float input = x_int / 64)
    acc1   = sum(round(w1*127) * x_int) + round(b1*127*64);  h1 = clamp(acc1 >> 6, 0, 127)
    acc2   = sum(round(w2*64) * h1)     + round(b2*127*64);  h2 = clamp(acc2 >> 6, 0, 127)
    raw    = sum(round(w3*64) * h2)     + round(b3*64*127)
    cp     = int(raw * OUT_SCALE / (64*127))

    python evalnet/export_eval_net.py --checkpoint evalnet/checkpoints/eval_net.pt --out src/eval_net/eval_net.bin
"""

import argparse
import struct

import numpy as np
import torch

from train_eval_net import EvalNet, IN_SCALE, load

MAGIC = b"AEVNET01"
VERSION = 1
S1 = 6
S2 = 6


def quant(w, scale):
    q = np.round(w * scale)
    clipped = np.abs(q) > 127
    if clipped.any():
        print(f"  warning: {int(clipped.sum())} weights clipped to i8 range")
    return np.clip(q, -127, 127).astype(np.int8)


def int_forward(net, x_int):
    """Integer forward pass on a batch, exactly as inference.rs computes it."""
    x = x_int.astype(np.int64)
    acc1 = x @ net["l1_w"].astype(np.int64).T + net["l1_b"].astype(np.int64)
    h1 = np.clip(acc1 >> S1, 0, 127)
    acc2 = h1 @ net["l2_w"].astype(np.int64).T + net["l2_b"].astype(np.int64)
    h2 = np.clip(acc2 >> S2, 0, 127)
    raw = h2 @ net["l3_w"].astype(np.int64) + int(net["l3_b"])
    return (raw.astype(np.float32) * net["out_scale"]).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="evalnet/checkpoints/eval_net.pt")
    ap.add_argument("--out", default="src/eval_net/eval_net.bin")
    ap.add_argument("--data", default="evalnet/eval_net_data.bin", help="records for the int-vs-float check")
    ap.add_argument("--check-samples", type=int, default=200000)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    n_in, hidden = ck["n_features"], ck["hidden"]
    hidden2 = ck.get("hidden2", hidden)
    model = EvalNet(n_in, hidden, hidden2)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.qat = True  # compare against the quantization-aware float path
    assert ck["in_scale"] == IN_SCALE == 64.0
    out_scale_f = float(ck["out_scale"])

    w1 = model.l1.weight.detach().numpy()
    b1 = model.l1.bias.detach().numpy()
    w2 = model.l2.weight.detach().numpy()
    b2 = model.l2.bias.detach().numpy()
    w3 = model.l3.weight.detach().numpy()[0]
    b3 = float(model.l3.bias.detach().numpy()[0])

    net = {
        "l1_w": quant(w1, 127.0),
        "l1_b": np.round(b1 * 127.0 * 64.0).astype(np.int32),
        "l2_w": quant(w2, 64.0),
        "l2_b": np.round(b2 * 127.0 * 64.0).astype(np.int32),
        "l3_w": quant(w3, 64.0),
        "l3_b": int(round(b3 * 64.0 * 127.0)),
        "out_scale": np.float32(out_scale_f / (64.0 * 127.0)),
    }

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        version = 2 if ck.get("perspective") else VERSION
        f.write(struct.pack("<IIIIII", version, n_in, hidden, hidden2, S1, S2))
        f.write(struct.pack("<Q", ck["schema"]))
        f.write(struct.pack("<f", float(net["out_scale"])))
        f.write(net["l1_w"].tobytes(order="C"))
        f.write(net["l1_b"].astype("<i4").tobytes())
        f.write(net["l2_w"].tobytes(order="C"))
        f.write(net["l2_b"].astype("<i4").tobytes())
        f.write(net["l3_w"].tobytes())
        f.write(struct.pack("<i", net["l3_b"]))
    params = w1.size + b1.size + w2.size + b2.size + w3.size + 1
    print(f"wrote {args.out}: {n_in}->{hidden}->{hidden2}->1, {params} params, schema {ck['schema']:#x}")

    try:
        _, arr = load(args.data, args.check_samples)
    except FileNotFoundError:
        print("no data file for the int/float check; skipped")
        return
    x_int = np.asarray(arr["x"])[:, :n_in].astype(np.int32) * int(ck.get("x_mult", 1))
    x_int = x_int.clip(-32767, 32767).astype(np.int16)
    with torch.no_grad():
        f_out = (model(torch.from_numpy(x_int.astype(np.float32)) / IN_SCALE) * out_scale_f).numpy()
    i_out = int_forward(net, x_int)
    diff = np.abs(i_out - f_out)
    print(
        f"int vs float on {len(x_int):,} records: mean |diff| {diff.mean():.2f} cp, "
        f"max {diff.max():.1f} cp; residual mean {f_out.mean():+.1f}, std {f_out.std():.1f}, "
        f"|res|>250: {100 * (np.abs(f_out) > 250).mean():.2f}%"
    )


if __name__ == "__main__":
    main()
