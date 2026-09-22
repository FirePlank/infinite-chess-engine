#!/usr/bin/env python3
"""Copy depth-N teacher labels from a relabelled export onto a re-export of the same
positions in a newer layout, matching records by their leading feature columns and
static eval (layouts only append, so those columns are identical).

    python nnue/join_labels.py labels.bin positions.bin out.bin
"""
import struct
import sys

import numpy as np

from train_eval_net import HEADER, read_header, record_dtype


def open_any(path):
    h = read_header(path)
    dt = record_dtype(h["n_features"], h["record_size"])
    return h, np.memmap(path, dtype=dt, mode="r", offset=HEADER.size, shape=(h["count"],))


def keys(arr, n):
    x = np.ascontiguousarray(arr["x"][:, :n])
    st = np.ascontiguousarray(arr["static"]).view(np.uint16).astype(np.uint64)
    # 64-bit FNV-style fold over the leading columns plus the static eval.
    h = np.full(len(arr), 0xCBF29CE484222325, dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for c in range(n):
        h = (h ^ x[:, c].view(np.uint16).astype(np.uint64)) * prime
    return (h ^ st) * prime


def main():
    if sys.argv[1] == "--dump-keys":
        h, arr = open_any(sys.argv[2])
        n = int(sys.argv[4]) if len(sys.argv) > 4 else h["n_features"]
        keys(arr, n).astype("<u8").tofile(sys.argv[3])
        print(f"wrote {len(arr):,} keys")
        return
    lab_path, pos_path, out_path = sys.argv[1:4]
    hl, lab = open_any(lab_path)
    hp, pos = open_any(pos_path)
    n = int(sys.argv[4]) if len(sys.argv) > 4 else min(hl["n_features"], hp["n_features"])
    kl, kp = keys(lab, n), keys(pos, n)
    order = np.argsort(kl)
    kl_sorted = kl[order]
    idx = np.searchsorted(kl_sorted, kp)
    idx = np.clip(idx, 0, len(kl_sorted) - 1)
    hit = kl_sorted[idx] == kp
    src = order[idx[hit]]
    out = np.array(pos[hit])
    out["teacher"] = lab["teacher"][src]
    out["source"] = 0
    with open(out_path, "wb") as f:
        f.write(HEADER.pack(b"AEVDAT01", hp["version"], hp["n_features"], hp["schema"],
                            hp["record_size"], len(out)))
        out.tofile(f)
    print(f"labels {len(lab):,}  positions {len(pos):,}  joined {len(out):,} "
          f"({100 * hit.mean():.1f}% of positions)")


if __name__ == "__main__":
    main()
