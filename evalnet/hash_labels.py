#!/usr/bin/env python3
"""Carry depth-N labels across an HCE change, which alters every feature key, by
matching positions on the Zobrist hashes the exporter writes with --hash-out.

    python evalnet/hash_labels.py table labels.bin positions.bin positions.hash table.bin [n]
        key-join positions to labels (as join_labels.py), then write (hash, teacher)
        pairs; the hashes alone (first 8 bytes of each pair) feed --keep-hashes
    python evalnet/hash_labels.py apply table.bin positions.bin positions.hash out.bin
        give each position the teacher of its hash and write the labelled records
"""
import sys

import numpy as np

from join_labels import keys, open_any
from train_eval_net import HEADER

PAIR = np.dtype([("hash", "<u8"), ("teacher", "<i2")])


def read_hashes(path, n):
    h = np.fromfile(path, dtype="<u8")
    assert len(h) == n, f"{path}: {len(h):,} hashes for {n:,} records"
    return h


def write_records(path, header, recs):
    with open(path, "wb") as f:
        f.write(HEADER.pack(b"AEVDAT01", header["version"], header["n_features"], header["schema"],
                            header["record_size"], len(recs)))
        recs.tofile(f)


def table(lab_path, pos_path, hash_path, out_path, n=None):
    hl, lab = open_any(lab_path)
    hp, pos = open_any(pos_path)
    n = int(n) if n else min(hl["n_features"], hp["n_features"])
    hashes = read_hashes(hash_path, len(pos))
    kl, kp = keys(lab, n), keys(pos, n)
    order = np.argsort(kl)
    idx = np.clip(np.searchsorted(kl[order], kp), 0, len(kl) - 1)
    hit = kl[order][idx] == kp
    pairs = np.empty(int(hit.sum()), dtype=PAIR)
    pairs["hash"] = hashes[hit]
    pairs["teacher"] = np.asarray(lab["teacher"])[order[idx[hit]]]
    pairs.tofile(out_path)
    pairs["hash"].tofile(out_path + ".keep")
    print(f"positions {len(pos):,}  joined {len(pairs):,} ({100 * hit.mean():.1f}%)  "
          f"-> {out_path}, {out_path}.keep")


def apply(table_path, pos_path, hash_path, out_path):
    pairs = np.fromfile(table_path, dtype=PAIR)
    hp, pos = open_any(pos_path)
    hashes = read_hashes(hash_path, len(pos))
    order = np.argsort(pairs["hash"])
    th = pairs["hash"][order]
    idx = np.clip(np.searchsorted(th, hashes), 0, len(th) - 1)
    hit = th[idx] == hashes
    out = np.array(pos[hit])
    out["teacher"] = pairs["teacher"][order[idx[hit]]]
    out["source"] = 0
    write_records(out_path, hp, out)
    print(f"positions {len(pos):,}  labelled {len(out):,} ({100 * hit.mean():.1f}%) -> {out_path}")


if __name__ == "__main__":
    {"table": table, "apply": apply}[sys.argv[1]](*sys.argv[2:])
