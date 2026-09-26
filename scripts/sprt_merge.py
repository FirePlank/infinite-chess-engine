#!/usr/bin/env python3
"""Merge SPRT game files (remote shards, local runs) into one games JSON and print the
harness's Final Summary: same formulas as src/bin/sprt.rs (pentanomial Elo/nElo, the
fastchess normalized/logistic GSPRT LLR, trinomial per-variant Elo), same line format.

    python scripts/sprt_merge.py --out games/sprt/games_X.json [--elo0 0 --elo1 5]
        [--model normalized] [--label NEW] [--old OLD] shard1.json shard2.json ...

Pairs are the harness's (2k, 2k+1) game indices within each shard. Every shard numbers
its games from 0, so a repeated index starts a new shard, even inside a merged file.
"""
import argparse
import json
import math
import re


def tag(icn, name):
    m = re.search(r'\[' + name + r' "([^"]*)"\]', icn)
    return m.group(1) if m else None


def result_for_new(icn):
    r, new_white = tag(icn, "Result"), tag(icn, "White") == "Apeiron New"
    if r == "1-0":
        return 1.0 if new_white else 0.0
    if r == "0-1":
        return 0.0 if new_white else 1.0
    return 0.5


def itp(f, a, b, f_a, f_b, k1, k2, n0, eps):
    if f_a > 0:
        a, b, f_a, f_b = b, a, f_b, f_a
    n_max = math.ceil(math.log2(abs(b - a) / (2 * eps))) + n0
    i = 0.0
    while abs(b - a) > 2 * eps:
        x_half = (a + b) / 2
        r = eps * 2 ** (n_max - i) - (b - a) / 2
        delta = k1 * abs(b - a) ** k2
        x_f = (f_b * a - f_a * b) / (f_b - f_a)
        sigma = math.copysign(1.0, x_half - x_f)
        x_t = x_f + sigma * delta if delta <= abs(x_half - x_f) else x_half
        x_itp = x_t if abs(x_t - x_half) <= r else x_half - sigma * r
        f_itp = f(x_itp)
        if f_itp == 0:
            a = b = x_itp
        elif f_itp < 0:
            a, f_a = x_itp, f_itp
        else:
            b, f_b = x_itp, f_itp
        i += 1
    return (a + b) / 2


def mle_logistic(scores, probs, s):
    n = len(scores)
    theta = itp(lambda x: sum(probs[i] * (scores[i] - s) / (1 + x * (scores[i] - s)) for i in range(n)),
                -1 / (scores[-1] - s), -1 / (scores[0] - s), math.inf, -math.inf, 0.1, 2.0, 0.99, 1e-3)
    return [probs[i] / (1 + theta * (scores[i] - s)) for i in range(n)]


def mle_normalized(scores, probs, mu_ref, t_star):
    n = len(scores)
    p = [1 / n] * n
    for _ in range(10):
        mu = sum(scores[i] * p[i] for i in range(n))
        sigma = math.sqrt(sum(p[i] * (scores[i] - mu) ** 2 for i in range(n)))
        phi = [scores[i] - mu_ref - 0.5 * t_star * sigma * (1 + ((scores[i] - mu) / sigma) ** 2) for i in range(n)]
        theta = itp(lambda x: sum(probs[i] * phi[i] / (1 + x * phi[i]) for i in range(n)),
                    -1 / max(phi), -1 / min(phi), math.inf, -math.inf, 0.1, 2.0, 0.99, 1e-7)
        new = [probs[i] / (1 + theta * phi[i]) for i in range(n)]
        done = max(abs(new[i] - p[i]) for i in range(n)) < 1e-4
        p = new
        if done:
            break
    return p


def score_to_elo(s):
    s = min(max(s, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / s - 1)


def penta_llr(pc, elo0, elo1, model):
    reg = lambda v: 1e-3 if v == 0 else float(v)
    ll, ld, mid, wd, ww = reg(pc[0]), reg(pc[1]), reg(pc[2]), reg(pc[3]), reg(pc[4])
    total = ll + ld + mid + wd + ww
    probs = [ll / total, ld / total, mid / total, wd / total, ww / total]
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    if model == "logistic":
        s0, s1 = (1 / (1 + 10 ** (-e / 400)) for e in (elo0, elo1))
        p0, p1 = mle_logistic(scores, probs, s0), mle_logistic(scores, probs, s1)
    else:
        c = math.sqrt(2) / (800 / math.log(10))
        p0, p1 = mle_normalized(scores, probs, 0.5, c * elo0), mle_normalized(scores, probs, 0.5, c * elo1)
    return total * sum(probs[i] * (math.log(p1[i]) - math.log(p0[i])) for i in range(5))


def penta_elo(pc):
    pairs = sum(pc)
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = sum(v * c for v, c in zip(vals, pc)) / pairs
    var = sum(c * (v - score) ** 2 for v, c in zip(vals, pc)) / pairs
    vpp = var / pairs
    z = 1.959963984540054
    up, lo = score + z * math.sqrt(vpp), score - z * math.sqrt(vpp)
    elo = -999.0 if score <= 0 else 999.0 if score >= 1 else score_to_elo(score)
    err = min((score_to_elo(up) - score_to_elo(lo)) / 2, 200.0)
    s2n = lambda s: (s - 0.5) / math.sqrt(2 * var) * (800 / math.log(10))
    nelo, nerr = (0.0, 0.0) if var <= 0 else (s2n(score), (s2n(up) - s2n(lo)) / 2)
    return elo, err, nelo, nerr


def trinomial_elo(w, l, d):
    n = w + l + d
    if n == 0:
        return 0.0, 0.0
    s = (w + 0.5 * d) / n
    if s <= 0:
        return -999.0, 0.0
    if s >= 1:
        return 999.0, 0.0
    var = (w * (1 - s) ** 2 + l * s ** 2 + d * (0.5 - s) ** 2) / n
    err = math.sqrt(var / n) * 400 / (math.log(10) * s * (1 - s))
    return -400 * math.log10(1 / s - 1), min(err, 200.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True, help="merged games JSON (one list of ICNs)")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--model", default="normalized")
    ap.add_argument("--label", default="NEW")
    ap.add_argument("--old", default="OLD")
    a = ap.parse_args()

    allg, pc = [], [0, 0, 0, 0, 0]  # ll, ld, (wl + dd), wd, ww
    w = l = d = timeouts = new_timeouts = 0
    per_var, tc = {}, None
    for path in a.inputs:
        games = json.load(open(path, encoding="utf-8"))
        allg += games
        by_idx, seg = {}, 0
        for icn in games:
            r = result_for_new(icn)
            w, l, d = w + (r == 1), l + (r == 0), d + (r == 0.5)
            if tag(icn, "Termination") == "Loss on time":
                timeouts += 1
                new_timeouts += r == 0
            v = per_var.setdefault(tag(icn, "Variant") or "?", [0, 0, 0])
            v[0 if r == 1 else 1 if r == 0 else 2] += 1
            tc = tc or tag(icn, "TimeControl")
            ev = tag(icn, "Event") or ""
            if ev.startswith("SPRT Test Game "):
                idx = int(ev.split()[-1])
                if (seg, idx) in by_idx:
                    seg += 1
                by_idx[(seg, idx)] = r
        for sg, k in {(sg, i // 2) for sg, i in by_idx}:
            if (sg, 2 * k) in by_idx and (sg, 2 * k + 1) in by_idx:
                pc[int(round((by_idx[(sg, 2 * k)] + by_idx[(sg, 2 * k + 1)]) * 2))] += 1
    json.dump(allg, open(a.out, "w", encoding="utf-8"))

    elo, err, nelo, nerr = penta_elo(pc)
    llr = penta_llr(pc, a.elo0, a.elo1, a.model)
    bound = math.log((1 - 0.05) / 0.05)
    print("Final Summary:")
    print(f"  NEW: {a.label}  vs  OLD: {a.old}")
    print(f"  TC: {tc} | Merged from {len(a.inputs)} file(s)")
    print(f"  Elo: {elo:.2f} +/- {err:.2f}")
    print(f"  nElo: {nelo:.2f} +/- {nerr:.2f}")
    print(f"  Games: {w + l + d} | W: {w} L: {l} D: {d}")
    print(f"  Pentanomial [{sum(pc)} pairs] (0-2): {pc[0]}, {pc[1]}, {pc[2]}, {pc[3]}, {pc[4]}")
    print(f"  LLR: {llr:.3f}  bounds [{-bound:.2f}, {bound:.2f}] ({a.model} model, [{a.elo0:g}, {a.elo1:g}])")
    if timeouts:
        print(f"  ALERT: {timeouts} games ended by timeout ({new_timeouts} from new) ")
    print("\nPer-Variant Breakdown:")
    for name in sorted(per_var):
        vw, vl, vd = per_var[name]
        ve, vr = trinomial_elo(vw, vl, vd)
        print(f"  [{name}]: {vw}W - {vl}L - {vd}D, Elo: {ve:.1f} +/- {vr:.1f}")


if __name__ == "__main__":
    main()
