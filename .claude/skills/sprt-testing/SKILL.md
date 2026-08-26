---
name: sprt-testing
description: Rules and exact commands for running an SPRT match test on the Apeiron engine. Use BEFORE launching ANY SPRT (target/release/sprt.exe run ...) — covers which binary to build, variant scoping, model/bounds selection, launch mechanics, LLR-based decisions, and the commit-message format. Invoke every time before testing an engine change.
---

# Running an SPRT test (Apeiron)

Follow these rules exactly.

## 1. The engine binary is the `sprt` binary — NOT `uci.exe`

The harness runs the engine via `serve` (one persistent process per game, TT/history
stay warm move to move), falling back to per-move `search` only if either binary
predates it. Both subcommands exist ONLY in `src/bin/sprt.rs` behind `--features sprt`.
`uci.exe` has neither — passing it makes EVERY game "Loss on engine failure" (the
tell-tale that you used the wrong binary).

Build a baseline/engine binary:
`cargo build --release --features sprt --bin sprt`, then copy `target/release/sprt.exe`.

## 2. THE HARNESS DOES NOT REBUILD ANYTHING — you must build the sprt bin yourself

"No --new-bin provided. Using current binary..." means it literally uses the
`target/release/sprt.exe` FILE AS-IS. And that bin is feature-gated
(`required-features = ["sprt"]` in Cargo.toml), so a plain `cargo build --release`
SILENTLY SKIPS IT — the exe stays stale and you run a NULL TEST (engine vs itself;
2026-07-21 incident: several "results" including a committed change were noise from this).

**Mandatory sequence after ANY source edit, before launching:**
```
cargo build --release --features sprt --bin sprt
```
**Null-test tripwire — SKIP IT IF THE NODE ORACLE ALREADY ANSWERED.** The point is only "is the
change actually in the binary". If you already ran the deterministic oracle
(`cargo test --release --test perft run_search_only_suite`, or `negamax_node_count_for_depth` on a
relevant position) and it moved — or provably stayed identical because the change is gated to
variants the oracle doesn't cover — that IS the liveness proof. Don't also run a per-position
tripwire; go straight to the SPRT.

⚠️ **`sprt.exe search --max-depth N` does NOT drive a deep search** (it passes time 0, so it
returns after a trivial number of nodes — an Obstocean position gave the same 47 nodes at depths
8→20). So it is useless as a tripwire for anything but the shallowest change, and `--fixed-time`
node counts jitter run-to-run, so a fixed-time diff is not proof either. Use the oracle. Only fall
back to a per-position comparison when no oracle path covers the change, and then compare via
`negamax_node_count_for_depth` (deterministic), not the `search` subcommand.
For an eval-only change, a pure static/positional change may leave `nodes` equal but shift `score`
at fixed depth — the `score` delta is the valid liveness signal (e.g. cache-collision fixes only
diverge mid-search, so pick a position that actually searches). Prefer building BOTH old and new
from git commits over trusting a saved/copied baseline `.exe` of uncertain provenance.

### `--new-bin` timeout artifact (separate trap)
A freshly-copied exe passed via `--new-bin` is AV-scanned per subprocess spawn → NEW loses on
time (`ALERT: N games ended by timeout (NEW ENGINE ONLY)` + low draw count = contaminated,
discard). So: NEW = the featured-built `target/release/sprt.exe` (no `--new-bin`), OLD = a
pre-built baseline via `--old-bin` (repo root, does not show the artifact).

## 3. Build/run location — repo dir ONLY

Never build or run an SPRT engine binary from Temp/scratchpad — per-move AV-scan latency
causes timeout losses. Build into `target/release/`; keep baseline binaries in the REPO ROOT
(e.g. `sprt_old.exe`). Pass `--old-bin`/`--new-bin` as ABSOLUTE paths (Rust `Command::new`
won't find a bare relative name in cwd on Windows).

**Reuse one baseline name (`sprt_old.exe`, `nps_old.exe`/`nps_new.exe`) and delete it once
the run reports** — untracked exes named per-commit accumulate invisibly (one session left
33 stray exes, 52 MB). Kill engine processes first or the delete fails ("Access is denied").

**Games/results JSON go in `<REPO>/games/sprt/`, NEVER the repo root and never a
session scratchpad.** `games/` is gitignored, so the corpus accumulates there and
stays out of `git status`; a scratchpad copy dies with the session, and the root
fills with stray `games_*.json` that later have to be swept up by hand. That
corpus is reused by `puzzle_gen`/`texel` as training data, so keeping every run
in one place is what makes it worth anything.

`--concurrency` defaults to physical core count (`num_cpus::get_physical()`), not logical —
each game is single-threaded, so SMT siblings only add contention, not real parallelism.
Passing a manual `--concurrency` above physical core count reproduces the same timeout
inflation as the zombie incident below, without needing any zombies to cause it.

## 4. Scope variants to what the change touches

Testing all site variants on a change that affects a few buries the signal in noise.

- Whole-engine change (search / movegen / TT / eval wrapper): `--variants site`.
- base.rs-only (Generic evaluator): default preset (no `--variants`).
- Piece/variant-specific: list only the relevant variants.

**NEVER include Abundance.** It draws almost everything (measured 153D of 170 games, ±8.4 Elo), so it
contributes dilution instead of signal and drags the aggregate toward zero. Exclude it even when it
contains the exact piece the change targets — it has a chancellor, and it still had to be filtered out
of the archbishop/chancellor movegen test afterwards (+8.7 → +11.1 once removed).

Fairy-piece location map (from `starting_icn` in src/lib.rs):
- **Knightrider (`nr`)**: `CoaIP_NO`, `Scattered_Leapers`. (Knightline has KNIGHTS `n`, not knightriders.)
- **Rose (`ro`)**: `CoaIP_RO`, `Scattered_Leapers`.
- **Huygen (`hu`)**: `CoaIP_HO`.
- **Amazon**: Palace. **Chancellor/Archbishop**: Standarch, Space, CoaIP*. **Hawk/Centaur**: Space, CoaIP*.

## 5. Model & bounds — pick per the change

The SPRT is **pentanomial + normalized Elo (nElo)** by default (`--model normalized`). nElo
bounds are draw-rate/TC independent — pick them by the change's intent:

| Scenario | Bounds | Meaning |
|----------|--------|---------|
| Gainer, short TC (default new feature/tune) | `--elo0 0 --elo1 2` | prove a real gain |
| Gainer, long TC | `--elo0 0.5 --elo1 2.5` | gain that survives depth |
| Simplification / refactor / cleanup (prove NOT a regression) | `--elo0 -1.75 --elo1 0.25` | accept small losses, reject real ones |
| Risky rewrite where a tiny loss is unacceptable | `--elo0 -0.5 --elo1 1.5` | tight non-regression |

Keep α=β=0.05 (⇒ LLR decision bounds ≈ **[−2.94, +2.94]**). Adjudication stays OFF.
Use `--model logistic` only to reproduce old-style raw-Elo bounds — do not mix scales.

## 6. Game counts (initial + follow-up)

Set `--min-games` ~400 (noise floor) and `--max-games` to the initial batch below. The harness
auto-stops the moment the LLR crosses a formal bound, so decisive changes end early on their
own — the batch size is just the cap for the undecided case.

| Scope | Initial `--max-games` | Follow-up (`--resume`, if still undecided) |
|-------|----------------------|--------------------------------------------|
| Scoped (few variants) | 1500 | +1500 |
| Whole engine (`site`) | 2500 | +2500 |

Extend only when it pays off — decide from |LLR| at the end of a batch (§8). Follow-ups use
`--resume <games.json>` with the SAME binaries.

## 7. Launch mechanics

- **KILL EVERY EXISTING `sprt` PROCESS FIRST — MANDATORY.** Stopping a background task does NOT
  reap the engine children it spawned; they keep burning CPU forever. Leftovers oversubscribe the
  box and BOTH engines start losing on time, which reads as a huge fake regression (2026-07-29:
  zombies from an aborted run produced a bogus **−40 Elo** with **71% of games lost on time** —
  the baseline alone timed out 27% vs 2% clean). Run before every launch:
  ```
  Get-Process -Name sprt -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
  ```
  Then confirm it returns nothing. Zombies also lock `target/release/sprt.exe`, so a rebuild fails
  with "Access is denied (os error 5)" / "Device or resource busy" — that error means zombies, kill
  and retry. **Contamination tell:** timeouts ≫ a few % or the baseline timing out at a similar
  rate to NEW ⇒ the run is VOID, kill everything and rerun; do not interpret the Elo.
- Run in the background (`run_in_background: true`). Do NOT append `&` and do NOT pipe through
  `head`/`tail` — either kills or truncates the streaming SPRT. (`nohup ... &` is the classic way
  to strand zombies: the wrapper returns, the match keeps running unsupervised.)
- **NEVER set a watch/monitor/poll loop on a running SPRT, and never sleep waiting on one.** The
  background task auto-notifies on completion — just END THE TURN after launching. Extra watchers
  burn a core (the match is already sized to the physical cores), add nothing, and a polling loop
  is the same oversubscription that fakes regressions. Same rule for the post-launch "is it really
  running" check: one glance at the output file is fine, a loop is not.
- **Let the run FINISH (or `--resume` it to completion) before quoting numbers** — the
  `Final Summary` block and the `--results` JSON are only written at the end. Killing early leaves
  you with no per-variant breakdown to paste into the commit (§9).
- **Negative bounds MUST use `="..."` syntax** (`--elo0="-1.75"`), else clap parses `-1.75` as a
  flag (`unexpected argument '-1'`) AND the shell can mangle it. Always quote: `--elo0="-1.75" --elo1="0.25"`.
- Standard invocation:
  ```
  ./target/release/sprt.exe run \
    --old-bin "<REPO>/sprt_old.exe" [--new-bin "<REPO>/base_new.exe"] \
    --old-commit <sha> --new-commit <label> \
    --variants "<scoped,list>" \
    --elo0 <lo> --elo1 <hi> \
    --games "<REPO>/games/sprt/games_<tag>.json" --results "<REPO>/games/sprt/results_<tag>.json" \
    --max-games <N>
  ```

## 8. Deciding — by LLR (time-optimal, not always ±2.94)

The LLR is the decision statistic; read it live from `... | LLR: X [-2.94, 2.94]`. The formal
bound is ±2.94 (95%), but **don't reflexively wait for it** — accepting needs confidence
scaled to risk, while rejecting is cheap (you just don't ship), so abandon losers early.

**Accept threshold — by how risky/reversible the change is:**

| Change type | Accept when LLR ≥ |
|-------------|-------------------|
| Simple / low-risk / trivially reversible (cleanup, small tweak, obvious speedup) | **+1.0** |
| Normal | **+2.94** |
| Risky / has revert history / hard to verify (search reworks, multi-royal, TT) | **+2.94** AND clean per-variant breakdown |

**Reject threshold (any gainer test):** LLR ≤ **−0.5** → revert. No need to prove a loss to
95%; stop burning games on it. (Simplification tests use their own `[-1.75, 0.25]` bounds)

**Extend vs stop — from |LLR| at end of a batch:**

- **|LLR| ≥ 1.0**, heading toward a bound → almost there; let it finish / one small follow-up.
  If a follow-up finishes and LLR is STILL ≥ +1.0 (even short of +2.94), accept — it has
  already survived more games without dropping below the accept line.
- **0.5 ≤ |LLR| < 1.0** → genuinely undecided; this is where games pay off → run a full follow-up.
- **|LLR| < 0.5** after the initial batch, still < 0.5 after one follow-up → effectively
  **neutral** (a flat LLR won't move with more games — stop). Decide by intent: **accept
  simplifications** (free simplicity), **reject gainers** (complexity for no gain).
- **Neutrality cap:** stop chasing after ~2 follow-ups with |LLR| < 1.0.

**Large-sample override:** at **≥4000 total games**, a solidly positive point estimate is
enough on its own — accept if nElo/Elo is clearly positive (e.g. ≥ +5 Elo-equivalent) and its
error bar does not cross 0, even if LLR hasn't hit +2.94. That many games without the sign
flipping negative IS the confidence; don't keep grinding past this just to satisfy a formal
bound.

To stop early, kill the run (Ctrl-C) once the threshold is met; `--resume` can continue it
later if you change your mind. The nElo/Elo point-estimates are context only — decide on LLR.

Then check the **per-variant breakdown**: a change can PASS overall while wrecking one variant
class → prefer variant-gating (eval_kind) over reverting the whole change.

Correctness/rare-endgame fixes read as neutral (LLR near 0) — validate those with unit tests
and mate suites, not by waiting for an SPRT bound.

## 9. Commit message (when a change PASSES)

Subject line, then a **1–2 line description only**, then paste the harness's `Final Summary`
block **verbatim** — do NOT hand-condense it. `.github/workflows/auto-release.yml` parses the
commit body with a regex that requires each variant on its own `[Name]: …, Elo: X +/- Y` line;
a condensed multi-per-line summary (no brackets) is invisible to it and silently breaks
auto-release. **No AI attribution / Co-Authored-By / "Generated with" trailer.**

Correct — paste exactly what `sprt.exe` printed:
```
Elo: 3.6 +/- 4.8
Record: 870W - 839L - 1291D (3000 total)

Per-Variant Breakdown:
  [Palace]: 80W - 81L - 15D, Elo: -2.0 +/- 25.0
  [Knightline]: 81W - 73L - 22D, Elo: 15.8 +/- 24.5
  [Space]: 51W - 50L - 75D, Elo: 2.0 +/- 19.8
  ... (one line per variant, brackets required)
```

## Pre-flight checklist

1. Engine binary is the `sprt` binary, repo root, absolute path.
2. Variants scoped to what the change touches (§4).
3. Bounds chosen for the change's intent (§5); model = normalized.
4. Game count per §6; background, no `&`, no pipe.
5. Decide by LLR per §8 (risk-scaled accept, early reject, neutral→intent); extend with `--resume` only when 0.5 ≤ |LLR| < 2.94.
