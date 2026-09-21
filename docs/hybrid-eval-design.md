# Hybrid HCE + Small-Net Evaluation — Design Document

Status: Stage A implemented (see §11); Stages B/C remain proposals. Scope: `evaluation/` for the generic eval path,
new training tooling, new inference module. Everything here is SPRT-gated and
individually disableable.

---

## 1. Goal and constraints

Augment the hand-crafted evaluation (`src/evaluation/base.rs`) with one or more
small neural networks that:

- add strength without replacing the HCE (residual corrections, not absolute eval);
- are cheap enough to run at every eval call (< a few % NPS);
- train on data volumes we can actually produce (millions of positions, not billions);
- cover as many variants/pieces as possible — ideally including fairy pieces
  (huygen, knightrider, rose, ...) and custom positions never seen in training;
- degrade gracefully: any net can be skipped per-position or disabled entirely,
  falling back to pure HCE.

## 2. Why InfNNUE-v1 failed (diagnosis)

The previous attempt (`src/nnue/`, trainer in `nnue/`) was a two-stream NNUE:
RelKP (25,450 → 256) + ThreatEdges (6,768 → 64), head 640→32→32→1.

1. **Parameter count vs data.** The feature transformers alone hold
   ~6.9M parameters (25,450×256 + 6,768×64). Nets this size are trained in
   standard chess on hundreds of millions to billions of positions. We can't
   produce that.
2. **Absolute evaluation target.** The net had to re-learn material, king
   safety, passers — everything the HCE already knows — before it could add
   anything. That is precisely the most data-hungry part of the task.
3. **Closed piece vocabulary.** One-hot piece codes for P/N/B/R/Q/K only
   (`src/nnue/features.rs::get_piece_code`), single king per side
   (`src/nnue/mod.rs::is_applicable`). No fairy pieces, no multi-royal, no
   generalization mechanism — a huygen has no representation at all.
4. **Runtime cost.** The threat stream was rebuilt from scratch per eval and
   the accumulator refresh is O(pieces × 256).

Every one of these is addressed by going small, residual, and
attribute-encoded rather than identity-encoded.

## 3. Design principles

1. **Residual on top of HCE.** The net predicts a clamped correction added to
   the HCE score. The target (what the HCE *misses*) has far lower variance
   than the eval itself, so a small net with little data suffices. Prior art:
   Ethereal 12.58's pawn-king net — a [224,32,1] network (~7.2k params)
   *augmenting* the HCE, trained as `sigmoid((net + static_eval)/K) → win rate`,
   gained **+26 Elo** with <1% slowdown (AndyGrant/Ethereal commit `513a93c`).
2. **Small.** 5–50k parameters per net. Trainable overnight on CPU/consumer GPU
   from a few million positions.
3. **Translation invariance.** The board is unbounded (`i64` coords); there are
   no absolute squares. All spatial inputs are king-relative or cloud-relative
   with log-binned distances (the `relkp_bucket` near/far scheme in
   `src/nnue/features.rs` is the right shape and is reusable).
4. **Generalize by construction.** Where piece identity matters, describe
   pieces by *movement attributes* (ortho/diag slider, leaper class, value,
   royal, colorbound, ...) instead of one-hot identity. An unseen piece maps
   into the input space automatically. Prior art direction: Winter's
   piece-relation GNN.
5. **Reuse what the eval already computes.** The single-pass loop in
   `evaluate_inner_traced` already produces king-ray arrays, tropism units,
   pawn metrics, cloud statistics, threat counts. Feeding these to a net costs
   nearly nothing.
6. **Quantized integer inference.** The existing i8/i16 kernels in
   `src/nnue/inference.rs` (`dot_product_i8_i16_chunked` etc.) are reusable
   as-is.
7. **Bounded failure.** Output clamp (±200cp initially), cargo feature +
   runtime toggle per net, tracer row so `debug_evaluate` shows the residual,
   per-variant SPRT before merging (per the `sprt-testing` skill; eval-term
   changes are historically SPRT-fragile — see AGENTS.md).

## 4. Staged architecture

Three nets, one shared data/training pipeline. Each stage ships (or dies)
independently.

### 4.1 Stage A — HCE-feature residual MLP ("term mixer") — build first

**What it is.** A tiny MLP over a fixed vector of scalars the eval loop
already computes. It learns the *non-linear interactions between existing
terms* — exactly the thing that keeps failing when hand-tuned (king safety ×
phase × counterplay, development × board spread, ...).

**Why first.**
- Universal by construction: every fairy piece, multi-royal setup, and custom
  position is already abstracted into these features. No applicability gate
  needed (unlike InfNNUE's `is_applicable`).
- Near-zero runtime cost: features are already computed; the net is one
  ~128→32→32→1 quantized forward pass (~5k multiply-adds, well under 1µs).
- Smallest implementation surface; validates the entire
  data→train→quantize→SPRT pipeline that B and C then reuse.

**Input vector (~96–128 dims).** All from the side-to-move's perspective as
(us, them) pairs plus shared globals. Sourced from locals in
`evaluate_inner_traced` and `compute_pawn_core`/`score_passed_pawns`:

- *Global*: effective phase (`effective_phase`), initial-phase ratio, total
  piece count (log-binned), pawn rank spread (`pawn_max_y - pawn_min_y`,
  log-binned), wall/void density (`wall_count`, `void_count`), bounded-board
  flag + log world size (`moves::get_world_size`), win-condition flags
  (checkmate / allpiecescaptured / allroyalscaptured per side), royal count
  per side.
- *Material*: `material_score` (clamped, in pawns), non-pawn-non-royal count,
  pawn counts, bishop-pair/colorbound flags, diag/ortho slider counts
  (`w_diag_count` etc.), counterplay units (`white_cp`/`black_cp`).
- *King safety* (per side, first royal + aggregates over extra royals):
  the 8 king-ray tuples reduced to: count of open rays, count of rays covered
  by cheap friendly piece at dist ≤2, nearest-enemy-slider distance bin and
  value bucket per line class (diag/ortho), ring-cover flag
  (`king_rays_from_indices`), attacking/defender tropism units
  (`RoyalTropismMetrics.attacking_units`, `defender_units`,
  `defender_units_in_distance` collapsed to 2–3 bins), sliders-in-zone
  (`w_sliders_in_zone`), additional attack units, pawn-storm totals.
- *Pawns*: doubled/connected/candidate counts, passer count, best-passer
  promo-distance bin, passer king-distance bins (from `score_passed_pawns`
  inputs), pawns-past-promo count.
- *Activity/threats*: threat points (`w_threat_points`), queen-threat flags,
  pawn/minor/slider threat totals, undeveloped counts, cloud spread
  (`cloud_spread_sum/cloud_count`), mean cloud distance excess.

Normalization: counts log1p-binned to small ints; distances via the existing
`dist_bin` log scheme; everything clamped to i16 input range at extraction
time. The exact list is frozen in one Rust function (see §7) used by both the
exporter and inference so train/infer can never disagree (the mistake the old
NNUE guarded against by hand-duplication in `gen_nnue_data.rs`).

**Net.** `[N≈128] → 32 (CReLU) → 32 (CReLU) → 1`, i8 weights / i16
activations, ≈ 5–9k params. Output scaled to centipawns, clamped to ±200,
added to the generic eval before mop-up/damping (see §7).

**Expected data need.** ~2–5M positions is comfortable (≥200 samples/param
even at the high end).

### 4.2 Stage B — Pawn-King net (Ethereal-style, unbounded-board adaptation)

**What it is.** A network over pawn and king placement only — the aspect of
eval that linear HCE terms are worst at (pawn-structure interactions, king
shelter shapes, passer races), and the one with the strongest prior art
(+26 Elo in Ethereal, on top of an already strong pawn eval).

**Inputs.** Ethereal used 4×56/64 absolute-square bitboard planes; we cannot
(unbounded board). Instead, per pawn, sum-pooled per color:

- king-relative bucket to own king and to enemy king, using a *coarsened*
  `relkp_bucket` (near zone ±4 exact = 81 buckets, far zone sign×log ≈ 30
  buckets → ~111 per king; pawns beyond huge distances saturate);
- promo-distance bin (existing 8-bin scheme from `features.rs::promo_dist_bin`
  — already handles shifted/multiple promotion ranks);
- file-parity / adjacency summary relative to nearest friendly pawn
  (connected/isolated/doubled indicator bits — cheap since
  `EVAL_WHITE_PAWNS`/`EVAL_BLACK_PAWNS` are sorted).

Plus king-vs-king relative bucket and royal-count flags.
Feature space ≈ 2 sides × (111×2 buckets × 8 promo bins + flags) ≈ ~4k sparse
features → 32-dim embedding (≈130k params in the embedding is fine — it's
sparse and pawn-hash-cached; if data proves thin, drop to promo-bin × coarse
bucket ≈ 1k features / 32k params).

**Net.** sparse-embed → 32 (CReLU) → (mg, eg) pair, tapered like other pawn
terms with `effective_phase`.

**Caching — the reason this net is nearly free.** Split into two streams:
1. *Pawn-only stream* (promo bins, adjacency, pawn-vs-pawn shape): pure
   function of the pawn set → cached in the existing 2-bucket pawn cache
   keyed by `game.pawn_hash` (`PawnCacheEntry` gains an `[i16; 32]`
   accumulator; `evaluate_pawn_structure_traced` already has the probe/store
   plumbing).
2. *King-relative stream*: recomputed live — O(pawns) embedding adds — or
   cached under `pawn_hash ^ hash(king buckets)`. Ethereal saw 92% hit rates
   on the equivalent cache.

**Coverage.** Any position with ≥1 pawn; king-relative stream needs ≥1 royal
per side (all current variants qualify; multi-royal uses the first royal +
count flag, same convention as the HCE's ray code). Fairy pieces are
irrelevant to it — full variant coverage.

**Interaction with Stage A.** Fully compatible: disjoint inputs (A sees
computed scalar terms, B sees raw pawn/king geometry), both add residuals.
Train sequentially on residuals: train A, freeze, train B on what remains
(prevents double-counting). Expect roughly additive gains. If A's pawn-count
features start shadowing B, drop the few overlapping scalars from A's input
and retrain — the feature list is versioned (§7).

### 4.3 Stage C (optional, after A/B verdicts) — piece-descriptor deep-set net

**What it is.** The full-generality option: a permutation-invariant
("deep set") net over all pieces, where a piece is encoded by *what it moves
like*, not its name — so huygens, knightriders, roses, and even pieces absent
from training data get a meaningful representation.

**Per-piece encoding.**
- *Descriptor* (~12–16 dims, hand-built table over `PieceType`): ortho-slider,
  diag-slider, rider-of-leap flag, leap-pattern class (knight/camel/zebra/
  giraffe/king-step), max-range class (1 / n / ∞), value bucket
  (`get_piece_value_base` log-binned), phase weight (`get_piece_phase`),
  royal, colorbound, pawn-like (promotes), capture-only-diagonal.
  A compound (amazon = Q+N) is the OR/sum of its parts' flags — new compounds
  compose automatically.
- *Position features*: log-binned (dx,dy) sign+magnitude to own king, to enemy
  king (existing `dist_bin`/`sign_code`), distance-to-cloud-center bin,
  promo-distance bin when pawn-like.

Per piece: descriptor (d) and position (p) vectors combine via a factorized
bilinear map `W(d ⊗ p)` → 32-dim per-piece embedding (keeps params at
~d×p×32 ≈ 16×24×32 ≈ 12k rather than a full outer-product table), CReLU,
sum-pool per side → head `64 → 32 → 1`, clamped residual. Total ≈ 20–40k
params, cost O(pieces × ~48 MAdds) — same order as one extra eval term.

**Risks.** Highest of the three: descriptors are a lossy hand-designed
bottleneck (a rose's spiral ≠ a knightrider's ray, yet both are
"rider-of-knight-leap"); interactions between pieces are only captured via
pooling. Only attempt after A/B have proven the pipeline, and expect to need
the larger corpus (§5). Judge per variant class — a net that helps
ScatteredLeapers/CoaIP but is neutral elsewhere is still a win if gated.

### 4.4 What was considered and rejected

- **Full NNUE replacement (retry).** Same data wall as before; rejected.
- **Threat-edge stream (reviving ThreatEdges).** Redundant with Stage A's
  threat scalars at far lower cost; the 6,768-feature version is the old
  data-hungry design. Not retried before C.
- **Per-variant nets / variant one-hot input.** Fragments the training data
  and dies on custom positions. Variant-*property* inputs (win condition,
  boundedness, initial phase) are used instead — they generalize.
- **Multiple heads sharing a trunk (user's "combine multiple nets").** The
  staged A+B (+C) design *is* this, just trained sequentially on residuals
  instead of jointly — same expressive power, much simpler to ship/gate/SPRT
  one piece at a time. A joint fine-tune pass over all active nets is a
  possible later refinement.

## 5. Training data

### 5.1 Sources (both exist today)

1. **`games/texel_corpus.jsonl`** — `data_gen` output: ~10.3k games,
   fixed-depth (default depth 15, 15s/move cap), per-position records with
   White-ahead static eval, search score, depth, quiet flag, phase
   (`data_gen.rs::PositionRecord`). High label quality, modest volume
   (~0.3–0.5M quiet positions).
2. **`games/sprt/*.json`** — 661 files, **5.4 GB** of annotated ICN games from
   SPRT runs: full move lists with per-move `[%eval]` and results, across all
   variants including fairy-heavy ones. Order of 1–2M games → tens of millions
   of positions. Caveats: 10+0.1 evals (shallow), mixed engine versions across
   months (label noise), duplicated openings, and both engines in a pair are
   near-identical (fine — we want on-policy positions).

### 5.2 Export strategy (one tool serves all stages)

New binary `src/bin/export_eval_features.rs` (feature-gated like the other
tools) that:

1. Reads either corpus format, replays each game from its start ICN
   (`setup_position_from_icn` + move application — same machinery as
   `texel.rs` extraction; **one variant/bounds group at a time**, since
   `moves::set_world_bounds` is process-global — parallelism stays inside a
   group, per AGENTS.md).
2. At each recorded position: recomputes the **current** HCE static eval and
   the Stage-A feature vector (and Stage-B pawn/king features), so features
   and the residual baseline always reflect today's `base.rs`, regardless of
   which engine version played the game. Recorded evals are only ever
   *targets*, never features.
3. Filters: quiet only (texel corpus: `quiet` flag; sprt corpus: reconstruct —
   |recorded eval − recomputed static| ≤ 150cp, not in check, ply ≥ 14,
   |score| < 2000cp, ≥4 pieces — mirroring `data_gen`'s quiet definition),
   dedupe by zobrist within a file, skip variants routed to non-generic
   evaluators (`EvalKind::Chess/Obstocean/PawnHorde`) since the net won't run
   there (§7).
4. Writes a flat binary (feature vector + static_cp + teacher_cp + wdl +
   variant tag + phase), versioned header with a feature-schema hash.

### 5.3 Targets and loss

WDL-space, Ethereal/texel style, using the engine's fitted scale
K = 531.9 (`texel.rs::DEFAULT_K_SCALE`):

```
p_target = λ · sigmoid(teacher_cp / K) + (1 − λ) · wdl        λ ≈ 0.7
loss     = MSE( sigmoid((static_cp + net(x)) / K), p_target )
```

The net is inside the sigmoid with the frozen HCE value — it directly learns
the correction that best calibrates the *combined* eval to outcomes. For the
noisy sprt corpus, lower λ (~0.4–0.5) so game results dominate over stale
evals; per-source λ is a training-script flag. Mate-adjacent scores
(|cp| ≥ `search::MATE_SCORE` floor) are excluded.

### 5.4 Volumes and fresh generation

- **Stage A** (5–9k params): existing texel corpus + a filtered slice of the
  sprt corpus is already ample (≥2M quiet positions). A fresh `data_gen` run
  is *optional* quality insurance: `data_gen --games 30000 --depth 9
  --variants base_only` (~1 day) gives ~1.5M+ high-quality quiet positions
  with consistent labels. Recommended but not blocking.
- **Stage B** (~30–130k params): add the sprt corpus in full (pawn/king
  features are cheap to extract). Target ≥5–10M positions.
- **Stage C** (20–40k params but a much harder function class): the sprt
  corpus's fairy-variant games are the key asset (CoaIP*, Space, Abundance,
  Palace, ScatteredLeapers...). Target ≥10–30M positions; supplement with a
  multi-day `data_gen --variants base_full` run if C shows promise on the
  existing data.

## 6. Training recipe

Python/PyTorch under `nnue/` alongside the existing scripts (reuse the
quantization/export skeleton of `export_innue.py`):

- Optimizer AdamW, LR 1e-3, cosine decay, batch 8192, 20–40 epochs (minutes
  to an hour at these sizes, CPU-feasible).
- Quantization-aware clamps during training matching inference (CReLU [0,127],
  weight clipping to i8 range / scale) — same discipline as the old trainer.
- Validation: held-out *games* (not positions — positions within a game are
  correlated), plus a per-variant residual-MSE report so a net that only helps
  Classical is visible before SPRT.
- Export: `export_eval_net.py` → little-endian binary with magic, dims,
  feature-schema hash, scales; embedded via `include_bytes!` like
  `src/nnue/innue.bin`.

## 7. Rust integration

New module `src/eval_net/` (the old `src/nnue/` stays untouched until A ships,
then can be retired):

- `features.rs` — `fn stage_a_features(game, &EvalAccumulators) -> [i16; N]`.
  The eval loop's locals are gathered into a small `EvalAccumulators` struct
  populated at the end of `evaluate_inner_traced` (they're all in scope
  there); the same function is called by the exporter binary, so the schema
  cannot fork. Schema hash checked against the weights file at startup.
- `inference.rs` — quantized forward pass reusing the chunked dot-product
  kernels; no allocation, no floats until the final scale.
- Hook: in `evaluation/mod.rs::evaluate`, `EvalKind::Generic` arm only —
  `score += net_residual.clamp(-200, 200)` computed inside `base::evaluate`
  (where the accumulators live), *before* `compute_mop_up_term`,
  `apply_*_scale` and rule50 damping, so all existing safety scaling applies
  to the combined value. Tracer row `"NetResidual"` in `debug_evaluate`.
- Gating: cargo feature `eval_net` (on by default once passed) + a runtime
  kill-switch (UCI option / env) for A/B testing; `EvalKind::Chess/Obstocean/
  PawnHorde` untouched initially (their evaluators wrap base with different
  term mixes; extending the net there is a follow-up decision).
- Stage B: extend `PawnCacheEntry` with the pawn-stream accumulator (the
  clone-on-store cost of +64 bytes is trivial next to the existing passer
  SmallVecs); king-stream computed in `evaluate_pawn_structure_traced` after
  the cache probe.
- Skill limiter (`EvalStyle`): residual is a full-strength term; apply no
  style scaling (weak levels already misjudge via attack/defense scaling).

## 8. Rollout and verification

Per stage, in order:

1. Implement exporter path + feature schema; unit test: exporter features ==
   inference features on a set of ICN positions (the train/infer-consistency
   test the old NNUE needed).
2. Train; sanity-check residual MSE beats the zero-residual baseline on
   held-out games, per variant.
3. Integrate; `cargo test --release --lib -- --test-threads=1`, clippy, node
   oracle (shift expected — it's an eval change).
4. **Invoke the `sprt-testing` skill** and run the standard SPRT. Decision by
   LLR per its rules; check per-variant lines — a net may be gated off for a
   variant class rather than rejected outright.
5. Commit with the Final Summary block per AGENTS.md conventions.

Kill criteria: Stage A rejected twice (different feature sets) → stop, write
up in ATTEMPTS.md. Stage B is attempted regardless of A's verdict (independent
mechanism, strongest prior art). Stage C only after at least one of A/B passes.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Eval-term SPRT fragility (AGENTS.md: most plausible eval changes test negative) | Residual clamp; net inside existing damping/scaling chain; per-variant gating; kill criteria |
| Train/infer feature drift | Single Rust feature function used by both; schema hash in weights file |
| Label noise in sprt corpus (mixed versions, shallow evals) | Recompute static/features at export; lower λ toward WDL; texel corpus & fresh fixed-depth run as clean core |
| Net double-counts HCE terms (A) or pawn terms (B) | Residual-on-frozen-HCE target makes double-counting unprofitable in training; sequential residual training between nets |
| NPS regression | A: features already computed, ~5k MAdds. B: pawn-hash cached. Measure with `nps_bench` before SPRT |
| Overfit to variant mix of training data | Variant-property inputs instead of variant identity; per-variant validation report; custom-position spot checks via `eval_icn` |
| `PlayerColor::Neutral=0/White=1/Black=2` indexing bug (recurring, per AGENTS.md) | Feature extraction indexes per-color arrays via explicit match, never `as usize` arithmetic |

## 10. Decision summary

- **Build Stage A first**: universal (all fairy pieces, multi-royal, custom
  positions), cheapest, validates the pipeline; it attacks the exact pain
  point (non-linear term interactions that hand-tuning keeps failing at).
- **Stage B second**: strongest prior art (+26 Elo class in Ethereal),
  near-zero cost via the pawn cache; combines additively with A.
- **Stage C**: the fairy-generalization moonshot, attempted only on top of a
  proven pipeline and only with the sprt-corpus data volume.

## 11. Stage A implementation status (2026-09-21)

- Code: `src/eval_net/{features,weights,inference,mod}.rs`; tracer handoff
  `record_inputs` in `base.rs` (compiled out unless `T::WANTS_INPUTS`); residual
  added in `base::evaluate` under cargo feature `eval_net` (default on), runtime
  kill-switch `APEIRON_EVAL_NET=0`; `debug_evaluate` shows a `Net Residual` row.
- Data: `src/bin/export_eval_features.rs` (`--features data_gen`) replays the
  texel corpus and every `games/sprt/*.json` archive, recomputes today's static
  eval + features, keeps quiet non-check positions, Generic eval kind only.
  10.3M records from 517k games in 87 s.
- Training: `nnue/train_eval_net.py` (WDL-space residual loss, K=531.9,
  λ=0.7 texel / 0.5 sprt, game-level holdout, quantization-aware from epoch 4),
  `nnue/export_eval_net.py` (AEVNET01 blob + integer/float agreement check).
- First net: 99→32→32→1, 4,289 params; held-out loss −4.7% vs zero residual
  (−7.9% on the fixed-depth corpus); int vs float 0.67 cp mean; ~770 ns/eval.
- SPRT: `games/sprt/games_evalnet_a1.json` (base_only preset, bounds [0, 5]).
