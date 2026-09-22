# Stage-A eval net: exhaustion plan

Goal: squeeze the proven HCE-residual net (docs/hybrid-eval-design.md §4.1, §11)
before moving to the pawn-king net. Ordered by expected gain per hour of
machine time. Every training experiment is first screened OFFLINE (held-out
WDL loss vs the zero-residual baseline, ~2 min per run); only candidates that
beat the incumbent offline go to SPRT. Runtime-only changes never need an
SPRT: identical `eval_bench` checksum (the eval oracle) plus interleaved
old/new ns-per-eval pairs decide them.

Standing rules for this loop:
- SPRT: `--concurrency 12`, base_only preset (Generic evaluator only),
  gainer bounds `--elo0 0 --elo1 5`, 1000 games then `--resume` to 2500.
  NEW = the freshly trained net, OLD = the last committed net (not HEAD-before-net),
  so each step is measured against the incumbent.
- Keep a test running at all times; while one runs, do only non-CPU work
  (training data screening on the GPU is fine, builds are not).
- Commit each accepted step with the Final Summary block; retrain data stays
  out of git (`nnue/eval_net_data.bin`, `nnue/checkpoints/` are ignored).

## Tier 1: model capacity and inputs (net is under-fitting: train loss == val loss)

1. **More inputs (A2).** Add ~30 scalars the eval already has but the vector
   omits: pawn-structure counts per side (doubled, isolated, connected,
   candidate, passed, backward) routed through `EvalNetInputs` instead of the
   cache-bypassing tracer rows; king-to-king Chebyshev distance bin; passer
   promo-distance min and count; total mobility per side (`Piece: Activity`
   components); `Pawn: Doubled/Candidate/Connected/Isolated/Backward` tapered
   rows; enemy-slider count on each ray class; halfmove clock bucket.
   Bump `SCHEMA_VERSION`. Screen offline against A1's 4.68%.
2. **Wider net.** 64 and 128 hidden in layer 1 (layer 2 stays 32). Cost
   ~+400 ns per doubling before vectorization; screen offline, SPRT the best
   size only after Tier 3 item 1 has cut the per-neuron cost.
3. **Cap 250 → 500.** ~4.5% of outputs clip today. Screen by applying the
   clamp inside the offline loss; if it helps offline, fold it into the next
   SPRT'd net rather than testing alone.
4. **Recipe knobs (offline only, take the best):** λ_texel/λ_sprt grid
   (0.5–1.0 / 0.3–0.7), K (400/532/700), 60–100 epochs, weight decay, lr,
   oversampling the fixed-depth corpus 3×, dropping archive positions with
   |teacher − static| > 250, WDL-only and teacher-only ablations.
5. **Phase-split output.** Two outputs (mg, eg) tapered by `effective_phase`
   like the HCE terms; cheap, common NNUE trick. Offline screen.

## Tier 2: data

6. **On-policy archive growth.** Every SPRT run of this loop lands in
   `games/sprt/games_evalnet_*.json`; re-export includes them automatically,
   so each generation trains on the stronger engine's positions.
7. **Fresh fixed-depth corpus with the net engine.** Overnight
   `data_gen --variants base_only --depth 9` (the fixed-depth corpus gave
   7.9% offline vs 4.5% on the 10+0.1 archives: label quality > volume).
   Requires a `data_gen` build of the incumbent; run while no SPRT is active,
   or at reduced threads alongside one only if timeouts stay at baseline.
8. **Filter tuning on export:** `--min-ply`, `--quiet-tolerance`,
   `--max-abs-cp`, sample rate; screen offline.

## Tier 3: runtime cost (no SPRT; checksum-identical + interleaved NPS pairs)

9. **Vectorize layer 1** (i8 × i16 → i32 with 16-lane chunks; native AVX2 via
   autovectorization or `std::arch`, wasm via simd128 which the build already
   enables). Layer 1 is 3,168 of the 4,257 MACs.
10. **Tracer row lookup:** resolve the 15 row names to indices without runtime
    string compares (const table keyed by pointer/len or an enum passed by the
    eval); measure, it may already be folded.
11. **Feature vector build:** write the i16 vector straight from the collector
    instead of clamping twice; skip `feature_vector` allocation churn.
12. **Skip the net where the HCE result is decisive** (|score| > 1500 cp or
    insufficient-material paths): saves the forward pass in mop-up trees;
    behaviour-changing, so this one DOES need an SPRT (non-regression bounds).

## Tier 4: coverage

13. **Specialized evaluators.** Chess / Obstocean / PawnHorde wrap base with
    different term mixes; train one net per evaluator on their own archive
    positions (exporter gains an `--eval-kind` switch) and hook into
    `variants/*.rs`. SPRT on the matching variants only (`--variants site` for
    the combined run). Possible gating per evaluator.
14. **Per-variant gating check** after each SPRT: a class that is negative
    across two fresh runs gets the residual disabled by `eval_kind` pattern,
    not a revert.

## Exit criterion → Stage B (pawn-king net)

Two consecutive Tier 1/2 iterations that fail to beat the incumbent offline by
> 0.3% relative, or that pass offline and fail SPRT, mean A is saturated. Then
build the pawn-king net on the residual that A leaves (design doc §4.2).

## Log

- 2026-09-22 A2 screen (129 features, offline held-out gain; A1 shape 4.6%): h32 4.79,
  h64 5.20, h128/32 5.42, h256/32 5.64, h256/64 5.79, h512/32 5.86, h256/128 5.91.
  Cap 250→500 +0.1 (clipping 7%→0.4%), cap 1000 no further gain. 60 epochs +0.18;
  100 epochs, lr, batch, weight decay, phase-split head: within seed noise (~0.06).
  Texel oversampling ×3 and |teacher−static|≤250 filtering both LOSE. λ and K
  change the target, so they are SPRT-only questions.
- 2026-09-22 A2 (h256/64, 30 ep) vs A1: +23 ± 18 over 1000 games (LLR 1.03).
  Net forward 1.9 µs of a 6.2 µs eval; MAC count is not the limit (i8→i16 widening
  and 4-row kernels changed <5%), pointing at misaligned 258-byte row strides.
