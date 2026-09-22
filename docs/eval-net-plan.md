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
- 2026-09-22 A2 (60 ep) vs A1: **+43.3 ± 15.0** over 1452 games, LLR 2.95, committed 5fea1cf.
  Pawndard −21/−67 and CoaIP_NO −58/−52 in both A2 runs (gating watch).
- 2026-09-22 λ 1.0/0.7 (teacher-heavier targets) vs A2: **−29.9 ± 13.1**, LLR −2.96, REJECTED.
  Offline "gain" is not comparable across target changes; game results carry real
  signal, so the next target test goes the other way (λ 0.5/0.3), then K 400.
- 2026-09-22 λ 0.5/0.3 vs A2: +1.7 ± 11.5 over 2500 games, LLR −0.06, neutral → λ axis
  closed at 0.7/0.5. Pawndard negative for the third net-vs-net run (−51): running the
  A2-vs-A1 Pawndard-only gating check before K 400.
- 2026-09-22 Pawndard-only check, A2 vs A1: **+27.5 ± 17.4** over 1000 games. The three
  negative Pawndard lines were per-variant noise; no gating. Next: K 400 vs A2.
- 2026-09-22 K 400 vs A2: −1.5 ± 11.6 at 2296 games (LLR −0.41), neutral, stopped. Target
  axis closed (λ 0.7/0.5, K 532). Chess/Obstocean/PawnHorde nets shelved by decision: the
  base evaluator is the priority. Next: fixed-depth data_gen with the A2 engine
  (`games/texel_corpus_a2.jsonl`, depth 9), retrain on all sources, SPRT; then Stage B.
- 2026-09-22 Stage B screen (dense king-relative pawn histograms, 310 inputs, 64x32 net
  trained on A2's remaining residual, 5.1M records): **0.00% held-out gain**, train loss falls
  while validation does not. The residual A2 leaves is label noise at this data quality, so
  inputs cannot help; only better labels can. Stage-B code dropped (it would cost eval time).
  Lever now = label quality: fixed-depth data_gen with the A2 engine (running), then retrain.
- 2026-09-22 **Fresh-data referee.** 86k positions from the first 851 depth-9 games of the
  A2 engine (never trained on) rank the nets in the SPRT order: A2 (r_ep60) 17.5%,
  h128/32 15.9%, λ0.5/0.3 15.8% (SPRT neutral), texel×3 15.3%, K400 14.1% (neutral),
  λ1.0/0.7 13.7% (−30 Elo), texel-only nets 3–4% (overfit the small off-policy corpus).
  Rule from here: screen on fresh on-policy fixed-depth data, never on a split of the
  training corpus. `train_eval_net.py --eval-only` does this.
- 2026-09-22 Data composition on the fresh referee: archives-only 16.4%, archives at 2×
  sample 16.6%, low-LR fine-tune of A2 17.4%, vs A2 17.5%/16.7% (two seeds). Nothing in
  the existing sources moves the fresh metric; the old texel corpus neither helps nor hurts.
  Only fresh on-policy fixed-depth data remains as a lever; data_gen (depth 9) runs at
  ~400 games/h.
- 2026-09-22 A3 (A2 recipe + first 2000 fresh depth-9 games ≈ 200k records) on a 113k
  fresh holdout: ×1 18.3%, ×3 17.1%, ×8 15.2% vs A2 seeds 19.4%/17.9%. Fresh data is 2% of
  the corpus and cannot move the net yet; up-weighting a small set overfits. Continue
  data_gen; retrain at ≥1M fresh records. Seed spread (1.5 pt) is exploitable: keep the
  best of N seeds on the holdout.
- 2026-09-22 A3 seed sweep (seeds 3-6) on the holdout: 18.4/19.1/19.1/18.4; committed A2
  = 19.4. No candidate clears the incumbent; waiting on more fresh data (4.3k games).
- 2026-09-22 A3 with 540k fresh records (5% of corpus), seeds 1/4/5: 18.9/18.6/18.4,
  ×2 fresh weight 18.3 — still inside A2's band (19.4/17.9). Testing whether the archive
  noise now caps the net: fresh-only and fixed-depth-only trainings on the same holdout.
- 2026-09-22 Without archives the net collapses on the fresh holdout: fresh-only 9.3/10.6%,
  fixed-depth-only (1.6M clean records) 10.5/9.9%, A2 warm-started then tuned on fixed-depth
  18.4% (= A2). Volume on the archive distribution is what the net learns from; clean
  labels help only at that volume. Plan: re-label sampled archive positions with a depth-9
  search (~200k positions/h vs ~40k/h from new games) and train on the re-labelled set.
- 2026-09-22 `export_eval_features --relabel-depth 9`: re-labels each kept archive position
  with a fixed-depth search of the current engine (70 positions/s on 16 threads). Full run
  on ~1M sampled archive positions started (`nnue/relabel_d9.bin`); data_gen paused at
  7370 games (resumable: same command appends).
- 2026-09-23 Relabel results on the holdout: relabelled-only (898k) 16.9/15.6%, all
  fixed-depth 14.8%, mixed+relabel fresh seeds 18.2/18.1%, **A2 warm-started and fine-tuned
  20 epochs at lr 2e-4 on mixed+relabel: 20.05%** vs A2 19.4% from the same weights.
  First candidate above the incumbent → SPRT (A4 = fine-tuned A2).
- 2026-09-23 Fine-tunes from A2 on the holdout: mixed 40ep lr1e-4 19.8, mixed 20ep lr5e-4
  20.1, relabel-only 20ep 20.3, fixed-depth-all 19.1, mixed ×2 19.2, mixed seed2 20.2.
  Warm-start seed noise ≈0.15, so the +0.7–0.9 over A2 (19.4) is real but small; A4 (mixed
  fine-tune, 20.05) is in SPRT, relabel-only fine-tune is the backup candidate.
- 2026-09-23 128×64 on mixed+relabel, seeds 1/2: 18.4/19.1 on the holdout (A2 19.4) at
  ~half the net cost; seed 2 (+ relabel fine-tune) is the size-vs-speed SPRT candidate.
- 2026-09-23 128×64 seed 2 + relabel fine-tune: **19.7%** on the holdout (A2 19.4) at ~half
  the net cost → SPRT candidate right after A4 (`h128_64_ft.pt`).
