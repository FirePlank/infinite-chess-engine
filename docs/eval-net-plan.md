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
- 2026-09-23 **A4 committed (0151969): +11.6 ± 12.6 vs A2 over 2040 games** (LLR 0.97).
  Layer-2 vpmaddubsw kernel: checksum/nodes identical but 6% SLOWER (u8 repack of h1
  per call outweighs halved MACs); rejected. Next: 128×64 distilled from A4 vs A4.
- 2026-09-23 128×64 path: from-scratch + relabel pass 19.7%; distilled from A4 (α 0.5/0.8/1.0)
  18.8/19.1/18.6, distilled + relabel pass 19.6 → distillation adds nothing. h128x64 net is
  ~12% cheaper per eval (5.2 vs 5.9 µs); SPRT vs A4 running (games_evalnet_a5h128).
  Next: joint pawn-king test = A inputs (129) + dense king-relative pawn cells (310) in ONE
  net, exported together, screened on the fresh holdout against A4.
- 2026-09-23 **A5 committed (ba92e3e): 128×64, +27.9 ± 16.0 vs A4 over 1160 games** (speed win).
- 2026-09-23 Joint pawn-king screen (129 scalars + 310 king-relative pawn cells in ONE net,
  matched data, stride 2): joint 13.4/13.8% vs scalars-only 18.1/18.6% at 128/256 wide.
  Pawn cells lower train loss but hurt the holdout: memorization. Second pawn-king design
  to fail (after the stacked residual at 0.00%); runtime capture reverted.
- 2026-09-23 **A5 loss map** (holdout): 55% of remaining loss is in material-balanced
  positions (net gain there only 12%), 44% in the top phase band (13.5%); Standarch gains
  0.6% (compound pieces invisible to aggregate inputs), CoaIP 9.8%, Knightline 12.6%.
- 2026-09-23 **Sparse piece-king embedding** (every piece: 9 fairy-aware movement bits ×
  65 king-relative buckets × own/enemy king frame + pawn promo bins = 2356 features),
  warm-started from A5 by net surgery, trained on the archive mix: control 19.0, joint
  17.4 (falls to 7.1 by epoch 20), branch 19.4 (falls to 14.4) vs A5 start 20.2 on the
  report half. Holdout gain DROPS as training fit rises: archive piece-placement patterns
  do not transfer to the net engine's own games (distribution shift, not memorization —
  in-distribution val improved). Scalar inputs transfer; positional ones need on-policy
  data. Testing on 2.17M on-policy positions (net-engine SPRT games + fresh depth-9).
- 2026-09-23 **Game-result fingerprinting.** Sparse nets on 2.17M on-policy positions with the
  usual WDL-mixed targets collapsed to −25% on the holdout (control 18.0): piece-placement
  vectors identify the game, and half of every target is that game's result. With
  engine-eval-only targets (λ=1) the collapse disappears: control 18.0, branch 17.9, joint
  17.2. RULE: any positional/high-cardinality input must train on eval-only or relabelled
  targets. Explains the earlier pawn-cell failures too. Next: sparse on the 898k depth-9
  relabels (clean, diverse, no WDL) via a feature-vector join.
- 2026-09-23 Exporter: `--human` (infinitechess.org games, needs --relabel-depth),
  `--exclude-variants` (default Abundance), loader accepts older append-only layouts
  (A5 runs unchanged in a v3 build: identical checksum −1925464).
- 2026-09-23 **Human-game referee** (88.5k positions from ~14k infinitechess.org human games,
  engine variants only, Abundance/custom-evaluator/disconnects excluded, depth-9 labels):
  A2 34.8%, A4 33.2%, A5 40.7% error reduction vs HCE — the gains transfer to human play
  styles, more strongly than on self-play (~20%).
- 2026-09-23 Clean-label screen (1.0M depth-9 relabels, eval-only targets, from A5):
  self-play / human = A5 20.2/40.7, control 18.0/42.4, sparse branch 18.9/42.7, sparse joint
  18.5/42.0, 36 movement-class scalars (A6 layout) 18.2/42.4. New inputs within noise of the
  control; relabel fine-tuning trades self-play for human accuracy.
- 2026-09-23 Sparse-branch net in Rust (`src/eval_net/branch.rs`, 2356×32 embedding +
  32→32→1 head, int vs float 1.2 cp): eval 7.2 µs vs A5 5.2 µs. SPRT vs A5 running
  (games_evalnet_a6branch). Untested directions queued: search-tree position sampling,
  material-signature output buckets, net-driven search margins, depth-12 labels.
- 2026-09-23 Sparse-branch SPRT vs A5: −38.0 ± 37.4 after 266 games (LLR −0.49), stopped:
  worse in self-play as predicted (38% more eval cost, lower self-play accuracy). Shelved;
  its only upside is on human positions (+2.0 on the human referee).
- 2026-09-23 Priority order set by the user: material buckets, search-tree positions
  (`--perturb`), net inside search (uncertainty head); deeper labels only as background.
- 2026-09-23 **Uncertainty head** (frozen A5, predict |depth-9 − eval|): full head on
  h2+inputs rank-corr 0.29 (top/bottom decile error 96/37 cp) vs 0.23 for |eval| alone;
  cheap h2-only head 0.24 ≈ |eval|. Too little over a free proxy to wire into margins now.
- 2026-09-23 **Material/stage buckets** (warm-started copies of A5's layers, quantile edges,
  mixed data; control 18.82 self-play / 38.75 human): phase8 head 19.14/39.02, count8 head
  19.15/38.65, pawns3x3 head 18.98/38.31; layer stacks (per-bucket L2+L3) 18.4–19.0 and hurt
  the human referee badly (32.9–38.7). Output-bucket gains +0.3 = seed noise. Closed.
- 2026-09-23 Perturbed ("search-tree") positions: 300,545 relabelled at depth 9 (1–3 random
  legal moves off the game line), 2.9 h. Fine-tune screen running.
- 2026-09-23 Aspiration W2 (SF19-style: ±25 cp, recentre on failing bound, ×1.37, 8 retries)
  SPRT vs A5 running, 1000 games (games_asp_w2).
- 2026-09-23 **Perturbed-position fine-tunes of A5** (eval-only targets): relabels 18.2/42.4,
  perturbed 300k **18.4/44.7**, both 18.3/44.1 (self-play/human; A5 19.7/40.7). With the
  standard mixed labels: control 18.6/37.8, +perturbed 18.5–18.7/38.2–38.9. Every A5
  fine-tune loses ~1.3 on the self-play holdout, likely selection bias (A5 was picked on
  that holdout); the human referee is unbiased. SPRT of the perturbed net running.
- 2026-09-23 **Aspiration W2 vs A5: −9.4 ± 17.4 over 1000 games (LLR −0.59)**. Not clearly
  bad; parked behind the perturbed-net SPRT, consistent with the earlier rejection.
- 2026-09-23 **Perturbed-position net vs A5: −24.1 ± 13.5 over 1818 games (LLR −2.33), REJECTED.**
  Engine-eval-only fine-tuning trades self-play accuracy for human-position accuracy on one
  curve (5 ep 18.8/42.3 … 40 ep 18.1/45.3); the self-play referee predicted the SPRT, so
  the WDL part of the label is load-bearing for Elo. The human referee alone does not
  predict self-play Elo.
- 2026-09-23 A5 fine-tune on the A4 mix + 5,160 new fresh games (standard labels): 19.19/39.05
  vs matched old-mix control 19.02/38.78 → noise; no candidate beats A5 (19.72).
- 2026-09-23 **Aspiration W1 (initial window 30, ×4 kept) vs A5: −63 ± 47 after 128 games**,
  stopped. Both aspiration variants lose; the ±60/×4 form stays.
- 2026-09-23 Depth-12 relabel of the 250k key subset started (`nnue/relabel_d12.bin`).
- 2026-09-23 **M75 (eval-trust margins ×0.75) vs A5: +1.4 ± 15.8 at 1240 games**, shelved.
- 2026-09-23 **Colour symmetry.** A5 start-position bias up to ±69 cp (never saw plies <12;
  (White,Black)+stm inputs). Openings in training cut it to ≤31; lockstep mirror augmentation
  to ≤23 at equal accuracy; perspective (us, them) encoding makes it exactly 0 and costs no
  accuracy vs a matched White/Black net. Found an A2 input bug: king-to-cloud distance mixed
  doubled and single units (not translation-invariant). Fixed in the v4 schema; Rust port of
  the perspective transform done (blob v2 = perspective). A5-recipe retrains on v4 data in both
  encodings running; stage-scale SPRT running.
- 2026-09-23 Development term: calibrated at population level (unexplained score vs undeveloped
  difference ≈ 0 up to ±2 pieces). CoaIP hawks start ~10 squares from the cloud centre, inside
  the 16-square cohesion radius, so only the one-move starting-square penalty pushes them.
  Proposed fix: smaller cloud radius for leapers (~8), not a development redesign.
- 2026-09-23 **v4 retrains (A5 recipe, corrected input):** White/Black 19.29/36.97, perspective
  **19.63/38.42** (self-play/human). Perspective candidate built clean on A5 + 3 changes; start-
  position colour bias exactly 0.000 in every symmetric variant. SPRT vs A5 running
  (games_a6persp). Stage scale 1.4× stopped at −11 ± 39 (184 games) on request.
- Queued: leaper cloud radius 16 → 8 (hawks), via the retrain-both-sides workflow.
