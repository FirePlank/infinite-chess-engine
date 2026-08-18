# Changelog

All notable changes to Apeiron (formerly known as HydroChess) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/), matching the `version` field in `Cargo.toml`.

### Versioning policy

Releases up to and including `v1.3.0` were numbered manually, matching the historical "Apeiron N" milestones announced on the site and elsewhere. `v2.0.0` is a deliberate baseline reset that coincides with the HydroChess → Apeiron rename. From `v2.0.0` onward, version bumps are decided automatically by [`scripts/elo_release.py`](scripts/elo_release.py) based on accumulated SPRT-measured Elo since the last release:

- **+30 accumulated Elo** since the last release → minor bump
- **major bumps are manual**, triggered by running the Auto Release workflow by hand

The accumulator sums each commit's own SPRT-reported Elo, scaled across the 17 site variants. Those figures are *nominal*: per-commit SPRT results are measured against different baselines and don't add up to an A/B measurement, so they consistently overstate the real gain. The bold Elo line under each release is instead the accumulator rescaled against a directly measured head-to-head match between the two releases, so consecutive entries add up to what an actual game would show.

## v3.0.0 (2026-08-18)
Commit: `dc89a0e41da6616dd8b678c93c9166796c99fb46` • [compare to v2.6.0](https://github.com/FirePlank/infinite-chess-engine/compare/a92199f9ffc95261b617cf0c93b18937e6390493...dc89a0e41da6616dd8b678c93c9166796c99fb46)

**It is about 8 Elo better than v2.6.0, and about 110 Elo better than the v2.0.0 baseline.**

### Added
- Live SPRT dashboard: per-variant rows, LOS, abort-on-fault, and an honest ETA, with full/compact/plain views chosen by terminal detection so the live and final printouts can't drift apart
- Max-ply adjudication option in the SPRT CLI and web UI
- `AGENTS.md`/`CLAUDE.md` contributor documentation and an SPRT-testing skill
- A test asserting a colour-mirrored position evaluates to exactly zero

### Changed
- Weak site skill levels misjudge the position instead of just picking a worse move: attack-recognizing terms are damped, defensive ones amplified. Full strength stays bit-identical
- The first move of a PV node gets the full search window instead of a null-window scout, so a fail-low there is no longer returned as a real score that seeds aspiration windows and the TT
- SPRT reuses a persistent engine process per game instead of respawning per move, cutting ~50ms of spawn overhead and a cold TT/history from every move; at concurrency 16 that overhead alone had turned a 300ms search into 857ms wall-clock
- SPRT plain-mode output reports every game instead of every 50, so a backgrounded run is watchable through its log
- `[Variant "Omega"]` resolves to no-variant instead of silently falling back to Classical
- Smarter puzzle generator

### Fixed
- Empty tiles are reclaimed, so the fixed-capacity tile table can't fill and spin forever in `get_or_create`'s unbounded probe. That probe sits below the search's stop-flag poll, which is why game review could hang uninterruptibly
- The world border resets on every ICN parse, so a missing border token means "no border" instead of inheriting the previous position's
- The bishop long-diagonal bonus is anchored to the kings' midpoint instead of absolute 8x8 lines; all 15 mirror-symmetric variants were biased before and now evaluate to exactly 0
- The piece-cloud centre is kept at half-square precision. Truncating it had biased every cloud distance by side, and Classical read +4 for White in a mirrored start
- En passant is filed as a capture in staged generation, so evasion scoring no longer ranks it below every capture and ProbCut no longer rejects it as a TT move
- En-passant captures are scored against their real victim beside `m.to` instead of falling through to the no-victim branch
- SPRT creates parent directories for output files

### Removed
- The continuation-history and capture-ordering additions. Each had passed its own SPRT at LLR 0.24-0.62, far under the accept bar, so the set was selected on noise: their measured gains summed to +56 nElo while removing all eight commits cost nothing

## v2.6.0 (2026-08-14)
Commit: `a92199f9ffc95261b617cf0c93b18937e6390493` • [compare to v2.5.0](https://github.com/FirePlank/infinite-chess-engine/compare/4b3ad18c949322259e24e945cd58db51d39da9bf...a92199f9ffc95261b617cf0c93b18937e6390493)

**It is about 14 Elo better than v2.5.0.**

### Changed
- Internal iterative reduction starts at depth 3 instead of 6, cutting ~30% of nodes at fixed depth and buying real depth back; the dense variants that lost at depth 4 recover at 3
- Root moves are generated without the stale slider candidate cache. Over 2401 positions, 84% of root lists both lost legal moves and gained impossible ones, and `is_move_illegal` only checks king safety, so a stale slide through a blocker would have been played
- Riders are priced against leapers by bounded-world geometry, since a bounded world truncates rays; inert on unbounded boards
- SPSA discards whitewash iterations instead of tuning on them, and surfaces the swallowed panic and rejected parameters

### Fixed
- Exact generation restricted to evasions whenever the side was in check, but under `AllRoyalsCaptured` a check need not be answered: 5-7 moves were offered where 31-42 are legal
- `minor_hash` did not follow the castling partner, so every CoaIP Guard castle corrupted the minor correction-history bucket for that subtree
- The en-passant victim was always booked as a pawn, even when a double push had promoted and left a promoted piece on the landing square, injecting a phantom pawn into `pawn_hash`
- Zero royals ended the game regardless of win condition, but `AllPiecesCaptured` requires taking every piece, so a bare army fights on
- The insufficient-material cache was keyed on material hash alone, so it survived a variant switch and could return a verdict for the wrong boundedness
- Camel/Giraffe/Zebra/Hawk and king-step adjacency were absent from the fast check test, so real checks scored as quiets and could be futility/history-pruned, over-reduced, or dropped from qsearch
- The LMR depth clamp panicked at depth 1 when `lmr_min_depth` was tuned to 1; the SPSA harness turned the panic into "bestmove none", silently forfeiting all 200 games of an iteration

### Improved
- King tropism weights come from const tables instead of three runtime integer divisions per piece per royal, with piece-type classification hoisted out of the royal loops: 3470.6 → 3412.1 ns/eval

## v2.5.0 (2026-08-11)
Commit: `4b3ad18c949322259e24e945cd58db51d39da9bf` • [compare to v2.4.0](https://github.com/FirePlank/infinite-chess-engine/compare/22510bd1228528d800a0816ae1794d66202d5a23...4b3ad18c949322259e24e945cd58db51d39da9bf)

**It is about 21 Elo better than v2.4.0**, mostly from a native Texel retune of the evaluation.

### Added
- Native Texel tuner: fits base.rs's additive eval weights to a 1.1M-position self-play corpus (10.3k games over 14 base-eval variants) by extracting per-parameter derivatives once, then running Adam with held-out early stopping. 96 terms retuned; the two terms the fit drove to zero on both taper ends were removed outright
- King-safety and pawn-structure tapers exposed to tuning. Eight MG/EG pairs were hardcoded, and their endgame halves had been calibrated back when big-army variants couldn't reach the endgame at all
- `calc_load` and `generated_date` in generated puzzles

### Changed
- The taper clock runs on each game's own starting material instead of Classical's `MAX_PHASE=24`, which had pinned big-army variants at full middlegame so the endgame half of every tapered term never executed there
- The score is damped by material complexity beyond classical phase: a cp edge cashes far less with a big army aboard (81k-game analysis: p>=0.8 evals win 61% in CoaIP vs 79% in Classical). Classical-sized positions are byte-identical, and damping also makes trading while ahead raise the scaled score, the missing conversion gradient
- The undeveloped-minor penalty is split by piece type
- Distant squares are reached by target value rather than distance, so heavy pieces and undefended targets bypass the 16-square cross-ray filter; a rook previously couldn't see a winning 23-square attack on a lone passer
- Slider squares that knight-attack from compound pieces are proposed, so an archbishop or chancellor can generate a fork or check along an otherwise-empty line; knight-attack candidates extended to the amazon
- Puzzles are rated by how hard each move is to find rather than by ply count (max 2660 → 2310, median ~1350 → 1090), with long quiet combinations ranking highest
- `--recook` and `--deep-verify` checkpoint per candidate and bound each search, so a killed run resumes instead of restarting and one unsearchable position can't stall its batch
- Contempt is dropped in analysis so review scores stay objective; play paths keep 15cp

### Fixed
- Lazy SMP helper threads kept serving pawn/material cache values from the previous variant's rules after a variant switch, since the cache key has no rules component
- Shared pawn history was not reset on `reset_engine_state`

### Improved
- Short spatial lines are scanned instead of binary-searched: 90% hold <= 4 pieces, none over 16, where `partition_point`'s branchy search loses to a predictable scan. NPS 201k → 207k
- King rays are derived from the spatial index in four lookups per king, instead of testing every piece on the board for alignment
- Both ends of a slider line share one binary search, so four lookups cover all eight rays: 1.74 → 1.63 µs
- The `sprt` feature no longer pulls `param_tuning`, which turned every eval-parameter read into a runtime load; matches had been measuring a slower engine than actually ships

## v2.4.0 (2026-08-07)
Commit: `22510bd1228528d800a0816ae1794d66202d5a23` • [compare to v2.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/6f73f04989220e714b41eddbb6c858dfd7090600...22510bd1228528d800a0816ae1794d66202d5a23)

**It is about 14 Elo better than v2.3.0.**

### Added
- 15cp contempt to avoid draws. Avoidable draws (threefold + 50-move) drop from 13.25% to 10.78% of games at no Elo cost in self-play, where both sides decline them
- Obstocean pawn-obstacle breakouts treated as tactical. Taking a neutral obstacle opens the line the variant is built around, but the neutral victim had made it score as quiet; SEE and delta pruning still gate it
- A test asserting `min <= default <= max` for every tunable spec

### Changed
- Late moves reduced harder (`lmr_divisor` 3 → 2); root widths of 100-400 make near-full-depth late moves unaffordable
- Late move pruning tightened on small bounded boards, where Chess branches ~29 against the ~101-wide open plane LMP was tuned for. Obstocean is excluded, since its breakout is a quiet LMP would discard
- Low-history far quiet sliders pruned on bounded boards; they're 55% of generated and 30% of played moves. Confined_Classical is excluded as obstacle-ringed but unbounded
- Slider threats scored by victim-minus-attacker value gap rather than bucketed, so they can't go stale when piece values are refitted
- Continuation history keyed on 4-bit coordinate hashes: 25MB → 3.1MB per searcher, which is per-thread and matters for wasm. Elo-neutral; the cache-miss motivation wasn't borne out
- Pawn history shrunk to 2048 buckets (128MiB → 32MiB). More aliasing is the right direction here, since widening main-history buckets 256 → 4096 previously lost 9.3 Elo
- The countermove table widened from i16 to i32 destination coordinates, since a piece past ±32767 could alias to the wrong entry
- The multipv scout is skipped while candidate slots are unfilled, where the threshold is -infinity and the full re-search was guaranteed anyway
- UCI: `movetime` spends its full budget (2000ms → ~1950ms, previously ~870ms), `movestogo` is honoured instead of assuming ~50.5 moves left, and `Hash` has a real `setoption` handler
- Skill levels refined over several passes, with better win conversion at lower levels
- The setup-time attack pass keeps only the pin scan; the check-square, slider-ray, discovered-check and checker-count tables lost their last readers once check detection went live

### Fixed
- Non-pawn-material flags latched true at setup and never cleared on capture, so the NMP zugzwang guard and shallow-pruning gate saw "has pieces" for whole games. They're now derived from the live piece counters
- i64 → i32 overflow in distance-based tropism: past ~2^31 the cast wraps, driving queen line-tropism and king-pawn tropism into multi-million-cp scores that cross `MATE_SCORE` and corrupt search and TT
- Qsearch stood pat on its depth cap while in check, claiming a possibly-mated position was fine, even though stand-pat is suppressed precisely because we're in check; it now fails low
- The Rose SPSA range excluded its own default (997 against a 250..650 range), so the first perturbation would have clamped both candidates below it and silently discarded two prior positive changes
- Castling never required the partner to be >= 3 squares from the king, and never checked the king's own landing square when a partner sat adjacent, so the king could land on an occupied square
- The TT prefetch key toggled side and moved the piece but never removed the capture victim, so every capture prefetched the wrong bucket
- The eval debug trace is now an accounting identity. Material seeded the score but was never recorded, and pawn penalties were traced through `.abs()` so a penalty showed as a positive contribution

### Improved
- `is_piece_attacking_square()` early-exits for pure sliders and slider+knight compounds instead of generating moves (#32)
- Displacement checks for the remaining fairy attackers (Camel, Giraffe, Zebra, Hawk, Centaur, RoyalCentaur, RoyalQueen), which had fallen through to full move generation
- `get_piece().is_none()` replaced with the faster `is_occupied()` check

## v2.3.0 (2026-07-27)
Commit: `6f73f04989220e714b41eddbb6c858dfd7090600` • [compare to v2.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/2d5e7fd99aaddfe84f4eeb7dee83a7f144487751...6f73f04989220e714b41eddbb6c858dfd7090600)

**It is about 16 Elo better than v2.2.0**, mostly from an empirical revaluation of the fairy pieces.

### Changed
- Fairy piece values refitted by logistic regression of game outcome on material imbalance over 294k positions from 41k games. The method self-validated (the queen, rook and knightrider fits matched the engine's existing values) and corrected hawk 632 → 450, guard 224 → 180, huygen 363 → 330, archbishop 908 → 1060, chancellor bonus 116 → 245. Bishop was left alone, since it was the term that had dragged orthodox variants down in an earlier joint test
- The rose raised 700 → 775 → 880 → 997 in three steps, every one of which gained
- The amazon given a compound premium. It was the only compound priced at the bare sum of its parts, while the chancellor carries +245 over rook+knight and the archbishop +371 over bishop+knight
- Hawk and huygen devaluations half-stepped. The pooled fit was dominated by open variants, and the full devaluation cost ~40 Elo across the CoaIP family where these pieces are worth more
- The skill limiter now applies to every level below the maximum. `MAX_SITE_SKILL` is 8 but both dispatch sites gated on `s < 3`, so levels 3-7 ran the full-strength search; depth caps are now an explicit 2/3/4/6/8/10/12 ladder
- Far quiets along a ray are skipped at depth <= 3
- Better auto-release workflow

### Fixed
- Knightriders generated at most two hops on an open ray, so every maneuver longer than two hops was invisible to the search at every depth; open rays now reach 5 hops, still gated to 2 at tight-generation depths
- Exact legal-move lists bypass the stale slider cache, which is keyed only on `(square, direction)` and never invalidated: startpos perft D3 gave 8842 instead of 8902
- Knight and pawn checks were tested against a check-square set built once at setup, so they were wrong at every node where a royal had moved; testing the royals directly is exact and cheaper than the hash probe it replaces
- Both slider branches returned true on alignment alone, so blocked rays counted as checks, were exempted from pruning, and paid for a full SEE in the ordering
- `is_shuffling` probed the board after `make_move`, where the destination always holds the mover, so the capture probe was always true and the detector always false

## v2.2.0 (2026-07-23)
Commit: `2d5e7fd99aaddfe84f4eeb7dee83a7f144487751` • [compare to v2.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/c758bdd4d08bb9b402136abdfe1267b550fb690a...2d5e7fd99aaddfe84f4eeb7dee83a7f144487751)

**It is about 19 Elo better than v2.1.0**, largely from a mop-up and king-danger overhaul.

### Added
- Quiet moves that lift a valuable (>= knight) piece off an enemy-attacked square onto a safe one are ordered by the saved piece's value at depth >= 4, surfacing the escape before LMR/LMP can bury it

### Changed
- King danger grows at quarter slope past 400, capped at 800, instead of clamping flat. The hard `.min(400)` had zeroed the gradient exactly where attacks escalate, so past the cap no amount of extra pressure could veto a material grab
- Mop-up edge-push, king-approach and KBN corner-drive each get a gentle linear tail past their caps. Mid-board on larger bounded boards (Obstocean 21x15) had zero gradient there, pure shuffle
- Mop-up station anchoring smoothed to the max over the 3x3 anchor neighbourhood, which one king step can never drop. Stations had been anchored to a single grid cell, so the defender crossing a boundary teleported all of them
- Mop-up conversion cliffs removed. The kill-zone bonus adds on top of the target-box score instead of replacing it (holding the assembled box forever had been optimal), and the shaping downside now saturates at 250 so the simplifying capture that activates mop-up can't read as a >1000cp drop
- Passed-pawn scoring computed live. The pawn cache is keyed on pawn positions only but had baked in phase taper, king distances and blockers; it now stores untapered `(mg, eg)` shape terms plus passed-pawn coordinates
- King-ray shelter, attack bonus and attack readiness share `ray_pressured()`. An enemy slider parked on a king ray had counted as a shield worth 40% penalty relief, so achieving alignment lost the attack credit it should gain
- Time checks every 4096 nodes instead of 8192. In slider-heavy low-NPS positions one 8192-node interval could exceed the whole near-flag budget, firing the first check after the flag
- The low-clock survival budget reserves move overhead with an increment/4 floor. It had spent 0.9x increment with no reserve, while every move is charged spawn/reply latency, draining the clock monotonically toward a flag

### Fixed
- Legality is verified against knightrider and huygen pins. Knightriders pin along knight-rays and huygens along prime-distance files, neither of which lies on a queen ray, so the pin map and the `is_legal_fast` queen-ray test cleared moves that leave the king in check
- Bishop square-colour parity mixed into the material hash. The insufficient-material cache is keyed by it, but the verdict depends on the light/dark split (an opposite-colour pair wins where a same-colour pair draws), so a winning set could be cached as a draw
- `undo_move` never restored the non-pawn-material flags that promotion sets, so one promotion subtree corrupted NMP gating and mop-up classification for the rest of the search
- The precise-mode castling hash pair was computed after the mover was removed, and it skips rights-squares with no piece, so the mover's own rights key was never XORed out, desyncing hash and rep_hash for the whole subtree. Affects multi-partner positions such as double-king

## v2.1.0 (2026-07-21)
Commit: `c758bdd4d08bb9b402136abdfe1267b550fb690a` • [compare to v2.0.0](https://github.com/FirePlank/infinite-chess-engine/compare/6a407eeb02ac3e7b25659ad146c08b123517963e...c758bdd4d08bb9b402136abdfe1267b550fb690a)

**It is about 18 Elo better than v2.0.0.**

### Added
- Pentanomial SPRT
- Threat-aware quiet move ordering at depth >= 4: quiet moves attacking an enemy piece are boosted by victim value, doubled when the victim is undefended. Unlike dest-hashed history this signal doesn't alias at deep nodes, so LMR/LMP reduce the right late moves; shallow nodes skip the cost
- SPRT adjudicates a max-ply game as decisive when both engines' last search score agrees one side is ahead by >= 10 pawns
- This changelog (#29)

### Changed
- Quiet moves pruned one move earlier (`lmp_base` 3 → 2); the net gain is driven by high-branching open and pawn-heavy variants
- The mating net applies whenever the defender is pawnless, so piece-up endings like K+R+P vs K+B no longer stall for the whole move budget
- The slider interception cache is stored as `Arc<[i64]>`, making a hit a refcount bump rather than two `Vec` copies, dropping the miss-path clone, and letting per-thread game clones share the arc

### Fixed
- Long infinite-chess games (200-300 plies, past the allocator's ~50-move horizon) survive on the increment instead of flagging on time
- `negamax_root` omitted the per-node reset a ply-0 node would do, so `cutoff_cnt[2]` never cleared and ply-1 late moves drifted toward permanent over-reduction
- Undefined behaviour in local TT move decode: a key16 collision could XOR foreign bits into an out-of-range `PieceType`/`PlayerColor` and transmute them; those are now rejected and the move dropped

## v2.0.0 (2026-07-15)
Commit: `6a407eeb02ac3e7b25659ad146c08b123517963e` • [compare to v1.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/be4c3931e05506fe46018e2d15a8710baaf13f02...6a407eeb02ac3e7b25659ad146c08b123517963e)

**It is about 150 Elo better than v1.3.0, with an additional 50 Elo improvement from making multithreading the default.**

### Renamed: HydroChess → Apeiron
This release renames the project from **HydroChess** to **Apeiron**, a Greek word that means _“the unlimited”_ or _“the boundless.”_ [(Reference: Wikipedia)](https://en.wikipedia.org/wiki/Apeiron "Open the Apeiron page on Wikipedia") The new name is reflected across the codebase, build artifacts, and documentation.

### Added
- UCI protocol support, for interoperability with standard chess GUIs
- Real-time analysis protocol
- Game review web feature
- Multi-king variants and an “All Pieces Classical” variant in SPRT
- SPRT presets: `all`, `base_only`, `base_full`, `site`, `multi_king`, and `coaip`
- Secondary Zobrist hash for more reliable repetition detection
- Insufficient-material detection for additional material configurations
- Engine version now exposed through a WASM-callable function
- Strength-based auto-release pipeline with a `v2.0.0` baseline (`scripts/elo_release.py` + GitHub workflow)
- Multithreaded Lazy SMP is now the default build, currently supporting up to **4 threads**
- Thread-aggregated NPS reporting for multithreaded search

### Changed
- The evaluation function is now selected based on positional characteristics instead of variant metadata
- Multi-royal positions now correctly go through the full legality verifier instead of a single-royal fast path, including tropism and check/pin handling
- Reworked mop-up logic, then extended it to apply in “chess” and other bounded variants
- Reworked Pawn Horde evaluation to be stronger than the base evaluator, with adjusted pawn bonuses and faster pawn-structure evaluation
- The variant-specific evaluation is now a lot stronger in Obstocean and Pawn Horde
- Adopted the Ethereal push-square model for evaluation
- Better pawn shelter evaluation
- Dynamically adjusted attacking/defensive tropism
- Improved leaper, knightrider, and knight-mobility evaluation
- Improved principal-variation and TT best-move handling
- Better NNUE handling
- Better king-pawn proximity evaluation
- Account for minor “fairy” pieces and neutral pieces in evaluation and move generation
- SEE (Static Exchange Evaluation) refactored to use `see_ge` for pruning and to account for pinned pieces
- `compute_pins` is now computed once per node instead of repeatedly
- Centralized per-ply child-state installation in search
- Bumped the maximum site skill level from 3 to **8**
- Bumped wasm-bindgen dependencies
- `.gitignore`: exclude local dev dotfile markers that aren't build output

### Fixed
- Buggy thread-voting algorithm in multithreaded search, plus related multithreading fixes (shared-TT replacement gate, promotion ordering, TT mate-score clamp)
- Crash when pieces were far away from each other during evaluation
- SEE returning 0 for every quiet move
- Qsearch ignoring the TT move, and failing to store static eval to TT on a non-fail-high stand-pat
- Dead “best move effort” time-management term
- Wrong move receiving a mate score, producing short/garbage PVs
- Root low-ply history hash mismatch
- Zobrist piece keys that were incorrectly XOR-separable
- Asymmetric attack-readiness scaling and asymmetric capture-history updates
- En passant incorrectly classified as a quiet move
- Second killer move being overwritten
- Same move being searched twice
- TT/killer-move castling rights not rebuilt correctly
- Upcoming-repetition checker: fixed a previously-disabled check, then replaced it with a smarter implementation
- Correction-history color-index lookup bug
- Void piece handling bugs, including missed neutral Void occupancy checks during pawn pushes
- Royal-capture logic errors in SPRT and evaluation, and incorrect evaluation of royal captures/threats for the RoyalCapture variant
- Castling-partner check and win-condition handling
- White's promotion-square-attacker off-by-one error
- 7th-rank connector bonus miscalculated for a chess variant
- Obstocean bishop-pawn support term and horde advancement logic
- Missing bounded-only helpmate checks
- Per-node move link not maintained in qsearch
- Clearing of all per-square board planes
- Singular extension polluting the TT entry
- `build.rs` using the wrong commit hash
- SPRT web UI using incorrect variant strings

### Removed
- Dead code from an internal refactor pass

### Improved
- Faster analysis slicing and faster local tile-probe path
- Faster Obstocean quiescence search and PSQT evaluation; added an outside-passed-pawn bonus and adjusted lane bonuses
- Split eval clamping into a pack/unpack pair for clarity
- Fixed several `cargo clippy` warnings
- Reduced unnecessary recompiles by detecting unchanged commit info at build time

## v1.3.0 (2026-04-04)
Commit: `be4c3931e05506fe46018e2d15a8710baaf13f02` • [compare to v1.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168...be4c3931e05506fe46018e2d15a8710baaf13f02)

**It is about 50 Elo better than v1.2.0.**

### Added
- Support for multiple royals per side (game rules, move generation, search, NNUE evaluation)
- Puzzle and game generation tooling (`puzzle_gen`, `game_gen` binaries)
- Native (non-wasm) Engine API for search and clock control, alongside the existing wasm API
- `ARCHITECTURE.md` and expanded project docs
- CLI-based SPRT tester (replacing the old script) and a CLI-based SPSA tuner
- “Scattered Leapers” - a variant that tests the engine's ability to use fairy pieces, in SPRT
- Elo-gain graph in the README
- Commit IDs recorded in SPRT logs
- A handful of bounded helpmate scenarios and a pawn-storm evaluation bonus

### Changed
- Rewrote insufficient-material/helpmate detection (added R+single-bishop, 2N+B, refined R+N/R+B-vs-Q handling) and restructured mop-up evaluation; insufficient-material checks now only apply when both sides use checkmate as their win condition
- Rewrote the “chess” variant evaluation
- Tuned piece values, general evaluation weights, and the mop-up threshold
- Retuned the ray-attack bonus for open diagonal/orthogonal lines, settling on stronger orthogonal weighting; several other evaluation experiments (distance-based king pawn shelter, king-escape penalty, leaper mobility bonus, king attack-unit system) were tried and reverted after failing SPRT testing
- Tuned internal iterative reduction (IIR) parameters in search
- Improved PV extension logic
- SPRT material adjudication now requires both engines to agree on the winner and at least 20 plies played; adjudication is now disabled by default
- SPRT CLI: automatic concurrency detection, unlimited max games by default on the web UI, removed the default game limit, default `elo0` changed to 0.0, wider opening-noise window (8 plies instead of 4)
- GitHub Actions now drive the SPRT CLI directly
- Most tests migrated from manual board construction to ICN-based setup

### Fixed
- A data race where world-border bounds were stored in a shared mutable static, corrupting concurrent SPRT games; made thread-local
- Occasional engine hangs and time losses, by capping quiescence search depth (`MAX_QSEARCH_DEPTH`)
- Repetition-detection bugs via improved position hashing
- Pawn Horde stalemate occurring on move 1
- Huygen blocker/attack detection
- Royal (non-king) piece castling issues, including a missing royal check for the castling partner
- ICN move-list parsing to support capture (`x`) notation
- SPRT file-locking issues when `--new-bin` isn't given, and false “engine failure” results when stopping a run mid-way
- SPRT binary path handling to not hardcode `.exe`

### Removed
- Capture futility pruning

### Improved
- Expanded test coverage for game state, move generation, search parameters, and Zobrist hashing
- SPRT now alerts on game timeouts and orders CI dependencies more reliably

## v1.2.0 (2026-03-02)
Commit: `e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168` • [compare to v1.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/fe5640d774e8baca5a9516e650ef846deb6b34c2...e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168)

**Has a +200 improvement in the Classical variant and a +140 Elo average improvement for all variants compared to v1.1.0.** Its offensive capabilities are quicker and more pronounced, and it's better at producing passed pawns and escorting them to promotion.

### Added
- Neural network evaluation (NNUE): initial framework, feature extraction, and inference for infinite chess
- Helpmate solver: both sides cooperate to help one side get checkmated (new df-pn-based solver, later sped up repeatedly with better hashing and a bounding-box optimization)
- New `gen_nnue_data` and `spsa_tuner` binaries, plus an expanded `perft_icn` test suite
- Difficulty option in SPRT matchmaking, toggled by pressing the `D` key, with improved multi-PV speed and time usage
- Depth limiting based on configured skill level
- Outpost bonus for bishops and knights; open/semi-open file bonuses; king open-file penalty
- Pawn connectivity evaluation term
- NNUE-aware "statscore" move-ordering signal in search/movegen

### Changed
- Search: more aggressive internal iterative reductions (IIR), smarter LMR/singular extensions, dynamic SE margins, NMP verification search, fail-low bonus and ttPv propagation on fail-low, per-offset weights in continuation history
- Transposition tables: Hash-XOR integrity checking, better replacement logic, TT usage in qsearch, TT depth storage for zero-move nodes
- Evaluation: reworked pawn evaluation multiple times, reworked king safety handling for neutral/void pieces, scaled mop-up values to avoid inadvertent underpromotion
- Move ordering: synced movegen and ordering capture scoring, switched history updates to bit shifts instead of division, reduced the capture-history divisor, skip SEE pruning when giving check, precomputed LMR table
- JS/wasm interface switched to use ICN (Infinite Chess Notation) for board interchange, with corresponding SPRT web tooling updates
- Internal data structures: removed the pieces hashmap in favor of full tilemap usage, switched `SpatialIndices` to a struct-of-arrays layout, replaced slow `Vec`/`RefCell` usage in hot paths

### Fixed
- Eval inconsistencies when pieces sit at very large board coordinates
- Movegen getting stuck near far promotion ranks, and other promotion-rank edge cases
- Helpmate solver correctness, including mate scores being incorrectly replaced in its TT, and a bug in `parse_icn_pieces` affecting double promotions
- Several panics in search and Zobrist hashing

### Removed
- Confined Classical custom eval

### Improved
- Faster SEE, evaluation, move generation, piece encoding/decoding, and single-threaded TT access
- Faster/better hashing overall, plus pawn-hash-specific optimizations
- Cached SEE piece values and added 7-dimensional continuation history for move ordering

## v1.1.0 (2026-01-26)
Commit: `fe5640d774e8baca5a9516e650ef846deb6b34c2` • [compare to v1.0.0](https://github.com/FirePlank/infinite-chess-engine/compare/eb24c6d911a69ed388dfab963d648ed59d6d9c61...fe5640d774e8baca5a9516e650ef846deb6b34c2)

**It is notable for the ~300 Elo improvement from v1.0.0.** The engine now prioritizes king safety and doesn't miss simple tactics.

### Added
- Multithreading support with Lazy SMP (currently experimental)
- Support for multiple win conditions
- Persistent transposition table reused across searches, with PV reconstruction that extends from the TT when the recorded PV is incomplete
- Seedable PRNG for reproducible SPRT games
- Minor-piece correction history and pawn-history search heuristics
- Committed the developer `bin/` tools (`apply_params`, `generate_magics`, `spsa_tuner`) to the repository

### Changed
- Rewrote evaluation as a single unified pass instead of many separate scans, including incremental phase calculation
- Reworked king safety evaluation and made material values relative to other piece values
- Overhauled the “chess” variant evaluation and improved defense-urgency/attack-readiness terms
- Discouraged the engine from shuffling/wasting moves without purpose
- Reworked movegen: pruned seemingly useless moves, sped up rose movegen, made cross-ray attack handling smarter
- Shrunk transposition table entries (32 → 24 bytes) and aligned `TTBucket` to 64 bytes; unified TT probing logic and improved TT replacement strategy; TT now also stores static eval and the PV flag
- Reworked move-ordering histories and switched move buffers to `SmallVec`
- Tuned LMR further, added a shuffling guard for singular extensions, and improved the repetition cut-off check
- Reworked time management: spend nearly all available time under the soft limit, and cap time spent on the first move / any single move
- Normalized skill levels; removed the old `noisy.rs` move-selection module in favor of multi-PV-based strength limiting
- SPRT web UI: added support for testing older engine versions, safer defaults, and a confirmation prompt before closing mid-run

### Fixed
- Draw detection no longer misses mate
- A rare stack overflow (increased heap boxing to avoid stack overflows during search)
- Out-of-bounds moves being generated by movegen
- A rare panic when constructing the PV from the TT
- Win-condition checks that were evaluated in reverse
- Pawn count not being restored after undoing a promotion
- En passant handling on promotion, and an issue where the starting board incorrectly had en passant available
- Game state leaking between games during SPRT runs
- Rose blocker detection

### Removed
- The unused knightrider evaluation term and another unimportant heuristic
- `noisy.rs`; its move-selection logic was replaced by multi-PV-based strength limiting

### Improved
- Faster pawn evaluation and a tapered passed-pawn bonus
- Faster rose movegen and removal of a redundant TT check
- Sliders no longer need to be centered in the “cloud” for tropism/positional evaluation
- Improved the internal ICN parser used for test/tooling position construction

## v1.0.0 (2026-01-08)
Commit: `eb24c6d911a69ed388dfab963d648ed59d6d9c61` • [compare to v0.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/e6803a73732817a8f0729fe1ee8cfc8505affae7...eb24c6d911a69ed388dfab963d648ed59d6d9c61)

**The first public release of the engine**, featured in [this video](https://youtu.be/vpE7u6ya1k8). This is **~400 Elo better** than v0.2.0.

### Added
- ProbCut pruning added to search
- TT-move extension added to search
- Cutoff-count tracking for search diagnostics
- New threat-evaluation term, including weighted slider-threat and weighted cloud-center variants
- Connected pawn bonus added to evaluation
- Per-tile piece-type bitmask precomputed alongside occupancy bitboards
- Mate is now emitted as a separate tag in ICN output; a “wb” tag was also added, along with an option to export SPRT games to JSON
- Castling with any piece is now supported (for variants that need it)
- All-pieces-captured win condition now applies to a side that has no royals
- Stalemate detection added to the SPRT harness

### Changed
- Reworked search reductions/extensions and smarter time management
- Pawn evaluation code unified into a single implementation
- SEE logic adjusted
- Move generation reworked to directly produce capture-only and quiet-only move lists
- King safety values retuned
- Cloud-center penalty/bonus tuning refined
- Made the lower difficulty levels easier
- Custom eval variants disabled by default in SPRT
- SPRT web tool: JSON export format simplified to a flat ICN array, download buttons now enable only once a completed game exists

### Fixed
- Sign error in the SEE-based pruning margin (capture pruning compared against the wrong threshold)
- Long-standing transposition-table and neutral-piece bugs
- State restoration bug after unmaking a move
- Knightrider movement bug
- Rose and Huygen check-detection bugs, distant slider capture-detection bug, unhandled orthogonal checks from orthogonal rays, an evasion-generation bug, and a “friendly wiggle room” logic bug, collectively fixing the majority of illegal moves the engine could submit
- Mate finding when the king is far away from sliders
- Mate score could be returned incorrectly when the search was stopped before completing depth 1
- World border was not being reset between SPRT games
- Threefold-repetition detection and position hashing solidified
- En passant, double-move, and promotion bugs, plus related SPRT game-handling fixes

### Removed
- Slider mobility scoring removed from evaluation
- Custom evaluation removed from the palace variant

### Improved
- Legality checking sped up, with fast-check used more broadly
- Transposition table and search internals improved, including better state clearing between searches
- Huygen piece move generation, evasion generation, and check detection substantially improved across several passes
- SPRT now accounts for stalemates
- Resolved nearly all clippy warnings and errors

## v0.2.0 (2025-12-26)
Commit: `e6803a73732817a8f0729fe1ee8cfc8505affae7` • [compare to v0.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/ee4943c08f6f262fe3bfba3e0c424ec5b1785266...e6803a73732817a8f0729fe1ee8cfc8505affae7)

**A large batch of optimizations, bug fixes, and refactoring, improving the engine by ~400 Elo.**

### Added
- Support for all win conditions (`Checkmate`, `RoyalCapture`, `AllRoyalsCaptured`, and `AllPiecesCaptured`)
- SIMD-accelerated evaluation routines
- Internal iterative reductions (IIR)
- Singular extensions
- Continuation history heuristic
- TT move history tracking
- Capture history used in quiet-move pruning
- Hindsight depth adjustment (increase/decrease depth based on prior reduction and opponent response)
- Good/bad quiet move separation in move ordering
- O(1) null-move zugzwang detection
- Opponent-worsening heuristic
- History-adjusted late move reductions (LMR)
- Razoring for depth <= 3
- Multi-cut pruning
- Static exchange evaluation (SEE) module, used to prune bad quiet/capture moves
- Node-type tracking (PV/cut/all) to guide pruning decisions
- Staged move generation with move exclusion, replacing the simpler generator
- Initial Lazy SMP (multithreaded) search support, experimental and not yet strong enough for the default build
- Dynamic correction history (corrhist), tuned per variant
- Triangular principal variation tracking
- MultiPV support
- RNG seeding so the engine doesn't repeat identical games
- TT prefetching on x86_64
- Variant-specific evaluation for Chess, Confined Classical, Obstocean, Palace, and Pawn Horde
- Dedicated mop-up evaluation for king+material endgames, including an improved 2-rook checkmating technique
- Native SPRT runner and SPSA hyperparameter tuner, replacing the old browser-only tuner
- Difficulty setting exposed through SPRT/engine config
- Code coverage tooling and an expanded test suite

### Changed
- Search core rewritten around a Stockfish-style negamax structure: explicit node-type classification, mate-distance pruning, TT mate-score adjustment, and 50-move-rule-aware TT cutoffs
- Move ordering and TT logic split out into dedicated `search/ordering.rs` and `search/tt.rs` modules
- Insufficient-material detection pulled into its own module, backed by a material hash
- Repetition detection/checking logic rewritten for correctness
- Pawn evaluation (advancement and structure) reworked
- Obstacle-piece handling and the Obstocean variant overhauled
- Rust edition bumped to 2024
- SPRT web UI redesigned; SPRT/SPSA now supports all variants and uses randomized/seeded opening moves for reproducibility
- README and sprt/README rewritten and reorganized

### Fixed
- Rose piece move generation and its check-detection logic
- Centaur move generation missing a move
- Huygen fallback producing an illegal move
- Obstacle-piece bugs and a magic-bitboard initialization bug
- Knightrider move generation bug; this and the above collectively reduce the number of illegal moves the engine can produce
- A `material_hash` bug in insufficient-material detection
- Engine getting stuck when pieces were at extremely large coordinate distances
- SPRT reliability issues on some devices/environments

### Removed
- Legacy JS tuner, replaced by the native SPSA tooling
- Dead/unused move-ordering helper functions

### Improved
- Movegen performance via slider caching and cache-friendly hot-path data layout
- Eval/search hot-path data grouped for better cache locality
- SPRT in web now supports all variants, plus a number of other additions
- A better JavaScript API

## v0.1.0 (2025-11-28)
Commit: `ee4943c08f6f262fe3bfba3e0c424ec5b1785266` • [initial version](https://github.com/FirePlank/infinite-chess-engine/commit/ee4943c08f6f262fe3bfba3e0c424ec5b1785266)

**The first released version of the infinite chess engine.** It was not very good at the time.

### Added
- Support for fairy pieces
- A JavaScript API for the engine
- SPRT in web to test improvements (only supports the Classical variant)
- A tuner to adjust values
