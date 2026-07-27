# Changelog

All notable changes to Apeiron (formerly known as HydroChess) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/) — matching the `version` field in `Cargo.toml`.

### Versioning policy

Releases up to and including `v1.3.0` were numbered manually, matching the historical “Apeiron N” milestones announced on the site and elsewhere. `v2.0.0` is a deliberate baseline reset that coincides with the HydroChess → Apeiron rename. From `v2.0.0` onward, version bumps are decided automatically by [`scripts/elo_release.py`](scripts/elo_release.py) based on accumulated SPRT-measured Elo since the last release:

- **+150 Elo** since the last major (`X.0.0`) → major bump
- **+30 Elo** since the last release → minor bump

## [Unreleased] v2.0.0 (2026-07-15)
Commit: `6a407eeb02ac3e7b25659ad146c08b123517963e` • [compare to v1.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/be4c3931e05506fe46018e2d15a8710baaf13f02...6a407eeb02ac3e7b25659ad146c08b123517963e)

**It is currently ~150 Elo better than v1.3.0, with an additional 50 Elo improvement from making multithreading the default.**

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
- A data race where world-border bounds were stored in a shared mutable static, corrupting concurrent SPRT games — made thread-local
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

**The first public release of the engine** — featured in [this video](https://youtu.be/vpE7u6ya1k8). This is **~400 Elo better** than v0.2.0.

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
- Rose and Huygen check-detection bugs, distant slider capture-detection bug, unhandled orthogonal checks from orthogonal rays, an evasion-generation bug, and a “friendly wiggle room” logic bug — collectively fixing the majority of illegal moves the engine could submit
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
- Initial Lazy SMP (multithreaded) search support — experimental, not yet strong enough for the default build
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
- Knightrider move generation bug — this and the above collectively reduce the number of illegal moves the engine can produce
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
