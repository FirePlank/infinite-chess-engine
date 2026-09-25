# Changelog

All notable changes to Apeiron (formerly known as HydroChess) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/), matching the `version` field in `Cargo.toml`.

### Versioning policy

Releases up to and including `v1.3.0` were numbered manually, matching the historical "Apeiron N" milestones announced on the site and elsewhere. `v2.0.0` is a deliberate baseline reset that coincides with the HydroChess → Apeiron rename. From `v2.0.0` onward, version bumps are decided automatically by [`scripts/elo_release.py`](scripts/elo_release.py) based on accumulated SPRT-measured Elo since the last release:

- **+30 accumulated Elo** since the last release → minor bump
- **major bumps are manual**, triggered by running the Auto Release workflow by hand

The accumulator sums each commit's own SPRT-reported Elo, scaled across the 17 site variants. Those figures are *nominal*: per-commit SPRT results are measured against different baselines and don't add up to an A/B measurement, so they consistently overstate the real gain. The bold Elo line under each release is instead the accumulator rescaled against a directly measured head-to-head match between the two releases, so consecutive entries add up to what an actual game would show.

## v6.0.0 (2026-09-25)
Commit: `bfeb847601b2fa5804594542bc4c421e032d46f5` • [compare to v5.5.0](https://github.com/FirePlank/infinite-chess-engine/compare/51544372373ab8d00fd513fcb0ec389dc222b0a0...bfeb847601b2fa5804594542bc4c421e032d46f5)

**It is about 23 Elo better than v5.5.0, and about 250 Elo better than the v5.0.0 baseline.**

Most of that gain since v5.0.0 comes from the eval net: a small quantized MLP that reads terms the hand-crafted evaluation already computes and adds a learned, capped residual to its score, so it covers every fairy piece and custom position the HCE does. It arrived as one generic net in v5.1.0 and now has its own net for each specialized evaluator (Chess, Obstocean, Pawn Horde), trained on depth-9 relabels of archive games.

Search is also faster, even though it now pays for the net on every evaluation. Across the node-oracle sweep it runs about 12% more nodes per second than v5.0.0 and reaches a fixed depth 12-19% sooner. With the net switched off (`APEIRON_EVAL_NET=0`) it runs about 25% more nodes per second, so the net costs about 10% of search speed. Most of this release's speed comes from a behaviour-identical pass over the board representation, measured on both the native and the shipped wasm build. Native and wasm now also search identical trees.

### Added
- `wasm-harness/`, which builds the engine to wasm the way the site ships it (shared memory, Lazy SMP compiled in) and times two builds interleaved under Node; each run first checks the net's SIMD kernel against its scalar reference inside the build being measured
- The Pawn Horde net reads doubled pawns and Black's king cover, two generic-eval terms the Pawn Horde evaluator never scored

### Changed
- An iteration may run to the full move budget; it used to be aborted once it alone had used half, which left the rest of the clock unspent. No new iteration starts past half
- Futility and SEE pruning judge the depth the move would actually be searched at (after the LMR reduction), as in Stockfish; the unreduced depth had left the quiet futility margin about twice as loose
- Draw detection stays on below a null move: the repetition scan stops at the null move, which starts a fresh repetition window, instead of repetition, rule-50 and upcoming-repetition checks all being disabled under it
- Pawns of a side with no promotion rank enter the pawn lists (structure, shelter, open files) but are never scored as passers or candidates
- The wasm build strips panic file/line strings (1,180,139 -> 1,161,883 bytes, no change to generated code)

### Fixed
- SEE sorted a Checkmate king (282) below a knight (315), so it recaptured with the king early and then ended the exchange without using the side's remaining minors; royals now recapture last
- Quiet promotions were emitted by the capture stage, the quiet stage and a promotion killer, inflating the move count that LMR and LMP key off
- Razoring ran inside the singular exclusion search, where quiescence has no excluded move and returned the node's own TT bound on the move under test, so singular moves read as non-singular
- Evasion lists could hold the same capture of the checker up to three times (41% of in-check positions, 8.6% of evasion moves)
- Obstacle boards generated evasions by walking a hash set of squares whose order depends on the width of `usize`, so native and wasm searched different trees; the scans now walk the tiles' colour occupancy, in the same order on every platform
- A skill style change now drops cached scores, a tuning-parameter change drops the pawn cache, and passer king distances are safe in 32 bits

### Improved
- Tiles in a 32x32-tile block around the origin (256x256 squares, where every variant starts) map straight to their table slots, skipping the hash and the probe (-3.3% cycles)
- Rank, file and diagonal indices keep flat arrays in front of their hash maps, and at setup and each search root those windows move onto the densest cluster of pieces, so a few far-off pieces cannot drag them into empty space (-1.9% cycles; -3.0% on Classical shifted 10,000 files)
- The eval net's dense layers use an eight-row AVX2 kernel (forward pass 670 -> 400 ns) with u8 x i8 products in layer 2 (~370 ns), and an eight-row simd128 kernel on wasm (wasm eval -10.5%, search -6.4%); all integer arithmetic, so outputs are bit-identical
- Knightrider movegen sorts pieces onto rays in one pass instead of a division test per piece per direction (+41% NPS on CoaIP_NO), and both it and the eval's ray fill read only the tile squares on the rider's lines through precomputed masks (another -5.5% cycles on CoaIP_NO)
- Rose moves are deduplicated with a bitmask over its 32 distinct spiral squares instead of scanning up to 64 pairs and clearing a 1 KB array per call (-18.8% cycles on CoaIP_RO)
- Move lists (6 KB inline) are no longer copied by value in evasion and castling generation, and line scans read plain slices instead of checking SmallVec storage on every element (-2.7% and -3.7% cycles)
- The shared TT's fields up to 32 bits use plain stores on wasm, like Stockfish's wasm build; wasm has only sequentially consistent atomics, so each relaxed store had been a locked exchange, six per TT write (-9.3% on wasm through the shared table)
- Special rights (castling, pawn double-step) live in a 128x128 bitboard around the origin instead of a hash set, and castling reads only the king's rank
- SEE finds its eight slider rays with four line scans, slider generation reads its per-call state once, and evasions test a leaper's capture of the checker directly instead of generating its full move list
- The eval's pawn lists are insertion-sorted, since the tile walk already emits them in near-sorted runs, and line congestion scans its short lines instead of binary-searching them (wasm -2.0% and -1.2%)

## v5.5.0 (2026-09-24)
Commit: `51544372373ab8d00fd513fcb0ec389dc222b0a0` • [compare to v5.4.0](https://github.com/FirePlank/infinite-chess-engine/compare/1d381cb4961fc44572f695416c1d0d7ee37e2e81...51544372373ab8d00fd513fcb0ec389dc222b0a0)

**It is about 19 Elo better than v5.4.0.**

### Changed
- The Chess, Obstocean and Pawn Horde nets read their own evaluator's terms, written during its pass, instead of running a second generic evaluation per node; the generic net drops eight dead columns (129 -> 121 inputs, so layer 1 pads to 128 lanes instead of 160)
- `nnue/` is renamed to `evalnet/`: the net is a small MLP over HCE terms, not an NNUE

### Fixed
- Checking obstacle captures in Obstocean were searched at every qsearch ply and filled the first-three exemption, chaining 16 plies deep (650k nodes at depth 1); only the first qsearch ply keeps them now, and pawn breakouts are unchanged

### Removed
- Three tuning parameters the eval never read (the middlegame and endgame king tropism bonus and the queen's ideal line distance)

## v5.4.0 (2026-09-24)
Commit: `1d381cb4961fc44572f695416c1d0d7ee37e2e81` • [compare to v5.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/d66c67e0734960f52e994cc3a7bb48678757775e...1d381cb4961fc44572f695416c1d0d7ee37e2e81)

**It is about 28 Elo better than v5.3.0.**

### Added
- Eval nets for the specialized Chess, Pawn Horde and Obstocean evaluators, each a residual over its own evaluator's score trained on that variant's archive games (24.1k, 24.7k and 26.3k games)
- Parameter sweeps through the offline screen, and joint training of several seeds on one copy of the data (about 2.3x faster at the same holdout loss)

### Changed
- A leaper counts as out of play beyond 8 squares from the piece cloud instead of the shared 16, since it only reaches one jump; sliders and riders keep 16. Aimed at far-starting pieces like the CoaIP hawks
- Knightrider 800 -> 900, queen 1380 -> 1518, chancellor 1125 -> 1060, camel 195 -> 175, and the endgame connected-pawn bonus 15 -> 30: the best set from the offline parameter sweeps
- The leaper and rider cloud radii are tunable parameters; sweeps confirmed the defaults of 8 and 16

### Fixed
- An Obstocean board below 60% obstacle fill fell to the generic evaluator and its net, which never trained on obstacles and blew up quiescence: a depth-1 search went from 149 to 10,379 nodes and 39% of Obstocean games were lost on time. Any bounded board with obstacles now stays on the Obstocean evaluator

## v5.3.0 (2026-09-23)
Commit: `d66c67e0734960f52e994cc3a7bb48678757775e` • [compare to v5.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/73ede4800b314d3c7edba103738acf231d21e988...d66c67e0734960f52e994cc3a7bb48678757775e)

**It is about 42 Elo better than v5.2.0.**

### Added
- An offline screen for HCE changes: a reduced export and short training over several seeds for HEAD and the change, about a minute per seed, which ranks variants of an idea before any SPRT
- Exporter tooling for the net: fixed-depth relabelling of any source, perturbed search-tree positions, human games, colour-mirrored twins, and hash-keyed label joins so depth-9 labels carry across an HCE change

### Changed
- The eval net reads the position from the side to move's perspective instead of White's, so a position and its colour mirror evaluate identically; the start-position colour bias (up to +-69cp) is exactly 0
- The net also trains on opening plies (from ply 0 instead of 12), the positions the engine actually searches first
- The net is skipped against a bare king, where it never trains and cannot see the mating geometry, so its residual was only noise on the mop-up gradient
- The eval net is always built; the `eval_net` cargo feature only let `--no-default-features` builds silently lose it, and `APEIRON_EVAL_NET=0` still switches it off at runtime
- SPRT pairs play their opening once: the first game searches its first 8 plies at fixed depth with the pair's noise seed and the second game replays them, since per-engine noise had given only 20% of pairs the same opening
- The training recipe runs 120 base epochs (swept 30-240; 180 and up overfit)

### Fixed
- The pawn hash XORed colour in as a constant, so it survived only as each side's pawn-count parity: swapping ownership of pawns kept the hash, and the pawn cache, pawn history and pawn correction history treated the two structures as one
- The net's king-to-cloud distance input mixed doubled and single units
- A resumed SPRT restarted its timeout counters at zero, so the Final Summary's timeout alert undercounted

### Removed
- The starting-square development term: whether a piece has left its starting square says nothing about where it stands, and in custom setups some pieces are best left home. Starting-square tracking in make/undo goes with it

## v5.2.0 (2026-09-22)
Commit: `73ede4800b314d3c7edba103738acf231d21e988` • [compare to v5.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/884b35fcc9afa596269e8c594a4475207e3712b3...73ede4800b314d3c7edba103738acf231d21e988)

**It is about 54 Elo better than v5.1.0.**

### Changed
- The eval net widens to 129 inputs (pawn structure through the pawn cache, diagonal/orthogonal ray splits, defender histograms, king geometry) with a residual cap of 500cp
- The net is fine-tuned on 898k archive positions relabelled by a depth-9 search
- The net's hidden layers shrink from 256x64 to 128x64: half the forward cost for a slightly less accurate net, and the speed wins

### Improved
- The net's forward pass multiplies 64-byte-aligned i16 rows with `pmaddwd` (1.37 us vs 1.94)

### Removed
- The InfNNUE-v1 full-replacement network: it never shipped, its 14 MB weights were untracked so fresh checkouts could not build it, and it gated 40 sites across search and the tools

## v5.1.0 (2026-09-21)
Commit: `884b35fcc9afa596269e8c594a4475207e3712b3` • [compare to v5.0.0](https://github.com/FirePlank/infinite-chess-engine/compare/8cf0c7a621672bc743ebd1c0237babe644f4b1f3...884b35fcc9afa596269e8c594a4475207e3712b3)

**It is about 84 Elo better than v5.0.0.**

### Added
- The hybrid eval net: a 99 -> 32 -> 32 -> 1 quantized MLP over scalars the HCE already computes, adding a capped residual to the generic evaluator's score, so it works for every fairy piece and custom position; exporter, trainer and quantizer included

### Changed
- Capture history is aged between searches like main history, instead of keeping opening credit at full weight hundreds of plies later
- The puzzle generator drops its per-variant and ply-count caps, which were discarding most candidates once the corpus grew, and excludes Huygen solutions, since its movegen only enumerates fixed prime-distance stops and such a line is usually not really forced

### Fixed
- A singular exclusion search that ran out of moves scored 0, which a negative beta read as a multi-cut for the whole node; it now returns alpha, as in Stockfish
- SEE consulted pin maps built at setup and never updated by make/undo, so inside the tree it reasoned about the root position's pins
- The exact legal-move list offered rose-pinned moves, since a rose pins along a spiral the queen-ray pin test cannot see (208 illegal moves in 7.3M across 24k CoaIP_RO positions)
- The root never filled its own move context, so at ply 1 continuation history was empty, fail-low credit never reached the root move, and qsearch could not see a recapture

### Improved
- The root legal-move filter no longer clones the whole game state per move for strict verification; one scratch state with make/undo serves them all (root movegen 9.4x faster)
- Legality checking trusts the cached royal list instead of rescanning the board for royals it already had (+3.4% NPS on multi-royal variants)
- The fast legality test only gives up for a knightrider that actually stands on a knight ray through the royal, instead of for every move whenever one stood anywhere

### Removed
- Dead code: `is_legal_fast`'s never-returned `Ok(false)` arm, the singular block's always-zero PV bonus, an uncalled history decay, and 1,269 lines of unused magic-bitboard code

## v5.0.0 (2026-09-20)
Commit: `8cf0c7a621672bc743ebd1c0237babe644f4b1f3` • [compare to v4.5.0](https://github.com/FirePlank/infinite-chess-engine/compare/036259218b96859d7ffd5161c4e349a71c8cd66a...8cf0c7a621672bc743ebd1c0237babe644f4b1f3)

**It is about 14 Elo better than v4.5.0, and about 156 Elo better than the v4.0.0 baseline.**

### Changed
- The pawn history table is halved to 1024 buckets; narrowing has consistently won here and widening has consistently lost

### Fixed
- The pawn cache's backward-pawn and candidate-passer terms probed live board occupancy for the stop square, so a piece blocker poisoned an entry keyed on the pawn hash alone; 24.7% of hits disagreed with a fresh call before the fix, 0.1% after

### Improved
- Helper threads under lazy SMP no longer allocate a full local transposition table alongside the shared one they actually read and write; sixteen threads had held about a gigabyte nothing ever consulted

### Removed
- The secondary repetition Zobrist key: it hashed exactly the same state as the primary and both share one coordinate hash, so it guarded against nothing a coordinate collision could not already slip past either; zero vetoes over 22,600 recorded games, and `hash_coordinate` measured collision-free over 58M coordinates spanning the far-escape shells and the i64 play border

## v4.5.0 (2026-09-20)
Commit: `036259218b96859d7ffd5161c4e349a71c8cd66a` • [compare to v4.4.0](https://github.com/FirePlank/infinite-chess-engine/compare/1fb98c8b122bbfef2de46e85139395a953f0c736...036259218b96859d7ffd5161c4e349a71c8cd66a)

**It is about 26 Elo better than v4.4.0.**

### Added
- Every finite leaper (camel, zebra, giraffe, hawk) now has safe-check detection: reversing the leaper's own offsets from the royal gives its checking squares as a finite set whatever the coordinates, whereas only knight geometry was recognised before
- A pin-opportunity-cost penalty: a friendly piece is only charged for being pinned when an enemy slider actually stands behind it on the ray, scaled by how much the piece's own mobility is lost (a pinned rook keeps most of its job, a pinned bishop loses everything), kept beside the existing tied-defender term rather than replacing it

### Changed
- A compound piece (archbishop, chancellor, amazon) can now deliver a safe check through any of its component's geometry instead of only the component that occupies the checking square, since it previously could never be credited for a knight-check square its bishop half can't reach but its knight half can
- Pawn Horde breach scoring replaced Chebyshev distance-to-nearest-pawn with real attack geometry: the distinct horde pawns Black can actually reach (slider first-occupant per ray, knight/king offsets), weighting an unsupported base above a pawn-defended one
- King shelter is now divided by the number of friendly royals standing within two squares of each other, so paired kings that share a pawn wall aren't scored as if each held an independent post the way maze kings do
- SPRT bounds for eval-term tests widened from Stockfish's `[0,2]`/`[-1.75,0.25]` (sized for 20k-100k games) since this repo's budget could never resolve them

### Fixed
- The play border defaulted to +/-1e15 instead of the site's actual `PLAY_BORDER.cap = i64::MAX - 1000`, which hid two overflow bugs: `ray_border_distance` wrapped negative for a slider parked at the far escape square and generated zero moves along that ray (126 legal moves became 92), and mop-up edge distance wrapped the same way and paid a cornered king's full edge bonus to a king standing in open space
- Confined-board (8x8-style) slider/leaper adjustments leaked onto an unbounded world: `(30 - world_size) * 100` saturated and wrapped to 3100 for a huge world size, so an infinite board silently got queen -165/rook -74/bishop -54/knight +31
- A defending royal's mop-up target was chosen by list order instead of distance, so reordering an otherwise-identical position moved the scaled term from 2538 to -207; now requires a single defending royal and picks the mating royal by distance, also fixing a second king being scored twice in 2-king endings
- Kingless cloud-shape geometry anchored to the origin `(0,0)` when no royals remained on either side, so translating a kingless position (reachable once both sides lose every royal under capture-all rules) changed its score; now anchored to the piece centroid
- A merged per-direction nearest-occupant array was shared across all royals for shelter scoring, so one king's shelter was partly computed from another king's lines instead of its own rays
- Fairy pieces (centaur, rose, camel, giraffe, zebra) fell through insufficient-material detection's default classification arm and vanished, so an army made entirely of them read as royals-only and was adjudicated a draw
- The bounded rook/minor drawish scale applied to capture-all endings, discounting a capture-all material edge eightfold on checkmate reasoning that doesn't apply to that win condition

### Improved
- Knightrider attack scoring now walks the board once for all riders instead of once per rider (CoaIP_NO NPS +3.1%)
- The staged move picker's scored-move buffer is pooled from a thread-local instead of allocated fresh per node (NPS +1.0-3.4%)
- The picker's repeated from-square attack tests are memoised per node instead of rescanned for every quiet from the same square (NPS +4.6-6.6%)
- The picker finds its best capture victim from slider/leaper geometry directly instead of generating a full capture list and re-probing the target square (wall-clock -2.5% to fixed depth; SPRT +20.2 Elo)
- `is_square_attacked` answers both directions of a line in one scan instead of hashing and probing each sign separately (NPS +1.4%)
- Qsearch's pooled 8 KB move list is boxed and swapped by an 8-byte handle instead of memcpy'd in and out of a local (NPS +1.0-1.3%)
- Qsearch capture legality is decided with a single threshold-gated SEE test instead of a full exchange walk (NPS +0.4%)
- The slider candidate cache is read through the borrowed slice on a hit instead of a cloned `Arc`, removing the matched refcount increment/decrement
- The Pawn Horde evaluator reuses a thread-local pawn set instead of allocating a fresh `FxHashSet` per evaluation (NPS +1.8% on Pawn Horde)
- The Obstocean board scan reads directly from the colour planes instead of walking every occupied square and dropping the obstacles found there (NPS +2.9% on Obstocean)
- The world size is read once per node instead of once per move in both pruning gates that use it (NPS +2.2%)
- The pawn cache is scored from inside the borrow instead of cloning the whole entry (both passer lists included) to read four fields, and tropism's integer divide is skipped once the divisor exceeds the numerator (common on an unbounded board)
- The cross-ray scan replaces its runtime modulo/divide (always +-1 or +-2) with a dedicated shift-based helper
- The picker's low-ply history index reuses the move-destination hash already computed for `score_quiet` instead of rehashing it
- `Move`'s castling-partner field shrinks from a 24-byte `Option<Coordinate>` to an 8-byte file-plus-sentinel, since castling is always same-rank
- Rose and Knightrider attack scans locate the real pieces through the tile type mask and index a precomputed offset/spiral table instead of probing all 112 (Rose) or sliding up to 160 hashed lookups (Knightrider) per call (NPS +8.8% CoaIP_RO, +3.1% CoaIP_NO)
- A tombstoned tile slot's redundant `clear()` memset is dropped, since a fresh `Tile` is written on reuse anyway (NPS +1.8%)
- The move-ordering picker's already-computed gives-check bit is carried on `ScoredMove` instead of being recomputed by negamax on the same move (NPS +1.2%)
- The TT generation byte is only rewritten on a probe hit when it's actually stale, avoiding a redundant atomic store (neutral natively, real saving under wasm threads where every atomic store is a full seq_cst exchange)
- The spatial index's forward-neighbour loop drops a redundant continuation past the first match, since `insert` guarantees unique coordinates (NPS +0.8%)
- `safe_check_units` precomputes each leaper kind's axis span and step set instead of recomputing them per piece and testing membership with an O(offsets²) search (eval_bench -4.8% cycles; this function was 18.6% of eval time by ablation)

## v4.4.0 (2026-09-19)
Commit: `1fb98c8b122bbfef2de46e85139395a953f0c736` • [compare to v4.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/3ee85705e0e6c19d2a4c3d88ed601ea6ddf1417d...1fb98c8b122bbfef2de46e85139395a953f0c736)

**It is about 24 Elo better than v4.3.0.**

### Added
- Safe checks priced into king danger: a check square is where an enemy piece's line crosses the king's ray, so the cost scales with piece count rather than ray length on an unbounded board (Palace and Scattered_Leapers regressed and are flagged as follow-up)

### Changed
- qsearch keeps captures down to -37 SEE instead of pruning every losing capture at 0, so a sacrifice that is the point of a combination is no longer skipped before it's examined (Stockfish's see_ge(move, -74), halved into this engine's value scale)
- LMR reduces more when the TT move is a capture, so a quiet move competing against a tactical refutation is no longer reduced as if nothing were going on
- Razoring's margin is now Stockfish's linear per-ply form guarded by seek_mate, replacing a quadratic margin that reached ~148 pawns by depth 8 and only ever fired at depth 1-2, with a depth cap papering over it
- The ttPv term is dropped from the LMR reduction entirely; a three-point measurement showed neither this engine's prior sign (-1) nor Stockfish's own sign (+1) beats removing it

### Fixed
- The public legal-move API and the SPRT harness's own terminal detection both filtered the cached slider candidate list, which can omit legal moves that filtering cannot recover
- Neutral Voids were ignored by the evaluator dispatcher, so a bounded 8x8 board carrying them was sent to the Chess evaluator, whose piece-index fallback scored them as black pawns
- setup_position_from_icn left the variant tag, promotion types, win conditions and initial royal counts from the previous position in place, so a reused GameState could report has_lost_by_royal_capture on a position it had just loaded
- is_clear_line_between's diagonal cross product wrapped to a false zero past 2^31, well inside the default world bounds
- LocalTranspositionTable claimed Sync while probe() and penalize() write through a shared reference; it is thread-local in practice and the claim was the only thing making a racy use compile
- TileTable was a fixed 512-slot table that panicked once 512 8x8 tiles were occupied, and never bounded tombstones, which could leave a probe unable to reach an Empty and silently report a live tile as absent

### Improved
- The chess evaluator's pre-pass over every piece is dropped: the occupancy window and pawn file masks now come straight from tile bitboards, leaving the main pass as the only piece walk (+5.4% NPS on Chess)
- Doubled, phalanx, supported, backward and passed pawn predicates are answered from per-colour bitboards with shifts instead of five linear rescans of the pawn list per pawn, while every pawn stays inside 1..=8 (+9.5% NPS on Chess)
- Chess mobility counters read occupancy from a packed u64 window instead of a per-square tile probe through board.get_piece (+1.8% NPS on Chess)
- The move list is iterated by reference instead of copied by value at each of the three generation sites, and the scored list is given a starting capacity so it stops reallocating at every node (+3.4% NPS)
- TileTable's slots are split into parallel states/keys/tiles arrays so a probe scans 1 byte per slot instead of a 64-byte line embedded in each tile, and the table rehashes at a 3/4 load factor instead of the old fixed 512-slot cap

## v4.3.0 (2026-09-17)
Commit: `3ee85705e0e6c19d2a4c3d88ed601ea6ddf1417d` • [compare to v4.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/e0cf5a1c3e39a148504c79b3c61b79076406e790...3ee85705e0e6c19d2a4c3d88ed601ea6ddf1417d)

**It is about 36 Elo better than v4.2.0.**

### Fixed
- Quiet pruning's prune threshold, far-slider gate and `adj_lmr_depth` read main history alone, so a move that followed well after the previous plies was pruned like a stranger; pool pawn and continuation history in, as move ordering and the LMR reduction already do
- The five correction-history sources summed to exactly 100 and had never been scaled as a group, leaving corrections about 30% too weak; scale the sum 1.30x with the mix untouched
- The root scorer keyed its capture branch on the target square, so en passant sorted among the quiets, and promotion gain was only added for capture promotions, even though both were already handled in the staged interior picker
- `capture_sort_key` took no searcher, so quiescence sorted on bare MVV-LVA with no history of any kind across roughly half the tree; thread the searcher in and add the capture-history term the interior picker already applies
- Main history had no decay path outside a full reset, so credit earned in the opening still counted at full weight 200 plies later; scale it by 729/1024 at the start of each search, as Stockfish does
- In-move pruning was gated on `!is_pv`, so a PV node was never pruned at all; prune once a node has diverged from the previous iteration's completed PV, matching Stockfish's `ss->followPV`-based gate

## v4.2.0 (2026-09-16)
Commit: `e0cf5a1c3e39a148504c79b3c61b79076406e790` • [compare to v4.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/ffc8153616bf2399db392b3696c8faf7912f78d2...e0cf5a1c3e39a148504c79b3c61b79076406e790)

**It is about 17 Elo better than v4.1.0.**

### Changed
- The defender's-remaining-room (cage) eval term was keyed only to armies of minors; it now applies to every army at half weight, since it also reads as a bare king's escape signal from any closing net, whatever is doing the closing
- Armies with no wall-capable piece (leapers, same-coloured bishops) scored nothing until fully enclosed, since they cannot cut a line; the flood now finishes the search instead of bailing at the rim and pays gradually for the room within six squares of the king
- Opposite-colour bishops on adjacent diagonals now score as a battery that penalizes separation instead of saturating at nine lines regardless of range, and an archbishop in a thin-wall army is rewarded for bringing its knight component into range

### Fixed
- Reverse futility pruning required no TT move or a capturing one, gating it off at every node whose TT move was quiet — the common case — while its margin multiplier had been tuned for the excluded node class
- Past the sixteen-square slider candidate filter, the move generator never produced the square beside a bare king that builds a mating wall, so no mop-up evaluation could steer the search toward that formation
- In-check nodes fell through to the `ply >= 2` else-arm and came out unconditionally "improving"; they now report not-improving, matching Stockfish, since late-move pruning doesn't gate on check
- Castling validation demanded the square three files out be empty, which at the minimum legal distance is the rook itself, so a legal castle failed validation and a killer naming it was dropped; it now checks rights on both ends, a partner at least three squares away on the same row, and nothing in between
- `recompute_hash` never hashed obstacles even though `make_move` xors a captured one out of the key, so two boards differing only in whether an obstacle still stands hashed identically
- Continuation-history slots for a killer move were built in the quiet stage, which runs after both killer stages return their moves, so a killer searched late got none of the reduction relief its siblings get; the build now happens one stage earlier
- The attack query used for legality caps at 20 hops, but a knightrider attacks without limit, so the root could return a king step onto a square only a far rider covers; the root now also runs an uncapped check for the few riders present
- The ProbCut loop inside singular-extension verification never excluded the move under test, letting it prove an alternative "good" using the very move whose singularity it was checking
- A pawn's double-step right was validated by square geometry but never by the right itself, so a killer move could be replayed for a different pawn now standing on that square, carrying a phantom en-passant state into the tree
- Both halves of the last-move correction-history key were masked to a byte before combining, reaching only 256 of the table's 4096 slots and colliding a move with its own reverse; widened to the full key at no extra memory cost
- A pawn promoting with check was tested using pawn geometry (still its piece at that point), so it was misread as a quiet move: futility-prunable, over-reducible, and droppable by quiescence
- Each ply's PV row was zeroed after three early returns, the hottest being the depth-0 hand-off to quiescence, so a leaf could leave its parent an earlier sibling's row to copy onto the real line; garbage propagated to the root on every alpha raise (scores were always honest, only the displayed line was fiction)

### Improved
- The good-quiet move stage scanned its whole span and the bad-quiet stage then rescanned the same span to pick up what it skipped; bad quiets now compact forward as they're passed, so each stage walks only its own moves
- The knightrider reach term scanned the whole board once per rider; ray tables for every rider on the board are now built together on the first one met, so boards without a knightrider pay nothing
- The knightrider scan buffer was reallocated fresh per rider per stage; it's now reused

### Removed
- The king-attack eval terms that needing a royal on a clear ray or landing square — the queen's clear-line bonus, pawn/leaper royal-threat constants, and the huygen/rose/knightrider/compound royal-arm terms — since static evaluation is never called on a position where a royal is under attack, making all of them permanently inert (measured zero occurrences of a clear slider-to-royal line across 320 sampled positions)

## v4.1.0 (2026-09-13)
Commit: `ffc8153616bf2399db392b3696c8faf7912f78d2` • [compare to v4.0.0](https://github.com/FirePlank/infinite-chess-engine/compare/4285d73b8f8e62f876647cc1abe0860426eccf4e...ffc8153616bf2399db392b3696c8faf7912f78d2)

**It is about 39 Elo better than v4.0.0.**

### Changed
- Quiet history was keyed on piece type and destination only, so White's and Black's moves shared every slot; it now gets a side of its own
- Restored the defensive-tropism king guard behind pawn rank spread instead of raw piece distance, recovering the term the widest boards (Space) need without reintroducing the wash it was on compact boards

### Fixed
- A fail-low node's best move — the least-bad of a set of null-window bounds — was being stored into the TT, evicting whatever a real search had already learned for that position; kept apart from `best_move`, which still feeds the futility margin and the tt-move statistic
- A scoreless (eval-only) TT entry decodes to a real 0 score, and ProbCut's gate read that as proof the node was already below beta
- `eval_stack[0]` (the root's static eval) was never written, so every ply-2 node's "improving" flag effectively read "is the score positive" while ply-1 read the opposite sign; this flag sets the RFP margin, halves the LMP count, adds a reduction, and shifts ProbCut
- The FEN→ICN converter in the UCI binary dropped a coordinate separator, producing malformed ICN tokens
- Razoring's margin carried constants sized for a pawn worth 208; this engine's pawn is worth 100, so it demanded a 40-pawn deficit by depth 3 and fired on 0.128% of the nodes it was offered instead of the ~0.8% it was written for; rescaled to a plain quadratic (232 per depth squared), matching Stockfish's own removal of the linear term
- A blockaded passer (e.g. a knight parked directly in front of it) still collected the full unstoppable-passer bonus, because the race-to-promotion test only ruled out enemy pawns and never looked at what actually occupies the squares in between

### Improved
- Guarded the qsearch prefetch-address computation the way the main loop's already is, removing dead child-hash arithmetic on targets where prefetching is a no-op
- Six behaviour-identical cuts proven by an unchanged node oracle: an unused pin map every capture generator ignores, a redundant in-check scan, per-line allocation in make/undo, a 127-hop rider walk capped to match the legality check's own 20-hop limit, and two dead code paths

## v4.0.0 (2026-09-08)
Commit: `4285d73b8f8e62f876647cc1abe0860426eccf4e` • [compare to v3.7.0](https://github.com/FirePlank/infinite-chess-engine/compare/08ce10a57db8dc58976f32dc2a3cd33bb647335f...4285d73b8f8e62f876647cc1abe0860426eccf4e)

**It is about 21 Elo better than v3.7.0, and about 105 Elo better than the v3.0.0 baseline.**

### Added
- A square-rule term for a passer nothing can catch: reach is computed per piece type, and any enemy slider or rider silences it since those intercept a file in one move. One Knightline position had scored a runaway pawn as a 300cp passer, taking 75x the match budget to resolve by search
- A test asserting every configuration the site's practice mode lists as matable is never scored as a draw

### Changed
- The move-time allocator sizes its horizon from the corpus's real game length (median 224 plies, not the ~50-move chess figure it assumed), spending 1.23x more through the opening and middlegame and holding increment-only late, where games are usually already decided
- Correction history unified onto one style for every position instead of a pawn-based/non-pawn-based split keyed on the `[Variant]` tag, and it no longer learns from a quiescence node's tactical swing (except Obstocean, whose widened quiescence generator genuinely resolves position) or from the singular-exclusion search
- The continuation correction-history table (4MB, read and written on every corrected eval) is dropped; an ablation showed the engine is better without it

### Fixed
- Complexity damping keyed on total phase, so the attacker's own material damped the defender's score; one reported position removed 22,111cp of a 67,980cp material edge. It now fades out only as the weaker side is stripped, matching a 1.17M-position corpus showing the attacker's material never raises the defender's save rate while a well-armed defender needs 2.03x the gap
- Only the mover's own non-pawn correction-history slot was consulted, so half the available correction went permanently unread
- Obstocean's evaluator/quiescence handling and the pawn-based correction mode's last-move/continuation history were keyed on the `[Variant]` tag instead of the position, so an omitted or mistyped tag silently changed evaluation; CoaIP, Classical and Chess had never built or read tables that already existed for the other mode (wiring them cuts CoaIP nodes 26%)
- En passant priced as a victimless move in the qsearch sort key, and a quiet promotion scored as worthless there, so both searched far later than the tactics they are
- A knightrider's attack-reach term credited it through pawns and obstacles its own movegen cannot ride past
- The persistent searcher kept a warm TT and histories from the wrong evaluator family across an Obstocean/Pawn Horde evaluator switch mid-game
- The slider candidate cache's periodic clear ran on the main search's node counter but not qsearch's, making its effective lifetime unbounded in practice (reproduced as a false mate-in-3 in 8x8 Chess); it now clears on both entry paths
- An 8x8 position with a fairy promotion available, or a custom ICN overflowing a specialised evaluator's fixed piece buffers, could be routed to an evaluator that cannot see it or panic mid-evaluation
- The staged move generator kept an invalid TT move for deduplication instead of dropping it, so the real move with the same squares was silently skipped by the killer and quiet stages
- A pawnless leader whose remaining force cannot mate a bare king (K+Q, K+R) kept a full material claim until the trade that made the draw official
- A manual major-release dispatch was blocked by the same `[skip-release]` guard meant only to stop the bot's own bump commit from re-triggering a build
- A Ctrl+C landing during a fault's wind-down swallowed the abort report and left a fabricated "Loss on engine failure" game behind instead of voiding the game

### Removed
- The continuation correction-history table and the pawn-based/non-pawn-based variant split (see Changed)

## v3.7.0 (2026-09-02)
Commit: `08ce10a57db8dc58976f32dc2a3cd33bb647335f` • [compare to v3.6.0](https://github.com/FirePlank/infinite-chess-engine/compare/c6274d1524a408d3f81d505c2febca12914c43df...08ce10a57db8dc58976f32dc2a3cd33bb647335f)

**It is about 8 Elo better than v3.6.0.**

### Changed
- An own piece walls a slider ray at half the congestion cost of a neutral or enemy pawn, since it can step aside
- The attack half of king safety, previously priced at a third of its shelter half, now presses the enemy king half again harder

### Fixed
- A neutral piece one or two squares down a slider ray was not counted as a wall at all, even though it slows development and blocks the attack just like an enemy pawn
- The defensive half of global tropism (crediting a piece for standing near its own king) duplicated what the king-defender bonus and ring cover already price and was pure gravity toward passivity; halving it was Elo-neutral, dropping it entirely was not
- Razoring had no depth cap, so past depth 8 only mate-valued windows could clear its quadratic margin, hollowing the tree exactly where short mates live

## v3.6.0 (2026-08-31)
Commit: `c6274d1524a408d3f81d505c2febca12914c43df` • [compare to v3.5.0](https://github.com/FirePlank/infinite-chess-engine/compare/2e6133e22bdeddd34848763c2e55562332c6c00b...c6274d1524a408d3f81d505c2febca12914c43df)

**It is about 19 Elo better than v3.5.0.**

### Added
- A ported Stockfish `seekMate`: once the root is deep and the score decisive, deep reverse-futility cutoffs and singular extensions switch off, since a static eval cannot tell mate-in-9 from mate-in-10

### Changed
- Late-move reductions read the 1- and 2-ply continuation-history planes the way move ordering already did, weighted up a further step afterward
- ProbCut's verification search shaved a ply, since the SEE gate and beta margin already screen the candidates; the null-move verification search goes a ply deeper now that its steepened margin screens out marginal attempts up front
- Razoring's floor raised alongside its slope, and it razors less as depth grows, since a static deficit is weaker proof at depth on a board this wide
- A null-move cutoff is taken (score clamped to beta) even when its mate distance is unproven, instead of forfeiting the cutoff outright

### Fixed
- The pawn was the last attacker still scored on flat value buckets, and the lowest gate sat above the odd leapers, so a pawn attacking a camel scored nothing
- Nothing priced a slider walled in by its own pieces one or two steps down its rays, the mirror image of the far-slider penalty already in place

## v3.5.0 (2026-08-29)
Commit: `2e6133e22bdeddd34848763c2e55562332c6c00b` • [compare to v3.4.0](https://github.com/FirePlank/infinite-chess-engine/compare/68913072bbdb9fee951c9ffb3131c8f170a589fc...2e6133e22bdeddd34848763c2e55562332c6c00b)

**It is about 9 Elo better than v3.4.0.**

### Changed
- Null-move pruning runs at every non-PV node instead of only cut nodes, with its depth-scaled margin and depth tax raised in three further steps to match how much static surplus a deep cutoff needs on a board this wide; the reverse-futility margin steepened with depth for the same reason
- Mop-up activation no longer caps the winner at ten non-pawn pieces; a bare defender alone identifies a conversion position, so a bigger army still gets the conversion shaping now
- Sliders can generate the long run to the far shell of an open board, where before they could not reach the far edge in one move, hiding an escape or a switch to the other side

### Fixed
- The far-slider penalty ramped to 280cp for a rook, so shifting one far down an empty line read as losing half the piece even though a slider returns to the fight in one move; it now stops at an eighth of the piece's value
- A royal could not step onto an attacked square even under a win condition where the opponent wins by capturing it rather than by mate, so a side whose every royal move was attacked could return no move at all in a non-terminal position

## v3.4.0 (2026-08-27)
Commit: `68913072bbdb9fee951c9ffb3131c8f170a589fc` • [compare to v3.3.0](https://github.com/FirePlank/infinite-chess-engine/compare/a2e6899bff6c43667a6adc476400fe5e3cfdc07c...68913072bbdb9fee951c9ffb3131c8f170a589fc)

**It is about 10 Elo better than v3.3.0.**

### Changed
- Knight threat scoring replaced with a gradient tracking the victim's real value instead of two flat buckets stepping at 400 and 600cp, and the centaur is no longer excluded from its own branch
- Threat credit raised generally, since it read 0.9-2.1 mean centipawns against piece activity's 30.9 (attacking a hanging piece scored far less than merely standing near the board's centre)
- Camel, giraffe and zebra earn value-scaled threat credit on their own offset tables, and read position density the way a knight does; the hawk keeps its own figures on both, since testing showed its old scoring was already correct
- Camel/giraffe/zebra lowered in three steps toward what a 130k-corpus regression says these pieces are worth, since a coarser lattice trades reach for the precision needed to actually land on a target

### Fixed
- The guard sat outside all three threat-scoring paths (not the knight bucket, not a slider, not the leaper arm), so a guard attacking a hanging rook scored nothing at all
- Sizing the huygen sniper's prime-sieve candidate list to exactly its try-count budget silently shrank the set as pieces spread out, and could produce zero candidates past coordinate 719, dropping landings the search relied on; it also carried a per-call heap allocation and an O(line) nearest-prime scan, both removed (+5.6% NPS on the reproduction position)

### Improved
- Faster `evaluate_knightrider_reach` slot assignment (+6.6% NPS averaged over 2 variants)

## v3.3.0 (2026-08-25)
Commit: `a2e6899bff6c43667a6adc476400fe5e3cfdc07c` • [compare to v3.2.0](https://github.com/FirePlank/infinite-chess-engine/compare/bf9c542ead7907fe98c6212d1d3c98fbc6578dc4...a2e6899bff6c43667a6adc476400fe5e3cfdc07c)

**It is about 19 Elo better than v3.2.0.**

### Changed
- The huygen sniper's candidate search asks each landing which enemy it hits in one pass instead of re-deriving the same 24 landings once per target, paying for a candidate cap raised 24 -> 128 and keeping the landing hardest to interpose against
- The cloud-centre bonus is weighted by what the piece is worth instead of a per-type table, so an archbishop no longer pulls as hard as a bishop and a chancellor no longer pulls as hard as a rook worth half as much
- King-defence value cutoffs now ramp with piece value on both ends instead of a flat penalty past 600cp and a hard flip at 400cp
- A knightrider now earns most of the way to what a corpus regression puts it at (824 vs. the prior 720)
- Piece values for the short pieces moved two thirds of the way toward what 130k corpus games say they're worth

### Fixed
- A royal centaur's legal (2,1) leap satisfied the "moved two files" castling test with no same-rank check, so a knight-like leap dragged the rook along and later crashed when undo restored it from the wrong square
- The cloud-distance penalty could reach 128% of a piece's own value at maximum distance and moved a whole step per centipawn at its boundary, so a far-flung piece could score worse than no piece at all; it's now capped and scaled by value directly

### Improved
- The huygen snipe path no longer runs Miller-Rabin per candidate; a sieve now covers the snipe range instead (CoaIP_HO NPS +1.41%, node counts byte-identical)

## v3.2.0 (2026-08-24)
Commit: `bf9c542ead7907fe98c6212d1d3c98fbc6578dc4` • [compare to v3.1.0](https://github.com/FirePlank/infinite-chess-engine/compare/9efd72b32f319a23bc20a4d484ca838ae28c3fc1...bf9c542ead7907fe98c6212d1d3c98fbc6578dc4)

**It is about 10 Elo better than v3.1.0.**

### Changed
- Threat scoring extended to three fairy pieces that had none: the rose (its sixteen spirals resolve to their first occupant, deduped across spirals), the knightrider (each ray resolves to its closest occupant, an enemy a target at any distance) and the huygen (each prime-distance ray terminates at its first occupant); none of the three are sliders, so the existing threat scan never saw them
- A chancellor, archbishop or amazon is credited for the eight knight-leap squares it attacks directly, instead of only having its line moves scanned, since half its attack surface was scoring nothing
- An open knightrider ray rides to the step limit instead of stopping at 5 hops while a blocked ray ran to 10, making longer maneuvers on an empty ray generatable at any depth
- Late-move reductions read whether a node was ever on a principal variation, and reduce harder at cut nodes, on top of the signals already consulted
- Quiet moves are pruned one move earlier (late-move-pruning base 2 -> 3)

### Fixed
- SPRT's ETA estimate improved, and a Ctrl+C no longer counts as an engine crash

### Removed
- The reverse-futility-pruning correction term, which never fired at its actual scale and tested Elo-neutral once corrected

## v3.1.0 (2026-08-21)
Commit: `9efd72b32f319a23bc20a4d484ca838ae28c3fc1` • [compare to v3.0.0](https://github.com/FirePlank/infinite-chess-engine/compare/dc89a0e41da6616dd8b678c93c9166796c99fb46...9efd72b32f319a23bc20a4d484ca838ae28c3fc1)

**It is about 9 Elo better than v3.0.0.**

### Changed
- A transposition-table entry deep enough to cut but holding the wrong bound is penalized a ply instead of being re-probed forever without ever cutting, ported from Stockfish's `TTWriter::penalize`
- The pawn relative-rank penalty ramps gradually for a pawn more than 6 ranks from promotion instead of a flat -48, capping at -100
- Mop-up: the king marches in when it's the army's only wall besides the pieces, and leapers are pulled in hard when the queen is the army's only slider, since a lone queen can't mate and the leaper is the actual mating piece
- A check delivered at the search horizon extends one ply so the reply gets searched, instead of handing resolution to the qsearch boundary
- SPRT saves its games file every 10 games instead of 50, so an interrupted run loses less progress

### Fixed
- Secondary TT aging decayed every deep entry in the local table and never aged the shared table at all; both now decay only a decisive bound, so a stale mate score loses depth while ordinary bounds keep theirs

### Removed
- The dead `cutoff_score` TT field, computed on every probe in both tables but read by no caller

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
