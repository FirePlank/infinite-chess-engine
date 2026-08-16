# Apeiron

A chess engine for infinite/unbounded-board chess variants (fairy pieces,
custom win conditions), not standard 8x8. Coordinates are `i64` and can be
huge — don't assume a bounded board or a fixed edge to scan to.

## Commands

- Build the engine: `cargo build --release`
- Build the SPRT test binary (feature-gated, easy to forget):
  `cargo build --release --features sprt --bin sprt`
- Test: `cargo test --release --lib -- --test-threads=1` (some tests share
  global board-bounds state and are flaky under parallel threads)
- Lint (must be clean before committing): `cargo clippy --release --all-targets --all-features`
- Any search/eval/movegen behavior change: invoke the `sprt-testing` skill
  BEFORE testing anything — it has the exact binary/variant/bounds/decision
  rules. Don't improvise an SPRT invocation from memory.

## Workflow

- Work autonomously: decide and act, don't stop to ask permission for
  routine engine-improvement work. Keep a test running whenever possible.
- Comments: 2-3 lines max, hard limit. Explain the non-obvious WHY only
  (a hidden constraint, a workaround, something that'll surprise a reader).
  Never write history/changelog comments ("previously X", "fixed Y") —
  that belongs in the commit message.
- Before claiming something is "faster", name what actually dominates the
  cost. Division/allocation/cache-miss swamp micro-choices like branch vs
  branchless; don't optimize the wrong thing.
- Commit messages for engine changes: subject line, 1-3 line body, then the
  SPRT harness's `Final Summary` block pasted verbatim (not hand-condensed —
  `auto-release.yml` parses per-variant `[Name]: ..., Elo: X` lines by regex).
  No AI attribution.
- Verification: be lean, not exhaustive. One full pass (`cargo test --release
  --lib -- --test-threads=1`, clippy, and the node oracle if the change
  touches eval/search/movegen) right before commit/SPRT is enough — don't
  re-run it after every micro-edit. Don't manually confirm two binaries
  differ before an SPRT; the node-oracle shift (or a documented reason it's
  expected to be zero, e.g. a variant-gated change) already IS that proof.
  Don't invent extra sanity checks (isolated eval_icn probes, stash/pop
  before/after comparisons, etc.) unless the oracle genuinely can't cover
  the change — most of the time it can.

## Architecture gotchas (read before touching related code)

- `GameState::get_legal_moves()` is a misnomer: it returns PSEUDO-legal moves
  and keeps the slider candidate cache. An exact list needs
  `get_legal_moves_into` (which bypasses the cache) plus a make/`is_move_illegal`
  /undo filter — what `search_with_searcher` does at its root. So
  `get_legal_moves().is_empty()` is NOT a checkmate test and never fires; any
  mate/stalemate detection written that way silently finds nothing.
- `moves::set_world_bounds` writes process-global atomics (not thread-local), and
  `setup_position_from_icn` calls it whenever the ICN carries a bounds token.
  Tools that process several variants must do so one variant at a time and keep
  parallelism inside a variant, never across.
- `PlayerColor` is `Neutral=0, White=1, Black=2` — NOT 0/1. Code using
  `(color as usize).min(1)` or similar silently pools White and Black into
  one bucket. This exact bug has recurred multiple times (history tables,
  corrhist indexing) and is the first thing to check in any per-color array
  indexed by `PlayerColor as usize`.
- `moves.rs`'s slider candidate cache (keyed `(square, direction)`) is
  DELIBERATELY never invalidated for the interior search — it trades
  occasional staleness for avoiding an O(all-pieces) rescan per slider per
  node. Removing/bypassing it in the interior search is a large NPS
  regression, tested repeatedly. Root/exact move lists (perft, legality,
  the public API) must still bypass it via `set_slider_cache_bypass`, since
  a missed root move is fatal in a way a missed interior move isn't.
- History/pawn-history/killer tables: narrowing (fewer buckets, more
  aliasing) has consistently WON in SPRT; widening (more buckets, wider
  rays, bigger caps) has consistently LOST. Aliasing acts as useful
  cross-region generalization on an unbounded board — don't assume more
  precision is better here.
- Eval-term tuning (piece values, king-safety terms, mobility weights,
  colorboundness, tempo) is extremely SPRT-fragile: many plausible-looking
  changes have tested negative. Treat any eval-term proposal as needing
  real SPRT evidence, not just sound reasoning.
- A piece-value/variant change can pass SPRT in one variant while being
  wrong for another if it changes that value's ordering against pieces
  actually present there — check per-variant piece inventories
  (`starting_icn` in `src/lib.rs`), don't assume.

## Already tried and rejected (don't re-propose without new evidence)

- Removing/weakening the slider candidate cache in the interior search
  (NPS regression, see above).
- Widening history-table buckets/dest hashing, KR ray limits, Rose spiral
  caps, or the 16-square slider-candidate limit.
- NMP margin sign-flip to match Stockfish's exact form (the existing form
  is a deliberate ICE-specific adaptation, not a port bug).
- Stockfish-style gentle aspiration-window widening (the current aggressive
  x4 + 4-retry-cap form is tuned for this engine's eval volatility).
- Root quiet pruning by rank (keep-top-N, the Arimaa mechanic), −26.5 Elo over
  670 games even though it removed the 30-46% of nodes it targeted. A root miss
  is played directly, so ~2% best-move loss ≈ 1 blunder/game and swamps the
  depth gained; interior pruning of the same moves won +41.5. Judge root-side
  pruning by best-move-loss rate per game, never by node share.
- Root-move reordering by previous-iteration score in single-PV search
  (single-PV only gets a real score for the PV move; the rest are noisy
  fail-low bounds, so this replaces good static ordering with noise).
- Blanket qsearch quiet-move exemption for promotions (node-count
  explosion in pawn-dense variants). A narrow, SEE-gated exemption for one
  variant's specific tactic (see the Obstocean breakout handling) is fine.
- Re-keying the tropism addend from royal-owner to attacking-side. The side
  mismatch vs the original +19.4 commit is REAL (confirmed in git history),
  but restoring it tested -6.4 Elo over 2500 games with no winning variant
  class — every later eval change was calibrated against the current form.
- Rescaling/removing the mop-up damping cap without also fixing the
  underlying rule50-clock-baked-into-TT-static-eval bug it papers over —
  tested clean in isolation, but +48% nodes with no Elo gain at 2500 games,
  so it isn't worth it alone.
