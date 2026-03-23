# Architecture

**[← Back to README](../README.md)** | **[Setup Guide](SETUP.md)** | **[Contributing Guide](CONTRIBUTING.md)** | **[SPRT Testing](../sprt/README.md)**

---

This document is a map of the engine.

It is meant to answer two questions for someone new to the codebase:

- where does a given kind of logic live?
- what assumptions is the rest of the engine built around?

If you want implementation detail, use symbol search and the [DeepWiki](https://deepwiki.com/FirePlank/infinite-chess-engine). This file stays at the level of stable structure.

## Bird's-eye view

HydroChess is an engine for chess on an effectively unbounded plane, with support for fairy-piece variants.

The architecture is built around one central decision: **`GameState` owns the position**. Search, evaluation, and the public API all work through it.

The design is shaped by the unique challenge of the infinite plane. We avoid the pitfalls of naive coordinate-stepping and dense arrays by decoupling storage from geometry:

- A **sparse tiled board** stores pieces without paying for empty space.
- **Spatial line indices** for rows, files, and diagonals resolve long-range movement and attacks in logarithmic time.

Everything else hangs off that:

- `src/lib.rs` is the boundary to the outside world;
- `src/game.rs` owns state, rules, and make/undo;
- `src/board.rs` and `src/tiles/` store pieces;
- `src/moves.rs` and `src/attacks.rs` handle movement geometry and attack queries;
- `src/search.rs` and `src/search/` decide what to search;
- `src/evaluation/` scores positions.

If you are trying to understand the project quickly, start with `src/lib.rs`, then `src/game.rs`, then the board/move layers.

## Code map

### `src/lib.rs`

This is the engine boundary.

It exposes the engine to the outside world, especially the WASM/JS side, and is the place where external position/config input turns into a `GameState` plus a search request.

Keep API glue here. Do not let search or evaluation take on UI-facing concerns.

### `src/game.rs`

This is the center of the engine.

`GameState` is the authoritative position object. It owns the board, side to move, special rights, en passant, move counters, repetition state, hashes, cached piece lists, spatial indices, royal locations, check-related caches, and variant rule data.

This is also where move execution lives. If a change affects what a move means, how a rule is enforced, or what needs to survive make/undo during search, it probably belongs here.

When in doubt, look here first.

### `src/board.rs` and `src/tiles/`

This is the physical board representation.

`src/board.rs` defines the low-level language of the engine: coordinates, colors, piece types, and the `Board` abstraction.

`src/tiles/` is the storage strategy. The board is partitioned into 8x8 tiles. Only occupied tiles exist. Inside each tile, occupancy and piece-class information is kept in bitboard form, so local queries stay cheap even though the global board is sparse.

A good mental model is: **sparse globally, dense locally**.

### `src/moves.rs`

This is the geometry layer.

It defines `Move`, move lists, legal/pseudo-legal move generation helpers, world bounds, and `SpatialIndices`.

`SpatialIndices` are the other half of the infinite-board design. They let the engine answer "what is the next blocker on this row/file/diagonal?" without walking across arbitrary distance square by square.

If you are adding a fairy piece, line movement, or anything that depends on blockers, this is one of the first files to inspect.

### `src/attacks.rs`

This is the attack-query layer.

It answers square-attacked questions and related checks used by legality, king safety, and move filtering. It sits between raw movement rules and `GameState`'s legality logic.

If a move is mysteriously legal or illegal, this file and `src/game.rs` are usually the place to start.

### `src/search.rs`

This is the search spine.

It owns iterative deepening, the alpha-beta/negamax loop, quiescence, stopping, and the top-level interaction between search and evaluation.

The important architectural point is that search is a **consumer** of state. It decides what to try, in what order, and how aggressively to prune, but it is not where chess rules should be redefined.

### `src/search/`

This directory holds the search-side policy and machinery:

- `movegen.rs` — staged move generation for search;
- `ordering.rs` — move ordering heuristics;
- `see.rs` — static exchange evaluation;
- `tt.rs`, `shared_tt.rs`, `tt_defs.rs` — transposition-table machinery;
- `zobrist.rs` — hashing keys and helpers;
- `params.rs` — tunable search constants.

This split matters. The general move/rule layer lives in `src/moves.rs`; search-specific ordering and heuristics live here.

### `src/evaluation/`

This is the static evaluation layer.

`base.rs` contains the default hand-crafted evaluation. `helpers.rs`, `mop_up.rs`, and `insufficient_material.rs` support it. `variants/` exists for cases where a variant needs genuinely different scoring rather than a tiny rules tweak.

Evaluation should mostly read already-maintained state and turn it into a score. If it has to rediscover basic positional facts from scratch, something is probably in the wrong place.

### `tests/`

This is the semantic safety net.

`perft.rs` and `perft_icn.rs` check move generation and parsing paths. `endgame_mates.rs` covers mating logic. `static_eval_bench.rs` exists for eval-facing measurement.

If you touch move execution, legality, or attack logic, this directory should move with you.

### `sprt/`

This is for strength testing and tuning.

Use it for answering "did this actually make the engine stronger?" rather than "is this move generator still correct?"

## Boundaries that matter

A few boundaries are easy to miss if you only read files one by one.

### State vs. search

`GameState` defines the position. Search consumes it.

If you are tempted to sneak rule logic into pruning or move ordering, stop and push that logic back toward `src/game.rs`, `src/moves.rs`, or `src/attacks.rs`.

### Storage vs. geometry

The tiled board is about **where pieces are stored**.

`SpatialIndices` are about **how distant relations are queried**.

Those are different jobs. Do not turn the storage layer into the movement layer, and do not reintroduce distance-proportional scans where the index layer exists to avoid them.

### Evaluation vs. rules

Evaluation scores the position it is given.

It should not become a second implementation of move rules, check logic, or board reconstruction.

### API vs. engine core

`src/lib.rs` is the external boundary.

WASM/JS-facing concerns should stop there. The rest of the engine should read like an engine, not like frontend glue.

## A few things worth knowing before moving code around

`GameState` is intentionally "fat". That is not an accident. Search wants a position object with a lot of cached, incrementally maintained facts already attached.

The make/undo path is sacred. New rule state, new caches, and new bookkeeping all have to survive that path cleanly.

Empty space should stay cheap. On an infinite board, any change that makes work scale with geometric distance rather than occupied structure is usually a mistake.

Search-specific move ordering belongs in `src/search/`, not in the generic move layer.

## Cross-cutting concerns

### Hashing and repetition

Zobrist hashing is part of the engine's core plumbing, not an optional optimization. It ties together repetition detection, TT usage, and make/undo correctness.

### Variant support

Variants are not a bolt-on. Rule differences live in `GameRules`/`GameState`, and evaluation has a separate `variants/` layer for cases that need custom scoring.

### Infinite-board assumptions

The engine is not "normal chess with bigger coordinates". Sparse storage, world bounds, and line-indexed blocker queries are foundational design choices.

### Correctness vs. strength

This repository has both kinds of validation:

- correctness checks in `tests/`;
- strength testing in `sprt/`.

Both matter. Passing perft is not the same thing as being a better engine, and gaining Elo is not the same thing as preserving legality.

## Further reading

For a deeper walkthrough of specific subsystems, use the DeepWiki:

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FirePlank/infinite-chess-engine)