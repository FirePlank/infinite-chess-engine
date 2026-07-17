# Changelog

## (Unreleased) Apeiron 3 (2026-07-15)
Commit: `6a407eeb02ac3e7b25659ad146c08b123517963e` • [compare to v2.1](https://github.com/FirePlank/infinite-chess-engine/compare/be4c3931e05506fe46018e2d15a8710baaf13f02...6a407eeb02ac3e7b25659ad146c08b123517963e)

- **It is currently ~150 Elo better than v2.1, with an additional 50 Elo improvement with multithreading.**
- Changed the name from “HydroChess” to **“Apeiron”**, a Greek word that means _“the unlimited”_ or _“the boundless.”_ [(Reference: Wikipedia)](https://en.wikipedia.org/wiki/Apeiron "Open the Apeiron page on Wikipedia")
- **Added:**
  - Multi-king variants and “All Pieces Classical” variant in SPRT
  - Presets in SPRT: `all`, `base_only`, `base_full`, `site`, `multi_king`, and `coaip`
  - UCI protocol support
  - Game review web feature
- **Improved:**
  - The evaluation function is now selected based on positional characteristics instead of variant metadata.
  - The variant-specific evaluation is now a lot stronger in Obstocean and Pawn horde
- **Changed:**
  - Made multithreaded Lazy SMP the default build. It currently supports up to **4 threads.**
  - Bump the maximum site skill level from 3 to **8**
- _...Plus many improvements in search and evaluation functions, refactoring, many bug fixes, and other things I forgot to include._

### Apeiron 2.1 (2026-04-04)
Commit: `be4c3931e05506fe46018e2d15a8710baaf13f02` • [compare to v2](https://github.com/FirePlank/infinite-chess-engine/compare/e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168...be4c3931e05506fe46018e2d15a8710baaf13f02)

- **It is about 50 Elo better than Apeiron 2.**
- **Added:**
  - Puzzle generation feature
  - Support for multiple royals per side
  - CLI-based SPRT
  - “Scattered Leapers” - a variant that tests the engine’s ability to use fairy pieces, in SPRT
- **Changed:**
  - Removed the default SPRT game limit, set the `elo0` to 0, and disabled the adjudication by default
  - SPSA is now CLI-based
- **Fixed:**
  - Some issues related to castling
  - Occasional engine hang
  - Resolved most time losses by adding `MAX_QSEARCH_DEPTH`
- _...Plus a couple of evaluation improvements and bug fixes._

## Apeiron 2 (2026-03-02)
Commit: `e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168` • [compare to v1.5](https://github.com/FirePlank/infinite-chess-engine/compare/fe5640d774e8baca5a9516e650ef846deb6b34c2...e9415a4a2adc4581de9bfb3eacc8e60d8d9e9168)

- **Has a +200 improvement in the Classical variant and a +140 Elo average improvement for all variants compared to v1.5.**
- Its offensive capabilities are quicker and more pronounced. In addition, it is better at producing passed pawns and escorting them to promotion!
- **Added:**
  - A helpmate solver: where both sides cooperate to help one side get checkmated
  - A difficulty option in SPRT matchmaking, which can be toggled if a user presses the `D` key
  - An experimental NNUE architecture for infinite chess
- **Removed:**
  - Confined Classical custom eval
- _...Plus a couple of optimizations in search and base evaluation._

### Apeiron 1.5 (2026-01-26)
Commit: `fe5640d774e8baca5a9516e650ef846deb6b34c2` • [compare to v1](https://github.com/FirePlank/infinite-chess-engine/compare/eb24c6d911a69ed388dfab963d648ed59d6d9c61...fe5640d774e8baca5a9516e650ef846deb6b34c2)

- **It is notable for the ~300 Elo improvement from the Apeiron 1!**
- The engine now prioritizes king safety and doesn’t miss simple tactics
- **Added:**
  - Multithreading support with Lazy SMP (currently experimental)
  - Support for multiple win conditions
- **Improved:**
  - A smarter move generation + a few fixes there
  - Better defaults in SPRT
- _...Plus bug fixes and improvements in search and evaluation, and a few things I forgot to include._

## Apeiron 1 (2026-01-08)
Commit: `eb24c6d911a69ed388dfab963d648ed59d6d9c61` • [compare to v0.5](https://github.com/FirePlank/infinite-chess-engine/compare/e6803a73732817a8f0729fe1ee8cfc8505affae7...eb24c6d911a69ed388dfab963d648ed59d6d9c61)

- **The first public release of the infinite chess engine — featured in https://youtu.be/vpE7u6ya1k8**
- This is **~400 Elo better** than Apeiron 0.5.
- **Added:**
  - The ability to download SPRT games in JSON
- **Improved:**
  - SPRT now accounts for stalemates
- **Changed:**
  - Made the lower difficulty levels easier
- **Fixed:**
  - Huygen, rose, and knightrider check detection
  - Couple of problems in move generation, which should hopefully fix the majority of the bugs, where the engine would submit an illegal move
- **Removed:**
  - Palace custom eval
- _...Plus bug fixes and many improvements in search and evaluation._

### Apeiron 0.5 (2025-12-26)
Commit: `e6803a73732817a8f0729fe1ee8cfc8505affae7` • [compare to v0](https://github.com/FirePlank/infinite-chess-engine/compare/ee4943c08f6f262fe3bfba3e0c424ec5b1785266...e6803a73732817a8f0729fe1ee8cfc8505affae7)

- **Did a bunch of optimizations, bug fixes refactoring, and other improvements, which improved the engine by ~400 Elo.**
- **Added:**
  - Variant-specific evaluation for Chess, Confined Classical, Obstocean, Palace, and Pawn Horde
  - Implemented Lazy SMP, which is only experimental as it is not strong enough to be in the default build
  - A bunch of things in search that other chess engines like Stockfish have in common, such as continuation history and extensions
  - Mop-up evaluation: used when one side has an overwhelming material advantage
  - Support for all win conditions (`Checkmate`, `RoyalCapture`, `AllRoyalsCaptured`, and `AllPiecesCaptured`)
- **Improved:**
  - SPRT in web now supports all variants + a bunch of other additions
  - A better JavaScript API
- **Fixed:**
  - Few movegen problems, which fixes some problems with the knightrider, rose, and other pieces. This reduces the number of illegal moves the engine makes.

## Apeiron 0 (2025-11-28)
Commit: `ee4943c08f6f262fe3bfba3e0c424ec5b1785266` • [initial version](https://github.com/FirePlank/infinite-chess-engine/commit/ee4943c08f6f262fe3bfba3e0c424ec5b1785266)

- **The first released version of the infinite chess engine!**
- The current engine was not very good at that time.
- **Added:**
  - Support for fairy pieces
  - A JavaScript API for the engine
  - SPRT in web to test improvements (only supports the Classical variant)
  - A tuner to adjust values
