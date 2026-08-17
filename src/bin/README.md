# Utility Binaries

**[← Back to README](../../README.md)** | **[Setup Guide](../../docs/SETUP.md)** | **[Engine Architecture](../../docs/ARCHITECTURE.md)** | **[Contributing Guide](../../docs/CONTRIBUTING.md)** | **[SPRT Testing](../../sprt/README.md)**

Standalone scripts and tools for development, debugging, and tuning.

## List of Binaries

### 1. Helpmate Solver (`helpmate_solver.rs`)
A solver for Helpmate problems in Infinite Chess using the DF-PN (Depth-First Proof-Number Search) algorithm.

```bash
cargo run --release --bin helpmate_solver --features parallel_solver -- --icn "<ICN>" --mate-in <PLIES> --mated-side <COLOR>
```

### 2. Evaluation Debugger (`eval_icn.rs`)
Prints a detailed breakdown of evaluation scoring for a specific position.

```bash
cargo run --bin eval_icn "<ICN_STRING>"
```

### 3. SPRT CLI (`sprt.rs`)
A high-performance SPRT tool for comparing engine versions using native execution and subprocess communication.

```bash
cargo run --release --bin sprt --features sprt -- run --old-bin target\release\sprt_old.exe
```

### 4. SPSA Tuner (`spsa.rs`)
A match-based SPSA tuner that runs self-play directly from the CLI and can also apply or revert tuned search constants.

```bash
cargo run --release --bin spsa --features param_tuning -- run
```

### 5. Magic Bitboard Generator (`generate_magics.rs`)
Computes magic number constants for sliding piece attack generation. Currently unused.

```bash
cargo run --release --bin generate_magics
```

### 6. Game Generator (`game_gen.rs`)
Generates sample games for use in puzzle generation.

```bash
cargo run --release --bin game_gen --features puzzle_gen,rand
```

### 7. Puzzle Generator (`puzzle_gen.rs`)
Mines a self-play game corpus (the `games*.json` files an SPRT run's `--pgn`/game
logging produces) for sound tactical puzzles: a position where one move wins (or
saves a lost game, or forces a draw) and every alternative provably does not.
Every puzzle is deep-verified against the current engine, not just the shallow
eval the games were originally annotated with, and is rated on an absolute scale
based on how much there actually is to calculate -- not merely how many plies.

```bash
# Generate from every games*.json under one or more directories
cargo run --release --bin puzzle_gen --features puzzle_gen -- \
  --corpus path/to/games --out puzzles.csv

# Re-search each stored puzzle at high depth and drop anything no longer sound
cargo run --release --bin puzzle_gen --features puzzle_gen -- \
  --deep-verify --out puzzles.csv

# Point at every session/work directory under a shared root instead of listing
# each --corpus by hand (root via --auto-corpus-root or PUZZLE_GEN_AUTO_CORPUS_ROOT)
cargo run --release --bin puzzle_gen --features puzzle_gen -- \
  --auto-corpus --auto-corpus-root /path/to/parent --out puzzles.csv
```

Runs are resumable throughout: candidates already tried are checkpointed
(`<out>.progress`), corpus files already fully mined are recorded in a persistent
manifest (default `corpus_seen.jsonl`) so a second invocation only looks at what's
new, and `--recook`/`--deep-verify` each keep their own checkpoint too. Pass
`--explain "<move prefix>"` against an existing CSV to print the per-move
difficulty breakdown behind a puzzle's rating. Run with `--help` for the full
flag list.

### 8. UCI Protocol Bridge (`uci.rs`)
A UCI-compliant chess engine interface for standard 8×8 chess. Accepts UCI commands on stdin and outputs moves/info to stdout. Compatible with any UCI GUI (Cutechess, Arena, Lichess, etc.).

```bash
cargo build --bin uci --release
./target/release/uci.exe
```

### 9. Texel Tuner (texel.rs)

A static-eval Texel tuner for `src/evaluation/base.rs` constants. Fits eval parameters to a `data_gen` corpus and applies the tuned values back to the source. Use one bounds-group corpus (the `data_gen` default `base_only` preset is the usual choice).

```bash
cargo run --release --bin data_gen --features data_gen -- --games 100000
cargo run --release --bin texel --features eval_tuning -- run
cargo run --release --bin texel --features eval_tuning -- apply
```
