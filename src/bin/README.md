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
Extracts tactical puzzles from a game database using win-chance metrics and theme detection.

```bash
cargo run --release --bin puzzle_gen --features puzzle_gen
```