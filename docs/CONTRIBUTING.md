# Contributing Guide

Thank you for your interest in contributing to HydroChess! This guide explains the workflow for making changes to the engine.

**[← Back to README](../README.md)** | **[Setup Guide](SETUP.md)** | **[SPRT Testing](../sprt/README.md)** | **[Roadmap](ROADMAP.md)**

---

## Overview

The contribution workflow has three stages:

1. **Implement** - Write your feature or fix
2. **Test** - Verify correctness with unit tests
3. **Validate** - Run SPRT if the change affects playing strength

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Implement   │ --> │    Test      │ --> │    SPRT      │
│  Feature     │     │  (cargo)     │     │  (optional)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Step 1: Implement Your Change

### Types of Changes

| Change Type | Examples | Needs SPRT? |
|-------------|----------|-------------|
| **Bug fix** | Crash fix, move generation error | Usually no |
| **Refactor** | Code cleanup, no behavior change | No |
| **Search improvement** | New pruning, better move ordering | **Yes** |
| **Evaluation change** | New eval term, tuned values | **Yes** |
| **New feature** | New variant, new piece type | Depends |

### Code Style

- Use `cargo fmt` before committing
- Run `cargo clippy` to catch common issues
- Write meaningful commit messages

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy --lib
```

---

## Step 2: Run Tests

All changes must pass the existing test suite.

### Run Unit Tests

```bash
# Run all library tests
cargo test --lib

# Run with output shown
cargo test --lib -- --nocapture

# Run a specific test module
cargo test search::tests --lib
```

### Run Perft Tests

Perft validates that move generation is correct:

```bash
cargo test --test perft
```

### Check Coverage

```bash
# Generate coverage report
cargo llvm-cov --lib

# Aim for >80% line coverage on modified files
```

---

## Step 3: SPRT Testing (For Logic Changes)

If your change affects search or evaluation, you **must** run SPRT to prove it doesn't weaken the engine.

### When to Run SPRT

Run SPRT for:
- ✅ Search algorithm changes (LMR, pruning, extensions)
- ✅ Evaluation term additions or modifications
- ✅ Move ordering improvements
- ✅ Time management changes

Skip SPRT for:
- ❌ Bug fixes (already broken behavior)
- ❌ Code refactoring (no behavior change)
- ❌ Documentation updates
- ❌ Test additions

### Running SPRT

1. **Build your baseline** (before changes):

```bash
# Save current state as "old" engine
wasm-pack build --target web --out-dir pkg-old
```

2. **Make your changes** and rebuild:

```bash
# This happens automatically when you run SPRT
```

3. **Run the SPRT test**:

```bash
cd sprt
npm run dev
```

4. **Open** `http://localhost:3000` and configure your test:
   - Use `all` preset for most changes
   - Mode: `Gainer` (proving improvement) or `Non-Regression` (proving no regression)

5. **Wait for result**:
   - ✅ **PASSED**: Your change is an improvement (or at least not worse)
   - ❌ **FAILED**: Your change weakens the engine
   - ⚠️ **INCONCLUSIVE**: Need more games or different bounds

See **[SPRT Documentation](../sprt/README.md)** for full details.

---

## Pull Request Checklist

Before submitting a PR, verify:

- [ ] Code compiles without warnings (`cargo build --lib`)
- [ ] All tests pass (`cargo test --lib`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Clippy is happy (`cargo clippy --lib`)
- [ ] Logic changes have SPRT results (if applicable)
- [ ] New code has test coverage

---

## Project Architecture

Understanding the codebase:

### Core Modules

| Module | Purpose |
|--------|---------|
| `lib.rs` | WASM bindings, Engine struct |
| `board.rs` | Piece types, coordinates, board representation |
| `game.rs` | GameState, make/undo moves, repetition detection |
| `moves.rs` | Legal move generation for all pieces |

### Search

| Module | Purpose |
|--------|---------|
| `search.rs` | Main iterative deepening + alpha-beta |
| `search/tt.rs` | Transposition table |
| `search/ordering.rs` | Move ordering heuristics |
| `search/see.rs` | Static exchange evaluation |

### Evaluation

| Module | Purpose |
|--------|---------|
| `evaluation/base.rs` | Core evaluation + piece-square logic |
| `evaluation/mop_up.rs` | Endgame evaluation for mating |
| `evaluation/insufficient_material.rs` | Draw detection |
| `evaluation/variants/*.rs` | Variant-specific evaluation |

### Utilities

| Module | Purpose |
|--------|---------|
| `src/bin/*.rs` | Standalone tools: Helpmate solver, SPSA tuner, debuggers (see **[src/bin/README.md](../src/bin/README.md)**) |

---

## Common Tasks

### Adding a New Evaluation Term

1. Add the term in `src/evaluation/base.rs`
2. Add tests to verify the term works correctly
3. Run SPRT to validate it improves play

### Adding a New Piece Type

1. Add the piece to `src/board.rs` (`PieceType` enum)
2. Add move generation in `src/moves.rs`
3. Add evaluation in `src/evaluation/base.rs`
4. Add tests for move generation and evaluation

### Tuning Parameters

Use SPSA for automatic parameter tuning:

```bash
cd sprt
npm run spsa -- --games 60 --iterations 100
```

See **[SPSA Documentation](../sprt/README.md#spsa-logic-tuning)**.

---

## Getting Help

- Check existing tests for examples
- Review similar code in the codebase
- Open an issue for questions

---

## Navigation

- **[← Main README](../README.md)**
- **[Setup Guide](SETUP.md)**
- **[SPRT Testing](../sprt/README.md)**
