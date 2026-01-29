# Roadmap

This document outlines the high-level goals, planned features, and known technical debt for HydroChess. It serves as a guide for contributors to understand the project's direction and where they can help.

**[← Back to README](../README.md)** | **[Contribution Guide](CONTRIBUTING.md)**

---

## Priority 1: Parameter Fine-tuning

The single biggest opportunity for strength gain is fine-tuning the engine's parameters. Currently, almost all values in `src/search/params.rs`, `src/search.rs`, and `src/evaluation/base.rs` are rough estimates or hand-picked defaults.

### The Plan
1.  **Optimization of SPSA Tuner**:
    - We have an SPSA tuner (`sprt/spsa.mjs`, alternatively `src/bin/spsa_tuner.rs`), but it is currently quite slow on a single machine to yield reliable results in a reasonable amount of time.
    - *Goal*: Optimize the tuner or distribute the workload.
2.  **Hand-tuning via SPRT**:
    - **Alternative**: Contributors can hand-tune the most relevant parameters (e.g., piece values, evaluation weights, LMR reductions) and verify them using the SPRT test suite.
    - This allows for quick gains without too much effort.
    - See `sprt/README.md` for how to run these tests.

### Relevant Files
- `src/search/params.rs`
- `src/evaluation/base.rs`
- `src/bin/spsa_tuner.rs`

---

## Priority 2: Evaluation Logic Improvements

Beyond just tuning numbers, the evaluation function itself needs better metrics to understand Infinite Chess positions.

### The Problem
The current evaluation is a simple adaptation of standard chess rules with a few infinite-specific tweaks. It lacks "smart" metrics for an infinite board, such as better understanding of piece safety, long-range attacks, or unique pawn structures in unbounded space.

### The Plan
- Implement smarter evaluation terms in `src/evaluation/base.rs`.
- experiment with new metrics unique to infinite chess geometry.
- *Note*: Any logic change here **must** be verified with SPRT.

### Relevant Files
- `src/evaluation/base.rs`

---

## Priority 3: Multithreading Optimization

We have a Lazy SMP implementation, but it's not faster or stronger than single-threaded execution due to overhead.

### The Problem
The infrastructure exists (`src/search/shared_tt.rs`, `get_best_move_parallel`), but it needs optimization to actually scale.

### The Plan
1.  **Optimize Shared State**: Improve Transposition Table access patterns for threads.
2.  **Refine Threading Logic**: Better work distribution in `get_best_move_parallel`.

### Relevant Files
- `src/search/shared_tt.rs`
- `src/search.rs`

---

## Good First Issues

If you are looking to contribute but aren't ready to tackle the big items above, here are some smaller tasks:

- **Add Unit Tests**: Increase coverage for the codebase. Run `cargo llvm-cov --lib` to see the current status.
- **Documentation**: Improve documentation for the codebase.
- **Refactoring**: Verify simple refactors with SPRT.

---

## Backlog Ideas

- **NNUE**: Explore Neural Network evaluation for infinite chess.
