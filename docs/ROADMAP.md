# Roadmap

This document outlines the high-level goals, planned features, and known technical debt for Apeiron. It serves as a guide for contributors to understand the project's direction and where they can help.

**[← Back to README](../README.md)** | **[Contribution Guide](CONTRIBUTING.md)**

---

## Priority 1: Parameter Fine-tuning

Most of the engine's parameters have already been bracketed by SPRT (razoring, NMP, RFP, LMR, IIR,
late-move pruning, threat weights, correction-history mix) and piece values have had an SPSA pass.
What's still open:

### The Plan
1.  **SPSA on search parameters**: The tuner (`src/bin/spsa.rs`) works, but the ~49 search parameters
    have only ever been hand-bracketed one at a time; a full SPSA pass needs a long uninterrupted run.
2.  **Joint tuning**: every parameter so far was bracketed against a fixed baseline, so interactions
    between them are unexplored.
    - See `sprt/README.md` for how to run these tests.

### Relevant Files
- `src/search/params.rs`
- `src/evaluation/base.rs`
- `src/bin/spsa.rs`

---

## Priority 2: Evaluation Logic Improvements

Beyond just tuning numbers, the evaluation function itself needs better metrics to understand Infinite Chess positions.

### The Problem
The current evaluation is an adaptation of standard chess rules with a few infinite-specific tweaks. It lacks "smart" metrics for an infinite board, such as better understanding of piece safety, long-range attacks, or unique structures in unbounded space.

### The Plan
- Implement smarter evaluation terms in `src/evaluation/base.rs`.
- experiment with new metrics unique to infinite chess geometry.
- *Note*: Any logic change here **must** be verified with SPRT.

### Relevant Files
- `src/evaluation/base.rs`

---

## Good First Issues

If you are looking to contribute but aren't ready to tackle the big items above, here are some smaller tasks:

- **Add Unit Tests**: Increase coverage for the codebase. Run `cargo llvm-cov --lib` to see the current status.
- **Documentation**: Improve documentation for the codebase.
- **Refactoring**: Verify simple refactors with SPRT.

---

## Backlog Ideas

- **Joint parameter tuning**: see Priority 1.
