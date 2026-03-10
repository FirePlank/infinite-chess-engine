# SPRT Testing Tool

Sequential Probability Ratio Test (SPRT) tool for validating engine strength changes.

**[← Back to README](../README.md)** | **[Setup Guide](../docs/SETUP.md)** | **[Engine Architecture](../docs/ARCHITECTURE.md)** | **[Contributing Guide](../docs/CONTRIBUTING.md)**

## Overview

SPRT is a statistical test used to determine if a change to the engine results in a strength gain, loss, or is neutral. It is used for tuning search algorithms, evaluation terms, and other parameters.

## Quick Start

### 1. Build Baseline

```bash
wasm-pack build --target web --out-dir pkg-old
```

### 2. Build Modified Engine

After making changes to the source code, run:

```bash
cd sprt
npm run dev
```

This builds the current state into `sprt/web/pkg-new` and starts the test server at `http://localhost:3000`.

### 3. Run Test

1. Open `http://localhost:3000` in your browser.
2. Select a bounds preset (e.g., `all` for [0, 10] Elo).
3. Set your time control and concurrency.
4. Start the test.

## Configuration

### Bounds Presets

| Preset | Gainer Bounds | Non-Reg Bounds |
|--------|---------------|----------------|
| `all` | [0, 10] | [-10, 0] |
| `top200` | [0, 5] | [-5, 0] |
| `top30` | [0, 3] | [-3, 1] |
| `stockfish_stc` | [0, 2] | [-1.75, 0.25] |

### Modes

- **Gainer**: Prove the new version is stronger (H1: new > old).
- **Non-Regression**: Prove the new version is not weaker (H1: new ≥ old).

### Other Settings

| Setting | Default | Description |
|---------|---------|-------------|
| TC Mode | Smart Mix | Randomized mix of time controls and search depths |
| Time Control | 10+0.1 | Base time + increment |
| Concurrency | available cores | Parallel games (Web Workers) |
| Min Games | 250 | Minimum games before stopping |
| Max Games | 1000 | Maximum games limit |
| Max Moves | 300 | Moves before forced draw |
| Material Adjudication | 2000 | Eval difference to auto win (cp) |
| Search Noise | 50 | Randomness amplitude for first 4 ply (cp) |
| Alpha / Beta | 0.05 | False positive / False negative rates |

## SPSA Parameter Tuning

SPSA (Simultaneous Perturbation Stochastic Approximation) is used to automatically tune engine constants through self-play.

### Usage

```bash
cd sprt

# Start tuning
npm run spsa

# Options
npm run spsa -- --games 100 --iterations 500 --concurrency 20
```

Checkpoints are saved to `sprt/checkpoints/` and can be resumed by running the command again. Use `--fresh` to start from scratch.

## Project Structure

- `sprt.js`: Build and server script.
- `spsa.mjs`: SPSA tuning logic.
- `web/`: Web UI for running SPRT tests.
- `web/pkg-old/`: Baseline WebAssembly package.
- `web/pkg-new/`: Modified WebAssembly package.

### References

- [SPRT on Chess Programming Wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test)
- [SPSA on Chess Programming Wiki](https://www.chessprogramming.org/SPSA)
- [Stockfish Testing](https://tests.stockfishchess.org/) - Production SPRT system