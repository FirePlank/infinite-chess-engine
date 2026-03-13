# SPRT Testing Tool

Sequential Probability Ratio Test (SPRT) tool for validating engine strength changes.

**[← Back to README](../README.md)** | **[Setup Guide](../docs/SETUP.md)** | **[Engine Architecture](../docs/ARCHITECTURE.md)** | **[Contributing Guide](../docs/CONTRIBUTING.md)**

## Overview

SPRT is a statistical test used to determine if a change to the engine results in a strength gain, loss, or is neutral. It is used for tuning search algorithms, evaluation terms, and other parameters.

There are two ways to run SPRT: the **native CLI** (lightweight) and the **web UI** (visual).

---

## Native CLI

The CLI is built directly into the `sprt` binary. It manages game pairs, subprocess engines, clocks, adjudication, and reports results.

### Step 1: Build the baseline

Before making your changes, build the current source as the baseline:

```bash
cargo build --release --features sprt --bin sprt
```

Copy or rename the binary so it doesn't get overwritten:

```bash
# Windows
copy target\release\sprt.exe target\release\sprt_old.exe

# Linux/macOS
cp target/release/sprt target/release/sprt_old
```

### Step 2: Make your changes

Edit the engine source code with whatever changes you want to test.

### Step 3: Run the SPRT

The CLI will automatically build the new binary from the current source:

```bash
cargo run --release --bin sprt --features sprt -- run --old-bin target/release/sprt_old
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--new-bin <PATH>` | auto-build | Path to the new engine binary |
| `--old-bin <PATH>` | **required** | Path to the old (baseline) engine binary |
| `--tc <TC>` | `10+0.1` | Time control: `base+inc` (seconds), `depth N`, or `fixed Ns` |
| `--concurrency <N>` | `16` | Number of parallel games |
| `--max-games <N>` | unlimited | Maximum games to run |
| `--min-games <N>` | `250` | Minimum games before SPRT can terminate |
| `--elo0 <F>` | `0.0` | H0 bound (Elo where new is NOT better) |
| `--elo1 <F>` | `5.0` | H1 bound (Elo where new IS better) |
| `--alpha <F>` | `0.05` | Type I error rate (false positive) |
| `--beta <F>` | `0.05` | Type II error rate (false negative) |
| `--adjudication <N>` | `2000` | Material eval difference (cp) to auto-adjudicate |
| `--max-moves <N>` | `300` | Max plies before forced draw |
| `--search-noise <N>` | `50` | Noise amplitude (cp) for first 8 ply |
| `--old-strength <N>` | `3` | Strength level for old engine (1-3) |
| `--games <PATH>` | — | Write game ICNs to a JSON |
| `--results <PATH>` | — | Write results to a JSON |
| `--variants <LIST>` | all except custom eval | Comma-separated variant list |
| `--verbose` | off | Print detailed game info |

### Example: Small regression test

```bash
cargo run --release --bin sprt --features sprt -- run --old-bin target/release/sprt_old \
  --tc 1+0.01 \
  --concurrency 8 \
  --max-games 200 \
  --games games.json
```

Afterwards you can drop the games JSON into [the ICN validator](https://infinitechess.org/icnvalidator) to catch illegal moves or bad terminations, though some discrepancies are expected in certain insufficient material and Huygen mate cases.

---

## Web UI

For visual feedback and interactive configuration, use the browser-based SPRT.

### Step 1: Build the baseline

```bash
wasm-pack build --target web --out-dir pkg-old
```

### Step 2: Build & Run

After making changes:

```bash
cd sprt
npm run dev
```

This builds the current source into `sprt/web/pkg-new` and starts the test server at `http://localhost:3000`.

### Step 3: Configure & Run

1. Open `http://localhost:3000` in your browser
2. Select bounds preset, time control, and concurrency
3. Start the test

---

## SPSA Parameter Tuning

SPSA (Simultaneous Perturbation Stochastic Approximation) is used to automatically tune engine constants through self-play.

```bash
cd sprt

# Start tuning
npm run spsa

# Options
npm run spsa -- --games 100 --iterations 500 --concurrency 20
```

Checkpoints are saved to `sprt/checkpoints/` and can be resumed by running the command again. Use `--fresh` to start from scratch.

## Project Structure

- `src/bin/sprt.rs` — Native CLI (SPRT manager + search subprocess)
- `sprt.js` — Build and server script (web UI)
- `spsa.mjs` — SPSA tuning logic
- `web/` — Web UI for running SPRT tests
- `web/pkg-old/` — Baseline WebAssembly package
- `web/pkg-new/` — Modified WebAssembly package

### References

- [SPRT on Chess Programming Wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test)
- [SPSA on Chess Programming Wiki](https://www.chessprogramming.org/SPSA)
- [Stockfish Testing](https://tests.stockfishchess.org/) — Production SPRT system
