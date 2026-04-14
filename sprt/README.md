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
| `--concurrency <N>` | logical CPU count | Number of parallel games |
| `--max-games <N>` | unlimited | Maximum games to run |
| `--min-games <N>` | `250` | Minimum games before SPRT can terminate |
| `--elo0 <F>` | `0.0` | H0 bound (Elo where new is NOT better) |
| `--elo1 <F>` | `5.0` | H1 bound (Elo where new IS better) |
| `--alpha <F>` | `0.05` | Type I error rate (false positive) |
| `--beta <F>` | `0.05` | Type II error rate (false negative) |
| `--adjudication <N>` | `0` | Eval difference (cp) to auto-adjudicate (0 = disabled) |
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

For visual feedback and interactive configuration, use the browser-based SPRT. You'll need Node for this.

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

SPSA (Simultaneous Perturbation Stochastic Approximation) is used to automatically tune search and evaluation constants through self-play.

The tuner lives in `src/bin/spsa.rs` and uses a single feature gate for dynamic parameter injection during tuning:

```bash
cargo run --release --bin spsa --features sprt,param_tuning -- run
```

### Parameter Selection

`--params` controls which knobs are tuned:

| Selector | Meaning |
|----------|---------|
| `all` | Tune every exposed search and eval parameter |
| `search` | Tune only search parameters from `src/search/params.rs` |
| `eval` | Tune only evaluation parameters from `src/evaluation/base.rs` |
| `piece-values` | Tune only evaluation material / piece-value style knobs |
| `pawn,knight,...` | Tune only the explicitly named parameters |

### SPSA CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `run --iterations <N>` | `100` | Number of SPSA iterations |
| `run --pairs <N>` | `400` | Paired openings per iteration; total games = `pairs * 2` |
| `run --checkpoint-every <N>` | `1` | Save a checkpoint every N iterations |
| `run --resume <PATH>` | latest checkpoint | Resume from a specific checkpoint |
| `run --fresh` | off | Ignore checkpoints and start from defaults |
| `run --tc <TC>` | `3+0.03` | Time control: `base+inc`, `depth N`, or `fixed Ns` |
| `run --concurrency <N>` | `16` | Number of parallel game workers |
| `run --variants <LIST>` | default set | Comma-separated variant list |
| `run --adjudication <N>` | `2000` | Eval threshold for adjudication |
| `run --max-moves <N>` | `300` | Maximum plies before forced draw |
| `run --search-noise <N>` | `50` | Noise amplitude for first 8 ply |
| `run --params <SELECTOR>` | `all` | Parameter preset or comma-separated names |
| `run --config <PATH>` | none | Optional JSON override for bounds/defaults/`c_end`/`r_end` |
| `run --results <PATH>` | `sprt/spsa_final.json` | Final result JSON output |
| `run --games <PATH>` | off | Write latest iteration ICNs as JSON |
| `run --big-a <F>` | `iterations / 10` | SPSA stability constant `A` |
| `run --alpha <F>` | `0.602` | SPSA learning-rate decay |
| `run --gamma <F>` | `0.101` | SPSA perturbation decay |
| `run --verbose` | off | Inherit search subprocess stderr |
| `list --params <SELECTOR>` | `all` | Print selected tunables with bounds, `c_end`, and `R_end` |
| `apply --input <PATH>` | latest checkpoint | Apply tuned constants back into Rust source |
| `revert --params <SELECTOR>` | `all` | Revert selected constants back to defaults |

### Tuning Config Overrides

`--config` accepts a JSON object keyed by parameter name. Each entry can override any subset of `default`, `min`, `max`, `c_end`, and `r_end`.

```json
{
  "knight": { "min": 180, "max": 340, "c_end": 4.0, "r_end": 0.0020 },
  "bishop": { "default": 430, "c_end": 4.0, "r_end": 0.0015 },
  "razoring_linear": { "min": 300, "max": 650, "c_end": 16.0, "r_end": 0.0020 }
}
```

### Examples

Tune all exposed params on the default variant set:

```bash
cargo run --release --bin spsa --features param_tuning -- run --pairs 100 --iterations 500 --concurrency 20
```

Tune only piece values at `5+0.1` across the default variants:

```bash
cargo run --release --bin spsa --features param_tuning -- run --tc 5+0.1 --params piece-values
```

Tune only a hand-picked subset:

```bash
cargo run --release --bin spsa --features param_tuning -- run --params pawn,knight,bishop,rook,mg_bishop_pair_bonus
```

Inspect the tunable set before a run:

```bash
cargo run --bin spsa --features param_tuning -- list --params eval
```

Apply the latest checkpoint back into source:

```bash
cargo run --bin spsa --features param_tuning -- apply
```

Checkpoints are saved to `sprt/spsa_checkpoints/` by default and resume automatically unless `--fresh` is passed.

---

## Game Review Tool

The web UI also includes an interactive game review tool (`web/review/`) for analyzing game positions and moves:

### Features

- **Move Classification**: Automatically classifies moves as Best, Excellent, Good, Inaccuracy, Mistake, Blunder, or Forced
- **Evaluation Graph**: Visual representation of position evaluation throughout the game with hover tooltips
- **Move List**: Interactive move list showing evaluation after each move with classification symbols
- **Accuracy Stats**: Per-side accuracy percentages and classification breakdowns

---

## Project Structure

- `src/bin/sprt.rs` — Native CLI (SPRT manager + search subprocess)
- `sprt.js` — Build and server script (web UI)
- `src/bin/spsa.rs` — Match-based SPSA CLI (runner + search subprocess + apply/revert)
- `web/` — Web UI for running SPRT and game review
- `web/review/` — Game review tool

### References

- [SPRT on Chess Programming Wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test)
- [SPSA on Chess Programming Wiki](https://www.chessprogramming.org/SPSA)
- [Stockfish Testing](https://tests.stockfishchess.org/) — Production SPRT system

