# SPRT Testing Tool

Sequential Probability Ratio Test (SPRT) for validating HydroChess engine changes.

**[← Back to README](../README.md)** | **[Contributing Guide](../docs/CONTRIBUTING.md)** | **[Setup Guide](../docs/SETUP.md)**

---

## What is SPRT?

SPRT is a statistical test that determines whether a new engine version is stronger, weaker, or equivalent to a baseline. It's the industry standard for chess engine development, used by Stockfish, Leela Chess Zero, and others.

**Use SPRT when:**
- Changing search algorithms (LMR, pruning, extensions)
- Modifying evaluation terms
- Tuning piece values or parameters

---

## Quick Start

### 1. Build Your Baseline

```bash
# From the engine root directory
wasm-pack build --target web --out-dir pkg-old
```

> [!TIP]
> If you want to test multithreading, build with: `node build_mt.js`.

### 2. Make Your Changes

Edit the code and save.

### 3. Run SPRT

```bash
cd sprt
# For single-threaded testing:
npm run dev

# For multi-threaded testing:
node sprt.js --mt
```

This will:
- Build your modified engine → `sprt/web/pkg-new` (with threads if --mt is used)
- Copy baseline → `sprt/web/pkg-old`
- Start a local server at `http://localhost:3000`

### 4. Open the Web UI

Navigate to `http://localhost:3000` in your browser.

### 5. Configure and Run

1. Select a **Bounds Preset** (see table below)
2. Choose **Mode**: Gainer or Non-Regression
3. Set **TC Mode** and **Time Control** (e.g., Smart Mix or 10+0.1)
4. Click **Run SPRT**

---

## Configuration Options

### Bounds Presets

| Preset | Gainer Bounds | Non-Reg Bounds | Best For |
|--------|---------------|----------------|----------|
| `all` (default) | [0, 10] | [-10, 0] | Most changes |
| `top200` | [0, 5] | [-5, 0] | Small improvements |
| `top30` | [0, 3] | [-3, 1] | Subtle changes |
| `stockfish_stc` | [0, 2] | [-1.75, 0.25] | Very refined changes |
| `stockfish_ltc` | [0.5, 2.5] | [-1.75, 0.25] | Long time control |

### Modes

- **Gainer**: Prove the new engine is stronger (H1: new > old)
- **Non-Regression**: Prove the new engine isn't weaker (H1: new ≥ old)

### Other Settings

| Setting | Default | Description |
|---------|---------|-------------|
| TC Mode | Smart Mix | Randomized mix of time controls |
| Time Control | 10+0.1 | Base time + increment (for Standard) |
| Concurrency | 50 | Parallel games (Web Workers) |
| Min Games | 250 | Minimum games before stopping |
| Max Games | 1000 | Maximum games limit |
| Max Moves | 300 | Moves before forced draw |
| Material Adjudication | 2000 | Eval difference to auto win (cp) |
| Search Noise | 50 | Randomness amplitude for first 4 ply (cp) |
| Alpha / Beta | 0.05 | False positive / False negative rates |

---

## Understanding Results

### Status Colors

| Status | Meaning |
|--------|---------|
| 🟢 **PASSED** | New engine is stronger (LLR ≥ upper bound) |
| 🔴 **FAILED** | New engine is weaker (LLR ≤ lower bound) |
| 🟡 **INCONCLUSIVE** | Need more games |
| ⚪ **ABORTED** | Test stopped manually |

### Statistics Shown

- **W/L/D**: Wins, Losses, Draws for the new engine
- **Elo ± Error**: Estimated rating difference
- **LLR**: Log-likelihood ratio (test statistic)
- **Bounds**: [lower, upper] threshold for decision

### Example Output

```
Game 100: win (W:35 L:25 D:40) Elo≈20.5±12.3 LLR=2.89 [5,4->5,5]

═══════════ SPRT PASSED ═══════════
Total games: 100
Score: 55.0% (35W 25L 40D)
Elo difference: +20.5 ± 12.3
LLR: 2.89 [0.00, 10.00]
Verdict: New engine is stronger
═══════════════════════════════════
```

---

## Game Pairing

Games are run in **color-reversed pairs** to reduce first-move bias:

- Games 0 & 1: Same opening, new engine plays White then Black
- Games 2 & 3: Different opening, same pattern
- ...

SPRT only makes decisions after completing full pairs.

---

## Downloads

The UI provides download buttons:

| Button | Content |
|--------|---------|
| **Copy Log** | Copy game log to clipboard |
| **Download Logs** | Save game logs in ICN format |
| **Download Games (TXT)** | Save games as a `.txt` |
| **Download Games (JSON)** | Save games as a `.json` |

ICN format includes headers like:
```
[Event "SPRT Test Game 12"]
[Result "1-0"]
[Opening "5,2->5,4"]
```

---

## SPSA Parameter Tuning

SPSA automatically optimizes search parameters through self-play.

### Quick Start

```bash
cd sprt
npm run spsa
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--games <n>` | 100 | Games per side per iteration |
| `--iterations <n>` | 100 | Total iterations |
| `--tc <ms>` | 100 | Time control per move (ms) |
| `--concurrency <n>` | 20 | Parallel workers |
| `--fresh` | false | Ignore checkpoints |
| `--verbose` | false | Show detailed updates |
| `--native` | false | Use fast native iwasm execution |
| `--browser` | false | Force browser execution (Puppeteer) |
| `--variant <v>` | Classical | Variant to use for tuning |

### Examples

```bash
# Quick test run
npm run spsa -- --games 20 --iterations 10

# Full tuning session
npm run spsa -- --games 100 --iterations 500

# Resume from checkpoint
npm run spsa  # Auto-resumes if checkpoint exists

# Start fresh
npm run spsa -- --fresh
```

### Checkpoints

- Saved to `sprt/checkpoints/spsa_N.json` every 5% of iterations
- Auto-resumed on next run
- Use `--fresh` to ignore checkpoints

### Configuration

Edit `sprt/spsa_config.mjs` to:
- Add/remove tunable parameters
- Set parameter bounds
- Adjust SPSA hyperparameters (learning rate, perturbation)

---

## Directory Structure

```
sprt/
├── sprt.js              # Build script + dev server
├── spsa.mjs             # SPSA tuner
├── spsa_config.mjs      # Tunable parameters
├── package.json         # npm scripts
├── checkpoints/         # SPSA checkpoints
└── web/
    ├── index.html       # SPRT web UI
    ├── main.js          # UI logic
    ├── sprt-worker.js   # Game worker
    ├── pkg-old/         # Baseline engine
    └── pkg-new/         # Modified engine
```

---

## References

- [SPRT on Chess Programming Wiki](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test)
- [SPSA on Chess Programming Wiki](https://www.chessprogramming.org/SPSA)
- [Stockfish Testing](https://tests.stockfishchess.org/) - Production SPRT system

---

## Navigation

- **[← Main README](../README.md)**
- **[Setup Guide](../docs/SETUP.md)**
- **[Contributing Guide](../docs/CONTRIBUTING.md)**
