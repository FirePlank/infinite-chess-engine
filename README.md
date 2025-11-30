# HydroChess WASM

A Rust-based chess engine compiled to WebAssembly for infinite chess variants.

## Features

### Engine Capabilities

- **Coordinate-based board**: Supports arbitrary piece positions (not limited to 8x8)
- **Multiple piece types**: Standard chess + fairy pieces (Amazon, Chancellor, Archbishop, Centaur, Hawk, Knightrider, etc.)
- **Iterative deepening search** with time management
- **Alpha-beta pruning** with aspiration windows
- **Null move pruning** and **late move reductions (LMR)**
- **Transposition table** with Zobrist hashing
- **Killer moves** and **history heuristic** for move ordering
- **Quiescence search** for tactical accuracy
- **Static Exchange Evaluation (SEE)** for accurate capture pruning and move ordering
- **Coordinate normalization** for infinite board positions
- **Gravity-style history heuristic** with bonuses/maluses and **history leaf pruning** in non-PV nodes
- **Per-ply move buffer reuse** to avoid repeated allocations in the search tree
- **Incremental spatial indices** and **incremental Zobrist hashing** for faster move generation and hashing

### WASM Interface

```rust
// Create engine from game state
let engine = Engine::new(json_state)?;

// Get best move (default time control)
let best_move = engine.get_best_move();

// Get best move with custom time limit
let best_move = engine.get_best_move_with_time(500); // 500ms

// Get all legal moves (for opening generation, UI, etc.)
let moves = engine.get_legal_moves_js();

// Run perft for testing
let nodes = engine.perft(5);
```

### Evaluation

- **Material counting** with piece values
- **Piece-square tables** for positional evaluation
- **King safety** evaluation
- **Pawn structure** analysis
- **Endgame detection** and specialized evaluation
- **Insufficient material** draw detection

## Building

### Browser Target (default)

```bash
wasm-pack build --target web
```

Output in `pkg/` - use with bundlers (esbuild, webpack, etc.)

### Node.js Target

```bash
wasm-pack build --target nodejs --out-dir pkg-node
```

Output in `pkg-node/` - use with Node.js directly

**NOTE: The Node.js target has been seen to be unstable and is not recommended for use. The engine might occasionally hang indefinitely regardless of the thinking time given.**

## Usage in JavaScript

### Engine Protocol

The engine uses a simplified UCI-like protocol:

1. **Initial state**: Send the starting position and move history (no dynamic state)
2. **Move format**: Engine returns moves in coordinate notation with promotion letter
3. **Special moves**: Client reconstructs special flags (en passant, castling) from the move

```javascript
// Game state format for Engine constructor
const gameState = {
    board: {
        pieces: [
            { x: "1", y: "1", piece_type: "r", player: "w" },
            { x: "5", y: "1", piece_type: "k", player: "w" },
            // ... all pieces from STARTING position
        ]
    },
    turn: "w",  // side to move at start of game
    special_rights: ["1,1", "5,1", "8,1"],  // Initial castling/pawn rights only
    en_passant: null,  // Initial en passant only (usually null)
    halfmove_clock: 0,
    fullmove_number: 1,
    move_history: [
        { from: "5,2", to: "5,4", promotion: null },
        { from: "7,7", to: "7,5", promotion: null },
        // ... all moves played so far
    ],
    game_rules: {
        promotion_ranks: {
            white: ["8"],
            black: ["1"]
        },
        promotions_allowed: ["q", "r", "b", "n"]  // Piece letters
    },
    world_bounds: null  // Optional playable region
};

const engine = new Engine(gameState);
const bestMove = engine.get_best_move();
// Returns: { from: "5,4", to: "5,5", promotion: "q" }

// Timed search with evaluation (time in milliseconds). This returns the same
// move shape plus an `eval` field with the final root score in centipawns
// from the side-to-move's perspective.
const bestMoveTimed = engine.get_best_move_with_time(500);
// Returns: { from: "5,4", to: "5,5", promotion: "q", eval: 34 }
```

### Move Conversion

The engine returns moves in a minimal format without special move flags. The client can reconstruct special flags (en passant, castling, etc.) using the game's legal move utilities if needed.

## SPRT Testing

The engine includes a comprehensive SPRT (Sequential Probability Ratio Test) tool, exposed through a
**web-based UI** that compares an old vs new WASM build directly in the browser.

### Start the web SPRT helper (builds web WASM + starts dev server)

```bash
cd sprt
npm run dev
```

This script:

- Treats the root `pkg` directory as the **OLD** engine snapshot
- Builds a new **web** WASM into `pkg-new` via `wasm-pack build --target web --out-dir pkg-new`
- Copies both into `sprt/web/pkg-old` and `sprt/web/pkg-new`
- Starts `npx serve .` in `sprt/web` so the browser UI can import both

Then in your browser, open:

- **URL**: `http://localhost:3000/`

From there you can configure and run SPRT entirely in the UI.

Features:
- **Web-based control panel**: Configure bounds preset/mode, alpha/beta, time per move, concurrency, min/max games
- **Parallel game playing in browser**: Uses Web Workers for concurrency
- **Standard SPRT bounds presets**: `stockfish_ltc`, `stockfish_stc`, `top30`, `top200`, `all`
- **Gainer vs non-regression modes**: Hypothesis testing for different scenarios
- **Random coordinate-based openings**: Random legal white first move, shared by a **pair** of games
- **Color-reversed pairs**: For each opening, one game with the new engine as White, one with the old engine as White
- **Even-game termination**: SPRT will only stop after completing full pairs
- **Rich logging and downloads**:
    - Game progress log with W/L/D, Elo, LLR and opening tag
    - Final / aborted summary blocks
    - Downloadable plain-text logs
    - Downloadable ICN-style game list with `[Result "..."]` metadata

See [sprt/README.md](sprt/README.md) for full documentation and screenshots.

## Project Structure

```
hydrochess-wasm/
├── src/
│   ├── lib.rs          # WASM bindings and Engine struct
│   ├── board.rs        # Board representation and piece types
│   ├── game.rs         # GameState and move making/unmaking
│   ├── moves.rs        # Move generation for all piece types
│   ├── search.rs       # Search entry points and main search logic
│   ├── search/         # Search internals (TT, move ordering, helpers)
│   │   ├── tt.rs       # Transposition table implementation
│   │   └── ordering.rs # Move ordering heuristics
│   ├── evaluation.rs   # Position evaluation
│   └── utils.rs        # Utilities and panic hook
├── sprt/               # SPRT testing helper + web UI
│   ├── sprt.js         # Web helper: builds web WASM & serves sprt/web
│   └── web/            # Browser UI, workers, and WASM packages
├── pkg/                # Browser WASM build (generated)
└── Cargo.toml          # Rust dependencies
```

## Piece Type Codes

| Code | Piece | Code | Piece |
|------|-------|------|-------|
| `p` | Pawn | `m` | Amazon |
| `n` | Knight | `c` | Chancellor |
| `b` | Bishop | `a` | Archbishop |
| `r` | Rook | `e` | Centaur |
| `q` | Queen | `d` | Royal Centaur |
| `k` | King | `h` | Hawk |
| `g` | Guard | `s` | Knightrider |
| `l` | Camel | `o` | Rose |
| `i` | Giraffe | `u` | Huygen |
| `z` | Zebra | `y` | Royal Queen |

