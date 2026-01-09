# HydroChess WASM

A high-performance Rust chess engine compiled to WebAssembly, designed for [Infinite Chess](https://www.infinitechess.org/) variants.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Setup Guide](docs/SETUP.md)** | Install Rust, wasm-pack, and build the engine |
| **[Contributing Guide](docs/CONTRIBUTING.md)** | Workflow for adding features and testing changes |
| **[SPRT Testing](sprt/README.md)** | Run strength tests to validate engine changes |

---

## ✨ Features

### Search
- **Iterative deepening** with aspiration windows
- **Alpha-beta pruning** with principal variation search
- **Null move pruning** and **late move reductions (LMR)**
- **Transposition table** with Zobrist hashing
- **Killer moves** and **history heuristic** for move ordering
- **Quiescence search** for tactical accuracy
- **Static Exchange Evaluation (SEE)** for capture pruning

### Evaluation
- Material counting with tuned piece values
- Piece-square considerations for positional play
- King safety evaluation
- Pawn structure analysis (isolated, doubled, passed pawns)
- Endgame detection with specialized mop-up evaluation
- Insufficient material draw detection

### Infinite Chess Support
- **Coordinate-based board**: Arbitrary positions (not limited to 8x8)
- **Fairy pieces**: Amazon, Chancellor, Archbishop, Centaur, Hawk, Knightrider, Rose, Huygen, and more
- **Variant-specific evaluation**: Chess, Confined Classical, Obstocean, Pawn Horde, Palace

---

## 🚀 Quick Start

```bash
# 1. Install Rust and wasm-pack (see docs/SETUP.md for details)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 2. Build for browser
wasm-pack build --target web

# 3. Output is in pkg/ - ready for use with your bundler
```

For detailed setup instructions, see **[docs/SETUP.md](docs/SETUP.md)**.

---

## 📖 Usage

### JavaScript API

```javascript
import init, { Engine } from './pkg/hydrochess_wasm.js';

await init();

const engine = new Engine(gameStateJson);

// Get best move with time limit (milliseconds)
const result = engine.get_best_move_with_time(500);
// Returns: { from: "5,2", to: "5,4", promotion: null, eval: 34 }

// Get all legal moves
const moves = engine.get_legal_moves_js();
```

### Game State Format

```javascript
const gameState = {
    board: {
        pieces: [
            { x: "1", y: "1", piece_type: "r", player: "w" },
            { x: "5", y: "1", piece_type: "k", player: "w" },
            // ... all pieces
        ]
    },
    turn: "w",
    special_rights: ["1,1", "5,1", "8,1"],
    en_passant: null,
    halfmove_clock: 0,
    fullmove_number: 1,
    move_history: [
        { from: "5,2", to: "5,4", promotion: null },
        // ... moves played
    ],
    game_rules: {
        promotion_ranks: { white: ["8"], black: ["1"] },
        promotions_allowed: ["q", "r", "b", "n"]
    }
};
```

---

## 🧪 Testing

```bash
# Run all unit tests
cargo test --lib

# Run with coverage
cargo llvm-cov --lib

# Run perft tests (move generation validation)
cargo test --test perft
```

For testing engine strength changes, see **[sprt/README.md](sprt/README.md)**.

---

## 📁 Project Structure

```
hydrochess-wasm/
├── src/
│   ├── lib.rs              # WASM bindings and Engine struct
│   ├── board.rs            # Board representation and piece types
│   ├── game.rs             # GameState, make/undo moves, repetition
│   ├── moves.rs            # Move generation (legal / pseudo-legal)
│   ├── attacks.rs          # Fast attack bitboard lookups
│   ├── search.rs           # Main IDDFS search loop
│   ├── search/             # Search internals
│   │   ├── tt.rs           # Transposition table
│   │   ├── ordering.rs     # Move ordering heuristics
│   │   ├── see.rs          # Static exchange evaluation
│   │   ├── noisy.rs        # search.rs but with eval noise
│   │   ├── movegen.rs      # Search-specific move ordering
│   │   └── params.rs       # Tunable search constants
│   ├── evaluation/         # Position evaluation
│   │   ├── base.rs         # Core evaluation + piece-square logic
│   │   ├── mop_up.rs       # Endgame evaluation for mating
│   │   ├── insufficient_material.rs # Draw detection
│   │   └── variants/       # Variant-specific evaluation
│   ├── tiles/              # Bitboard tile management
│   │   ├── mod.rs          # Core tile structure
│   │   └── magic.rs        # Magic bitboards (sliding pieces)
│   └── simd.rs             # Hardware acceleration (optional)
├── sprt/                   # Testing and Tuning tools
│   ├── sprt.js             # Web SPRT helper
│   ├── spsa.mjs            # SPSA parameter tuner
│   └── texel_tuner.js      # Texel evaluation tuner
├── tests/                  # Integration tests
│   ├── perft.rs            # Move generation accuracy
│   └── endgame_mates.rs    # Mating pattern validation
├── docs/                   # Documentation
│   ├── SETUP.md            # Installation guide
│   └── CONTRIBUTING.md     # Contribution workflow
└── pkg/                    # WASM build output (generated)
```

---

## 🎯 Piece Type Codes

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

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Infinite Chess](https://www.infinitechess.org/) - Play infinite chess online
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Engine development resources
- [Stockfish](https://stockfishchess.org/) - The world's strongest open-source chess engine