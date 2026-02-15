# HydroChess WASM

A high-performance Rust chess engine compiled to WebAssembly, designed for [Infinite Chess](https://www.infinitechess.org/).

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Setup Guide](docs/SETUP.md)** | Install Rust, wasm-pack, and build the engine |
| **[Contributing Guide](docs/CONTRIBUTING.md)** | Workflow for adding features and testing changes |
| **[SPRT Testing](sprt/README.md)** | Run strength tests to validate engine changes |
| **[Roadmap](docs/ROADMAP.md)** | Planned features, technical debt, and backlog |
| **[Utility Binaries](src/bin/README.md)** | Solvers, debuggers, and tuning tools |

---

## ✨ Features

- **Infinite Architecture**: Coordinate-based board supporting arbitrary world sizes and non-standard geometries.
- **Advanced Search**: Iterative deepening PVS with Aspiration Windows, Null Move Pruning, LMR, history-based move ordering, and more.
- **Evaluation**: Modular **HCE** (Material, Cloud Centrality, Pawn Advancement, King Safety, etc.) with experimental **NNUE** support.
- **Scalable Performance**: High-performance Rust core with **Lazy SMP** multithreading.
- **Variants & Fairy Pieces**: Full support for all fairy pieces and unique infinite chess variants.

---

## 🚀 Quick Start

```bash
# 1. Install Rust and wasm-pack (see docs/SETUP.md for details)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 2. Build for browser (Single-threaded)
wasm-pack build --target web

# 3. Build for browser (Multi-threaded)
node build_mt.js

# 4. Output is in pkg/ - ready for use with your bundler
```

For detailed setup instructions, see **[docs/SETUP.md](docs/SETUP.md)**.

---

## 📖 Usage

### JavaScript API

```javascript
import init, { Engine } from './pkg/hydrochess_wasm.js';

await init();

// Initialize engine from ICN string
const icnString = "w 0/100 1 (8|1) P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|...|K5,1+|k5,8+";
const engineConfig = {
    strength_level: 3, // 1=Easy, 2=Medium, 3=Hard (default)
    wtime: 60000,      // White clock in ms
    btime: 60000,      // Black clock in ms
    winc: 1000,        // White increment in ms
    binc: 1000         // Black increment in ms
};

const engine = Engine.from_icn(icnString, engineConfig);

// Get best move
const result = engine.get_best_move();
// Example result: { from: "1,2", to: "1,4", promotion: null, eval: 34, depth: 12 }

// Get best move with time limit (milliseconds)
const result = engine.get_best_move_with_time(500);

// Get all legal moves
const moves = engine.get_legal_moves_js();
```

### Multithreaded usage

To use parallel search, you must initialize the WASM module's thread pool:

```javascript
import init, { Engine, initThreadPool } from './pkg/hydrochess_wasm.js';

await init();
await initThreadPool(navigator.hardwareConcurrency);

const engine = Engine.from_icn(icnString, engineConfig);
const result = engine.get_best_move(); // Now uses all available cores
```

> [!NOTE]
> Parallel WASM requires specific HTTP headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`) to be served by your web server.

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

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Infinite Chess](https://www.infinitechess.org/) - Play infinite chess online
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Engine development resources
- [Stockfish](https://stockfishchess.org/) - The world's strongest open-source chess engine