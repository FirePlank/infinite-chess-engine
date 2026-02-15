# Setup Guide

This guide walks you through setting up your development environment for HydroChess WASM.

**[← Back to README](../README.md)** | **[Contributing Guide](CONTRIBUTING.md)** | **[Roadmap](ROADMAP.md)**

---

## Tools You'll Need

The following tools are required to build and test the engine. If you don't have them yet, follow the installation steps in the next section.

- **Git** - For version control
- **Rust** - The programming language
- **wasm-pack** - Tool for building Rust to WebAssembly
- **Node.js** (optional) - For running SPRT and SPSA tests

---

## 1. Install Rust (If not installed)

### Windows

Download and run the installer from [rustup.rs](https://rustup.rs/):

```powershell
# Or use winget:
winget install Rustlang.Rustup
```

### macOS / Linux

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, restart your terminal and verify:

```bash
rustc --version
cargo --version
```

---

## 2. Add WebAssembly Target

```bash
rustup target add wasm32-unknown-unknown
```

---

## 3. Install wasm-pack (If not installed)

```bash
cargo install wasm-pack
```

Verify installation:

```bash
wasm-pack --version
```

---

## 4. Clone and Build

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Build for browser
wasm-pack build --target web
```

The built WASM package will be in the `pkg/` directory.

---

## Running Tests

```bash
# Run all unit tests
cargo test --lib

# Run tests with output
cargo test --lib -- --nocapture

# Run a specific test
cargo test test_name --lib
```

### Code Coverage

```bash
# Install llvm-cov
cargo install cargo-llvm-cov

# Run coverage report
cargo llvm-cov --lib
```

---

## 5. Multi-threaded Build

The engine supports parallel search in WebAssembly. This requires a nightly Rust toolchain and specific compilation flags to enable shared memory and atomics.

### Setup for Multithreading

1.  **Install Nightly Rust**:
    ```bash
    rustup toolchain install nightly
    rustup component add rust-src --toolchain nightly
    ```

2.  **Build with Helper Script**:
    Use the provided `build_mt.js` script to build the engine with the correct flags and feature set:
    ```bash
    node build_mt.js
    ```

This script handles the complex configuration required for WASM threads and uses `wasm-pack` to generate the correct JS/WASM bindings in the `pkg/` directory.

---

## IDE Setup

### VS Code

Recommended extensions:

1. **rust-analyzer** - Rust language support

Settings (`.vscode/settings.json`):

```json
{
    "rust-analyzer.check.command": "clippy"
}
```

### IntelliJ / CLion

Install the **Rust** plugin from JetBrains Marketplace.

---

## Next Steps

- **[Contributing Guide](CONTRIBUTING.md)** - Learn the development workflow
- **[Roadmap](ROADMAP.md)** - See planned features and where to help
- **[SPRT Testing](../sprt/README.md)** - Validate engine strength changes
- **[Main README](../README.md)** - Project overview

---

## Useful Links

- [The Rust Book](https://doc.rust-lang.org/book/)
- [wasm-pack Documentation](https://drager.github.io/wasm-pack/book/)
- [Chess Programming Wiki](https://www.chessprogramming.org/)
