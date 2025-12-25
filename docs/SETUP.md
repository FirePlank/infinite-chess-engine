# Setup Guide

This guide walks you through setting up your development environment for HydroChess WASM.

**[← Back to README](../README.md)**

---

## Prerequisites

- **Git** - For version control
- **Node.js** (optional) - For running SPRT tests

---

## 1. Install Rust

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

## 3. Install wasm-pack

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
# Clone the repository (if not already done)
git clone <repository-url>
cd <repository-name>

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

## IDE Setup

### VS Code

Recommended extensions:

1. **rust-analyzer** - Rust language support
2. **CodeLLDB** - Debugging support

### IntelliJ / CLion

Install the **Rust** plugin from JetBrains Marketplace.

---

## Troubleshooting

### "wasm-pack: command not found"

Ensure `~/.cargo/bin` is in your PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.cargo/bin:$PATH"
```

### "error[E0463]: can't find crate"

Run:

```bash
rustup update
rustup target add wasm32-unknown-unknown
```

---

## Next Steps

- **[Contributing Guide](CONTRIBUTING.md)** - Learn the development workflow
- **[SPRT Testing](../sprt/README.md)** - Validate engine strength changes
- **[Main README](../README.md)** - Project overview

---

## Useful Links

- [The Rust Book](https://doc.rust-lang.org/book/)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)
- [Chess Programming Wiki](https://www.chessprogramming.org/)
