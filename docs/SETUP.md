# Setup Guide

This guide walks you through setting up your development environment for Apeiron WASM.

**[← Back to README](../README.md)** | **[Contributing Guide](CONTRIBUTING.md)** | **[Roadmap](ROADMAP.md)**

---

## Tools You'll Need

The following tools are required to build and test the engine. If you don't have them yet, follow the installation steps in the next section.

- **Git** - For version control
- **Rust** - The programming language
- **wasm-pack** - Tool for building Rust to WebAssembly

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

## 2. WebAssembly Target & Toolchain

`rust-toolchain.toml` pins the nightly toolchain and lists the `wasm32-unknown-unknown`
target and `rust-src` component, so rustup provisions them automatically the first time you
build in this directory. No manual `rustup target add` or `rustup toolchain install` is needed.

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

# Build for browser (multi-threaded / Lazy SMP by default)
wasm-pack build --target web --release
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

## 5. Threading

Parallel search (Lazy SMP) in WebAssembly needs shared memory and atomics, which require a
nightly toolchain, `build-std`, and specific link flags. These are all committed to the
repository — `rust-toolchain.toml` pins the toolchain and `.cargo/config.toml` carries the
`build-std` and target flags — and the `multithreading` Cargo feature is on by default. So the
standard build produces the multi-threaded engine:

```bash
wasm-pack build --target web --release            # multi-threaded (default)
wasm-pack build --target web --release --no-default-features   # single-threaded
```

Parallel WASM requires the page to be cross-origin isolated (`Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: require-corp`); without those headers the
engine still loads but runs single-threaded.

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
