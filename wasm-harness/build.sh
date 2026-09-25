#!/bin/bash
# build.sh <out-dir> [mt]
# Builds the harness to wasm, runs wasm-bindgen (web target) and wasm-opt -O4 like
# the shipped build. Default is single-threaded; `mt` uses the repo's shared-memory
# rustflags and compiles Lazy SMP in, exactly as `wasm-pack build` ships it.
# WASM_BINDGEN / WASM_OPT override the tool paths (the CLI must match wasm-bindgen 0.2.126).
# PROFILE=1 keeps function names so `node --cpu-prof` attributes wasm frames.
set -euo pipefail
# wasm-pack caches the tools it downloads; use its wasm-bindgen that matches the lock.
cache="${LOCALAPPDATA:-$HOME/.cache}/.wasm-pack"
if [ -z "${WASM_BINDGEN:-}" ]; then
  for b in "$cache"/wasm-bindgen-*/wasm-bindgen*; do
    case "$b" in *test-runner*) continue ;; esac
    if [ -x "$b" ] && "$b" --version 2>/dev/null | grep -q " 0.2.126$"; then
      WASM_BINDGEN="$b"
      break
    fi
  done
fi
if [ -z "${WASM_OPT:-}" ]; then
  for b in "$cache"/wasm-opt-*/bin/wasm-opt*; do
    [ -x "$b" ] && WASM_OPT="$b" && break
  done
fi
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "$1" && cd "$1" && pwd)"
cd "$here"
# Resolve dependencies exactly as the engine's own build does.
if [ ! -f Cargo.lock ] && [ -f ../Cargo.lock ]; then
  cp ../Cargo.lock Cargo.lock
fi
if [ "${2:-}" = "mt" ]; then
  cargo build --release --target wasm32-unknown-unknown --features mt
else
  RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
fi
bindgen_flags=(--target web)
opt_flags=(-O4)
if [ "${PROFILE:-}" = 1 ]; then
  bindgen_flags+=(--keep-debug)
  opt_flags+=(-g)
fi
"${WASM_BINDGEN:-wasm-bindgen}" "${bindgen_flags[@]}" --out-dir "$out" \
  target/wasm32-unknown-unknown/release/apeiron_wasm_harness.wasm
"${WASM_OPT:-wasm-opt}" "${opt_flags[@]}" --enable-simd --enable-threads --enable-bulk-memory \
  --enable-mutable-globals --enable-sign-ext --enable-nontrapping-float-to-int \
  --enable-reference-types --enable-multivalue \
  "$out/apeiron_wasm_harness_bg.wasm" -o "$out/apeiron_wasm_harness_bg.wasm"
echo "built $out"
