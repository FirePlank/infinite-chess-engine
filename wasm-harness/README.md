# wasm harness

Runs the engine compiled to wasm (the build the site ships) under Node, so wasm
speed can be measured and wasm-only code paths (the simd128 net kernel) checked.

It exports nps_bench's fixed-depth search sweep and eval_bench's position corpus.
A behaviour-identical change must leave both results unchanged between two builds.
Wasm search differs slightly from native (FxHash on a 32-bit `usize` changes
hash-set iteration order), so compare wasm builds with each other. Eval sums
match native exactly: `cargo test --release -- --ignored --nocapture` in this
directory prints the native reference.

```bash
./build.sh pkg-new mt        # shipped config: shared memory + Lazy SMP compiled in
./build.sh pkg-st            # single-threaded simd128 build
PROFILE=1 ./build.sh pkg-prof mt   # keeps names for node --cpu-prof

node run.mjs search 10 12 pkg-old pkg-new   # interleaved A/B, depth 10, 12 pairs
node run.mjs eval 20 12 pkg-old pkg-new     # 20 corpus passes per run
NPS_ONLY=CoaIP_RO node run.mjs search 12 8 pkg-old pkg-new
node --cpu-prof --cpu-prof-dir=prof run.mjs search 10 1 pkg-prof
```

Each run first calls `net_selftest()`, which checks the net's SIMD dense layer
against its scalar reference inside the build being measured.

`build.sh` finds the wasm-bindgen CLI matching the lockfile (0.2.126) and
wasm-opt in wasm-pack's download cache; set `WASM_BINDGEN` / `WASM_OPT` to
override. On first build it copies the engine's `Cargo.lock` so dependency
versions match the shipped build.
