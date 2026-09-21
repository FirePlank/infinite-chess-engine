# Eval-net training pipeline

Training tooling for the Stage-A hybrid evaluation net: a small quantized MLP
over scalars the hand-crafted evaluation already computes, adding a capped
residual to the Generic evaluator. Design and status: `docs/hybrid-eval-design.md`,
improvement plan: `docs/eval-net-plan.md`. Runtime code: `src/eval_net/`.

## Loop

```
cargo build --release --features data_gen --bin export_eval_features
./target/release/export_eval_features.exe --texel games/texel_corpus.jsonl \
    --sprt-dir games/sprt --sprt-sample 0.15 --out nnue/eval_net_data.bin
python nnue/train_eval_net.py --data nnue/eval_net_data.bin --epochs 30 --hidden 32 --cap 250
python nnue/export_eval_net.py --checkpoint nnue/checkpoints/eval_net.pt --out src/eval_net/eval_net.bin
cargo build --release
```

- The exporter recomputes today's static eval and feature vector at every kept
  position; recorded evals are only targets. It prints the teacher/static slope
  (~0.98) so a flipped eval sign is obvious.
- The trainer reports held-out loss against the zero-residual baseline, split by
  GAME so correlated plies never leak. Quantization-aware training starts at
  epoch 4; the exporter checks the integer forward pass against the float model.
- `--hidden` and `--cap` must match `src/eval_net/inference.rs` (`RESIDUAL_CAP`)
  and the loader's dimension checks; `SCHEMA_VERSION` in `features.rs` seals the
  feature layout into the blob.

Data files (`*.bin`) and `checkpoints/` are ignored by git.
