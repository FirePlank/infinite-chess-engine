# Eval-net training pipeline

Training tooling for the hybrid evaluation net: a small quantized MLP over
scalars the hand-crafted evaluation already computes, adding a capped residual
to the Generic evaluator. Design: `docs/hybrid-eval-design.md`, results log:
`docs/eval-net-plan.md`. Runtime code: `src/eval_net/`.

## Current recipe

129 inputs -> 128 -> 64 -> 1, residual capped at 500 cp, inputs read as
(side to move, opponent). Two stages: a base run on a large mixed set, then a
fine-tune on positions relabelled by a depth-9 search.

```
cargo build --release --features data_gen --bin export_eval_features
X=./target/release/export_eval_features.exe

# base set: texel corpora plus a 15% sample of every SPRT archive, from ply 0
$X --min-ply 0 --texel games/texel_corpus.jsonl --texel nnue/fresh_train.jsonl \
   --sprt-dir games/sprt --sprt-sample 0.15 --out nnue/mix.bin
# fine-tune set: ~1M archive positions searched to depth 9 (about 4 h on 16 threads)
$X --sprt-dir games/sprt --sprt-sample 0.016 --relabel-depth 9 --out nnue/rel.bin
python nnue/merge_data.py nnue/mixrel.bin nnue/mix.bin nnue/rel.bin

T="python nnue/train_eval_net.py --perspective --keep-cloud --n-cols 129 --hidden 128 --hidden2 64 --cap 500"
$T --data nnue/mixrel.bin --epochs 60 --out nnue/checkpoints/base.pt
$T --data nnue/rel.bin --init nnue/checkpoints/base.pt --epochs 20 --lr 2e-4 \
   --qat-from 1 --val-frac 0.1 --out nnue/checkpoints/net.pt
python nnue/export_eval_net.py --checkpoint nnue/checkpoints/net.pt \
   --data nnue/holdout.bin --out src/eval_net/eval_net.bin
```

## Judging a net

Score candidates on a fresh self-play holdout (`--texel nnue/fresh_holdout.jsonl`
export) with `train_eval_net.py --eval-only`. It ranks nets the way SPRT does;
splits of the training corpus and the human-game set do not. Only nets that
beat the incumbent there go to SPRT.

## After an HCE change

The net's inputs are HCE terms, so a changed eval needs a retrained net on both
sides of the test (see `docs/CONTRIBUTING.md`). Depth-9 labels are expensive and
the change alters every feature key, so carry them over by Zobrist hash
(`$OLD`/`$NEW` are exporters built without and with the change):

```
REL="--sprt-dir games/sprt --sprt-sample 1.0 --quiet-tolerance 100000"
$OLD $REL --keep-keys nnue/relabel_keys78.bin --key-columns 78 --hash-out old.hash --out old_relpos.bin
python nnue/hash_labels.py table nnue/relabel_d9.bin old_relpos.bin old.hash d9.table 78
$NEW $REL --keep-hashes d9.table.keep --hash-out new.hash --out new_relpos.bin
python nnue/hash_labels.py apply d9.table new_relpos.bin new.hash new_rel.bin
```

## Notes

- The exporter recomputes today's static eval and features at every kept
  position; recorded evals are only targets. It prints the teacher/static slope
  so a flipped eval sign is obvious.
- Validation is split by game, so correlated plies never leak. The exporter
  checks the integer forward pass against the float model.
- Positions where the engine switches the net off (bare-king mop-up) are
  dropped at export and again in the trainer.
- `--hidden`/`--cap` must match `src/eval_net/inference.rs`; `SCHEMA_VERSION` in
  `features.rs` seals the feature layout into the blob.

Data files (`*.bin`) and `checkpoints/` are ignored by git.
