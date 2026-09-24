# Eval-net training pipeline

Training tooling for the hybrid evaluation net: a small quantized MLP over
scalars the hand-crafted evaluation already computes, adding a capped residual
to the Generic evaluator. Design: `docs/hybrid-eval-design.md`, results log:
`docs/eval-net-plan.md`. Runtime code: `src/eval_net/`.

## Current recipe

121 inputs -> 128 -> 64 -> 1, residual capped at 500 cp, inputs read as
(side to move, opponent). Two stages: a base run on a large mixed set, then a
fine-tune on positions relabelled by a depth-9 search.
Base-run length was swept: 120 epochs beats 60/90 and 180/240 (which overfit);
weight EMA and SWA gave no gain over six seeds. Train several seeds, keep the best:
`evalnet/train_seeds.py --seeds 1,2,3,4,5,6` trains them together on one copy of the data
(same flags and checkpoints as `train_eval_net.py`, about 2.3x faster than one by one).

```
cargo build --release --features data_gen --bin export_eval_features
X=./target/release/export_eval_features.exe

# base set: texel corpora plus a 15% sample of every SPRT archive, from ply 0
$X --min-ply 0 --texel games/texel_corpus.jsonl --texel evalnet/fresh_train.jsonl \
   --sprt-dir games/sprt --sprt-sample 0.15 --out evalnet/mix.bin
# fine-tune set: ~1M archive positions searched to depth 9 (about 4 h on 16 threads)
$X --sprt-dir games/sprt --sprt-sample 0.016 --relabel-depth 9 --out evalnet/rel.bin
python evalnet/merge_data.py evalnet/mixrel.bin evalnet/mix.bin evalnet/rel.bin

T="python evalnet/train_eval_net.py --perspective --hidden 128 --hidden2 64 --cap 500"
$T --data evalnet/mixrel.bin --epochs 120 --out evalnet/checkpoints/base.pt
$T --data evalnet/rel.bin --init evalnet/checkpoints/base.pt --epochs 20 --lr 2e-4 \
   --qat-from 1 --val-frac 0.1 --out evalnet/checkpoints/net.pt
python evalnet/export_eval_net.py --checkpoint evalnet/checkpoints/net.pt \
   --data evalnet/holdout.bin --out src/eval_net/eval_net.bin
```

## Judging a net

Score candidates on a fresh self-play holdout (`--texel evalnet/fresh_holdout.jsonl`
export) with `train_eval_net.py --eval-only`. It ranks nets the way SPRT does;
splits of the training corpus and the human-game set do not. Only nets that
beat the incumbent there go to SPRT.

## Screening an HCE change

`evalnet/screen.sh` is the pre-filter: an export of the two self-play corpora and a
40-epoch base run per seed, scored on the fresh holdout, about a minute per seed.
Run it for HEAD and for the change with the same seeds and compare the mean losses:

```
bash evalnet/screen.sh head ./export_head.exe 3
bash evalnet/screen.sh change ./target/release/export_eval_features.exe 3
```

Single seeds differ by about 0.3%, so judge means over 3 or more. The screen ranks
variants of one idea well; as a verdict on a single change it is only trusted for
clear losses (about 1% or more worse). Closer calls go to SPRT: A6 was 0.1% worse
offline and won +20, and dropping the development term screened 0.45% worse and
won +13.

## After an HCE change

The net's inputs are HCE terms, so a changed eval needs its own retrained net,
tested against HEAD as committed (see `docs/CONTRIBUTING.md`). Depth-9 labels are
expensive and the change alters every feature key, so carry them over by Zobrist
hash (`$OLD`/`$NEW` are exporters built without and with the change):

```
# once: map the labelled positions to their hashes (d9.table is reused afterwards)
$OLD --sprt-dir games/sprt --sprt-sample 1.0 --quiet-tolerance 100000      --keep-keys evalnet/relabel_keys78.bin --key-columns 78 --hash-out old.hash --out old_relpos.bin
python evalnet/hash_labels.py table evalnet/relabel_d9.bin old_relpos.bin old.hash d9.table 78

# per change: both training sets from one replay of the corpora
$NEW --min-ply 0 --texel games/texel_corpus.jsonl --texel evalnet/fresh_train.jsonl      --sprt-dir games/sprt --sprt-sample 0.15 --out mix.bin      --rel-out relpos.bin --rel-keep-hashes d9.table.keep --rel-hash-out relpos.hash
python evalnet/hash_labels.py apply d9.table relpos.bin relpos.hash rel.bin
```

## Specialized-evaluator nets

Chess, Obstocean and Pawn Horde each train on their evaluator's own layout
(`NET_LAYOUT` in its `variants/` file). `--layout fixed,neg` tells the trainer
how that layout mirrors: fixed columns, then White-ahead singles, then pairs.
Pawn Horde has no colour mirror, so it trains without `--perspective`.

```
$X --eval-kind chess --min-ply 0 --sprt-dir games/sprt --sprt-sample 1.0 --out chess.bin
python evalnet/train_seeds.py --perspective --layout 1,0 --hidden 128 --hidden2 64 --cap 500 \
   --data chess.bin --seeds 1,2,3 --epochs 120 --out chess_s{s}.pt
```

Layouts: Chess `1,0`, Obstocean `1,3`, Pawn Horde `20,0`. To see which generic
inputs a variant net would use, export with `--generic-features` and rank the
input groups by permutation importance.

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
