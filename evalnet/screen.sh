#!/usr/bin/env bash
# Offline pre-filter for an HCE change: a reduced export + short base training, several
# seeds trained together, scored on the fresh self-play holdout. Run it for HEAD and for
# the change with the same seeds and compare the mean losses (see evalnet/README.md).
#
#   bash evalnet/screen.sh <tag> <export_eval_features.exe> [seeds=3] [threads=2]
#
# EXTRA passes more exporter arguments, e.g. EXTRA="--param rook=650" with an exporter
# built with --features data_gen,eval_tuning, to screen parameter values without a rebuild.
set -e
TAG=$1; EXE=$2; SEEDS=${3:-3}; THREADS=${4:-2}
D=evalnet/screen; mkdir -p "$D"
"$EXE" --threads "$THREADS" $EXTRA --min-ply 0 --texel games/texel_corpus.jsonl --texel evalnet/fresh_train.jsonl \
    --out "$D/$TAG.bin" > "$D/export_$TAG.log" 2>&1
"$EXE" --threads "$THREADS" $EXTRA --texel evalnet/fresh_holdout.jsonl --out "$D/${TAG}_ho.bin" > /dev/null 2>&1
python -u evalnet/train_seeds.py --perspective --hidden 128 --hidden2 64 --cap 500 \
    --data "$D/$TAG.bin" --epochs 40 --qat-from 30 --seeds "$(seq -s, 1 "$SEEDS")" \
    --out "$D/${TAG}_s{s}.pt" > "$D/train_$TAG.log" 2>&1
for s in $(seq 1 "$SEEDS"); do
    echo "$TAG s$s $(python evalnet/train_eval_net.py --data "$D/${TAG}_ho.bin" --lambda-texel 0.7 \
        --eval-only "$D/${TAG}_s$s.pt" 2>&1 | grep -v Warn | sed 's/.*: //')" | tee -a "$D/results.txt"
done
rm -f "$D/$TAG.bin"
