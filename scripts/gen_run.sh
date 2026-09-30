#!/usr/bin/env bash
# Self-play data generation on GitHub runners: commits `gen: <name>` (= HEAD + an optional
# patch + .github/gen.json) on the `gen` branch, waits for the
# run, and concatenates every shard's JSONL into evalnet/gen/<name>.jsonl.
#   scripts/gen_run.sh <name> <games> [nodes] [variants] [seed] [shards] [patch]
# 40 shards fill GitHub Pro's concurrent-job limit; use fewer while an SPRT is running.
set -euo pipefail
NAME=$1 GAMES=$2 NODES=${3:-100000} VARIANTS=${4:-site} SEED=${5:-1} SHARDS=${6:-40}
PATCH=${7:--}
R=$(git rev-parse --show-toplevel); W="$R/../ice-gen-branch"
ghr() { local o; for _ in $(seq 40); do o=$("$@") && { echo "$o"; return 0; }; sleep 30; done; return 1; }
REPO=$(ghr gh repo view --json nameWithOwner -q .nameWithOwner)
BASE=$(git -C "$R" rev-parse HEAD)
if [ -n "${RUN_ID:-}" ]; then
  ID=$RUN_ID
else
  [ -d "$W" ] || git -C "$R" worktree add -q --detach "$W" "$BASE"
  cd "$W"
  git checkout -q -B gen "$BASE"
  [ "$PATCH" = "-" ] || git apply "$PATCH"
  jq -n --arg n "$NAME" --arg v "$VARIANTS" --argjson g "$GAMES" --argjson nodes "$NODES" --argjson s "$SEED" \
        --argjson sh "$SHARDS" '{name:$n, games:$g, nodes:$nodes, variants:$v, seed:$s, shards:$sh, minutes:330}' > .github/gen.json
  git add -A && git commit -q -m "gen: $NAME"
  SHA=$(git rev-parse HEAD)
  git push -q -f origin gen
  ID=""
  for _ in $(seq 60); do
    ID=$(gh run list --branch gen --limit 10 --json databaseId,headSha,workflowName \
         -q ".[] | select(.headSha==\"$SHA\" and .workflowName==\"Gen remote\") | .databaseId" | head -1)
    [ -n "$ID" ] && break; sleep 5
  done
fi
echo "run $ID  https://github.com/$REPO/actions/runs/$ID"
until [ "$(ghr gh run view "$ID" --json status -q .status)" = completed ]; do sleep 120; done
# A folder per watcher process, as in sprt_run.sh.
T="$R/evalnet/gen/.remote_${NAME}_$$"; rm -rf "$T"
until gh run download "$ID" -p "shard-*" -D "$T"; do
  echo "shard download failed; retrying in 60s" >&2; rm -rf "$T"; sleep 60
done
cat "$T"/shard-*/shard_*.jsonl > "$R/evalnet/gen/$NAME.jsonl"
echo "$(ls "$T" | wc -l) shards, $(wc -l < "$R/evalnet/gen/$NAME.jsonl") games -> evalnet/gen/$NAME.jsonl"
rm -rf "$T"
