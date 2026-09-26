#!/usr/bin/env bash
# SPRT on GitHub runners: pushes sprt/<name> = <base> + patch + .github/sprt.json, waits for
# the sharded run, and saves its merged games and Final Summary under games/sprt/remote_<name>.
#   scripts/sprt_remote.sh <name> <base-sha> <patch-file|-> <games> [variants] [elo0] [elo1] [tc]
set -euo pipefail
NAME=$1 BASE=$2 PATCH=$3 GAMES=$4 VARIANTS=${5:-site} E0=${6:-0} E1=${7:-5} TC=${8:-10+0.1}
R=$(git rev-parse --show-toplevel); W="$R/../ice-remote"
[ -d "$W" ] || git -C "$R" worktree add -q --detach "$W" "$BASE"
cd "$W"
git checkout -q --detach "$BASE" && git reset -q --hard "$BASE"
[ "$PATCH" = "-" ] || git apply "$PATCH"
jq -n --arg n "$NAME" --arg o "$BASE" --arg v "$VARIANTS" --arg tc "$TC" \
      --argjson g "$GAMES" --argjson e0 "$E0" --argjson e1 "$E1" \
      '{name:$n, old:$o, games:$g, shards:20, variants:$v, tc:$tc, elo0:$e0, elo1:$e1}' > .github/sprt.json
git add -A && git commit -q -m "sprt: $NAME"
SHA=$(git rev-parse HEAD)
git push -f -q origin "HEAD:refs/heads/sprt/$NAME"
ID=""
for _ in $(seq 60); do
  ID=$(gh run list --branch "sprt/$NAME" --workflow sprt-remote.yml --limit 5 --json databaseId,headSha \
       -q ".[] | select(.headSha==\"$SHA\") | .databaseId" | head -1)
  [ -n "$ID" ] && break; sleep 5
done
echo "run $ID  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions/runs/$ID"
gh run watch "$ID" --interval 30 > /dev/null || true
OUT="$R/games/sprt/remote_$NAME"; rm -rf "$OUT"
gh run download "$ID" -n merged -D "$OUT"
cat "$OUT/summary.txt"
