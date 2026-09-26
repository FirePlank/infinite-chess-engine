#!/usr/bin/env bash
# SPRT on GitHub runners: commits `test: <name>` (= <base> + patch + .github/sprt.json) on the
# one `sprt` branch, sums the shards' live pair counts, cancels the run once the LLR
# crosses a bound, and saves games + Final Summary under games/sprt. Record the decision
# afterwards with scripts/sprt_done.sh.
#   scripts/sprt_run.sh <name> <base-sha> <patch-file|-> <games> [variants] [elo0] [elo1] [tc]
set -euo pipefail
NAME=$1 BASE=$2 PATCH=$3 GAMES=$4 VARIANTS=${5:-site} E0=${6:-0} E1=${7:-5} TC=${8:-10+0.1}
R=$(git rev-parse --show-toplevel); W="$R/../ice-sprt-branch"
git -C "$R" fetch -q origin
if [ ! -d "$W" ]; then
  if git -C "$R" ls-remote --exit-code --heads origin sprt > /dev/null; then
    git -C "$R" worktree add -q -B sprt "$W" origin/sprt
  else
    git -C "$R" worktree add -q -b sprt "$W" "$BASE"
  fi
fi
cd "$W"
git ls-remote --exit-code --heads origin sprt > /dev/null && git reset -q --hard origin/sprt
git merge -q --no-edit -m "merge main into sprt" "$BASE"
# The branch must be exactly main plus the test config, or results would be skewed.
if ! git diff --quiet "$BASE" HEAD -- . ':!.github/sprt.json'; then
  echo "sprt branch differs from $BASE outside .github/sprt.json; record or revert the last test first" >&2
  exit 1
fi
[ "$PATCH" = "-" ] || git apply "$PATCH"
jq -n --arg n "$NAME" --arg o "$BASE" --arg v "$VARIANTS" --arg tc "$TC" \
      --argjson g "$GAMES" --argjson e0 "$E0" --argjson e1 "$E1" \
      '{name:$n, old:$o, games:$g, shards:20, variants:$v, tc:$tc, elo0:$e0, elo1:$e1}' > .github/sprt.json
key() {
  { git ls-tree -r "$1" -- src Cargo.toml Cargo.lock build.rs .cargo/config.toml rust-toolchain.toml \
      | grep -v $'\tsrc/bin/'
    git ls-tree "$1" -- src/bin/sprt.rs; } | sha256sum | cut -c1-24
}
git add -A && git commit -q -m "test: $NAME"
if [ "$(key HEAD)" = "$(key "$BASE")" ]; then
  echo "null test: the patch leaves the engine source identical to $BASE; not pushing" >&2
  git reset -q --hard HEAD~1; exit 1
fi
SHA=$(git rev-parse HEAD)
git push -q origin sprt
ID=""
for _ in $(seq 60); do
  ID=$(gh run list --branch sprt --limit 10 --json databaseId,headSha,workflowName \
       -q ".[] | select(.headSha==\"$SHA\" and .workflowName==\"SPRT remote\") | .databaseId" | head -1)
  [ -n "$ID" ] && break; sleep 5
done
echo "run $ID  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions/runs/$ID"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
# Each shard posts its pair counts (ll,ld,wl+dd,wd,ww) as a commit status every few
# minutes; sum the newest per shard and stop the run once the LLR crosses a bound.
# `gh run watch` is avoided: it can hang without a terminal.
while :; do
  STATUS=$(gh run view "$ID" --json status -q .status)
  SUM=$(gh api "repos/$REPO/commits/$SHA/statuses?per_page=100"           -q '[.[] | select(.context | startswith("sprt/shard-"))] | group_by(.context)
              | map(max_by(.updated_at).description | split(",") | map(tonumber))
              | if length == 0 then [0,0,0,0,0] else transpose | map(add) end | map(tostring) | join(",")'           2> /dev/null || echo "0,0,0,0,0")
  OUT=$(python "$R/scripts/sprt_merge.py" --from-counts "$SUM" --elo0 "$E0" --elo1 "$E1")
  if grep -q '^stop=true' <<< "$OUT"; then
    gh api -X POST "repos/$REPO/actions/runs/$ID/cancel" > /dev/null
    echo "LLR bound crossed ($(grep '^llr=' <<< "$OUT"), $(grep '^pairs=' <<< "$OUT")): run cancelled"
    until [ "$(gh run view "$ID" --json status -q .status)" = completed ]; do sleep 15; done
    break
  fi
  [ "$STATUS" = completed ] && break
  sleep 60
done
# Keep every game, stopped early or not: one JSON per test under games/sprt.
T="$R/games/sprt/.remote_$NAME"; rm -rf "$T"
gh run download "$ID" -p "shard-*" -D "$T"
python "$R/scripts/sprt_merge.py" --label "$NAME" --old "$BASE" --elo0 "$E0" --elo1 "$E1"   --out "$R/games/sprt/games_${NAME}_remote.json" "$T"/shard-*/shard_*.json   | tee "$R/games/sprt/summary_${NAME}_remote.txt"
rm -rf "$T"
