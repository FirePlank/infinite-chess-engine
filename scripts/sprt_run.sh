#!/usr/bin/env bash
# SPRT on GitHub runners: commits `test: <name>` (= <base> + patch + .github/sprt.json) on the
# one `sprt` branch, sums the shards' live pair counts, cancels the run once the LLR
# crosses a bound, and saves games + Final Summary under games/sprt. Record the decision
# afterwards with scripts/sprt_done.sh.
#   scripts/sprt_run.sh <name> <base-sha> <patch-file|-> <games> [variants] [elo0] [elo1] [tc]
#   Env: SHARDS (default 40); ADJ=0 turns max-ply adjudication off, which any change to
#   eval magnitude needs, since the 1000 cp threshold reads each engine's own score.
set -euo pipefail
NAME=$1 BASE=$2 PATCH=$3 GAMES=$4 VARIANTS=${5:-site} E0=${6:-0} E1=${7:-5} TC=${8:-10+0.1}
R=$(git rev-parse --show-toplevel); W="$R/../ice-sprt-branch"
# A generic-evaluator-only patch cannot change Chess, Obstocean or Pawn_Horde (their own
# evaluators), so their games are noise: force the default preset, whatever was asked.
if [ "$PATCH" != "-" ] && [ "$(python "$R/scripts/sprt_scope.py" "$PATCH")" = generic ] && [ "$VARIANTS" = site ]; then
  VARIANTS=Classical,Confined_Classical,Classical_Plus,Core,CoaIP,CoaIP_HO,CoaIP_RO,CoaIP_NO,Palace,Pawndard,Standarch,Space_Classic,Space,Knightline,Scattered_Leapers
  echo "generic-evaluator patch: variants set to the default preset (no Chess/Obstocean/Pawn_Horde)"
fi
# GitHub calls retry for ~20 min, so a network drop cannot abort the watch.
ghr() { local o; for _ in $(seq 40); do o=$("$@") && { echo "$o"; return 0; }; sleep 30; done; return 1; }
REPO=$(ghr gh repo view --json nameWithOwner -q .nameWithOwner)
# RUN_ID=<id> reattaches to a run already pushed (after a lost watch): no commit, no push.
if [ -n "${RUN_ID:-}" ]; then
  ID=$RUN_ID; SHA=$(ghr gh run view "$ID" --json headSha -q .headSha)
else
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
      --argjson g "$GAMES" --argjson e0 "$E0" --argjson e1 "$E1" --argjson sh "${SHARDS:-40}" \
      --argjson adj "${ADJ:-1000}" \
      '{name:$n, old:$o, games:$g, shards:$sh, variants:$v, tc:$tc, elo0:$e0, elo1:$e1, adjudication:$adj}' > .github/sprt.json
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
fi
echo "run $ID  https://github.com/$REPO/actions/runs/$ID"
# Each shard posts its pair counts (ll,ld,wl+dd,wd,ww) as a commit status every few
# minutes; sum the newest per shard and stop the run once the LLR crosses a bound.
# `gh run watch` is avoided: it can hang without a terminal.
# A shard whose count stays unchanged while still pending is stuck (the harness saves
# every 10 games, minutes apart): cancel, so one shard cannot hold a test for hours.
STALL_S=1200
declare -A LAST_SEEN LAST_CHANGE
while :; do
  STATUS=$(ghr gh run view "$ID" --json status -q .status)
  SUM=$(gh api "repos/$REPO/commits/$SHA/status?per_page=100"           -q '[.statuses[] | select(.context | startswith("sprt/shard-"))] | group_by(.context)
              | map(max_by(.updated_at).description | split(",") | map(tonumber))
              | if length == 0 then [0,0,0,0,0] else transpose | map(add) end | map(tostring) | join(",")'           2> /dev/null || echo "0,0,0,0,0")
  OUT=$(python "$R/scripts/sprt_merge.py" --from-counts "$SUM" --elo0 "$E0" --elo1 "$E1")
  NOW=$(date +%s); STUCK=""
  while IFS='|' read -r CTX STATE DESC; do
    [ -n "$CTX" ] || continue
    if [ "${LAST_SEEN[$CTX]:-}" != "$DESC" ]; then LAST_SEEN[$CTX]=$DESC; LAST_CHANGE[$CTX]=$NOW; fi
    if [ "$STATE" = pending ] && [ $(( NOW - ${LAST_CHANGE[$CTX]} )) -ge $STALL_S ]; then STUCK="$STUCK $CTX"; fi
  done < <(gh api "repos/$REPO/commits/$SHA/status?per_page=100"              -q '[.statuses[] | select(.context | startswith("sprt/shard-"))] | group_by(.context)
                 | map(max_by(.updated_at)) | .[] | "\(.context)|\(.state)|\(.description)"' 2> /dev/null)
  if [ -n "$STUCK" ]; then
    ghr gh api -X POST "repos/$REPO/actions/runs/$ID/cancel" > /dev/null
    echo "stalled for $((STALL_S / 60)) min:$STUCK; run cancelled ($(grep '^pairs=' <<< "$OUT"))"
    until [ "$(ghr gh run view "$ID" --json status -q .status)" = completed ]; do sleep 15; done
    break
  fi
  if grep -q '^stop=true' <<< "$OUT"; then
    ghr gh api -X POST "repos/$REPO/actions/runs/$ID/cancel" > /dev/null
    echo "LLR bound crossed ($(grep '^llr=' <<< "$OUT"), $(grep '^pairs=' <<< "$OUT")): run cancelled"
    until [ "$(ghr gh run view "$ID" --json status -q .status)" = completed ]; do sleep 15; done
    break
  fi
  [ "$STATUS" = completed ] && break
  sleep 60
done
# Keep every game, stopped early or not: one JSON per test under games/sprt.
# A folder per watcher process: a second watcher on the same run (a lost session's, say)
# would otherwise extract into the same files ("The file exists") and wipe them.
T="$R/games/sprt/.remote_${NAME}_$$"; rm -rf "$T"
until gh run download "$ID" -p "shard-*" -D "$T"; do
  echo "shard download failed; retrying in 60s" >&2; rm -rf "$T"; sleep 60
done
python "$R/scripts/sprt_merge.py" --label "$NAME" --old "$BASE" --elo0 "$E0" --elo1 "$E1"   --out "$R/games/sprt/games_${NAME}_remote.json" "$T"/shard-*/shard_*.json   | tee "$R/games/sprt/summary_${NAME}_remote.txt"
rm -rf "$T"
