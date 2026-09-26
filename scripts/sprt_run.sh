#!/usr/bin/env bash
# SPRT on GitHub runners: commits `test: <name>` (= <base> + patch + .github/sprt.json) on the
# one `sprt` branch, merges the shards' games as they upload, cancels the run once the LLR
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
T="$R/games/sprt/.remote_$NAME"; rm -rf "$T"; mkdir -p "$T"
declare -A SEEN
fetch() {
  gh api "repos/$REPO/actions/runs/$ID/artifacts?per_page=100"     -q '.artifacts[] | select(.name | startswith("shard-")) | "\(.id) \(.name) \(.updated_at)"' > "$T/list" || return 0
  while read -r aid aname aupd; do
    [ "${SEEN[$aname]:-}" = "$aupd" ] && continue
    gh api "repos/$REPO/actions/artifacts/$aid/zip" > "$T/z.zip"       && rm -rf "$T/$aname" && unzip -q -o "$T/z.zip" -d "$T/$aname" && SEEN[$aname]=$aupd
  done < "$T/list"
}
merge() {
  python "$R/scripts/sprt_merge.py" --label "$NAME" --old "$BASE" --elo0 "$E0" --elo1 "$E1" "$@"     --out "$R/games/sprt/games_${NAME}_remote.json" "$T"/shard-*/shard_*.json
}
# Shards upload after each third of their games; stop the run as soon as the aggregate
# LLR crosses a bound. `gh run watch` is avoided: it can hang without a terminal.
while :; do
  STATUS=$(gh run view "$ID" --json status -q .status)
  fetch
  if compgen -G "$T/shard-*/shard_*.json" > /dev/null; then
    rm -f "$T/out"; merge --gh-output "$T/out" > /dev/null
    if grep -q '^stop=true' "$T/out"; then
      gh api -X POST "repos/$REPO/actions/runs/$ID/cancel" > /dev/null
      echo "LLR bound crossed ($(grep '^llr=' "$T/out")): run cancelled"; break
    fi
  fi
  [ "$STATUS" = completed ] && break
  sleep 60
done
fetch
merge | tee "$R/games/sprt/summary_${NAME}_remote.txt"
rm -rf "$T"
