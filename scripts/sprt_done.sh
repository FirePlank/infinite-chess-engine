#!/usr/bin/env bash
# Record a finished test on the sprt branch: `reject` reverts its commit, `accept` leaves it
# (the change lands on main separately). Either way the record carries the result summary.
#   scripts/sprt_done.sh <name> accept|reject [summary-file]
set -euo pipefail
NAME=$1 VERDICT=$2
R=$(git rev-parse --show-toplevel); W="$R/../ice-sprt-branch"
SUMMARY=${3:-"$R/games/sprt/summary_${NAME}_remote.txt"}
cd "$W"
git fetch -q origin && git reset -q --hard origin/sprt
C=$(git log --format=%H --grep="^test: $NAME\$" -1)
[ -n "$C" ] || { echo "no 'test: $NAME' commit on sprt" >&2; exit 1; }
MSG=$(mktemp); { echo "$VERDICT: $NAME"; echo; [ -f "$SUMMARY" ] && cat "$SUMMARY"; } > "$MSG"
if [ "$VERDICT" = reject ]; then
  git revert -q --no-edit "$C" && git commit -q --amend -F "$MSG"
else
  git commit -q --allow-empty -F "$MSG"
fi
rm -f "$MSG"
git push -q origin sprt
echo "recorded $VERDICT: $NAME"
