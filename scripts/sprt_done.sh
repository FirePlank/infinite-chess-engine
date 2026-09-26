#!/usr/bin/env bash
# Delete a finished test's sprt/* branches (after it is committed or rejected).
#   scripts/sprt_done.sh <name> [<name> ...]
set -uo pipefail
for n in "$@"; do
  git push -q origin --delete "sprt/$n" 2>/dev/null && echo "deleted sprt/$n" || echo "no sprt/$n on origin"
done
