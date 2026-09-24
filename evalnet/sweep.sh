#!/usr/bin/env bash
# Screen every arm of a sweep file (lines "tag|exporter args"): bash evalnet/sweep.sh <file> <exporter> [seeds]
# Arms already in evalnet/screen/results.txt are skipped, so a stopped sweep resumes.
cd "$(dirname "$0")/.."
while IFS='|' read -r tag args; do
    args=${args%$'\r'}
    [ -z "$tag" ] && continue
    grep -q "^$tag s1 " evalnet/screen/results.txt 2>/dev/null && continue
    EXTRA="$args" bash evalnet/screen.sh "$tag" "$2" "${3:-3}" 2
done < "$1"
