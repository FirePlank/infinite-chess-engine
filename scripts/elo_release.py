#!/usr/bin/env python3
"""Decide and apply automatic strength-based version bumps.

Scans commit messages since the last release tags, extracts each commit's
SPRT-reported Elo, scales it to a whole-engine estimate over the 17 canonical
site variants, and cuts a minor or major bump when the accumulated gain crosses
a threshold.

Bump rule (major is checked first):
  * cum_elo since last vX.0.0 tag   >= MAJOR_THRESHOLD  -> (X+1).0.0
  * cum_elo since last release tag   >= MINOR_THRESHOLD  -> X.(Y+1).0
  * otherwise                                            -> no release

Elo per commit is nominal (per-commit SPRT deltas do not add up to a real A/B
measurement), but variant coverage is detected per commit: only canonical
variants present in the breakdown are summed, then divided by 17. Negative
totals clamp to 0. A `Elo-Override: <n>` trailer forces a commit's value.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# The 17 site variants the engine is rated across. Anything else in a
# breakdown (Scattered_Leapers, Double_King_*, Triple_King_Maze, ...) is a
# test-only variant and does not count toward the whole-engine estimate.
CANONICAL_VARIANTS = {
    "Chess", "Classical", "Classical_Plus", "CoaIP", "CoaIP_HO", "CoaIP_NO",
    "CoaIP_RO", "Confined_Classical", "Core", "Knightline", "Obstocean",
    "Palace", "Pawn_Horde", "Pawndard", "Space", "Space_Classic", "Standarch",
}
NUM_SITE_VARIANTS = 17

MINOR_THRESHOLD = 30.0
MAJOR_THRESHOLD = 150.0

# A per-variant line: "  [Obstocean]: 503W - 287L - 580D, Elo: 55.2 +/- 7.2"
VARIANT_RE = re.compile(
    r"\[([A-Za-z_]+)\]:.*?Elo:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE
)
# A standalone overall figure, used only when no breakdown is present.
OVERALL_RE = re.compile(
    r"Elo(?:\s+Difference)?:\s*([-+]?\d+(?:\.\d+)?)\s*(?:\+/-|±)", re.IGNORECASE
)
OVERRIDE_RE = re.compile(r"^\s*Elo-Override:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)

SKIP_TOKEN = "[skip-release]"


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(
        args, cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def variant_blocks(message: str) -> list[list[tuple[str, float]]]:
    """Split a message into maximal runs of consecutive per-variant lines."""
    blocks: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    for line in message.splitlines():
        m = VARIANT_RE.search(line)
        if m:
            current.append((m.group(1), float(m.group(2))))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def scaled_elo(message: str) -> float:
    """Whole-engine Elo estimate for one commit message, clamped to >= 0."""
    override = OVERRIDE_RE.search(message)
    if override:
        return max(0.0, float(override.group(1)))

    blocks = variant_blocks(message)
    if blocks:
        # Use the last breakdown: for multi-test PRs it is the shipped state.
        # Sum only canonical variants, spread the gain across all 17.
        block = blocks[-1]
        total = sum(elo for name, elo in block if name in CANONICAL_VARIANTS)
        return max(0.0, total / NUM_SITE_VARIANTS)

    # No breakdown: fall back to a lone overall figure (rare, small in practice).
    overall = OVERALL_RE.search(message)
    if overall:
        return max(0.0, float(overall.group(1)))
    return 0.0


def commit_messages(rev_range: str, cwd: Path) -> list[str]:
    if not rev_range:
        return []
    out = run(
        ["git", "log", "--no-merges", "--format=%H%x1f%B%x1e", rev_range], cwd=cwd
    )
    if not out:
        return []
    records = [r for r in out.split("\x1e") if r.strip()]
    return [r.split("\x1f", 1)[1] for r in records if "\x1f" in r]


def cum_elo(rev_range: str, cwd: Path) -> float:
    return sum(scaled_elo(m) for m in commit_messages(rev_range, cwd))


def list_semver_tags(cwd: Path) -> list[tuple[int, int, int, str]]:
    try:
        raw = run(["git", "tag", "--list", "v*"], cwd=cwd)
    except subprocess.CalledProcessError:
        return []
    tags = []
    for tag in raw.splitlines():
        m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag.strip())
        if m:
            tags.append((int(m[1]), int(m[2]), int(m[3]), tag.strip()))
    return sorted(tags)


def read_cargo_version(cargo: Path) -> tuple[int, int, int]:
    text = cargo.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("could not find package version in Cargo.toml")
    return int(m[1]), int(m[2]), int(m[3])


def write_cargo_version(cargo: Path, version: str) -> None:
    text = cargo.read_text(encoding="utf-8")
    new, n = re.subn(
        r'^(version\s*=\s*)"\d+\.\d+\.\d+"',
        rf'\1"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit("failed to rewrite Cargo.toml version")
    cargo.write_text(new, encoding="utf-8")


def emit_output(key: str, value: str) -> None:
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report the decision without touching Cargo.toml or notes",
    )
    ap.add_argument("--notes-file", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root
    cargo = root / "Cargo.toml"
    major, minor, patch = read_cargo_version(cargo)

    if SKIP_TOKEN in run(["git", "log", "-1", "--format=%B"], cwd=root):
        print(f"HEAD carries {SKIP_TOKEN}; skipping.")
        emit_output("release", "false")
        return 0

    tags = list_semver_tags(root)
    if not tags:
        print(
            "No release tags found. Bootstrap by tagging the baseline, e.g. "
            f"git tag v{major}.{minor}.{patch} && git push --tags"
        )
        emit_output("release", "false")
        return 0
    last_release_ref = tags[-1][3]
    majors = [t for t in tags if t[1] == 0 and t[2] == 0]
    last_major_ref = majors[-1][3] if majors else tags[0][3]

    cum_major = cum_elo(f"{last_major_ref}..HEAD", root)
    cum_minor = cum_elo(f"{last_release_ref}..HEAD", root)

    print(f"current version : {major}.{minor}.{patch}")
    print(f"since major {last_major_ref}: {cum_major:.1f} Elo (threshold {MAJOR_THRESHOLD:.0f})")
    print(f"since release {last_release_ref}: {cum_minor:.1f} Elo (threshold {MINOR_THRESHOLD:.0f})")

    if cum_major >= MAJOR_THRESHOLD:
        level = "major"
        new_version = f"{major + 1}.0.0"
    elif cum_minor >= MINOR_THRESHOLD:
        level = "minor"
        new_version = f"{major}.{minor + 1}.0"
    else:
        print("no threshold reached; no release")
        emit_output("release", "false")
        return 0

    print(f"-> {level} bump: {new_version}")

    subjects = run(
        ["git", "log", "--no-merges", "--format=- %s", f"{last_release_ref}..HEAD"],
        cwd=root,
    )
    gain = cum_major if level == "major" else cum_minor
    notes = (
        f"## v{new_version} ({level} release)\n\n"
        f"Estimated nominal strength gain since v{major}.{minor}.{patch}: "
        f"**+{gain:.0f} Elo** (aggregated across the 17 site variants).\n\n"
        f"### Changes\n{subjects}\n"
    )

    emit_output("release", "true")
    emit_output("level", level)
    emit_output("version", new_version)
    emit_output("gain", f"{gain:.0f}")

    if args.dry_run:
        print("\n--- dry run: notes ---\n" + notes)
        return 0

    write_cargo_version(cargo, new_version)
    notes_path = args.notes_file or (root / "RELEASE_NOTES.md")
    notes_path.write_text(notes, encoding="utf-8")
    print(f"wrote {cargo} and {notes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
