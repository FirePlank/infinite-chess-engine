#!/usr/bin/env python3
"""Variant scope an SPRT patch needs: prints "generic" when every change sits in the
generic evaluator (base.rs, the generic net, or eval params no specialized evaluator
reads), else "site". Chess, Obstocean and Pawn_Horde have their own evaluators, so their
games are noise for a generic-only change.

    python scripts/sprt_scope.py <patch>
"""
import pathlib
import re
import sys

GENERIC_FILES = {
    "src/evaluation/base.rs",
    "src/eval_net/eval_net.bin",
    "src/eval_net/features.rs",
    "src/eval_net/inference.rs",
    "src/eval_net/mod.rs",
    "src/eval_net/weights.rs",
}
ENGINE_BUILD_FILES = {"Cargo.toml", "Cargo.lock", "build.rs", ".cargo/config.toml", "rust-toolchain.toml"}
PARAMS = "src/evaluation/params.rs"
# Files outside the generic evaluator that read eval params.
OTHER_READERS = ["src/evaluation/mod.rs", "src/evaluation/piece_reach.rs", "src/evaluation/variants",
                 "src/search.rs", "src/search", "src/lib.rs", "src/game.rs", "src/moves.rs"]


def changed_params(patch_text):
    names = set()
    in_params = False
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            in_params = line.endswith(" b/" + PARAMS)
        elif in_params and line[:1] in "+-" and not line.startswith(("+++", "---")):
            m = re.match(r"[+-]\s+(\w+)\s*(?::\s*\w+)?\s*=", line)
            if m:
                names.add(m.group(1))
    return names


def read_elsewhere(name, root):
    pat = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    for rel in OTHER_READERS:
        p = root / rel
        files = [p] if p.is_file() else sorted(p.rglob("*.rs")) if p.is_dir() else []
        for f in files:
            if pat.search(f.read_text(encoding="utf-8", errors="replace")):
                return True
    return False


def main():
    patch = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    root = pathlib.Path(__file__).resolve().parent.parent
    files = set(re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M))
    for f in files:
        # Only src/ and the build files reach the engine; trainer scripts and tool bins
        # (src/bin/ other than the harness, as sprt_run.sh's build key) do not.
        tool_bin = f.startswith("src/bin/") and f != "src/bin/sprt.rs"
        if f in GENERIC_FILES or tool_bin or not (f.startswith("src/") or f in ENGINE_BUILD_FILES):
            continue
        if f == PARAMS and not any(read_elsewhere(n, root) for n in changed_params(patch)):
            continue
        print("site")
        return
    print("generic")


if __name__ == "__main__":
    main()
