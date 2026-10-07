//! Builds `src/evaluation/insuffmat_unbounded.bin` from infinitechess.org's
//! insufficient-material tables (`mates-N.tsv` smallest mating sets and
//! `draws-N.txt` dead sets, labels like `K,B0 vs k,n`, mirror images listed once).
//!
//! Every set is checked to be exactly one of: a listed draw, or a superset of a
//! listed mate. Listed mates must be smallest (no listed mate inside another).
//! The optional helpmate-only list (`forced_gen`'s `escape.txt`: White's army vs
//! Black's royals, labels like `K,GU,GU vs k`) is appended as a one-probe hash of set indices.
//!
//! Usage: cargo run --release --bin insuffmat_build -- <tables dir> [helpmate-only list] [out file]

use apeiron::evaluation::mating_sets::{
    self, CAP, KIND_CODES, KINDS, SET_COUNT, SYMBOLS, for_each_set, set_index,
};
use std::collections::HashSet;

fn parse(label: &str) -> Vec<u8> {
    let label = label.split(" @").next().unwrap();
    let (white, black) = label
        .split_once(" vs")
        .unwrap_or_else(|| panic!("no ' vs' in {label:?}"));
    let mut set = Vec::new();
    for (color, side) in [(0u8, white), (1u8, black)] {
        for code in side.trim().split(',').filter(|c| !c.is_empty()) {
            let kind = KIND_CODES
                .iter()
                .position(|k| k.eq_ignore_ascii_case(code))
                .unwrap_or_else(|| panic!("unknown piece {code:?} in {label:?}"));
            set.push(color * KINDS + kind as u8);
        }
    }
    set
}

/// The set and its mirror images: colors swapped, bishop colors swapped, or both.
fn orientations(set: &[u8]) -> [Vec<u8>; 4] {
    let swap_colors = |s: u8| (s + KINDS) % SYMBOLS;
    let swap_bishops = |s: u8| match s % KINDS {
        5 => s + 1,
        6 => s - 1,
        _ => s,
    };
    [
        set.to_vec(),
        set.iter().map(|&s| swap_colors(s)).collect(),
        set.iter().map(|&s| swap_bishops(s)).collect(),
        set.iter().map(|&s| swap_bishops(swap_colors(s))).collect(),
    ]
}

fn read(dir: &str, prefix: &str, ext: &str) -> (HashSet<usize>, usize) {
    let mut out = HashSet::new();
    let mut lines = 0;
    for n in 1..=CAP {
        let path = format!("{dir}/{prefix}-{n}.{ext}");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let set = parse(line.split('\t').next().unwrap());
            assert_eq!(set.len(), n, "{path}: {line}");
            assert!(
                set.iter().any(|&s| mating_sets::is_royal_symbol(s)),
                "{path}: royal-less set {line}"
            );
            lines += 1;
            for o in orientations(&set) {
                out.insert(set_index(&o));
            }
        }
    }
    (out, lines)
}

/// A multiplier that spreads `sets` over the buckets with none overflowing, then the
/// buckets, so every lookup reads one bucket.
fn helpmate_hash(sets: &std::collections::BTreeSet<u32>) -> Vec<u8> {
    use mating_sets::{HELPMATE_BUCKET_BITS, HELPMATE_SLOTS};
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    loop {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mul = (state >> 32) as u32 | 1;
        let mut buckets = vec![Vec::new(); 1 << HELPMATE_BUCKET_BITS];
        for &i in sets {
            buckets[(i.wrapping_mul(mul) >> (32 - HELPMATE_BUCKET_BITS)) as usize].push(i);
        }
        if buckets.iter().all(|b| b.len() <= HELPMATE_SLOTS) {
            let mut out = mul.to_le_bytes().to_vec();
            for b in &buckets {
                for k in 0..HELPMATE_SLOTS {
                    out.extend(b.get(k).copied().unwrap_or(u32::MAX).to_le_bytes());
                }
            }
            return out;
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: insuffmat_build <tables dir> [helpmate-only list] [out file]");
    let helpmate_list = args.next();
    let out = args
        .next()
        .unwrap_or_else(|| "src/evaluation/insuffmat_unbounded.bin".to_string());

    let (mates, mate_lines) = read(&dir, "mates", "tsv");
    let (draws, draw_lines) = read(&dir, "draws", "txt");

    let mut mate = vec![false; SET_COUNT];
    let mut bits = vec![0u8; mating_sets::DEAD_BYTES];
    let (mut dead, mut errors) = (0usize, 0usize);
    for_each_set(|set| {
        let i = set_index(set);
        let listed_mate = mates.contains(&i);
        let mut sub = [0u8; CAP];
        let contains_mate = (0..set.len()).any(|skip| {
            let mut n = 0;
            for (j, &s) in set.iter().enumerate() {
                if j != skip {
                    sub[n] = s;
                    n += 1;
                }
            }
            let sub = &sub[..n];
            sub.iter().any(|&s| mating_sets::is_royal_symbol(s)) && mate[set_index(sub)]
        });
        if listed_mate && contains_mate {
            eprintln!("listed mate is not smallest: {set:?}");
            errors += 1;
        }
        mate[i] = listed_mate || contains_mate;
        if mate[i] == draws.contains(&i) {
            let why = if mate[i] { "listed draw can mate" } else { "neither draw nor mate" };
            eprintln!("{why}: {set:?}");
            errors += 1;
        }
        if !mate[i] {
            bits[i >> 3] |= 1 << (i & 7);
            dead += 1;
        }
    });
    assert_eq!(errors, 0, "{errors} inconsistent sets");

    let mut helpmate = std::collections::BTreeSet::new();
    if let Some(path) = &helpmate_list {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let set = parse(line.split('\t').next().unwrap());
            let black_royals_only = set
                .iter()
                .filter(|&&s| s / KINDS == 1)
                .all(|&s| mating_sets::is_royal_symbol(s));
            assert!(black_royals_only && set.iter().any(|&s| s / KINDS == 0), "not a core: {line}");
            // Both bishop-colour images; never the colour swap, which is the other side attacking.
            for o in &orientations(&set)[..] {
                if o.iter().zip(&set).all(|(a, b)| a / KINDS == b / KINDS) {
                    let i = set_index(o);
                    assert!(mate[i], "helpmate-only core cannot mate at all: {line}");
                    helpmate.insert(i as u32);
                }
            }
        }
    }

    let helpmate_sets = helpmate.len();
    bits.extend(helpmate_hash(&helpmate));
    assert_eq!(bits.len(), mating_sets::TABLE_BYTES);
    std::fs::write(&out, &bits).unwrap_or_else(|e| panic!("{out}: {e}"));
    println!(
        "{mate_lines} smallest mates ({} with mirrors), {draw_lines} draws ({} with mirrors)",
        mates.len(),
        draws.len()
    );
    println!(
        "{SET_COUNT} sets: {dead} dead, {} can mate, {helpmate_sets} helpmate-only sets; wrote {} bytes to {out}",
        SET_COUNT - dead,
        bits.len()
    );
}
