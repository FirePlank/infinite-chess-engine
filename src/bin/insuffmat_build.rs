//! Builds `src/evaluation/insuffmat.bin` from infinitechess.org's generated
//! `matingsets.ts` (src/shared/chess/logic/insuffmat/): every smallest piece set that
//! can mate, per board kind, as the sorted canonical ranks' gaps in base-32 digits
//! (materials.ts's `decodeRanks`). A set can mate exactly when a listed set fits
//! inside it. The optional helpmate-only lists (`forced_gen`'s `escape.txt` and
//! `bounded_helpmate`'s `helpmate_only.txt`: White's army vs Black's royals, labels
//! like `K,GU,GU vs k`) follow the dead bits as delta-coded keys.
//!
//! Usage: cargo run --release --bin insuffmat_build -- <matingsets.ts> [helpmate-only list]
//!        [bounded helpmate-only list] [out file]

use apeiron::evaluation::mating_sets::{
    self, BOUNDED_CAP, BOUNDED_SET_COUNT, CAP, KIND_CODES, KINDS, SET_COUNT, SYMBOLS, attack_key,
    bounded_attack_key, bounded_set_index, for_each_set, set_index,
};
use std::collections::{BTreeSet, HashSet};

/// matingsets.ts's piece order for ranks: royals last.
const RANKED_CODES: &str = "ro hu nr ce ze gi ca1 ca0 gu ar ch ha am p n b1 b0 r q RO HU NR CE ZE GI CA1 CA0 GU AR CH HA AM P N B1 B0 R Q rq rc k RQ RC K";
/// matingsets.ts's digits: a number's last digit is one of the first 32, its others of the last 32.
const RANK_DIGITS: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_";

/// The set with colors swapped.
fn swap_colors(set: &[u8]) -> Vec<u8> {
    set.iter().map(|&s| (s + KINDS) % SYMBOLS).collect()
}

/// A label's symbols: `K,B0 vs k,n`, uppercase White, lowercase Black.
fn parse(label: &str) -> Vec<u8> {
    let (white, black) = label
        .split_once(" vs")
        .unwrap_or_else(|| panic!("no side separator in {label:?}"));
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

/// The set and its mirror images: colors swapped, square colors swapped, or both.
fn orientations(set: &[u8]) -> [Vec<u8>; 4] {
    let swap_colors = |s: u8| (s + KINDS) % SYMBOLS;
    let swap_squares = |s: u8| match s % KINDS {
        5 | 14 => s + 1,
        6 | 15 => s - 1,
        _ => s,
    };
    [
        set.to_vec(),
        set.iter().map(|&s| swap_colors(s)).collect(),
        set.iter().map(|&s| swap_squares(s)).collect(),
        set.iter().map(|&s| swap_squares(swap_colors(s))).collect(),
    ]
}

fn binomial(n: u64, k: u64) -> u64 {
    (0..k).fold(1, |r, i| r * (n - i) / (i + 1))
}

/// matingsets.ts's rank of a set, shared by its mirror images: the least colex rank
/// over [`RANKED_CODES`] among them.
fn site_rank(set: &[u8], order: &[u64; SYMBOLS as usize]) -> u64 {
    let codes = RANKED_CODES.split(' ').count() as u64;
    orientations(set)
        .iter()
        .map(|o| {
            let mut values: Vec<u64> = o.iter().map(|&s| order[s as usize]).collect();
            values.sort_unstable();
            let fewer = binomial(codes - 1 + values.len() as u64, values.len() as u64 - 1);
            fewer
                + values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| binomial(v + i as u64, i as u64 + 1))
                    .sum::<u64>()
        })
        .min()
        .unwrap()
}

/// Per symbol, its place in [`RANKED_CODES`].
fn ranked_order() -> [u64; SYMBOLS as usize] {
    let codes: Vec<&str> = RANKED_CODES.split(' ').collect();
    let mut order = [0; SYMBOLS as usize];
    for s in 0..SYMBOLS {
        let code = KIND_CODES[(s % KINDS) as usize];
        let code = if s < KINDS {
            code.to_string()
        } else {
            code.to_lowercase()
        };
        order[s as usize] = codes
            .iter()
            .position(|&c| c == code)
            .unwrap_or_else(|| panic!("{code} unranked")) as u64;
    }
    order
}

/// The ranks of the list quoted after `kind:` in matingsets.ts.
fn listed(ts: &str, kind: &str) -> HashSet<u64> {
    let body = &ts[ts
        .find(&format!("\t{kind}:"))
        .unwrap_or_else(|| panic!("no {kind} list"))..];
    let body = &body[body.find('\'').unwrap() + 1..];
    let (mut ranks, mut rank, mut gap) = (HashSet::new(), 0u64, 0u64);
    for c in body[..body.find('\'').unwrap()].chars() {
        let digit = RANK_DIGITS
            .find(c)
            .unwrap_or_else(|| panic!("bad digit {c:?}")) as u64;
        gap = gap * 32 + digit % 32;
        if digit < 32 {
            rank += gap;
            ranks.insert(rank);
            gap = 0;
        }
    }
    ranks
}

/// The `cap: { unbounded: N, bounded: M }` entry for `kind`.
fn listed_cap(ts: &str, kind: &str) -> usize {
    let caps = &ts[ts.find("cap:").expect("no cap")..];
    let caps = &caps[..caps.find('}').unwrap()];
    let n = &caps[caps.find(&format!(" {kind}:")).unwrap() + kind.len() + 2..];
    n.trim_start()
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// Per set up to `cap` (by `index`), whether it can mate, from the ranks of the
/// smallest mating sets, which must each be one of these sets and smallest.
fn closure(
    listed: &HashSet<u64>,
    cap: usize,
    count: usize,
    index: fn(&[u8]) -> usize,
) -> Vec<bool> {
    let order = ranked_order();
    let mut found = HashSet::new();
    let mut mate = vec![false; count];
    let mut errors = 0;
    for_each_set(cap, |set| {
        let i = index(set);
        let rank = site_rank(set, &order);
        let is_listed = listed.contains(&rank);
        if is_listed {
            found.insert(rank);
        }
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
            sub.iter().any(|&s| mating_sets::is_royal_symbol(s)) && mate[index(sub)]
        });
        if is_listed && contains_mate {
            eprintln!("listed mate is not smallest: {set:?}");
            errors += 1;
        }
        mate[i] |= is_listed || contains_mate;
    });
    assert_eq!(errors, 0, "{errors} listed mates are not smallest");
    assert_eq!(
        found.len(),
        listed.len(),
        "listed ranks that are no set of at most {cap} pieces with a royal"
    );
    mate
}

/// One bit per set, set when it cannot mate.
fn dead_bits(mate: &[bool]) -> Vec<u8> {
    let mut bits = vec![0u8; mate.len().div_ceil(8)];
    for (i, _) in mate.iter().enumerate().filter(|(_, m)| !**m) {
        bits[i >> 3] |= 1 << (i & 7);
    }
    bits
}

/// The keys as `mating_sets` reads them: a multiplier that spreads them over the
/// buckets with none overflowing, their count, and the sorted keys' gaps as varints.
fn helpmate_section(keys: &BTreeSet<u32>, bucket_bits: u32) -> Vec<u8> {
    use mating_sets::HELPMATE_SLOTS;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mul = loop {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mul = (state >> 32) as u32 | 1;
        let mut fill = vec![0usize; 1 << bucket_bits];
        for &k in keys {
            fill[(k.wrapping_mul(mul) >> (32 - bucket_bits)) as usize] += 1;
        }
        if fill.iter().all(|&n| n <= HELPMATE_SLOTS) {
            break mul;
        }
    };
    let mut out = mul.to_le_bytes().to_vec();
    out.extend((keys.len() as u32).to_le_bytes());
    let mut last = 0;
    for &k in keys {
        let mut gap = k - last;
        last = k;
        while gap >= 128 {
            out.push(gap as u8 | 128);
            gap >>= 7;
        }
        out.push(gap as u8);
    }
    out
}

/// The attack keys of a helpmate-only list, skipping cores the table calls dead: a
/// newer table can prove more draws.
fn helpmate_keys(
    path: &str,
    mate: &[bool],
    index: fn(&[u8]) -> usize,
    key: fn(&[u8], bool) -> u32,
) -> (BTreeSet<u32>, usize) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (mut keys, mut dead) = (BTreeSet::new(), 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let set = parse(line.split('\t').next().unwrap());
        let black_royals_only = set
            .iter()
            .filter(|&&s| s / KINDS == 1)
            .all(|&s| mating_sets::is_royal_symbol(s));
        assert!(
            black_royals_only && set.iter().any(|&s| s / KINDS == 0),
            "not a core: {line}"
        );
        if !mate[index(&set)] {
            dead += 1;
            continue;
        }
        // A color-symmetric set keys each attacking side apart.
        keys.insert(key(&set, true));
        keys.insert(key(&swap_colors(&set), false));
    }
    (keys, dead)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ts_path = args.next().expect(
        "usage: insuffmat_build <matingsets.ts> [helpmate-only list] [bounded helpmate-only list] [out file]",
    );
    let helpmate_list = args.next();
    let bounded_helpmate_list = args.next();
    let out = args
        .next()
        .unwrap_or_else(|| "src/evaluation/insuffmat.bin".to_string());
    let ts = std::fs::read_to_string(&ts_path).unwrap_or_else(|e| panic!("{ts_path}: {e}"));
    assert_eq!(listed_cap(&ts, "unbounded"), CAP);
    assert_eq!(listed_cap(&ts, "bounded"), BOUNDED_CAP);

    let (unbounded, bounded) = (listed(&ts, "unbounded"), listed(&ts, "bounded"));
    let mate = closure(&unbounded, CAP, SET_COUNT, set_index);
    let bounded_mate = closure(&bounded, BOUNDED_CAP, BOUNDED_SET_COUNT, bounded_set_index);
    let (helpmate, dead_helpmate) = helpmate_list.as_deref().map_or_else(Default::default, |p| {
        helpmate_keys(p, &mate, set_index, attack_key)
    });
    let (bounded_helpmate, dead_bounded_helpmate) = bounded_helpmate_list
        .as_deref()
        .map_or_else(Default::default, |p| {
            helpmate_keys(p, &bounded_mate, bounded_set_index, bounded_attack_key)
        });

    let mut bytes = dead_bits(&mate);
    bytes.extend(dead_bits(&bounded_mate));
    assert_eq!(bytes.len(), mating_sets::HELPMATES_START);
    bytes.extend(helpmate_section(
        &helpmate,
        mating_sets::HELPMATE_BUCKET_BITS,
    ));
    bytes.extend(helpmate_section(
        &bounded_helpmate,
        mating_sets::BOUNDED_HELPMATE_BUCKET_BITS,
    ));
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("{out}: {e}"));
    let count = |m: &[bool]| m.iter().filter(|&&m| !m).count();
    println!(
        "unbounded: {} smallest mates, {SET_COUNT} sets, {} dead, {} helpmate-only keys ({dead_helpmate} listed cores now dead)",
        unbounded.len(),
        count(&mate),
        helpmate.len()
    );
    println!(
        "bounded: {} smallest mates, {BOUNDED_SET_COUNT} sets, {} dead, {} helpmate-only keys ({dead_bounded_helpmate} listed cores now dead); wrote {} bytes to {out}",
        bounded.len(),
        count(&bounded_mate),
        bounded_helpmate.len(),
        bytes.len()
    );
}
