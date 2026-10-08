//! Builds `src/evaluation/insuffmat.bin` from infinitechess.org's generated
//! `matingsets.ts` (src/shared/chess/logic/insuffmat/): every smallest piece set that
//! can mate, per board kind, labels like `K,B0/k,n` (White's pieces, then Black's),
//! mirror images listed once. A set can mate exactly when a listed set fits inside it.
//! The optional helpmate-only list (`forced_gen`'s `escape.txt`: White's army vs
//! Black's royals, labels like `K,GU,GU vs k`) is appended as a one-probe hash of set
//! indices, then the bounded table's dead bits and the same hash of the bounded
//! helpmate-only list (`bounded_helpmate`'s `helpmate_only.txt`).
//!
//! Usage: cargo run --release --bin insuffmat_build -- <matingsets.ts> [helpmate-only list]
//!        [bounded helpmate-only list] [out file]

use apeiron::evaluation::mating_sets::{
    self, BOUNDED_CAP, BOUNDED_SET_COUNT, CAP, KIND_CODES, KINDS, SET_COUNT, SYMBOLS, attack_key,
    bounded_attack_key, bounded_set_index, for_each_set, set_index,
};

/// The set with colors swapped.
fn swap_colors(set: &[u8]) -> Vec<u8> {
    set.iter().map(|&s| (s + KINDS) % SYMBOLS).collect()
}

/// A label's symbols: `K,B0/k,n` or `K,B0 vs k,n`, uppercase White, lowercase Black.
fn parse(label: &str) -> Vec<u8> {
    let (white, black) = label
        .split_once('/')
        .or_else(|| label.split_once(" vs"))
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

/// The space-separated labels quoted after `kind:` in matingsets.ts.
fn listed(ts: &str, kind: &str) -> Vec<Vec<u8>> {
    let body = &ts[ts.find(&format!("\t{kind}:")).unwrap_or_else(|| panic!("no {kind} list"))..];
    let body = &body[body.find('\'').unwrap() + 1..];
    body[..body.find('\'').unwrap()].split(' ').map(parse).collect()
}

/// The `cap: { unbounded: N, bounded: M }` entry for `kind`.
fn listed_cap(ts: &str, kind: &str) -> usize {
    let caps = &ts[ts.find("cap:").expect("no cap")..];
    let caps = &caps[..caps.find('}').unwrap()];
    let n = &caps[caps.find(&format!(" {kind}:")).unwrap() + kind.len() + 2..];
    n.trim_start().split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()
}

/// Per set up to `cap` (by `index`), whether it can mate, from the smallest mating
/// sets, which must each be smallest and hold a royal.
fn closure(smallest: &[Vec<u8>], cap: usize, count: usize, index: fn(&[u8]) -> usize) -> Vec<bool> {
    let mut listed = vec![false; count];
    for set in smallest {
        assert!(set.len() <= cap && set.iter().any(|&s| mating_sets::is_royal_symbol(s)), "{set:?}");
        for o in orientations(set) {
            listed[index(&o)] = true;
        }
    }
    let mut mate = vec![false; count];
    let mut errors = 0;
    for_each_set(cap, |set| {
        let i = index(set);
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
        if listed[i] && contains_mate {
            eprintln!("listed mate is not smallest: {set:?}");
            errors += 1;
        }
        mate[i] = listed[i] || contains_mate;
    });
    assert_eq!(errors, 0, "{errors} listed mates are not smallest");
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

/// A multiplier that spreads `sets` over the buckets with none overflowing, then the
/// buckets, so every lookup reads one bucket.
fn helpmate_hash(sets: &std::collections::BTreeSet<u32>, bucket_bits: u32) -> Vec<u8> {
    use mating_sets::HELPMATE_SLOTS;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    loop {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mul = (state >> 32) as u32 | 1;
        let mut buckets = vec![Vec::new(); 1 << bucket_bits];
        for &i in sets {
            buckets[(i.wrapping_mul(mul) >> (32 - bucket_bits)) as usize].push(i);
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
    let mut bits = dead_bits(&mate);

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
                    assert!(mate[set_index(o)], "helpmate-only core cannot mate at all: {line}");
                    // A color-symmetric set keys each attacking side apart.
                    helpmate.insert(attack_key(o, true));
                    helpmate.insert(attack_key(&swap_colors(o), false));
                }
            }
        }
    }

    let mut bounded_helpmate = std::collections::BTreeSet::new();
    if let Some(path) = &bounded_helpmate_list {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        // Each bishop-color image is solved and listed on its own.
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let set = parse(line);
            assert!(bounded_mate[bounded_set_index(&set)], "bounded helpmate-only core cannot mate at all: {line}");
            bounded_helpmate.insert(bounded_attack_key(&set, true));
            bounded_helpmate.insert(bounded_attack_key(&swap_colors(&set), false));
        }
    }

    let helpmate_sets = helpmate.len();
    bits.extend(helpmate_hash(&helpmate, mating_sets::HELPMATE_BUCKET_BITS));
    bits.extend(dead_bits(&bounded_mate));
    bits.extend(helpmate_hash(&bounded_helpmate, mating_sets::BOUNDED_HELPMATE_BUCKET_BITS));
    assert_eq!(bits.len(), mating_sets::TABLE_BYTES);
    std::fs::write(&out, &bits).unwrap_or_else(|e| panic!("{out}: {e}"));
    let count = |m: &[bool]| m.iter().filter(|&&m| !m).count();
    println!(
        "unbounded: {} smallest mates, {SET_COUNT} sets, {} dead, {helpmate_sets} helpmate-only",
        unbounded.len(),
        count(&mate)
    );
    println!(
        "bounded: {} smallest mates, {BOUNDED_SET_COUNT} sets, {} dead, {} helpmate-only; wrote {} bytes to {out}",
        bounded.len(),
        count(&bounded_mate),
        bounded_helpmate.len(),
        bits.len()
    );
}
