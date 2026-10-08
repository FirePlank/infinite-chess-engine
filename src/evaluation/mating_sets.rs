//! Which piece sets can be arranged into a legally reachable checkmate, helpmates
//! included, from infinitechess.org's generated tables: up to [`CAP`] pieces on an
//! unbounded board, up to [`BOUNDED_CAP`] on a bounded one. Unbounded, also which
//! armies can only mate with the defender's help. Built by the `insuffmat_build` binary.

use crate::board::{PieceType, PlayerColor};

/// Most pieces in a tabled set, counting both sides and the royals.
pub const CAP: usize = 5;
/// The same for bounded boards, whose table was generated on 8x8.
pub const BOUNDED_CAP: usize = 4;
/// Piece kinds in table order: the 3 royals first, then the 18 others.
pub const KIND_CODES: [&str; 21] = [
    "K", "RC", "RQ", "Q", "R", "B0", "B1", "N", "P", "AM", "HA", "CH", "AR", "GU", "CA", "GI",
    "ZE", "CE", "NR", "HU", "RO",
];
pub const KINDS: u8 = 21;
pub const ROYAL_KINDS: u8 = 3;
const OTHER_KINDS: u8 = KINDS - ROYAL_KINDS;
/// Symbols are `color * KINDS + kind`, white 0, black 1.
pub const SYMBOLS: u8 = 2 * KINDS;
pub const PAWN_KIND: u8 = 8;

const fn binom(n: u32, k: u32) -> u32 {
    let mut r: u64 = 1;
    let mut i = 0;
    while i < k {
        r = r * (n - i) as u64 / (i + 1) as u64;
        i += 1;
    }
    r as u32
}

/// `COLEX[i][v] = C(v + i, i + 1)`: slot `i` of one side's sorted values holding `v`.
const COLEX: [[u32; KINDS as usize]; CAP] = {
    let mut t = [[0u32; KINDS as usize]; CAP];
    let mut i = 0;
    while i < CAP {
        let mut v = 0;
        while v < KINDS as usize {
            t[i][v] = binom((v + i) as u32, (i + 1) as u32);
            v += 1;
        }
        i += 1;
    }
    t
};

/// Offsets for indexing sets of at most `cap` pieces up to a color swap. A side's
/// pieces are ranked colex over the values `KINDS - 1 - kind`, so for `m` pieces the
/// royal-less sides take ranks below `royal_less[m]` and the rest hold a royal.
struct Layout {
    royal_less: [u32; CAP + 1],
    /// Royal-less sides of fewer than `s` pieces.
    royal_less_before: [u32; CAP + 2],
    /// Royal-holding sides of fewer than `s` pieces.
    royal_before: [u32; CAP + 2],
    /// Index of the first set whose royal side (the other has none) has `a` pieces.
    one_royal_side: [u32; CAP + 2],
    /// Index of the first set with royals on both sides whose larger side has `s`
    /// pieces, for `s` past half the cap.
    two_royal_sides: [u32; CAP + 2],
    count: u32,
}

const fn layout(cap: usize) -> Layout {
    let mut l = Layout {
        royal_less: [0; CAP + 1],
        royal_less_before: [0; CAP + 2],
        royal_before: [0; CAP + 2],
        one_royal_side: [0; CAP + 2],
        two_royal_sides: [0; CAP + 2],
        count: 0,
    };
    let mut royal = [0u32; CAP + 1];
    let mut m = 0;
    while m <= cap {
        l.royal_less[m] = binom(OTHER_KINDS as u32 - 1 + m as u32, m as u32);
        royal[m] = binom(KINDS as u32 - 1 + m as u32, m as u32) - l.royal_less[m];
        l.royal_less_before[m + 1] = l.royal_less_before[m] + l.royal_less[m];
        l.royal_before[m + 1] = l.royal_before[m] + royal[m];
        m += 1;
    }
    let mut a = 1;
    while a <= cap {
        l.one_royal_side[a + 1] = l.one_royal_side[a] + royal[a] * l.royal_less_before[cap - a + 1];
        a += 1;
    }
    let half = cap / 2;
    let paired = l.royal_before[half + 1];
    l.two_royal_sides[half + 1] = l.one_royal_side[cap + 1] + paired * (paired + 1) / 2;
    let mut s = half + 1;
    while s <= cap {
        l.two_royal_sides[s + 1] = l.two_royal_sides[s] + royal[s] * l.royal_before[cap - s + 1];
        s += 1;
    }
    l.count = l.two_royal_sides[cap + 1];
    l
}

const LAYOUT: Layout = layout(CAP);
const BOUNDED_LAYOUT: Layout = layout(BOUNDED_CAP);

/// Sets with at least one royal, up to a color swap, which never changes whether
/// a mate is possible; royal-less sets are never mates and have no index.
pub const SET_COUNT: usize = LAYOUT.count as usize;
pub const BOUNDED_SET_COUNT: usize = BOUNDED_LAYOUT.count as usize;

/// [`DEAD_BYTES`] of one bit per unbounded set, set when no mate is possible. Then
/// [`HELPMATE_BYTES`] hashing the sets where Black has only royals and mate is
/// possible, but White can never force it (`forced_gen` proved it): a `u32`
/// multiplier, then buckets of [`HELPMATE_SLOTS`] `u32` [`attack_key`]s padded
/// with `u32::MAX`, all little-endian. Then [`BOUNDED_DEAD_BYTES`] of bounded dead bits,
/// and the same hash of the bounded helpmate-only sets (`bounded_helpmate` proved it).
static TABLE: &[u8] = include_bytes!("insuffmat.bin");
pub const DEAD_BYTES: usize = SET_COUNT.div_ceil(8);
pub const HELPMATE_BUCKET_BITS: u32 = 7;
pub const HELPMATE_SLOTS: usize = 8;
pub const HELPMATE_BYTES: usize = 4 + (HELPMATE_SLOTS << HELPMATE_BUCKET_BITS) * 4;
pub const BOUNDED_DEAD_BYTES: usize = BOUNDED_SET_COUNT.div_ceil(8);
pub const BOUNDED_HELPMATE_BUCKET_BITS: u32 = 8;
pub const BOUNDED_HELPMATE_BYTES: usize = 4 + (HELPMATE_SLOTS << BOUNDED_HELPMATE_BUCKET_BITS) * 4;
const BOUNDED_START: usize = DEAD_BYTES + HELPMATE_BYTES;
const BOUNDED_HELPMATE_START: usize = BOUNDED_START + BOUNDED_DEAD_BYTES;
pub const TABLE_BYTES: usize = BOUNDED_HELPMATE_START + BOUNDED_HELPMATE_BYTES;

#[inline]
pub fn is_royal_symbol(s: u8) -> bool {
    s % KINDS < ROYAL_KINDS
}

/// The symbol of a colored piece, `None` for neutral pieces. `x + y` picks the
/// bishop's square color.
#[inline]
pub fn symbol(pt: PieceType, color: PlayerColor, x: i64, y: i64) -> Option<u8> {
    let kind = match pt {
        PieceType::King => 0,
        PieceType::RoyalCentaur => 1,
        PieceType::RoyalQueen => 2,
        PieceType::Queen => 3,
        PieceType::Rook => 4,
        PieceType::Bishop => 5 + ((x ^ y) & 1) as u8,
        PieceType::Knight => 7,
        PieceType::Pawn => 8,
        PieceType::Amazon => 9,
        PieceType::Hawk => 10,
        PieceType::Chancellor => 11,
        PieceType::Archbishop => 12,
        PieceType::Guard => 13,
        PieceType::Camel => 14,
        PieceType::Giraffe => 15,
        PieceType::Zebra => 16,
        PieceType::Centaur => 17,
        PieceType::Knightrider => 18,
        PieceType::Huygen => 19,
        PieceType::Rose => 20,
        PieceType::Void | PieceType::Obstacle => return None,
    };
    match color {
        PlayerColor::White => Some(kind),
        PlayerColor::Black => Some(KINDS + kind),
        PlayerColor::Neutral => None,
    }
}

/// The index of a set with 1 to [`CAP`] symbols, at least one royal, in any order;
/// a set and its color swap share it.
#[inline]
pub fn set_index(symbols: &[u8]) -> usize {
    canonical(symbols, CAP, &LAYOUT).expect("a royal").0
}

/// [`set_index`] among the sets of at most [`BOUNDED_CAP`] symbols.
#[inline]
pub fn bounded_set_index(symbols: &[u8]) -> usize {
    canonical(symbols, BOUNDED_CAP, &BOUNDED_LAYOUT).expect("a royal").0
}

/// The hash key of `symbols` with the given side attacking: its [`set_index`] and
/// whether the attacker is the side the index puts first.
#[inline]
pub fn attack_key(symbols: &[u8], white_attacks: bool) -> u32 {
    let (i, swapped) = canonical(symbols, CAP, &LAYOUT).expect("a royal");
    key(i, swapped, white_attacks)
}

/// [`attack_key`] among the sets of at most [`BOUNDED_CAP`] symbols.
#[inline]
pub fn bounded_attack_key(symbols: &[u8], white_attacks: bool) -> u32 {
    let (i, swapped) = canonical(symbols, BOUNDED_CAP, &BOUNDED_LAYOUT).expect("a royal");
    key(i, swapped, white_attacks)
}

#[inline]
fn key(i: usize, swapped: bool, white_attacks: bool) -> u32 {
    (i as u32) << 1 | (swapped != white_attacks) as u32
}

/// The set's index, and whether Black's side comes first in it; `None` without a
/// royal. A side holding a royal comes first; with royals on both, the lower-ranked
/// side does.
#[inline]
fn canonical(symbols: &[u8], cap: usize, l: &Layout) -> Option<(usize, bool)> {
    debug_assert!(symbols.len() <= cap);
    let (mut white, mut black) = ([0u8; CAP], [0u8; CAP]);
    let (mut w, mut b) = (0, 0);
    for &s in symbols {
        if s < KINDS {
            white[w] = KINDS - 1 - s;
            w += 1;
        } else {
            black[b] = SYMBOLS - 1 - s;
            b += 1;
        }
    }
    let rank = |v: &mut [u8]| {
        sort_small(v);
        v.iter().enumerate().map(|(i, &x)| COLEX[i][x as usize]).sum::<u32>()
    };
    let (rw, rb) = (rank(&mut white[..w]), rank(&mut black[..b]));
    let (white_royal, black_royal) = (rw >= l.royal_less[w], rb >= l.royal_less[b]);
    if !white_royal && !black_royal {
        return None;
    }
    if white_royal != black_royal {
        let ((a, ra), (n, rn)) = if white_royal { ((w, rw), (b, rb)) } else { ((b, rb), (w, rw)) };
        let i = l.one_royal_side[a] + (ra - l.royal_less[a]) * l.royal_less_before[cap - a + 1] + l.royal_less_before[n] + rn;
        return Some((i as usize, black_royal));
    }
    let gw = l.royal_before[w] + rw - l.royal_less[w];
    let gb = l.royal_before[b] + rb - l.royal_less[b];
    let swapped = gb < gw;
    let (low, high, size) = if swapped { (gb, gw, w) } else { (gw, gb, b) };
    let i = if 2 * size <= cap {
        l.one_royal_side[cap + 1] + high * (high + 1) / 2 + low
    } else {
        l.two_royal_sides[size] + (high - l.royal_before[size]) * l.royal_before[cap - size + 1] + low
    };
    Some((i as usize, swapped))
}

#[inline]
fn sort_small(v: &mut [u8]) {
    for i in 1..v.len() {
        let mut j = i;
        while j > 0 && v[j - 1] > v[j] {
            v.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Whether no checkmate of either side is possible with exactly these pieces,
/// not even a helpmate. Sets above [`CAP`] are never called dead.
#[inline]
pub fn is_dead(symbols: &[u8]) -> bool {
    if symbols.len() > CAP {
        return false;
    }
    canonical(symbols, CAP, &LAYOUT).is_none_or(|(i, _)| dead_bit(i))
}

/// Whether no checkmate of either side is possible with exactly these pieces on a
/// bounded board, not even a helpmate. Sets above [`BOUNDED_CAP`] are never dead.
#[inline]
pub fn is_dead_bounded(symbols: &[u8]) -> bool {
    if symbols.len() > BOUNDED_CAP {
        return false;
    }
    canonical(symbols, BOUNDED_CAP, &BOUNDED_LAYOUT).is_none_or(|(i, _)| bounded_dead_bit(i))
}

#[inline]
fn bounded_dead_bit(i: usize) -> bool {
    TABLE[BOUNDED_START + (i >> 3)] >> (i & 7) & 1 == 1
}

#[inline]
fn dead_bit(i: usize) -> bool {
    TABLE[i >> 3] >> (i & 7) & 1 == 1
}

/// Whether `key` is in the helpmate-only hash at `start` with `bucket_bits`.
#[inline]
fn in_hash(key: u32, start: usize, bucket_bits: u32) -> bool {
    let mul = u32::from_le_bytes(TABLE[start..start + 4].try_into().unwrap());
    let bucket = key.wrapping_mul(mul) >> (32 - bucket_bits);
    let base = start + 4 + bucket as usize * HELPMATE_SLOTS * 4;
    let slots: &[u8; HELPMATE_SLOTS * 4] = TABLE[base..base + HELPMATE_SLOTS * 4].try_into().unwrap();
    // Every slot is compared, so the cost is the same for every set.
    slots.chunks_exact(4).fold(false, |hit, w| hit | (u32::from_le_bytes(w.try_into().unwrap()) == key))
}

/// Whether the attacking side's pieces in `symbols` (White's when `white_attacks`)
/// can never force mate against the other side's royals, which are all the other
/// side holds: no mate is possible, or only with the defender's help.
#[inline]
pub fn helpless(symbols: &[u8], white_attacks: bool) -> bool {
    if symbols.len() > CAP {
        return false;
    }
    canonical(symbols, CAP, &LAYOUT).is_none_or(|(i, swapped)| {
        dead_bit(i) || in_hash(key(i, swapped, white_attacks), DEAD_BYTES, HELPMATE_BUCKET_BITS)
    })
}

/// [`helpless`] on a bounded board, for sets of up to [`BOUNDED_CAP`] pieces.
#[inline]
pub fn helpless_bounded(symbols: &[u8], white_attacks: bool) -> bool {
    if symbols.len() > BOUNDED_CAP {
        return false;
    }
    canonical(symbols, BOUNDED_CAP, &BOUNDED_LAYOUT).is_none_or(|(i, swapped)| {
        bounded_dead_bit(i)
            || in_hash(key(i, swapped, white_attacks), BOUNDED_HELPMATE_START, BOUNDED_HELPMATE_BUCKET_BITS)
    })
}

/// Calls `f` with every set of at least one royal and at most `cap` pieces,
/// smallest sets first.
pub fn for_each_set(cap: usize, mut f: impl FnMut(&[u8])) {
    fn multisets(n: u8, k: usize, start: u8, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for v in start..n {
            cur.push(v);
            multisets(n, k, v, cur, out);
            cur.pop();
        }
    }
    let royal_symbols: Vec<u8> = (0..SYMBOLS).filter(|&s| is_royal_symbol(s)).collect();
    let other_symbols: Vec<u8> = (0..SYMBOLS).filter(|&s| !is_royal_symbol(s)).collect();
    for total in 1..=cap {
        for a in 1..=total {
            let (mut rs, mut os) = (Vec::new(), Vec::new());
            multisets(royal_symbols.len() as u8, a, 0, &mut Vec::new(), &mut rs);
            multisets(other_symbols.len() as u8, total - a, 0, &mut Vec::new(), &mut os);
            for r in &rs {
                for o in &os {
                    let set: Vec<u8> = r
                        .iter()
                        .map(|&i| royal_symbols[i as usize])
                        .chain(o.iter().map(|&i| other_symbols[i as usize]))
                        .collect();
                    f(&set);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(label: &str) -> Vec<u8> {
        let (w, b) = label.split_once(" vs ").unwrap();
        let side = |s: &str, color: u8| -> Vec<u8> {
            s.split(',')
                .filter(|c| !c.is_empty())
                .map(|c| {
                    let k = KIND_CODES.iter().position(|k| k.eq_ignore_ascii_case(c));
                    color * KINDS + k.unwrap() as u8
                })
                .collect()
        };
        let mut v = side(w, 0);
        v.extend(side(b, 1));
        v
    }

    /// Every index is used, by a set and its color swap alone.
    #[test]
    fn index_covers_each_color_pair_once() {
        for (cap, count, index) in [
            (CAP, SET_COUNT, set_index as fn(&[u8]) -> usize),
            (BOUNDED_CAP, BOUNDED_SET_COUNT, bounded_set_index),
        ] {
            let mut owner: Vec<Option<Vec<u8>>> = vec![None; count];
            for_each_set(cap, |s| {
                let mut swap: Vec<u8> = s.iter().map(|&x| (x + KINDS) % SYMBOLS).collect();
                swap.sort_unstable();
                let mut key = s.to_vec();
                key.sort_unstable();
                let pair = key.min(swap);
                let slot = &mut owner[index(s)];
                assert!(slot.as_ref().is_none_or(|o| *o == pair), "{s:?}");
                *slot = Some(pair);
            });
            assert!(owner.iter().all(Option::is_some));
        }
        assert_eq!(SET_COUNT, 392_302);
        assert_eq!(BOUNDED_SET_COUNT, 35_929);
    }

    #[test]
    fn known_verdicts() {
        for dead in ["K vs k", "K,Q vs k", "K,R vs k,r", "Q,R vs k", " vs RQ,Q,N", "K,N,N vs k"] {
            assert!(is_dead(&set(dead)), "{dead}");
        }
        for mate in ["K,R,R vs k", "Q,Q vs k", "K,AM vs k", "K,Q vs k,b0,b0", "K vs rq,q"] {
            assert!(!is_dead(&set(mate)), "{mate}");
        }
        // Order and mirror images do not matter.
        assert!(!is_dead(&set("K,B1,B1 vs q,k")));
        assert!(is_dead(&set("Q,N vs ")));
    }

    #[test]
    fn known_bounded_verdicts() {
        for dead in ["K vs k", "K,N vs k", "AM,Q vs rq", "K,B0,B0 vs k"] {
            assert!(is_dead_bounded(&set(dead)), "{dead}");
        }
        for mate in ["K,R vs k", "K vs k,rc", "AM,AM vs rq", "K,B0,B1 vs k", "K,N,N vs k"] {
            assert!(!is_dead_bounded(&set(mate)), "{mate}");
        }
        assert!(!is_dead_bounded(&set("K,R,R,R,R vs k")), "past the bounded cap");
        assert!(helpless_bounded(&set("K,N,N vs k"), true));
        assert!(helpless_bounded(&set("k vs K,N,N"), false));
        assert!(!helpless_bounded(&set("K,B0,N vs k"), true));
        assert!(!helpless_bounded(&set("K,R vs k"), true));
    }

    #[test]
    fn helpmate_only_armies() {
        assert!(helpless(&set("K,GU,GU vs k"), true));
        assert!(helpless(&set("k vs K,GU,GU"), false));
        assert!(!helpless(&set("K,R,R vs k"), true));
        assert!(!helpless(&set("K,AM vs k"), true));
        assert!(helpless(&set("K,Q vs k"), true));
    }
}
