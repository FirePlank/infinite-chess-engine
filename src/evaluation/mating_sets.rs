//! Which piece sets of up to [`CAP`] pieces can be arranged into a legally reachable
//! checkmate on an unbounded board, helpmates included, from infinitechess.org's
//! generated tables, and which armies can only mate with the defender's help. A dead
//! bit per set, indexed by [`set_index`], then the few helpmate-only sets in a small
//! one-probe hash; built by the `insuffmat_build` binary.

use crate::board::{PieceType, PlayerColor};

/// Most pieces in a tabled set, counting both sides and the royals.
pub const CAP: usize = 5;
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

/// `BIN[i][c] = C(c + i, i + 1)`: slot `i` of a sorted multiset holding value `c`.
const BIN: [[u32; 2 * OTHER_KINDS as usize]; CAP] = {
    let mut t = [[0u32; 2 * OTHER_KINDS as usize]; CAP];
    let mut i = 0;
    while i < CAP {
        let mut c = 0;
        while c < 2 * OTHER_KINDS as usize {
            t[i][c] = binom((c + i) as u32, (i + 1) as u32);
            c += 1;
        }
        i += 1;
    }
    t
};

/// `CUM_OTHER[m]`: multisets of at most `m` non-royal symbols.
const CUM_OTHER: [u32; CAP + 1] = {
    let mut t = [0u32; CAP + 1];
    let n = 2 * OTHER_KINDS as u32;
    let mut m = 0;
    let mut acc = 0;
    while m <= CAP {
        acc += binom(n + m as u32 - 1, m as u32);
        t[m] = acc;
        m += 1;
    }
    t
};

/// `ROYAL_BASE[a]`: index of the first set with `a` royals.
const ROYAL_BASE: [u32; CAP + 2] = {
    let mut t = [0u32; CAP + 2];
    let n = 2 * ROYAL_KINDS as u32;
    let mut a = 1;
    while a <= CAP {
        t[a + 1] = t[a] + binom(n + a as u32 - 1, a as u32) * CUM_OTHER[CAP - a];
        a += 1;
    }
    t
};

/// Sets with at least one royal; royal-less sets are never mates and have no index.
pub const SET_COUNT: usize = ROYAL_BASE[CAP + 1] as usize;

/// [`DEAD_BYTES`] of one bit per set, set when no mate is possible. Then a hash of
/// the sets where Black has only royals and mate is possible, but White can never
/// force it (`forced_gen` proved it): a `u32` multiplier, then buckets of
/// [`HELPMATE_SLOTS`] `u32` set indices padded with `u32::MAX`, all little-endian.
static TABLE: &[u8] = include_bytes!("insuffmat_unbounded.bin");
pub const DEAD_BYTES: usize = SET_COUNT.div_ceil(8);
pub const HELPMATE_BUCKET_BITS: u32 = 7;
pub const HELPMATE_SLOTS: usize = 8;
pub const TABLE_BYTES: usize = DEAD_BYTES + 4 + (HELPMATE_SLOTS << HELPMATE_BUCKET_BITS) * 4;

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

/// The index of a set with 1 to [`CAP`] symbols, at least one royal, in any order.
#[inline]
pub fn set_index(symbols: &[u8]) -> usize {
    debug_assert!(symbols.len() <= CAP);
    let mut royals = [0u8; CAP];
    let mut others = [0u8; CAP];
    let (mut a, mut b) = (0, 0);
    for &s in symbols {
        let (color, kind) = (s / KINDS, s % KINDS);
        if kind < ROYAL_KINDS {
            royals[a] = color * ROYAL_KINDS + kind;
            a += 1;
        } else {
            others[b] = color * OTHER_KINDS + kind - ROYAL_KINDS;
            b += 1;
        }
    }
    debug_assert!(a >= 1);
    sort_small(&mut royals[..a]);
    sort_small(&mut others[..b]);
    let mut royal_rank = 0;
    for (i, &c) in royals[..a].iter().enumerate() {
        royal_rank += BIN[i][c as usize];
    }
    let mut other_rank = if b == 0 { 0 } else { CUM_OTHER[b - 1] };
    for (i, &c) in others[..b].iter().enumerate() {
        other_rank += BIN[i][c as usize];
    }
    (ROYAL_BASE[a] + royal_rank * CUM_OTHER[CAP - a] + other_rank) as usize
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
    if !symbols.iter().any(|&s| is_royal_symbol(s)) {
        return true;
    }
    dead_bit(set_index(symbols))
}

#[inline]
fn dead_bit(i: usize) -> bool {
    TABLE[i >> 3] >> (i & 7) & 1 == 1
}

#[inline]
fn helpmate_only(i: usize) -> bool {
    let mul = u32::from_le_bytes(TABLE[DEAD_BYTES..DEAD_BYTES + 4].try_into().unwrap());
    let bucket = (i as u32).wrapping_mul(mul) >> (32 - HELPMATE_BUCKET_BITS);
    let base = DEAD_BYTES + 4 + bucket as usize * HELPMATE_SLOTS * 4;
    let slots: &[u8; HELPMATE_SLOTS * 4] = TABLE[base..base + HELPMATE_SLOTS * 4].try_into().unwrap();
    // Every slot is compared, so the cost is the same for every set.
    slots.chunks_exact(4).fold(false, |hit, w| hit | (u32::from_le_bytes(w.try_into().unwrap()) == i as u32))
}

/// Whether the attacking side's pieces in `symbols` (White's when `white_attacks`)
/// can never force mate against the other side's royals, which are all the other
/// side holds: no mate is possible, or only with the defender's help.
#[inline]
pub fn helpless(symbols: &[u8], white_attacks: bool) -> bool {
    if symbols.len() > CAP {
        return false;
    }
    if !symbols.iter().any(|&s| is_royal_symbol(s)) {
        return true;
    }
    let mut oriented = [0u8; CAP];
    for (o, &s) in oriented.iter_mut().zip(symbols) {
        *o = if white_attacks { s } else { (s + KINDS) % SYMBOLS };
    }
    let i = set_index(&oriented[..symbols.len()]);
    dead_bit(i) || helpmate_only(i)
}

/// Calls `f` with every set of at least one royal, smallest sets first.
pub fn for_each_set(mut f: impl FnMut(&[u8])) {
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
    for total in 1..=CAP {
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

    #[test]
    fn index_is_a_bijection() {
        let mut seen = vec![false; SET_COUNT];
        let mut n = 0;
        for_each_set(|s| {
            let i = set_index(s);
            assert!(!seen[i], "{s:?}");
            seen[i] = true;
            n += 1;
        });
        assert_eq!(n, SET_COUNT);
        assert_eq!(SET_COUNT, 784_541);
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
    fn helpmate_only_armies() {
        assert!(helpless(&set("K,GU,GU vs k"), true));
        assert!(helpless(&set("k vs K,GU,GU"), false));
        assert!(!helpless(&set("K,R,R vs k"), true));
        assert!(!helpless(&set("K,AM vs k"), true));
        assert!(helpless(&set("K,Q vs k"), true));
    }
}
