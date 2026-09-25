//! Special rights (castling and pawn double-push) as a bitboard over the squares
//! around the origin. A right only ever sits on a piece's starting square, and every
//! variant starts well inside the window; squares outside it use an ordered set, so
//! iteration is in (rank, file) order on every platform.

use crate::board::Coordinate;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Files and ranks covered by the bitboard: [-HALF, HALF).
const HALF: i64 = 64;
const SIDE: usize = 2 * HALF as usize;
const ROW_WORDS: usize = SIDE / 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialRights {
    bits: [u64; SIDE * ROW_WORDS],
    /// Bit per rank that holds any right, so full iteration skips empty ranks.
    ranks: [u64; SIDE / 64],
    /// Rights outside the window, keyed (y, x) so they iterate rank by rank.
    far: BTreeSet<(i64, i64)>,
    len: usize,
}

impl Default for SpecialRights {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecialRights {
    pub const fn new() -> Self {
        Self {
            bits: [0; SIDE * ROW_WORDS],
            ranks: [0; SIDE / 64],
            far: BTreeSet::new(),
            len: 0,
        }
    }

    /// (rank index, file index) inside the window, or None.
    #[inline(always)]
    fn slot(x: i64, y: i64) -> Option<(usize, usize)> {
        let fx = x.wrapping_add(HALF) as u64;
        let fy = y.wrapping_add(HALF) as u64;
        (fx < SIDE as u64 && fy < SIDE as u64).then_some((fy as usize, fx as usize))
    }

    #[inline(always)]
    pub fn contains(&self, c: &Coordinate) -> bool {
        match Self::slot(c.x, c.y) {
            Some((r, f)) => (self.bits[r * ROW_WORDS + f / 64] >> (f % 64)) & 1 != 0,
            None => !self.far.is_empty() && self.far.contains(&(c.y, c.x)),
        }
    }

    /// Adds the right; true if it was not already held.
    #[inline]
    pub fn insert(&mut self, c: Coordinate) -> bool {
        let added = match Self::slot(c.x, c.y) {
            Some((r, f)) => {
                let (word, bit) = (&mut self.bits[r * ROW_WORDS + f / 64], 1u64 << (f % 64));
                let added = *word & bit == 0;
                *word |= bit;
                self.ranks[r / 64] |= 1 << (r % 64);
                added
            }
            None => self.far.insert((c.y, c.x)),
        };
        self.len += added as usize;
        added
    }

    /// Drops the right; true if it was held.
    #[inline]
    pub fn remove(&mut self, c: &Coordinate) -> bool {
        let removed = match Self::slot(c.x, c.y) {
            Some((r, f)) => {
                let row = r * ROW_WORDS;
                let (word, bit) = (&mut self.bits[row + f / 64], 1u64 << (f % 64));
                let removed = *word & bit != 0;
                *word &= !bit;
                if self.bits[row..row + ROW_WORDS].iter().all(|&w| w == 0) {
                    self.ranks[r / 64] &= !(1 << (r % 64));
                }
                removed
            }
            None => self.far.remove(&(c.y, c.x)),
        };
        self.len -= removed as usize;
        removed
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Keeps only the rights `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(&Coordinate) -> bool) {
        let dropped: Vec<Coordinate> = self.iter().filter(|c| !keep(c)).collect();
        for c in &dropped {
            self.remove(c);
        }
    }

    /// Rights on rank `y`, by ascending file.
    pub fn iter_rank(&self, y: i64) -> impl Iterator<Item = Coordinate> + '_ {
        let (words, inside) = match Self::slot(0, y) {
            Some((r, _)) => {
                let mut w = [0u64; ROW_WORDS];
                w.copy_from_slice(&self.bits[r * ROW_WORDS..(r + 1) * ROW_WORDS]);
                (w, true)
            }
            None => ([0; ROW_WORDS], false),
        };
        let dense = words.into_iter().enumerate().flat_map(move |(i, mut w)| {
            std::iter::from_fn(move || {
                (w != 0).then(|| {
                    let f = w.trailing_zeros() as i64;
                    w &= w - 1;
                    Coordinate::new(i as i64 * 64 + f - HALF, y)
                })
            })
        });
        // On a rank inside the window, far rights lie left or right of the dense files.
        let (left, right) = if inside {
            (
                self.far.range((y, i64::MIN)..(y, -HALF)),
                self.far.range((y, HALF)..=(y, i64::MAX)),
            )
        } else {
            (self.far.range((y, i64::MIN)..=(y, i64::MAX)), self.far.range((y, 0)..(y, 0)))
        };
        let far = |&(y, x): &(i64, i64)| Coordinate::new(x, y);
        left.map(far).chain(dense).chain(right.map(far))
    }

    /// Every right, rank by rank and by ascending file within a rank.
    pub fn iter(&self) -> impl Iterator<Item = Coordinate> + '_ {
        let mut ranks: Vec<i64> = Vec::new();
        for (i, &word) in self.ranks.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                ranks.push(i as i64 * 64 + w.trailing_zeros() as i64 - HALF);
                w &= w - 1;
            }
        }
        ranks.extend(self.far.iter().map(|&(y, _)| y));
        ranks.sort_unstable();
        ranks.dedup();
        ranks.into_iter().flat_map(move |y| self.iter_rank(y))
    }
}

impl Serialize for SpecialRights {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for SpecialRights {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let coords = Vec::<Coordinate>::deserialize(deserializer)?;
        let mut rights = SpecialRights::new();
        for c in coords {
            rights.insert(c);
        }
        Ok(rights)
    }
}

impl FromIterator<Coordinate> for SpecialRights {
    fn from_iter<I: IntoIterator<Item = Coordinate>>(iter: I) -> Self {
        let mut rights = SpecialRights::new();
        for c in iter {
            rights.insert(c);
        }
        rights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hash_set_inside_and_outside_the_window() {
        let mut rights = SpecialRights::new();
        let mut reference = std::collections::HashSet::new();
        let mut seed = 0x9E3779B97F4A7C15u64;
        for i in 0..20_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let span = if i % 7 == 0 { 1_000_000_000_000 } else { 80 };
            let x = (seed >> 20) as i64 % span - span / 2;
            let y = (seed >> 40) as i64 % span - span / 2;
            let c = Coordinate::new(x, y);
            if seed >> 63 == 0 {
                assert_eq!(rights.insert(c), reference.insert(c));
            } else {
                assert_eq!(rights.remove(&c), reference.remove(&c));
            }
            assert_eq!(rights.len(), reference.len());
        }
        for &c in &reference {
            assert!(rights.contains(&c));
        }
        let listed: Vec<_> = rights.iter().collect();
        assert_eq!(listed.len(), reference.len());
        assert!(listed.windows(2).all(|w| (w[0].y, w[0].x) < (w[1].y, w[1].x)));
        let ranks: std::collections::BTreeSet<i64> = reference.iter().map(|c| c.y).collect();
        for y in ranks.into_iter().take(200).chain([-70, -64, -1, 0, 63, 64, 79]) {
            let row: Vec<_> = rights.iter_rank(y).collect();
            assert!(row.windows(2).all(|w| w[0].x < w[1].x), "rank {y} out of order");
            assert_eq!(row.len(), reference.iter().filter(|c| c.y == y).count());
        }
    }
}
