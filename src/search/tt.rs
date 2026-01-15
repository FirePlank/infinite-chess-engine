use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::Move;

use super::tt_defs::{TTFlag, TTProbeParams, TTStoreParams, value_from_tt, value_to_tt};
use super::{INFINITY, MATE_SCORE};

const ENTRIES_PER_BUCKET: usize = 4;
const NO_MOVE_SENTINEL: i32 = i32::MIN;

// ============================================================================
// TT Entry (32 bytes)
// ============================================================================

/// Transposition Table entry - optimized for memory efficiency and cache alignment.
///
/// Layout (32 bytes total):
/// - score: i32       - Evaluation score (4 bytes)
/// - eval: i32        - Static evaluation (4 bytes)
/// - from_x: i32      - Move from X (4 bytes)
/// - from_y: i32      - Move from Y (4 bytes)
/// - to_x: i32        - Move to X (4 bytes)
/// - to_y: i32        - Move to Y (4 bytes)
/// - key32: u32       - Signature key (32 bits, bits above the index) (4 bytes)
/// - depth: u8        - Search depth (1 byte)
/// - gen_bound: u8    - Generation (5 bits) + PV (1 bit) + Bound (2 bits) (1 byte)
/// - move_info: u16   - Packed move info: Type (5), Color (2), Promotion (5) (2 bytes)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTEntry {
    pub score: i32,
    pub eval: i32,
    pub from_x: i32,
    pub from_y: i32,
    pub to_x: i32,
    pub to_y: i32,
    pub key32: u32,
    pub depth: u8,
    pub gen_bound: u8,
    pub move_info: u16,
}

impl TTEntry {
    /// Create an empty/invalid entry
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            score: 0,
            eval: INFINITY + 1,
            from_x: NO_MOVE_SENTINEL,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            key32: 0,
            depth: 0,
            gen_bound: 0,
            move_info: 0,
        }
    }

    /// Check if entry is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.from_x == NO_MOVE_SENTINEL && self.key32 == 0
    }

    /// Extract the bound type from gen_bound
    #[inline]
    pub fn flag(&self) -> TTFlag {
        TTFlag::from_u8(self.gen_bound)
    }

    /// Extract the is_pv flag
    #[inline]
    pub fn is_pv(&self) -> bool {
        (self.gen_bound & 0x04) != 0
    }

    /// Extract the generation from gen_bound
    #[inline]
    pub fn generation(&self) -> u8 {
        self.gen_bound >> 3
    }

    /// Create packed gen_bound from generation, is_pv and flag
    #[inline]
    fn pack_gen_bound(generation: u8, is_pv: bool, flag: TTFlag) -> u8 {
        ((generation & 0x1F) << 3) | (if is_pv { 0x04 } else { 0 }) | (flag as u8 & 0x03)
    }

    /// Get the best move as Option<Move>
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        if self.from_x == NO_MOVE_SENTINEL {
            return None;
        }

        let info = self.move_info;
        let pt_val = (info & 0x1F) as u8;
        let cl_val = ((info >> 5) & 0x03) as u8;
        let pr_val = ((info >> 7) & 0x1F) as u8;

        let piece_type = PieceType::from_u8(pt_val);
        let color = PlayerColor::from_u8(cl_val);
        let piece = Piece::new(piece_type, color);

        let promotion = if pr_val == 0 {
            None
        } else {
            Some(PieceType::from_u8(pr_val))
        };

        Some(Move {
            from: Coordinate {
                x: self.from_x as i64,
                y: self.from_y as i64,
            },
            to: Coordinate {
                x: self.to_x as i64,
                y: self.to_y as i64,
            },
            piece,
            promotion,
            rook_coord: None,
        })
    }
}

// ============================================================================
// TT Bucket (Cluster of Entries)
// ============================================================================

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
}

impl TTBucket {
    #[inline]
    pub const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
        }
    }
}

// ============================================================================
// Transposition Table
// ============================================================================

use std::cell::UnsafeCell;

pub struct LocalTranspositionTable {
    buckets: UnsafeCell<Vec<TTBucket>>,
    mask: usize,
    index_bits: u32,
    generation: UnsafeCell<u8>,
    used: UnsafeCell<usize>,
}

unsafe impl Sync for LocalTranspositionTable {}

impl LocalTranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        let size_mb = size_mb.min(64);

        let bytes = size_mb.max(1) * 1024 * 1024;
        let bucket_size = std::mem::size_of::<TTBucket>();
        let num_buckets = (bytes / bucket_size).max(1);

        let mut cap_pow2 = 1usize;
        let mut index_bits = 0u32;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
            index_bits += 1;
        }

        LocalTranspositionTable {
            buckets: UnsafeCell::new(vec![TTBucket::empty(); cap_pow2]),
            mask: cap_pow2 - 1,
            index_bits,
            generation: UnsafeCell::new(1),
            used: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        let buckets = unsafe { &*self.buckets.get() };
        buckets.len() * ENTRIES_PER_BUCKET
    }

    #[inline]
    pub fn used_entries(&self) -> usize {
        unsafe { *self.used.get() }
    }

    #[inline]
    pub fn fill_permille(&self) -> u32 {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0;
        }
        ((self.used_entries() as u64 * 1000) / capacity as u64) as u32
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Signature for verification. Uses bits right above the index to minimize collisions.
    #[inline]
    fn hash_signature(&self, hash: u64) -> u32 {
        (hash >> self.index_bits) as u32
    }

    /// Prefetch the TT bucket for the given hash into L1 cache.
    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = self.bucket_index(hash);
        let buckets = unsafe { &*self.buckets.get() };
        let ptr = buckets.as_ptr().wrapping_add(idx) as *const i8;
        // SAFETY: ptr points into a valid, allocated slice
        unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
    }

    #[inline]
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    pub fn prefetch_entry(&self, _hash: u64) {}

    pub fn probe(&self, params: &TTProbeParams) -> Option<(i32, i32, Option<Move>, bool)> {
        let signature = self.hash_signature(params.hash);
        let idx = self.bucket_index(params.hash);
        let buckets = unsafe { &*self.buckets.get() };
        let bucket = &buckets[idx];

        for entry in &bucket.entries {
            if entry.key32 != signature || entry.is_empty() {
                continue;
            }

            let best_move = entry.best_move();

            if entry.depth as usize >= params.depth {
                let score = value_from_tt(
                    entry.score,
                    params.ply,
                    params.rule50_count,
                    params.rule_limit,
                );
                let usable_score = match entry.flag() {
                    TTFlag::Exact => Some(score),
                    TTFlag::LowerBound if score >= params.beta => Some(score),
                    TTFlag::UpperBound if score <= params.alpha => Some(score),
                    _ => None,
                };

                if let Some(s) = usable_score {
                    return Some((s, entry.eval, best_move, entry.is_pv()));
                }
            }

            return Some((INFINITY + 1, entry.eval, best_move, entry.is_pv()));
        }
        None
    }

    pub fn probe_for_singular(
        &self,
        hash: u64,
        ply: usize,
    ) -> Option<(TTFlag, u8, i32, i32, Option<Move>, bool)> {
        let signature = self.hash_signature(hash);
        let idx = self.bucket_index(hash);
        let buckets = unsafe { &*self.buckets.get() };
        let bucket = &buckets[idx];

        for entry in &bucket.entries {
            if entry.key32 != signature || entry.is_empty() {
                continue;
            }

            let mut score = entry.score;
            if score > MATE_SCORE {
                score -= ply as i32;
            } else if score < -MATE_SCORE {
                score += ply as i32;
            }

            return Some((
                entry.flag(),
                entry.depth,
                score,
                entry.eval,
                entry.best_move(),
                entry.is_pv(),
            ));
        }
        None
    }

    pub fn store(&self, params: &TTStoreParams) {
        let signature = self.hash_signature(params.hash);
        let adjusted_score = value_to_tt(params.score, params.ply);
        let idx = self.bucket_index(params.hash);

        let (generation, buckets, used_ptr) = unsafe {
            (
                *self.generation.get(),
                &mut *self.buckets.get(),
                self.used.get(),
            )
        };

        let bucket = &mut buckets[idx];
        let mut replace_idx = 0;
        let mut worst_score = i32::MAX;

        for (i, entry) in bucket.entries.iter_mut().enumerate() {
            if entry.key32 == signature && !entry.is_empty() {
                let move_info = if let Some(m) = &params.best_move {
                    if m.from.x >= i32::MIN as i64
                        && m.from.x <= i32::MAX as i64
                        && m.from.y >= i32::MIN as i64
                        && m.from.y <= i32::MAX as i64
                        && m.to.x >= i32::MIN as i64
                        && m.to.x <= i32::MAX as i64
                        && m.to.y >= i32::MIN as i64
                        && m.to.y <= i32::MAX as i64
                    {
                        let pt = m.piece.piece_type() as u16;
                        let cl = m.piece.color() as u16;
                        let pr = m.promotion.map_or(0, |p| p as u16);
                        (pt & 0x1F) | ((cl & 0x03) << 5) | ((pr & 0x1F) << 7)
                    } else {
                        entry.move_info
                    }
                } else {
                    entry.move_info
                };

                let from_x = if params.best_move.is_some() {
                    params.best_move.as_ref().unwrap().from.x as i32
                } else {
                    entry.from_x
                };
                let from_y = if params.best_move.is_some() {
                    params.best_move.as_ref().unwrap().from.y as i32
                } else {
                    entry.from_y
                };
                let to_x = if params.best_move.is_some() {
                    params.best_move.as_ref().unwrap().to.x as i32
                } else {
                    entry.to_x
                };
                let to_y = if params.best_move.is_some() {
                    params.best_move.as_ref().unwrap().to.y as i32
                } else {
                    entry.to_y
                };

                let eval_to_store = if params.static_eval != INFINITY + 1 {
                    params.static_eval
                } else {
                    entry.eval
                };
                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };
                let age_diff = (generation.wrapping_sub(entry.generation())) & 0x1F;

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (entry.depth as i32 - 4)
                    || age_diff != 0
                    || params.depth == 0
                {
                    *entry = TTEntry {
                        score: adjusted_score,
                        eval: eval_to_store,
                        from_x,
                        from_y,
                        to_x,
                        to_y,
                        key32: signature,
                        depth: params.depth as u8,
                        gen_bound: TTEntry::pack_gen_bound(generation, params.is_pv, params.flag),
                        move_info,
                    };
                } else if entry.depth >= 5 && entry.flag() != TTFlag::Exact {
                    entry.depth = entry.depth.saturating_sub(1);
                }
                return;
            }

            let entry_priority = Self::calculate_replacement_score(entry, generation);
            if entry_priority < worst_score {
                worst_score = entry_priority;
                replace_idx = i;
            }
        }

        let mut move_info = 0;
        let mut from_x = NO_MOVE_SENTINEL;
        let mut from_y = 0;
        let mut to_x = 0;
        let mut to_y = 0;

        if let Some(m) = &params.best_move {
            if m.from.x >= i32::MIN as i64
                && m.from.x <= i32::MAX as i64
                && m.from.y >= i32::MIN as i64
                && m.from.y <= i32::MAX as i64
                && m.to.x >= i32::MIN as i64
                && m.to.x <= i32::MAX as i64
                && m.to.y >= i32::MIN as i64
                && m.to.y <= i32::MAX as i64
            {
                let pt = m.piece.piece_type() as u16;
                let cl = m.piece.color() as u16;
                let pr = m.promotion.map_or(0, |p| p as u16);
                move_info = (pt & 0x1F) | ((cl & 0x03) << 5) | ((pr & 0x1F) << 7);
                from_x = m.from.x as i32;
                from_y = m.from.y as i32;
                to_x = m.to.x as i32;
                to_y = m.to.y as i32;
            }
        }

        let new_entry = TTEntry {
            score: adjusted_score,
            eval: params.static_eval,
            from_x,
            from_y,
            to_x,
            to_y,
            key32: signature,
            depth: params.depth as u8,
            gen_bound: TTEntry::pack_gen_bound(generation, params.is_pv, params.flag),
            move_info,
        };

        if Self::calculate_replacement_score(&new_entry, generation) >= worst_score {
            if bucket.entries[replace_idx].is_empty() {
                unsafe { *used_ptr += 1 };
            }
            bucket.entries[replace_idx] = new_entry;
        }
    }

    #[inline]
    fn calculate_replacement_score(entry: &TTEntry, current_gen: u8) -> i32 {
        if entry.is_empty() {
            return i32::MIN;
        }
        let mut score = entry.depth as i32;
        score -= ((current_gen.wrapping_sub(entry.generation())) & 0x1F) as i32 * 2;
        if entry.flag() == TTFlag::Exact || entry.is_pv() {
            score += 2;
        }
        score
    }

    pub fn increment_age(&self) {
        unsafe {
            let next_gen = (*self.generation.get() + 1) & 0x1F;
            *self.generation.get() = if next_gen == 0 { 1 } else { next_gen };
        }
    }

    pub fn clear(&self) {
        let buckets = unsafe { &mut *self.buckets.get() };
        for b in buckets {
            *b = TTBucket::empty();
        }
        unsafe {
            *self.generation.get() = 1;
            *self.used.get() = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<TTEntry>(), 32);
        assert_eq!(std::mem::size_of::<TTBucket>(), 128);
    }

    #[test]
    fn test_tt_basic_operations() {
        let tt = LocalTranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0u64;
        tt.store(&TTStoreParams {
            hash,
            depth: 5,
            flag: TTFlag::Exact,
            score: 100,
            static_eval: 90,
            is_pv: true,
            best_move: None,
            ply: 0,
        });
        let res = tt.probe(&TTProbeParams {
            hash,
            alpha: -1000,
            beta: 1000,
            depth: 5,
            ply: 0,
            rule50_count: 0,
            rule_limit: 100,
        });
        assert!(res.is_some());
        let (s, e, _, _) = res.unwrap();
        assert_eq!(s, 100);
        assert_eq!(e, 90);
    }

    #[test]
    fn test_move_info_packing() {
        // Test a Knight (16) with Black color (2) and Queen promotion (8)
        let pt = PieceType::Knight;
        let cl = PlayerColor::Black;
        let pr = Some(PieceType::Queen);

        // Pack
        let info =
            (pt as u16 & 0x1F) | ((cl as u16 & 0x03) << 5) | ((pr.unwrap() as u16 & 0x1F) << 7);
        let mut entry = TTEntry::empty();
        entry.move_info = info;
        entry.from_x = 0; // Not empty

        let m = entry.best_move().unwrap();
        assert_eq!(m.piece.piece_type(), PieceType::Knight);
        assert_eq!(m.piece.color(), PlayerColor::Black);
        assert_eq!(m.promotion, Some(PieceType::Queen));
    }
}
