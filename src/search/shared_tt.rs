use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, value_from_tt, value_to_tt,
};
use super::{INFINITY, MATE_SCORE};

const ENTRIES_PER_BUCKET: usize = 4;
const NO_MOVE_SENTINEL: i64 = i64::MIN;

// ============================================================================
// Compact Move Representation (Atomic Friendly)
// ============================================================================

/// Compact move for TT storage. Uses i64 coordinates (required for infinite chess).
/// Size: 40 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTMove {
    pub from_x: i64,
    pub from_y: i64,
    pub to_x: i64,
    pub to_y: i64,
    pub piece_type: u8,
    pub piece_color: u8,
    pub promotion: u8, // 0 = none, else PieceType as u8
}

impl TTMove {
    /// Create a "no move" sentinel
    #[inline]
    pub const fn none() -> Self {
        TTMove {
            from_x: NO_MOVE_SENTINEL,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        }
    }

    /// Check if this represents a valid move
    #[inline]
    pub fn is_some(&self) -> bool {
        self.from_x != NO_MOVE_SENTINEL
    }

    /// Create from a Move
    #[inline]
    pub fn from_move(m: &Move) -> Self {
        TTMove {
            from_x: m.from.x,
            from_y: m.from.y,
            to_x: m.to.x,
            to_y: m.to.y,
            piece_type: m.piece.piece_type() as u8,
            piece_color: m.piece.color() as u8,
            promotion: m.promotion.map_or(0, |p| p as u8),
        }
    }

    /// Convert back to Option<Move> for use in search
    #[inline]
    pub fn to_move(self) -> Option<Move> {
        if !self.is_some() {
            return None;
        }

        let piece_type = PieceType::from_u8(self.piece_type);
        let color = PlayerColor::from_u8(self.piece_color);
        let piece = Piece::new(piece_type, color);

        let promotion = if self.promotion == 0 {
            None
        } else {
            // SAFE: Pieces are valid u8s [0, PieceType::MAX]
            Some(unsafe { std::mem::transmute::<u8, PieceType>(self.promotion) })
        };

        Some(Move {
            from: Coordinate {
                x: self.from_x,
                y: self.from_y,
            },
            to: Coordinate {
                x: self.to_x,
                y: self.to_y,
            },
            piece,
            promotion,
            rook_coord: None,
        })
    }
}
// ============================================================================
// TT Entry (Lock-Free Storage - Like Stockfish)
// ============================================================================

use std::cell::UnsafeCell;

/// Lock-free Transposition Table entry.
/// Like Stockfish, we use completely racy read/write access for maximum speed.
/// No atomics, no synchronization - just raw memory access.
/// This is safe because:
/// 1. Torn reads are detected via key mismatch
/// 2. Invalid data is handled gracefully by the search
/// 3. The trade-off (occasional wrong data) is worth the massive speedup
#[repr(C)]
pub struct TTEntry {
    /// Full 64-bit hash key (for verification)
    key: UnsafeCell<u64>,
    /// Packed: score (32) | eval (32)
    score_eval: UnsafeCell<u64>,
    /// Packed: depth (8) | gen_bound (8) | padding (48)
    meta: UnsafeCell<u64>,
    /// Move data (5 words = 40 bytes)
    move_words: [UnsafeCell<u64>; 5],
}

// SAFETY: TTEntry is designed for racy multi-threaded access like Stockfish.
// Torn reads are acceptable and handled via key verification.
unsafe impl Sync for TTEntry {}
unsafe impl Send for TTEntry {}

impl TTEntry {
    /// Create an empty/invalid entry
    pub fn empty() -> Self {
        TTEntry {
            key: UnsafeCell::new(0),
            score_eval: UnsafeCell::new(0),
            meta: UnsafeCell::new(0),
            move_words: [
                UnsafeCell::new(NO_MOVE_SENTINEL as u64),
                UnsafeCell::new(0),
                UnsafeCell::new(0),
                UnsafeCell::new(0),
                UnsafeCell::new(0),
            ],
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        unsafe {
            std::ptr::read_volatile(self.key.get()) == 0
                && std::ptr::read_volatile(self.meta.get()) == 0
        }
    }

    /// Read entry using raw memory access (like Stockfish).
    /// No synchronization - accepts torn reads.
    #[inline]
    pub fn read(&self, hash: u64) -> Option<(i32, i32, u8, u8, Option<Move>)> {
        unsafe {
            // Fast path: check key first
            let key = std::ptr::read_volatile(self.key.get());
            if key != hash {
                return None;
            }

            let score_eval = std::ptr::read_volatile(self.score_eval.get());
            let meta = std::ptr::read_volatile(self.meta.get());

            let depth = (meta >> 56) as u8;
            let gen_bound = ((meta >> 48) & 0xFF) as u8;
            let score = (score_eval >> 32) as i32;
            let eval = (score_eval & 0xFFFFFFFF) as i32;

            // Skip move loading for depth 0 entries
            if depth == 0 {
                return Some((score, eval, depth, gen_bound, None));
            }

            let m_words = [
                std::ptr::read_volatile(self.move_words[0].get()),
                std::ptr::read_volatile(self.move_words[1].get()),
                std::ptr::read_volatile(self.move_words[2].get()),
                std::ptr::read_volatile(self.move_words[3].get()),
                std::ptr::read_volatile(self.move_words[4].get()),
            ];

            let tt_move: TTMove = std::mem::transmute(m_words);
            Some((score, eval, depth, gen_bound, tt_move.to_move()))
        }
    }

    /// Write entry using raw memory access (like Stockfish).
    /// No synchronization - accepts racy writes.
    #[inline]
    pub fn write(
        &self,
        hash: u64,
        score: i32,
        eval: i32,
        depth: u8,
        gen_bound: u8,
        tt_move: TTMove,
    ) {
        let score_eval = ((score as u64) << 32) | (eval as u32 as u64);
        let meta = ((depth as u64) << 56) | ((gen_bound as u64) << 48);
        let m_words: [u64; 5] = unsafe { std::mem::transmute(tt_move) };

        unsafe {
            std::ptr::write_volatile(self.key.get(), hash);
            std::ptr::write_volatile(self.score_eval.get(), score_eval);
            std::ptr::write_volatile(self.meta.get(), meta);
            std::ptr::write_volatile(self.move_words[0].get(), m_words[0]);
            std::ptr::write_volatile(self.move_words[1].get(), m_words[1]);
            std::ptr::write_volatile(self.move_words[2].get(), m_words[2]);
            std::ptr::write_volatile(self.move_words[3].get(), m_words[3]);
            std::ptr::write_volatile(self.move_words[4].get(), m_words[4]);
        }
    }

    /// Raw key read for replacement logic
    #[inline]
    pub fn raw_key(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.key.get()) }
    }

    /// Raw meta read for replacement logic
    #[inline]
    pub fn raw_meta(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.meta.get()) }
    }

    /// Clear this entry (for TT reset)
    #[inline]
    pub fn clear(&self) {
        unsafe {
            std::ptr::write_volatile(self.key.get(), 0);
            std::ptr::write_volatile(self.meta.get(), 0);
        }
    }

    #[inline]
    pub fn flag(gen_bound: u8) -> TTFlag {
        TTFlag::from_u8(gen_bound)
    }

    #[inline]
    pub fn is_pv(gen_bound: u8) -> bool {
        (gen_bound & 0x04) != 0
    }

    #[inline]
    pub fn generation(gen_bound: u8) -> u8 {
        gen_bound >> 3
    }

    #[inline]
    pub fn pack_gen_bound(generation: u8, is_pv: bool, flag: TTFlag) -> u8 {
        (generation << 3) | (if is_pv { 0x04 } else { 0 }) | (flag as u8)
    }
}

// ============================================================================
// TT Bucket (Cluster of Entries)
// ============================================================================

/// A bucket/cluster containing multiple TT entries.
pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
}

impl TTBucket {
    pub fn empty() -> Self {
        TTBucket {
            entries: [
                TTEntry::empty(),
                TTEntry::empty(),
                TTEntry::empty(),
                TTEntry::empty(),
            ],
        }
    }
}

/// Transposition Table with bucket-based collision handling.
/// Thread-safe for Lazy SMP using lock-free architecture.
pub struct SharedTranspositionTable {
    buckets: Vec<TTBucket>,
    /// Bitmask for indexing
    mask: usize,
    /// Current generation (wrapped in UnsafeCell for racy access like Stockfish)
    generation: UnsafeCell<u8>,
}

unsafe impl Sync for SharedTranspositionTable {}
unsafe impl Send for SharedTranspositionTable {}

impl SharedTranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        let size_mb = size_mb.min(64);

        let bytes = size_mb.max(1) * 1024 * 1024;
        let bucket_size = std::mem::size_of::<TTBucket>();
        let num_buckets = (bytes / bucket_size).max(1);

        let mut cap_pow2 = 1usize;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
        }

        let mut buckets = Vec::with_capacity(cap_pow2);
        for _ in 0..cap_pow2 {
            buckets.push(TTBucket::empty());
        }

        SharedTranspositionTable {
            buckets,
            mask: cap_pow2 - 1,
            generation: UnsafeCell::new(1),
        }
    }

    /// Get the hash for the current position
    #[inline]
    pub fn generate_hash(game: &GameState) -> u64 {
        game.hash
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buckets.len() * ENTRIES_PER_BUCKET
    }

    #[inline]
    pub fn used_entries(&self) -> usize {
        // Approximate used entries by sampling (slow to count all)
        (self.hashfull() as usize * self.capacity()) / 1000
    }

    #[inline]
    pub fn fill_permille(&self) -> u32 {
        self.hashfull()
    }

    /// Calculate hash table occupancy in permille (0-1000).
    /// Samples first 1000 buckets like Stockfish to avoid expensive full scans.
    pub fn hashfull(&self) -> u32 {
        let mut occupied = 0;
        let sample_size = self.buckets.len().min(1000);
        let generation = unsafe { *self.generation.get() };

        for i in 0..sample_size {
            for entry in &self.buckets[i].entries {
                let meta = entry.raw_meta();
                let e_gen_bound = ((meta >> 48) & 0xFF) as u8;
                let age_diff = (generation.wrapping_sub(TTEntry::generation(e_gen_bound))) & 0x1F;

                if meta != 0 && age_diff == 0 {
                    occupied += 1;
                }
            }
        }

        if sample_size == 0 {
            return 0;
        }
        (occupied * 1000 / (sample_size * ENTRIES_PER_BUCKET)) as u32
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Prefetch the TT bucket for the given hash into L1 cache.
    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = self.bucket_index(hash);
        let ptr = self.buckets.as_ptr().wrapping_add(idx) as *const i8;
        // SAFETY: ptr points into a valid, allocated slice
        unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
    }

    #[inline]
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    pub fn prefetch_entry(&self, _hash: u64) {}

    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        let hash = params.hash;
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];

        for entry in &bucket.entries {
            // Atomic read with verification
            if let Some((score, eval, depth, gen_bound, best_move)) = entry.read(hash) {
                // Found a matching entry
                let score =
                    value_from_tt(score, params.ply, params.rule50_count, params.rule_limit);
                let flag = TTEntry::flag(gen_bound);
                let is_pv = TTEntry::is_pv(gen_bound);

                let mut cutoff_score = INFINITY + 1;
                if depth as usize >= params.depth {
                    let usable = match flag {
                        TTFlag::Exact => true,
                        TTFlag::LowerBound if score >= params.beta => true,
                        TTFlag::UpperBound if score <= params.alpha => true,
                        _ => false,
                    };
                    if usable {
                        cutoff_score = score;
                    }
                }

                return Some(TTProbeResult {
                    cutoff_score,
                    tt_score: score,
                    eval,
                    depth,
                    flag,
                    is_pv,
                    best_move,
                });
            }
        }
        None
    }

    /// Probe only for the best move (used for PV extraction)
    pub fn probe_move(&self, hash: u64) -> Option<Move> {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];

        for entry in &bucket.entries {
            if let Some((_, _, _, _, best_move)) = entry.read(hash) {
                return best_move;
            }
        }
        None
    }

    pub fn store(&self, params: &TTStoreParams) {
        let hash = params.hash;
        let adjusted_score = value_to_tt(params.score, params.ply);
        let idx = self.bucket_index(hash);
        let generation = unsafe { *self.generation.get() };
        let bucket = &self.buckets[idx];

        let mut replace_idx = 0;
        let mut worst_score = i32::MAX;

        for (i, entry) in bucket.entries.iter().enumerate() {
            // Atomic read for comparison/replacement logic
            if let Some((_, old_eval, old_depth, old_gen_bound, old_move)) = entry.read(hash) {
                let move_to_store = if params.best_move.is_some() {
                    TTMove::from_move(&params.best_move.as_ref().unwrap())
                } else {
                    old_move.map_or(TTMove::none(), |m| TTMove::from_move(&m))
                };

                let eval_to_store = if params.static_eval != INFINITY + 1 {
                    params.static_eval
                } else {
                    old_eval
                };
                let _old_flag = TTEntry::flag(old_gen_bound);
                let old_gen = TTEntry::generation(old_gen_bound);

                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };
                let relative_age = (generation.wrapping_sub(old_gen)) & 0x1F;

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (old_depth as i32 - 4)
                    || relative_age != 0
                    || params.depth == 0
                {
                    entry.write(
                        hash,
                        adjusted_score,
                        eval_to_store,
                        params.depth as u8,
                        TTEntry::pack_gen_bound(generation, params.is_pv, params.flag),
                        move_to_store,
                    );
                }
                return;
            }

            // Entry didn't match hash or was torn. Use standard replacement metric.
            let key = entry.raw_key();
            let meta = entry.raw_meta();
            let e_depth = (meta >> 56) as u8;
            let e_gen_bound = ((meta >> 48) & 0xFF) as u8;

            let mut entry_priority = e_depth as i32;
            let age_diff = (generation.wrapping_sub(TTEntry::generation(e_gen_bound))) & 0x1F;
            entry_priority -= (age_diff as i32) * 2;
            if TTEntry::flag(e_gen_bound) == TTFlag::Exact || TTEntry::is_pv(e_gen_bound) {
                entry_priority += 2;
            }
            if key == 0 {
                entry_priority = i32::MIN;
            }

            if entry_priority < worst_score {
                worst_score = entry_priority;
                replace_idx = i;
            }
        }

        // Prepare new entry
        let move_to_store = params
            .best_move
            .as_ref()
            .map_or(TTMove::none(), TTMove::from_move);
        let new_priority = params.depth as i32
            + (if params.flag == TTFlag::Exact || params.is_pv {
                2
            } else {
                0
            });

        if new_priority >= worst_score {
            bucket.entries[replace_idx].write(
                hash,
                adjusted_score,
                params.static_eval,
                params.depth as u8,
                TTEntry::pack_gen_bound(generation, params.is_pv, params.flag),
                move_to_store,
            );
        }
    }

    pub fn increment_age(&self) {
        unsafe {
            let old = *self.generation.get();
            let mut new = (old + 1) & 0x1F;
            if new == 0 {
                new = 1;
            }
            *self.generation.get() = new;
        }
    }

    pub fn clear(&self) {
        for bucket in &self.buckets {
            for entry in &bucket.entries {
                entry.clear();
            }
        }
        unsafe {
            *self.generation.get() = 1;
        }
    }
}

// ============================================================================
// Re-export old types for API compatibility
// ============================================================================

// TTFlag is already defined above with the same variants (compatible)

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Verify our compact structs are the expected sizes
        assert_eq!(
            std::mem::size_of::<TTMove>(),
            40,
            "TTMove should be 40 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTEntry>(),
            72,
            "TTEntry should be 72 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTBucket>(),
            72 * ENTRIES_PER_BUCKET,
            "TTBucket should be 288 bytes (4 x 72)"
        );
    }

    #[test]
    fn test_tt_basic_operations() {
        let tt = SharedTranspositionTable::new(1); // 1 MB table

        // Store and probe
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

        let result = tt.probe(&TTProbeParams {
            hash,
            alpha: -1000,
            beta: 1000,
            depth: 5,
            ply: 0,
            rule50_count: 0,
            rule_limit: 100,
        });
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.cutoff_score, 100);
        assert_eq!(res.eval, 90);
    }

    #[test]
    fn test_tt_gen_bound_packing() {
        // Test that generation and bound are packed correctly
        for r#gen in [0u8, 1, 31] {
            for flag in [
                TTFlag::None,
                TTFlag::Exact,
                TTFlag::LowerBound,
                TTFlag::UpperBound,
            ] {
                // Testing with is_pv = true
                let packed_pv = TTEntry::pack_gen_bound(r#gen, true, flag);
                assert_eq!(TTEntry::generation(packed_pv), r#gen & 0x1F);
                assert_eq!(TTEntry::flag(packed_pv), flag);
                assert!(TTEntry::is_pv(packed_pv));

                // Testing with is_pv = false
                let packed_nopv = TTEntry::pack_gen_bound(r#gen, false, flag);
                assert_eq!(TTEntry::generation(packed_nopv), r#gen & 0x1F);
                assert_eq!(TTEntry::flag(packed_nopv), flag);
                assert!(!TTEntry::is_pv(packed_nopv));
            }
        }
    }

    #[test]
    fn test_ttmove_sentinel() {
        let none = TTMove::none();
        assert!(!none.is_some());
        assert!(none.to_move().is_none());

        // A real move should be_some
        let real = TTMove {
            from_x: 0,
            from_y: 0,
            to_x: 1,
            to_y: 1,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        };
        assert!(real.is_some());
    }
}
