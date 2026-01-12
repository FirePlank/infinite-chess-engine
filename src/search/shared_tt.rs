//! Shared Transposition Table for Lazy SMP
//!
//! This module provides a lock-free transposition table that can be shared across
//! multiple Web Workers using SharedArrayBuffer. The design follows the Lazy SMP
//! approach where multiple threads search the same position with different move
//! orderings (via different history tables), sharing only the TT.
//!
//! ## WASM Considerations
//! - Uses AtomicU64 for lock-free read/write
//! - TT entries are packed into 64-bit atomic values for atomic operations
//! - Designed to work with SharedArrayBuffer passed from JavaScript
//! - Uses raw pointer to shared memory instead of owned Vec for true sharing

use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::Move;

use super::{INFINITY, MATE_SCORE};

// ============================================================================
// Packed Entry Format
// ============================================================================
//
// We pack TT data into two 64-bit atomics per entry:
//
// Word 0 (data):
//   - Bits 63-48: key16 (16 bits) - Hash verification
//   - Bits 47-40: depth (8 bits) - Search depth
//   - Bits 39-32: gen_bound (8 bits) - Generation + bound type
//   - Bits 31-0:  score (32 bits) - Evaluation score (as i32)
//
// Word 1 (move):
//   - Bits 63-48: from_x low 16 bits
//   - Bits 47-32: from_y low 16 bits
//   - Bits 31-16: to_x low 16 bits
//   - Bits 15-0:  to_y low 16 bits
//
// Word 2 (move high bits + piece info):
//   - Bits 63-48: from_x high 16 bits
//   - Bits 47-32: from_y high 16 bits
//   - Bits 31-16: to_x high 16 bits
//   - Bits 15-8:  to_y high 8 bits (remaining high bits)
//   - Bits 7-5:   piece_type (3 bits - up to 8 types, may need adjustment)
//   - Bits 4-3:   piece_color (2 bits)
//   - Bits 2-0:   promotion (3 bits - 0=none, else piece type)
//
// Actually, for simplicity and infinite chess i64 coords, let's use a simpler approach:
// 3 atomic u64s per entry (24 bytes per entry, simpler to handle)

/// Number of atomic u64s per entry
const WORDS_PER_ENTRY: usize = 3;

/// Entries per bucket for collision handling
const ENTRIES_PER_BUCKET: usize = 2;

/// Sentinel for no move
const NO_MOVE_SENTINEL: u64 = u64::MAX;

/// TT bound type (compatible with regular TT)
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum SharedTTFlag {
    None = 0,
    Exact = 1,
    LowerBound = 2,
    UpperBound = 3,
}

impl SharedTTFlag {
    #[inline]
    fn from_u8(v: u8) -> Self {
        match v & 0b11 {
            0 => SharedTTFlag::None,
            1 => SharedTTFlag::Exact,
            2 => SharedTTFlag::LowerBound,
            _ => SharedTTFlag::UpperBound,
        }
    }
}

// ============================================================================
// Shared Transposition Table (Owned Version)
// ============================================================================

/// Lock-free transposition table for Lazy SMP.
///
/// Uses atomic operations for thread-safe read/write without locks.
/// This version owns its data - for true cross-worker sharing, use SharedTTView.
pub struct SharedTT {
    /// Atomic storage: groups of WORDS_PER_ENTRY * ENTRIES_PER_BUCKET per bucket
    data: Vec<AtomicU64>,
    /// Number of buckets (kept for potential future use)
    #[allow(dead_code)]
    num_buckets: usize,
    /// Mask for bucket indexing (power of 2 - 1)
    mask: usize,
}

impl SharedTT {
    /// Create a new SharedTT with approximately `size_mb` megabytes.
    pub fn new(size_mb: usize) -> Self {
        // Cap for WASM
        #[cfg(target_arch = "wasm32")]
        let size_mb = size_mb.min(64);

        let bytes = size_mb.max(1) * 1024 * 1024;
        let bytes_per_bucket = WORDS_PER_ENTRY * ENTRIES_PER_BUCKET * 8; // 8 bytes per u64
        let num_buckets = (bytes / bytes_per_bucket).max(1);

        // Round to power of 2
        let mut cap_pow2 = 1usize;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
        }

        let total_words = cap_pow2 * ENTRIES_PER_BUCKET * WORDS_PER_ENTRY;
        let mut data = Vec::with_capacity(total_words);
        for _ in 0..total_words {
            data.push(AtomicU64::new(0));
        }

        SharedTT {
            data,
            num_buckets: cap_pow2,
            mask: cap_pow2 - 1,
        }
    }

    /// Get raw pointer to data (for passing to other threads)
    pub fn data_ptr(&mut self) -> *mut u64 {
        self.data.as_mut_ptr() as *mut u64
    }

    /// Get data length in words
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Get bucket index from hash
    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Get the offset into data array for a bucket
    #[inline]
    fn bucket_offset(&self, bucket_idx: usize) -> usize {
        bucket_idx * ENTRIES_PER_BUCKET * WORDS_PER_ENTRY
    }

    /// Extract key verification bits
    #[inline]
    fn key16(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    /// Pack entry word 0: key16 | depth | gen_bound | score
    #[inline]
    fn pack_word0(key16: u16, depth: u8, gen_bound: u8, score: i32) -> u64 {
        ((key16 as u64) << 48)
            | ((depth as u64) << 40)
            | ((gen_bound as u64) << 32)
            | ((score as u32) as u64)
    }

    /// Unpack word 0
    #[inline]
    fn unpack_word0(w: u64) -> (u16, u8, u8, i32) {
        let key16 = (w >> 48) as u16;
        let depth = (w >> 40) as u8;
        let gen_bound = (w >> 32) as u8;
        let score = w as i32;
        (key16, depth, gen_bound, score)
    }

    /// Pack move into two words (for i64 coordinates)
    #[inline]
    fn pack_move(m: &Move) -> (u64, u64) {
        // Word 1: from_x (32 bits) | from_y (32 bits)
        let w1 = ((m.from.x as u64) << 32) | ((m.from.y as u32) as u64);

        // Word 2: to_x (16 bits) | to_y (16 bits) | piece_type (8 bits) | piece_color (8 bits) | promotion (8 bits) | padding (8 bits)
        let to_x_lo = (m.to.x as u16) as u64;
        let to_y_lo = (m.to.y as u16) as u64;
        let piece_type = m.piece.piece_type() as u8;
        let piece_color = m.piece.color() as u8;
        let promotion = m.promotion.map_or(0u8, |p| p as u8);

        let w2 = (to_x_lo << 48)
            | (to_y_lo << 32)
            | ((piece_type as u64) << 24)
            | ((piece_color as u64) << 16)
            | ((promotion as u64) << 8);

        (w1, w2)
    }

    /// Pack "no move" sentinel
    #[inline]
    fn pack_no_move() -> (u64, u64) {
        (NO_MOVE_SENTINEL, 0)
    }

    /// Unpack move from two words
    #[inline]
    fn unpack_move(w1: u64, w2: u64) -> Option<Move> {
        if w1 == NO_MOVE_SENTINEL {
            return None;
        }

        let from_x = (w1 >> 32) as i32 as i64;
        let from_y = w1 as i32 as i64;

        let to_x = (w2 >> 48) as i16 as i64;
        let to_y = (w2 >> 32) as i16 as i64;
        let piece_type = PieceType::from_u8((w2 >> 24) as u8);
        let piece_color = PlayerColor::from_u8((w2 >> 16) as u8);
        let promo_raw = (w2 >> 8) as u8;
        let promotion = if promo_raw == 0 {
            None
        } else {
            Some(PieceType::from_u8(promo_raw))
        };

        Some(Move {
            from: Coordinate {
                x: from_x,
                y: from_y,
            },
            to: Coordinate { x: to_x, y: to_y },
            piece: Piece::new(piece_type, piece_color),
            promotion,
            rook_coord: None,
        })
    }

    /// Probe the TT for a position (lock-free).
    pub fn probe(
        &self,
        hash: u64,
        alpha: i32,
        beta: i32,
        depth: usize,
        ply: usize,
        rule50_count: u32,
        rule_limit: i32,
    ) -> Option<(i32, Option<Move>, bool)> {
        let bucket_idx = self.bucket_index(hash);
        let offset = self.bucket_offset(bucket_idx);
        let key16 = Self::key16(hash);

        for entry_idx in 0..ENTRIES_PER_BUCKET {
            let base = offset + entry_idx * WORDS_PER_ENTRY;

            // Read atomically (Relaxed is fine for Lazy SMP - we accept races)
            let w0 = self.data[base].load(Ordering::Relaxed);
            let w1 = self.data[base + 1].load(Ordering::Relaxed);
            let w2 = self.data[base + 2].load(Ordering::Relaxed);

            if w0 == 0 {
                continue; // Empty entry
            }

            let (stored_key16, stored_depth, gen_bound, mut score) = Self::unpack_word0(w0);

            if stored_key16 != key16 {
                continue; // Hash mismatch
            }

            let best_move = Self::unpack_move(w1, w2);
            let flag = SharedTTFlag::from_u8(gen_bound);

            // Adjust score from TT to search value, handling 50-move rule
            // We'll use the logic from tt.rs:value_from_tt
            let score = crate::search::tt::value_from_tt(score, ply, rule50_count, rule_limit);

            // Check if score is usable
            if stored_depth as usize >= depth {
                let usable_score = match flag {
                    SharedTTFlag::Exact => Some(score),
                    SharedTTFlag::LowerBound if score >= beta => Some(score),
                    SharedTTFlag::UpperBound if score <= alpha => Some(score),
                    _ => None,
                };

                if let Some(s) = usable_score {
                    let is_pv = (gen_bound & 0b100) != 0;
                    return Some((s, best_move, is_pv));
                }
            }

            // Return move for ordering even if score not usable
            let is_pv = (gen_bound & 0b100) != 0;
            return Some((INFINITY + 1, best_move, is_pv));
        }

        None
    }

    /// Store an entry in the TT (lock-free).
    pub fn store(
        &self,
        hash: u64,
        depth: usize,
        flag: SharedTTFlag,
        score: i32,
        best_move: Option<&Move>,
        ply: usize,
        generation: u8,
        is_pv: bool,
    ) {
        // Adjust mate scores
        let mut adjusted_score = score;
        if score > MATE_SCORE {
            adjusted_score += ply as i32;
        } else if score < -MATE_SCORE {
            adjusted_score -= ply as i32;
        }

        let bucket_idx = self.bucket_index(hash);
        let offset = self.bucket_offset(bucket_idx);
        let key16 = Self::key16(hash);
        let gen_bound = (generation << 3) | ((is_pv as u8) << 2) | (flag as u8);

        let w0 = Self::pack_word0(key16, depth as u8, gen_bound, adjusted_score);
        let (w1, w2) = best_move.map_or_else(Self::pack_no_move, Self::pack_move);

        // Find best slot to replace
        let mut replace_idx = 0;
        let mut worst_priority = i32::MAX;

        for entry_idx in 0..ENTRIES_PER_BUCKET {
            let base = offset + entry_idx * WORDS_PER_ENTRY;
            let existing_w0 = self.data[base].load(Ordering::Relaxed);

            if existing_w0 == 0 {
                // Empty slot - use it
                replace_idx = entry_idx;
                break;
            }

            let (stored_key16, stored_depth, stored_gen_bound, _) = Self::unpack_word0(existing_w0);

            // If same position, always replace if deeper or exact
            if stored_key16 == key16 {
                if depth >= stored_depth as usize || flag == SharedTTFlag::Exact {
                    replace_idx = entry_idx;
                    break;
                }
                return; // Don't replace with worse info
            }

            // Calculate replacement priority
            let stored_gen = stored_gen_bound >> 3;
            let age_diff = generation.wrapping_sub(stored_gen) & 0x1F;
            let priority = stored_depth as i32 * 8 - age_diff as i32 * 4;

            if priority < worst_priority {
                worst_priority = priority;
                replace_idx = entry_idx;
            }
        }

        // Write atomically (Relaxed is fine - we accept some races)
        let base = offset + replace_idx * WORDS_PER_ENTRY;
        self.data[base].store(w0, Ordering::Relaxed);
        self.data[base + 1].store(w1, Ordering::Relaxed);
        self.data[base + 2].store(w2, Ordering::Relaxed);
    }

    /// Clear the table
    pub fn clear(&self) {
        for atomic in &self.data {
            atomic.store(0, Ordering::Relaxed);
        }
    }

    /// Get the size in bytes (for SharedArrayBuffer allocation)
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 8
    }

    /// Get the number of words (for SharedArrayBuffer allocation)
    pub fn num_words(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// Shared TT View (for true cross-WASM instance sharing)
// ============================================================================

/// A view into a SharedArrayBuffer for cross-worker TT sharing.
///
/// This struct holds a raw pointer to shared memory and performs atomic
/// operations directly on that memory. All WASM instances sharing the
/// same SharedArrayBuffer will see each other's writes.
#[cfg(target_arch = "wasm32")]
pub struct SharedTTView {
    /// Raw pointer to the shared memory
    ptr: *mut AtomicU64,
    /// Number of u64 words in the shared memory
    len: usize,
    /// Mask for bucket indexing  
    mask: usize,
}

#[cfg(target_arch = "wasm32")]
impl SharedTTView {
    /// Create a view into shared memory.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - `ptr` points to valid SharedArrayBuffer memory
    /// - The memory remains valid for the lifetime of this view
    /// - The memory is properly aligned for AtomicU64
    pub unsafe fn new(ptr: *mut u64, len: usize) -> Self {
        let words_per_bucket = WORDS_PER_ENTRY * ENTRIES_PER_BUCKET;
        let num_buckets = len / words_per_bucket;

        // Round to power of 2
        let mut cap_pow2 = 1usize;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
        }

        SharedTTView {
            ptr: ptr as *mut AtomicU64,
            len: cap_pow2 * words_per_bucket,
            mask: cap_pow2 - 1,
        }
    }

    /// Get the capacity (number of entries) in the shared TT
    #[inline]
    pub fn capacity(&self) -> usize {
        self.len / WORDS_PER_ENTRY
    }

    /// Get the length in words
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    #[inline]
    fn bucket_offset(&self, bucket_idx: usize) -> usize {
        bucket_idx * ENTRIES_PER_BUCKET * WORDS_PER_ENTRY
    }

    #[inline]
    unsafe fn load_word(&self, offset: usize) -> u64 {
        (*self.ptr.add(offset)).load(Ordering::Relaxed)
    }

    #[inline]
    unsafe fn store_word(&self, offset: usize, value: u64) {
        (*self.ptr.add(offset)).store(value, Ordering::Relaxed);
    }

    /// Probe the shared TT
    pub unsafe fn probe(
        &self,
        hash: u64,
        alpha: i32,
        beta: i32,
        depth: usize,
        ply: usize,
        rule50_count: u32,
        rule_limit: i32,
    ) -> Option<(i32, Option<Move>, bool)> {
        let bucket_idx = self.bucket_index(hash);
        let base_offset = self.bucket_offset(bucket_idx);
        let key16 = SharedTT::key16(hash);

        for entry_idx in 0..ENTRIES_PER_BUCKET {
            let offset = base_offset + entry_idx * WORDS_PER_ENTRY;

            let w0 = self.load_word(offset);
            if w0 == 0 {
                continue;
            }

            let (stored_key16, stored_depth, gen_bound, mut score) = SharedTT::unpack_word0(w0);
            if stored_key16 != key16 {
                continue;
            }

            let w1 = self.load_word(offset + 1);
            let w2 = self.load_word(offset + 2);
            let best_move = SharedTT::unpack_move(w1, w2);
            let flag = SharedTTFlag::from_u8(gen_bound);

            // Adjust score from TT to search value, handling 50-move rule
            let score = crate::search::tt::value_from_tt(score, ply, rule50_count, rule_limit);

            if stored_depth as usize >= depth {
                let usable = match flag {
                    SharedTTFlag::Exact => Some(score),
                    SharedTTFlag::LowerBound if score >= beta => Some(score),
                    SharedTTFlag::UpperBound if score <= alpha => Some(score),
                    _ => None,
                };
                if let Some(s) = usable {
                    let is_pv = (gen_bound & 0b100) != 0;
                    return Some((s, best_move, is_pv));
                }
            }

            let is_pv = (gen_bound & 0b100) != 0;
            return Some((INFINITY + 1, best_move, is_pv));
        }

        None
    }

    /// Store to the shared TT
    pub unsafe fn store(
        &self,
        hash: u64,
        depth: usize,
        flag: SharedTTFlag,
        score: i32,
        best_move: Option<&Move>,
        ply: usize,
        generation: u8,
        is_pv: bool,
    ) {
        let mut adjusted_score = score;
        if score > MATE_SCORE {
            adjusted_score += ply as i32;
        } else if score < -MATE_SCORE {
            adjusted_score -= ply as i32;
        }

        let bucket_idx = self.bucket_index(hash);
        let base_offset = self.bucket_offset(bucket_idx);
        let key16 = SharedTT::key16(hash);
        let gen_bound = (generation << 3) | ((is_pv as u8) << 2) | (flag as u8);

        let w0 = SharedTT::pack_word0(key16, depth as u8, gen_bound, adjusted_score);
        let (w1, w2) = best_move.map_or_else(SharedTT::pack_no_move, SharedTT::pack_move);

        // Find replacement slot
        let mut replace_idx = 0;
        let mut worst_priority = i32::MAX;

        for entry_idx in 0..ENTRIES_PER_BUCKET {
            let offset = base_offset + entry_idx * WORDS_PER_ENTRY;
            let existing_w0 = self.load_word(offset);

            if existing_w0 == 0 {
                replace_idx = entry_idx;
                break;
            }

            let (stored_key16, stored_depth, stored_gen_bound, _) =
                SharedTT::unpack_word0(existing_w0);

            if stored_key16 == key16 {
                if depth >= stored_depth as usize || flag == SharedTTFlag::Exact {
                    replace_idx = entry_idx;
                    break;
                }
                return;
            }

            // Calculate replacement priority
            let stored_gen = stored_gen_bound >> 3;
            let age_diff = generation.wrapping_sub(stored_gen) & 0x1F;
            let priority = stored_depth as i32 * 8 - age_diff as i32 * 4;

            if priority < worst_priority {
                worst_priority = priority;
                replace_idx = entry_idx;
            }
        }

        let offset = base_offset + replace_idx * WORDS_PER_ENTRY;
        self.store_word(offset, w0);
        self.store_word(offset + 1, w1);
        self.store_word(offset + 2, w2);
    }

    /// Clear the shared TT
    pub unsafe fn clear(&self) {
        for i in 0..self.len {
            self.store_word(i, 0);
        }
    }

    /// Estimate the fill rate in permille (0-1000) by sampling.
    /// Accessing all memory is too slow during search, so we sample.
    pub unsafe fn fill_permille(&self) -> u32 {
        // Sample 1000 entries
        let sample_size = 1000;
        let total_buckets = self.len / (WORDS_PER_ENTRY * ENTRIES_PER_BUCKET);

        // If table is small, check everything
        if total_buckets < sample_size {
            let used = self.used_entries();
            let capacity = self.capacity();
            return if capacity > 0 {
                ((used as u64 * 1000) / capacity as u64) as u32
            } else {
                0
            };
        }

        let step = total_buckets / sample_size;
        let mut used_samples = 0;

        for i in 0..sample_size {
            let bucket_idx = i * step;
            let base_offset = self.bucket_offset(bucket_idx);

            // Check if any entry in the bucket is used
            let mut bucket_used = false;
            for entry_idx in 0..ENTRIES_PER_BUCKET {
                let offset = base_offset + entry_idx * WORDS_PER_ENTRY;
                if self.load_word(offset) != 0 {
                    bucket_used = true;
                    break;
                }
            }
            if bucket_used {
                used_samples += 1;
            }
        }

        (used_samples * 1000) / sample_size as u32
    }

    /// Calculate exactly how many entries are used (slow).
    pub unsafe fn used_entries(&self) -> usize {
        let mut used = 0;
        let total_entries = self.len / WORDS_PER_ENTRY;

        for i in 0..total_entries {
            let offset = i * WORDS_PER_ENTRY;
            if self.load_word(offset) != 0 {
                used += 1;
            }
        }
        used
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_tt_basic() {
        let tt = SharedTT::new(1);

        let hash = 0x123456789ABCDEF0u64;
        tt.store(hash, 5, SharedTTFlag::Exact, 100, None, 0, 1);

        let result = tt.probe(hash, -1000, 1000, 5, 0, 1);
        assert!(result.is_some());
        let (score, _) = result.unwrap();
        assert_eq!(score, 100);
    }

    #[test]
    fn test_word_packing() {
        let w0 = SharedTT::pack_word0(0xABCD, 10, 5, -500);
        let (key16, depth, gen_bound, score) = SharedTT::unpack_word0(w0);
        assert_eq!(key16, 0xABCD);
        assert_eq!(depth, 10);
        assert_eq!(gen_bound, 5);
        assert_eq!(score, -500);
    }
}
