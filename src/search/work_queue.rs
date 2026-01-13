//! Shared Work Queue for Root Move Splitting (Lazy SMP)
//!
//! This module provides a lock-free work queue for distributing root moves
//! across multiple Web Workers. Each worker atomically claims the next unsearched
//! root move, searches it, and reports results back.
//!
//! Memory Layout (in shared WASM memory after TT):
//! - Offset 0: next_move_index (AtomicU64) - next move to claim
//! - Offset 1: total_moves (u64) - number of root moves
//! - Offset 2: best_score (AtomicI64) - best score found so far
//! - Offset 3: best_move_index (AtomicU64) - index of best move
//! - Offset 4: search_depth (u64) - current iteration depth
//! - Offset 5: moves_completed (AtomicU64) - how many moves finished
//! - Offset 6+: packed root moves (48 bytes each)

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::Move;

/// Maximum root moves we can handle
pub const MAX_ROOT_MOVES: usize = 256;

/// Size of packed move in u64 words (6 words = 48 bytes)
/// from_x, from_y, to_x, to_y, piece_info, promotion_info
const WORDS_PER_PACKED_MOVE: usize = 6;

/// Header size in u64 words
const HEADER_SIZE: usize = 6;

/// Sentinel for "no more moves"
pub const NO_MORE_MOVES: usize = usize::MAX;

/// View into shared work queue memory
#[cfg(target_arch = "wasm32")]
pub struct SharedWorkQueue {
    ptr: *mut AtomicU64,
    len: usize,
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for SharedWorkQueue {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for SharedWorkQueue {}

#[cfg(target_arch = "wasm32")]
impl SharedWorkQueue {
    /// Create a view into shared work queue memory
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and properly aligned.
    pub unsafe fn new(ptr: *mut u64, len: usize) -> Self {
        SharedWorkQueue {
            ptr: ptr as *mut AtomicU64,
            len,
        }
    }

    /// Initialize the work queue with root moves
    /// Called by thread 0 before other threads start
    pub unsafe fn init_with_moves(&self, moves: &[Move], depth: usize) {
        // Reset counters
        (*self.ptr.add(0)).store(0, Ordering::Release); // next_move_index
        (*self.ptr.add(1)).store(moves.len() as u64, Ordering::Release); // total_moves
        (*(self.ptr.add(2) as *mut AtomicI64)).store(i64::MIN, Ordering::Release); // best_score
        (*self.ptr.add(3)).store(u64::MAX, Ordering::Release); // best_move_index (none yet)
        (*self.ptr.add(4)).store(depth as u64, Ordering::Release); // search_depth
        (*self.ptr.add(5)).store(0, Ordering::Release); // moves_completed

        // Pack moves into shared memory
        for (i, m) in moves.iter().enumerate() {
            if i >= MAX_ROOT_MOVES {
                break;
            }
            self.store_move(i, m);
        }
    }

    /// Claim the next move to search. Returns NO_MORE_MOVES if all claimed.
    pub unsafe fn claim_next_move(&self) -> usize {
        let total = (*self.ptr.add(1)).load(Ordering::Acquire) as usize;
        let idx = (*self.ptr.add(0)).fetch_add(1, Ordering::AcqRel) as usize;

        if idx < total { idx } else { NO_MORE_MOVES }
    }

    /// Get the move at a given index
    pub unsafe fn get_move(&self, index: usize) -> Option<Move> {
        let total = (*self.ptr.add(1)).load(Ordering::Acquire) as usize;
        if index >= total || index >= MAX_ROOT_MOVES {
            return None;
        }
        self.load_move(index)
    }

    /// Report a result for a move
    pub unsafe fn report_result(&self, move_index: usize, score: i32) {
        // Atomically update best if this is better
        let best_ptr = self.ptr.add(2) as *mut AtomicI64;
        let mut current_best = (*best_ptr).load(Ordering::Acquire);

        loop {
            if score as i64 <= current_best {
                break; // Not better
            }

            match (*best_ptr).compare_exchange_weak(
                current_best,
                score as i64,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Successfully updated best score, now update best move index
                    (*self.ptr.add(3)).store(move_index as u64, Ordering::Release);
                    break;
                }
                Err(actual) => {
                    current_best = actual;
                }
            }
        }

        // Increment moves completed
        (*self.ptr.add(5)).fetch_add(1, Ordering::AcqRel);
    }

    /// Check if all moves have been searched
    pub unsafe fn all_moves_complete(&self) -> bool {
        let total = (*self.ptr.add(1)).load(Ordering::Acquire);
        let completed = (*self.ptr.add(5)).load(Ordering::Acquire);
        completed >= total
    }

    /// Get the best move found so far
    pub unsafe fn get_best_move(&self) -> Option<(Move, i32)> {
        let best_idx = (*self.ptr.add(3)).load(Ordering::Acquire);
        if best_idx == u64::MAX {
            return None;
        }

        let score = (*(self.ptr.add(2) as *mut AtomicI64)).load(Ordering::Acquire) as i32;
        self.load_move(best_idx as usize).map(|m| (m, score))
    }

    /// Get the current search depth
    pub unsafe fn get_depth(&self) -> usize {
        (*self.ptr.add(4)).load(Ordering::Acquire) as usize
    }

    /// Get total number of root moves
    pub unsafe fn total_moves(&self) -> usize {
        (*self.ptr.add(1)).load(Ordering::Acquire) as usize
    }

    /// Get number of moves completed
    pub unsafe fn moves_completed(&self) -> usize {
        (*self.ptr.add(5)).load(Ordering::Acquire) as usize
    }

    /// Reset for next iteration (keeps moves, resets counters)
    pub unsafe fn reset_for_iteration(&self, depth: usize) {
        (*self.ptr.add(0)).store(0, Ordering::Release); // next_move_index
        (*(self.ptr.add(2) as *mut AtomicI64)).store(i64::MIN, Ordering::Release); // best_score
        (*self.ptr.add(3)).store(u64::MAX, Ordering::Release); // best_move_index
        (*self.ptr.add(4)).store(depth as u64, Ordering::Release); // search_depth
        (*self.ptr.add(5)).store(0, Ordering::Release); // moves_completed
    }

    /// Reorder moves to put best move first (for next iteration)
    pub unsafe fn reorder_best_first(&self) {
        let best_idx = (*self.ptr.add(3)).load(Ordering::Acquire) as usize;
        if best_idx == usize::MAX || best_idx == 0 {
            return; // No best or already first
        }

        let total = self.total_moves();
        if best_idx >= total {
            return;
        }

        // Swap best move with first move
        let first_move = self.load_move(0);
        let best_move = self.load_move(best_idx);

        if let (Some(fm), Some(bm)) = (first_move, best_move) {
            self.store_move(0, &bm);
            self.store_move(best_idx, &fm);
        }
    }

    // Internal helpers for packing/unpacking moves

    unsafe fn store_move(&self, index: usize, m: &Move) {
        let base = HEADER_SIZE + index * WORDS_PER_PACKED_MOVE;

        // Word 0: from_x as i64
        (*self.ptr.add(base)).store(m.from.x as u64, Ordering::Release);
        // Word 1: from_y as i64
        (*self.ptr.add(base + 1)).store(m.from.y as u64, Ordering::Release);
        // Word 2: to_x as i64
        (*self.ptr.add(base + 2)).store(m.to.x as u64, Ordering::Release);
        // Word 3: to_y as i64
        (*self.ptr.add(base + 3)).store(m.to.y as u64, Ordering::Release);
        // Word 4: piece info (type, color)
        let piece_info = ((m.piece.piece_type() as u64) << 8) | (m.piece.color() as u64);
        (*self.ptr.add(base + 4)).store(piece_info, Ordering::Release);
        // Word 5: promotion (0 if none)
        let promo = m.promotion.map_or(0u64, |p| p as u64 + 1);
        (*self.ptr.add(base + 5)).store(promo, Ordering::Release);
    }

    unsafe fn load_move(&self, index: usize) -> Option<Move> {
        let base = HEADER_SIZE + index * WORDS_PER_PACKED_MOVE;

        if base + WORDS_PER_PACKED_MOVE > self.len {
            return None;
        }

        let from_x = (*self.ptr.add(base)).load(Ordering::Acquire) as i64;
        let from_y = (*self.ptr.add(base + 1)).load(Ordering::Acquire) as i64;
        let to_x = (*self.ptr.add(base + 2)).load(Ordering::Acquire) as i64;
        let to_y = (*self.ptr.add(base + 3)).load(Ordering::Acquire) as i64;
        let piece_info = (*self.ptr.add(base + 4)).load(Ordering::Acquire);
        let promo = (*self.ptr.add(base + 5)).load(Ordering::Acquire);

        let piece_type = PieceType::from_u8(((piece_info >> 8) & 0xFF) as u8);
        let color = PlayerColor::from_u8((piece_info & 0xFF) as u8);

        let promotion = if promo > 0 {
            Some(PieceType::from_u8((promo - 1) as u8))
        } else {
            None
        };

        Some(Move {
            from: Coordinate::new(from_x, from_y),
            to: Coordinate::new(to_x, to_y),
            piece: Piece::new(piece_type, color),
            promotion,
            rook_coord: None, // Work queue doesn't preserve castling info
        })
    }
}

/// Required size in u64 words for the work queue
pub const WORK_QUEUE_SIZE_WORDS: usize = HEADER_SIZE + MAX_ROOT_MOVES * WORDS_PER_PACKED_MOVE;
