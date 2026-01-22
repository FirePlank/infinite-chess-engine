use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::Move;

use super::INFINITY;
use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, value_from_tt, value_to_tt,
};

const ENTRIES_PER_BUCKET: usize = 2; // 2 × 24 = 48 bytes + padding = 64

// Generation management
const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
#[allow(clippy::identity_op)]
const GENERATION_MASK: u8 = (0xFF << GENERATION_BITS) & 0xFF;
const GENERATION_CYCLE: u16 = 255 + GENERATION_DELTA as u16;

// TT entry structure uses i16 coordinates to save space while supporting
// a massive board range. Size is 24 bytes to allow for 64-byte bucket alignment.

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTEntry {
    pub key16: u16,    // Upper hash bits for collision detection
    pub depth: u8,     // Search depth
    pub gen_bound: u8, // Aging generation + PV flag + Bound type
    pub score16: i16,  // Node score
    pub eval16: i16,   // Static evaluation
    pub move16: u16,   // Piece type, color, and promotion
    pub _pad: u16,
    pub from_x: i16,
    pub from_y: i16,
    pub to_x: i16,
    pub to_y: i16,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<TTEntry>() == 24);

const NO_MOVE: u16 = 0;

#[inline]
fn clamp_to_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

impl TTEntry {
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            key16: 0,
            depth: 0,
            gen_bound: 0,
            score16: 0,
            eval16: 0,
            move16: NO_MOVE,
            _pad: 0,
            from_x: 0,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            _pad2: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key16 == 0 && self.gen_bound == 0
    }

    #[inline]
    pub fn flag(&self) -> TTFlag {
        TTFlag::from_u8(self.gen_bound)
    }

    #[inline]
    pub fn is_pv(&self) -> bool {
        (self.gen_bound & 0x04) != 0
    }

    #[inline]
    fn pack_gen_bound(generation: u8, is_pv: bool, flag: TTFlag) -> u8 {
        (generation & GENERATION_MASK) | (if is_pv { 0x04 } else { 0 }) | (flag as u8 & 0x03)
    }

    #[inline]
    pub fn relative_age(&self, current_gen: u8) -> u8 {
        ((GENERATION_CYCLE + current_gen as u16 - (self.gen_bound as u16))
            & (GENERATION_MASK as u16)) as u8
    }

    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        if self.move16 == NO_MOVE {
            return None;
        }

        let pt = PieceType::from_u8((self.move16 & 0x1F) as u8);
        let cl = PlayerColor::from_u8(((self.move16 >> 5) & 0x03) as u8);
        let pr = ((self.move16 >> 7) & 0x1F) as u8;

        Some(Move {
            from: Coordinate {
                x: self.from_x as i64,
                y: self.from_y as i64,
            },
            to: Coordinate {
                x: self.to_x as i64,
                y: self.to_y as i64,
            },
            piece: Piece::new(pt, cl),
            promotion: if pr == 0 {
                None
            } else {
                Some(PieceType::from_u8(pr))
            },
            rook_coord: None,
        })
    }

    #[inline]
    fn encode_move(&mut self, m: &Move) -> bool {
        if m.from.x < i16::MIN as i64
            || m.from.x > i16::MAX as i64
            || m.from.y < i16::MIN as i64
            || m.from.y > i16::MAX as i64
            || m.to.x < i16::MIN as i64
            || m.to.x > i16::MAX as i64
            || m.to.y < i16::MIN as i64
            || m.to.y > i16::MAX as i64
        {
            self.move16 = NO_MOVE;
            return false;
        }

        let pt = m.piece.piece_type() as u16;
        let cl = m.piece.color() as u16;
        let pr = m.promotion.map_or(0, |p| p as u16);
        self.move16 = (pt & 0x1F) | ((cl & 0x03) << 5) | ((pr & 0x1F) << 7);
        self.from_x = m.from.x as i16;
        self.from_y = m.from.y as i16;
        self.to_x = m.to.x as i16;
        self.to_y = m.to.y as i16;
        true
    }
}

// Buckets are 64-byte aligned to fit exactly one CPU cache line.
// This minimizes cache misses when probing a hash.

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET], // 48 bytes
    _pad: [u8; 16],                         // 16 bytes padding
}

const _: () = assert!(std::mem::size_of::<TTBucket>() == 64);

impl TTBucket {
    #[inline]
    pub const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
            _pad: [0; 16],
        }
    }
}

// Thread-local Transposition Table using UnsafeCell for performance.
// Masking is used for fast bucket indexing.

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
        let num_buckets = (bytes / 64).max(1);
        let mut cap = 1usize;
        let mut bits = 0u32;
        while cap * 2 <= num_buckets {
            cap *= 2;
            bits += 1;
        }

        LocalTranspositionTable {
            buckets: UnsafeCell::new(vec![TTBucket::empty(); cap]),
            mask: cap - 1,
            index_bits: bits,
            generation: UnsafeCell::new(1),
            used: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { (*self.buckets.get()).len() * ENTRIES_PER_BUCKET }
    }
    #[inline]
    pub fn used_entries(&self) -> usize {
        unsafe { *self.used.get() }
    }
    #[inline]
    pub fn fill_permille(&self) -> u32 {
        let cap = self.capacity();
        if cap == 0 {
            0
        } else {
            ((self.used_entries() as u64 * 1000) / cap as u64) as u32
        }
    }
    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }
    #[inline]
    fn hash_key16(&self, hash: u64) -> u16 {
        (hash >> self.index_bits) as u16
    }

    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = self.bucket_index(hash);
        let ptr = unsafe { (*self.buckets.get()).as_ptr().wrapping_add(idx) as *const i8 };
        unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
    }
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    pub fn prefetch_entry(&self, _hash: u64) {}

    pub fn probe_move(&self, hash: u64) -> Option<Move> {
        let key16 = self.hash_key16(hash);
        let bucket = unsafe { &(&(*self.buckets.get()))[self.bucket_index(hash)] };
        for e in &bucket.entries {
            if e.key16 == key16 && !e.is_empty() {
                return e.best_move();
            }
        }
        None
    }

    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        let key16 = self.hash_key16(params.hash);
        let bucket = unsafe { &(&(*self.buckets.get()))[self.bucket_index(params.hash)] };

        for e in &bucket.entries {
            if e.key16 != key16 || e.is_empty() {
                continue;
            }

            let score = value_from_tt(
                e.score16 as i32,
                params.ply,
                params.rule50_count,
                params.rule_limit,
            );
            let mut cutoff = INFINITY + 1;

            if e.depth as usize >= params.depth {
                let usable = match e.flag() {
                    TTFlag::Exact => true,
                    TTFlag::LowerBound if score >= params.beta => true,
                    TTFlag::UpperBound if score <= params.alpha => true,
                    _ => false,
                };
                if usable {
                    cutoff = score;
                }
            }

            return Some(TTProbeResult {
                cutoff_score: cutoff,
                tt_score: score,
                eval: e.eval16 as i32,
                depth: e.depth,
                flag: e.flag(),
                is_pv: e.is_pv(),
                best_move: e.best_move(),
            });
        }
        None
    }

    /// Stores results in the TT, replacing existing entries based on
    /// search depth and relative age (generation).
    pub fn store(&self, params: &TTStoreParams) {
        let key16 = self.hash_key16(params.hash);
        let adj_score = value_to_tt(params.score, params.ply);
        let (curr_gen, bucket, used) = unsafe {
            (
                *self.generation.get(),
                &mut (&mut (*self.buckets.get()))[self.bucket_index(params.hash)],
                self.used.get(),
            )
        };

        let mut replace_idx = 0;
        let mut worst = i32::MAX;

        for (i, e) in bucket.entries.iter_mut().enumerate() {
            if e.key16 == key16 && !e.is_empty() {
                let store_eval = if params.static_eval != INFINITY + 1 {
                    clamp_to_i16(params.static_eval)
                } else {
                    e.eval16
                };
                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (e.depth as i32 - 4)
                    || e.relative_age(curr_gen) != 0
                {
                    let (old_m16, old_fx, old_fy, old_tx, old_ty) =
                        (e.move16, e.from_x, e.from_y, e.to_x, e.to_y);
                    *e = TTEntry {
                        key16,
                        depth: params.depth as u8,
                        gen_bound: TTEntry::pack_gen_bound(curr_gen, params.is_pv, params.flag),
                        score16: clamp_to_i16(adj_score),
                        eval16: store_eval,
                        move16: old_m16,
                        _pad: 0,
                        from_x: old_fx,
                        from_y: old_fy,
                        to_x: old_tx,
                        to_y: old_ty,
                        _pad2: 0,
                    };
                    if let Some(m) = &params.best_move {
                        e.encode_move(m);
                    }
                } else if e.depth >= 5 && e.flag() != TTFlag::Exact {
                    e.depth = e.depth.saturating_sub(1);
                }
                return;
            }
            let priority = (e.depth as i32) - (e.relative_age(curr_gen) as i32);
            if priority < worst {
                worst = priority;
                replace_idx = i;
            }
        }

        let mut new_e = TTEntry {
            key16,
            depth: params.depth as u8,
            gen_bound: TTEntry::pack_gen_bound(curr_gen, params.is_pv, params.flag),
            score16: clamp_to_i16(adj_score),
            eval16: clamp_to_i16(params.static_eval),
            move16: NO_MOVE,
            _pad: 0,
            from_x: 0,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            _pad2: 0,
        };
        if let Some(m) = &params.best_move {
            new_e.encode_move(m);
        }

        if bucket.entries[replace_idx].is_empty() {
            unsafe {
                *used += 1;
            }
        }
        bucket.entries[replace_idx] = new_e;
    }

    pub fn increment_age(&self) {
        unsafe {
            *self.generation.get() = (*self.generation.get()).wrapping_add(GENERATION_DELTA);
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
        assert_eq!(std::mem::size_of::<TTEntry>(), 24);
        assert_eq!(std::mem::size_of::<TTBucket>(), 64);
    }

    #[test]
    fn test_tt_basic() {
        let tt = LocalTranspositionTable::new(1);
        let hash = 0x123456789ABCDEFu64;
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
        let res = tt
            .probe(&TTProbeParams {
                hash,
                alpha: -1000,
                beta: 1000,
                depth: 5,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            })
            .unwrap();
        assert_eq!(res.cutoff_score, 100);
        assert_eq!(res.eval, 90);
    }

    #[test]
    fn test_move_roundtrip() {
        let tt = LocalTranspositionTable::new(1);
        let hash = 0xABCDEF123456789u64;
        let m = Move {
            from: Coordinate::new(4, 2),
            to: Coordinate::new(4, 4),
            piece: Piece::new(PieceType::Pawn, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };
        tt.store(&TTStoreParams {
            hash,
            depth: 10,
            flag: TTFlag::Exact,
            score: 50,
            static_eval: 40,
            is_pv: true,
            best_move: Some(m),
            ply: 0,
        });

        let res = tt
            .probe(&TTProbeParams {
                hash,
                alpha: -1000,
                beta: 1000,
                depth: 0,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            })
            .unwrap();
        let decoded = res.best_move.unwrap();
        assert_eq!(decoded.from, m.from);
        assert_eq!(decoded.to, m.to);
    }

    #[test]
    fn test_extreme_coords() {
        let mut e = TTEntry::empty();
        let m = Move {
            from: Coordinate::new(30000, -30000),
            to: Coordinate::new(-30000, 30000),
            piece: Piece::new(PieceType::Rook, PlayerColor::Black),
            promotion: None,
            rook_coord: None,
        };
        assert!(e.encode_move(&m));
        let decoded = e.best_move().unwrap();
        assert_eq!(decoded.from.x, 30000);
        assert_eq!(decoded.from.y, -30000);
    }
}
