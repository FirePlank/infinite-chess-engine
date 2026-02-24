use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::moves::Move;

use super::INFINITY;
use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, clamp_to_i16, pack_coord, unpack_coord,
    value_from_tt, value_to_tt,
};

const ENTRIES_PER_BUCKET: usize = 4; // 4 × 16 = 64 bytes

// Generation management
const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
#[allow(clippy::identity_op)]
const GENERATION_MASK: u8 = (0xFF << GENERATION_BITS) & 0xFF;

use super::tt_defs::{MAX_TT_COORD, MIN_TT_COORD};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTEntry {
    pub key16: u16,     // Upper hash bits
    pub depth: u8,      // Search depth
    pub gen_bound: u8,  // Aging generation + PV flag + Bound type
    pub score16: i16,   // Node score
    pub eval16: i16,    // Static evaluation
    pub move_data: u64, // Packed move info
}

const _: () = assert!(std::mem::size_of::<TTEntry>() == 16);

const NO_MOVE: u64 = 0;

impl TTEntry {
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            key16: 0,
            depth: 0,
            gen_bound: 0,
            score16: 0,
            eval16: 0,
            move_data: NO_MOVE,
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
        current_gen.wrapping_sub(self.gen_bound & GENERATION_MASK) & GENERATION_MASK
    }

    #[inline]
    pub fn best_move(&self, hash: u64) -> Option<Move> {
        if self.move_data == NO_MOVE {
            return None;
        }

        let hash_key = hash >> 16;
        let m = self.move_data ^ hash_key;
        if m == NO_MOVE {
            return None;
        }

        let pt = PieceType::from_u8((m & 0x1F) as u8);
        let cl = PlayerColor::from_u8(((m >> 5) & 0x03) as u8);
        let pr = ((m >> 7) & 0x1F) as u8;

        let from_x = unpack_coord(m >> 12);
        let from_y = unpack_coord(m >> 25);
        let to_x = unpack_coord(m >> 38);
        let to_y = unpack_coord(m >> 51);

        Some(Move {
            from: Coordinate {
                x: from_x,
                y: from_y,
            },
            to: Coordinate { x: to_x, y: to_y },
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
    fn encode_move(&mut self, m: &Move, hash: u64) -> bool {
        if m.from.x < MIN_TT_COORD
            || m.from.x > MAX_TT_COORD
            || m.from.y < MIN_TT_COORD
            || m.from.y > MAX_TT_COORD
            || m.to.x < MIN_TT_COORD
            || m.to.x > MAX_TT_COORD
            || m.to.y < MIN_TT_COORD
            || m.to.y > MAX_TT_COORD
        {
            self.move_data = NO_MOVE;
            return false;
        }

        let pt = m.piece.piece_type() as u64;
        let cl = m.piece.color() as u64;
        let pr = m.promotion.map_or(0, |p| p as u64);

        let mdata = (pt & 0x1F)
            | ((cl & 0x03) << 5)
            | ((pr & 0x1F) << 7)
            | (pack_coord(m.from.x) << 12)
            | (pack_coord(m.from.y) << 25)
            | (pack_coord(m.to.x) << 38)
            | (pack_coord(m.to.y) << 51);

        let hash_key = hash >> 16;
        let self_move_data = mdata ^ hash_key;
        self.move_data = self_move_data;
        true
    }
}

// Buckets are 64-byte aligned to fit exactly one CPU cache line.

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct TTBucket {
    pub entries: [TTEntry; ENTRIES_PER_BUCKET],
}

const _: () = assert!(std::mem::size_of::<TTBucket>() == 64);

impl TTBucket {
    #[inline]
    pub const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
        }
    }
}

// Thread-local Transposition Table optimized for speed.
// Using raw pointers and bit-level management for fast access.

pub struct LocalTranspositionTable {
    buckets: *mut TTBucket,
    capacity: usize,
    mask: usize,
    index_bits: u32,
    pub generation: u8,
    pub used: usize,
    // Keep the Vec to manage the underlying memory lifecycle
    _mem_anchor: Vec<TTBucket>,
}

unsafe impl Sync for LocalTranspositionTable {}
unsafe impl Send for LocalTranspositionTable {}

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

        let mut _mem_anchor = vec![TTBucket::empty(); cap];
        let buckets = _mem_anchor.as_mut_ptr();

        LocalTranspositionTable {
            buckets,
            capacity: cap * ENTRIES_PER_BUCKET,
            mask: cap - 1,
            index_bits: bits,
            generation: 1,
            used: 0,
            _mem_anchor,
        }
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn used_entries(&self) -> usize {
        self.used
    }

    #[inline(always)]
    pub fn fill_permille(&self) -> u32 {
        if self.capacity == 0 {
            0
        } else {
            ((self.used as u64 * 1000) / self.capacity as u64) as u32
        }
    }

    #[inline(always)]
    fn hash_key16(&self, hash: u64) -> u16 {
        (hash >> self.index_bits) as u16
    }

    #[inline(always)]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = (hash as usize) & self.mask;
        unsafe {
            let ptr = self.buckets.add(idx) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }

    #[inline(always)]
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    pub fn prefetch_entry(&self, _hash: u64) {}

    pub fn probe_move(&self, hash: u64) -> Option<Move> {
        let key16 = self.hash_key16(hash);
        let idx = (hash as usize) & self.mask;

        unsafe {
            let bucket_ptr = self.buckets.add(idx);
            let entries = &(*bucket_ptr).entries;

            for e in entries {
                if e.key16 == key16 && !e.is_empty() {
                    return e.best_move(hash);
                }
            }
        }
        None
    }

    #[inline(always)]
    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        let key16 = self.hash_key16(params.hash);
        let idx = (params.hash as usize) & self.mask;

        unsafe {
            let bucket_ptr = self.buckets.add(idx);
            // Access entries directly without intermediate copy
            let entries = &(*bucket_ptr).entries;

            for e in entries {
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
                    let flag = e.flag();
                    let usable = match flag {
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
                    best_move: e.best_move(params.hash),
                });
            }
        }
        None
    }

    /// Stores results in the TT, replacing existing entries based on
    /// search depth and relative age (generation).
    #[inline(always)]
    pub fn store(&mut self, params: &TTStoreParams) {
        let key16 = self.hash_key16(params.hash);
        let adj_score = value_to_tt(params.score, params.ply);
        let idx = (params.hash as usize) & self.mask;

        unsafe {
            let bucket_ptr = self.buckets.add(idx);
            let entries = &mut (*bucket_ptr).entries;

            let mut replace_idx = 0;
            let mut worst = i32::MAX;

            for (i, e) in entries.iter_mut().enumerate() {
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
                        || e.relative_age(self.generation) != 0
                    {
                        let old_move_data = e.move_data;
                        *e = TTEntry {
                            key16,
                            depth: params.depth as u8,
                            gen_bound: TTEntry::pack_gen_bound(
                                self.generation,
                                params.is_pv,
                                params.flag,
                            ),
                            score16: clamp_to_i16(adj_score),
                            eval16: store_eval,
                            move_data: old_move_data,
                        };
                        if let Some(m) = &params.best_move {
                            e.encode_move(m, params.hash);
                        }
                    } else if e.depth >= 5 && e.flag() != TTFlag::Exact {
                        e.depth = e.depth.saturating_sub(1);
                    }
                    return;
                }
                let priority = (e.depth as i32 + 3 + if e.is_pv() { 2 } else { 0 })
                    - (e.relative_age(self.generation) as i32);
                if priority < worst {
                    worst = priority;
                    replace_idx = i;
                }
            }

            let mut new_e = TTEntry {
                key16,
                depth: params.depth as u8,
                gen_bound: TTEntry::pack_gen_bound(self.generation, params.is_pv, params.flag),
                score16: clamp_to_i16(adj_score),
                eval16: clamp_to_i16(params.static_eval),
                move_data: NO_MOVE,
            };
            if let Some(m) = &params.best_move {
                new_e.encode_move(m, params.hash);
            }

            if entries[replace_idx].is_empty() {
                self.used += 1;
            }
            entries[replace_idx] = new_e;
        }
    }

    #[inline(always)]
    pub fn increment_age(&mut self) {
        self.generation = self.generation.wrapping_add(GENERATION_DELTA);
    }

    pub fn clear(&mut self) {
        unsafe {
            for i in 0..=self.mask {
                *self.buckets.add(i) = TTBucket::empty();
            }
        }
        self.generation = 1;
        self.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<TTEntry>(), 16);
        assert_eq!(std::mem::size_of::<TTBucket>(), 64);
    }

    #[test]
    fn test_tt_basic() {
        let mut tt = LocalTranspositionTable::new(1);
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
        let mut tt = LocalTranspositionTable::new(1);
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
            from: Coordinate::new(4000, -4000),
            to: Coordinate::new(-4000, 4000),
            piece: Piece::new(PieceType::Rook, PlayerColor::Black),
            promotion: None,
            rook_coord: None,
        };
        assert!(e.encode_move(&m, 0));
        let decoded = e.best_move(0).unwrap();
        assert_eq!(decoded.from.x, 4000);
        assert_eq!(decoded.from.y, -4000);
    }
}
