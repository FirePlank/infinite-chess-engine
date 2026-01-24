use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::INFINITY;
use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, value_from_tt, value_to_tt,
};

const ENTRIES_PER_BUCKET: usize = 3;

const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
#[allow(clippy::identity_op)]
const GENERATION_MASK: u8 = (0xFF << GENERATION_BITS) & 0xFF;

const NO_MOVE: u16 = 0;

use std::cell::UnsafeCell;

// Entries are stored as 3 × u64 words for atomic-friendly access without explicit locks.
// word0: key16 (16) | depth (8) | gen_bound (8) | score (16) | eval (16)
// word1: move16 (16) | from_x (16) | from_y (16) | to_x (16)
// word2: to_y (16) | padding (48)

#[repr(C, align(8))]
pub struct TTEntry {
    word0: UnsafeCell<u64>,
    word1: UnsafeCell<u64>,
    word2: UnsafeCell<u64>,
}

unsafe impl Sync for TTEntry {}
unsafe impl Send for TTEntry {}

#[inline]
fn clamp_to_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

impl TTEntry {
    pub fn empty() -> Self {
        TTEntry {
            word0: UnsafeCell::new(0),
            word1: UnsafeCell::new(0),
            word2: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub fn read(&self, key16: u16) -> Option<(i32, i32, u8, u8, Option<Move>)> {
        unsafe {
            let w0 = std::ptr::read_volatile(self.word0.get());
            let stored_key = (w0 & 0xFFFF) as u16;
            if stored_key != key16 || w0 == 0 {
                return None;
            }

            let depth = ((w0 >> 16) & 0xFF) as u8;
            let gen_bound = ((w0 >> 24) & 0xFF) as u8;
            let score = ((w0 >> 32) & 0xFFFF) as i16 as i32;
            let eval = ((w0 >> 48) & 0xFFFF) as i16 as i32;

            let w1 = std::ptr::read_volatile(self.word1.get());
            let move16 = (w1 & 0xFFFF) as u16;

            let best_move = if move16 == NO_MOVE {
                None
            } else {
                let from_x = ((w1 >> 16) & 0xFFFF) as i16;
                let from_y = ((w1 >> 32) & 0xFFFF) as i16;
                let to_x = ((w1 >> 48) & 0xFFFF) as i16;
                let w2 = std::ptr::read_volatile(self.word2.get());
                let to_y = (w2 & 0xFFFF) as i16;

                let pt = PieceType::from_u8((move16 & 0x1F) as u8);
                let cl = PlayerColor::from_u8(((move16 >> 5) & 0x03) as u8);
                let pr = ((move16 >> 7) & 0x1F) as u8;

                Some(Move {
                    from: Coordinate {
                        x: from_x as i64,
                        y: from_y as i64,
                    },
                    to: Coordinate {
                        x: to_x as i64,
                        y: to_y as i64,
                    },
                    piece: Piece::new(pt, cl),
                    promotion: if pr == 0 {
                        None
                    } else {
                        Some(PieceType::from_u8(pr))
                    },
                    rook_coord: None,
                })
            };

            Some((score, eval, depth, gen_bound, best_move))
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn write(
        &self,
        key16: u16,
        score: i16,
        eval: i16,
        depth: u8,
        gen_bound: u8,
        move16: u16,
        from_x: i16,
        from_y: i16,
        to_x: i16,
        to_y: i16,
    ) {
        let w0 = (key16 as u64)
            | ((depth as u64) << 16)
            | ((gen_bound as u64) << 24)
            | (((score as u16) as u64) << 32)
            | (((eval as u16) as u64) << 48);
        let w1 = (move16 as u64)
            | (((from_x as u16) as u64) << 16)
            | (((from_y as u16) as u64) << 32)
            | (((to_x as u16) as u64) << 48);
        let w2 = (to_y as u16) as u64;

        unsafe {
            std::ptr::write_volatile(self.word0.get(), w0);
            std::ptr::write_volatile(self.word1.get(), w1);
            std::ptr::write_volatile(self.word2.get(), w2);
        }
    }

    #[inline]
    pub fn raw_word0(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.word0.get()) }
    }
    #[inline]
    pub fn clear(&self) {
        unsafe {
            std::ptr::write_volatile(self.word0.get(), 0);
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
        (gen_bound & GENERATION_MASK) >> GENERATION_BITS
    }
    #[inline]
    pub fn pack_gen_bound(r#gen: u8, is_pv: bool, flag: TTFlag) -> u8 {
        ((r#gen << GENERATION_BITS) & GENERATION_MASK)
            | (if is_pv { 0x04 } else { 0 })
            | (flag as u8 & 0x03)
    }
}

pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
}
impl TTBucket {
    pub fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(), TTEntry::empty(), TTEntry::empty()],
        }
    }
}

pub struct SharedTranspositionTable {
    buckets: Vec<TTBucket>,
    mask: usize,
    index_bits: u32,
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
        let mut cap = 1usize;
        let mut bits = 0u32;
        while cap * 2 <= num_buckets {
            cap *= 2;
            bits += 1;
        }

        let mut buckets = Vec::with_capacity(cap);
        for _ in 0..cap {
            buckets.push(TTBucket::empty());
        }

        SharedTranspositionTable {
            buckets,
            mask: cap - 1,
            index_bits: bits,
            generation: UnsafeCell::new(1),
        }
    }

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
        (self.hashfull() as usize * self.capacity()) / 1000
    }
    #[inline]
    pub fn fill_permille(&self) -> u32 {
        self.hashfull()
    }

    /// Approximate fill level in permille (0-1000).
    /// Samples a portion of the table for efficiency.
    pub fn hashfull(&self) -> u32 {
        let sample = self.buckets.len().min(1000);
        let r#gen = unsafe { *self.generation.get() };
        let mut occ = 0u32;
        for i in 0..sample {
            for e in &self.buckets[i].entries {
                let w0 = e.raw_word0();
                if w0 != 0 {
                    let gb = ((w0 >> 24) & 0xFF) as u8;
                    if (r#gen.wrapping_sub(TTEntry::generation(gb))) & 0x1F == 0 {
                        occ += 1;
                    }
                }
            }
        }
        if sample == 0 {
            0
        } else {
            (occ * 1000) / (sample * ENTRIES_PER_BUCKET) as u32
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

    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let ptr = self.buckets.as_ptr().wrapping_add(self.bucket_index(hash)) as *const i8;
        unsafe {
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    pub fn prefetch_entry(&self, _hash: u64) {}

    pub fn probe_move(&self, hash: u64) -> Option<Move> {
        let key16 = self.hash_key16(hash);
        for e in &self.buckets[self.bucket_index(hash)].entries {
            if let Some((_, _, _, _, m)) = e.read(key16) {
                return m;
            }
        }
        None
    }

    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        let key16 = self.hash_key16(params.hash);
        for e in &self.buckets[self.bucket_index(params.hash)].entries {
            if let Some((score, eval, depth, gen_bound, best_move)) = e.read(key16) {
                let score =
                    value_from_tt(score, params.ply, params.rule50_count, params.rule_limit);
                let flag = TTEntry::flag(gen_bound);
                let mut cutoff = INFINITY + 1;
                if depth as usize >= params.depth {
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
                    eval,
                    depth,
                    flag,
                    is_pv: TTEntry::is_pv(gen_bound),
                    best_move,
                });
            }
        }
        None
    }

    /// Stores an entry in the multithreaded table.
    /// Priority is given to deeper searches and newer generation entries.
    pub fn store(&self, params: &TTStoreParams) {
        let key16 = self.hash_key16(params.hash);
        let adj_score = value_to_tt(params.score, params.ply);
        let r#gen = unsafe { *self.generation.get() };
        let bucket = &self.buckets[self.bucket_index(params.hash)];

        let (m16, fx, fy, tx, ty) = params
            .best_move
            .as_ref()
            .map(|m| {
                if m.from.x >= i16::MIN as i64
                    && m.from.x <= i16::MAX as i64
                    && m.from.y >= i16::MIN as i64
                    && m.from.y <= i16::MAX as i64
                    && m.to.x >= i16::MIN as i64
                    && m.to.x <= i16::MAX as i64
                    && m.to.y >= i16::MIN as i64
                    && m.to.y <= i16::MAX as i64
                {
                    let pt = m.piece.piece_type() as u16;
                    let cl = m.piece.color() as u16;
                    let pr = m.promotion.map_or(0, |p| p as u16);
                    (
                        (pt & 0x1F) | ((cl & 0x03) << 5) | ((pr & 0x1F) << 7),
                        m.from.x as i16,
                        m.from.y as i16,
                        m.to.x as i16,
                        m.to.y as i16,
                    )
                } else {
                    (0, 0, 0, 0, 0)
                }
            })
            .unwrap_or((0, 0, 0, 0, 0));

        let mut replace_idx = 0;
        let mut worst = i32::MAX;

        for (i, e) in bucket.entries.iter().enumerate() {
            if let Some((_, old_eval, old_depth, old_gb, old_move)) = e.read(key16) {
                let (sm16, sfx, sfy, stx, sty) = if m16 != 0 {
                    (m16, fx, fy, tx, ty)
                } else if let Some(m) = old_move {
                    let pt = m.piece.piece_type() as u16;
                    let cl = m.piece.color() as u16;
                    let pr = m.promotion.map_or(0, |p| p as u16);
                    (
                        (pt & 0x1F) | ((cl & 0x03) << 5) | ((pr & 0x1F) << 7),
                        m.from.x as i16,
                        m.from.y as i16,
                        m.to.x as i16,
                        m.to.y as i16,
                    )
                } else {
                    (0, 0, 0, 0, 0)
                };

                let store_eval = if params.static_eval != INFINITY + 1 {
                    clamp_to_i16(params.static_eval)
                } else {
                    clamp_to_i16(old_eval)
                };
                let old_gen = TTEntry::generation(old_gb);
                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };
                let rel_age = (r#gen.wrapping_sub(old_gen)) & 0x1F;

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (old_depth as i32 - 4)
                    || rel_age != 0
                    || params.depth == 0
                {
                    e.write(
                        key16,
                        clamp_to_i16(adj_score),
                        store_eval,
                        params.depth as u8,
                        TTEntry::pack_gen_bound(r#gen, params.is_pv, params.flag),
                        sm16,
                        sfx,
                        sfy,
                        stx,
                        sty,
                    );
                }
                return;
            }

            let w0 = e.raw_word0();
            let ed = ((w0 >> 16) & 0xFF) as u8;
            let egb = ((w0 >> 24) & 0xFF) as u8;
            let mut prio =
                ed as i32 - ((r#gen.wrapping_sub(TTEntry::generation(egb))) & 0x1F) as i32 * 2;
            if TTEntry::flag(egb) == TTFlag::Exact || TTEntry::is_pv(egb) {
                prio += 2;
            }
            if w0 == 0 {
                prio = i32::MIN;
            }
            if prio < worst {
                worst = prio;
                replace_idx = i;
            }
        }

        let new_prio = params.depth as i32
            + if params.flag == TTFlag::Exact || params.is_pv {
                2
            } else {
                0
            };
        if new_prio >= worst {
            bucket.entries[replace_idx].write(
                key16,
                clamp_to_i16(adj_score),
                clamp_to_i16(params.static_eval),
                params.depth as u8,
                TTEntry::pack_gen_bound(r#gen, params.is_pv, params.flag),
                m16,
                fx,
                fy,
                tx,
                ty,
            );
        }
    }

    pub fn increment_age(&self) {
        unsafe {
            *self.generation.get() = (*self.generation.get()).wrapping_add(GENERATION_DELTA);
        }
    }
    pub fn clear(&self) {
        for b in &self.buckets {
            for e in &b.entries {
                e.clear();
            }
        }
        unsafe {
            *self.generation.get() = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_basic() {
        let tt = SharedTranspositionTable::new(1);
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
    }

    #[test]
    fn test_move_roundtrip() {
        let tt = SharedTranspositionTable::new(1);
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
}
