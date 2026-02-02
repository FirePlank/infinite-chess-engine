use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::INFINITY;
use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, clamp_to_i16, pack_coord, unpack_coord,
    value_from_tt, value_to_tt,
};

const ENTRIES_PER_BUCKET: usize = 4;

const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
#[allow(clippy::identity_op)]
const GENERATION_MASK: u8 = (0xFF << GENERATION_BITS) & 0xFF;

const NO_MOVE: u64 = 0;

use std::cell::UnsafeCell;

// TT entry structure uses 16 bytes.
// Metadata: key16 | depth8 | gen_bound8 | score16 | eval16 (64 bits)
// Move: Packed pieces and 13-bit coordinates (64 bits)

#[repr(C, align(8))]
pub struct TTEntry {
    metadata: UnsafeCell<u64>,
    move_data: UnsafeCell<u64>,
}

unsafe impl Sync for TTEntry {}
unsafe impl Send for TTEntry {}

use super::tt_defs::{MAX_TT_COORD, MIN_TT_COORD};

impl TTEntry {
    pub fn empty() -> Self {
        TTEntry {
            metadata: UnsafeCell::new(0),
            move_data: UnsafeCell::new(NO_MOVE),
        }
    }

    #[inline]
    pub fn read(&self, key16: u16) -> Option<(i32, i32, u8, u8, Option<Move>)> {
        unsafe {
            let meta = std::ptr::read_volatile(self.metadata.get());
            if (meta & 0xFFFF) as u16 != key16 || meta == 0 {
                return None;
            }

            let mdata = std::ptr::read_volatile(self.move_data.get());
            if std::ptr::read_volatile(self.metadata.get()) != meta {
                return None;
            }

            let d = (meta >> 16) as u8;
            let gb = (meta >> 24) as u8;
            let score = (meta >> 32) as u16 as i16 as i32;
            let eval = (meta >> 48) as u16 as i16 as i32;

            let best_move = if mdata == NO_MOVE {
                None
            } else {
                let pt = PieceType::from_u8((mdata & 0x1F) as u8);
                let cl = PlayerColor::from_u8(((mdata >> 5) & 0x03) as u8);
                let pr = ((mdata >> 7) & 0x1F) as u8;

                let fx = unpack_coord(mdata >> 12);
                let fy = unpack_coord(mdata >> 25);
                let tx = unpack_coord(mdata >> 38);
                let ty = unpack_coord(mdata >> 51);

                Some(Move {
                    from: Coordinate { x: fx, y: fy },
                    to: Coordinate { x: tx, y: ty },
                    piece: Piece::new(pt, cl),
                    promotion: if pr == 0 {
                        None
                    } else {
                        Some(PieceType::from_u8(pr))
                    },
                    rook_coord: None,
                })
            };

            Some((score, eval, d, gb, best_move))
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
        best_move: &Option<Move>,
    ) {
        let mdata = if let Some(m) = best_move {
            if m.from.x >= MIN_TT_COORD
                && m.from.x <= MAX_TT_COORD
                && m.from.y >= MIN_TT_COORD
                && m.from.y <= MAX_TT_COORD
                && m.to.x >= MIN_TT_COORD
                && m.to.x <= MAX_TT_COORD
                && m.to.y >= MIN_TT_COORD
                && m.to.y <= MAX_TT_COORD
            {
                let pt = m.piece.piece_type() as u64;
                let cl = m.piece.color() as u64;
                let pr = m.promotion.map_or(0, |p| p as u64);
                (pt & 0x1F)
                    | ((cl & 0x03) << 5)
                    | ((pr & 0x1F) << 7)
                    | (pack_coord(m.from.x) << 12)
                    | (pack_coord(m.from.y) << 25)
                    | (pack_coord(m.to.x) << 38)
                    | (pack_coord(m.to.y) << 51)
            } else {
                NO_MOVE
            }
        } else {
            NO_MOVE
        };

        let meta = (key16 as u64)
            | ((depth as u64) << 16)
            | ((gen_bound as u64) << 24)
            | (((score as u16) as u64) << 32)
            | (((eval as u16) as u64) << 48);

        unsafe {
            std::ptr::write_volatile(self.move_data.get(), mdata);
            std::ptr::write_volatile(self.metadata.get(), meta);
        }
    }

    #[inline]
    pub fn raw_key(&self) -> u16 {
        unsafe { (std::ptr::read_volatile(self.metadata.get()) & 0xFFFF) as u16 }
    }
    #[inline]
    pub fn raw_depth(&self) -> u8 {
        unsafe { (std::ptr::read_volatile(self.metadata.get()) >> 16) as u8 }
    }
    #[inline]
    pub fn raw_gen_bound(&self) -> u8 {
        unsafe { (std::ptr::read_volatile(self.metadata.get()) >> 24) as u8 }
    }

    #[inline]
    pub fn clear(&self) {
        unsafe {
            std::ptr::write_volatile(self.metadata.get(), 0);
        }
    }
    #[inline]
    pub fn flag(gen_bound: u8) -> TTFlag {
        TTFlag::from_u8(gen_bound & 0x03)
    }
    #[inline]
    pub fn is_pv(gen_bound: u8) -> bool {
        (gen_bound & 0x04) != 0
    }
    #[inline]
    pub fn generation(gen_bound: u8) -> u8 {
        gen_bound & GENERATION_MASK
    }
    #[inline]
    pub fn pack_gen_bound(r#gen: u8, is_pv: bool, flag: TTFlag) -> u8 {
        (r#gen & GENERATION_MASK) | (if is_pv { 0x04 } else { 0 }) | (flag as u8 & 0x03)
    }
}

#[repr(C, align(64))]
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
                let meta = unsafe { std::ptr::read_volatile(e.metadata.get()) };
                if meta != 0 {
                    let gb = (meta >> 24) as u8;
                    if TTEntry::generation(gb) == r#gen {
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

        let mut replace_idx = 0;
        let mut worst = i32::MAX;

        for (i, e) in bucket.entries.iter().enumerate() {
            // Read metadata ONCE strictly for this iteration
            let meta = unsafe { std::ptr::read_volatile(e.metadata.get()) };

            // Check if key matches (and entry is not empty)
            if (meta & 0xFFFF) as u16 == key16 && meta != 0 {
                let mdata = unsafe { std::ptr::read_volatile(e.move_data.get()) };

                // Verify consistency: re-read metadata and fail if changed.
                let old_depth = (meta >> 16) as u8;
                let old_gb = (meta >> 24) as u8;
                let old_eval = (meta >> 48) as i16;

                // Decode old move for preservation
                let old_move_data = mdata;

                let store_move = params.best_move.as_ref();
                let mdata_to_write = if let Some(m) = store_move {
                    // Encode new move
                    if m.from.x >= MIN_TT_COORD
                        && m.from.x <= MAX_TT_COORD
                        && m.from.y >= MIN_TT_COORD
                        && m.from.y <= MAX_TT_COORD
                        && m.to.x >= MIN_TT_COORD
                        && m.to.x <= MAX_TT_COORD
                        && m.to.y >= MIN_TT_COORD
                        && m.to.y <= MAX_TT_COORD
                    {
                        let pt = m.piece.piece_type() as u64;
                        let cl = m.piece.color() as u64;
                        let pr = m.promotion.map_or(0, |p| p as u64);
                        (pt & 0x1F)
                            | ((cl & 0x03) << 5)
                            | ((pr & 0x1F) << 7)
                            | (pack_coord(m.from.x) << 12)
                            | (pack_coord(m.from.y) << 25)
                            | (pack_coord(m.to.x) << 38)
                            | (pack_coord(m.to.y) << 51)
                    } else {
                        NO_MOVE
                    }
                } else {
                    old_move_data
                };

                let store_eval = if params.static_eval != INFINITY + 1 {
                    clamp_to_i16(params.static_eval)
                } else {
                    old_eval
                };

                let old_gen = TTEntry::generation(old_gb);
                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };
                let rel_age = (r#gen.wrapping_sub(old_gen)) & GENERATION_MASK;

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (old_depth as i32 - 4)
                    || rel_age != 0
                    || params.depth == 0
                {
                    let new_meta = (key16 as u64)
                        | ((params.depth as u64) << 16)
                        | ((TTEntry::pack_gen_bound(r#gen, params.is_pv, params.flag) as u64)
                            << 24)
                        | (((clamp_to_i16(adj_score) as u16) as u64) << 32)
                        | (((store_eval as u16) as u64) << 48);

                    unsafe {
                        std::ptr::write_volatile(e.move_data.get(), mdata_to_write);
                        std::ptr::write_volatile(e.metadata.get(), new_meta);
                    }
                }
                return;
            }

            // Calculation for replacement strategy
            let ed = (meta >> 16) as u8;
            let egb = (meta >> 24) as u8;
            let rel_age = (r#gen.wrapping_sub(TTEntry::generation(egb))) & GENERATION_MASK;

            // Fix: remove * 2 multiplier to match Stockfish/LocalTT
            let mut prio = ed as i32 - (rel_age as i32);

            if TTEntry::flag(egb) == TTFlag::Exact || TTEntry::is_pv(egb) {
                prio += 2;
            }
            if (meta & 0xFFFF) == 0 && egb == 0 {
                // Is empty check
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
                &params.best_move,
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
