use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::INFINITY;
use super::tt_defs::{
    TTFlag, TTProbeParams, TTProbeResult, TTStoreParams, eval_from_i16, eval_to_i16, pack_coord,
    score_from_i16, score_to_i16, unpack_coord, value_from_tt, value_to_tt,
};

const ENTRIES_PER_BUCKET: usize = 4;

const GENERATION_BITS: u8 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
#[allow(clippy::identity_op)]
const GENERATION_MASK: u8 = (0xFF << GENERATION_BITS) & 0xFF;

const NO_MOVE: u64 = 0;

use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, AtomicU64, Ordering};

/// Relaxed, per-field access. Concurrent readers and writers can interleave, so a
/// probe may return a self-inconsistent copy; every consumer must tolerate that.
/// Fields are separate so refreshing the generation cannot disturb the payload.
const REL: Ordering = Ordering::Relaxed;

// TT entry structure uses 16 bytes: key16 | depth8 | gen_bound8 | score16 | eval16,
// plus the packed 13-bit-coordinate move (64 bits).

#[repr(C, align(8))]
pub struct TTEntry {
    key16: AtomicU16,
    depth8: AtomicU8,
    gen_bound8: AtomicU8,
    score16: AtomicI16,
    eval16: AtomicI16,
    move_data: AtomicU64,
}

use super::tt_defs::{MAX_TT_COORD, MIN_TT_COORD};

impl TTEntry {
    #[inline]
    pub fn read(&self, key16: u16, params_hash: u64) -> Option<(i32, i32, u8, u8, Option<Move>)> {
        {
            if self.key16.load(REL) != key16 {
                return None;
            }

            let d = self.depth8.load(REL);
            let gb = self.gen_bound8.load(REL);
            let score = self.score16.load(REL) as i32;
            let eval = self.eval16.load(REL) as i32;

            // A never-written entry is all zeroes, which only survives the key check
            // when the probing key is itself zero.
            if key16 == 0 && d == 0 && gb == 0 && score == 0 && eval == 0 {
                return None;
            }

            let mdata = self.move_data.load(REL);

            // XOR the probing key into move_data: a move stored by a colliding/other position
            // decodes to garbage and gets rejected by the guards below.
            let hash_key = params_hash >> 16; // Use matching bits from the full hash
            let decoded_mdata = mdata ^ hash_key;

            let pt_raw = (decoded_mdata & 0x1F) as u8;
            let cl_raw = ((decoded_mdata >> 5) & 0x03) as u8;
            let pr = ((decoded_mdata >> 7) & 0x1F) as u8;

            // Guarded decode: invalid discriminants (torn/foreign move) → no move, keep the hit.
            let best_move = if decoded_mdata == NO_MOVE
                || pt_raw > PieceType::Pawn as u8
                || cl_raw > PlayerColor::Black as u8
                || pr > PieceType::Pawn as u8
            {
                None
            } else {
                let pt = PieceType::from_u8(pt_raw);
                let cl = PlayerColor::from_u8(cl_raw);

                let fx = unpack_coord(decoded_mdata >> 12);
                let fy = unpack_coord(decoded_mdata >> 25);
                let tx = unpack_coord(decoded_mdata >> 38);
                let ty = unpack_coord(decoded_mdata >> 51);

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
        hash: u64,
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

        // XOR the hash key into move_data for integrity
        let hash_key = hash >> 16;
        let protected_mdata = mdata ^ hash_key;

        // Payload before key: a reader that matches the key then sees data for this
        // position rather than the previous occupant's.
        self.move_data.store(protected_mdata, REL);
        self.depth8.store(depth, REL);
        self.gen_bound8.store(gen_bound, REL);
        self.score16.store(score, REL);
        self.eval16.store(eval, REL);
        self.key16.store(key16, REL);
    }

    #[inline]
    pub fn clear(&self) {
        self.key16.store(0, REL);
        self.depth8.store(0, REL);
        self.gen_bound8.store(0, REL);
        self.score16.store(0, REL);
        self.eval16.store(0, REL);
        self.move_data.store(0, REL);
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

pub struct SharedTranspositionTable {
    buckets: Vec<TTBucket>,
    mask: usize,
    index_bits: u32,
    generation: AtomicU8,
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

        // Zeroed allocation, no memset: all-zero bytes are the empty state, and calloc'd
        // pages fault in lazily, so construction cost is independent of hash size.
        let buckets: Vec<TTBucket> = unsafe {
            let layout = std::alloc::Layout::array::<TTBucket>(cap).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut TTBucket;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, cap, cap)
        };

        SharedTranspositionTable {
            buckets,
            mask: cap - 1,
            index_bits: bits,
            generation: AtomicU8::new(1),
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
        let r#gen = self.generation.load(REL);
        let mut occ = 0u32;
        for i in 0..sample {
            for e in &self.buckets[i].entries {
                let gb = e.gen_bound8.load(REL);
                let occupied =
                    e.key16.load(REL) != 0 || gb != 0 || e.depth8.load(REL) != 0;
                if occupied && TTEntry::generation(gb) == r#gen {
                    occ += 1;
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
            if let Some((_, _, _, _, m)) = e.read(key16, hash) {
                return m;
            }
        }
        None
    }

    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        let key16 = self.hash_key16(params.hash);
        for e in &self.buckets[self.bucket_index(params.hash)].entries {
            if let Some((score, eval, depth, gen_bound, best_move)) = e.read(key16, params.hash) {
                // Keeping used entries current-generation lets them win replacement
                // fights while untouched ones age out. Only this byte is written, so a
                // concurrent store's score and depth cannot be reverted by the refresh.
                let r#gen = self.generation.load(REL);
                let gb = e.gen_bound8.load(REL);
                e.gen_bound8
                    .store((r#gen & GENERATION_MASK) | (gb & 0x07), REL);
                let score = value_from_tt(
                    score_from_i16(score),
                    params.ply,
                    params.rule50_count,
                    params.rule_limit,
                );
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
                    eval: eval_from_i16(eval),
                    depth,
                    flag,
                    is_pv: TTEntry::is_pv(gen_bound),
                    best_move,
                });
            }
        }
        None
    }

    /// Shaves plies off an entry that was deep enough to cut but carried the wrong
    /// bound, so a real search can replace it instead of it being re-probed for a
    /// cutoff it can never give.
    pub fn penalize(&self, hash: u64, penalty: u8) {
        let key16 = self.hash_key16(hash);
        for e in &self.buckets[self.bucket_index(hash)].entries {
            if e.key16.load(REL) == key16 {
                let d = e.depth8.load(REL);
                e.depth8.store(d.saturating_sub(penalty), REL);
                return;
            }
        }
    }

    /// Stores an entry in the multithreaded table.
    /// Priority is given to deeper searches and newer generation entries.
    pub fn store(&self, params: &TTStoreParams) {
        let key16 = self.hash_key16(params.hash);
        let adj_score = value_to_tt(params.score, params.ply);
        let r#gen = self.generation.load(REL);
        let bucket = &self.buckets[self.bucket_index(params.hash)];

        let mut replace_idx = 0;
        let mut worst = i32::MAX;

        for (i, e) in bucket.entries.iter().enumerate() {
            let e_key = e.key16.load(REL);
            let old_depth = e.depth8.load(REL);
            let old_gb = e.gen_bound8.load(REL);

            // Check if key matches (and entry is not empty)
            let empty = e_key == 0
                && old_depth == 0
                && old_gb == 0
                && e.score16.load(REL) == 0
                && e.eval16.load(REL) == 0;
            if e_key == key16 && !empty {
                let mdata = e.move_data.load(REL);
                let old_eval = e.eval16.load(REL);

                // Decode old move for preservation
                let old_move_data = mdata ^ (params.hash >> 16);

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
                    eval_to_i16(params.static_eval)
                } else {
                    old_eval
                };

                let old_gen = old_gb & GENERATION_MASK;
                let pv_bonus = if params.flag == TTFlag::Exact || params.is_pv {
                    2
                } else {
                    0
                };
                let rel_age = (r#gen.wrapping_sub(old_gen)) & GENERATION_MASK;

                if params.flag == TTFlag::Exact
                    || (params.depth as i32 + pv_bonus) > (old_depth as i32 - 4)
                    || rel_age != 0
                {
                    e.move_data
                        .store(mdata_to_write ^ (params.hash >> 16), REL);
                    e.depth8.store(params.depth as u8, REL);
                    e.gen_bound8.store(
                        TTEntry::pack_gen_bound(r#gen, params.is_pv, params.flag),
                        REL,
                    );
                    e.score16.store(score_to_i16(adj_score), REL);
                    e.eval16.store(store_eval, REL);
                    e.key16.store(key16, REL);
                } else if old_depth >= 5
                    && TTEntry::flag(old_gb) != TTFlag::Exact
                    && super::is_decisive(score_from_i16(e.score16.load(REL) as i32))
                {
                    // Only a decisive bound decays. Aging ordinary deep bounds costs
                    // cutoffs table-wide; a stale mate bound is what has to lose depth
                    // so a fresher search can replace it.
                    e.depth8.store(old_depth - 1, REL);
                }
                return;
            }

            // Calculation for replacement strategy
            let ed = old_depth;
            let egb = old_gb;
            let rel_age = (r#gen.wrapping_sub(egb & GENERATION_MASK)) & GENERATION_MASK;

            // Age weighted by 1 (matches the local TT).
            let mut prio =
                (ed as i32 + 3 + if TTEntry::is_pv(egb) { 2 } else { 0 }) - rel_age as i32;
            if e_key == 0 && egb == 0 {
                // Is empty check
                prio = i32::MIN;
            }
            if prio < worst {
                worst = prio;
                replace_idx = i;
            }
        }

        bucket.entries[replace_idx].write(
            key16,
            score_to_i16(adj_score),
            eval_to_i16(params.static_eval),
            params.depth as u8,
            TTEntry::pack_gen_bound(r#gen, params.is_pv, params.flag),
            &params.best_move,
            params.hash,
        );
    }

    pub fn increment_age(&self) {
        self.generation
            .store(self.generation.load(REL).wrapping_add(GENERATION_DELTA), REL);
    }
    pub fn clear(&self) {
        for b in &self.buckets {
            for e in &b.entries {
                e.clear();
            }
        }
        self.generation.store(1, REL);
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

    #[test]
    fn test_move_survives_best_move_none_restore() {
        // A key-match store with best_move=None must preserve the existing move: it
        // has to un-protect the stored move_data before the final re-protect, or the
        // double XOR turns it into garbage.
        let tt = SharedTranspositionTable::new(1);
        let hash = 0x1357924680ABCDEFu64;
        let m = Move {
            from: Coordinate::new(3, 1),
            to: Coordinate::new(3, 3),
            piece: Piece::new(PieceType::Pawn, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };
        tt.store(&TTStoreParams {
            hash,
            depth: 8,
            flag: TTFlag::Exact,
            score: 20,
            static_eval: 15,
            is_pv: true,
            best_move: Some(m),
            ply: 0,
        });
        // A deeper store with no move must take the key-match path and keep the move.
        tt.store(&TTStoreParams {
            hash,
            depth: 9,
            flag: TTFlag::Exact,
            score: 25,
            static_eval: 15,
            is_pv: true,
            best_move: None,
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
        let decoded = res
            .best_move
            .expect("move should survive a best_move=None re-store");
        assert_eq!(decoded.from, m.from);
        assert_eq!(decoded.to, m.to);
    }

    #[test]
    fn test_shallow_store_does_not_clobber_deep_entry() {
        // A depth-0 store, such as an eager static-eval store, must not evict the
        // metadata of a deep same-generation entry.
        let tt = SharedTranspositionTable::new(1);
        let hash = 0x1122334455667788u64;
        tt.store(&TTStoreParams {
            hash,
            depth: 10,
            flag: TTFlag::Exact,
            score: 100,
            static_eval: 90,
            is_pv: false,
            best_move: None,
            ply: 0,
        });
        // Same generation, non-exact, shallow: none of the replacement conditions
        // should be met, so this must NOT overwrite the deep entry above.
        tt.store(&TTStoreParams {
            hash,
            depth: 0,
            flag: TTFlag::UpperBound,
            score: -500,
            static_eval: -500,
            is_pv: false,
            best_move: None,
            ply: 0,
        });
        let res = tt
            .probe(&TTProbeParams {
                hash,
                alpha: -1000,
                beta: 1000,
                depth: 10,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            })
            .unwrap();
        assert_eq!(
            res.depth, 10,
            "a depth-0 store must not clobber a deep entry's metadata"
        );
        assert_eq!(res.tt_score, 100);
        assert_eq!(res.flag, TTFlag::Exact);
    }

    /// Secondary aging is confined to decisive bounds. An ordinary deep bound must
    /// keep its depth, or every deep entry decays and cutoffs are lost table-wide.
    #[test]
    fn secondary_aging_only_decays_decisive_bounds() {
        // A depth-5 store against a stored depth 10 fails every overwrite condition
        // (non-exact, 5 <= 10 - 4, same generation), so the aging branch runs.
        fn probe_depth(tt: &SharedTranspositionTable, hash: u64) -> u8 {
            tt.probe(&TTProbeParams {
                hash,
                alpha: -30_000,
                beta: 30_000,
                depth: 1,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            })
            .expect("entry present")
            .depth
        }
        fn seed(tt: &SharedTranspositionTable, hash: u64, score: i32) {
            tt.store(&TTStoreParams {
                hash,
                depth: 10,
                flag: TTFlag::LowerBound,
                score,
                static_eval: 0,
                is_pv: false,
                best_move: None,
                ply: 0,
            });
        }
        fn shallow(tt: &SharedTranspositionTable, hash: u64) {
            tt.store(&TTStoreParams {
                hash,
                depth: 5,
                flag: TTFlag::LowerBound,
                score: 0,
                static_eval: 0,
                is_pv: false,
                best_move: None,
                ply: 0,
            });
        }

        let tt = SharedTranspositionTable::new(1);

        // Ordinary score: must NOT decay.
        let quiet = 0xABCD_0000_1111_2222u64;
        seed(&tt, quiet, 120);
        assert_eq!(probe_depth(&tt, quiet), 10);
        shallow(&tt, quiet);
        assert_eq!(
            probe_depth(&tt, quiet),
            10,
            "an ordinary deep bound must keep its depth"
        );

        // Decisive score: must decay by one.
        let mate = 0x1234_5555_6666_7777u64;
        seed(&tt, mate, crate::search::MATE_VALUE - 10);
        assert_eq!(probe_depth(&tt, mate), 10);
        shallow(&tt, mate);
        assert_eq!(
            probe_depth(&tt, mate),
            9,
            "a stale mate bound must lose a ply so a fresher search can replace it"
        );
    }

    /// The generation refresh on probe writes only its own byte. When it shared a
    /// packed word with the score, a probe racing a store wrote the pre-store value
    /// back, so a just-proven mate score could silently revert.
    #[test]
    fn concurrent_probe_does_not_revert_a_store() {
        use std::sync::Arc;
        const N: i32 = 20_000;
        let hash = 0x9E3779B97F4A7C15u64;

        fn store_n(tt: &SharedTranspositionTable, hash: u64, v: i32) {
            tt.store(&TTStoreParams {
                hash,
                depth: 8,
                flag: TTFlag::Exact,
                score: v,
                static_eval: v,
                is_pv: false,
                best_move: None,
                ply: 0,
            });
        }
        fn probe(tt: &SharedTranspositionTable, hash: u64) -> Option<TTProbeResult> {
            tt.probe(&TTProbeParams {
                hash,
                alpha: -30_000,
                beta: 30_000,
                depth: 8,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            })
        }

        // Repeat: the interleaving that reverts a store is timing dependent.
        for _round in 0..8 {
            let tt = Arc::new(SharedTranspositionTable::new(1));
            let writer = {
                let tt = Arc::clone(&tt);
                std::thread::spawn(move || {
                    for i in 1..=N {
                        store_n(&tt, hash, i);
                    }
                })
            };
            let reader = {
                let tt = Arc::clone(&tt);
                std::thread::spawn(move || {
                    let mut seen_max = 0;
                    for _ in 0..N * 2 {
                        if let Some(r) = probe(&tt, hash) {
                            // A score must never move backwards once observed.
                            assert!(
                                r.tt_score >= seen_max,
                                "score went backwards: {} after {seen_max}",
                                r.tt_score
                            );
                            seen_max = r.tt_score;
                        }
                    }
                })
            };
            writer.join().expect("writer");
            reader.join().expect("reader");

            let final_score = probe(&tt, hash).expect("entry present").tt_score;
            assert_eq!(
                final_score, N,
                "the last store must survive concurrent probe refreshes"
            );
        }
    }
}

const _: () = assert!(std::mem::size_of::<TTEntry>() == 16);
