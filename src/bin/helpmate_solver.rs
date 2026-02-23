//! Helpmate Solver for Infinite Chess
//!
//! A cooperative chess problem solver where both sides work together to achieve checkmate.
//! Uses parallel exhaustive search with a thread-safe Transposition Table.

use hydrochess_wasm::{
    board::{Coordinate, PlayerColor},
    game::GameState,
    moves::{Move, MoveList},
    search::{INFINITY, MATE_VALUE},
};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// TRANSPOSITION TABLE
// ============================================================================

mod parallel_tt {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const PN_INF: u32 = 1_000_000_000;
    const ENTRIES_PER_BUCKET: usize = 4;

    pub struct TTEntry {
        pub key: AtomicU64,
        pub data1: AtomicU64,
        pub data2: AtomicU64,
        pub data3: AtomicU64,
    }

    impl TTEntry {
        pub fn new() -> Self {
            TTEntry {
                key: AtomicU64::new(0),
                data1: AtomicU64::new(2013265922), // Pack 1 and 1 as default PN/DN
                data2: AtomicU64::new(0),
                data3: AtomicU64::new(0),
            }
        }
    }

    pub struct TTBucket {
        pub entries: [TTEntry; ENTRIES_PER_BUCKET],
    }

    pub struct TranspositionTable {
        buckets: Vec<TTBucket>,
        mask: usize,
    }

    unsafe impl Sync for TranspositionTable {}
    unsafe impl Send for TranspositionTable {}

    impl TranspositionTable {
        pub fn new(size_mb: usize) -> Self {
            let bytes = size_mb.max(1) * 1024 * 1024;
            let bucket_size = std::mem::size_of::<TTBucket>();
            let num_buckets = (bytes / bucket_size).max(1).next_power_of_two();
            let mut buckets = Vec::with_capacity(num_buckets);
            for _ in 0..num_buckets {
                buckets.push(TTBucket {
                    entries: [
                        TTEntry::new(),
                        TTEntry::new(),
                        TTEntry::new(),
                        TTEntry::new(),
                    ],
                });
            }
            TranspositionTable {
                buckets,
                mask: num_buckets - 1,
            }
        }

        #[inline]
        fn bucket_idx(&self, hash: u64) -> usize {
            (hash as usize) & self.mask
        }

        pub fn probe(&self, hash: u64) -> Option<(u32, u32, Option<(i16, i16, i16, i16)>, u32)> {
            let bucket = &self.buckets[self.bucket_idx(hash)];

            for e in &bucket.entries {
                let key = e.key.load(Ordering::Relaxed);
                if key == 0 {
                    continue;
                }

                let d1 = e.data1.load(Ordering::Relaxed);
                let d2 = e.data2.load(Ordering::Relaxed);
                let d3 = e.data3.load(Ordering::Relaxed);

                if (key ^ d1 ^ d2 ^ d3) == hash {
                    let pn = (d1 & 0xFFFFFFFF) as u32;
                    let dn = ((d1 >> 32) & 0xFFFFFFFF) as u32;
                    let from_x = (d2 & 0xFFFF) as i16;
                    let from_y = ((d2 >> 16) & 0xFFFF) as i16;
                    let to_x = ((d2 >> 32) & 0xFFFF) as i16;
                    let to_y = ((d2 >> 48) & 0xFFFF) as i16;

                    let move_coords = if from_x == 0 && from_y == 0 && to_x == 0 && to_y == 0 {
                        None
                    } else {
                        Some((from_x, from_y, to_x, to_y))
                    };

                    let depth = (d3 & 0xFFFFFFFF) as u32;
                    return Some((pn, dn, move_coords, depth));
                }
            }
            None
        }

        pub fn store(
            &self,
            hash: u64,
            pn: u32,
            dn: u32,
            depth: u32,
            m: Option<(i16, i16, i16, i16)>,
        ) {
            let bucket = &self.buckets[self.bucket_idx(hash)];

            let mut replace_idx = 0;
            let mut found_slot = false;

            for (i, e) in bucket.entries.iter().enumerate() {
                let key = e.key.load(Ordering::Relaxed);
                if key == 0 {
                    replace_idx = i;
                    found_slot = true;
                    break;
                }

                let d1 = e.data1.load(Ordering::Relaxed);
                let d2 = e.data2.load(Ordering::Relaxed);
                let d3 = e.data3.load(Ordering::Relaxed);

                if (key ^ d1 ^ d2 ^ d3) == hash {
                    replace_idx = i;
                    found_slot = true;
                    break;
                }
            }

            if !found_slot {
                let mut min_work_depth = u32::MAX;
                let mut replace_candidate = (hash as usize >> 32) % ENTRIES_PER_BUCKET;
                for (i, e) in bucket.entries.iter().enumerate() {
                    let e_d3 = e.data3.load(Ordering::Relaxed);
                    let e_depth = (e_d3 & 0xFFFFFFFF) as u32;

                    if e_depth < min_work_depth {
                        min_work_depth = e_depth;
                        replace_candidate = i;
                    }
                }
                replace_idx = replace_candidate;
            }

            let entry = &bucket.entries[replace_idx];

            let (from_x, from_y, to_x, to_y) = if let Some((fx, fy, tx, ty)) = m {
                (
                    fx as u16 as u64,
                    fy as u16 as u64,
                    tx as u16 as u64,
                    ty as u16 as u64,
                )
            } else {
                (0, 0, 0, 0)
            };

            let d1 = (pn as u64) | ((dn as u64) << 32);
            let d2 = from_x | (from_y << 16) | (to_x << 32) | (to_y << 48);
            let d3 = depth as u64;
            let new_key = hash ^ d1 ^ d2 ^ d3;

            // Invalidate key first
            entry.key.store(0, Ordering::Relaxed);

            entry.data1.store(d1, Ordering::Relaxed);
            entry.data2.store(d2, Ordering::Relaxed);
            entry.data3.store(d3, Ordering::Relaxed);

            // Write key last
            entry.key.store(new_key, Ordering::Release);
        }
    }
}

// ============================================================================
// SOLVER
// ============================================================================

struct HelpmateSolver {
    tt: parallel_tt::TranspositionTable,
    nodes: AtomicU64,
    found_mate: AtomicBool,
    target_depth: u32,
    target_mated_side: PlayerColor,
    killers: Vec<[AtomicU64; 2]>,
    history: Vec<AtomicI32>,
}

impl HelpmateSolver {
    pub fn new(mate_in: u32, target_mated_side: PlayerColor) -> Self {
        let mut history = Vec::with_capacity(32768);
        for _ in 0..32768 {
            history.push(AtomicI32::new(0));
        }
        let mut killers = Vec::with_capacity(64);
        for _ in 0..64 {
            killers.push([AtomicU64::new(0), AtomicU64::new(0)]);
        }
        Self {
            tt: parallel_tt::TranspositionTable::new(128),
            target_depth: mate_in,
            target_mated_side,
            nodes: AtomicU64::new(0),
            found_mate: AtomicBool::new(false),
            history,
            killers,
        }
    }

    fn solve(&mut self, state: &mut GameState) -> Option<i32> {
        self.found_mate.store(false, Ordering::SeqCst);

        // Force Hash Consistency at Root
        state.recompute_hash();

        let start = Instant::now();

        // Iterative Deepening DFS
        for current_target in 1..=self.target_depth {
            // Pre-collect work items to avoid Sync issues with GameState capture
            let moves = self.generate_helpmate_moves(state, current_target as i32);
            let work_items: Vec<(GameState, Move)> = moves
                .iter()
                .filter_map(|m| {
                    let mut local_state = state.clone();
                    let _undo = local_state.make_move(m);
                    if local_state.is_move_illegal() {
                        return None;
                    }
                    Some((local_state, *m))
                })
                .collect();

            if work_items.is_empty() {
                continue;
            }

            let results: Vec<(u32, u32, Move)> = work_items
                .into_par_iter()
                .map(|(mut local_state, m)| {
                    let (rpn, rdn) = self.mid_node(
                        &mut local_state,
                        parallel_tt::PN_INF,
                        parallel_tt::PN_INF,
                        1,
                        current_target,
                    );
                    (rpn, rdn, m)
                })
                .collect();

            // Aggregate results for root PN/DN
            let mut min_pn = parallel_tt::PN_INF;
            let mut sum_dn: u64 = 0;
            let mut best_move = None;

            for (rpn, rdn, m) in results {
                if rpn < min_pn {
                    min_pn = rpn;
                    best_move = Some(m);
                }
                sum_dn += rdn as u64;
            }
            let pn = min_pn;
            let dn = sum_dn.min(parallel_tt::PN_INF as u64) as u32;

            let nodes = self.nodes.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            println!(
                "info depth {} nodes {} time {:.2}s nps {:.0} pn {} dn {}",
                current_target,
                nodes,
                elapsed,
                nodes as f64 / elapsed,
                pn,
                dn
            );

            if pn == 0 {
                if let Some(m) = best_move {
                    let hash = state.hash;
                    self.tt.store(
                        hash,
                        0,
                        parallel_tt::PN_INF,
                        current_target,
                        Some((
                            m.from.x as i16,
                            m.from.y as i16,
                            m.to.x as i16,
                            m.to.y as i16,
                        )),
                    );
                }
                self.found_mate.store(true, Ordering::SeqCst);
                return Some(MATE_VALUE);
            }

            if elapsed > 300.0 {
                break;
            }
        }

        None
    }

    fn mid_node(
        &self,
        state: &mut GameState,
        pn_limit: u32,
        dn_limit: u32,
        ply: u32,
        max_plies: u32,
    ) -> (u32, u32) {
        self.nodes.fetch_add(1, Ordering::Relaxed);

        if self.found_mate.load(Ordering::Relaxed) {
            return (0, parallel_tt::PN_INF);
        }

        // Helpmate is purely cooperative -> Every node behaves as an OR node.
        // PN = min(child PN), DN = sum(child DN)
        let depth_left = max_plies.saturating_sub(ply);

        let hash = state.hash;
        let mut tt_move_coords = None;
        if let Some((pn, dn, move_coords, cached_depth)) = self.tt.probe(hash) {
            tt_move_coords = move_coords;
            let use_bounds = cached_depth >= depth_left || pn == 0;
            if use_bounds {
                if pn >= pn_limit || dn >= dn_limit {
                    return (pn, dn);
                }
            }
        }

        let target_king_pos = match self.target_mated_side {
            PlayerColor::White => state.white_king_pos.unwrap_or(Coordinate::new(0, 0)),
            PlayerColor::Black => state.black_king_pos.unwrap_or(Coordinate::new(0, 0)),
            _ => Coordinate::new(0, 0),
        };

        // Isolation Pruning: If king is too far from any piece, mate is impossible.
        if ply > 0 && self.is_king_isolated(state, target_king_pos, depth_left as i32) {
            let res = (parallel_tt::PN_INF, 0);
            self.tt.store(hash, res.0, res.1, depth_left, None);
            return res;
        }

        if ply >= max_plies {
            let score = self.terminal_score(state, ply);
            let (pn, dn) = if score >= MATE_VALUE - 1000 {
                (0, parallel_tt::PN_INF)
            } else {
                (parallel_tt::PN_INF, 0)
            };
            self.tt.store(hash, pn, dn, depth_left, None);
            return (pn, dn);
        }

        // Depth 1 Optimization: specialized mating search
        if ply + 1 == max_plies && state.turn != self.target_mated_side {
            if let Some((_score, m)) = self.find_mating_move(state, ply) {
                let res = (0, parallel_tt::PN_INF);
                self.tt.store(
                    hash,
                    res.0,
                    res.1,
                    depth_left,
                    Some((
                        m.from.x as i16,
                        m.from.y as i16,
                        m.to.x as i16,
                        m.to.y as i16,
                    )),
                );
                return res;
            } else {
                // If we checked all moves and found no mate involved, we failed.
                let res = (parallel_tt::PN_INF, 0);
                self.tt.store(hash, res.0, res.1, depth_left, None);
                return res;
            }
        }

        let moves = self.generate_helpmate_moves(state, depth_left as i32);
        if moves.is_empty() {
            let score = self.terminal_score(state, ply);
            let (pn, dn) = if score >= MATE_VALUE - 1000 {
                (0, parallel_tt::PN_INF)
            } else {
                (parallel_tt::PN_INF, 0)
            };
            self.tt.store(hash, pn, dn, depth_left, None);
            return (pn, dn);
        }

        let ply_idx = (ply as usize).min(63);
        let k1 = self.killers[ply_idx][0].load(Ordering::Relaxed);
        let k2 = self.killers[ply_idx][1].load(Ordering::Relaxed);

        let mut scored_moves: Vec<(Move, i32)> = moves
            .into_iter()
            .map(|m| {
                let mut score = 0;
                let m_val = (m.from.x as u16 as u64)
                    | ((m.from.y as u16 as u64) << 16)
                    | ((m.to.x as u16 as u64) << 32)
                    | ((m.to.y as u16 as u64) << 48);

                if let Some((fx, fy, tx, ty)) = tt_move_coords {
                    if m.from.x as i16 == fx
                        && m.from.y as i16 == fy
                        && m.to.x as i16 == tx
                        && m.to.y as i16 == ty
                    {
                        score += 10_000_000;
                    }
                }
                if m_val == k1 {
                    score += 5_000_000;
                } else if m_val == k2 {
                    score += 4_000_000;
                }

                let h_idx = (((((m.from.x as u32).wrapping_mul(31) ^ (m.from.y as u32))
                    .wrapping_mul(31)
                    ^ (m.to.x as u32))
                    .wrapping_mul(31)
                    ^ (m.to.y as u32)) as usize)
                    & 32767;
                score += self.history[h_idx].load(Ordering::Relaxed);

                (m, score)
            })
            .collect();
        scored_moves.sort_unstable_by_key(|(_, s)| -s);

        let mut total_dn: u64 = 0;
        let mut best_pn = parallel_tt::PN_INF;

        for (i_usize, (m, _)) in scored_moves.clone().into_iter().enumerate() {
            let undo = state.make_move(&m);
            if state.is_move_illegal() {
                state.undo_move(&m, undo);
                continue;
            }

            let child_hash = state.hash;
            let child_depth_left = depth_left.saturating_sub(1);
            let i_u32 = i_usize as u32;

            let (mut cpn, mut cdn) =
                if let Some((tt_pn, tt_dn, _, cached_depth)) = self.tt.probe(child_hash) {
                    if cached_depth >= child_depth_left || tt_pn == 0 {
                        (tt_pn, tt_dn)
                    } else {
                        let init_pn = 1 + i_u32 * i_u32 * 10;
                        (init_pn, 1)
                    }
                } else {
                    let init_pn = 1 + i_u32 * i_u32 * 10;
                    (init_pn, 1)
                };

            if cpn > 0 && cpn < parallel_tt::PN_INF && cdn < parallel_tt::PN_INF {
                let (new_cpn, new_cdn) = self.mid_node(
                    state,
                    parallel_tt::PN_INF,
                    parallel_tt::PN_INF,
                    ply + 1,
                    max_plies,
                );
                cpn = new_cpn;
                cdn = new_cdn;
            }

            state.undo_move(&m, undo);
            total_dn += cdn as u64;

            if cpn < best_pn {
                best_pn = cpn;
            }

            if cpn == 0 {
                let h_idx = (((((m.from.x as u32).wrapping_mul(31) ^ (m.from.y as u32))
                    .wrapping_mul(31)
                    ^ (m.to.x as u32))
                    .wrapping_mul(31)
                    ^ (m.to.y as u32)) as usize)
                    & 32767;
                self.history[h_idx].fetch_add(100, Ordering::Relaxed);

                let m_val = (m.from.x as u16 as u64)
                    | ((m.from.y as u16 as u64) << 16)
                    | ((m.to.x as u16 as u64) << 32)
                    | ((m.to.y as u16 as u64) << 48);
                if self.killers[ply_idx][0].load(Ordering::Relaxed) != m_val {
                    self.killers[ply_idx][1].store(
                        self.killers[ply_idx][0].load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    self.killers[ply_idx][0].store(m_val, Ordering::Relaxed);
                }

                self.tt.store(
                    hash,
                    0,
                    parallel_tt::PN_INF,
                    depth_left,
                    Some((
                        m.from.x as i16,
                        m.from.y as i16,
                        m.to.x as i16,
                        m.to.y as i16,
                    )),
                );
                return (0, parallel_tt::PN_INF);
            }

            if self.found_mate.load(Ordering::Relaxed) {
                return (0, parallel_tt::PN_INF);
            }
        }

        if total_dn == 0 && best_pn == parallel_tt::PN_INF {
            let res = (parallel_tt::PN_INF, 0);
            self.tt.store(hash, res.0, res.1, depth_left, None);
            return res;
        }

        let final_dn = total_dn.min(parallel_tt::PN_INF as u64) as u32;
        let res = (best_pn, final_dn);

        let best_move_coords = if let Some((_, _, m, _)) = self.tt.probe(hash) {
            m
        } else {
            None
        };
        self.tt
            .store(hash, res.0, res.1, depth_left, best_move_coords);
        res
    }

    fn terminal_score(&self, state: &mut GameState, ply: u32) -> i32 {
        if state.is_in_check() && state.turn == self.target_mated_side {
            if !state.has_legal_evasions() {
                return MATE_VALUE - ply as i32;
            }
        }
        -INFINITY + ply as i32
    }

    fn generate_helpmate_moves(&self, state: &GameState, depth: i32) -> SmallVec<[Move; 128]> {
        let mut moves = SmallVec::new();

        let tk = match self.target_mated_side {
            PlayerColor::White => state.white_king_pos.unwrap_or(Coordinate::new(0, 0)),
            PlayerColor::Black => state.black_king_pos.unwrap_or(Coordinate::new(0, 0)),
            _ => Coordinate::new(0, 0),
        };
        let ok = if state.turn == PlayerColor::White {
            state.black_king_pos.unwrap_or(Coordinate::new(0, 0))
        } else {
            state.white_king_pos.unwrap_or(Coordinate::new(0, 0))
        };

        // At depth 1, if mating side to move, must give check.
        let must_check = depth == 1 && state.turn != self.target_mated_side;

        // At depth 2, if defending side to move, must self-block or move king.
        let strict_neighborhood = depth == 2 && state.turn == self.target_mated_side;

        let ml = (depth + 1) / 2;
        let b = 2;
        let min_x = tk.x.min(ok.x) - (ml * 2 + b) as i64;
        let max_x = tk.x.max(ok.x) + (ml * 2 + b) as i64;
        let min_y = tk.y.min(ok.y) - (ml * 2 + b) as i64;
        let max_y = tk.y.max(ok.y) + (ml * 2 + b) as i64;

        let us_king = if state.turn == PlayerColor::White {
            state.white_king_pos
        } else {
            state.black_king_pos
        };
        let pinned = if let Some(kp) = us_king {
            state.compute_pins(&kp, state.turn)
        } else {
            rustc_hash::FxHashMap::default()
        };

        let ctx = hydrochess_wasm::moves::MoveGenContext {
            special_rights: &state.special_rights,
            en_passant: &state.en_passant,
            game_rules: &state.game_rules,
            indices: &state.spatial_indices,
            enemy_king_pos: Some(&ok),
            pinned: &pinned,
        };

        let is_white = state.turn == PlayerColor::White;
        let mut piece_buf = MoveList::new();

        for (px, py, piece) in state.board.iter_pieces_by_color(is_white) {
            let in_z = px >= min_x && px <= max_x && py >= min_y && py <= max_y;
            let pt = piece.piece_type();

            if !in_z {
                // Optimization: Sliders can influence from far away, but only if they are somewhat aligned
                if !hydrochess_wasm::attacks::is_slider(pt) {
                    continue;
                }

                let ax = (px >= min_x && px <= max_x) || px == tk.x || px == ok.x;
                let ay = (py >= min_y && py <= max_y) || py == tk.y || py == ok.y;
                let ad = (px - tk.x).abs() == (py - tk.y).abs()
                    || (px - ok.x).abs() == (py - ok.y).abs();

                if !ax && !ay && !ad {
                    continue;
                }
            }

            piece_buf.clear();
            hydrochess_wasm::moves::get_pseudo_legal_moves_for_piece_into(
                &state.board,
                &piece,
                &Coordinate::new(px, py),
                &ctx,
                &mut piece_buf,
            );

            if must_check {
                for m in &piece_buf {
                    // Fast check filter before full validation
                    if hydrochess_wasm::search::movegen::StagedMoveGen::move_gives_check_fast(
                        state, m,
                    ) {
                        moves.push(*m);
                    }
                }
            } else if strict_neighborhood {
                for m in &piece_buf {
                    // Strict Neighborhood Filter:
                    // 1. King moves are allowed
                    // 2. Non-King moves MUST end near the King (Chebyshev dist <= 3)
                    let is_k = m.piece.piece_type() == hydrochess_wasm::board::PieceType::King;
                    if is_k {
                        moves.push(*m);
                    } else {
                        let dist = (m.to.x - tk.x).abs().max((m.to.y - tk.y).abs());
                        if dist <= 3 {
                            moves.push(*m);
                        }
                    }
                }
            } else {
                moves.extend_from_slice(&piece_buf);
            }
        }
        moves
    }

    fn find_mating_move(&self, state: &mut GameState, ply: u32) -> Option<(i32, Move)> {
        // Only attacking side can deliver mate
        if state.turn == self.target_mated_side {
            return None;
        }

        // Depth 1 specialized search: Find ANY move that gives check and leads to mate.
        let tk = match self.target_mated_side {
            PlayerColor::White => state.white_king_pos.unwrap_or(Coordinate::new(0, 0)),
            PlayerColor::Black => state.black_king_pos.unwrap_or(Coordinate::new(0, 0)),
            _ => Coordinate::new(0, 0),
        };
        let ok = if state.turn == PlayerColor::White {
            state.black_king_pos.unwrap_or(Coordinate::new(0, 0))
        } else {
            state.white_king_pos.unwrap_or(Coordinate::new(0, 0))
        };

        // Windowing (generous for sliders)
        let min_x = tk.x.min(ok.x) - 10;
        let max_x = tk.x.max(ok.x) + 10;
        let min_y = tk.y.min(ok.y) - 10;
        let max_y = tk.y.max(ok.y) + 10;

        let is_white = state.turn == PlayerColor::White;

        // Collect pieces to avoid borrow conflicts
        let pieces: Vec<_> = state.board.iter_pieces_by_color(is_white).collect();
        let mut piece_buf = MoveList::new();

        let us_king = if state.turn == PlayerColor::White {
            state.white_king_pos
        } else {
            state.black_king_pos
        };
        let pinned = if let Some(kp) = us_king {
            state.compute_pins(&kp, state.turn)
        } else {
            rustc_hash::FxHashMap::default()
        };

        for (px, py, piece) in pieces {
            let in_z = px >= min_x && px <= max_x && py >= min_y && py <= max_y;
            if !in_z {
                if !hydrochess_wasm::attacks::is_slider(piece.piece_type()) {
                    continue;
                }
                let ax = px == tk.x || px == ok.x;
                let ay = py == tk.y || py == ok.y;
                let ad = (px - tk.x).abs() == (py - tk.y).abs()
                    || (px - ok.x).abs() == (py - ok.y).abs();
                if !ax && !ay && !ad {
                    continue;
                }
            }

            piece_buf.clear();

            {
                let ctx = hydrochess_wasm::moves::MoveGenContext {
                    special_rights: &state.special_rights,
                    en_passant: &state.en_passant,
                    game_rules: &state.game_rules,
                    indices: &state.spatial_indices,
                    enemy_king_pos: Some(&ok),
                    pinned: &pinned,
                };

                hydrochess_wasm::moves::get_pseudo_legal_moves_for_piece_into(
                    &state.board,
                    &piece,
                    &Coordinate::new(px, py),
                    &ctx,
                    &mut piece_buf,
                );
            }

            let len = piece_buf.len();
            for i in 0..len {
                // SAFETY: i is within bounds 0..len
                let m = unsafe { piece_buf.get_unchecked(i) };

                // FAST CHECK: Only process moves that give check
                if hydrochess_wasm::search::movegen::StagedMoveGen::move_gives_check_fast(state, m)
                {
                    let undo = state.make_move(m);

                    // Check for legal evasions
                    if state.is_move_illegal() {
                        state.undo_move(m, undo);
                        continue;
                    }

                    if state.is_in_check() && !state.has_legal_evasions() {
                        state.undo_move(m, undo);
                        return Some((MATE_VALUE - (ply + 1) as i32, *m));
                    }
                    state.undo_move(m, undo);
                }
            }
        }
        None
    }

    fn is_king_isolated(&self, state: &GameState, target_pos: Coordinate, max_plies: i32) -> bool {
        // Fast isolation check using SpatialIndices
        if max_plies >= 5 {
            return false;
        }

        let moves_available = (max_plies + 1) / 2;
        let threshold = (moves_available + 1) as i64;

        // Check pieces in rows [y-threshold, y+threshold]
        let min_y = target_pos.y - threshold;
        let max_y = target_pos.y + threshold;

        for y in min_y..=max_y {
            if let Some(row) = state.spatial_indices.rows.get(&y) {
                // We want pieces with x in [x-threshold, x+threshold]
                let min_x = target_pos.x - threshold;
                let max_x = target_pos.x + threshold;

                // Use binary search to find starting index
                let start_idx = row.coords.partition_point(|x| *x < min_x);

                for (x, packed) in row.iter().skip(start_idx) {
                    if x > max_x {
                        break;
                    }
                    // Keep skipping the king itself
                    let piece = hydrochess_wasm::board::Piece::from_packed(packed);
                    if piece.piece_type() == hydrochess_wasm::board::PieceType::King
                        && piece.color() == self.target_mated_side
                    {
                        continue;
                    }

                    // Found a piece within threshold box!
                    let dist = (x - target_pos.x).abs().max((y - target_pos.y).abs());

                    let effective_dist = match piece.piece_type() {
                        hydrochess_wasm::board::PieceType::Knight => (dist + 1) / 2,
                        hydrochess_wasm::board::PieceType::Pawn => dist,
                        _ => dist, // Sliders are powerful, count as normal distance (or 1)
                    };

                    if effective_dist <= threshold {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn extract_pv(&self, state: &mut GameState) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current_state = state.clone();

        for _ in 0..self.target_depth {
            let hash = current_state.hash;
            let res = self.tt.probe(hash);

            if let Some((pn, _dn, move_coords, _depth)) = res {
                // If this node is proven (PN=0), follow the move that proved it
                if pn != 0 {
                    break;
                }

                if let Some((fx, fy, tx, ty)) = move_coords {
                    let mut moves = MoveList::new();
                    current_state.get_legal_moves_into(&mut moves);

                    let mut found_move = None;
                    for &m in moves.iter() {
                        if m.from.x as i16 == fx
                            && m.from.y as i16 == fy
                            && m.to.x as i16 == tx
                            && m.to.y as i16 == ty
                        {
                            let undo = current_state.make_move(&m);
                            if !current_state.is_move_illegal() {
                                found_move = Some(m);
                                break;
                            }
                            current_state.undo_move(&m, undo);
                        }
                    }

                    if let Some(m) = found_move {
                        pv.push(m);
                        if self.terminal_score(&mut current_state, 0) >= MATE_VALUE - 1000 {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        pv
    }
}

// ============================================================================
// UTILITIES & MAIN
// ============================================================================

struct Args {
    icn: String,
    mate_in: Option<u32>,
    mated_side: Option<PlayerColor>,
}

fn print_help() {
    println!("=== Infinite Chess Helpmate Solver ===");
    println!("Usage: helpmate_solver --icn \"<ICN>\" --mate-in <N> --mated-side <w|b>");
    println!();
    println!("Required Arguments:");
    println!("  --icn \"<string>\"    The ICN string for the position.");
    println!("  --mate-in <N>       Target plies to find a helpmate in.");
    println!("  --mated-side <w|b>  The side to be mated (white/w or black/b).");
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        print_help();
        std::process::exit(0);
    }
    let mut icn = String::new();
    let mut mate_in = None;
    let mut mated_side = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--icn" if i + 1 < args.len() => {
                icn = args[i + 1].clone();
                i += 2;
            }
            "--mate-in" if i + 1 < args.len() => {
                mate_in = args[i + 1].parse().ok();
                i += 2;
            }
            "--mated-side" if i + 1 < args.len() => {
                let s = args[i + 1].to_lowercase();
                if s.starts_with('w') {
                    mated_side = Some(PlayerColor::White);
                } else if s.starts_with('b') {
                    mated_side = Some(PlayerColor::Black);
                }
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }
    Args {
        icn,
        mate_in,
        mated_side,
    }
}

fn main() {
    #[cfg(debug_assertions)]
    {
        println!("⚠️  WARNING: Running in DEBUG mode. Performance will be significantly reduced.");
        println!(
            "   For production solving, use: cargo run --bin helpmate_solver --release --features parallel_solver -- <ARGS>"
        );
        println!();
    }

    let args = parse_args();
    if args.icn.is_empty() || args.mate_in.is_none() || args.mated_side.is_none() {
        print_help();
        std::process::exit(1);
    }
    let mate_in = args.mate_in.unwrap();
    let mated_side = args.mated_side.unwrap();

    let mut game = GameState::new();
    game.setup_position_from_icn(&args.icn);

    // Find the bounding box of all pieces and add a small buffer.
    // Intersect this with the existing world border to get the tightest possible bounds.
    let mut min_x = i64::MAX;
    let mut max_x = i64::MIN;
    let mut min_y = i64::MAX;
    let mut max_y = i64::MIN;
    let mut has_pieces = false;

    for (x, y, _piece) in game.board.iter() {
        has_pieces = true;
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    if has_pieces {
        let buffer = 2;
        min_x = min_x.saturating_sub(buffer);
        max_x = max_x.saturating_add(buffer);
        min_y = min_y.saturating_sub(buffer);
        max_y = max_y.saturating_add(buffer);

        let (cur_min_x, cur_max_x, cur_min_y, cur_max_y) =
            hydrochess_wasm::moves::get_coord_bounds();

        let final_min_x = min_x.max(cur_min_x);
        let final_max_x = max_x.min(cur_max_x);
        let final_min_y = min_y.max(cur_min_y);
        let final_max_y = max_y.min(cur_max_y);

        hydrochess_wasm::moves::set_world_bounds(
            final_min_x,
            final_max_x,
            final_min_y,
            final_max_y,
        );
    }

    println!(
        "\n=== HELPMATE SOLVER ===\nBoard: {} pieces\nTurn: {:?}\nTarget: Helpmate in {} plies (Mate {:?})\nThreads: {}\n",
        game.board.iter().count(),
        game.turn,
        mate_in,
        mated_side,
        rayon::current_num_threads()
    );

    let mut solver = HelpmateSolver::new(mate_in, mated_side);
    let start = Instant::now();
    let result = solver.solve(&mut game.clone());
    let elapsed = start.elapsed();

    if let Some(score) = result {
        if score >= MATE_VALUE - 1000 {
            let pv = solver.extract_pv(&mut game);
            println!("\n=== RESULT ===\n✓ FOUND HELPMATE in {} plies!", pv.len());
            let pv_str: Vec<_> = pv
                .iter()
                .map(|m| format!("({},{})->({},{})", m.from.x, m.from.y, m.to.x, m.to.y))
                .collect();
            println!("  PV: {}", pv_str.join(" "));
            println!(
                "\nTime: {:.2?}\nNodes: {}\nNPS: {:.0}",
                elapsed,
                solver.nodes.load(Ordering::Relaxed),
                solver.nodes.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64().max(0.001)
            );
            return;
        }
    }

    println!(
        "\n=== RESULT ===\n✗ No helpmate found in {} plies\n\nTime: {:.2?}\nNodes: {}\nNPS: {:.0}",
        mate_in,
        elapsed,
        solver.nodes.load(Ordering::Relaxed),
        solver.nodes.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64().max(0.001)
    );
}
