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
    use super::*;

    const ENTRIES_PER_BUCKET: usize = 8;

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct TTEntry {
        pub key: u64,
        pub score: i32,
        pub from_x: i16,
        pub from_y: i16,
        pub to_x: i16,
        pub to_y: i16,
        pub depth: u8,
        pub generation: u8,
    }

    impl TTEntry {
        pub const fn empty() -> Self {
            TTEntry {
                key: 0,
                score: 0,
                from_x: 0,
                from_y: 0,
                to_x: 0,
                to_y: 0,
                depth: 0,
                generation: 0,
            }
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.key == 0
        }
    }

    pub struct TTBucket {
        pub entries: [TTEntry; ENTRIES_PER_BUCKET],
        pub lock: AtomicBool,
    }

    impl TTBucket {
        pub fn lock(&self) {
            while self
                .lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::hint::spin_loop();
            }
        }
        pub fn unlock(&self) {
            self.lock.store(false, Ordering::Release);
        }
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
                    entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
                    lock: AtomicBool::new(false),
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

        pub fn probe(&self, hash: u64, depth: i32) -> Option<(i32, Option<(i16, i16, i16, i16)>)> {
            let bucket = &self.buckets[self.bucket_idx(hash)];
            bucket.lock();
            let mut res = None;
            for e in &bucket.entries {
                if e.key == hash && !e.is_empty() {
                    if e.depth as i32 >= depth {
                        res = Some((e.score, Some((e.from_x, e.from_y, e.to_x, e.to_y))));
                    } else {
                        res = Some((i32::MIN, Some((e.from_x, e.from_y, e.to_x, e.to_y))));
                    }
                    break;
                }
            }
            bucket.unlock();
            res
        }

        pub fn store(&self, hash: u64, depth: i32, score: i32, m: Option<&Move>, generation: u8) {
            let bucket = &self.buckets[self.bucket_idx(hash)];
            bucket.lock();

            let mut replace_idx = 0;
            let mut found_slot = false;
            for (i, e) in bucket.entries.iter().enumerate() {
                if e.key == hash && !e.is_empty() {
                    // Match found! Only overwrite if NEW is 'better'
                    let old_is_mate = e.score >= MATE_VALUE - 1000;
                    let new_is_mate = score >= MATE_VALUE - 1000;

                    if old_is_mate && !new_is_mate {
                        bucket.unlock();
                        return; // Protect mate score
                    }

                    if (depth as u8) < e.depth && e.generation == generation && !new_is_mate {
                        bucket.unlock();
                        return; // Don't downgrade depth in same generation
                    }

                    replace_idx = i;
                    found_slot = true;
                    break;
                }
                if e.is_empty() {
                    replace_idx = i;
                    found_slot = true;
                    break;
                }
            }

            if !found_slot {
                let mut worst_v = i32::MAX;
                for (i, e) in bucket.entries.iter().enumerate() {
                    let mut v =
                        (e.depth as i32) - (generation.wrapping_sub(e.generation) as i32 * 10);
                    // Strictly protect mate scores
                    if e.score >= MATE_VALUE - 1000 {
                        v += 10000;
                    }
                    if v < worst_v {
                        worst_v = v;
                        replace_idx = i;
                    }
                }
            }

            let entry = unsafe {
                let ptr = bucket.entries.as_ptr() as *mut TTEntry;
                &mut *ptr.add(replace_idx)
            };

            entry.key = hash;
            entry.score = score;
            entry.depth = depth as u8;
            entry.generation = generation;
            if let Some(m) = m {
                entry.from_x = m.from.x as i16;
                entry.from_y = m.from.y as i16;
                entry.to_x = m.to.x as i16;
                entry.to_y = m.to.y as i16;
            } else {
                entry.from_x = 0;
                entry.from_y = 0;
                entry.to_x = 0;
                entry.to_y = 0;
            }

            bucket.unlock();
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
    best_score: AtomicI32,
    target_depth: u32,
    target_mated_side: PlayerColor,
    generation: u8,
    history: Vec<AtomicI32>,
    killers: Vec<[AtomicU64; 2]>,
    iteration_depth: AtomicI32,
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
            best_score: AtomicI32::new(-INFINITY),
            generation: 0,
            history,
            killers,
            iteration_depth: AtomicI32::new(0),
        }
    }

    fn solve(&mut self, state: &mut GameState) -> Option<i32> {
        self.found_mate.store(false, Ordering::SeqCst);
        self.best_score.store(-INFINITY, Ordering::SeqCst);

        // Force Hash Consistency at Root
        state.recompute_hash();

        let target = self.target_depth as i32;
        let mut best = -INFINITY;
        let start = Instant::now();

        // Iterative deepening (for move ordering and PV consistency)
        for depth in 1..=target {
            self.iteration_depth.store(depth as i32, Ordering::Relaxed);
            self.generation = self.generation.wrapping_add(1);
            let score = self.parallel_root_search(state, depth);
            best = score;

            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            let nodes = self.nodes.load(Ordering::Relaxed);
            let nps = nodes as f64 / elapsed;

            println!(
                "info depth {} {} nodes {} time {:.2}s nps {:.0}",
                depth,
                format_score(score),
                nodes,
                elapsed,
                nps
            );

            if score >= MATE_VALUE - target {
                self.found_mate.store(true, Ordering::SeqCst);
                break;
            }
        }

        if best > -INFINITY { Some(best) } else { None }
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

    fn parallel_root_search(&self, state: &mut GameState, depth: i32) -> i32 {
        // Use custom generator
        let moves = self.generate_helpmate_moves(state, depth);

        if moves.is_empty() {
            return self.terminal_score(state, 0);
        }

        let work_items: Vec<_> = moves
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
            // In root, if no moves that are legal, check terminal state
            return self.terminal_score(state, 0);
        }

        let results: Vec<(i32, Move)> = work_items
            .into_par_iter()
            .map(|(mut local_state, m)| {
                if self.found_mate.load(Ordering::Relaxed) {
                    return (-INFINITY, m);
                }
                let score = self.search_sequential(&mut local_state, depth - 1, 1);
                (score, m)
            })
            .collect();

        let mut best_score = -INFINITY;
        let mut best_move = None;

        for (score, m) in results {
            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }
        }

        self.tt.store(
            state.hash
                ^ (self.iteration_depth.load(Ordering::Relaxed) as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15),
            depth,
            best_score,
            best_move.as_ref(),
            self.generation,
        );

        best_score
    }

    #[inline(always)]
    fn score_to_tt(score: i32, ply: u32) -> i32 {
        if score >= MATE_VALUE - 1000 {
            score + ply as i32
        } else if score <= -MATE_VALUE + 1000 {
            score - ply as i32
        } else {
            score
        }
    }

    #[inline(always)]
    fn score_from_tt(val: i32, ply: u32) -> i32 {
        if val >= MATE_VALUE - 1000 {
            val - ply as i32
        } else if val <= -MATE_VALUE + 1000 {
            val + ply as i32
        } else {
            val
        }
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

    fn search_sequential(&self, state: &mut GameState, depth: i32, ply: u32) -> i32 {
        self.nodes.fetch_add(1, Ordering::Relaxed);

        if self.found_mate.load(Ordering::Relaxed) {
            return -INFINITY;
        }

        let hash = state.hash
            ^ (self.iteration_depth.load(Ordering::Relaxed) as u64)
                .wrapping_mul(0x9E3779B97F4A7C15);

        if let Some((raw_score, _)) = self.tt.probe(hash, depth) {
            if raw_score != i32::MIN {
                let tt_score = Self::score_from_tt(raw_score, ply);
                if tt_score < MATE_VALUE - 1000 {
                    return tt_score;
                }
            }
        }

        // Get TT move for ordering (probe at depth 0 to always get move hint)
        let tt_move = self.tt.probe(hash, 0).and_then(|(_, m)| m);

        if depth <= 0 {
            return self.terminal_score(state, ply);
        }

        let target_king_pos = match self.target_mated_side {
            PlayerColor::White => state.white_king_pos.unwrap_or(Coordinate::new(0, 0)),
            PlayerColor::Black => state.black_king_pos.unwrap_or(Coordinate::new(0, 0)),
            _ => Coordinate::new(0, 0),
        };

        // Isolation Pruning
        // If the King is too far from ANY piece (friend or foe), mate is impossible.
        if self.is_king_isolated(state, target_king_pos, depth) {
            return -INFINITY + ply as i32 + 1000;
        }

        // Leaf Node Optimization
        if depth == 1 && state.turn != self.target_mated_side {
            if let Some((score, best_move)) = self.find_mating_move(state, ply) {
                // Found mate! Store the mating move in TT for PV extraction.
                self.tt.store(
                    hash,
                    depth,
                    Self::score_to_tt(score, ply),
                    Some(&best_move),
                    self.generation,
                );
                return score;
            }
            // If we checked all moves and found no mate involved, we failed.
            // We return a "not mated" score.
            let s = -INFINITY + ply as i32;
            self.tt.store(
                hash,
                depth,
                Self::score_to_tt(s, ply),
                None,
                self.generation,
            );
            return s;
        }

        let moves = self.generate_helpmate_moves(state, depth);
        if moves.is_empty() {
            return self.terminal_score(state, ply);
        }

        let opp_king = if state.turn == PlayerColor::White {
            state.black_king_pos.unwrap_or(Coordinate::new(0, 0))
        } else {
            state.white_king_pos.unwrap_or(Coordinate::new(0, 0))
        };

        // Score and Order
        let ply_idx = (ply as usize).min(63);
        let k1_val = self.killers[ply_idx][0].load(Ordering::Relaxed);
        let k2_val = self.killers[ply_idx][1].load(Ordering::Relaxed);

        let mut scored_moves: SmallVec<[(Move, i32); 64]> = moves
            .iter()
            .map(|m| {
                let mut score = 0;
                let dist = (m.to.x - opp_king.x).abs().max((m.to.y - opp_king.y).abs());
                let target_dist = (m.to.x - target_king_pos.x)
                    .abs()
                    .max((m.to.y - target_king_pos.y).abs());

                // Atomic Killers check
                let m_val = (m.from.x as u16 as u64)
                    | ((m.from.y as u16 as u64) << 16)
                    | ((m.to.x as u16 as u64) << 32)
                    | ((m.to.y as u16 as u64) << 48);
                if m_val == k1_val {
                    score += 900_000;
                } else if m_val == k2_val {
                    score += 800_000;
                }

                if state.turn == self.target_mated_side {
                    if target_dist <= 2 {
                        score += 50_000;
                    }
                } else {
                    if hydrochess_wasm::search::movegen::StagedMoveGen::move_gives_check_fast(
                        state, m,
                    ) {
                        score += 1_000_000;
                    }
                    if target_dist <= 2 {
                        score += 30_000;
                    }
                    if dist <= 2 {
                        score += 20_000;
                    }
                }

                if let Some((fx, fy, tx, ty)) = tt_move {
                    if m.from.x as i16 == fx
                        && m.from.y as i16 == fy
                        && m.to.x as i16 == tx
                        && m.to.y as i16 == ty
                    {
                        score += 2_000_000;
                    }
                }

                let h_idx = (((((m.from.x as u32).wrapping_mul(31) ^ (m.from.y as u32))
                    .wrapping_mul(31)
                    ^ (m.to.x as u32))
                    .wrapping_mul(31)
                    ^ (m.to.y as u32)) as usize)
                    & 32767;
                let h_score = self.history[h_idx].load(Ordering::Relaxed);
                score += h_score.min(200_000);

                (*m, score)
            })
            .collect();

        scored_moves.sort_unstable_by_key(|(_, s)| -s);

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut searched_count = 0;
        let mut real_legal = false;

        for (m, _score) in scored_moves {
            // Depth 2: Mated side moves, then mating side must check. Prioritize moves near target king.
            if depth == 2 && state.turn == self.target_mated_side {
                let king_dist = (m.to.x - target_king_pos.x)
                    .abs()
                    .max((m.to.y - target_king_pos.y).abs());

                // If this is a king move that escapes too far, skip
                if m.piece.piece_type() == hydrochess_wasm::board::PieceType::King && king_dist > 3
                {
                    continue;
                }
            }

            let undo = state.make_move(&m);
            if state.is_move_illegal() {
                state.undo_move(&m, undo);
                continue;
            }
            real_legal = true;
            searched_count += 1;

            let score = self.search_sequential(state, depth - 1, ply + 1);
            state.undo_move(&m, undo);

            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }

            if score >= MATE_VALUE - ply as i32 {
                self.found_mate.store(true, Ordering::SeqCst);
                break;
            }
        }

        if searched_count == 0 {
            if real_legal {
                // We had legal moves but pruned them all. This is a search failure, not a mate.
                let s = -INFINITY + ply as i32;
                self.tt.store(
                    hash,
                    depth,
                    Self::score_to_tt(s, ply),
                    None,
                    self.generation,
                );
                return s;
            } else {
                let s = self.terminal_score(state, ply);
                self.tt.store(
                    hash,
                    depth,
                    Self::score_to_tt(s, ply),
                    None,
                    self.generation,
                );
                return s;
            }
        }

        self.tt.store(
            hash,
            depth,
            Self::score_to_tt(best_score, ply),
            best_move.as_ref(),
            self.generation,
        );

        if let Some(ref m) = best_move {
            // Update history (atomic)
            let h_idx = (((((m.from.x as u32).wrapping_mul(31) ^ (m.from.y as u32))
                .wrapping_mul(31)
                ^ (m.to.x as u32))
                .wrapping_mul(31)
                ^ (m.to.y as u32)) as usize)
                & 32767;
            self.history[h_idx].fetch_add(depth * depth, Ordering::Relaxed);

            // Update killers (atomic)
            if ply_idx < 64 {
                let m_val = (m.from.x as u16 as u64)
                    | ((m.from.y as u16 as u64) << 16)
                    | ((m.to.x as u16 as u64) << 32)
                    | ((m.to.y as u16 as u64) << 48);
                let k = &self.killers[ply_idx];
                if k[0].load(Ordering::Relaxed) != m_val {
                    k[1].store(k[0].load(Ordering::Relaxed), Ordering::Relaxed);
                    k[0].store(m_val, Ordering::Relaxed);
                }
            }
        }
        best_score
    }

    fn extract_pv(&self, state: &mut GameState, _target_score: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current_state = state.clone();

        for ply in 0..self.target_depth {
            let hash = current_state.hash
                ^ (self.iteration_depth.load(Ordering::Relaxed) as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15);
            let res = self.tt.probe(hash, 0);

            if let Some((raw_score, move_coords)) = res {
                // If the score is no longer a mate or is just i32::MIN, we lost the thread
                if raw_score == i32::MIN {
                    break;
                }

                let actual_score = Self::score_from_tt(raw_score, ply);
                if actual_score < MATE_VALUE - 1000 {
                    // This entry is not a mate, don't follow it for PV
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
                        // If this move just delivered mate, we are done
                        if self.terminal_score(&mut current_state, ply + 1) >= MATE_VALUE - 1000 {
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

fn format_score(score: i32) -> String {
    if score >= MATE_VALUE - 1000 {
        format!("mate {}", (MATE_VALUE - score + 1) / 2)
    } else if score <= -INFINITY + 1000 {
        "score fail".to_string()
    } else {
        format!("cp {}", score)
    }
}

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
            let pv = solver.extract_pv(&mut game, score);
            println!(
                "\n=== RESULT ===\n✓ FOUND HELPMATE in {} plies!",
                MATE_VALUE - score
            );
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
