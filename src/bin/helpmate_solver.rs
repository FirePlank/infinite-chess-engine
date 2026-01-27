//! Helpmate Solver for Infinite Chess
//!
//! A cooperative chess problem solver where both sides work together to achieve checkmate.
//! Uses parallel exhaustive search with a thread-safe Transposition Table.

use hydrochess_wasm::{
    board::{PieceType, PlayerColor},
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

    const ENTRIES_PER_BUCKET: usize = 4;

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
                if e.key == hash || e.is_empty() {
                    replace_idx = i;
                    found_slot = true;
                    break;
                }
            }

            if !found_slot {
                let mut worst_v = i32::MAX;
                for (i, e) in bucket.entries.iter().enumerate() {
                    let v = (e.depth as i32) - (generation.wrapping_sub(e.generation) as i32 * 20);
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

use rustc_hash::FxHashMap;
use std::sync::Mutex;

// ============================================================================
// PV TABLE - Stores best moves for PV extraction (score+depth priority)
// ============================================================================

struct PvTable {
    table: Mutex<FxHashMap<u64, (i32, i32, i16, i16, i16, i16)>>, // hash -> (depth, score, fx, fy, tx, ty)
}

impl PvTable {
    fn new() -> Self {
        PvTable {
            table: Mutex::new(FxHashMap::default()),
        }
    }

    fn store(&self, hash: u64, depth: i32, score: i32, m: &Move) {
        let mut t = self.table.lock().unwrap();
        let should_replace = match t.get(&hash) {
            Some(&(d, s, _, _, _, _)) => depth > d || (depth == d && score > s),
            None => true,
        };
        if should_replace {
            t.insert(
                hash,
                (
                    depth,
                    score,
                    m.from.x as i16,
                    m.from.y as i16,
                    m.to.x as i16,
                    m.to.y as i16,
                ),
            );
        }
    }

    fn probe(&self, hash: u64) -> Option<(i16, i16, i16, i16)> {
        let t = self.table.lock().unwrap();
        t.get(&hash).map(|&(_, _, fx, fy, tx, ty)| (fx, fy, tx, ty))
    }
}

// ============================================================================
// SOLVER
// ============================================================================

struct HelpmateSolver {
    tt: parallel_tt::TranspositionTable,
    pv_table: PvTable,
    nodes: AtomicU64,
    found_mate: AtomicBool,
    best_score: AtomicI32,
    target_depth: u32,
    target_mated_side: PlayerColor,
    generation: u8,
}

impl HelpmateSolver {
    fn new(target_depth: u32, target_mated_side: PlayerColor) -> Self {
        HelpmateSolver {
            tt: parallel_tt::TranspositionTable::new(512),
            pv_table: PvTable::new(),
            nodes: AtomicU64::new(0),
            found_mate: AtomicBool::new(false),
            best_score: AtomicI32::new(-INFINITY),
            target_depth,
            target_mated_side,
            generation: 0,
        }
    }

    fn solve(&mut self, state: &mut GameState) -> Option<i32> {
        self.found_mate.store(false, Ordering::SeqCst);
        self.best_score.store(-INFINITY, Ordering::SeqCst);

        let target = self.target_depth as i32;
        let mut best = -INFINITY;
        let start = Instant::now();

        // Iterative deepening (for move ordering and PV consistency)
        for d in 1..=target {
            self.generation = self.generation.wrapping_add(1);
            let score = self.parallel_root_search(state, d);
            best = score;

            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            let nodes = self.nodes.load(Ordering::Relaxed);
            let nps = nodes as f64 / elapsed;

            println!(
                "info depth {} score {} nodes {} time {:.2}s nps {:.0}",
                d,
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

    fn parallel_root_search(&self, state: &mut GameState, depth: i32) -> i32 {
        let mut moves = MoveList::new();
        state.get_legal_moves_into(&mut moves);

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
            return self.terminal_score(state, 0);
        }

        let results: Vec<(i32, Move)> = work_items
            .into_par_iter()
            .map(|(mut local_state, m)| {
                if self.found_mate.load(Ordering::Relaxed) {
                    return (-INFINITY, m);
                }
                let mut history = FxHashMap::default();
                let mut killers = [[None; 2]; 64];
                let score = self.search_sequential(
                    &mut local_state,
                    depth - 1,
                    1,
                    &mut history,
                    &mut killers,
                );
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

        if let Some(m) = best_move {
            self.tt
                .store(state.hash, depth, best_score, Some(&m), self.generation);
            self.pv_table.store(state.hash, depth, best_score, &m);
        } else {
            self.tt
                .store(state.hash, depth, best_score, None, self.generation);
        }

        best_score
    }

    fn search_sequential(
        &self,
        state: &mut GameState,
        depth: i32,
        ply: u32,
        history: &mut FxHashMap<(i16, i16, i16, i16), i32>,
        killers: &mut [[Option<Move>; 2]; 64],
    ) -> i32 {
        self.nodes.fetch_add(1, Ordering::Relaxed);

        if self.found_mate.load(Ordering::Relaxed) {
            return -INFINITY;
        }

        let hash = state.hash;
        let tt_move = self.tt.probe(hash, 0).and_then(|(_, m)| m);

        let mut moves = MoveList::new();
        state.get_legal_moves_into(&mut moves);

        if moves.is_empty() {
            let s = self.terminal_score(state, ply);
            self.tt.store(hash, depth, s, None, self.generation);
            return s;
        }

        if depth <= 0 {
            // Horizon reached without mate - failure for helpmate
            let s = -INFINITY + ply as i32;
            self.tt.store(hash, 0, s, None, self.generation);
            return s;
        }

        // Simple move ordering
        let mut scored_moves: SmallVec<[(Move, i32); 128]> = moves
            .iter()
            .map(|m| {
                let mut s = 0i32;
                if let Some((fx, fy, tx, ty)) = tt_move {
                    if m.from.x as i16 == fx
                        && m.from.y as i16 == fy
                        && m.to.x as i16 == tx
                        && m.to.y as i16 == ty
                    {
                        s += 1000000;
                    }
                }
                // History score
                if let Some(&h) = history.get(&(
                    m.from.x as i16,
                    m.from.y as i16,
                    m.to.x as i16,
                    m.to.y as i16,
                )) {
                    s += h.min(1_000_000);
                }

                // Killer Move score
                if ply < 64 {
                    if let Some(km) = killers[ply as usize][0] {
                        if km.from == m.from && km.to == m.to {
                            s += 2_000_000;
                        }
                    } else if let Some(km) = killers[ply as usize][1] {
                        if km.from == m.from && km.to == m.to {
                            s += 1_900_000;
                        }
                    }
                }

                // Check bonus & Capture bonus
                if hydrochess_wasm::search::movegen::StagedMoveGen::move_gives_check_fast(state, m)
                {
                    s += 50_000;
                }

                // Capture bonus (simple lookahead)
                if let Some(p) = state.board.get_piece(m.to.x, m.to.y) {
                    s += 10_000 + piece_value(p.piece_type()) * 10;
                }

                (*m, s)
            })
            .collect();
        scored_moves.sort_unstable_by_key(|(_, s)| -s);

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut searched_count = 0;
        let mut real_legal_move_exists = false;

        for (m, _) in scored_moves {
            // At depth 1 (last move), moves MUST check the opponent to allow mate.
            if depth == 1
                && !hydrochess_wasm::search::movegen::StagedMoveGen::move_gives_check_fast(
                    state, &m,
                )
            {
                continue;
            }

            let undo = state.make_move(&m);
            if state.is_move_illegal() {
                state.undo_move(&m, undo);
                continue;
            }
            real_legal_move_exists = true;

            searched_count += 1;

            let score = self.search_sequential(state, depth - 1, ply + 1, history, killers);
            state.undo_move(&m, undo);

            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }

            if best_score >= MATE_VALUE - ply as i32 - 1 {
                break;
            }
        }

        if searched_count == 0 {
            if real_legal_move_exists {
                // We had legal moves but pruned them all. This is a search failure, not a mate.
                let s = -INFINITY + ply as i32;
                self.tt.store(hash, depth, s, None, self.generation);
                return s;
            } else {
                let s = self.terminal_score(state, ply);
                self.tt.store(hash, depth, s, None, self.generation);
                return s;
            }
        }

        self.tt
            .store(hash, depth, best_score, best_move.as_ref(), self.generation);
        if let Some(ref m) = best_move {
            self.pv_table.store(hash, depth, best_score, m);
            // Update history
            *history
                .entry((
                    m.from.x as i16,
                    m.from.y as i16,
                    m.to.x as i16,
                    m.to.y as i16,
                ))
                .or_insert(0) += depth as i32 * depth as i32;

            // Update killers
            if ply < 64 {
                if killers[ply as usize][0].map_or(true, |k| k.from != m.from || k.to != m.to) {
                    killers[ply as usize][1] = killers[ply as usize][0];
                    killers[ply as usize][0] = Some(*m);
                }
            }
        }
        best_score
    }

    fn terminal_score(&self, state: &mut GameState, ply: u32) -> i32 {
        if state.is_in_check() {
            // Valid mate only if the correct side is mated
            if state.turn == self.target_mated_side {
                // Verify it's actually checkmate
                let mut moves = hydrochess_wasm::moves::MoveList::new();
                state.get_legal_moves_into(&mut moves);

                if moves.is_empty() {
                    MATE_VALUE - ply as i32
                } else {
                    // Check but not mate
                    -INFINITY + ply as i32
                }
            } else {
                // Wrong side mated - treat as failure
                -INFINITY + ply as i32
            }
        } else {
            -INFINITY + ply as i32
        }
    }

    fn extract_pv(&self, state: &mut GameState, _target_score: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current_state = state.clone();

        for _ in 0..self.target_depth {
            // Get move from PV table
            if let Some((fx, fy, tx, ty)) = self.pv_table.probe(current_state.hash) {
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

fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 1,
        PieceType::Knight => 3,
        PieceType::Bishop => 4,
        PieceType::Hawk => 4,
        PieceType::Rook => 5,
        PieceType::Archbishop => 7,
        PieceType::Chancellor => 8,
        PieceType::Queen => 9,
        PieceType::RoyalQueen => 9,
        PieceType::Amazon => 9,
        PieceType::King => 100,
        _ => 0,
    }
}

fn format_score(score: i32) -> String {
    if score >= MATE_VALUE - 1000 {
        format!("mate {} plies", MATE_VALUE - score)
    } else if score <= -INFINITY + 1000 {
        "fail".to_string()
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
    let args = parse_args();
    if args.icn.is_empty() || args.mate_in.is_none() || args.mated_side.is_none() {
        print_help();
        std::process::exit(1);
    }
    let mate_in = args.mate_in.unwrap();
    let mated_side = args.mated_side.unwrap();

    let mut game = GameState::new();
    game.setup_position_from_icn(&args.icn);

    println!();
    println!("=== HELPMATE SOLVER ===");
    println!("Board: {} pieces", game.board.iter().count());
    println!("Turn: {:?}", game.turn);
    println!(
        "Target: Helpmate in {} plies (Mate {:?})",
        mate_in, mated_side
    );
    println!("Threads: {}", rayon::current_num_threads());
    println!();

    let mut solver = HelpmateSolver::new(mate_in, mated_side);
    let start = Instant::now();
    let result = solver.solve(&mut game);
    let elapsed = start.elapsed();

    println!();
    println!("=== RESULT ===");
    match result {
        Some(score) if score >= MATE_VALUE - 1000 => {
            println!("✓ FOUND HELPMATE in {} plies!", MATE_VALUE - score);
            let pv = solver.extract_pv(&mut game, score);
            let pv_str: Vec<_> = pv
                .iter()
                .map(|m| format!("({},{})->({},{})", m.from.x, m.from.y, m.to.x, m.to.y))
                .collect();
            println!("  PV: {}", pv_str.join(" "));
        }
        _ => println!("✗ No helpmate found in {} plies", mate_in),
    }

    println!();
    println!("Time: {:.2?}", elapsed);
    println!("Nodes: {}", solver.nodes.load(Ordering::Relaxed));
    let nps = solver.nodes.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64().max(0.001);
    println!("NPS: {:.0}", nps);
}
