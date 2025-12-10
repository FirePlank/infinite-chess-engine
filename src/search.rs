use crate::board::PieceType;
use crate::evaluation::evaluate;
use crate::game::GameState;
use crate::moves::{get_quiescence_captures, Move};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

// For web WASM (browser), use js_sys::Date for timing
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use js_sys::Date;
// For native builds and WASI, use std::time::Instant
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use std::time::Instant;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn now_ms() -> f64 {
    // Simple wall-clock timer for wasm; keeps the hot path small and avoids
    // repeated window()/performance() lookups.
    Date::now()
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

pub const MAX_PLY: usize = 64;
pub const INFINITY: i32 = 1_000_000;
pub const MATE_VALUE: i32 = 900_000;
pub const MATE_SCORE: i32 = 800_000;
pub const THINK_TIME_MS: u128 = 3000; // 3 seconds per move (default, may be overridden by caller)

// ============================================================================
// Tunable search parameters - accessed via param accessor functions
// ============================================================================
pub mod params;
use params::{
    aspiration_fail_mult, aspiration_window, futility_margin, history_bonus_base,
    history_bonus_cap, history_bonus_sub, hlp_history_leaf, hlp_history_reduce, hlp_max_depth,
    hlp_min_moves, lmp_threshold, lmr_divisor, lmr_min_depth, lmr_min_moves, max_history,
    nmp_min_depth, nmp_reduction, repetition_penalty, rfp_margin_per_depth, rfp_max_depth,
};

mod tt;
pub use tt::{TTEntry, TTFlag, TranspositionTable};

mod ordering;
use ordering::{
    hash_coord_32, hash_move_dest, hash_move_from, sort_captures, sort_moves, sort_moves_root,
};

mod see;
pub(crate) use see::static_exchange_eval_impl as static_exchange_eval;

pub mod zobrist;
pub use zobrist::{en_passant_key, piece_key, special_right_key, SIDE_KEY};

mod noisy;
pub use noisy::get_best_move_with_noise;

/// Timer abstraction to handle platform differences
#[derive(Clone)]
pub struct Timer {
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    start: f64,
    #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            Self { start: now_ms() }
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            Self {
                start: Instant::now(),
            }
        }
    }

    pub fn reset(&mut self) {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            self.start = now_ms();
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            self.start = Instant::now();
        }
    }

    pub fn elapsed_ms(&self) -> u128 {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            (now_ms() - self.start) as u128
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            self.start.elapsed().as_millis()
        }
    }
}

/// Lightweight statistics about the transposition table after a search.
#[derive(Clone, Debug)]
pub struct SearchStats {
    pub tt_capacity: usize,
    pub tt_used: usize,
    pub tt_fill_permille: u32,
}

/// A single PV line with its score and depth.
#[derive(Clone, Debug)]
pub struct PVLine {
    pub mv: Move,
    pub score: i32,
    pub depth: usize,
    pub pv: Vec<Move>,
}

/// Result of a MultiPV search.
#[derive(Clone, Debug)]
pub struct MultiPVResult {
    pub lines: Vec<PVLine>,
    pub stats: SearchStats,
}

thread_local! {
    static GLOBAL_SEARCHER: RefCell<Option<Searcher>> = RefCell::new(None);
}

#[allow(dead_code)]
fn with_global_searcher<F, R>(time_limit_ms: u128, silent: bool, f: F) -> R
where
    F: FnOnce(&mut Searcher) -> R,
{
    // Disable persistent searcher: create a fresh one per call so TT does not persist across searches.
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    searcher.stopped = false;
    searcher.timer.reset();

    f(&mut searcher)
}

fn build_search_stats(searcher: &Searcher) -> SearchStats {
    SearchStats {
        tt_capacity: searcher.tt.capacity(),
        tt_used: searcher.tt.used_entries(),
        tt_fill_permille: searcher.tt.fill_permille(),
    }
}

/// Return current TT statistics from the persistent global searcher, if any.
/// When no global searcher exists yet, this returns zeros.
pub fn get_current_tt_stats() -> SearchStats {
    GLOBAL_SEARCHER.with(|cell| {
        let opt = cell.borrow();
        if let Some(ref searcher) = *opt {
            build_search_stats(searcher)
        } else {
            SearchStats {
                tt_capacity: 0,
                tt_used: 0,
                tt_fill_permille: 0,
            }
        }
    })
}

/// Reset the global search state.
/// Call this when starting a brand new game so old entries don't carry over.
pub fn reset_search_state() {
    GLOBAL_SEARCHER.with(|cell| {
        *cell.borrow_mut() = None;
    });

    let seed = (random() * 1.8446744073709552e19) as u64;
    crate::search::noisy::reset_noise_seed(seed);
}

/// Search state that persists across the search
pub struct Searcher {
    pub nodes: u64,
    pub qnodes: u64,

    pub timer: Timer,
    pub time_limit_ms: u128,
    pub stopped: bool,
    pub seldepth: usize,

    // Transposition table
    pub tt: TranspositionTable,

    // Triangular PV table: flat array indexed by pv_table[ply * MAX_PLY + offset]
    // Using Box to avoid stack overflow with 64*64 = 4096 Move entries
    pub pv_table: Box<[Option<Move>; MAX_PLY * MAX_PLY]>,
    pub pv_length: [usize; MAX_PLY],

    // Killer moves (2 per ply)
    pub killers: Vec<[Option<Move>; 2]>,

    // History heuristic [piece_type][to_square_hash]
    pub history: [[i32; 256]; 32],

    // Capture history [moving_piece_type][captured_piece_type]
    // Used to improve capture ordering beyond pure MVV-LVA
    pub capture_history: [[i32; 32]; 32],

    // Countermove heuristic [prev_from_hash][prev_to_hash] -> (piece_type, to_x, to_y)
    // Stores the move that refuted the previous move (for quiet beta cutoffs).
    // Using (u8, i16, i16) to store piece type and destination coords.
    pub countermoves: [[(u8, i16, i16); 256]; 256],

    // Previous move info for countermove heuristic (from_hash, to_hash)
    pub prev_move_stack: Vec<(usize, usize)>,

    // Static eval stack for "improving" heuristic
    // Stores eval at each ply to detect if position is improving
    pub eval_stack: Vec<i32>,

    // Best move from previous iteration
    pub best_move_root: Option<Move>,

    // Previous iteration score for aspiration windows
    pub prev_score: i32,

    // Silent mode - no info output
    pub silent: bool,

    // Per-ply reusable move buffers to avoid Vec allocations in the search
    pub move_buffers: Vec<Vec<Move>>,

    // Move history stack for continuation history (move at each ply)
    pub move_history: Vec<Option<Move>>,

    // Moved piece history stack (piece type that moved at each ply)
    pub moved_piece_history: Vec<u8>,

    // Continuation history: [prev_piece_type][prev_to_hash][cur_from_hash][cur_to_hash]
    // Using smaller dimensions (16*32*32*32*4 = 2MB) to fit in WASM memory
    pub cont_history: Vec<[[[i32; 32]; 32]; 32]>,

    // MultiPV: moves to exclude from root search (for finding 2nd, 3rd, etc. best moves)
    // Stored as (from_x, from_y, to_x, to_y) tuples for fast comparison without cloning
    pub excluded_moves: Vec<(i64, i64, i64, i64)>,
}

impl Searcher {
    pub fn new(time_limit_ms: u128) -> Self {
        // Triangular PV table - heap allocated to avoid stack overflow
        let pv_table = Box::new([None; MAX_PLY * MAX_PLY]);

        let mut killers = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            killers.push([None, None]);
        }

        let mut move_buffers = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            move_buffers.push(Vec::with_capacity(64));
        }

        Searcher {
            nodes: 0,
            qnodes: 0,
            timer: Timer::new(),
            time_limit_ms,
            stopped: false,
            seldepth: 0,
            tt: TranspositionTable::new(32),
            pv_table,
            pv_length: [0; MAX_PLY],
            killers,
            history: [[0; 256]; 32],
            capture_history: [[0; 32]; 32],
            countermoves: [[(0, 0, 0); 256]; 256],
            prev_move_stack: vec![(0, 0); MAX_PLY],
            eval_stack: vec![0; MAX_PLY],
            best_move_root: None,
            prev_score: 0,
            silent: false,
            move_buffers,
            move_history: vec![None; MAX_PLY],
            moved_piece_history: vec![0; MAX_PLY],
            cont_history: vec![[[[0i32; 32]; 32]; 32]; 16],
            excluded_moves: Vec::new(),
        }
    }

    pub fn reset_for_iteration(&mut self) {
        // Note: DO NOT reset timer here - we want global time limit across all iterations
        self.nodes = 0;
        self.qnodes = 0;
        self.stopped = false;
        self.seldepth = 0;

        // Reset PV lengths only - much faster than clearing entire array
        // The PV entries will be overwritten as needed during search
        self.pv_length = [0; MAX_PLY];
    }

    /// Decay history scores at the start of each iteration
    pub fn decay_history(&mut self) {
        for row in &mut self.history {
            for val in row.iter_mut() {
                *val = *val * 9 / 10; // Decay by 10%
            }
        }
    }

    /// Gravity-style history update: scales updates based on current value and clamps to [-MAX_HISTORY, MAX_HISTORY].
    #[inline]
    pub fn update_history(&mut self, piece: PieceType, idx: usize, bonus: i32) {
        let max_h = max_history();
        let mut clamped = bonus;
        if clamped > max_h {
            clamped = max_h;
        }
        if clamped < -max_h {
            clamped = -max_h;
        }

        let entry = &mut self.history[piece as usize][idx];
        *entry += clamped - *entry * clamped.abs() / max_h;
    }

    #[inline]
    pub fn check_time(&mut self) -> bool {
        // Fast-path: no time limit (used by offline test/perft helpers).
        if self.time_limit_ms == u128::MAX {
            return false;
        }

        // Check time only every N nodes to keep the hot path cheap, especially
        // on wasm where elapsed_ms() crosses the JS boundary.
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        const TIME_CHECK_MASK: u64 = 8191; // every 8192 nodes
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        const TIME_CHECK_MASK: u64 = 2047; // every 2048 nodes

        if self.nodes & TIME_CHECK_MASK == 0 {
            if self.timer.elapsed_ms() >= self.time_limit_ms {
                self.stopped = true;
            }
        }
        self.stopped
    }

    /// Format PV line as string
    pub fn format_pv(&self) -> String {
        let mut pv_str = String::new();
        // Root PV is at pv_table[0..pv_length[0]]
        for i in 0..self.pv_length[0] {
            if let Some(m) = self.pv_table[i] {
                if !pv_str.is_empty() {
                    pv_str.push(' ');
                }
                pv_str.push_str(&format!("{},{}->{},{}", m.from.x, m.from.y, m.to.x, m.to.y));
            }
        }
        pv_str
    }

    /// Print UCI-style info string with optional MultiPV index
    pub fn print_info(&self, depth: usize, score: i32) {
        self.print_info_multipv(depth, score, 1);
    }

    /// Print UCI-style info string with MultiPV index
    pub fn print_info_multipv(&self, depth: usize, score: i32, multipv: usize) {
        let time_ms = self.timer.elapsed_ms();
        let nps = if time_ms > 0 {
            (self.nodes as u128 * 1000) / time_ms
        } else {
            0
        };
        let tt_fill = self.tt.fill_permille();

        // Proper mate score display
        let score_str = if score > MATE_SCORE {
            // Positive mate score = we are mating
            let mate_in = (MATE_VALUE - score + 1) / 2;
            format!("mate {}", mate_in)
        } else if score < -MATE_SCORE {
            // Negative mate score = we are getting mated
            let mate_in = (MATE_VALUE + score + 1) / 2;
            format!("mate -{}", mate_in)
        } else {
            format!("cp {}", score)
        };

        let pv = self.format_pv();

        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            use crate::log;
            if multipv > 1 {
                log(&format!(
                    "info depth {} seldepth {} multipv {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.seldepth,
                    multipv,
                    score_str,
                    self.nodes,
                    self.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                ));
            } else {
                log(&format!(
                    "info depth {} seldepth {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.seldepth,
                    score_str,
                    self.nodes,
                    self.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                ));
            }
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            if multipv > 1 {
                eprintln!(
                    "info depth {} seldepth {} multipv {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.seldepth,
                    multipv,
                    score_str,
                    self.nodes,
                    self.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                );
            } else {
                eprintln!(
                    "info depth {} seldepth {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.seldepth,
                    score_str,
                    self.nodes,
                    self.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                );
            }
        }
    }
}

/// Core timed search implementation using a provided searcher.
fn search_with_searcher(
    searcher: &mut Searcher,
    game: &mut GameState,
    max_depth: usize,
) -> Option<(Move, i32)> {
    let moves = game.get_legal_moves();
    if moves.is_empty() {
        return None;
    }

    // If only one move, return immediately with a simple static eval as score.
    if moves.len() == 1 {
        let single = moves[0].clone();
        let score = evaluate(game);
        return Some((single, score));
    }

    // Find first legal move as ultimate fallback.
    // CRITICAL: We must ALWAYS check legality, even under time pressure.
    // Accepting an illegal move (one that leaves our king in check) is never acceptable.
    let fallback_move = moves
        .iter()
        .find(|m| {
            let undo = game.make_move(m);
            let legal = !game.is_move_illegal();
            game.undo_move(m, undo);
            legal
        })
        .cloned();

    let mut best_move: Option<Move> = fallback_move.clone();
    let mut best_score = -INFINITY;
    let mut stability: usize = 0;
    let mut prev_iter_score: i32 = 0;
    let mut has_prev_iter_score = false;
    let mut prev_root_move_coords: Option<(i64, i64, i64, i64)> = None;

    // Iterative deepening with aspiration windows
    for depth in 1..=max_depth {
        searcher.reset_for_iteration();
        searcher.decay_history();

        // Immediate time check at start of each iteration
        if searcher.timer.elapsed_ms() >= searcher.time_limit_ms {
            searcher.stopped = true;
            break;
        }

        let score = if depth == 1 {
            // First iteration: full window
            negamax_root(searcher, game, depth, -INFINITY, INFINITY)
        } else {
            // Aspiration window search
            let asp_win = aspiration_window();
            let mut alpha = searcher.prev_score - asp_win;
            let mut beta = searcher.prev_score + asp_win;
            let mut window_size = asp_win;
            let mut result;
            let mut retries = 0;

            loop {
                result = negamax_root(searcher, game, depth, alpha, beta);
                retries += 1;

                if searcher.stopped {
                    break;
                }

                if result <= alpha {
                    // Failed low - widen alpha
                    window_size *= aspiration_fail_mult();
                    alpha = searcher.prev_score - window_size;
                } else if result >= beta {
                    // Failed high - widen beta
                    window_size *= aspiration_fail_mult();
                    beta = searcher.prev_score + window_size;
                } else {
                    // Score within window
                    break;
                }

                // Fallback to full window if window gets too large or too many retries
                if window_size > 1000 || retries >= 4 {
                    result = negamax_root(searcher, game, depth, -INFINITY, INFINITY);
                    break;
                }
            }
            result
        };

        // Update best move - even if stopped, use best from this iteration if found
        // Root PV is at pv_table[0]
        if let Some(pv_move) = searcher.pv_table[0] {
            best_move = Some(pv_move);
            best_score = score;
            searcher.best_move_root = Some(pv_move);
            searcher.prev_score = score;

            let coords = (pv_move.from.x, pv_move.from.y, pv_move.to.x, pv_move.to.y);
            if let Some(prev_coords) = prev_root_move_coords {
                if prev_coords == coords {
                    stability += 1;
                } else {
                    stability = 0;
                }
            } else {
                stability = 0;
            }
            prev_root_move_coords = Some(coords);
        }

        if !searcher.stopped && !searcher.silent {
            searcher.print_info(depth, score);
        }

        // Check if we found mate or time is up
        if searcher.stopped || best_score.abs() > MATE_SCORE {
            break;
        }

        // If we've used more than 50% of time, don't start another iteration
        if searcher.time_limit_ms != u128::MAX {
            let elapsed = searcher.timer.elapsed_ms();
            let limit = searcher.time_limit_ms;

            if best_move.is_some() {
                let mut factor = 1.1_f64 - 0.03_f64 * (stability as f64);
                if factor < 0.5 {
                    factor = 0.5;
                }

                if has_prev_iter_score && best_score - prev_iter_score > aspiration_window() {
                    factor *= 1.1;
                }

                if factor > 1.0 {
                    factor = 1.0;
                }

                let ideal_ms = (limit as f64 * factor) as u128;
                let soft_limit = std::cmp::min(limit, ideal_ms);

                if elapsed >= soft_limit {
                    break;
                }

                prev_iter_score = best_score;
                has_prev_iter_score = true;
            }

            let cutoff = if limit <= 300 {
                // Very short thinks: keep ~50% heuristic
                limit / 2
            } else if limit <= 2000 {
                // Short blitz: leave ~250ms safety buffer
                limit.saturating_sub(250)
            } else if limit <= 8000 {
                // Rapid-ish: leave ~500ms safety buffer (e.g. 4s -> 3.5s)
                limit.saturating_sub(500)
            } else {
                // Long thinks: use almost all allotted time but keep a small buffer
                limit.saturating_sub(2000)
            };

            if elapsed >= cutoff {
                break;
            }
        }
    }

    // Increment TT age for next search
    searcher.tt.increment_age();

    best_move.map(|m| (m, best_score))
}

/// Time-limited search that returns the best move, its evaluation (cp from side-to-move's
/// perspective), and simple TT statistics. This is the main public search entry point.
pub fn get_best_move(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    silent: bool,
) -> Option<(Move, i32, SearchStats)> {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();

    // Use a fresh searcher per call (no persistent TT/state between searches).
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    let result = search_with_searcher(&mut searcher, game, max_depth);
    let stats = build_search_stats(&searcher);
    result.map(|(m, eval)| (m, eval, stats))
}

/// MultiPV-enabled search that returns up to `multi_pv` best moves with their evaluations.
///
/// When `multi_pv` is 1, this has zero overhead - it's equivalent to `get_best_move`.
/// For `multi_pv` > 1, at each depth all root moves are searched and the top N are returned.
pub fn get_best_moves_multipv(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    multi_pv: usize,
    silent: bool,
) -> MultiPVResult {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();

    let multi_pv = multi_pv.max(1);

    // MultiPV = 1: Zero overhead path - just do normal search
    if multi_pv == 1 {
        let mut searcher = Searcher::new(time_limit_ms);
        searcher.time_limit_ms = time_limit_ms;
        searcher.silent = silent;

        let mut lines: Vec<PVLine> = Vec::with_capacity(1);
        if let Some((best_move, score)) = search_with_searcher(&mut searcher, game, max_depth) {
            let pv = extract_pv(&searcher);
            let depth = max_depth.min(searcher.seldepth.max(1));
            lines.push(PVLine {
                mv: best_move,
                score,
                depth,
                pv,
            });
        }
        let stats = build_search_stats(&searcher);
        return MultiPVResult { lines, stats };
    }

    // MultiPV > 1: Search with special root handling to collect multiple best moves
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.time_limit_ms = time_limit_ms;
    searcher.silent = silent;

    // Get all legal moves upfront
    let moves = game.get_legal_moves();
    if moves.is_empty() {
        let stats = build_search_stats(&searcher);
        return MultiPVResult {
            lines: Vec::new(),
            stats,
        };
    }

    // If only one move, return it immediately
    if moves.len() == 1 {
        let single = moves[0].clone();
        let score = crate::evaluation::evaluate(game);
        let stats = build_search_stats(&searcher);
        return MultiPVResult {
            lines: vec![PVLine {
                mv: single,
                score,
                depth: 0,
                pv: Vec::new(),
            }],
            stats,
        };
    }

    // Find legal moves only (filter pseudo-legal)
    let mut legal_root_moves: Vec<Move> = Vec::with_capacity(moves.len());
    for m in &moves {
        let undo = game.make_move(m);
        let legal = !game.is_move_illegal();
        game.undo_move(m, undo);
        if legal {
            legal_root_moves.push(m.clone());
        }
    }

    if legal_root_moves.is_empty() {
        let stats = build_search_stats(&searcher);
        return MultiPVResult {
            lines: Vec::new(),
            stats,
        };
    }

    // Cap multi_pv at number of legal moves
    let multi_pv = multi_pv.min(legal_root_moves.len());

    // Store (move, score, pv) for each root move at current depth
    let mut root_scores: Vec<(Move, i32, Vec<Move>)> = Vec::with_capacity(legal_root_moves.len());
    let mut best_lines: Vec<PVLine> = Vec::with_capacity(multi_pv);

    // Iterative deepening
    for depth in 1..=max_depth {
        searcher.reset_for_iteration();
        searcher.decay_history();

        // Time check at start of each iteration
        if searcher.timer.elapsed_ms() >= searcher.time_limit_ms {
            searcher.stopped = true;
            break;
        }

        root_scores.clear();

        // Search each root move
        for (move_idx, m) in legal_root_moves.iter().enumerate() {
            if searcher.stopped {
                break;
            }

            let undo = game.make_move(m);

            // Set up prev move info for child search
            let prev_from_hash = hash_move_from(m);
            let prev_to_hash = hash_move_dest(m);
            searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

            // For MultiPV, we need to search all moves to get their scores.
            // First move gets full window, subsequent moves use PVS.
            let score = if move_idx == 0 {
                -negamax(&mut searcher, game, depth - 1, 1, -INFINITY, INFINITY, true)
            } else {
                // Use PVS for efficiency, but we need to re-search with full window
                // if the move might be in the top-N
                let alpha = if root_scores.len() >= multi_pv {
                    // We already have enough candidates; use the worst score as lower bound
                    root_scores
                        .iter()
                        .map(|(_, s, _)| *s)
                        .min()
                        .unwrap_or(-INFINITY)
                } else {
                    -INFINITY
                };

                let mut s = -negamax(&mut searcher, game, depth - 1, 1, -alpha - 1, -alpha, true);
                if s > alpha && !searcher.stopped {
                    // Re-search with full window to get accurate score
                    s = -negamax(&mut searcher, game, depth - 1, 1, -INFINITY, INFINITY, true);
                }
                s
            };

            game.undo_move(m, undo);

            if !searcher.stopped {
                // Extract PV for this move from ply 1's triangular row
                let child_base = MAX_PLY; // ply 1 base offset
                let mut pv = Vec::with_capacity(searcher.pv_length[1] + 1);
                pv.push(*m);
                for i in 0..searcher.pv_length[1] {
                    if let Some(pv_move) = searcher.pv_table[child_base + i] {
                        pv.push(pv_move);
                    }
                }
                root_scores.push((m.clone(), score, pv));
            }
        }

        if searcher.stopped && root_scores.is_empty() {
            break;
        }

        // Sort by score descending
        root_scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Update best_lines with results from this depth
        best_lines.clear();
        for (idx, (mv, score, pv)) in root_scores.iter().take(multi_pv).enumerate() {
            best_lines.push(PVLine {
                mv: mv.clone(),
                score: *score,
                depth,
                pv: pv.clone(),
            });

            // Print info for each PV line
            if !silent && !searcher.stopped {
                // Format PV string
                let pv_str: String = pv
                    .iter()
                    .map(|m| format!("{},{}->{},{}", m.from.x, m.from.y, m.to.x, m.to.y))
                    .collect::<Vec<_>>()
                    .join(" ");

                let time_ms = searcher.timer.elapsed_ms();
                let nps = if time_ms > 0 {
                    (searcher.nodes as u128 * 1000) / time_ms
                } else {
                    0
                };

                let score_str = if *score > MATE_SCORE {
                    let mate_in = (MATE_VALUE - score + 1) / 2;
                    format!("mate {}", mate_in)
                } else if *score < -MATE_SCORE {
                    let mate_in = (MATE_VALUE + score + 1) / 2;
                    format!("mate -{}", mate_in)
                } else {
                    format!("cp {}", score)
                };

                #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
                {
                    use crate::log;
                    log(&format!(
                        "info depth {} seldepth {} multipv {} score {} nodes {} nps {} time {} pv {}",
                        depth,
                        searcher.seldepth,
                        idx + 1,
                        score_str,
                        searcher.nodes,
                        nps,
                        time_ms,
                        pv_str
                    ));
                }
                #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
                {
                    eprintln!(
                        "info depth {} seldepth {} multipv {} score {} nodes {} nps {} time {} pv {}",
                        depth,
                        searcher.seldepth,
                        idx + 1,
                        score_str,
                        searcher.nodes,
                        nps,
                        time_ms,
                        pv_str
                    );
                }
            }
        }

        // Check for mate or time up
        if searcher.stopped {
            break;
        }
        if !best_lines.is_empty() && best_lines[0].score.abs() > MATE_SCORE {
            break;
        }

        // Soft time limit check - don't start next iteration if past 50%
        if searcher.time_limit_ms != u128::MAX {
            let elapsed = searcher.timer.elapsed_ms();
            if elapsed >= searcher.time_limit_ms / 2 {
                break;
            }
        }
    }

    // Update PV table with best move for stats
    if !best_lines.is_empty() {
        searcher.pv_table[0] = Some(best_lines[0].mv);
        searcher.pv_length[0] = 1;
    }

    searcher.tt.increment_age();
    let stats = build_search_stats(&searcher);
    MultiPVResult {
        lines: best_lines,
        stats,
    }
}

/// Extract the PV line from the searcher's PV table as a Vec<Move>.
fn extract_pv(searcher: &Searcher) -> Vec<Move> {
    // Root PV is at pv_table[0..pv_length[0]]
    let mut pv = Vec::with_capacity(searcher.pv_length[0]);
    for i in 0..searcher.pv_length[0] {
        if let Some(m) = searcher.pv_table[i] {
            pv.push(m);
        }
    }
    pv
}

pub fn negamax_node_count_for_depth(game: &mut GameState, depth: usize) -> u64 {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();

    let mut searcher = Searcher::new(u128::MAX);
    searcher.reset_for_iteration();
    searcher.decay_history();
    searcher.tt.clear();
    let _ = negamax_root(&mut searcher, game, depth, -INFINITY, INFINITY);
    searcher.nodes
}

/// Root negamax - special handling for root node
fn negamax_root(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    // Save original alpha for TT flag determination
    let alpha_orig = alpha;

    searcher.pv_length[0] = 0;

    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;

    // Probe TT for best move from previous search
    if let Some((_, best)) = searcher.tt.probe(hash, alpha, beta, depth, 0) {
        tt_move = best;
    }

    let in_check = game.is_in_check();

    // Reuse per-ply move buffer at root (ply 0)
    let mut moves = Vec::new();
    std::mem::swap(&mut moves, &mut searcher.move_buffers[0]);
    game.get_legal_moves_into(&mut moves);

    // Sort moves at root (TT move first, then by score)
    sort_moves_root(searcher, game, &mut moves, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut fallback_move: Option<Move> = None; // First legal move as fallback
    let mut legal_moves = 0;

    for m in &moves {
        // Skip excluded moves (for MultiPV subsequent passes)
        if !searcher.excluded_moves.is_empty() {
            let coords = (m.from.x, m.from.y, m.to.x, m.to.y);
            if searcher.excluded_moves.contains(&coords) {
                continue;
            }
        }

        let undo = game.make_move(m);

        // Check if move is illegal (leaves our king in check)
        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        // At the root, this move becomes the previous move for child ply 1,
        // stored as (from_hash, to_hash).
        let prev_entry_backup = searcher.prev_move_stack[0];
        let prev_from_hash = hash_move_from(m);
        let prev_to_hash = hash_move_dest(m);
        searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

        legal_moves += 1;

        // Save first legal move as fallback
        if fallback_move.is_none() {
            fallback_move = Some(m.clone());
        }

        let score;
        if legal_moves == 1 {
            // Full window search for first legal move
            score = -negamax(searcher, game, depth - 1, 1, -beta, -alpha, true);
        } else {
            // PVS: Null window first, then re-search if it improves alpha
            let mut s = -negamax(searcher, game, depth - 1, 1, -alpha - 1, -alpha, true);
            if s > alpha && s < beta {
                s = -negamax(searcher, game, depth - 1, 1, -beta, -alpha, true);
            }
            score = s;
        }

        game.undo_move(m, undo);

        // Restore previous-move stack entry for root after returning from child.
        searcher.prev_move_stack[0] = prev_entry_backup;

        if searcher.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());

            if score > alpha {
                alpha = score;

                // Update PV using triangular indexing
                // Root (ply 0) stores PV at pv_table[0..], child (ply 1) at pv_table[MAX_PLY..]
                searcher.pv_table[0] = Some(*m); // Head of PV is this move
                let child_len = searcher.pv_length[1];
                let child_base = MAX_PLY;
                for j in 0..child_len {
                    searcher.pv_table[1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[0] = child_len + 1;
            }
        }

        if alpha >= beta {
            break;
        }
    }

    // Checkmate or stalemate (or loss by capture-all-pieces variants)
    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        return if in_check || no_pieces {
            -MATE_VALUE
        } else {
            0
        };
    }

    // Swap back move buffer for reuse in future searches
    std::mem::swap(&mut searcher.move_buffers[0], &mut moves);

    // Store in TT with correct flag based on original alpha
    let tt_flag = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    searcher
        .tt
        .store(hash, depth, tt_flag, best_score, best_move, 0);

    best_score
}

/// Main negamax with alpha-beta pruning, NMP, LMR, and TT
fn negamax(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    ply: usize,
    mut alpha: i32,
    mut beta: i32,
    allow_null: bool,
) -> i32 {
    // Save original alpha/beta for TT flag determination (per Wikipedia pseudocode)
    let alpha_orig = alpha;
    let beta_orig = beta;

    searcher.nodes += 1;
    // Initialize PV length to 0; will be updated if alpha is raised
    searcher.pv_length[ply] = 0;

    // Update seldepth
    if ply > searcher.seldepth {
        searcher.seldepth = ply;
    }

    // Time check
    if searcher.check_time() {
        return 0;
    }

    // Check for max ply
    if ply >= MAX_PLY - 1 {
        return evaluate(game);
    }

    // Fifty-move rule: 100 half-moves without pawn move or capture is a draw
    if game.is_fifty() {
        return 0;
    }

    // Generate hash for TT
    let hash = TranspositionTable::generate_hash(game);

    // Threefold repetition detection (uses hash_stack built during make_move/undo_move)
    if ply > 0 && game.is_threefold() {
        // Treat repetition as a slightly worse outcome than a neutral eval
        // from the current side's perspective. This nudges the search away
        // from pointless threefolds when other equal moves exist, while still
        // allowing repetition in clearly worse positions.
        return -repetition_penalty();
    }

    // Mate distance pruning (not at root)
    if ply > 0 {
        let mate_score = MATE_VALUE - ply as i32;
        if alpha < -mate_score {
            alpha = -mate_score;
        }
        if beta > mate_score - 1 {
            beta = mate_score - 1;
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let in_check = game.is_in_check();
    let is_pv = beta > alpha + 1;

    // Base case: quiescence search at leaf nodes
    if depth == 0 {
        return quiescence(searcher, game, ply, alpha, beta);
    }

    // Depth may be adjusted by check extensions and internal iterative
    // reductions (IIR). Start from the caller-provided depth.
    let mut depth = depth;

    // Check extension (limited to avoid infinite recursion)
    // Only extend if we're not too deep already
    if in_check && ply < MAX_PLY / 2 {
        depth += 1;
    }
    let mut tt_move: Option<Move> = None;

    if let Some((score, best)) = searcher.tt.probe(hash, alpha, beta, depth, ply) {
        tt_move = best;

        // Use TT cutoff in non-PV nodes
        if !is_pv && score != INFINITY + 1 {
            return score;
        }
    }

    // Static evaluation for pruning decisions
    let static_eval = if in_check {
        -MATE_VALUE + ply as i32
    } else {
        evaluate(game)
    };

    // Store eval for "improving" heuristic
    searcher.eval_stack[ply] = static_eval;

    // Check if position is improving (eval better than 2 plies ago)
    // This is used to adjust pruning aggressiveness
    let improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true // Assume improving at root or when in check
    };

    // Internal Iterative Reductions (IIR): if we have no TT move in an
    // expected cut-node (Stockfish-style cutNode: non-PV, beta == alpha+1),
    // and we are not in check, reduce depth slightly. The idea is that the
    // node is likely unimportant if no best move was stored previously.
    let cut_node = !is_pv && beta == alpha + 1;
    if tt_move.is_none() && cut_node && !in_check && depth >= 4 {
        depth -= 1;
    }

    // Pruning techniques (not in check, not PV node)
    if !in_check && !is_pv {
        // Reverse Futility Pruning (Static Null Move Pruning)
        if depth < rfp_max_depth() && static_eval - rfp_margin_per_depth() * depth as i32 >= beta {
            return static_eval;
        }

        // Null Move Pruning
        if allow_null && depth >= nmp_min_depth() && static_eval >= beta {
            // Check if we have non-pawn material (avoid zugzwang)
            let has_pieces = game.board.pieces.iter().any(|(_, p)| {
                p.color() == game.turn
                    && p.piece_type() != PieceType::Pawn
                    && p.piece_type() != PieceType::King
            });

            if has_pieces {
                // Make null move (proper tracking for repetition detection)
                let saved_ep = game.en_passant.clone();
                game.make_null_move();

                let r = nmp_reduction() + depth / 6;
                let null_score = -negamax(
                    searcher,
                    game,
                    depth.saturating_sub(1 + r),
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                );

                game.unmake_null_move();
                game.en_passant = saved_ep;

                if searcher.stopped {
                    return 0;
                }

                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    // Reuse per-ply move buffer for this node
    let mut moves = Vec::new();
    std::mem::swap(&mut moves, &mut searcher.move_buffers[ply]);
    game.get_legal_moves_into(&mut moves);

    // Sort moves for better pruning
    sort_moves(searcher, game, &mut moves, ply, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    // Track quiet moves searched at this node for history maluses
    let mut quiets_searched: Vec<Move> = Vec::new();

    // Futility pruning flag
    let futility_pruning = !in_check && !is_pv && depth <= 3;
    let futility_base = if futility_pruning {
        static_eval + futility_margin(depth)
    } else {
        0
    };

    for m in &moves {
        let captured_piece = game.board.get_piece(&m.to.x, &m.to.y);
        let is_capture = captured_piece.map_or(false, |p| !p.piece_type().is_neutral_type());
        let captured_type = captured_piece.map(|p| p.piece_type());
        let is_promotion = m.promotion.is_some();

        // Futility pruning - skip quiet moves that can't raise alpha
        if futility_pruning && legal_moves > 0 && !is_capture && !is_promotion {
            if futility_base <= alpha {
                continue;
            }
        }

        // Late Move Pruning (LMP) - skip quiet moves late in the move list at shallow depths
        // Only prune after we have at least one legal move (best_score != -INFINITY)
        if !in_check && !is_pv && depth <= 4 && depth > 0 && !is_capture && !is_promotion {
            let threshold = lmp_threshold(depth);
            if legal_moves >= threshold && best_score > -MATE_SCORE {
                continue;
            }
        }

        let undo = game.make_move(m);

        // Check if move is illegal (leaves our king in check)
        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        // Record quiet moves searched at this node for history maluses
        if !is_capture && !is_promotion {
            quiets_searched.push(m.clone());
        }

        // For this node at `ply`, this move becomes the previous move for child
        // ply + 1, stored as (from_hash, to_hash).
        let prev_entry_backup = searcher.prev_move_stack[ply];
        let from_hash = hash_move_from(m);
        let to_hash = hash_move_dest(m);
        searcher.prev_move_stack[ply] = (from_hash, to_hash);

        // Store move and piece info for continuation history
        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        searcher.move_history[ply] = Some(m.clone());
        searcher.moved_piece_history[ply] = m.piece.piece_type() as u8;

        legal_moves += 1;

        let score;
        if legal_moves == 1 {
            // Full window search for first legal move
            score = -negamax(searcher, game, depth - 1, ply + 1, -beta, -alpha, true);
        } else {
            // Late Move Reductions
            let mut reduction = 0;
            if depth >= lmr_min_depth()
                && legal_moves >= lmr_min_moves()
                && !in_check
                && !is_capture
            {
                reduction = 1
                    + (legal_moves as f32).ln() as usize * (depth as f32).ln() as usize
                        / lmr_divisor();

                // Reduce more when position is not improving
                if !improving {
                    reduction += 1;
                }

                reduction = reduction.min(depth - 2);
            }

            // Base child depth after LMR
            let mut new_depth = depth as i32 - 1 - reduction as i32;

            // History Leaf Pruning (Fruit-style)
            // Only in non-PV, quiet, shallow nodes and after enough moves
            if !in_check
                && !is_pv
                && !is_capture
                && !is_promotion
                && depth <= hlp_max_depth()
                && legal_moves >= hlp_min_moves()
                && best_score > -MATE_SCORE
            {
                let idx = hash_move_dest(m);
                let value = searcher.history[m.piece.piece_type() as usize][idx];

                if value < hlp_history_reduce() {
                    // Extra reduction based on poor history
                    new_depth -= 1;

                    // If depth after reductions would drop to quiescence or below
                    // and history is really bad, prune this move entirely.
                    if new_depth <= 0 && value < hlp_history_leaf() {
                        game.undo_move(m, undo);
                        continue;
                    }
                }
            }

            // Allow new_depth to reach 0 so that the child call will
            // transition to quiescence (depth == 0) instead of being
            // artificially clamped to 1, which can cause very deep
            // "depth 1" trees and huge node counts.
            let search_depth = if new_depth <= 0 {
                0
            } else {
                new_depth as usize
            };

            // Null window search with possible reduction
            let mut s = -negamax(
                searcher,
                game,
                search_depth,
                ply + 1,
                -alpha - 1,
                -alpha,
                true,
            );

            // Re-search at full depth if it looks promising
            if s > alpha && (reduction > 0 || s < beta) {
                s = -negamax(searcher, game, depth - 1, ply + 1, -beta, -alpha, true);
            }
            score = s;
        }

        game.undo_move(m, undo);

        // Restore previous-move stack entry for this ply after child returns.
        searcher.prev_move_stack[ply] = prev_entry_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());

            if score > alpha {
                alpha = score;

                // Update PV using triangular indexing
                // ply stores PV at pv_table[ply * MAX_PLY..], child at pv_table[(ply+1) * MAX_PLY..]
                let ply_base = ply * MAX_PLY;
                let child_base = (ply + 1) * MAX_PLY;

                searcher.pv_table[ply_base] = Some(*m); // Head of PV is this move
                let child_len = searcher.pv_length[ply + 1];
                for j in 0..child_len {
                    searcher.pv_table[ply_base + 1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[ply] = child_len + 1;
            }
        }

        if alpha >= beta {
            if !is_capture {
                // History bonus for quiet cutoff move, with maluses for previously searched quiets
                let idx = hash_move_dest(m);
                let bonus = history_bonus_base() * depth as i32 - history_bonus_sub();
                let adj = bonus.min(history_bonus_cap());
                let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;

                searcher.update_history(m.piece.piece_type(), idx, bonus);

                for quiet in &quiets_searched {
                    let qidx = hash_move_dest(quiet);
                    if quiet.piece.piece_type() == m.piece.piece_type() && qidx == idx {
                        continue;
                    }
                    searcher.update_history(quiet.piece.piece_type(), qidx, -bonus);
                }

                // Killer move heuristic (for non-captures)
                searcher.killers[ply][1] = searcher.killers[ply][0].clone();
                searcher.killers[ply][0] = Some(*m);

                // Countermove heuristic: on a quiet beta cutoff, record this move
                // as the countermove to the move that led into this node.
                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (m.piece.piece_type() as u8, m.to.x as i16, m.to.y as i16);
                    }
                }

                // Continuation history update (like Zig: 1-ply, 2-ply, 4-ply back)

                for &plies_ago in &[0usize, 1, 3] {
                    if ply >= plies_ago + 1 {
                        if let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1] {
                            let prev_piece =
                                searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_piece < 16 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);

                                // Update all searched quiets (best with bonus, others with malus)
                                for quiet in &quiets_searched {
                                    let q_from_hash = hash_coord_32(quiet.from.x, quiet.from.y);
                                    let q_to_hash = hash_coord_32(quiet.to.x, quiet.to.y);
                                    let is_best = quiet.from == m.from && quiet.to == m.to;

                                    let entry = &mut searcher.cont_history[prev_piece]
                                        [prev_to_hash][q_from_hash][q_to_hash];
                                    if is_best {
                                        *entry += adj - *entry * adj / max_history;
                                    } else {
                                        *entry += -adj - *entry * adj / max_history;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if let Some(cap_type) = captured_type {
                // Update capture history on beta cutoff
                let bonus = (depth * depth) as i32;
                searcher.capture_history[m.piece.piece_type() as usize][cap_type as usize] += bonus;
            }
            break;
        }
    }

    // Swap back move buffer for this ply before returning
    std::mem::swap(&mut searcher.move_buffers[ply], &mut moves);

    // Checkmate or stalemate detection (also treat no-pieces as loss)
    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        if in_check || no_pieces {
            return -MATE_VALUE + ply as i32;
        } else {
            return 0; // Stalemate
        }
    }

    // Store in TT with correct flag based on original alpha/beta (per Wikipedia pseudocode)
    // - UPPERBOUND: best_score <= alpha_orig (failed low, didn't improve alpha)
    // - LOWERBOUND: best_score >= beta_orig (failed high, caused beta cutoff)
    // - EXACT: alpha_orig < best_score < beta_orig (true minimax value)
    let tt_flag = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta_orig {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    searcher
        .tt
        .store(hash, depth, tt_flag, best_score, best_move, ply);

    best_score
}

/// Quiescence search - only search captures to avoid horizon effect
fn quiescence(
    searcher: &mut Searcher,
    game: &mut GameState,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    searcher.nodes += 1;
    searcher.qnodes += 1;

    // Update seldepth
    if ply > searcher.seldepth {
        searcher.seldepth = ply;
    }

    if searcher.check_time() {
        return 0;
    }

    let in_check = game.is_in_check();

    // Stand pat (not when in check)
    let stand_pat = if in_check {
        -MATE_VALUE + ply as i32
    } else {
        evaluate(game)
    };

    if !in_check {
        if stand_pat >= beta {
            return beta;
        }

        if alpha < stand_pat {
            alpha = stand_pat;
        }
    }

    if ply >= MAX_PLY - 1 {
        return stand_pat;
    }

    // When in check, generate all pseudo-legal moves (evasions) via the normal generator.
    // When not in check, use a specialized capture-only generator to avoid creating
    // thousands of quiet moves only to filter them out.
    // Reuse per-ply move buffer to avoid Vec allocations inside quiescence.
    let mut tactical_moves = Vec::new();
    std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);

    if in_check {
        game.get_evasion_moves_into(&mut tactical_moves);
    } else {
        get_quiescence_captures(
            &game.board,
            game.turn,
            &game.special_rights,
            &game.en_passant,
            &game.game_rules,
            &game.spatial_indices,
            &mut tactical_moves,
        );
    }

    // Sort captures by MVV-LVA
    sort_captures(game, &mut tactical_moves);

    let mut best_score = stand_pat;
    let mut legal_moves = 0;

    // Delta pruning margin (safety buffer for positional factors)
    const DELTA_MARGIN: i32 = 200;

    for m in &tactical_moves {
        // SEE-based pruning and delta pruning for captures when not in check.
        // static_exchange_eval returns 0 for non-captures or special cases
        // (e.g. en passant target squares), so it is safe to call unconditionally.
        if !in_check {
            let see_gain = static_exchange_eval(game, m);

            // Prune clearly losing captures that don't even break even materially.
            if see_gain < 0 {
                continue;
            }

            // Delta pruning: if stand_pat + best possible material swing from this
            // capture (SEE gain) plus a small margin cannot beat alpha, skip.
            if stand_pat + see_gain + DELTA_MARGIN < alpha {
                continue;
            }
        }

        let undo = game.make_move(m);

        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        legal_moves += 1;

        let score = -quiescence(searcher, game, ply + 1, -beta, -alpha);

        game.undo_move(m, undo);

        if searcher.stopped {
            // Swap back move buffer before returning early
            std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);
            return best_score;
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                alpha = score;
            }
        }

        if alpha >= beta {
            break;
        }
    }

    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        if in_check || no_pieces {
            // Swap back move buffer before returning mate score
            std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);
            return -MATE_VALUE + ply as i32;
        }
    }

    // Swap back move buffer for this ply before returning
    std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);

    best_score
}
