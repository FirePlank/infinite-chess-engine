use crate::board::PieceType;
use crate::evaluation::evaluate;
use crate::game::GameState;
use crate::moves::{Move, get_quiescence_captures};

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    // Simple wall-clock timer for wasm; keeps the hot path small and avoids
    // repeated window()/performance() lookups.
    Date::now()
}

pub const MAX_PLY: usize = 64;
pub const INFINITY: i32 = 1_000_000;
pub const MATE_VALUE: i32 = 900_000;
pub const MATE_SCORE: i32 = 800_000;
pub const THINK_TIME_MS: u128 = 3000; // 3 seconds per move

// Maximum absolute value for history scores (used by gravity-style updates)
const MAX_HISTORY: i32 = 4000;

// Null Move Pruning parameters
const NMP_REDUCTION: usize = 3;
const NMP_MIN_DEPTH: usize = 3;

// Late Move Reduction parameters
const LMR_MIN_DEPTH: usize = 3;
const LMR_MIN_MOVES: usize = 4;

// History Leaf Pruning (Fruit-style) parameters
// Only active in non-PV, shallow nodes to keep it conservative but effective.
const HLP_MAX_DEPTH: usize = 3;        // only apply at depth <= 3
const HLP_MIN_MOVES: usize = 5;        // played_nb >= 5
const HLP_HISTORY_REDUCE: i32 = 300;   // history < this gets extra reduction
const HLP_HISTORY_LEAF: i32 = 0;       // and if depth<=0 and history < this, prune move

// Late Move Pruning thresholds per depth (skip quiet moves after this many moves searched)
// Index is depth (0=unused, 1=depth1, etc). At depth d, skip after LMP_THRESHOLD[d] quiet moves.
const LMP_THRESHOLD: [usize; 5] = [0, 4, 8, 12, 16];

// Aspiration window
const ASPIRATION_WINDOW: i32 = 50;

// Futility pruning margins
const FUTILITY_MARGIN: [i32; 4] = [0, 100, 200, 300];

// Internal Iterative Deepening
const IID_MIN_DEPTH: usize = 4;
const IID_REDUCTION: usize = 2;

mod tt;
pub use tt::{TTEntry, TTFlag, TranspositionTable};

mod ordering;
use ordering::{hash_move_dest, sort_captures, sort_moves, sort_moves_root};

mod see;
pub(crate) use see::static_exchange_eval_impl as static_exchange_eval;

pub mod zobrist;
pub use zobrist::{piece_key, special_right_key, en_passant_key, SIDE_KEY};

/// Timer abstraction to handle platform differences
#[derive(Clone)]
pub struct Timer {
    #[cfg(target_arch = "wasm32")]
    start: f64,
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self { start: now_ms() }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self { start: Instant::now() }
        }
    }

    pub fn reset(&mut self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.start = now_ms();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start = Instant::now();
        }
    }

    pub fn elapsed_ms(&self) -> u128 {
        #[cfg(target_arch = "wasm32")]
        {
            (now_ms() - self.start) as u128
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_millis()
        }
    }
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
    
    // PV tracking
    pub pv_table: Vec<Vec<Option<Move>>>,
    pub pv_length: Vec<usize>,
    
    // Killer moves (2 per ply)
    pub killers: Vec<[Option<Move>; 2]>,
    
    // History heuristic [piece_type][to_square_hash]
    pub history: [[i32; 256]; 32],
    
    // Capture history [moving_piece_type][captured_piece_type]
    // Used to improve capture ordering beyond pure MVV-LVA
    pub capture_history: [[i32; 32]; 32],
    
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
}

impl Searcher {
    pub fn new(time_limit_ms: u128) -> Self {
        let mut pv_table = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            let mut row = Vec::with_capacity(MAX_PLY);
            for _ in 0..MAX_PLY {
                row.push(None);
            }
            pv_table.push(row);
        }
        
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
            tt: TranspositionTable::new(32), // 32 MB TT
            pv_table,
            pv_length: vec![0; MAX_PLY],
            killers,
            history: [[0; 256]; 32],
            capture_history: [[0; 32]; 32],
            eval_stack: vec![0; MAX_PLY],
            best_move_root: None,
            prev_score: 0,
            silent: false,
            move_buffers,
        }
    }
    
    pub fn reset_for_iteration(&mut self) {
        // Note: DO NOT reset timer here - we want global time limit across all iterations
        self.nodes = 0;
        self.qnodes = 0;
        self.stopped = false;
        self.seldepth = 0;
        
        // Reset PV table
        for i in 0..MAX_PLY {
            self.pv_length[i] = 0;
            for j in 0..MAX_PLY {
                self.pv_table[i][j] = None;
            }
        }
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
        let mut clamped = bonus;
        if clamped > MAX_HISTORY { clamped = MAX_HISTORY; }
        if clamped < -MAX_HISTORY { clamped = -MAX_HISTORY; }

        let entry = &mut self.history[piece as usize][idx];
        *entry += clamped - *entry * clamped.abs() / MAX_HISTORY;
    }
    
    #[inline]
    pub fn check_time(&mut self) -> bool {
        // Fast-path: no time limit (used by offline test/perft helpers).
        if self.time_limit_ms == u128::MAX {
            return false;
        }

        // Check time only every N nodes to keep the hot path cheap, especially
        // on wasm where elapsed_ms() crosses the JS boundary.
        #[cfg(target_arch = "wasm32")]
        const TIME_CHECK_MASK: u64 = 8191; // every 8192 nodes
        #[cfg(not(target_arch = "wasm32"))]
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
        for i in 0..self.pv_length[0] {
            if let Some(m) = &self.pv_table[0][i] {
                if !pv_str.is_empty() {
                    pv_str.push_str(" ");
                }
                pv_str.push_str(&format!("{},{}->{},{}", m.from.x, m.from.y, m.to.x, m.to.y));
            }
        }
        pv_str
    }
    
    /// Print UCI-style info string
    pub fn print_info(&self, depth: usize, score: i32) {
        let time_ms = self.timer.elapsed_ms();
        let nps = if time_ms > 0 { (self.nodes as u128 * 1000) / time_ms } else { 0 };
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
        
        #[cfg(target_arch = "wasm32")]
        {
             use crate::log;
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
        #[cfg(not(target_arch = "wasm32"))]
        {
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

/// Main entry point - iterative deepening search with aspiration windows.
/// Uses the default THINK_TIME_MS and returns only the best move; the
/// underlying implementation is shared with the timed-with-eval helper.
pub fn get_best_move(game: &mut GameState, max_depth: usize) -> Option<Move> {
    get_best_move_timed_with_eval(game, max_depth, THINK_TIME_MS, false)
        .map(|(m, _)| m)
}

/// Time-limited search that also returns the final root score (cp from side-to-move's perspective).
pub fn get_best_move_timed_with_eval(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    silent: bool,
) -> Option<(Move, i32)> {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();

    let mut searcher = Searcher::new(time_limit_ms);
    searcher.silent = silent;
    
    let moves = game.get_legal_moves();
    if moves.is_empty() {
        return None;
    }
    
    // If only one move, return immediately with a simple static eval as score
    if moves.len() == 1 {
        let single = moves[0].clone();
        let score = evaluate(game);
        return Some((single, score));
    }
    
    // Find first legal move as ultimate fallback (with time check)
    let fallback_move = moves.iter().find(|m| {
        // Time check during legal move validation
        if searcher.timer.elapsed_ms() >= searcher.time_limit_ms {
            return true; // Just accept this move
        }
        let undo = game.make_move(m);
        let legal = !game.is_move_illegal();
        game.undo_move(m, undo);
        legal
    }).cloned();
    
    let mut best_move: Option<Move> = fallback_move.clone();
    let mut best_score = -INFINITY;
    
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
            negamax_root(&mut searcher, game, depth, -INFINITY, INFINITY)
        } else {
            // Aspiration window search
            let mut alpha = searcher.prev_score - ASPIRATION_WINDOW;
            let mut beta = searcher.prev_score + ASPIRATION_WINDOW;
            let mut window_size = ASPIRATION_WINDOW;
            let mut result;
            let mut retries = 0;
            
            loop {
                result = negamax_root(&mut searcher, game, depth, alpha, beta);
                retries += 1;
                
                if searcher.stopped {
                    break;
                }
                
                if result <= alpha {
                    // Failed low - widen alpha
                    window_size *= 4;
                    alpha = searcher.prev_score - window_size;
                } else if result >= beta {
                    // Failed high - widen beta
                    window_size *= 4;
                    beta = searcher.prev_score + window_size;
                } else {
                    // Score within window
                    break;
                }
                
                // Fallback to full window if window gets too large or too many retries
                if window_size > 1000 || retries >= 4 {
                    result = negamax_root(&mut searcher, game, depth, -INFINITY, INFINITY);
                    break;
                }
            }
            result
        };
        
        // Update best move - even if stopped, use best from this iteration if found
        if let Some(pv_move) = &searcher.pv_table[0][0] {
            best_move = Some(pv_move.clone());
            best_score = score;
            searcher.best_move_root = Some(pv_move.clone());
            searcher.prev_score = score;
        }
        
        // Print info after each depth (even if stopped, for debugging)
        if !searcher.stopped && !searcher.silent {
            searcher.print_info(depth, score);
        }
        
        // Check if we found mate or time is up
        if searcher.stopped || best_score.abs() > MATE_SCORE {
            break;
        }
        
        // If we've used more than 80% of time, don't start another iteration
        if searcher.timer.elapsed_ms() > searcher.time_limit_ms * 8 / 10 {
            break;
        }
    }
    
    // Increment TT age for next search
    searcher.tt.increment_age();
    
    if let Some(m) = best_move {
        Some((m, best_score))
    } else {
        None
    }
}

/// Time-limited search entry point. Delegates to the core
/// get_best_move_timed_with_eval implementation and discards the eval.
pub fn get_best_move_timed(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    silent: bool,
) -> Option<Move> {
    get_best_move_timed_with_eval(game, max_depth, time_limit_ms, silent)
        .map(|(m, _)| m)
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
fn negamax_root(searcher: &mut Searcher, game: &mut GameState, depth: usize, mut alpha: i32, beta: i32) -> i32 {
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
        let undo = game.make_move(m);
        
        // Check if move is illegal (leaves our king in check)
        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }
        
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
        
        if searcher.stopped {
            return best_score;
        }
        
        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());
            
            if score > alpha {
                alpha = score;
                
                // Update PV
                searcher.pv_table[0][0] = Some(m.clone());
                searcher.pv_length[0] = searcher.pv_length[1] + 1;
                
                for j in 0..searcher.pv_length[1] {
                    searcher.pv_table[0][j + 1] = searcher.pv_table[1][j].clone();
                }
            }
        }
        
        if alpha >= beta {
            break;
        }
    }
    
    // Checkmate or stalemate (or loss by capture-all-pieces variants)
    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        return if in_check || no_pieces { -MATE_VALUE } else { 0 };
    }
    
    // Swap back move buffer for reuse in future searches
    std::mem::swap(&mut searcher.move_buffers[0], &mut moves);
    
    // Store in TT
    let flag = if best_score >= beta {
        TTFlag::LowerBound
    } else if best_score <= alpha {
        TTFlag::UpperBound
    } else {
        TTFlag::Exact
    };
    searcher.tt.store(hash, depth, flag, best_score, best_move, 0);
    
    best_score
}

/// Main negamax with alpha-beta pruning, NMP, LMR, and TT
fn negamax(searcher: &mut Searcher, game: &mut GameState, depth: usize, ply: usize, 
           mut alpha: i32, mut beta: i32, allow_null: bool) -> i32 {
    searcher.nodes += 1;
    searcher.pv_length[ply] = ply;
    
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
        return 0; // Draw by repetition
    }
    
    // Mate distance pruning (not at root)
    if ply > 0 {
        let mate_score = MATE_VALUE - ply as i32;
        if alpha < -mate_score { alpha = -mate_score; }
        if beta > mate_score - 1 { beta = mate_score - 1; }
        if alpha >= beta { return alpha; }
    }
    
    let in_check = game.is_in_check();
    let is_pv = beta > alpha + 1;
    
    // Base case: quiescence search at leaf nodes
    if depth == 0 {
        return quiescence(searcher, game, ply, alpha, beta);
    }
    
    // Check extension (limited to avoid infinite recursion)
    // Only extend if we're not too deep already
    let depth = if in_check && ply < MAX_PLY / 2 { depth + 1 } else { depth };
    let mut tt_move: Option<Move> = None;
    
    if let Some((score, best)) = searcher.tt.probe(hash, alpha, beta, depth, ply) {
        tt_move = best;
        
        // Use TT cutoff in non-PV nodes
        if !is_pv && score != INFINITY + 1 {
            return score;
        }
    }
    
    // Static evaluation for pruning decisions
    let static_eval = if in_check { -MATE_VALUE + ply as i32 } else { evaluate(game) };
    
    // Store eval for "improving" heuristic
    searcher.eval_stack[ply] = static_eval;
    
    // Check if position is improving (eval better than 2 plies ago)
    // This is used to adjust pruning aggressiveness
    let improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true // Assume improving at root or when in check
    };
    
    // Internal Iterative Deepening - search with reduced depth if no TT move
    if tt_move.is_none() && depth >= IID_MIN_DEPTH && is_pv {
        negamax(searcher, game, depth - IID_REDUCTION, ply, alpha, beta, false);
        if let Some((_, best)) = searcher.tt.probe(hash, alpha, beta, depth, ply) {
            tt_move = best;
        }
    }
    
    // Pruning techniques (not in check, not PV node)
    if !in_check && !is_pv {
        // Reverse Futility Pruning (Static Null Move Pruning)
        if depth < 3 && static_eval - 120 * depth as i32 >= beta {
            return static_eval;
        }
        
        // Null Move Pruning
        if allow_null && depth >= NMP_MIN_DEPTH && static_eval >= beta {
            // Check if we have non-pawn material (avoid zugzwang)
            let has_pieces = game.board.pieces.iter().any(|(_, p)| {
                p.color == game.turn && 
                p.piece_type != PieceType::Pawn && 
                p.piece_type != PieceType::King
            });
            
            if has_pieces {
                // Make null move (proper tracking for repetition detection)
                let saved_ep = game.en_passant.clone();
                game.make_null_move();
                
                let r = NMP_REDUCTION + depth / 6;
                let null_score = -negamax(
                    searcher, game, 
                    depth.saturating_sub(1 + r), 
                    ply + 1, -beta, -beta + 1, false
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
    let mut hash_flag = TTFlag::UpperBound;
    // Track quiet moves searched at this node for history maluses
    let mut quiets_searched: Vec<(PieceType, usize)> = Vec::new();
    
    // Futility pruning flag
    let futility_pruning = !in_check && !is_pv && depth <= 3;
    let futility_base = if futility_pruning {
        static_eval + FUTILITY_MARGIN[depth.min(3)]
    } else {
        0
    };
    
    for m in &moves {
        let captured_piece = game.board.get_piece(&m.to.x, &m.to.y);
        let is_capture = captured_piece.is_some();
        let captured_type = captured_piece.map(|p| p.piece_type);
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
            let threshold = LMP_THRESHOLD[depth.min(4)];
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
            let idx = hash_move_dest(m);
            quiets_searched.push((m.piece.piece_type, idx));
        }

        legal_moves += 1;
        
        let score;
        if legal_moves == 1 {
            // Full window search for first legal move
            score = -negamax(searcher, game, depth - 1, ply + 1, -beta, -alpha, true);
        } else {
            // Late Move Reductions
            let mut reduction = 0;
            if depth >= LMR_MIN_DEPTH && legal_moves >= LMR_MIN_MOVES && !in_check && !is_capture {
                reduction = 1 + (legal_moves as f32).ln() as usize * (depth as f32).ln() as usize / 3;
                
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
            if !in_check && !is_pv && !is_capture && !is_promotion &&
               depth <= HLP_MAX_DEPTH && legal_moves >= HLP_MIN_MOVES &&
               best_score > -MATE_SCORE
            {
                let idx = hash_move_dest(m);
                let value = searcher.history[m.piece.piece_type as usize][idx];

                if value < HLP_HISTORY_REDUCE {
                    // Extra reduction based on poor history
                    new_depth -= 1;

                    // If depth after reductions would drop to quiescence or below
                    // and history is really bad, prune this move entirely.
                    if new_depth <= 0 && value < HLP_HISTORY_LEAF {
                        game.undo_move(m, undo);
                        continue;
                    }
                }
            }

            // Allow new_depth to reach 0 so that the child call will
            // transition to quiescence (depth == 0) instead of being
            // artificially clamped to 1, which can cause very deep
            // "depth 1" trees and huge node counts.
            let search_depth = if new_depth <= 0 { 0 } else { new_depth as usize };

            // Null window search with possible reduction
            let mut s = -negamax(
                searcher, game,
                search_depth,
                ply + 1, -alpha - 1, -alpha, true
            );
            
            // Re-search at full depth if it looks promising
            if s > alpha && (reduction > 0 || s < beta) {
                s = -negamax(searcher, game, depth - 1, ply + 1, -beta, -alpha, true);
            }
            score = s;
        }
        
        game.undo_move(m, undo);
        
        if searcher.stopped {
            return best_score;
        }
        
        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());
            
            if score > alpha {
                alpha = score;
                hash_flag = TTFlag::Exact;
                
                // Update PV
                searcher.pv_table[ply][ply] = Some(m.clone());
                searcher.pv_length[ply] = searcher.pv_length[ply + 1];
                
                for j in (ply + 1)..searcher.pv_length[ply + 1] {
                    searcher.pv_table[ply][j] = searcher.pv_table[ply + 1][j].clone();
                }
            }
        }
        
        if alpha >= beta {
            hash_flag = TTFlag::LowerBound;
            
            if !is_capture {
                // History bonus for quiet cutoff move, with maluses for previously searched quiets
                let idx = hash_move_dest(m);
                let bonus = 300 * depth as i32 - 250;
                searcher.update_history(m.piece.piece_type, idx, bonus);

                for &(ptype, hidx) in &quiets_searched {
                    if ptype == m.piece.piece_type && hidx == idx {
                        continue;
                    }
                    searcher.update_history(ptype, hidx, -bonus);
                }

                // Killer move heuristic (for non-captures)
                searcher.killers[ply][1] = searcher.killers[ply][0].clone();
                searcher.killers[ply][0] = Some(*m);
            } else if let Some(cap_type) = captured_type {
                // Update capture history on beta cutoff
                let bonus = (depth * depth) as i32;
                searcher.capture_history[m.piece.piece_type as usize][cap_type as usize] += bonus;
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
    
    // Store in TT
    searcher.tt.store(hash, depth, hash_flag, best_score, best_move, ply);
    
    best_score
}

/// Quiescence search - only search captures to avoid horizon effect
fn quiescence(searcher: &mut Searcher, game: &mut GameState, ply: usize, mut alpha: i32, beta: i32) -> i32 {
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
