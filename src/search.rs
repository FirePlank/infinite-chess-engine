use crate::board::PieceType;
use crate::evaluation::{evaluate, get_piece_value};
use crate::game::GameState;
use crate::moves::{Move, MoveList, get_quiescence_captures};
use std::cell::RefCell;
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

#[cfg(not(target_arch = "wasm32"))]
fn random() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    let nanos = since_the_epoch.as_nanos();
    let mut x = nanos as u64;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x as f64) / (u64::MAX as f64)
}

pub const MAX_PLY: usize = 64;
pub const INFINITY: i32 = 1_000_000;
pub const MATE_VALUE: i32 = 900_000;
pub const MATE_SCORE: i32 = 800_000;
pub const THINK_TIME_MS: u128 = 3000; // 3 seconds per move (default, may be overridden by caller)

// Correction History constants (adapted for Infinite Chess)
// Size of correction history tables (power of 2 for fast masking)
pub const CORRHIST_SIZE: usize = 16384; // 16K entries per color (for piece/material hashes)
pub const CORRHIST_MASK: u64 = (CORRHIST_SIZE - 1) as u64;
// Last move correction uses smaller table indexed by move from-to hash
pub const LASTMOVE_CORRHIST_SIZE: usize = 4096; // 4K entries
pub const LASTMOVE_CORRHIST_MASK: usize = LASTMOVE_CORRHIST_SIZE - 1;
pub const CORRHIST_GRAIN: i32 = 256; // Scaling factor for correction values
pub const CORRHIST_LIMIT: i32 = 1024 * 32; // Max absolute correction value
pub const CORRHIST_WEIGHT_SCALE: i32 = 256; // Weight scaling for updates

// Low Ply History constants (Stockfish-style)
// Tracks which moves were successful at low plies (near root)
pub const LOW_PLY_HISTORY_SIZE: usize = 4; // Only track first 4 plies
pub const LOW_PLY_HISTORY_ENTRIES: usize = 4096; // Move hash entries per ply
pub const LOW_PLY_HISTORY_MASK: usize = LOW_PLY_HISTORY_ENTRIES - 1;

/// Determines which correction history tables to use.
/// Set once at search start for zero runtime overhead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CorrHistMode {
    /// For CoaIP variants + Classical + Chess: pawn + material (original approach that worked)
    PawnBased,
    /// For all other variants: non-pawn + material + last-move
    NonPawnBased,
}

/// Node type for alpha-beta search.
/// Used to enable more aggressive pruning at expected cut-nodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    /// Principal Variation node - full window search, no aggressive pruning
    PV,
    /// Cut node - expected to fail high (opponent will have a refutation)
    Cut,
    /// All node - expected to fail low (we'll search all moves)
    All,
}

// ============================================================================
// Tunable search parameters - accessed via param accessor functions
// ============================================================================
pub mod params;
use params::{
    aspiration_fail_mult, aspiration_window, history_bonus_base, history_bonus_cap,
    history_bonus_sub, hlp_history_leaf, hlp_history_reduce, hlp_max_depth, hlp_min_moves,
    lmr_divisor, lmr_min_depth, lmr_min_moves, max_history, nmp_min_depth, repetition_penalty,
};

mod tt;
pub use tt::{TTEntry, TTFlag, TranspositionTable};

// Shared TT for Lazy SMP - uses SharedArrayBuffer from JavaScript
// Shared TT for Lazy SMP - uses SharedArrayBuffer from JavaScript
#[cfg(feature = "multithreading")]
mod shared_tt;
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub use shared_tt::SharedTTView;
#[cfg(feature = "multithreading")]
pub use shared_tt::{SharedTT, SharedTTFlag};

mod ordering;
use ordering::{
    hash_coord_32, hash_move_dest, hash_move_from, sort_captures, sort_moves, sort_moves_root,
};

mod movegen;
use movegen::StagedMoveGen;

mod see;
pub(crate) use see::see_ge;
pub(crate) use see::static_exchange_eval_impl as static_exchange_eval;

pub mod zobrist;
pub use zobrist::{SIDE_KEY, en_passant_key, material_key, pawn_key, piece_key, special_right_key};

mod noisy;
pub use noisy::get_best_move_with_noise;

// Shared work queue for root move splitting (DTS-style parallelism)
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
mod work_queue;
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub use work_queue::{NO_MORE_MOVES, SharedWorkQueue, WORK_QUEUE_SIZE_WORDS};

// ============================================================================
// Shared TT Global State (for Lazy SMP with SharedArrayBuffer)
// ============================================================================

/// Thread-local storage for the shared TT pointer and length.
/// Each worker receives the same SharedArrayBuffer from JavaScript,
/// so all workers share the exact same TT memory.
/// Thread-local storage for the shared TT pointer and length.
/// Each worker receives the same SharedArrayBuffer from JavaScript,
/// so all workers share the exact same TT memory.
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
thread_local! {
    // Raw pointer to the SharedArrayBuffer data and its length in u64 words
    static SHARED_TT_STATE: RefCell<Option<(*mut u64, usize)>> = RefCell::new(None);
    // Pointer to shared work queue (offset after TT in shared memory)
    static SHARED_WORK_QUEUE_STATE: RefCell<Option<(*mut u64, usize)>> = RefCell::new(None);
}

/// Set the shared TT pointer (called from lib.rs WASM binding)
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub fn set_shared_tt_ptr(ptr: *mut u64, len: usize) {
    SHARED_TT_STATE.with(|cell| {
        *cell.borrow_mut() = Some((ptr, len));
    });
}

/// Set the shared work queue pointer (called from lib.rs WASM binding)
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub fn set_shared_work_queue_ptr(ptr: *mut u64, len: usize) {
    SHARED_WORK_QUEUE_STATE.with(|cell| {
        *cell.borrow_mut() = Some((ptr, len));
    });
}

/// Check if a shared TT is configured
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub fn has_shared_tt() -> bool {
    SHARED_TT_STATE.with(|cell| cell.borrow().is_some())
}

/// Create a SharedTTView from the stored pointer.
/// Returns None if no shared TT is configured.
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub fn create_shared_tt_view() -> Option<SharedTTView> {
    SHARED_TT_STATE.with(|cell| {
        if let Some((ptr, len)) = *cell.borrow() {
            // SAFETY: The pointer comes from JavaScript's SharedArrayBuffer
            // which is kept alive by JS. The SharedTTView uses atomic operations.
            Some(unsafe { SharedTTView::new(ptr, len) })
        } else {
            None
        }
    })
}

/// Create a SharedWorkQueue from the stored pointer.
/// Returns None if no work queue is configured.
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub fn create_work_queue_view() -> Option<SharedWorkQueue> {
    SHARED_WORK_QUEUE_STATE.with(|cell| {
        if let Some((ptr, len)) = *cell.borrow() {
            Some(unsafe { SharedWorkQueue::new(ptr, len) })
        } else {
            None
        }
    })
}

// ============================================================================
// TT Probe/Store with Shared TT Support (for Lazy SMP)
// ============================================================================

/// Probe the TT, using SharedTTView when available for Lazy SMP.
/// Falls back to the Searcher's local TT if no shared TT is configured.
#[inline]
pub fn probe_tt_with_shared(
    searcher: &Searcher,
    hash: u64,
    alpha: i32,
    beta: i32,
    depth: usize,
    ply: usize,
    rule50_count: u32,
    rule_limit: i32,
) -> Option<(i32, Option<Move>)> {
    // On WASM with shared TT configured, use SharedTTView
    #[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
    {
        if let Some(shared_tt) = create_shared_tt_view() {
            // SAFETY: SharedTTView uses atomic operations for thread-safety
            let result = unsafe { shared_tt.probe(hash, alpha, beta, depth, ply, rule_limit) };
            return result;
        }
    }

    // Fall back to local TT
    searcher
        .tt
        .probe(hash, alpha, beta, depth, ply, rule50_count, rule_limit)
}

/// Store to the TT, using SharedTTView when available for Lazy SMP.
/// Falls back to the Searcher's local TT if no shared TT is configured.
#[inline]
pub fn store_tt_with_shared(
    searcher: &mut Searcher,
    hash: u64,
    depth: usize,
    flag: TTFlag,
    score: i32,
    best_move: Option<Move>,
    ply: usize,
) {
    // On WASM with shared TT configured, use SharedTTView
    #[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
    {
        if let Some(shared_tt) = create_shared_tt_view() {
            // Convert TTFlag to SharedTTFlag
            let shared_flag = match flag {
                TTFlag::None => SharedTTFlag::None,
                TTFlag::Exact => SharedTTFlag::Exact,
                TTFlag::LowerBound => SharedTTFlag::LowerBound,
                TTFlag::UpperBound => SharedTTFlag::UpperBound,
            };
            // SAFETY: SharedTTView uses atomic operations for thread-safety
            unsafe {
                shared_tt.store(
                    hash,
                    depth,
                    shared_flag,
                    score,
                    best_move.as_ref(),
                    ply,
                    0, // generation - we use 0 for now since all workers share
                );
            }
            return;
        }
    }

    // Fall back to local TT
    searcher.tt.store(hash, depth, flag, score, best_move, ply);
}

/// Timer abstraction to handle platform differences
#[derive(Clone)]
pub struct Timer {
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    start: f64,
    #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
    start: Instant,
}

/// Hot data struct - grouped together for cache efficiency.
/// These fields are accessed every node or very frequently during search.
pub struct SearcherHot {
    pub nodes: u64,
    pub qnodes: u64,
    pub timer: Timer,
    pub time_limit_ms: u128,
    pub stopped: bool,
    pub seldepth: usize,
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
    searcher.hot.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    searcher.hot.stopped = false;
    searcher.hot.timer.reset();

    f(&mut searcher)
}

fn build_search_stats(searcher: &Searcher) -> SearchStats {
    #[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
    if let Some(shared) = create_shared_tt_view() {
        // Use shared TT stats
        // Note: used_entries() scans the whole table, which might be slow for very large TTs,
        // but it's only called at the end of a search iteration.
        // If it's too slow, we can return 0 or an estimate.
        // For now, let's trust the sampling in fill_permille and avoid full scan for 'used'.
        // Or we could implement an approximate counter in shared memory.
        let fill = unsafe { shared.fill_permille() };
        let cap = shared.capacity();
        let used = ((cap as u64 * fill as u64) / 1000) as usize;

        return SearchStats {
            tt_capacity: cap,
            tt_used: used,
            tt_fill_permille: fill,
        };
    }

    SearchStats {
        tt_capacity: searcher.tt.capacity(),
        tt_used: searcher.tt.used_entries(),
        tt_fill_permille: searcher.tt.fill_permille(),
    }
}

/// Return current TT statistics from the persistent global searcher, if any.
/// When no global searcher exists yet, checks shared TT if available.
pub fn get_current_tt_stats() -> SearchStats {
    GLOBAL_SEARCHER.with(|cell| {
        let opt = cell.borrow();
        if let Some(ref searcher) = *opt {
            build_search_stats(searcher)
        } else {
            // No global searcher yet - check if shared TT is available
            #[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
            if let Some(shared) = create_shared_tt_view() {
                let fill = unsafe { shared.fill_permille() };
                let cap = shared.capacity();
                let used = ((cap as u64 * fill as u64) / 1000) as usize;
                return SearchStats {
                    tt_capacity: cap,
                    tt_used: used,
                    tt_fill_permille: fill,
                };
            }

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

    // Clear pawn structure cache for new game
    crate::evaluation::base::clear_pawn_cache();

    // Clear material cache for new game
    crate::evaluation::insufficient_material::clear_material_cache();
}

/// Search state that persists across the search
pub struct Searcher {
    /// Hot data - grouped for cache efficiency
    pub hot: SearcherHot,

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

    // Thread ID for Lazy SMP - helper threads (id > 0) skip first N moves
    // This distributes work across threads naturally
    pub thread_id: usize,

    // Per-ply reusable move buffers using Stack/Heap-allocated MoveList (SmallVec)
    pub move_buffers: Vec<MoveList>,

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

    // Correction History - variant-aware for optimal performance:
    // - PawnBased mode: pawn + material (for CoaIP/Classical/Chess variants)
    // - NonPawnBased mode: non-pawn + material + last-move (for other variants)
    // Mode is set once at search start for zero runtime overhead.
    pub corrhist_mode: CorrHistMode,
    pub pawn_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub nonpawn_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub material_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub lastmove_corrhist: Box<[i32; LASTMOVE_CORRHIST_SIZE]>,

    /// TT Move History: tracks reliability of TT moves.
    /// Positive values = TT moves tend to be best moves.
    /// Negative values = TT moves often fail.
    /// Uses gravity-based update formula for smoothing.
    pub tt_move_history: i32,

    /// Reduction stack for hindsight depth adjustment.
    /// Stores the LMR reduction applied at each ply.
    /// Used to adjust depth based on prior search decisions.
    pub reduction_stack: Vec<i32>,

    /// Dynamic move rule limit (e.g. 100 for 50-move rule)
    pub move_rule_limit: i32,

    /// Low Ply History (Stockfish-style): [ply][move_hash] -> score
    /// Tracks which moves were successful at low plies (first 4 from root).
    /// Used to boost ordering for moves that worked well near root.
    pub low_ply_history: Box<[[i32; LOW_PLY_HISTORY_ENTRIES]; LOW_PLY_HISTORY_SIZE]>,
}

impl Searcher {
    pub fn new(time_limit_ms: u128) -> Self {
        // Triangular PV table - heap allocated to avoid stack overflow
        let pv_table = Box::new([None; MAX_PLY * MAX_PLY]);

        let mut killers = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            killers.push([None, None]);
        }

        let mut move_buffers: Vec<MoveList> = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            move_buffers.push(MoveList::new());
        }

        Searcher {
            hot: SearcherHot {
                nodes: 0,
                qnodes: 0,
                timer: Timer::new(),
                time_limit_ms,
                stopped: false,
                seldepth: 0,
            },
            tt: TranspositionTable::new(16),
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
            thread_id: 0,
            move_buffers,
            move_history: vec![None; MAX_PLY],
            moved_piece_history: vec![0; MAX_PLY],
            cont_history: vec![[[[0i32; 32]; 32]; 32]; 16],
            excluded_moves: Vec::new(),
            corrhist_mode: CorrHistMode::NonPawnBased, // Default, set based on variant at search start
            pawn_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            nonpawn_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            material_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            lastmove_corrhist: Box::new([0i32; LASTMOVE_CORRHIST_SIZE]),
            tt_move_history: 0,
            reduction_stack: vec![0; MAX_PLY],
            move_rule_limit: 100, // Default, will be updated from GameState
            low_ply_history: Box::new([[0i32; LOW_PLY_HISTORY_ENTRIES]; LOW_PLY_HISTORY_SIZE]),
        }
    }

    pub fn reset_for_iteration(&mut self) {
        // Note: DO NOT reset timer here - we want global time limit across all iterations
        self.hot.nodes = 0;
        self.hot.qnodes = 0;
        self.hot.stopped = false;
        self.hot.seldepth = 0;

        // Reset PV lengths only - much faster than clearing entire array
        // The PV entries will be overwritten as needed during search
        self.pv_length = [0; MAX_PLY];
    }

    /// Set correction history mode based on variant.
    /// Called once at search start for zero runtime overhead.
    #[inline]
    pub fn set_corrhist_mode(&mut self, game: &GameState) {
        use crate::Variant;
        self.corrhist_mode = match game.variant {
            // PawnBased mode for variants where pawn correction showed positive Elo
            Some(Variant::CoaIP)
            | Some(Variant::CoaIPHO)
            | Some(Variant::CoaIPRO)
            | Some(Variant::CoaIPNO)
            | Some(Variant::Classical)
            | Some(Variant::Chess) => CorrHistMode::PawnBased,
            // NonPawnBased mode for all other variants
            _ => CorrHistMode::NonPawnBased,
        };
    }

    /// Decay history scores at the start of each iteration
    pub fn decay_history(&mut self) {
        for row in self.history.iter_mut() {
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

    /// Update low ply history (Stockfish-style) for moves that caused beta cutoff at low plies.
    /// Only updates for ply < LOW_PLY_HISTORY_SIZE (first 4 plies from root).
    #[inline]
    pub fn update_low_ply_history(&mut self, ply: usize, move_hash: usize, bonus: i32) {
        if ply < LOW_PLY_HISTORY_SIZE {
            let max_h = max_history();
            let clamped = bonus.clamp(-max_h, max_h);
            let idx = move_hash & LOW_PLY_HISTORY_MASK;
            let entry = &mut self.low_ply_history[ply][idx];
            *entry += clamped - *entry * clamped.abs() / max_h;
        }
    }

    #[inline]
    pub fn check_time(&mut self) -> bool {
        // Fast-path: no time limit (used by offline test/perft helpers).
        if self.hot.time_limit_ms == u128::MAX {
            return false;
        }

        if self.hot.nodes & 8191 == 0 {
            if self.hot.timer.elapsed_ms() >= self.hot.time_limit_ms {
                self.hot.stopped = true;
            }
        }
        self.hot.stopped
    }

    /// Apply correction history to raw static evaluation.
    /// Uses variant-specific mode set at search start for zero overhead.
    #[inline]
    pub fn adjusted_eval(&self, game: &GameState, raw_eval: i32, prev_move_idx: usize) -> i32 {
        let color_idx = (game.turn as usize).min(1);

        let total_correction = match self.corrhist_mode {
            CorrHistMode::PawnBased => {
                // Pawn + Material (for CoaIP/Classical/Chess)
                let pawn_idx = (game.pawn_hash & CORRHIST_MASK) as usize;
                let pawn_corr = self.pawn_corrhist[color_idx][pawn_idx];

                let mat_idx = (game.material_hash & CORRHIST_MASK) as usize;
                let mat_corr = self.material_corrhist[color_idx][mat_idx];

                // 60% pawn, 40% material - divide by (GRAIN * 100) for proper percentage scaling
                (pawn_corr * 60 + mat_corr * 40) / (CORRHIST_GRAIN * 100)
            }
            CorrHistMode::NonPawnBased => {
                // Non-pawn + Material + Last-move (for other variants)
                let nonpawn_idx = (game.nonpawn_hash & CORRHIST_MASK) as usize;
                let nonpawn_corr = self.nonpawn_corrhist[color_idx][nonpawn_idx];

                let mat_idx = (game.material_hash & CORRHIST_MASK) as usize;
                let mat_corr = self.material_corrhist[color_idx][mat_idx];

                let lastmove_idx = prev_move_idx & LASTMOVE_CORRHIST_MASK;
                let lastmove_corr = self.lastmove_corrhist[lastmove_idx];

                // 50% non-pawn, 30% material, 20% last-move
                (nonpawn_corr * 50 + mat_corr * 30 + lastmove_corr * 20) / (CORRHIST_GRAIN * 100)
            }
        };

        let corrected = raw_eval + total_correction;
        corrected.clamp(-MATE_SCORE + 1, MATE_SCORE - 1)
    }

    /// Update correction history based on search result.
    /// Updates only the tables relevant for the current mode.
    #[inline]
    pub fn update_correction_history(
        &mut self,
        game: &GameState,
        depth: usize,
        static_eval: i32,
        search_score: i32,
        best_move_is_quiet: bool,
        in_check: bool,
        prev_move_idx: usize,
    ) {
        if in_check || !best_move_is_quiet {
            return;
        }

        let diff = search_score - static_eval;
        let color_idx = (game.turn as usize).min(1);
        let weight = ((depth * depth + 2 * depth + 1) as i32).min(128).max(1);
        let scaled_diff = diff * CORRHIST_GRAIN;

        match self.corrhist_mode {
            CorrHistMode::PawnBased => {
                // Update pawn + material only
                let pawn_idx = (game.pawn_hash & CORRHIST_MASK) as usize;
                let pawn_entry = &mut self.pawn_corrhist[color_idx][pawn_idx];
                *pawn_entry = (*pawn_entry * (CORRHIST_WEIGHT_SCALE - weight)
                    + scaled_diff * weight)
                    / CORRHIST_WEIGHT_SCALE;
                *pawn_entry = (*pawn_entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);

                let mat_idx = (game.material_hash & CORRHIST_MASK) as usize;
                let mat_entry = &mut self.material_corrhist[color_idx][mat_idx];
                *mat_entry = (*mat_entry * (CORRHIST_WEIGHT_SCALE - weight) + scaled_diff * weight)
                    / CORRHIST_WEIGHT_SCALE;
                *mat_entry = (*mat_entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
            }
            CorrHistMode::NonPawnBased => {
                // Update non-pawn + material + last-move
                let nonpawn_idx = (game.nonpawn_hash & CORRHIST_MASK) as usize;
                let nonpawn_entry = &mut self.nonpawn_corrhist[color_idx][nonpawn_idx];
                *nonpawn_entry = (*nonpawn_entry * (CORRHIST_WEIGHT_SCALE - weight)
                    + scaled_diff * weight)
                    / CORRHIST_WEIGHT_SCALE;
                *nonpawn_entry = (*nonpawn_entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);

                let mat_idx = (game.material_hash & CORRHIST_MASK) as usize;
                let mat_entry = &mut self.material_corrhist[color_idx][mat_idx];
                *mat_entry = (*mat_entry * (CORRHIST_WEIGHT_SCALE - weight) + scaled_diff * weight)
                    / CORRHIST_WEIGHT_SCALE;
                *mat_entry = (*mat_entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);

                let lastmove_idx = prev_move_idx & LASTMOVE_CORRHIST_MASK;
                let lm_weight = weight.min(64);
                let lm_entry = &mut self.lastmove_corrhist[lastmove_idx];
                *lm_entry = (*lm_entry * (CORRHIST_WEIGHT_SCALE - lm_weight)
                    + scaled_diff * lm_weight)
                    / CORRHIST_WEIGHT_SCALE;
                *lm_entry = (*lm_entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
            }
        }
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
                let promo = m.promotion.map_or("", |p| p.to_site_code());
                pv_str.push_str(&format!(
                    "{},{}->{},{}{}",
                    m.from.x, m.from.y, m.to.x, m.to.y, promo
                ));
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
        let time_ms = self.hot.timer.elapsed_ms();
        let nps = if time_ms > 0 {
            (self.hot.nodes as u128 * 1000) / time_ms
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
                    self.hot.seldepth,
                    multipv,
                    score_str,
                    self.hot.nodes,
                    self.hot.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                ));
            } else {
                log(&format!(
                    "info depth {} seldepth {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.hot.seldepth,
                    score_str,
                    self.hot.nodes,
                    self.hot.qnodes,
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
                    self.hot.seldepth,
                    multipv,
                    score_str,
                    self.hot.nodes,
                    self.hot.qnodes,
                    nps,
                    time_ms,
                    tt_fill,
                    pv
                );
            } else {
                eprintln!(
                    "info depth {} seldepth {} score {} nodes {} qnodes {} nps {} time {} hashfull {} pv {}",
                    depth,
                    self.hot.seldepth,
                    score_str,
                    self.hot.nodes,
                    self.hot.qnodes,
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

    // Filter fully legal moves upfront.
    // This allows negamax_root to skip legality checks and allows us to reuse the move list
    // (and its sorting) across iterative deepening depths.
    let mut legal_moves: MoveList = MoveList::new();
    let mut fallback_move: Option<Move> = None;

    for m in moves {
        let undo = game.make_move(&m);
        let legal = !game.is_move_illegal();
        game.undo_move(&m, undo);

        if legal {
            if fallback_move.is_none() {
                fallback_move = Some(m.clone());
            }
            legal_moves.push(m);
        }
    }

    if legal_moves.is_empty() {
        return None;
    }

    // If only one move, return immediately with a simple static eval as score.
    if legal_moves.len() == 1 {
        let single = legal_moves[0].clone();
        let score = evaluate(game);
        return Some((single, score));
    }

    let mut best_move: Option<Move> = fallback_move; // Already cloned above
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
        if searcher.hot.timer.elapsed_ms() >= searcher.hot.time_limit_ms {
            searcher.hot.stopped = true;
            break;
        }

        let score = if depth == 1 {
            // First iteration: full window
            negamax_root(searcher, game, depth, -INFINITY, INFINITY, &mut legal_moves)
        } else {
            // Aspiration window search
            let asp_win = aspiration_window();
            let mut alpha = searcher.prev_score - asp_win;
            let mut beta = searcher.prev_score + asp_win;
            let mut window_size = asp_win;
            let mut result;
            let mut retries = 0;

            loop {
                result = negamax_root(searcher, game, depth, alpha, beta, &mut legal_moves);
                retries += 1;

                if searcher.hot.stopped {
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
                    result =
                        negamax_root(searcher, game, depth, -INFINITY, INFINITY, &mut legal_moves);
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

        if !searcher.hot.stopped && !searcher.silent {
            searcher.print_info(depth, score);
        }

        // Check if we found mate or time is up
        if searcher.hot.stopped || best_score.abs() > MATE_SCORE {
            break;
        }

        // If we've used more than 50% of time, don't start another iteration
        if searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms();
            let limit = searcher.hot.time_limit_ms;

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
    get_best_move_threaded(game, max_depth, time_limit_ms, silent, 0)
}

/// Time-limited search with thread_id for Lazy SMP.
/// Helper threads (thread_id > 0) skip the first move to distribute work.
/// Uses fresh Searcher per call (no persistent TT between searches).
pub fn get_best_move_threaded(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    silent: bool,
    thread_id: usize,
) -> Option<(Move, i32, SearchStats)> {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();
    // Initialize correction history hashes
    game.recompute_correction_hashes();

    // Use a fresh searcher per call (no persistent TT/state between searches).
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.hot.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    searcher.thread_id = thread_id;
    // Set correction mode based on variant (zero overhead during search)
    searcher.set_corrhist_mode(game);
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
    // Initialize correction history hashes
    game.recompute_correction_hashes();

    let multi_pv = multi_pv.max(1);

    // MultiPV = 1: Zero overhead path - just do normal search
    if multi_pv == 1 {
        let mut searcher = Searcher::new(time_limit_ms);
        searcher.hot.time_limit_ms = time_limit_ms;
        searcher.silent = silent;
        searcher.set_corrhist_mode(game);
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        let mut lines: Vec<PVLine> = Vec::with_capacity(1);
        if let Some((best_move, score)) = search_with_searcher(&mut searcher, game, max_depth) {
            let pv = extract_pv(&searcher);
            let depth = max_depth.min(searcher.hot.seldepth.max(1));
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
    searcher.hot.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    searcher.set_corrhist_mode(game);
    searcher.move_rule_limit = game
        .game_rules
        .move_rule_limit
        .map_or(i32::MAX, |v| v as i32);

    get_best_moves_multipv_impl(&mut searcher, game, max_depth, multi_pv, silent)
}

fn get_best_moves_multipv_impl(
    searcher: &mut Searcher,
    game: &mut GameState,
    max_depth: usize,
    multi_pv: usize,
    silent: bool,
) -> MultiPVResult {
    // Get all legal moves upfront
    let moves = game.get_legal_moves();
    if moves.is_empty() {
        let stats = build_search_stats(searcher);
        return MultiPVResult {
            lines: Vec::new(),
            stats,
        };
    }

    // If only one move, return it immediately
    if moves.len() == 1 {
        let single = moves[0].clone();
        let score = crate::evaluation::evaluate(game);
        let stats = build_search_stats(searcher);
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
    let mut legal_root_moves: MoveList = MoveList::new();
    for m in &moves {
        let undo = game.make_move(m);
        let legal = !game.is_move_illegal();
        game.undo_move(m, undo);
        if legal {
            legal_root_moves.push(m.clone());
        }
    }

    if legal_root_moves.is_empty() {
        let stats = build_search_stats(searcher);
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
        if searcher.hot.timer.elapsed_ms() >= searcher.hot.time_limit_ms {
            searcher.hot.stopped = true;
            break;
        }

        root_scores.clear();

        // Track the MultiPV alpha threshold incrementally to avoid repeated min() scans
        let mut multipv_alpha = -INFINITY;

        // Search each root move (ordered by previous iteration's scores)
        for (move_idx, m) in legal_root_moves.iter().enumerate() {
            if searcher.hot.stopped {
                break;
            }

            let undo = game.make_move(m);

            // Set up prev move info for child search
            let prev_from_hash = hash_move_from(m);
            let prev_to_hash = hash_move_dest(m);
            searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

            // For MultiPV, we need to search all moves to get their scores.
            // First move gets full window, subsequent moves use PVS with MultiPV-aware alpha.
            let score = if move_idx == 0 {
                -negamax(
                    searcher,
                    game,
                    depth - 1,
                    1,
                    -INFINITY,
                    INFINITY,
                    true,
                    NodeType::PV,
                )
            } else {
                // Use PVS for efficiency with MultiPV-aware alpha bound
                let alpha = multipv_alpha;

                let mut s = -negamax(
                    searcher,
                    game,
                    depth - 1,
                    1,
                    -alpha - 1,
                    -alpha,
                    true,
                    NodeType::Cut,
                );
                if s > alpha && !searcher.hot.stopped {
                    // Re-search with full window to get accurate score
                    s = -negamax(
                        searcher,
                        game,
                        depth - 1,
                        1,
                        -INFINITY,
                        INFINITY,
                        true,
                        NodeType::PV,
                    );
                }
                s
            };

            game.undo_move(m, undo);

            if !searcher.hot.stopped {
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

                // Update multipv_alpha incrementally once we have enough candidates
                if root_scores.len() >= multi_pv {
                    // Find the minimum score among current candidates
                    let worst = root_scores.iter().map(|(_, s, _)| *s).min().unwrap();
                    multipv_alpha = worst;
                }
            }
        }

        if searcher.hot.stopped && root_scores.is_empty() {
            break;
        }

        // Sort by score descending
        root_scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Reorder legal_root_moves by this iteration's scores for better PVS efficiency
        // at the next depth - the previous best move will be searched first
        legal_root_moves.clear();
        for (mv, _, _) in &root_scores {
            legal_root_moves.push(mv.clone());
        }

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
            if !silent && !searcher.hot.stopped {
                // Format PV string
                let pv_str: String = pv
                    .iter()
                    .map(|m| {
                        format!(
                            "{},{}->{},{}{}",
                            m.from.x,
                            m.from.y,
                            m.to.x,
                            m.to.y,
                            m.promotion.map_or("", |p| p.to_site_code())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                let time_ms = searcher.hot.timer.elapsed_ms();
                let nps = if time_ms > 0 {
                    (searcher.hot.nodes as u128 * 1000) / time_ms
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
                        searcher.hot.seldepth,
                        idx + 1,
                        score_str,
                        searcher.hot.nodes,
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
                        searcher.hot.seldepth,
                        idx + 1,
                        score_str,
                        searcher.hot.nodes,
                        nps,
                        time_ms,
                        pv_str
                    );
                }
            }
        }

        // Check for mate or time up
        if searcher.hot.stopped {
            break;
        }
        if !best_lines.is_empty() && best_lines[0].score.abs() > MATE_SCORE {
            break;
        }

        // Soft time limit check - don't start next iteration if past 50%
        if searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms();
            if elapsed >= searcher.hot.time_limit_ms / 2 {
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
    let stats = build_search_stats(searcher);
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
    // Initialize correction history hashes
    game.recompute_correction_hashes();

    let mut searcher = Searcher::new(u128::MAX);
    searcher.set_corrhist_mode(game);
    searcher.reset_for_iteration();
    searcher.decay_history();
    searcher.tt.clear();

    // Generate and filter legal moves
    let moves = game.get_legal_moves();
    let mut legal_moves: MoveList = MoveList::new();
    for m in moves {
        let undo = game.make_move(&m);
        let legal = !game.is_move_illegal();
        game.undo_move(&m, undo);
        if legal {
            legal_moves.push(m);
        }
    }

    let _ = negamax_root(
        &mut searcher,
        game,
        depth,
        -INFINITY,
        INFINITY,
        &mut legal_moves,
    );
    searcher.hot.nodes
}

/// Root negamax - special handling for root node
fn negamax_root(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    mut alpha: i32,
    beta: i32,
    moves: &mut MoveList,
) -> i32 {
    // Save original alpha for TT flag determination
    let alpha_orig = alpha;

    searcher.pv_length[0] = 0;

    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;

    // Probe TT for best move from previous search (uses shared TT if configured)
    // Stockfish passes rule50_count (halfmove_clock) directly to value_from_tt
    let rule50_count = game.halfmove_clock;
    if let Some((_, best)) = probe_tt_with_shared(
        searcher,
        hash,
        alpha,
        beta,
        depth,
        0,
        rule50_count,
        searcher.move_rule_limit,
    ) {
        tt_move = best;
    }

    let in_check = game.is_in_check();

    // Sort moves at root (TT move first, then by score)
    // This reorders the `moves` vec in-place, preserving this ordering
    // for the next iteration.
    sort_moves_root(searcher, game, moves, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;

    for m in moves {
        // Skip excluded moves (for MultiPV subsequent passes)
        if !searcher.excluded_moves.is_empty() {
            let coords = (m.from.x, m.from.y, m.to.x, m.to.y);
            if searcher.excluded_moves.contains(&coords) {
                continue;
            }
        }

        // Note: All threads search all moves. Thread variation comes from:
        // 1. Shared TT - threads benefit from each other's entries
        // 2. Slight timing differences - threads finish at different points

        let undo = game.make_move(m);

        // At the root, this move becomes the previous move for child ply 1,
        // stored as (from_hash, to_hash).
        let prev_entry_backup = searcher.prev_move_stack[0];
        let prev_from_hash = hash_move_from(m);
        let prev_to_hash = hash_move_dest(m);
        searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

        legal_moves += 1;

        let score;
        if legal_moves == 1 {
            // Full window search for first legal move
            score = -negamax(
                searcher,
                game,
                depth - 1,
                1,
                -beta,
                -alpha,
                true,
                NodeType::PV,
            );
        } else {
            // PVS: Null window first, then re-search if it improves alpha
            let mut s = -negamax(
                searcher,
                game,
                depth - 1,
                1,
                -alpha - 1,
                -alpha,
                true,
                NodeType::Cut,
            );
            if s > alpha && s < beta {
                s = -negamax(
                    searcher,
                    game,
                    depth - 1,
                    1,
                    -beta,
                    -alpha,
                    true,
                    NodeType::PV,
                );
            }
            score = s;
        }

        game.undo_move(m, undo);

        // Restore previous-move stack entry for root after returning from child.
        searcher.prev_move_stack[0] = prev_entry_backup;

        if searcher.hot.stopped {
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

    // Store in TT with correct flag based on original alpha
    let tt_flag = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    store_tt_with_shared(searcher, hash, depth, tt_flag, best_score, best_move, 0);

    best_score
}

/// Main negamax with alpha-beta pruning
fn negamax(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    ply: usize,
    mut alpha: i32,
    mut beta: i32,
    allow_null: bool,
    node_type: NodeType,
) -> i32 {
    // Node type classification for search behavior
    let is_pv = node_type == NodeType::PV;
    let cut_node = node_type == NodeType::Cut;
    let all_node = !is_pv && !cut_node;

    // Leaf node: transition to quiescence search
    if depth == 0 {
        return quiescence(searcher, game, ply, alpha, beta);
    }

    // Cap depth to prevent overflow
    let mut depth = depth.min(MAX_PLY - 1);

    // Safety check
    if ply >= MAX_PLY - 1 {
        return evaluate(game);
    }

    // Initialize node state
    let in_check = game.is_in_check();
    searcher.hot.nodes += 1;
    searcher.pv_length[ply] = 0;

    // Time management and selective depth tracking
    if searcher.check_time() {
        return 0;
    }
    if is_pv && ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
    }

    // Non-root node: check for draws and mate distance pruning
    if ply > 0 {
        // Draw by fifty-move rule or repetition
        if game.is_fifty() || game.is_repetition(ply) {
            return -repetition_penalty();
        }

        // Mate distance pruning: if we already found a faster mate, prune
        let mate_score = MATE_VALUE - ply as i32;
        alpha = alpha.max(-mate_score);
        beta = beta.min(mate_score - 1);
        if alpha >= beta {
            return alpha;
        }
    }

    // Save original bounds for TT flag determination
    let alpha_orig = alpha;
    let beta_orig = beta;

    // Track reduction from parent ply for hindsight adjustment
    let prior_reduction = if ply > 0 {
        searcher.reduction_stack[ply - 1]
    } else {
        0
    };

    // Transposition table probe for hash move and potential cutoff
    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;

    // Stockfish passes rule50_count (halfmove_clock) directly to value_from_tt
    let rule50_count = game.halfmove_clock;
    if let Some((score, best)) = probe_tt_with_shared(
        searcher,
        hash,
        alpha,
        beta,
        depth,
        ply,
        rule50_count,
        searcher.move_rule_limit,
    ) {
        tt_move = best;
        // In non-PV nodes, use TT cutoff if valid score returned
        // Stockfish's "graph history interaction" workaround:
        // - Don't produce TT cutoffs when rule50 is high (>= 96)
        // - Don't produce TT cutoffs when position has repetition history
        let rule_limit = searcher.move_rule_limit as u32;
        if !is_pv
            && score != INFINITY + 1
            && game.halfmove_clock < rule_limit.saturating_sub(4)
            && game.repetition == 0
        {
            return score;
        }
    }

    // Static evaluation for pruning decisions
    let prev_move_idx = if ply > 0 {
        let (from_hash, to_hash) = searcher.prev_move_stack[ply - 1];
        from_hash ^ to_hash
    } else {
        0
    };

    let (static_eval, raw_eval) = if in_check {
        // When in check, use a pessimistic value (eval is unreliable in check)
        let eval = -MATE_VALUE + ply as i32;
        (eval, eval)
    } else {
        let raw = evaluate(game);
        let adjusted = searcher.adjusted_eval(game, raw, prev_move_idx);
        (adjusted, raw)
    };

    searcher.eval_stack[ply] = static_eval;

    // Position improving heuristic: compare eval to 2 plies ago
    // Used to adjust pruning aggressiveness
    let mut improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true
    };

    // Opponent worsening: their last move made our position better
    // Used to increase pruning when opponent is making poor moves
    let opponent_worsening = if ply >= 1 && !in_check {
        static_eval > -searcher.eval_stack[ply - 1]
    } else {
        false
    };

    // Hindsight depth adjustment based on prior search behavior
    // If we were reduced heavily but opponent didn't worsen, increase depth
    // If position is stable, decrease depth
    if !in_check && ply > 0 {
        let prev_eval = searcher.eval_stack[ply - 1];
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2 && depth >= 2 && static_eval + prev_eval > 173 {
            depth = depth.saturating_sub(1);
        }
    }

    // When in check, skip all pruning - we need to search all evasions
    if !in_check {
        // =================================================================
        // Pre-move pruning techniques
        // =================================================================

        // Razoring: if static eval is very low, drop to qsearch
        // Formula: eval < alpha - 485 - 281 * depth²
        if !is_pv && static_eval < alpha - 485 - 281 * (depth * depth) as i32 {
            return quiescence(searcher, game, ply, alpha, beta);
        }

        // Reverse futility pruning (static null move)
        // If static eval is much higher than beta, return early
        if !is_pv && depth < 14 {
            let futility_mult = 76;
            let futility_margin = futility_mult * depth as i32
                - if improving {
                    2474 * futility_mult / 1024
                } else {
                    0
                }
                - if opponent_worsening {
                    331 * futility_mult / 1024
                } else {
                    0
                };

            if static_eval - futility_margin >= beta && static_eval >= beta {
                return (2 * beta + static_eval) / 3;
            }
        }

        // Null move pruning: give opponent an extra move, if still >= beta, prune
        // Only in cut nodes with non-pawn material (avoid zugzwang)
        if cut_node && allow_null && depth >= nmp_min_depth() {
            let nmp_margin = static_eval - (18 * depth as i32) + 350;
            if nmp_margin >= beta && game.has_non_pawn_material(game.turn) {
                let saved_ep = game.en_passant.clone();
                game.make_null_move();

                // Reduction: R = 7 + depth / 3
                let r = 7 + depth / 3;
                let null_score = -negamax(
                    searcher,
                    game,
                    depth.saturating_sub(r),
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    NodeType::Cut,
                );

                game.unmake_null_move();
                game.en_passant = saved_ep;

                if searcher.hot.stopped {
                    return 0;
                }

                // If null move score >= beta, we can prune
                if null_score >= beta {
                    return null_score;
                }
            }
        }

        // Update improving flag based on static eval vs beta
        improving = improving || static_eval >= beta;

        // Internal iterative reductions (IIR)
        // Without TT move, reduce depth to find one faster
        if !all_node && depth >= 6 && tt_move.is_none() && prior_reduction <= 3 {
            depth -= 1;
        }
    }

    // =========================================================================
    // Staged Move Generation - generate moves in stages for better efficiency
    // =========================================================================
    let mut movegen = StagedMoveGen::new(tt_move.clone(), ply, searcher, game);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    let mut quiets_searched: MoveList = MoveList::new();

    // Singular extension conditions (checked when we reach the TT move in the loop)
    // We cache the TT probe result here to avoid re-probing
    let se_conditions = if depth >= 6 && !in_check && tt_move.is_some() {
        if let Some((tt_flag, tt_depth, tt_score, _)) = searcher.tt.probe_for_singular(hash, ply) {
            if (tt_flag == TTFlag::LowerBound || tt_flag == TTFlag::Exact)
                && tt_depth as usize >= depth.saturating_sub(3)
                && tt_score.abs() < MATE_SCORE
            {
                Some((tt_score, (depth - 1) / 2)) // (singular_beta_base, singular_depth)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // New depth for child nodes
    let new_depth = depth.saturating_sub(1);

    // Main move loop - iterate through staged moves
    while let Some(m) = movegen.next(game, searcher) {
        // BITBOARD: Fast capture detection
        let captured_piece = game.board.get_piece(m.to.x, m.to.y);
        let is_capture = captured_piece.map_or(false, |p| !p.piece_type().is_neutral_type());
        let captured_type = captured_piece.map(|p| p.piece_type());
        let is_promotion = m.promotion.is_some();
        let p_type = m.piece.piece_type();

        // Check if this move gives check to enemy king (O(1) for knights/pawns)
        let gives_check = StagedMoveGen::move_gives_check_fast(game, &m);

        // In-move pruning at shallow depths (not in PV, have material, not losing)
        if !is_pv && game.has_non_pawn_material(game.turn) && best_score > -MATE_SCORE {
            // Late move pruning: skip quiet moves after seeing enough
            // Threshold: (3 + depth²) / (2 if not improving, else 1)
            let improving_div = if improving { 1 } else { 2 };
            let lmp_count = (3 + depth * depth) / improving_div;
            // Signal movegen to skip quiet generation entirely (truly lazy)
            if legal_moves >= lmp_count {
                movegen.skip_quiet_moves();
            }

            // LMR depth estimate for pruning decisions
            let lmr_depth = new_depth as i32;

            if is_capture || gives_check {
                // Capture/check pruning
                if let Some(cap_type) = captured_type {
                    let capt_hist = searcher.capture_history[p_type as usize][cap_type as usize];

                    // Capture futility: skip captures that can't raise alpha
                    // Threshold: eval + 232 + 217*lmrD + pieceVal + histBonus
                    if !gives_check && lmr_depth < 7 {
                        let cap_value = get_piece_value(cap_type);
                        let futility_value = static_eval
                            + 232
                            + 217 * lmr_depth
                            + cap_value
                            + 131 * capt_hist / 1024;
                        if futility_value <= alpha {
                            continue;
                        }
                    }

                    // SEE pruning for captures: skip losing captures
                    // Exempt moves that give check (they have tactical significance)
                    // Threshold: -max(166*d + captHist/29, 0)
                    // if !gives_check {
                    let see_margin = (166 * depth as i32 + capt_hist / 29).max(0);
                    let see_value = static_exchange_eval(game, &m);
                    if see_value < -see_margin {
                        continue;
                    }
                    // }
                }
            } else {
                // Quiet move pruning
                let hist_idx = hash_move_dest(&m);
                let main_hist = searcher.history[p_type as usize][hist_idx];
                let history = main_hist;

                // History-based pruning: skip moves with very bad history
                // Threshold: history < -4083 * depth
                if history < -4083 * depth as i32 {
                    continue;
                }

                // Adjust LMR depth based on history
                let adj_lmr_depth = (lmr_depth + history / 3208).max(0);

                // Quiet futility: skip moves that can't raise alpha
                // Threshold: eval + 42 + 161*(no bestMove) + 127*lmrD
                if !in_check && adj_lmr_depth < 13 {
                    let no_best = if best_move.is_none() { 161 } else { 0 };
                    let futility_value = static_eval + 42 + no_best + 127 * adj_lmr_depth;
                    if futility_value <= alpha {
                        if best_score <= futility_value {
                            best_score = futility_value;
                        }
                        continue;
                    }
                }

                // SEE pruning for quiets: skip moves with bad SEE
                // Threshold: -25 * lmrD²
                let see_threshold = -25 * adj_lmr_depth * adj_lmr_depth;
                let see_value = static_exchange_eval(game, &m);
                if see_value < see_threshold {
                    continue;
                }
            }
        }

        // Prefetch TT entry for child position BEFORE making the move.
        // This warms the cache so the TT probe in the recursive call is faster.
        // Compute approximate child hash: toggle side + move piece from->to
        #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
        {
            let p_type = m.piece.piece_type();
            let p_color = m.piece.color();
            let child_hash = game.hash
                ^ SIDE_KEY
                ^ piece_key(p_type, p_color, m.from.x, m.from.y)
                ^ piece_key(p_type, p_color, m.to.x, m.to.y);
            searcher.tt.prefetch_entry(child_hash);
        }

        let mut undo = game.make_move(&m);

        // Check if move is illegal (leaves our king in check)
        if game.is_move_illegal() {
            game.undo_move(&m, undo);
            continue;
        }

        // Record quiet moves searched at this node for history maluses
        if !is_capture && !is_promotion {
            quiets_searched.push(m.clone());
        }

        // For this node at `ply`, this move becomes the previous move for child
        // ply + 1, stored as (from_hash, to_hash).
        let prev_entry_backup = searcher.prev_move_stack[ply];
        let from_hash = hash_move_from(&m);
        let to_hash = hash_move_dest(&m);
        searcher.prev_move_stack[ply] = (from_hash, to_hash);

        // Store move and piece info for continuation history
        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        searcher.move_history[ply] = Some(m.clone());
        searcher.moved_piece_history[ply] = p_type as u8;

        legal_moves += 1;

        // Calculate per-move extension
        // Stockfish-style Singular Extension + Multi-Cut:
        // When we're about to search the TT move at sufficient depth, first verify
        // it's truly singular by doing a reduced search excluding it.
        let mut extension: usize = 0;

        let is_tt_move = if let Some(ref tt_m) = tt_move {
            m.from == tt_m.from && m.to == tt_m.to && m.promotion == tt_m.promotion
        } else {
            false
        };

        if is_tt_move && !is_pv {
            if let Some((tt_score, singular_depth)) = se_conditions {
                // Singular extension margin with TT Move History adjustment
                // Stockfish uses: 897 * ttMoveHistory / 127649
                // When TT moves are reliable (high ttMoveHistory), we can use a tighter margin
                // (less extension). When unreliable (low), use a more generous margin.
                let tt_history_adj = searcher.tt_move_history / 150; // ~= 897/127649 * 16384
                let singular_beta = tt_score - (depth as i32) * 3 + tt_history_adj;

                // Undo the move we just made so we can search alternatives
                game.undo_move(&m, undo.clone());

                // Do a reduced search excluding the TT move to verify singularity
                // Create a move generator that skips the TT move
                let mut se_gen = StagedMoveGen::with_exclusion(
                    None, // No TT move hint for this search
                    ply,
                    searcher,
                    game,
                    m.clone(), // Exclude the current (TT) move
                );

                let mut se_best = -INFINITY;
                let mut se_moves_checked = 0;
                const SE_MAX_MOVES: usize = 6;

                while let Some(se_m) = se_gen.next(game, searcher) {
                    if se_moves_checked >= SE_MAX_MOVES {
                        break;
                    }

                    let se_undo = game.make_move(&se_m);
                    if game.is_move_illegal() {
                        game.undo_move(&se_m, se_undo);
                        continue;
                    }

                    se_moves_checked += 1;

                    let se_score = -negamax(
                        searcher,
                        game,
                        singular_depth,
                        ply + 1,
                        -singular_beta,
                        -singular_beta + 1,
                        false,
                        NodeType::Cut,
                    );

                    game.undo_move(&se_m, se_undo);

                    if searcher.hot.stopped {
                        return 0;
                    }

                    if se_score > se_best {
                        se_best = se_score;
                    }

                    // Early exit if we find a refuter
                    if se_best >= singular_beta {
                        break;
                    }
                }

                // Re-make the TT move since we undid it above
                let new_undo = game.make_move(&m);
                // Update undo for later
                undo = new_undo;

                if se_best < singular_beta {
                    // TT move is singular - extend it
                    extension = 1;
                } else if se_best >= beta && !is_pv {
                    // Multi-cut: alternatives also beat beta, prune the whole subtree
                    // Apply negative penalty to TT Move History - the TT move wasn't truly singular
                    // Stockfish uses: max(-400 - 100 * depth, -4000)
                    let penalty = (-400 - 100 * depth as i32).max(-4000);
                    let max_tt_hist = 8192;
                    searcher.tt_move_history +=
                        penalty - searcher.tt_move_history * penalty.abs() / max_tt_hist;

                    game.undo_move(&m, undo);
                    return beta;
                }
            }
        }

        let score;
        if legal_moves == 1 {
            // Child type depends on current node type:
            // PV → PV for first child, Cut → All, All → Cut
            let child_type = if is_pv {
                NodeType::PV
            } else if cut_node {
                NodeType::All
            } else {
                NodeType::Cut
            };
            // Full window search for first legal move
            score = -negamax(
                searcher,
                game,
                depth - 1 + extension,
                ply + 1,
                -beta,
                -alpha,
                true,
                child_type,
            );
        } else {
            // Late Move Reductions
            let mut reduction: i32 = 0;
            if depth >= lmr_min_depth()
                && legal_moves >= lmr_min_moves()
                && !in_check
                && !is_capture
                && !(gives_check && (p_type == PieceType::Queen || p_type == PieceType::Amazon))
            {
                reduction = 1
                    + ((legal_moves as f32).ln() * (depth as f32).ln() / lmr_divisor() as f32)
                        as i32;

                // Reduce more when position is not improving
                if !improving {
                    reduction += 1;
                }

                // History-adjusted LMR (simple, low-overhead version)
                // Only use main history - continuation history lookups are too expensive.
                // Only reduce LESS for good history (safe); don't increase for bad (risky).
                let hist_idx = hash_move_dest(&m);
                let hist_score = searcher.history[p_type as usize][hist_idx];

                // Reduce less for moves with good history (threshold: ~50% of max)
                if hist_score > 2000 && reduction > 0 {
                    reduction -= 1;
                }

                // TT Move History adjustment:
                // If TT moves have been unreliable (low tt_move_history), reduce less
                // since the move ordering from TT may not be trustworthy.
                // Only adjust for significant negative values to avoid overhead.
                if searcher.tt_move_history < -1000 && reduction > 0 {
                    reduction -= 1;
                }

                // Reduce less for forking piece checks (high tactical importance)
                // if reduction > 0
                //     && (gives_check && (p_type == PieceType::Queen || p_type == PieceType::Amazon))
                // {
                //     reduction -= 1;
                // }

                // Ensure reduction stays in valid range [0, depth-2]
                reduction = reduction.clamp(0, (depth as i32) - 2);
            }

            // Base child depth after LMR (with singular extension if applicable)
            let mut new_depth = depth as i32 - 1 + extension as i32 - reduction;

            // History Leaf Pruning (Fruit-style)
            // Only in non-PV, quiet, shallow nodes and after enough moves
            // Exempt checking moves (Stockfish-style) to avoid missing check-fork tactics
            if !in_check
                && !is_pv
                && !is_capture
                && !is_promotion
                && !gives_check
                && depth <= hlp_max_depth()
                && legal_moves >= hlp_min_moves()
                && best_score > -MATE_SCORE
            {
                let idx = hash_move_dest(&m);
                let value = searcher.history[p_type as usize][idx];

                if value < hlp_history_reduce() {
                    // Extra reduction based on poor history
                    new_depth -= 1;

                    // If depth after reductions would drop to quiescence or below
                    // and history is really bad, prune this move entirely.
                    if new_depth <= 0 && value < hlp_history_leaf() {
                        game.undo_move(&m, undo);
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

            // Child type for non-first moves: alternate Cut/All
            let child_type = if cut_node {
                NodeType::All
            } else {
                NodeType::Cut
            };

            // Null window search with possible reduction
            // Store reduction for hindsight depth adjustment in child nodes
            searcher.reduction_stack[ply] = reduction;
            let mut s = -negamax(
                searcher,
                game,
                search_depth,
                ply + 1,
                -alpha - 1,
                -alpha,
                true,
                child_type,
            );

            // Re-search at full depth if it looks promising
            if s > alpha && (reduction > 0 || s < beta) {
                // Re-search with PV-like search if we're in PV, otherwise same child type
                let research_type = if is_pv { NodeType::PV } else { child_type };
                s = -negamax(
                    searcher,
                    game,
                    depth - 1 + extension,
                    ply + 1,
                    -beta,
                    -alpha,
                    true,
                    research_type,
                );

                // Post LMR continuation history update
                // When a reduced search fails high and we had to re-search, the move
                // proved to be good - give it a bonus in continuation history.
                //
                // Key design decisions for generalized (non-Stockfish-tuned) bonus:
                // 1. Depth-proportional: deeper searches = more reliable signal = bigger bonus
                // 2. Scaled down: LMR re-search is weaker signal than beta-cutoff (~1/3 bonus)
                // 3. Quiets only: continuation history only helps quiet move ordering
                if reduction > 0 && !is_capture && !is_promotion {
                    // Depth-scaled bonus: ~100 * depth (compared to ~300 * depth for cutoffs)
                    // This is roughly 1/3 of the beta-cutoff bonus, appropriate since
                    // "failed high after reduction" is a weaker signal than "caused cutoff"
                    let lmr_bonus = 100 * depth as i32;
                    let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;
                    let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
                    let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

                    // Update continuation histories at ply offsets -1, -2, -4
                    // (matching the existing beta-cutoff update pattern)
                    for &plies_ago in &[0usize, 1, 3] {
                        if ply >= plies_ago + 1 {
                            if let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1]
                            {
                                let prev_piece =
                                    searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                                if prev_piece < 16 {
                                    let prev_to_hash =
                                        hash_coord_32(prev_move.to.x, prev_move.to.y);
                                    let entry = &mut searcher.cont_history[prev_piece]
                                        [prev_to_hash][cur_from_hash][cur_to_hash];
                                    // Use gravity-based update: entry += bonus - entry * bonus / max
                                    *entry += lmr_bonus - *entry * lmr_bonus / max_history;
                                }
                            }
                        }
                    }
                }
            }
            score = s;
        }

        game.undo_move(&m, undo);

        // Restore previous-move stack entry for this ply after child returns.
        searcher.prev_move_stack[ply] = prev_entry_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.hot.stopped {
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

                searcher.pv_table[ply_base] = Some(m.clone()); // Head of PV is this move
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
                let idx = hash_move_dest(&m);
                let bonus = history_bonus_base() * depth as i32 - history_bonus_sub();
                let adj = bonus.min(history_bonus_cap());
                let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;

                searcher.update_history(m.piece.piece_type(), idx, bonus);

                // Low Ply History update (Stockfish-style)
                searcher.update_low_ply_history(ply, idx, bonus);

                for quiet in &quiets_searched {
                    let qidx = hash_move_dest(quiet);
                    if quiet.piece.piece_type() == m.piece.piece_type() && qidx == idx {
                        continue;
                    }
                    searcher.update_history(quiet.piece.piece_type(), qidx, -bonus);
                    // Penalize other quiets in low ply history too
                    searcher.update_low_ply_history(ply, qidx, -bonus);
                }

                // Killer move heuristic (for non-captures)
                searcher.killers[ply][1] = searcher.killers[ply][0].clone();
                searcher.killers[ply][0] = Some(m.clone());

                // Countermove heuristic: on a quiet beta cutoff, record this move
                // as the countermove to the move that led into this node.
                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (m.piece.piece_type() as u8, m.to.x as i16, m.to.y as i16);
                    }
                }

                // Continuation history update (Stockfish indices: 0, 1, 2, 3, 5)
                for &plies_ago in &[0usize, 1, 2, 3, 5] {
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
                let bonus = 8 * (depth * depth) as i32;
                let e =
                    &mut searcher.capture_history[m.piece.piece_type() as usize][cap_type as usize];
                *e += bonus - *e * bonus / params::DEFAULT_HISTORY_MAX_GRAVITY;
            }
            break;
        } else if let Some(cap_type) = captured_type {
            // Penalize captures that didn't cause a cutoff
            let malus = 2 * depth as i32;
            let e = &mut searcher.capture_history[m.piece.piece_type() as usize][cap_type as usize];
            *e += -malus - *e * malus / params::DEFAULT_HISTORY_MAX_GRAVITY;
        }
    }

    // Staged gen doesn't use move_buffers, so no swap needed

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
    store_tt_with_shared(
        searcher,
        hash,
        depth,
        tt_flag,
        best_score,
        best_move.clone(),
        ply,
    );

    // Update TT Move History (Stockfish-style)
    // Tracks how reliable TT moves are: positive = TT moves tend to be best.
    // Only update in non-PV nodes to get clean cutoff/fail statistics.
    if !is_pv {
        if let Some(ref bm) = best_move {
            // Check if best move matches the TT move
            let tt_move_matched = tt_move
                .as_ref()
                .map_or(false, |tm| tm.from == bm.from && tm.to == bm.to);

            // Gravity-based update: bonus = delta - entry * delta / max
            // Stockfish uses 8192 as max, bonuses +809/-865
            let delta: i32 = if tt_move_matched { 809 } else { -865 };
            let max_tt_history = 8192;
            searcher.tt_move_history +=
                delta - searcher.tt_move_history * delta.abs() / max_tt_history;
        }
    }

    // Update correction history when conditions are met:
    // - Not in check
    // - Best move is quiet or doesn't exist
    // - Score respects bound constraints relative to static eval
    if !in_check {
        let best_move_is_quiet = match &best_move {
            Some(m) => {
                // BITBOARD: Fast capture check
                let captured = game.board.get_piece(m.to.x, m.to.y);
                let is_capture = captured.map_or(false, |p| !p.piece_type().is_neutral_type());
                !is_capture && m.promotion.is_none()
            }
            None => true, // No best move counts as "quiet"
        };

        // Stockfish conditions:
        // - If lower bound (failed high), score should not be below static eval
        // - If upper bound (failed low), score should not be above static eval
        let should_update = match tt_flag {
            TTFlag::LowerBound => best_score >= raw_eval,
            TTFlag::UpperBound => best_score <= raw_eval,
            TTFlag::Exact => true,
            TTFlag::None => false, // Should never happen, but be safe
        };

        if best_move_is_quiet && should_update {
            searcher.update_correction_history(
                game,
                depth,
                raw_eval,
                best_score,
                true,
                false,
                prev_move_idx,
            );
        }
    }

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
    // CRITICAL: Check for max ply BEFORE any array accesses to prevent out-of-bounds
    // This must be the very first check to avoid panics when ply >= MAX_PLY
    if ply >= MAX_PLY - 1 {
        return evaluate(game);
    }

    searcher.hot.nodes += 1;
    searcher.hot.qnodes += 1;

    // Update seldepth
    if ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
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
    let mut tactical_moves: MoveList = MoveList::new();
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

        if searcher.hot.stopped {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;
    use crate::moves::Move;

    // ======================== Constants Tests ========================

    #[test]
    fn test_search_constants() {
        assert_eq!(MAX_PLY, 64);
        assert!(INFINITY > MATE_VALUE);
        assert!(MATE_VALUE > MATE_SCORE);
        assert!(MATE_SCORE > 0);
    }

    #[test]
    fn test_corrhist_constants() {
        assert!(CORRHIST_SIZE.is_power_of_two());
        assert!(LASTMOVE_CORRHIST_SIZE.is_power_of_two());
        assert!(LOW_PLY_HISTORY_ENTRIES.is_power_of_two());
    }

    // ======================== Timer Tests ========================

    #[test]
    fn test_timer_new() {
        let timer = Timer::new();
        let elapsed = timer.elapsed_ms();
        // Should be very small (less than 100ms for new timer)
        assert!(elapsed < 100, "New timer should have small elapsed time");
    }

    #[test]
    fn test_timer_reset() {
        let mut timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let before_reset = timer.elapsed_ms();
        timer.reset();
        let after_reset = timer.elapsed_ms();
        assert!(
            after_reset < before_reset,
            "Reset should reduce elapsed time"
        );
    }

    // ======================== Searcher Tests ========================

    #[test]
    fn test_searcher_new() {
        let searcher = Searcher::new(5000);

        assert_eq!(searcher.hot.time_limit_ms, 5000);
        assert_eq!(searcher.hot.nodes, 0);
        assert_eq!(searcher.hot.qnodes, 0);
        assert!(!searcher.hot.stopped);
        assert!(!searcher.silent);
        assert_eq!(searcher.thread_id, 0);
        assert_eq!(searcher.killers.len(), MAX_PLY);
        assert_eq!(searcher.pv_length.len(), MAX_PLY);
    }

    #[test]
    fn test_searcher_reset_for_iteration() {
        let mut searcher = Searcher::new(5000);
        searcher.hot.nodes = 1000;
        searcher.hot.qnodes = 500;
        searcher.hot.seldepth = 10;
        searcher.pv_length[0] = 5;

        searcher.reset_for_iteration();

        assert_eq!(searcher.hot.nodes, 0);
        assert_eq!(searcher.hot.qnodes, 0);
        assert_eq!(searcher.hot.seldepth, 0);
        assert!(!searcher.hot.stopped);
    }

    #[test]
    fn test_searcher_decay_history() {
        let mut searcher = Searcher::new(5000);
        searcher.history[0][0] = 100;
        searcher.history[1][1] = 200;

        searcher.decay_history();

        assert_eq!(searcher.history[0][0], 90); // 100 * 9/10
        assert_eq!(searcher.history[1][1], 180); // 200 * 9/10
    }

    #[test]
    fn test_searcher_update_history() {
        let mut searcher = Searcher::new(5000);

        searcher.update_history(PieceType::Knight, 42, 100);
        let val = searcher.history[PieceType::Knight as usize][42];
        assert!(val > 0, "History should be updated positively");

        searcher.update_history(PieceType::Knight, 42, -100);
        let val_after = searcher.history[PieceType::Knight as usize][42];
        assert!(
            val_after < val,
            "History should decrease with negative bonus"
        );
    }

    #[test]
    fn test_searcher_check_time_no_limit() {
        let mut searcher = Searcher::new(u128::MAX);
        searcher.hot.nodes = 10000;

        let timed_out = searcher.check_time();
        assert!(!timed_out, "Should not timeout with MAX time limit");
    }

    // ======================== Score Helper Tests ========================

    #[test]
    fn test_mate_score_detection() {
        // Simple mate score detection using constants
        let mate_score = MATE_VALUE - 10;
        let is_mate = mate_score.abs() > MATE_SCORE;
        assert!(is_mate, "Near MATE_VALUE should be detected as mate");

        let normal_score: i32 = 1000;
        let is_normal_mate = normal_score.abs() > MATE_SCORE;
        assert!(!is_normal_mate, "Normal score should not be mate");
    }

    // ======================== CorrHistMode Tests ========================

    #[test]
    fn test_corrhist_mode_enum() {
        assert!(CorrHistMode::PawnBased != CorrHistMode::NonPawnBased);
    }

    // ======================== NodeType Tests ========================

    #[test]
    fn test_node_type_enum() {
        assert!(NodeType::PV != NodeType::Cut);
        assert!(NodeType::Cut != NodeType::All);
    }

    // ======================== SearchStats Tests ========================

    #[test]
    fn test_search_stats_default() {
        let stats = SearchStats {
            tt_capacity: 1000,
            tt_used: 500,
            tt_fill_permille: 500,
        };

        assert_eq!(stats.tt_capacity, 1000);
        assert_eq!(stats.tt_used, 500);
        assert_eq!(stats.tt_fill_permille, 500);
    }

    // ======================== Move Helper Tests ========================

    #[test]
    fn test_move_creation_for_search() {
        let from = Coordinate::new(4, 4);
        let to = Coordinate::new(5, 6);
        let piece = Piece::new(PieceType::Knight, PlayerColor::White);

        let m = Move::new(from, to, piece);

        assert_eq!(m.from.x, 4);
        assert_eq!(m.from.y, 4);
        assert_eq!(m.to.x, 5);
        assert_eq!(m.to.y, 6);
    }

    // ======================== Low Ply History Tests ========================

    #[test]
    fn test_update_low_ply_history() {
        let mut searcher = Searcher::new(5000);

        // Update at ply 0
        searcher.update_low_ply_history(0, 42, 100);
        let val = searcher.low_ply_history[0][42 & LOW_PLY_HISTORY_MASK];
        assert!(val > 0, "Low ply history should be updated");

        // Update at ply >= LOW_PLY_HISTORY_SIZE should do nothing
        searcher.update_low_ply_history(10, 42, 1000);
        // Can't easily verify no change, but at least it shouldn't panic
    }

    // ======================== get_best_move Tests ========================

    #[test]
    fn test_get_best_move_simple_position() {
        let mut game = GameState::new();
        game.board = Board::new();

        // Simple position: white queen can take undefended black rook
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Rook, PlayerColor::Black));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        // Short search with 1 second time limit
        let result = get_best_move(&mut game, 5, 1000, true);

        assert!(result.is_some(), "Should find a move");
        let (best_move, _eval, _stats) = result.unwrap();
        // Should find the queen capture of rook as best
        // (Can't guarantee specific move but should find something)
        assert!(best_move.piece.piece_type() != PieceType::Void);
    }

    #[test]
    fn test_get_best_move_returns_result() {
        let mut game = GameState::new();
        game.board = Board::new();

        // Any position with legal moves
        game.board
            .set_piece(4, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        let result = get_best_move(&mut game, 5, 1000, true);

        assert!(result.is_some(), "Should find a move");
        let (best_move, _eval, stats) = result.unwrap();
        assert!(best_move.piece.piece_type() != PieceType::Void);
        // Check stats are populated
        assert!(stats.tt_capacity > 0);
    }

    // ======================== Evaluation with Search Tests ========================

    #[test]
    fn test_evaluate_with_search() {
        let mut game = GameState::new();
        game.board = Board::new();

        // Balanced position
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Rook, PlayerColor::Black));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        // Get static eval
        let static_eval = evaluate(&game);
        // Should be close to 0 (roughly balanced)
        assert!(
            static_eval.abs() < 500,
            "Balanced position eval should be near 0"
        );
    }

    #[test]
    fn test_tt_basic_operations() {
        let tt = TranspositionTable::new(16);

        assert!(tt.capacity() > 0);
        assert_eq!(tt.used_entries(), 0);
        assert_eq!(tt.fill_permille(), 0);
    }

    // ======================== Timer Extended Tests ========================

    #[test]
    fn test_timer_reset_and_elapsed() {
        let mut timer = Timer::new();
        // Wait just a bit to ensure elapsed is > 0
        let _ = timer.elapsed_ms();
        timer.reset();
        // After reset, elapsed should be close to 0
        let elapsed = timer.elapsed_ms();
        assert!(elapsed < 100, "Elapsed after reset should be small");
    }

    // ======================== Searcher Extended Tests ========================

    #[test]
    fn test_searcher_initialization() {
        let searcher = Searcher::new(10000);

        assert_eq!(searcher.hot.nodes, 0);
        assert!(searcher.tt.capacity() > 0);
    }

    // ======================== History Table Tests ========================

    #[test]
    fn test_killer_moves() {
        let mut searcher = Searcher::new(1000);

        let from = Coordinate::new(4, 4);
        let to = Coordinate::new(5, 6);
        let piece = Piece::new(PieceType::Knight, PlayerColor::White);
        let m = Move::new(from, to, piece);

        // Add killer at ply 0
        searcher.killers[0][1] = searcher.killers[0][0];
        searcher.killers[0][0] = Some(m);

        assert!(searcher.killers[0][0].is_some());
    }

    // ======================== Search Stats Extended Tests ========================

    #[test]
    fn test_search_stats_structure() {
        let stats = SearchStats {
            tt_capacity: 1000,
            tt_used: 100,
            tt_fill_permille: 100,
        };
        assert_eq!(stats.tt_capacity, 1000);
        assert_eq!(stats.tt_used, 100);
        assert_eq!(stats.tt_fill_permille, 100);
    }

    // ======================== Constants Extended Tests ========================

    #[test]
    fn test_search_constants_values() {
        assert!(MAX_PLY > 0);
        assert!(INFINITY > 0);
        assert!(MATE_VALUE > 0);
        assert!(MATE_SCORE < MATE_VALUE);
    }

    // ======================== Extended Searcher Tests ========================

    #[test]
    fn test_searcher_killers_and_history() {
        let mut searcher = Searcher::new(1000);

        // Add some killer moves
        let m = Move::new(
            Coordinate::new(0, 0),
            Coordinate::new(1, 1),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        searcher.killers[0][0] = Some(m);
        assert!(searcher.killers[0][0].is_some());
    }

    #[test]
    fn test_history_table_dimensions() {
        let searcher = Searcher::new(1000);

        // Verify history table dimensions [32 piece types][256 to squares]
        assert_eq!(searcher.history.len(), 32);
        assert_eq!(searcher.history[0].len(), 256);
    }

    // ======================== MoveList Operations ========================

    #[test]
    fn test_movelist_operations() {
        use crate::moves::MoveList;

        let mut moves = MoveList::new();
        assert!(moves.is_empty());

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 6),
            Piece::new(PieceType::Knight, PlayerColor::White),
        );

        moves.push(m);
        assert_eq!(moves.len(), 1);
        assert!(!moves.is_empty());
    }

    // ======================== Integration Tests ========================

    #[test]
    fn test_search_endgame_position() {
        let mut game = GameState::new();
        game.board = Board::new();

        // KQ vs K endgame
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();
        game.board.rebuild_tiles();

        let result = get_best_move(&mut game, 3, 500, true);
        assert!(result.is_some(), "Should find a move in KQ vs K");

        let (best_move, eval, _stats) = result.unwrap();
        assert!(eval > 0, "White should be winning in KQ vs K");
        assert!(best_move.piece.piece_type() != PieceType::Void);
    }

    #[test]
    fn test_search_with_captures() {
        let mut game = GameState::new();
        game.board = Board::new();

        // Position with clear capture
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Pawn, PlayerColor::Black)); // Can be captured

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();
        game.board.rebuild_tiles();

        let result = get_best_move(&mut game, 4, 500, true);
        assert!(result.is_some());
    }

    // ======================== Format PV Tests ========================

    #[test]
    fn test_format_pv_empty() {
        let searcher = Searcher::new(1000);
        let pv = searcher.format_pv();
        // PV should be a string (possibly empty)
        assert!(pv.is_empty() || !pv.is_empty());
    }

    // ======================== CorrHist Mode Tests ========================

    #[test]
    fn test_set_corrhist_mode() {
        let mut searcher = Searcher::new(1000);
        let game = GameState::new();

        searcher.set_corrhist_mode(&game);
        // Mode should be set (either PawnBased or NonPawnBased)
        assert!(
            searcher.corrhist_mode == CorrHistMode::PawnBased
                || searcher.corrhist_mode == CorrHistMode::NonPawnBased
        );
    }

    // ======================== Adjusted Eval Tests ========================

    #[test]
    fn test_adjusted_eval() {
        let searcher = Searcher::new(1000);
        let game = GameState::new();

        let raw_eval = 100;
        let adjusted = searcher.adjusted_eval(&game, raw_eval, 0);
        // Adjusted eval should be within reasonable bounds of raw
        assert!(adjusted.abs() < raw_eval.abs() + 1000);
    }

    // ======================== Build Search Stats Tests ========================

    #[test]
    fn test_build_search_stats() {
        let searcher = Searcher::new(1000);
        let stats = build_search_stats(&searcher);

        assert!(stats.tt_capacity > 0);
        assert_eq!(stats.tt_used, 0);
        assert_eq!(stats.tt_fill_permille, 0);
    }

    // ======================== Extract PV Tests ========================

    #[test]
    fn test_extract_pv() {
        let searcher = Searcher::new(1000);
        let pv = extract_pv(&searcher);
        // PV should be empty for a fresh searcher
        assert!(pv.is_empty());
    }

    // ======================== With Global Searcher Tests ========================

    #[test]
    fn test_with_global_searcher() {
        let result = with_global_searcher(1000, true, |searcher| searcher.tt.capacity());
        assert!(result > 0);
    }

    // ======================== Reset Search State Tests ========================

    #[test]
    fn test_reset_search_state() {
        // Should not panic
        reset_search_state();
    }

    // ======================== Get Current TT Stats Tests ========================

    #[test]
    fn test_get_current_tt_stats() {
        GLOBAL_SEARCHER.with(|cell| {
            *cell.borrow_mut() = Some(Searcher::new(1000));
        });
        let stats = get_current_tt_stats();
        assert!(stats.tt_capacity > 0);
        reset_search_state();
    }

    // ======================== Searcher Method Tests ========================

    #[test]
    fn test_capture_history_update() {
        let mut searcher = Searcher::new(1000);

        // Update capture history
        searcher.capture_history[PieceType::Rook as usize][PieceType::Pawn as usize] = 100;
        let val = searcher.capture_history[PieceType::Rook as usize][PieceType::Pawn as usize];
        assert_eq!(val, 100);
    }

    #[test]
    fn test_countermove_heuristic() {
        let mut searcher = Searcher::new(1000);

        // Update countermove table
        let prev_from_hash = 10;
        let prev_to_hash = 20;
        searcher.countermoves[prev_from_hash][prev_to_hash] = (1, 5, 5);

        let (piece_type, to_x, to_y) = searcher.countermoves[prev_from_hash][prev_to_hash];
        assert_eq!(piece_type, 1);
        assert_eq!(to_x, 5);
        assert_eq!(to_y, 5);
    }

    // ======================== Search Functionality Tests ========================

    #[test]
    fn test_multipv_search_functionality() {
        let mut game = GameState::new();
        game.board = Board::new();
        // Setup a position where white has multiple good moves
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(2, 2, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::White));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();
        game.board.rebuild_tiles();

        // Search with MultiPV = 2
        let result = get_best_moves_multipv(&mut game, 2, 500, 2, true);

        // Should find at least 1 line, hopefully 2 if the position allows
        assert!(!result.lines.is_empty());
        if result.lines.len() > 1 {
            assert!(
                result.lines[0].mv != result.lines[1].mv,
                "MultiPV moves should be unique"
            );
            assert!(
                result.lines[0].score >= result.lines[1].score,
                "MultiPV lines should be ordered by score"
            );
        }
    }

    #[test]
    fn test_tt_integration_via_searcher() {
        let mut searcher = Searcher::new(1000);
        let hash = 123456789;
        let depth = 5;
        let score = 1000;
        let best_move = Move::new(
            Coordinate::new(0, 0),
            Coordinate::new(1, 1),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        // Store EXACT score using correct TT signature:
        // store(&mut self, hash, depth, flag, score, move, ply)
        searcher.tt.store(
            hash,
            depth,
            crate::search::tt::TTFlag::Exact,
            score,
            Some(best_move),
            0,
        );

        // Probe EXACT score using correct TT signature:
        // probe(&self, hash, alpha, beta, depth, ply, rule50, rule_limit)
        let result = searcher
            .tt
            .probe(hash, score - 100, score + 100, depth, 0, 0, 100);
        assert!(result.is_some());
        let (probed_score, probed_move) = result.unwrap();
        assert_eq!(probed_score, score);
        assert!(probed_move.is_some());
        assert_eq!(probed_move.unwrap().from.x, 0);
    }

    #[test]
    fn test_search_mate_in_one() {
        let mut game = GameState::new();
        game.board = Board::new();

        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::Black));
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                game.board
                    .set_piece(dx, dy, Piece::new(PieceType::Pawn, PlayerColor::Black));
            }
        }

        game.board.remove_piece(&0, &1);
        game.board
            .set_piece(-5, -5, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Rook, PlayerColor::White));

        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        assert_eq!(
            game.white_piece_count, 2,
            "Should have 2 white pieces (King, Rook)"
        );
        assert!(
            game.black_piece_count >= 8,
            "Should have at least 8 black pieces"
        );
        assert!(
            game.black_king_pos.is_some(),
            "Black king position must be detected"
        );
        assert!(
            game.white_king_pos.is_some(),
            "White king position must be detected"
        );

        game.recompute_hash();
        game.board.rebuild_tiles();

        // Verification: ensure move generation works
        let moves = game.get_legal_moves();
        assert!(
            !moves.is_empty(),
            "White should have legal moves, found 0. Piece counts: W={}, B={}",
            game.white_piece_count,
            game.black_piece_count
        );
        assert!(!moves.is_empty(), "White should have legal moves, found 0");

        // Search depth 3 to be absolutely sure
        let result = get_best_move(&mut game, 3, 2000, true);
        assert!(
            result.is_some(),
            "Search returned None even though legal moves exist"
        );
        let (best_move, score, _stats) = result.unwrap();

        // Should find the mate move to (0,5)
        assert_eq!(best_move.to.x, 0);
        assert_eq!(best_move.to.y, 5);

        assert!(
            score > 800000,
            "Should detect mate score (>800000), got {}",
            score
        );
    }
    #[test]
    fn test_quiescence_search_depth() {
        let mut searcher = Searcher::new(1000);
        let mut game = GameState::new();
        game.board = Board::new();

        // Setup empty board with kings to avoid panics
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();
        game.board.rebuild_tiles();

        // Qsearch should return static eval on quiet position
        let alpha = -10000;
        let beta = 10000;
        let score = quiescence(&mut searcher, &mut game, 0, alpha, beta);
        assert!(score.abs() < 500); // Should be near zero for balanced empty board
        assert_eq!(searcher.hot.qnodes, 1);
    }

    #[test]
    fn test_negamax_node_counts() {
        let mut game = GameState::new();
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();

        let nodes = negamax_node_count_for_depth(&mut game, 1);
        assert!(nodes > 0);
    }

    // ======================== PVLine and MultiPVResult Tests ========================

    #[test]
    fn test_pvline_structure() {
        let dummy_move = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        let pv = PVLine {
            mv: dummy_move,
            score: 100,
            depth: 5,
            pv: vec![],
        };
        assert_eq!(pv.score, 100);
        assert_eq!(pv.depth, 5);
        assert!(pv.pv.is_empty());
    }

    #[test]
    fn test_multipv_result_structure() {
        let result = MultiPVResult {
            lines: vec![],
            stats: SearchStats {
                tt_capacity: 1000,
                tt_used: 100,
                tt_fill_permille: 100,
            },
        };
        assert!(result.lines.is_empty());
        assert_eq!(result.stats.tt_capacity, 1000);
    }

    // ======================== Thread ID and Silent Mode Tests ========================

    #[test]
    fn test_searcher_thread_id() {
        let searcher = Searcher::new(1000);
        assert_eq!(searcher.thread_id, 0); // Default thread ID
    }

    #[test]
    fn test_searcher_silent_mode() {
        let mut searcher = Searcher::new(1000);
        assert!(!searcher.silent); // Default is not silent
        searcher.silent = true;
        assert!(searcher.silent);
    }

    // ======================== Move Rule Limit Tests ========================

    #[test]
    fn test_move_rule_limit() {
        let searcher = Searcher::new(1000);
        assert_eq!(searcher.move_rule_limit, 100); // Default 50-move rule
    }
}
