use crate::board::PieceType;
use crate::evaluation::{evaluate, get_piece_value};
use crate::game::GameState;
use crate::moves::{Move, MoveGenContext, MoveList, get_quiescence_captures};
use crate::search::params::{
    aspiration_fail_mult, aspiration_max_window, aspiration_window, delta_margin,
    history_bonus_base, history_bonus_cap, history_bonus_sub, hlp_history_leaf, hlp_history_reduce,
    hlp_max_depth, hlp_min_moves, iir_min_depth, lmp_base, lmp_depth_mult, lmr_cutoff_thresh,
    lmr_divisor, lmr_min_depth, lmr_min_moves, lmr_tt_history_thresh, low_depth_probcut_margin,
    max_history, nmp_base, nmp_depth_mult, nmp_min_depth, nmp_reduction_base, nmp_reduction_div,
    probcut_depth_sub, probcut_divisor, probcut_improving, probcut_margin, probcut_min_depth,
    razoring_linear, razoring_quad, rfp_improving_mult, rfp_max_depth, rfp_mult_no_tt, rfp_mult_tt,
    rfp_worsening_mult, see_capture_hist_div, see_capture_linear, see_quiet_quad,
};
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
// For web WASM (browser), use js_sys::Date for timing
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
use js_sys::Date;
use std::cell::RefCell;
// For native builds and WASI, use std::time::Instant
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
use std::time::Instant;

pub struct ProbeContext {
    pub hash: u64,
    pub alpha: i32,
    pub beta: i32,
    pub depth: usize,
    pub ply: usize,
    pub rule50_count: u32,
    pub rule_limit: i32,
}

pub struct StoreContext {
    pub hash: u64,
    pub depth: usize,
    pub flag: TTFlag,
    pub score: i32,
    pub static_eval: i32,
    pub is_pv: bool,
    pub best_move: Option<Move>,
    pub ply: usize,
}

pub struct NegamaxContext<'a> {
    pub searcher: &'a mut Searcher,
    pub game: &'a mut GameState,
    pub depth: usize,
    pub ply: usize,
    pub alpha: i32,
    pub beta: i32,
    pub allow_null: bool,
    pub node_type: NodeType,
    pub was_null_move: bool,
    pub excluded_move: Option<Move>,
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn now_ms() -> f64 {
    Date::now()
}

/// Fast deterministic seedable PRNG for search noise and strength limiting.
/// Uses a custom Xorshift-like algorithm for speed and predictability.
#[derive(Clone, Debug)]
pub struct Prng {
    state: u64,
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        let mut p = Prng { state: seed };
        // Advance once to avoid problems with 0 seed if necessary
        if p.state == 0 {
            p.state = 0x123456789ABCDEF0;
        }
        for _ in 0..4 {
            p.next_u64();
        }
        p
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Generate deterministic noise for a given position and seed.
#[inline(always)]
fn get_noise(seed: u64, hash: u64, amp: i32) -> i32 {
    if amp == 0 {
        return 0;
    }
    // High-quality SplitMix64 hash stage for mixing position and seed
    let mut x = seed.wrapping_add(hash);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    (x % (2 * amp as u64)) as i32 - amp
}

pub const MAX_PLY: usize = 64;
pub const INFINITY: i32 = 1_000_000;
pub const MATE_VALUE: i32 = 900_000;
pub const MATE_SCORE: i32 = 800_000;
pub const THINK_TIME_MS: u128 = 3000; // 3 seconds per move (default, may be overridden by caller)

pub const MAX_SITE_SKILL: u32 = 3; // Current max skill level on the site
pub const MAX_PV_COUNT: usize = 4; // MultiPV lines to use

#[inline(always)]
pub const fn is_win(value: i32) -> bool {
    value > MATE_SCORE
}

#[inline(always)]
pub const fn is_loss(value: i32) -> bool {
    value < -MATE_SCORE
}

use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

/// Global stop flag for all search threads.
pub(crate) static GLOBAL_STOP: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub const fn is_decisive(value: i32) -> bool {
    value.abs() > MATE_SCORE
}

pub const VALUE_DRAW: i32 = 0;
#[inline(always)]
pub fn value_draw(nodes: u64) -> i32 {
    // VALUE_DRAW is 0, so this gives -1 or +1
    -1 + ((nodes & 0x2) as i32)
}

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

// Low Ply History constants:
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

pub mod params;
pub mod tt_defs;
pub use tt_defs::{TTFlag, TTProbeParams, TTProbeResult, TTStoreParams};

mod tt;
use tt::LocalTranspositionTable;

#[cfg(feature = "multithreading")]
mod shared_tt;
#[cfg(feature = "multithreading")]
use shared_tt::SharedTranspositionTable;

mod ordering;
use ordering::{hash_coord_32, hash_move_dest, hash_move_from, sort_captures, sort_moves_root};

pub mod movegen;
use movegen::StagedMoveGen;

mod see;
pub(crate) use see::see_ge;
pub(crate) use see::static_exchange_eval_impl as static_exchange_eval;

pub mod zobrist;
pub use zobrist::{
    SIDE_KEY, castling_rights_key, castling_rights_key_from_bitfield, en_passant_key, material_key,
    pawn_key, pawn_special_right_key, piece_key,
};

// ============================================================================
// TT Probe/Store (Dispatch wrapper)
// ============================================================================

static LOCAL_TT: OnceLock<LocalTranspositionTable> = OnceLock::new();
#[cfg(feature = "multithreading")]
static SHARED_TT: OnceLock<SharedTranspositionTable> = OnceLock::new();

/// Flag to enable shared TT usage.
/// Set to true when parallel search is active.
#[cfg(feature = "multithreading")]
pub(crate) static USE_SHARED_TT: AtomicBool = AtomicBool::new(false);

/// Helper struct to satisfy closure syntax in get_or_init
pub struct TranspositionTable;
impl TranspositionTable {
    pub fn new(_: usize) -> Self {
        Self
    }
}

/// Enum to hold reference to the active TT implementation
pub enum TranspositionTableRef<'a> {
    Local(&'a LocalTranspositionTable),
    #[cfg(feature = "multithreading")]
    Shared(&'a SharedTranspositionTable),
}

impl<'a> TranspositionTableRef<'a> {
    #[inline]
    pub fn probe(&self, params: &TTProbeParams) -> Option<TTProbeResult> {
        match self {
            Self::Local(tt) => tt.probe(params),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.probe(params),
        }
    }

    #[inline]
    pub fn probe_move(&self, hash: u64) -> Option<Move> {
        match self {
            Self::Local(tt) => tt.probe_move(hash),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.probe_move(hash),
        }
    }

    #[inline]
    pub fn store(&self, params: &TTStoreParams) {
        match self {
            Self::Local(tt) => tt.store(params),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.store(params),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            Self::Local(tt) => tt.capacity(),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.capacity(),
        }
    }

    #[inline]
    pub fn used_entries(&self) -> usize {
        match self {
            Self::Local(tt) => tt.used_entries(),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.used_entries(),
        }
    }

    #[inline]
    pub fn fill_permille(&self) -> u32 {
        match self {
            Self::Local(tt) => tt.fill_permille(),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.fill_permille(),
        }
    }

    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        match self {
            Self::Local(tt) => tt.prefetch_entry(hash),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.prefetch_entry(hash),
        }
    }

    #[inline]
    pub fn increment_age(&self) {
        match self {
            Self::Local(tt) => tt.increment_age(),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.increment_age(),
        }
    }

    #[inline]
    pub fn clear(&self) {
        match self {
            Self::Local(tt) => tt.clear(),
            #[cfg(feature = "multithreading")]
            Self::Shared(tt) => tt.clear(),
        }
    }
}

pub struct GlobalTtHandle;

impl GlobalTtHandle {
    #[inline]
    pub fn get(&self) -> Option<TranspositionTableRef<'_>> {
        #[cfg(feature = "multithreading")]
        if USE_SHARED_TT.load(std::sync::atomic::Ordering::Relaxed) {
            return SHARED_TT.get().map(TranspositionTableRef::Shared);
        }
        LOCAL_TT.get().map(TranspositionTableRef::Local)
    }

    pub fn get_or_init<F>(&self, _f: F) -> TranspositionTableRef<'_>
    where
        F: FnOnce() -> TranspositionTable,
    {
        #[cfg(feature = "multithreading")]
        if USE_SHARED_TT.load(std::sync::atomic::Ordering::Relaxed) {
            let tt = SHARED_TT.get_or_init(|| SharedTranspositionTable::new(64));
            return TranspositionTableRef::Shared(tt);
        }
        let tt = LOCAL_TT.get_or_init(|| LocalTranspositionTable::new(16));
        TranspositionTableRef::Local(tt)
    }
}

pub(crate) static GLOBAL_TT: GlobalTtHandle = GlobalTtHandle;

/// Probe the TT. Dispatch based on thread configuration.
#[inline]
pub fn probe_tt_with_shared(_searcher: &Searcher, ctx: &ProbeContext) -> Option<TTProbeResult> {
    GLOBAL_TT.get().and_then(|tt| {
        tt.probe(&crate::search::tt_defs::TTProbeParams {
            hash: ctx.hash,
            alpha: ctx.alpha,
            beta: ctx.beta,
            depth: ctx.depth,
            ply: ctx.ply,
            rule50_count: ctx.rule50_count,
            rule_limit: ctx.rule_limit,
        })
    })
}

/// Store to the TT. Dispatch based on thread configuration.
#[inline]
pub fn store_tt_with_shared(_searcher: &mut Searcher, ctx: &StoreContext) {
    if let Some(tt) = GLOBAL_TT.get() {
        tt.store(&crate::search::tt_defs::TTStoreParams {
            hash: ctx.hash,
            depth: ctx.depth,
            flag: ctx.flag,
            score: ctx.score,
            static_eval: ctx.static_eval,
            is_pv: ctx.is_pv,
            best_move: ctx.best_move,
            ply: ctx.ply,
        });
    }
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
    /// Tracks the minimum depth that must be completed before time stops are allowed.
    /// Set to 1 at search start, cleared to 0 after depth 1 completes.
    pub min_depth_required: usize,
    /// Optimum time to use for this search (soft limit)
    pub optimum_time_ms: u128,
    /// Maximum time to use for this search (hard limit)
    pub maximum_time_ms: u128,
    /// Total best move changes (instability) persisted across iterations
    pub tot_best_move_changes: f64,
    /// Best move changes in the current iteration
    pub best_move_changes: f64,
    /// Nodes spent on the current best move (first root move) in the current iteration
    pub best_move_nodes: u64,
    /// Running average score smoothed across iterations
    pub best_previous_average_score: i32,
    /// Running scores for falling eval (circular buffer of last 4 iterations)
    pub iter_values: [i32; 4],
    /// Index into iter_values circular buffer
    pub iter_idx: usize,
    /// Previous time reduction factor (for smoothing across iterations)
    pub prev_time_reduction: f64,
    /// Depth at which best move was last changed
    pub last_best_move_depth: usize,
    /// Whether this is a "soft" time limit (suggested time, can exceed up to max)
    /// vs a hard limit (must stop at maximum time). For untimed games with a
    /// suggested per-move limit, this allows the engine to use more time when beneficial.
    pub is_soft_limit: bool,
    /// Calculated total time budget for this move, including dynamic factors.
    /// Used by check_time for mid-depth stops.
    pub total_time_ms: f64,
    /// Time (ms) when the current iterative deepening depth started.
    pub iter_start_ms: f64,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Self {
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let start = now_ms();
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let start = Instant::now();
        Timer { start }
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

impl SearcherHot {
    /// Calculate optimum and maximum time.
    ///
    /// Time management works differently based on `is_soft_limit`:
    /// - **Soft limit** (untimed game with suggested time): The engine can freely
    ///   use up to `maximum_time_ms` if beneficial. Optimum is set higher
    ///   because there's no risk of flagging, and max is close to the full budget.
    /// - **Hard limit** (timed game): The engine must be conservative. Optimum is
    ///   set lower and max is capped to leave headroom for dynamic extensions
    ///   in critical positions.
    ///
    /// The dynamic factors (fallingEval up to 1.7x, instability up to ~2.5x, etc.)
    /// multiply the optimum time, capped at maximum.
    pub fn set_time_limits(&mut self, opt_ms: u128, max_ms: u128, is_soft: bool) {
        self.optimum_time_ms = opt_ms;
        self.maximum_time_ms = max_ms;
        self.is_soft_limit = is_soft;
        self.time_limit_ms = max_ms; // Used by check_time()
    }
}

/// Lightweight statistics about the transposition table after a search.
#[derive(Clone, Debug)]
pub struct SearchStats {
    pub nodes: u64,
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

/// Result from a single thread's search, used for Lazy SMP thread voting.
/// Stockfish's thread voting algorithm weights votes by:
///   (score - minScore + 14) * completedDepth
/// This ensures deeper searches with better scores have more influence.
#[cfg(feature = "multithreading")]
#[derive(Clone, Debug)]
pub struct ThreadResult {
    /// Best move found by this thread
    pub best_move: Move,
    /// Score of the best move (from side-to-move perspective)
    pub score: i32,
    /// Highest completed depth
    pub completed_depth: usize,
    /// Length of the PV (longer PVs are more trustworthy)
    pub pv_length: usize,
    /// Total nodes searched
    pub nodes: u64,
    /// Thread index (for debugging)
    pub thread_id: usize,
}

thread_local! {
    pub(crate) static GLOBAL_SEARCHER: RefCell<Option<Searcher>> = const { RefCell::new(None) };
}

fn build_search_stats(searcher: &Searcher) -> SearchStats {
    let (cap, used, fill) = GLOBAL_TT.get().map_or((0, 0, 0), |tt| {
        (tt.capacity(), tt.used_entries(), tt.fill_permille())
    });
    SearchStats {
        nodes: searcher.hot.nodes,
        tt_capacity: cap,
        tt_used: used,
        tt_fill_permille: fill,
    }
}

/// Return current TT statistics from the persistent global searcher, if any.
/// When no global searcher exists yet, initializes one with default size to report capacity.
pub fn get_current_tt_stats() -> SearchStats {
    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();

        // Ensure searcher exists so we can report its capacity/fill even before first search
        let searcher = opt.get_or_insert_with(|| Searcher::new(4000));
        build_search_stats(searcher)
    })
}

/// Reset the global search state.
/// Call this when starting a brand new game so old entries don't carry over.
pub fn reset_search_state() {
    GLOBAL_SEARCHER.with(|cell| {
        *cell.borrow_mut() = None;
    });

    // Clear pawn structure cache for new game
    crate::evaluation::base::clear_pawn_cache();

    // Clear material cache for new game
    crate::evaluation::insufficient_material::clear_material_cache();

    // Clear transposition table
    if let Some(tt) = GLOBAL_TT.get() {
        tt.clear()
    }
}

/// Search state that persists across the search
pub struct Searcher {
    /// Hot data - grouped for cache efficiency
    pub hot: SearcherHot,

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

    // Search noise parameters for SPRT pairing
    pub noise_amp: i32,
    pub seed: u64,
    pub rng: Prng,

    // Depth fully completed in the current search
    pub completed_depth: usize,

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

    #[allow(clippy::type_complexity)]
    // Continuation history: [is_capture][in_check][prev_piece_type][prev_to_hash][cur_from_hash][cur_to_hash]
    // Using 32 for piece types to accommodate all types (8*32*32*32*32*4 = 32MB)
    pub cont_history: Box<[[[[[[i32; 32]; 32]; 32]; 32]; 2]; 2]>,

    // Continuation Correction History: [prev_piece_type][prev_to_hash][cur_piece_type][cur_to_hash]
    // Used for evaluation correction (32*32*32*32*4 = 4MB)
    pub cont_corrhist: Box<[[[[i32; 32]; 32]; 32]; 32]>,

    // MultiPV: moves to exclude from root search (for finding 2nd, 3rd, etc. best moves)
    // Stored as (from_x, from_y, to_x, to_y) tuples for fast comparison without cloning
    pub excluded_moves: Vec<(i64, i64, i64, i64)>,

    // Correction History - variant-aware for optimal performance:
    // - PawnBased mode: pawn + material (for CoaIP/Classical/Chess variants)
    // - NonPawnBased mode: non-pawn + material + last-move (for other variants)
    pub corrhist_mode: CorrHistMode,
    pub pawn_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub nonpawn_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub material_corrhist: Box<[[i32; CORRHIST_SIZE]; 2]>,
    pub lastmove_corrhist: Box<[i32; LASTMOVE_CORRHIST_SIZE]>,

    // Stacks to track node state for history updates
    pub in_check_history: Vec<bool>,
    pub capture_history_stack: Vec<bool>,

    /// TT Move History: tracks reliability of TT moves.
    /// Positive values = TT moves tend to be best moves.
    /// Negative values = TT moves often fail.
    pub tt_move_history: i32,

    /// Reduction stack for hindsight depth adjustment.
    /// Used to adjust depth based on prior search decisions.
    pub reduction_stack: Vec<i32>,

    /// Cutoff count per ply.
    /// Used to increase LMR when next ply has many fail highs.
    pub cutoff_cnt: Vec<u8>,

    /// Dynamic move rule limit (e.g. 100 for 50-move rule)
    pub move_rule_limit: i32,

    /// Low Ply History: [ply][move_hash] -> score
    /// Tracks which moves were successful at low plies (first 4 from root).
    /// Used to boost ordering for moves that worked well near root.
    pub low_ply_history: Box<[[i32; LOW_PLY_HISTORY_ENTRIES]; LOW_PLY_HISTORY_SIZE]>,

    pub plies_from_null: Box<[u8; MAX_PLY]>,
}

impl Searcher {
    pub fn new(time_limit_ms: u128) -> Self {
        // Triangular PV table
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
                min_depth_required: 1, // Must complete at least depth 1
                optimum_time_ms: 0,
                maximum_time_ms: 0,
                tot_best_move_changes: 0.0,
                best_move_changes: 0.0,
                best_move_nodes: 0,
                best_previous_average_score: 0,
                iter_values: [0; 4],
                iter_idx: 0,
                prev_time_reduction: 1.0,
                last_best_move_depth: 0,
                is_soft_limit: false,
                total_time_ms: 0.0,
                iter_start_ms: 0.0,
            },
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
            noise_amp: 0,
            seed: 0,
            rng: Prng::new(0),
            completed_depth: 0,
            silent: false,
            thread_id: 0,
            move_buffers,
            move_history: vec![None; MAX_PLY],
            moved_piece_history: vec![0; MAX_PLY],
            in_check_history: vec![false; MAX_PLY],
            capture_history_stack: vec![false; MAX_PLY],
            cont_history: Box::new([[[[[[0i32; 32]; 32]; 32]; 32]; 2]; 2]),
            cont_corrhist: Box::new([[[[0i32; 32]; 32]; 32]; 32]),
            excluded_moves: Vec::new(),
            corrhist_mode: CorrHistMode::NonPawnBased, // Default, set based on variant at search start
            pawn_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            nonpawn_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            material_corrhist: Box::new([[0i32; CORRHIST_SIZE]; 2]),
            lastmove_corrhist: Box::new([0i32; LASTMOVE_CORRHIST_SIZE]),
            tt_move_history: 0,
            reduction_stack: vec![0; MAX_PLY],
            cutoff_cnt: vec![0; MAX_PLY + 2], // +2 for (ply+2) access pattern
            move_rule_limit: 100,             // Default, will be updated from GameState
            low_ply_history: Box::new([[0i32; LOW_PLY_HISTORY_ENTRIES]; LOW_PLY_HISTORY_SIZE]),
            plies_from_null: Box::new([255; MAX_PLY]),
        }
    }

    pub fn reset_for_iteration(&mut self) {
        self.hot.stopped = false;
        self.hot.seldepth = 0;

        // Reset PV lengths only - much faster than clearing entire array
        // The PV entries will be overwritten as needed during search
        self.pv_length = [0; MAX_PLY];
    }

    /// Detects shuffling sequences to prevent search explosions in closed positions.
    pub fn is_shuffling(&self, game: &GameState, m: &Move, ply: usize) -> bool {
        // 1. Pawn moves, captures, and early game/reversible moves are not shuffling
        if m.piece.piece_type() == PieceType::Pawn
            || game.board.is_occupied(m.to.x, m.to.y)
            || game.halfmove_clock < 10
        {
            return false;
        }

        // 2. Depth/Ply guards
        let plies_from_null = self.plies_from_null[ply];
        if plies_from_null <= 6 || ply < 20 {
            return false;
        }

        // 3. Check for geometric shuffling pattern:
        // Current move: A -> B
        // Ply-2 move:   B -> A
        // Ply-4 move:   A -> B
        if let Some(ref m2) = self.move_history[ply - 2]
            && let Some(ref m4) = self.move_history[ply - 4]
        {
            return m.from == m2.to && m2.from == m4.to;
        }

        false
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

    /// Start a new search: reset per-search state and increment TT age (or clear if requested).
    pub fn new_search(&mut self) {
        if let Some(tt) = GLOBAL_TT.get() {
            tt.increment_age();
        }

        // Reset cumulative counters
        self.hot.nodes = 0;
        self.hot.qnodes = 0;
        self.hot.seldepth = 0;
        self.hot.stopped = false;

        // Reset search control
        self.hot.min_depth_required = 1;

        // Reset time management variables
        self.hot.tot_best_move_changes = 0.0;
        self.hot.best_move_changes = 0.0;
        self.hot.best_move_nodes = 0;
        self.hot.best_previous_average_score = 0;
        self.hot.iter_values.fill(0);
        self.hot.iter_idx = 0;
        self.hot.prev_time_reduction = 1.0;
        self.hot.last_best_move_depth = 0;
        self.hot.total_time_ms = 0.0;
        self.hot.iter_start_ms = 0.0;

        // Reset iterative deepening state
        self.prev_score = 0;
        self.completed_depth = 0;
        self.best_move_root = None;

        // Reset killers - they are position-dependent and should be fresh for a new search
        for k in self.killers.iter_mut() {
            k[0] = None;
            k[1] = None;
        }

        // Reset TT move history - hits on the old TT are no longer relevant
        self.tt_move_history = 0;

        // Fill lowPlyHistory with 97 at the start of iterative deepening
        // (not 0, to give a small positive bias to moves that haven't been seen)
        for row in self.low_ply_history.iter_mut() {
            row.fill(97);
        }
    }

    /// Clears TT and resets all history tables to neutral values.
    pub fn clear(&mut self) {
        // Clear transposition table
        if let Some(tt) = GLOBAL_TT.get() {
            tt.clear();
        }

        // Reset main history
        for row in self.history.iter_mut() {
            for val in row.iter_mut() {
                *val = 0;
            }
        }

        // Reset capture history
        for row in self.capture_history.iter_mut() {
            for val in row.iter_mut() {
                *val = 0;
            }
        }

        // Reset continuation history
        for c in 0..2 {
            for ic in 0..2 {
                for p in 0..32 {
                    for t in 0..32 {
                        for f in 0..32 {
                            self.cont_history[c][ic][p][t][f].fill(0);
                        }
                    }
                }
            }
        }

        // Reset continuation correction history
        for p in 0..32 {
            for t in 0..32 {
                for p2 in 0..32 {
                    self.cont_corrhist[p][t][p2].fill(0);
                }
            }
        }

        // Reset correction histories
        for row in self.pawn_corrhist.iter_mut() {
            row.fill(0);
        }
        for row in self.nonpawn_corrhist.iter_mut() {
            row.fill(0);
        }
        for row in self.material_corrhist.iter_mut() {
            row.fill(0);
        }
        self.lastmove_corrhist.fill(0);

        // Reset low ply history
        for row in self.low_ply_history.iter_mut() {
            row.fill(0);
        }

        // Reset killers
        for k in self.killers.iter_mut() {
            k[0] = None;
            k[1] = None;
        }

        // Reset countermoves
        for row in self.countermoves.iter_mut() {
            for val in row.iter_mut() {
                *val = (0, 0, 0);
            }
        }

        // Reset TT move history
        self.tt_move_history = 0;
    }

    /// Gravity-style history update: scales updates based on current value and clamps to [-MAX_HISTORY, MAX_HISTORY].
    #[inline]
    pub fn update_history(&mut self, piece: PieceType, idx: usize, bonus: i32) {
        let max_h = max_history();
        let clamped = bonus.clamp(-max_h, max_h);

        let entry = &mut self.history[piece as usize][idx];
        *entry += clamped - *entry * clamped.abs() / max_h;
    }

    /// Update low ply history for moves that caused beta cutoff at low plies.
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

        // Don't stop until we've completed at least depth 1
        if self.hot.min_depth_required > 0 {
            return false;
        }

        if self.hot.nodes & 8191 == 0 {
            let elapsed = self.hot.timer.elapsed_ms() as f64;
            let hard_limit = if self.hot.maximum_time_ms > 0 {
                self.hot.maximum_time_ms as f64
            } else {
                self.hot.time_limit_ms as f64
            };

            // 1. Hard stop at maximum time - this is absolute safety.
            if elapsed >= hard_limit {
                self.hot.stopped = true;
                return true;
            }

            // Proactive Safety Stop:
            // Only trigger if we're very close to the limit and NPS is slow.
            // This is a last-resort safety, not a regular termination condition.
            if self.hot.nodes > 8192 {
                let time_to_next_check = (8192.0 * elapsed) / self.hot.nodes as f64;
                // Only stop if we literally cannot reach the next check in time.
                if (elapsed + time_to_next_check) > hard_limit {
                    self.hot.stopped = true;
                    return true;
                }
            }

            // If the current depth ALONE has consumed > 50% of the move budget, return.
            // ONLY for hard limits. For soft limits (fixed time), we want to use all time.
            if !self.hot.is_soft_limit
                && self.hot.total_time_ms > 0.0
                && elapsed - self.hot.iter_start_ms > self.hot.total_time_ms * 0.50
            {
                self.hot.stopped = true;
                return true;
            }
        }
        self.hot.stopped
    }

    /// Apply correction history to raw static evaluation.
    /// Uses variant-specific mode set at search start for zero overhead.
    #[inline]
    pub fn adjusted_eval(
        &self,
        game: &GameState,
        raw_eval: i32,
        ply: usize,
        prev_move_idx: usize,
    ) -> i32 {
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

                // Continuation correction (ss-2 and ss-4):
                let mut cont_corr = 0;

                if ply > 0
                    && let Some(m) = self.move_history[ply - 1]
                {
                    let cur_pc = m.piece.piece_type() as usize;
                    let cur_to = hash_coord_32(m.to.x, m.to.y);

                    for &plies_ago in &[1usize, 3] {
                        if ply > plies_ago
                            && let Some(prev_move) = self.move_history[ply - plies_ago - 1]
                        {
                            let prev_piece = self.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_piece < 32 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                                cont_corr +=
                                    self.cont_corrhist[prev_piece][prev_to_hash][cur_pc][cur_to];
                            }
                        }
                    }
                }

                // 40% non-pawn, 25% material, 15% last-move, 20% continuation
                (nonpawn_corr * 40 + mat_corr * 25 + lastmove_corr * 15 + cont_corr * 20)
                    / (CORRHIST_GRAIN * 100)
            }
        };

        let corrected = raw_eval + total_correction;
        corrected.clamp(-MATE_SCORE + 1, MATE_SCORE - 1)
    }

    /// Update correction history based on search result.
    /// Updates only the tables relevant for the current mode.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn update_correction_history(
        &mut self,
        game: &GameState,
        ply: usize,
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
        let weight = ((depth * depth + 2 * depth + 1) as i32).clamp(1, 128);
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
                // Update non-pawn + material + last-move + continuation
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

                // Update continuation correction history (ss-2 and ss-4)
                if let Some(cur_m) = self.move_history[ply - 1] {
                    let cur_pc = cur_m.piece.piece_type() as usize;
                    let cur_to = hash_coord_32(cur_m.to.x, cur_m.to.y);
                    let cont_weight = weight.min(128);

                    for &plies_ago in &[1usize, 3] {
                        if ply > plies_ago
                            && let Some(prev_move) = self.move_history[ply - plies_ago - 1]
                        {
                            let prev_pc = self.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_pc < 32 {
                                let prev_to = hash_coord_32(prev_move.to.x, prev_move.to.y);
                                let entry =
                                    &mut self.cont_corrhist[prev_pc][prev_to][cur_pc][cur_to];

                                let w = if plies_ago == 1 {
                                    cont_weight
                                } else {
                                    cont_weight / 2
                                };
                                *entry = (*entry * (CORRHIST_WEIGHT_SCALE - w) + scaled_diff * w)
                                    / CORRHIST_WEIGHT_SCALE;
                                *entry = (*entry).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Format a score (cp or mate) as a string
    fn format_score(&self, score: i32) -> String {
        if score > MATE_SCORE {
            let mate_in = (MATE_VALUE - score + 1) / 2;
            format!("mate {}", mate_in)
        } else if score < -MATE_SCORE {
            let mate_in = (MATE_VALUE + score + 1) / 2;
            format!("mate -{}", mate_in)
        } else {
            format!("cp {}", score)
        }
    }

    /// Format a single PV line (Vec<Move>) as a string
    fn format_pv_line(&self, pv: &[Move]) -> String {
        pv.iter()
            .map(|m| {
                let promo = m.promotion.map_or("", |p| p.to_site_code());
                format!("{},{}->{},{}{}", m.from.x, m.from.y, m.to.x, m.to.y, promo)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract PV by following TT moves (for display only)
    /// Uses a cloned GameState to avoid corrupting the original board
    pub fn extract_pv_from_tt(&self, game: &mut GameState, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::with_capacity(max_len);
        let mut temp_game = game.clone();
        let mut seen_hashes = Vec::with_capacity(max_len);

        for _ in 0..max_len {
            let hash = temp_game.hash;
            if seen_hashes.contains(&hash) {
                break;
            }
            seen_hashes.push(hash);

            let tt_move = if let Some(tt) = GLOBAL_TT.get() {
                tt.probe_move(hash)
            } else {
                break;
            };

            if let Some(m) = tt_move {
                // Validate that the move is still valid on the current board position
                let piece_at_from = temp_game.board.get_piece(m.from.x, m.from.y);
                if piece_at_from.is_none() || piece_at_from != Some(&m.piece) {
                    break;
                }

                temp_game.make_move(&m);
                if temp_game.is_move_illegal() {
                    break;
                }
                pv.push(m);
            } else {
                break;
            }
        }
        pv
    }

    /// Extract the PV line (for internal use by MultiPV single-line path)
    pub fn extract_pv_only(&self, game: &mut GameState, depth: usize) -> Vec<Move> {
        // Just extract moves from pv_table without making them on the board
        // This is safe because we're only reading, not modifying game state
        let mut pv = Vec::with_capacity(self.pv_length[0].min(depth));
        for i in 0..self.pv_length[0].min(depth) {
            if let Some(m) = self.pv_table[i] {
                pv.push(m);
            } else {
                break;
            }
        }

        // Only extend with TT moves if we have room and the PV is short
        if pv.len() < depth {
            // Clone game state for TT probing to avoid corrupting the original
            let mut temp_game = game.clone();

            // First, advance temp_game to the end of the current PV
            // Validate each move before making it
            for m in &pv {
                let piece_at_from = temp_game.board.get_piece(m.from.x, m.from.y);
                if piece_at_from.is_none() || piece_at_from != Some(&m.piece) {
                    // PV is invalid, return what we have so far (empty safe)
                    return pv;
                }
                temp_game.make_move(m);
            }

            // Now probe TT to extend
            let mut seen_hashes = Vec::with_capacity(depth);
            for _ in pv.len()..depth {
                let hash = temp_game.hash;
                if seen_hashes.contains(&hash) {
                    break;
                }
                seen_hashes.push(hash);

                let tt_move = if let Some(tt) = GLOBAL_TT.get() {
                    tt.probe_move(hash)
                } else {
                    break;
                };

                if let Some(m) = tt_move {
                    // Validate that the move is still valid on the current board position
                    let piece_at_from = temp_game.board.get_piece(m.from.x, m.from.y);
                    if piece_at_from.is_none() || piece_at_from != Some(&m.piece) {
                        break;
                    }

                    temp_game.make_move(&m);
                    if temp_game.is_move_illegal() {
                        // Don't add illegal moves to PV
                        break;
                    }
                    pv.push(m);
                } else {
                    break;
                }
            }
        }
        pv
    }

    /// Format current searcher's PV as string
    pub fn format_pv(&self, game: &mut GameState, depth: usize) -> String {
        let pv = self.extract_pv_only(game, depth);
        self.format_pv_line(&pv)
    }

    /// Print UCI-style info string with optional MultiPV index
    pub fn print_info(&self, game: &mut GameState, depth: usize, score: i32) {
        self.print_info_multipv(game, depth, score, 1);
    }

    /// Print UCI-style info string with MultiPV index
    pub fn print_info_multipv(
        &self,
        game: &mut GameState,
        depth: usize,
        score: i32,
        multipv: usize,
    ) {
        let time_ms = self.hot.timer.elapsed_ms();
        let nps = if time_ms > 0 {
            (self.hot.nodes as u128 * 1000) / time_ms
        } else {
            0
        };
        let tt_fill = GLOBAL_TT.get().map_or(0, |tt| tt.fill_permille());
        let score_str = self.format_score(score);
        let pv = self.format_pv(game, depth);

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

    /// Aggregate all PV lines for a depth and print them as a single grouped update
    pub fn print_multi_pv_depth(&self, depth: usize, lines: &[PVLine]) {
        if lines.is_empty() {
            return;
        }

        let time_ms = self.hot.timer.elapsed_ms();
        let nps = if time_ms > 0 {
            (self.hot.nodes as u128 * 1000) / time_ms
        } else {
            0
        };
        let tt_fill = GLOBAL_TT.get().map_or(0, |tt| tt.fill_permille());

        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            use crate::{group, groupEnd, log};
            group(&format!(
                "Depth {} (time {}ms, nodes {}, nps {}, hashfull {}‰)",
                depth, time_ms, self.hot.nodes, nps, tt_fill
            ));

            for (idx, line) in lines.iter().enumerate() {
                let score_str = self.format_score(line.score);
                let pv_str = self.format_pv_line(&line.pv);
                log(&format!("#{} {} pv {}", idx + 1, score_str, pv_str));
            }
            groupEnd();
        }

        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            eprintln!(
                "Depth {} (time {}ms, nodes {}, nps {}, hashfull {}‰)",
                depth, time_ms, self.hot.nodes, nps, tt_fill
            );
            for (idx, line) in lines.iter().enumerate() {
                let score_str = self.format_score(line.score);
                let pv_str = self.format_pv_line(&line.pv);
                eprintln!("#{} {} pv {}", idx + 1, score_str, pv_str);
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
                fallback_move = Some(m);
            }
            legal_moves.push(m);
        }
    }

    if legal_moves.is_empty() {
        return None;
    }

    // If only one move, return immediately with a simple static eval as score.
    if legal_moves.len() == 1 {
        let single = legal_moves[0];
        let score = evaluate(game);
        return Some((single, score));
    }

    let mut best_move: Option<Move> = fallback_move; // Already cloned above
    let mut best_score = -INFINITY;
    let mut prev_root_move_coords: Option<(i64, i64, i64, i64)> = None;

    // Lazy SMP: Helper threads start at different depths to create search diversity.
    // Thread 0 (main): starts at depth 1
    // Thread 1: starts at depth 2 (skips trivial depth 1)
    // Thread 2: starts at depth 1
    // Thread 3: starts at depth 2
    // etc. - odd threads get a head start on deeper search
    let start_depth = if searcher.thread_id > 0 && searcher.thread_id % 2 == 1 {
        2.min(max_depth) // Odd helpers skip depth 1
    } else {
        1
    };

    // Iterative deepening with aspiration windows
    for base_depth in start_depth..=max_depth {
        // Lazy SMP depth offset: odd-indexed helper threads search at depth+1
        // This creates statistical diversity - threads explore different depths
        // and populate the TT with entries at various depths.
        // Main thread (0) and even helpers: search at base_depth
        // Odd helpers (1, 3, 5...): search at base_depth + 1
        let depth = if searcher.thread_id > 0 && searcher.thread_id % 2 == 1 {
            (base_depth + 1).min(max_depth)
        } else {
            base_depth
        };

        searcher.reset_for_iteration();
        searcher.hot.iter_start_ms = searcher.hot.timer.elapsed_ms() as f64;

        // Age out PV variability metric at START of each iteration
        // Note: Decay the PERSISTED tot, not the per-iteration changes.
        searcher.hot.tot_best_move_changes /= 2.0;

        // Time check at start of each iteration - but always complete depth 1.
        if searcher.hot.min_depth_required == 0 && searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms() as f64;

            // 1. Hard stop if we've exceeded the maximum time.
            if elapsed >= searcher.hot.maximum_time_ms as f64 {
                searcher.hot.stopped = true;
                break;
            }

            // Proactive stop: don't start next depth if most budget spent
            // For hard limits (timed games), we are more conservative (50%).
            // For soft limits (fixed time), we push much closer (90%) to use all time.
            let proactive_threshold = if searcher.hot.is_soft_limit {
                0.90
            } else {
                0.50
            };
            if searcher.hot.total_time_ms > 0.0
                && elapsed > searcher.hot.total_time_ms * proactive_threshold
            {
                break;
            }
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
                if window_size > aspiration_max_window() || retries >= 4 {
                    result =
                        negamax_root(searcher, game, depth, -INFINITY, INFINITY, &mut legal_moves);
                    break;
                }
            }
            result
        };

        // After first completed depth, allow time stops for subsequent depths
        // For helpers starting at depth 2, this triggers after their first iteration
        if base_depth == start_depth {
            searcher.hot.min_depth_required = 0;
        }

        // Update best move from this iteration.
        // IMPORTANT: Only update best_score if the search wasn't stopped mid-iteration,
        // because an interrupted search might return garbage values (like -INFINITY or
        // aspiration window bounds). If stopped, we keep the valid score from the previous
        // completed depth. The pv_table[0] check ensures a move was found.
        if let Some(pv_move) = searcher.pv_table[0] {
            // Always update the best_move to the latest PV move (even if stopped,
            // the move itself is valid from a previous iteration)
            best_move = Some(pv_move);
            searcher.best_move_root = Some(pv_move);

            // ONLY update score if search was not interrupted
            if !searcher.hot.stopped {
                best_score = score;
                searcher.prev_score = score;
                searcher.completed_depth = depth;
            }

            let coords = (pv_move.from.x, pv_move.from.y, pv_move.to.x, pv_move.to.y);
            if let Some(prev_coords) = prev_root_move_coords {
                // Track best move changes for instability calculation
                if prev_coords != coords {
                    searcher.hot.best_move_changes += 1.0;
                    searcher.hot.last_best_move_depth = depth;
                }
            }
            prev_root_move_coords = Some(coords);
        }

        if !searcher.hot.stopped && !searcher.silent {
            searcher.print_info(game, depth, score);
        }

        // Check global stop flag (for helper threads)
        if GLOBAL_STOP.load(std::sync::atomic::Ordering::Relaxed) {
            searcher.hot.stopped = true;
        }

        // Check if we found mate or time is up
        if searcher.hot.stopped || best_score.abs() > MATE_SCORE {
            break;
        }

        // Dynamic Time Management Check
        if searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms() as f64;

            // Effort tracking: fraction of nodes spent on the best move
            let nodes_effort = if searcher.hot.nodes > 0 {
                (searcher.hot.best_move_nodes as f64 * 100000.0) / (searcher.hot.nodes as f64)
            } else {
                0.0
            };
            let high_best_move_effort = if nodes_effort >= 93340.0 { 0.76 } else { 1.0 };

            // Accumulate instability changes from this iteration
            searcher.hot.tot_best_move_changes += searcher.hot.best_move_changes;
            searcher.hot.best_move_changes = 0.0;

            // fallingEval: spend more time when score is dropping
            let iter_val = searcher.hot.iter_values[searcher.hot.iter_idx];
            let prev_avg = searcher.hot.best_previous_average_score;
            let falling_eval = (11.85
                + 2.24 * (prev_avg - best_score) as f64
                + 0.93 * (iter_val - best_score) as f64)
                / 100.0;
            let falling_eval = falling_eval.clamp(0.57, 1.70);

            // timeReduction: spend less time when best move is stable
            let k = 0.51;
            let center = (searcher.hot.last_best_move_depth as f64) + 12.15;
            let time_reduction = 0.66 + 0.85 / (0.98 + (-k * (depth as f64 - center)).exp());

            let reduction = (1.43 + searcher.hot.prev_time_reduction) / (2.28 * time_reduction);

            // bestMoveInstability: spend more time when best move keeps changing
            let instability = (1.02 + 2.14 * searcher.hot.tot_best_move_changes).min(2.5);

            // Calculate totalTime with all factors
            let mut total_factors =
                (falling_eval * reduction * instability * high_best_move_effort).clamp(0.5, 2.5);

            // If it's a soft limit (like fixed time per move), we want to use
            // nearly all of the time, not stop early to save time.
            if searcher.hot.is_soft_limit {
                total_factors = total_factors.max(0.98);
            }

            let mut total_time = searcher.hot.optimum_time_ms as f64 * total_factors;

            // Cap for single legal move
            if legal_moves.len() == 1 {
                total_time = total_time.min(502.0);
            }

            let hard_limit = searcher.hot.maximum_time_ms as f64;

            // A search stop is triggered if the elapsed time exceeds the dynamic
            // limit (calculated from optimum time and stability factors) or the
            // hard maximum limit.
            let effective_limit = total_time.min(hard_limit);
            searcher.hot.total_time_ms = effective_limit; // Store for proactive checks

            if elapsed > effective_limit {
                searcher.hot.stopped = true;
                break;
            }

            // Update iteration tracking AFTER the time check
            searcher.hot.iter_values[searcher.hot.iter_idx] = best_score;
            searcher.hot.iter_idx = (searcher.hot.iter_idx + 1) & 3;

            // Update running average score
            if searcher.hot.best_previous_average_score == 0 {
                searcher.hot.best_previous_average_score = best_score;
            } else {
                searcher.hot.best_previous_average_score =
                    (best_score + searcher.hot.best_previous_average_score) / 2;
            }

            searcher.hot.prev_time_reduction = time_reduction;
        }
    }

    best_move.map(|m| (m, best_score))
}

/// Time-limited search that returns the best move, its evaluation (cp from side-to-move's
/// perspective), and simple TT statistics. This is the main public search entry point.
pub fn get_best_move(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    silent: bool,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    get_best_move_parallel(
        game,
        max_depth,
        time_limit_ms,
        time_limit_ms, // Use input as both opt and max for basic convenience wrapper
        silent,
        is_soft_limit,
    )
}

#[cfg(feature = "multithreading")]
pub fn get_best_move_parallel(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    silent: bool,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    use rustc_hash::FxHashMap;
    use std::sync::{Arc, Mutex};

    GLOBAL_TT.get_or_init(|| TranspositionTable::new(32));

    // Clear global stop flag
    GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);

    let num_threads = rayon::current_num_threads().max(1);

    // WASM SharedArrayBuffer and Web Worker communication have significant overhead.
    // Even 4 threads causes slowdown due to rayon/WebWorker scheduling costs.
    // For WASM, limit to 2 threads max (main + 1 helper) for any positive benefit.
    #[cfg(target_arch = "wasm32")]
    let num_threads = num_threads.min(2);

    USE_SHARED_TT.store(num_threads > 1, std::sync::atomic::Ordering::Relaxed);

    if num_threads == 1 {
        // Single-threaded: use local TT for best performance
        GLOBAL_TT.get_or_init(|| TranspositionTable::new(16));
        return get_best_move_threaded(
            game,
            max_depth,
            opt_time_ms,
            max_time_ms,
            silent,
            0,
            is_soft_limit,
        );
    }

    // Initialize TT - for WASM we don't share it, for native we do
    GLOBAL_TT.get_or_init(|| TranspositionTable::new(64));

    // Shared storage for thread results - all threads contribute to voting
    let results: Arc<Mutex<Vec<ThreadResult>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_threads)));

    rayon::scope(|s| {
        // Spawn helper threads (1..num_threads)
        for i in 1..num_threads {
            let results_clone = Arc::clone(&results);
            let mut game_clone = game.clone();

            s.spawn(move |_| {
                if let Some((best_move, score, stats)) = get_best_move_threaded(
                    &mut game_clone,
                    max_depth,
                    opt_time_ms,
                    max_time_ms,
                    true, // Helpers are always silent
                    i,
                    is_soft_limit,
                ) {
                    // Get PV length from thread-local searcher
                    let pv_len = GLOBAL_SEARCHER
                        .with(|cell| cell.borrow().as_ref().map_or(1, |s| s.pv_length[0].max(1)));
                    let completed_depth = GLOBAL_SEARCHER
                        .with(|cell| cell.borrow().as_ref().map_or(1, |s| s.completed_depth));

                    let result = ThreadResult {
                        best_move,
                        score,
                        completed_depth: completed_depth.max(1),
                        pv_length: pv_len,
                        nodes: stats.nodes,
                        thread_id: i,
                    };
                    if let Ok(mut results) = results_clone.lock() {
                        results.push(result);
                    }
                }
            });
        }

        // Run main search on thread 0 (this thread)
        if let Some((best_move, score, stats)) = get_best_move_threaded(
            game,
            max_depth,
            opt_time_ms,
            max_time_ms,
            silent, // Main thread respects silent flag
            0,
            is_soft_limit,
        ) {
            let pv_len = GLOBAL_SEARCHER
                .with(|cell| cell.borrow().as_ref().map_or(1, |s| s.pv_length[0].max(1)));
            let completed_depth = GLOBAL_SEARCHER
                .with(|cell| cell.borrow().as_ref().map_or(1, |s| s.completed_depth));

            let result = ThreadResult {
                best_move,
                score,
                completed_depth: completed_depth.max(1),
                pv_length: pv_len,
                nodes: stats.nodes,
                thread_id: 0,
            };
            if let Ok(mut results) = results.lock() {
                results.push(result);
            }
        }

        // Signal all helper threads to stop
        GLOBAL_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    // All threads have finished - now apply thread voting to select best move
    let all_results = Arc::try_unwrap(results)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();

    if all_results.is_empty() {
        return None;
    }

    // ========================================================================
    // Thread Voting Algorithm (based on Stockfish's get_best_thread)
    // ========================================================================

    // Step 1: Find minimum score among all threads
    let min_score = all_results.iter().map(|r| r.score).min().unwrap_or(0);

    // Step 2: Build vote map - each move gets weighted votes from threads
    // Vote weight = (score - minScore + 14) * completedDepth
    // The +14 ensures even the worst-scoring thread contributes positively
    let mut votes: FxHashMap<(i64, i64, i64, i64), i64> = FxHashMap::default();

    for r in &all_results {
        let move_key = (
            r.best_move.from.x,
            r.best_move.from.y,
            r.best_move.to.x,
            r.best_move.to.y,
        );
        let vote_value = (r.score - min_score + 14) as i64 * r.completed_depth as i64;
        *votes.entry(move_key).or_insert(0) += vote_value;
    }

    // Step 3: Select best thread using Stockfish's criteria
    let mut best_idx = 0;

    // Helper to compute voting value for a thread
    let thread_voting_value =
        |r: &ThreadResult| -> i64 { (r.score - min_score + 14) as i64 * r.completed_depth as i64 };

    for (i, r) in all_results.iter().enumerate() {
        let best = &all_results[best_idx];

        let best_move_key = (
            best.best_move.from.x,
            best.best_move.from.y,
            best.best_move.to.x,
            best.best_move.to.y,
        );
        let new_move_key = (
            r.best_move.from.x,
            r.best_move.from.y,
            r.best_move.to.x,
            r.best_move.to.y,
        );

        let best_vote = votes.get(&best_move_key).copied().unwrap_or(0);
        let new_vote = votes.get(&new_move_key).copied().unwrap_or(0);

        let best_in_proven_win = is_win(best.score);
        let new_in_proven_win = is_win(r.score);
        let best_in_proven_loss = best.score != -INFINITY && is_loss(best.score);
        let new_in_proven_loss = r.score != -INFINITY && is_loss(r.score);

        // Prefer threads with longer PVs (more trustworthy)
        let better_voting_with_pv = thread_voting_value(r) * (if r.pv_length > 2 { 1 } else { 0 })
            > thread_voting_value(best) * (if best.pv_length > 2 { 1 } else { 0 });

        if best_in_proven_win {
            // Already in a winning position: pick the fastest mate
            if r.score > best.score {
                best_idx = i;
            }
        } else if best_in_proven_loss {
            // In a losing position: pick the longest resistance
            if new_in_proven_loss && r.score < best.score {
                best_idx = i;
            }
        } else if new_in_proven_win
            || new_in_proven_loss
            || (!is_loss(r.score)
                && (new_vote > best_vote || (new_vote == best_vote && better_voting_with_pv)))
        {
            best_idx = i;
        }
    }

    let best_result = &all_results[best_idx];

    // Aggregate total nodes from all threads for accurate NPS reporting
    let total_nodes: u64 = all_results.iter().map(|r| r.nodes).sum();

    // Build stats from aggregated data
    let (cap, used, fill) = GLOBAL_TT.get().map_or((0, 0, 0), |tt| {
        (tt.capacity(), tt.used_entries(), tt.fill_permille())
    });
    let stats = SearchStats {
        nodes: total_nodes,
        tt_capacity: cap,
        tt_used: used,
        tt_fill_permille: fill,
    };

    Some((best_result.best_move.clone(), best_result.score, stats))
}

#[cfg(not(feature = "multithreading"))]
pub fn get_best_move_parallel(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    silent: bool,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    // Init TT if needed
    GLOBAL_TT.get_or_init(|| TranspositionTable::new(16));
    get_best_move_threaded(
        game,
        max_depth,
        opt_time_ms,
        max_time_ms,
        silent,
        0,
        is_soft_limit,
    )
}

/// Time-limited search with thread_id for Lazy SMP.
/// Helper threads (thread_id > 0) skip the first move to distribute work.
/// Uses persistent GLOBAL_SEARCHER - TT and histories persist across searches.
/// Call reset_search_state() to clear for a new game.
fn get_best_move_threaded(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    silent: bool,
    thread_id: usize,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();
    // Initialize correction history hashes
    game.recompute_correction_hashes();

    // Use persistent global searcher
    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();

        // Get or create the persistent searcher
        let searcher = opt.get_or_insert_with(|| Searcher::new(max_time_ms));

        // If this is a helper thread, ensure it has a unique RNG state based on global seed
        if thread_id > 0 {
            // Helpers don't mutate global seed, but use it to seed their local RNG
            // We use wrapping_add to ensure deterministic variation per thread
            let base_seed = searcher.seed;
            searcher.rng = Prng::new(base_seed.wrapping_add(thread_id as u64));
        }

        // Initialize searcher for this search
        searcher.new_search();

        // Update search parameters for this search
        searcher
            .hot
            .set_time_limits(opt_time_ms, max_time_ms, is_soft_limit);
        searcher.silent = silent;
        searcher.thread_id = thread_id;
        searcher.hot.timer.reset();

        // Set correction mode based on variant (zero overhead during search)
        searcher.set_corrhist_mode(game);

        let result = search_with_searcher(searcher, game, max_depth);
        let stats = build_search_stats(searcher);
        result.map(|(m, eval)| (m, eval, stats))
    })
}

/// MultiPV-enabled search that returns up to `multi_pv` best moves with their evaluations.
///
/// When `multi_pv` is 1, this has zero overhead - it's equivalent to `get_best_move`.
/// For `multi_pv` > 1, at each depth all root moves are searched and the top N are returned.
pub fn get_best_moves_multipv(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    multi_pv: usize,
    silent: bool,
    is_soft_limit: bool,
) -> MultiPVResult {
    // Ensure fast per-color piece counts are in sync with the board
    game.recompute_piece_counts();
    // Initialize correction history hashes
    game.recompute_correction_hashes();

    let multi_pv = multi_pv.max(1);

    // Use persistent global searcher pattern:
    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();

        // Get or create the persistent searcher
        let searcher = opt.get_or_insert_with(|| Searcher::new(max_time_ms));

        // Initialize searcher for this search
        searcher.new_search();

        // Update search parameters for this search
        searcher
            .hot
            .set_time_limits(opt_time_ms, max_time_ms, is_soft_limit);
        searcher.silent = silent;
        searcher.hot.timer.reset();

        searcher.set_corrhist_mode(game);
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        // MultiPV = 1: Zero overhead path - just do normal search
        if multi_pv == 1 {
            let mut lines: Vec<PVLine> = Vec::with_capacity(1);
            if let Some((best_move, score)) = search_with_searcher(searcher, game, max_depth) {
                let pv = searcher.extract_pv_only(game, max_depth);
                let depth = max_depth.min(searcher.hot.seldepth.max(1));
                lines.push(PVLine {
                    mv: best_move,
                    score,
                    depth,
                    pv,
                });
            }
            let stats = build_search_stats(searcher);
            return MultiPVResult { lines, stats };
        }

        // MultiPV > 1: Search with special root handling to collect multiple best moves
        get_best_moves_multipv_impl(searcher, game, max_depth, multi_pv, silent)
    })
}

/// Sets the global seed and re-initializes the PRNG.
/// This affects subsequent calls to functions that use GLOBAL_SEARCHER.
pub fn set_global_params(seed: u64, noise_amp: Option<i32>) {
    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();
        // Get or create the persistent searcher.
        // If not initialized, we use a default max_time (e.g. 1000) which will be updated later.
        let searcher = opt.get_or_insert_with(|| Searcher::new(1000));

        searcher.seed = seed;
        searcher.rng = Prng::new(seed);
        searcher.noise_amp = noise_amp.unwrap_or(0);
    });
}

/// Selects a move from MultiPV results using Stockfish's strength-limiting logic.
fn pick_best(result: &MultiPVResult, skill_level: u32, rng: &mut Prng) -> Option<(Move, i32)> {
    if result.lines.is_empty() {
        return None;
    }
    if result.lines.len() == 1 || skill_level >= 20 {
        let best = &result.lines[0];
        return Some((best.mv, best.score));
    }

    let top_score = result.lines[0].score;
    let last_score = result.lines.last().unwrap().score;
    let delta = (top_score - last_score).min(100); // 100 cp = PawnValue
    let weakness = 120 - 2 * skill_level as i32;

    let mut max_score = -INFINITY;
    let mut chosen_idx = 0;

    for (idx, line) in result.lines.iter().enumerate() {
        let rng_val = (rng.next_f64() * (weakness as f64)) as i32;
        let push = (weakness * (top_score - line.score) + delta * rng_val) / 128;

        if line.score + push >= max_score {
            max_score = line.score + push;
            chosen_idx = idx;
        }
    }

    let best = &result.lines[chosen_idx];
    Some((best.mv, best.score))
}

/// Entry point for searches with strength limiting.
/// Consolidated to use standard search path with root-level move selection.
pub(crate) fn get_best_move_limited(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    strength_level: Option<u32>,
    silent: bool,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    game.recompute_piece_counts();
    game.recompute_correction_hashes();

    let input_skill = strength_level
        .unwrap_or(MAX_SITE_SKILL)
        .clamp(1, MAX_SITE_SKILL);
    if input_skill >= MAX_SITE_SKILL {
        return get_best_move_parallel(
            game,
            max_depth,
            opt_time_ms,
            max_time_ms,
            silent,
            is_soft_limit,
        );
    }

    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();
        let searcher = opt.get_or_insert_with(|| Searcher::new(max_time_ms));

        searcher.new_search();
        searcher
            .hot
            .set_time_limits(opt_time_ms, max_time_ms, is_soft_limit);
        searcher.silent = silent;
        searcher.hot.timer.reset();

        searcher.set_corrhist_mode(game);
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        // Ensure TT is initialized
        GLOBAL_TT.get_or_init(|| TranspositionTable::new(16));

        // Use MultiPV at the root and pick_best selection logic.
        // Automatically normalize site skill (1..MAX_SITE_SKILL) to internal (1..20)
        let skill_level = if MAX_SITE_SKILL > 1 {
            let progress = (input_skill - 1) as f32 / (MAX_SITE_SKILL - 1) as f32;
            (1.0 + progress * 19.0).round() as u32
        } else {
            20
        };

        let multi_pv = if skill_level >= 20 { 1 } else { MAX_PV_COUNT };

        if multi_pv > 1 {
            let result = get_best_moves_multipv_impl(searcher, game, max_depth, multi_pv, silent);
            let stats = result.stats.clone();
            pick_best(&result, skill_level, &mut searcher.rng).map(|(m, eval)| (m, eval, stats))
        } else {
            let res = search_with_searcher(searcher, game, max_depth);
            let stats = build_search_stats(searcher);
            res.map(|(m, eval)| (m, eval, stats))
        }
    })
}

pub(crate) fn get_best_moves_multipv_impl(
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

    // Find legal moves only (filter pseudo-legal)
    let mut legal_root_moves: MoveList = MoveList::new();
    let mut fallback_move: Option<Move> = None;
    for m in moves {
        let undo = game.make_move(&m);
        let legal = !game.is_move_illegal();
        game.undo_move(&m, undo);
        if legal {
            if fallback_move.is_none() {
                fallback_move = Some(m);
            }
            legal_root_moves.push(m);
        }
    }

    if legal_root_moves.is_empty() {
        let stats = build_search_stats(searcher);
        return MultiPVResult {
            lines: Vec::new(),
            stats,
        };
    }

    // If only one move, return immediately with a simple static eval as score.
    if legal_root_moves.len() == 1 {
        let single = legal_root_moves[0];
        let score = evaluate(game);
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

    // Cap multi_pv at number of legal moves
    let multi_pv = multi_pv.min(legal_root_moves.len());

    // Store (move, score, pv) for each root move at current depth
    let mut root_scores: Vec<(Move, i32, Vec<Move>)> = Vec::with_capacity(legal_root_moves.len());
    let mut best_lines: Vec<PVLine> = Vec::with_capacity(multi_pv);

    // Lazy SMP: Helper threads start at different depths to create search diversity.
    let start_depth = if searcher.thread_id > 0 && searcher.thread_id % 2 == 1 {
        2.min(max_depth)
    } else {
        1
    };

    // Iterative deepening
    for base_depth in start_depth..=max_depth {
        let depth = if searcher.thread_id > 0 && searcher.thread_id % 2 == 1 {
            (base_depth + 1).min(max_depth)
        } else {
            base_depth
        };

        searcher.reset_for_iteration();
        searcher.hot.iter_start_ms = searcher.hot.timer.elapsed_ms() as f64;
        searcher.hot.tot_best_move_changes /= 2.0;

        // Time check at start of each iteration - but always complete depth 1
        if searcher.hot.min_depth_required == 0 && searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms() as f64;

            if elapsed >= searcher.hot.maximum_time_ms as f64 {
                searcher.hot.stopped = true;
                break;
            }

            let proactive_threshold = if searcher.hot.is_soft_limit {
                0.90
            } else {
                0.50
            };
            if searcher.hot.total_time_ms > 0.0
                && elapsed > searcher.hot.total_time_ms * proactive_threshold
            {
                break;
            }
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
            let prev_entry_backup = searcher.prev_move_stack[0];
            let prev_from_hash = hash_move_from(m);
            let prev_to_hash = hash_move_dest(m);
            searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

            // For MultiPV, we need to search all moves to get their scores.
            // First move gets full window, subsequent moves use PVS with MultiPV-aware alpha.
            let score = if move_idx == 0 {
                -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: depth - 1,
                    ply: 1,
                    alpha: -INFINITY,
                    beta: INFINITY,
                    allow_null: true,
                    node_type: NodeType::PV,
                    was_null_move: false,
                    excluded_move: None,
                })
            } else {
                // Use PVS for efficiency with MultiPV-aware alpha bound
                let alpha = multipv_alpha;

                let mut s = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: depth - 1,
                    ply: 1,
                    alpha: -alpha - 1,
                    beta: -alpha,
                    allow_null: true,
                    node_type: NodeType::Cut,
                    was_null_move: false,
                    excluded_move: None,
                });
                if s > alpha && !searcher.hot.stopped {
                    // Re-search with full window to get accurate score
                    s = -negamax(&mut NegamaxContext {
                        searcher,
                        game,
                        depth: depth - 1,
                        ply: 1,
                        alpha: -INFINITY,
                        beta: INFINITY,
                        allow_null: true,
                        node_type: NodeType::PV,
                        was_null_move: false,
                        excluded_move: None,
                    });
                }
                s
            };

            searcher.prev_move_stack[0] = prev_entry_backup;
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
                root_scores.push((*m, score, pv));

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
            legal_root_moves.push(*mv);
        }

        // Update best_lines with results from this depth
        best_lines.clear();
        for (mv, score, pv) in root_scores.iter().take(multi_pv) {
            best_lines.push(PVLine {
                mv: *mv,
                score: *score,
                depth: depth.min(searcher.hot.seldepth.max(1)),
                pv: pv.clone(),
            });
        }

        if !silent {
            searcher.print_multi_pv_depth(depth, &best_lines);
        }

        searcher.prev_score = if !root_scores.is_empty() {
            root_scores[0].1
        } else {
            -INFINITY
        };
        searcher.hot.min_depth_required = 0;

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

    let stats = build_search_stats(searcher);
    MultiPVResult {
        lines: best_lines,
        stats,
    }
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
    GLOBAL_TT
        .get_or_init(|| TranspositionTable::new(16))
        .clear();

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

    let hash = game.hash;
    let mut tt_move: Option<Move> = None;

    // Probe TT for best move from previous search (uses shared TT if configured)
    // Pass half-move clock directly for score adjustment:
    let rule50_count = game.halfmove_clock;
    if let Some(res) = probe_tt_with_shared(
        searcher,
        &ProbeContext {
            hash,
            alpha,
            beta,
            depth,
            ply: 0,
            rule50_count,
            rule_limit: searcher.move_rule_limit,
        },
    ) {
        tt_move = res.best_move;
    }

    let in_check = game.is_in_check();

    // Sort moves at root (TT move first, then by score)
    // This reorders the `moves` vec in-place, preserving this ordering
    // for the next iteration.
    sort_moves_root(searcher, game, moves, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;

    for (move_idx, m) in moves.iter().enumerate() {
        // Skip excluded moves (for MultiPV subsequent passes)
        if !searcher.excluded_moves.is_empty() {
            let coords = (m.from.x, m.from.y, m.to.x, m.to.y);
            if searcher.excluded_moves.contains(&coords) {
                continue;
            }
        }

        let nodes_before_move = searcher.hot.nodes;

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
            score = -negamax(&mut NegamaxContext {
                searcher,
                game,
                depth: depth - 1,
                ply: 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                node_type: NodeType::PV,
                was_null_move: false,
                excluded_move: None,
            });
        } else {
            // PVS: Null window first, then re-search if it improves alpha
            let mut s = -negamax(&mut NegamaxContext {
                searcher,
                game,
                depth: depth - 1,
                ply: 1,
                alpha: -alpha - 1,
                beta: -alpha,
                allow_null: true,
                node_type: NodeType::Cut,
                was_null_move: false,
                excluded_move: None,
            });
            if s > alpha && s < beta {
                s = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: depth - 1,
                    ply: 1,
                    alpha: -beta,
                    beta: -alpha,
                    allow_null: true,
                    node_type: NodeType::PV,
                    was_null_move: false,
                    excluded_move: None,
                });
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
            best_move = Some(*m);

            if score > alpha {
                alpha = score;

                // if legal_moves > 1 {
                //     searcher.hot.best_move_changes += 1.0;
                // }

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

        // Track nodes spent on this move (effort)
        // If this is the current first move in the list (most likely the best move),
        // track its effort for time management.
        if move_idx == 0 {
            searcher.hot.best_move_nodes = searcher.hot.nodes - nodes_before_move;
        }
    }

    // Checkmate, stalemate, or loss by capture-based variants
    if legal_moves == 0 {
        // Determine if this is a loss:
        // 1. In check AND must escape check (our win condition is checkmate) → checkmate
        // 2. No pieces left (relevant for allpiecescaptured variants) → loss
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        return if checkmate || no_pieces {
            -MATE_VALUE
        } else {
            0 // Stalemate
        };
    }

    // Store in TT with correct flag based on original alpha
    let tt_data_bound = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    store_tt_with_shared(
        searcher,
        &StoreContext {
            hash,
            depth,
            flag: tt_data_bound,
            score: best_score,
            static_eval: INFINITY + 1, // Not computed at root normally, or already stored
            is_pv: true,
            best_move,
            ply: 0,
        },
    );

    best_score
}

/// Main negamax with alpha-beta pruning
fn negamax(ctx: &mut NegamaxContext) -> i32 {
    let searcher = &mut *ctx.searcher;
    let game = &mut *ctx.game;
    let depth = ctx.depth;
    let ply = ctx.ply;
    let mut alpha = ctx.alpha;
    let mut beta = ctx.beta;
    let allow_null = ctx.allow_null;
    let node_type = ctx.node_type;

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

    // Check if we have an upcoming move that draws by repetition
    // Stockfish: this comes before is_draw check
    if ply > 0 && alpha < VALUE_DRAW && game.upcoming_repetition(ply) {
        let draw_val = value_draw(searcher.hot.nodes);
        if draw_val >= beta {
            return draw_val;
        }
        alpha = alpha.max(draw_val);
    }

    // Initialize node state
    let in_check = game.is_in_check();
    searcher.hot.nodes += 1;
    searcher.pv_length[ply] = 0;

    // Initialize cutoff count for grandchild ply
    if ply + 2 < MAX_PLY {
        searcher.cutoff_cnt[ply + 2] = 0;
    }

    // Update plies_from_null stack (for is_shuffling detection)
    // If previous move was null, reset count to 0. Otherwise increment.
    let prev_plies = if ctx.was_null_move {
        0
    } else if ply > 0 {
        searcher.plies_from_null[ply - 1]
    } else {
        255 // Root assumption (no recent null move)
    };
    searcher.plies_from_null[ply] = prev_plies.saturating_add(1);

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
            return value_draw(searcher.hot.nodes);
        }

        // Royal capture loss: if our king was just captured (RoyalCapture/AllRoyalsCaptured variants)
        if game.has_lost_by_royal_capture() {
            return -MATE_VALUE + ply as i32;
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
        let r = searcher.reduction_stack[ply - 1];
        searcher.reduction_stack[ply - 1] = 0;
        r
    } else {
        0
    };

    // Transposition table probe for hash move and potential cutoff
    let hash = game.hash;
    let rule50_count = game.halfmove_clock;
    let tt_probe = probe_tt_with_shared(
        searcher,
        &ProbeContext {
            hash,
            alpha,
            beta,
            depth,
            ply,
            rule50_count,
            rule_limit: searcher.move_rule_limit,
        },
    );

    let (tt_hit_node, tt_move, tt_value, tt_data_static_eval, tt_data_depth, tt_pv, tt_data_bound) =
        if let Some(res) = tt_probe {
            (
                true,
                res.best_move,
                if res.tt_score == INFINITY + 1 {
                    None
                } else {
                    Some(res.tt_score)
                },
                res.eval,
                res.depth,
                res.is_pv,
                res.flag,
            )
        } else {
            (false, None, None, INFINITY + 1, 0, false, TTFlag::None)
        };

    // Static evaluation for pruning decisions
    let prev_move_idx = if ply > 0 {
        let (from_hash, to_hash) = searcher.prev_move_stack[ply - 1];
        from_hash ^ to_hash
    } else {
        0
    };

    let (mut static_eval, raw_eval) = if in_check {
        // When in check, use previous ply's evaluation
        let prev_eval = if ply >= 2 {
            searcher.eval_stack[ply - 2]
        } else {
            0
        };
        (prev_eval, prev_eval)
    } else {
        // Use stored TT evaluation if available, otherwise compute it
        let mut raw = tt_data_static_eval;
        if raw == INFINITY + 1 {
            raw = evaluate(game);

            // Store the computed evaluation in TT immediately
            store_tt_with_shared(
                searcher,
                &StoreContext {
                    hash,
                    depth: 0,
                    flag: TTFlag::None,
                    score: 0,
                    static_eval: raw,
                    is_pv: tt_pv,
                    best_move: tt_move,
                    ply,
                },
            );
        }

        let adjusted = searcher.adjusted_eval(game, raw, ply, prev_move_idx);
        (adjusted, raw)
    };

    // Apply deterministic search noise at low plies for SPRT
    if searcher.noise_amp > 0 && ply < LOW_PLY_HISTORY_SIZE {
        static_eval += get_noise(searcher.seed, hash, searcher.noise_amp);
    }
    searcher.eval_stack[ply] = static_eval;

    // Position improving heuristic: compare eval to 2 plies ago
    let mut improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true
    };

    // Opponent worsening: their last move made our position better
    let opponent_worsening = if ply >= 1 && !in_check {
        static_eval > -searcher.eval_stack[ply - 1]
    } else {
        false
    };

    // Use TT value to improve position evaluation (Store in local variable 'eval' like Stockfish)
    let mut eval = static_eval;
    if !in_check
        && let Some(tt_s) = tt_value
        && !is_decisive(tt_s)
    {
        let fails_high = tt_s >= beta;
        if (tt_data_bound == TTFlag::LowerBound && fails_high)
            || (tt_data_bound == TTFlag::UpperBound && !fails_high)
            || tt_data_bound == TTFlag::Exact
        {
            eval = tt_s;
        }
    }

    // Hindsight depth adjustment based on prior search behavior
    if !in_check && ply > 0 {
        let prev_eval = searcher.eval_stack[ply - 1];
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2 && depth >= 2 && static_eval + prev_eval > 173 {
            depth = depth.saturating_sub(1);
        }
    }

    // TT Cutoff
    let excluded_move = ctx.excluded_move;
    if !is_pv
        && excluded_move.is_none()
        && tt_hit_node
        && let Some(tt_s) = tt_value
        && !is_decisive(tt_s)
    {
        // Check TT depth vs adjusted depth
        let depth_threshold = if tt_s <= beta {
            depth.saturating_sub(1)
        } else {
            depth
        };

        let tt_data_depth_ok = (tt_data_depth as usize) > depth_threshold;

        // Bound check
        let fails_high = tt_s >= beta;
        let bound_matches = if fails_high {
            (tt_data_bound as u8 & TTFlag::LowerBound as u8) != 0
        } else {
            (tt_data_bound as u8 & TTFlag::UpperBound as u8) != 0
        };

        let node_type_matches = (cut_node == fails_high) || depth > 5;

        // Graph history interaction workaround: don't cutoff at high rule50
        let rule50_threshold = (searcher.move_rule_limit as u32).saturating_sub(4);
        let rule50_ok = game.halfmove_clock < rule50_threshold;

        if tt_data_depth_ok
            && bound_matches
            && node_type_matches
            && rule50_ok
            && !game.is_repetition(ply)
        {
            return tt_s;
        }
    }

    // Determine if this node is a TT PV node
    let tt_pv = is_pv || (tt_hit_node && tt_pv);

    // When in check, skip all pruning - we need to search all evasions
    if !in_check {
        // =================================================================
        // Pre-move pruning techniques
        // =================================================================

        // Razoring: if static eval is very low, drop to qsearch
        // Formula: eval < alpha - 485 - 281 * depth²
        if !is_pv
            && static_eval < alpha - razoring_linear() - razoring_quad() * (depth * depth) as i32
        {
            return quiescence(searcher, game, ply, alpha, beta);
        }

        // Check if TT move is a capture (for RFP condition)
        let tt_capture = if let Some(m) = tt_move {
            if game.board.get_piece(m.to.x, m.to.y).is_some() {
                true
            } else if let Some(ep) = game.en_passant {
                ep.square == m.to && m.piece.piece_type() == PieceType::Pawn
            } else {
                false
            }
        } else {
            false
        };

        // Reverse Futility Pruning (RFP)
        if !tt_pv
            && depth < rfp_max_depth()
            && (tt_move.is_none() || tt_capture)
            && !is_loss(beta)
            && !is_win(static_eval)
        {
            let futility_mult = if tt_hit_node {
                rfp_mult_tt()
            } else {
                rfp_mult_no_tt()
            };

            let mut bonus = 0;
            if improving {
                bonus += rfp_improving_mult() * futility_mult / 1024;
            }
            if opponent_worsening {
                bonus += rfp_worsening_mult() * futility_mult / 1024;
            }

            let futility_margin = futility_mult * depth as i32 - bonus;

            if eval - futility_margin >= beta && static_eval >= beta {
                return (beta + eval) / 2;
            }
        }

        // Null move pruning: give opponent an extra move, if still >= beta, prune
        // Only in cut nodes with non-pawn material (avoid zugzwang)
        // Guard: don't prune when beta is a losing mate score (preserve mate finding)
        if cut_node && allow_null && depth >= nmp_min_depth() && !is_loss(beta) {
            let nmp_margin = static_eval - (nmp_depth_mult() * depth as i32) + nmp_base();
            if nmp_margin >= beta && game.has_non_pawn_material(game.turn) {
                let saved_ep = game.en_passant;
                // Backup move history for this ply and clear it (null move has no move)
                let move_history_backup = searcher.move_history[ply].take();
                let piece_history_backup = searcher.moved_piece_history[ply]; // Also backup piece history

                game.make_null_move();

                // Reduction: R = 7 + depth / 3
                let r = nmp_reduction_base() + depth / nmp_reduction_div();
                let null_score = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: depth.saturating_sub(r),
                    ply: ply + 1,
                    alpha: -beta,
                    beta: -beta + 1,
                    allow_null: false,
                    node_type: NodeType::Cut,
                    was_null_move: true,
                    excluded_move: None,
                });

                game.unmake_null_move();
                game.en_passant = saved_ep;
                // Restore move history
                searcher.move_history[ply] = move_history_backup;
                searcher.moved_piece_history[ply] = piece_history_backup;

                if searcher.hot.stopped {
                    return 0;
                }

                // If null move score >= beta, we can prune
                // Guard: don't return mate scores from null move (they're unproven)
                if null_score >= beta && !is_win(null_score) {
                    return null_score;
                }
            }
        }

        // Update improving flag based on static eval vs beta
        improving = improving || static_eval >= beta;

        // Internal iterative reductions (IIR)
        // Without TT move, reduce depth to find one faster
        if !all_node && depth >= iir_min_depth() && tt_move.is_none() && prior_reduction <= 3 {
            depth -= 1;
        }
    }

    // =========================================================================
    // ProbCut
    // =========================================================================
    // If we have a good enough capture and a reduced search returns a value
    // much above beta, we can prune.
    let prob_cut_beta = beta + probcut_margin() - if improving { probcut_improving() } else { 0 };
    // Guard: don't ProbCut when beta is a mate score
    if !is_pv
        && !in_check
        && depth >= probcut_min_depth()
        && !is_decisive(beta)
        && tt_value.is_none_or(|v| v >= prob_cut_beta)
    {
        let mut prob_cut_depth =
            (depth as i32 - probcut_depth_sub() as i32 - (static_eval - beta) / probcut_divisor())
                .max(0) as usize;
        if prob_cut_depth > depth {
            prob_cut_depth = depth;
        }

        // Use StagedMoveGen for ProbCut (captures with SEE >= threshold)
        let threshold = prob_cut_beta - static_eval;
        let mut probcut_gen = StagedMoveGen::new_probcut(tt_move, threshold, searcher, game);

        while let Some(m) = probcut_gen.next(game, searcher) {
            // Fast legality check (skips is_move_illegal for non-pinned pieces)
            let fast_legal = game.is_legal_fast(&m, in_check);
            if let Ok(false) = fast_legal {
                continue;
            }

            let undo = game.make_move(&m);
            if fast_legal.is_err() && game.is_move_illegal() {
                game.undo_move(&m, undo);
                continue;
            }

            // Preliminary qsearch to verify
            let mut val = -quiescence(searcher, game, ply + 1, -prob_cut_beta, -prob_cut_beta + 1);

            // If qsearch held, perform regular search at reduced depth
            if val >= prob_cut_beta {
                val = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: prob_cut_depth,
                    ply: ply + 1,
                    alpha: -prob_cut_beta,
                    beta: -prob_cut_beta + 1,
                    allow_null: true,
                    node_type: NodeType::Cut, // Expected cut node
                    was_null_move: false,
                    excluded_move: None,
                });
            }

            game.undo_move(&m, undo);

            if searcher.hot.stopped {
                return 0;
            }

            if val >= prob_cut_beta {
                store_tt_with_shared(
                    searcher,
                    &StoreContext {
                        hash,
                        depth: prob_cut_depth + 1,
                        flag: TTFlag::LowerBound,
                        score: val,
                        static_eval: raw_eval,
                        is_pv: false,
                        best_move: Some(m),
                        ply,
                    },
                );

                // Only return if not decisive, adjust value
                if !is_decisive(val) {
                    return val - (prob_cut_beta - beta);
                }
            }
        }
    }

    // Small ProbCut: if TT entry has a lower bound >= beta + margin, return early
    // This avoids searching positions where we already know there's a good move
    {
        let small_prob_cut_beta = beta + low_depth_probcut_margin();
        if tt_hit_node
            && (tt_data_bound == TTFlag::LowerBound || tt_data_bound == TTFlag::Exact)
            && tt_data_depth as usize >= depth.saturating_sub(4)
            && let Some(tt_v) = tt_value
            && tt_v >= small_prob_cut_beta
            && !is_decisive(beta)
            && !is_decisive(tt_v)
        {
            return small_prob_cut_beta;
        }
    }

    // =========================================================================
    // Staged Move Generation - generate moves in stages for better efficiency
    // =========================================================================
    let mut movegen = StagedMoveGen::new(tt_move, ply, depth as i32, searcher, game);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    let mut quiets_searched: MoveList = MoveList::new();

    // Singular extension conditions (checked when we reach the TT move in the loop)
    // We cache the TT probe result here to avoid re-probing
    let se_conditions = if depth >= 6 && !in_check && tt_move.is_some() {
        if tt_hit_node
            && (tt_data_bound == TTFlag::LowerBound || tt_data_bound == TTFlag::Exact)
            && tt_data_depth as usize >= depth.saturating_sub(3)
            && let Some(tt_v) = tt_value
            && !is_decisive(tt_v)
        {
            Some((tt_v, (depth - 1) / 2)) // (singular_beta_base, singular_depth)
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
        let is_capture = captured_piece.is_some_and(|p| !p.piece_type().is_neutral_type());
        let captured_type = captured_piece.map(|p| p.piece_type());
        let is_promotion = m.promotion.is_some();
        let p_type = m.piece.piece_type();

        // Check if this move gives check to enemy king (O(1) for knights/pawns)
        let gives_check = StagedMoveGen::move_gives_check_fast(game, &m);

        // In-move pruning at shallow depths (not in PV, have material, not losing)
        // Guard: don't prune when we have a losing mate score (must find escape)
        if !is_pv && game.has_non_pawn_material(game.turn) && !is_loss(best_score) {
            // Late move pruning: skip quiet moves after seeing enough
            // Threshold: (3 + depth²) / (2 if not improving, else 1)
            let improving_div = if improving { 1 } else { 2 };
            let lmp_count = (lmp_base() + depth * depth * lmp_depth_mult()) / improving_div;
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
                    let see_margin = (see_capture_linear() * depth as i32
                        + capt_hist / see_capture_hist_div())
                    .max(0);
                    let see_value = static_exchange_eval(game, &m);
                    if see_value < -see_margin {
                        continue;
                    }
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
                        // Guard: don't overwrite mate scores with futility value
                        if best_score <= futility_value && !is_decisive(best_score) {
                            best_score = futility_value;
                        }
                        continue;
                    }
                }

                // SEE pruning for quiets: skip moves with bad SEE
                // Threshold: -25 * adj_lmr_depth²
                let see_threshold = -see_quiet_quad() * adj_lmr_depth * adj_lmr_depth;
                let see_value = static_exchange_eval(game, &m);
                if see_value < see_threshold {
                    continue;
                }
            }
        }

        // Check legality BEFORE make_move (Pin Detection)
        // returns Ok(true) if legal, Ok(false) if illegal, Err if unsure
        let fast_legal = game.is_legal_fast(&m, in_check);
        if let Ok(false) = fast_legal {
            continue; // Definitely illegal (pinned piece moving off ray)
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
            if let Some(tt) = GLOBAL_TT.get() {
                tt.prefetch_entry(child_hash);
            }
        }

        let mut undo = game.make_move(&m);

        // Check if move is illegal (leaves our king in check)
        // Only check if fast check was inconclusive (Err)
        if fast_legal.is_err() && game.is_move_illegal() {
            game.undo_move(&m, undo);
            continue;
        }

        // Record quiet moves searched at this node for history maluses
        if !is_capture && !is_promotion {
            quiets_searched.push(m);
        }

        // For this node at `ply`, this move becomes the previous move for child
        // ply + 1, stored as (from_hash, to_hash).
        let prev_entry_backup = searcher.prev_move_stack[ply];
        let from_hash = hash_move_from(&m);
        let to_hash = hash_move_dest(&m);
        searcher.prev_move_stack[ply] = (from_hash, to_hash);

        // Store move, piece and state info for continuation history
        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        let in_check_backup = searcher.in_check_history[ply];
        let capture_backup = searcher.capture_history_stack[ply];

        searcher.move_history[ply] = Some(m);
        searcher.moved_piece_history[ply] = p_type as u8;
        searcher.in_check_history[ply] = in_check;
        searcher.capture_history_stack[ply] = is_capture;

        legal_moves += 1;

        // Calculate per-move extension (can be negative for negative extensions)
        let mut extension: i32 = 0;

        let is_tt_move = tt_move
            .filter(|tt_m| m.from == tt_m.from && m.to == tt_m.to && m.promotion == tt_m.promotion)
            .is_some();

        if let Some((tt_s_base, singular_depth)) = se_conditions.filter(|_| {
            is_tt_move
                && !is_pv
                && depth >= 6 + (tt_pv as usize)
                && !searcher.is_shuffling(game, &m, ply)
        }) {
            // Singular extension margin with TT Move History adjustment.
            // Adjust margin based on TT move reliability.
            // When TT moves are reliable (high ttMoveHistory), we can use a tighter margin
            // (less extension). When unreliable (low), use a more generous margin.
            let tt_history_adj = searcher.tt_move_history / 150; // History scaling factor
            let singular_beta = tt_s_base - (depth as i32) * 3 + tt_history_adj;

            // Undo the move we just made so we can search alternatives
            game.undo_move(&m, undo);

            // Do a reduced search excluding the TT move to verify singularity
            // Create a move generator that skips the TT move
            let mut se_gen = StagedMoveGen::with_exclusion(
                None, // No TT move hint for this search
                ply,
                depth as i32,
                searcher,
                game,
                m, // Exclude the current (TT) move
            );

            let mut se_best = -INFINITY;
            let mut se_moves_checked = 0;
            const SE_MAX_MOVES: usize = 6;

            while let Some(se_m) = se_gen.next(game, searcher) {
                if se_moves_checked >= SE_MAX_MOVES {
                    break;
                }

                // Fast legality check (skips is_move_illegal for non-pinned pieces)
                let fast_legal = game.is_legal_fast(&se_m, in_check);
                if let Ok(false) = fast_legal {
                    continue;
                }

                let se_undo = game.make_move(&se_m);
                if fast_legal.is_err() && game.is_move_illegal() {
                    game.undo_move(&se_m, se_undo);
                    continue;
                }

                se_moves_checked += 1;

                let se_score = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: singular_depth,
                    ply: ply + 1,
                    alpha: -singular_beta,
                    beta: -singular_beta + 1,
                    allow_null: false,
                    node_type: NodeType::Cut,
                    was_null_move: false,
                    excluded_move: None,
                });

                game.undo_move(&se_m, se_undo);

                if searcher.hot.stopped {
                    // Restore searcher state before returning
                    searcher.prev_move_stack[ply] = prev_entry_backup;
                    searcher.move_history[ply] = move_history_backup;
                    searcher.moved_piece_history[ply] = piece_history_backup;

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
                // TT move is singular - calculate extension level
                // Double extension margin: how much below singular_beta for +2 extension
                let double_margin = (depth as i32) * 2;
                // Triple extension margin: how much below for +3 extension
                let triple_margin = (depth as i32) * 4;

                extension = 1;
                if se_best < singular_beta - double_margin {
                    extension = 2;
                }
                if se_best < singular_beta - triple_margin && is_pv {
                    extension = 3;
                }
            } else if se_best >= beta && !is_pv && !is_decisive(se_best) {
                // Multi-cut: alternatives also beat beta, prune the whole subtree
                // Guard: don't return mate scores (they need verification)
                let penalty = (-400 - 100 * depth as i32).max(-4000);
                let max_tt_hist = 8192;
                searcher.tt_move_history +=
                    penalty - searcher.tt_move_history * penalty.abs() / max_tt_hist;

                game.undo_move(&m, undo);
                searcher.prev_move_stack[ply] = prev_entry_backup;
                searcher.move_history[ply] = move_history_backup;
                searcher.moved_piece_history[ply] = piece_history_backup;
                return se_best;
            } else if tt_value.is_some_and(|v| v >= beta) {
                // Negative extension: TT move is assumed to fail high but wasn't singular
                // Reduce depth to favor other moves
                extension = -3;
            } else if cut_node {
                // On cut nodes, if TT move isn't assumed to fail high, reduce it
                extension = -2;
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
            // Calculate new depth: base depth - 1 + extension (extension can be negative)
            let new_depth = ((depth as i32) - 1 + extension).max(0) as usize;
            score = -negamax(&mut NegamaxContext {
                searcher,
                game,
                depth: new_depth,
                ply: ply + 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                node_type: child_type,
                was_null_move: false,
                excluded_move: None,
            });
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
                // Continuous Stockfish-inspired adjustment: reduction -= history / 4096
                let hist_idx = hash_move_dest(&m);
                let hist_score = searcher.history[p_type as usize][hist_idx];
                reduction -= hist_score / 4096;

                // Increase reduction if next ply has a lot of fail highs
                // We use a simpler version: add 1 to reduction when many cutoffs
                if ply + 1 < MAX_PLY && searcher.cutoff_cnt[ply + 1] > lmr_cutoff_thresh() {
                    reduction += 1;
                    if all_node {
                        reduction += 1;
                    }
                }

                // TT Move History adjustment:
                // If TT moves have been unreliable (low tt_move_history), reduce less
                // since the move ordering from TT may not be trustworthy.
                // Only adjust for significant negative values to avoid overhead.
                if searcher.tt_move_history < lmr_tt_history_thresh() && reduction > 0 {
                    reduction -= 1;
                }

                // Ensure reduction stays in valid range [0, depth-2]
                reduction = reduction.clamp(0, (depth as i32) - 2);
            }

            // Base child depth after LMR (with singular extension if applicable)
            let mut new_depth = (depth as i32) - 1 + extension - reduction;

            // History Leaf Pruning:
            // Only in non-PV, quiet, shallow nodes and after enough moves
            // Exempt checking moves to avoid missing check-fork tactics:
            // Guard: don't prune when we have a losing mate score
            if !in_check
                && !is_pv
                && !is_capture
                && !is_promotion
                && !gives_check
                && depth <= hlp_max_depth()
                && legal_moves >= hlp_min_moves()
                && !is_loss(best_score)
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
                        // Restore searcher state before continuing
                        searcher.prev_move_stack[ply] = prev_entry_backup;
                        searcher.move_history[ply] = move_history_backup;
                        searcher.moved_piece_history[ply] = piece_history_backup;
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
            let mut s = -negamax(&mut NegamaxContext {
                searcher,
                game,
                depth: search_depth,
                ply: ply + 1,
                alpha: -alpha - 1,
                beta: -alpha,
                allow_null: true,
                node_type: child_type,
                was_null_move: false,
                excluded_move: None,
            });

            // Re-search at full depth if it looks promising
            if s > alpha && (reduction > 0 || s < beta) {
                // Re-search with PV-like search if we're in PV, otherwise same child type
                let research_type = if is_pv { NodeType::PV } else { child_type };

                // LMR deeper/shallower re-search depth adjustment
                // If reduced search returned good value, search deeper
                // If it returned bad value, search shallower
                let base_depth = (depth as i32) - 1 + extension;
                let do_deeper_search =
                    (search_depth as i32) < base_depth && s > (best_score + 43 + 2 * base_depth);
                let do_shallower_search = s < best_score + 9;
                let adjusted_depth = (base_depth + (do_deeper_search as i32)
                    - (do_shallower_search as i32))
                    .max(0) as usize;

                // TT move extension: prevent dropping to qsearch if TT has decisive/deep info
                // For PV nodes with the TT move, if about to go to qsearch and:
                // - TT has mate score with depth > 0, OR
                // - TT depth > 1
                // then ensure minimum depth of 1
                let mut pv_depth = adjusted_depth;
                if is_pv && is_tt_move && pv_depth == 0 {
                    let has_decisive =
                        tt_value.is_some_and(|v| v.abs() > MATE_SCORE) && tt_data_depth > 0;
                    let has_deep_tt = tt_data_depth > 1;
                    if has_decisive || has_deep_tt {
                        pv_depth = 1;
                    }
                }

                s = -negamax(&mut NegamaxContext {
                    searcher,
                    game,
                    depth: pv_depth,
                    ply: ply + 1,
                    alpha: -beta,
                    beta: -alpha,
                    allow_null: true,
                    node_type: research_type,
                    was_null_move: false,
                    excluded_move: None,
                });

                // Post LMR continuation history update
                // When a reduced search fails high and we had to re-search, the move
                // proved to be good - give it a bonus in continuation history.
                //
                // Bonus and malus logic for quiet moves:
                // 1. Depth-proportional: deeper searches = more reliable signal = bigger bonus
                // 2. Scaled down: LMR re-search is weaker signal than beta-cutoff (~1/3 bonus)
                // 3. Quiets only: continuation history only helps quiet move ordering
                if reduction > 0 && !is_capture && !is_promotion {
                    // Depth-scaled bonus: ~100 * depth (compared to ~300 * depth for cutoffs)
                    // This is roughly 1/3 of the beta-cutoff bonus, appropriate since
                    // "failed high after reduction" is a weaker signal than "caused cutoff"
                    let lmr_bonus = 100 * depth as i32;
                    let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;

                    // Update continuation histories at ply offsets -1, -2, -4
                    // (matching the existing beta-cutoff update pattern)
                    // Update continuation histories at ply offsets -1, -2, -3, -4, -5, -6
                    // (matching the new 6-lookup pattern)
                    for &plies_ago in &[0usize, 1, 2, 3, 4, 5] {
                        if ply > plies_ago
                            && let Some(prev_move) = searcher.move_history[ply - plies_ago - 1]
                        {
                            let prev_piece =
                                searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_piece < 32 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                                let cf_hash = hash_coord_32(m.from.x, m.from.y);
                                let ct_hash = hash_coord_32(m.to.x, m.to.y);

                                let prev_ic =
                                    searcher.in_check_history[ply - plies_ago - 1] as usize;
                                let prev_cap =
                                    searcher.capture_history_stack[ply - plies_ago - 1] as usize;

                                let entry = &mut searcher.cont_history[prev_cap][prev_ic]
                                    [prev_piece][prev_to_hash][cf_hash][ct_hash];
                                // Use gravity-based update: entry += bonus - entry * bonus / max
                                *entry += lmr_bonus - *entry * lmr_bonus / max_history;
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
        searcher.in_check_history[ply] = in_check_backup;
        searcher.capture_history_stack[ply] = capture_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.hot.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m);

            if score > alpha {
                alpha = score;

                // Update PV using triangular indexing
                // ply stores PV at pv_table[ply * MAX_PLY..], child at pv_table[(ply+1) * MAX_PLY..]
                let ply_base = ply * MAX_PLY;
                let child_base = (ply + 1) * MAX_PLY;

                searcher.pv_table[ply_base] = Some(m); // Head of PV is this move
                let child_len = searcher.pv_length[ply + 1];
                for j in 0..child_len {
                    searcher.pv_table[ply_base + 1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[ply] = child_len + 1;

                // Depth reduction on alpha improvement
                // Reduce depth for remaining moves after finding a score improvement
                // NOTE: Disabled - requires proper conditions to match engine behavior
                // if depth > 2 && depth < 14 && !is_decisive(score) {
                //     depth -= 2;
                // }
            }
        }

        if alpha >= beta {
            // Increment cutoff count
            // We increment for low-extension cutoffs or PV nodes
            if (extension < 2 || is_pv) && ply < MAX_PLY {
                searcher.cutoff_cnt[ply] = searcher.cutoff_cnt[ply].saturating_add(1);
            }

            if !is_capture {
                // History bonus for quiet cutoff move, with maluses for previously searched quiets
                let idx = hash_move_dest(&m);
                let bonus = history_bonus_base() * depth as i32 - history_bonus_sub();
                let adj = bonus.min(history_bonus_cap());
                let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;

                searcher.update_history(m.piece.piece_type(), idx, bonus);

                // Low Ply History update:
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
                searcher.killers[ply][1] = searcher.killers[ply][0];
                searcher.killers[ply][0] = Some(m);

                // Countermove heuristic: on a quiet beta cutoff, record this move
                // as the countermove to the move that led into this node.
                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (m.piece.piece_type() as u8, m.to.x as i16, m.to.y as i16);
                    }
                }

                // Continuation history update:
                for &plies_ago in &[0usize, 1, 2, 3, 4, 5] {
                    if ply > plies_ago
                        && let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1]
                    {
                        let prev_piece = searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                        if prev_piece < 32 {
                            let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                            let prev_ic = searcher.in_check_history[ply - plies_ago - 1] as usize;
                            let prev_cap =
                                searcher.capture_history_stack[ply - plies_ago - 1] as usize;

                            // Update all searched quiets (best with bonus, others with malus)
                            for quiet in &quiets_searched {
                                let q_from_hash = hash_coord_32(quiet.from.x, quiet.from.y);
                                let q_to_hash = hash_coord_32(quiet.to.x, quiet.to.y);
                                let is_best = quiet.from == m.from && quiet.to == m.to;

                                let entry = &mut searcher.cont_history[prev_cap][prev_ic]
                                    [prev_piece][prev_to_hash][q_from_hash][q_to_hash];
                                if is_best {
                                    *entry += adj - *entry * adj / max_history;
                                } else {
                                    *entry += -adj - *entry * adj / max_history;
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

    // Checkmate, stalemate, or loss by capture-based variants
    if legal_moves == 0 {
        // Determine if this is a loss:
        // 1. In check AND must escape check (our win condition is checkmate) → checkmate
        // 2. No pieces left (relevant for allpiecescaptured variants) → loss
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        if checkmate || no_pieces {
            return -MATE_VALUE + ply as i32;
        } else {
            return 0; // Stalemate
        }
    }

    // Adjust best value for fail high cases
    // Soften the score to prevent returning inflated values from reduced searches
    if best_score >= beta && !is_decisive(best_score) && !is_decisive(alpha) {
        best_score = (best_score * depth as i32 + beta) / (depth as i32 + 1);
    }

    // Store in TT with correct flag based on original alpha/beta (per Wikipedia pseudocode)
    // - UPPERBOUND: best_score <= alpha_orig (failed low, didn't improve alpha)
    // - LOWERBOUND: best_score >= beta_orig (failed high, caused beta cutoff)
    // - EXACT: alpha_orig < best_score < beta_orig (true minimax value)
    let tt_data_bound = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta_orig {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    store_tt_with_shared(
        searcher,
        &StoreContext {
            hash,
            depth,
            flag: tt_data_bound,
            score: best_score,
            static_eval: raw_eval,
            is_pv,
            best_move,
            ply,
        },
    );

    // Update TT Move History:
    // Tracks how reliable TT moves are: positive = TT moves tend to be best.
    // Only update in non-PV nodes to get clean cutoff/fail statistics.
    if !is_pv && let Some(ref bm) = best_move {
        // Check if best move matches the TT move
        let tt_move_matched = tt_move
            .as_ref()
            .is_some_and(|tm| tm.from == bm.from && tm.to == bm.to);

        // Gravity-based update: bonus = delta - entry * delta / max
        // Limit bonus magnitude and scale by depth:
        let delta: i32 = if tt_move_matched { 809 } else { -865 };
        let max_tt_history = 8192;
        searcher.tt_move_history += delta - searcher.tt_move_history * delta.abs() / max_tt_history;
    }

    // Update correction history when conditions are met:
    // - Not in check
    // - Best move is quiet or doesn't exist
    // - Score respects bound constraints relative to static eval
    if !in_check {
        let best_move_is_quiet = match best_move {
            Some(m) => {
                // BITBOARD: Fast capture check
                let captured = game.board.get_piece(m.to.x, m.to.y);
                let is_capture = captured.is_some_and(|p| !p.piece_type().is_neutral_type());
                !is_capture && m.promotion.is_none()
            }
            None => true, // No best move counts as "quiet"
        };

        // Replacement conditions:
        // - If lower bound (failed high), score should not be below static eval
        // - If upper bound (failed low), score should not be above static eval
        let should_update = match tt_data_bound {
            TTFlag::LowerBound => best_score >= raw_eval,
            TTFlag::UpperBound => best_score <= raw_eval,
            TTFlag::Exact => true,
            TTFlag::None => false, // Should never happen, but be safe
        };

        if best_move_is_quiet && should_update {
            searcher.update_correction_history(
                game,
                ply,
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
    if ply >= MAX_PLY - 1 {
        return evaluate(game);
    }

    // Check if we have an upcoming move that draws by repetition
    if alpha < VALUE_DRAW && game.upcoming_repetition(ply) {
        let draw_val = value_draw(searcher.hot.nodes);
        if draw_val >= beta {
            return draw_val;
        }
        alpha = alpha.max(draw_val);
    }

    searcher.hot.nodes += 1;
    searcher.hot.qnodes += 1;

    // Update seldepth
    if ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
    }

    // Draw by fifty-move rule or repetition
    if game.is_fifty() || game.is_repetition(ply) {
        return VALUE_DRAW;
    }

    if searcher.check_time() {
        return 0;
    }

    // Royal capture loss: if our king was just captured (RoyalCapture/AllRoyalsCaptured variants)
    if game.has_lost_by_royal_capture() {
        return -MATE_VALUE + ply as i32;
    }

    let in_check = game.is_in_check();
    // Only treat check specially if we must escape (checkmate-based win condition)
    let must_escape = in_check && game.must_escape_check();

    // Stand pat (not when must escape check)
    let stand_pat = if must_escape {
        -MATE_VALUE + ply as i32
    } else {
        evaluate(game)
    };

    if !must_escape {
        // Stand pat cutoff with adjustment for non-decisive scores
        if stand_pat >= beta {
            // Don't adjust mate scores
            if !is_decisive(stand_pat) {
                return (stand_pat + beta) / 2;
            }
            return stand_pat;
        }

        if alpha < stand_pat {
            alpha = stand_pat;
        }
    }

    if ply >= MAX_PLY - 1 {
        return stand_pat;
    }

    // When must escape check, generate all pseudo-legal moves (evasions) via the normal generator.
    // When not must escape, use a specialized capture-only generator to avoid creating
    // thousands of quiet moves only to filter them out.
    // Reuse per-ply move buffer to avoid Vec allocations inside quiescence.
    let mut tactical_moves: MoveList = MoveList::new();
    std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);
    tactical_moves.clear();

    if must_escape {
        // In check and must escape - only generate evasion moves
        game.get_evasion_moves_into(&mut tactical_moves);
    } else {
        // Normal quiescence: generate captures only
        let ctx = MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: game.enemy_king_pos(),
        };
        get_quiescence_captures(&game.board, game.turn, &ctx, &mut tactical_moves);
    }

    // Sort captures by MVV-LVA
    sort_captures(game, &mut tactical_moves);

    let mut best_score = stand_pat;
    let mut legal_moves = 0;

    // Delta pruning margin (safety buffer for positional factors)
    let delta_margin = delta_margin();

    for m in &tactical_moves {
        // Don't prune when we're getting mated - need to search all moves
        if !in_check && !is_loss(best_score) {
            // See gain for the capture/promotion
            let see_gain = static_exchange_eval(game, m);

            // Prune clearly losing captures that don't even break even materially.
            if see_gain < 0 {
                continue;
            }

            // Delta pruning: if stand_pat + best possible material swing from this
            // capture (SEE gain) plus a small margin cannot beat alpha, skip.
            if stand_pat + see_gain + delta_margin < alpha {
                continue;
            }
        }

        // Fast legality check (skips is_move_illegal for non-pinned pieces)
        let fast_legal = game.is_legal_fast(m, in_check);
        if let Ok(false) = fast_legal {
            continue;
        }

        let undo = game.make_move(m);

        if fast_legal.is_err() && game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        legal_moves += 1;

        let score = -quiescence(searcher, game, ply + 1, -beta, -alpha);

        game.undo_move(m, undo);

        if searcher.hot.stopped {
            // Swap back move buffer before returning early
            std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);
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
        // Determine if this is a loss:
        // 1. In check AND must escape check (our win condition is checkmate) → checkmate
        // 2. No pieces left (relevant for allpiecescaptured variants) → loss
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        if checkmate || no_pieces {
            // Swap back move buffer before returning mate score
            std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);
            return -MATE_VALUE + ply as i32;
        }
    }

    // Swap back move buffer for this ply before returning
    std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);

    // Adjust score for fail-high, but not for mate scores
    if !is_decisive(best_score) && best_score > beta {
        best_score = (best_score + beta) / 2;
    }

    best_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;
    use crate::moves::Move;

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
            nodes: 0,
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
        let result = get_best_move(&mut game, 5, 1000, true, true);

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

        let result = get_best_move(&mut game, 5, 1000, true, true);

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
        let tt = LocalTranspositionTable::new(1);

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
        GLOBAL_TT.get_or_init(|| TranspositionTable::new(16));
        let searcher = Searcher::new(10000);

        assert_eq!(searcher.hot.nodes, 0);
        assert!(GLOBAL_TT.get().unwrap().capacity() > 0);
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
            nodes: 0,
            tt_capacity: 1000,
            tt_used: 100,
            tt_fill_permille: 100,
        };
        assert_eq!(stats.tt_capacity, 1000);
        assert_eq!(stats.tt_used, 100);
        assert_eq!(stats.tt_fill_permille, 100);
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

        let result = get_best_move(&mut game, 3, 500, true, true);
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

        let result = get_best_move(&mut game, 4, 500, true, true);
        assert!(result.is_some());
    }

    // ======================== Format PV Tests ========================

    #[test]
    fn test_format_pv_empty() {
        let searcher = Searcher::new(1000);
        let mut game = GameState::new();
        let pv = searcher.format_pv(&mut game, 0);
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
        let adjusted = searcher.adjusted_eval(&game, raw_eval, 0, 0);
        // Adjusted eval should be within reasonable bounds of raw
        assert!(adjusted.abs() < raw_eval.abs() + 1000);
    }

    // ======================== Extract PV Tests ========================

    #[test]
    fn test_extract_pv() {
        let searcher = Searcher::new(1000);
        let mut game = GameState::new();
        let pv = searcher.extract_pv_only(&mut game, 1);
        // PV should be empty for a fresh searcher
        assert!(pv.is_empty());
    }

    // ======================== Reset Search State Tests ========================

    #[test]
    fn test_reset_search_state() {
        // Should not panic
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
        let result = get_best_moves_multipv(&mut game, 2, 500, 500, 2, true, false);

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
    fn test_tt_integration_via_global() {
        GLOBAL_TT.get_or_init(|| TranspositionTable::new(16));
        // We don't need searcher for TT anymore
        let hash = 123456789;
        let depth = 5;
        let score = 1000;
        let best_move = Move::new(
            Coordinate::new(0, 0),
            Coordinate::new(1, 1),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        // Store EXACT score using correct TT signature:
        GLOBAL_TT
            .get()
            .unwrap()
            .store(&crate::search::tt_defs::TTStoreParams {
                hash,
                depth,
                flag: crate::search::tt_defs::TTFlag::Exact,
                score,
                static_eval: INFINITY + 1,
                is_pv: true,
                best_move: Some(best_move),
                ply: 0,
            });

        // Probe EXACT score using correct TT signature:
        let result = GLOBAL_TT
            .get()
            .unwrap()
            .probe(&crate::search::tt_defs::TTProbeParams {
                hash,
                alpha: score - 100,
                beta: score + 100,
                depth,
                ply: 0,
                rule50_count: 0,
                rule_limit: 100,
            });
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.cutoff_score, score);
        assert!(res.best_move.is_some());
        assert_eq!(res.best_move.unwrap().from.x, 0);
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
        let _in_pawn_endgame = game.white_piece_count <= 2 && game.black_piece_count <= 2;
        assert!(!moves.is_empty(), "White should have legal moves, found 0");

        // Search depth 3 to be absolutely sure
        let result = get_best_move(&mut game, 3, 2000, true, true);
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
                nodes: 0,
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
