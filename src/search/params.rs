//! Search Parameters Module
//!
//! Centralizes all tunable search constants. During normal compilation, these
//! are compile-time constants for maximum performance. When the `search_tuning`
//! feature is enabled, they become runtime-configurable via a global struct.

#[cfg(feature = "search_tuning")]
use once_cell::sync::Lazy;
#[cfg(feature = "search_tuning")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "search_tuning")]
use std::sync::RwLock;

// ============================================================================
// DEFAULT VALUES - These are the baseline constants used in production
// ============================================================================

// Null Move Pruning
pub const DEFAULT_NMP_REDUCTION: usize = 3;
pub const DEFAULT_NMP_MIN_DEPTH: usize = 3;

// Late Move Reductions
pub const DEFAULT_LMR_MIN_DEPTH: usize = 3;
pub const DEFAULT_LMR_MIN_MOVES: usize = 4;
pub const DEFAULT_LMR_DIVISOR: usize = 3; // ln(moves) * ln(depth) / divisor

// History Leaf Pruning (Fruit-style)
pub const DEFAULT_HLP_MAX_DEPTH: usize = 3;
pub const DEFAULT_HLP_MIN_MOVES: usize = 4;
pub const DEFAULT_HLP_HISTORY_REDUCE: i32 = 300;
pub const DEFAULT_HLP_HISTORY_LEAF: i32 = 0;

// Late Move Pruning thresholds by depth [0, 1, 2, 3, 4]
pub const DEFAULT_LMP_THRESHOLD: [usize; 5] = [0, 4, 7, 11, 16];

// Aspiration Window
pub const DEFAULT_ASPIRATION_WINDOW: i32 = 45;
pub const DEFAULT_ASPIRATION_FAIL_MULT: i32 = 4; // Window *= this on fail
pub const DEFAULT_ASPIRATION_MAX_WINDOW: i32 = 1000;

// Futility margins by depth [0, 1, 2, 3]
pub const DEFAULT_FUTILITY_MARGIN: [i32; 4] = [0, 95, 190, 285];

// Reverse Futility Pruning
pub const DEFAULT_RFP_MAX_DEPTH: usize = 4;
pub const DEFAULT_RFP_MARGIN_PER_DEPTH: i32 = 105;

// Internal Iterative Reductions
pub const DEFAULT_IIR_MIN_DEPTH: usize = 4;

// Move Ordering scores (higher = searched first)
pub const DEFAULT_SORT_HASH: i32 = 6000000;
pub const DEFAULT_SORT_WINNING_CAPTURE: i32 = 1000000;
pub const DEFAULT_SORT_LOSING_CAPTURE: i32 = 0;
pub const DEFAULT_SORT_QUIET: i32 = 0;
pub const DEFAULT_SORT_KILLER1: i32 = 900000;
pub const DEFAULT_SORT_KILLER2: i32 = 800000;
pub const DEFAULT_SORT_COUNTERMOVE: i32 = 600000;
pub const DEFAULT_SORT_GIVES_CHECK: i32 = 700000;

// SEE threshold for "winning" captures
pub const DEFAULT_SEE_WINNING_THRESHOLD: i32 = -80;

// History heuristic
pub const DEFAULT_MAX_HISTORY: i32 = 4000;
pub const DEFAULT_HISTORY_BONUS_BASE: i32 = 300;
pub const DEFAULT_HISTORY_BONUS_SUB: i32 = 250;
pub const DEFAULT_HISTORY_BONUS_CAP: i32 = 1536;
pub const DEFAULT_HISTORY_MAX_GRAVITY: i32 = 16384;
pub const DEFAULT_HISTORY_DECAY_NUMER: i32 = 9;
pub const DEFAULT_HISTORY_DECAY_DENOM: i32 = 10;

// Capture history divisor for move ordering
pub const DEFAULT_CAPTURE_HISTORY_DIVISOR: i32 = 10;

// Repetition/Draw penalties
pub const DEFAULT_REPETITION_PENALTY: i32 = 8;

// Quiescence
pub const DEFAULT_DELTA_MARGIN: i32 = 180;

// Time management
pub const DEFAULT_STABILITY_FACTOR_BASE: f64 = 1.1;
pub const DEFAULT_STABILITY_FACTOR_DECAY: f64 = 0.03;
pub const DEFAULT_STABILITY_MIN_FACTOR: f64 = 0.5;
pub const DEFAULT_SCORE_JUMP_FACTOR: f64 = 1.1;

// ============================================================================
// FEATURE-GATED RUNTIME CONFIGURATION
// ============================================================================

#[cfg(feature = "search_tuning")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchParams {
    // Null Move Pruning
    pub nmp_reduction: usize,
    pub nmp_min_depth: usize,

    // Late Move Reductions
    pub lmr_min_depth: usize,
    pub lmr_min_moves: usize,
    pub lmr_divisor: usize,

    // History Leaf Pruning
    pub hlp_max_depth: usize,
    pub hlp_min_moves: usize,
    pub hlp_history_reduce: i32,
    pub hlp_history_leaf: i32,

    // Late Move Pruning
    pub lmp_threshold_1: usize,
    pub lmp_threshold_2: usize,
    pub lmp_threshold_3: usize,
    pub lmp_threshold_4: usize,

    // Aspiration
    pub aspiration_window: i32,
    pub aspiration_fail_mult: i32,
    pub aspiration_max_window: i32,

    // Futility
    pub futility_margin_1: i32,
    pub futility_margin_2: i32,
    pub futility_margin_3: i32,

    // Reverse Futility
    pub rfp_max_depth: usize,
    pub rfp_margin_per_depth: i32,

    // IIR
    pub iir_min_depth: usize,

    // Move Ordering
    pub sort_hash: i32,
    pub sort_winning_capture: i32,
    pub sort_killer1: i32,
    pub sort_killer2: i32,
    pub sort_countermove: i32,
    pub see_winning_threshold: i32,

    // History
    pub max_history: i32,
    pub history_bonus_base: i32,
    pub history_bonus_sub: i32,
    pub history_bonus_cap: i32,

    // Other
    pub repetition_penalty: i32,
    pub delta_margin: i32,
}

#[cfg(feature = "search_tuning")]
impl Default for SearchParams {
    fn default() -> Self {
        Self {
            nmp_reduction: DEFAULT_NMP_REDUCTION,
            nmp_min_depth: DEFAULT_NMP_MIN_DEPTH,
            lmr_min_depth: DEFAULT_LMR_MIN_DEPTH,
            lmr_min_moves: DEFAULT_LMR_MIN_MOVES,
            lmr_divisor: DEFAULT_LMR_DIVISOR,
            hlp_max_depth: DEFAULT_HLP_MAX_DEPTH,
            hlp_min_moves: DEFAULT_HLP_MIN_MOVES,
            hlp_history_reduce: DEFAULT_HLP_HISTORY_REDUCE,
            hlp_history_leaf: DEFAULT_HLP_HISTORY_LEAF,
            lmp_threshold_1: DEFAULT_LMP_THRESHOLD[1],
            lmp_threshold_2: DEFAULT_LMP_THRESHOLD[2],
            lmp_threshold_3: DEFAULT_LMP_THRESHOLD[3],
            lmp_threshold_4: DEFAULT_LMP_THRESHOLD[4],
            aspiration_window: DEFAULT_ASPIRATION_WINDOW,
            aspiration_fail_mult: DEFAULT_ASPIRATION_FAIL_MULT,
            aspiration_max_window: DEFAULT_ASPIRATION_MAX_WINDOW,
            futility_margin_1: DEFAULT_FUTILITY_MARGIN[1],
            futility_margin_2: DEFAULT_FUTILITY_MARGIN[2],
            futility_margin_3: DEFAULT_FUTILITY_MARGIN[3],
            rfp_max_depth: DEFAULT_RFP_MAX_DEPTH,
            rfp_margin_per_depth: DEFAULT_RFP_MARGIN_PER_DEPTH,
            iir_min_depth: DEFAULT_IIR_MIN_DEPTH,
            sort_hash: DEFAULT_SORT_HASH,
            sort_winning_capture: DEFAULT_SORT_WINNING_CAPTURE,
            sort_killer1: DEFAULT_SORT_KILLER1,
            sort_killer2: DEFAULT_SORT_KILLER2,
            sort_countermove: DEFAULT_SORT_COUNTERMOVE,
            see_winning_threshold: DEFAULT_SEE_WINNING_THRESHOLD,
            max_history: DEFAULT_MAX_HISTORY,
            history_bonus_base: DEFAULT_HISTORY_BONUS_BASE,
            history_bonus_sub: DEFAULT_HISTORY_BONUS_SUB,
            history_bonus_cap: DEFAULT_HISTORY_BONUS_CAP,
            repetition_penalty: DEFAULT_REPETITION_PENALTY,
            delta_margin: DEFAULT_DELTA_MARGIN,
        }
    }
}

#[cfg(feature = "search_tuning")]
pub static SEARCH_PARAMS: Lazy<RwLock<SearchParams>> =
    Lazy::new(|| RwLock::new(SearchParams::default()));

/// Set search parameters from a JSON string. Returns true on success.
#[cfg(feature = "search_tuning")]
pub fn set_search_params_from_json(json: &str) -> bool {
    match serde_json::from_str::<SearchParams>(json) {
        Ok(params) => {
            if let Ok(mut guard) = SEARCH_PARAMS.write() {
                *guard = params;
                true
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

/// Get current search parameters as a JSON string.
#[cfg(feature = "search_tuning")]
pub fn get_search_params_as_json() -> String {
    if let Ok(guard) = SEARCH_PARAMS.read() {
        serde_json::to_string(&*guard).unwrap_or_else(|_| "{}".to_string())
    } else {
        "{}".to_string()
    }
}

// ============================================================================
// ACCESSOR MACROS/FUNCTIONS
//
// When `search_tuning` is enabled, these read from the global struct.
// When disabled, they are compile-time constants (zero overhead).
// ============================================================================

#[cfg(feature = "search_tuning")]
macro_rules! param {
    ($field:ident) => {{ SEARCH_PARAMS.read().unwrap().$field }};
}

// Null Move Pruning
#[cfg(feature = "search_tuning")]
#[inline]
pub fn nmp_reduction() -> usize {
    param!(nmp_reduction)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn nmp_reduction() -> usize {
    DEFAULT_NMP_REDUCTION
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn nmp_min_depth() -> usize {
    param!(nmp_min_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn nmp_min_depth() -> usize {
    DEFAULT_NMP_MIN_DEPTH
}

// Late Move Reductions
#[cfg(feature = "search_tuning")]
#[inline]
pub fn lmr_min_depth() -> usize {
    param!(lmr_min_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn lmr_min_depth() -> usize {
    DEFAULT_LMR_MIN_DEPTH
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn lmr_min_moves() -> usize {
    param!(lmr_min_moves)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn lmr_min_moves() -> usize {
    DEFAULT_LMR_MIN_MOVES
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn lmr_divisor() -> usize {
    param!(lmr_divisor)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn lmr_divisor() -> usize {
    DEFAULT_LMR_DIVISOR
}

// History Leaf Pruning
#[cfg(feature = "search_tuning")]
#[inline]
pub fn hlp_max_depth() -> usize {
    param!(hlp_max_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn hlp_max_depth() -> usize {
    DEFAULT_HLP_MAX_DEPTH
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn hlp_min_moves() -> usize {
    param!(hlp_min_moves)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn hlp_min_moves() -> usize {
    DEFAULT_HLP_MIN_MOVES
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn hlp_history_reduce() -> i32 {
    param!(hlp_history_reduce)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn hlp_history_reduce() -> i32 {
    DEFAULT_HLP_HISTORY_REDUCE
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn hlp_history_leaf() -> i32 {
    param!(hlp_history_leaf)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn hlp_history_leaf() -> i32 {
    DEFAULT_HLP_HISTORY_LEAF
}

// LMP Thresholds
#[cfg(feature = "search_tuning")]
#[inline]
pub fn lmp_threshold(depth: usize) -> usize {
    let guard = SEARCH_PARAMS.read().unwrap();
    match depth {
        0 => 0,
        1 => guard.lmp_threshold_1,
        2 => guard.lmp_threshold_2,
        3 => guard.lmp_threshold_3,
        _ => guard.lmp_threshold_4,
    }
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn lmp_threshold(depth: usize) -> usize {
    if depth == 0 {
        0
    } else if depth < DEFAULT_LMP_THRESHOLD.len() {
        DEFAULT_LMP_THRESHOLD[depth]
    } else {
        DEFAULT_LMP_THRESHOLD[4]
    }
}

// Aspiration Window
#[cfg(feature = "search_tuning")]
#[inline]
pub fn aspiration_window() -> i32 {
    param!(aspiration_window)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn aspiration_window() -> i32 {
    DEFAULT_ASPIRATION_WINDOW
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn aspiration_fail_mult() -> i32 {
    param!(aspiration_fail_mult)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn aspiration_fail_mult() -> i32 {
    DEFAULT_ASPIRATION_FAIL_MULT
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn aspiration_max_window() -> i32 {
    param!(aspiration_max_window)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn aspiration_max_window() -> i32 {
    DEFAULT_ASPIRATION_MAX_WINDOW
}

// Futility Margins
#[cfg(feature = "search_tuning")]
#[inline]
pub fn futility_margin(depth: usize) -> i32 {
    let guard = SEARCH_PARAMS.read().unwrap();
    match depth {
        0 => 0,
        1 => guard.futility_margin_1,
        2 => guard.futility_margin_2,
        _ => guard.futility_margin_3,
    }
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn futility_margin(depth: usize) -> i32 {
    if depth < DEFAULT_FUTILITY_MARGIN.len() {
        DEFAULT_FUTILITY_MARGIN[depth]
    } else {
        DEFAULT_FUTILITY_MARGIN[3]
    }
}

// Reverse Futility Pruning
#[cfg(feature = "search_tuning")]
#[inline]
pub fn rfp_max_depth() -> usize {
    param!(rfp_max_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn rfp_max_depth() -> usize {
    DEFAULT_RFP_MAX_DEPTH
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn rfp_margin_per_depth() -> i32 {
    param!(rfp_margin_per_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn rfp_margin_per_depth() -> i32 {
    DEFAULT_RFP_MARGIN_PER_DEPTH
}

// IIR
#[cfg(feature = "search_tuning")]
#[inline]
pub fn iir_min_depth() -> usize {
    param!(iir_min_depth)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn iir_min_depth() -> usize {
    DEFAULT_IIR_MIN_DEPTH
}

// Move Ordering
#[cfg(feature = "search_tuning")]
#[inline]
pub fn sort_hash() -> i32 {
    param!(sort_hash)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn sort_hash() -> i32 {
    DEFAULT_SORT_HASH
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn sort_winning_capture() -> i32 {
    param!(sort_winning_capture)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn sort_winning_capture() -> i32 {
    DEFAULT_SORT_WINNING_CAPTURE
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn sort_killer1() -> i32 {
    param!(sort_killer1)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn sort_killer1() -> i32 {
    DEFAULT_SORT_KILLER1
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn sort_killer2() -> i32 {
    param!(sort_killer2)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn sort_killer2() -> i32 {
    DEFAULT_SORT_KILLER2
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn sort_countermove() -> i32 {
    param!(sort_countermove)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn sort_countermove() -> i32 {
    DEFAULT_SORT_COUNTERMOVE
}

// Gives Check bonus (not tunable, fixed constant)
#[inline]
pub const fn sort_gives_check() -> i32 {
    DEFAULT_SORT_GIVES_CHECK
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn see_winning_threshold() -> i32 {
    param!(see_winning_threshold)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn see_winning_threshold() -> i32 {
    DEFAULT_SEE_WINNING_THRESHOLD
}

// History
#[cfg(feature = "search_tuning")]
#[inline]
pub fn max_history() -> i32 {
    param!(max_history)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn max_history() -> i32 {
    DEFAULT_MAX_HISTORY
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn history_bonus_base() -> i32 {
    param!(history_bonus_base)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn history_bonus_base() -> i32 {
    DEFAULT_HISTORY_BONUS_BASE
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn history_bonus_sub() -> i32 {
    param!(history_bonus_sub)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn history_bonus_sub() -> i32 {
    DEFAULT_HISTORY_BONUS_SUB
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn history_bonus_cap() -> i32 {
    param!(history_bonus_cap)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn history_bonus_cap() -> i32 {
    DEFAULT_HISTORY_BONUS_CAP
}

// Other
#[cfg(feature = "search_tuning")]
#[inline]
pub fn repetition_penalty() -> i32 {
    param!(repetition_penalty)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn repetition_penalty() -> i32 {
    DEFAULT_REPETITION_PENALTY
}

#[cfg(feature = "search_tuning")]
#[inline]
pub fn delta_margin() -> i32 {
    param!(delta_margin)
}
#[cfg(not(feature = "search_tuning"))]
#[inline]
pub const fn delta_margin() -> i32 {
    DEFAULT_DELTA_MARGIN
}
