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

// Razoring
pub const DEFAULT_RAZORING_LINEAR: i32 = 485;
pub const DEFAULT_RAZORING_QUAD: i32 = 281;

// Null Move Pruning
pub const DEFAULT_NMP_REDUCTION: usize = 3;
pub const DEFAULT_NMP_MIN_DEPTH: usize = 3;
pub const DEFAULT_NMP_BASE: i32 = 350;
pub const DEFAULT_NMP_DEPTH_MULT: i32 = 18;
pub const DEFAULT_NMP_REDUCTION_BASE: usize = 7;
pub const DEFAULT_NMP_REDUCTION_DIV: usize = 3;

// Late Move Reductions
pub const DEFAULT_LMR_MIN_DEPTH: usize = 3;
pub const DEFAULT_LMR_MIN_MOVES: usize = 4;
pub const DEFAULT_LMR_DIVISOR: usize = 3; // ln(moves) * ln(depth) / divisor
pub const DEFAULT_LMR_HISTORY_THRESH: i32 = 2000;
pub const DEFAULT_LMR_CUTOFF_THRESH: u8 = 2;
pub const DEFAULT_LMR_TT_HISTORY_THRESH: i32 = -1000;

// History Leaf Pruning:
pub const DEFAULT_HLP_MAX_DEPTH: usize = 3;
pub const DEFAULT_HLP_MIN_MOVES: usize = 4;
pub const DEFAULT_HLP_HISTORY_REDUCE: i32 = 300;
pub const DEFAULT_HLP_HISTORY_LEAF: i32 = 0;

// Late Move Pruning
pub const DEFAULT_LMP_BASE: usize = 3;
pub const DEFAULT_LMP_DEPTH_MULT: usize = 1;

// Aspiration Window
pub const DEFAULT_ASPIRATION_WINDOW: i32 = 60;
pub const DEFAULT_ASPIRATION_FAIL_MULT: i32 = 4; // Window *= this on fail
pub const DEFAULT_ASPIRATION_MAX_WINDOW: i32 = 1000;

// Futility margins by depth [0, 1, 2, 3] (Legacy - kept for compatibility if needed, but RFP replaces logic mostly)
pub const DEFAULT_FUTILITY_MARGIN: [i32; 4] = [0, 95, 190, 285];

// Reverse Futility Pruning
pub const DEFAULT_RFP_MAX_DEPTH: usize = 14;
pub const DEFAULT_RFP_MULT_TT: i32 = 76;
pub const DEFAULT_RFP_MULT_NO_TT: i32 = 53;
pub const DEFAULT_RFP_IMPROVING_MULT: i32 = 2474;
pub const DEFAULT_RFP_WORSENING_MULT: i32 = 331;

// ProbCut
pub const DEFAULT_PROBCUT_MARGIN: i32 = 235;
pub const DEFAULT_PROBCUT_IMPROVING: i32 = 63;
pub const DEFAULT_PROBCUT_MIN_DEPTH: usize = 5;
pub const DEFAULT_PROBCUT_DEPTH_SUB: usize = 4;
pub const DEFAULT_PROBCUT_DIVISOR: i32 = 315;
pub const DEFAULT_LOW_DEPTH_PROBCUT_MARGIN: i32 = 800;

// Internal Iterative Reductions
pub const DEFAULT_IIR_MIN_DEPTH: usize = 6; // Adjusted to match search.rs:2326

// SEE Pruning
pub const DEFAULT_SEE_CAPTURE_LINEAR: i32 = 166;
pub const DEFAULT_SEE_CAPTURE_HIST_DIV: i32 = 29;
pub const DEFAULT_SEE_QUIET_QUAD: i32 = 25;
pub const DEFAULT_SEE_WINNING_THRESHOLD: i32 = -80;

// Move Ordering scores (higher = searched first)
pub const DEFAULT_SORT_HASH: i32 = 6000000;
pub const DEFAULT_SORT_WINNING_CAPTURE: i32 = 1000000;
pub const DEFAULT_SORT_LOSING_CAPTURE: i32 = 0;
pub const DEFAULT_SORT_QUIET: i32 = 0;
pub const DEFAULT_SORT_KILLER1: i32 = 900000;
pub const DEFAULT_SORT_KILLER2: i32 = 800000;
pub const DEFAULT_SORT_COUNTERMOVE: i32 = 600000;

// History heuristic
pub const DEFAULT_MAX_HISTORY: i32 = 4000;
pub const DEFAULT_HISTORY_BONUS_BASE: i32 = 300;
pub const DEFAULT_HISTORY_BONUS_SUB: i32 = 250;
pub const DEFAULT_HISTORY_BONUS_CAP: i32 = 1536;
pub const DEFAULT_HISTORY_MAX_GRAVITY: i32 = 16384;

// Repetition/Draw penalties
pub const DEFAULT_REPETITION_PENALTY: i32 = 8;

// Quiescence
pub const DEFAULT_DELTA_MARGIN: i32 = 200;

// ============================================================================
// FEATURE-GATED RUNTIME CONFIGURATION
// ============================================================================

#[cfg(feature = "search_tuning")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchParams {
    // Razoring
    pub razoring_linear: i32,
    pub razoring_quad: i32,

    // Null Move Pruning
    pub nmp_reduction: usize,
    pub nmp_min_depth: usize,
    pub nmp_base: i32,
    pub nmp_depth_mult: i32,
    pub nmp_reduction_base: usize,
    pub nmp_reduction_div: usize,

    // Late Move Reductions
    pub lmr_min_depth: usize,
    pub lmr_min_moves: usize,
    pub lmr_divisor: usize,
    pub lmr_history_thresh: i32,
    pub lmr_cutoff_thresh: u8,
    pub lmr_tt_history_thresh: i32,

    // History Leaf Pruning
    pub hlp_max_depth: usize,
    pub hlp_min_moves: usize,
    pub hlp_history_reduce: i32,
    pub hlp_history_leaf: i32,

    // Late Move Pruning
    pub lmp_base: usize,
    pub lmp_depth_mult: usize,

    // Aspiration
    pub aspiration_window: i32,
    pub aspiration_fail_mult: i32,
    pub aspiration_max_window: i32,

    // Reverse Futility (Expanded)
    pub rfp_max_depth: usize,
    pub rfp_mult_tt: i32,
    pub rfp_mult_no_tt: i32,
    pub rfp_improving_mult: i32,
    pub rfp_worsening_mult: i32,

    // ProbCut
    pub probcut_margin: i32,
    pub probcut_improving: i32,
    pub probcut_min_depth: usize,
    pub probcut_depth_sub: usize,
    pub probcut_divisor: i32,
    pub low_depth_probcut_margin: i32,

    // IIR
    pub iir_min_depth: usize,

    // SEE Pruning
    pub see_capture_linear: i32,
    pub see_capture_hist_div: i32,
    pub see_quiet_quad: i32,
    pub see_winning_threshold: i32,

    // Move Ordering
    pub sort_hash: i32,
    pub sort_winning_capture: i32,
    pub sort_killer1: i32,
    pub sort_killer2: i32,
    pub sort_countermove: i32,

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
            razoring_linear: DEFAULT_RAZORING_LINEAR,
            razoring_quad: DEFAULT_RAZORING_QUAD,
            nmp_reduction: DEFAULT_NMP_REDUCTION,
            nmp_min_depth: DEFAULT_NMP_MIN_DEPTH,
            nmp_base: DEFAULT_NMP_BASE,
            nmp_depth_mult: DEFAULT_NMP_DEPTH_MULT,
            nmp_reduction_base: DEFAULT_NMP_REDUCTION_BASE,
            nmp_reduction_div: DEFAULT_NMP_REDUCTION_DIV,
            lmr_min_depth: DEFAULT_LMR_MIN_DEPTH,
            lmr_min_moves: DEFAULT_LMR_MIN_MOVES,
            lmr_divisor: DEFAULT_LMR_DIVISOR,
            lmr_history_thresh: DEFAULT_LMR_HISTORY_THRESH,
            lmr_cutoff_thresh: DEFAULT_LMR_CUTOFF_THRESH,
            lmr_tt_history_thresh: DEFAULT_LMR_TT_HISTORY_THRESH,
            hlp_max_depth: DEFAULT_HLP_MAX_DEPTH,
            hlp_min_moves: DEFAULT_HLP_MIN_MOVES,
            hlp_history_reduce: DEFAULT_HLP_HISTORY_REDUCE,
            hlp_history_leaf: DEFAULT_HLP_HISTORY_LEAF,
            lmp_base: DEFAULT_LMP_BASE,
            lmp_depth_mult: DEFAULT_LMP_DEPTH_MULT,
            aspiration_window: DEFAULT_ASPIRATION_WINDOW,
            aspiration_fail_mult: DEFAULT_ASPIRATION_FAIL_MULT,
            aspiration_max_window: DEFAULT_ASPIRATION_MAX_WINDOW,
            rfp_max_depth: DEFAULT_RFP_MAX_DEPTH,
            rfp_mult_tt: DEFAULT_RFP_MULT_TT,
            rfp_mult_no_tt: DEFAULT_RFP_MULT_NO_TT,
            rfp_improving_mult: DEFAULT_RFP_IMPROVING_MULT,
            rfp_worsening_mult: DEFAULT_RFP_WORSENING_MULT,
            probcut_margin: DEFAULT_PROBCUT_MARGIN,
            probcut_improving: DEFAULT_PROBCUT_IMPROVING,
            probcut_min_depth: DEFAULT_PROBCUT_MIN_DEPTH,
            probcut_depth_sub: DEFAULT_PROBCUT_DEPTH_SUB,
            probcut_divisor: DEFAULT_PROBCUT_DIVISOR,
            low_depth_probcut_margin: DEFAULT_LOW_DEPTH_PROBCUT_MARGIN,
            iir_min_depth: DEFAULT_IIR_MIN_DEPTH,
            see_capture_linear: DEFAULT_SEE_CAPTURE_LINEAR,
            see_capture_hist_div: DEFAULT_SEE_CAPTURE_HIST_DIV,
            see_quiet_quad: DEFAULT_SEE_QUIET_QUAD,
            see_winning_threshold: DEFAULT_SEE_WINNING_THRESHOLD,
            sort_hash: DEFAULT_SORT_HASH,
            sort_winning_capture: DEFAULT_SORT_WINNING_CAPTURE,
            sort_killer1: DEFAULT_SORT_KILLER1,
            sort_killer2: DEFAULT_SORT_KILLER2,
            sort_countermove: DEFAULT_SORT_COUNTERMOVE,
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
// ============================================================================

#[cfg(feature = "search_tuning")]
macro_rules! param {
    ($field:ident) => {{ SEARCH_PARAMS.read().unwrap().$field }};
}

// Helper macro to generate accessors
macro_rules! define_accessor {
    ($name:ident, $type:ty, $default:ident) => {
        #[cfg(feature = "search_tuning")]
        #[inline]
        pub fn $name() -> $type {
            param!($name)
        }
        #[cfg(not(feature = "search_tuning"))]
        #[inline]
        pub const fn $name() -> $type {
            $default
        }
    };
}

// Razoring
define_accessor!(razoring_linear, i32, DEFAULT_RAZORING_LINEAR);
define_accessor!(razoring_quad, i32, DEFAULT_RAZORING_QUAD);

// Null Move Pruning
define_accessor!(nmp_reduction, usize, DEFAULT_NMP_REDUCTION);
define_accessor!(nmp_min_depth, usize, DEFAULT_NMP_MIN_DEPTH);
define_accessor!(nmp_base, i32, DEFAULT_NMP_BASE);
define_accessor!(nmp_depth_mult, i32, DEFAULT_NMP_DEPTH_MULT);
define_accessor!(nmp_reduction_base, usize, DEFAULT_NMP_REDUCTION_BASE);
define_accessor!(nmp_reduction_div, usize, DEFAULT_NMP_REDUCTION_DIV);

// Late Move Reductions
define_accessor!(lmr_min_depth, usize, DEFAULT_LMR_MIN_DEPTH);
define_accessor!(lmr_min_moves, usize, DEFAULT_LMR_MIN_MOVES);
define_accessor!(lmr_divisor, usize, DEFAULT_LMR_DIVISOR);
define_accessor!(lmr_history_thresh, i32, DEFAULT_LMR_HISTORY_THRESH);
define_accessor!(lmr_cutoff_thresh, u8, DEFAULT_LMR_CUTOFF_THRESH);
define_accessor!(lmr_tt_history_thresh, i32, DEFAULT_LMR_TT_HISTORY_THRESH);

// History Leaf Pruning
define_accessor!(hlp_max_depth, usize, DEFAULT_HLP_MAX_DEPTH);
define_accessor!(hlp_min_moves, usize, DEFAULT_HLP_MIN_MOVES);
define_accessor!(hlp_history_reduce, i32, DEFAULT_HLP_HISTORY_REDUCE);
define_accessor!(hlp_history_leaf, i32, DEFAULT_HLP_HISTORY_LEAF);

// Late Move Pruning
define_accessor!(lmp_base, usize, DEFAULT_LMP_BASE);
define_accessor!(lmp_depth_mult, usize, DEFAULT_LMP_DEPTH_MULT);

// Aspiration
define_accessor!(aspiration_window, i32, DEFAULT_ASPIRATION_WINDOW);
define_accessor!(aspiration_fail_mult, i32, DEFAULT_ASPIRATION_FAIL_MULT);
define_accessor!(aspiration_max_window, i32, DEFAULT_ASPIRATION_MAX_WINDOW);

// Reverse Futility Pruning
define_accessor!(rfp_max_depth, usize, DEFAULT_RFP_MAX_DEPTH);
define_accessor!(rfp_mult_tt, i32, DEFAULT_RFP_MULT_TT);
define_accessor!(rfp_mult_no_tt, i32, DEFAULT_RFP_MULT_NO_TT);
define_accessor!(rfp_improving_mult, i32, DEFAULT_RFP_IMPROVING_MULT);
define_accessor!(rfp_worsening_mult, i32, DEFAULT_RFP_WORSENING_MULT);

// ProbCut
define_accessor!(probcut_margin, i32, DEFAULT_PROBCUT_MARGIN);
define_accessor!(probcut_improving, i32, DEFAULT_PROBCUT_IMPROVING);
define_accessor!(probcut_min_depth, usize, DEFAULT_PROBCUT_MIN_DEPTH);
define_accessor!(probcut_depth_sub, usize, DEFAULT_PROBCUT_DEPTH_SUB);
define_accessor!(probcut_divisor, i32, DEFAULT_PROBCUT_DIVISOR);
define_accessor!(
    low_depth_probcut_margin,
    i32,
    DEFAULT_LOW_DEPTH_PROBCUT_MARGIN
);

// IIR
define_accessor!(iir_min_depth, usize, DEFAULT_IIR_MIN_DEPTH);

// SEE Pruning
define_accessor!(see_capture_linear, i32, DEFAULT_SEE_CAPTURE_LINEAR);
define_accessor!(see_capture_hist_div, i32, DEFAULT_SEE_CAPTURE_HIST_DIV);
define_accessor!(see_quiet_quad, i32, DEFAULT_SEE_QUIET_QUAD);
define_accessor!(see_winning_threshold, i32, DEFAULT_SEE_WINNING_THRESHOLD);

// Move Ordering
define_accessor!(sort_hash, i32, DEFAULT_SORT_HASH);
define_accessor!(sort_winning_capture, i32, DEFAULT_SORT_WINNING_CAPTURE);
define_accessor!(sort_killer1, i32, DEFAULT_SORT_KILLER1);
define_accessor!(sort_killer2, i32, DEFAULT_SORT_KILLER2);
define_accessor!(sort_countermove, i32, DEFAULT_SORT_COUNTERMOVE);

// History
define_accessor!(max_history, i32, DEFAULT_MAX_HISTORY);
define_accessor!(history_bonus_base, i32, DEFAULT_HISTORY_BONUS_BASE);
define_accessor!(history_bonus_sub, i32, DEFAULT_HISTORY_BONUS_SUB);
define_accessor!(history_bonus_cap, i32, DEFAULT_HISTORY_BONUS_CAP);

// Other
define_accessor!(repetition_penalty, i32, DEFAULT_REPETITION_PENALTY);
define_accessor!(delta_margin, i32, DEFAULT_DELTA_MARGIN);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        assert_eq!(razoring_linear(), DEFAULT_RAZORING_LINEAR);
    }
}
