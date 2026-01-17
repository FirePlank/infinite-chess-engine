use super::{MATE_SCORE, MATE_VALUE};
use crate::moves::Move;

// ============================================================================
// Value Adjustment Constants & Helpers
// ============================================================================

/// Value adjustment for storage:
/// Adjusts a mate score from "plies to mate from the root" to
/// "plies to mate from the current position" for storage in TT.
/// Standard scores are unchanged.
#[inline]
pub fn value_to_tt(value: i32, ply: usize) -> i32 {
    // is_win: value > MATE_SCORE (positive mate score)
    if value > MATE_SCORE {
        value + ply as i32
    }
    // is_loss: value < -MATE_SCORE (negative mate score, being mated)
    else if value < -MATE_SCORE {
        value - ply as i32
    } else {
        value
    }
}

/// Value adjustment for retrieval:
/// Inverse of value_to_tt: adjusts TT score back to root-relative.
/// Downgrades mate scores that are unreachable due to the 50-move rule.
#[inline]
pub fn value_from_tt(value: i32, ply: usize, rule50_count: u32, rule_limit: i32) -> i32 {
    // Handle winning mate scores (we are giving mate)
    if value > MATE_SCORE {
        // mate_distance = how many plies until mate from the stored position
        let mate_distance = MATE_VALUE - value;

        // Downgrade a potentially false mate score:
        // If mate_distance + rule50_count > rule_limit, the game would be drawn
        // by the 50-move rule before we can deliver checkmate.
        if mate_distance + rule50_count as i32 > rule_limit {
            // Downgrade to non-mate winning score (just below mate threshold)
            return MATE_SCORE - 1;
        }

        // Adjust back to root-relative
        return value - ply as i32;
    }

    // Handle losing mate scores (we are being mated)
    if value < -MATE_SCORE {
        let mate_distance = MATE_VALUE + value;

        // Downgrade a potentially false mate score
        if mate_distance + rule50_count as i32 > rule_limit {
            // Downgrade to non-mate losing score
            return -MATE_SCORE + 1;
        }

        return value + ply as i32;
    }

    value
}

// ============================================================================
// TT Types
// ============================================================================

/// TT bound type (2 bits)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    None = 0,
    UpperBound = 1, // Score <= alpha (all node, failed low)
    LowerBound = 2, // Score >= beta (cut node, failed high)
    Exact = 3,      // Exact score (PV node)
}

impl TTFlag {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v & 0b11) }
    }
}

/// Parameters for probing the TT
pub struct TTProbeParams {
    pub hash: u64,
    pub alpha: i32,
    pub beta: i32,
    pub depth: usize,
    pub ply: usize,
    pub rule50_count: u32,
    pub rule_limit: i32,
}

/// Parameters for storing to the TT
pub struct TTStoreParams {
    pub hash: u64,
    pub depth: usize,
    pub flag: TTFlag,
    pub score: i32,
    pub static_eval: i32,
    pub is_pv: bool,
    pub best_move: Option<Move>,
    pub ply: usize,
}

/// Result from a TT probe
#[derive(Debug, Clone, Copy)]
pub struct TTProbeResult {
    pub cutoff_score: i32, // Result of old probe(): includes depth check, bounds, and 50-move rule
    pub tt_score: i32, // Result of old probe_for_singular(): only hash check, simplified mate logic
    pub eval: i32,
    pub depth: u8,
    pub flag: TTFlag,
    pub is_pv: bool,
    pub best_move: Option<Move>,
}
