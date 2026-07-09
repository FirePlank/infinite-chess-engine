use super::{INFINITY, MATE_SCORE, MATE_VALUE, MAX_PLY};
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

/// The i16 eval-field value reserved to mean "no static eval stored". Maps
/// to/from the INFINITY+1 sentinel that search uses for an uncomputed eval.
pub const TT_NO_EVAL: i16 = i16::MIN;

/// Pack a static eval into the i16 eval field. The "not computed" sentinel
/// (INFINITY + 1) becomes TT_NO_EVAL; real evals clamp into the remaining range
/// (i16::MIN is reserved, so the floor is i16::MIN + 1). Inverse of [`eval_from_i16`].
#[inline]
pub fn eval_to_i16(v: i32) -> i16 {
    if v == INFINITY + 1 {
        TT_NO_EVAL
    } else {
        v.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16
    }
}

/// Expand a stored i16 eval (sign-extended to i32) back to an engine eval.
/// TT_NO_EVAL maps to the INFINITY+1 sentinel so callers recompute the eval.
#[inline]
pub fn eval_from_i16(s: i32) -> i32 {
    if s == TT_NO_EVAL as i32 {
        INFINITY + 1
    } else {
        s
    }
}


// The engine's decisive constants are large so that even huge evaluations on an unbounded
// board cannot be mistaken for a mate. They do NOT fit in i16.
//
// Real mate scores always lie within MAX_PLY of MATE_VALUE, and `value_to_tt`
// can shift them by up to another MAX_PLY, so they occupy a band of width
// 2*MAX_PLY+1. We reserve that band at each end of the i16 range and map mate
// scores into it losslessly; normal scores are clamped to the remaining range.

/// Number of distinct i16 slots reserved for mate scores at each end of the range.
const TT_MATE_BAND: i32 = 2 * MAX_PLY as i32 + 1;
/// Largest magnitude a normal (non-mate) score may occupy in the i16 field.
const TT_SCORE_NORMAL_MAX: i32 = i16::MAX as i32 - TT_MATE_BAND;

/// Pack a (ply-adjusted) node score into the i16 TT score field, preserving mate
/// scores. Inverse of [`score_from_i16`].
#[inline]
pub fn score_to_i16(v: i32) -> i16 {
    if v > MATE_SCORE {
        // Closer mates (larger v) map to larger stored values, just below i16::MAX.
        let offset = (MATE_VALUE - v + MAX_PLY as i32).clamp(0, TT_MATE_BAND - 1);
        (i16::MAX as i32 - offset) as i16
    } else if v < -MATE_SCORE {
        let offset = (MATE_VALUE + v + MAX_PLY as i32).clamp(0, TT_MATE_BAND - 1);
        (-(i16::MAX as i32) + offset) as i16
    } else {
        v.clamp(-TT_SCORE_NORMAL_MAX, TT_SCORE_NORMAL_MAX) as i16
    }
}

/// Expand a stored i16 TT score back to the engine's score range. The argument
/// is the stored value sign-extended to i32. Inverse of [`score_to_i16`].
#[inline]
pub fn score_from_i16(s: i32) -> i32 {
    if s > TT_SCORE_NORMAL_MAX {
        let offset = i16::MAX as i32 - s;
        MATE_VALUE - (offset - MAX_PLY as i32)
    } else if s < -TT_SCORE_NORMAL_MAX {
        let offset = i16::MAX as i32 + s;
        -MATE_VALUE + (offset - MAX_PLY as i32)
    } else {
        s
    }
}

#[cfg(test)]
mod score_pack_tests {
    use super::*;

    #[test]
    fn normal_scores_round_trip() {
        for v in [0, 1, -1, 100, -100, 5000, -5000, TT_SCORE_NORMAL_MAX, -TT_SCORE_NORMAL_MAX] {
            assert_eq!(score_from_i16(score_to_i16(v) as i32), v, "v = {v}");
        }
    }

    #[test]
    fn mate_scores_round_trip() {
        // Cover mate scores as produced by value_to_tt across the legal ply range.
        for ply in 0..=MAX_PLY {
            for dist in 0..=MAX_PLY {
                let win = MATE_VALUE - dist as i32; // mate in `dist`
                let adj = value_to_tt(win, ply);
                let back = score_from_i16(score_to_i16(adj) as i32);
                assert_eq!(back, adj, "win mate dist={dist} ply={ply}");

                let loss = -(MATE_VALUE - dist as i32);
                let adj = value_to_tt(loss, ply);
                let back = score_from_i16(score_to_i16(adj) as i32);
                assert_eq!(back, adj, "loss mate dist={dist} ply={ply}");
            }
        }
    }

    #[test]
    fn value_from_tt_downgrades_unreachable_mate() {
        // A mate in 5 stored at ply 0. With a finite move rule that the mate
        // would overrun (rule50 already high), it must downgrade to the largest
        // non-mate score; with no/ample rule it stays a true mate.
        let mate_in_5 = MATE_VALUE - 5;

        // Unreachable: 5 plies to mate + 98 half-moves already > 100 limit.
        assert_eq!(value_from_tt(mate_in_5, 0, 98, 100), MATE_SCORE - 1);
        let mated_in_5 = -(MATE_VALUE - 5);
        assert_eq!(value_from_tt(mated_in_5, 0, 98, 100), -MATE_SCORE + 1);

        // Reachable under the limit: passes through as a true mate.
        assert!(value_from_tt(mate_in_5, 0, 10, 100) > MATE_SCORE);

        // No move rule (rule_limit = i32::MAX): a realistic (even large)
        // halfmove clock never downgrades, and the i64 compare does not overflow.
        assert!(value_from_tt(mate_in_5, 0, 1_000_000, i32::MAX) > MATE_SCORE);
    }

    #[test]
    fn mate_and_normal_bands_are_disjoint() {
        // A near-mate value survives as a recognizable mate (> MATE_SCORE)...
        let stored = score_to_i16(MATE_VALUE - 3);
        assert!(score_from_i16(stored as i32) > MATE_SCORE);
        // ...while the largest normal score stays below the mate threshold.
        let stored = score_to_i16(TT_SCORE_NORMAL_MAX);
        assert!(score_from_i16(stored as i32) <= MATE_SCORE);
    }

    #[test]
    fn eval_no_eval_sentinel_round_trips() {
        // The "not computed" sentinel packs to TT_NO_EVAL and unpacks back to it,
        // so callers recompute instead of trusting a bogus clamped value.
        assert_eq!(eval_to_i16(INFINITY + 1), TT_NO_EVAL);
        assert_eq!(eval_from_i16(TT_NO_EVAL as i32), INFINITY + 1);
    }

    #[test]
    fn eval_real_values_never_collide_with_sentinel() {
        // Real evals (including ones that clamp to the i16 extremes) must never
        // pack to TT_NO_EVAL, and must round-trip to their clamped value.
        for v in [0, 1, -1, 250, -250, 30000, -30000, 100_000, -100_000, i32::MAX, i32::MIN] {
            let packed = eval_to_i16(v);
            assert_ne!(packed, TT_NO_EVAL, "real eval {v} collided with NO_EVAL");
            let clamped = v.clamp(i16::MIN as i32 + 1, i16::MAX as i32);
            assert_eq!(eval_from_i16(packed as i32), clamped, "v = {v}");
        }
        // Previously the sentinel clamped to +32767; verify a genuine +32767 eval
        // is still distinguishable from "no eval".
        assert_ne!(eval_to_i16(32767), TT_NO_EVAL);
        assert_eq!(eval_from_i16(eval_to_i16(32767) as i32), 32767);
    }
}

#[inline]
pub fn pack_coord(c: i64) -> u64 {
    (c.clamp(MIN_TT_COORD, MAX_TT_COORD) & COORD_MASK as i64) as u64
}

#[inline]
pub fn unpack_coord(v: u64) -> i64 {
    let mut val = (v & COORD_MASK) as i64;
    if val >= (1 << (COORD_BITS - 1)) {
        val -= 1 << COORD_BITS;
    }
    val
}

/// Value adjustment for retrieval:
/// Inverse of value_to_tt: adjusts TT score back to root-relative.
/// Downgrades mate scores that are unreachable due to the 50-move rule.
#[inline]
pub fn value_from_tt(value: i32, ply: usize, rule50_count: u32, rule_limit: i32) -> i32 {
    // Handle winning mate scores (we are giving mate).
    if value > MATE_SCORE {
        // Plies until mate from the stored position.
        let mate_distance = MATE_VALUE - value;

        // If the 50-move rule forces a draw before mate is delivered, the mate
        // is unproven: report the largest non-mate winning score. i64 avoids
        // overflow when rule_limit is i32::MAX (variants with no move rule).
        if mate_distance as i64 + rule50_count as i64 > rule_limit as i64 {
            return MATE_SCORE - 1;
        }

        // Adjust back to root-relative
        return value - ply as i32;
    }

    // Handle losing mate scores (we are being mated).
    if value < -MATE_SCORE {
        let mate_distance = MATE_VALUE + value;

        if mate_distance as i64 + rule50_count as i64 > rule_limit as i64 {
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

/// Promotion type is 5 bits (supporting 32 types)
pub const PROMO_BITS: u32 = 5;

/// Coordinate bit-packing (13 bits = +/- 4096 range)
pub const COORD_BITS: u32 = 13;
pub const COORD_MASK: u64 = (1 << COORD_BITS) - 1;
pub const MAX_TT_COORD: i64 = (1 << (COORD_BITS - 1)) - 1;
pub const MIN_TT_COORD: i64 = -MAX_TT_COORD - 1;

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
    pub cutoff_score: i32,
    pub tt_score: i32,
    pub eval: i32,
    pub depth: u8,
    pub flag: TTFlag,
    pub is_pv: bool,
    pub best_move: Option<Move>,
}
