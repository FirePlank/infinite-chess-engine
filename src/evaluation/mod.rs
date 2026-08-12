//! Evaluation dispatch. `base` holds every default heuristic; a file under
//! `variants/` exists only where a variant needs something base cannot express.

pub mod base;
pub mod eval_kind;
pub mod helpers;
pub mod insufficient_material;
pub mod mop_up;
pub mod variants;

use crate::board::PlayerColor;
use crate::game::GameState;
use eval_kind::EvalKind;

pub use base::{calculate_initial_material, get_piece_phase, get_piece_value_base};

/// Largest slice of the evaluation halfmove-clock damping may remove. Big enough to
/// keep the winner pressing against the move-rule clock, small enough per tick that
/// stale TT scores never outweigh real progress.
const RULE50_DAMP_CAP: i32 = 700;

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub use crate::search::params::{
    EVAL_PARAMS, EvalParamSpec, EvalParams, TUNABLE_EVAL_PARAM_SPECS, get_eval_params_as_json,
    set_eval_params, set_eval_params_from_json,
};
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub use base::{EVAL_FEATURES, EvalFeatures, reset_eval_features, snapshot_eval_features};

/// Mop-up bonus from the side-to-move's perspective. Activation, scaling and
/// saturation live in `mop_up`, shared with the search's skill policy.
#[inline]
fn compute_mop_up_term(game: &GameState) -> i32 {
    let Some((winner, scale)) = crate::evaluation::mop_up::active_mop_up(game) else {
        return 0;
    };
    let term = crate::evaluation::mop_up::evaluate_mop_up_scaled(game, winner, scale);
    let raw = if winner == PlayerColor::White {
        term
    } else {
        -term
    };
    if game.turn == PlayerColor::Black {
        -raw
    } else {
        raw
    }
}

/// Bounded-board rook/minor endings that are drawn with correct defense are
/// scaled hard toward the draw.
fn apply_bounded_drawish_scale(game: &GameState, eval: i32) -> i32 {
    bounded_drawish_scale_inner(game, eval, crate::moves::get_world_size())
}

fn bounded_drawish_scale_inner(game: &GameState, eval: i32, world_size: i64) -> i32 {
    if eval == 0 || world_size > 200 {
        return eval;
    }
    if game.white_royals.len() != 1 || game.black_royals.len() != 1 {
        return eval;
    }
    // Small endings only (up to two non-king pieces per side).
    if game.white_piece_count + game.black_piece_count > 6 {
        return eval;
    }

    use crate::board::PieceType;
    // Only rook, minor and pawn endings qualify: with a queen or other heavy on the
    // board the advantage is a genuine win the normal eval already handles.
    let (mut w_npm, mut b_npm) = (0i32, 0i32);
    let (mut w_pawns, mut b_pawns) = (false, false);
    for (_, _, piece) in game.board.iter() {
        let pt = piece.piece_type();
        if pt.is_royal() {
            continue;
        }
        let white = piece.color() == PlayerColor::White;
        match pt {
            PieceType::Rook | PieceType::Knight | PieceType::Bishop => {
                let v = get_piece_value_base(pt);
                if white {
                    w_npm += v;
                } else {
                    b_npm += v;
                }
            }
            PieceType::Pawn => {
                if white {
                    w_pawns = true;
                } else {
                    b_pawns = true;
                }
            }
            _ => return eval,
        }
    }

    // Identify the stronger side by non-pawn material; an equal split (e.g.
    // R+P vs R) is left to the normal eval, which knows KRPKR can be won.
    let (strong_npm, weak_npm, strong_has_pawns, strong_is_white) = if w_npm > b_npm {
        (w_npm, b_npm, w_pawns, true)
    } else if b_npm > w_npm {
        (b_npm, w_npm, b_pawns, false)
    } else {
        return eval;
    };

    // A pawn for the stronger side is a real winning try (it can promote).
    if strong_has_pawns {
        return eval;
    }
    // Only scale the stronger side's own advantage claim (eval is from the
    // side-to-move's perspective).
    let strong_to_move = (game.turn == PlayerColor::White) == strong_is_white;
    if (eval > 0) != strong_to_move {
        return eval;
    }
    // Bare piece edge (≤ a minor) with no pawns cannot force mate: fortress.
    if strong_npm - weak_npm <= get_piece_value_base(PieceType::Bishop) {
        eval / 8
    } else {
        eval
    }
}

/// Halfmove-clock damping, applying rising pressure to make progress. Mop-up
/// conversion damps only a capped slice: TT entries do not know the clock, so fully
/// damping a huge mop-up eval lets stale shuffle scores beat fresh progress.
#[inline]
fn apply_rule50_damping(game: &GameState, raw_eval: i32, mop_up_active: bool) -> i32 {
    match game.game_rules.move_rule_limit {
        Some(limit) if limit > 0 => {
            let divisor = 2 * limit as i32 - 1;
            let clock = (game.halfmove_clock as i32).min(divisor);
            let dampable = if mop_up_active {
                raw_eval.clamp(-RULE50_DAMP_CAP, RULE50_DAMP_CAP)
            } else {
                raw_eval
            };
            raw_eval - (dampable * clock) / divisor
        }
        _ => raw_eval,
    }
}

/// Main evaluation entry point - NNUE Enabled
#[inline]
#[cfg(feature = "nnue")]
pub fn evaluate(game: &GameState, nnue_state: Option<&crate::nnue::NnueState>) -> i32 {
    if insufficient_material::evaluate_insufficient_material(game) {
        return 0;
    }
    let raw_eval = match game.eval_kind {
        EvalKind::Chess => variants::chess::evaluate(game),
        EvalKind::Obstocean => variants::obstocean::evaluate(game),
        EvalKind::PawnHorde => variants::pawn_horde::evaluate(game),
        EvalKind::Generic => {
            // Try NNUE first if applicable (standard pieces, kings present, weights loaded)
            if crate::nnue::is_applicable(game) {
                if let Some(state) = nnue_state {
                    crate::nnue::evaluate_with_state(game, state)
                } else {
                    crate::nnue::evaluate(game)
                }
            } else {
                base::evaluate(game)
            }
        }
    };
    let mop_up = compute_mop_up_term(game);

    apply_rule50_damping(
        game,
        apply_bounded_drawish_scale(game, raw_eval + mop_up),
        mop_up != 0,
    )
}

/// Main evaluation entry point - NNUE Disabled
#[inline]
#[cfg(not(feature = "nnue"))]
pub fn evaluate(game: &GameState) -> i32 {
    if insufficient_material::evaluate_insufficient_material(game) {
        return 0;
    }
    let raw_eval = match game.eval_kind {
        EvalKind::Chess => variants::chess::evaluate(game),
        EvalKind::Obstocean => variants::obstocean::evaluate(game),
        EvalKind::PawnHorde => variants::pawn_horde::evaluate(game),
        EvalKind::Generic => base::evaluate(game),
    };
    let mop_up = compute_mop_up_term(game);

    apply_rule50_damping(
        game,
        apply_bounded_drawish_scale(game, raw_eval + mop_up),
        mop_up != 0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{PieceType, PlayerColor};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8")
    }

    fn create_test_game_from_icn(icn: &str) -> GameState {
        let mut game = GameState::new();
        game.setup_position_from_icn(icn);
        game
    }

    #[inline]
    fn evaluate_wrapper(game: &GameState) -> i32 {
        #[cfg(feature = "nnue")]
        return evaluate(game, None);
        #[cfg(not(feature = "nnue"))]
        return evaluate(game);
    }

    #[test]
    fn test_bounded_drawish_endings_scaled() {
        // R+minor vs R and R vs lone minor are drawn with correct defense on a
        // bounded board, so the stronger pawnless side must not keep its full
        // material claim.
        for icn in [
            "w (8;q|1;q) K2,2|R4,4|N5,5|k7,7|r7,1", // R+N vs R
            "w (8;q|1;q) K2,2|R4,4|B5,4|k7,7|r7,1", // R+B vs R
            "w (8;q|1;q) K2,2|R4,4|k7,7|n6,1",      // R vs N
            "w (8;q|1;q) K2,2|R4,4|k7,7|b6,1",      // R vs B
        ] {
            let game = create_test_game_from_icn(icn);
            assert_eq!(
                bounded_drawish_scale_inner(&game, 400, 8),
                50,
                "strong side's winning claim must scale toward the draw: {}",
                icn
            );
        }

        // A defender pawn's counterplay is never masked: an eval favoring the
        // weaker (pawned) side keeps full volume.
        let game = create_test_game_from_icn("w (8;q|1;q) K2,2|R4,4|N5,5|k7,7|r7,1|p6,2");
        assert_eq!(
            bounded_drawish_scale_inner(&game, -547, 8),
            -547,
            "danger from the defender's promoting pawn must not be scaled"
        );
        // But the strong side's own claim still scales in the same ending.
        assert_eq!(bounded_drawish_scale_inner(&game, 400, 8), 50);

        // Unbounded world: never scaled.
        let unb = create_test_game_from_icn("w (8;q|1;q) K2,2|R4,4|N5,5|k7,7|r7,1");
        assert_eq!(
            bounded_drawish_scale_inner(&unb, 400, 1_000_000),
            400,
            "unbounded boards are handled by insufficient-material, not scaling"
        );
    }

    #[test]
    fn test_evaluate_returns_value() {
        let game = create_test_game();
        let score = evaluate_wrapper(&game);
        // K vs K should be close to 0
        assert!(score.abs() < 1000, "K vs K should be near 0");
    }

    #[test]
    fn test_evaluate_material_advantage() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,4|r1,8");

        let eval = evaluate_wrapper(&game);
        // Just verify we get a reasonable value (queen > rook typically)
        // Exact values depend on evaluation logic
        assert!(
            eval.abs() < 100000,
            "Eval should be in reasonable range, got {}",
            eval
        );
    }

    #[test]
    fn test_evaluate_with_variant() {
        let mut game = create_test_game();
        game.variant = Some(crate::Variant::Chess);

        // Should dispatch to Chess evaluator
        let _score = evaluate_wrapper(&game);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_evaluate_rule50_damping() {
        let mut game = create_test_game();
        game.setup_position_from_icn("w (8;q|1;q) K5,1|k5,8|R0,0|R1,0");
        game.material_score = 1000;
        game.turn = PlayerColor::White;
        game.game_rules.move_rule_limit = Some(100);

        let eval_0 = evaluate_wrapper(&game);

        game.halfmove_clock = 50;
        let eval_50 = evaluate_wrapper(&game);

        game.halfmove_clock = 100;
        let eval_100 = evaluate_wrapper(&game);

        assert!(eval_50 < eval_0, "Eval should decrease at clock=50");
        assert!(
            eval_100 < eval_50,
            "Eval should decrease further at clock=100"
        );
        assert!(eval_100 > 0, "Eval should not drop to 0 at the limit");

        // Only a capped slice of the eval is damped:
        // delta(clock) = min(|eval_0|, cap) * clock / (2 * limit - 1)
        let dampable = eval_0.clamp(-RULE50_DAMP_CAP, RULE50_DAMP_CAP);
        let delta_50 = eval_0 - eval_50;
        let expected_delta_50 = (dampable * 50) / 199;
        assert!(
            (delta_50 - expected_delta_50).abs() < 2,
            "Delta 50: expected {}, got {}",
            expected_delta_50,
            delta_50
        );

        let delta_100 = eval_0 - eval_100;
        let expected_delta_100 = (dampable * 100) / 199;
        assert!(
            (delta_100 - expected_delta_100).abs() < 2,
            "Delta 100: expected {}, got {}",
            expected_delta_100,
            delta_100
        );
    }

    #[test]
    fn test_get_piece_value() {
        // Test piece values are reasonable
        let queen_val = get_piece_value_base(PieceType::Queen);
        let rook_val = get_piece_value_base(PieceType::Rook);
        let bishop_val = get_piece_value_base(PieceType::Bishop);
        let knight_val = get_piece_value_base(PieceType::Knight);
        let pawn_val = get_piece_value_base(PieceType::Pawn);

        assert!(queen_val > rook_val, "Queen should be worth more than rook");
        assert!(
            rook_val > bishop_val,
            "Rook should be worth more than bishop"
        );
        assert!(
            bishop_val > knight_val,
            "Bishop should be worth more than knight"
        );
        assert!(
            knight_val > pawn_val,
            "Knight should be worth more than pawn"
        );
        assert!(pawn_val > 0, "Pawn should have positive value");
    }
}
