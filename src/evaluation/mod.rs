// Modular Evaluation System
//
// Design Principles:
// 1. `base` contains ALL default heuristics
// 2. Variant files in `variants/` ONLY exist if they have special logic

pub mod base;
pub mod helpers;
pub mod insufficient_material;
pub mod mop_up;
pub mod variants;

use crate::Variant;
use crate::game::GameState;

pub use base::{calculate_initial_material, get_piece_phase, get_piece_value};

#[cfg(feature = "eval_tuning")]
pub use base::{EVAL_FEATURES, EvalFeatures, reset_eval_features, snapshot_eval_features};

/// Main evaluation entry point - NNUE Enabled
#[inline]
#[cfg(feature = "nnue")]
pub fn evaluate(game: &GameState, nnue_state: Option<&crate::nnue::NnueState>) -> i32 {
    let raw_eval = match game.variant {
        Some(Variant::Chess) => variants::chess::evaluate(game),
        Some(Variant::Obstocean) => variants::obstocean::evaluate(game),
        Some(Variant::PawnHorde) => variants::pawn_horde::evaluate(game),
        // Add new variants here as they get custom evaluators
        _ => {
            // Try NNUE first if applicable (standard pieces, kings present, weights loaded)
            if crate::nnue::is_applicable(game) {
                if let Some(state) = nnue_state {
                    crate::nnue::evaluate_with_state(game, state) + base::compute_mop_up_term(game)
                } else {
                    crate::nnue::evaluate(game) + base::compute_mop_up_term(game)
                }
            } else {
                base::evaluate(game)
            }
        } // Default: use base for all others
    };

    // As the halfmove clock increases during shuffling, we slightly damp the
    // evaluation. This provides a gentle pressure to "get on with it" and
    // avoid unnecessary repetitions or shuffling.
    let rule_limit = game.game_rules.move_rule_limit.unwrap_or(100) as i32;
    if rule_limit > 0 {
        let divisor = 2 * rule_limit - 1;
        raw_eval - (raw_eval * game.halfmove_clock as i32) / divisor
    } else {
        raw_eval
    }
}

/// Main evaluation entry point - NNUE Disabled
#[inline]
#[cfg(not(feature = "nnue"))]
pub fn evaluate(game: &GameState) -> i32 {
    let raw_eval = match game.variant {
        Some(Variant::Chess) => variants::chess::evaluate(game),
        Some(Variant::Obstocean) => variants::obstocean::evaluate(game),
        Some(Variant::PawnHorde) => variants::pawn_horde::evaluate(game),
        // Add new variants here as they get custom evaluators
        _ => base::evaluate(game), // Default: use base for all others
    };

    // As the halfmove clock increases during shuffling, we slightly damp the
    // evaluation. This provides a gentle pressure to "get on with it" and
    // avoid unnecessary repetitions or shuffling.
    let rule_limit = game.game_rules.move_rule_limit.unwrap_or(100) as i32;
    if rule_limit > 0 {
        let divisor = 2 * rule_limit - 1;
        raw_eval - (raw_eval * game.halfmove_clock as i32) / divisor
    } else {
        raw_eval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece, PieceType, PlayerColor};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();
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
    fn test_evaluate_returns_value() {
        let game = create_test_game();
        let score = evaluate_wrapper(&game);
        // K vs K should be close to 0
        assert!(score.abs() < 1000, "K vs K should be near 0");
    }

    #[test]
    fn test_evaluate_material_advantage() {
        let mut game = GameState::new();
        game.board = Board::new();

        // More complex position to avoid lone king detection
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();

        // Recalculate material score
        let mut score = 0i32;
        for (_, _, piece) in game.board.iter() {
            let val = get_piece_value(piece.piece_type());
            match piece.color() {
                PlayerColor::White => score += val,
                PlayerColor::Black => score -= val,
                PlayerColor::Neutral => {}
            }
        }
        game.material_score = score;

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
        game.board
            .set_piece(0, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(1, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        game.recompute_piece_counts();
        game.material_score = 1000;
        game.turn = PlayerColor::White;
        game.game_rules.move_rule_limit = Some(100);

        let eval_0 = evaluate_wrapper(&game);

        game.halfmove_clock = 50;
        let eval_50 = evaluate_wrapper(&game);

        game.halfmove_clock = 100;
        let eval_100 = evaluate_wrapper(&game);

        // With 100-move rule limit (200 halfmoves), at 100 halfmoves
        // the damping should be roughly halving the evaluation.
        // Formula: v -= v * clock / (2 * limit - 1)
        // v = 1000, clock = 50, limit = 100 -> 1000 - (1000 * 50 / 199) = 1000 - 251 = 749
        // v = 1000, clock = 100, limit = 100 -> 1000 - (1000 * 100 / 199) = 1000 - 502 = 498

        assert!(eval_50 < eval_0, "Eval should decrease at clock=50");
        assert!(
            eval_100 < eval_50,
            "Eval should decrease further at clock=100"
        );
        assert!(eval_100 > 0, "Eval should not drop to 0 at the limit");

        // Approximate values check
        let delta_50 = eval_0 - eval_50;
        let expected_delta_50 = (eval_0 * 50) / 199;
        assert!(
            (delta_50 - expected_delta_50).abs() < 2,
            "Delta 50: expected {}, got {}",
            expected_delta_50,
            delta_50
        );

        let delta_100 = eval_0 - eval_100;
        let expected_delta_100 = (eval_0 * 100) / 199;
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
        let queen_val = get_piece_value(PieceType::Queen);
        let rook_val = get_piece_value(PieceType::Rook);
        let bishop_val = get_piece_value(PieceType::Bishop);
        let knight_val = get_piece_value(PieceType::Knight);
        let pawn_val = get_piece_value(PieceType::Pawn);

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
