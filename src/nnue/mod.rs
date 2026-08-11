//! Quantized NNUE evaluation with two streams: RelKP (king-relative piece positions,
//! 25450 to 256) and ThreatEdges (attack/defense relationships, 6768 to 64).
//! Accumulators update incrementally on make/undo; weights are embedded at compile time.

mod features;
mod inference;
mod state;
mod weights;

#[cfg(test)]
mod tests;

pub use features::{build_relkp_active_lists, build_threat_active_lists};
pub use inference::{evaluate, evaluate_with_state};
pub use state::NnueState;
pub use weights::{NNUE_WEIGHTS, NnueWeights};

use crate::board::{PieceType, PlayerColor};
use crate::game::GameState;

/// NNUE applies only to positions of standard chess pieces with exactly one king per
/// side and no obstacles, voids, or fairy pieces.
#[inline]
pub fn is_applicable(gs: &GameState) -> bool {
    // The RelKP encoding anchors on a single king per side; a second same-color
    // king cannot be represented, so multi-royal positions fall back to HCE.
    if gs.white_royals.len() != 1 || gs.black_royals.len() != 1 {
        return false;
    }

    // Check all pieces are standard chess pieces
    for (_x, _y, piece) in gs.board.iter_all_pieces() {
        let pt = piece.piece_type();
        let color = piece.color();

        // Skip neutral pieces (obstacles, voids)
        if color == PlayerColor::Neutral {
            return false;
        }

        // Only standard chess pieces allowed
        match pt {
            PieceType::King
            | PieceType::Queen
            | PieceType::Rook
            | PieceType::Bishop
            | PieceType::Knight
            | PieceType::Pawn => {}
            _ => return false,
        }
    }

    true
}

/// Initialize NNUE state from scratch for a position.
pub fn init_state(gs: &GameState) -> NnueState {
    NnueState::from_position(gs)
}

#[cfg(test)]
mod mod_tests {
    use super::*;
    use crate::board::{Coordinate, Piece, PieceType, PlayerColor};

    #[test]
    fn test_is_applicable() {
        let mut gs = GameState::new();
        // Missing kings
        assert!(!is_applicable(&gs));

        gs.white_royals.push(Coordinate::new(4, 0));
        gs.black_royals.push(Coordinate::new(4, 7));
        // Kings present, but board empty
        assert!(is_applicable(&gs));

        // Standard piece
        gs.board
            .set_piece(0, 0, Piece::new(PieceType::Pawn, PlayerColor::White));
        assert!(is_applicable(&gs));

        // Fairy piece
        gs.board
            .set_piece(0, 1, Piece::new(PieceType::Amazon, PlayerColor::White));
        assert!(!is_applicable(&gs));

        // Obstacle
        gs.board.remove_piece(&0, &1);
        gs.board
            .set_piece(0, 1, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
        assert!(!is_applicable(&gs));
    }

    #[test]
    fn test_init_state_no_panic() {
        let mut gs = GameState::new();
        gs.white_royals.push(Coordinate::new(4, 0));
        gs.black_royals.push(Coordinate::new(4, 7));
        let _state = init_state(&gs);
    }
}
