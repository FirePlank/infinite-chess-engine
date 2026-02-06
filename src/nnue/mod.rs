//! NNUE (Efficiently Updatable Neural Network) Evaluation for Infinite Chess
//!
//! This module provides a quantized neural network evaluation trained on
//! self-play games. The NNUE uses a two-stream architecture:
//!
//! - **RelKP Stream**: Translation-invariant piece positions relative to king (25450 → 256)
//! - **ThreatEdges Stream**: Attack/defense relationships (6768 → 64)
//!
//! The accumulators are updated incrementally during make/undo for O(1) feature updates.
//! For WASM compatibility, weights are embedded at compile time using `include_bytes!`.

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

/// Check if NNUE evaluation is applicable to this position.
///
/// NNUE is only used when:
/// - Every piece is a standard chess piece (K, Q, R, B, N, P)
/// - Exactly one king per side exists
/// - No obstacles, voids, or fairy pieces are present
#[inline]
pub fn is_applicable(gs: &GameState) -> bool {
    // Must have exactly one king per side
    if gs.white_king_pos.is_none() || gs.black_king_pos.is_none() {
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

        gs.white_king_pos = Some(Coordinate::new(4, 0));
        gs.black_king_pos = Some(Coordinate::new(4, 7));
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
        gs.white_king_pos = Some(Coordinate::new(4, 0));
        gs.black_king_pos = Some(Coordinate::new(4, 7));
        let _state = init_state(&gs);
    }
}
