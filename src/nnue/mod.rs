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