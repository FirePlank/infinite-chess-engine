//! NNUE Accumulator State
//!
//! Stores the RelKP accumulator state for incremental updates.
//! The threat stream is computed on-the-fly since it's fast enough.

use super::features::build_relkp_active_lists;
use super::weights::NNUE_WEIGHTS;
use crate::game::GameState;

/// Accumulator dimensions
pub const RELKP_DIM: usize = 256;

/// NNUE accumulator state for a position.
/// Contains pre-computed RelKP accumulator for both perspectives.
#[derive(Clone)]
pub struct NnueState {
    /// White perspective RelKP accumulator
    pub rel_acc_white: [i16; RELKP_DIM],
    /// Black perspective RelKP accumulator
    pub rel_acc_black: [i16; RELKP_DIM],
}

impl Default for NnueState {
    fn default() -> Self {
        Self {
            rel_acc_white: [0; RELKP_DIM],
            rel_acc_black: [0; RELKP_DIM],
        }
    }
}

impl NnueState {
    /// Create a new NNUE state by computing accumulators from scratch.
    pub fn from_position(gs: &GameState) -> Self {
        let weights = match NNUE_WEIGHTS.as_ref() {
            Some(w) => w,
            None => return Self::default(),
        };

        let (white_features, black_features) = build_relkp_active_lists(gs);

        let mut state = Self::default();

        // Initialize with bias
        // Fix: Use correct bias for each dimension! Previously used [0] for all.
        state.rel_acc_white.copy_from_slice(&weights.rel_bias);
        state.rel_acc_black.copy_from_slice(&weights.rel_bias);

        // Accumulate white perspective features
        for &feat_id in &white_features {
            let offset = (feat_id as usize) * RELKP_DIM;
            for (i, v) in state.rel_acc_white.iter_mut().enumerate() {
                *v = v.saturating_add(weights.rel_embed[offset + i]);
            }
        }

        // Accumulate black perspective features
        for &feat_id in &black_features {
            let offset = (feat_id as usize) * RELKP_DIM;
            for (i, v) in state.rel_acc_black.iter_mut().enumerate() {
                *v = v.saturating_add(weights.rel_embed[offset + i]);
            }
        }

        state
    }

    /// Add a feature to the accumulator (for incremental updates).
    #[inline]
    pub fn add_feature(
        &mut self,
        weights: &super::weights::NnueWeights,
        feat_id: u32,
        is_white: bool,
    ) {
        let offset = (feat_id as usize) * RELKP_DIM;
        let acc = if is_white {
            &mut self.rel_acc_white
        } else {
            &mut self.rel_acc_black
        };

        for (i, v) in acc.iter_mut().enumerate() {
            *v = v.saturating_add(weights.rel_embed[offset + i]);
        }
    }

    /// Remove a feature from the accumulator (for incremental updates).
    #[inline]
    pub fn remove_feature(
        &mut self,
        weights: &super::weights::NnueWeights,
        feat_id: u32,
        is_white: bool,
    ) {
        let offset = (feat_id as usize) * RELKP_DIM;
        let acc = if is_white {
            &mut self.rel_acc_white
        } else {
            &mut self.rel_acc_black
        };

        for (i, v) in acc.iter_mut().enumerate() {
            *v = v.saturating_sub(weights.rel_embed[offset + i]);
        }
    }

    /// Incrementally update the accumulator for a move.
    /// MUST be called BEFORE the move is applied to the GameState.
    pub fn update_for_move(&mut self, gs: &GameState, m: crate::moves::Move) {
        let weights = match NNUE_WEIGHTS.as_ref() {
            Some(w) => w,
            None => return,
        };

        // If King moves, rebuild the entire state from scratch.
        // We manually patch a cloned board rather than calling make_move,
        // which avoids heavy GameState bookkeeping and potential side-effects.
        if m.piece.piece_type() == crate::board::PieceType::King {
            let mut tmp = gs.clone();

            // 1. Remove king from source
            tmp.board.remove_piece(&m.from.x, &m.from.y);

            // 2. Handle capture at destination
            tmp.board.remove_piece(&m.to.x, &m.to.y);

            // 3. Place king at destination
            tmp.board.set_piece(m.to.x, m.to.y, m.piece);

            // 4. Update king position
            if m.piece.color() == crate::board::PlayerColor::White {
                tmp.white_king_pos = Some(m.to);
            } else {
                tmp.black_king_pos = Some(m.to);
            }

            // 5. Handle castling rook
            if (m.to.x - m.from.x).abs() > 1 {
                if let Some(rook_from) = m.rook_coord {
                    if let Some(rook) = tmp.board.remove_piece(&rook_from.x, &rook_from.y) {
                        let rook_to_x = m.from.x + if m.to.x > m.from.x { 1 } else { -1 };
                        tmp.board.set_piece(rook_to_x, m.from.y, rook);
                    }
                }
            }

            // 6. Update side to move (from_position uses gs.turn for perspective)
            tmp.turn = tmp.turn.opponent();

            *self = NnueState::from_position(&tmp);
            return;
        }

        // Standard incremental update (non-King move)
        let us = m.piece.color();
        // Friendly King is at...
        let white_king = if let Some(k) = gs.white_king_pos {
            k
        } else {
            return;
        };
        let black_king = if let Some(k) = gs.black_king_pos {
            k
        } else {
            return;
        };

        let mut update = |piece: crate::board::Piece, sq: crate::board::Coordinate, add: bool| {
            // White View
            if let Some(idx) = super::features::relkp_feature_id(
                crate::board::PlayerColor::White,
                piece,
                sq,
                white_king,
                gs,
            ) {
                if add {
                    self.add_feature(weights, idx, true);
                } else {
                    self.remove_feature(weights, idx, true);
                }
            }
            // Black View
            if let Some(idx) = super::features::relkp_feature_id(
                crate::board::PlayerColor::Black,
                piece,
                sq,
                black_king,
                gs,
            ) {
                if add {
                    self.add_feature(weights, idx, false);
                } else {
                    self.remove_feature(weights, idx, false);
                }
            }
        };

        // 1. Remove from source
        update(m.piece, m.from, false);

        // 2. Add to dest (maybe promoted)
        let new_piece = if let Some(pt) = m.promotion {
            crate::board::Piece::new(pt, us)
        } else {
            m.piece
        };
        update(new_piece, m.to, true);

        // 3. Handle Capture
        if let Some(captured) = gs.board.get_piece(m.to.x, m.to.y) {
            if captured.color() != us {
                update(captured, m.to, false);
            }
        } else if let Some(eps) = gs.en_passant
            && m.piece.piece_type() == crate::board::PieceType::Pawn
            && m.to == eps.square
        {
            // EP Capture
            let cap_sq = eps.pawn_square;
            if let Some(captured) = gs.board.get_piece(cap_sq.x, cap_sq.y) {
                update(captured, cap_sq, false);
            }
        }

        // 4. Castling (Rook update) is handled in handle_king_move because Castling IS a King move.
        // So we don't need to handle it here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Coordinate, Piece, PieceType, PlayerColor};

    #[test]
    fn test_accumulator_ops() {
        let weights = match NNUE_WEIGHTS.as_ref() {
            Some(w) => w,
            None => return,
        };

        let mut state = NnueState::default();
        state.add_feature(weights, 0, true);
        assert_ne!(state.rel_acc_white[0], 0);

        state.remove_feature(weights, 0, true);
        // Assuming 0 embed for feat 0 in mock or it cancels out
        assert_eq!(state.rel_acc_white[0], 0);
    }

    #[test]
    fn test_from_position_consistency() {
        let mut gs = GameState::new();
        gs.white_king_pos = Some(Coordinate::new(4, 0));
        gs.black_king_pos = Some(Coordinate::new(4, 7));
        gs.board
            .set_piece(4, 1, Piece::new(PieceType::Pawn, PlayerColor::White));

        let state = NnueState::from_position(&gs);
        // Bias + feature
        assert!(state.rel_acc_white.iter().any(|&v| v != 0));
    }
}
