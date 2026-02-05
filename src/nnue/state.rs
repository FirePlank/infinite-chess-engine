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

        // If King moves, we must handle perspective updates carefully (recompute our side)
        if m.piece.piece_type() == crate::board::PieceType::King {
            self.handle_king_move(gs, m, weights);
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
        } else if let Some(eps) = gs.en_passant {
            if m.piece.piece_type() == crate::board::PieceType::Pawn && m.to == eps.square {
                // EP Capture
                let cap_sq = eps.pawn_square;
                if let Some(captured) = gs.board.get_piece(cap_sq.x, cap_sq.y) {
                    update(captured, cap_sq, false);
                }
            }
        }

        // 4. Castling (Rook update) is handled in handle_king_move because Castling IS a King move.
        // So we don't need to handle it here.
    }

    /// Handle complex updates when the King moves (requires recomputing relative features for that side).
    fn handle_king_move(
        &mut self,
        gs: &GameState,
        m: crate::moves::Move,
        weights: &super::weights::NnueWeights,
    ) {
        let us = m.piece.color();

        // 1. Update Opponent's perspective (Incremental)
        // For opponent, this is just an enemy piece (our King) moving.
        let them = if us == crate::board::PlayerColor::White {
            crate::board::PlayerColor::Black
        } else {
            crate::board::PlayerColor::White
        };
        let them_king = if us == crate::board::PlayerColor::White {
            gs.black_king_pos
        } else {
            gs.white_king_pos
        };

        if let Some(t_king) = them_king {
            // Helper for single perspective
            let mut update_them = |piece: crate::board::Piece,
                                   sq: crate::board::Coordinate,
                                   add: bool| {
                if let Some(idx) = super::features::relkp_feature_id(them, piece, sq, t_king, gs) {
                    if them == crate::board::PlayerColor::White {
                        if add {
                            self.add_feature(weights, idx, true);
                        } else {
                            self.remove_feature(weights, idx, true);
                        }
                    } else {
                        if add {
                            self.add_feature(weights, idx, false);
                        } else {
                            self.remove_feature(weights, idx, false);
                        }
                    }
                }
            };

            // Remove King from From
            update_them(m.piece, m.from, false);
            // Add King to To
            update_them(m.piece, m.to, true);

            // Capture handling for THEM perspective
            if let Some(captured) = gs.board.get_piece(m.to.x, m.to.y) {
                if captured.color() != us {
                    update_them(captured, m.to, false);
                }
            }

            // Castling Rook?
            if (m.from.x - m.to.x).abs() > 1 {
                if let Some(rook_from) = m.rook_coord {
                    let rook_to_x = if m.to.x > m.from.x {
                        m.to.x - 1
                    } else {
                        m.to.x + 1
                    };
                    let rook_to = crate::board::Coordinate::new(rook_to_x, m.to.y);
                    if let Some(rook) = gs.board.get_piece(rook_from.x, rook_from.y) {
                        update_them(rook, rook_from, false);
                        update_them(rook, rook_to, true);
                    }
                }
            }
        }

        // 2. Recompute Our Perspective (Scratch)
        // Need to clear and re-accumulate because ALL relative coordinates changed.
        if us == crate::board::PlayerColor::White {
            self.rel_acc_white.copy_from_slice(&weights.rel_bias);
        } else {
            self.rel_acc_black.copy_from_slice(&weights.rel_bias);
        }

        let new_king_pos = m.to;

        // Re-accumulate ALL pieces on the board based on the new state
        for (px, py, piece) in gs.board.iter() {
            let sq = crate::board::Coordinate::new(px, py);

            // Skip our own King from the feature list (limit to perspective side)
            // Feature set is relative TO the king, so the king itself is not a feature.
            if piece.color() == us && piece.piece_type() == crate::board::PieceType::King {
                continue;
            }

            if let Some(idx) = super::features::relkp_feature_id(us, piece, sq, new_king_pos, gs) {
                if us == crate::board::PlayerColor::White {
                    self.add_feature(weights, idx, true);
                } else {
                    self.add_feature(weights, idx, false);
                }
            }
        }
    }
}
