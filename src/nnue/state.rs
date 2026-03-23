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

        if m.piece.piece_type() == crate::board::PieceType::King {
            let us = m.piece.color();
            let them = us.opponent();

            let (white_king, black_king) = if us == crate::board::PlayerColor::White {
                (
                    m.to,
                    gs.black_royals
                        .first()
                        .copied()
                        .unwrap_or(crate::board::Coordinate::new(0, 0)),
                )
            } else {
                (
                    gs.white_royals
                        .first()
                        .copied()
                        .unwrap_or(crate::board::Coordinate::new(0, 0)),
                    m.to,
                )
            };

            // Reset the friendly accumulator to bias
            let friendly_acc = if us == crate::board::PlayerColor::White {
                &mut self.rel_acc_white
            } else {
                &mut self.rel_acc_black
            };
            friendly_acc.copy_from_slice(&weights.rel_bias);

            // Re-accumulate all features for the friendly perspective
            // We iterate manually to handle the "virtual" king move without modifying GameState
            let white_promo = &gs.game_rules.promotion_ranks.white;
            let black_promo = &gs.game_rules.promotion_ranks.black;

            let my_king_pos = m.to;

            for (px, py, piece) in gs.board.iter_all_pieces() {
                let piece_color = piece.color();
                if piece_color == crate::board::PlayerColor::Neutral {
                    continue;
                }

                // Skip the King at its OLD position (it's moving)
                if piece.piece_type() == crate::board::PieceType::King && piece_color == us {
                    continue;
                }

                // If this is a piece being captured, skip it (it won't exist in new state)
                if px == m.to.x && py == m.to.y {
                    continue;
                }

                // If this is an EP capture, skip the captured pawn
                if let Some(eps) = gs.en_passant
                    && m.piece.piece_type() == crate::board::PieceType::Pawn
                    && m.to == eps.square
                    && px == eps.pawn_square.x
                    && py == eps.pawn_square.y
                {
                    continue;
                }

                let is_friendly = piece_color == us;

                // Adjust coordinates for perspective
                let (dx, dy) = if us == crate::board::PlayerColor::White {
                    (px - my_king_pos.x, py - my_king_pos.y)
                } else {
                    (-(px - my_king_pos.x), -(py - my_king_pos.y))
                };

                let bucket = super::features::relkp_bucket(dx, dy);

                if let Some(code) = super::features::get_piece_code(
                    piece,
                    is_friendly,
                    py,
                    piece_color,
                    white_promo,
                    black_promo,
                ) {
                    let feat_id = code * super::features::NUM_RELKP_BUCKETS + bucket;

                    // Add to friendly accumulator
                    let offset = (feat_id as usize) * super::state::RELKP_DIM;
                    for (i, v) in friendly_acc.iter_mut().enumerate() {
                        *v = v.saturating_add(weights.rel_embed[offset + i]);
                    }
                }
            }


            if (m.to.x - m.from.x).abs() > 1
                && let Some(rook_from) = m.rook_coord
                && let Some(rook) = gs.board.get_piece(rook_from.x, rook_from.y)
            {
                let _is_friendly = true; // Rook is same color as King

                // Recalculate what the loop added for the rook
                let (dx_old, dy_old) = if us == crate::board::PlayerColor::White {
                    (rook_from.x - my_king_pos.x, rook_from.y - my_king_pos.y)
                } else {
                    (
                        -(rook_from.x - my_king_pos.x),
                        -(rook_from.y - my_king_pos.y),
                    )
                };
                let bucket_old = super::features::relkp_bucket(dx_old, dy_old);
                if let Some(code) = super::features::get_piece_code(
                    rook,
                    true,
                    rook_from.y,
                    us,
                    white_promo,
                    black_promo,
                ) {
                    let feat_id = code * super::features::NUM_RELKP_BUCKETS + bucket_old;
                    let offset = (feat_id as usize) * super::state::RELKP_DIM;
                    for (i, v) in friendly_acc.iter_mut().enumerate() {
                        *v = v.saturating_sub(weights.rel_embed[offset + i]);
                    }
                }

                // Add the rook feature at `rook_to`
                let rook_to_x = m.from.x + if m.to.x > m.from.x { 1 } else { -1 };
                let rook_to_y = m.from.y;

                let (dx_new, dy_new) = if us == crate::board::PlayerColor::White {
                    (rook_to_x - my_king_pos.x, rook_to_y - my_king_pos.y)
                } else {
                    (-(rook_to_x - my_king_pos.x), -(rook_to_y - my_king_pos.y))
                };
                let bucket_new = super::features::relkp_bucket(dx_new, dy_new);
                if let Some(code) = super::features::get_piece_code(
                    rook,
                    true,
                    rook_to_y,
                    us,
                    white_promo,
                    black_promo,
                ) {
                    let feat_id = code * super::features::NUM_RELKP_BUCKETS + bucket_new;
                    let offset = (feat_id as usize) * super::state::RELKP_DIM;
                    for (i, v) in friendly_acc.iter_mut().enumerate() {
                        *v = v.saturating_add(weights.rel_embed[offset + i]);
                    }
                }
            }
            
            // Remove old king pos
            if let Some(idx) = super::features::relkp_feature_id(
                them,
                m.piece,
                m.from,
                if them == crate::board::PlayerColor::White {
                    white_king
                } else {
                    black_king
                }, // Enemy king position (static)
                gs,
            ) {
                // Remove from enemy acc
                self.remove_feature(weights, idx, them == crate::board::PlayerColor::White);
            }

            // Add new king pos
            if let Some(idx) = super::features::relkp_feature_id(
                them,
                m.piece,
                m.to,
                if them == crate::board::PlayerColor::White {
                    white_king
                } else {
                    black_king
                },
                gs,
            ) {
                // Add to enemy acc
                self.add_feature(weights, idx, them == crate::board::PlayerColor::White);
            }

            // Note: If we captured something, we must also remove it from the enemy accumulator
            // because it's no longer on the board!
            //
            // Case A: Standard Capture
            // Case A: Standard Capture
            if let Some(captured) = gs.board.get_piece(m.to.x, m.to.y)
                && let Some(idx) = super::features::relkp_feature_id(
                    them,
                    captured,
                    m.to,
                    if them == crate::board::PlayerColor::White {
                        white_king
                    } else {
                        black_king
                    },
                    gs,
                )
            {
                self.remove_feature(weights, idx, them == crate::board::PlayerColor::White);
            }

            // Case B: En Passant Capture (King cannot do EP in chess, but for safety/completeness)
            if let Some(_eps) = gs.en_passant
                && m.piece.piece_type() == crate::board::PieceType::Pawn
            // King can't be pawn
            {
                // Unreachable for King move
            }

            // Case C: Castling Rook Move
            // From ENEMY perspective, the Rook also moved.
            // We must update the Rook's position in the enemy accumulator too.
            if (m.to.x - m.from.x).abs() > 1
                && let Some(rook_from) = m.rook_coord
                && let Some(rook) = gs.board.get_piece(rook_from.x, rook_from.y)
            {
                // Remove Rook from old pos
                if let Some(idx) = super::features::relkp_feature_id(
                    them,
                    rook,
                    rook_from,
                    if them == crate::board::PlayerColor::White {
                        white_king
                    } else {
                        black_king
                    },
                    gs,
                ) {
                    self.remove_feature(weights, idx, them == crate::board::PlayerColor::White);
                }

                // Add Rook to new pos
                let rook_to_x = m.from.x + if m.to.x > m.from.x { 1 } else { -1 };
                let rook_to = crate::board::Coordinate::new(rook_to_x, m.from.y);

                if let Some(idx) = super::features::relkp_feature_id(
                    them,
                    rook,
                    rook_to,
                    if them == crate::board::PlayerColor::White {
                        white_king
                    } else {
                        black_king
                    },
                    gs,
                ) {
                    self.add_feature(weights, idx, them == crate::board::PlayerColor::White);
                }
            }

            return;
        }

        // Standard incremental update (non-King move)
        let us = m.piece.color();
        // Friendly King is at...
        let white_king = if let Some(k) = gs.white_royals.first().copied() {
            k
        } else {
            return;
        };
        let black_king = if let Some(k) = gs.black_royals.first().copied() {
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
    use crate::game::GameState;

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
        let gs = {
            let mut gs = GameState::new();
            gs.setup_position_from_icn("w (8;q|1;q) K4,0|k4,7|P4,1");
            gs
        };

        let state = NnueState::from_position(&gs);
        // Bias + feature
        assert!(state.rel_acc_white.iter().any(|&v| v != 0));
    }
}
