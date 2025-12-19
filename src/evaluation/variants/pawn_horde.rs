// Pawn Horde Variant Evaluation
//
// In Pawn Horde, White (Pawns) must advance and promote. Black (Pieces) must capture all pawns.
// This evaluation optimizes for:
// 1. White: Phalanx structure, advancement, promotion threats.
// 2. Black: Breaking the wall, back-rank penetration, picking off weak pawns.

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::base::get_piece_value;
use crate::game::GameState;

// ==================== Constants ====================

// Piece Values
const PAWN_VALUE: i32 = 90; // Higher base value for horde pawns

// White (Horde) Bonuses
const PHALANX_BONUS: i32 = 12; // Side-by-side pawns
const SUPPORT_BONUS: i32 = 15; // Protected pawns
const KING_ATTACK_BONUS: i32 = 20; // Pawns near enemy king

// Black (Pieces) Bonuses
const BREAKTHROUGH_BONUS: i32 = 150; // Major piece behind the pawn wall
const ATTACKING_PAWN_BONUS: i32 = 25; // Attacking a pawn

// Rank-based pawn advancement curve (0-indexed, relative to promotion)
// Closer to 0 means closer to promotion
fn get_pawn_advance_bonus(dist_to_promo: i32) -> i32 {
    match dist_to_promo {
        0 => 0,   // Promoted (not a pawn anymore)
        1 => 250, // Rank 7 - Huge threat
        2 => 100, // Rank 6 - Major threat
        3 => 40,  // Rank 5
        4 => 20,  // Rank 4
        5 => 10,  // Rank 3
        _ => 5,   // Back ranks
    }
}

pub fn evaluate(game: &GameState) -> i32 {
    let mut score = 0;

    // 1. Gather Piece Lists
    let mut white_pawns: Vec<Coordinate> = Vec::with_capacity(36);
    let mut black_pieces: Vec<(Coordinate, PieceType)> = Vec::with_capacity(16);
    let mut black_king_pos = Coordinate::new(5, 8); // Default fallback

    // Map for quick lookup of pawn locations
    // Using a simple vector check is fast enough for ~36 items

    for ((x, y), piece) in game.board.iter() {
        if piece.color() == PlayerColor::Neutral {
            continue;
        }

        let coord = Coordinate::new(*x, *y);
        match piece.color() {
            PlayerColor::White => {
                if piece.piece_type() == PieceType::Pawn {
                    white_pawns.push(coord);
                    score += PAWN_VALUE; // Material count
                } else {
                    // Promoted piece! Huge value.
                    score += get_piece_value(piece.piece_type());
                }
            }
            PlayerColor::Black => {
                if piece.piece_type().is_royal() {
                    black_king_pos = coord;
                }
                black_pieces.push((coord, piece.piece_type()));
                score -= get_piece_value(piece.piece_type());
            }
            _ => {}
        }
    }

    // Win conditions: No pawns = Black wins
    if white_pawns.is_empty() && game.white_piece_count == 0 {
        // Check piece count too in case of promotions
        return if game.turn == PlayerColor::Black {
            30000
        } else {
            -30000
        };
    }

    // 2. White Logic (Horde)
    // Find the "front line" (minimum Y of pawns) to detect breakthroughs
    let mut min_pawn_y = 1000;

    let promo_rank = game.white_promo_rank;

    for pawn in &white_pawns {
        if pawn.y < min_pawn_y {
            min_pawn_y = pawn.y;
        }

        // Advancement
        let dist = (promo_rank - pawn.y).max(0) as i32;
        score += get_pawn_advance_bonus(dist);

        // Phalanx (Horizontal neighbors) - creates a wall
        // We scan the list - O(N^2) but N is small (36) => ~1000 ops, totally fine
        let mut neighbors = 0;
        let mut supported = false;

        for other in &white_pawns {
            if other == pawn {
                continue;
            }

            // Phalanx: Same Y, adjacent X
            if other.y == pawn.y && (other.x - pawn.x).abs() == 1 {
                neighbors += 1;
            }

            // Support: Behind by 1 rank, adjacent X
            // Assuming White moves UP (increasing Y) towards promo_rank > start_rank
            // If promo is 8, support is at y-1.
            let support_y = pawn.y - 1;
            if other.y == support_y && (other.x - pawn.x).abs() == 1 {
                supported = true;
            }
        }

        if neighbors > 0 {
            score += PHALANX_BONUS + (neighbors * 5);
        }
        if supported {
            score += SUPPORT_BONUS;
        }

        // King Attack Tropism
        let dist_to_king = (pawn.x - black_king_pos.x).abs() + (pawn.y - black_king_pos.y).abs();
        if dist_to_king <= 3 {
            score += KING_ATTACK_BONUS * (4 - dist_to_king) as i32;
        }
    }

    // 3. Black Logic (Pieces)
    for (pos, ptype) in &black_pieces {
        // Breakthrough: Are we behind the pawn wall?
        if pos.y < min_pawn_y {
            if *ptype == PieceType::Rook || *ptype == PieceType::Queen {
                score -= BREAKTHROUGH_BONUS; // Score is absolute, so subtract for Black advantage
            }
        }

        // Attacks on Pawns
        // Simple heuristic: distance to nearest pawn
        let mut min_dist_to_pawn = 100;
        for pawn in &white_pawns {
            let d = (pos.x - pawn.x).abs().max((pos.y - pawn.y).abs());
            if d < min_dist_to_pawn {
                min_dist_to_pawn = d;
            }

            // Direct attack checks would be better but expensive without movegen.
            // Distance is a good proxy for "activity against horde".
        }

        if min_dist_to_pawn <= 2 {
            score -= ATTACKING_PAWN_BONUS;
        } else if min_dist_to_pawn > 5 {
            // Piece inactive/far from horde penalty
            score += 10;
        }
    }

    // King Safety (Black)
    // King should be far from the horde front line
    let king_safety_dist = (black_king_pos.y - min_pawn_y).abs(); // Vertical distance to pawn front
    if king_safety_dist < 3 {
        // King is dangerously close to the front
        score += 50; // Penalty for Black (positive score)
    }

    // Return perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}
