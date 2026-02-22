// Chess Variant Evaluation (Standard 8x8 Chess)
// Uses an improved version of the PeSTO-based evaluation.

use crate::board::{PieceType, PlayerColor};
use crate::game::GameState;
use arrayvec::ArrayVec;

// Constants

const MG_VALUES: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUES: [i32; 6] = [94, 281, 297, 512, 936, 0];

// Mobility bonuses
const MG_MOBILITY: [i32; 6] = [0, 4, 3, 2, 1, 0];
const EG_MOBILITY: [i32; 6] = [0, 4, 3, 4, 2, 0];

// King Safety Weights (simple)
const MG_KING_ATTACK_WEIGHT: [i32; 6] = [0, 10, 10, 20, 40, 0];
const KING_SAFETY_COEFF: i32 = 2; // Scaler for safety penalty

// Pawn Structure
const MG_ISOLATED_PENALTY: i32 = 10;
const EG_ISOLATED_PENALTY: i32 = 15;
const MG_DOUBLED_PENALTY: i32 = 10;
const EG_DOUBLED_PENALTY: i32 = 10;
const PASSED_PAWN_BONUS: [i32; 8] = [0, 5, 10, 20, 40, 70, 120, 200];

const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

// Piece-Square Tables (Top-Down, a8=0)

#[rustfmt::skip]
const MG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
];

#[rustfmt::skip]
const EG_PAWN_PST: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT_PST: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
];

#[rustfmt::skip]
const EG_KNIGHT_PST: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];

#[rustfmt::skip]
const MG_BISHOP_PST: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];

#[rustfmt::skip]
const EG_BISHOP_PST: [i32; 64] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
];

#[rustfmt::skip]
const MG_ROOK_PST: [i32; 64] = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
];

#[rustfmt::skip]
const EG_ROOK_PST: [i32; 64] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
];

#[rustfmt::skip]
const MG_QUEEN_PST: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];

#[rustfmt::skip]
const EG_QUEEN_PST: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];

#[rustfmt::skip]
const MG_KING_PST: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];

#[rustfmt::skip]
const EG_KING_PST: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
];

const MG_PST: [[i32; 64]; 6] = [
    MG_PAWN_PST,
    MG_KNIGHT_PST,
    MG_BISHOP_PST,
    MG_ROOK_PST,
    MG_QUEEN_PST,
    MG_KING_PST,
];

const EG_PST: [[i32; 64]; 6] = [
    EG_PAWN_PST,
    EG_KNIGHT_PST,
    EG_BISHOP_PST,
    EG_ROOK_PST,
    EG_QUEEN_PST,
    EG_KING_PST,
];

// ==================== Helper Functions ====================

#[inline]
fn coord_to_pst_index(x: i64, y: i64) -> usize {
    // top-down: Rank 8 is row 0, Rank 1 is row 7
    // DEFENSIVE: Clamp to 1-8 range to avoid underflow/overflow panics
    let cx = x.clamp(1, 8);
    let cy = y.clamp(1, 8);
    let row = (8 - cy) as usize;
    let col = (cx - 1) as usize;
    row * 8 + col
}

#[inline]
fn get_piece_idx(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
        _ => 0,
    }
}

// Main Evaluation

#[allow(clippy::needless_range_loop)]
pub fn evaluate(game: &GameState) -> i32 {
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut game_phase = 0;

    let mut w_pawn_files = 0u8;
    let mut b_pawn_files = 0u8;
    let mut pawns: ArrayVec<(i64, i64, bool), 16> = ArrayVec::new();

    // Tracker for king safety
    let mut w_king_attackers = 0;
    let mut w_king_attack_weight = 0;
    let mut b_king_attackers = 0;
    let mut b_king_attack_weight = 0;

    use crate::board::Coordinate;

    let white_king = game.white_king_pos.unwrap_or(Coordinate { x: 5, y: 1 });
    let black_king = game.black_king_pos.unwrap_or(Coordinate { x: 5, y: 8 });

    // SINGLE PASS: Iterating over board pieces once
    for (x, y, piece) in game.board.iter_all_pieces() {
        let pt = piece.piece_type();
        let pc_idx = get_piece_idx(pt);
        let is_white = piece.color() == PlayerColor::White;
        let color_idx = if is_white { 0 } else { 1 };

        // 1. PST & Material
        let mut sq = coord_to_pst_index(x, y);
        if !is_white {
            sq ^= 56;
        }
        mg[color_idx] += MG_VALUES[pc_idx] + MG_PST[pc_idx][sq];
        eg[color_idx] += EG_VALUES[pc_idx] + EG_PST[pc_idx][sq];

        // 2. Phase
        game_phase += PHASE_INC[pc_idx];

        if pt == PieceType::Pawn {
            if is_white {
                w_pawn_files |= 1 << ((x - 1).clamp(0, 7));
            } else {
                b_pawn_files |= 1 << ((x - 1).clamp(0, 7));
            }
            pawns.push((x, y, is_white));
        } else if (1..=4).contains(&pc_idx) {
            // 3. Mobility (including capture squares)
            let mobility = count_mobility(&game.board, x, y, piece);
            mg[color_idx] += mobility * MG_MOBILITY[pc_idx];
            eg[color_idx] += mobility * EG_MOBILITY[pc_idx];

            // 4. King Safety contribution
            let target_king = if is_white { &black_king } else { &white_king };
            if (x - target_king.x).abs() <= 3 && (y - target_king.y).abs() <= 3 {
                if is_white {
                    b_king_attackers += 1;
                    b_king_attack_weight += MG_KING_ATTACK_WEIGHT[pc_idx];
                } else {
                    w_king_attackers += 1;
                    w_king_attack_weight += MG_KING_ATTACK_WEIGHT[pc_idx];
                }
            }
        }
    }

    // Pawn Structure (Second Pass on pawns only)
    for i in 0..pawns.len() {
        let (x, y, is_white) = pawns[i];
        let color_idx = if is_white { 0 } else { 1 };
        let f = (x - 1).clamp(0, 7) as usize;
        let files = if is_white { w_pawn_files } else { b_pawn_files };

        // 1. Isolated
        let has_neighbor =
            (f > 0 && (files & (1 << (f - 1))) != 0) || (f < 7 && (files & (1 << (f + 1))) != 0);
        if !has_neighbor {
            mg[color_idx] -= MG_ISOLATED_PENALTY;
            eg[color_idx] -= EG_ISOLATED_PENALTY;
        }

        // 2. Doubled
        let mut is_doubled = false;
        for j in 0..pawns.len() {
            if i != j {
                let (nx, _, nw) = pawns[j];
                if nw == is_white && nx == x {
                    is_doubled = true;
                    break;
                }
            }
        }
        if is_doubled {
            mg[color_idx] -= MG_DOUBLED_PENALTY;
            eg[color_idx] -= EG_DOUBLED_PENALTY;
        }

        // 3. Passed (No enemy pawns on same/adj files AHEAD)
        let mut is_passed = true;
        for j in 0..pawns.len() {
            let (nx, ny, nw) = pawns[j];
            if nw != is_white
                && (nx - x).abs() <= 1
                && ((is_white && ny > y) || (!is_white && ny < y))
            {
                is_passed = false;
                break;
            }
        }

        if is_passed {
            let rank = if is_white { y } else { 9 - y };
            let bonus = PASSED_PAWN_BONUS[(rank - 1).clamp(0, 7) as usize];
            mg[color_idx] += bonus / 2;
            eg[color_idx] += bonus;
        }
    }

    // Apply King Safety Penalty (Non-linear)
    if w_king_attackers >= 2 {
        let penalty = (w_king_attackers * w_king_attackers * w_king_attack_weight) / 40;
        mg[0] -= penalty * KING_SAFETY_COEFF;
    }
    if b_king_attackers >= 2 {
        let penalty = (b_king_attackers * b_king_attackers * b_king_attack_weight) / 40;
        mg[1] -= penalty * KING_SAFETY_COEFF;
    }

    // Perspective relative to current player
    let side = if game.turn == PlayerColor::White {
        0
    } else {
        1
    };
    let other = side ^ 1;

    let mg_score = mg[side] - mg[other];
    let eg_score = eg[side] - eg[other];

    let mg_phase = game_phase.min(MAX_PHASE);
    let eg_phase = MAX_PHASE - mg_phase;

    (mg_score * mg_phase + eg_score * eg_phase) / MAX_PHASE
}

fn count_mobility(board: &crate::board::Board, x: i64, y: i64, piece: crate::board::Piece) -> i32 {
    let mut count = 0;
    let pt = piece.piece_type();
    let our_color = piece.color();
    let dirs = match pt {
        PieceType::Knight => vec![
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ],
        PieceType::Bishop => vec![(1, 1), (1, -1), (-1, 1), (-1, -1)],
        PieceType::Rook => vec![(1, 0), (-1, 0), (0, 1), (0, -1)],
        PieceType::Queen => vec![
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ],
        _ => return 0,
    };

    let sliding = pt == PieceType::Bishop || pt == PieceType::Rook || pt == PieceType::Queen;

    for (dx, dy) in dirs {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while (1..=8).contains(&nx) && (1..=8).contains(&ny) {
            if let Some(p) = board.get_piece(nx, ny) {
                if p.color() != our_color && p.color() != PlayerColor::Neutral {
                    count += 1; // Count capture square
                }
                break;
            }
            count += 1;
            if !sliding {
                break;
            }
            nx += dx;
            ny += dy;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_chess_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.variant = Some(crate::Variant::Chess);
        game
    }

    fn setup_standard_chess_opening(game: &mut GameState) {
        for file in 1..=8 {
            game.board
                .set_piece(file, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
            game.board
                .set_piece(file, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        }
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.board
            .set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));

        game.board
            .set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.board
            .set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.board
            .set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        game.board
            .set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        game.board
            .set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        game.board
            .set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

        game.recompute_piece_counts();
    }

    #[test]
    fn test_evaluate_starting_position() {
        let mut game = create_chess_game();
        setup_standard_chess_opening(&mut game);
        game.turn = PlayerColor::White;
        let score = evaluate(&game);
        assert_eq!(score, 0, "Starting position should be perfectly equal");
    }

    #[test]
    fn test_evaluate_material_advantage() {
        let mut game = create_chess_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        let score = evaluate(&game);
        assert!(
            score > 800,
            "White should have significant advantage with extra queen"
        );
    }
}
