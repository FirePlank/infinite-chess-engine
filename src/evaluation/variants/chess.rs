// Chess Variant Evaluation (Standard 8x8 Chess)

use crate::board::{PieceType, PlayerColor};
use crate::game::GameState;
use arrayvec::ArrayVec;

// ==================== Material Values ====================

const MG_VALUES: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUES: [i32; 6] = [94, 281, 297, 512, 936, 0];

// ==================== Non-linear Mobility Tables ====================

// Knight: 0-8 squares
#[rustfmt::skip]
const MG_KNIGHT_MOB: [i32; 9] = [-62, -36, -12,  0,  8, 14, 18, 20, 22];
#[rustfmt::skip]
const EG_KNIGHT_MOB: [i32; 9] = [-81, -46, -26, -8,  4, 10, 14, 16, 18];

// Bishop: 0-13 squares
#[rustfmt::skip]
const MG_BISHOP_MOB: [i32; 14] = [-48, -20,  6, 14, 20, 26, 30, 32, 32, 34, 36, 36, 38, 40];
#[rustfmt::skip]
const EG_BISHOP_MOB: [i32; 14] = [-59, -23, -3,  8, 16, 22, 28, 30, 34, 36, 38, 40, 42, 44];

// Rook: 0-14 squares
#[rustfmt::skip]
const MG_ROOK_MOB: [i32; 15] = [-60, -20, 0, 2, 4,  8, 14, 18, 22, 22, 24, 26, 28, 28, 30];
#[rustfmt::skip]
const EG_ROOK_MOB: [i32; 15] = [-78, -17, 16, 28, 48, 62, 66, 74, 78, 80, 84, 86, 88, 88, 90];

// Queen: 0-27 squares
#[rustfmt::skip]
const MG_QUEEN_MOB: [i32; 28] = [
    -30, -12, -8, -8, 10, 12, 12, 18, 18, 26, 30, 30,
     30, 30, 30, 30, 34, 34, 36, 36, 40, 42, 42, 42,
     44, 46, 46, 48,
];
#[rustfmt::skip]
const EG_QUEEN_MOB: [i32; 28] = [
    -48, -30,  -7, 14, 30, 40, 44, 50, 52, 58, 60, 64,
     72, 76, 80, 82, 84, 86, 90, 92, 94, 96, 96, 100,
    102, 104, 106, 112,
];

// ==================== Piece-Square Tables (Top-Down, a8=0) ====================

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

// ==================== Pawn Structure ====================

const MG_ISOLATED_PENALTY: i32 = 8;
const EG_ISOLATED_PENALTY: i32 = 16;
const MG_DOUBLED_PENALTY: i32 = 8;
const EG_DOUBLED_PENALTY: i32 = 16;
const MG_BACKWARD_PENALTY: i32 = 9;
const EG_BACKWARD_PENALTY: i32 = 20;

// Connected pawn bonus by rank (rank 1..=8, index 0 unused)
#[rustfmt::skip]
const CONNECTED_BONUS: [i32; 8] = [0, 0, 7, 8, 12, 29, 48, 0];

// Passed pawn bonus by relative rank [0..8], rank 0/1 unused
#[rustfmt::skip]
const MG_PASSED_BONUS: [i32; 8] = [0, 0,  5, 10, 20,  40,  70, 120];
#[rustfmt::skip]
const EG_PASSED_BONUS: [i32; 8] = [0, 0, 10, 20, 40,  80, 140, 220];

// ==================== Piece Bonuses ====================

const MG_BISHOP_PAIR: i32 = 30;
const EG_BISHOP_PAIR: i32 = 55;

const MG_ROOK_OPEN_FILE: i32 = 48;
const EG_ROOK_OPEN_FILE: i32 = 29;
const MG_ROOK_SEMI_OPEN_FILE: i32 = 19;
const EG_ROOK_SEMI_OPEN_FILE: i32 = 7;

// Outpost bonus: knight/bishop on a square protected by own pawn,
// ranks 4-6 (relative), not attackable by enemy pawns.
const MG_OUTPOST_KNIGHT: i32 = 54;
const EG_OUTPOST_KNIGHT: i32 = 34;
const MG_OUTPOST_BISHOP: i32 = 28;
const EG_OUTPOST_BISHOP: i32 = 20;

// Minor piece behind a pawn
const MG_MINOR_BEHIND_PAWN: i32 = 18;

// ==================== King Safety ====================

// King attack weights per piece type [pawn, knight, bishop, rook, queen, king]
const KING_ATTACK_WEIGHT: [i32; 6] = [0, 20, 20, 40, 80, 0];

// Pawn shelter bonus by distance (1 = adjacent rank, 2 = two ranks away, 3 = three)
// and by file relationship (0 = king file, 1 = adjacent file, 2 = outer file)
const SHELTER_BONUS: [[i32; 3]; 4] = [
    [0, 0, 0],   // distance 0 (unused)
    [20, 14, 8], // distance 1
    [10,  6, 2], // distance 2
    [ 4,  2, 0], // distance 3
];
// Penalty if no pawn covering the king's file/adjacent at all
const SHELTER_MISSING_PENALTY: [i32; 3] = [22, 14, 6];

// ==================== Phase ====================

const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;

// ==================== Helper Functions ====================

#[inline]
fn coord_to_pst_index(x: i64, y: i64) -> usize {
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

/// Chebyshev distance between two squares.
#[inline]
fn cheb(ax: i64, ay: i64, bx: i64, by: i64) -> i64 {
    (ax - bx).abs().max((ay - by).abs())
}

// ==================== Main Evaluation ====================

#[allow(clippy::needless_range_loop)]
pub fn evaluate(game: &GameState) -> i32 {
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut game_phase = 0i32;

    // Pawn file occupancy bitmasks (bits 0-7 = files a-h)
    let mut w_pawn_files = 0u8;
    let mut b_pawn_files = 0u8;
    // Pawns stored as (x, y, is_white)
    let mut pawns: ArrayVec<(i64, i64, bool), 16> = ArrayVec::new();

    // Bishop tracking for bishop pair
    let mut w_bishop_light = false;
    let mut w_bishop_dark = false;
    let mut b_bishop_light = false;
    let mut b_bishop_dark = false;

    // King safety accumulators
    let mut w_king_danger = 0i32; // danger TO white king (attackers = black)
    let mut b_king_danger = 0i32; // danger TO black king (attackers = white)

    use crate::board::Coordinate;

    let white_king = game
        .white_royals
        .first()
        .copied()
        .unwrap_or(Coordinate { x: 5, y: 1 });
    let black_king = game
        .black_royals
        .first()
        .copied()
        .unwrap_or(Coordinate { x: 5, y: 8 });

    // ---- PRE-PASS: collect pawn file masks (needed by piece evaluation) ----
    for (x, _, piece) in game.board.iter_all_pieces() {
        if piece.piece_type() == PieceType::Pawn {
            let file_bit = 1u8 << ((x - 1).clamp(0, 7));
            if piece.color() == PlayerColor::White {
                w_pawn_files |= file_bit;
            } else {
                b_pawn_files |= file_bit;
            }
        }
    }

    // ---- FIRST PASS: material, PST, phase, mobility, bishop pair, rook files ----
    for (x, y, piece) in game.board.iter_all_pieces() {
        let pt = piece.piece_type();
        let pc_idx = get_piece_idx(pt);
        let is_white = piece.color() == PlayerColor::White;
        let ci = if is_white { 0 } else { 1 };

        // PST index: white uses normal orientation, black flips vertically
        let mut sq = coord_to_pst_index(x, y);
        if !is_white {
            sq ^= 56;
        }

        mg[ci] += MG_VALUES[pc_idx] + MG_PST[pc_idx][sq];
        eg[ci] += EG_VALUES[pc_idx] + EG_PST[pc_idx][sq];
        game_phase += PHASE_INC[pc_idx];

        match pt {
            PieceType::Pawn => {
                pawns.push((x, y, is_white));
            }

            PieceType::Bishop => {
                // Track bishop square color for pair detection
                let light = (x + y) % 2 == 0;
                if is_white {
                    if light { w_bishop_light = true; } else { w_bishop_dark = true; }
                } else if light { b_bishop_light = true; } else { b_bishop_dark = true; }

                let mob = count_sliding_mobility(&game.board, x, y, piece);
                let mob_idx = mob.min(13) as usize;
                mg[ci] += MG_BISHOP_MOB[mob_idx];
                eg[ci] += EG_BISHOP_MOB[mob_idx];

                // King safety: bishop near enemy king
                let ek = if is_white { &black_king } else { &white_king };
                if cheb(x, y, ek.x, ek.y) <= 4 {
                    if is_white { b_king_danger += KING_ATTACK_WEIGHT[2]; }
                    else        { w_king_danger += KING_ATTACK_WEIGHT[2]; }
                }

                // Outpost (ranks 4-6 for white = y 4-6, for black = y 3-5)
                let rel_rank = if is_white { y } else { 9 - y };
                if (4..=6).contains(&rel_rank) {
                    let f = (x - 1).clamp(0, 7) as usize;
                    let own_pawns = if is_white { w_pawn_files } else { b_pawn_files };
                    let enemy_pawns = if is_white { b_pawn_files } else { w_pawn_files };
                    let pawn_protected = (f > 0 && (own_pawns & (1 << (f - 1))) != 0)
                        || (f < 7 && (own_pawns & (1 << (f + 1))) != 0);
                    let enemy_can_attack = (f > 0 && (enemy_pawns & (1 << (f - 1))) != 0)
                        || (f < 7 && (enemy_pawns & (1 << (f + 1))) != 0);
                    if pawn_protected && !enemy_can_attack {
                        mg[ci] += MG_OUTPOST_BISHOP;
                        eg[ci] += EG_OUTPOST_BISHOP;
                    }
                }
            }

            PieceType::Knight => {
                let mob = count_knight_mobility(&game.board, x, y, piece);
                let mob_idx = mob.min(8) as usize;
                mg[ci] += MG_KNIGHT_MOB[mob_idx];
                eg[ci] += EG_KNIGHT_MOB[mob_idx];

                // King safety
                let ek = if is_white { &black_king } else { &white_king };
                if cheb(x, y, ek.x, ek.y) <= 3 {
                    if is_white { b_king_danger += KING_ATTACK_WEIGHT[1]; }
                    else        { w_king_danger += KING_ATTACK_WEIGHT[1]; }
                }

                // Outpost
                let rel_rank = if is_white { y } else { 9 - y };
                if (4..=6).contains(&rel_rank) {
                    let f = (x - 1).clamp(0, 7) as usize;
                    let own_pawns = if is_white { w_pawn_files } else { b_pawn_files };
                    let enemy_pawns = if is_white { b_pawn_files } else { w_pawn_files };
                    let pawn_protected = (f > 0 && (own_pawns & (1 << (f - 1))) != 0)
                        || (f < 7 && (own_pawns & (1 << (f + 1))) != 0);
                    let enemy_can_attack = (f > 0 && (enemy_pawns & (1 << (f - 1))) != 0)
                        || (f < 7 && (enemy_pawns & (1 << (f + 1))) != 0);
                    if pawn_protected && !enemy_can_attack {
                        mg[ci] += MG_OUTPOST_KNIGHT;
                        eg[ci] += EG_OUTPOST_KNIGHT;
                    }
                }

                // Minor behind pawn
                let pawn_ahead_y = if is_white { y + 1 } else { y - 1 };
                if (1..=8).contains(&pawn_ahead_y) {
                    if let Some(p) = game.board.get_piece(x, pawn_ahead_y) {
                        if p.piece_type() == PieceType::Pawn && p.color() == piece.color() {
                            mg[ci] += MG_MINOR_BEHIND_PAWN;
                        }
                    }
                }
            }

            PieceType::Rook => {
                let mob = count_sliding_mobility(&game.board, x, y, piece);
                let mob_idx = mob.min(14) as usize;
                mg[ci] += MG_ROOK_MOB[mob_idx];
                eg[ci] += EG_ROOK_MOB[mob_idx];

                // King safety
                let ek = if is_white { &black_king } else { &white_king };
                if cheb(x, y, ek.x, ek.y) <= 4 {
                    if is_white { b_king_danger += KING_ATTACK_WEIGHT[3]; }
                    else        { w_king_danger += KING_ATTACK_WEIGHT[3]; }
                }

                // Open / semi-open file bonus
                let f = ((x - 1).clamp(0, 7)) as u8;
                let file_bit = 1u8 << f;
                let own_pawn = if is_white { w_pawn_files } else { b_pawn_files };
                let enemy_pawn = if is_white { b_pawn_files } else { w_pawn_files };
                if own_pawn & file_bit == 0 {
                    if enemy_pawn & file_bit == 0 {
                        mg[ci] += MG_ROOK_OPEN_FILE;
                        eg[ci] += EG_ROOK_OPEN_FILE;
                    } else {
                        mg[ci] += MG_ROOK_SEMI_OPEN_FILE;
                        eg[ci] += EG_ROOK_SEMI_OPEN_FILE;
                    }
                }
            }

            PieceType::Queen => {
                let mob = count_sliding_mobility(&game.board, x, y, piece);
                let mob_idx = mob.min(27) as usize;
                mg[ci] += MG_QUEEN_MOB[mob_idx];
                eg[ci] += EG_QUEEN_MOB[mob_idx];

                // King safety
                let ek = if is_white { &black_king } else { &white_king };
                if cheb(x, y, ek.x, ek.y) <= 5 {
                    if is_white { b_king_danger += KING_ATTACK_WEIGHT[4]; }
                    else        { w_king_danger += KING_ATTACK_WEIGHT[4]; }
                }
            }

            _ => {}
        }
    }

    // ---- Bishop pair ----
    if w_bishop_light && w_bishop_dark {
        mg[0] += MG_BISHOP_PAIR;
        eg[0] += EG_BISHOP_PAIR;
    }
    if b_bishop_light && b_bishop_dark {
        mg[1] += MG_BISHOP_PAIR;
        eg[1] += EG_BISHOP_PAIR;
    }

    // ---- SECOND PASS: pawn structure ----
    for i in 0..pawns.len() {
        let (x, y, is_white) = pawns[i];
        let ci = if is_white { 0 } else { 1 };
        let f = (x - 1).clamp(0, 7) as usize;
        let own_files = if is_white { w_pawn_files } else { b_pawn_files };
        let enemy_files = if is_white { b_pawn_files } else { w_pawn_files };

        // -- Isolated --
        let has_neighbor = (f > 0 && (own_files & (1 << (f - 1))) != 0)
            || (f < 7 && (own_files & (1 << (f + 1))) != 0);
        if !has_neighbor {
            mg[ci] -= MG_ISOLATED_PENALTY;
            eg[ci] -= EG_ISOLATED_PENALTY;
        }

        // -- Doubled --
        let is_doubled = pawns.iter().enumerate().any(|(j, &(nx, _, nw))| {
            j != i && nw == is_white && nx == x
        });
        if is_doubled {
            mg[ci] -= MG_DOUBLED_PENALTY;
            eg[ci] -= EG_DOUBLED_PENALTY;
        }

        // -- Connected (phalanx or supported) --
        // Phalanx: friendly pawn on same rank, adjacent file
        let phalanx = pawns.iter().any(|&(nx, ny, nw)| {
            nw == is_white && ny == y && (nx - x).abs() == 1
        });
        // Supported: friendly pawn one rank behind on adjacent file
        let support_y = if is_white { y - 1 } else { y + 1 };
        let supported = pawns.iter().any(|&(nx, ny, nw)| {
            nw == is_white && ny == support_y && (nx - x).abs() <= 1
        });
        if phalanx || supported {
            let rank = if is_white { y } else { 9 - y };
            let rank_idx = (rank as usize).clamp(0, 7);
            let v = CONNECTED_BONUS[rank_idx];
            mg[ci] += v;
            eg[ci] += v * (rank_idx as i32 - 2).max(0) / 4;
        }

        // -- Backward pawn --
        // A pawn is backward if it has no friendly pawns supporting it from behind
        // on adjacent files, and the stop square is contested by an enemy pawn.
        // Only applied when not already isolated (no neighbor), to avoid double-counting.
        if !supported && !phalanx && has_neighbor {
            let no_support_behind = !pawns.iter().any(|&(nx, ny, nw)| {
                nw == is_white
                    && (nx - x).abs() == 1
                    && if is_white { ny < y } else { ny > y }
            });
            let enemy_stop_file = (f > 0 && (enemy_files & (1 << (f - 1))) != 0)
                || (f < 7 && (enemy_files & (1 << (f + 1))) != 0);
            if no_support_behind && enemy_stop_file {
                mg[ci] -= MG_BACKWARD_PENALTY;
                eg[ci] -= EG_BACKWARD_PENALTY;
            }
        }

        // -- Passed pawn --
        let is_passed = !pawns.iter().any(|&(nx, ny, nw)| {
            nw != is_white
                && (nx - x).abs() <= 1
                && if is_white { ny > y } else { ny < y }
        });

        if is_passed {
            let rel_rank = if is_white { y } else { 9 - y };
            let rank_idx = (rel_rank as usize).clamp(0, 7);
            let mg_bonus = MG_PASSED_BONUS[rank_idx];
            let mut eg_bonus = EG_PASSED_BONUS[rank_idx];

            // King proximity adjustment (endgame): friendly king close = +bonus, enemy king close = -bonus
            if rank_idx >= 3 {
                let promo_x = x;
                let promo_y = if is_white { 8i64 } else { 1i64 };
                let (own_king, opp_king) = if is_white {
                    (&white_king, &black_king)
                } else {
                    (&black_king, &white_king)
                };
                let own_dist = cheb(own_king.x, own_king.y, promo_x, promo_y).min(5) as i32;
                let opp_dist = cheb(opp_king.x, opp_king.y, promo_x, promo_y).min(5) as i32;
                let w = (rank_idx as i32 - 2) * 5;
                eg_bonus += (opp_dist - own_dist) * w;
            }

            mg[ci] += mg_bonus;
            eg[ci] += eg_bonus;
        }
    }

    // ---- King Safety: pawn shelter ----
    // For each king evaluate the 3 files around it (king file, ±1)
    for color_idx in 0..2usize {
        let (king, own_pawns_arr, is_white_king): (&Coordinate, &[_], bool) = if color_idx == 0 {
            (&white_king, pawns.as_slice(), true)
        } else {
            (&black_king, pawns.as_slice(), false)
        };

        let kf = (king.x - 1).clamp(0, 7) as usize;
        let mut shelter_mg = 0i32;

        for df in -1i64..=1i64 {
            let f = (kf as i64 + df).clamp(0, 7) as usize;
            let file_dist = df.unsigned_abs() as usize; // 0 = king file, 1 = adjacent
            let file_rel = file_dist.min(2);

            // Find closest own pawn ahead (toward promotion)
            let best_pawn_rank_dist: Option<i32> = own_pawns_arr
                .iter()
                .filter(|&&(px, py, pw)| {
                    pw == is_white_king
                        && (px - 1).clamp(0, 7) as usize == f
                        && if is_white_king { py > king.y } else { py < king.y }
                })
                .map(|&(_, py, _)| (py - king.y).abs() as i32)
                .min();

            match best_pawn_rank_dist {
                Some(dist) if dist <= 3 => {
                    let d = dist as usize;
                    shelter_mg += SHELTER_BONUS[d][file_rel];
                }
                _ => {
                    shelter_mg -= SHELTER_MISSING_PENALTY[file_rel];
                }
            }
        }
        // Shelter applies to mg only
        mg[color_idx] += shelter_mg;
    }

    // ---- King Safety: attacker danger ----
    // Apply as quadratic penalty (only in midgame)
    if w_king_danger > 0 {
        let penalty = w_king_danger * w_king_danger / 256;
        mg[0] -= penalty;
    }
    if b_king_danger > 0 {
        let penalty = b_king_danger * b_king_danger / 256;
        mg[1] -= penalty;
    }

    // ---- Tapered score ----
    let side = if game.turn == PlayerColor::White { 0 } else { 1 };
    let other = side ^ 1;

    let mg_score = mg[side] - mg[other];
    let eg_score = eg[side] - eg[other];

    let mg_phase = game_phase.min(MAX_PHASE);
    let eg_phase = MAX_PHASE - mg_phase;

    (mg_score * mg_phase + eg_score * eg_phase) / MAX_PHASE
}

// ==================== Mobility Counting ====================

fn count_knight_mobility(board: &crate::board::Board, x: i64, y: i64, piece: crate::board::Piece) -> i32 {
    let our_color = piece.color();
    let mut count = 0i32;
    for (dx, dy) in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)] {
        let nx = x + dx;
        let ny = y + dy;
        if !(1..=8).contains(&nx) || !(1..=8).contains(&ny) { continue; }
        if let Some(p) = board.get_piece(nx, ny) {
            if p.color() != our_color && p.color() != PlayerColor::Neutral {
                count += 1;
            }
        } else {
            count += 1;
        }
    }
    count
}

fn count_sliding_mobility(board: &crate::board::Board, x: i64, y: i64, piece: crate::board::Piece) -> i32 {
    let pt = piece.piece_type();
    let our_color = piece.color();
    let dirs: &[(i64, i64)] = match pt {
        PieceType::Bishop => &[(1,1),(1,-1),(-1,1),(-1,-1)],
        PieceType::Rook   => &[(1,0),(-1,0),(0,1),(0,-1)],
        PieceType::Queen  => &[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)],
        _ => return 0,
    };
    let mut count = 0i32;
    for &(dx, dy) in dirs {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while (1..=8).contains(&nx) && (1..=8).contains(&ny) {
            if let Some(p) = board.get_piece(nx, ny) {
                if p.color() != our_color && p.color() != PlayerColor::Neutral {
                    count += 1;
                }
                break;
            }
            count += 1;
            nx += dx;
            ny += dy;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    fn create_chess_game() -> GameState {
        let mut game = GameState::new();
        game.variant = Some(crate::Variant::Chess);
        game
    }

    fn setup_standard_chess_opening(game: &mut GameState) {
        game.setup_position_from_icn(
            "w (8;q|1;q) R1,1|N2,1|B3,1|Q4,1|K5,1|B6,1|N7,1|R8,1|\
            P1,2|P2,2|P3,2|P4,2|P5,2|P6,2|P7,2|P8,2|\
            r1,8|n2,8|b3,8|q4,8|k5,8|b6,8|n7,8|r8,8|\
            p1,7|p2,7|p3,7|p4,7|p5,7|p6,7|p7,7|p8,7",
        );
    }

    fn create_chess_game_from_icn(icn: &str) -> GameState {
        let mut game = create_chess_game();
        game.setup_position_from_icn(icn);
        game
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
        let mut game = create_chess_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,4");
        game.turn = PlayerColor::White;
        let score = evaluate(&game);
        assert!(
            score > 800,
            "White should have significant advantage with extra queen"
        );
    }
}
