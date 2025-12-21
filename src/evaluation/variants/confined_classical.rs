// Confined Classical Variant Evaluator
//
// Special logic for Confined Classical:
// 1. Obstacles surround pieces making early activation harder
// 2. DON'T eat own back-rank obstacles (y=0 for white, y=9 for black) - they protect king
// 3. Less aggressive queen - don't rush out early
// 4. King prefers CENTER instead of castled corners (obstacles provide cover)
// 5. Focus on piece development - get all minor pieces out before attacking

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::game::GameState;

use crate::evaluation::base;

// Confined Classical constants - STRONG values to enforce positional play
const QUEEN_EARLY_PENALTY: i32 = 60; // Strong penalty for queen out before minors
const KING_CENTER_BONUS: i32 = 35; // Bonus for king in center (safer with obstacles)
const DEVELOPMENT_BONUS_PER_PIECE: i32 = 40; // Strong bonus per developed minor
const UNDEVELOPED_PIECE_PENALTY: i32 = 25; // Penalty per piece still on back rank
const BACK_OBSTACLE_VALUE: i32 = 30; // Value of each back-rank obstacle (protection)

/// Main Confined Classical evaluation
#[inline]
pub fn evaluate(game: &GameState) -> i32 {
    // Check for insufficient material draw
    match crate::evaluation::insufficient_material::evaluate_insufficient_material(game) {
        Some(0) => return 0, // Dead draw
        Some(divisor) => {
            // Drawish - dampen eval
            return evaluate_inner(game) / divisor;
        }
        None => {} // Sufficient - continue to normal eval
    }

    evaluate_inner(game)
}

/// Core evaluation logic - skips insufficient material check
#[inline]
fn evaluate_inner(game: &GameState) -> i32 {
    // Start with material score
    let mut score = game.material_score;

    // Use cached king positions (O(1) instead of O(n) board scan)
    let (white_king, black_king) = (game.white_king_pos, game.black_king_pos);

    // Try scaled mop-up evaluation based on material imbalance
    let mut mop_up_applied = false;

    // Check if black is losing
    if let Some(_scale) =
        crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black)
    {
        if let (Some(wk), Some(bk)) = (&white_king, &black_king) {
            score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                wk,
                bk,
                PlayerColor::White,
                PlayerColor::Black,
            );
            mop_up_applied = true;
        }
    }

    // Check if white is losing
    if !mop_up_applied {
        if let Some(_scale) =
            crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White)
        {
            if let (Some(wk), Some(bk)) = (&white_king, &black_king) {
                score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
                    game,
                    bk,
                    wk,
                    PlayerColor::Black,
                    PlayerColor::White,
                );
                mop_up_applied = true;
            }
        }
    }

    // If mop-up wasn't applied, use normal Confined Classical evaluation
    if !mop_up_applied {
        // Custom piece evaluation for Confined Classical
        score += evaluate_pieces_confined(game, &white_king, &black_king);

        // Custom king safety - prefer center
        score += evaluate_king_position_confined(&white_king, &black_king);

        // Development and structure
        score += evaluate_development_confined(game);
        score += base::evaluate_pawn_structure(game);
    }

    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Evaluate pieces for Confined Classical
fn evaluate_pieces_confined(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut score: i32 = 0;
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let mut white_bishop_colors: (bool, bool) = (false, false);
    let mut black_bishop_colors: (bool, bool) = (false, false);

    // Track development
    let mut white_minors_developed = 0;
    let mut black_minors_developed = 0;
    let mut white_queen_out = false;
    let mut black_queen_out = false;

    for ((x, y), piece) in game.board.iter() {
        if piece.color() == PlayerColor::Neutral {
            continue;
        }

        let is_white = piece.color() == PlayerColor::White;
        let mut piece_score: i32 = 0;

        match piece.piece_type() {
            PieceType::Rook => {
                piece_score +=
                    base::evaluate_rook(game, *x, *y, piece.color(), white_king, black_king);

                // Bonus for rook invading enemy territory (behind their obstacles)
                let in_enemy_territory = if is_white { *y >= 8 } else { *y <= 1 };
                if in_enemy_territory {
                    piece_score += 40;
                }
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                // Less aggressive queen evaluation for Confined Classical
                piece_score +=
                    evaluate_queen_confined(game, *x, *y, piece.color(), white_king, black_king);

                // Track if queen has left starting position
                let start_y = if is_white { 1 } else { 8 };
                if *y != start_y {
                    if is_white {
                        white_queen_out = true;
                    } else {
                        black_queen_out = true;
                    }
                }
            }
            PieceType::Knight => {
                // Custom knight eval: FORWARD development, not huddling around king
                piece_score += evaluate_knight_confined(*x, *y, piece.color());

                // Track development
                let back_rank = if is_white { 1 } else { 8 };
                if *y != back_rank {
                    if is_white {
                        white_minors_developed += 1;
                    } else {
                        black_minors_developed += 1;
                    }
                }
            }
            PieceType::Bishop => {
                // Custom bishop eval: forward and into enemy territory
                piece_score += evaluate_bishop_confined(*x, *y, piece.color());

                if is_white {
                    white_bishops += 1;
                    if (*x + *y) % 2 == 0 {
                        white_bishop_colors.0 = true;
                    } else {
                        white_bishop_colors.1 = true;
                    }
                } else {
                    black_bishops += 1;
                    if (*x + *y) % 2 == 0 {
                        black_bishop_colors.0 = true;
                    } else {
                        black_bishop_colors.1 = true;
                    }
                }

                // Track development
                let back_rank = if is_white { 1 } else { 8 };
                if *y != back_rank {
                    if is_white {
                        white_minors_developed += 1;
                    } else {
                        black_minors_developed += 1;
                    }
                }
            }
            PieceType::Pawn => {
                // Custom pawn eval: center pawn pushes are great
                piece_score += evaluate_pawn_confined(
                    *x,
                    *y,
                    piece.color(),
                    game.white_promo_rank,
                    game.black_promo_rank,
                );
            }
            _ => {
                // Other pieces get no special evaluation
            }
        }

        if is_white {
            score += piece_score;
        } else {
            score -= piece_score;
        }
    }

    // Bishop pair bonus
    if white_bishops >= 2 {
        score += 60;
        if white_bishop_colors.0 && white_bishop_colors.1 {
            score += 20;
        }
    }
    if black_bishops >= 2 {
        score -= 60;
        if black_bishop_colors.0 && black_bishop_colors.1 {
            score -= 20;
        }
    }

    // Penalize queen out before minors developed
    if white_queen_out && white_minors_developed < 3 {
        score -= QUEEN_EARLY_PENALTY;
    }
    if black_queen_out && black_minors_developed < 3 {
        score += QUEEN_EARLY_PENALTY;
    }

    // Bonus for developing minors AND penalty for undeveloped pieces
    // White: 4 minors (2 knights + 2 bishops), Black: 4 minors
    let white_undeveloped = (4 - white_minors_developed).max(0);
    let black_undeveloped = (4 - black_minors_developed).max(0);

    score += white_minors_developed * DEVELOPMENT_BONUS_PER_PIECE;
    score -= black_minors_developed * DEVELOPMENT_BONUS_PER_PIECE;
    score -= white_undeveloped * UNDEVELOPED_PIECE_PENALTY;
    score += black_undeveloped * UNDEVELOPED_PIECE_PENALTY;

    score
}

/// Queen evaluation for Confined Classical - less aggressive
fn evaluate_queen_confined(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut bonus: i32 = 0;

    // Basic tropism to enemy king (but weaker than normal)
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let capped = dist.min(20);
        // Much weaker tropism than normal queen eval
        bonus += ((20 - capped) as i32) / 2;
    }

    // Slight centralization bonus instead of aggressive positioning
    let center_dist = (x - 4).abs() + (y - 4).abs();
    if center_dist <= 3 {
        bonus += (4 - center_dist as i32) * 3;
    }

    // Far slider penalty still applies
    let own_king = if color == PlayerColor::White {
        white_king
    } else {
        black_king
    };
    if let (Some(ek), Some(ok)) = (enemy_king, own_king) {
        let cheb_enemy = (x - ek.x).abs().max((y - ek.y).abs());
        let cheb_own = (x - ok.x).abs().max((y - ok.y).abs());
        let cheb = cheb_enemy.min(cheb_own);
        if cheb > 18 {
            let excess = (cheb - 18).min(40) as i32;
            bonus -= excess * 2;
        }
    }

    // Penalize queen in front of own king (blocking development)
    if let Some(ok) = own_king {
        let in_front = match color {
            PlayerColor::White => y > ok.y && y <= ok.y + 2 && (x - ok.x).abs() <= 1,
            PlayerColor::Black => y < ok.y && y >= ok.y - 2 && (x - ok.x).abs() <= 1,
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        };
        if in_front {
            bonus -= 20;
        }
    }

    // Idle penalty from pawns on file (same as base)
    let (own_pawns_on_file, enemy_pawns_on_file) = base::count_pawns_on_file(game, x, color);
    if own_pawns_on_file > 0 && enemy_pawns_on_file > 0 {
        bonus -= 15;
    }

    bonus
}

/// King position for Confined Classical - prefer center over corners
fn evaluate_king_position_confined(
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut score: i32 = 0;

    // White king - prefer center files (3-6)
    if let Some(wk) = white_king {
        // Center bonus
        if wk.x >= 3 && wk.x <= 6 {
            score += KING_CENTER_BONUS;
        }
        // Mild penalty for corner castling (less safe in Confined)
        if wk.x <= 2 || wk.x >= 7 {
            score -= KING_CENTER_BONUS / 2;
        }
    }

    // Black king - same logic
    if let Some(bk) = black_king {
        if bk.x >= 3 && bk.x <= 6 {
            score -= KING_CENTER_BONUS;
        }
        if bk.x <= 2 || bk.x >= 7 {
            score += KING_CENTER_BONUS / 2;
        }
    }

    score
}

/// Evaluate development for Confined Classical
/// Penalizes capturing own back-rank obstacles (y=0 for white, y=9 for black)
fn evaluate_development_confined(game: &GameState) -> i32 {
    let mut score: i32 = 0;

    // Count obstacles on back ranks
    // White's protective obstacles are at y=0, black's are at y=9
    let mut white_back_obstacles = 0;
    let mut black_back_obstacles = 0;

    for ((_, y), piece) in game.board.iter() {
        if piece.piece_type() == PieceType::Obstacle {
            if *y == 0 {
                white_back_obstacles += 1;
            } else if *y == 9 {
                black_back_obstacles += 1;
            }
        }
    }

    // Bonus for keeping back obstacles (they protect the king)
    // In starting position there should be ~8 obstacles per side on back rank
    // Having more is good (we haven't eaten our own protective obstacles)
    score += white_back_obstacles * BACK_OBSTACLE_VALUE;
    score -= black_back_obstacles * BACK_OBSTACLE_VALUE;

    score
}

// ==================== Forward-Focused Piece Evaluation ====================

/// Knight evaluation for Confined Classical - push FORWARD, not huddle around king
fn evaluate_knight_confined(x: i64, y: i64, color: PlayerColor) -> i32 {
    let mut bonus: i32 = 0;

    // Forward advancement bonus - knights should push into enemy territory
    match color {
        PlayerColor::White => {
            // Bonus for advancing: y=3 is ok (+10), y=4-5 is good (+20), y>=6 is great (+35)
            if y >= 6 {
                bonus += 35;
            } else if y >= 4 {
                bonus += 20;
            } else if y >= 3 {
                bonus += 10;
            }
            // Behind opponent's obstacle line (y>=9) is excellent
            if y >= 9 {
                bonus += 25;
            }
        }
        PlayerColor::Black => {
            if y <= 3 {
                bonus += 35;
            } else if y <= 5 {
                bonus += 20;
            } else if y <= 6 {
                bonus += 10;
            }
            // Behind opponent's obstacle line (y<=0) is excellent
            if y <= 0 {
                bonus += 25;
            }
        }
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }

    // Central file bonus (files 3-6 are central)
    if x >= 3 && x <= 6 {
        bonus += 10;
    }

    bonus
}

/// Bishop evaluation for Confined Classical - forward diagonals and enemy territory
fn evaluate_bishop_confined(x: i64, y: i64, color: PlayerColor) -> i32 {
    let mut bonus: i32 = 0;

    // Forward advancement bonus
    match color {
        PlayerColor::White => {
            if y >= 6 {
                bonus += 25; // Deep in enemy territory
            } else if y >= 4 {
                bonus += 15; // Crossing center
            }
            // Behind enemy obstacles
            if y >= 9 {
                bonus += 30;
            }
        }
        PlayerColor::Black => {
            if y <= 3 {
                bonus += 25;
            } else if y <= 5 {
                bonus += 15;
            }
            if y <= 0 {
                bonus += 30;
            }
        }
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }

    // Diagonal control (on main diagonals is good for long-range pressure)
    if (x - y).abs() <= 2 || (x + y - 8).abs() <= 2 {
        bonus += 8;
    }

    bonus
}

/// Pawn evaluation for Confined Classical - center pawn advancement is key
fn evaluate_pawn_confined(
    x: i64,
    y: i64,
    color: PlayerColor,
    white_promo: i64,
    black_promo: i64,
) -> i32 {
    let mut bonus: i32 = 0;

    // Base advancement bonus
    match color {
        PlayerColor::White => {
            let dist = (white_promo - y).max(0);
            bonus += ((8 - dist.min(8)) as i32) * 3;
        }
        PlayerColor::Black => {
            let dist = (y - black_promo).max(0);
            bonus += ((8 - dist.min(8)) as i32) * 3;
        }
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }

    // CENTER PAWN BONUS - d and e pawn pushes open up the position
    // Files 4 and 5 (d and e in standard notation)
    if x == 4 || x == 5 {
        match color {
            PlayerColor::White => {
                // d4/e4 opening gives +20, d5/e5 even better
                if y == 4 {
                    bonus += 25; // d4/e4 - opening the position
                } else if y >= 5 {
                    bonus += 30; // Even more advanced center pawns
                } else if y == 3 {
                    bonus += 15; // d3/e3 is ok but d4/e4 is better
                }
            }
            PlayerColor::Black => {
                if y == 5 {
                    bonus += 25; // d5/e5 opening
                } else if y <= 4 {
                    bonus += 30;
                } else if y == 6 {
                    bonus += 15;
                }
            }
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    // Wing pawns are less valuable in the opening (keep focus on center)
    if x <= 2 || x >= 7 {
        bonus -= 5;
    }

    bonus
}
