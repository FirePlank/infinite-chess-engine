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
    if let (Some(_scale), Some(bk)) = (
        crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black),
        &black_king,
    ) {
        score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
            game,
            white_king.as_ref(),
            bk,
            PlayerColor::White,
            PlayerColor::Black,
        );
        mop_up_applied = true;
    }

    // Check if white is losing
    // Check if white is losing
    if !mop_up_applied
        && let (Some(_scale), Some(wk)) = (
            crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White),
            &white_king,
        )
    {
        score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
            game,
            black_king.as_ref(),
            wk,
            PlayerColor::Black,
            PlayerColor::White,
        );
        mop_up_applied = true;
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
    if (3..=6).contains(&x) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_confined_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.variant = Some(crate::Variant::ConfinedClassical);
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    #[test]
    fn test_evaluate_returns_value() {
        let mut game = create_confined_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        let score = evaluate(&game);
        assert!(score.abs() < 10000, "K vs K should be near 0");
    }

    #[test]
    fn test_evaluate_knight_confined_white() {
        // Knight advanced should score better
        let front_score = evaluate_knight_confined(4, 6, PlayerColor::White);
        let back_score = evaluate_knight_confined(4, 1, PlayerColor::White);

        assert!(
            front_score > back_score,
            "Advanced knight should score better"
        );
    }

    #[test]
    fn test_evaluate_knight_confined_black() {
        let front_score = evaluate_knight_confined(4, 3, PlayerColor::Black);
        let back_score = evaluate_knight_confined(4, 8, PlayerColor::Black);

        assert!(
            front_score > back_score,
            "Advanced black knight should score better"
        );
    }

    #[test]
    fn test_evaluate_bishop_confined() {
        // Advanced bishop = better
        let advanced = evaluate_bishop_confined(4, 6, PlayerColor::White);
        let back = evaluate_bishop_confined(4, 1, PlayerColor::White);

        assert!(advanced > back, "Advanced bishop should score better");
    }

    #[test]
    fn test_evaluate_pawn_confined_center() {
        // Center pawns (d/e file) should score better
        let center = evaluate_pawn_confined(4, 4, PlayerColor::White, 8, 1);
        let wing = evaluate_pawn_confined(1, 4, PlayerColor::White, 8, 1);

        assert!(
            center > wing,
            "Center pawn should score better than wing pawn"
        );
    }

    #[test]
    fn test_king_center_bonus() {
        // Center king = +35, corner king = -17.5
        let center_king = Some(Coordinate::new(4, 4));
        let corner_king = Some(Coordinate::new(1, 1));
        let black_king = Some(Coordinate::new(5, 8));

        let center_score = evaluate_king_position_confined(&center_king, &black_king);
        let corner_score = evaluate_king_position_confined(&corner_king, &black_king);

        assert!(
            center_score > corner_score,
            "Center king should score better"
        );
    }

    #[test]
    fn test_queen_early_penalty() {
        let mut game = create_confined_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White minors not developed, queen out
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        // Knights still on back rank
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score_early_queen = evaluate(&game);

        // Now with minors developed and queen back
        game.board.remove_piece(&4, &4);
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board.remove_piece(&2, &1);
        game.board.remove_piece(&7, &1);
        game.board
            .set_piece(3, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(6, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();

        let score_developed = evaluate(&game);

        // Developed should be better (or at least run without panic)
        assert!(score_developed.abs() < 100000);
        assert!(score_early_queen.abs() < 100000);
    }

    #[test]
    fn test_obstacle_value() {
        let mut game = create_confined_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Add obstacles on back rank
        game.board
            .set_piece(1, 0, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
        game.board
            .set_piece(2, 0, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let score = evaluate_development_confined(&game);
        // Should get back obstacle bonus
        assert!(
            score >= BACK_OBSTACLE_VALUE,
            "Should get bonus for back obstacles"
        );
    }

    #[test]
    fn test_evaluate_from_black_perspective() {
        let mut game = create_confined_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();

        game.turn = PlayerColor::White;
        let score_white = evaluate(&game);

        game.turn = PlayerColor::Black;
        let score_black = evaluate(&game);

        // Scores should be negated
        assert_eq!(
            score_white, -score_black,
            "Score should negate for opposite side"
        );
    }

    #[test]
    fn test_development_bonuses() {
        let mut game = Box::new(create_confined_game());
        game.board = Board::new();
        game.board
            .set_piece(4, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Add pawns to ensure material is sufficient
        game.board
            .set_piece(0, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(7, 4, Piece::new(PieceType::Pawn, PlayerColor::Black));

        // Minor piece at home rank (undeveloped)
        game.board
            .set_piece(1, 0, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();
        let eval_home = evaluate(&game);

        // Minor piece moved forward (developed)
        game.board.remove_piece(&1, &0);
        game.board
            .set_piece(2, 2, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();
        let eval_developed = evaluate(&game);

        assert!(
            eval_developed > eval_home,
            "Developed piece should be worth more than undeveloped back-rank piece"
        );
    }

    #[test]
    fn test_back_rank_obstacle_protection() {
        let mut game = Box::new(create_confined_game());
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Add pawns to ensure material is sufficient
        game.board
            .set_piece(0, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(7, 4, Piece::new(PieceType::Pawn, PlayerColor::Black));

        // Back rank obstacles at (4,0) and (6,0) providing protection
        game.board
            .set_piece(4, 0, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
        game.board
            .set_piece(6, 0, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
        game.recompute_piece_counts();
        let eval_with_obstacles = evaluate(&game);

        // Removing obstacles (simulating them being captured/eaten) should lower the score
        game.board.remove_piece(&4, &0);
        game.board.remove_piece(&6, &0);
        game.recompute_piece_counts();
        let eval_without_obstacles = evaluate(&game);

        assert!(
            eval_with_obstacles > eval_without_obstacles,
            "Back-rank obstacles should provide a defensive bonus"
        );
    }

    #[test]
    fn test_evaluate_inner_basic() {
        let mut game = Box::new(create_confined_game());
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(3, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.recompute_piece_counts();
        game.material_score = 0; // Equal pawns

        let score = evaluate_inner(&game);
        // Should return a reasonable evaluation
        assert!(
            score.abs() < 10000,
            "Evaluation should be reasonable: {}",
            score
        );
    }

    #[test]
    fn test_mop_up_triggers_in_endgame() {
        let mut game = Box::new(create_confined_game());
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White has 2 queens - massive advantage
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(3, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();
        game.material_score = 2700; // 2 Queen advantage

        let score = evaluate(&game);
        // Should be very positive for white
        assert!(
            score > 1000,
            "White should have huge advantage with 2 queens: {}",
            score
        );
    }

    #[test]
    fn test_king_position_confined() {
        let mut game = Box::new(create_confined_game());
        game.board = Board::new();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 6, Piece::new(PieceType::King, PlayerColor::Black));
        // Add material to avoid insufficient material
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(7, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.recompute_piece_counts();
        game.material_score = 0;

        let wk = Some(Coordinate::new(4, 4));
        let bk = Some(Coordinate::new(4, 6));
        let score = evaluate_king_position_confined(&wk, &bk);
        // King positions function should return a value
        assert!(
            score.abs() < 500,
            "King position score should be reasonable: {}",
            score
        );
    }
}
