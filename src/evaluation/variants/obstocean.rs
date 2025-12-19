// Obstocean Variant Evaluator
//
// Special logic for Obstocean:
// 1. Skip Cloud/Development penalties (knights move freely)
// 2. Edge pawn priority (edge > center)
// 3. Knight Rush (y-advancement instead of centralization)
// 4. Tunneling Pawn bonus (obstacles ahead = safe passage)
// 5. Blocking Bonus (penalize enemy free runners)
// 6. Free Runner / Intercept Gradient

use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;

use crate::evaluation::base;

/// Main Obstocean evaluation - delegates most logic to base, overrides pawn structure
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

    // Check for endgame with lone king
    let white_only_king = base::is_lone_king(game, PlayerColor::White);
    let black_only_king = base::is_lone_king(game, PlayerColor::Black);

    // Handle lone king endgames
    if black_only_king && black_king.is_some() {
        let our_king = white_king
            .as_ref()
            .cloned()
            .unwrap_or(Coordinate { x: 4, y: 4 });
        score += base::evaluate_lone_king_endgame(
            game,
            &our_king,
            black_king.as_ref().unwrap(),
            PlayerColor::White,
        );
    } else if white_only_king && white_king.is_some() {
        let our_king = black_king
            .as_ref()
            .cloned()
            .unwrap_or(Coordinate { x: 4, y: 4 });
        score -= base::evaluate_lone_king_endgame(
            game,
            &our_king,
            white_king.as_ref().unwrap(),
            PlayerColor::Black,
        );
    } else {
        // Obstocean-specific piece and pawn evaluation
        score += evaluate_pieces_obstocean(game, &white_king, &black_king);
        score += base::evaluate_king_safety(game, &white_king, &black_king);
        score += evaluate_pawn_structure_obstocean(game);
    }

    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Obstocean piece evaluation - skips Cloud and Development penalties
fn evaluate_pieces_obstocean(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut score: i32 = 0;
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let mut white_bishop_colors: (bool, bool) = (false, false);
    let mut black_bishop_colors: (bool, bool) = (false, false);

    for ((x, y), piece) in game.board.iter() {
        if piece.color() == PlayerColor::Neutral {
            continue;
        }

        let mut piece_score: i32 = 0;

        match piece.piece_type() {
            PieceType::Rook | PieceType::Chancellor | PieceType::Amazon => {
                piece_score +=
                    base::evaluate_rook(game, *x, *y, piece.color(), white_king, black_king);
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                piece_score +=
                    base::evaluate_queen(game, *x, *y, piece.color(), white_king, black_king);
            }
            PieceType::Knight
            | PieceType::Hawk
            | PieceType::Archbishop
            | PieceType::Centaur
            | PieceType::RoyalCentaur => {
                // Obstocean Knight Rush: Just push forward
                piece_score += evaluate_knight_obstocean(*x, *y, piece.color());
            }
            PieceType::Bishop => {
                piece_score +=
                    base::evaluate_bishop(game, *x, *y, piece.color(), white_king, black_king);
                if piece.color() == PlayerColor::White {
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
            }
            PieceType::Pawn => {
                piece_score += evaluate_pawn_position_obstocean(
                    *x,
                    *y,
                    piece.color(),
                    game.white_promo_rank,
                    game.black_promo_rank,
                );
            }
            _ => {}
        }

        // NO CLOUD PENALTY for Obstocean
        // NO DEVELOPMENT PENALTY for Obstocean

        if piece.color() == PlayerColor::White {
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

    score
}

/// Obstocean Knight: Just push forward, no centralization or king safety
#[inline]
fn evaluate_knight_obstocean(_x: i64, y: i64, color: PlayerColor) -> i32 {
    match color {
        PlayerColor::White => y as i32 * 10,
        PlayerColor::Black => (20 - y).max(0) as i32 * 10,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }
}

/// Obstocean Pawn Position: Edge > Center
#[inline]
fn evaluate_pawn_position_obstocean(
    x: i64,
    y: i64,
    color: PlayerColor,
    white_promo_rank: i64,
    black_promo_rank: i64,
) -> i32 {
    let mut bonus: i32 = 0;

    // Advancement bonus
    match color {
        PlayerColor::White => {
            let dist = (white_promo_rank - y).max(0);
            bonus += ((8 - dist.min(8)) as i32) * 3;
        }
        PlayerColor::Black => {
            let dist = (y - black_promo_rank).max(0);
            bonus += ((8 - dist.min(8)) as i32) * 3;
        }
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }

    // Edge pawn priority for Obstocean
    if x.abs() > 3 {
        bonus += 20; // Big bonus for edge pawns
    }
    if x >= 3 && x <= 5 {
        bonus -= 10; // Penalty for center pawns (clogged)
    }

    bonus
}

/// Obstocean pawn structure with all the custom bonuses
fn evaluate_pawn_structure_obstocean(game: &GameState) -> i32 {
    let mut score: i32 = 0;

    let mut white_pawns: Vec<(i64, i64)> = Vec::new();
    let mut black_pawns: Vec<(i64, i64)> = Vec::new();
    let mut white_pawn_files: Vec<i64> = Vec::new();
    let mut black_pawn_files: Vec<i64> = Vec::new();

    for ((x, y), piece) in game.board.iter() {
        if piece.piece_type() == PieceType::Pawn {
            if piece.color() == PlayerColor::White {
                white_pawns.push((*x, *y));
                white_pawn_files.push(*x);
            } else if piece.color() == PlayerColor::Black {
                black_pawns.push((*x, *y));
                black_pawn_files.push(*x);
            }
        }
    }

    // Doubled pawn penalty
    white_pawn_files.sort();
    let mut prev_file: Option<i64> = None;
    for &file in &white_pawn_files {
        if prev_file == Some(file) {
            score -= 8;
        }
        prev_file = Some(file);
    }

    black_pawn_files.sort();
    prev_file = None;
    for &file in &black_pawn_files {
        if prev_file == Some(file) {
            score += 8;
        }
        prev_file = Some(file);
    }

    // ===== WHITE PAWNS =====
    for (wx, wy) in &white_pawns {
        // Passed pawn check
        let mut is_passed = true;
        for (bx, by) in &black_pawns {
            if (*bx - wx).abs() <= 1 && *by > *wy {
                is_passed = false;
                break;
            }
        }
        if is_passed {
            score += 20;
        }

        // Tunneling Pawn Heuristic
        score += evaluate_tunnel(*wx, *wy, game.white_promo_rank, &game.board, true);

        // Blocking Bonus
        score += evaluate_blocking(
            *wx,
            *wy,
            &black_pawns,
            game.black_promo_rank,
            &game.board,
            true,
        );

        // Free Runner Bonus
        score += evaluate_free_runner(
            *wx,
            *wy,
            game.white_promo_rank,
            &game.board,
            PlayerColor::Black,
            true,
        );
    }

    // ===== BLACK PAWNS =====
    for (bx, by) in &black_pawns {
        // Passed pawn check
        let mut is_passed = true;
        for (wx, wy) in &white_pawns {
            if (*wx - bx).abs() <= 1 && *wy < *by {
                is_passed = false;
                break;
            }
        }
        if is_passed {
            score -= 20;
        }

        // Tunneling Pawn Heuristic
        score -= evaluate_tunnel(*bx, *by, game.black_promo_rank, &game.board, false);

        // Blocking Bonus
        score -= evaluate_blocking(
            *bx,
            *by,
            &white_pawns,
            game.white_promo_rank,
            &game.board,
            false,
        );

        // Free Runner Bonus
        score -= evaluate_free_runner(
            *bx,
            *by,
            game.black_promo_rank,
            &game.board,
            PlayerColor::White,
            false,
        );
    }

    score
}

/// Tunnel detection: Bonus for pawns with obstacles forming a "safe corridor"
fn evaluate_tunnel(px: i64, py: i64, promo_rank: i64, board: &Board, is_white: bool) -> i32 {
    let limit = if is_white {
        (promo_rank - py).max(0).min(20)
    } else {
        (py - promo_rank).max(0).min(20)
    };

    if limit == 0 {
        return 0;
    }

    let mut weighted_density = 0.0;
    let mut total_steps: f64 = 0.0;

    for i in 1..=limit {
        let check_y = if is_white { py + i } else { py - i };

        // Center obstacle
        if let Some(p) = board.get_piece(px, check_y) {
            if p.piece_type() == PieceType::Obstacle {
                weighted_density += 1.0;
            }
        }

        // Adjacent obstacles (eating shield) - worth more
        if let Some(p) = board.get_piece(px - 1, check_y) {
            if p.piece_type() == PieceType::Obstacle {
                weighted_density += 2.0;
            }
        }
        if let Some(p) = board.get_piece(px + 1, check_y) {
            if p.piece_type() == PieceType::Obstacle {
                weighted_density += 2.0;
            }
        }

        total_steps += 1.0;
    }

    if weighted_density > 0.0 {
        let advancement_factor = (20.0 - limit as f64).max(1.0);
        let base_bonus = 1.5 * advancement_factor;
        let density_score = weighted_density / total_steps.max(1.0);
        let multiplier = 1.0 + density_score;
        let final_bonus = (base_bonus * multiplier) as i32;
        return final_bonus.min(80);
    }

    0
}

/// Blocking bonus: Reward pawns that block enemy pawns
fn evaluate_blocking(
    px: i64,
    py: i64,
    enemy_pawns: &[(i64, i64)],
    enemy_promo_rank: i64,
    _board: &Board,
    is_white: bool,
) -> i32 {
    let scan_limit = 20i64;

    for i in 1..=scan_limit {
        let check_y = if is_white { py + i } else { py - i };

        for (ex, ey) in enemy_pawns {
            if *ex == px && *ey == check_y {
                // Found enemy pawn on our file
                let enemy_dist = if is_white {
                    (check_y - enemy_promo_rank).max(0)
                } else {
                    (enemy_promo_rank - check_y).max(0)
                };
                let urgency = (20 - enemy_dist).max(0);
                let proximity = (scan_limit - i).max(0);

                if urgency > 0 {
                    return ((urgency * 5) + (proximity * 2)) as i32;
                }
                return 0;
            }
        }
    }

    0
}

/// Free Runner bonus: Reward pawns with no enemy pieces ahead
fn evaluate_free_runner(
    px: i64,
    py: i64,
    promo_rank: i64,
    board: &Board,
    enemy_color: PlayerColor,
    is_white: bool,
) -> i32 {
    let limit = if is_white {
        (promo_rank - py).max(0).min(50)
    } else {
        (py - promo_rank).max(0).min(50)
    };

    // Check if contested
    for i in 1..=limit {
        let check_y = if is_white { py + i } else { py - i };
        if let Some(p) = board.get_piece(px, check_y) {
            if p.color() == enemy_color {
                return 0; // Contested, no bonus
            }
        }
    }

    // Free Runner!
    let dist = if is_white {
        (promo_rank - py).max(0)
    } else {
        (py - promo_rank).max(0)
    };
    let urgency = (20 - dist).max(0);

    if urgency > 0 {
        let target_y = if is_white { py + 1 } else { py - 1 };
        let intercept_dist = get_closest_piece_distance(board, px, target_y, enemy_color);

        // Steeper gradient: bonus scales with distance
        let dist_factor = (intercept_dist.min(20) as f32) / 20.0;
        let effective_bonus = (urgency as f32 * 45.0 * dist_factor) as i32;

        return effective_bonus;
    }

    0
}

/// Helper: Find closest enemy piece to a target square
fn get_closest_piece_distance(
    board: &Board,
    target_x: i64,
    target_y: i64,
    color: PlayerColor,
) -> i64 {
    let mut min_dist = 1000i64;
    for (x, y, _piece) in board
        .tiles
        .iter_pieces_by_color(color == PlayerColor::White)
    {
        let dist = (x - target_x).abs().max((y - target_y).abs());
        if dist < min_dist {
            min_dist = dist;
        }
    }
    min_dist
}
