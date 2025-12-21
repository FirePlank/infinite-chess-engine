// Mop-Up Evaluation - Ultra-optimized version
//
// Specialized endgame evaluation for positions where opponent has few pieces.
// Only runs when:
// - Opponent has < 20% of starting non-pawn pieces (pawns NOT counted)
// - Winning side has at least one non-pawn piece

use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;

/// Don't run mop-up eval if opponent has >= 20% of starting non-pawn pieces
const MOP_UP_THRESHOLD_PERCENT: u32 = 20;

// ==================== Entry Points ====================

/// Check if a side only has a king (no other pieces)
#[inline(always)]
pub fn is_lone_king(game: &GameState, color: PlayerColor) -> bool {
    if color == PlayerColor::White {
        game.white_pawn_count == 0 && !game.white_non_pawn_material
    } else {
        game.black_pawn_count == 0 && !game.black_non_pawn_material
    }
}

/// Check if a side has any pawn that can still promote
#[inline(always)]
pub fn has_promotable_pawn(board: &Board, color: PlayerColor, promo_rank: i64) -> bool {
    let is_white = color == PlayerColor::White;
    for (_cx, cy, tile) in board.tiles.iter() {
        let color_pawns = tile.occ_pawns
            & if is_white {
                tile.occ_white
            } else {
                tile.occ_black
            };
        if color_pawns == 0 {
            continue;
        }
        let mut bits = color_pawns;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let y = cy * 8 + (idx / 8) as i64;
            if is_white {
                if y < promo_rank {
                    return true;
                }
            } else {
                if y > promo_rank {
                    return true;
                }
            }
        }
    }
    false
}

/// Calculate mop-up scaling factor (0-100). Returns None if:
/// - Opponent has >= 20% of starting non-pawn pieces
/// - Winning side has no non-pawn pieces (only king/pawns)
#[inline(always)]
pub fn calculate_mop_up_scale(game: &GameState, losing_color: PlayerColor) -> Option<u32> {
    // Count NON-PAWN pieces only (excluding king)
    let (losing_pieces, losing_starting) = if losing_color == PlayerColor::White {
        // white_piece_count includes all pieces, subtract pawns and king
        let current_non_pawn = game.white_piece_count.saturating_sub(game.white_pawn_count);
        let current_non_king = current_non_pawn.saturating_sub(1); // -1 for king
        let starting = game.starting_white_pieces.saturating_sub(1); // starting already excludes pawns, -1 for king
        (current_non_king, starting)
    } else {
        let current_non_pawn = game.black_piece_count.saturating_sub(game.black_pawn_count);
        let current_non_king = current_non_pawn.saturating_sub(1);
        let starting = game.starting_black_pieces.saturating_sub(1);
        (current_non_king, starting)
    };

    // Check winning side has at least one non-pawn piece
    let winning_has_pieces = if losing_color == PlayerColor::White {
        game.black_non_pawn_material
    } else {
        game.white_non_pawn_material
    };

    if !winning_has_pieces {
        return None; // Don't mop-up with just king+pawns
    }

    // If starting pieces is 0 and losing side has 0, it's lone king
    if losing_starting == 0 {
        return if losing_pieces == 0 { Some(100) } else { None };
    }

    // Calculate percentage of NON-PAWN material remaining
    let percent_remaining = (losing_pieces as u32 * 100) / (losing_starting as u32);

    if percent_remaining >= MOP_UP_THRESHOLD_PERCENT {
        return None;
    }

    // Scale: 0% = 100, 20% = 0
    Some(100 - (percent_remaining * 100 / MOP_UP_THRESHOLD_PERCENT).min(100))
}

/// Legacy entry point - unscaled evaluation
#[inline(always)]
pub fn evaluate_lone_king_endgame(
    game: &GameState,
    our_king: &Coordinate,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    evaluate_mop_up_core(game, our_king, enemy_king, winning_color)
}

/// Scaled mop-up evaluation - main entry point
#[inline(always)]
pub fn evaluate_mop_up_scaled(
    game: &GameState,
    our_king: &Coordinate,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
    losing_color: PlayerColor,
) -> i32 {
    let scale = match calculate_mop_up_scale(game, losing_color) {
        Some(s) if s > 0 => s,
        _ => return 0,
    };

    let raw = evaluate_mop_up_core(game, our_king, enemy_king, winning_color);
    (raw * scale as i32) / 100
}

// ==================== Core Evaluation ====================

/// Core mop-up evaluation - no allocations, minimal branching
#[inline(always)]
fn evaluate_mop_up_core(
    game: &GameState,
    our_king: &Coordinate,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    let mut bonus: i32 = 0;

    // Track closest fences in each direction using scalar min/max
    let mut ortho_y_min_above: i64 = i64::MAX;
    let mut ortho_y_max_below: i64 = i64::MIN;
    let mut ortho_x_min_right: i64 = i64::MAX;
    let mut ortho_x_max_left: i64 = i64::MIN;

    let mut diag_pos_min_above: i64 = i64::MAX;
    let mut diag_pos_max_below: i64 = i64::MIN;
    let mut diag_neg_min_above: i64 = i64::MAX;
    let mut diag_neg_max_below: i64 = i64::MIN;

    let mut ortho_count: u8 = 0;
    let mut diag_count: u8 = 0;
    let mut short_range_bonus: i32 = 0;

    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    let enemy_diag_pos = enemy_x + enemy_y;
    let enemy_diag_neg = enemy_x - enemy_y;

    // Single pass - collect fence positions and short-range bonuses
    let is_white = winning_color == PlayerColor::White;
    for (x, y, piece) in game.board.iter_pieces_by_color(is_white) {
        let pt = piece.piece_type();
        if pt.is_royal() || pt == PieceType::Pawn {
            continue;
        }

        // Orthogonal sliders
        let has_ortho = matches!(
            pt,
            PieceType::Rook
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Chancellor
                | PieceType::Amazon
        );

        if has_ortho {
            ortho_count += 1;
            if y > enemy_y && y < ortho_y_min_above {
                ortho_y_min_above = y;
            }
            if y < enemy_y && y > ortho_y_max_below {
                ortho_y_max_below = y;
            }
            if x > enemy_x && x < ortho_x_min_right {
                ortho_x_min_right = x;
            }
            if x < enemy_x && x > ortho_x_max_left {
                ortho_x_max_left = x;
            }
        }

        // Diagonal sliders
        let has_diag = matches!(
            pt,
            PieceType::Bishop
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Archbishop
                | PieceType::Amazon
        );

        if has_diag {
            diag_count += 1;
            let dp = x + y;
            let dn = x - y;
            if dp > enemy_diag_pos && dp < diag_pos_min_above {
                diag_pos_min_above = dp;
            }
            if dp < enemy_diag_pos && dp > diag_pos_max_below {
                diag_pos_max_below = dp;
            }
            if dn > enemy_diag_neg && dn < diag_neg_min_above {
                diag_neg_min_above = dn;
            }
            if dn < enemy_diag_neg && dn > diag_neg_max_below {
                diag_neg_max_below = dn;
            }
        }

        // Short-range pieces proximity bonus
        if !has_ortho && !has_diag {
            let dist = (x - enemy_x).abs() + (y - enemy_y).abs();
            if dist < 12 {
                short_range_bonus += ((12 - dist) * 4) as i32;
            }
        }
    }

    let total_sliders = ortho_count.max(diag_count);
    let few_sliders = total_sliders <= 1;

    bonus += short_range_bonus * if few_sliders { 2 } else { 1 };

    // ========== ORTHOGONAL FENCING ==========
    if ortho_count > 0 {
        // Vertical sandwich
        if ortho_y_min_above != i64::MAX && ortho_y_max_below != i64::MIN {
            let gap = ortho_y_min_above - ortho_y_max_below - 1;
            bonus += gap_bonus(gap);
        }
        // Horizontal sandwich
        if ortho_x_min_right != i64::MAX && ortho_x_max_left != i64::MIN {
            let gap = ortho_x_min_right - ortho_x_max_left - 1;
            bonus += gap_bonus(gap);
        }

        // Fence closeness
        if ortho_y_min_above != i64::MAX {
            bonus += distance_bonus(ortho_y_min_above - enemy_y);
        }
        if ortho_y_max_below != i64::MIN {
            bonus += distance_bonus(enemy_y - ortho_y_max_below);
        }
        if ortho_x_min_right != i64::MAX {
            bonus += distance_bonus(ortho_x_min_right - enemy_x);
        }
        if ortho_x_max_left != i64::MIN {
            bonus += distance_bonus(enemy_x - ortho_x_max_left);
        }

        // Cutting off escape
        let our_dx = our_king.x - enemy_x;
        let our_dy = our_king.y - enemy_y;

        if our_dx > 0 && ortho_x_max_left != i64::MIN {
            let cut = enemy_x - ortho_x_max_left;
            bonus += 250 + if cut <= 3 { (4 - cut) as i32 * 60 } else { 0 };
        } else if our_dx < 0 && ortho_x_min_right != i64::MAX {
            let cut = ortho_x_min_right - enemy_x;
            bonus += 250 + if cut <= 3 { (4 - cut) as i32 * 60 } else { 0 };
        }

        if our_dy > 0 && ortho_y_max_below != i64::MIN {
            let cut = enemy_y - ortho_y_max_below;
            bonus += 250 + if cut <= 3 { (4 - cut) as i32 * 60 } else { 0 };
        } else if our_dy < 0 && ortho_y_min_above != i64::MAX {
            let cut = ortho_y_min_above - enemy_y;
            bonus += 250 + if cut <= 3 { (4 - cut) as i32 * 60 } else { 0 };
        }

        if ortho_count >= 2 {
            bonus += 240;
        }
    }

    // ========== DIAGONAL FENCING ==========
    if diag_count > 0 {
        if diag_pos_min_above != i64::MAX && diag_pos_max_below != i64::MIN {
            let gap = diag_pos_min_above - diag_pos_max_below - 2;
            bonus += gap_bonus(gap);
        }
        if diag_neg_min_above != i64::MAX && diag_neg_max_below != i64::MIN {
            let gap = diag_neg_min_above - diag_neg_max_below - 2;
            bonus += gap_bonus(gap);
        }

        if diag_pos_min_above != i64::MAX {
            bonus += distance_bonus(diag_pos_min_above - enemy_diag_pos);
        }
        if diag_pos_max_below != i64::MIN {
            bonus += distance_bonus(enemy_diag_pos - diag_pos_max_below);
        }
        if diag_neg_min_above != i64::MAX {
            bonus += distance_bonus(diag_neg_min_above - enemy_diag_neg);
        }
        if diag_neg_max_below != i64::MIN {
            bonus += distance_bonus(enemy_diag_neg - diag_neg_max_below);
        }

        let our_dx = our_king.x - enemy_x;
        let our_dy = our_king.y - enemy_y;
        let diag_pos_dir = our_dx + our_dy;
        let diag_neg_dir = our_dx - our_dy;

        if diag_pos_dir > 0 && diag_pos_max_below != i64::MIN {
            let cut = enemy_diag_pos - diag_pos_max_below;
            bonus += 200 + if cut <= 4 { (5 - cut) as i32 * 50 } else { 0 };
        } else if diag_pos_dir < 0 && diag_pos_min_above != i64::MAX {
            let cut = diag_pos_min_above - enemy_diag_pos;
            bonus += 200 + if cut <= 4 { (5 - cut) as i32 * 50 } else { 0 };
        }

        if diag_neg_dir > 0 && diag_neg_max_below != i64::MIN {
            let cut = enemy_diag_neg - diag_neg_max_below;
            bonus += 200 + if cut <= 4 { (5 - cut) as i32 * 50 } else { 0 };
        } else if diag_neg_dir < 0 && diag_neg_min_above != i64::MAX {
            let cut = diag_neg_min_above - enemy_diag_neg;
            bonus += 200 + if cut <= 4 { (5 - cut) as i32 * 50 } else { 0 };
        }

        if diag_count >= 2 {
            bonus += 240;
        }
    }

    // ========== KING INVOLVEMENT ==========
    let king_dist = (our_king.x - enemy_x).abs() + (our_king.y - enemy_y).abs();

    if total_sliders < 3 {
        let prox = (30 - king_dist.min(30)) as i32;
        bonus += prox * 45;

        let dx = (our_king.x - enemy_x).abs();
        let dy = (our_king.y - enemy_y).abs();

        if (dx == 2 && dy == 0) || (dx == 0 && dy == 2) {
            bonus += 280;
        }
        if dx == 2 && dy == 2 {
            bonus += 200;
        }
        if dx <= 2 && dy <= 2 {
            bonus += 180;
        }
        if dx <= 1 && dy <= 1 && dx + dy > 0 {
            bonus += 120;
        }
    } else {
        bonus += ((15 - king_dist.min(15)) * 3) as i32;
    }

    if total_sliders >= 2 {
        bonus += 100;
    }
    if total_sliders >= 3 {
        bonus += 150;
    }
    if ortho_count >= 1 && diag_count >= 1 {
        bonus += 80;
    }

    bonus
}

// ==================== Helper Functions ====================

#[inline(always)]
fn gap_bonus(gap: i64) -> i32 {
    if gap <= 1 {
        650
    } else if gap <= 2 {
        500
    } else if gap <= 3 {
        380
    } else if gap <= 5 {
        280
    } else if gap <= 8 {
        180
    } else {
        100
    }
}

#[inline(always)]
fn distance_bonus(dist: i64) -> i32 {
    if dist <= 1 {
        220
    } else if dist <= 2 {
        160
    } else if dist <= 4 {
        110
    } else if dist <= 6 {
        60
    } else {
        25
    }
}

/// Determine if king is needed for mate based on material
#[inline(always)]
pub fn needs_king_for_mate(board: &Board, color: PlayerColor) -> bool {
    let mut queens: u8 = 0;
    let mut rooks: u8 = 0;
    let mut bishops: u8 = 0;
    let mut knights: u8 = 0;
    let mut chancellors: u8 = 0;
    let mut archbishops: u8 = 0;
    let mut hawks: u8 = 0;
    let mut guards: u8 = 0;

    let is_white = color == PlayerColor::White;
    for (_, _, piece) in board.iter_pieces_by_color(is_white) {
        match piece.piece_type() {
            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => bishops += 1,
            PieceType::Knight => knights += 1,
            PieceType::Chancellor => chancellors += 1,
            PieceType::Archbishop => archbishops += 1,
            PieceType::Hawk => hawks += 1,
            PieceType::Guard => guards += 1,
            _ => {}
        }
        // Quick exits for common cases
        if queens >= 2 {
            return false;
        }
        if rooks >= 3 {
            return false;
        }
    }

    // Strong material combinations that don't need king
    if chancellors >= 2 {
        return false;
    }
    if archbishops >= 3 {
        return false;
    }
    if hawks >= 4 {
        return false;
    }
    if bishops >= 6 {
        return false;
    }
    if queens >= 1 && chancellors >= 1 {
        return false;
    }
    if queens >= 1 && bishops >= 2 {
        return false;
    }
    if queens >= 1 && knights >= 2 {
        return false;
    }
    if queens >= 1 && guards >= 2 {
        return false;
    }
    if queens >= 1 && rooks >= 1 && (bishops >= 1 || knights >= 1) {
        return false;
    }
    if chancellors >= 1 && bishops >= 2 {
        return false;
    }
    if rooks >= 2 && (bishops >= 2 || knights >= 2 || guards >= 1) {
        return false;
    }
    if rooks >= 1 && bishops >= 3 {
        return false;
    }
    if rooks >= 1 && knights >= 4 {
        return false;
    }
    if rooks >= 1 && guards >= 2 {
        return false;
    }

    true
}
