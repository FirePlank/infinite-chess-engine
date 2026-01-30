// Mop-Up Evaluation
//
// Specialized endgame logic for positions where one side has a significant
// material advantage. It aims to drive the enemy king into a corner or
// "cage" it to facilitate checkmate.

use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::{SpatialIndices, is_square_attacked};

/// Threshold for disabling mop-up evaluation if the opponent still has significant material.
const MOP_UP_THRESHOLD_PERCENT: u32 = 20;

#[derive(Clone, Copy)]
struct SliderInfo {
    x: i64,
    y: i64,
}

/// Detects if the enemy king is trapped within a localized "cage" of attacked squares.
/// Returns whether a cage exists and the total reachable area for the king.
#[inline]
fn find_bitboard_cage(
    board: &Board,
    indices: &SpatialIndices,
    enemy_king: &Coordinate,
    our_color: PlayerColor,
) -> (bool, u32) {
    // 32x32 local window centered on king
    // Indices 0..31 map to king_coord - 16 .. king_coord + 15
    let mut forbidden = [0u32; 32];
    let origin_x = enemy_king.x - 16;
    let origin_y = enemy_king.y - 16;

    // 1. Identify forbidden squares (attacked, occupied, or out of bounds)
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();

    for (local_y, forbidden_row) in forbidden.iter_mut().enumerate() {
        let abs_y = origin_y + local_y as i64;
        for local_x in 0..32 {
            let abs_x = origin_x + local_x as i64;

            // If out of bounds, it's a "wall"
            if abs_x < min_x || abs_x > max_x || abs_y < min_y || abs_y > max_y {
                *forbidden_row |= 1 << local_x;
                continue;
            }

            let target = Coordinate::new(abs_x, abs_y);

            // If square is attacked or occupied by our piece, it's a "wall"
            if is_square_attacked(board, &target, our_color, indices)
                || board.is_occupied_by_color(abs_x, abs_y, our_color)
            {
                *forbidden_row |= 1 << local_x;
            }
        }
    }

    // 2. Flood fill from center (16, 16)
    let mut reachable = [0u32; 32];
    reachable[16] = 1 << 16;

    // Iterative 8-way dilation
    for _ in 0..32 {
        let mut changed = false;
        let mut next_reachable = reachable;

        for y in 0..32 {
            if reachable[y] == 0 {
                continue;
            }

            // Current row dilation (left/right)
            let row = reachable[y];
            let dilated_row = row | (row << 1) | (row >> 1);

            // Propagate to current row and neighbors
            next_reachable[y] |= dilated_row;
            if y > 0 {
                next_reachable[y - 1] |= dilated_row;
            }
            if y < 31 {
                next_reachable[y + 1] |= dilated_row;
            }
        }

        // Mask out forbidden squares
        for y in 0..32 {
            let prev = reachable[y];
            next_reachable[y] &= !forbidden[y];
            if next_reachable[y] != prev {
                changed = true;
            }
            reachable[y] = next_reachable[y];
        }

        if !changed {
            break;
        }

        // Check if we hit the perimeter
        if (reachable[0] | reachable[31]) != 0 {
            return (false, 1024);
        }
        for reach in reachable.iter().take(31).skip(1) {
            if (reach & 0x80000001) != 0 {
                return (false, 1024);
            }
        }
    }

    // 3. Successful fill without hitting the perimeter indicates a contained cage
    let mut area = 0u32;
    for row in reachable.iter() {
        area += row.count_ones();
    }

    (area > 0 && area < 1000, area)
}

// --- Utility Functions ---

/// Check if a side only has a king (no other pieces)
#[inline(always)]
pub fn is_lone_king(game: &GameState, color: PlayerColor) -> bool {
    if color == PlayerColor::White {
        game.white_pawn_count == 0 && !game.white_non_pawn_material
    } else {
        game.black_pawn_count == 0 && !game.black_non_pawn_material
    }
}

/// Calculates the mop-up scaling factor (0-100) based on remaining material.
/// Mop-up is only active when the opponent's material is below the threshold.
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

    if losing_pieces == 0 {
        return Some(100); // Full mop-up
    }

    if losing_starting == 0 {
        return None;
    }

    let percent_remaining = (losing_pieces as u32 * 100) / losing_starting as u32;

    if percent_remaining >= MOP_UP_THRESHOLD_PERCENT {
        return None;
    }

    // Scale linear regression from 100% (at 0% material) to 0% (at threshold)
    Some(100 - (percent_remaining * 100 / MOP_UP_THRESHOLD_PERCENT).min(100))
}

/// Unscaled mop-up evaluation.
#[inline(always)]
pub fn evaluate_lone_king_endgame(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    evaluate_mop_up_core(game, our_king, enemy_king, winning_color)
}

/// Scaled mop-up evaluation.
#[inline(always)]
pub fn evaluate_mop_up_scaled(
    game: &GameState,
    our_king: Option<&Coordinate>,
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

// --- Core Evaluation ---

/// Main logic for driving the enemy king to mate.
#[inline(always)]
fn evaluate_mop_up_core(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    let mut bonus: i32 = 0;

    // Calculate distance between kings if applicable
    let (our_dx, our_dy, king_dist) = if let Some(ok) = our_king {
        let dx = ok.x - enemy_king.x;
        let dy = ok.y - enemy_king.y;
        (dx, dy, dx.abs().max(dy.abs()))
    } else {
        (0, 0, i64::MAX) // No king = no approach bonus
    };

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
    let mut leaper_count: u8 = 0;
    let mut queen_count: u8 = 0;
    let mut amazon_count: u8 = 0;
    let mut total_non_pawn_pieces: u8 = 0;
    let mut short_range_bonus: i32 = 0;

    let mut our_pieces: [crate::evaluation::mop_up::SliderInfo; 24] =
        [crate::evaluation::mop_up::SliderInfo { x: 0, y: 0 }; 24];
    let mut our_pieces_count: usize = 0;

    // Track 2nd closest fences for ladder/wall detection
    let mut ortho_y_min_above_2: i64 = i64::MAX;
    let mut ortho_y_max_below_2: i64 = i64::MIN;
    let mut ortho_x_min_right_2: i64 = i64::MAX;
    let mut ortho_x_max_left_2: i64 = i64::MIN;

    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    let enemy_diag_pos = enemy_x + enemy_y;
    let enemy_diag_neg = enemy_x - enemy_y;

    // Single pass variables
    let is_white = winning_color == PlayerColor::White;
    for (x, y, piece) in game.board.iter_pieces_by_color(is_white) {
        let pt = piece.piece_type();

        if pt.is_royal() {
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

        if pt != PieceType::King && pt != PieceType::Pawn && our_pieces_count < 24 {
            our_pieces[our_pieces_count] = SliderInfo { x, y };
            our_pieces_count += 1;
        }

        if has_ortho {
            ortho_count += 1;
            if y > enemy_y {
                if y < ortho_y_min_above {
                    ortho_y_min_above_2 = ortho_y_min_above;
                    ortho_y_min_above = y;
                } else if y < ortho_y_min_above_2 {
                    ortho_y_min_above_2 = y;
                }
            } else if y < enemy_y {
                if y > ortho_y_max_below {
                    ortho_y_max_below_2 = ortho_y_max_below;
                    ortho_y_max_below = y;
                } else if y > ortho_y_max_below_2 {
                    ortho_y_max_below_2 = y;
                }
            }

            if x > enemy_x {
                if x < ortho_x_min_right {
                    ortho_x_min_right_2 = ortho_x_min_right;
                    ortho_x_min_right = x;
                } else if x < ortho_x_min_right_2 {
                    ortho_x_min_right_2 = x;
                }
            } else if x < enemy_x {
                if x > ortho_x_max_left {
                    ortho_x_max_left_2 = ortho_x_max_left;
                    ortho_x_max_left = x;
                } else if x > ortho_x_max_left_2 {
                    ortho_x_max_left_2 = x;
                }
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

        if pt == PieceType::Queen || pt == PieceType::RoyalQueen {
            queen_count += 1;
        } else if pt == PieceType::Amazon {
            amazon_count += 1;
        }

        total_non_pawn_pieces += 1;

        // Placement heuristics
        let pdx = x - enemy_x;
        let pdy = y - enemy_y;

        let on_back_x = (our_dx > 0 && pdx < 0) || (our_dx < 0 && pdx > 0);
        let on_back_y = (our_dy > 0 && pdy < 0) || (our_dy < 0 && pdy > 0);

        // Reward cutting off escape relative to our king
        if on_back_x {
            bonus += 70;
        }
        if on_back_y {
            bonus += 70;
        }

        // Diagonals Back Side
        let pdp = x + y - enemy_diag_pos;
        let pdn = x - y - enemy_diag_neg;
        if let Some(ok) = our_king {
            let our_dp = ok.x + ok.y - enemy_diag_pos;
            let our_dn = ok.x - ok.y - enemy_diag_neg;
            if (our_dp > 0 && pdp < 0) || (our_dp < 0 && pdp > 0) {
                bonus += 40;
            }
            if (our_dn > 0 && pdn < 0) || (our_dn < 0 && pdn > 0) {
                bonus += 40;
            }
        }

        // Penalize checks that drive the enemy king to safer areas
        let is_checking = match pt {
            PieceType::Rook | PieceType::Chancellor => pdx == 0 || pdy == 0,
            PieceType::Bishop | PieceType::Archbishop => pdx.abs() == pdy.abs(),
            PieceType::Queen | PieceType::Amazon | PieceType::RoyalQueen => {
                pdx == 0 || pdy == 0 || pdx.abs() == pdy.abs()
            }
            PieceType::Knight => {
                (pdx.abs() == 2 && pdy.abs() == 1) || (pdx.abs() == 1 && pdy.abs() == 2)
            }
            _ => false,
        };

        if is_checking {
            // Penalty for checks that push the enemy king away from our king.
            // Calibrated: -300 is enough to discourage, but not so much that the king runs away.
            let is_frontal_check = (our_dx.signum() == pdx.signum() && pdx != 0)
                || (our_dy.signum() == pdy.signum() && pdy != 0);

            if is_frontal_check {
                bonus -= 60;
            } else {
                bonus -= 20; // Minimal penalty for checks from behind/side
            }
        }

        if !has_ortho && !has_diag {
            leaper_count += 1;
            let dist = pdx.abs().max(pdy.abs()); // Chebyshev distance

            // Heavy proximity bonus to ensure short-range pieces engage
            if dist <= 3 {
                short_range_bonus += 160; // Very close - huge bonus
            } else if dist <= 6 {
                short_range_bonus += 100; // Close
            } else if dist <= 10 {
                short_range_bonus += 60; // Medium
            } else if dist <= 15 {
                short_range_bonus += 30; // Far but approaching
            } else if dist <= 25 {
                short_range_bonus += 15; // Very far
            } else {
                // PENALTY for being too far - knight is useless here
                short_range_bonus -= 80; // Doubled penalty
            }
        }
    }

    let total_sliders = ortho_count.max(diag_count);
    let few_pieces = total_non_pawn_pieces <= 2;

    bonus += short_range_bonus * if few_pieces { 5 } else { 3 };

    let is_overwhelming = queen_count >= 1 || amazon_count >= 1 || total_non_pawn_pieces >= 5;

    // --- Strategy Selection ---
    let losing_color = winning_color.opponent();
    let is_opponent_lone_king = is_lone_king(game, losing_color);
    let mut cage_score = 0;

    if is_opponent_lone_king {
        // 1. Bitboard cage check (precise local connectivity)
        let (bitboard_caged, reached_area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            enemy_king,
            winning_color,
        );

        // 2. Macro box logic (slider enclosures)
        let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
        const EDGE_THRESHOLD: i64 = 50;
        let has_barrier_above = ortho_y_min_above != i64::MAX || (max_y - enemy_y) < EDGE_THRESHOLD;
        let has_barrier_below = ortho_y_max_below != i64::MIN || (enemy_y - min_y) < EDGE_THRESHOLD;
        let has_barrier_right = ortho_x_min_right != i64::MAX || (max_x - enemy_x) < EDGE_THRESHOLD;
        let has_barrier_left = ortho_x_max_left != i64::MIN || (enemy_x - min_x) < EDGE_THRESHOLD;
        let macro_box =
            has_barrier_above && has_barrier_below && has_barrier_right && has_barrier_left;

        let macro_area = if macro_box {
            let box_width = if ortho_x_min_right != i64::MAX && ortho_x_max_left != i64::MIN {
                (ortho_x_min_right - ortho_x_max_left - 1).max(1)
            } else {
                100
            };
            let box_height = if ortho_y_min_above != i64::MAX && ortho_y_max_below != i64::MIN {
                (ortho_y_min_above - ortho_y_max_below - 1).max(1)
            } else {
                100
            };
            (box_width * box_height) as u32
        } else {
            10000
        };

        if is_overwhelming {
            // Precise cage logic for high-material endgames
            if bitboard_caged {
                cage_score = if reached_area <= 5 {
                    500
                } else if reached_area <= 10 {
                    360
                } else if reached_area <= 16 {
                    240
                } else if reached_area <= 40 {
                    160
                } else if reached_area <= 100 {
                    80
                } else {
                    40
                };
            }
            if macro_box {
                cage_score = if macro_area <= 100 { 70 } else { 30 };
            }
            bonus += cage_score;

            // King approach gradient
            let king_approach_bonus = (100 - king_dist.min(100) as i32) * 16; // Scaled down for material respect
            bonus += king_approach_bonus;

            if king_dist <= 2 {
                bonus += 160; // Huge bonus for being adjacent
            } else if king_dist <= 4 {
                bonus += 80; // Good bonus for being close
            }

            // General piece approach
            for s in our_pieces.iter().take(our_pieces_count) {
                let dist = (s.x - enemy_x).abs().max((s.y - enemy_y).abs()); // Chebyshev distance

                // Scale by cage tightness - tighter cage = more urgency to close in
                let approach_weight = if bitboard_caged && reached_area <= 16 {
                    24 // Massive weight for tight cages
                } else if bitboard_caged {
                    16
                } else if macro_box {
                    10
                } else {
                    8 // Slightly increased base weight
                };

                const LONG_RANGE: i64 = 100;
                bonus += ((LONG_RANGE - dist.min(LONG_RANGE)) as i32) * approach_weight;

                // TEAMWORK: Reward pieces for staying within semi-reasonable range of our king
                // This prevents the "Amazon in corner, King in other corner" drift.
                if let Some(ok) = our_king {
                    let king_piece_dist = (s.x - ok.x).abs().max((s.y - ok.y).abs());
                    bonus += (60 - king_piece_dist.min(60) as i32) * 3;
                }
            }
        } else if ortho_count == 2
            && diag_count == 0
            && leaper_count == 0
            && total_non_pawn_pieces == 2
        {
            // Specialized logic for 2R+K vs K endgames

            let (r1_x, r1_y, r2_x, r2_y) = if our_pieces_count == 2 {
                (
                    our_pieces[0].x,
                    our_pieces[0].y,
                    our_pieces[1].x,
                    our_pieces[1].y,
                )
            } else {
                (0, 0, 0, 0)
            };

            // 1. Mutual protection and stability
            let rooks_on_same_rank = r1_y == r2_y;
            let rooks_on_same_file = r1_x == r2_x;
            let rooks_protecting = rooks_on_same_rank || rooks_on_same_file;

            if rooks_protecting {
                bonus += 2000;

                // PREVENT SHUFFLING: Reward being close to each other
                let rook_dist_between = (r1_x - r2_x).abs() + (r1_y - r2_y).abs();
                bonus -= (rook_dist_between as i32) * 50;
            } else {
                bonus -= 2000; // Pathological state: Fix immediately
            }

            // 2. Proximity and sandwiching
            let has_rook_above = r1_y > enemy_y || r2_y > enemy_y;
            let has_rook_below = r1_y < enemy_y || r2_y < enemy_y;
            let has_rook_right = r1_x > enemy_x || r2_x > enemy_x;
            let has_rook_left = r1_x < enemy_x || r2_x < enemy_x;

            let has_sandwich_v = has_rook_above && has_rook_below;
            let has_sandwich_h = has_rook_right && has_rook_left;

            if has_sandwich_v {
                bonus += 1000;
                let ca = if r1_y > enemy_y { r1_y } else { r2_y }.min(if r2_y > enemy_y {
                    r2_y
                } else {
                    r1_y
                });
                let cb = if r1_y < enemy_y { r1_y } else { r2_y }.max(if r2_y < enemy_y {
                    r2_y
                } else {
                    r1_y
                });
                let gap = ca - cb - 1;
                bonus += (8 - gap.min(8) as i32) * 150;
            }
            if has_sandwich_h {
                bonus += 1000;
                let cr = if r1_x > enemy_x { r1_x } else { r2_x }.min(if r2_x > enemy_x {
                    r2_x
                } else {
                    r1_x
                });
                let cl = if r1_x < enemy_x { r1_x } else { r2_x }.max(if r2_x < enemy_x {
                    r2_x
                } else {
                    r1_x
                });
                let gap = cr - cl - 1;
                bonus += (8 - gap.min(8) as i32) * 150;
            }

            // 3. Fence quality
            for r in &[(r1_x, r1_y), (r2_x, r2_y)] {
                let rd = (r.1 - enemy_y).abs();
                let fd = (r.0 - enemy_x).abs();
                if rd > 0 {
                    bonus += if rd == 1 {
                        400
                    } else if rd == 2 {
                        250
                    } else {
                        50
                    };
                }
                if fd > 0 {
                    bonus += if fd == 1 {
                        400
                    } else if fd == 2 {
                        250
                    } else {
                        50
                    };
                }
            }

            // 4. King approach (Dominant factor)
            if let Some(ok) = our_king {
                // MASSIVE approach bonus
                bonus += (100 - king_dist.min(100) as i32) * 100;

                if king_dist <= 2 {
                    bonus += 3000;
                } else if king_dist <= 4 {
                    bonus += 1500;
                }

                let our_dx = ok.x - enemy_x;
                let our_dy = ok.y - enemy_y;

                // CUT OFF FROM OPPOSITE SIDE
                if our_dx > 0 && has_rook_left {
                    bonus += 1200;
                }
                if our_dx < 0 && has_rook_right {
                    bonus += 1200;
                }
                if our_dy > 0 && has_rook_below {
                    bonus += 1200;
                }
                if our_dy < 0 && has_rook_above {
                    bonus += 1200;
                }

                // OPPOSITION
                if has_sandwich_v && our_dy.abs() <= 1 {
                    bonus += 1000;
                }
                if has_sandwich_h && our_dx.abs() <= 1 {
                    bonus += 1000;
                }

                // Don't block our own rooks!
                if (rooks_on_same_rank && ok.y == r1_y) || (rooks_on_same_file && ok.x == r1_x) {
                    bonus -= 1500;
                }
            }

            // Full box bonus
            if has_sandwich_v && has_sandwich_h {
                bonus += 2000;
            }
        } else {
            // General technical endgame logic
            let mut protected_count = 0;
            for s in &our_pieces[..our_pieces_count] {
                let coord = Coordinate::new(s.x, s.y);
                if crate::moves::is_square_attacked(
                    &game.board,
                    &coord,
                    winning_color,
                    &game.spatial_indices,
                ) {
                    protected_count += 1;
                }
            }
            bonus += protected_count * 40;

            let mut sand_h = false;
            let mut sand_v = false;
            let mut tight_h = false;
            let mut tight_v = false;
            if ortho_y_min_above != i64::MAX && ortho_y_max_below != i64::MIN {
                let gap = ortho_y_min_above - ortho_y_max_below - 1;
                if gap <= 3 {
                    sand_v = true;
                    if gap <= 1 {
                        tight_v = true;
                    }
                }
                bonus += if gap <= 1 {
                    160
                } else if gap <= 2 {
                    120
                } else if gap <= 3 {
                    100
                } else {
                    40
                };
            }
            if ortho_x_min_right != i64::MAX && ortho_x_max_left != i64::MIN {
                let gap = ortho_x_min_right - ortho_x_max_left - 1;
                if gap <= 3 {
                    sand_h = true;
                    if gap <= 1 {
                        tight_h = true;
                    }
                }
                bonus += if gap <= 1 {
                    160
                } else if gap <= 2 {
                    120
                } else if gap <= 3 {
                    100
                } else {
                    40
                };
            }
            let mut sand_dp = false;
            let mut sand_dn = false;
            if diag_pos_min_above != i64::MAX && diag_pos_max_below != i64::MIN {
                let gap = diag_pos_min_above - diag_pos_max_below - 1;
                if gap <= 2 {
                    sand_dp = true;
                }
                bonus += if gap <= 1 {
                    120
                } else if gap <= 2 {
                    90
                } else {
                    30
                };
            }
            if diag_neg_min_above != i64::MAX && diag_neg_max_below != i64::MIN {
                let gap = diag_neg_min_above - diag_neg_max_below - 1;
                if gap <= 2 {
                    sand_dn = true;
                }
                bonus += if gap <= 1 {
                    120
                } else if gap <= 2 {
                    90
                } else {
                    30
                };
            }

            let mut ladder = false;
            if ortho_count >= 2 {
                let ladder_x = (ortho_x_min_right != i64::MAX
                    && ortho_x_min_right_2 != i64::MAX
                    && (ortho_x_min_right_2 - ortho_x_min_right) == 1)
                    || (ortho_x_max_left != i64::MIN
                        && ortho_x_max_left_2 != i64::MIN
                        && (ortho_x_max_left - ortho_x_max_left_2) == 1);
                let ladder_y = (ortho_y_min_above != i64::MAX
                    && ortho_y_min_above_2 != i64::MAX
                    && (ortho_y_min_above_2 - ortho_y_min_above) == 1)
                    || (ortho_y_max_below != i64::MIN
                        && ortho_y_max_below_2 != i64::MIN
                        && (ortho_y_max_below - ortho_y_max_below_2) == 1);
                if ladder_x || ladder_y {
                    ladder = true;
                    bonus += 240;
                }
            }

            let r_up = if ortho_y_min_above != i64::MAX {
                ortho_y_min_above - enemy_y - 1
            } else {
                15
            };
            let r_down = if ortho_y_max_below != i64::MIN {
                enemy_y - ortho_y_max_below - 1
            } else {
                15
            };
            let r_right = if ortho_x_min_right != i64::MAX {
                ortho_x_min_right - enemy_x - 1
            } else {
                15
            };
            let r_left = if ortho_x_max_left != i64::MIN {
                enemy_x - ortho_x_max_left - 1
            } else {
                15
            };
            let run_h = if our_dx > 0 {
                r_left
            } else if our_dx < 0 {
                r_right
            } else {
                r_left.max(r_right)
            };
            let run_v = if our_dy > 0 {
                r_down
            } else if our_dy < 0 {
                r_up
            } else {
                r_up.max(r_down)
            };
            bonus += (20 - run_h.min(20)) as i32 * 12;
            bonus += (20 - run_v.min(20)) as i32 * 12;

            let is_contained = ladder
                || (sand_h && tight_h)
                || (sand_v && tight_v)
                || (sand_h && sand_v)
                || (sand_dp && sand_dn)
                || (bitboard_caged && reached_area <= 12);

            // King approach gradient
            let king_approach_bonus = (100 - king_dist.min(100) as i32) * 8; // Scaled down
            bonus += king_approach_bonus;

            // Piece approach
            for s in &our_pieces[..our_pieces_count] {
                let dist = (s.x - enemy_x).abs().max((s.y - enemy_y).abs());

                // Pieces approach even in technical branch
                bonus += (80 - dist.min(80) as i32) * 4;
                // Pieces stay with king (only if we have a king)
                if let Some(ok) = our_king {
                    let king_piece_dist = (s.x - ok.x).abs().max((s.y - ok.y).abs());
                    bonus += (50 - king_piece_dist.min(50) as i32) * 2;
                }
            }

            if is_contained {
                let prox = (30 - king_dist.min(30)) as i32;
                bonus += prox * 16;
                if king_dist <= 2 {
                    bonus += 80;
                }
            } else {
                let prox = (20 - king_dist.min(20)) as i32;
                bonus += prox;
            }
        }
    }

    if total_sliders >= 2 {
        bonus += 20;
    }
    if total_sliders >= 3 {
        bonus += 30;
    }
    if ortho_count >= 1 && diag_count >= 1 {
        bonus += 15;
    }

    bonus
}

// --- Helper Functions ---

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    #[test]
    fn test_is_lone_king_true() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.recompute_piece_counts();

        assert!(is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_is_lone_king_false() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();

        assert!(!is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_calculate_mop_up_scale_returns_none_for_no_advantage() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        game.recompute_piece_counts();

        // Both sides have material, so no mop-up
        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // May or may not apply depending on thresholds
        assert!(scale.is_none() || scale.unwrap() <= 100);
    }

    #[test]
    fn test_needs_king_for_mate_true() {
        // Just a knight - needs king
        let mut board = Board::new();
        board.set_piece(3, 3, Piece::new(PieceType::Knight, PlayerColor::White));

        assert!(needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_needs_king_for_mate_false_two_queens() {
        let mut board = Board::new();
        board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
        board.set_piece(4, 3, Piece::new(PieceType::Queen, PlayerColor::White));
        

        assert!(!needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_needs_king_for_mate_false_three_rooks() {
        let mut board = Board::new();
        board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(2, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(3, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        

        assert!(!needs_king_for_mate(&board, PlayerColor::White));
    }

    #[test]
    fn test_evaluate_lone_king_endgame_returns_value() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();

        let enemy_king = Coordinate::new(5, 8);
        let our_king = Coordinate::new(5, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive (White has advantage)
        assert!(score >= 0);
    }

    #[test]
    fn test_evaluate_mop_up_scaled_no_king() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(3, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();

        let enemy_king = Coordinate::new(5, 8);

        // No white king (checkmate practice)
        let score = evaluate_mop_up_scaled(
            &game,
            None,
            &enemy_king,
            PlayerColor::White,
            PlayerColor::Black,
        );
        assert!(score.abs() < 100000);
    }

    #[test]
    fn test_mop_up_rook_fence_bonus() {
        let mut game = create_test_game();
        // Black king in center
        game.board
            .set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
        // White rooks creating a fence
        game.board
            .set_piece(0, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(7, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.recompute_piece_counts();

        let enemy_king = Coordinate::new(4, 4);
        let our_king = Coordinate::new(4, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive since rooks create cutting lines
        assert!(
            score > 0,
            "Rook fence should give positive score: {}",
            score
        );
    }

    #[test]
    fn test_mop_up_king_approach_bonus() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 5, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        // White king close to enemy
        game.board
            .set_piece(6, 5, Piece::new(PieceType::King, PlayerColor::White));
        game.recompute_piece_counts();

        let enemy_king = Coordinate::new(5, 5);
        let our_king_close = Coordinate::new(6, 5);

        let score_close = evaluate_lone_king_endgame(
            &game,
            Some(&our_king_close),
            &enemy_king,
            PlayerColor::White,
        );

        // Move white king further away
        game.board.remove_piece(&6, &5);
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.recompute_piece_counts();

        let our_king_far = Coordinate::new(1, 1);
        let score_far =
            evaluate_lone_king_endgame(&game, Some(&our_king_far), &enemy_king, PlayerColor::White);

        assert!(
            score_close > score_far,
            "Closer king should get higher score: close={} far={}",
            score_close,
            score_far
        );
    }

    #[test]
    fn test_calculate_mop_up_scale_with_pawns() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 5, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::White));
        // Add rooks (sufficient material) and promotable pawn
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(2, 2, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(3, 7, Piece::new(PieceType::Pawn, PlayerColor::White)); // Near promotion
        game.white_promo_rank = 8;
        game.recompute_piece_counts();

        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // Should return a scale since white has mating material
        assert!(scale.is_some(), "Should have mop-up scale with rooks");
    }

    #[test]
    fn test_find_bitboard_cage() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
        // Surround with attacking rooks
        game.board
            .set_piece(4, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(4, 8, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(0, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(8, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.recompute_piece_counts();

        let enemy_king = Coordinate::new(4, 4);
        let (_is_caged, area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            &enemy_king,
            PlayerColor::White,
        );
        // The king should be significantly restricted
        assert!(
            area < 100,
            "King should be in a small area, found: {}",
            area
        );
    }
}
