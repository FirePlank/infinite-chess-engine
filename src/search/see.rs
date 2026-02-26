use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;

/// Tests if SEE value of move is >= threshold.
/// Uses early cutoffs to avoid full SEE calculation when possible.
#[inline(always)]
pub(crate) fn see_ge(game: &GameState, m: &Move, threshold: i32) -> bool {
    // BITBOARD: Fast piece check
    let captured = match game.board.get_piece(m.to.x, m.to.y) {
        Some(p) => p,
        None => return 0 >= threshold, // No capture: SEE = 0
    };

    let victim_val = get_piece_value(captured.piece_type());
    let attacker_val = get_piece_value(m.piece.piece_type());

    // Early cutoff 1: if capturing loses material even if undefended, fail
    let swap = victim_val - threshold;
    if swap < 0 {
        return false;
    }

    // Early cutoff 2: if capturing wins material even if we lose attacker, pass
    let swap = attacker_val - swap;
    if swap <= 0 {
        return true;
    }

    // Need full SEE for complex cases
    static_exchange_eval_impl(game, m) >= threshold
}

/// Static Exchange Evaluation implementation for a capture move on a single square.
///
/// Returns the net material gain (in centipawns) for the side to move if both
/// sides optimally capture/recapture on the destination square of `m`.
pub(crate) fn static_exchange_eval_impl(game: &GameState, m: &Move) -> i32 {
    let captured = match game.board.get_piece(m.to.x, m.to.y) {
        Some(p) => p,
        None => return 0,
    };

    let target_x = m.to.x;
    let target_y = m.to.y;

    #[derive(Clone, Copy, Debug)]
    struct Attacker {
        value: i32,
        color: PlayerColor,
        pos: Coordinate,
        ray_idx: Option<usize>,
    }

    // 1. Initial State
    let mut gain: [i32; 32] = [0; 32];
    let mut depth = 1;
    gain[0] = get_piece_value(captured.piece_type());

    let mut side = game.turn;
    let mut occ_val = get_piece_value(m.piece.piece_type());

    // 2. Active Attacker Collection
    // We use a SmallVec for the active attackers (those we've already found)
    let mut attackers: smallvec::SmallVec<[Attacker; 32]> = smallvec::SmallVec::new();

    // Directions for 16-ray lazy discovery
    // 0-3 Ortho, 4-7 Diag, 8-15 Knightrider
    use crate::attacks::*;
    let ray_dirs: [(i64, i64); 16] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (1, 2),
        (1, -2),
        (2, 1),
        (2, -1),
        (-1, 2),
        (-1, -2),
        (-2, 1),
        (-2, -1),
    ];

    // A. 3x3 Neighborhood Bitboard Scan (Covers all pawns, knights, kings, and exotic leapers)
    // This is O(1) and covers most tactic-heavy areas.
    let neighborhood = game.board.get_neighborhood(target_x, target_y);
    let local_idx = crate::tiles::local_index(target_x, target_y);
    use crate::tiles::masks;

    for (n, maybe_tile) in neighborhood.iter().enumerate() {
        let Some(tile) = maybe_tile else { continue };
        let occ = tile.occ_all;
        if occ == 0 {
            continue;
        }

        let (tx, ty) = crate::tiles::tile_coords(target_x, target_y);
        let nx = tx + (n as i64 % 3) - 1;
        let ny = ty + (n as i64 / 3) - 1;

        let masks_to_check = [
            (masks::KNIGHT_MASKS[local_idx][n], KNIGHT_MASK),
            (masks::KING_MASKS[local_idx][n], KING_MASK),
            (masks::CAMEL_MASKS[local_idx][n], CAMEL_MASK),
            (masks::GIRAFFE_MASKS[local_idx][n], GIRAFFE_MASK),
            (masks::ZEBRA_MASKS[local_idx][n], ZEBRA_MASK),
            (masks::HAWK_MASKS[local_idx][n], HAWK_MASK),
        ];

        for (attack_mask, req_mask) in masks_to_check {
            let mut bits = occ & attack_mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = Piece::from_packed(tile.piece[i]);
                if matches_mask(p.piece_type(), req_mask) {
                    let pos = Coordinate::new(nx * 8 + (i % 8) as i64, ny * 8 + (i / 8) as i64);
                    if pos != m.from {
                        attackers.push(Attacker {
                            value: get_piece_value(p.piece_type()),
                            color: p.color(),
                            pos,
                            ray_idx: None,
                        });
                    }
                }
            }
        }

        // Special Case: Pawns (they attack differently for White vs Black)
        for (color, mask) in [
            (
                PlayerColor::White,
                masks::pawn_attacker_masks(true)[local_idx][n],
            ),
            (
                PlayerColor::Black,
                masks::pawn_attacker_masks(false)[local_idx][n],
            ),
        ] {
            let mut bits = (if color == PlayerColor::White {
                tile.occ_white
            } else {
                tile.occ_black
            }) & tile.occ_pawns
                & mask;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let pos = Coordinate::new(nx * 8 + (i % 8) as i64, ny * 8 + (i / 8) as i64);
                if pos != m.from {
                    attackers.push(Attacker {
                        value: get_piece_value(PieceType::Pawn),
                        color,
                        pos,
                        ray_idx: None,
                    });
                }
            }
        }
    }

    // B. Lazy Ray Discovery (Sliding Pieces + Distant Knights/Kings)
    // We only find the FIRST blocker on each ray.
    for r in 0..16 {
        let (dx, dy) = ray_dirs[r];
        let mut found_pos: Option<(i64, i64, Piece)> = None;

        if r < 8 {
            // Cardinal/Diagonal via SpatialIndices (Infinite range)
            found_pos = game
                .spatial_indices
                .find_first_blocker(target_x, target_y, dx, dy);
        } else if game.spatial_indices.has_knightrider[0] || game.spatial_indices.has_knightrider[1]
        {
            // Knightrider Rays (Step-based with Tile skipping)
            let mut k = 1;
            while k < 128 {
                // Practical limit for Infinite Chess SEE
                let x = target_x + dx * k;
                let y = target_y + dy * k;
                // Optimization: Tile boundary check
                if let Some(p) = game.board.get_piece(x, y) {
                    found_pos = Some((x, y, p));
                    break;
                }
                k += 1;
            }
        }

        if let Some((vx, vy, p)) = found_pos {
            let pos = Coordinate::new(vx, vy);
            if pos == m.from {
                continue;
            }

            let pt = p.piece_type();
            let dist = (vx - target_x).abs().max((vy - target_y).abs());

            let can_attack = if r < 4 {
                is_ortho_slider(pt) || (dist == 1 && attacks_like_king(pt))
            } else if r < 8 {
                is_diag_slider(pt) || (dist == 1 && attacks_like_king(pt))
            } else {
                pt == PieceType::Knightrider || (dist == 1 && attacks_like_knight(pt))
            };

            if can_attack {
                // Check if we already found this piece in the 3x3 local scan (to avoid double-counting)
                if dist > 8 || !attackers.iter().any(|a| a.pos == pos) {
                    attackers.push(Attacker {
                        value: get_piece_value(pt),
                        color: p.color(),
                        pos,
                        ray_idx: Some(r),
                    });
                }
            }
        }
    }

    // 3. Recapture Sequence Loop
    loop {
        side = side.opponent();
        if depth >= 32 {
            break;
        }

        let mut best_i: Option<usize> = None;
        let mut best_val = i32::MAX;

        for i in 0..attackers.len() {
            let a = &attackers[i];
            if a.color == side && a.value < best_val {
                best_val = a.value;
                best_i = Some(i);
            }
        }

        if let Some(i) = best_i {
            let chosen = attackers.swap_remove(i);
            gain[depth] = occ_val - gain[depth - 1];
            occ_val = best_val;

            // X-Ray Discovery!
            if let Some(r) = chosen.ray_idx {
                let (dx, dy) = ray_dirs[r];
                let mut next_blocker: Option<(i64, i64, Piece)> = None;

                if r < 8 {
                    next_blocker =
                        game.spatial_indices
                            .find_first_blocker(chosen.pos.x, chosen.pos.y, dx, dy);
                } else if game.spatial_indices.has_knightrider[0]
                    || game.spatial_indices.has_knightrider[1]
                {
                    let mut k = 1;
                    while k < 128 {
                        let nx = chosen.pos.x + dx * k;
                        let ny = chosen.pos.y + dy * k;
                        if let Some(np) = game.board.get_piece(nx, ny) {
                            next_blocker = Some((nx, ny, np));
                            break;
                        }
                        k += 1;
                    }
                }

                if let Some((nx, ny, np)) = next_blocker {
                    let npt = np.piece_type();
                    let can_xray = if r < 4 {
                        is_ortho_slider(npt)
                    } else if r < 8 {
                        is_diag_slider(npt)
                    } else {
                        npt == PieceType::Knightrider
                    };

                    if can_xray {
                        attackers.push(Attacker {
                            value: get_piece_value(npt),
                            color: np.color(),
                            pos: Coordinate::new(nx, ny),
                            ray_idx: Some(r),
                        });
                    }
                }
            }
            depth += 1;
        } else {
            break;
        }
    }

    // 4. Negamax to find optimal outcome
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -std::cmp::max(-gain[depth - 1], gain[depth]);
    }
    gain[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;
    use crate::moves::Move;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game
    }

    #[test]
    fn test_see_simple_pawn_takes_pawn() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        assert_eq!(see_val, 100, "Pawn takes pawn should yield 100 cp");
    }

    #[test]
    fn test_see_queen_takes_defended_pawn() {
        let mut game = create_test_game();
        // White queen takes black pawn defended by black pawn
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.board
            .set_piece(6, 6, Piece::new(PieceType::Pawn, PlayerColor::Black)); // Defends 5,5
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Queen, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        // Queen takes pawn (+100), then pawn takes queen (-1350), net = -1250
        assert!(
            see_val < 0,
            "Queen taking defended pawn should be negative: {}",
            see_val
        );
    }

    #[test]
    fn test_see_rook_takes_rook() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 1),
            Coordinate::new(4, 7),
            Piece::new(PieceType::Rook, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        assert_eq!(see_val, 650, "Rook takes rook should yield rook value");
    }

    #[test]
    fn test_see_ge_threshold_pass() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Queen, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        // Pawn takes queen = +1350, easily passes threshold 0
        assert!(see_ge(&game, &m, 0));
        assert!(see_ge(&game, &m, 1000));
    }

    #[test]
    fn test_see_ge_threshold_fail() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Queen, PlayerColor::White),
        );

        // Queen takes pawn = +100, but very high threshold should fail
        assert!(!see_ge(&game, &m, 500));
    }

    #[test]
    fn test_see_no_capture_returns_zero() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 5), // Empty square
            Piece::new(PieceType::Rook, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        assert_eq!(see_val, 0, "Non-capture should return 0");
    }

    #[test]
    fn test_see_knight_takes_bishop() {
        let mut game = create_test_game();
        game.board
            .set_piece(3, 3, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(4, 5, Piece::new(PieceType::Bishop, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let m = Move::new(
            Coordinate::new(3, 3),
            Coordinate::new(4, 5),
            Piece::new(PieceType::Knight, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        // Knight (250) takes bishop (450) = +450 (undefended)
        assert_eq!(see_val, 450, "Knight takes bishop should yield 450");
    }
}
