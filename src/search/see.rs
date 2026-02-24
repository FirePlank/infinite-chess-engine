use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;
use arrayvec::ArrayVec;

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
        piece_type: PieceType,
        color: PlayerColor,
        pos: Coordinate,
    }

    // 1. Gather sliders into sorted rays using SpatialIndices
    // Rays: 0-3 Ortho (R, B, L, U), 4-7 Diag (UR, DR, DL, UL), 8-15 Knightrider
    let mut rays: [ArrayVec<Attacker, 16>; 16] = Default::default();
    let indices = &game.spatial_indices;

    for (r, &(dx, dy)) in crate::attacks::ORTHO_DIRS
        .iter()
        .chain(crate::attacks::DIAG_DIRS.iter())
        .enumerate()
    {
        let line = if dx == 0 {
            indices.cols.get(&target_x)
        } else if dy == 0 {
            indices.rows.get(&target_y)
        } else if dx == dy {
            indices.diag1.get(&(target_x - target_y))
        } else {
            indices.diag2.get(&(target_x + target_y))
        };

        if let Some(spatial_line) = line {
            let target_coord = if dx == 0 { target_y } else { target_x };
            let direction = if dx == 0 { dy } else { dx };

            if direction > 0 {
                let start_idx = spatial_line.coords.partition_point(|&c| c <= target_coord);
                for i in start_idx..spatial_line.coords.len().min(start_idx + 16) {
                    let p = crate::board::Piece::from_packed(spatial_line.pieces[i]);
                    let px = spatial_line.coords[i];
                    let py = if dx == 0 {
                        px
                    } else if dy == 0 {
                        target_y
                    } else if dx == dy {
                        px - (target_x - target_y)
                    } else {
                        (target_x + target_y) - px
                    };
                    // Actually, if dx == 0, the coord is y.
                    let (final_px, final_py) = if dx == 0 { (target_x, px) } else { (px, py) };

                    rays[r].push(Attacker {
                        value: get_piece_value(p.piece_type()),
                        piece_type: p.piece_type(),
                        color: p.color(),
                        pos: Coordinate::new(final_px, final_py),
                    });
                }
            } else {
                let end_idx = spatial_line.coords.partition_point(|&c| c < target_coord);
                let start_idx = end_idx.saturating_sub(16);
                for i in (start_idx..end_idx).rev() {
                    let p = crate::board::Piece::from_packed(spatial_line.pieces[i]);
                    let px = spatial_line.coords[i];
                    let py = if dx == 0 {
                        px
                    } else if dy == 0 {
                        target_y
                    } else if dx == dy {
                        px - (target_x - target_y)
                    } else {
                        (target_x + target_y) - px
                    };
                    let (final_px, final_py) = if dx == 0 { (target_x, px) } else { (px, py) };

                    rays[r].push(Attacker {
                        value: get_piece_value(p.piece_type()),
                        piece_type: p.piece_type(),
                        color: p.color(),
                        pos: Coordinate::new(final_px, final_py),
                    });
                }
            }
        }
    }

    // 2. Gather leapers using bitboard neighborhood (O(1) area check)
    let mut leapers: [ArrayVec<Attacker, 24>; 2] = Default::default(); // 0: White, 1: Black
    let neighborhood = game.board.get_neighborhood(target_x, target_y);
    let local_idx = crate::tiles::local_index(target_x, target_y);

    use crate::attacks::*;
    use crate::tiles::masks;

    for (c_idx, color) in [PlayerColor::White, PlayerColor::Black].iter().enumerate() {
        let is_white = *color == PlayerColor::White;
        let p_masks = masks::pawn_attacker_masks(is_white);
        let pawn_bit = 1u32 << (PieceType::Pawn as u8);

        for n in 0..9 {
            let Some(tile) = neighborhood[n] else {
                continue;
            };
            let (occ, type_mask) = if is_white {
                (tile.occ_white, tile.type_mask_white)
            } else {
                (tile.occ_black, tile.type_mask_black)
            };
            if occ == 0 {
                continue;
            }

            let masks_to_check = [
                (masks::KNIGHT_MASKS[local_idx][n], KNIGHT_MASK),
                (masks::KING_MASKS[local_idx][n], KING_MASK),
                (masks::CAMEL_MASKS[local_idx][n], CAMEL_MASK),
                (masks::GIRAFFE_MASKS[local_idx][n], GIRAFFE_MASK),
                (masks::ZEBRA_MASKS[local_idx][n], ZEBRA_MASK),
                (masks::HAWK_MASKS[local_idx][n], HAWK_MASK),
                (p_masks[local_idx][n], pawn_bit),
            ];

            for (attack_mask, req_mask) in masks_to_check {
                if (type_mask & req_mask) == 0 {
                    continue;
                }
                let mut bits = occ & attack_mask;
                while bits != 0 {
                    let b_idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let pt = Piece::from_packed(tile.piece[b_idx]).piece_type();
                    if matches_mask(pt, req_mask) {
                        let (lx, ly) = (b_idx % 8, b_idx / 8);
                        let (cx, cy) = crate::tiles::tile_coords(target_x, target_y);
                        // Neighbor tile cx,cy adjustment
                        let n_dx = (n % 3) as i64 - 1;
                        let n_dy = (n / 3) as i64 - 1;
                        let px = (cx + n_dx) * 8 + lx as i64;
                        let py = (cy + n_dy) * 8 + ly as i64;

                        leapers[c_idx].push(Attacker {
                            value: get_piece_value(pt),
                            piece_type: pt,
                            color: *color,
                            pos: Coordinate::new(px, py),
                        });
                    }
                }
            }
        }
        leapers[c_idx].sort_by_key(|a| a.value);
    }

    // 2b. Gather Knightriders (8 sliding knight directions)
    if indices.has_knightrider[0] || indices.has_knightrider[1] {
        for (i, &(dx, dy)) in crate::attacks::KNIGHTRIDER_DIRS.iter().enumerate() {
            let mut k = 1i64;
            loop {
                let x = target_x + dx * k;
                let y = target_y + dy * k;
                if let Some(p) = game.board.get_piece(x, y) {
                    if p.piece_type() == PieceType::Knightrider {
                        rays[8 + i].push(Attacker {
                            value: get_piece_value(p.piece_type()),
                            piece_type: p.piece_type(),
                            color: p.color(),
                            pos: Coordinate::new(x, y),
                        });
                    }
                    break;
                }
                k += 1;
                if k > 20 {
                    break;
                }
            }
        }
    }

    // 3. Recapture sequence simulation
    let mut gain: [i32; 32] = [0; 32];
    let mut depth = 1;
    gain[0] = get_piece_value(captured.piece_type());

    let mut side = game.turn;
    let mut occ_val = get_piece_value(m.piece.piece_type());
    let mut ray_ptrs = [0usize; 16];

    // Remove moving piece from lists to avoid duplication
    let mut found_initial = false;
    for r in 0..16 {
        if let Some(i) = rays[r].iter().position(|a| a.pos == m.from) {
            rays[r].remove(i);
            found_initial = true;
            break;
        }
    }
    if !found_initial {
        let s_idx = if side == PlayerColor::White { 0 } else { 1 };
        if let Some(i) = leapers[s_idx].iter().position(|a| a.pos == m.from) {
            leapers[s_idx].remove(i);
        }
    }

    loop {
        side = side.opponent();
        if depth >= 32 {
            break;
        }

        let s_idx = if side == PlayerColor::White { 0 } else { 1 };
        let mut best_val = i32::MAX;
        let mut best_src: Option<LvaSource> = None;

        enum LvaSource {
            Ray(usize),
            Leaper(usize),
        }

        for r in 0..16 {
            let ptr = ray_ptrs[r];
            if ptr < rays[r].len() {
                let a = &rays[r][ptr];
                if a.color == side {
                    let can_attack = if r < 4 {
                        is_ortho_slider(a.piece_type)
                    } else if r < 8 {
                        is_diag_slider(a.piece_type)
                    } else {
                        a.piece_type == PieceType::Knightrider
                    };
                    if can_attack && a.value < best_val {
                        best_val = a.value;
                        best_src = Some(LvaSource::Ray(r));
                    }
                }
            }
        }

        if !leapers[s_idx].is_empty() {
            let a = &leapers[s_idx][0];
            if a.value < best_val {
                best_val = a.value;
                best_src = Some(LvaSource::Leaper(0));
            }
        }

        if let Some(src) = best_src {
            gain[depth] = occ_val - gain[depth - 1];
            occ_val = best_val;
            match src {
                LvaSource::Ray(r) => ray_ptrs[r] += 1,
                LvaSource::Leaper(i) => {
                    leapers[s_idx].remove(i);
                }
            }
            depth += 1;
        } else {
            break;
        }
    }

    // Negamax the gain list to find optimal outcome
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
