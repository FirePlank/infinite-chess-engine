use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

/// Tests if SEE value of move is >= threshold.
/// Uses early cutoffs to avoid full SEE calculation when possible.
#[inline(always)]
pub(crate) fn see_ge(game: &GameState, m: &Move, threshold: i32) -> bool {
    let mover_color = m.piece.color();

    // Material won by the move before any recapture: the captured piece (0 for a
    // quiet, non-capturing move; a pawn for en passant) plus the promotion gain
    // if the move promotes. For a promotion the piece left standing on the square
    // is the promoted piece, so that is what is at risk of recapture.
    let captured_val = if game.is_en_passant(m) {
        game.get_piece_value(PieceType::Pawn, mover_color.opponent())
    } else {
        match game.board.get_piece(m.to.x, m.to.y) {
            Some(p) => game.get_piece_value(p.piece_type(), p.color()),
            None => 0,
        }
    };
    let (promo_gain, mover_val) = match m.promotion {
        Some(promo) => {
            let pv = game.get_piece_value(promo, mover_color);
            (pv - game.get_piece_value(m.piece.piece_type(), mover_color), pv)
        }
        None => (0, game.get_piece_value(m.piece.piece_type(), mover_color)),
    };
    let victim_val = captured_val + promo_gain;
    let attacker_val = mover_val;

    // Early cutoff 1: if the move cannot reach the threshold even when the moved
    // piece is never recaptured, fail. For a quiet move victim_val == 0, so any
    // positive threshold fails here without touching the board.
    let swap = victim_val - threshold;
    if swap < 0 {
        return false;
    }

    // Early cutoff 2: if we still meet the threshold even after losing the moved
    // piece, pass.
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
    let target_x = m.to.x;
    let target_y = m.to.y;

    // Material captured by the move itself. 0 for a quiet (non-capturing) move
    // (SEE then measures whether the moved piece can be safely placed on the
    // target square); a pawn for en passant (whose victim sits off m.to).
    let captured_val = if game.is_en_passant(m) {
        game.get_piece_value(PieceType::Pawn, m.piece.color().opponent())
    } else {
        match game.board.get_piece(m.to.x, m.to.y) {
            Some(p) => game.get_piece_value(p.piece_type(), p.color()),
            None => 0,
        }
    };

    // The piece that ends up standing on the target square (and is thus exposed
    // to recapture). For a promotion this is the promoted piece, not the pawn,
    // and the promotion is itself an immediate material gain.
    let mover_color = m.piece.color();
    let (promo_gain, mover_val) = match m.promotion {
        Some(promo) => {
            let pv = game.get_piece_value(promo, mover_color);
            (pv - game.get_piece_value(m.piece.piece_type(), mover_color), pv)
        }
        None => (0, game.get_piece_value(m.piece.piece_type(), mover_color)),
    };

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
    gain[0] = captured_val + promo_gain;

    let mut side = game.turn;
    let mut occ_val = mover_val;

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
                            value: game.get_piece_value(p.piece_type(), p.color()),
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
                        value: game.get_piece_value(PieceType::Pawn, color),
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
                        value: game.get_piece_value(pt, p.color()),
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
                            value: game.get_piece_value(npt, np.color()),
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
    use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;
    use crate::moves::Move;

    fn create_test_game() -> GameState {
        GameState::new()
    }

    fn create_test_game_from_icn(icn: &str) -> GameState {
        let mut game = create_test_game();
        game.setup_position_from_icn(icn);
        game
    }

    #[test]
    fn test_see_simple_pawn_takes_pawn() {
        let mut game = create_test_game_from_icn("w (8;q|1;q) P4,4|p5,5");
        game.turn = PlayerColor::White;

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
        let mut game = create_test_game_from_icn("w (8;q|1;q) Q4,4|p5,5|p6,6");
        game.turn = PlayerColor::White;

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
    fn test_see_ge_threshold_pass() {
        let mut game = create_test_game_from_icn("w (8;q|1;q) P4,4|q5,5");
        game.turn = PlayerColor::White;

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
        let mut game = create_test_game_from_icn("w (8;q|1;q) Q4,4|p5,5");
        game.turn = PlayerColor::White;

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
        let mut game = create_test_game_from_icn("w (8;q|1;q) R4,4");
        game.turn = PlayerColor::White;

        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 5), // Empty square
            Piece::new(PieceType::Rook, PlayerColor::White),
        );

        let see_val = static_exchange_eval_impl(&game, &m);
        assert_eq!(see_val, 0, "Non-capture should return 0");
    }

    #[test]
    fn test_see_quiet_safe_thresholds() {
        // Rook to an empty, unattacked square: SEE == 0, and see_ge reflects that
        // a quiet move can never *gain* material.
        let mut game = create_test_game_from_icn("w (8;q|1;q) R5,1");
        game.turn = PlayerColor::White;
        let m = Move::new(
            Coordinate::new(5, 1),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Rook, PlayerColor::White),
        );
        assert_eq!(static_exchange_eval_impl(&game, &m), 0);
        assert!(see_ge(&game, &m, 0), "Safe quiet move meets threshold 0");
        assert!(!see_ge(&game, &m, 1), "Quiet move can never gain material");
    }

    #[test]
    fn test_see_quiet_hanging_piece() {
        // White rook moves to an empty square attacked by a black pawn with no
        // white defender: the rook is lost for nothing.
        let mut game = create_test_game_from_icn("w (8;q|1;q) R5,1|p6,6");
        game.turn = PlayerColor::White;
        let m = Move::new(
            Coordinate::new(5, 1),
            Coordinate::new(5, 5), // empty, attacked by p6,6
            Piece::new(PieceType::Rook, PlayerColor::White),
        );

        let rook = game.get_piece_value(PieceType::Rook, PlayerColor::White);
        assert_eq!(
            static_exchange_eval_impl(&game, &m),
            -rook,
            "Quiet move hanging the rook to a pawn should be -rook"
        );
        assert!(!see_ge(&game, &m, 0), "Hanging quiet move must fail see_ge(>= 0)");
        assert!(
            see_ge(&game, &m, -rook),
            "Hanging quiet move still meets a threshold equal to the loss"
        );
    }

    #[test]
    fn test_see_promotion_safe() {
        // Pawn promotes to a queen on a safe empty square: net gain ~ queen - pawn.
        let mut game = create_test_game_from_icn("w (8;q|1;q) P5,7");
        game.turn = PlayerColor::White;
        let mut m = Move::new(
            Coordinate::new(5, 7),
            Coordinate::new(5, 8),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        m.promotion = Some(PieceType::Queen);

        let q = game.get_piece_value(PieceType::Queen, PlayerColor::White);
        let p = game.get_piece_value(PieceType::Pawn, PlayerColor::White);
        assert_eq!(
            static_exchange_eval_impl(&game, &m),
            q - p,
            "Safe promotion should net queen minus pawn"
        );
        assert!(see_ge(&game, &m, 0), "Safe promotion is winning");
    }

    #[test]
    fn test_see_promotion_hanging() {
        // Pawn promotes on a square attacked by a black pawn, undefended: we
        // promote (+queen-pawn) then lose the queen (-queen) => net -pawn.
        let mut game = create_test_game_from_icn("w (8;q|1;q) P5,7|p6,9");
        game.turn = PlayerColor::White;
        let mut m = Move::new(
            Coordinate::new(5, 7),
            Coordinate::new(5, 8),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        m.promotion = Some(PieceType::Queen);

        let p = game.get_piece_value(PieceType::Pawn, PlayerColor::White);
        assert_eq!(
            static_exchange_eval_impl(&game, &m),
            -p,
            "Promotion hanging the new queen to a pawn nets -pawn"
        );
    }

    #[test]
    fn test_see_en_passant_wins_a_pawn() {
        use crate::game::EnPassantState;
        // Black pawn just double-pushed to (4,5); white pawn on (5,5) can capture
        // en passant to the empty square (4,6), winning the pawn (undefended).
        let mut game = create_test_game_from_icn("w (8;q|1;q) P5,5|p4,5");
        game.turn = PlayerColor::White;
        game.en_passant = Some(EnPassantState {
            square: Coordinate::new(4, 6),
            pawn_square: Coordinate::new(4, 5),
        });
        let m = Move::new(
            Coordinate::new(5, 5),
            Coordinate::new(4, 6),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        assert!(game.is_en_passant(&m), "move should be detected as en passant");
        let p = game.get_piece_value(PieceType::Pawn, PlayerColor::White);
        assert_eq!(
            static_exchange_eval_impl(&game, &m),
            p,
            "En passant should be valued as winning a pawn, not 0 (quiet)"
        );
        assert!(see_ge(&game, &m, 0));
        assert!(!see_ge(&game, &m, p + 1));
    }
}
