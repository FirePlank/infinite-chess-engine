use crate::board::{PieceType, PlayerColor};
use crate::evaluation::base;
use crate::game::GameState;

#[inline]
pub fn evaluate(game: &GameState) -> i32 {
    match crate::evaluation::insufficient_material::evaluate_insufficient_material(game) {
        Some(0) => return 0,
        Some(d) => return evaluate_inner(game) / d,
        None => {}
    }
    evaluate_inner(game)
}

#[inline]
fn evaluate_inner(game: &GameState) -> i32 {
    let mut score = game.material_score;
    let (wk, bk) = (game.white_king_pos, game.black_king_pos);

    base::EVAL_WHITE_PAWNS.with(|wp_cell| {
        base::EVAL_BLACK_PAWNS.with(|bp_cell| {
            base::EVAL_PIECE_LIST.with(|pl_cell| {
                base::EVAL_WHITE_RQ.with(|wrq_cell| {
                    base::EVAL_BLACK_RQ.with(|brq_cell| {
                        let mut white_pawns = wp_cell.borrow_mut();
                        let mut black_pawns = bp_cell.borrow_mut();
                        let mut heavy_pieces = pl_cell.borrow_mut();
                        let mut white_rq = wrq_cell.borrow_mut();
                        let mut black_rq = brq_cell.borrow_mut();

                        white_pawns.clear();
                        black_pawns.clear();
                        heavy_pieces.clear();
                        white_rq.clear();
                        black_rq.clear();

                        // 1. Single Board Pass: Collect and evaluate Simple Pieces (Pawns)
                        for (cx, cy, tile) in game.board.tiles.iter() {
                            if crate::simd::both_zero(tile.occ_white, tile.occ_black) {
                                continue;
                            }

                            let mut bits = tile.occ_all;
                            while bits != 0 {
                                let idx = bits.trailing_zeros() as usize;
                                bits &= bits - 1;
                                let packed = tile.piece[idx];
                                if packed == 0 {
                                    continue;
                                }
                                let p = crate::board::Piece::from_packed(packed);
                                if p.color() == PlayerColor::Neutral {
                                    continue;
                                }

                                let pt = p.piece_type();
                                let x = cx * 8 + (idx % 8) as i64;
                                let y = cy * 8 + (idx / 8) as i64;

                                if pt == PieceType::Pawn {
                                    let v = eval_pawn(x, y, p.color(), game);
                                    if p.color() == PlayerColor::White {
                                        score += v;
                                        white_pawns.push((x, y));
                                    } else {
                                        score -= v;
                                        black_pawns.push((x, y));
                                    }
                                } else {
                                    heavy_pieces.push((x, y, p));
                                    // Track RQ for pawn structure evaluation
                                    match pt {
                                        PieceType::Rook
                                        | PieceType::Queen
                                        | PieceType::Amazon
                                        | PieceType::Chancellor
                                        | PieceType::RoyalQueen => {
                                            if p.color() == PlayerColor::White {
                                                white_rq.push((x, y));
                                            } else {
                                                black_rq.push((x, y));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        // 2. Sort lists for count_pawns_on_file O(log N) lookups
                        white_pawns.sort_unstable();
                        black_pawns.sort_unstable();

                        // 3. Deferred Evaluation for Sliders (now with full pawn data)
                        for &(x, y, p) in heavy_pieces.iter() {
                            let is_white = p.color() == PlayerColor::White;
                            let pt = p.piece_type();

                            let v = match pt {
                                PieceType::Rook | PieceType::Chancellor | PieceType::Amazon => {
                                    base::evaluate_rook(
                                        game,
                                        x,
                                        y,
                                        p.color(),
                                        &wk,
                                        &bk,
                                        base::MAX_PHASE,
                                        &white_pawns,
                                        &black_pawns,
                                    )
                                }
                                PieceType::Queen | PieceType::RoyalQueen => base::evaluate_queen(
                                    game,
                                    x,
                                    y,
                                    p.color(),
                                    &wk,
                                    &bk,
                                    base::MAX_PHASE,
                                    &white_pawns,
                                    &black_pawns,
                                ),
                                PieceType::Bishop => base::evaluate_bishop(
                                    game,
                                    x,
                                    y,
                                    p.color(),
                                    &wk,
                                    &bk,
                                    base::MAX_PHASE,
                                    &white_pawns,
                                    &black_pawns,
                                ),
                                _ => 0,
                            };

                            if is_white {
                                score += v;
                            } else {
                                score -= v;
                            }
                        }

                        // 4. Mop-up (uses collected info)
                        if crate::evaluation::mop_up::calculate_mop_up_scale(
                            game,
                            PlayerColor::Black,
                        )
                        .is_some()
                        {
                            if let Some(b) = &bk {
                                score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
                                    game,
                                    wk.as_ref(),
                                    b,
                                    PlayerColor::White,
                                    PlayerColor::Black,
                                );
                            }
                        } else if crate::evaluation::mop_up::calculate_mop_up_scale(
                            game,
                            PlayerColor::White,
                        )
                        .is_some()
                            && let Some(w) = &wk
                        {
                            score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
                                game,
                                bk.as_ref(),
                                w,
                                PlayerColor::Black,
                                PlayerColor::White,
                            );
                        }

                        // Pawn Structure (uses collected info to avoid RefCell borrow panic)
                        score += base::evaluate_pawn_structure_traced(
                            game,
                            game.total_phase.min(base::MAX_PHASE),
                            &wk,
                            &bk,
                            &mut base::NoTrace,
                            &white_pawns,
                            &black_pawns,
                            &white_rq,
                            &black_rq,
                        );

                        // 5. optimized race_eval
                        score += race_eval_optimized(game, &white_pawns, &black_pawns);
                    });
                });
            });
        });
    });

    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// Pawn eval: HUGE bonus for edge/outside, penalty for center
#[inline]
fn eval_pawn(x: i64, y: i64, color: PlayerColor, game: &GameState) -> i32 {
    let dist = if color == PlayerColor::White {
        (game.white_promo_rank - y).max(0)
    } else {
        (y - game.black_promo_rank).max(0)
    };

    let mut b: i32 = 0;

    // Advancement: 10cp per rank (max 80cp)
    b += (8 - dist.min(8)) as i32 * 10;

    // LANE BONUS (the whole point)
    if x < 1 {
        // LEFT OUTSIDE: 100 + 15 per file out
        b += 100 + ((1 - x) as i32 * 15).min(90);
    } else if x > 8 {
        // RIGHT OUTSIDE: 100 + 15 per file out
        b += 100 + ((x - 8) as i32 * 15).min(90);
    } else if x == 1 || x == 8 {
        // EDGE FILES: Strong priority
        b += 80;
    } else if x == 2 || x == 7 {
        // NEAR EDGE
        b += 25;
    } else {
        // CENTER (x=3,4,5,6): PENALTY
        b -= 40;
    }

    b
}

/// Race evaluation: Who's closest to promoting on edge/outside?
fn race_eval_optimized(
    game: &GameState,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let mut w_min: i64 = 100;
    let mut b_min: i64 = 100;

    for &(x, y) in white_pawns.iter() {
        if x > 1 && x < 8 {
            continue;
        }
        let d = (game.white_promo_rank - y).max(0);
        if d < w_min {
            w_min = d;
        }
    }

    for &(x, y) in black_pawns.iter() {
        if x > 1 && x < 8 {
            continue;
        }
        let d = (y - game.black_promo_rank).max(0);
        if d < b_min {
            b_min = d;
        }
    }

    let mut s: i32 = 0;

    // Race comparison: 100cp per move, max 500cp
    if w_min < 100 && b_min < 100 {
        let diff = b_min - w_min;
        s += (diff as i32 * 100).clamp(-500, 500);
    } else if w_min < 100 {
        s += (10 - w_min).max(0) as i32 * 40;
    } else if b_min < 100 {
        s -= (10 - b_min).max(0) as i32 * 40;
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_obstocean_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.variant = Some(crate::Variant::Obstocean);
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    #[test]
    fn test_evaluate_returns_value() {
        let mut game = create_obstocean_game();
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
    fn test_edge_pawn_bonus() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.recompute_piece_counts();

        // Test eval_pawn directly (avoids mop-up interference)
        let edge_score = eval_pawn(1, 4, PlayerColor::White, &game);
        let center_score = eval_pawn(4, 4, PlayerColor::White, &game);

        // Edge pawn should score better (80 vs -40)
        assert!(
            edge_score > center_score,
            "Edge pawn ({}) should score better than center pawn ({})",
            edge_score,
            center_score
        );
    }

    #[test]
    fn test_eval_pawn_function() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.recompute_piece_counts();

        // Edge file should give big bonus
        let edge_score = eval_pawn(1, 3, PlayerColor::White, &game);
        let center_score = eval_pawn(4, 3, PlayerColor::White, &game);

        assert!(edge_score > center_score, "Edge pawn should score higher");
    }

    #[test]
    fn test_race_eval_basic() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White edge pawn near promotion
        game.board
            .set_piece(1, 7, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game.recompute_piece_counts();

        let mut w_pawns = Vec::new();
        let mut b_pawns = Vec::new();
        for (x, y, p) in game.board.iter() {
            if p.piece_type() == PieceType::Pawn {
                if p.color() == PlayerColor::White {
                    w_pawns.push((x, y));
                } else if p.color() == PlayerColor::Black {
                    b_pawns.push((x, y));
                }
            }
        }
        let race = race_eval_optimized(&game, &w_pawns, &b_pawns);
        // White should be winning the race
        assert!(
            race > 0,
            "White pawn near promo should give positive race eval"
        );
    }

    #[test]
    fn test_outside_file_bonus() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.recompute_piece_counts();

        // x=0 is "outside" (left of a-file)
        let outside_score = eval_pawn(0, 4, PlayerColor::White, &game);
        let edge_score = eval_pawn(1, 4, PlayerColor::White, &game);

        // Outside should be even better than edge
        assert!(
            outside_score > edge_score,
            "Outside file pawn should be best"
        );
    }

    #[test]
    fn test_race_eval_both_sides_racing() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White pawn closer to promotion
        game.board
            .set_piece(1, 7, Piece::new(PieceType::Pawn, PlayerColor::White));
        // Black pawn further from promotion
        game.board
            .set_piece(1, 4, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game.recompute_piece_counts();

        let mut w_pawns = Vec::new();
        let mut b_pawns = Vec::new();
        for (x, y, p) in game.board.iter() {
            if p.piece_type() == PieceType::Pawn {
                if p.color() == PlayerColor::White {
                    w_pawns.push((x, y));
                } else if p.color() == PlayerColor::Black {
                    b_pawns.push((x, y));
                }
            }
        }
        let race = race_eval_optimized(&game, &w_pawns, &b_pawns);
        // White should be winning the race (1 move vs 3 moves)
        assert!(race > 0, "White closer to promo should win race: {}", race);
    }

    #[test]
    fn test_evaluate_inner_returns_value() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(3, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game.recompute_piece_counts();

        let score = evaluate_inner(&game);
        // Should return a valid evaluation (not panic or overflow)
        assert!(
            score.abs() < 100000,
            "Score should be reasonable: {}",
            score
        );
    }

    #[test]
    fn test_black_advantage_race() {
        let mut game = create_obstocean_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // White pawn far from promotion
        game.board
            .set_piece(1, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        // Black pawn very close to promotion
        game.board
            .set_piece(1, 2, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game.recompute_piece_counts();

        let mut w_pawns = Vec::new();
        let mut b_pawns = Vec::new();
        for (x, y, p) in game.board.iter() {
            if p.piece_type() == PieceType::Pawn {
                if p.color() == PlayerColor::White {
                    w_pawns.push((x, y));
                } else if p.color() == PlayerColor::Black {
                    b_pawns.push((x, y));
                }
            }
        }
        let race = race_eval_optimized(&game, &w_pawns, &b_pawns);
        // Black should be winning the race
        assert!(race < 0, "Black closer to promo should win race: {}", race);
    }
}
