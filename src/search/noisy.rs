use super::params;
use super::*;
use crate::evaluation::evaluate;
use std::cell::RefCell;

thread_local! {
    static NOISE_SEED: RefCell<u64> = RefCell::new(0);
}

pub fn reset_noise_seed(seed: u64) {
    NOISE_SEED.with(|c| *c.borrow_mut() = seed);
}

#[inline]
fn evaluate_with_noise(game: &GameState, noise_amp: i32) -> i32 {
    let base = evaluate(game);
    if noise_amp <= 0 {
        return base;
    }
    if base.abs() >= MATE_SCORE {
        return base;
    }

    // Generate a determenistic wiggle to the eval based on the game state
    let hash = TranspositionTable::generate_hash(game);
    let seed = NOISE_SEED.with(|c| *c.borrow());
    let mut x = (hash as u64) ^ seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let span = (2 * noise_amp + 1) as i64;
    let noise_raw = (x % (span as u64)) as i64;
    let noise = (noise_raw - noise_amp as i64) as i32;
    base.saturating_add(noise)
}

pub fn get_best_move_with_noise(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    noise_amp: i32,
    silent: bool,
) -> Option<(Move, i32, SearchStats)> {
    game.recompute_piece_counts();
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.time_limit_ms = time_limit_ms;
    searcher.silent = silent;

    if noise_amp <= 0 {
        let result = super::get_best_move(game, max_depth, time_limit_ms, silent);
        return result;
    }

    let result = search_with_searcher_noisy(&mut searcher, game, max_depth, noise_amp);
    let stats = super::build_search_stats(&searcher);
    result.map(|(m, eval)| (m, eval, stats))
}

fn search_with_searcher_noisy(
    searcher: &mut Searcher,
    game: &mut GameState,
    max_depth: usize,
    noise_amp: i32,
) -> Option<(Move, i32)> {
    let moves = game.get_legal_moves();
    if moves.is_empty() {
        return None;
    }

    if moves.len() == 1 {
        let single = moves[0].clone();
        let score = evaluate_with_noise(game, noise_amp);
        return Some((single, score));
    }

    let fallback_move = moves
        .iter()
        .find(|m| {
            let undo = game.make_move(m);
            let legal = !game.is_move_illegal();
            game.undo_move(m, undo);
            legal
        })
        .cloned();

    let mut best_move: Option<Move> = fallback_move.clone();
    let mut best_score = -INFINITY;
    let mut stability: usize = 0;
    let mut prev_iter_score: i32 = 0;
    let mut has_prev_iter_score = false;
    let mut prev_root_move_coords: Option<(i64, i64, i64, i64)> = None;

    for depth in 1..=max_depth {
        searcher.reset_for_iteration();
        searcher.decay_history();

        if searcher.timer.elapsed_ms() >= searcher.time_limit_ms {
            searcher.stopped = true;
            break;
        }

        let score = if depth == 1 {
            negamax_root_noisy(searcher, game, depth, -INFINITY, INFINITY, noise_amp)
        } else {
            let asp_win = aspiration_window();
            let mut alpha = searcher.prev_score - asp_win;
            let mut beta = searcher.prev_score + asp_win;
            let mut window_size = asp_win;
            let mut result;
            let mut retries = 0;

            loop {
                result = negamax_root_noisy(searcher, game, depth, alpha, beta, noise_amp);
                retries += 1;

                if searcher.stopped {
                    break;
                }

                if result <= alpha {
                    window_size *= aspiration_fail_mult();
                    alpha = searcher.prev_score - window_size;
                } else if result >= beta {
                    window_size *= aspiration_fail_mult();
                    beta = searcher.prev_score + window_size;
                } else {
                    break;
                }

                if window_size > 1000 || retries >= 4 {
                    result =
                        negamax_root_noisy(searcher, game, depth, -INFINITY, INFINITY, noise_amp);
                    break;
                }
            }
            result
        };

        // Root PV is at pv_table[0]
        if let Some(pv_move) = searcher.pv_table[0] {
            best_move = Some(pv_move);
            best_score = score;
            searcher.best_move_root = Some(pv_move);
            searcher.prev_score = score;

            let coords = (pv_move.from.x, pv_move.from.y, pv_move.to.x, pv_move.to.y);
            if let Some(prev_coords) = prev_root_move_coords {
                if prev_coords == coords {
                    stability += 1;
                } else {
                    stability = 0;
                }
            } else {
                stability = 0;
            }
            prev_root_move_coords = Some(coords);
        }

        if !searcher.stopped && !searcher.silent {
            searcher.print_info(depth, score);
        }

        if searcher.stopped || best_score.abs() > MATE_SCORE {
            break;
        }

        if searcher.time_limit_ms != u128::MAX {
            let elapsed = searcher.timer.elapsed_ms();
            let limit = searcher.time_limit_ms;

            if best_move.is_some() {
                let mut factor = 1.1_f64 - 0.03_f64 * (stability as f64);
                if factor < 0.5 {
                    factor = 0.5;
                }

                if has_prev_iter_score && best_score - prev_iter_score > aspiration_window() {
                    factor *= 1.1;
                }

                if factor > 1.0 {
                    factor = 1.0;
                }

                let ideal_ms = (limit as f64 * factor) as u128;
                let soft_limit = std::cmp::min(limit, ideal_ms);

                if elapsed >= soft_limit {
                    break;
                }

                prev_iter_score = best_score;
                has_prev_iter_score = true;
            }

            let cutoff = if limit <= 300 {
                limit / 2
            } else if limit <= 2000 {
                limit.saturating_sub(250)
            } else if limit <= 8000 {
                limit.saturating_sub(500)
            } else {
                limit.saturating_sub(2000)
            };

            if elapsed >= cutoff {
                break;
            }
        }
    }

    searcher.tt.increment_age();

    best_move.map(|m| (m, best_score))
}

fn negamax_root_noisy(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    mut alpha: i32,
    beta: i32,
    noise_amp: i32,
) -> i32 {
    let alpha_orig = alpha;

    searcher.pv_length[0] = 0;

    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;

    if let Some((_, best)) = searcher.tt.probe(hash, alpha, beta, depth, 0) {
        tt_move = best;
    }

    let in_check = game.is_in_check();

    let mut moves = Vec::new();
    std::mem::swap(&mut moves, &mut searcher.move_buffers[0]);
    game.get_legal_moves_into(&mut moves);

    sort_moves_root(searcher, game, &mut moves, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut fallback_move: Option<Move> = None;
    let mut legal_moves = 0;

    for m in &moves {
        if !searcher.excluded_moves.is_empty() {
            let coords = (m.from.x, m.from.y, m.to.x, m.to.y);
            if searcher.excluded_moves.contains(&coords) {
                continue;
            }
        }

        let undo = game.make_move(m);

        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        let prev_entry_backup = searcher.prev_move_stack[0];
        let prev_from_hash = hash_move_from(m);
        let prev_to_hash = hash_move_dest(m);
        searcher.prev_move_stack[0] = (prev_from_hash, prev_to_hash);

        legal_moves += 1;

        if fallback_move.is_none() {
            fallback_move = Some(m.clone());
        }

        let score = if legal_moves == 1 {
            -negamax_noisy(searcher, game, depth - 1, 1, -beta, -alpha, true, noise_amp)
        } else {
            let mut s = -negamax_noisy(
                searcher,
                game,
                depth - 1,
                1,
                -alpha - 1,
                -alpha,
                true,
                noise_amp,
            );
            if s > alpha && s < beta {
                s = -negamax_noisy(searcher, game, depth - 1, 1, -beta, -alpha, true, noise_amp);
            }
            s
        };

        game.undo_move(m, undo);

        searcher.prev_move_stack[0] = prev_entry_backup;

        if searcher.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());

            if score > alpha {
                alpha = score;

                // Update PV using triangular indexing (root)
                searcher.pv_table[0] = Some(*m);
                let child_len = searcher.pv_length[1];
                let child_base = MAX_PLY;
                for j in 0..child_len {
                    searcher.pv_table[1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[0] = child_len + 1;
            }
        }

        if alpha >= beta {
            break;
        }
    }

    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        return if in_check || no_pieces {
            -MATE_VALUE
        } else {
            0
        };
    }

    std::mem::swap(&mut searcher.move_buffers[0], &mut moves);

    let tt_flag = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    searcher
        .tt
        .store(hash, depth, tt_flag, best_score, best_move, 0);

    best_score
}

fn negamax_noisy(
    searcher: &mut Searcher,
    game: &mut GameState,
    depth: usize,
    ply: usize,
    mut alpha: i32,
    mut beta: i32,
    allow_null: bool,
    noise_amp: i32,
) -> i32 {
    let alpha_orig = alpha;
    let beta_orig = beta;

    searcher.nodes += 1;
    // Initialize PV length to 0; will be updated if alpha is raised
    searcher.pv_length[ply] = 0;

    if ply > searcher.seldepth {
        searcher.seldepth = ply;
    }

    if searcher.check_time() {
        return 0;
    }

    if ply >= MAX_PLY - 1 {
        return evaluate_with_noise(game, noise_amp);
    }

    if game.is_fifty() {
        return 0;
    }

    let hash = TranspositionTable::generate_hash(game);

    if ply > 0 && game.is_threefold() {
        return -repetition_penalty();
    }

    if ply > 0 {
        let mate_score = MATE_VALUE - ply as i32;
        if alpha < -mate_score {
            alpha = -mate_score;
        }
        if beta > mate_score - 1 {
            beta = mate_score - 1;
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let in_check = game.is_in_check();
    let is_pv = beta > alpha + 1;

    if depth == 0 {
        return quiescence_noisy(searcher, game, ply, alpha, beta, noise_amp);
    }

    let mut depth = depth;

    if in_check && ply < MAX_PLY / 2 {
        depth += 1;
    }
    let mut tt_move: Option<Move> = None;

    if let Some((score, best)) = searcher.tt.probe(hash, alpha, beta, depth, ply) {
        tt_move = best;

        if !is_pv && score != INFINITY + 1 {
            return score;
        }
    }

    let static_eval = if in_check {
        -MATE_VALUE + ply as i32
    } else {
        evaluate_with_noise(game, noise_amp)
    };

    searcher.eval_stack[ply] = static_eval;

    let improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true
    };

    let cut_node = !is_pv && beta == alpha + 1;
    if tt_move.is_none() && cut_node && !in_check && depth >= 4 {
        depth -= 1;
    }

    if !in_check && !is_pv {
        if depth < rfp_max_depth() && static_eval - rfp_margin_per_depth() * depth as i32 >= beta {
            return static_eval;
        }

        if allow_null && depth >= nmp_min_depth() && static_eval >= beta {
            let has_pieces = game.board.pieces.iter().any(|(_, p)| {
                p.color() == game.turn
                    && p.piece_type() != PieceType::Pawn
                    && p.piece_type() != PieceType::King
            });

            if has_pieces {
                let saved_ep = game.en_passant.clone();
                game.make_null_move();

                let r = nmp_reduction() + depth / 6;
                let null_score = -negamax_noisy(
                    searcher,
                    game,
                    depth.saturating_sub(1 + r),
                    ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    noise_amp,
                );

                game.unmake_null_move();
                game.en_passant = saved_ep;

                if searcher.stopped {
                    return 0;
                }

                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    let mut moves = Vec::new();
    std::mem::swap(&mut moves, &mut searcher.move_buffers[ply]);
    game.get_legal_moves_into(&mut moves);

    sort_moves(searcher, game, &mut moves, ply, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    let mut quiets_searched: Vec<Move> = Vec::new();

    let futility_pruning = !in_check && !is_pv && depth <= 3;
    let futility_base = if futility_pruning {
        static_eval + futility_margin(depth)
    } else {
        0
    };

    for m in &moves {
        let captured_piece = game.board.get_piece(&m.to.x, &m.to.y);
        let is_capture = captured_piece.map_or(false, |p| !p.piece_type().is_neutral_type());
        let captured_type = captured_piece.map(|p| p.piece_type());
        let is_promotion = m.promotion.is_some();

        if futility_pruning && legal_moves > 0 && !is_capture && !is_promotion {
            if futility_base <= alpha {
                continue;
            }
        }

        if !in_check && !is_pv && depth <= 4 && depth > 0 && !is_capture && !is_promotion {
            let threshold = lmp_threshold(depth);
            if legal_moves >= threshold && best_score > -MATE_SCORE {
                continue;
            }
        }

        let undo = game.make_move(m);

        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        if !is_capture && !is_promotion {
            quiets_searched.push(m.clone());
        }

        let prev_entry_backup = searcher.prev_move_stack[ply];
        let from_hash = hash_move_from(m);
        let to_hash = hash_move_dest(m);
        searcher.prev_move_stack[ply] = (from_hash, to_hash);

        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        searcher.move_history[ply] = Some(m.clone());
        searcher.moved_piece_history[ply] = m.piece.piece_type() as u8;

        legal_moves += 1;

        let score;
        if legal_moves == 1 {
            score = -negamax_noisy(
                searcher,
                game,
                depth - 1,
                ply + 1,
                -beta,
                -alpha,
                true,
                noise_amp,
            );
        } else {
            let mut reduction = 0;
            if depth >= lmr_min_depth()
                && legal_moves >= lmr_min_moves()
                && !in_check
                && !is_capture
            {
                reduction = 1
                    + (legal_moves as f32).ln() as usize * (depth as f32).ln() as usize
                        / lmr_divisor();

                if !improving {
                    reduction += 1;
                }

                reduction = reduction.min(depth - 2);
            }

            let mut new_depth = depth as i32 - 1 - reduction as i32;

            if !in_check
                && !is_pv
                && !is_capture
                && !is_promotion
                && depth <= hlp_max_depth()
                && legal_moves >= hlp_min_moves()
                && best_score > -MATE_SCORE
            {
                let idx = hash_move_dest(m);
                let value = searcher.history[m.piece.piece_type() as usize][idx];

                if value < hlp_history_reduce() {
                    new_depth -= 1;

                    if new_depth <= 0 && value < hlp_history_leaf() {
                        game.undo_move(m, undo);
                        searcher.prev_move_stack[ply] = prev_entry_backup;
                        searcher.move_history[ply] = move_history_backup;
                        searcher.moved_piece_history[ply] = piece_history_backup;
                        continue;
                    }
                }
            }

            let search_depth = if new_depth <= 0 {
                0
            } else {
                new_depth as usize
            };

            let mut s = -negamax_noisy(
                searcher,
                game,
                search_depth,
                ply + 1,
                -alpha - 1,
                -alpha,
                true,
                noise_amp,
            );

            if s > alpha && (reduction > 0 || s < beta) {
                s = -negamax_noisy(
                    searcher,
                    game,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    true,
                    noise_amp,
                );
            }
            score = s;
        }

        game.undo_move(m, undo);

        searcher.prev_move_stack[ply] = prev_entry_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.stopped {
            std::mem::swap(&mut searcher.move_buffers[ply], &mut moves);
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m.clone());

            if score > alpha {
                alpha = score;

                // Update PV using triangular indexing
                let ply_base = ply * MAX_PLY;
                let child_base = (ply + 1) * MAX_PLY;

                searcher.pv_table[ply_base] = Some(*m);
                let child_len = searcher.pv_length[ply + 1];
                for j in 0..child_len {
                    searcher.pv_table[ply_base + 1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[ply] = child_len + 1;
            }
        }

        if alpha >= beta {
            if !is_capture {
                let idx = hash_move_dest(m);
                let bonus = history_bonus_base() * depth as i32 - history_bonus_sub();
                let adj = bonus.min(history_bonus_cap());
                let max_history: i32 = params::DEFAULT_HISTORY_MAX_GRAVITY;

                searcher.update_history(m.piece.piece_type(), idx, bonus);

                for quiet in &quiets_searched {
                    let qidx = hash_move_dest(quiet);
                    if quiet.piece.piece_type() == m.piece.piece_type() && qidx == idx {
                        continue;
                    }
                    searcher.update_history(quiet.piece.piece_type(), qidx, -bonus);
                }

                searcher.killers[ply][1] = searcher.killers[ply][0].clone();
                searcher.killers[ply][0] = Some(*m);

                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (m.piece.piece_type() as u8, m.to.x as i16, m.to.y as i16);
                    }
                }

                for &plies_ago in &[0usize, 1, 3] {
                    if ply >= plies_ago + 1 {
                        if let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1] {
                            let prev_piece =
                                searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_piece < 16 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);

                                for quiet in &quiets_searched {
                                    let q_from_hash = hash_coord_32(quiet.from.x, quiet.from.y);
                                    let q_to_hash = hash_coord_32(quiet.to.x, quiet.to.y);
                                    let is_best = quiet.from == m.from && quiet.to == m.to;

                                    let entry = &mut searcher.cont_history[prev_piece]
                                        [prev_to_hash][q_from_hash][q_to_hash];
                                    if is_best {
                                        *entry += adj - *entry * adj / max_history;
                                    } else {
                                        *entry += -adj - *entry * adj / max_history;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if let Some(cap_type) = captured_type {
                let bonus = (depth * depth) as i32;
                searcher.capture_history[m.piece.piece_type() as usize][cap_type as usize] += bonus;
            }
            break;
        }
    }

    std::mem::swap(&mut searcher.move_buffers[ply], &mut moves);

    if best_move.is_none() {
        return 0;
    }

    let flag = if best_score <= alpha_orig {
        TTFlag::UpperBound
    } else if best_score >= beta_orig {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };

    searcher
        .tt
        .store(hash, depth, flag, best_score, best_move.clone(), ply);

    best_score
}

fn quiescence_noisy(
    searcher: &mut Searcher,
    game: &mut GameState,
    ply: usize,
    mut alpha: i32,
    beta: i32,
    noise_amp: i32,
) -> i32 {
    searcher.nodes += 1;
    searcher.qnodes += 1;

    if ply > searcher.seldepth {
        searcher.seldepth = ply;
    }

    if searcher.check_time() {
        return 0;
    }

    let in_check = game.is_in_check();

    let stand_pat = if in_check {
        -MATE_VALUE + ply as i32
    } else {
        evaluate_with_noise(game, noise_amp)
    };

    if !in_check {
        if stand_pat >= beta {
            return beta;
        }

        if alpha < stand_pat {
            alpha = stand_pat;
        }
    }

    if ply >= MAX_PLY - 1 {
        return stand_pat;
    }

    let mut tactical_moves = Vec::new();
    std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);

    if in_check {
        game.get_evasion_moves_into(&mut tactical_moves);
    } else {
        get_quiescence_captures(
            &game.board,
            game.turn,
            &game.special_rights,
            &game.en_passant,
            &game.game_rules,
            &game.spatial_indices,
            &mut tactical_moves,
        );
    }

    sort_captures(game, &mut tactical_moves);

    let mut best_score = stand_pat;
    let mut legal_moves = 0;

    const DELTA_MARGIN: i32 = 200;

    for m in &tactical_moves {
        if !in_check {
            let see_gain = static_exchange_eval(game, m);

            if see_gain < 0 {
                continue;
            }

            if stand_pat + see_gain + DELTA_MARGIN < alpha {
                continue;
            }
        }

        let undo = game.make_move(m);

        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        legal_moves += 1;

        let score = -quiescence_noisy(searcher, game, ply + 1, -beta, -alpha, noise_amp);

        game.undo_move(m, undo);

        if searcher.stopped {
            std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);
            return best_score;
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                alpha = score;
            }
        }

        if alpha >= beta {
            break;
        }
    }

    if legal_moves == 0 {
        let no_pieces = !game.has_pieces(game.turn);
        if in_check || no_pieces {
            std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);
            return -MATE_VALUE + ply as i32;
        }
    }

    std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);

    best_score
}
