//! Noisy search module - adds stochastic noise to evaluation for SPRT testing.
//! This module largely mirrors search.rs with noise injection.
//! Coverage exclusion is handled via CI ignore flags.

use super::ordering::*;
use super::params::*;
use super::*;
use crate::board::PlayerColor;
use crate::evaluation::evaluate;
use crate::moves::{MoveGenContext, MoveList, get_quiescence_captures};
use std::cell::RefCell;

pub struct NegamaxNoisyContext<'a> {
    pub searcher: &'a mut Searcher,
    pub game: &'a mut GameState,
    pub depth: usize,
    pub ply: usize,
    pub alpha: i32,
    pub beta: i32,
    pub allow_null: bool,
    pub noise_amp: i32,
}

thread_local! {
    static NOISE_SEED: RefCell<u64> = const { RefCell::new(0) };
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
    let mut x = hash ^ seed;
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
    // Initialize correction history hashes
    game.recompute_correction_hashes();
    let mut searcher = Searcher::new(time_limit_ms);
    searcher.hot.time_limit_ms = time_limit_ms;
    searcher.silent = silent;
    searcher.set_corrhist_mode(game);
    searcher.move_rule_limit = game
        .game_rules
        .move_rule_limit
        .map_or(i32::MAX, |v| v as i32);

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
        let single = moves[0];
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

    let mut best_move: Option<Move> = fallback_move;
    let mut best_score = -INFINITY;
    let mut stability: usize = 0;
    let mut prev_iter_score: i32 = 0;
    let mut has_prev_iter_score = false;
    let mut prev_root_move_coords: Option<(i64, i64, i64, i64)> = None;

    for depth in 1..=max_depth {
        searcher.reset_for_iteration();
        searcher.decay_history();

        // Time check at iteration start - but note that check_time won't stop
        // during depth 1 due to min_depth_required
        if searcher.hot.min_depth_required == 0
            && searcher.hot.timer.elapsed_ms() >= searcher.hot.time_limit_ms
        {
            searcher.hot.stopped = true;
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

                if searcher.hot.stopped {
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

        // After depth 1 completes, allow time stops for subsequent depths
        if depth == 1 {
            searcher.hot.min_depth_required = 0;
        }

        // Update best move from this iteration (guaranteed to have a result since
        // check_time doesn't stop during depth 1)
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

        if !searcher.hot.stopped && !searcher.silent {
            searcher.print_info(depth, score);
        }

        if searcher.hot.stopped || best_score.abs() > MATE_SCORE {
            break;
        }

        if searcher.hot.time_limit_ms != u128::MAX {
            let elapsed = searcher.hot.timer.elapsed_ms();
            let limit = searcher.hot.time_limit_ms;

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

    // Probe TT for best move from previous search (uses shared TT if configured)
    let rule50_count = game.halfmove_clock;
    if let Some((_, best)) = super::probe_tt_with_shared(
        searcher,
        &super::ProbeContext {
            hash,
            alpha,
            beta,
            depth,
            ply: 0,
            rule50_count,
            rule_limit: searcher.move_rule_limit,
        },
    ) {
        tt_move = best;
    }

    let in_check = game.is_in_check();

    let mut moves: MoveList = MoveList::new();
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
            fallback_move = Some(*m);
        }

        let score = if legal_moves == 1 {
            -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: depth - 1,
                ply: 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                noise_amp,
            })
        } else {
            let mut s = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: depth - 1,
                ply: 1,
                alpha: -alpha - 1,
                beta: -alpha,
                allow_null: true,
                noise_amp,
            });
            if s > alpha && s < beta {
                s = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: depth - 1,
                    ply: 1,
                    alpha: -beta,
                    beta: -alpha,
                    allow_null: true,
                    noise_amp,
                });
            }
            s
        };

        game.undo_move(m, undo);

        searcher.prev_move_stack[0] = prev_entry_backup;

        if searcher.hot.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(*m);

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
        // Determine if this is a loss:
        // 1. In check AND must escape check (our win condition is checkmate) → checkmate
        // 2. No pieces left (relevant for allpiecescaptured variants) → loss
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        return if checkmate || no_pieces {
            -MATE_VALUE
        } else {
            0 // Stalemate
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
    super::store_tt_with_shared(
        searcher,
        &super::StoreContext {
            hash,
            depth,
            flag: tt_flag,
            score: best_score,
            best_move,
            ply: 0,
        },
    );

    best_score
}

fn negamax_noisy(ctx: &mut NegamaxNoisyContext) -> i32 {
    let depth = ctx.depth;
    let ply = ctx.ply;
    let mut alpha = ctx.alpha;
    let mut beta = ctx.beta;
    let allow_null = ctx.allow_null;
    let noise_amp = ctx.noise_amp;

    // Re-borrow searcher and game to avoid moving them out of ctx which is borrowed
    let searcher = &mut *ctx.searcher;
    let game = &mut *ctx.game;
    // Node type classification for search behavior
    let is_pv = beta > alpha + 1;
    let cut_node = !is_pv && beta == alpha + 1;
    let all_node = !is_pv && !cut_node;

    // Leaf node: transition to quiescence search
    if depth == 0 {
        return quiescence_noisy(searcher, game, ply, alpha, beta, noise_amp);
    }

    // Cap depth to prevent overflow
    let mut depth = depth.min(MAX_PLY - 1);

    // Safety check
    if ply >= MAX_PLY - 1 {
        return evaluate_with_noise(game, noise_amp);
    }

    // Initialize node state
    let in_check = game.is_in_check();
    searcher.hot.nodes += 1;
    searcher.pv_length[ply] = 0;

    // Time management and selective depth tracking
    if searcher.check_time() {
        return 0;
    }
    if is_pv && ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
    }

    // Non-root node: check for draws and mate distance pruning
    if ply > 0 {
        if game.is_fifty() || game.is_repetition(ply) {
            return -repetition_penalty();
        }

        // Royal capture loss: if our king was just captured (RoyalCapture/AllRoyalsCaptured variants)
        if game.has_lost_by_royal_capture() {
            return -MATE_VALUE + ply as i32;
        }

        let mate_score = MATE_VALUE - ply as i32;
        alpha = alpha.max(-mate_score);
        beta = beta.min(mate_score - 1);
        if alpha >= beta {
            return alpha;
        }
    }

    // Save original bounds for TT flag determination
    let alpha_orig = alpha;
    let beta_orig = beta;

    // Track reduction from parent ply for hindsight adjustment
    let prior_reduction = if ply > 0 {
        searcher.reduction_stack[ply - 1]
    } else {
        0
    };

    // Transposition table probe
    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;

    let rule50_count = game.halfmove_clock;
    if let Some((score, best)) = super::probe_tt_with_shared(
        searcher,
        &super::ProbeContext {
            hash,
            alpha,
            beta,
            depth,
            ply,
            rule50_count,
            rule_limit: searcher.move_rule_limit,
        },
    ) {
        tt_move = best;
        // Stockfish's "graph history interaction" workaround:
        let rule_limit = searcher.move_rule_limit as u32;
        if !is_pv
            && score != super::INFINITY + 1
            && game.halfmove_clock < rule_limit.saturating_sub(4)
            && game.repetition == 0
        {
            return score;
        }
    }

    // Static evaluation for pruning decisions
    // Only use -MATE_VALUE if we're in check AND must escape (checkmate win condition)
    let must_escape = in_check && game.must_escape_check();
    let static_eval = if must_escape {
        -MATE_VALUE + ply as i32
    } else {
        evaluate_with_noise(game, noise_amp)
    };

    searcher.eval_stack[ply] = static_eval;

    // Position improving heuristic
    let mut improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true
    };

    // Opponent worsening heuristic
    let opponent_worsening = if ply >= 1 && !in_check {
        static_eval > -searcher.eval_stack[ply - 1]
    } else {
        false
    };

    // Hindsight depth adjustment
    if !in_check && ply > 0 {
        let prev_eval = searcher.eval_stack[ply - 1];
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2 && depth >= 2 && static_eval + prev_eval > 173 {
            depth = depth.saturating_sub(1);
        }
    }

    // When in check, skip all pruning
    if !in_check {
        // Razoring: if static eval is very low, drop to qsearch
        if !is_pv && static_eval < alpha - 485 - 281 * (depth * depth) as i32 {
            return quiescence_noisy(searcher, game, ply, alpha, beta, noise_amp);
        }

        // Reverse futility pruning
        if !is_pv && depth < 14 {
            let futility_mult = 76;
            let futility_margin = futility_mult * depth as i32
                - if improving {
                    2474 * futility_mult / 1024
                } else {
                    0
                }
                - if opponent_worsening {
                    331 * futility_mult / 1024
                } else {
                    0
                };

            if static_eval - futility_margin >= beta && static_eval >= beta {
                return (2 * beta + static_eval) / 3;
            }
        }

        // Null move pruning
        if cut_node && allow_null && depth >= nmp_min_depth() {
            let nmp_margin = static_eval - (18 * depth as i32) + 350;
            if nmp_margin >= beta && game.has_non_pawn_material(game.turn) {
                let saved_ep = game.en_passant;
                game.make_null_move();

                let r = 7 + depth / 3;
                let null_score = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: depth.saturating_sub(r),
                    ply: ply + 1,
                    alpha: -beta,
                    beta: -beta + 1,
                    allow_null: false,
                    noise_amp,
                });

                game.unmake_null_move();
                game.en_passant = saved_ep;

                if searcher.hot.stopped {
                    return 0;
                }

                if null_score >= beta {
                    return null_score;
                }
            }
        }

        // Update improving flag
        improving = improving || static_eval >= beta;

        // Internal iterative reductions
        if !all_node && depth >= 6 && tt_move.is_none() && prior_reduction <= 3 {
            depth -= 1;
        }
    }

    let mut moves: MoveList = MoveList::new();
    std::mem::swap(&mut moves, &mut searcher.move_buffers[ply]);
    game.get_legal_moves_into(&mut moves);

    sort_moves(searcher, game, &mut moves, ply, &tt_move);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    let mut quiets_searched: MoveList = MoveList::new();

    // New depth for child nodes
    let new_depth = depth.saturating_sub(1);

    for m in &moves {
        let captured_piece = game.board.get_piece(m.to.x, m.to.y);
        let is_capture = captured_piece.is_some_and(|p| !p.piece_type().is_neutral_type());
        let captured_type = captured_piece.map(|p| p.piece_type());
        let is_promotion = m.promotion.is_some();

        // Check if this move gives check (O(1) for knights/pawns)
        let gives_check = super::movegen::StagedMoveGen::move_gives_check_fast(game, m);

        // In-move pruning at shallow depths
        if !is_pv && game.has_non_pawn_material(game.turn) && best_score > -MATE_SCORE {
            // LMP: (3+d²) / (2 if not improving)
            let improving_div = if improving { 1 } else { 2 };
            let lmp_count = (3 + depth * depth) / improving_div;
            if legal_moves >= lmp_count && !is_capture && !is_promotion && !gives_check {
                continue;
            }

            let lmr_depth = new_depth as i32;

            if is_capture || gives_check {
                // Capture/check pruning
                if let Some(cap_type) = captured_type {
                    let capt_hist =
                        searcher.capture_history[m.piece.piece_type() as usize][cap_type as usize];

                    // Capture futility
                    if !gives_check && lmr_depth < 7 {
                        let cap_value = get_piece_value(cap_type);
                        let futility_value = static_eval
                            + 232
                            + 217 * lmr_depth
                            + cap_value
                            + 131 * capt_hist / 1024;
                        if futility_value <= alpha {
                            continue;
                        }
                    }

                    // SEE pruning for captures
                    let see_margin = (166 * depth as i32 + capt_hist / 29).max(0);
                    let see_value = static_exchange_eval(game, m);
                    if see_value < -see_margin {
                        continue;
                    }
                }
            } else {
                // Quiet move pruning
                let hist_idx = hash_move_dest(m);
                let history = searcher.history[m.piece.piece_type() as usize][hist_idx];

                // History-based pruning
                if history < -4083 * depth as i32 {
                    continue;
                }

                // Adjust LMR depth based on history
                let adj_lmr_depth = (lmr_depth + history / 3208).max(0);

                // Quiet futility
                if !in_check && adj_lmr_depth < 13 {
                    let no_best = if best_move.is_none() { 161 } else { 0 };
                    let futility_value = static_eval + 42 + no_best + 127 * adj_lmr_depth;
                    if futility_value <= alpha {
                        if best_score <= futility_value {
                            best_score = futility_value;
                        }
                        continue;
                    }
                }

                // SEE pruning for quiets
                let see_threshold = -25 * adj_lmr_depth * adj_lmr_depth;
                let see_value = static_exchange_eval(game, m);
                if see_value < see_threshold {
                    continue;
                }
            }
        }

        let undo = game.make_move(m);

        if game.is_move_illegal() {
            game.undo_move(m, undo);
            continue;
        }

        if !is_capture && !is_promotion {
            quiets_searched.push(*m);
        }

        let prev_entry_backup = searcher.prev_move_stack[ply];
        let from_hash = hash_move_from(m);
        let to_hash = hash_move_dest(m);
        searcher.prev_move_stack[ply] = (from_hash, to_hash);

        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        searcher.move_history[ply] = Some(*m);
        searcher.moved_piece_history[ply] = m.piece.piece_type() as u8;

        legal_moves += 1;

        let score;
        if legal_moves == 1 {
            score = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: depth - 1,
                ply: ply + 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                noise_amp,
            });
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

            let mut s = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: search_depth,
                ply: ply + 1,
                alpha: -alpha - 1,
                beta: -alpha,
                allow_null: true,
                noise_amp,
            });

            if s > alpha && (reduction > 0 || s < beta) {
                s = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: depth - 1,
                    ply: ply + 1,
                    alpha: -beta,
                    beta: -alpha,
                    allow_null: true,
                    noise_amp,
                });
            }
            score = s;
        }

        game.undo_move(m, undo);

        searcher.prev_move_stack[ply] = prev_entry_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.hot.stopped {
            std::mem::swap(&mut searcher.move_buffers[ply], &mut moves);
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(*m);

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

                // Low Ply History update (Stockfish-style)
                searcher.update_low_ply_history(ply, idx, bonus);

                for quiet in &quiets_searched {
                    let qidx = hash_move_dest(quiet);
                    if quiet.piece.piece_type() == m.piece.piece_type() && qidx == idx {
                        continue;
                    }
                    searcher.update_history(quiet.piece.piece_type(), qidx, -bonus);
                    // Penalize other quiets in low ply history too
                    searcher.update_low_ply_history(ply, qidx, -bonus);
                }

                searcher.killers[ply][1] = searcher.killers[ply][0];
                searcher.killers[ply][0] = Some(*m);

                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (m.piece.piece_type() as u8, m.to.x as i16, m.to.y as i16);
                    }
                }

                // Continuation history update (Stockfish indices: 0, 1, 2, 3, 5)
                for &plies_ago in &[0usize, 1, 2, 3, 5] {
                    if ply > plies_ago
                        && let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1]
                    {
                        let prev_piece = searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                        if prev_piece < 16 {
                            let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);

                            for quiet in &quiets_searched {
                                let q_from_hash = hash_coord_32(quiet.from.x, quiet.from.y);
                                let q_to_hash = hash_coord_32(quiet.to.x, quiet.to.y);
                                let is_best = quiet.from == m.from && quiet.to == m.to;

                                let entry = &mut searcher.cont_history[prev_piece][prev_to_hash]
                                    [q_from_hash][q_to_hash];
                                if is_best {
                                    *entry += adj - *entry * adj / max_history;
                                } else {
                                    *entry += -adj - *entry * adj / max_history;
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

    super::store_tt_with_shared(
        searcher,
        &super::StoreContext {
            hash,
            depth,
            flag,
            score: best_score,
            best_move,
            ply,
        },
    );

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
    // CRITICAL: Check for max ply BEFORE any array accesses to prevent out-of-bounds
    // This must be the very first check to avoid panics when ply >= MAX_PLY
    if ply >= MAX_PLY - 1 {
        return evaluate_with_noise(game, noise_amp);
    }

    searcher.hot.nodes += 1;
    searcher.hot.qnodes += 1;

    if ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
    }

    if searcher.check_time() {
        return 0;
    }

    // Royal capture loss: if our king was just captured (RoyalCapture/AllRoyalsCaptured variants)
    // Zero overhead: has_lost_by_royal_capture returns false immediately for Checkmate variants
    if game.has_lost_by_royal_capture() {
        return -MATE_VALUE + ply as i32;
    }

    let in_check = game.is_in_check();
    // Only treat check specially if we must escape (checkmate-based win condition)
    let must_escape = in_check && game.must_escape_check();

    let stand_pat = if must_escape {
        -MATE_VALUE + ply as i32
    } else {
        evaluate_with_noise(game, noise_amp)
    };

    if !must_escape {
        if stand_pat >= beta {
            return beta;
        }

        if alpha < stand_pat {
            alpha = stand_pat;
        }
    }

    let mut tactical_moves: MoveList = MoveList::new();
    std::mem::swap(&mut tactical_moves, &mut searcher.move_buffers[ply]);

    if must_escape {
        // In check and must escape - only generate evasion moves
        game.get_evasion_moves_into(&mut tactical_moves);
    } else {
        // Normal quiescence: generate captures only
        let enemy_king_pos = match game.turn.opponent() {
            PlayerColor::White => game.white_king_pos,
            PlayerColor::Black => game.black_king_pos,
            _ => None,
        };
        let ctx = MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: enemy_king_pos.as_ref(),
        };
        get_quiescence_captures(&game.board, game.turn, &ctx, &mut tactical_moves);
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

        if searcher.hot.stopped {
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
        // Determine if this is a loss:
        // 1. In check AND must escape check (our win condition is checkmate) → checkmate
        // 2. No pieces left (relevant for allpiecescaptured variants) → loss
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        if checkmate || no_pieces {
            std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);
            return -MATE_VALUE + ply as i32;
        }
    }

    std::mem::swap(&mut searcher.move_buffers[ply], &mut tactical_moves);

    best_score
}
