//! Noisy search module - adds stochastic noise to evaluation for SPRT testing.
//! This module largely mirrors search.rs with noise injection.
//! Coverage exclusion is handled via CI ignore flags.

use super::movegen::StagedMoveGen;
use super::ordering::*;
use super::*;
use crate::board::PlayerColor;
use crate::evaluation::{evaluate, get_piece_value};
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
    pub node_type: NodeType,
}

thread_local! {
    static NOISE_SEED: RefCell<u64> = const { RefCell::new(0) };
}

pub fn reset_noise_seed(seed: u64) {
    NOISE_SEED.with(|c| *c.borrow_mut() = seed);
}

#[inline]
fn apply_noise(base: i32, game: &GameState, noise_amp: i32) -> i32 {
    if noise_amp <= 0 {
        return base;
    }
    if base.abs() >= MATE_SCORE {
        return base;
    }

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

#[inline]
fn evaluate_with_noise(game: &GameState, noise_amp: i32) -> i32 {
    let base = evaluate(game);
    apply_noise(base, game, noise_amp)
}

pub fn get_best_move_with_noise(
    game: &mut GameState,
    max_depth: usize,
    time_limit_ms: u128,
    noise_amp: i32,
    silent: bool,
) -> Option<(Move, i32, SearchStats)> {
    game.recompute_piece_counts();
    game.recompute_correction_hashes();

    if noise_amp <= 0 {
        return super::get_best_move(game, max_depth, time_limit_ms, silent);
    }

    super::GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();
        let searcher = opt.get_or_insert_with(|| super::Searcher::new(time_limit_ms));

        searcher.new_search();
        searcher.hot.time_limit_ms = time_limit_ms;
        searcher.silent = silent;
        searcher.hot.timer.reset();

        searcher.set_corrhist_mode(game);
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        let result = search_with_searcher_noisy(searcher, game, max_depth, noise_amp);
        let stats = super::build_search_stats(searcher);
        result.map(|(m, eval)| (m, eval, stats))
    })
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

        if searcher.hot.min_depth_required == 0
            && searcher.hot.timer.elapsed_ms() >= searcher.hot.time_limit_ms
        {
            searcher.hot.stopped = true;
            break;
        }

        let score = negamax_root_noisy(searcher, game, depth, -INFINITY, INFINITY, noise_amp);

        if depth == 1 {
            searcher.hot.min_depth_required = 0;
        }

        if let Some(pv_move) = searcher.pv_table[0] {
            best_move = Some(pv_move);
            searcher.best_move_root = Some(pv_move);

            if !searcher.hot.stopped {
                best_score = score;
                searcher.prev_score = score;
            }

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
                if has_prev_iter_score
                    && best_score - prev_iter_score > super::params::aspiration_max_window()
                {
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

        let score;
        if legal_moves == 1 {
            score = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: depth - 1,
                ply: 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                noise_amp,
                node_type: NodeType::PV,
            });
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
                node_type: NodeType::Cut,
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
                    node_type: NodeType::PV,
                });
            }
            score = s;
        }

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
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        return if checkmate || no_pieces {
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
    let node_type = ctx.node_type;

    let searcher = &mut *ctx.searcher;
    let game = &mut *ctx.game;
    let is_pv = node_type == NodeType::PV;
    let cut_node = node_type == NodeType::Cut;
    let all_node = node_type == NodeType::All;

    if depth == 0 {
        return quiescence_noisy(searcher, game, ply, alpha, beta, noise_amp);
    }

    let mut depth = depth.min(MAX_PLY - 1);
    if ply >= MAX_PLY - 1 {
        return evaluate_with_noise(game, noise_amp);
    }

    let in_check = game.is_in_check();
    searcher.hot.nodes += 1;
    searcher.pv_length[ply] = 0;
    searcher.killers[ply + 1][0] = None;
    searcher.killers[ply + 1][1] = None;

    if searcher.check_time() {
        return 0;
    }
    if is_pv && ply > searcher.hot.seldepth {
        searcher.hot.seldepth = ply;
    }

    if ply > 0 {
        if game.is_fifty() || game.is_repetition(ply) {
            return -super::params::repetition_penalty();
        }
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

    let alpha_orig = alpha;
    let _beta_orig = beta;
    let prior_reduction = if ply > 0 {
        searcher.reduction_stack[ply - 1]
    } else {
        0
    };

    let hash = TranspositionTable::generate_hash(game);
    let mut tt_move: Option<Move> = None;
    let mut tt_value: Option<i32> = None;

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
        tt_value = Some(score);

        let rule_limit = searcher.move_rule_limit as u32;
        if !is_pv
            && score != super::INFINITY + 1
            && score.abs() < super::MATE_SCORE
            && game.halfmove_clock < rule_limit.saturating_sub(4)
            && game.repetition == 0
        {
            return score;
        }
    }

    let must_escape = in_check && game.must_escape_check();
    let (static_eval, raw_eval) = if must_escape {
        (-MATE_VALUE + ply as i32, -MATE_VALUE + ply as i32)
    } else {
        let raw = evaluate(game);
        let prev_move_hash = if ply > 0 {
            let (from_hash, to_hash) = searcher.prev_move_stack[ply - 1];
            from_hash ^ to_hash
        } else {
            0
        };
        let adjusted = searcher.adjusted_eval(game, raw, prev_move_hash);
        let noisy = apply_noise(adjusted, game, noise_amp);
        (noisy, raw)
    };

    searcher.eval_stack[ply] = static_eval;

    let mut improving = if ply >= 2 && !in_check {
        static_eval > searcher.eval_stack[ply - 2]
    } else {
        true
    };

    let opponent_worsening = if ply >= 1 && !in_check {
        static_eval > -searcher.eval_stack[ply - 1]
    } else {
        false
    };

    if !in_check && ply > 0 {
        let prev_eval = searcher.eval_stack[ply - 1];
        if prior_reduction >= 3 && !opponent_worsening {
            depth += 1;
        }
        if prior_reduction >= 2 && depth >= 2 && static_eval + prev_eval > 173 {
            depth = depth.saturating_sub(1);
        }
    }

    if !in_check {
        if !is_pv && static_eval < alpha - 485 - 281 * (depth * depth) as i32 {
            let v = quiescence_noisy(searcher, game, ply, alpha, beta, noise_amp);
            if v <= alpha {
                return v;
            }
        }

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

        if node_type != NodeType::PV && allow_null && depth >= super::params::nmp_min_depth() {
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
                    node_type: NodeType::Cut,
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

        improving = improving || static_eval >= beta;
        if !all_node && depth >= 6 && tt_move.is_none() && prior_reduction <= 3 {
            depth -= 1;
        }
    }

    let prob_cut_beta = beta + 235 - if improving { 63 } else { 0 };
    if !is_pv
        && !in_check
        && depth >= 5
        && beta.abs() < super::MATE_SCORE
        && !tt_value.is_some_and(|v| v < prob_cut_beta)
    {
        let mut prob_cut_depth = (depth as i32 - 4 - (static_eval - beta) / 315).max(0) as usize;
        if prob_cut_depth > depth {
            prob_cut_depth = depth;
        }

        let mut captures: MoveList = MoveList::new();
        let ctx = crate::moves::MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: game.enemy_king_pos(),
        };
        crate::moves::get_quiescence_captures(&game.board, game.turn, &ctx, &mut captures);
        super::sort_captures(game, &mut captures);

        for m in &captures {
            if super::static_exchange_eval(game, m) < prob_cut_beta - static_eval {
                continue;
            }
            let fast_legal = game.is_legal_fast(m, in_check);
            if let Ok(false) = fast_legal {
                continue;
            }
            let undo = game.make_move(m);
            if fast_legal.is_err() && game.is_move_illegal() {
                game.undo_move(m, undo);
                continue;
            }

            let mut val = -quiescence_noisy(
                searcher,
                game,
                ply + 1,
                -prob_cut_beta,
                -prob_cut_beta + 1,
                noise_amp,
            );
            if val >= prob_cut_beta {
                val = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: prob_cut_depth,
                    ply: ply + 1,
                    alpha: -prob_cut_beta,
                    beta: -prob_cut_beta + 1,
                    allow_null: true,
                    noise_amp,
                    node_type: NodeType::Cut,
                });
            }
            game.undo_move(m, undo);
            if searcher.hot.stopped {
                return 0;
            }
            if val >= prob_cut_beta {
                super::store_tt_with_shared(
                    searcher,
                    &super::StoreContext {
                        hash,
                        depth: prob_cut_depth + 1,
                        flag: TTFlag::LowerBound,
                        score: val,
                        best_move: Some(*m),
                        ply,
                    },
                );
                return val;
            }
        }
    }

    let se_conditions = if depth >= 6 && !in_check {
        tt_move.as_ref().and_then(|_| {
            searcher.tt.probe_for_singular(hash, ply).and_then(
                |(tt_flag, tt_depth, tt_score, _)| {
                    let depth_val = tt_depth as usize;
                    if (tt_flag == TTFlag::LowerBound || tt_flag == TTFlag::Exact)
                        && depth_val >= depth.saturating_sub(3)
                        && tt_score.abs() < MATE_SCORE
                    {
                        Some((tt_score, (depth - 1) / 2))
                    } else {
                        None
                    }
                },
            )
        })
    } else {
        None
    };

    let mut movegen = StagedMoveGen::new(tt_move, ply, searcher, game);

    let mut best_score = -INFINITY;
    let mut best_move: Option<Move> = None;
    let mut legal_moves = 0;
    let mut quiets_searched: MoveList = MoveList::new();
    let mut moves_searched = 0;

    let new_depth = depth.saturating_sub(1);

    while let Some(m) = movegen.next(game, searcher) {
        let p_type = m.piece.piece_type();
        let is_strict_capture = game.board.is_occupied(m.to.x, m.to.y);
        let gives_check = StagedMoveGen::move_gives_check_fast(game, &m);

        if !is_pv && game.has_non_pawn_material(game.turn) && best_score > -MATE_SCORE {
            let improving_div = if improving { 1 } else { 2 };
            let lmp_count = (3 + depth * depth) / improving_div;
            if legal_moves >= lmp_count {
                movegen.skip_quiet_moves();
            }

            let lmr_depth = new_depth as i32;
            if is_strict_capture || gives_check {
                if let Some(cap_type) = game.board.get_piece(m.to.x, m.to.y).map(|p| p.piece_type())
                {
                    let capt_hist = searcher.capture_history[p_type as usize][cap_type as usize];
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
                    let see_margin = (166 * depth as i32 + capt_hist / 29).max(0);
                    if static_exchange_eval(game, &m) < -see_margin {
                        continue;
                    }
                }
            } else {
                let hist_idx = hash_move_dest(&m);
                let history = searcher.history[p_type as usize][hist_idx];
                if history < -4083 * depth as i32 {
                    continue;
                }

                let adj_lmr_depth = (lmr_depth + history / 3208).max(0);
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

                let see_threshold = -25 * adj_lmr_depth * adj_lmr_depth;
                if static_exchange_eval(game, &m) < see_threshold {
                    continue;
                }
            }
        }

        let fast_legal = game.is_legal_fast(&m, in_check);
        if let Ok(false) = fast_legal {
            continue;
        }

        let mut undo = game.make_move(&m);
        if fast_legal.is_err() && game.is_move_illegal() {
            game.undo_move(&m, undo);
            continue;
        }

        if !is_strict_capture && m.promotion.is_none() {
            quiets_searched.push(m);
        }

        let prev_entry_backup = searcher.prev_move_stack[ply];
        searcher.prev_move_stack[ply] = (hash_move_from(&m), hash_move_dest(&m));

        let move_history_backup = searcher.move_history[ply].take();
        let piece_history_backup = searcher.moved_piece_history[ply];
        searcher.move_history[ply] = Some(m);
        searcher.moved_piece_history[ply] = p_type as u8;

        legal_moves += 1;
        moves_searched += 1;

        // Singular Extension
        let mut extension: usize = 0;
        let is_tt_move = tt_move
            .filter(|tt_m| m.from == tt_m.from && m.to == tt_m.to && m.promotion == tt_m.promotion)
            .is_some();
        if let Some((tt_score, singular_depth)) = se_conditions.filter(|_| is_tt_move && !is_pv) {
            let tt_history_adj = searcher.tt_move_history / 150;
            let singular_beta = tt_score - (depth as i32) * 3 + tt_history_adj;

            game.undo_move(&m, undo);
            let mut se_gen = StagedMoveGen::with_exclusion(None, ply, searcher, game, m);
            let mut se_moves_checked = 0;
            const SE_MAX_MOVES: usize = 6;
            let mut se_extension_found = true;

            while let Some(se_m) = se_gen.next(game, searcher) {
                if se_moves_checked >= SE_MAX_MOVES {
                    break;
                }

                let fast_legal = game.is_legal_fast(&se_m, in_check);
                if let Ok(false) = fast_legal {
                    continue;
                }
                let se_undo = game.make_move(&se_m);
                if fast_legal.is_err() && game.is_move_illegal() {
                    game.undo_move(&se_m, se_undo);
                    continue;
                }

                se_moves_checked += 1;
                let se_score = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: singular_depth,
                    ply: ply + 1,
                    alpha: -singular_beta,
                    beta: -singular_beta + 1,
                    allow_null: false,
                    noise_amp,
                    node_type: NodeType::Cut,
                });
                game.undo_move(&se_m, se_undo);
                if searcher.hot.stopped {
                    return best_score;
                }

                if se_score >= singular_beta {
                    se_extension_found = false;
                    break;
                }
            }
            if se_extension_found {
                extension = 1;
            }
            undo = game.make_move(&m);
        }

        let mut reduction = 0;
        if depth >= super::params::lmr_min_depth()
            && moves_searched >= super::params::lmr_min_moves()
            && !in_check
            && !is_strict_capture
            && !(m.promotion.is_some())
        {
            reduction = 1
                + ((moves_searched as f32).ln() * (depth as f32).ln()
                    / super::params::lmr_divisor() as f32) as i32;
            if !improving {
                reduction += 1;
            }
            let h_idx = hash_move_dest(&m);
            let h_score = searcher.history[p_type as usize][h_idx];
            if h_score > 2000 && reduction > 0 {
                reduction -= 1;
            }
            if searcher.tt_move_history < -1000 && reduction > 0 {
                reduction -= 1;
            }
            reduction = reduction.clamp(0, depth as i32 - 2);
        }

        let mut next_depth = depth as i32 - 1 + extension as i32 - reduction;

        if !in_check
            && !is_pv
            && !is_strict_capture
            && m.promotion.is_none()
            && !gives_check
            && depth <= super::params::hlp_max_depth()
            && moves_searched >= super::params::hlp_min_moves()
            && best_score > -MATE_SCORE
        {
            let idx = hash_move_dest(&m);
            let value = searcher.history[p_type as usize][idx];
            if value < super::params::hlp_history_reduce() {
                next_depth -= 1;
                if next_depth <= 0 && value < super::params::hlp_history_leaf() {
                    game.undo_move(&m, undo);
                    searcher.prev_move_stack[ply] = prev_entry_backup;
                    searcher.move_history[ply] = move_history_backup;
                    searcher.moved_piece_history[ply] = piece_history_backup;
                    continue;
                }
            }
        }

        let search_depth = if next_depth <= 0 {
            0
        } else {
            next_depth as usize
        };

        let mut score;
        if moves_searched == 1 {
            score = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: (depth as i32 - 1 + extension as i32) as usize,
                ply: ply + 1,
                alpha: -beta,
                beta: -alpha,
                allow_null: true,
                noise_amp,
                node_type: if cut_node {
                    NodeType::All
                } else {
                    NodeType::PV
                },
            });
        } else {
            let child_type = if cut_node {
                NodeType::All
            } else {
                NodeType::Cut
            };
            searcher.reduction_stack[ply] = reduction;
            score = -negamax_noisy(&mut NegamaxNoisyContext {
                searcher,
                game,
                depth: search_depth,
                ply: ply + 1,
                alpha: -alpha - 1,
                beta: -alpha,
                allow_null: true,
                noise_amp,
                node_type: child_type,
            });
            if score > alpha && (reduction > 0 || score < beta) {
                let research_type = if is_pv { NodeType::PV } else { child_type };
                score = -negamax_noisy(&mut NegamaxNoisyContext {
                    searcher,
                    game,
                    depth: (depth as i32 - 1 + extension as i32) as usize,
                    ply: ply + 1,
                    alpha: -beta,
                    beta: -alpha,
                    allow_null: true,
                    noise_amp,
                    node_type: research_type,
                });
            }
        }

        game.undo_move(&m, undo);
        searcher.prev_move_stack[ply] = prev_entry_backup;
        searcher.move_history[ply] = move_history_backup;
        searcher.moved_piece_history[ply] = piece_history_backup;

        if searcher.hot.stopped {
            return best_score;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m);
            if score > alpha {
                alpha = score;
                let ply_base = ply * MAX_PLY;
                let child_base = (ply + 1) * MAX_PLY;
                searcher.pv_table[ply_base] = Some(m);
                let child_len = searcher.pv_length[ply + 1];
                for j in 0..child_len {
                    searcher.pv_table[ply_base + 1 + j] = searcher.pv_table[child_base + j];
                }
                searcher.pv_length[ply] = child_len + 1;
            }
        }

        if alpha >= beta {
            if !is_strict_capture {
                let idx = hash_move_dest(&m);
                let bonus_base = super::params::history_bonus_base() as i32;
                let bonus_sub = super::params::history_bonus_sub() as i32;
                let bonus: i32 = bonus_base * (depth as i32) - bonus_sub;

                let cap = super::params::history_bonus_cap() as i32;
                let adj = bonus.min(cap);
                let max_history: i32 = super::params::DEFAULT_HISTORY_MAX_GRAVITY;

                searcher.update_history(p_type, idx, bonus);
                searcher.update_low_ply_history(ply, idx, bonus);

                for quiet in &quiets_searched {
                    let qidx = hash_move_dest(quiet);
                    if qidx != idx && quiet.piece.piece_type() == m.piece.piece_type() {
                        continue;
                    }
                    if qidx == idx {
                        continue;
                    }
                    searcher.update_history(quiet.piece.piece_type(), qidx, -bonus);
                    searcher.update_low_ply_history(ply, qidx, -bonus);
                }
                searcher.killers[ply][1] = searcher.killers[ply][0];
                searcher.killers[ply][0] = Some(m);
                if ply > 0 {
                    let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                    if prev_from_hash < 256 && prev_to_hash < 256 {
                        searcher.countermoves[prev_from_hash][prev_to_hash] =
                            (p_type as u8, m.to.x as i16, m.to.y as i16);
                    }
                }
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
            } else if let Some(cap_type) =
                game.board.get_piece(m.to.x, m.to.y).map(|p| p.piece_type())
            {
                let bonus = 8 * ((depth * depth) as i32);
                let e = &mut searcher.capture_history[p_type as usize][cap_type as usize];
                *e += bonus - *e * bonus / super::params::DEFAULT_HISTORY_MAX_GRAVITY;
            }
            break;
        } else if let Some(cap_type) = game.board.get_piece(m.to.x, m.to.y).map(|p| p.piece_type())
        {
            let malus = 2 * (depth as i32);
            let e = &mut searcher.capture_history[p_type as usize][cap_type as usize];
            *e += -malus - *e * malus / super::params::DEFAULT_HISTORY_MAX_GRAVITY;
        }
    }

    if legal_moves == 0 {
        let checkmate = in_check && game.must_escape_check();
        let no_pieces = !game.has_pieces(game.turn);
        return if checkmate || no_pieces {
            -MATE_VALUE + ply as i32
        } else {
            0
        };
    }

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

    if !in_check && best_move.is_some() {
        let bm = best_move.unwrap();
        let is_quiet = !game.board.is_occupied(bm.to.x, bm.to.y) && bm.promotion.is_none();

        let should_update = match tt_flag {
            TTFlag::LowerBound => best_score >= raw_eval,
            TTFlag::UpperBound => best_score <= raw_eval,
            TTFlag::Exact => true,
            TTFlag::None => false,
        };

        if is_quiet && should_update {
            let prev_move_idx = if ply > 0 {
                let (from_hash, to_hash) = searcher.prev_move_stack[ply - 1];
                from_hash ^ to_hash
            } else {
                0
            };
            searcher.update_correction_history(
                game,
                depth,
                raw_eval,
                best_score,
                true,
                false,
                prev_move_idx,
            );
        }
    }

    if node_type != NodeType::PV
        && let Some(ref bm) = best_move
    {
        let tt_move_matched = tt_move
            .as_ref()
            .is_some_and(|tm| tm.from == bm.from && tm.to == bm.to);
        let delta: i32 = if tt_move_matched { 809 } else { -865 };
        let max_tt_history = 8192;
        searcher.tt_move_history += delta - searcher.tt_move_history * delta.abs() / max_tt_history;
    }

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
    if game.has_lost_by_royal_capture() {
        return -MATE_VALUE + ply as i32;
    }

    let in_check = game.is_in_check();
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
        game.get_evasion_moves_into(&mut tactical_moves);
    } else {
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

        let fast_legal = game.is_legal_fast(m, in_check);
        if let Ok(false) = fast_legal {
            continue;
        }
        let undo = game.make_move(m);
        if fast_legal.is_err() && game.is_move_illegal() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    #[test]
    fn test_noisy_search_smoke() {
        use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
        let mut game = GameState::default();

        // Setup minimal board: White King at e1 (4,0), Black King at e8 (4,7), White Pawn at e2 (4,1)
        game.board
            .set_piece(4, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Pawn, PlayerColor::White));

        game.white_king_pos = Some(Coordinate::new(4, 0));
        game.black_king_pos = Some(Coordinate::new(4, 7));
        game.turn = PlayerColor::White;

        game.recompute_piece_counts();
        game.recompute_check_squares();
        game.recompute_hash();

        let moves = game.get_legal_moves();
        println!("Legal moves available: {}", moves.len());
        assert!(!moves.is_empty(), "Should have legal moves");

        // Depth 2, sufficient time, some noise
        let res = get_best_move_with_noise(&mut game, 2, 5000, 10, true);
        if res.is_none() {
            println!("Search returned None!");
        }
        assert!(res.is_some(), "Search should return a result");
        let (m, score, _stats) = res.unwrap();
        println!("Best move: {:?}, score: {}", m, score);
    }
}
