//! Site strength-level system: per-level candidate sampling, human-plausibility
//! filters, and the deep-tactic/hang gates.
use super::*;

pub const MAX_SITE_SKILL: u32 = 8; // Current max skill level on the site
pub const MAX_PV_COUNT: usize = 96; // Safety cap for MultiPV candidate collection

/// Infinite-chess strength limiting needs a much wider candidate set than Stockfish's
/// fixed MultiPV=4. On a large board, the first several moves are often effectively
/// equivalent, so picking one of four does not create a meaningful error.
#[derive(Clone, Copy, Debug)]
struct SkillConfig {
    depth_cap: Option<usize>,
    endgame_depth_bonus: usize,
    mop_up_depth_bonus: usize,
    candidates: usize,
    best_move_chance: f64,
    mean_loss_permille: f64,
    max_loss_permille: i32,
    max_conversion_cp_loss: i32,
    mate_distance_temperature: f64,
    obvious_blunder_chance: f64,
    conversion_odds_multiplier: f64,
    conversion_loss_multiplier: f64,
    /// Chance to trust a move whose payoff only appears deep in the search
    /// (a forced combination invisible at depth 2), rather than avoiding it.
    deep_tactic_chance: f64,
}

/// Site levels 1..7; level 8 bypasses the limiter for the full parallel search.
/// `mean_loss_permille` targets a loss in winning probability rather than centipawns,
/// so weak levels keep giving ground instead of turning near-perfect once ahead.
const SKILL_CONFIGS: [SkillConfig; 7] = [
    SkillConfig {
        depth_cap: Some(3),
        endgame_depth_bonus: 0,
        mop_up_depth_bonus: 2,
        candidates: 24,
        best_move_chance: 0.16,
        mean_loss_permille: 110.0,
        max_loss_permille: 350,
        max_conversion_cp_loss: 600,
        mate_distance_temperature: 8.0,
        obvious_blunder_chance: 0.025,
        conversion_odds_multiplier: 3.0,
        conversion_loss_multiplier: 0.75,
        deep_tactic_chance: 0.05,
    },
    SkillConfig {
        depth_cap: Some(4),
        endgame_depth_bonus: 0,
        mop_up_depth_bonus: 2,
        candidates: 20,
        best_move_chance: 0.23,
        mean_loss_permille: 80.0,
        max_loss_permille: 280,
        max_conversion_cp_loss: 475,
        mate_distance_temperature: 6.0,
        obvious_blunder_chance: 0.020,
        conversion_odds_multiplier: 3.5,
        conversion_loss_multiplier: 0.70,
        deep_tactic_chance: 0.10,
    },
    SkillConfig {
        depth_cap: Some(5),
        endgame_depth_bonus: 1,
        mop_up_depth_bonus: 2,
        candidates: 16,
        best_move_chance: 0.32,
        mean_loss_permille: 58.0,
        max_loss_permille: 220,
        max_conversion_cp_loss: 375,
        mate_distance_temperature: 4.5,
        obvious_blunder_chance: 0.015,
        conversion_odds_multiplier: 4.0,
        conversion_loss_multiplier: 0.65,
        deep_tactic_chance: 0.20,
    },
    SkillConfig {
        depth_cap: Some(7),
        endgame_depth_bonus: 2,
        mop_up_depth_bonus: 3,
        candidates: 12,
        best_move_chance: 0.42,
        mean_loss_permille: 42.0,
        max_loss_permille: 160,
        max_conversion_cp_loss: 285,
        mate_distance_temperature: 3.0,
        obvious_blunder_chance: 0.010,
        conversion_odds_multiplier: 5.0,
        conversion_loss_multiplier: 0.55,
        deep_tactic_chance: 0.35,
    },
    SkillConfig {
        depth_cap: Some(8),
        endgame_depth_bonus: 3,
        mop_up_depth_bonus: 3,
        candidates: 8,
        best_move_chance: 0.54,
        mean_loss_permille: 29.0,
        max_loss_permille: 110,
        max_conversion_cp_loss: 210,
        mate_distance_temperature: 1.8,
        obvious_blunder_chance: 0.006,
        conversion_odds_multiplier: 7.0,
        conversion_loss_multiplier: 0.45,
        deep_tactic_chance: 0.55,
    },
    SkillConfig {
        depth_cap: Some(10),
        endgame_depth_bonus: 3,
        mop_up_depth_bonus: 3,
        candidates: 6,
        best_move_chance: 0.62,
        mean_loss_permille: 22.0,
        max_loss_permille: 80,
        max_conversion_cp_loss: 145,
        mate_distance_temperature: 0.8,
        obvious_blunder_chance: 0.003,
        conversion_odds_multiplier: 9.0,
        conversion_loss_multiplier: 0.35,
        deep_tactic_chance: 0.75,
    },
    SkillConfig {
        depth_cap: Some(14),
        endgame_depth_bonus: 3,
        mop_up_depth_bonus: 3,
        candidates: 2,
        best_move_chance: 0.68,
        mean_loss_permille: 22.0,
        max_loss_permille: 70,
        max_conversion_cp_loss: 90,
        mate_distance_temperature: 0.25,
        obvious_blunder_chance: 0.001,
        conversion_odds_multiplier: 12.0,
        conversion_loss_multiplier: 0.25,
        deep_tactic_chance: 0.90,
    },
];

// Fitted from 11,582 human moves at depth 4. Humans find a clear, depth-stable best
// move far more often, and far less often when it changes between shallow and final
// depth.
const CLEAR_BEST_GAP_PERMILLE: i32 = 50;
const CLEAR_BEST_ODDS_MULTIPLIER: f64 = 3.0;
const UNSTABLE_ODDS_MULTIPLIER: f64 = 0.40;
const MISSED_CLEAR_LOSS_MULTIPLIER: f64 = 1.25;
const SHALLOW_RANK_LOG_PENALTY: f64 = 14.0;
const LEVEL_7_CLEAN_SEARCH_CHANCE: f64 = 0.30;
const CLEAR_CONVERSION_WIN_CHANCE: i32 = 700;
const CONVERSION_SCORE_FLOOR: i32 = 100;

/// Win-probability gain (permille) between the depth-2 shallow score and the
/// final depth needed to count a move as a "deep-only" forced tactic.
const DEEP_TACTIC_SURPRISE_PERMILLE: i32 = 120;

/// Maps an engine score to winning probability in permille, on the deliberately
/// shallow curve the site's review model uses. Mate scores return the endpoints
/// directly, so their sentinels never reach exp().
#[inline]
pub fn score_to_win_chance_permille(score: i32) -> i32 {
    if score > MATE_SCORE {
        return 1000;
    }
    if score < -MATE_SCORE {
        return 0;
    }

    let cp = score.clamp(-1800, 1800) as f64;
    (1000.0 / (1.0 + (-0.003 * cp).exp())).round() as i32
}

fn scale_probability_odds(probability: f64, multiplier: f64) -> f64 {
    if probability <= 0.0 || probability >= 1.0 {
        return probability;
    }
    let odds = probability / (1.0 - probability) * multiplier;
    odds / (1.0 + odds)
}

#[derive(Clone, Copy, Debug)]
struct AdjustedSkillBehavior {
    best_move_chance: f64,
    mean_loss_permille: f64,
    max_loss_permille: i32,
}

fn adjusted_skill_behavior(
    result: &MultiPVResult,
    config: SkillConfig,
    conversion_position: bool,
) -> AdjustedSkillBehavior {
    let top_chance = score_to_win_chance_permille(result.lines[0].score);
    let second_loss = result.lines.get(1).map_or(0, |line| {
        (top_chance - score_to_win_chance_permille(line.score)).max(0)
    });

    let (mut best_move_chance, mut mean_loss_permille) = if result.shallow_best_changed {
        (
            scale_probability_odds(config.best_move_chance, UNSTABLE_ODDS_MULTIPLIER),
            config.mean_loss_permille,
        )
    } else if second_loss >= CLEAR_BEST_GAP_PERMILLE {
        (
            scale_probability_odds(config.best_move_chance, CLEAR_BEST_ODDS_MULTIPLIER),
            config.mean_loss_permille * MISSED_CLEAR_LOSS_MULTIPLIER,
        )
    } else {
        (config.best_move_chance, config.mean_loss_permille)
    };
    let mut max_loss_permille = config.max_loss_permille;

    if conversion_position && top_chance >= CLEAR_CONVERSION_WIN_CHANCE {
        best_move_chance =
            scale_probability_odds(best_move_chance, config.conversion_odds_multiplier);
        mean_loss_permille *= config.conversion_loss_multiplier;
        max_loss_permille =
            (max_loss_permille as f64 * config.conversion_loss_multiplier.sqrt()).round() as i32;
    }

    AdjustedSkillBehavior {
        best_move_chance,
        mean_loss_permille,
        max_loss_permille,
    }
}

fn is_reduced_material_position(game: &GameState) -> bool {
    let total_pieces = game.white_piece_count as usize + game.black_piece_count as usize;
    let current_non_pawns = game.white_piece_count.saturating_sub(game.white_pawn_count) as usize
        + game.black_piece_count.saturating_sub(game.black_pawn_count) as usize;
    let starting_non_pawns =
        game.starting_white_pieces as usize + game.starting_black_pieces as usize;

    total_pieces <= 12 || (starting_non_pawns >= 4 && current_non_pawns * 2 <= starting_non_pawns)
}

/// The side to move, when it is the one converting a won endgame.
fn active_conversion(game: &GameState) -> Option<PlayerColor> {
    let (winner, _) = crate::evaluation::mop_up::active_mop_up(game)?;
    (winner == game.turn).then_some(winner)
}

/// All proven wins form a hard outcome class. Once this level's own capped
/// search has proved mate, strength limiting may choose a slower proof but may
/// not leave the mating set.
fn pick_proven_mate(
    result: &MultiPVResult,
    config: SkillConfig,
    rng: &mut Prng,
) -> Option<(Move, i32)> {
    let best_score = result.lines.first()?.score;
    if best_score <= MATE_SCORE {
        return None;
    }

    // A mate delivered by this move is not a stylistic preference or a
    // calculation detail. Once the capped search has one, every level must
    // finish immediately instead of sampling a slower mating continuation.
    if best_score >= mate_in(1) {
        let best = &result.lines[0];
        return Some((best.mv, best.score));
    }

    let candidates: Vec<(usize, f64)> = result
        .lines
        .iter()
        .enumerate()
        .take_while(|(_, line)| line.score > MATE_SCORE)
        .map(|(idx, line)| {
            let extra_plies = (best_score - line.score).max(0) as f64;
            let shallow_rank = result
                .shallow_order
                .iter()
                .position(|mv| *mv == line.mv)
                .unwrap_or(idx);
            let distance_weight = (-extra_plies / config.mate_distance_temperature).exp();
            let plausibility_weight = ((shallow_rank + 1) as f64).powf(-0.35);
            (idx, distance_weight * plausibility_weight)
        })
        .collect();

    sample_weighted_line(result, &candidates, rng)
}

/// If every available move is a proven loss, skill affects how well the engine
/// resists. The top line survives longest; weak levels are allowed to shorten
/// the loss, but mate scores are never flattened into an arbitrary tie.
fn pick_proven_loss(
    result: &MultiPVResult,
    config: SkillConfig,
    rng: &mut Prng,
) -> Option<(Move, i32)> {
    let best_score = result.lines.first()?.score;
    if best_score >= -MATE_SCORE {
        return None;
    }

    let candidates: Vec<(usize, f64)> = result
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.score < -MATE_SCORE)
        .map(|(idx, line)| {
            let lost_survival_plies = (best_score - line.score).max(0) as f64;
            (
                idx,
                (-lost_survival_plies / config.mate_distance_temperature).exp(),
            )
        })
        .collect();

    sample_weighted_line(result, &candidates, rng)
}

fn sample_weighted_line(
    result: &MultiPVResult,
    candidates: &[(usize, f64)],
    rng: &mut Prng,
) -> Option<(Move, i32)> {
    let &(fallback_idx, _) = candidates.first()?;
    let total_weight: f64 = candidates.iter().map(|(_, weight)| weight).sum();
    let mut sample = rng.next_f64() * total_weight;
    let mut chosen_idx = fallback_idx;
    for &(idx, weight) in candidates {
        chosen_idx = idx;
        sample -= weight;
        if sample <= 0.0 {
            break;
        }
    }
    let chosen = &result.lines[chosen_idx];
    Some((chosen.mv, chosen.score))
}

fn conversion_loss_permille(
    top_score: i32,
    line_score: i32,
    ordinary_loss: i32,
    behavior: AdjustedSkillBehavior,
    config: SkillConfig,
) -> i32 {
    let cp_loss = (top_score - line_score).max(0);
    let scaled_cp_loss = (cp_loss as i64 * behavior.max_loss_permille as i64
        / config.max_conversion_cp_loss.max(1) as i64)
        .min(i32::MAX as i64) as i32;
    ordinary_loss.max(scaled_cp_loss)
}

fn effective_skill_depth(game: &GameState, max_depth: usize, config: SkillConfig) -> usize {
    let reduced_bonus = is_reduced_material_position(game).then_some(config.endgame_depth_bonus);
    let mop_up_bonus = active_conversion(game).map(|_| config.mop_up_depth_bonus);
    let bonus = reduced_bonus
        .into_iter()
        .chain(mop_up_bonus)
        .max()
        .unwrap_or(0);
    config
        .depth_cap
        .map_or(max_depth, |cap| max_depth.min(cap + bonus))
}

fn is_obvious_material_hang(game: &GameState, line: &PVLine, loss_permille: i32) -> bool {
    // Never suppress a sound sacrifice or a best-equivalent move merely because
    // a one-square exchange evaluator cannot see its tactical compensation.
    if loss_permille <= 2 {
        return false;
    }

    let mover_value = game.get_piece_value(line.mv.piece.piece_type(), line.mv.piece.color());
    if mover_value < 450 {
        return false;
    }

    let see = static_exchange_eval(game, &line.mv);
    see <= -300 && -see * 5 >= mover_value * 3
}

/// Halfway to max_depth, so "deep" scales with this search's own depth budget
/// instead of a fixed depth that's trivial once caps rise into double digits.
pub(crate) fn deep_tactic_reference_depth(max_depth: usize) -> usize {
    max_depth
        .div_ceil(2)
        .max(2)
        .min(max_depth.saturating_sub(1).max(1))
}

/// Mirror image of [`is_obvious_material_hang`]: instead of a move that looks
/// worse than it is, this flags one that looks ordinary until a forced line
/// resolves it several plies later. Missing shallow data means no surprise.
fn deep_tactic_surprise_permille(result: &MultiPVResult, mv: Move, final_score: i32) -> i32 {
    let Some(&(_, shallow_score)) = result.deep_ref_scores.iter().find(|(m, _)| *m == mv) else {
        return 0;
    };
    score_to_win_chance_permille(final_score) - score_to_win_chance_permille(shallow_score)
}

fn is_deep_only_tactic(result: &MultiPVResult, mv: Move, final_score: i32) -> bool {
    deep_tactic_surprise_permille(result, mv, final_score) >= DEEP_TACTIC_SURPRISE_PERMILLE
}

/// Selects a plausible move near a sampled loss in winning probability. A point mass
/// at zero lets every level play good moves; errors draw an exponential regret, then
/// pick softly among moves that already ranked well at depth 2.
fn pick_best(
    game: &GameState,
    result: &MultiPVResult,
    config: SkillConfig,
    rng: &mut Prng,
) -> Option<(Move, i32)> {
    if result.lines.is_empty() {
        return None;
    }
    if result.lines[0].score > MATE_SCORE {
        return pick_proven_mate(result, config, rng);
    }
    if result.lines[0].score < -MATE_SCORE {
        return pick_proven_loss(result, config, rng);
    }
    if result.lines.len() == 1 {
        let best = &result.lines[0];
        return Some((best.mv, best.score));
    }

    let conversion_position = active_conversion(game).is_some();
    let behavior = adjusted_skill_behavior(result, config, conversion_position);
    let top_chance = score_to_win_chance_permille(result.lines[0].score);
    let clear_conversion = conversion_position && top_chance >= CLEAR_CONVERSION_WIN_CHANCE;
    let conversion_score_floor = if clear_conversion {
        CONVERSION_SCORE_FLOOR.max(result.lines[0].score - config.max_conversion_cp_loss)
    } else {
        -INFINITY
    };
    // Rolled once per decision, like allow_obvious_hang, so it governs both
    // branches below rather than re-rolling per candidate.
    let allow_deep_tactic = rng.next_f64() < config.deep_tactic_chance;

    if rng.next_f64() < behavior.best_move_chance {
        let mut chosen_idx = None;
        let mut equivalent_count = 0u64;
        for (idx, line) in result.lines.iter().enumerate() {
            if line.score < conversion_score_floor {
                continue;
            }
            let ordinary_loss = (top_chance - score_to_win_chance_permille(line.score)).max(0);
            let loss = if clear_conversion {
                conversion_loss_permille(
                    result.lines[0].score,
                    line.score,
                    ordinary_loss,
                    behavior,
                    config,
                )
            } else {
                ordinary_loss
            };
            if loss > 2 {
                continue;
            }
            if !allow_deep_tactic && is_deep_only_tactic(result, line.mv, line.score) {
                continue;
            }
            equivalent_count += 1;
            if rng.next_u64().is_multiple_of(equivalent_count) {
                chosen_idx = Some(idx);
            }
        }
        // If every best-equivalent line required deeper calculation than this
        // level gets credit for, fall through to the regret sampler below
        // instead of blindly playing the top line anyway.
        if let Some(idx) = chosen_idx {
            let chosen = &result.lines[idx];
            return Some((chosen.mv, chosen.score));
        }
    }

    // Inverse-CDF sampling of an exponential distribution. Keep the input
    // below one because next_f64() can very rarely return exactly 1.0.
    let u = rng.next_f64().min(1.0 - f64::EPSILON);
    let target_loss = (-behavior.mean_loss_permille * (1.0 - u).ln())
        .round()
        .clamp(1.0, behavior.max_loss_permille as f64) as i32;
    let allow_obvious_hang = rng.next_f64() < config.obvious_blunder_chance;
    let temperature = (behavior.mean_loss_permille * 0.22).clamp(4.0, 24.0);
    let mut choices: Vec<(usize, f64)> = Vec::with_capacity(result.lines.len());
    let mut minimum_cost = f64::INFINITY;

    for (idx, line) in result.lines.iter().enumerate() {
        if line.score < conversion_score_floor {
            continue;
        }
        let ordinary_loss = (top_chance - score_to_win_chance_permille(line.score)).max(0);
        let loss = if clear_conversion {
            conversion_loss_permille(
                result.lines[0].score,
                line.score,
                ordinary_loss,
                behavior,
                config,
            )
        } else {
            ordinary_loss
        };
        if loss > behavior.max_loss_permille {
            continue;
        }
        if !allow_obvious_hang && is_obvious_material_hang(game, line, loss) {
            continue;
        }
        if !allow_deep_tactic && is_deep_only_tactic(result, line.mv, line.score) {
            continue;
        }

        let shallow_rank = result
            .shallow_order
            .iter()
            .position(|mv| *mv == line.mv)
            .unwrap_or(idx);
        let plausibility_penalty = ((shallow_rank + 1) as f64).ln() * SHALLOW_RANK_LOG_PENALTY;
        let cost = (loss - target_loss).abs() as f64 + plausibility_penalty;
        minimum_cost = minimum_cost.min(cost);
        choices.push((idx, cost));
    }

    if choices.is_empty() {
        let best = &result.lines[0];
        return Some((best.mv, best.score));
    }

    let total_weight: f64 = choices
        .iter()
        .map(|(_, cost)| (-(cost - minimum_cost) / temperature).exp())
        .sum();
    let mut sample = rng.next_f64() * total_weight;
    let mut chosen_idx = choices[0].0;
    for (idx, cost) in choices {
        chosen_idx = idx;
        sample -= (-(cost - minimum_cost) / temperature).exp();
        if sample <= 0.0 {
            break;
        }
    }

    let best = &result.lines[chosen_idx];
    Some((best.mv, best.score))
}

/// Entry point for searches with strength limiting.
/// Consolidated to use standard search path with root-level move selection.
pub(crate) fn get_best_move_limited(
    game: &mut GameState,
    max_depth: usize,
    opt_time_ms: u128,
    max_time_ms: u128,
    strength_level: Option<u32>,
    silent: bool,
    is_soft_limit: bool,
) -> Option<(Move, i32, SearchStats)> {
    // Clear any stale stop request (check_time polls GLOBAL_STOP).
    GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);

    game.recompute_piece_counts();
    game.recompute_correction_hashes();

    let input_skill = strength_level
        .unwrap_or(MAX_SITE_SKILL)
        .clamp(1, MAX_SITE_SKILL);
    if input_skill >= MAX_SITE_SKILL {
        return get_best_move_parallel(
            game,
            max_depth,
            opt_time_ms,
            max_time_ms,
            silent,
            is_soft_limit,
        );
    }

    GLOBAL_SEARCHER.with(|cell| {
        let mut opt = cell.borrow_mut();
        let searcher = opt.get_or_insert_with(|| Searcher::new(max_time_ms));

        searcher.new_search();
        searcher.silent = silent;
        searcher.hot.timer.reset();

        searcher.set_corrhist_mode(game);
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        // Local TT is already initialized in Searcher::new.
        let config = SKILL_CONFIGS[(input_skill - 1) as usize];
        let multi_pv = config.candidates.min(MAX_PV_COUNT);
        let effective_depth = effective_skill_depth(game, max_depth, config);

        // For MultiPV, we use the same optimum/maximum but disable dynamic extensions
        searcher
            .hot
            .set_time_limits(opt_time_ms, max_time_ms, is_soft_limit);

        // Level 7 pays MultiPV only on the decisions it lapses on: clean ones get a
        // plain single-PV search. That keeps it below perfect without an artificial
        // compute cliff right before level 8's full parallel search.
        let skilled_clean_search = input_skill == MAX_SITE_SKILL - 1
            && searcher.rng.next_f64() < LEVEL_7_CLEAN_SEARCH_CHANCE;
        if skilled_clean_search {
            let res = search_with_searcher(searcher, game, effective_depth);
            let stats = build_search_stats(searcher);
            return res.map(|(m, eval)| (m, eval, stats));
        }

        if multi_pv > 1 {
            // For MultiPV: cap dynamic time at optimum to prevent runaway extensions
            searcher.hot.total_time_ms = opt_time_ms as f64;
        }

        if multi_pv > 1 {
            let result = get_best_moves_multipv_impl(
                searcher,
                game,
                effective_depth,
                multi_pv,
                silent,
                None,
                None,
                None,
            );
            let stats = result.stats.clone();
            let picker_config = if input_skill == MAX_SITE_SKILL - 1 {
                SkillConfig {
                    best_move_chance: 0.0,
                    ..config
                }
            } else {
                config
            };
            pick_best(game, &result, picker_config, &mut searcher.rng)
                .map(|(m, eval)| (m, eval, stats))
        } else {
            let res = search_with_searcher(searcher, game, effective_depth);
            let stats = build_search_stats(searcher);
            res.map(|(m, eval)| (m, eval, stats))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;

    fn skill_test_result(scores: &[i32]) -> MultiPVResult {
        let piece = Piece::new(PieceType::Pawn, PlayerColor::White);
        let lines: Vec<PVLine> = scores
            .iter()
            .enumerate()
            .map(|(i, score)| {
                let mv = Move::new(
                    Coordinate::new(i as i64, 0),
                    Coordinate::new(i as i64, 1),
                    piece,
                );
                PVLine {
                    mv,
                    score: *score,
                    depth: 1,
                    pv: vec![mv],
                }
            })
            .collect();
        MultiPVResult {
            shallow_order: lines.iter().map(|line| line.mv).collect(),
            deep_ref_scores: lines.iter().map(|line| (line.mv, line.score)).collect(),
            lines,
            stats: SearchStats {
                nodes: 0,
                tt_capacity: 0,
                tt_used: 0,
                tt_fill_permille: 0,
            },
            shallow_best_changed: false,
        }
    }

    #[test]
    fn test_skill_config_ladder_is_monotonic() {
        for pair in SKILL_CONFIGS.windows(2) {
            let weaker = pair[0];
            let stronger = pair[1];
            assert!(weaker.candidates >= stronger.candidates);
            assert!(weaker.best_move_chance <= stronger.best_move_chance);
            assert!(weaker.mean_loss_permille >= stronger.mean_loss_permille);
            assert!(weaker.max_loss_permille >= stronger.max_loss_permille);
            assert!(weaker.max_conversion_cp_loss >= stronger.max_conversion_cp_loss);
            assert!(weaker.mate_distance_temperature >= stronger.mate_distance_temperature);
            assert!(weaker.obvious_blunder_chance >= stronger.obvious_blunder_chance);
            assert!(weaker.deep_tactic_chance <= stronger.deep_tactic_chance);
            if let (Some(weak_depth), Some(strong_depth)) = (weaker.depth_cap, stronger.depth_cap) {
                assert!(weak_depth <= strong_depth);
                assert!(
                    weak_depth + weaker.endgame_depth_bonus
                        <= strong_depth + stronger.endgame_depth_bonus
                );
                assert!(
                    weak_depth + weaker.mop_up_depth_bonus
                        <= strong_depth + stronger.mop_up_depth_bonus
                );
            }
        }
    }

    #[test]
    fn test_skill_behavior_responds_to_position_difficulty() {
        let config = SKILL_CONFIGS[2];
        let stable = skill_test_result(&[0, -10]);
        let clear = skill_test_result(&[0, -100]);
        let mut unstable = stable.clone();
        unstable.shallow_best_changed = true;

        let stable_behavior = adjusted_skill_behavior(&stable, config, false);
        let clear_behavior = adjusted_skill_behavior(&clear, config, false);
        let unstable_behavior = adjusted_skill_behavior(&unstable, config, false);

        assert!(clear_behavior.best_move_chance > stable_behavior.best_move_chance);
        assert!(stable_behavior.best_move_chance > unstable_behavior.best_move_chance);
        assert!(clear_behavior.mean_loss_permille > stable_behavior.mean_loss_permille);
        assert_eq!(
            unstable_behavior.mean_loss_permille,
            stable_behavior.mean_loss_permille
        );
    }

    #[test]
    fn test_skill_conversion_behavior_protects_winning_endgames() {
        let result = skill_test_result(&[500, 0, -200]);
        let config = SKILL_CONFIGS[4]; // Site level 5
        let normal = adjusted_skill_behavior(&result, config, false);
        let conversion = adjusted_skill_behavior(&result, config, true);

        assert!(conversion.best_move_chance > normal.best_move_chance);
        assert!(conversion.mean_loss_permille < normal.mean_loss_permille);
        assert!(conversion.max_loss_permille < normal.max_loss_permille);
    }

    #[test]
    fn test_skill_uses_actual_mop_up_activation() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9");
        let winner = active_conversion(&game).expect("white should be converting");
        assert_eq!(winner, PlayerColor::White);

        game.turn = PlayerColor::Black;
        assert!(
            active_conversion(&game).is_none(),
            "the defending side must not receive conversion assistance"
        );
    }

    #[test]
    fn test_proven_mate_is_never_abandoned_or_deep_gated() {
        let mut result = skill_test_result(&[mate_in(3), mate_in(7), 1800]);
        result.deep_ref_scores[0].1 = 0;
        assert!(is_deep_only_tactic(
            &result,
            result.lines[0].mv,
            result.lines[0].score
        ));

        let game = GameState::new();
        for config in SKILL_CONFIGS {
            for seed in 1..=500 {
                let mut rng = Prng::new(seed);
                let (_, score) = pick_best(&game, &result, config, &mut rng).unwrap();
                assert!(
                    score > MATE_SCORE,
                    "a proved mate must stay in the mating outcome class"
                );
            }
        }
    }

    #[test]
    fn test_limited_search_plays_mate_in_one_at_every_level() {
        for level in 1..MAX_SITE_SKILL {
            reset_search_state();
            set_global_params(level as u64, None);
            let mut game = GameState::new();
            game.setup_position_from_icn(
                "w K-5,-5|R5,5|k0,0|p-1,-1|p0,-1|p1,-1|p-1,0|p1,0|p-1,1|p1,1",
            );

            let (mv, score, _) =
                get_best_move_limited(&mut game, 3, 2000, 2000, Some(level), true, false)
                    .expect("position has legal moves");
            assert!(score > MATE_SCORE, "site level {level}: score={score}");
            assert_eq!(
                (mv.to.x, mv.to.y),
                (0, 5),
                "site level {level}: selected {mv:?} with mate score {score}"
            );
        }
    }

    #[test]
    fn test_mate_distance_preference_scales_by_level() {
        let result = skill_test_result(&[mate_in(3), mate_in(7)]);
        let game = GameState::new();
        let shortest_count = |config: SkillConfig| {
            (1..=2000)
                .filter(|seed| {
                    let mut rng = Prng::new(*seed);
                    pick_best(&game, &result, config, &mut rng).unwrap().1 == mate_in(3)
                })
                .count()
        };

        let weakest_shortest = shortest_count(SKILL_CONFIGS[0]);
        let strongest_shortest = shortest_count(SKILL_CONFIGS[6]);
        assert!(
            weakest_shortest < 1800,
            "weak levels should allow slower mates"
        );
        assert!(
            strongest_shortest > 1990,
            "strong limited play should overwhelmingly choose the shortest mate"
        );
    }

    #[test]
    fn test_proven_loss_prefers_longer_resistance_by_level() {
        let result = skill_test_result(&[mated_in(10), mated_in(3)]);
        let game = GameState::new();
        let survival_count = |config: SkillConfig| {
            (1..=2000)
                .filter(|seed| {
                    let mut rng = Prng::new(*seed);
                    pick_best(&game, &result, config, &mut rng).unwrap().1 == mated_in(10)
                })
                .count()
        };
        assert!(survival_count(SKILL_CONFIGS[6]) > survival_count(SKILL_CONFIGS[0]));
    }

    #[test]
    fn test_mop_up_conversion_rejects_drawing_regression() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9");
        let result = skill_test_result(&[1500, 1000, 0, -500]);

        for config in SKILL_CONFIGS {
            for seed in 1..=500 {
                let mut rng = Prng::new(seed);
                let (_, score) = pick_best(&game, &result, config, &mut rng).unwrap();
                assert!(
                    score >= CONVERSION_SCORE_FLOOR,
                    "conversion sampling must not cross into draw/loss"
                );
                assert!(
                    result.lines[0].score - score <= config.max_conversion_cp_loss,
                    "conversion sampling exceeded the level's score-loss corridor"
                );
            }
        }
    }

    #[test]
    fn test_mop_up_gets_targeted_extra_depth() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9");
        for config in SKILL_CONFIGS {
            let cap = config.depth_cap.unwrap();
            assert_eq!(
                effective_skill_depth(&game, 64, config),
                cap + config.mop_up_depth_bonus
            );
        }
    }

    #[test]
    fn test_skill_recognizes_direct_major_piece_hang() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K0,0|Q4,4|k7,7|r4,7");
        let queen = Piece::new(PieceType::Queen, PlayerColor::White);
        let hanging_move = Move::new(Coordinate::new(4, 4), Coordinate::new(4, 6), queen);
        let line = PVLine {
            mv: hanging_move,
            score: -500,
            depth: 2,
            pv: vec![hanging_move],
        };

        assert!(is_obvious_material_hang(&game, &line, 100));
        assert!(!is_obvious_material_hang(&game, &line, 0));
    }

    #[test]
    fn test_skill_deep_only_tactic_gate() {
        let piece = Piece::new(PieceType::Pawn, PlayerColor::White);
        let deep_mv = Move::new(Coordinate::new(0, 0), Coordinate::new(0, 1), piece);
        let plain_mv = Move::new(Coordinate::new(1, 0), Coordinate::new(1, 1), piece);
        // Both end on the same score, but deep_mv only looks good at full depth
        // (-100 shallow, 200 final) while plain_mv has no forced line behind it.
        let result = MultiPVResult {
            lines: vec![
                PVLine {
                    mv: deep_mv,
                    score: 200,
                    depth: 5,
                    pv: vec![deep_mv],
                },
                PVLine {
                    mv: plain_mv,
                    score: 200,
                    depth: 5,
                    pv: vec![plain_mv],
                },
            ],
            stats: SearchStats {
                nodes: 0,
                tt_capacity: 0,
                tt_used: 0,
                tt_fill_permille: 0,
            },
            shallow_best_changed: false,
            shallow_order: vec![plain_mv, deep_mv],
            deep_ref_scores: vec![(deep_mv, -100), (plain_mv, 200)],
        };
        assert!(is_deep_only_tactic(&result, deep_mv, 200));
        assert!(!is_deep_only_tactic(&result, plain_mv, 200));

        let game = GameState::new();
        let mut config = SKILL_CONFIGS[0];
        config.best_move_chance = 1.0;

        config.deep_tactic_chance = 0.0;
        let never_deep = (1..=200u64).all(|seed| {
            let mut rng = Prng::new(seed);
            pick_best(&game, &result, config, &mut rng).unwrap().0 != deep_mv
        });
        assert!(
            never_deep,
            "deep-only tactic must not surface when disallowed"
        );

        config.deep_tactic_chance = 1.0;
        let sometimes_deep = (1..=200u64).any(|seed| {
            let mut rng = Prng::new(seed);
            pick_best(&game, &result, config, &mut rng).unwrap().0 == deep_mv
        });
        assert!(
            sometimes_deep,
            "deep-only tactic must be reachable when allowed"
        );
    }

    #[test]
    fn test_deep_tactic_reference_depth_scales_with_budget() {
        // Reference stays strictly below max_depth and strictly grows as the
        // level's own depth budget grows, across every real skill ladder cap.
        let mut prev = 0;
        for config in SKILL_CONFIGS {
            let Some(cap) = config.depth_cap else {
                continue;
            };
            let max_depth = cap + config.endgame_depth_bonus;
            let reference = deep_tactic_reference_depth(max_depth);
            assert!(reference < max_depth);
            assert!(
                reference >= prev,
                "reference depth must not shrink up the ladder"
            );
            prev = reference;
        }
        // A beginner's 1-ply gap stays tight; the top of the ladder gets a much
        // wider gap, so "deep" is relative to what that level can afford.
        assert_eq!(deep_tactic_reference_depth(3), 2);
        assert!(max_depth_gap(17) > max_depth_gap(3));

        fn max_depth_gap(max_depth: usize) -> usize {
            max_depth - deep_tactic_reference_depth(max_depth)
        }
    }

    #[test]
    fn test_skill_picker_uses_wide_candidate_set() {
        let scores: Vec<i32> = (0..64).map(|i| -(i * 10)).collect();
        let result = skill_test_result(&scores);
        let config = SKILL_CONFIGS[2]; // Site level 3
        let game = GameState::new();

        let reached_beyond_stockfish_multipv = (1..=1000).any(|seed| {
            let mut rng = Prng::new(seed);
            let (mv, _) = pick_best(&game, &result, config, &mut rng).unwrap();
            mv.from.x >= 4
        });

        assert!(reached_beyond_stockfish_multipv);
    }

    #[test]
    fn test_skill_picker_expected_loss_decreases_by_level() {
        let scores: Vec<i32> = (0..MAX_PV_COUNT).map(|i| -(i as i32 * 12)).collect();
        let result = skill_test_result(&scores);
        let game = GameState::new();
        let sample_average = |config: SkillConfig| -> f64 {
            let mut total = 0u64;
            let candidates = MultiPVResult {
                lines: result.lines[..config.candidates].to_vec(),
                stats: result.stats.clone(),
                shallow_best_changed: result.shallow_best_changed,
                shallow_order: result.shallow_order.clone(),
                deep_ref_scores: result.deep_ref_scores.clone(),
            };
            for seed in 1..=5000 {
                let mut rng = Prng::new(seed);
                let (_, score) = pick_best(&game, &candidates, config, &mut rng).unwrap();
                total +=
                    (score_to_win_chance_permille(0) - score_to_win_chance_permille(score)) as u64;
            }
            total as f64 / 5000.0
        };

        let level_1 = sample_average(SKILL_CONFIGS[0]);
        let level_3 = sample_average(SKILL_CONFIGS[2]);
        let level_6 = sample_average(SKILL_CONFIGS[5]);
        let level_7 = sample_average(SKILL_CONFIGS[6]);
        assert!(level_1 > level_3);
        assert!(level_3 > level_6);
        assert!(level_6 > level_7);
    }

    #[test]
    fn test_skill_picker_respects_maximum_loss() {
        let result = skill_test_result(&[0, -67]);
        let config = SkillConfig {
            depth_cap: None,
            endgame_depth_bonus: 0,
            mop_up_depth_bonus: 0,
            candidates: 2,
            best_move_chance: 0.0,
            mean_loss_permille: 1000.0,
            max_loss_permille: 30,
            max_conversion_cp_loss: 1000,
            mate_distance_temperature: 1.0,
            obvious_blunder_chance: 0.0,
            conversion_odds_multiplier: 1.0,
            conversion_loss_multiplier: 1.0,
            deep_tactic_chance: 1.0,
        };
        let game = GameState::new();

        for seed in 1..=100 {
            let mut rng = Prng::new(seed);
            let (mv, _) = pick_best(&game, &result, config, &mut rng).unwrap();
            assert_eq!(mv.from.x, 0);
        }
    }
}
