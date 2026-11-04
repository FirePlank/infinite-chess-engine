//! Site strength levels built from what a player at that level knows. Basic chess
//! (material, simple threats) is always understood; development, structure and
//! infinite-board ideas are learned later, textbook piece values give way to the
//! engine's, and calculation is short and narrow. A weak level loses because it does
//! not know what a good position is, not because it rolled a blunder.
use super::*;

pub const MAX_SITE_SKILL: u32 = 8; // Current max skill level on the site
pub const MAX_PV_COUNT: usize = 96; // Safety cap for MultiPV candidate collection

#[derive(Clone, Copy, Debug)]
struct SkillConfig {
    depth_cap: Option<usize>,
    endgame_depth_bonus: usize,
    mop_up_depth_bonus: usize,
    candidates: usize,
    /// Percent of each evaluation tier the level understands; see `EvalStyle`.
    intermediate: i32,
    advanced: i32,
    infinite: i32,
    /// Percent of the way from the engine's piece values to textbook ones.
    textbook: i32,
    /// Extra plies lost calculating a move longer than any on an 8x8 board, for
    /// either side: long-range ideas and threats take more calculation to find.
    far_reduction: i32,
    /// Captures read into an exchange before taking the position as it stands;
    /// `Some(0)` does not look at the reply to its own move at all.
    qs_limit: Option<usize>,
    /// Percent of a capture from across the board the level registers, both when
    /// checking what a move leaves en prise and inside its search: long-range threats
    /// and combinations are harder to see, and a big one still registers.
    far_sight: i32,
    /// Scale of the tie-break between moves the level judges about equal, in
    /// centipawns. Never reaches past four times this below the best move.
    temperature: f64,
    /// Extra material a candidate may leave en prise, beyond the safest candidate.
    material_slack: i32,
    max_conversion_cp_loss: i32,
    mate_distance_temperature: f64,
}

/// Site levels 1..7; level 8 bypasses the limiter for the full parallel search.
const SKILL_CONFIGS: [SkillConfig; 7] = [
    SkillConfig {
        depth_cap: Some(1),
        endgame_depth_bonus: 0,
        mop_up_depth_bonus: 4,
        candidates: 24,
        intermediate: 0,
        advanced: 0,
        infinite: 0,
        textbook: 100,
        far_reduction: 2,
        qs_limit: Some(1),
        far_sight: 10,
        temperature: 25.0,
        material_slack: 250,
        max_conversion_cp_loss: 600,
        mate_distance_temperature: 8.0,
    },
    SkillConfig {
        depth_cap: Some(1),
        endgame_depth_bonus: 0,
        mop_up_depth_bonus: 5,
        candidates: 20,
        intermediate: 30,
        advanced: 0,
        infinite: 0,
        textbook: 100,
        far_reduction: 2,
        qs_limit: Some(1),
        far_sight: 15,
        temperature: 20.0,
        material_slack: 200,
        max_conversion_cp_loss: 475,
        mate_distance_temperature: 6.0,
    },
    SkillConfig {
        depth_cap: Some(2),
        endgame_depth_bonus: 0,
        mop_up_depth_bonus: 5,
        candidates: 16,
        intermediate: 30,
        advanced: 0,
        infinite: 0,
        textbook: 100,
        far_reduction: 2,
        qs_limit: Some(2),
        far_sight: 25,
        temperature: 16.0,
        material_slack: 200,
        max_conversion_cp_loss: 375,
        mate_distance_temperature: 4.5,
    },
    SkillConfig {
        depth_cap: Some(2),
        endgame_depth_bonus: 1,
        mop_up_depth_bonus: 8,
        candidates: 12,
        intermediate: 40,
        advanced: 10,
        infinite: 10,
        textbook: 85,
        far_reduction: 2,
        qs_limit: Some(2),
        far_sight: 45,
        temperature: 12.0,
        material_slack: 180,
        max_conversion_cp_loss: 285,
        mate_distance_temperature: 3.0,
    },
    SkillConfig {
        depth_cap: Some(3),
        endgame_depth_bonus: 2,
        mop_up_depth_bonus: 8,
        candidates: 10,
        intermediate: 55,
        advanced: 15,
        infinite: 15,
        textbook: 70,
        far_reduction: 1,
        qs_limit: Some(3),
        far_sight: 60,
        temperature: 10.0,
        material_slack: 130,
        max_conversion_cp_loss: 210,
        mate_distance_temperature: 1.8,
    },
    SkillConfig {
        depth_cap: Some(4),
        endgame_depth_bonus: 3,
        mop_up_depth_bonus: 9,
        candidates: 8,
        intermediate: 100,
        advanced: 50,
        infinite: 50,
        textbook: 40,
        far_reduction: 1,
        qs_limit: Some(4),
        far_sight: 100,
        temperature: 8.0,
        material_slack: 60,
        max_conversion_cp_loss: 145,
        mate_distance_temperature: 0.8,
    },
    SkillConfig {
        depth_cap: Some(14),
        endgame_depth_bonus: 3,
        mop_up_depth_bonus: 3,
        candidates: 4,
        intermediate: 100,
        advanced: 100,
        infinite: 100,
        textbook: 0,
        far_reduction: 0,
        qs_limit: None,
        far_sight: 100,
        temperature: 2.0,
        material_slack: 10,
        max_conversion_cp_loss: 90,
        mate_distance_temperature: 0.25,
    },
];

const CLEAR_CONVERSION_WIN_CHANCE: i32 = 700;
/// Moves per side over which the tie-break is widened for opening variety.
const OPENING_VARIETY_MOVES: u32 = 6;
const CONVERSION_SCORE_FLOOR: i32 = 100;

/// A move this long cannot be played on an 8x8 board.
const FAR_MOVE_DISTANCE: u64 = 8;

/// Whether a move is longer than any on an 8x8 board.
#[inline]
pub(crate) fn is_far_move(m: &Move) -> bool {
    m.to.x.abs_diff(m.from.x).max(m.to.y.abs_diff(m.from.y)) >= FAR_MOVE_DISTANCE
}

/// Whether an 8x8 player would not look for this move: longer than any on an 8x8
/// board, or into or out of the board they know.
#[inline]
pub(crate) fn is_unfamiliar_move(m: &Move, area: Option<[i64; 4]>) -> bool {
    let outside = |x: i64, y: i64| {
        area.is_some_and(|[x0, x1, y0, y1]| x < x0 || x > x1 || y < y0 || y > y1)
    };
    is_far_move(m) || outside(m.from.x, m.from.y) || outside(m.to.x, m.to.y)
}

/// The board an 8x8 player knows: where the pawns stand, one square wider all round.
/// None without pawns, where no such board exists.
fn familiar_area(game: &GameState) -> Option<[i64; 4]> {
    let mut area: Option<[i64; 4]> = None;
    for (x, y, piece) in game.board.iter() {
        if piece.piece_type() != PieceType::Pawn {
            continue;
        }
        let a = area.get_or_insert([x, x, y, y]);
        a[0] = a[0].min(x);
        a[1] = a[1].max(x);
        a[2] = a[2].min(y);
        a[3] = a[3].max(y);
    }
    area.map(|[x0, x1, y0, y1]| [x0 - 1, x1 + 1, y0 - 1, y1 + 1])
}

/// Slope of the logistic from an internal score to expected score (draws counting
/// half), fitted on the current net's evals in its 3699 full-strength self-play games
/// that ended on the board (310k plies, 10+0.1; games stopped at the move cap have no
/// result): +1.20 is about 59%, +3.55 about 75%.
pub const WIN_CHANCE_K: f64 = 0.003_1;

/// Maps an internal score to expected score in permille, draws counting half. Mate
/// scores return the endpoints directly, so their sentinels never reach exp().
#[inline]
pub fn score_to_win_chance_permille(score: i32) -> i32 {
    if score > MATE_SCORE {
        return 1000;
    }
    if score < -MATE_SCORE {
        return 0;
    }

    let cp = score.clamp(-1800, 1800) as f64;
    (1000.0 / (1.0 + (-WIN_CHANCE_K * cp).exp())).round() as i32
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
            (idx, (-extra_plies / config.mate_distance_temperature).exp())
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

/// Halfway to max_depth, so "deep" scales with this search's own depth budget
/// instead of a fixed depth that's trivial once caps rise into double digits.
pub(crate) fn deep_tactic_reference_depth(max_depth: usize) -> usize {
    max_depth
        .div_ceil(2)
        .max(2)
        .min(max_depth.saturating_sub(1).max(1))
}

/// Material the opponent can win by a capture right after `mv`, recaptures included,
/// less what `mv` itself took; a capture from across the board counts `far_sight`
/// percent, unless it takes a piece just put in the way of a check, which any player
/// looks at. This is what any chess player checks before letting
/// go of a piece: an even trade or a recapture costs nothing here, a piece left en
/// prise costs its value.
fn material_risk(game: &mut GameState, mv: &Move, far_sight: i32) -> i32 {
    let mover = mv.piece.color();
    let evading = game.is_in_check();
    let won = if game.is_en_passant(mv) {
        game.get_piece_value(PieceType::Pawn, mover.opponent())
    } else {
        game.board
            .get_piece(mv.to.x, mv.to.y)
            .filter(|p| !p.piece_type().is_neutral_type() && !p.piece_type().is_royal())
            .map_or(0, |p| game.get_piece_value(p.piece_type(), p.color()))
    } + mv.promotion.map_or(0, |promo| {
        game.get_piece_value(promo, mover) - game.get_piece_value(mv.piece.piece_type(), mover)
    });

    let undo = game.make_move(mv);
    let pinned = rustc_hash::FxHashMap::default();
    let ctx = MoveGenContext {
        pinned: &pinned,
        special_rights: &game.special_rights,
        en_passant: &game.en_passant,
        game_rules: &game.game_rules,
        indices: &game.spatial_indices,
        enemy_king_pos: game.enemy_king_pos(),
    };
    let mut captures = MoveList::new();
    get_quiescence_captures(&game.board, game.turn, &ctx, &mut captures);
    let lost = captures
        .iter()
        .filter(|c| {
            game.board
                .get_piece(c.to.x, c.to.y)
                .is_some_and(|p| !p.piece_type().is_royal())
        })
        .map(|c| {
            let see = see::static_exchange_eval_impl(game, c);
            if is_far_move(c) && !(evading && c.to == mv.to) {
                see * far_sight / 100
            } else {
                see
            }
        })
        .max()
        .unwrap_or(0)
        .max(0);
    game.undo_move(mv, undo);
    (lost - won).max(0)
}

/// A Gumbel draw of the given scale that depends only on `seed` and the move.
fn seeded_gumbel(seed: u64, mv: &Move, temperature: f64) -> f64 {
    let mut z = seed
        ^ (mv.from.x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (mv.from.y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (mv.to.x as u64).wrapping_mul(0x1656_67b1_9e37_79f9)
        ^ (mv.to.y as u64).wrapping_mul(0x27d4_eb2f_1656_67c5);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    -temperature * (-u.ln()).ln()
}

/// Chooses among the searched candidates, which the level has already judged with
/// its own knowledge. Every candidate must be tactically plausible to a chess player:
/// it may not leave more material en prise than the safest one does (plus the level's
/// slack) unless its searched score pays for it, and weak levels trust that payment
/// less. The best plausible move is played; a small tie-break, varying from search to
/// search, separates moves the level cannot tell apart.
fn pick_best(
    game: &mut GameState,
    result: &MultiPVResult,
    config: SkillConfig,
    seed: u64,
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

    let lines = &result.lines;
    let top = lines[0].score;
    let conversion = active_conversion(game).is_some();
    let floor = if conversion && score_to_win_chance_permille(top) >= CLEAR_CONVERSION_WIN_CHANCE
    {
        CONVERSION_SCORE_FLOOR.max(top - config.max_conversion_cp_loss)
    } else {
        -MATE_SCORE
    };
    // A tie-break, not a mistake generator: nothing beyond a few of its widths.
    let max_loss = (config.temperature * 4.0) as i32;
    // Opening moves really are close, and a weak level cannot rank them, so within
    // the same window it picks among them more evenly there and games differ.
    let temperature = if game.fullmove_number <= OPENING_VARIETY_MOVES {
        config.temperature * 3.0
    } else {
        config.temperature
    };

    let risks: Vec<i32> = lines
        .iter()
        .map(|line| material_risk(game, &line.mv, if conversion { 100 } else { config.far_sight }))
        .collect();
    let viable = |k: usize| lines[k].score >= floor;
    let safest = (0..lines.len()).filter(|&k| viable(k)).map(|k| risks[k]).min()?;
    let safe_score = (0..lines.len())
        .filter(|&k| viable(k) && risks[k] <= safest + config.material_slack)
        .map(|k| lines[k].score)
        .max()?;
    let plausible = |k: usize| {
        let compensation = (lines[k].score - safe_score).max(0) * config.infinite / 100;
        viable(k) && risks[k] <= safest + config.material_slack + compensation
    };
    let best_plausible = (0..lines.len())
        .filter(|&k| plausible(k))
        .map(|k| lines[k].score)
        .max()?;

    // An unfamiliar candidate (long, or into or out of the known board) is judged only
    // partly on what it gains over the best ordinary move: the level searched it, but
    // does not trust a long-range idea the way it trusts an ordinary one.
    let area = if conversion { None } else { familiar_area(game) };
    let unfamiliar = |k: usize| !conversion && is_unfamiliar_move(&lines[k].mv, area);
    let familiar_best = (0..lines.len())
        .filter(|&k| plausible(k) && !unfamiliar(k))
        .map(|k| lines[k].score)
        .max();
    let judged = |k: usize| match familiar_best {
        Some(base) if unfamiliar(k) && lines[k].score > base => {
            base + (lines[k].score - base) * config.far_sight / 100
        }
        _ => lines[k].score,
    };
    let best_plausible = (0..lines.len())
        .filter(|&k| plausible(k))
        .map(judged)
        .max()
        .unwrap_or(best_plausible);

    let mut chosen = None;
    let mut chosen_value = f64::NEG_INFINITY;
    for (k, line) in lines.iter().enumerate() {
        if !plausible(k) || best_plausible - judged(k) > max_loss {
            continue;
        }
        let value = judged(k) as f64 + seeded_gumbel(seed, &line.mv, temperature);
        if value > chosen_value {
            chosen_value = value;
            chosen = Some(k);
        }
    }

    let line = &lines[chosen.unwrap_or(0)];
    Some((line.mv, line.score))
}

/// Installs a level's evaluation and far-move reduction for one search and restores
/// full strength on the way out, including on an early return. Nothing outside a
/// limited search may ever observe a scaled evaluation.
struct EvalStyleGuard;

impl EvalStyleGuard {
    fn install(style: crate::evaluation::EvalStyle) -> Self {
        crate::evaluation::set_eval_style(style);
        EvalStyleGuard
    }
}

impl Drop for EvalStyleGuard {
    fn drop(&mut self) {
        crate::evaluation::set_eval_style(crate::evaluation::EvalStyle::NEUTRAL);
    }
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
        searcher.move_rule_limit = game
            .game_rules
            .move_rule_limit
            .map_or(i32::MAX, |v| v as i32);

        let config = SKILL_CONFIGS[(input_skill - 1) as usize];
        let multi_pv = config.candidates.min(MAX_PV_COUNT);
        let effective_depth = effective_skill_depth(game, max_depth, config);

        // Conversion stays full strength. Decided at the root like the depth bonus:
        // the tree below a converting root is overwhelmingly still converting, and a
        // per-node check would cost a mop-up probe on every evaluation.
        let converting = active_conversion(game).is_some();
        let _style = EvalStyleGuard::install(if converting {
            crate::evaluation::EvalStyle::NEUTRAL
        } else {
            crate::evaluation::EvalStyle {
                intermediate: config.intermediate,
                advanced: config.advanced,
                infinite: config.infinite,
                textbook: config.textbook,
            }
        });
        searcher.skill_far_reduction = if converting { 0 } else { config.far_reduction };
        searcher.skill_qs_limit = if converting { None } else { config.qs_limit };
        searcher.skill_far_sight = if converting { 100 } else { config.far_sight };
        searcher.skill_area = if converting || (config.far_reduction == 0 && config.far_sight >= 100) {
            None
        } else {
            familiar_area(game)
        };
        // After the style is installed, so a style change drops the other style's scores.
        searcher.adopt_eval_kind(game.eval_kind);

        searcher
            .hot
            .set_time_limits(opt_time_ms, max_time_ms, is_soft_limit);
        // For MultiPV: cap dynamic time at optimum to prevent runaway extensions
        searcher.hot.total_time_ms = opt_time_ms as f64;

        let result = get_best_moves_multipv_impl(
            searcher,
            game,
            effective_depth,
            multi_pv,
            silent,
            None,
            None,
        );
        searcher.skill_far_reduction = 0;
        searcher.skill_qs_limit = None;
        searcher.skill_far_sight = 100;
        searcher.skill_area = None;
        let stats = result.stats.clone();
        // Each search arrives with a fresh seed, so games vary among the moves a level
        // judges equal; nothing outside the tie-break window is ever reachable.
        let seed = game.hash
            ^ (input_skill as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ searcher.rng.next_u64();
        pick_best(game, &result, config, seed, &mut searcher.rng)
            .map(|(m, eval)| (m, eval, stats))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;

    fn line(mv: Move, score: i32) -> PVLine {
        PVLine {
            mv,
            score,
            depth: 4,
            pv: vec![mv],
        }
    }

    fn multipv(lines: Vec<PVLine>) -> MultiPVResult {
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

    fn legal_moves(game: &mut GameState) -> Vec<Move> {
        game.get_pseudo_legal_moves()
            .into_iter()
            .filter(|m| {
                let undo = game.make_move(m);
                let ok = !game.is_move_illegal();
                game.undo_move(m, undo);
                ok
            })
            .collect()
    }

    fn find(moves: &[Move], from: (i64, i64), to: (i64, i64)) -> Move {
        moves
            .iter()
            .find(|m| (m.from.x, m.from.y, m.to.x, m.to.y) == (from.0, from.1, to.0, to.1))
            .copied()
            .unwrap_or_else(|| {
                let from_here: Vec<_> = moves
                    .iter()
                    .filter(|m| (m.from.x, m.from.y) == from)
                    .map(|m| (m.to.x, m.to.y))
                    .collect();
                panic!("no move {from:?}->{to:?}; have {from_here:?}")
            })
    }

    fn count_picks(
        game: &mut GameState,
        result: &MultiPVResult,
        config: SkillConfig,
        mv: Move,
    ) -> usize {
        (1..=500u64)
            .filter(|seed| {
                let mut rng = Prng::new(*seed);
                pick_best(game, result, config, *seed, &mut rng).unwrap().0 == mv
            })
            .count()
    }

    #[test]
    fn test_skill_config_ladder_is_monotonic() {
        for pair in SKILL_CONFIGS.windows(2) {
            let weaker = pair[0];
            let stronger = pair[1];
            assert!(weaker.candidates >= stronger.candidates);
            assert!(weaker.intermediate <= stronger.intermediate);
            assert!(weaker.advanced <= stronger.advanced);
            assert!(weaker.infinite <= stronger.infinite);
            assert!(weaker.textbook >= stronger.textbook);
            assert!(weaker.far_reduction >= stronger.far_reduction);
            assert!(weaker.far_sight <= stronger.far_sight);
            assert!(weaker.temperature >= stronger.temperature);
            assert!(weaker.material_slack >= stronger.material_slack);
            let reads = |c: SkillConfig| c.qs_limit.unwrap_or(usize::MAX);
            assert!(reads(weaker) <= reads(stronger));
            assert!(weaker.max_conversion_cp_loss >= stronger.max_conversion_cp_loss);
            assert!(weaker.mate_distance_temperature >= stronger.mate_distance_temperature);
            let (weak_depth, strong_depth) =
                (weaker.depth_cap.unwrap(), stronger.depth_cap.unwrap());
            assert!(weak_depth <= strong_depth);
            assert!(
                weak_depth + weaker.endgame_depth_bonus
                    <= strong_depth + stronger.endgame_depth_bonus
            );
            assert!(
                weak_depth + weaker.mop_up_depth_bonus <= strong_depth + stronger.mop_up_depth_bonus
            );
        }
        let top = SKILL_CONFIGS[6];
        assert_eq!(top.intermediate, 100);
        assert_eq!((top.far_reduction, top.qs_limit), (0, None));
    }

    /// Conversion searches exactly as deep as it always has: whatever `depth_cap`
    /// a level gets, `mop_up_depth_bonus` pays it back.
    #[test]
    fn test_mop_up_depth_ladder_is_pinned() {
        const MOP_UP_DEPTHS: [usize; 7] = [5, 6, 7, 10, 11, 13, 17];
        for (config, expected) in SKILL_CONFIGS.iter().zip(MOP_UP_DEPTHS) {
            assert_eq!(
                config.depth_cap.unwrap() + config.mop_up_depth_bonus,
                expected,
                "a depth_cap change was not paid back in mop_up_depth_bonus"
            );
        }
    }

    #[test]
    fn test_moves_off_the_known_board_are_unfamiliar() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K5,1|Q4,1|P1,2|P8,2|k5,8|p1,7|p8,7");
        let area = familiar_area(&game);
        assert_eq!(area, Some([0, 9, 1, 8]));
        let queen = Piece::new(PieceType::Queen, PlayerColor::White);
        let mv = |fx, fy, tx, ty| Move::new(Coordinate::new(fx, fy), Coordinate::new(tx, ty), queen);
        assert!(!is_unfamiliar_move(&mv(4, 1, 4, 7), area));
        assert!(is_unfamiliar_move(&mv(4, 1, 4, 13), area), "behind the back rank");
        assert!(is_unfamiliar_move(&mv(9, 13, 9, 8), area), "attacking from behind");
        assert!(!is_unfamiliar_move(&mv(4, 13, 9, 13), None));
    }

    #[test]
    fn test_far_moves_are_the_ones_8x8_cannot_play() {
        let piece = Piece::new(PieceType::Rook, PlayerColor::White);
        let mv = |dx: i64, dy: i64| Move::new(Coordinate::new(1, 1), Coordinate::new(1 + dx, 1 + dy), piece);
        assert!(!is_far_move(&mv(7, 0)));
        assert!(!is_far_move(&mv(0, -7)));
        assert!(is_far_move(&mv(8, 0)));
        assert!(is_far_move(&mv(-3, 40)));
        let huge = Move::new(
            Coordinate::new(-1_000_000_000_000_000, 0),
            Coordinate::new(1_000_000_000_000_000, 0),
            piece,
        );
        assert!(is_far_move(&huge));
    }

    fn eval_with(game: &GameState, style: crate::evaluation::EvalStyle) -> i32 {
        crate::evaluation::set_eval_style(style);
        let score = evaluate(game);
        crate::evaluation::set_eval_style(crate::evaluation::EvalStyle::NEUTRAL);
        score
    }

    /// Basic chess survives at every level: a level that knows nothing beyond it
    /// still counts material in full, and the style ends with its search.
    #[test]
    fn test_basic_chess_survives_every_tier_removed() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K1,1|Q6,6|R7,1|R7,2|k7,7|p6,7|p7,6|n2,8");
        let full = evaluate(&game);
        let basic = eval_with(
            &game,
            crate::evaluation::EvalStyle {
                intermediate: 0,
                advanced: 0,
                infinite: 0,
                textbook: 0,
            },
        );
        assert!(basic > 800, "material must survive with every tier removed: {basic}");
        assert_ne!(basic, full, "the learned tiers must be blended out");
        assert_eq!(evaluate(&game), full, "the style must not outlive its search");
    }

    /// Textbook values misprice an open board the way an 8x8 player does: a rook
    /// and bishop look like more than a queen.
    #[test]
    fn test_textbook_values_misprice_the_queen() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K5,1|Q4,1|P1,2|P2,2|P3,2|k5,8|r1,8|b3,8|p1,7|p2,7|p3,7");
        let textbook = crate::evaluation::EvalStyle {
            textbook: 100,
            ..crate::evaluation::EvalStyle::NEUTRAL
        };
        assert!(
            eval_with(&game, textbook) < evaluate(&game) - 300,
            "textbook values see R+B as worth about a queen"
        );
    }

    /// A capture of a piece is answered: no level ignores a recapture to play a
    /// quiet move a few centipawns better on paper.
    #[test]
    fn test_every_level_keeps_its_queen_and_recaptures() {
        let mut game = GameState::new();
        // Black's rook attacks the white queen: the king move leaves it, the queen
        // step saves it.
        game.setup_position_from_icn("w K1,1|Q4,4|R1,2|P2,2|N3,1|k7,7|r4,7|b6,6|p6,7|n5,8");
        let moves = legal_moves(&mut game);
        let save = find(&moves, (4, 4), (6, 4));
        let ignore = find(&moves, (1, 1), (0, 1));
        // Even if the search misjudged them as close, the king move loses the queen.
        let result = multipv(vec![line(save, 20), line(ignore, 0)]);
        for config in SKILL_CONFIGS {
            assert_eq!(count_picks(&mut game, &result, config, ignore), 0);
        }
    }

    /// An even trade is not a giveaway: SEE with recaptures prices it at zero.
    #[test]
    fn test_material_risk_understands_exchanges() {
        let mut game = GameState::new();
        // White queen takes a black queen defended by the king's rook; white's rook
        // recaptures behind it.
        game.setup_position_from_icn("w K0,0|Q4,4|R4,1|k8,9|q4,7|r1,7");
        let moves = legal_moves(&mut game);
        let trade = find(&moves, (4, 4), (4, 7));
        assert_eq!(material_risk(&mut game, &trade, 100), 0);
        let hang = find(&moves, (4, 4), (7, 7));
        assert!(material_risk(&mut game, &hang, 100) >= 800);
    }

    /// A weak level barely notices a capture from across the board, but it always
    /// sees the checking piece take what it just put in the way.
    #[test]
    fn test_blocking_a_far_check_with_a_loose_piece_is_seen() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K5,1|N3,4|P1,2|P8,2|k9,60|r5,40|p1,7|p8,7");
        let moves = legal_moves(&mut game);
        let block = find(&moves, (3, 4), (5, 5));
        assert!(material_risk(&mut game, &block, 10) >= 250);

        let mut game = GameState::new();
        game.setup_position_from_icn("w K4,1|N3,4|P1,2|P8,2|k9,60|r5,40|p1,7|p8,7");
        let moves = legal_moves(&mut game);
        let loose = find(&moves, (3, 4), (5, 5));
        assert!(material_risk(&mut game, &loose, 10) < 100, "no check: a far capture is hard to see");
    }

    /// Variety lives only among equal moves: different seeds pick different equal
    /// moves, and a clearly worse one is never picked.
    #[test]
    fn test_tie_break_varies_only_among_equal_moves() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K1,1|R2,2|P5,5|k20,20|p30,30");
        let moves = legal_moves(&mut game);
        let mut lines: Vec<PVLine> = moves.iter().take(4).map(|m| line(*m, 0)).collect();
        let worse = moves[4];
        lines.push(line(worse, -300));
        let result = multipv(lines);
        for config in SKILL_CONFIGS {
            let picks: std::collections::HashSet<_> = (0..200u64)
                .map(|seed| {
                    let mv = pick_best(&mut game, &result, config, seed, &mut Prng::new(1)).unwrap().0;
                    assert_ne!(mv, worse, "a clearly worse move is never the tie-break");
                    (mv.from.x, mv.from.y, mv.to.x, mv.to.y)
                })
                .collect();
            assert!(picks.len() > 1, "equal moves should vary across searches");
        }
    }

    /// Given a seed the choice is reproducible.
    #[test]
    fn test_choice_is_fixed_per_seed() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K1,1|R2,2|P5,5|k20,20|p30,30");
        let moves = legal_moves(&mut game);
        let result = multipv(moves.iter().take(6).map(|m| line(*m, 0)).collect());
        for config in SKILL_CONFIGS {
            let first = pick_best(&mut game, &result, config, 42, &mut Prng::new(1));
            for rng_seed in 2..50 {
                let again = pick_best(&mut game, &result, config, 42, &mut Prng::new(rng_seed));
                assert_eq!(again.map(|c| c.0), first.map(|c| c.0));
            }
        }
    }

    #[test]
    fn test_stronger_levels_lose_less() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K1,1|R2,2|P5,5|k20,20|p30,30");
        let moves = legal_moves(&mut game);
        let quiet: Vec<Move> = moves
            .iter()
            .copied()
            .filter(|m| !is_far_move(m) && m.piece.piece_type() != PieceType::King)
            .take(8)
            .collect();
        let scores = [200, 180, 150, 110, 60, 0, -80, -200];
        let result = multipv(
            quiet
                .iter()
                .zip(scores)
                .map(|(m, s)| line(*m, s))
                .collect(),
        );
        // Depth and noise trade off between neighbours, so compare levels far apart.
        let mut previous = i64::MIN;
        for config in [SKILL_CONFIGS[0], SKILL_CONFIGS[3], SKILL_CONFIGS[6]] {
            let total: i64 = (1..=2000u64)
                .map(|seed| {
                    let mut rng = Prng::new(seed);
                    pick_best(&mut game, &result, config, seed, &mut rng).unwrap().1 as i64
                })
                .sum();
            assert!(total > previous, "{total} <= {previous}");
            previous = total;
        }
    }

    #[test]
    fn test_proven_mate_is_never_abandoned() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K1,1|R2,2|P5,5|k20,20|p30,30");
        let moves = legal_moves(&mut game);
        let result = multipv(vec![
            line(moves[0], mate_in(3)),
            line(moves[1], mate_in(7)),
            line(moves[2], 1800),
        ]);
        for config in SKILL_CONFIGS {
            for seed in 1..=300 {
                let mut rng = Prng::new(seed);
                let (_, score) = pick_best(&mut game, &result, config, seed, &mut rng).unwrap();
                assert!(score > MATE_SCORE, "a proved mate must stay in the mating set");
            }
        }
    }

    /// The hard floor: whatever a level gives up, it still finishes a mate it sees.
    /// Ignored because it drives the global searcher, TT and seed, which the rest of
    /// the suite races with. Run with `--ignored --test-threads=1`.
    #[test]
    #[ignore = "needs exclusive global search state; run with --test-threads=1"]
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
    fn test_mop_up_conversion_rejects_drawing_regression() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9");
        assert!(active_conversion(&game).is_some());
        let moves = legal_moves(&mut game);
        let result = multipv(vec![
            line(moves[0], 1500),
            line(moves[1], 1000),
            line(moves[2], 0),
            line(moves[3], -500),
        ]);
        for config in SKILL_CONFIGS {
            for seed in 1..=300 {
                let mut rng = Prng::new(seed);
                let (_, score) = pick_best(&mut game, &result, config, seed, &mut rng).unwrap();
                assert!(score >= CONVERSION_SCORE_FLOOR);
                assert!(1500 - score <= config.max_conversion_cp_loss);
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
                effective_skill_depth(&game, MAX_PLY, config),
                cap + config.mop_up_depth_bonus
            );
        }
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
}
