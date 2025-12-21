use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

// Thread-local pawn structure cache: pawn_hash -> evaluation score
// This avoids recomputing pawn structure for positions with identical pawn configurations.
thread_local! {
    static PAWN_CACHE: RefCell<FxHashMap<u64, i32>> = RefCell::new(FxHashMap::default());
}

/// Clear the pawn structure cache. Call at the start of a new game.
pub fn clear_pawn_cache() {
    PAWN_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(feature = "eval_tuning")]
use once_cell::sync::Lazy;
#[cfg(feature = "eval_tuning")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "eval_tuning")]
use std::sync::RwLock;

#[cfg(feature = "eval_tuning")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvalFeatures {
    // King safety
    pub king_ring_missing_penalty: i32,
    pub king_open_ray_penalty: i32,
    pub king_enemy_slider_penalty: i32,

    // Development & piece order
    pub dev_queen_back_rank_penalty: i32,
    pub dev_rook_back_rank_penalty: i32,
    pub dev_minor_back_rank_penalty: i32,

    // Rook activity
    pub rook_idle_penalty: i32,

    // Slider mobility
    pub slider_mobility_bonus: i32,
    pub bishop_mobility_bonus: i32,

    // Pawn structure
    pub doubled_pawn_penalty: i32,

    // Bishop pair & queen heuristics
    pub bishop_pair_bonus: i32,
    pub queen_too_close_to_king_penalty: i32,
    pub queen_fork_zone_bonus: i32,
}

#[cfg(feature = "eval_tuning")]
static EVAL_FEATURES: Lazy<RwLock<EvalFeatures>> =
    Lazy::new(|| RwLock::new(EvalFeatures::default()));

#[cfg(feature = "eval_tuning")]
pub fn reset_eval_features() {
    if let Ok(mut guard) = EVAL_FEATURES.write() {
        *guard = EvalFeatures::default();
    }
}

#[cfg(feature = "eval_tuning")]
pub fn snapshot_eval_features() -> EvalFeatures {
    EVAL_FEATURES.read().map(|g| g.clone()).unwrap_or_default()
}

#[cfg(feature = "eval_tuning")]
macro_rules! bump_feat {
    ($field:ident, $amount:expr) => {{
        if let Ok(mut f) = $crate::evaluation::EVAL_FEATURES.write() {
            f.$field += $amount;
        }
    }};
}

#[cfg(not(feature = "eval_tuning"))]
macro_rules! bump_feat {
    ($($tt:tt)*) => {};
}

// ==================== Piece Values ====================

pub fn get_piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        // neutral/blocking pieces - no material value
        PieceType::Void => 0,
        PieceType::Obstacle => 0,

        // orthodox - adjusted for infinite chess where sliders dominate (baseline scale)
        PieceType::Pawn => 100,
        PieceType::Knight => 250, // Weak in infinite chess - limited range
        PieceType::Bishop => 450, // Strong slider - worth knight + 1.5 pawns
        PieceType::Rook => 650,   // Very strong in infinite chess
        PieceType::Queen | PieceType::RoyalQueen => 1350, // > 2 rooks
        PieceType::King | PieceType::Guard => 220,

        // short / medium range
        PieceType::Camel => 270,   // (1,3) leaper
        PieceType::Giraffe => 260, // (1,4) leaper
        PieceType::Zebra => 260,   // (2,3) leaper

        // riders / compounds
        PieceType::Knightrider => 700,
        PieceType::Amazon => 1550,
        PieceType::Hawk => 600,
        PieceType::Chancellor => 1000,
        PieceType::Archbishop => 900,
        PieceType::Centaur => 550,

        // royal compound
        PieceType::RoyalCentaur => 620,

        // special infinite-board pieces
        PieceType::Rose => 450,
        PieceType::Huygen => 355,
    }
}

// Rook heuristics
// Slightly increased based on Texel tuning (optimum around ~37), but kept
// moderate so rooks are encouraged to activate without over-penalizing
// reasonable defensive placements.
const ROOK_IDLE_PENALTY: i32 = 20;

// King attack heuristics - back near original scale
// These should be impactful but not dominate material.
const BEHIND_KING_BONUS: i32 = 40;
const KING_TROPISM_BONUS: i32 = 4;
const KNIGHT_NEAR_KING_BONUS: i32 = 15;
const SLIDER_NET_BONUS: i32 = 20;

// King safety ring and ray penalties - slightly stronger than original.
// Enemy slider penalty is bumped up toward the Texel suggestion (~75) but
// kept below that to remain in line with other positional terms.
const KING_RING_MISSING_PENALTY: i32 = 30;
const KING_OPEN_RAY_PENALTY: i32 = 10;
const KING_ENEMY_SLIDER_PENALTY: i32 = 40;

// King pawn shield heuristics (mild influence only)
// Reward having pawns in front of the king and penalize the king walking
// in front of its pawn chain.
const KING_PAWN_SHIELD_BONUS: i32 = 10;
const KING_PAWN_AHEAD_PENALTY: i32 = 20;

// Mobility tuning (kept modest so pieces don't run to infinity for raw space)
const SLIDER_MOBILITY_BONUS: i32 = 1; // Used only for queen/bishop mobility
const BISHOP_MOBILITY_BONUS: i32 = 1;

// Distance penalties to discourage sliders far away from the king "zone".
// We look at distance to both own and enemy king and penalize pieces that
// drift too far from either.
const FAR_SLIDER_CHEB_RADIUS: i64 = 18;
const FAR_SLIDER_CHEB_MAX_EXCESS: i64 = 40;
const FAR_QUEEN_PENALTY: i32 = 3;
const FAR_BISHOP_PENALTY: i32 = 2;
const FAR_ROOK_PENALTY: i32 = 2;
const PIECE_CLOUD_CHEB_RADIUS: i64 = 16;
const PIECE_CLOUD_CHEB_MAX_EXCESS: i64 = 64;
// Cloud penalty now scaled by piece value
const CLOUD_PENALTY_PER_100_VALUE: i32 = 1; // 1cp penalty per 100 piece value per excess square

// Pawn structure
const DOUBLED_PAWN_PENALTY: i32 = 8;

// Bishop pair & queen heuristics
const BISHOP_PAIR_BONUS: i32 = 60;
const QUEEN_IDEAL_LINE_DIST: i32 = 4;

// ==================== Fairy Piece Evaluation ====================

// Leaper positioning (tropism to kings and piece cloud)
const LEAPER_TROPISM_DIVISOR: i32 = 400; // piece_value / 400 = tropism multiplier
const LEAPER_MAX_TROPISM_DIST: i64 = 20; // Beyond this, bonus is capped

// Compound piece weight scaling (fraction of base piece eval to inherit)
const CHANCELLOR_ROOK_SCALE: i32 = 90; // 90% of rook eval
const ARCHBISHOP_BISHOP_SCALE: i32 = 90; // 90% of bishop eval
const AMAZON_ROOK_SCALE: i32 = 50; // 50% of rook eval (also has queen)
const AMAZON_QUEEN_SCALE: i32 = 70; // 70% of queen eval
const CENTAUR_KNIGHT_SCALE: i32 = 80; // 80% of knight eval
const CENTAUR_GUARD_SCALE: i32 = 50; // 50% of guard/leaper eval

// Knightrider specific
const KNIGHTRIDER_RAY_BONUS: i32 = 3; // Per square of knight-ray mobility

// ==================== Pawn Distance Scaling ====================

// Pawns far from promotion are worth much less in infinite chess
const PAWN_FULL_VALUE_THRESHOLD: i64 = 6; // Within 6 ranks = full value
const PAWN_PAST_PROMO_PENALTY: i32 = 80; // Massive penalty for pawns that can't promote
const PAWN_FAR_FROM_PROMO_PENALTY: i32 = 50; // Flat penalty for back pawns (no benefit from advancing)

// ==================== King Infinite Exposure ====================

// ==================== Development ====================

// Minimum starting square penalty for minors
const MIN_DEVELOPMENT_PENALTY: i32 = 6; // Moderate - not too aggressive

// King exposure: penalize kings with too many open directions
const KING_OPEN_DIRECTION_THRESHOLD: i32 = 4;
const KING_EXPOSURE_PENALTY_PER_DIR: i32 = 8;

// King defender bonuses/penalties
// Low-value pieces near own king = good (defense)
// High-value pieces near own king = bad (should be attacking)
const KING_DEFENDER_VALUE_THRESHOLD: i32 = 400; // Pieces below this value are defensive
const KING_DEFENDER_BONUS: i32 = 6; // Reduced - prefer center activity over defensive huddle
const KING_ATTACKER_NEAR_OWN_KING_PENALTY: i32 = 8; // Penalty for high-value pieces near own king

// ==================== Game Phase ====================

// Phase based on piece count (excluding pawns)
// Opening: >= 70% of pieces remain -> no pawn advancement bonus, development focus
// Middlegame: 30-70% of pieces remain -> partial pawn advancement bonus
// Endgame: < 30% of pieces remain -> full pawn evaluation
const ENDGAME_PIECE_THRESHOLD: i32 = 30; // Less than 30% = endgame
const MIDDLEGAME_PIECE_THRESHOLD: i32 = 70; // More than 70% = opening

// Development thresholds - for attack scaling only
const UNDEVELOPED_MINORS_THRESHOLD: i32 = 2;
const DEVELOPMENT_PHASE_ATTACK_SCALE: i32 = 50;
const DEVELOPED_PHASE_ATTACK_SCALE: i32 = 100;

/// Returns game phase as percentage (100 = opening, 0 = pure endgame)
/// Uses cached piece counts for O(1) performance
#[inline]
fn calculate_game_phase(game: &GameState) -> i32 {
    // Use the cached counts from GameState (set at game start, updated by make/undo)
    // We only count non-pawn pieces for game phase.
    let current_pieces = (game.white_piece_count.saturating_sub(game.white_pawn_count)
        + game.black_piece_count.saturating_sub(game.black_pawn_count))
        as i32;
    let starting_pieces = (game.starting_white_pieces + game.starting_black_pieces) as i32;

    if starting_pieces == 0 {
        return 50; // Default to middlegame
    }

    ((current_pieces * 100) / starting_pieces).min(100)
}

/// Compute the centroid of all non-obstacle, non-void pieces on the board.
/// Used for piece cloud calculations. (Made public for variant modules.)
pub fn compute_cloud_center(board: &Board) -> Option<Coordinate> {
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;
    let mut count: i64 = 0;

    // BITBOARD: O(1) per tile summation using bitwise helpers
    for (cx, cy, tile) in board.tiles.iter() {
        let bits = tile.occ_all & !tile.occ_void & !tile.occ_pawns;
        if bits == 0 {
            continue;
        }

        let n = bits.count_ones() as i64;
        sum_x += n * cx * 8 + tile.sum_lx(bits) as i64;
        sum_y += n * cy * 8 + tile.sum_ly(bits) as i64;
        count += n;
    }

    if count > 0 {
        Some(Coordinate {
            x: sum_x / count,
            y: sum_y / count,
        })
    } else {
        None
    }
}

// ==================== Main Evaluation ====================

/// Lazy evaluation: Material + Simple Position (PST)
/// Used for Null Move Pruning and other heuristics where speed is critical.
pub fn evaluate_lazy(game: &GameState) -> i32 {
    let mut score = game.material_score;

    // BITBOARD: Use tile-based CTZ iteration for O(popcount) positional scoring
    for (cx, cy, tile) in game.board.tiles.iter() {
        // SIMD: Fast skip empty tiles
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
            let piece = crate::board::Piece::from_packed(packed);

            let mut positional_bonus = 0;
            let x = cx * 8 + (idx % 8) as i64;
            let y = cy * 8 + (idx / 8) as i64;

            match piece.piece_type() {
                PieceType::Pawn => {
                    let rank = y;
                    let promotion_rank = if piece.color() == PlayerColor::White {
                        game.white_promo_rank
                    } else {
                        game.black_promo_rank
                    };
                    let distance_to_promo = (promotion_rank - rank).abs();
                    if distance_to_promo < 6 {
                        positional_bonus += (6 - distance_to_promo) as i32 * 5;
                    }
                    if x >= 2 && x <= 5 {
                        positional_bonus += 5;
                    }
                }
                PieceType::Knight | PieceType::Bishop => {
                    if x >= 2 && x <= 5 && y >= 2 && y <= 5 {
                        positional_bonus += 10;
                    }
                }
                _ => {
                    if x >= 2 && x <= 5 && y >= 2 && y <= 5 {
                        positional_bonus += 2;
                    }
                }
            }

            if piece.color() == PlayerColor::White {
                score += positional_bonus;
            } else {
                score -= positional_bonus;
            }
        }
    }

    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

pub fn evaluate(game: &GameState) -> i32 {
    // Check for insufficient material draw
    match crate::evaluation::insufficient_material::evaluate_insufficient_material(game) {
        Some(0) => return 0, // Dead draw
        Some(divisor) => {
            // Drawish - dampen eval
            return evaluate_inner(game) / divisor;
        }
        None => {} // Sufficient - continue to normal eval
    }

    evaluate_inner(game)
}

/// Core evaluation logic - skips insufficient material check
#[inline]
fn evaluate_inner(game: &GameState) -> i32 {
    // Start with material score
    let mut score = game.material_score;

    // Use cached king positions (O(1) instead of O(n) board scan)
    let (white_king, black_king) = (game.white_king_pos, game.black_king_pos);

    // Try scaled mop-up evaluation based on material imbalance
    // This runs when one side has < 40% of starting material
    let mut mop_up_applied = false;

    // Check if black is losing (white has material advantage or black has few pieces)
    if let Some(_scale) =
        crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black)
    {
        if let (Some(wk), Some(bk)) = (&white_king, &black_king) {
            // Don't prioritize mop-up if we have promotable pawns
            let white_has_promo = crate::evaluation::mop_up::has_promotable_pawn(
                &game.board,
                PlayerColor::White,
                game.white_promo_rank,
            );
            if !white_has_promo {
                score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
                    game,
                    wk,
                    bk,
                    PlayerColor::White,
                    PlayerColor::Black,
                );
                mop_up_applied = true;
            }
        }
    }

    // Check if white is losing (black has material advantage or white has few pieces)
    if !mop_up_applied {
        if let Some(_scale) =
            crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White)
        {
            if let (Some(wk), Some(bk)) = (&white_king, &black_king) {
                let black_has_promo = crate::evaluation::mop_up::has_promotable_pawn(
                    &game.board,
                    PlayerColor::Black,
                    game.black_promo_rank,
                );
                if !black_has_promo {
                    score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
                        game,
                        bk,
                        wk,
                        PlayerColor::Black,
                        PlayerColor::White,
                    );
                    mop_up_applied = true;
                }
            }
        }
    }

    // If mop-up wasn't applied, use normal positional evaluation
    if !mop_up_applied {
        score += evaluate_pieces(game, &white_king, &black_king);
        score += evaluate_king_safety(game, &white_king, &black_king);
        score += evaluate_pawn_structure(game);
    }

    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

// ==================== Piece Evaluation ====================

pub fn evaluate_pieces(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut score: i32 = 0;

    let white_promo_rank = game.white_promo_rank;
    let black_promo_rank = game.black_promo_rank;

    let game_phase = calculate_game_phase(game);
    let is_opening = game_phase >= MIDDLEGAME_PIECE_THRESHOLD;
    let is_endgame = game_phase < ENDGAME_PIECE_THRESHOLD;

    // BITBOARD: Pass 1 - Single metadata collection
    let mut white_undeveloped = 0;
    let mut black_undeveloped = 0;
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let mut white_bishop_colors: (bool, bool) = (false, false);
    let mut black_bishop_colors: (bool, bool) = (false, false);
    let mut cloud_sum_x: i64 = 0;
    let mut cloud_sum_y: i64 = 0;
    let mut cloud_count: i64 = 0;

    for (cx, cy, tile) in game.board.tiles.iter() {
        // SIMD: Fast skip empty tiles using parallel zero check
        if crate::simd::both_zero(tile.occ_white, tile.occ_black) {
            continue;
        }

        let cloud_bits = tile.occ_all & !tile.occ_void & !tile.occ_pawns;
        if cloud_bits != 0 {
            let n = cloud_bits.count_ones() as i64;
            cloud_sum_x += n * cx * 8 + tile.sum_lx(cloud_bits) as i64;
            cloud_sum_y += n * cy * 8 + tile.sum_ly(cloud_bits) as i64;
            cloud_count += n;
        }

        let w_minors = tile.occ_white & (tile.occ_knights | tile.occ_bishops);
        if w_minors != 0 {
            let mut bits = w_minors;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                if game.starting_squares.contains(&Coordinate::new(x, y)) {
                    white_undeveloped += 1;
                }
                if ((1 << idx) & tile.occ_bishops) != 0 {
                    white_bishops += 1;
                    if (x + y) % 2 == 0 {
                        white_bishop_colors.0 = true;
                    } else {
                        white_bishop_colors.1 = true;
                    }
                }
            }
        }

        let b_minors = tile.occ_black & (tile.occ_knights | tile.occ_bishops);
        if b_minors != 0 {
            let mut bits = b_minors;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                if game.starting_squares.contains(&Coordinate::new(x, y)) {
                    black_undeveloped += 1;
                }
                if ((1 << idx) & tile.occ_bishops) != 0 {
                    black_bishops += 1;
                    if (x + y) % 2 == 0 {
                        black_bishop_colors.0 = true;
                    } else {
                        black_bishop_colors.1 = true;
                    }
                }
            }
        }
    }

    let cloud_center = if cloud_count > 0 {
        Some(Coordinate {
            x: cloud_sum_x / cloud_count,
            y: cloud_sum_y / cloud_count,
        })
    } else {
        None
    };

    let white_attack_scale = if white_undeveloped >= UNDEVELOPED_MINORS_THRESHOLD {
        DEVELOPMENT_PHASE_ATTACK_SCALE
    } else {
        DEVELOPED_PHASE_ATTACK_SCALE
    };
    let black_attack_scale = if black_undeveloped >= UNDEVELOPED_MINORS_THRESHOLD {
        DEVELOPMENT_PHASE_ATTACK_SCALE
    } else {
        DEVELOPED_PHASE_ATTACK_SCALE
    };

    // BITBOARD: Pass 2 - Main evaluation
    for (cx, cy, tile) in game.board.tiles.iter() {
        // SIMD: Fast skip empty tiles
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
            let piece = crate::board::Piece::from_packed(packed);
            let x = cx * 8 + (idx % 8) as i64;
            let y = cy * 8 + (idx / 8) as i64;

            let mut piece_score = match piece.piece_type() {
                PieceType::Rook => evaluate_rook(game, x, y, piece.color(), white_king, black_king),
                PieceType::Queen => {
                    evaluate_queen(game, x, y, piece.color(), white_king, black_king)
                }
                PieceType::Knight => evaluate_knight(x, y, piece.color(), black_king, white_king),
                PieceType::Bishop => {
                    evaluate_bishop(game, x, y, piece.color(), white_king, black_king)
                }
                PieceType::Pawn => {
                    let pawn_eval = evaluate_pawn_position(
                        x,
                        y,
                        piece.color(),
                        white_promo_rank,
                        black_promo_rank,
                    );
                    if is_opening {
                        if x >= 3 && x <= 6 { 5 } else { 0 }
                    } else if is_endgame {
                        pawn_eval
                    } else {
                        let scale = 100
                            - ((game_phase - ENDGAME_PIECE_THRESHOLD) * 100
                                / (MIDDLEGAME_PIECE_THRESHOLD - ENDGAME_PIECE_THRESHOLD));
                        pawn_eval * scale / 100
                    }
                }
                PieceType::Chancellor => {
                    let rook_eval =
                        evaluate_rook(game, x, y, piece.color(), white_king, black_king);
                    let ek = if piece.color() == PlayerColor::White {
                        black_king
                    } else {
                        white_king
                    };
                    let mut b = 0;
                    if let Some(k) = ek {
                        let d = (x - k.x).abs() + (y - k.y).abs();
                        if d <= 4 {
                            b = 15;
                        } else if d <= 8 {
                            b = 8;
                        }
                    }
                    (rook_eval * CHANCELLOR_ROOK_SCALE / 100) + b
                }
                PieceType::Archbishop => {
                    let bishop_eval =
                        evaluate_bishop(game, x, y, piece.color(), white_king, black_king);
                    let ek = if piece.color() == PlayerColor::White {
                        black_king
                    } else {
                        white_king
                    };
                    let mut b = 0;
                    if let Some(k) = ek {
                        let d = (x - k.x).abs() + (y - k.y).abs();
                        if d <= 4 {
                            b = 15;
                        } else if d <= 8 {
                            b = 8;
                        }
                    }
                    (bishop_eval * ARCHBISHOP_BISHOP_SCALE / 100) + b
                }
                PieceType::Amazon => {
                    let queen_eval =
                        evaluate_queen(game, x, y, piece.color(), white_king, black_king);
                    let rook_eval =
                        evaluate_rook(game, x, y, piece.color(), white_king, black_king);
                    let ek = if piece.color() == PlayerColor::White {
                        black_king
                    } else {
                        white_king
                    };
                    let mut b = 0;
                    if let Some(k) = ek {
                        let d = (x - k.x).abs() + (y - k.y).abs();
                        if d <= 4 {
                            b = 20;
                        } else if d <= 8 {
                            b = 10;
                        }
                    }
                    (queen_eval * AMAZON_QUEEN_SCALE / 100)
                        + (rook_eval * AMAZON_ROOK_SCALE / 100)
                        + b
                }
                PieceType::RoyalQueen => {
                    evaluate_queen(game, x, y, piece.color(), white_king, black_king)
                }
                PieceType::Knightrider => {
                    evaluate_knightrider(x, y, piece.color(), white_king, black_king, &game.board)
                }
                PieceType::Hawk
                | PieceType::Rose
                | PieceType::Camel
                | PieceType::Giraffe
                | PieceType::Zebra => evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    cloud_center.as_ref(),
                    get_piece_value(piece.piece_type()),
                ),
                PieceType::Centaur | PieceType::RoyalCentaur => {
                    let knight_eval = evaluate_knight(x, y, piece.color(), black_king, white_king);
                    let leaper_eval = evaluate_leaper_positioning(
                        x,
                        y,
                        piece.color(),
                        white_king,
                        black_king,
                        cloud_center.as_ref(),
                        get_piece_value(piece.piece_type()),
                    );
                    (knight_eval * CENTAUR_KNIGHT_SCALE / 100)
                        + (leaper_eval * CENTAUR_GUARD_SCALE / 100)
                }
                PieceType::Huygen => evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    cloud_center.as_ref(),
                    get_piece_value(PieceType::Huygen),
                ),
                PieceType::Guard => evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    cloud_center.as_ref(),
                    get_piece_value(PieceType::Guard),
                ),
                _ => 0,
            };

            if let Some(center) = &cloud_center {
                let cheb = (x - center.x).abs().max((y - center.y).abs());
                if piece.piece_type() != PieceType::Pawn && !piece.piece_type().is_royal() {
                    if cheb > PIECE_CLOUD_CHEB_RADIUS {
                        let piece_val = get_piece_value(piece.piece_type());
                        let value_factor = (piece_val / 100).max(1);
                        let excess = (cheb - PIECE_CLOUD_CHEB_RADIUS)
                            .min(PIECE_CLOUD_CHEB_MAX_EXCESS)
                            as i32;
                        piece_score -= excess * CLOUD_PENALTY_PER_100_VALUE * value_factor;
                    }
                }
            }

            if piece.piece_type() != PieceType::Pawn && !piece.piece_type().is_royal() {
                if game.starting_squares.contains(&Coordinate::new(x, y)) {
                    piece_score -= match piece.piece_type() {
                        PieceType::Knight | PieceType::Bishop => MIN_DEVELOPMENT_PENALTY + 3,
                        PieceType::Archbishop => MIN_DEVELOPMENT_PENALTY,
                        _ => 0,
                    };
                }
            }

            let own_king = if piece.color() == PlayerColor::White {
                &white_king
            } else {
                &black_king
            };
            if let Some(ok) = own_king {
                if !piece.piece_type().is_royal() && piece.piece_type() != PieceType::Pawn {
                    let dist = (x - ok.x).abs().max((y - ok.y).abs());
                    if dist <= 3 {
                        if get_piece_value(piece.piece_type()) < KING_DEFENDER_VALUE_THRESHOLD {
                            piece_score += KING_DEFENDER_BONUS;
                        } else {
                            piece_score -= KING_ATTACKER_NEAR_OWN_KING_PENALTY;
                        }
                    }
                }
            }

            let is_attacking_piece = matches!(
                piece.piece_type(),
                PieceType::Rook
                    | PieceType::Queen
                    | PieceType::RoyalQueen
                    | PieceType::Bishop
                    | PieceType::Chancellor
                    | PieceType::Archbishop
                    | PieceType::Amazon
            );
            if is_attacking_piece && piece_score > 0 {
                let attack_scale = if piece.color() == PlayerColor::White {
                    white_attack_scale
                } else {
                    black_attack_scale
                };
                piece_score = piece_score * attack_scale / 100;
            }

            if piece.color() == PlayerColor::White {
                score += piece_score;
            } else {
                score -= piece_score;
            }
        }
    }

    if white_bishops >= 2 {
        score += BISHOP_PAIR_BONUS;
        bump_feat!(bishop_pair_bonus, 1);
        if white_bishop_colors.0 && white_bishop_colors.1 {
            score += 20;
        }
    }
    if black_bishops >= 2 {
        score -= BISHOP_PAIR_BONUS;
        bump_feat!(bishop_pair_bonus, -1);
        if black_bishop_colors.0 && black_bishop_colors.1 {
            score -= 20;
        }
    }

    score
}

pub fn evaluate_rook(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut bonus: i32 = 0;

    // Behind enemy king bonus and rook tropism.
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        // Behind enemy king along the rank direction.
        if (color == PlayerColor::White && y > ek.y) || (color == PlayerColor::Black && y < ek.y) {
            bonus += BEHIND_KING_BONUS;
        }

        // On same or adjacent file to enemy king: strong attacking potential.
        if (x - ek.x).abs() <= 1 {
            bonus += 50;
        }

        // Rook tropism: closer to enemy king is better, but capped.
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let capped = dist.min(25);
        let trop = (25 - capped) as i32;
        bonus += trop * KING_TROPISM_BONUS;

        // Simplified confinement bonus - just reward rooks controlling key squares near king
        // Much faster than the previous complex detection
        let mut confinement_bonus = 0;

        // Rook on same rank as king - controls king's horizontal movement
        if y == ek.y {
            if (x - ek.x).abs() <= 3 {
                confinement_bonus += 30;
            }
        }

        // Rook on same file as king - controls king's vertical movement
        if x == ek.x {
            if (y - ek.y).abs() <= 3 {
                confinement_bonus += 30;
            }
        }

        // Rook adjacent to king - immediate pressure
        if (x - ek.x).abs() <= 1 && (y - ek.y).abs() <= 1 {
            confinement_bonus += 40;
        }

        bonus += confinement_bonus;

        // Simplified slider coordination - just count nearby sliders without iteration
        // Much faster than previous full board scan
        let nearby_slider_bonus = if (x - ek.x).abs() <= 4 && (y - ek.y).abs() <= 4 {
            // This rook is close to king, assume some coordination exists
            SLIDER_NET_BONUS / 2
        } else {
            0
        };

        bonus += nearby_slider_bonus;

        // Penalize rooks that have drifted very far from the king zone (both
        // own and enemy kings). On an infinite board, there is rarely value
        // in a rook being dozens of squares away from *both* kings.
        let mut cheb = (x - ek.x).abs().max((y - ek.y).abs());
        let own_king_ref = if color == PlayerColor::White {
            white_king
        } else {
            black_king
        };
        if let Some(ok) = own_king_ref {
            let cheb_own = (x - ok.x).abs().max((y - ok.y).abs());
            cheb = cheb.min(cheb_own);
        }

        if cheb > FAR_SLIDER_CHEB_RADIUS {
            let excess = (cheb - FAR_SLIDER_CHEB_RADIUS).min(FAR_SLIDER_CHEB_MAX_EXCESS) as i32;
            bonus -= excess * FAR_ROOK_PENALTY;
        }
    }

    // Penalize completely idle rooks stuck behind both own and enemy pawns on their file.
    let (own_pawns_on_file, enemy_pawns_on_file) = count_pawns_on_file(game, x, color);
    if own_pawns_on_file > 0 && enemy_pawns_on_file > 0 {
        bonus -= ROOK_IDLE_PENALTY;
        bump_feat!(rook_idle_penalty, -1);
    }

    bonus
}

pub fn evaluate_queen(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut bonus: i32 = 0;

    // Queen should aggressively aim at the enemy king from a safe distance.
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        let dx = ek.x - x;
        let dy = ek.y - y;
        let same_file = dx == 0;
        let same_rank = dy == 0;
        let same_diag = dx.abs() == dy.abs();

        let from = Coordinate { x, y };

        if same_file || same_rank || same_diag {
            // Reward only if the line is clear between queen and king (direct pressure).
            if is_clear_line_between(&game.board, &from, ek) {
                // Base line-attack bonus - reduced to avoid queen chasing king too eagerly
                let mut line_bonus: i32 = 15;
                let lin_dist = dx.abs().max(dy.abs()) as i32;
                let max_lin: i32 = 20;
                let clamped = lin_dist.min(max_lin);
                let diff = (clamped - QUEEN_IDEAL_LINE_DIST).abs();
                let base = (max_lin - diff * 2).max(0);
                // Reduce the distance score weight
                let distance_score = base * (KING_TROPISM_BONUS / 2).max(1);

                line_bonus += distance_score;
                bonus += line_bonus;

                // Small bonus for being "behind" the king - reduced from 30 to avoid backrank obsession
                if (color == PlayerColor::White && y > ek.y)
                    || (color == PlayerColor::Black && y < ek.y)
                {
                    bonus += 10;
                }
            }
        }

        // General tropism bonus based on Manhattan distance (king proximity).
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let manhattan_score = (20 - dist.min(20)) as i32;
        let manhattan_weight = (KING_TROPISM_BONUS / 2).max(1);
        bonus += manhattan_score * manhattan_weight;

        // Penalize queens that are extremely far from the *king zone*.
        // We take the minimum Chebyshev distance to own and enemy kings so
        // that wandering far away from both is discouraged.
        let mut cheb = (x - ek.x).abs().max((y - ek.y).abs());
        let own_king_ref = if color == PlayerColor::White {
            white_king
        } else {
            black_king
        };
        if let Some(ok) = own_king_ref {
            let cheb_own = (x - ok.x).abs().max((y - ok.y).abs());
            cheb = cheb.min(cheb_own);
        }

        if cheb > FAR_SLIDER_CHEB_RADIUS {
            let excess = (cheb - FAR_SLIDER_CHEB_RADIUS).min(FAR_SLIDER_CHEB_MAX_EXCESS) as i32;
            bonus -= excess * FAR_QUEEN_PENALTY;
        }
    }

    // Queen mobility
    // Directional: we mostly care about mobility towards the enemy side and
    // along central files/ranks, not running off to infinity behind our own
    // king.
    let queen_dirs_white: &[(i64, i64)] = &[
        (1, 0),  // horizontal
        (-1, 0), // horizontal
        (0, 1),  // forward (towards black)
        (1, 1),  // forward diagonals
        (-1, 1),
    ];
    let queen_dirs_black: &[(i64, i64)] = &[
        (1, 0),  // horizontal
        (-1, 0), // horizontal
        (0, -1), // forward (towards white)
        (1, -1), // forward diagonals
        (-1, -1),
    ];
    let queen_dirs_neutral: &[(i64, i64)] = &[
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];

    let queen_dirs: &[(i64, i64)] = match color {
        PlayerColor::White => queen_dirs_white,
        PlayerColor::Black => queen_dirs_black,
        PlayerColor::Neutral => queen_dirs_neutral,
    };

    let mobility = slider_mobility(&game.board, x, y, queen_dirs, 12);
    bonus += mobility * SLIDER_MOBILITY_BONUS;
    bump_feat!(slider_mobility_bonus, mobility);

    bonus
}

pub fn evaluate_knight(
    x: i64,
    y: i64,
    color: PlayerColor,
    black_king: &Option<Coordinate>,
    white_king: &Option<Coordinate>,
) -> i32 {
    let mut bonus: i32 = 0;

    // Knights are weak in infinite chess overall, but we still reward good king-related placement.

    // Small bonus for being near friendly king (defensive knight).
    let own_king = if color == PlayerColor::White {
        white_king
    } else {
        black_king
    };
    if let Some(ok) = own_king {
        let dist = (x - ok.x).abs() + (y - ok.y).abs();
        if dist <= 3 {
            bonus += KNIGHT_NEAR_KING_BONUS;
        } else if dist <= 5 {
            bonus += KNIGHT_NEAR_KING_BONUS / 2;
        }
    }

    // Small bonus for being near enemy king (fork and mating net potential).
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        if dist <= 3 {
            bonus += 10;
        }
    }

    // Mild centralization bonus.
    if x >= 3 && x <= 5 && y >= 3 && y <= 5 {
        bonus += 5;
    }

    bonus
}

pub fn evaluate_bishop(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut bonus: i32 = 0;

    // Long diagonal control bonus: bishops near "main" diagonals get a small bonus.
    if (x - y).abs() <= 1 || (x + y - 8).abs() <= 1 {
        bonus += 8;
    }

    // Behind enemy king bonus and bishop tropism.
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        // Bishop behind enemy king along the rank direction (less direct than rook/queen).
        if (color == PlayerColor::White && y > ek.y) || (color == PlayerColor::Black && y < ek.y) {
            bonus += BEHIND_KING_BONUS / 2;
        }

        // Bishop tropism: closer diagonally to enemy king is good.
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let capped = dist.min(15);
        let trop = (15 - capped) as i32;
        bonus += trop * (KING_TROPISM_BONUS / 2).max(1);

        // Penalize bishops that are extremely far from the king zone
        // (minimum of distance to own and enemy kings).
        let mut cheb = (x - ek.x).abs().max((y - ek.y).abs());
        let own_king_ref = if color == PlayerColor::White {
            white_king
        } else {
            black_king
        };
        if let Some(ok) = own_king_ref {
            let cheb_own = (x - ok.x).abs().max((y - ok.y).abs());
            cheb = cheb.min(cheb_own);
        }

        if cheb > FAR_SLIDER_CHEB_RADIUS {
            let excess = (cheb - FAR_SLIDER_CHEB_RADIUS).min(FAR_SLIDER_CHEB_MAX_EXCESS) as i32;
            bonus -= excess * FAR_BISHOP_PENALTY;
        }
    }

    // Bishop mobility: prefer activity on diagonals pointing towards the
    // enemy side. Backward diagonals are mostly defensive and should not
    // encourage bishops to run off to infinity behind our own king.
    let bishop_dirs_white: &[(i64, i64)] = &[(1, 1), (-1, 1)];
    let bishop_dirs_black: &[(i64, i64)] = &[(1, -1), (-1, -1)];
    let bishop_dirs_neutral: &[(i64, i64)] = &[(1, 1), (1, -1), (-1, 1), (-1, -1)];

    let bishop_dirs: &[(i64, i64)] = match color {
        PlayerColor::White => bishop_dirs_white,
        PlayerColor::Black => bishop_dirs_black,
        PlayerColor::Neutral => bishop_dirs_neutral,
    };

    let mobility = slider_mobility(&game.board, x, y, bishop_dirs, 12);
    bonus += mobility * BISHOP_MOBILITY_BONUS;
    bump_feat!(bishop_mobility_bonus, mobility);

    bonus
}

pub fn evaluate_pawn_position(
    x: i64,
    y: i64,
    color: PlayerColor,
    white_promo_rank: i64,
    black_promo_rank: i64,
) -> i32 {
    let mut bonus: i32 = 0;

    // Advancement bonus and distance penalty based on distance to promotion.
    // In infinite chess, back pawns are nearly worthless - they take many
    // tempos to promote and don't contribute to the position.
    match color {
        PlayerColor::White => {
            let dist = (white_promo_rank - y).max(0);

            // Check if pawn is PAST promotion rank (can't promote - worthless!)
            if y > white_promo_rank {
                bonus -= PAWN_PAST_PROMO_PENALTY;
            } else if dist > PAWN_FULL_VALUE_THRESHOLD {
                // Far from promotion - flat penalty, NO benefit from advancing
                // This prevents the engine from wasting tempos pushing back pawns
                bonus -= PAWN_FAR_FROM_PROMO_PENALTY;
            } else {
                // Close to promotion - advancement bonus applies
                bonus += ((PAWN_FULL_VALUE_THRESHOLD - dist) as i32) * 4;
            }
        }
        PlayerColor::Black => {
            let dist = (y - black_promo_rank).max(0);

            // Past promotion rank check
            if y < black_promo_rank {
                bonus -= PAWN_PAST_PROMO_PENALTY;
            } else if dist > PAWN_FULL_VALUE_THRESHOLD {
                // Far from promotion - flat penalty, NO benefit from advancing
                bonus -= PAWN_FAR_FROM_PROMO_PENALTY;
            } else {
                // Close to promotion - advancement bonus
                bonus += ((PAWN_FULL_VALUE_THRESHOLD - dist) as i32) * 4;
            }
        }
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    }

    // Central pawns are valuable
    if x >= 3 && x <= 5 {
        bonus += 5;
    }

    bonus
}

// ==================== Fairy Piece Evaluation ====================

/// Evaluate leaper pieces (Hawk, Rose, Camel, Giraffe, Zebra, etc.)
/// Uses tropism (distance to kings) and cloud proximity since mobility is meaningless on infinite board
fn evaluate_leaper_positioning(
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    cloud_center: Option<&Coordinate>,
    piece_value: i32,
) -> i32 {
    let mut bonus: i32 = 0;

    // Scale factor based on piece value (stronger pieces get bigger positioning bonuses)
    let scale = (piece_value / LEAPER_TROPISM_DIVISOR).max(1);

    // Tropism: reward being close to enemy king
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs().max((y - ek.y).abs()); // Chebyshev distance
        let capped_dist = dist.min(LEAPER_MAX_TROPISM_DIST);
        let tropism_bonus = (LEAPER_MAX_TROPISM_DIST - capped_dist) as i32;
        bonus += tropism_bonus * scale;
    }

    // Reward being close to own king (defensive potential)
    let own_king = if color == PlayerColor::White {
        white_king
    } else {
        black_king
    };
    if let Some(ok) = own_king {
        let dist = (x - ok.x).abs().max((y - ok.y).abs());
        if dist <= 4 {
            bonus += (5 - dist as i32) * (scale / 2).max(1);
        }
    }

    // Reward being close to piece cloud center (activity)
    if let Some(center) = cloud_center {
        let dist = (x - center.x).abs().max((y - center.y).abs());
        if dist <= 10 {
            bonus += (11 - dist as i32) * (scale / 3).max(1);
        }
    }

    bonus
}

/// Evaluate knightrider (extended knight that can continue in knight patterns)
fn evaluate_knightrider(
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    board: &Board,
) -> i32 {
    let mut bonus: i32 = 0;

    // Knight offsets for all 8 directions
    const KNIGHT_DIRS: [(i64, i64); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];

    // Count how far the knightrider can travel in each direction
    let mut total_reach = 0;
    for (dx, dy) in KNIGHT_DIRS {
        let mut nx = x + dx;
        let mut ny = y + dy;
        let mut steps = 0;
        // Count empty squares along the knight-ray (max 5 steps for efficiency)
        while steps < 5 {
            if board.get_piece(nx, ny).is_some() {
                break;
            }
            steps += 1;
            nx += dx;
            ny += dy;
        }
        total_reach += steps;
    }
    bonus += total_reach * KNIGHTRIDER_RAY_BONUS;

    // Tropism to enemy king
    let enemy_king = if color == PlayerColor::White {
        black_king
    } else {
        white_king
    };
    if let Some(ek) = enemy_king {
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        if dist <= 10 {
            bonus += (11 - dist as i32) * 3;
        }
    }

    // Near own king bonus
    let own_king = if color == PlayerColor::White {
        white_king
    } else {
        black_king
    };
    if let Some(ok) = own_king {
        let dist = (x - ok.x).abs() + (y - ok.y).abs();
        if dist <= 5 {
            bonus += KNIGHT_NEAR_KING_BONUS;
        }
    }

    bonus
}

// ==================== King Safety ====================

pub fn evaluate_king_safety(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
) -> i32 {
    let mut score: i32 = 0;

    // White king safety
    if let Some(wk) = white_king {
        score += evaluate_king_shelter(game, wk, PlayerColor::White);
    }

    // Black king safety
    if let Some(bk) = black_king {
        score -= evaluate_king_shelter(game, bk, PlayerColor::Black);
    }

    score
}

fn evaluate_king_shelter(game: &GameState, king: &Coordinate, color: PlayerColor) -> i32 {
    let mut safety: i32 = 0;

    // 1. Local pawn / guard cover (only track presence; scoring is via missing-penalty)
    let mut has_ring_cover = false;
    for dx in -1..=1_i64 {
        for dy in -1..=1_i64 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let cx = king.x + dx;
            let cy = king.y + dy;
            if let Some(piece) = game.board.get_piece(cx, cy) {
                if piece.color() == color {
                    if piece.piece_type() == PieceType::Pawn
                        || piece.piece_type() == PieceType::Guard
                        || piece.piece_type() == PieceType::Void
                    {
                        has_ring_cover = true;
                    }
                }
            }
        }
    }
    if !has_ring_cover {
        safety -= KING_RING_MISSING_PENALTY;
        bump_feat!(king_ring_missing_penalty, -1);
    }

    // 1b. King relative to own pawn chain: prefer being behind pawns rather than ahead of them.
    // We look for pawns roughly on the same files as the king (+/- 2 files) to keep it local.
    let mut has_pawn_ahead = false;
    let mut has_pawn_behind = false;

    // BITBOARD: Use tile-based pawn iteration
    let is_white = color == PlayerColor::White;
    for (cx, cy, tile) in game.board.tiles.iter() {
        let color_pawns = tile.occ_pawns
            & if is_white {
                tile.occ_white
            } else {
                tile.occ_black
            };
        if color_pawns == 0 {
            continue;
        }

        let mut bits = color_pawns;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let px = cx * 8 + (idx % 8) as i64;
            let py = cy * 8 + (idx / 8) as i64;

            // Only consider pawns near the king in file-space
            if (px - king.x).abs() > 2 {
                continue;
            }

            match color {
                PlayerColor::White => {
                    if py > king.y {
                        has_pawn_ahead = true;
                    } else if py < king.y {
                        has_pawn_behind = true;
                    }
                }
                PlayerColor::Black => {
                    if py < king.y {
                        has_pawn_ahead = true;
                    } else if py > king.y {
                        has_pawn_behind = true;
                    }
                }
                PlayerColor::Neutral => {}
            }
        }
    }

    // If there are pawns both ahead and behind, we assume a mixed structure and stay neutral.
    if has_pawn_ahead && !has_pawn_behind {
        // King is safely tucked behind its pawn shield.
        safety += KING_PAWN_SHIELD_BONUS;
    } else if !has_pawn_ahead && has_pawn_behind {
        safety -= KING_PAWN_AHEAD_PENALTY;
    }

    // 2. Open rays (more open lines = more vulnerable)
    // Count how many directions have no friendly piece cover
    let directions: &[(i64, i64)] = &[
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let mut open_direction_count = 0;
    for &(dx, dy) in directions {
        let mut cx = king.x;
        let mut cy = king.y;
        let mut ray_open = true;
        // limit ray length; tactics around king are local
        for _step in 0..8 {
            cx += dx;
            cy += dy;
            if let Some(piece) = game.board.get_piece(cx, cy) {
                // Friendly piece stops the ray and provides some cover
                if piece.color() == color {
                    ray_open = false;
                }
                break;
            }
        }
        if ray_open {
            open_direction_count += 1;
            safety -= KING_OPEN_RAY_PENALTY;
            bump_feat!(king_open_ray_penalty, -1);
        }
    }

    // Extra penalty for kings exposed in too many directions
    // This makes it dangerous to leave the king unprotected
    if open_direction_count >= KING_OPEN_DIRECTION_THRESHOLD {
        let excess = open_direction_count - KING_OPEN_DIRECTION_THRESHOLD;
        safety -= excess * KING_EXPOSURE_PENALTY_PER_DIR;
    }

    // 3. Enemy sliders attacking king zone
    let mut enemy_slider_threats = 0;
    for (cx, cy, tile) in game.board.tiles.iter() {
        let occ = if color == PlayerColor::White {
            tile.occ_black
        } else {
            tile.occ_white
        };
        let sliders = occ & (tile.occ_ortho_sliders | tile.occ_diag_sliders);
        if sliders == 0 {
            continue;
        }

        let mut bits = sliders;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let x = cx * 8 + (idx % 8) as i64;
            let y = cy * 8 + (idx / 8) as i64;

            // First check: is the slider even roughly in line with the king?
            let dx = x - king.x;
            let dy = y - king.y;

            let same_file = dx == 0;
            let same_rank = dy == 0;
            let same_diag = dx.abs() == dy.abs();

            if !(same_file || same_rank || same_diag) {
                continue;
            }

            // In infinite chess, ignore ridiculously far sliders for safety.
            let chebyshev = dx.abs().max(dy.abs());
            if chebyshev > 32 {
                continue;
            }

            // Now use the O(#pieces) line-of-sight check
            let from = Coordinate { x, y };
            if is_clear_line_between(&game.board, &from, king) {
                enemy_slider_threats += 1;
            }
        }
    }
    safety -= enemy_slider_threats * KING_ENEMY_SLIDER_PENALTY;
    bump_feat!(king_enemy_slider_penalty, -enemy_slider_threats);

    safety
}

// ==================== Pawn Structure ====================

pub fn evaluate_pawn_structure(game: &GameState) -> i32 {
    // Check cache first using game's pawn_hash
    let pawn_hash = game.pawn_hash;

    let cached = PAWN_CACHE.with(|cache| cache.borrow().get(&pawn_hash).copied());

    if let Some(score) = cached {
        return score;
    }

    // Cache miss - compute pawn structure
    let score = compute_pawn_structure(game);

    // Store in cache (limit cache size to avoid unbounded growth)
    PAWN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        // Clear cache if it grows too large (simple LRU alternative)
        if cache.len() > 16384 {
            cache.clear();
        }
        cache.insert(pawn_hash, score);
    });

    score
}

/// Core pawn structure computation. Called on cache miss.
fn compute_pawn_structure(game: &GameState) -> i32 {
    let mut score: i32 = 0;

    // Track pawns per file for each color
    let mut white_pawn_files: Vec<i64> = Vec::new(); // For inter-tile doubled pawn check
    let mut black_pawn_files: Vec<i64> = Vec::new();
    let mut white_pawns: Vec<(i64, i64)> = Vec::new();
    let mut black_pawns: Vec<(i64, i64)> = Vec::new();

    // Column masks for bitwise doubled pawn check
    const COL_MASKS: [u64; 8] = [
        0x0101010101010101,
        0x0202020202020202,
        0x0404040404040404,
        0x0808080808080808,
        0x1010101010101010,
        0x2020202020202020,
        0x4040404040404040,
        0x8080808080808080,
    ];

    // BITBOARD: Use per-tile occ_pawns for faster collection and intra-tile doubled checks
    for (cx, cy, tile) in game.board.tiles.iter() {
        let w_pawns = tile.occ_pawns & tile.occ_white;
        let b_pawns = tile.occ_pawns & tile.occ_black;

        if w_pawns != 0 {
            // Intra-tile doubled pawn check
            for mask in COL_MASKS {
                let count = (w_pawns & mask).count_ones();
                if count > 1 {
                    score -= (count - 1) as i32 * DOUBLED_PAWN_PENALTY;
                }
                if count > 0 {
                    white_pawn_files.push(cx * 8 + (mask.trailing_zeros() % 8) as i64);
                }
            }

            // Collect for passed pawn check
            let mut bits = w_pawns;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                white_pawns.push((cx * 8 + (idx % 8) as i64, cy * 8 + (idx / 8) as i64));
            }
        }

        if b_pawns != 0 {
            // Same for black
            for mask in COL_MASKS {
                let count = (b_pawns & mask).count_ones();
                if count > 1 {
                    score += (count - 1) as i32 * DOUBLED_PAWN_PENALTY;
                }
                if count > 0 {
                    black_pawn_files.push(cx * 8 + (mask.trailing_zeros() % 8) as i64);
                }
            }

            let mut bits = b_pawns;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                black_pawns.push((cx * 8 + (idx % 8) as i64, cy * 8 + (idx / 8) as i64));
            }
        }
    }

    // Doubled pawns penalty and Passed Pawn / Tunnel detection
    // Note: We reuse the coordinate lists built above.
    // white_pawns and black_pawns are Vec<(x, y)>

    // --- WHITE PAWNS ---
    let mut prev_file: Option<i64> = None;
    white_pawn_files.sort();
    for &file in &white_pawn_files {
        if prev_file == Some(file) {
            score -= DOUBLED_PAWN_PENALTY;
            bump_feat!(doubled_pawn_penalty, -1);
        }
        prev_file = Some(file);
    }

    // Check passed pawns for White
    for (wx, wy) in &white_pawns {
        // 1. Is it passed? (No black pawns ahead on files x-1, x, x+1)
        let mut is_passed = true;
        for (bx, by) in &black_pawns {
            if (*bx - wx).abs() <= 1 && *by > *wy {
                is_passed = false;
                break;
            }
        }

        if is_passed {
            // Base bonus for passed pawn
            score += 20;
        }
    }

    // --- BLACK PAWNS ---
    let mut prev_file: Option<i64> = None;
    black_pawn_files.sort();
    for &file in &black_pawn_files {
        if prev_file == Some(file) {
            score += DOUBLED_PAWN_PENALTY;
            bump_feat!(doubled_pawn_penalty, 1);
        }
        prev_file = Some(file);
    }

    // Check passed pawns for Black
    for (bx, by) in &black_pawns {
        // 1. Is it passed? (No white pawns ahead (y < by) on files x-1, x, x+1)
        let mut is_passed = true;
        for (wx, wy) in &white_pawns {
            if (*wx - bx).abs() <= 1 && *wy < *by {
                is_passed = false;
                break;
            }
        }

        if is_passed {
            score -= 20;
        }
    }

    score
}

pub fn count_pawns_on_file(game: &GameState, file: i64, color: PlayerColor) -> (i32, i32) {
    let mut own_pawns = 0;
    let mut enemy_pawns = 0;

    // BITBOARD: Only check the tile(s) that contain this file
    let file_tile_x = file >> 3; // Which tile column contains this file
    let local_x = (file & 7) as usize;

    for (cx, _, tile) in game.board.tiles.iter() {
        if cx != file_tile_x {
            continue; // Skip tiles not on this file
        }

        // Check each row in the tile for pawns in the target column
        for row in 0..8 {
            let idx = row * 8 + local_x;
            if (tile.occ_pawns >> idx) & 1 != 0 {
                if (tile.occ_white >> idx) & 1 != 0 {
                    if color == PlayerColor::White {
                        own_pawns += 1;
                    } else {
                        enemy_pawns += 1;
                    }
                } else if (tile.occ_black >> idx) & 1 != 0 {
                    if color == PlayerColor::Black {
                        own_pawns += 1;
                    } else {
                        enemy_pawns += 1;
                    }
                }
            }
        }
    }

    (own_pawns, enemy_pawns)
}

fn is_between(a: i64, b: i64, c: i64) -> bool {
    let (minv, maxv) = if b < c { (b, c) } else { (c, b) };
    a > minv && a < maxv
}

pub fn slider_mobility(
    board: &Board,
    x: i64,
    y: i64,
    directions: &[(i64, i64)],
    max_steps: i64,
) -> i32 {
    let mut count: i32 = 0;

    for (dx, dy) in directions {
        let mut cx = x;
        let mut cy = y;
        for _ in 0..max_steps {
            cx += dx;
            cy += dy;
            if board.get_piece(cx, cy).is_some() {
                break;
            }
            count += 1;
        }
    }

    count
}

/// Returns true if the straight line between `from` and `to` is not blocked by any piece.
/// Works for ranks, files, and diagonals on an unbounded board by checking only existing pieces.
pub fn is_clear_line_between(board: &Board, from: &Coordinate, to: &Coordinate) -> bool {
    let dx = to.x - from.x;
    let dy = to.y - from.y;

    // Not collinear in rook/bishop directions -> we don't consider it a line for sliders.
    if !(dx == 0 || dy == 0 || dx.abs() == dy.abs()) {
        return false;
    }

    for ((px, py), _) in board.iter() {
        // Skip the endpoints themselves
        if *px == from.x && *py == from.y {
            continue;
        }
        if *px == to.x && *py == to.y {
            continue;
        }

        // Same file
        if dx == 0 && *px == from.x && is_between(*py, from.y, to.y) {
            return false;
        }

        // Same rank
        if dy == 0 && *py == from.y && is_between(*px, from.x, to.x) {
            return false;
        }

        // Same diagonal
        if dx.abs() == dy.abs() {
            let vx = *px - from.x;
            let vy = *py - from.y;
            // Collinear and between
            if vx * dy == vy * dx && is_between(*px, from.x, to.x) && is_between(*py, from.y, to.y)
            {
                return false;
            }
        }
    }

    true
}

pub fn calculate_initial_material(board: &Board) -> i32 {
    let mut score: i32 = 0;

    // BITBOARD: Use tile-based CTZ iteration for O(popcount) scan
    for (cx, cy, tile) in board.tiles.iter() {
        // SIMD: Fast skip empty tiles
        if crate::simd::both_zero(tile.occ_white, tile.occ_black) {
            continue;
        }

        // Process white pieces
        let mut white_bits = tile.occ_white;
        while white_bits != 0 {
            let idx = white_bits.trailing_zeros() as usize;
            white_bits &= white_bits - 1;

            let packed = tile.piece[idx];
            if packed != 0 {
                let piece = crate::board::Piece::from_packed(packed);
                score += get_piece_value(piece.piece_type());
            }
        }

        // Process black pieces
        let mut black_bits = tile.occ_black;
        while black_bits != 0 {
            let idx = black_bits.trailing_zeros() as usize;
            black_bits &= black_bits - 1;

            let packed = tile.piece[idx];
            if packed != 0 {
                let piece = crate::board::Piece::from_packed(packed);
                score -= get_piece_value(piece.piece_type());
            }
        }

        // Suppress unused variable warnings
        let _ = (cx, cy);
    }
    score
}
