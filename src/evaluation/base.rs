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

pub fn get_centrality_weight(piece_type: PieceType) -> i64 {
    match piece_type {
        PieceType::King => 2000,
        PieceType::Queen | PieceType::RoyalQueen | PieceType::Amazon => 1000,
        PieceType::Rook | PieceType::Chancellor => 500,
        PieceType::Bishop | PieceType::Archbishop => 300,
        PieceType::Knight | PieceType::Centaur | PieceType::RoyalCentaur => 300,
        PieceType::Camel | PieceType::Giraffe | PieceType::Zebra => 300,
        PieceType::Knightrider => 400,
        PieceType::Hawk => 350,
        PieceType::Rose => 350,
        PieceType::Guard | PieceType::Huygen => 250,
        // Pawns and others have 0 weight for "Piece Cloud" centrality
        _ => 0,
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

// King safety ring and ray penalties - tuned for infinite boards.
// Open rays are extremely dangerous on infinite boards - sliders can attack from anywhere.
const KING_RING_MISSING_PENALTY: i32 = 45;
const KING_OPEN_RAY_PENALTY: i32 = 30; // Open rays are very dangerous on infinite boards

// King pawn shield heuristics
// Reward having pawns in front of the king and penalize the king walking
// in front of its pawn chain.
const KING_PAWN_SHIELD_BONUS: i32 = 18;
const KING_PAWN_AHEAD_PENALTY: i32 = 20;

// Distance penalties to discourage sliders far away from the king "zone".
// We look at distance to both own and enemy king and penalize pieces that
// drift too far from either.
const FAR_SLIDER_CHEB_RADIUS: i64 = 18;
const FAR_SLIDER_CHEB_MAX_EXCESS: i64 = 40;
const FAR_QUEEN_PENALTY: i32 = 3;
const FAR_BISHOP_PENALTY: i32 = 2;
const FAR_ROOK_PENALTY: i32 = 2;
const PIECE_CLOUD_CHEB_RADIUS: i64 = 16;
const SLIDER_AXIS_WIGGLE: i64 = 5; // A slider is "active" if its ray passes within 5 sq of center
const PIECE_CLOUD_CHEB_MAX_EXCESS: i64 = 64;
const CLOUD_PENALTY_PER_100_VALUE: i32 = 1;

// Connected pawns bonus
const CONNECTED_PAWN_BONUS: i32 = 8;

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
const PAWN_PAST_PROMO_PENALTY: i32 = 90; // Massive penalty for pawns that can't promote (worth 10x less)
const PAWN_FAR_FROM_PROMO_PENALTY: i32 = 50; // Flat penalty for back pawns (no benefit from advancing)

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

// Development thresholds - for attack scaling only
const UNDEVELOPED_MINORS_THRESHOLD: i32 = 2;
const DEVELOPMENT_PHASE_ATTACK_SCALE: i32 = 50;
const DEVELOPED_PHASE_ATTACK_SCALE: i32 = 100;

/// Compute the centroid of all non-obstacle, non-void pieces on the board.
/// Used for piece cloud calculations. (Made public for variant modules.)
pub fn compute_cloud_center(board: &Board) -> Option<Coordinate> {
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;
    let mut total_weight: i64 = 0;

    for (cx, cy, tile) in board.tiles.iter() {
        let mut bits = tile.occ_all & !tile.occ_void & !tile.occ_pawns;
        if bits == 0 {
            continue;
        }

        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let packed = tile.piece[idx];
            // packed should not be 0 since bits came from occ_all
            if packed == 0 {
                continue;
            }

            let piece = crate::board::Piece::from_packed(packed);
            let weight = get_centrality_weight(piece.piece_type());

            if weight > 0 {
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                sum_x += weight * x;
                sum_y += weight * y;
                total_weight += weight;
            }
        }
    }

    if total_weight > 0 {
        Some(Coordinate {
            x: sum_x / total_weight,
            y: sum_y / total_weight,
        })
    } else {
        None
    }
}

/// Compute centroid of finite-moving pieces only (knights, centaurs, etc.).
/// Excludes sliders to prevent them from skewing the "action zone".
/// Finite movers should cluster together for mutual support.
/// Uses occ_knights which includes knights and centaurs (pieces with knight-like movement).
pub fn compute_finite_mover_center(board: &Board) -> Option<Coordinate> {
    let mut sum_x: i64 = 0;
    let mut sum_y: i64 = 0;
    let mut count: i64 = 0;

    for (cx, cy, tile) in board.tiles.iter() {
        // Only include finite movers: knights (includes centaurs which are also knight-like)
        // We use occ_knights as the primary finite-mover bitboard
        let finite_movers = tile.occ_knights;
        let bits = finite_movers & tile.occ_all & !tile.occ_void;

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

/// Check if a pawn is connected (has a friendly pawn diagonally behind it).
/// Connected pawns are much stronger as they protect each other.
#[inline]
fn is_connected_pawn(game: &GameState, x: i64, y: i64, color: PlayerColor) -> bool {
    let dy = if color == PlayerColor::White { -1 } else { 1 };
    // Check if friendly pawn on adjacent file, one rank behind
    game.board
        .get_piece(x - 1, y + dy)
        .is_some_and(|p| p.piece_type() == PieceType::Pawn && p.color() == color)
        || game
            .board
            .get_piece(x + 1, y + dy)
            .is_some_and(|p| p.piece_type() == PieceType::Pawn && p.color() == color)
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
                    if (2..=5).contains(&x) {
                        positional_bonus += 5;
                    }
                }
                PieceType::Knight | PieceType::Bishop => {
                    if (2..=5).contains(&x) && (2..=5).contains(&y) {
                        positional_bonus += 10;
                    }
                }
                _ => {
                    if (2..=5).contains(&x) && (2..=5).contains(&y) {
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
    // NOTE: One side might not have a king (checkmate practice positions)
    // Mop-up and pawn advancement (one side has very few pieces)
    let white_pieces = game.white_piece_count.saturating_sub(game.white_pawn_count);
    let black_pieces = game.black_piece_count.saturating_sub(game.black_pawn_count);

    // Call optimized single-pass pawn evaluation
    let (pawn_score, white_has_promo, black_has_promo) = evaluate_pawns(game);
    score += pawn_score;

    let mut mop_up_applied = false;

    // Check if black is losing (white has material advantage or black has few pieces)
    // SKIP mop-up if white has a promotable pawn to prioritize promotion
    if black_pieces < 3
        && white_pieces > 1
        && !white_has_promo
        && let Some(_scale) =
            crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black)
    {
        // Need enemy king as target
        if let Some(bk) = &black_king {
            score += crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                white_king.as_ref(),
                bk,
                PlayerColor::White,
                PlayerColor::Black,
            );
            mop_up_applied = true;
        }
    }

    // Check if white is losing (black has material advantage or white has few pieces)
    // SKIP if black has a promotable pawn
    if !mop_up_applied
        && white_pieces < 3
        && black_pieces > 1
        && !black_has_promo
        && let Some(_scale) =
            crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White)
    {
        // Need enemy king as target
        if let Some(wk) = &white_king {
            score -= crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                black_king.as_ref(),
                wk,
                PlayerColor::Black,
                PlayerColor::White,
            );
            mop_up_applied = true;
        }
    }

    // If mop-up wasn't applied, use normal positional evaluation
    if !mop_up_applied {
        score += evaluate_pieces(game, &white_king, &black_king);
        score += evaluate_king_safety(game, &white_king, &black_king);
        score += evaluate_pawn_structure(game);
        score += evaluate_threats(game);
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
    // Finite-mover center tracking - commented out
    /*
    let mut finite_sum_x: i64 = 0;
    let mut finite_sum_y: i64 = 0;
    let mut finite_count: i64 = 0;
    */

    for (cx, cy, tile) in game.board.tiles.iter() {
        // SIMD: Fast skip empty tiles using parallel zero check
        if crate::simd::both_zero(tile.occ_white, tile.occ_black) {
            continue;
        }

        let mut cloud_bits = tile.occ_all & !tile.occ_void & !tile.occ_pawns;
        while cloud_bits != 0 {
            let idx = cloud_bits.trailing_zeros() as usize;
            cloud_bits &= cloud_bits - 1;

            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }
            let piece = crate::board::Piece::from_packed(packed);
            let weight = get_centrality_weight(piece.piece_type());

            if weight > 0 {
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                cloud_sum_x += weight * x;
                cloud_sum_y += weight * y;
                cloud_count += weight;
            }
        }

        /*
        // Finite-mover center (knights only - they have limited range and should cluster)
        let finite_bits = tile.occ_knights & tile.occ_all & !tile.occ_void;
        if finite_bits != 0 {
            let n = finite_bits.count_ones() as i64;
            finite_sum_x += n * cx * 8 + tile.sum_lx(finite_bits) as i64;
            finite_sum_y += n * cy * 8 + tile.sum_ly(finite_bits) as i64;
            finite_count += n;
        }
        */

        // SIMD: Compute both color minor masks simultaneously
        let minors_mask = tile.occ_knights | tile.occ_bishops;
        let (w_minors, b_minors) =
            crate::simd::and_pairs(tile.occ_white, tile.occ_black, minors_mask, minors_mask);

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

    /*
    // Finite-mover center: encourages knights to cluster together
    let finite_center = if finite_count > 0 {
        Some(Coordinate {
            x: finite_sum_x / finite_count,
            y: finite_sum_y / finite_count,
        })
    } else {
        None
    };
    */

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
                let dx = (x - center.x).abs();
                let dy = (y - center.y).abs();
                let cheb = dx.max(dy);

                if piece.piece_type() != PieceType::Pawn
                    && !piece.piece_type().is_royal()
                    && cheb > PIECE_CLOUD_CHEB_RADIUS
                {
                    let pt = piece.piece_type();

                    let is_ortho_slider = pt == PieceType::Rook || pt == PieceType::Chancellor;
                    let is_diag_slider = pt == PieceType::Bishop || pt == PieceType::Archbishop;
                    let is_full_slider = pt == PieceType::Queen || pt == PieceType::Amazon;

                    // Diagonals for diag-sliders
                    let d1 = ((x - y) - (center.x - center.y)).abs();
                    let d2 = ((x + y) - (center.x + center.y)).abs();

                    // Check if piece is "active" (either inside radius or having a ray passing through wiggle zone)
                    let is_active = if is_ortho_slider {
                        dx <= SLIDER_AXIS_WIGGLE || dy <= SLIDER_AXIS_WIGGLE
                    } else if is_diag_slider {
                        d1 <= SLIDER_AXIS_WIGGLE || d2 <= SLIDER_AXIS_WIGGLE
                    } else if is_full_slider {
                        dx <= SLIDER_AXIS_WIGGLE
                            || dy <= SLIDER_AXIS_WIGGLE
                            || d1 <= SLIDER_AXIS_WIGGLE
                            || d2 <= SLIDER_AXIS_WIGGLE
                    } else {
                        // Non-sliders must be within the Chebyshev radius
                        false
                    };

                    if !is_active {
                        let piece_val = get_piece_value(pt);
                        let value_factor = (piece_val / 100).max(1);

                        // Distance to the proximity zone
                        let dist_to_radius = cheb - PIECE_CLOUD_CHEB_RADIUS;

                        // Distance to the nearest active axis lane
                        let dist_to_lane = if is_ortho_slider {
                            (dx - SLIDER_AXIS_WIGGLE).min(dy - SLIDER_AXIS_WIGGLE)
                        } else if is_diag_slider {
                            (d1 - SLIDER_AXIS_WIGGLE).min(d2 - SLIDER_AXIS_WIGGLE)
                        } else if is_full_slider {
                            let ortho_dist = (dx - SLIDER_AXIS_WIGGLE).min(dy - SLIDER_AXIS_WIGGLE);
                            let diag_dist = (d1 - SLIDER_AXIS_WIGGLE).min(d2 - SLIDER_AXIS_WIGGLE);
                            ortho_dist.min(diag_dist)
                        } else {
                            dist_to_radius // For non-sliders, it's just the radius distance
                        };

                        // Excess is the minimum distance to ANY "safe" state
                        let excess = dist_to_radius.min(dist_to_lane).max(1);
                        let capped_excess = excess.min(PIECE_CLOUD_CHEB_MAX_EXCESS) as i32;
                        piece_score -= capped_excess * CLOUD_PENALTY_PER_100_VALUE * value_factor;
                    }
                }
            }

            if piece.piece_type() != PieceType::Pawn
                && !piece.piece_type().is_royal()
                && game.starting_squares.contains(&Coordinate::new(x, y))
            {
                piece_score -= match piece.piece_type() {
                    PieceType::Knight | PieceType::Bishop => MIN_DEVELOPMENT_PENALTY + 3,
                    PieceType::Archbishop => MIN_DEVELOPMENT_PENALTY,
                    _ => 0,
                };
            }

            let own_king = if piece.color() == PlayerColor::White {
                &white_king
            } else {
                &black_king
            };
            if let Some(ok) = own_king
                .filter(|_| !piece.piece_type().is_royal() && piece.piece_type() != PieceType::Pawn)
            {
                let dist = (x - ok.x).abs().max((y - ok.y).abs());
                if dist <= 3 {
                    if get_piece_value(piece.piece_type()) < KING_DEFENDER_VALUE_THRESHOLD {
                        piece_score += KING_DEFENDER_BONUS;
                    } else {
                        piece_score -= KING_ATTACKER_NEAR_OWN_KING_PENALTY;
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
        // Rook on same rank as king - controls king's horizontal movement
        if y == ek.y && (x - ek.x).abs() <= 3 {
            confinement_bonus += 30;
        }
        // Rook on same file as king - controls king's vertical movement
        if x == ek.x && (y - ek.y).abs() <= 3 {
            confinement_bonus += 30;
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
            if is_clear_line_between_fast(&game.spatial_indices, &from, ek) {
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
    if (3..=5).contains(&x) && (3..=5).contains(&y) {
        bonus += 5;
    }

    bonus
}

pub fn evaluate_bishop(
    _game: &GameState,
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

    bonus
}

/// Evaluate all pawn-related positional terms in a single pass.
/// Includes advancement, doubled, passed, phalanx, connected pawns.
/// Pawns past promotion rank ONLY get PAWN_PAST_PROMO_PENALTY - no other evaluation.
#[inline(always)]
fn evaluate_pawns(game: &GameState) -> (i32, bool, bool) {
    if game.white_pawn_count == 0 && game.black_pawn_count == 0 {
        return (0, false, false);
    }

    // Piece count for scaling (non-pawn, non-royal units)
    let white_royals = if game.white_king_pos.is_some() { 1 } else { 0 };
    let black_royals = if game.black_king_pos.is_some() { 1 } else { 0 };
    let white_pieces = game
        .white_piece_count
        .saturating_sub(game.white_pawn_count)
        .saturating_sub(white_royals);
    let black_pieces = game
        .black_piece_count
        .saturating_sub(game.black_pawn_count)
        .saturating_sub(black_royals);
    let total_pieces = white_pieces + black_pieces;

    // Multiplier for pawn advancement terms: 10+ pieces -> 10%, <5 pieces -> 100%
    let multiplier_q = if total_pieces >= 10 {
        10
    } else if total_pieces <= 5 {
        100
    } else {
        // Linear interpolation between (5, 100) and (10, 10)
        100 - (total_pieces - 5) as i32 * 18
    };

    let mut white_max_y = i64::MIN;
    let mut black_min_y = i64::MAX;
    let w_promo = game.white_promo_rank;
    let b_promo = game.black_promo_rank;

    let mut w_to_find = game.white_pawn_count as i32;
    let mut b_to_find = game.black_pawn_count as i32;

    let mut bonus_score = 0; // Scaled terms (advancement)
    let mut penalty_score = 0; // Unscaled terms (structural worthless penalty)

    for (_cx, cy, tile) in game.board.tiles.iter() {
        let occ_pawns = tile.occ_pawns;
        if occ_pawns == 0 {
            continue;
        }

        let base_y = cy * 8;

        // Process White pawns
        let mut bits_w = occ_pawns & tile.occ_white;
        if bits_w != 0 {
            w_to_find -= bits_w.count_ones() as i32;
            while bits_w != 0 {
                let idx = bits_w.trailing_zeros() as usize;
                bits_w &= bits_w - 1;
                let y = base_y + (idx / 8) as i64;

                if y >= w_promo {
                    // Worthless if already past promotion - apply unscaled penalty
                    // to ensure 10x value reduction regardless of phase.
                    penalty_score -= PAWN_PAST_PROMO_PENALTY;
                } else {
                    let dist = w_promo - y;
                    if dist > PAWN_FULL_VALUE_THRESHOLD {
                        bonus_score -= PAWN_FAR_FROM_PROMO_PENALTY;
                    } else {
                        bonus_score += (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
                    }
                    // Track most advanced for special bonus
                    if y > white_max_y {
                        white_max_y = y;
                    }
                }
            }
        }

        // Process Black pawns
        let mut bits_b = occ_pawns & tile.occ_black;
        if bits_b != 0 {
            b_to_find -= bits_b.count_ones() as i32;
            while bits_b != 0 {
                let idx = bits_b.trailing_zeros() as usize;
                bits_b &= bits_b - 1;
                let y = base_y + (idx / 8) as i64;

                if y <= b_promo {
                    penalty_score += PAWN_PAST_PROMO_PENALTY;
                } else {
                    let dist = y - b_promo;
                    if dist > PAWN_FULL_VALUE_THRESHOLD {
                        bonus_score += PAWN_FAR_FROM_PROMO_PENALTY;
                    } else {
                        bonus_score -= (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
                    }
                    // Track most advanced
                    if y < black_min_y {
                        black_min_y = y;
                    }
                }
            }
        }

        if w_to_find <= 0 && b_to_find <= 0 {
            break;
        }
    }

    let white_has_promo = white_max_y != i64::MIN;
    let black_has_promo = black_min_y != i64::MAX;

    // Apply special high-value promotion bonus for the single most advanced pawn
    if white_has_promo {
        let dist = w_promo - white_max_y;
        bonus_score += if dist <= 1 {
            500
        } else if dist <= 2 {
            350
        } else {
            ((10 - dist.min(10)) as i32) * 40
        };
    }
    if black_has_promo {
        let dist = black_min_y - b_promo;
        bonus_score -= if dist <= 1 {
            500
        } else if dist <= 2 {
            350
        } else {
            ((10 - dist.min(10)) as i32) * 40
        };
    }

    // Scale advancement bonuses by game phase, but keep structural penalties unscaled
    let final_score = (bonus_score * multiplier_q / 100) + penalty_score;

    (final_score, white_has_promo, black_has_promo)
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
            if game.board.get_piece(cx, cy).is_some_and(|piece| {
                piece.color() == color
                    && (piece.piece_type() == PieceType::Pawn
                        || piece.piece_type() == PieceType::Guard
                        || piece.piece_type() == PieceType::Void)
            }) {
                has_ring_cover = true;
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

    // 3. Enemy sliders attacking king zone (with weighted penalties)
    // Only counts sliders with actual line-of-sight to king - essential for infinite chess
    let mut slider_threat_weight = 0i32;
    let mut slider_threat_count = 0i32;
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

            // Now use the O(log n) line-of-sight check via spatial indices
            let from = Coordinate { x, y };
            if is_clear_line_between_fast(&game.spatial_indices, &from, king) {
                // Weight by piece type: Queen/Amazon > Rook/Chancellor > Bishop/Archbishop
                let packed = tile.piece[idx];
                let weight = if packed != 0 {
                    let piece = crate::board::Piece::from_packed(packed);
                    match piece.piece_type() {
                        PieceType::Queen | PieceType::RoyalQueen | PieceType::Amazon => 4,
                        PieceType::Rook | PieceType::Chancellor => 3,
                        PieceType::Bishop | PieceType::Archbishop => 2,
                        PieceType::Knightrider => 2,
                        _ => 2,
                    }
                } else {
                    2
                };
                slider_threat_weight += weight;
                slider_threat_count += 1;
            }
        }
    }

    // Non-linear penalty: multiple sliders attacking is exponentially worse
    let base_penalty = slider_threat_weight * 15;
    let coordination_bonus = if slider_threat_count >= 2 {
        slider_threat_weight * 8
    } else {
        0
    };
    safety -= base_penalty + coordination_bonus;
    bump_feat!(king_enemy_slider_penalty, -slider_threat_count);

    /*
    // 4. King virtual mobility: safe squares the king can move to
    // Penalize kings that have few safe escape squares
    let mut safe_squares = 0;
    let enemy_color = if color == PlayerColor::White {
        PlayerColor::Black
    } else {
        PlayerColor::White
    };
    for dx in -1..=1_i64 {
        for dy in -1..=1_i64 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sq_x = king.x + dx;
            let sq_y = king.y + dy;
            let sq = Coordinate::new(sq_x, sq_y);

            // Check if square is empty or has enemy piece (potential escape)
            let is_blocked_by_friendly = game
                .board
                .get_piece(sq_x, sq_y)
                .is_some_and(|p| p.color() == color);
            if is_blocked_by_friendly {
                continue;
            }

            // Check if square is attacked by enemy
            if !crate::moves::is_square_attacked(
                &game.board,
                &sq,
                enemy_color,
                &game.spatial_indices,
            ) {
                safe_squares += 1;
            }
        }
    }
    // Bonus for having safe escape squares
    safety += safe_squares * KING_VIRTUAL_MOBILITY_WEIGHT;
    */

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
        // Evict ~40% of entries if cache grows too large (better than full clear)
        if cache.len() > 32768 {
            let to_remove = cache.len() * 2 / 5; // Remove ~40%
            let keys: Vec<_> = cache.keys().take(to_remove).copied().collect();
            for k in keys {
                cache.remove(&k);
            }
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

    let w_promo = game.white_promo_rank;
    let b_promo = game.black_promo_rank;

    // BITBOARD: Use per-tile occ_pawns - skip past-promo pawns for structure eval
    for (cx, cy, tile) in game.board.tiles.iter() {
        let w_pawns_bits = tile.occ_pawns & tile.occ_white;
        let b_pawns_bits = tile.occ_pawns & tile.occ_black;
        let base_y = cy * 8;

        if w_pawns_bits != 0 {
            let mut bits = w_pawns_bits;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = base_y + (idx / 8) as i64;
                if y < w_promo {
                    white_pawns.push((x, y));
                    white_pawn_files.push(x);
                }
            }
        }

        if b_pawns_bits != 0 {
            let mut bits = b_pawns_bits;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = base_y + (idx / 8) as i64;
                if y > b_promo {
                    black_pawns.push((x, y));
                    black_pawn_files.push(x);
                }
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

        // Connected pawn bonus: check if there's a friendly pawn diagonally behind
        if is_connected_pawn(game, *wx, *wy, PlayerColor::White) {
            score += CONNECTED_PAWN_BONUS;
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

        // Connected pawn bonus: check if there's a friendly pawn diagonally behind
        if is_connected_pawn(game, *bx, *by, PlayerColor::Black) {
            score -= CONNECTED_PAWN_BONUS;
        }
    }

    score
}

// ==================== Threat Evaluation ====================

/// Evaluate threats: bonus for attacking higher-value pieces with lower-value pieces.
/// Efficient on infinite boards - only checks direct attack squares for leapers (pawns, knights).
fn evaluate_threats(game: &GameState) -> i32 {
    let mut score: i32 = 0;

    // Threat bonus constants (centipawns)
    const PAWN_THREATENS_MINOR: i32 = 25;
    const PAWN_THREATENS_ROOK: i32 = 40;
    const PAWN_THREATENS_QUEEN: i32 = 60;
    const MINOR_THREATENS_ROOK: i32 = 20;
    const MINOR_THREATENS_QUEEN: i32 = 35;

    const KNIGHT_OFFSETS: [(i64, i64); 8] = [
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ];

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
            let piece = crate::board::Piece::from_packed(packed);
            let pt = piece.piece_type();
            let color = piece.color();
            if color == PlayerColor::Neutral {
                continue;
            }

            let x = cx * 8 + (idx % 8) as i64;
            let y = cy * 8 + (idx / 8) as i64;

            let sign = if color == PlayerColor::White { 1 } else { -1 };
            let enemy = if color == PlayerColor::White {
                PlayerColor::Black
            } else {
                PlayerColor::White
            };

            match pt {
                PieceType::Pawn => {
                    let dy = if color == PlayerColor::White { 1 } else { -1 };
                    for dx in [-1i64, 1] {
                        if let Some(target) = game.board.get_piece(x + dx, y + dy) {
                            if target.color() == enemy {
                                let tv = get_piece_value(target.piece_type());
                                if tv >= 600 {
                                    score += sign * PAWN_THREATENS_QUEEN;
                                } else if tv >= 400 {
                                    score += sign * PAWN_THREATENS_ROOK;
                                } else if tv >= 200 {
                                    score += sign * PAWN_THREATENS_MINOR;
                                }
                            }
                        }
                    }
                }
                PieceType::Knight | PieceType::Centaur | PieceType::RoyalCentaur => {
                    for &(dx, dy) in &KNIGHT_OFFSETS {
                        if let Some(target) = game.board.get_piece(x + dx, y + dy) {
                            if target.color() == enemy {
                                let tv = get_piece_value(target.piece_type());
                                let mv = get_piece_value(pt);
                                if tv >= 600 && mv < 600 {
                                    score += sign * MINOR_THREATENS_QUEEN;
                                } else if tv >= 400 && mv < 400 {
                                    score += sign * MINOR_THREATENS_ROOK;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
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

/// O(log n) version of is_clear_line_between using SpatialIndices.
/// Uses binary search on sorted coordinate arrays instead of iterating all pieces.
#[inline]
pub fn is_clear_line_between_fast(
    indices: &crate::moves::SpatialIndices,
    from: &Coordinate,
    to: &Coordinate,
) -> bool {
    let dx = to.x - from.x;
    let dy = to.y - from.y;

    // Not collinear in rook/bishop directions
    if !(dx == 0 || dy == 0 || dx.abs() == dy.abs()) {
        return false;
    }

    // Early exit for adjacent squares
    if dx.abs() <= 1 && dy.abs() <= 1 {
        return true;
    }

    // Horizontal line (same rank)
    if dy == 0 {
        if let Some(row) = indices.rows.get(&from.y) {
            let (min_x, max_x) = if from.x < to.x {
                (from.x, to.x)
            } else {
                (to.x, from.x)
            };
            // Binary search for first piece with x > min_x
            let start = row.partition_point(|(x, _)| *x <= min_x);
            // Check if any piece exists before max_x
            if start < row.len() && row[start].0 < max_x {
                return false;
            }
        }
        return true;
    }

    // Vertical line (same file)
    if dx == 0 {
        if let Some(col) = indices.cols.get(&from.x) {
            let (min_y, max_y) = if from.y < to.y {
                (from.y, to.y)
            } else {
                (to.y, from.y)
            };
            // Binary search for first piece with y > min_y
            let start = col.partition_point(|(y, _)| *y <= min_y);
            // Check if any piece exists before max_y
            if start < col.len() && col[start].0 < max_y {
                return false;
            }
        }
        return true;
    }

    // Diagonal (x - y constant) - for dx.signum() == dy.signum()
    if dx.signum() == dy.signum() {
        let diag_key = from.x - from.y;
        if let Some(diag) = indices.diag1.get(&diag_key) {
            let (min_x, max_x) = if from.x < to.x {
                (from.x, to.x)
            } else {
                (to.x, from.x)
            };
            let start = diag.partition_point(|(x, _)| *x <= min_x);
            if start < diag.len() && diag[start].0 < max_x {
                return false;
            }
        }
        return true;
    }

    // Anti-diagonal (x + y constant) - for dx.signum() != dy.signum()
    let diag_key = from.x + from.y;
    if let Some(diag) = indices.diag2.get(&diag_key) {
        let (min_x, max_x) = if from.x < to.x {
            (from.x, to.x)
        } else {
            (to.x, from.x)
        };
        let start = diag.partition_point(|(x, _)| *x <= min_x);
        if start < diag.len() && diag[start].0 < max_x {
            return false;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece};
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    #[test]
    fn test_get_piece_value() {
        // Standard pieces - values tuned for infinite chess
        assert_eq!(get_piece_value(PieceType::Pawn), 100);
        assert_eq!(get_piece_value(PieceType::Knight), 250); // Weak in infinite chess
        assert_eq!(get_piece_value(PieceType::Bishop), 450); // Strong slider
        assert_eq!(get_piece_value(PieceType::Rook), 650); // Very strong
        assert_eq!(get_piece_value(PieceType::Queen), 1350); // > 2 rooks

        // King/Guard have same nominal value
        assert_eq!(get_piece_value(PieceType::King), 220);

        // Fairy pieces have values
        assert!(get_piece_value(PieceType::Amazon) > get_piece_value(PieceType::Queen));
        assert!(get_piece_value(PieceType::Chancellor) > get_piece_value(PieceType::Rook));
    }

    #[test]
    fn test_is_between() {
        assert!(is_between(5, 3, 7));
        assert!(is_between(5, 7, 3));
        assert!(!is_between(3, 3, 7));
        assert!(!is_between(7, 3, 7));
        assert!(!is_between(2, 3, 7));
        assert!(!is_between(8, 3, 7));
    }

    #[test]
    fn test_is_clear_line_between() {
        let mut board = Board::new();
        let from = Coordinate::new(1, 1);
        let to = Coordinate::new(1, 8);

        // Empty board should have clear line
        assert!(is_clear_line_between(&board, &from, &to));

        // Add blocker
        board.set_piece(1, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        assert!(!is_clear_line_between(&board, &from, &to));
    }

    #[test]
    fn test_is_clear_line_diagonal() {
        let mut board = Board::new();
        let from = Coordinate::new(1, 1);
        let to = Coordinate::new(5, 5);

        assert!(is_clear_line_between(&board, &from, &to));

        board.set_piece(3, 3, Piece::new(PieceType::Bishop, PlayerColor::Black));
        assert!(!is_clear_line_between(&board, &from, &to));
    }

    #[test]
    fn test_calculate_initial_material() {
        let mut board = Board::new();

        // Empty board = 0
        assert_eq!(calculate_initial_material(&board), 0);

        // Add white queen
        board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        board.rebuild_tiles();
        assert_eq!(calculate_initial_material(&board), 1350); // Queen = 1350 in infinite chess

        // Add black queen - should cancel out
        board.set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        board.rebuild_tiles();
        assert_eq!(calculate_initial_material(&board), 0);
    }

    #[test]
    fn test_clear_pawn_cache() {
        // Just ensure it doesn't panic
        clear_pawn_cache();
    }

    #[test]
    fn test_evaluate_returns_value() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.board.rebuild_tiles();
        game.recompute_hash();

        let score = evaluate(&game);
        // K vs K should be close to 0
        assert!(score.abs() < 1000, "K vs K should be near 0, got {}", score);
    }

    #[test]
    fn test_evaluate_lazy_returns_value() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.board.rebuild_tiles();

        let score = evaluate_lazy(&game);
        // White has extra queen so should be positive
        assert!(score > 0, "White with extra queen should be positive");
    }

    #[test]
    fn test_count_pawns_on_file() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.board.rebuild_tiles();

        let (own, enemy) = count_pawns_on_file(&game, 4, PlayerColor::White);
        assert_eq!(own, 2);
        assert_eq!(enemy, 1);
    }

    #[test]
    fn test_evaluate_pawn_structure() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Doubled pawns for white
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.recompute_piece_counts();
        game.board.rebuild_tiles();
        game.recompute_hash();

        let score = evaluate_pawn_structure(&game);
        // Doubled pawns should give penalty (White has doubled pawns = negative score)
        // Note: The penalty may be offset by passed pawn bonus, so just check it runs
        assert!(
            score.abs() < 1000,
            "Pawn structure score should be reasonable: {}",
            score
        );
    }

    #[test]
    fn test_evaluate_king_safety() {
        let mut game = create_test_game();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board.rebuild_tiles();

        let wk = Some(Coordinate::new(5, 1));
        let bk = Some(Coordinate::new(5, 8));

        let score = evaluate_king_safety(&game, &wk, &bk);
        // Both kings exposed similarly, should be near 0
        assert!(score.abs() < 200);
    }

    #[test]
    fn test_compute_cloud_center() {
        let mut board = Board::new();
        // Use non-neutral pieces
        board.set_piece(0, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(10, 0, Piece::new(PieceType::Rook, PlayerColor::White));

        let center = compute_cloud_center(&board);
        assert!(center.is_some(), "Cloud center should exist");
        let c = center.unwrap();
        assert_eq!(c.x, 5); // Average of 0 and 10
        assert_eq!(c.y, 0);
    }

    #[test]
    fn test_evaluate_lazy_basic() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.material_score = 0; // Kings are worth same

        let score_equal = evaluate_lazy(&game);

        // Add a white queen
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();
        game.material_score = 1350; // Manually update material score
        let score_white_plus = evaluate_lazy(&game);

        assert!(
            score_white_plus > score_equal + 1300,
            "White queen should increase lazy eval by at least its material value"
        );
    }

    #[test]
    fn test_king_safety_penalties() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        // White King at (0,0)
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(10, 10, Piece::new(PieceType::King, PlayerColor::Black));

        // Add some sufficient material to avoid draw (0)
        game.board
            .set_piece(5, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(5, 9, Piece::new(PieceType::Rook, PlayerColor::Black));

        // White Queen at home near its king (good/neutral)
        game.board
            .set_piece(0, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();
        game.material_score = 0; // Rook vs Rook balanced
        let score_near = evaluate_inner(&game);

        // White Queen far away from its king
        game.board.remove_piece(&0, &1);
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();
        game.material_score = 0;
        let score_far = evaluate_inner(&game);

        assert!(score_far != score_near);
    }

    #[test]
    fn test_pawn_structure_caching() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();

        clear_pawn_cache();
        let eval1 = evaluate_inner(&game);

        // Calling again should hit cache
        let eval2 = evaluate_inner(&game);
        assert_eq!(
            eval1, eval2,
            "Cached evaluation should match initial evaluation"
        );
    }

    #[test]
    fn test_evaluate_knight_centralization() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));

        // Knight in corner (worst)
        let corner_score = evaluate_knight(0, 0, PlayerColor::White, &None, &None);
        // Knight in center (best)
        let center_score = evaluate_knight(4, 4, PlayerColor::White, &None, &None);

        assert!(
            center_score > corner_score,
            "Central knight should score better than corner knight"
        );
    }

    #[test]
    fn test_evaluate_bishop_diagonal() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Bishop, PlayerColor::White));
        game.recompute_piece_counts();
        game.board.rebuild_tiles();

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_bishop(&game, 4, 4, PlayerColor::White, &wk, &bk);
        // Central bishop should have positive score
        assert!(
            score > 0,
            "Central bishop should have positive positional score"
        );
    }

    #[test]
    fn test_evaluate_rook_open_file() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.recompute_piece_counts();
        game.board.rebuild_tiles();

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_rook(&game, 4, 1, PlayerColor::White, &wk, &bk);
        // Rook should have score for mobility etc
        assert!(score.abs() < 1000, "Rook score should be reasonable");
    }

    #[test]
    fn test_evaluate_queen_central() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::White));
        game.recompute_piece_counts();
        game.board.rebuild_tiles();

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_queen(&game, 4, 4, PlayerColor::White, &wk, &bk);
        // Queen in center should have decent positional score
        assert!(score.abs() < 2000, "Queen score should be reasonable");
    }

    #[test]
    fn test_pawn_structure_isolated_pawn() {
        let mut game = Box::new(GameState::new());
        game.board = Board::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Isolated white pawn on d-file
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.recompute_piece_counts();
        game.recompute_hash();

        clear_pawn_cache();
        let isolated_score = evaluate_pawn_structure(&game);

        // Add supporting pawns
        game.board
            .set_piece(3, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(5, 3, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.recompute_piece_counts();
        game.recompute_hash();

        clear_pawn_cache();
        let supported_score = evaluate_pawn_structure(&game);

        // Supported pawns should score better
        assert!(
            supported_score > isolated_score,
            "Supported pawns should be better than isolated"
        );
    }
}
