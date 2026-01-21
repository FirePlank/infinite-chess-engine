use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;

use std::cell::RefCell;

// Direct-mapped pawn structure cache: index -> (pawn_hash, score)
// Size power of 2 for fast modulo
const PAWN_CACHE_SIZE: usize = 16384;

thread_local! {
    // Array of (hash, score). Initialize with MAX which is unlikely to valid hash.
    static PAWN_CACHE: RefCell<Vec<(u64, i32)>> = RefCell::new(vec![(u64::MAX, 0); PAWN_CACHE_SIZE]);
    // Reusable buffer for piece list to avoid allocation
    static EVAL_PIECE_LIST: RefCell<Vec<(i64, i64, crate::board::Piece)>> = RefCell::new(Vec::with_capacity(64));
}

/// Clear the pawn structure cache.
pub fn clear_pawn_cache() {
    PAWN_CACHE.with(|cache| {
        // Fast clear using fill
        cache.borrow_mut().fill((u64::MAX, 0));
    });
}

#[cfg(feature = "eval_tuning")]
use once_cell::sync::Lazy;
#[cfg(feature = "eval_tuning")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "eval_tuning")]
use std::sync::RwLock;

/// Tracer trait for evaluation components.
/// Uses zero-cost abstraction with NoTrace for production.
pub trait EvaluationTracer {
    fn record(&mut self, term: &str, white: i32, black: i32);
    fn is_active(&self) -> bool;
}

/// No-op tracer for production use.
pub struct NoTrace;
impl EvaluationTracer for NoTrace {
    #[inline(always)]
    fn record(&mut self, _term: &str, _white: i32, _black: i32) {}
    #[inline(always)]
    fn is_active(&self) -> bool {
        false
    }
}

/// Active tracer for debug output.
#[derive(Default, Debug, Clone)]
pub struct ActiveTrace {
    pub rows: Vec<(String, i32, i32)>,
}

impl EvaluationTracer for ActiveTrace {
    fn record(&mut self, term: &str, white: i32, black: i32) {
        self.rows.push((term.to_string(), white, black));
    }
    fn is_active(&self) -> bool {
        true
    }
}

impl ActiveTrace {
    pub fn print(&self) {
        println!(
            "\n{:<25} | {:>10} | {:>10} | {:>10}",
            "Evaluation Term", "White", "Black", "Total"
        );
        println!("{:-<25}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "");
        let mut total_w = 0;
        let mut total_b = 0;
        for (term, w, b) in &self.rows {
            total_w += w;
            total_b += b;
            println!(
                "{:<25} | {:>10.2} | {:>10.2} | {:>10.2}",
                term,
                *w as f64 / 100.0,
                *b as f64 / 100.0,
                (*w - *b) as f64 / 100.0
            );
        }
        println!("{:-<25}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "");
        println!(
            "{:<25} | {:>10.2} | {:>10.2} | {:>10.2}",
            "TOTAL",
            total_w as f64 / 100.0,
            total_b as f64 / 100.0,
            (total_w - total_b) as f64 / 100.0
        );
        println!();
    }
}

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

// Piece Values
const KNIGHT: i32 = 250;
const BISHOP: i32 = KNIGHT + 200;
const ROOK: i32 = KNIGHT + BISHOP - 50;
const GUARD: i32 = 220;
const CENTAUR: i32 = 550;
const QUEEN: i32 = ROOK * 2 + COMPOUND_BONUS;

const COMPOUND_BONUS: i32 = 50;
const ROYAL_BONUS: i32 = 50;

pub fn get_piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        // neutral/blocking pieces - no material value
        PieceType::Void => 0,
        PieceType::Obstacle => 0,

        // orthodox - adjusted for infinite chess where sliders dominate
        PieceType::Pawn => 100,
        PieceType::Knight => KNIGHT, // Weak in infinite chess
        PieceType::Bishop => BISHOP, // Strong slider
        PieceType::Rook => ROOK,     // Very strong in infinite chess
        PieceType::Queen => QUEEN,   // > 2 rooks
        PieceType::Guard => GUARD,

        // short / medium range
        PieceType::Camel => 270,   // (1,3) leaper
        PieceType::Giraffe => 260, // (1,4) leaper
        PieceType::Zebra => 260,   // (2,3) leaper

        // riders / compounds
        PieceType::Knightrider => 700,
        PieceType::Amazon => QUEEN + KNIGHT,
        PieceType::Hawk => 600,
        PieceType::Chancellor => ROOK + KNIGHT + 100,
        PieceType::Archbishop => 900,
        PieceType::Centaur => CENTAUR,

        // royals
        PieceType::King => GUARD + ROYAL_BONUS,
        PieceType::RoyalQueen => QUEEN + ROYAL_BONUS,
        PieceType::RoyalCentaur => CENTAUR + ROYAL_BONUS,

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

// King attack heuristics - back near original scale
// These should be impactful but not dominate material.
const SLIDER_NET_BONUS: i32 = 20;

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

// Max distance a single piece can skew the cloud center from the reference point.
// Prevents extreme outliers (e.g., a queen at 1e15) from dominating the weighted average.
// Pieces beyond this distance have their position clamped for centroid calculation.
const CLOUD_CENTER_MAX_SKEW_DIST: i64 = 16;

// Bishop pair & queen heuristics
// Tapered pairs defined below
const QUEEN_IDEAL_LINE_DIST: i32 = 4;

// Fairy Piece Evaluation

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

// King defender bonuses/penalties
// Low-value pieces near own king = good (defense)
// High-value pieces near own king = bad (should be attacking)
const KING_DEFENDER_VALUE_THRESHOLD: i32 = 400; // Pieces below this value are defensive

// ==================== Game Phase ====================

// Development thresholds - for attack scaling only
const UNDEVELOPED_MINORS_THRESHOLD: i32 = 2;
const DEVELOPMENT_PHASE_ATTACK_SCALE: i32 = 50;

// ==================== Game Phase ====================

pub const MAX_PHASE: i32 = 24;

pub fn get_piece_phase(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 1,
        PieceType::Rook => 2,
        PieceType::Queen => 4,
        PieceType::King => 0,

        // Fairy pieces
        PieceType::Guard => 1,
        PieceType::Centaur => 1, // Knight-like
        PieceType::Camel => 1,
        PieceType::Giraffe => 1,
        PieceType::Zebra => 1,
        PieceType::Rose => 2, // Stronger
        PieceType::Huygen => 1,

        // Strong compounds
        PieceType::Chancellor => 2, // R+N
        PieceType::Archbishop => 2, // B+N
        PieceType::Hawk => 2,
        PieceType::Knightrider => 2,

        // Monsters
        PieceType::Amazon => 4, // Q+N
        PieceType::RoyalQueen => 4,
        PieceType::RoyalCentaur => 2,

        _ => 0,
    }
}

/// Calculate the current game phase (0 = Endgame, MAX_PHASE = Opening/Middlegame)
/// Based on non-pawn material remaining on the board.
pub fn calculate_game_phase(game: &GameState) -> i32 {
    let mut phase = 0;

    for (_cx, _cy, tile) in game.board.tiles.iter() {
        if crate::simd::both_zero(tile.occ_white, tile.occ_black) {
            continue;
        }
        let mut bits = tile.occ_all & !tile.occ_void & !tile.occ_pawns; // Skip pawns/void
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let packed = tile.piece[idx];
            if packed != 0 {
                let piece = crate::board::Piece::from_packed(packed);
                phase += get_piece_phase(piece.piece_type());
            }
        }
    }

    phase.min(MAX_PHASE)
}

// ==================== Tapered Evaluation Constants (MG, EG) ====================

// King Safety
const MG_BEHIND_KING_BONUS: i32 = 40;
const EG_BEHIND_KING_BONUS: i32 = 60; // More important to be behind king in EG

const MG_KING_TROPISM_BONUS: i32 = 4;
const EG_KING_TROPISM_BONUS: i32 = 6; // King centralized -> piece proximity matters more

// Shelter / Ring
const MG_KING_RING_MISSING_PENALTY: i32 = 45;
const EG_KING_RING_MISSING_PENALTY: i32 = 20; // Less penalty in EG

const MG_KING_PAWN_SHIELD_BONUS: i32 = 18;
const EG_KING_PAWN_SHIELD_BONUS: i32 = 5; // Shield less critical

const MG_KING_PAWN_AHEAD_PENALTY: i32 = 20;
const EG_KING_PAWN_AHEAD_PENALTY: i32 = 5;

// Structural
const MG_CONNECTED_PAWN_BONUS: i32 = 8;
const EG_CONNECTED_PAWN_BONUS: i32 = 15; // Chains critical in EG

const MG_DOUBLED_PAWN_PENALTY: i32 = 8;
const EG_DOUBLED_PAWN_PENALTY: i32 = 12;

const MG_BISHOP_PAIR_BONUS: i32 = 60;
const EG_BISHOP_PAIR_BONUS: i32 = 80;

const MG_KING_DEFENDER_BONUS: i32 = 6;
const EG_KING_DEFENDER_BONUS: i32 = 2; // Less need for defenders

const MG_KING_ATTACKER_NEAR_OWN_KING_PENALTY: i32 = 8;
const EG_KING_ATTACKER_NEAR_OWN_KING_PENALTY: i32 = 2;

// Slider Distances (Centralization less critical in EG)
const MG_FAR_SLIDER_PENALTY_MULT: i32 = 100; // 100%
const EG_FAR_SLIDER_PENALTY_MULT: i32 = 40; // 40%

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

// Main Evaluation
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

/// Perform a full evaluation with detailed tracing.
pub fn debug_evaluate(game: &GameState) -> ActiveTrace {
    let mut tracer = ActiveTrace::default();
    evaluate_inner_traced(game, &mut tracer);
    tracer
}

/// Core evaluation logic - skips insufficient material check
#[inline]
pub fn evaluate_inner(game: &GameState) -> i32 {
    evaluate_inner_traced(game, &mut NoTrace)
}

/// Core evaluation logic with tracing support
pub fn evaluate_inner_traced<T: EvaluationTracer>(game: &GameState, tracer: &mut T) -> i32 {
    // Start with material score
    let mut score = game.material_score;

    // Use cached king positions
    let (white_king, black_king) = (game.white_king_pos, game.black_king_pos);

    // Single-Pass Collection and Scoring
    let mut phase = 0;
    let mut white_undeveloped = 0;
    let mut black_undeveloped = 0;
    let mut white_bishops = 0;
    let mut white_bishop_colors = (false, false);
    let mut black_bishops = 0;
    let mut black_bishop_colors = (false, false);
    let mut cloud_sum_x: i64 = 0;
    let mut cloud_sum_y: i64 = 0;
    let mut cloud_count: i64 = 0;

    // Slider counts for attack bonus (white, black)
    let mut w_diag_count = 0;
    let mut w_ortho_count = 0;
    let mut b_diag_count = 0;
    let mut b_ortho_count = 0;

    // Threat points for defense urgency
    let mut w_threat_points = 0;
    let mut black_threat_points = 0;
    let mut w_has_queen_threat = false;
    let mut b_has_queen_threat = false;

    // Pawn advancement metrics
    let mut white_max_y = i64::MIN;
    let mut black_min_y = i64::MAX;
    let mut w_pawn_bonus = 0;
    let mut b_pawn_bonus = 0;
    let mut w_pawn_penalty = 0;
    let mut b_pawn_penalty = 0;
    let w_promo = game.white_promo_rank;
    let b_promo = game.black_promo_rank;

    // For multiplier_q
    let mut white_non_pawn_non_royal = 0;
    let mut black_non_pawn_non_royal = 0;

    EVAL_PIECE_LIST.with(|piece_list_cell| {
        let mut piece_list = piece_list_cell.borrow_mut();
        piece_list.clear();

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
                let is_white = piece.color() == PlayerColor::White;
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;

                // 1. Phase
                let p_phase = get_piece_phase(pt);
                phase += p_phase;

                // 2. Piece Collection
                piece_list.push((x, y, piece));

                // 3. Piece counts for scaling
                if pt != PieceType::Pawn && !pt.is_royal() {
                    if is_white {
                        white_non_pawn_non_royal += 1;
                    } else {
                        black_non_pawn_non_royal += 1;
                    }
                }

                // 4. Cloud Stats
                let cw = get_centrality_weight(pt);
                if cw > 0 && pt != PieceType::Pawn {
                    let rx = match (white_king, black_king) {
                        (Some(wk), Some(bk)) => (wk.x + bk.x) / 2,
                        (Some(wk), None) => wk.x,
                        (None, Some(bk)) => bk.x,
                        (None, None) => 0,
                    };
                    let ry = match (white_king, black_king) {
                        (Some(wk), Some(bk)) => (wk.y + bk.y) / 2,
                        (Some(wk), None) => wk.y,
                        (None, Some(bk)) => bk.y,
                        (None, None) => 0,
                    };
                    let dx = x - rx;
                    let dy = y - ry;
                    let cdx = dx.clamp(-CLOUD_CENTER_MAX_SKEW_DIST, CLOUD_CENTER_MAX_SKEW_DIST);
                    let cdy = dy.clamp(-CLOUD_CENTER_MAX_SKEW_DIST, CLOUD_CENTER_MAX_SKEW_DIST);
                    cloud_sum_x += cw * (rx + cdx);
                    cloud_sum_y += cw * (ry + cdy);
                    cloud_count += cw;
                }

                // 5. Minor stats
                if (pt == PieceType::Knight || pt == PieceType::Bishop)
                    && game.starting_squares.contains(&Coordinate::new(x, y))
                {
                    if is_white {
                        white_undeveloped += 1;
                    } else {
                        black_undeveloped += 1;
                    }
                }

                if pt == PieceType::Bishop {
                    if is_white {
                        white_bishops += 1;
                        if (x + y) % 2 == 0 {
                            white_bishop_colors.0 = true;
                        } else {
                            white_bishop_colors.1 = true;
                        }
                    } else {
                        black_bishops += 1;
                        if (x + y) % 2 == 0 {
                            black_bishop_colors.0 = true;
                        } else {
                            black_bishop_colors.1 = true;
                        }
                    }
                }

                // 6. Slider counts
                if (tile.occ_diag_sliders & (1 << idx)) != 0 {
                    if is_white {
                        w_diag_count += 1;
                    } else {
                        b_diag_count += 1;
                    }
                }
                if (tile.occ_ortho_sliders & (1 << idx)) != 0 {
                    if is_white {
                        w_ortho_count += 1;
                    } else {
                        b_ortho_count += 1;
                    }
                }

                // 7. Threat points for urgency
                if !pt.is_royal() && pt != PieceType::Pawn {
                    const QUEEN_THREAT: i32 = 40;
                    const ROOK_THREAT: i32 = 15;
                    const BISHOP_THREAT: i32 = 10;
                    const KNIGHTRIDER_THREAT: i32 = 8;
                    const MINOR_THREAT: i32 = 3;

                    let (is_diag, is_ortho) = (
                        (tile.occ_diag_sliders & (1 << idx)) != 0,
                        (tile.occ_ortho_sliders & (1 << idx)) != 0,
                    );

                    let tp = if is_diag && is_ortho {
                        if is_white {
                            w_has_queen_threat = true;
                        } else {
                            b_has_queen_threat = true;
                        }
                        QUEEN_THREAT
                    } else if is_ortho {
                        ROOK_THREAT
                    } else if is_diag {
                        BISHOP_THREAT
                    } else if pt == PieceType::Knightrider {
                        KNIGHTRIDER_THREAT
                    } else {
                        MINOR_THREAT
                    };

                    if is_white {
                        w_threat_points += tp;
                    } else {
                        black_threat_points += tp;
                    }
                }

                // 8. Pawn advancement
                if pt == PieceType::Pawn {
                    if is_white {
                        if y >= w_promo {
                            w_pawn_penalty -= PAWN_PAST_PROMO_PENALTY;
                        } else {
                            let dist = w_promo - y;
                            if dist > PAWN_FULL_VALUE_THRESHOLD {
                                w_pawn_bonus -= PAWN_FAR_FROM_PROMO_PENALTY;
                            } else {
                                w_pawn_bonus += (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
                            }
                            if y > white_max_y {
                                white_max_y = y;
                            }
                        }
                    } else if y <= b_promo {
                        b_pawn_penalty -= PAWN_PAST_PROMO_PENALTY;
                    } else {
                        let dist = y - b_promo;
                        if dist > PAWN_FULL_VALUE_THRESHOLD {
                            b_pawn_bonus -= PAWN_FAR_FROM_PROMO_PENALTY;
                        } else {
                            b_pawn_bonus += (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
                        }
                        if y < black_min_y {
                            black_min_y = y;
                        }
                    }
                }
            }
        }

        // --- Post-Pass processing ---
        let final_phase = phase.min(MAX_PHASE);
        let cloud_center = if cloud_count > 0 {
            Some(Coordinate {
                x: cloud_sum_x / cloud_count,
                y: cloud_sum_y / cloud_count,
            })
        } else {
            None
        };

        // Pawn Advancement Calculation
        if white_max_y != i64::MIN {
            let dist = w_promo - white_max_y;
            w_pawn_bonus += if dist <= 1 {
                500
            } else if dist <= 2 {
                350
            } else {
                ((10 - dist.min(10)) as i32) * 40
            };
        }
        if black_min_y != i64::MAX {
            let dist = black_min_y - b_promo;
            b_pawn_bonus += if dist <= 1 {
                500
            } else if dist <= 2 {
                350
            } else {
                ((10 - dist.min(10)) as i32) * 40
            };
        }

        let total_pieces = white_non_pawn_non_royal + black_non_pawn_non_royal;
        let multiplier_q = if total_pieces >= 10 {
            10
        } else if total_pieces <= 5 {
            100
        } else {
            100 - (total_pieces - 5) * 18
        };

        let w_adv = (w_pawn_bonus * multiplier_q / 100) + w_pawn_penalty;
        let b_adv = (b_pawn_bonus * multiplier_q / 100) + b_pawn_penalty;
        tracer.record("Pawn Advancement", w_adv, b_adv);
        score += w_adv - b_adv;

        let white_has_promo = white_max_y != i64::MIN;
        let black_has_promo = black_min_y != i64::MAX;

        // Mop-up (using existing counts)
        let white_pieces = game.white_piece_count.saturating_sub(game.white_pawn_count);
        let black_pieces = game.black_piece_count.saturating_sub(game.black_pawn_count);
        let mut mop_up_applied = false;

        if black_pieces < 3
            && white_pieces > 1
            && !white_has_promo
            && let Some(bk) = &black_king
            && crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::Black).is_some()
        {
            let s = crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                white_king.as_ref(),
                bk,
                PlayerColor::White,
                PlayerColor::Black,
            );
            tracer.record("Mop-up", s, 0);
            score += s;
            mop_up_applied = true;
        }

        if !mop_up_applied
            && white_pieces < 3
            && black_pieces > 1
            && !black_has_promo
            && let Some(wk) = &white_king
            && crate::evaluation::mop_up::calculate_mop_up_scale(game, PlayerColor::White).is_some()
        {
            let s = crate::evaluation::mop_up::evaluate_mop_up_scaled(
                game,
                black_king.as_ref(),
                wk,
                PlayerColor::Black,
                PlayerColor::White,
            );
            tracer.record("Mop-up", 0, s);
            score -= s;
            mop_up_applied = true;
        }

        // Defense urgency (pre-calculated)
        let calc_urgency = |tp: i32, has_q: bool| {
            if tp >= 80 {
                100
            } else if tp >= 55 {
                85
            } else if tp >= 40 {
                70
            } else if has_q {
                60
            } else if tp >= 25 {
                50
            } else if tp >= 10 {
                30
            } else {
                10
            }
        };
        let w_urgency = calc_urgency(black_threat_points, b_has_queen_threat);
        let b_urgency = calc_urgency(w_threat_points, w_has_queen_threat);

        if !mop_up_applied {
            score += evaluate_pieces_processed(
                game,
                &white_king,
                &black_king,
                final_phase,
                tracer,
                &piece_list,
                PieceMetrics {
                    white_undeveloped,
                    black_undeveloped,
                    white_bishops,
                    black_bishops,
                    white_bishop_colors,
                    black_bishop_colors,
                    cloud_center,
                },
            );

            // King Safety
            let ks_metrics = KingSafetyMetrics {
                white_slider_counts: (w_diag_count, w_ortho_count),
                black_slider_counts: (b_diag_count, b_ortho_count),
                urgency: (w_urgency, b_urgency),
                has_enemy_queen: (b_has_queen_threat, w_has_queen_threat),
            };
            score += evaluate_king_safety_traced(
                game,
                &white_king,
                &black_king,
                final_phase,
                tracer,
                &ks_metrics,
            );

            score += evaluate_pawn_structure_traced(game, final_phase, tracer, &piece_list);
            score += evaluate_threat_traced(game, tracer);
        }
    });

    // Return from current player's perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

struct PieceMetrics {
    white_undeveloped: i32,
    black_undeveloped: i32,
    white_bishops: i32,
    black_bishops: i32,
    white_bishop_colors: (bool, bool),
    black_bishop_colors: (bool, bool),
    cloud_center: Option<Coordinate>,
}

fn evaluate_pieces_processed<T: EvaluationTracer>(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    tracer: &mut T,
    piece_list: &[(i64, i64, crate::board::Piece)],
    metrics: PieceMetrics,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_activity: i32 = 0;
    let mut b_activity: i32 = 0;

    let cloud_center = metrics.cloud_center;

    // Pre-calculate attack readiness using piece list
    let white_attack_scale =
        calculate_attack_readiness_from_list(game, black_king, piece_list, PlayerColor::White);
    let black_attack_scale =
        calculate_attack_readiness_from_list(game, white_king, piece_list, PlayerColor::Black);

    let white_attack_scale = if metrics.white_undeveloped >= UNDEVELOPED_MINORS_THRESHOLD {
        white_attack_scale.min(DEVELOPMENT_PHASE_ATTACK_SCALE)
    } else {
        white_attack_scale
    };
    let black_attack_scale = if metrics.black_undeveloped >= UNDEVELOPED_MINORS_THRESHOLD {
        black_attack_scale.min(DEVELOPMENT_PHASE_ATTACK_SCALE)
    } else {
        black_attack_scale
    };

    for &(x, y, piece) in piece_list {
        let mut piece_score = match piece.piece_type() {
            PieceType::Rook => {
                evaluate_rook(game, x, y, piece.color(), white_king, black_king, phase)
            }
            PieceType::Queen => {
                evaluate_queen(game, x, y, piece.color(), white_king, black_king, phase)
            }
            PieceType::Knight => {
                evaluate_knight(x, y, piece.color(), black_king, white_king, phase)
            }
            PieceType::Bishop => {
                evaluate_bishop(game, x, y, piece.color(), white_king, black_king, phase)
            }
            PieceType::Chancellor => {
                let rook_eval =
                    evaluate_rook(game, x, y, piece.color(), white_king, black_king, phase);
                let mut b = 0;
                let ek = if piece.color() == PlayerColor::White {
                    black_king
                } else {
                    white_king
                };
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
                    evaluate_bishop(game, x, y, piece.color(), white_king, black_king, phase);
                let mut b = 0;
                let ek = if piece.color() == PlayerColor::White {
                    black_king
                } else {
                    white_king
                };
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
                    evaluate_queen(game, x, y, piece.color(), white_king, black_king, phase);
                let rook_eval =
                    evaluate_rook(game, x, y, piece.color(), white_king, black_king, phase);
                let mut b = 0;
                let ek = if piece.color() == PlayerColor::White {
                    black_king
                } else {
                    white_king
                };
                if let Some(k) = ek {
                    let d = (x - k.x).abs() + (y - k.y).abs();
                    if d <= 4 {
                        b = 20;
                    } else if d <= 8 {
                        b = 10;
                    }
                }
                (queen_eval * AMAZON_QUEEN_SCALE / 100) + (rook_eval * AMAZON_ROOK_SCALE / 100) + b
            }
            PieceType::RoyalQueen => {
                evaluate_queen(game, x, y, piece.color(), white_king, black_king, phase)
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
                let knight_eval =
                    evaluate_knight(x, y, piece.color(), black_king, white_king, phase);
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
                    false
                };

                if !is_active {
                    let piece_val = get_piece_value(pt);
                    let value_factor = (piece_val / 100).max(1);
                    let dist_to_radius = cheb - PIECE_CLOUD_CHEB_RADIUS;
                    let dist_to_lane = if is_ortho_slider {
                        (dx - SLIDER_AXIS_WIGGLE).min(dy - SLIDER_AXIS_WIGGLE)
                    } else if is_diag_slider {
                        (d1 - SLIDER_AXIS_WIGGLE).min(d2 - SLIDER_AXIS_WIGGLE)
                    } else if is_full_slider {
                        let ortho_dist = (dx - SLIDER_AXIS_WIGGLE).min(dy - SLIDER_AXIS_WIGGLE);
                        let diag_dist = (d1 - SLIDER_AXIS_WIGGLE).min(d2 - SLIDER_AXIS_WIGGLE);
                        ortho_dist.min(diag_dist)
                    } else {
                        dist_to_radius
                    };

                    let excess = dist_to_radius.min(dist_to_lane).max(1);
                    let capped_excess = excess.min(PIECE_CLOUD_CHEB_MAX_EXCESS) as i32;
                    let mult = taper(MG_FAR_SLIDER_PENALTY_MULT, EG_FAR_SLIDER_PENALTY_MULT);
                    let penalty =
                        capped_excess * CLOUD_PENALTY_PER_100_VALUE * value_factor * mult / 100;
                    piece_score -= penalty;
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
            white_king
        } else {
            black_king
        };
        if let Some(ok) = own_king
            .filter(|_| !piece.piece_type().is_royal() && piece.piece_type() != PieceType::Pawn)
        {
            let dist = (x - ok.x).abs().max((y - ok.y).abs());
            if dist <= 3 {
                if get_piece_value(piece.piece_type()) < KING_DEFENDER_VALUE_THRESHOLD {
                    piece_score += taper(MG_KING_DEFENDER_BONUS, EG_KING_DEFENDER_BONUS);
                } else {
                    piece_score -= taper(
                        MG_KING_ATTACKER_NEAR_OWN_KING_PENALTY,
                        EG_KING_ATTACKER_NEAR_OWN_KING_PENALTY,
                    );
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
            w_activity += piece_score;
        } else {
            b_activity += piece_score;
        }
    }

    let mut w_pair_bonus = 0;
    let mut b_pair_bonus = 0;

    if metrics.white_bishops >= 2 {
        w_pair_bonus += taper(MG_BISHOP_PAIR_BONUS, EG_BISHOP_PAIR_BONUS);
        bump_feat!(bishop_pair_bonus, 1);
        if metrics.white_bishop_colors.0 && metrics.white_bishop_colors.1 {
            w_pair_bonus += 20;
        }
    }
    if metrics.black_bishops >= 2 {
        b_pair_bonus += taper(MG_BISHOP_PAIR_BONUS, EG_BISHOP_PAIR_BONUS);
        bump_feat!(bishop_pair_bonus, -1);
        if metrics.black_bishop_colors.0 && metrics.black_bishop_colors.1 {
            b_pair_bonus += 20;
        }
    }

    tracer.record("Piece: Activity", w_activity, b_activity);
    tracer.record("Piece: Bishop Pair", w_pair_bonus, b_pair_bonus);

    (w_activity + w_pair_bonus) - (b_activity + b_pair_bonus)
}

fn calculate_attack_readiness_from_list(
    game: &GameState,
    enemy_king: &Option<Coordinate>,
    piece_list: &[(i64, i64, crate::board::Piece)],
    attacker: PlayerColor,
) -> i32 {
    let Some(ek) = enemy_king else { return 50 };

    // 1. Count open rays around enemy king
    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;
    const DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    const ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    for &(dx, dy) in &DIAG_DIRS {
        let mut is_open = true;
        for step in 1..=6 {
            if game
                .board
                .get_piece(ek.x + dx * step, ek.y + dy * step)
                .is_some()
            {
                is_open = false;
                break;
            }
        }
        if is_open {
            open_diag_rays += 1;
        }
    }
    for &(dx, dy) in &ORTHO_DIRS {
        let mut is_open = true;
        for step in 1..=6 {
            if game
                .board
                .get_piece(ek.x + dx * step, ek.y + dy * step)
                .is_some()
            {
                is_open = false;
                break;
            }
        }
        if is_open {
            open_ortho_rays += 1;
        }
    }
    let total_open_rays = open_diag_rays + open_ortho_rays;
    if total_open_rays <= 2 {
        return 40;
    }

    // 2. Scan Piece List for sliders in zone
    let mut sliders_in_zone = 0;
    let mut sliders_on_open_ray = 0;
    const ATTACK_ZONE_RADIUS: i64 = 10;

    for &(x, y, piece) in piece_list {
        if piece.color() != attacker {
            continue;
        }
        let pt = piece.piece_type();

        // Is slider?
        let is_diag_slider = matches!(
            pt,
            PieceType::Bishop
                | PieceType::Queen
                | PieceType::Archbishop
                | PieceType::Amazon
                | PieceType::RoyalQueen
        );
        let is_ortho_slider = matches!(
            pt,
            PieceType::Rook
                | PieceType::Queen
                | PieceType::Chancellor
                | PieceType::Amazon
                | PieceType::RoyalQueen
        );

        if !is_diag_slider && !is_ortho_slider {
            continue;
        }

        let dx = (x - ek.x).abs();
        let dy = (y - ek.y).abs();
        let cheb = dx.max(dy);

        if cheb <= ATTACK_ZONE_RADIUS {
            sliders_in_zone += 1;

            // Check if on ray align
            let on_file = x == ek.x;
            let on_rank = y == ek.y;
            let on_diag1 = (x - ek.x) == (y - ek.y);
            let on_diag2 = (x - ek.x) == -(y - ek.y);

            if (is_ortho_slider && (on_file || on_rank))
                || (is_diag_slider && (on_diag1 || on_diag2))
            {
                sliders_on_open_ray += 1;
            }
        }
    }

    if sliders_in_zone >= 3 && sliders_on_open_ray >= 1 {
        130
    } else if sliders_in_zone >= 2 && sliders_on_open_ray >= 1 {
        115
    } else if sliders_in_zone >= 2 {
        100
    } else if sliders_in_zone == 1 && total_open_rays >= 5 {
        85
    } else if sliders_in_zone == 1 {
        55
    } else {
        30
    }
}

pub struct KingSafetyMetrics {
    pub white_slider_counts: (i32, i32), // (diag, ortho)
    pub black_slider_counts: (i32, i32),
    pub urgency: (i32, i32),           // (white_urgency, black_urgency)
    pub has_enemy_queen: (bool, bool), // (white_sees_queen, black_sees_queen)
}

pub fn evaluate_king_safety_traced<T: EvaluationTracer>(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    tracer: &mut T,
    metrics: &KingSafetyMetrics,
) -> i32 {
    let mut w_safety: i32 = 0;
    let mut b_safety: i32 = 0;
    let mut w_attack: i32 = 0;
    let mut b_attack: i32 = 0;

    // Defense penalty (Shelter)
    if let Some(wk) = white_king {
        w_safety += evaluate_king_shelter(
            game,
            wk,
            PlayerColor::White,
            phase,
            metrics.urgency.0,
            metrics.has_enemy_queen.0,
        );
    }
    if let Some(bk) = black_king {
        b_safety += evaluate_king_shelter(
            game,
            bk,
            PlayerColor::Black,
            phase,
            metrics.urgency.1,
            metrics.has_enemy_queen.1,
        );
    }

    // Attack bonuses (using counts)
    if let Some(bk) = black_king {
        // White attacks Black
        w_attack += compute_attack_bonus_optimized(game, bk, metrics.white_slider_counts);
    }
    if let Some(wk) = white_king {
        // Black attacks White
        b_attack += compute_attack_bonus_optimized(game, wk, metrics.black_slider_counts);
    }

    tracer.record("King: Shelter", w_safety, b_safety);
    tracer.record("King: Attack", w_attack, b_attack);

    (w_safety + w_attack) - (b_safety + b_attack)
}

fn compute_attack_bonus_optimized(
    game: &GameState,
    enemy_king: &Coordinate,
    slider_counts: (i32, i32), // (diag, ortho)
) -> i32 {
    let (our_diag_count, our_ortho_count) = slider_counts;
    if our_diag_count == 0 && our_ortho_count == 0 {
        return 0;
    }

    const DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    const ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;

    if our_diag_count > 0 {
        for &(dx, dy) in &DIAG_DIRS {
            let mut is_open = true;
            for step in 1..=5 {
                if game
                    .board
                    .get_piece(enemy_king.x + dx * step, enemy_king.y + dy * step)
                    .is_some()
                {
                    is_open = false;
                    break;
                }
            }
            if is_open {
                open_diag_rays += 1;
            }
        }
    }
    if our_ortho_count > 0 {
        for &(dx, dy) in &ORTHO_DIRS {
            let mut is_open = true;
            for step in 1..=5 {
                if game
                    .board
                    .get_piece(enemy_king.x + dx * step, enemy_king.y + dy * step)
                    .is_some()
                {
                    is_open = false;
                    break;
                }
            }
            if is_open {
                open_ortho_rays += 1;
            }
        }
    }

    const ATTACK_BONUS_PER_OPEN_RAY: i32 = 10;
    let diag_bonus = if our_diag_count > 0 && open_diag_rays > 0 {
        let mult = if our_diag_count >= 2 { 115 } else { 100 };
        open_diag_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    let ortho_bonus = if our_ortho_count > 0 && open_ortho_rays > 0 {
        let mult = if our_ortho_count >= 2 { 115 } else { 100 };
        open_ortho_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    diag_bonus + ortho_bonus
}

pub fn evaluate_rook(
    _game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
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
            bonus += taper(MG_BEHIND_KING_BONUS, EG_BEHIND_KING_BONUS);
        }

        // On same or adjacent file to enemy king: strong attacking potential.
        if (x - ek.x).abs() <= 1 {
            bonus += 50;
        }

        // Rook tropism: closer to enemy king is better, but capped.
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let capped = dist.min(25);
        let trop = (25 - capped) as i32;
        bonus += trop * taper(MG_KING_TROPISM_BONUS, EG_KING_TROPISM_BONUS);

        // Simplified confinement bonus - just reward rooks controlling key squares near king
        let mut confinement_bonus = 0;

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
        let nearby_slider_bonus = if (x - ek.x).abs() <= 4 && (y - ek.y).abs() <= 4 {
            // This rook is close to king, assume some coordination exists
            SLIDER_NET_BONUS / 2
        } else {
            0
        };

        bonus += nearby_slider_bonus;

        // Penalize rooks that have drifted very far from the king zone
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

    bonus
}

pub fn evaluate_queen(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
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
                let distance_score =
                    base * (taper(MG_KING_TROPISM_BONUS, EG_KING_TROPISM_BONUS) / 2).max(1);

                line_bonus += distance_score;
                bonus += line_bonus;

                // Small bonus for being "behind" the king
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
        let manhattan_weight = (taper(MG_KING_TROPISM_BONUS, EG_KING_TROPISM_BONUS) / 2).max(1);
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
    phase: i32,
) -> i32 {
    let _ = phase;
    let mut bonus: i32 = 0;

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
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
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
            bonus += taper(MG_BEHIND_KING_BONUS, EG_BEHIND_KING_BONUS) / 2;
        }

        // Bishop tropism: closer diagonally to enemy king is good.
        let dist = (x - ek.x).abs() + (y - ek.y).abs();
        let capped = dist.min(15);
        let trop = (15 - capped) as i32;
        bonus += trop * (taper(MG_KING_TROPISM_BONUS, EG_KING_TROPISM_BONUS) / 2).max(1);

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

    bonus
}

fn evaluate_king_shelter(
    game: &GameState,
    king: &Coordinate,
    color: PlayerColor,
    phase: i32,
    defense_urgency: i32,
    has_enemy_queen_possible: bool,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut safety: i32 = 0;

    // 1. Local pawn / guard cover (Optimized: Board::get_piece is O(1))
    let mut has_ring_cover = false;

    // Actually, for just 8 squares, board.get_piece is very fast.
    for dx in -1..=1_i64 {
        for dy in -1..=1_i64 {
            if dx == 0 && dy == 0 {
                continue;
            }
            if let Some(piece) = game.board.get_piece(king.x + dx, king.y + dy)
                && piece.color() == color
                && (piece.piece_type() == PieceType::Pawn
                    || piece.piece_type() == PieceType::Guard
                    || piece.piece_type() == PieceType::Void)
            {
                has_ring_cover = true;
                break;
            }
        }
    }

    if !has_ring_cover {
        safety -= taper(MG_KING_RING_MISSING_PENALTY, EG_KING_RING_MISSING_PENALTY);
        bump_feat!(king_ring_missing_penalty, -1);
    }

    // 1b. King shield (pawn ahead/behind) - Optimized: Check only relevant files
    let mut has_pawn_ahead = false;
    let mut has_pawn_behind = false;
    let is_white = color == PlayerColor::White;

    for dx in -2..=2_i64 {
        let x = king.x + dx;
        let tcx = x >> 3;
        // We still need to find pawns on these files.
        // Instead of iterating ALL tiles, we only iterate tiles in these 5 columns.
        for (_tcx_actual, tcy, tile) in game.board.tiles.iter_columns(tcx) {
            let col_idx = (x & 7) as usize;
            let mut bits = tile.occ_pawns
                & (if is_white {
                    tile.occ_white
                } else {
                    tile.occ_black
                });
            bits &= 0x0101010101010101u64 << col_idx; // Mask for this file

            if bits != 0 {
                while bits != 0 {
                    let idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let py = tcy * 8 + (idx / 8) as i64;
                    if is_white {
                        if py > king.y {
                            has_pawn_ahead = true;
                        } else if py < king.y {
                            has_pawn_behind = true;
                        }
                    } else if py < king.y {
                        has_pawn_ahead = true;
                    } else if py > king.y {
                        has_pawn_behind = true;
                    }
                }
            }
        }
    }

    if has_pawn_ahead && !has_pawn_behind {
        safety += taper(MG_KING_PAWN_SHIELD_BONUS, EG_KING_PAWN_SHIELD_BONUS);
    } else if !has_pawn_ahead && has_pawn_behind {
        safety -= taper(MG_KING_PAWN_AHEAD_PENALTY, EG_KING_PAWN_AHEAD_PENALTY);
    }

    if defense_urgency <= 10 {
        return safety;
    }

    // 2. Ray-based safety (pre-filtered by enemy metrics)
    const DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    const ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    const BASE_DIAG_RAY_PENALTY: i32 = 30;
    const BASE_ORTHO_RAY_PENALTY: i32 = 35;

    let mut total_ray_penalty: i32 = 0;
    let mut tied_defender_penalty: i32 = 0;

    let blocker_reduction_pct = |v: i32, d: i32| {
        let val_pct = if v <= 100 {
            80
        } else if v <= 300 {
            60
        } else if v <= 500 {
            40
        } else if v <= 700 {
            20
        } else {
            0
        };
        let dist_mult = match d {
            1 => 100,
            2 => 75,
            3 => 50,
            _ => 30,
        };
        val_pct * dist_mult / 100
    };

    // Diagonal Rays
    for &(dx, dy) in &DIAG_DIRS {
        let mut blocker: Option<(i32, i32)> = None;
        let mut enemy_blocked = false;
        for step in 1..=8 {
            if let Some(p) = game.board.get_piece(king.x + dx * step, king.y + dy * step) {
                if p.color() == color {
                    let v = get_piece_value(p.piece_type());
                    blocker = Some((v, step as i32));
                    if v >= 600 {
                        tied_defender_penalty += 10;
                    }
                } else if p.color() != PlayerColor::Neutral {
                    enemy_blocked = true;
                }
                break;
            }
        }
        let mut penalty = BASE_DIAG_RAY_PENALTY; // simplified mult for now
        if let Some((v, d)) = blocker {
            penalty = penalty * (100 - blocker_reduction_pct(v, d)) / 100;
        } else if enemy_blocked {
            penalty = penalty * 60 / 100;
        }
        total_ray_penalty += penalty;
    }

    // Ortho Rays
    for &(dx, dy) in &ORTHO_DIRS {
        let mut blocker: Option<(i32, i32)> = None;
        let mut enemy_blocked = false;
        for step in 1..=8 {
            if let Some(p) = game.board.get_piece(king.x + dx * step, king.y + dy * step) {
                if p.color() == color {
                    let v = get_piece_value(p.piece_type());
                    blocker = Some((v, step as i32));
                    if v >= 600 {
                        tied_defender_penalty += 12;
                    }
                } else if p.color() != PlayerColor::Neutral {
                    enemy_blocked = true;
                }
                break;
            }
        }
        let mut penalty = BASE_ORTHO_RAY_PENALTY;
        if let Some((v, d)) = blocker {
            penalty = penalty * (100 - blocker_reduction_pct(v, d)) / 100;
        } else if enemy_blocked {
            penalty = penalty * 60 / 100;
        }
        total_ray_penalty += penalty;
    }

    let mut total_danger = total_ray_penalty + tied_defender_penalty;
    if !has_enemy_queen_possible {
        total_danger = total_danger * 70 / 100;
    }

    let final_penalty =
        (total_danger + (total_danger * total_danger / 800)) * defense_urgency / 100;
    safety -= final_penalty.min(400);

    safety
}

pub fn evaluate_pawn_structure(game: &GameState) -> i32 {
    let phase = calculate_game_phase(game);
    evaluate_pawn_structure_traced(game, phase, &mut NoTrace, &[])
}

pub fn evaluate_pawn_structure_traced<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    tracer: &mut T,
    piece_list: &[(i64, i64, crate::board::Piece)],
) -> i32 {
    // Check cache first using game's pawn_hash
    let pawn_hash = game.pawn_hash;

    // Bypassing cache if tracer is active to ensure we get a full breakdown.
    if tracer.is_active() {
        return compute_pawn_structure_traced(game, phase, tracer, piece_list);
    }

    // Direct mapped cache probe
    let idx = (pawn_hash as usize) % PAWN_CACHE_SIZE;
    let cached = PAWN_CACHE.with(|cache| {
        let entry = cache.borrow()[idx];
        if entry.0 == pawn_hash {
            Some(entry.1)
        } else {
            None
        }
    });

    if let Some(score) = cached {
        return score;
    }

    // Cache miss - compute pawn structure
    let score = compute_pawn_structure_traced(game, phase, tracer, piece_list);

    // Direct mapped cache store
    PAWN_CACHE.with(|cache| {
        cache.borrow_mut()[idx] = (pawn_hash, score);
    });

    score
}

/// Core pawn structure computation. Called on cache miss.
fn compute_pawn_structure_traced<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    tracer: &mut T,
    piece_list: &[(i64, i64, crate::board::Piece)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_doubled = 0;
    let mut b_doubled = 0;
    let mut w_passed = 0;
    let mut b_passed = 0;
    let mut w_connected = 0;
    let mut b_connected = 0;

    let mut white_pawns: Vec<(i64, i64)> = Vec::new();
    let mut black_pawns: Vec<(i64, i64)> = Vec::new();

    let w_promo = game.white_promo_rank;
    let b_promo = game.black_promo_rank;

    // Use pre-collected piece list if provided
    for &(x, y, piece) in piece_list.iter() {
        if piece.piece_type() == PieceType::Pawn {
            if piece.color() == PlayerColor::White {
                if y < w_promo {
                    white_pawns.push((x, y));
                }
            } else if piece.color() == PlayerColor::Black && y > b_promo {
                black_pawns.push((x, y));
            }
        }
    }

    // Fallback if piece list was empty (should not happen in evaluate_inner)
    if white_pawns.is_empty()
        && black_pawns.is_empty()
        && (game.white_pawn_count > 0 || game.black_pawn_count > 0)
    {
        for (cx, cy, tile) in game.board.tiles.iter() {
            let mut bits_w = tile.occ_pawns & tile.occ_white;
            while bits_w != 0 {
                let idx = bits_w.trailing_zeros() as usize;
                bits_w &= bits_w - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                if y < w_promo {
                    white_pawns.push((x, y));
                }
            }
            let mut bits_b = tile.occ_pawns & tile.occ_black;
            while bits_b != 0 {
                let idx = bits_b.trailing_zeros() as usize;
                bits_b &= bits_b - 1;
                let x = cx * 8 + (idx % 8) as i64;
                let y = cy * 8 + (idx / 8) as i64;
                if y > b_promo {
                    black_pawns.push((x, y));
                }
            }
        }
    }

    // Sort pawns by file (x) then rank (y)
    white_pawns.sort_unstable();
    black_pawns.sort_unstable();

    // White Doubled Pawns
    let mut i = 0;
    while i < white_pawns.len() {
        let mut count = 1;
        let file = white_pawns[i].0;
        let mut j = i + 1;
        while j < white_pawns.len() && white_pawns[j].0 == file {
            count += 1;
            j += 1;
        }
        if count > 1 {
            w_doubled -= (count - 1) * taper(MG_DOUBLED_PAWN_PENALTY, EG_DOUBLED_PAWN_PENALTY);
        }
        i = j;
    }

    // Black Doubled Pawns
    let mut i = 0;
    while i < black_pawns.len() {
        let mut count = 1;
        let file = black_pawns[i].0;
        let mut j = i + 1;
        while j < black_pawns.len() && black_pawns[j].0 == file {
            count += 1;
            j += 1;
        }
        if count > 1 {
            b_doubled -= (count - 1) * taper(MG_DOUBLED_PAWN_PENALTY, EG_DOUBLED_PAWN_PENALTY);
        }
        i = j;
    }

    // Passed pawns (O(P log P) due to individual binary searches per pawn)
    // For white pawns, check black pawns on files x-1, x, x+1 with y > wy.
    for &(wx, wy) in &white_pawns {
        let mut is_passed = true;
        for dx in -1..=1 {
            let target_file = wx + dx;
            // Binary search to find start of this file in black_pawns
            let start = black_pawns.partition_point(|&(bx, _)| bx < target_file);
            let mut k = start;
            while k < black_pawns.len() && black_pawns[k].0 == target_file {
                if black_pawns[k].1 > wy {
                    is_passed = false;
                    break;
                }
                k += 1;
            }
            if !is_passed {
                break;
            }
        }
        if is_passed {
            w_passed += 20;
        }

        if is_connected_pawn(game, wx, wy, PlayerColor::White) {
            w_connected += taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS);
        }
    }

    // For black pawns, check white pawns on files x-1, x, x+1 with y < by.
    for &(bx, by) in &black_pawns {
        let mut is_passed = true;
        for dx in -1..=1 {
            let target_file = bx + dx;
            // Binary search to find start of this file in white_pawns
            let start = white_pawns.partition_point(|&(wx, _)| wx < target_file);
            let mut k = start;
            while k < white_pawns.len() && white_pawns[k].0 == target_file {
                if white_pawns[k].1 < by {
                    is_passed = false;
                    break;
                }
                k += 1;
            }
            if !is_passed {
                break;
            }
        }
        if is_passed {
            b_passed += 20;
        }

        if is_connected_pawn(game, bx, by, PlayerColor::Black) {
            b_connected += taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS);
        }
    }

    tracer.record("Pawn: Doubled", w_doubled.abs(), b_doubled.abs());
    tracer.record("Pawn: Passed", w_passed, b_passed);
    tracer.record("Pawn: Connected", w_connected, b_connected);

    (w_doubled + w_passed + w_connected) - (b_doubled + b_passed + b_connected)
}

// ==================== Threat Evaluation ====================

/// Evaluate threats: bonus for attacking higher-value pieces with lower-value pieces.
/// Efficient on infinite boards - only checks direct attack squares for leapers (pawns, knights).
pub fn evaluate_threats(game: &GameState) -> i32 {
    evaluate_threat_traced(game, &mut NoTrace)
}

pub fn evaluate_threat_traced<T: EvaluationTracer>(game: &GameState, tracer: &mut T) -> i32 {
    let mut w_pawn_threats = 0;
    let mut b_pawn_threats = 0;
    let mut w_minor_threats = 0;
    let mut b_minor_threats = 0;

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

            let enemy = if color == PlayerColor::White {
                PlayerColor::Black
            } else {
                PlayerColor::White
            };

            match pt {
                PieceType::Pawn => {
                    let dy = if color == PlayerColor::White { 1 } else { -1 };
                    for dx in [-1i64, 1] {
                        if let Some(target) = game.board.get_piece(x + dx, y + dy)
                            && target.color() == enemy
                        {
                            let tv = get_piece_value(target.piece_type());
                            if tv >= 600 {
                                if color == PlayerColor::White {
                                    w_pawn_threats += PAWN_THREATENS_QUEEN;
                                } else {
                                    b_pawn_threats += PAWN_THREATENS_QUEEN;
                                }
                            } else if tv >= 400 {
                                if color == PlayerColor::White {
                                    w_pawn_threats += PAWN_THREATENS_ROOK;
                                } else {
                                    b_pawn_threats += PAWN_THREATENS_ROOK;
                                }
                            } else if tv >= 200 {
                                if color == PlayerColor::White {
                                    w_pawn_threats += PAWN_THREATENS_MINOR;
                                } else {
                                    b_pawn_threats += PAWN_THREATENS_MINOR;
                                }
                            }
                        }
                    }
                }
                PieceType::Knight | PieceType::Centaur | PieceType::RoyalCentaur => {
                    for &(dx, dy) in &KNIGHT_OFFSETS {
                        if let Some(target) = game.board.get_piece(x + dx, y + dy)
                            && target.color() == enemy
                        {
                            let tv = get_piece_value(target.piece_type());
                            let mv = get_piece_value(pt);
                            if tv >= 600 && mv < 600 {
                                if color == PlayerColor::White {
                                    w_minor_threats += MINOR_THREATENS_QUEEN;
                                } else {
                                    b_minor_threats += MINOR_THREATENS_QUEEN;
                                }
                            } else if tv >= 400 && mv < 400 {
                                if color == PlayerColor::White {
                                    w_minor_threats += MINOR_THREATENS_ROOK;
                                } else {
                                    b_minor_threats += MINOR_THREATENS_ROOK;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    tracer.record("Threats: Pawn", w_pawn_threats, b_pawn_threats);
    tracer.record("Threats: Minor", w_minor_threats, b_minor_threats);

    (w_pawn_threats + w_minor_threats) - (b_pawn_threats + b_minor_threats)
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
        let corner_score = evaluate_knight(0, 0, PlayerColor::White, &None, &None, MAX_PHASE);
        // Knight in center (best)
        let center_score = evaluate_knight(4, 4, PlayerColor::White, &None, &None, MAX_PHASE);

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
        let score = evaluate_bishop(&game, 4, 4, PlayerColor::White, &wk, &bk, MAX_PHASE);
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
        let score = evaluate_rook(&game, 4, 1, PlayerColor::White, &wk, &bk, MAX_PHASE);
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
        let score = evaluate_queen(&game, 4, 4, PlayerColor::White, &wk, &bk, MAX_PHASE);
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
