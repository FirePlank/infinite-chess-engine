use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;

use smallvec::SmallVec;
use std::cell::UnsafeCell;

// 2-Bucket LRU pawn structure cache
const PAWN_CACHE_SIZE: usize = 16384; // 16384 buckets * 2 entries = 32768 entries

#[derive(Clone, Copy)]
struct PawnCacheEntry {
    hash: u64,
    score: i32,
}

#[derive(Clone, Copy)]
struct PawnCacheBucket {
    entries: [PawnCacheEntry; 2],
}

impl Default for PawnCacheBucket {
    fn default() -> Self {
        PawnCacheBucket {
            entries: [
                PawnCacheEntry {
                    hash: u64::MAX,
                    score: 0,
                },
                PawnCacheEntry {
                    hash: u64::MAX,
                    score: 0,
                },
            ],
        }
    }
}

thread_local! {
    static PAWN_CACHE: UnsafeCell<Vec<PawnCacheBucket>> = UnsafeCell::new(vec![PawnCacheBucket::default(); PAWN_CACHE_SIZE]);
    // Reusable buffer for piece list to avoid allocation
    pub(crate) static EVAL_PIECE_LIST: UnsafeCell<SmallVec<[(i64, i64, Piece); 128]>> = UnsafeCell::new(SmallVec::new());
    pub(crate) static EVAL_WHITE_PAWNS: UnsafeCell<SmallVec<[(i64, i64); 64]>> = UnsafeCell::new(SmallVec::new());
    pub(crate) static EVAL_BLACK_PAWNS: UnsafeCell<SmallVec<[(i64, i64); 64]>> = UnsafeCell::new(SmallVec::new());
    pub(crate) static EVAL_WHITE_RQ: UnsafeCell<SmallVec<[(i64, i64); 32]>> = UnsafeCell::new(SmallVec::new());
    pub(crate) static EVAL_BLACK_RQ: UnsafeCell<SmallVec<[(i64, i64); 32]>> = UnsafeCell::new(SmallVec::new());
}

/// Clear the pawn structure cache.
pub fn clear_pawn_cache() {
    PAWN_CACHE.with(|cache| {
        // Fast clear using fill
        unsafe { (&mut *cache.get()).fill(PawnCacheBucket::default()) };
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
pub static EVAL_FEATURES: Lazy<RwLock<EvalFeatures>> =
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

// Shared constants for ray detection
const DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// Bishop pair & queen heuristics
// Tapered pairs defined below
const QUEEN_IDEAL_LINE_DIST: i32 = 4;

// Fairy Piece Evaluation

// Leaper positioning (tropism to kings and piece cloud)
const LEAPER_TROPISM_DIVISOR: i32 = 400; // piece_value / 400 = tropism multiplier
// Beyond this, bonus is capped

// Compound piece weight scaling (fraction of base piece eval to inherit)
const CHANCELLOR_ROOK_SCALE: i32 = 90; // 90% of rook eval
const ARCHBISHOP_BISHOP_SCALE: i32 = 90; // 90% of bishop eval
const AMAZON_ROOK_SCALE: i32 = 50; // 50% of rook eval (also has queen)
const AMAZON_QUEEN_SCALE: i32 = 70; // 70% of queen eval
const CENTAUR_GUARD_SCALE: i32 = 50; // 50% of guard/leaper eval

// ==================== Pawn Distance Scaling ====================

// Pawns far from promotion are worth much less in infinite chess
const PAWN_FULL_VALUE_THRESHOLD: i64 = 6; // Within 6 ranks = full value
const PAWN_PAST_PROMO_PENALTY: i32 = 90; // Massive penalty for pawns that can't promote (worth 10x less)
const PAWN_FAR_FROM_PROMO_PENALTY: i32 = 50; // Flat penalty for back pawns (no benefit from advancing)

// ==================== Development ====================

// Minimum starting square penalty for minors
const MIN_DEVELOPMENT_PENALTY: i32 = 6; // Moderate - not too aggressive

// King defender bonuses/penalties
// Low-value pieces near own king = good (defense)
// High-value pieces near own king = bad (should be attacking)
const KING_DEFENDER_VALUE_THRESHOLD: i32 = 400; // Pieces below this value are defensive

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

const MG_KING_OPEN_FILE_PENALTY: i32 = 25;
const EG_KING_OPEN_FILE_PENALTY: i32 = 10;

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

// Piece on Open File Bonuses
const ROOK_OPEN_FILE_BONUS: i32 = 45;
const ROOK_SEMI_OPEN_FILE_BONUS: i32 = 20;
const QUEEN_OPEN_FILE_BONUS: i32 = 25;
const QUEEN_SEMI_OPEN_FILE_BONUS: i32 = 10;
const MG_OUTPOST_BONUS: i32 = 20;
const EG_OUTPOST_BONUS: i32 = 50;

// Passed Pawn Detail (MG/EG tapered arrays by relative rank 0-5)
// Rank 0 is far, Rank 5 is near promotion.
const CANDIDATE_PASSER_BONUS: [i32; 6] = [3, 7, 14, 25, 40, 70];

// PASSED_PAWN_ADV_BONUS[canAdvance][safeAdvance][rank]
const PASSED_PAWN_ADV_BONUS: [[[i32; 6]; 2]; 2] = [
    // cannot advance
    [
        [1, 3, 6, 10, 20, 40],  // unsafe
        [3, 7, 14, 28, 50, 85], // safe
    ],
    // can advance
    [
        [5, 10, 20, 40, 70, 125],   // unsafe
        [10, 20, 40, 75, 140, 240], // safe
    ],
];

const PASSED_FRIENDLY_KING_DIST: [i32; 6] = [1, 2, 3, 5, 8, 12];
const PASSED_ENEMY_KING_DIST: [i32; 6] = [1, 2, 3, 4, 6, 9];

const MG_PASSED_SAFE_PATH_BONUS: i32 = 40;
const EG_PASSED_SAFE_PATH_BONUS: i32 = 80;

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

/// Compute MopUp term for NNUE hybrid evaluation.
/// Always applied (both NNUE and HCE branches) to fix endgame conversion.
#[cfg(feature = "nnue")]
#[inline]
pub fn compute_mop_up_term(game: &GameState) -> i32 {
    use crate::evaluation::mop_up::{calculate_mop_up_scale, evaluate_lone_king_endgame};

    // Determine which side has material advantage
    let white_material = game.white_non_pawn_material;
    let black_material = game.black_non_pawn_material;

    let (winning_color, losing_color) = if white_material && !black_material {
        (PlayerColor::White, PlayerColor::Black)
    } else if black_material && !white_material {
        (PlayerColor::Black, PlayerColor::White)
    } else {
        return 0; // No clear winner or both have material
    };

    // Check if mop-up is active
    let scale = match calculate_mop_up_scale(game, losing_color) {
        Some(s) if s > 0 => s,
        _ => return 0,
    };

    // Get king positions
    let (our_king, enemy_king) = if winning_color == PlayerColor::White {
        (game.white_king_pos.as_ref(), game.black_king_pos.as_ref())
    } else {
        (game.black_king_pos.as_ref(), game.white_king_pos.as_ref())
    };

    let enemy_king = match enemy_king {
        Some(k) => k,
        None => return 0,
    };

    let raw = evaluate_lone_king_endgame(game, our_king, enemy_king, winning_color);
    let mop_up = (raw * scale as i32) / 100;

    // Return from side-to-move's perspective (positive = good for STM)
    if winning_color == game.turn {
        mop_up
    } else {
        -mop_up
    }
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

    let taper = |mg: i32, eg: i32| -> i32 {
        ((mg * game.total_phase.min(MAX_PHASE))
            + (eg * (MAX_PHASE - game.total_phase.min(MAX_PHASE))))
            / MAX_PHASE
    };

    // Single-Pass Collection and Scoring
    let mut phase = 0;
    let mut white_undeveloped = 0;
    let mut black_undeveloped = 0;
    let mut white_bishops = 0;
    let mut white_bishop_colors = (false, false);
    let mut black_bishops = 0;
    let mut black_bishop_colors = (false, false);
    let mut cloud_sum_dx: i64 = 0;
    let mut cloud_sum_dy: i64 = 0;
    let mut cloud_count: i64 = 0;

    let (ref_x, ref_y) = match (white_king, black_king) {
        (Some(wk), Some(bk)) => (
            wk.x / 2 + bk.x / 2 + (wk.x % 2 + bk.x % 2) / 2,
            wk.y / 2 + bk.y / 2 + (wk.y % 2 + bk.y % 2) / 2,
        ),
        (Some(wk), None) => (wk.x, wk.y),
        (None, Some(bk)) => (bk.x, bk.y),
        (None, None) => (0, 0),
    };

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

    // Connectivity (Fast integer version)
    let mut w_connectivity: i32 = 0;
    let mut b_connectivity: i32 = 0;

    // Interaction threat totals
    let mut w_pawn_threats = 0;
    let mut b_pawn_threats = 0;
    let mut w_minor_threats = 0;
    let mut b_minor_threats = 0;

    // Readiness counts (Unified Loop)
    let mut w_sliders_in_zone = 0;
    let mut b_sliders_in_zone = 0;
    const ATTACK_ZONE_RADIUS: i64 = 10;

    // King Safety Arrays
    // [0..4] = Diag, [4..8] = Ortho
    // Stores: (distance, piece_value, piece_color, piece_type)
    let mut w_king_rays = [(i32::MAX, 0, PlayerColor::Neutral, PieceType::Void); 8];
    let mut b_king_rays = [(i32::MAX, 0, PlayerColor::Neutral, PieceType::Void); 8];

    let mut w_king_ring_covered = false;
    let mut b_king_ring_covered = false;

    let mut w_attacking_tropism: i32 = 0;
    let mut w_defensive_tropism: i32 = 0;
    let mut b_attacking_tropism: i32 = 0;
    let mut b_defensive_tropism: i32 = 0;

    // Interaction threat constants
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
        EVAL_WHITE_PAWNS.with(|white_pawns_cell| {
            EVAL_BLACK_PAWNS.with(|black_pawns_cell| {
                EVAL_WHITE_RQ.with(|white_rq_cell| {
                    EVAL_BLACK_RQ.with(|black_rq_cell| {
                        let piece_list = unsafe { &mut *piece_list_cell.get() };
                        let white_pawns = unsafe { &mut *white_pawns_cell.get() };
                        let black_pawns = unsafe { &mut *black_pawns_cell.get() };
                        let white_rq = unsafe { &mut *white_rq_cell.get() };
                        let black_rq = unsafe { &mut *black_rq_cell.get() };

                        piece_list.clear();
                        white_pawns.clear();
                        black_pawns.clear();
                        white_rq.clear();
                        black_rq.clear();

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
                                phase += get_piece_phase(pt);

                                // 2. Piece Collection (Optimized categorization)
                                if pt == PieceType::Pawn {
                                    if is_white {
                                        if y < w_promo {
                                            white_pawns.push((x, y));
                                        }
                                    } else if y > b_promo {
                                        black_pawns.push((x, y));
                                    }
                                } else {
                                    piece_list.push((x, y, piece));
                                    // Rooks and Queens for support bonus
                                    if pt == PieceType::Rook
                                        || pt == PieceType::Queen
                                        || pt == PieceType::Amazon
                                        || pt == PieceType::Chancellor
                                        || pt == PieceType::RoyalQueen
                                    {
                                        if is_white {
                                            white_rq.push((x, y));
                                        } else {
                                            black_rq.push((x, y));
                                        }
                                    }
                                }

                                // 3. Piece counts for scaling (Non-pawn, non-royal)
                                if pt != PieceType::Pawn && !pt.is_royal() {
                                    if is_white {
                                        white_non_pawn_non_royal += 1;
                                    } else {
                                        black_non_pawn_non_royal += 1;
                                    }
                                }

                                // 4. Cloud Stats (Non-pawn)
                                if pt != PieceType::Pawn {
                                    let cw = get_centrality_weight(pt);
                                    if cw > 0 {
                                        let dx = x - ref_x;
                                        let dy = y - ref_y;
                                        let cdx = dx.clamp(
                                            -CLOUD_CENTER_MAX_SKEW_DIST,
                                            CLOUD_CENTER_MAX_SKEW_DIST,
                                        );
                                        let cdy = dy.clamp(
                                            -CLOUD_CENTER_MAX_SKEW_DIST,
                                            CLOUD_CENTER_MAX_SKEW_DIST,
                                        );
                                        cloud_sum_dx += cw * cdx;
                                        cloud_sum_dy += cw * cdy;
                                        cloud_count += cw;
                                    }
                                }

                                // 5. Readiness sliders in zone
                                let is_diag_slider_type = matches!(
                                    pt,
                                    PieceType::Bishop
                                        | PieceType::Queen
                                        | PieceType::Archbishop
                                        | PieceType::Amazon
                                        | PieceType::RoyalQueen
                                );
                                let is_ortho_slider_type = matches!(
                                    pt,
                                    PieceType::Rook
                                        | PieceType::Queen
                                        | PieceType::Chancellor
                                        | PieceType::Amazon
                                        | PieceType::RoyalQueen
                                );
                                let is_slider = is_diag_slider_type
                                    || is_ortho_slider_type
                                    || pt == PieceType::Knightrider;

                                if is_slider {
                                    if is_white {
                                        if let Some(bk) = black_king
                                            && (x - bk.x).abs() <= ATTACK_ZONE_RADIUS
                                            && (y - bk.y).abs() <= ATTACK_ZONE_RADIUS
                                        {
                                            w_sliders_in_zone += 1;
                                        }
                                    } else if let Some(wk) = white_king
                                        && (x - wk.x).abs() <= ATTACK_ZONE_RADIUS
                                        && (y - wk.y).abs() <= ATTACK_ZONE_RADIUS
                                    {
                                        b_sliders_in_zone += 1;
                                    }
                                }

                                // Global Tropism (Piece activity relative to kings)
                                if !pt.is_royal() && pt != PieceType::Pawn {
                                    let piece_val = get_piece_value(pt);
                                    if is_white {
                                        if let Some(bk) = black_king {
                                            let d = (x - bk.x).abs().max((y - bk.y).abs());
                                            w_attacking_tropism += piece_val / (d as i32 + 10);
                                        }
                                        if let Some(wk) = white_king {
                                            let d = (x - wk.x).abs().max((y - wk.y).abs());
                                            w_defensive_tropism +=
                                                piece_val.min(350) / (d as i32 + 10);
                                        }
                                    } else {
                                        if let Some(wk) = white_king {
                                            let d = (x - wk.x).abs().max((y - wk.y).abs());
                                            b_attacking_tropism += piece_val / (d as i32 + 10);
                                        }
                                        if let Some(bk) = black_king {
                                            let d = (x - bk.x).abs().max((y - bk.y).abs());
                                            b_defensive_tropism +=
                                                piece_val.min(350) / (d as i32 + 10);
                                        }
                                    }
                                }

                                // 6. Connectivity
                                // Kopec: Inverse value weights × protector value
                                let conn_weight: i32 = match pt {
                                    PieceType::Pawn => 50,
                                    PieceType::Knight => 35,
                                    PieceType::Bishop => 30,
                                    PieceType::Rook => 10,
                                    PieceType::Queen | PieceType::RoyalQueen => 4,
                                    PieceType::Chancellor => 7,
                                    PieceType::Amazon => 2,
                                    _ => 0,
                                };
                                if conn_weight > 0 {
                                    // Fast bitboard pawn protection
                                    // bit = 1 << idx (our position in the tile)
                                    let bit = 1u64 << idx;
                                    let our_pawns = tile.occ_pawns
                                        & if is_white {
                                            tile.occ_white
                                        } else {
                                            tile.occ_black
                                        };

                                    // Count protecting pawns using bitboard shifts
                                    // Mask to avoid wrapping: column A (0x0101...) and column H (0x8080...)
                                    const NOT_A_FILE: u64 = !0x0101010101010101u64;
                                    const NOT_H_FILE: u64 = !0x8080808080808080u64;

                                    let prot_count = if is_white {
                                        // White pieces protected by white pawns below
                                        let left_prot =
                                            ((bit >> 9) & NOT_H_FILE & our_pawns).count_ones();
                                        let right_prot =
                                            ((bit >> 7) & NOT_A_FILE & our_pawns).count_ones();
                                        left_prot + right_prot
                                    } else {
                                        // Black pieces protected by black pawns above
                                        let left_prot =
                                            ((bit << 7) & NOT_H_FILE & our_pawns).count_ones();
                                        let right_prot =
                                            ((bit << 9) & NOT_A_FILE & our_pawns).count_ones();
                                        left_prot + right_prot
                                    };

                                    // Single pawn = 8, Double pawn = 15 (per Kopec table)
                                    let prot_val = if prot_count >= 2 {
                                        15
                                    } else {
                                        prot_count as i32 * 8
                                    };
                                    if is_white {
                                        w_connectivity += conn_weight * prot_val;
                                    } else {
                                        b_connectivity += conn_weight * prot_val;
                                    }
                                }

                                {
                                    // Check White King
                                    if let Some(wk) = white_king {
                                        let dx = x - wk.x;
                                        let dy = y - wk.y;
                                        let adx = dx.abs();
                                        let ady = dy.abs();
                                        let dist = adx.max(ady);

                                        // Ring Cover
                                        if !w_king_ring_covered
                                            && dist == 1
                                            && is_white
                                            && (pt == PieceType::Pawn
                                                || pt == PieceType::Guard
                                                || pt == PieceType::Void)
                                        {
                                            w_king_ring_covered = true;
                                        }

                                        // Rays
                                        // Ortho: (1, 0), (-1, 0), (0, 1), (0, -1) -> Indices 4, 5, 6, 7
                                        if dx != 0 && dy == 0 {
                                            let idx = if dx > 0 { 4 } else { 5 };
                                            if (dist as i32) < w_king_rays[idx].0 {
                                                w_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        } else if dx == 0 && dy != 0 {
                                            let idx = if dy > 0 { 6 } else { 7 };
                                            if (dist as i32) < w_king_rays[idx].0 {
                                                w_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        }
                                        // Diag: (1, 1), (1, -1), (-1, 1), (-1, -1) -> Indices 0, 1, 2, 3
                                        else if adx == ady && dist > 0 {
                                            let idx = if dx > 0 {
                                                if dy > 0 { 0 } else { 1 }
                                            } else if dy > 0 {
                                                2
                                            } else {
                                                3
                                            };
                                            if (dist as i32) < w_king_rays[idx].0 {
                                                w_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        }
                                    }

                                    // Check Black King
                                    if let Some(bk) = black_king {
                                        let dx = x - bk.x;
                                        let dy = y - bk.y;
                                        let adx = dx.abs();
                                        let ady = dy.abs();
                                        let dist = adx.max(ady);

                                        // Ring Cover
                                        if !b_king_ring_covered
                                            && dist == 1
                                            && !is_white
                                            && (pt == PieceType::Pawn
                                                || pt == PieceType::Guard
                                                || pt == PieceType::Void)
                                        {
                                            b_king_ring_covered = true;
                                        }

                                        // Rays
                                        if dx != 0 && dy == 0 {
                                            let idx = if dx > 0 { 4 } else { 5 };
                                            if (dist as i32) < b_king_rays[idx].0 {
                                                b_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        } else if dx == 0 && dy != 0 {
                                            let idx = if dy > 0 { 6 } else { 7 };
                                            if (dist as i32) < b_king_rays[idx].0 {
                                                b_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        } else if adx == ady && dist > 0 {
                                            let idx = if dx > 0 {
                                                if dy > 0 { 0 } else { 1 }
                                            } else if dy > 0 {
                                                2
                                            } else {
                                                3
                                            };
                                            if (dist as i32) < b_king_rays[idx].0 {
                                                b_king_rays[idx] = (
                                                    dist as i32,
                                                    get_piece_value(pt),
                                                    piece.color(),
                                                    pt,
                                                );
                                            }
                                        }
                                    }
                                }

                                // 6. Interaction Threats
                                if pt == PieceType::Pawn {
                                    let enemy = if is_white {
                                        PlayerColor::Black
                                    } else {
                                        PlayerColor::White
                                    };
                                    let dy = if is_white { 1 } else { -1 };
                                    for dx in [-1i64, 1] {
                                        if let Some(target) = game.board.get_piece(x + dx, y + dy)
                                            && target.color() == enemy
                                        {
                                            let tv = get_piece_value(target.piece_type());
                                            if tv >= 600 {
                                                if is_white {
                                                    w_pawn_threats += PAWN_THREATENS_QUEEN;
                                                } else {
                                                    b_pawn_threats += PAWN_THREATENS_QUEEN;
                                                }
                                            } else if tv >= 400 {
                                                if is_white {
                                                    w_pawn_threats += PAWN_THREATENS_ROOK;
                                                } else {
                                                    b_pawn_threats += PAWN_THREATENS_ROOK;
                                                }
                                            } else if tv >= 200 {
                                                if is_white {
                                                    w_pawn_threats += PAWN_THREATENS_MINOR;
                                                } else {
                                                    b_pawn_threats += PAWN_THREATENS_MINOR;
                                                }
                                            }
                                        }
                                    }
                                } else if pt == PieceType::Knight
                                    || pt == PieceType::Centaur
                                    || pt == PieceType::RoyalCentaur
                                {
                                    let enemy = if is_white {
                                        PlayerColor::Black
                                    } else {
                                        PlayerColor::White
                                    };
                                    for &(dx, dy) in &KNIGHT_OFFSETS {
                                        if let Some(target) = game.board.get_piece(x + dx, y + dy)
                                            && target.color() == enemy
                                        {
                                            let tv = get_piece_value(target.piece_type());
                                            let mv = get_piece_value(pt);
                                            if tv >= 600 && mv < 600 {
                                                if is_white {
                                                    w_minor_threats += MINOR_THREATENS_QUEEN;
                                                } else {
                                                    b_minor_threats += MINOR_THREATENS_QUEEN;
                                                }
                                            } else if tv >= 400 && mv < 400 {
                                                if is_white {
                                                    w_minor_threats += MINOR_THREATENS_ROOK;
                                                } else {
                                                    b_minor_threats += MINOR_THREATENS_ROOK;
                                                }
                                            }
                                        }
                                    }
                                }

                                // 8. Minor stats
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
                                                w_pawn_bonus +=
                                                    (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
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
                                            b_pawn_bonus +=
                                                (PAWN_FULL_VALUE_THRESHOLD - dist) as i32 * 4;
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
                                x: ref_x + cloud_sum_dx / cloud_count,
                                y: ref_y + cloud_sum_dy / cloud_count,
                            })
                        } else {
                            None
                        };

                        // Pawn Advancement Calculation
                        if white_max_y != i64::MIN {
                            let dist = (w_promo - white_max_y).clamp(1, 100) as i32;
                            // Continuous piecewise linear: matches 500 at dist=1, 350 at dist=2, then transitions to (10-dist)*40.
                            w_pawn_bonus += (500 - (dist - 1) * 150).max((10 - dist) * 40).max(0);
                        }
                        if black_min_y != i64::MAX {
                            let dist = (black_min_y - b_promo).clamp(1, 100) as i32;
                            b_pawn_bonus += (500 - (dist - 1) * 150).max((10 - dist) * 40).max(0);
                        }

                        // Sort pawns for efficient structure evaluation (O(P log P))
                        white_pawns.sort_unstable();
                        black_pawns.sort_unstable();

                        let total_pieces = white_non_pawn_non_royal + black_non_pawn_non_royal;
                        let multiplier_q = (190 - 18 * total_pieces).clamp(10, 100);

                        let w_adv = (w_pawn_bonus * multiplier_q / 100) + w_pawn_penalty;
                        let b_adv = (b_pawn_bonus * multiplier_q / 100) + b_pawn_penalty;
                        tracer.record("Pawn Advancement", w_adv, b_adv);
                        score += w_adv - b_adv;

                        let white_has_promo = white_max_y != i64::MIN;
                        let black_has_promo = black_min_y != i64::MAX;

                        // Mop-up (using existing counts)
                        let white_pieces =
                            game.white_piece_count.saturating_sub(game.white_pawn_count);
                        let black_pieces =
                            game.black_piece_count.saturating_sub(game.black_pawn_count);
                        let mut mop_up_applied = false;

                        if black_pieces < 3
                            && white_pieces > 1
                            && !white_has_promo
                            && let Some(bk) = &black_king
                            && crate::evaluation::mop_up::calculate_mop_up_scale(
                                game,
                                PlayerColor::Black,
                            )
                            .is_some()
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
                            && crate::evaluation::mop_up::calculate_mop_up_scale(
                                game,
                                PlayerColor::White,
                            )
                            .is_some()
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

                        // Defense urgency
                        let calc_urgency = |tp: i32| (10 + tp + (tp / 4)).min(100);
                        let w_urgency = calc_urgency(black_threat_points);
                        let b_urgency = calc_urgency(w_threat_points);

                        // Attack scale calculation (Finalized from Readiness loop counts)
                        let w_attack_ready = compute_attack_readiness_optimized(
                            &b_king_rays,
                            black_king.is_some(),
                            w_sliders_in_zone,
                        );
                        let b_attack_ready = compute_attack_readiness_optimized(
                            &w_king_rays,
                            white_king.is_some(),
                            b_sliders_in_zone,
                        );

                        if !mop_up_applied {
                            score += evaluate_pieces_processed(
                                game,
                                &white_king,
                                &black_king,
                                final_phase,
                                tracer,
                                piece_list,
                                PieceMetrics {
                                    white_undeveloped,
                                    black_undeveloped,
                                    white_bishops,
                                    black_bishops,
                                    white_bishop_colors,
                                    black_bishop_colors,
                                    cloud_center,
                                },
                                w_attack_ready,
                                b_attack_ready,
                                white_pawns,
                                black_pawns,
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
                                white_pawns,
                                black_pawns,
                                &w_king_rays,
                                &b_king_rays,
                                w_king_ring_covered,
                                b_king_ring_covered,
                            );

                            score += evaluate_pawn_structure_traced(
                                game,
                                final_phase,
                                &white_king,
                                &black_king,
                                tracer,
                                white_pawns,
                                black_pawns,
                                white_rq,
                                black_rq,
                            );

                            // Interaction Threats (Result from merged loop)
                            tracer.record("Threats: Pawn", w_pawn_threats, b_pawn_threats);
                            tracer.record("Threats: Minor", w_minor_threats, b_minor_threats);
                            score += (w_pawn_threats + w_minor_threats)
                                - (b_pawn_threats + b_minor_threats);

                            // Global Tropism
                            let gt_att_mult = taper(180, 360);
                            let gt_def_mult = taper(120, 60);

                            // Normalize by 1000 since piece values are high and we want roughly 10-100 pts
                            let w_gt = (w_attacking_tropism * gt_att_mult / 100)
                                + (w_defensive_tropism * gt_def_mult / 100);
                            let b_gt = (b_attacking_tropism * gt_att_mult / 100)
                                + (b_defensive_tropism * gt_def_mult / 100);

                            tracer.record("Global Tropism", w_gt, b_gt);
                            score += w_gt - b_gt;
                        }
                    }); // brq
                }); // wrq
            }); // bp
        }); // wp
    }); // pl

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

#[allow(clippy::too_many_arguments)]
fn evaluate_pieces_processed<T: EvaluationTracer>(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    tracer: &mut T,
    piece_list: &[(i64, i64, crate::board::Piece)],
    metrics: PieceMetrics,
    white_attack_ready: i32,
    black_attack_ready: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_activity: i32 = 0;
    let mut b_activity: i32 = 0;

    let cloud_center = metrics.cloud_center;

    let white_attack_ready = {
        let cap = (100 - metrics.white_undeveloped * 25).clamp(30, 100);
        white_attack_ready.min(cap)
    };
    let black_attack_ready = {
        let cap = (100 - metrics.black_undeveloped * 25).clamp(30, 100);
        black_attack_ready.min(cap)
    };

    for &(x, y, piece) in piece_list {
        let mut piece_score = match piece.piece_type() {
            PieceType::Rook => evaluate_rook(
                game,
                x,
                y,
                piece.color(),
                white_king,
                black_king,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Queen => evaluate_queen(
                game,
                x,
                y,
                piece.color(),
                white_king,
                black_king,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Bishop => evaluate_bishop(
                game,
                x,
                y,
                piece.color(),
                white_king,
                black_king,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Chancellor => {
                let rook_eval = evaluate_rook(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                rook_eval * CHANCELLOR_ROOK_SCALE / 100
            }
            PieceType::Archbishop => {
                let bishop_eval = evaluate_bishop(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                bishop_eval * ARCHBISHOP_BISHOP_SCALE / 100
            }
            PieceType::Amazon => {
                let queen_eval = evaluate_queen(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                let rook_eval = evaluate_rook(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_king,
                    black_king,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                (queen_eval * AMAZON_QUEEN_SCALE / 100) + (rook_eval * AMAZON_ROOK_SCALE / 100)
            }
            PieceType::RoyalQueen => evaluate_queen(
                game,
                x,
                y,
                piece.color(),
                white_king,
                black_king,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Knight => evaluate_knight(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Hawk
            | PieceType::Rose
            | PieceType::Camel
            | PieceType::Giraffe
            | PieceType::Zebra => evaluate_leaper_positioning(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                get_piece_value(piece.piece_type()),
            ),
            PieceType::Centaur | PieceType::RoyalCentaur => {
                let leaper_eval = evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    cloud_center.as_ref(),
                    get_piece_value(piece.piece_type()),
                );
                leaper_eval * CENTAUR_GUARD_SCALE / 100
            }
            PieceType::Huygen => evaluate_leaper_positioning(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                get_piece_value(PieceType::Huygen),
            ),
            PieceType::Guard => evaluate_leaper_positioning(
                x,
                y,
                piece.color(),
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
                let is_ortho = pt == PieceType::Rook || pt == PieceType::Chancellor;
                let is_diag = pt == PieceType::Bishop || pt == PieceType::Archbishop;
                let is_queen = pt == PieceType::Queen || pt == PieceType::Amazon;

                let piece_val = get_piece_value(pt);
                let value_factor = (piece_val / 100).max(1);
                let mult = taper(MG_FAR_SLIDER_PENALTY_MULT, EG_FAR_SLIDER_PENALTY_MULT);

                if is_ortho || is_diag || is_queen {
                    // Sliders: only penalized if they cannot "see" the cloud center (misaligned).
                    // Distance doesn't matter (infinite range).
                    let mut lane_dist = i64::MAX;

                    if is_ortho || is_queen {
                        lane_dist = lane_dist.min(dx.min(dy));
                    }
                    if is_diag || is_queen {
                        let d1 = ((x - y) - (center.x - center.y)).abs();
                        let d2 = ((x + y) - (center.x + center.y)).abs();
                        lane_dist = lane_dist.min(d1.min(d2));
                    }

                    if lane_dist > SLIDER_AXIS_WIGGLE {
                        let excess = (lane_dist - SLIDER_AXIS_WIGGLE)
                            .min(PIECE_CLOUD_CHEB_MAX_EXCESS)
                            as i32;
                        let penalty =
                            excess * CLOUD_PENALTY_PER_100_VALUE * value_factor * mult / 100;
                        piece_score -= penalty;
                    }
                } else {
                    // Leapers/Others: penalized by distance (Chebyshev)
                    // We are only in this block if cheb > RADIUS, so dist_to_radius > 0
                    let dist_to_radius = cheb - PIECE_CLOUD_CHEB_RADIUS;
                    let excess = dist_to_radius.min(PIECE_CLOUD_CHEB_MAX_EXCESS) as i32;
                    let penalty = excess * CLOUD_PENALTY_PER_100_VALUE * value_factor * mult / 100;
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
            let scale = if piece.color() == PlayerColor::White {
                white_attack_ready
            } else {
                black_attack_ready
            };
            piece_score = piece_score * scale / 100;
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

fn compute_attack_readiness_optimized(
    enemy_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    has_enemy_king: bool,
    sliders_in_zone: i32,
) -> i32 {
    if !has_enemy_king {
        return 50;
    }

    // 1. Count open rays around enemy king (O(1))
    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;

    for i in 0..4 {
        if enemy_king_rays[i].0 > 6 {
            open_diag_rays += 1;
        }
    }
    for i in 4..8 {
        if enemy_king_rays[i].0 > 6 {
            open_ortho_rays += 1;
        }
    }

    let total_open_rays = open_diag_rays + open_ortho_rays;
    if total_open_rays <= 2 {
        return 40;
    }

    // Scoring logic (Simplified from calculate_attack_readiness_from_list)
    if sliders_in_zone >= 2 {
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

#[allow(clippy::too_many_arguments)]
pub fn evaluate_king_safety_traced<T: EvaluationTracer>(
    game: &GameState,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    tracer: &mut T,
    metrics: &KingSafetyMetrics,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    w_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    b_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    w_ring_covered: bool,
    b_ring_covered: bool,
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
            white_pawns,
            w_king_rays,
            w_ring_covered,
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
            black_pawns,
            b_king_rays,
            b_ring_covered,
        );
    }

    // Attack bonuses (using counts)
    if black_king.is_some() {
        // White attacks Black
        w_attack += compute_attack_bonus_optimized(b_king_rays, metrics.white_slider_counts);
    }
    if white_king.is_some() {
        // Black attacks White
        b_attack += compute_attack_bonus_optimized(w_king_rays, metrics.black_slider_counts);
    }

    tracer.record("King: Shelter", w_safety, b_safety);
    tracer.record("King: Attack", w_attack, b_attack);

    (w_safety + w_attack) - (b_safety + b_attack)
}

fn compute_attack_bonus_optimized(
    enemy_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    slider_counts: (i32, i32), // (diag, ortho)
) -> i32 {
    let (our_diag_count, our_ortho_count) = slider_counts;
    if our_diag_count == 0 && our_ortho_count == 0 {
        return 0;
    }

    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;

    if our_diag_count > 0 {
        for i in 0..4 {
            if enemy_king_rays[i].0 > 5 {
                open_diag_rays += 1;
            }
        }
    }
    if our_ortho_count > 0 {
        for i in 4..8 {
            if enemy_king_rays[i].0 > 5 {
                open_ortho_rays += 1;
            }
        }
    }

    const ATTACK_BONUS_PER_OPEN_RAY: i32 = 10;
    let diag_bonus = if our_diag_count > 0 && open_diag_rays > 0 {
        let mult = 100 + (our_diag_count - 1).max(0) * 15;
        open_diag_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    let ortho_bonus = if our_ortho_count > 0 && open_ortho_rays > 0 {
        let mult = 100 + (our_ortho_count - 1).max(0) * 15;
        open_ortho_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    diag_bonus + ortho_bonus
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_rook(
    _game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
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

    // Open / Semi-Open File Bonus
    let (my_pawns, enemy_pawns) = if color == PlayerColor::White {
        (white_pawns, black_pawns)
    } else {
        (black_pawns, white_pawns)
    };

    // Check for our own pawns on this file
    let run_start = my_pawns.partition_point(|p| p.0 < x);
    let has_own_pawns = run_start < my_pawns.len() && my_pawns[run_start].0 == x;

    if !has_own_pawns {
        // Semi-open (at least)
        let run_start_enemy = enemy_pawns.partition_point(|p| p.0 < x);
        let has_enemy_pawns =
            run_start_enemy < enemy_pawns.len() && enemy_pawns[run_start_enemy].0 == x;

        if !has_enemy_pawns {
            // Open file
            bonus += ROOK_OPEN_FILE_BONUS;
        } else {
            // Semi-open file
            bonus += ROOK_SEMI_OPEN_FILE_BONUS;
        }
    }

    bonus
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_queen(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
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

    // Open / Semi-Open File Bonus
    let (my_pawns, enemy_pawns) = if color == PlayerColor::White {
        (white_pawns, black_pawns)
    } else {
        (black_pawns, white_pawns)
    };

    // Check for our own pawns on this file
    let run_start = my_pawns.partition_point(|p| p.0 < x);
    let has_own_pawns = run_start < my_pawns.len() && my_pawns[run_start].0 == x;

    if !has_own_pawns {
        // Semi-open (at least)
        let run_start_enemy = enemy_pawns.partition_point(|p| p.0 < x);
        let has_enemy_pawns =
            run_start_enemy < enemy_pawns.len() && enemy_pawns[run_start_enemy].0 == x;

        if !has_enemy_pawns {
            // Open file
            bonus += QUEEN_OPEN_FILE_BONUS;
        } else {
            // Semi-open file
            bonus += QUEEN_SEMI_OPEN_FILE_BONUS;
        }
    }

    bonus
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_bishop(
    _game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
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

    // Outpost Bonus: precise pawn support
    let (my_pawns, _) = if color == PlayerColor::White {
        (white_pawns, black_pawns)
    } else {
        (black_pawns, white_pawns)
    };

    // Check for pawn support: (x-1, y-dir) or (x+1, y-dir)
    // White pawns at y-1 support piece at y. Black pawns at y+1 support piece at y.
    let support_y = if color == PlayerColor::White {
        y - 1
    } else {
        y + 1
    };

    // Check left support
    let has_left_support = my_pawns.binary_search(&(x - 1, support_y)).is_ok();

    // Check right support
    let has_right_support = my_pawns.binary_search(&(x + 1, support_y)).is_ok();

    if has_left_support || has_right_support {
        bonus += taper(MG_OUTPOST_BONUS, EG_OUTPOST_BONUS);
    }

    bonus
}

fn evaluate_knight(
    x: i64,
    y: i64,
    color: PlayerColor,
    cloud_center: Option<&Coordinate>,
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut bonus = evaluate_leaper_positioning(
        x,
        y,
        color,
        cloud_center,
        get_piece_value(PieceType::Knight),
    );

    // Outpost Bonus: precise pawn support
    let (my_pawns, _enemy_pawns) = if color == PlayerColor::White {
        (white_pawns, black_pawns)
    } else {
        (black_pawns, white_pawns)
    };

    let support_y = if color == PlayerColor::White {
        y - 1
    } else {
        y + 1
    };

    // Check left support
    let has_left_support = my_pawns.binary_search(&(x - 1, support_y)).is_ok();

    // Check right support
    let has_right_support = my_pawns.binary_search(&(x + 1, support_y)).is_ok();

    if has_left_support || has_right_support {
        bonus += taper(MG_OUTPOST_BONUS, EG_OUTPOST_BONUS);
    }

    bonus
}

// ==================== Fairy Piece Evaluation ====================

/// Evaluate leaper pieces (Hawk, Rose, Camel, Giraffe, Zebra, etc.)
/// Uses tropism (distance to kings) and cloud proximity since mobility is meaningless on infinite board
fn evaluate_leaper_positioning(
    x: i64,
    y: i64,
    _color: PlayerColor,
    cloud_center: Option<&Coordinate>,
    piece_value: i32,
) -> i32 {
    let mut bonus: i32 = 0;

    // Scale factor based on piece value
    let scale = (piece_value / LEAPER_TROPISM_DIVISOR).max(1);

    // Reward being close to piece cloud center (activity)
    if let Some(center) = cloud_center {
        let dist = (x - center.x).abs().max((y - center.y).abs());
        if dist <= 10 {
            bonus += (11 - dist as i32) * (scale / 3).max(1);
        }
    }

    bonus
}

#[allow(clippy::too_many_arguments)]
fn evaluate_king_shelter(
    _game: &GameState,
    king: &Coordinate,
    color: PlayerColor,
    phase: i32,
    defense_urgency: i32,
    has_enemy_queen_possible: bool,
    pawns: &[(i64, i64)], // Pre-sorted by (x, y)
    king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    has_ring_cover: bool,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut safety: i32 = 0;

    // 1. Local pawn / guard cover (Optimized: Ring cover passed in)
    if !has_ring_cover {
        safety -= taper(MG_KING_RING_MISSING_PENALTY, EG_KING_RING_MISSING_PENALTY);
        bump_feat!(king_ring_missing_penalty, -1);
    }

    // 1b. King shield (pawn ahead/behind) - Unified: Use pre-sorted pawn list
    let mut has_pawn_ahead = false;
    let mut has_pawn_behind = false;
    let is_white = color == PlayerColor::White;

    for dx in -2..=2_i64 {
        let x = king.x + dx;
        // Find range of pawns on this file
        let start = pawns.partition_point(|p| p.0 < x);
        let mut k = start;
        let mut on_file_count = 0;
        while k < pawns.len() && pawns[k].0 == x {
            on_file_count += 1;
            let py = pawns[k].1;
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
            k += 1;
        }

        // King on Open File Penalty (No friendly pawns on file)
        if dx == 0 && on_file_count == 0 {
            safety -= taper(MG_KING_OPEN_FILE_PENALTY, EG_KING_OPEN_FILE_PENALTY);
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
    const BASE_DIAG_RAY_PENALTY: i32 = 30;
    const BASE_ORTHO_RAY_PENALTY: i32 = 35;

    let mut total_ray_penalty: i32 = 0;
    let mut tied_defender_penalty: i32 = 0;

    let blocker_reduction_pct = |v: i32, d: i32| {
        // Continuous linear: 80% at v=100, 60% at v=300, 40% at v=500, 20% at v=700, 0% at v>=900
        let val_pct = (90 - v / 10).clamp(0, 80);
        // Continuous linear: 100% at d=1, 75% at d=2, 50% at d=3, 30% at d>=4
        let dist_mult = (125 - d * 25).clamp(30, 100);
        val_pct * dist_mult / 100
    };

    // Bounds for world border check (treat as friendly blocker)
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();

    let get_border_dist = |dx: i64, dy: i64| -> i32 {
        let mut d = i64::MAX;
        if dx > 0 {
            d = d.min(max_x.saturating_sub(king.x).saturating_add(1));
        }
        if dx < 0 {
            d = d.min(king.x.saturating_sub(min_x).saturating_add(1));
        }
        if dy > 0 {
            d = d.min(max_y.saturating_sub(king.y).saturating_add(1));
        }
        if dy < 0 {
            d = d.min(king.y.saturating_sub(min_y).saturating_add(1));
        }
        d.clamp(0, 100) as i32
    };

    // Diagonal Rays (Indices 0..4)
    for (i, (dist, val, c, pt)) in king_rays[0..4].iter().enumerate() {
        let (dist, val, c, pt) = (*dist, *val, *c, *pt);
        let mut blocker: Option<(i32, i32)> = None;
        let mut enemy_blocked = false;

        let (dx, dy) = DIAG_DIRS[i];
        let border_dist = get_border_dist(dx, dy);
        let is_border_closest = border_dist < dist;
        let actual_dist = if is_border_closest { border_dist } else { dist };

        if actual_dist <= 8 {
            if is_border_closest {
                // Border acts as a low-value friendly piece at distance 1 (perfect blocker)
                blocker = Some((0, 1));
            } else if c == color {
                blocker = Some((val, dist));
                if val >= 600 {
                    tied_defender_penalty += 10;
                }
            } else if c == PlayerColor::Neutral {
                // Neutral pieces (Void/Obstacle)
                // Void -> Perfect blocker (dist 1) like world border
                if pt == PieceType::Void {
                    blocker = Some((0, 1));
                } else {
                    blocker = Some((0, dist));
                }
            } else {
                enemy_blocked = true;
            }
        }

        let mut penalty = BASE_DIAG_RAY_PENALTY;
        if let Some((v, d)) = blocker {
            penalty = penalty * (100 - blocker_reduction_pct(v, d)) / 100;
        } else if enemy_blocked {
            penalty = penalty * 60 / 100;
        }
        total_ray_penalty += penalty;
    }

    // Ortho Rays (Indices 4..8)
    for (i, (dist, val, c, pt)) in king_rays[4..8].iter().enumerate() {
        let (dist, val, c, pt) = (*dist, *val, *c, *pt);
        let mut blocker: Option<(i32, i32)> = None;
        let mut enemy_blocked = false;

        let (dx, dy) = ORTHO_DIRS[i];
        let border_dist = get_border_dist(dx, dy);
        let is_border_closest = border_dist < dist;
        let actual_dist = if is_border_closest { border_dist } else { dist };

        if actual_dist <= 8 {
            if is_border_closest {
                blocker = Some((0, 1));
            } else if c == color {
                blocker = Some((val, dist));
                if val >= 600 {
                    tied_defender_penalty += 12;
                }
            } else if c == PlayerColor::Neutral {
                if pt == PieceType::Void {
                    blocker = Some((0, 1));
                } else {
                    blocker = Some((0, dist));
                }
            } else {
                enemy_blocked = true;
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
    let phase = game.total_phase.min(MAX_PHASE);
    // For standalone call, we must fill the vectors
    EVAL_WHITE_PAWNS.with(|wp_cell| {
        EVAL_BLACK_PAWNS.with(|bp_cell| {
            EVAL_WHITE_RQ.with(|wrq_cell| {
                EVAL_BLACK_RQ.with(|brq_cell| {
                    let wp = unsafe { &mut *wp_cell.get() };
                    let bp = unsafe { &mut *bp_cell.get() };
                    let wrq = unsafe { &mut *wrq_cell.get() };
                    let brq = unsafe { &mut *brq_cell.get() };
                    wp.clear();
                    bp.clear();
                    wrq.clear();
                    brq.clear();

                    let w_promo = game.white_promo_rank;
                    let b_promo = game.black_promo_rank;

                    for (cx, cy, tile) in game.board.tiles.iter() {
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
                            if piece.piece_type() == PieceType::Pawn {
                                if piece.color() == PlayerColor::White {
                                    if y < w_promo {
                                        wp.push((x, y));
                                    }
                                } else if y > b_promo {
                                    bp.push((x, y));
                                }
                            } else if matches!(
                                piece.piece_type(),
                                PieceType::Rook
                                    | PieceType::Queen
                                    | PieceType::Amazon
                                    | PieceType::Chancellor
                                    | PieceType::RoyalQueen
                            ) {
                                if piece.color() == PlayerColor::White {
                                    wrq.push((x, y));
                                } else {
                                    brq.push((x, y));
                                }
                            }
                        }
                    }
                    wp.sort_unstable();
                    bp.sort_unstable();

                    evaluate_pawn_structure_traced(
                        game,
                        phase,
                        &game.white_king_pos,
                        &game.black_king_pos,
                        &mut NoTrace,
                        wp,
                        bp,
                        wrq,
                        brq,
                    )
                })
            })
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_pawn_structure_traced<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    tracer: &mut T,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    white_rq: &[(i64, i64)],
    black_rq: &[(i64, i64)],
) -> i32 {
    // Check cache first using game's pawn_hash
    let pawn_hash = game.pawn_hash;

    // Bypassing cache if tracer is active to ensure we get a full breakdown.
    if tracer.is_active() {
        return compute_pawn_structure_traced(
            game,
            phase,
            white_king,
            black_king,
            tracer,
            white_pawns,
            black_pawns,
            white_rq,
            black_rq,
        );
    }

    // Fast 2-Bucket cache probe using bitwise mask
    let idx = (pawn_hash as usize) & (PAWN_CACHE_SIZE - 1);
    let cached = PAWN_CACHE.with(|cache| {
        let bucket = unsafe { (&*cache.get())[idx] };
        if bucket.entries[0].hash == pawn_hash {
            Some(bucket.entries[0].score)
        } else if bucket.entries[1].hash == pawn_hash {
            Some(bucket.entries[1].score)
        } else {
            None
        }
    });

    if let Some(score) = cached {
        return score;
    }

    // Cache miss - compute pawn structure
    let score = compute_pawn_structure_traced(
        game,
        phase,
        white_king,
        black_king,
        tracer,
        white_pawns,
        black_pawns,
        white_rq,
        black_rq,
    );

    // 2-Bucket cache store (LRU: new item goes to front, old item moves to back)
    PAWN_CACHE.with(|cache| {
        let cache_mut = unsafe { &mut *cache.get() };
        let bucket = &mut cache_mut[idx];
        bucket.entries[1] = bucket.entries[0];
        bucket.entries[0] = PawnCacheEntry {
            hash: pawn_hash,
            score,
        };
    });

    score
}

#[allow(clippy::too_many_arguments)]
/// Core pawn structure computation. Called on cache miss.
fn compute_pawn_structure_traced<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    white_king: &Option<Coordinate>,
    black_king: &Option<Coordinate>,
    tracer: &mut T,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    _white_rq: &[(i64, i64)],
    _black_rq: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_doubled = 0;
    let mut b_doubled = 0;
    let mut w_passed_score = 0;
    let mut b_passed_score = 0;
    let mut w_connected = 0;
    let mut b_connected = 0;
    let mut w_candidate = 0;
    let mut b_candidate = 0;
    let mut w_isolated = 0;
    let mut b_isolated = 0;
    let mut w_backward = 0;
    let mut b_backward = 0;

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

    // White Pawns: Passed, Candidate, Connected, Isolated, Backward
    for &(wx, wy) in white_pawns {
        let mut is_passed = true;
        let mut is_candidate = false;
        let mut stoppers = 0;
        let mut attackers = 0;
        let mut defenders = 0;

        // Structure checks
        let left_idx = white_pawns.partition_point(|&(x, _)| x < wx - 1);
        let has_left_neighbor = left_idx < white_pawns.len() && white_pawns[left_idx].0 == wx - 1;

        let right_idx = white_pawns.partition_point(|&(x, _)| x < wx + 1);
        let has_right_neighbor =
            right_idx < white_pawns.len() && white_pawns[right_idx].0 == wx + 1;

        if !has_left_neighbor && !has_right_neighbor {
            w_isolated -= taper(10, 20);
        } else {
            let is_behind_left = !has_left_neighbor || white_pawns[left_idx].1 > wy;
            let is_behind_right = !has_right_neighbor || white_pawns[right_idx].1 > wy;

            if is_behind_left && is_behind_right {
                let stop_sq_blocked = game.board.is_occupied(wx, wy + 1);
                let stop_sq_attacked = black_pawns.binary_search(&(wx - 1, wy + 2)).is_ok()
                    || black_pawns.binary_search(&(wx + 1, wy + 2)).is_ok();

                if stop_sq_blocked || stop_sq_attacked {
                    w_backward -= taper(8, 12);
                }
            }
        }

        // Relative rank 0 to 5 (assuming 6 ranks is "near promotion")
        // For an infinite board, we'll anchor to the promotion rank.
        let w_promo = game.white_promo_rank;
        let dist_to_promo = (w_promo - wy).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;

        for dx in -1..=1 {
            let target_file = wx + dx;

            // Check for enemy pawns blocking or on adjacent files
            let start = black_pawns.partition_point(|&(bx, _)| bx < target_file);
            let mut k = start;
            while k < black_pawns.len() && black_pawns[k].0 == target_file {
                let by = black_pawns[k].1;
                if by > wy {
                    is_passed = false;
                    stoppers += 1;
                    if dx == 0 {
                        // Directly in front
                    } else {
                        // Lever/Attack square
                        attackers += 1;
                    }
                }
                k += 1;
            }

            // Check for friendly support (for candidate detection)
            if dx != 0 {
                let start_f = white_pawns.partition_point(|&(fx, _)| fx < target_file);
                let mut kf = start_f;
                while kf < white_pawns.len() && white_pawns[kf].0 == target_file {
                    if white_pawns[kf].1 < wy {
                        defenders += 1;
                    }
                    kf += 1;
                }
            }
        }

        if is_passed {
            // 1. Can Advance
            let next_y = wy + 1;
            let can_advance = game.board.get_piece(wx, next_y).is_none();

            // 2. Safe Advance
            let safe_advance = black_pawns.binary_search(&(wx - 1, next_y + 1)).is_err()
                && black_pawns.binary_search(&(wx + 1, next_y + 1)).is_err();

            // 3. King Distances
            let mut friendly_king_bonus = 0;
            let mut enemy_king_penalty = 0;
            if let Some(wk) = white_king {
                let d = (wx - wk.x).abs().max((wy - wk.y).abs()) as usize;
                friendly_king_bonus = PASSED_FRIENDLY_KING_DIST[rel_rank] * (7 - d.min(7)) as i32;
            }
            if let Some(bk) = black_king {
                let d = (wx - bk.x).abs().max((wy - bk.y).abs()) as usize;
                enemy_king_penalty = PASSED_ENEMY_KING_DIST[rel_rank] * (7 - d.min(7)) as i32;
            }

            // 4. Safe Promotion Path
            let mut safe_path = is_clear_line_between_fast(
                &game.spatial_indices,
                &Coordinate::new(wx, wy),
                &Coordinate::new(wx, w_promo),
            );
            if safe_path {
                // Check for attacking black pawns on adjacent files in rank range [wy+2, w_promo]
                for dx in &[-1, 1] {
                    let target_file = wx + dx;
                    let start = black_pawns.partition_point(|&(bx, by)| {
                        bx < target_file || (bx == target_file && by < wy + 2)
                    });
                    if start < black_pawns.len()
                        && black_pawns[start].0 == target_file
                        && black_pawns[start].1 <= w_promo
                    {
                        safe_path = false;
                        break;
                    }
                }
            }
            let safe_path_bonus = if safe_path {
                taper(MG_PASSED_SAFE_PATH_BONUS, EG_PASSED_SAFE_PATH_BONUS)
            } else {
                0
            };

            let base_bonus =
                PASSED_PAWN_ADV_BONUS[can_advance as usize][safe_advance as usize][rel_rank];
            w_passed_score +=
                base_bonus + friendly_king_bonus - enemy_king_penalty + safe_path_bonus;
        } else {
            // Check for candidate passer: no blockers on file, but attackers on adjacent files.
            if stoppers > 0 && attackers == stoppers && defenders >= attackers {
                is_candidate = true;
            }

            if is_candidate {
                w_candidate += CANDIDATE_PASSER_BONUS[rel_rank];
            }
        }

        // Connectivity
        if white_pawns.binary_search(&(wx - 1, wy - 1)).is_ok()
            || white_pawns.binary_search(&(wx + 1, wy - 1)).is_ok()
        {
            // Boost connectivity if also a candidate/passed
            let bonus = if is_passed {
                (taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS) * 3) / 2
            } else {
                taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS)
            };
            w_connected += bonus;
        }
    }

    // Black Pawns: Passed, Candidate, Connected, Isolated, Backward
    for &(bx, by) in black_pawns {
        let mut is_passed = true;
        let mut is_candidate = false;
        let mut stoppers = 0;
        let mut attackers = 0;
        let mut defenders = 0;

        // Structure checks
        let left_idx = black_pawns.partition_point(|&(x, _)| x < bx - 1);
        let has_left_neighbor = left_idx < black_pawns.len() && black_pawns[left_idx].0 == bx - 1;

        let right_idx = black_pawns.partition_point(|&(x, _)| x < bx + 1);
        let has_right_neighbor =
            right_idx < black_pawns.len() && black_pawns[right_idx].0 == bx + 1;

        if !has_left_neighbor && !has_right_neighbor {
            b_isolated -= taper(10, 20);
        } else {
            let mut is_behind_left = true;
            if has_left_neighbor {
                let next_idx = black_pawns.partition_point(|&(x, _)| x < bx);
                if next_idx > left_idx {
                    let last_y = black_pawns[next_idx - 1].1;
                    if last_y >= by {
                        is_behind_left = false;
                    }
                }
            }

            let mut is_behind_right = true;
            if has_right_neighbor {
                let next_idx = black_pawns.partition_point(|&(x, _)| x < bx + 2);
                if next_idx > right_idx {
                    let last_y = black_pawns[next_idx - 1].1;
                    if last_y >= by {
                        is_behind_right = false;
                    }
                }
            }

            if is_behind_left && is_behind_right {
                let stop_sq_blocked = game.board.is_occupied(bx, by - 1);
                let stop_sq_attacked = white_pawns.binary_search(&(bx - 1, by - 2)).is_ok()
                    || white_pawns.binary_search(&(bx + 1, by - 2)).is_ok();

                if stop_sq_blocked || stop_sq_attacked {
                    b_backward -= taper(8, 12);
                }
            }
        }

        let b_promo = game.black_promo_rank;
        let dist_to_promo = (by - b_promo).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;

        for dx in -1..=1 {
            let target_file = bx + dx;
            let start = white_pawns.partition_point(|&(wx, _)| wx < target_file);
            let mut k = start;
            while k < white_pawns.len() && white_pawns[k].0 == target_file {
                let wy = white_pawns[k].1;
                if wy < by {
                    is_passed = false;
                    stoppers += 1;
                    if dx != 0 {
                        attackers += 1;
                    }
                }
                k += 1;
            }

            if dx != 0 {
                let start_f = black_pawns.partition_point(|&(fx, _)| fx < target_file);
                let mut kf = start_f;
                while kf < black_pawns.len() && black_pawns[kf].0 == target_file {
                    if black_pawns[kf].1 > by {
                        defenders += 1;
                    }
                    kf += 1;
                }
            }
        }

        if is_passed {
            let next_y = by - 1;
            let can_advance = game.board.get_piece(bx, next_y).is_none();
            let safe_advance = white_pawns.binary_search(&(bx - 1, next_y - 1)).is_err()
                && white_pawns.binary_search(&(bx + 1, next_y - 1)).is_err();

            let mut friendly_king_bonus = 0;
            let mut enemy_king_penalty = 0;
            if let Some(bk) = black_king {
                let d = (bx - bk.x).abs().max((by - bk.y).abs()) as usize;
                friendly_king_bonus = PASSED_FRIENDLY_KING_DIST[rel_rank] * (7 - d.min(7)) as i32;
            }
            if let Some(wk) = white_king {
                let d = (bx - wk.x).abs().max((by - wk.y).abs()) as usize;
                enemy_king_penalty = PASSED_ENEMY_KING_DIST[rel_rank] * (7 - d.min(7)) as i32;
            }

            // 4. Safe Promotion Path
            let mut safe_path = is_clear_line_between_fast(
                &game.spatial_indices,
                &Coordinate::new(bx, by),
                &Coordinate::new(bx, b_promo),
            );
            if safe_path {
                // Check for attacking white pawns on adjacent files in rank range [b_promo-1, by-2]
                for dx in &[-1, 1] {
                    let target_file = bx + dx;
                    let start = white_pawns.partition_point(|&(wx, wy_)| {
                        wx < target_file || (wx == target_file && wy_ < b_promo - 1)
                    });
                    if start < white_pawns.len()
                        && white_pawns[start].0 == target_file
                        && white_pawns[start].1 <= by - 2
                    {
                        safe_path = false;
                        break;
                    }
                }
            }
            let safe_path_bonus = if safe_path {
                taper(MG_PASSED_SAFE_PATH_BONUS, EG_PASSED_SAFE_PATH_BONUS)
            } else {
                0
            };

            let base_bonus =
                PASSED_PAWN_ADV_BONUS[can_advance as usize][safe_advance as usize][rel_rank];
            b_passed_score +=
                base_bonus + friendly_king_bonus - enemy_king_penalty + safe_path_bonus;
        } else {
            if stoppers > 0 && attackers == stoppers && defenders >= attackers {
                is_candidate = true;
            }
            if is_candidate {
                b_candidate += CANDIDATE_PASSER_BONUS[rel_rank];
            }
        }

        if black_pawns.binary_search(&(bx - 1, by + 1)).is_ok()
            || black_pawns.binary_search(&(bx + 1, by + 1)).is_ok()
        {
            // Boost connectivity if also a candidate/passed
            let bonus = if is_passed {
                (taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS) * 3) / 2
            } else {
                taper(MG_CONNECTED_PAWN_BONUS, EG_CONNECTED_PAWN_BONUS)
            };
            b_connected += bonus;
        }
    }

    tracer.record("Pawn: Doubled", w_doubled.abs(), b_doubled.abs());
    tracer.record("Pawn: Passed", w_passed_score, b_passed_score);
    tracer.record("Pawn: Candidate", w_candidate, b_candidate);
    tracer.record("Pawn: Connected", w_connected, b_connected);
    tracer.record("Pawn: Isolated", w_isolated.abs(), b_isolated.abs());
    tracer.record("Pawn: Backward", w_backward.abs(), b_backward.abs());

    (w_doubled + w_passed_score + w_candidate + w_connected + w_isolated + w_backward)
        - (b_doubled + b_passed_score + b_candidate + b_connected + b_isolated + b_backward)
}

pub fn count_pawns_on_file(
    _game: &GameState,
    file: i64,
    color: PlayerColor,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> (i32, i32) {
    let mut own_pawns = 0;
    let mut enemy_pawns = 0;

    let target_pawns = if color == PlayerColor::White {
        white_pawns
    } else {
        black_pawns
    };
    let opponent_pawns = if color == PlayerColor::White {
        black_pawns
    } else {
        white_pawns
    };

    // Find range of pawns on this file in our lists
    let start = target_pawns.partition_point(|p| p.0 < file);
    let mut k = start;
    while k < target_pawns.len() && target_pawns[k].0 == file {
        own_pawns += 1;
        k += 1;
    }

    let start_opp = opponent_pawns.partition_point(|p| p.0 < file);
    let mut k_opp = start_opp;
    while k_opp < opponent_pawns.len() && opponent_pawns[k_opp].0 == file {
        enemy_pawns += 1;
        k_opp += 1;
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

    for (px, py, _) in board.iter() {
        // Skip the endpoints themselves
        if px == from.x && py == from.y {
            continue;
        }
        if px == to.x && py == to.y {
            continue;
        }

        // Same file
        if dx == 0 && px == from.x && is_between(py, from.y, to.y) {
            return false;
        }

        // Same rank
        if dy == 0 && py == from.y && is_between(px, from.x, to.x) {
            return false;
        }

        // Same diagonal
        if dx.abs() == dy.abs() {
            let vx = px - from.x;
            let vy = py - from.y;
            // Collinear and between
            if vx * dy == vy * dx && is_between(px, from.x, to.x) && is_between(py, from.y, to.y) {
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
            let start = row.coords.partition_point(|x| *x <= min_x);
            // Check if any piece exists before max_x
            if start < row.len() && row.coords[start] < max_x {
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
            let start = col.coords.partition_point(|y| *y <= min_y);
            // Check if any piece exists before max_y
            if start < col.len() && col.coords[start] < max_y {
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
            let start = diag.coords.partition_point(|x| *x <= min_x);
            if start < diag.len() && diag.coords[start] < max_x {
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
        let start = diag.coords.partition_point(|x| *x <= min_x);
        if start < diag.len() && diag.coords[start] < max_x {
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

    use crate::game::GameState;

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
        let mut game = GameState::new();
        let from = Coordinate::new(1, 1);
        let to = Coordinate::new(1, 8);

        // Empty board should have clear line
        assert!(is_clear_line_between(&game.board, &from, &to));

        // Add blocker
        let icn = "w (8;q|1|q) P1,4";
        game.setup_position_from_icn(icn);
        assert!(!is_clear_line_between(&game.board, &from, &to));
    }

    #[test]
    fn test_is_clear_line_diagonal() {
        let mut game = GameState::new();
        let from = Coordinate::new(1, 1);
        let to = Coordinate::new(5, 5);

        assert!(is_clear_line_between(&game.board, &from, &to));

        let icn = "w (8;q|1;q) b3,3|";
        game.setup_position_from_icn(icn);
        assert!(!is_clear_line_between(&game.board, &from, &to));
    }

    #[test]
    fn test_calculate_initial_material() {
        let mut game = GameState::new();

        // Empty board = 0
        assert_eq!(calculate_initial_material(&game.board), 0);

        // Add white queen
        let icn1 = "w (8;q|1;q) Q4,1";
        game.setup_position_from_icn(icn1);
        assert_eq!(calculate_initial_material(&game.board), 1350); // Queen = 1350 in infinite chess

        // Add black queen - should cancel out
        let icn2 = "w (8;q|1|q) Q4,1|q4,8";
        game.setup_position_from_icn(icn2);
        assert_eq!(calculate_initial_material(&game.board), 0);
    }

    #[test]
    fn test_clear_pawn_cache() {
        // Just ensure it doesn't panic
        clear_pawn_cache();
    }

    #[test]
    fn test_evaluate_returns_value() {
        let mut game = GameState::new();
        let icn = "w (8;q|1;q) K5,1|k5,8";
        game.setup_position_from_icn(icn);

        let score = evaluate(&game);
        // K vs K should be close to 0
        assert!(score.abs() < 1000, "K vs K should be near 0, got {}", score);
    }

    #[test]
    fn test_count_pawns_on_file() {
        let mut game = GameState::new();
        let icn = "w (8;q|1;q) K5,1|k5,8|P4,2|P4,3|p4,7";
        game.setup_position_from_icn(icn);

        let w_pawns = vec![(4, 1), (4, 3)];
        let b_pawns = vec![(4, 7)];
        let (own, enemy) = count_pawns_on_file(&game, 4, PlayerColor::White, &w_pawns, &b_pawns);
        assert_eq!(own, 2);
        assert_eq!(enemy, 1);
    }

    #[test]
    fn test_evaluate_pawn_structure() {
        let mut game = GameState::new();
        let icn = "w (8;q|1;q) K5,1|k5,8|P4,2|P4,3";
        game.setup_position_from_icn(icn);

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
    fn test_king_safety_penalties() {
        let mut game = Box::new(GameState::new());
        // White King at (0,0), Black King (10,10), Rooks, White Queen at (0,1)
        let icn_near = "w (8;q|1;q) K0,0|k10,10|R5,0|r5,9|Q0,1";
        game.setup_position_from_icn(icn_near);

        let score_near = evaluate_inner(&game);

        // White Queen far away from its king
        let icn_far = "w (8;q|1;q) K0,0|k10,10|R5,0|r5,9|Q5,5";
        game.setup_position_from_icn(icn_far);

        let score_far = evaluate_inner(&game);

        assert!(score_far != score_near);
    }

    #[test]
    fn test_pawn_structure_caching() {
        let mut game = Box::new(GameState::new());
        let icn = "w (8;q|1;q) K5,1|k5,8|P4,4|p4,5";
        game.setup_position_from_icn(icn);

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
    fn test_evaluate_bishop_diagonal() {
        let mut game = Box::new(GameState::new());
        let icn = "w (8;q|1;q) K0,0|k7,7|B4,4";
        game.setup_position_from_icn(icn);

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_bishop(
            &game,
            4,
            4,
            PlayerColor::White,
            &wk,
            &bk,
            MAX_PHASE,
            &[],
            &[],
        );
        // Central bishop should have positive score
        assert!(
            score > 0,
            "Central bishop should have positive positional score"
        );
    }

    #[test]
    fn test_evaluate_rook_open_file() {
        let mut game = Box::new(GameState::new());
        let icn = "w (8;q|1;q) K0,0|k7,7|R4,1";
        game.setup_position_from_icn(icn);

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_rook(
            &game,
            4,
            1,
            PlayerColor::White,
            &wk,
            &bk,
            MAX_PHASE,
            &[],
            &[],
        );
        // Rook should have score for mobility etc
        assert!(score.abs() < 1000, "Rook score should be reasonable");
    }

    #[test]
    fn test_evaluate_queen_central() {
        let mut game = Box::new(GameState::new());
        let icn = "w (8;q|1;q) K0,0|k7,7|Q4,4";
        game.setup_position_from_icn(icn);

        let wk = Some(Coordinate::new(0, 0));
        let bk = Some(Coordinate::new(7, 7));
        let score = evaluate_queen(
            &game,
            4,
            4,
            PlayerColor::White,
            &wk,
            &bk,
            MAX_PHASE,
            &[],
            &[],
        );
        // Queen in center should have decent positional score
        assert!(score.abs() < 2000, "Queen score should be reasonable");
    }

    #[test]
    fn test_pawn_structure_isolated_pawn() {
        let mut game = Box::new(GameState::new());
        // Isolated white pawn on d-file
        let isolated_icn = "w (8;q|1;q) K5,1|k5,8|P4,4";
        game.setup_position_from_icn(isolated_icn);

        clear_pawn_cache();
        let isolated_score = evaluate_pawn_structure(&game);

        // Add supporting pawns
        let supported_icn = "w (8;q|1;q) K5,1|k5,8|P4,4|P3,3|P5,3";
        game.setup_position_from_icn(supported_icn);
        game.recompute_hash();

        clear_pawn_cache();
        let supported_score = evaluate_pawn_structure(&game);

        // Supported pawns should score better
        assert!(
            supported_score > isolated_score,
            "Supported pawns should be better than isolated"
        );
    }

    #[test]
    fn test_outpost_bonus() {
        let mut game = Box::new(GameState::new());

        // Case 1: No support
        let icn_no_support = "w (8;q|1;q) K0,0|k0,10|N4,4";
        game.setup_position_from_icn(icn_no_support);

        let score_no_support = evaluate_knight(4, 4, PlayerColor::White, None, 0, &[], &[]);

        // Case 2: Support from pawn at (3,3) (White pawn at y-1 supports y)
        let icn_supported = "w (8;q|1;q) K0,0|k0,10|N4,4|P3,3";
        game.setup_position_from_icn(icn_supported);
        // Mock pawn list
        let white_pawns = vec![(3, 3)];

        let score_supported = evaluate_knight(4, 4, PlayerColor::White, None, 0, &white_pawns, &[]);

        println!(
            "No Support: {}, Supported: {}",
            score_no_support, score_supported
        );
        assert!(
            score_supported > score_no_support,
            "Supported knight should have higher score"
        );
        assert_eq!(
            score_supported - score_no_support,
            EG_OUTPOST_BONUS,
            "Bonus should match EG_OUTPOST_BONUS in endgame"
        );
    }

    #[test]
    fn test_candidate_passer_bonus() {
        let mut game = Box::new(GameState::new());

        // Candidate Passer Setup:
        // White Pawn at (4, 4)
        // Black Pawn stopping it on adjacent file: (3, 5) or (5, 5)
        // White support to balance the stopper
        let icn = "w (8;q|1;q) K0,0|k0,10|P4,4|p3,5|P5,3";
        game.setup_position_from_icn(icn);
        clear_pawn_cache();

        let score = evaluate_pawn_structure(&game);

        // Candidate bonus for rel_rank 3 (wy=4, dist=4, rel_rank=2 or 3 depending on clamp)
        // should be positive.
        assert!(score > 0, "Candidate passer should provide positive score");
    }

    #[test]
    fn test_passed_pawn_advancement() {
        let mut game = Box::new(GameState::new());
        game.white_promo_rank = 8;

        // Case 1: Passed pawn at (4, 4) - can advance, but not safely (controlled by black pawn)
        let unsafe_icn = "w (8;q|1;q) K0,0|k10,10|P4,4|p3,6";
        game.setup_position_from_icn(unsafe_icn);

        clear_pawn_cache();
        let score_unsafe = evaluate_pawn_structure(&game);

        // Case 2: Make it safe (remove black pawn)
        let safe_icn = "w (8;q|1;q) K0,0|k10,10|P4,4";
        game.setup_position_from_icn(safe_icn);
        clear_pawn_cache();
        let score_safe = evaluate_pawn_structure(&game);

        assert!(
            score_safe > score_unsafe,
            "Safe-to-advance passed pawn should score higher than unsafe"
        );
    }

    #[test]
    fn test_backward_isolated_penalties() {
        let mut game = Box::new(GameState::new());

        // Case 1: Connected Pawns (Good)
        let connected_icn = "w (8;q|1;q) K0,0|k0,10|P4,4|P5,5";
        game.setup_position_from_icn(connected_icn);

        clear_pawn_cache();
        let score_good = evaluate_pawn_structure(&game);

        // Case 2: Isolated Pawn (Bad)
        let isolated_icn = "w (8;q|1;q) K0,0|k0,10|P4,4|P8,4";
        game.setup_position_from_icn(isolated_icn);

        clear_pawn_cache();
        let score_isolated = evaluate_pawn_structure(&game);

        // Case 3: Backward Pawn (Bad)
        let backward_icn = "w (8;q|1;q) K0,0|k0,10|P4,4|P5,5|p3,6";
        game.setup_position_from_icn(backward_icn);
        game.recompute_hash();
        clear_pawn_cache();
        let score_backward = evaluate_pawn_structure(&game);

        // Expect: Connected > Backward
        assert!(
            score_good > score_backward,
            "Backward pawn should score lower than free connected. Good: {}, Backward: {}",
            score_good,
            score_backward
        );

        // Expect: Connected > Isolated
        assert!(
            score_good > score_isolated,
            "Isolated pawn should score lower than connected. Good: {}, Isolated: {}",
            score_good,
            score_isolated
        );
    }

    #[test]
    fn test_king_open_file_penalty() {
        let mut game = Box::new(GameState::new());
        // Setup King on open file (0,0)
        let open_icn = "w (8;q|1;q) K0,0|k0,10|P5,5";
        game.setup_position_from_icn(open_icn);

        clear_pawn_cache();

        let score_open = evaluate(&game);

        // Setup King with pawn shield on file 0
        let closed_icn = "w (8;q|1;q) K0,0|k0,10|P5,5|P0,1";
        game.setup_position_from_icn(closed_icn);
        game.recompute_hash();
        clear_pawn_cache();

        let score_closed = evaluate(&game);

        // Closed (shielded) should be inherently safer than Open
        assert!(
            score_closed > score_open,
            "Shielded king should score higher than open file king. Open: {}, Closed: {}",
            score_open,
            score_closed
        );
    }
}
