use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::{GameState, WinCondition};

use smallvec::SmallVec;
use std::cell::{Cell, UnsafeCell};

use super::piece_reach::{
    evaluate_compound_leap_threats, evaluate_huygen_reach, evaluate_knightrider_reach,
    evaluate_rose_reach,
};
use crate::search::params::{
    amazon_compound_bonus, amazon_queen_scale, amazon_rook_scale, archbishop,
    archbishop_bishop_scale, bishop, camel, candidate_passer_bonus, centaur, centaur_guard_scale,
    chancellor_bonus, chancellor_rook_scale, cloud_center_max_skew_dist,
    cloud_penalty_per_100_value, complexity_damp, complexity_excess_max, eg_bishop_pair_bonus,
    eg_doubled_pawn_penalty, eg_far_slider_penalty_mult, eg_king_pawn_ahead_penalty,
    eg_outpost_bonus, far_queen_penalty, far_rook_penalty, far_slider_cheb_max_excess,
    far_slider_cheb_radius, giraffe, guard, hawk, huygen, king_defender_value_threshold,
    king_shield_ahead_max_dist, knight, knightrider, leaper_tropism_divisor, mg_bishop_pair_bonus,
    mg_doubled_pawn_penalty, mg_far_slider_penalty_mult, mg_king_pawn_ahead_penalty,
    mg_outpost_bonus, min_major_development_penalty,
    minor_development_penalty_threshold, passed_enemy_king_dist, passed_friendly_king_dist,
    passed_pawn_adv_bonus, pawn, pawn_enemy_king_dist, pawn_far_from_promo_max_penalty,
    pawn_friendly_king_dist, pawn_full_value_threshold, pawn_past_promo_penalty,
    piece_cloud_cheb_max_excess, piece_cloud_cheb_radius, queen_ideal_line_dist,
    queen_open_file_bonus, queen_semi_open_file_bonus, queen_value, rook, rook_open_file_bonus,
    rook_semi_open_file_bonus, rose, slider_axis_wiggle, slider_net_bonus, slider_threat_cap,
    slider_threat_div, zebra,
min_fairy_development_penalty,
};

// 2-Bucket LRU pawn structure cache
const PAWN_CACHE_SIZE: usize = 16384; // 16384 buckets * 2 entries = 32768 entries

// Caches pawn-hash-pure terms as untapered (mg, eg) plus the passed-pawn coordinate
// lists; passed-pawn scoring (king distance, blockers, path) is recomputed live.
#[derive(Clone)]
struct PawnCacheEntry {
    hash: u64,
    mg: i32,
    eg: i32,
    w_passed: SmallVec<[(i64, i64); 4]>,
    b_passed: SmallVec<[(i64, i64); 4]>,
}

impl Default for PawnCacheEntry {
    fn default() -> Self {
        PawnCacheEntry {
            hash: u64::MAX,
            mg: 0,
            eg: 0,
            w_passed: SmallVec::new(),
            b_passed: SmallVec::new(),
        }
    }
}

#[derive(Clone, Default)]
struct PawnCacheBucket {
    entries: [PawnCacheEntry; 2],
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

/// Per-level play-style weighting, in percent of the full-strength term. Damping
/// attack and amplifying defense makes a weak level misjudge the position instead
/// of misplaying a correct ranking. `NEUTRAL` is full strength, and only the
/// generic evaluation honours it — `variants/` evaluators are unscaled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EvalStyle {
    pub attack_scale: i32,
    pub defense_scale: i32,
}

impl EvalStyle {
    pub const NEUTRAL: Self = Self {
        attack_scale: 100,
        defense_scale: 100,
    };

    /// Scale a term that rewards generating pressure on the enemy king.
    #[inline]
    pub fn attack(&self, term: i32) -> i32 {
        if self.attack_scale == 100 {
            term
        } else {
            term * self.attack_scale / 100
        }
    }

    /// Scale a term that rewards keeping material home and the king sheltered.
    #[inline]
    pub fn defense(&self, term: i32) -> i32 {
        if self.defense_scale == 100 {
            term
        } else {
            term * self.defense_scale / 100
        }
    }
}

thread_local! {
    static EVAL_STYLE: Cell<EvalStyle> = const { Cell::new(EvalStyle::NEUTRAL) };
}

/// The style in force on this thread. Read once per evaluation, not per term.
#[inline]
pub fn eval_style() -> EvalStyle {
    EVAL_STYLE.with(|style| style.get())
}

/// Install a style for this thread. The skill limiter owns this and restores
/// `NEUTRAL` when its search ends; nothing else may leave it set.
#[inline]
pub fn set_eval_style(style: EvalStyle) {
    EVAL_STYLE.with(|cell| cell.set(style));
}

/// Clear the pawn structure cache.
pub fn clear_pawn_cache() {
    PAWN_CACHE.with(|cache| {
        // Fast clear using fill
        unsafe { (&mut *cache.get()).fill(PawnCacheBucket::default()) };
    });
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
use once_cell::sync::Lazy;
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
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

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
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

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub static EVAL_FEATURES: Lazy<RwLock<EvalFeatures>> =
    Lazy::new(|| RwLock::new(EvalFeatures::default()));

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn reset_eval_features() {
    if let Ok(mut guard) = EVAL_FEATURES.write() {
        *guard = EvalFeatures::default();
    }
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn snapshot_eval_features() -> EvalFeatures {
    EVAL_FEATURES.read().map(|g| g.clone()).unwrap_or_default()
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
macro_rules! bump_feat {
    ($field:ident, $amount:expr) => {{
        if let Ok(mut f) = $crate::evaluation::EVAL_FEATURES.write() {
            f.$field += $amount;
        }
    }};
}

#[cfg(not(any(feature = "param_tuning", feature = "eval_tuning")))]
macro_rules! bump_feat {
    ($($tt:tt)*) => {};
}

pub const DEFAULT_EVAL_PAWN: i32 = 100;
pub const DEFAULT_EVAL_KNIGHT: i32 = 255;
pub const DEFAULT_EVAL_BISHOP: i32 = 434;
pub const DEFAULT_EVAL_ROOK: i32 = 646;
pub const DEFAULT_EVAL_GUARD: i32 = 180;
pub const DEFAULT_EVAL_CENTAUR: i32 = 566;
pub const DEFAULT_EVAL_COMPOUND_BONUS: i32 = 46;
pub const DEFAULT_EVAL_CAMEL: i32 = 270;
pub const DEFAULT_EVAL_GIRAFFE: i32 = 268;
pub const DEFAULT_EVAL_ZEBRA: i32 = 272;
pub const DEFAULT_EVAL_KNIGHTRIDER: i32 = 720;
pub const DEFAULT_EVAL_HAWK: i32 = 540;
pub const DEFAULT_EVAL_ARCHBISHOP: i32 = 1060;
pub const DEFAULT_EVAL_ROSE: i32 = 997;
pub const DEFAULT_EVAL_HUYGEN: i32 = 330;
pub const DEFAULT_EVAL_CHANCELLOR_BONUS: i32 = 245;
/// Amazon was the only compound priced at the bare sum of its parts, while the
/// chancellor carries +245 over rook+knight and the archbishop +371.
pub const DEFAULT_EVAL_MG_DOUBLED_PAWN_PENALTY: i32 = 10;
pub const DEFAULT_EVAL_EG_DOUBLED_PAWN_PENALTY: i32 = 15;
pub const DEFAULT_EVAL_MG_BISHOP_PAIR_BONUS: i32 = 57;
pub const DEFAULT_EVAL_EG_BISHOP_PAIR_BONUS: i32 = 101;
pub const DEFAULT_EVAL_ROOK_OPEN_FILE_BONUS: i32 = 57;
pub const DEFAULT_EVAL_ROOK_SEMI_OPEN_FILE_BONUS: i32 = 29;
pub const DEFAULT_EVAL_QUEEN_OPEN_FILE_BONUS: i32 = 33;
pub const DEFAULT_EVAL_QUEEN_SEMI_OPEN_FILE_BONUS: i32 = 19;
pub const DEFAULT_EVAL_MG_OUTPOST_BONUS: i32 = 33;
pub const DEFAULT_EVAL_EG_OUTPOST_BONUS: i32 = 56;
pub const DEFAULT_EVAL_AMAZON_COMPOUND_BONUS: i32 = 200;
pub const DEFAULT_EVAL_SLIDER_NET_BONUS: i32 = 21;
pub const DEFAULT_EVAL_FAR_SLIDER_CHEB_RADIUS: i32 = 18;
pub const DEFAULT_EVAL_FAR_SLIDER_CHEB_MAX_EXCESS: i32 = 40;
pub const DEFAULT_EVAL_FAR_QUEEN_PENALTY: i32 = 5;
pub const DEFAULT_EVAL_FAR_ROOK_PENALTY: i32 = 7;
pub const DEFAULT_EVAL_PIECE_CLOUD_CHEB_RADIUS: i32 = 16;
pub const DEFAULT_EVAL_SLIDER_AXIS_WIGGLE: i32 = 5;
pub const DEFAULT_EVAL_PIECE_CLOUD_CHEB_MAX_EXCESS: i32 = 64;
pub const DEFAULT_EVAL_CLOUD_PENALTY_PER_100_VALUE: i32 = 2;
pub const DEFAULT_EVAL_CLOUD_CENTER_MAX_SKEW_DIST: i32 = 16;
pub const DEFAULT_EVAL_QUEEN_IDEAL_LINE_DIST: i32 = 4;
pub const DEFAULT_EVAL_LEAPER_TROPISM_DIVISOR: i32 = 400;
pub const DEFAULT_EVAL_CHANCELLOR_ROOK_SCALE: i32 = 90;
pub const DEFAULT_EVAL_ARCHBISHOP_BISHOP_SCALE: i32 = 90;
pub const DEFAULT_EVAL_AMAZON_ROOK_SCALE: i32 = 50;
pub const DEFAULT_EVAL_AMAZON_QUEEN_SCALE: i32 = 70;
pub const DEFAULT_EVAL_CENTAUR_GUARD_SCALE: i32 = 50;
pub const DEFAULT_EVAL_PAWN_FULL_VALUE_THRESHOLD: i32 = 6;
pub const DEFAULT_EVAL_PAWN_PAST_PROMO_PENALTY: i32 = 90;
pub const DEFAULT_EVAL_PAWN_FAR_FROM_PROMO_MAX_PENALTY: i32 = 100;
pub const DEFAULT_EVAL_MINOR_DEVELOPMENT_PENALTY_THRESHOLD: i32 = 400;
pub const DEFAULT_EVAL_MIN_MAJOR_DEVELOPMENT_PENALTY: i32 = 16;
pub const DEFAULT_EVAL_MIN_FAIRY_DEVELOPMENT_PENALTY: i32 = 80;
pub const DEFAULT_EVAL_KING_DEFENDER_VALUE_THRESHOLD: i32 = 400;
pub const DEFAULT_EVAL_COMPLEXITY_DAMP: i32 = 8;
pub const DEFAULT_EVAL_COMPLEXITY_EXCESS_MAX: i32 = 40;
pub const DEFAULT_EVAL_KING_SHIELD_AHEAD_MAX_DIST: i32 = 3;
pub const DEFAULT_EVAL_MG_KING_PAWN_AHEAD_PENALTY: i32 = 20;
pub const DEFAULT_EVAL_EG_KING_PAWN_AHEAD_PENALTY: i32 = 0;
pub const DEFAULT_EVAL_MG_FAR_SLIDER_PENALTY_MULT: i32 = 100;
pub const DEFAULT_EVAL_EG_FAR_SLIDER_PENALTY_MULT: i32 = 44;
pub const DEFAULT_EVAL_SLIDER_THREAT_DIV: i32 = 12;
pub const DEFAULT_EVAL_SLIDER_THREAT_CAP: i32 = 41;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_0: i32 = 2;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_1: i32 = 0;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_2: i32 = 12;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_3: i32 = 25;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_4: i32 = 42;
pub const DEFAULT_EVAL_CANDIDATE_PASSER_BONUS_5: i32 = 74;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_0: i32 = 7;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_1: i32 = 3;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_2: i32 = 6;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_3: i32 = 0;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_4: i32 = 3;
pub const DEFAULT_EVAL_PAWN_FRIENDLY_KING_DIST_5: i32 = 14;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_0: i32 = 4;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_1: i32 = 4;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_2: i32 = 0;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_3: i32 = 8;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_4: i32 = 9;
pub const DEFAULT_EVAL_PAWN_ENEMY_KING_DIST_5: i32 = 19;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_0: i32 = 0;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_1: i32 = 0;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_2: i32 = 0;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_3: i32 = 5;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_4: i32 = 8;
pub const DEFAULT_EVAL_PASSED_FRIENDLY_KING_DIST_5: i32 = 4;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_0: i32 = 0;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_1: i32 = 10;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_2: i32 = 1;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_3: i32 = 3;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_4: i32 = 3;
pub const DEFAULT_EVAL_PASSED_ENEMY_KING_DIST_5: i32 = 9;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_0: i32 = 0;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_1: i32 = 3;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_2: i32 = 4;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_3: i32 = 13;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_4: i32 = 21;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_0_5: i32 = 37;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_0: i32 = 0;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_1: i32 = 4;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_2: i32 = 13;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_3: i32 = 26;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_4: i32 = 57;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_0_1_5: i32 = 81;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_0: i32 = 2;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_1: i32 = 0;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_2: i32 = 16;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_3: i32 = 36;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_4: i32 = 70;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_0_5: i32 = 124;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_0: i32 = 0;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_1: i32 = 8;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_2: i32 = 38;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_3: i32 = 81;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_4: i32 = 148;
pub const DEFAULT_EVAL_PASSED_PAWN_ADV_BONUS_1_1_5: i32 = 238;

// Piece Values

/// Nearest piece along each of the king's 8 rays, plus whether the ring is covered.
/// Index map matches the per-piece form it replaces: 0=NE 1=SE 2=NW 3=SW 4=E 5=W 6=N 7=S.
/// One index lookup per line replaces a scan of every piece on the board.
fn king_rays_from_indices(
    indices: &crate::moves::SpatialIndices,
    kx: i64,
    ky: i64,
    own: PlayerColor,
) -> ([(i32, i32, PlayerColor, PieceType); 8], bool) {
    let mut rays = [(i32::MAX, 0, PlayerColor::Neutral, PieceType::Void); 8];
    let mut ring = false;
    let mut put = |slot: usize, along: i64, end: crate::moves::LineEnd, base: i64| {
        if let Some((coord, packed)) = end {
            let p = Piece::from_packed(packed);
            let pt = p.piece_type();
            let dist = saturating_dist_i32((coord - base).abs());
            if dist < rays[slot].0 {
                rays[slot] = (dist, get_piece_value_base(pt), p.color(), pt);
            }
            if dist == 1
                && ((p.color() == own && (pt == PieceType::Pawn || pt == PieceType::Guard))
                    || pt.is_neutral_type())
            {
                ring = true;
            }
            let _ = along;
        }
    };
    if let Some(l) = indices.rows.get(&ky) {
        let (f, b) = l.neighbors(kx);
        put(4, 0, f, kx);
        put(5, 0, b, kx);
    }
    if let Some(l) = indices.cols.get(&kx) {
        let (f, b) = l.neighbors(ky);
        put(6, 0, f, ky);
        put(7, 0, b, ky);
    }
    if let Some(l) = indices.diag1.get(&(kx - ky)) {
        let (f, b) = l.neighbors(kx);
        put(0, 0, f, kx);
        put(3, 0, b, kx);
    }
    if let Some(l) = indices.diag2.get(&(kx + ky)) {
        let (f, b) = l.neighbors(kx);
        put(1, 0, f, kx);
        put(2, 0, b, kx);
    }
    (rays, ring)
}

pub fn get_piece_value_base(piece_type: PieceType) -> i32 {
    match piece_type {
        // neutral/blocking pieces - no material value
        PieceType::Void => 0,
        PieceType::Obstacle => 0,

        // orthodox - adjusted for infinite chess where sliders dominate
        PieceType::Pawn => pawn(),
        PieceType::Knight => knight(),     // Weak in infinite chess
        PieceType::Bishop => bishop(),     // Strong slider
        PieceType::Rook => rook(),         // Very strong in infinite chess
        PieceType::Queen => queen_value(), // > 2 rooks
        PieceType::Guard => guard(),

        // short / medium range
        PieceType::Camel => camel(),     // (1,3) leaper
        PieceType::Giraffe => giraffe(), // (1,4) leaper
        PieceType::Zebra => zebra(),     // (2,3) leaper

        // riders / compounds
        PieceType::Knightrider => knightrider(),
        PieceType::Amazon => queen_value() + knight() + amazon_compound_bonus(),
        PieceType::Hawk => hawk(),
        PieceType::Chancellor => rook() + knight() + chancellor_bonus(),
        PieceType::Archbishop => archbishop(),
        PieceType::Centaur => centaur(),

        PieceType::King => guard(),
        PieceType::RoyalQueen => queen_value(),
        PieceType::RoyalCentaur => centaur(),

        // special infinite-board pieces
        PieceType::Rose => rose(),
        PieceType::Huygen => huygen(),
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

// Distance penalties to discourage sliders far away from the king "zone".
// We look at distance to both own and enemy king and penalize pieces that
// drift too far from either.

// Max distance a single piece can skew the cloud center from the reference point.
// Prevents extreme outliers (e.g., a queen at 1e15) from dominating the weighted average.
// Pieces beyond this distance have their position clamped for centroid calculation.

// Shared constants for ray detection
const DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// Bishop pair & queen heuristics
// Tapered pairs defined below

// Fairy Piece Evaluation

// Leaper positioning (tropism to kings and piece cloud)
// Beyond this, bonus is capped

// Compound piece weight scaling (fraction of base piece eval to inherit)

// Pawn Distance Scaling

// Pawns far from promotion are worth much less in infinite chess

// Development

// Minimum starting square penalty for minors

// King defender bonuses/penalties
// Low-value pieces near own king = good (defense)
// High-value pieces near own king = bad (should be attacking)

// Game Phase

pub const MAX_PHASE: i32 = 24;

/// The taper clock runs on the game's own starting material: a variant that
/// begins with double an army reaches "endgame" at double the removals.
#[inline(always)]
pub fn effective_phase(raw_phase: i32, initial_phase: i32) -> i32 {
    (raw_phase * MAX_PHASE / initial_phase.max(MAX_PHASE)).min(MAX_PHASE)
}
// A cp edge cashes far less often with a big army still aboard (measured:
// p>=0.8 evals win 61% in CoaIP vs 79% in Classical), so damp by excess phase.
pub const MAX_KING_PHASE: i32 = 8;

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

// Tapered Evaluation Constants (MG, EG)

// King Safety
pub const DEFAULT_EVAL_MG_BEHIND_KING_BONUS: i32 = 45;
pub const DEFAULT_EVAL_EG_BEHIND_KING_BONUS: i32 = 59; // More important to be behind king in EG

pub const DEFAULT_EVAL_MG_KING_TROPISM_BONUS: i32 = 10;
pub const DEFAULT_EVAL_EG_KING_TROPISM_BONUS: i32 = 6; // King centralized -> piece proximity matters more

// Shelter / Ring
pub const DEFAULT_EVAL_MG_KING_RING_MISSING_PENALTY: i32 = 52;
pub const DEFAULT_EVAL_EG_KING_RING_MISSING_PENALTY: i32 = 11; // Less penalty in EG

pub const DEFAULT_EVAL_MG_KING_PAWN_SHIELD_BONUS: i32 = 20;
pub const DEFAULT_EVAL_EG_KING_PAWN_SHIELD_BONUS: i32 = 0; // Shield less critical

// A pawn only shelters the king when it is close in front; on an unbounded
// board an ahead pawn could otherwise be arbitrarily far and fabricate cover.

pub const DEFAULT_EVAL_MG_KING_OPEN_FILE_PENALTY: i32 = 28;
pub const DEFAULT_EVAL_EG_KING_OPEN_FILE_PENALTY: i32 = 0;

// Structural
pub const DEFAULT_EVAL_MG_CONNECTED_PAWN_BONUS: i32 = 0;
pub const DEFAULT_EVAL_EG_CONNECTED_PAWN_BONUS: i32 = 15; // Chains critical in EG

pub const DEFAULT_EVAL_MG_KING_DEFENDER_BONUS: i32 = 18;
pub const DEFAULT_EVAL_EG_KING_DEFENDER_BONUS: i32 = 0; // Less need for defenders

// Slider Distances (Centralization less critical in EG)

// Piece on Open File Bonuses

// Passed Pawn Detail (MG/EG tapered arrays by relative rank 0-5)
// Rank 0 is far, Rank 5 is near promotion.

// passed_pawn_adv_bonus()[canAdvance][safeAdvance][rank]

pub const DEFAULT_EVAL_MG_PASSED_SAFE_PATH_BONUS: i32 = 27;
pub const DEFAULT_EVAL_EG_PASSED_SAFE_PATH_BONUS: i32 = 67;

/// Probe a square offset (dx, dy) from a piece at local tile index `idx`.
/// Targets that stay inside the current 8x8 tile are read straight from the
/// tile's bitboard/piece array, skipping the TileTable hash probe.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_local_probe(
    board: &crate::board::Board,
    occ_all: u64,
    piece_arr: &[u8; 64],
    idx: usize,
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
) -> Option<crate::board::Piece> {
    let lx = (idx % 8) as i64 + dx;
    let ly = (idx / 8) as i64 + dy;
    if (0..8).contains(&lx) && (0..8).contains(&ly) {
        let ti = (ly * 8 + lx) as usize;
        if (occ_all >> ti) & 1 != 0 {
            Some(crate::board::Piece::from_packed(piece_arr[ti]))
        } else {
            None
        }
    } else {
        board.get_piece(x + dx, y + dy)
    }
}

// Main Evaluation
pub fn evaluate(game: &GameState) -> i32 {
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
    // Read once per evaluation, then threaded to each term it weights.
    let style = eval_style();
    let mut score = game.material_score;
    // Seeds the score, so it has to appear as a row or TOTAL is not the eval.
    tracer.record("Material (net)", game.material_score, 0);

    let (white_royals, black_royals) = (game.white_royals.as_slice(), game.black_royals.as_slice());
    let white_king = white_royals.first().copied();
    let black_king = black_royals.first().copied();

    let eff_phase = effective_phase(game.total_phase, game.initial_phase);
    let taper = |mg: i32, eg: i32| -> i32 {
        ((mg * eff_phase) + (eg * (MAX_PHASE - eff_phase))) / MAX_PHASE
    };

    // Single-Pass Collection and Scoring
    let mut phase = 0; // decreases with fewer pieces
    let mut white_undeveloped = 0;
    let mut black_undeveloped = 0;
    let mut white_bishops = 0;
    let mut white_bishop_colors = (false, false);
    let mut black_bishops = 0;
    let mut black_bishop_colors = (false, false);
    let mut cloud_sum_dx: i64 = 0;
    let mut cloud_sum_dy: i64 = 0;
    let mut cloud_count: i64 = 0;
    let mut cloud_spread_sum: i64 = 0;

    // Doubled units throughout the cloud: the kings' midpoint lands on a half
    // square whenever they sit an odd distance apart, and truncating it biases
    // every distance measured from it by colour.
    let (ref_x, ref_y) = match (white_king, black_king) {
        (Some(wk), Some(bk)) => (wk.x + bk.x, wk.y + bk.y),
        (Some(wk), None) => (2 * wk.x, 2 * wk.y),
        (None, Some(bk)) => (2 * bk.x, 2 * bk.y),
        (None, None) => (0, 0),
    };

    // Slider counts for attack bonus (white, black) and attacking units
    let mut w_diag_count = 0;
    let mut w_ortho_count = 0;
    let mut b_diag_count = 0;
    let mut b_ortho_count = 0;
    let mut w_additional_attack_units = 0;
    let mut b_additional_attack_units = 0;
    // Wall-density gate below: Void+Obstacle choke sliders now, but only the
    // uncapturable Void permanently confines a leaper's cell.
    let mut wall_count: i64 = 0;
    let mut void_count: i64 = 0;

    // Threat points for defense urgency
    let mut w_threat_points = 0;
    let mut black_threat_points = 0;
    let mut w_has_queen_threat = false;
    let mut b_has_queen_threat = false;

    // Interaction threat totals
    let mut w_pawn_threats = 0;
    let mut b_pawn_threats = 0;
    let mut w_minor_threats = 0;
    let mut w_slider_threats = 0;
    let mut b_minor_threats = 0;
    let mut b_slider_threats = 0;

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

    let mut white_royal_tropisms: SmallVec<[_; 1]> = game
        .white_royals
        .iter()
        .map(|r| RoyalTropismMetrics {
            piece_type: game
                .board
                .get_piece(r.x, r.y)
                .map(|p| p.piece_type())
                .unwrap_or(PieceType::Void),
            x: r.x,
            y: r.y,

            // placeholders
            tropism_addend: 0,
            attacking_units: 0,
            defender_units: 0,
            defender_units_in_distance: [0; 8],
        })
        .collect();
    let mut black_royal_tropisms: SmallVec<[_; 1]> = game
        .black_royals
        .iter()
        .map(|r| RoyalTropismMetrics {
            piece_type: game
                .board
                .get_piece(r.x, r.y)
                .map(|p| p.piece_type())
                .unwrap_or(PieceType::Void),
            x: r.x,
            y: r.y,

            // placeholders
            tropism_addend: 0,
            attacking_units: 0,
            defender_units: 0,
            defender_units_in_distance: [0; 8],
        })
        .collect();

    // Interaction threat constants
    const PAWN_THREATENS_MINOR: i32 = 25;
    const PAWN_THREATENS_ROOK: i32 = 40;
    const PAWN_THREATENS_QUEEN: i32 = 60;
    // Sliders are graded by value gap rather than bucketed: on an unbounded
    // board they are the pieces that can threaten from anywhere, and a fixed
    // tier would have to be re-cut every time the piece values are refitted.
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

    // Unified pawn metrics accumulation
    let mut w_pawn_storm_total: i32 = 0;
    let mut b_pawn_storm_total: i32 = 0;
    let mut w_storm_count: i32 = 0;
    let mut b_storm_count: i32 = 0;

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

                        // Main piece loop
                        for (cx, cy, tile) in game.board.tiles.iter() {
                            if tile.occ_all == 0 {
                                continue;
                            }

                            // Slider counts from this tile's bitboards.
                            w_diag_count +=
                                (tile.occ_diag_sliders & tile.occ_white).count_ones() as i32;
                            b_diag_count +=
                                (tile.occ_diag_sliders & tile.occ_black).count_ones() as i32;
                            w_ortho_count +=
                                (tile.occ_ortho_sliders & tile.occ_white).count_ones() as i32;
                            b_ortho_count +=
                                (tile.occ_ortho_sliders & tile.occ_black).count_ones() as i32;

                            let mut bits = tile.occ_all;
                            while bits != 0 {
                                let idx = bits.trailing_zeros() as usize;
                                bits &= bits - 1;
                                let packed = tile.piece[idx];
                                let piece = crate::board::Piece::from_packed(packed);
                                let pt = piece.piece_type();
                                let piece_val = get_piece_value_base(pt);
                                let is_white = piece.color() == PlayerColor::White;
                                let is_neutral = pt.is_neutral_type();
                                let x = cx * 8 + (idx % 8) as i64;
                                let y = cy * 8 + (idx / 8) as i64;

                                // Attack and defender units for king tropism.
                                if !is_neutral
                                    && matches!(
                                        pt,
                                        PieceType::Amazon
                                            | PieceType::Chancellor
                                            | PieceType::Archbishop
                                            | PieceType::Knightrider
                                    )
                                {
                                    if is_white {
                                        w_additional_attack_units += 100;
                                    } else {
                                        b_additional_attack_units += 100;
                                    }
                                }
                                {
                                    // Tropism weights divided by Chebyshev distance 1..=7.
                                    // Indexing replaces a per-piece-per-royal idiv.
                                    const DEF_NEUTRAL: [i32; 8] = [0, 25, 12, 8, 6, 5, 4, 3];
                                    const DEF_PAWN: [i32; 8] = [0, 33, 16, 11, 8, 6, 5, 4];
                                    const DEF_PIECE: [i32; 8] = [0, 100, 50, 33, 25, 20, 16, 14];

                                    let (your_royals, enemy_royals) = if is_white {
                                        (&mut white_royal_tropisms, &mut black_royal_tropisms)
                                    } else {
                                        (&mut black_royal_tropisms, &mut white_royal_tropisms)
                                    };
                                    let leaper_attack_units =
                                        if matches!(pt, PieceType::Hawk | PieceType::Rose) {
                                            100
                                        } else if matches!(
                                            pt,
                                            PieceType::Knight
                                                | PieceType::Centaur
                                                | PieceType::Camel
                                                | PieceType::Giraffe
                                                | PieceType::Zebra
                                                | PieceType::Huygen
                                        ) {
                                            50
                                        } else {
                                            0
                                        };
                                    if leaper_attack_units != 0 {
                                        for ek in enemy_royals {
                                            let dx = (x - ek.x).abs();
                                            let dy = (y - ek.y).abs();
                                            if dx <= 20 && dy <= 20 {
                                                ek.attacking_units += leaper_attack_units;
                                            }
                                        }
                                    }
                                    if piece.color() == PlayerColor::Neutral {
                                        for king in white_royal_tropisms
                                            .iter_mut()
                                            .chain(black_royal_tropisms.iter_mut())
                                        {
                                            let d = (x - king.x).abs().max((y - king.y).abs());
                                            if d <= 7 {
                                                king.defender_units_in_distance[d as usize] +=
                                                    DEF_NEUTRAL[d as usize];
                                            }
                                        }
                                    } else {
                                        let table = if matches!(pt, PieceType::Pawn) {
                                            &DEF_PAWN
                                        } else if !pt.is_royal() {
                                            &DEF_PIECE
                                        } else {
                                            &[0i32; 8]
                                        };
                                        for yk in your_royals {
                                            let d = (x - yk.x).abs().max((y - yk.y).abs());
                                            if d <= 7 {
                                                yk.defender_units_in_distance[d as usize] +=
                                                    table[d as usize];
                                            }
                                        }
                                    }
                                }

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
                                } else if !pt.is_neutral_type() {
                                    // Neutral pieces score no activity or attack;
                                    // they only help king safety defensively.
                                    piece_list.push((x, y, piece));
                                } else if pt == PieceType::Void {
                                    wall_count += 1;
                                    void_count += 1;
                                } else if pt == PieceType::Obstacle {
                                    wall_count += 1;
                                }

                                // 3. Piece counts for scaling (Non-pawn, non-royal)
                                if !is_neutral && pt != PieceType::Pawn && !pt.is_royal() {
                                    if is_white {
                                        white_non_pawn_non_royal += 1;
                                    } else {
                                        black_non_pawn_non_royal += 1;
                                    }
                                }

                                // 4. Cloud Stats (Non-pawn)
                                if !is_neutral && pt != PieceType::Pawn {
                                    let cw = get_centrality_weight(pt);
                                    if cw > 0 {
                                        let dx = 2 * x - ref_x;
                                        let dy = 2 * y - ref_y;
                                        let skew2 = 2 * cloud_center_max_skew_dist() as i64;
                                        let cdx = dx.clamp(-skew2, skew2);
                                        let cdy = dy.clamp(-skew2, skew2);
                                        cloud_sum_dx += cw * cdx;
                                        cloud_sum_dy += cw * cdy;
                                        cloud_count += cw;
                                        cloud_spread_sum += cw * cdx.abs().max(cdy.abs());
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

                                // Count this slider if it sits within an enemy royal's attack zone.
                                if is_slider {
                                    if is_white {
                                        for bk in &black_royal_tropisms {
                                            if (x - bk.x).abs() <= ATTACK_ZONE_RADIUS
                                                && (y - bk.y).abs() <= ATTACK_ZONE_RADIUS
                                            {
                                                w_sliders_in_zone += 1;
                                                break;
                                            }
                                        }
                                    } else {
                                        for wk in &white_royal_tropisms {
                                            if (x - wk.x).abs() <= ATTACK_ZONE_RADIUS
                                                && (y - wk.y).abs() <= ATTACK_ZONE_RADIUS
                                            {
                                                b_sliders_in_zone += 1;
                                                break;
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
                                        if let Some(target) = tile_local_probe(
                                            &game.board,
                                            tile.occ_all,
                                            &tile.piece,
                                            idx,
                                            x,
                                            y,
                                            dx,
                                            dy,
                                        ) && target.color() == enemy
                                        {
                                            let tv = get_piece_value_base(target.piece_type());
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
                                        if let Some(target) = tile_local_probe(
                                            &game.board,
                                            tile.occ_all,
                                            &tile.piece,
                                            idx,
                                            x,
                                            y,
                                            dx,
                                            dy,
                                        ) && target.color() == enemy
                                        {
                                            let tv = get_piece_value_base(target.piece_type());
                                            let mv = piece_val;
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
                                } else if crate::attacks::is_slider(pt) {
                                    // A slider attacks exactly the first piece on each of
                                    // its rays; both ends of a line share one binary search.
                                    let idx_sp = &game.spatial_indices;
                                    let own = piece.color();
                                    let mut bonus = 0;
                                    if crate::attacks::is_ortho_slider(pt) {
                                        if let Some(l) = idx_sp.rows.get(&y) {
                                            let (f, b) = l.neighbors(x);
                                            bonus += slider_threat_bonus(f, own, piece_val)
                                                + slider_threat_bonus(b, own, piece_val);
                                        }
                                        if let Some(l) = idx_sp.cols.get(&x) {
                                            let (f, b) = l.neighbors(y);
                                            bonus += slider_threat_bonus(f, own, piece_val)
                                                + slider_threat_bonus(b, own, piece_val);
                                        }
                                    }
                                    if crate::attacks::is_diag_slider(pt) {
                                        if let Some(l) = idx_sp.diag1.get(&(x - y)) {
                                            let (f, b) = l.neighbors(x);
                                            bonus += slider_threat_bonus(f, own, piece_val)
                                                + slider_threat_bonus(b, own, piece_val);
                                        }
                                        if let Some(l) = idx_sp.diag2.get(&(x + y)) {
                                            let (f, b) = l.neighbors(x);
                                            bonus += slider_threat_bonus(f, own, piece_val)
                                                + slider_threat_bonus(b, own, piece_val);
                                        }
                                    }
                                    if bonus > 0 {
                                        if is_white {
                                            w_slider_threats += bonus;
                                        } else {
                                            b_slider_threats += bonus;
                                        }
                                    }
                                }

                                // 8. Minor stats
                                if (pt.is_minor() || pt == PieceType::Archbishop)
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

                                // 7. Threat points for urgency
                                if !is_neutral && !pt.is_royal() && pt != PieceType::Pawn {
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

                                // 8. Pawn advancement, storm, and space metrics (unified pass)
                                if pt == PieceType::Pawn {
                                    if is_white {
                                        if y >= w_promo {
                                            w_pawn_penalty -= pawn_past_promo_penalty();
                                        } else {
                                            let dist = w_promo - y;
                                            let bonus =
                                                (pawn_full_value_threshold() - dist.min(255) as i32) * 6;

                                            w_pawn_bonus += bonus.max(-pawn_far_from_promo_max_penalty());
                                            if y > white_max_y {
                                                white_max_y = y;
                                            }
                                        }
                                        // Pawn storm: check distance to all black royals
                                        for bk in black_royals {
                                            let file_dist = (x - bk.x).abs();
                                            if file_dist <= 3 {
                                                let rank_dist = bk.y - y;
                                                if (1..=6).contains(&rank_dist) {
                                                    let adv_bonus: i32 = match rank_dist {
                                                        1 => 30,
                                                        2 => 20,
                                                        3 => 12,
                                                        4 => 6,
                                                        5 => 3,
                                                        _ => 1,
                                                    };
                                                    let file_scale: i32 = match file_dist {
                                                        0 => 110,
                                                        1 => 100,
                                                        2 => 80,
                                                        _ => 60,
                                                    };
                                                    w_pawn_storm_total +=
                                                        adv_bonus * file_scale / 100;
                                                    w_storm_count += 1;
                                                }
                                            }
                                        }
                                    } else {
                                        if y <= b_promo {
                                            b_pawn_penalty -= pawn_past_promo_penalty();
                                        } else {
                                            let dist = y - b_promo;
                                            let bonus =
                                                (pawn_full_value_threshold() - dist.min(255) as i32) * 6;

                                            b_pawn_bonus += bonus.max(-pawn_far_from_promo_max_penalty());
                                            if y < black_min_y {
                                                black_min_y = y;
                                            }
                                        }
                                        // Pawn storm: check distance to all white royals
                                        for wk in white_royals {
                                            let file_dist = (x - wk.x).abs();
                                            if file_dist <= 3 {
                                                let rank_dist = y - wk.y;
                                                if (1..=6).contains(&rank_dist) {
                                                    let adv_bonus: i32 = match rank_dist {
                                                        1 => 30,
                                                        2 => 20,
                                                        3 => 12,
                                                        4 => 6,
                                                        5 => 3,
                                                        _ => 1,
                                                    };
                                                    let file_scale: i32 = match file_dist {
                                                        0 => 110,
                                                        1 => 100,
                                                        2 => 80,
                                                        _ => 60,
                                                    };
                                                    b_pawn_storm_total +=
                                                        adv_bonus * file_scale / 100;
                                                    b_storm_count += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // King rays come from the spatial index: one lookup per line
                        // replaces testing every piece for alignment.
                        for &wk in white_royals {
                            let (r, ring) = king_rays_from_indices(
                                &game.spatial_indices,
                                wk.x,
                                wk.y,
                                PlayerColor::White,
                            );
                            for i in 0..8 {
                                if r[i].0 < w_king_rays[i].0 {
                                    w_king_rays[i] = r[i];
                                }
                            }
                            w_king_ring_covered |= ring;
                        }
                        for &bk in black_royals {
                            let (r, ring) = king_rays_from_indices(
                                &game.spatial_indices,
                                bk.x,
                                bk.y,
                                PlayerColor::Black,
                            );
                            for i in 0..8 {
                                if r[i].0 < b_king_rays[i].0 {
                                    b_king_rays[i] = r[i];
                                }
                            }
                            b_king_ring_covered |= ring;
                        }

                        // Finalize slider totals, defender units, and king tropism addends.
                        let w_total_sliders = w_diag_count + w_ortho_count;
                        let b_total_sliders = b_diag_count + b_ortho_count;
                        for king in &mut white_royal_tropisms {
                            let mut defender_bonus = 0;
                            for d in 1..8 {
                                let defender_units_to_add =
                                    king.defender_units_in_distance[d as usize];
                                king.defender_units += defender_units_to_add;
                                if defender_units_to_add >= 200 {
                                    defender_bonus = (300 + 100 * d - 100 * b_total_sliders).max(0);
                                }
                            }
                            king.defender_units += defender_bonus;
                        }
                        for king in &mut black_royal_tropisms {
                            let mut defender_bonus = 0;
                            for d in 1..8 {
                                let defender_units_to_add =
                                    king.defender_units_in_distance[d as usize];
                                king.defender_units += defender_units_to_add;
                                if defender_units_to_add >= 200 {
                                    defender_bonus = (300 + 100 * d - 100 * w_total_sliders).max(0);
                                }
                            }
                            king.defender_units += defender_bonus;
                        }
                        for king in &mut white_royal_tropisms {
                            let total_effective_units = w_total_sliders
                                + (w_additional_attack_units + king.attacking_units
                                    - king.defender_units)
                                    / 100;
                            king.tropism_addend = compute_tropism_addend(total_effective_units);
                        }
                        for king in &mut black_royal_tropisms {
                            let total_effective_units = b_total_sliders
                                + (b_additional_attack_units + king.attacking_units
                                    - king.defender_units)
                                    / 100;
                            king.tropism_addend = compute_tropism_addend(total_effective_units);
                        }

                        // Accumulate piece-to-king tropism using the finalized addends.
                        for &(px, py, ppiece) in piece_list.iter() {
                            let ppt = ppiece.piece_type();
                            let piece_val = if !ppt.is_royal() && ppt != PieceType::Pawn {
                                get_piece_value_base(ppt)
                            } else {
                                0
                            };
                            if piece_val == 0 {
                                continue;
                            }
                            if ppiece.color() == PlayerColor::White {
                                for bk in &black_royal_tropisms {
                                    let d = (px - bk.x).abs().max((py - bk.y).abs());
                                    w_attacking_tropism +=
                                        tropism_contribution(piece_val, d, bk.tropism_addend);
                                }
                                for wk in &white_royal_tropisms {
                                    let d = (px - wk.x).abs().max((py - wk.y).abs());
                                    w_defensive_tropism += tropism_contribution(
                                        piece_val.min(350),
                                        d,
                                        wk.tropism_addend,
                                    );
                                }
                            } else {
                                for wk in &white_royal_tropisms {
                                    let d = (px - wk.x).abs().max((py - wk.y).abs());
                                    b_attacking_tropism +=
                                        tropism_contribution(piece_val, d, wk.tropism_addend);
                                }
                                for bk in &black_royal_tropisms {
                                    let d = (px - bk.x).abs().max((py - bk.y).abs());
                                    b_defensive_tropism += tropism_contribution(
                                        piece_val.min(350),
                                        d,
                                        bk.tropism_addend,
                                    );
                                }
                            }
                        }

                        // Post-Pass processing
                        let final_phase = effective_phase(phase, game.initial_phase);
                        // Doubled units: the centroid of a symmetric position lands on a
                        // half square, and truncating it to an integer moves the centre
                        // toward one side, biasing every cloud distance by colour.
                        let cloud_center = if cloud_count > 0 {
                            Some(Coordinate {
                                x: ref_x + cloud_sum_dx / cloud_count,
                                y: ref_y + cloud_sum_dy / cloud_count,
                            })
                        } else {
                            None
                        };
                        // Weighted average Chebyshev distance of pieces from the kings' midpoint.
                        // Low = tight/closed position (leapers thrive), High = spread/open (sliders dominate).
                        let cloud_avg_spread = if cloud_count > 0 {
                            (cloud_spread_sum / cloud_count / 2) as i32
                        } else {
                            8 // neutral fallback
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

                        // Defense urgency
                        let calc_urgency = |tp: i32| (10 + tp + (tp / 4)).min(100);
                        let w_urgency = calc_urgency(black_threat_points);
                        let b_urgency = calc_urgency(w_threat_points);

                        // Attack scale calculation (Finalized from Readiness loop counts)
                        let w_attack_ready = compute_attack_readiness_optimized(
                            &b_king_rays,
                            black_king.is_some(),
                            w_sliders_in_zone,
                            PlayerColor::White,
                        );
                        let b_attack_ready = compute_attack_readiness_optimized(
                            &w_king_rays,
                            white_king.is_some(),
                            b_sliders_in_zone,
                            PlayerColor::Black,
                        );

                        // Sliders lose value only where rays are genuinely open;
                        // leapers lose their jump immunity only to a permanent Void.
                        let (slider_geometry_ctx, leaper_geometry_ctx) = {
                            let (bmin_x, bmax_x, bmin_y, bmax_y) =
                                crate::moves::get_coord_bounds();
                            let world_size = (bmax_x.saturating_sub(bmin_x))
                                .max(bmax_y.saturating_sub(bmin_y));
                            // 100 on 8x8-class boards, 0 once the world exceeds 30.
                            let size_ctx = ((30 - world_size) * 100 / 20).clamp(0, 100) as i32;
                            let total_squares = (bmax_x - bmin_x + 1) * (bmax_y - bmin_y + 1);
                            let real_pieces =
                                (game.white_piece_count + game.black_piece_count) as i64;
                            let non_piece = total_squares.saturating_sub(real_pieces);
                            // Zero out past ~10% wall fill: below that a few scattered
                            // walls don't meaningfully pre-block rays/cells.
                            let openness_pct = |count: i64| -> i32 {
                                if non_piece > 0 {
                                    (100 - (count * 100 / non_piece) * 10).clamp(0, 100) as i32
                                } else {
                                    100
                                }
                            };
                            let slider_openness = openness_pct(wall_count);
                            let leaper_openness = openness_pct(void_count);
                            (
                                size_ctx * slider_openness / 100,
                                size_ctx * leaper_openness / 100,
                            )
                        };

                        score += evaluate_pieces_processed(
                            game,
                            white_royals,
                            black_royals,
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
                                cloud_avg_spread,
                                slider_geometry_ctx,
                                leaper_geometry_ctx,
                            },
                            w_attack_ready,
                            b_attack_ready,
                            white_pawns,
                            black_pawns,
                        );

                        let ks_metrics = KingSafetyMetrics {
                            white_slider_counts: (w_diag_count, w_ortho_count),
                            black_slider_counts: (b_diag_count, b_ortho_count),
                            urgency: (w_urgency, b_urgency),
                            has_enemy_queen: (b_has_queen_threat, w_has_queen_threat),
                        };
                        score += evaluate_king_safety_traced(
                            game,
                            white_royals,
                            black_royals,
                            final_phase,
                            tracer,
                            &ks_metrics,
                            white_pawns,
                            black_pawns,
                            &w_king_rays,
                            &b_king_rays,
                            w_king_ring_covered,
                            b_king_ring_covered,
                            style,
                        );

                        score += evaluate_pawn_structure_traced(
                            game,
                            final_phase,
                            white_royals,
                            black_royals,
                            tracer,
                            white_pawns,
                            black_pawns,
                            white_rq,
                            black_rq,
                        );

                        if phase < MAX_KING_PHASE {
                            score += evaluate_king_positioning_traced(
                                game,
                                MAX_KING_PHASE - phase,
                                white_royal_tropisms.as_ref(),
                                black_royal_tropisms.as_ref(),
                                tracer,
                                white_pawns,
                                black_pawns,
                            );
                        }

                        // Interaction Threats (Result from merged loop). Seeing a
                        // threat is the first thing a weak level gives up, so these
                        // are scaled per-component and the trace rows follow.
                        let w_pawn_threats = style.attack(w_pawn_threats);
                        let b_pawn_threats = style.attack(b_pawn_threats);
                        let w_minor_threats = style.attack(w_minor_threats);
                        let b_minor_threats = style.attack(b_minor_threats);
                        let w_slider_threats = style.attack(w_slider_threats);
                        let b_slider_threats = style.attack(b_slider_threats);
                        tracer.record("Threats: Pawn", w_pawn_threats, b_pawn_threats);
                        tracer.record("Threats: Minor", w_minor_threats, b_minor_threats);
                        tracer.record("Threats: Slider", w_slider_threats, b_slider_threats);
                        score += (w_pawn_threats + w_minor_threats + w_slider_threats)
                            - (b_pawn_threats + b_minor_threats + b_slider_threats);

                        // Global Tropism. The attack/defense split already has a
                        // per-side percentage, so the style composes into it: weak
                        // levels crowd their own king instead of the enemy's.
                        let gt_att_mult = taper(180, 360);
                        let gt_def_mult = taper(120, 60);

                        let w_att_scale = style.attack(match game.game_rules.white_win_condition {
                            WinCondition::AllRoyalsCaptured => 80,
                            _ => 100,
                        });
                        let b_att_scale = style.attack(match game.game_rules.black_win_condition {
                            WinCondition::AllRoyalsCaptured => 80,
                            _ => 100,
                        });
                        let w_def_scale = style.defense(100);
                        let b_def_scale = style.defense(100);

                        // Normalize by 1000 since piece values are high and we want roughly 10-100 pts
                        let w_gt = (w_attacking_tropism * gt_att_mult * w_att_scale / 10000)
                            + (w_defensive_tropism * gt_def_mult * w_def_scale / 10000);
                        let b_gt = (b_attacking_tropism * gt_att_mult * b_att_scale / 10000)
                            + (b_defensive_tropism * gt_def_mult * b_def_scale / 10000);

                        tracer.record("Global Tropism", w_gt, b_gt);
                        score += w_gt - b_gt;

                        // Finalize unified pawn storm metrics (collected during main loop)
                        // Apply synergy multiplier when 2+ pawns threaten, then taper by phase.
                        let mut w_storm = w_pawn_storm_total;
                        let mut b_storm = b_pawn_storm_total;
                        if w_storm_count >= 2 {
                            w_storm = w_storm * (100 + (w_storm_count - 1) * 12) / 100;
                        }
                        if b_storm_count >= 2 {
                            b_storm = b_storm * (100 + (b_storm_count - 1) * 12) / 100;
                        }
                        let w_storm_scale =
                            style.attack(match game.game_rules.white_win_condition {
                                WinCondition::AllRoyalsCaptured => 55,
                                _ => 100,
                            });
                        let b_storm_scale =
                            style.attack(match game.game_rules.black_win_condition {
                                WinCondition::AllRoyalsCaptured => 55,
                                _ => 100,
                            });
                        let w_storm = taper(w_storm, w_storm * 40 / 100) * w_storm_scale / 100;
                        let b_storm = taper(b_storm, b_storm * 40 / 100) * b_storm_scale / 100;

                        tracer.record("King: Pawn Storm", w_storm, b_storm);
                        score += w_storm - b_storm;
                    }); // brq
                }); // wrq
            }); // bp
        }); // wp
    }); // pl

    // Damp the whole score by material complexity: the same advantage is worth
    // less with more resistance left, and trading it away raises the scaled score.
    let excess = (phase - MAX_PHASE).clamp(0, complexity_excess_max());
    if excess > 0 {
        let scaled = score * (1024 - complexity_damp() * excess) / 1024;
        tracer.record("Complexity scale", scaled - score, 0);
        score = scaled;
    }

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
    cloud_avg_spread: i32,
    /// 0-100: how much a bounded world truncates slider rays. Always 0 on an
    /// unbounded board, where a ray reaches regardless of how pieces clump.
    slider_geometry_ctx: i32,
    leaper_geometry_ctx: i32,
}

/// Peak context adjustment as % of base value. Uniform per class so it shifts
/// rider vs leaper without re-ranking pieces within a class.
fn geometry_ctx_pct(pt: PieceType) -> i32 {
    match pt {
        // Pure riders: every move is a blockable ray.
        PieceType::Rook
        | PieceType::Bishop
        | PieceType::Queen
        | PieceType::Knightrider
        | PieceType::Huygen => -12,
        // Compounds keep a jump component, so only the rider half pays.
        PieceType::Chancellor | PieceType::Archbishop | PieceType::Amazon => -6,
        // True leapers: jumps ignore the blockers that stop riders.
        PieceType::Knight
        | PieceType::Camel
        | PieceType::Zebra
        | PieceType::Giraffe
        | PieceType::Guard
        | PieceType::Hawk
        | PieceType::Centaur => 10,
        // Rose rides a curved path (neither cleanly), royals are priced by
        // their life rather than their mobility.
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_pieces_processed<T: EvaluationTracer>(
    game: &GameState,
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
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
    let cloud_avg_spread = metrics.cloud_avg_spread;

    let white_attack_ready = {
        let cap = (100 - metrics.white_undeveloped * 25).clamp(30, 100);
        white_attack_ready.min(cap)
    };
    let black_attack_ready = {
        let cap = (100 - metrics.black_undeveloped * 25).clamp(30, 100);
        black_attack_ready.min(cap)
    };

    for &(x, y, piece) in piece_list {
        let pt = piece.piece_type();
        let mut piece_score = match pt {
            PieceType::Rook => evaluate_rook(
                game,
                x,
                y,
                piece.color(),
                white_royals,
                black_royals,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Queen => evaluate_queen(
                game,
                x,
                y,
                piece.color(),
                white_royals,
                black_royals,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Bishop => evaluate_bishop(
                game,
                x,
                y,
                piece.color(),
                white_royals,
                black_royals,
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
                    white_royals,
                    black_royals,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                rook_eval * chancellor_rook_scale() / 100
                    + evaluate_compound_leap_threats(
                        game,
                        x,
                        y,
                        piece.color(),
                        get_piece_value_base(pt),
                        phase,
                    )
            }
            PieceType::Archbishop => {
                let bishop_eval = evaluate_bishop(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_royals,
                    black_royals,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                bishop_eval * archbishop_bishop_scale() / 100
                    + evaluate_compound_leap_threats(
                        game,
                        x,
                        y,
                        piece.color(),
                        get_piece_value_base(pt),
                        phase,
                    )
            }
            PieceType::Amazon => {
                let queen_eval = evaluate_queen(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_royals,
                    black_royals,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                let rook_eval = evaluate_rook(
                    game,
                    x,
                    y,
                    piece.color(),
                    white_royals,
                    black_royals,
                    phase,
                    white_pawns,
                    black_pawns,
                );
                (queen_eval * amazon_queen_scale() / 100)
                    + (rook_eval * amazon_rook_scale() / 100)
                    + evaluate_compound_leap_threats(
                        game,
                        x,
                        y,
                        piece.color(),
                        get_piece_value_base(pt),
                        phase,
                    )
            }
            PieceType::RoyalQueen => evaluate_queen(
                game,
                x,
                y,
                piece.color(),
                white_royals,
                black_royals,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Knight => evaluate_knight(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                cloud_avg_spread,
                phase,
                white_pawns,
                black_pawns,
            ),
            PieceType::Rose => {
                evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    cloud_center.as_ref(),
                    PieceType::Rose,
                    cloud_avg_spread,
                    phase,
                ) + evaluate_rose_reach(game, x, y, piece.color(), phase)
            }
            PieceType::Hawk
            | PieceType::Camel
            | PieceType::Giraffe
            | PieceType::Zebra => evaluate_leaper_positioning(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                pt,
                cloud_avg_spread,
                phase,
            ),
            PieceType::Centaur | PieceType::RoyalCentaur => {
                let leaper_eval = evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    cloud_center.as_ref(),
                    pt,
                    cloud_avg_spread,
                    phase,
                );
                leaper_eval * centaur_guard_scale() / 100
            }
            PieceType::Huygen => evaluate_huygen_reach(
                &game.spatial_indices,
                x,
                y,
                piece.color(),
                if piece.color() == PlayerColor::White {
                    black_royals
                } else {
                    white_royals
                },
                phase,
            ),
            PieceType::Guard => evaluate_leaper_positioning(
                x,
                y,
                piece.color(),
                cloud_center.as_ref(),
                PieceType::Guard,
                cloud_avg_spread,
                phase,
            ),
            // A knightrider rides along knight rays; on an unbounded board its
            // reach is unbounded so mobility-counting is meaningless. Use the
            // board-aware cloud-proximity/density shaping like the other riders.
            PieceType::Knightrider => {
                evaluate_leaper_positioning(
                    x,
                    y,
                    piece.color(),
                    cloud_center.as_ref(),
                    PieceType::Knightrider,
                    cloud_avg_spread,
                    phase,
                ) + evaluate_knightrider_reach(x, y, piece.color(), piece_list, phase)
            }
            _ => 0,
        };
        let piece_val = get_piece_value_base(pt);

        if let Some(center) = &cloud_center {
            let dx = (2 * x - center.x).abs() / 2;
            let dy = (2 * y - center.y).abs() / 2;
            let cheb = dx.max(dy);

            if pt != PieceType::Pawn && !pt.is_royal() && cheb > piece_cloud_cheb_radius() as i64 {
                let is_ortho = pt == PieceType::Rook || pt == PieceType::Chancellor;
                let is_diag = pt == PieceType::Bishop || pt == PieceType::Archbishop;
                let is_queen = pt == PieceType::Queen || pt == PieceType::Amazon;

                let value_factor = (piece_val / 100).max(1);
                let mult = taper(mg_far_slider_penalty_mult(), eg_far_slider_penalty_mult());

                if is_ortho || is_diag || is_queen {
                    // Sliders: only penalized if they cannot "see" the cloud center (misaligned).
                    // Distance doesn't matter (infinite range).
                    let mut lane_dist = i64::MAX;

                    if is_ortho || is_queen {
                        lane_dist = lane_dist.min(dx.min(dy));
                    }
                    if is_diag || is_queen {
                        let d1 = (2 * (x - y) - (center.x - center.y)).abs() / 2;
                        let d2 = (2 * (x + y) - (center.x + center.y)).abs() / 2;
                        lane_dist = lane_dist.min(d1.min(d2));
                    }

                    if lane_dist > slider_axis_wiggle() as i64 {
                        let excess = (lane_dist - slider_axis_wiggle() as i64)
                            .min(piece_cloud_cheb_max_excess() as i64)
                            as i32;
                        let penalty =
                            excess * cloud_penalty_per_100_value() * value_factor * mult / 100;
                        piece_score -= penalty;
                    }
                } else {
                    // Leapers/Others: penalized by distance (Chebyshev)
                    // We are only in this block if cheb > RADIUS, so dist_to_radius > 0
                    let dist_to_radius = cheb - piece_cloud_cheb_radius() as i64;
                    let excess = dist_to_radius.min(piece_cloud_cheb_max_excess() as i64) as i32;
                    let penalty =
                        excess * cloud_penalty_per_100_value() * value_factor * mult / 100;
                    piece_score -= penalty;
                }
            }
        }

        if (pt.is_minor() || pt == PieceType::Archbishop)
            && game.starting_squares.contains(&Coordinate::new(x, y))
        {
            // A fairy leaper is near-useless from its starting square -- an odd leap
            // pattern only pays once it has room -- while a knight or bishop at home
            // is far less urgent. One shared value has to compromise between the two,
            // which suits neither, so they are priced apart.
            piece_score -= if pt.is_minor() {
                if matches!(pt, PieceType::Knight | PieceType::Bishop) {
                    // A knight or bishop below the threshold costs nothing for
                    // sitting at home -- tuned to exactly 0, so the branch and its
                    // threshold check are skipped rather than computed and discarded.
                    if piece_val < minor_development_penalty_threshold() {
                        0
                    } else {
                        min_major_development_penalty()
                    }
                } else {
                    min_fairy_development_penalty()
                }
            } else {
                min_major_development_penalty()
            };
        }

        if !pt.is_royal() && pt != PieceType::Pawn {
            let own_royals = if piece.color() == PlayerColor::White {
                white_royals
            } else {
                black_royals
            };
            for &ok in own_royals {
                let dist = (x - ok.x).abs().max((y - ok.y).abs());
                if dist <= 3 {
                    if piece_val < king_defender_value_threshold() {
                        piece_score += taper(
                            crate::search::params::mg_king_defender_bonus(),
                            crate::search::params::eg_king_defender_bonus(),
                        );
                    }
                    break; // Count once
                }
            }
        }

        let is_attacking_piece = matches!(
            pt,
            PieceType::Rook
                | PieceType::Queen
                | PieceType::RoyalQueen
                | PieceType::Bishop
                | PieceType::Chancellor
                | PieceType::Archbishop
                | PieceType::Amazon
        );
        if is_attacking_piece {
            let scale = if piece.color() == PlayerColor::White {
                white_attack_ready
            } else {
                black_attack_ready
            };
            piece_score = piece_score * scale / 100;
        }

        // Riders converge toward their confined worth and leapers are paid the
        // difference. Both ctx are 0 on any unbounded board, so one test skips
        // the piece-value lookup entirely there.
        if (metrics.slider_geometry_ctx | metrics.leaper_geometry_ctx) != 0 && !pt.is_royal() {
            let ctx_pct = geometry_ctx_pct(pt);
            if ctx_pct != 0 {
                let ctx = if ctx_pct < 0 {
                    metrics.slider_geometry_ctx
                } else {
                    metrics.leaper_geometry_ctx
                };
                piece_score += get_piece_value_base(pt) * ctx_pct * ctx / 10000;
            }
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
        w_pair_bonus += taper(mg_bishop_pair_bonus(), eg_bishop_pair_bonus());
        bump_feat!(bishop_pair_bonus, 1);
        if metrics.white_bishop_colors.0 && metrics.white_bishop_colors.1 {
            w_pair_bonus += 20;
        }
    }
    if metrics.black_bishops >= 2 {
        b_pair_bonus += taper(mg_bishop_pair_bonus(), eg_bishop_pair_bonus());
        bump_feat!(bishop_pair_bonus, -1);
        if metrics.black_bishop_colors.0 && metrics.black_bishop_colors.1 {
            b_pair_bonus += 20;
        }
    }

    tracer.record("Piece: Activity", w_activity, b_activity);
    tracer.record("Piece: Bishop Pair", w_pair_bonus, b_pair_bonus);

    (w_activity + w_pair_bonus) - (b_activity + b_pair_bonus)
}

pub struct RoyalTropismMetrics {
    piece_type: PieceType,
    tropism_addend: i32,
    attacking_units: i32,
    defender_units: i32,
    defender_units_in_distance: [i32; 8],
    x: i64,
    y: i64,
}

/// Calculates the right distance to attack the other side and defend your own king.
#[inline(always)]
fn compute_tropism_addend(slider_count: i32) -> i32 {
    12 - slider_count.clamp(1, 7)
}

/// Threat bonus for the piece a slider meets at one end of one of its lines.
#[inline]
fn slider_threat_bonus(end: Option<(i64, u8)>, own: PlayerColor, piece_val: i32) -> i32 {
    let Some((_, packed)) = end else {
        return 0;
    };
    let victim = Piece::from_packed(packed);
    let vt = victim.piece_type();
    if victim.color() == own || vt.is_royal() || vt.is_neutral_type() {
        return 0;
    }
    let gain = get_piece_value_base(vt) - piece_val;
    if gain <= 0 {
        return 0;
    }
    (gain / slider_threat_div()).min(slider_threat_cap())
}

/// Saturating i64->i32 cast for a (non-negative) Chebyshev distance.
#[inline(always)]
fn saturating_dist_i32(d: i64) -> i32 {
    d.min(i32::MAX as i64) as i32
}

/// One piece's king-tropism contribution: `numerator / (chebyshev_dist + addend)`.
#[inline(always)]
fn tropism_contribution(numerator: i32, d: i64, addend: i32) -> i32 {
    let denom = saturating_dist_i32(d).saturating_add(addend).max(1);
    numerator / denom
}

/// A ray toward a king is attack-relevant when it's open past `open_dist`, or
/// its nearest piece is already an attacker slider that moves along it.
#[inline]
fn ray_pressured(
    ray: &(i32, i32, PlayerColor, PieceType),
    attacker: PlayerColor,
    slider_mask: u32,
    open_dist: i32,
) -> bool {
    ray.0 > open_dist || (ray.2 == attacker && (1u32 << (ray.3 as u8)) & slider_mask != 0)
}

fn compute_attack_readiness_optimized(
    enemy_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    has_enemy_king: bool,
    sliders_in_zone: i32,
    attacker: PlayerColor,
) -> i32 {
    if !has_enemy_king {
        return 50;
    }

    // 1. Count pressured rays around enemy king (O(1))
    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;

    for ray in &enemy_king_rays[0..4] {
        if ray_pressured(ray, attacker, crate::attacks::DIAG_MASK, 6) {
            open_diag_rays += 1;
        }
    }
    for ray in &enemy_king_rays[4..8] {
        if ray_pressured(ray, attacker, crate::attacks::ORTHO_MASK, 6) {
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
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    phase: i32,
    tracer: &mut T,
    metrics: &KingSafetyMetrics,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    w_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    b_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    w_ring_covered: bool,
    b_ring_covered: bool,
    style: EvalStyle,
) -> i32 {
    let mut w_safety: i32 = 0;
    let mut b_safety: i32 = 0;
    let mut w_attack: i32 = 0;
    let mut b_attack: i32 = 0;

    // Defense penalty (Shelter)
    for &wk in white_royals {
        w_safety += evaluate_king_shelter(
            game,
            &wk,
            PlayerColor::White,
            phase,
            metrics.urgency.0,
            metrics.has_enemy_queen.0,
            white_pawns,
            w_king_rays,
            w_ring_covered,
        );
    }
    for &bk in black_royals {
        b_safety += evaluate_king_shelter(
            game,
            &bk,
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
    if !black_royals.is_empty() {
        // White attacks Black
        w_attack += compute_attack_bonus_optimized(
            b_king_rays,
            metrics.white_slider_counts,
            PlayerColor::White,
        );
    }
    if !white_royals.is_empty() {
        // Black attacks White
        b_attack += compute_attack_bonus_optimized(
            w_king_rays,
            metrics.black_slider_counts,
            PlayerColor::Black,
        );
    }

    // Under AllRoyalsCaptured losing one royal is survivable, so the reduction is
    // moderate. Each side's multiplier gates both its own aggression and the
    // opponent's safety concern.
    let white_rc_mult = match game.game_rules.white_win_condition {
        WinCondition::AllRoyalsCaptured => 65,
        _ => 100,
    };
    let black_rc_mult = match game.game_rules.black_win_condition {
        WinCondition::AllRoyalsCaptured => 65,
        _ => 100,
    };

    // Shelter is the defensive half and the attack bonus the offensive half, so the
    // style weights them apart: a weak level over-values sitting behind its own
    // pawns and barely notices a king it could be attacking — or one being attacked.
    let w_shelter = style.defense(w_safety * black_rc_mult / 100);
    let b_shelter = style.defense(b_safety * white_rc_mult / 100);
    let w_pressure = style.attack(w_attack * white_rc_mult / 100);
    let b_pressure = style.attack(b_attack * black_rc_mult / 100);

    let w_total = w_shelter + w_pressure;
    let b_total = b_shelter + b_pressure;

    tracer.record("King: Shelter", w_shelter, b_shelter);
    tracer.record("King: Attack", w_pressure, b_pressure);

    w_total - b_total
}

/// Ray-based attack bonus: open rays toward enemy king with slider presence.
fn compute_attack_bonus_optimized(
    enemy_king_rays: &[(i32, i32, PlayerColor, PieceType); 8],
    slider_counts: (i32, i32), // (diag, ortho)
    attacker: PlayerColor,
) -> i32 {
    let (our_diag_count, our_ortho_count) = slider_counts;
    if our_diag_count == 0 && our_ortho_count == 0 {
        return 0;
    }

    let mut open_diag_rays = 0;
    let mut open_ortho_rays = 0;

    if our_diag_count > 0 {
        for ray in &enemy_king_rays[0..4] {
            if ray_pressured(ray, attacker, crate::attacks::DIAG_MASK, 5) {
                open_diag_rays += 1;
            }
        }
    }
    if our_ortho_count > 0 {
        for ray in &enemy_king_rays[4..8] {
            if ray_pressured(ray, attacker, crate::attacks::ORTHO_MASK, 5) {
                open_ortho_rays += 1;
            }
        }
    }

    const ATTACK_BONUS_PER_OPEN_RAY: i32 = 12;
    let diag_bonus = if our_diag_count > 0 && open_diag_rays > 0 {
        let mult = 100 + (our_diag_count - 1).max(0) * 25;
        open_diag_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    let ortho_bonus = if our_ortho_count > 0 && open_ortho_rays > 0 {
        let mult = 110 + (our_ortho_count - 1).max(0) * 30;
        open_ortho_rays * ATTACK_BONUS_PER_OPEN_RAY * mult / 100
    } else {
        0
    };

    diag_bonus + ortho_bonus
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_rook(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut bonus: i32 = 0;

    // Scale king-targeting bonuses based on own win condition.
    // AllRoyalsCaptured: partial focus on enemy king.
    let own_win_cond = if color == PlayerColor::White {
        game.game_rules.white_win_condition
    } else {
        game.game_rules.black_win_condition
    };
    let king_mult = match own_win_cond {
        WinCondition::AllRoyalsCaptured => 70,
        _ => 100,
    };

    let enemy_royals = if color == PlayerColor::White {
        black_royals
    } else {
        white_royals
    };

    let mut king_bonus: i32 = 0;

    for &ek in enemy_royals {
        // Behind enemy king along the rank direction.
        if (color == PlayerColor::White && y > ek.y) || (color == PlayerColor::Black && y < ek.y) {
            king_bonus += taper(
                crate::search::params::mg_behind_king_bonus(),
                crate::search::params::eg_behind_king_bonus(),
            );
            break;
        }
    }

    for &ek in enemy_royals {
        // On same or adjacent file to enemy king: strong attacking potential.
        if (x - ek.x).abs() <= 1 {
            king_bonus += 50;
            break;
        }
    }

    for &ek in enemy_royals {
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

        king_bonus += confinement_bonus;
        if confinement_bonus > 0 {
            break;
        }
    }

    for &ek in enemy_royals {
        // Simplified slider coordination - just count nearby sliders without iteration
        if (x - ek.x).abs() <= 4 && (y - ek.y).abs() <= 4 {
            // This rook is close to king, assume some coordination exists
            king_bonus += slider_net_bonus() / 2;
            break;
        }
    }

    bonus += king_bonus * king_mult / 100;

    // Penalize rooks that have drifted very far from the king zone
    let mut min_cheb = i64::MAX;
    for &ek in enemy_royals {
        min_cheb = min_cheb.min((x - ek.x).abs().max((y - ek.y).abs()));
    }
    let own_royals = if color == PlayerColor::White {
        white_royals
    } else {
        black_royals
    };
    for &ok in own_royals {
        min_cheb = min_cheb.min((x - ok.x).abs().max((y - ok.y).abs()));
    }

    if min_cheb != i64::MAX && min_cheb > far_slider_cheb_radius() as i64 {
        let excess = (min_cheb - far_slider_cheb_radius() as i64)
            .min(far_slider_cheb_max_excess() as i64) as i32;
        bonus -= excess * far_rook_penalty();
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
            bonus += rook_open_file_bonus();
        } else {
            bonus += rook_semi_open_file_bonus();
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
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut bonus: i32 = 0;

    // Scale king-targeting bonuses based on own win condition.
    let own_win_cond = if color == PlayerColor::White {
        game.game_rules.white_win_condition
    } else {
        game.game_rules.black_win_condition
    };
    let king_mult = match own_win_cond {
        WinCondition::AllRoyalsCaptured => 70,
        _ => 100,
    };

    // Queen should aggressively aim at the enemy king from a safe distance.
    let enemy_royals = if color == PlayerColor::White {
        black_royals
    } else {
        white_royals
    };

    let from = Coordinate { x, y };
    for ek in enemy_royals {
        let dx = ek.x - x;
        let dy = ek.y - y;
        let same_file = dx == 0;
        let same_rank = dy == 0;
        let same_diag = dx.abs() == dy.abs();

        if (same_file || same_rank || same_diag)
            && is_clear_line_between_fast(&game.spatial_indices, &from, ek)
        {
            let mut line_bonus = 15;
            let lin_dist = saturating_dist_i32(dx.abs().max(dy.abs()));
            let max_lin = 20;
            let clamped = lin_dist.min(max_lin);
            let diff = (clamped - queen_ideal_line_dist()).abs();
            let base = (max_lin - diff * 2).max(0);
            line_bonus += base
                * (taper(
                    crate::search::params::mg_king_tropism_bonus(),
                    crate::search::params::eg_king_tropism_bonus(),
                ) / 2)
                    .max(1);
            let line_bonus = line_bonus
                + if (color == PlayerColor::White && y > ek.y)
                    || (color == PlayerColor::Black && y < ek.y)
                {
                    10
                } else {
                    0
                };
            bonus += line_bonus * king_mult / 100;
            break;
        }
    }

    let mut min_cheb = i64::MAX;
    for ek in enemy_royals {
        min_cheb = min_cheb.min((x - ek.x).abs().max((y - ek.y).abs()));
    }
    let own_royals = if color == PlayerColor::White {
        white_royals
    } else {
        black_royals
    };
    for &ok in own_royals {
        min_cheb = min_cheb.min((x - ok.x).abs().max((y - ok.y).abs()));
    }

    if min_cheb != i64::MAX && min_cheb > far_slider_cheb_radius() as i64 {
        let excess = (min_cheb - far_slider_cheb_radius() as i64)
            .min(far_slider_cheb_max_excess() as i64) as i32;
        bonus -= excess * far_queen_penalty();
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
            bonus += queen_open_file_bonus();
        } else {
            bonus += queen_semi_open_file_bonus();
        }
    }

    bonus
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_bishop(
    game: &GameState,
    x: i64,
    y: i64,
    color: PlayerColor,
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    phase: i32,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut bonus: i32 = 0;

    // Scale king-targeting bonuses based on own win condition.
    let own_win_cond = if color == PlayerColor::White {
        game.game_rules.white_win_condition
    } else {
        game.game_rules.black_win_condition
    };
    let king_mult = match own_win_cond {
        WinCondition::AllRoyalsCaptured => 70,
        _ => 100,
    };

    // Long diagonal control bonus. A diagonal at a fixed absolute position means
    // nothing on an unbounded board and is not colour-symmetric, so anchor both to
    // the kings' midpoint. Doubled units keep a half-square midpoint exact.
    let (rx2, ry2) = match (white_royals.first(), black_royals.first()) {
        (Some(wk), Some(bk)) => (wk.x + bk.x, wk.y + bk.y),
        (Some(k), None) | (None, Some(k)) => (2 * k.x, 2 * k.y),
        (None, None) => (0, 0),
    };
    if (2 * (x - y) - (rx2 - ry2)).abs() <= 2 || (2 * (x + y) - (rx2 + ry2)).abs() <= 2 {
        bonus += 8;
    }

    // Behind enemy king bonus and bishop tropism.
    let enemy_royals = if color == PlayerColor::White {
        black_royals
    } else {
        white_royals
    };

    for &ek in enemy_royals {
        // Bishop behind enemy king along the rank direction (less direct than rook/queen).
        if (color == PlayerColor::White && y > ek.y) || (color == PlayerColor::Black && y < ek.y) {
            bonus += taper(
                crate::search::params::mg_behind_king_bonus(),
                crate::search::params::eg_behind_king_bonus(),
            ) / 2
                * king_mult
                / 100;
            break;
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

    let has_left_support = my_pawns.binary_search(&(x - 1, support_y)).is_ok();

    let has_right_support = my_pawns.binary_search(&(x + 1, support_y)).is_ok();

    if has_left_support || has_right_support {
        bonus += taper(mg_outpost_bonus(), eg_outpost_bonus());
    }

    bonus
}

fn evaluate_knight(
    x: i64,
    y: i64,
    color: PlayerColor,
    cloud_center: Option<&Coordinate>,
    cloud_avg_spread: i32,
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
        PieceType::Knight,
        cloud_avg_spread,
        phase,
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

    let has_left_support = my_pawns.binary_search(&(x - 1, support_y)).is_ok();

    let has_right_support = my_pawns.binary_search(&(x + 1, support_y)).is_ok();

    if has_left_support || has_right_support {
        bonus += taper(mg_outpost_bonus(), eg_outpost_bonus());
    }

    bonus
}

fn evaluate_leaper_positioning(
    x: i64,
    y: i64,
    _color: PlayerColor,
    cloud_center: Option<&Coordinate>,
    piece_type: PieceType,
    cloud_avg_spread: i32,
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let piece_value = get_piece_value_base(piece_type);
    let mut bonus: i32 = 0;

    // 1. CLOUD PROXIMITY: reward being near the piece cloud center
    let scale = (piece_value / leaper_tropism_divisor()).max(1);
    if let Some(center) = cloud_center {
        let dist = (2 * x - center.x).abs().max((2 * y - center.y).abs()) / 2;
        if dist <= 10 {
            bonus += (11 - dist as i32) * (scale / 3).max(1);
        }
    }

    // Spread runs 0..=cloud_center_max_skew_dist() as i64 and is neutral at 8, so a positive
    // density_adj means a clustered position and a leaper bonus.
    let density_sensitivity: i32 = match piece_type {
        PieceType::Knight => 35,
        PieceType::Camel => 30,
        PieceType::Zebra => 25,
        PieceType::Giraffe => 20,
        PieceType::Guard => 25,
        PieceType::Hawk => 15,
        PieceType::Centaur | PieceType::RoyalCentaur => 22,
        PieceType::Huygen => 15,
        PieceType::Rose => 5,
        _ => 10,
    };
    let density_adj = (8_i32 - cloud_avg_spread).clamp(-8, 8);
    bonus += density_adj * density_sensitivity / 10;

    // 3. PHASE TAPER
    let (mg_bonus, eg_bonus): (i32, i32) = match piece_type {
        PieceType::Knight => (0, 30),
        PieceType::Camel => (0, 23),
        PieceType::Zebra => (0, 20),
        PieceType::Giraffe => (0, 15),
        PieceType::Guard => (0, 20),
        PieceType::Hawk => (0, 10),
        PieceType::Centaur | PieceType::RoyalCentaur => (0, 20),
        PieceType::Huygen => (0, 0),
        PieceType::Rose => (0, 0),
        _ => (0, 10),
    };
    bonus += taper(mg_bonus, eg_bonus);

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
        safety -= taper(
            crate::search::params::mg_king_ring_missing_penalty(),
            crate::search::params::eg_king_ring_missing_penalty(),
        );
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
                if py > king.y && py - king.y <= king_shield_ahead_max_dist() as i64 {
                    has_pawn_ahead = true;
                } else if py < king.y {
                    has_pawn_behind = true;
                }
            } else if py < king.y && king.y - py <= king_shield_ahead_max_dist() as i64 {
                has_pawn_ahead = true;
            } else if py > king.y {
                has_pawn_behind = true;
            }
            k += 1;
        }

        // King on Open File Penalty (No friendly pawns on file)
        if dx == 0 && on_file_count == 0 {
            safety -= taper(
                crate::search::params::mg_king_open_file_penalty(),
                crate::search::params::eg_king_open_file_penalty(),
            );
        }
    }

    // A pawn ahead shelters the king regardless of any pawn behind it; only the
    // absence of a forward pawn (with one behind) draws the penalty.
    if has_pawn_ahead {
        safety += taper(
            crate::search::params::mg_king_pawn_shield_bonus(),
            crate::search::params::eg_king_pawn_shield_bonus(),
        );
    } else if has_pawn_behind {
        safety -= taper(mg_king_pawn_ahead_penalty(), eg_king_pawn_ahead_penalty());
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
        let mut enemy_slider_aligned = false;

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
                // A diagonal slider on this ray is an actual attacker, not a shield.
                enemy_slider_aligned = (1u32 << (pt as u8)) & crate::attacks::DIAG_MASK != 0;
            }
        }

        let mut penalty = BASE_DIAG_RAY_PENALTY;
        if let Some((v, d)) = blocker {
            penalty = penalty * (100 - blocker_reduction_pct(v, d)) / 100;
        } else if enemy_blocked && !enemy_slider_aligned {
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
        let mut enemy_slider_aligned = false;

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
                // An orthogonal slider on this ray is an actual attacker, not a shield.
                enemy_slider_aligned = (1u32 << (pt as u8)) & crate::attacks::ORTHO_MASK != 0;
            }
        }

        let mut penalty = BASE_ORTHO_RAY_PENALTY;
        if let Some((v, d)) = blocker {
            penalty = penalty * (100 - blocker_reduction_pct(v, d)) / 100;
        } else if enemy_blocked && !enemy_slider_aligned {
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
    // Quarter-slope past 400 instead of a hard cap: a flat cap zeroes the
    // danger gradient exactly where attacks escalate, so safety could never
    // veto a piece grab once the cap was hit. Identical below 400.
    let capped = if final_penalty > 400 {
        (400 + (final_penalty - 400) / 4).min(800)
    } else {
        final_penalty
    };
    safety -= capped;

    safety
}

pub fn evaluate_pawn_structure(game: &GameState) -> i32 {
    let phase = effective_phase(game.total_phase, game.initial_phase);
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
                            }
                        }
                    }
                    wp.sort_unstable();
                    bp.sort_unstable();

                    evaluate_pawn_structure_traced(
                        game,
                        phase,
                        game.white_royals.as_slice(),
                        game.black_royals.as_slice(),
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
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    tracer: &mut T,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    white_rq: &[(i64, i64)],
    black_rq: &[(i64, i64)],
) -> i32 {
    let _ = (white_rq, black_rq);
    let pawn_hash = game.pawn_hash;
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };

    // Bypassing cache if tracer is active to ensure we get a full breakdown.
    if tracer.is_active() {
        let core = compute_pawn_core(game, phase, tracer, white_pawns, black_pawns);
        return taper(core.mg, core.eg)
            + score_passed_pawns(
                game,
                phase,
                white_royals,
                black_royals,
                white_pawns,
                black_pawns,
                &core.w_passed,
                &core.b_passed,
                tracer,
            );
    }

    // Fast 2-Bucket cache probe using bitwise mask. Only pawn-hash-pure data is
    // cached; taper and passed-pawn scoring happen live so phase, king positions
    // and blockers are always current.
    let idx = (pawn_hash as usize) & (PAWN_CACHE_SIZE - 1);
    let cached = PAWN_CACHE.with(|cache| {
        let bucket = unsafe { &(&*cache.get())[idx] };
        if bucket.entries[0].hash == pawn_hash {
            Some(bucket.entries[0].clone())
        } else if bucket.entries[1].hash == pawn_hash {
            Some(bucket.entries[1].clone())
        } else {
            None
        }
    });

    if let Some(entry) = cached {
        return taper(entry.mg, entry.eg)
            + score_passed_pawns(
                game,
                phase,
                white_royals,
                black_royals,
                white_pawns,
                black_pawns,
                &entry.w_passed,
                &entry.b_passed,
                tracer,
            );
    }

    // Cache miss - compute pawn structure
    let core = compute_pawn_core(game, phase, tracer, white_pawns, black_pawns);
    let score = taper(core.mg, core.eg)
        + score_passed_pawns(
            game,
            phase,
            white_royals,
            black_royals,
            white_pawns,
            black_pawns,
            &core.w_passed,
            &core.b_passed,
            tracer,
        );

    // 2-Bucket cache store (LRU: new item goes to front, old item moves to back)
    PAWN_CACHE.with(|cache| {
        let cache_mut = unsafe { &mut *cache.get() };
        let bucket = &mut cache_mut[idx];
        bucket.entries[1] = bucket.entries[0].clone();
        bucket.entries[0] = PawnCacheEntry {
            hash: pawn_hash,
            mg: core.mg,
            eg: core.eg,
            w_passed: core.w_passed,
            b_passed: core.b_passed,
        };
    });

    score
}

/// Pawn-hash-pure structure terms (untapered) plus passed-pawn locations.
struct PawnCoreOut {
    mg: i32,
    eg: i32,
    w_passed: SmallVec<[(i64, i64); 4]>,
    b_passed: SmallVec<[(i64, i64); 4]>,
}

/// Computes the cacheable pawn terms: doubled, isolated, backward, candidate,
/// connected, and passed-pawn detection. Passed-pawn scoring is done live.
fn compute_pawn_core<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    tracer: &mut T,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> PawnCoreOut {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_doubled = (0, 0);
    let mut b_doubled = (0, 0);
    let mut w_connected = (0, 0);
    let mut b_connected = (0, 0);
    let mut w_candidate = (0, 0);
    let mut b_candidate = (0, 0);
    let mut w_isolated = (0, 0);
    let mut b_isolated = (0, 0);
    let mut w_backward = (0, 0);
    let mut b_backward = (0, 0);
    let mut w_passed: SmallVec<[(i64, i64); 4]> = SmallVec::new();
    let mut b_passed: SmallVec<[(i64, i64); 4]> = SmallVec::new();

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
            w_doubled.0 -= (count - 1) * mg_doubled_pawn_penalty();
            w_doubled.1 -= (count - 1) * eg_doubled_pawn_penalty();
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
            b_doubled.0 -= (count - 1) * mg_doubled_pawn_penalty();
            b_doubled.1 -= (count - 1) * eg_doubled_pawn_penalty();
        }
        i = j;
    }

    // White Pawns: Passed, Candidate, Connected, Isolated, Backward
    for &(wx, wy) in white_pawns {
        let mut is_passed = true;
        let mut is_candidate = false;
        let mut stoppers = 0;

        // Structure checks
        let left_idx = white_pawns.partition_point(|&(x, _)| x < wx - 1);
        let has_left_neighbor = left_idx < white_pawns.len() && white_pawns[left_idx].0 == wx - 1;

        let right_idx = white_pawns.partition_point(|&(x, _)| x < wx + 1);
        let has_right_neighbor =
            right_idx < white_pawns.len() && white_pawns[right_idx].0 == wx + 1;

        if !has_left_neighbor && !has_right_neighbor {
            w_isolated.0 -= 10;
            w_isolated.1 -= 20;
        } else {
            let is_behind_left = !has_left_neighbor || white_pawns[left_idx].1 > wy;
            let is_behind_right = !has_right_neighbor || white_pawns[right_idx].1 > wy;

            if is_behind_left && is_behind_right {
                let stop_sq_blocked = game.board.is_occupied(wx, wy + 1);
                let stop_sq_attacked = black_pawns.binary_search(&(wx - 1, wy + 2)).is_ok()
                    || black_pawns.binary_search(&(wx + 1, wy + 2)).is_ok();

                if stop_sq_blocked || stop_sq_attacked {
                    w_backward.0 -= 8;
                    w_backward.1 -= 12;
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
                }
                k += 1;
            }
        }

        if is_passed {
            w_passed.push((wx, wy));
        } else {
            // Candidate passer: not passed, but pushing it reaches a square our side
            // defends at least as heavily as the enemy attacks, so the pawn can force
            // a passer by trading.
            let can_advance = !game.board.is_occupied(wx, wy + 1);
            let push_support = white_pawns.binary_search(&(wx - 1, wy)).is_ok() as i32
                + white_pawns.binary_search(&(wx + 1, wy)).is_ok() as i32;
            let push_threats = black_pawns.binary_search(&(wx - 1, wy + 2)).is_ok() as i32
                + black_pawns.binary_search(&(wx + 1, wy + 2)).is_ok() as i32;
            if stoppers > 0 && can_advance && push_support >= push_threats {
                is_candidate = true;
            }

            if is_candidate {
                w_candidate.0 += candidate_passer_bonus()[rel_rank];
                w_candidate.1 += candidate_passer_bonus()[rel_rank];
            }
        }

        // Connectivity (passed pawns get a 3/2 boost)
        if white_pawns.binary_search(&(wx - 1, wy - 1)).is_ok()
            || white_pawns.binary_search(&(wx + 1, wy - 1)).is_ok()
        {
            if is_passed {
                w_connected.0 += (crate::search::params::mg_connected_pawn_bonus() * 3) / 2;
                w_connected.1 += (crate::search::params::eg_connected_pawn_bonus() * 3) / 2;
            } else {
                w_connected.0 += crate::search::params::mg_connected_pawn_bonus();
                w_connected.1 += crate::search::params::eg_connected_pawn_bonus();
            }
        }
    }

    // Black Pawns: Passed, Candidate, Connected, Isolated, Backward
    for &(bx, by) in black_pawns {
        let mut is_passed = true;
        let mut is_candidate = false;
        let mut stoppers = 0;

        // Structure checks
        let left_idx = black_pawns.partition_point(|&(x, _)| x < bx - 1);
        let has_left_neighbor = left_idx < black_pawns.len() && black_pawns[left_idx].0 == bx - 1;

        let right_idx = black_pawns.partition_point(|&(x, _)| x < bx + 1);
        let has_right_neighbor =
            right_idx < black_pawns.len() && black_pawns[right_idx].0 == bx + 1;

        if !has_left_neighbor && !has_right_neighbor {
            b_isolated.0 -= 10;
            b_isolated.1 -= 20;
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
                    b_backward.0 -= 8;
                    b_backward.1 -= 12;
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
                }
                k += 1;
            }
        }

        if is_passed {
            b_passed.push((bx, by));
        } else {
            // Candidate passer at the push square (see white candidate branch).
            let can_advance = !game.board.is_occupied(bx, by - 1);
            let push_support = black_pawns.binary_search(&(bx - 1, by)).is_ok() as i32
                + black_pawns.binary_search(&(bx + 1, by)).is_ok() as i32;
            let push_threats = white_pawns.binary_search(&(bx - 1, by - 2)).is_ok() as i32
                + white_pawns.binary_search(&(bx + 1, by - 2)).is_ok() as i32;
            if stoppers > 0 && can_advance && push_support >= push_threats {
                is_candidate = true;
            }
            if is_candidate {
                b_candidate.0 += candidate_passer_bonus()[rel_rank];
                b_candidate.1 += candidate_passer_bonus()[rel_rank];
            }
        }

        // Connectivity (passed pawns get a 3/2 boost)
        if black_pawns.binary_search(&(bx - 1, by + 1)).is_ok()
            || black_pawns.binary_search(&(bx + 1, by + 1)).is_ok()
        {
            if is_passed {
                b_connected.0 += (crate::search::params::mg_connected_pawn_bonus() * 3) / 2;
                b_connected.1 += (crate::search::params::eg_connected_pawn_bonus() * 3) / 2;
            } else {
                b_connected.0 += crate::search::params::mg_connected_pawn_bonus();
                b_connected.1 += crate::search::params::eg_connected_pawn_bonus();
            }
        }
    }

    if tracer.is_active() {
        tracer.record(
            "Pawn: Doubled",
            taper(w_doubled.0, w_doubled.1),
            taper(b_doubled.0, b_doubled.1),
        );
        tracer.record(
            "Pawn: Candidate",
            taper(w_candidate.0, w_candidate.1),
            taper(b_candidate.0, b_candidate.1),
        );
        tracer.record(
            "Pawn: Connected",
            taper(w_connected.0, w_connected.1),
            taper(b_connected.0, b_connected.1),
        );
        tracer.record(
            "Pawn: Isolated",
            taper(w_isolated.0, w_isolated.1),
            taper(b_isolated.0, b_isolated.1),
        );
        tracer.record(
            "Pawn: Backward",
            taper(w_backward.0, w_backward.1),
            taper(b_backward.0, b_backward.1),
        );
    }

    PawnCoreOut {
        mg: (w_doubled.0 + w_candidate.0 + w_connected.0 + w_isolated.0 + w_backward.0)
            - (b_doubled.0 + b_candidate.0 + b_connected.0 + b_isolated.0 + b_backward.0),
        eg: (w_doubled.1 + w_candidate.1 + w_connected.1 + w_isolated.1 + w_backward.1)
            - (b_doubled.1 + b_candidate.1 + b_connected.1 + b_isolated.1 + b_backward.1),
        w_passed,
        b_passed,
    }
}

/// Scores passed pawns live (never cached): king distances, blockers and the
/// promotion path change every move and must stay current for conversion play.
#[allow(clippy::too_many_arguments)]
fn score_passed_pawns<T: EvaluationTracer>(
    game: &GameState,
    phase: i32,
    white_royals: &[Coordinate],
    black_royals: &[Coordinate],
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
    w_passed: &[(i64, i64)],
    b_passed: &[(i64, i64)],
    tracer: &mut T,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut w_passed_score = 0;
    let mut b_passed_score = 0;

    for &(wx, wy) in w_passed {
        let w_promo = game.white_promo_rank;
        let dist_to_promo = (w_promo - wy).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;

        // 1. Can Advance
        let next_y = wy + 1;
        let can_advance = !game.board.is_occupied(wx, next_y);

        // 2. Safe Advance
        let safe_advance = black_pawns.binary_search(&(wx - 1, next_y + 1)).is_err()
            && black_pawns.binary_search(&(wx + 1, next_y + 1)).is_err();

        // 3. King Distances (find max bonus across all royals)
        let mut friendly_king_bonus = 0;
        let mut enemy_king_penalty = 0;
        for wk in white_royals {
            let d = (wx - wk.x).abs().max((wy - wk.y).abs()) as usize;
            let b = passed_friendly_king_dist()[rel_rank] * (7 - d.min(7)) as i32;
            friendly_king_bonus = friendly_king_bonus.max(b);
        }
        for bk in black_royals {
            let d = (wx - bk.x).abs().max((wy - bk.y).abs()) as usize;
            let p = passed_enemy_king_dist()[rel_rank] * (7 - d.min(7)) as i32;
            enemy_king_penalty = enemy_king_penalty.max(p);
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
                    && black_pawns[start].1 <= w_promo.saturating_add(1)
                {
                    safe_path = false;
                    break;
                }
            }
        }
        let safe_path_bonus = if safe_path {
            taper(
                crate::search::params::mg_passed_safe_path_bonus(),
                crate::search::params::eg_passed_safe_path_bonus(),
            )
        } else {
            0
        };

        let base_bonus =
            passed_pawn_adv_bonus()[can_advance as usize][safe_advance as usize][rel_rank];
        w_passed_score += base_bonus + friendly_king_bonus - enemy_king_penalty + safe_path_bonus;
    }

    for &(bx, by) in b_passed {
        let b_promo = game.black_promo_rank;
        let dist_to_promo = (by - b_promo).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;

        let next_y = by - 1;
        let can_advance = !game.board.is_occupied(bx, next_y);
        let safe_advance = white_pawns.binary_search(&(bx - 1, next_y - 1)).is_err()
            && white_pawns.binary_search(&(bx + 1, next_y - 1)).is_err();

        let mut friendly_king_bonus = 0;
        let mut enemy_king_penalty = 0;
        for bk in black_royals {
            let d = (bx - bk.x).abs().max((by - bk.y).abs()) as usize;
            let b = passed_friendly_king_dist()[rel_rank] * (7 - d.min(7)) as i32;
            friendly_king_bonus = friendly_king_bonus.max(b);
        }
        for wk in white_royals {
            let d = (bx - wk.x).abs().max((by - wk.y).abs()) as usize;
            let p = passed_enemy_king_dist()[rel_rank] * (7 - d.min(7)) as i32;
            enemy_king_penalty = enemy_king_penalty.max(p);
        }

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
            taper(
                crate::search::params::mg_passed_safe_path_bonus(),
                crate::search::params::eg_passed_safe_path_bonus(),
            )
        } else {
            0
        };

        let base_bonus =
            passed_pawn_adv_bonus()[can_advance as usize][safe_advance as usize][rel_rank];
        b_passed_score += base_bonus + friendly_king_bonus - enemy_king_penalty + safe_path_bonus;
    }

    if tracer.is_active() {
        tracer.record("Pawn: Passed", w_passed_score, b_passed_score);
    }

    w_passed_score - b_passed_score
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

/// Evaluates how close the king is to the pawns which is very important in endgames.
pub fn evaluate_king_positioning_traced<T: EvaluationTracer>(
    game: &GameState,
    king_phase: i32,
    white_royals: &[RoyalTropismMetrics],
    black_royals: &[RoyalTropismMetrics],
    tracer: &mut T,
    white_pawns: &[(i64, i64)],
    black_pawns: &[(i64, i64)],
) -> i32 {
    let mut w_activity = 0;
    let mut b_activity = 0;
    let mut nearest_pawn_distance = 255; // arbitrary number

    // King distances for each pawn (friendly_dist, enemy_dist))
    let mut w_king_pawn_distances = SmallVec::<[(i32, i32); 64]>::new();
    let mut b_king_pawn_distances = SmallVec::<[(i32, i32); 64]>::new();

    // Find the closes distance from each pawn to the kings.
    for &(wx, wy) in white_pawns {
        let w_promo = game.white_promo_rank;
        let dist_to_promo = (w_promo - wy).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;
        let mut min_d = 255; // Chebyshev distance

        // Manhattan distance
        let mut min_friendly_md = 255;
        let mut min_enemy_md = 255;

        // First, check if it is a passed pawn
        let mut is_passed = true;
        for dx in -1..=1 {
            let target_file = wx + dx;

            // Check for enemy pawns blocking or on adjacent files
            let start = black_pawns.partition_point(|&(bx, _)| bx < target_file);
            let mut k = start;
            while k < black_pawns.len() && black_pawns[k].0 == target_file {
                let by = black_pawns[k].1;
                if by > wy {
                    is_passed = false;
                }
                k += 1;
            }
        }

        // Apply weights
        let weight = if is_passed {
            3
        } else if white_pawns.binary_search(&(wx - 1, wy - 1)).is_ok()
            || white_pawns.binary_search(&(wx + 1, wy - 1)).is_ok()
        {
            2
        } else {
            // base of the pawn chain
            3
        };

        // Find the nearest king to the pawn.
        for wk in white_royals {
            if matches!(wk.piece_type, PieceType::King) {
                let d = saturating_dist_i32((wx - wk.x).abs().max((wy - wk.y).abs()));
                min_d = min_d.min(d);
                let md = saturating_dist_i32((wx - wk.x).abs() + (wy - wk.y).abs());
                min_friendly_md = min_friendly_md.min(md);
            }
        }
        let near_friendly_king_bonus = pawn_friendly_king_dist()[rel_rank] * (7 - min_d.min(7));

        min_d = 255; // reset it back
        for bk in black_royals {
            if matches!(bk.piece_type, PieceType::King) {
                let d = saturating_dist_i32((wx - bk.x).abs().max((wy - bk.y).abs()));
                min_d = min_d.min(d);
                let md = saturating_dist_i32((wx - bk.x).abs() + (wy - bk.y).abs());
                min_enemy_md = min_enemy_md.min(md);
            }
        }
        let near_enemy_king_penalty = pawn_enemy_king_dist()[rel_rank] * (7 - min_d.min(7));

        w_king_pawn_distances.push((min_friendly_md, min_enemy_md));
        nearest_pawn_distance = nearest_pawn_distance.min(min_friendly_md).min(min_enemy_md);
        w_activity += (near_friendly_king_bonus - near_enemy_king_penalty) * weight / 3;
    }
    for &(bx, by) in black_pawns {
        let b_promo = game.black_promo_rank;
        let dist_to_promo = (by - b_promo).max(1);
        let rel_rank = (6 - dist_to_promo).clamp(0, 5) as usize;
        let mut min_d = 255; // Chebyshev distance

        // Manhattan distance
        let mut min_friendly_md = 255;
        let mut min_enemy_md = 255;

        // First, check if it is a passed pawn
        let mut is_passed = true;
        for dx in -1..=1 {
            let target_file = bx + dx;
            let start = white_pawns.partition_point(|&(wx, _)| wx < target_file);
            let mut k = start;
            while k < white_pawns.len() && white_pawns[k].0 == target_file {
                let wy = white_pawns[k].1;
                if wy < by {
                    is_passed = false;
                }
                k += 1;
            }
        }

        // Apply weights
        let weight = if is_passed {
            3
        } else if black_pawns.binary_search(&(bx - 1, by + 1)).is_ok()
            || black_pawns.binary_search(&(bx + 1, by + 1)).is_ok()
        {
            2
        } else {
            // base of the pawn chain
            3
        };

        // Find the nearest king to the pawn.
        for bk in black_royals {
            if matches!(bk.piece_type, PieceType::King) {
                let d = saturating_dist_i32((bx - bk.x).abs().max((by - bk.y).abs()));
                min_d = min_d.min(d);
                let md = saturating_dist_i32((bx - bk.x).abs() + (by - bk.y).abs());
                min_friendly_md = min_friendly_md.min(md);
            }
        }
        let near_friendly_king_bonus = pawn_friendly_king_dist()[rel_rank] * (7 - min_d.min(7));

        min_d = 255; // reset it back
        for wk in white_royals {
            if matches!(wk.piece_type, PieceType::King) {
                let d = saturating_dist_i32((bx - wk.x).abs().max((by - wk.y).abs()));
                min_d = min_d.min(d);
                let md = saturating_dist_i32((bx - wk.x).abs() + (by - wk.y).abs());
                min_enemy_md = min_enemy_md.min(md);
            }
        }
        let near_enemy_king_penalty = pawn_enemy_king_dist()[rel_rank] * (7 - min_d.min(7));

        b_king_pawn_distances.push((min_friendly_md, min_enemy_md));
        nearest_pawn_distance = nearest_pawn_distance.min(min_friendly_md).min(min_enemy_md);
        b_activity += (near_friendly_king_bonus - near_enemy_king_penalty) * weight / 3;
    }

    // Apply adjustments to nullify the effect if both friendly and enemy kings are far.
    let max_effective_dist = nearest_pawn_distance.saturating_add(5);
    for (friendly_dist, enemy_dist) in w_king_pawn_distances {
        w_activity += (enemy_dist.min(max_effective_dist) - friendly_dist.min(max_effective_dist))
            .clamp(-30, 30);
    }
    for (friendly_dist, enemy_dist) in b_king_pawn_distances {
        b_activity += (enemy_dist.min(max_effective_dist) - friendly_dist.min(max_effective_dist))
            .clamp(-30, 30);
    }

    // Apply a scale factor before outputting the value.
    w_activity = w_activity * king_phase / MAX_KING_PHASE;
    b_activity = b_activity * king_phase / MAX_KING_PHASE;

    tracer.record("Pawn: King Pawn Tropism", w_activity, b_activity);
    w_activity - b_activity
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
                score += get_piece_value_base(piece.piece_type());
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
                score -= get_piece_value_base(piece.piece_type());
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

    /// A colour mirror must evaluate to exactly 0. Any gap means a term reads an
    /// absolute board position rather than one derived from the pieces -- an 8x8
    /// assumption that also fires arbitrarily once play drifts from the origin.
    #[test]
    fn mirror_symmetric_positions_evaluate_to_zero() {
        use crate::board::PlayerColor;
        use std::collections::BTreeMap;

        let mut offenders: Vec<String> = Vec::new();
        let mut checked = 0usize;

        // One variant at a time: setup_position_from_icn writes global world bounds.
        let variants = [
            crate::Variant::Classical,
            crate::Variant::CoaIP,
            crate::Variant::CoaIPHO,
            crate::Variant::CoaIPRO,
            crate::Variant::CoaIPNO,
            crate::Variant::Palace,
            crate::Variant::Standarch,
            crate::Variant::Obstocean,
            crate::Variant::Knightline,
            crate::Variant::Core,
            crate::Variant::ConfinedClassical,
            crate::Variant::Chess,
            crate::Variant::ScatteredLeapers,
            crate::Variant::ClassicalPlus,
            crate::Variant::DoubleKingClassical,
        ];
        for v in variants {
            let mut game = GameState::new();
            game.setup_position_from_icn(v.starting_icn());

            let mut white: BTreeMap<(i64, i64), u8> = BTreeMap::new();
            let mut black: BTreeMap<(i64, i64), u8> = BTreeMap::new();
            for (x, y, p) in game.board.iter_all_pieces() {
                match p.color() {
                    PlayerColor::White => white.insert((x, y), p.piece_type() as u8),
                    PlayerColor::Black => black.insert((x, y), p.piece_type() as u8),
                    PlayerColor::Neutral => continue,
                };
            }
            if white.len() != black.len() || white.is_empty() {
                continue;
            }
            let (Some(wmin), Some(bmax)) = (
                white.keys().map(|k| k.1).min(),
                black.keys().map(|k| k.1).max(),
            ) else {
                continue;
            };
            let axis = wmin + bmax;
            if white
                .iter()
                .any(|((x, y), pt)| black.get(&(*x, axis - *y)) != Some(pt))
            {
                continue; // not a mirror; an imbalance here is legitimate
            }

            checked += 1;
            let score = evaluate_inner(&game);
            if score != 0 {
                offenders.push(format!("{v:?} evaluates {score} in a colour mirror"));
            }
        }

        assert!(checked >= 10, "expected several mirror variants, saw {checked}");
        assert!(offenders.is_empty(), "{}", offenders.join("; "));
    }

    /// The trace must be an accounting identity: summing the rows has to
    /// reproduce the base evaluation exactly, or every CP tuned against it is
    /// tuned against the wrong number.
    #[test]
    fn trace_rows_sum_to_the_base_evaluation() {
        for icn in [
            "w (8|1) K5,1|k5,8|Q4,4|R1,1|P2,2|p7,7|n6,6",
            "w (8|1) K5,1|k5,8|R1,1|R8,1|P1,2|P2,2|P3,2|p1,7|p2,7|b4,5",
            "b (8|1) K5,1|k5,8|P4,7|p4,2|N2,3|n7,6",
        ] {
            let mut game = GameState::new();
            game.setup_position_from_icn(icn);

            let mut trace = ActiveTrace::default();
            let traced = evaluate_inner_traced(&game, &mut trace);
            let summed: i32 = trace.rows.iter().map(|(_, w, b)| w - b).sum();

            // Rows are white-relative; the return is side-to-move relative.
            let white_relative = if game.turn == PlayerColor::Black {
                -traced
            } else {
                traced
            };
            assert_eq!(
                summed, white_relative,
                "{icn}: rows sum to {summed} but the white-relative evaluation is {white_relative}"
            );
        }
    }

    /// A non-mating, non-insufficient-material position must never evaluate
    /// past MATE_SCORE: crossing it makes is_decisive() true and lets a
    /// coordinate-overflow bug pass as a real forced mate into the TT.
    fn assert_sane_envelope(icn: &str) {
        let mut game = GameState::new();
        game.setup_position_from_icn(icn);
        let score = evaluate(&game);
        assert!(
            score.abs() < crate::search::MATE_SCORE,
            "{icn}: eval {score} crosses MATE_SCORE ({})",
            crate::search::MATE_SCORE
        );
    }

    /// Distance-based terms (queen line-tropism, king-pawn tropism) must stay
    /// bounded and roughly constant in SIGN/magnitude class as coordinates
    /// grow, instead of wrapping through i32 at huge but legitimate distances.
    #[test]
    fn far_coordinates_do_not_overflow() {
        for coord in [
            0i64,
            1,
            1_000_000,
            1_000_000_000,
            1_000_000_000_000_000,
            i32::MAX as i64,
            i32::MAX as i64 + 1,
            i64::MAX / 4,
        ] {
            assert_sane_envelope(&format!("w (8|1) K1,1|k5,8|Q5,{}", -coord.max(1)));
            assert_sane_envelope(&format!(
                "w (8|1) K{},1|k{},8|P4,7",
                -coord.max(1),
                coord.max(1)
            ));
        }
    }

    /// Only for debugging purposes, excluded during tests or releases
    #[test]
    #[ignore]
    fn test_evaluate_traced() {
        let mut game = GameState::new();
        let icn = "b 1/100 1 (8|1) p6,7+|P5,2+|p4,4|p8,6|P8,5|p3,5|P4,3|P1,5|k3,6|p6,5|K5,1";
        game.setup_position_from_icn(icn);

        let traced_evaluation = debug_evaluate(&game);

        traced_evaluation.print();
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
    fn test_neutral_only_void_tile_contributes_king_shelter() {
        // Put the Void in the adjacent tile, rather than the king's tile. Its
        // packed representation is zero, so this exercises both neutral-only
        // tile traversal and occupied-packed-zero decoding.
        let mut open = GameState::new();
        open.setup_position_from_icn("w (8;q|1;q) K7,0|k30,30");
        let open_trace = debug_evaluate(&open);
        let open_shelter = open_trace
            .rows
            .iter()
            .find(|(term, _, _)| term == "King: Shelter")
            .map(|(_, white, _)| *white)
            .expect("king shelter trace row");

        let mut walled = GameState::new();
        walled.setup_position_from_icn("w (8;q|1;q) K7,0|k30,30|VO8,0");
        let walled_trace = debug_evaluate(&walled);
        let walled_shelter = walled_trace
            .rows
            .iter()
            .find(|(term, _, _)| term == "King: Shelter")
            .map(|(_, white, _)| *white)
            .expect("king shelter trace row");

        assert!(
            walled_shelter > open_shelter,
            "a neutral Void wall must provide white king shelter: open={open_shelter}, walled={walled_shelter}"
        );
    }

    #[test]
    fn test_mixed_tile_obstacle_has_no_colored_activity() {
        // The obstacle shares a tile with the white rook so it reaches the
        // main loop, but it is deliberately outside both kings' local/ray
        // geometry. It must not add black threat points or non-pawn material.
        let mut plain = GameState::new();
        plain.setup_position_from_icn("w (100;q|1;q) R0,0|K20,8|r20,0|k-20,-20");

        let mut with_obstacle = GameState::new();
        with_obstacle.setup_position_from_icn("w (100;q|1;q) R0,0|OB7,2|K20,8|r20,0|k-20,-20");

        assert_eq!(
            evaluate_inner(&with_obstacle),
            evaluate_inner(&plain),
            "an unrelated neutral obstacle must not be evaluated as black activity"
        );
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

        let wk = [Coordinate::new(0, 0)];
        let bk = [Coordinate::new(7, 7)];
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

        let wk = [Coordinate::new(0, 0)];
        let bk = [Coordinate::new(7, 7)];
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

        let wk = [Coordinate::new(0, 0)];
        let bk = [Coordinate::new(7, 7)];
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

        let score_no_support = evaluate_knight(4, 4, PlayerColor::White, None, 8, 0, &[], &[]);

        // Case 2: Support from pawn at (3,3) (White pawn at y-1 supports y)
        let icn_supported = "w (8;q|1;q) K0,0|k0,10|N4,4|P3,3";
        game.setup_position_from_icn(icn_supported);
        // Mock pawn list
        let white_pawns = vec![(3, 3)];

        let score_supported =
            evaluate_knight(4, 4, PlayerColor::White, None, 8, 0, &white_pawns, &[]);

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
            eg_outpost_bonus(),
            "Bonus should match eg_outpost_bonus() in endgame"
        );
    }

    #[test]
    fn test_candidate_passer_bonus() {
        let mut game = Box::new(GameState::new());

        // A white pawn at (4,4) stopped by a black pawn on an adjacent file, with
        // white support balancing the stopper.
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
