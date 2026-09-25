//! Evaluation tuning parameters, each declared once in the `eval_params!` table:
//! `name = default` for a fixed value, plus `=> (min, max, c_end, r_end, description)`
//! for one the tuners may move. The table generates the accessors, `EvalParams`
//! and `TUNABLE_EVAL_PARAM_SPECS`; spsa/texel rewrite its defaults in place.

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
use once_cell::sync::Lazy;
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
use std::sync::RwLock;

#[derive(Debug, Clone, Copy)]
pub struct EvalParamSpec {
    pub name: &'static str,
    pub default: i64,
    pub min: i64,
    pub max: i64,
    pub c_end: f64,
    pub r_end: f64,
    pub description: &'static str,
}

impl EvalParamSpec {
    pub const fn new(
        name: &'static str,
        default: i64,
        min: i64,
        max: i64,
        c_end: f64,
        r_end: f64,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            default,
            min,
            max,
            c_end,
            r_end,
            description,
        }
    }

    #[inline]
    pub fn clamp_value(self, value: i64) -> i64 {
        value.clamp(self.min, self.max)
    }
}

macro_rules! eval_params {
    ($(
        $(#[$meta:meta])*
        $name:ident = $default:literal
        $(=> ($min:literal, $max:literal, $c_end:literal, $r_end:literal, $desc:literal))?;
    )*) => {
        #[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(default)]
        pub struct EvalParams {
            $(pub $name: i32,)*
        }

        #[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
        impl Default for EvalParams {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        $(
            $(#[$meta])*
            #[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
            #[inline]
            pub fn $name() -> i32 {
                eval_param!($name)
            }

            $(#[$meta])*
            #[cfg(not(any(feature = "param_tuning", feature = "eval_tuning")))]
            #[inline]
            pub const fn $name() -> i32 {
                $default
            }
        )*

        pub const TUNABLE_EVAL_PARAM_SPECS: &[EvalParamSpec] = &[
            $($(
                EvalParamSpec::new(stringify!($name), $default, $min, $max, $c_end, $r_end, $desc),
            )?)*
        ];
    };
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub static EVAL_PARAMS: Lazy<RwLock<EvalParams>> = Lazy::new(|| RwLock::new(EvalParams::default()));

/// Bumped on every write to [`EVAL_PARAMS`]. Threads compare it against their
/// cached copy's generation to know whether that copy is stale.
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub static EVAL_PARAMS_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Replace the live eval parameters. Always use this (never write through
/// [`EVAL_PARAMS`] directly) so the generation counter invalidates thread caches.
#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn set_eval_params(params: EvalParams) -> bool {
    match EVAL_PARAMS.write() {
        Ok(mut guard) => {
            *guard = params;
            EVAL_PARAMS_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            true
        }
        Err(_) => false,
    }
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn set_eval_params_from_json(json: &str) -> bool {
    match serde_json::from_str::<EvalParams>(json) {
        Ok(params) => set_eval_params(params),
        Err(_) => false,
    }
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn get_eval_params_as_json() -> String {
    match EVAL_PARAMS.read() {
        Ok(guard) => serde_json::to_string(&*guard).unwrap_or_else(|_| "{}".to_string()),
        Err(_) => "{}".to_string(),
    }
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
thread_local! {
    /// Per-thread snapshot, refreshed only when the global generation moves.
    /// A single eval reads 130+ params, so a shared `RwLock` per read thrashed one
    /// cache line across threads; this is a relaxed atomic load plus a field access.
    static EVAL_PARAMS_TLS: std::cell::RefCell<(u64, EvalParams)> =
        std::cell::RefCell::new((u64::MAX, EvalParams::default()));
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
macro_rules! eval_param {
    ($field:ident) => {{
        EVAL_PARAMS_TLS.with(|cell| {
            let generation = EVAL_PARAMS_GEN.load(std::sync::atomic::Ordering::Acquire);
            let mut cached = cell.borrow_mut();
            if cached.0 != generation {
                if let Ok(guard) = EVAL_PARAMS.read() {
                    cached.1 = guard.clone();
                    cached.0 = generation;
                }
            }
            cached.1.$field
        })
    }};
}

// Indexed families (`_N`, `_A_S_R`) are read through the grouped helpers below;
// the per-index names are what the tuners and their JSON address.
eval_params! {
    pawn = 100;
    knight = 315 => (150, 450, 4.0, 0.002, "Knight value");
    bishop = 450 => (250, 650, 4.0, 0.002, "Bishop value");
    rook = 618 => (450, 850, 6.0, 0.002, "Rook value");
    guard = 232 => (120, 420, 4.0, 0.002, "Guard value");
    centaur = 640 => (350, 750, 6.0, 0.002, "Centaur value");
    queen = 1518 => (700, 2100, 4.0, 0.002, "Queen value");
    camel = 175 => (120, 470, 4.0, 0.002, "Camel value");
    giraffe = 165 => (120, 460, 4.0, 0.002, "Giraffe value");
    zebra = 180 => (120, 460, 4.0, 0.002, "Zebra value");
    knightrider = 900 => (500, 900, 8.0, 0.002, "Knightrider value");
    hawk = 540 => (400, 800, 6.0, 0.002, "Hawk value");
    archbishop = 1080 => (700, 1100, 8.0, 0.002, "Archbishop value");
    rose = 997 => (700, 1250, 6.0, 0.002, "Rose value");
    huygen = 330 => (155, 555, 4.0, 0.002, "Huygen value");
    chancellor = 1060 => (600, 1900, 4.0, 0.002, "Chancellor value");
    mg_doubled_pawn_penalty = 10 => (0, 208, 2.0, 0.002, "Middlegame doubled pawn penalty");
    eg_doubled_pawn_penalty = 15 => (0, 212, 2.0, 0.002, "Endgame doubled pawn penalty");
    mg_bishop_pair_bonus = 57 => (0, 260, 2.0, 0.002, "Middlegame bishop pair bonus");
    eg_bishop_pair_bonus = 101 => (0, 280, 2.0, 0.002, "Endgame bishop pair bonus");
    rook_open_file_bonus = 57 => (0, 245, 2.0, 0.002, "Rook open file bonus");
    rook_semi_open_file_bonus = 29 => (0, 220, 2.0, 0.002, "Rook semi-open file bonus");
    queen_open_file_bonus = 33 => (0, 225, 2.0, 0.002, "Queen open file bonus");
    queen_semi_open_file_bonus = 19 => (0, 210, 2.0, 0.002, "Queen semi-open file bonus");
    mg_king_ring_missing_penalty = 52 => (0, 160, 2.0, 0.002, "Middlegame king ring missing penalty");
    eg_king_ring_missing_penalty = 11 => (0, 160, 2.0, 0.002, "Endgame king ring missing penalty");
    mg_king_pawn_shield_bonus = 20 => (0, 100, 2.0, 0.002, "Middlegame king pawn shield bonus");
    eg_king_pawn_shield_bonus = 0 => (0, 100, 2.0, 0.002, "Endgame king pawn shield bonus");
    mg_behind_king_bonus = 45 => (0, 200, 2.0, 0.002, "Middlegame piece behind king bonus");
    eg_behind_king_bonus = 59 => (0, 200, 2.0, 0.002, "Endgame piece behind king bonus");
    mg_connected_pawn_bonus = 0 => (0, 60, 2.0, 0.002, "Middlegame connected pawn bonus");
    eg_connected_pawn_bonus = 30 => (0, 60, 2.0, 0.002, "Endgame connected pawn bonus");
    mg_passed_safe_path_bonus = 27 => (0, 240, 2.0, 0.002, "Middlegame passed pawn safe path bonus");
    eg_passed_safe_path_bonus = 67 => (0, 240, 2.0, 0.002, "Endgame passed pawn safe path bonus");
    mg_king_open_file_penalty = 28 => (0, 120, 2.0, 0.002, "Middlegame king open file penalty");
    eg_king_open_file_penalty = 0 => (0, 120, 2.0, 0.002, "Endgame king open file penalty");
    mg_king_defender_bonus = 18 => (0, 50, 2.0, 0.002, "Middlegame king defender bonus");
    eg_king_defender_bonus = 0 => (0, 50, 2.0, 0.002, "Endgame king defender bonus");
    mg_outpost_bonus = 33 => (0, 220, 2.0, 0.002, "Middlegame outpost bonus");
    eg_outpost_bonus = 56 => (0, 250, 2.0, 0.002, "Endgame outpost bonus");
    /// Amazon was the only compound priced at the bare sum of its parts, while the
    /// chancellor carries +245 over rook+knight and the archbishop +371.
    amazon = 1793 => (900, 2800, 4.0, 0.002, "Amazon value");
    slider_net_bonus = 21 => (0, 80, 2.0, 0.002, "Slider net-control bonus");
    far_slider_cheb_radius = 18 => (6, 40, 2.0, 0.002, "Chebyshev radius beyond which a slider is far from the action");
    far_slider_cheb_max_excess = 40 => (10, 100, 2.0, 0.002, "Max excess distance counted for the far-slider penalty");
    far_queen_penalty = 5 => (0, 30, 2.0, 0.002, "Per-excess-square penalty for a far queen");
    far_rook_penalty = 7 => (0, 25, 2.0, 0.002, "Per-excess-square penalty for a far rook");
    piece_cloud_cheb_radius = 16 => (4, 40, 2.0, 0.002, "Chebyshev radius of the piece-cloud cohesion zone");
    /// Tighter than the rider radius because a leaper's reach is one jump.
    leaper_cloud_radius = 8 => (2, 40, 2.0, 0.002, "Cloud radius beyond which a leaper counts as out of play");
    /// Riders are the knightrider, rose and huygen.
    rider_cloud_radius = 16 => (2, 40, 2.0, 0.002, "Cloud radius beyond which a rider counts as out of play");
    slider_axis_wiggle = 5 => (1, 20, 2.0, 0.002, "Wiggle room for a slider ray to count as passing through center");
    piece_cloud_cheb_max_excess = 64 => (16, 160, 2.0, 0.002, "Max excess distance counted for the piece-cloud penalty");
    centrality_value_scale = 72 => (20, 200, 4.0, 0.002, "Cloud-centre weight as a percent of piece value");
    cloud_penalty_max_pct = 50 => (10, 130, 4.0, 0.002, "Cloud penalty ceiling, as a percent of the piece's value");
    cloud_penalty_per_100_value = 2 => (0, 6, 2.0, 0.002, "Cloud-spread penalty per 100 value of piece worth");
    cloud_center_max_skew_dist = 16 => (4, 40, 2.0, 0.002, "Max skew distance for the cloud-center reference point");
    leaper_tropism_divisor = 400 => (100, 1000, 4.0, 0.002, "Divisor turning leaper value into a tropism multiplier");
    chancellor_rook_scale = 90 => (20, 150, 2.0, 0.002, "Chancellor rook-component scale (% of rook eval)");
    archbishop_bishop_scale = 90 => (20, 150, 2.0, 0.002, "Archbishop bishop-component scale (% of bishop eval)");
    amazon_rook_scale = 50 => (10, 120, 2.0, 0.002, "Amazon rook-component scale (% of rook eval)");
    amazon_queen_scale = 70 => (10, 150, 2.0, 0.002, "Amazon queen-component scale (% of queen eval)");
    centaur_guard_scale = 50 => (10, 120, 2.0, 0.002, "Centaur guard/leaper-component scale (% of guard eval)");
    pawn_full_value_threshold = 6 => (2, 16, 2.0, 0.002, "Ranks from promotion within which a pawn keeps full value");
    pawn_past_promo_penalty = 90 => (0, 250, 2.0, 0.002, "Penalty for a pawn that can never promote");
    pawn_far_from_promo_max_penalty = 100 => (0, 150, 2.0, 0.002, "Max penalty for a pawn far from promotion");
    tied_defender_ref_value = 600 => (150, 1600, 8.0, 0.002, "Reference value at which a king-tied piece pays the full penalty");
    king_defender_ref_value = 250 => (100, 800, 4.0, 0.002, "Piece value below which a piece counts as a king defender");
    complexity_damp = 8 => (0, 40, 2.0, 0.002, "Per-excess-phase damping applied to the whole score");
    complexity_excess_max = 40 => (8, 100, 2.0, 0.002, "Cap on phase excess counted for complexity damping");
    /// A pawn only shelters the king when close in front; on an unbounded board an
    /// ahead pawn could otherwise be arbitrarily far and fabricate cover.
    king_shield_ahead_max_dist = 3 => (1, 10, 2.0, 0.002, "Max distance ahead of the king counted for pawn shield");
    mg_king_pawn_ahead_penalty = 20 => (0, 80, 2.0, 0.002, "Middlegame penalty for a pawn stuck ahead of its own king");
    eg_king_pawn_ahead_penalty = 0 => (0, 60, 2.0, 0.002, "Endgame penalty for a pawn stuck ahead of its own king");
    mg_far_slider_penalty_mult = 100 => (20, 200, 2.0, 0.002, "Middlegame far-slider penalty scale (%)");
    eg_far_slider_penalty_mult = 44 => (0, 150, 2.0, 0.002, "Endgame far-slider penalty scale (%)");
    slider_threat_div = 5 => (2, 40, 2.0, 0.002, "Divisor for slider-threat scoring");
    slider_threat_cap = 100 => (5, 150, 2.0, 0.002, "Cap on slider-threat scoring");
    /// Cost of a piece frozen by a real absolute pin, per tied_defender_ref_value.
    pin_opportunity_cost = 26 => (0, 80, 2.0, 0.002, "Cost of an absolutely pinned piece");
    pin_opportunity_cap = 70 => (0, 300, 2.0, 0.002, "Cap on total pin opportunity cost");
    candidate_passer_bonus_0 = 2 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [0]");
    candidate_passer_bonus_1 = 0 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [1]");
    candidate_passer_bonus_2 = 12 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [2]");
    candidate_passer_bonus_3 = 25 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [3]");
    candidate_passer_bonus_4 = 42 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [4]");
    candidate_passer_bonus_5 = 74 => (0, 200, 2.0, 0.002, "Candidate passer bonus by relative rank [5]");
    pawn_friendly_king_dist_0 = 7 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [0]");
    pawn_friendly_king_dist_1 = 3 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [1]");
    pawn_friendly_king_dist_2 = 6 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [2]");
    pawn_friendly_king_dist_3 = 0 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [3]");
    pawn_friendly_king_dist_4 = 3 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [4]");
    pawn_friendly_king_dist_5 = 14 => (0, 40, 2.0, 0.002, "Friendly-king-distance weight by relative rank [5]");
    pawn_enemy_king_dist_0 = 4 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [0]");
    pawn_enemy_king_dist_1 = 4 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [1]");
    pawn_enemy_king_dist_2 = 0 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [2]");
    pawn_enemy_king_dist_3 = 8 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [3]");
    pawn_enemy_king_dist_4 = 9 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [4]");
    pawn_enemy_king_dist_5 = 19 => (0, 40, 2.0, 0.002, "Enemy-king-distance weight by relative rank [5]");
    passed_friendly_king_dist_0 = 0 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [0]");
    passed_friendly_king_dist_1 = 0 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [1]");
    passed_friendly_king_dist_2 = 0 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [2]");
    passed_friendly_king_dist_3 = 5 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [3]");
    passed_friendly_king_dist_4 = 8 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [4]");
    passed_friendly_king_dist_5 = 4 => (0, 40, 2.0, 0.002, "Passed-pawn friendly-king-distance weight by relative rank [5]");
    passed_enemy_king_dist_0 = 0 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [0]");
    passed_enemy_king_dist_1 = 10 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [1]");
    passed_enemy_king_dist_2 = 1 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [2]");
    passed_enemy_king_dist_3 = 3 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [3]");
    passed_enemy_king_dist_4 = 3 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [4]");
    passed_enemy_king_dist_5 = 9 => (0, 40, 2.0, 0.002, "Passed-pawn enemy-king-distance weight by relative rank [5]");
    passed_pawn_adv_bonus_0_0_0 = 0 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=0]");
    passed_pawn_adv_bonus_0_0_1 = 3 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=1]");
    passed_pawn_adv_bonus_0_0_2 = 4 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=2]");
    passed_pawn_adv_bonus_0_0_3 = 13 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=3]");
    passed_pawn_adv_bonus_0_0_4 = 21 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=4]");
    passed_pawn_adv_bonus_0_0_5 = 37 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=0][rank=5]");
    passed_pawn_adv_bonus_0_1_0 = 0 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=0]");
    passed_pawn_adv_bonus_0_1_1 = 4 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=1]");
    passed_pawn_adv_bonus_0_1_2 = 13 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=2]");
    passed_pawn_adv_bonus_0_1_3 = 26 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=3]");
    passed_pawn_adv_bonus_0_1_4 = 57 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=4]");
    passed_pawn_adv_bonus_0_1_5 = 81 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=0][safeAdvance=1][rank=5]");
    passed_pawn_adv_bonus_1_0_0 = 2 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=0]");
    passed_pawn_adv_bonus_1_0_1 = 0 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=1]");
    passed_pawn_adv_bonus_1_0_2 = 16 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=2]");
    passed_pawn_adv_bonus_1_0_3 = 36 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=3]");
    passed_pawn_adv_bonus_1_0_4 = 70 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=4]");
    passed_pawn_adv_bonus_1_0_5 = 124 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=0][rank=5]");
    passed_pawn_adv_bonus_1_1_0 = 0 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=0]");
    passed_pawn_adv_bonus_1_1_1 = 8 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=1]");
    passed_pawn_adv_bonus_1_1_2 = 38 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=2]");
    passed_pawn_adv_bonus_1_1_3 = 81 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=3]");
    passed_pawn_adv_bonus_1_1_4 = 148 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=4]");
    passed_pawn_adv_bonus_1_1_5 = 238 => (0, 300, 2.0, 0.002, "Passed pawn advance bonus [canAdvance=1][safeAdvance=1][rank=5]");
}

#[inline]
pub fn candidate_passer_bonus() -> [i32; 6] {
    [candidate_passer_bonus_0(), candidate_passer_bonus_1(), candidate_passer_bonus_2(), candidate_passer_bonus_3(), candidate_passer_bonus_4(), candidate_passer_bonus_5()]
}

#[inline]
pub fn pawn_friendly_king_dist() -> [i32; 6] {
    [pawn_friendly_king_dist_0(), pawn_friendly_king_dist_1(), pawn_friendly_king_dist_2(), pawn_friendly_king_dist_3(), pawn_friendly_king_dist_4(), pawn_friendly_king_dist_5()]
}

#[inline]
pub fn pawn_enemy_king_dist() -> [i32; 6] {
    [pawn_enemy_king_dist_0(), pawn_enemy_king_dist_1(), pawn_enemy_king_dist_2(), pawn_enemy_king_dist_3(), pawn_enemy_king_dist_4(), pawn_enemy_king_dist_5()]
}

#[inline]
pub fn passed_friendly_king_dist() -> [i32; 6] {
    [passed_friendly_king_dist_0(), passed_friendly_king_dist_1(), passed_friendly_king_dist_2(), passed_friendly_king_dist_3(), passed_friendly_king_dist_4(), passed_friendly_king_dist_5()]
}

#[inline]
pub fn passed_enemy_king_dist() -> [i32; 6] {
    [passed_enemy_king_dist_0(), passed_enemy_king_dist_1(), passed_enemy_king_dist_2(), passed_enemy_king_dist_3(), passed_enemy_king_dist_4(), passed_enemy_king_dist_5()]
}

/// Indexed `[can_advance][safe_advance][relative_rank]`; rank 5 is next to promotion.
#[inline]
pub fn passed_pawn_adv_bonus() -> [[[i32; 6]; 2]; 2] {
    [
    [
        [passed_pawn_adv_bonus_0_0_0(), passed_pawn_adv_bonus_0_0_1(), passed_pawn_adv_bonus_0_0_2(), passed_pawn_adv_bonus_0_0_3(), passed_pawn_adv_bonus_0_0_4(), passed_pawn_adv_bonus_0_0_5()],
        [passed_pawn_adv_bonus_0_1_0(), passed_pawn_adv_bonus_0_1_1(), passed_pawn_adv_bonus_0_1_2(), passed_pawn_adv_bonus_0_1_3(), passed_pawn_adv_bonus_0_1_4(), passed_pawn_adv_bonus_0_1_5()],
    ],
    [
        [passed_pawn_adv_bonus_1_0_0(), passed_pawn_adv_bonus_1_0_1(), passed_pawn_adv_bonus_1_0_2(), passed_pawn_adv_bonus_1_0_3(), passed_pawn_adv_bonus_1_0_4(), passed_pawn_adv_bonus_1_0_5()],
        [passed_pawn_adv_bonus_1_1_0(), passed_pawn_adv_bonus_1_1_1(), passed_pawn_adv_bonus_1_1_2(), passed_pawn_adv_bonus_1_1_3(), passed_pawn_adv_bonus_1_1_4(), passed_pawn_adv_bonus_1_1_5()],
    ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default outside its own range makes the first SPSA perturbation clamp
    /// both candidates to the same wrong side.
    #[test]
    fn eval_param_specs_have_valid_ranges_and_defaults() {
        for spec in TUNABLE_EVAL_PARAM_SPECS {
            assert!(
                spec.min <= spec.default && spec.default <= spec.max,
                "{}: default {} outside [{}, {}]",
                spec.name, spec.default, spec.min, spec.max
            );
            assert_eq!(spec.clamp_value(spec.min - 1000), spec.min);
            assert_eq!(spec.clamp_value(spec.max + 1000), spec.max);
        }
    }
}
