//! Search Parameters Module
//!
//! This file is the single source of truth for search tuning:
//! - production defaults live here as compile-time constants
//! - runtime overrides (behind `search_tuning`) deserialize into `SearchParams`
//! - SPSA tuning metadata lives here to prevent drift with external scripts

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
use once_cell::sync::Lazy;
#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchParamKind {
    I32,
    Usize,
    U8,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchParamSpec {
    pub name: &'static str,
    pub kind: SearchParamKind,
    pub default: i64,
    pub min: i64,
    pub max: i64,
    pub c_end: f64,
    pub r_end: f64,
    pub description: &'static str,
}

impl SearchParamSpec {
    pub const fn new(
        name: &'static str,
        kind: SearchParamKind,
        default: i64,
        min: i64,
        max: i64,
        c_end: f64,
        r_end: f64,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            kind,
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

pub const DEFAULT_RAZORING_LINEAR: i32 = 485;
pub const DEFAULT_RAZORING_QUAD: i32 = 281;

pub const DEFAULT_NMP_MIN_DEPTH: usize = 3;
pub const DEFAULT_NMP_BASE: i32 = 350;
pub const DEFAULT_NMP_DEPTH_MULT: i32 = 18;
pub const DEFAULT_NMP_REDUCTION_BASE: usize = 7;
pub const DEFAULT_NMP_REDUCTION_DIV: usize = 3;

pub const DEFAULT_LMR_MIN_DEPTH: usize = 3;
pub const DEFAULT_LMR_MIN_MOVES: usize = 4;
pub const DEFAULT_LMR_DIVISOR: usize = 3;
pub const DEFAULT_LMR_CUTOFF_THRESH: u8 = 2;
pub const DEFAULT_LMR_TT_HISTORY_THRESH: i32 = -1000;

pub const DEFAULT_HLP_MAX_DEPTH: usize = 3;
pub const DEFAULT_HLP_MIN_MOVES: usize = 4;
pub const DEFAULT_HLP_HISTORY_REDUCE: i32 = 300;
pub const DEFAULT_HLP_HISTORY_LEAF: i32 = 0;

pub const DEFAULT_LMP_BASE: usize = 3;
pub const DEFAULT_LMP_DEPTH_MULT: usize = 1;

pub const DEFAULT_ASPIRATION_WINDOW: i32 = 60;
pub const DEFAULT_ASPIRATION_FAIL_MULT: i32 = 4;
pub const DEFAULT_ASPIRATION_MAX_WINDOW: i32 = 1000;

pub const DEFAULT_RFP_MAX_DEPTH: usize = 14;
pub const DEFAULT_RFP_MULT_TT: i32 = 76;
pub const DEFAULT_RFP_MULT_NO_TT: i32 = 53;
pub const DEFAULT_RFP_IMPROVING_MULT: i32 = 2474;
pub const DEFAULT_RFP_WORSENING_MULT: i32 = 331;

pub const DEFAULT_PROBCUT_MARGIN: i32 = 235;
pub const DEFAULT_PROBCUT_IMPROVING: i32 = 63;
pub const DEFAULT_PROBCUT_MIN_DEPTH: usize = 5;
pub const DEFAULT_PROBCUT_DEPTH_SUB: usize = 4;
pub const DEFAULT_PROBCUT_DIVISOR: i32 = 315;
pub const DEFAULT_LOW_DEPTH_PROBCUT_MARGIN: i32 = 800;

pub const DEFAULT_IIR_MIN_DEPTH: usize = 6;

pub const DEFAULT_SEE_CAPTURE_LINEAR: i32 = 166;
pub const DEFAULT_SEE_CAPTURE_HIST_DIV: i32 = 29;
pub const DEFAULT_SEE_QUIET_QUAD: i32 = 25;
pub const DEFAULT_SEE_WINNING_THRESHOLD: i32 = 0;

pub const DEFAULT_SORT_HASH: i32 = 6_000_000;
pub const DEFAULT_SORT_WINNING_CAPTURE: i32 = 1_000_000;
pub const DEFAULT_SORT_LOSING_CAPTURE: i32 = 0;
pub const DEFAULT_SORT_QUIET: i32 = 0;
pub const DEFAULT_SORT_KILLER1: i32 = 900_000;
pub const DEFAULT_SORT_KILLER2: i32 = 800_000;
pub const DEFAULT_SORT_COUNTERMOVE: i32 = 600_000;

pub const DEFAULT_HISTORY_BONUS_BASE: i32 = 300;
pub const DEFAULT_HISTORY_BONUS_SUB: i32 = 250;
pub const DEFAULT_HISTORY_BONUS_CAP: i32 = 1536;
pub const DEFAULT_HISTORY_MAX_GRAVITY: i32 = 16384;

pub const DEFAULT_PAWN_HISTORY_BONUS_SCALE: i32 = 2;
pub const DEFAULT_PAWN_HISTORY_MALUS_SCALE: i32 = 1;

pub const DEFAULT_DELTA_MARGIN: i32 = 200;

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

pub const TUNABLE_EVAL_PARAM_SPECS: &[EvalParamSpec] = &[
    EvalParamSpec::new("knight", crate::evaluation::base::DEFAULT_EVAL_KNIGHT as i64, 150, 450, 4.0, 0.002, "Knight value"),
    EvalParamSpec::new("bishop", crate::evaluation::base::DEFAULT_EVAL_BISHOP as i64, 250, 650, 4.0, 0.002, "Bishop value"),
    EvalParamSpec::new("rook", crate::evaluation::base::DEFAULT_EVAL_ROOK as i64, 450, 850, 6.0, 0.002, "Rook value"),
    EvalParamSpec::new("guard", crate::evaluation::base::DEFAULT_EVAL_GUARD as i64, 120, 420, 4.0, 0.002, "Guard value"),
    EvalParamSpec::new("centaur", crate::evaluation::base::DEFAULT_EVAL_CENTAUR as i64, 350, 750, 6.0, 0.002, "Centaur value"),
    EvalParamSpec::new("compound_bonus", crate::evaluation::base::DEFAULT_EVAL_COMPOUND_BONUS as i64, 0, 250, 4.0, 0.002, "Compound piece bonus"),
    EvalParamSpec::new("camel", crate::evaluation::base::DEFAULT_EVAL_CAMEL as i64, 120, 470, 4.0, 0.002, "Camel value"),
    EvalParamSpec::new("giraffe", crate::evaluation::base::DEFAULT_EVAL_GIRAFFE as i64, 120, 460, 4.0, 0.002, "Giraffe value"),
    EvalParamSpec::new("zebra", crate::evaluation::base::DEFAULT_EVAL_ZEBRA as i64, 120, 460, 4.0, 0.002, "Zebra value"),
    EvalParamSpec::new("knightrider", crate::evaluation::base::DEFAULT_EVAL_KNIGHTRIDER as i64, 500, 900, 8.0, 0.002, "Knightrider value"),
    EvalParamSpec::new("hawk", crate::evaluation::base::DEFAULT_EVAL_HAWK as i64, 400, 800, 6.0, 0.002, "Hawk value"),
    EvalParamSpec::new("archbishop", crate::evaluation::base::DEFAULT_EVAL_ARCHBISHOP as i64, 700, 1100, 8.0, 0.002, "Archbishop value"),
    EvalParamSpec::new("rose", crate::evaluation::base::DEFAULT_EVAL_ROSE as i64, 250, 650, 6.0, 0.002, "Rose value"),
    EvalParamSpec::new("huygen", crate::evaluation::base::DEFAULT_EVAL_HUYGEN as i64, 155, 555, 4.0, 0.002, "Huygen value"),
    EvalParamSpec::new("chancellor_bonus", crate::evaluation::base::DEFAULT_EVAL_CHANCELLOR_BONUS as i64, 0, 300, 4.0, 0.002, "Chancellor extra value bonus"),
    EvalParamSpec::new("mg_doubled_pawn_penalty", crate::evaluation::base::DEFAULT_EVAL_MG_DOUBLED_PAWN_PENALTY as i64, 0, 208, 2.0, 0.002, "Middlegame doubled pawn penalty"),
    EvalParamSpec::new("eg_doubled_pawn_penalty", crate::evaluation::base::DEFAULT_EVAL_EG_DOUBLED_PAWN_PENALTY as i64, 0, 212, 2.0, 0.002, "Endgame doubled pawn penalty"),
    EvalParamSpec::new("mg_bishop_pair_bonus", crate::evaluation::base::DEFAULT_EVAL_MG_BISHOP_PAIR_BONUS as i64, 0, 260, 2.0, 0.002, "Middlegame bishop pair bonus"),
    EvalParamSpec::new("eg_bishop_pair_bonus", crate::evaluation::base::DEFAULT_EVAL_EG_BISHOP_PAIR_BONUS as i64, 0, 280, 2.0, 0.002, "Endgame bishop pair bonus"),
    EvalParamSpec::new("rook_open_file_bonus", crate::evaluation::base::DEFAULT_EVAL_ROOK_OPEN_FILE_BONUS as i64, 0, 245, 2.0, 0.002, "Rook open file bonus"),
    EvalParamSpec::new("rook_semi_open_file_bonus", crate::evaluation::base::DEFAULT_EVAL_ROOK_SEMI_OPEN_FILE_BONUS as i64, 0, 220, 2.0, 0.002, "Rook semi-open file bonus"),
    EvalParamSpec::new("queen_open_file_bonus", crate::evaluation::base::DEFAULT_EVAL_QUEEN_OPEN_FILE_BONUS as i64, 0, 225, 2.0, 0.002, "Queen open file bonus"),
    EvalParamSpec::new("queen_semi_open_file_bonus", crate::evaluation::base::DEFAULT_EVAL_QUEEN_SEMI_OPEN_FILE_BONUS as i64, 0, 210, 2.0, 0.002, "Queen semi-open file bonus"),
    EvalParamSpec::new("mg_outpost_bonus", crate::evaluation::base::DEFAULT_EVAL_MG_OUTPOST_BONUS as i64, 0, 220, 2.0, 0.002, "Middlegame outpost bonus"),
    EvalParamSpec::new("eg_outpost_bonus", crate::evaluation::base::DEFAULT_EVAL_EG_OUTPOST_BONUS as i64, 0, 250, 2.0, 0.002, "Endgame outpost bonus"),
];

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalParams {
    pub pawn: i32,
    pub knight: i32,
    pub bishop: i32,
    pub rook: i32,
    pub guard: i32,
    pub centaur: i32,
    pub compound_bonus: i32,
    pub camel: i32,
    pub giraffe: i32,
    pub zebra: i32,
    pub knightrider: i32,
    pub hawk: i32,
    pub archbishop: i32,
    pub rose: i32,
    pub huygen: i32,
    pub chancellor_bonus: i32,
    pub mg_doubled_pawn_penalty: i32,
    pub eg_doubled_pawn_penalty: i32,
    pub mg_bishop_pair_bonus: i32,
    pub eg_bishop_pair_bonus: i32,
    pub rook_open_file_bonus: i32,
    pub rook_semi_open_file_bonus: i32,
    pub queen_open_file_bonus: i32,
    pub queen_semi_open_file_bonus: i32,
    pub mg_outpost_bonus: i32,
    pub eg_outpost_bonus: i32,
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
impl Default for EvalParams {
    fn default() -> Self {
        Self {
            pawn: crate::evaluation::base::DEFAULT_EVAL_PAWN,
            knight: crate::evaluation::base::DEFAULT_EVAL_KNIGHT,
            bishop: crate::evaluation::base::DEFAULT_EVAL_BISHOP,
            rook: crate::evaluation::base::DEFAULT_EVAL_ROOK,
            guard: crate::evaluation::base::DEFAULT_EVAL_GUARD,
            centaur: crate::evaluation::base::DEFAULT_EVAL_CENTAUR,
            compound_bonus: crate::evaluation::base::DEFAULT_EVAL_COMPOUND_BONUS,
            camel: crate::evaluation::base::DEFAULT_EVAL_CAMEL,
            giraffe: crate::evaluation::base::DEFAULT_EVAL_GIRAFFE,
            zebra: crate::evaluation::base::DEFAULT_EVAL_ZEBRA,
            knightrider: crate::evaluation::base::DEFAULT_EVAL_KNIGHTRIDER,
            hawk: crate::evaluation::base::DEFAULT_EVAL_HAWK,
            archbishop: crate::evaluation::base::DEFAULT_EVAL_ARCHBISHOP,
            rose: crate::evaluation::base::DEFAULT_EVAL_ROSE,
            huygen: crate::evaluation::base::DEFAULT_EVAL_HUYGEN,
            chancellor_bonus: crate::evaluation::base::DEFAULT_EVAL_CHANCELLOR_BONUS,
            mg_doubled_pawn_penalty: crate::evaluation::base::DEFAULT_EVAL_MG_DOUBLED_PAWN_PENALTY,
            eg_doubled_pawn_penalty: crate::evaluation::base::DEFAULT_EVAL_EG_DOUBLED_PAWN_PENALTY,
            mg_bishop_pair_bonus: crate::evaluation::base::DEFAULT_EVAL_MG_BISHOP_PAIR_BONUS,
            eg_bishop_pair_bonus: crate::evaluation::base::DEFAULT_EVAL_EG_BISHOP_PAIR_BONUS,
            rook_open_file_bonus: crate::evaluation::base::DEFAULT_EVAL_ROOK_OPEN_FILE_BONUS,
            rook_semi_open_file_bonus: crate::evaluation::base::DEFAULT_EVAL_ROOK_SEMI_OPEN_FILE_BONUS,
            queen_open_file_bonus: crate::evaluation::base::DEFAULT_EVAL_QUEEN_OPEN_FILE_BONUS,
            queen_semi_open_file_bonus: crate::evaluation::base::DEFAULT_EVAL_QUEEN_SEMI_OPEN_FILE_BONUS,
            mg_outpost_bonus: crate::evaluation::base::DEFAULT_EVAL_MG_OUTPOST_BONUS,
            eg_outpost_bonus: crate::evaluation::base::DEFAULT_EVAL_EG_OUTPOST_BONUS,
        }
    }
}

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub static EVAL_PARAMS: Lazy<RwLock<EvalParams>> = Lazy::new(|| RwLock::new(EvalParams::default()));

#[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
pub fn set_eval_params_from_json(json: &str) -> bool {
    match serde_json::from_str::<EvalParams>(json) {
        Ok(params) => match EVAL_PARAMS.write() {
            Ok(mut guard) => {
                *guard = params;
                true
            }
            Err(_) => false,
        },
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
macro_rules! eval_param {
    ($field:ident) => {{ EVAL_PARAMS.read().unwrap().$field }};
}

macro_rules! define_eval_accessor {
    ($name:ident, $default:path) => {
        #[cfg(any(feature = "param_tuning", feature = "eval_tuning"))]
        #[inline]
        pub fn $name() -> i32 {
            eval_param!($name)
        }

        #[cfg(not(any(feature = "param_tuning", feature = "eval_tuning")))]
        #[inline]
        pub const fn $name() -> i32 {
            $default
        }
    };
}

define_eval_accessor!(pawn, crate::evaluation::base::DEFAULT_EVAL_PAWN);
define_eval_accessor!(knight, crate::evaluation::base::DEFAULT_EVAL_KNIGHT);
define_eval_accessor!(bishop, crate::evaluation::base::DEFAULT_EVAL_BISHOP);
define_eval_accessor!(rook, crate::evaluation::base::DEFAULT_EVAL_ROOK);
define_eval_accessor!(guard, crate::evaluation::base::DEFAULT_EVAL_GUARD);
define_eval_accessor!(centaur, crate::evaluation::base::DEFAULT_EVAL_CENTAUR);
define_eval_accessor!(compound_bonus, crate::evaluation::base::DEFAULT_EVAL_COMPOUND_BONUS);
define_eval_accessor!(camel, crate::evaluation::base::DEFAULT_EVAL_CAMEL);
define_eval_accessor!(giraffe, crate::evaluation::base::DEFAULT_EVAL_GIRAFFE);
define_eval_accessor!(zebra, crate::evaluation::base::DEFAULT_EVAL_ZEBRA);
define_eval_accessor!(knightrider, crate::evaluation::base::DEFAULT_EVAL_KNIGHTRIDER);
define_eval_accessor!(hawk, crate::evaluation::base::DEFAULT_EVAL_HAWK);
define_eval_accessor!(archbishop, crate::evaluation::base::DEFAULT_EVAL_ARCHBISHOP);
define_eval_accessor!(rose, crate::evaluation::base::DEFAULT_EVAL_ROSE);
define_eval_accessor!(huygen, crate::evaluation::base::DEFAULT_EVAL_HUYGEN);
define_eval_accessor!(chancellor_bonus, crate::evaluation::base::DEFAULT_EVAL_CHANCELLOR_BONUS);
define_eval_accessor!(mg_doubled_pawn_penalty, crate::evaluation::base::DEFAULT_EVAL_MG_DOUBLED_PAWN_PENALTY);
define_eval_accessor!(eg_doubled_pawn_penalty, crate::evaluation::base::DEFAULT_EVAL_EG_DOUBLED_PAWN_PENALTY);
define_eval_accessor!(mg_bishop_pair_bonus, crate::evaluation::base::DEFAULT_EVAL_MG_BISHOP_PAIR_BONUS);
define_eval_accessor!(eg_bishop_pair_bonus, crate::evaluation::base::DEFAULT_EVAL_EG_BISHOP_PAIR_BONUS);
define_eval_accessor!(rook_open_file_bonus, crate::evaluation::base::DEFAULT_EVAL_ROOK_OPEN_FILE_BONUS);
define_eval_accessor!(rook_semi_open_file_bonus, crate::evaluation::base::DEFAULT_EVAL_ROOK_SEMI_OPEN_FILE_BONUS);
define_eval_accessor!(queen_open_file_bonus, crate::evaluation::base::DEFAULT_EVAL_QUEEN_OPEN_FILE_BONUS);
define_eval_accessor!(queen_semi_open_file_bonus, crate::evaluation::base::DEFAULT_EVAL_QUEEN_SEMI_OPEN_FILE_BONUS);
define_eval_accessor!(mg_outpost_bonus, crate::evaluation::base::DEFAULT_EVAL_MG_OUTPOST_BONUS);
define_eval_accessor!(eg_outpost_bonus, crate::evaluation::base::DEFAULT_EVAL_EG_OUTPOST_BONUS);

#[inline]
pub fn queen_value() -> i32 {
    rook() * 2 + compound_bonus()
}

pub const TUNABLE_PARAM_SPECS: &[SearchParamSpec] = &[
    SearchParamSpec::new("razoring_linear", SearchParamKind::I32, DEFAULT_RAZORING_LINEAR as i64, 200, 700, 16.0, 0.002, "Razoring linear margin"),
    SearchParamSpec::new("razoring_quad", SearchParamKind::I32, DEFAULT_RAZORING_QUAD as i64, 100, 500, 12.0, 0.002, "Razoring quadratic margin"),
    SearchParamSpec::new("nmp_min_depth", SearchParamKind::Usize, DEFAULT_NMP_MIN_DEPTH as i64, 1, 8, 1.0, 0.002, "Null move minimum depth"),
    SearchParamSpec::new("nmp_base", SearchParamKind::I32, DEFAULT_NMP_BASE as i64, 100, 600, 16.0, 0.002, "Null move base margin"),
    SearchParamSpec::new("nmp_depth_mult", SearchParamKind::I32, DEFAULT_NMP_DEPTH_MULT as i64, 8, 48, 2.0, 0.002, "Null move depth multiplier"),
    SearchParamSpec::new("nmp_reduction_base", SearchParamKind::Usize, DEFAULT_NMP_REDUCTION_BASE as i64, 2, 12, 1.0, 0.002, "Null move reduction numerator"),
    SearchParamSpec::new("nmp_reduction_div", SearchParamKind::Usize, DEFAULT_NMP_REDUCTION_DIV as i64, 1, 8, 1.0, 0.002, "Null move reduction divisor"),
    SearchParamSpec::new("lmr_min_depth", SearchParamKind::Usize, DEFAULT_LMR_MIN_DEPTH as i64, 1, 8, 1.0, 0.002, "Late move reduction minimum depth"),
    SearchParamSpec::new("lmr_min_moves", SearchParamKind::Usize, DEFAULT_LMR_MIN_MOVES as i64, 1, 16, 1.0, 0.002, "Late move reduction minimum move count"),
    SearchParamSpec::new("lmr_divisor", SearchParamKind::Usize, DEFAULT_LMR_DIVISOR as i64, 1, 8, 1.0, 0.002, "Late move reduction divisor"),
    SearchParamSpec::new("lmr_cutoff_thresh", SearchParamKind::U8, DEFAULT_LMR_CUTOFF_THRESH as i64, 1, 8, 1.0, 0.002, "Late move reduction cutoff threshold"),
    SearchParamSpec::new("lmr_tt_history_thresh", SearchParamKind::I32, DEFAULT_LMR_TT_HISTORY_THRESH as i64, -4000, 0, 64.0, 0.002, "Late move reduction TT history threshold"),
    SearchParamSpec::new("hlp_max_depth", SearchParamKind::Usize, DEFAULT_HLP_MAX_DEPTH as i64, 1, 8, 1.0, 0.002, "History leaf pruning maximum depth"),
    SearchParamSpec::new("hlp_min_moves", SearchParamKind::Usize, DEFAULT_HLP_MIN_MOVES as i64, 1, 16, 1.0, 0.002, "History leaf pruning minimum move count"),
    SearchParamSpec::new("hlp_history_reduce", SearchParamKind::I32, DEFAULT_HLP_HISTORY_REDUCE as i64, -2000, 2000, 64.0, 0.002, "History threshold for extra late-move reduction"),
    SearchParamSpec::new("hlp_history_leaf", SearchParamKind::I32, DEFAULT_HLP_HISTORY_LEAF as i64, -2000, 2000, 64.0, 0.002, "History threshold for pruning leaf moves"),
    SearchParamSpec::new("lmp_base", SearchParamKind::Usize, DEFAULT_LMP_BASE as i64, 1, 12, 1.0, 0.002, "Late move pruning base"),
    SearchParamSpec::new("lmp_depth_mult", SearchParamKind::Usize, DEFAULT_LMP_DEPTH_MULT as i64, 0, 6, 1.0, 0.002, "Late move pruning depth multiplier"),
    SearchParamSpec::new("aspiration_window", SearchParamKind::I32, DEFAULT_ASPIRATION_WINDOW as i64, 8, 256, 8.0, 0.002, "Initial aspiration window"),
    SearchParamSpec::new("aspiration_fail_mult", SearchParamKind::I32, DEFAULT_ASPIRATION_FAIL_MULT as i64, 2, 8, 1.0, 0.002, "Aspiration fail expansion multiplier"),
    SearchParamSpec::new("aspiration_max_window", SearchParamKind::I32, DEFAULT_ASPIRATION_MAX_WINDOW as i64, 256, 4000, 64.0, 0.002, "Maximum aspiration window"),
    SearchParamSpec::new("rfp_max_depth", SearchParamKind::Usize, DEFAULT_RFP_MAX_DEPTH as i64, 1, 20, 1.0, 0.002, "Reverse futility maximum depth"),
    SearchParamSpec::new("rfp_mult_tt", SearchParamKind::I32, DEFAULT_RFP_MULT_TT as i64, 1, 256, 4.0, 0.002, "Reverse futility TT multiplier"),
    SearchParamSpec::new("rfp_mult_no_tt", SearchParamKind::I32, DEFAULT_RFP_MULT_NO_TT as i64, 1, 256, 4.0, 0.002, "Reverse futility non-TT multiplier"),
    SearchParamSpec::new("rfp_improving_mult", SearchParamKind::I32, DEFAULT_RFP_IMPROVING_MULT as i64, 256, 4096, 64.0, 0.002, "Reverse futility improving multiplier"),
    SearchParamSpec::new("rfp_worsening_mult", SearchParamKind::I32, DEFAULT_RFP_WORSENING_MULT as i64, 0, 2048, 32.0, 0.002, "Reverse futility worsening multiplier"),
    SearchParamSpec::new("probcut_margin", SearchParamKind::I32, DEFAULT_PROBCUT_MARGIN as i64, 0, 512, 8.0, 0.002, "ProbCut margin"),
    SearchParamSpec::new("probcut_improving", SearchParamKind::I32, DEFAULT_PROBCUT_IMPROVING as i64, 0, 256, 4.0, 0.002, "ProbCut improving adjustment"),
    SearchParamSpec::new("probcut_min_depth", SearchParamKind::Usize, DEFAULT_PROBCUT_MIN_DEPTH as i64, 1, 12, 1.0, 0.002, "ProbCut minimum depth"),
    SearchParamSpec::new("probcut_depth_sub", SearchParamKind::Usize, DEFAULT_PROBCUT_DEPTH_SUB as i64, 1, 8, 1.0, 0.002, "ProbCut depth subtraction"),
    SearchParamSpec::new("probcut_divisor", SearchParamKind::I32, DEFAULT_PROBCUT_DIVISOR as i64, 32, 1024, 16.0, 0.002, "ProbCut static-eval divisor"),
    SearchParamSpec::new("low_depth_probcut_margin", SearchParamKind::I32, DEFAULT_LOW_DEPTH_PROBCUT_MARGIN as i64, 128, 2048, 32.0, 0.002, "Low-depth ProbCut margin"),
    SearchParamSpec::new("iir_min_depth", SearchParamKind::Usize, DEFAULT_IIR_MIN_DEPTH as i64, 1, 12, 1.0, 0.002, "Internal iterative reduction minimum depth"),
    SearchParamSpec::new("see_capture_linear", SearchParamKind::I32, DEFAULT_SEE_CAPTURE_LINEAR as i64, 0, 512, 8.0, 0.002, "SEE capture pruning linear term"),
    SearchParamSpec::new("see_capture_hist_div", SearchParamKind::I32, DEFAULT_SEE_CAPTURE_HIST_DIV as i64, 1, 128, 2.0, 0.002, "SEE capture history divisor"),
    SearchParamSpec::new("see_quiet_quad", SearchParamKind::I32, DEFAULT_SEE_QUIET_QUAD as i64, 1, 128, 2.0, 0.002, "SEE quiet pruning quadratic term"),
    SearchParamSpec::new("see_winning_threshold", SearchParamKind::I32, DEFAULT_SEE_WINNING_THRESHOLD as i64, -256, 256, 8.0, 0.002, "SEE threshold for classifying winning captures"),
    SearchParamSpec::new("sort_hash", SearchParamKind::I32, DEFAULT_SORT_HASH as i64, 1_000_000, 10_000_000, 100_000.0, 0.002, "Hash move ordering bonus"),
    SearchParamSpec::new("sort_winning_capture", SearchParamKind::I32, DEFAULT_SORT_WINNING_CAPTURE as i64, 100_000, 4_000_000, 50_000.0, 0.002, "Winning capture ordering bonus"),
    SearchParamSpec::new("sort_killer1", SearchParamKind::I32, DEFAULT_SORT_KILLER1 as i64, 100_000, 4_000_000, 50_000.0, 0.002, "Primary killer ordering bonus"),
    SearchParamSpec::new("sort_killer2", SearchParamKind::I32, DEFAULT_SORT_KILLER2 as i64, 100_000, 4_000_000, 50_000.0, 0.002, "Secondary killer ordering bonus"),
    SearchParamSpec::new("sort_countermove", SearchParamKind::I32, DEFAULT_SORT_COUNTERMOVE as i64, 100_000, 4_000_000, 50_000.0, 0.002, "Countermove ordering bonus"),
    SearchParamSpec::new("history_bonus_base", SearchParamKind::I32, DEFAULT_HISTORY_BONUS_BASE as i64, 0, 1024, 16.0, 0.002, "History bonus base"),
    SearchParamSpec::new("history_bonus_sub", SearchParamKind::I32, DEFAULT_HISTORY_BONUS_SUB as i64, 0, 1024, 16.0, 0.002, "History bonus subtraction"),
    SearchParamSpec::new("history_bonus_cap", SearchParamKind::I32, DEFAULT_HISTORY_BONUS_CAP as i64, 64, 8192, 64.0, 0.002, "History bonus cap"),
    SearchParamSpec::new("history_max_gravity", SearchParamKind::I32, DEFAULT_HISTORY_MAX_GRAVITY as i64, 1024, 32768, 256.0, 0.002, "History gravity clamp"),
    SearchParamSpec::new("pawn_history_bonus_scale", SearchParamKind::I32, DEFAULT_PAWN_HISTORY_BONUS_SCALE as i64, 0, 8, 1.0, 0.002, "Pawn history bonus scale"),
    SearchParamSpec::new("pawn_history_malus_scale", SearchParamKind::I32, DEFAULT_PAWN_HISTORY_MALUS_SCALE as i64, 0, 8, 1.0, 0.002, "Pawn history malus scale"),
    SearchParamSpec::new("delta_margin", SearchParamKind::I32, DEFAULT_DELTA_MARGIN as i64, 0, 512, 8.0, 0.002, "Quiescence delta pruning margin"),
];

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchParams {
    pub razoring_linear: i32,
    pub razoring_quad: i32,
    pub nmp_min_depth: usize,
    pub nmp_base: i32,
    pub nmp_depth_mult: i32,
    pub nmp_reduction_base: usize,
    pub nmp_reduction_div: usize,
    pub lmr_min_depth: usize,
    pub lmr_min_moves: usize,
    pub lmr_divisor: usize,
    pub lmr_cutoff_thresh: u8,
    pub lmr_tt_history_thresh: i32,
    pub hlp_max_depth: usize,
    pub hlp_min_moves: usize,
    pub hlp_history_reduce: i32,
    pub hlp_history_leaf: i32,
    pub lmp_base: usize,
    pub lmp_depth_mult: usize,
    pub aspiration_window: i32,
    pub aspiration_fail_mult: i32,
    pub aspiration_max_window: i32,
    pub rfp_max_depth: usize,
    pub rfp_mult_tt: i32,
    pub rfp_mult_no_tt: i32,
    pub rfp_improving_mult: i32,
    pub rfp_worsening_mult: i32,
    pub probcut_margin: i32,
    pub probcut_improving: i32,
    pub probcut_min_depth: usize,
    pub probcut_depth_sub: usize,
    pub probcut_divisor: i32,
    pub low_depth_probcut_margin: i32,
    pub iir_min_depth: usize,
    pub see_capture_linear: i32,
    pub see_capture_hist_div: i32,
    pub see_quiet_quad: i32,
    pub see_winning_threshold: i32,
    pub sort_hash: i32,
    pub sort_winning_capture: i32,
    pub sort_killer1: i32,
    pub sort_killer2: i32,
    pub sort_countermove: i32,
    pub history_bonus_base: i32,
    pub history_bonus_sub: i32,
    pub history_bonus_cap: i32,
    pub history_max_gravity: i32,
    pub pawn_history_bonus_scale: i32,
    pub pawn_history_malus_scale: i32,
    pub delta_margin: i32,
}

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
impl Default for SearchParams {
    fn default() -> Self {
        Self {
            razoring_linear: DEFAULT_RAZORING_LINEAR,
            razoring_quad: DEFAULT_RAZORING_QUAD,
            nmp_min_depth: DEFAULT_NMP_MIN_DEPTH,
            nmp_base: DEFAULT_NMP_BASE,
            nmp_depth_mult: DEFAULT_NMP_DEPTH_MULT,
            nmp_reduction_base: DEFAULT_NMP_REDUCTION_BASE,
            nmp_reduction_div: DEFAULT_NMP_REDUCTION_DIV,
            lmr_min_depth: DEFAULT_LMR_MIN_DEPTH,
            lmr_min_moves: DEFAULT_LMR_MIN_MOVES,
            lmr_divisor: DEFAULT_LMR_DIVISOR,
            lmr_cutoff_thresh: DEFAULT_LMR_CUTOFF_THRESH,
            lmr_tt_history_thresh: DEFAULT_LMR_TT_HISTORY_THRESH,
            hlp_max_depth: DEFAULT_HLP_MAX_DEPTH,
            hlp_min_moves: DEFAULT_HLP_MIN_MOVES,
            hlp_history_reduce: DEFAULT_HLP_HISTORY_REDUCE,
            hlp_history_leaf: DEFAULT_HLP_HISTORY_LEAF,
            lmp_base: DEFAULT_LMP_BASE,
            lmp_depth_mult: DEFAULT_LMP_DEPTH_MULT,
            aspiration_window: DEFAULT_ASPIRATION_WINDOW,
            aspiration_fail_mult: DEFAULT_ASPIRATION_FAIL_MULT,
            aspiration_max_window: DEFAULT_ASPIRATION_MAX_WINDOW,
            rfp_max_depth: DEFAULT_RFP_MAX_DEPTH,
            rfp_mult_tt: DEFAULT_RFP_MULT_TT,
            rfp_mult_no_tt: DEFAULT_RFP_MULT_NO_TT,
            rfp_improving_mult: DEFAULT_RFP_IMPROVING_MULT,
            rfp_worsening_mult: DEFAULT_RFP_WORSENING_MULT,
            probcut_margin: DEFAULT_PROBCUT_MARGIN,
            probcut_improving: DEFAULT_PROBCUT_IMPROVING,
            probcut_min_depth: DEFAULT_PROBCUT_MIN_DEPTH,
            probcut_depth_sub: DEFAULT_PROBCUT_DEPTH_SUB,
            probcut_divisor: DEFAULT_PROBCUT_DIVISOR,
            low_depth_probcut_margin: DEFAULT_LOW_DEPTH_PROBCUT_MARGIN,
            iir_min_depth: DEFAULT_IIR_MIN_DEPTH,
            see_capture_linear: DEFAULT_SEE_CAPTURE_LINEAR,
            see_capture_hist_div: DEFAULT_SEE_CAPTURE_HIST_DIV,
            see_quiet_quad: DEFAULT_SEE_QUIET_QUAD,
            see_winning_threshold: DEFAULT_SEE_WINNING_THRESHOLD,
            sort_hash: DEFAULT_SORT_HASH,
            sort_winning_capture: DEFAULT_SORT_WINNING_CAPTURE,
            sort_killer1: DEFAULT_SORT_KILLER1,
            sort_killer2: DEFAULT_SORT_KILLER2,
            sort_countermove: DEFAULT_SORT_COUNTERMOVE,
            history_bonus_base: DEFAULT_HISTORY_BONUS_BASE,
            history_bonus_sub: DEFAULT_HISTORY_BONUS_SUB,
            history_bonus_cap: DEFAULT_HISTORY_BONUS_CAP,
            history_max_gravity: DEFAULT_HISTORY_MAX_GRAVITY,
            pawn_history_bonus_scale: DEFAULT_PAWN_HISTORY_BONUS_SCALE,
            pawn_history_malus_scale: DEFAULT_PAWN_HISTORY_MALUS_SCALE,
            delta_margin: DEFAULT_DELTA_MARGIN,
        }
    }
}

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
pub static SEARCH_PARAMS: Lazy<RwLock<SearchParams>> =
    Lazy::new(|| RwLock::new(SearchParams::default()));

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
pub fn set_search_params_from_json(json: &str) -> bool {
    match serde_json::from_str::<SearchParams>(json) {
        Ok(params) => match SEARCH_PARAMS.write() {
            Ok(mut guard) => {
                *guard = params;
                true
            }
            Err(_) => false,
        },
        Err(_) => false,
    }
}

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
pub fn get_search_params_as_json() -> String {
    match SEARCH_PARAMS.read() {
        Ok(guard) => serde_json::to_string(&*guard).unwrap_or_else(|_| "{}".to_string()),
        Err(_) => "{}".to_string(),
    }
}

#[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
macro_rules! param {
    ($field:ident) => {{ SEARCH_PARAMS.read().unwrap().$field }};
}

macro_rules! define_accessor {
    ($name:ident, $type:ty, $default:ident) => {
        #[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
        #[inline]
        pub fn $name() -> $type {
            param!($name)
        }

        #[cfg(not(any(feature = "param_tuning", feature = "search_tuning")))]
        #[inline]
        pub const fn $name() -> $type {
            $default
        }
    };
}

define_accessor!(razoring_linear, i32, DEFAULT_RAZORING_LINEAR);
define_accessor!(razoring_quad, i32, DEFAULT_RAZORING_QUAD);
define_accessor!(nmp_min_depth, usize, DEFAULT_NMP_MIN_DEPTH);
define_accessor!(nmp_base, i32, DEFAULT_NMP_BASE);
define_accessor!(nmp_depth_mult, i32, DEFAULT_NMP_DEPTH_MULT);
define_accessor!(nmp_reduction_base, usize, DEFAULT_NMP_REDUCTION_BASE);
define_accessor!(nmp_reduction_div, usize, DEFAULT_NMP_REDUCTION_DIV);
define_accessor!(lmr_min_depth, usize, DEFAULT_LMR_MIN_DEPTH);
define_accessor!(lmr_min_moves, usize, DEFAULT_LMR_MIN_MOVES);
define_accessor!(lmr_divisor, usize, DEFAULT_LMR_DIVISOR);
define_accessor!(lmr_cutoff_thresh, u8, DEFAULT_LMR_CUTOFF_THRESH);
define_accessor!(lmr_tt_history_thresh, i32, DEFAULT_LMR_TT_HISTORY_THRESH);
define_accessor!(hlp_max_depth, usize, DEFAULT_HLP_MAX_DEPTH);
define_accessor!(hlp_min_moves, usize, DEFAULT_HLP_MIN_MOVES);
define_accessor!(hlp_history_reduce, i32, DEFAULT_HLP_HISTORY_REDUCE);
define_accessor!(hlp_history_leaf, i32, DEFAULT_HLP_HISTORY_LEAF);
define_accessor!(lmp_base, usize, DEFAULT_LMP_BASE);
define_accessor!(lmp_depth_mult, usize, DEFAULT_LMP_DEPTH_MULT);
define_accessor!(aspiration_window, i32, DEFAULT_ASPIRATION_WINDOW);
define_accessor!(aspiration_fail_mult, i32, DEFAULT_ASPIRATION_FAIL_MULT);
define_accessor!(aspiration_max_window, i32, DEFAULT_ASPIRATION_MAX_WINDOW);
define_accessor!(rfp_max_depth, usize, DEFAULT_RFP_MAX_DEPTH);
define_accessor!(rfp_mult_tt, i32, DEFAULT_RFP_MULT_TT);
define_accessor!(rfp_mult_no_tt, i32, DEFAULT_RFP_MULT_NO_TT);
define_accessor!(rfp_improving_mult, i32, DEFAULT_RFP_IMPROVING_MULT);
define_accessor!(rfp_worsening_mult, i32, DEFAULT_RFP_WORSENING_MULT);
define_accessor!(probcut_margin, i32, DEFAULT_PROBCUT_MARGIN);
define_accessor!(probcut_improving, i32, DEFAULT_PROBCUT_IMPROVING);
define_accessor!(probcut_min_depth, usize, DEFAULT_PROBCUT_MIN_DEPTH);
define_accessor!(probcut_depth_sub, usize, DEFAULT_PROBCUT_DEPTH_SUB);
define_accessor!(probcut_divisor, i32, DEFAULT_PROBCUT_DIVISOR);
define_accessor!(low_depth_probcut_margin, i32, DEFAULT_LOW_DEPTH_PROBCUT_MARGIN);
define_accessor!(iir_min_depth, usize, DEFAULT_IIR_MIN_DEPTH);
define_accessor!(see_capture_linear, i32, DEFAULT_SEE_CAPTURE_LINEAR);
define_accessor!(see_capture_hist_div, i32, DEFAULT_SEE_CAPTURE_HIST_DIV);
define_accessor!(see_quiet_quad, i32, DEFAULT_SEE_QUIET_QUAD);
define_accessor!(see_winning_threshold, i32, DEFAULT_SEE_WINNING_THRESHOLD);
define_accessor!(sort_hash, i32, DEFAULT_SORT_HASH);
define_accessor!(sort_winning_capture, i32, DEFAULT_SORT_WINNING_CAPTURE);
define_accessor!(sort_killer1, i32, DEFAULT_SORT_KILLER1);
define_accessor!(sort_killer2, i32, DEFAULT_SORT_KILLER2);
define_accessor!(sort_countermove, i32, DEFAULT_SORT_COUNTERMOVE);
define_accessor!(history_bonus_base, i32, DEFAULT_HISTORY_BONUS_BASE);
define_accessor!(history_bonus_sub, i32, DEFAULT_HISTORY_BONUS_SUB);
define_accessor!(history_bonus_cap, i32, DEFAULT_HISTORY_BONUS_CAP);
define_accessor!(history_max_gravity, i32, DEFAULT_HISTORY_MAX_GRAVITY);
define_accessor!(pawn_history_bonus_scale, i32, DEFAULT_PAWN_HISTORY_BONUS_SCALE);
define_accessor!(pawn_history_malus_scale, i32, DEFAULT_PAWN_HISTORY_MALUS_SCALE);
define_accessor!(delta_margin, i32, DEFAULT_DELTA_MARGIN);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        assert_eq!(razoring_linear(), DEFAULT_RAZORING_LINEAR);
        assert!(TUNABLE_PARAM_SPECS.iter().any(|spec| spec.name == "delta_margin"));
        assert!(!TUNABLE_PARAM_SPECS.iter().any(|spec| spec.name == "nmp_reduction"));
    }

    #[test]
    fn search_param_specs_have_valid_ranges_and_defaults() {
        for spec in TUNABLE_PARAM_SPECS {
            assert!(spec.min <= spec.default);
            assert!(spec.default <= spec.max);
            assert_eq!(spec.clamp_value(spec.min - 1000), spec.min);
            assert_eq!(spec.clamp_value(spec.max + 1000), spec.max);
            assert_eq!(spec.clamp_value(spec.default), spec.default);
        }
    }

    #[test]
    fn eval_param_specs_have_valid_ranges_and_queen_value_matches_formula() {
        for spec in TUNABLE_EVAL_PARAM_SPECS {
            assert!(spec.min <= spec.max);
            assert_eq!(spec.clamp_value(spec.min - 1000), spec.min);
            assert_eq!(spec.clamp_value(spec.max + 1000), spec.max);
            assert!((spec.min..=spec.max).contains(&spec.clamp_value(spec.default)));
        }

        assert_eq!(queen_value(), rook() * 2 + compound_bonus());
    }
}
