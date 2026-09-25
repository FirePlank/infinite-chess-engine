//! Search tuning parameters, each declared once in the `search_params!` table as
//! `name: type = default => (min, max, c_end, r_end, description)`. The table
//! generates the accessors, `SearchParams` and `TUNABLE_PARAM_SPECS`, so external
//! scripts cannot drift from the engine; spsa rewrites its defaults in place.

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

macro_rules! param_kind {
    (i32) => {
        SearchParamKind::I32
    };
    (usize) => {
        SearchParamKind::Usize
    };
    (u8) => {
        SearchParamKind::U8
    };
}

macro_rules! search_params {
    ($(
        $(#[$meta:meta])*
        $name:ident: $ty:ident = $default:literal
        => ($min:literal, $max:literal, $c_end:literal, $r_end:literal, $desc:literal);
    )*) => {
        #[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(default)]
        pub struct SearchParams {
            $(pub $name: $ty,)*
        }

        #[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
        impl Default for SearchParams {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        $(
            $(#[$meta])*
            #[cfg(any(feature = "param_tuning", feature = "search_tuning"))]
            #[inline]
            pub fn $name() -> $ty {
                param!($name)
            }

            $(#[$meta])*
            #[cfg(not(any(feature = "param_tuning", feature = "search_tuning")))]
            #[inline]
            pub const fn $name() -> $ty {
                $default
            }
        )*

        pub const TUNABLE_PARAM_SPECS: &[SearchParamSpec] = &[
            $(
                SearchParamSpec::new(
                    stringify!($name), param_kind!($ty), $default, $min, $max, $c_end, $r_end, $desc,
                ),
            )*
        ];
    };
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

search_params! {
    /// Razoring margin per ply of depth.
    razoring_quad: i32 = 232 => (100, 500, 12.0, 0.002, "Razoring quadratic margin");
    nmp_min_depth: usize = 3 => (1, 8, 1.0, 0.002, "Null move minimum depth");
    nmp_base: i32 = 350 => (100, 600, 16.0, 0.002, "Null move base margin");
    nmp_depth_mult: i32 = 36 => (8, 48, 2.0, 0.002, "Null move depth multiplier");
    nmp_reduction_base: usize = 7 => (2, 12, 1.0, 0.002, "Null move reduction numerator");
    nmp_reduction_div: usize = 2 => (1, 8, 1.0, 0.002, "Null move reduction divisor");
    lmr_min_depth: usize = 3 => (1, 8, 1.0, 0.002, "Late move reduction minimum depth");
    lmr_min_moves: usize = 4 => (1, 16, 1.0, 0.002, "Late move reduction minimum move count");
    lmr_divisor: usize = 2 => (1, 8, 1.0, 0.002, "Late move reduction divisor");
    lmr_cutoff_thresh: u8 = 2 => (1, 8, 1.0, 0.002, "Late move reduction cutoff threshold");
    lmr_tt_history_thresh: i32 = -1000 => (-4000, 0, 64.0, 0.002, "Late move reduction TT history threshold");
    hlp_max_depth: usize = 3 => (1, 8, 1.0, 0.002, "History leaf pruning maximum depth");
    hlp_min_moves: usize = 4 => (1, 16, 1.0, 0.002, "History leaf pruning minimum move count");
    hlp_history_reduce: i32 = 300 => (-2000, 2000, 64.0, 0.002, "History threshold for extra late-move reduction");
    hlp_history_leaf: i32 = 0 => (-2000, 2000, 64.0, 0.002, "History threshold for pruning leaf moves");
    lmp_base: usize = 3 => (1, 12, 1.0, 0.002, "Late move pruning base");
    lmp_depth_mult: usize = 1 => (0, 6, 1.0, 0.002, "Late move pruning depth multiplier");
    aspiration_window: i32 = 60 => (8, 256, 8.0, 0.002, "Initial aspiration window");
    aspiration_fail_mult: i32 = 4 => (2, 8, 1.0, 0.002, "Aspiration fail expansion multiplier");
    aspiration_max_window: i32 = 1000 => (256, 4000, 64.0, 0.002, "Maximum aspiration window");
    rfp_max_depth: usize = 14 => (1, 20, 1.0, 0.002, "Reverse futility maximum depth");
    rfp_mult_tt: i32 = 101 => (1, 256, 4.0, 0.002, "Reverse futility TT multiplier");
    rfp_mult_no_tt: i32 = 70 => (1, 256, 4.0, 0.002, "Reverse futility non-TT multiplier");
    rfp_improving_mult: i32 = 2474 => (256, 4096, 64.0, 0.002, "Reverse futility improving multiplier");
    rfp_worsening_mult: i32 = 331 => (0, 2048, 32.0, 0.002, "Reverse futility worsening multiplier");
    probcut_margin: i32 = 235 => (0, 512, 8.0, 0.002, "ProbCut margin");
    probcut_improving: i32 = 63 => (0, 256, 4.0, 0.002, "ProbCut improving adjustment");
    probcut_min_depth: usize = 5 => (1, 12, 1.0, 0.002, "ProbCut minimum depth");
    probcut_depth_sub: usize = 5 => (1, 8, 1.0, 0.002, "ProbCut depth subtraction");
    probcut_divisor: i32 = 315 => (32, 1024, 16.0, 0.002, "ProbCut static-eval divisor");
    low_depth_probcut_margin: i32 = 800 => (128, 2048, 32.0, 0.002, "Low-depth ProbCut margin");
    iir_min_depth: usize = 3 => (1, 12, 1.0, 0.002, "Internal iterative reduction minimum depth");
    see_capture_linear: i32 = 166 => (0, 512, 8.0, 0.002, "SEE capture pruning linear term");
    see_capture_hist_div: i32 = 29 => (1, 128, 2.0, 0.002, "SEE capture history divisor");
    see_quiet_quad: i32 = 25 => (1, 128, 2.0, 0.002, "SEE quiet pruning quadratic term");
    see_winning_threshold: i32 = 0 => (-256, 256, 8.0, 0.002, "SEE threshold for classifying winning captures");
    sort_hash: i32 = 6_000_000 => (1_000_000, 10_000_000, 100_000.0, 0.002, "Hash move ordering bonus");
    sort_winning_capture: i32 = 1_000_000 => (100_000, 4_000_000, 50_000.0, 0.002, "Winning capture ordering bonus");
    sort_killer1: i32 = 900_000 => (100_000, 4_000_000, 50_000.0, 0.002, "Primary killer ordering bonus");
    sort_killer2: i32 = 800_000 => (100_000, 4_000_000, 50_000.0, 0.002, "Secondary killer ordering bonus");
    sort_countermove: i32 = 600_000 => (100_000, 4_000_000, 50_000.0, 0.002, "Countermove ordering bonus");
    history_bonus_base: i32 = 300 => (0, 1024, 16.0, 0.002, "History bonus base");
    history_bonus_sub: i32 = 250 => (0, 1024, 16.0, 0.002, "History bonus subtraction");
    history_bonus_cap: i32 = 1536 => (64, 8192, 64.0, 0.002, "History bonus cap");
    history_max_gravity: i32 = 16384 => (1024, 32768, 256.0, 0.002, "History gravity clamp");
    pawn_history_bonus_scale: i32 = 2 => (0, 8, 1.0, 0.002, "Pawn history bonus scale");
    pawn_history_malus_scale: i32 = 1 => (0, 8, 1.0, 0.002, "Pawn history malus scale");
    delta_margin: i32 = 200 => (0, 512, 8.0, 0.002, "Quiescence delta pruning margin");
}

// Fixed ordering offsets, never tuned.
pub const DEFAULT_SORT_LOSING_CAPTURE: i32 = 0;
pub const DEFAULT_SORT_QUIET: i32 = 0;

/// Rewrites the default of every parameter-table row that `new_default` names,
/// leaving all other bytes of `source` (line endings included) untouched.
/// Returns the new source and how many rows changed.
pub fn rewrite_table_defaults(
    source: &str,
    new_default: impl Fn(&str) -> Option<i64>,
) -> (String, usize) {
    let mut out = String::with_capacity(source.len());
    let mut changed = 0;
    for line in source.split_inclusive('\n') {
        match rewrite_row(line, &new_default) {
            Some(row) => {
                out.push_str(&row);
                changed += 1;
            }
            None => out.push_str(line),
        }
    }
    (out, changed)
}

fn rewrite_row(line: &str, new_default: &impl Fn(&str) -> Option<i64>) -> Option<String> {
    let (head, rest) = line.split_once(" = ")?;
    let name = head.split(':').next()?.trim();
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return None;
    }
    let value = new_default(name)?;
    let end = rest.find(" =>").or_else(|| rest.find(';'))?;
    rest[..end].trim().replace('_', "").parse::<i64>().ok()?;
    Some(format!("{head} = {value}{}", &rest[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default outside its own range makes the first SPSA perturbation clamp
    /// both candidates to the same wrong side.
    #[test]
    fn search_param_specs_have_valid_ranges_and_defaults() {
        for spec in TUNABLE_PARAM_SPECS {
            assert!(
                spec.min <= spec.default && spec.default <= spec.max,
                "{}: default {} outside [{}, {}]",
                spec.name, spec.default, spec.min, spec.max
            );
            assert_eq!(spec.clamp_value(spec.min - 1000), spec.min);
            assert_eq!(spec.clamp_value(spec.max + 1000), spec.max);
        }
    }

    #[test]
    fn test_params_default() {
        assert!(
            TUNABLE_PARAM_SPECS
                .iter()
                .any(|spec| spec.name == "delta_margin")
        );
        assert!(
            !TUNABLE_PARAM_SPECS
                .iter()
                .any(|spec| spec.name == "nmp_reduction")
        );
    }

    #[test]
    fn rewrite_table_defaults_touches_only_named_rows() {
        let src = "    a: i32 = -5 => (1, 2, 1.0, 0.002, \"x = 1;\");\r\n    b = 1_000;\n    let c = 3;\n";
        let (out, n) = rewrite_table_defaults(src, |name| match name {
            "a" => Some(7),
            "b" => Some(-2),
            "c" => Some(9),
            _ => None,
        });
        assert_eq!(n, 2);
        assert_eq!(
            out,
            "    a: i32 = 7 => (1, 2, 1.0, 0.002, \"x = 1;\");\r\n    b = -2;\n    let c = 3;\n"
        );
    }
}
