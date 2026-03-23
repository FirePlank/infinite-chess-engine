use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::{Parser, Subcommand};
use hydrochess_wasm::board::PlayerColor;
use hydrochess_wasm::evaluation::{self};
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::search::params::{
    self, EvalParams, SearchParams, TUNABLE_EVAL_PARAM_SPECS, TUNABLE_PARAM_SPECS,
};
use hydrochess_wasm::{Engine, Variant};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

static STOP: AtomicBool = AtomicBool::new(false);
static USER_STOP: AtomicBool = AtomicBool::new(false);

const DEFAULT_VARIANTS: &str = "Classical,Confined_Classical,Classical_Plus,Core,CoaIP,CoaIP_HO,CoaIP_RO,CoaIP_NO,Palace,Pawndard,Standarch,Space_Classic,Space,Scattered_Leapers";
const DEFAULT_CHECKPOINT_DIR: &str = "sprt/spsa_checkpoints";
const DEFAULT_RESULTS_PATH: &str = "sprt/spsa_final.json";
const SEARCH_PARAMS_RS_PATH: &str = "src/search/params.rs";
const EVAL_BASE_RS_PATH: &str = "src/evaluation/base.rs";

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run SPSA self-play tuning over a selected parameter set.
    Run {
        /// Number of SPSA iterations to run.
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        /// Number of paired openings per iteration. Total games = `pairs * 2`.
        #[arg(long, default_value_t = 400)]
        pairs: usize,
        /// Save a checkpoint every N iterations.
        #[arg(long, default_value_t = 1)]
        checkpoint_every: usize,
        /// Resume from a specific checkpoint JSON. If omitted, the latest checkpoint is auto-used.
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Ignore checkpoints and start from defaults for the selected parameter set.
        #[arg(long, default_value_t = false)]
        fresh: bool,
        /// Time control: `base+inc`, `depth N`, or `fixed Ns`.
        #[arg(long, default_value = "3+0.03")]
        tc: String,
        /// Number of concurrent game workers.
        #[arg(long, default_value_t = 16)]
        concurrency: usize,
        /// Comma-separated variant list to test.
        #[arg(long, default_value = DEFAULT_VARIANTS)]
        variants: String,
        /// Material-eval threshold for adjudication.
        #[arg(long, default_value_t = 2000)]
        adjudication: i32,
        /// Maximum plies before forcing a draw.
        #[arg(long, default_value_t = 300)]
        max_moves: usize,
        /// Search noise amplitude for the first 8 ply.
        #[arg(long, default_value_t = 50)]
        search_noise: i32,
        /// Directory where checkpoint JSON files are stored.
        #[arg(long, default_value = DEFAULT_CHECKPOINT_DIR)]
        checkpoint_dir: PathBuf,
        /// Output path for the final SPSA result JSON.
        #[arg(long, default_value = DEFAULT_RESULTS_PATH)]
        results: PathBuf,
        /// Optional path to write the latest iteration's ICNs as JSON.
        #[arg(long)]
        games: Option<PathBuf>,
        /// SPSA stability parameter `A`. Defaults to `iterations / 10`.
        #[arg(long)]
        big_a: Option<f64>,
        /// SPSA learning-rate decay exponent `alpha`.
        #[arg(long, default_value_t = 0.602)]
        alpha: f64,
        /// SPSA perturbation decay exponent `gamma`.
        #[arg(long, default_value_t = 0.101)]
        gamma: f64,
        /// Print verbose search subprocess stderr.
        #[arg(long, default_value_t = false)]
        verbose: bool,
        /// Parameter selection preset or comma-separated names: `all`, `search`, `eval`, `piece-values`, or explicit names.
        #[arg(long, default_value = "all")]
        params: String,
        /// Optional JSON file overriding bounds, defaults, `c_end`, or `r_end` for selected params.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Internal search subprocess used by the SPSA match runner.
    Search {
        /// ICN string of the position to search.
        #[arg(long, required = true)]
        icn: String,
        /// White time remaining in milliseconds.
        #[arg(long, default_value_t = 0)]
        wtime: u64,
        /// Black time remaining in milliseconds.
        #[arg(long, default_value_t = 0)]
        btime: u64,
        /// White increment in milliseconds.
        #[arg(long, default_value_t = 0)]
        winc: u64,
        /// Black increment in milliseconds.
        #[arg(long, default_value_t = 0)]
        binc: u64,
        /// Variant name.
        #[arg(long, default_value = "Classical")]
        variant: String,
        /// Maximum search depth override.
        #[arg(long)]
        max_depth: Option<usize>,
        /// Fixed time budget in milliseconds.
        #[arg(long)]
        fixed_time: Option<u32>,
        /// Search noise amplitude.
        #[arg(long)]
        noise_amp: Option<i32>,
        /// Random seed.
        #[arg(long)]
        seed: Option<u64>,
        /// JSON-encoded search params override for this subprocess.
        #[arg(long)]
        search_params_json: Option<String>,
        /// JSON-encoded eval params override for this subprocess.
        #[arg(long)]
        eval_params_json: Option<String>,
    },
    /// Apply tuned values from a result or checkpoint JSON back into the Rust source constants.
    Apply {
        /// Optional result/checkpoint JSON path. Defaults to the latest checkpoint.
        #[arg(long)]
        input: Option<PathBuf>,
    },
    Revert {
        /// Parameter selection preset or comma-separated names: `all`, `search`, `eval`, `piece-values`, or explicit names.
        #[arg(long, default_value = "all")]
        params: String,
    },
    List {
        /// Parameter selection preset or comma-separated names: `all`, `search`, `eval`, `piece-values`, or explicit names.
        #[arg(long, default_value = "all")]
        params: String,
    },
}

#[derive(Clone, Debug)]
struct RunConfig {
    tc: String,
    tc_base_ms: u64,
    tc_inc_ms: u64,
    tc_fixed_ms: Option<u32>,
    tc_max_depth: Option<usize>,
    concurrency: usize,
    variants: Vec<Variant>,
    adjudication_threshold: i32,
    max_moves: usize,
    search_noise: i32,
    verbose: bool,
    engine_bin: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GameResult {
    Win,
    Loss,
    Draw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameOutcome {
    result: GameResult,
    variant_name: String,
    termination_reason: String,
    icn: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IterationRecord {
    iteration: usize,
    plus_wins: usize,
    minus_wins: usize,
    draws: usize,
    score: f64,
    elo: f64,
    elapsed_ms: u128,
    changed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    iteration: usize,
    theta: BTreeMap<String, f64>,
    best_theta: BTreeMap<String, f64>,
    best_iteration: usize,
    best_score: f64,
    best_elo: f64,
    history: Vec<IterationRecord>,
    selected: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BatchResults {
    plus_wins: usize,
    minus_wins: usize,
    draws: usize,
    games: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Domain {
    Search,
    Eval,
}

#[derive(Debug, Clone)]
struct UnifiedSpec {
    domain: Domain,
    name: &'static str,
    default: i64,
    min: i64,
    max: i64,
    c_end: f64,
    r_end: f64,
    description: &'static str,
}

impl UnifiedSpec {
    fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min as f64, self.max as f64)
    }

    fn quantize(&self, value: f64) -> i64 {
        self.clamp(value).round() as i64
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SpecOverride {
    min: Option<i64>,
    max: Option<i64>,
    default: Option<i64>,
    c_end: Option<f64>,
    r_end: Option<f64>,
}

fn all_specs() -> Vec<UnifiedSpec> {
    let mut specs = Vec::new();
    for spec in TUNABLE_PARAM_SPECS {
        specs.push(UnifiedSpec {
            domain: Domain::Search,
            name: spec.name,
            default: spec.default,
            min: spec.min,
            max: spec.max,
            c_end: spec.c_end,
            r_end: spec.r_end,
            description: spec.description,
        });
    }
    for spec in TUNABLE_EVAL_PARAM_SPECS {
        specs.push(UnifiedSpec {
            domain: Domain::Eval,
            name: spec.name,
            default: spec.default,
            min: spec.min,
            max: spec.max,
            c_end: spec.c_end,
            r_end: spec.r_end,
            description: spec.description,
        });
    }
    specs
}

fn piece_value_names() -> &'static [&'static str] {
    &[
        "pawn",
        "knight",
        "bishop",
        "rook",
        "guard",
        "centaur",
        "compound_bonus",
        "camel",
        "giraffe",
        "zebra",
        "knightrider",
        "hawk",
        "archbishop",
        "rose",
        "huygen",
        "chancellor_bonus",
    ]
}

fn select_specs(selector: &str, override_path: Option<&Path>) -> Vec<UnifiedSpec> {
    let all = all_specs();
    let by_name: HashMap<&str, UnifiedSpec> =
        all.iter().map(|spec| (spec.name, spec.clone())).collect();
    let mut names = Vec::new();
    for token in selector
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        match token {
            "all" => names.extend(all.iter().map(|spec| spec.name)),
            "search" => names.extend(
                all.iter()
                    .filter(|spec| spec.domain == Domain::Search)
                    .map(|spec| spec.name),
            ),
            "eval" => names.extend(
                all.iter()
                    .filter(|spec| spec.domain == Domain::Eval)
                    .map(|spec| spec.name),
            ),
            "piece-values" | "material" => names.extend(piece_value_names().iter().copied()),
            other => names.push(other),
        }
    }
    if names.is_empty() {
        names.extend(all.iter().map(|spec| spec.name));
    }
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for name in names {
        if seen.insert(name) {
            if name == "pawn" {
                eprintln!(
                    "note: `pawn` is fixed at 100 and is not a tunable parameter; ignoring selector"
                );
                continue;
            }
            let spec = by_name
                .get(name)
                .unwrap_or_else(|| panic!("unknown parameter selection: {}. Use `spsa list --params all` to inspect valid names.", name));
            selected.push(spec.clone());
        }
    }
    if let Some(path) = override_path {
        let root: HashMap<String, SpecOverride> =
            serde_json::from_str(&fs::read_to_string(path).expect("read tuning config"))
                .expect("parse tuning config");
        for spec in &mut selected {
            if let Some(ov) = root.get(spec.name) {
                if let Some(v) = ov.default {
                    spec.default = v;
                }
                if let Some(v) = ov.min {
                    spec.min = v;
                }
                if let Some(v) = ov.max {
                    spec.max = v;
                }
                if let Some(v) = ov.c_end {
                    spec.c_end = v.max(f64::EPSILON);
                }
                if let Some(v) = ov.r_end {
                    spec.r_end = v.max(f64::EPSILON);
                }
                if spec.min > spec.max {
                    std::mem::swap(&mut spec.min, &mut spec.max);
                }
                spec.default = spec.default.clamp(spec.min, spec.max);
            }
        }
    }
    selected
}

fn score_to_elo(score: f64) -> f64 {
    let clamped = score.clamp(0.001, 0.999);
    -400.0 * ((1.0 / clamped) - 1.0).log10()
}
fn move_to_string(m: &hydrochess_wasm::moves::Move) -> String {
    let mut s = format!("{},{} {},{}", m.from.x, m.from.y, m.to.x, m.to.y);
    if let Some(p) = m.promotion {
        s.push_str(&format!(" {}", p.to_site_code().to_lowercase()));
    }
    s
}

fn parse_bestmove_to_icn(bestmove_str: &str, turn: PlayerColor) -> Option<String> {
    let parts: Vec<&str> = bestmove_str.split_whitespace().collect();
    if parts.len() < 2 || parts[0] == "none" {
        return None;
    }
    let mut result = format!("{}>{}", parts[0], parts[1]);
    if parts.len() > 2 {
        let promo = if turn == PlayerColor::White {
            parts[2].to_uppercase()
        } else {
            parts[2].to_lowercase()
        };
        result.push('=');
        result.push_str(&promo);
    }
    Some(result)
}

fn has_any_fully_legal_move(game: &mut GameState) -> bool {
    for m in game.get_legal_moves() {
        let undo = game.make_move(&m);
        let legal = !game.is_move_illegal();
        game.undo_move(&m, undo);
        if legal {
            return true;
        }
    }
    false
}

fn terminal_reason(game: &mut GameState) -> Option<(&'static str, Option<bool>)> {
    let in_check = game.is_in_check();
    if !has_any_fully_legal_move(game) {
        let lost_by_mate = in_check && game.must_escape_check();
        let lost_by_piece_capture = !game.has_pieces(game.turn);
        if lost_by_mate || lost_by_piece_capture {
            return Some(("checkmate", Some(game.turn == PlayerColor::Black)));
        }
        return Some(("stalemate", None));
    }
    if game.is_draw(0, in_check) {
        if game.is_fifty() {
            return Some(("fifty-move rule", None));
        }
        if game.is_repetition(0) {
            return Some(("threefold repetition", None));
        }
    }
    None
}

fn play_game(
    config: &RunConfig,
    variant: Variant,
    plus_white: bool,
    seeds: &[u64],
    plus_search_json: Option<&str>,
    minus_search_json: Option<&str>,
    plus_eval_json: Option<&str>,
    minus_eval_json: Option<&str>,
) -> GameOutcome {
    let mut game = GameState::new();
    game.setup_position_from_icn(variant.starting_icn());
    game.variant = Some(variant);
    let starting = variant.starting_icn().to_string();
    let mut white_clock = config.tc_base_ms;
    let mut black_clock = config.tc_base_ms;
    let mut moves = Vec::new();

    let eval_fn = |g: &GameState| {
        #[cfg(feature = "nnue")]
        {
            hydrochess_wasm::evaluation::evaluate(g, None)
        }
        #[cfg(not(feature = "nnue"))]
        {
            hydrochess_wasm::evaluation::evaluate(g)
        }
    };

    for ply in 0..config.max_moves {
        if STOP.load(Ordering::SeqCst) {
            return GameOutcome {
                result: GameResult::Draw,
                variant_name: variant.to_str().to_string(),
                termination_reason: "interrupted".to_string(),
                icn: String::new(),
            };
        }
        if let Some((reason, white_won)) = terminal_reason(&mut game) {
            let result = match white_won {
                Some(white_won) => {
                    if white_won == plus_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    }
                }
                None => GameResult::Draw,
            };
            let icn = if moves.is_empty() {
                starting.clone()
            } else {
                format!("{} {}", starting, moves.join("|"))
            };
            return GameOutcome {
                result,
                variant_name: variant.to_str().to_string(),
                termination_reason: reason.to_string(),
                icn,
            };
        }
        let eval = eval_fn(&game);
        if eval.abs() >= config.adjudication_threshold {
            let white_won = eval > 0;
            let result = if white_won == plus_white {
                GameResult::Win
            } else {
                GameResult::Loss
            };
            let icn = if moves.is_empty() {
                starting.clone()
            } else {
                format!("{} {}", starting, moves.join("|"))
            };
            return GameOutcome {
                result,
                variant_name: variant.to_str().to_string(),
                termination_reason: "material adjudication".to_string(),
                icn,
            };
        }

        let is_plus_turn = (game.turn == PlayerColor::White) == plus_white;
        let search_json = if is_plus_turn {
            plus_search_json
        } else {
            minus_search_json
        };
        let eval_json = if is_plus_turn {
            plus_eval_json
        } else {
            minus_eval_json
        };
        let subprocess_icn = if moves.is_empty() {
            starting.clone()
        } else {
            format!("{} {}", starting, moves.join("|"))
        };

        let mut cmd = Command::new(&config.engine_bin);
        cmd.env("RAYON_NUM_THREADS", "1")
            .arg("search")
            .arg("--icn")
            .arg(&subprocess_icn)
            .arg("--wtime")
            .arg(white_clock.to_string())
            .arg("--btime")
            .arg(black_clock.to_string())
            .arg("--winc")
            .arg(config.tc_inc_ms.to_string())
            .arg("--binc")
            .arg(config.tc_inc_ms.to_string())
            .arg("--variant")
            .arg(variant.to_str())
            .arg("--seed")
            .arg(seeds[ply].to_string());
        if let Some(json) = search_json {
            cmd.arg("--search-params-json").arg(json);
        }
        if let Some(json) = eval_json {
            cmd.arg("--eval-params-json").arg(json);
        }
        if let Some(depth) = config.tc_max_depth {
            cmd.arg("--max-depth").arg(depth.to_string());
        }
        if let Some(fixed) = config.tc_fixed_ms {
            cmd.arg("--fixed-time").arg(fixed.to_string());
        }
        if ply < 8 {
            cmd.arg("--noise-amp").arg(config.search_noise.to_string());
        }
        if config.verbose {
            cmd.stderr(Stdio::inherit());
        }

        let started = Instant::now();
        let output = match cmd.output() {
            Ok(output) => output,
            Err(_) => {
                let result = if is_plus_turn {
                    GameResult::Loss
                } else {
                    GameResult::Win
                };
                let icn = if moves.is_empty() {
                    starting.clone()
                } else {
                    format!("{} {}", starting, moves.join("|"))
                };
                return GameOutcome {
                    result,
                    variant_name: variant.to_str().to_string(),
                    termination_reason: "engine failure".to_string(),
                    icn,
                };
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let bestmove_icn = stdout
            .lines()
            .find(|line| line.starts_with("bestmove"))
            .and_then(|line| {
                parse_bestmove_to_icn(line.trim_start_matches("bestmove").trim(), game.turn)
            });
        let elapsed = (started.elapsed().as_millis() as u64).saturating_sub(20);
        let current_clock = if game.turn == PlayerColor::White {
            white_clock
        } else {
            black_clock
        };
        if current_clock < elapsed {
            let result = if is_plus_turn {
                GameResult::Loss
            } else {
                GameResult::Win
            };
            let icn = if moves.is_empty() {
                starting.clone()
            } else {
                format!("{} {}", starting, moves.join("|"))
            };
            return GameOutcome {
                result,
                variant_name: variant.to_str().to_string(),
                termination_reason: "timeout".to_string(),
                icn,
            };
        }
        let remaining_clock = current_clock - elapsed + config.tc_inc_ms;
        if let Some(move_icn) = bestmove_icn {
            moves.push(move_icn);
            let new_icn = format!("{} {}", starting, moves.join("|"));
            let old_turn = game.turn;
            game = GameState::new();
            game.setup_position_from_icn(&new_icn);
            game.variant = Some(variant);
            if game.turn == old_turn {
                let result = if is_plus_turn {
                    GameResult::Loss
                } else {
                    GameResult::Win
                };
                return GameOutcome {
                    result,
                    variant_name: variant.to_str().to_string(),
                    termination_reason: "illegal move".to_string(),
                    icn: new_icn,
                };
            }
            if game.turn == PlayerColor::Black {
                white_clock = remaining_clock;
            } else {
                black_clock = remaining_clock;
            }
        } else {
            let result = if is_plus_turn {
                GameResult::Loss
            } else {
                GameResult::Win
            };
            let icn = if moves.is_empty() {
                starting.clone()
            } else {
                format!("{} {}", starting, moves.join("|"))
            };
            return GameOutcome {
                result,
                variant_name: variant.to_str().to_string(),
                termination_reason: "engine failure".to_string(),
                icn,
            };
        }
    }

    let icn = if moves.is_empty() {
        starting
    } else {
        format!("{} {}", starting, moves.join("|"))
    };
    GameOutcome {
        result: GameResult::Draw,
        variant_name: variant.to_str().to_string(),
        termination_reason: "max_moves".to_string(),
        icn,
    }
}
fn default_search_map() -> serde_json::Map<String, Value> {
    serde_json::to_value(SearchParams::default())
        .expect("serialize search defaults")
        .as_object()
        .unwrap()
        .clone()
}

fn default_eval_map() -> serde_json::Map<String, Value> {
    serde_json::to_value(EvalParams::default())
        .expect("serialize eval defaults")
        .as_object()
        .unwrap()
        .clone()
}

fn selected_defaults(specs: &[UnifiedSpec]) -> BTreeMap<String, f64> {
    specs
        .iter()
        .map(|spec| (spec.name.to_string(), spec.default as f64))
        .collect()
}

fn clamp_theta(theta: &BTreeMap<String, f64>, selected: &[UnifiedSpec]) -> BTreeMap<String, f64> {
    let map: HashMap<&str, &UnifiedSpec> = selected.iter().map(|spec| (spec.name, spec)).collect();
    theta
        .iter()
        .filter_map(|(name, value)| {
            map.get(name.as_str())
                .map(|spec| (name.clone(), spec.clamp(*value)))
        })
        .collect()
}

fn generate_delta(selected: &[UnifiedSpec]) -> BTreeMap<String, i64> {
    selected
        .iter()
        .map(|spec| {
            (
                spec.name.to_string(),
                if rand::random::<bool>() { 1 } else { -1 },
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct SpsaCoefficients {
    ck: f64,
    rk: f64,
}

fn ck_for(spec: &UnifiedSpec, iteration: usize, iterations: usize, gamma: f64) -> f64 {
    let c = spec.c_end * (iterations as f64).powf(gamma);
    c / (iteration as f64).powf(gamma)
}

fn ak_for(spec: &UnifiedSpec, iteration: usize, iterations: usize, big_a: f64, alpha: f64) -> f64 {
    let a_end = spec.r_end * spec.c_end * spec.c_end;
    let a = a_end * (big_a + iterations as f64).powf(alpha);
    a / (big_a + iteration as f64).powf(alpha)
}

fn compute_coefficients(
    selected: &[UnifiedSpec],
    iteration: usize,
    iterations: usize,
    big_a: f64,
    alpha: f64,
    gamma: f64,
) -> BTreeMap<String, SpsaCoefficients> {
    selected
        .iter()
        .map(|spec| {
            let ck = ck_for(spec, iteration, iterations, gamma);
            let ak = ak_for(spec, iteration, iterations, big_a, alpha);
            let rk = ak / (ck * ck);
            (
                spec.name.to_string(),
                SpsaCoefficients { ck, rk },
            )
        })
        .collect()
}

fn perturb(
    theta: &BTreeMap<String, f64>,
    delta: &BTreeMap<String, i64>,
    coeffs: &BTreeMap<String, SpsaCoefficients>,
    sign: f64,
    selected: &[UnifiedSpec],
) -> BTreeMap<String, f64> {
    let map: HashMap<&str, &UnifiedSpec> = selected.iter().map(|spec| (spec.name, spec)).collect();
    theta
        .iter()
        .map(|(name, value)| {
            let spec = map[name.as_str()];
            let ck = coeffs[name].ck;
            let offset = sign * ck * delta[name] as f64;
            (
                name.clone(),
                spec.clamp(*value + offset),
            )
        })
        .collect()
}

fn update(
    theta: &BTreeMap<String, f64>,
    delta: &BTreeMap<String, i64>,
    coeffs: &BTreeMap<String, SpsaCoefficients>,
    result: f64,
    selected: &[UnifiedSpec],
) -> BTreeMap<String, f64> {
    let map: HashMap<&str, &UnifiedSpec> = selected.iter().map(|spec| (spec.name, spec)).collect();
    theta
        .iter()
        .map(|(name, value)| {
            let spec = map[name.as_str()];
            let rk = coeffs[name].rk;
            let ck = coeffs[name].ck;
            let update = rk * ck * result / delta[name] as f64;
            let next = *value + update;
            (name.clone(), spec.clamp(next))
        })
        .collect()
}

fn build_json_for_domain(theta: &BTreeMap<String, f64>, domain: Domain) -> Option<String> {
    let mut map = match domain {
        Domain::Search => default_search_map(),
        Domain::Eval => default_eval_map(),
    };
    let mut touched = false;
    for spec in all_specs().iter().filter(|spec| spec.domain == domain) {
        if let Some(value) = theta.get(spec.name) {
            map.insert(
                spec.name.to_string(),
                Value::Number(Number::from(spec.quantize(*value))),
            );
            touched = true;
        }
    }
    if touched {
        Some(Value::Object(map).to_string())
    } else {
        None
    }
}

fn run_batch(
    config: &RunConfig,
    pairs: usize,
    plus_theta: &BTreeMap<String, f64>,
    minus_theta: &BTreeMap<String, f64>,
) -> BatchResults {
    let plus_search_json = build_json_for_domain(plus_theta, Domain::Search);
    let minus_search_json = build_json_for_domain(minus_theta, Domain::Search);
    let plus_eval_json = build_json_for_domain(plus_theta, Domain::Eval);
    let minus_eval_json = build_json_for_domain(minus_theta, Domain::Eval);
    let (tx, rx) = std::sync::mpsc::channel();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.concurrency.max(1))
        .build()
        .expect("build thread pool");
    let config_clone = config.clone();
    std::thread::spawn(move || {
        pool.install(|| {
            rayon::scope(|scope| {
                for worker_idx in 0..config_clone.concurrency.max(1) {
                    let tx = tx.clone();
                    let config = config_clone.clone();
                    let plus_search_json = plus_search_json.clone();
                    let minus_search_json = minus_search_json.clone();
                    let plus_eval_json = plus_eval_json.clone();
                    let minus_eval_json = minus_eval_json.clone();
                    scope.spawn(move |_| {
                        let mut pair_idx = worker_idx;
                        let step = config.concurrency.max(1);
                        while pair_idx < pairs && !STOP.load(Ordering::SeqCst) {
                            let variant = config.variants[pair_idx % config.variants.len()];
                            let mut seeds = Vec::with_capacity(config.max_moves);
                            for _ in 0..config.max_moves {
                                seeds.push(rand::random::<u64>());
                            }
                            let mut outcomes = Vec::with_capacity(2);
                            if rand::random::<bool>() {
                                outcomes.push(play_game(
                                    &config,
                                    variant,
                                    true,
                                    &seeds,
                                    plus_search_json.as_deref(),
                                    minus_search_json.as_deref(),
                                    plus_eval_json.as_deref(),
                                    minus_eval_json.as_deref(),
                                ));
                                if !STOP.load(Ordering::SeqCst) {
                                    outcomes.push(play_game(
                                        &config,
                                        variant,
                                        false,
                                        &seeds,
                                        plus_search_json.as_deref(),
                                        minus_search_json.as_deref(),
                                        plus_eval_json.as_deref(),
                                        minus_eval_json.as_deref(),
                                    ));
                                }
                            } else {
                                outcomes.push(play_game(
                                    &config,
                                    variant,
                                    false,
                                    &seeds,
                                    plus_search_json.as_deref(),
                                    minus_search_json.as_deref(),
                                    plus_eval_json.as_deref(),
                                    minus_eval_json.as_deref(),
                                ));
                                if !STOP.load(Ordering::SeqCst) {
                                    outcomes.push(play_game(
                                        &config,
                                        variant,
                                        true,
                                        &seeds,
                                        plus_search_json.as_deref(),
                                        minus_search_json.as_deref(),
                                        plus_eval_json.as_deref(),
                                        minus_eval_json.as_deref(),
                                    ));
                                }
                            }
                            if tx.send(outcomes).is_err() {
                                break;
                            }
                            pair_idx += step;
                        }
                    });
                }
            });
        });
    });

    let mut results = BatchResults::default();
    for outcomes in rx {
        for outcome in outcomes {
            if outcome.termination_reason == "interrupted" {
                continue;
            }
            match outcome.result {
                GameResult::Win => results.plus_wins += 1,
                GameResult::Loss => results.minus_wins += 1,
                GameResult::Draw => results.draws += 1,
            }
            results.games.push(outcome.icn);
        }
        if results.plus_wins + results.minus_wins + results.draws >= pairs * 2 {
            break;
        }
    }
    results
}

fn parse_tc(config: &mut RunConfig) {
    if config.tc.contains('+') {
        let parts: Vec<&str> = config.tc.split('+').collect();
        config.tc_base_ms = (parts[0].parse::<f64>().expect("Invalid base time") * 1000.0) as u64;
        config.tc_inc_ms = (parts[1].parse::<f64>().expect("Invalid increment") * 1000.0) as u64;
    } else if config.tc.starts_with("depth ") {
        config.tc_max_depth = Some(
            config
                .tc
                .replace("depth ", "")
                .parse()
                .expect("Invalid depth"),
        );
    } else if config.tc.starts_with("fixed ") {
        config.tc_fixed_ms = Some(
            (config
                .tc
                .replace("fixed ", "")
                .replace('s', "")
                .parse::<f64>()
                .expect("Invalid fixed time")
                * 1000.0) as u32,
        );
    }
}

fn checkpoint_path(dir: &Path, iteration: usize) -> PathBuf {
    dir.join(format!("spsa_{:06}.json", iteration))
}

fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut entries = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect::<Vec<_>>();
    entries.sort();
    entries.pop()
}

fn save_checkpoint(path: &Path, checkpoint: &Checkpoint) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(
        path,
        serde_json::to_string_pretty(checkpoint).expect("serialize checkpoint"),
    )
    .expect("write checkpoint");
}

fn load_checkpoint(path: &Path) -> Checkpoint {
    serde_json::from_str(&fs::read_to_string(path).expect("read checkpoint"))
        .expect("parse checkpoint")
}
fn search_const_name(name: &str) -> String {
    format!("DEFAULT_{}", name.to_ascii_uppercase())
}
fn eval_const_name(name: &str) -> String {
    format!("DEFAULT_EVAL_{}", name.to_ascii_uppercase())
}

fn apply_constants(path: &str, values: &[(String, String, i64)]) {
    let mut content = fs::read_to_string(path).expect("read constants file");
    for (_, const_name, value) in values {
        let lines: Vec<String> = content.lines().map(|line| line.to_string()).collect();
        let mut out = Vec::with_capacity(lines.len());
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("pub const ")
                && let Some(rest) = trimmed.strip_prefix("pub const ")
                && let Some((name_part, _)) = rest.split_once(':')
                && name_part.trim() == const_name
                && let Some(eq_idx) = line.find('=')
                && let Some(semi_idx) = line.find(';')
                && eq_idx < semi_idx
            {
                out.push(format!(
                    "{} {}{}",
                    &line[..eq_idx + 1],
                    value,
                    &line[semi_idx..]
                ));
            } else {
                out.push(line);
            }
        }
        content = out.join("\n");
        content.push('\n');
    }
    fs::write(path, content).expect("write constants file");
}

fn apply_selected_values(theta: &BTreeMap<String, f64>) {
    let mut search_updates = Vec::new();
    let mut eval_updates = Vec::new();
    for spec in all_specs() {
        if let Some(value) = theta.get(spec.name) {
            let quantized = spec.quantize(*value);
            match spec.domain {
                Domain::Search => search_updates.push((
                    spec.name.to_string(),
                    search_const_name(spec.name),
                    quantized,
                )),
                Domain::Eval => {
                    eval_updates.push((spec.name.to_string(), eval_const_name(spec.name), quantized))
                }
            }
        }
    }
    if !search_updates.is_empty() {
        apply_constants(SEARCH_PARAMS_RS_PATH, &search_updates);
    }
    if !eval_updates.is_empty() {
        apply_constants(EVAL_BASE_RS_PATH, &eval_updates);
    }
}

fn load_values(path: &Path) -> BTreeMap<String, f64> {
    let root: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("read json")).expect("parse json");
    let obj = root.as_object().expect("json object");
    let source = if let Some(theta) = obj.get("theta") {
        theta.as_object().expect("theta object")
    } else {
        obj
    };
    let specs: HashMap<&str, UnifiedSpec> = all_specs()
        .into_iter()
        .map(|spec| (spec.name, spec))
        .collect();
    source
        .iter()
        .filter_map(|(name, value)| {
            specs.get(name.as_str()).and_then(|spec| {
                value
                    .as_i64()
                    .or_else(|| value.as_u64().map(|v| v as i64))
                    .or_else(|| value.as_f64().map(|v| v.round() as i64))
                    .map(|raw| (name.clone(), spec.clamp(raw as f64)))
            })
        })
        .collect()
}

fn run_spsa(
    iterations: usize,
    pairs: usize,
    checkpoint_every: usize,
    resume: Option<PathBuf>,
    fresh: bool,
    mut config: RunConfig,
    checkpoint_dir: PathBuf,
    results_path: PathBuf,
    games_path: Option<PathBuf>,
    big_a: f64,
    alpha: f64,
    gamma: f64,
    params: String,
    spec_config: Option<PathBuf>,
) {
    if cfg!(debug_assertions) {
        println!("Warning: debug build. Use --release for real tuning throughput.");
    }
    parse_tc(&mut config);
    ctrlc::set_handler(|| {
        USER_STOP.store(true, Ordering::SeqCst);
        STOP.store(true, Ordering::SeqCst);
    })
    .expect("install ctrl-c handler");

    let selected = select_specs(&params, spec_config.as_deref());
    println!(
        "Tuning {} params: {}",
        selected.len(),
        selected
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut theta = selected_defaults(&selected);
    let mut best_theta = theta.clone();
    let mut best_iteration = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_elo = f64::NEG_INFINITY;
    let mut history = Vec::new();
    let mut start_iteration = 1;
    if !fresh && let Some(path) = resume.or_else(|| latest_checkpoint(&checkpoint_dir)) {
        let checkpoint = load_checkpoint(&path);
        theta = clamp_theta(&checkpoint.theta, &selected);
        best_theta = clamp_theta(&checkpoint.best_theta, &selected);
        best_iteration = checkpoint.best_iteration;
        best_score = checkpoint.best_score;
        best_elo = checkpoint.best_elo;
        history = checkpoint.history;
        start_iteration = checkpoint.iteration + 1;
        println!("Resuming from {}", path.display());
    }

    let mut latest_games = Vec::new();
    for k in start_iteration..=iterations {
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        let started = Instant::now();
        let delta = generate_delta(&selected);
        let coeffs = compute_coefficients(&selected, k, iterations, big_a, alpha, gamma);
        let theta_plus = perturb(&theta, &delta, &coeffs, 1.0, &selected);
        let theta_minus = perturb(&theta, &delta, &coeffs, -1.0, &selected);
        let batch = run_batch(&config, pairs, &theta_plus, &theta_minus);
        latest_games = batch.games.clone();
        let total = batch.plus_wins + batch.minus_wins + batch.draws;
        if total == 0 {
            println!("Iteration {} produced no completed games", k);
            continue;
        }
        let score = (batch.plus_wins as f64 + 0.5 * batch.draws as f64) / total as f64;
        let match_result = batch.plus_wins as f64 - batch.minus_wins as f64;
        let next_theta = clamp_theta(
            &update(&theta, &delta, &coeffs, match_result, &selected),
            &selected,
        );
        let changed = next_theta
            .iter()
            .filter(|(name, value)| theta.get(*name) != Some(*value))
            .count();
        theta = next_theta;
        let elo = score_to_elo(score);
        if score > best_score {
            best_score = score;
            best_elo = elo;
            best_iteration = k;
            best_theta = theta.clone();
        }
        let elapsed_ms = started.elapsed().as_millis();
        println!(
            "iter {:>5} | + {:>4} - {:>4} = {:>4} | score {:.3} | elo {:>6.1} | changed {}",
            k, batch.plus_wins, batch.minus_wins, batch.draws, score, elo, changed
        );
        history.push(IterationRecord {
            iteration: k,
            plus_wins: batch.plus_wins,
            minus_wins: batch.minus_wins,
            draws: batch.draws,
            score,
            elo,
            elapsed_ms,
            changed,
        });
        if checkpoint_every > 0 && k % checkpoint_every == 0 {
            let checkpoint = Checkpoint {
                iteration: k,
                theta: theta.clone(),
                best_theta: best_theta.clone(),
                best_iteration,
                best_score,
                best_elo,
                history: history.clone(),
                selected: selected.iter().map(|s| s.name.to_string()).collect(),
            };
            save_checkpoint(&checkpoint_path(&checkpoint_dir, k), &checkpoint);
        }
    }

    if let Some(parent) = results_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let root = serde_json::json!({ "theta": theta, "best_theta": best_theta, "best_iteration": best_iteration, "best_score": best_score, "best_elo": best_elo, "history": history, "selected": selected.iter().map(|s| s.name).collect::<Vec<_>>() });
    fs::write(
        &results_path,
        serde_json::to_string_pretty(&root).expect("serialize final"),
    )
    .expect("write final results");
    println!("Saved {}", results_path.display());
    if let Some(path) = games_path {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(
            &path,
            serde_json::to_string_pretty(&latest_games).expect("serialize games"),
        )
        .expect("write games");
    }
    if USER_STOP.load(Ordering::SeqCst) {
        println!("Stopped by user");
    }
}

fn main() {
    match Cli::parse().command {
        Some(Commands::Run {
            iterations,
            pairs,
            checkpoint_every,
            resume,
            fresh,
            tc,
            concurrency,
            variants,
            adjudication,
            max_moves,
            search_noise,
            checkpoint_dir,
            results,
            games,
            big_a,
            alpha,
            gamma,
            verbose,
            params,
            config,
        }) => {
            let config_run = RunConfig {
                tc,
                tc_base_ms: 10_000,
                tc_inc_ms: 100,
                tc_fixed_ms: None,
                tc_max_depth: None,
                concurrency,
                variants: variants.split(',').map(Variant::parse).collect(),
                adjudication_threshold: adjudication,
                max_moves,
                search_noise,
                verbose,
                engine_bin: std::env::current_exe().expect("current exe"),
            };
            run_spsa(
                iterations,
                pairs,
                checkpoint_every,
                resume,
                fresh,
                config_run,
                checkpoint_dir,
                results,
                games,
                big_a.unwrap_or(iterations as f64 / 10.0),
                alpha,
                gamma,
                params,
                config,
            );
        }
        Some(Commands::Search {
            icn,
            wtime,
            btime,
            winc,
            binc,
            variant,
            max_depth,
            fixed_time,
            noise_amp,
            seed,
            search_params_json,
            eval_params_json,
        }) => {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Some(json) = search_params_json.as_deref() {
                    let _ = params::set_search_params_from_json(json);
                }
                if let Some(json) = eval_params_json.as_deref() {
                    let _ = evaluation::set_eval_params_from_json(json);
                }
                let mut engine = Engine::from_icn_native(icn.as_str(), None);
                engine.set_clock(wtime, btime, winc, binc);
                engine.game_mut().variant = Some(Variant::parse(&variant));
                if terminal_reason(engine.game_mut()).is_some() {
                    println!("bestmove none");
                    return;
                }
                let search_res = if let Some(ft) = fixed_time {
                    engine.search_native(ft, max_depth, true, noise_amp, seed)
                } else {
                    engine.search_native(0, max_depth, true, noise_amp, seed)
                };
                if let Some((m, score, stats)) = search_res {
                    println!("bestmove {}", move_to_string(&m));
                    let pv = engine.current_pv_native(max_depth.unwrap_or(50));
                    if pv.is_empty() {
                        eprintln!("info score {} nodes {}", score, stats.nodes);
                    } else {
                        eprintln!("info score {} nodes {} pv {}", score, stats.nodes, pv);
                    }
                } else {
                    println!("bestmove none");
                }
            }));
            if result.is_err() {
                println!("bestmove none");
            }
        }
        Some(Commands::Apply { input }) => {
            let path = input
                .or_else(|| latest_checkpoint(Path::new(DEFAULT_CHECKPOINT_DIR)))
                .expect("No checkpoint json found");
            apply_selected_values(&load_values(&path));
        }
        Some(Commands::Revert { params }) => {
            let selected = select_specs(&params, None);
            apply_selected_values(&selected_defaults(&selected));
        }
        Some(Commands::List { params }) => {
            for spec in select_specs(&params, None) {
                println!(
                    "{:<28} {:<6} [{:>5}, {:>5}] c_end {:>8.3} R_end {:>7.4} {}",
                    spec.name,
                    match spec.domain {
                        Domain::Search => "search",
                        Domain::Eval => "eval",
                    },
                    spec.min,
                    spec.max,
                    spec.c_end,
                    spec.r_end,
                    spec.description
                );
            }
        }
        None => println!("Use --help for usage."),
    }
}
