// `erf` is stable for all our other targets but nightly-gated on f64; we already
// pin nightly (rust-toolchain.toml), so use it directly instead of the libm crate.
#![feature(float_erf)]

use std::io::{IsTerminal, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use apeiron::Engine;
use apeiron::Variant;
use apeiron::board::{Coordinate, PieceType, PlayerColor};
use apeiron::game::GameState;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

// Commit SHA and date baked in at compile time by build.rs.
const BUILD_COMMIT: Option<&str> = option_env!("SPRT_GIT_COMMIT");
const BUILD_DATE: Option<&str> = option_env!("SPRT_GIT_DATE");
const BUILD_DIRTY: Option<&str> = option_env!("SPRT_GIT_DIRTY");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

// Parsed once at startup, so the variant size gap costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand, Debug)]
enum Commands {
    /// Run an SPRT session comparing two engine versions
    Run {
        /// Path to the new engine binary (if omitted, builds the current source)
        #[arg(long)]
        new_bin: Option<String>,

        /// Path to the old engine binary
        #[arg(long, required = true)]
        old_bin: String,

        /// SPRT bound H0 (Elo difference where new is NOT better)
        #[arg(long, default_value_t = 0.0)]
        elo0: f64,

        /// SPRT bound H1 (Elo difference where new IS better).
        /// Interpreted in the units of --model (normalized nElo by default).
        #[arg(long, default_value_t = 2.0)]
        elo1: f64,

        /// SPRT statistical model: "normalized" (nElo bounds, draw-rate/TC
        /// independent) or "logistic" (classic Elo-point bounds)
        #[arg(long, default_value = "normalized")]
        model: String,

        /// SPRT alpha (type I error probability)
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,

        /// SPRT beta (type II error probability)
        #[arg(long, default_value_t = 0.05)]
        beta: f64,

        /// Time control (e.g., "10+0.1", "depth 6", "fixed 0.1s")
        #[arg(long, default_value = "10+0.1")]
        tc: String,

        /// Number of parallel games (defaults to logical CPU count)
        #[arg(long)]
        concurrency: Option<usize>,

        /// Maximum games to run (omit for no limit)
        #[arg(long)]
        max_games: Option<usize>,

        /// Minimum games before SPRT can pass/fail
        #[arg(long, default_value_t = 250)]
        min_games: usize,

        /// Variants to test (comma-separated list)
        #[arg(
            long,
            default_value = "Classical,Confined_Classical,Classical_Plus,Core,CoaIP,CoaIP_HO,CoaIP_RO,CoaIP_NO,Palace,Pawndard,Standarch,Space_Classic,Space,Knightline,Scattered_Leapers"
        )]
        variants: String,

        /// Material threshold for draws
        #[arg(long, default_value_t = 0)]
        adjudication: i32,

        /// Max-ply adjudication threshold in White-ahead centipawns
        #[arg(long, default_value_t = 1000.0)]
        maxply_adjudication: f64,

        /// Path to output game ICNs
        #[arg(long)]
        games: Option<String>,

        /// Path to output results summary
        #[arg(long)]
        results: Option<String>,

        /// Maximum moves per game (game is drawn if reached)
        #[arg(long, default_value_t = 300)]
        max_moves: usize,

        /// Search noise amplitude for first 8 ply
        #[arg(long, default_value_t = 50)]
        search_noise: i32,

        /// Old engine strength level (1-8; 8 = full strength, the default)
        #[arg(long, default_value_t = 8)]
        old_strength: u32,

        /// New engine strength level (1-8; 8 = full strength, the default)
        #[arg(long, alias = "new-strenght", default_value_t = 8)]
        new_strength: u32,

        /// Print verbose engine info
        #[arg(long, default_value_t = false)]
        verbose: bool,

        /// Use the single-line status view instead of the live dashboard.
        /// Forced on automatically when stdout is not a terminal.
        #[arg(long, default_value_t = false)]
        compact: bool,

        /// Git commit SHA for the new engine (overrides the build-time embedded value)
        #[arg(long)]
        new_commit: Option<String>,

        /// Git commit SHA for the old engine (overrides the value embedded in the old binary)
        #[arg(long)]
        old_commit: Option<String>,

        /// Path to a games JSON file to resume from; reconstructs W/L/D and auto-detects TC and variants
        #[arg(long)]
        resume: Option<String>,

        /// Save the --games file every N completed games when --games is set (0 = every game)
        #[arg(long, default_value_t = 50)]
        save_interval: usize,
    },

    /// Print the commit SHA and date baked into this binary at build time (JSON output).
    /// Used internally by the run manager to identify which snapshot the old binary was built from.
    CommitInfo,

    /// Internal interface for subprocess move generation
    Search {
        /// ICN string of the position
        #[arg(long, required = true)]
        icn: String,

        /// White time remaining in ms
        #[arg(long, default_value_t = 0)]
        wtime: u64,

        /// Black time remaining in ms
        #[arg(long, default_value_t = 0)]
        btime: u64,

        /// White increment in ms
        #[arg(long, default_value_t = 0)]
        winc: u64,

        /// Black increment in ms
        #[arg(long, default_value_t = 0)]
        binc: u64,

        /// Variant name
        #[arg(long, default_value = "Classical")]
        variant: String,

        /// Maximum depth for search
        #[arg(long)]
        max_depth: Option<usize>,

        /// Fixed time for search in ms
        #[arg(long)]
        fixed_time: Option<u32>,

        /// Search noise amplitude
        #[arg(long)]
        noise_amp: Option<i32>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Engine strength Level
        #[arg(long)]
        strength_level: Option<u32>,
    },
}

/// Commit identity: short SHA plus an optional date string (YYYY-MM-DD) and dirty flag.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CommitInfo {
    commit: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    dirty: bool,
}

impl CommitInfo {
    /// Format for display: "abc12345 (2024-01-15)" or "abc12345 (dirty)" or "abc12345 (2024-01-15, dirty)".
    fn display_str(&self) -> String {
        let mut result = self.commit.clone();
        let mut suffix = String::new();

        if !self.date.is_empty() {
            suffix.push_str(&self.date);
        }

        if self.dirty {
            if !suffix.is_empty() {
                suffix.push_str(", dirty");
            } else {
                suffix.push_str("dirty");
            }
        }

        if !suffix.is_empty() {
            result.push_str(&format!(" ({})", suffix));
        }

        result
    }
}

/// Best-effort: get the author-date (YYYY-MM-DD) for the given git revision.
/// Returns an empty string when git is unavailable or the revision is unknown.
fn get_commit_date_from_git(sha: &str) -> String {
    Command::new("git")
        .args(["log", "-1", "--format=%cs", sha])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Query `bin_path commit-info` to get the commit info embedded in that binary at build time.
/// Returns `None` if the binary doesn't support the subcommand or its output can't be parsed.
fn try_query_binary_commit_info(bin_path: &str) -> Option<CommitInfo> {
    let output = Command::new(bin_path)
        .arg("commit-info")
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let json = String::from_utf8_lossy(&output.stdout).trim().to_string();
    serde_json::from_str::<CommitInfo>(&json)
        .ok()
        .filter(|c| !c.commit.is_empty())
}

/// Try to resolve a git commit SHA (short 8-char) and author-date for `rev`.
/// Silently returns `None` if git is unavailable or `rev` cannot be resolved.
fn try_get_commit_info_from_git(rev: &str) -> Option<CommitInfo> {
    let sha_out = Command::new("git")
        .args(["rev-parse", "--short=7", rev])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let commit = String::from_utf8_lossy(&sha_out.stdout).trim().to_string();
    if commit.is_empty() {
        return None;
    }
    let date = get_commit_date_from_git(rev);
    Some(CommitInfo {
        commit,
        date,
        dirty: false,
    })
}

/// Print the "NEW: … vs OLD: …" commit identity line (shared by startup banner and final summary).
fn print_commit_context(new_info: &Option<CommitInfo>, old_info: &Option<CommitInfo>) {
    match (new_info, old_info) {
        (Some(nc), Some(oc)) => {
            println!("  NEW: {}  vs  OLD: {}", nc.display_str(), oc.display_str())
        }
        (Some(nc), None) => println!("  NEW: {}  vs  OLD: (unknown)", nc.display_str()),
        (None, Some(oc)) => println!("  NEW: (unknown)  vs  OLD: {}", oc.display_str()),
        (None, None) => {}
    }
}

/// Print the compact settings lines (shared by startup banner and final summary).
fn print_settings_context(config: &Config) {
    let adjudication_str = if config.adjudication_threshold <= 0 {
        "Off".to_string()
    } else {
        format!("{} cp", config.adjudication_threshold)
    };
    let maxply_adjudication_str = if config.maxply_adjudication <= 0.0 {
        "Off".to_string()
    } else {
        format!("{} cp", config.maxply_adjudication)
    };
    println!(
        "  TC: {} | Concurrency: {} | Variants: {} | Strength: {} vs {} | Adjudication: {} | Max-ply adjudication: {}",
        config.tc,
        config.concurrency,
        config.variants.len(),
        config.new_strength,
        config.old_strength,
        adjudication_str,
        maxply_adjudication_str,
    );
}

#[derive(Clone, Debug)]
struct Config {
    elo0: f64,
    elo1: f64,
    model: SprtModel,
    alpha: f64,
    beta: f64,
    tc: String,
    tc_base_ms: u64,
    tc_inc_ms: u64,
    tc_fixed_ms: Option<u32>,
    tc_max_depth: Option<usize>,
    concurrency: usize,
    max_games: Option<usize>,
    min_games: usize,
    variants: Vec<Variant>,
    adjudication_threshold: i32,
    maxply_adjudication: f64,
    new_bin: String,
    old_bin: String,
    max_moves: usize,
    search_noise: i32,
    new_strength: u32,
    old_strength: u32,
    verbose: bool,
    compact: bool,
    new_commit_info: Option<CommitInfo>,
    old_commit_info: Option<CommitInfo>,
    resume_pair_offset: usize,
    save_interval: usize,
}

static STOP: AtomicBool = AtomicBool::new(false);
static USER_STOP: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GameResult {
    Win,
    Loss,
    Draw,
}

struct GameOutcome {
    result: GameResult,
    icn: String,
    variant_name: String,
    game_idx: usize,
    termination_reason: String,
    new_engine_timed_out: bool,
}

#[derive(Clone, Copy)]
enum TerminalState {
    Checkmate { white_won: bool },
    AllPiecesCaptured { white_won: bool },
    AllRoyalsCaptured { white_won: bool },
    RoyalCapture { white_won: bool },
    Draw(&'static str),
}

fn elo_to_score(elo_diff: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-elo_diff / 400.0))
}
fn score_to_elo(s: f64) -> f64 {
    -400.0 * (1.0 / s.clamp(1e-9, 1.0 - 1e-9) - 1.0).log10()
}

fn estimate_elo(wins: usize, losses: usize, draws: usize) -> (f64, f64) {
    let total = wins + losses + draws;
    if total == 0 {
        return (0.0, 0.0);
    }
    let score = (wins as f64 + draws as f64 * 0.5) / total as f64;
    if score <= 0.0 {
        return (-999.0, 0.0);
    }
    if score >= 1.0 {
        return (999.0, 0.0);
    }
    let elo = -400.0 * (1.0 / score - 1.0).log10();
    let variance = (wins as f64 * (1.0 - score).powi(2)
        + losses as f64 * (0.0 - score).powi(2)
        + draws as f64 * (0.5 - score).powi(2))
        / total as f64;
    let std_dev = (variance / total as f64).sqrt();
    let elo_error = std_dev * 400.0 / (10.0f64.ln() * score * (1.0 - score));
    (elo, elo_error.min(200.0))
}

// Pentanomial GSPRT
#[derive(Clone, Copy, Default, Debug)]
struct PentaCounts {
    ww: usize,
    wd: usize,
    wl: usize,
    dd: usize,
    ld: usize,
    ll: usize,
}

impl PentaCounts {
    fn total_pairs(&self) -> usize {
        self.ww + self.wd + self.wl + self.dd + self.ld + self.ll
    }

    fn score(&self) -> f64 {
        (self.ww as f64
            + 0.75 * self.wd as f64
            + 0.5 * (self.wl as f64 + self.dd as f64)
            + 0.25 * self.ld as f64
        ) / self.total_pairs() as f64
    }

    fn variance(&self) -> f64 {
        let score = self.score();
        (self.ww as f64 * (1.0 - score).powi(2)
            + self.wd as f64 * (0.75 - score).powi(2)
            + (self.wl as f64 + self.dd as f64) * (0.5 - score).powi(2)
            + self.ld as f64 * (0.25 - score).powi(2)
            + self.ll as f64 * (0.0 - score).powi(2)
        ) / self.total_pairs() as f64
    }

    /// Bucket a completed pair from the two NEW-perspective game results.
    fn add_pair(&mut self, a: GameResult, b: GameResult) {
        let (mut w, mut d, mut l) = (0u32, 0u32, 0u32);
        for r in [a, b] {
            match r {
                GameResult::Win => w += 1,
                GameResult::Draw => d += 1,
                GameResult::Loss => l += 1,
            }
        }
        if w == 2 {
            self.ww += 1;
        } else if w == 1 && d == 1 {
            self.wd += 1;
        } else if w == 1 && l == 1 {
            self.wl += 1;
        } else if d == 2 {
            self.dd += 1;
        } else if l == 1 && d == 1 {
            self.ld += 1;
        } else {
            self.ll += 1;
        }
    }
}

/// fastchess `regularize`: a zero bucket becomes 1e-3 so log-likelihoods stay finite.
fn regularize(v: usize) -> f64 {
    if v == 0 { 1e-3 } else { v as f64 }
}

/// ITP root-finder (Oliveira & Takahashi 2020), ported from fastchess `itp()`.
/// Solves f(x)=0 on the bracket [a,b] with f_a<0<f_b (after the initial swap).
fn itp<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    mut f_a: f64,
    mut f_b: f64,
    k_1: f64,
    k_2: f64,
    n_0: f64,
    epsilon: f64,
) -> f64 {
    if f_a > 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut f_a, &mut f_b);
    }

    let n_half = ((b - a).abs() / (2.0 * epsilon)).log2().ceil();
    let n_max = n_half + n_0;
    let mut i = 0.0;
    while (b - a).abs() > 2.0 * epsilon {
        let x_half = (a + b) / 2.0;
        let r = epsilon * 2.0f64.powf(n_max - i) - (b - a) / 2.0;
        let delta = k_1 * (b - a).powf(k_2);

        let x_f = (f_b * a - f_a * b) / (f_b - f_a);

        let sigma = (x_half - x_f) / (x_half - x_f).abs();
        let x_t = if delta <= (x_half - x_f).abs() {
            x_f + sigma * delta
        } else {
            x_half
        };

        let x_itp = if (x_t - x_half).abs() <= r {
            x_t
        } else {
            x_half - sigma * r
        };

        let f_itp = f(x_itp);
        if f_itp == 0.0 {
            a = x_itp;
            b = x_itp;
        } else if f_itp < 0.0 {
            a = x_itp;
            f_a = f_itp;
        } else {
            b = x_itp;
            f_b = f_itp;
        }
        i += 1.0;
    }

    (a + b) / 2.0
}

/// Maximum-likelihood outcome distribution constrained to expected score `s`
/// (fastchess `getLLR_logistic`'s inner `mle`, logistic model).
fn mle_logistic(scores: &[f64], probs: &[f64], s: f64) -> Vec<f64> {
    let n = scores.len();
    let theta_epsilon = 1e-3;
    let min_theta = -1.0 / (scores[n - 1] - s);
    let max_theta = -1.0 / (scores[0] - s);
    let theta = itp(
        |x| {
            let mut result = 0.0;
            for i in 0..n {
                let a_i = scores[i];
                result += probs[i] * (a_i - s) / (1.0 + x * (a_i - s));
            }
            result
        },
        min_theta,
        max_theta,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.1,
        2.0,
        0.99,
        theta_epsilon,
    );
    (0..n)
        .map(|i| probs[i] / (1.0 + theta * (scores[i] - s)))
        .collect()
}

/// Log-likelihood ratio for the logistic GSPRT (fastchess `getLLR_logistic`).
fn llr_logistic(total: f64, scores: &[f64], probs: &[f64], s0: f64, s1: f64) -> f64 {
    let p0 = mle_logistic(scores, probs, s0);
    let p1 = mle_logistic(scores, probs, s1);
    let mut acc = 0.0;
    for i in 0..scores.len() {
        acc += probs[i] * (p1[i].ln() - p0[i].ln());
    }
    total * acc
}

fn mean(x: &[f64], p: &[f64]) -> f64 {
    let mut result = 0.0;
    for i in 0..x.len() {
        result += x[i] * p[i];
    }
    result
}

fn mean_and_variance(x: &[f64], p: &[f64]) -> (f64, f64) {
    let mu = mean(x, p);
    let mut var = 0.0;
    for i in 0..x.len() {
        var += p[i] * (x[i] - mu) * (x[i] - mu);
    }
    (mu, var)
}

/// MLE distribution for the normalized model (fastchess `getLLR_normalized`'s inner
/// `mle`): iteratively re-fits `phi` from the running mean/variance, targeting the
/// standardized effect `t_star` at reference score `mu_ref`.
fn mle_normalized(scores: &[f64], probs: &[f64], mu_ref: f64, t_star: f64) -> Vec<f64> {
    let n = scores.len();
    let theta_epsilon = 1e-7;
    let mle_epsilon = 1e-4;
    let mut p = vec![1.0 / n as f64; n];

    for _ in 0..10 {
        let (mu, var) = mean_and_variance(scores, &p);
        let sigma = var.sqrt();
        let phi: Vec<f64> = (0..n)
            .map(|i| {
                let a_i = scores[i];
                a_i - mu_ref
                    - 0.5 * t_star * sigma * (1.0 + ((a_i - mu) / sigma) * ((a_i - mu) / sigma))
            })
            .collect();

        let u = phi.iter().cloned().fold(f64::INFINITY, f64::min);
        let v = phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_theta = -1.0 / v;
        let max_theta = -1.0 / u;

        let theta = itp(
            |x| {
                let mut result = 0.0;
                for i in 0..n {
                    result += probs[i] * phi[i] / (1.0 + x * phi[i]);
                }
                result
            },
            min_theta,
            max_theta,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.1,
            2.0,
            0.99,
            theta_epsilon,
        );

        let mut max_diff = 0.0f64;
        for i in 0..n {
            let newp = probs[i] / (1.0 + theta * phi[i]);
            max_diff = max_diff.max((newp - p[i]).abs());
            p[i] = newp;
        }
        if max_diff < mle_epsilon {
            break;
        }
    }
    p
}

/// Log-likelihood ratio for the normalized (nElo) GSPRT (fastchess `getLLR_normalized`).
fn llr_normalized(total: f64, scores: &[f64], probs: &[f64], t0: f64, t1: f64) -> f64 {
    let p0 = mle_normalized(scores, probs, 0.5, t0);
    let p1 = mle_normalized(scores, probs, 0.5, t1);
    let mut acc = 0.0;
    for i in 0..scores.len() {
        acc += probs[i] * (p1[i].ln() - p0[i].ln());
    }
    total * acc
}

/// SPRT statistical model for interpreting the elo0/elo1 bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SprtModel {
    /// Normalized Elo (nElo): bounds are signal-to-noise, draw-rate/TC independent.
    Normalized,
    /// Logistic (classic) Elo: bounds are raw Elo points.
    Logistic,
}

impl SprtModel {
    fn parse(s: &str) -> SprtModel {
        match s.trim().to_lowercase().as_str() {
            "logistic" => SprtModel::Logistic,
            _ => SprtModel::Normalized,
        }
    }
}

/// Pentanomial LLR, the SPRT decision statistic, matching fastchess and Fishtest.
fn calculate_pentanomial_llr(p: &PentaCounts, elo0: f64, elo1: f64, model: SprtModel) -> f64 {
    if p.total_pairs() == 0 {
        return 0.0;
    }
    let ll = regularize(p.ll);
    let ld = regularize(p.ld);
    let wl_dd = regularize(p.dd + p.wl);
    let wd = regularize(p.wd);
    let ww = regularize(p.ww);
    let total = ww + wd + wl_dd + ld + ll;
    let probs = [
        ll / total,
        ld / total,
        wl_dd / total,
        wd / total,
        ww / total,
    ];
    let scores = [0.0, 0.25, 0.5, 0.75, 1.0];
    match model {
        SprtModel::Normalized => {
            // Pentanomial scale factor sqrt(2) (fastchess), 800/ln10 = logistic constant.
            let t0 = 2.0f64.sqrt() * elo0 / (800.0 / 10.0f64.ln());
            let t1 = 2.0f64.sqrt() * elo1 / (800.0 / 10.0f64.ln());
            llr_normalized(total, &scores, &probs, t0, t1)
        }
        SprtModel::Logistic => {
            let s0 = elo_to_score(elo0);
            let s1 = elo_to_score(elo1);
            llr_logistic(total, &scores, &probs, s0, s1)
        }
    }
}

/// Likelihood of superiority: probability that the new engine is better than the
/// old. Matches fastchess implementation.
fn calculate_los(score: f64, variance_per_pair: f64) -> f64 {
    (1.0 - (-(score - 0.5) / (2.0 * variance_per_pair).sqrt()).erf()) / 2.0
}

/// Pentanomial Elo estimate, matching fastchess `EloPentanomial`. Both the
/// logistic Elo (`elo`) and normalized Elo (`nelo`) point-estimates are computed
/// from the same pair-score distribution — they do not depend on the SPRT model.
#[derive(Clone, Copy, Debug, Default)]
struct PentaElo {
    elo: f64,
    elo_err: f64,
    nelo: f64,
    nelo_err: f64,
}

fn estimate_pentanomial_elo(p: &PentaCounts) -> PentaElo {
    let pairs = p.total_pairs() as f64;
    if pairs == 0.0 {
        return PentaElo::default();
    }

    let score = p.score();
    let variance = p.variance();
    let variance_per_pair = variance / pairs;

    const CI95: f64 = 1.959963984540054;
    // Normalized Elo (fastchess scoreToNeloDiff): uses the per-pair variance.
    let s2n = |s: f64| (s - 0.5) / (2.0 * variance).sqrt() * (800.0 / 10.0f64.ln());
    let upper = score + CI95 * variance_per_pair.sqrt();
    let lower = score - CI95 * variance_per_pair.sqrt();
    let elo = if score <= 0.0 {
        -999.0
    } else if score >= 1.0 {
        999.0
    } else {
        score_to_elo(score)
    };
    let elo_err = (score_to_elo(upper) - score_to_elo(lower)) / 2.0;
    let (nelo, nelo_err) = if variance <= 0.0 {
        (0.0, 0.0)
    } else {
        (s2n(score), (s2n(upper) - s2n(lower)) / 2.0)
    };
    PentaElo {
        elo,
        elo_err: elo_err.min(200.0),
        nelo,
        nelo_err,
    }
}

fn format_clock(ms: u64) -> String {
    let total_seconds = ms / 1000;
    let h = total_seconds / 3600;
    let m = (total_seconds % 3600) / 60;
    let s = total_seconds % 60;
    let dec = (ms % 1000) / 100;
    format!("{}:{:02}:{:02}.{}", h, m, s, dec)
}

fn move_to_string(m: &apeiron::moves::Move) -> String {
    let mut s = format!("{},{} {},{}", m.from.x, m.from.y, m.to.x, m.to.y);
    if let Some(p) = m.promotion {
        s.push_str(&format!(" {}", p.to_site_code().to_lowercase()));
    }
    s
}

fn parse_icn_tag(icn: &str, tag: &str) -> Option<String> {
    let prefix = format!("[{} \"", tag);
    let start = icn.find(&prefix)? + prefix.len();
    let end = start + icn[start..].find('"')?;
    Some(icn[start..end].to_string())
}

struct ResumeState {
    games: Vec<String>,
    wins: usize,
    losses: usize,
    draws: usize,
    penta: PentaCounts,
    per_variant_stats: HashMap<String, (usize, usize, usize)>,
    resume_pair_offset: usize,
    detected_tc: Option<String>,
    detected_variants: Vec<String>,
}

fn load_resume_state(path: &str) -> ResumeState {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read resume file '{}': {}", path, e));
    let games: Vec<String> = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse resume file '{}': {}", path, e));

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut per_variant_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut max_game_idx: Option<usize> = None;
    let mut detected_tc: Option<String> = None;
    let mut seen_variants: HashSet<String> = HashSet::new();
    let mut detected_variants: Vec<String> = Vec::new();
    // Per-game NEW-perspective result keyed by game index, for pentanomial pairing.
    let mut results_by_idx: HashMap<usize, GameResult> = HashMap::new();

    for icn in &games {
        let mut this_idx: Option<usize> = None;
        if let Some(event) = parse_icn_tag(icn, "Event")
            && let Some(idx_str) = event.strip_prefix("SPRT Test Game ")
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            max_game_idx = Some(max_game_idx.map_or(idx, |m| m.max(idx)));
            this_idx = Some(idx);
        }

        let result_tag = parse_icn_tag(icn, "Result");
        let white_player = parse_icn_tag(icn, "White");
        let new_plays_white = white_player.as_deref() == Some("Apeiron New");

        let result = match result_tag.as_deref() {
            Some("1-0") => {
                if new_plays_white {
                    GameResult::Win
                } else {
                    GameResult::Loss
                }
            }
            Some("0-1") => {
                if new_plays_white {
                    GameResult::Loss
                } else {
                    GameResult::Win
                }
            }
            _ => GameResult::Draw,
        };

        match result {
            GameResult::Win => wins += 1,
            GameResult::Loss => losses += 1,
            GameResult::Draw => draws += 1,
        }

        if let Some(idx) = this_idx {
            results_by_idx.insert(idx, result);
        }

        if let Some(variant) = parse_icn_tag(icn, "Variant") {
            if seen_variants.insert(variant.clone()) {
                detected_variants.push(variant.clone());
            }
            let stats = per_variant_stats.entry(variant).or_insert((0, 0, 0));
            match result {
                GameResult::Win => stats.0 += 1,
                GameResult::Loss => stats.1 += 1,
                GameResult::Draw => stats.2 += 1,
            }
        }

        if detected_tc.is_none() {
            detected_tc = parse_icn_tag(icn, "TimeControl");
        }
    }

    let resume_pair_offset = max_game_idx.map_or(0, |idx| idx / 2 + 1);

    // Rebuild pentanomial buckets from complete pairs (2k, 2k+1); drop orphans.
    let mut penta = PentaCounts::default();
    if let Some(max_idx) = max_game_idx {
        for k in 0..=(max_idx / 2) {
            if let (Some(&a), Some(&b)) = (
                results_by_idx.get(&(2 * k)),
                results_by_idx.get(&(2 * k + 1)),
            ) {
                penta.add_pair(a, b);
            }
        }
    }

    ResumeState {
        games,
        wins,
        losses,
        draws,
        penta,
        per_variant_stats,
        resume_pair_offset,
        detected_tc,
        detected_variants,
    }
}

fn save_games_file(path: &str, game_logs: &[String]) {
    let tmp_path = format!("{}.tmp", path);
    if let Ok(json_data) = serde_json::to_string_pretty(game_logs)
        && std::fs::write(&tmp_path, &json_data).is_ok()
    {
        let _ = std::fs::rename(&tmp_path, path);
    }
}

/// Parse a bestmove output line like "8,8 6,8" or "1,7 1,8 q" into
/// an ICN move string like "8,8>6,8" or "1,7>1,8=Q" (case-sensitive by turn).
fn parse_bestmove_to_icn(bestmove_str: &str, turn: PlayerColor) -> Option<String> {
    let parts: Vec<&str> = bestmove_str.split_whitespace().collect();
    if parts.len() < 2 || parts[0] == "none" {
        return None;
    }
    let from = parts[0]; // "fx,fy"
    let to = parts[1]; // "tx,ty"
    // Validate coordinate format
    let from_parts: Vec<&str> = from.split(',').collect();
    let to_parts: Vec<&str> = to.split(',').collect();
    if from_parts.len() != 2 || to_parts.len() != 2 {
        return None;
    }
    from_parts[0].parse::<i64>().ok()?;
    from_parts[1].parse::<i64>().ok()?;
    to_parts[0].parse::<i64>().ok()?;
    to_parts[1].parse::<i64>().ok()?;

    let mut result = format!("{}>{}", from, to);
    if parts.len() > 2 {
        // ICN uses uppercase for White pieces, lowercase for Black
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

fn has_any_fully_legal_move(game: &GameState) -> bool {
    let moves = game.get_legal_moves();
    for m in moves {
        let mut game_copy = game.clone();
        game_copy.make_move(&m);
        let legal = !game_copy.is_move_illegal();
        if legal {
            return true;
        }
    }
    false
}

fn make_position_key(game: &GameState) -> String {
    // Build piece list sorted by position
    let mut pieces: Vec<String> = game
        .board
        .iter()
        .map(|(x, y, piece)| {
            let color_char = if piece.color() == PlayerColor::White {
                'w'
            } else {
                'b'
            };
            let piece_char = piece.piece_type().to_site_code().to_lowercase();
            format!("{}{}{},{}", color_char, piece_char, x, y)
        })
        .collect();
    pieces.sort();

    // Compute effective castling rights following FIDE rules
    let mut castling_rights = String::new();

    for color in [PlayerColor::White, PlayerColor::Black] {
        let color_char = if color == PlayerColor::White {
            'w'
        } else {
            'b'
        };

        // Find king
        let king_pos = game
            .board
            .iter()
            .find(|(_, _, piece)| piece.color() == color && piece.piece_type() == PieceType::King);

        if let Some((king_x, king_y, _)) = king_pos {
            let king_coord = Coordinate::new(king_x, king_y);
            let king_has_rights = game.has_special_right(&king_coord);

            if king_has_rights {
                // King has rights - check which castling partners have rights
                let mut left_partner = false;
                let mut right_partner = false;

                for (px, py, piece) in game.board.iter() {
                    if piece.color() != color {
                        continue;
                    }
                    if piece.piece_type() == PieceType::Pawn
                        || piece.piece_type() == PieceType::King
                    {
                        continue;
                    }

                    if py != king_y {
                        continue;
                    }

                    let partner_coord = Coordinate::new(px, py);
                    if game.has_special_right(&partner_coord) {
                        if px < king_x {
                            left_partner = true;
                        } else {
                            right_partner = true;
                        }
                    }
                }

                if left_partner {
                    castling_rights.push_str(&format!("{}L", color_char));
                }
                if right_partner {
                    castling_rights.push_str(&format!("{}R", color_char));
                }
            }
        }
    }

    // Compute pawn special rights (double-push rights)
    let mut pawn_rights = String::new();
    let mut pawn_coords: Vec<String> = game
        .board
        .iter()
        .filter_map(|(x, y, piece)| {
            if piece.piece_type() == PieceType::Pawn {
                let coord = Coordinate::new(x, y);
                if game.has_special_right(&coord) {
                    return Some(format!("{},{}", x, y));
                }
            }
            None
        })
        .collect();
    pawn_coords.sort();
    if !pawn_coords.is_empty() {
        pawn_rights = pawn_coords.join(";");
    }

    // Include en passant square if present
    let ep = if let Some(ep_info) = game.en_passant {
        format!("{},{}", ep_info.square.x, ep_info.square.y)
    } else {
        String::new()
    };

    // Combine all components
    let turn_char = if game.turn == PlayerColor::White {
        'w'
    } else {
        'b'
    };
    format!(
        "{}|{}|{}|{}|{}",
        turn_char,
        pieces.join(";"),
        castling_rights,
        pawn_rights,
        ep
    )
}

fn detect_terminal_state(game: &GameState) -> Option<TerminalState> {
    use apeiron::game::WinCondition;

    // Determine what the opponent must do to beat the current player.
    let opp_wc = match game.turn {
        PlayerColor::White => game.game_rules.black_win_condition,
        PlayerColor::Black => game.game_rules.white_win_condition,
        PlayerColor::Neutral => return None,
    };

    // AllPiecesCaptured must be checked before has_lost_by_royal_capture: taking the
    // last piece also zeroes royals, which sends that function down a branch that
    // skips termination and the fifty-move check entirely.
    if opp_wc == WinCondition::AllPiecesCaptured {
        let has_royals = match game.turn {
            PlayerColor::White => !game.white_royals.is_empty(),
            PlayerColor::Black => !game.black_royals.is_empty(),
            _ => false,
        };
        if !game.has_pieces(game.turn) && !has_royals {
            return Some(TerminalState::AllPiecesCaptured {
                white_won: game.turn == PlayerColor::Black,
            });
        }
        // Not yet over (opponent still has pieces/royals left to capture).
        // Fall through so the fifty-move rule and other draw checks still run.

        // RoyalCapture / AllRoyalsCaptured
        // Only run for win conditions that actually target royals.
    } else if game.has_lost_by_royal_capture() {
        return match opp_wc {
            WinCondition::RoyalCapture => Some(TerminalState::RoyalCapture {
                white_won: game.turn == PlayerColor::Black,
            }),
            WinCondition::AllRoyalsCaptured => Some(TerminalState::AllRoyalsCaptured {
                white_won: game.turn == PlayerColor::Black,
            }),
            // Checkmate: losing royals is not a direct win condition here;
            // fall through to legal-move detection below.
            _ => return None,
        };
    }

    // No legal moves → checkmate / stalemate / piece-loss
    let in_check = game.is_in_check();
    let has_legal_move = has_any_fully_legal_move(game);
    if !has_legal_move {
        let lost_by_mate = in_check && game.must_escape_check();
        let lost_by_piece_capture = !game.has_pieces(game.turn);

        if lost_by_mate {
            return Some(TerminalState::Checkmate {
                white_won: game.turn == PlayerColor::Black,
            });
        }

        if lost_by_piece_capture {
            return Some(TerminalState::AllPiecesCaptured {
                white_won: game.turn == PlayerColor::Black,
            });
        }

        return Some(TerminalState::Draw("stalemate"));
    }

    // Draw conditions
    if apeiron::evaluation::insufficient_material::evaluate_insufficient_material_game_handler(game)
    {
        return Some(TerminalState::Draw("insufficient_material"));
    }

    if game.is_fifty() {
        return Some(TerminalState::Draw("fifty-move rule"));
    }

    None
}

fn with_variant_bounds<T>(variant: Variant, f: impl FnOnce() -> T) -> T {
    static WORLD_BOUNDS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = WORLD_BOUNDS_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("world bounds lock poisoned");
    let bounds = variant.get_default_bounds();
    apeiron::moves::set_world_bounds(bounds.0, bounds.1, bounds.2, bounds.3);
    f()
}

fn play_game(
    config: &Config,
    variant: Variant,
    new_plays_white: bool,
    game_idx: usize,
    seeds: Vec<u64>,
) -> GameOutcome {
    let mut game = with_variant_bounds(variant, || {
        let mut game = GameState::new();
        game.setup_position_from_icn(variant.starting_icn());
        game.variant = Some(variant);
        game
    });

    let starting_board_setup = get_board_setup_icn(&game);

    let mut white_clock = config.tc_base_ms;
    let mut black_clock = config.tc_base_ms;
    let mut move_info_log = Vec::new();
    let mut move_history_clean: Vec<String> = Vec::new();
    let mut repetition_counts: HashMap<String, usize> = HashMap::new();
    let mut last_eval_new: Option<i32> = None;
    let mut last_eval_old: Option<i32> = None;
    // Each engine's last search score in White-ahead centipawns (positive =
    // White winning), for the max-ply adjudication when both engines agree.
    let mut last_wscore_new: Option<f64> = None;
    let mut last_wscore_old: Option<f64> = None;
    let termination_reason;

    let get_eval = |g: &GameState| {
        #[cfg(feature = "nnue")]
        return with_variant_bounds(variant, || apeiron::evaluation::evaluate(g, None));
        #[cfg(not(feature = "nnue"))]
        return with_variant_bounds(variant, || apeiron::evaluation::evaluate(g));
    };

    /// Helper to create an outcome return value
    macro_rules! game_outcome {
        ($result:expr, $reason:expr, $result_str:expr) => {
            GameOutcome {
                result: $result,
                icn: generate_icn(
                    &variant,
                    &move_info_log,
                    game_idx,
                    new_plays_white,
                    Some($reason),
                    config,
                    $result_str,
                    &starting_board_setup,
                ),
                variant_name: variant.to_str().to_string(),
                game_idx,
                termination_reason: $reason.to_string(),
                new_engine_timed_out: false,
            }
        };
    }

    // Record initial position
    {
        let key = make_position_key(&game);
        *repetition_counts.entry(key).or_insert(0) += 1;
    }

    for (ply, &seed_val) in seeds.iter().enumerate().take(config.max_moves) {
        if STOP.load(Ordering::SeqCst) {
            if USER_STOP.load(Ordering::SeqCst) {
                return GameOutcome {
                    result: GameResult::Draw, // Dummy result
                    icn: String::new(),
                    variant_name: variant.to_str().to_string(),
                    game_idx,
                    termination_reason: "interrupted".to_string(),
                    new_engine_timed_out: false,
                };
            }
            break;
        }

        // Check for threefold repetition using manual position key tracking
        let current_key = make_position_key(&game);
        let repetition_count = *repetition_counts.get(&current_key).unwrap_or(&0);
        if repetition_count >= 3 {
            return game_outcome!(GameResult::Draw, "threefold repetition", "1/2-1/2");
        }

        // Terminal state checks always run before adjudication or engine search.
        if let Some(terminal) = with_variant_bounds(variant, || detect_terminal_state(&game)) {
            match terminal {
                TerminalState::Checkmate { white_won } => {
                    let result = if white_won == new_plays_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    };
                    return game_outcome!(
                        result,
                        "checkmate",
                        if white_won { "1-0" } else { "0-1" }
                    );
                }
                TerminalState::AllPiecesCaptured { white_won } => {
                    let result = if white_won == new_plays_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    };
                    return game_outcome!(
                        result,
                        "allpiecescaptured",
                        if white_won { "1-0" } else { "0-1" }
                    );
                }
                TerminalState::AllRoyalsCaptured { white_won } => {
                    let result = if white_won == new_plays_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    };
                    return game_outcome!(
                        result,
                        "allroyalscaptured",
                        if white_won { "1-0" } else { "0-1" }
                    );
                }
                TerminalState::RoyalCapture { white_won } => {
                    let result = if white_won == new_plays_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    };
                    return game_outcome!(
                        result,
                        "royalcapture",
                        if white_won { "1-0" } else { "0-1" }
                    );
                }
                TerminalState::Draw(reason) => {
                    return game_outcome!(GameResult::Draw, reason, "1/2-1/2");
                }
            }
        }

        // Material adjudication (after terminal checks, only if both engines agree)
        // Requires at least 20 plies and both engines to have provided evals
        if variant != Variant::PawnHorde
            && move_history_clean.len() >= 20
            && let (Some(eval_new), Some(eval_old)) = (last_eval_new, last_eval_old)
        {
            let threshold = config.adjudication_threshold;
            if threshold > 0 {
                // Determine winner from each engine's eval
                let new_winner = if eval_new >= threshold {
                    Some('w')
                } else if eval_new <= -threshold {
                    Some('b')
                } else {
                    None
                };

                let old_winner = if eval_old >= threshold {
                    Some('w')
                } else if eval_old <= -threshold {
                    Some('b')
                } else {
                    None
                };

                // Only adjudicate if both engines agree on the same winner
                if let (Some(new_w), Some(old_w)) = (new_winner, old_winner)
                    && new_w == old_w
                {
                    let white_winning = new_w == 'w';
                    let result = if white_winning == new_plays_white {
                        GameResult::Win
                    } else {
                        GameResult::Loss
                    };
                    let result_str = if white_winning { "1-0" } else { "0-1" };
                    return game_outcome!(result, "material adjudication", result_str);
                }
            }
        }

        // Engine search
        let is_new_turn = (game.turn == PlayerColor::White) == new_plays_white;

        let bin = if is_new_turn {
            &config.new_bin
        } else {
            &config.old_bin
        };

        let subprocess_icn = if move_history_clean.is_empty() {
            starting_board_setup.clone()
        } else {
            format!("{} {}", starting_board_setup, move_history_clean.join("|"))
        };

        let mut cmd = Command::new(bin);
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
            .arg(variant.to_str());

        if let Some(d) = config.tc_max_depth {
            cmd.arg("--max-depth").arg(d.to_string());
        }
        if let Some(ft) = config.tc_fixed_ms {
            cmd.arg("--fixed-time").arg(ft.to_string());
        }

        if ply < 8 {
            cmd.arg("--noise-amp").arg(config.search_noise.to_string());
        }

        cmd.arg("--seed").arg(seed_val.to_string());

        let strength = if is_new_turn {
            config.new_strength
        } else {
            config.old_strength
        };
        if strength < apeiron::search::MAX_SITE_SKILL {
            cmd.arg("--strength-level").arg(strength.to_string());
        }

        if config.verbose {
            cmd.stderr(Stdio::inherit());
        }

        let start_time = Instant::now();
        let output = cmd
            .output()
            .unwrap_or_else(|e| panic!("Failed to execute engine binary {}: {}", bin, e));
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // A crashed engine invalidates every remaining game, so stop the run and
        // keep its stderr — that's where a panic in the engine binary lands.
        // Once STOP is set the run is already winding down and in-flight engines
        // are being torn down (Ctrl-C kills the whole process group), so a
        // non-zero exit there is expected noise rather than a fault.
        if !(output.status.success() || STOP.load(Ordering::SeqCst)) {
            abort_run(AbortReason::EngineFault {
                kind: "engine crashed",
                engine: if is_new_turn { "NEW" } else { "OLD" },
                game_idx,
                variant: variant.to_str().to_string(),
                detail: format!(
                    "exit code {:?}\n\nengine stderr:\n{}",
                    output.status.code(),
                    if stderr.trim().is_empty() {
                        "(empty — rerun without --verbose to capture it)"
                    } else {
                        stderr.trim()
                    }
                ),
            });
        }

        // Parse bestmove into ICN move format
        let bestmove_icn = if let Some(line) = stdout.lines().find(|l| l.starts_with("bestmove")) {
            let move_str = line.trim_start_matches("bestmove").trim();
            parse_bestmove_to_icn(move_str, game.turn)
        } else {
            None
        };

        // Parse score/depth from stderr
        let mut score = None;
        if !config.verbose
            && let Some(line) = stderr.lines().find(|l| l.contains("score"))
        {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for i in 0..parts.len() {
                if parts[i] == "score" && i + 1 < parts.len() {
                    score = parts[i + 1].parse::<f64>().ok();
                }
            }
        }

        let elapsed = (start_time.elapsed().as_millis() as u64).saturating_sub(20);

        let current_clock = if game.turn == PlayerColor::White {
            white_clock
        } else {
            black_clock
        };
        let (flagged_on_time, remaining_clock) = account_move_time(
            current_clock,
            elapsed,
            config.tc_inc_ms,
            config.tc_fixed_ms.is_some(),
        );

        if flagged_on_time {
            let result = if is_new_turn {
                GameResult::Loss
            } else {
                GameResult::Win
            };
            let white_won = (result == GameResult::Win) == new_plays_white;
            let result_str = if white_won { "1-0" } else { "0-1" };
            return GameOutcome {
                result,
                icn: generate_icn(
                    &variant,
                    &move_info_log,
                    game_idx,
                    new_plays_white,
                    Some("timeout"),
                    config,
                    result_str,
                    &starting_board_setup,
                ),
                variant_name: variant.to_str().to_string(),
                game_idx,
                termination_reason: "timeout".to_string(),
                new_engine_timed_out: is_new_turn,
            };
        }

        if let Some(move_icn) = bestmove_icn {
            // Build annotated move for the output log
            let mut comment = format!("[%clk {}]", format_clock(remaining_clock));
            if let Some(mut s) = score {
                // Flip score to White's perspective if Black just moved
                if game.turn == PlayerColor::White {
                    s = -s;
                }
                // `s` here is Black-ahead; -s is White-ahead. Record per engine.
                if is_new_turn {
                    last_wscore_new = Some(-s);
                } else {
                    last_wscore_old = Some(-s);
                }
                // Convert mate scores (>= 800000 cp) to [%mate N] format
                if s.abs() >= 800000.0 {
                    let mate_in = if s > 0.0 {
                        ((900000.0 - s + 1.0) / 2.0).floor() as i32
                    } else {
                        ((900000.0 + s + 1.0) / 2.0).floor() as i32
                    };
                    if s > 0.0 {
                        comment.push_str(&format!(" [%mate {}]", mate_in));
                    } else {
                        comment.push_str(&format!(" [%mate -{}]", mate_in));
                    }
                } else {
                    comment.push_str(&format!(" [%eval {:+.2}]", s / 100.0));
                }
            }
            move_info_log.push(format!("{}{{{}}}", move_icn, comment));
            move_history_clean.push(move_icn);

            // Reconstruct game state from the full ICN (starting position + all moves).
            let new_icn = format!("{} {}", starting_board_setup, move_history_clean.join("|"));
            let old_turn = game.turn;
            game = with_variant_bounds(variant, || {
                let mut game = GameState::new();
                game.setup_position_from_icn(&new_icn);
                game.variant = Some(variant);
                game
            });

            // If the turn didn't change, the move wasn't applied (illegal or unparseable)
            if game.turn == old_turn {
                abort_run(AbortReason::EngineFault {
                    kind: "illegal move",
                    engine: if is_new_turn { "NEW" } else { "OLD" },
                    game_idx,
                    variant: variant.to_str().to_string(),
                    detail: format!(
                        "move {} was rejected by the rules\n\nposition:\n{}",
                        move_history_clean.last().map_or("(none)", |m| m.as_str()),
                        new_icn
                    ),
                });
                let result = if is_new_turn {
                    GameResult::Loss
                } else {
                    GameResult::Win
                };
                let white_won = (result == GameResult::Win) == new_plays_white;
                let result_str = if white_won { "1-0" } else { "0-1" };
                return game_outcome!(result, "illegal move", result_str);
            }

            // Record the new position for threefold repetition tracking
            {
                let key = make_position_key(&game);
                *repetition_counts.entry(key).or_insert(0) += 1;
            }

            // Update evaluation tracking for the engine that just moved
            if variant != Variant::PawnHorde {
                let eval = get_eval(&game);
                if is_new_turn {
                    last_eval_new = Some(eval);
                } else {
                    last_eval_old = Some(eval);
                }
            }

            // Update clocks (after the move, it's now the other side's turn)
            if game.turn == PlayerColor::Black {
                // White just moved
                white_clock = remaining_clock;
            } else {
                // Black just moved
                black_clock = remaining_clock;
            }
        } else {
            if let Some(terminal) = with_variant_bounds(variant, || detect_terminal_state(&game)) {
                match terminal {
                    TerminalState::Checkmate { white_won } => {
                        let result = if white_won == new_plays_white {
                            GameResult::Win
                        } else {
                            GameResult::Loss
                        };
                        return game_outcome!(
                            result,
                            "checkmate",
                            if white_won { "1-0" } else { "0-1" }
                        );
                    }
                    TerminalState::AllPiecesCaptured { white_won } => {
                        let result = if white_won == new_plays_white {
                            GameResult::Win
                        } else {
                            GameResult::Loss
                        };
                        return game_outcome!(
                            result,
                            "allpiecescaptured",
                            if white_won { "1-0" } else { "0-1" }
                        );
                    }
                    TerminalState::AllRoyalsCaptured { white_won } => {
                        let result = if white_won == new_plays_white {
                            GameResult::Win
                        } else {
                            GameResult::Loss
                        };
                        return game_outcome!(
                            result,
                            "allroyalscaptured",
                            if white_won { "1-0" } else { "0-1" }
                        );
                    }
                    TerminalState::RoyalCapture { white_won } => {
                        let result = if white_won == new_plays_white {
                            GameResult::Win
                        } else {
                            GameResult::Loss
                        };
                        return game_outcome!(
                            result,
                            "royalcapture",
                            if white_won { "1-0" } else { "0-1" }
                        );
                    }
                    TerminalState::Draw(reason) => {
                        return game_outcome!(GameResult::Draw, reason, "1/2-1/2");
                    }
                }
            }

            if USER_STOP.load(Ordering::SeqCst) {
                return GameOutcome {
                    result: GameResult::Draw, // Dummy result
                    icn: String::new(),
                    variant_name: variant.to_str().to_string(),
                    game_idx,
                    termination_reason: "interrupted".to_string(),
                    new_engine_timed_out: false,
                };
            }

            termination_reason = Some("engine failure");
            abort_run(AbortReason::EngineFault {
                kind: "engine failure",
                engine: if is_new_turn { "NEW" } else { "OLD" },
                game_idx,
                variant: variant.to_str().to_string(),
                detail: "no usable bestmove returned in a non-terminal position \
                         (the engine exited cleanly without moving, or its output was unparseable)"
                    .to_string(),
            });
            let result = if is_new_turn {
                GameResult::Loss
            } else {
                GameResult::Win
            };
            let white_won = (result == GameResult::Win) == new_plays_white;
            return GameOutcome {
                result,
                icn: generate_icn(
                    &variant,
                    &move_info_log,
                    game_idx,
                    new_plays_white,
                    termination_reason,
                    config,
                    if white_won { "1-0" } else { "0-1" },
                    &starting_board_setup,
                ),
                variant_name: variant.to_str().to_string(),
                game_idx,
                termination_reason: termination_reason.unwrap_or("engine failure").to_string(),
                new_engine_timed_out: false,
            };
        }
    }

    // Final check: all terminal conditions before declaring max_moves draw
    if let Some(terminal) = with_variant_bounds(variant, || detect_terminal_state(&game)) {
        match terminal {
            TerminalState::Checkmate { white_won } => {
                let result = if white_won == new_plays_white {
                    GameResult::Win
                } else {
                    GameResult::Loss
                };
                return game_outcome!(result, "checkmate", if white_won { "1-0" } else { "0-1" });
            }
            TerminalState::AllPiecesCaptured { white_won } => {
                let result = if white_won == new_plays_white {
                    GameResult::Win
                } else {
                    GameResult::Loss
                };
                return game_outcome!(
                    result,
                    "allpiecescaptured",
                    if white_won { "1-0" } else { "0-1" }
                );
            }
            TerminalState::AllRoyalsCaptured { white_won } => {
                let result = if white_won == new_plays_white {
                    GameResult::Win
                } else {
                    GameResult::Loss
                };
                return game_outcome!(
                    result,
                    "allroyalscaptured",
                    if white_won { "1-0" } else { "0-1" }
                );
            }
            TerminalState::RoyalCapture { white_won } => {
                let result = if white_won == new_plays_white {
                    GameResult::Win
                } else {
                    GameResult::Loss
                };
                return game_outcome!(
                    result,
                    "royalcapture",
                    if white_won { "1-0" } else { "0-1" }
                );
            }
            TerminalState::Draw(reason) => {
                return game_outcome!(GameResult::Draw, reason, "1/2-1/2");
            }
        }
    }

    // Check for threefold repetition at end of loop
    let final_key = make_position_key(&game);
    let final_repetition_count = *repetition_counts.get(&final_key).unwrap_or(&0);
    if final_repetition_count >= 3 {
        return game_outcome!(GameResult::Draw, "threefold repetition", "1/2-1/2");
    }

    // Max-ply adjudication: if both engines' last score agrees one side is ahead,
    // award that side the point instead of scoring a draw.
    if config.maxply_adjudication > 0.0
        && let (Some(wn), Some(wo)) = (last_wscore_new, last_wscore_old)
    {
        let side = |w: f64| {
            if w >= config.maxply_adjudication {
                Some(true)
            } else if w <= -config.maxply_adjudication {
                Some(false)
            } else {
                None
            }
        };
        if let (Some(sn), Some(so)) = (side(wn), side(wo))
            && sn == so
        {
            let white_won = sn;
            let result = if white_won == new_plays_white {
                GameResult::Win
            } else {
                GameResult::Loss
            };
            return game_outcome!(
                result,
                "max-ply adjudication",
                if white_won { "1-0" } else { "0-1" }
            );
        }
    }

    game_outcome!(GameResult::Draw, "max_moves", "1/2-1/2")
}

fn get_board_setup_icn(game: &GameState) -> String {
    let turn_str = "w";
    let move_limit = game.game_rules.move_rule_limit.unwrap_or(100);
    let promo_token = {
        let white_rank = game.white_promo_rank;
        let black_rank = game.black_promo_rank;
        let promos = if let Some(p_types) = &game.game_rules.promotion_types {
            p_types
                .iter()
                .map(|pt| pt.to_site_code().to_lowercase())
                .collect::<Vec<_>>()
                .join(",")
        } else {
            "q,r,b,n".to_string()
        };
        format!("({};{}|{};{})", white_rank, promos, black_rank, promos)
    };
    let bounds_token = if let Some(v) = &game.variant {
        let bounds = v.get_default_bounds();
        format!("{},{},{},{}", bounds.0, bounds.1, bounds.2, bounds.3)
    } else {
        "-999999999999999,1000000000000008,-999999999999999,1000000000000008".to_string()
    };

    let mut pieces: Vec<_> = game.board.iter().collect();
    // Sort by Y descending, then X ascending
    pieces.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let pieces_str = pieces
        .iter()
        .map(|(x, y, piece)| {
            let mut s = piece.piece_type().to_site_code().to_string();
            if piece.color() == PlayerColor::Black || piece.color() == PlayerColor::Neutral {
                s = s.to_lowercase();
            }
            let mut y_str = y.to_string();
            if game.has_special_right(&Coordinate::new(*x, *y)) {
                y_str.push('+');
            }
            format!("{}{},{}", s, x, y_str)
        })
        .collect::<Vec<_>>()
        .join("|");

    let variant_tag = if let Some(v) = &game.variant {
        format!("[Variant \"{}\"] ", v.to_str())
    } else {
        String::new()
    };

    // Include win conditions if they differ from standard checkmate
    let win_cond_token = if game.game_rules.white_win_condition
        != apeiron::game::WinCondition::Checkmate
        || game.game_rules.black_win_condition != apeiron::game::WinCondition::Checkmate
    {
        format!(
            "{:?},{:?}",
            game.game_rules.white_win_condition, game.game_rules.black_win_condition
        )
        .to_lowercase()
    } else {
        String::new()
    };

    if win_cond_token.is_empty() {
        format!(
            "{}{} 0/{} 1 {} {} {}",
            variant_tag, turn_str, move_limit, promo_token, bounds_token, pieces_str
        )
    } else {
        format!(
            "{}{} 0/{} 1 {} {} {} {}",
            variant_tag,
            turn_str,
            move_limit,
            promo_token,
            bounds_token,
            win_cond_token,
            pieces_str
        )
    }
}

fn generate_icn(
    variant: &Variant,
    move_log: &[String],
    game_idx: usize,
    new_plays_white: bool,
    reason: Option<&str>,
    config: &Config,
    result_str: &str,
    starting_board_setup: &str,
) -> String {
    let mut icn = String::new();
    icn.push_str(&format!("[Event \"SPRT Test Game {}\"] ", game_idx));
    icn.push_str(&format!("[Variant \"{}\"] ", variant.to_str()));
    icn.push_str(&format!("[Result \"{}\"] ", result_str));
    icn.push_str(&format!("[TimeControl \"{}\"] ", config.tc));

    let white = if new_plays_white {
        "Apeiron New"
    } else {
        "Apeiron Old"
    };
    let black = if new_plays_white {
        "Apeiron Old"
    } else {
        "Apeiron New"
    };
    icn.push_str(&format!("[White \"{}\"] ", white));
    icn.push_str(&format!("[Black \"{}\"] ", black));
    let white_strength = if new_plays_white {
        config.new_strength
    } else {
        config.old_strength
    };
    let black_strength = if new_plays_white {
        config.old_strength
    } else {
        config.new_strength
    };
    icn.push_str(&format!("[WhiteStrength \"{}\"] ", white_strength));
    icn.push_str(&format!("[BlackStrength \"{}\"] ", black_strength));

    if let Some(r) = reason {
        let term = match r {
            "material adjudication" => {
                format!(
                    "Material adjudication (|eval| >= {} cp)",
                    config.adjudication_threshold
                )
            }
            "max-ply adjudication" => {
                format!(
                    "Max-ply adjudication (|eval| >= {} cp)",
                    config.maxply_adjudication
                )
            }
            "checkmate" => "Checkmate".to_string(),
            "allpiecescaptured" => "All pieces captured".to_string(),
            "allroyalscaptured" => "All royals captured".to_string(),
            "royalcapture" => "Royal capture".to_string(),
            "stalemate" => "Stalemate".to_string(),
            "fifty-move rule" => "50-move rule".to_string(),
            "threefold repetition" => "Threefold repetition".to_string(),
            "insufficient_material" => "Insufficient material".to_string(),
            "timeout" => "Loss on time".to_string(),
            "illegal move" => "Loss on illegal move".to_string(),
            "engine failure" => "Loss on engine failure".to_string(),
            "max_moves" => "Maximum moves reached".to_string(),
            _ => r.to_string(),
        };
        icn.push_str(&format!("[Termination \"{}\"] ", term));
    }

    icn.push_str(starting_board_setup);

    if !move_log.is_empty() {
        icn.push(' ');
        icn.push_str(&move_log.join("|"));
    }
    icn
}

// ─── Run abort ────────────────────────────────────────────────────────────────

/// Why a run stopped early. Recorded once: the first cause wins, so a lock
/// poisoned by an earlier panic can't overwrite the panic that poisoned it.
enum AbortReason {
    Panic {
        message: String,
        location: String,
        backtrace: String,
    },
    /// The engine under test misbehaved. Always a bug worth stopping for — a
    /// timeout is a legitimate loss, but a crash, a missing move or an illegal
    /// move means the remaining games would measure a broken engine.
    EngineFault {
        kind: &'static str,
        engine: &'static str,
        game_idx: usize,
        variant: String,
        detail: String,
    },
}

static ABORT: OnceLock<AbortReason> = OnceLock::new();
/// Set while a redraw-in-place view owns the bottom of the screen, so the panic
/// hook knows whether printing directly would corrupt it.
static LIVE_VIEW_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Record the first fatal cause and signal every worker to wind down.
fn abort_run(reason: AbortReason) {
    let _ = ABORT.set(reason);
    STOP.store(true, Ordering::SeqCst);
}

/// Capture panics from every thread. Without this a panicking worker dies
/// quietly, its channel closes, and the harness prints a normal-looking summary
/// over partial data.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };
        let location = info
            .location()
            .map_or_else(|| "unknown location".to_string(), |l| l.to_string());
        abort_run(AbortReason::Panic {
            message,
            location,
            backtrace: std::backtrace::Backtrace::force_capture().to_string(),
        });

        // A worker panic is reported by main once the live view is torn down.
        // A main-thread panic never reaches that path, so print it immediately.
        if std::thread::current().name() == Some("main") {
            if LIVE_VIEW_ACTIVE.load(Ordering::SeqCst) {
                println!();
            }
            default_hook(info);
        }
    }));
}

// ─── Rendering ────────────────────────────────────────────────────────────────

/// ANSI colour codes, empty when colour is disabled.
#[derive(Clone, Copy)]
struct Colors {
    green: &'static str,
    red: &'static str,
    yellow: &'static str,
    gray: &'static str,
    reset: &'static str,
}

impl Colors {
    fn new(enabled: bool) -> Self {
        if enabled {
            Self {
                green: "\x1b[32m",
                red: "\x1b[31m",
                yellow: "\x1b[33m",
                gray: "\x1b[90m",
                reset: "\x1b[0m",
            }
        } else {
            Self { green: "", red: "", yellow: "", gray: "", reset: "" }
        }
    }

    /// Green above +1, red below -1, grey in between — the "is this a gain"
    /// signal. Only ever applied to a point estimate, never to its error
    /// margin: the margin is precision, not a verdict.
    fn by_elo(&self, elo: f64) -> &'static str {
        if elo > 1.0 {
            self.green
        } else if elo < -1.0 {
            self.red
        } else {
            self.gray
        }
    }

    fn by_llr(&self, llr: f64, lower: f64, upper: f64) -> &'static str {
        if llr >= upper {
            self.green
        } else if llr <= lower {
            self.red
        } else {
            self.yellow
        }
    }
}

/// Where the live status is drawn, decided once at startup.
#[derive(Clone, Copy, PartialEq)]
enum OutputMode {
    /// Multi-line dashboard redrawn in place. Interactive terminals only.
    Full,
    /// One status line rewritten with `\r`. Interactive terminals only.
    Compact,
    /// Append-only lines, no cursor movement. Pipes, files, CI, `TERM=dumb`.
    Plain,
}

impl OutputMode {
    fn detect(compact_flag: bool) -> Self {
        let dumb = std::env::var("TERM").is_ok_and(|t| t == "dumb");
        if !std::io::stdout().is_terminal() || dumb {
            // Redrawing needs a terminal: `\r` into a log file is garbage, and
            // animations turn CI logs into christmas trees.
            OutputMode::Plain
        } else if compact_flag {
            OutputMode::Compact
        } else {
            OutputMode::Full
        }
    }
}

/// Every number the run reports. The live view and the final summary both
/// derive from this, so the two can never drift apart.
struct SprtStats {
    wins: usize,
    losses: usize,
    draws: usize,
    penta: PentaCounts,
    per_variant: HashMap<String, (usize, usize, usize)>,
    /// Display order for `per_variant` rows: the order variants were selected
    /// in, fixed for the run's lifetime. Sorting by live game count instead
    /// would reshuffle rows every update as counts leapfrog each other.
    variant_order: Vec<String>,
    /// Timeout losses for either engine.
    timeout_losses: usize,
    /// The subset of `timeout_losses` where the new engine was the one that
    /// ran out of time — the figure that actually matters for judging it.
    new_engine_timeouts: usize,
    /// Games already complete when a run resumed, excluded from throughput so
    /// the rate reflects this session rather than the whole history.
    resumed_games: usize,
    started: Instant,
    elo0: f64,
    elo1: f64,
    model: SprtModel,
    lower: f64,
    upper: f64,
    max_games: Option<usize>,
    /// Trailing (games played, LLR) samples, used to fit the ETA's trend line.
    llr_history: VecDeque<(f64, f64)>,
    /// The ETA is smoothed frame-to-frame (see [`Self::update_eta`]), so it is
    /// stored rather than recomputed fresh — a raw per-update estimate swings
    /// wildly early in a run.
    smoothed_eta: Option<Duration>,
}

impl SprtStats {
    fn total_games(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    fn elo(&self) -> PentaElo {
        estimate_pentanomial_elo(&self.penta)
    }

    fn llr(&self) -> f64 {
        calculate_pentanomial_llr(&self.penta, self.elo0, self.elo1, self.model)
    }

    fn los(&self) -> f64 {
        let pairs = self.penta.total_pairs() as f64;
        let los = calculate_los(self.penta.score(), self.penta.variance() / pairs);
        if los.is_nan() { 0.5 } else { los }
    }

    fn record(&mut self, outcome: &GameOutcome) {
        match outcome.result {
            GameResult::Win => self.wins += 1,
            GameResult::Loss => self.losses += 1,
            GameResult::Draw => self.draws += 1,
        }
        let entry = self
            .per_variant
            .entry(outcome.variant_name.clone())
            .or_insert((0, 0, 0));
        match outcome.result {
            GameResult::Win => entry.0 += 1,
            GameResult::Loss => entry.1 += 1,
            GameResult::Draw => entry.2 += 1,
        }
    }

    fn games_per_min(&self) -> f64 {
        let played = self.total_games().saturating_sub(self.resumed_games) as f64;
        let mins = self.started.elapsed().as_secs_f64() / 60.0;
        if mins > 0.0 { played / mins } else { 0.0 }
    }

    /// Last computed ETA. Call [`Self::update_eta`] once per status update to
    /// refresh it; this just reads the stored value.
    fn eta(&self) -> Option<Duration> {
        self.smoothed_eta
    }

    /// Recompute the ETA and fold it into the smoothed value. Call exactly
    /// once per status update (not from render code, which may run more than
    /// once per update) — otherwise the smoothing window means nothing.
    fn update_eta(&mut self) {
        let played = self.total_games().saturating_sub(self.resumed_games) as f64;
        let llr = self.llr();
        self.llr_history.push_back((played, llr));
        const HISTORY_CAP: usize = 60;
        if self.llr_history.len() > HISTORY_CAP {
            self.llr_history.pop_front();
        }

        let fresh = self.fresh_eta(played, llr);
        self.smoothed_eta = match (self.smoothed_eta, fresh) {
            // Blend in seconds so the displayed number drifts smoothly instead
            // of jumping to whatever the latest noisy sample says.
            (Some(prev), Some(new)) => {
                const SMOOTHING: f64 = 0.15; // weight given to the new estimate
                Some(Duration::from_secs_f64(
                    prev.as_secs_f64() * (1.0 - SMOOTHING) + new.as_secs_f64() * SMOOTHING,
                ))
            }
            // No prior estimate to blend with, or the trend just vanished
            // (flat/reversed) — show the fresh read (possibly None) directly.
            (_, fresh) => fresh,
        };
    }

    /// One fresh ETA estimate, before smoothing. Prefers the hard `--max-games`
    /// cap; otherwise fits a trend line through recent (games, LLR) samples and
    /// projects it to whichever bound it's actually heading for. A regression
    /// over a window is far steadier than extrapolating from a single current
    /// LLR value, which swings wildly — and can divide by near-zero — early in
    /// a run. Returns None rather than a number when the trend gives no honest
    /// answer: too little history, a flat slope, or heading away from both bounds.
    fn fresh_eta(&self, played: f64, llr: f64) -> Option<Duration> {
        let rate = self.games_per_min();
        if rate <= 0.0 {
            return None;
        }
        if let Some(max) = self.max_games {
            let remaining = max.saturating_sub(self.total_games()) as f64;
            return Some(Duration::from_secs_f64((remaining / rate * 60.0).min(u32::MAX as f64)));
        }

        const MIN_SAMPLES: usize = 10;
        if self.llr_history.len() < MIN_SAMPLES || played < 20.0 {
            return None;
        }

        // Least-squares slope of LLR against games played.
        let n = self.llr_history.len() as f64;
        let mean_x = self.llr_history.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_y = self.llr_history.iter().map(|(_, y)| y).sum::<f64>() / n;
        let (mut cov, mut var_x) = (0.0, 0.0);
        for &(x, y) in &self.llr_history {
            cov += (x - mean_x) * (y - mean_y);
            var_x += (x - mean_x).powi(2);
        }
        if var_x <= 0.0 {
            return None;
        }
        let slope = cov / var_x; // LLR gained per game, at the recent trend
        if slope.abs() < 1e-6 {
            return None; // flat trend: no meaningful ETA
        }

        let target = if slope > 0.0 { self.upper } else { self.lower };
        let remaining_llr = target - llr;
        // Matching signs means the trend is actually closing on that bound;
        // opposite signs mean it has turned away from it.
        if remaining_llr.signum() != slope.signum() {
            return None;
        }
        let remaining_games = (remaining_llr / slope).max(0.0);
        Some(Duration::from_secs_f64((remaining_games / rate * 60.0).min(u32::MAX as f64)))
    }
}

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// The live status region. Owns nothing else — the final summary is printed by
/// [`print_final_summary`] after this is torn down.
enum LiveView {
    Full(Box<FullView>),
    Compact { previous_len: usize },
    Plain { last_reported: usize },
}

/// Games between status lines in [`OutputMode::Plain`]. Sparse on purpose: this
/// output ends up in log files and CI transcripts.
const PLAIN_REPORT_INTERVAL: usize = 50;

/// How often the live view redraws when no game has finished, keeping the clock,
/// throughput and resize handling responsive without busy-looping.
const REFRESH_INTERVAL: Duration = Duration::from_millis(250);

impl LiveView {
    fn new(mode: OutputMode, variant_count: usize, colors: Colors) -> Self {
        match mode {
            OutputMode::Full => match FullView::new(variant_count, colors) {
                Ok(view) => {
                    LIVE_VIEW_ACTIVE.store(true, Ordering::SeqCst);
                    LiveView::Full(Box::new(view))
                }
                // A terminal that won't give us a viewport still gets a status line.
                Err(_) => LiveView::Compact { previous_len: 0 },
            },
            OutputMode::Compact => {
                LIVE_VIEW_ACTIVE.store(true, Ordering::SeqCst);
                LiveView::Compact { previous_len: 0 }
            }
            OutputMode::Plain => LiveView::Plain { last_reported: 0 },
        }
    }

    fn update(&mut self, stats: &SprtStats, colors: &Colors) {
        match self {
            LiveView::Full(view) => view.draw(stats),
            LiveView::Compact { previous_len } => {
                // Cap to the terminal width, or a `\r` overwrite only clears the
                // final wrapped row — the overflow from earlier, wider lines
                // strands itself above as permanent stuck fragments.
                let max_width = ratatui::crossterm::terminal::size()
                    .ok()
                    .map(|(cols, _)| cols.saturating_sub(1) as usize);
                let line = compact_status_line(stats, colors, max_width);
                // Pad to the previous width so leftovers from a longer line are cleared.
                let width = (*previous_len).max(line.len());
                print!("\r{line:<width$}");
                let _ = std::io::stdout().flush();
                *previous_len = line.len();
            }
            LiveView::Plain { last_reported } => {
                let total = stats.total_games();
                if total.saturating_sub(*last_reported) >= PLAIN_REPORT_INTERVAL {
                    *last_reported = total;
                    // No width cap: this is a log line, not a display — it's
                    // fine for it to wrap in whatever views the log later.
                    println!("{}", compact_status_line(stats, &Colors::new(false), None));
                }
            }
        }
    }

    /// Emit a line that scrolls above the live region into real scrollback.
    fn log(&mut self, message: &str) {
        match self {
            LiveView::Full(view) => view.log(message),
            LiveView::Compact { previous_len } => {
                // Overwrite the status line so it isn't left half-erased above the message.
                println!("\r{:<width$}", message, width = *previous_len);
                *previous_len = 0;
            }
            LiveView::Plain { .. } => println!("{message}"),
        }
    }

    /// Release the terminal. Must run before anything else prints, including a
    /// panic report, or the output lands inside the viewport.
    fn finish(self) {
        LIVE_VIEW_ACTIVE.store(false, Ordering::SeqCst);
        match self {
            LiveView::Full(view) => view.finish(),
            LiveView::Compact { .. } => println!(),
            LiveView::Plain { .. } => {}
        }
    }
}

/// One-line status, used by both the compact and plain modes.
/// One-line status. `max_width`, when given, drops trailing (less essential)
/// fields — richest first — until the line's *visible* width fits: never
/// mid-string truncation, which could slice through a color escape code and
/// bleed that color onto the rest of the terminal.
fn compact_status_line(stats: &SprtStats, colors: &Colors, max_width: Option<usize>) -> String {
    let pe = stats.elo();
    let color = colors.by_elo(pe.elo);
    // The only ANSI bytes in this line, so the overhead is exactly this much —
    // no need for a general-purpose escape-code scanner.
    let color_overhead = if colors.reset.is_empty() { 0 } else { color.len() + colors.reset.len() };

    let core = format!(
        "Games: {} ({} pairs) | W: {} L: {} D: {} | Elo: {color}{:.2}{} +/- {:.2}",
        stats.total_games(),
        stats.penta.total_pairs(),
        stats.wins,
        stats.losses,
        stats.draws,
        pe.elo,
        colors.reset,
        pe.elo_err,
    );
    let los = format!(" | LOS: {:.1}%", stats.los() * 100.0);
    let llr = format!(" | LLR: {:.2}", stats.llr());
    let bounds = format!(" [{:.2}, {:.2}]", stats.lower, stats.upper);

    let Some(max_width) = max_width else {
        return format!("{core}{los}{llr}{bounds}");
    };
    [
        format!("{core}{los}{llr}{bounds}"),
        format!("{core}{los}{llr}"),
        format!("{core}{llr}"), // LOS drops before LLR, the more decision-relevant figure
        core.clone(),
    ]
    .into_iter()
    .find(|line| line.len() - color_overhead <= max_width)
    .unwrap_or(core)
}

/// Multi-line dashboard drawn in an inline viewport: a fixed block at the
/// bottom of the terminal that is redrawn in place, leaving scrollback intact.
/// Deliberately not an alternate-screen TUI — those erase themselves on exit,
/// taking the run's output with them.
struct FullView {
    terminal: ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    /// Selected variant count, kept to re-derive the row budget on resize.
    variant_count: usize,
    /// Per-variant rows that fit; the remainder collapse into an "and N more" row.
    variant_rows: usize,
    colors: Colors,
}

/// Dashboard lines that are not per-variant rows (header, metrics, alerts, footer).
const FULL_VIEW_CHROME: usize = 9;

impl FullView {
    /// How many variant rows fit, and the resulting viewport height, for the
    /// terminal's *current* size — recomputed on every resize, not just at
    /// startup.
    fn layout_for(variant_count: usize) -> (usize, u16) {
        let term_height = ratatui::crossterm::terminal::size().map_or(24, |(_, h)| h) as usize;
        let budget = term_height.saturating_sub(4).max(6);
        let variant_rows = variant_count.min(budget.saturating_sub(FULL_VIEW_CHROME));
        let height = (FULL_VIEW_CHROME + variant_rows).min(budget) as u16;
        (variant_rows, height)
    }

    fn new(variant_count: usize, colors: Colors) -> std::io::Result<Self> {
        let (variant_rows, height) = Self::layout_for(variant_count);
        let terminal = ratatui::Terminal::with_options(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions { viewport: ratatui::Viewport::Inline(height) },
        )?;
        Ok(Self { terminal, variant_count, variant_rows, colors })
    }

    /// Ratatui's inline viewport height is fixed at construction: `Terminal::resize`
    /// only repositions it for the new terminal size, clamped to that *original*
    /// height (see `compute_inline_size` upstream) — it never grows back or
    /// shrinks further. So a real height change tears down and recreates the
    /// viewport instead.
    fn resize_if_needed(&mut self) {
        let (variant_rows, height) = Self::layout_for(self.variant_count);
        if variant_rows == self.variant_rows {
            return;
        }
        // `clear` erases relative to ratatui's *stored* viewport position, which
        // the window resize just invalidated. Sync it to the real terminal size
        // first, or the clear wipes the wrong rows and strands the old frame in
        // the scrollback above the new viewport.
        let _ = self.terminal.autoresize();
        // Erase the old viewport and park the cursor at its top, so the new one
        // starts exactly there instead of duplicating space below it.
        let _ = self.terminal.clear();
        let _ = std::io::stdout().flush();
        if let Ok(new_terminal) = ratatui::Terminal::with_options(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions { viewport: ratatui::Viewport::Inline(height) },
        ) {
            self.terminal = new_terminal;
            self.variant_rows = variant_rows;
        }
    }

    fn draw(&mut self, stats: &SprtStats) {
        self.resize_if_needed();
        let lines = self.compose(stats);
        let _ = self.terminal.draw(|frame| {
            frame.render_widget(ratatui::widgets::Paragraph::new(lines), frame.area());
        });
    }

    fn log(&mut self, message: &str) {
        use ratatui::widgets::Widget;
        let owned = message.to_string();
        let _ = self.terminal.insert_before(1, move |buf| {
            ratatui::widgets::Paragraph::new(owned).render(buf.area, buf);
        });
    }

    fn finish(mut self) {
        // Collapse the viewport so the final summary is printed over it rather
        // than below a stale dashboard.
        let _ = self.terminal.clear();
        let _ = std::io::stdout().flush();
    }

    fn compose(&self, stats: &SprtStats) -> Vec<ratatui::text::Line<'static>> {
        use ratatui::style::{Color, Modifier, Style};
        use ratatui::text::{Line, Span};

        let dim = if self.colors.reset.is_empty() {
            Style::default()
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let styled = |color: Color| {
            if self.colors.reset.is_empty() {
                Style::default()
            } else {
                Style::default().fg(color)
            }
        };
        let elo_color = |elo: f64| {
            if elo > 1.0 {
                Color::Green
            } else if elo < -1.0 {
                Color::Red
            } else {
                Color::DarkGray
            }
        };

        let pe = stats.elo();
        let llr = stats.llr();
        let total = stats.total_games();
        let width = self.terminal.size().map_or(78, |s| s.width) as usize;
        let rule = Line::from(Span::styled("─".repeat(width.min(78)), dim));

        let mut lines = vec![
            Line::from(vec![
                Span::styled(
                    "SPRT",
                    if self.colors.reset.is_empty() {
                        Style::default()
                    } else {
                        Style::default().add_modifier(Modifier::BOLD)
                    },
                ),
                Span::raw(format!(
                    "  {} games  ·  {} pairs  ·  {:.1} games/min{}",
                    total,
                    stats.penta.total_pairs(),
                    stats.games_per_min(),
                    stats
                        .eta()
                        .map_or_else(String::new, |eta| format!("  ·  eta ~{}", fmt_duration(eta))),
                )),
            ]),
            rule.clone(),
            Line::from(format!(
                "  W {}   L {}   D {}      elapsed {}",
                stats.wins,
                stats.losses,
                stats.draws,
                fmt_duration(stats.started.elapsed()),
            )),
            Line::from(vec![
                Span::raw("  Elo   "),
                Span::styled(format!("{:+.2}", pe.elo), styled(elo_color(pe.elo))),
                Span::raw(format!(" ± {:.2}", pe.elo_err)),
                Span::raw(format!("        LOS  {:.1}%", stats.los() * 100.0)),
            ]),
            Line::from(vec![
                Span::raw("  nElo  "),
                Span::styled(format!("{:+.2}", pe.nelo), styled(elo_color(pe.elo))),
                Span::raw(format!(" ± {:.2}", pe.nelo_err)),
            ]),
            Line::from(vec![
                Span::raw("  LLR   "),
                Span::styled(
                    format!("{llr:+.2}"),
                    styled(if llr >= stats.upper {
                        Color::Green
                    } else if llr <= stats.lower {
                        Color::Red
                    } else {
                        Color::Yellow
                    }),
                ),
                Span::raw(format!("  [{:.2}, {:.2}]", stats.lower, stats.upper)),
            ]),
            rule.clone(),
        ];

        // Per-variant rows, in selection order — fixed for the run's lifetime, so
        // a row's position never changes as its game count edges past another
        // variant's. Not sorted by live count: variants are assigned round-robin
        // (see play_game), so counts stay roughly even anyway, and re-sorting
        // would only reshuffle rows for no real benefit.
        let variants: Vec<_> = stats
            .variant_order
            .iter()
            .map(|name| (name, stats.per_variant.get(name).copied().unwrap_or_default()))
            .collect();
        for (name, (w, l, d)) in variants.iter().take(self.variant_rows) {
            let (velo, verr) = estimate_elo(*w, *l, *d);
            lines.push(Line::from(vec![
                Span::raw(format!("  {:<22}{:>5}  ", truncate(name, 21), w + l + d)),
                Span::styled(format!("{:>3}/{:>3}/{:>3}", w, l, d), dim),
                Span::raw("  "),
                Span::styled(format!("{velo:>+7.1}"), styled(elo_color(velo))),
                Span::styled(format!(" ± {verr:.0}"), dim),
            ]));
        }
        if variants.len() > self.variant_rows {
            lines.push(Line::from(Span::styled(
                format!("  …and {} more", variants.len() - self.variant_rows),
                dim,
            )));
        }
        for _ in variants.len()..self.variant_rows {
            lines.push(Line::from(""));
        }

        lines.push(rule);
        lines.push(if stats.timeout_losses > 0 {
            Line::from(Span::styled(
                // Two spaces after the glyph: it renders double-width, so a
                // single space leaves it touching the count.
                format!(
                    "  ⚠  {} timeout losses ({} from new)",
                    stats.timeout_losses, stats.new_engine_timeouts
                ),
                styled(Color::Red),
            ))
        } else {
            Line::from(Span::styled("  no timeouts", dim))
        });
        lines
    }
}

/// Report a fatal cause loudly, with enough context to reproduce it, and make
/// clear the partial numbers are not a result.
fn print_abort_report(reason: &AbortReason, stats: &SprtStats, colors: &Colors) {
    let Colors { red, yellow, reset, .. } = *colors;
    eprintln!("\n{red}════════════════════════ SPRT ABORTED ════════════════════════{reset}");
    match reason {
        AbortReason::Panic { message, location, backtrace } => {
            eprintln!("{red}The harness panicked. This is a bug in sprt.rs, not in the engine.{reset}");
            eprintln!("\n  panic: {message}");
            eprintln!("  at:    {location}");
            eprintln!("\nbacktrace:\n{backtrace}");
        }
        AbortReason::EngineFault { kind, engine, game_idx, variant, detail } => {
            eprintln!("{red}The {engine} engine hit a fault: {kind}{reset}");
            eprintln!("\n  game:    {game_idx}");
            eprintln!("  variant: {variant}");
            eprintln!("\n{detail}");
        }
    }
    eprintln!(
        "\n{yellow}Partial results after {} games ({} pairs) — NOT a valid measurement:{reset}",
        stats.total_games(),
        stats.penta.total_pairs()
    );
    let pe = stats.elo();
    eprintln!(
        "  W {} L {} D {} | Elo: {:.2} +/- {:.2} | LLR: {:.2}",
        stats.wins,
        stats.losses,
        stats.draws,
        pe.elo,
        pe.elo_err,
        stats.llr()
    );
    eprintln!("{red}══════════════════════════════════════════════════════════════{reset}");
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max.saturating_sub(1)).chain(['…']).collect()
    }
}

fn account_move_time(
    current_clock: u64,
    elapsed: u64,
    increment: u64,
    fixed_time: bool,
) -> (bool, u64) {
    // A fixed time control is a fresh per-move budget, not a cumulative game
    // clock. The engine enforces --fixed-time internally.
    if fixed_time {
        return (false, current_clock);
    }
    if current_clock < elapsed {
        return (true, 0);
    }
    (false, current_clock - elapsed + increment)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Run {
            new_bin,
            old_bin,
            elo0,
            elo1,
            model,
            alpha,
            beta,
            tc,
            concurrency,
            max_games,
            min_games,
            variants,
            adjudication,
            maxply_adjudication,
            games,
            results,
            max_moves,
            search_noise,
            old_strength,
            new_strength,
            verbose,
            compact,
            new_commit,
            old_commit,
            resume,
            save_interval,
        }) => {
            let concurrency = concurrency.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            let actual_new_bin = if let Some(path) = new_bin {
                path
            } else {
                println!("No --new-bin provided. Using current binary...");
                let ext = std::env::consts::EXE_EXTENSION;
                if ext.is_empty() {
                    "target/release/sprt".to_string()
                } else {
                    format!("target/release/sprt.{}", ext)
                }
            };

            // Compared against the parsed args to tell an explicit flag from a default.
            const DEFAULT_TC_STR: &str = "10+0.1";
            const DEFAULT_VARIANTS_STR: &str = "Classical,Confined_Classical,Classical_Plus,Core,CoaIP,CoaIP_HO,CoaIP_RO,CoaIP_NO,Palace,Pawndard,Standarch,Space_Classic,Space,Knightline,Scattered_Leapers";

            // Load resume state if --resume was provided
            let resume_state_opt: Option<ResumeState> = resume.as_deref().and_then(|path| {
                if std::path::Path::new(path).exists() {
                    Some(load_resume_state(path))
                } else {
                    println!("Resume file '{}' not found, starting fresh.", path);
                    None
                }
            });

            let resume_pair_offset_val = resume_state_opt
                .as_ref()
                .map_or(0, |rs| rs.resume_pair_offset);

            // Auto-detect TC from resume if the user didn't supply an explicit value
            let tc = match &resume_state_opt {
                Some(rs) if tc == DEFAULT_TC_STR => rs.detected_tc.clone().unwrap_or(tc),
                _ => tc,
            };

            // Auto-detect variants from resume if the user didn't supply an explicit value
            let variants = match &resume_state_opt {
                Some(rs)
                    if variants == DEFAULT_VARIANTS_STR && !rs.detected_variants.is_empty() =>
                {
                    rs.detected_variants.join(",")
                }
                _ => variants,
            };

            // ── Variant presets ──────────────────────────────────────────────────────
            // all        — every variant in the engine
            // site       — variants live on the public site (image order), no Abundance/Showcase
            // base_only  — base-eval standard variants; no multi-king, no AllPieces, no Abundance
            //              Default for testing base.rs changes against typical positions.
            // base_full  — all base-eval variants including multi-king and AllPiecesClassical
            // multi_king — only the double/triple-king variants
            // coaip_set  — only the Chess-on-an-Infinite-Plane family

            // Every variant in the engine
            const ALL_VARIANTS: &[Variant] = &[
                Variant::Classical,
                Variant::ConfinedClassical,
                Variant::ClassicalPlus,
                Variant::CoaIP,
                Variant::CoaIPHO,
                Variant::CoaIPRO,
                Variant::CoaIPNO,
                Variant::Palace,
                Variant::Pawndard,
                Variant::Core,
                Variant::Standarch,
                Variant::SpaceClassic,
                Variant::Space,
                Variant::Abundance,
                Variant::PawnHorde,
                Variant::Knightline,
                Variant::Obstocean,
                Variant::Chess,
                Variant::ScatteredLeapers,
                Variant::DoubleKingClassical,
                Variant::DoubleKingChess,
                Variant::TripleKingMaze,
                Variant::AllPiecesClassical,
            ];

            // Variants live on the public site (image order), minus Abundance and Showcase
            const SITE_VARIANTS: &[Variant] = &[
                Variant::Classical,
                Variant::ConfinedClassical,
                Variant::ClassicalPlus,
                Variant::CoaIP,
                Variant::CoaIPHO,
                Variant::CoaIPRO,
                Variant::CoaIPNO,
                Variant::Palace,
                Variant::Pawndard,
                Variant::Core,
                Variant::Standarch,
                Variant::SpaceClassic,
                Variant::Space,
                Variant::PawnHorde,
                Variant::Knightline,
                Variant::Obstocean,
                Variant::Chess,
            ];

            // Base-eval variants only: no custom evaluators, multi-king,
            // AllPiecesClassical or Abundance. The best default for base.rs changes.
            const BASE_ONLY_VARIANTS: &[Variant] = &[
                Variant::Classical,
                Variant::ConfinedClassical,
                Variant::ClassicalPlus,
                Variant::CoaIP,
                Variant::CoaIPHO,
                Variant::CoaIPRO,
                Variant::CoaIPNO,
                Variant::Palace,
                Variant::Pawndard,
                Variant::Core,
                Variant::Standarch,
                Variant::SpaceClassic,
                Variant::Space,
                Variant::Knightline,
                Variant::ScatteredLeapers,
            ];

            // All base-eval variants including multi-king and AllPiecesClassical.
            // Use when you specifically want coverage of the exotic base-eval positions.
            const BASE_FULL_VARIANTS: &[Variant] = &[
                Variant::Classical,
                Variant::ConfinedClassical,
                Variant::ClassicalPlus,
                Variant::CoaIP,
                Variant::CoaIPHO,
                Variant::CoaIPRO,
                Variant::CoaIPNO,
                Variant::Palace,
                Variant::Pawndard,
                Variant::Core,
                Variant::Standarch,
                Variant::SpaceClassic,
                Variant::Space,
                Variant::Knightline,
                Variant::ScatteredLeapers,
                Variant::DoubleKingClassical,
                Variant::DoubleKingChess,
                Variant::TripleKingMaze,
                Variant::AllPiecesClassical,
            ];

            // Only the double/triple-king variants
            const MULTI_KING_VARIANTS: &[Variant] = &[
                Variant::DoubleKingClassical,
                Variant::DoubleKingChess,
                Variant::TripleKingMaze,
            ];

            // Only the Chess-on-an-Infinite-Plane family
            const COAIP_VARIANTS: &[Variant] = &[
                Variant::CoaIP,
                Variant::CoaIPHO,
                Variant::CoaIPRO,
                Variant::CoaIPNO,
            ];

            let parsed_variants = match variants.trim().to_lowercase().as_str() {
                "all" => ALL_VARIANTS.to_vec(),
                "site" => SITE_VARIANTS.to_vec(),
                "base_only" => BASE_ONLY_VARIANTS.to_vec(),
                "base_full" => BASE_FULL_VARIANTS.to_vec(),
                "multi_king" => MULTI_KING_VARIANTS.to_vec(),
                "coaip_set" => COAIP_VARIANTS.to_vec(),
                _ => {
                    let mut parsed = Vec::new();
                    for name in variants.split(',') {
                        let name_trimmed = name.trim();
                        let name_lower = name_trimmed.to_lowercase().replace(' ', "_");
                        let known = matches!(
                            name_lower.as_str(),
                            "classical"
                                | "confined_classical"
                                | "classical_plus"
                                | "coaip"
                                | "coaip_ho"
                                | "coaip_ro"
                                | "coaip_no"
                                | "palace"
                                | "pawndard"
                                | "core"
                                | "standarch"
                                | "space_classic"
                                | "space"
                                | "abundance"
                                | "pawn_horde"
                                | "knightline"
                                | "obstocean"
                                | "chess"
                                | "scattered_leapers"
                                | "double_king_classical"
                                | "double_king_chess"
                                | "triple_king_maze"
                                | "all_pieces_classical"
                        );
                        if !known {
                            eprintln!(
                                "Error: Unknown variant or preset '{}'. \
                                 Valid presets: all, site, base_only, base_full, multi_king, coaip_set",
                                name_trimmed
                            );
                            std::process::exit(1);
                        }
                        parsed.push(Variant::parse(name_trimmed).unwrap_or(Variant::Classical));
                    }
                    parsed
                }
            };

            let mut config = Config {
                elo0,
                elo1,
                model: SprtModel::parse(&model),
                alpha,
                beta,
                tc: tc.clone(),
                tc_base_ms: 10000,
                tc_inc_ms: 100,
                tc_fixed_ms: None,
                tc_max_depth: None,
                concurrency,
                max_games,
                min_games,
                variants: parsed_variants,
                adjudication_threshold: adjudication,
                maxply_adjudication,
                new_bin: actual_new_bin,
                old_bin,
                max_moves,
                search_noise,
                new_strength: new_strength.clamp(1, apeiron::search::MAX_SITE_SKILL),
                old_strength: old_strength.clamp(1, apeiron::search::MAX_SITE_SKILL),
                verbose,
                compact,
                new_commit_info: None,
                old_commit_info: None,
                resume_pair_offset: resume_pair_offset_val,
                save_interval: save_interval.max(1),
            };

            // Resolve old commit info: explicit CLI arg > query the old binary itself.
            config.old_commit_info = if let Some(sha) = old_commit {
                let date = get_commit_date_from_git(&sha);
                Some(CommitInfo {
                    commit: sha,
                    date,
                    dirty: false,
                })
            } else {
                try_query_binary_commit_info(&config.old_bin)
            };

            // Resolve new commit info: explicit CLI arg > build-time embedded value > git HEAD.
            config.new_commit_info = if let Some(sha) = new_commit {
                let date = get_commit_date_from_git(&sha);
                Some(CommitInfo {
                    commit: sha,
                    date,
                    dirty: false,
                })
            } else if let Some(commit) = BUILD_COMMIT.filter(|s| !s.is_empty()) {
                let is_dirty = BUILD_DIRTY.map(|d| d == "1").unwrap_or(false);
                Some(CommitInfo {
                    commit: commit.to_string(),
                    date: BUILD_DATE.unwrap_or("").to_string(),
                    dirty: is_dirty,
                })
            } else {
                try_get_commit_info_from_git("HEAD")
            };

            let games_path = games.or_else(|| resume.clone());
            let results_path = results;

            install_panic_hook();

            ctrlc::set_handler(move || {
                USER_STOP.store(true, Ordering::SeqCst);
                STOP.store(true, Ordering::SeqCst);
            })
            .expect("Error setting Ctrl-C handler");

            if tc.contains('+') {
                let parts: Vec<&str> = tc.split('+').collect();
                config.tc_base_ms =
                    (parts[0].parse::<f64>().expect("Invalid base time") * 1000.0) as u64;
                config.tc_inc_ms =
                    (parts[1].parse::<f64>().expect("Invalid increment") * 1000.0) as u64;
            } else if tc.starts_with("depth ") {
                config.tc_max_depth =
                    Some(tc.replace("depth ", "").parse().expect("Invalid depth"));
            } else if tc.starts_with("fixed ") {
                config.tc_fixed_ms = Some(
                    (tc.replace("fixed ", "")
                        .replace("s", "")
                        .parse::<f64>()
                        .expect("Invalid fixed time")
                        * 1000.0) as u32,
                );
            }

            let (lower, upper) = (
                (config.beta / (1.0 - config.alpha)).ln(),
                ((1.0 - config.beta) / config.alpha).ln(),
            );
            let mut game_logs: Vec<String> = Vec::new();
            // Fixed display order, deduplicated, so a repeated preset entry
            // doesn't produce a doubled-up row.
            let mut variant_order = Vec::with_capacity(config.variants.len());
            let mut seen_variants = HashSet::new();
            for variant in &config.variants {
                let name = variant.to_str().to_string();
                if seen_variants.insert(name.clone()) {
                    variant_order.push(name);
                }
            }
            let mut stats = SprtStats {
                wins: 0,
                losses: 0,
                draws: 0,
                penta: PentaCounts::default(),
                per_variant: variant_order.iter().map(|name| (name.clone(), (0, 0, 0))).collect(),
                variant_order,
                timeout_losses: 0,
                new_engine_timeouts: 0,
                resumed_games: 0,
                started: Instant::now(),
                elo0: config.elo0,
                elo1: config.elo1,
                model: config.model,
                lower,
                upper,
                max_games: config.max_games,
                llr_history: VecDeque::new(),
                smoothed_eta: None,
            };

            // Apply resume state: seed W/L/D, pentanomial pairs, per-variant stats, prior logs
            if let Some(rs) = resume_state_opt {
                stats.wins = rs.wins;
                stats.losses = rs.losses;
                stats.draws = rs.draws;
                stats.penta = rs.penta;
                // Resumed variants merge into the pre-seeded map rather than
                // replacing it, so the fixed row order still covers every variant.
                for (name, counts) in rs.per_variant_stats {
                    stats.per_variant.insert(name, counts);
                }
                stats.resumed_games = rs.wins + rs.losses + rs.draws;
                game_logs = rs.games;
            }

            // No color when stdout isn't a terminal (piped/redirected) or NO_COLOR is
            // set (https://no-color.org), so raw escape codes never leak as garbage text.
            let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
            let colors = Colors::new(use_color);
            let Colors { green, red, yellow, reset, .. } = colors;
            let output_mode = OutputMode::detect(config.compact);

            println!("\nStarting SPRT with Configuration:");
            print_commit_context(&config.new_commit_info, &config.old_commit_info);
            print_settings_context(&config);
            if config.resume_pair_offset > 0 {
                println!(
                    "  Resuming: {} games loaded ({}W / {}L / {}D), next pair = {}",
                    stats.total_games(),
                    stats.wins,
                    stats.losses,
                    stats.draws,
                    config.resume_pair_offset
                );
            }
            println!();

            let mut view = LiveView::new(output_mode, config.variants.len(), colors);

            let (tx, rx) = std::sync::mpsc::channel();
            let config_clone = config.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.concurrency.max(1))
                .build()
                .expect("Failed to build Rayon thread pool");

            let producer = std::thread::spawn(move || {
                pool.install(|| {
                    rayon::scope(|scope| {
                        for worker_idx in 0..config_clone.concurrency.max(1) {
                            let tx = tx.clone();
                            let config = config_clone.clone();
                            scope.spawn(move |_| {
                                let mut pair_idx = config.resume_pair_offset + worker_idx;
                                let step = config.concurrency.max(1);

                                loop {
                                    if STOP.load(Ordering::SeqCst) {
                                        break;
                                    }

                                    let game_idx_even = pair_idx * 2;
                                    let game_idx_odd = game_idx_even + 1;

                                    if let Some(max_games) = config.max_games
                                        && game_idx_even >= max_games
                                    {
                                        break;
                                    }

                                    let variant = config.variants[pair_idx % config.variants.len()];
                                    let mut seeds = Vec::with_capacity(config.max_moves);
                                    for _ in 0..config.max_moves {
                                        seeds.push(rand::random::<u64>());
                                    }

                                    let play_new_white_first = rand::random::<bool>();
                                    let mut pair_outcomes = Vec::with_capacity(2);

                                    if play_new_white_first {
                                        pair_outcomes.push(play_game(
                                            &config,
                                            variant,
                                            true,
                                            game_idx_even,
                                            seeds.clone(),
                                        ));
                                        if STOP.load(Ordering::SeqCst) {
                                            let _ = tx.send(pair_outcomes);
                                            break;
                                        }
                                        if config.max_games.is_none_or(|max| game_idx_odd < max) {
                                            pair_outcomes.push(play_game(
                                                &config,
                                                variant,
                                                false,
                                                game_idx_odd,
                                                seeds,
                                            ));
                                        }
                                    } else {
                                        if config.max_games.is_none_or(|max| game_idx_odd < max) {
                                            pair_outcomes.push(play_game(
                                                &config,
                                                variant,
                                                false,
                                                game_idx_odd,
                                                seeds.clone(),
                                            ));
                                        }
                                        if STOP.load(Ordering::SeqCst) {
                                            let _ = tx.send(pair_outcomes);
                                            break;
                                        }
                                        pair_outcomes.push(play_game(
                                            &config,
                                            variant,
                                            true,
                                            game_idx_even,
                                            seeds,
                                        ));
                                    }

                                    if tx.send(pair_outcomes).is_err() {
                                        break;
                                    }

                                    pair_idx += step;
                                }
                            });
                        }
                    });
                });
            });

            // Track how many games were saved so we know when the next batch is due
            let mut last_save_len = game_logs.len();

            let mut verdict: Option<&str> = None;
            loop {
                // A fault recorded by a worker invalidates everything after it.
                if ABORT.get().is_some() {
                    break;
                }

                // Receive on a timer rather than blocking: the Full dashboard's
                // clock, throughput, ETA and terminal-resize check all need to
                // keep ticking between game completions, which at slow time
                // controls are far apart. Compact and Plain only ever redraw on
                // an actual result — a single status line has nothing to animate,
                // and Plain's output is a log, not a display.
                let pair_outcomes = match rx.recv_timeout(REFRESH_INTERVAL) {
                    Ok(outcomes) => outcomes,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if output_mode == OutputMode::Full {
                            view.update(&stats, &colors);
                        }
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };

                // NEW-perspective results of this pair's completed games, for
                // pentanomial bucketing (only full 2-game pairs count).
                let mut pair_results: Vec<GameResult> = Vec::with_capacity(2);
                for outcome in pair_outcomes {
                    if outcome.termination_reason == "interrupted" {
                        continue;
                    }

                    if outcome.termination_reason == "timeout" {
                        stats.timeout_losses += 1;
                        if outcome.new_engine_timed_out {
                            stats.new_engine_timeouts += 1;
                        }
                        if outcome.new_engine_timed_out && config.verbose {
                            view.log(&format!(
                                "ALERT: Game {} ended by timeout [{}] - NEW ENGINE TIMED OUT",
                                outcome.game_idx, outcome.variant_name
                            ));
                        }
                    }

                    stats.record(&outcome);
                    pair_results.push(outcome.result);
                    game_logs.push(outcome.icn);
                }

                if pair_results.len() == 2 {
                    stats.penta.add_pair(pair_results[0], pair_results[1]);
                }

                // Batch save: write the games file periodically so progress is not lost on crash
                if let Some(ref path) = games_path
                    && game_logs.len().saturating_sub(last_save_len) >= config.save_interval
                {
                    save_games_file(path, &game_logs);
                    last_save_len = game_logs.len();
                }

                stats.update_eta();
                view.update(&stats, &colors);

                if stats.total_games() >= config.min_games {
                    let llr = stats.llr();
                    if llr >= upper {
                        verdict = Some("PASS");
                    } else if llr <= lower {
                        verdict = Some("FAIL");
                    }
                    if verdict.is_some() {
                        STOP.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }

            // Release the terminal before anything else prints: a report written
            // while the live view owns the bottom of the screen is unreadable.
            view.finish();

            // Surface a worker panic that the channel closing would otherwise hide.
            let producer_panicked = producer.join().is_err();

            if let Some(reason) = ABORT.get() {
                // Persist whatever was played, so an abort doesn't cost the games.
                if let Some(ref path) = games_path {
                    save_games_file(path, &game_logs);
                }
                print_abort_report(reason, &stats, &colors);
                std::process::exit(1);
            }
            if producer_panicked {
                eprintln!(
                    "{red}SPRT ABORTED: a worker thread panicked but no cause was recorded.{reset}"
                );
                std::process::exit(1);
            }

            match verdict {
                Some("PASS") => println!("\n{green}SPRT: PASS {reset}"),
                Some("FAIL") => println!("\n{red}SPRT: FAIL {reset}"),
                _ => {}
            }
            if USER_STOP.load(Ordering::SeqCst) {
                println!("\n{yellow}SPRT: INCONCLUSIVE {reset}");
                println!("\nRun stopped by user.");
            }

            println!("\n\nFinal Summary:");
            print_commit_context(&config.new_commit_info, &config.old_commit_info);
            print_settings_context(&config);
            let pe = stats.elo();
            let final_penta_llr = stats.llr();
            let model_name = match config.model {
                SprtModel::Normalized => "normalized",
                SprtModel::Logistic => "logistic",
            };

            let text_color = colors.by_elo(pe.elo);
            println!("  Elo: {text_color}{:.2}{reset} +/- {:.2}", pe.elo, pe.elo_err);
            println!("  nElo: {text_color}{:.2}{reset} +/- {:.2}", pe.nelo, pe.nelo_err);

            println!(
                "  Games: {} | W: {} L: {} D: {}",
                stats.total_games(),
                stats.wins,
                stats.losses,
                stats.draws
            );
            println!(
                "  Pentanomial [{} pairs] (0-2): {}, {}, {}, {}, {}",
                stats.penta.total_pairs(),
                stats.penta.ll,
                stats.penta.ld,
                stats.penta.wl + stats.penta.dd,
                stats.penta.wd,
                stats.penta.ww
            );

            let text_color = colors.by_llr(final_penta_llr, lower, upper);
            println!(
                "  LLR: {text_color}{final_penta_llr:.3}{reset}  bounds [{lower:.2}, {upper:.2}] ({model_name} model, [{}, {}])",
                config.elo0, config.elo1
            );

            // Engine failures and illegal moves abort the run outright, so a run
            // that reaches this point can only have accumulated timeout losses.
            if stats.timeout_losses > 0 {
                println!(
                    "{red}  ALERT: {} games ended by timeout ({} from new) {reset}",
                    stats.timeout_losses, stats.new_engine_timeouts
                );
            }
            println!("\nPer-Variant Breakdown:");
            let mut variant_names: Vec<_> = stats.per_variant.keys().collect();
            variant_names.sort();
            for name in variant_names {
                let (vw, vl, vd) = stats.per_variant[name];
                let (velo, verr) = estimate_elo(vw, vl, vd);
                println!(
                    "  [{}]: {}W - {}L - {}D, Elo: {:.1} +/- {:.1}",
                    name, vw, vl, vd, velo, verr
                );
            }

            if let Some(path) = games_path {
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    std::fs::create_dir_all(parent).expect("Failed to create games output directory");
                }
                let json_data = serde_json::to_string_pretty(&game_logs).unwrap();
                std::fs::write(path, json_data).expect("Failed to write JSON output");
            }
            if let Some(path) = results_path {
                #[derive(Serialize)]
                struct ResultSettings {
                    tc: String,
                    elo0: f64,
                    elo1: f64,
                    alpha: f64,
                    beta: f64,
                    concurrency: usize,
                    variant_count: usize,
                    adjudication: i32,
                    maxply_adjudication: f64,
                    min_games: usize,
                    max_games: Option<usize>,
                    new_strength: u32,
                    old_strength: u32,
                }
                #[derive(Serialize)]
                struct FinalResults {
                    #[serde(skip_serializing_if = "Option::is_none")]
                    new_commit: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    new_commit_date: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    old_commit: Option<String>,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    old_commit_date: Option<String>,
                    settings: ResultSettings,
                    wins: usize,
                    losses: usize,
                    draws: usize,
                    timeout_losses: usize,
                    new_engine_timeouts: usize,
                    // elo/elo_error/llr are the pentanomial (Fishtest-model) values;
                    // field names kept stable for existing consumers.
                    elo: f64,
                    elo_error: f64,
                    nelo: f64,
                    nelo_error: f64,
                    llr: f64,
                    model: String,
                    total_pairs: usize,
                    penta_ll: usize,
                    penta_ld: usize,
                    penta_wl_dd: usize,
                    penta_wd: usize,
                    penta_ww: usize,
                    total_games: usize,
                    per_variant: HashMap<String, (usize, usize, usize)>,
                }
                let final_llr = stats.llr();
                let final_pe = stats.elo();
                let model_str = match config.model {
                    SprtModel::Normalized => "normalized",
                    SprtModel::Logistic => "logistic",
                }
                .to_string();
                let res = FinalResults {
                    new_commit: config.new_commit_info.as_ref().map(|c| c.commit.clone()),
                    new_commit_date: config
                        .new_commit_info
                        .as_ref()
                        .filter(|c| !c.date.is_empty())
                        .map(|c| c.date.clone()),
                    old_commit: config.old_commit_info.as_ref().map(|c| c.commit.clone()),
                    old_commit_date: config
                        .old_commit_info
                        .as_ref()
                        .filter(|c| !c.date.is_empty())
                        .map(|c| c.date.clone()),
                    settings: ResultSettings {
                        tc: config.tc.clone(),
                        elo0: config.elo0,
                        elo1: config.elo1,
                        alpha: config.alpha,
                        beta: config.beta,
                        concurrency: config.concurrency,
                        variant_count: config.variants.len(),
                        adjudication: config.adjudication_threshold,
                        maxply_adjudication: config.maxply_adjudication,
                        min_games: config.min_games,
                        max_games: config.max_games,
                        new_strength: config.new_strength,
                        old_strength: config.old_strength,
                    },
                    wins: stats.wins,
                    losses: stats.losses,
                    draws: stats.draws,
                    timeout_losses: stats.timeout_losses,
                    new_engine_timeouts: stats.new_engine_timeouts,
                    elo: final_pe.elo,
                    elo_error: final_pe.elo_err,
                    nelo: final_pe.nelo,
                    nelo_error: final_pe.nelo_err,
                    llr: final_llr,
                    model: model_str,
                    total_pairs: stats.penta.total_pairs(),
                    penta_ll: stats.penta.ll,
                    penta_ld: stats.penta.ld,
                    penta_wl_dd: stats.penta.wl + stats.penta.dd,
                    penta_wd: stats.penta.wd,
                    penta_ww: stats.penta.ww,
                    total_games: stats.total_games(),
                    per_variant: stats.per_variant,
                };
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    std::fs::create_dir_all(parent).expect("Failed to create results output directory");
                }
                let json_data = serde_json::to_string_pretty(&res).unwrap();
                std::fs::write(path, json_data).expect("Failed to write results output");
            }
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
            strength_level,
        }) => {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut engine = Engine::from_icn_native(icn.as_str(), strength_level);
                engine.set_clock(wtime, btime, winc, binc);
                engine.game_mut().variant = Variant::parse(&variant);
                if let Some(terminal) = detect_terminal_state(engine.game_mut()) {
                    match terminal {
                        TerminalState::Checkmate { white_won } => {
                            eprintln!(
                                "terminal checkmate winner {}",
                                if white_won { "white" } else { "black" }
                            );
                        }
                        TerminalState::AllPiecesCaptured { white_won } => {
                            eprintln!(
                                "terminal allpiecescaptured winner {}",
                                if white_won { "white" } else { "black" }
                            );
                        }
                        TerminalState::AllRoyalsCaptured { white_won } => {
                            eprintln!(
                                "terminal allroyalscaptured winner {}",
                                if white_won { "white" } else { "black" }
                            );
                        }
                        TerminalState::RoyalCapture { white_won } => {
                            eprintln!(
                                "terminal royalcapture winner {}",
                                if white_won { "white" } else { "black" }
                            );
                        }
                        TerminalState::Draw(reason) => {
                            eprintln!("terminal {}", reason);
                        }
                    }
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
                    eprintln!("search returned None for icn: {}", icn);
                    println!("bestmove none");
                }
            }));
            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                eprintln!("PANIC in search subprocess: {}", msg);
                println!("bestmove none");
            }
        }
        Some(Commands::CommitInfo) => {
            let is_dirty = BUILD_DIRTY.map(|d| d == "1").unwrap_or(false);
            let info = CommitInfo {
                commit: BUILD_COMMIT.unwrap_or("").to_string(),
                date: BUILD_DATE.unwrap_or("").to_string(),
                dirty: is_dirty,
            };
            println!("{}", serde_json::to_string(&info).unwrap());
        }
        None => {
            println!("Use --help for usage. SPRT CLI requires a subcommand.");
        }
    }
}
#[cfg(test)]
mod pentanomial_tests {
    use super::*;

    // Reference values from fastchess app/tests/sprt_test.cpp (logistic model).
    // Stats(ll, ld, wl, dd, wd, ww) → PentaCounts fields.
    #[test]
    fn logistic_pentanomial_matches_fastchess() {
        // "logistic pentanomial 1": Stats(223, 9863, 20279, 1000, 10037, 246), elo [0.5, 2.5] → -3.07
        let p1 = PentaCounts {
            ww: 246,
            wd: 10037,
            wl: 20279,
            dd: 1000,
            ld: 9863,
            ll: 223,
        };
        let llr1 = calculate_pentanomial_llr(&p1, 0.5, 2.5, SprtModel::Logistic);
        assert!((llr1 - (-3.07)).abs() < 0.02, "case 1 got {}", llr1);

        // "logistic pentanomial 2": Stats(871, 26175, 55003, 980, 26678, 821), elo [0, 2] → -4.98
        let p2 = PentaCounts {
            ww: 821,
            wd: 26678,
            wl: 55003,
            dd: 980,
            ld: 26175,
            ll: 871,
        };
        let llr2 = calculate_pentanomial_llr(&p2, 0.0, 2.0, SprtModel::Logistic);
        assert!((llr2 - (-4.98)).abs() < 0.02, "case 2 got {}", llr2);
    }

    #[test]
    fn normalized_pentanomial_matches_fastchess() {
        // "normalized pentanomial 1": Stats(365, 16618, 36029, 200, 16974, 390), elo [0, 2] → 2.25
        let p1 = PentaCounts {
            ww: 390,
            wd: 16974,
            wl: 36029,
            dd: 200,
            ld: 16618,
            ll: 365,
        };
        let llr1 = calculate_pentanomial_llr(&p1, 0.0, 2.0, SprtModel::Normalized);
        assert!((llr1 - 2.25).abs() < 0.02, "case 1 got {}", llr1);

        // "normalized pentanomial 2": Stats(127, 4883, 10311, 401, 5150, 104), elo [-1.75, 0.25] → 3.01
        let p2 = PentaCounts {
            ww: 104,
            wd: 5150,
            wl: 10311,
            dd: 401,
            ld: 4883,
            ll: 127,
        };
        let llr2 = calculate_pentanomial_llr(&p2, -1.75, 0.25, SprtModel::Normalized);
        assert!((llr2 - 3.01).abs() < 0.02, "case 2 got {}", llr2);

        // "normalized pentanomial 3": Stats(0, 0, 0, 0, 0, 5550), elo [0, 5] → 111.82
        let p3 = PentaCounts {
            ww: 5550,
            wd: 0,
            wl: 0,
            dd: 0,
            ld: 0,
            ll: 0,
        };
        let llr3 = calculate_pentanomial_llr(&p3, 0.0, 5.0, SprtModel::Normalized);
        assert!((llr3 - 111.82).abs() < 0.1, "case 3 got {}", llr3);
    }

    #[test]
    fn add_pair_buckets_correctly() {
        let mut p = PentaCounts::default();
        p.add_pair(GameResult::Win, GameResult::Win);
        p.add_pair(GameResult::Win, GameResult::Draw);
        p.add_pair(GameResult::Win, GameResult::Loss);
        p.add_pair(GameResult::Draw, GameResult::Draw);
        p.add_pair(GameResult::Loss, GameResult::Draw);
        p.add_pair(GameResult::Loss, GameResult::Loss);
        assert_eq!((p.ww, p.wd, p.wl, p.dd, p.ld, p.ll), (1, 1, 1, 1, 1, 1));
        assert_eq!(p.total_pairs(), 6);
    }

    #[test]
    fn fixed_time_does_not_consume_the_game_clock() {
        assert_eq!(account_move_time(10_000, 75, 0, true), (false, 10_000));
        assert_eq!(account_move_time(10_000, 20_000, 0, true), (false, 10_000));
    }

    #[test]
    fn cumulative_time_updates_and_flags_normally() {
        assert_eq!(account_move_time(1_000, 250, 50, false), (false, 800));
        assert_eq!(account_move_time(100, 101, 0, false), (true, 0));
    }
}
