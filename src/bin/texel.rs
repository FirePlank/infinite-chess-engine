//! Texel-style tuner for `evaluation/base.rs`, fitting every tunable eval
//! parameter to the data_gen self-play corpus by minimizing the logistic
//! negative log-likelihood of game results against the STATIC eval.
//!
//! World bounds (`moves::set_world_bounds`) are a process global, so extraction is
//! grouped by variant and the corpus must not mix variants with different bounds.

use apeiron::Variant;
use apeiron::board::PlayerColor;
use apeiron::evaluation::{EvalParamSpec, EvalParams, TUNABLE_EVAL_PARAM_SPECS, set_eval_params};
use apeiron::game::GameState;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const EVAL_BASE_RS_PATH: &str = "src/evaluation/base.rs";
/// Any |score| beyond this is a confirmed forced mate, not a real evaluation.
const MATE_FLOOR: i32 = apeiron::search::MATE_SCORE;
/// Fitted on this engine's own score scale (`puzzle_gen`'s WC_K=0.00188 → 1/K),
/// roughly half of Stockfish's 400: this engine's scores are steeper per centipawn.
const DEFAULT_K_SCALE: f64 = 531.9;
/// Positions materialized at once during extraction. Bounds peak memory to
/// roughly `CHUNK * 100KB` of live `GameState`s.
const EXTRACT_CHUNK: usize = 3_072;

static STOP: AtomicBool = AtomicBool::new(false);

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Texel-style tuner for evaluation/base.rs constants"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Tune eval parameters against a data_gen corpus.
    Run {
        /// data_gen JSONL corpus path.
        #[arg(long, default_value = "games/texel_corpus.jsonl")]
        corpus: String,
        /// Comma-separated parameter names to tune; omit to tune every tunable param.
        #[arg(long)]
        params: Option<String>,
        /// Output JSON path for tuned values + diagnostics.
        #[arg(long, default_value = "games/eval_params_tuned.json")]
        output: String,
        /// Ceiling on gradient-descent epochs per outer round. Training normally
        /// stops earlier, when held-out loss stops improving.
        #[arg(long, default_value_t = 20_000)]
        epochs: usize,
        /// Stop a round after this many epochs with no held-out improvement.
        #[arg(long, default_value_t = 600)]
        patience: usize,
        /// Hold out 1 in N GAMES for validation (never a fraction of positions:
        /// positions in one game share a result and would leak). 0 disables, which
        /// also disables early stopping.
        #[arg(long, default_value_t = 10)]
        val_every: usize,
        /// Outer rounds. Each re-extracts coefficients at the current weights,
        /// which is what corrects for the bilinear/threshold parameters whose
        /// local slope shifts as the weights move.
        #[arg(long, default_value_t = 3)]
        outer: usize,
        /// Adam learning rate, in centipawns per step.
        #[arg(long, default_value_t = 1.0)]
        lr: f64,
        /// Sigmoid K-scale: win_prob = 1 / (1 + exp(-score / k_scale)).
        #[arg(long, default_value_t = DEFAULT_K_SCALE)]
        k_scale: f64,
        /// Re-fit K to the data at the start of each outer round.
        #[arg(long, default_value_t = true)]
        fit_k: bool,
        /// Only train on positions data_gen marked `quiet: true`.
        #[arg(long, default_value_t = true)]
        quiet_only: bool,
        /// Cap on training positions; 0 = use the entire corpus (the default —
        /// sparse rows are ~200 bytes, so millions of positions are fine).
        #[arg(long, default_value_t = 0)]
        max_samples: usize,
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Patch tuned values from a Run's output JSON into base.rs's `DEFAULT_EVAL_*`.
    Apply {
        #[arg(long, default_value = "games/eval_params_tuned.json")]
        input: String,
    },
    /// Sweep one or more parameters and report the loss curve PER VARIANT.
    ///
    /// This is the high-power alternative to inferring a parameter's per-variant
    /// optimum from match results: an SPRT resolves a single variant to about
    /// +/-50 Elo at a few dozen games, which is the same size as the effects worth
    /// chasing, whereas each variant here is measured over tens of thousands of
    /// positions. Use it to find out whether a parameter genuinely wants different
    /// values in different variants before building any gating for it.
    Sweep {
        #[arg(long, default_value = "games/texel_corpus.jsonl")]
        corpus: String,
        /// Comma-separated parameter names to sweep, each independently with the
        /// others held at their current values.
        #[arg(long)]
        params: String,
        /// Sweep points per parameter, spread across its declared range.
        #[arg(long, default_value_t = 21)]
        points: usize,
        #[arg(long, default_value_t = DEFAULT_K_SCALE)]
        k_scale: f64,
        #[arg(long, default_value_t = true)]
        quiet_only: bool,
        /// Optional JSON output path for the raw curves.
        #[arg(long)]
        output: Option<String>,
    },
    /// Print every tunable parameter with its current default and search range.
    List,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\x1b[31m[texel] error:\x1b[0m {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    ctrlc::set_handler(|| STOP.store(true, Ordering::Relaxed))
        .map_err(|e| format!("failed to install Ctrl-C handler: {}", e))?;

    match Cli::parse().command {
        Commands::Run {
            corpus,
            params,
            output,
            epochs,
            patience,
            val_every,
            outer,
            lr,
            k_scale,
            fit_k,
            quiet_only,
            max_samples,
            verbose,
        } => {
            let cfg = RunCfg {
                corpus,
                params,
                output,
                epochs,
                patience,
                val_every,
                outer,
                lr,
                k_scale,
                fit_k,
                quiet_only,
                max_samples,
                verbose,
            };
            run_tuner(&cfg)
        }
        Commands::Apply { input } => apply_tuned(&input),
        Commands::Sweep {
            corpus,
            params,
            points,
            k_scale,
            quiet_only,
            output,
        } => run_sweep(&corpus, &params, points, k_scale, quiet_only, output.as_deref()),
        Commands::List => {
            for spec in TUNABLE_EVAL_PARAM_SPECS {
                println!(
                    "{:<44} default={:<6} range=[{}, {}]  {}",
                    spec.name, spec.default, spec.min, spec.max, spec.description
                );
            }
            println!("\n{} tunable parameters", TUNABLE_EVAL_PARAM_SPECS.len());
            Ok(())
        }
    }
}

struct RunCfg {
    corpus: String,
    params: Option<String>,
    output: String,
    epochs: usize,
    patience: usize,
    val_every: usize,
    outer: usize,
    lr: f64,
    k_scale: f64,
    fit_k: bool,
    quiet_only: bool,
    max_samples: usize,
    verbose: bool,
}

// ---------------------------------------------------------------------------
// corpus format (matches data_gen)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PositionRecord {
    ply: usize,
    score: i32,
    quiet: bool,
}

#[derive(Debug, Deserialize)]
struct GameRecord {
    variant: String,
    wdl: f32,
    start_icn: String,
    moves: Vec<String>,
    positions: Vec<PositionRecord>,
}

/// Every variant the engine knows, needed to map a corpus variant name back to
/// its world bounds.
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

fn canon(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn resolve_variant(name: &str) -> Option<Variant> {
    let want = canon(name);
    ALL_VARIANTS
        .iter()
        .find(|v| canon(v.to_str()) == want)
        .copied()
}

fn parse_move(mv: &str) -> Option<(i64, i64, i64, i64, Option<String>)> {
    let (coords, promo) = match mv.split_once('=') {
        Some((c, p)) => (c, Some(p.to_lowercase())),
        None => (mv, None),
    };
    let (from, to) = coords.split_once('>')?;
    let mut fp = from.split(',');
    let mut tp = to.split(',');
    let fx = fp.next()?.parse().ok()?;
    let fy = fp.next()?.parse().ok()?;
    let tx = tp.next()?.parse().ok()?;
    let ty = tp.next()?.parse().ok()?;
    Some((fx, fy, tx, ty, promo))
}

/// Replays a game once and returns the positions to train on, paired with the
/// game result from each position's side-to-move perspective.
fn replay_game(record: &GameRecord, quiet_only: bool) -> Vec<(GameState, f32)> {
    let mut out = Vec::new();
    let wanted: HashSet<usize> = record
        .positions
        .iter()
        .filter(|p| (!quiet_only || p.quiet) && p.score.abs() < MATE_FLOOR)
        .map(|p| p.ply)
        .collect();
    if wanted.is_empty() {
        return out;
    }

    let mut game = GameState::new();
    game.setup_position_from_icn(&record.start_icn);
    game.recompute_piece_counts();
    game.recompute_hash();

    for (ply, mv) in record.moves.iter().enumerate() {
        if wanted.contains(&ply) {
            let result = if game.turn == PlayerColor::White {
                record.wdl
            } else {
                1.0 - record.wdl
            };
            out.push((game.clone(), result));
        }
        let Some((fx, fy, tx, ty, promo)) = parse_move(mv) else {
            break;
        };
        game.make_move_coords(fx, fy, tx, ty, promo.as_deref());
    }
    out
}

// ---------------------------------------------------------------------------
// sparse coefficient dataset (CSR layout)
// ---------------------------------------------------------------------------

/// Sparse per-position eval derivatives, CSR-packed: one flat coefficient array
/// plus row offsets, avoiding a `Vec` header per position.
struct Dataset {
    /// Row offsets into `coeffs`, len = rows + 1.
    offsets: Vec<u32>,
    /// `(param index, d_eval / d_param)` pairs.
    coeffs: Vec<(u16, f32)>,
    /// Eval at the weights coefficients were extracted at, side-to-move relative.
    baseline: Vec<f32>,
    /// Game result from the side-to-move's perspective (0 / 0.5 / 1).
    result: Vec<f32>,
    /// Row indices held out for validation, and those trained on. The split is by
    /// GAME, never by position: positions from one game share a result label and
    /// are highly correlated, so splitting mid-game leaks the label into validation
    /// and makes held-out loss look better than it is.
    train_rows: Vec<u32>,
    val_rows: Vec<u32>,
}

impl Dataset {
    fn rows(&self) -> usize {
        self.baseline.len()
    }

    fn nonzeros(&self) -> usize {
        self.coeffs.len()
    }

    /// Predicted eval at `w`, given coefficients extracted at `w0`.
    #[inline]
    fn predict(&self, row: usize, w: &[f64], w0: &[f64]) -> f64 {
        let (lo, hi) = (self.offsets[row] as usize, self.offsets[row + 1] as usize);
        let mut pred = self.baseline[row] as f64;
        for &(idx, coeff) in &self.coeffs[lo..hi] {
            let i = idx as usize;
            pred += coeff as f64 * (w[i] - w0[i]);
        }
        pred
    }
}

#[inline]
fn static_eval(game: &GameState) -> i32 {
    #[cfg(feature = "nnue")]
    return apeiron::evaluation::evaluate(game, None);
    #[cfg(not(feature = "nnue"))]
    return apeiron::evaluation::evaluate(game);
}

/// Get/set an `EvalParams` field by name through its JSON encoding: 132
/// hand-written match arms would drift from the struct, this cannot.
fn get_field(params: &EvalParams, name: &str) -> i64 {
    serde_json::to_value(params)
        .ok()
        .and_then(|v| v.get(name).and_then(Value::as_i64))
        .unwrap_or(0)
}

fn set_field(params: &EvalParams, name: &str, value: i64) -> EvalParams {
    let mut v = serde_json::to_value(params).unwrap_or_else(|_| serde_json::json!({}));
    v[name] = serde_json::json!(value);
    serde_json::from_value(v).unwrap_or_else(|_| params.clone())
}

/// Probe step for a parameter's finite difference. The eval is integer-valued and
/// internally tapered by integer division, so a step of 1 can truncate a real
/// slope to zero; a wider step averages that out. Kept inside the parameter's own
/// range so the probe never evaluates an out-of-range configuration.
fn probe_delta(spec: &EvalParamSpec) -> i64 {
    let span = spec.max - spec.min;
    (span / 8).clamp(1, 16)
}

/// Extracts sparse coefficients for the whole corpus at weights `w0`.
///
/// Positions are materialized in chunks and dropped as soon as their row is
/// written, so peak memory is bounded by `EXTRACT_CHUNK` live `GameState`s rather
/// than by corpus size. Within a chunk, every parameter's probe is one parallel
/// pass over the chunk — the global `EVAL_PARAMS` is a single shared value, so
/// probes must be sequential over parameters and parallel over positions.
#[allow(clippy::too_many_arguments)]
fn extract_dataset(
    by_variant: &[(String, Vec<GameRecord>)],
    specs: &[&EvalParamSpec],
    base_params: &EvalParams,
    quiet_only: bool,
    max_samples: usize,
    val_every: usize,
    verbose: bool,
) -> Dataset {
    let mut data = Dataset {
        offsets: vec![0],
        coeffs: Vec::new(),
        baseline: Vec::new(),
        result: Vec::new(),
        train_rows: Vec::new(),
        val_rows: Vec::new(),
    };
    // Running game counter drives the by-game validation split.
    let mut game_seq = 0usize;

    let deltas: Vec<i64> = specs.iter().map(|s| probe_delta(s)).collect();
    let started = Instant::now();

    // EXTRACT_CHUNK bounds live POSITIONS, but chunking happens over games, so the
    // game-chunk has to be derived from how many positions a game actually yields
    // (~100 in this corpus). Chunking by games directly would hold thousands of
    // ~100KB GameStates at once, which is exactly the blow-up to avoid.
    let total_games: usize = by_variant.iter().map(|(_, g)| g.len()).sum();
    let total_positions: usize = by_variant
        .iter()
        .flat_map(|(_, g)| g.iter())
        .map(|g| g.positions.len())
        .sum();
    let per_game = (total_positions / total_games.max(1)).max(1);
    let chunk_games = (EXTRACT_CHUNK / per_game).max(1);
    if verbose {
        eprintln!(
            "[texel]   ~{} positions/game -> {} games/chunk (~{} live positions)",
            per_game,
            chunk_games,
            chunk_games * per_game
        );
    }

    'outer: for (variant_name, games) in by_variant {
        let Some(variant) = resolve_variant(variant_name) else {
            eprintln!("[texel] unknown variant '{}', skipping", variant_name);
            continue;
        };
        let b = variant.get_default_bounds();
        apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);

        for chunk in games.chunks(chunk_games) {
            if STOP.load(Ordering::Relaxed) {
                break 'outer;
            }
            // `GameState` holds a `RefCell`-backed spatial-index cache, so it is
            // `Send` but not `Sync`: shared refs can't cross threads, but the
            // exclusive `&mut` slices `par_iter_mut` hands out can.
            //
            // Replayed per game rather than flat-mapped, so each row still knows
            // which game it came from and the validation split can cut on game
            // boundaries.
            let per_game: Vec<Vec<(GameState, f32)>> = chunk
                .par_iter()
                .map(|g| replay_game(g, quiet_only))
                .collect();
            let mut positions: Vec<(GameState, f32)> = Vec::new();
            let mut row_is_val: Vec<bool> = Vec::new();
            for game_rows in per_game {
                let is_val = val_every > 0 && game_seq % val_every == 0;
                game_seq += 1;
                row_is_val.extend(std::iter::repeat_n(is_val, game_rows.len()));
                positions.extend(game_rows);
            }
            if positions.is_empty() {
                continue;
            }

            // Baseline pass at w0.
            set_eval_params(base_params.clone());
            let baseline: Vec<i32> = positions
                .par_iter_mut()
                .map(|(g, _)| static_eval(g))
                .collect();

            // One probe pass per parameter, accumulating this chunk's rows.
            let mut rows: Vec<Vec<(u16, f32)>> = vec![Vec::new(); positions.len()];
            for (pi, spec) in specs.iter().enumerate() {
                let delta = deltas[pi];
                let current = get_field(base_params, spec.name);
                // Probe upward unless that would leave the range.
                let (probe, signed) = if current + delta <= spec.max {
                    (current + delta, delta)
                } else {
                    (current - delta, -delta)
                };
                if probe == current {
                    continue;
                }
                set_eval_params(set_field(base_params, spec.name, probe));
                let probed: Vec<i32> = positions
                    .par_iter_mut()
                    .map(|(g, _)| static_eval(g))
                    .collect();
                for (ri, row) in rows.iter_mut().enumerate() {
                    let diff = probed[ri] - baseline[ri];
                    if diff != 0 {
                        row.push((pi as u16, diff as f32 / signed as f32));
                    }
                }
            }

            for (ri, (_, result)) in positions.iter().enumerate() {
                let row_idx = data.baseline.len() as u32;
                data.coeffs.extend_from_slice(&rows[ri]);
                data.offsets.push(data.coeffs.len() as u32);
                data.baseline.push(baseline[ri] as f32);
                data.result.push(*result);
                if row_is_val[ri] {
                    data.val_rows.push(row_idx);
                } else {
                    data.train_rows.push(row_idx);
                }
            }

            if verbose {
                eprintln!(
                    "[texel]   extracted {} rows ({} nnz) in {:.1}s",
                    data.rows(),
                    data.nonzeros(),
                    started.elapsed().as_secs_f64()
                );
            }
            if max_samples > 0 && data.rows() >= max_samples {
                println!(
                    "[texel] hit --max-samples {}, stopping extraction",
                    max_samples
                );
                break 'outer;
            }
        }
    }

    // Restore the caller's weights; probing left the global on the last probe value.
    set_eval_params(base_params.clone());
    data
}

// ---------------------------------------------------------------------------
// loss + Adam
// ---------------------------------------------------------------------------

/// Mean NLL over the given rows.
fn loss_on(data: &Dataset, rows: &[u32], w: &[f64], w0: &[f64], k: f64) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = rows
        .par_iter()
        .map(|&r| {
            let r = r as usize;
            let pred = data.predict(r, w, w0);
            let p = (1.0 / (1.0 + (-pred / k).exp())).clamp(1e-12, 1.0 - 1e-12);
            let res = data.result[r] as f64;
            -(res * p.ln() + (1.0 - res) * (1.0 - p).ln())
        })
        .sum();
    sum / rows.len() as f64
}

/// Golden-section fit of K on the training rows. K only rescales the sigmoid, so
/// its optimum is one-dimensional and cheap to locate exactly.
fn fit_k_scale(data: &Dataset, w: &[f64], w0: &[f64]) -> f64 {
    let (mut lo, mut hi) = (80.0f64, 1200.0f64);
    let phi = 0.618_033_988_75;
    let mut c = hi - (hi - lo) * phi;
    let mut d = lo + (hi - lo) * phi;
    let mut fc = loss_on(data, &data.train_rows, w, w0, c);
    let mut fd = loss_on(data, &data.train_rows, w, w0, d);
    for _ in 0..24 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - (hi - lo) * phi;
            fc = loss_on(data, &data.train_rows, w, w0, c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + (hi - lo) * phi;
            fd = loss_on(data, &data.train_rows, w, w0, d);
        }
    }
    (lo + hi) / 2.0
}

/// What a training call converged to.
struct TrainOutcome {
    train_loss: f64,
    val_loss: f64,
    epochs_run: usize,
    stopped_early: bool,
}

/// Full-batch Adam on the sparse coefficients. No `evaluate()` calls here, which
/// is what makes many thousands of epochs affordable.
///
/// `epochs` is a CEILING, not a target: training stops when held-out loss stops
/// improving, and the weights returned are the best-validation ones rather than
/// whatever the last epoch happened to leave behind. A fixed epoch count either
/// stops short of convergence or keeps fitting noise; neither is visible without
/// a held-out split.
#[allow(clippy::too_many_arguments)]
fn train(
    data: &Dataset,
    specs: &[&EvalParamSpec],
    w: &mut [f64],
    w0: &[f64],
    k: f64,
    epochs: usize,
    lr: f64,
    patience: usize,
    verbose: bool,
) -> TrainOutcome {
    let n = specs.len();
    let train_rows = &data.train_rows;
    let has_val = !data.val_rows.is_empty();
    let mut m = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    /// Below this the steps are far smaller than the 1cp quantization of the
    /// final integer constants, so further epochs cannot change the result.
    const LR_MIN: f64 = 0.01;

    let mut lr = lr;
    let mut train_loss = f64::INFINITY;
    let mut best_val = f64::INFINITY;
    let mut best_w = w.to_vec();
    let mut since_improved = 0usize;
    let mut epochs_run = 0usize;
    let mut stopped_early = false;
    // Halve the rate well before giving up entirely: a stalled validation loss
    // usually means the step is too coarse to resolve the optimum, not that the
    // optimum is reached (ReduceLROnPlateau, then early stop).
    let plateau = (patience / 3).max(1);

    for epoch in 1..=epochs {
        if STOP.load(Ordering::Relaxed) {
            break;
        }
        epochs_run = epoch;

        // Gradient of mean NLL w.r.t. each weight: sum over rows of
        // (p - result) * coeff / k, the sigmoid's clean derivative (the
        // sigma*(1-sigma) factor cancels for cross-entropy).
        let (grad, total) = train_rows
            .par_iter()
            .fold(
                || (vec![0.0f64; n], 0.0f64),
                |(mut g, mut acc), &r| {
                    let r = r as usize;
                    let pred = data.predict(r, w, w0);
                    let p = (1.0 / (1.0 + (-pred / k).exp())).clamp(1e-12, 1.0 - 1e-12);
                    let res = data.result[r] as f64;
                    acc += -(res * p.ln() + (1.0 - res) * (1.0 - p).ln());
                    let dl = (p - res) / k;
                    let (lo, hi) = (data.offsets[r] as usize, data.offsets[r + 1] as usize);
                    for &(idx, coeff) in &data.coeffs[lo..hi] {
                        g[idx as usize] += dl * coeff as f64;
                    }
                    (g, acc)
                },
            )
            .reduce(
                || (vec![0.0f64; n], 0.0f64),
                |(mut ga, aa), (gb, ab)| {
                    for (a, b) in ga.iter_mut().zip(gb.iter()) {
                        *a += *b;
                    }
                    (ga, aa + ab)
                },
            );

        train_loss = total / train_rows.len() as f64;
        let scale = 1.0 / train_rows.len() as f64;
        for i in 0..n {
            let g = grad[i] * scale;
            m[i] = B1 * m[i] + (1.0 - B1) * g;
            v[i] = B2 * v[i] + (1.0 - B2) * g * g;
            let mhat = m[i] / (1.0 - B1.powi(epoch as i32));
            let vhat = v[i] / (1.0 - B2.powi(epoch as i32));
            w[i] -= lr * mhat / (vhat.sqrt() + EPS);
            w[i] = w[i].clamp(specs[i].min as f64, specs[i].max as f64);
        }

        // Held-out check. Weights are quantized to integers before scoring, since
        // that is what actually ships — a fractional improvement that rounds away
        // is not an improvement.
        let probe: Vec<f64> = w.iter().map(|x| x.round()).collect();
        let val = if has_val {
            loss_on(data, &data.val_rows, &probe, w0, k)
        } else {
            train_loss
        };

        if val + 1e-9 < best_val {
            best_val = val;
            best_w.copy_from_slice(&probe);
            since_improved = 0;
        } else {
            since_improved += 1;
            if since_improved % plateau == 0 && lr > LR_MIN {
                lr *= 0.5;
                if verbose {
                    eprintln!("[texel]   epoch {:>6}  val plateau -> lr {:.4}", epoch, lr);
                }
            }
            if since_improved >= patience || lr <= LR_MIN {
                stopped_early = true;
                if verbose {
                    eprintln!(
                        "[texel]   epoch {:>6}  converged (no val gain in {} epochs)",
                        epoch, since_improved
                    );
                }
                break;
            }
        }

        if verbose && (epoch % 250 == 0 || epoch == 1) {
            eprintln!(
                "[texel]   epoch {:>6}  train {:.6}  val {:.6}  lr {:.4}",
                epoch, train_loss, val, lr
            );
        }
    }

    // Ship the best-validation weights, not the last epoch's.
    w.copy_from_slice(&best_w);
    TrainOutcome {
        train_loss,
        val_loss: best_val,
        epochs_run,
        stopped_early,
    }
}

#[derive(Serialize)]
struct TunerOutput {
    params: BTreeMap<String, i64>,
    changed: BTreeMap<String, String>,
    zero_signal: Vec<String>,
    baseline_loss: f64,
    final_loss: f64,
    /// Held-out loss, the number that actually indicates generalization.
    baseline_val_loss: f64,
    final_val_loss: f64,
    samples: usize,
    nonzeros: usize,
    variants: Vec<String>,
    epochs: usize,
    outer: usize,
    k_scale: f64,
    timestamp: u64,
}

fn run_tuner(cfg: &RunCfg) -> Result<(), String> {
    // ---- load + group the corpus (games only; positions are materialized later) ----
    let file =
        File::open(&cfg.corpus).map_err(|e| format!("failed to open {}: {}", cfg.corpus, e))?;
    let mut grouped: HashMap<String, Vec<GameRecord>> = HashMap::new();
    let mut total_games = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<GameRecord>(trimmed) {
            Ok(r) => {
                total_games += 1;
                grouped.entry(r.variant.clone()).or_default().push(r);
            }
            Err(e) => eprintln!("[texel] skipping malformed line: {}", e),
        }
    }
    if grouped.is_empty() {
        return Err(format!("no usable games in {}", cfg.corpus));
    }

    // evaluate() reads the process-global world bounds, so a corpus spanning
    // several bounds settings cannot be tuned in one run.
    let mut bounds_groups: HashMap<(i64, i64, i64, i64), Vec<String>> = HashMap::new();
    for name in grouped.keys() {
        if let Some(v) = resolve_variant(name) {
            bounds_groups
                .entry(v.get_default_bounds())
                .or_default()
                .push(name.clone());
        }
    }
    if bounds_groups.len() > 1 {
        let groups: Vec<String> = bounds_groups.values().map(|n| n.join("+")).collect();
        return Err(format!(
            "corpus mixes variants with different world bounds ({}); tune one bounds-group per run",
            groups.join("  VS  ")
        ));
    }

    // Sorted, not HashMap order: row order decides float summation order, so leaving
    // it to a randomly-seeded HashMap makes the reported loss wobble between runs.
    let mut by_variant: Vec<(String, Vec<GameRecord>)> = grouped.into_iter().collect();
    by_variant.sort_by(|a, b| a.0.cmp(&b.0));
    let variants: Vec<String> = by_variant.iter().map(|(n, _)| n.clone()).collect();
    println!(
        "[texel] {} games across {} variant(s): {}",
        total_games,
        variants.len(),
        variants.join(", ")
    );

    // ---- parameter selection ----
    let specs: Vec<&EvalParamSpec> = match &cfg.params {
        Some(sel) => {
            let wanted: HashSet<String> = sel.split(',').map(|s| s.trim().to_lowercase()).collect();
            TUNABLE_EVAL_PARAM_SPECS
                .iter()
                .filter(|s| wanted.contains(&s.name.to_lowercase()))
                .collect()
        }
        None => TUNABLE_EVAL_PARAM_SPECS.iter().collect(),
    };
    if specs.is_empty() {
        return Err("no matching tunable parameters selected".to_string());
    }
    println!(
        "[texel] tuning {} parameters, {} outer round(s), <={} epochs each \
         (early stop after {} stale), lr {}, val 1-in-{} games",
        specs.len(),
        cfg.outer,
        cfg.epochs,
        cfg.patience,
        cfg.lr,
        cfg.val_every
    );

    let mut params = EvalParams::default();
    let mut w: Vec<f64> = specs
        .iter()
        .map(|s| get_field(&params, s.name) as f64)
        .collect();
    let mut k = cfg.k_scale;
    let mut baseline_loss = f64::NAN;
    let mut baseline_val = f64::NAN;
    let mut final_loss = f64::NAN;
    let mut final_val = f64::NAN;
    let mut zero_signal: Vec<String> = Vec::new();
    let mut last_rows = 0usize;
    let mut last_nnz = 0usize;

    for round in 1..=cfg.outer {
        if STOP.load(Ordering::Relaxed) {
            break;
        }
        println!("\n[texel] ===== outer round {}/{} =====", round, cfg.outer);

        // Coefficients are only valid near the weights they were taken at, so each
        // round re-extracts at the current weights.
        for (i, spec) in specs.iter().enumerate() {
            params = set_field(&params, spec.name, w[i].round() as i64);
        }
        let t0 = Instant::now();
        let data = extract_dataset(
            &by_variant,
            &specs,
            &params,
            cfg.quiet_only,
            cfg.max_samples,
            cfg.val_every,
            cfg.verbose,
        );
        if data.train_rows.is_empty() {
            return Err("no training positions extracted (check --quiet-only / corpus)".into());
        }
        last_rows = data.rows();
        last_nnz = data.nonzeros();
        println!(
            "[texel] extracted {} positions ({} train / {} val), {} nonzero coefficients \
             ({:.1} avg/pos) in {:.1}s",
            data.rows(),
            data.train_rows.len(),
            data.val_rows.len(),
            data.nonzeros(),
            data.nonzeros() as f64 / data.rows() as f64,
            t0.elapsed().as_secs_f64()
        );

        // Parameters no position responds to cannot be moved by a gradient; report
        // them rather than pretending they were tuned.
        let mut seen = vec![false; specs.len()];
        for &(idx, _) in &data.coeffs {
            seen[idx as usize] = true;
        }
        zero_signal = specs
            .iter()
            .zip(seen.iter())
            .filter(|&(_, &s)| !s)
            .map(|(spec, _)| spec.name.to_string())
            .collect();

        let w0 = w.clone();
        if cfg.fit_k {
            k = fit_k_scale(&data, &w, &w0);
            println!("[texel] fitted K = {:.1}", k);
        }
        let start_loss = loss_on(&data, &data.train_rows, &w, &w0, k);
        let start_val = loss_on(&data, &data.val_rows, &w, &w0, k);
        if round == 1 {
            baseline_loss = start_loss;
            baseline_val = start_val;
        }
        println!(
            "[texel] round start: train {:.6}  val {:.6}",
            start_loss, start_val
        );

        let t1 = Instant::now();
        let outcome = train(
            &data,
            &specs,
            &mut w,
            &w0,
            k,
            cfg.epochs,
            cfg.lr,
            cfg.patience,
            cfg.verbose,
        );
        final_loss = outcome.train_loss;
        final_val = outcome.val_loss;
        println!(
            "[texel] round {} done: train {:.6} -> {:.6}, val {:.6} -> {:.6}  ({} epochs{}, {:.1}s)",
            round,
            start_loss,
            outcome.train_loss,
            start_val,
            outcome.val_loss,
            outcome.epochs_run,
            if outcome.stopped_early {
                ", converged"
            } else {
                ", hit ceiling"
            },
            t1.elapsed().as_secs_f64()
        );
    }

    // ---- results ----
    let defaults = EvalParams::default();
    let mut out_params = BTreeMap::new();
    let mut changed = BTreeMap::new();
    for (i, spec) in specs.iter().enumerate() {
        let tuned = (w[i].round() as i64).clamp(spec.min, spec.max);
        out_params.insert(spec.name.to_string(), tuned);
        let old = get_field(&defaults, spec.name);
        if tuned != old {
            changed.insert(spec.name.to_string(), format!("{} -> {}", old, tuned));
        }
    }
    // Every tunable param is emitted, so `apply` writes a self-consistent set even
    // when the run tuned a subset.
    for spec in TUNABLE_EVAL_PARAM_SPECS {
        out_params
            .entry(spec.name.to_string())
            .or_insert_with(|| get_field(&defaults, spec.name));
    }

    println!(
        "\n[texel] {} of {} parameters changed",
        changed.len(),
        specs.len()
    );
    for (name, delta) in &changed {
        println!("  {:<44} {}", name, delta);
    }
    if !zero_signal.is_empty() {
        println!(
            "\n[texel] {} parameter(s) had NO gradient signal in this corpus and keep their \
             defaults (no position's eval responded to them — either the term never fires in \
             these variants, or it is a threshold whose step no position straddles):",
            zero_signal.len()
        );
        println!("  {}", zero_signal.join(", "));
    }

    if let Some(parent) = std::path::Path::new(&cfg.output).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let result = TunerOutput {
        params: out_params,
        changed,
        zero_signal,
        baseline_loss,
        final_loss,
        baseline_val_loss: baseline_val,
        final_val_loss: final_val,
        samples: last_rows,
        nonzeros: last_nnz,
        variants,
        epochs: cfg.epochs,
        outer: cfg.outer,
        k_scale: k,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    let mut f = File::create(&cfg.output).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;

    println!(
        "\n\x1b[32m[texel] done: train {:.6} -> {:.6}, held-out {:.6} -> {:.6}, wrote {}\x1b[0m",
        baseline_loss, final_loss, baseline_val, final_val, cfg.output
    );
    if final_val.is_finite() && baseline_val.is_finite() && final_val >= baseline_val {
        println!(
            "\x1b[33m[texel] WARNING: held-out loss did not improve — this fit does not \
             generalize, do not ship it.\x1b[0m"
        );
    }
    println!(
        "[texel] loss is a proxy: eval-term changes in this engine are SPRT-fragile, so \
         confirm with an SPRT before trusting these values."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// per-variant sweep
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct SweepCurve {
    param: String,
    values: Vec<i64>,
    /// variant -> loss at each swept value, aligned with `values`.
    per_variant: BTreeMap<String, Vec<f64>>,
    /// variant -> position count the curve was measured over.
    counts: BTreeMap<String, usize>,
    default: i64,
}

/// Measures, for each swept value of each parameter, the mean NLL PER VARIANT.
///
/// Positions are materialized once per chunk and every candidate value is scored
/// against that same chunk, so the cost is one eval per (position, value) rather
/// than a rebuild per value.
fn run_sweep(
    corpus: &str,
    params: &str,
    points: usize,
    k_scale: f64,
    quiet_only: bool,
    output: Option<&str>,
) -> Result<(), String> {
    let wanted: Vec<&EvalParamSpec> = {
        let names: Vec<String> = params.split(',').map(|s| s.trim().to_lowercase()).collect();
        let mut v = Vec::new();
        for n in &names {
            match TUNABLE_EVAL_PARAM_SPECS.iter().find(|s| s.name.to_lowercase() == *n) {
                Some(s) => v.push(s),
                None => return Err(format!("unknown parameter '{}'", n)),
            }
        }
        v
    };

    // Load and group games by variant.
    let file = File::open(corpus).map_err(|e| format!("failed to open {}: {}", corpus, e))?;
    let mut grouped: HashMap<String, Vec<GameRecord>> = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<GameRecord>(line.trim()) {
            grouped.entry(r.variant.clone()).or_default().push(r);
        }
    }
    let mut by_variant: Vec<(String, Vec<GameRecord>)> = grouped.into_iter().collect();
    by_variant.sort_by(|a, b| a.0.cmp(&b.0));

    let base = EvalParams::default();
    let mut curves = Vec::new();

    for spec in &wanted {
        if STOP.load(Ordering::Relaxed) {
            break;
        }
        // Candidate values spread across the parameter's declared range, with its
        // current value forced in so the baseline is always on the curve.
        let mut values: Vec<i64> = (0..points.max(2))
            .map(|i| {
                spec.min + (spec.max - spec.min) * i as i64 / (points.max(2) - 1) as i64
            })
            .collect();
        let current = get_field(&base, spec.name);
        if !values.contains(&current) {
            values.push(current);
        }
        values.sort_unstable();
        values.dedup();

        let mut sums: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let started = Instant::now();

        for (variant_name, games) in &by_variant {
            let Some(variant) = resolve_variant(variant_name) else { continue };
            let b = variant.get_default_bounds();
            apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);

            let acc = sums.entry(variant_name.clone()).or_insert_with(|| vec![0.0; values.len()]);
            let cnt = counts.entry(variant_name.clone()).or_insert(0);

            for chunk in games.chunks(24) {
                if STOP.load(Ordering::Relaxed) {
                    break;
                }
                let mut positions: Vec<(GameState, f32)> = chunk
                    .par_iter()
                    .flat_map(|g| replay_game(g, quiet_only))
                    .collect();
                if positions.is_empty() {
                    continue;
                }
                *cnt += positions.len();

                for (vi, &val) in values.iter().enumerate() {
                    set_eval_params(set_field(&base, spec.name, val));
                    let s: f64 = positions
                        .par_iter_mut()
                        .map(|(g, res)| {
                            let e = static_eval(g) as f64;
                            let p = (1.0 / (1.0 + (-e / k_scale).exp())).clamp(1e-12, 1.0 - 1e-12);
                            let r = *res as f64;
                            -(r * p.ln() + (1.0 - r) * (1.0 - p).ln())
                        })
                        .sum();
                    acc[vi] += s;
                }
            }
        }
        set_eval_params(base.clone());

        // Turn the accumulated sums into mean loss.
        let mut per_variant: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (v, s) in &sums {
            let n = counts[v].max(1) as f64;
            per_variant.insert(v.clone(), s.iter().map(|x| x / n).collect());
        }

        // Report: each variant's own optimum, and how far it sits from the current
        // global value. A parameter whose optima cluster wants one global value; one
        // whose optima split across variants is a gating candidate.
        println!("\n=== {} (current {}, range [{}, {}]) ===",
                 spec.name, current, spec.min, spec.max);
        println!("{:<22} {:>8} {:>10} {:>12} {:>12}", "variant", "best", "positions", "loss@best", "loss@current");
        let cur_idx = values.iter().position(|&v| v == current).unwrap_or(0);
        let mut best_vals = Vec::new();
        let mut rows: Vec<(String, i64, usize, f64, f64)> = Vec::new();
        for (v, losses) in &per_variant {
            let (bi, _) = losses
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            rows.push((v.clone(), values[bi], counts[v], losses[bi], losses[cur_idx]));
            best_vals.push(values[bi]);
        }
        rows.sort_by_key(|r| r.1);
        for (v, best, n, lb, lc) in &rows {
            let gain = lc - lb;
            println!(
                "{:<22} {:>8} {:>10} {:>12.6} {:>12.6}  {}",
                v, best, n, lb, lc,
                if gain > 0.0005 { format!("(-{:.4} available)", gain) } else { String::new() }
            );
        }
        best_vals.sort_unstable();
        let spread = best_vals.last().unwrap_or(&0) - best_vals.first().unwrap_or(&0);
        println!(
            "per-variant optima span {} .. {} (spread {}) — {}",
            best_vals.first().unwrap_or(&0),
            best_vals.last().unwrap_or(&0),
            spread,
            if spread * 4 > (spec.max - spec.min) {
                "SPLIT: variants disagree, a single global value cannot serve them"
            } else {
                "clustered: one global value is fine"
            }
        );
        println!("  measured in {:.1}s", started.elapsed().as_secs_f64());

        curves.push(SweepCurve {
            param: spec.name.to_string(),
            values,
            per_variant,
            counts,
            default: current,
        });
    }

    if let Some(path) = output {
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(path, serde_json::to_string_pretty(&curves).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        println!("\n[texel] wrote curves to {}", path);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

fn apply_tuned(input: &str) -> Result<(), String> {
    let raw = fs::read_to_string(input).map_err(|e| format!("failed to read {}: {}", input, e))?;
    let root: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let obj = root
        .get("params")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{} has no \"params\" object", input))?;

    let base_text = fs::read_to_string(EVAL_BASE_RS_PATH)
        .map_err(|e| format!("failed to read {}: {}", EVAL_BASE_RS_PATH, e))?;

    let updates: HashMap<String, i64> = obj
        .iter()
        .filter_map(|(n, v)| {
            v.as_i64()
                .map(|x| (format!("DEFAULT_EVAL_{}", n.to_uppercase()), x))
        })
        .collect();
    if updates.is_empty() {
        return Err(format!("no numeric parameters found in {}", input));
    }

    // base.rs is CRLF and `.lines()` strips both styles, so the ending has to be
    // restored explicitly or every line reads as changed.
    let eol = if base_text.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut applied = 0;
    let mut out = String::with_capacity(base_text.len());
    for line in base_text.lines() {
        let mut replaced = None;
        if let Some(rest) = line.trim().strip_prefix("pub const ")
            && let Some((name, _)) = rest.split_once(':')
            && let Some(&value) = updates.get(name.trim())
            && let Some(eq) = line.find('=')
            && let Some(semi) = line.find(';')
        {
            replaced = Some(format!(
                "{}= {}{}",
                &line[..=eq].trim_end_matches('='),
                value,
                &line[semi..]
            ));
            applied += 1;
        }
        out.push_str(&replaced.unwrap_or_else(|| line.to_string()));
        out.push_str(eol);
    }

    fs::write(EVAL_BASE_RS_PATH, out)
        .map_err(|e| format!("failed to write {}: {}", EVAL_BASE_RS_PATH, e))?;
    println!(
        "\x1b[32m[texel] applied {} constant(s) from {} to {}\x1b[0m",
        applied, input, EVAL_BASE_RS_PATH
    );
    Ok(())
}
