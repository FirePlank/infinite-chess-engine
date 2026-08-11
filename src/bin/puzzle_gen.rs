//! Puzzle generator: mines the self-play corpus for tactical puzzles.
//!
//! Five stages, cheapest filter first. Stage 0 uses only the `[%eval]`/`[%mate]`
//! annotations the SPRT harness already wrote, so 99.8% of plies are rejected
//! before the engine is ever started.
//!
//!   0 scan   - annotation trace -> candidate plies (no engine)
//!   1 replay - reconstruct only those positions, board filters + hash dedup
//!   2 verify - shallow MultiPV: is there a single clearly-best winning move?
//!   3 cook   - deep walk building the forced line, only-move checked every winner ply
//!   4 rate   - difficulty features -> rank-normalised rating, plus theme tagging
//!
//! World bounds are a process-global (`moves::set_world_bounds`), so variants are
//! processed one at a time and parallelism lives inside a variant, never across.

use apeiron::Variant;
use apeiron::board::{Coordinate, PieceType, PlayerColor};
use apeiron::evaluation::get_piece_value_base;
use apeiron::game::{GameState, WinCondition};
use apeiron::moves::{Move, MoveGenContext, MoveList};
use apeiron::search;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

// ---------------------------------------------------------------------------
// annotation scale
// ---------------------------------------------------------------------------

/// Mate encoding for annotation-space scores: `|s| = ANN_MATE - dist`.
const ANN_MATE: i32 = 1_000_000;
/// Anything above this in annotation space is a mate, not a centipawn score.
const ANN_MATE_FLOOR: i32 = ANN_MATE - 1_000;

#[inline]
fn ann_is_win(s: i32) -> bool {
    s >= ANN_MATE_FLOOR
}

#[inline]
fn ann_mate_dist(s: i32) -> i32 {
    ANN_MATE - s.abs()
}

/// Fitted on this corpus (62k self-play games, known results, 2.4M sampled plies)
/// rather than borrowed from Stockfish, whose 0.003682 is roughly twice as steep:
/// on this engine's scale +400cp is a 0.36 edge, not 0.63.
const WC_K: f64 = 0.001_880;

fn win_chances(score: i32) -> f64 {
    if search::is_win(score) {
        return 1.0;
    }
    if search::is_loss(score) {
        return -1.0;
    }
    2.0 / (1.0 + (-WC_K * score as f64).exp()) - 1.0
}

// ---------------------------------------------------------------------------
// tunables
// ---------------------------------------------------------------------------

/// A "turning point" needs this many prior plies below `TURN_QUIET_CEIL`. One-ply
/// swing tests misfire here: 0.15s/move evals oscillate by several hundred cp.
const QUIET_WINDOW: usize = 6;
const TURN_QUIET_CEIL: i32 = 300;
const TURN_WIN_FLOOR: i32 = 400;
const MATE_MAX_DIST: i32 = 10;
/// Below this the position is a trivial mop-up, not a puzzle.
const MIN_PIECES: u32 = 6;
/// Winner-perspective material cap. Lichess drops anything where the winner is
/// already ahead, as a proxy for "the win is not already in hand". The only-move
/// rule tests that directly and far better, so this only has to cut the hopeless
/// conversions where a search would be wasted.
const MAX_LEAD: i32 = 3000;
/// Candidates from one game must sit this far apart, or they are near-copies.
const MIN_PLY_GAP: usize = 4;
/// A move wins past this on this engine's scale; at or under `HELD_CP` the win is
/// gone. A puzzle has to *cross* that boundary: +5 -> 0 is a puzzle, +20 -> +10 is
/// not, because every move there still wins. A win-chance gap cannot express this —
/// it passes +2000 vs +900, where missing the "solution" costs nothing real.
const WON_CP: i32 = 500;
/// +300 is a ~64% expected score on the fitted curve — clearly better, not won —
/// so an alternative there still means the win was thrown away.
const HELD_CP: i32 = 300;
/// The same crossing read from the defending side: one move keeps the game
/// playable, every other move loses it outright.
const HOLD_FLOOR: i32 = -200;
const LOST_CEIL: i32 = -700;
/// Source C: the mover was no worse than this, then dropped below `DEF_LOST_CP`.
const DEF_HOLD_CP: i32 = 400;
const DEF_LOST_CP: i32 = 600;
/// Screen threshold, a deliberately slack version of `HELD_CP` so a shallow
/// search's noise cannot throw away a genuine puzzle.
const SCREEN_SECOND_CEIL: i32 = 1200;
/// How many full moves the puzzle start may be rewound to reach the beginning of
/// an already-forced sequence.
const MAX_REWIND: usize = 6;

/// Variants worth building a puzzle set from: everything live on the public site,
/// plus Scattered_Leapers for fairy-piece coverage. Custom test variants are out.
const ALLOWED_VARIANTS: &[Variant] = &[
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
    Variant::ScatteredLeapers,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// The game turned here: quiet for `QUIET_WINDOW` plies, then decisive.
    TurningPoint,
    /// A forced mate that is not the previous ply's mate counted down one.
    MateShot,
    /// The mover was holding and then lost the game outright, so the position
    /// before their move asks for the only defence.
    MissedSave,
}

impl Source {
    fn as_str(self) -> &'static str {
        match self {
            Source::TurningPoint => "turning_point",
            Source::MateShot => "mate_shot",
            Source::MissedSave => "missed_save",
        }
    }
    fn idx(self) -> usize {
        match self {
            Source::TurningPoint => 0,
            Source::MateShot => 1,
            Source::MissedSave => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Stage-0 detector thresholds. These decide only which positions get *looked at*;
/// every candidate still faces the same validity rule, so widening them buys more
/// puzzles at the cost of CPU without weakening a single one.
#[derive(Clone, Copy)]
struct Scan {
    quiet_window: usize,
    turn_quiet_ceil: i32,
    turn_win_floor: i32,
    mate_max_dist: i32,
    def_hold_cp: i32,
    def_lost_cp: i32,
    min_ply_gap: usize,
}

impl Scan {
    /// Widens only what adds *distinct* positions. The eval thresholds are left
    /// alone on purpose: dropping them measured 5x worse acceptance (turning
    /// points went 4.2% -> 0.2%), so relaxing those costs CPU and yields fewer
    /// puzzles, not more.
    fn wide(self) -> Self {
        Self {
            mate_max_dist: 12,
            min_ply_gap: 2,
            ..self
        }
    }
}

impl Default for Scan {
    fn default() -> Self {
        Self {
            quiet_window: QUIET_WINDOW,
            turn_quiet_ceil: TURN_QUIET_CEIL,
            turn_win_floor: TURN_WIN_FLOOR,
            mate_max_dist: MATE_MAX_DIST,
            def_hold_cp: DEF_HOLD_CP,
            def_lost_cp: DEF_LOST_CP,
            min_ply_gap: MIN_PLY_GAP,
        }
    }
}

struct Cfg {
    scan: Scan,
    corpus: Vec<PathBuf>,
    out: PathBuf,
    skip: FxHashSet<String>,
    per_variant: usize,
    per_game: usize,
    screen_depth: usize,
    verify_depth: usize,
    cook_depth: usize,
    defend_depth: usize,
    rate_depth: usize,
    cap_ms: u128,
    budget_ms: u64,
    defend_pv: usize,
    defence_window: f64,
    recook: bool,
    max_plies: usize,
    hash_mb: usize,
    threads: usize,
    dry_run: bool,
    rate_only: bool,
    refresh: bool,
    deep_verify: bool,
    deep_depth: usize,
    deep_cap_ms: u128,
    fresh: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        let mut skip = FxHashSet::default();
        skip.insert("Chess".to_string());
        Self {
            scan: Scan::default(),
            corpus: Vec::new(),
            out: PathBuf::from("puzzles.csv"),
            skip,
            per_variant: 4_000,
            per_game: 8,
            screen_depth: 7,
            verify_depth: 11,
            cook_depth: 13,
            defend_depth: 10,
            rate_depth: 13,
            cap_ms: 1_500,
            budget_ms: 20_000,
            defend_pv: 4,
            defence_window: 0.10,
            recook: false,
            max_plies: 15, // a mate in 8, the deepest the scanner accepts
            hash_mb: 32,
            threads: 0,
            dry_run: false,
            rate_only: false,
            refresh: false,
            deep_verify: false,
            deep_depth: 20,
            deep_cap_ms: 20_000,
            fresh: false,
        }
    }
}

fn parse_args() -> Cfg {
    let mut cfg = Cfg::default();
    let mut skip_overridden = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_default();
        match a.as_str() {
            "--corpus" => cfg.corpus.push(PathBuf::from(val())),
            "--out" => cfg.out = PathBuf::from(val()),
            "--skip-variant" => {
                if !skip_overridden {
                    cfg.skip.clear();
                    skip_overridden = true;
                }
                cfg.skip.insert(val());
            }
            "--keep-all-variants" => {
                cfg.skip.clear();
                skip_overridden = true;
            }
            "--per-variant" => cfg.per_variant = val().parse().unwrap_or(cfg.per_variant),
            "--per-game" => cfg.per_game = val().parse().unwrap_or(cfg.per_game).max(1),
            "--screen-depth" => cfg.screen_depth = val().parse().unwrap_or(cfg.screen_depth),
            "--verify-depth" => cfg.verify_depth = val().parse().unwrap_or(cfg.verify_depth),
            "--cook-depth" => cfg.cook_depth = val().parse().unwrap_or(cfg.cook_depth),
            "--defend-depth" => cfg.defend_depth = val().parse().unwrap_or(cfg.defend_depth),
            "--rate-depth" => cfg.rate_depth = val().parse().unwrap_or(cfg.rate_depth),
            "--cap-ms" => cfg.cap_ms = val().parse().unwrap_or(cfg.cap_ms),
            "--budget-ms" => cfg.budget_ms = val().parse().unwrap_or(cfg.budget_ms),
            "--defend-pv" => cfg.defend_pv = val().parse().unwrap_or(cfg.defend_pv),
            "--defence-window" => cfg.defence_window = val().parse().unwrap_or(cfg.defence_window),
            "--recook" => {
                cfg.recook = true;
                cfg.rate_only = true;
                cfg.refresh = true;
            }
            "--max-plies" => cfg.max_plies = val().parse().unwrap_or(cfg.max_plies),
            "--hash" => cfg.hash_mb = val().parse().unwrap_or(cfg.hash_mb),
            "--threads" => cfg.threads = val().parse().unwrap_or(0),
            "--wide" => cfg.scan = cfg.scan.wide(),
            "--rate-only" => cfg.rate_only = true,
            "--refresh" => {
                cfg.rate_only = true;
                cfg.refresh = true;
            }
            "--deep-verify" => {
                cfg.rate_only = true;
                cfg.refresh = true;
                cfg.deep_verify = true;
            }
            "--deep-depth" => cfg.deep_depth = val().parse().unwrap_or(cfg.deep_depth),
            "--deep-cap-ms" => cfg.deep_cap_ms = val().parse().unwrap_or(cfg.deep_cap_ms),
            "--fresh" => cfg.fresh = true,
            "--dry-run" => cfg.dry_run = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => eprintln!("warning: ignoring unknown argument {other}"),
        }
    }
    if cfg.corpus.is_empty() {
        cfg.corpus.push(PathBuf::from("."));
    }
    cfg
}

fn print_help() {
    println!(
        "puzzle_gen - mine tactical puzzles from annotated self-play games

  --corpus <dir>        directory holding games*.json (repeatable, default \".\")
  --out <file>          output CSV (default puzzles.csv)
  --skip-variant <name> exclude a variant (default: Chess); repeat to add more
  --keep-all-variants   do not exclude anything
  --per-variant <n>     cap candidates per variant (default 4000)
  --per-game <n>        max spaced candidates from one game (default 4)
  --screen-depth <n>    cheap pre-screen depth (default 7)
  --verify-depth <n>    stage-2 depth (default 11)
  --cook-depth <n>      winner-ply depth (default 13)
  --defend-depth <n>    loser-ply depth (default 10)
  --rate-depth <n>      depth-to-find probe depth (default 13)
  --cap-ms <n>          per-search wall-clock cap (default 1500)
  --budget-ms <n>       per-candidate cook budget (default 20000)
  --max-plies <n>       longest solution in plies (default 13)
  --hash <mb>           TT size per thread (default 32)
  --threads <n>         worker threads (default: all cores)
  --wide                relax the stage-0 detectors: more positions examined,
                        same validity rule, so more puzzles for more CPU
  --fresh               discard an existing CSV and its .progress and restart
  --rate-only           just re-rate an existing CSV and exit
  --refresh             re-rate AND recompute the search-free difficulty features,
                        so the rating model can be retuned with no re-searching
  --recook              rebuild each stored puzzle's line, letting the defender
                        pick near-equal replies that keep the attacker on a single
                        move, so solutions run as long as they really are forced
  --defend-pv <n>       defensive replies considered per ply (default 4)
  --defence-window <f>  win-chance a defence may concede to extend the line (0.10)
  --deep-verify         re-search every stored puzzle deeply: records the true
                        score and mate distance, drops any whose answer is no
                        longer uniquely best, then re-rates (implies --refresh)
  --deep-depth <n>      depth for --deep-verify (default 20)
  --deep-cap-ms <n>     per-search cap for --deep-verify (default 20000)
  --dry-run             stages 0-1 only, print candidate stats

Rows are appended as they are found and every processed candidate is recorded in
<out>.progress, so re-running the same command resumes instead of redoing work."
    );
}

// ---------------------------------------------------------------------------
// stage 0: scan the annotation trace
// ---------------------------------------------------------------------------

struct Cand {
    ply: usize,
    source: Source,
    ann_score: i32,
}

struct GameRec {
    variant: Variant,
    position_icn: String,
    moves: Vec<String>,
    cands: Vec<Cand>,
}

fn header<'a>(raw: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("[{key} \"");
    let start = raw.find(&pat)? + pat.len();
    let end = raw[start..].find('"')? + start;
    Some(&raw[start..end])
}

/// Splits a corpus record into (position ICN, move blob). Position tokens never
/// contain `>`, so the first one locates the move list even though the `{...}`
/// comments contain spaces.
fn split_record(raw: &str) -> Option<(&str, &str)> {
    // Walk the leading tag block: the `{[%clk ...]}` comments also contain "] ",
    // so scanning from the end lands in the middle of the move list.
    let mut body = raw.trim();
    while body.starts_with('[') {
        let end = body.find(']')?;
        body = body[end + 1..].trim_start();
    }
    let gt = body.find('>')?;
    let cut = body[..gt].rfind(' ').map_or(0, |i| i + 1);
    Some((body[..cut].trim_end(), &body[cut..]))
}

/// Annotation score in White-ahead centipawns. The harness writes Black-ahead.
fn parse_annotation(mv: &str) -> Option<i32> {
    if let Some(p) = mv.find("[%mate ") {
        let rest = &mv[p + 7..];
        let end = rest.find(']')?;
        let n: i32 = rest[..end].trim().parse().ok()?;
        let dist = n.abs().clamp(1, 999);
        return Some(if n > 0 {
            -(ANN_MATE - dist)
        } else {
            ANN_MATE - dist
        });
    }
    let p = mv.find("[%eval ")?;
    let rest = &mv[p + 7..];
    let end = rest.find(']')?;
    let v: f64 = rest[..end].trim().parse().ok()?;
    Some(-(v * 100.0).round() as i32)
}

fn scan_game(
    raw: &str,
    skip: &FxHashSet<String>,
    max_per_game: usize,
    sc: &Scan,
) -> Option<GameRec> {
    let name = header(raw, "Variant")?;
    if skip.contains(name) {
        return None;
    }
    let variant = Variant::parse(name);
    // Variant::parse falls back to a default on unknown input; a round-trip
    // mismatch means we would silently mine the wrong ruleset.
    if variant.to_str() != name || !ALLOWED_VARIANTS.iter().any(|v| v.to_str() == name) {
        return None;
    }

    let (position_icn, blob) = split_record(raw)?;
    let mut moves: Vec<String> = Vec::new();
    let mut evals: Vec<i32> = Vec::new();
    for part in blob.split('|') {
        let Some(score) = parse_annotation(part) else {
            break;
        };
        let txt = part.split('{').next()?.trim();
        if txt.is_empty() {
            break;
        }
        moves.push(txt.to_string());
        evals.push(score);
    }
    let n = evals.len();
    if n < sc.quiet_window + 4 {
        return None;
    }

    // Winner-perspective score after ply i (the side to move there is White iff i is odd).
    let wpov = |i: usize| if i % 2 == 1 { evals[i] } else { -evals[i] };
    let last = n - 1; // the final ply ends the game: no puzzle there

    // Every qualifying ply is scored, then a spaced-out top few are kept: only a
    // small fraction survives the only-move test later, so one candidate per game
    // leaves far too little to work with.
    let mut scored: Vec<(i32, Cand)> = Vec::new();

    // Source A - the game turned here.
    for i in sc.quiet_window..last {
        let s = wpov(i);
        if !ann_is_win(s) && s < sc.turn_win_floor {
            continue;
        }
        let sgn = if i % 2 == 1 { 1 } else { -1 };
        if (i - sc.quiet_window..i).all(|j| sgn * evals[j] < sc.turn_quiet_ceil) {
            // Turning points are the best source, and the earliest is the moment
            // the position actually broke rather than a later confirmation.
            scored.push((
                10_000 - i as i32,
                Cand {
                    ply: i,
                    source: Source::TurningPoint,
                    ann_score: s,
                },
            ));
        }
    }

    // Source B - a mate that is not the previous ply's mate counted down by one.
    // That single test drops the entire mop-up tail: 93% of mate-scored plies in
    // this corpus sit behind a >=+20 evaluation and are pure conversion.
    for i in 2..last {
        let s = wpov(i);
        if !ann_is_win(s) {
            continue;
        }
        let dist = ann_mate_dist(s);
        if !(1..=sc.mate_max_dist).contains(&dist) {
            continue;
        }
        let prev = wpov(i - 2);
        let surprise = if ann_is_win(prev) {
            let pd = ann_mate_dist(prev);
            if pd <= dist + 1 {
                continue; // already knew, or found a slower mate
            }
            (pd - dist).min(50)
        } else {
            100 // mate materialised out of a non-mate evaluation
        };
        scored.push((
            surprise * 10 + dist,
            Cand {
                ply: i,
                source: Source::MateShot,
                ann_score: s,
            },
        ));
    }

    // Source C - the mover was holding, then handed the game away. The position
    // *before* that move is the puzzle, and its answer is the only defence.
    for j in sc.quiet_window..last {
        let sgn = if j % 2 == 1 { 1 } else { -1 };
        let here = sgn * evals[j];
        if !(-sc.def_hold_cp..sc.turn_win_floor).contains(&here) {
            continue; // already lost, or already winning: not a defensive task
        }
        let after = sgn * evals[j + 1];
        if after > -sc.def_lost_cp {
            continue;
        }
        // Had they been holding? Read their OWN previous searches, which sit at odd
        // offsets from j; the even offsets are the opponent's engine talking.
        if !(1..=2).all(|k| sgn * evals[j + 1 - 2 * k] >= -sc.def_hold_cp) {
            continue;
        }
        scored.push((
            5_000 - j as i32,
            Cand {
                ply: j,
                source: Source::MissedSave,
                ann_score: here,
            },
        ));
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    let mut cands: Vec<Cand> = Vec::new();
    for (_, c) in scored {
        if cands.len() >= max_per_game {
            break;
        }
        if cands.iter().all(|k| k.ply.abs_diff(c.ply) >= sc.min_ply_gap) {
            cands.push(c);
        }
    }
    if cands.is_empty() {
        return None;
    }
    let keep = cands.iter().map(|c| c.ply).max().unwrap_or(0) + 1;
    moves.truncate(keep);
    Some(GameRec {
        variant,
        position_icn: position_icn.to_string(),
        moves,
        cands,
    })
}

fn collect_corpus_files(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for d in dirs {
        let Ok(rd) = fs::read_dir(d) else {
            eprintln!("warning: cannot read {}", d.display());
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with("games") && name.ends_with(".json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn scan(cfg: &Cfg) -> Vec<GameRec> {
    let files = collect_corpus_files(&cfg.corpus);
    println!("stage 0: scanning {} corpus files", files.len());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  [{elapsed_precise}] [{bar:32.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let recs: Vec<GameRec> = files
        .par_iter()
        .flat_map(|path| {
            let out = match fs::read_to_string(path) {
                Err(e) => {
                    pb.suspend(|| eprintln!("  ! read {}: {e}", path.display()));
                    Vec::new()
                }
                Ok(txt) => match serde_json::from_str::<Vec<String>>(&txt) {
                    Err(e) => {
                        pb.suspend(|| eprintln!("  ! parse {}: {e}", path.display()));
                        Vec::new()
                    }
                    Ok(games) => games
                        .iter()
                        .filter_map(|g| scan_game(g, &cfg.skip, cfg.per_game, &cfg.scan))
                        .collect::<Vec<_>>(),
                },
            };
            pb.inc(1);
            out
        })
        .collect();
    pb.finish_and_clear();
    recs
}

// ---------------------------------------------------------------------------
// stage 1: replay to the candidate positions
// ---------------------------------------------------------------------------

struct Candidate {
    game: std::sync::Arc<GameData>,
    ply: usize,
    source: Source,
    ann_score: i32,
    hash: u64,
}

/// One game's replayable text, shared by every candidate taken from it so the
/// puzzle start can be rewound without re-parsing anything.
struct GameData {
    start_icn: String,
    moves: Vec<String>,
}

impl GameData {
    /// Game-prefix ICN through `ply` inclusive.
    fn prefix(&self, ply: usize) -> String {
        let mut s = self.start_icn.clone();
        for (i, m) in self.moves[..=ply].iter().enumerate() {
            s.push(if i == 0 { ' ' } else { '|' });
            s.push_str(m);
        }
        s
    }

    fn state_at(&self, ply: usize) -> GameState {
        let mut st = GameState::new();
        st.setup_position_from_icn(&self.start_icn);
        for m in &self.moves[..=ply] {
            if apply_icn_move(&mut st, m).is_none() {
                break;
            }
        }
        forget_history(&mut st);
        st
    }

    /// Where the move at `ply` landed and whether it captured there.
    fn move_info(&self, ply: usize) -> Option<(Coordinate, bool)> {
        let mut st = GameState::new();
        st.setup_position_from_icn(&self.start_icn);
        for m in &self.moves[..ply] {
            apply_icn_move(&mut st, m)?;
        }
        apply_icn_move(&mut st, self.moves.get(ply)?)
    }
}

/// Applies one ICN move, returning where it landed and whether it captured.
fn apply_icn_move(st: &mut GameState, mv: &str) -> Option<(Coordinate, bool)> {
    let (from, rest) = mv.split_once('>')?;
    let (to, promo) = match rest.split_once('=') {
        Some((t, p)) => (t, Some(p)),
        None => (rest, None),
    };
    let coord = |s: &str| -> Option<(i64, i64)> {
        let (x, y) = s.split_once(',')?;
        let y = y.trim_matches(|c| c == '+' || c == '!' || c == '#' || c == '?');
        Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
    };
    let ((fx, fy), (tx, ty)) = (coord(from)?, coord(to)?);
    let captured = st.board.is_occupied(tx, ty);
    st.make_move_coords(fx, fy, tx, ty, promo);
    Some((Coordinate::new(tx, ty), captured))
}

/// A solver sees a position, not the game that produced it, so the puzzle must not
/// inherit a half-played 50-move clock or a repetition history it cannot know
/// about. Both still accrue normally from moves made *inside* the puzzle.
fn forget_history(st: &mut GameState) {
    st.halfmove_clock = 0;
    st.hash_stack.clear();
    st.rep_hash_stack.clear();
    st.null_moves = 0;
}

fn material_lead(st: &GameState, side: PlayerColor) -> i32 {
    let mut total = 0;
    for (_, _, p) in st.board.iter() {
        let v = get_piece_value_base(p.piece_type());
        match p.color() {
            c if c == side => total += v,
            PlayerColor::Neutral => {}
            _ => total -= v,
        }
    }
    total
}

fn replay(rec: &GameRec) -> Vec<Candidate> {
    let mut st = GameState::new();
    st.setup_position_from_icn(&rec.position_icn);
    // A puzzle whose goal is "capture every piece" or "capture the royal" reads as
    // a broken checkmate puzzle, so only pure-checkmate rulesets qualify.
    if st.game_rules.white_win_condition != WinCondition::Checkmate
        || st.game_rules.black_win_condition != WinCondition::Checkmate
    {
        return Vec::new();
    }
    let game = std::sync::Arc::new(GameData {
        start_icn: rec.position_icn.clone(),
        moves: rec.moves.clone(),
    });
    let mut out = Vec::new();

    for (i, mv) in rec.moves.iter().enumerate() {
        if apply_icn_move(&mut st, mv).is_none() {
            break;
        }

        let Some(c) = rec.cands.iter().find(|c| c.ply == i) else {
            continue;
        };
        if u32::from(st.white_piece_count + st.black_piece_count) < MIN_PIECES {
            continue;
        }
        if legal_moves(&mut st).len() < 2 {
            continue;
        }
        if material_lead(&st, st.turn) > MAX_LEAD {
            continue;
        }
        out.push(Candidate {
            game: std::sync::Arc::clone(&game),
            ply: i,
            source: c.source,
            ann_score: c.ann_score,
            hash: st.hash,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// stages 2-3: verify and cook
// ---------------------------------------------------------------------------

/// Soft limit at `cap_ms` so whole depths complete, hard ceiling at 4x so a single
/// dense position cannot run for minutes and strand its variant batch on one core.
fn mpv(st: &mut GameState, depth: usize, cap_ms: u128, lines: usize) -> search::MultiPVResult {
    search::get_best_moves_multipv(st, depth, cap_ms, cap_ms.saturating_mul(4), lines, true, true)
}

/// Fully legal moves. `get_legal_moves` is pseudo-legal and keeps the slider
/// candidate cache, so an exact list needs the buffer form (which bypasses it)
/// plus a legality filter -- the same two steps the search does at its root.
/// Without this an empty list never appears and checkmate is never detected.
fn legal_moves(st: &mut GameState) -> MoveList {
    let mut pseudo = MoveList::new();
    st.get_legal_moves_into(&mut pseudo);
    let mut out = MoveList::new();
    for m in pseudo {
        let undo = st.make_move(&m);
        let ok = !st.is_move_illegal();
        st.undo_move(&m, undo);
        if ok {
            out.push(m);
        }
    }
    out
}

/// The one rule, applied at every solver ply: the move has to put the game on the
/// good side of an outcome boundary and every alternative on the bad side. Read as
/// an attack (win vs not-win) or as a defence (playable vs lost). A gap that stays
/// inside one band -- +20 down to +10, say -- is not a puzzle, because missing the
/// move costs nothing that matters.
fn valid_attack(best: i32, second: Option<i32>) -> bool {
    let wins = search::is_win(best) || best >= WON_CP;
    let holds = best >= HOLD_FLOOR;
    if !wins && !holds {
        return false;
    }
    let Some(s) = second else {
        return true;
    };
    (wins && !search::is_win(s) && s <= HELD_CP)
        || (holds && (search::is_loss(s) || s <= LOST_CEIL))
}

/// Where candidates die, so thresholds get tuned against counts instead of guesses.
mod rej {
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub const NO_LINES: usize = 0;
    pub const SCREENED: usize = 1;
    pub const WEAK: usize = 2;
    pub const NOT_ONLY: usize = 3;
    pub const COOK_FAILED: usize = 4;
    pub const TRIVIAL: usize = 5;
    pub const UNSOUND: usize = 6;
    pub const OK: usize = 7;
    pub const N: usize = 8;
    pub static COUNTS: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
    /// Runner-up score at the root in 100cp buckets (index 0 = also a mate,
    /// 1 = <= 0, then +100 each), for choosing HELD_CP against real counts.
    pub static SECOND: [AtomicUsize; 24] = [const { AtomicUsize::new(0) }; 24];

    /// Seen/accepted per source, so the pool can be rebalanced toward whichever
    /// one actually survives the only-move rule.
    pub static SRC_SEEN: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
    pub static SRC_OK: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

    pub fn hit(k: usize) {
        COUNTS[k].fetch_add(1, Ordering::Relaxed);
    }
    pub fn src(i: usize, accepted: bool) {
        if accepted {
            SRC_OK[i].fetch_add(1, Ordering::Relaxed);
        } else {
            SRC_SEEN[i].fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn second(s: i32, is_mate: bool) {
        let b = if is_mate {
            0
        } else {
            (1 + (s.max(0) / 100)).clamp(1, 23) as usize
        };
        SECOND[b].fetch_add(1, Ordering::Relaxed);
    }
    /// Why a mate line stopped short: 0 not-only-move, 1 ply cap, 2 no defence,
    /// 3 score fell out of mate, 4 reached mate.
    pub static MATE_STOP: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
    pub fn mate_stop(k: usize) {
        MATE_STOP[k].fetch_add(1, Ordering::Relaxed);
    }

    pub fn report() {
        const STOP: [&str; 5] = [
            "not an only-move",
            "hit ply cap",
            "no defence found",
            "mate score lost",
            "delivered mate",
        ];
        println!("\nmate lines stopped because:");
        for i in 0..5 {
            println!("  {:<24}{:>7}", STOP[i], MATE_STOP[i].load(Ordering::Relaxed));
        }

        const NAMES: [&str; N] = [
            "no search lines",
            "screened out (shallow)",
            "best move does not win",
            "an alternative also wins",
            "cook produced nothing",
            "trivial recapture",
            "failed soundness check",
            "accepted",
        ];
        println!("\ncandidate outcomes:");
        for i in 0..N {
            println!("  {:<26}{:>7}", NAMES[i], COUNTS[i].load(Ordering::Relaxed));
        }
        println!("  by source:");
        for (i, n) in ["turning_point", "mate_shot", "missed_save"].iter().enumerate() {
            let ok = SRC_OK[i].load(Ordering::Relaxed);
            let seen = SRC_SEEN[i].load(Ordering::Relaxed) + ok;
            let pct = if seen > 0 { 100.0 * ok as f64 / seen as f64 } else { 0.0 };
            println!("    {n:<16}{seen:>7} seen{ok:>7} kept  {pct:>5.1}%");
        }
        println!("  runner-up score at the root (puzzle needs this below HELD_CP):");
        for (i, c) in SECOND.iter().enumerate() {
            let n = c.load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            match i {
                0 => println!("    also mates      {n:>6}"),
                1 => println!("    <= 0            {n:>6}"),
                _ => println!("    >= {:<13}{:>6}", (i - 1) * 100, n),
            }
        }
    }
}

struct Cooked {
    line: Vec<Move>,
    final_score: i32,
    ends_in_mate: bool,
    shallow_rank: usize,
    root_margin: f64,
    defender_replies: Vec<usize>,
}

fn cook(st: &mut GameState, winner: PlayerColor, looking_for_mate: bool, cfg: &Cfg) -> Option<Cooked> {
    let mut line: Vec<Move> = Vec::new();
    let mut defender_replies = Vec::new();
    // Trailing winner moves with no alternative are filler, not part of the puzzle.
    let mut had_choice: Vec<bool> = Vec::new();
    let mut shallow_rank = 0usize;
    let mut root_margin = 1.0f64;
    let mut final_score = 0i32;
    let mut ends_in_mate = false;

    // Time limits are soft, so a single search only stops between depths and a
    // dense position can run far past `cap_ms`. Without a budget here one slow
    // candidate holds up its whole variant batch while the other cores idle.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(cfg.budget_ms);

    // Positions seen inside the puzzle. A "solution" that shuffles back and forth
    // is not a solution -- the defender just claims the repetition.
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    seen.insert(st.hash);

    let mut stop = 1usize; // assume the ply cap until something else ends the walk
    while line.len() < cfg.max_plies {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let legal = legal_moves(st);
        if legal.is_empty() {
            ends_in_mate = st.is_in_check();
            stop = 4;
            break;
        }
        if st.turn == winner {
            let r = mpv(st, cfg.cook_depth, cfg.cap_ms, 2);
            let Some(best) = r.lines.first() else {
                stop = 2;
                break;
            };
            let second = r.lines.get(1).map(|l| l.score);
            // One rule at every winner ply: this move wins, nothing else does.
            // Where that stops holding the line stops, and the prefix is still a
            // sound puzzle -- just a shorter one.
            if !valid_attack(best.score, second) {
                stop = 0;
                break;
            }
            if line.is_empty() {
                shallow_rank = r
                    .shallow_order
                    .iter()
                    .position(|m| *m == best.mv)
                    .unwrap_or(0);
                root_margin = match second {
                    Some(s) => (win_chances(best.score) - win_chances(s)).clamp(0.0, 2.0),
                    None => 2.0,
                };
            }
            final_score = best.score;
            had_choice.push(legal.len() > 1 && second.is_some());
            line.push(best.mv);
            st.make_move(&best.mv);
        } else {
            defender_replies.push(legal.len());
            let r = mpv(st, cfg.defend_depth, cfg.cap_ms, cfg.defend_pv);
            let Some(top) = r.lines.first() else {
                stop = 2;
                break;
            };
            // The strongest defence is not always the best one to show. If a reply
            // that concedes almost nothing leaves the attacker exactly one winning
            // move, while the top reply leaves several, the near-equal move makes the
            // better puzzle: it forces the solver to find more precise moves. So
            // among replies that do not throw the game away, prefer one that keeps
            // the attacker on a single move, and take the strongest such reply.
            let mut choice = top.mv;
            let best_def = top.score;
            for l in r.lines.iter() {
                if win_chances(best_def) - win_chances(l.score) > cfg.defence_window {
                    break; // lines are sorted, so everything after is worse still
                }
                let undo = st.make_move(&l.mv);
                let a = mpv(st, cfg.verify_depth, cfg.cap_ms, 2);
                let forced = a.lines.first().is_some_and(|b| {
                    valid_attack(b.score, a.lines.get(1).map(|x| x.score))
                });
                st.undo_move(&l.mv, undo);
                if forced {
                    choice = l.mv;
                    break;
                }
            }
            line.push(choice);
            st.make_move(&choice);
        }
        if !seen.insert(st.hash) {
            return None; // the line repeats: not a puzzle at all
        }
    }
    if looking_for_mate {
        rej::mate_stop(stop);
    }

    // The while-loop also exits on the ply cap, so the terminal test has to run here
    // too or a mate delivered on the last allowed ply is missed.
    if !ends_in_mate {
        ends_in_mate = legal_moves(st).is_empty() && st.is_in_check();
    }

    // Keep an odd length so the solver always makes the last move, and drop a
    // trailing winner move that had no alternative to begin with. A mating move is
    // never filler, so a line that ends in mate is left alone.
    loop {
        if line.len().is_multiple_of(2) && !line.is_empty() {
            line.pop();
            ends_in_mate = false;
            continue;
        }
        if !ends_in_mate && line.len() > 1 && !had_choice.last().copied().unwrap_or(true) {
            line.pop();
            had_choice.pop();
            continue;
        }
        break;
    }
    if line.is_empty() {
        return None;
    }
    Some(Cooked {
        line,
        final_score,
        ends_in_mate,
        shallow_rank,
        root_margin,
        defender_replies,
    })
}

/// Shallowest depth from which iterative deepening never leaves the solution move.
fn depth_to_find(st: &mut GameState, solution: Move, cfg: &Cfg) -> usize {
    let mut last_wrong = 0usize;
    let mut cb = |info: &search::DepthInfo| {
        if let Some(l) = info.lines.first()
            && l.mv != solution
        {
            last_wrong = info.depth;
        }
    };
    // slice_ms 0 means no deadline whatsoever, which let one dense position stall
    // a whole variant batch on a single core.
    search::analyse_position(st, cfg.rate_depth, 1, cfg.cap_ms, 1, &mut cb);
    last_wrong + 1
}

// ---------------------------------------------------------------------------
// stage 4: features, themes, rating
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Features {
    shallow_rank: usize,
    depth_to_find: usize,
    margin: f64,
    plies: usize,
    mean_replies: f64,
    quiet_key: bool,
    sacrifice: i32,
    travel: i64,
    remoteness: i64,
    pieces: u32,
    relevant: usize,
    fairy_used: f64,
    fairy_present: f64,
    /// Share of solver moves that check or capture. A forcing line prunes itself as
    /// you calculate it, so it is easier per ply than a quiet one of equal length.
    forcing: f64,
    ends_in_mate: bool,
    /// Plies to the mate when the position is a forced one. The cook can truncate
    /// its line well short of the mate, but the solver still has to see it through,
    /// so this is what the calculation burden should be measured against.
    mate_plies: usize,
    root_moves: usize,
    root_forcing: usize,
}

/// How unintuitive a piece is to calculate with, which is not the same as how
/// strong it is. A chancellor is just rook+knight and reads easily; a huygen
/// jumps prime distances and a knightrider slides in knight steps, and neither
/// has any counterpart in normal chess.
fn fairy_complexity(pt: PieceType) -> f64 {
    match pt {
        PieceType::Huygen => 1.00,
        PieceType::Rose => 0.90,
        PieceType::Knightrider => 0.85,
        PieceType::Camel | PieceType::Giraffe | PieceType::Zebra => 0.60,
        PieceType::Hawk => 0.55,
        PieceType::Centaur | PieceType::RoyalCentaur => 0.45,
        PieceType::Amazon => 0.40,
        PieceType::Archbishop | PieceType::Chancellor => 0.35,
        PieceType::Guard | PieceType::RoyalQueen => 0.20,
        _ => 0.0,
    }
}

/// Pieces that bear on a square from arbitrarily far away, so distance alone does
/// not tell you whether they matter.
fn is_long_range(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Queen
            | PieceType::Rook
            | PieceType::Bishop
            | PieceType::Amazon
            | PieceType::Chancellor
            | PieceType::Archbishop
            | PieceType::Knightrider
            | PieceType::RoyalQueen
            | PieceType::Huygen
            | PieceType::Rose
    )
}

/// How far from the action a piece can sit and still be part of the problem.
const ACTION_RADIUS: i64 = 6;

/// Everything about a puzzle's difficulty that needs no search. Shared by
/// generation and `--refresh` so a retune reproduces the original numbers exactly.
struct BoardFeats {
    pieces: u32,
    /// Pieces that actually bear on the tactic: near the squares the solution
    /// touches, or lined up on them from a distance. A board carrying forty pawns
    /// twenty squares away is not forty pawns harder to read.
    relevant: usize,
    /// Complexity of the most exotic piece the solver actually moves or takes.
    fairy_used: f64,
    /// Complexity merely sitting on the board, which counts for much less.
    fairy_present: f64,
    /// Solver moves that check or capture. Both prune the tree as you calculate.
    forcing: f64,
    ends_in_mate: bool,
    /// Legal moves in the puzzle position: how many candidates the solver has to
    /// sift before finding the one that works.
    root_moves: usize,
    /// Of those, the ones that check or capture -- the set actually scanned when
    /// the answer is known to be forcing.
    root_forcing: usize,
}

/// Replays a solution to derive its search-free features. `None` means the line
/// repeats a position, which makes it no solution at all.
fn board_features(icn: &str, solution: &[String]) -> Option<BoardFeats> {
    let mut st = GameState::new();
    st.setup_position_from_icn(icn);
    forget_history(&mut st);
    let pieces = u32::from(st.white_piece_count) + u32::from(st.black_piece_count);

    let mut action: Vec<Coordinate> = Vec::new();
    if let Some(r) = opposing_royal(&st, st.turn) {
        action.push(r);
    }

    let roots = legal_moves(&mut st);
    let root_moves = roots.len();
    let root_forcing = roots
        .iter()
        .filter(|m| {
            let cap = st.board.is_occupied(m.to.x, m.to.y);
            let undo = st.make_move(m);
            let chk = st.is_in_check();
            st.undo_move(m, undo);
            cap || chk
        })
        .count();
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    seen.insert(st.hash);
    let (mut forcing_moves, mut solver_moves) = (0usize, 0usize);
    let mut fairy_used: f64 = 0.0;

    for (k, mv) in solution.iter().enumerate() {
        let from = mv.split_once('>').and_then(|(f, _)| parse_coord(f));
        let solver = k.is_multiple_of(2);
        if solver && let Some(f) = from {
            if let Some(p) = st.board.get_piece(f.x, f.y) {
                fairy_used = fairy_used.max(fairy_complexity(p.piece_type()));
            }
            action.push(f);
        }
        let captured = mv
            .split_once('>')
            .and_then(|(_, t)| parse_coord(t))
            .map(|t| {
                if solver && let Some(p) = st.board.get_piece(t.x, t.y) {
                    fairy_used = fairy_used.max(fairy_complexity(p.piece_type()));
                }
                st.board.is_occupied(t.x, t.y)
            })
            .unwrap_or(false);

        // Guards against a movegen change invalidating a stored line, not just
        // against a bad parse.
        if !legal_moves(&mut st).iter().any(|m| move_to_icn(m) == *mv) {
            return None;
        }
        let (to, _) = apply_icn_move(&mut st, mv)?;
        if !seen.insert(st.hash) {
            return None;
        }
        if solver {
            action.push(to);
            solver_moves += 1;
            if captured || st.is_in_check() {
                forcing_moves += 1;
            }
        }
    }
    if solver_moves == 0 {
        return None;
    }

    let relevant = st
        .board
        .iter()
        .filter(|(x, y, p)| {
            if matches!(p.piece_type(), PieceType::Void | PieceType::Obstacle) {
                return false;
            }
            let here = Coordinate::new(*x, *y);
            action.iter().any(|a| {
                chebyshev(here, *a) <= ACTION_RADIUS
                    || (is_long_range(p.piece_type())
                        && (a.x == here.x
                            || a.y == here.y
                            || (a.x - here.x).abs() == (a.y - here.y).abs()))
            })
        })
        .count();

    let mut types: FxHashSet<u8> = FxHashSet::default();
    let fairy_present: f64 = st
        .board
        .iter()
        .filter(|(_, _, p)| types.insert(p.piece_type() as u8))
        .map(|(_, _, p)| fairy_complexity(p.piece_type()))
        .sum();

    Some(BoardFeats {
        pieces,
        relevant,
        fairy_used,
        fairy_present,
        forcing: forcing_moves as f64 / solver_moves as f64,
        ends_in_mate: legal_moves(&mut st).is_empty() && st.is_in_check(),
        root_moves,
        root_forcing,
    })
}

fn parse_coord(s: &str) -> Option<Coordinate> {
    let s = s.split('=').next()?;
    let (x, y) = s.split_once(',')?;
    let y = y.trim_matches(|c| c == '+' || c == '!' || c == '#' || c == '?');
    Some(Coordinate::new(x.trim().parse().ok()?, y.trim().parse().ok()?))
}

fn chebyshev(a: Coordinate, b: Coordinate) -> i64 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

/// Weighted blend of the signals that make a tactic hard to see. Only the ordering
/// matters: the rating itself comes from a rank transform below.
///
/// How much has to be calculated dominates -- solution length, then how crowded the
/// board is and how many fairy pieces are on it, which is what actually separates a
/// long forcing sequence from a one-move shot. The engine-derived signals
/// (shallow_rank, depth-to-find) rank *within* that rather than over it.
/// Absolute rating in Elo-like points, judged on the puzzle alone.
///
/// Deliberately NOT a rank transform. Mapping the set onto a fixed distribution
/// guarantees something is rated 2800 every run whether or not anything that hard
/// exists. Here every term is a fixed number of points, so 2800 has to be earned:
/// reaching it needs a long quiet line, an invisible key move, a crowded board and
/// an exotic piece all at once, which almost never co-occur.
///
/// Anchors it is tuned against: an obvious forced mate in 1 lands near 600; a
/// forcing mate in 3 near 1000; a mid-length half-forcing tactic near 1500; a long
/// quiet combination with an unnatural key move near 2400.
fn puzzle_rating(f: &Features) -> i32 {
    let n = |v: f64, hi: f64| (v / hi).clamp(0.0, 1.0);
    let forcing = f.forcing.clamp(0.0, 1.0);

    // Forcing only makes a line cheap if you can see the moves at a glance. A
    // sequence of huygen jumps has to be verified square by square, so an exotic
    // key piece takes most of that discount back.
    let eff_forcing = forcing * (1.0 - 0.50 * f.fairy_used.clamp(0.0, 1.0));
    // How much has to be calculated. 0.45, not 0.60: forcing shortens the work but
    // every move still has to be checked, and over-discounting put forced mates in
    // 3 below mates in 2.
    let depth = f.plies.max(f.mate_plies);
    let eff_plies = depth.saturating_sub(1) as f64 * (1.0 - 0.45 * eff_forcing);
    let calc = n(eff_plies, 6.0);
    // How invisible the key move is. Engine-centric, and it can read zero on a
    // position humans find hard, so it no longer dominates the scale.
    let obscure = 0.60 * n(f.shallow_rank as f64, 10.0)
        + 0.40 * n(f.depth_to_find.saturating_sub(2) as f64, 12.0);
    // How much board the solver has to hold in their head. Exotic pieces count for
    // most when the solution uses one, but a huygen merely sitting there still
    // warps what is possible, and `fairy_present` is complexity-weighted so a
    // huygen or knightrider counts far above a chancellor.
    let complex = 0.55 * n(f.relevant as f64, 45.0)
        + 0.25 * f.fairy_used.clamp(0.0, 1.0)
        + 0.20 * n(f.fairy_present, 2.0);
    // How exactly it has to be played.
    let precision = 0.50 * (1.0 - n(f.margin, 1.4)) + 0.50 * n(f.mean_replies, 30.0);
    // Red herrings: how many moves actually have to be looked at. When the answer
    // is forcing the solver scans only checks and captures -- but on a crowded
    // board that is still a long list, so count it rather than zeroing it out.
    // Scaled by depth as well: sifting fifty candidates is only expensive when each
    // one has to be calculated out. For a mate in 1 you glance at the checks and
    // you are done, so a crowded board must not inflate it.
    let candidates =
        eff_forcing * f.root_forcing as f64 + (1.0 - eff_forcing) * f.root_moves as f64;
    let branching = n(candidates, 50.0) * n(depth as f64, 3.0);

    let r = 640.0
        + 660.0 * calc
        + 460.0 * obscure
        + 620.0 * complex
        + 300.0 * branching
        + 260.0 * precision
        + 180.0 * if f.quiet_key { 1.0 } else { 0.0 }
        + 120.0 * n(f.sacrifice as f64, 900.0)
        - 120.0 * eff_forcing
        - 100.0 * if f.ends_in_mate { 1.0 } else { 0.0 };
    ((r / 10.0).round() as i32 * 10).clamp(400, 3000)
}

fn raw_difficulty(f: &Features) -> f64 {
    let n = |v: f64, hi: f64| (v / hi).clamp(0.0, 1.0);
    // Forcing shortens the line you actually have to calculate rather than shaving a
    // fixed amount off the end: at every check or capture the replies collapse to a
    // handful. So it scales the length term instead of being subtracted from it --
    // a seven-ply line of checks is about as much work as a three-ply quiet one.
    let eff_plies = f.plies.saturating_sub(1) as f64 * (1.0 - 0.60 * f.forcing.clamp(0.0, 1.0));
    0.30 * n(eff_plies, 6.0)
        + 0.16 * n(f.shallow_rank as f64, 8.0)
        + 0.14 * n(f.depth_to_find.saturating_sub(2) as f64, 12.0)
        // Only pieces that bear on the tactic. Raw piece count rewards boards that
        // merely have a lot of far-away pawns on them.
        + 0.12 * n(f.relevant as f64, 30.0)
        // A fairy piece the solver has to calculate with is worth far more than one
        // standing somewhere on the board.
        + 0.10 * f.fairy_used.clamp(0.0, 1.0)
        + 0.04 * n(f.fairy_present, 3.0)
        + 0.06 * (1.0 - n(f.margin, 1.4))
        + 0.05 * n(f.mean_replies, 30.0)
        + 0.03 * if f.quiet_key { 1.0 } else { 0.0 }
        // A residual discount on top of the shortened line, plus one for mate: you
        // know what you are looking for, and you know when you have found it.
        - 0.10 * f.forcing.clamp(0.0, 1.0)
        - 0.06 * if f.ends_in_mate { 1.0 } else { 0.0 }
}

fn is_capture(st: &GameState, m: &Move) -> bool {
    st.board.is_occupied(m.to.x, m.to.y)
        || (m.piece.piece_type() == PieceType::Pawn && m.from.x != m.to.x)
}

fn opposing_royal(st: &GameState, side: PlayerColor) -> Option<Coordinate> {
    if side == PlayerColor::White {
        st.black_royals.first().copied()
    } else {
        st.white_royals.first().copied()
    }
}

fn checkers(st: &GameState) -> usize {
    let defender = st.turn;
    let Some(king) = (if defender == PlayerColor::White {
        st.white_royals.first().copied()
    } else {
        st.black_royals.first().copied()
    }) else {
        return 0;
    };
    st.board
        .iter()
        .filter(|(ax, ay, p)| {
            p.color() == defender.opponent()
                && apeiron::moves::is_piece_attacking_square(
                    &st.board,
                    p,
                    &Coordinate::new(*ax, *ay),
                    &king,
                    &st.spatial_indices,
                    &st.game_rules,
                )
        })
        .count()
}

/// Counts enemy pieces the just-moved piece now hits that are worth hitting:
/// royals, undefended pieces, or anything more valuable than the attacker.
fn fork_targets(st: &GameState, at: Coordinate, winner: PlayerColor) -> usize {
    let Some(piece) = st.board.get_piece(at.x, at.y) else {
        return 0;
    };
    let empty_pinned = FxHashMap::default();
    let ctx = MoveGenContext {
        special_rights: &st.special_rights,
        en_passant: &st.en_passant,
        game_rules: &st.game_rules,
        indices: &st.spatial_indices,
        enemy_king_pos: None,
        pinned: &empty_pinned,
    };
    let mut list = MoveList::new();
    apeiron::moves::get_pseudo_legal_moves_for_piece_into(&st.board, &piece, &at, &ctx, &mut list);

    let mine = get_piece_value_base(piece.piece_type());
    let mut seen = FxHashSet::default();
    let mut hits = 0;
    for m in list.iter() {
        let Some(target) = st.board.get_piece(m.to.x, m.to.y) else {
            continue;
        };
        if target.color() != winner.opponent() || !seen.insert((m.to.x, m.to.y)) {
            continue;
        }
        let defended = apeiron::moves::is_square_attacked(
            &st.board,
            &m.to,
            winner.opponent(),
            &st.spatial_indices,
        );
        if target.piece_type().is_royal()
            || !defended
            || get_piece_value_base(target.piece_type()) > mine
        {
            hits += 1;
        }
    }
    hits
}

/// True when removing `from` would expose the enemy royal, i.e. the moved piece
/// was screening a line. Only tested for squares aligned with the royal.
fn reveals_line(st: &GameState, from: Coordinate, winner: PlayerColor) -> bool {
    let Some(royal) = opposing_royal(st, winner) else {
        return false;
    };
    let (dx, dy) = (royal.x - from.x, royal.y - from.y);
    if dx != 0 && dy != 0 && dx.abs() != dy.abs() {
        return false;
    }
    let mut probe = st.board.clone();
    probe.remove_piece(&from.x, &from.y);
    apeiron::moves::is_square_attacked(&probe, &royal, winner, &st.spatial_indices)
}

fn detect_themes(
    icn: &str,
    line: &[Move],
    winner: PlayerColor,
    f: &Features,
    ends_in_mate: bool,
    final_score: i32,
    bounded: bool,
    defensive: bool,
) -> BTreeSet<String> {
    let mut t: BTreeSet<String> = BTreeSet::new();
    let mut st = GameState::new();
    st.setup_position_from_icn(icn);

    let full = line.len();
    if ends_in_mate {
        t.insert("mate".into());
        let n = full.div_ceil(2);
        t.insert(match n {
            1 => "mateIn1".into(),
            2 => "mateIn2".into(),
            3 => "mateIn3".into(),
            4 => "mateIn4".into(),
            _ => "mateIn5plus".into(),
        });
        if bounded {
            t.insert("boundedMate".into());
        } else {
            t.insert("openBoardMate".into());
        }
    } else if defensive {
        // The move does not win anything; it is the only one that avoids losing.
        t.insert("defensiveMove".into());
        t.insert("onlyMove".into());
    } else if final_score.abs() > 600 {
        t.insert("crushing".into());
    } else {
        t.insert("advantage".into());
    }
    match full {
        1 => t.insert("oneMove".into()),
        3 => t.insert("short".into()),
        5 | 7 => t.insert("long".into()),
        _ => t.insert("veryLong".into()),
    };

    let base_lead = material_lead(&st, winner);
    let mut low_lead = base_lead;

    for (i, m) in line.iter().enumerate() {
        let winner_ply = i % 2 == 0;
        if winner_ply {
            let captured = is_capture(&st, m);
            if captured {
                t.insert("capture".into());
            }
            if i == 0 {
                if m.piece.piece_type().to_site_code().len() > 1 {
                    t.insert("fairyPiece".into());
                }
                if f.travel >= 8 {
                    t.insert("longRangeShot".into());
                }
                if f.remoteness >= 12 {
                    t.insert("distantAttack".into());
                }
                if f.quiet_key {
                    t.insert("quietMove".into()); // neither a capture nor a check
                }
            }
            if m.promotion.is_some() {
                t.insert("promotion".into());
                if m.promotion != Some(PieceType::Queen) {
                    t.insert("underPromotion".into());
                }
            }
            if captured
                && !apeiron::moves::is_square_attacked(
                    &st.board,
                    &m.to,
                    winner.opponent(),
                    &st.spatial_indices,
                )
            {
                t.insert("hangingPiece".into());
            }
            if reveals_line(&st, m.from, winner) && !st.is_in_check() {
                t.insert("discoveredAttack".into());
            }
        }

        st.make_move(m);

        if winner_ply {
            if fork_targets(&st, m.to, winner) >= 2 {
                t.insert("fork".into());
            }
            if st.is_in_check() {
                t.insert("check".into());
                if checkers(&st) > 1 {
                    t.insert("doubleCheck".into());
                }
            }
            low_lead = low_lead.min(material_lead(&st, winner));
        }
    }

    if base_lead - low_lead >= 200 {
        t.insert("sacrifice".into());
        if base_lead - low_lead >= 800 {
            t.insert("bigSacrifice".into());
        }
    }
    if t.is_empty() {
        t.insert("tactical".into());
    }
    t
}

// ---------------------------------------------------------------------------
// output
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct PuzzleRecord {
    variant: String,
    /// The puzzle position on its own, no move list, so there is no ambiguity about
    /// which ply it starts on. `solution_moves` is played from here.
    position_icn: String,
    /// Where it came from: starting position plus the game moves up to the puzzle.
    game_icn: String,
    side_to_move: String,
    solution_moves: String,
    rating: i32,
    rating_deviation: i32,
    themes: String,
    source: String,
    solution_plies: usize,
    ends_in_mate: bool,
    final_eval: i32,
    difficulty_raw: f64,
    shallow_rank: usize,
    depth_to_find: usize,
    only_move_margin: f64,
    mean_defender_replies: f64,
    sacrifice_cp: i32,
    key_move_travel: i64,
    piece_count: u32,
    /// Pieces that actually bear on the tactic, unlike `piece_count`.
    #[serde(default)]
    relevant_pieces: usize,
    /// Legal moves in the puzzle position: the solver's candidate load.
    #[serde(default)]
    root_moves: usize,
    /// Of those, the ones that check or capture.
    #[serde(default)]
    root_forcing: usize,
    /// Complexity (x100) of the most exotic piece the solution actually uses.
    fairy_count: usize,
    /// Percent of solver moves that give check. Stored so a re-rate never has to
    /// run the engine again; defaulted so CSVs written before it existed still load.
    #[serde(default)]
    forcing_pct: i32,
    /// From --deep-verify: forced mate in this many moves, 0 if not a mate. The
    /// cook's line can stop short of a mate, so this is the authority for rating.
    #[serde(default)]
    mate_in: i32,
    #[serde(default)]
    deep_eval: i32,
    #[serde(default)]
    verified: bool,
    /// What the annotation-only scan thought, before any search. Handy for
    /// re-tuning stage 0 without re-running the engine.
    scan_eval: i32,
}

/// Rows stream to disk as they are found, so a stopped run keeps its work. The
/// sibling `.progress` file records every candidate already looked at -- puzzle or
/// not -- which is what makes a resume skip them instead of re-searching.
enum Emit {
    Puzzle(Box<PuzzleRecord>),
    Done(u64),
}

fn progress_path(out: &Path) -> PathBuf {
    let mut p = out.to_path_buf();
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("csv");
    p.set_extension(format!("{ext}.progress"));
    p
}

fn load_progress(path: &Path) -> FxHashSet<u64> {
    let mut done = FxHashSet::default();
    if let Ok(f) = fs::File::open(path) {
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            if let Ok(h) = line.trim().parse::<u64>() {
                done.insert(h);
            }
        }
    }
    done
}

fn spawn_writer(out: PathBuf, prog: PathBuf, rx: mpsc::Receiver<Emit>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let fresh = out.metadata().map(|m| m.len() == 0).unwrap_or(true);
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .expect("failed to open output CSV");
        let mut w = csv::WriterBuilder::new().has_headers(fresh).from_writer(f);
        let pf = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&prog)
            .expect("failed to open progress file");
        let mut pw = BufWriter::new(pf);
        let mut n = 0usize;
        for e in rx {
            match e {
                Emit::Puzzle(p) => {
                    w.serialize(&*p).expect("failed to serialize puzzle");
                    let _ = w.flush();
                }
                Emit::Done(h) => {
                    let _ = writeln!(pw, "{h}");
                }
            }
            n += 1;
            if n.is_multiple_of(128) {
                let _ = pw.flush();
            }
        }
        let _ = w.flush();
        let _ = pw.flush();
    })
}

/// Re-examines every stored puzzle at a much greater depth than generation used.
/// Two things come out of it: the true score, including whether the position is a
/// forced mate at all -- the cook truncates a line as soon as two moves both mate,
/// so a mate in 4 can end up recorded as a quiet material win and miss the mate
/// discount -- and a final check that the stored answer is still the unique one.
fn deep_verify(puzzles: &mut [PuzzleRecord], cfg: &Cfg) -> (usize, usize) {
    let mut by_variant: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    for (i, p) in puzzles.iter().enumerate() {
        by_variant.entry(p.variant.as_str()).or_default().push(i);
    }
    let groups: Vec<(Variant, Vec<usize>)> = by_variant
        .iter()
        .filter_map(|(name, idx)| {
            let v = Variant::parse(name);
            (v.to_str() == *name).then(|| (v, idx.clone()))
        })
        .collect();

    let pb = ProgressBar::new(puzzles.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  deep verify [{elapsed_precise}] [{bar:28.green/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let results: Mutex<Vec<(usize, i32, i32, bool)>> = Mutex::new(Vec::new());

    for (v, idx) in groups {
        let b = v.get_default_bounds();
        apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);
        idx.par_iter().for_each(|&i| {
            let p = &puzzles[i];
            let mut st = GameState::new();
            st.setup_position_from_icn(&p.position_icn);
            forget_history(&mut st);
            let r = mpv(&mut st, cfg.deep_depth, cfg.deep_cap_ms, 2);
            let out = match r.lines.first() {
                None => (i, 0, 0, false),
                Some(best) => {
                    let mate_in = if search::is_win(best.score) {
                        (search::MATE_VALUE - best.score + 1) / 2
                    } else {
                        0
                    };
                    let second = r.lines.get(1).map(|l| l.score);
                    let first_ok = p
                        .solution_moves
                        .split_whitespace()
                        .next()
                        .is_some_and(|m| m == move_to_icn(&best.mv));
                    (i, mate_in, best.score, first_ok && valid_attack(best.score, second))
                }
            };
            results.lock().unwrap().push(out);
            pb.inc(1);
        });
    }
    pb.finish_and_clear();

    let (mut mates, mut failed) = (0usize, 0usize);
    for (i, mate_in, score, ok) in results.into_inner().unwrap() {
        puzzles[i].mate_in = mate_in;
        puzzles[i].deep_eval = score;
        puzzles[i].verified = ok;
        if mate_in > 0 {
            mates += 1;
        }
        if !ok {
            failed += 1;
        }
    }
    (mates, failed)
}

/// Rebuilds every stored puzzle's line from its own position, with the smarter
/// defence. Cheap next to generation because the candidate hunt is already done:
/// only the cook re-runs. Lines usually get longer, never less forced.
fn recook_all(puzzles: &mut [PuzzleRecord], cfg: &Cfg) -> (usize, usize) {
    let mut by_variant: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    for (i, p) in puzzles.iter().enumerate() {
        by_variant.entry(p.variant.as_str()).or_default().push(i);
    }
    let groups: Vec<(Variant, Vec<usize>)> = by_variant
        .iter()
        .filter_map(|(name, idx)| {
            let v = Variant::parse(name);
            (v.to_str() == *name).then(|| (v, idx.clone()))
        })
        .collect();

    let pb = ProgressBar::new(puzzles.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  recook [{elapsed_precise}] [{bar:28.green/blue}] {pos}/{len} {per_sec}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    let out: Mutex<Vec<(usize, Option<Cooked>)>> = Mutex::new(Vec::new());

    for (v, idx) in groups {
        let b = v.get_default_bounds();
        apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);
        idx.par_iter().for_each(|&i| {
            let icn = &puzzles[i].position_icn;
            let mut st = GameState::new();
            st.setup_position_from_icn(icn);
            forget_history(&mut st);
            let winner = st.turn;
            let looking_for_mate = puzzles[i].mate_in > 0 || puzzles[i].ends_in_mate;
            let c = cook(&mut st, winner, looking_for_mate, cfg);
            out.lock().unwrap().push((i, c));
            pb.inc(1);
        });
    }
    pb.finish_and_clear();

    let (mut longer, mut kept) = (0usize, 0usize);
    for (i, c) in out.into_inner().unwrap() {
        match c {
            Some(c) => {
                if c.line.len() > puzzles[i].solution_plies {
                    longer += 1;
                }
                puzzles[i].solution_moves =
                    c.line.iter().map(move_to_icn).collect::<Vec<_>>().join(" ");
                puzzles[i].solution_plies = c.line.len();
                puzzles[i].ends_in_mate = c.ends_in_mate;
                puzzles[i].final_eval = c.final_score;
                puzzles[i].shallow_rank = c.shallow_rank;
                puzzles[i].only_move_margin = (c.root_margin * 1000.0).round() / 1000.0;
                puzzles[i].mean_defender_replies = if c.defender_replies.is_empty() {
                    0.0
                } else {
                    (c.defender_replies.iter().sum::<usize>() as f64
                        / c.defender_replies.len() as f64
                        * 10.0)
                        .round()
                        / 10.0
                };
            }
            // A re-cook can fail two ways: the reply it now prefers runs into a
            // repetition, or the root fails the only-move test at cook depth. Neither
            // says the puzzle is unsound -- the stored line already passed at depth
            // 20, which is deeper. So keep what we had rather than throwing it away.
            None => kept += 1,
        }
    }
    (longer, kept)
}

/// Recomputes the difficulty features that need no search -- forcing fraction,
/// mate flag, piece and fairy counts -- by replaying each stored solution, then
/// rebuilds `difficulty_raw`. Lets the rating model be retuned against an existing
/// CSV without redoing a single search. Bounds are global, so rows are grouped by
/// variant exactly as generation does.
/// Rewrites the `<halfmove>/<limit>` token to a fresh clock.
fn zero_move_clock(icn: &str) -> String {
    let mut toks: Vec<String> = icn.split_whitespace().map(str::to_string).collect();
    if toks.len() > 1
        && let Some((_, limit)) = toks[1].split_once('/')
    {
        toks[1] = format!("0/{limit}");
    }
    toks.join(" ")
}

fn refresh_features(puzzles: &mut [PuzzleRecord]) -> usize {
    let mut by_variant: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    for (i, p) in puzzles.iter().enumerate() {
        by_variant.entry(p.variant.as_str()).or_default().push(i);
    }
    let groups: Vec<(Variant, Vec<usize>)> = by_variant
        .iter()
        .filter_map(|(name, idx)| {
            let v = Variant::parse(name);
            (v.to_str() == *name).then(|| (v, idx.clone()))
        })
        .collect();

    let mut updated = 0;
    for (v, idx) in groups {
        let b = v.get_default_bounds();
        apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);
        for i in idx {
            let p = &mut puzzles[i];
            // Zero an inherited 50-move clock: the solver cannot know it, and it
            // leaks into the engine's evaluation of the puzzle position.
            p.position_icn = zero_move_clock(&p.position_icn);
            let sol: Vec<String> = p
                .solution_moves
                .split_whitespace()
                .map(str::to_string)
                .collect();
            // A `None` here means the line repeats a position, so the defender just
            // claims the draw: whatever it is, it is not a winning solution.
            let Some(bf) = board_features(&p.position_icn, &sol) else {
                p.verified = false;
                continue;
            };
            p.piece_count = bf.pieces;
            p.relevant_pieces = bf.relevant;
            p.root_moves = bf.root_moves;
            p.root_forcing = bf.root_forcing;
            p.fairy_count = (bf.fairy_used * 100.0).round() as usize;
            p.ends_in_mate = bf.ends_in_mate;
            p.forcing_pct = (bf.forcing * 100.0).round() as i32;

            // A position deep-verified as a forced mate is a mate puzzle even when
            // the recorded line stops short, so the themes have to say so.
            if p.mate_in > 0 {
                let mut t: BTreeSet<String> = p
                    .themes
                    .split(',')
                    .filter(|s| !s.is_empty() && *s != "crushing" && *s != "advantage")
                    .map(str::to_string)
                    .collect();
                t.insert("mate".into());
                t.retain(|s| !s.starts_with("mateIn"));
                t.insert(match p.mate_in {
                    1 => "mateIn1".into(),
                    2 => "mateIn2".into(),
                    3 => "mateIn3".into(),
                    4 => "mateIn4".into(),
                    _ => "mateIn5plus".to_string(),
                });
                p.themes = t.into_iter().collect::<Vec<_>>().join(",");
            }

            let feats = Features {
                shallow_rank: p.shallow_rank,
                depth_to_find: p.depth_to_find,
                margin: p.only_move_margin,
                plies: p.solution_plies,
                mean_replies: p.mean_defender_replies,
                quiet_key: p.themes.split(',').any(|t| t == "quietMove"),
                sacrifice: p.sacrifice_cp,
                travel: p.key_move_travel,
                remoteness: 0, // unused by the model
                pieces: bf.pieces,
                relevant: bf.relevant,
                fairy_used: bf.fairy_used,
                fairy_present: bf.fairy_present,
                forcing: bf.forcing,
                // A deep-verified forced mate counts even when the recorded line
                // stops short of delivering it.
                ends_in_mate: bf.ends_in_mate || p.mate_in > 0,
                mate_plies: if p.mate_in > 0 {
                    (p.mate_in as usize * 2).saturating_sub(1)
                } else {
                    0
                },
                root_moves: bf.root_moves,
                root_forcing: bf.root_forcing,
            };
            p.difficulty_raw = raw_difficulty(&feats);
            p.rating = puzzle_rating(&feats);
            updated += 1;
        }
    }
    updated
}

/// The rating is a puzzle's place in the difficulty ordering, so it can only be
/// set once every puzzle exists. Streamed rows carry rating 0 until this rewrites
/// the file; it is also exposed as `--rate-only` to finish an interrupted run.
fn finalize_ratings(path: &Path, cfg: &Cfg) -> Vec<PuzzleRecord> {
    let Ok(mut rdr) = csv::Reader::from_path(path) else {
        return Vec::new();
    };
    let mut puzzles: Vec<PuzzleRecord> = rdr.deserialize().filter_map(Result::ok).collect();
    if puzzles.is_empty() {
        return puzzles;
    }
    if cfg.refresh {
        for p in puzzles.iter_mut() {
            p.position_icn = zero_move_clock(&p.position_icn);
        }
    }
    if cfg.recook {
        let (longer, kept) = recook_all(&mut puzzles, cfg);
        println!("recook: {longer} lines extended, {kept} kept as previously verified");
    }
    if cfg.deep_verify {
        let (mates, failed) = deep_verify(&mut puzzles, cfg);
        println!(
            "deep verify at depth {}: {mates} forced mates, {failed} no longer uniquely best",
            cfg.deep_depth
        );
    }
    if cfg.refresh {
        let n = refresh_features(&mut puzzles);
        println!("recomputed features for {n}/{} puzzles", puzzles.len());
    }
    if cfg.deep_verify || cfg.recook {
        let before = puzzles.len();
        puzzles.retain(|p| p.verified);
        println!("dropped {} unsound or repeating puzzles", before - puzzles.len());
    }
    // Two candidates can rewind onto the same root, which the pre-rewind hash dedup
    // cannot see. Collapse them here, keeping the longest solution found.
    puzzles.sort_by(|a, b| {
        a.position_icn
            .cmp(&b.position_icn)
            .then(b.solution_plies.cmp(&a.solution_plies))
    });
    let before = puzzles.len();
    puzzles.dedup_by(|a, b| a.position_icn == b.position_icn);
    if puzzles.len() < before {
        println!("dropped {} duplicate positions", before - puzzles.len());
    }
    // Ratings are absolute, set per puzzle in `puzzle_rating`; this sort is only so
    // the file reads hardest-last.
    puzzles.sort_by_key(|p| p.rating);
    let tmp = path.with_extension("csv.tmp");
    {
        let mut w = csv::Writer::from_path(&tmp).expect("failed to open temp CSV");
        for p in &puzzles {
            w.serialize(p).expect("failed to serialize puzzle");
        }
        w.flush().expect("failed to flush temp CSV");
    }
    // Rename is atomic but fails outright if anything holds the file open (a
    // spreadsheet, typically), so fall back to writing through it in place.
    if let Err(e) = fs::rename(&tmp, path) {
        eprintln!("note: could not replace {} ({e}); writing in place", path.display());
        let mut w = csv::Writer::from_path(path).expect("failed to open output CSV");
        for p in &puzzles {
            w.serialize(p).expect("failed to serialize puzzle");
        }
        w.flush().expect("failed to flush output CSV");
        let _ = fs::remove_file(&tmp);
    }
    puzzles
}

fn move_to_icn(m: &Move) -> String {
    let mut s = format!("{},{}>{},{}", m.from.x, m.from.y, m.to.x, m.to.y);
    if let Some(p) = m.promotion {
        s.push('=');
        s.push_str(p.to_site_code());
    }
    s
}

// ---------------------------------------------------------------------------

/// Last line of defence before a puzzle is written: the position must be legal
/// (a side to move that can capture the enemy royal means the replay desynced),
/// every solution move must actually be legal where it is played, and a line
/// claiming mate must really end in one.
fn line_is_sound(icn: &str, line: &[Move], ends_in_mate: bool) -> bool {
    let mut st = GameState::new();
    st.setup_position_from_icn(icn);
    if let Some(royal) = opposing_royal(&st, st.turn)
        && apeiron::moves::is_square_attacked(&st.board, &royal, st.turn, &st.spatial_indices)
    {
        return false;
    }
    for m in line {
        if !legal_moves(&mut st).contains(m) {
            return false;
        }
        st.make_move(m);
    }
    !ends_in_mate || (legal_moves(&mut st).is_empty() && st.is_in_check())
}

/// True when the answer is just "take it back". The opponent traded on a square
/// and the solution recaptures there; not finding that is not a puzzle, it is not
/// knowing the rules. A recapture only survives when the solver still has to pick
/// *which* piece takes and the shallow-obvious choice is the wrong one.
/// Serialises the position itself, with no move list, so there is no question of
/// which ply the puzzle starts on. The promotion/bounds/win-condition tokens are
/// copied verbatim from the game's own starting ICN: they never change, and
/// rebuilding them risks a dialect mismatch.
fn position_to_icn(st: &GameState, start_icn: &str) -> String {
    let toks: Vec<&str> = start_icn.split_whitespace().collect();
    let middle = if toks.len() > 4 {
        toks[3..toks.len() - 1].join(" ")
    } else {
        String::new()
    };
    let turn = if st.turn == PlayerColor::White { "w" } else { "b" };
    let limit = st.game_rules.move_rule_limit.unwrap_or(100);

    let mut pieces: Vec<_> = st.board.iter().collect();
    pieces.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let pieces_str = pieces
        .iter()
        .map(|(x, y, p)| {
            let mut code = p.piece_type().to_site_code().to_string();
            if p.color() != PlayerColor::White {
                code = code.to_lowercase();
            }
            let mut ys = y.to_string();
            if st.has_special_right(&Coordinate::new(*x, *y)) {
                ys.push('+');
            }
            format!("{code}{x},{ys}")
        })
        .collect::<Vec<_>>()
        .join("|");

    let head = format!("{turn} {}/{limit} {}", st.halfmove_clock, st.fullmove_number);
    if middle.is_empty() {
        format!("{head} {pieces_str}")
    } else {
        format!("{head} {middle} {pieces_str}")
    }
}

/// The scan fires where the evaluation swing *shows up*, which can be several plies
/// into a combination that was already forced. Walk back two plies at a time --
/// same side to move -- while the earlier position is still an only-move win, so the
/// puzzle starts where the sequence starts rather than in the middle of it.
fn rewind_to_start(cand: &Candidate, winner: PlayerColor, cfg: &Cfg) -> usize {
    let mut ply = cand.ply;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(cfg.budget_ms);
    for _ in 0..MAX_REWIND {
        if std::time::Instant::now() >= deadline {
            break;
        }
        if ply < 2 {
            break;
        }
        let mut st = cand.game.state_at(ply - 2);
        if st.turn != winner {
            break;
        }
        let r = mpv(&mut st, cfg.verify_depth, cfg.cap_ms, 2);
        let Some(b) = r.lines.first() else { break };
        if !valid_attack(b.score, r.lines.get(1).map(|l| l.score)) {
            break;
        }
        ply -= 2;
    }
    ply
}

fn trivial_recapture(cand: &Candidate, ply: usize, key: &Move, solution_plies: usize) -> bool {
    let Some((last_to, last_was_capture)) = cand.game.move_info(ply) else {
        return false;
    };
    // Only genuine trades. Capturing a piece the opponent merely moved into reach
    // can be a real tactic; the obviousness gate below handles the easy ones.
    if !last_was_capture || key.to != last_to {
        return false;
    }
    let mut st = cand.game.state_at(ply);
    if !st.board.is_occupied(key.to.x, key.to.y) {
        return false; // not a capture at all
    }
    let takers = legal_moves(&mut st)
        .iter()
        .filter(|m| m.to == last_to)
        .count();
    // Purely structural, on purpose. Engine ordering is no guide here: a move can
    // top the shallow list merely for being a check and still be a fine puzzle.
    // What makes a recapture worthless is having no choice about it, or being the
    // whole answer -- they took, you took back, nothing else happened.
    takers < 2 || solution_plies == 1
}

fn solve(cand: &Candidate, variant: Variant, cfg: &Cfg) -> Option<PuzzleRecord> {
    let mut st = cand.game.state_at(cand.ply);
    let winner = st.turn;

    // stage 2a - cheap screen. Over 90% of candidates die here, so the evidence
    // that kills them should not cost a full-depth search. Thresholds are set well
    // clear of the real ones so a shallow misread cannot discard a real puzzle.
    {
        let s = mpv(&mut st, cfg.screen_depth, cfg.cap_ms, 2);
        let Some(b) = s.lines.first() else {
            rej::hit(rej::NO_LINES);
            return None;
        };
        // Only reject on evidence a deeper search cannot overturn: a runner-up that
        // already wins big keeps winning. Screening on the *best* move looking weak
        // would throw away precisely the deep sacrifices worth the most as puzzles.
        if let Some(sec) = s.lines.get(1).map(|l| l.score)
            && ((search::is_win(b.score) && search::is_win(sec)) || sec >= SCREEN_SECOND_CEIL)
        {
            rej::hit(rej::SCREENED);
            return None;
        }
    }

    // stage 2b - is there a single clearly-best winning move at all?
    let verify = mpv(&mut st, cfg.verify_depth, cfg.cap_ms, 2);
    let Some(best) = verify.lines.first() else {
        rej::hit(rej::NO_LINES);
        return None;
    };
    let looking_for_mate = search::is_win(best.score);
    let defensive = !looking_for_mate && best.score < WON_CP;
    let second = verify.lines.get(1).map(|l| l.score);
    if let Some(s) = second {
        rej::second(s, search::is_win(s));
    }
    if !valid_attack(best.score, second) {
        rej::hit(if best.score < HOLD_FLOOR {
            rej::WEAK
        } else {
            rej::NOT_ONLY
        });
        return None;
    }

    // stage 2c - rewind to where the forced sequence actually begins.
    let ply = rewind_to_start(cand, winner, cfg);
    let root_icn = cand.game.prefix(ply);

    // stage 3 - build the forced line.
    let mut cook_state = cand.game.state_at(ply);
    let Some(cooked) = cook(&mut cook_state, winner, looking_for_mate, cfg) else {
        rej::hit(rej::COOK_FAILED);
        return None;
    };
    if trivial_recapture(cand, ply, &cooked.line[0], cooked.line.len()) {
        rej::hit(rej::TRIVIAL);
        return None;
    }
    if !line_is_sound(&root_icn, &cooked.line, cooked.ends_in_mate) {
        rej::hit(rej::UNSOUND);
        return None;
    }

    // stage 4 - features.
    let key = cooked.line[0];
    let mut probe = cand.game.state_at(ply);
    let dtf = depth_to_find(&mut probe, key, cfg);
    rej::hit(rej::OK);

    let mut fs = cand.game.state_at(ply);
    let base_lead = material_lead(&fs, winner);
    let standalone = position_to_icn(&fs, &cand.game.start_icn);
    // Derived through the same path `--refresh` uses, so a later retune reproduces
    // these numbers exactly instead of drifting from them.
    let sol: Vec<String> = cooked.line.iter().map(move_to_icn).collect();
    let Some(bf) = board_features(&standalone, &sol) else {
        rej::hit(rej::TRIVIAL); // the line repeats; the defender just draws
        return None;
    };
    // Measured in the puzzle position, not after the line: this asks how far from
    // the enemy royal the solver has to look, which is what makes an unbounded
    // board hard to read.
    let remoteness = opposing_royal(&fs, winner)
        .map(|r| chebyshev(key.to, r))
        .unwrap_or(0);
    let quiet_key = {
        let cap = is_capture(&fs, &key);
        let undo = fs.make_move(&key);
        let gives_check = fs.is_in_check();
        fs.undo_move(&key, undo);
        !cap && !gives_check
    };
    let mut low = base_lead;
    for (i, m) in cooked.line.iter().enumerate() {
        fs.make_move(m);
        if i.is_multiple_of(2) {
            low = low.min(material_lead(&fs, winner));
        }
    }

    let feats = Features {
        shallow_rank: cooked.shallow_rank,
        depth_to_find: dtf,
        margin: cooked.root_margin,
        plies: cooked.line.len(),
        mean_replies: if cooked.defender_replies.is_empty() {
            0.0
        } else {
            cooked.defender_replies.iter().sum::<usize>() as f64
                / cooked.defender_replies.len() as f64
        },
        quiet_key,
        sacrifice: (base_lead - low).max(0),
        travel: chebyshev(key.from, key.to),
        remoteness,
        pieces: bf.pieces,
        relevant: bf.relevant,
        fairy_used: bf.fairy_used,
        fairy_present: bf.fairy_present,
        forcing: bf.forcing,
        ends_in_mate: bf.ends_in_mate,
        mate_plies: 0, // --deep-verify supplies the true distance
        root_moves: bf.root_moves,
        root_forcing: bf.root_forcing,
    };

    let bounded = apeiron::moves::get_world_size() < 1_000_000;
    let themes = detect_themes(
        &root_icn,
        &cooked.line,
        winner,
        &feats,
        cooked.ends_in_mate,
        cooked.final_score,
        bounded,
        defensive,
    );

    Some(PuzzleRecord {
        variant: variant.to_str().to_string(),
        position_icn: standalone,
        game_icn: root_icn,
        side_to_move: if winner == PlayerColor::White {
            "White".into()
        } else {
            "Black".into()
        },
        solution_moves: cooked
            .line
            .iter()
            .map(move_to_icn)
            .collect::<Vec<_>>()
            .join(" "),
        rating: puzzle_rating(&feats),
        rating_deviation: 500,
        themes: themes.into_iter().collect::<Vec<_>>().join(","),
        source: cand.source.as_str().to_string(),
        solution_plies: cooked.line.len(),
        ends_in_mate: cooked.ends_in_mate,
        final_eval: cooked.final_score,
        difficulty_raw: raw_difficulty(&feats),
        shallow_rank: feats.shallow_rank,
        depth_to_find: feats.depth_to_find,
        only_move_margin: (feats.margin * 1000.0).round() / 1000.0,
        mean_defender_replies: (feats.mean_replies * 10.0).round() / 10.0,
        sacrifice_cp: feats.sacrifice,
        key_move_travel: feats.travel,
        piece_count: feats.pieces,
        relevant_pieces: feats.relevant,
        root_moves: feats.root_moves,
        root_forcing: feats.root_forcing,
        fairy_count: (feats.fairy_used * 100.0).round() as usize,
        forcing_pct: (feats.forcing * 100.0).round() as i32,
        mate_in: 0,
        deep_eval: 0,
        verified: false,
        scan_eval: cand.ann_score,
    })
}

fn main() {
    let cfg = parse_args();
    if cfg.fresh {
        let _ = fs::remove_file(&cfg.out);
        let _ = fs::remove_file(progress_path(&cfg.out));
    }
    if cfg.rate_only {
        let puzzles = finalize_ratings(&cfg.out, &cfg);
        if puzzles.is_empty() {
            println!("nothing to rate in {}", cfg.out.display());
        } else {
            report(&puzzles, &cfg.out);
        }
        return;
    }
    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .expect("failed to build thread pool");
    }
    search::set_tt_size_mb(cfg.hash_mb);

    let recs = scan(&cfg);
    // Keyed by name because Variant is not Hash.
    let mut by_variant: FxHashMap<&'static str, (Variant, Vec<GameRec>)> = FxHashMap::default();
    let mut total_cands = 0;
    for r in recs {
        total_cands += r.cands.len();
        let v = r.variant;
        by_variant
            .entry(v.to_str())
            .or_insert_with(|| (v, Vec::new()))
            .1
            .push(r);
    }
    let mut variants: Vec<&'static str> = by_variant.keys().copied().collect();
    variants.sort_unstable();
    println!(
        "stage 0: {} candidate plies across {} variants",
        total_cands,
        variants.len()
    );

    let done = AtomicUsize::new(0);
    let prog = progress_path(&cfg.out);
    let already = if cfg.dry_run {
        FxHashSet::default()
    } else {
        load_progress(&prog)
    };
    if !already.is_empty() {
        println!("resuming: {} candidates already processed", already.len());
    }
    let (tx, rx) = mpsc::channel::<Emit>();
    let writer = if cfg.dry_run {
        drop(rx);
        None
    } else {
        Some(spawn_writer(cfg.out.clone(), prog.clone(), rx))
    };
    // Sender is Send but not Sync, and rayon needs shared access across workers.
    let tx = Mutex::new(tx);

    for name in &variants {
        let (v, games) = &by_variant[name];
        // World bounds are process-global, so a variant owns the process while
        // it runs. Parallelism stays inside this block.
        let b = v.get_default_bounds();
        apeiron::moves::set_world_bounds(b.0, b.1, b.2, b.3);

        let mut cands: Vec<Candidate> = games.par_iter().flat_map(replay).collect();
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        cands.retain(|c| seen.insert(c.hash));
        cands.truncate(cfg.per_variant);
        // Truncate first, then drop the resumed ones, so the cap always selects the
        // same candidate set whether or not this is a resume.
        cands.retain(|c| !already.contains(&c.hash));
        if cands.is_empty() {
            continue;
        }
        if cfg.dry_run {
            println!("  {:<22} {:>6} unique candidate positions", v.to_str(), cands.len());
            done.fetch_add(cands.len(), Ordering::Relaxed);
            continue;
        }

        let pb = ProgressBar::new(cands.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "  {msg:<22} [{elapsed_precise}] [{bar:28.green/blue}] {pos}/{len} {per_sec}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message(v.to_str().to_string());

        let hits = AtomicUsize::new(0);
        cands.par_iter().for_each(|c| {
            let r = solve(c, *v, &cfg);
            rej::src(c.source.idx(), r.is_some());
            let g = tx.lock().unwrap();
            if let Some(p) = r {
                hits.fetch_add(1, Ordering::Relaxed);
                let _ = g.send(Emit::Puzzle(Box::new(p)));
            }
            let _ = g.send(Emit::Done(c.hash));
            drop(g);
            pb.inc(1);
        });
        pb.finish_and_clear();
        println!(
            "  {:<22} {:>6} candidates -> {:>5} puzzles",
            v.to_str(),
            cands.len(),
            hits.load(Ordering::Relaxed)
        );
        done.fetch_add(cands.len(), Ordering::Relaxed);
    }

    drop(tx); // closes the channel so the writer can finish
    if let Some(h) = writer {
        let _ = h.join();
    }
    rej::report();
    if cfg.dry_run {
        println!(
            "dry run: {} candidate positions total",
            done.load(Ordering::Relaxed)
        );
        return;
    }

    let puzzles = finalize_ratings(&cfg.out, &cfg);
    if puzzles.is_empty() {
        println!("no puzzles produced");
        return;
    }
    report(&puzzles, &cfg.out);
}

fn report(puzzles: &[PuzzleRecord], out: &Path) {
    let mut by_variant: FxHashMap<&str, usize> = FxHashMap::default();
    let mut by_theme: FxHashMap<&str, usize> = FxHashMap::default();
    let mut bands = [0usize; 5];
    let mut mates = 0;
    for p in puzzles {
        *by_variant.entry(p.variant.as_str()).or_default() += 1;
        for t in p.themes.split(',') {
            *by_theme.entry(t).or_default() += 1;
        }
        let b = match p.rating {
            r if r < 1000 => 0,
            r if r < 1400 => 1,
            r if r < 1800 => 2,
            r if r < 2200 => 3,
            _ => 4,
        };
        bands[b] += 1;
        if p.ends_in_mate {
            mates += 1;
        }
    }
    println!("\n{} puzzles written to {}", puzzles.len(), out.display());
    println!("  mate puzzles: {mates}   tactical: {}", puzzles.len() - mates);
    println!(
        "  rating bands: <1000 {} | 1000-1399 {} | 1400-1799 {} | 1800-2199 {} | 2200+ {}",
        bands[0], bands[1], bands[2], bands[3], bands[4]
    );

    let mut vs: Vec<_> = by_variant.into_iter().collect();
    vs.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("  by variant:");
    for (v, n) in vs {
        println!("    {v:<22} {n:>5}");
    }
    let mut ts: Vec<_> = by_theme.into_iter().collect();
    ts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("  top themes:");
    for (t, n) in ts.iter().take(20) {
        println!("    {t:<22} {n:>5}");
    }
}
