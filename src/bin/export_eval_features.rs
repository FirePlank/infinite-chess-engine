//! Exports Stage-A eval-net training records by replaying the texel corpus
//! (`data_gen` JSONL) and the SPRT game archives (`games/sprt/*.json`), then
//! recomputing the CURRENT static eval and feature vector at every kept
//! position. Recorded evals are only ever targets, never inputs, so a corpus
//! played by any older engine version still trains today's residual.
//!
//! World bounds are process-global, so games are grouped by variant and each
//! group runs as its own parallel pass.

use apeiron::Variant;
use apeiron::board::PlayerColor;
use apeiron::eval_net::{FeatureCollector, NUM_FEATURES, feature_vector, schema_hash};
use apeiron::evaluation::{base, eval_kind::EvalKind, insufficient_material};
use apeiron::game::GameState;
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAGIC: &[u8; 8] = b"AEVDAT01";
const VERSION: u32 = 1;
/// Feature vector + static/teacher cp + flags + game id + ply, padded to 216.
const RECORD_SIZE: usize = NUM_FEATURES * 2 + 18;
const HEADER_SIZE: u64 = 36;
const MATE_FLOOR: i32 = apeiron::search::MATE_SCORE;

/// Games set up sequentially per parallel replay pass; bounds peak memory.
const SETUP_CHUNK: usize = 512;
static UNPARSED_MOVES: AtomicU64 = AtomicU64::new(0);
static MIRRORED: AtomicU64 = AtomicU64::new(0);
static MIRROR_REJECTED: AtomicU64 = AtomicU64::new(0);
static KEEP_KEYS: std::sync::OnceLock<HashSet<u64>> = std::sync::OnceLock::new();
static KEEP_HASHES: std::sync::OnceLock<HashSet<u64>> = std::sync::OnceLock::new();
static HASH_OUT: std::sync::OnceLock<Mutex<BufWriter<File>>> = std::sync::OnceLock::new();

/// FNV-style fold over the leading feature columns and the static eval; must
/// match `keys()` in `nnue/join_labels.py`.
fn record_key(x: &[i16], n: usize, static_white: i16) -> u64 {
    let prime = 0x0000_0100_0000_01B3u64;
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &v in &x[..n] {
        h = (h ^ v as u16 as u64).wrapping_mul(prime);
    }
    (h ^ static_white as u16 as u64).wrapping_mul(prime)
}
static EXCLUDED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
const SOURCE_TEXEL: u8 = 0;
const SOURCE_SPRT: u8 = 1;
/// Internal only: relabelling always rewrites the stored source to fixed-depth.
const SOURCE_HUMAN: u8 = 2;

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

#[derive(Parser, Debug)]
#[command(about = "Export Stage-A eval-net training records from game corpora")]
struct Cli {
    /// data_gen JSONL corpora (repeatable).
    #[arg(long)]
    texel: Vec<PathBuf>,
    /// Human games JSON (infinitechess.org export: array of {variant, result,
    /// termination, icn}). Carries no evals, so pair it with --relabel-depth.
    #[arg(long)]
    human: Option<PathBuf>,
    /// Comma-separated variants to drop from every source.
    #[arg(long, default_value = "Abundance")]
    exclude_variants: String,
    /// Fraction of eligible human-game positions to keep.
    #[arg(long, default_value_t = 0.05)]
    human_sample: f64,
    /// Directory of SPRT games*.json archives.
    #[arg(long)]
    sprt_dir: Option<PathBuf>,
    /// Only SPRT files whose name contains this substring.
    #[arg(long)]
    sprt_filter: Option<String>,
    /// Stop after this many SPRT files (0 = all).
    #[arg(long, default_value_t = 0)]
    sprt_max_files: usize,
    /// Fraction of eligible SPRT positions to keep (decorrelates plies).
    #[arg(long, default_value_t = 0.35)]
    sprt_sample: f64,
    /// Fraction of eligible texel positions to keep.
    #[arg(long, default_value_t = 1.0)]
    texel_sample: f64,
    /// Sign multiplier turning a recorded [%eval] into White-ahead cp. The
    /// archives store Black-ahead values, hence -1; the run prints the
    /// teacher/static correlation so a wrong sign is obvious.
    #[arg(long, default_value_t = -1)]
    sprt_eval_sign: i32,
    /// Skip positions before this ply.
    #[arg(long, default_value_t = 12)]
    min_ply: usize,
    /// Skip positions whose recorded teacher differs from the recomputed
    /// static eval by more than this (cp); crude tactical-noise filter.
    #[arg(long, default_value_t = 400)]
    quiet_tolerance: i32,
    /// Skip positions with |teacher| above this (cp).
    #[arg(long, default_value_t = 2000)]
    max_abs_cp: i32,
    #[arg(long, default_value = "nnue/eval_net_data.bin")]
    out: PathBuf,
    /// Re-label every kept position with a fixed-depth search of the CURRENT engine
    /// instead of the recorded eval (0 = keep recorded). Records are then tagged as
    /// fixed-depth (source 0).
    #[arg(long, default_value_t = 0)]
    relabel_depth: usize,
    /// Play 1..=N random legal moves off each sampled game position before labelling
    /// (needs --relabel-depth, since recorded evals no longer apply).
    #[arg(long, default_value_t = 0)]
    perturb: usize,
    /// Continue an interrupted export: reopen `--out`, restore the record count and
    /// game-id counter from `<out>.progress`, and skip archive files already done.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Also write every kept position colour-mirrored (colours swapped, board
    /// reflected between the promotion ranks, labels negated) so the net learns
    /// White and Black symmetrically.
    #[arg(long, default_value_t = false)]
    mirror: bool,
    /// Keep only positions whose key (see `record_key`) is in this file of LE u64s;
    /// used to re-export an earlier sample in a newer layout.
    #[arg(long)]
    keep_keys: Option<PathBuf>,
    /// Write each record's Zobrist hash (LE u64, record order) to this sidecar, so
    /// labels can be carried across an eval change that alters every feature key.
    #[arg(long)]
    hash_out: Option<PathBuf>,
    /// Keep only positions whose Zobrist hash is in this file of LE u64s.
    #[arg(long)]
    keep_hashes: Option<PathBuf>,
    /// Leading feature columns the key covers (the older layout's width).
    #[arg(long, default_value_t = 129)]
    key_columns: usize,
    /// Hard time cap per re-label search in ms.
    #[arg(long, default_value_t = 3000)]
    relabel_ms: u64,
    #[arg(long, default_value_t = 8)]
    tt_mb: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

// ---------------------------------------------------------------------------
// corpus records
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TexelPosition {
    ply: usize,
    score: i32,
    #[serde(default)]
    hmc: u32,
    #[serde(default)]
    cap: bool,
    #[serde(default)]
    promo: bool,
    #[serde(default)]
    chk: bool,
    #[serde(default)]
    quiet: bool,
}

#[derive(Deserialize)]
struct TexelGame {
    variant: String,
    wdl: f32,
    start_icn: String,
    moves: Vec<String>,
    positions: Vec<TexelPosition>,
}

/// One replayable game with a White-ahead teacher score per ply (None where
/// the engine reported a mate or nothing).
struct Game {
    variant: String,
    /// White's result: 1.0, 0.5, 0.0.
    wdl: f32,
    start_icn: String,
    moves: Vec<String>,
    teacher: Vec<Option<i32>>,
    /// Plies excluded up front (texel's non-quiet flag, captures, checks).
    skip: Vec<bool>,
    source: u8,
}

fn canon(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn variant_id(name: &str) -> u8 {
    let want = canon(name);
    ALL_VARIANTS
        .iter()
        .position(|v| canon(v.to_str()) == want)
        .map_or(255, |i| i as u8)
}

fn parse_move(mv: &str) -> Option<(i64, i64, i64, i64, Option<String>)> {
    let (coords, promo) = match mv.split_once('=') {
        Some((c, p)) => (c, Some(p.to_lowercase())),
        None => (mv, None),
    };
    let (from, to) = coords.split_once('>')?;
    let mut fp = from.split(',');
    let mut tp = to.split(',');
    let fx = fp.next()?.trim().parse().ok()?;
    let fy = fp.next()?.trim().parse().ok()?;
    let tx = tp.next()?.trim().parse().ok()?;
    let ty = tp.next()?.trim().parse().ok()?;
    Some((fx, fy, tx, ty, promo))
}

#[derive(Deserialize)]
struct HumanGame {
    variant: String,
    result: String,
    #[serde(default)]
    termination: String,
    icn: String,
}

/// Human games start from the variant's standard position; the ICN only carries
/// headers, the side/clock prefix and the move list.
fn human_to_game(h: HumanGame) -> Option<Game> {
    let term = h.termination.to_lowercase();
    if term.contains("disconnect") || term.contains("abort") || term.contains("abandon") {
        return None;
    }
    let wdl = match h.result.as_str() {
        "1-0" => 1.0,
        "0-1" => 0.0,
        "1/2-1/2" => 0.5,
        _ => return None,
    };
    let variant = ALL_VARIANTS
        .iter()
        .find(|v| canon(v.to_str()) == canon(&h.variant))?;
    let blob = h.icn.split_whitespace().find(|t| t.contains('>'))?;
    let moves: Vec<String> = blob.split('|').map(|m| m.trim().to_string()).collect();
    if moves.len() < 20 {
        return None;
    }
    let n = moves.len();
    Some(Game {
        variant: variant.to_str().to_string(),
        wdl,
        start_icn: format!("[Variant \"{}\"] {}", variant.to_str(), variant.starting_icn()),
        moves,
        // Placeholder teacher: only --relabel-depth gives these positions a label.
        teacher: vec![Some(0); n],
        skip: vec![false; n],
        source: SOURCE_HUMAN,
    })
}

fn texel_to_game(t: TexelGame) -> Game {
    let n = t.moves.len();
    let mut teacher = vec![None; n];
    let mut skip = vec![true; n];
    for p in &t.positions {
        if p.ply < n && p.score.abs() < MATE_FLOOR {
            teacher[p.ply] = Some(p.score);
            skip[p.ply] = !p.quiet || p.cap || p.promo || p.chk || p.hmc >= 40;
        }
    }
    Game {
        variant: t.variant,
        wdl: t.wdl,
        start_icn: t.start_icn,
        moves: t.moves,
        teacher,
        skip,
        source: SOURCE_TEXEL,
    }
}

/// Parses one SPRT archive entry: `[Tag "v"]... <position ICN> <move blob>`.
/// The `{[%clk ..] [%eval ..]}` comments contain spaces, so the body is found
/// by walking the leading tags and then splitting at the first `>` token.
fn parse_sprt_game(s: &str, eval_sign: i32) -> Option<Game> {
    let mut rest = s.trim_start();
    let mut tags: HashMap<String, String> = HashMap::new();
    while let Some(after) = rest.strip_prefix('[') {
        let key_end = after.find(' ')?;
        let key = &after[..key_end];
        let q1 = after.find('"')?;
        let q2 = q1 + 1 + after[q1 + 1..].find('"')?;
        let close = q2 + 1 + after[q2 + 1..].find(']')?;
        tags.insert(key.to_string(), after[q1 + 1..q2].to_string());
        rest = after[close + 1..].trim_start();
    }
    let variant = tags.get("Variant")?.clone();
    let wdl = match tags.get("Result")?.as_str() {
        "1-0" => 1.0,
        "0-1" => 0.0,
        "1/2-1/2" => 0.5,
        _ => return None,
    };
    if let Some(term) = tags.get("Termination") {
        let t = term.to_lowercase();
        if t.contains("time") || t.contains("illegal") || t.contains("failure") {
            return None;
        }
    }

    // Position tokens never contain '>'; the first one that does starts the moves.
    let body_start = rest
        .split_whitespace()
        .find(|tok| tok.contains('>'))
        .map(|tok| tok.as_ptr() as usize - rest.as_ptr() as usize);
    let (position, blob) = match body_start {
        Some(off) => (rest[..off].trim(), rest[off..].trim()),
        None => return None,
    };
    let start_icn = format!("[Variant \"{variant}\"] {position}");

    let mut moves = Vec::new();
    let mut teacher = Vec::new();
    for entry in blob.split('|') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (mv, comment) = match entry.split_once('{') {
            Some((m, c)) => (m.trim(), c.trim_end_matches('}')),
            None => (entry, ""),
        };
        let mut t = None;
        if let Some(pos) = comment.find("[%eval ") {
            let after = &comment[pos + 7..];
            if let Some(end) = after.find(']')
                && let Ok(v) = after[..end].trim().parse::<f64>()
            {
                t = Some(((v * 100.0).round() as i32) * eval_sign);
            }
        }
        moves.push(mv.to_string());
        teacher.push(t);
    }
    if moves.is_empty() {
        return None;
    }
    let n = moves.len();
    Some(Game {
        variant,
        wdl,
        start_icn,
        moves,
        teacher,
        skip: vec![false; n],
        source: SOURCE_SPRT,
    })
}

// ---------------------------------------------------------------------------
// replay + record building
// ---------------------------------------------------------------------------

struct Stats {
    kept: AtomicU64,
    games: AtomicU64,
    dup: AtomicU64,
    /// Sums for the teacher/static correlation, one per source.
    corr: Mutex<[[f64; 5]; 2]>,
    per_variant: Mutex<HashMap<String, u64>>,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Plays `steps` uniformly random legal moves from `g`; None if a side runs out.
fn perturb(g: &GameState, steps: usize, seed: u64) -> Option<GameState> {
    let mut pos = g.clone();
    let mut rng = seed;
    for _ in 0..steps {
        let mut list = apeiron::moves::MoveList::new();
        pos.get_pseudo_legal_moves_into(&mut list);
        let n = list.len();
        if n == 0 {
            return None;
        }
        rng = splitmix(rng);
        let start = (rng % n as u64) as usize;
        let mut played = false;
        for k in 0..n {
            let m = list[(start + k) % n];
            let undo = pos.make_move(&m);
            if pos.is_move_illegal() {
                pos.undo_move(&m, undo);
            } else {
                played = true;
                break;
            }
        }
        if !played {
            return None;
        }
    }
    Some(pos)
}

/// Deterministic per-(game, ply) sample in [0, 1).
fn unit_interval(seed: u64) -> f64 {
    (splitmix(seed) >> 11) as f64 / (1u64 << 53) as f64
}

fn replay(
    game: &Game,
    mut g: GameState,
    mut mirror: Option<(GameState, i64)>,
    game_id: u32,
    cli: &Cli,
    seen: &[Mutex<HashSet<u64>>],
    stats: &Stats,
    out: &mut Vec<u8>,
    hashes: &mut Vec<u8>,
) {
    if g.eval_kind != EvalKind::Generic {
        return;
    }
    let sample = match game.source {
        SOURCE_TEXEL => cli.texel_sample,
        SOURCE_HUMAN => cli.human_sample,
        _ => cli.sprt_sample,
    };
    let vid = variant_id(&game.variant);
    stats.games.fetch_add(1, Ordering::Relaxed);
    let mut kept_here = 0u64;
    let mut corr = [0f64; 5];

    for (ply, mv) in game.moves.iter().enumerate() {
        let Some((fx, fy, tx, ty, promo)) = parse_move(mv) else {
            // Coordinates beyond i64 (possible in site games) end the replay here.
            UNPARSED_MOVES.fetch_add(1, Ordering::Relaxed);
            break;
        };
        let eligible = ply >= cli.min_ply
            && !game.skip[ply]
            && game.teacher[ply].is_some_and(|t| t.abs() < cli.max_abs_cp)
            && g.halfmove_clock < 40
            && (g.white_piece_count + g.black_piece_count) >= 4
            // The played move must be quiet: no capture, no promotion.
            && promo.is_none()
            && g.board.get_piece(tx, ty).is_none()
            && (sample >= 1.0 || unit_interval(((game_id as u64) << 20) | ply as u64) < sample);
        if eligible && !g.is_in_check() && !insufficient_material::evaluate_insufficient_material(&g)
        {
            // With --perturb, step off the game line by a few random legal moves, so the
            // net also trains on the unbalanced positions the search evaluates.
            let pos = if cli.perturb > 0 {
                let steps = 1 + (splitmix(((game_id as u64) << 24) ^ ply as u64 ^ 0xA5A5) % cli.perturb as u64) as usize;
                match perturb(&g, steps, ((game_id as u64) << 20) | ply as u64) {
                    Some(p) => p,
                    None => {
                        advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
                        continue;
                    }
                }
            } else {
                g.clone()
            };
            if cli.perturb > 0 && (pos.is_in_check() || insufficient_material::evaluate_insufficient_material(&pos)) {
                advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
                continue;
            }
            // Positions where the engine skips the net are never trained on.
            if base::net_off(&pos) {
                advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
                continue;
            }
            let mut teacher = game.teacher[ply].unwrap();
            let mut source = game.source;
            if cli.relabel_depth > 0 {
                let mut gs = pos.clone();
                let Some((bm, score, _)) = apeiron::search::get_best_move(
                    &mut gs,
                    cli.relabel_depth,
                    cli.relabel_ms as u128,
                    true,
                    false,
                ) else {
                    advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
                    continue;
                };
                // Same quiet definition as data_gen: the chosen move must not capture
                // or promote, and the score must be a real evaluation.
                if score.abs() >= MATE_FLOOR
                    || bm.promotion.is_some()
                    || pos.board.get_piece(bm.to.x, bm.to.y).is_some()
                {
                    advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
                    continue;
                }
                teacher = if pos.turn == PlayerColor::Black { -score } else { score };
                source = SOURCE_TEXEL;
            }
            let mut fc = FeatureCollector::default();
            let stm_score = base::evaluate_inner_traced(&pos, &mut fc);
            let static_white = if pos.turn == PlayerColor::Black {
                -stm_score
            } else {
                stm_score
            };
            if (teacher - static_white).abs() <= cli.quiet_tolerance {
                let shard = &seen[(pos.hash % seen.len() as u64) as usize];
                let x = feature_vector(&pos, &fc);
                let keep = KEEP_KEYS.get().is_none_or(|k| {
                    let st = static_white.clamp(-20000, 20000) as i16;
                    k.contains(&record_key(&x, cli.key_columns, st))
                }) && KEEP_HASHES.get().is_none_or(|k| k.contains(&pos.hash));
                let fresh = keep && shard.lock().unwrap().insert(pos.hash);
                if fresh {
                    for f in x {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                    out.extend_from_slice(&(static_white.clamp(-20000, 20000) as i16).to_le_bytes());
                    out.extend_from_slice(&(teacher.clamp(-20000, 20000) as i16).to_le_bytes());
                    out.push(if game.wdl > 0.75 {
                        2
                    } else if game.wdl > 0.25 {
                        1
                    } else {
                        0
                    });
                    out.push(pos.turn as u8);
                    out.push(vid);
                    out.push(source);
                    out.push(fc.inputs.phase.clamp(0, 255) as u8);
                    out.push(0);
                    out.extend_from_slice(&game_id.to_le_bytes());
                    out.extend_from_slice(&(ply.min(65535) as u16).to_le_bytes());
                    out.extend_from_slice(&[0u8; 2]);
                    hashes.extend_from_slice(&pos.hash.to_le_bytes());
                    kept_here += 1;
                    if cli.perturb == 0
                        && let Some((m, _)) = mirror.as_ref()
                    {
                        let mut mfc = FeatureCollector::default();
                        let m_stm = base::evaluate_inner_traced(m, &mut mfc);
                        let m_static = if m.turn == PlayerColor::Black { -m_stm } else { m_stm };
                        // A sound mirror scores exactly the negated static eval.
                        if m.eval_kind == EvalKind::Generic && m_static == -static_white {
                            for f in feature_vector(m, &mfc) {
                                out.extend_from_slice(&f.to_le_bytes());
                            }
                            out.extend_from_slice(&(m_static.clamp(-20000, 20000) as i16).to_le_bytes());
                            out.extend_from_slice(&((-teacher).clamp(-20000, 20000) as i16).to_le_bytes());
                            out.push(if game.wdl > 0.75 { 0 } else if game.wdl > 0.25 { 1 } else { 2 });
                            out.push(m.turn as u8);
                            out.push(vid);
                            out.push(source);
                            out.push(mfc.inputs.phase.clamp(0, 255) as u8);
                            out.push(0);
                            out.extend_from_slice(&game_id.to_le_bytes());
                            out.extend_from_slice(&(ply.min(65535) as u16).to_le_bytes());
                            out.extend_from_slice(&[0u8; 2]);
                            hashes.extend_from_slice(&m.hash.to_le_bytes());
                            kept_here += 1;
                            MIRRORED.fetch_add(1, Ordering::Relaxed);
                        } else {
                            MIRROR_REJECTED.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let (t, s) = (teacher as f64, static_white as f64);
                    let _ = source;
                    corr[0] += 1.0;
                    corr[1] += t;
                    corr[2] += s;
                    corr[3] += t * s;
                    corr[4] += t * t;
                } else {
                    stats.dup.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        advance(&mut g, &mut mirror, (fx, fy, tx, ty), promo.as_deref());
    }

    if kept_here > 0 {
        stats.kept.fetch_add(kept_here, Ordering::Relaxed);
        let mut c = stats.corr.lock().unwrap();
        let row = &mut c[usize::from(game.source != SOURCE_TEXEL)];
        for k in 0..5 {
            row[k] += corr[k];
        }
        *stats
            .per_variant
            .lock()
            .unwrap()
            .entry(game.variant.clone())
            .or_default() += kept_here;
    }
}

/// Runs one bounds-homogeneous group in parallel and appends its records.
fn run_group(
    games: &[Game],
    first_id: u32,
    cli: &Cli,
    stats: &Stats,
    writer: &Mutex<BufWriter<File>>,
) {
    if games.is_empty() {
        return;
    }
    // `setup_position_from_icn` resets the process-global bounds to unbounded
    // before applying the ICN's token, so setups must never overlap a replay:
    // set up a chunk sequentially, then replay it in parallel.
    let seen: Vec<Mutex<HashSet<u64>>> = (0..64).map(|_| Mutex::new(HashSet::new())).collect();
    for (ci, chunk) in games.chunks(SETUP_CHUNK).enumerate() {
        let states: Vec<(GameState, Option<(GameState, i64)>)> = chunk
            .iter()
            .map(|game| {
                let mut g = GameState::new();
                g.setup_position_from_icn(&game.start_icn);
                g.recompute_piece_counts();
                g.recompute_hash();
                // The mirror replays the whole game reflected, so starting squares and
                // special rights match; setup stays sequential for the global bounds.
                let m = if cli.mirror {
                    mirror_icn(&g, &game.start_icn).map(|icn| {
                        let mut m = GameState::new();
                        m.setup_position_from_icn(&icn);
                        m.recompute_piece_counts();
                        m.recompute_hash();
                        (m, g.white_promo_rank + g.black_promo_rank)
                    })
                } else {
                    None
                };
                // Setting up the mirror may leave its own bounds; restore the original's.
                if m.is_some() {
                    let mut again = GameState::new();
                    again.setup_position_from_icn(&game.start_icn);
                }
                (g, m)
            })
            .collect();
        let base_id = first_id + (ci * SETUP_CHUNK) as u32;
        let outputs: Vec<(Vec<u8>, Vec<u8>)> = chunk
            .par_iter()
            .zip(states.into_par_iter())
            .enumerate()
            .map(|(i, (game, (g, m)))| {
                let (mut out, mut hashes) = (Vec::new(), Vec::new());
                replay(game, g, m, base_id + i as u32, cli, &seen, stats, &mut out, &mut hashes);
                (out, hashes)
            })
            .collect();
        let mut w = writer.lock().unwrap();
        let mut hw = HASH_OUT.get().map(|h| h.lock().unwrap());
        for (c, h) in outputs {
            w.write_all(&c).unwrap();
            if let Some(hw) = hw.as_mut() {
                hw.write_all(&h).unwrap();
            }
        }
    }
}

/// ICN of `g` with colours swapped and the board reflected across the midline of the
/// two promotion ranks. None when the start ICN carries anything this cannot mirror
/// exactly (asymmetric bounds, win conditions, unknown tokens) or ranks are missing.
fn mirror_icn(g: &GameState, start_icn: &str) -> Option<String> {
    if g.white_promo_rank == i64::MIN || g.black_promo_rank == i64::MAX {
        return None;
    }
    let s = g.white_promo_rank + g.black_promo_rank;
    let body = start_icn.rsplit(']').next()?.trim();
    let toks: Vec<&str> = body.split_whitespace().collect();
    let mut middle = Vec::new();
    // Skip turn, clock and fullmove; stop at the piece list.
    for t in toks.iter().skip(3) {
        if t.contains('>') || (t.contains(',') && t.chars().any(|c| c.is_ascii_alphabetic()) && !t.starts_with('(')) {
            break;
        }
        if let Some(inner) = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
            let (w, b) = inner.split_once('|')?;
            let flip = |side: &str| -> Option<String> {
                let (ranks, types) = match side.split_once(';') {
                    Some((r, ty)) => (r, Some(ty)),
                    None => (side, None),
                };
                let rs: Option<Vec<String>> = ranks
                    .split(',')
                    .filter(|r| !r.is_empty())
                    .map(|r| r.parse::<i64>().ok().map(|v| (s - v).to_string()))
                    .collect();
                let rs = rs?.join(",");
                Some(match types {
                    Some(ty) => format!("{rs};{ty}"),
                    None => rs,
                })
            };
            middle.push(format!("({}|{})", flip(b)?, flip(w)?));
        } else if t.split(',').count() == 4 && t.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '-') {
            let v: Vec<i64> = t.split(',').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            // Only a vertically symmetric border keeps the process-global bounds identical.
            if v[2] + v[3] != s && !(v[2] == -v[3] && v[3] >= 1_000_000) {
                return None;
            }
            middle.push(t.to_string());
        } else {
            return None;
        }
    }
    let mut pieces: Vec<String> = Vec::new();
    for (x, y, p) in g.board.iter() {
        let code = p.piece_type().to_site_code();
        let code = match p.color() {
            PlayerColor::White => code.to_lowercase(),
            PlayerColor::Black => code.to_uppercase(),
            PlayerColor::Neutral => code.to_string(),
        };
        let plus = if g.has_special_right(&apeiron::board::Coordinate::new(x, y)) { "+" } else { "" };
        pieces.push(format!("{code}{x},{}{plus}", s - y));
    }
    let turn = if g.turn == PlayerColor::White { "b" } else { "w" };
    let limit = g.game_rules.move_rule_limit.unwrap_or(100);
    Some(format!(
        "{turn} {}/{limit} {} {} {}",
        g.halfmove_clock,
        g.fullmove_number,
        middle.join(" "),
        pieces.join("|")
    ))
}

/// Plays one game move, and its reflection on the mirrored game when there is one.
fn advance(
    g: &mut GameState,
    mirror: &mut Option<(GameState, i64)>,
    (fx, fy, tx, ty): (i64, i64, i64, i64),
    promo: Option<&str>,
) {
    g.make_move_coords(fx, fy, tx, ty, promo);
    if let Some((m, s)) = mirror.as_mut() {
        m.make_move_coords(fx, *s - fy, tx, *s - ty, promo);
    }
}

/// Resume state kept beside the output file.
struct Progress {
    kept: u64,
    next_id: u32,
    sources_done: bool,
    done: HashSet<String>,
}

impl Progress {
    fn load(path: &PathBuf) -> Option<Progress> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut p = Progress { kept: 0, next_id: 1, sources_done: false, done: HashSet::new() };
        for line in text.lines() {
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            match k {
                "kept" => p.kept = v.parse().ok()?,
                "next_id" => p.next_id = v.parse().ok()?,
                "sources_done" => p.sources_done = true,
                "done" => {
                    p.done.insert(v.to_string());
                }
                _ => {}
            }
        }
        Some(p)
    }
}

/// Flushes every record so far, patches the header count, and appends `event`
/// plus the counters to the progress file, so a kill loses at most one archive file.
fn checkpoint(
    writer: &Mutex<BufWriter<File>>,
    stats: &Stats,
    next_id: u32,
    progress: &PathBuf,
    event: &str,
) {
    let mut w = writer.lock().unwrap();
    w.flush().unwrap();
    let kept = stats.kept.load(Ordering::Relaxed);
    let f = w.get_mut();
    f.seek(SeekFrom::Start(HEADER_SIZE - 8)).unwrap();
    f.write_all(&kept.to_le_bytes()).unwrap();
    f.seek(SeekFrom::End(0)).unwrap();
    f.sync_data().unwrap();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(progress).unwrap();
    writeln!(log, "{event}
kept {kept}
next_id {next_id}").unwrap();
}

fn group_by_variant(games: Vec<Game>) -> Vec<Vec<Game>> {
    let excluded: Vec<String> = EXCLUDED.get().map_or_else(Vec::new, |v| v.clone());
    let mut map: HashMap<String, Vec<Game>> = HashMap::new();
    for g in games {
        if excluded.contains(&canon(&g.variant)) {
            continue;
        }
        map.entry(canon(&g.variant)).or_default().push(g);
    }
    let mut groups: Vec<Vec<Game>> = map.into_values().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    groups
}

fn main() {
    let cli = Cli::parse();
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .unwrap();
    }
    if let Some(dir) = cli.out.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    apeiron::search::set_tt_size_mb(cli.tt_mb);
    if let Some(path) = &cli.keep_hashes {
        let bytes = std::fs::read(path).unwrap();
        let set: HashSet<u64> =
            bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        eprintln!("keep-hashes: {} hashes", set.len());
        let _ = KEEP_HASHES.set(set);
    }
    if let Some(path) = &cli.hash_out {
        assert!(!cli.resume, "--hash-out cannot resume: the sidecar would lose alignment");
        let f = File::create(path).unwrap();
        let _ = HASH_OUT.set(Mutex::new(BufWriter::new(f)));
    }
    if let Some(path) = &cli.keep_keys {
        let bytes = std::fs::read(path).unwrap();
        let set: HashSet<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        eprintln!("keep-keys: {} keys", set.len());
        let _ = KEEP_KEYS.set(set);
    }
    let _ = EXCLUDED.set(
        cli.exclude_variants
            .split(',')
            .filter(|v| !v.trim().is_empty())
            .map(canon)
            .collect(),
    );
    let progress_path = PathBuf::from(format!("{}.progress", cli.out.display()));
    let prior = if cli.resume { Progress::load(&progress_path) } else { None };
    let rec = RECORD_SIZE;
    let file = if let Some(p) = &prior {
        let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&cli.out).unwrap();
        // Drop any partial record written after the last completed file.
        f.set_len(HEADER_SIZE + p.kept * rec as u64).unwrap();
        f.seek(SeekFrom::End(0)).unwrap();
        eprintln!("resume: {} records, {} archive files already done", p.kept, p.done.len());
        f
    } else {
        let mut f = File::create(&cli.out).unwrap();
        f.write_all(MAGIC).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&(NUM_FEATURES as u32).to_le_bytes()).unwrap();
        f.write_all(&schema_hash().to_le_bytes()).unwrap();
        f.write_all(&(rec as u32).to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        assert_eq!(f.stream_position().unwrap(), HEADER_SIZE);
        let _ = std::fs::remove_file(&progress_path);
        f
    };
    let writer = Mutex::new(BufWriter::with_capacity(1 << 24, file));

    let stats = Stats {
        kept: AtomicU64::new(0),
        games: AtomicU64::new(0),
        dup: AtomicU64::new(0),
        corr: Mutex::new([[0.0; 5]; 2]),
        per_variant: Mutex::new(HashMap::new()),
    };
    let start = Instant::now();
    let mut next_id: u32 = prior.as_ref().map_or(1, |p| p.next_id);
    if let Some(p) = &prior {
        stats.kept.store(p.kept, Ordering::Relaxed);
    }
    // Texel and human sources are fast and run first; a resume past them skips both.
    let resumed_past_sources = prior.as_ref().is_some_and(|p| p.sources_done);
    let texel_sources: &[PathBuf] = if resumed_past_sources { &[] } else { &cli.texel };

    for path in texel_sources {
        let t0 = Instant::now();
        let reader = BufReader::new(File::open(path).unwrap());
        let games: Vec<Game> = reader
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str::<TexelGame>(&l).ok())
            .map(texel_to_game)
            .collect();
        let n = games.len();
        for group in group_by_variant(games) {
            run_group(&group, next_id, &cli, &stats, &writer);
            next_id += group.len() as u32;
        }
        eprintln!(
            "[texel] {} : {} games, {} records total ({:.1}s)",
            path.display(),
            n,
            stats.kept.load(Ordering::Relaxed),
            t0.elapsed().as_secs_f64()
        );
    }

    if let Some(path) = cli.human.as_ref().filter(|_| !resumed_past_sources) {
        assert!(
            cli.relabel_depth > 0 || cli.keep_keys.is_some() || cli.keep_hashes.is_some(),
            "--human needs --relabel-depth (or --keep-keys to re-export labelled positions)"
        );
        let t0 = Instant::now();
        let text = std::fs::read_to_string(path).unwrap();
        let raw: Vec<HumanGame> = serde_json::from_str(&text).unwrap();
        let games: Vec<Game> = raw.into_iter().filter_map(human_to_game).collect();
        let n = games.len();
        for group in group_by_variant(games) {
            run_group(&group, next_id, &cli, &stats, &writer);
            next_id += group.len() as u32;
        }
        eprintln!("[human] {} games, {} records total ({:.1}s)", n, stats.kept.load(Ordering::Relaxed), t0.elapsed().as_secs_f64());
    }

    if let Some(dir) = &cli.sprt_dir {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p.file_name().unwrap().to_string_lossy();
                name.contains("games") && name.ends_with(".json")
            })
            .filter(|p| {
                cli.sprt_filter
                    .as_ref()
                    .is_none_or(|f| p.to_string_lossy().contains(f.as_str()))
            })
            .collect();
        files.sort();
        if cli.sprt_max_files > 0 {
            files.truncate(cli.sprt_max_files);
        }
        let total = files.len();
        if !resumed_past_sources {
            checkpoint(&writer, &stats, next_id, &progress_path, "sources_done");
        }
        let done: HashSet<String> = prior.as_ref().map_or_else(HashSet::new, |p| p.done.clone());
        for (fi, path) in files.iter().enumerate() {
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            if done.contains(&fname) {
                continue;
            }
            let t0 = Instant::now();
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let Ok(entries) = serde_json::from_str::<Vec<String>>(&text) else {
                eprintln!("[sprt] {} : not a JSON string array, skipped", path.display());
                continue;
            };
            drop(text);
            let games: Vec<Game> = entries
                .iter()
                .filter_map(|s| parse_sprt_game(s, cli.sprt_eval_sign))
                .collect();
            let n = games.len();
            for group in group_by_variant(games) {
                run_group(&group, next_id, &cli, &stats, &writer);
                next_id += group.len() as u32;
            }
            eprintln!(
                "[sprt {}/{}] {} : {} games, {} records total ({:.1}s)",
                fi + 1,
                total,
                path.file_name().unwrap().to_string_lossy(),
                n,
                stats.kept.load(Ordering::Relaxed),
                t0.elapsed().as_secs_f64()
            );
            checkpoint(&writer, &stats, next_id, &progress_path, &format!("done {fname}"));
        }
    }

    // Patch the record count into the header.
    let kept = stats.kept.load(Ordering::Relaxed);
    let mut w = writer.into_inner().unwrap();
    w.flush().unwrap();
    let mut file = w.into_inner().unwrap();
    file.seek(SeekFrom::Start(HEADER_SIZE - 8)).unwrap();
    file.write_all(&kept.to_le_bytes()).unwrap();
    file.flush().unwrap();
    if let Some(h) = HASH_OUT.get() {
        h.lock().unwrap().flush().unwrap();
    }

    eprintln!(
        "done: {} games replayed, {} records, {} zobrist dups skipped, {:.1}s",
        stats.games.load(Ordering::Relaxed),
        kept,
        stats.dup.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    eprintln!("replays cut short by an unparseable move: {}", UNPARSED_MOVES.load(Ordering::Relaxed));
    if cli.mirror {
        eprintln!(
            "mirror: {} twins written, {} rejected (static eval not exactly negated)",
            MIRRORED.load(Ordering::Relaxed),
            MIRROR_REJECTED.load(Ordering::Relaxed)
        );
    }
    let corr = stats.corr.lock().unwrap();
    for (src, name) in [(SOURCE_TEXEL, "texel"), (SOURCE_SPRT, "sprt")] {
        let c = corr[src as usize];
        if c[0] < 2.0 {
            continue;
        }
        let n = c[0];
        let (mt, ms) = (c[1] / n, c[2] / n);
        let cov = c[3] / n - mt * ms;
        // Covariance normalized by the teacher variance: the regression slope of
        // static on teacher, ~1 when the sign is right and negative when flipped.
        let vt = c[4] / n - mt * mt;
        eprintln!(
            "[{name}] n={n:.0} mean teacher={mt:.1} mean static={ms:.1} cov/var_t={:.3} (must be clearly positive; negative = eval sign is flipped)",
            if vt > 0.0 { cov / vt } else { 0.0 }
        );
    }
    let mut pv: Vec<(String, u64)> = stats.per_variant.lock().unwrap().clone().into_iter().collect();
    pv.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (v, n) in pv {
        eprintln!("  {v:<24} {n:>10}");
    }
}
