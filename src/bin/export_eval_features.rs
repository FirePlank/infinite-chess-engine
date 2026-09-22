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
const SOURCE_TEXEL: u8 = 0;
const SOURCE_SPRT: u8 = 1;

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

/// Deterministic per-(game, ply) sample in [0, 1).
fn unit_interval(seed: u64) -> f64 {
    (splitmix(seed) >> 11) as f64 / (1u64 << 53) as f64
}

fn replay(
    game: &Game,
    mut g: GameState,
    game_id: u32,
    cli: &Cli,
    seen: &[Mutex<HashSet<u64>>],
    stats: &Stats,
    out: &mut Vec<u8>,
) {
    if g.eval_kind != EvalKind::Generic {
        return;
    }
    let sample = if game.source == SOURCE_TEXEL {
        cli.texel_sample
    } else {
        cli.sprt_sample
    };
    let vid = variant_id(&game.variant);
    stats.games.fetch_add(1, Ordering::Relaxed);
    let mut kept_here = 0u64;
    let mut corr = [0f64; 5];

    for (ply, mv) in game.moves.iter().enumerate() {
        let Some((fx, fy, tx, ty, promo)) = parse_move(mv) else {
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
            let mut teacher = game.teacher[ply].unwrap();
            let mut source = game.source;
            if cli.relabel_depth > 0 {
                let mut gs = g.clone();
                let Some((bm, score, _)) = apeiron::search::get_best_move(
                    &mut gs,
                    cli.relabel_depth,
                    cli.relabel_ms as u128,
                    true,
                    false,
                ) else {
                    g.make_move_coords(fx, fy, tx, ty, promo.as_deref());
                    continue;
                };
                // Same quiet definition as data_gen: the chosen move must not capture
                // or promote, and the score must be a real evaluation.
                if score.abs() >= MATE_FLOOR
                    || bm.promotion.is_some()
                    || g.board.get_piece(bm.to.x, bm.to.y).is_some()
                {
                    g.make_move_coords(fx, fy, tx, ty, promo.as_deref());
                    continue;
                }
                teacher = if g.turn == PlayerColor::Black { -score } else { score };
                source = SOURCE_TEXEL;
            }
            let mut fc = FeatureCollector::default();
            let stm_score = base::evaluate_inner_traced(&g, &mut fc);
            let static_white = if g.turn == PlayerColor::Black {
                -stm_score
            } else {
                stm_score
            };
            if (teacher - static_white).abs() <= cli.quiet_tolerance {
                let shard = &seen[(g.hash % seen.len() as u64) as usize];
                let fresh = shard.lock().unwrap().insert(g.hash);
                if fresh {
                    let x = feature_vector(&g, &fc);
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
                    out.push(g.turn as u8);
                    out.push(vid);
                    out.push(source);
                    out.push(fc.inputs.phase.clamp(0, 255) as u8);
                    out.push(0);
                    out.extend_from_slice(&game_id.to_le_bytes());
                    out.extend_from_slice(&(ply.min(65535) as u16).to_le_bytes());
                    out.extend_from_slice(&[0u8; 2]);
                    debug_assert_eq!(out.len() % RECORD_SIZE, 0);
                    kept_here += 1;
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
        g.make_move_coords(fx, fy, tx, ty, promo.as_deref());
    }

    if kept_here > 0 {
        stats.kept.fetch_add(kept_here, Ordering::Relaxed);
        let mut c = stats.corr.lock().unwrap();
        let row = &mut c[game.source as usize];
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
        let states: Vec<GameState> = chunk
            .iter()
            .map(|game| {
                let mut g = GameState::new();
                g.setup_position_from_icn(&game.start_icn);
                g.recompute_piece_counts();
                g.recompute_hash();
                g
            })
            .collect();
        let base_id = first_id + (ci * SETUP_CHUNK) as u32;
        let outputs: Vec<Vec<u8>> = chunk
            .par_iter()
            .zip(states.into_par_iter())
            .enumerate()
            .map(|(i, (game, g))| {
                let mut out = Vec::new();
                replay(game, g, base_id + i as u32, cli, &seen, stats, &mut out);
                out
            })
            .collect();
        let mut w = writer.lock().unwrap();
        for c in outputs {
            w.write_all(&c).unwrap();
        }
    }
}

fn group_by_variant(games: Vec<Game>) -> Vec<Vec<Game>> {
    let mut map: HashMap<String, Vec<Game>> = HashMap::new();
    for g in games {
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
    let mut file = File::create(&cli.out).unwrap();
    file.write_all(MAGIC).unwrap();
    file.write_all(&VERSION.to_le_bytes()).unwrap();
    file.write_all(&(NUM_FEATURES as u32).to_le_bytes()).unwrap();
    file.write_all(&schema_hash().to_le_bytes()).unwrap();
    file.write_all(&(RECORD_SIZE as u32).to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    assert_eq!(file.stream_position().unwrap(), HEADER_SIZE);
    let writer = Mutex::new(BufWriter::with_capacity(1 << 24, file));

    let stats = Stats {
        kept: AtomicU64::new(0),
        games: AtomicU64::new(0),
        dup: AtomicU64::new(0),
        corr: Mutex::new([[0.0; 5]; 2]),
        per_variant: Mutex::new(HashMap::new()),
    };
    let start = Instant::now();
    let mut next_id: u32 = 1;

    for path in &cli.texel {
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
        for (fi, path) in files.iter().enumerate() {
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

    eprintln!(
        "done: {} games replayed, {} records, {} zobrist dups skipped, {:.1}s",
        stats.games.load(Ordering::Relaxed),
        kept,
        stats.dup.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
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
