//! Self-play corpus generator for Texel tuning (and puzzle mining).
//!
//! Plays fixed-depth self-play games in-process (no UCI subprocess) and writes one
//! fully-annotated JSON record per game to a JSONL file, flushed as each game
//! finishes so a kill never loses more than the games still in flight.
//!
//! Every position carries everything a downstream tuner needs decided HERE, at
//! generation time: static eval, search score, depth actually reached, think time,
//! nodes, piece count, phase, and the quiet-classification flags. Nothing has to be
//! re-searched later.
//!
//! World bounds are a process-global (`moves::set_world_bounds`), so variants are
//! grouped by bounds and each group dispatches as one flat, interleaved parallel
//! pass (round-robin across the group's variants) rather than a per-variant batch —
//! a per-variant batch barrier let one slow/stalled game stall an entire chunk
//! while every other thread sat idle. Each game also runs under a hard wall-clock
//! deadline (`with_hard_timeout`): `check_time`'s internal polling only re-checks
//! every 4096 nodes, so a single pathologically expensive node can stall a search
//! well past its intended cap with no internal chance to notice.

use apeiron::Variant;
use apeiron::board::{Coordinate, PlayerColor};
use apeiron::evaluation::{get_piece_phase, insufficient_material};
use apeiron::game::GameState;
use apeiron::moves::{Move, MoveList};
use apeiron::search;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

/// Record format version, bumped whenever a field's meaning changes so a mixed
/// corpus stays interpretable.
const FORMAT_VERSION: u32 = 1;

/// Scores at or above this are mates, not centipawn evaluations.
const MATE_FLOOR: i32 = search::MATE_SCORE;

static STOP: AtomicBool = AtomicBool::new(false);
static GAMES_DONE: AtomicU64 = AtomicU64::new(0);
static POSITIONS_KEPT: AtomicU64 = AtomicU64::new(0);
static POSITIONS_TOTAL: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// variant presets (mirrors sprt.rs so the two corpora stay comparable)
// ---------------------------------------------------------------------------

/// Base-eval variants only: no custom evaluators, no multi-king. The right set for
/// tuning `evaluation/base.rs`, since Chess/Obstocean/PawnHorde positions are scored
/// by their own evaluators and would pull base terms toward the wrong optimum.
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

fn resolve_variants(spec: &str) -> Result<Vec<Variant>, String> {
    match spec.trim().to_lowercase().as_str() {
        "base" | "base_only" => return Ok(BASE_ONLY_VARIANTS.to_vec()),
        "base_full" => return Ok(BASE_FULL_VARIANTS.to_vec()),
        "site" => return Ok(SITE_VARIANTS.to_vec()),
        "all" => return Ok(ALL_VARIANTS.to_vec()),
        _ => {}
    }

    let mut out = Vec::new();
    for name in spec.split(',') {
        // Separator-agnostic: "CoaIP_HO", "coaip-ho" and "CoaIPHO" must all match
        // the same variant, so strip every separator instead of normalizing to one.
        let want: String = name
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        if want.is_empty() {
            continue;
        }
        let found = ALL_VARIANTS
            .iter()
            .find(|v| {
                let canon: String = v
                    .to_str()
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                canon == want
            })
            .copied();
        match found {
            Some(v) => out.push(v),
            None => return Err(format!("unknown variant '{}'", name.trim())),
        }
    }
    if out.is_empty() {
        return Err("no variants selected".to_string());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Generate a self-play corpus (annotated per-position) for Texel tuning"
)]
struct Cli {
    /// Total games to have in the corpus when finished (existing games count).
    #[arg(long, default_value_t = 100_000)]
    games: usize,

    /// Fixed search depth per move.
    #[arg(long, default_value_t = 15)]
    depth: usize,

    /// Hard per-move think cap; on expiry the last completed depth is used.
    #[arg(long, default_value_t = 15_000)]
    max_move_ms: u64,

    /// Variant preset (`base_only`, `base_full`, `site`, `all`) or a comma list.
    #[arg(long, default_value = "base_only")]
    variants: String,

    /// Output JSONL path (appended to; resumes from what is already there).
    #[arg(long, default_value = "games/texel_corpus.jsonl")]
    out: String,

    /// Worker threads; 0 uses all cores.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Per-thread TT size. Small on purpose: many concurrent searchers, and a
    /// fixed shallow depth does not need a big table.
    #[arg(long, default_value_t = 8)]
    tt_mb: usize,

    /// Ply cap before a game is scored as a draw (or adjudicated).
    #[arg(long, default_value_t = 300)]
    max_plies: usize,

    /// Opening plies played from a randomized MultiPV pick, for structural diversity.
    #[arg(long, default_value_t = 12)]
    opening_plies: usize,

    /// MultiPV width used during the randomized opening.
    #[arg(long, default_value_t = 5)]
    opening_multipv: usize,

    /// Depth used during the randomized opening (cheap: these plies are discarded).
    #[arg(long, default_value_t = 5)]
    opening_depth: usize,

    /// Opening picks are limited to lines within this many cp of the best line.
    #[arg(long, default_value_t = 120)]
    opening_window: i32,

    /// Resign adjudication threshold in cp; 0 disables.
    #[arg(long, default_value_t = 2500)]
    resign_cp: i32,

    /// Consecutive plies above `resign_cp` (same side ahead) before adjudicating.
    #[arg(long, default_value_t = 4)]
    resign_plies: usize,

    /// Draw adjudication: |score| at or below this counts as dead-equal. 0 disables.
    #[arg(long, default_value_t = 10)]
    draw_cp: i32,

    /// Consecutive dead-equal plies before adjudicating a draw.
    #[arg(long, default_value_t = 16)]
    draw_plies: usize,

    /// Draw adjudication only applies from this ply onward.
    #[arg(long, default_value_t = 80)]
    draw_min_ply: usize,

    /// Max-ply fallback: if the ply cap is hit with no other terminal, a |last
    /// score| at or above this (White-ahead cp) awards the win instead of a draw.
    #[arg(long, default_value_t = 1000)]
    maxply_cp: i32,

    // ---- quiet classification (decided at generation time) ----
    /// Plies before this are never marked quiet (covers the randomized opening).
    #[arg(long, default_value_t = 14)]
    quiet_min_ply: usize,

    /// Positions with fewer pieces than this are never marked quiet.
    #[arg(long, default_value_t = 4)]
    quiet_min_pieces: usize,

    /// Max |search score - static eval| for a position to count as quiet. This is
    /// the tactics filter: a large gap means the search found something the static
    /// eval cannot see, so the position teaches the eval nothing.
    #[arg(long, default_value_t = 150)]
    quiet_max_gap: i32,

    /// Positions with |score| above this are never marked quiet (already decided).
    #[arg(long, default_value_t = 2000)]
    quiet_max_score: i32,

    /// Seed base; the per-game seed is derived from it, the variant and the index.
    #[arg(long, default_value_t = 0x5DEE_CE66_D125_A03B)]
    seed: u64,

    /// Store every position, not just quiet ones (`quiet` flag is recorded either
    /// way). Off by default: noisy plies triple corpus size for no tuning value.
    #[arg(long, default_value_t = false)]
    keep_noisy: bool,

    /// Also emit the sprt-style annotated ICN string per game, so `puzzle_gen`
    /// can consume the corpus via `export-icn`.
    #[arg(long, default_value_t = true)]
    with_icn: bool,
}

// ---------------------------------------------------------------------------
// output records
// ---------------------------------------------------------------------------

/// One position. All scores are White-ahead centipawns (positive = White better)
/// regardless of `stm`, so a consumer never has to guess the sign convention.
#[derive(Serialize)]
struct PositionRecord {
    ply: u32,
    /// Side to move: "w" or "b".
    stm: &'static str,
    /// Static eval, White-ahead cp.
    eval: i32,
    /// Search score at `depth`, White-ahead cp.
    score: i32,
    /// Depth actually completed (may be below the target if the time cap hit).
    depth: u32,
    /// Think time for this move in ms.
    ms: u64,
    nodes: u64,
    pieces: u32,
    /// Game phase, 0 (bare endgame) to 24 (full middlegame material).
    phase: i32,
    /// Halfmove clock; high values mean the eval is rule50-damped.
    hmc: u32,
    /// Chosen move captures.
    cap: bool,
    /// Chosen move promotes.
    promo: bool,
    /// Side to move is in check.
    chk: bool,
    /// Passed every quiet filter and is safe to train on.
    quiet: bool,
    /// Chosen move, ICN coordinate form.
    bm: String,
}

#[derive(Serialize)]
struct GameRecord {
    v: u32,
    variant: &'static str,
    /// "1-0", "0-1" or "1/2-1/2".
    result: &'static str,
    /// White's score: 1.0, 0.5 or 0.0.
    wdl: f32,
    termination: &'static str,
    plies: u32,
    depth_target: u32,
    max_move_ms: u64,
    /// Board setup ICN; replay `moves` from here to reach any position.
    start_icn: String,
    moves: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    icn: Option<String>,
    positions: Vec<PositionRecord>,
}

// ---------------------------------------------------------------------------
// game playing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Terminal {
    WhiteWin(&'static str),
    BlackWin(&'static str),
    Draw(&'static str),
}

impl Terminal {
    fn parts(self) -> (&'static str, f32, &'static str) {
        match self {
            Terminal::WhiteWin(r) => ("1-0", 1.0, r),
            Terminal::BlackWin(r) => ("0-1", 0.0, r),
            Terminal::Draw(r) => ("1/2-1/2", 0.5, r),
        }
    }
}

fn loser_to_terminal(loser: PlayerColor, reason: &'static str) -> Terminal {
    if loser == PlayerColor::White {
        Terminal::BlackWin(reason)
    } else {
        Terminal::WhiteWin(reason)
    }
}

/// Runs `f` on a fresh OS thread and waits up to `dur`. `check_time`'s hard limit
/// only re-polls every 4096 nodes, so a single pathologically expensive node (this
/// board's movegen can produce them) can stall a search well past its intended cap
/// with no internal chance to notice — puzzle_gen hit multi-hour stalls this way.
/// This is the backstop: past the deadline the game is abandoned (its thread keeps
/// running until process exit, but no longer blocks the rayon pool behind it) rather
/// than letting one stuck game stall an entire batch with idle cores everywhere else.
fn with_hard_timeout<T: Send + 'static>(
    dur: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(dur).ok()
}

/// Exact legality test. `get_pseudo_legal_moves` is pseudo-legal and keeps the slider
/// candidate cache, so a terminal check written on it silently never fires;
/// `get_pseudo_legal_moves_into` bypasses the cache, and each move still needs a
/// make/`is_move_illegal`/undo filter.
fn has_any_legal_move(game: &GameState) -> bool {
    let mut moves = MoveList::new();
    game.get_pseudo_legal_moves_into(&mut moves);
    for m in moves.iter() {
        let mut probe = game.clone();
        probe.make_move(m);
        if !probe.is_move_illegal() {
            return true;
        }
    }
    false
}

fn detect_terminal(game: &GameState) -> Option<Terminal> {
    use apeiron::game::WinCondition;

    let stm = game.turn;
    let opp_wc = match stm {
        PlayerColor::White => game.game_rules.black_win_condition,
        PlayerColor::Black => game.game_rules.white_win_condition,
        PlayerColor::Neutral => return None,
    };

    // AllPiecesCaptured first: taking the last piece also zeroes the royals, which
    // sends has_lost_by_royal_capture down a branch that skips termination entirely.
    if opp_wc == WinCondition::AllPiecesCaptured {
        let has_royals = match stm {
            PlayerColor::White => !game.white_royals.is_empty(),
            PlayerColor::Black => !game.black_royals.is_empty(),
            _ => false,
        };
        if !game.has_pieces(stm) && !has_royals {
            return Some(loser_to_terminal(stm, "allpiecescaptured"));
        }
    } else if game.has_lost_by_royal_capture() {
        match opp_wc {
            WinCondition::RoyalCapture => {
                return Some(loser_to_terminal(stm, "royalcapture"));
            }
            WinCondition::AllRoyalsCaptured => {
                return Some(loser_to_terminal(stm, "allroyalscaptured"));
            }
            // Checkmate: losing royals is not itself the win condition, so fall
            // through to the legal-move test below.
            _ => return None,
        }
    }

    if !has_any_legal_move(game) {
        if game.is_in_check() && game.must_escape_check() {
            return Some(loser_to_terminal(stm, "checkmate"));
        }
        if !game.has_pieces(stm) {
            return Some(loser_to_terminal(stm, "allpiecescaptured"));
        }
        return Some(Terminal::Draw("stalemate"));
    }

    if insufficient_material::evaluate_insufficient_material_game_handler(game) {
        return Some(Terminal::Draw("insufficient_material"));
    }
    if game.is_fifty() {
        return Some(Terminal::Draw("fifty-move rule"));
    }
    None
}

/// Board-setup ICN for the starting position, matching the SPRT harness's form so
/// the two corpora are interchangeable.
fn board_setup_icn(game: &GameState, variant: Variant) -> String {
    let move_limit = game.game_rules.move_rule_limit.unwrap_or(100);
    let promos = match &game.game_rules.promotion_types {
        Some(types) => types
            .iter()
            .map(|pt| pt.to_site_code().to_lowercase())
            .collect::<Vec<_>>()
            .join(","),
        None => "q,r,b,n".to_string(),
    };
    let promo_token = format!(
        "({};{}|{};{})",
        game.white_promo_rank, promos, game.black_promo_rank, promos
    );
    let b = variant.get_default_bounds();
    let bounds_token = format!("{},{},{},{}", b.0, b.1, b.2, b.3);

    let mut pieces: Vec<_> = game.board.iter().collect();
    pieces.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let pieces_str = pieces
        .iter()
        .map(|(x, y, piece)| {
            let mut code = piece.piece_type().to_site_code().to_string();
            if piece.color() != PlayerColor::White {
                code = code.to_lowercase();
            }
            let mut y_str = y.to_string();
            if game.has_special_right(&Coordinate::new(*x, *y)) {
                y_str.push('+');
            }
            format!("{}{},{}", code, x, y_str)
        })
        .collect::<Vec<_>>()
        .join("|");

    let win_cond = if game.game_rules.white_win_condition != apeiron::game::WinCondition::Checkmate
        || game.game_rules.black_win_condition != apeiron::game::WinCondition::Checkmate
    {
        format!(
            "{:?},{:?} ",
            game.game_rules.white_win_condition, game.game_rules.black_win_condition
        )
        .to_lowercase()
    } else {
        String::new()
    };

    format!(
        "[Variant \"{}\"] w 0/{} 1 {} {} {}{}",
        variant.to_str(),
        move_limit,
        promo_token,
        bounds_token,
        win_cond,
        pieces_str
    )
}

fn move_to_icn(m: &Move, mover: PlayerColor) -> String {
    let mut s = format!("{},{}>{},{}", m.from.x, m.from.y, m.to.x, m.to.y);
    if let Some(promo) = m.promotion {
        let code = promo.to_site_code();
        s.push('=');
        s.push_str(&if mover == PlayerColor::White {
            code.to_uppercase()
        } else {
            code.to_lowercase()
        });
    }
    s
}

fn total_phase(game: &GameState) -> i32 {
    let sum: i32 = game
        .board
        .iter()
        .map(|(_, _, piece)| get_piece_phase(piece.piece_type()))
        .sum();
    sum.min(apeiron::evaluation::base::MAX_PHASE)
}

#[inline]
fn static_eval(game: &GameState) -> i32 {
    #[cfg(feature = "nnue")]
    return apeiron::evaluation::evaluate(game, None);
    #[cfg(not(feature = "nnue"))]
    return apeiron::evaluation::evaluate(game);
}

/// Convert a side-to-move-relative score to White-ahead.
#[inline]
fn to_white(score: i32, stm: PlayerColor) -> i32 {
    if stm == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// xorshift64*; a full RNG crate is overkill for opening picks and this keeps the
/// per-game stream reproducible from (seed, variant, index).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

/// Cheap position key for threefold detection. The zobrist hash already folds in
/// turn, castling and en passant, so it needs nothing added.
#[inline]
fn position_key(game: &GameState) -> u64 {
    game.hash
}

fn play_game(cfg: &Cli, variant: Variant, game_idx: usize) -> Option<GameRecord> {
    let mut rng = Rng::new(
        cfg.seed
            ^ (variant as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (game_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
    );

    let mut game = GameState::new();
    game.setup_position_from_icn(variant.starting_icn());
    game.variant = Some(variant);
    game.recompute_piece_counts();
    game.recompute_hash();

    let start_icn = board_setup_icn(&game, variant);

    let mut moves_clean: Vec<String> = Vec::new();
    let mut moves_annotated: Vec<String> = Vec::new();
    let mut positions: Vec<PositionRecord> = Vec::new();
    let mut reps: HashMap<u64, u32> = HashMap::new();
    *reps.entry(position_key(&game)).or_insert(0) += 1;

    // Adjudication streaks, tracked in White-ahead cp.
    let mut resign_streak = 0usize;
    let mut resign_side_white = false;
    let mut draw_streak = 0usize;
    // Last played (non-opening) score, for the max-ply fallback below.
    let mut last_score_white: Option<i32> = None;

    let mut terminal: Option<Terminal> = None;

    for ply in 0..cfg.max_plies {
        if STOP.load(Ordering::Relaxed) {
            return None;
        }

        if let Some(t) = detect_terminal(&game) {
            terminal = Some(t);
            break;
        }
        if *reps.get(&position_key(&game)).unwrap_or(&0) >= 3 {
            terminal = Some(Terminal::Draw("threefold repetition"));
            break;
        }

        let stm = game.turn;
        let in_opening = ply < cfg.opening_plies;
        let (depth, multipv) = if in_opening {
            (cfg.opening_depth, cfg.opening_multipv.max(1))
        } else {
            (cfg.depth, 1)
        };

        let eval_before = to_white(static_eval(&game), stm);
        let started = Instant::now();
        let result = search::get_best_moves_multipv(
            &mut game,
            depth,
            cfg.max_move_ms as u128,
            cfg.max_move_ms as u128,
            multipv,
            true,
            false,
        );
        let elapsed_ms = started.elapsed().as_millis() as u64;

        if result.lines.is_empty() {
            // The search found nothing playable; trust the exact terminal check.
            terminal = Some(detect_terminal(&game).unwrap_or(Terminal::Draw("no moves")));
            break;
        }

        // Opening diversity: pick uniformly among lines that are not clearly worse
        // than the best, so the corpus spans many structures without losing openings.
        let best_score = result.lines[0].score;
        let chosen = if in_opening && result.lines.len() > 1 {
            let eligible = result
                .lines
                .iter()
                .filter(|l| best_score - l.score <= cfg.opening_window)
                .count()
                .max(1);
            &result.lines[rng.below(eligible)]
        } else {
            &result.lines[0]
        };

        let best_move = chosen.mv;
        let score_white = to_white(chosen.score, stm);
        let captures = game.board.is_occupied(best_move.to.x, best_move.to.y)
            || game
                .en_passant
                .as_ref()
                .is_some_and(|ep| ep.square.x == best_move.to.x && ep.square.y == best_move.to.y);

        // The opening plies are deliberately randomized, so their scores describe a
        // move nobody would play; never record them as training positions.
        if !in_opening {
            let in_check = game.is_in_check();
            let pieces = game.board.len();
            let gap = (score_white - eval_before).abs();
            let quiet = !in_check
                && !captures
                && best_move.promotion.is_none()
                && ply >= cfg.quiet_min_ply
                && pieces >= cfg.quiet_min_pieces
                && score_white.abs() < MATE_FLOOR
                && score_white.abs() <= cfg.quiet_max_score
                && gap <= cfg.quiet_max_gap;

            POSITIONS_TOTAL.fetch_add(1, Ordering::Relaxed);
            if quiet || cfg.keep_noisy {
                if quiet {
                    POSITIONS_KEPT.fetch_add(1, Ordering::Relaxed);
                }
                positions.push(PositionRecord {
                    ply: ply as u32,
                    stm: if stm == PlayerColor::White { "w" } else { "b" },
                    eval: eval_before,
                    score: score_white,
                    depth: chosen.depth as u32,
                    ms: elapsed_ms,
                    nodes: result.stats.nodes,
                    pieces: pieces as u32,
                    phase: total_phase(&game),
                    hmc: game.halfmove_clock,
                    cap: captures,
                    promo: best_move.promotion.is_some(),
                    chk: in_check,
                    quiet,
                    bm: move_to_icn(&best_move, stm),
                });
            }
        }

        // Adjudication, on the played score only (a randomized opening pick would
        // otherwise resign games that are perfectly fine).
        if !in_opening {
            last_score_white = Some(score_white);

            if cfg.resign_cp > 0 && score_white.abs() >= cfg.resign_cp {
                let side_white = score_white > 0;
                if resign_streak > 0 && side_white == resign_side_white {
                    resign_streak += 1;
                } else {
                    resign_streak = 1;
                    resign_side_white = side_white;
                }
            } else {
                resign_streak = 0;
            }

            if cfg.draw_cp > 0 && score_white.abs() <= cfg.draw_cp && ply >= cfg.draw_min_ply {
                draw_streak += 1;
            } else {
                draw_streak = 0;
            }
        }

        let mover = stm;
        game.make_move(&best_move);
        let icn_move = move_to_icn(&best_move, mover);
        if cfg.with_icn {
            let annotation = if score_white.abs() >= MATE_FLOOR {
                let dist = (search::MATE_VALUE - score_white.abs() + 1) / 2;
                format!(
                    "{{[%mate {}{}]}}",
                    if score_white > 0 { "" } else { "-" },
                    dist
                )
            } else {
                format!("{{[%eval {:+.2}]}}", score_white as f64 / 100.0)
            };
            moves_annotated.push(format!("{}{}", icn_move, annotation));
        }
        moves_clean.push(icn_move);
        *reps.entry(position_key(&game)).or_insert(0) += 1;

        if resign_streak >= cfg.resign_plies.max(1) {
            terminal = Some(if resign_side_white {
                Terminal::WhiteWin("resign adjudication")
            } else {
                Terminal::BlackWin("resign adjudication")
            });
            break;
        }
        if draw_streak >= cfg.draw_plies.max(1) {
            terminal = Some(Terminal::Draw("draw adjudication"));
            break;
        }
    }

    // Max-ply fallback: a game that never tripped a streak (e.g. it was still
    // climbing) must not default to a draw when the last score was clearly
    // decisive. Mirrors sprt.rs's own max-ply adjudication, just without a second
    // engine to agree with since self-play only has the one score to trust.
    let terminal = terminal.or_else(|| detect_terminal(&game)).or_else(|| {
        last_score_white.and_then(|s| {
            if s >= cfg.maxply_cp {
                Some(Terminal::WhiteWin("max-ply adjudication"))
            } else if s <= -cfg.maxply_cp {
                Some(Terminal::BlackWin("max-ply adjudication"))
            } else {
                None
            }
        })
    });
    let terminal = terminal.unwrap_or(Terminal::Draw("max_plies"));
    let (result_str, wdl, reason) = terminal.parts();

    let icn = cfg.with_icn.then(|| {
        let mut s = format!(
            "[Event \"Apeiron data_gen {}\"] [Result \"{}\"] [Termination \"{}\"] \
             [White \"Apeiron\"] [Black \"Apeiron\"] ",
            game_idx, result_str, reason
        );
        s.push_str(&start_icn);
        if !moves_annotated.is_empty() {
            s.push(' ');
            s.push_str(&moves_annotated.join("|"));
        }
        s
    });

    Some(GameRecord {
        v: FORMAT_VERSION,
        variant: variant.to_str(),
        result: result_str,
        wdl,
        termination: reason,
        plies: moves_clean.len() as u32,
        depth_target: cfg.depth as u32,
        max_move_ms: cfg.max_move_ms,
        start_icn,
        moves: moves_clean,
        icn,
        positions,
    })
}

// ---------------------------------------------------------------------------
// resume
// ---------------------------------------------------------------------------

/// Count existing games per variant so a resumed run refills the round-robin
/// instead of restarting it. Reads the variant tag textually: parsing 100k full
/// records just to count them would be minutes of JSON work.
fn scan_existing(path: &str) -> (usize, HashMap<String, usize>) {
    let mut per_variant: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    let Ok(file) = File::open(path) else {
        return (0, per_variant);
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        if let Some(rest) = line.split_once("\"variant\":\"").map(|(_, r)| r)
            && let Some((name, _)) = rest.split_once('"')
        {
            *per_variant.entry(name.to_string()).or_insert(0) += 1;
        }
    }
    (total, per_variant)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = Cli::parse();

    let variants = match resolve_variants(&cfg.variants) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[data_gen] {}", e);
            std::process::exit(1);
        }
    };

    let threads = if cfg.threads == 0 {
        num_cpus::get()
    } else {
        cfg.threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(16 * 1024 * 1024)
        .build_global()
        .expect("failed to build rayon pool");
    search::set_tt_size_mb(cfg.tt_mb);

    ctrlc::set_handler(|| {
        if STOP.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        eprintln!("\n[data_gen] stopping after in-flight games finish (Ctrl-C again to force)");
    })
    .expect("failed to install Ctrl-C handler");

    if let Some(parent) = std::path::Path::new(&cfg.out).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).expect("failed to create output directory");
    }

    let (existing_total, mut done_per_variant) = scan_existing(&cfg.out);
    if existing_total >= cfg.games {
        println!(
            "[data_gen] {} already has {} games (target {}); nothing to do.",
            cfg.out, existing_total, cfg.games
        );
        return;
    }

    // Even coverage across variants, as required when the same eval serves all of
    // them: a corpus dominated by one variant tunes shared terms for that variant.
    let target_per_variant = cfg.games.div_ceil(variants.len());

    println!(
        "[data_gen] {} variants x {} games (target {}), depth {} cap {}ms, {} threads",
        variants.len(),
        target_per_variant,
        cfg.games,
        cfg.depth,
        cfg.max_move_ms,
        threads
    );
    if existing_total > 0 {
        println!(
            "[data_gen] resuming: {} games already in {}",
            existing_total, cfg.out
        );
    }

    let writer = Mutex::new(BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&cfg.out)
            .expect("failed to open output file"),
    ));

    let progress = ProgressBar::new(cfg.games as u64);
    progress.set_position(existing_total as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:36.cyan/blue}] {pos}/{len} \
             ({percent}%) eta {eta} | {msg}",
        )
        .expect("bad progress template")
        .progress_chars("#>-"),
    );
    GAMES_DONE.store(existing_total as u64, Ordering::Relaxed);

    // A per-game hard wall-clock deadline: the theoretical worst case if every ply
    // hit the cap, doubled for slack. Bounds a stalled search (see with_hard_timeout)
    // to "abandon this one game" instead of "block a thread indefinitely".
    let game_deadline = Duration::from_millis(
        (cfg.max_plies as u64)
            .saturating_mul(cfg.max_move_ms)
            .saturating_mul(2)
            .max(120_000),
    );

    // Group requested variants by world bounds: `set_world_bounds` is a process
    // global, so only variants sharing identical bounds may be dispatched together
    // in one flat parallel pass. In practice this run's variant list is almost
    // always one group (every unbounded base-eval variant shares the same bounds),
    // which is what actually matters: a single flat dispatch across the whole
    // group removes the old per-variant batch barrier that let one slow/stalled
    // game stall an entire chunk while every other thread sat idle.
    let mut bounds_groups: Vec<((i64, i64, i64, i64), Vec<Variant>)> = Vec::new();
    for &v in &variants {
        let b = v.get_default_bounds();
        match bounds_groups.iter_mut().find(|(gb, _)| *gb == b) {
            Some((_, vs)) => vs.push(v),
            None => bounds_groups.push((b, vec![v])),
        }
    }

    for (bounds, group) in &bounds_groups {
        if STOP.load(Ordering::Relaxed) {
            break;
        }
        apeiron::moves::set_world_bounds(bounds.0, bounds.1, bounds.2, bounds.3);

        // Interleave every variant in this bounds-group into one flat work list
        // (round-robin by index) so the corpus stays balanced even if the run is
        // killed partway, while still dispatching it as a single parallel pass.
        // `Variant` has no `Hash` impl, so track remaining counts by position in
        // `group` rather than keying a map on the variant itself.
        let done_by_pos: Vec<usize> = group
            .iter()
            .map(|v| *done_per_variant.entry(v.to_str().to_string()).or_insert(0))
            .collect();
        let remaining_by_pos: Vec<usize> = done_by_pos
            .iter()
            .map(|&done| target_per_variant.saturating_sub(done))
            .collect();
        let max_remaining = remaining_by_pos.iter().copied().max().unwrap_or(0);

        let mut work: Vec<(Variant, usize)> = Vec::new();
        for offset in 0..max_remaining {
            for (pos, &v) in group.iter().enumerate() {
                if offset < remaining_by_pos[pos] {
                    work.push((v, done_by_pos[pos] + offset));
                }
            }
        }

        work.into_par_iter().for_each(|(variant, idx)| {
            if STOP.load(Ordering::Relaxed) {
                return;
            }
            let cfg_owned = cfg.clone();
            let record =
                with_hard_timeout(game_deadline, move || play_game(&cfg_owned, variant, idx))
                    .flatten();
            let Some(record) = record else {
                return;
            };
            let Ok(line) = serde_json::to_string(&record) else {
                return;
            };

            // Flush per game: a killed run must lose nothing already played.
            if let Ok(mut w) = writer.lock() {
                let _ = w.write_all(line.as_bytes());
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }

            let games = GAMES_DONE.fetch_add(1, Ordering::Relaxed) + 1;
            progress.set_position(games);
            let kept = POSITIONS_KEPT.load(Ordering::Relaxed);
            let seen = POSITIONS_TOTAL.load(Ordering::Relaxed).max(1);
            progress.set_message(format!(
                "{} | {} quiet pos ({:.1}%)",
                variant.to_str(),
                kept,
                100.0 * kept as f64 / seen as f64
            ));
        });

        for &v in group {
            *done_per_variant.get_mut(v.to_str()).unwrap() = target_per_variant;
        }
    }

    if let Ok(mut w) = writer.lock() {
        let _ = w.flush();
    }
    progress.finish_and_clear();

    println!(
        "[data_gen] wrote {} games total to {} ({} quiet positions this run, {:.1}% of {} scored)",
        GAMES_DONE.load(Ordering::Relaxed),
        cfg.out,
        POSITIONS_KEPT.load(Ordering::Relaxed),
        100.0 * POSITIONS_KEPT.load(Ordering::Relaxed) as f64
            / POSITIONS_TOTAL.load(Ordering::Relaxed).max(1) as f64,
        POSITIONS_TOTAL.load(Ordering::Relaxed)
    );
}
