use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use clap::{Parser, Subcommand};
use hydrochess_wasm::Engine;
use hydrochess_wasm::Variant;
use hydrochess_wasm::board::{Coordinate, PlayerColor};
use hydrochess_wasm::game::GameState;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

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

        /// SPRT bound H1 (Elo difference where new IS better)
        #[arg(long, default_value_t = 5.0)]
        elo1: f64,

        /// SPRT alpha (type I error probability)
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,

        /// SPRT beta (type II error probability)
        #[arg(long, default_value_t = 0.05)]
        beta: f64,

        /// Time control (e.g., "10+0.1", "depth 6", "fixed 0.1s")
        #[arg(long, default_value = "10+0.1")]
        tc: String,

        /// Number of parallel games
        #[arg(long, default_value_t = 16)]
        concurrency: usize,

        /// Maximum games to run (omit for no limit)
        #[arg(long)]
        max_games: Option<usize>,

        /// Minimum games before SPRT can pass/fail
        #[arg(long, default_value_t = 250)]
        min_games: usize,

        /// Variants to test (comma-separated list)
        #[arg(
            long,
            default_value = "Classical,Confined_Classical,Classical_Plus,Core,CoaIP,CoaIP_HO,CoaIP_RO,CoaIP_NO,Palace,Pawndard,Standarch,Space_Classic,Space,Knightline"
        )]
        variants: String,

        /// Material threshold for draws
        #[arg(long, default_value_t = 2000)]
        adjudication: i32,

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

        /// Old engine strength level (1-3)
        #[arg(long, default_value_t = 3)]
        old_strength: u32,

        /// Print verbose engine info
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },

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

#[derive(Clone, Debug)]
struct Config {
    elo0: f64,
    elo1: f64,
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
    new_bin: String,
    old_bin: String,
    max_moves: usize,
    search_noise: i32,
    old_strength: u32,
    verbose: bool,
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
}

#[derive(Clone, Copy)]
enum TerminalState {
    Checkmate { white_won: bool },
    Draw(&'static str),
}

fn elo_to_score(elo_diff: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-elo_diff / 400.0))
}

fn calculate_llr(wins: usize, losses: usize, draws: usize, elo0: f64, elo1: f64) -> f64 {
    let total = wins + losses + draws;
    if total == 0 {
        return 0.0;
    }
    let score = (wins as f64 + draws as f64 * 0.5) / total as f64;
    let s0 = elo_to_score(elo0);
    let s1 = elo_to_score(elo1);
    let clamped_score = score.clamp(0.001, 0.999);
    total as f64
        * (clamped_score * (s1 / s0).ln() + (1.0 - clamped_score) * ((1.0 - s1) / (1.0 - s0)).ln())
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

fn format_clock(ms: u64) -> String {
    let total_seconds = ms / 1000;
    let h = total_seconds / 3600;
    let m = (total_seconds % 3600) / 60;
    let s = total_seconds % 60;
    let dec = (ms % 1000) / 100;
    format!("{}:{:02}:{:02}.{}", h, m, s, dec)
}

fn move_to_string(m: &hydrochess_wasm::moves::Move) -> String {
    let mut s = format!("{},{} {},{}", m.from.x, m.from.y, m.to.x, m.to.y);
    if let Some(p) = m.promotion {
        s.push_str(&format!(" {}", p.to_site_code().to_lowercase()));
    }
    s
}

fn is_ctrl_c_exit_code(code: Option<i32>) -> bool {
    matches!(code, Some(130) | Some(-1073741510))
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

fn has_any_fully_legal_move(game: &mut GameState) -> bool {
    let moves = game.get_legal_moves();
    for m in moves {
        let undo = game.make_move(&m);
        let legal = !game.is_move_illegal();
        game.undo_move(&m, undo);
        if legal {
            return true;
        }
    }
    false
}

fn detect_terminal_state(game: &mut GameState) -> Option<TerminalState> {
    let in_check = game.is_in_check();
    let has_legal_move = has_any_fully_legal_move(game);
    if !has_legal_move {
        let lost_by_mate = in_check && game.must_escape_check();
        let lost_by_piece_capture = !game.has_pieces(game.turn);

        if lost_by_mate || lost_by_piece_capture {
            return Some(TerminalState::Checkmate {
                white_won: game.turn == PlayerColor::Black,
            });
        }

        return Some(TerminalState::Draw("stalemate"));
    }

    if game.is_draw(0, in_check) {
        if game.is_fifty() {
            return Some(TerminalState::Draw("fifty-move rule"));
        }
        if game.is_repetition(0) {
            return Some(TerminalState::Draw("threefold repetition"));
        }
    }

    if game.is_fifty() {
        return Some(TerminalState::Draw("fifty-move rule"));
    }

    if game.is_repetition(0) {
        return Some(TerminalState::Draw("threefold repetition"));
    }

    None
}

fn play_game(
    config: &Config,
    variant: Variant,
    new_plays_white: bool,
    game_idx: usize,
    seeds: Vec<u64>,
) -> GameOutcome {
    let mut game = GameState::new();
    game.setup_position_from_icn(variant.starting_icn());
    game.variant = Some(variant);

    let starting_board_setup = get_board_setup_icn(&game);

    let mut white_clock = config.tc_base_ms;
    let mut black_clock = config.tc_base_ms;
    let mut move_info_log = Vec::new();
    let mut move_history_clean: Vec<String> = Vec::new();
    let termination_reason;

    let get_eval = |g: &GameState| {
        #[cfg(feature = "nnue")]
        return hydrochess_wasm::evaluation::evaluate(g, None);
        #[cfg(not(feature = "nnue"))]
        return hydrochess_wasm::evaluation::evaluate(g);
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
            }
        };
    }

    for ply in 0..config.max_moves {
        if STOP.load(Ordering::SeqCst) {
            break;
        }

        // Terminal state checks always run before adjudication or engine search.
        if let Some(terminal) = detect_terminal_state(&mut game) {
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
                TerminalState::Draw(reason) => {
                    return game_outcome!(GameResult::Draw, reason, "1/2-1/2");
                }
            }
        }

        //  Material adjudication (after terminal checks)
        let eval = get_eval(&game);
        if eval.abs() >= config.adjudication_threshold {
            let white_winning = eval > 0;
            let result = if white_winning == new_plays_white {
                GameResult::Win
            } else {
                GameResult::Loss
            };
            let result_str = if white_winning { "1-0" } else { "0-1" };
            return game_outcome!(result, "material adjudication", result_str);
        }

        //  Engine search
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

        let seed_val = seeds[ply];
        cmd.arg("--seed").arg(seed_val.to_string());

        if !is_new_turn && config.old_strength < 3 {
            cmd.arg("--strength-level")
                .arg(config.old_strength.to_string());
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

        // Check for subprocess crash
        if !(output.status.success()
            || USER_STOP.load(Ordering::SeqCst) && is_ctrl_c_exit_code(output.status.code()))
        {
            eprintln!(
                "[Game {}] Subprocess crashed! exit={:?}\n  stderr={}",
                game_idx,
                output.status.code(),
                stderr.trim()
            );
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
        let mut flagged_on_time = false;
        let mut remaining_clock = 0;

        if current_clock < elapsed {
            flagged_on_time = true;
        } else {
            remaining_clock = current_clock - elapsed;
        }
        remaining_clock += config.tc_inc_ms;

        if flagged_on_time {
            let result = if is_new_turn {
                GameResult::Loss
            } else {
                GameResult::Win
            };
            let white_won = (result == GameResult::Win) == new_plays_white;
            let result_str = if white_won { "1-0" } else { "0-1" };
            return game_outcome!(result, "timeout", result_str);
        }

        if let Some(move_icn) = bestmove_icn {
            // Build annotated move for the output log
            let mut comment = format!("[%clk {}]", format_clock(remaining_clock));
            if let Some(s) = score {
                comment.push_str(&format!(" [%eval {:+.2}]", s / 100.0));
            }
            move_info_log.push(format!("{}{{{}}}", move_icn, comment));
            move_history_clean.push(move_icn);

            // Reconstruct game state from the full ICN (starting position + all moves).
            let new_icn = format!("{} {}", starting_board_setup, move_history_clean.join("|"));
            let old_turn = game.turn;
            game = GameState::new();
            game.setup_position_from_icn(&new_icn);
            game.variant = Some(variant);

            // If the turn didn't change, the move wasn't applied (illegal or unparseable)
            if game.turn == old_turn {
                let result = if is_new_turn {
                    GameResult::Loss
                } else {
                    GameResult::Win
                };
                let white_won = (result == GameResult::Win) == new_plays_white;
                let result_str = if white_won { "1-0" } else { "0-1" };
                return game_outcome!(result, "illegal move", result_str);
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
            if let Some(terminal) = detect_terminal_state(&mut game) {
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
                    TerminalState::Draw(reason) => {
                        return game_outcome!(GameResult::Draw, reason, "1/2-1/2");
                    }
                }
            }

            termination_reason = Some("engine failure");
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
            };
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

    format!(
        "{} 0/{} 1 {} {} {}",
        turn_str, move_limit, promo_token, bounds_token, pieces_str
    )
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
        "HydroChess New"
    } else {
        "HydroChess Old"
    };
    let black = if new_plays_white {
        "HydroChess Old"
    } else {
        "HydroChess New"
    };
    icn.push_str(&format!("[White \"{}\"] ", white));
    icn.push_str(&format!("[Black \"{}\"] ", black));

    if let Some(r) = reason {
        let term = match r {
            "material adjudication" => {
                format!(
                    "Material adjudication (|eval| >= {} cp)",
                    config.adjudication_threshold
                )
            }
            "checkmate" => "Checkmate".to_string(),
            "stalemate" => "Draw by stalemate".to_string(),
            "fifty-move rule" => "Draw by fifty-move rule".to_string(),
            "threefold repetition" => "Draw by threefold repetition".to_string(),
            "timeout" => "Loss on time".to_string(),
            "illegal move" => "Loss on illegal move".to_string(),
            "engine failure" => "Loss on engine failure".to_string(),
            "max_moves" => "Draw by maximum moves reached".to_string(),
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

fn print_status_line(previous_len: &mut usize, line: &str) {
    let clear_width = (*previous_len).max(line.len());
    print!("\r{:<width$}", line, width = clear_width);
    std::io::stdout().flush().unwrap();
    *previous_len = line.len();
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Run {
            new_bin,
            old_bin,
            elo0,
            elo1,
            alpha,
            beta,
            tc,
            concurrency,
            max_games,
            min_games,
            variants,
            adjudication,
            games,
            results,
            max_moves,
            search_noise,
            old_strength,
            verbose,
        }) => {
            let actual_new_bin = if let Some(path) = new_bin {
                path
            } else {
                println!("No --new-bin provided. Building current source...");
                let status = Command::new("cargo")
                    .args(["build", "--release", "--features=sprt", "--bin", "sprt"])
                    .status()
                    .expect("Failed to execute cargo build");
                if !status.success() {
                    panic!("Failed to build new binary automatically.");
                }
                // Copy to a unique name to avoid file-locking when the manager
                let ext = std::env::consts::EXE_EXTENSION;
                let src = if ext.is_empty() {
                    "target/release/sprt".to_string()
                } else {
                    format!("target/release/sprt.{}", ext)
                };
                let dst = if ext.is_empty() {
                    "target/release/sprt_new".to_string()
                } else {
                    format!("target/release/sprt_new.{}", ext)
                };

                std::fs::copy(&src, &dst).unwrap_or_else(|e| {
                    panic!("Failed to copy {} to {}: {}", src, dst, e);
                });
                dst
            };

            let mut config = Config {
                elo0,
                elo1,
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
                variants: variants.split(',').map(Variant::parse).collect(),
                adjudication_threshold: adjudication,
                new_bin: actual_new_bin,
                old_bin,
                max_moves,
                search_noise,
                old_strength,
                verbose,
            };
            let games_path = games;
            let results_path = results;

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

            println!(
                "Starting SPRT: elo0={}, elo1={}, tc={}, concurrency={}\n",
                config.elo0, config.elo1, tc, config.concurrency
            );

            let (lower, upper) = (
                (config.beta / (1.0 - config.alpha)).ln(),
                ((1.0 - config.beta) / config.alpha).ln(),
            );
            let mut wins = 0;
            let mut losses = 0;
            let mut draws = 0;
            let mut timeout_losses = 0;
            let mut game_logs = Vec::new();
            let mut per_variant_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();
            let mut last_status_len = 0;

            let (tx, rx) = std::sync::mpsc::channel();
            let config_clone = config.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.concurrency.max(1))
                .build()
                .expect("Failed to build Rayon thread pool");

            std::thread::spawn(move || {
                pool.install(|| {
                    rayon::scope(|scope| {
                        for worker_idx in 0..config_clone.concurrency.max(1) {
                            let tx = tx.clone();
                            let config = config_clone.clone();
                            scope.spawn(move |_| {
                                let mut pair_idx = worker_idx;
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

            for pair_outcomes in rx {
                for outcome in pair_outcomes {
                    if outcome.termination_reason == "timeout" {
                        timeout_losses += 1;
                        if config.verbose {
                            println!(
                                "\nALERT: Game {} ended by timeout [{}].",
                                outcome.game_idx, outcome.variant_name
                            );
                        }
                    }

                    match outcome.result {
                        GameResult::Win => wins += 1,
                        GameResult::Loss => losses += 1,
                        GameResult::Draw => draws += 1,
                    }
                    game_logs.push(outcome.icn);
                    let stats = per_variant_stats
                        .entry(outcome.variant_name)
                        .or_insert((0, 0, 0));
                    match outcome.result {
                        GameResult::Win => stats.0 += 1,
                        GameResult::Loss => stats.1 += 1,
                        GameResult::Draw => stats.2 += 1,
                    }
                }

                let llr = calculate_llr(wins, losses, draws, config.elo0, config.elo1);
                let (elo, err) = estimate_elo(wins, losses, draws);
                let status_line = format!(
                    "Games: {} | W: {} L: {} D: {} | Elo: {:.1} +/- {:.1} | LLR: {:.2} [{:.2}, {:.2}]",
                    wins + losses + draws,
                    wins,
                    losses,
                    draws,
                    elo,
                    err,
                    llr,
                    lower,
                    upper
                );
                print_status_line(&mut last_status_len, &status_line);
                if wins + losses + draws >= config.min_games {
                    if llr >= upper {
                        println!("\nSPRT: PASS");
                        STOP.store(true, Ordering::SeqCst);
                        break;
                    } else if llr <= lower {
                        println!("\nSPRT: FAIL");
                        STOP.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
            if USER_STOP.load(Ordering::SeqCst) {
                println!("\nRun stopped by user.");
            }

            println!("\nFinal Summary:");
            let (elo, err) = estimate_elo(wins, losses, draws);
            println!("  Elo: {:.1} +/- {:.1}", elo, err);
            println!("  Record: {}W - {}L - {}D", wins, losses, draws);
            if timeout_losses > 0 {
                println!("  ALERT: {} games ended by timeout", timeout_losses);
            }
            println!("\nPer-Variant Breakdown:");
            let mut variant_names: Vec<_> = per_variant_stats.keys().collect();
            variant_names.sort();
            for name in variant_names {
                let (vw, vl, vd) = per_variant_stats[name];
                let (velo, verr) = estimate_elo(vw, vl, vd);
                println!(
                    "  [{}]: {}W - {}L - {}D, Elo: {:.1} +/- {:.1}",
                    name, vw, vl, vd, velo, verr
                );
            }

            if let Some(path) = games_path {
                let json_data = serde_json::to_string_pretty(&game_logs).unwrap();
                std::fs::write(path, json_data).expect("Failed to write JSON output");
            }
            if let Some(path) = results_path {
                #[derive(Serialize)]
                struct FinalResults {
                    wins: usize,
                    losses: usize,
                    draws: usize,
                    timeout_losses: usize,
                    elo: f64,
                    elo_error: f64,
                    llr: f64,
                    total_games: usize,
                    per_variant: HashMap<String, (usize, usize, usize)>,
                }
                let final_llr = calculate_llr(wins, losses, draws, config.elo0, config.elo1);
                let (final_elo, final_err) = estimate_elo(wins, losses, draws);
                let res = FinalResults {
                    wins,
                    losses,
                    draws,
                    timeout_losses,
                    elo: final_elo,
                    elo_error: final_err,
                    llr: final_llr,
                    total_games: wins + losses + draws,
                    per_variant: per_variant_stats,
                };
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
                let v = Variant::parse(&variant);
                let mut engine = Engine::from_icn_native(icn.as_str(), strength_level);
                engine.set_clock(wtime, btime, winc, binc);
                engine.game_mut().variant = Some(v);
                if let Some(terminal) = detect_terminal_state(engine.game_mut()) {
                    match terminal {
                        TerminalState::Checkmate { white_won } => {
                            eprintln!(
                                "terminal checkmate winner {}",
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
        None => {
            println!("Use --help for usage. SPRT CLI requires a subcommand.");
        }
    }
}


