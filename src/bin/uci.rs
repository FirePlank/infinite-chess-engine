/// UCI protocol bridge for the infinite-chess engine, constrained to standard 8x8 chess.
///
/// Coordinate mapping:
///   UCI file a-h  <->  internal x 1-8
///   UCI rank 1-8  <->  internal y 1-8
///
/// The world border is fixed at (1, 8, 1, 8) and the Chess variant starting ICN is used.

use hydrochess_wasm::board::PieceType;
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::moves::set_world_bounds;
use hydrochess_wasm::search;
use std::io::{self, BufRead, Write};

const ENGINE_NAME: &str = "HydroChess";
const ENGINE_AUTHOR: &str = "FirePlank";

/// The standard Chess variant ICN (8x8, bounded 1,8,1,8).
const CHESS_START_ICN: &str =
    "[Variant \"Chess\"] w 0/100 1 (8|1) 1,8,1,8 P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|\
     p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|\
     R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|\
     B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+";

// ---------------------------------------------------------------------------
// Coordinate conversion helpers
// ---------------------------------------------------------------------------

/// UCI square string (e.g. "e2") -> internal (x, y).
/// Returns None on invalid input.
fn uci_sq_to_xy(sq: &str) -> Option<(i64, i64)> {
    let mut chars = sq.chars();
    let file = chars.next()?;
    let rank = chars.next()?;
    let x = match file {
        'a' => 1,
        'b' => 2,
        'c' => 3,
        'd' => 4,
        'e' => 5,
        'f' => 6,
        'g' => 7,
        'h' => 8,
        _ => return None,
    };
    let y = rank.to_digit(10)? as i64;
    if !(1..=8).contains(&y) {
        return None;
    }
    Some((x, y))
}

/// Internal (x, y) -> UCI square string.
fn xy_to_uci_sq(x: i64, y: i64) -> String {
    let file = match x {
        1 => 'a',
        2 => 'b',
        3 => 'c',
        4 => 'd',
        5 => 'e',
        6 => 'f',
        7 => 'g',
        8 => 'h',
        _ => '?',
    };
    format!("{}{}", file, y)
}

/// Promotion piece type -> UCI promotion char (lowercase).
fn promo_to_uci(pt: PieceType) -> char {
    match pt {
        PieceType::Queen => 'q',
        PieceType::Rook => 'r',
        PieceType::Bishop => 'b',
        PieceType::Knight => 'n',
        _ => 'q', // fallback
    }
}

/// UCI promotion char -> internal PieceType.
fn uci_promo_to_piece(c: char) -> Option<PieceType> {
    match c.to_ascii_lowercase() {
        'q' => Some(PieceType::Queen),
        'r' => Some(PieceType::Rook),
        'b' => Some(PieceType::Bishop),
        'n' => Some(PieceType::Knight),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// FEN parsing -> ICN string
// ---------------------------------------------------------------------------

/// Convert a FEN string into our ICN format, ready for `setup_position_from_icn`.
/// Returns a full ICN string on success, or an error message on failure.
fn fen_to_icn(fen: &str) -> Result<String, String> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(format!("Invalid FEN (too few fields): {}", fen));
    }

    let piece_placement = parts[0];
    let active_color = parts[1];
    let castling = parts[2];
    let en_passant_str = parts[3];
    let halfmove: u32 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let fullmove: u32 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    // --- Turn ---
    let turn = match active_color {
        "w" => "w",
        "b" => "b",
        _ => return Err(format!("Invalid active color: {}", active_color)),
    };

    // --- Parse piece placement (rank 8 down to rank 1 in FEN) ---
    // We'll collect pieces as ICN tokens.
    // FEN ranks are ordered 8..1 (top to bottom), files a..h (left to right).
    let mut piece_tokens: Vec<String> = Vec::new();

    let fen_ranks: Vec<&str> = piece_placement.split('/').collect();
    if fen_ranks.len() != 8 {
        return Err(format!(
            "Expected 8 ranks in FEN, got {}",
            fen_ranks.len()
        ));
    }

    // Track which squares have pieces that have moved (to determine special rights).
    // In chess, rooks and kings on their starting squares get special rights (castling)
    // and pawns on their starting ranks get double-push rights.
    // We approximate this from FEN castling availability.
    let white_ks = castling.contains('K');
    let white_qs = castling.contains('Q');
    let black_ks = castling.contains('k');
    let black_qs = castling.contains('q');

    // En passant square
    let ep_sq: Option<(i64, i64)> = if en_passant_str == "-" {
        None
    } else {
        uci_sq_to_xy(en_passant_str)
    };

    for (rank_idx, rank_str) in fen_ranks.iter().enumerate() {
        let y = 8 - rank_idx as i64; // FEN rank 0 = board rank 8
        let mut x: i64 = 1;

        for ch in rank_str.chars() {
            if ch.is_ascii_digit() {
                x += ch as i64 - '0' as i64;
            } else {
                // Determine piece type and color from FEN char
                let (piece_char, is_white) = if ch.is_uppercase() {
                    (ch.to_ascii_lowercase(), true)
                } else {
                    (ch, false)
                };

                let (icn_code, has_special_right) = fen_char_to_icn(
                    piece_char,
                    is_white,
                    x,
                    y,
                    white_ks,
                    white_qs,
                    black_ks,
                    black_qs,
                    ep_sq,
                );

                let token = if has_special_right {
                    format!("{}{},{}+", icn_code, x, y)
                } else {
                    format!("{},{},{}", icn_code, x, y)
                };
                // Reformat: ICN uses `CODE x,y[+]` where CODE is uppercase for white
                // Actually our ICN format is: `Px,y+` for white pawn with special right
                // The code already includes the case letter, just assemble correctly:
                piece_tokens.push(token);
                x += 1;
            }
        }
    }

    let ep_move_token: Option<String> = ep_sq.map(|(_ex, ey)| {
        if ey == 6 {
            // Black pawn double-pushed from y=7 to y=5; ep target is y=6
            let ex = ep_sq.unwrap().0;
            format!("{},7->{},{}", ex, ex, 5)
        } else {
            // White pawn double-pushed from y=2 to y=4; ep target is y=3
            let ex = ep_sq.unwrap().0;
            format!("{},2->{},{}", ex, ex, 4)
        }
    });

    // Assemble ICN
    let pieces_str = piece_tokens.join("|");
    let mut icn = format!(
        "[Variant \"Chess\"] {} {}/100 {} (8|1) 1,8,1,8 {}",
        turn, halfmove, fullmove, pieces_str
    );

    if let Some(ep) = ep_move_token {
        icn.push(' ');
        icn.push_str(&ep);
    }

    Ok(icn)
}

/// Map a FEN piece character + context to ICN piece code and whether it has special rights.
/// Returns (icn_code_string, has_special_right).
fn fen_char_to_icn(
    piece_char: char,
    is_white: bool,
    x: i64,
    y: i64,
    white_ks: bool,
    white_qs: bool,
    black_ks: bool,
    black_qs: bool,
    ep_sq: Option<(i64, i64)>,
) -> (String, bool) {
    let code = match piece_char {
        'p' => {
            if is_white {
                "P".to_string()
            } else {
                "p".to_string()
            }
        }
        'r' => {
            if is_white {
                "R".to_string()
            } else {
                "r".to_string()
            }
        }
        'n' => {
            if is_white {
                "N".to_string()
            } else {
                "n".to_string()
            }
        }
        'b' => {
            if is_white {
                "B".to_string()
            } else {
                "b".to_string()
            }
        }
        'q' => {
            if is_white {
                "Q".to_string()
            } else {
                "q".to_string()
            }
        }
        'k' => {
            if is_white {
                "K".to_string()
            } else {
                "k".to_string()
            }
        }
        _ => "P".to_string(),
    };

    // Determine special rights:
    // - Pawns on their starting rank (white y=2, black y=7) always get double-push right.
    // - Kings/Rooks get castling rights based on FEN castling field.
    let has_special = match piece_char {
        'p' => {
            let starting_rank = if is_white { 2 } else { 7 };
            if y == starting_rank {
                true
            } else {
                false
            }
        }
        'k' => {
            // King has castling right if any castling right for that color exists
            if is_white {
                white_ks || white_qs
            } else {
                black_ks || black_qs
            }
        }
        'r' => {
            // Rook has castling right based on position
            if is_white && y == 1 {
                if x == 1 && white_qs {
                    true
                } else if x == 8 && white_ks {
                    true
                } else {
                    false
                }
            } else if !is_white && y == 8 {
                if x == 1 && black_qs {
                    true
                } else if x == 8 && black_ks {
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    };

    let _ = ep_sq; // used via pattern above
    (code, has_special)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct UciState {
    game: GameState,
}

impl UciState {
    fn new() -> Self {
        set_world_bounds(1, 8, 1, 8);
        let mut game = GameState::new();
        game.setup_position_from_icn(CHESS_START_ICN);
        UciState { game }
    }

    fn reset_to_startpos(&mut self) {
        self.game = GameState::new();
        self.game.setup_position_from_icn(CHESS_START_ICN);
    }

    fn set_fen(&mut self, fen: &str) {
        match fen_to_icn(fen) {
            Ok(icn) => {
                self.game = GameState::new();
                self.game.setup_position_from_icn(&icn);
            }
            Err(e) => {
                eprintln!("info string FEN parse error: {}", e);
            }
        }
    }

    /// Apply a sequence of UCI moves (e.g. ["e2e4", "e7e5"]) to the current position.
    fn apply_moves(&mut self, moves: &[&str]) {
        for mv_str in moves {
            let mv_str = mv_str.trim();
            if mv_str.len() < 4 {
                eprintln!("info string invalid move: {}", mv_str);
                continue;
            }
            let from_sq = &mv_str[0..2];
            let to_sq = &mv_str[2..4];
            let promo_char = mv_str.chars().nth(4);

            let (from_x, from_y) = match uci_sq_to_xy(from_sq) {
                Some(c) => c,
                None => {
                    eprintln!("info string invalid from square: {}", from_sq);
                    continue;
                }
            };
            let (to_x, to_y) = match uci_sq_to_xy(to_sq) {
                Some(c) => c,
                None => {
                    eprintln!("info string invalid to square: {}", to_sq);
                    continue;
                }
            };

            let promo: Option<PieceType> = promo_char.and_then(uci_promo_to_piece);
            let promo_str: Option<String> = promo.map(|p| p.to_str().to_string());
            self.game.make_move_coords(
                from_x,
                from_y,
                to_x,
                to_y,
                promo_str.as_deref(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// "go" command handling
// ---------------------------------------------------------------------------

struct GoParams {
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movetime: Option<u64>,
    depth: Option<usize>,
    infinite: bool,
}

impl GoParams {
    fn parse(tokens: &[&str]) -> Self {
        let mut p = GoParams {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movetime: None,
            depth: None,
            infinite: false,
        };
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => {
                    i += 1;
                    p.wtime = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "btime" => {
                    i += 1;
                    p.btime = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "winc" => {
                    i += 1;
                    p.winc = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "binc" => {
                    i += 1;
                    p.binc = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "movetime" => {
                    i += 1;
                    p.movetime = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "depth" => {
                    i += 1;
                    p.depth = tokens.get(i).and_then(|s| s.parse().ok());
                }
                "infinite" => {
                    p.infinite = true;
                }
                _ => {}
            }
            i += 1;
        }
        p
    }
}

fn run_go(state: &mut UciState, params: GoParams) {
    use hydrochess_wasm::board::PlayerColor;

    // Set clock on a temporary Engine wrapper via search directly.
    // We call search_native equivalent by building time limits ourselves.
    let max_depth = params.depth.unwrap_or(50).clamp(1, 100);

    // Compute time limits (mirrors Engine::effective_time_limit_ms logic, simplified).
    let (opt_ms, max_ms, is_soft): (u128, u128, bool) = if params.infinite {
        (u128::MAX, u128::MAX, true)
    } else if let Some(mt) = params.movetime {
        let ms = mt as u128;
        (ms, ms, false)
    } else if params.depth.is_some() && params.wtime.is_none() && params.btime.is_none() {
        // Fixed depth, no clock: treat as infinite time
        (u128::MAX, u128::MAX, true)
    } else {
        // Clock-based time allocation
        let (remaining_ms_raw, inc_ms_raw) = match state.game.turn {
            PlayerColor::White => (
                params.wtime.unwrap_or(0),
                params.winc.unwrap_or(0),
            ),
            PlayerColor::Black => (
                params.btime.unwrap_or(0),
                params.binc.unwrap_or(0),
            ),
            PlayerColor::Neutral => (0, 0),
        };

        if remaining_ms_raw == 0 && inc_ms_raw == 0 {
            (5000, 5000, true) // fallback: 5 seconds
        } else {
            let move_overhead: u64 = 50;
            let remaining_ms = if remaining_ms_raw > 0 {
                remaining_ms_raw
            } else {
                inc_ms_raw.max(500)
            };
            let scaled_time = remaining_ms.saturating_sub(move_overhead);
            let centi_mtg: i64 = if scaled_time >= 1000 { 5051 } else { (scaled_time as f64 * 5.051) as i64 }.max(100);
            let time_left = (remaining_ms as i64
                + (inc_ms_raw as i64 * (centi_mtg - 100) - move_overhead as i64 * (200 + centi_mtg)) / 100)
                .max(1) as f64;
            let log_adj = (0.3128 * time_left.max(1.0).log10() - 0.4354).max(0.1);
            let log_time_sec = (scaled_time as f64 / 1000.0).max(0.001).log10();
            let opt_constant = (0.0032116 + 0.000321123 * log_time_sec).min(0.00508017);
            let max_constant = (3.3977 + 3.0395 * log_time_sec).max(2.94761);
            let ply = state.game.fullmove_number.saturating_sub(1).saturating_mul(2) as f64
                + if state.game.turn == PlayerColor::Black { 1.0 } else { 0.0 };
            let opt_scale = ((0.0121431 + (ply + 2.94693_f64).powf(0.461073) * opt_constant)
                .min(0.213035 * remaining_ms as f64 / time_left))
                * log_adj;
            let max_scale = max_constant.min(6.67704) + ply / 11.9847;
            let optimum = (opt_scale * time_left) as u64;
            let maximum = ((max_scale * optimum as f64)
                .min(0.825179 * remaining_ms as f64 - move_overhead as f64) - 10.0)
                .max(0.0) as u64;
            let min_think: u64 = 10;
            let optimum = optimum.max(min_think);
            let maximum = maximum.max(optimum);
            let abs_cap = ((remaining_ms as f64) * 0.825 - move_overhead as f64) as u64;
            let optimum = optimum.min(abs_cap.max(min_think));
            let maximum = maximum.min(abs_cap.max(min_think));
            (optimum as u128, maximum as u128, false)
        }
    };

    // Initialize randomness
    search::set_global_params(get_random_seed(), None);

    let result = search::get_best_move_parallel(
        &mut state.game,
        max_depth,
        opt_ms,
        max_ms,
        true, // silent: we emit UCI info ourselves to stdout
        is_soft,
    );

    match result {
        Some((best_move, eval, stats)) => {
            let completed_depth = search::get_completed_depth().max(1);

            let score_str = if eval > search::MATE_SCORE {
                let mate_in = (search::MATE_VALUE - eval + 1) / 2;
                format!("mate {}", mate_in)
            } else if eval < -search::MATE_SCORE {
                let mate_in = (search::MATE_VALUE + eval + 1) / 2;
                format!("mate -{}", mate_in)
            } else {
                format!("cp {}", eval)
            };

            let from_uci = xy_to_uci_sq(best_move.from.x, best_move.from.y);
            let to_uci = xy_to_uci_sq(best_move.to.x, best_move.to.y);
            let promo_str = best_move
                .promotion
                .map(promo_to_uci)
                .map(|c| c.to_string())
                .unwrap_or_default();
            let bm_uci = format!("{}{}{}", from_uci, to_uci, promo_str);

            println!(
                "info depth {} score {} nodes {} hashfull {} pv {}",
                completed_depth,
                score_str,
                stats.nodes,
                stats.tt_fill_permille,
                bm_uci,
            );
            println!("bestmove {}", bm_uci);
        }
        None => {
            // No legal moves (checkmate / stalemate)
            println!("bestmove 0000");
        }
    }
    let _ = io::stdout().flush();
}

fn get_random_seed() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

fn main() {
    let stdin = io::stdin();
    let mut state = UciState::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name {}", ENGINE_NAME);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default 64 min 1 max 65536");
                println!("uciok");
                let _ = io::stdout().flush();
            }
            "isready" => {
                println!("readyok");
                let _ = io::stdout().flush();
            }
            "ucinewgame" => {
                search::reset_search_state();
                state.reset_to_startpos();
            }
            "position" => {
                handle_position(&mut state, &tokens[1..]);
            }
            "go" => {
                let params = GoParams::parse(&tokens[1..]);
                run_go(&mut state, params);
            }
            "stop" => {
                // Signal the search to stop (if running in a thread in the future)
                // For now the search is synchronous so this is a no-op.
            }
            "quit" => {
                break;
            }
            _ => {
                // Unknown command - ignore per UCI spec
            }
        }
    }
}

fn handle_position(state: &mut UciState, tokens: &[&str]) {
    // Syntax: position startpos [moves m1 m2 ...]
    //         position fen <fen_string> [moves m1 m2 ...]
    if tokens.is_empty() {
        return;
    }

    let moves_idx: Option<usize> = tokens.iter().position(|&t| t == "moves");

    match tokens[0] {
        "startpos" => {
            state.reset_to_startpos();
        }
        "fen" => {
            // FEN occupies tokens[1..moves_idx] (or tokens[1..] if no moves keyword)
            let fen_end = moves_idx.unwrap_or(tokens.len());
            let fen_str = tokens[1..fen_end].join(" ");
            state.set_fen(&fen_str);
        }
        _ => {
            // Try treating the whole thing as a FEN
            let fen_end = moves_idx.unwrap_or(tokens.len());
            let fen_str = tokens[0..fen_end].join(" ");
            state.set_fen(&fen_str);
        }
    }

    if let Some(mi) = moves_idx {
        let move_list = &tokens[mi + 1..];
        state.apply_moves(move_list);
    }
}
