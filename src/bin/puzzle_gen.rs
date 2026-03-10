use csv::Writer;
use hydrochess_wasm::Variant;
use hydrochess_wasm::board::{Coordinate, PieceType, PlayerColor};
use hydrochess_wasm::evaluation::get_piece_value_base;
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::moves::{Move, MoveGenContext, MoveList};
use hydrochess_wasm::search;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Serialize, Clone)]
struct PuzzleRecord {
    position_icn: String,
    side_to_move: String,
    solution_moves: String,
    initial_eval: i32,
    final_eval: i32,
    themes: String,
    difficulty: i32,
}

fn win_chances(score: i32) -> f64 {
    2.0 / (1.0 + (-0.00368208 * score as f64).exp()) - 1.0
}

fn is_valid_attack(best_score: i32, second_score: Option<i32>) -> bool {
    let best_win = win_chances(best_score);
    match second_score {
        None => true,
        Some(s) => {
            let second_win = win_chances(s);
            best_win > second_win + 0.7
        }
    }
}

fn main() {
    println!("Starting Puzzle Generator...");

    let games_path = Path::new("generated_games.txt");
    if !games_path.exists() {
        eprintln!("Error: generated_games.txt not found.");
        return;
    }

    let file = File::open(games_path).expect("Failed to open generated_games.txt");
    let reader = BufReader::new(file);

    println!("Reading generated_games.txt...");
    let lines: Vec<String> = reader
        .lines()
        .map(|l| l.expect("Could not read line"))
        .collect();
    println!("Successfully loaded {} games.", lines.len());

    let seen_positions = Arc::new(Mutex::new(FxHashSet::default()));

    // Initialize real-time CSV writer on a background thread
    let (tx, rx) = mpsc::channel::<PuzzleRecord>();
    let writer_handle = thread::spawn(move || {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("puzzles.csv")
            .expect("Failed to open/create puzzles.csv");

        let is_empty = file.metadata().map(|m| m.len() == 0).unwrap_or(true);
        let mut wtr = Writer::from_writer(file);

        if is_empty {
            wtr.write_record([
                "position_icn",
                "side_to_move",
                "solution_moves",
                "initial_eval",
                "final_eval",
                "themes",
                "difficulty",
            ])
            .expect("Failed to write CSV headers");
        }

        for puzzle in rx {
            wtr.serialize(puzzle).expect("Failed to serialize puzzle");
            let _ = wtr.flush();
        }
    });

    println!(
        "Generating puzzles using {} threads...",
        rayon::current_num_threads()
    );

    let puzzles_generated = AtomicUsize::new(0);
    let completed_indices = Arc::new(Mutex::new(FxHashSet::default()));

    // Load checkpoint
    let checkpoint_path = Path::new("checkpoint.txt");
    let start_idx = if checkpoint_path.exists() {
        std::fs::read_to_string(checkpoint_path)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0)
    } else {
        0
    };

    if start_idx > 0 {
        println!("Resuming from game index {}...", start_idx);
    }
    let last_safe_idx = Arc::new(AtomicUsize::new(start_idx));

    let pb = ProgressBar::new(lines.len() as u64);
    pb.set_position(start_idx as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {msg} | {per_sec}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    lines
        .par_iter()
        .enumerate()
        .skip(start_idx)
        .for_each(|(idx, line)| {
            let seen_positions = Arc::clone(&seen_positions);
            let tx = tx.clone();
            let puzzles_generated = &puzzles_generated;
            let pb = &pb;
            let lines_len = lines.len();
            let last_safe_idx = Arc::clone(&last_safe_idx);
            let completed_indices = Arc::clone(&completed_indices);
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 2 {
                let variant_str = parts[0];
                if let Ok(variant) = Variant::from_str(variant_str) {
                    let actual_variant = if variant == Variant::Chess {
                        None
                    } else {
                        Some(variant)
                    };

                    if let Some(variant) = actual_variant {
                        let move_strs = &parts[2..];
                        let mut state = GameState::new();
                        state.setup_position_from_icn(variant.starting_icn());

                        let starting_icn = variant.starting_icn();
                        let mut current_icn = starting_icn.to_string();

                        for (i, m_str) in move_strs.iter().enumerate() {
                            if !is_suitable_for_puzzle(&state) {
                                if let Some(mv) = parse_compact_move_fast(m_str, &state) {
                                    state.make_move(&mv);
                                    if i == 0 {
                                        current_icn.push(' ');
                                    } else {
                                        current_icn.push('|');
                                    }
                                    current_icn.push_str(m_str);
                                }
                                continue;
                            }

                            if seen_positions.lock().unwrap().insert(current_icn.clone())
                                && let Some(puzzle) =
                                    analyze_position(&mut state, current_icn.clone())
                            {
                                puzzles_generated.fetch_add(1, Ordering::SeqCst);
                                let _ = tx.send(puzzle);
                            }

                            if let Some(mv) = parse_compact_move_fast(m_str, &state) {
                                state.make_move(&mv);
                                if i == 0 {
                                    current_icn.push(' ');
                                } else {
                                    current_icn.push('|');
                                }
                                current_icn.push_str(m_str);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }

            // Update progress bar
            pb.inc(1);

            // Monotonic checkpoint logic
            completed_indices.lock().unwrap().insert(idx);
            let mut current_safe = last_safe_idx.load(Ordering::SeqCst);
            while completed_indices.lock().unwrap().contains(&(current_safe)) {
                if last_safe_idx
                    .compare_exchange(
                        current_safe,
                        current_safe + 1,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
                {
                    completed_indices.lock().unwrap().remove(&current_safe);
                    current_safe += 1;
                } else {
                    current_safe = last_safe_idx.load(Ordering::SeqCst);
                }
            }

            if idx % 10 == 0 || idx == lines_len - 1 {
                let pg = puzzles_generated.load(Ordering::Relaxed);
                let gp = pb.position();
                let ratio = if pg > 0 { gp as f64 / pg as f64 } else { 0.0 };
                pb.set_message(format!("Puzzles: {} | Ratio: {:.1} G/P", pg, ratio));

                // Periodically save checkpoint using the monotonic high-water mark
                if idx % 10 == 0 {
                    let safe_to_save = last_safe_idx.load(Ordering::SeqCst);
                    let mut f =
                        File::create("checkpoint.txt").expect("Failed to create checkpoint.txt");
                    write!(f, "{}", safe_to_save).expect("Failed to write checkpoint");
                }
            }
        });

    drop(tx); // Signal writer to finish
    let _ = writer_handle.join();

    pb.finish_with_message("Done!");
    println!("Puzzle Generator finished.");
}

#[inline]
fn parse_compact_move_fast(s: &str, state: &GameState) -> Option<Move> {
    let sep = s.find('>')?;
    let from_s = &s[..sep];
    let to_s = &s[sep + 1..];

    let f_comma = from_s.find(',')?;
    let fx = from_s[..f_comma].parse::<i64>().ok()?;
    let fy = from_s[f_comma + 1..].parse::<i64>().ok()?;

    let t_comma = to_s.find(',')?;
    let tx = to_s[..t_comma].parse::<i64>().ok()?;

    let rest = &to_s[t_comma + 1..];
    let mut y_end = 0;
    let bytes = rest.as_bytes();
    if y_end < bytes.len() && bytes[y_end] == b'-' {
        y_end += 1;
    }
    while y_end < bytes.len() && bytes[y_end].is_ascii_digit() {
        y_end += 1;
    }

    let ty = rest[..y_end].parse::<i64>().ok()?;
    let promo_s = &rest[y_end..];
    let promo = if promo_s.is_empty() {
        None
    } else {
        Some(PieceType::from_site_code(promo_s))
    };

    let moving_piece = state.board.get_piece(fx, fy)?;

    let mut rook_coord = None;
    if moving_piece.piece_type() == PieceType::King && (tx - fx).abs() > 1 {
        let is_white = moving_piece.color() == PlayerColor::White;
        if tx > fx {
            let mut max_x = tx;
            for (x, y, p) in state.board.iter_pieces_by_color(is_white) {
                if y == fy && x > tx && p.piece_type() != PieceType::Void {
                    max_x = max_x.max(x);
                }
            }
            rook_coord = Some(Coordinate { x: max_x, y: fy });
        } else {
            let mut min_x = tx;
            for (x, y, p) in state.board.iter_pieces_by_color(is_white) {
                if y == fy && x < tx && p.piece_type() != PieceType::Void {
                    min_x = min_x.min(x);
                }
            }
            rook_coord = Some(Coordinate { x: min_x, y: fy });
        }
    }

    Some(Move {
        from: Coordinate { x: fx, y: fy },
        to: Coordinate { x: tx, y: ty },
        piece: moving_piece,
        promotion: promo,
        rook_coord,
    })
}

fn material_count(state: &GameState, color: PlayerColor) -> i32 {
    let mut total = 0;
    for (_, _, p) in state.board.iter() {
        if p.color() == color {
            total += get_piece_value_base(p.piece_type());
        }
    }
    total
}

fn material_diff(state: &GameState, color: PlayerColor) -> i32 {
    material_count(state, color) - material_count(state, color.opponent())
}

fn is_suitable_for_puzzle(state: &GameState) -> bool {
    let total_pieces = state.white_piece_count + state.black_piece_count;
    if total_pieces < 6 {
        return false;
    }

    true
}

fn detect_themes(
    initial_state: &mut GameState,
    final_state: &GameState,
    pv: &[hydrochess_wasm::moves::Move],
    winner: PlayerColor,
    themes: &mut std::collections::HashSet<String>,
) {
    if final_state.is_in_check() && final_state.get_legal_moves().is_empty() {
        themes.insert("mate".to_string());
        if pv.len() == 1 {
            themes.insert("mateIn1".to_string());
        } else if pv.len() == 3 {
            themes.insert("mateIn2".to_string());
        } else if pv.len() == 5 {
            themes.insert("mateIn3".to_string());
        }
    } else {
        let final_cp = search::get_best_moves_multipv(
            &mut final_state.clone(),
            10,
            u128::MAX,
            u128::MAX,
            1,
            true,
            true,
        )
        .lines
        .first()
        .map(|l| l.score)
        .unwrap_or(0);
        if final_cp.abs() > 600 {
            themes.insert("crushing".to_string());
        } else if final_cp.abs() > 200 {
            themes.insert("advantage".to_string());
        }
    }

    let mut undo_stack = Vec::new();
    for (i, mv) in pv.iter().enumerate() {
        if i % 2 == 0 {
            let before_diff = material_diff(initial_state, winner);
            undo_stack.push(initial_state.make_move(mv));
            let after_diff = material_diff(initial_state, winner);
            if after_diff < before_diff - 100 {
                themes.insert("sacrifice".to_string());
            }
        } else {
            undo_stack.push(initial_state.make_move(mv));
        }
    }

    // Undo all moves in reverse order
    for mv in pv.iter().rev() {
        if let Some(undo) = undo_stack.pop() {
            initial_state.undo_move(mv, undo);
        }
    }
}

fn additional_theme_logic(
    initial_state: &mut GameState,
    pv: &[hydrochess_wasm::moves::Move],
    winner: PlayerColor,
    themes: &mut std::collections::HashSet<String>,
) {
    let first_move = &pv[0];
    let undo = initial_state.make_move(first_move);

    if let Some(piece) = initial_state
        .board
        .get_piece(first_move.to.x, first_move.to.y)
    {
        let empty_rights = std::collections::HashSet::default();
        let empty_pinned = FxHashMap::default();
        let ctx = MoveGenContext {
            special_rights: &empty_rights,
            en_passant: &None,
            game_rules: &initial_state.game_rules,
            indices: &initial_state.spatial_indices,
            enemy_king_pos: None,
            pinned: &empty_pinned,
        };

        let mut move_list = MoveList::new();
        hydrochess_wasm::moves::get_pseudo_legal_moves_for_piece_into(
            &initial_state.board,
            &piece,
            &first_move.to,
            &ctx,
            &mut move_list,
        );

        let mut targets = 0;
        for m in move_list {
            if let Some(target_piece) = initial_state.board.get_piece(m.to.x, m.to.y)
                && target_piece.color() == winner.opponent()
            {
                let is_king = target_piece.piece_type() == PieceType::King;
                let is_undefended = !hydrochess_wasm::moves::is_square_attacked(
                    &initial_state.board,
                    &m.to,
                    winner.opponent(),
                    &initial_state.spatial_indices,
                );
                let is_value_jump = get_piece_value_base(target_piece.piece_type())
                    > get_piece_value_base(piece.piece_type());

                if is_king || is_undefended || is_value_jump {
                    targets += 1;
                }
            }
        }
        if targets >= 2 {
            themes.insert("fork".to_string());
        }

        if first_move_in_check(initial_state) {
            let checkers = get_checkers_count(initial_state);
            if checkers > 1 {
                themes.insert("doubleCheck".to_string());
            }
        }
    }
    initial_state.undo_move(first_move, undo);

    if pv
        .iter()
        .step_by(2)
        .any(|m| initial_state.board.get_piece(m.to.x, m.to.y).is_some())
    {
        themes.insert("capture".to_string());
    }

    // Pin / Discovered attack detection
    let mut undo_stack = Vec::new();
    for (i, mv) in pv.iter().enumerate() {
        if i % 2 == 0 {
            let from_piece = initial_state.board.get_piece(mv.from.x, mv.from.y);
            let to_target = initial_state.board.get_piece(mv.to.x, mv.to.y);

            if to_target.is_some()
                && !hydrochess_wasm::moves::is_square_attacked(
                    &initial_state.board,
                    &mv.to,
                    winner.opponent(),
                    &initial_state.spatial_indices,
                )
            {
                themes.insert("hangingPiece".to_string());
            }

            if let Some(_p) = from_piece {
                let mut sandbox_board = initial_state.board.clone();
                sandbox_board.remove_piece(&mv.from.x, &mv.from.y);

                let opponent_king = if winner == PlayerColor::White {
                    initial_state.black_royals.first().copied()
                } else {
                    initial_state.white_royals.first().copied()
                };
                if let Some(ksq) = opponent_king {
                    let revealed_check = hydrochess_wasm::moves::is_square_attacked(
                        &sandbox_board,
                        &ksq,
                        winner,
                        &initial_state.spatial_indices,
                    );
                    if revealed_check && !initial_state.is_in_check() {
                        themes.insert("discoveredAttack".to_string());
                    }
                }
            }

            let opponent_color = winner.opponent();
            let opp_king_pos = if opponent_color == PlayerColor::White {
                initial_state.white_royals.first().copied()
            } else {
                initial_state.black_royals.first().copied()
            };
            if let Some(okp) = opp_king_pos {
                for (ox, oy, op) in initial_state
                    .board
                    .iter_pieces_by_color(opponent_color == PlayerColor::White)
                {
                    if op.piece_type() == PieceType::King {
                        continue;
                    }
                    let mut sandbox = initial_state.board.clone();
                    sandbox.remove_piece(&ox, &oy);
                    if hydrochess_wasm::moves::is_square_attacked(
                        &sandbox,
                        &okp,
                        winner,
                        &initial_state.spatial_indices,
                    ) {
                        themes.insert("pin".to_string());
                        break;
                    }
                }
            }
            undo_stack.push(initial_state.make_move(mv));
        } else {
            undo_stack.push(initial_state.make_move(mv));
        }
    }

    // Undo all moves in reverse order
    for mv in pv.iter().rev() {
        if let Some(undo) = undo_stack.pop() {
            initial_state.undo_move(mv, undo);
        }
    }
}

fn first_move_in_check(state: &GameState) -> bool {
    state.is_in_check()
}

fn get_checkers_count(state: &GameState) -> usize {
    let mut count = 0;
    let our_color = state.turn;
    let their_color = our_color.opponent();
    let king_sq = if our_color == PlayerColor::White {
        state.white_royals.first().copied()
    } else {
        state.black_royals.first().copied()
    };
    let king_sq = if let Some(ks) = king_sq {
        ks
    } else {
        return 0;
    };

    for (ax, ay, p) in state.board.iter() {
        if p.color() == their_color
            && hydrochess_wasm::moves::is_piece_attacking_square(
                &state.board,
                &p,
                &Coordinate::new(ax, ay),
                &king_sq,
                &state.spatial_indices,
                &state.game_rules,
            )
        {
            count += 1;
        }
    }
    count
}

fn cook_mate(
    state: &mut GameState,
    winner: PlayerColor,
    depth: i32,
) -> Option<Vec<hydrochess_wasm::moves::Move>> {
    if state.is_in_check() && state.get_legal_moves().is_empty() {
        return Some(vec![]);
    }
    if depth <= 0 {
        return None;
    }

    if state.turn == winner {
        let res = search::get_best_moves_multipv(state, 12, u128::MAX, u128::MAX, 2, true, true);
        if res.lines.is_empty() {
            return None;
        }
        let best = &res.lines[0];
        if !search::is_win(best.score) {
            return None;
        }
        if res.lines.len() >= 2 && !is_valid_attack(best.score, Some(res.lines[1].score)) {
            return None;
        }
        let mv = best.mv;
        let undo = state.make_move(&mv);
        let pv = cook_mate(state, winner, depth - 1);
        state.undo_move(&mv, undo);

        let mut pv = pv?;
        pv.insert(0, mv);
        Some(pv)
    } else {
        let res = search::get_best_moves_multipv(state, 10, u128::MAX, u128::MAX, 1, true, true);
        if res.lines.is_empty() {
            return None;
        }
        let best = &res.lines[0];
        let mv = best.mv;
        let undo = state.make_move(&mv);
        let pv = cook_mate(state, winner, depth - 1);
        state.undo_move(&mv, undo);

        let mut pv = pv?;
        pv.insert(0, mv);
        Some(pv)
    }
}

fn cook_advantage(
    state: &mut GameState,
    winner: PlayerColor,
    depth: i32,
) -> Option<Vec<hydrochess_wasm::moves::Move>> {
    if depth <= 0 {
        return Some(vec![]);
    }

    if state.turn == winner {
        let res = search::get_best_moves_multipv(state, 12, u128::MAX, u128::MAX, 2, true, true);
        if res.lines.is_empty() {
            return None;
        }
        let best = &res.lines[0];
        if best.score < 200 {
            return Some(vec![]);
        }
        if res.lines.len() >= 2 && !is_valid_attack(best.score, Some(res.lines[1].score)) {
            return Some(vec![]);
        }
        let mv = best.mv;
        let undo = state.make_move(&mv);
        let pv = cook_advantage(state, winner, depth - 1);
        state.undo_move(&mv, undo);

        let mut pv = pv?;
        pv.insert(0, mv);
        Some(pv)
    } else {
        let res = search::get_best_moves_multipv(state, 10, u128::MAX, u128::MAX, 1, true, true);
        if res.lines.is_empty() {
            return Some(vec![]);
        }
        let best = &res.lines[0];
        let mv = best.mv;
        let undo = state.make_move(&mv);
        let pv = cook_advantage(state, winner, depth - 1);
        state.undo_move(&mv, undo);

        let mut pv = pv?;
        pv.insert(0, mv);
        Some(pv)
    }
}

fn analyze_position(state: &mut GameState, current_icn: String) -> Option<PuzzleRecord> {
    let winner = state.turn;
    let init_res = search::get_best_moves_multipv(state, 10, u128::MAX, u128::MAX, 1, true, true);
    if init_res.lines.is_empty() {
        return None;
    }
    let initial_eval = init_res.lines[0].score;
    if initial_eval.abs() > 300 {
        return None;
    }

    let tactical_res =
        search::get_best_moves_multipv(state, 14, u128::MAX, u128::MAX, 2, true, true);
    if tactical_res.lines.is_empty() {
        return None;
    }
    let best = &tactical_res.lines[0];
    let is_mate = search::is_win(best.score);
    if !is_mate && best.score < 300 {
        return None;
    }
    if win_chances(best.score) < win_chances(initial_eval) + 0.6 && !is_mate {
        return None;
    }
    if !is_mate && best.score < 400 && material_diff(state, winner) > -1 {
        return None;
    }
    if tactical_res.lines.len() >= 2
        && !is_valid_attack(best.score, Some(tactical_res.lines[1].score))
    {
        return None;
    }

    let mut solution_pv = if is_mate {
        cook_mate(state, winner, 12)?
    } else {
        cook_advantage(state, winner, 10)?
    };

    while solution_pv.len() % 2 == 0 {
        solution_pv.pop();
    }
    if solution_pv.len() < 3 {
        return None;
    }

    let mut themes = std::collections::HashSet::new();

    // Theme detection functions handle their own make/undo cycles to restore state.
    detect_themes(state, &state.clone(), &solution_pv, winner, &mut themes);
    additional_theme_logic(state, &solution_pv, winner, &mut themes);

    if themes.is_empty() {
        themes.insert("tactical".to_string());
    }

    let solution_moves = solution_pv
        .iter()
        .map(|m| {
            let prom = m.promotion.map(|p| p.to_site_code()).unwrap_or_default();
            format!("{},{}>{},{}{}", m.from.x, m.from.y, m.to.x, m.to.y, prom)
        })
        .collect::<Vec<_>>()
        .join(" ");

    let side = if state.turn == PlayerColor::White {
        "White"
    } else {
        "Black"
    };
    let difficulty = (solution_pv.len() as i32 * 100)
        + (best.score.abs().min(1000) / 10)
        + (if is_mate { 300 } else { 0 });

    Some(PuzzleRecord {
        position_icn: current_icn,
        side_to_move: side.to_string(),
        solution_moves,
        initial_eval,
        final_eval: best.score,
        themes: themes.into_iter().collect::<Vec<_>>().join(", "),
        difficulty: difficulty.clamp(600, 3000),
    })
}
