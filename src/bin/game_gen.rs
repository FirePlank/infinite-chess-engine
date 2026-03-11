use hydrochess_wasm::Variant;
use hydrochess_wasm::board::PlayerColor;
use hydrochess_wasm::evaluation::insufficient_material::evaluate_insufficient_material;
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::search;
use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

const NUM_GAMES: u64 = 100_000;
const BATCH_SIZE: usize = 100;
const MAX_MOVES: usize = 300;
const ADJUDICATION_THRESHOLD: i32 = 2000; // cp

const VARIANTS: [Variant; 16] = [
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
];

fn main() {
    // Set stack size for search recursion
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .unwrap();

    let mut current_count = 0;
    if let Ok(file) = File::open("generated_games.txt") {
        use std::io::{BufRead, BufReader};
        current_count = BufReader::new(file).lines().count() as u64;
    }

    if current_count >= NUM_GAMES {
        println!("Already generated {} games. Exiting.", current_count);
        return;
    }

    let remaining_games = NUM_GAMES - current_count;
    println!(
        "Starting generation of {} games in batches of {}...",
        remaining_games, BATCH_SIZE
    );

    let output_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("generated_games.txt")
        .expect("Failed to open generated_games.txt");
    let writer = Mutex::new(BufWriter::new(output_file));
    let games_buffer = Mutex::new(Vec::with_capacity(BATCH_SIZE));

    let pb = ProgressBar::new(NUM_GAMES);
    pb.set_position(current_count);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {msg} | {per_sec}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message("Generating Games");

    (0..remaining_games).into_par_iter().for_each(|_| {
        let mut rng = rand::rng();
        let variant = VARIANTS[rng.random_range(0..VARIANTS.len())];
        let mut state = GameState::new();
        state.setup_position_from_icn(variant.starting_icn());

        let mut game_moves = Vec::new();
        let mut ply = 0;

        while ply < MAX_MOVES {
            // Draw detection: Insufficient material
            if let Some(divisor) = evaluate_insufficient_material(&state)
                && divisor == 0
            {
                break;
            }

            // Draw detection: Threefold repetition
            if state.is_repetition(3) {
                break;
            }

            // Draw detection: 50-move rule
            if let Some(limit) = state.game_rules.move_rule_limit
                && state.halfmove_clock >= limit
            {
                break;
            }

            let depth = (5 + rng.random_range(-1i32..=1)) as usize;

            let chosen_move = if ply < 14 {
                // MultiPV = 4 to ensure diversification in opening
                let multi_pv_result = search::get_best_moves_multipv(
                    &mut state,
                    depth,
                    u128::MAX,
                    u128::MAX,
                    4,
                    true,
                    true,
                );

                if multi_pv_result.lines.is_empty() {
                    break;
                }

                // Randomly pick from top moves
                let idx = rng.random_range(0..multi_pv_result.lines.len());
                multi_pv_result.lines[idx].mv
            } else {
                // MultiPV = 1 for speed after opening
                if rng.random_bool(0.08) {
                    // 8% random legal move
                    let legal_moves = state.get_legal_moves();
                    if legal_moves.is_empty() {
                        break;
                    }
                    legal_moves[rng.random_range(0..legal_moves.len())]
                } else {
                    let res = search::get_best_moves_multipv(
                        &mut state,
                        depth,
                        u128::MAX,
                        u128::MAX,
                        1,
                        true,
                        true,
                    );
                    if res.lines.is_empty() {
                        break;
                    }

                    // Adjudication check
                    let best_score = res.lines[0].score;
                    if best_score.abs() >= ADJUDICATION_THRESHOLD && best_score.abs() < 20000 {
                        break;
                    }

                    res.lines[0].mv
                }
            };

            // Format: fx,fy>tx,tyPROMO
            let move_str = format!(
                "{},{}>{},{}",
                chosen_move.from.x, chosen_move.from.y, chosen_move.to.x, chosen_move.to.y
            );
            let promo_str = chosen_move
                .promotion
                .map(|p| p.to_site_code())
                .unwrap_or_default();
            game_moves.push(format!("{}{}", move_str, promo_str));

            state.make_move(&chosen_move);
            ply += 1;

            if state.get_legal_moves().is_empty() {
                break;
            }
        }

        let mut result = 0; // Draw
        if ply < MAX_MOVES {
            // Re-evaluate end score for result classification
            let res =
                search::get_best_moves_multipv(&mut state, 10, u128::MAX, u128::MAX, 1, true, true);
            if !res.lines.is_empty() {
                let score = res.lines[0].score;
                if score >= 2000 {
                    result = if state.turn == PlayerColor::White {
                        1
                    } else {
                        -1
                    };
                } else if score <= -2000 {
                    result = if state.turn == PlayerColor::White {
                        -1
                    } else {
                        1
                    };
                }
            }
            if state.get_legal_moves().is_empty() && state.is_in_check() {
                result = if state.turn == PlayerColor::White {
                    -1
                } else {
                    1
                };
            }
        }

        if !game_moves.is_empty() {
            let mut buffer = games_buffer.lock().unwrap();
            buffer.push(format!("{:?}|{}|{}", variant, result, game_moves.join("|")));

            if buffer.len() >= BATCH_SIZE {
                let mut w = writer.lock().unwrap();
                for game in buffer.drain(..) {
                    writeln!(w, "{}", game).unwrap();
                }
                w.flush().unwrap();
            }
        }

        pb.inc(1);
    });

    pb.finish_with_message("Done!");

    // Final flush
    let mut buffer = games_buffer.lock().unwrap();
    let mut w = writer.lock().unwrap();
    for game in buffer.drain(..) {
        writeln!(w, "{}", game).unwrap();
    }
    w.flush().unwrap();

    println!("Generation complete.");
}
