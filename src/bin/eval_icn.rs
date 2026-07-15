use apeiron::evaluation::base::debug_evaluate;
use apeiron::game::GameState;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --bin eval_icn \"<ICN_STRING>\"");
        return;
    }

    let icn = &args[1];
    let mut game = GameState::new();

    // Use the ICN parser in GameState
    game.setup_position_from_icn(icn);

    println!("Evaluating position: {}", icn);
    println!("Turn: {:?}, Fullmove: {}", game.turn, game.fullmove_number);
    println!(
        "Pieces: White {}, Black {}",
        game.white_piece_count, game.black_piece_count
    );

    let trace = debug_evaluate(&game);
    trace.print();
}
