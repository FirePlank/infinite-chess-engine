use hydrochess_wasm::game::GameState;
use std::env;
use std::io::{self, Read};

fn main() -> io::Result<()> {
    #[cfg(debug_assertions)]
    {
        println!(
            "⚠️  WARNING: Running in DEBUG mode. Performance measurements will be inaccurate."
        );
        println!("   For accurate benchmarking, use: cargo run --bin eval_nnue --release -- <ICN>");
        println!();
    }

    let args: Vec<String> = env::args().collect();

    // Read ICN from arg or stdin
    let icn = if args.len() > 1 {
        args[1].trim().to_string()
    } else {
        let mut buffer = String::new();
        if io::stdin().read_to_string(&mut buffer).is_err() {
            println!("Usage: eval_nnue \"<ICN>\" or pipe ICN via stdin");
            return Ok(());
        }
        buffer.trim().to_string()
    };

    if icn.is_empty() {
        println!("Error: Empty ICN string.");
        return Ok(());
    }

    println!("Parsing ICN: {}", icn);
    let mut game = GameState::new();

    game.setup_position_from_icn(&icn);
    game.recompute_piece_counts();
    game.recompute_hash();

    println!("\n--- NNUE Diagnostics ---");

    // Check feature flag
    #[cfg(feature = "nnue")]
    println!("Feature 'nnue': ENABLED");
    #[cfg(not(feature = "nnue"))]
    println!("Feature 'nnue': DISABLED");

    // Check applicability
    #[cfg(feature = "nnue")]
    {
        let applicable = hydrochess_wasm::nnue::is_applicable(&game);
        println!("NNUE Applicable: {}", applicable);

        if !applicable {
            println!("  Reasons (potential):");
            let mut reason_found = false;

            // Check kings
            if game.white_king_pos.is_none() {
                println!("  - Missing White King");
                reason_found = true;
            }
            if game.black_king_pos.is_none() {
                println!("  - Missing Black King");
                reason_found = true;
            }

            let standard_pieces = [
                hydrochess_wasm::board::PieceType::King,
                hydrochess_wasm::board::PieceType::Queen,
                hydrochess_wasm::board::PieceType::Rook,
                hydrochess_wasm::board::PieceType::Bishop,
                hydrochess_wasm::board::PieceType::Knight,
                hydrochess_wasm::board::PieceType::Pawn,
            ];

            for (_x, _y, p) in game.board.iter() {
                if !standard_pieces.contains(&p.piece_type()) {
                    println!("  - Non-standard piece detected: {:?}", p.piece_type());
                    reason_found = true;
                }
            }

            if !reason_found {
                println!("  - Unknown (check nnue::is_applicable source logic)");
            }
        } else {
            // Compute NNUE score
            let start = std::time::Instant::now();
            let nnue_score = hydrochess_wasm::nnue::evaluate(&game);
            let duration = start.elapsed();
            println!("NNUE Raw Score: {} cp", nnue_score);
            println!("NNUE Compute Time: {:?}", duration);

            // Compute Hybrid (Base) Score
            let hybrid_score = hydrochess_wasm::evaluation::evaluate(&game, None);
            println!("Hybrid/Total Score (evaluate()): {} cp", hybrid_score);

            if nnue_score == 0 && hybrid_score != 0 {
                println!(
                    "WARNING: NNUE score is exactly 0. Check weight loading or quantization scales."
                );
            }
        }
    }

    println!("\n--- HCE Diagnostics ---");
    // HCE fallback
    let hce_score = hydrochess_wasm::evaluation::base::evaluate_inner(&game);
    println!("HCE Score (evaluate_inner): {} cp", hce_score);

    Ok(())
}
