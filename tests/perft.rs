#![cfg(not(coverage))]
use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor};
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::search::negamax_node_count_for_depth;
use std::time::Instant;

fn format_duration(duration: std::time::Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 1_000 {
        format!("{:.2} ns", nanos as f64)
    } else if nanos < 1_000_000 {
        format!("{:.2} µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
    }
}

fn run_single_search_test(
    max_depth: usize,
) -> (u128, std::time::Duration, u64, std::time::Duration) {
    let mut total_nodes: u128 = 0;
    let mut total_duration = std::time::Duration::new(0, 0);
    let mut depth7_nodes: u64 = 0;
    let mut depth7_duration = std::time::Duration::new(0, 0);

    for depth in 1..=max_depth {
        let mut game = GameState::new();
        game.setup_standard_chess();

        let search_start = Instant::now();
        let searched_nodes = std::hint::black_box(negamax_node_count_for_depth(
            std::hint::black_box(&mut game),
            depth,
        ));
        let search_duration = search_start.elapsed();

        total_nodes += searched_nodes as u128;
        total_duration += search_duration;

        if depth == max_depth {
            depth7_nodes = searched_nodes;
            depth7_duration = search_duration;
        }
    }

    (total_nodes, total_duration, depth7_nodes, depth7_duration)
}

#[test]
fn run_search_only_suite() {
    println!("\n================================================================");
    println!("Running Search-Only Suite for HydroChess (5 runs, averaged)");
    println!("================================================================");

    const NUM_RUNS: usize = 5;
    const MAX_DEPTH: usize = 11;

    println!("Warming up...");
    for _ in 0..20 {
        let mut g = GameState::new();
        g.setup_standard_chess();
        std::hint::black_box(negamax_node_count_for_depth(
            std::hint::black_box(&mut g),
            5,
        ));
    }

    println!(
        "Running {} sequential search tests to depth {}...",
        NUM_RUNS, MAX_DEPTH
    );
    println!("----------------------------------------------------------------");

    // Run tests sequentially to avoid CPU contention
    let mut results: Vec<(u128, std::time::Duration, u64, std::time::Duration)> =
        Vec::with_capacity(NUM_RUNS);
    for _ in 0..NUM_RUNS {
        results.push(run_single_search_test(MAX_DEPTH));
    }

    // Calculate averages
    let mut sum_total_nodes: u128 = 0;
    let mut sum_total_micros: u128 = 0;
    let mut sum_dmax_nodes: u128 = 0;
    let mut sum_dmax_micros: u128 = 0;

    for (i, (total_nodes, total_duration, dmax_nodes, dmax_duration)) in results.iter().enumerate()
    {
        let total_micros = total_duration.as_micros();
        let dmax_micros = dmax_duration.as_micros();

        let nps = (*total_nodes * 1_000_000) / total_micros.max(1);
        let dmax_nps = (*dmax_nodes as u128 * 1_000_000) / dmax_micros.max(1);
        println!(
            "Run {}: total_nodes={:10} | total_time={:>10} | NPS={:10} | D{}_nodes={:10} | D{}_NPS={:10}",
            i + 1,
            total_nodes,
            format_duration(*total_duration),
            nps,
            MAX_DEPTH,
            dmax_nodes,
            MAX_DEPTH,
            dmax_nps
        );
        sum_total_nodes += *total_nodes;
        sum_total_micros += total_micros;
        sum_dmax_nodes += *dmax_nodes as u128;
        sum_dmax_micros += dmax_micros;
    }

    let avg_total_nodes = sum_total_nodes / NUM_RUNS as u128;
    let avg_total_micros = sum_total_micros / NUM_RUNS as u128;
    let avg_dmax_nodes = sum_dmax_nodes / NUM_RUNS as u128;
    let avg_dmax_micros = sum_dmax_micros / NUM_RUNS as u128;

    let avg_nps = (avg_total_nodes * 1_000_000) / avg_total_micros.max(1);
    let avg_dmax_nps = (avg_dmax_nodes * 1_000_000) / avg_dmax_micros.max(1);

    println!("================================================================");
    println!("AVERAGED RESULTS ({} runs):", NUM_RUNS);
    println!(
        "  Avg total nodes (depths 1-{}): {}",
        MAX_DEPTH, avg_total_nodes
    );
    println!(
        "  Avg total time: {}",
        format_duration(std::time::Duration::from_micros(avg_total_micros as u64))
    );
    println!("  Avg NPS: {}", avg_nps);
    println!("  Avg Depth {} nodes: {}", MAX_DEPTH, avg_dmax_nodes);
    println!(
        "  Avg Depth {} time: {}",
        MAX_DEPTH,
        format_duration(std::time::Duration::from_micros(avg_dmax_micros as u64))
    );
    println!("  Avg Depth {} NPS: {}", MAX_DEPTH, avg_dmax_nps);
    println!("================================================================");
}

#[test]
fn perft_fairy_piece_mix() {
    println!("\n================================================================");
    println!("Perft: Fairy Piece Mix");
    println!("================================================================");

    let mut game = GameState::new();

    // Clear any default setup and place a mix of fairy and classical pieces.
    game.board = Board::new();
    game.special_rights.clear();
    game.en_passant = None;
    game.turn = PlayerColor::White;
    game.halfmove_clock = 0;
    game.fullmove_number = 1;
    game.hash_stack.clear();
    game.null_moves = 0;

    // White king to keep position legal
    game.board
        .set_piece(4, 1, Piece::new(PieceType::King, PlayerColor::White));
    // Black king far away
    game.board
        .set_piece(4, 100, Piece::new(PieceType::King, PlayerColor::Black));

    // Place a row of various non-pawn pieces to stress move generation.
    // Classical
    game.board
        .set_piece(1, 2, Piece::new(PieceType::Queen, PlayerColor::White));
    game.board
        .set_piece(2, 2, Piece::new(PieceType::Rook, PlayerColor::White));
    game.board
        .set_piece(3, 2, Piece::new(PieceType::Bishop, PlayerColor::White));
    game.board
        .set_piece(5, 2, Piece::new(PieceType::Knight, PlayerColor::White));

    // Fairy sliders / leapers
    game.board
        .set_piece(6, 2, Piece::new(PieceType::Amazon, PlayerColor::White));
    game.board
        .set_piece(7, 2, Piece::new(PieceType::Chancellor, PlayerColor::White));
    game.board
        .set_piece(8, 2, Piece::new(PieceType::Archbishop, PlayerColor::White));

    game.board
        .set_piece(1, 3, Piece::new(PieceType::Centaur, PlayerColor::White));
    game.board.set_piece(
        2,
        3,
        Piece::new(PieceType::RoyalCentaur, PlayerColor::White),
    );
    game.board
        .set_piece(3, 3, Piece::new(PieceType::Hawk, PlayerColor::White));
    game.board
        .set_piece(4, 3, Piece::new(PieceType::Guard, PlayerColor::White));
    game.board
        .set_piece(5, 3, Piece::new(PieceType::Camel, PlayerColor::White));
    game.board
        .set_piece(6, 3, Piece::new(PieceType::Giraffe, PlayerColor::White));
    game.board
        .set_piece(7, 3, Piece::new(PieceType::Zebra, PlayerColor::White));
    game.board
        .set_piece(8, 3, Piece::new(PieceType::Knightrider, PlayerColor::White));

    game.board
        .set_piece(2, 4, Piece::new(PieceType::Rose, PlayerColor::White));
    game.board
        .set_piece(3, 4, Piece::new(PieceType::Huygen, PlayerColor::White));
    game.board
        .set_piece(5, 4, Piece::new(PieceType::RoyalQueen, PlayerColor::White));

    // place obstacles across the 3rd rank
    for x in -50..=50 {
        game.board
            .set_piece(x, 5, Piece::new(PieceType::Obstacle, PlayerColor::Neutral));
    }
    // Keep a clean copy of the fairy position so we can reuse it for search tests.
    let base_game = game.clone();

    // run a search-only speed test on the same fairy position
    const NUM_RUNS: usize = 3;
    const MAX_SEARCH_DEPTH: usize = 8;

    println!(
        "\nFairy mix search-only benchmark ({} runs, depth 1..{}):",
        NUM_RUNS, MAX_SEARCH_DEPTH
    );

    println!("Warming up...");
    for _ in 0..2 {
        let mut g = base_game.clone();
        g.recompute_piece_counts();
        g.recompute_hash();
        std::hint::black_box(negamax_node_count_for_depth(
            std::hint::black_box(&mut g),
            5,
        ));
    }

    let mut sum_total_nodes: u128 = 0;
    let mut sum_total_micros: u128 = 0;
    let mut sum_dmax_nodes: u128 = 0;
    let mut sum_dmax_micros: u128 = 0;

    for run in 0..NUM_RUNS {
        let mut g = base_game.clone();
        g.recompute_piece_counts();
        g.recompute_hash();

        let mut total_nodes: u128 = 0;
        let mut total_duration = std::time::Duration::new(0, 0);
        let mut dmax_nodes: u64 = 0;
        let mut dmax_duration = std::time::Duration::new(0, 0);

        for depth in 1..=MAX_SEARCH_DEPTH {
            let start = Instant::now();
            let searched = std::hint::black_box(negamax_node_count_for_depth(
                std::hint::black_box(&mut g),
                depth,
            ));
            let duration = start.elapsed();

            total_nodes += searched as u128;
            total_duration += duration;

            if depth == MAX_SEARCH_DEPTH {
                dmax_nodes = searched;
                dmax_duration = duration;
            }
        }

        let total_micros = total_duration.as_micros();
        let dmax_micros = dmax_duration.as_micros();

        let nps = (total_nodes * 1_000_000) / total_micros.max(1);
        let dmax_nps = (dmax_nodes as u128 * 1_000_000) / dmax_micros.max(1);

        println!(
            "  Run {}: total_nodes={:10} | total_time={:>10} | NPS={:10} | D{}_nodes={:10} | D{}_NPS={:10}",
            run + 1,
            total_nodes,
            format_duration(total_duration),
            nps,
            MAX_SEARCH_DEPTH,
            dmax_nodes,
            MAX_SEARCH_DEPTH,
            dmax_nps,
        );

        sum_total_nodes += total_nodes;
        sum_total_micros += total_micros;
        sum_dmax_nodes += dmax_nodes as u128;
        sum_dmax_micros += dmax_micros;
    }

    let avg_total_nodes = sum_total_nodes / NUM_RUNS as u128;
    let avg_total_micros = sum_total_micros / NUM_RUNS as u128;
    let avg_dmax_nodes = sum_dmax_nodes / NUM_RUNS as u128;
    let avg_dmax_micros = sum_dmax_micros / NUM_RUNS as u128;

    let avg_nps = (avg_total_nodes * 1_000_000) / avg_total_micros.max(1);
    let avg_dmax_nps = (avg_dmax_nodes * 1_000_000) / avg_dmax_micros.max(1);

    println!("----------------------------------------------------------------");
    println!(
        "Fairy mix AVG: total_nodes={} | total_time={} | NPS={}",
        avg_total_nodes,
        format_duration(std::time::Duration::from_micros(avg_total_micros as u64)),
        avg_nps,
    );
    println!(
        "Fairy mix AVG depth {}: nodes={} | time={} | NPS={}",
        MAX_SEARCH_DEPTH,
        avg_dmax_nodes,
        format_duration(std::time::Duration::from_micros(avg_dmax_micros as u64)),
        avg_dmax_nps,
    );

    println!("================================================================");
}
