#![cfg(not(coverage))]
use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor};
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::moves::Move;

use hydrochess_wasm::search::negamax_node_count_for_depth;
use std::time::Instant;

#[test]
fn run_perft_suite() {
    println!("\n================================================================");
    println!("Running Perft Suite for HydroChess WASM");
    println!("================================================================");
    println!("NOTE: Move counts vary based on slider move heuristics");
    println!("================================================================");

    // Setup Standard Chess
    let mut game = GameState::new();
    game.setup_standard_chess();

    println!("Initial Board Setup: Standard Chess");
    println!("----------------------------------------------------------------");

    let max_depth = 3;
    let mut last_duration = None;

    let mut total_perft_nodes: u128 = 0;
    let mut total_perft_micros: u128 = 0;
    let mut total_search_nodes: u128 = 0;
    let mut total_search_micros: u128 = 0;

    for depth in 1..=max_depth {
        // Perft timing
        let perft_start = Instant::now();
        let perft_nodes = game.perft(depth);
        let perft_duration = perft_start.elapsed();
        let perft_micros = perft_duration.as_micros().max(1);
        let perft_nps = (perft_nodes as u128 * 1_000_000) / perft_micros;

        total_perft_nodes += perft_nodes as u128;
        total_perft_micros += perft_micros;

        // Search timing
        let search_start = Instant::now();
        let searched_nodes = negamax_node_count_for_depth(&mut game, depth);
        let search_duration = search_start.elapsed();
        let search_micros = search_duration.as_micros().max(1);
        let search_nps = (searched_nodes as u128 * 1_000_000) / search_micros;

        total_search_nodes += searched_nodes as u128;
        total_search_micros += search_micros;

        println!(
            "Depth {}: perft {:10} | searched {:10} | \
             Time P: {:?} | NPS P: {:10} | Time S: {:?} | NPS S: {:10}",
            depth,
            perft_nodes,
            searched_nodes,
            perft_duration,
            perft_nps,
            search_duration,
            search_nps
        );

        last_duration = Some(perft_duration);
    }

    println!("================================================================");
    println!("Performance Summary:");
    if total_perft_micros > 0 {
        let avg_perft_nps = (total_perft_nodes * 1_000_000) / total_perft_micros.max(1);
        println!(
            "  Avg perft NPS over depths 1..{}: {}",
            max_depth, avg_perft_nps
        );
    }
    if total_search_micros > 0 {
        let avg_search_nps = (total_search_nodes * 1_000_000) / total_search_micros.max(1);
        println!(
            "  Avg search NPS over depths 1..{}: {}",
            max_depth, avg_search_nps
        );
    }
    if let Some(duration_d_max) = last_duration {
        println!(
            "  Depth {} perft completed in {:?}",
            max_depth, duration_d_max
        );
    }
    println!("================================================================");
}

#[test]
fn compare_evasion_generators_depth_5() {
    // Custom minimal position to exercise evasion logic without huge trees:
    // White: K,Q,R,B,N; Black: K. No pawns, no castling.
    let mut game = GameState::new();

    // White pieces
    game.board
        .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White)); // e1
    game.board
        .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White)); // d1
    game.board
        .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White)); // a1
    game.board
        .set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White)); // c1
    game.board
        .set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White)); // g1

    // Black king
    game.board
        .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black)); // e8

    game.turn = PlayerColor::White;
    game.special_rights.clear();
    game.en_passant = None;
    game.halfmove_clock = 0;
    game.fullmove_number = 1;
    game.hash_stack.clear();
    game.null_moves = 0;
    game.recompute_piece_counts();
    game.recompute_hash();

    let max_depth = 4usize;
    let mut total_new: u64 = 0;
    let mut total_old: u64 = 0;

    fn dfs_compare(game: &mut GameState, depth: usize, total_new: &mut u64, total_old: &mut u64) {
        if depth == 0 {
            return;
        }

        let mut all_moves: Vec<Move> = Vec::new();
        game.get_legal_moves_into(&mut all_moves);

        if game.is_in_check() {
            let mut cand_new: Vec<Move> = Vec::new();
            game.get_evasion_moves_into(&mut cand_new);

            // Helper to filter moves that actually get out of check
            fn filter_legal_evasions(game: &mut GameState, moves: &[Move]) -> Vec<Move> {
                let mut res = Vec::new();
                for m in moves {
                    let undo = game.make_move(m);
                    let illegal = game.is_move_illegal();
                    game.undo_move(m, undo);
                    if !illegal {
                        res.push(*m);
                    }
                }
                res
            }

            let mut new_legal = filter_legal_evasions(game, &cand_new);
            let mut old_legal = filter_legal_evasions(game, &all_moves);

            let key = |m: &Move| {
                (
                    m.from.x,
                    m.from.y,
                    m.to.x,
                    m.to.y,
                    m.promotion.map(|p| p as u8),
                )
            };

            new_legal.sort_by_key(|m| key(m));
            old_legal.sort_by_key(|m| key(m));

            *total_new += new_legal.len() as u64;
            *total_old += old_legal.len() as u64;

            assert_eq!(
                new_legal.len(),
                old_legal.len(),
                "Evasion move count mismatch at some node"
            );
            assert_eq!(
                new_legal.iter().map(|m| key(m)).collect::<Vec<_>>(),
                old_legal.iter().map(|m| key(m)).collect::<Vec<_>>(),
                "Evasion move sets differ at some node",
            );
        }

        for m in all_moves {
            let undo = game.make_move(&m);
            if game.is_move_illegal() {
                game.undo_move(&m, undo);
                continue;
            }
            dfs_compare(game, depth - 1, total_new, total_old);
            game.undo_move(&m, undo);
        }
    }

    dfs_compare(&mut game, max_depth, &mut total_new, &mut total_old);

    println!(
        "Total legal evasions up to depth {}: new = {}, old = {}",
        max_depth, total_new, total_old
    );

    assert_eq!(total_new, total_old);
}

/// Run a single search test and return (total_nodes, total_micros, depth7_nodes, depth7_micros)
fn run_single_search_test(max_depth: usize) -> (u128, u128, u64, u128) {
    let mut total_nodes: u128 = 0;
    let mut total_micros: u128 = 0;
    let mut depth7_nodes: u64 = 0;
    let mut depth7_micros: u128 = 0;

    for depth in 1..=max_depth {
        let mut game = GameState::new();
        game.setup_standard_chess();

        let search_start = Instant::now();
        let searched_nodes = negamax_node_count_for_depth(&mut game, depth);
        let search_micros = search_start.elapsed().as_micros().max(1);

        total_nodes += searched_nodes as u128;
        total_micros += search_micros;

        if depth == max_depth {
            depth7_nodes = searched_nodes;
            depth7_micros = search_micros;
        }
    }

    (total_nodes, total_micros, depth7_nodes, depth7_micros)
}

#[test]
fn run_search_only_suite() {
    println!("\n================================================================");
    println!("Running Search-Only Suite for HydroChess (5 parallel runs, averaged)");
    println!("================================================================");

    const NUM_RUNS: usize = 5;
    const MAX_DEPTH: usize = 10;

    println!(
        "Running {} sequential search tests to depth {}...",
        NUM_RUNS, MAX_DEPTH
    );
    println!("----------------------------------------------------------------");

    // Run tests sequentially to avoid CPU contention
    let mut results: Vec<(u128, u128, u64, u128)> = Vec::with_capacity(NUM_RUNS);
    for _ in 0..NUM_RUNS {
        results.push(run_single_search_test(MAX_DEPTH));
    }

    // Calculate averages
    let mut sum_total_nodes: u128 = 0;
    let mut sum_total_micros: u128 = 0;
    let mut sum_depth7_nodes: u128 = 0;
    let mut sum_depth7_micros: u128 = 0;

    for (i, (total_nodes, total_micros, d7_nodes, d7_micros)) in results.iter().enumerate() {
        let nps = (*total_nodes * 1_000_000) / (*total_micros).max(1);
        let d7_nps = (*d7_nodes as u128 * 1_000_000) / (*d7_micros).max(1);
        println!(
            "Run {}: total_nodes={:10} | total_time={:8}µs | NPS={:8} | D7_nodes={:8} | D7_NPS={:8}",
            i + 1,
            total_nodes,
            total_micros,
            nps,
            d7_nodes,
            d7_nps
        );
        sum_total_nodes += *total_nodes;
        sum_total_micros += *total_micros;
        sum_depth7_nodes += *d7_nodes as u128;
        sum_depth7_micros += *d7_micros;
    }

    let avg_total_nodes = sum_total_nodes / NUM_RUNS as u128;
    let avg_total_micros = sum_total_micros / NUM_RUNS as u128;
    let avg_depth7_nodes = sum_depth7_nodes / NUM_RUNS as u128;
    let avg_depth7_micros = sum_depth7_micros / NUM_RUNS as u128;

    let avg_nps = (avg_total_nodes * 1_000_000) / avg_total_micros.max(1);
    let avg_d7_nps = (avg_depth7_nodes * 1_000_000) / avg_depth7_micros.max(1);

    println!("================================================================");
    println!("AVERAGED RESULTS ({} runs):", NUM_RUNS);
    println!(
        "  Avg total nodes (depths 1-{}): {}",
        MAX_DEPTH, avg_total_nodes
    );
    println!(
        "  Avg total time: {}µs ({:.2}s)",
        avg_total_micros,
        avg_total_micros as f64 / 1_000_000.0
    );
    println!("  Avg NPS: {}", avg_nps);
    println!("  Avg Depth {} nodes: {}", MAX_DEPTH, avg_depth7_nodes);
    println!(
        "  Avg Depth {} time: {}µs ({:.2}s)",
        MAX_DEPTH,
        avg_depth7_micros,
        avg_depth7_micros as f64 / 1_000_000.0
    );
    println!("  Avg Depth {} NPS: {}", MAX_DEPTH, avg_d7_nps);
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

    let mut sum_total_nodes: u128 = 0;
    let mut sum_total_micros: u128 = 0;
    let mut sum_dmax_nodes: u128 = 0;
    let mut sum_dmax_micros: u128 = 0;

    for run in 0..NUM_RUNS {
        let mut g = base_game.clone();
        g.recompute_piece_counts();
        g.recompute_hash();

        let mut total_nodes: u128 = 0;
        let mut total_micros: u128 = 0;
        let mut dmax_nodes: u64 = 0;
        let mut dmax_micros: u128 = 0;

        for depth in 1..=MAX_SEARCH_DEPTH {
            let start = Instant::now();
            let searched = negamax_node_count_for_depth(&mut g, depth);
            let micros = start.elapsed().as_micros().max(1);

            total_nodes += searched as u128;
            total_micros += micros;

            if depth == MAX_SEARCH_DEPTH {
                dmax_nodes = searched;
                dmax_micros = micros;
            }
        }

        let nps = (total_nodes * 1_000_000) / total_micros.max(1);
        let dmax_nps = (dmax_nodes as u128 * 1_000_000) / dmax_micros.max(1);

        println!(
            "  Run {}: total_nodes={:10} | total_time={:8}µs | NPS={:10} | D{}_nodes={:10} | D{}_NPS={:10}",
            run + 1,
            total_nodes,
            total_micros,
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
        "Fairy mix AVG: total_nodes={} | total_time={}µs ({:.2}s) | NPS={}",
        avg_total_nodes,
        avg_total_micros,
        avg_total_micros as f64 / 1_000_000.0,
        avg_nps,
    );
    println!(
        "Fairy mix AVG depth {}: nodes={} | time={}µs ({:.2}s) | NPS={}",
        MAX_SEARCH_DEPTH,
        avg_dmax_nodes,
        avg_dmax_micros,
        avg_dmax_micros as f64 / 1_000_000.0,
        avg_dmax_nps,
    );

    println!("================================================================");
}

/// Test that Rose check detection works correctly.
/// This test reproduces the bug where a rose at (3,7) should check a king at (5,8)
/// but the engine was not detecting it.
#[test]
fn test_rose_check_detection() {
    use hydrochess_wasm::board::Coordinate;
    use hydrochess_wasm::moves::{SpatialIndices, is_square_attacked};

    println!("\n================================================================");
    println!("Testing Rose Check Detection");
    println!("================================================================");

    let mut game = GameState::new();
    game.board = Board::new();
    game.special_rights.clear();
    game.turn = PlayerColor::Black;

    // Set up the position from the bug report:
    // Rose at (3,7) should be able to check king at (5,8)

    // White rose at (3,7)
    game.board
        .set_piece(3, 7, Piece::new(PieceType::Rose, PlayerColor::White));

    // Black king at (5,8)
    game.board
        .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

    // White king somewhere else
    game.board
        .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));

    game.recompute_piece_counts();
    game.recompute_hash();

    // Build spatial indices
    let indices = SpatialIndices::new(&game.board);

    // Test: The king at (5,8) should be attacked by the white rose at (3,7)
    let king_pos = Coordinate::new(5, 8);
    let is_attacked = is_square_attacked(&game.board, &king_pos, PlayerColor::White, &indices);

    println!("Rose at (3,7), King at (5,8)");
    println!("King should be in check from Rose: {}", is_attacked);
    assert!(
        is_attacked,
        "Rose at (3,7) should give check to king at (5,8)"
    );

    // Also test that the game correctly identifies it's in check
    assert!(game.is_in_check(), "Game should report being in check");

    // And test that non-king moves are illegal (must respond to check)
    let mut all_moves = Vec::new();
    game.get_legal_moves_into(&mut all_moves);
    println!("Legal moves available: {}", all_moves.len());

    // All legal moves should either move the king or block/capture the rose
    for m in &all_moves {
        if m.piece.piece_type() != PieceType::King {
            // If it's not a king move, it must capture the rose
            assert_eq!(
                (m.to.x, m.to.y),
                (3, 7),
                "Non-king move {:?} should capture the rose at (3,7)",
                m
            );
        }
    }

    println!("Rose check detection test PASSED!");
    println!("================================================================");
}

/// Test that sliders can block a distant diagonal check.
/// This reproduces a bug where a rook 15 squares away couldn't block a bishop check.
#[test]
fn test_distant_slider_block() {
    use hydrochess_wasm::moves::MoveList;

    println!("\n================================================================");
    println!("Testing Distant Slider Block");
    println!("================================================================");

    let mut game = GameState::new();
    game.board = Board::new();
    game.special_rights.clear();
    game.turn = PlayerColor::White;

    // White King at (-4, -2)
    game.board
        .set_piece(-4, -2, Piece::new(PieceType::King, PlayerColor::White));

    // White Rook at (7, 2) - can block at (7, -13) which is 15 squares down
    game.board
        .set_piece(7, 2, Piece::new(PieceType::Rook, PlayerColor::White));

    // Black Bishop at (26, -32) - giving diagonal check
    game.board
        .set_piece(26, -32, Piece::new(PieceType::Bishop, PlayerColor::Black));

    // Black King somewhere else
    game.board
        .set_piece(100, 100, Piece::new(PieceType::King, PlayerColor::Black));

    game.recompute_piece_counts();
    game.recompute_hash();

    // Verify the king is in check
    assert!(
        game.is_in_check(),
        "White king should be in check from distant bishop"
    );

    // Get evasion moves
    let mut evasion_moves: MoveList = MoveList::new();
    game.get_evasion_moves_into(&mut evasion_moves);

    println!("Evasion moves generated: {}", evasion_moves.len());
    for m in &evasion_moves {
        println!("  {:?} -> {:?}", m.from, m.to);
    }

    // Check that the rook blocking move is present
    // Rook at (7, 2) should be able to move to (7, -13) to block
    // Check ray: from (-4,-2) towards (26,-32), step (1,-1)
    // At k=11: (-4+11, -2-11) = (7, -13) - this is where rook blocks!
    let rook_block = evasion_moves
        .iter()
        .any(|m| m.from.x == 7 && m.from.y == 2 && m.to.x == 7 && m.to.y == -13);

    assert!(
        rook_block,
        "Rook at (7,2) should be able to block at (7,-13)"
    );

    println!("Distant slider block test PASSED!");
    println!("================================================================");
}
