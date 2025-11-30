use hydrochess_wasm::game::GameState;
use hydrochess_wasm::moves::Move;
use hydrochess_wasm::board::{Piece, PieceType, PlayerColor};

use std::time::Instant;
use hydrochess_wasm::search::negamax_node_count_for_depth;

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
        println!("  Avg perft NPS over depths 1..{}: {}", max_depth, avg_perft_nps);
    }
    if total_search_micros > 0 {
        let avg_search_nps = (total_search_nodes * 1_000_000) / total_search_micros.max(1);
        println!("  Avg search NPS over depths 1..{}: {}", max_depth, avg_search_nps);
    }
    if let Some(duration_d_max) = last_duration {
        println!("  Depth {} perft completed in {:?}", max_depth, duration_d_max);
    }
    println!("================================================================");
}

#[test]
fn compare_evasion_generators_depth_5() {
    // Custom minimal position to exercise evasion logic without huge trees:
    // White: K,Q,R,B,N; Black: K. No pawns, no castling.
    let mut game = GameState::new();

    // White pieces
    game.board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));   // e1
    game.board.set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));  // d1
    game.board.set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));   // a1
    game.board.set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White)); // c1
    game.board.set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White)); // g1

    // Black king
    game.board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));   // e8

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

            let key = |m: &Move| (m.from.x, m.from.y, m.to.x, m.to.y, m.promotion.map(|p| p as u8));

            new_legal.sort_by_key(|m| key(m));
            old_legal.sort_by_key(|m| key(m));

            *total_new += new_legal.len() as u64;
            *total_old += old_legal.len() as u64;

            assert_eq!(new_legal.len(), old_legal.len(), "Evasion move count mismatch at some node");
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

#[test]
fn run_search_only_suite() {
    println!("\n================================================================");
    println!("Running Search-Only Suite for HydroChess WASM");
    println!("================================================================");

    // Setup Standard Chess
    let mut game = GameState::new();
    game.setup_standard_chess();

    println!("Initial Board Setup: Standard Chess");
    println!("----------------------------------------------------------------");

    let max_depth = 7;
    let mut last_duration = None;

    let mut total_search_nodes: u128 = 0;
    let mut total_search_micros: u128 = 0;

    for depth in 1..=max_depth {
        let search_start = Instant::now();
        let searched_nodes = negamax_node_count_for_depth(&mut game, depth);
        let search_duration = search_start.elapsed();
        let search_micros = search_duration.as_micros().max(1);
        let search_nps = (searched_nodes as u128 * 1_000_000) / search_micros;

        total_search_nodes += searched_nodes as u128;
        total_search_micros += search_micros;

        println!(
            "Depth {}: searched {:10} | Time: {:?} | NPS: {:10}",
            depth,
            searched_nodes,
            search_duration,
            search_nps
        );

        last_duration = Some(search_duration);
    }

    println!("================================================================");
    println!("Search-Only Performance Summary:");
    if total_search_micros > 0 {
        let avg_search_nps = (total_search_nodes * 1_000_000) / total_search_micros.max(1);
        println!("  Avg search NPS over depths 1..{}: {}", max_depth, avg_search_nps);
    }
    if let Some(duration_d_max) = last_duration {
        println!("  Depth {} search completed in {:?}", max_depth, duration_d_max);
    }
    println!("================================================================");
}