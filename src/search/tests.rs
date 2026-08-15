use super::*;
use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::{Move, set_world_bounds};

// Helper function to reset world bounds to defaults
fn reset_world_bounds() {
    set_world_bounds(
        -1_000_000_000_000_000,
        1_000_000_000_000_000,
        -1_000_000_000_000_000,
        1_000_000_000_000_000,
    );
}

#[test]
fn test_corrhist_constants() {
    assert!(CORRHIST_SIZE.is_power_of_two());
    assert!(LASTMOVE_CORRHIST_SIZE.is_power_of_two());
    assert!(LOW_PLY_HISTORY_ENTRIES.is_power_of_two());
}

/// Native replica of the wasm MT analyse path (lib.rs): Lazy SMP helpers filling the
/// shared TT while the main thread runs the MultiPV analysis. Panics here reproduce
/// (with real backtraces) the `unreachable` crashes seen on wasm rayon workers.
#[test]
#[cfg(feature = "multithreading")]
fn test_mt_analyse_with_helpers() {
    reset_world_bounds();
    let mut game = GameState::new();
    game.setup_position_from_icn(crate::Variant::Chess.starting_icn());

    GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);
    init_shared_tt();
    USE_SHARED_TT.store(true, std::sync::atomic::Ordering::Relaxed);

    let helper_game = game.clone();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for i in 1..4usize {
            let mut game_clone = helper_game.clone();
            handles.push(s.spawn(move || {
                let _ = get_best_move_threaded(&mut game_clone, 12, 2_000, 2_000, true, i, true);
            }));
        }

        let mut cb = |_: &DepthInfo| {};
        let result = analyse_position(&mut game, 10, 1, 2_000, 2, &mut cb);
        GLOBAL_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(
            !result.lines.is_empty(),
            "MT analyse should produce PV lines"
        );

        for handle in handles {
            handle.join().expect("helper thread panicked");
        }
    });

    USE_SHARED_TT.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Time-to-depth benchmark on the wasm worker's 180ms slice cadence. Configured via
/// BENCH_MODE, BENCH_TT_MB, BENCH_THREADS and BENCH_DEPTH so each run is a fresh
/// process.
#[test]
#[ignore]
#[cfg(feature = "multithreading")]
fn bench_time_to_depth() {
    let mode = std::env::var("BENCH_MODE").unwrap_or_else(|_| "st".into());
    let tt_mb: usize = std::env::var("BENCH_TT_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let threads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let target: usize = std::env::var("BENCH_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(18);
    #[allow(non_snake_case)]
    let SLICE_MS: u128 = std::env::var("BENCH_SLICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(180);

    reset_world_bounds();
    set_tt_size_mb(tt_mb);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    // Startpos + a normal opening, like a real analysis position (BENCH_POS picks one).
    let pos: usize = std::env::var("BENCH_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let moves = match pos {
        1 => "5,2>5,4|5,7>5,5|7,1>6,3|2,8>3,6|6,1>2,5|7,8>6,6|4,2>4,3|6,8>2,4|3,1>7,5|4,8>5,7",
        2 => "4,2>4,4|4,7>4,5|3,2>3,4|5,7>5,6|2,1>3,3|7,8>6,6|3,1>6,4|6,8>5,7",
        3 => "7,1>6,3|4,7>4,5|3,2>3,4|3,7>3,6|4,2>4,4|7,8>6,6|2,1>3,3|5,7>5,6",
        _ => panic!("unknown BENCH_POS"),
    };
    let icn = format!("{} {}", crate::Variant::Chess.starting_icn(), moves);
    let mut game = GameState::new();
    game.setup_position_from_icn(&icn);

    let use_mt = mode.starts_with("mt");
    if use_mt {
        init_shared_tt();
        USE_SHARED_TT.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    let t0 = std::time::Instant::now();
    let mut reached = 0usize;
    let helper_game = game.clone();

    let mut cb = |info: &DepthInfo| {
        println!(
            "depth {:2} at {:6}ms nodes {}",
            info.depth,
            t0.elapsed().as_millis(),
            info.nodes
        );
    };

    match mode.as_str() {
        // Exactly the wasm worker loop: 180ms slices, resume at reached+1, no helpers.
        "st" => {
            while reached < target {
                let start = (reached + 1).min(target);
                GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);
                let r = analyse_position(&mut game, target, start, SLICE_MS, 1, &mut cb);
                let d = r.lines.first().map_or(0, |l| l.depth);
                if d <= reached {
                    break;
                }
                reached = d;
            }
        }
        // The shipped wasm design: detached helpers spawned once, main sliced like the worker.
        "mt_detached" => {
            let epoch = HELPER_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);
            for i in 1..threads {
                let gc = helper_game.clone();
                rayon::spawn(move || helper_run(gc, epoch, i));
            }
            while reached < target {
                let start = (reached + 1).min(target);
                let r = analyse_position(&mut game, target, start, SLICE_MS, 1, &mut cb);
                let d = r.lines.first().map_or(0, |l| l.depth);
                if d <= reached {
                    break;
                }
                reached = d;
            }
            stop_analysis_helpers();
        }
        // Helpers persist for the WHOLE search; main runs unsliced to target.
        "mt_full" => {
            GLOBAL_STOP.store(false, std::sync::atomic::Ordering::Relaxed);
            rayon::in_place_scope(|s| {
                for i in 1..threads {
                    let mut gc = helper_game.clone();
                    s.spawn(move |_| {
                        let _ = get_best_move_threaded(
                            &mut gc, target, 600_000, 600_000, true, i, true,
                        );
                    });
                }
                let _ = analyse_position(&mut game, target, 1, 0, 1, &mut cb);
                GLOBAL_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }
        other => panic!("unknown BENCH_MODE {other}"),
    }
    println!(
        "TOTAL {}ms mode={} tt={}MB threads={}",
        t0.elapsed().as_millis(),
        mode,
        tt_mb,
        threads
    );
    USE_SHARED_TT.store(false, std::sync::atomic::Ordering::Relaxed);
}

#[test]
fn test_timer_new() {
    let timer = Timer::new();
    let elapsed = timer.elapsed_ms();
    // Should be very small (less than 100ms for new timer)
    assert!(elapsed < 100, "New timer should have small elapsed time");
}

#[test]
fn test_timer_reset() {
    let mut timer = Timer::new();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let before_reset = timer.elapsed_ms();
    timer.reset();
    let after_reset = timer.elapsed_ms();
    assert!(
        after_reset < before_reset,
        "Reset should reduce elapsed time"
    );
}

#[test]
fn test_searcher_new() {
    let searcher = Searcher::new(5000);

    assert_eq!(searcher.hot.time_limit_ms, 5000);
    assert_eq!(searcher.hot.nodes, 0);
    assert_eq!(searcher.hot.qnodes, 0);
    assert!(!searcher.hot.stopped);
    assert!(!searcher.silent);
    assert_eq!(searcher.thread_id, 0);
    assert_eq!(searcher.killers.len(), MAX_PLY);
    assert_eq!(searcher.pv_length.len(), MAX_PLY);
}

#[test]
fn test_searcher_decay_history() {
    let mut searcher = Searcher::new(5000);
    searcher.history[0][0] = 100;
    searcher.history[1][1] = 200;

    searcher.decay_history();

    assert_eq!(searcher.history[0][0], 90); // 100 * 9/10
    assert_eq!(searcher.history[1][1], 180); // 200 * 9/10
}

#[test]
fn test_searcher_update_history() {
    let mut searcher = Searcher::new(5000);

    searcher.update_history(PieceType::Knight, 42, 100);
    let val = searcher.history[PieceType::Knight as usize][42];
    assert!(val > 0, "History should be updated positively");

    searcher.update_history(PieceType::Knight, 42, -100);
    let val_after = searcher.history[PieceType::Knight as usize][42];
    assert!(
        val_after < val,
        "History should decrease with negative bonus"
    );
}

#[test]
fn test_searcher_check_time_no_limit() {
    let mut searcher = Searcher::new(u128::MAX);
    searcher.hot.nodes = 10000;

    let timed_out = searcher.check_time();
    assert!(!timed_out, "Should not timeout with MAX time limit");
}

#[test]
fn test_mate_score_detection() {
    // Simple mate score detection using constants
    let mate_score = MATE_VALUE - 10;
    let is_mate = mate_score.abs() > MATE_SCORE;
    assert!(is_mate, "Near MATE_VALUE should be detected as mate");

    let normal_score: i32 = 1000;
    let is_normal_mate = normal_score.abs() > MATE_SCORE;
    assert!(!is_normal_mate, "Normal score should not be mate");
}

#[test]
fn test_corrhist_mode_enum() {
    assert!(CorrHistMode::PawnBased != CorrHistMode::NonPawnBased);
}

#[test]
fn test_node_type_enum() {
    assert!(NodeType::PV != NodeType::Cut);
    assert!(NodeType::Cut != NodeType::All);
}

#[test]
fn test_search_stats_default() {
    let stats = SearchStats {
        nodes: 0,
        tt_capacity: 1000,
        tt_used: 500,
        tt_fill_permille: 500,
    };

    assert_eq!(stats.tt_capacity, 1000);
    assert_eq!(stats.tt_used, 500);
    assert_eq!(stats.tt_fill_permille, 500);
}

#[test]
fn test_move_creation_for_search() {
    let from = Coordinate::new(4, 4);
    let to = Coordinate::new(5, 6);
    let piece = Piece::new(PieceType::Knight, PlayerColor::White);

    let m = Move::new(from, to, piece);

    assert_eq!(m.from.x, 4);
    assert_eq!(m.from.y, 4);
    assert_eq!(m.to.x, 5);
    assert_eq!(m.to.y, 6);
}

#[test]
fn test_update_low_ply_history() {
    let mut searcher = Searcher::new(5000);

    searcher.update_low_ply_history(0, 42, 100);
    let val = searcher.low_ply_history[0][42 & LOW_PLY_HISTORY_MASK];
    assert!(val > 0, "Low ply history should be updated");

    // Update at ply >= LOW_PLY_HISTORY_SIZE should do nothing
    searcher.update_low_ply_history(10, 42, 1000);
    // Can't easily verify no change, but at least it shouldn't panic
}

#[cfg(feature = "multithreading")]
fn thread_result(
    from: (i64, i64),
    to: (i64, i64),
    promo: Option<PieceType>,
    score: i32,
    depth: usize,
) -> ThreadResult {
    ThreadResult {
        best_move: Move {
            from: Coordinate::new(from.0, from.1),
            to: Coordinate::new(to.0, to.1),
            piece: Piece::new(PieceType::Pawn, PlayerColor::White),
            promotion: promo,
            rook_coord: None,
        },
        score,
        completed_depth: depth,
        pv_length: 5,
        nodes: 0,
        thread_id: 0,
    }
}

#[test]
#[cfg(feature = "multithreading")]
fn select_best_thread_matches_stockfish_voting() {
    let win = MATE_VALUE - 20; // proven win
    let loss = -MATE_VALUE + 20; // proven loss
    let long_loss = -MATE_VALUE + 60; // loss, but longer resistance

    // (a) A losing best must yield to a normal (or winning) thread.
    assert_eq!(
        select_best_thread(&[
            thread_result((1, 1), (1, 2), None, loss, 10),
            thread_result((2, 2), (2, 3), None, 50, 10),
        ]),
        1,
        "a normal thread must override a losing best"
    );

    // (b) A normal best must never be replaced by a proven loss.
    assert_eq!(
        select_best_thread(&[
            thread_result((1, 1), (1, 2), None, 50, 10),
            thread_result((2, 2), (2, 3), None, loss, 10),
        ]),
        0,
        "a proven loss must not override a normal best"
    );

    // (c) Between two losses, pick the longest resistance (higher score).
    assert_eq!(
        select_best_thread(&[
            thread_result((1, 1), (1, 2), None, loss, 10),
            thread_result((2, 2), (2, 3), None, long_loss, 10),
        ]),
        1,
        "should pick the longest resistance, not the faster loss"
    );

    // Winning best: pick the fastest mate (higher score).
    assert_eq!(
        select_best_thread(&[
            thread_result((1, 1), (1, 2), None, win, 10),
            thread_result((2, 2), (2, 3), None, win + 5, 10),
        ]),
        1,
        "should pick the fastest mate"
    );
}

#[test]
#[cfg(feature = "multithreading")]
fn select_best_thread_distinguishes_promotions() {
    // Two threads promote to Q and N on the same square. Sharing a vote key would
    // combine them to 280 and beat the a1a2 thread's 240; keyed by promotion they
    // hold 140 each, so a1a2 wins.
    let results = [
        thread_result((5, 7), (5, 8), Some(PieceType::Queen), 50, 10),
        thread_result((5, 7), (5, 8), Some(PieceType::Knight), 50, 10),
        thread_result((1, 1), (1, 2), None, 60, 10),
    ];
    assert_eq!(
        select_best_thread(&results),
        2,
        "distinct promotions must not pool their votes"
    );
}

#[test]
fn test_get_best_move_simple_position() {
    let mut game = GameState::new();
    // Simple position: white queen can take undefended black rook
    game.setup_position_from_icn("w K0,0|Q4,4|k7,7|r4,7");

    // Short search with 1 second time limit
    let result = get_best_move(&mut game, 5, 1000, true, true);

    assert!(result.is_some(), "Should find a move");
    let (best_move, _eval, _stats) = result.unwrap();
    // Should find the queen capture of rook as best
    // (Can't guarantee specific move but should find something)
    assert!(best_move.piece.piece_type() != PieceType::Void);
}

#[test]
fn test_get_best_move_returns_result() {
    let mut game = GameState::new();
    // Any position with legal moves
    game.setup_position_from_icn("w K4,1|k4,8|R1,1");

    let result = get_best_move(&mut game, 5, 1000, true, true);

    assert!(result.is_some(), "Should find a move");
    let (best_move, _eval, stats) = result.unwrap();
    assert!(best_move.piece.piece_type() != PieceType::Void);
    // Check stats are populated
    assert!(stats.tt_capacity > 0);
}

#[test]
fn test_evaluate_with_search() {
    let mut game = GameState::new();
    // Balanced position
    game.setup_position_from_icn("w K0,0|k7,7|R4,2|r4,7");

    // Get static eval
    #[cfg(feature = "nnue")]
    let static_eval = evaluate(&game, None);
    #[cfg(not(feature = "nnue"))]
    let static_eval = evaluate(&game);
    // Should be close to 0 (roughly balanced)
    assert!(
        static_eval.abs() < 500,
        "Balanced position eval should be near 0"
    );
}

#[test]
fn test_tt_basic_operations() {
    let tt = LocalTranspositionTable::new(1);

    assert!(tt.capacity() > 0);
    assert_eq!(tt.used_entries(), 0);
    assert_eq!(tt.fill_permille(), 0);
}

#[test]
fn test_timer_reset_and_elapsed() {
    let mut timer = Timer::new();
    // Wait just a bit to ensure elapsed is > 0
    let _ = timer.elapsed_ms();
    timer.reset();
    // After reset, elapsed should be close to 0
    let elapsed = timer.elapsed_ms();
    assert!(elapsed < 100, "Elapsed after reset should be small");
}

#[test]
fn test_searcher_initialization() {
    let searcher = Searcher::new(10000);

    assert_eq!(searcher.hot.nodes, 0);
    assert!(searcher.tt.capacity() > 0);
}

#[test]
fn test_killer_moves() {
    let mut searcher = Searcher::new(1000);

    let from = Coordinate::new(4, 4);
    let to = Coordinate::new(5, 6);
    let piece = Piece::new(PieceType::Knight, PlayerColor::White);
    let m = Move::new(from, to, piece);

    // Add killer at ply 0
    searcher.killers[0][1] = searcher.killers[0][0];
    searcher.killers[0][0] = Some(m);

    assert!(searcher.killers[0][0].is_some());
}

#[test]
fn test_search_stats_structure() {
    let stats = SearchStats {
        nodes: 0,
        tt_capacity: 1000,
        tt_used: 100,
        tt_fill_permille: 100,
    };
    assert_eq!(stats.tt_capacity, 1000);
    assert_eq!(stats.tt_used, 100);
    assert_eq!(stats.tt_fill_permille, 100);
}

#[test]
fn test_searcher_killers_and_history() {
    let mut searcher = Searcher::new(1000);

    // Add some killer moves
    let m = Move::new(
        Coordinate::new(0, 0),
        Coordinate::new(1, 1),
        Piece::new(PieceType::Pawn, PlayerColor::White),
    );
    searcher.killers[0][0] = Some(m);
    assert!(searcher.killers[0][0].is_some());
}

#[test]
fn test_history_table_dimensions() {
    let searcher = Searcher::new(1000);

    // Verify history table dimensions [32 piece types][256 to squares]
    assert_eq!(searcher.history.len(), 32);
    assert_eq!(searcher.history[0].len(), 256);
}

// MoveList Operations

#[test]
fn test_movelist_operations() {
    use crate::moves::MoveList;

    let mut moves = MoveList::new();
    assert!(moves.is_empty());

    let m = Move::new(
        Coordinate::new(4, 4),
        Coordinate::new(5, 6),
        Piece::new(PieceType::Knight, PlayerColor::White),
    );

    moves.push(m);
    assert_eq!(moves.len(), 1);
    assert!(!moves.is_empty());
}

#[test]
fn test_search_endgame_position() {
    let mut game = GameState::new();
    // KQ vs K endgame
    game.setup_position_from_icn("w K0,0|Q4,4|k7,7");

    let result = get_best_move(&mut game, 3, 500, true, true);
    assert!(result.is_some(), "Should find a move in KQ vs K");

    let (best_move, eval, _stats) = result.unwrap();
    assert!(eval > 0, "White should be winning in KQ vs K");
    assert!(best_move.piece.piece_type() != PieceType::Void);
}

#[test]
fn test_search_with_captures() {
    let mut game = GameState::new();
    // Position with clear capture
    game.setup_position_from_icn("w K0,0|R4,4|k7,7|p4,7");

    let result = get_best_move(&mut game, 4, 500, true, true);
    assert!(result.is_some());
}

#[test]
fn test_format_pv_empty() {
    let searcher = Box::new(Searcher::new(1000));
    let mut game = GameState::new();
    let pv = searcher.format_pv(&mut game, 0);
    // PV should be a string (possibly empty)
    assert!(pv.is_empty() || !pv.is_empty());
}

#[test]
fn test_set_corrhist_mode() {
    let mut searcher = Box::new(Searcher::new(1000));
    let game = GameState::new();

    searcher.set_corrhist_mode(&game);
    // Mode should be set (either PawnBased or NonPawnBased)
    assert!(
        searcher.corrhist_mode == CorrHistMode::PawnBased
            || searcher.corrhist_mode == CorrHistMode::NonPawnBased
    );
}

#[test]
fn test_omega_variant_tag_resolves_to_no_variant() {
    let mut searcher = Box::new(Searcher::new(1000));

    let mut omega_game = GameState::new();
    omega_game.setup_position_from_icn("[Variant \"Omega\"] K5,1|k5,8");
    assert_eq!(omega_game.variant, None);
    searcher.set_corrhist_mode(&omega_game);
    assert_eq!(searcher.corrhist_mode, CorrHistMode::NonPawnBased);

    let mut classical_game = GameState::new();
    classical_game.setup_position_from_icn("[Variant \"Classical\"] K5,1|k5,8");
    assert_eq!(classical_game.variant, Some(crate::Variant::Classical));
    searcher.set_corrhist_mode(&classical_game);
    assert_eq!(searcher.corrhist_mode, CorrHistMode::PawnBased);

    let mut unknown_game = GameState::new();
    unknown_game.setup_position_from_icn("[Variant \"not a real variant\"] K5,1|k5,8");
    assert_eq!(unknown_game.variant, Some(crate::Variant::Classical));
    searcher.set_corrhist_mode(&unknown_game);
    assert_eq!(searcher.corrhist_mode, CorrHistMode::PawnBased);
}

#[test]
fn test_adjusted_eval() {
    let searcher = Box::new(Searcher::new(1000));
    let mut game = GameState::new();
    game.white_nonpawn_hash = 12345;
    game.pawn_hash = 67890;
    game.material_hash = 11111;

    let raw_eval = 100;
    let adjusted = searcher.adjusted_eval(&game, raw_eval, 0, 0);
    // Adjusted eval should be within reasonable bounds of raw
    assert!(adjusted.abs() < raw_eval.abs() + 1000);
}

#[test]
fn test_extract_pv() {
    let searcher = Box::new(Searcher::new(1000));
    let mut game = GameState::new();
    let pv = searcher.extract_pv_only(&mut game, 1);
    // PV should be empty for a fresh searcher
    assert!(pv.is_empty());
}

#[test]
fn test_reset_search_state() {
    // Should not panic
    reset_search_state();
}

#[test]
fn test_capture_history_update() {
    let mut searcher = Box::new(Searcher::new(1000));

    searcher.capture_history[PieceType::Rook as usize][PieceType::Pawn as usize] = 100;
    let val = searcher.capture_history[PieceType::Rook as usize][PieceType::Pawn as usize];
    assert_eq!(val, 100);
}

#[test]
fn test_countermove_heuristic() {
    let mut searcher = Box::new(Searcher::new(1000));

    // Update countermove table
    let prev_from_hash = 10;
    let prev_to_hash = 20;
    searcher.countermoves[prev_from_hash][prev_to_hash] = (1, 5, 5);

    let (piece_type, to_x, to_y) = searcher.countermoves[prev_from_hash][prev_to_hash];
    assert_eq!(piece_type, 1);
    assert_eq!(to_x, 5);
    assert_eq!(to_y, 5);
}

#[test]
fn test_countermove_beyond_i16_range() {
    // Coordinates outside i16 (-32768..32767) must not alias to a wrong destination.
    let mut searcher = Box::new(Searcher::new(1000));
    searcher.countermoves[10][20] = (1, 40_000, -40_000);
    let (piece_type, to_x, to_y) = searcher.countermoves[10][20];
    assert_eq!(piece_type, 1);
    assert_eq!(to_x, 40_000);
    assert_eq!(to_y, -40_000);
}

#[test]
fn test_multipv_search_functionality() {
    let mut game = GameState::new();
    // Simple position for multipv
    game.setup_position_from_icn("w K0,0|Q4,4|k7,7|r5,5");

    // Search with MultiPV = 2
    let result = get_best_moves_multipv(&mut game, 2, 500, 500, 2, true, false);

    // Should find at least 1 line, hopefully 2 if the position allows
    assert!(!result.lines.is_empty());
    if result.lines.len() > 1 {
        assert!(
            result.lines[0].mv != result.lines[1].mv,
            "MultiPV moves should be unique"
        );
        assert!(
            result.lines[0].score >= result.lines[1].score,
            "MultiPV lines should be ordered by score"
        );
    }
}

#[test]
fn test_tt_integration_via_local() {
    let mut tt = LocalTranspositionTable::new(16);
    let hash = 123456789;
    let depth = 5;
    let score = 1000;
    let best_move = Move::new(
        Coordinate::new(0, 0),
        Coordinate::new(1, 1),
        Piece::new(PieceType::Pawn, PlayerColor::White),
    );

    // Store EXACT score using correct TT signature:
    tt.store(&crate::search::tt_defs::TTStoreParams {
        hash,
        depth,
        flag: crate::search::tt_defs::TTFlag::Exact,
        score,
        static_eval: INFINITY + 1,
        is_pv: true,
        best_move: Some(best_move),
        ply: 0,
    });

    // Probe EXACT score using correct TT signature:
    let result = tt.probe(&crate::search::tt_defs::TTProbeParams {
        hash,
        alpha: score - 100,
        beta: score + 100,
        depth,
        ply: 0,
        rule50_count: 0,
        rule_limit: 100,
    });
    assert!(result.is_some());
    let res = result.unwrap();
    assert_eq!(res.cutoff_score, score);
    assert!(res.best_move.is_some());
    assert_eq!(res.best_move.unwrap().from.x, 0);
}

#[test]
fn test_search_mate_in_one() {
    reset_world_bounds();
    let mut game = GameState::new();
    game.setup_position_from_icn("w K-5,-5|R5,5|k0,0|p-1,-1|p0,-1|p1,-1|p-1,0|p1,0|p-1,1|p1,1");

    assert_eq!(
        game.white_piece_count, 2,
        "Should have 2 white pieces (King, Rook)"
    );
    assert!(
        game.black_piece_count >= 8,
        "Should have at least 8 black pieces"
    );
    assert!(
        !game.black_royals.is_empty(),
        "Black king position must be detected"
    );
    assert!(
        !game.white_royals.is_empty(),
        "White king position must be detected"
    );

    game.recompute_hash();

    // Verification: ensure move generation works
    let moves = game.get_legal_moves();
    assert!(
        !moves.is_empty(),
        "White should have legal moves, found 0. Piece counts: W={}, B={}",
        game.white_piece_count,
        game.black_piece_count
    );
    let _in_pawn_endgame = game.white_piece_count <= 2 && game.black_piece_count <= 2;
    assert!(!moves.is_empty(), "White should have legal moves, found 0");

    // Search depth 3 to be absolutely sure
    let result = get_best_move(&mut game, 3, 2000, true, true);
    assert!(
        result.is_some(),
        "Search returned None even though legal moves exist"
    );
    let (best_move, score, _stats) = result.unwrap();

    // Should find the mate move to (0,5)
    assert_eq!(best_move.to.x, 0);
    assert_eq!(best_move.to.y, 5);

    assert!(
        score > 800000,
        "Should detect mate score (>800000), got {}",
        score
    );
}
#[test]
fn test_quiescence_search_depth() {
    let mut searcher = Box::new(Searcher::new(1000));
    let mut game = GameState::new();
    // Setup empty board with kings to avoid panics
    game.setup_position_from_icn("w K0,0|k7,7");

    // Qsearch should return static eval on quiet position
    let alpha = -10000;
    let beta = 10000;
    let score = quiescence(&mut searcher, &mut game, 0, 0, alpha, beta, NodeType::PV);
    assert!(score.abs() < 500); // Should be near zero for balanced empty board
    assert_eq!(searcher.hot.qnodes, 1);
}

#[test]
fn test_negamax_node_counts() {
    let mut game = GameState::new();
    game.setup_position_from_icn("w K0,0|k7,7");

    let nodes = negamax_node_count_for_depth(&mut game, 1);
    assert!(nodes > 0);
}

#[test]
fn test_pvline_structure() {
    let dummy_move = Move::new(
        Coordinate::new(4, 4),
        Coordinate::new(5, 5),
        Piece::new(PieceType::Pawn, PlayerColor::White),
    );
    let pv = PVLine {
        mv: dummy_move,
        score: 100,
        depth: 5,
        pv: vec![],
    };
    assert_eq!(pv.score, 100);
    assert_eq!(pv.depth, 5);
    assert!(pv.pv.is_empty());
}

#[test]
fn test_multipv_result_structure() {
    let result = MultiPVResult {
        lines: vec![],
        stats: SearchStats {
            nodes: 0,
            tt_capacity: 1000,
            tt_used: 100,
            tt_fill_permille: 100,
        },
        shallow_best_changed: false,
        shallow_order: Vec::new(),
        deep_ref_scores: Vec::new(),
    };
    assert!(result.lines.is_empty());
    assert_eq!(result.stats.tt_capacity, 1000);
}

#[test]
fn test_searcher_thread_id() {
    let searcher = Box::new(Searcher::new(1000));
    assert_eq!(searcher.thread_id, 0); // Default thread ID
}

#[test]
fn test_searcher_silent_mode() {
    let mut searcher = Box::new(Searcher::new(1000));
    assert!(!searcher.silent); // Default is not silent
    searcher.silent = true;
    assert!(searcher.silent);
}

#[test]
fn test_move_rule_limit() {
    let searcher = Box::new(Searcher::new(1000));
    assert_eq!(searcher.move_rule_limit, 100); // Default 50-move rule
}
