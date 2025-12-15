use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor};
use hydrochess_wasm::evaluation::insufficient_material::evaluate_insufficient_material;
use std::time::Instant;

/// Create various test positions with different piece configurations
fn create_test_positions() -> Vec<Board> {
    let mut positions = Vec::new();

    // Position 1: King vs King (insufficient)
    let mut board1 = Board::new();
    board1.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board1.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board1);

    // Position 2: King + Bishop vs King (insufficient - same color bishops)
    let mut board2 = Board::new();
    board2.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board2.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White)); // light square
    board2.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board2);

    // Position 3: King + Knight vs King (insufficient)
    let mut board3 = Board::new();
    board3.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board3.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board3.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board3);

    // Position 4: King + Rook vs King (sufficient)
    let mut board4 = Board::new();
    board4.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board4.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board4.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board4);

    // Position 5: King + Queen vs King (sufficient)
    let mut board5 = Board::new();
    board5.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board5.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board5.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board5);

    // Position 6: King + 2 Knights vs King (insufficient with standard rules)
    let mut board6 = Board::new();
    board6.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board6.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board6.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board6.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board6);

    // Position 7: Lone Queen (0K) (insufficient)
    let mut board7 = Board::new();
    board7.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board7.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board7);

    // Position 8: 2 Rooks (0K) (insufficient)
    let mut board8 = Board::new();
    board8.set_piece(0, 0, Piece::new(PieceType::Rook, PlayerColor::White));
    board8.set_piece(1, 0, Piece::new(PieceType::Rook, PlayerColor::White));
    board8.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board8);

    // Position 9: King + Bishop + Knight vs King (insufficient)
    let mut board9 = Board::new();
    board9.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board9.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White));
    board9.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board9.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board9);

    // Position 10: King + 2 Bishops (opposite colors) vs King (insufficient)
    let mut board10 = Board::new();
    board10.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board10.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White)); // light
    board10.set_piece(2, 3, Piece::new(PieceType::Bishop, PlayerColor::White)); // dark
    board10.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board10);

    positions
}

#[test]
fn bench_insufficient_material_100k() {
    let positions = create_test_positions();
    let iterations = 100_000;

    println!("\n=== Insufficient Material Benchmark ===");
    println!(
        "Testing {} positions, {} iterations each",
        positions.len(),
        iterations
    );

    let start = Instant::now();

    let mut results = vec![false; positions.len()];

    for _ in 0..iterations {
        for (i, board) in positions.iter().enumerate() {
            results[i] = evaluate_insufficient_material(board).is_some();
        }
    }

    let elapsed = start.elapsed();
    let total_checks = iterations * positions.len();
    let per_check_ns = elapsed.as_nanos() / total_checks as u128;

    println!("\nResults:");
    println!("  Total time: {:?}", elapsed);
    println!("  Total checks: {}", total_checks);
    println!("  Time per check: {} ns", per_check_ns);
    println!(
        "  Checks per second: {:.2}M",
        (total_checks as f64) / elapsed.as_secs_f64() / 1_000_000.0
    );

    // Print individual position results
    println!("\nPosition results:");
    for (i, result) in results.iter().enumerate() {
        let status = if *result {
            "DRAW (insufficient)"
        } else {
            "PLAYABLE (sufficient)"
        };
        println!("  Position {}: {}", i + 1, status);
    }
}

#[test]
fn test_insufficient_material_correctness() {
    // Test specific scenarios for correctness

    // K vs K (Zero extra on both sides) -> Some(0) (dead draw)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K vs K should be Some(0) (dead draw)"
    );

    // K+B vs K (White 1 extra, Black 0). Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K+B vs K should be Some(0) (Lone King)"
    );

    // K+N vs K (White 1 extra, Black 0). Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K+N vs K should be Some(0) (Lone King)"
    );

    // K vs K+N (White 0 extra, Black 1). Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    board.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K vs K+N should be Some(0) (Lone King)"
    );

    // K+N vs K+N (Equal extra). Both have pieces -> Some(8) (drawish)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(0, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    board.set_piece(4, 5, Piece::new(PieceType::Knight, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(8),
        "K+N vs K+N should be Some(8) (Drawish)"
    );

    // K+2N vs K+N (White 2, Black 1). Both have pieces -> Some(8) (drawish)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(0, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(0, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    board.set_piece(4, 5, Piece::new(PieceType::Knight, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(8),
        "K+2N vs K+N should be Some(8) (Drawish)"
    );

    // K+R vs K (White 1, Black 0). Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K+R vs K should be Some(0) (Lone King)"
    );

    // K+Q vs K (White 1, Black 0). Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "K+Q vs K should be Some(0) (Lone King)"
    );

    // K+2Q vs K. sufficient -> None
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(3, 4, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        None,
        "K+2Q vs K should be None (Sufficient)"
    );

    // Lone Rook (0K) vs K. Black is Lone King -> Some(0)
    let mut board = Board::new();
    board.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(0),
        "Lone R vs K should be Some(0) (Lone King)"
    );

    // K+R vs Q+R (White 1, Black 2). Both have pieces -> Some(8) (drawish)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(0, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    // Black: Q+R (no King)
    board.set_piece(7, 7, Piece::new(PieceType::Queen, PlayerColor::Black));
    board.set_piece(7, 6, Piece::new(PieceType::Rook, PlayerColor::Black));
    assert_eq!(
        evaluate_insufficient_material(&board),
        Some(8),
        "K+R vs Q+R (No Black King) should be Some(8) (Drawish)"
    );

    println!("All correctness tests passed!");
}
