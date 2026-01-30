//! Endgame Mate Tests - Excluded from coverage
//!
//! These tests verify the engine can find checkmate in specific endgames
//! within the 50-move rule. They are excluded from coverage reports due to their
//! time-consuming nature.
//!
//! INFINITE CHESS NOTES:
//! - Some classical chess mates don't work on infinite board (e.g., 2R vs K)
//! - Tests include positions where one side has no king

#![cfg(not(coverage))]

use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor};
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::search::get_best_move;

/// Helper to create a minimal endgame position
/// white_has_king: if false, white has no king (only pieces)
/// black_has_king: if false, black has no king (only pieces)
fn create_endgame(
    white_pieces: &[(i64, i64, PieceType)],
    black_pieces: &[(i64, i64, PieceType)],
) -> GameState {
    let mut game = GameState::new();
    game.board = Board::new();
    game.special_rights.clear();
    game.en_passant = None;
    game.turn = PlayerColor::White;
    game.halfmove_clock = 0;
    game.fullmove_number = 1;
    game.hash_stack.clear();
    game.null_moves = 0;

    for (x, y, pt) in white_pieces {
        game.board
            .set_piece(*x, *y, Piece::new(*pt, PlayerColor::White));
    }
    for (x, y, pt) in black_pieces {
        game.board
            .set_piece(*x, *y, Piece::new(*pt, PlayerColor::Black));
    }

    game.recompute_piece_counts();
    game.recompute_hash();
    game.init_starting_piece_counts();

    // Recalculate material score
    let mut score = 0;
    for (_, _, piece) in game.board.iter() {
        let val = hydrochess_wasm::evaluation::get_piece_value(piece.piece_type());
        match piece.color() {
            PlayerColor::White => score += val,
            PlayerColor::Black => score -= val,
            PlayerColor::Neutral => {}
        }
    }
    game.material_score = score;

    game
}

/// Verify that the engine can find checkmate within max_ply moves
fn verify_can_mate(mut game: GameState, max_ply: u32, description: &str) {
    let mut ply = 0;

    println!("\n=== Testing: {} ===", description);

    while ply < max_ply {
        // Check if already checkmate or stalemate
        let moves = game.get_legal_moves();
        let legal_count = moves
            .iter()
            .filter(|m| {
                let undo = game.make_move(m);
                let illegal = game.is_move_illegal();
                game.undo_move(m, undo);
                !illegal
            })
            .count();

        if legal_count == 0 {
            if game.is_in_check() {
                println!("  Checkmate found at ply {}", ply);
                return; // Checkmate!
            } else {
                panic!(
                    "{}: Stalemate at ply {} - engine failed to avoid stalemate",
                    description, ply
                );
            }
        }

        // Get best move with reasonable think time
        let result = get_best_move(&mut game, 20, 2000, true, true);

        let best_move = match result {
            Some((m, _eval, _stats)) => m,
            None => panic!("{}: No move found at ply {}", description, ply),
        };

        // Make the move
        let undo = game.make_move(&best_move);
        if game.is_move_illegal() {
            game.undo_move(&best_move, undo);
            panic!(
                "{}: Engine returned illegal move at ply {}",
                description, ply
            );
        }

        ply += 1;

        // Switch sides for next iteration
        // (game.turn is already updated by make_move)
    }

    panic!("{}: Failed to mate within {} ply", description, max_ply);
}

// =============================================================================
// INFINITE CHESS MATE TESTS
// These positions work on infinite boards
// =============================================================================

/// 2Q + K vs K - Two queens can always mate (even on infinite board)
#[test]
fn test_mate_2queen_vs_king() {
    let game = create_endgame(
        &[(4, 4, PieceType::Queen), (6, 4, PieceType::Queen)],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "2Q vs K");
}

/// Q + R + K vs K - Queen and rook can mate (enough to cut off king)
#[test]
fn test_mate_queen_rook_vs_king() {
    let game = create_endgame(
        &[
            (5, 1, PieceType::King),
            (4, 4, PieceType::Queen),
            (1, 1, PieceType::Rook),
        ],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "Q + R + K vs K");
}

// 2R + K vs K
#[test]
fn test_mate_2rook_vs_king() {
    let game = create_endgame(
        &[
            (5, 1, PieceType::King),
            (4, 4, PieceType::Rook),
            (6, 4, PieceType::Rook),
        ],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "2R + K vs K");
}

/// Amazon + K vs K - Amazon (Queen + Knight) is stronger than queen
#[test]
fn test_mate_amazon_vs_king() {
    let game = create_endgame(
        &[(5, 1, PieceType::King), (4, 4, PieceType::Amazon)],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "Amazon + K vs K");
}

// =============================================================================
// COMPOUND PIECE MATE TESTS
// =============================================================================

/// Q + R + B  vs K
#[test]
fn test_mate_queen_rook_bishop_vs_king() {
    let game = create_endgame(
        &[
            (4, 4, PieceType::Queen),
            (1, 1, PieceType::Rook),
            (3, 1, PieceType::Bishop),
        ],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "Q + R + B vs K");
}

/// Q + R + N + K vs K  
#[test]
fn test_mate_queen_rook_knight_vs_king() {
    let game = create_endgame(
        &[
            (4, 4, PieceType::Queen),
            (1, 1, PieceType::Rook),
            (2, 1, PieceType::Knight),
        ],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "Q + R + N vs K");
}

/// Archbishop + K vs K - Archbishop = Bishop + Knight
#[test]
fn test_mate_archbishop_vs_king() {
    // Archbishop alone can't force mate, need additional material
    let game = create_endgame(
        &[
            (5, 1, PieceType::King),
            (4, 4, PieceType::Archbishop),
            (1, 1, PieceType::Rook),
        ],
        &[(5, 8, PieceType::King)],
    );
    verify_can_mate(game, 100, "Archbishop + R + K vs K");
}
