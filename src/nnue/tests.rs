use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::nnue::{self, NnueState};

// Setup game from ICN string
fn setup_game(icn: &str) -> GameState {
    let mut game = GameState::new();
    game.setup_position_from_icn(icn);
    game
}

// Verify RelKP bucket calculation
#[test]
fn test_relkp_encoding_buckets() {
    // White King at 5,1 (e1), White Pawn at 5,2 (e2)
    let icn = "w 0/100 1 (8|1) K5,1+|P5,2+";
    let game = setup_game(icn);

    let (white_features, _) = nnue::features::build_relkp_active_lists(&game);

    // Expect features for the pawn
    assert!(
        !white_features.is_empty(),
        "Should have features for the pawn"
    );
}

// Ensure incremental updates match scratch generation
#[test]
fn test_incremental_vs_scratch() {
    // Position with various pieces
    let start_icn = "w 0/100 1 (8|1) K5,1+|P5,2+|k5,8+";
    let game = setup_game(start_icn);

    // Initial state
    let mut state = NnueState::from_position(&game);

    // Move: White Pawn e2-e4
    let from = Coordinate::new(5, 2);
    let to = Coordinate::new(5, 4);
    let piece = Piece::new(PieceType::Pawn, PlayerColor::White);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: None,
    };

    // Update state incrementally
    state.update_for_move(&game, m);

    // Apply move to board manually for verification
    let mut game_after = game.clone();
    game_after.board.remove_piece(&from.x, &from.y);
    game_after.board.set_piece(to.x, to.y, piece);
    game_after.turn = PlayerColor::Black;
    game_after.recompute_piece_counts();

    // Calculate state from scratch
    let state_scratch = NnueState::from_position(&game_after);

    // Verify consistency
    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "White accumulator mismatch"
    );
    assert_eq!(
        state.rel_acc_black, state_scratch.rel_acc_black,
        "Black accumulator mismatch"
    );
}

#[test]
fn test_incremental_capture() {
    // White P5,4 captures Black P4,5
    let start_icn = "w 0/100 1 (8|1) K5,1+|P5,4+|k5,8+|p4,5+";
    let game = setup_game(start_icn);

    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(5, 4);
    let to = Coordinate::new(4, 5);
    let piece = Piece::new(PieceType::Pawn, PlayerColor::White);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: None,
    };

    state.update_for_move(&game, m);

    // Apply capture to board
    let mut game_after = game.clone();
    game_after.board.remove_piece(&from.x, &from.y);
    game_after.board.set_piece(to.x, to.y, piece); // Overwrites captured piece
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);

    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "White acc mismatch capture"
    );
    assert_eq!(
        state.rel_acc_black, state_scratch.rel_acc_black,
        "Black acc mismatch capture"
    );
}

#[test]
fn test_incremental_king_move() {
    // King move (expensive update)
    let start_icn = "w 0/100 1 (8|1) K5,1+|P5,2+|k5,8+";
    let game = setup_game(start_icn);

    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(5, 1);
    let to = Coordinate::new(6, 1);
    let piece = Piece::new(PieceType::King, PlayerColor::White);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: None,
    };

    state.update_for_move(&game, m);

    let mut game_after = game.clone();
    game_after.board.remove_piece(&from.x, &from.y);
    game_after.board.set_piece(to.x, to.y, piece);
    game_after.white_king_pos = Some(to);
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);

    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "White acc mismatch King move"
    );
    assert_eq!(
        state.rel_acc_black, state_scratch.rel_acc_black,
        "Black acc mismatch King move"
    );
}

#[test]
fn test_incremental_king_move_black() {
    // Black King move: k5,8 -> k6,8
    let start_icn = "b 0/100 1 (8|1) K1,1+|k5,8+|p5,7+";
    let game = setup_game(start_icn);
    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(5, 8);
    let to = Coordinate::new(6, 8);
    let piece = Piece::new(PieceType::King, PlayerColor::Black);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: None,
    };

    state.update_for_move(&game, m);

    let mut game_after = game.clone();
    game_after.board.remove_piece(&5, &8);
    game_after.board.set_piece(6, 8, piece);
    game_after.black_king_pos = Some(to);
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);
    assert_eq!(
        state.rel_acc_black, state_scratch.rel_acc_black,
        "Black King acc mismatch"
    );
}

// Verify ThreatEdges features are generated
#[test]
fn test_threat_active_lists() {
    // Setup position with various threats
    // White: Rook (4,4), Knight (8,4), King (1,1), Queen (10,10), Bishop (2,2)
    // Black: Pawn (4,5), Slider blocker (5,5), King (8,8)
    let icn = "w 0/100 1 (8|1) K1,1+|R4,4+|N8,4+|Q10,10+|B2,2+|k8,8+|p4,5+|n5,5+";
    let game = setup_game(icn);

    let (white_threats, black_threats) = nnue::features::build_threat_active_lists(&game);

    assert!(
        !white_threats.is_empty(),
        "White should have threat features (Rook, Knight, Queen, Bishop)"
    );
    assert!(
        !black_threats.is_empty(),
        "Black should have threat features (Pawn, King)"
    );
}

// Smoke test for the main evaluate entry point
#[test]
fn test_evaluate_smoke() {
    let icn = "w 0/100 1 (8|1) K5,1+|P5,2+|k5,8+";
    let game = setup_game(icn);

    // Calls the public API
    // If weights are missing, it returns 0. If present, some score.
    // Just ensuring it doesn't panic.
    let score = nnue::evaluate(&game);

    // In test environment, weights might be missing -> 0
    // But verification is simply "no panic".
    assert!(
        (-10000..=10000).contains(&score),
        "Score within reasonable bounds"
    );
}

#[test]
fn test_incremental_en_passant() {
    // White Pawn captures Black Pawn via EP
    // White P5,5, Black p4,5, EP square 4,6
    let start_icn = "w 0/100 1 (8|1) K1,1+|P5,5+|k8,8+|p4,5+ 4,6";
    let game = setup_game(start_icn);
    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(5, 5);
    let to = Coordinate::new(4, 6);
    let piece = Piece::new(PieceType::Pawn, PlayerColor::White);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: None,
    };

    state.update_for_move(&game, m);

    let mut game_after = game.clone();
    game_after.board.remove_piece(&5, &5);
    game_after.board.remove_piece(&4, &5); // EP captured pawn
    game_after.board.set_piece(4, 6, piece);
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);
    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "EP white mismatch"
    );
}

#[test]
fn test_incremental_promotion() {
    // White Pawn promotes on rank 8
    let start_icn = "w 0/100 1 (8|1) K1,1+|P4,7+|k8,8+";
    let game = setup_game(start_icn);
    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(4, 7);
    let to = Coordinate::new(4, 8);
    let piece = Piece::new(PieceType::Pawn, PlayerColor::White);
    let promo = PieceType::Queen;

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: Some(promo),
        rook_coord: None,
    };

    state.update_for_move(&game, m);

    let mut game_after = game.clone();
    game_after.board.remove_piece(&4, &7);
    game_after
        .board
        .set_piece(4, 8, Piece::new(promo, PlayerColor::White));
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);
    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "Promotion white mismatch"
    );
}

#[test]
fn test_incremental_castling() {
    // White King e1-g1 (5,1 to 7,1), Rook h1-f1 (8,1 to 6,1)
    let start_icn = "w 0/100 1 (8|1) K5,1+|R8,1+|k5,8+";
    let game = setup_game(start_icn);
    let mut state = NnueState::from_position(&game);

    let from = Coordinate::new(5, 1);
    let to = Coordinate::new(7, 1);
    let piece = Piece::new(PieceType::King, PlayerColor::White);
    let rook_from = Coordinate::new(8, 1);

    let m = crate::moves::Move {
        from,
        to,
        piece,
        promotion: None,
        rook_coord: Some(rook_from),
    };

    state.update_for_move(&game, m);

    let mut game_after = game.clone();
    game_after.board.remove_piece(&5, &1);
    game_after.board.remove_piece(&8, &1);
    game_after.board.set_piece(7, 1, piece);
    game_after
        .board
        .set_piece(6, 1, Piece::new(PieceType::Rook, PlayerColor::White));
    game_after.white_king_pos = Some(to);
    game_after.recompute_piece_counts();

    let state_scratch = NnueState::from_position(&game_after);
    assert_eq!(
        state.rel_acc_white, state_scratch.rel_acc_white,
        "Castling white mismatch"
    );
    assert_eq!(
        state.rel_acc_black, state_scratch.rel_acc_black,
        "Castling black mismatch"
    );
}
