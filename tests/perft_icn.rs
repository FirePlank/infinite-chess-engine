#![cfg(not(coverage))]
use hydrochess_wasm::board::{Coordinate, PieceType, PlayerColor};
use hydrochess_wasm::game::GameState;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PerftStats {
    pub nodes: u64,
    pub captures: u64,
    pub en_passant: u64,
    pub castles: u64,
    pub promotions: u64,
    pub checks: u64,
    pub discovery_checks: u64,
    pub double_checks: u64,
    pub checkmates: u64,
}

impl std::ops::Add for PerftStats {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            nodes: self.nodes + other.nodes,
            captures: self.captures + other.captures,
            en_passant: self.en_passant + other.en_passant,
            castles: self.castles + other.castles,
            promotions: self.promotions + other.promotions,
            checks: self.checks + other.checks,
            discovery_checks: self.discovery_checks + other.discovery_checks,
            double_checks: self.double_checks + other.double_checks,
            checkmates: self.checkmates + other.checkmates,
        }
    }
}

impl std::ops::AddAssign for PerftStats {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Converts a standard FEN string to Infinite Chess ICN string.
fn fen_to_icn(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let piece_str = parts[0];
    let turn = parts.get(1).unwrap_or(&"w");
    let castling = parts.get(2).unwrap_or(&"-");
    let ep = parts.get(3).unwrap_or(&"-");
    let halfmove = parts.get(4).unwrap_or(&"0");
    let fullmove = parts.get(5).unwrap_or(&"1");

    let mut pieces = Vec::new();

    let mut y = 8;
    for rank in piece_str.split('/') {
        let mut x = 1;
        for c in rank.chars() {
            if c.is_ascii_digit() {
                x += c.to_digit(10).unwrap() as i64;
            } else {
                let code = c.to_string(); // e.g. "P", "n"

                let mut is_special = false;

                // King castling rights
                if c == 'K' && (castling.contains('K') || castling.contains('Q')) {
                    is_special = true;
                }
                if c == 'k' && (castling.contains('k') || castling.contains('q')) {
                    is_special = true;
                }

                // Rooks: simple heuristic
                if c == 'R' && y == 1 && x == 8 && castling.contains('K') {
                    is_special = true;
                }
                if c == 'R' && y == 1 && x == 1 && castling.contains('Q') {
                    is_special = true;
                }
                if c == 'r' && y == 8 && x == 8 && castling.contains('k') {
                    is_special = true;
                }
                if c == 'r' && y == 8 && x == 1 && castling.contains('q') {
                    is_special = true;
                }

                // Pawns: Standard chess pawns on 2nd and 7th rank have double move rights
                if c == 'P' && y == 2 {
                    is_special = true;
                }
                if c == 'p' && y == 7 {
                    is_special = true;
                }

                let suffix = if is_special { "+" } else { "" };
                pieces.push(format!("{}{},{}{}", code, x, y, suffix));

                x += 1;
            }
        }
        y -= 1;
    }

    let pieces_str = pieces.join("|");

    let mut icn = format!(
        "{} {}/100 {} (8;Q,R,B,N|1;q,r,b,n) 1,8,1,8 {}",
        turn, halfmove, fullmove, pieces_str
    );

    if *ep != "-" {
        let col = ep.chars().nth(0).unwrap() as u8 - b'a' + 1;
        let row = ep.chars().nth(1).unwrap().to_digit(10).unwrap();
        icn.push_str(&format!(" {},{}", col, row));
    }

    icn
}

/// Helper to fix castling rights initialization which setup_position_from_icn might miss
fn recompute_castling_rights(game: &mut GameState) {
    game.castling_partner_counts = [0; 4];
    game.effective_castling_rights = 0;

    let wk = game.white_king_pos;
    let bk = game.black_king_pos;

    // Iterate all special rights
    for coord in &game.special_rights {
        let piece = game.board.get_piece(coord.x, coord.y);
        if let Some(p) = piece {
            if p.piece_type() == PieceType::Pawn {
                continue;
            }
            if p.piece_type().is_royal() {
                continue;
            } // Kings handled separately

            // It's a partner (e.g. Rook)
            if p.color() == PlayerColor::White && wk.is_some() {
                let k = wk.unwrap();
                if coord.y == k.y {
                    if coord.x > k.x {
                        game.castling_partner_counts[0] += 1;
                    } else {
                        game.castling_partner_counts[1] += 1;
                    }
                }
            } else if p.color() == PlayerColor::Black && bk.is_some() {
                let k = bk.unwrap();
                if coord.y == k.y {
                    if coord.x > k.x {
                        game.castling_partner_counts[2] += 1;
                    } else {
                        game.castling_partner_counts[3] += 1;
                    }
                }
            }
        }
    }

    // Set meaningful effective rights bits
    // White Kingside
    if let Some(k) = wk {
        if game.special_rights.contains(&k) && game.castling_partner_counts[0] > 0 {
            game.effective_castling_rights |= 1;
        }
        if game.special_rights.contains(&k) && game.castling_partner_counts[1] > 0 {
            game.effective_castling_rights |= 2;
        }
    }
    // Black Kingside
    if let Some(k) = bk {
        if game.special_rights.contains(&k) && game.castling_partner_counts[2] > 0 {
            game.effective_castling_rights |= 4;
        }
        if game.special_rights.contains(&k) && game.castling_partner_counts[3] > 0 {
            game.effective_castling_rights |= 8;
        }
    }
}

pub fn perft(game: &mut GameState, depth: usize) -> PerftStats {
    let mut stats = PerftStats::default();

    if depth == 0 {
        stats.nodes = 1;
        return stats;
    }

    let legal_moves = game.get_legal_moves();

    for m in &legal_moves {
        // Strict Castling Check
        if m.piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1 {
            // 1. Cannot castle out of check
            if game.is_in_check() {
                continue;
            }

            // 2. Cannot castle through check
            let dir = (m.to.x - m.from.x).signum();
            let mid_x = m.from.x + dir;
            let mid_y = m.from.y;
            let mid_coord = Coordinate::new(mid_x, mid_y);

            if hydrochess_wasm::moves::is_square_attacked(
                &game.board,
                &mid_coord,
                game.turn.opponent(),
                &game.spatial_indices,
            ) {
                continue;
            }
        }

        let undo = game.make_move(m);

        let mover = game.turn.opponent();
        let current = game.turn;

        // Strict Check Legality (did we move into check?)
        game.turn = mover;
        let in_check = game.is_in_check();
        game.turn = current;

        if !in_check {
            if depth == 1 {
                // Counts for leaf move
                stats.nodes += 1;

                // Captures
                if undo.captured_piece.is_some() {
                    stats.captures += 1;
                }

                // En Passant
                if undo.ep_captured_piece.is_some() {
                    stats.en_passant += 1;
                }

                // Castles
                if m.piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1 {
                    stats.castles += 1;
                }

                // Promotions
                if m.promotion.is_some() {
                    stats.promotions += 1;
                }

                // Checks & Checkmates
                if game.is_in_check() {
                    stats.checks += 1;

                    // Optimization: Use has_legal_moves if available, or just get_legal_moves().is_empty()
                    // But get_legal_moves() is pseudo-legal...
                    // Wait, get_legal_moves SHOULD return legal moves if strict?
                    // No, "get_legal_moves returns pseudo-legal moves".
                    // So we MUST strict check them.

                    let opponent_moves = game.get_legal_moves();
                    let mut has_legal = false;

                    for op_m in &opponent_moves {
                        if op_m.piece.piece_type() == PieceType::King
                            && (op_m.to.x - op_m.from.x).abs() > 1
                        {
                            if game.is_in_check() {
                                continue;
                            }
                            let dir = (op_m.to.x - op_m.from.x).signum();
                            let mid_coord = Coordinate::new(op_m.from.x + dir, op_m.from.y);
                            if hydrochess_wasm::moves::is_square_attacked(
                                &game.board,
                                &mid_coord,
                                game.turn.opponent(),
                                &game.spatial_indices,
                            ) {
                                continue;
                            }
                        }

                        let op_undo = game.make_move(op_m);
                        let op_mover = game.turn.opponent();
                        let op_cur = game.turn;
                        game.turn = op_mover;
                        let op_in_check = game.is_in_check();
                        game.turn = op_cur;
                        game.undo_move(op_m, op_undo);

                        if !op_in_check {
                            has_legal = true;
                            break;
                        }
                    }

                    if !has_legal {
                        stats.checkmates += 1;
                    }
                }
            } else {
                stats += perft(game, depth - 1);
            }
        }

        game.undo_move(m, undo);
    }

    stats
}

fn run_fen_perft(
    name: &str,
    fen: &str,
    depth: usize,
    expected_nodes: u64,
    expected_stats: Option<PerftStats>,
) {
    let icn = fen_to_icn(fen);
    let mut game = GameState::new();

    game.setup_position_from_icn(&icn);
    // Fix: Recompute castling rights properly
    recompute_castling_rights(&mut game);

    let start = std::time::Instant::now();
    let stats = perft(&mut game, depth);
    let duration = start.elapsed();

    println!(
        "{} D{}: Nodes: {}, Time: {:?}",
        name, depth, stats.nodes, duration
    );
    println!("  Captures: {}", stats.captures);
    println!("  E.p.: {}", stats.en_passant);
    println!("  Castles: {}", stats.castles);
    println!("  Promotions: {}", stats.promotions);
    println!("  Checks: {}", stats.checks);
    println!("  Checkmates: {}", stats.checkmates);

    assert_eq!(
        stats.nodes, expected_nodes,
        "Failed perft node count for {} at depth {}",
        name, depth
    );

    if let Some(expected) = expected_stats {
        assert_eq!(stats.captures, expected.captures, "Captures mismatch");
        assert_eq!(stats.en_passant, expected.en_passant, "EP mismatch");
        assert_eq!(stats.castles, expected.castles, "Castles mismatch");
        assert_eq!(stats.promotions, expected.promotions, "Promotions mismatch");
        assert_eq!(stats.checks, expected.checks, "Checks mismatch");
        assert_eq!(stats.checkmates, expected.checkmates, "Checkmates mismatch");
    }
}

#[test]
fn perft_initial_position() {
    let expected = PerftStats {
        nodes: 197_281,
        captures: 1576,
        en_passant: 0,
        castles: 0,
        promotions: 0,
        checks: 469,
        discovery_checks: 0,
        double_checks: 0,
        checkmates: 8,
    };
    run_fen_perft(
        "Start Pos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        4,
        197_281,
        Some(expected),
    );
}

#[test]
fn perft_kiwipete() {
    let expected = PerftStats {
        nodes: 97_862,
        captures: 17_102,
        en_passant: 45,
        castles: 3_162,
        promotions: 0,
        checks: 993,
        discovery_checks: 0,
        double_checks: 0,
        checkmates: 1,
    };
    run_fen_perft(
        "Kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        3,
        97_862,
        Some(expected),
    );
}

#[test]
fn perft_position_3() {
    let expected = PerftStats {
        nodes: 43_238,
        captures: 3_348,
        en_passant: 123,
        castles: 0,
        promotions: 0,
        checks: 1_680,
        discovery_checks: 0,
        double_checks: 0,
        checkmates: 17,
    };
    run_fen_perft(
        "Pos 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        4,
        43_238,
        Some(expected),
    );
}

#[test]
fn perft_position_4() {
    let expected = PerftStats {
        nodes: 9_467,
        captures: 1_021,
        en_passant: 4,
        castles: 0,
        promotions: 120,
        checks: 38,
        discovery_checks: 0,
        double_checks: 0,
        checkmates: 22,
    };
    run_fen_perft(
        "Pos 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        3,
        9_467,
        Some(expected),
    );
}

#[test]
fn perft_position_5_depth_1_debug() {
    // Debug for Pos 5 Depth 1
    // Expected nodes: 44.

    // Convert and setup
    let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    let icn = fen_to_icn(fen);
    println!("ICN: {}", icn);

    let mut game = GameState::new();
    game.setup_position_from_icn(&icn);
    // Fix castling rights
    recompute_castling_rights(&mut game);

    // Check occupancy at d8 (4, 8)
    let d8 = game.board.get_piece(4, 8);
    println!("Piece at d8 (4,8): {:?}", d8);

    // Print Moves
    println!("Pos 5 D1 Moves (Manual):");
    let legal = game.get_legal_moves();
    let mut count = 0;
    for m in &legal {
        // Strict legality check
        if m.piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1 {
            if game.is_in_check() {
                continue;
            }
            let dir = (m.to.x - m.from.x).signum();
            let mid_coord = Coordinate::new(m.from.x + dir, m.from.y);
            if hydrochess_wasm::moves::is_square_attacked(
                &game.board,
                &mid_coord,
                game.turn.opponent(),
                &game.spatial_indices,
            ) {
                continue;
            }
        }

        let undo = game.make_move(m);
        let mover = game.turn.opponent();
        let cur = game.turn;
        game.turn = mover;
        let in_check = game.is_in_check();
        game.turn = cur;

        if !in_check {
            count += 1;
            if count >= 30 {
                println!(
                    "{}: {:?} -> {:?} (Prom: {:?})",
                    count, m.from, m.to, m.promotion
                );
            }
        }
        game.undo_move(m, undo);
    }

    println!("Total Manual Count: {}", count);
    assert_eq!(count, 44, "Manual count mismatch");

    // Standard run
    // run_fen_perft("Pos 5 Debug", fen, 1, 44, None);
}

#[test]
fn perft_initial_position_debug_d3() {
    // Start Pos Depth 3
    // Nodes: 8,902
    // Captures: 34
    // E.p.: 0
    // Castles: 0
    // Promotions: 0
    // Checks: 12
    // Checkmates: 0

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    run_fen_perft(
        "Start Pos Debug D3",
        fen,
        3,
        8_902,
        Some(PerftStats {
            nodes: 8_902,
            captures: 34,
            en_passant: 0,
            castles: 0,
            promotions: 0,
            checks: 12,
            discovery_checks: 0, // Unknown standard, usually part of checks
            double_checks: 0,
            checkmates: 0,
        }),
    );
}

#[test]
fn perft_position_5() {
    run_fen_perft(
        "Pos 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        62_379,
        None,
    );
}

#[test]
fn perft_initial_position_divide_d3() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let icn = fen_to_icn(fen);
    let mut game = GameState::new();
    game.setup_position_from_icn(&icn);
    recompute_castling_rights(&mut game);

    println!("Start Pos Divide D3:");
    let moves = game.get_legal_moves();
    for m in moves {
        let undo = game.make_move(&m);
        let nodes = perft(&mut game, 2).nodes;
        println!("{:?} -> {:?}: {}", m.from, m.to, nodes);
        game.undo_move(&m, undo);
    }
}
