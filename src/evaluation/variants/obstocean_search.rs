//! Modified quiescence search for Obstocean
//! 
//! This is quite similar to the standard quiescence search, but it now accounts for pawn captures
//! away from the center and outside the "board", as it is a common motif here.

use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::{EnPassantState, GameRules};
use crate::moves::{Move, MoveGenContext, MoveGenType, MoveList,
    generate_compass_moves_into, generate_huygen_moves_into, generate_knightrider_moves_into, generate_leaper_moves_into,
    generate_pawn_quiet_promotions, generate_rose_moves_into, generate_sliding_capture_moves, is_enemy_piece};
use rustc_hash::{FxHashSet};

/// Generate only capturing moves for quiescence search when the side to move is **not** in check.
/// This avoids generating and then filtering thousands of quiet moves.
pub fn get_quiescence_captures(
    board: &Board,
    turn: PlayerColor,
    ctx: &MoveGenContext,
    out: &mut MoveList,
) {
    use crate::tiles::TILE_SIZE;

    out.clear();

    // BITBOARD: CTZ iteration for O(popcount) piece enumeration
    let is_white = turn == PlayerColor::White;

    for (cx, cy, tile) in board.tiles.iter() {
        let occ = if is_white {
            tile.occ_white
        } else {
            tile.occ_black
        };
        if occ == 0 {
            continue;
        }

        let mut bits = occ;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }

            let piece = Piece::from_packed(packed);

            let lx = (idx % 8) as i64;
            let ly = (idx / 8) as i64;
            let x = cx * TILE_SIZE + lx;
            let y = cy * TILE_SIZE + ly;
            let from = Coordinate::new(x, y);

            generate_captures_for_piece(board, &piece, &from, ctx, out);
        }
    }
}

// Helper to avoid duplicating the switch logic
fn generate_captures_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    ctx: &MoveGenContext,
    out: &mut MoveList,
) {
    let special_rights = ctx.special_rights;
    let en_passant = ctx.en_passant;
    let game_rules = ctx.game_rules;
    let indices = ctx.indices;
    match piece.piece_type() {
        PieceType::Void | PieceType::Obstacle => {}

        // Pawns: only capture and en-passant moves (with promotions when applicable)
        PieceType::Pawn => {
            generate_pawn_capture_moves(
                board,
                from,
                piece,
                special_rights,
                en_passant,
                game_rules,
                out,
            );
            generate_pawn_quiet_promotions(board, from, piece, special_rights, game_rules, out);
        }

        // Knight-like leapers
        PieceType::Knight => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Camel => {
            generate_leaper_moves_into(board, from, piece, 1, 3, MoveGenType::Captures, out);
        }
        PieceType::Giraffe => {
            generate_leaper_moves_into(board, from, piece, 1, 4, MoveGenType::Captures, out);
        }
        PieceType::Zebra => {
            generate_leaper_moves_into(board, from, piece, 2, 3, MoveGenType::Captures, out);
        }

        // King/Guard/Centaur/RoyalCentaur/Hawk: use compass moves, then filter captures
        PieceType::King | PieceType::Guard => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Captures, out);
        }
        PieceType::Centaur | PieceType::RoyalCentaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Captures, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Hawk => {
            generate_compass_moves_into(board, from, piece, 2, MoveGenType::Captures, out);
            generate_compass_moves_into(board, from, piece, 3, MoveGenType::Captures, out);
        }

        // Standard sliders and slider-leaper compounds
        PieceType::Rook => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
        }
        PieceType::Bishop => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
        }
        PieceType::Chancellor => {
            // Rook + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Archbishop => {
            // Bishop + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Amazon => {
            // Queen + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }

        // Knightrider: sliding along knight vectors
        PieceType::Knightrider => {
            generate_knightrider_moves_into(board, from, piece, MoveGenType::Captures, out);
        }

        // Huygen: use existing generator and keep only captures
        PieceType::Huygen => {
            generate_huygen_moves_into(board, from, piece, indices, MoveGenType::Captures, out);
        }

        // Rose: use existing generator and keep only captures
        PieceType::Rose => {
            generate_rose_moves_into(board, from, piece, MoveGenType::Captures, out);
        }
    }
}

/// Generate only pawn captures (including en passant) for quiescence.
fn generate_pawn_capture_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    _special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    out: &mut MoveList,
) {
    let direction = match piece.color() {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get promotion ranks for this color
    let ranks = &game_rules.promotion_ranks;
    let promotion_ranks = match piece.color() {
        PlayerColor::White => &ranks.white,
        PlayerColor::Black => &ranks.black,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get allowed promotion pieces (use pre-converted types, default to Q, R, B, N)
    let default_promos = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    let promotion_pieces: &[PieceType] = game_rules
        .promotion_types
        .as_deref()
        .unwrap_or(&default_promos);

    // Local helper mirroring generate_pawn_moves promotion handling
    fn add_pawn_cap_move(
        out: &mut MoveList,
        from: Coordinate,
        to_x: i64,
        to_y: i64,
        piece: Piece,
        promotion_ranks: &[i64],
        promotion_pieces: &[PieceType],
    ) {
        if promotion_ranks.contains(&to_y) {
            for &promo in promotion_pieces {
                let mut m = Move::new(from, Coordinate::new(to_x, to_y), piece);
                m.promotion = Some(promo);
                out.push(m);
            }
        } else {
            out.push(Move::new(from, Coordinate::new(to_x, to_y), piece));
        }
    }

    // Captures (including neutral pieces - they can be captured)
    for dx in [-1i64, 1] {
        let capture_x = from.x + dx;
        let capture_y = from.y + direction;

        if let Some(target) = board.get_piece(capture_x, capture_y) {
            if is_enemy_piece(&target, piece.color()) {
                // In Obstocean, we allow pawn captures that promote to queen, are outside the "board", or go away from the center.
                let is_neutral = target.piece_type().is_neutral_type();
                let capturing_away = (capture_x <= 3 && capture_x < from.x) || (capture_x >= 6 && capture_x > from.x);
                let capturing_outside = capture_x < 1 || capture_x > 8;

                if !is_neutral || promotion_ranks.contains(&capture_y) || capturing_away || capturing_outside {
                    add_pawn_cap_move(
                        out,
                        *from,
                        capture_x,
                        capture_y,
                        *piece,
                        promotion_ranks,
                        promotion_pieces,
                    );
                }
            }
        } else if en_passant
            .as_ref()
            .is_some_and(|ep| ep.square.x == capture_x && ep.square.y == capture_y)
        {
            add_pawn_cap_move(
                out,
                *from,
                capture_x,
                capture_y,
                *piece,
                promotion_ranks,
                promotion_pieces,
            );
        }
    }
}
