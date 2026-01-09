use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, MoveList};

use super::Searcher;
use super::params::{
    DEFAULT_SORT_LOSING_CAPTURE, DEFAULT_SORT_QUIET, see_winning_threshold, sort_countermove,
    sort_hash, sort_killer1, sort_killer2, sort_winning_capture,
};

/// Low-ply history table size for moves at shallow depths
pub const LOW_PLY_HISTORY_SIZE: usize = 4;
pub const LOW_PLY_HISTORY_ENTRIES: usize = 1024;
pub const LOW_PLY_HISTORY_MASK: usize = LOW_PLY_HISTORY_ENTRIES - 1;

/// Score a single move for ordering purposes.
/// Returns higher score for better moves.
#[inline]
pub fn score_move(
    searcher: &Searcher,
    game: &GameState,
    m: &Move,
    ply: usize,
    tt_move: &Option<Move>,
) -> i32 {
    let mut score: i32 = 0;

    // Hash move bonus (highest priority)
    if let Some(ttm) = tt_move
        && m.from == ttm.from
        && m.to == ttm.to
        && m.promotion == ttm.promotion
    {
        return sort_hash(); // Early return - TT move always first
    }

    // Capture scoring
    if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
        let victim_val = get_piece_value(target.piece_type());
        let attacker_val = get_piece_value(m.piece.piece_type());
        let mvv_lva = victim_val * 10 - attacker_val;

        // SEE threshold check
        let is_winning = super::see_ge(game, m, see_winning_threshold());

        score += mvv_lva;
        if is_winning {
            score += sort_winning_capture();
        } else {
            score += DEFAULT_SORT_LOSING_CAPTURE;
        }

        // Capture history
        let cap_hist =
            searcher.capture_history[m.piece.piece_type() as usize][target.piece_type() as usize];
        score += cap_hist / 10;
    } else {
        // Quiet move scoring

        // Killer moves
        if searcher.killers[ply][0]
            .as_ref()
            .is_some_and(|k| m.from == k.from && m.to == k.to && m.promotion == k.promotion)
        {
            score += sort_killer1();
        } else if searcher.killers[ply][1]
            .as_ref()
            .is_some_and(|k| m.from == k.from && m.to == k.to && m.promotion == k.promotion)
        {
            score += sort_killer2();
        } else {
            // Countermove check
            if ply > 0 {
                let (prev_from_hash, prev_to_hash) = searcher.prev_move_stack[ply - 1];
                if prev_from_hash < 256 && prev_to_hash < 256 {
                    let (cm_piece, cm_to_x, cm_to_y) =
                        searcher.countermoves[prev_from_hash][prev_to_hash];
                    if cm_piece != 0
                        && cm_piece == m.piece.piece_type() as u8
                        && cm_to_x == m.to.x as i16
                        && cm_to_y == m.to.y as i16
                    {
                        score += sort_countermove();
                    }
                }
            }

            score += DEFAULT_SORT_QUIET;

            // Main history heuristic
            let idx = hash_move_dest(m);
            score += searcher.history[m.piece.piece_type() as usize][idx];

            // Continuation history
            let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
            let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

            for &plies_ago in &[0usize, 1, 2, 3, 5] {
                if ply > plies_ago
                    && let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1]
                {
                    let prev_piece = searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                    if prev_piece < 16 {
                        let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                        score += searcher.cont_history[prev_piece][prev_to_hash][cur_from_hash]
                            [cur_to_hash];
                    }
                }
            }

            // Low-ply history bonus:
            if ply < LOW_PLY_HISTORY_SIZE {
                let move_hash = hash_move_for_lowply(m);
                score += searcher.low_ply_history[ply][move_hash] / 4;
            }
        }
    }

    score
}

/// Sort all moves using selection sort approach.
/// This is equivalent to calling pick_best_move repeatedly but more efficient
/// when we need all moves sorted (e.g., at root).
pub fn sort_moves(
    searcher: &Searcher,
    game: &GameState,
    moves: &mut MoveList,
    ply: usize,
    tt_move: &Option<Move>,
) {
    // For small move lists, use selection sort (faster than allocation overhead)
    if moves.len() <= 32 {
        // Compute scores inline to avoid allocation
        let mut scores = [0i32; 256];
        for (i, m) in moves.iter().enumerate() {
            scores[i] = score_move(searcher, game, m, ply, tt_move);
        }

        // Selection sort - find best for each position
        for i in 0..moves.len().saturating_sub(1) {
            let mut best_idx = i;
            let mut best_score = scores[i];

            for j in (i + 1)..moves.len() {
                if scores[j] > best_score {
                    best_score = scores[j];
                    best_idx = j;
                }
            }

            if best_idx != i {
                moves.swap(i, best_idx);
                scores.swap(i, best_idx);
            }
        }
    } else {
        // For larger lists, use cached key sort (allocation amortized)
        moves.sort_by_cached_key(|m| -score_move(searcher, game, m, ply, tt_move));
    }
}

/// Sort moves at root - always full sort since we examine all moves
pub fn sort_moves_root(
    searcher: &Searcher,
    game: &GameState,
    moves: &mut MoveList,
    tt_move: &Option<Move>,
) {
    sort_moves(searcher, game, moves, 0, tt_move);
}

/// Fast capture sorting using MVV-LVA only (no SEE for qsearch captures)
pub fn sort_captures(game: &GameState, moves: &mut MoveList) {
    // For captures, use selection sort since qsearch usually has few captures
    if moves.len() <= 16 {
        let mut scores = [0i32; 128];
        for (i, m) in moves.iter().enumerate() {
            if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
                scores[i] = get_piece_value(target.piece_type()) * 10
                    - get_piece_value(m.piece.piece_type());
            }
        }

        for i in 0..moves.len().saturating_sub(1) {
            let mut best_idx = i;
            let mut best_score = scores[i];

            for j in (i + 1)..moves.len() {
                if scores[j] > best_score {
                    best_score = scores[j];
                    best_idx = j;
                }
            }

            if best_idx != i {
                moves.swap(i, best_idx);
                scores.swap(i, best_idx);
            }
        }
    } else {
        moves.sort_by_cached_key(|m| {
            if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
                -(get_piece_value(target.piece_type()) * 10 - get_piece_value(m.piece.piece_type()))
            } else {
                0
            }
        });
    }
}

/// Hash move destination to 256-size index (for main history)
#[inline]
pub fn hash_move_dest(m: &Move) -> usize {
    ((m.to.x.wrapping_abs() ^ m.to.y.wrapping_abs()) & 0xFF) as usize
}

/// Hash move source to 256-size index
#[inline]
pub fn hash_move_from(m: &Move) -> usize {
    ((m.from.x.wrapping_abs() ^ m.from.y.wrapping_abs()) & 0xFF) as usize
}

/// Hash coordinate to 32-size index (for continuation history)
#[inline]
pub fn hash_coord_32(x: i64, y: i64) -> usize {
    let ux = x.wrapping_abs() as u64;
    let uy = y.wrapping_abs() as u64;
    ((ux ^ uy) & 0x1F) as usize
}

/// Hash move for low-ply history table (1024 entries)
#[inline]
pub fn hash_move_for_lowply(m: &Move) -> usize {
    let piece = m.piece.piece_type() as u64;
    let from_hash =
        (m.from.x.wrapping_abs() as u64) ^ (m.from.y.wrapping_abs() as u64).rotate_left(8);
    let to_hash = (m.to.x.wrapping_abs() as u64) ^ (m.to.y.wrapping_abs() as u64).rotate_left(16);
    ((piece ^ from_hash ^ to_hash) & LOW_PLY_HISTORY_MASK as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
    use crate::game::GameState;
    use crate::moves::Move;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();
        game
    }

    #[test]
    fn test_hash_move_dest() {
        let m = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 5),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        let hash = hash_move_dest(&m);
        assert!(hash < 256);
    }

    #[test]
    fn test_hash_move_dest_negative() {
        let m = Move::new(
            Coordinate::new(-100, -200),
            Coordinate::new(-50, -150),
            Piece::new(PieceType::Queen, PlayerColor::Black),
        );
        let hash = hash_move_dest(&m);
        assert!(hash < 256);
    }

    #[test]
    fn test_hash_coord_32() {
        let hash = hash_coord_32(1000, -2000);
        assert!(hash < 32);
    }

    #[test]
    fn test_hash_move_for_lowply() {
        let m = Move::new(
            Coordinate::new(4, 2),
            Coordinate::new(4, 4),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        let hash = hash_move_for_lowply(&m);
        assert!(hash < LOW_PLY_HISTORY_ENTRIES);
    }

    #[test]
    fn test_sort_captures_mvv_lva() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Queen, PlayerColor::Black));
        game.board
            .set_piece(5, 5, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.board
            .set_piece(0, 0, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board.rebuild_tiles();

        let mut moves = vec![
            Move::new(
                Coordinate::new(1, 1),
                Coordinate::new(5, 5),
                Piece::new(PieceType::Pawn, PlayerColor::White),
            ),
            Move::new(
                Coordinate::new(0, 0),
                Coordinate::new(4, 4),
                Piece::new(PieceType::Knight, PlayerColor::White),
            ),
        ];

        sort_captures(&game, &mut moves);

        assert_eq!(moves[0].to.x, 4);
        assert_eq!(moves[0].to.y, 4);
    }
}
