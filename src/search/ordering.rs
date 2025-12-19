use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, MoveList};

use super::Searcher;
use super::params::{
    DEFAULT_SORT_LOSING_CAPTURE, DEFAULT_SORT_QUIET, see_winning_threshold, sort_countermove,
    sort_gives_check, sort_hash, sort_killer1, sort_killer2, sort_winning_capture,
};

/// Find the enemy king's position for the current side to move.
/// Uses cached king positions from GameState for O(1) lookup.
#[inline]
fn find_enemy_king(game: &GameState) -> Option<Coordinate> {
    // We're sorting moves for the player about to move (game.turn).
    // The enemy is the opponent of the current turn.
    match game.turn {
        PlayerColor::White => game.black_king_pos,
        PlayerColor::Black => game.white_king_pos,
        PlayerColor::Neutral => None,
    }
}

/// Check if a move gives check to the enemy king.
/// This is an efficient check that doesn't require making the move.
/// It checks if the piece at 'to' would attack the enemy king position.
///
/// Note: This is a simplified check that handles direct checks but not discovered checks.
/// Discovered checks are rarer and more expensive to detect.
#[inline]
fn move_gives_check(game: &GameState, m: &Move, enemy_king: &Coordinate) -> bool {
    let to = &m.to;
    let kx = enemy_king.x;
    let ky = enemy_king.y;
    let dx = kx - to.x;
    let dy = ky - to.y;

    // The piece that will be on 'to' after the move
    let piece_type = m.promotion.unwrap_or(m.piece.piece_type());

    match piece_type {
        // Pawn: checks on diagonals one square away
        PieceType::Pawn => {
            let dir = if m.piece.color() == PlayerColor::White {
                1
            } else {
                -1
            };
            dy == dir && (dx == 1 || dx == -1)
        }

        // Knight: (1,2) or (2,1) pattern
        PieceType::Knight => {
            let adx = dx.abs();
            let ady = dy.abs();
            (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }

        // Bishop: diagonal alignment + path clear
        PieceType::Bishop => {
            if dx.abs() != dy.abs() || dx == 0 {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Rook: orthogonal alignment + path clear
        PieceType::Rook => {
            if !((dx == 0 && dy != 0) || (dy == 0 && dx != 0)) {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Queen/RoyalQueen: orthogonal or diagonal + path clear
        PieceType::Queen | PieceType::RoyalQueen => {
            let is_ortho = (dx == 0 && dy != 0) || (dy == 0 && dx != 0);
            let is_diag = dx.abs() == dy.abs() && dx != 0;
            if !is_ortho && !is_diag {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Chancellor (Rook + Knight)
        PieceType::Chancellor => {
            // Knight check
            let adx = dx.abs();
            let ady = dy.abs();
            if (adx == 1 && ady == 2) || (adx == 2 && ady == 1) {
                return true;
            }
            // Rook check
            if !((dx == 0 && dy != 0) || (dy == 0 && dx != 0)) {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Archbishop (Bishop + Knight)
        PieceType::Archbishop => {
            // Knight check
            let adx = dx.abs();
            let ady = dy.abs();
            if (adx == 1 && ady == 2) || (adx == 2 && ady == 1) {
                return true;
            }
            // Bishop check
            if dx.abs() != dy.abs() || dx == 0 {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Amazon (Queen + Knight)
        PieceType::Amazon => {
            // Knight check
            let adx = dx.abs();
            let ady = dy.abs();
            if (adx == 1 && ady == 2) || (adx == 2 && ady == 1) {
                return true;
            }
            // Queen check
            let is_ortho = (dx == 0 && dy != 0) || (dy == 0 && dx != 0);
            let is_diag = dx.abs() == dy.abs() && dx != 0;
            if !is_ortho && !is_diag {
                return false;
            }
            path_clear_to_king(game, to, enemy_king, dx.signum(), dy.signum())
        }

        // Centaur/RoyalCentaur (King + Knight): Knight check only (king range is 1)
        PieceType::Centaur | PieceType::RoyalCentaur => {
            let adx = dx.abs();
            let ady = dy.abs();
            (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }

        // Knightrider: extended knight pattern on same ray
        PieceType::Knightrider => {
            // Check if king is on a knight-ray from destination
            // Knight directions: (1,2), (2,1), etc.
            for (ndx, ndy) in &[
                (1i64, 2),
                (2, 1),
                (-1, 2),
                (-2, 1),
                (1, -2),
                (2, -1),
                (-1, -2),
                (-2, -1),
            ] {
                if *ndx == 0 || *ndy == 0 {
                    continue;
                }
                // Check if dx/dy aligns with this knight direction
                if dx % ndx == 0 && dy % ndy == 0 {
                    let kx_steps = dx / ndx;
                    let ky_steps = dy / ndy;
                    if kx_steps == ky_steps && kx_steps > 0 {
                        // King is on this knight-ray, check path is clear
                        let mut clear = true;
                        for step in 1..kx_steps {
                            let cx = to.x + ndx * step;
                            let cy = to.y + ndy * step;
                            // BITBOARD: Fast occupancy check
                            if game.board.is_occupied(cx, cy) {
                                clear = false;
                                break;
                            }
                        }
                        if clear {
                            return true;
                        }
                    }
                }
            }
            false
        }

        // Other leapers: Camel (1,3), Giraffe (1,4), Zebra (2,3), Hawk (2,3 jumps)
        PieceType::Camel => {
            let adx = dx.abs();
            let ady = dy.abs();
            (adx == 1 && ady == 3) || (adx == 3 && ady == 1)
        }
        PieceType::Giraffe => {
            let adx = dx.abs();
            let ady = dy.abs();
            (adx == 1 && ady == 4) || (adx == 4 && ady == 1)
        }
        PieceType::Zebra => {
            let adx = dx.abs();
            let ady = dy.abs();
            (adx == 2 && ady == 3) || (adx == 3 && ady == 2)
        }
        PieceType::Hawk => {
            // Hawk: jumps 2 or 3 squares orthogonally or diagonally
            let adx = dx.abs();
            let ady = dy.abs();
            // Orthogonal: (2,0), (3,0), (0,2), (0,3)
            if (adx == 2 || adx == 3) && ady == 0 {
                return true;
            }
            if (ady == 2 || ady == 3) && adx == 0 {
                return true;
            }
            // Diagonal: (2,2), (3,3)
            if adx == ady && (adx == 2 || adx == 3) {
                return true;
            }
            false
        }

        // King/Guard: can only "check" if adjacent (rare, but possible in some variants)
        PieceType::King | PieceType::Guard => {
            dx.abs() <= 1 && dy.abs() <= 1 && (dx != 0 || dy != 0)
        }

        // Rose, Huygen: complex piece movements, skip for now (rare)
        // These could be added but would need more complex logic
        _ => false,
    }
}

/// Check if the path from 'from' to 'to' is clear (no pieces blocking).
/// step_x and step_y are the direction signs (-1, 0, or 1).
#[inline]
fn path_clear_to_king(
    game: &GameState,
    from: &Coordinate,
    to: &Coordinate,
    step_x: i64,
    step_y: i64,
) -> bool {
    // Limit path checking to avoid infinite loops on infinite boards
    const MAX_PATH_CHECK: i64 = 50;

    let mut x = from.x + step_x;
    let mut y = from.y + step_y;
    let mut steps = 0;

    while (x != to.x || y != to.y) && steps < MAX_PATH_CHECK {
        // BITBOARD: Fast occupancy check
        if game.board.is_occupied(x, y) {
            return false;
        }
        x += step_x;
        y += step_y;
        steps += 1;
    }

    steps < MAX_PATH_CHECK
}

// Move ordering helpers
pub fn sort_moves(
    searcher: &Searcher,
    game: &GameState,
    moves: &mut MoveList,
    ply: usize,
    tt_move: &Option<Move>,
) {
    // Get previous move info for countermove lookup, indexed by hashed
    // from/to squares as in the classic counter-move heuristic.
    let (prev_from_hash, prev_to_hash) = if ply > 0 {
        searcher.prev_move_stack[ply - 1]
    } else {
        (0, 0)
    };

    // Find enemy king once before sorting (avoids repeated lookups per move)
    let enemy_king = find_enemy_king(game);

    moves.sort_by_cached_key(|m| {
        let mut score: i32 = 0;

        // Hash move bonus
        if let Some(ttm) = tt_move {
            if m.from == ttm.from && m.to == ttm.to && m.promotion == ttm.promotion {
                score += sort_hash();
            }
        }

        // BITBOARD: Fast capture detection
        if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
            // Capture: MVV-LVA + SEE threshold + capture history.
            let victim_val = get_piece_value(target.piece_type());
            let attacker_val = get_piece_value(m.piece.piece_type());
            let mvv_lva = victim_val * 10 - attacker_val;

            // Use see_ge with early cutoffs (Stockfish pattern)
            let is_winning = super::see_ge(game, m, see_winning_threshold());

            score += mvv_lva;
            if is_winning {
                score += sort_winning_capture();
            } else {
                score += DEFAULT_SORT_LOSING_CAPTURE;
            }

            let cap_hist = searcher.capture_history[m.piece.piece_type() as usize]
                [target.piece_type() as usize];
            score += cap_hist / 10;

            // Check bonus for captures that also give check (rare but very forcing)
            if let Some(ref ek) = enemy_king {
                if move_gives_check(game, m, ek) {
                    score += sort_gives_check();
                }
            }
        } else {
            // Quiet move: killers + countermove + check bonus + history + continuation history
            if searcher.killers[ply][0].as_ref().map_or(false, |k| {
                m.from == k.from && m.to == k.to && m.promotion == k.promotion
            }) {
                score += sort_killer1();
            } else if searcher.killers[ply][1].as_ref().map_or(false, |k| {
                m.from == k.from && m.to == k.to && m.promotion == k.promotion
            }) {
                score += sort_killer2();
            } else {
                // Check bonus for quiet checks (very important for tactics!)
                // This is applied before countermove/history to prioritize forcing moves
                if let Some(ref ek) = enemy_king {
                    if move_gives_check(game, m, ek) {
                        score += sort_gives_check();
                    }
                }

                // Check if this is the countermove for the previous move
                if ply > 0 && prev_from_hash < 256 && prev_to_hash < 256 {
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
                score += DEFAULT_SORT_QUIET;

                // Main history heuristic
                let idx = hash_move_dest(m);
                score += searcher.history[m.piece.piece_type() as usize][idx];

                // Continuation history: [prev_piece][prev_to][cur_from][cur_to]
                // Use 1-ply, 2-ply, and 4-ply back (like Zig: plies_ago = 0, 1, 3)
                let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
                let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

                for &plies_ago in &[0usize, 1, 3] {
                    if ply >= plies_ago + 1 {
                        if let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1] {
                            let prev_piece =
                                searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                            if prev_piece < 16 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                                score += searcher.cont_history[prev_piece][prev_to_hash]
                                    [cur_from_hash][cur_to_hash];
                            }
                        }
                    }
                }
            }
        }

        // We sort by ascending key, so negate to get highest-score moves first.
        -score
    });
}

pub fn sort_moves_root(
    searcher: &Searcher,
    game: &GameState,
    moves: &mut MoveList,
    tt_move: &Option<Move>,
) {
    sort_moves(searcher, game, moves, 0, tt_move);
}

pub fn sort_captures(game: &GameState, moves: &mut MoveList) {
    moves.sort_by_cached_key(|m| {
        let mut score = 0;
        // BITBOARD: Fast piece lookup
        if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
            // MVV-LVA: prioritize capturing high value pieces with low value attackers
            score -=
                get_piece_value(target.piece_type()) * 10 - get_piece_value(m.piece.piece_type());
        }
        score
    });
}

/// Hash move destination to 256-size index (for main history)
/// Uses wrapping_abs to safely handle negative coordinates in infinite chess
#[inline]
pub fn hash_move_dest(m: &Move) -> usize {
    ((m.to.x.wrapping_abs() ^ m.to.y.wrapping_abs()) & 0xFF) as usize
}

/// Hash move source to 256-size index
#[inline]
pub fn hash_move_from(m: &Move) -> usize {
    // Use wrapping for consistency with infinite coordinates
    ((m.from.x.wrapping_abs() ^ m.from.y.wrapping_abs()) & 0xFF) as usize
}

/// Hash coordinate to 32-size index (for continuation history)
/// Uses wrapping_abs to safely handle negative coordinates in infinite chess
#[inline]
pub fn hash_coord_32(x: i64, y: i64) -> usize {
    // Use wrapping operations to avoid issues with i64::MIN
    let ux = x.wrapping_abs() as u64;
    let uy = y.wrapping_abs() as u64;
    ((ux ^ uy) & 0x1F) as usize
}
