use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;
use crate::search::static_exchange_eval;

use super::Searcher;

// Zig-style move ordering constants (scaled for our key direction where
// lower scores are searched first).
const SORT_HASH: i32 = 6_000_000;
const SORT_WINNING_CAPTURE: i32 = 1_000_000;
const SORT_LOSING_CAPTURE: i32 = 0;
const SORT_QUIET: i32 = 0;
const SORT_KILLER1: i32 = 900_000;
const SORT_KILLER2: i32 = 800_000;

// Move ordering helpers
pub fn sort_moves(searcher: &Searcher, game: &GameState, moves: &mut Vec<Move>, ply: usize, tt_move: &Option<Move>) {
    moves.sort_by_cached_key(|m| {
        let mut score: i32 = 0;

        // Hash move bonus
        if let Some(ttm) = tt_move {
            if m.from == ttm.from && m.to == ttm.to && m.promotion == ttm.promotion {
                score += SORT_HASH;
            }
        }

        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            // Capture: MVV-LVA + SEE threshold + capture history.
            let victim_val = get_piece_value(target.piece_type);
            let attacker_val = get_piece_value(m.piece.piece_type);
            let mvv_lva = victim_val * 10 - attacker_val;

            // Approximate Zig's see_threshold(pos, move, -90): treat captures
            // that lose more than ~a pawn as "losing" and others as winning.
            let see_gain = static_exchange_eval(game, m);
            let is_winning = see_gain >= -90;

            score += mvv_lva;
            if is_winning {
                score += SORT_WINNING_CAPTURE;
            } else {
                score += SORT_LOSING_CAPTURE;
            }

            let cap_hist = searcher.capture_history[m.piece.piece_type as usize][target.piece_type as usize];
            score += cap_hist / 10;
        } else {
            // Quiet move: killers + history, similar to Zig's ordering.
            if searcher.killers[ply][0].as_ref().map_or(false, |k| m.from == k.from && m.to == k.to && m.promotion == k.promotion) {
                score += SORT_KILLER1;
            } else if searcher.killers[ply][1].as_ref().map_or(false, |k| m.from == k.from && m.to == k.to && m.promotion == k.promotion) {
                score += SORT_KILLER2;
            } else {
                score += SORT_QUIET;
            }

            let idx = hash_move_dest(m);
            let history_score = searcher.history[m.piece.piece_type as usize][idx];
            score += history_score;
        }

        // We sort by ascending key, so negate to get highest-score moves first.
        -score
    });
}

pub fn sort_moves_root(searcher: &Searcher, game: &GameState, moves: &mut Vec<Move>, tt_move: &Option<Move>) {
    sort_moves(searcher, game, moves, 0, tt_move);
}

pub fn sort_captures(game: &GameState, moves: &mut Vec<Move>) {
    moves.sort_by_cached_key(|m| {
        let mut score = 0;
        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            // MVV-LVA: prioritize capturing high value pieces with low value attackers
            score -= get_piece_value(target.piece_type) * 10 - get_piece_value(m.piece.piece_type);
        }
        score
    });
}

#[inline]
pub fn hash_move_dest(m: &Move) -> usize {
    ((m.to.x ^ m.to.y) & 0xFF) as usize
}
