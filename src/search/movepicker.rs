//! TRUE Staged move generation with optimized pruning and perfect ordering.
//!
//! Stages:
//! 1. TT Move (tried immediately, no generation)
//! 2. Captures (generated via get_quiescence_captures, sorted by MVV/LVA + SEE)
//! 3. Killers (tried if quiet and pseudo-legal)
//! 4. Quiets (generated via get_quiet_moves_into, sorted by History + Check bonus)
//! 5. Bad Captures (SEE < 0, tried last)
//!
//! This implementation avoids generating quiet moves if a cutoff occurs in TT or Captures.
//! It also reuses the exact move ordering heuristics from the original sort_moves.

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{get_quiescence_captures, Move};
use crate::search::ordering::{hash_coord_32, hash_move_dest};
use crate::search::params::{
    see_winning_threshold, sort_countermove, sort_gives_check, sort_winning_capture,
    DEFAULT_SORT_LOSING_CAPTURE, DEFAULT_SORT_QUIET,
};
use crate::search::{static_exchange_eval, Searcher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    GenerateQuiets,
    Quiets,
    BadCaptures,
    Done,
}

pub struct MovePicker {
    stage: Stage,

    // TT move
    tt_move: Option<Move>,
    tt_is_capture: bool,
    tt_returned: bool,

    // Killers
    killer1: Option<Move>,
    killer2: Option<Move>,
    killer1_valid: bool,
    killer2_valid: bool,

    // Moves buffers
    captures: Vec<Move>,
    captures_idx: usize,
    bad_captures_start: usize,

    quiets: Vec<Move>,
    quiets_idx: usize,

    // Context
    ply: usize,
    enemy_king: Option<Coordinate>,
}

impl MovePicker {
    pub fn new(
        game: &GameState,
        tt_move: Option<Move>,
        killers: &[Option<Move>; 2],
        ply: usize,
        _searcher: &Searcher,
    ) -> Self {
        let tt_is_capture = if let Some(ref m) = tt_move {
            Self::is_capture_static(game, m)
        } else {
            false
        };

        let enemy_king = match game.turn {
            PlayerColor::White => game.black_king_pos,
            PlayerColor::Black => game.white_king_pos,
            PlayerColor::Neutral => None,
        };

        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            tt_is_capture,
            tt_returned: false,
            killer1: killers[0].clone(),
            killer2: killers[1].clone(),
            killer1_valid: false, // validated later
            killer2_valid: false,
            captures: Vec::new(),
            captures_idx: 0,
            bad_captures_start: 0,
            quiets: Vec::new(),
            quiets_idx: 0,
            ply,
            enemy_king,
        }
    }

    #[inline]
    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    if let Some(ref ttm) = self.tt_move {
                        if self.is_pseudo_legal(game, ttm) {
                            self.tt_returned = true;
                            return Some(ttm.clone());
                        }
                    }
                }

                Stage::GenerateCaptures => {
                    self.generate_captures(game, searcher);
                    self.stage = Stage::GoodCaptures;
                }

                Stage::GoodCaptures => {
                    while self.captures_idx < self.bad_captures_start {
                        let idx = self.captures_idx;
                        self.captures_idx += 1;
                        let m = &self.captures[idx];

                        // Filter TT move
                        if self.tt_returned
                            && self.tt_is_capture
                            && self.is_same_move(m, self.tt_move.as_ref())
                        {
                            continue;
                        }

                        return Some(m.clone());
                    }
                    self.stage = Stage::Killer1;
                }

                Stage::Killer1 => {
                    self.stage = Stage::Killer2;
                    if let Some(ref k1) = self.killer1.clone() {
                        // Killers must be quiet and pseudo-legal
                        if !Self::is_capture_static(game, k1) && self.is_pseudo_legal(game, k1) {
                            // Check overlap with TT (if TT was quiet)
                            if !self.tt_returned
                                || self.tt_is_capture
                                || !self.is_same_move(k1, self.tt_move.as_ref())
                            {
                                self.killer1_valid = true;
                                return Some(k1.clone());
                            }
                        }
                    }
                }

                Stage::Killer2 => {
                    self.stage = Stage::GenerateQuiets;
                    if let Some(ref k2) = self.killer2.clone() {
                        if !Self::is_capture_static(game, k2) && self.is_pseudo_legal(game, k2) {
                            // Check overlap with TT and Killer1
                            // TT check
                            let overlap_tt = self.tt_returned
                                && !self.tt_is_capture
                                && self.is_same_move(k2, self.tt_move.as_ref());
                            // Killer1 check
                            let overlap_k1 =
                                self.killer1_valid && self.is_same_move(k2, self.killer1.as_ref());

                            if !overlap_tt && !overlap_k1 {
                                self.killer2_valid = true;
                                return Some(k2.clone());
                            }
                        }
                    }
                }

                Stage::GenerateQuiets => {
                    self.generate_quiets(game, searcher);
                    self.stage = Stage::Quiets;
                }

                Stage::Quiets => {
                    while self.quiets_idx < self.quiets.len() {
                        let idx = self.quiets_idx;
                        self.quiets_idx += 1;
                        let m = &self.quiets[idx];

                        // Filter TT
                        if self.tt_returned
                            && !self.tt_is_capture
                            && self.is_same_move(m, self.tt_move.as_ref())
                        {
                            continue;
                        }
                        // Filter Killers
                        if self.killer1_valid && self.is_same_move(m, self.killer1.as_ref()) {
                            continue;
                        }
                        if self.killer2_valid && self.is_same_move(m, self.killer2.as_ref()) {
                            continue;
                        }

                        return Some(m.clone());
                    }
                    self.stage = Stage::BadCaptures;
                }

                Stage::BadCaptures => {
                    while self.captures_idx < self.captures.len() {
                        let idx = self.captures_idx;
                        self.captures_idx += 1;
                        let m = &self.captures[idx];

                        // Filter TT
                        if self.tt_returned
                            && self.tt_is_capture
                            && self.is_same_move(m, self.tt_move.as_ref())
                        {
                            continue;
                        }

                        return Some(m.clone());
                    }
                    self.stage = Stage::Done;
                }

                Stage::Done => {
                    return None;
                }
            }
        }
    }

    fn generate_captures(&mut self, game: &GameState, searcher: &Searcher) {
        // Use optimized capture generator
        get_quiescence_captures(
            &game.board,
            game.turn,
            &game.special_rights,
            &game.en_passant,
            &game.game_rules,
            &game.spatial_indices,
            &mut self.captures,
        );

        // Score captures
        let mut scored: Vec<(Move, i32, i32)> = self
            .captures
            .drain(..)
            .map(|m| {
                let mut score: i32 = 0;

                // MVV-LVA
                let target_val = game
                    .board
                    .get_piece(&m.to.x, &m.to.y)
                    .map(|p| get_piece_value(p.piece_type()))
                    .unwrap_or(100); // En passant
                let attacker_val = get_piece_value(m.piece.piece_type());
                score += target_val * 10 - attacker_val;

                // Capture History
                if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
                    let cap_hist = searcher.capture_history[m.piece.piece_type() as usize]
                        [target.piece_type() as usize];
                    score += cap_hist / 10;
                }

                // SEE
                let see_val = static_exchange_eval(game, &m);
                if see_val >= see_winning_threshold() {
                    score += sort_winning_capture();
                } else {
                    score += DEFAULT_SORT_LOSING_CAPTURE;
                }

                // Check Bonus
                if let Some(ref ek) = self.enemy_king {
                    if Self::move_gives_check(game, &m, ek) {
                        score += sort_gives_check();
                    }
                }

                (m, score, see_val)
            })
            .collect();

        // Sort
        scored.sort_by(|a, b| b.1.cmp(&a.1));

        // Partition bad captures (SEE < 0)
        // We actually used see_threshold logic above for scoring.
        // True "bad" captures are those with SEE < 0.
        // We'll partition strictly on SEE < 0 for the "BadCaptures" stage.
        // The sorted list puts high scores first. Bad captures have DEFAULT_SORT_LOSING_CAPTURE (low).
        // But some losing captures might be sorted higher than others?
        // Let's find the split point where SEE < 0 starts?
        // No, mixed scores. We must re-order or just flag them.
        // Easiest is to keep them sorted by score, but conceptually "bad" ones are searched later.
        // However, standard staged search usually puts SEE < 0 captures strictly last.
        // So we partition: Good (SEE >= 0) vs Bad (SEE < 0).
        // And Sort each partition.

        let (mut good, mut bad): (Vec<_>, Vec<_>) =
            scored.into_iter().partition(|(_, _, see)| *see >= 0);

        good.sort_by(|a, b| b.1.cmp(&a.1));
        bad.sort_by(|a, b| b.1.cmp(&a.1));

        self.bad_captures_start = good.len();
        self.captures = good
            .into_iter()
            .map(|x| x.0)
            .chain(bad.into_iter().map(|x| x.0))
            .collect();
    }

    fn generate_quiets(&mut self, game: &GameState, searcher: &Searcher) {
        // Use optimized quiet generator
        game.get_quiet_moves_into(&mut self.quiets, false);

        // Score quiets
        let (prev_from, prev_to) = if self.ply > 0 {
            searcher.prev_move_stack[self.ply - 1]
        } else {
            (0, 0)
        };

        let mut scored: Vec<(Move, i32)> = self
            .quiets
            .drain(..)
            .map(|m| {
                let mut score = DEFAULT_SORT_QUIET;

                // Check Bonus
                if let Some(ref ek) = self.enemy_king {
                    if Self::move_gives_check(game, &m, ek) {
                        score += sort_gives_check();
                    }
                }

                // Countermove
                if self.ply > 0 && prev_from < 256 && prev_to < 256 {
                    let (cm_piece, cm_to_x, cm_to_y) = searcher.countermoves[prev_from][prev_to];
                    if cm_piece != 0
                        && cm_piece == m.piece.piece_type() as u8
                        && cm_to_x == m.to.x as i16
                        && cm_to_y == m.to.y as i16
                    {
                        score += sort_countermove();
                    }
                }

                // History
                let idx = hash_move_dest(&m);
                score += searcher.history[m.piece.piece_type() as usize][idx];

                // Cont History
                let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
                let cur_to_hash = hash_coord_32(m.to.x, m.to.y);
                for &plies_ago in &[0usize, 1, 3] {
                    if self.ply >= plies_ago + 1 {
                        if let Some(ref prev_move) = searcher.move_history[self.ply - plies_ago - 1]
                        {
                            let prev_piece =
                                searcher.moved_piece_history[self.ply - plies_ago - 1] as usize;
                            if prev_piece < 16 {
                                let prev_to_hash = hash_coord_32(prev_move.to.x, prev_move.to.y);
                                score += searcher.cont_history[prev_piece][prev_to_hash]
                                    [cur_from_hash][cur_to_hash];
                            }
                        }
                    }
                }

                (m, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        self.quiets = scored.into_iter().map(|x| x.0).collect();
    }

    // Helpers
    #[inline]
    fn is_capture_static(game: &GameState, m: &Move) -> bool {
        game.board
            .get_piece(&m.to.x, &m.to.y)
            .map_or(false, |p| !p.piece_type().is_neutral_type())
            || Self::is_en_passant_static(game, m)
    }

    #[inline]
    fn is_en_passant_static(game: &GameState, m: &Move) -> bool {
        if m.piece.piece_type() != PieceType::Pawn {
            return false;
        }
        game.en_passant
            .as_ref()
            .map_or(false, |ep| m.to.x == ep.square.x && m.to.y == ep.square.y)
    }

    #[inline]
    fn is_same_move(&self, a: &Move, b: Option<&Move>) -> bool {
        match b {
            Some(b) => a.from == b.from && a.to == b.to && a.promotion == b.promotion,
            None => false,
        }
    }

    #[inline]
    fn is_pseudo_legal(&self, game: &GameState, m: &Move) -> bool {
        let piece = match game.board.get_piece(&m.from.x, &m.from.y) {
            Some(p) => p,
            None => return false,
        };
        if piece.color() != game.turn || piece.piece_type() != m.piece.piece_type() {
            return false;
        }
        // Basic destination check
        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            if target.color() == game.turn {
                return false;
            }
        }
        true
    }

    // Copied from ordering.rs to ensure self-contained correctness
    #[inline]
    fn move_gives_check(game: &GameState, m: &Move, enemy_king: &Coordinate) -> bool {
        let to = &m.to;
        let kx = enemy_king.x;
        let ky = enemy_king.y;

        // Discovered checks are not handled here (requires board update), purely direct checks

        let piece_type = m.promotion.unwrap_or(m.piece.piece_type());
        match piece_type {
            PieceType::Pawn => {
                let dir = if m.piece.color() == PlayerColor::White {
                    1
                } else {
                    -1
                };
                let dy = ky - to.y;
                let dx = (kx - to.x).abs();
                dy == dir && dx == 1
            }
            PieceType::Knight => {
                let dx = (kx - to.x).abs();
                let dy = (ky - to.y).abs();
                (dx == 1 && dy == 2) || (dx == 2 && dy == 1)
            }
            // ... For other pieces, we check ray alignment ...
            PieceType::Rook => kx == to.x || ky == to.y,
            PieceType::Bishop => (kx - to.x).abs() == (ky - to.y).abs(),
            PieceType::Queen => {
                (kx == to.x || ky == to.y) || ((kx - to.x).abs() == (ky - to.y).abs())
            }
            // Simplified check: if it aligns, assume check
            _ => false,
        }
    }
}
