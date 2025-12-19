//! Stockfish-style staged move generation for efficient alpha-beta search.
//!
//! Implements exact Stockfish movepick.cpp pattern:
//! 1. TT move (hash move) - highest priority
//! 2. CAPTURE_INIT: Generate + score + sort_unstable_by
//! 3. GOOD_CAPTURE: Select with SEE filter, bad captures collected
//! 4. QUIET_INIT: Generate + score + sort_unstable_by (skip if skipQuiets)
//! 5. GOOD_QUIET: Select with score > goodQuietThreshold
//! 6. BAD_CAPTURE: Iterate collected bad captures
//! 7. BAD_QUIET: Select with score <= goodQuietThreshold

use super::params::{
    DEFAULT_SORT_QUIET, see_winning_threshold, sort_countermove, sort_gives_check, sort_killer1,
    sort_killer2,
};
use super::{Searcher, hash_coord_32, hash_move_dest, static_exchange_eval};
use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, MoveList, get_quiescence_captures, get_quiet_moves_into};

/// Good quiet threshold (original -4000, not Stockfish's -14000)
const GOOD_QUIET_THRESHOLD: i32 = -4000;

/// Stages of move generation (hybrid: Stockfish optimizations + trusted killer stages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveStage {
    TTMove,
    CaptureInit,
    GoodCapture,
    Killer1,
    Killer2,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,
    Done,
}

/// Move with score for sorting
#[derive(Clone)]
struct ScoredMove {
    m: Move,
    score: i32,
}

/// Staged move generator with unified buffer and sort_unstable_by.
pub struct StagedMoveGen {
    stage: MoveStage,

    // TT move
    tt_move: Option<Move>,

    // Unified move buffer (like Stockfish's moves[])
    moves: Vec<ScoredMove>,
    cur: usize,              // Current position in buffer
    end_bad_captures: usize, // End of bad captures section
    end_captures: usize,     // End of captures section
    end_generated: usize,    // End of generated moves

    // Ply for scoring
    ply: usize,

    // Previous move info for countermove lookup
    prev_from_hash: usize,
    prev_to_hash: usize,

    // Cached enemy king for check detection
    enemy_king: Option<Coordinate>,

    // Killers (checked during quiet iteration)
    killer1: Option<Move>,
    killer2: Option<Move>,

    // Skip quiets flag (Stockfish-style LMP)
    skip_quiets: bool,

    // Excluded move (for singular extension)
    excluded_move: Option<Move>,
}

/// Sort scored moves by score descending (highest first).
/// Uses sort_unstable_by which is O(n log n) and faster than stable sort.
#[inline]
fn sort_moves_by_score(moves: &mut [ScoredMove]) {
    moves.sort_unstable_by(|a, b| b.score.cmp(&a.score));
}

impl StagedMoveGen {
    pub fn new(tt_move: Option<Move>, ply: usize, searcher: &Searcher, game: &GameState) -> Self {
        // Get previous move info for countermove lookup
        let (prev_from_hash, prev_to_hash) = if ply > 0 {
            searcher.prev_move_stack[ply - 1]
        } else {
            (0, 0)
        };

        // Find enemy king for check detection
        let enemy_king = match game.turn {
            PlayerColor::White => game.black_king_pos,
            PlayerColor::Black => game.white_king_pos,
            PlayerColor::Neutral => None,
        };

        // Get killers
        let killer1 = searcher.killers[ply][0].clone();
        let killer2 = searcher.killers[ply][1].clone();

        Self {
            stage: MoveStage::TTMove,
            tt_move,
            moves: Vec::with_capacity(96),
            cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            ply,
            prev_from_hash,
            prev_to_hash,
            enemy_king,
            killer1,
            killer2,
            skip_quiets: false,
            excluded_move: None,
        }
    }

    /// Create with exclusion for singular extension.
    pub fn with_exclusion(
        tt_move: Option<Move>,
        ply: usize,
        searcher: &Searcher,
        game: &GameState,
        excluded: Move,
    ) -> Self {
        let mut r#gen = Self::new(tt_move, ply, searcher, game);
        r#gen.excluded_move = Some(excluded);
        r#gen
    }

    /// Signal to skip quiet moves entirely (Stockfish-style LMP).
    #[inline]
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    /// Check if a move matches another
    #[inline]
    fn moves_match(a: &Move, b: &Option<Move>) -> bool {
        match b {
            Some(bm) => a.from == bm.from && a.to == bm.to && a.promotion == bm.promotion,
            None => false,
        }
    }

    /// Check if move is excluded
    #[inline]
    fn is_excluded(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.excluded_move)
    }

    /// Check if move is TT move
    #[inline]
    fn is_tt_move(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.tt_move)
    }

    /// Check if move is a capture (BITBOARD: O(1) bit check)
    #[inline]
    fn is_capture(game: &GameState, m: &Move) -> bool {
        game.board.is_occupied(m.to.x, m.to.y)
    }
    #[inline]
    fn is_pseudo_legal(game: &GameState, m: &Move) -> bool {
        // BITBOARD: Fast piece check using tile array
        if let Some(piece) = game.board.get_piece_fast(m.from.x, m.from.y) {
            if piece.color() != game.turn || piece.piece_type() != m.piece.piece_type() {
                return false;
            }

            // Castling check
            if piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1 {
                if let Some(rook_coord) = &m.rook_coord {
                    // BITBOARD: Check rook exists and is our color
                    if !game
                        .board
                        .is_occupied_by_color(rook_coord.x, rook_coord.y, game.turn)
                    {
                        return false;
                    }
                    let dir = if m.to.x > m.from.x { 1 } else { -1 };
                    // BITBOARD: Fast occupancy checks for castling path
                    if game.board.is_occupied(m.from.x + dir, m.from.y) {
                        return false;
                    }
                    if game.board.is_occupied(m.to.x, m.from.y) {
                        return false;
                    }
                    if dir < 0 && game.board.is_occupied(m.from.x - 3, m.from.y) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Score capture move (BITBOARD: uses fast piece retrieval)
    fn score_capture(game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if let Some(target) = game.board.get_piece_fast(m.to.x, m.to.y) {
            let victim_val = get_piece_value(target.piece_type());
            let cap_hist = searcher.capture_history[m.piece.piece_type() as usize]
                [target.piece_type() as usize];
            cap_hist + 7 * victim_val
        } else {
            0
        }
    }

    /// Score quiet move (Stockfish formula with histories)
    fn score_quiet(&self, game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        let mut score: i32 = DEFAULT_SORT_QUIET;
        let ply = self.ply;

        // Killer bonus
        if Self::moves_match(m, &self.killer1) {
            return sort_killer1();
        }
        if Self::moves_match(m, &self.killer2) {
            return sort_killer2();
        }

        // Check bonus
        if let Some(ref ek) = self.enemy_king {
            if Self::move_gives_check_simple(game, m, ek) {
                score += sort_gives_check();
            }
        }

        // Countermove bonus
        if self.ply > 0 && self.prev_from_hash < 256 && self.prev_to_hash < 256 {
            let (cm_piece, cm_to_x, cm_to_y) =
                searcher.countermoves[self.prev_from_hash][self.prev_to_hash];
            if cm_piece != 0
                && cm_piece == m.piece.piece_type() as u8
                && cm_to_x == m.to.x as i16
                && cm_to_y == m.to.y as i16
            {
                score += sort_countermove();
            }
        }

        // Main history (2x weight like Stockfish)
        let idx = hash_move_dest(m);
        score += 2 * searcher.history[m.piece.piece_type() as usize][idx];

        // Continuation history
        let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
        let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

        for &plies_ago in &[0usize, 1, 3] {
            if ply >= plies_ago + 1 {
                if let Some(ref prev_move) = searcher.move_history[ply - plies_ago - 1] {
                    let prev_piece = searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                    if prev_piece < 16 {
                        let prev_to_h = hash_coord_32(prev_move.to.x, prev_move.to.y);
                        score += searcher.cont_history[prev_piece][prev_to_h][cur_from_hash]
                            [cur_to_hash];
                    }
                }
            }
        }

        score
    }

    /// Simple check detection
    #[inline]
    pub fn move_gives_check_simple(_game: &GameState, m: &Move, enemy_king: &Coordinate) -> bool {
        let to = &m.to;
        let dx = (enemy_king.x - to.x).abs();
        let dy = (enemy_king.y - to.y).abs();

        match m.piece.piece_type() {
            PieceType::Queen => dx == 0 || dy == 0 || dx == dy,
            PieceType::Rook => dx == 0 || dy == 0,
            PieceType::Bishop => dx == dy,
            PieceType::Knight => (dx == 1 && dy == 2) || (dx == 2 && dy == 1),
            PieceType::Pawn => {
                let direction = if m.piece.color() == PlayerColor::White {
                    1
                } else {
                    -1
                };
                dy == 1 && (to.y - enemy_king.y) == -direction && dx == 1
            }
            _ => false,
        }
    }

    /// Get next move (Stockfish-style next_move())
    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                MoveStage::TTMove => {
                    self.stage = MoveStage::CaptureInit;
                    if let Some(ref m) = self.tt_move {
                        if !self.is_excluded(m) && Self::is_pseudo_legal(game, m) {
                            return Some(m.clone());
                        }
                    }
                }

                MoveStage::CaptureInit => {
                    // Generate all captures
                    let mut captures: MoveList = MoveList::new();
                    get_quiescence_captures(
                        &game.board,
                        game.turn,
                        &game.special_rights,
                        &game.en_passant,
                        &game.game_rules,
                        &game.spatial_indices,
                        &mut captures,
                    );

                    // Score captures
                    for m in captures {
                        if self.is_tt_move(&m) || self.is_excluded(&m) {
                            continue;
                        }
                        let score = Self::score_capture(game, searcher, &m);
                        self.moves.push(ScoredMove { m, score });
                    }

                    self.end_captures = self.moves.len();
                    self.end_bad_captures = 0;
                    self.cur = 0;

                    // Full sort for captures (usually small number)
                    if !self.moves.is_empty() {
                        sort_moves_by_score(&mut self.moves[..self.end_captures]);
                    }

                    self.stage = MoveStage::GoodCapture;
                }

                MoveStage::GoodCapture => {
                    // Select captures with SEE filter
                    while self.cur < self.end_captures {
                        let sm = &self.moves[self.cur];
                        // Use original fixed threshold, not Stockfish's -score/18
                        if static_exchange_eval(game, &sm.m) >= see_winning_threshold() {
                            let m = self.moves[self.cur].m.clone();
                            self.cur += 1;
                            return Some(m);
                        } else {
                            // Move bad capture to end_bad_captures section
                            // We'll iterate them later
                            self.end_bad_captures += 1;
                        }
                        self.cur += 1;
                    }
                    self.stage = MoveStage::Killer1;
                }

                MoveStage::Killer1 => {
                    self.stage = MoveStage::Killer2;
                    // Skip killers if LMP is active (killers are quiet moves)
                    if self.skip_quiets {
                        continue;
                    }
                    if let Some(ref k) = self.killer1 {
                        // Killer must be: not TT move, not a capture, pseudo-legal
                        if !self.is_tt_move(k)
                            && !self.is_excluded(k)
                            && !Self::is_capture(game, k)
                            && Self::is_pseudo_legal(game, k)
                        {
                            return Some(k.clone());
                        }
                    }
                }

                MoveStage::Killer2 => {
                    self.stage = MoveStage::QuietInit;
                    // Skip killers if LMP is active
                    if self.skip_quiets {
                        continue;
                    }
                    if let Some(ref k) = self.killer2 {
                        if !self.is_tt_move(k)
                            && !Self::moves_match(k, &self.killer1)
                            && !self.is_excluded(k)
                            && !Self::is_capture(game, k)
                            && Self::is_pseudo_legal(game, k)
                        {
                            return Some(k.clone());
                        }
                    }
                }

                MoveStage::QuietInit => {
                    if self.skip_quiets {
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    // Generate quiets
                    let mut quiets: MoveList = MoveList::new();
                    get_quiet_moves_into(
                        &game.board,
                        game.turn,
                        &game.special_rights,
                        &game.en_passant,
                        &game.game_rules,
                        &game.spatial_indices,
                        &mut quiets,
                    );

                    // Score quiets
                    let quiet_start = self.moves.len();
                    for m in quiets {
                        if self.is_tt_move(&m) || self.is_excluded(&m) {
                            continue;
                        }
                        let score = self.score_quiet(game, searcher, &m);
                        self.moves.push(ScoredMove { m, score });
                    }

                    self.end_generated = self.moves.len();
                    self.cur = quiet_start;

                    // Full sort for quiets (like original)
                    if quiet_start < self.end_generated {
                        sort_moves_by_score(&mut self.moves[quiet_start..self.end_generated]);
                    }

                    self.stage = MoveStage::GoodQuiet;
                }

                MoveStage::GoodQuiet => {
                    if self.skip_quiets {
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    // Select quiets with score > goodQuietThreshold
                    while self.cur < self.end_generated {
                        if self.moves[self.cur].score > GOOD_QUIET_THRESHOLD {
                            let m = self.moves[self.cur].m.clone();
                            self.cur += 1;
                            return Some(m);
                        }
                        self.cur += 1;
                    }

                    // Setup for bad captures
                    self.cur = 0;
                    self.stage = MoveStage::BadCapture;
                }

                MoveStage::BadCapture => {
                    // Iterate through captures that failed SEE
                    while self.cur < self.end_captures {
                        let sm = &self.moves[self.cur];
                        let see_threshold = -sm.score / 18;

                        // This capture failed SEE earlier, return it now
                        if static_exchange_eval(game, &sm.m) < see_threshold {
                            let m = self.moves[self.cur].m.clone();
                            self.cur += 1;
                            return Some(m);
                        }
                        self.cur += 1;
                    }

                    // Setup for bad quiets
                    self.cur = self.end_captures;
                    self.stage = MoveStage::BadQuiet;
                }

                MoveStage::BadQuiet => {
                    if self.skip_quiets {
                        self.stage = MoveStage::Done;
                        return None;
                    }

                    // Select quiets with score <= goodQuietThreshold
                    while self.cur < self.end_generated {
                        if self.moves[self.cur].score <= GOOD_QUIET_THRESHOLD {
                            let m = self.moves[self.cur].m.clone();
                            self.cur += 1;
                            return Some(m);
                        }
                        self.cur += 1;
                    }
                    self.stage = MoveStage::Done;
                }

                MoveStage::Done => {
                    return None;
                }
            }
        }
    }
}
