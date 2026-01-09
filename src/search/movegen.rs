//! Staged move generation for efficient alpha-beta search.
//!
//! Implements a staged move generation pattern:
//! 1. TT move (hash move) - highest priority
//! 2. CAPTURE_INIT: Generate + score + sort
//! 3. GOOD_CAPTURE: Select with SEE filter, bad captures collected
//! 4. QUIET_INIT: Generate + score + sort (skip if skip_quiets)
//! 5. GOOD_QUIET: Select with score > boundary
//! 6. BAD_CAPTURE: Iterate collected bad captures
//! 7. BAD_QUIET: Select with score <= boundary

use super::params::{DEFAULT_SORT_QUIET, sort_countermove, sort_killer1, sort_killer2};
use super::{
    LOW_PLY_HISTORY_MASK, LOW_PLY_HISTORY_SIZE, Searcher, hash_coord_32, hash_move_dest,
    static_exchange_eval,
};
use crate::board::{PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, MoveGenContext, MoveList, get_quiescence_captures, get_quiet_moves_into};

/// Good quiet threshold - controls the boundary between good and bad quiet moves.
const GOOD_QUIET_THRESHOLD: i32 = -14000;

/// Stages of move generation (includes captures, killers, and quiet moves).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveStage {
    TTMove,
    EvasionInit,
    Evasion,
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

    // Unified move buffer
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

    // Killers (checked during quiet iteration)
    killer1: Option<Move>,
    killer2: Option<Move>,

    // Skip quiets flag (for LMP)
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
    pub fn new(tt_move: Option<Move>, ply: usize, searcher: &Searcher, _game: &GameState) -> Self {
        // Get previous move info for countermove lookup
        let (prev_from_hash, prev_to_hash) = if ply > 0 {
            searcher.prev_move_stack[ply - 1]
        } else {
            (0, 0)
        };

        // Get killers
        let killer1 = searcher.killers[ply][0];
        let killer2 = searcher.killers[ply][1];

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

            killer1,
            killer2,
            skip_quiets: false,
            excluded_move: None,
        }
    }

    /// Check if the position is in check and should use evasion stages
    fn is_in_check(game: &GameState) -> bool {
        game.is_in_check() && game.must_escape_check()
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

    /// Signal to skip quiet moves entirely (LMP).
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
        if let Some(piece) = game.board.get_piece(m.from.x, m.from.y) {
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

    /// Score capture move (combines capture history and piece values)
    fn score_capture(game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
            let victim_val = get_piece_value(target.piece_type());
            let cap_hist = searcher.capture_history[m.piece.piece_type() as usize]
                [target.piece_type() as usize];
            cap_hist + 7 * victim_val
        } else {
            0
        }
    }

    /// Score quiet move using multiple history heuristics:
    /// - main history
    /// - continuation history
    /// - check bonus if move gives check and SEE >= -75
    fn score_quiet(&self, game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        let mut score: i32 = DEFAULT_SORT_QUIET;
        let ply = self.ply;

        // Killer bonus - handled separately with special scores
        if Self::moves_match(m, &self.killer1) {
            return sort_killer1();
        }
        if Self::moves_match(m, &self.killer2) {
            return sort_killer2();
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

        // Main history
        let idx = hash_move_dest(m);
        score += 2 * searcher.history[m.piece.piece_type() as usize][idx];

        // Continuation history using multiple plies of history
        let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
        let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

        for &plies_ago in &[0usize, 1, 2, 3, 5] {
            if let Some(prev_move) = ply
                .checked_sub(plies_ago + 1)
                .and_then(|i| searcher.move_history[i])
            {
                let prev_piece = searcher.moved_piece_history[ply - plies_ago - 1] as usize;
                if prev_piece < 16 {
                    let prev_to_h = hash_coord_32(prev_move.to.x, prev_move.to.y);
                    score +=
                        searcher.cont_history[prev_piece][prev_to_h][cur_from_hash][cur_to_hash];
                }
            }
        }

        // Give a bonus for moves that place the king in check
        // if SEE >= -75 (to avoid giving bonus to bad checks)
        // Use O(1) hash lookup for knights/pawns, inline check for sliders
        let gives_check = Self::move_gives_check_fast(game, m);
        if gives_check {
            // Verify the check isn't losing material with SEE
            if super::see_ge(game, m, -75) {
                score += 16384;
            }
        }

        // Low Ply History:
        if ply < LOW_PLY_HISTORY_SIZE {
            let move_hash = hash_move_dest(m) & LOW_PLY_HISTORY_MASK;
            score += 8 * searcher.low_ply_history[ply][move_hash] / (1 + ply as i32);
        }

        score
    }

    /// Ultra-fast check detection using precomputed data and bit operations.
    /// Ultra-fast check detection using precomputed data and bit operations.
    /// Handles core piece types: Knights, Pawns, and Sliders/Compounds.
    #[inline(always)]
    pub fn move_gives_check_fast(game: &GameState, m: &Move) -> bool {
        let pt = m.piece.piece_type();
        let color = m.piece.color();
        let tx = m.to.x;
        let ty = m.to.y;

        // Fast path: Knights and Pawns use O(1) precomputed hash lookup
        if pt == PieceType::Knight || pt == PieceType::Pawn {
            let check_squares = if color == PlayerColor::White {
                &game.check_squares_black
            } else {
                &game.check_squares_white
            };
            return check_squares.contains(&(tx, ty, pt as u8));
        }

        // Get enemy king position
        let king_pos = if color == PlayerColor::White {
            match &game.black_king_pos {
                Some(k) => k,
                None => return false,
            }
        } else {
            match &game.white_king_pos {
                Some(k) => k,
                None => return false,
            }
        };

        let dx = tx - king_pos.x;
        let dy = ty - king_pos.y;
        let adx = dx.abs();
        let ady = dy.abs();

        // Compute piece type bit for O(1) mask checks
        use crate::attacks::{DIAG_MASK, KNIGHT_MASK, ORTHO_MASK};
        let pt_bit = 1u32 << (pt as u8);

        // 1. Knight-like check (including compounds like Amazon, Chancellor, etc.)
        if (pt_bit & KNIGHT_MASK) != 0 && ((adx == 1 && ady == 2) || (adx == 2 && ady == 1)) {
            return true;
        }

        // 2. Orthogonal slider check (including Queen, Rook, Chancellor, etc.)
        if (pt_bit & ORTHO_MASK) != 0 && (dx == 0 || dy == 0) {
            return true;
        }

        // 3. Diagonal slider check (including Queen, Bishop, Archbishop, etc.)
        if (pt_bit & DIAG_MASK) != 0 && (adx == ady && adx > 0) {
            return true;
        }

        false
    }

    /// Score an evasion move using standard heuristics
    fn score_evasion(&self, game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if game.board.is_occupied(m.to.x, m.to.y) {
            // Evasion capture: high priority + capture heuristics
            30_000_000 + Self::score_capture(game, searcher, m)
        } else {
            // Evasion quiet: use score_quiet (includes Killers, History, etc.)
            self.score_quiet(game, searcher, m)
        }
    }

    /// Get next move from the current stage.
    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                MoveStage::TTMove => {
                    if Self::is_in_check(game) {
                        self.stage = MoveStage::EvasionInit;
                    } else {
                        self.stage = MoveStage::CaptureInit;
                    }

                    if let Some(m) = self
                        .tt_move
                        .filter(|m| !self.is_excluded(m) && Self::is_pseudo_legal(game, m))
                    {
                        return Some(m);
                    }
                }

                MoveStage::EvasionInit => {
                    // Generate all legal evasions
                    let mut evasions: MoveList = MoveList::new();
                    game.get_evasion_moves_into(&mut evasions);

                    // Score evasions
                    for m in evasions {
                        if self.is_tt_move(&m) || self.is_excluded(&m) {
                            continue;
                        }
                        let score = self.score_evasion(game, searcher, &m);
                        self.moves.push(ScoredMove { m, score });
                    }

                    if !self.moves.is_empty() {
                        sort_moves_by_score(&mut self.moves);
                    }
                    self.cur = 0;
                    self.stage = MoveStage::Evasion;
                }

                MoveStage::Evasion => {
                    if self.cur < self.moves.len() {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }
                    self.stage = MoveStage::Done;
                }

                MoveStage::CaptureInit => {
                    // Generate all captures
                    let mut captures: MoveList = MoveList::new();
                    let ctx = MoveGenContext {
                        special_rights: &game.special_rights,
                        en_passant: &game.en_passant,
                        game_rules: &game.game_rules,
                        indices: &game.spatial_indices,
                        enemy_king_pos: game.enemy_king_pos(),
                    };
                    get_quiescence_captures(&game.board, game.turn, &ctx, &mut captures);

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
                    // Bad captures are swapped to end_bad_captures region at the front
                    while self.cur < self.end_captures {
                        let see_threshold = -self.moves[self.cur].score / 18;
                        if static_exchange_eval(game, &self.moves[self.cur].m) >= see_threshold {
                            let m = self.moves[self.cur].m;
                            self.cur += 1;
                            return Some(m);
                        } else {
                            // Swap this bad capture to the end_bad_captures position
                            self.moves.swap(self.end_bad_captures, self.cur);
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
                    if let Some(k) = self.killer1.filter(|k| {
                        !self.is_tt_move(k)
                            && !self.is_excluded(k)
                            && !Self::is_capture(game, k)
                            && Self::is_pseudo_legal(game, k)
                    }) {
                        return Some(k);
                    }
                }

                MoveStage::Killer2 => {
                    self.stage = MoveStage::QuietInit;
                    // Skip killers if LMP is active
                    if self.skip_quiets {
                        continue;
                    }
                    if let Some(k) = self.killer2.filter(|k| {
                        !self.is_tt_move(k)
                            && !Self::moves_match(k, &self.killer1)
                            && !self.is_excluded(k)
                            && !Self::is_capture(game, k)
                            && Self::is_pseudo_legal(game, k)
                    }) {
                        return Some(k);
                    }
                }

                MoveStage::QuietInit => {
                    if self.skip_quiets {
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    // Generate quiets
                    let mut quiets: MoveList = MoveList::new();
                    let ctx = MoveGenContext {
                        special_rights: &game.special_rights,
                        en_passant: &game.en_passant,
                        game_rules: &game.game_rules,
                        indices: &game.spatial_indices,
                        enemy_king_pos: game.enemy_king_pos(),
                    };
                    get_quiet_moves_into(&game.board, game.turn, &ctx, &mut quiets);

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
                    if self.cur < self.end_generated {
                        if self.moves[self.cur].score > GOOD_QUIET_THRESHOLD {
                            let m = self.moves[self.cur].m;
                            self.cur += 1;
                            return Some(m);
                        }
                        self.cur += 1;
                        continue; // Go to next quiet or next stage
                    }

                    // Setup for bad captures
                    self.cur = 0;
                    self.stage = MoveStage::BadCapture;
                }

                MoveStage::BadCapture => {
                    // Iterate bad captures (swapped to front during GOOD_CAPTURE)
                    if self.cur < self.end_bad_captures {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
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
                    if self.cur < self.end_generated {
                        if self.moves[self.cur].score <= GOOD_QUIET_THRESHOLD {
                            let m = self.moves[self.cur].m;
                            self.cur += 1;
                            return Some(m);
                        }
                        self.cur += 1;
                        continue; // Go to next quiet or next stage
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
