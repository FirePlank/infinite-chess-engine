//! Staged move generation for efficient alpha-beta search.
//!
//! Implements multi-stage move generation:
//!
//! Main Search: MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE → QUIET_INIT →
//!              GOOD_QUIET → BAD_CAPTURE → BAD_QUIET
//!
//! Evasions:    EVASION_TT → EVASION_INIT → EVASION
//!
//! ProbCut:     PROBCUT_TT → PROBCUT_INIT → PROBCUT
//!
//! QSearch:     QSEARCH_TT → QCAPTURE_INIT → QCAPTURE

use super::params::{DEFAULT_SORT_QUIET, sort_countermove, sort_killer1, sort_killer2};
use super::{
    LOW_PLY_HISTORY_MASK, LOW_PLY_HISTORY_SIZE, PAWN_HISTORY_MASK, Searcher, hash_coord_32,
    hash_move_dest,
};
use crate::board::{PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, MoveGenContext, MoveList, get_quiescence_captures, get_quiet_moves_into};

/// Good quiet threshold
const GOOD_QUIET_THRESHOLD: i32 = -14000;

/// Stages of move generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveStage {
    // Main search
    MainTT,
    CaptureInit,
    GoodCapture,
    Killer1,
    Killer2,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,

    // Evasion (when in check)
    EvasionTT,
    EvasionInit,
    Evasion,

    // ProbCut
    ProbCutTT,
    ProbCutInit,
    ProbCut,

    // QSearch
    QSearchTT,
    QCaptureInit,
    QCapture,

    Done,
}

/// Move with score for sorting
#[derive(Clone, Copy)]
struct ScoredMove {
    m: Move,
    score: i32,
}

/// Staged move generator
pub struct StagedMoveGen {
    stage: MoveStage,
    tt_move: Option<Move>,

    // Move buffer
    moves: Vec<ScoredMove>,
    cur: usize,
    end_bad_captures: usize,
    end_captures: usize,
    end_generated: usize,

    // Search parameters
    ply: usize,
    depth: i32,
    threshold: i32, // For ProbCut SEE threshold

    // Previous move info for countermove lookup
    prev_from_hash: usize,
    prev_to_hash: usize,

    // Killers (scored in score_quiet, not separate stages)
    killer1: Option<Move>,
    killer2: Option<Move>,

    // Flags
    skip_quiets: bool,
    excluded_move: Option<Move>,
}

/// Partial insertion sort - sorts moves with score >= limit to the front in descending order.
/// Performs partial insertion sort.
#[inline]
fn partial_insertion_sort(moves: &mut [ScoredMove], limit: i32) {
    let mut sorted_end = 0;
    for i in 0..moves.len() {
        if moves[i].score >= limit {
            let tmp = moves[i];
            moves[i] = moves[sorted_end];

            // Insertion sort into sorted portion
            let mut j = sorted_end;
            while j > 0 && moves[j - 1].score < tmp.score {
                moves[j] = moves[j - 1];
                j -= 1;
            }
            moves[j] = tmp;
            sorted_end += 1;
        }
    }
}

impl StagedMoveGen {
    /// Create MovePicker for main search or quiescence search.
    /// Primary constructor.
    pub fn new(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        searcher: &Searcher,
        game: &GameState,
    ) -> Self {
        let is_in_check = Self::is_in_check(game);
        let tt_valid = tt_move.is_some() && Self::is_pseudo_legal(game, &tt_move.unwrap());

        // Set initial stage based on TT move availability
        // If TT move is valid, start at TT stage; otherwise skip to Init stage
        let start_stage = if is_in_check {
            if tt_valid {
                MoveStage::EvasionTT
            } else {
                MoveStage::EvasionInit
            }
        } else if depth > 0 {
            if tt_valid {
                MoveStage::MainTT
            } else {
                MoveStage::CaptureInit
            }
        } else {
            // QSearch
            if tt_valid {
                MoveStage::QSearchTT
            } else {
                MoveStage::QCaptureInit
            }
        };

        Self::init(tt_move, ply, depth, 0, searcher, start_stage)
    }

    /// Create MovePicker for ProbCut - captures with SEE >= threshold.
    /// Secondary constructor.
    pub fn new_probcut(
        tt_move: Option<Move>,
        threshold: i32,
        searcher: &Searcher,
        game: &GameState,
    ) -> Self {
        debug_assert!(!Self::is_in_check(game), "ProbCut not used when in check");

        // TT move valid only if it's a capture and pseudo-legal
        let tt_valid = tt_move.is_some()
            && Self::is_capture(game, &tt_move.unwrap())
            && Self::is_pseudo_legal(game, &tt_move.unwrap());

        let start_stage = if tt_valid {
            MoveStage::ProbCutTT
        } else {
            MoveStage::ProbCutInit
        };

        Self::init(tt_move, 0, 0, threshold, searcher, start_stage)
    }

    fn init(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        threshold: i32,
        searcher: &Searcher,
        stage: MoveStage,
    ) -> Self {
        let (prev_from_hash, prev_to_hash) = if ply > 0 {
            searcher.prev_move_stack[ply - 1]
        } else {
            (0, 0)
        };

        let killer1 = if ply < searcher.killers.len() {
            searcher.killers[ply][0]
        } else {
            None
        };
        let killer2 = if ply < searcher.killers.len() {
            searcher.killers[ply][1]
        } else {
            None
        };

        Self {
            stage,
            tt_move,
            moves: Vec::new(),
            cur: 0,
            end_bad_captures: 0,
            end_captures: 0,
            end_generated: 0,
            ply,
            depth,
            threshold,
            prev_from_hash,
            prev_to_hash,
            killer1,
            killer2,
            skip_quiets: false,
            excluded_move: None,
        }
    }

    /// Create with exclusion for singular extension
    pub fn with_exclusion(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        searcher: &Searcher,
        game: &GameState,
        excluded: Move,
    ) -> Self {
        let mut generator = Self::new(tt_move, ply, depth, searcher, game);
        generator.excluded_move = Some(excluded);
        generator
    }

    /// Signal to skip quiet moves (LMP)
    #[inline]
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    #[inline]
    fn is_in_check(game: &GameState) -> bool {
        game.is_in_check() && game.must_escape_check()
    }

    #[inline]
    fn moves_match(a: &Move, b: &Option<Move>) -> bool {
        match b {
            Some(bm) => a.from == bm.from && a.to == bm.to && a.promotion == bm.promotion,
            None => false,
        }
    }

    #[inline]
    fn is_excluded(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.excluded_move)
    }

    #[inline]
    fn is_tt_move(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.tt_move)
    }

    #[inline]
    fn is_capture(game: &GameState, m: &Move) -> bool {
        game.board.is_occupied(m.to.x, m.to.y)
    }

    /// Pseudo-legal check - verifies piece exists, correct color/type, and path validation
    #[inline]
    fn is_pseudo_legal(game: &GameState, m: &Move) -> bool {
        use crate::tiles::{local_index, tile_coords};

        // 1. Fast Tile Access
        let (cx, cy) = tile_coords(m.from.x, m.from.y);
        let from_idx = local_index(m.from.x, m.from.y);

        let tile = match game.board.tiles.get_tile(cx, cy) {
            Some(t) => t,
            None => return false,
        };

        // 2. Identity Check
        let packed = tile.piece[from_idx];
        if packed == 0 {
            return false;
        }

        let piece = crate::board::Piece::from_packed(packed);

        if piece.color() != game.turn || piece.piece_type() != m.piece.piece_type() {
            return false;
        }

        // 3. Target Occupancy Check (Generic)
        let (tx, ty) = tile_coords(m.to.x, m.to.y);
        let target_packed = if tx == cx && ty == cy {
            tile.piece[local_index(m.to.x, m.to.y)]
        } else {
            match game.board.tiles.get_tile(tx, ty) {
                Some(t) => t.piece[local_index(m.to.x, m.to.y)],
                None => 0,
            }
        };

        if target_packed != 0 {
            let target = crate::board::Piece::from_packed(target_packed);
            if target.color() == game.turn {
                return false;
            }
        }

        // 4. Piece-Specific Logic
        match piece.piece_type() {
            PieceType::Pawn => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = m.to.y - m.from.y;
                let dir = if game.turn == PlayerColor::White {
                    1
                } else {
                    -1
                };

                if dx == 0 {
                    // Push
                    if dy == dir {
                        target_packed == 0
                    } else if dy == 2 * dir {
                        // 2 steps
                        if target_packed != 0 {
                            return false;
                        }
                        // Check intermediate
                        let mid_y = m.from.y + dir;
                        let (mx, my) = tile_coords(m.from.x, mid_y);
                        let mid_packed = if mx == cx && my == cy {
                            tile.piece[local_index(m.from.x, mid_y)]
                        } else {
                            match game.board.tiles.get_tile(mx, my) {
                                Some(t) => t.piece[local_index(m.from.x, mid_y)],
                                None => 0,
                            }
                        };
                        mid_packed == 0
                    } else {
                        false
                    }
                } else {
                    // Capture
                    if dx == 1 && dy == dir {
                        if target_packed != 0 {
                            return true;
                        }
                        if let Some(ep) = &game.en_passant
                            && ep.square.x == m.to.x
                            && ep.square.y == m.to.y
                        {
                            return true;
                        }
                        return false;
                    }
                    false
                }
            }
            PieceType::Knight => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = (m.to.y - m.from.y).abs();
                (dx == 1 && dy == 2) || (dx == 2 && dy == 1)
            }
            PieceType::King => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = (m.to.y - m.from.y).abs();

                if dx > 1 {
                    if let Some(rook_coord) = &m.rook_coord {
                        if !game
                            .board
                            .is_occupied_by_color(rook_coord.x, rook_coord.y, game.turn)
                        {
                            return false;
                        }
                        let dir = if m.to.x > m.from.x { 1 } else { -1 };
                        if game.board.is_occupied(m.from.x + dir, m.from.y)
                            || game.board.is_occupied(m.to.x, m.from.y)
                            || (dir < 0 && game.board.is_occupied(m.from.x - 3, m.from.y))
                        {
                            return false;
                        }
                        return true;
                    }
                    return false;
                }
                dx <= 1 && dy <= 1
            }
            PieceType::Rook | PieceType::Bishop | PieceType::Queen => {
                let dx = m.to.x - m.from.x;
                let dy = m.to.y - m.from.y;

                let is_ortho = dx == 0 || dy == 0;
                let is_diag = dx.abs() == dy.abs();

                let valid_type = match piece.piece_type() {
                    PieceType::Rook => is_ortho,
                    PieceType::Bishop => is_diag,
                    PieceType::Queen => is_ortho || is_diag,
                    _ => false,
                };
                if !valid_type {
                    return false;
                }

                // Same-Tile Fast Path
                let step_x = dx.signum();
                let step_y = dy.signum();
                if let Some(is_clear) =
                    crate::moves::is_path_clear_locally(&game.board, &m.from, &m.to, step_x, step_y)
                {
                    return is_clear;
                }

                crate::moves::is_piece_attacking_square(
                    &game.board,
                    &piece,
                    &m.from,
                    &m.to,
                    &game.spatial_indices,
                    &game.game_rules,
                )
            }
            _ => crate::moves::is_piece_attacking_square(
                &game.board,
                &piece,
                &m.from,
                &m.to,
                &game.spatial_indices,
                &game.game_rules,
            ),
        }
    }

    /// Score capture: captureHistory + 7 * PieceValue
    fn score_capture(game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
            let victim_val = get_piece_value(target.piece_type());
            let cap_hist = searcher
                .capture_history
                .get(m.piece.piece_type() as usize)
                .and_then(|row| row.get(target.piece_type() as usize))
                .copied()
                .unwrap_or(0);
            cap_hist + 7 * victim_val
        } else {
            0
        }
    }

    /// Score quiet move using history heuristics (includes killer/countermove bonuses)
    fn score_quiet(&self, game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        let mut score: i32 = DEFAULT_SORT_QUIET;
        let ply = self.ply;

        // Killer bonus (integrated into scoring, not separate stages)
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

        // Main history: 2 * mainHistory[us][move]
        let idx = hash_move_dest(m);
        let pt_idx = m.piece.piece_type() as usize;
        if pt_idx < searcher.history.len() {
            score += 2 * searcher.history[pt_idx][idx];
        }

        // Pawn history: 2 * pawnHistory[pawn_hash % SIZE][piece][to]
        let ph_idx = (game.pawn_hash & PAWN_HISTORY_MASK) as usize;
        score += 2 * searcher.pawn_history[ph_idx][pt_idx][idx];

        // Continuation history
        let cur_from_hash = hash_coord_32(m.from.x, m.from.y);
        let cur_to_hash = hash_coord_32(m.to.x, m.to.y);

        for &plies_ago in &[0usize, 1, 2, 3, 4, 5] {
            if let Some(prev_move) = ply
                .checked_sub(plies_ago + 1)
                .and_then(|i| searcher.move_history.get(i).copied().flatten())
                && let Some(&prev_piece) = searcher.moved_piece_history.get(ply - plies_ago - 1)
            {
                let prev_piece = prev_piece as usize;
                if prev_piece < 32 {
                    let prev_to_h = hash_coord_32(prev_move.to.x, prev_move.to.y);
                    let prev_ic = searcher.in_check_history[ply - plies_ago - 1] as usize;
                    let prev_cap = searcher.capture_history_stack[ply - plies_ago - 1] as usize;

                    let val = searcher.cont_history[prev_cap][prev_ic][prev_piece][prev_to_h]
                        [cur_from_hash][cur_to_hash];
                    score += val;
                }
            }
        }

        // Check bonus (if move gives check and SEE >= -75)
        if Self::move_gives_check_fast(game, m) && super::see_ge(game, m, -75) {
            score += 16384;
        }

        // Low ply history
        if ply < LOW_PLY_HISTORY_SIZE {
            let move_hash = hash_move_dest(m) & LOW_PLY_HISTORY_MASK;
            if let Some(val) = searcher
                .low_ply_history
                .get(ply)
                .and_then(|row| row.get(move_hash))
            {
                score += 8 * val / (1 + ply as i32);
            }
        }

        score
    }

    /// Score evasion move
    fn score_evasion(&self, game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if Self::is_capture(game, m) {
            // Capture: PieceValue + (1 << 28)
            let captured_val = game
                .board
                .get_piece(m.to.x, m.to.y)
                .map(|p| get_piece_value(p.piece_type()))
                .unwrap_or(0);
            captured_val + (1 << 28)
        } else {
            // Quiet: use history
            self.score_quiet(game, searcher, m)
        }
    }

    /// Fast check detection
    #[inline(always)]
    pub fn move_gives_check_fast(game: &GameState, m: &Move) -> bool {
        let pt = m.piece.piece_type();
        let color = m.piece.color();
        let tx = m.to.x;
        let ty = m.to.y;

        // Knights and Pawns use precomputed hash lookup
        if pt == PieceType::Knight || pt == PieceType::Pawn {
            let check_squares = if color == PlayerColor::White {
                &game.check_squares_black
            } else {
                &game.check_squares_white
            };
            return check_squares.contains(&(tx, ty, pt as u8));
        }

        // Get enemy king position
        let king_pos = match if color == PlayerColor::White {
            &game.black_king_pos
        } else {
            &game.white_king_pos
        } {
            Some(k) => k,
            None => return false,
        };

        let dx = tx - king_pos.x;
        let dy = ty - king_pos.y;
        let adx = dx.abs();
        let ady = dy.abs();

        use crate::attacks::{DIAG_MASK, KNIGHT_MASK, ORTHO_MASK};
        let pt_bit = 1u32 << (pt as u8);

        // Knight-like check
        if (pt_bit & KNIGHT_MASK) != 0 && ((adx == 1 && ady == 2) || (adx == 2 && ady == 1)) {
            return true;
        }

        // Orthogonal slider check
        if (pt_bit & ORTHO_MASK) != 0 && (dx == 0 || dy == 0) {
            return true;
        }

        // Diagonal slider check
        if (pt_bit & DIAG_MASK) != 0 && adx == ady && adx > 0 {
            return true;
        }

        false
    }

    fn generate_captures(&mut self, game: &GameState, searcher: &Searcher) {
        let mut captures = MoveList::new();
        let ctx = MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: game.enemy_king_pos(),
        };
        get_quiescence_captures(&game.board, game.turn, &ctx, &mut captures);

        for m in captures {
            if self.is_tt_move(&m) || self.is_excluded(&m) {
                continue;
            }
            let score = Self::score_capture(game, searcher, &m);
            self.moves.push(ScoredMove { m, score });
        }
    }

    fn generate_quiets(&mut self, game: &GameState, searcher: &Searcher) {
        let mut quiets = MoveList::new();
        let ctx = MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: game.enemy_king_pos(),
        };
        get_quiet_moves_into(&game.board, game.turn, &ctx, &mut quiets);

        for m in quiets {
            if self.is_tt_move(&m)
                || self.is_excluded(&m)
                || Self::moves_match(&m, &self.killer1)
                || Self::moves_match(&m, &self.killer2)
            {
                continue;
            }
            let score = self.score_quiet(game, searcher, &m);
            self.moves.push(ScoredMove { m, score });
        }
    }

    fn generate_evasions(&mut self, game: &GameState, searcher: &Searcher) {
        let mut evasions = MoveList::new();
        game.get_evasion_moves_into(&mut evasions);

        for m in evasions {
            if self.is_tt_move(&m) || self.is_excluded(&m) {
                continue;
            }
            let score = self.score_evasion(game, searcher, &m);
            self.moves.push(ScoredMove { m, score });
        }
    }

    /// Get next move using multi-stage generation
    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                MoveStage::MainTT
                | MoveStage::EvasionTT
                | MoveStage::QSearchTT
                | MoveStage::ProbCutTT => {
                    // Advance to next stage
                    self.stage = match self.stage {
                        MoveStage::MainTT => MoveStage::CaptureInit,
                        MoveStage::EvasionTT => MoveStage::EvasionInit,
                        MoveStage::QSearchTT => MoveStage::QCaptureInit,
                        MoveStage::ProbCutTT => MoveStage::ProbCutInit,
                        _ => unreachable!(),
                    };

                    // Return TT move (already validated in constructor)
                    if let Some(tt_m) = self.tt_move
                        && !self.is_excluded(&tt_m)
                    {
                        return Some(tt_m);
                    }
                }

                MoveStage::CaptureInit | MoveStage::QCaptureInit | MoveStage::ProbCutInit => {
                    self.generate_captures(game, searcher);

                    self.cur = 0;
                    self.end_bad_captures = 0;
                    self.end_captures = self.moves.len();

                    // Sort all captures (limit = MIN to include all)
                    partial_insertion_sort(&mut self.moves, i32::MIN);

                    self.stage = match self.stage {
                        MoveStage::CaptureInit => MoveStage::GoodCapture,
                        MoveStage::QCaptureInit => MoveStage::QCapture,
                        MoveStage::ProbCutInit => MoveStage::ProbCut,
                        _ => unreachable!(),
                    };
                }

                MoveStage::GoodCapture => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];

                        if super::see_ge(game, &sm.m, -sm.score / 18) {
                            self.cur += 1;
                            return Some(sm.m);
                        } else {
                            // Bad capture - swap to front for later
                            self.moves.swap(self.end_bad_captures, self.cur);
                            self.end_bad_captures += 1;
                        }
                        self.cur += 1;
                    }

                    self.stage = MoveStage::Killer1;
                }

                MoveStage::Killer1 => {
                    self.stage = MoveStage::Killer2;
                    // Lazy Killer 1
                    if self.skip_quiets {
                        continue;
                    }

                    if let Some(m) = self.killer1
                        && !self.is_tt_move(&m)
                        && !self.is_excluded(&m)
                        && Self::is_pseudo_legal(game, &m)
                    {
                        return Some(m);
                    }
                }

                MoveStage::Killer2 => {
                    self.stage = MoveStage::QuietInit;
                    // Lazy Killer 2
                    if self.skip_quiets {
                        continue;
                    }

                    if let Some(m) = self.killer2
                        && !self.is_tt_move(&m)
                        && !self.is_excluded(&m)
                        && !Self::moves_match(&m, &self.killer1)
                        && Self::is_pseudo_legal(game, &m)
                    {
                        return Some(m);
                    }
                }

                MoveStage::QuietInit => {
                    if self.skip_quiets {
                        // Prepare for bad captures
                        self.cur = 0;
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    let quiet_start = self.moves.len();
                    self.generate_quiets(game, searcher);
                    self.end_generated = self.moves.len();

                    // Partial sort with depth-based limit
                    let limit = -3560 * self.depth;
                    partial_insertion_sort(&mut self.moves[quiet_start..], limit);

                    self.cur = quiet_start;
                    self.stage = MoveStage::GoodQuiet;
                }

                MoveStage::GoodQuiet => {
                    if self.skip_quiets {
                        self.cur = 0;
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    while self.cur < self.end_generated {
                        let sm = self.moves[self.cur];
                        self.cur += 1;

                        if sm.score > GOOD_QUIET_THRESHOLD {
                            return Some(sm.m);
                        }
                    }

                    // Prepare for bad captures
                    self.cur = 0;
                    self.stage = MoveStage::BadCapture;
                }

                MoveStage::BadCapture => {
                    if self.cur < self.end_bad_captures {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }

                    // Prepare for bad quiets
                    self.cur = self.end_captures;
                    self.stage = MoveStage::BadQuiet;
                }

                MoveStage::BadQuiet => {
                    if self.skip_quiets {
                        self.stage = MoveStage::Done;
                        return None;
                    }

                    while self.cur < self.end_generated {
                        let sm = self.moves[self.cur];
                        self.cur += 1;

                        if sm.score <= GOOD_QUIET_THRESHOLD {
                            return Some(sm.m);
                        }
                    }

                    self.stage = MoveStage::Done;
                }

                MoveStage::EvasionInit => {
                    self.generate_evasions(game, searcher);
                    self.end_generated = self.moves.len();
                    self.cur = 0;

                    partial_insertion_sort(&mut self.moves, i32::MIN);

                    self.stage = MoveStage::Evasion;
                }

                MoveStage::Evasion | MoveStage::QCapture => {
                    if self.cur < self.end_generated.max(self.end_captures) {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }
                    self.stage = MoveStage::Done;
                }

                MoveStage::ProbCut => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];
                        self.cur += 1;

                        if super::see_ge(game, &sm.m, self.threshold) {
                            return Some(sm.m);
                        }
                    }
                    self.stage = MoveStage::Done;
                }

                // =================================================================
                // DONE
                // =================================================================
                MoveStage::Done => {
                    return None;
                }
            }
        }
    }
}
