//! Staged move generation for efficient alpha-beta search.
//!
//! Generates moves in the exact same order as sort_moves:
//! 1. TT move (hash move) - highest priority
//! 2. Good captures (SEE >= threshold, MVV-LVA + capture history sorted)
//! 3. Killer 1 (if valid quiet move)
//! 4. Killer 2 (if valid quiet move)  
//! 5. Quiets (history + continuation history sorted)
//! 6. Bad captures (SEE < threshold)

use super::params::{
    DEFAULT_SORT_QUIET, see_winning_threshold, sort_countermove, sort_gives_check, sort_killer1,
    sort_killer2,
};
use super::{Searcher, hash_coord_32, hash_move_dest, static_exchange_eval};
use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::{Move, get_quiescence_captures, get_quiet_moves_into};

/// Stages of move generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveStage {
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

/// Staged move generator with exact same ordering as sort_moves.
pub struct StagedMoveGen {
    stage: MoveStage,

    // TT move
    tt_move: Option<Move>,
    tt_move_yielded: bool,

    // Ply for history lookups
    ply: usize,

    // Previous move info for countermove lookup
    prev_from_hash: usize,
    prev_to_hash: usize,

    // Cached enemy king for check detection
    enemy_king: Option<Coordinate>,

    // Good captures (sorted)
    good_captures: Vec<(Move, i32)>, // (move, score)
    good_capture_idx: usize,

    // Bad captures (deferred)
    bad_captures: Vec<Move>,
    bad_capture_idx: usize,

    // Killers
    killer1: Option<Move>,
    killer2: Option<Move>,
    killer1_yielded: bool,
    killer2_yielded: bool,

    // Quiets (sorted by history)
    quiets: Vec<(Move, i32)>, // (move, score)
    quiet_idx: usize,

    // Excluded move (for singular extension verification)
    excluded_move: Option<Move>,

    // Skip quiet moves flag (for FutilityMoveCount pruning)
    // When set, the generator will skip generating/returning quiet moves
    skip_quiets: bool,
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
            tt_move_yielded: false,
            ply,
            prev_from_hash,
            prev_to_hash,
            enemy_king,
            good_captures: Vec::with_capacity(32),
            good_capture_idx: 0,
            bad_captures: Vec::with_capacity(16),
            bad_capture_idx: 0,
            killer1,
            killer2,
            killer1_yielded: false,
            killer2_yielded: false,
            quiets: Vec::with_capacity(64),
            quiet_idx: 0,
            excluded_move: None,
            skip_quiets: false,
        }
    }

    /// Create a staged move generator that excludes a specific move.
    /// Used for Singular Extension verification searches.
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

    /// Skip generating and returning quiet moves.
    /// Called from search when moveCount exceeds FutilityMoveCount threshold.
    /// This prevents wasting time generating quiets that will be pruned anyway.
    #[inline]
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    /// Check if a move is the excluded move
    #[inline]
    fn is_excluded(&self, m: &Move) -> bool {
        if let Some(ref excl) = self.excluded_move {
            m.from == excl.from && m.to == excl.to && m.promotion == excl.promotion
        } else {
            false
        }
    }

    /// Check if move is pseudo-legal
    #[inline]
    fn is_pseudo_legal(game: &GameState, m: &Move) -> bool {
        if let Some(piece) = game.board.get_piece(&m.from.x, &m.from.y) {
            if piece.color() != game.turn || piece.piece_type() != m.piece.piece_type() {
                return false;
            }

            // O(1) castling integrity check:
            // TT moves may be invalid if path is blocked or rook moved. Check:
            // 1. Rook exists at rook_coord
            // 2. King's landing squares are clear (m.to and one square between)
            // 3. Rook's landing square is clear
            if piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1 {
                if let Some(rook_coord) = &m.rook_coord {
                    if let Some(rook) = game.board.get_piece(&rook_coord.x, &rook_coord.y) {
                        if rook.color() != game.turn {
                            return false;
                        }

                        // Calculate direction and critical squares
                        let dir = if m.to.x > m.from.x { 1 } else { -1 };

                        // King passes through m.from+dir and lands on m.to (m.from + 2*dir)
                        let king_path_1 = m.from.x + dir;
                        let king_path_2 = m.to.x; // = m.from.x + 2*dir

                        // Check king's path squares (these must be empty for castling)
                        if game.board.get_piece(&king_path_1, &m.from.y).is_some() {
                            return false;
                        }
                        if game.board.get_piece(&king_path_2, &m.from.y).is_some() {
                            return false;
                        }

                        // For queenside, also check the square next to rook (b-file)
                        // The rook passes through this square
                        if dir < 0 {
                            let extra_path = m.from.x - 3; // b1 for standard chess
                            if game.board.get_piece(&extra_path, &m.from.y).is_some() {
                                return false;
                            }
                        }
                    } else {
                        return false; // Rook missing
                    }
                } else {
                    return false; // Rook coord missing
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if move is a capture
    #[inline]
    fn is_capture(game: &GameState, m: &Move) -> bool {
        game.board.get_piece(&m.to.x, &m.to.y).is_some()
    }

    /// Check if two moves match
    #[inline]
    fn moves_match(a: &Move, b: &Option<Move>) -> bool {
        match b {
            Some(bm) => {
                a.from == bm.from
                    && a.to == bm.to
                    && a.promotion == bm.promotion
                    && a.rook_coord == bm.rook_coord
            }
            None => false,
        }
    }

    /// Score capture move (MVV-LVA + capture history + check bonus)
    fn score_capture(
        game: &GameState,
        searcher: &Searcher,
        m: &Move,
        enemy_king: &Option<Coordinate>,
    ) -> i32 {
        let mut score: i32 = 0;

        if let Some(target) = game.board.get_piece(&m.to.x, &m.to.y) {
            let victim_val = get_piece_value(target.piece_type());
            let attacker_val = get_piece_value(m.piece.piece_type());
            let mvv_lva = victim_val * 10 - attacker_val;
            score += mvv_lva;

            // Capture history
            let cap_hist = searcher.capture_history[m.piece.piece_type() as usize]
                [target.piece_type() as usize];
            score += cap_hist / 10;

            // Check bonus for captures that give check
            if let Some(ek) = enemy_king {
                if Self::move_gives_check_simple(game, m, ek) {
                    score += sort_gives_check();
                }
            }
        }

        score
    }

    /// Score quiet move (killer + countermove + check + history + continuation)
    fn score_quiet(
        game: &GameState,
        searcher: &Searcher,
        m: &Move,
        ply: usize,
        prev_from_hash: usize,
        prev_to_hash: usize,
        enemy_king: &Option<Coordinate>,
        killer1: &Option<Move>,
        killer2: &Option<Move>,
    ) -> i32 {
        let mut score: i32 = DEFAULT_SORT_QUIET;

        // Killer bonus
        if Self::moves_match(m, killer1) {
            return sort_killer1();
        }
        if Self::moves_match(m, killer2) {
            return sort_killer2();
        }

        // Check bonus
        if let Some(ek) = enemy_king {
            if Self::move_gives_check_simple(game, m, ek) {
                score += sort_gives_check();
            }
        }

        // Countermove bonus
        if ply > 0 && prev_from_hash < 256 && prev_to_hash < 256 {
            let (cm_piece, cm_to_x, cm_to_y) = searcher.countermoves[prev_from_hash][prev_to_hash];
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
        score += searcher.history[m.piece.piece_type() as usize][idx];

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

    /// Simple check detection (direct attacks to enemy king)
    #[inline]
    fn move_gives_check_simple(game: &GameState, m: &Move, enemy_king: &Coordinate) -> bool {
        let to = &m.to;
        let kx = enemy_king.x;
        let ky = enemy_king.y;
        let dx = kx - to.x;
        let dy = ky - to.y;

        let piece_type = m.promotion.unwrap_or(m.piece.piece_type());

        match piece_type {
            PieceType::Pawn => {
                let dir = if m.piece.color() == PlayerColor::White {
                    1
                } else {
                    -1
                };
                dx.abs() == 1 && dy == dir
            }
            PieceType::Knight => {
                let adx = dx.abs();
                let ady = dy.abs();
                (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
            }
            PieceType::Bishop => {
                dx.abs() == dy.abs()
                    && dx != 0
                    && Self::path_clear(game, to, enemy_king, dx.signum(), dy.signum())
            }
            PieceType::Rook => {
                ((dx == 0 && dy != 0) || (dy == 0 && dx != 0))
                    && Self::path_clear(game, to, enemy_king, dx.signum(), dy.signum())
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                let is_ortho = (dx == 0 && dy != 0) || (dy == 0 && dx != 0);
                let is_diag = dx.abs() == dy.abs() && dx != 0;
                (is_ortho || is_diag)
                    && Self::path_clear(game, to, enemy_king, dx.signum(), dy.signum())
            }
            _ => false,
        }
    }

    #[inline]
    fn path_clear(
        game: &GameState,
        from: &Coordinate,
        to: &Coordinate,
        step_x: i64,
        step_y: i64,
    ) -> bool {
        let mut x = from.x + step_x;
        let mut y = from.y + step_y;
        let mut steps = 0;
        const MAX_PATH: i64 = 50;

        while (x != to.x || y != to.y) && steps < MAX_PATH {
            if game.board.get_piece(&x, &y).is_some() {
                return false;
            }
            x += step_x;
            y += step_y;
            steps += 1;
        }
        steps < MAX_PATH
    }

    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        loop {
            match self.stage {
                MoveStage::TTMove => {
                    self.stage = MoveStage::GenerateCaptures;
                    if let Some(ref m) = self.tt_move {
                        // Skip if this is the excluded move (for SE verification)
                        if !self.is_excluded(m) && Self::is_pseudo_legal(game, m) {
                            self.tt_move_yielded = true;
                            return Some(m.clone());
                        }
                    }
                }

                MoveStage::GenerateCaptures => {
                    // Generate all captures
                    let mut captures = Vec::with_capacity(32);
                    get_quiescence_captures(
                        &game.board,
                        game.turn,
                        &game.special_rights,
                        &game.en_passant,
                        &game.game_rules,
                        &game.spatial_indices,
                        &mut captures,
                    );

                    // Separate good and bad captures with scores
                    let enemy_king = self.enemy_king;
                    for m in captures {
                        // Skip TT move or excluded move
                        if Self::moves_match(&m, &self.tt_move) || self.is_excluded(&m) {
                            continue;
                        }

                        let see_val = static_exchange_eval(game, &m);
                        if see_val >= see_winning_threshold() {
                            let score = Self::score_capture(game, searcher, &m, &enemy_king);
                            self.good_captures.push((m, score));
                        } else {
                            self.bad_captures.push(m);
                        }
                    }

                    // Sort good captures by score (highest first)
                    self.good_captures.sort_by(|a, b| b.1.cmp(&a.1));

                    self.stage = MoveStage::GoodCaptures;
                }

                MoveStage::GoodCaptures => {
                    while self.good_capture_idx < self.good_captures.len() {
                        let (m, _) = self.good_captures[self.good_capture_idx].clone();
                        self.good_capture_idx += 1;
                        // Skip excluded move
                        if self.is_excluded(&m) {
                            continue;
                        }
                        return Some(m);
                    }
                    self.stage = MoveStage::Killer1;
                }

                MoveStage::Killer1 => {
                    self.stage = MoveStage::Killer2;
                    // Skip killers if we're skipping quiets
                    if self.skip_quiets {
                        continue;
                    }
                    if !self.killer1_yielded {
                        self.killer1_yielded = true;
                        if let Some(ref k) = self.killer1 {
                            // Killer must be: not TT move, not a capture, pseudo-legal
                            if !Self::moves_match(k, &self.tt_move)
                                && !self.is_excluded(k)
                                && !Self::is_capture(game, k)
                                && Self::is_pseudo_legal(game, k)
                            {
                                return Some(k.clone());
                            }
                        }
                    }
                }

                MoveStage::Killer2 => {
                    self.stage = MoveStage::GenerateQuiets;
                    // Skip killers if we're skipping quiets
                    if self.skip_quiets {
                        continue;
                    }
                    if !self.killer2_yielded {
                        self.killer2_yielded = true;
                        if let Some(ref k) = self.killer2 {
                            if !Self::moves_match(k, &self.tt_move)
                                && !Self::moves_match(k, &self.killer1)
                                && !self.is_excluded(k)
                                && !Self::is_capture(game, k)
                                && Self::is_pseudo_legal(game, k)
                            {
                                return Some(k.clone());
                            }
                        }
                    }
                }

                MoveStage::GenerateQuiets => {
                    // Skip quiet generation entirely if we're skipping quiets
                    if self.skip_quiets {
                        self.stage = MoveStage::BadCaptures;
                        continue;
                    }

                    let mut quiets_raw = Vec::with_capacity(64);
                    get_quiet_moves_into(
                        &game.board,
                        game.turn,
                        &game.special_rights,
                        &game.en_passant,
                        &game.game_rules,
                        &game.spatial_indices,
                        &mut quiets_raw,
                    );

                    // Score and collect quiets
                    let ply = self.ply;
                    let prev_from = self.prev_from_hash;
                    let prev_to = self.prev_to_hash;
                    let enemy_king = self.enemy_king;
                    let killer1 = self.killer1.clone();
                    let killer2 = self.killer2.clone();

                    for m in quiets_raw {
                        let score = Self::score_quiet(
                            game,
                            searcher,
                            &m,
                            ply,
                            prev_from,
                            prev_to,
                            &enemy_king,
                            &killer1,
                            &killer2,
                        );
                        self.quiets.push((m, score));
                    }

                    // Sort quiets by score (highest first)
                    self.quiets.sort_by(|a, b| b.1.cmp(&a.1));

                    self.stage = MoveStage::Quiets;
                }

                MoveStage::Quiets => {
                    while self.quiet_idx < self.quiets.len() {
                        let (m, _) = self.quiets[self.quiet_idx].clone();
                        self.quiet_idx += 1;

                        // Skip TT move and killers (already yielded)
                        if Self::moves_match(&m, &self.tt_move) {
                            continue;
                        }
                        if self.killer1_yielded && Self::moves_match(&m, &self.killer1) {
                            continue;
                        }
                        if self.killer2_yielded && Self::moves_match(&m, &self.killer2) {
                            continue;
                        }
                        // Skip excluded move
                        if self.is_excluded(&m) {
                            continue;
                        }

                        return Some(m);
                    }
                    self.stage = MoveStage::BadCaptures;
                }

                MoveStage::BadCaptures => {
                    while self.bad_capture_idx < self.bad_captures.len() {
                        let m = self.bad_captures[self.bad_capture_idx].clone();
                        self.bad_capture_idx += 1;
                        // Skip excluded move
                        if self.is_excluded(&m) {
                            continue;
                        }
                        return Some(m);
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
