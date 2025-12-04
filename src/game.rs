use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::evaluation::{calculate_initial_material, get_piece_value};
use crate::moves::{
    get_legal_moves, get_legal_moves_into, get_pseudo_legal_moves_for_piece, is_square_attacked,
    Move, SpatialIndices,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct EnPassantState {
    pub square: Coordinate,
    pub pawn_square: Coordinate,
}

/// Promotion ranks configuration for a variant
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct PromotionRanks {
    pub white: Vec<i64>,
    pub black: Vec<i64>,
}

/// Game rules that can vary between chess variants
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct GameRules {
    pub promotion_ranks: Option<PromotionRanks>,
    #[serde(skip)]
    pub promotion_types: Option<Vec<PieceType>>, // Pre-converted promotion piece types (fast)
    pub promotions_allowed: Option<Vec<String>>, // Piece type codes (only for serialization)
}

impl GameRules {
    /// Convert promotions_allowed strings to PieceTypes once
    pub fn init_promotion_types(&mut self) {
        if let Some(ref allowed) = self.promotions_allowed {
            self.promotion_types = Some(
                allowed
                    .iter()
                    .filter_map(|s| PieceType::from_str(s.as_str()))
                    .collect(),
            );
        }
    }
}

#[derive(Clone)]
pub struct UndoMove {
    pub captured_piece: Option<Piece>,
    pub old_en_passant: Option<EnPassantState>,
    pub old_halfmove_clock: u32,
    pub old_hash: u64,                           // Hash before the move was made
    pub special_rights_removed: Vec<Coordinate>, // Track which special rights were removed (re-insert on undo)
    /// If this move caused a piece to leave its original starting square,
    /// we remove that coordinate from starting_squares. Store it here so
    /// undo_move can restore starting_squares exactly.
    pub starting_square_restored: Option<Coordinate>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub turn: PlayerColor,
    /// Special rights for pieces - includes both castling rights (kings/rooks) AND
    /// pawn double-move rights. A piece with its coordinate in this set has its special rights.
    pub special_rights: HashSet<Coordinate>,
    pub en_passant: Option<EnPassantState>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub material_score: i32,   // Positive = White advantage
    pub game_rules: GameRules, // Variant-specific rules
    /// Optional variant identifier (e.g. "Classical", "Pawn_Horde"), used for
    /// variant-specific evaluation and tuning. Not serialized.
    #[serde(skip)]
    pub variant: Option<crate::Variant>,
    #[serde(skip)]
    pub hash: u64, // Incrementally maintained Zobrist hash
    #[serde(skip)]
    pub hash_stack: Vec<u64>, // Position hashes for repetition detection
    #[serde(skip)]
    pub null_moves: u8, // Counter for null moves (for repetition detection)
    #[serde(skip)]
    pub white_piece_count: u16,
    #[serde(skip)]
    pub black_piece_count: u16,
    /// Piece coordinates per color for fast iteration (avoid scanning full HashMap)
    #[serde(skip)]
    pub white_pieces: Vec<(i64, i64)>,
    #[serde(skip)]
    pub black_pieces: Vec<(i64, i64)>,
    /// Spatial indices for fast sliding move and attack queries
    #[serde(skip)]
    pub spatial_indices: SpatialIndices,
    /// Starting squares for development: coordinates where non-pawn,
    /// non-royal pieces began the game. Used to apply a one-time
    /// development penalty while a piece remains on its original square.
    #[serde(skip)]
    pub starting_squares: HashSet<Coordinate>,
    /// Cached dynamic back ranks derived from promotion_ranks. These are
    /// computed once when the game is created.
    #[serde(skip)]
    pub white_back_rank: i64,
    #[serde(skip)]
    pub black_back_rank: i64,
    /// Cached effective promotion ranks per color, computed once when the
    /// game is created. Used by pawn evaluation to avoid per-eval scans of
    /// game_rules.promotion_ranks.
    #[serde(skip)]
    pub white_promo_rank: i64,
    #[serde(skip)]
    pub black_promo_rank: i64,
}

// For backwards compatibility, keep castling_rights as an alias
impl GameState {
    /// Returns pieces that can castle (kings and rooks with special rights)
    pub fn castling_rights(&self) -> HashSet<Coordinate> {
        let mut rights = HashSet::new();
        for coord in &self.special_rights {
            if let Some(piece) = self.board.get_piece(&coord.x, &coord.y) {
                // Only include kings and rooks (not pawns) in castling rights
                if piece.piece_type == PieceType::King
                    || piece.piece_type == PieceType::Rook
                    || piece.piece_type == PieceType::RoyalCentaur
                {
                    rights.insert(coord.clone());
                }
            }
        }
        rights
    }

    /// Check if a piece at the given coordinate has its special rights
    pub fn has_special_right(&self, coord: &Coordinate) -> bool {
        self.special_rights.contains(coord)
    }
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: HashSet::new(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            game_rules: GameRules::default(),
            variant: None,
            hash: 0,
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
            starting_squares: HashSet::new(),
            white_back_rank: 1,
            black_back_rank: 8,
            white_promo_rank: 8,
            black_promo_rank: 1,
        }
    }

    pub fn new_with_rules(game_rules: GameRules) -> Self {
        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: HashSet::new(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            game_rules,
            variant: None,
            hash: 0,
            hash_stack: Vec::with_capacity(128),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
            starting_squares: HashSet::new(),
            white_back_rank: 1,
            black_back_rank: 8,
            white_promo_rank: 8,
            black_promo_rank: 1,
        }
    }

    /// Recompute piece counts and rebuild piece lists from the board
    pub fn recompute_piece_counts(&mut self) {
        let mut white: u16 = 0;
        let mut black: u16 = 0;
        self.white_pieces.clear();
        self.black_pieces.clear();

        for ((x, y), piece) in &self.board.pieces {
            match piece.color {
                PlayerColor::White => {
                    white = white.saturating_add(1);
                    self.white_pieces.push((*x, *y));
                }
                PlayerColor::Black => {
                    black = black.saturating_add(1);
                    self.black_pieces.push((*x, *y));
                }
                PlayerColor::Neutral => {}
            }
        }
        self.white_piece_count = white;
        self.black_piece_count = black;
        // Rebuild spatial indices from current board
        self.spatial_indices = SpatialIndices::new(&self.board);
    }

    /// Initialize starting_squares from the current board: all non-pawn,
    /// non-royal pieces' current coordinates are treated as their original
    /// squares. Intended to be called once when constructing a GameState
    /// from an initial position before replaying move history.
    pub fn init_starting_squares(&mut self) {
        self.starting_squares.clear();
        for ((x, y), piece) in &self.board.pieces {
            if piece.piece_type != PieceType::Pawn && !piece.piece_type.is_royal() {
                self.starting_squares.insert(Coordinate::new(*x, *y));
            }
        }
    }

    #[inline]
    pub fn has_pieces(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::White => self.white_piece_count > 0,
            PlayerColor::Black => self.black_piece_count > 0,
            PlayerColor::Neutral => false,
        }
    }

    /// Check for threefold repetition
    pub fn is_threefold(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }

        // Need at least 6 positions to have a potential threefold
        if self.hash_stack.len() < 6 {
            return false;
        }

        // Generate current position hash
        let current_hash = self.generate_hash();

        let mut repetitions_count = 1;
        // Only look back as far as halfmove_clock allows (captures/pawn moves reset repetition)
        let lookback = (self.halfmove_clock as usize).min(self.hash_stack.len());
        let from = self.hash_stack.len().saturating_sub(lookback);
        let to = self.hash_stack.len().saturating_sub(1);

        if to <= from {
            return false;
        }

        // Check every other position (same side to move)
        for hash_index in (from..to).rev().step_by(2) {
            if self.hash_stack[hash_index] == current_hash {
                repetitions_count += 1;

                if repetitions_count >= 3 {
                    return true;
                }
            }
        }

        false
    }

    /// Check if this is a lone king endgame (one side only has a king)
    pub fn is_lone_king_endgame(&self) -> bool {
        use crate::board::{PieceType, PlayerColor};

        let mut white_has_non_king = false;
        let mut black_has_non_king = false;

        for (_, piece) in &self.board.pieces {
            if piece.piece_type != PieceType::King {
                if piece.color == PlayerColor::White {
                    white_has_non_king = true;
                } else {
                    black_has_non_king = true;
                }
            }
        }

        // One side has only a king (or nothing)
        !white_has_non_king || !black_has_non_king
    }

    /// Check if position is a draw by 50-move rule
    pub fn is_fifty(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        self.halfmove_clock >= 100
    }

    /// Make a null move (just flip turn, for null move pruning)
    pub fn make_null_move(&mut self) {
        use crate::search::zobrist::{en_passant_key, SIDE_KEY};

        // Push current hash
        self.hash_stack.push(self.hash);

        // Update hash: remove en passant
        if let Some(ep) = &self.en_passant {
            self.hash ^= en_passant_key(ep.square.x, ep.square.y);
        }

        // Clear en passant
        self.en_passant = None;

        // Update hash: flip turn
        self.hash ^= SIDE_KEY;

        // Flip turn
        self.turn = self.turn.opponent();

        self.null_moves += 1;
    }

    /// Unmake a null move
    pub fn unmake_null_move(&mut self) {
        // Pop hash (restores the old hash)
        if let Some(old_hash) = self.hash_stack.pop() {
            self.hash = old_hash;
        }

        // Flip turn back
        self.turn = self.turn.opponent();

        self.null_moves -= 1;
    }

    /// Return the incrementally maintained hash (fast)
    #[inline]
    pub fn generate_hash(&self) -> u64 {
        self.hash
    }

    /// Recompute the hash from scratch (slow, use sparingly)
    pub fn recompute_hash(&mut self) {
        use crate::search::zobrist::{en_passant_key, piece_key, special_right_key, SIDE_KEY};

        let mut h: u64 = 0;

        // Hash all pieces
        for ((x, y), piece) in &self.board.pieces {
            h ^= piece_key(piece.piece_type, piece.color, *x, *y);
        }

        // Hash special rights
        for coord in &self.special_rights {
            h ^= special_right_key(coord);
        }

        // Hash en passant
        if let Some(ep) = &self.en_passant {
            h ^= en_passant_key(ep.square.x, ep.square.y);
        }

        // Hash side to move
        if self.turn == PlayerColor::Black {
            h ^= SIDE_KEY;
        }

        self.hash = h;
    }

    /// Returns pseudo-legal moves. Legality (not leaving king in check)
    /// is checked in the search after making each move.
    pub fn get_legal_moves(&self) -> Vec<Move> {
        get_legal_moves(
            &self.board,
            self.turn,
            &self.special_rights,
            &self.en_passant,
            &self.game_rules,
            &self.spatial_indices,
        )
    }

    /// Fill a pre-allocated buffer with pseudo-legal moves for the current side.
    pub fn get_legal_moves_into(&self, out: &mut Vec<Move>) {
        get_legal_moves_into(
            &self.board,
            self.turn,
            &self.special_rights,
            &self.en_passant,
            &self.game_rules,
            &self.spatial_indices,
            out,
        );
    }

    pub fn get_evasion_moves_into(&self, out: &mut Vec<Move>) {
        out.clear();

        let our_color = self.turn;
        let their_color = our_color.opponent();

        for (_, piece) in &self.board.pieces {
            if piece.color == PlayerColor::Neutral {
                continue;
            }
            match piece.piece_type {
                PieceType::King
                | PieceType::Queen
                | PieceType::Rook
                | PieceType::Bishop
                | PieceType::Knight
                | PieceType::Pawn
                | PieceType::RoyalQueen
                | PieceType::RoyalCentaur => {}
                _ => {
                    self.get_legal_moves_into(out);
                    return;
                }
            }
        }

        let mut royal_pos: Option<(Coordinate, Piece)> = None;
        for ((x, y), piece) in &self.board.pieces {
            if piece.color == our_color && piece.piece_type.is_royal() {
                if royal_pos.is_some() {
                    self.get_legal_moves_into(out);
                    return;
                }
                royal_pos = Some((Coordinate::new(*x, *y), *piece));
            }
        }

        let (king_sq, king_piece) = match royal_pos {
            Some(v) => v,
            None => {
                self.get_legal_moves_into(out);
                return;
            }
        };

        let mut checkers: Vec<Coordinate> = Vec::new();

        let pawn_dir: i64 = match their_color {
            PlayerColor::White => 1,
            PlayerColor::Black => -1,
            PlayerColor::Neutral => 0,
        };
        if pawn_dir != 0 {
            let pawn_y = king_sq.y - pawn_dir;
            for dx in [-1i64, 1] {
                let px = king_sq.x + dx;
                if let Some(p) = self.board.get_piece(&px, &pawn_y) {
                    if p.color == their_color && p.piece_type == PieceType::Pawn {
                        checkers.push(Coordinate::new(px, pawn_y));
                    }
                }
            }
        }

        let knight_offsets = [
            (1i64, 2),
            (2, 1),
            (-1, 2),
            (-2, 1),
            (1, -2),
            (2, -1),
            (-1, -2),
            (-2, -1),
        ];
        for (dx, dy) in knight_offsets.iter() {
            let x = king_sq.x + dx;
            let y = king_sq.y + dy;
            if let Some(p) = self.board.get_piece(&x, &y) {
                if p.color == their_color && p.piece_type == PieceType::Knight {
                    checkers.push(Coordinate::new(x, y));
                }
            }
        }

        let king_offsets = [
            (-1i64, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        for (dx, dy) in king_offsets.iter() {
            let x = king_sq.x + dx;
            let y = king_sq.y + dy;
            if let Some(p) = self.board.get_piece(&x, &y) {
                if p.color == their_color && p.piece_type == PieceType::King {
                    checkers.push(Coordinate::new(x, y));
                }
            }
        }

        let ortho_dirs = [(1i64, 0), (-1, 0), (0, 1), (0, -1)];
        let diag_dirs = [(1i64, 1), (1, -1), (-1, 1), (-1, -1)];

        for (dx, dy) in ortho_dirs.iter() {
            let mut x = king_sq.x + dx;
            let mut y = king_sq.y + dy;
            loop {
                if let Some(p) = self.board.get_piece(&x, &y) {
                    if p.color == their_color
                        && (p.piece_type == PieceType::Rook || p.piece_type == PieceType::Queen)
                    {
                        checkers.push(Coordinate::new(x, y));
                    }
                    break;
                }
                x += dx;
                y += dy;
                if (x - king_sq.x).abs() > 50 || (y - king_sq.y).abs() > 50 {
                    break;
                }
            }
        }

        for (dx, dy) in diag_dirs.iter() {
            let mut x = king_sq.x + dx;
            let mut y = king_sq.y + dy;
            loop {
                if let Some(p) = self.board.get_piece(&x, &y) {
                    if p.color == their_color
                        && (p.piece_type == PieceType::Bishop || p.piece_type == PieceType::Queen)
                    {
                        checkers.push(Coordinate::new(x, y));
                    }
                    break;
                }
                x += dx;
                y += dy;
                if (x - king_sq.x).abs() > 50 || (y - king_sq.y).abs() > 50 {
                    break;
                }
            }
        }

        if checkers.is_empty() {
            self.get_legal_moves_into(out);
            return;
        }

        let king_moves = get_pseudo_legal_moves_for_piece(
            &self.board,
            &king_piece,
            &king_sq,
            &self.special_rights,
            &self.en_passant,
            Some(&self.spatial_indices),
            &self.game_rules,
        );
        out.extend(king_moves);

        if checkers.len() >= 2 {
            return;
        }

        let checker_sq = checkers[0];
        let dx_check = checker_sq.x - king_sq.x;
        let dy_check = checker_sq.y - king_sq.y;

        let mut block_squares: Vec<Coordinate> = Vec::new();
        let is_slider = dx_check == 0 || dy_check == 0 || dx_check.abs() == dy_check.abs();
        if is_slider {
            let step_x = dx_check.signum();
            let step_y = dy_check.signum();
            let mut x = king_sq.x + step_x;
            let mut y = king_sq.y + step_y;
            while x != checker_sq.x || y != checker_sq.y {
                block_squares.push(Coordinate::new(x, y));
                x += step_x;
                y += step_y;
            }
        }

        let mut targets: Vec<Coordinate> = Vec::new();
        targets.push(checker_sq);
        targets.extend(block_squares.iter().copied());

        if targets.is_empty() {
            return;
        }

        for ((x, y), piece) in &self.board.pieces {
            if piece.color != our_color {
                continue;
            }
            if *x == king_sq.x && *y == king_sq.y {
                continue;
            }
            let from = Coordinate::new(*x, *y);

            let mut may_reach = false;
            for target in &targets {
                let tx = target.x;
                let ty = target.y;
                let ddx = tx - from.x;
                let ddy = ty - from.y;

                match piece.piece_type {
                    PieceType::Pawn => {
                        let dir = match our_color {
                            PlayerColor::White => 1,
                            PlayerColor::Black => -1,
                            PlayerColor::Neutral => 0,
                        };
                        if dir != 0 {
                            if ddy == dir && (ddx == -1 || ddx == 1) {
                                may_reach = true;
                            } else if ddx == 0 && (ddy == dir || ddy == dir * 2) {
                                may_reach = true;
                            }
                        }
                    }
                    PieceType::Knight => {
                        if (ddx.abs() == 1 && ddy.abs() == 2) || (ddx.abs() == 2 && ddy.abs() == 1)
                        {
                            may_reach = true;
                        }
                    }
                    PieceType::Bishop => {
                        if ddx.abs() == ddy.abs() && ddx != 0 {
                            may_reach = true;
                        }
                    }
                    PieceType::Rook => {
                        if (ddx == 0 && ddy != 0) || (ddy == 0 && ddx != 0) {
                            may_reach = true;
                        }
                    }
                    PieceType::Queen => {
                        if (ddx == 0 && ddy != 0)
                            || (ddy == 0 && ddx != 0)
                            || (ddx.abs() == ddy.abs() && ddx != 0)
                        {
                            may_reach = true;
                        }
                    }
                    _ => {}
                }

                if may_reach {
                    break;
                }
            }

            if !may_reach {
                continue;
            }

            let pseudo = get_pseudo_legal_moves_for_piece(
                &self.board,
                piece,
                &from,
                &self.special_rights,
                &self.en_passant,
                Some(&self.spatial_indices),
                &self.game_rules,
            );
            for m in pseudo {
                for target in &targets {
                    if m.to.x == target.x && m.to.y == target.y {
                        out.push(m);
                        break;
                    }
                }
            }
        }
    }

    /// Check if the side that just moved left their royal piece(s) in check (illegal move).
    /// Call this AFTER make_move to verify legality.
    /// Checks all royal pieces: King, RoyalQueen, RoyalCentaur
    pub fn is_move_illegal(&self) -> bool {
        // After make_move, self.turn is the opponent.
        // We need to check if the side that just moved (opponent of current turn) has any royal in check.
        let moved_color = self.turn.opponent();
        let indices = &self.spatial_indices;

        // Find ALL royal pieces of the side that just moved and check if any are attacked
        for ((x, y), piece) in &self.board.pieces {
            if piece.color == moved_color && piece.piece_type.is_royal() {
                let pos = Coordinate::new(*x, *y);
                if is_square_attacked(&self.board, &pos, self.turn, Some(indices)) {
                    return true;
                }
            }
        }
        false
    }

    pub fn is_in_check(&self) -> bool {
        let indices = &self.spatial_indices;
        let attacker_color = self.turn.opponent();

        // Check if ANY royal piece of current player is attacked
        for ((x, y), piece) in &self.board.pieces {
            if piece.color == self.turn && piece.piece_type.is_royal() {
                let pos = Coordinate::new(*x, *y);
                if is_square_attacked(&self.board, &pos, attacker_color, Some(indices)) {
                    return true;
                }
            }
        }
        false
    }

    /// Make a move given just from/to coordinates and optional promotion.
    /// Like UCI - we trust the input is valid and just execute it directly.
    /// This is much faster than generating all legal moves for history replay.
    pub fn make_move_coords(
        &mut self,
        from_x: i64,
        from_y: i64,
        to_x: i64,
        to_y: i64,
        promotion: Option<&str>,
    ) {
        // Push current position hash BEFORE making the move
        let current_hash = self.generate_hash();
        self.hash_stack.push(current_hash);

        // Once a piece moves from its original square, we no longer treat
        // that coordinate as an undeveloped starting square.
        self.starting_squares
            .remove(&Coordinate::new(from_x, from_y));

        let piece = match self.board.remove_piece(&from_x, &from_y) {
            Some(p) => p,
            None => return, // No piece at from - invalid move, just skip
        };
        // Update spatial indices: remove piece from source square
        self.spatial_indices.remove(from_x, from_y);

        // Handle capture
        let captured = self.board.remove_piece(&to_x, &to_y);
        let is_capture = captured.is_some();

        if let Some(ref cap) = captured {
            let value = get_piece_value(cap.piece_type);
            if cap.color == PlayerColor::White {
                self.material_score -= value;
                self.white_piece_count = self.white_piece_count.saturating_sub(1);
            } else {
                self.material_score += value;
                self.black_piece_count = self.black_piece_count.saturating_sub(1);
            }
        }

        // Handle en passant capture
        let mut is_ep_capture = false;
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &self.en_passant {
                if to_x == ep.square.x && to_y == ep.square.y {
                    if let Some(captured_pawn) = self
                        .board
                        .remove_piece(&ep.pawn_square.x, &ep.pawn_square.y)
                    {
                        is_ep_capture = true;
                        // Update spatial indices for EP captured pawn
                        self.spatial_indices
                            .remove(ep.pawn_square.x, ep.pawn_square.y);

                        let value = get_piece_value(captured_pawn.piece_type);
                        if captured_pawn.color == PlayerColor::White {
                            self.material_score -= value;
                            self.white_piece_count = self.white_piece_count.saturating_sub(1);
                        } else {
                            self.material_score += value;
                            self.black_piece_count = self.black_piece_count.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // Handle promotion material
        if let Some(promo_str) = promotion {
            let pawn_val = get_piece_value(PieceType::Pawn);
            if piece.color == PlayerColor::White {
                self.material_score -= pawn_val;
            } else {
                self.material_score += pawn_val;
            }

            let promo_type = PieceType::from_str(promo_str).unwrap_or(PieceType::Queen);
            let promo_val = get_piece_value(promo_type);
            if piece.color == PlayerColor::White {
                self.material_score += promo_val;
            } else {
                self.material_score -= promo_val;
            }
        }

        // Update special rights - moving piece loses its rights
        self.special_rights.remove(&Coordinate::new(from_x, from_y));
        // Captured piece (if any) loses its rights
        if is_capture {
            self.special_rights.remove(&Coordinate::new(to_x, to_y));
        }

        // Handle castling (king moves more than 1 square horizontally)
        if piece.piece_type == PieceType::King || piece.piece_type == PieceType::RoyalCentaur {
            let dx = to_x - from_x;
            if dx.abs() > 1 {
                // Find the rook BEYOND the king's destination (rook is outside the path)
                // e.g., kingside: king 5,1->7,1, rook at 8,1 moves to 6,1
                let rook_dir = if dx > 0 { 1 } else { -1 };
                let mut rook_x = to_x + rook_dir; // Start searching past king's destination
                while rook_x >= -1_000_000 && rook_x <= 1_000_000 {
                    if let Some(r) = self.board.get_piece(&rook_x, &from_y) {
                        if r.piece_type == PieceType::Rook && r.color == piece.color {
                            // Found the rook - move it to the square the king jumped over
                            let rook = self.board.remove_piece(&rook_x, &from_y).unwrap();
                            let rook_to_x = to_x - rook_dir; // Rook goes on the other side of king
                            self.board.set_piece(rook_to_x, from_y, rook);
                            self.special_rights.remove(&Coordinate::new(rook_x, from_y));
                            break;
                        }
                        break; // Hit a non-rook piece, stop searching
                    }
                    rook_x += rook_dir;
                }
            }
        }

        // Place the piece (with promotion if applicable)
        let final_piece = if let Some(promo_str) = promotion {
            if let Some(promo_type) = PieceType::from_str(promo_str) {
                Piece::new(promo_type, piece.color)
            } else {
                piece.clone()
            }
        } else {
            piece.clone()
        };
        self.board.set_piece(to_x, to_y, final_piece);
        // Update spatial indices for piece on destination square
        self.spatial_indices.add(to_x, to_y);

        // Update en passant state
        self.en_passant = None;
        if piece.piece_type == PieceType::Pawn {
            let dy = to_y - from_y;
            if dy.abs() == 2 {
                let ep_y = from_y + (dy / 2);
                self.en_passant = Some(EnPassantState {
                    square: Coordinate::new(from_x, ep_y),
                    pawn_square: Coordinate::new(to_x, to_y),
                });
            }
        }

        // Update clocks
        if piece.piece_type == PieceType::Pawn || is_capture || is_ep_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if self.turn == PlayerColor::Black {
            self.fullmove_number += 1;
        }

        self.turn = self.turn.opponent();
    }

    pub fn make_move(&mut self, m: &Move) -> UndoMove {
        use crate::search::zobrist::{en_passant_key, piece_key, special_right_key, SIDE_KEY};

        // Push current position hash BEFORE making the move (for repetition detection)
        self.hash_stack.push(self.hash);

        let from_coord = Coordinate::new(m.from.x, m.from.y);

        let piece = self.board.remove_piece(&m.from.x, &m.from.y).unwrap();
        // Update spatial indices: remove moving piece from source square
        self.spatial_indices.remove(m.from.x, m.from.y);

        // Hash: remove piece from source
        self.hash ^= piece_key(piece.piece_type, piece.color, m.from.x, m.from.y);

        let mut undo_info = UndoMove {
            captured_piece: self.board.get_piece(&m.to.x, &m.to.y).copied(),
            old_en_passant: self.en_passant.clone(),
            old_halfmove_clock: self.halfmove_clock,
            old_hash: self.hash_stack.last().copied().unwrap_or(0), // Save original hash
            special_rights_removed: Vec::new(),
            starting_square_restored: None,
        };

        // Once a piece moves from its original square, we no longer treat
        // that coordinate as an undeveloped starting square. Record this so
        // undo_move can restore starting_squares.
        if self.starting_squares.remove(&from_coord) {
            undo_info.starting_square_restored = Some(from_coord);
        }

        // Handle captures
        let is_capture = undo_info.captured_piece.is_some();

        if let Some(captured) = &undo_info.captured_piece {
            // Hash: remove captured piece
            self.hash ^= piece_key(captured.piece_type, captured.color, m.to.x, m.to.y);
            // Update spatial indices for captured piece on destination square
            self.spatial_indices.remove(m.to.x, m.to.y);

            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score -= value;
                self.white_piece_count = self.white_piece_count.saturating_sub(1);
            } else {
                self.material_score += value;
                self.black_piece_count = self.black_piece_count.saturating_sub(1);
            }
        }

        // Handle En Passant capture
        let mut is_ep_capture = false;
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &self.en_passant {
                if m.to.x == ep.square.x && m.to.y == ep.square.y {
                    if let Some(captured_pawn) = self
                        .board
                        .remove_piece(&ep.pawn_square.x, &ep.pawn_square.y)
                    {
                        is_ep_capture = true;
                        // Hash: remove EP captured pawn
                        self.hash ^= piece_key(
                            captured_pawn.piece_type,
                            captured_pawn.color,
                            ep.pawn_square.x,
                            ep.pawn_square.y,
                        );
                        // Update spatial indices for EP captured pawn
                        self.spatial_indices
                            .remove(ep.pawn_square.x, ep.pawn_square.y);

                        let value = get_piece_value(captured_pawn.piece_type);
                        if captured_pawn.color == PlayerColor::White {
                            self.material_score -= value;
                            self.white_piece_count = self.white_piece_count.saturating_sub(1);
                        } else {
                            self.material_score += value;
                            self.black_piece_count = self.black_piece_count.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // Handle Promotion material update
        if let Some(promo_type) = m.promotion {
            let pawn_val = get_piece_value(PieceType::Pawn);
            let promo_val = get_piece_value(promo_type);
            if piece.color == PlayerColor::White {
                self.material_score -= pawn_val;
                self.material_score += promo_val;
            } else {
                self.material_score += pawn_val;
                self.material_score -= promo_val;
            }
        }

        // Hash: remove old en passant
        if let Some(ep) = &self.en_passant {
            self.hash ^= en_passant_key(ep.square.x, ep.square.y);
        }

        // Update special rights (hash update before removal)
        if self.special_rights.remove(&m.from) {
            self.hash ^= special_right_key(&m.from);
            undo_info.special_rights_removed.push(m.from);
        }
        if is_capture && self.special_rights.remove(&m.to) {
            self.hash ^= special_right_key(&m.to);
            undo_info.special_rights_removed.push(m.to);
        }

        // Handle Castling Move (King moves > 1 square)
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                if let Some(rook_coord) = &m.rook_coord {
                    if let Some(rook) = self.board.remove_piece(&rook_coord.x, &rook_coord.y) {
                        let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                        // Hash: remove rook from original, add at new position
                        self.hash ^=
                            piece_key(rook.piece_type, rook.color, rook_coord.x, rook_coord.y);
                        self.hash ^= piece_key(rook.piece_type, rook.color, rook_to_x, m.from.y);
                        self.board.set_piece(rook_to_x, m.from.y, rook);
                        // Update spatial indices for rook move
                        self.spatial_indices.remove(rook_coord.x, rook_coord.y);
                        self.spatial_indices.add(rook_to_x, m.from.y);

                        // Rook also loses special rights
                        if self.special_rights.remove(rook_coord) {
                            self.hash ^= special_right_key(rook_coord);
                            undo_info.special_rights_removed.push(*rook_coord);
                        }
                    }
                }
            }
        }

        // Move piece (handle promotion if needed)
        let final_piece = if let Some(promo_type) = m.promotion {
            Piece::new(promo_type, piece.color)
        } else {
            piece
        };

        // Hash: add piece at destination
        self.hash ^= piece_key(final_piece.piece_type, final_piece.color, m.to.x, m.to.y);
        self.board.set_piece(m.to.x, m.to.y, final_piece);
        // Update spatial indices for moved piece on destination square
        self.spatial_indices.add(m.to.x, m.to.y);

        // Update En Passant state
        self.en_passant = None;
        if piece.piece_type == PieceType::Pawn {
            let dy = m.to.y - m.from.y;
            if dy.abs() == 2 {
                let ep_y = m.from.y + (dy / 2);
                self.en_passant = Some(EnPassantState {
                    square: Coordinate::new(m.from.x, ep_y),
                    pawn_square: m.to,
                });
                // Hash: add new en passant
                self.hash ^= en_passant_key(m.from.x, ep_y);
            }
        }

        // Update clocks
        if piece.piece_type == PieceType::Pawn || is_capture || is_ep_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if self.turn == PlayerColor::Black {
            self.fullmove_number += 1;
        }

        // Hash: flip side to move
        self.hash ^= SIDE_KEY;
        self.turn = self.turn.opponent();

        undo_info
    }

    pub fn undo_move(&mut self, m: &Move, undo: UndoMove) {
        // Pop the hash that was pushed in make_move and restore the saved hash
        self.hash_stack.pop();
        self.hash = undo.old_hash;

        // Revert turn
        self.turn = self.turn.opponent();

        if self.turn == PlayerColor::Black {
            self.fullmove_number -= 1;
        }

        // Revert piece move
        // Get the piece from the 'to' square
        let mut piece = self.board.remove_piece(&m.to.x, &m.to.y).unwrap();
        // Update spatial indices: remove piece from destination square
        self.spatial_indices.remove(m.to.x, m.to.y);

        // Handle Promotion Revert
        if m.promotion.is_some() {
            // Convert back to pawn
            let promo_val = get_piece_value(piece.piece_type);
            let pawn_val = get_piece_value(PieceType::Pawn);

            if piece.color == PlayerColor::White {
                self.material_score -= promo_val;
                self.material_score += pawn_val;
            } else {
                self.material_score += promo_val;
                self.material_score -= pawn_val;
            }
            piece.piece_type = PieceType::Pawn;
        }

        // Move back to 'from'
        self.board.set_piece(m.from.x, m.from.y, piece);
        // Update spatial indices for moved piece back on source square
        self.spatial_indices.add(m.from.x, m.from.y);

        // Restore captured piece
        if let Some(captured) = undo.captured_piece {
            let value = get_piece_value(captured.piece_type);
            if captured.color == PlayerColor::White {
                self.material_score += value;
                self.white_piece_count = self.white_piece_count.saturating_add(1);
            } else {
                self.material_score -= value;
                self.black_piece_count = self.black_piece_count.saturating_add(1);
            }
            self.board.set_piece(m.to.x, m.to.y, captured);
            // Update spatial indices for restored captured piece
            self.spatial_indices.add(m.to.x, m.to.y);
        }

        // Handle En Passant Capture Revert
        // If it was an EP capture, the captured pawn was on 'pawn_square' of the OLD en_passant state
        // But wait, we don't store "is_ep_capture" in UndoMove.
        // We can infer it: if piece is pawn, and to_square matches old_ep.square
        if piece.piece_type == PieceType::Pawn {
            if let Some(ep) = &undo.old_en_passant {
                if m.to.x == ep.square.x && m.to.y == ep.square.y {
                    // It was an EP capture!
                    // Restore the captured pawn
                    let captured_pawn = Piece::new(PieceType::Pawn, piece.color.opponent());

                    self.board
                        .set_piece(ep.pawn_square.x, ep.pawn_square.y, captured_pawn);
                    // Update spatial indices for restored EP pawn
                    self.spatial_indices.add(ep.pawn_square.x, ep.pawn_square.y);

                    // Restore material
                    let value = get_piece_value(PieceType::Pawn);
                    if captured_pawn.color == PlayerColor::White {
                        self.material_score += value;
                        self.white_piece_count = self.white_piece_count.saturating_add(1);
                    } else {
                        self.material_score -= value;
                        self.black_piece_count = self.black_piece_count.saturating_add(1);
                    }
                }
            }
        }

        // Handle Castling Revert
        if piece.piece_type == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling was performed. Move rook back.
                if let Some(rook_coord) = &m.rook_coord {
                    let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                    if let Some(rook) = self.board.remove_piece(&rook_to_x, &m.from.y) {
                        self.board.set_piece(rook_coord.x, rook_coord.y, rook);
                        // Update spatial indices for rook moved back
                        self.spatial_indices.remove(rook_to_x, m.from.y);
                        self.spatial_indices.add(rook_coord.x, rook_coord.y);
                    }
                }
            }
        }

        // Restore state
        self.en_passant = undo.old_en_passant;
        // Re-insert removed special rights instead of restoring entire HashSet
        for coord in undo.special_rights_removed {
            self.special_rights.insert(coord);
        }
        // If this move caused a piece to leave its original starting square,
        // restore that coordinate in starting_squares.
        if let Some(coord) = undo.starting_square_restored {
            self.starting_squares.insert(coord);
        }
        self.halfmove_clock = undo.old_halfmove_clock;
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.get_legal_moves();
        let mut nodes = 0;

        for m in moves {
            let undo = self.make_move(&m);
            nodes += self.perft(depth - 1);
            self.undo_move(&m, undo);
        }

        nodes
    }

    pub fn setup_position_from_icn(&mut self, position_icn: &str) {
        self.board = Board::new();
        self.special_rights.clear();
        self.en_passant = None;
        self.turn = PlayerColor::White;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;
        self.material_score = 0;

        // Parse ICN format: "PieceType,x,y|PieceType,x,y|..."
        // Example: "P1,2|r2,3|K4,5" where:
        // - P = white pawn at (1,2)
        // - r = black rook at (2,3)
        // - K = white king at (4,5)
        // Optional + after piece indicates special rights: "P1,2+|r2,3+"
        for piece_str in position_icn.split('|') {
            if piece_str.is_empty() {
                continue;
            }

            // Split into piece_info and coordinates: "P1,2" -> ["P1", "2"]
            let parts: Vec<&str> = piece_str.split(',').collect();
            if parts.len() != 2 {
                continue; // Skip invalid pieces
            }

            let (piece_info, y_str) = (parts[0], parts[1]);

            // Extract piece type and x coordinate from piece_info: "P1" -> ('P', '1')
            let mut chars = piece_info.chars();
            let piece_char = chars.next();
            let x_str: String = chars.collect();

            if piece_char.is_none() {
                continue;
            }

            let x: i64 = x_str.parse().unwrap_or(0);
            let y: i64 = y_str.parse().unwrap_or(0);

            // Extract piece type and check for special rights
            let piece_char = piece_char.unwrap();
            let (actual_piece_char, has_special_rights) = if x_str.ends_with('+') {
                // Format like "P1+,2" - special rights indicated
                let _clean_x = &x_str[..x_str.len() - 1]; // Prefix with underscore to indicate unused
                (piece_char, true)
            } else {
                (piece_char, false)
            };

            let is_white = actual_piece_char.is_uppercase();
            let piece_type = match actual_piece_char.to_ascii_lowercase() {
                'k' => PieceType::King,
                'q' => PieceType::Queen,
                'r' => PieceType::Rook,
                'b' => PieceType::Bishop,
                'n' => PieceType::Knight,
                'p' => PieceType::Pawn,
                // Extended pieces for variants
                'a' => PieceType::Amazon,
                'c' => PieceType::Chancellor,
                'h' => PieceType::Archbishop,
                'v' => PieceType::Void,
                'x' => PieceType::Obstacle,
                'g' => PieceType::Giraffe,
                'l' => PieceType::Camel, // 'l' for camel ( avoid 'c' conflict with chancellor
                'z' => PieceType::Zebra,
                'm' => PieceType::Knightrider, // 'm' for knightrider
                _ => continue,                 // Skip unknown piece types
            };

            let color = if is_white {
                PlayerColor::White
            } else {
                PlayerColor::Black
            };
            let piece = Piece::new(piece_type, color);

            // Use the cleaned x coordinate if we had special rights
            let final_x = if has_special_rights && x_str.ends_with('+') {
                let clean_x = &x_str[..x_str.len() - 1];
                clean_x.parse().unwrap_or(x)
            } else {
                x
            };

            self.board.set_piece(final_x, y, piece);

            // Add special rights if indicated by +
            if has_special_rights {
                // coord_key was unused, removing the format string since we only need the coordinate
                self.special_rights.insert(Coordinate::new(final_x, y));
            }
        }

        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);

        // Rebuild piece lists and counts
        self.recompute_piece_counts();

        // Compute initial hash
        self.recompute_hash();
    }

    pub fn setup_standard_chess(&mut self) {
        self.board = Board::new();
        self.special_rights.clear();
        self.en_passant = None;
        self.turn = PlayerColor::White;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;
        self.material_score = 0;

        // White Pieces
        self.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        self.board
            .set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board
            .set_piece(3, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board
            .set_piece(4, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        self.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        self.board
            .set_piece(6, 1, Piece::new(PieceType::Bishop, PlayerColor::White));
        self.board
            .set_piece(7, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        self.board
            .set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));

        for x in 1..=8 {
            self.board
                .set_piece(x, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        }

        // Black Pieces
        self.board
            .set_piece(1, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        self.board
            .set_piece(2, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board
            .set_piece(3, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board
            .set_piece(4, 8, Piece::new(PieceType::Queen, PlayerColor::Black));
        self.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        self.board
            .set_piece(6, 8, Piece::new(PieceType::Bishop, PlayerColor::Black));
        self.board
            .set_piece(7, 8, Piece::new(PieceType::Knight, PlayerColor::Black));
        self.board
            .set_piece(8, 8, Piece::new(PieceType::Rook, PlayerColor::Black));

        for x in 1..=8 {
            self.board
                .set_piece(x, 7, Piece::new(PieceType::Pawn, PlayerColor::Black));
        }

        // Special Rights - Kings, Rooks (castling) and Pawns (double move)
        self.special_rights.insert(Coordinate::new(1, 1)); // Rook
        self.special_rights.insert(Coordinate::new(5, 1)); // King
        self.special_rights.insert(Coordinate::new(8, 1)); // Rook

        self.special_rights.insert(Coordinate::new(1, 8)); // Rook
        self.special_rights.insert(Coordinate::new(5, 8)); // King
        self.special_rights.insert(Coordinate::new(8, 8)); // Rook

        // Pawn double-move rights
        for x in 1..=8 {
            self.special_rights.insert(Coordinate::new(x, 2)); // White pawns
            self.special_rights.insert(Coordinate::new(x, 7)); // Black pawns
        }

        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);

        // Rebuild piece lists and counts
        self.recompute_piece_counts();

        // Compute initial hash
        self.recompute_hash();
    }
}
