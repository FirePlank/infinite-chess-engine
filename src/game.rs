use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::evaluation::{calculate_initial_material, get_piece_value};
use crate::moves::{
    Move, MoveList, SpatialIndices, get_legal_moves, get_legal_moves_into,
    get_pseudo_legal_moves_for_piece_into, is_square_attacked,
};
use crate::utils::{is_prime_fast, is_prime_i64};
use arrayvec::ArrayVec;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

/// Win conditions for a player. Determines how they win the game.
/// - Checkmate: Standard - win by checkmating the opponent
/// - RoyalCapture: Win by capturing the opponent's only royal piece
/// - AllRoyalsCaptured: Win when all opponent's royal pieces are captured
/// - AllPiecesCaptured: Win when all opponent's pieces are captured
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WinCondition {
    #[default]
    Checkmate,
    RoyalCapture,
    AllRoyalsCaptured,
    AllPiecesCaptured,
}

impl std::str::FromStr for WinCondition {
    type Err = ();

    /// Parse a win condition from a string (as received from JS).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "checkmate" => Ok(WinCondition::Checkmate),
            "royalcapture" => Ok(WinCondition::RoyalCapture),
            "allroyalscaptured" => Ok(WinCondition::AllRoyalsCaptured),
            "allpiecescaptured" => Ok(WinCondition::AllPiecesCaptured),
            _ => Err(()),
        }
    }
}

impl WinCondition {
    /// Returns true if this win condition requires the opponent to respond to check.
    /// For Checkmate, checks must be addressed. For capture-based conditions, king can be taken.
    #[inline]
    pub fn requires_check_evasion(&self) -> bool {
        matches!(self, WinCondition::Checkmate)
    }

    /// Returns true if this win condition is based on capturing royal pieces.
    /// Used to gate royal capture checks for zero overhead on standard chess.
    #[inline]
    pub fn is_royal_capture_based(&self) -> bool {
        matches!(
            self,
            WinCondition::RoyalCapture | WinCondition::AllRoyalsCaptured
        )
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EnPassantState {
    pub square: Coordinate,
    pub pawn_square: Coordinate,
}

/// Promotion ranks configuration for a variant
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromotionRanks {
    pub white: Vec<i64>,
    pub black: Vec<i64>,
}

/// Game rules that can vary between chess variants
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GameRules {
    pub promotion_ranks: Option<PromotionRanks>,
    #[serde(skip)]
    pub promotion_types: Option<Vec<PieceType>>, // Pre-converted promotion piece types (fast)
    pub promotions_allowed: Option<Vec<String>>, // Piece type codes (only for serialization)
    pub move_rule_limit: Option<u32>,            // 50-move rule limit in halfmoves (default 100)
    /// Win condition for White: what Black must do to defeat White.
    /// E.g., "checkmate" means Black must checkmate White to win.
    #[serde(skip)]
    pub white_win_condition: WinCondition,
    /// Win condition for Black: what White must do to defeat Black.
    /// E.g., "allpiecescaptured" means White must capture all of Black's pieces to win.
    #[serde(skip)]
    pub black_win_condition: WinCondition,
}

impl GameRules {
    /// Convert promotions_allowed strings to PieceTypes once
    pub fn init_promotion_types(&mut self) {
        if let Some(ref allowed) = self.promotions_allowed {
            self.promotion_types = Some(
                allowed
                    .iter()
                    .filter_map(|s| s.parse::<PieceType>().ok())
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
    pub old_hash: u64, // Hash before the move was made
    pub special_rights_removed: ArrayVec<Coordinate, 4>, // Track which special rights were removed (re-insert on undo)
    /// If this move caused a piece to leave its original starting square,
    /// we remove that coordinate from starting_squares. Store it here so
    /// undo_move can restore starting_squares exactly.
    pub starting_square_restored: Option<Coordinate>,
    /// King positions before move (only set if piece was a king or royal piece)
    pub old_white_king_pos: Option<Coordinate>,
    pub old_black_king_pos: Option<Coordinate>,
    /// Old repetition value for restoration
    pub old_repetition: i32,
    /// Piece captured via en passant (could be a promoted piece, not just a pawn)
    pub ep_captured_piece: Option<Piece>,
    /// Incremental castling state for O(1) restoration
    pub old_effective_castling_rights: u8,
    pub old_castling_partner_counts: [u16; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Board,
    pub turn: PlayerColor,
    /// Special rights for pieces - includes both castling rights (kings/rooks) AND
    /// pawn double-move rights. A piece with its coordinate in this set has its special rights.
    pub special_rights: FxHashSet<Coordinate>,
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
    #[serde(skip)]
    pub white_pawn_count: u16,
    #[serde(skip)]
    pub black_pawn_count: u16,
    /// Starting piece counts (non-pawn) for game phase calculation
    /// Set once when game is initialized, never changes
    #[serde(skip)]
    pub starting_white_pieces: u16,
    #[serde(skip)]
    pub starting_black_pieces: u16,
    /// Piece coordinates per color for fast iteration
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
    pub starting_squares: FxHashSet<Coordinate>,
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
    /// Cached king positions for O(1) lookup. Updated incrementally in make/undo.
    #[serde(skip)]
    pub white_king_pos: Option<Coordinate>,
    #[serde(skip)]
    pub black_king_pos: Option<Coordinate>,
    /// Precomputed check squares for white king (squares from which enemy pieces give check)
    /// Uses hash for O(1) lookup. Stores (x, y, piece_type) as key.
    #[serde(skip)]
    pub check_squares_white: rustc_hash::FxHashSet<(i64, i64, u8)>,
    /// Precomputed check squares for black king
    #[serde(skip)]
    pub check_squares_black: rustc_hash::FxHashSet<(i64, i64, u8)>,
    /// Slider rays from white king: [direction_index] -> Option<(blocker_x, blocker_y)>
    /// Direction indices: 0=N, 1=S, 2=E, 3=W, 4=NE, 5=NW, 6=SE, 7=SW
    /// None = infinite ray (no blocker), Some = first blocker position
    #[serde(skip)]
    pub slider_rays_white: [Option<(i64, i64)>; 8],
    /// Slider rays from black king
    #[serde(skip)]
    pub slider_rays_black: [Option<(i64, i64)>; 8],

    /// Squares from which a piece move discovers a check on the enemy king.
    /// Stores (x, y) coordinates of the potentially blocking piece.
    #[serde(skip)]
    pub discovered_check_squares_white: FxHashSet<(i64, i64)>,
    #[serde(skip)]
    pub discovered_check_squares_black: FxHashSet<(i64, i64)>,
    /// Pawn structure hash for correction history (helps CoaIP variants).
    #[serde(skip)]
    pub pawn_hash: u64,
    /// Non-pawn piece position hash for correction history.
    #[serde(skip)]
    pub nonpawn_hash: u64,
    /// Material configuration hash for correction history.
    #[serde(skip)]
    pub material_hash: u64,
    /// Stockfish-style repetition info: distance to previous occurrence of same position.
    /// 0 = no repetition, positive = distance to first occurrence, negative = threefold.
    /// Computed during make_move for O(1) is_repetition check.
    #[serde(skip)]
    pub repetition: i32,
    /// Cached effective castling rights bitfield: 0=WKS, 1=WQS, 2=BKS, 3=BQS
    #[serde(skip)]
    pub effective_castling_rights: u8,
    /// Number of castling partners for each color/direction that still have special rights.
    /// Indexed by: 0=WKS, 1=WQS, 2=BKS, 3=BQS
    #[serde(skip)]
    pub castling_partner_counts: [u16; 4],
    /// Fast non-pawn material flags for NMP zugzwang detection.
    /// True if the side has at least one non-pawn, non-king piece.
    /// Updated incrementally in make_move/undo_move.
    #[serde(skip)]
    pub white_non_pawn_material: bool,
    #[serde(skip)]
    pub black_non_pawn_material: bool,
}

// For backwards compatibility, keep castling_rights as an alias
impl GameState {
    /// Returns pieces that can castle (kings/royals and any non-pawn partner with special rights)
    pub fn castling_rights(&self) -> FxHashSet<Coordinate> {
        let mut rights = FxHashSet::default();
        for coord in &self.special_rights {
            if let Some(piece) = self.board.get_piece(coord.x, coord.y) {
                // Include royals (kings) and any non-pawn piece as potential castling partners
                // This matches the move generation logic which accepts any non-pawn, non-royal piece
                if piece.piece_type().is_royal() || piece.piece_type() != PieceType::Pawn {
                    rights.insert(*coord);
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

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        // crate::tiles::magic::init();

        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: FxHashSet::default(),
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
            white_pawn_count: 0,
            black_pawn_count: 0,
            starting_white_pieces: 0,
            starting_black_pieces: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
            starting_squares: FxHashSet::default(),
            white_back_rank: 1,
            black_back_rank: 8,
            white_promo_rank: 2_000_000_000_000_000,
            black_promo_rank: -2_000_000_000_000_000,
            white_king_pos: None,
            black_king_pos: None,
            check_squares_white: FxHashSet::default(),
            check_squares_black: FxHashSet::default(),
            slider_rays_white: [None; 8],
            slider_rays_black: [None; 8],
            discovered_check_squares_white: FxHashSet::default(),
            discovered_check_squares_black: FxHashSet::default(),
            pawn_hash: 0,
            nonpawn_hash: 0,
            material_hash: 0,
            repetition: 0,
            white_non_pawn_material: false,
            black_non_pawn_material: false,
            effective_castling_rights: 0,
            castling_partner_counts: [0; 4],
        }
    }

    pub fn new_with_rules(game_rules: GameRules) -> Self {
        // crate::tiles::magic::init();

        GameState {
            board: Board::new(),
            turn: PlayerColor::White,
            special_rights: FxHashSet::default(),
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
            white_pawn_count: 0,
            black_pawn_count: 0,
            starting_white_pieces: 0,
            starting_black_pieces: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
            starting_squares: FxHashSet::default(),
            white_back_rank: 1,
            black_back_rank: 8,
            white_promo_rank: 2_000_000_000_000_000,
            black_promo_rank: -2_000_000_000_000_000,
            white_king_pos: None,
            black_king_pos: None,
            check_squares_white: FxHashSet::default(),
            check_squares_black: FxHashSet::default(),
            slider_rays_white: [None; 8],
            slider_rays_black: [None; 8],
            discovered_check_squares_white: FxHashSet::default(),
            discovered_check_squares_black: FxHashSet::default(),
            pawn_hash: 0,
            nonpawn_hash: 0,
            material_hash: 0,
            repetition: 0,
            white_non_pawn_material: false,
            black_non_pawn_material: false,
            effective_castling_rights: 0,
            castling_partner_counts: [0; 4],
        }
    }

    /// Recompute piece counts, rebuild piece lists, and find king positions from the board
    /// Recompute piece counts, rebuild piece lists, and find king positions from the board
    pub fn recompute_piece_counts(&mut self) {
        let mut white: u16 = 0;
        let mut black: u16 = 0;
        let mut white_pawns: u16 = 0;
        let mut black_pawns: u16 = 0;
        let mut white_npm = false;
        let mut black_npm = false;
        self.white_pieces.clear();
        self.black_pieces.clear();
        self.white_king_pos = None;
        self.black_king_pos = None;

        if let Some(active) = &self.board.active_coords {
            for (x, y) in active {
                let piece = match self.board.get_piece(*x, *y) {
                    Some(p) => p,
                    None => continue,
                };
                // Track king positions (any royal piece)
                if piece.piece_type().is_royal() {
                    if piece.color() == PlayerColor::White {
                        self.white_king_pos = Some(Coordinate::new(*x, *y));
                    } else if piece.color() == PlayerColor::Black {
                        self.black_king_pos = Some(Coordinate::new(*x, *y));
                    }
                }
                match piece.color() {
                    PlayerColor::White => {
                        white = white.saturating_add(1);
                        self.white_pieces.push((*x, *y));
                        // Track pawns and non-pawn material
                        if piece.piece_type() == PieceType::Pawn {
                            white_pawns += 1;
                        } else if !piece.piece_type().is_royal() {
                            white_npm = true;
                        }
                    }
                    PlayerColor::Black => {
                        black = black.saturating_add(1);
                        self.black_pieces.push((*x, *y));
                        // Track pawns and non-pawn material
                        if piece.piece_type() == PieceType::Pawn {
                            black_pawns += 1;
                        } else if !piece.piece_type().is_royal() {
                            black_npm = true;
                        }
                    }
                    PlayerColor::Neutral => {}
                }
            }
        } else {
            for ((x, y), piece) in self.board.iter() {
                // Track king positions (any royal piece)
                if piece.piece_type().is_royal() {
                    if piece.color() == PlayerColor::White {
                        self.white_king_pos = Some(Coordinate::new(*x, *y));
                    } else if piece.color() == PlayerColor::Black {
                        self.black_king_pos = Some(Coordinate::new(*x, *y));
                    }
                }
                match piece.color() {
                    PlayerColor::White => {
                        white = white.saturating_add(1);
                        self.white_pieces.push((*x, *y));
                        // Track pawns and non-pawn material
                        if piece.piece_type() == PieceType::Pawn {
                            white_pawns += 1;
                        } else if !piece.piece_type().is_royal() {
                            white_npm = true;
                        }
                    }
                    PlayerColor::Black => {
                        black = black.saturating_add(1);
                        self.black_pieces.push((*x, *y));
                        // Track pawns and non-pawn material
                        if piece.piece_type() == PieceType::Pawn {
                            black_pawns += 1;
                        } else if !piece.piece_type().is_royal() {
                            black_npm = true;
                        }
                    }
                    PlayerColor::Neutral => {}
                }
            }
        }
        self.white_piece_count = white;
        self.black_piece_count = black;
        self.white_pawn_count = white_pawns;
        self.black_pawn_count = black_pawns;
        self.white_non_pawn_material = white_npm;
        self.black_non_pawn_material = black_npm;

        // Reset and recompute initial effective castling rights and partner counts
        self.castling_partner_counts = [0; 4];
        self.effective_castling_rights = 0;

        // Find white castling partners with rights
        if let Some(wk_pos) = self.white_king_pos {
            let wk_has_rights = self.special_rights.contains(&wk_pos);
            for coord in &self.special_rights {
                if coord.y != wk_pos.y || coord.x == wk_pos.x {
                    continue;
                }
                if let Some(piece) = self.board.get_piece(coord.x, coord.y) {
                    if piece.color() == PlayerColor::White
                        && piece.piece_type() != PieceType::Pawn
                        && !piece.piece_type().is_royal()
                    {
                        if coord.x > wk_pos.x {
                            self.castling_partner_counts[0] += 1;
                        } else {
                            self.castling_partner_counts[1] += 1;
                        }
                    }
                }
            }
            if wk_has_rights {
                if self.castling_partner_counts[0] > 0 {
                    self.effective_castling_rights |= 1;
                }
                if self.castling_partner_counts[1] > 0 {
                    self.effective_castling_rights |= 2;
                }
            }
        }

        // Find black castling partners with rights
        if let Some(bk_pos) = self.black_king_pos {
            let bk_has_rights = self.special_rights.contains(&bk_pos);
            for coord in &self.special_rights {
                if coord.y != bk_pos.y || coord.x == bk_pos.x {
                    continue;
                }
                if let Some(piece) = self.board.get_piece(coord.x, coord.y) {
                    if piece.color() == PlayerColor::Black
                        && piece.piece_type() != PieceType::Pawn
                        && !piece.piece_type().is_royal()
                    {
                        if coord.x > bk_pos.x {
                            self.castling_partner_counts[2] += 1;
                        } else {
                            self.castling_partner_counts[3] += 1;
                        }
                    }
                }
            }
            if bk_has_rights {
                if self.castling_partner_counts[2] > 0 {
                    self.effective_castling_rights |= 4;
                }
                if self.castling_partner_counts[3] > 0 {
                    self.effective_castling_rights |= 8;
                }
            }
        }
        // Rebuild spatial indices from current board
        self.spatial_indices = SpatialIndices::new(&self.board);
        // Recompute check squares for O(1) check detection
        self.recompute_check_squares();
    }

    /// Precompute check squares for both kings.
    /// For each king, stores the (x, y, piece_type) tuples for squares from which
    /// knights and pawns can give check. Also computes slider rays for O(1) slider check.
    #[inline]
    pub fn recompute_check_squares(&mut self) {
        // Knight offsets
        const KNIGHT_OFFSETS: [(i64, i64); 8] = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ];

        // 8 directions for slider rays: N, S, E, W, NE, NW, SE, SW
        const DIRECTIONS: [(i64, i64); 8] = [
            (0, 1),   // N (index 0)
            (0, -1),  // S (index 1)
            (1, 0),   // E (index 2)
            (-1, 0),  // W (index 3)
            (1, 1),   // NE (index 4)
            (-1, 1),  // NW (index 5)
            (1, -1),  // SE (index 6)
            (-1, -1), // SW (index 7)
        ];

        self.check_squares_white.clear();
        self.check_squares_black.clear();
        self.slider_rays_white = [None; 8];
        self.slider_rays_black = [None; 8];
        self.discovered_check_squares_white.clear();
        self.discovered_check_squares_black.clear();

        // White King Rays (squares from which Black can check White)
        if let Some(ref wk) = self.white_king_pos {
            for (dx, dy) in KNIGHT_OFFSETS {
                self.check_squares_white
                    .insert((wk.x + dx, wk.y + dy, PieceType::Knight as u8));
            }
            // Black pawns attack downward
            self.check_squares_white
                .insert((wk.x - 1, wk.y + 1, PieceType::Pawn as u8));
            self.check_squares_white
                .insert((wk.x + 1, wk.y + 1, PieceType::Pawn as u8));

            for (dir_idx, (dx, dy)) in DIRECTIONS.iter().enumerate() {
                if let Some((bx, by)) = self.find_first_blocker_on_ray(wk.x, wk.y, *dx, *dy) {
                    self.slider_rays_white[dir_idx] = Some((bx, by));
                    // Discovered check: if bx,by is a BLACK piece, does it block a BLACK slider?
                    if let Some(p2) = self
                        .board
                        .get_piece(bx, by)
                        .filter(|p| p.color() == PlayerColor::Black)
                        .and_then(|_| self.find_first_blocker_on_ray(bx, by, *dx, *dy))
                        .and_then(|(bx2, by2)| self.board.get_piece(bx2, by2))
                        .filter(|p| p.color() == PlayerColor::Black)
                    {
                        let is_ortho = dir_idx < 4;
                        let pt2 = p2.piece_type();
                        use crate::attacks::{is_diag_slider, is_ortho_slider};
                        if (is_ortho && is_ortho_slider(pt2)) || (!is_ortho && is_diag_slider(pt2))
                        {
                            self.discovered_check_squares_black.insert((bx, by));
                        }
                    }
                }
            }
        }

        // Black King Rays (squares from which White can check Black)
        if let Some(ref bk) = self.black_king_pos {
            for (dx, dy) in KNIGHT_OFFSETS {
                self.check_squares_black
                    .insert((bk.x + dx, bk.y + dy, PieceType::Knight as u8));
            }
            // White pawns attack upward
            self.check_squares_black
                .insert((bk.x - 1, bk.y - 1, PieceType::Pawn as u8));
            self.check_squares_black
                .insert((bk.x + 1, bk.y - 1, PieceType::Pawn as u8));

            for (dir_idx, (dx, dy)) in DIRECTIONS.iter().enumerate() {
                if let Some((bx, by)) = self.find_first_blocker_on_ray(bk.x, bk.y, *dx, *dy) {
                    self.slider_rays_black[dir_idx] = Some((bx, by));
                    // Discovered check: if bx,by is a WHITE piece, does it block a WHITE slider?
                    if let Some(p2) = self
                        .board
                        .get_piece(bx, by)
                        .filter(|p| p.color() == PlayerColor::White)
                        .and_then(|_| self.find_first_blocker_on_ray(bx, by, *dx, *dy))
                        .and_then(|(bx2, by2)| self.board.get_piece(bx2, by2))
                        .filter(|p| p.color() == PlayerColor::White)
                    {
                        let is_ortho = dir_idx < 4;
                        let pt2 = p2.piece_type();
                        use crate::attacks::{is_diag_slider, is_ortho_slider};
                        if (is_ortho && is_ortho_slider(pt2)) || (!is_ortho && is_diag_slider(pt2))
                        {
                            self.discovered_check_squares_white.insert((bx, by));
                        }
                    }
                }
            }
        }
    }

    /// Find the first blocker on a ray from (start_x, start_y) in direction (dx, dy).
    /// Uses spatial indices for O(1) lookup per direction.
    /// Returns Some((x, y)) if a blocker exists, None if ray is infinite.
    #[inline]
    fn find_first_blocker_on_ray(
        &self,
        start_x: i64,
        start_y: i64,
        dx: i64,
        dy: i64,
    ) -> Option<(i64, i64)> {
        // Use spatial indices to find nearest piece in direction
        // SpatialIndices stores (coord, packed_piece) tuples sorted by coord
        // Field names: cols (x -> list of y), rows (y -> list of x),
        //              diag1 (x-y -> list of x), diag2 (x+y -> list of x)

        if dx == 0 {
            // Vertical ray (N or S) - use cols[start_x] to get all y coords
            if let Some(col_vec) = self.spatial_indices.cols.get(&start_x) {
                // Use find_nearest with direction (+1 or -1)
                if let Some((found_y, _packed)) = SpatialIndices::find_nearest(col_vec, start_y, dy)
                {
                    return Some((start_x, found_y));
                }
            }
        } else if dy == 0 {
            // Horizontal ray (E or W) - use rows[start_y] to get all x coords
            if let Some((found_x, _packed)) = self
                .spatial_indices
                .rows
                .get(&start_y)
                .and_then(|row_vec| SpatialIndices::find_nearest(row_vec, start_x, dx))
            {
                return Some((found_x, start_y));
            }
        } else {
            // Diagonal rays
            let diag_key = start_x - start_y;
            let anti_key = start_x + start_y;

            if dx == dy {
                // Main diagonal (NE or SW: dx == dy) - use diag1
                if let Some(diag_vec) = self.spatial_indices.diag1.get(&diag_key) {
                    // diag1 is indexed by x, so search for x in direction dx
                    if let Some((found_x, _packed)) =
                        SpatialIndices::find_nearest(diag_vec, start_x, dx)
                    {
                        // Reconstruct y from x: on main diagonal, x - y = diag_key
                        let found_y = found_x - diag_key;
                        return Some((found_x, found_y));
                    }
                }
            } else {
                // Anti-diagonal (NW or SE: dx != dy) - use diag2
                if let Some(anti_vec) = self.spatial_indices.diag2.get(&anti_key) {
                    // diag2 is indexed by x, search in direction dx
                    if let Some((found_x, _packed)) =
                        SpatialIndices::find_nearest(anti_vec, start_x, dx)
                    {
                        // Reconstruct y from x: on anti-diagonal, x + y = anti_key
                        let found_y = anti_key - found_x;
                        return Some((found_x, found_y));
                    }
                }
            }
        }
        None // Infinite ray
    }

    fn is_on_check_ray(
        &self,
        target: &Coordinate,
        king_sq: &Coordinate,
        step_x: i64,
        step_y: i64,
        check_dist: i64,
        checker_type: PieceType,
    ) -> bool {
        let dx = target.x - king_sq.x;
        let dy = target.y - king_sq.y;

        // For the target to be on the check ray, it must satisfy:
        // target = king + k * (step_x, step_y) for some k in 1..=check_dist
        // This means: dx = k * step_x and dy = k * step_y for the same k

        // Handle horizontal/vertical/diagonal rays
        let k = if step_x != 0 && step_y != 0 {
            // Diagonal: both dx and dy must give the same k
            if dx % step_x != 0 || dy % step_y != 0 {
                return false;
            }
            let kx = dx / step_x;
            let ky = dy / step_y;
            if kx != ky || kx <= 0 {
                return false;
            }
            kx
        } else if step_x != 0 {
            // Horizontal ray (step_y == 0): dy must be 0
            if dy != 0 || dx % step_x != 0 {
                return false;
            }
            let kx = dx / step_x;
            if kx <= 0 {
                return false;
            }
            kx
        } else if step_y != 0 {
            // Vertical ray (step_x == 0): dx must be 0
            if dx != 0 || dy % step_y != 0 {
                return false;
            }
            let ky = dy / step_y;
            if ky <= 0 {
                return false;
            }
            ky
        } else {
            // step_x == 0 && step_y == 0 shouldn't happen (not a valid ray)
            return false;
        };

        if k < 1 || k > check_dist {
            return false;
        }

        if checker_type == PieceType::Huygen {
            // A piece blocks a Huygen only if it lands on a prime distance from it
            let dist_from_checker = check_dist - k;
            crate::utils::is_prime_fast(dist_from_checker)
        } else {
            true
        }
    }

    /// Check if a rook/queen can move from `from` to `to` (orthogonal move).
    /// Uses spatial indices to find the first blocker on the ray.
    #[inline]
    fn is_path_clear_for_rook(&self, from: &Coordinate, to: &Coordinate) -> bool {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        // Must be orthogonal
        if dx != 0 && dy != 0 {
            return false;
        }
        let step_x = dx.signum();
        let step_y = dy.signum();
        let dist = dx.abs().max(dy.abs());
        if dist <= 1 {
            // Target is adjacent, check if occupied by friendly
            if let Some(p) = self.board.get_piece(to.x, to.y) {
                return p.color() != self.turn; // Can capture enemy
            }
            return true;
        }
        // Use find_first_blocker_on_ray to check path
        if let Some((bx, by)) = self.find_first_blocker_on_ray(from.x, from.y, step_x, step_y) {
            let blocker_dist = (bx - from.x).abs().max((by - from.y).abs());
            // Path is clear if blocker is at or beyond target (i.e., blocker_dist >= dist)
            if blocker_dist < dist {
                return false;
            }
            // If blocker is exactly at target, check if it's enemy (capture) or friendly
            if blocker_dist == dist
                && self
                    .board
                    .get_piece(bx, by)
                    .is_some_and(|p| p.color() != self.turn)
            {
                return true;
            }
            if blocker_dist == dist {
                return false;
            }
        }
        true
    }

    /// Check if a bishop/queen can move from `from` to `to` (diagonal move).
    /// Uses spatial indices to find the first blocker on the ray.
    #[inline]
    fn is_path_clear_for_bishop(&self, from: &Coordinate, to: &Coordinate) -> bool {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        // Must be diagonal
        if dx.abs() != dy.abs() || dx == 0 {
            return false;
        }
        let step_x = dx.signum();
        let step_y = dy.signum();
        let dist = dx.abs();
        if dist <= 1 {
            if let Some(p) = self.board.get_piece(to.x, to.y) {
                return p.color() != self.turn;
            }
            return true;
        }
        if let Some((bx, by)) = self.find_first_blocker_on_ray(from.x, from.y, step_x, step_y) {
            let blocker_dist = (bx - from.x).abs();
            if blocker_dist < dist {
                return false;
            }
            if blocker_dist == dist
                && self
                    .board
                    .get_piece(bx, by)
                    .is_some_and(|p| p.color() != self.turn)
            {
                return true;
            }
            if blocker_dist == dist {
                return false;
            }
        }
        true
    }

    /// Get blocking squares for non-linear checkers (Rose, etc.)
    /// Returns a list of intermediate squares that can be blocked to stop the attack.
    /// For Rose: finds the spiral path used to attack and returns intermediate squares.
    ///
    /// This is the generalized equivalent of `is_on_check_ray` for pieces that don't
    /// attack in straight lines.
    fn get_nonlinear_blocking_squares(
        &self,
        checker_sq: &Coordinate,
        king_sq: &Coordinate,
        checker_type: PieceType,
    ) -> arrayvec::ArrayVec<Coordinate, 8> {
        use crate::moves::ROSE_SPIRALS;

        let mut blocking = arrayvec::ArrayVec::<Coordinate, 8>::new();

        // Rose blocking squares: find the spiral path from checker to king
        // and return the intermediate squares
        // NOTE: Other non-linear piece types can be added here in the future
        // with their own blocking square logic
        if checker_type == PieceType::Rose {
            // Find which spiral the Rose used to reach the king
            let dx = king_sq.x - checker_sq.x;
            let dy = king_sq.y - checker_sq.y;

            // Search through all spirals to find one that matches
            for spiral_dirs in &ROSE_SPIRALS {
                for spiral in spiral_dirs {
                    for (hop_idx, &(cum_dx, cum_dy)) in spiral.iter().enumerate() {
                        if cum_dx == dx && cum_dy == dy {
                            // Found the spiral path! Verify it's unblocked and collect
                            // intermediate squares
                            let mut valid = true;
                            for (i, &(prev_dx, prev_dy)) in spiral.iter().take(hop_idx).enumerate()
                            {
                                let check_x = checker_sq.x + prev_dx;
                                let check_y = checker_sq.y + prev_dy;

                                // Check if this intermediate square is blocked
                                if self.board.get_piece(check_x, check_y).is_some() {
                                    // This path is blocked, try next spiral
                                    valid = false;
                                    break;
                                }

                                // Collect intermediate squares (excluding if i == hop_idx-1
                                // since that's the one already covered, but we take(hop_idx))
                                if i < hop_idx {
                                    blocking.push(Coordinate::new(check_x, check_y));
                                }
                            }

                            if valid && !blocking.is_empty() {
                                return blocking;
                            } else if valid {
                                // Direct hit (hop 0), no blocking squares
                                return blocking;
                            }
                            // Clear and try next spiral
                            blocking.clear();
                        }
                    }
                }
            }
        }

        blocking
    }

    /// Initialize starting_squares from the current board: all non-pawn,
    /// non-royal pieces' current coordinates are treated as their original
    /// squares. Intended to be called once when constructing a GameState
    /// from an initial position before replaying move history.
    pub fn init_starting_squares(&mut self) {
        self.starting_squares.clear();
        for ((x, y), piece) in self.board.iter() {
            if piece.piece_type() != PieceType::Pawn && !piece.piece_type().is_royal() {
                self.starting_squares.insert(Coordinate::new(*x, *y));
            }
        }
    }

    /// Initialize starting piece counts for game phase calculation.
    /// Should be called once when the game is created before move history replay.
    /// Counts non-pawn pieces (since pawns don't contribute to game phase).
    pub fn init_starting_piece_counts(&mut self) {
        let mut white: u16 = 0;
        let mut black: u16 = 0;
        for (_, piece) in self.board.iter() {
            if piece.piece_type() != PieceType::Pawn && piece.color() != PlayerColor::Neutral {
                match piece.color() {
                    PlayerColor::White => white += 1,
                    PlayerColor::Black => black += 1,
                    PlayerColor::Neutral => {}
                }
            }
        }
        self.starting_white_pieces = white;
        self.starting_black_pieces = black;
    }

    #[inline]
    pub fn has_pieces(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::White => self.white_piece_count > 0,
            PlayerColor::Black => self.black_piece_count > 0,
            PlayerColor::Neutral => false,
        }
    }

    /// O(1) check for non-pawn material (for NMP zugzwang detection).
    /// Returns true if the specified color has at least one non-pawn, non-king piece.
    #[inline]
    pub fn has_non_pawn_material(&self, color: PlayerColor) -> bool {
        match color {
            PlayerColor::White => self.white_non_pawn_material,
            PlayerColor::Black => self.black_non_pawn_material,
            PlayerColor::Neutral => false,
        }
    }

    /// Returns true if the side-to-move must respond to check.
    ///
    /// In standard chess (checkmate win condition), you must escape check.
    /// In capture-based variants (royalcapture, allroyalscaptured, allpiecescaptured),
    /// the opponent wins by capturing pieces, not by giving checkmate, so checks
    /// don't need to be escaped (the king can be captured).
    ///
    /// The logic: YOUR OWN win condition determines if YOU must escape check.
    /// - Your win condition specifies what the opponent must do to beat you.
    /// - If your win condition is Checkmate → opponent beats you via checkmate → you must escape
    /// - If your win condition is capture-based → opponent beats you via capture → you don't need to escape
    #[inline]
    pub fn must_escape_check(&self) -> bool {
        // Our own win condition tells us how the opponent beats us
        let our_win_condition = match self.turn {
            PlayerColor::White => self.game_rules.white_win_condition,
            PlayerColor::Black => self.game_rules.black_win_condition,
            PlayerColor::Neutral => return true, // Safe default
        };
        our_win_condition.requires_check_evasion()
    }

    /// Returns true if the given color's king can be captured (no check evasion needed).
    /// This is the opposite of must_escape_check but for a specific color.
    #[inline]
    pub fn king_capturable(&self, color: PlayerColor) -> bool {
        // The color's own win condition tells us if their king can be captured
        // (i.e., if their opponent wins via capture rather than checkmate)
        let win_condition = match color {
            PlayerColor::White => self.game_rules.white_win_condition,
            PlayerColor::Black => self.game_rules.black_win_condition,
            PlayerColor::Neutral => return false,
        };
        !win_condition.requires_check_evasion()
    }

    /// Check if the side-to-move has lost by royal capture.
    /// This is only relevant for RoyalCapture and AllRoyalsCaptured win conditions.
    /// Returns true if the opponent (who just moved) has captured all required royals.
    ///
    /// Zero overhead: This method checks the win condition first and returns false
    /// immediately for Checkmate and AllPiecesCaptured variants.
    #[inline]
    pub fn has_lost_by_royal_capture(&self) -> bool {
        // Get the side-to-move's win condition (what the opponent must do to beat them)
        let our_win_condition = match self.turn {
            PlayerColor::White => self.game_rules.white_win_condition,
            PlayerColor::Black => self.game_rules.black_win_condition,
            PlayerColor::Neutral => return false,
        };

        // Zero overhead: only check for royal loss if win condition is royal-capture based
        if !our_win_condition.is_royal_capture_based() {
            return false;
        }

        // Check if we still have our king
        let has_king = match self.turn {
            PlayerColor::White => self.white_king_pos.is_some(),
            PlayerColor::Black => self.black_king_pos.is_some(),
            PlayerColor::Neutral => true,
        };

        // If we have no king, we've lost (in RoyalCapture/AllRoyalsCaptured variants)
        !has_king
    }

    /// Stockfish-style repetition detection for search.
    /// Returns true if the current position should be treated as a draw due to repetition.
    ///
    /// Matches Stockfish's logic exactly: `repetition != 0 && repetition < ply`
    ///
    /// For twofold (repetition > 0): Only a draw if the repetition distance is less than ply,
    /// meaning the first occurrence is within the search tree.
    ///
    /// For threefold (repetition < 0): The negative value is always less than any positive ply,
    /// so threefold is always detected as a draw at ply > 0.
    #[inline]
    pub fn is_repetition(&self, ply: usize) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        // Stockfish: return st->repetition && st->repetition < ply;
        // This works for both positive (twofold) and negative (threefold) values.
        // Negative values are always < positive ply, so threefold always returns true for ply > 0.
        self.repetition != 0 && self.repetition < (ply as i32)
    }

    /// Check if this is a lone king endgame (one side only has a king)
    pub fn is_lone_king_endgame(&self) -> bool {
        use crate::board::{PieceType, PlayerColor};

        let mut white_has_non_king = false;
        let mut black_has_non_king = false;

        if let Some(active) = &self.board.active_coords {
            for (x, y) in active {
                let piece = match self.board.get_piece(*x, *y) {
                    Some(p) => p,
                    None => continue,
                };
                if piece.piece_type() != PieceType::King {
                    if piece.color() == PlayerColor::White {
                        white_has_non_king = true;
                    } else if piece.color() == PlayerColor::Black {
                        black_has_non_king = true;
                    }
                }
            }
        } else {
            for (_, piece) in self.board.iter() {
                if piece.piece_type() != PieceType::King {
                    if piece.color() == PlayerColor::White {
                        white_has_non_king = true;
                    } else if piece.color() == PlayerColor::Black {
                        black_has_non_king = true;
                    }
                }
            }
        }

        // One side has only a king (or nothing)
        !white_has_non_king || !black_has_non_king
    }

    /// Check if position is a draw by 50-move rule (or variant specific limit)
    pub fn is_fifty(&self) -> bool {
        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }
        // If no move rule is defined, never trigger a draw
        match self.game_rules.move_rule_limit {
            Some(limit) => self.halfmove_clock >= limit,
            None => false,
        }
    }

    /// Make a null move (just flip turn, for null move pruning)
    pub fn make_null_move(&mut self) {
        use crate::search::zobrist::{SIDE_KEY, en_passant_key};

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
        use crate::search::zobrist::{
            SIDE_KEY, castling_rights_key_from_bitfield, en_passant_key, pawn_special_right_key,
            piece_key,
        };

        let mut h: u64 = 0;

        // Hash all pieces (excluding obstacles/voids for performance)
        if let Some(active) = &self.board.active_coords {
            for (x, y) in active {
                let piece = match self.board.get_piece(*x, *y) {
                    Some(p) => p,
                    None => continue,
                };
                h ^= piece_key(piece.piece_type(), piece.color(), *x, *y);
            }
        } else {
            for ((x, y), piece) in self.board.iter() {
                if piece.color() == PlayerColor::Neutral {
                    continue;
                }
                h ^= piece_key(piece.piece_type(), piece.color(), *x, *y);
            }
        }

        // Hash EFFECTIVE castling rights using cached bitfield (O(1))
        h ^= castling_rights_key_from_bitfield(self.effective_castling_rights);

        // Hash individual PAWN special rights (double-push rights)
        for coord in &self.special_rights {
            if let Some(piece) = self.board.get_piece(coord.x, coord.y) {
                if piece.piece_type() == PieceType::Pawn {
                    h ^= pawn_special_right_key(coord.x, coord.y);
                }
            }
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

    /// Recompute pawn_hash, nonpawn_hash, and material_hash from scratch.
    /// These are used by correction history for indexing.
    /// All three are computed for comprehensive variant coverage.
    pub fn recompute_correction_hashes(&mut self) {
        use crate::search::zobrist::{material_key, pawn_key, piece_key};

        let mut ph: u64 = 0; // Pawn structure hash
        let mut nph: u64 = 0; // Non-pawn piece hash
        let mut mh: u64 = 0; // Material hash

        for ((x, y), piece) in self.board.iter() {
            if piece.color() == PlayerColor::Neutral {
                continue;
            }

            // Material hash: Additive to distinguish counts (avoid XOR cancellation)
            mh = mh.wrapping_add(material_key(piece.piece_type(), piece.color()));

            if piece.piece_type() == PieceType::Pawn {
                // Pawn hash: only pawns (helps CoaIP variants)
                ph ^= pawn_key(piece.color(), *x, *y);
            } else {
                // Non-pawn hash: all pieces except pawns
                nph ^= piece_key(piece.piece_type(), piece.color(), *x, *y);
            }
        }

        self.pawn_hash = ph;
        self.nonpawn_hash = nph;
        self.material_hash = mh;
    }

    #[inline(always)]
    pub fn enemy_king_pos(&self) -> Option<&Coordinate> {
        if self.turn == PlayerColor::White {
            self.black_king_pos.as_ref()
        } else {
            self.white_king_pos.as_ref()
        }
    }

    /// Returns pseudo-legal moves. Legality (not leaving king in check)
    /// is checked in the search after making each move.
    /// When in check and must escape, uses the optimized evasion generator.
    pub fn get_legal_moves(&self) -> MoveList {
        // Check if we're in check and need to use the optimized evasion generator
        if self.is_in_check() && self.must_escape_check() {
            let mut out = MoveList::new();
            self.get_evasion_moves_into(&mut out);
            return out;
        }

        let ctx = crate::moves::MoveGenContext {
            special_rights: &self.special_rights,
            en_passant: &self.en_passant,
            game_rules: &self.game_rules,
            indices: &self.spatial_indices,
            enemy_king_pos: self.enemy_king_pos(),
        };

        get_legal_moves(&self.board, self.turn, &ctx)
    }

    /// Fill a pre-allocated buffer with pseudo-legal moves for the current side.
    /// When in check and must escape (checkmate win condition), uses the optimized
    /// evasion generator that handles long-range blocking moves correctly.
    pub fn get_legal_moves_into(&self, out: &mut MoveList) {
        if self.is_in_check() {
            self.get_evasion_moves_into(out);
            // Strict legality filtering (pins/leaving king in check)
            let mut i = 0;
            let mut s_mut = self.clone();
            while i < out.len() {
                let m = out[i];
                let undo = s_mut.make_move(&m);
                if s_mut.is_move_illegal() {
                    s_mut.undo_move(&m, undo);
                    out.remove(i);
                } else {
                    s_mut.undo_move(&m, undo);
                    i += 1;
                }
            }
            return;
        }

        let ctx = crate::moves::MoveGenContext {
            special_rights: &self.special_rights,
            en_passant: &self.en_passant,
            game_rules: &self.game_rules,
            indices: &self.spatial_indices,
            enemy_king_pos: self.enemy_king_pos(),
        };

        get_legal_moves_into(&self.board, self.turn, &ctx, out, false);

        if out.is_empty() {
            get_legal_moves_into(&self.board, self.turn, &ctx, out, true);
        }
    }

    pub fn get_evasion_moves_into(&self, out: &mut MoveList) {
        out.clear();

        let our_color = self.turn;
        let their_color = our_color.opponent();

        // Use cached king position for efficiency
        let king_sq = match our_color {
            PlayerColor::White => self.white_king_pos,
            PlayerColor::Black => self.black_king_pos,
            PlayerColor::Neutral => None,
        };

        let king_sq = match king_sq {
            Some(pos) => pos,
            None => return, // No king found, no evasion possible
        };

        let king_piece = match self.board.get_piece(king_sq.x, king_sq.y) {
            Some(p) => *p,
            None => return,
        };

        // Use stack-allocated array for checkers
        let mut checkers: [Coordinate; 16] = [Coordinate::new(0, 0); 16];
        let mut checker_count = 0;

        // COMPREHENSIVE CHECKER DETECTION (Sync with is_square_attacked)
        // Check all enemy pieces to see if they attack our king
        let indices = &self.spatial_indices;
        if let Some(active) = &self.board.active_coords {
            for &(ax, ay) in active {
                if self.board.get_piece(ax, ay).is_some_and(|p| {
                    p.color() == their_color
                        && crate::moves::is_piece_attacking_square(
                            &self.board,
                            p,
                            &Coordinate::new(ax, ay),
                            &king_sq,
                            indices,
                            &self.game_rules,
                        )
                }) && checker_count < 16
                {
                    checkers[checker_count] = Coordinate::new(ax, ay);
                    checker_count += 1;
                }
            }
        } else {
            for (&(ax, ay), p) in self.board.iter() {
                if p.color() == their_color
                    && crate::moves::is_piece_attacking_square(
                        &self.board,
                        p,
                        &Coordinate::new(ax, ay),
                        &king_sq,
                        indices,
                        &self.game_rules,
                    )
                    && checker_count < 16
                {
                    checkers[checker_count] = Coordinate::new(ax, ay);
                    checker_count += 1;
                }
            }
        }

        if checker_count == 0 {
            return; // No checkers found
        }

        // 1. King escapes (Legal regardless of checker count, as long as target not attacked)
        let ctx = crate::moves::MoveGenContext {
            special_rights: &self.special_rights,
            en_passant: &self.en_passant,
            game_rules: &self.game_rules,
            indices: &self.spatial_indices,
            enemy_king_pos: self.enemy_king_pos(),
        };
        get_pseudo_legal_moves_for_piece_into(
            &self.board,
            &king_piece,
            &king_sq,
            &ctx,
            true, // Allow check-related filtering
            out,
        );

        if checker_count >= 2 {
            return; // Double check - only king moves can escape
        }

        // 2. Capture checker or block attack (Only in single check)
        let checker_sq = checkers[0];
        let dx_check = checker_sq.x - king_sq.x;
        let dy_check = checker_sq.y - king_sq.y;

        // Identify if checker is a slider (Ortho, Diag, or Knightrider)
        use crate::attacks::{DIAG_MASK, KNIGHTRIDER_MASK, ORTHO_MASK, matches_mask};
        let checker_p = self.board.get_piece(checker_sq.x, checker_sq.y).unwrap();
        let checker_type = checker_p.piece_type();
        let is_slider = matches_mask(checker_type, ORTHO_MASK | DIAG_MASK | KNIGHTRIDER_MASK)
            || checker_type == PieceType::Huygen;

        // Identify if checker is a non-linear attacker (Rose, etc.)
        // These pieces have blocking squares but not along a straight line
        let is_nonlinear_checker = checker_type == PieceType::Rose;

        // Identify if checker is a Huygen (prime-distance attacker)
        // Huygen blocking is special: only squares at prime distances FROM THE HUYGEN can block
        let is_huygen_checker = checker_type == PieceType::Huygen;

        // Identify if checker is a Knightrider
        // Knightrider blocking is special: blocking squares are along the knight hop path
        let is_knightrider_checker = checker_type == PieceType::Knightrider;

        // Pre-compute knightrider blocking squares: the intermediate knight hops between checker and king
        // The knightrider attacks along a line of repeated knight moves: (dx, dy) * n
        // For a check from checker_sq to king_sq, we need to find which knight direction was used
        let knightrider_blocking_squares: arrayvec::ArrayVec<Coordinate, 32> =
            if is_knightrider_checker {
                use crate::attacks::KNIGHTRIDER_DIRS;
                let mut blocking = arrayvec::ArrayVec::new();
                // Find which knight direction matches the check vector
                for &(ndx, ndy) in &KNIGHTRIDER_DIRS {
                    // Check if dx_check = ndx * n and dy_check = ndy * n for some positive n
                    if ndx != 0 && ndy != 0 {
                        let n_x = dx_check / ndx;
                        let n_y = dy_check / ndy;
                        if n_x == n_y && n_x > 0 && dx_check == ndx * n_x && dy_check == ndy * n_y {
                            // Found the knight direction! Compute intermediate squares
                            let n = n_x;
                            for i in 1..n {
                                let bx = king_sq.x + ndx * i;
                                let by = king_sq.y + ndy * i;
                                blocking.push(Coordinate::new(bx, by));
                            }
                            break;
                        }
                    }
                }
                blocking
            } else {
                arrayvec::ArrayVec::new()
            };

        // For non-linear checkers, compute blocking squares up front
        let nonlinear_blocking_squares = if is_nonlinear_checker {
            self.get_nonlinear_blocking_squares(&checker_sq, &king_sq, checker_type)
        } else {
            arrayvec::ArrayVec::<Coordinate, 8>::new()
        };

        let check_dist = dx_check.abs().max(dy_check.abs());
        let step_x = dx_check.signum();
        let step_y = dy_check.signum();

        // SPECIAL CASE: Huygen checker - Fast Path
        // Always active. Checks the first 128 primes from the checker towards the king.
        // Handles both Huygens between and Huygens outside (jumping over checker).
        if is_huygen_checker {
            use crate::utils::PRIMES_UNDER_128;

            let is_horizontal = dy_check == 0;
            let checker_coord = if is_horizontal {
                checker_sq.x
            } else {
                checker_sq.y
            };
            let king_coord = if is_horizontal { king_sq.x } else { king_sq.y };
            let dir_from_checker_to_king = (king_coord - checker_coord).signum();

            let line_vec = if is_horizontal {
                self.spatial_indices.rows.get(&checker_sq.y)
            } else {
                self.spatial_indices.cols.get(&checker_sq.x)
            };

            if let Some(vec) = line_vec {
                for &(coord, packed) in vec {
                    let piece = Piece::from_packed(packed);
                    if piece.color() != our_color || piece.piece_type() != PieceType::Huygen {
                        continue;
                    }

                    let our_huygen_coord = coord;
                    let from_sq = if is_horizontal {
                        Coordinate::new(our_huygen_coord, checker_sq.y)
                    } else {
                        Coordinate::new(checker_sq.x, our_huygen_coord)
                    };

                    'outer_huygen_fast: for &p_from_checker in &PRIMES_UNDER_128 {
                        let block_coord = checker_coord + dir_from_checker_to_king * p_from_checker;

                        // Must be between king and checker
                        let block_between = if checker_coord > king_coord {
                            block_coord > king_coord && block_coord < checker_coord
                        } else {
                            block_coord < king_coord && block_coord > checker_coord
                        };
                        if !block_between {
                            let past_king = if checker_coord > king_coord {
                                block_coord <= king_coord
                            } else {
                                block_coord >= king_coord
                            };
                            if past_king {
                                break;
                            }
                            continue;
                        }

                        // Check if square is occupied by friendly piece (cannot block there)
                        let bx = if is_horizontal {
                            block_coord
                        } else {
                            checker_sq.x
                        };
                        let by = if is_horizontal {
                            checker_sq.y
                        } else {
                            block_coord
                        };
                        if let Some(p) = self.board.get_piece(bx, by) {
                            if p.color() == our_color {
                                continue;
                            }
                        }

                        let dist_from_huygen = (block_coord - our_huygen_coord).abs();
                        if dist_from_huygen == 0 {
                            continue;
                        }

                        // Use robust prime check for distance from Huygen
                        if !is_prime_i64(dist_from_huygen) {
                            continue;
                        }

                        // Blocker check (ignore checker since it's the checker)
                        let dir_to_block = (block_coord - our_huygen_coord).signum();
                        for &(other_coord, _) in vec {
                            if other_coord == our_huygen_coord || other_coord == checker_coord {
                                continue;
                            }

                            let other_dir = (other_coord - our_huygen_coord).signum();
                            if other_dir != dir_to_block {
                                continue;
                            }

                            let other_dist = (other_coord - our_huygen_coord).abs();
                            if other_dist >= dist_from_huygen {
                                continue;
                            }

                            if is_prime_i64(other_dist) {
                                continue 'outer_huygen_fast;
                            }
                        }

                        let to_sq = Coordinate::new(bx, by);
                        out.push(Move::new(from_sq, to_sq, piece));
                    }
                }
            }
        }

        let process_piece = |s: &GameState, from: Coordinate, piece: &Piece, out: &mut MoveList| {
            if piece.color() != our_color || (from.x == king_sq.x && from.y == king_sq.y) {
                return;
            }

            let pt = piece.piece_type();

            // Import attack utilities
            use crate::attacks::{
                CAMEL_OFFSETS, GIRAFFE_OFFSETS, HAWK_OFFSETS, KING_OFFSETS, KNIGHT_OFFSETS,
                KNIGHTRIDER_DIRS, ZEBRA_OFFSETS, attacks_like_king, attacks_like_knight,
                is_diag_slider, is_ortho_slider,
            };

            let can_ortho = is_ortho_slider(pt);
            let can_diag = is_diag_slider(pt);
            let can_knight = attacks_like_knight(pt);
            let can_king = attacks_like_king(pt);

            // Helper to check if target is valid for blocking (empty or enemy)
            let can_block_at = |tx: i64, ty: i64| -> bool {
                if let Some(target_p) = s.board.get_piece(tx, ty) {
                    target_p.color() != our_color
                } else {
                    true
                }
            };

            // Helper to check if a target square is on the check blocking path
            // For knightrider checkers, use the pre-computed knight hop path
            // For other checkers, use the standard check ray logic
            let is_valid_blocking_square = |tx: i64, ty: i64| -> bool {
                if is_knightrider_checker {
                    // For knightrider checkers, blocking squares are along the knight hop path
                    knightrider_blocking_squares
                        .iter()
                        .any(|sq| sq.x == tx && sq.y == ty)
                } else {
                    // For other sliders, use standard check ray logic
                    s.is_on_check_ray(
                        &Coordinate::new(tx, ty),
                        &king_sq,
                        step_x,
                        step_y,
                        check_dist,
                        checker_type,
                    )
                }
            };

            // ==========================================
            // SLIDER BLOCKING (Rook/Bishop/Queen/etc)
            // Direct intersection calculation - O(1), works for infinite distances
            // ==========================================

            // HUYGEN CHECKER: Special handling for ortho sliders
            // Compute where the slider intersects the check ray and verify it's at prime distance from checker
            if is_huygen_checker && can_ortho {
                // Huygen attacks orthogonally, so check ray is horizontal (dy=0) or vertical (dx=0)
                let is_horizontal_check = dy_check == 0;

                if is_horizontal_check {
                    // Check ray is horizontal at y = king_sq.y = checker_sq.y
                    // Ortho slider can block by moving vertically to y = king_sq.y
                    if from.y != king_sq.y {
                        // Slider is NOT on the check ray - can it move to it?
                        // Destination is (from.x, king_sq.y) if path is clear
                        let tx = from.x;
                        let ty = king_sq.y;

                        // Check if this x is between king and checker
                        let between = if checker_sq.x > king_sq.x {
                            tx > king_sq.x && tx < checker_sq.x
                        } else {
                            tx < king_sq.x && tx > checker_sq.x
                        };

                        if between {
                            // Check if distance from checker is prime
                            let dist_from_checker = (checker_sq.x - tx).abs();
                            if is_prime_fast(dist_from_checker) {
                                // Check path is clear (vertical move)
                                if s.is_path_clear_for_rook(&from, &Coordinate::new(tx, ty)) {
                                    out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                                }
                            }
                        }
                    } else {
                        // Slider IS on the check ray - can slide horizontally to prime distance
                        // Already handled by general Huygen blocking below
                    }
                } else {
                    // Check ray is vertical at x = king_sq.x = checker_sq.x
                    // Ortho slider can block by moving horizontally to x = king_sq.x
                    if from.x != king_sq.x {
                        let tx = king_sq.x;
                        let ty = from.y;

                        let between = if checker_sq.y > king_sq.y {
                            ty > king_sq.y && ty < checker_sq.y
                        } else {
                            ty < king_sq.y && ty > checker_sq.y
                        };

                        if between {
                            let dist_from_checker = (checker_sq.y - ty).abs();
                            if is_prime_fast(dist_from_checker) {
                                if s.is_path_clear_for_rook(&from, &Coordinate::new(tx, ty)) {
                                    out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                                }
                            }
                        }
                    }
                }
            }

            // Regular slider blocking for non-Huygen checkers
            if is_slider && can_ortho && !is_huygen_checker {
                // Horizontal line y=from.y intersects check ray
                if step_y != 0 {
                    let k = (from.y - king_sq.y) / step_y;
                    let rem = (from.y - king_sq.y) % step_y;
                    if rem == 0 && k >= 1 && k <= check_dist {
                        let tx = king_sq.x + k * step_x;
                        let ty = from.y;
                        if tx != from.x && s.is_path_clear_for_rook(&from, &Coordinate::new(tx, ty))
                        {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }
                // Vertical line x=from.x intersects check ray
                if step_x != 0 {
                    let k = (from.x - king_sq.x) / step_x;
                    let rem = (from.x - king_sq.x) % step_x;
                    if rem == 0 && k >= 1 && k <= check_dist {
                        let tx = from.x;
                        let ty = king_sq.y + k * step_y;
                        if ty != from.y && s.is_path_clear_for_rook(&from, &Coordinate::new(tx, ty))
                        {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }
            }

            // ==========================================
            // OPTIMIZED HUYGEN BLOCKING & CAPTURE
            // Uses intersection logic for O(1) checks on cross-rays.
            // Falls back to O(N) loop only for parallel rays.
            // Pre-fetches spatial indices for max performance.
            // ==========================================
            if pt == PieceType::Huygen && is_slider {
                use crate::utils::is_prime_fast;
                // Pre-fetch spatial indices
                let row_pieces = s.spatial_indices.rows.get(&from.y);
                let col_pieces = s.spatial_indices.cols.get(&from.x);

                // --- Helper: Check and push move to (tx, ty) ---
                // Returns true if move was valid and pushed
                let mut try_add_move = |tx: i64, ty: i64, verify_checker_dist: bool| {
                    let d_x = tx - from.x;
                    let d_y = ty - from.y;
                    let dist = d_x.abs().max(d_y.abs());

                    // Prime distance check (Huygen constraint)
                    if dist == 0 || !is_prime_fast(dist) {
                        return false;
                    }

                    // For Huygen checker: blocking sq must be prime dist from checker
                    if verify_checker_dist {
                        let d_cx = tx - checker_sq.x;
                        let d_cy = ty - checker_sq.y;
                        let dist_c = d_cx.abs().max(d_cy.abs());
                        if !is_prime_fast(dist_c) {
                            return false;
                        }
                    }

                    // Check blockers
                    let mut blocked = false;
                    if d_y == 0 {
                        // Horizontal move
                        if let Some(vec) = row_pieces {
                            let dir = d_x.signum();
                            for &(coord, _) in vec {
                                let dp = (coord - from.x) * dir;
                                if dp > 0 && dp < dist && is_prime_fast(dp) {
                                    blocked = true;
                                    break;
                                }
                            }
                        }
                    } else {
                        // Vertical move
                        if let Some(vec) = col_pieces {
                            let dir = d_y.signum();
                            for &(coord, _) in vec {
                                let dp = (coord - from.y) * dir;
                                if dp > 0 && dp < dist && is_prime_fast(dp) {
                                    blocked = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !blocked {
                        out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        return true;
                    }
                    false
                };

                // 1. CAPTURE CHECKER
                // Check if checker is strictly orthogonal to us
                let dx_c = checker_sq.x - from.x;
                let dy_c = checker_sq.y - from.y;
                if (dx_c == 0) != (dy_c == 0) {
                    try_add_move(checker_sq.x, checker_sq.y, false);
                }

                // 2. BLOCKING
                let max_k = check_dist.min(100);

                // A. Check Horizontal Intercepts (Huygen moves Vertically | to block)
                // We want intersection where bx == from.x
                if step_x != 0 {
                    // Check Ray is not vertical, so it crosses x = from.x exactly once
                    let k = (from.x - king_sq.x) / step_x;
                    if k >= 1 && k < check_dist && (from.x - king_sq.x) % step_x == 0 {
                        let by = king_sq.y + k * step_y;
                        if by != from.y && can_block_at(from.x, by) {
                            try_add_move(from.x, by, is_huygen_checker);
                        }
                    }
                } else {
                    // Check Ray is Vertical (step_x == 0).
                    // If from.x == king.x, we are PARALLEL (Huygen on the check line).
                    if from.x == king_sq.x {
                        for k in 1..max_k {
                            let by = king_sq.y + k * step_y;
                            // For parallel vertical, we only care about vertical moves along the ray
                            if by != from.y && can_block_at(from.x, by) {
                                try_add_move(from.x, by, is_huygen_checker);
                            }
                        }
                    }
                }

                // B. Check Vertical Intercepts (Huygen moves Horizontally - to block)
                // We want intersection where by == from.y
                if step_y != 0 {
                    // Check Ray is not horizontal, so it crosses y = from.y exactly once
                    let k = (from.y - king_sq.y) / step_y;
                    if k >= 1 && k < check_dist && (from.y - king_sq.y) % step_y == 0 {
                        let bx = king_sq.x + k * step_x;
                        if bx != from.x && can_block_at(bx, from.y) {
                            try_add_move(bx, from.y, is_huygen_checker);
                        }
                    }
                } else {
                    // Check Ray is Horizontal (step_y == 0).
                    // If from.y == king.y, we are PARALLEL (Huygen on the check line).
                    if from.y == king_sq.y {
                        for k in 1..max_k {
                            let bx = king_sq.x + k * step_x;
                            // For parallel horizontal, we only care about horizontal moves along the ray
                            if bx != from.x && can_block_at(bx, from.y) {
                                try_add_move(bx, from.y, is_huygen_checker);
                            }
                        }
                    }
                }
            }

            if is_slider && can_diag {
                // Diagonal x-y=c intersects check ray
                let s_diff = step_x - step_y;
                if s_diff != 0 {
                    let d_diff = (from.x - from.y) - (king_sq.x - king_sq.y);
                    let k = d_diff / s_diff;
                    let rem = d_diff % s_diff;
                    if rem == 0 && k >= 1 && k <= check_dist {
                        let tx = king_sq.x + k * step_x;
                        let ty = king_sq.y + k * step_y;
                        if !(tx == from.x && ty == from.y)
                            && s.is_path_clear_for_bishop(&from, &Coordinate::new(tx, ty))
                            && (!is_huygen_checker || is_prime_fast(check_dist - k))
                        {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }
                // Anti-diagonal x+y=c intersects check ray
                let s_sum = step_x + step_y;
                if s_sum != 0 {
                    let d_sum = (from.x + from.y) - (king_sq.x + king_sq.y);
                    let k = d_sum / s_sum;
                    let rem = d_sum % s_sum;
                    if rem == 0 && k >= 1 && k <= check_dist {
                        let tx = king_sq.x + k * step_x;
                        let ty = king_sq.y + k * step_y;
                        if !(tx == from.x && ty == from.y)
                            && s.is_path_clear_for_bishop(&from, &Coordinate::new(tx, ty))
                            && (!is_huygen_checker || is_prime_fast(check_dist - k))
                        {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }
            }

            // ==========================================
            // KNIGHTRIDER BLOCKING
            // Slider along knight directions - intersection calculation
            // ==========================================
            if is_slider && pt == PieceType::Knightrider {
                for &(ndx, ndy) in &KNIGHTRIDER_DIRS {
                    // Knightrider line: from + t*(ndx, ndy) for t >= 1
                    // Check ray: king + k*(step_x, step_y) for k in 1..=check_dist
                    // Find t, k such that both equations are satisfied
                    let det = ndx * step_y - ndy * step_x;
                    if det != 0 {
                        let dx = king_sq.x - from.x;
                        let dy = king_sq.y - from.y;
                        let t_num = dx * step_y - dy * step_x;
                        let k_num = dx * ndy - dy * ndx;
                        if t_num % det == 0 && k_num % det == 0 {
                            let t = t_num / det;
                            let k = k_num / det;
                            if t >= 1 && k >= 1 && k <= check_dist {
                                let tx = from.x + t * ndx;
                                let ty = from.y + t * ndy;
                                // Check path is clear for knightrider
                                let mut path_clear = true;
                                for i in 1..t {
                                    if s.board
                                        .get_piece(from.x + i * ndx, from.y + i * ndy)
                                        .is_some()
                                    {
                                        path_clear = false;
                                        break;
                                    }
                                }
                                if path_clear
                                    && can_block_at(tx, ty)
                                    && (!is_huygen_checker || is_prime_fast(check_dist - k))
                                {
                                    out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                                }
                            }
                        }
                    }
                }
            }

            // ==========================================
            // LEAPER BLOCKING (Knight, Camel, Zebra, Giraffe, Guard, Hawk)
            // Enumerate fixed jump patterns - O(jumps)
            // ==========================================
            if is_slider {
                // Knight-like blocking
                if can_knight {
                    for &(dx, dy) in &KNIGHT_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }

                // King/Guard-like blocking (1-square moves)
                if can_king {
                    for &(dx, dy) in &KING_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }

                // Camel blocking
                if pt == PieceType::Camel {
                    for &(dx, dy) in &CAMEL_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }

                // Zebra blocking
                if pt == PieceType::Zebra {
                    for &(dx, dy) in &ZEBRA_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }

                // Giraffe blocking
                if pt == PieceType::Giraffe {
                    for &(dx, dy) in &GIRAFFE_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }

                // Hawk blocking
                if pt == PieceType::Hawk {
                    for &(dx, dy) in &HAWK_OFFSETS {
                        let tx = from.x + dx;
                        let ty = from.y + dy;
                        if is_valid_blocking_square(tx, ty) && can_block_at(tx, ty) {
                            out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                        }
                    }
                }
            }

            // ==========================================
            // DIRECT SLIDER CAPTURE OF CHECKER
            // O(1) checks for infinite-range slider captures - no move gen needed
            // ==========================================
            if can_ortho {
                // Check if checker is on same row or column
                if from.x == checker_sq.x && from.y != checker_sq.y {
                    // Same column - vertical capture
                    if s.is_path_clear_for_rook(&from, &checker_sq) {
                        out.push(Move::new(from, checker_sq, *piece));
                    }
                } else if from.y == checker_sq.y && from.x != checker_sq.x {
                    // Same row - horizontal capture
                    if s.is_path_clear_for_rook(&from, &checker_sq) {
                        out.push(Move::new(from, checker_sq, *piece));
                    }
                }
            }

            if can_diag {
                // Check if checker is on same diagonal
                let dx = checker_sq.x - from.x;
                let dy = checker_sq.y - from.y;
                if dx != 0 && dx.abs() == dy.abs() {
                    // On a diagonal
                    if s.is_path_clear_for_bishop(&from, &checker_sq) {
                        out.push(Move::new(from, checker_sq, *piece));
                    }
                }
            }

            // ==========================================
            // HUYGEN CAPTURE OF CHECKER
            // O(n) check for Huygen capturing at prime distance
            // ==========================================
            if pt == PieceType::Huygen {
                let dx = checker_sq.x - from.x;
                let dy = checker_sq.y - from.y;

                // Must be orthogonal (same row or column)
                if (dx == 0 && dy != 0) || (dy == 0 && dx != 0) {
                    let dist = dx.abs().max(dy.abs());

                    // Must be at prime distance
                    if is_prime_fast(dist) {
                        // Check for blockers at closer prime distances
                        let is_horizontal = dy == 0;
                        let line_vec = if is_horizontal {
                            s.spatial_indices.rows.get(&from.y)
                        } else {
                            s.spatial_indices.cols.get(&from.x)
                        };

                        let our_coord = if is_horizontal { from.x } else { from.y };
                        let checker_coord = if is_horizontal {
                            checker_sq.x
                        } else {
                            checker_sq.y
                        };
                        let dir = (checker_coord - our_coord).signum();

                        let mut blocked = false;
                        if let Some(vec) = line_vec {
                            for &(coord, _) in vec {
                                let d = (coord - our_coord) * dir;
                                if d > 0 && d < dist && is_prime_fast(d) {
                                    blocked = true;
                                    break;
                                }
                            }
                        }

                        if !blocked {
                            out.push(Move::new(from, checker_sq, *piece));
                        }
                    }
                }
            }

            // ==========================================
            // CAPTURE & BLOCKING DETECTION (for remaining pieces)
            // Uses pseudo-legal move generation for captures
            // ==========================================
            let mut pseudo = MoveList::new();
            let ctx = crate::moves::MoveGenContext {
                special_rights: &s.special_rights,
                en_passant: &s.en_passant,
                game_rules: &s.game_rules,
                indices: &s.spatial_indices,
                enemy_king_pos: s.enemy_king_pos(),
            };
            get_pseudo_legal_moves_for_piece_into(&s.board, piece, &from, &ctx, true, &mut pseudo);

            // Check if this piece has optimized blocking (already handled above)
            let has_optimized_blocking = can_ortho
                || can_diag
                || can_knight
                || can_king
                || pt == PieceType::Camel
                || pt == PieceType::Zebra
                || pt == PieceType::Giraffe
                || pt == PieceType::Hawk
                || pt == PieceType::Knightrider;

            for m in pseudo {
                // Capture of checker
                if m.to.x == checker_sq.x && m.to.y == checker_sq.y {
                    out.push(m);
                    continue;
                }
                // Special case: en passant capture removes piece at ep.pawn_square, not m.to
                // If the checker is at ep.pawn_square, the EP move captures it and escapes check
                if m.piece.piece_type() == PieceType::Pawn {
                    if let Some(ep) = &s.en_passant {
                        if m.to.x == ep.square.x && m.to.y == ep.square.y {
                            // This is an EP capture - check if it captures the checker
                            if ep.pawn_square.x == checker_sq.x && ep.pawn_square.y == checker_sq.y
                            {
                                out.push(m);
                                continue;
                            }
                        }
                    }
                }
                // Blocking moves for sliders (straight line check rays) and knightrider checkers
                if is_slider && !has_optimized_blocking && is_valid_blocking_square(m.to.x, m.to.y)
                {
                    out.push(m);
                    continue;
                }
                // Blocking moves for knightrider checkers (for pieces not covered above)
                if is_knightrider_checker
                    && knightrider_blocking_squares
                        .iter()
                        .any(|sq| sq.x == m.to.x && sq.y == m.to.y)
                {
                    out.push(m);
                    continue;
                }
                // Blocking moves for non-linear checkers (Rose spirals, etc.)
                if is_nonlinear_checker
                    && nonlinear_blocking_squares
                        .iter()
                        .any(|sq| sq.x == m.to.x && sq.y == m.to.y)
                {
                    out.push(m);
                    continue;
                }
            }
        };

        if let Some(active) = &self.board.active_coords {
            for &(ax, ay) in active {
                if let Some(p) = self.board.get_piece(ax, ay) {
                    process_piece(self, Coordinate::new(ax, ay), p, out);
                }
            }
        } else {
            for (&(ax, ay), p) in self.board.iter() {
                process_piece(self, Coordinate::new(ax, ay), p, out);
            }
        }
    }

    /// Check if the side that just moved left their royal piece(s) in check (illegal move).
    /// Call this AFTER make_move to verify legality.
    /// Checks all royal pieces: King, RoyalQueen, RoyalCentaur
    ///
    /// Note: In capture-based win condition variants (royalcapture, allroyalscaptured,
    /// allpiecescaptured), leaving your king in check is NOT illegal since the opponent
    /// wins by capturing, not by checkmate.
    pub fn is_move_illegal(&self) -> bool {
        // After make_move, self.turn is the opponent (the side that will move next).
        // We need to check if the side that just moved (opponent of current turn) has any royal in check.
        let moved_color = self.turn.opponent();

        // Check if the side that moved needs to escape check.
        // moved_color's win condition tells us what their opponent does to beat them.
        // If moved_color's win condition is capture-based (not checkmate), then
        // leaving the king in check is NOT illegal (king can be captured).
        if self.king_capturable(moved_color) {
            return false; // Leaving king in check is legal in capture-based variants
        }

        let indices = &self.spatial_indices;

        // Fast path: use cached king position for the side that just moved
        let cached_king = match moved_color {
            PlayerColor::White => self.white_king_pos,
            PlayerColor::Black => self.black_king_pos,
            PlayerColor::Neutral => None,
        };

        if let Some(king_pos) = cached_king {
            // Most common case: single royal piece with cached position
            if is_square_attacked(&self.board, &king_pos, self.turn, indices) {
                return true;
            }
            // For standard variants with just a King, we're done
            if self
                .board
                .get_piece(king_pos.x, king_pos.y)
                .is_some_and(|p| p.piece_type() == PieceType::King)
            {
                return false;
            }
        }

        // Fallback: full scan for variants with multiple royals or no cached position
        self.is_move_illegal_full_scan(moved_color, indices)
    }

    /// Full board scan for illegal move detection. Used as fallback for variants
    /// with multiple royal pieces or when cached king position is unavailable.
    #[inline(never)]
    fn is_move_illegal_full_scan(
        &self,
        moved_color: PlayerColor,
        indices: &SpatialIndices,
    ) -> bool {
        if let Some(active) = &self.board.active_coords {
            for (x, y) in active {
                let piece = match self.board.get_piece(*x, *y) {
                    Some(p) => p,
                    None => continue,
                };
                if piece.color() == moved_color && piece.piece_type().is_royal() {
                    let pos = Coordinate::new(*x, *y);
                    if is_square_attacked(&self.board, &pos, self.turn, indices) {
                        return true;
                    }
                }
            }
        } else {
            for ((x, y), piece) in self.board.iter() {
                if piece.color() == moved_color && piece.piece_type().is_royal() {
                    let pos = Coordinate::new(*x, *y);
                    if is_square_attacked(&self.board, &pos, self.turn, indices) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline(always)]
    pub fn is_in_check(&self) -> bool {
        let indices = &self.spatial_indices;
        let attacker_color = self.turn.opponent();

        // Fast path: use cached king position for the side to move
        let cached_king = match self.turn {
            PlayerColor::White => self.white_king_pos,
            PlayerColor::Black => self.black_king_pos,
            PlayerColor::Neutral => None,
        };

        if let Some(king_pos) = cached_king {
            // Most common case: single royal piece with cached position
            if is_square_attacked(&self.board, &king_pos, attacker_color, indices) {
                return true;
            }
            // For standard variants, we're done. But for variants with multiple royals
            // (e.g., RoyalQueen, RoyalCentaur), we need to check all of them.
            // We can skip the full scan if the piece at cached position is the only royal.
            if let Some(piece) = self.board.get_piece(king_pos.x, king_pos.y) {
                // If it's a standard King, there's typically only one - skip full scan
                if piece.piece_type() == PieceType::King {
                    return false;
                }
            }
        }

        // Fallback: full scan for variants with multiple royals or no cached position
        self.is_in_check_full_scan(attacker_color, indices)
    }

    /// Full board scan for check detection. Used as fallback for variants with
    /// multiple royal pieces or when cached king position is unavailable.
    #[inline(always)]
    pub fn is_in_check_full_scan(
        &self,
        attacker_color: PlayerColor,
        indices: &SpatialIndices,
    ) -> bool {
        if let Some(active) = &self.board.active_coords {
            for (x, y) in active {
                let piece = match self.board.get_piece(*x, *y) {
                    Some(p) => p,
                    None => continue,
                };
                if piece.color() == self.turn && piece.piece_type().is_royal() {
                    let pos = Coordinate::new(*x, *y);
                    if is_square_attacked(&self.board, &pos, attacker_color, indices) {
                        return true;
                    }
                }
            }
        } else {
            for ((x, y), piece) in self.board.iter() {
                if piece.color() == self.turn && piece.piece_type().is_royal() {
                    let pos = Coordinate::new(*x, *y);
                    if is_square_attacked(&self.board, &pos, attacker_color, indices) {
                        return true;
                    }
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
        let piece = match self.board.get_piece(from_x, from_y) {
            Some(p) => *p,
            None => return, // No piece at from - invalid move, just skip
        };

        let mut m = Move {
            from: Coordinate::new(from_x, from_y),
            to: Coordinate::new(to_x, to_y),
            piece,
            promotion: promotion.and_then(|s| s.parse().ok()),
            rook_coord: None,
        };

        // Detect if this is a castling move to populate rook_coord (partner_coord)
        // Castling works with any non-pawn, non-royal piece that has special rights
        if piece.piece_type().is_royal() {
            let dx = to_x - from_x;
            if dx.abs() > 1 {
                // Use spatial indices to find castling partner - O(log n) instead of O(distance)
                let partner_dir = if dx > 0 { 1i64 } else { -1i64 };
                if let Some(row_pieces) = self.spatial_indices.rows.get(&from_y) {
                    // Find nearest piece past king's destination
                    if let Some((partner_x, packed)) =
                        SpatialIndices::find_nearest(row_pieces, to_x, partner_dir)
                    {
                        let partner = Piece::from_packed(packed);
                        let partner_coord = Coordinate::new(partner_x, from_y);
                        // Accept any non-pawn, non-royal piece with special rights as a castling partner
                        if partner.color() == piece.color()
                            && partner.piece_type() != PieceType::Pawn
                            && !partner.piece_type().is_royal()
                            && self.special_rights.contains(&partner_coord)
                        {
                            m.rook_coord = Some(partner_coord);
                        }
                    }
                }
            }
        }

        // Execute the move using the centralized logic
        self.make_move(&m);
    }

    pub fn make_move(&mut self, m: &Move) -> UndoMove {
        use crate::search::zobrist::{
            SIDE_KEY, en_passant_key, material_key, pawn_special_right_key, piece_key,
        };

        // Push current position hash BEFORE making the move (for repetition detection)
        self.hash_stack.push(self.hash);

        let from_coord = Coordinate::new(m.from.x, m.from.y);

        let piece = self.board.remove_piece(&m.from.x, &m.from.y).unwrap();
        // Update spatial indices: remove moving piece from source square
        self.spatial_indices.remove(m.from.x, m.from.y);

        // Hash: remove piece from source
        self.hash ^= piece_key(piece.piece_type(), piece.color(), m.from.x, m.from.y);

        let mut undo_info = UndoMove {
            captured_piece: self.board.get_piece(m.to.x, m.to.y).copied(),
            old_en_passant: self.en_passant,
            old_halfmove_clock: self.halfmove_clock,
            old_hash: self.hash_stack.last().copied().unwrap_or(0),
            special_rights_removed: ArrayVec::new(),
            starting_square_restored: None,
            old_white_king_pos: None,
            old_black_king_pos: None,
            old_repetition: self.repetition,
            ep_captured_piece: None,
            old_effective_castling_rights: self.effective_castling_rights,
            old_castling_partner_counts: self.castling_partner_counts,
        };

        // Track king position updates for undo
        if piece.piece_type().is_royal() {
            if piece.color() == PlayerColor::White {
                undo_info.old_white_king_pos = self.white_king_pos;
                self.white_king_pos = Some(m.to);
            } else if piece.color() == PlayerColor::Black {
                undo_info.old_black_king_pos = self.black_king_pos;
                self.black_king_pos = Some(m.to);
            }
        }

        // Once a piece moves from its original square, we no longer treat
        // that coordinate as an undeveloped starting square. Record this so
        // undo_move can restore starting_squares.
        if self.starting_squares.remove(&from_coord) {
            undo_info.starting_square_restored = Some(from_coord);
        }

        // Handle captures
        let is_capture = undo_info.captured_piece.is_some();

        if let Some(captured) = &undo_info.captured_piece {
            // Hash: For non-neutral pieces, XOR out (remove from hash).
            // For neutral pieces (obstacles), they weren't in the initial hash,
            // so XORing their key IN when captured creates a unique hash for
            // positions where different obstacles have been removed.
            // The XOR operation is symmetric, so the same line works for both:
            // - Non-neutral: was in hash, XOR removes it
            // - Neutral: wasn't in hash, XOR adds "removed obstacle" marker
            self.hash ^= piece_key(captured.piece_type(), captured.color(), m.to.x, m.to.y);

            // Update spatial indices for captured piece on destination square
            self.spatial_indices.remove(m.to.x, m.to.y);

            // If a royal piece was captured, clear the king position for that side
            // This is critical for has_lost_by_royal_capture() to detect wins
            if captured.piece_type().is_royal() {
                if captured.color() == PlayerColor::White {
                    undo_info.old_white_king_pos = self.white_king_pos;
                    self.white_king_pos = None;
                } else if captured.color() == PlayerColor::Black {
                    undo_info.old_black_king_pos = self.black_king_pos;
                    self.black_king_pos = None;
                }
            }

            // Only update material/piece counts for non-neutral pieces
            if captured.color() != PlayerColor::Neutral {
                // Update material hash (subtractive)
                self.material_hash = self
                    .material_hash
                    .wrapping_sub(material_key(captured.piece_type(), captured.color()));

                let value = get_piece_value(captured.piece_type());
                if captured.color() == PlayerColor::White {
                    self.material_score -= value;
                    self.white_piece_count = self.white_piece_count.saturating_sub(1);
                    if captured.piece_type() == PieceType::Pawn {
                        self.white_pawn_count = self.white_pawn_count.saturating_sub(1);
                    }
                } else {
                    self.material_score += value;
                    self.black_piece_count = self.black_piece_count.saturating_sub(1);
                    if captured.piece_type() == PieceType::Pawn {
                        self.black_pawn_count = self.black_pawn_count.saturating_sub(1);
                    }
                }
            }
        }

        // Handle En Passant capture
        let mut is_ep_capture = false;
        if let Some((ep, captured_pawn)) = self
            .en_passant
            .as_ref()
            .filter(|ep| {
                piece.piece_type() == PieceType::Pawn
                    && m.to.x == ep.square.x
                    && m.to.y == ep.square.y
            })
            .and_then(|ep| {
                self.board
                    .remove_piece(&ep.pawn_square.x, &ep.pawn_square.y)
                    .map(|p| (ep, p))
            })
        {
            is_ep_capture = true;
            // Store the actual captured piece for proper undo restoration
            undo_info.ep_captured_piece = Some(captured_pawn);
            // Hash: remove EP captured pawn
            self.hash ^= piece_key(
                captured_pawn.piece_type(),
                captured_pawn.color(),
                ep.pawn_square.x,
                ep.pawn_square.y,
            );
            // Update spatial indices for EP captured pawn
            self.spatial_indices
                .remove(ep.pawn_square.x, ep.pawn_square.y);

            // Update material hash (subtractive) for EP capture
            self.material_hash = self.material_hash.wrapping_sub(material_key(
                captured_pawn.piece_type(),
                captured_pawn.color(),
            ));

            let value = get_piece_value(captured_pawn.piece_type());
            if captured_pawn.color() == PlayerColor::White {
                self.material_score -= value;
                self.white_piece_count = self.white_piece_count.saturating_sub(1);
                self.white_pawn_count = self.white_pawn_count.saturating_sub(1);
            } else {
                self.material_score += value;
                self.black_piece_count = self.black_piece_count.saturating_sub(1);
                self.black_pawn_count = self.black_pawn_count.saturating_sub(1);
            }
        }

        // Handle Promotion material update
        if let Some(promo_type) = m.promotion {
            // Update material hash: remove pawn, add promoted piece
            self.material_hash = self
                .material_hash
                .wrapping_sub(material_key(PieceType::Pawn, piece.color()));
            self.material_hash = self
                .material_hash
                .wrapping_add(material_key(promo_type, piece.color()));

            let pawn_val = get_piece_value(PieceType::Pawn);
            let promo_val = get_piece_value(promo_type);
            if piece.color() == PlayerColor::White {
                self.material_score -= pawn_val;
                self.material_score += promo_val;
                self.white_pawn_count = self.white_pawn_count.saturating_sub(1);
                self.white_non_pawn_material = true;
            } else {
                self.material_score += pawn_val;
                self.material_score -= promo_val;
                self.black_pawn_count = self.black_pawn_count.saturating_sub(1);
                self.black_non_pawn_material = true;
            }
        }

        // Hash: remove old en passant
        if let Some(ep) = &self.en_passant {
            self.hash ^= en_passant_key(ep.square.x, ep.square.y);
        }

        // === INCREMENTAL CASTLING UPDATES (O(1)) ===
        // XOR out the old effective castling keys
        self.hash ^= crate::search::zobrist::castling_rights_key_from_bitfield(
            self.effective_castling_rights,
        );

        // Update rights for the moving piece
        if self.special_rights.remove(&m.from) {
            undo_info.special_rights_removed.push(m.from);

            if piece.piece_type() == PieceType::Pawn {
                self.hash ^= pawn_special_right_key(m.from.x, m.from.y);
            } else if piece.piece_type().is_royal() {
                // King moves: loses ALL castling rights for its side
                if piece.color() == PlayerColor::White {
                    self.effective_castling_rights &= !3;
                } else {
                    self.effective_castling_rights &= !12;
                }
            } else {
                // Non-pawn piece moves: could be a castling partner
                if let Some(k_pos) = if piece.color() == PlayerColor::White {
                    self.white_king_pos
                } else {
                    self.black_king_pos
                } {
                    if m.from.y == k_pos.y {
                        let idx = if piece.color() == PlayerColor::White {
                            if m.from.x > k_pos.x { 0 } else { 1 }
                        } else {
                            if m.from.x > k_pos.x { 2 } else { 3 }
                        };
                        self.castling_partner_counts[idx] =
                            self.castling_partner_counts[idx].saturating_sub(1);
                        if self.castling_partner_counts[idx] == 0 {
                            self.effective_castling_rights &= !(1 << idx);
                        }
                    }
                }
            }
        }

        // Update rights for the captured piece
        if is_capture && self.special_rights.remove(&m.to) {
            let captured = undo_info.captured_piece.unwrap();
            undo_info.special_rights_removed.push(m.to);

            if captured.piece_type() == PieceType::Pawn {
                self.hash ^= pawn_special_right_key(m.to.x, m.to.y);
            } else if captured.piece_type().is_royal() {
                // King captured: its side loses ALL castling rights
                if captured.color() == PlayerColor::White {
                    self.effective_castling_rights &= !3;
                } else {
                    self.effective_castling_rights &= !12;
                }
            } else {
                // Non-pawn partner captured
                if let Some(k_pos) = if captured.color() == PlayerColor::White {
                    self.white_king_pos
                } else {
                    self.black_king_pos
                } {
                    if m.to.y == k_pos.y {
                        let idx = if captured.color() == PlayerColor::White {
                            if m.to.x > k_pos.x { 0 } else { 1 }
                        } else {
                            if m.to.x > k_pos.x { 2 } else { 3 }
                        };
                        self.castling_partner_counts[idx] =
                            self.castling_partner_counts[idx].saturating_sub(1);
                        if self.castling_partner_counts[idx] == 0 {
                            self.effective_castling_rights &= !(1 << idx);
                        }
                    }
                }
            }
        }

        // Handle Castling Move (King moves > 1 square)
        if piece.piece_type() == PieceType::King
            && (m.to.x - m.from.x).abs() > 1
            && let Some(rook_coord) = &m.rook_coord
            && let Some(rook) = self.board.remove_piece(&rook_coord.x, &rook_coord.y)
        {
            let dx = m.to.x - m.from.x;
            let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
            // Hash: remove rook from original, add at new position
            self.hash ^= piece_key(rook.piece_type(), rook.color(), rook_coord.x, rook_coord.y);
            self.hash ^= piece_key(rook.piece_type(), rook.color(), rook_to_x, m.from.y);
            self.board.set_piece(rook_to_x, m.from.y, rook);
            // Update spatial indices for rook move
            self.spatial_indices.remove(rook_coord.x, rook_coord.y);
            self.spatial_indices.add(rook_to_x, m.from.y, rook.packed());

            // Rook also loses special rights
            if self.special_rights.remove(rook_coord) {
                undo_info.special_rights_removed.push(*rook_coord);
                // This rook was a castling partner, decrement its count
                if let Some(k_pos) = if rook.color() == PlayerColor::White {
                    self.white_king_pos
                } else {
                    self.black_king_pos
                } {
                    if rook_coord.y == k_pos.y {
                        let idx = if rook.color() == PlayerColor::White {
                            if rook_coord.x > k_pos.x { 0 } else { 1 }
                        } else {
                            if rook_coord.x > k_pos.x { 2 } else { 3 }
                        };
                        self.castling_partner_counts[idx] =
                            self.castling_partner_counts[idx].saturating_sub(1);
                        if self.castling_partner_counts[idx] == 0 {
                            self.effective_castling_rights &= !(1 << idx);
                        }
                    }
                }
            }
        }

        // XOR in the new effective castling keys
        self.hash ^= crate::search::zobrist::castling_rights_key_from_bitfield(
            self.effective_castling_rights,
        );

        // Move piece (handle promotion if needed)
        let final_piece = if let Some(promo_type) = m.promotion {
            Piece::new(promo_type, piece.color())
        } else {
            piece
        };

        // Hash: add piece at destination
        self.hash ^= piece_key(
            final_piece.piece_type(),
            final_piece.color(),
            m.to.x,
            m.to.y,
        );
        self.board.set_piece(m.to.x, m.to.y, final_piece);
        // Update spatial indices for moved piece on destination square
        self.spatial_indices
            .add(m.to.x, m.to.y, final_piece.packed());

        // Update En Passant state
        self.en_passant = None;
        if piece.piece_type() == PieceType::Pawn {
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
        if piece.piece_type() == PieceType::Pawn || is_capture || is_ep_capture {
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

        // Stockfish-style repetition detection: compute distance to previous occurrence
        // of same position. 0 = no repetition, positive = distance to twofold, negative = threefold.
        self.repetition = 0;
        let end = (self.halfmove_clock as usize).min(self.hash_stack.len());
        if end >= 4 {
            let current_hash = self.hash;
            // Check every 2 plies (same side to move)
            let mut i = 4usize;
            let mut first_match: Option<i32> = None;
            while i <= end {
                let idx = self.hash_stack.len().saturating_sub(i);
                if idx < self.hash_stack.len() && self.hash_stack[idx] == current_hash {
                    if first_match.is_none() {
                        // First match: store distance as positive (twofold)
                        first_match = Some(i as i32);
                        // Continue searching for a second match (threefold)
                    } else if let Some(m) = first_match {
                        // Second match: this is threefold! Store as negative.
                        self.repetition = -m;
                        break;
                    }
                }
                i += 2;
            }
            // If we only found one match, store it as positive (twofold)
            if self.repetition == 0
                && let Some(dist) = first_match
            {
                self.repetition = dist;
            }
        }

        undo_info
    }

    pub fn undo_move(&mut self, m: &Move, undo: UndoMove) {
        use crate::search::zobrist::material_key;

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
            // Convert back to pawn: Remove promo type, Add pawn type
            self.material_hash = self
                .material_hash
                .wrapping_sub(material_key(piece.piece_type(), piece.color()));
            self.material_hash = self
                .material_hash
                .wrapping_add(material_key(PieceType::Pawn, piece.color()));

            // Convert back to pawn
            let promo_val = get_piece_value(piece.piece_type());
            let pawn_val = get_piece_value(PieceType::Pawn);

            if piece.color() == PlayerColor::White {
                self.material_score -= promo_val;
                self.material_score += pawn_val;
            } else {
                self.material_score += promo_val;
                self.material_score -= pawn_val;
            }
            piece = Piece::new(PieceType::Pawn, piece.color());
        }

        // Move back to 'from'
        self.board.set_piece(m.from.x, m.from.y, piece);
        // Update spatial indices for moved piece back on source square
        self.spatial_indices.add(m.from.x, m.from.y, piece.packed());

        // Restore captured piece
        if let Some(captured) = undo.captured_piece {
            // Restore material hash
            if captured.color() != PlayerColor::Neutral {
                self.material_hash = self
                    .material_hash
                    .wrapping_add(material_key(captured.piece_type(), captured.color()));
            }

            // Only update piece counts and material for non-neutral pieces
            if captured.color() != PlayerColor::Neutral {
                let value = get_piece_value(captured.piece_type());
                if captured.color() == PlayerColor::White {
                    self.material_score += value;
                    self.white_piece_count = self.white_piece_count.saturating_add(1);
                    if captured.piece_type() == PieceType::Pawn {
                        self.white_pawn_count = self.white_pawn_count.saturating_add(1);
                    }
                } else {
                    self.material_score -= value;
                    self.black_piece_count = self.black_piece_count.saturating_add(1);
                    if captured.piece_type() == PieceType::Pawn {
                        self.black_pawn_count = self.black_pawn_count.saturating_add(1);
                    }
                }
            }
            self.board.set_piece(m.to.x, m.to.y, captured);
            // Update spatial indices for restored captured piece
            self.spatial_indices.add(m.to.x, m.to.y, captured.packed());
        }

        // Handle En Passant capture undo - use the stored captured piece
        if let Some(captured_pawn) = undo.ep_captured_piece {
            if let Some(ep) = &undo.old_en_passant {
                // Restore the captured piece (could be a pawn or promoted piece)
                self.board
                    .set_piece(ep.pawn_square.x, ep.pawn_square.y, captured_pawn);
                self.spatial_indices.add(
                    ep.pawn_square.x,
                    ep.pawn_square.y,
                    captured_pawn.packed(),
                );

                // Restore material hash
                self.material_hash = self.material_hash.wrapping_add(material_key(
                    captured_pawn.piece_type(),
                    captured_pawn.color(),
                ));

                // Restore material value and piece counts
                let value = get_piece_value(captured_pawn.piece_type());
                if captured_pawn.color() == PlayerColor::White {
                    self.material_score += value;
                    self.white_piece_count = self.white_piece_count.saturating_add(1);
                    self.white_pawn_count = self.white_pawn_count.saturating_add(1);
                } else {
                    self.material_score -= value;
                    self.black_piece_count = self.black_piece_count.saturating_add(1);
                    self.black_pawn_count = self.black_pawn_count.saturating_add(1);
                }
            }
        }

        // Handle Castling Revert
        if piece.piece_type() == PieceType::King {
            let dx = m.to.x - m.from.x;
            if dx.abs() > 1 {
                // Castling was performed. Move rook back.
                if let Some(rook_coord) = &m.rook_coord {
                    let rook_to_x = m.from.x + (if dx > 0 { 1 } else { -1 });
                    if let Some(rook) = self.board.remove_piece(&rook_to_x, &m.from.y) {
                        self.board.set_piece(rook_coord.x, rook_coord.y, rook);
                        // Update spatial indices for rook moved back
                        self.spatial_indices.remove(rook_to_x, m.from.y);
                        self.spatial_indices
                            .add(rook_coord.x, rook_coord.y, rook.packed());
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
        // Restore king positions if they were saved (i.e., a king moved)
        if let Some(pos) = undo.old_white_king_pos {
            self.white_king_pos = Some(pos);
        }
        if let Some(pos) = undo.old_black_king_pos {
            self.black_king_pos = Some(pos);
        }
        self.halfmove_clock = undo.old_halfmove_clock;
        self.repetition = undo.old_repetition;

        // Restore castling state
        self.effective_castling_rights = undo.old_effective_castling_rights;
        self.castling_partner_counts = undo.old_castling_partner_counts;
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }
        // One move buffer per ply so recursion doesn't overwrite
        let mut bufs: Vec<MoveList> = (0..=depth).map(|_| MoveList::new()).collect();
        self.perft_buf(depth, 0, &mut bufs)
    }

    fn perft_buf(&mut self, depth: usize, ply: usize, bufs: &mut [MoveList]) -> u64 {
        if depth == 0 {
            return 1;
        }

        bufs[ply].clear();
        self.get_legal_moves_into(&mut bufs[ply]);

        let move_count = bufs[ply].len();
        let mut nodes = 0u64;
        for i in 0..move_count {
            let m = bufs[ply][i];
            let undo = self.make_move(&m);
            nodes += self.perft_buf(depth - 1, ply + 1, bufs);
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

        // Explicitly set standard promotion ranks (8 for white, 1 for black)
        self.game_rules.promotion_ranks = Some(PromotionRanks {
            white: vec![8],
            black: vec![1],
        });
        self.white_promo_rank = 8;
        self.black_promo_rank = 1;

        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);

        // Rebuild piece lists and counts
        self.recompute_piece_counts();

        // Compute initial hash
        self.recompute_hash();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a minimal game state for testing
    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        // Add kings so the position is "legal"
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();
        game
    }

    // ======================== 50-Move Rule Tests ========================

    #[test]
    fn test_is_fifty_returns_false_when_no_rule() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = None;
        game.halfmove_clock = 200;
        assert!(
            !game.is_fifty(),
            "Should never trigger draw when no move rule limit"
        );
    }

    #[test]
    fn test_is_fifty_returns_false_below_limit() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = Some(100);
        game.halfmove_clock = 99;
        assert!(!game.is_fifty(), "Should not trigger draw below limit");
    }

    #[test]
    fn test_is_fifty_returns_true_at_limit() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = Some(100);
        game.halfmove_clock = 100;
        assert!(game.is_fifty(), "Should trigger draw at limit");
    }

    #[test]
    fn test_is_fifty_returns_true_above_limit() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = Some(100);
        game.halfmove_clock = 150;
        assert!(game.is_fifty(), "Should trigger draw above limit");
    }

    #[test]
    fn test_is_fifty_respects_null_move() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = Some(100);
        game.halfmove_clock = 100;
        game.null_moves = 1;
        assert!(
            !game.is_fifty(),
            "Should not trigger during null move search"
        );
    }

    #[test]
    fn test_is_fifty_custom_limit() {
        let mut game = create_test_game();
        game.game_rules.move_rule_limit = Some(150); // 75-move rule
        game.halfmove_clock = 149;
        assert!(!game.is_fifty());
        game.halfmove_clock = 150;
        assert!(game.is_fifty());
    }

    // ======================== Repetition Tests ========================

    #[test]
    fn test_rose_check_detection() {
        let mut game = GameState::new();
        game.board = Board::new();
        game.special_rights.clear();
        game.turn = PlayerColor::Black;

        // White rose at (3,7)
        game.board
            .set_piece(3, 7, Piece::new(PieceType::Rose, PlayerColor::White));

        // Black king at (5,8)
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));

        // White king somewhere else
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));

        game.recompute_piece_counts();
        game.recompute_hash();

        // Build spatial indices
        game.spatial_indices = SpatialIndices::new(&game.board);

        // Test: The king at (5,8) should be attacked by the white rose at (3,7)
        let king_pos = Coordinate::new(5, 8);
        let is_attacked = crate::moves::is_square_attacked(
            &game.board,
            &king_pos,
            PlayerColor::White,
            &game.spatial_indices,
        );

        assert!(
            is_attacked,
            "Rose at (3,7) should give check to king at (5,8)"
        );

        // Also test that the game correctly identifies it's in check
        assert!(game.is_in_check(), "Game should report being in check");

        // And test that non-king moves are illegal (must respond to check)
        let mut all_moves = MoveList::new();
        game.get_legal_moves_into(&mut all_moves);

        // All legal moves should either move the king or block/capture the rose
        for m in all_moves.iter() {
            if m.piece.piece_type() != PieceType::King {
                // If it's not a king move, it must capture the rose
                assert_eq!(
                    (m.to.x, m.to.y),
                    (3, 7),
                    "Non-king move {:?} should capture the rose at (3,7)",
                    m
                );
            }
        }
    }

    #[test]
    fn test_distant_slider_block() {
        let mut game = GameState::new();
        game.board = Board::new();
        game.special_rights.clear();
        game.turn = PlayerColor::White;

        // White King at (-4, -2)
        game.board
            .set_piece(-4, -2, Piece::new(PieceType::King, PlayerColor::White));
        // White Rook at (7, 2)
        game.board
            .set_piece(7, 2, Piece::new(PieceType::Rook, PlayerColor::White));
        // Black Bishop at (26, -32) - giving diagonal check
        game.board
            .set_piece(26, -32, Piece::new(PieceType::Bishop, PlayerColor::Black));
        // Black King
        game.board
            .set_piece(100, 100, Piece::new(PieceType::King, PlayerColor::Black));

        game.recompute_piece_counts();
        game.recompute_hash();

        assert!(game.is_in_check(), "White king should be in check");

        let mut moves = MoveList::new();
        game.get_legal_moves_into(&mut moves);

        // Rook at (7, 2) should be able to move to (7, -13) to block
        let can_block = moves
            .iter()
            .any(|m| m.from.x == 7 && m.from.y == 2 && m.to.x == 7 && m.to.y == -13);
        assert!(
            can_block,
            "Rook should be able to block distant bishop check"
        );
    }

    #[test]
    fn test_is_repetition_no_repetition() {
        let game = create_test_game();
        assert!(!game.is_repetition(5), "No repetition initially");
    }

    #[test]
    fn test_is_repetition_twofold_within_search() {
        let mut game = create_test_game();
        game.repetition = 2; // Positive = twofold, distance = 2
        // Only draw if repetition < ply (both occurrences in search tree)
        assert!(!game.is_repetition(1), "ply=1 < repetition=2, not a draw");
        assert!(!game.is_repetition(2), "ply=2 == repetition=2, not a draw");
        assert!(game.is_repetition(3), "ply=3 > repetition=2, is a draw");
    }

    #[test]
    fn test_is_repetition_threefold() {
        let mut game = create_test_game();
        game.repetition = -3; // Negative = threefold
        // Threefold is always a draw (negative is always < positive ply)
        assert!(game.is_repetition(1), "Threefold should always draw");
        assert!(game.is_repetition(10), "Threefold should always draw");
    }

    #[test]
    fn test_is_repetition_during_null_move() {
        let mut game = create_test_game();
        game.repetition = -3;
        game.null_moves = 1;
        assert!(!game.is_repetition(5), "Should not detect during null move");
    }

    // ======================== Null Move Tests ========================

    #[test]
    fn test_null_move_flips_turn() {
        let mut game = create_test_game();
        game.turn = PlayerColor::White;
        game.make_null_move();
        assert_eq!(game.turn, PlayerColor::Black);
        game.unmake_null_move();
        assert_eq!(game.turn, PlayerColor::White);
    }

    #[test]
    fn test_null_move_clears_en_passant() {
        let mut game = create_test_game();
        game.en_passant = Some(EnPassantState {
            square: Coordinate::new(4, 3),
            pawn_square: Coordinate::new(4, 4),
        });
        game.make_null_move();
        assert!(game.en_passant.is_none(), "En passant should be cleared");
        game.unmake_null_move();
        // Note: en_passant is not restored by unmake_null_move (standard behavior)
    }

    #[test]
    fn test_null_move_increments_counter() {
        let mut game = create_test_game();
        assert_eq!(game.null_moves, 0);
        game.make_null_move();
        assert_eq!(game.null_moves, 1);
        game.make_null_move();
        assert_eq!(game.null_moves, 2);
        game.unmake_null_move();
        assert_eq!(game.null_moves, 1);
    }

    #[test]
    fn test_null_move_hash_restored() {
        let mut game = create_test_game();
        let original_hash = game.hash;
        game.make_null_move();
        assert_ne!(game.hash, original_hash, "Hash should change");
        game.unmake_null_move();
        assert_eq!(game.hash, original_hash, "Hash should be restored");
    }

    // ======================== King Position Tests ========================

    #[test]
    fn test_king_positions_tracked() {
        let mut game = GameState::new();
        game.board
            .set_piece(3, 3, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();

        assert_eq!(game.white_king_pos, Some(Coordinate::new(3, 3)));
        assert_eq!(game.black_king_pos, Some(Coordinate::new(7, 7)));
    }

    #[test]
    fn test_royal_centaur_tracked_as_king() {
        let mut game = GameState::new();
        game.board.set_piece(
            4,
            4,
            Piece::new(PieceType::RoyalCentaur, PlayerColor::White),
        );
        game.board
            .set_piece(6, 6, Piece::new(PieceType::RoyalQueen, PlayerColor::Black));
        game.recompute_piece_counts();

        assert_eq!(game.white_king_pos, Some(Coordinate::new(4, 4)));
        assert_eq!(game.black_king_pos, Some(Coordinate::new(6, 6)));
    }

    // ======================== Piece Count Tests ========================

    #[test]
    fn test_piece_counts_accurate() {
        let mut game = GameState::new();
        // Add pieces
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(3, 1, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.board
            .set_piece(4, 1, Piece::new(PieceType::Pawn, PlayerColor::White));

        game.board
            .set_piece(1, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(2, 8, Piece::new(PieceType::Rook, PlayerColor::Black));

        game.recompute_piece_counts();

        assert_eq!(game.white_piece_count, 4);
        assert_eq!(game.black_piece_count, 2);
        assert_eq!(game.white_pawn_count, 2);
        assert_eq!(game.black_pawn_count, 0);
    }

    #[test]
    fn test_non_pawn_material_tracked() {
        let mut game = GameState::new();
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();

        assert!(
            !game.has_non_pawn_material(PlayerColor::White),
            "Only king = no NPM"
        );

        game.board
            .set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();

        assert!(
            game.has_non_pawn_material(PlayerColor::White),
            "Knight = has NPM"
        );
    }

    // ======================== Hash Consistency Tests ========================

    #[test]
    fn test_hash_changes_on_move() {
        let mut game = GameState::new();
        game.setup_standard_chess();
        let initial_hash = game.hash;

        // Make a simple pawn move
        let moves = game.get_legal_moves();
        if let Some(m) = moves.first() {
            let _undo = game.make_move(m);
            assert_ne!(game.hash, initial_hash, "Hash should change after move");
        }
    }

    #[test]
    fn test_hash_restored_on_unmake() {
        let mut game = GameState::new();
        game.setup_standard_chess();
        let initial_hash = game.hash;

        let moves = game.get_legal_moves();
        if let Some(m) = moves.first() {
            let undo = game.make_move(m);
            game.undo_move(m, undo);
            assert_eq!(
                game.hash, initial_hash,
                "Hash should be restored after undo"
            );
        }
    }

    #[test]
    fn test_recompute_hash_matches_incremental() {
        let mut game = GameState::new();
        game.setup_standard_chess();

        // Make several moves
        for _ in 0..5 {
            let moves = game.get_legal_moves();
            let legal_moves: Vec<_> = moves
                .iter()
                .filter(|m| {
                    let undo = game.make_move(m);
                    let illegal = game.is_move_illegal();
                    game.undo_move(m, undo);
                    !illegal
                })
                .collect();

            if let Some(m) = legal_moves.first() {
                let _undo = game.make_move(m);
            } else {
                break;
            }
        }

        let incremental_hash = game.hash;
        game.recompute_hash();
        assert_eq!(
            game.hash, incremental_hash,
            "Recomputed hash should match incremental"
        );
    }

    // ======================== Move Make/Unmake Tests ========================

    #[test]
    fn test_halfmove_clock_increments() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();
        game.recompute_hash();

        // Knight move (not pawn, not capture) should increment
        let m = Move {
            from: Coordinate::new(4, 4),
            to: Coordinate::new(5, 6),
            piece: Piece::new(PieceType::Knight, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };

        game.halfmove_clock = 10;
        let _undo = game.make_move(&m);
        assert_eq!(game.halfmove_clock, 11, "Should increment on quiet move");
    }

    #[test]
    fn test_halfmove_clock_resets_on_pawn() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.recompute_piece_counts();
        game.recompute_hash();

        let m = Move {
            from: Coordinate::new(4, 2),
            to: Coordinate::new(4, 3),
            piece: Piece::new(PieceType::Pawn, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };

        game.halfmove_clock = 50;
        let _undo = game.make_move(&m);
        assert_eq!(game.halfmove_clock, 0, "Should reset on pawn move");
    }

    #[test]
    fn test_halfmove_clock_resets_on_capture() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        game.board
            .set_piece(5, 6, Piece::new(PieceType::Pawn, PlayerColor::Black));
        game.recompute_piece_counts();
        game.recompute_hash();

        let m = Move {
            from: Coordinate::new(4, 4),
            to: Coordinate::new(5, 6),
            piece: Piece::new(PieceType::Knight, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };

        game.halfmove_clock = 50;
        let _undo = game.make_move(&m);
        assert_eq!(game.halfmove_clock, 0, "Should reset on capture");
    }

    #[test]
    fn test_halfmove_clock_restored_on_unmake() {
        let mut game = create_test_game();
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        game.recompute_piece_counts();
        game.recompute_hash();

        let m = Move {
            from: Coordinate::new(4, 4),
            to: Coordinate::new(5, 6),
            piece: Piece::new(PieceType::Knight, PlayerColor::White),
            promotion: None,
            rook_coord: None,
        };

        game.halfmove_clock = 42;
        let undo = game.make_move(&m);
        game.undo_move(&m, undo);
        assert_eq!(game.halfmove_clock, 42, "Should restore halfmove clock");
    }

    // ======================== Lone King Endgame Tests ========================

    #[test]
    fn test_is_lone_king_endgame_both_have_pieces() {
        let mut game = GameState::new();
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(2, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.recompute_piece_counts();

        assert!(!game.is_lone_king_endgame());
    }

    #[test]
    fn test_is_lone_king_endgame_white_lone_king() {
        let mut game = GameState::new();
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(2, 8, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.recompute_piece_counts();

        assert!(game.is_lone_king_endgame());
    }

    #[test]
    fn test_is_lone_king_endgame_black_lone_king() {
        let mut game = GameState::new();
        game.board
            .set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(2, 1, Piece::new(PieceType::Queen, PlayerColor::White));
        game.board
            .set_piece(1, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.recompute_piece_counts();

        assert!(game.is_lone_king_endgame());
    }

    // ======================== Check Detection Tests ========================

    #[test]
    fn test_is_in_check_basic() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(5, 4, Piece::new(PieceType::Rook, PlayerColor::Black)); // Checking white king
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        assert!(
            game.is_in_check(),
            "White king should be in check from rook"
        );
    }

    #[test]
    fn test_is_in_check_knight() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 3, Piece::new(PieceType::Knight, PlayerColor::Black)); // Checking white king
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        assert!(
            game.is_in_check(),
            "White king should be in check from knight"
        );
    }

    #[test]
    fn test_is_in_check_no_check() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        // Place rook NOT on any line with the king (not same row, column, or diagonal)
        game.board
            .set_piece(3, 4, Piece::new(PieceType::Rook, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        assert!(!game.is_in_check(), "White king should not be in check");
    }

    // ======================== Standard Chess Setup Tests ========================

    #[test]
    fn test_setup_standard_chess() {
        let mut game = GameState::new();
        game.setup_standard_chess();

        // Check piece counts
        assert_eq!(game.white_piece_count, 16);
        assert_eq!(game.black_piece_count, 16);

        // Check king positions
        assert_eq!(game.white_king_pos, Some(Coordinate::new(5, 1)));
        assert_eq!(game.black_king_pos, Some(Coordinate::new(5, 8)));

        // Check it's white's turn
        assert_eq!(game.turn, PlayerColor::White);

        // Check promotion ranks set
        assert_eq!(game.white_promo_rank, 8);
        assert_eq!(game.black_promo_rank, 1);
    }

    #[test]
    fn test_standard_chess_has_moves() {
        let mut game = GameState::new();
        game.setup_standard_chess();

        let moves = game.get_legal_moves();
        // In infinite chess, sliders can have many more moves than classical chess
        // Just verify we have some moves available
        assert!(!moves.is_empty(), "Should have legal moves at start");
    }

    #[test]
    fn test_make_undo_move_restores_state() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        let original_hash = game.hash;
        let original_piece_count = game.white_piece_count;

        // Create and make a move
        let mv = Move::new(
            Coordinate::new(4, 2),
            Coordinate::new(4, 3),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );

        let undo = game.make_move(&mv);
        assert!(
            game.board.get_piece(4, 3).is_some(),
            "Pawn should be at new position"
        );
        assert!(
            game.board.get_piece(4, 2).is_none(),
            "Original position should be empty"
        );

        game.undo_move(&mv, undo);
        assert!(
            game.board.get_piece(4, 2).is_some(),
            "Pawn should be restored"
        );
        assert!(
            game.board.get_piece(4, 3).is_none(),
            "New position should be empty after undo"
        );
        assert_eq!(game.hash, original_hash, "Hash should be restored");
        assert_eq!(
            game.white_piece_count, original_piece_count,
            "Piece count should match"
        );
    }

    #[test]
    fn test_has_special_right() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.special_rights.insert(Coordinate::new(5, 1));

        assert!(game.has_special_right(&Coordinate::new(5, 1)));
        assert!(!game.has_special_right(&Coordinate::new(1, 1)));
    }

    #[test]
    fn test_halfmove_clock_increment() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();

        let initial_clock = game.halfmove_clock;

        // Make a quiet rook move (non-pawn, non-capture)
        let mv = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 5),
            Piece::new(PieceType::Rook, PlayerColor::White),
        );
        let _undo = game.make_move(&mv);

        assert_eq!(
            game.halfmove_clock,
            initial_clock + 1,
            "Halfmove clock should increment for quiet move"
        );
    }

    #[test]
    fn test_halfmove_clock_resets_on_pawn_move() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        game.turn = PlayerColor::White;
        game.halfmove_clock = 10; // Simulate some moves having been made
        game.recompute_piece_counts();

        let mv = Move::new(
            Coordinate::new(4, 2),
            Coordinate::new(4, 3),
            Piece::new(PieceType::Pawn, PlayerColor::White),
        );
        let _undo = game.make_move(&mv);

        assert_eq!(
            game.halfmove_clock, 0,
            "Halfmove clock should reset on pawn move"
        );
    }

    #[test]
    fn test_is_repetition_basic() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.turn = PlayerColor::White;
        game.recompute_piece_counts();
        game.recompute_hash();

        // At ply 0, no repetition initially
        assert!(!game.is_repetition(0));
    }

    #[test]
    fn test_has_non_pawn_material() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        game.board
            .set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        game.recompute_piece_counts();

        assert!(
            game.has_non_pawn_material(PlayerColor::White),
            "White should have non-pawn material"
        );
        assert!(
            !game.has_non_pawn_material(PlayerColor::Black),
            "Black should not have non-pawn material (only king)"
        );
    }

    #[test]
    fn test_init_starting_squares() {
        let mut game = GameState::new();
        game.board
            .set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        game.board
            .set_piece(1, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.board
            .set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White));
        game.init_starting_squares();

        assert!(game.starting_squares.contains(&Coordinate::new(1, 1)));
        assert!(game.starting_squares.contains(&Coordinate::new(8, 1)));
    }
}
