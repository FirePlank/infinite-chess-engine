#[cfg(any(test, not(target_arch = "wasm32")))]
use crate::Variant;
#[cfg(any(test, not(target_arch = "wasm32")))]
use crate::evaluation::calculate_initial_material;

use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
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
/// - RoyalCapture: Win by capturing one of the opponent's royal pieces
/// - AllRoyalsCaptured: Win when all of the opponent's royal pieces are captured
/// - AllPiecesCaptured: Win when all of the opponent's pieces are captured
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

    /// Parse a win condition from a string.
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
    /// Select the most appropriate win condition from a list based on priority.
    /// Priority: Checkmate > RoyalCapture > AllRoyalsCaptured > AllPiecesCaptured.
    pub fn select(conditions: &[WinCondition], opponent_has_royal: bool) -> Self {
        if !opponent_has_royal {
            return WinCondition::AllPiecesCaptured;
        }
        if conditions.contains(&WinCondition::Checkmate) {
            WinCondition::Checkmate
        } else if conditions.contains(&WinCondition::RoyalCapture) {
            WinCondition::RoyalCapture
        } else if conditions.contains(&WinCondition::AllRoyalsCaptured) {
            WinCondition::AllRoyalsCaptured
        } else if conditions.contains(&WinCondition::AllPiecesCaptured) {
            WinCondition::AllPiecesCaptured
        } else {
            WinCondition::Checkmate // Default
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
    /// Win condition for White: what White must do to beat Black.
    /// E.g., "checkmate" means White must checkmate Black to win.
    #[serde(skip)]
    pub white_win_condition: WinCondition,
    /// Win condition for Black: what Black must do to beat White.
    /// E.g., "allpiecescaptured" means Black must capture all of White's pieces to win.
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

/// Entry in move history for repetition detection.
/// Stores move details to check if a reversal would repeat a position.
#[derive(Clone, Copy, Debug)]
pub struct MoveHistoryEntry {
    pub from_x: i64,
    pub from_y: i64,
    pub to_x: i64,
    pub to_y: i64,
    pub piece_type: PieceType,
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
    /// Repetition information: distance to previous occurrence of same position.
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
    /// Pinned pieces for white: maps (x, y) of a WHITE piece to (dx, dy) pin direction.
    /// A piece at (x,y) pinned with direction (dx,dy) can only move along that line.
    /// This is the direction FROM the king TO the pinner (through the pinned piece).
    /// Updated by recompute_check_squares(). Used for fast legality checks (C1 optimization).
    #[serde(skip)]
    pub pinned_white: rustc_hash::FxHashMap<(i64, i64), (i64, i64)>,
    /// Pinned pieces for black
    #[serde(skip)]
    pub pinned_black: rustc_hash::FxHashMap<(i64, i64), (i64, i64)>,
    /// Number of pieces currently checking the white king.
    #[serde(skip)]
    pub checkers_count_white: u8,
    /// Number of pieces currently checking the black king.
    #[serde(skip)]
    pub checkers_count_black: u8,
    /// Move history for repetition detection.
    /// Stores (from, to, piece_type) for each move.
    #[serde(skip)]
    pub move_history: Vec<MoveHistoryEntry>,
    /// Search ply distance from last null move.
    #[serde(skip)]
    pub plies_from_null: u32,
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
            pinned_white: rustc_hash::FxHashMap::default(),
            pinned_black: rustc_hash::FxHashMap::default(),
            checkers_count_white: 0,
            checkers_count_black: 0,
            move_history: Vec::with_capacity(128),
            plies_from_null: 0,
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
            pinned_white: rustc_hash::FxHashMap::default(),
            pinned_black: rustc_hash::FxHashMap::default(),
            checkers_count_white: 0,
            checkers_count_black: 0,
            move_history: Vec::with_capacity(128),
            plies_from_null: 0,
        }
    }

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
                if let Some(piece) = self.board.get_piece(coord.x, coord.y)
                    && piece.color() == PlayerColor::White
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
                if let Some(piece) = self.board.get_piece(coord.x, coord.y)
                    && piece.color() == PlayerColor::Black
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
        self.pinned_white.clear();
        self.pinned_black.clear();
        self.checkers_count_white = 0;
        self.checkers_count_black = 0;

        use crate::attacks::{is_diag_slider, is_ortho_slider};

        // White King Status (Attacks by Black pieces)
        if let Some(wk) = self.white_king_pos {
            // 1. Knight Checkers
            for (dx, dy) in KNIGHT_OFFSETS {
                let tx = wk.x + dx;
                let ty = wk.y + dy;
                if let Some(p) = self.board.get_piece(tx, ty)
                    && p.color() == PlayerColor::Black
                    && p.piece_type() == PieceType::Knight
                {
                    self.checkers_count_white += 1;
                }
                self.check_squares_white
                    .insert((tx, ty, PieceType::Knight as u8));
            }
            // 2. Pawn Checkers (Black pawns attack downward: y+1)
            for dx in [-1, 1] {
                let tx = wk.x + dx;
                let ty = wk.y + 1;
                if let Some(p) = self.board.get_piece(tx, ty)
                    && p.color() == PlayerColor::Black
                    && p.piece_type() == PieceType::Pawn
                {
                    self.checkers_count_white += 1;
                }
                self.check_squares_white
                    .insert((tx, ty, PieceType::Pawn as u8));
            }

            // 3. Slider Rays (Sliders & Pinned pieces)
            for (dir_idx, (dx, dy)) in DIRECTIONS.iter().enumerate() {
                if let Some((bx, by)) = self.find_first_blocker_on_ray(wk.x, wk.y, *dx, *dy) {
                    self.slider_rays_white[dir_idx] = Some((bx, by));
                    let p1 = self.board.get_piece(bx, by).unwrap();
                    let is_ortho = dir_idx < 4;

                    if p1.color() == PlayerColor::Black {
                        // Immediate Checker?
                        let pt1 = p1.piece_type();
                        if (is_ortho && is_ortho_slider(pt1)) || (!is_ortho && is_diag_slider(pt1))
                        {
                            self.checkers_count_white += 1;
                        }

                        // Potential Discovered check for Black (if bx,by moves)
                        if let Some((bx2, by2)) = self.find_first_blocker_on_ray(bx, by, *dx, *dy)
                            && let Some(p2) = self.board.get_piece(bx2, by2)
                            && p2.color() == PlayerColor::Black
                        {
                            let pt2 = p2.piece_type();
                            if (is_ortho && is_ortho_slider(pt2))
                                || (!is_ortho && is_diag_slider(pt2))
                            {
                                self.discovered_check_squares_black.insert((bx, by));
                            }
                        }
                    } else {
                        // Friendly piece - could be pinned?
                        if let Some((bx2, by2)) = self.find_first_blocker_on_ray(bx, by, *dx, *dy)
                            && let Some(p2) = self.board.get_piece(bx2, by2)
                            && p2.color() == PlayerColor::Black
                        {
                            let pt2 = p2.piece_type();
                            if (is_ortho && is_ortho_slider(pt2))
                                || (!is_ortho && is_diag_slider(pt2))
                            {
                                self.pinned_white.insert((bx, by), (*dx, *dy));
                            }
                        }
                    }
                }
            }
        }

        // Black King Status (Attacks by White pieces)
        if let Some(bk) = self.black_king_pos {
            // 1. Knight Checkers
            for (dx, dy) in KNIGHT_OFFSETS {
                let tx = bk.x + dx;
                let ty = bk.y + dy;
                if let Some(p) = self.board.get_piece(tx, ty)
                    && p.color() == PlayerColor::White
                    && p.piece_type() == PieceType::Knight
                {
                    self.checkers_count_black += 1;
                }
                self.check_squares_black
                    .insert((tx, ty, PieceType::Knight as u8));
            }
            // 2. Pawn Checkers (White pawns attack upward: y-1)
            for dx in [-1, 1] {
                let tx = bk.x + dx;
                let ty = bk.y - 1;
                if let Some(p) = self.board.get_piece(tx, ty)
                    && p.color() == PlayerColor::White
                    && p.piece_type() == PieceType::Pawn
                {
                    self.checkers_count_black += 1;
                }
                self.check_squares_black
                    .insert((tx, ty, PieceType::Pawn as u8));
            }

            // 3. Slider Rays (Sliders & Pinned pieces)
            for (dir_idx, (dx, dy)) in DIRECTIONS.iter().enumerate() {
                if let Some((bx, by)) = self.find_first_blocker_on_ray(bk.x, bk.y, *dx, *dy) {
                    self.slider_rays_black[dir_idx] = Some((bx, by));
                    let p1 = self.board.get_piece(bx, by).unwrap();
                    let is_ortho = dir_idx < 4;

                    if p1.color() == PlayerColor::White {
                        // Immediate Checker?
                        let pt1 = p1.piece_type();
                        if (is_ortho && is_ortho_slider(pt1)) || (!is_ortho && is_diag_slider(pt1))
                        {
                            self.checkers_count_black += 1;
                        }

                        // Potential Discovered check for White (if bx,by moves)
                        if let Some((bx2, by2)) = self.find_first_blocker_on_ray(bx, by, *dx, *dy)
                            && let Some(p2) = self.board.get_piece(bx2, by2)
                            && p2.color() == PlayerColor::White
                        {
                            let pt2 = p2.piece_type();
                            if (is_ortho && is_ortho_slider(pt2))
                                || (!is_ortho && is_diag_slider(pt2))
                            {
                                self.discovered_check_squares_white.insert((bx, by));
                            }
                        }
                    } else {
                        // Friendly piece - could be pinned?
                        if let Some((bx2, by2)) = self.find_first_blocker_on_ray(bx, by, *dx, *dy)
                            && let Some(p2) = self.board.get_piece(bx2, by2)
                            && p2.color() == PlayerColor::White
                        {
                            let pt2 = p2.piece_type();
                            if (is_ortho && is_ortho_slider(pt2))
                                || (!is_ortho && is_diag_slider(pt2))
                            {
                                self.pinned_black.insert((bx, by), (*dx, *dy));
                            }
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
            if let Some(col_vec) = self.spatial_indices.cols.get(&start_x)
                && let Some((found_y, _packed)) = SpatialIndices::find_nearest(col_vec, start_y, dy)
            {
                return Some((start_x, found_y));
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

    /// Compute blocking squares for non-linear checkers (Rose).
    ///
    /// Algorithm:
    /// 1. Find ALL spiral paths that can reach the target offset
    /// 2. For each path, check if it's currently blocked by any piece
    /// 3. If blocked, skip that path
    /// 4. Collect ALL intermediate squares from unblocked paths
    /// 5. If 1 path: return all intermediates (any block works)
    /// 6. If N paths: return intersection (must block all)
    fn get_nonlinear_blocking_squares(
        &self,
        checker_sq: &Coordinate,
        king_sq: &Coordinate,
        checker_type: PieceType,
    ) -> arrayvec::ArrayVec<Coordinate, 64> {
        use crate::moves::ROSE_SPIRALS;

        let mut result = arrayvec::ArrayVec::<Coordinate, 64>::new();

        if checker_type != PieceType::Rose {
            return result;
        }

        let dx = king_sq.x - checker_sq.x;
        let dy = king_sq.y - checker_sq.y;

        // Collect all unblocked paths with their intermediate squares
        // Each path is a list of (absolute) intermediate coordinates
        let mut valid_paths: arrayvec::ArrayVec<arrayvec::ArrayVec<Coordinate, 8>, 16> =
            arrayvec::ArrayVec::new();

        // Search all 16 spirals (8 start directions × 2 rotations)
        for spiral_dirs in &ROSE_SPIRALS {
            for spiral in spiral_dirs {
                // Find which hop reaches target offset (if any)
                let mut target_hop: Option<usize> = None;
                for (hop_idx, &(cum_dx, cum_dy)) in spiral.iter().enumerate() {
                    if cum_dx == dx && cum_dy == dy {
                        target_hop = Some(hop_idx);
                        break;
                    }
                }

                let target_hop = match target_hop {
                    Some(h) => h,
                    None => continue,
                };

                // Collect intermediate squares and check if path is blocked
                let mut path_blocked = false;
                let mut intermediates = arrayvec::ArrayVec::<Coordinate, 8>::new();

                for &(int_dx, int_dy) in spiral.iter().take(target_hop) {
                    let sq = Coordinate::new(checker_sq.x + int_dx, checker_sq.y + int_dy);

                    // Check if this intermediate square is occupied
                    if self.board.get_piece(sq.x, sq.y).is_some() {
                        path_blocked = true;
                        break;
                    }

                    intermediates.push(sq);
                }

                if !path_blocked {
                    valid_paths.push(intermediates);
                }
            }
        }

        // No unblocked paths - shouldn't happen if we're in check
        if valid_paths.is_empty() {
            return result;
        }

        // Single path: return ALL its intermediate squares
        if valid_paths.len() == 1 {
            for sq in &valid_paths[0] {
                result.push(*sq);
            }
            return result;
        }

        // Multiple paths: return INTERSECTION of all paths
        // Only squares common to all paths can block all attacks
        if let Some((first, rest)) = valid_paths.split_first() {
            for sq in first {
                let on_all_paths = rest
                    .iter()
                    .all(|path| path.iter().any(|p| p.x == sq.x && p.y == sq.y));

                if on_all_paths && result.len() < 64 {
                    result.push(*sq);
                }
            }
        }

        result
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
    /// The logic: The OPPONENT's win condition determines if WE must escape check.
    /// - white_win_condition = how White beats Black (what White must do to win)
    /// - black_win_condition = how Black beats White (what Black must do to win)
    /// - If White is to move: Black beats White via black_win_condition → if Checkmate, White must escape
    /// - If Black is to move: White beats Black via white_win_condition → if Checkmate, Black must escape
    #[inline]
    pub fn must_escape_check(&self) -> bool {
        // The OPPONENT's win condition tells us how they beat us
        // If they beat us via checkmate, we must escape check
        let opponent_win_condition = match self.turn {
            PlayerColor::White => self.game_rules.black_win_condition, // How Black beats White
            PlayerColor::Black => self.game_rules.white_win_condition, // How White beats Black
            PlayerColor::Neutral => return true,                       // Safe default
        };
        opponent_win_condition.requires_check_evasion()
    }

    /// Returns true if the given color's king can be captured (no check evasion needed).
    /// This is the opposite of must_escape_check but for a specific color.
    ///
    /// The OPPONENT's win condition against this color determines if the king can be captured:
    /// - If White's king can be captured: check black_win_condition (how Black beats White)
    /// - If Black's king can be captured: check white_win_condition (how White beats Black)
    #[inline]
    pub fn king_capturable(&self, color: PlayerColor) -> bool {
        // The OPPONENT's win condition tells us how they beat this color
        // If they beat via capture (not checkmate), the king can be captured
        let opponent_win_condition = match color {
            PlayerColor::White => self.game_rules.black_win_condition, // How Black beats White
            PlayerColor::Black => self.game_rules.white_win_condition, // How White beats Black
            PlayerColor::Neutral => return false,
        };
        !opponent_win_condition.requires_check_evasion()
    }

    /// Check if the side-to-move has lost by royal capture.
    /// This is only relevant for RoyalCapture and AllRoyalsCaptured win conditions.
    /// Returns true if the opponent (who just moved) has captured all required royals.
    ///
    /// The OPPONENT's win condition against us determines if we can lose by royal capture:
    /// - If White is to move: check black_win_condition (how Black beats White)
    /// - If Black is to move: check white_win_condition (how White beats Black)
    #[inline]
    pub fn has_lost_by_royal_capture(&self) -> bool {
        // The OPPONENT's win condition tells us how they beat us
        let opponent_win_condition = match self.turn {
            PlayerColor::White => self.game_rules.black_win_condition, // How Black beats White
            PlayerColor::Black => self.game_rules.white_win_condition, // How White beats Black
            PlayerColor::Neutral => return false,
        };

        // Only check for royal loss if opponent's win condition is royal-capture based
        if !opponent_win_condition.is_royal_capture_based() {
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

    /// Repetition detection for search.
    /// Returns true if the current position should be treated as a draw due to repetition.
    ///
    /// Logic for draw detection: `repetition != 0 && repetition < ply`
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
        // Result is true if a repetition occurred within the current search tree.
        // This works for both positive (twofold) and negative (threefold) values.
        // Negative values are always < positive ply, so threefold always returns true for ply > 0.
        self.repetition != 0 && self.repetition < (ply as i32)
    }

    /// Check if we have an upcoming move that draws by repetition.
    /// Checks if a single reversible move would lead to a repeated position.
    /// Used for pruning and early draw detection.
    #[inline]
    pub fn upcoming_repetition(&self, ply: usize) -> bool {
        use crate::search::zobrist::{SIDE_KEY, piece_key};

        // Don't check during null move search
        if self.null_moves > 0 {
            return false;
        }

        // Use minimum of halfmove_clock and plies_from_null like Stockfish
        let end = (self.halfmove_clock as usize).min(self.plies_from_null as usize);
        if end < 3 {
            return false;
        }

        // Check if positions differ by exactly two moves that allow reversal.
        // We look for recent moves that can be reversed to create a repetition.
        let current_hash = self.hash;
        let stack_len = self.hash_stack.len();
        let history_len = self.move_history.len();

        if history_len < 2 || stack_len < 3 {
            return false;
        }

        // For each odd distance (opposite side to move), check if we can make a move
        // that transforms current position to match an earlier position.
        for i in (3..=end.min(stack_len)).step_by(2) {
            let hist_idx = if history_len >= i {
                history_len - i
            } else {
                continue;
            };
            let hash_idx = if stack_len >= i {
                stack_len - i
            } else {
                continue;
            };
            let target_hash = self.hash_stack[hash_idx];

            // Compute what single move would transform current position to target position
            // move_key = current_hash XOR target_hash XOR SIDE_KEY (since opposite side to move)
            let move_key = current_hash ^ target_hash ^ SIDE_KEY;

            // Check if any of our pieces can make a move that produces this hash difference.
            // For a piece at (from_x, from_y) moving to (to_x, to_y):
            // hash_diff = piece_key(type, color, from) XOR piece_key(type, color, to)
            //
            // The most common case: the piece that moved i plies ago can move back.
            // Check our move from (i-1)/2 moves ago (rounded).
            if hist_idx < history_len {
                let entry = &self.move_history[hist_idx];
                let pt = entry.piece_type;

                // Check if the piece is still at entry.to and can move back to entry.from
                if let Some(piece) = self.board.get_piece(entry.to_x, entry.to_y)
                    && piece.piece_type() == pt
                    && piece.color() == self.turn
                {
                    // Check if entry.from is empty (piece can move back)
                    if self.board.get_piece(entry.from_x, entry.from_y).is_none() {
                        // Compute hash difference for this reverse move
                        let from_key = piece_key(pt, self.turn, entry.to_x, entry.to_y);
                        let to_key = piece_key(pt, self.turn, entry.from_x, entry.from_y);
                        let expected_key = from_key ^ to_key;

                        if expected_key == move_key {
                            // This move would create the repetition!
                            // Check if repetition is within search tree
                            if ply > i {
                                return true;
                            }

                            // For root nodes, check if target position was already repeated
                            // This requires checking if hash_stack[hash_idx] had repetition
                            // For simplicity, just check the most recent few positions
                            for j in (0..hash_idx).rev().step_by(2).take(4) {
                                if self.hash_stack[j] == target_hash {
                                    return true;
                                }
                            }
                        }
                    }
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

        // Reset plies from last null move.
        self.plies_from_null = 0;
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
            if let Some(piece) = self.board.get_piece(coord.x, coord.y)
                && piece.piece_type() == PieceType::Pawn
            {
                h ^= pawn_special_right_key(coord.x, coord.y);
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

        get_legal_moves_into(&self.board, self.turn, &ctx, out);

        if out.is_empty() {
            get_legal_moves_into(&self.board, self.turn, &ctx, out);
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
        get_pseudo_legal_moves_for_piece_into(&self.board, &king_piece, &king_sq, &ctx, out);

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
            arrayvec::ArrayVec::<Coordinate, 64>::new()
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
                        if let Some(p) = self.board.get_piece(bx, by)
                            && p.color() == our_color
                        {
                            continue;
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
                            if is_prime_fast(dist_from_checker)
                                && s.is_path_clear_for_rook(&from, &Coordinate::new(tx, ty))
                            {
                                out.push(Move::new(from, Coordinate::new(tx, ty), *piece));
                            }
                        }
                    }
                }
            }

            // ==========================================
            // OPTIMIZED SLIDER BLOCKING FOR ROSE CHECKERS
            // For Rose checks, blocking squares are precomputed.
            // Generate moves to each reachable blocking square.
            // ==========================================
            if is_nonlinear_checker && (can_ortho || can_diag) {
                for block_sq in &nonlinear_blocking_squares {
                    // Check if this slider can reach the blocking square
                    if can_ortho {
                        // Check same row or column
                        if from.x == block_sq.x && from.y != block_sq.y {
                            // Same column - check path clear
                            if s.is_path_clear_for_rook(&from, block_sq) {
                                out.push(Move::new(from, *block_sq, *piece));
                            }
                        } else if from.y == block_sq.y && from.x != block_sq.x {
                            // Same row - check path clear
                            if s.is_path_clear_for_rook(&from, block_sq) {
                                out.push(Move::new(from, *block_sq, *piece));
                            }
                        }
                    }
                    if can_diag {
                        // Check same diagonal
                        let dx = block_sq.x - from.x;
                        let dy = block_sq.y - from.y;
                        if dx != 0
                            && dx.abs() == dy.abs()
                            && s.is_path_clear_for_bishop(&from, block_sq)
                        {
                            out.push(Move::new(from, *block_sq, *piece));
                        }
                    }
                }
            }

            // Regular slider blocking for non-Huygen, non-Rose checkers (linear attack patterns)
            if is_slider && can_ortho && !is_huygen_checker && !is_nonlinear_checker {
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
            get_pseudo_legal_moves_for_piece_into(&s.board, piece, &from, &ctx, &mut pseudo);

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
                if m.piece.piece_type() == PieceType::Pawn
                    && let Some(ep) = &s.en_passant
                    && m.to.x == ep.square.x
                    && m.to.y == ep.square.y
                {
                    // This is an EP capture - check if it captures the checker
                    if ep.pawn_square.x == checker_sq.x && ep.pawn_square.y == checker_sq.y {
                        out.push(m);
                        continue;
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

    /// Ultra-fast legality check (C1 Optimization).
    /// Only does simple arithmetic - NO spatial index lookups for the common case.
    /// Returns:
    /// - Ok(true): Move is DEFINITELY LEGAL (piece not on any slider ray from king AND side not in check)
    /// - Err(()): Cannot determine fast, must use full is_move_illegal check
    #[inline(always)]
    #[allow(clippy::result_unit_err)]
    pub fn is_legal_fast(&self, m: &Move, in_check: bool) -> Result<bool, ()> {
        // 1. If currently in check, any move could be illegal or fail to escape check.
        if in_check {
            return Err(());
        }

        // 2. King moves: always need full check (must check for attacked squares)
        if m.piece.piece_type().is_royal() {
            return Err(());
        }

        // 3. En Passant: always need full check (rank clearing can expose king behind)
        if m.piece.piece_type() == PieceType::Pawn {
            let dx = (m.to.x - m.from.x).abs();
            let dy = (m.to.y - m.from.y).abs();
            if dx != 0 && dy != 0 && !self.board.is_occupied(m.to.x, m.to.y) {
                return Err(());
            }
        }

        // 4. Get king position (fast - already cached)
        let king_pos = if self.turn == PlayerColor::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };

        let Some(king) = king_pos else {
            // No king - can't be pinned
            return Ok(true);
        };

        // 5. FAST CHECK: Is piece on a slider ray from king?
        // Only arithmetic - no hash lookups!
        let dx = m.from.x - king.x;
        let dy = m.from.y - king.y;

        // Same square as king (shouldn't happen for non-king piece)
        if dx == 0 && dy == 0 {
            return Err(());
        }

        // Check if on a slider ray (vertical, horizontal, or diagonal)
        let on_slider_ray = dx == 0  // Vertical (same file)
            || dy == 0               // Horizontal (same rank)  
            || dx.abs() == dy.abs(); // Diagonal

        if on_slider_ray {
            // Piece MIGHT be pinned - fall back to full is_move_illegal check
            Err(())
        } else {
            // Piece is NOT on any slider ray from king - CANNOT be pinned!
            // Side is NOT in check (verified above), and it's NOT a king move/EP.
            // Therefore, this move is DEFINITELY LEGAL. Skip is_move_illegal entirely.
            Ok(true)
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
                } && m.from.y == k_pos.y
                {
                    let idx = if piece.color() == PlayerColor::White {
                        if m.from.x > k_pos.x { 0 } else { 1 }
                    } else if m.from.x > k_pos.x {
                        2
                    } else {
                        3
                    };
                    self.castling_partner_counts[idx] =
                        self.castling_partner_counts[idx].saturating_sub(1);
                    if self.castling_partner_counts[idx] == 0 {
                        self.effective_castling_rights &= !(1 << idx);
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
                } && m.to.y == k_pos.y
                {
                    let idx = if captured.color() == PlayerColor::White {
                        if m.to.x > k_pos.x { 0 } else { 1 }
                    } else if m.to.x > k_pos.x {
                        2
                    } else {
                        3
                    };
                    self.castling_partner_counts[idx] =
                        self.castling_partner_counts[idx].saturating_sub(1);
                    if self.castling_partner_counts[idx] == 0 {
                        self.effective_castling_rights &= !(1 << idx);
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
                } && rook_coord.y == k_pos.y
                {
                    let idx = if rook.color() == PlayerColor::White {
                        if rook_coord.x > k_pos.x { 0 } else { 1 }
                    } else if rook_coord.x > k_pos.x {
                        2
                    } else {
                        3
                    };
                    self.castling_partner_counts[idx] =
                        self.castling_partner_counts[idx].saturating_sub(1);
                    if self.castling_partner_counts[idx] == 0 {
                        self.effective_castling_rights &= !(1 << idx);
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

        // Track move for repetition detection.
        self.move_history.push(MoveHistoryEntry {
            from_x: m.from.x,
            from_y: m.from.y,
            to_x: m.to.x,
            to_y: m.to.y,
            piece_type: piece.piece_type(),
        });
        self.plies_from_null += 1;

        // Compute distance to previous occurrence for repetition detection:
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
                self.white_pawn_count = self.white_pawn_count.saturating_add(1);
            } else {
                self.material_score += promo_val;
                self.material_score -= pawn_val;
                self.black_pawn_count = self.black_pawn_count.saturating_add(1);
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
        if let Some(captured_pawn) = undo.ep_captured_piece
            && let Some(ep) = &undo.old_en_passant
        {
            // Restore the captured piece (could be a pawn or promoted piece)
            self.board
                .set_piece(ep.pawn_square.x, ep.pawn_square.y, captured_pawn);
            self.spatial_indices
                .add(ep.pawn_square.x, ep.pawn_square.y, captured_pawn.packed());

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

        // Pop move history (reverse of make_move push)
        self.move_history.pop();
        self.plies_from_null = self.plies_from_null.saturating_sub(1);
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

    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub fn setup_position_from_icn(&mut self, position_icn: &str) {
        self.board = Board::new();
        self.special_rights.clear();
        self.en_passant = None;
        self.turn = PlayerColor::White;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;
        self.material_score = 0;

        // Step 1: Strip [...] metadata tags
        let content = if let Some(idx) = position_icn.rfind(']') {
            &position_icn[idx + 1..]
        } else {
            position_icn
        };

        let content = content.trim();
        if content.is_empty() {
            return;
        }

        // Tokenize by whitespace
        let tokens: Vec<&str> = content.split_whitespace().collect();

        // Handle the case where it's JUST pieces
        if tokens.len() == 1 {
            self.parse_icn_pieces(tokens[0]);
            self.finalize_setup();
            return;
        }

        let mut wc_list = Vec::new();
        let mut pieces_token = None;

        for token in tokens {
            if token == "-" {
                continue;
            }

            // identify token by structure
            if token == "w" {
                self.turn = PlayerColor::White;
            } else if token == "b" {
                self.turn = PlayerColor::Black;
            } else if token.contains('/') {
                // Clocks: halfmove/limit
                let parts: Vec<&str> = token.split('/').collect();
                if let Some(hm_str) = parts.first() {
                    self.halfmove_clock = hm_str.parse().unwrap_or(0);
                }
                if parts.len() > 1 {
                    self.game_rules.move_rule_limit = parts[1].parse::<u32>().ok();
                }
            } else if token.starts_with('(') && token.ends_with(')') {
                // Promotion Rules: (w_rank;w_pieces|b_rank;b_pieces)
                let inner = &token[1..token.len() - 1];
                let sides: Vec<&str> = inner.split('|').collect();
                let mut promo_types = Vec::new();

                for (idx, side_str) in sides.iter().enumerate() {
                    let parts: Vec<&str> = side_str.split(';').collect();
                    if parts.is_empty() {
                        continue;
                    }

                    if let Ok(rank) = parts[0].parse::<i64>() {
                        if idx == 0 {
                            self.white_promo_rank = rank;
                        } else {
                            self.black_promo_rank = rank;
                        }

                        if self.game_rules.promotion_ranks.is_none() {
                            self.game_rules.promotion_ranks = Some(PromotionRanks::default());
                        }
                        if let Some(ref mut pr) = self.game_rules.promotion_ranks {
                            if idx == 0 {
                                pr.white = vec![rank];
                            } else {
                                pr.black = vec![rank];
                            }
                        }
                    }

                    if parts.len() > 1 {
                        let types: Vec<&str> = parts[1].split(',').collect();
                        for t in &types {
                            let pt = PieceType::from_site_code(&t.to_uppercase());
                            if pt != PieceType::Void {
                                promo_types.push(pt);
                            }
                        }
                        self.game_rules.promotions_allowed =
                            Some(types.iter().map(|s| s.to_string()).collect());
                        self.game_rules.promotion_types = Some(promo_types.clone());
                    }
                }
            } else if token.split(',').count() == 4
                && token
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == ',' || c == '-')
            {
                // World Border: left,right,bottom,top
                let bounds: Vec<&str> = token.split(',').collect();
                if let (Ok(l), Ok(r), Ok(b), Ok(t)) = (
                    bounds[0].parse::<i64>(),
                    bounds[1].parse::<i64>(),
                    bounds[2].parse::<i64>(),
                    bounds[3].parse::<i64>(),
                ) {
                    crate::moves::set_world_bounds(l, r, b, t);
                }
            } else if token.contains('|')
                || (token.contains(',') && token.chars().any(|c| c.is_ascii_uppercase()))
            {
                // Pieces segment typically contains '|' or pieces like 'P1,2' (uppercase P)
                pieces_token = Some(token);
            } else if token.contains(',')
                && token.split(',').count() == 2
                && !token.chars().any(|c| c.is_ascii_alphabetic())
            {
                // En Passant: x,y
                let parts: Vec<&str> = token.split(',').collect();
                if let (Ok(x), Ok(y)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                    let pawn_y = if self.turn == PlayerColor::White {
                        y - 1
                    } else {
                        y + 1
                    };
                    self.en_passant = Some(EnPassantState {
                        square: Coordinate::new(x, y),
                        pawn_square: Coordinate::new(x, pawn_y),
                    });
                }
            } else if let Ok(val) = token.parse::<u32>() {
                // Fullmove number
                self.fullmove_number = val;
            } else {
                // Check if it's a win condition list
                for wc_str in token.split(',') {
                    if let Ok(wc) = wc_str.parse::<WinCondition>() {
                        wc_list.push(wc);
                    }
                }
            }
        }

        // Now parse pieces if found
        if let Some(p) = pieces_token {
            self.parse_icn_pieces(p);
        }

        // Recompute piece counts/lists BEFORE selecting win conditions
        self.recompute_piece_counts();

        // Finalize win conditions based on piece presence
        let white_has_royal = self.white_pieces.iter().any(|&(px, py)| {
            self.board
                .get_piece(px, py)
                .map(|p| p.piece_type().is_royal())
                .unwrap_or(false)
        });
        let black_has_royal = self.black_pieces.iter().any(|&(px, py)| {
            self.board
                .get_piece(px, py)
                .map(|p| p.piece_type().is_royal())
                .unwrap_or(false)
        });

        self.game_rules.white_win_condition = WinCondition::select(&wc_list, black_has_royal);
        self.game_rules.black_win_condition = WinCondition::select(&wc_list, white_has_royal);

        self.finalize_setup();
    }

    #[cfg(any(test, not(target_arch = "wasm32")))]
    fn parse_icn_pieces(&mut self, piece_segment: &str) {
        for piece_def in piece_segment.split('|') {
            if piece_def.is_empty() {
                continue;
            }

            let parts: Vec<&str> = piece_def.split(',').collect();
            if parts.len() != 2 {
                continue;
            }

            let piece_info_raw = parts[0];
            let y_raw = parts[1];

            let split_idx = piece_info_raw
                .find(|c: char| c.is_ascii_digit() || c == '-')
                .unwrap_or(piece_info_raw.len());
            let (code_str, x_str_raw) = piece_info_raw.split_at(split_idx);

            if code_str.is_empty() {
                continue;
            }

            let mut has_special_rights = false;
            let x_str = if let Some(stripped) = x_str_raw.strip_suffix('+') {
                has_special_rights = true;
                stripped
            } else {
                x_str_raw
            };

            let y_str = if let Some(stripped) = y_raw.strip_suffix('+') {
                has_special_rights = true;
                stripped
            } else {
                y_raw
            };

            let x: i64 = x_str.parse().unwrap_or(0);
            let y: i64 = y_str.parse().unwrap_or(0);

            let first_char = code_str.chars().next().unwrap();
            let is_white = first_char.is_uppercase();
            let color = if is_white {
                PlayerColor::White
            } else {
                PlayerColor::Black
            };

            let piece_type = PieceType::from_site_code(&code_str.to_uppercase());

            self.board.set_piece(x, y, Piece::new(piece_type, color));

            if has_special_rights {
                self.special_rights.insert(Coordinate::new(x, y));
            }
        }
    }

    #[cfg(any(test, not(target_arch = "wasm32")))]
    fn finalize_setup(&mut self) {
        // Calculate initial material
        self.material_score = calculate_initial_material(&self.board);

        // Rebuild piece lists and counts
        self.recompute_piece_counts();

        // Compute initial hash
        self.recompute_hash();

        // Rebuild spatial indices
        self.spatial_indices = SpatialIndices::new(&self.board);
    }

    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub fn setup_variant(&mut self, variant: Variant) {
        self.setup_position_from_icn(variant.starting_icn());
    }

    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub fn setup_standard_chess(&mut self) {
        self.setup_variant(Variant::Classical);
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

    #[test]
    fn test_parse_icn_full() {
        let icn = "[Event \"Complex Game\"] w 10,3 5/100 1 (8;am,q|1;am,q) -100,500,-35,100 checkmate,royalcapture,allroyalscaptured,allpiecescaptured K5,1+|k5,8+";

        let mut game = GameState::new();
        game.setup_position_from_icn(icn);

        // Check header info
        assert_eq!(game.turn, PlayerColor::White);
        assert_eq!(game.halfmove_clock, 5);
        assert_eq!(game.game_rules.move_rule_limit, Some(100));
        assert_eq!(game.fullmove_number, 1);
        assert_eq!(game.white_promo_rank, 8);
        assert_eq!(game.black_promo_rank, 1);

        // En passant square (10,3). White turn, so Black pawn just moved 10,4->10,2.
        // Pawn being captured is at 10,2.
        let ep = game.en_passant.unwrap();
        assert_eq!(ep.square, Coordinate::new(10, 3));
        assert_eq!(ep.pawn_square, Coordinate::new(10, 2));

        // Check world bounds
        let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
        assert_eq!(min_x, -100);
        assert_eq!(max_x, 500);
        assert_eq!(min_y, -35);
        assert_eq!(max_y, 100);

        // Check win conditions
        // Priority: Checkmate
        assert_eq!(game.game_rules.white_win_condition, WinCondition::Checkmate);
        assert_eq!(game.game_rules.black_win_condition, WinCondition::Checkmate);

        // Check allowed promotions
        let allowed = game.game_rules.promotions_allowed.as_ref().unwrap();
        assert!(allowed.contains(&"am".to_string()));
        assert!(allowed.contains(&"q".to_string()));

        // Check pieces
        let k = game.board.get_piece(5, 1).unwrap();
        assert_eq!(k.piece_type(), PieceType::King);
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
    fn test_all_variants_setup() {
        let variants = [
            Variant::Classical,
            Variant::ConfinedClassical,
            Variant::ClassicalPlus,
            Variant::CoaIP,
            Variant::CoaIPHO,
            Variant::CoaIPRO,
            Variant::CoaIPNO,
            Variant::Palace,
            Variant::Pawndard,
            Variant::Core,
            Variant::Standarch,
            Variant::SpaceClassic,
            Variant::Space,
            Variant::Abundance,
            Variant::PawnHorde,
            Variant::Knightline,
            Variant::Obstocean,
            Variant::Chess,
        ];

        for &v in &variants {
            let mut game = GameState::new();
            game.setup_variant(v);
            assert!(
                game.white_piece_count > 0,
                "Variant {:?} should have white pieces",
                v
            );
            assert!(
                game.black_piece_count > 0,
                "Variant {:?} should have black pieces",
                v
            );
        }
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
