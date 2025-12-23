use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::{EnPassantState, GameRules};
use crate::utils::is_prime_i64;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

/// Stack-allocated move list with inline capacity of 128 moves.
/// Spills to heap if this limit is exceeded, preventing panics.
pub type MoveList = Vec<Move>;

// World border for infinite chess. These are initialized to a very large box,
// but can be overridden from JS via the playableRegion values.
static mut COORD_MIN_X: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_X: i64 = 1_000_000_000_000_000; // default  1e15
static mut COORD_MIN_Y: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_Y: i64 = 1_000_000_000_000_000; // default  1e15

/// Update world borders from JS playableRegion (left, right, bottom, top).
pub fn set_world_bounds(left: i64, right: i64, bottom: i64, top: i64) {
    unsafe {
        COORD_MIN_X = left.min(right);
        COORD_MAX_X = left.max(right);
        COORD_MIN_Y = bottom.min(top);
        COORD_MAX_Y = bottom.max(top);
    }
}

/// Get the maximum dimension of the current world border.
/// Returns the larger of (max_x - min_x, max_y - min_y).
/// Used for determining if standard chess mating patterns apply (bounded board).
#[inline]
pub fn get_world_size() -> i64 {
    unsafe {
        let width = COORD_MAX_X - COORD_MIN_X;
        let height = COORD_MAX_Y - COORD_MIN_Y;
        width.max(height)
    }
}

/// Get all coordinate bounds (min_x, max_x, min_y, max_y).
/// Used for cage detection in mop-up evaluation.
#[inline]
pub fn get_coord_bounds() -> (i64, i64, i64, i64) {
    unsafe { (COORD_MIN_X, COORD_MAX_X, COORD_MIN_Y, COORD_MAX_Y) }
}

/// Generate all pseudo-legal moves for a Knightrider.
/// A Knightrider slides like a knight repeated along its direction until blocked or out of bounds.
fn generate_knightrider_moves(board: &Board, from: &Coordinate, piece: &Piece) -> MoveList {
    // All 8 knight directions
    const KR_DIRS: [(i64, i64); 8] = [
        (1, 2),
        (1, -2),
        (2, 1),
        (2, -1),
        (-1, 2),
        (-1, -2),
        (-2, 1),
        (-2, -1),
    ];

    let piece_count = board.len();
    let mut moves = MoveList::new();

    // Pre-collect piece data once
    let mut pieces_data: Vec<(i64, i64, bool)> = Vec::with_capacity(piece_count);
    // BITBOARD: Use tile-based CTZ iteration for O(popcount) piece enumeration
    for (cx, cy, tile) in board.tiles.iter() {
        let mut bits = tile.occ_all;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }
            let p = Piece::from_packed(packed);
            let lx = (idx % 8) as i64;
            let ly = (idx / 8) as i64;
            let px = cx * 8 + lx;
            let py = cy * 8 + ly;
            let is_enemy = is_enemy_piece(&p, piece.color());
            pieces_data.push((px, py, is_enemy));
        }
    }

    for (dx, dy) in KR_DIRS {
        // 1. Find closest blocker along this knight ray, in units of knight-steps (k)
        let mut closest_k: i64 = i64::MAX;
        let mut closest_is_enemy = false;

        for &(px, py, is_enemy) in &pieces_data {
            let rx = px - from.x;
            let ry = py - from.y;

            // Solve (rx, ry) = k * (dx, dy) with integer k > 0
            if rx == 0 && ry == 0 {
                continue;
            }

            // dx,dy are non-zero for all knight directions
            if rx % dx != 0 || ry % dy != 0 {
                continue;
            }

            let kx = rx / dx;
            let ky = ry / dy;
            if kx <= 0 || ky <= 0 || kx != ky {
                continue;
            }

            let k = kx; // steps along this knight ray
            if k < closest_k {
                closest_k = k;
                closest_is_enemy = is_enemy;
            }
        }

        // 2. Generate moves along this ray.
        // Cap at 10 for performance - captures at distance handled separately
        const KR_STEP_LIMIT: i64 = 10;
        let max_steps: i64 = if closest_k < i64::MAX {
            if closest_is_enemy {
                closest_k.min(KR_STEP_LIMIT)
            } else {
                closest_k.saturating_sub(1).min(KR_STEP_LIMIT)
            }
        } else {
            2
        };

        // CRITICAL: If enemy is beyond step limit, still add the direct capture
        if closest_k < i64::MAX && closest_is_enemy && closest_k > KR_STEP_LIMIT {
            let x = from.x + dx * closest_k;
            let y = from.y + dy * closest_k;
            if in_bounds(x, y) {
                moves.push(Move::new(*from, Coordinate::new(x, y), *piece));
            }
        }

        if max_steps <= 0 {
            continue;
        }

        let mut k = 1i64;
        while k <= max_steps {
            let x = from.x + dx * k;
            let y = from.y + dy * k;

            if !in_bounds(x, y) {
                break;
            }

            if let Some(blocker) = board.get_piece(x, y) {
                // Enemy: can capture on this square.
                if blocker.color() != piece.color() && blocker.piece_type() != PieceType::Void {
                    moves.push(Move::new(*from, Coordinate::new(x, y), *piece));
                }
                // Either way, ray stops at first blocker.
                break;
            } else {
                // Empty square: normal quiet move (only within the window).
                moves.push(Move::new(*from, Coordinate::new(x, y), *piece));
            }

            k += 1;
        }
    }

    moves
}

/// Check if a coordinate is within valid bounds (world border)
#[inline]
pub fn in_bounds(x: i64, y: i64) -> bool {
    unsafe { x >= COORD_MIN_X && x <= COORD_MAX_X && y >= COORD_MIN_Y && y <= COORD_MAX_Y }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SpatialIndices {
    /// Row index: y -> [(x, packed_piece), ...] sorted by x
    pub rows: FxHashMap<i64, Vec<(i64, u8)>>,
    /// Column index: x -> [(y, packed_piece), ...] sorted by y
    pub cols: FxHashMap<i64, Vec<(i64, u8)>>,
    /// Diagonal (x-y constant): key -> [(x, packed_piece), ...] sorted by x
    pub diag1: FxHashMap<i64, Vec<(i64, u8)>>,
    /// Anti-diagonal (x+y constant): key -> [(x, packed_piece), ...] sorted by x
    pub diag2: FxHashMap<i64, Vec<(i64, u8)>>,
    /// Lazily-populated slider interception cache.
    /// Key: (x, y, dir_index) where dir_index encodes the 8 cardinal/diagonal directions.
    /// Value: Sorted list of valid interception distances for that slider position/direction.
    #[serde(skip)]
    pub slider_cache: std::cell::RefCell<FxHashMap<(i64, i64, u8), Vec<i64>>>,

    // Fairy piece existence flags per color for O(1) early-exit in attack detection
    // [0] = white, [1] = black
    #[serde(skip)]
    pub has_huygen: [bool; 2],
    #[serde(skip)]
    pub has_rose: [bool; 2],
    #[serde(skip)]
    pub has_knightrider: [bool; 2],
}

impl SpatialIndices {
    pub fn new(board: &Board) -> Self {
        let mut rows: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut cols: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut diag1: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut diag2: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();

        // Fairy piece flags: [0] = white, [1] = black
        let mut has_huygen = [false, false];
        let mut has_rose = [false, false];
        let mut has_knightrider = [false, false];

        // BITBOARD: Use tile-based CTZ iteration for O(popcount) enumeration
        for (cx, cy, tile) in board.tiles.iter() {
            let mut bits = tile.occ_all;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let packed = tile.piece[idx];
                // Note: packed==0 is valid for Void pieces (Neutral*22+Void=0)
                // occ_all bitboard guarantees this is an occupied square
                let lx = (idx % 8) as i64;
                let ly = (idx / 8) as i64;
                let x = cx * 8 + lx;
                let y = cy * 8 + ly;
                rows.entry(y).or_default().push((x, packed));
                cols.entry(x).or_default().push((y, packed));
                diag1.entry(x - y).or_default().push((x, packed));
                diag2.entry(x + y).or_default().push((x, packed));

                // Track fairy piece existence for O(1) early-exit in attack detection
                let piece = Piece::from_packed(packed);
                let color_idx = if piece.color() == PlayerColor::White {
                    0
                } else {
                    1
                };
                match piece.piece_type() {
                    PieceType::Huygen => has_huygen[color_idx] = true,
                    PieceType::Rose => has_rose[color_idx] = true,
                    PieceType::Knightrider => has_knightrider[color_idx] = true,
                    _ => {}
                }
            }
        }

        // Sort vectors by coordinate for binary search
        for list in rows.values_mut() {
            list.sort_by_key(|(coord, _)| *coord);
        }
        for list in cols.values_mut() {
            list.sort_by_key(|(coord, _)| *coord);
        }
        for list in diag1.values_mut() {
            list.sort_by_key(|(coord, _)| *coord);
        }
        for list in diag2.values_mut() {
            list.sort_by_key(|(coord, _)| *coord);
        }

        SpatialIndices {
            rows,
            cols,
            diag1,
            diag2,
            slider_cache: std::cell::RefCell::new(FxHashMap::default()),
            has_huygen,
            has_rose,
            has_knightrider,
        }
    }

    #[inline]
    fn insert_sorted(vec: &mut Vec<(i64, u8)>, coord: i64, packed: u8) {
        match vec.binary_search_by_key(&coord, |(c, _)| *c) {
            Ok(pos) => {
                // Update existing entry with new piece
                vec[pos].1 = packed;
            }
            Err(pos) => vec.insert(pos, (coord, packed)),
        }
    }

    #[inline]
    fn remove_sorted(vec: &mut Vec<(i64, u8)>, coord: i64) {
        if let Ok(pos) = vec.binary_search_by_key(&coord, |(c, _)| *c) {
            vec.remove(pos);
        }
    }

    /// Incrementally add a piece at (x, y) to the indices.
    pub fn add(&mut self, x: i64, y: i64, packed: u8) {
        Self::insert_sorted(self.rows.entry(y).or_default(), x, packed);
        Self::insert_sorted(self.cols.entry(x).or_default(), y, packed);

        let d1 = x - y;
        let d2 = x + y;
        Self::insert_sorted(self.diag1.entry(d1).or_default(), x, packed);
        Self::insert_sorted(self.diag2.entry(d2).or_default(), x, packed);
    }

    /// Incrementally remove a piece at (x, y) from the indices.
    pub fn remove(&mut self, x: i64, y: i64) {
        if let Some(v) = self.rows.get_mut(&y) {
            Self::remove_sorted(v, x);
            if v.is_empty() {
                self.rows.remove(&y);
            }
        }
        if let Some(v) = self.cols.get_mut(&x) {
            Self::remove_sorted(v, y);
            if v.is_empty() {
                self.cols.remove(&x);
            }
        }

        let d1 = x - y;
        if let Some(v) = self.diag1.get_mut(&d1) {
            Self::remove_sorted(v, x);
            if v.is_empty() {
                self.diag1.remove(&d1);
            }
        }
        let d2 = x + y;
        if let Some(v) = self.diag2.get_mut(&d2) {
            Self::remove_sorted(v, x);
            if v.is_empty() {
                self.diag2.remove(&d2);
            }
        }
    }

    /// Helper to find nearest piece in a direction from SpatialIndices.
    /// Returns (coord, packed_piece) if found.
    #[inline]
    pub fn find_nearest(vec: &[(i64, u8)], from: i64, direction: i64) -> Option<(i64, u8)> {
        let pos = vec.binary_search_by_key(&from, |(c, _)| *c);
        let idx = match pos {
            Ok(i) => i,
            Err(i) => i,
        };

        if direction > 0 {
            // Look forward
            let next_idx = if pos.is_ok() { idx + 1 } else { idx };
            if next_idx < vec.len() {
                return Some(vec[next_idx]);
            }
        } else {
            // Look backward
            if idx > 0 {
                return Some(vec[idx - 1]);
            }
        }
        None
    }
}

impl Default for SpatialIndices {
    fn default() -> Self {
        SpatialIndices {
            rows: FxHashMap::default(),
            cols: FxHashMap::default(),
            diag1: FxHashMap::default(),
            diag2: FxHashMap::default(),
            slider_cache: std::cell::RefCell::new(FxHashMap::default()),
            has_huygen: [false, false],
            has_rose: [false, false],
            has_knightrider: [false, false],
        }
    }
}

/// Compact move representation - Copy-able for zero-allocation cloning in hot loops.
/// Uses Option<PieceType> instead of Option<String> for promotion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Move {
    pub from: Coordinate,
    pub to: Coordinate,
    pub piece: Piece,
    pub promotion: Option<PieceType>,
    pub rook_coord: Option<Coordinate>, // For castling: stores the rook's coordinate
}

impl Move {
    pub fn new(from: Coordinate, to: Coordinate, piece: Piece) -> Self {
        Move {
            from,
            to,
            piece,
            promotion: None,
            rook_coord: None,
        }
    }
}

#[inline]
fn is_enemy_piece(piece: &Piece, our_color: PlayerColor) -> bool {
    piece.color() != our_color && piece.piece_type() != PieceType::Void
}

pub fn get_legal_moves_into(
    board: &Board,
    turn: PlayerColor,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    out: &mut MoveList,
    fallback: bool,
    enemy_king_pos: Option<&Coordinate>, // For check target computation
) {
    use crate::tiles::TILE_SIZE;

    out.clear();

    // BITBOARD: Use tile-based CTZ iteration for O(popcount) piece enumeration
    // This is the Stockfish pattern: iterate occupied tiles, CTZ through occupancy bits
    let is_white = turn == PlayerColor::White;

    for (cx, cy, tile) in board.tiles.iter() {
        // Get occupancy bitboard for our color
        let occ = if is_white {
            tile.occ_white
        } else {
            tile.occ_black
        };
        if occ == 0 {
            continue;
        } // Fast skip empty tiles

        // CTZ loop: extract each set bit (piece position)
        let mut bits = occ;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1; // Clear lowest bit

            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }

            let piece = Piece::from_packed(packed);

            // Convert tile-local index to world coordinates
            let lx = (idx % 8) as i64;
            let ly = (idx / 8) as i64;
            let x = cx * TILE_SIZE + lx;
            let y = cy * TILE_SIZE + ly;
            let from = Coordinate::new(x, y);

            get_pseudo_legal_moves_for_piece_into(
                board,
                &piece,
                &from,
                special_rights,
                en_passant,
                indices,
                game_rules,
                fallback,
                out,
                enemy_king_pos,
            );
        }
    }
}

pub fn get_legal_moves(
    board: &Board,
    turn: PlayerColor,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    enemy_king_pos: Option<&Coordinate>,
) -> MoveList {
    let mut moves = MoveList::new();
    get_legal_moves_into(
        board,
        turn,
        special_rights,
        en_passant,
        game_rules,
        indices,
        &mut moves,
        false, // Normal mode
        enemy_king_pos,
    );

    // Fallback: if no pseudo-legal moves found, try short-range slider fallback
    // Note: This logic triggers on 0 *pseudo-legal* moves.
    // If strict generation yields 0 moves, we might be stuck, so we retry.
    if moves.is_empty() {
        get_legal_moves_into(
            board,
            turn,
            special_rights,
            en_passant,
            game_rules,
            indices,
            &mut moves,
            true, // Fallback mode
            enemy_king_pos,
        );
    }

    moves
}

/// Generate only capturing moves for quiescence search when the side to move is **not** in check.
/// This avoids generating and then filtering thousands of quiet moves.
pub fn get_quiescence_captures(
    board: &Board,
    turn: PlayerColor,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    out: &mut MoveList,
) {
    use crate::tiles::TILE_SIZE;

    out.clear();

    // BITBOARD: CTZ iteration for O(popcount) piece enumeration
    let is_white = turn == PlayerColor::White;

    for (cx, cy, tile) in board.tiles.iter() {
        let occ = if is_white {
            tile.occ_white
        } else {
            tile.occ_black
        };
        if occ == 0 {
            continue;
        }

        let mut bits = occ;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }

            let piece = Piece::from_packed(packed);

            let lx = (idx % 8) as i64;
            let ly = (idx / 8) as i64;
            let x = cx * TILE_SIZE + lx;
            let y = cy * TILE_SIZE + ly;
            let from = Coordinate::new(x, y);

            generate_captures_for_piece(
                board,
                &piece,
                &from,
                special_rights,
                en_passant,
                game_rules,
                indices,
                out,
            );
        }
    }
}

// Helper to avoid duplicating the switch logic
fn generate_captures_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    out: &mut MoveList,
) {
    match piece.piece_type() {
        PieceType::Void | PieceType::Obstacle => {}

        // Pawns: only capture and en-passant moves (with promotions when applicable)
        PieceType::Pawn => {
            generate_pawn_capture_moves(
                board,
                from,
                piece,
                special_rights,
                en_passant,
                game_rules,
                out,
            );
        }

        // Knight-like leapers
        PieceType::Knight => {
            let m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Camel => {
            let m = generate_leaper_moves(board, from, piece, 1, 3);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Giraffe => {
            let m = generate_leaper_moves(board, from, piece, 1, 4);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Zebra => {
            let m = generate_leaper_moves(board, from, piece, 2, 3);
            extend_captures_only(board, piece.color(), m, out);
        }

        // King/Guard/Centaur/RoyalCentaur/Hawk: use compass moves, then filter captures
        PieceType::King | PieceType::Guard => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Centaur | PieceType::RoyalCentaur => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_captures_only(board, piece.color(), m, out);
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_captures_only(board, piece.color(), knight_m, out);
        }
        PieceType::Hawk => {
            let mut m = generate_compass_moves(board, from, piece, 2);
            m.extend(generate_compass_moves(board, from, piece, 3));
            extend_captures_only(board, piece.color(), m, out);
        }

        // Standard sliders and slider-leaper compounds
        PieceType::Rook => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
        }
        PieceType::Bishop => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
        }
        PieceType::Chancellor => {
            // Rook + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            let m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Archbishop => {
            // Bishop + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            let m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_captures_only(board, piece.color(), m, out);
        }
        PieceType::Amazon => {
            // Queen + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            let m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_captures_only(board, piece.color(), m, out);
        }

        // Knightrider: sliding along knight vectors
        PieceType::Knightrider => {
            let m = generate_knightrider_moves(board, from, piece);
            extend_captures_only(board, piece.color(), m, out);
        }

        // Huygen: use existing generator and keep only captures
        PieceType::Huygen => {
            let m = generate_huygen_moves(board, from, piece, indices, false);
            extend_captures_only(board, piece.color(), m, out);
        }

        // Rose: use existing generator and keep only captures
        PieceType::Rose => {
            let m = generate_rose_moves(board, from, piece);
            extend_captures_only(board, piece.color(), m, out);
        }
    }
}

/// Generate pseudo-legal moves for a piece directly into an output buffer.
/// This avoids per-piece allocations during move generation.
#[inline]
pub fn get_pseudo_legal_moves_for_piece_into(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    indices: &SpatialIndices,
    game_rules: &GameRules,
    fallback: bool,
    out: &mut MoveList,
    enemy_king_pos: Option<&Coordinate>,
) {
    match piece.piece_type() {
        // Neutral/blocking pieces cannot move
        PieceType::Void | PieceType::Obstacle => {}
        PieceType::Pawn => {
            generate_pawn_moves_into(
                board,
                from,
                piece,
                special_rights,
                en_passant,
                game_rules,
                out,
            );
        }
        PieceType::Knight => generate_leaper_moves_into(board, from, piece, 1, 2, out),
        PieceType::Hawk => {
            generate_compass_moves_into(board, from, piece, 2, out);
            generate_compass_moves_into(board, from, piece, 3, out);
        }
        PieceType::King => {
            generate_compass_moves_into(board, from, piece, 1, out);
            generate_castling_moves_into(board, from, piece, special_rights, indices, out);
        }
        PieceType::Guard => generate_compass_moves_into(board, from, piece, 1, out),
        PieceType::Rook => {
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Bishop => {
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Chancellor => {
            generate_leaper_moves_into(board, from, piece, 1, 2, out);
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Archbishop => {
            generate_leaper_moves_into(board, from, piece, 1, 2, out);
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Amazon => {
            generate_leaper_moves_into(board, from, piece, 1, 2, out);
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
                enemy_king_pos,
            );
        }
        PieceType::Camel => generate_leaper_moves_into(board, from, piece, 1, 3, out),
        PieceType::Giraffe => generate_leaper_moves_into(board, from, piece, 1, 4, out),
        PieceType::Zebra => generate_leaper_moves_into(board, from, piece, 2, 3, out),
        // Knightrider: slide along all 8 knight directions until blocked
        PieceType::Knightrider => generate_knightrider_moves_into(board, from, piece, out),
        PieceType::Centaur => {
            generate_compass_moves_into(board, from, piece, 1, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, out);
        }
        PieceType::RoyalCentaur => {
            generate_compass_moves_into(board, from, piece, 1, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, out);
            generate_castling_moves_into(board, from, piece, special_rights, indices, out);
        }
        PieceType::Huygen => generate_huygen_moves_into(board, from, piece, indices, fallback, out),
        PieceType::Rose => generate_rose_moves_into(board, from, piece, out),
    }
}

/// Legacy wrapper that allocates a new Vec. Prefer `get_pseudo_legal_moves_for_piece_into` for performance.
pub fn get_pseudo_legal_moves_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    indices: &SpatialIndices,
    game_rules: &GameRules,
    fallback: bool,
    enemy_king_pos: Option<&Coordinate>,
) -> MoveList {
    let mut out = MoveList::new();
    get_pseudo_legal_moves_for_piece_into(
        board,
        piece,
        from,
        special_rights,
        en_passant,
        indices,
        game_rules,
        fallback,
        &mut out,
        enemy_king_pos,
    );
    out
}

/// Ultra-fast attack detection using tile bitboards and spatial indices.
/// O(1) for leapers via precomputed masks, O(log n) for sliders via sorted indices.
#[inline(always)]
pub fn is_square_attacked(
    board: &Board,
    target: &Coordinate,
    attacker_color: PlayerColor,
    indices: &SpatialIndices,
) -> bool {
    use crate::attacks::*;
    use crate::tiles::{local_index, masks};

    // Early exit for neutral
    if attacker_color == PlayerColor::Neutral {
        return false;
    }

    let is_white = attacker_color == PlayerColor::White;
    let neighborhood = board.get_neighborhood(target.x, target.y);
    let local_idx = local_index(target.x, target.y);

    // Get pawn masks (depends on attacker color)
    let pawn_masks = masks::pawn_attacker_masks(is_white);
    let pawn_type_mask = 1u32 << (PieceType::Pawn as u8);

    // SINGLE-PASS: Check all tiles once, checking all leaper+pawn types per tile
    for n in 0..9 {
        let Some(tile) = neighborhood[n] else {
            continue;
        };

        // Get attacker occupancy for this tile
        let occ = if is_white {
            tile.occ_white
        } else {
            tile.occ_black
        };
        if occ == 0 {
            continue;
        }

        // Check each leaper type mask against occupancy
        let masks_to_check = [
            (masks::KNIGHT_MASKS[local_idx][n], KNIGHT_MASK),
            (masks::KING_MASKS[local_idx][n], KING_MASK),
            (masks::CAMEL_MASKS[local_idx][n], CAMEL_MASK),
            (masks::GIRAFFE_MASKS[local_idx][n], GIRAFFE_MASK),
            (masks::ZEBRA_MASKS[local_idx][n], ZEBRA_MASK),
            (masks::HAWK_MASKS[local_idx][n], HAWK_MASK),
            (pawn_masks[local_idx][n], pawn_type_mask),
        ];

        for (attack_mask, type_mask) in masks_to_check {
            let candidates = occ & attack_mask;
            if candidates != 0 {
                let mut bits = candidates;
                while bits != 0 {
                    let bit_idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;

                    let packed = tile.piece[bit_idx];
                    if packed != 0 {
                        let pt = Piece::from_packed(packed).piece_type();
                        if matches_mask(pt, type_mask) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // Slider check using spatial indices (O(log n) per direction)
    #[inline(always)]
    fn check_slider_ray(
        indices: &SpatialIndices,
        target: &Coordinate,
        dx: i64,
        dy: i64,
        attacker_color: PlayerColor,
        type_mask: PieceTypeMask,
    ) -> bool {
        let line_vec = if dx == 0 {
            indices.cols.get(&target.x)
        } else if dy == 0 {
            indices.rows.get(&target.y)
        } else if dx == dy {
            indices.diag1.get(&(target.x - target.y))
        } else {
            indices.diag2.get(&(target.x + target.y))
        };

        if let Some(vec) = line_vec {
            let val = if dx == 0 { target.y } else { target.x };
            let step_dir = if dx == 0 { dy } else { dx };

            if let Some((_, packed)) = SpatialIndices::find_nearest(vec, val, step_dir) {
                let piece = Piece::from_packed(packed);
                if piece.color() == attacker_color && matches_mask(piece.piece_type(), type_mask) {
                    return true;
                }
            }
        }
        false
    }

    // Orthogonal sliders
    for &(dx, dy) in &ORTHO_DIRS {
        if check_slider_ray(indices, target, dx, dy, attacker_color, ORTHO_MASK) {
            return true;
        }
    }

    // Diagonal sliders
    for &(dx, dy) in &DIAG_DIRS {
        if check_slider_ray(indices, target, dx, dy, attacker_color, DIAG_MASK) {
            return true;
        }
    }

    // Knightrider check (sliding knight) - O(1) early exit if no Knightriders exist
    let attacker_idx = if attacker_color == PlayerColor::White {
        0
    } else {
        1
    };
    if indices.has_knightrider[attacker_idx] {
        for &(dx, dy) in &KNIGHTRIDER_DIRS {
            let mut k = 1i64;
            loop {
                let x = target.x + dx * k;
                let y = target.y + dy * k;
                if let Some(piece) = board.get_piece(x, y) {
                    if piece.color() == attacker_color
                        && piece.piece_type() == PieceType::Knightrider
                    {
                        return true;
                    }
                    break;
                }
                k += 1;
                if k > 20 {
                    break;
                }
            }
        }
    }

    // Huygen check (prime distances) - O(1) early exit if no Huygens exist
    // CRITICAL: is_prime_i64 is O(sqrt(n)) - skip huge distances to avoid 31M+ iterations
    if indices.has_huygen[attacker_idx] {
        const HUYGEN_ATTACK_LIMIT: i64 = 100; // Huygen pieces realistically attack within this range
        for &(dx, dy) in &ORTHO_DIRS {
            let line_vec = if dx == 0 {
                indices.cols.get(&target.x)
            } else {
                indices.rows.get(&target.y)
            };
            if let Some(vec) = line_vec {
                for &(coord, packed) in vec {
                    let dist = if dx == 0 {
                        coord - target.y
                    } else {
                        coord - target.x
                    };
                    let abs_dist = dist.abs();
                    // Skip huge distances - is_prime_i64 is O(sqrt(n))!
                    if abs_dist > HUYGEN_ATTACK_LIMIT {
                        continue;
                    }
                    if abs_dist > 0 && is_prime_i64(abs_dist) {
                        let sign = if dist > 0 { 1 } else { -1 };
                        let dir_check = if dx == 0 { dy == sign } else { dx == sign };
                        if dir_check {
                            let piece = Piece::from_packed(packed);
                            if piece.color() == attacker_color
                                && piece.piece_type() == PieceType::Huygen
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Rose check - O(1) early exit if no Roses exist
    // Rose attacks in circular knight patterns; check if any Rose piece can reach target
    if indices.has_rose[attacker_idx] {
        for &(dx, dy) in &ROSE_OFFSETS {
            let x = target.x + dx;
            let y = target.y + dy;
            if let Some(piece) = board.get_piece(x, y) {
                if piece.color() == attacker_color && piece.piece_type() == PieceType::Rose {
                    return true;
                }
            }
        }
    }

    false
}

#[allow(dead_code)]
fn generate_pawn_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
) -> MoveList {
    let mut moves = MoveList::new();
    let direction = match piece.color() {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get promotion ranks for this color
    // If promotion_ranks is not set AND promotions_allowed is not set, use empty (no promotions)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else if game_rules
        .promotions_allowed
        .as_ref()
        .map_or(true, |v| v.is_empty())
    {
        // No promotion_ranks AND no promotions_allowed = no promotions anywhere
        vec![]
    } else {
        // promotions_allowed is set but no ranks = use classical defaults
        match piece.color() {
            PlayerColor::White => vec![8],
            PlayerColor::Black => vec![1],
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    };

    // Get allowed promotion pieces (use pre-converted types, default to Q, R, B, N)
    let default_promos = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    let promotion_pieces: &[PieceType] = game_rules
        .promotion_types
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&default_promos);

    // Helper function to add pawn move with promotion handling
    fn add_pawn_move_inner(
        moves: &mut MoveList,
        from: Coordinate,
        to_x: i64,
        to_y: i64,
        piece: Piece,
        promotion_ranks: &[i64],
        promotion_pieces: &[PieceType],
    ) {
        if promotion_ranks.contains(&to_y) {
            // Generate a move for each possible promotion piece
            for &promo in promotion_pieces {
                let mut m = Move::new(from, Coordinate::new(to_x, to_y), piece);
                m.promotion = Some(promo);
                moves.push(m);
            }
        } else {
            moves.push(Move::new(from, Coordinate::new(to_x, to_y), piece));
        }
    }

    // Move forward 1
    let to_y = from.y + direction;
    let to_x = from.x;

    // Check if square is blocked
    let forward_blocked = board.get_piece(to_x, to_y).is_some();

    if !forward_blocked {
        add_pawn_move_inner(
            &mut moves,
            *from,
            to_x,
            to_y,
            *piece,
            &promotion_ranks,
            &promotion_pieces,
        );

        // Move forward 2 if pawn has special rights (double-move available)
        // This is now dynamic - based on special_rights set, not hardcoded starting rank
        // Note: double-move cannot result in promotion, so no need to check
        if special_rights.contains(from) {
            let to_y_2 = from.y + (direction * 2);
            // Must also check that the target square isn't blocked
            if board.get_piece(to_x, to_y_2).is_none() {
                moves.push(Move::new(*from, Coordinate::new(to_x, to_y_2), *piece));
            }
        }
    }

    // Captures (including neutral pieces - they can be captured)
    for dx in [-1i64, 1] {
        let capture_x = from.x + dx;
        let capture_y = from.y + direction;

        if let Some(target) = board.get_piece(capture_x, capture_y) {
            // Can capture any piece that's not the same color as us
            // This includes neutral pieces (obstacles can be captured)
            if is_enemy_piece(target, piece.color()) {
                add_pawn_move_inner(
                    &mut moves,
                    *from,
                    capture_x,
                    capture_y,
                    *piece,
                    &promotion_ranks,
                    &promotion_pieces,
                );
            }
        } else {
            // En Passant - cannot result in promotion so no promotion check needed
            if let Some(ep) = en_passant {
                if ep.square.x == capture_x && ep.square.y == capture_y {
                    moves.push(Move::new(
                        *from,
                        Coordinate::new(capture_x, capture_y),
                        *piece,
                    ));
                }
            }
        }
    }

    moves
}

/// Generate only pawn captures (including en passant) for quiescence.
fn generate_pawn_capture_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    _special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    out: &mut MoveList,
) {
    let direction = match piece.color() {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get promotion ranks for this color
    // If promotion_ranks is not set AND promotions_allowed is not set, use empty (no promotions)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else if game_rules
        .promotions_allowed
        .as_ref()
        .map_or(true, |v| v.is_empty())
    {
        // No promotion_ranks AND no promotions_allowed = no promotions anywhere
        vec![]
    } else {
        // promotions_allowed is set but no ranks = use classical defaults
        match piece.color() {
            PlayerColor::White => vec![8],
            PlayerColor::Black => vec![1],
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    };

    // Get allowed promotion pieces (use pre-converted types, default to Q, R, B, N)
    let default_promos = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    let promotion_pieces: &[PieceType] = game_rules
        .promotion_types
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&default_promos);

    // Local helper mirroring generate_pawn_moves promotion handling
    fn add_pawn_cap_move(
        out: &mut MoveList,
        from: Coordinate,
        to_x: i64,
        to_y: i64,
        piece: Piece,
        promotion_ranks: &[i64],
        promotion_pieces: &[PieceType],
    ) {
        if promotion_ranks.contains(&to_y) {
            for &promo in promotion_pieces {
                let mut m = Move::new(from, Coordinate::new(to_x, to_y), piece);
                m.promotion = Some(promo);
                out.push(m);
            }
        } else {
            out.push(Move::new(from, Coordinate::new(to_x, to_y), piece));
        }
    }

    // Captures (including neutral pieces - they can be captured)
    for dx in [-1i64, 1] {
        let capture_x = from.x + dx;
        let capture_y = from.y + direction;

        if let Some(target) = board.get_piece(capture_x, capture_y) {
            if is_enemy_piece(&target, piece.color()) {
                // Obstocean Optimization:
                // If it's a neutral piece (Obstacle), capturing it is a "quiet" move in material terms (0 -> 0).
                // Doing this for all obstacles causes a QS explosion.
                // We ONLY allow capturing obstacles in QS if it results in PROMOTION (Tactical win).
                let is_neutral = target.piece_type().is_neutral_type();
                if !is_neutral || promotion_ranks.contains(&capture_y) {
                    add_pawn_cap_move(
                        out,
                        *from,
                        capture_x,
                        capture_y,
                        *piece,
                        &promotion_ranks,
                        &promotion_pieces,
                    );
                }
            }
        } else if let Some(ep) = en_passant {
            // En passant capture square must match
            if ep.square.x == capture_x && ep.square.y == capture_y {
                out.push(Move::new(
                    *from,
                    Coordinate::new(capture_x, capture_y),
                    *piece,
                ));
            }
        }
    }
}

fn generate_castling_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    special_rights: &FxHashSet<Coordinate>,
    indices: &SpatialIndices,
) -> MoveList {
    let mut moves = MoveList::new();

    // King must have special rights to castle
    if !special_rights.contains(from) {
        return moves;
    }

    // Find all pieces with special rights that could be castling partners
    for coord in special_rights.iter() {
        if let Some(target_piece) = board.get_piece(coord.x, coord.y) {
            // Must be same color and a valid castling partner (rook-like piece, not pawn)
            if target_piece.color() == piece.color()
                && target_piece.piece_type() != PieceType::Pawn
                && !target_piece.piece_type().is_royal()
            {
                let dx = coord.x - from.x;
                let dy = coord.y - from.y;

                if dy == 0 {
                    let dir = if dx > 0 { 1i64 } else { -1i64 };

                    // Use spatial indices to check path - O(log n) instead of O(distance)
                    let mut clear = true;
                    if let Some(row_pieces) = indices.rows.get(&from.y) {
                        // Find nearest piece in direction from king
                        if let Some((nearest_x, _)) =
                            SpatialIndices::find_nearest(row_pieces, from.x, dir)
                        {
                            // Path is clear only if no piece between king and rook
                            // nearest_x should equal coord.x (the rook) for clear path
                            if (dir > 0 && nearest_x < coord.x) || (dir < 0 && nearest_x > coord.x)
                            {
                                clear = false; // There's a piece between king and rook
                            }
                        }
                    }

                    if clear {
                        let opponent = piece.color().opponent();

                        let path_1 = from.x + dir;
                        let path_2 = from.x + (dir * 2);

                        let pos_1 = Coordinate::new(path_1, from.y);
                        let pos_2 = Coordinate::new(path_2, from.y);

                        {
                            if !is_square_attacked(board, from, opponent, indices)
                                && !is_square_attacked(board, &pos_1, opponent, indices)
                                && !is_square_attacked(board, &pos_2, opponent, indices)
                            {
                                let to_x = from.x + (dir * 2);
                                let mut castling_move = Move::new(
                                    from.clone(),
                                    Coordinate::new(to_x, from.y),
                                    piece.clone(),
                                );
                                castling_move.rook_coord = Some(coord.clone());
                                moves.push(castling_move);
                            }
                        }
                    }
                }
            }
        }
    }
    moves
}

/// Check if ray is blocked within tile before reaching edge (for magic optimization)
#[inline]
// fn is_ray_blocked_in_tile(from_local: usize, dx: i64, dy: i64, occ: u64) -> bool {
//     let r = (from_local / 8) as i64;
//     let f = (from_local % 8) as i64;
//     let mut cr = r + dy;
//     let mut cf = f + dx;
//     while cr >= 0 && cr < 8 && cf >= 0 && cf < 8 {
//         if (occ >> ((cr as usize) * 8 + cf as usize)) & 1 != 0 {
//             return true;
//         }
//         cr += dy;
//         cf += dx;
//     }
//     false
// }

/// Generate only sliding captures for quiescence search.
/// Uses step-by-step ray tracing with bitboard occupancy fast-path.
fn generate_sliding_capture_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    directions: &[(i64, i64)],
    _indices: &SpatialIndices,
    out: &mut MoveList,
) {
    use crate::tiles::{local_index, tile_coords};

    let our_color = piece.color();

    for &(dx_raw, dy_raw) in directions {
        for sign in [1i64, -1i64] {
            let dx = dx_raw * sign;
            let dy = dy_raw * sign;
            if dx == 0 && dy == 0 {
                continue;
            }

            let mut step = 1i64;
            loop {
                let x = from.x + dx * step;
                let y = from.y + dy * step;

                if !in_bounds(x, y) {
                    break;
                }

                // BITBOARD FAST PATH: Check tile occupancy for O(1) empty detection
                let (cx, cy) = tile_coords(x, y);
                let local_idx = local_index(x, y);

                // Get tile - if no tile exists, square is empty
                let is_occupied = if let Some(tile) = board.tiles.get_tile(cx, cy) {
                    (tile.occ_all >> local_idx) & 1 != 0
                } else {
                    false
                };

                if !is_occupied {
                    step += 1;
                    if step > 50 {
                        break;
                    }
                    continue;
                }

                // Square is occupied - get piece details from tile
                if let Some(tile) = board.tiles.get_tile(cx, cy) {
                    let packed = tile.piece[local_idx];
                    if packed != 0 {
                        let target = Piece::from_packed(packed);
                        // Obstacles are neutral but capturable - check is_uncapturable()
                        if target.color() != our_color && !target.piece_type().is_uncapturable() {
                            out.push(Move::new(*from, Coordinate::new(x, y), *piece));
                        }
                    }
                }
                break; // Square occupied = ray blocked
            }
        }
    }
}

/// Distance from local square to tile edge in given direction
#[inline]
// fn distance_to_tile_edge(from_local: usize, dx: i64, dy: i64) -> i64 {
//     let r = (from_local / 8) as i64;
//     let f = (from_local % 8) as i64;
//     let dist_r = if dy > 0 {
//         7 - r
//     } else if dy < 0 {
//         r
//     } else {
//         i64::MAX
//     };
//     let dist_f = if dx > 0 {
//         7 - f
//     } else if dx < 0 {
//         f
//     } else {
//         i64::MAX
//     };
//     dist_r.min(dist_f)
// }

/// Extend out with only capturing moves from a pre-generated move list.
fn extend_captures_only(
    board: &Board,
    our_color: PlayerColor,
    moves_in: MoveList,
    out: &mut MoveList,
) {
    for m in moves_in {
        if let Some(target) = board.get_piece(m.to.x, m.to.y) {
            // Allow capturing obstacles (neutral but capturable), block only Voids
            if is_enemy_piece(&target, our_color) && !target.piece_type().is_uncapturable() {
                out.push(m);
            }
        }
    }
}

/// Extend out with only quiet (non-capturing) moves from a pre-generated move list.
fn extend_quiets_only(board: &Board, moves_in: MoveList, out: &mut MoveList) {
    for m in moves_in {
        if board.get_piece(m.to.x, m.to.y).is_none() {
            out.push(m);
        }
    }
}

/// Generate only quiet (non-capturing) moves for staged move generation.
/// This is the complement of get_quiescence_captures.
pub fn get_quiet_moves_into(
    board: &Board,
    turn: PlayerColor,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    out: &mut MoveList,
    enemy_king_pos: Option<&Coordinate>, // For check target computation
) {
    out.clear();

    // BITBOARD: Use fast color-specific bitboard iteration
    let is_white = turn == PlayerColor::White;
    for (x, y, piece) in board.iter_pieces_by_color(is_white) {
        if piece.color() == PlayerColor::Neutral {
            continue;
        }

        let from = Coordinate::new(x, y);
        generate_quiets_for_piece(
            board,
            &piece,
            &from,
            special_rights,
            en_passant,
            game_rules,
            indices,
            out,
            enemy_king_pos,
        );
    }
}

/// Generate only quiet moves for a single piece.
fn generate_quiets_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    special_rights: &FxHashSet<Coordinate>,
    _en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    indices: &SpatialIndices,
    out: &mut MoveList,
    enemy_king_pos: Option<&Coordinate>,
) {
    match piece.piece_type() {
        PieceType::Void | PieceType::Obstacle => {}

        // Pawns: only forward moves (single and double push), no captures
        PieceType::Pawn => {
            generate_pawn_quiet_moves(board, from, piece, special_rights, game_rules, out);
        }

        // Knight-like leapers: filter to empty squares
        PieceType::Knight => {
            let m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, m, out);
        }
        PieceType::Camel => {
            let m = generate_leaper_moves(board, from, piece, 1, 3);
            extend_quiets_only(board, m, out);
        }
        PieceType::Giraffe => {
            let m = generate_leaper_moves(board, from, piece, 1, 4);
            extend_quiets_only(board, m, out);
        }
        PieceType::Zebra => {
            let m = generate_leaper_moves(board, from, piece, 2, 3);
            extend_quiets_only(board, m, out);
        }

        // King: compass + castling
        PieceType::King => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_quiets_only(board, m, out);
            // Castling is always a quiet move
            let castling = generate_castling_moves(board, from, piece, special_rights, indices);
            out.extend(castling);
        }
        PieceType::Guard => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_quiets_only(board, m, out);
        }
        PieceType::Centaur => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_quiets_only(board, m, out);
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
        }
        PieceType::RoyalCentaur => {
            let m = generate_compass_moves(board, from, piece, 1);
            extend_quiets_only(board, m, out);
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let castling = generate_castling_moves(board, from, piece, special_rights, indices);
            out.extend(castling);
        }
        PieceType::Hawk => {
            let mut m = generate_compass_moves(board, from, piece, 2);
            m.extend(generate_compass_moves(board, from, piece, 3));
            extend_quiets_only(board, m, out);
        }

        // Sliders
        PieceType::Rook => {
            let m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                false,
                enemy_king_pos,
            );
            extend_quiets_only(board, m, out);
        }
        PieceType::Bishop => {
            let m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
                enemy_king_pos,
            );
            extend_quiets_only(board, m, out);
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            let mut m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                false,
                enemy_king_pos,
            );
            m.extend(generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
                enemy_king_pos,
            ));
            extend_quiets_only(board, m, out);
        }
        PieceType::Chancellor => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let rook_m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                false,
                enemy_king_pos,
            );
            extend_quiets_only(board, rook_m, out);
        }
        PieceType::Archbishop => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let bishop_m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
                enemy_king_pos,
            );
            extend_quiets_only(board, bishop_m, out);
        }
        PieceType::Amazon => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let mut queen_m = generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 0), (0, 1)],
                indices,
                false,
                enemy_king_pos,
            );
            queen_m.extend(generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
                enemy_king_pos,
            ));
            extend_quiets_only(board, queen_m, out);
        }

        PieceType::Knightrider => {
            let m = generate_knightrider_moves(board, from, piece);
            extend_quiets_only(board, m, out);
        }
        PieceType::Huygen => {
            let m = generate_huygen_moves(board, from, piece, indices, false);
            extend_quiets_only(board, m, out);
        }
        PieceType::Rose => {
            let m = generate_rose_moves(board, from, piece);
            extend_quiets_only(board, m, out);
        }
    }
}

/// Generate pawn quiet moves (forward pushes only, no captures)
fn generate_pawn_quiet_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    special_rights: &FxHashSet<Coordinate>,
    game_rules: &GameRules,
    out: &mut MoveList,
) {
    let direction = match piece.color() {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get promotion ranks
    let promotion_ranks: Vec<i64> = match piece.color() {
        PlayerColor::White => game_rules
            .promotion_ranks
            .as_ref()
            .map(|p| p.white.clone())
            .unwrap_or_else(|| vec![8]),
        PlayerColor::Black => game_rules
            .promotion_ranks
            .as_ref()
            .map(|p| p.black.clone())
            .unwrap_or_else(|| vec![1]),
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    let default_promos = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    let promotion_pieces: &[PieceType] = game_rules
        .promotion_types
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&default_promos);

    // Single push
    let to_y = from.y + direction;
    let to_x = from.x;

    if board.get_piece(to_x, to_y).is_none() {
        // Square is empty, can push
        if promotion_ranks.contains(&to_y) {
            for &promo in promotion_pieces {
                let mut m = Move::new(*from, Coordinate::new(to_x, to_y), *piece);
                m.promotion = Some(promo);
                out.push(m);
            }
        } else {
            out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
        }

        // Double push if pawn has special rights
        if special_rights.contains(from) {
            let double_y = from.y + 2 * direction;
            if board.get_piece(to_x, double_y).is_none() {
                out.push(Move::new(*from, Coordinate::new(to_x, double_y), *piece));
            }
        }
    }
}

fn generate_compass_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    distance: i64,
) -> MoveList {
    let mut moves = MoveList::new();
    let dist = distance;
    let offsets = [
        (-dist, dist),
        (0, dist),
        (dist, dist),
        (-dist, 0),
        (dist, 0),
        (-dist, -dist),
        (0, -dist),
        (dist, -dist),
    ];

    for (dx, dy) in offsets {
        let to_x = from.x + dx;
        let to_y = from.y + dy;

        // Skip if outside world border
        if !in_bounds(to_x, to_y) {
            continue;
        }

        if let Some(target) = board.get_piece(to_x, to_y) {
            if is_enemy_piece(&target, piece.color()) {
                moves.push(Move::new(
                    from.clone(),
                    Coordinate::new(to_x, to_y),
                    piece.clone(),
                ));
            }
        } else {
            moves.push(Move::new(
                from.clone(),
                Coordinate::new(to_x, to_y),
                piece.clone(),
            ));
        }
    }

    moves
}

fn generate_leaper_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    m: i64,
    n: i64,
) -> MoveList {
    let mut moves = MoveList::new();
    generate_leaper_moves_into(board, from, piece, m, n, &mut moves);
    moves
}

/// Generate leaper moves directly into an output buffer
#[inline]
fn generate_leaper_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    m: i64,
    n: i64,
    out: &mut MoveList,
) {
    let offsets = [
        (-n, m),
        (-m, n),
        (m, n),
        (n, m),
        (-n, -m),
        (-m, -n),
        (m, -n),
        (n, -m),
    ];

    for (dx, dy) in offsets {
        let to_x = from.x + dx;
        let to_y = from.y + dy;

        // Skip if outside world border
        if !in_bounds(to_x, to_y) {
            continue;
        }

        if let Some(target) = board.get_piece(to_x, to_y) {
            if is_enemy_piece(&target, piece.color()) {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
        } else {
            out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
        }
    }
}

/// Generate compass moves directly into an output buffer
#[inline]
fn generate_compass_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    distance: i64,
    out: &mut MoveList,
) {
    let dist = distance;
    let offsets = [
        (-dist, dist),
        (0, dist),
        (dist, dist),
        (-dist, 0),
        (dist, 0),
        (-dist, -dist),
        (0, -dist),
        (dist, -dist),
    ];

    for (dx, dy) in offsets {
        let to_x = from.x + dx;
        let to_y = from.y + dy;

        // Skip if outside world border
        if !in_bounds(to_x, to_y) {
            continue;
        }

        if let Some(target) = board.get_piece(to_x, to_y) {
            if is_enemy_piece(&target, piece.color()) {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
        } else {
            out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
        }
    }
}

#[inline]
fn ray_border_distance(from: &Coordinate, dir_x: i64, dir_y: i64) -> Option<i64> {
    if dir_x == 0 && dir_y == 0 {
        return None;
    }

    unsafe {
        let min_x = COORD_MIN_X;
        let max_x = COORD_MAX_X;
        let min_y = COORD_MIN_Y;
        let max_y = COORD_MAX_Y;

        // Allow one extra "infinite" move, but clamp it so we don't shoot pieces
        // off to absurd coordinates. 256 steps is effectively infinite for any
        // reasonable board while keeping coordinates well-behaved, even when the
        // underlying world box is huge.
        const MAX_INF_DISTANCE: i64 = 256;

        if dir_x == 0 {
            let raw = if dir_y > 0 {
                max_y - from.y
            } else {
                from.y - min_y
            };
            let limit = raw.min(MAX_INF_DISTANCE);
            if limit > 0 { Some(limit) } else { None }
        } else if dir_y == 0 {
            let raw = if dir_x > 0 {
                max_x - from.x
            } else {
                from.x - min_x
            };
            let limit = raw.min(MAX_INF_DISTANCE);
            if limit > 0 { Some(limit) } else { None }
        } else if dir_x.abs() == dir_y.abs() {
            let raw_x = if dir_x > 0 {
                max_x - from.x
            } else {
                from.x - min_x
            };
            let raw_y = if dir_y > 0 {
                max_y - from.y
            } else {
                from.y - min_y
            };
            let raw = raw_x.min(raw_y);
            let limit = raw.min(MAX_INF_DISTANCE);
            if limit > 0 { Some(limit) } else { None }
        } else {
            None
        }
    }
}

/// Find cross-ray attack targets for sliders (ULTRA-FAST).
///
/// Mathematical Intersection Approach:
/// 1. Iterate over ALL pieces on the board (O(N), usually < 60).
/// 2. For each piece P, calculate if any of its 8 attack rays intersect our slider's ray.
/// 3. If an intersection S exists within max_dist, verify reachability.
#[inline]
fn find_cross_ray_targets_into(
    board: &Board,
    from: &Coordinate,
    dir_x: i64,
    dir_y: i64,
    max_dist: i64,
    indices: &SpatialIndices,
    our_color: PlayerColor,
    piece_type: PieceType,
    enemy_wiggle: i64,
    friend_wiggle: i64,
    out: &mut Vec<i64>,
) {
    // Filter by piece type attack capabilities
    let checks_ortho = matches!(
        piece_type,
        PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Rook
            | PieceType::Chancellor
            | PieceType::Amazon
    );
    let checks_diag = matches!(
        piece_type,
        PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Bishop
            | PieceType::Archbishop
            | PieceType::Amazon
    );

    if !checks_ortho && !checks_diag {
        return;
    }

    // Precompute constant ray properties
    let ray_diff = dir_x - dir_y;
    let ray_sum = dir_x + dir_y;

    // Iterate all pieces on the board once
    for (px, py, p) in board.tiles.iter_all_pieces() {
        // Skip same position or same color
        if (px == from.x && py == from.y) || p.color() == our_color {
            continue;
        }

        let is_enemy = p.color() != our_color && !p.piece_type().is_uncapturable();
        let wiggle = if is_enemy {
            enemy_wiggle
        } else {
            friend_wiggle
        };

        // 1. Orthogonal Cross-Rays (Vertical/Horizontal)
        if checks_ortho {
            // Vertical cross: S.x = px
            if dir_x != 0 {
                let num = px - from.x;
                if num.signum() == dir_x.signum() && num % dir_x == 0 {
                    let d = num / dir_x;
                    if d > 0 && d <= max_dist {
                        let sy = from.y + d * dir_y;
                        if py != sy {
                            let step = (py - sy).signum();
                            if let Some(pieces_in_col) = indices.cols.get(&px) {
                                if let Some((nearest_y, _)) =
                                    SpatialIndices::find_nearest(pieces_in_col, sy, step)
                                {
                                    if nearest_y == py {
                                        // Wiggle allowed for orthogonal
                                        for w in -wiggle..=wiggle {
                                            let wd = d + w;
                                            if wd > 0 && wd <= max_dist {
                                                out.push(wd);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Horizontal cross: S.y = py
            if dir_y != 0 {
                let num = py - from.y;
                if num.signum() == dir_y.signum() && num % dir_y == 0 {
                    let d = num / dir_y;
                    if d > 0 && d <= max_dist {
                        let sx = from.x + d * dir_x;
                        if px != sx {
                            let step = (px - sx).signum();
                            if let Some(pieces_in_row) = indices.rows.get(&py) {
                                if let Some((nearest_x, _)) =
                                    SpatialIndices::find_nearest(pieces_in_row, sx, step)
                                {
                                    if nearest_x == px {
                                        // Wiggle allowed for orthogonal
                                        for w in -wiggle..=wiggle {
                                            let wd = d + w;
                                            if wd > 0 && wd <= max_dist {
                                                out.push(wd);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Diagonal Cross-Rays
        if checks_diag {
            // Diag1: S.x - S.y = px - py
            if ray_diff != 0 {
                let num = (px - py) - (from.x - from.y);
                if num.signum() == ray_diff.signum() && num % ray_diff == 0 {
                    let d = num / ray_diff;
                    if d > 0 && d <= max_dist {
                        let sx = from.x + d * dir_x;
                        if px != sx {
                            let step = (px - sx).signum();
                            if let Some(pieces_on_diag) = indices.diag1.get(&(px - py)) {
                                if let Some((nearest_x, _)) =
                                    SpatialIndices::find_nearest(pieces_on_diag, sx, step)
                                {
                                    if nearest_x == px {
                                        // NO wiggle for diagonal
                                        out.push(d);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Diag2: S.x + S.y = px + py
            if ray_sum != 0 {
                let num = (px + py) - (from.x + from.y);
                if num.signum() == ray_sum.signum() && num % ray_sum == 0 {
                    let d = num / ray_sum;
                    if d > 0 && d <= max_dist {
                        let sx = from.x + d * dir_x;
                        if px != sx {
                            let step = (px - sx).signum();
                            if let Some(pieces_on_diag) = indices.diag2.get(&(px + py)) {
                                if let Some((nearest_x, _)) =
                                    SpatialIndices::find_nearest(pieces_on_diag, sx, step)
                                {
                                    if nearest_x == px {
                                        // NO wiggle for diagonal
                                        out.push(d);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn generate_sliding_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    directions: &[(i64, i64)],
    indices: &SpatialIndices,
    fallback: bool,
    enemy_king_pos: Option<&Coordinate>, // Cached enemy king position for O(1) check target computation
) -> MoveList {
    // Original wiggle values - important for tactics
    const ENEMY_WIGGLE: i64 = 2;
    const FRIEND_WIGGLE: i64 = 1;
    // Maximum distance for interception - 50 is needed for long-range tactics
    // We only cap "interception" moves (crossing an enemy's line), not direct captures!
    const MAX_INTERCEPTION_DIST: i64 = 50;

    // Fallback limit for short-range slider moves
    const FALLBACK_LIMIT: i64 = 10;

    let mut moves = MoveList::new();
    let our_color = piece.color();

    let ek_ref = enemy_king_pos;

    for &(dx_raw, dy_raw) in directions {
        for sign in [1i64, -1i64] {
            let dir_x = dx_raw * sign;
            let dir_y = dy_raw * sign;

            if dir_x == 0 && dir_y == 0 {
                continue;
            }

            if fallback {
                for dist in 1..=FALLBACK_LIMIT {
                    let tx = from.x + dir_x * dist;
                    let ty = from.y + dir_y * dist;

                    // Stop if out of bounds (though unlikely with infinite board coords)
                    if !in_bounds(tx, ty) {
                        break;
                    }

                    if let Some(target) = board.get_piece(tx, ty) {
                        // If blocked, check if we can capture
                        // Obstacles are neutral but capturable - check is_uncapturable()
                        let is_enemy =
                            target.color() != our_color && !target.piece_type().is_uncapturable();
                        if is_enemy {
                            moves.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                        }
                        break; // blocked
                    } else {
                        moves.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                    }
                }
                continue;
            }

            let is_vertical = dir_x == 0;
            let is_horizontal = dir_y == 0;

            // Use spatial indices for O(log n) blocker finding when available
            let (closest_dist, closest_is_enemy) =
                find_blocker_via_indices(board, from, dir_x, dir_y, indices, our_color);

            let max_dist = if closest_dist < i64::MAX {
                if closest_is_enemy {
                    closest_dist
                } else {
                    closest_dist - 1
                }
            } else {
                match ray_border_distance(from, dir_x, dir_y) {
                    Some(d) if d > 0 => d,
                    _ => 0,
                }
            };

            if max_dist <= 0 {
                continue;
            }

            // Efficient interception limit for OFF-RAY pieces
            let interception_limit = max_dist.min(MAX_INTERCEPTION_DIST);

            // Use Vec for target distances to avoid overflow
            let mut target_dists: Vec<i64> = Vec::with_capacity(64);

            // Start wiggle room (always add these)
            for d in 1..=ENEMY_WIGGLE {
                target_dists.push(d);
            }

            // CRITICAL: Always add direct capture distance if there's an enemy piece
            // This is O(1) via spatial indices and covers ANY distance
            if closest_dist < i64::MAX && closest_is_enemy {
                target_dists.push(closest_dist);
            }

            // Cache key: (x, y, direction_index) where direction_index encodes (dir_x, dir_y)
            // Direction encoding: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
            let dir_index: u8 = match (dir_x.signum(), dir_y.signum()) {
                (1, 0) => 0,   // East
                (1, 1) => 1,   // NE
                (0, 1) => 2,   // North
                (-1, 1) => 3,  // NW
                (-1, 0) => 4,  // West
                (-1, -1) => 5, // SW
                (0, -1) => 6,  // South
                (1, -1) => 7,  // SE
                _ => 0,        // fallback
            };
            let cache_key = (from.x, from.y, dir_index);

            // Check cache first
            let cached = indices.slider_cache.borrow().get(&cache_key).cloned();
            if let Some(cached_dists) = cached {
                // Use cached interception distances, but still add blocker wiggle room
                target_dists.extend(cached_dists.iter().filter(|&&d| d <= interception_limit));
            } else {
                // Compute interception distances by iterating PIECES on the ray, not squares
                // This is O(pieces_on_line) which is typically small

                if is_horizontal {
                    // Horizontal ray: iterate pieces on the same row
                    if let Some(pieces_on_row) = indices.rows.get(&from.y) {
                        for &(px, packed) in pieces_on_row {
                            let dx = px - from.x;
                            // Skip pieces not in our direction or at origin
                            if dx == 0 || dx.signum() != dir_x.signum() {
                                continue;
                            }
                            let piece_dist = dx.abs();
                            // Only interception for pieces within limit (direct capture handled separately)
                            if piece_dist > interception_limit {
                                continue;
                            }
                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                if d > 0 && d <= interception_limit {
                                    target_dists.push(d);
                                }
                            }
                        }
                    }
                } else if is_vertical {
                    // Vertical ray: iterate pieces on the same column
                    if let Some(pieces_on_col) = indices.cols.get(&from.x) {
                        for &(py, packed) in pieces_on_col {
                            let dy = py - from.y;
                            if dy == 0 || dy.signum() != dir_y.signum() {
                                continue;
                            }
                            let piece_dist = dy.abs();
                            if piece_dist > interception_limit {
                                continue;
                            }
                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                if d > 0 && d <= interception_limit {
                                    target_dists.push(d);
                                }
                            }
                        }
                    }
                } else {
                    // Diagonal ray: iterate pieces on the same diagonal
                    let is_diag1_dir = dir_x == dir_y;
                    let diag_key = if is_diag1_dir {
                        from.x - from.y
                    } else {
                        from.x + from.y
                    };
                    let diag_map = if is_diag1_dir {
                        &indices.diag1
                    } else {
                        &indices.diag2
                    };

                    if let Some(pieces_on_diag) = diag_map.get(&diag_key) {
                        for &(px, packed) in pieces_on_diag {
                            let dx = px - from.x;
                            if dx == 0 || dx.signum() != dir_x.signum() {
                                continue;
                            }
                            let piece_dist = dx.abs();
                            if piece_dist > interception_limit {
                                continue;
                            }
                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                if d > 0 && d <= interception_limit {
                                    target_dists.push(d);
                                }
                            }
                        }
                    }
                }

                // Add cross-ray attack targets: pieces on perpendicular rays that we could
                // attack from squares along our current ray direction
                find_cross_ray_targets_into(
                    board,
                    from,
                    dir_x,
                    dir_y,
                    interception_limit,
                    indices,
                    our_color,
                    piece.piece_type(),
                    ENEMY_WIGGLE,
                    FRIEND_WIGGLE,
                    &mut target_dists,
                );

                // Store computed distances in cache
                indices
                    .slider_cache
                    .borrow_mut()
                    .insert(cache_key, target_dists.clone());
            } // end else (cache miss)

            // Add blocker wiggle room (up to max_dist)
            if closest_dist < i64::MAX {
                let wr = if closest_is_enemy {
                    ENEMY_WIGGLE
                } else {
                    FRIEND_WIGGLE
                };
                let start = closest_dist.saturating_sub(wr).max(1);
                for d in start..=closest_dist {
                    if d <= max_dist {
                        target_dists.push(d);
                    }
                }
            }

            // Add check target: if we have enemy king position, compute distance for check
            // This is O(1) per direction - much simpler and faster than scanning
            if let Some(ek) = ek_ref {
                let kx = ek.x;
                let ky = ek.y;
                let piece_type = piece.piece_type();

                // For orthogonal rays, check if we can give diagonal check
                if is_horizontal {
                    // Horizontal ray: find column where diagonal check is possible
                    // From (tx, from.y), can attack (kx, ky) diagonally if |tx - kx| == |from.y - ky|
                    if from.y != ky {
                        if matches!(
                            piece_type,
                            crate::board::PieceType::Queen
                                | crate::board::PieceType::Bishop
                                | crate::board::PieceType::Archbishop
                                | crate::board::PieceType::Amazon
                        ) {
                            let diff = (from.y - ky).abs();
                            for target_x in [kx + diff, kx - diff] {
                                let dx = target_x - from.x;
                                if dx != 0 && dx.signum() == dir_x.signum() {
                                    let d = dx.abs();
                                    if d <= max_dist && d <= MAX_INTERCEPTION_DIST {
                                        target_dists.push(d);
                                    }
                                }
                            }
                        }
                    }
                } else if is_vertical {
                    // Vertical ray: find row where diagonal check is possible
                    // From (from.x, ty), can attack (kx, ky) diagonally if |from.x - kx| == |ty - ky|
                    if from.x != kx {
                        if matches!(
                            piece_type,
                            crate::board::PieceType::Queen
                                | crate::board::PieceType::Bishop
                                | crate::board::PieceType::Archbishop
                                | crate::board::PieceType::Amazon
                        ) {
                            let diff = (from.x - kx).abs();
                            for target_y in [ky + diff, ky - diff] {
                                let dy = target_y - from.y;
                                if dy != 0 && dy.signum() == dir_y.signum() {
                                    let d = dy.abs();
                                    if d <= max_dist && d <= MAX_INTERCEPTION_DIST {
                                        target_dists.push(d);
                                    }
                                }
                            }
                        }
                    }
                }
                // TODO: diagonal rays could check for orthogonal attacks if needed
            }

            // Sort and deduplicate
            target_dists.sort_unstable();
            target_dists.dedup();

            // Generate moves
            for d in target_dists {
                if d <= 0 || d > max_dist {
                    continue;
                }

                // Skip if this is a friendly blocker square
                if d == closest_dist && !closest_is_enemy {
                    continue;
                }

                let sq_x = from.x + dir_x * d;
                let sq_y = from.y + dir_y * d;

                if !in_bounds(sq_x, sq_y) {
                    continue;
                }

                moves.push(Move::new(*from, Coordinate::new(sq_x, sq_y), *piece));
            }
        }
    }

    moves
}

/// Find the closest blocker on a ray using spatial indices (O(log n))
/// Now uses the new SpatialIndices format with inline piece data
#[inline]
fn find_blocker_via_indices(
    _board: &Board,
    from: &Coordinate,
    dir_x: i64,
    dir_y: i64,
    indices: &SpatialIndices,
    our_color: PlayerColor,
) -> (i64, bool) {
    let is_vertical = dir_x == 0;
    let is_horizontal = dir_y == 0;
    let is_diag1 = dir_x == dir_y; // Moving along x-y = const

    let line_vec = if is_vertical {
        indices.cols.get(&from.x)
    } else if is_horizontal {
        indices.rows.get(&from.y)
    } else if is_diag1 {
        indices.diag1.get(&(from.x - from.y))
    } else {
        indices.diag2.get(&(from.x + from.y))
    };

    if let Some(vec) = line_vec {
        let search_val = if is_vertical { from.y } else { from.x };
        let step_dir = if is_vertical { dir_y } else { dir_x };

        // Use the new find_nearest helper
        if let Some((next_coord, packed)) = SpatialIndices::find_nearest(vec, search_val, step_dir)
        {
            let dist = (next_coord - search_val).abs();

            // Verify this is actually in the correct direction
            if (next_coord > search_val) != (step_dir > 0) {
                return (i64::MAX, false);
            }

            let piece = Piece::from_packed(packed);
            // Obstacles are neutral but capturable - check is_uncapturable()
            let is_enemy = piece.color() != our_color && !piece.piece_type().is_uncapturable();
            return (dist, is_enemy);
        }
    }

    (i64::MAX, false)
}

fn generate_huygen_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    indices: &SpatialIndices,
    fallback: bool,
) -> MoveList {
    let mut moves = MoveList::new();
    let directions = [(1, 0), (0, 1)];
    const FALLBACK_LIMIT: i64 = 10;

    for (dx_raw, dy_raw) in directions {
        for sign in [1, -1] {
            let dir_x = dx_raw * sign;
            let dir_y = dy_raw * sign;

            let mut closest_prime_dist: Option<i64> = None;
            let mut closest_piece_color: Option<PlayerColor> = None;

            let mut found_via_indices = false;

            // Fallback mode: scan step-by-step but still respect prime distance requirement.
            if fallback {
                const PRIMES: [i64; 4] = [2, 3, 5, 7];

                for &target_dist in &PRIMES {
                    if target_dist > FALLBACK_LIMIT {
                        break;
                    }

                    let tx = from.x + dir_x * target_dist;
                    let ty = from.y + dir_y * target_dist;

                    // Check if path is clear up to target_dist - 1
                    let mut path_blocked = false;
                    for step in 1..target_dist {
                        let sx = from.x + dir_x * step;
                        let sy = from.y + dir_y * step;
                        if board.get_piece(sx, sy).is_some() {
                            path_blocked = true;
                            break;
                        }
                    }

                    if path_blocked {
                        break; // Can't reach any further primes in this direction
                    }

                    // Check target square
                    if let Some(target) = board.get_piece(tx, ty) {
                        // Void blocks like friendly
                        if target.piece_type() == PieceType::Void {
                            break;
                        }
                        // Obstacles are neutral but capturable - check is_uncapturable()
                        let is_enemy = target.color() != piece.color()
                            && !target.piece_type().is_uncapturable();
                        if is_enemy {
                            moves.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                        }
                        break; // blocked at this prime, can't go further
                    } else {
                        // Empty square at prime distance - valid move
                        moves.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                    }
                }
                continue;
            }

            let line_vec = if dx_raw == 0 {
                indices.cols.get(&from.x)
            } else {
                indices.rows.get(&from.y)
            };
            if let Some(vec) = line_vec {
                let val = if dx_raw == 0 { from.y } else { from.x };
                // Binary search by coordinate only
                if let Ok(idx) = vec.binary_search_by_key(&val, |(c, _)| *c) {
                    let step_dir = if dx_raw == 0 { dir_y } else { dir_x };
                    if step_dir > 0 {
                        for i in (idx + 1)..vec.len() {
                            let (next_coord, packed) = vec[i];
                            let dist = next_coord - val;
                            if is_prime_i64(dist) {
                                closest_prime_dist = Some(dist);
                                let p = Piece::from_packed(packed);
                                // Treat Void as friendly for capture purposes
                                closest_piece_color = Some(if p.piece_type() == PieceType::Void {
                                    piece.color()
                                } else {
                                    p.color()
                                });
                                break;
                            }
                        }
                    } else {
                        for i in (0..idx).rev() {
                            let (prev_coord, packed) = vec[i];
                            let dist = val - prev_coord;
                            if is_prime_i64(dist) {
                                closest_prime_dist = Some(dist);
                                let p = Piece::from_packed(packed);
                                // Treat Void as friendly for capture purposes
                                closest_piece_color = Some(if p.piece_type() == PieceType::Void {
                                    piece.color()
                                } else {
                                    p.color()
                                });
                                break;
                            }
                        }
                    }
                    found_via_indices = true;
                }
            }

            if !found_via_indices {
                for ((px, py), target_piece) in board.iter() {
                    let dx = px - from.x;
                    let dy = py - from.y;
                    let k = if dir_x != 0 {
                        if dx % dir_x == 0 && dy == 0 {
                            Some(dx / dir_x)
                        } else {
                            None
                        }
                    } else {
                        if dy % dir_y == 0 && dx == 0 {
                            Some(dy / dir_y)
                        } else {
                            None
                        }
                    };

                    if let Some(dist) = k {
                        if dist > 0 {
                            if is_prime_i64(dist) {
                                if closest_prime_dist.as_ref().map_or(true, |d| dist < *d) {
                                    closest_prime_dist = Some(dist);
                                    // Treat Void as friendly for capture purposes
                                    closest_piece_color =
                                        Some(if target_piece.piece_type() == PieceType::Void {
                                            piece.color()
                                        } else {
                                            target_piece.color()
                                        });
                                }
                            }
                        }
                    }
                }
            }

            let limit = closest_prime_dist.unwrap_or(100);

            if closest_prime_dist.is_some() {
                // Original behavior: generate all prime-distance squares up to
                // the blocking piece at `limit`.
                for s in 2..=limit {
                    if is_prime_i64(s) {
                        let to_x = from.x + (dir_x * s);
                        let to_y = from.y + (dir_y * s);

                        if s == limit {
                            if closest_piece_color != Some(piece.color()) {
                                moves.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                            }
                        } else {
                            moves.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                        }
                    }
                }
            } else {
                // No blocking piece at any prime distance along this ray.
                // Instead of generating all prime squares up to 50, only keep
                // those whose destination is aligned with *some* piece on the
                // orthogonal axis (file for horizontal moves, rank for
                // vertical moves).
                let scan_limit = 50i64;

                for s in 2..=scan_limit {
                    if !is_prime_i64(s) {
                        continue;
                    }

                    let to_x = from.x + (dir_x * s);
                    let to_y = from.y + (dir_y * s);

                    let aligned = if dir_x != 0 {
                        indices.cols.get(&to_x).map_or(false, |v| !v.is_empty())
                    } else {
                        indices.rows.get(&to_y).map_or(false, |v| !v.is_empty())
                    };

                    if aligned {
                        moves.push(Move::new(
                            from.clone(),
                            Coordinate::new(to_x, to_y),
                            piece.clone(),
                        ));
                    }
                }
            }
        }
    }
    moves
}

fn generate_rose_moves(board: &Board, from: &Coordinate, piece: &Piece) -> MoveList {
    let mut moves = MoveList::new();
    generate_rose_moves_into(board, from, piece, &mut moves);
    moves
}

/// Rose movement offsets (precomputed):
/// - 8 knight moves (jumps, no blocking)
/// - 8 diagonal: skips first 2, lands on positions 3 and 4 in each diagonal direction
/// - 8 orthogonal: skips first 3, lands on position 4; skips 5, lands on 6
pub static ROSE_OFFSETS: [(i64, i64); 32] = [
    // Knight moves (8)
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    // Diagonal: skip 2, then positions 3 and 4
    (3, 3),
    (4, 4),
    (-3, -3),
    (-4, -4),
    (3, -3),
    (4, -4),
    (-3, 3),
    (-4, 4),
    // Orthogonal: skip 3, position 4; skip 5, position 6
    (4, 0),
    (6, 0),
    (-4, 0),
    (-6, 0),
    (0, 4),
    (0, 6),
    (0, -4),
    (0, -6),
    // Extended knight-like: (5,2) in all 8 directions
    (5, 2),
    (5, -2),
    (-5, 2),
    (-5, -2),
    (2, 5),
    (2, -5),
    (-2, 5),
    (-2, -5),
];

/// Generate rose moves directly into an output buffer using precomputed offsets.
#[inline(always)]
fn generate_rose_moves_into(board: &Board, from: &Coordinate, piece: &Piece, out: &mut MoveList) {
    let my_color = piece.color();

    for &(dx, dy) in &ROSE_OFFSETS {
        let tx = from.x + dx;
        let ty = from.y + dy;

        if let Some(target) = board.get_piece(tx, ty) {
            // Can capture enemy pieces
            if is_enemy_piece(target, my_color) {
                out.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
            }
        } else {
            // Empty square - can move there
            out.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
        }
    }
}

/// Generate pawn moves directly into an output buffer
#[inline]
fn generate_pawn_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    special_rights: &FxHashSet<Coordinate>,
    en_passant: &Option<EnPassantState>,
    game_rules: &GameRules,
    out: &mut MoveList,
) {
    let direction = match piece.color() {
        PlayerColor::White => 1,
        PlayerColor::Black => -1,
        PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
    };

    // Get promotion ranks for this color
    // If promotion_ranks is not set AND promotions_allowed is not set, use empty (no promotions)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else if game_rules
        .promotions_allowed
        .as_ref()
        .map_or(true, |v| v.is_empty())
    {
        // No promotion_ranks AND no promotions_allowed = no promotions anywhere
        vec![]
    } else {
        // promotions_allowed is set but no ranks = use classical defaults
        match piece.color() {
            PlayerColor::White => vec![8],
            PlayerColor::Black => vec![1],
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    };

    let default_promos = [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ];
    let promotion_pieces: &[PieceType] = game_rules
        .promotion_types
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&default_promos);

    // Helper function for promotion moves
    #[inline]
    fn add_pawn_move(
        out: &mut MoveList,
        from: Coordinate,
        to_x: i64,
        to_y: i64,
        piece: Piece,
        promotion_ranks: &[i64],
        promotion_pieces: &[PieceType],
    ) {
        if promotion_ranks.contains(&to_y) {
            for &promo in promotion_pieces {
                let mut m = Move::new(from, Coordinate::new(to_x, to_y), piece);
                m.promotion = Some(promo);
                out.push(m);
            }
        } else {
            out.push(Move::new(from, Coordinate::new(to_x, to_y), piece));
        }
    }

    // Move forward 1
    let to_y = from.y + direction;
    let to_x = from.x;
    let forward_blocked = board.get_piece(to_x, to_y).is_some();

    if !forward_blocked {
        add_pawn_move(
            out,
            *from,
            to_x,
            to_y,
            *piece,
            &promotion_ranks,
            promotion_pieces,
        );

        // Double push (can also result in promotion in some variants)
        if special_rights.contains(from) {
            let to_y_2 = from.y + (direction * 2);
            if board.get_piece(to_x, to_y_2).is_none() {
                add_pawn_move(
                    out,
                    *from,
                    to_x,
                    to_y_2,
                    *piece,
                    &promotion_ranks,
                    promotion_pieces,
                );
            }
        }
    }

    // Captures
    for dx in [-1i64, 1] {
        let capture_x = from.x + dx;
        let capture_y = from.y + direction;

        if let Some(target) = board.get_piece(capture_x, capture_y) {
            if is_enemy_piece(target, piece.color()) {
                add_pawn_move(
                    out,
                    *from,
                    capture_x,
                    capture_y,
                    *piece,
                    &promotion_ranks,
                    promotion_pieces,
                );
            }
        } else if let Some(ep) = en_passant {
            if ep.square.x == capture_x && ep.square.y == capture_y {
                out.push(Move::new(
                    *from,
                    Coordinate::new(capture_x, capture_y),
                    *piece,
                ));
            }
        }
    }
}

/// Generate castling moves directly into an output buffer
#[inline]
fn generate_castling_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    special_rights: &FxHashSet<Coordinate>,
    indices: &SpatialIndices,
    out: &mut MoveList,
) {
    if !special_rights.contains(from) {
        return;
    }

    for coord in special_rights.iter() {
        if let Some(target_piece) = board.get_piece(coord.x, coord.y) {
            if target_piece.color() == piece.color()
                && target_piece.piece_type() != PieceType::Pawn
                && !target_piece.piece_type().is_royal()
            {
                let dx = coord.x - from.x;
                let dy = coord.y - from.y;

                if dy == 0 {
                    let dir = if dx > 0 { 1i64 } else { -1i64 };

                    // Use spatial indices to check path - O(log n) instead of O(distance)
                    let mut clear = true;
                    if let Some(row_pieces) = indices.rows.get(&from.y) {
                        if let Some((nearest_x, _)) =
                            SpatialIndices::find_nearest(row_pieces, from.x, dir)
                        {
                            if (dir > 0 && nearest_x < coord.x) || (dir < 0 && nearest_x > coord.x)
                            {
                                clear = false;
                            }
                        }
                    }

                    if clear {
                        let opponent = piece.color().opponent();
                        let pos_1 = Coordinate::new(from.x + dir, from.y);
                        let pos_2 = Coordinate::new(from.x + dir * 2, from.y);

                        if !is_square_attacked(board, from, opponent, indices)
                            && !is_square_attacked(board, &pos_1, opponent, indices)
                            && !is_square_attacked(board, &pos_2, opponent, indices)
                        {
                            let mut castling_move =
                                Move::new(*from, Coordinate::new(from.x + dir * 2, from.y), *piece);
                            castling_move.rook_coord = Some(*coord);
                            out.push(castling_move);
                        }
                    }
                }
            }
        }
    }
}

/// Generate sliding moves directly into an output buffer
#[inline]
pub fn generate_sliding_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    directions: &[(i64, i64)],
    indices: &SpatialIndices,
    fallback: bool,
    out: &mut MoveList,
    enemy_king_pos: Option<&Coordinate>,
) {
    // Reuse implementation by delegating to existing function and extending
    let moves = generate_sliding_moves(
        board,
        from,
        piece,
        directions,
        indices,
        fallback,
        enemy_king_pos,
    );
    out.extend(moves);
}

/// Generate knightrider moves directly into an output buffer
#[inline]
fn generate_knightrider_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    out: &mut MoveList,
) {
    let moves = generate_knightrider_moves(board, from, piece);
    out.extend(moves);
}

/// Generate huygen moves directly into an output buffer
#[inline]
fn generate_huygen_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    indices: &SpatialIndices,
    fallback: bool,
    out: &mut MoveList,
) {
    let moves = generate_huygen_moves(board, from, piece, indices, fallback);
    out.extend(moves);
}
