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
/// Rounding errors from BigInt -> i64 conversion on the JS side are acceptable.
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
        // If we have a blocker, generate *all* intermediate steps up to it.
        // If we have no blocker, only generate the first two consecutive steps.
        let max_steps: i64 = if closest_k < i64::MAX {
            if closest_is_enemy {
                closest_k
            } else {
                closest_k.saturating_sub(1)
            }
        } else {
            2
        };

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
}

impl SpatialIndices {
    pub fn new(board: &Board) -> Self {
        let mut rows: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut cols: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut diag1: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();
        let mut diag2: FxHashMap<i64, Vec<(i64, u8)>> = FxHashMap::default();

        // BITBOARD: Use tile-based CTZ iteration for O(popcount) enumeration
        for (cx, cy, tile) in board.tiles.iter() {
            let mut bits = tile.occ_all;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let packed = tile.piece[idx];
                if packed == 0 {
                    continue;
                }
                let lx = (idx % 8) as i64;
                let ly = (idx / 8) as i64;
                let x = cx * 8 + lx;
                let y = cy * 8 + ly;
                rows.entry(y).or_default().push((x, packed));
                cols.entry(x).or_default().push((y, packed));
                diag1.entry(x - y).or_default().push((x, packed));
                diag2.entry(x + y).or_default().push((x, packed));
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
            );
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
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
            );
            generate_sliding_moves_into(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                fallback,
                out,
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
    );
    out
}

pub fn is_square_attacked(
    board: &Board,
    target: &Coordinate,
    attacker_color: PlayerColor,
    indices: &SpatialIndices,
) -> bool {
    use crate::attacks::*;
    use crate::tiles::{local_index, masks};

    // OPTIMIZATION: Single-pass tile neighborhood check for all leapers+pawns
    let neighborhood = board.get_neighborhood(target.x, target.y);
    let local_idx = local_index(target.x, target.y);

    // Select attacker color bitboard accessor
    let is_white = attacker_color == PlayerColor::White;
    if attacker_color == PlayerColor::Neutral {
        return false; // Neutral can't attack
    }

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
        } // Fast skip empty tiles

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
                // Found potential attackers - verify piece type
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

    // 3. Check Sliding Pieces (Orthogonal and Diagonal) using SpatialIndices
    // Helper to check rays using indices for fast O(log n) nearest-piece lookup
    let check_ray = |dirs: &[(i64, i64)], type_mask: PieceTypeMask| -> bool {
        for &(dx, dy) in dirs {
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
                    if piece.color() == attacker_color
                        && matches_mask(piece.piece_type(), type_mask)
                    {
                        return true;
                    }
                }
            }
        }
        false
    };

    if check_ray(&ORTHO_DIRS, ORTHO_MASK) {
        return true;
    }
    if check_ray(&DIAG_DIRS, DIAG_MASK) {
        return true;
    }

    // 4. Check Knightrider (Sliding Knight)
    for &(dx, dy) in &KNIGHTRIDER_DIRS {
        let mut k = 1;
        loop {
            let x = target.x + dx * k;
            let y = target.y + dy * k;
            if let Some(piece) = board.get_piece(x, y) {
                if piece.color() == attacker_color && piece.piece_type() == PieceType::Knightrider {
                    return true;
                }
                break; // Blocked
            }
            k += 1;
            if k > 25 {
                break;
            } // Safety
        }
    }

    // 5. Check Huygen (Prime Leaper/Slider)
    // Orthogonal directions. Check all prime distances.
    for &(dx, dy) in &ORTHO_DIRS {
        let line_vec = if dx == 0 {
            indices.cols.get(&target.x)
        } else {
            indices.rows.get(&target.y)
        };
        if let Some(vec) = line_vec {
            // Iterate all pieces on this line - now includes piece data
            for &(coord, packed) in vec {
                let dist = if dx == 0 {
                    coord - target.y
                } else {
                    coord - target.x
                };
                let abs_dist = dist.abs();
                if abs_dist > 0 && is_prime_i64(abs_dist) {
                    // Check direction
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

    // 6. Check Rose (Circular Knight)
    let defender_color = attacker_color.opponent();
    let rose_moves =
        generate_rose_moves(board, target, &Piece::new(PieceType::Rose, defender_color));
    for m in rose_moves {
        if let Some(piece) = board.get_piece(m.to.x, m.to.y) {
            if piece.color() == attacker_color && piece.piece_type() == PieceType::Rose {
                return true;
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

    // Get promotion ranks for this color (default to 8 for white, 1 for black if not specified)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else {
        // Default promotion ranks for standard chess
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

    // Get promotion ranks for this color (default to 8 for white, 1 for black if not specified)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else {
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
                    let dir = if dx > 0 { 1 } else { -1 };

                    let mut clear = true;
                    let mut current_x = from.x + dir;
                    while current_x != coord.x {
                        if board.get_piece(current_x, from.y).is_some() {
                            clear = false;
                            break;
                        }
                        current_x += dir;
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
            let m = generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices, false);
            extend_quiets_only(board, m, out);
        }
        PieceType::Bishop => {
            let m = generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices, false);
            extend_quiets_only(board, m, out);
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            let mut m =
                generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices, false);
            m.extend(generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
            ));
            extend_quiets_only(board, m, out);
        }
        PieceType::Chancellor => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let rook_m =
                generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices, false);
            extend_quiets_only(board, rook_m, out);
        }
        PieceType::Archbishop => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let bishop_m =
                generate_sliding_moves(board, from, piece, &[(1, 1), (1, -1)], indices, false);
            extend_quiets_only(board, bishop_m, out);
        }
        PieceType::Amazon => {
            let knight_m = generate_leaper_moves(board, from, piece, 1, 2);
            extend_quiets_only(board, knight_m, out);
            let mut queen_m =
                generate_sliding_moves(board, from, piece, &[(1, 0), (0, 1)], indices, false);
            queen_m.extend(generate_sliding_moves(
                board,
                from,
                piece,
                &[(1, 1), (1, -1)],
                indices,
                false,
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

pub fn generate_sliding_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    directions: &[(i64, i64)],
    indices: &SpatialIndices,
    fallback: bool,
) -> MoveList {
    // Original wiggle values - important for tactics
    const ENEMY_WIGGLE: i64 = 2;
    const FRIEND_WIGGLE: i64 = 1;
    // Maximum distance for interception - 50 is needed for long-range tactics
    // We only cap "interception" moves (crossing an enemy's line), not direct captures!
    const MAX_INTERCEPTION_DIST: i64 = 50;

    // Fallback limit for short-range slider moves
    const FALLBACK_LIMIT: i64 = 10;

    let _piece_count = board.len();
    let mut moves = MoveList::new();
    let our_color = piece.color();

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

            // Process ALL pieces for interception (needed for tactics)
            for ((px, py), p) in board.iter() {
                // Obstacles are neutral but capturable - check is_uncapturable()
                let is_enemy = p.color() != our_color && !p.piece_type().is_uncapturable();
                let wiggle = if is_enemy {
                    ENEMY_WIGGLE
                } else {
                    FRIEND_WIGGLE
                };

                let pdx = *px - from.x;
                let pdy = *py - from.y;

                if is_horizontal {
                    // Check x coordinates
                    for w in -wiggle..=wiggle {
                        let tx = *px + w;
                        let dx = tx - from.x;
                        if dx != 0 && dx.signum() == dir_x.signum() {
                            let d = dx.abs();
                            // CRITICAL: specific check
                            // - If enemy is exactly on the ray (capture), use max_dist
                            // - If enemy is offset (interception), use interception_limit
                            let is_direct_capture = is_enemy && *py == from.y && w == 0;
                            let limit = if is_direct_capture {
                                max_dist
                            } else {
                                interception_limit
                            };

                            if d <= limit {
                                target_dists.push(d);
                            }
                        }
                    }
                } else if is_vertical {
                    // Check y coordinates
                    for w in -wiggle..=wiggle {
                        let ty = *py + w;
                        let dy = ty - from.y;
                        if dy != 0 && dy.signum() == dir_y.signum() {
                            let d = dy.abs();
                            // CRITICAL: specific check
                            let is_direct_capture = is_enemy && *px == from.x && w == 0;
                            let limit = if is_direct_capture {
                                max_dist
                            } else {
                                interception_limit
                            };

                            if d <= limit {
                                target_dists.push(d);
                            }
                        }
                    }
                } else {
                    // Diagonal movement
                    // Check if this piece is on the SAME ray we're moving along
                    let on_this_ray = pdx.abs() == pdy.abs()
                        && pdx != 0
                        && pdx.signum() == dir_x.signum()
                        && pdy.signum() == dir_y.signum();

                    if on_this_ray {
                        // Piece is on our ray - handled by blocker wiggle (which uses max_dist/closest_dist)
                        continue;
                    }

                    // Orthogonal interception
                    for w in -wiggle..=wiggle {
                        let tx = *px + w;
                        let dx = tx - from.x;
                        if dx != 0 && dx.signum() == dir_x.signum() {
                            let d = dx.abs();
                            if d <= interception_limit {
                                target_dists.push(d);
                            }
                        }

                        let ty = *py + w;
                        let dy = ty - from.y;
                        if dy != 0 && dy.signum() == dir_y.signum() {
                            let d = dy.abs();
                            if d <= interception_limit {
                                target_dists.push(d);
                            }
                        }
                    }

                    // Diagonal proximity
                    let diag_wiggle: i64 = 1;

                    if dir_x * dir_y > 0 {
                        let from_sum = from.x + from.y;
                        let piece_sum = *px + *py;
                        let diff = piece_sum - from_sum;
                        let base_d = if dir_x > 0 { diff / 2 } else { -diff / 2 };

                        for dw in -diag_wiggle..=diag_wiggle {
                            let d = base_d + dw;
                            if d > 0 && d <= interception_limit {
                                target_dists.push(d);
                            }
                        }
                    } else {
                        let from_diff = from.x - from.y;
                        let piece_diff = *px - *py;
                        let diff = piece_diff - from_diff;
                        let base_d = if dir_x > 0 { diff / 2 } else { -diff / 2 };

                        for dw in -diag_wiggle..=diag_wiggle {
                            let d = base_d + dw;
                            if d > 0 && d <= interception_limit {
                                target_dists.push(d);
                            }
                        }
                    }
                }
            }

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

/// Generate rose moves directly into an output buffer
#[inline]
fn generate_rose_moves_into(board: &Board, from: &Coordinate, piece: &Piece, out: &mut MoveList) {
    let knight_moves = [
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1),
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
    ];

    for (start_idx, _) in knight_moves.iter().enumerate() {
        for direction in [1, -1] {
            let mut current_x = from.x;
            let mut current_y = from.y;
            let mut current_idx = start_idx as i32;

            for _ in 0..7 {
                let idx = (current_idx as usize) % 8;
                let (dx, dy) = knight_moves[idx];

                current_x += dx;
                current_y += dy;

                if let Some(target) = board.get_piece(current_x, current_y) {
                    if is_enemy_piece(target, piece.color()) {
                        out.push(Move::new(
                            *from,
                            Coordinate::new(current_x, current_y),
                            *piece,
                        ));
                    }
                    break;
                } else {
                    out.push(Move::new(
                        *from,
                        Coordinate::new(current_x, current_y),
                        *piece,
                    ));
                }

                current_idx += direction;
                if current_idx < 0 {
                    current_idx += 8;
                }
            }
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
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else {
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

        // Double push
        if special_rights.contains(from) {
            let to_y_2 = from.y + (direction * 2);
            if board.get_piece(to_x, to_y_2).is_none() {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y_2), *piece));
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
                    let dir = if dx > 0 { 1 } else { -1 };
                    let mut clear = true;
                    let mut current_x = from.x + dir;
                    while current_x != coord.x {
                        if board.get_piece(current_x, from.y).is_some() {
                            clear = false;
                            break;
                        }
                        current_x += dir;
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
) {
    // Reuse implementation by delegating to existing function and extending
    let moves = generate_sliding_moves(board, from, piece, directions, indices, fallback);
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
