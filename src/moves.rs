use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::{EnPassantState, GameRules};
use crate::utils::{PRIMES_UNDER_128, is_prime_fast, is_prime_i64};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveGenType {
    All,
    Quiets,
    Captures,
}

/// Stack-allocated move list with inline capacity of 128 moves.
/// Spills to heap if this limit is exceeded, preventing panics.
pub type MoveList = Vec<Move>;

#[derive(Debug, Clone)]
pub struct MoveGenContext<'a> {
    pub special_rights: &'a FxHashSet<Coordinate>,
    pub en_passant: &'a Option<EnPassantState>,
    pub game_rules: &'a GameRules,
    pub indices: &'a SpatialIndices,
    pub enemy_king_pos: Option<&'a Coordinate>,
}

// World border for infinite chess. These are initialized to a very large box,
// but can be overridden from JS via the playableRegion values.
static mut COORD_MIN_X: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_X: i64 = 1_000_000_000_000_000; // default  1e15
static mut COORD_MIN_Y: i64 = -1_000_000_000_000_000; // default -1e15
static mut COORD_MAX_Y: i64 = 1_000_000_000_000_000; // default  1e15

struct CrossRayContext<'a> {
    board: &'a Board,
    from: &'a Coordinate,
    max_dist: i64,
    indices: &'a SpatialIndices,
    our_color: PlayerColor,
    piece_type: PieceType,
    enemy_wiggle: i64,
    friend_wiggle: i64,
}

pub struct SlidingMoveContext<'a> {
    pub board: &'a Board,
    pub from: &'a Coordinate,
    pub piece: &'a Piece,
    pub directions: &'a [(i64, i64)],
    pub indices: &'a SpatialIndices,
    pub enemy_king_pos: Option<&'a Coordinate>,
}

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

/// Check if a piece at `from` attacks square `to`.
/// Optimized for sliders and leapers; falls back to full movegen for complex fairy pieces.
pub fn is_piece_attacking_square(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    to: &Coordinate,
    indices: &SpatialIndices,
    game_rules: &GameRules,
) -> bool {
    use crate::attacks::{is_diag_slider, is_ortho_slider, is_slider};

    let pt = piece.piece_type();
    let our_color = piece.color();

    // 1. Sliders (optimized via spatial indices)
    if is_slider(pt) {
        let dx = to.x - from.x;
        let dy = to.y - from.y;

        let mut on_ray = false;
        let mut step_x = 0;
        let mut step_y = 0;

        if dx == 0 && dy != 0 && is_ortho_slider(pt) {
            on_ray = true;
            step_y = dy.signum();
        } else if dy == 0 && dx != 0 && is_ortho_slider(pt) {
            on_ray = true;
            step_x = dx.signum();
        } else if dx.abs() == dy.abs() && dx != 0 && is_diag_slider(pt) {
            on_ray = true;
            step_x = dx.signum();
            step_y = dy.signum();
        }

        if on_ray {
            let (closest_dist, _) =
                find_blocker_via_indices(board, from, step_x, step_y, indices, our_color);
            let target_dist = dx.abs().max(dy.abs());
            return target_dist <= closest_dist;
        }
    }

    // 2. Leapers
    match pt {
        PieceType::Knight => {
            let dx = (to.x - from.x).abs();
            let dy = (to.y - from.y).abs();
            return (dx == 1 && dy == 2) || (dx == 2 && dy == 1);
        }
        PieceType::Pawn => {
            let direction = if our_color == PlayerColor::White {
                1
            } else {
                -1
            };
            let dy = to.y - from.y;
            let dx = (to.x - from.x).abs();
            return dy == direction && dx == 1;
        }
        PieceType::King | PieceType::Guard => {
            let dx = (to.x - from.x).abs();
            let dy = (to.y - from.y).abs();
            return dx <= 1 && dy <= 1 && (dx != 0 || dy != 0);
        }
        _ => {}
    }

    // 3. Optimized Huygen check (prime-distance orthogonal slider)
    // Avoids fallback to move generation which has limits
    if pt == PieceType::Huygen {
        let dx = to.x - from.x;
        let dy = to.y - from.y;

        // Must be on same row or column (orthogonal)
        if dx != 0 && dy != 0 {
            return false;
        }

        // Must be different square
        if dx == 0 && dy == 0 {
            return false;
        }

        let dist = dx.abs().max(dy.abs());

        // Must be at prime distance
        if !is_prime_fast(dist) {
            return false;
        }

        // Check for blocker at closer prime distance using spatial indices
        let is_horizontal = dy == 0;
        let line_vec = if is_horizontal {
            indices.rows.get(&from.y)
        } else {
            indices.cols.get(&from.x)
        };

        let our_coord = if is_horizontal { from.x } else { from.y };
        let target_coord = if is_horizontal { to.x } else { to.y };
        let sign = (target_coord - our_coord).signum();

        if let Some(vec) = line_vec {
            // Check all pieces between Huygen and target for blockers at prime distances
            for &(coord, _packed) in vec {
                let d = (coord - our_coord) * sign; // Distance in direction of target
                if d <= 0 || d >= dist {
                    continue; // Not between Huygen and target
                }

                // If this piece is at a prime distance from the Huygen, it blocks
                if is_prime_fast(d) {
                    return false;
                }
            }
        }

        return true;
    }

    // 4. Fallback for complex fairy pieces (Rose, Knightrider, etc.)
    let mut moves = MoveList::new();
    let ctx = MoveGenContext {
        special_rights: &FxHashSet::default(),
        en_passant: &None,
        game_rules,
        indices,
        enemy_king_pos: None,
    };
    get_pseudo_legal_moves_for_piece_into(board, piece, from, &ctx, &mut moves);
    moves.iter().any(|m| m.to.x == to.x && m.to.y == to.y)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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
    ctx: &MoveGenContext,
    out: &mut MoveList,
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

            get_pseudo_legal_moves_for_piece_into(board, &piece, &from, ctx, out);
        }
    }
}

pub fn get_legal_moves(board: &Board, turn: PlayerColor, ctx: &MoveGenContext) -> MoveList {
    let mut moves = MoveList::new();
    get_legal_moves_into(board, turn, ctx, &mut moves);
    moves
}

/// Generate only capturing moves for quiescence search when the side to move is **not** in check.
/// This avoids generating and then filtering thousands of quiet moves.
pub fn get_quiescence_captures(
    board: &Board,
    turn: PlayerColor,
    ctx: &MoveGenContext,
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

            generate_captures_for_piece(board, &piece, &from, ctx, out);
        }
    }
}

// Helper to avoid duplicating the switch logic
fn generate_captures_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    ctx: &MoveGenContext,
    out: &mut MoveList,
) {
    let special_rights = ctx.special_rights;
    let en_passant = ctx.en_passant;
    let game_rules = ctx.game_rules;
    let indices = ctx.indices;
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
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Camel => {
            generate_leaper_moves_into(board, from, piece, 1, 3, MoveGenType::Captures, out);
        }
        PieceType::Giraffe => {
            generate_leaper_moves_into(board, from, piece, 1, 4, MoveGenType::Captures, out);
        }
        PieceType::Zebra => {
            generate_leaper_moves_into(board, from, piece, 2, 3, MoveGenType::Captures, out);
        }

        // King/Guard/Centaur/RoyalCentaur/Hawk: use compass moves, then filter captures
        PieceType::King | PieceType::Guard => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Captures, out);
        }
        PieceType::Centaur | PieceType::RoyalCentaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Captures, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Hawk => {
            generate_compass_moves_into(board, from, piece, 2, MoveGenType::Captures, out);
            generate_compass_moves_into(board, from, piece, 3, MoveGenType::Captures, out);
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
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Archbishop => {
            // Bishop + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }
        PieceType::Amazon => {
            // Queen + knight
            generate_sliding_capture_moves(board, from, piece, &[(1, 0), (0, 1)], indices, out);
            generate_sliding_capture_moves(board, from, piece, &[(1, 1), (1, -1)], indices, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Captures, out);
        }

        // Knightrider: sliding along knight vectors
        PieceType::Knightrider => {
            generate_knightrider_moves_into(board, from, piece, MoveGenType::Captures, out);
        }

        // Huygen: use existing generator and keep only captures
        PieceType::Huygen => {
            generate_huygen_moves_into(board, from, piece, indices, MoveGenType::Captures, out);
        }

        // Rose: use existing generator and keep only captures
        PieceType::Rose => {
            generate_rose_moves_into(board, from, piece, MoveGenType::Captures, out);
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
    ctx: &MoveGenContext,
    out: &mut MoveList,
) {
    let special_rights = ctx.special_rights;
    let en_passant = ctx.en_passant;
    let game_rules = ctx.game_rules;
    let indices = ctx.indices;
    let enemy_king_pos = ctx.enemy_king_pos;
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
        PieceType::Knight => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out)
        }
        PieceType::Hawk => {
            generate_compass_moves_into(board, from, piece, 2, MoveGenType::All, out);
            generate_compass_moves_into(board, from, piece, 3, MoveGenType::All, out);
        }
        PieceType::King => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::All, out);
            generate_castling_moves_into(board, from, piece, special_rights, indices, out);
        }
        PieceType::Guard => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::All, out)
        }
        PieceType::Rook => {
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Bishop => {
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Chancellor => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out);
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Archbishop => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out);
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Amazon => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out);
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
            generate_sliding_moves_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Camel => {
            generate_leaper_moves_into(board, from, piece, 1, 3, MoveGenType::All, out)
        }
        PieceType::Giraffe => {
            generate_leaper_moves_into(board, from, piece, 1, 4, MoveGenType::All, out)
        }
        PieceType::Zebra => {
            generate_leaper_moves_into(board, from, piece, 2, 3, MoveGenType::All, out)
        }
        // Knightrider: slide along all 8 knight directions until blocked
        PieceType::Knightrider => {
            generate_knightrider_moves_into(board, from, piece, MoveGenType::All, out)
        }
        PieceType::Centaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::All, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out);
        }
        PieceType::RoyalCentaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::All, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::All, out);
            generate_castling_moves_into(board, from, piece, special_rights, indices, out);
        }
        PieceType::Huygen => {
            generate_huygen_moves_into(board, from, piece, indices, MoveGenType::All, out)
        }
        PieceType::Rose => generate_rose_moves_into(board, from, piece, MoveGenType::All, out),
    }
}

/// Legacy wrapper that allocates a new Vec. Prefer `get_pseudo_legal_moves_for_piece_into` for performance.
pub fn get_pseudo_legal_moves_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    ctx: &MoveGenContext,
) -> MoveList {
    let mut out = MoveList::new();
    get_pseudo_legal_moves_for_piece_into(board, piece, from, ctx, &mut out);
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
    // Combined mask of ALL leaper types that attack via nearby tiles
    const ALL_LEAPER_MASK: u32 = KNIGHT_MASK
        | KING_MASK
        | CAMEL_MASK
        | GIRAFFE_MASK
        | ZEBRA_MASK
        | HAWK_MASK
        | (1u32 << (PieceType::Pawn as u8));

    for n in 0..9 {
        let Some(tile) = neighborhood[n] else {
            continue;
        };

        // Get attacker occupancy and type mask for this tile
        let (occ, type_mask) = if is_white {
            (tile.occ_white, tile.type_mask_white)
        } else {
            (tile.occ_black, tile.type_mask_black)
        };

        // Fast early-exit: no attackers of any leaper type in this tile
        if occ == 0 || (type_mask & ALL_LEAPER_MASK) == 0 {
            continue;
        }

        // Check each leaper type - only if tile has that type
        let masks_to_check = [
            (masks::KNIGHT_MASKS[local_idx][n], KNIGHT_MASK),
            (masks::KING_MASKS[local_idx][n], KING_MASK),
            (masks::CAMEL_MASKS[local_idx][n], CAMEL_MASK),
            (masks::GIRAFFE_MASKS[local_idx][n], GIRAFFE_MASK),
            (masks::ZEBRA_MASKS[local_idx][n], ZEBRA_MASK),
            (masks::HAWK_MASKS[local_idx][n], HAWK_MASK),
            (pawn_masks[local_idx][n], pawn_type_mask),
        ];

        for (attack_mask, req_type_mask) in masks_to_check {
            // Skip if tile has no pieces of this type (fast O(1) check)
            if (type_mask & req_type_mask) == 0 {
                continue;
            }

            let candidates = occ & attack_mask;
            if candidates != 0 {
                let mut bits = candidates;
                while bits != 0 {
                    let bit_idx = bits.trailing_zeros() as usize;
                    bits &= bits - 1;

                    let packed = tile.piece[bit_idx];
                    if packed != 0 {
                        let pt = Piece::from_packed(packed).piece_type();
                        if matches_mask(pt, req_type_mask) {
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
    // CRITICAL: Blocking is from the HUYGENS's perspective, not the target's!
    // A Huygens at prime distance D attacks the target ONLY if there is no other piece
    // at any prime distance from the HUYGENS that is closer than D (between Huygens and target).
    if indices.has_huygen[attacker_idx] {
        // Check each orthogonal direction from the target to find Huygens
        for &(dx, dy) in &ORTHO_DIRS {
            let line_vec = if dx == 0 {
                indices.cols.get(&target.x)
            } else {
                indices.rows.get(&target.y)
            };
            if let Some(vec) = line_vec {
                // First pass: find any Huygens of attacker color in this direction
                for &(coord, packed) in vec {
                    let piece = Piece::from_packed(packed);
                    if piece.piece_type() != PieceType::Huygen || piece.color() != attacker_color {
                        continue;
                    }

                    // Calculate distance from target to this Huygens
                    let dist_to_target = if dx == 0 {
                        coord - target.y
                    } else {
                        coord - target.x
                    };

                    // Check direction: the Huygens must be in the direction we're checking
                    let in_right_direction = if dx == 0 {
                        (dy > 0 && dist_to_target > 0) || (dy < 0 && dist_to_target < 0)
                    } else {
                        (dx > 0 && dist_to_target > 0) || (dx < 0 && dist_to_target < 0)
                    };

                    if !in_right_direction {
                        continue;
                    }

                    let abs_dist_to_target = dist_to_target.abs();

                    // Target must be at a prime distance from the Huygens
                    // Use is_prime_i64 for arbitrary distances (handles extreme coordinates)
                    if !is_prime_i64(abs_dist_to_target) {
                        continue;
                    }

                    // Now check if any piece blocks at a CLOSER prime distance FROM THE HUYGENS
                    // The Huygens is at `coord`, target is at distance `abs_dist_to_target`
                    // We need to check all primes < abs_dist_to_target for blocking pieces
                    let huygen_coord = coord;
                    let mut blocked = false;

                    // Check all pieces in the line between Huygens and target
                    for &(other_coord, _other_packed) in vec {
                        // Calculate distance from HUYGENS to this piece
                        let dist_from_huygen = other_coord - huygen_coord;

                        // Piece must be between Huygens and target (in the direction toward target, closer distance)
                        // dist_to_target = huygen_coord - target_coord:
                        //   - If dist_to_target > 0: Huygens is at HIGHER coord than target, so blockers are TOWARD target (negative dist_from_huygen)
                        //   - If dist_to_target < 0: Huygens is at LOWER coord than target, so blockers are TOWARD target (positive dist_from_huygen)
                        let toward_target = if dist_to_target > 0 {
                            // Huygens at higher coord, target at lower coord -> blockers have negative dist (toward target)
                            dist_from_huygen < 0 && dist_from_huygen.abs() < abs_dist_to_target
                        } else {
                            // Huygens at lower coord, target at higher coord -> blockers have positive dist (toward target)
                            dist_from_huygen > 0 && dist_from_huygen < abs_dist_to_target
                        };

                        if !toward_target {
                            continue;
                        }

                        let abs_dist_from_huygen = dist_from_huygen.abs();
                        // If this piece is at a prime distance from the Huygens, it blocks!
                        // Use is_prime_i64 for arbitrary distances
                        if is_prime_i64(abs_dist_from_huygen) {
                            blocked = true;
                            break;
                        }
                    }

                    if !blocked {
                        return true; // Huygens attacks the target!
                    }
                }
            }
        }
    }

    // Rose check - O(1) early exit if no Roses exist
    // For attack detection, we need to find any Rose that can reach target via an unblocked spiral.
    // We check all positions that could host a Rose and verify if any spiral reaches target unblocked.
    if indices.has_rose[attacker_idx] {
        // Check every possible Rose position: positions on any spiral endpoint from target
        // A Rose at position P can attack target T if T is on one of P's spirals, unblocked.
        // Equivalently: there exists a spiral from some P that reaches T.

        // Iterate over all spiral endpoints from target (reverse direction)
        for spiral_dirs in &ROSE_SPIRALS {
            for spiral in spiral_dirs {
                // Check each position along this spiral from target
                for hop in 0..7 {
                    let (cum_dx, cum_dy) = spiral[hop];
                    // This is where a Rose would need to be to reach target at hop=hop
                    let rose_x = target.x - cum_dx;
                    let rose_y = target.y - cum_dy;
                    if board.get_piece(rose_x, rose_y).is_some_and(|p| {
                        p.color() == attacker_color && p.piece_type() == PieceType::Rose
                    }) {
                        // Found a Rose! Check if path to target is unblocked
                        let mut blocked = false;
                        for &(prev_dx, prev_dy) in spiral.iter().take(hop) {
                            let check_x = target.x - cum_dx + prev_dx;
                            let check_y = target.y - cum_dy + prev_dy;
                            if board.get_piece(check_x, check_y).is_some() {
                                blocked = true;
                                break;
                            }
                        }
                        if !blocked {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
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
    let promotion_ranks: Vec<i64> = if let Some(ranks) = game_rules.promotion_ranks.as_ref() {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else if game_rules
        .promotions_allowed
        .as_ref()
        .is_none_or(|v| v.is_empty())
    {
        // No promotion_ranks AND no promotions_allowed = no promotions anywhere
        vec![]
    } else {
        // promotions_allowed is set but no ranks = use classical defaults
        match piece.color() {
            PlayerColor::White => vec![],
            PlayerColor::Black => vec![],
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
        .as_deref()
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
            if is_enemy_piece(target, piece.color()) {
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
                        promotion_pieces,
                    );
                }
            }
        } else if en_passant
            .as_ref()
            .is_some_and(|ep| ep.square.x == capture_x && ep.square.y == capture_y)
        {
            out.push(Move::new(
                *from,
                Coordinate::new(capture_x, capture_y),
                *piece,
            ));
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
                                let mut castling_move =
                                    Move::new(*from, Coordinate::new(to_x, from.y), *piece);
                                castling_move.rook_coord = Some(*coord);
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
/// Uses O(log n) SpatialIndices for infinite-range blocker detection.
fn generate_sliding_capture_moves(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    directions: &[(i64, i64)],
    indices: &SpatialIndices,
    out: &mut MoveList,
) {
    let our_color = piece.color();

    for &(dx_raw, dy_raw) in directions {
        for sign in [1i64, -1i64] {
            let dx = dx_raw * sign;
            let dy = dy_raw * sign;
            if dx == 0 && dy == 0 {
                continue;
            }

            // O(log n) blocker lookup - handles infinite distance
            let (closest_dist, closest_is_enemy) =
                find_blocker_via_indices(board, from, dx, dy, indices, our_color);

            // Only add capture if blocker is an enemy piece
            if closest_dist < i64::MAX && closest_is_enemy {
                let x = from.x + dx * closest_dist;
                let y = from.y + dy * closest_dist;
                out.push(Move::new(*from, Coordinate::new(x, y), *piece));
            }
        }
    }
}

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

/// Generate only quiet (non-capturing) moves for staged move generation.
/// This is the complement of get_quiescence_captures.
pub fn get_quiet_moves_into(
    board: &Board,
    turn: PlayerColor,
    ctx: &MoveGenContext,
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
        generate_quiets_for_piece(board, &piece, &from, ctx, out);
    }
}

/// Generate only quiet moves for a single piece.
fn generate_quiets_for_piece(
    board: &Board,
    piece: &Piece,
    from: &Coordinate,
    ctx: &MoveGenContext,
    out: &mut MoveList,
) {
    let special_rights = ctx.special_rights;
    let game_rules = ctx.game_rules;
    let indices = ctx.indices;
    let enemy_king_pos = ctx.enemy_king_pos;
    match piece.piece_type() {
        PieceType::Void | PieceType::Obstacle => {}

        // Pawns: only forward moves (single and double push), no captures
        PieceType::Pawn => {
            generate_pawn_quiet_moves(board, from, piece, special_rights, game_rules, out);
        }

        // Knight-like leapers: filter to empty squares
        PieceType::Knight => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
        }
        PieceType::Camel => {
            generate_leaper_moves_into(board, from, piece, 1, 3, MoveGenType::Quiets, out);
        }
        PieceType::Giraffe => {
            generate_leaper_moves_into(board, from, piece, 1, 4, MoveGenType::Quiets, out);
        }
        PieceType::Zebra => {
            generate_leaper_moves_into(board, from, piece, 2, 3, MoveGenType::Quiets, out);
        }

        // King: compass + castling
        PieceType::King => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Quiets, out);
            // Castling is always a quiet move
            let castling = generate_castling_moves(board, from, piece, special_rights, indices);
            out.extend(castling);
        }
        PieceType::Guard => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Quiets, out);
        }
        PieceType::Centaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Quiets, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
        }
        PieceType::RoyalCentaur => {
            generate_compass_moves_into(board, from, piece, 1, MoveGenType::Quiets, out);
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
            let castling = generate_castling_moves(board, from, piece, special_rights, indices);
            out.extend(castling);
        }
        PieceType::Hawk => {
            generate_compass_moves_into(board, from, piece, 2, MoveGenType::Quiets, out);
            generate_compass_moves_into(board, from, piece, 3, MoveGenType::Quiets, out);
        }

        // Sliders
        PieceType::Rook => {
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Bishop => {
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Queen | PieceType::RoyalQueen => {
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Chancellor => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Archbishop => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }
        PieceType::Amazon => {
            generate_leaper_moves_into(board, from, piece, 1, 2, MoveGenType::Quiets, out);
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 0), (0, 1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
            generate_sliding_quiets_into(
                &SlidingMoveContext {
                    board,
                    from,
                    piece,
                    directions: &[(1, 1), (1, -1)],
                    indices,
                    enemy_king_pos,
                },
                out,
            );
        }

        PieceType::Knightrider => {
            generate_knightrider_moves_into(board, from, piece, MoveGenType::Quiets, out);
        }
        PieceType::Huygen => {
            generate_huygen_moves_into(board, from, piece, indices, MoveGenType::Quiets, out);
        }
        PieceType::Rose => {
            generate_rose_moves_into(board, from, piece, MoveGenType::Quiets, out);
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
            .unwrap_or_default(),
        PlayerColor::Black => game_rules
            .promotion_ranks
            .as_ref()
            .map(|p| p.black.clone())
            .unwrap_or_default(),
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
        .as_deref()
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

    // Single push
    let to_y = from.y + direction;
    let to_x = from.x;

    if board.get_piece(to_x, to_y).is_none() {
        // Square is empty, can push
        add_pawn_move(
            out,
            *from,
            to_x,
            to_y,
            *piece,
            &promotion_ranks,
            promotion_pieces,
        );

        // Double push if pawn has special rights
        if special_rights.contains(from) {
            let double_y = from.y + 2 * direction;
            if board.get_piece(to_x, double_y).is_none() {
                add_pawn_move(
                    out,
                    *from,
                    to_x,
                    double_y,
                    *piece,
                    &promotion_ranks,
                    promotion_pieces,
                );
            }
        }
    }
}

/// Generate leaper moves directly into an output buffer
/// gen_type controls which move types to generate: All, Quiets only, or Captures only
#[inline]
fn generate_leaper_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    m: i64,
    n: i64,
    gen_type: MoveGenType,
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
            // Target square occupied - this would be a capture
            let dominated = is_enemy_piece(target, piece.color());
            if dominated && gen_type != MoveGenType::Quiets {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
        } else {
            // Empty square - quiet move
            if gen_type != MoveGenType::Captures {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
        }
    }
}

/// Generate compass moves directly into an output buffer
/// gen_type controls which move types to generate: All, Quiets only, or Captures only
#[inline]
fn generate_compass_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    distance: i64,
    gen_type: MoveGenType,
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
            // Target square occupied - this would be a capture
            let dominated = is_enemy_piece(target, piece.color());
            if dominated && gen_type != MoveGenType::Quiets {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
        } else {
            // Empty square - quiet move
            if gen_type != MoveGenType::Captures {
                out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
            }
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

/// Find cross-ray attack targets for sliders - optimized for infinite chess.
///
/// Smart Filtering Approach:
/// 1. Iterate over ALL pieces on the board (O(N), usually < 60).
/// 2. For each piece P, calculate if any of its 8 attack rays intersect our slider's ray.
/// 3. Count how many pieces are reachable from each intersection distance.
/// 4. Only output distances where:
///    - 2+ pieces are reachable (forks/pins/batteries - unlimited distance)
///    - OR within SHORT_CROSS_RAY_LIMIT (single targets)
#[inline]
fn find_cross_ray_targets_into(
    ctx: &CrossRayContext,
    dir_x: i64,
    dir_y: i64,
    dist_counts: &mut FxHashMap<i64, u8>,
    royal_dists: &mut FxHashSet<i64>,
) {
    let board = ctx.board;
    let from = ctx.from;
    let max_dist = ctx.max_dist;
    let indices = ctx.indices;
    let our_color = ctx.our_color;
    let piece_type = ctx.piece_type;
    let enemy_wiggle = ctx.enemy_wiggle;
    let friend_wiggle = ctx.friend_wiggle;

    // Check OUR piece's attack capabilities
    let our_attacks_ortho = matches!(
        piece_type,
        PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Rook
            | PieceType::Chancellor
            | PieceType::Amazon
    );
    let our_attacks_diag = matches!(
        piece_type,
        PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Bishop
            | PieceType::Archbishop
            | PieceType::Amazon
    );

    // If our piece can't attack in any direction, no cross-ray targets
    if !our_attacks_ortho && !our_attacks_diag {
        return;
    }

    // Precompute constant ray properties
    let ray_diff = dir_x - dir_y;
    let ray_sum = dir_x + dir_y;

    // Helper to increment piece count for a distance
    #[inline(always)]
    fn add_dist(map: &mut FxHashMap<i64, u8>, d: i64, max_d: i64) {
        if d > 0 && d <= max_d {
            let entry = map.entry(d).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }

    // Iterate all pieces on the board once - count pieces reachable from each distance
    for (px, py, p) in board.tiles.iter_all_pieces() {
        // Skip only the piece at our exact position (can't target ourselves)
        if px == from.x && py == from.y {
            continue;
        }

        let is_enemy = p.color() != our_color && !p.piece_type().is_uncapturable();

        let wiggle = if is_enemy {
            enemy_wiggle
        } else {
            friend_wiggle
        };
        let is_royal = is_enemy && p.piece_type().is_royal();

        // 1. Orthogonal Cross-Rays (if OUR piece can attack orthogonally)
        if our_attacks_ortho {
            // Vertical cross: S.x = px
            if dir_x != 0 {
                let num = px - from.x;
                if num.signum() == dir_x.signum() && num % dir_x == 0 {
                    let d = num / dir_x;
                    if d > 0 && d <= max_dist {
                        let sy = from.y + d * dir_y;
                        if py != sy
                            && let Some((_nearest_y, _)) = indices
                                .cols
                                .get(&px)
                                .and_then(|pieces| {
                                    SpatialIndices::find_nearest(pieces, sy, (py - sy).signum())
                                })
                                .filter(|&(ny, _)| ny == py)
                        {
                            // Count this piece at distance d and wiggle distances
                            add_dist(dist_counts, d, max_dist);
                            if is_royal {
                                royal_dists.insert(d);
                            }

                            for w in 1..=wiggle {
                                add_dist(dist_counts, d + w, max_dist);
                                add_dist(dist_counts, d - w, max_dist);
                                if is_royal {
                                    royal_dists.insert(d + w);
                                    royal_dists.insert(d - w);
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
                        if px != sx
                            && let Some((_nearest_x, _)) = indices
                                .rows
                                .get(&py)
                                .and_then(|pieces| {
                                    SpatialIndices::find_nearest(pieces, sx, (px - sx).signum())
                                })
                                .filter(|&(nx, _)| nx == px)
                        {
                            add_dist(dist_counts, d, max_dist);
                            if is_royal {
                                royal_dists.insert(d);
                            }

                            for w in 1..=wiggle {
                                add_dist(dist_counts, d + w, max_dist);
                                add_dist(dist_counts, d - w, max_dist);
                                if is_royal {
                                    royal_dists.insert(d + w);
                                    royal_dists.insert(d - w);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Diagonal Cross-Rays (if OUR piece can attack diagonally)
        if our_attacks_diag {
            // Diagonal 1: x-y constant. S.x - S.y = px - py
            // (from.x + d*dir_x) - (from.y + d*dir_y) = px - py
            // d*(dir_x - dir_y) = (px - py) - (from.x - from.y)
            if ray_diff != 0 {
                let num = (px - py) - (from.x - from.y);
                if num.signum() == ray_diff.signum() && num % ray_diff == 0 {
                    let d = num / ray_diff;
                    if d > 0 && d <= max_dist {
                        let sx = from.x + d * dir_x;
                        let sy = from.y + d * dir_y;
                        let s_diag_diff = sx - sy;

                        if sx != px
                            && let Some((_nearest_x, _)) = indices
                                .diag1
                                .get(&s_diag_diff)
                                .and_then(|pieces| {
                                    SpatialIndices::find_nearest(pieces, sx, (px - sx).signum())
                                })
                                .filter(|&(nx, _)| nx == px)
                        {
                            add_dist(dist_counts, d, max_dist);
                            if is_royal {
                                royal_dists.insert(d);
                            }

                            for w in 1..=wiggle {
                                add_dist(dist_counts, d + w, max_dist);
                                add_dist(dist_counts, d - w, max_dist);
                                if is_royal {
                                    royal_dists.insert(d + w);
                                    royal_dists.insert(d - w);
                                }
                            }
                        }
                    }
                }
            }

            // Diagonal 2: x+y constant. S.x + S.y = px + py
            // (from.x + d*dir_x) + (from.y + d*dir_y) = px + py
            // d*(dir_x + dir_y) = (px + py) - (from.x + from.y)
            if ray_sum != 0 {
                let num = (px + py) - (from.x + from.y);
                if num.signum() == ray_sum.signum() && num % ray_sum == 0 {
                    let d = num / ray_sum;
                    if d > 0 && d <= max_dist {
                        let sx = from.x + d * dir_x;
                        let sy = from.y + d * dir_y;
                        let s_diag_sum = sx + sy;

                        if sx != px
                            && let Some((_nearest_x, _)) = indices
                                .diag2
                                .get(&s_diag_sum)
                                .and_then(|pieces| {
                                    SpatialIndices::find_nearest(pieces, sx, (px - sx).signum())
                                })
                                .filter(|&(nx, _)| nx == px)
                        {
                            add_dist(dist_counts, d, max_dist);
                            if is_royal {
                                royal_dists.insert(d);
                            }

                            for w in 1..=wiggle {
                                add_dist(dist_counts, d + w, max_dist);
                                add_dist(dist_counts, d - w, max_dist);
                                if is_royal {
                                    royal_dists.insert(d + w);
                                    royal_dists.insert(d - w);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn generate_sliding_moves_impl(
    ctx: &SlidingMoveContext,
    out: &mut MoveList,
    gen_type: MoveGenType,
) {
    let board = ctx.board;
    let from = ctx.from;
    let piece = ctx.piece;
    let directions = ctx.directions;
    let indices = ctx.indices;
    let enemy_king_pos = ctx.enemy_king_pos;

    // Original wiggle values - important for tactics
    const ENEMY_WIGGLE: i64 = 2;
    const FRIEND_WIGGLE: i64 = 1;
    // Unified base limit for short range
    const BASE_INTERCEPTION_DIST: i64 = 16;

    let our_color = piece.color();

    // Royal pieces: ALWAYS full wiggle (for safety/mate)
    #[inline(always)]
    fn distance_wiggle(dist: i64, is_enemy: bool, base_wiggle: i64, is_royal: bool) -> i64 {
        if is_royal || dist <= 10 {
            base_wiggle
        } else if is_enemy {
            1
        } else {
            0
        }
    }

    let ek_ref = enemy_king_pos;

    // Reuse maps across directions to avoid allocations
    let mut dist_counts: FxHashMap<i64, u8> = FxHashMap::default();
    let mut royal_dists: FxHashSet<i64> = FxHashSet::default();

    // Helper to increment piece count for a distance
    #[inline(always)]
    fn add_dist(map: &mut FxHashMap<i64, u8>, d: i64, max_d: i64) {
        if d > 0 && d <= max_d {
            let entry = map.entry(d).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }

    for &(dx_raw, dy_raw) in directions {
        for sign in [1i64, -1i64] {
            let dir_x = dx_raw * sign;
            let dir_y = dy_raw * sign;

            if dir_x == 0 && dir_y == 0 {
                continue;
            }

            let is_vertical = dir_x == 0;
            let is_horizontal = dir_y == 0;

            // Use spatial indices for O(log n) blocker finding
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

            // Direction encoding for cache: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
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
            let mut target_dists: Vec<i64> = Vec::with_capacity(64);

            if let Some(cached_dists) = cached {
                target_dists.extend(cached_dists);
            } else {
                dist_counts.clear();
                royal_dists.clear();

                // 1. Direct Ray iteration (O(pieces_on_line))
                if is_horizontal {
                    if let Some(pieces_on_row) = indices.rows.get(&from.y) {
                        for &(px, packed) in pieces_on_row {
                            let dx = px - from.x;
                            if dx == 0 || dx.signum() != dir_x.signum() {
                                continue;
                            }
                            let piece_dist = dx.abs();
                            if piece_dist > max_dist && piece_dist != closest_dist {
                                continue;
                            }

                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let is_target_royal = p.piece_type().is_royal();

                            // Friendly pieces only act as "targets" (interceptions) within short range
                            if !is_enemy && !is_target_royal && piece_dist > BASE_INTERCEPTION_DIST
                            {
                                continue;
                            }

                            let base_wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };
                            let is_our_royal = piece.piece_type().is_royal();
                            let wiggle = distance_wiggle(
                                piece_dist,
                                is_enemy,
                                base_wiggle,
                                is_our_royal || is_target_royal,
                            );

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                add_dist(&mut dist_counts, d, max_dist);
                                if is_target_royal {
                                    royal_dists.insert(d);
                                }
                            }
                        }
                    }
                } else if is_vertical {
                    if let Some(pieces_on_col) = indices.cols.get(&from.x) {
                        for &(py, packed) in pieces_on_col {
                            let dy = py - from.y;
                            if dy == 0 || dy.signum() != dir_y.signum() {
                                continue;
                            }
                            let piece_dist = dy.abs();
                            if piece_dist > max_dist && piece_dist != closest_dist {
                                continue;
                            }

                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let is_target_royal = p.piece_type().is_royal();

                            if !is_enemy && !is_target_royal && piece_dist > BASE_INTERCEPTION_DIST
                            {
                                continue;
                            }

                            let base_wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };
                            let is_our_royal = piece.piece_type().is_royal();
                            let wiggle = distance_wiggle(
                                piece_dist,
                                is_enemy,
                                base_wiggle,
                                is_our_royal || is_target_royal,
                            );

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                add_dist(&mut dist_counts, d, max_dist);
                                if is_target_royal {
                                    royal_dists.insert(d);
                                }
                            }
                        }
                    }
                } else {
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
                            if piece_dist > max_dist && piece_dist != closest_dist {
                                continue;
                            }

                            let p = Piece::from_packed(packed);
                            let is_enemy =
                                p.color() != our_color && !p.piece_type().is_uncapturable();
                            let is_target_royal = p.piece_type().is_royal();

                            if !is_enemy && !is_target_royal && piece_dist > BASE_INTERCEPTION_DIST
                            {
                                continue;
                            }

                            let base_wiggle = if is_enemy {
                                ENEMY_WIGGLE
                            } else {
                                FRIEND_WIGGLE
                            };
                            let is_our_royal = piece.piece_type().is_royal();
                            let wiggle = distance_wiggle(
                                piece_dist,
                                is_enemy,
                                base_wiggle,
                                is_our_royal || is_target_royal,
                            );

                            for w in -wiggle..=wiggle {
                                let d = piece_dist + w;
                                add_dist(&mut dist_counts, d, max_dist);
                                if is_target_royal {
                                    royal_dists.insert(d);
                                }
                            }
                        }
                    }
                }

                // 2. Cross-Ray pieces
                let cr_ctx = CrossRayContext {
                    board,
                    from,
                    max_dist,
                    indices,
                    our_color,
                    piece_type: piece.piece_type(),
                    enemy_wiggle: ENEMY_WIGGLE,
                    friend_wiggle: FRIEND_WIGGLE,
                };
                find_cross_ray_targets_into(
                    &cr_ctx,
                    dir_x,
                    dir_y,
                    &mut dist_counts,
                    &mut royal_dists,
                );

                // 3. Check targets (O(1))
                if let Some(ek) = ek_ref {
                    let kx = ek.x;
                    let ky = ek.y;
                    let pt = piece.piece_type();
                    let can_ortho = matches!(
                        pt,
                        PieceType::Queen
                            | PieceType::Rook
                            | PieceType::RoyalQueen
                            | PieceType::Chancellor
                            | PieceType::Amazon
                    );
                    let can_diag = matches!(
                        pt,
                        PieceType::Queen
                            | PieceType::Bishop
                            | PieceType::RoyalQueen
                            | PieceType::Archbishop
                            | PieceType::Amazon
                    );

                    if is_horizontal {
                        if can_ortho && kx != from.x && (kx - from.x).signum() == dir_x.signum() {
                            let d = (kx - from.x).abs();
                            add_dist(&mut dist_counts, d, max_dist);
                            royal_dists.insert(d);
                        }
                        if can_diag && from.y != ky {
                            let diff = (from.y - ky).abs();
                            for tx in [kx + diff, kx - diff] {
                                if tx != from.x && (tx - from.x).signum() == dir_x.signum() {
                                    let d = (tx - from.x).abs();
                                    add_dist(&mut dist_counts, d, max_dist);
                                    royal_dists.insert(d);
                                }
                            }
                        }
                    } else if is_vertical {
                        if can_ortho && ky != from.y && (ky - from.y).signum() == dir_y.signum() {
                            let d = (ky - from.y).abs();
                            add_dist(&mut dist_counts, d, max_dist);
                            royal_dists.insert(d);
                        }
                        if can_diag && from.x != kx {
                            let diff = (from.x - kx).abs();
                            for ty in [ky + diff, ky - diff] {
                                if ty != from.y && (ty - from.y).signum() == dir_y.signum() {
                                    let d = (ty - from.y).abs();
                                    add_dist(&mut dist_counts, d, max_dist);
                                    royal_dists.insert(d);
                                }
                            }
                        }
                    }
                }

                // 4. Final Filtering & Cache Storage
                let mut shared_targets = Vec::with_capacity(dist_counts.len());
                // Always add short-range wiggle room targets (up to max_dist)
                for d in 1..=ENEMY_WIGGLE {
                    if d <= max_dist {
                        shared_targets.push(d);
                    }
                }
                // Also always add direct capture distance if it's an enemy
                if closest_dist < i64::MAX && closest_is_enemy {
                    shared_targets.push(closest_dist);
                }

                for (&d, &count) in &dist_counts {
                    if d <= BASE_INTERCEPTION_DIST || count >= 2 || royal_dists.contains(&d) {
                        shared_targets.push(d);
                    }
                }
                shared_targets.sort_unstable();
                shared_targets.dedup();

                target_dists.extend(shared_targets.clone());
                indices
                    .slider_cache
                    .borrow_mut()
                    .insert(cache_key, shared_targets);
            }

            // Generate moves from target_dists
            for d in target_dists {
                if d <= 0 || d > max_dist {
                    continue;
                }
                // Skip friendly blocker
                if d == closest_dist && !closest_is_enemy {
                    continue;
                }

                let is_capture = d == closest_dist && closest_is_enemy;
                if (gen_type == MoveGenType::Captures && !is_capture)
                    || (gen_type == MoveGenType::Quiets && is_capture)
                {
                    continue;
                }

                let sq_x = from.x + dir_x * d;
                let sq_y = from.y + dir_y * d;
                if in_bounds(sq_x, sq_y) {
                    out.push(Move::new(*from, Coordinate::new(sq_x, sq_y), *piece));
                }
            }
        }
    }
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

/// OPTIMIZED Huygens move generation using precomputed primes and spatial indices.
///
/// Key optimizations:
/// 1. Uses PRIMES_UNDER_128 array for direct iteration instead of primality testing
/// 2. O(log n) binary search in spatial indices for blocker detection
/// 3. When no blocker, only generates moves to "interesting" squares aligned with cross-ray pieces
/// 4. is_prime_fast() for O(1) primality checks instead of O(√n)
/// gen_type controls which move types to generate: All, Quiets only, or Captures only
fn generate_huygen_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    indices: &SpatialIndices,
    gen_type: MoveGenType,
    out: &mut MoveList,
) {
    let my_color = piece.color();

    // Four orthogonal directions: right, left, up, down
    const ORTHO_DIRECTIONS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    // Limit for moves when no blocker is found (use cross-ray logic beyond this)
    const OPEN_RAY_LIMIT: i64 = 50;

    for &(dir_x, dir_y) in &ORTHO_DIRECTIONS {
        // Find the closest blocker at a prime distance in this direction
        let (blocker_dist, blocker_color) =
            find_huygen_blocker(board, from, dir_x, dir_y, indices, my_color);

        if blocker_dist < i64::MAX {
            // CASE 1: Blocker found at prime distance
            // Generate all prime-distance moves up to (and including if capturable) the blocker
            for &prime_dist in &PRIMES_UNDER_128 {
                if prime_dist > blocker_dist {
                    break;
                }

                let to_x = from.x + dir_x * prime_dist;
                let to_y = from.y + dir_y * prime_dist;

                if prime_dist == blocker_dist {
                    // At blocker - can only move here if enemy (capture)
                    if let Some(color) = blocker_color {
                        if color != my_color && gen_type != MoveGenType::Quiets {
                            out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                        }
                    }
                } else {
                    // Before blocker - empty square, valid move
                    if gen_type != MoveGenType::Captures {
                        out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                    }
                }
            }

            // IMPORTANT: Handle captures at prime distances > 127
            // The loop above only covers primes up to 127, but blocker could be further
            if blocker_dist > 127 && gen_type != MoveGenType::Quiets {
                if let Some(color) = blocker_color {
                    if color != my_color {
                        // Blocker is enemy at prime distance > 127 - generate capture
                        let to_x = from.x + dir_x * blocker_dist;
                        let to_y = from.y + dir_y * blocker_dist;
                        out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                    }
                }
            }
        } else {
            // CASE 2: No blocker found at any prime distance
            // Only generate moves to "interesting" squares that are aligned with pieces on cross-rays
            // This prevents move explosion on infinite boards
            if gen_type != MoveGenType::Captures {
                for &prime_dist in &PRIMES_UNDER_128 {
                    if prime_dist > OPEN_RAY_LIMIT {
                        break;
                    }

                    let to_x = from.x + dir_x * prime_dist;
                    let to_y = from.y + dir_y * prime_dist;

                    // Check if this destination is "interesting" (aligned with some piece on cross-ray)
                    // OR if it's one of the first 2 prime distances (2, 3) which are always generated.
                    let aligned = if dir_x != 0 {
                        indices.cols.get(&to_x).is_some_and(|v| !v.is_empty())
                    } else {
                        indices.rows.get(&to_y).is_some_and(|v| !v.is_empty())
                    };

                    if aligned || prime_dist <= 3 {
                        out.push(Move::new(*from, Coordinate::new(to_x, to_y), *piece));
                    }
                }
            }
        }
    }
}

/// Find the closest blocker at a prime distance for Huygens using spatial indices.
/// Returns (distance_to_blocker, blocker_color). If no blocker, returns (i64::MAX, None).
#[inline]
fn find_huygen_blocker(
    _board: &Board,
    from: &Coordinate,
    dir_x: i64,
    dir_y: i64,
    indices: &SpatialIndices,
    our_color: PlayerColor,
) -> (i64, Option<PlayerColor>) {
    // Get the appropriate spatial index line (row or column)
    let is_horizontal = dir_x != 0;
    let line_vec = if is_horizontal {
        indices.rows.get(&from.y)
    } else {
        indices.cols.get(&from.x)
    };

    let our_coord = if is_horizontal { from.x } else { from.y };

    if let Some(vec) = line_vec {
        // Binary search for our position in the sorted list
        match vec.binary_search_by_key(&our_coord, |(c, _)| *c) {
            Ok(idx) => {
                // Found our position, iterate in the direction to find first blocker at prime distance
                if (is_horizontal && dir_x > 0) || (!is_horizontal && dir_y > 0) {
                    // Positive direction: iterate forward from idx + 1
                    for (coord, packed) in vec.iter().skip(idx + 1) {
                        let dist = coord - our_coord;
                        // O(1) prime check
                        if is_prime_fast(dist) {
                            let p = Piece::from_packed(*packed);
                            // Void blocks like friendly
                            let effective_color = if p.piece_type() == PieceType::Void {
                                our_color
                            } else {
                                p.color()
                            };
                            return (dist, Some(effective_color));
                        }
                    }
                } else {
                    // Negative direction: iterate backward from idx - 1
                    for (coord, packed) in vec.iter().take(idx).rev() {
                        let dist = our_coord - coord;
                        // O(1) prime check
                        if is_prime_fast(dist) {
                            let p = Piece::from_packed(*packed);
                            let effective_color = if p.piece_type() == PieceType::Void {
                                our_color
                            } else {
                                p.color()
                            };
                            return (dist, Some(effective_color));
                        }
                    }
                }
            }
            Err(_) => {
                // Piece not in index (shouldn't happen)
            }
        }
    }

    (i64::MAX, None)
}

/// Rose movement - Circular knightrider that spirals along knight hops.
/// The 8 knight directions in counter-clockwise order:
const ROSE_KNIGHT_DELTAS: [(i64, i64); 8] = [
    (-2, -1), // index 0: SW-ish
    (-1, -2), // index 1: S-ish
    (1, -2),  // index 2: SE-ish
    (2, -1),  // index 3: E-ish
    (2, 1),   // index 4: NE-ish
    (1, 2),   // index 5: N-ish
    (-1, 2),  // index 6: NW-ish
    (-2, 1),  // index 7: W-ish
];

/// Precomputed cumulative offsets for all 16 Rose spirals.
/// Each spiral: 8 starting directions × 2 rotation directions (CCW=+1, CW=-1)
/// Each entry is a sequence of 7 cumulative (dx, dy) values from the origin.
/// Spiral stops if blocked at any intermediate square.
///
/// ROSE_SPIRALS[start_dir][rotation_dir][hop_index] = (cumulative_dx, cumulative_dy)
/// rotation_dir: 0 = CCW (+1), 1 = CW (-1)
pub static ROSE_SPIRALS: [[[(i64, i64); 7]; 2]; 8] = {
    // Build at compile time
    let mut spirals = [[[(0i64, 0i64); 7]; 2]; 8];
    let deltas = ROSE_KNIGHT_DELTAS;

    let mut start = 0usize;
    while start < 8 {
        // CCW direction (rotation +1)
        let mut cum_x = 0i64;
        let mut cum_y = 0i64;
        let mut idx = start;
        let mut hop = 0usize;
        while hop < 7 {
            let (dx, dy) = deltas[idx];
            cum_x += dx;
            cum_y += dy;
            spirals[start][0][hop] = (cum_x, cum_y);
            idx = (idx + 1) % 8; // CCW = next index
            hop += 1;
        }

        // CW direction (rotation -1)
        cum_x = 0;
        cum_y = 0;
        idx = start;
        hop = 0;
        while hop < 7 {
            let (dx, dy) = deltas[idx];
            cum_x += dx;
            cum_y += dy;
            spirals[start][1][hop] = (cum_x, cum_y);
            idx = (idx + 7) % 8; // CW = previous index (equiv to -1 mod 8)
            hop += 1;
        }

        start += 1;
    }
    spirals
};

/// Generate rose moves directly into an output buffer.
/// gen_type controls which move types to generate: All, Quiets only, or Captures only
#[inline(always)]
fn generate_rose_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    gen_type: MoveGenType,
    out: &mut MoveList,
) {
    let my_color = piece.color();
    let fx = from.x;
    let fy = from.y;

    // Dedup seen squares (same square reachable via CW and CCW spirals)
    let mut seen: [(i64, i64); 64] = [(i64::MAX, i64::MAX); 64];
    let mut seen_count = 0usize;

    #[inline(always)]
    fn is_seen_or_mark(seen: &mut [(i64, i64); 64], count: &mut usize, x: i64, y: i64) -> bool {
        for &s in seen.iter().take(*count) {
            if s == (x, y) {
                return true;
            }
        }
        if *count < 64 {
            seen[*count] = (x, y);
            *count += 1;
        }
        false
    }

    // Process all 16 spirals (8 start directions × 2 rotations)
    for spirals_for_dir in &ROSE_SPIRALS {
        for spiral_path in spirals_for_dir {
            // Single pass: walk spiral, generate moves, stop at blocker
            for &(cum_dx, cum_dy) in spiral_path.iter() {
                let tx = fx + cum_dx;
                let ty = fy + cum_dy;

                // Skip if already seen (dedup across spirals)
                if is_seen_or_mark(&mut seen, &mut seen_count, tx, ty) {
                    continue;
                }

                if let Some(target) = board.get_piece(tx, ty) {
                    // Capture opportunity - blocked after this
                    if is_enemy_piece(target, my_color) && gen_type != MoveGenType::Quiets {
                        out.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                    }
                    break; // Blocked - can't continue spiral
                } else {
                    // Empty square - quiet move
                    if gen_type != MoveGenType::Captures {
                        out.push(Move::new(*from, Coordinate::new(tx, ty), *piece));
                    }
                    // Continue spiraling
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
    // If promotion_ranks is not set AND promotions_allowed is not set, use empty (no promotions)
    let promotion_ranks: Vec<i64> = if let Some(ref ranks) = game_rules.promotion_ranks {
        match piece.color() {
            PlayerColor::White => ranks.white.clone(),
            PlayerColor::Black => ranks.black.clone(),
            PlayerColor::Neutral => unsafe { std::hint::unreachable_unchecked() },
        }
    } else if game_rules
        .promotions_allowed
        .as_deref()
        .unwrap_or(&[])
        .is_empty()
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
        .as_deref()
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
        } else if en_passant
            .as_ref()
            .is_some_and(|ep| ep.square.x == capture_x && ep.square.y == capture_y)
        {
            out.push(Move::new(
                *from,
                Coordinate::new(capture_x, capture_y),
                *piece,
            ));
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
        if board.get_piece(coord.x, coord.y).is_some_and(|p| {
            p.color() == piece.color()
                && p.piece_type() != PieceType::Pawn
                && !p.piece_type().is_royal()
        }) {
            let dx = coord.x - from.x;
            let dy = coord.y - from.y;

            if dy == 0 {
                let dir = if dx > 0 { 1i64 } else { -1i64 };

                // Use spatial indices to check path - O(log n) instead of O(distance)
                let mut clear = true;
                if let Some((nearest_x, _)) = indices
                    .rows
                    .get(&from.y)
                    .and_then(|row| SpatialIndices::find_nearest(row, from.x, dir))
                    && ((dir > 0 && nearest_x < coord.x) || (dir < 0 && nearest_x > coord.x))
                {
                    clear = false;
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

/// Generate sliding moves directly into an output buffer
#[inline]
pub fn generate_sliding_moves_into(ctx: &SlidingMoveContext, out: &mut MoveList) {
    generate_sliding_moves_impl(ctx, out, MoveGenType::All);
}

/// Generate only quiet (non-capture) sliding moves directly into output buffer.
#[inline]
pub fn generate_sliding_quiets_into(ctx: &SlidingMoveContext, out: &mut MoveList) {
    generate_sliding_moves_impl(ctx, out, MoveGenType::Quiets);
}

/// Generate knightrider moves directly into an output buffer
/// gen_type controls which move types to generate: All, Quiets only, or Captures only
#[inline]
fn generate_knightrider_moves_into(
    board: &Board,
    from: &Coordinate,
    piece: &Piece,
    gen_type: MoveGenType,
    out: &mut MoveList,
) {
    let moves = generate_knightrider_moves(board, from, piece);
    for m in moves {
        let is_capture = board.get_piece(m.to.x, m.to.y).is_some();
        // Filter based on gen_type
        if gen_type == MoveGenType::Quiets && is_capture {
            continue;
        }
        if gen_type == MoveGenType::Captures && !is_capture {
            continue;
        }
        out.push(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================== Bounds Tests ========================

    #[test]
    fn test_in_bounds_default() {
        // Default bounds are very large (-1e15 to 1e15)
        assert!(in_bounds(0, 0));
        assert!(in_bounds(1000, 1000));
        assert!(in_bounds(-1000, -1000));
        assert!(in_bounds(1_000_000_000, 1_000_000_000));
    }

    #[test]
    fn test_set_world_bounds() {
        // Set custom bounds
        set_world_bounds(-100, 100, -50, 50);

        assert!(in_bounds(0, 0));
        assert!(in_bounds(100, 50));
        assert!(in_bounds(-100, -50));
        assert!(!in_bounds(101, 0));
        assert!(!in_bounds(0, 51));

        // Reset to large defaults
        set_world_bounds(
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
        );
    }

    #[test]
    fn test_get_world_size() {
        set_world_bounds(-100, 100, -50, 50);
        let size = get_world_size();
        assert_eq!(size, 200, "Width is larger than height");

        // Reset
        set_world_bounds(
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
        );
    }

    #[test]
    fn test_get_coord_bounds() {
        set_world_bounds(-10, 20, -30, 40);
        let (min_x, max_x, min_y, max_y) = get_coord_bounds();
        assert_eq!(min_x, -10);
        assert_eq!(max_x, 20);
        assert_eq!(min_y, -30);
        assert_eq!(max_y, 40);

        // Reset
        set_world_bounds(
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
            -1_000_000_000_000_000,
            1_000_000_000_000_000,
        );
    }

    // ======================== SpatialIndices Tests ========================

    #[test]
    fn test_spatial_indices_new_empty() {
        let board = Board::new();
        let indices = SpatialIndices::new(&board);

        assert!(indices.rows.is_empty());
        assert!(indices.cols.is_empty());
        assert!(indices.diag1.is_empty());
        assert!(indices.diag2.is_empty());
    }

    #[test]
    fn test_spatial_indices_add_remove() {
        let mut indices = SpatialIndices::default();
        let packed = Piece::new(PieceType::Rook, PlayerColor::White).packed();

        // Add piece at (5, 10)
        indices.add(5, 10, packed);

        // Check it's in all the right indices
        assert!(indices.rows.contains_key(&10));
        assert!(indices.cols.contains_key(&5));
        assert!(indices.diag1.contains_key(&-5)); // 5 - 10 = -5
        assert!(indices.diag2.contains_key(&15)); // 5 + 10 = 15

        // Remove it
        indices.remove(5, 10);

        // Check it's removed from all indices
        assert!(indices.rows.get(&10).map(|v| v.is_empty()).unwrap_or(true));
        assert!(indices.cols.get(&5).map(|v| v.is_empty()).unwrap_or(true));
    }

    #[test]
    fn test_spatial_indices_find_nearest_forward() {
        let vec = vec![(0, 1), (5, 2), (10, 3), (20, 4)];

        // Find nearest forward from position 3
        let result = SpatialIndices::find_nearest(&vec, 3, 1);
        assert_eq!(result, Some((5, 2)), "Should find piece at coord 5");

        // Find nearest forward from position 10
        let result = SpatialIndices::find_nearest(&vec, 10, 1);
        assert_eq!(result, Some((20, 4)), "Should find piece at coord 20");
    }

    #[test]
    fn test_spatial_indices_find_nearest_backward() {
        let vec = vec![(0, 1), (5, 2), (10, 3), (20, 4)];

        // Find nearest backward from position 7
        let result = SpatialIndices::find_nearest(&vec, 7, -1);
        assert_eq!(result, Some((5, 2)), "Should find piece at coord 5");

        // Find nearest backward from position 0
        let result = SpatialIndices::find_nearest(&vec, 0, -1);
        assert_eq!(result, None, "No piece before 0");
    }

    #[test]
    fn test_spatial_indices_find_nearest_at_extreme_distance() {
        // Test with large coordinates (infinite chess scale)
        let vec = vec![(-1_000_000, 1), (0, 2), (1_000_000, 3)];

        let result = SpatialIndices::find_nearest(&vec, 0, 1);
        assert_eq!(result, Some((1_000_000, 3)), "Should find distant piece");

        let result = SpatialIndices::find_nearest(&vec, 0, -1);
        assert_eq!(
            result,
            Some((-1_000_000, 1)),
            "Should find distant piece backward"
        );
    }

    // ======================== Move Generation Tests ========================

    #[test]
    fn test_move_new() {
        let from = Coordinate::new(1, 2);
        let to = Coordinate::new(3, 4);
        let piece = Piece::new(PieceType::Knight, PlayerColor::White);

        let m = Move::new(from, to, piece);

        assert_eq!(m.from.x, 1);
        assert_eq!(m.from.y, 2);
        assert_eq!(m.to.x, 3);
        assert_eq!(m.to.y, 4);
        assert!(m.promotion.is_none());
        assert!(m.rook_coord.is_none());
    }

    #[test]
    fn test_is_enemy_piece_detection() {
        let white_knight = Piece::new(PieceType::Knight, PlayerColor::White);
        let black_knight = Piece::new(PieceType::Knight, PlayerColor::Black);
        let void = Piece::new(PieceType::Void, PlayerColor::Neutral);

        // White sees black as enemy
        assert!(is_enemy_piece(&black_knight, PlayerColor::White));
        // Black sees white as enemy
        assert!(is_enemy_piece(&white_knight, PlayerColor::Black));
        // Same color is not enemy
        assert!(!is_enemy_piece(&white_knight, PlayerColor::White));
        // Void is not enemy (it's neutral, but also blocked by piece type check)
        assert!(!is_enemy_piece(&void, PlayerColor::White));
    }

    #[test]
    fn test_slider_detection_at_distance() {
        // Test that SpatialIndices can find pieces at large distances
        // This is the foundation for slider attack detection in infinite chess
        let mut board = Board::new();
        board.set_piece(0, 0, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(1000, 0, Piece::new(PieceType::King, PlayerColor::Black));

        let indices = SpatialIndices::new(&board);

        // The row should have both pieces
        let row = indices.rows.get(&0).unwrap();
        assert_eq!(row.len(), 2, "Row should have 2 pieces");

        // Find nearest from rook position toward king
        let result = SpatialIndices::find_nearest(row, 0, 1);
        assert_eq!(
            result.map(|(c, _)| c),
            Some(1000),
            "Should find king at x=1000"
        );
    }

    #[test]
    fn test_knight_moves_generation() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Knight, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_leaper_moves_into(&board, &from, &piece, 1, 2, MoveGenType::All, &mut moves);

        // Knight has 8 possible moves from center
        assert_eq!(moves.len(), 8, "Knight should have 8 moves from (4,4)");

        // Check specific squares
        let expected = [
            (5, 6),
            (6, 5),
            (6, 3),
            (5, 2),
            (3, 2),
            (2, 3),
            (2, 5),
            (3, 6),
        ];
        for (x, y) in expected {
            assert!(
                moves.iter().any(|m| m.to.x == x && m.to.y == y),
                "Knight should be able to move to ({}, {})",
                x,
                y
            );
        }
    }

    #[test]
    fn test_king_moves_generation() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::King, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_compass_moves_into(&board, &from, &piece, 1, MoveGenType::All, &mut moves);

        // King has 8 possible moves from center
        assert_eq!(moves.len(), 8, "King should have 8 moves from (4,4)");
    }

    #[test]
    fn test_fairy_piece_camel() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Camel, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Camel, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_leaper_moves_into(&board, &from, &piece, 1, 3, MoveGenType::All, &mut moves);

        // Camel leaps (1,3) - 8 squares
        assert_eq!(moves.len(), 8, "Camel should have 8 moves from (4,4)");

        // Check a specific camel square
        assert!(
            moves.iter().any(|m| m.to.x == 5 && m.to.y == 7),
            "Camel should be able to move to (5, 7)"
        );
    }

    #[test]
    fn test_fairy_piece_zebra() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Zebra, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Zebra, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_leaper_moves_into(&board, &from, &piece, 2, 3, MoveGenType::All, &mut moves);

        // Zebra leaps (2,3) - 8 squares
        assert_eq!(moves.len(), 8, "Zebra should have 8 moves from (4,4)");
    }

    #[test]
    fn test_negative_coordinates() {
        // Test that piece at negative coordinates generates moves correctly
        let mut board = Board::new();
        board.set_piece(
            -100,
            -100,
            Piece::new(PieceType::Knight, PlayerColor::White),
        );

        let from = Coordinate::new(-100, -100);
        let piece = Piece::new(PieceType::Knight, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_leaper_moves_into(&board, &from, &piece, 1, 2, MoveGenType::All, &mut moves);

        assert_eq!(
            moves.len(),
            8,
            "Knight at negative coords should have 8 moves"
        );

        // Check one of the expected squares
        assert!(
            moves.iter().any(|m| m.to.x == -99 && m.to.y == -98),
            "Knight should be able to move to (-99, -98)"
        );
    }

    #[test]
    fn test_is_enemy_piece() {
        let white_pawn = Piece::new(PieceType::Pawn, PlayerColor::White);
        let black_pawn = Piece::new(PieceType::Pawn, PlayerColor::Black);

        assert!(!is_enemy_piece(&white_pawn, PlayerColor::White));
        assert!(is_enemy_piece(&black_pawn, PlayerColor::White));
        assert!(is_enemy_piece(&white_pawn, PlayerColor::Black));
    }

    #[test]
    fn test_generate_pawn_moves() {
        use crate::game::PromotionRanks;

        let mut board = Board::new();
        board.set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.set_piece(5, 3, Piece::new(PieceType::Pawn, PlayerColor::Black)); // Capture target

        let from = Coordinate::new(4, 2);
        let piece = Piece::new(PieceType::Pawn, PlayerColor::White);

        let rules = GameRules {
            promotions_allowed: Some(vec!["queens".to_string()]),
            promotion_ranks: Some(PromotionRanks {
                white: vec![8],
                black: vec![1],
            }),
            ..GameRules::default()
        };

        let special = FxHashSet::default();
        let mut moves = MoveList::new();
        generate_pawn_moves_into(&board, &from, &piece, &special, &None, &rules, &mut moves);

        assert!(moves.len() >= 2, "Pawn should have at least 2 moves");
        // Should include forward move and capture
        assert!(
            moves.iter().any(|m| m.to.y == 3 && m.to.x == 4),
            "Forward move"
        );
        assert!(moves.iter().any(|m| m.to.y == 3 && m.to.x == 5), "Capture");
    }

    #[test]
    fn test_generate_sliding_moves_rook() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        board.rebuild_tiles();

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Rook, PlayerColor::White);
        let indices = SpatialIndices::new(&board);

        let ortho = &[(1, 0), (-1, 0), (0, 1), (0, -1)];
        let mut moves = MoveList::new();
        generate_sliding_moves_into(
            &SlidingMoveContext {
                board: &board,
                from: &from,
                piece: &piece,
                directions: ortho,
                indices: &indices,
                enemy_king_pos: None,
            },
            &mut moves,
        );

        // Rook on empty board should have many moves (limited by fallback)
        assert!(!moves.is_empty(), "Rook should have some moves");
    }

    #[test]
    fn test_generate_sliding_moves_bishop() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Bishop, PlayerColor::White));
        board.rebuild_tiles();

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Bishop, PlayerColor::White);
        let indices = SpatialIndices::new(&board);

        let diag = &[(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut moves = MoveList::new();
        generate_sliding_moves_into(
            &SlidingMoveContext {
                board: &board,
                from: &from,
                piece: &piece,
                directions: diag,
                indices: &indices,
                enemy_king_pos: None,
            },
            &mut moves,
        );

        assert!(!moves.is_empty(), "Bishop should have some moves");
    }

    #[test]
    fn test_is_square_attacked_by_knight() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        board.rebuild_tiles();

        let indices = SpatialIndices::new(&board);
        let target_attacked = Coordinate::new(5, 6); // Knight can attack this
        let target_not_attacked = Coordinate::new(4, 5); // Knight cannot attack this

        assert!(is_square_attacked(
            &board,
            &target_attacked,
            PlayerColor::White,
            &indices
        ));
        assert!(!is_square_attacked(
            &board,
            &target_not_attacked,
            PlayerColor::White,
            &indices
        ));
    }

    #[test]
    fn test_is_square_attacked_by_rook() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        board.rebuild_tiles();

        let indices = SpatialIndices::new(&board);
        let target_file = Coordinate::new(4, 10); // Same file
        let target_rank = Coordinate::new(10, 4); // Same rank

        assert!(is_square_attacked(
            &board,
            &target_file,
            PlayerColor::White,
            &indices
        ));
        assert!(is_square_attacked(
            &board,
            &target_rank,
            PlayerColor::White,
            &indices
        ));
    }

    #[test]
    fn test_is_square_attacked_blocked() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(4, 6, Piece::new(PieceType::Pawn, PlayerColor::White)); // Blocker
        board.rebuild_tiles();

        let indices = SpatialIndices::new(&board);
        let target_blocked = Coordinate::new(4, 10); // Blocked by pawn at (4,6)

        assert!(!is_square_attacked(
            &board,
            &target_blocked,
            PlayerColor::White,
            &indices
        ));
    }

    #[test]
    fn test_generate_castling_moves() {
        let mut board = Board::new();
        board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        board.set_piece(8, 1, Piece::new(PieceType::Rook, PlayerColor::White)); // Kingside rook
        board.rebuild_tiles();

        let from = Coordinate::new(5, 1);
        let piece = Piece::new(PieceType::King, PlayerColor::White);

        let mut special = FxHashSet::default();
        special.insert(Coordinate::new(5, 1)); // King has special right
        special.insert(Coordinate::new(8, 1)); // Rook has special right

        let indices = SpatialIndices::new(&board);
        let moves = generate_castling_moves(&board, &from, &piece, &special, &indices);

        // Test that the function runs without panicking and returns a MoveList
        // Castling availability depends on variant rules and board state
        let _ = moves.len();
    }

    #[test]
    fn test_ray_border_distance() {
        let from = Coordinate::new(0, 0);

        // Moving right (positive x)
        let dist = ray_border_distance(&from, 1, 0);
        assert!(dist.is_some());
        assert!(dist.unwrap() > 0);
    }

    #[test]
    fn test_generate_compass_moves() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Hawk, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Hawk, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_compass_moves_into(&board, &from, &piece, 2, MoveGenType::All, &mut moves);

        // Distance 2 compass should have 8 moves (4 ortho + 4 diag)
        assert_eq!(moves.len(), 8);
    }

    #[test]
    fn test_spatial_indices_default() {
        let indices = SpatialIndices::default();
        assert!(indices.rows.is_empty());
        assert!(indices.cols.is_empty());
        assert!(indices.diag1.is_empty());
        assert!(indices.diag2.is_empty());
    }

    #[test]
    fn test_find_blocker_via_indices() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rook, PlayerColor::White));
        board.set_piece(4, 8, Piece::new(PieceType::Pawn, PlayerColor::White)); // Blocker
        board.rebuild_tiles();

        let from = Coordinate::new(4, 4);
        let indices = SpatialIndices::new(&board);

        // Looking up (positive y)
        let (dist, captures) =
            find_blocker_via_indices(&board, &from, 0, 1, &indices, PlayerColor::White);

        assert!(dist > 0, "Should find a blocker");
        assert!(!captures, "Own piece should not be a capture");
    }

    #[test]
    fn test_generate_knightrider_moves() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Knightrider, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Knightrider, PlayerColor::White);

        let moves = generate_knightrider_moves(&board, &from, &piece);

        // Knightrider should have at least 8 moves (the initial knight squares)
        assert!(moves.len() >= 8, "Knightrider should have at least 8 moves");
    }

    #[test]
    fn test_generate_rose_moves() {
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rose, PlayerColor::White));

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Rose, PlayerColor::White);

        let mut moves = MoveList::new();
        generate_rose_moves_into(&board, &from, &piece, MoveGenType::All, &mut moves);

        assert!(!moves.is_empty(), "Rose should have some moves");
    }

    #[test]
    fn test_get_legal_moves() {
        use crate::game::GameRules;

        let mut board = Board::new();
        board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        board.set_piece(4, 2, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.rebuild_tiles();

        let indices = SpatialIndices::new(&board);
        let special = FxHashSet::default();
        let rules = GameRules::default();

        let ctx = MoveGenContext {
            special_rights: &special,
            en_passant: &None,
            game_rules: &rules,
            indices: &indices,
            enemy_king_pos: Some(&Coordinate::new(5, 8)),
        };

        let moves = get_legal_moves(&board, PlayerColor::White, &ctx);

        assert!(!moves.is_empty(), "White should have legal moves");
    }

    #[test]
    fn test_get_quiescence_captures() {
        use crate::game::GameRules;

        let mut board = Board::new();
        board.set_piece(5, 1, Piece::new(PieceType::King, PlayerColor::White));
        board.set_piece(5, 8, Piece::new(PieceType::King, PlayerColor::Black));
        board.set_piece(4, 4, Piece::new(PieceType::Knight, PlayerColor::White));
        board.set_piece(5, 6, Piece::new(PieceType::Pawn, PlayerColor::Black)); // Capture target
        board.rebuild_tiles();

        let indices = SpatialIndices::new(&board);
        let special = FxHashSet::default();
        let rules = GameRules::default();

        let ctx = MoveGenContext {
            special_rights: &special,
            en_passant: &None,
            game_rules: &rules,
            indices: &indices,
            enemy_king_pos: None,
        };

        let mut captures = MoveList::new();
        get_quiescence_captures(&board, PlayerColor::White, &ctx, &mut captures);

        // Should find the knight capture
        assert!(!captures.is_empty(), "Should find capture moves");
    }

    #[test]
    fn test_generate_rose_moves_unblocked() {
        // Rose on empty board should have many moves
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rose, PlayerColor::White));
        board.rebuild_tiles();

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Rose, PlayerColor::White);
        let mut moves = MoveList::new();
        generate_rose_moves_into(&board, &from, &piece, MoveGenType::All, &mut moves);

        // Should have moves (each of 16 spirals can go up to 7 hops, though many overlap)
        assert!(!moves.is_empty(), "Rose should have moves on empty board");

        // First hop in any spiral should be a knight move
        // Check that (-2, -1) from origin is in the moves
        let has_knight_move = moves.iter().any(|m| m.to.x == 2 && m.to.y == 3);
        assert!(
            has_knight_move,
            "Rose should be able to make knight-like first hops"
        );
    }

    #[test]
    fn test_generate_rose_moves_blocked() {
        // Rose with a blocker that prevents some moves
        let mut board = Board::new();
        board.set_piece(4, 4, Piece::new(PieceType::Rose, PlayerColor::White));
        // Place blocker at first knight hop destination
        board.set_piece(3, 2, Piece::new(PieceType::Pawn, PlayerColor::White)); // (4-1, 4-2)
        board.rebuild_tiles();

        let from = Coordinate::new(4, 4);
        let piece = Piece::new(PieceType::Rose, PlayerColor::White);
        let mut moves = MoveList::new();
        generate_rose_moves_into(&board, &from, &piece, MoveGenType::All, &mut moves);

        // Should NOT have the blocked square as a move (friendly piece)
        let has_blocked_square = moves.iter().any(|m| m.to.x == 3 && m.to.y == 2);
        assert!(
            !has_blocked_square,
            "Rose should not move to square occupied by friendly piece"
        );
    }

    #[test]
    fn test_generate_rose_spirals_correct() {
        // Verify the spiral constants are computed correctly
        // Start direction 0, CCW: deltas[0] + deltas[1] + deltas[2] + ...
        // deltas[0] = (-2, -1)
        // deltas[1] = (-1, -2)
        // Cumulative: hop 0 = (-2, -1), hop 1 = (-3, -3)
        assert_eq!(ROSE_SPIRALS[0][0][0], (-2, -1), "First CCW hop from dir 0");
        assert_eq!(ROSE_SPIRALS[0][0][1], (-3, -3), "Second CCW hop from dir 0");
    }
    #[test]
    fn test_long_distance_royal_targeting() {
        use crate::game::GameState;

        let mut game = GameState::new();
        // White Queen at (10, -30)
        game.board
            .set_piece(10, -30, Piece::new(PieceType::Queen, PlayerColor::White));
        // Black King at (77, -41)
        game.board
            .set_piece(77, -41, Piece::new(PieceType::King, PlayerColor::Black));

        // Recompute indices and king positions
        game.recompute_piece_counts();

        let ctx = MoveGenContext {
            special_rights: &game.special_rights,
            en_passant: &game.en_passant,
            game_rules: &game.game_rules,
            indices: &game.spatial_indices,
            enemy_king_pos: game.black_king_pos.as_ref(),
        };

        let moves = get_legal_moves(&game.board, PlayerColor::White, &ctx);

        let target_from = Coordinate::new(10, -30);
        let target_to = Coordinate::new(77, -30);

        let found = moves
            .iter()
            .any(|m| m.from == target_from && m.to == target_to);

        assert!(
            found,
            "Move (10,-30) -> (77,-30) should be generated to target King at (77,-41)"
        );
    }
}
