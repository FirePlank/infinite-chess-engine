//! Sparse Tiled Bitboards for Infinite Chess
//!
//! This module implements an 8×8 tile-based representation that dramatically
//! reduces HashMap lookups for leaper attack detection and move generation.
//! Each tile contains u64 occupancy bitboards and packed piece arrays.
//!
//! The design is SIMD-ready: piece array scans can be vectorized with u8x16.

pub mod magic;
pub mod masks;

use crate::board::{Piece, PlayerColor};

// ============================================================================
// Tile Constants
// ============================================================================

/// Tile size is 8×8 (fits in u64 bitboard)
pub const TILE_SHIFT: i32 = 3;
pub const TILE_SIZE: i64 = 1 << TILE_SHIFT;
pub const TILE_MASK: i64 = TILE_SIZE - 1; // 0b111 = 7

/// TileTable capacity (power of 2) - 512 handles extreme positions during search
pub const TILE_TABLE_CAPACITY: usize = 512;
const TILE_TABLE_MASK: usize = TILE_TABLE_CAPACITY - 1;

// ============================================================================
// Tile Coordinate Math
// ============================================================================

/// Convert world coordinate to tile coordinate using arithmetic shift.
/// Works correctly for negative coordinates.
#[inline(always)]
pub fn tile_coord(v: i64) -> i64 {
    v >> TILE_SHIFT
}

/// Convert world coordinates to tile coordinates.
#[inline(always)]
pub fn tile_coords(x: i64, y: i64) -> (i64, i64) {
    (tile_coord(x), tile_coord(y))
}

/// Get local coordinate within tile (0..7).
/// Uses bitwise AND which works correctly for negative coordinates.
#[inline(always)]
pub fn local_coord(v: i64) -> usize {
    (v as u64 & TILE_MASK as u64) as usize
}

/// Get local x,y within tile.
#[inline(always)]
pub fn local_coords(x: i64, y: i64) -> (usize, usize) {
    (local_coord(x), local_coord(y))
}

/// Get index into tile's 64-element arrays (row-major: y*8 + x)
#[inline(always)]
pub fn local_index(x: i64, y: i64) -> usize {
    let lx = local_coord(x);
    let ly = local_coord(y);
    ly * 8 + lx
}

/// Get bit mask for a square within a tile.
#[inline(always)]
pub fn bit_mask(x: i64, y: i64) -> u64 {
    1u64 << local_index(x, y)
}

/// Get neighbor tile index from delta (-1, 0, 1).
/// Returns index in 0..9 for the 3×3 neighborhood.
/// Center tile is index 4.
#[inline(always)]
pub fn neighbor_index(dx: i64, dy: i64) -> usize {
    ((dy + 1) * 3 + (dx + 1)) as usize
}

// ============================================================================
// Tile Structure
// ============================================================================

/// An 8×8 tile containing occupancy bitboards and packed piece data.
/// Aligned to 64 bytes for cache efficiency.
///
/// BITBOARD ARCHITECTURE: Per-piece-type occupancy for Stockfish-style move gen.
/// This allows O(popcount) iteration over specific piece types without scanning.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Tile {
    /// Bitboard of all occupied squares (including voids/obstacles)
    pub occ_all: u64,
    /// Bitboard of white-occupied squares
    pub occ_white: u64,
    /// Bitboard of black-occupied squares
    pub occ_black: u64,
    /// Bitboard of void/obstacle squares (non-capturable blockers)
    pub occ_void: u64,

    // ===== PER-PIECE-TYPE BITBOARDS (Stockfish pattern) =====
    /// Bitboard of pawns
    pub occ_pawns: u64,
    /// Bitboard of knights (including fairy leapers that move like knights)
    pub occ_knights: u64,
    /// Bitboard of bishops
    pub occ_bishops: u64,
    /// Bitboard of rooks
    pub occ_rooks: u64,
    /// Bitboard of queens
    pub occ_queens: u64,
    /// Bitboard of kings (including royal pieces)
    pub occ_kings: u64,
    /// Bitboard of diagonal sliders (bishops + queens + archbishop)
    pub occ_diag_sliders: u64,
    /// Bitboard of orthogonal sliders (rooks + queens + chancellor)
    pub occ_ortho_sliders: u64,

    /// Packed piece codes for each square (0 = empty)
    /// Index: y*8 + x (row-major)
    pub piece: [u8; 64],
}

impl Default for Tile {
    fn default() -> Self {
        Tile {
            occ_all: 0,
            occ_white: 0,
            occ_black: 0,
            occ_void: 0,
            occ_pawns: 0,
            occ_knights: 0,
            occ_bishops: 0,
            occ_rooks: 0,
            occ_queens: 0,
            occ_kings: 0,
            occ_diag_sliders: 0,
            occ_ortho_sliders: 0,
            piece: [0; 64],
        }
    }
}

impl Tile {
    /// Create a new empty tile.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if tile is completely empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occ_all == 0
    }

    /// Set a piece at local index.
    /// Updates all piece-type specific bitboards for Stockfish-style move gen.
    #[inline]
    pub fn set_piece(&mut self, idx: usize, piece: Piece) {
        use crate::board::PieceType;

        let bit = 1u64 << idx;
        self.occ_all |= bit;
        self.piece[idx] = piece.packed();

        // Color bitboards
        match piece.color() {
            PlayerColor::White => self.occ_white |= bit,
            PlayerColor::Black => self.occ_black |= bit,
            PlayerColor::Neutral => {
                if piece.piece_type().is_neutral_type() {
                    self.occ_void |= bit;
                }
            }
        }

        // Per-piece-type bitboards (Stockfish pattern)
        match piece.piece_type() {
            PieceType::Pawn => self.occ_pawns |= bit,
            PieceType::Knight => self.occ_knights |= bit,
            PieceType::Bishop => {
                self.occ_bishops |= bit;
                self.occ_diag_sliders |= bit;
            }
            PieceType::Rook => {
                self.occ_rooks |= bit;
                self.occ_ortho_sliders |= bit;
            }
            PieceType::Queen | PieceType::RoyalQueen => {
                self.occ_queens |= bit;
                self.occ_diag_sliders |= bit;
                self.occ_ortho_sliders |= bit;
            }
            PieceType::King => self.occ_kings |= bit,
            // Compound pieces
            PieceType::Archbishop => {
                self.occ_diag_sliders |= bit; // Bishop component
            }
            PieceType::Chancellor => {
                self.occ_ortho_sliders |= bit; // Rook component
            }
            PieceType::Amazon => {
                self.occ_diag_sliders |= bit;
                self.occ_ortho_sliders |= bit;
            }
            // Royal pieces
            PieceType::RoyalCentaur => self.occ_kings |= bit,
            // Leapers that move like knights
            PieceType::Centaur => self.occ_knights |= bit,
            // Other pieces don't have standard bitboards
            _ => {}
        }
    }

    /// Remove a piece at local index. Returns the old packed piece code.
    /// Clears all per-piece-type bitboards for consistency.
    #[inline]
    pub fn remove_piece(&mut self, idx: usize) -> u8 {
        let bit = 1u64 << idx;
        let old_packed = self.piece[idx];

        if old_packed != 0 {
            // Clear all bitboards unconditionally (cheaper than checking type)
            self.occ_all &= !bit;
            self.occ_white &= !bit;
            self.occ_black &= !bit;
            self.occ_void &= !bit;
            self.occ_pawns &= !bit;
            self.occ_knights &= !bit;
            self.occ_bishops &= !bit;
            self.occ_rooks &= !bit;
            self.occ_queens &= !bit;
            self.occ_kings &= !bit;
            self.occ_diag_sliders &= !bit;
            self.occ_ortho_sliders &= !bit;
            self.piece[idx] = 0;
        }

        old_packed
    }

    /// Get piece at local index, if any.
    #[inline]
    pub fn get_piece(&self, idx: usize) -> Option<Piece> {
        let packed = self.piece[idx];
        if packed == 0 {
            None
        } else {
            Some(Piece::from_packed(packed))
        }
    }

    /// Get occupancy for a specific color.
    #[inline]
    pub fn occ_for_color(&self, color: PlayerColor) -> u64 {
        match color {
            PlayerColor::White => self.occ_white,
            PlayerColor::Black => self.occ_black,
            PlayerColor::Neutral => self.occ_void,
        }
    }

    /// Clear the tile completely.
    #[inline]
    pub fn clear(&mut self) {
        self.occ_all = 0;
        self.occ_white = 0;
        self.occ_black = 0;
        self.occ_void = 0;
        self.occ_pawns = 0;
        self.occ_knights = 0;
        self.occ_bishops = 0;
        self.occ_rooks = 0;
        self.occ_queens = 0;
        self.occ_kings = 0;
        self.occ_diag_sliders = 0;
        self.occ_ortho_sliders = 0;
        self.piece = [0; 64];
    }

    /// Bitwise sum of local X coordinates (0..7) for all set bits.
    /// Used for centroid calculation without CTZ loops.
    #[inline]
    pub fn sum_lx(&self, bits: u64) -> u32 {
        const COL_1: u64 = 0x0202020202020202;
        const COL_2: u64 = 0x0404040404040404;
        const COL_3: u64 = 0x0808080808080808;
        const COL_4: u64 = 0x1010101010101010;
        const COL_5: u64 = 0x2020202020202020;
        const COL_6: u64 = 0x4040404040404040;
        const COL_7: u64 = 0x8080808080808080;

        let mut sum = (bits & COL_1).count_ones();
        sum += (bits & COL_2).count_ones() * 2;
        sum += (bits & COL_3).count_ones() * 3;
        sum += (bits & COL_4).count_ones() * 4;
        sum += (bits & COL_5).count_ones() * 5;
        sum += (bits & COL_6).count_ones() * 6;
        sum += (bits & COL_7).count_ones() * 7;
        sum
    }

    /// Bitwise sum of local Y coordinates (0..7) for all set bits.
    /// Used for centroid calculation without CTZ loops.
    #[inline]
    pub fn sum_ly(&self, bits: u64) -> u32 {
        const ROW_1: u64 = 0x000000000000FF00;
        const ROW_2: u64 = 0x0000000000FF0000;
        const ROW_3: u64 = 0x00000000FF000000;
        const ROW_4: u64 = 0x000000FF00000000;
        const ROW_5: u64 = 0x0000FF0000000000;
        const ROW_6: u64 = 0x00FF000000000000;
        const ROW_7: u64 = 0xFF00000000000000;

        let mut sum = (bits & ROW_1).count_ones();
        sum += (bits & ROW_2).count_ones() * 2;
        sum += (bits & ROW_3).count_ones() * 3;
        sum += (bits & ROW_4).count_ones() * 4;
        sum += (bits & ROW_5).count_ones() * 5;
        sum += (bits & ROW_6).count_ones() * 6;
        sum += (bits & ROW_7).count_ones() * 7;
        sum
    }
}

// ============================================================================
// TileTable Bucket
// ============================================================================

/// Bucket states for open-addressing hash table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum BucketState {
    Empty = 0,
    Occupied = 1,
    Tombstone = 2,
}

/// A bucket in the tile table.
#[derive(Clone, Debug)]
struct Bucket {
    cx: i64,
    cy: i64,
    state: BucketState,
    tile: Tile,
}

impl Default for Bucket {
    fn default() -> Self {
        Bucket {
            cx: 0,
            cy: 0,
            state: BucketState::Empty,
            tile: Tile::new(),
        }
    }
}

// ============================================================================
// TileTable
// ============================================================================

/// Fixed-size open-addressing hash table for tiles.
/// Uses linear probing. Never grows (128 buckets is plenty for ~70 pieces).
pub struct TileTable {
    buckets: Box<[Bucket; TILE_TABLE_CAPACITY]>,
    count: usize,
    /// BITBOARD: Bitmask of occupied buckets (512 bits = 8 * u64)
    occ_mask: [u64; 8],
}

impl Clone for TileTable {
    fn clone(&self) -> Self {
        TileTable {
            buckets: self.buckets.clone(),
            count: self.count,
            occ_mask: self.occ_mask,
        }
    }
}

impl Default for TileTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TileTable {
    /// Create a new empty tile table.
    pub fn new() -> Self {
        // Use a boxed array to avoid stack overflow
        let buckets = vec![Bucket::default(); TILE_TABLE_CAPACITY]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        TileTable {
            buckets,
            count: 0,
            occ_mask: [0; 8],
        }
    }

    /// Hash tile coordinates to bucket index.
    #[inline]
    fn hash(cx: i64, cy: i64) -> usize {
        // FxHash-style mixing for (cx, cy)
        let mut h = cx as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= cy as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        (h as usize) & TILE_TABLE_MASK
    }

    /// Get a tile, if it exists.
    #[inline]
    pub fn get_tile(&self, cx: i64, cy: i64) -> Option<&Tile> {
        let mut idx = Self::hash(cx, cy);
        for _ in 0..TILE_TABLE_CAPACITY {
            let bucket = &self.buckets[idx];
            match bucket.state {
                BucketState::Empty => return None,
                BucketState::Occupied => {
                    if bucket.cx == cx && bucket.cy == cy {
                        return Some(&bucket.tile);
                    }
                }
                BucketState::Tombstone => {}
            }
            idx = (idx + 1) & TILE_TABLE_MASK;
        }
        None
    }

    /// Get a mutable tile, if it exists.
    #[inline]
    pub fn get_tile_mut(&mut self, cx: i64, cy: i64) -> Option<&mut Tile> {
        let mut idx = Self::hash(cx, cy);
        for _ in 0..TILE_TABLE_CAPACITY {
            let bucket = &self.buckets[idx];
            match bucket.state {
                BucketState::Empty => return None,
                BucketState::Occupied => {
                    if bucket.cx == cx && bucket.cy == cy {
                        return Some(&mut self.buckets[idx].tile);
                    }
                }
                BucketState::Tombstone => {}
            }
            idx = (idx + 1) & TILE_TABLE_MASK;
        }
        None
    }

    /// Get or create a tile at the given coordinates.
    #[inline]
    pub fn get_or_create(&mut self, cx: i64, cy: i64) -> &mut Tile {
        let mut idx = Self::hash(cx, cy);

        loop {
            match self.buckets[idx].state {
                BucketState::Empty | BucketState::Tombstone => {
                    self.buckets[idx] = Bucket {
                        cx,
                        cy,
                        state: BucketState::Occupied,
                        tile: Tile::new(),
                    };
                    self.count += 1;
                    self.occ_mask[idx / 64] |= 1u64 << (idx % 64);
                    return &mut self.buckets[idx].tile;
                }
                BucketState::Occupied => {
                    if self.buckets[idx].cx == cx && self.buckets[idx].cy == cy {
                        return &mut self.buckets[idx].tile;
                    }
                }
            }
            idx = (idx + 1) & TILE_TABLE_MASK;
        }
    }

    /// Remove a tile at the given coordinates (marks as tombstone).
    /// Used when a tile becomes completely empty.
    #[inline]
    pub fn remove(&mut self, cx: i64, cy: i64) {
        let mut idx = Self::hash(cx, cy);
        let start_idx = idx;

        loop {
            match self.buckets[idx].state {
                BucketState::Occupied => {
                    if self.buckets[idx].cx == cx && self.buckets[idx].cy == cy {
                        self.buckets[idx].state = BucketState::Tombstone;
                        self.buckets[idx].tile.clear();
                        self.count -= 1;
                        self.occ_mask[idx / 64] &= !(1u64 << (idx % 64));
                        return;
                    }
                }
                BucketState::Empty => return, // Not found
                _ => {}
            }
            idx = (idx + 1) & TILE_TABLE_MASK;
            if idx == start_idx {
                break;
            }
        }
    }

    /// Get the 3×3 neighborhood of tiles around (cx, cy).
    /// Returns array indexed by neighbor_index().
    /// Index 4 is the center tile.
    #[inline]
    pub fn get_neighborhood(&self, cx: i64, cy: i64) -> [Option<&Tile>; 9] {
        [
            self.get_tile(cx - 1, cy - 1), // 0: (-1, -1)
            self.get_tile(cx, cy - 1),     // 1: (0, -1)
            self.get_tile(cx + 1, cy - 1), // 2: (1, -1)
            self.get_tile(cx - 1, cy),     // 3: (-1, 0)
            self.get_tile(cx, cy),         // 4: (0, 0) - center
            self.get_tile(cx + 1, cy),     // 5: (1, 0)
            self.get_tile(cx - 1, cy + 1), // 6: (-1, 1)
            self.get_tile(cx, cy + 1),     // 7: (0, 1)
            self.get_tile(cx + 1, cy + 1), // 8: (1, 1)
        ]
    }

    /// Clear all tiles.
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.state = BucketState::Empty;
            bucket.tile.clear();
        }
        self.count = 0;
        self.occ_mask = [0; 8];
    }

    /// Get the number of occupied tiles.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate over all occupied tiles with their coordinates.
    /// Returns (tile_cx, tile_cy, &Tile) for each non-empty tile.
    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, &Tile)> {
        TileTableIter {
            table: self,
            mask_idx: 0,
            mask: self.occ_mask[0],
        }
    }

    /// Rebuild tiles from a HashMap of pieces.
    /// Used during initialization and recompute_piece_counts.
    pub fn rebuild_from_pieces<I>(&mut self, pieces: I)
    where
        I: IntoIterator<Item = ((i64, i64), Piece)>,
    {
        self.clear();
        for ((x, y), piece) in pieces {
            let (cx, cy) = tile_coords(x, y);
            let idx = local_index(x, y);
            let tile = self.get_or_create(cx, cy);
            tile.set_piece(idx, piece);
        }
    }

    /// Count total pieces across all tiles
    pub fn piece_count(&self) -> usize {
        self.buckets
            .iter()
            .filter(|b| b.state == BucketState::Occupied)
            .map(|b| b.tile.occ_all.count_ones() as usize)
            .sum()
    }
}

/// CTZ-based iterator over occupied buckets in the TileTable
struct TileTableIter<'a> {
    table: &'a TileTable,
    mask_idx: usize,
    mask: u64,
}

impl<'a> Iterator for TileTableIter<'a> {
    type Item = (i64, i64, &'a Tile);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.mask != 0 {
                let bit_idx = self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                let bucket_idx = self.mask_idx * 64 + bit_idx;
                let bucket = &self.table.buckets[bucket_idx];
                return Some((bucket.cx, bucket.cy, &bucket.tile));
            }

            self.mask_idx += 1;
            if self.mask_idx >= 8 {
                return None;
            }
            self.mask = self.table.occ_mask[self.mask_idx];
        }
    }
}

impl Tile {
    /// Iterate over all pieces in this tile.
    /// Yields (local_idx, Piece) for each occupied square using CTZ loop.
    #[inline]
    pub fn iter_pieces(&self) -> impl Iterator<Item = (usize, Piece)> + '_ {
        TilePieceIter {
            bits: self.occ_all,
            piece: &self.piece,
        }
    }
}

/// CTZ-based iterator over pieces in a tile
struct TilePieceIter<'a> {
    bits: u64,
    piece: &'a [u8; 64],
}

impl<'a> Iterator for TilePieceIter<'a> {
    type Item = (usize, Piece);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            return None;
        }
        let idx = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1; // Clear lowest bit
        let packed = self.piece[idx];
        if packed != 0 {
            Some((idx, Piece::from_packed(packed)))
        } else {
            self.next() // Skip if somehow piece array is inconsistent
        }
    }
}

// ============================================================================
// TileTable Fast Piece Iteration (for Evaluation)
// ============================================================================

impl TileTable {
    /// Iterate all pieces across all tiles.
    /// Yields (world_x, world_y, Piece) using CTZ bitboard iteration.
    /// Much faster than HashMap iteration for evaluation loops.
    #[inline]
    pub fn iter_all_pieces(&self) -> impl Iterator<Item = (i64, i64, Piece)> + '_ {
        TileTablePieceIter {
            table: self,
            bucket_mask_idx: 0,
            bucket_mask: self.occ_mask[0],
            current_bucket_idx: None,
            current_tile_bits: 0,
        }
    }

    /// Iterate pieces of a specific color (using color occupancy bitboard).
    #[inline]
    pub fn iter_pieces_by_color(
        &self,
        is_white: bool,
    ) -> impl Iterator<Item = (i64, i64, Piece)> + '_ {
        TileTableColorIter {
            table: self,
            is_white,
            bucket_mask_idx: 0,
            bucket_mask: self.occ_mask[0],
            current_bucket_idx: None,
            current_tile_bits: 0,
        }
    }
}

/// Iterator over all pieces in a TileTable
struct TileTablePieceIter<'a> {
    table: &'a TileTable,
    bucket_mask_idx: usize,
    bucket_mask: u64,
    current_bucket_idx: Option<usize>,
    current_tile_bits: u64,
}

impl<'a> Iterator for TileTablePieceIter<'a> {
    type Item = (i64, i64, Piece);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have bits in current tile, process them
            if self.current_tile_bits != 0 {
                let local_idx = self.current_tile_bits.trailing_zeros() as usize;
                self.current_tile_bits &= self.current_tile_bits - 1;

                if let Some(bucket_idx) = self.current_bucket_idx {
                    let bucket = &self.table.buckets[bucket_idx];
                    let packed = bucket.tile.piece[local_idx];
                    if packed != 0 {
                        let local_x = (local_idx % 8) as i64;
                        let local_y = (local_idx / 8) as i64;
                        let world_x = bucket.cx * TILE_SIZE + local_x;
                        let world_y = bucket.cy * TILE_SIZE + local_y;
                        return Some((world_x, world_y, Piece::from_packed(packed)));
                    }
                }
                continue;
            }

            // Find next occupied bucket
            while self.bucket_mask == 0 {
                self.bucket_mask_idx += 1;
                if self.bucket_mask_idx >= 8 {
                    return None;
                }
                self.bucket_mask = self.table.occ_mask[self.bucket_mask_idx];
            }

            // Get next bucket
            let bit_idx = self.bucket_mask.trailing_zeros() as usize;
            self.bucket_mask &= self.bucket_mask - 1;
            let bucket_idx = self.bucket_mask_idx * 64 + bit_idx;

            if bucket_idx < TILE_TABLE_CAPACITY {
                let bucket = &self.table.buckets[bucket_idx];
                if bucket.state == BucketState::Occupied {
                    self.current_bucket_idx = Some(bucket_idx);
                    self.current_tile_bits = bucket.tile.occ_all;
                }
            }
        }
    }
}

/// Iterator over pieces of a specific color
struct TileTableColorIter<'a> {
    table: &'a TileTable,
    is_white: bool,
    bucket_mask_idx: usize,
    bucket_mask: u64,
    current_bucket_idx: Option<usize>,
    current_tile_bits: u64,
}

impl<'a> Iterator for TileTableColorIter<'a> {
    type Item = (i64, i64, Piece);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have bits in current tile, process them
            if self.current_tile_bits != 0 {
                let local_idx = self.current_tile_bits.trailing_zeros() as usize;
                self.current_tile_bits &= self.current_tile_bits - 1;

                if let Some(bucket_idx) = self.current_bucket_idx {
                    let bucket = &self.table.buckets[bucket_idx];
                    let packed = bucket.tile.piece[local_idx];
                    if packed != 0 {
                        let local_x = (local_idx % 8) as i64;
                        let local_y = (local_idx / 8) as i64;
                        let world_x = bucket.cx * TILE_SIZE + local_x;
                        let world_y = bucket.cy * TILE_SIZE + local_y;
                        return Some((world_x, world_y, Piece::from_packed(packed)));
                    }
                }
                continue;
            }

            // Find next occupied bucket
            while self.bucket_mask == 0 {
                self.bucket_mask_idx += 1;
                if self.bucket_mask_idx >= 8 {
                    return None;
                }
                self.bucket_mask = self.table.occ_mask[self.bucket_mask_idx];
            }

            // Get next bucket
            let bit_idx = self.bucket_mask.trailing_zeros() as usize;
            self.bucket_mask &= self.bucket_mask - 1;
            let bucket_idx = self.bucket_mask_idx * 64 + bit_idx;

            if bucket_idx < TILE_TABLE_CAPACITY {
                let bucket = &self.table.buckets[bucket_idx];
                if bucket.state == BucketState::Occupied {
                    self.current_bucket_idx = Some(bucket_idx);
                    // Use color-specific occupancy
                    self.current_tile_bits = if self.is_white {
                        bucket.tile.occ_white
                    } else {
                        bucket.tile.occ_black
                    };
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Piece, PieceType, PlayerColor};

    #[test]
    fn test_tile_coords() {
        // Positive coordinates
        assert_eq!(tile_coord(0), 0);
        assert_eq!(tile_coord(7), 0);
        assert_eq!(tile_coord(8), 1);
        assert_eq!(tile_coord(15), 1);
        assert_eq!(tile_coord(16), 2);

        // Negative coordinates (arithmetic shift)
        assert_eq!(tile_coord(-1), -1);
        assert_eq!(tile_coord(-8), -1);
        assert_eq!(tile_coord(-9), -2);
    }

    #[test]
    fn test_local_coord() {
        // Positive
        assert_eq!(local_coord(0), 0);
        assert_eq!(local_coord(7), 7);
        assert_eq!(local_coord(8), 0);
        assert_eq!(local_coord(15), 7);

        // Negative (bitwise AND gives correct result)
        assert_eq!(local_coord(-1), 7);
        assert_eq!(local_coord(-8), 0);
        assert_eq!(local_coord(-9), 7);
    }

    #[test]
    fn test_local_index() {
        assert_eq!(local_index(0, 0), 0);
        assert_eq!(local_index(7, 0), 7);
        assert_eq!(local_index(0, 1), 8);
        assert_eq!(local_index(7, 7), 63);
    }

    #[test]
    fn test_neighbor_index() {
        assert_eq!(neighbor_index(-1, -1), 0);
        assert_eq!(neighbor_index(0, -1), 1);
        assert_eq!(neighbor_index(1, -1), 2);
        assert_eq!(neighbor_index(-1, 0), 3);
        assert_eq!(neighbor_index(0, 0), 4); // center
        assert_eq!(neighbor_index(1, 0), 5);
        assert_eq!(neighbor_index(-1, 1), 6);
        assert_eq!(neighbor_index(0, 1), 7);
        assert_eq!(neighbor_index(1, 1), 8);
    }

    #[test]
    fn test_tile_operations() {
        let mut tile = Tile::new();
        assert!(tile.is_empty());

        let piece = Piece::new(PieceType::Knight, PlayerColor::White);
        tile.set_piece(0, piece);

        assert!(!tile.is_empty());
        assert_eq!(tile.occ_all, 1);
        assert_eq!(tile.occ_white, 1);
        assert_eq!(tile.occ_black, 0);
        assert!(tile.get_piece(0).is_some());
        assert_eq!(tile.get_piece(0).unwrap().piece_type(), PieceType::Knight);

        tile.remove_piece(0);
        assert!(tile.is_empty());
        assert!(tile.get_piece(0).is_none());
    }

    #[test]
    fn test_tile_table_basic() {
        let mut table = TileTable::new();
        assert!(table.is_empty());

        // Create a tile
        let tile = table.get_or_create(0, 0);
        tile.set_piece(0, Piece::new(PieceType::Pawn, PlayerColor::White));

        assert_eq!(table.len(), 1);
        assert!(table.get_tile(0, 0).is_some());
        assert!(table.get_tile(1, 0).is_none());

        // Get same tile again
        let tile2 = table.get_or_create(0, 0);
        assert_eq!(tile2.get_piece(0).unwrap().piece_type(), PieceType::Pawn);
    }

    #[test]
    fn test_tile_table_negative_coords() {
        let mut table = TileTable::new();

        let tile = table.get_or_create(-1, -1);
        tile.set_piece(0, Piece::new(PieceType::King, PlayerColor::Black));

        assert!(table.get_tile(-1, -1).is_some());
        assert_eq!(
            table
                .get_tile(-1, -1)
                .unwrap()
                .get_piece(0)
                .unwrap()
                .piece_type(),
            PieceType::King
        );
    }

    #[test]
    fn test_tile_table_neighborhood() {
        let mut table = TileTable::new();

        // Create tiles at (0,0) and (1,0)
        table
            .get_or_create(0, 0)
            .set_piece(0, Piece::new(PieceType::Pawn, PlayerColor::White));
        table
            .get_or_create(1, 0)
            .set_piece(0, Piece::new(PieceType::Pawn, PlayerColor::Black));

        let neighborhood = table.get_neighborhood(0, 0);

        // Center tile (index 4) should exist
        assert!(neighborhood[4].is_some());
        // Right neighbor (index 5) should exist
        assert!(neighborhood[5].is_some());
        // Others should be None
        assert!(neighborhood[0].is_none());
        assert!(neighborhood[1].is_none());
    }

    #[test]
    fn test_rebuild_from_pieces() {
        let mut table = TileTable::new();

        let pieces = vec![
            ((0, 0), Piece::new(PieceType::King, PlayerColor::White)),
            ((8, 0), Piece::new(PieceType::Queen, PlayerColor::White)),
            ((-1, -1), Piece::new(PieceType::Knight, PlayerColor::Black)),
        ];

        table.rebuild_from_pieces(pieces);

        // (0,0) is in tile (0,0)
        assert!(table.get_tile(0, 0).is_some());
        // (8,0) is in tile (1,0)
        assert!(table.get_tile(1, 0).is_some());
        // (-1,-1) is in tile (-1,-1)
        assert!(table.get_tile(-1, -1).is_some());
    }
}
