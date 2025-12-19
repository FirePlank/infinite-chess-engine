use crate::tiles::{Tile, TileTable, local_index, tile_coords};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

// ============================================================================
// Number of piece types (for packed piece encoding)
// ============================================================================

/// Total number of piece types in the game (22 types: Void..Pawn)
pub const NUM_PIECE_TYPES: u8 = 22;

// ============================================================================
// Coordinate
// ============================================================================

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: i64,
    pub y: i64,
}

impl Coordinate {
    #[inline]
    pub fn new(x: i64, y: i64) -> Self {
        Coordinate { x, y }
    }

    #[inline]
    pub fn get_x(&self) -> i64 {
        self.x
    }

    #[inline]
    pub fn get_y(&self) -> i64 {
        self.y
    }
}

// ============================================================================
// PlayerColor
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PlayerColor {
    Neutral = 0,
    White = 1,
    Black = 2,
}

impl PlayerColor {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "n" | "neutral" => Some(PlayerColor::Neutral),
            "w" | "white" => Some(PlayerColor::White),
            "b" | "black" => Some(PlayerColor::Black),
            _ => None,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            PlayerColor::Neutral => "n",
            PlayerColor::White => "w",
            PlayerColor::Black => "b",
        }
    }

    #[inline]
    pub fn opponent(&self) -> Self {
        match self {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
            PlayerColor::Neutral => PlayerColor::Neutral,
        }
    }

    /// Convert from u8
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// ============================================================================
// PieceType
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PieceType {
    Void = 0,
    Obstacle = 1,
    King = 2,
    Giraffe = 3,
    Camel = 4,
    Zebra = 5,
    Knightrider = 6,
    Amazon = 7,
    Queen = 8,
    RoyalQueen = 9,
    Hawk = 10,
    Chancellor = 11,
    Archbishop = 12,
    Centaur = 13,
    RoyalCentaur = 14,
    Rose = 15,
    Knight = 16,
    Guard = 17,
    Huygen = 18,
    Rook = 19,
    Bishop = 20,
    Pawn = 21,
}

impl PieceType {
    /// Parse piece type from single-character string (matches JS typeutil)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v" => Some(PieceType::Void),
            "x" => Some(PieceType::Obstacle),
            "k" => Some(PieceType::King),
            "i" => Some(PieceType::Giraffe),
            "l" => Some(PieceType::Camel),
            "z" => Some(PieceType::Zebra),
            "s" => Some(PieceType::Knightrider),
            "m" => Some(PieceType::Amazon),
            "q" => Some(PieceType::Queen),
            "y" => Some(PieceType::RoyalQueen),
            "h" => Some(PieceType::Hawk),
            "c" => Some(PieceType::Chancellor),
            "a" => Some(PieceType::Archbishop),
            "e" => Some(PieceType::Centaur),
            "d" => Some(PieceType::RoyalCentaur),
            "o" => Some(PieceType::Rose),
            "n" => Some(PieceType::Knight),
            "g" => Some(PieceType::Guard),
            "u" => Some(PieceType::Huygen),
            "r" => Some(PieceType::Rook),
            "b" => Some(PieceType::Bishop),
            "p" => Some(PieceType::Pawn),
            _ => None,
        }
    }

    /// Convert piece type to single-character string
    pub fn to_str(&self) -> &'static str {
        match self {
            PieceType::Void => "v",
            PieceType::Obstacle => "x",
            PieceType::King => "k",
            PieceType::Giraffe => "i",
            PieceType::Camel => "l",
            PieceType::Zebra => "z",
            PieceType::Knightrider => "s",
            PieceType::Amazon => "m",
            PieceType::Queen => "q",
            PieceType::RoyalQueen => "y",
            PieceType::Hawk => "h",
            PieceType::Chancellor => "c",
            PieceType::Archbishop => "a",
            PieceType::Centaur => "e",
            PieceType::RoyalCentaur => "d",
            PieceType::Rose => "o",
            PieceType::Knight => "n",
            PieceType::Guard => "g",
            PieceType::Huygen => "u",
            PieceType::Rook => "r",
            PieceType::Bishop => "b",
            PieceType::Pawn => "p",
        }
    }

    /// Check if this piece type is a neutral/blocking type (can't be moved by players)
    #[inline]
    pub fn is_neutral_type(&self) -> bool {
        matches!(self, PieceType::Void | PieceType::Obstacle)
    }

    /// Check if this piece type is a royal (king-like) piece
    #[inline]
    pub fn is_royal(&self) -> bool {
        matches!(
            self,
            PieceType::King | PieceType::RoyalQueen | PieceType::RoyalCentaur
        )
    }

    /// Get all promotable piece types (for dynamic promotion)
    pub fn promotable_types() -> &'static [PieceType] {
        &[
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Chancellor,
            PieceType::Archbishop,
            PieceType::Amazon,
            PieceType::Hawk,
            PieceType::Guard,
            PieceType::Centaur,
            PieceType::Camel,
            PieceType::Giraffe,
            PieceType::Zebra,
            PieceType::Knightrider,
            PieceType::Rose,
            PieceType::Huygen,
        ]
    }

    /// Convert from u8
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// ============================================================================
// Piece - Packed representation (1 byte)
// ============================================================================

/// Packed piece representation: encodes both piece type and color in a single byte.
///
/// Encoding: `packed = color * NUM_PIECE_TYPES + piece_type`
/// - This matches the infinitechess.org JS encoding
/// - Allows 3 colors × 22 types = 66 values (fits in 7 bits)
///
/// For compatibility, we provide `piece_type()` and `color()` accessor methods
/// that decode the packed value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Piece(u8);

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece")
            .field("piece_type", &self.piece_type())
            .field("color", &self.color())
            .field("packed", &self.0)
            .finish()
    }
}

// Custom serde to preserve JSON compatibility with old format
impl Serialize for Piece {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Piece", 2)?;
        s.serialize_field("piece_type", &self.piece_type())?;
        s.serialize_field("color", &self.color())?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Piece {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PieceFields {
            piece_type: PieceType,
            color: PlayerColor,
        }
        let fields = PieceFields::deserialize(deserializer)?;
        Ok(Piece::new(fields.piece_type, fields.color))
    }
}

impl Piece {
    /// Create a new piece from type and color
    #[inline]
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece((color as u8) * NUM_PIECE_TYPES + (piece_type as u8))
    }

    /// Create a piece from a packed u8 value
    #[inline]
    pub fn from_packed(packed: u8) -> Self {
        Piece(packed)
    }

    /// Get the raw packed value
    #[inline]
    pub fn packed(&self) -> u8 {
        self.0
    }

    /// Decode the piece type
    #[inline]
    pub fn piece_type(&self) -> PieceType {
        PieceType::from_u8(self.0 % NUM_PIECE_TYPES)
    }

    /// Decode the color
    #[inline]
    pub fn color(&self) -> PlayerColor {
        PlayerColor::from_u8(self.0 / NUM_PIECE_TYPES)
    }
}

// ============================================================================
// Board
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "BoardRaw", into = "BoardRaw")]
pub struct Board {
    // HashMap for cold-path access and serialization
    pieces: FxHashMap<(i64, i64), Piece>,
    #[serde(skip)]
    pub active_coords: Option<FxHashSet<(i64, i64)>>,
    /// Sparse tiled bitboards for HOT PATH operations (attack detection, move gen, eval)
    /// NOT updated incrementally - call rebuild_tiles() before hot path usage
    #[serde(skip)]
    pub tiles: TileTable,
    /// Flag indicating tiles are in sync with pieces
    #[serde(skip)]
    tiles_valid: bool,
}

/// Raw representation for serialization
#[derive(Serialize, Deserialize)]
struct BoardRaw {
    pieces: FxHashMap<(i64, i64), Piece>,
}

impl From<BoardRaw> for Board {
    fn from(raw: BoardRaw) -> Self {
        let has_neutral = raw
            .pieces
            .values()
            .any(|p| p.piece_type().is_neutral_type());

        let active_coords = if has_neutral {
            let mut set = FxHashSet::default();
            for (pos, piece) in &raw.pieces {
                if !piece.piece_type().is_neutral_type() {
                    set.insert(*pos);
                }
            }
            Some(set)
        } else {
            None
        };

        // Build tiles immediately for hot path usage
        let mut tiles = TileTable::new();
        for (&(x, y), &piece) in &raw.pieces {
            let (cx, cy) = tile_coords(x, y);
            let idx = local_index(x, y);
            tiles.get_or_create(cx, cy).set_piece(idx, piece);
        }

        Board {
            pieces: raw.pieces,
            active_coords,
            tiles,
            tiles_valid: true,
        }
    }
}

impl From<Board> for BoardRaw {
    fn from(board: Board) -> Self {
        BoardRaw {
            pieces: board.pieces,
        }
    }
}

pub struct BoardIter<'a> {
    iter: std::collections::hash_map::Iter<'a, (i64, i64), Piece>,
}

impl<'a> Iterator for BoardIter<'a> {
    type Item = (&'a (i64, i64), &'a Piece);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for BoardIter<'_> {}

impl Board {
    pub fn new() -> Self {
        Board {
            pieces: FxHashMap::default(),
            active_coords: None,
            tiles: TileTable::new(),
            tiles_valid: true,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn iter(&self) -> BoardIter<'_> {
        BoardIter {
            iter: self.pieces.iter(),
        }
    }

    pub fn set_piece(&mut self, x: i64, y: i64, piece: Piece) {
        let pos = (x, y);

        // Active coords tracking
        if piece.piece_type().is_neutral_type() && self.active_coords.is_none() {
            // Need to initialize active_coords by scanning current pieces
            let mut set = FxHashSet::default();
            for (p_pos, p) in self.iter() {
                if !p.piece_type().is_neutral_type() {
                    set.insert(*p_pos);
                }
            }
            self.active_coords = Some(set);
        }

        self.pieces.insert(pos, piece);

        // INLINE TILE SYNC: Single fast update - set_piece handles overwrite
        let (cx, cy) = tile_coords(x, y);
        let idx = local_index(x, y);
        let tile = self.tiles.get_or_create(cx, cy);
        // Clear old occupancy bits first to handle overwrites correctly
        let bit = 1u64 << idx;
        tile.occ_all &= !bit;
        tile.occ_white &= !bit;
        tile.occ_black &= !bit;
        tile.occ_void &= !bit;
        // Set new piece
        tile.set_piece(idx, piece);

        if let Some(ref mut active) = self.active_coords {
            if !piece.piece_type().is_neutral_type() {
                active.insert(pos);
            } else {
                active.remove(&pos);
            }
        }
    }

    #[inline]
    pub fn get_piece(&self, x: &i64, y: &i64) -> Option<&Piece> {
        self.pieces.get(&(*x, *y))
    }

    /// BITBOARD: O(1) occupancy check using tile bitboards.
    /// Use this instead of get_piece().is_some() in hot paths.
    #[inline]
    pub fn is_occupied(&self, x: i64, y: i64) -> bool {
        let (cx, cy) = tile_coords(x, y);
        if let Some(tile) = self.tiles.get_tile(cx, cy) {
            let idx = local_index(x, y);
            (tile.occ_all >> idx) & 1 != 0
        } else {
            false
        }
    }

    /// BITBOARD: O(1) color-specific occupancy check.
    #[inline]
    pub fn is_occupied_by_color(&self, x: i64, y: i64, color: PlayerColor) -> bool {
        let (cx, cy) = tile_coords(x, y);
        if let Some(tile) = self.tiles.get_tile(cx, cy) {
            let idx = local_index(x, y);
            let occ = match color {
                PlayerColor::White => tile.occ_white,
                PlayerColor::Black => tile.occ_black,
                PlayerColor::Neutral => tile.occ_void,
            };
            (occ >> idx) & 1 != 0
        } else {
            false
        }
    }

    /// BITBOARD: O(1) piece retrieval using tile bitboards.
    /// Returns packed piece directly from tile array (no HashMap lookup).
    /// Use when you already know the square is occupied.
    #[inline]
    pub fn get_piece_fast(&self, x: i64, y: i64) -> Option<Piece> {
        let (cx, cy) = tile_coords(x, y);
        if let Some(tile) = self.tiles.get_tile(cx, cy) {
            let idx = local_index(x, y);
            if (tile.occ_all >> idx) & 1 != 0 {
                let packed = tile.piece[idx];
                if packed != 0 {
                    return Some(Piece::from_packed(packed));
                }
            }
        }
        None
    }

    pub fn remove_piece(&mut self, x: &i64, y: &i64) -> Option<Piece> {
        let pos = (*x, *y);
        let removed = self.pieces.remove(&pos);

        // INLINE TILE SYNC: Always keep tiles valid for hot path
        let (cx, cy) = tile_coords(*x, *y);
        let idx = local_index(*x, *y);
        if let Some(tile) = self.tiles.get_tile_mut(cx, cy) {
            tile.remove_piece(idx);
        }

        if let Some(ref piece) = removed {
            if let Some(ref mut active) = self.active_coords {
                if !piece.piece_type().is_neutral_type() {
                    active.remove(&pos);
                }
            }
        }
        removed
    }

    pub fn clear(&mut self) {
        self.pieces.clear();
        self.active_coords = None;
        self.tiles.clear();
        self.tiles_valid = true;
    }

    /// Rebuild tiles from HashMap. Call this ONCE before hot-path operations
    /// (attack detection, move generation, evaluation).
    /// This is O(n) where n = number of pieces, but only needs to be called
    /// once per position, not per operation.
    #[inline]
    pub fn rebuild_tiles(&mut self) {
        if self.tiles_valid {
            return;
        }
        self.tiles.clear();
        for (&(x, y), &piece) in &self.pieces {
            let (cx, cy) = tile_coords(x, y);
            let idx = local_index(x, y);
            self.tiles.get_or_create(cx, cy).set_piece(idx, piece);
        }
        self.tiles_valid = true;
    }

    /// Force rebuild tiles (for use after make_move/undo_move cycles)
    #[inline]
    pub fn ensure_tiles(&mut self) {
        self.rebuild_tiles();
    }

    /// Get 3x3 tile neighborhood around world coordinate.
    /// Caller should ensure tiles are valid via rebuild_tiles() first.
    #[inline]
    pub fn get_neighborhood(&self, x: i64, y: i64) -> [Option<&Tile>; 9] {
        let (cx, cy) = tile_coords(x, y);
        self.tiles.get_neighborhood(cx, cy)
    }

    /// Check if tiles are in sync with pieces
    #[inline]
    pub fn tiles_valid(&self) -> bool {
        self.tiles_valid
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_packing() {
        // Test all combinations of color and type
        for color in [PlayerColor::Neutral, PlayerColor::White, PlayerColor::Black] {
            for pt in [
                PieceType::Void,
                PieceType::Obstacle,
                PieceType::King,
                PieceType::Queen,
                PieceType::Rook,
                PieceType::Bishop,
                PieceType::Knight,
                PieceType::Pawn,
            ] {
                let piece = Piece::new(pt, color);
                assert_eq!(
                    piece.piece_type(),
                    pt,
                    "piece_type mismatch for {:?}/{:?}",
                    pt,
                    color
                );
                assert_eq!(
                    piece.color(),
                    color,
                    "color mismatch for {:?}/{:?}",
                    pt,
                    color
                );
            }
        }
    }

    #[test]
    fn test_piece_size() {
        assert_eq!(std::mem::size_of::<Piece>(), 1, "Piece should be 1 byte");
    }

    #[test]
    fn test_piece_packed_values() {
        // Test that packed values match JS encoding: color * NUM_TYPES + type
        let white_pawn = Piece::new(PieceType::Pawn, PlayerColor::White);
        assert_eq!(white_pawn.packed(), 1 * 22 + 21); // White=1, Pawn=21

        let black_king = Piece::new(PieceType::King, PlayerColor::Black);
        assert_eq!(black_king.packed(), 2 * 22 + 2); // Black=2, King=2
    }
}
