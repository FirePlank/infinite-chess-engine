use crate::tiles::{Tile, TileTable, local_index, tile_coords};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

/// Total number of piece types in the game (Void..Pawn)
pub const NUM_PIECE_TYPES: u8 = 22;

// Coordinate

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

// PlayerColor

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PlayerColor {
    Neutral = 0,
    White = 1,
    Black = 2,
}

impl std::str::FromStr for PlayerColor {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "n" | "neutral" => Ok(PlayerColor::Neutral),
            "w" | "white" => Ok(PlayerColor::White),
            "b" | "black" => Ok(PlayerColor::Black),
            _ => Err(()),
        }
    }
}

impl PlayerColor {
    pub fn get_char(&self) -> char {
        match self {
            PlayerColor::Neutral => 'n',
            PlayerColor::White => 'w',
            PlayerColor::Black => 'b',
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

    #[inline(always)]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// PieceType

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

impl std::str::FromStr for PieceType {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "v" => Ok(PieceType::Void),
            "x" => Ok(PieceType::Obstacle),
            "k" => Ok(PieceType::King),
            "i" => Ok(PieceType::Giraffe),
            "l" => Ok(PieceType::Camel),
            "z" => Ok(PieceType::Zebra),
            "s" => Ok(PieceType::Knightrider),
            "m" => Ok(PieceType::Amazon),
            "q" => Ok(PieceType::Queen),
            "y" => Ok(PieceType::RoyalQueen),
            "h" => Ok(PieceType::Hawk),
            "c" => Ok(PieceType::Chancellor),
            "a" => Ok(PieceType::Archbishop),
            "e" => Ok(PieceType::Centaur),
            "d" => Ok(PieceType::RoyalCentaur),
            "o" => Ok(PieceType::Rose),
            "n" => Ok(PieceType::Knight),
            "g" => Ok(PieceType::Guard),
            "u" => Ok(PieceType::Huygen),
            "r" => Ok(PieceType::Rook),
            "b" => Ok(PieceType::Bishop),
            "p" => Ok(PieceType::Pawn),
            _ => Err(()),
        }
    }
}

impl PieceType {
    pub fn get_char(&self) -> char {
        match self {
            PieceType::Void => 'v',
            PieceType::Obstacle => 'x',
            PieceType::King => 'k',
            PieceType::Giraffe => 'i',
            PieceType::Camel => 'l',
            PieceType::Zebra => 'z',
            PieceType::Knightrider => 's',
            PieceType::Amazon => 'm',
            PieceType::Queen => 'q',
            PieceType::RoyalQueen => 'y',
            PieceType::Hawk => 'h',
            PieceType::Chancellor => 'c',
            PieceType::Archbishop => 'a',
            PieceType::Centaur => 'e',
            PieceType::RoyalCentaur => 'd',
            PieceType::Rose => 'o',
            PieceType::Knight => 'n',
            PieceType::Guard => 'g',
            PieceType::Huygen => 'u',
            PieceType::Rook => 'r',
            PieceType::Bishop => 'b',
            PieceType::Pawn => 'p',
        }
    }

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

    /// Convert piece type to the site's two-letter code
    pub fn to_site_code(&self) -> &'static str {
        match self {
            PieceType::Void => "VO",
            PieceType::Obstacle => "OB",
            PieceType::King => "K",
            PieceType::Giraffe => "GI",
            PieceType::Camel => "CA",
            PieceType::Zebra => "ZE",
            PieceType::Knightrider => "NR",
            PieceType::Amazon => "AM",
            PieceType::Queen => "Q",
            PieceType::RoyalQueen => "RQ",
            PieceType::Hawk => "HA",
            PieceType::Chancellor => "CH",
            PieceType::Archbishop => "AR",
            PieceType::Centaur => "CE",
            PieceType::RoyalCentaur => "RC",
            PieceType::Rose => "RO",
            PieceType::Knight => "N",
            PieceType::Guard => "GU",
            PieceType::Huygen => "HU",
            PieceType::Rook => "R",
            PieceType::Bishop => "B",
            PieceType::Pawn => "P",
        }
    }

    pub fn from_site_code(code: &str) -> Self {
        match code {
            "VO" => PieceType::Void,
            "OB" => PieceType::Obstacle,
            "K" => PieceType::King,
            "GI" => PieceType::Giraffe,
            "CA" => PieceType::Camel,
            "ZE" => PieceType::Zebra,
            "NR" => PieceType::Knightrider,
            "AM" => PieceType::Amazon,
            "Q" => PieceType::Queen,
            "RQ" => PieceType::RoyalQueen,
            "HA" => PieceType::Hawk,
            "CH" => PieceType::Chancellor,
            "AR" => PieceType::Archbishop,
            "CE" => PieceType::Centaur,
            "RC" => PieceType::RoyalCentaur,
            "RO" => PieceType::Rose,
            "N" => PieceType::Knight,
            "GU" => PieceType::Guard,
            "HU" => PieceType::Huygen,
            "R" => PieceType::Rook,
            "B" => PieceType::Bishop,
            "P" => PieceType::Pawn,
            _ => PieceType::Void,
        }
    }

    /// Check if this piece type is a neutral/blocking type (cannot be moved by players)
    #[inline]
    pub fn is_neutral_type(&self) -> bool {
        matches!(self, PieceType::Void | PieceType::Obstacle)
    }

    /// Check if this piece type is truly uncapturable (Void blocks and cannot be taken).
    /// Obstacles CAN be captured despite being neutral.
    #[inline]
    pub fn is_uncapturable(&self) -> bool {
        matches!(self, PieceType::Void)
    }

    #[inline]
    pub fn is_royal(&self) -> bool {
        matches!(
            self,
            PieceType::King | PieceType::RoyalQueen | PieceType::RoyalCentaur
        )
    }

    /// Get all promotable piece types for dynamic promotion
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

    #[inline(always)]
    pub fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v) }
    }
}

// Piece - Packed representation (1 byte)

/// Packed piece representation: encodes both piece type and color in a single byte.
///
/// Encoding: `packed = color * NUM_PIECE_TYPES + piece_type`
/// Matches the infinitechess.org JS encoding.
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
    #[inline]
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece((color as u8) * NUM_PIECE_TYPES + (piece_type as u8))
    }

    #[inline(always)]
    pub fn from_packed(packed: u8) -> Self {
        Piece(packed)
    }

    #[inline]
    pub fn packed(&self) -> u8 {
        self.0
    }

    #[inline(always)]
    pub fn piece_type(&self) -> PieceType {
        PieceType::from_u8(self.0 % NUM_PIECE_TYPES)
    }

    #[inline(always)]
    pub fn color(&self) -> PlayerColor {
        PlayerColor::from_u8(self.0 / NUM_PIECE_TYPES)
    }
}

// ============================================================================
// Board
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "BoardRaw", into = "BoardRaw")]
pub struct Board {
    #[serde(skip)]
    pub tiles: TileTable,
    #[serde(skip)]
    pub piece_count: usize,
    #[serde(skip)]
    pub active_coords: Option<FxHashSet<(i64, i64)>>,
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

        let active_coords = has_neutral.then_some({
            let mut set = FxHashSet::default();
            for (pos, piece) in &raw.pieces {
                if !piece.piece_type().is_neutral_type() {
                    set.insert(*pos);
                }
            }
            set
        });

        // Build tiles and count pieces
        let mut tiles = TileTable::new();
        let mut piece_count = 0;
        for (&(x, y), &piece) in &raw.pieces {
            let (cx, cy) = tile_coords(x, y);
            let idx = local_index(x, y);
            tiles.get_or_create(cx, cy).set_piece(idx, piece);
            piece_count += 1;
        }

        Board {
            tiles,
            piece_count,
            active_coords,
        }
    }
}

impl From<Board> for BoardRaw {
    fn from(board: Board) -> Self {
        BoardRaw {
            pieces: board.iter().map(|(x, y, p)| ((x, y), p)).collect(),
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            active_coords: None,
            tiles: TileTable::new(),
            piece_count: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.piece_count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.piece_count == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, Piece)> + '_ {
        self.tiles.iter_all_pieces()
    }

    /// Fast bitboard-based iteration of pieces by color.
    #[inline]
    pub fn iter_pieces_by_color(
        &self,
        is_white: bool,
    ) -> impl Iterator<Item = (i64, i64, Piece)> + '_ {
        self.tiles.iter_pieces_by_color(is_white)
    }

    /// Iterate all pieces on the board using bitboards.
    #[inline]
    pub fn iter_all_pieces(&self) -> impl Iterator<Item = (i64, i64, Piece)> + '_ {
        self.tiles.iter_all_pieces()
    }

    pub fn set_piece(&mut self, x: i64, y: i64, piece: Piece) {
        let (cx, cy) = tile_coords(x, y);
        let idx = local_index(x, y);
        let tile = self.tiles.get_or_create(cx, cy);

        // Check if we are overwriting an existing piece
        let old_piece = tile.get_piece(idx);

        // Update bitmasks
        let bit = 1u64 << idx;
        tile.occ_all &= !bit;
        tile.occ_white &= !bit;
        tile.occ_black &= !bit;
        tile.occ_void &= !bit;

        tile.set_piece(idx, piece);

        // Update count
        if old_piece.is_none() {
            self.piece_count += 1;
        }

        // Active coords tracking
        if piece.piece_type().is_neutral_type() && self.active_coords.is_none() {
            let mut set = FxHashSet::default();

            // Iterate ALL pieces from tiles to populate set
            for (px, py, p) in self.tiles.iter_all_pieces() {
                if !p.piece_type().is_neutral_type() {
                    set.insert((px, py));
                }
            }
            self.active_coords = Some(set);
        }

        if let Some(ref mut active) = self.active_coords {
            if !piece.piece_type().is_neutral_type() {
                active.insert((x, y));
            } else {
                active.remove(&(x, y));
            }
        }
    }

    #[inline]
    pub fn get_piece(&self, x: i64, y: i64) -> Option<Piece> {
        let (cx, cy) = tile_coords(x, y);
        if let Some(tile) = self.tiles.get_tile(cx, cy) {
            let idx = local_index(x, y);
            tile.get_piece(idx)
        } else {
            None
        }
    }

    /// BITBOARD: O(1) occupancy check using tile bitboards.
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

    pub fn remove_piece(&mut self, x: &i64, y: &i64) -> Option<Piece> {
        let (cx, cy) = tile_coords(*x, *y);
        let idx = local_index(*x, *y);

        let mut removed = None;
        if let Some(tile) = self.tiles.get_tile_mut(cx, cy) {
            removed = tile.get_piece(idx);
            if removed.is_some() {
                tile.remove_piece(idx);
                self.piece_count -= 1;
            }
        }

        if let (Some(active), Some(p)) = (self.active_coords.as_mut(), removed)
            && !p.piece_type().is_neutral_type()
        {
            active.remove(&(*x, *y));
        }
        removed
    }

    pub fn clear(&mut self) {
        self.active_coords = None;
        self.tiles.clear();
        self.piece_count = 0;
    }

    /// Get 3x3 tile neighborhood around world coordinate.
    /// Caller should ensure tiles are valid via rebuild_tiles() first.
    #[inline]
    pub fn get_neighborhood(&self, x: i64, y: i64) -> [Option<&Tile>; 9] {
        let (cx, cy) = tile_coords(x, y);
        self.tiles.get_neighborhood(cx, cy)
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
        assert_eq!(white_pawn.packed(), 22 + 21); // White=1, Pawn=21

        let black_king = Piece::new(PieceType::King, PlayerColor::Black);
        assert_eq!(black_king.packed(), 2 * 22 + 2); // Black=2, King=2
    }

    #[test]
    fn test_coordinate() {
        let c = Coordinate::new(5, 10);
        assert_eq!(c.x, 5);
        assert_eq!(c.y, 10);
        assert_eq!(c.get_x(), 5);
        assert_eq!(c.get_y(), 10);
    }

    #[test]
    fn test_player_color_from_str() {
        assert_eq!(
            "white".parse::<PlayerColor>().ok(),
            Some(PlayerColor::White)
        );
        assert_eq!(
            "black".parse::<PlayerColor>().ok(),
            Some(PlayerColor::Black)
        );
        assert_eq!(
            "neutral".parse::<PlayerColor>().ok(),
            Some(PlayerColor::Neutral)
        );
        assert_eq!("invalid".parse::<PlayerColor>().ok(), None);
    }

    #[test]
    fn test_player_color_to_str() {
        // to_str returns the exact string that from_str accepts
        let w = PlayerColor::White.to_str();
        let b = PlayerColor::Black.to_str();
        let n = PlayerColor::Neutral.to_str();
        assert!(!w.is_empty());
        assert!(!b.is_empty());
        assert!(!n.is_empty());
    }

    #[test]
    fn test_player_color_opponent() {
        assert_eq!(PlayerColor::White.opponent(), PlayerColor::Black);
        assert_eq!(PlayerColor::Black.opponent(), PlayerColor::White);
        assert_eq!(PlayerColor::Neutral.opponent(), PlayerColor::Neutral);
    }

    #[test]
    fn test_player_color_from_u8() {
        assert_eq!(PlayerColor::from_u8(0), PlayerColor::Neutral);
        assert_eq!(PlayerColor::from_u8(1), PlayerColor::White);
        assert_eq!(PlayerColor::from_u8(2), PlayerColor::Black);
    }

    #[test]
    fn test_piece_type_from_str() {
        assert_eq!("k".parse::<PieceType>().ok(), Some(PieceType::King));
        assert_eq!("q".parse::<PieceType>().ok(), Some(PieceType::Queen));
        assert_eq!("r".parse::<PieceType>().ok(), Some(PieceType::Rook));
        assert_eq!("b".parse::<PieceType>().ok(), Some(PieceType::Bishop));
        assert_eq!("n".parse::<PieceType>().ok(), Some(PieceType::Knight));
        assert_eq!("p".parse::<PieceType>().ok(), Some(PieceType::Pawn));
        assert_eq!("x".parse::<PieceType>().ok(), Some(PieceType::Obstacle));
        assert_eq!("v".parse::<PieceType>().ok(), Some(PieceType::Void));
        assert_eq!("invalid".parse::<PieceType>().ok(), None);
    }

    #[test]
    fn test_piece_type_to_str() {
        assert_eq!(PieceType::King.to_str(), "k");
        assert_eq!(PieceType::Queen.to_str(), "q");
        assert_eq!(PieceType::Rook.to_str(), "r");
        assert_eq!(PieceType::Bishop.to_str(), "b");
        assert_eq!(PieceType::Knight.to_str(), "n");
        assert_eq!(PieceType::Pawn.to_str(), "p");
    }

    #[test]
    fn test_piece_type_to_site_code() {
        assert_eq!(PieceType::King.to_site_code(), "K");
        assert_eq!(PieceType::Queen.to_site_code(), "Q");
        assert_eq!(PieceType::Amazon.to_site_code(), "AM");
        assert_eq!(PieceType::Chancellor.to_site_code(), "CH");
    }

    #[test]
    fn test_piece_type_is_neutral_type() {
        assert!(PieceType::Void.is_neutral_type());
        assert!(PieceType::Obstacle.is_neutral_type());
        assert!(!PieceType::King.is_neutral_type());
        assert!(!PieceType::Pawn.is_neutral_type());
    }

    #[test]
    fn test_piece_type_is_uncapturable() {
        assert!(PieceType::Void.is_uncapturable());
        assert!(!PieceType::Obstacle.is_uncapturable());
        assert!(!PieceType::King.is_uncapturable());
    }

    #[test]
    fn test_piece_type_is_royal() {
        assert!(PieceType::King.is_royal());
        assert!(PieceType::RoyalQueen.is_royal());
        assert!(PieceType::RoyalCentaur.is_royal());
        assert!(!PieceType::Queen.is_royal());
        assert!(!PieceType::Pawn.is_royal());
    }

    #[test]
    fn test_piece_type_promotable_types() {
        let types = PieceType::promotable_types();
        assert!(types.contains(&PieceType::Queen));
        assert!(types.contains(&PieceType::Rook));
        assert!(types.contains(&PieceType::Bishop));
        assert!(types.contains(&PieceType::Knight));
        assert!(!types.contains(&PieceType::Pawn));
        assert!(!types.contains(&PieceType::King));
    }

    #[test]
    fn test_board_new() {
        let board = Board::new();
        assert!(board.is_empty());
        assert_eq!(board.len(), 0);
    }

    #[test]
    fn test_board_set_get_remove() {
        let mut board = Board::new();
        let piece = Piece::new(PieceType::King, PlayerColor::White);

        board.set_piece(5, 5, piece);
        assert_eq!(board.len(), 1);
        assert!(!board.is_empty());

        let got = board.get_piece(5, 5);
        assert!(got.is_some());
        assert_eq!(got.unwrap().piece_type(), PieceType::King);

        let removed = board.remove_piece(&5, &5);
        assert!(removed.is_some());
        assert_eq!(board.len(), 0);
    }

    #[test]
    fn test_board_is_occupied() {
        let mut board = Board::new();
        board.set_piece(3, 4, Piece::new(PieceType::Pawn, PlayerColor::White));

        assert!(board.is_occupied(3, 4));
        assert!(!board.is_occupied(3, 5));
    }

    #[test]
    fn test_board_is_occupied_by_color() {
        let mut board = Board::new();
        board.set_piece(3, 4, Piece::new(PieceType::Pawn, PlayerColor::White));
        board.set_piece(5, 6, Piece::new(PieceType::Pawn, PlayerColor::Black));

        assert!(board.is_occupied_by_color(3, 4, PlayerColor::White));
        assert!(!board.is_occupied_by_color(3, 4, PlayerColor::Black));
        assert!(board.is_occupied_by_color(5, 6, PlayerColor::Black));
    }

    #[test]
    fn test_board_clear() {
        let mut board = Board::new();
        board.set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        board.set_piece(2, 2, Piece::new(PieceType::Queen, PlayerColor::Black));
        assert_eq!(board.len(), 2);

        board.clear();
        assert!(board.is_empty());
        assert_eq!(board.len(), 0);
    }

    #[test]
    fn test_board_iter() {
        let mut board = Board::new();
        board.set_piece(1, 1, Piece::new(PieceType::King, PlayerColor::White));
        board.set_piece(8, 8, Piece::new(PieceType::King, PlayerColor::Black));

        let count = board.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_piece_from_packed() {
        let original = Piece::new(PieceType::Queen, PlayerColor::Black);
        let packed = original.packed();
        let reconstructed = Piece::from_packed(packed);

        assert_eq!(reconstructed.piece_type(), PieceType::Queen);
        assert_eq!(reconstructed.color(), PlayerColor::Black);
    }
}
