use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: i64,
    pub y: i64,
}

impl Coordinate {
    pub fn new(x: i64, y: i64) -> Self {
        Coordinate { x, y }
    }

    pub fn get_x(&self) -> i64 {
        self.x
    }

    pub fn get_y(&self) -> i64 {
        self.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerColor {
    Neutral,
    White,
    Black,
}

impl PlayerColor {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "n" => Some(PlayerColor::Neutral),
            "w" => Some(PlayerColor::White),
            "b" => Some(PlayerColor::Black),
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

    pub fn opponent(&self) -> Self {
        match self {
            PlayerColor::White => PlayerColor::Black,
            PlayerColor::Black => PlayerColor::White,
            PlayerColor::Neutral => PlayerColor::Neutral,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Void,
    Obstacle,
    King,
    Giraffe,
    Camel,
    Zebra,
    Knightrider,
    Amazon,
    Queen,
    RoyalQueen,
    Hawk,
    Chancellor,
    Archbishop,
    Centaur,
    RoyalCentaur,
    Rose,
    Knight,
    Guard,
    Huygen,
    Rook,
    Bishop,
    Pawn,
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
    pub fn is_neutral_type(&self) -> bool {
        matches!(self, PieceType::Void | PieceType::Obstacle)
    }

    /// Check if this piece type is a royal (king-like) piece
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: PlayerColor,
}

impl Piece {
    pub fn new(piece_type: PieceType, color: PlayerColor) -> Self {
        Piece { piece_type, color }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Board {
    pub pieces: HashMap<(i64, i64), Piece>,
}

impl Board {
    pub fn new() -> Self {
        Board {
            pieces: HashMap::new(),
        }
    }

    pub fn set_piece(&mut self, x: i64, y: i64, piece: Piece) {
        self.pieces.insert((x, y), piece);
    }

    pub fn get_piece(&self, x: &i64, y: &i64) -> Option<&Piece> {
        self.pieces.get(&(*x, *y))
    }

    pub fn remove_piece(&mut self, x: &i64, y: &i64) -> Option<Piece> {
        self.pieces.remove(&(*x, *y))
    }
}
