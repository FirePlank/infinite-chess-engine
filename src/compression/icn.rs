//! Reads the coordinates out of an ICN with arbitrarily large integers, and writes it back with
//! them replaced. Tokens are classified exactly as `GameState::setup_position_from_icn` classifies
//! them; everything that carries no coordinate is passed through verbatim.

use super::Square;
use super::bigint::Big;
use crate::board::{PieceType, PlayerColor};

/// A piece as the ICN writes it.
pub struct IcnPiece {
    /// The site code as written; its case gives the color.
    pub code: String,
    pub piece_type: PieceType,
    pub color: PlayerColor,
    pub square: Square,
    pub rights: bool,
}

/// A move of the history. `promotion` is the code as written.
#[derive(Clone)]
pub struct IcnMove {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<String>,
}

/// One side's promotion rule: its ranks, and its piece list as written.
pub struct PromotionSide {
    pub ranks: Vec<Big>,
    pub pieces: Option<String>,
}

enum Token {
    Raw(String),
    Pieces,
    EnPassant,
    Border,
    Promotion,
    Moves,
}

/// An ICN's coordinate-bearing parts, and enough of the rest to write it back.
pub struct Icn {
    tags: String,
    tokens: Vec<Token>,
    pub turn: PlayerColor,
    pub pieces: Vec<IcnPiece>,
    pub en_passant: Option<Square>,
    /// left, right, bottom, top; `None` for an unbounded (`_`) edge.
    pub border: Option<[Option<Big>; 4]>,
    pub promotion: Vec<PromotionSide>,
    pub moves: Vec<IcnMove>,
}

impl Icn {
    /// Parses `icn`, or `None` when it carries no explicit position.
    pub fn parse(icn: &str) -> Option<Icn> {
        let mut content = icn.trim();
        let mut tags = String::new();
        while content.starts_with('[') {
            let Some(end) = content.find(']') else { break };
            tags.push_str(&content[..=end]);
            tags.push(' ');
            content = content[end + 1..].trim();
        }
        let raw: Vec<&str> = content.split_whitespace().collect();
        let mut out = Icn {
            tags,
            tokens: Vec::new(),
            turn: PlayerColor::White,
            pieces: Vec::new(),
            en_passant: None,
            border: None,
            promotion: Vec::new(),
            moves: Vec::new(),
        };
        if raw.len() == 1 && !raw[0].contains('>') {
            out.pieces = parse_pieces(raw[0]);
            out.tokens.push(Token::Pieces);
            return Some(out);
        }
        for token in raw {
            let classified = if token == "-" {
                None
            } else if token.contains('>') || token.contains('x') {
                out.moves.extend(
                    token
                        .split('|')
                        .filter(|m| !m.is_empty())
                        .filter_map(parse_move),
                );
                Some(Token::Moves)
            } else if token == "w" || token == "b" {
                out.turn = if token == "w" {
                    PlayerColor::White
                } else {
                    PlayerColor::Black
                };
                None
            } else if (token.contains('/') && token.starts_with(|c: char| c.is_ascii_digit()))
                || (token.starts_with('(')
                    && token.ends_with(')')
                    && token[1..].starts_with(|c: char| c.is_ascii_alphabetic()))
            {
                None // Clocks, or win conditions.
            } else if token.starts_with('(') && token.ends_with(')') {
                out.promotion = parse_promotion(token);
                Some(Token::Promotion)
            } else if let Some(border) = parse_border(token) {
                out.border = Some(border);
                Some(Token::Border)
            } else if token.contains('|')
                || (token.contains(',') && token.chars().any(|c| c.is_ascii_uppercase()))
            {
                out.pieces = parse_pieces(token);
                Some(Token::Pieces)
            } else if token.contains(',')
                && token.split(',').count() == 2
                && !token.chars().any(|c| c.is_ascii_alphabetic())
            {
                let (x, y) = token.split_once(',').unwrap();
                out.en_passant = Big::parse(x)
                    .zip(Big::parse(y))
                    .map(|(x, y)| Square { x, y });
                Some(Token::EnPassant)
            } else {
                None
            };
            // A token class that repeats keeps only its last occurrence, as the engine does.
            if let Some(t) = classified {
                let same = |o: &Token| std::mem::discriminant(o) == std::mem::discriminant(&t);
                if !matches!(t, Token::Moves) || !out.tokens.iter().any(same) {
                    out.tokens.retain(|o| !same(o));
                    out.tokens.push(t);
                }
            } else {
                out.tokens.push(Token::Raw(token.to_string()));
            }
        }
        (!out.pieces.is_empty()).then_some(out)
    }

    /// Every coordinate the engine reads lies within ±`limit`.
    pub fn is_within(&self, limit: &Big) -> bool {
        let within = |v: &Big| v.abs() <= *limit;
        let square = |s: &Square| within(&s.x) && within(&s.y);
        self.pieces.iter().all(|p| square(&p.square))
            && self.moves.iter().all(|m| square(&m.from) && square(&m.to))
            && self.en_passant.as_ref().is_none_or(square)
            && self.promotion_ranks().all(within)
    }

    /// Every rank a side promotes on. The engine writes i64::MIN or i64::MAX for a side that never
    /// promotes, which is no line at all.
    pub fn promotion_ranks(&self) -> impl Iterator<Item = &Big> {
        let none = [Big::from(i64::MIN), Big::from(i64::MAX)];
        self.promotion
            .iter()
            .flat_map(|s| &s.ranks)
            .filter(move |r| !none.contains(r))
    }

    /// The ICN with every square mapped by `square`, every rank by `rank`, every border edge by
    /// `edge` (index: left, right, bottom, top), and the moves replaced by `moves`.
    pub fn write(
        &self,
        square: &dyn Fn(&Square) -> Square,
        rank: &dyn Fn(&Big) -> Big,
        edge: &dyn Fn(usize, Option<&Big>) -> Big,
    ) -> String {
        let mut out = self.tags.clone();
        for token in &self.tokens {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            match token {
                Token::Raw(s) => out.push_str(s),
                Token::Pieces => {
                    let entries: Vec<String> = self
                        .pieces
                        .iter()
                        .map(|p| {
                            let s = square(&p.square);
                            format!(
                                "{}{},{}{}",
                                p.code,
                                s.x,
                                s.y,
                                if p.rights { "+" } else { "" }
                            )
                        })
                        .collect();
                    out.push_str(&entries.join("|"));
                }
                Token::EnPassant => {
                    if let Some(ep) = &self.en_passant {
                        let s = square(ep);
                        out.push_str(&format!("{},{}", s.x, s.y));
                    }
                }
                Token::Border => {
                    let b = self.border.as_ref().unwrap();
                    let edges: Vec<String> =
                        (0..4).map(|i| edge(i, b[i].as_ref()).to_string()).collect();
                    out.push_str(&edges.join(","));
                }
                Token::Promotion => {
                    let sides: Vec<String> = self
                        .promotion
                        .iter()
                        .map(|side| {
                            let ranks: Vec<String> =
                                side.ranks.iter().map(|r| rank(r).to_string()).collect();
                            match &side.pieces {
                                Some(p) => format!("{};{}", ranks.join(","), p),
                                None => ranks.join(","),
                            }
                        })
                        .collect();
                    out.push_str(&format!("({})", sides.join("|")));
                }
                Token::Moves => {
                    let moves: Vec<String> = self
                        .moves
                        .iter()
                        .map(|m| {
                            let (f, t) = (square(&m.from), square(&m.to));
                            let promotion = m
                                .promotion
                                .as_ref()
                                .map_or(String::new(), |p| format!("={p}"));
                            format!("{},{}>{},{}{}", f.x, f.y, t.x, t.y, promotion)
                        })
                        .collect();
                    out.push_str(&moves.join("|"));
                }
            }
        }
        out.trim_end().to_string()
    }
}

/// The pieces segment, split as `parse_icn_pieces` splits it; malformed entries are dropped.
fn parse_pieces(segment: &str) -> Vec<IcnPiece> {
    segment
        .split('|')
        .filter_map(|def| {
            let (info, y_raw) = def.split_once(',').filter(|(_, y)| !y.contains(','))?;
            let split = info
                .find(|c: char| c.is_ascii_digit() || c == '-')
                .unwrap_or(info.len());
            let (code, x_raw) = info.split_at(split);
            if code.is_empty() {
                return None;
            }
            let (x_str, rx) = x_raw
                .strip_suffix('+')
                .map_or((x_raw, false), |s| (s, true));
            let (y_str, ry) = y_raw
                .strip_suffix('+')
                .map_or((y_raw, false), |s| (s, true));
            let piece_type = PieceType::from_site_code(&code.to_uppercase());
            let color = if piece_type.is_neutral_type() {
                PlayerColor::Neutral
            } else if code.starts_with(|c: char| c.is_uppercase()) {
                PlayerColor::White
            } else {
                PlayerColor::Black
            };
            Some(IcnPiece {
                code: code.to_string(),
                piece_type,
                color,
                square: Square {
                    x: Big::parse(x_str)?,
                    y: Big::parse(y_str)?,
                },
                rights: rx || ry,
            })
        })
        .collect()
}

/// One move, read as the engine's replay reads it.
fn parse_move(token: &str) -> Option<IcnMove> {
    let clean = token.trim_start_matches(|c: char| c.is_alphabetic() && c != '>' && c != 'x');
    let (from, to) = if clean.contains('>') {
        clean.split_once('>')?
    } else {
        clean.split_once('x')?
    };
    let (to, promotion) = match to.find('=') {
        Some(i) => (&to[..i], Some(to[i + 1..].to_string())),
        None => (to, None),
    };
    let coord = |s: &str| -> Option<Square> {
        let (x, y) = s.split_once(',')?;
        let y = y.trim_matches(|c| c == '+' || c == '!' || c == '#' || c == '?');
        Some(Square {
            x: Big::parse(x)?,
            y: Big::parse(y)?,
        })
    };
    Some(IcnMove {
        from: coord(from)?,
        to: coord(to)?,
        promotion,
    })
}

/// `(ranks;pieces|ranks;pieces)`, ranks comma-separated.
fn parse_promotion(token: &str) -> Vec<PromotionSide> {
    token[1..token.len() - 1]
        .split('|')
        .map(|side| {
            let (ranks, pieces) = match side.split_once(';') {
                Some((r, p)) => (r, Some(p.to_string())),
                None => (side, None),
            };
            PromotionSide {
                ranks: ranks.split(',').filter_map(Big::parse).collect(),
                pieces,
            }
        })
        .collect()
}

/// `left,right,bottom,top`, each an integer or `_` for unbounded.
fn parse_border(token: &str) -> Option<[Option<Big>; 4]> {
    let parts: Vec<&str> = token.split(',').collect();
    if parts.len() != 4
        || !token
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '-' || c == '_')
    {
        return None;
    }
    let mut edges: [Option<Big>; 4] = Default::default();
    for (i, part) in parts.iter().enumerate() {
        edges[i] = if *part == "_" {
            None
        } else {
            Some(Big::parse(part)?)
        };
    }
    Some(edges)
}
