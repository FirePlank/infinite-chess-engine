//! Brings a position whose coordinates lie beyond the engine's reach within it, and maps the
//! engine's moves back. An ICN already in range passes through untouched. One further out is
//! compressed together with its move history, so repetitions still register: under every line
//! form the engine reads, relations within [`EXACT_SPAN`] stay exact and farther ones keep their
//! order, so the engine perceives exactly what it would in the original.

mod bigint;
mod clustering;
mod compressor;
mod crossings;
mod history;
mod icn;
mod lifter;
mod lineforms;
mod placement;
mod simplex;
mod skeleton;
#[cfg(test)]
mod tests;

pub use bigint::Big;
pub use lifter::Lifter;

use crate::board::{PieceType, PlayerColor};
use crate::moves::{FAR_SHELL_MAX, FAR_SHELL_MIN, PLAY_BORDER_CAP, QUIET_RAY_CAP_RADIUS};
use crate::search::tt_defs::MAX_TT_COORD;
use clustering::{CompressionPoint, LoneLine};
use compressor::CompressionInput;
use crossings::Slider;
use icn::Icn;
use lifter::{LiftBoard, LiftLine, LiftPiece};
use lineforms::LineForm;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::Hash;

/// Past the furthest the engine reads a distance between two squares: its TT's move box around
/// the origin, a capped ray's quiet candidates (at most 5 radii under any form), and every eval
/// radius (Amazon tropism, at 1788, is the widest).
const EXACT_SPAN: i64 = 1 << 14;
const _: () = assert!(EXACT_SPAN > 2 * (MAX_TT_COORD + 1) && EXACT_SPAN > 5 * QUIET_RAY_CAP_RADIUS);

/// How near a third piece's line a far crossing must pass for the engine to notice: no line offset
/// it reads exceeds 216, and slider wiggles 2, leaving room for pieces to drift during a search.
const NEAR_LINE_SPAN: i64 = 4096;

/// A square with unbounded coordinates.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Square {
    pub x: Big,
    pub y: Big,
}

impl Square {
    pub fn from_i64(x: i64, y: i64) -> Square {
        Square {
            x: Big::from(x),
            y: Big::from(y),
        }
    }
}

/// A position as the engine receives it, and how to map its moves back.
pub struct Compressed {
    pub icn: String,
    pub lifter: Lifter,
}

/// Why a position can't be handed to the engine.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A Huygen reads exact distances at any range, which no compression keeps.
    ReadsExactDistances,
    /// Its orders can't all be kept within the engine's coordinates.
    Uncompressible,
}

/// `icn` within the engine's ±`limit` coordinates: `None` when it already is, else compressed.
pub fn prepare(icn_text: &str, limit: &Big) -> Result<Option<Compressed>, Refusal> {
    let Some(icn) = Icn::parse(icn_text) else {
        return Ok(None);
    };
    if icn.is_within(limit) {
        return Ok(None);
    }
    let current = replay(&icn);
    let mut types: Vec<PieceType> = Vec::new();
    let promotion_types = icn
        .promotion
        .iter()
        .filter_map(|s| s.pieces.as_deref())
        .flat_map(|p| p.split(','));
    for t in promotion_types
        .filter_map(PieceType::parse_promotion_code)
        .chain(icn.pieces.iter().map(|p| p.piece_type))
        .chain(current.values().map(|p| p.piece_type))
    {
        if !types.contains(&t) {
            types.push(t);
        }
    }
    if types.contains(&PieceType::Huygen) {
        return Err(Refusal::ReadsExactDistances);
    }
    let forms = forms_for(&types);
    if is_inert(&icn, &current, &forms, limit) {
        return Ok(None);
    }
    let split = history::split(&icn, &current);
    let attempt = match compress_with(&icn, &current, &forms, limit, Some(&split)) {
        Some(a) if a.placeholders.is_some() => Some(a),
        Some(_) => compress_with(&icn, &current, &forms, limit, None),
        None => None,
    };
    let attempt = attempt.ok_or(Refusal::Uncompressible)?;
    Ok(Some(attempt.finish(&icn, &current, &forms)))
}

/// Whether every point chains to the origin within the exact span under every form. Then every
/// point welds into the origin's cluster, which keeps its coordinates, so compression is the identity.
fn is_inert(icn: &Icn, current: &LiftBoard, forms: &[LineForm], limit: &Big) -> bool {
    let set = collect_points(icn, current, forms, limit, None);
    let span = Big::from(EXACT_SPAN);
    let origin = Square {
        x: Big::zero(),
        y: Big::zero(),
    };
    (0..forms.len()).all(|f| {
        let mut values: Vec<Big> = set
            .points
            .iter()
            .filter(|p| p.line.is_none_or(|l| l.form == f))
            .map(|p| forms[f].value(&p.coords))
            .collect();
        values.push(forms[f].value(&origin));
        values.sort_unstable();
        values.windows(2).all(|w| &w[1] - &w[0] <= span)
    })
}

/// One compression of the kept points, and the history's placeholders (`None` when they don't fit).
struct Attempt {
    point_set: PointSet,
    compressed: Vec<Square>,
    placeholders: Option<FxHashMap<Square, Square>>,
    limit: Big,
}

/// The points a compression keeps, and where each square and lone line sits among them.
struct PointSet {
    points: Vec<CompressionPoint>,
    perceived: usize,
    squares: FxHashMap<Square, usize>,
    lines: OrderedMap<(usize, Big), usize>,
}

fn compress_with(
    icn: &Icn,
    current: &LiftBoard,
    forms: &[LineForm],
    limit: &Big,
    split: Option<&history::HistorySplit>,
) -> Option<Attempt> {
    let point_set = collect_points(icn, current, forms, limit, split.map(|s| &s.keep));
    let input = CompressionInput {
        points: point_set.points.clone(),
        perceived: point_set.perceived,
        forms: forms.to_vec(),
        sliders: find_sliders(current, &point_set, forms),
    };
    let compressed = compressor::compress(&input)?;
    let mut attempt = Attempt {
        point_set,
        compressed,
        placeholders: Some(FxHashMap::default()),
        limit: limit.clone(),
    };
    if let Some(split) = split {
        let kept: Vec<Square> = attempt
            .point_set
            .points
            .iter()
            .zip(&attempt.compressed)
            .filter(|(p, _)| p.line.is_none())
            .map(|(_, c)| c.clone())
            .collect();
        let range = Big::from(compressor::RANGE);
        let edge = |i: usize| {
            icn.border
                .as_ref()
                .and_then(|b| b[i].as_ref())
                .map_or(range.clone(), |v| {
                    attempt.edge(forms, i, Some(v)).min(range.clone())
                })
        };
        attempt.placeholders = history::place(&split.groups, &kept, &edge(1), &edge(3))
            .map(|p| p.into_iter().collect());
    }
    Some(attempt)
}

impl Attempt {
    fn square(&self, s: &Square) -> Square {
        self.placeholders
            .as_ref()
            .and_then(|p| p.get(s))
            .cloned()
            .unwrap_or_else(|| self.compressed[self.point_set.squares[s]].clone())
    }

    fn line(&self, forms: &[LineForm], form: usize, value: &Big) -> Option<Big> {
        let p = *self.point_set.lines.get(&(form, value.clone()))?;
        Some(forms[form].value(&self.compressed[p]))
    }

    /// A border edge (left, right, bottom, top): compressed when collected, else clamped to the
    /// engine's own bound, which an edge beyond the limit is no nearer than.
    fn edge(&self, forms: &[LineForm], i: usize, value: Option<&Big>) -> Big {
        let cap = Big::from(PLAY_BORDER_CAP);
        let low_side = i.is_multiple_of(2);
        let form = if i < 2 {
            file_form(forms)
        } else {
            rank_form(forms)
        };
        match value {
            None => {
                if low_side {
                    -&cap
                } else {
                    cap
                }
            }
            Some(v) => self
                .line(forms, form, v)
                .unwrap_or_else(|| v.clone().min(cap.clone()).max(-&cap)),
        }
    }

    fn finish(self, icn: &Icn, current: &LiftBoard, forms: &[LineForm]) -> Compressed {
        let rank = rank_form(forms);
        let text = icn.write(
            &|s| self.square(s),
            // A "never promotes" rank was never collected, so it stays as written.
            &|r| self.line(forms, rank, r).unwrap_or_else(|| r.clone()),
            &|i, v| self.edge(forms, i, v),
        );
        let lines = self
            .point_set
            .lines
            .iter()
            .map(|((form, _), &p)| LiftLine {
                form: *form,
                original: forms[*form].value(&self.point_set.points[p].coords),
                compressed: forms[*form].value(&self.compressed[p]),
            })
            .collect();
        let mut board = LiftBoard::default();
        for piece in current.values() {
            let compressed = self.square(&piece.original);
            board.insert(
                compressed.clone(),
                LiftPiece::new(
                    piece.piece_type,
                    piece.color,
                    piece.original.clone(),
                    compressed,
                ),
            );
        }
        let _ = &self.limit;
        Compressed {
            icn: text,
            lifter: Lifter::new(forms.to_vec(), lines, board),
        }
    }
}

/// The start position, each piece on its original square on both boards, and its special rights.
fn replay_start(icn: &Icn) -> (LiftBoard, FxHashSet<Square>) {
    let mut board = LiftBoard::default();
    let mut rights = FxHashSet::default();
    for p in &icn.pieces {
        board.insert(
            p.square.clone(),
            LiftPiece::new(p.piece_type, p.color, p.square.clone(), p.square.clone()),
        );
        if p.rights {
            rights.insert(p.square.clone());
        }
    }
    (board, rights)
}

/// The position after the ICN's moves, the board the engine searches from.
fn replay(icn: &Icn) -> LiftBoard {
    let (mut board, _) = replay_start(icn);
    for m in &icn.moves {
        let promotion = m
            .promotion
            .as_deref()
            .and_then(PieceType::parse_promotion_code);
        lifter::play(&mut board, &m.from, &m.to, &m.to, promotion);
    }
    board
}

/// The directions each piece slides along, as the site's movesets list them.
fn sliding_vectors(piece_type: PieceType) -> &'static [(i64, i64)] {
    const ORTHO: [(i64, i64); 2] = [(1, 0), (0, 1)];
    const DIAG: [(i64, i64); 2] = [(1, 1), (1, -1)];
    const BOTH: [(i64, i64); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
    const KNIGHTRIDER: [(i64, i64); 4] = [(1, 2), (1, -2), (2, 1), (2, -1)];
    match piece_type {
        PieceType::Rook | PieceType::Chancellor | PieceType::Huygen => &ORTHO,
        PieceType::Bishop | PieceType::Archbishop => &DIAG,
        PieceType::Queen | PieceType::RoyalQueen | PieceType::Amazon => &BOTH,
        PieceType::Knightrider => &KNIGHTRIDER,
        _ => &[],
    }
}

/// Files, ranks and both diagonals, which the evaluation reads whatever the pieces, then every
/// further form a present piece slides along.
fn forms_for(types: &[PieceType]) -> Vec<LineForm> {
    let base = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut forms: Vec<LineForm> = Vec::new();
    for (vx, vy) in base.into_iter().chain(
        types
            .iter()
            .flat_map(|&t| sliding_vectors(t).iter().copied()),
    ) {
        let form = LineForm::from_vector(vx, vy);
        if !forms.contains(&form) {
            forms.push(form);
        }
    }
    forms
}

fn file_form(forms: &[LineForm]) -> usize {
    forms.iter().position(LineForm::is_file).unwrap()
}

fn rank_form(forms: &[LineForm]) -> usize {
    forms.iter().position(LineForm::is_rank).unwrap()
}

/// First what the engine reads lines through: the board after the moves, the en passant squares,
/// and each real border edge, promotion rank and far-escape shell line as a lone line. Then the
/// history squares `keep` admits (all of them without it), which the engine only replays.
fn collect_points(
    icn: &Icn,
    current: &LiftBoard,
    forms: &[LineForm],
    limit: &Big,
    keep: Option<&FxHashSet<Square>>,
) -> PointSet {
    let mut set = PointSet {
        points: Vec::new(),
        perceived: 0,
        squares: FxHashMap::default(),
        lines: OrderedMap::default(),
    };
    let add_square = |set: &mut PointSet, s: &Square| {
        if !set.squares.contains_key(s) {
            set.squares.insert(s.clone(), set.points.len());
            set.points.push(CompressionPoint {
                coords: s.clone(),
                line: None,
            });
        }
    };
    let add_line = |set: &mut PointSet, form: usize, value: &Big, boundary: bool| {
        if set.lines.get(&(form, value.clone())).is_none() {
            set.lines.insert((form, value.clone()), set.points.len());
            let coords = if forms[form].a == 0 {
                Square {
                    x: Big::zero(),
                    y: value.clone(),
                }
            } else {
                Square {
                    x: value.clone(),
                    y: Big::zero(),
                }
            };
            set.points.push(CompressionPoint {
                coords,
                line: Some(LoneLine { form, boundary }),
            });
        }
    };

    for piece in current.values() {
        add_square(&mut set, &piece.original);
    }
    if let Some(ep) = &icn.en_passant {
        add_square(&mut set, ep);
        let pawn_y = &ep.y
            + &Big::from(if icn.turn == PlayerColor::White {
                -1
            } else {
                1
            });
        add_square(
            &mut set,
            &Square {
                x: ep.x.clone(),
                y: pawn_y,
            },
        );
    }
    let (file, rank) = (file_form(forms), rank_form(forms));
    if let Some(border) = &icn.border {
        for (i, edge) in border.iter().enumerate() {
            if let Some(e) = edge.as_ref().filter(|e| e.abs() < *limit) {
                add_line(&mut set, if i < 2 { file } else { rank }, e, true);
            }
        }
    }
    for r in icn.promotion_ranks() {
        add_line(&mut set, rank, r, false);
    }
    for v in [FAR_SHELL_MIN, FAR_SHELL_MAX] {
        add_line(&mut set, file, &Big::from(v), false);
        add_line(&mut set, rank, &Big::from(v), false);
    }
    set.perceived = set.points.len();

    let kept = |s: &Square| keep.is_none_or(|k| k.contains(s));
    for s in icn
        .pieces
        .iter()
        .map(|p| &p.square)
        .chain(icn.moves.iter().flat_map(|m| [&m.from, &m.to]))
    {
        if kept(s) {
            add_square(&mut set, s);
        }
    }
    set
}

/// Every piece that slides, with the forms of the lines it slides along.
fn find_sliders(current: &LiftBoard, set: &PointSet, forms: &[LineForm]) -> Vec<Slider> {
    current
        .values()
        .filter(|p| !sliding_vectors(p.piece_type).is_empty())
        .map(|p| Slider {
            point: set.squares[&p.original],
            forms: sliding_vectors(p.piece_type)
                .iter()
                .map(|&(vx, vy)| {
                    forms
                        .iter()
                        .position(|f| *f == LineForm::from_vector(vx, vy))
                        .unwrap()
                })
                .collect(),
        })
        .collect()
}

/// A map that iterates in insertion order, where re-inserting a key keeps its place (as a JS `Map`
/// does), so every iteration matches the site's compressor exactly.
#[derive(Clone)]
pub struct OrderedMap<K, V> {
    entries: Vec<Option<(K, V)>>,
    index: FxHashMap<K, usize>,
}

impl<K, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        OrderedMap {
            entries: Vec::new(),
            index: FxHashMap::default(),
        }
    }
}

impl<K: Hash + Eq + Clone, V> OrderedMap<K, V> {
    pub fn get(&self, k: &K) -> Option<&V> {
        self.index
            .get(k)
            .map(|&i| &self.entries[i].as_ref().unwrap().1)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let i = *self.index.get(k)?;
        Some(&mut self.entries[i].as_mut().unwrap().1)
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.index.contains_key(k)
    }

    pub fn insert(&mut self, k: K, v: V) {
        match self.index.get(&k) {
            Some(&i) => self.entries[i].as_mut().unwrap().1 = v,
            None => {
                self.index.insert(k.clone(), self.entries.len());
                self.entries.push(Some((k, v)));
            }
        }
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let i = self.index.remove(k)?;
        self.entries[i].take().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().flatten().map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }
}
