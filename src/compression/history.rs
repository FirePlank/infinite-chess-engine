//! Splits a game's history squares by what the engine still reads of them. Positions before the
//! repetition window are never compared again, so a square only they use needs no geometry: just a
//! distinct square, rigid with its pawn-move partners. Everything else keeps full compression.

use super::bigint::Big;
use super::compressor::RANGE;
use super::icn::{Icn, IcnMove};
use super::lifter::{LiftBoard, play};
use super::{OrderedMap, Square, replay_start};
use crate::board::PieceType;
use rustc_hash::FxHashSet;

/// The history squares that keep full compression, and the rest in rigid groups.
pub struct HistorySplit {
    pub keep: FxHashSet<Square>,
    pub groups: Vec<Vec<Square>>,
}

/// The step between placeholder groups, and four times the widest group allowed.
const PLACEHOLDER_STEP: i64 = 1 << 16;

/// Keeps every square of every position from just before the window's last reset on, every
/// castling's rank, every promotion, the starting royals and the current position; groups the rest.
pub fn split(icn: &Icn, current: &LiftBoard) -> HistorySplit {
    let mut keep: FxHashSet<Square> = FxHashSet::default();
    let mut parent: OrderedMap<Square, Square> = OrderedMap::default();

    // Pass 1: the last ply that resets the repetition window, castlings and promotions.
    let (mut board, mut rights) = replay_start(icn);
    let mut last_reset = 0;
    for (i, m) in icn.moves.iter().enumerate() {
        let step = step_move(&mut board, &mut rights, m);
        if step.resets {
            last_reset = i + 1;
        }
        keep.extend(step.castling);
        if m.promotion.is_some() {
            keep.extend([m.from.clone(), m.to.clone()]);
        }
        if step.pawn {
            union(&mut parent, &m.from, &m.to);
        }
        if let Some(victim) = step.ep_victim {
            union(&mut parent, &m.from, &victim);
        }
    }
    // Pass 2: every square of every position from just before that ply on.
    let window_start = last_reset.saturating_sub(1);
    let (mut board, mut rights) = replay_start(icn);
    for m in &icn.moves[..window_start] {
        step_move(&mut board, &mut rights, m);
    }
    keep.extend(board.keys().cloned());
    for m in &icn.moves[window_start..] {
        keep.extend([m.from.clone(), m.to.clone()]);
    }
    keep.extend(
        icn.pieces
            .iter()
            .filter(|p| p.piece_type.is_royal())
            .map(|p| p.square.clone()),
    );
    keep.extend(current.keys().cloned());

    let mut history: OrderedMap<Square, ()> = OrderedMap::default();
    for s in icn
        .pieces
        .iter()
        .map(|p| &p.square)
        .chain(icn.moves.iter().flat_map(|m| [&m.from, &m.to]))
    {
        history.insert(s.clone(), ());
    }
    let mut by_root: OrderedMap<Square, Vec<Square>> = OrderedMap::default();
    let members: Vec<Square> = history.keys().chain(parent.keys()).cloned().collect();
    for k in members {
        let root = find(&parent, &k);
        if by_root.get(&root).is_none() {
            by_root.insert(root.clone(), Vec::new());
        }
        let group = by_root.get_mut(&root).unwrap();
        if !group.contains(&k) {
            group.push(k);
        }
    }
    let mut groups = Vec::new();
    for group in by_root.values() {
        if group.iter().any(|k| keep.contains(k)) {
            keep.extend(group.iter().cloned());
        } else {
            groups.push(group.clone());
        }
    }
    HistorySplit { keep, groups }
}

fn find(parent: &OrderedMap<Square, Square>, k: &Square) -> Square {
    let mut r = k;
    while let Some(p) = parent.get(r).filter(|p| *p != r) {
        r = p;
    }
    r.clone()
}

fn union(parent: &mut OrderedMap<Square, Square>, a: &Square, b: &Square) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent.insert(ra, rb);
    }
}

struct Step {
    resets: bool,
    pawn: bool,
    ep_victim: Option<Square>,
    castling: Vec<Square>,
}

/// Plays one move, reporting what the engine's replay reads from it: whether it resets the
/// repetition window, and the squares a castling or en passant capture relates.
fn step_move(board: &mut LiftBoard, rights: &mut FxHashSet<Square>, m: &IcnMove) -> Step {
    let Some(piece) = board.get(&m.from) else {
        return Step {
            resets: true,
            pawn: false,
            ep_victim: None,
            castling: Vec::new(),
        };
    };
    let piece_type = piece.piece_type;
    let dx = &m.to.x - &m.from.x;
    let is_capture = board.contains_key(&m.to);
    let pawn = piece_type == PieceType::Pawn;
    let ep_victim = (pawn && !is_capture && !dx.is_zero()).then(|| Square {
        x: m.to.x.clone(),
        y: m.from.y.clone(),
    });
    let mut lost = rights.remove(&m.from);
    if is_capture {
        lost |= rights.remove(&m.to);
    }
    let mut castling = Vec::new();
    if m.to.y == m.from.y && dx.abs() == Big::from(2) && piece_type.can_castle() {
        let step = if dx.is_neg() { -1 } else { 1 };
        castling.extend([
            m.from.clone(),
            m.to.clone(),
            Square {
                x: &m.to.x - &Big::from(step),
                y: m.to.y.clone(),
            },
        ]);
        let mut partner: Option<Square> = None;
        for other in board.values().filter(|o| o.original.y == m.from.y) {
            castling.push(other.original.clone());
            let beyond = (&(&other.original.x - &m.to.x) * step) > Big::zero();
            if beyond
                && partner
                    .as_ref()
                    .is_none_or(|p| (&(&other.original.x - &p.x) * step).is_neg())
            {
                partner = Some(other.original.clone());
            }
        }
        if let Some(p) = partner {
            lost |= rights.remove(&p);
        }
    }
    let promotion = m
        .promotion
        .as_deref()
        .and_then(PieceType::parse_promotion_code);
    play(board, &m.from, &m.to, &m.to, promotion);
    Step {
        resets: pawn || is_capture || ep_victim.is_some() || lost,
        pawn,
        ep_victim,
        castling,
    }
}

/// Each placeholder group on its own square past every kept point and inside `right` and `top`,
/// each group offset diagonally from the last so none shares a rank or file with a kept point.
pub fn place(
    groups: &[Vec<Square>],
    kept: &[Square],
    right: &Big,
    top: &Big,
) -> Option<Vec<(Square, Square)>> {
    if groups.is_empty() {
        return Some(Vec::new());
    }
    let step = Big::from(PLACEHOLDER_STEP);
    let floor = Big::from(-RANGE);
    let max_x = kept
        .iter()
        .map(|s| &s.x)
        .max()
        .unwrap_or(&floor)
        .max(&floor);
    let max_y = kept
        .iter()
        .map(|s| &s.y)
        .max()
        .unwrap_or(&floor)
        .max(&floor);
    let (x0, y0) = (max_x + &step, max_y + &step);
    let far = &step * (groups.len() as i64 + 1);
    if &(&x0 + &far) >= right || &(&y0 + &far) >= top {
        return None;
    }
    let widest = Big::from(PLACEHOLDER_STEP / 4);
    let mut placed = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        let anchor = &group[0];
        let base = &step * g as i64;
        for square in group {
            let (ox, oy) = (&square.x - &anchor.x, &square.y - &anchor.y);
            if ox.abs() >= widest || oy.abs() >= widest {
                return None;
            }
            placed.push((
                square.clone(),
                Square {
                    x: &(&x0 + &base) + &ox,
                    y: &(&y0 + &base) + &oy,
                },
            ));
        }
    }
    Some(placed)
}
