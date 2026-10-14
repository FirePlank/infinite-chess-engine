//! Maps the engine's moves on a compressed position back onto the original board. A capture lands
//! on its victim and a move short under every form keeps its offset; a long slide lands as near
//! every piece's line as the compressed landing does. Each landing is then checked against every
//! piece, so a line is only ever reported up to its last faithfully mapped move.

use super::bigint::Big;
use super::lineforms::{LineForm, ORIGIN_GRID};
use super::{EXACT_SPAN, NEAR_LINE_SPAN, OrderedMap, Square};
use crate::board::{PieceType, PlayerColor};
use crate::moves::Move;
use std::cell::OnceCell;
use std::sync::Arc;

/// A piece, where it stands on the original board and on the compressed one.
#[derive(Clone)]
pub struct LiftPiece {
    pub piece_type: PieceType,
    pub color: PlayerColor,
    pub original: Square,
    pub compressed: Square,
    /// Its squares' values under each form, original then compressed, computed on first use.
    values: OnceCell<Arc<(FormValues, FormValues)>>,
}

impl LiftPiece {
    pub fn new(
        piece_type: PieceType,
        color: PlayerColor,
        original: Square,
        compressed: Square,
    ) -> LiftPiece {
        LiftPiece {
            piece_type,
            color,
            original,
            compressed,
            values: OnceCell::new(),
        }
    }

    fn values(&self, forms: &[LineForm]) -> &(FormValues, FormValues) {
        self.values.get_or_init(|| {
            Arc::new((
                FormValues::of(forms, &self.original),
                FormValues::of(forms, &self.compressed),
            ))
        })
    }
}

/// The pieces in play, keyed by their compressed square, in the site's insertion order.
pub type LiftBoard = OrderedMap<Square, LiftPiece>;

/// A lone line (promotion rank, border edge, far-escape shell line) landings are also checked against.
#[derive(Clone)]
pub struct LiftLine {
    pub form: usize,
    pub original: Big,
    pub compressed: Big,
}

/// A square's value under each form, exactly and as a double for [`is_approx_near`].
#[derive(Clone)]
struct FormValues {
    exact: Vec<Big>,
    approx: Vec<f64>,
}

impl FormValues {
    fn of(forms: &[LineForm], square: &Square) -> FormValues {
        let exact: Vec<Big> = forms.iter().map(|f| f.value(square)).collect();
        let approx = exact.iter().map(Big::to_f64).collect();
        FormValues { exact, approx }
    }
}

/// Maps lines of moves on one compressed position back onto its original board.
#[derive(Clone)]
pub struct Lifter {
    forms: Vec<LineForm>,
    lines: Vec<LiftLine>,
    board: LiftBoard,
}

/// Bounds a double's rounding error relative to the values compared, with ample margin.
const APPROX_ERROR: f64 = 1.0 / (1u64 << 50) as f64;

impl Lifter {
    pub fn new(forms: Vec<LineForm>, lines: Vec<LiftLine>, board: LiftBoard) -> Lifter {
        Lifter {
            forms,
            lines,
            board,
        }
    }

    /// The longest prefix of an engine line, as site move tokens on the original board.
    pub fn lift_line(&self, line: &[Move]) -> Vec<String> {
        let mut board = self.board.clone();
        let mut lifted = Vec::with_capacity(line.len());
        for m in line {
            match self.lift_move(&mut board, m) {
                Some(token) => lifted.push(token),
                None => break,
            }
        }
        lifted
    }

    /// Lifts one move and plays it on `board`, or `None` if it can't be mapped faithfully.
    fn lift_move(&self, board: &mut LiftBoard, m: &Move) -> Option<String> {
        let start = Square::from_i64(m.from.x, m.from.y);
        let end = Square::from_i64(m.to.x, m.to.y);
        let piece = board.get(&start)?.clone();
        let victim = board.get(&end).cloned();
        let (dx, dy) = (
            m.to.x as i128 - m.from.x as i128,
            m.to.y as i128 - m.from.y as i128,
        );
        let is_short = self
            .forms
            .iter()
            .all(|f| (f.a as i128 * dx + f.b as i128 * dy).abs() <= EXACT_SPAN as i128);

        let end_original = match &victim {
            Some(v) => v.original.clone(),
            None if is_short => Square {
                x: &piece.original.x + &Big::from_i128(dx),
                y: &piece.original.y + &Big::from_i128(dy),
            },
            None => self.slide_landing(board, &piece, dx, dy)?,
        };
        if !self.keeps_relations(board, &piece, victim.as_ref(), &end_original, &end) {
            return None;
        }
        let promotion = m.promotion;
        play(board, &start, &end, &end_original, promotion);
        let mut token = format!(
            "{},{}>{},{}",
            piece.original.x, piece.original.y, end_original.x, end_original.y
        );
        if let Some(p) = promotion {
            let code = p.to_site_code();
            token.push('=');
            token.push_str(&if piece.color == PlayerColor::Black {
                code.to_lowercase()
            } else {
                code.to_string()
            });
        }
        Some(token)
    }

    /// Where a long slide lands on the original board. A landing near some piece's line (a
    /// crossing, or a wiggle off one) is pinned to the same offset from it, which compression
    /// keeps; a landing near no line keeps the compressed distance.
    fn slide_landing(
        &self,
        board: &LiftBoard,
        piece: &LiftPiece,
        dx: i128,
        dy: i128,
    ) -> Option<Square> {
        let steps = gcd128(dx, dy);
        let (ux, uy) = ((dx / steps) as i64, (dy / steps) as i64);
        let landing = FormValues::of(
            &self.forms,
            &Square {
                x: &piece.compressed.x + &Big::from_i128(dx),
                y: &piece.compressed.y + &Big::from_i128(dy),
            },
        );
        let start = &piece.values(&self.forms).0;
        let mut pinned: Option<Big> = None;
        let consistent = self.for_each_near(
            board,
            &[&piece.compressed],
            &landing,
            None,
            |f, original, compressed, _| {
                let along = self.forms[f].value_i(ux, uy);
                let compressed_offset = &landing.exact[f] - compressed;
                if along == 0 || compressed_offset.abs() > Big::from(NEAR_LINE_SPAN) {
                    return true;
                }
                let travel = &compressed_offset - &(&start.exact[f] - original);
                let (distance, remainder) = travel.div_rem(along);
                if remainder != 0 || pinned.as_ref().is_some_and(|p| *p != distance) {
                    return false;
                }
                pinned = Some(distance);
                true
            },
        );
        let distance = pinned.unwrap_or_else(|| Big::from_i128(steps));
        if !consistent || distance <= Big::zero() {
            return None;
        }
        Some(Square {
            x: &piece.original.x + &(&distance * ux),
            y: &piece.original.y + &(&distance * uy),
        })
    }

    /// Whether the original landing sits on the same residue, as near every other piece's line as
    /// the compressed one, and on the same side of every piece it shares a line with: nearness, and
    /// order along a line for blocks, pins and checks, are what the engine reads of a far landing.
    fn keeps_relations(
        &self,
        board: &LiftBoard,
        mover: &LiftPiece,
        victim: Option<&LiftPiece>,
        end_original: &Square,
        end: &Square,
    ) -> bool {
        let residue = |a: &Big, b: &Big| (a - b).div_rem(ORIGIN_GRID).1 == 0;
        if !residue(&end.x, &end_original.x) || !residue(&end.y, &end_original.y) {
            return false;
        }
        let compressed = FormValues::of(&self.forms, end);
        let original = FormValues::of(&self.forms, end_original);
        let near = Big::from(NEAR_LINE_SPAN);
        let mut excluded = vec![&mover.compressed];
        if let Some(v) = victim {
            excluded.push(&v.compressed);
        }
        self.for_each_near(
            board,
            &excluded,
            &compressed,
            Some(&original),
            |f, other_original, other_compressed, other| {
                let original_offset = &original.exact[f] - other_original;
                let compressed_offset = &compressed.exact[f] - other_compressed;
                let is_near = original_offset.abs() <= near || compressed_offset.abs() <= near;
                if is_near && original_offset != compressed_offset {
                    return false;
                }
                !original_offset.is_zero()
                    || other.is_none_or(|o| {
                        is_same_side_along(
                            self.forms[f],
                            end_original,
                            &o.original,
                            end,
                            &o.compressed,
                        )
                    })
            },
        )
    }

    /// Visits every other piece's and lone line's exact values under each form where its line may
    /// pass within the near-line span of the landing, until `visit` returns false. Doubles rule out
    /// the rest, so most pieces cost no exact arithmetic.
    fn for_each_near(
        &self,
        board: &LiftBoard,
        excluded: &[&Square],
        compressed: &FormValues,
        original: Option<&FormValues>,
        mut visit: impl FnMut(usize, &Big, &Big, Option<&LiftPiece>) -> bool,
    ) -> bool {
        let may_be_near = |f: usize, o: f64, c: f64| {
            is_approx_near(compressed.approx[f], c)
                || original.is_some_and(|v| is_approx_near(v.approx[f], o))
        };
        for (key, other) in board.iter() {
            if excluded.contains(&key) {
                continue;
            }
            let (ov, cv) = other.values(&self.forms);
            for f in 0..self.forms.len() {
                if may_be_near(f, ov.approx[f], cv.approx[f])
                    && !visit(f, &ov.exact[f], &cv.exact[f], Some(other))
                {
                    return false;
                }
            }
        }
        for line in &self.lines {
            if may_be_near(line.form, line.original.to_f64(), line.compressed.to_f64())
                && !visit(line.form, &line.original, &line.compressed, None)
            {
                return false;
            }
        }
        true
    }
}

/// Whether `a` lies the same way from `b` along their shared line of `form` as `a2` from `b2`.
fn is_same_side_along(form: LineForm, a: &Square, b: &Square, a2: &Square, b2: &Square) -> bool {
    // A file runs along y, every other line along x.
    if form.is_file() {
        (a.y > b.y) == (a2.y > b2.y)
    } else {
        (a.x > b.x) == (a2.x > b2.x)
    }
}

/// Whether two doubles may lie within the near-line span of each other, allowing for rounding.
/// Values beyond double range compare as infinite or NaN, which counts as near.
fn is_approx_near(a: f64, b: f64) -> bool {
    let bound = NEAR_LINE_SPAN as f64 + (a.abs() + b.abs()) * APPROX_ERROR + 1.0;
    (a - b).abs().partial_cmp(&bound) != Some(std::cmp::Ordering::Greater)
}

fn gcd128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Plays a move on both boards at once: the mover, its victim, a castling partner, an en passant
/// capture, a promotion. A move from an empty square is skipped, as the engine's replay skips it.
pub fn play(
    board: &mut LiftBoard,
    start: &Square,
    end: &Square,
    end_original: &Square,
    promotion: Option<PieceType>,
) {
    let Some(piece) = board.get(start).cloned() else {
        return;
    };
    let is_capture = board.contains_key(end);
    let dx = &end.x - &start.x;
    let dy_zero = end.y == start.y;
    let two = Big::from(2);
    if !is_capture && dy_zero && dx.abs() == two && piece.piece_type.can_castle() {
        move_partner(
            board,
            start,
            end,
            end_original,
            if dx.is_neg() { -1 } else { 1 },
        );
    } else if !is_capture && !dx.is_zero() && piece.piece_type == PieceType::Pawn {
        board.remove(&Square {
            x: end.x.clone(),
            y: start.y.clone(),
        }); // En passant
    }
    board.remove(start);
    let piece_type = promotion.unwrap_or(piece.piece_type);
    board.insert(
        end.clone(),
        LiftPiece::new(piece_type, piece.color, end_original.clone(), end.clone()),
    );
}

/// Castling: the nearest piece beyond the royal along its rank lands just behind its landing.
fn move_partner(
    board: &mut LiftBoard,
    start: &Square,
    end: &Square,
    end_original: &Square,
    step: i64,
) {
    let mut partner: Option<&LiftPiece> = None;
    for piece in board.values() {
        let c = &piece.compressed;
        if c.y != start.y || (&(&c.x - &start.x) * step) <= Big::zero() {
            continue;
        }
        if partner.is_none_or(|p| (&(&c.x - &p.compressed.x) * step).is_neg()) {
            partner = Some(piece);
        }
    }
    let Some(partner) = partner.cloned() else {
        return;
    };
    board.remove(&partner.compressed);
    let compressed = Square {
        x: &end.x - &Big::from(step),
        y: end.y.clone(),
    };
    let original = Square {
        x: &end_original.x - &Big::from(step),
        y: end_original.y.clone(),
    };
    board.insert(
        compressed.clone(),
        LiftPiece::new(partner.piece_type, partner.color, original, compressed),
    );
}
