//! Line families as linear forms: the form (a, b) holds every line `a·x + b·y = c`. Two squares
//! share a line when their values under its form match, and order along it by any other form,
//! so a position's line geometry is fully described by its squares' values under each form.

use super::Square;
use super::bigint::Big;

/// Mop-up anchors its target boxes to an 8-square grid at the origin, so every square keeps its
/// residue mod 8.
pub const ORIGIN_GRID: i64 = 8;

/// The line family `a·x + b·y = c`, with `a` and `b` coprime and the first nonzero one positive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LineForm {
    pub a: i64,
    pub b: i64,
}

impl LineForm {
    /// The family of lines running along `(vx, vy)`.
    pub fn from_vector(vx: i64, vy: i64) -> LineForm {
        let g = gcd(vx, vy);
        let (a, b) = (vy / g, -vx / g);
        if a > 0 || (a == 0 && b > 0) {
            LineForm { a, b }
        } else {
            LineForm { a: -a, b: -b }
        }
    }

    /// Which line of the form holds `square`.
    pub fn value(&self, square: &Square) -> Big {
        self.value_of(&square.x, &square.y)
    }

    pub fn value_of(&self, x: &Big, y: &Big) -> Big {
        match (self.a, self.b) {
            (1, 0) => x.clone(),
            (0, 1) => y.clone(),
            (a, b) => &(x * a) + &(y * b),
        }
    }

    /// The value of a small offset.
    pub fn value_i(&self, dx: i64, dy: i64) -> i64 {
        self.a * dx + self.b * dy
    }

    /// The primitive step along the form's lines, first nonzero component positive.
    pub fn direction(&self) -> (i64, i64) {
        if self.b < 0 || (self.b == 0 && self.a > 0) {
            (-self.b, self.a)
        } else {
            (self.b, -self.a)
        }
    }

    /// Whether this is the file form (constant x) or rank form (constant y).
    pub fn is_file(&self) -> bool {
        self.b == 0
    }

    pub fn is_rank(&self) -> bool {
        self.a == 0
    }
}

/// Zero exactly when the two families are parallel.
pub fn determinant(f: LineForm, g: LineForm) -> i64 {
    f.a * g.b - g.a * f.b
}

/// The square where line `cf` of `f` crosses line `cg` of `g`, if they meet on a square.
pub fn crossing(f: LineForm, cf: &Big, g: LineForm, cg: &Big) -> Option<Square> {
    let det = determinant(f, g);
    if det == 0 {
        return None;
    }
    let x = &(cf * g.b) - &(cg * f.b);
    let y = &(cg * f.a) - &(cf * g.a);
    let ((qx, rx), (qy, ry)) = (x.div_rem(det), y.div_rem(det));
    (rx == 0 && ry == 0).then_some(Square { x: qx, y: qy })
}

/// The modulus every translation must be a multiple of so that, among `forms`, the same lines
/// still cross on whole squares, and those crossings keep their residue mod [`ORIGIN_GRID`].
pub fn residue_modulus(forms: &[LineForm]) -> i64 {
    let mut determinants = 1i64;
    for i in 0..forms.len() {
        for j in i + 1..forms.len() {
            let det = determinant(forms[i], forms[j]).abs();
            determinants = determinants / gcd(determinants, det) * det;
        }
    }
    ORIGIN_GRID * determinants
}

pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
