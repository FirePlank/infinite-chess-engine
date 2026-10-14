//! A dense-tableau two-phase simplex over nonnegative variables: a line-for-line port of the
//! continuous path of yalps 0.6.4, so layouts match the site's solver exactly.

use rustc_hash::FxHashMap;

/// A minimization over nonnegative variables.
pub struct Model {
    /// (key, lower, upper) per constraint, in key order; key 0 is the objective.
    pub constraints: Vec<(u32, f64, f64)>,
    /// Each variable's (constraint key, coefficient) terms.
    pub variables: Vec<Vec<(u32, f64)>>,
}

const MAX_PIVOTS: usize = 8192;

struct Tableau {
    m: Vec<f64>,
    width: usize,
    height: usize,
    pos_of_var: Vec<u32>,
    var_at_pos: Vec<u32>,
    nonzero: Vec<usize>,
}

impl Tableau {
    #[inline(always)]
    fn at(&self, r: usize, c: usize) -> f64 {
        self.m[r * self.width + c]
    }

    #[inline(always)]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        let w = self.width;
        self.m[r * w + c] = v;
    }
}

/// Each variable's optimal value, or `None` when the model is infeasible or the solver stalls.
pub fn solve(model: &Model, precision: f64) -> Option<Vec<f64>> {
    // The objective's sign flips a minimization into the maximization the tableau runs.
    const SIGN: f64 = -1.0;
    let mut rows = FxHashMap::default();
    let mut height = 1usize;
    for &(key, lower, upper) in &model.constraints {
        rows.insert(key, (height, lower, upper));
        height += lower.is_finite() as usize + upper.is_finite() as usize;
    }
    let width = model.variables.len() + 1;
    let total = width + height;
    let mut t = Tableau {
        m: vec![0.0; width * height],
        width,
        height,
        pos_of_var: (0..total as u32).collect(),
        var_at_pos: (0..total as u32).collect(),
        nonzero: Vec::with_capacity(width),
    };
    for c in 1..width {
        for &(key, coef) in &model.variables[c - 1] {
            if key == 0 {
                t.set(0, c, SIGN * coef);
            }
            if let Some(&(row, lower, upper)) = rows.get(&key) {
                if upper.is_finite() {
                    t.set(row, c, coef);
                    if lower.is_finite() {
                        t.set(row + 1, c, -coef);
                    }
                } else if lower.is_finite() {
                    t.set(row, c, -coef);
                }
            }
        }
    }
    for &(row, lower, upper) in rows.values() {
        if upper.is_finite() {
            t.set(row, 0, upper);
            if lower.is_finite() {
                t.set(row + 1, 0, -lower);
            }
        } else if lower.is_finite() {
            t.set(row, 0, -lower);
        }
    }

    if !phase1(&mut t, precision) {
        return None;
    }
    Some(
        (0..model.variables.len())
            .map(|i| {
                let pos = t.pos_of_var[i + 1] as usize;
                let v = if pos >= width {
                    t.at(pos - width, 0)
                } else {
                    0.0
                };
                if v > precision {
                    round_to_precision(v, precision)
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

fn round_to_precision(num: f64, precision: f64) -> f64 {
    let rounding = js_round(1.0 / precision);
    js_round((num + f64::EPSILON) * rounding) / rounding
}

/// JS's `Math.round`: half rounds toward positive infinity.
pub fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 { f + 1.0 } else { f }
}

fn pivot(t: &mut Tableau, row: usize, col: usize) {
    let quotient = t.at(row, col);
    let leaving = t.var_at_pos[t.width + row];
    let entering = t.var_at_pos[col];
    t.var_at_pos[t.width + row] = entering;
    t.var_at_pos[col] = leaving;
    t.pos_of_var[leaving as usize] = col as u32;
    t.pos_of_var[entering as usize] = (t.width + row) as u32;

    let mut nonzero = std::mem::take(&mut t.nonzero);
    nonzero.clear();
    for c in 0..t.width {
        let value = t.at(row, c);
        if value.abs() > 1e-16 {
            t.set(row, c, value / quotient);
            nonzero.push(c);
        } else {
            t.set(row, c, 0.0);
        }
    }
    t.set(row, col, 1.0 / quotient);
    let w = t.width;
    let (before, rest) = t.m.split_at_mut(row * w);
    let (pivot_row, after) = rest.split_at_mut(w);
    for other in before.chunks_exact_mut(w).chain(after.chunks_exact_mut(w)) {
        let coef = other[col];
        if coef.abs() > 1e-16 {
            for &c in &nonzero {
                other[c] -= coef * pivot_row[c];
            }
            other[col] = -coef / quotient;
        }
    }
    t.nonzero = nonzero;
}

/// Restores feasibility, then optimizes; false when infeasible, unbounded or out of pivots.
fn phase1(t: &mut Tableau, precision: f64) -> bool {
    for _ in 0..MAX_PIVOTS {
        let mut row = 0;
        let mut rhs = -precision;
        for r in 1..t.height {
            let v = t.at(r, 0);
            if v < rhs {
                rhs = v;
                row = r;
            }
        }
        if row == 0 {
            return phase2(t, precision);
        }
        let mut col = 0;
        let mut max_ratio = f64::NEG_INFINITY;
        for c in 1..t.width {
            let coefficient = t.at(row, c);
            if coefficient < -precision {
                let ratio = -t.at(0, c) / coefficient;
                if ratio > max_ratio {
                    max_ratio = ratio;
                    col = c;
                }
            }
        }
        if col == 0 {
            return false;
        }
        pivot(t, row, col);
    }
    false
}

fn phase2(t: &mut Tableau, precision: f64) -> bool {
    for _ in 0..MAX_PIVOTS {
        let mut col = 0;
        let mut value = precision;
        for c in 1..t.width {
            let reduced = t.at(0, c);
            if reduced > value {
                value = reduced;
                col = c;
            }
        }
        if col == 0 {
            return true;
        }
        let mut row = 0;
        let mut min_ratio = f64::INFINITY;
        for r in 1..t.height {
            let v = t.at(r, col);
            if v <= precision {
                continue;
            }
            let ratio = t.at(r, 0) / v;
            if ratio < min_ratio {
                row = r;
                min_ratio = ratio;
                if ratio <= precision {
                    break;
                }
            }
        }
        if row == 0 {
            return false;
        }
        pivot(t, row, col);
    }
    false
}
