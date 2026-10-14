//! Lays out a position's rigid clusters as small as they go, measured in units of `unit` squares.
//! A linear program: links hold their clusters' shared values fixed, and each far gap shrinks to
//! about one unit, jittered so the compressed gaps relate to each other as generically as the
//! original astronomical ones did.

use super::bigint::Big;
use super::clustering::{Clustering, CompressionPoint, Link};
use super::lineforms::LineForm;
use super::simplex::{self, Model};
use super::{EXACT_SPAN, OrderedMap};

/// Each cluster's position in units, relative to the pinned home cluster.
pub struct Layout {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// A far gap the layout can't grow keeps this fraction of its original size as slack,
/// so a cluster wedged between another cluster's points still has room to sit.
const MODERATE_GAP_SLACK: u32 = 20;
/// The most units a layout may reach below home; more outgrows the solver's precision.
const MAX_REACH_UNITS: i64 = 1 << 22;
/// A prime below 2^26, so a product of two residues stays exact.
const RANK_PRIME: i64 = 67108859;
/// The solver's zero tolerance, in units. Placement makes the layout exact, so it need only be
/// close; a far tighter one reports wide layouts infeasible on rounding alone.
const SOLVER_PRECISION: f64 = 1e-6;

/// Solves the layout, or `None` when the solver finds none. `reach` is how far below home, in
/// squares, a cluster may be laid; `random` draws each far gap's jitter in [0, 1).
pub fn lay(
    points: &[CompressionPoint],
    forms: &[LineForm],
    clustering: &Clustering,
    home: usize,
    unit: i64,
    reach: i64,
    random: &mut impl FnMut() -> f64,
) -> Option<Layout> {
    let n = clustering.clusters.len();
    let mut model = ModelBuilder::new(n, (reach / unit).min(MAX_REACH_UNITS) as f64);
    model.fix(home);
    let anchor = |c: usize| &points[clustering.clusters[c][0]].coords;
    for link in independent_links(forms, &clustering.links, home, n) {
        let form = forms[link.form];
        let shared = &form.value(anchor(link.a)) - &form.value(anchor(link.b));
        model.add(
            form,
            link.a,
            link.b,
            Bound::Equal(shared.to_f64() / unit as f64),
        );
    }
    add_far_gaps(points, forms, clustering, unit, random, &mut model);
    model.minimize_span(forms, clustering);
    let values = simplex::solve(&model.build(), SOLVER_PRECISION)?;
    Some(model.read(&values))
}

/// The links whose equations don't follow from home's pin and the links kept before them. A cycle
/// of links makes its last one redundant, and the solver's elimination can trip over redundant
/// equalities; placement still holds every link exactly. Rank is found modulo [`RANK_PRIME`].
fn independent_links(forms: &[LineForm], links: &[Link], home: usize, n: usize) -> Vec<Link> {
    let mut basis: Vec<(usize, Vec<i64>)> = Vec::new();
    let mut keep_if_independent = |mut row: Vec<i64>| -> bool {
        for (pivot, basis_row) in &basis {
            let factor = row[*pivot];
            subtract_multiple(&mut row, basis_row, factor);
        }
        let Some(pivot) = row.iter().position(|&v| v != 0) else {
            return false;
        };
        let inverse = modular_inverse(row[pivot]);
        row.iter_mut().for_each(|v| *v = *v * inverse % RANK_PRIME);
        for (_, basis_row) in &mut basis {
            let factor = basis_row[pivot];
            subtract_multiple(basis_row, &row, factor);
        }
        basis.push((pivot, row));
        true
    };
    let row = |terms: &[(usize, i64)]| -> Vec<i64> {
        let mut values = vec![0; 2 * n];
        for &(variable, coefficient) in terms {
            values[variable] = (coefficient + RANK_PRIME) % RANK_PRIME;
        }
        values
    };
    keep_if_independent(row(&[(2 * home, 1)]));
    keep_if_independent(row(&[(2 * home + 1, 1)]));
    links
        .iter()
        .copied()
        .filter(|l| {
            let LineForm { a, b } = forms[l.form];
            keep_if_independent(row(&[
                (2 * l.a, a),
                (2 * l.b, -a),
                (2 * l.a + 1, b),
                (2 * l.b + 1, -b),
            ]))
        })
        .collect()
}

/// `row -= factor · other`, modulo [`RANK_PRIME`].
fn subtract_multiple(row: &mut [i64], other: &[i64], factor: i64) {
    if factor == 0 {
        return;
    }
    for (v, o) in row.iter_mut().zip(other) {
        *v = (*v - factor * o % RANK_PRIME + RANK_PRIME) % RANK_PRIME;
    }
}

/// The inverse of `value` modulo [`RANK_PRIME`], by the extended Euclidean algorithm.
fn modular_inverse(value: i64) -> i64 {
    let (mut r0, mut r1, mut s0, mut s1) = (RANK_PRIME, value, 0i64, 1i64);
    while r1 != 0 {
        let q = r0.div_euclid(r1);
        (r0, r1, s0, s1) = (r1, r0 - q * r1, s1, s0 - q * s1);
    }
    (s0 + RANK_PRIME) % RANK_PRIME
}

/// Requires every far gap between two clusters to stay far: at least one unit plus jitter, or its
/// original size when that's smaller, less [`MODERATE_GAP_SLACK`], but always beyond the exact span.
fn add_far_gaps(
    points: &[CompressionPoint],
    forms: &[LineForm],
    c: &Clustering,
    unit: i64,
    random: &mut impl FnMut() -> f64,
    model: &mut ModelBuilder,
) {
    let floor = Big::from(EXACT_SPAN + 1);
    for (f, order) in c.orders.iter().enumerate() {
        let form = forms[f];
        let anchor_value = |k: usize| form.value(&points[c.clusters[k][0]].coords);
        // The tightest bound between each pair of clusters, as several of their points may neighbour.
        let mut bounds: OrderedMap<(usize, usize), Big> = OrderedMap::default();
        for w in order.windows(2) {
            let (p, q) = (w[0], w[1]);
            let (cp, cq) = (c.cluster_of[p], c.cluster_of[q]);
            if c.classes[f][p] == c.classes[f][q] || cp == cq {
                continue;
            }
            let (vp, vq) = (
                c.values[f][p].as_ref().unwrap(),
                c.values[f][q].as_ref().unwrap(),
            );
            let gap = vq - vp;
            let jittered = Big::from(unit + (random() * unit as f64).floor() as i64);
            let moderate = (&gap - &gap.shr(MODERATE_GAP_SLACK)).max(floor.clone());
            // Points sit at their cluster's anchor plus a fixed offset, which the bound absorbs.
            let offsets = &(vq - &anchor_value(cq)) - &(vp - &anchor_value(cp));
            let minimum = &jittered.min(moderate) - &offsets;
            if bounds
                .get(&(cq, cp))
                .is_none_or(|existing| &minimum > existing)
            {
                bounds.insert((cq, cp), minimum);
            }
        }
        for ((above, below), minimum) in bounds.iter() {
            model.add(
                form,
                *above,
                *below,
                Bound::Min(minimum.to_f64() / unit as f64),
            );
        }
    }
}

enum Bound {
    Equal(f64),
    Min(f64),
}

/// Accumulates the linear program over each cluster's (x, y): variable `2c` is cluster c's x and
/// `2c + 1` its y. Constraint keys count up from 1; key 0 is the objective. The solver only takes
/// nonnegative variables, so each is offset, which bounds how far below home a cluster goes.
struct ModelBuilder {
    cluster_count: usize,
    offset: f64,
    /// Variables in first-use order (the solver's column order), each with its terms by key.
    variables: OrderedMap<usize, OrderedMap<u32, f64>>,
    constraints: Vec<(u32, f64, f64)>,
}

impl ModelBuilder {
    fn new(cluster_count: usize, offset: f64) -> ModelBuilder {
        ModelBuilder {
            cluster_count,
            offset,
            variables: OrderedMap::default(),
            constraints: Vec::new(),
        }
    }

    /// Pins cluster `c` at the origin.
    fn fix(&mut self, c: usize) {
        self.constrain(&[(2 * c, 1.0)], Bound::Equal(self.offset));
        self.constrain(&[(2 * c + 1, 1.0)], Bound::Equal(self.offset));
    }

    /// Bounds `form`'s value of cluster `a` minus cluster `b`.
    fn add(&mut self, form: LineForm, a: usize, b: usize, bound: Bound) {
        let mut terms = Vec::with_capacity(4);
        if form.a != 0 {
            terms.extend([(2 * a, form.a as f64), (2 * b, -form.a as f64)]);
        }
        if form.b != 0 {
            terms.extend([(2 * a + 1, form.b as f64), (2 * b + 1, -form.b as f64)]);
        }
        self.constrain(&terms, bound);
    }

    /// Minimizes the position's width plus height. Compression keeps every order, so the clusters
    /// at either end of each axis are known ahead, and the span is linear.
    fn minimize_span(&mut self, forms: &[LineForm], c: &Clustering) {
        for axis in 0..2 {
            let form = forms
                .iter()
                .position(|f| if axis == 0 { f.is_file() } else { f.is_rank() })
                .unwrap();
            let order = &c.orders[form];
            self.term(2 * c.cluster_of[*order.last().unwrap()] + axis, 0, 1.0);
            self.term(2 * c.cluster_of[order[0]] + axis, 0, -1.0);
        }
    }

    fn build(&self) -> Model {
        Model {
            constraints: self.constraints.clone(),
            variables: self
                .variables
                .iter()
                .map(|(_, terms)| terms.iter().map(|(&k, &v)| (k, v)).collect())
                .collect(),
        }
    }

    /// Each cluster's solved position, relative to the pinned home.
    fn read(&self, values: &[f64]) -> Layout {
        let mut layout = Layout {
            x: vec![0.0; self.cluster_count],
            y: vec![0.0; self.cluster_count],
        };
        for ((&variable, _), &value) in self.variables.iter().zip(values) {
            let axis = if variable % 2 == 0 {
                &mut layout.x
            } else {
                &mut layout.y
            };
            axis[variable >> 1] = value - self.offset;
        }
        layout
    }

    fn constrain(&mut self, terms: &[(usize, f64)], bound: Bound) {
        let key = self.constraints.len() as u32 + 1;
        let (lower, upper) = match bound {
            Bound::Equal(v) => (v, v),
            Bound::Min(v) => (v, f64::INFINITY),
        };
        self.constraints.push((key, lower, upper));
        for &(variable, coefficient) in terms {
            self.term(variable, key, coefficient);
        }
    }

    fn term(&mut self, variable: usize, key: u32, coefficient: f64) {
        if self.variables.get(&variable).is_none() {
            self.variables.insert(variable, OrderedMap::default());
        }
        let row = self.variables.get_mut(&variable).unwrap();
        let sum = row.get(&key).copied().unwrap_or(0.0) + coefficient;
        row.insert(key, sum);
    }
}
