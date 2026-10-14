//! Compresses a point set so the engine perceives exactly what it would in the original. Under
//! every line form, two points within [`EXACT_SPAN`] keep their exact difference, and points
//! further apart keep their order and stay beyond it. Every coordinate keeps its residue, so lines
//! still cross on the same squares, and the few crossings where a slider lands near a third line
//! far from every piece are kept too. Far gaps shrink to large but representable, jittered sizes.
//! The origin, which the engine anchors its far escapes and TT moves to, stays put. Every result is
//! verified exactly.

use super::bigint::Big;
use super::clustering::{self, Clustering, CompressionPoint};
use super::crossings::{self, Meetings, Slider};
use super::lineforms::{self, LineForm};
use super::{EXACT_SPAN, OrderedMap, Square, placement, skeleton};
use rustc_hash::FxHashSet;

/// A point set to compress.
pub struct CompressionInput {
    pub points: Vec<CompressionPoint>,
    /// How many leading points the engine reads lines through; the rest it only replays history over.
    pub perceived: usize,
    pub forms: Vec<LineForm>,
    pub sliders: Vec<Slider>,
}

/// Compressed coordinates stay within this, so the engine's unchecked sums of coordinates
/// (diagonal keys, the kings' midpoint, a mean of many pieces) fit i64.
pub const RANGE: i64 = 1 << 54;

/// The unit far gaps shrink to, and the jitter seed, of each attempt in turn. Smaller units fit
/// more clusters in [`RANGE`]; fresh jitter breaks an accidental alignment the last one made.
const ATTEMPTS: [(i64, u32); 6] = [
    (1 << 40, 1),
    (1 << 40, 2),
    (1 << 34, 3),
    (1 << 28, 4),
    (1 << 22, 5),
    (1 << 16, 6),
];

/// Each point's compressed coordinates, or `None` when no faithful compression fits.
pub fn compress(input: &CompressionInput) -> Option<Vec<Square>> {
    let modulus = lineforms::residue_modulus(&input.forms);
    let origin = input.points.len();
    let mut base = input.points.clone();
    base.push(CompressionPoint {
        coords: Square {
            x: Big::zero(),
            y: Big::zero(),
        },
        line: None,
    });
    base.extend(boundary_crossings(input));
    let (points, structure, meetings) = with_crossing_points(input, base);
    let home = structure.cluster_of[origin];
    for (unit, seed) in ATTEMPTS {
        let mut random = seeded_random(seed);
        let Some(layout) = skeleton::lay(
            &points,
            &input.forms,
            &structure,
            home,
            unit,
            2 * RANGE,
            &mut random,
        ) else {
            continue;
        };
        if let Some(coords) = placement::place(
            &points,
            &input.forms,
            &structure,
            &layout,
            home,
            unit,
            modulus,
        ) && verify(input, &points, &structure, &meetings, &coords, modulus)
        {
            return Some(coords[..input.points.len()].to_vec());
        }
    }
    None
}

/// Where every square's lines cross every boundary (a border edge). A crossing of two lines lands
/// on a boundary's inner side exactly when those lines cross the boundary in a given order, so
/// keeping the order of these points along it keeps every crossing's side.
fn boundary_crossings(input: &CompressionInput) -> Vec<CompressionPoint> {
    let squares: Vec<&Square> = input.points[..input.perceived]
        .iter()
        .filter(|p| p.line.is_none())
        .map(|p| &p.coords)
        .collect();
    let mut seen: FxHashSet<Square> = squares.iter().map(|&s| s.clone()).collect();
    let mut added = Vec::new();
    for edge in &input.points {
        if !edge.line.is_some_and(|l| l.boundary) {
            continue;
        }
        let fixed_x = clustering::uses_axis(&input.forms, edge, 0);
        let fixed = if fixed_x {
            &edge.coords.x
        } else {
            &edge.coords.y
        };
        for square in &squares {
            for form in &input.forms {
                let (fixed_coefficient, free_coefficient) = if fixed_x {
                    (form.a, form.b)
                } else {
                    (form.b, form.a)
                };
                if free_coefficient == 0 {
                    continue; // Parallel to the edge.
                }
                let numerator = &form.value(square) - &(fixed * fixed_coefficient);
                for free in bracket(&numerator, free_coefficient) {
                    let coords = if fixed_x {
                        Square {
                            x: fixed.clone(),
                            y: free,
                        }
                    } else {
                        Square {
                            x: free,
                            y: fixed.clone(),
                        }
                    };
                    if seen.insert(coords.clone()) {
                        added.push(CompressionPoint { coords, line: None });
                    }
                }
            }
        }
    }
    added
}

/// The integers at and around `numerator / denominator`: just it when exact, else the two either side.
fn bracket(numerator: &Big, denominator: i64) -> Vec<Big> {
    let (quotient, remainder) = numerator.div_rem(denominator);
    if remainder == 0 {
        return vec![quotient];
    }
    let below = if numerator.is_neg() != (denominator < 0) {
        &quotient - &Big::from(1)
    } else {
        quotient
    };
    let above = &below + &Big::from(1);
    vec![below, above]
}

/// Adds, as a point of its own, every square where a slider meets two other lines far from every
/// existing point, so compression keeps it. One within a quarter of the exact span of a point
/// already shares that point's classes in every form, which keeps it without help.
fn with_crossing_points(
    input: &CompressionInput,
    base_points: Vec<CompressionPoint>,
) -> (Vec<CompressionPoint>, Clustering, Meetings) {
    let base = clustering::build(&base_points, &input.forms);
    let meetings = crossings::find(
        &input.forms,
        &input.sliders,
        &base,
        &base.values,
        input.perceived,
    );
    let mut added: OrderedMap<Square, ()> = OrderedMap::default();
    for square in crossings::squares(&input.forms, &base.values, &meetings) {
        if !is_near_any_point(&input.forms, &base, &square) {
            added.insert(square, ());
        }
    }
    if added.is_empty() {
        return (base_points, base, meetings);
    }
    let mut points = base_points;
    points.extend(added.keys().map(|s| CompressionPoint {
        coords: s.clone(),
        line: None,
    }));
    let structure = clustering::build(&points, &input.forms);
    let meetings = crossings::find(
        &input.forms,
        &input.sliders,
        &structure,
        &structure.values,
        input.perceived,
    );
    (points, structure, meetings)
}

/// Whether a square of `structure` lies within a quarter of the exact span of `square`, on both axes.
fn is_near_any_point(forms: &[LineForm], structure: &Clustering, square: &Square) -> bool {
    let reach = Big::from(EXACT_SPAN / 4);
    let x_form = forms.iter().position(LineForm::is_file).unwrap();
    let y_form = forms.iter().position(LineForm::is_rank).unwrap();
    let xs = &structure.orders[x_form];
    let values = &structure.values[x_form];
    let (low_x, high_x) = (&square.x - &reach, &square.x + &reach);
    let low = xs.partition_point(|&i| values[i].as_ref().unwrap() < &low_x);
    xs[low..]
        .iter()
        .take_while(|&&i| values[i].as_ref().unwrap() <= &high_x)
        .any(|&i| {
            structure.values[y_form][i]
                .as_ref()
                .is_some_and(|y| (y - &square.y).abs() <= reach)
        })
}

/// A deterministic [0, 1) generator, so the same position always compresses the same way.
fn seeded_random(seed: u32) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x6d2b79f5);
        let mut t = (state ^ (state >> 15)).wrapping_mul(1 | state);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}

/// Whether `coords` keeps everything compression promises: within [`RANGE`], every residue, every
/// exact difference and far order under every form, and every slider's meetings.
fn verify(
    input: &CompressionInput,
    points: &[CompressionPoint],
    structure: &Clustering,
    meetings: &Meetings,
    coords: &[Square],
    modulus: i64,
) -> bool {
    let range = Big::from(RANGE);
    for (i, point) in points.iter().enumerate() {
        for axis in 0..2 {
            if !clustering::uses_axis(&input.forms, point, axis) {
                continue;
            }
            let (c, o) = if axis == 0 {
                (&coords[i].x, &point.coords.x)
            } else {
                (&coords[i].y, &point.coords.y)
            };
            if c.abs() > range || (c - o).div_rem(modulus).1 != 0 {
                return false;
            }
        }
    }
    let values: Vec<Vec<Option<Big>>> = structure
        .values
        .iter()
        .zip(&input.forms)
        .map(|(form_values, form)| {
            form_values
                .iter()
                .zip(coords)
                .map(|(v, c)| v.as_ref().map(|_| form.value(c)))
                .collect()
        })
        .collect();
    if !keeps_every_gap(structure, &values) {
        return false;
    }
    let compressed = crossings::find(
        &input.forms,
        &input.sliders,
        structure,
        &values,
        input.perceived,
    );
    compressed.len() == meetings.len() && meetings.iter().all(|(k, v)| compressed.get(k) == Some(v))
}

/// Whether every gap between neighbours under every form is kept: exactly within the exact span,
/// beyond it otherwise. Every pair's relation is a sum of such gaps, so this keeps them all.
fn keeps_every_gap(structure: &Clustering, compressed: &[Vec<Option<Big>>]) -> bool {
    let span = Big::from(EXACT_SPAN);
    structure.orders.iter().enumerate().all(|(f, order)| {
        let (original, values) = (&structure.values[f], &compressed[f]);
        order.windows(2).all(|w| {
            let gap = original[w[1]].as_ref().unwrap() - original[w[0]].as_ref().unwrap();
            let compressed_gap = values[w[1]].as_ref().unwrap() - values[w[0]].as_ref().unwrap();
            if gap <= span {
                compressed_gap == gap
            } else {
                compressed_gap > span
            }
        })
    })
}
