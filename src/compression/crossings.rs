//! Finds where a slider can land on two other pieces' lines at once. A slider stops where its line
//! crosses another piece's line, and a square there that also passes within [`NEAR_LINE_SPAN`] of a
//! third piece's line is a fork, a block, or a landing under attack. Per-form compression keeps
//! two-point relations exact, but such a meeting far from every piece depends on three far gaps at
//! once, so it is checked directly.

use super::bigint::Big;
use super::clustering::Clustering;
use super::lineforms::{self, LineForm};
use super::{EXACT_SPAN, NEAR_LINE_SPAN, OrderedMap, Square};
use rustc_hash::FxHashSet;

/// A point that slides, and the forms of the lines it slides along.
pub struct Slider {
    pub point: usize,
    pub forms: Vec<usize>,
}

/// `(slider, its form, second point, its form, third point, its form)`.
pub type MeetingKey = (usize, usize, usize, usize, usize, usize);

/// Every crossing within reach of a third line, valued by its offset from that line. A triple whose
/// three lines all share a class with one cluster is left out: the meeting moves with the cluster.
pub type Meetings = OrderedMap<MeetingKey, Big>;

/// Values are searched as doubles, then confirmed exactly. Only values past 2^APPROX_BITS are
/// scaled down, to stay within double range.
const APPROX_BITS: u32 = 1000;
/// Bounds a double's rounding error relative to the values it's computed from, with ample margin.
const RELATIVE_ERROR: f64 = 1.0 / (1u64 << 50) as f64;
/// Two lines through squares w apart cross within 6w of them, which moves any form's value at most
/// 18w, so in a cluster narrower than this every near third line shares one of its classes.
const COMPACT_WIDTH: i64 = (EXACT_SPAN - NEAR_LINE_SPAN) / 18;

/// Every crossing of a slider's line with a second point's line that passes within the near-line
/// span of a third point's line, under the given values (original or compressed) of each point.
/// Only points before `perceived` stand for lines the engine reads.
pub fn find(
    forms: &[LineForm],
    sliders: &[Slider],
    clustering: &Clustering,
    values: &[Vec<Option<Big>>],
    perceived: usize,
) -> Meetings {
    let (scaled, shift) = approximate(values);
    let compact = find_compact_clusters(forms, clustering, values);
    let class_clusters: Vec<Vec<FxHashSet<usize>>> = clustering
        .classes
        .iter()
        .map(|class_of| {
            let mut sets: Vec<FxHashSet<usize>> = Vec::new();
            for (i, &k) in class_of.iter().enumerate() {
                if k >= 0 {
                    if sets.len() <= k as usize {
                        sets.resize_with(k as usize + 1, Default::default);
                    }
                    sets[k as usize].insert(clustering.cluster_of[i]);
                }
            }
            sets
        })
        .collect();
    let context = Context {
        forms,
        clustering,
        values,
        scaled: &scaled,
        shift,
        compact: &compact,
        class_clusters: &class_clusters,
        perceived,
    }; // prettier-ignore
    let mut meetings = Meetings::default();
    for slider in sliders {
        for &fd in &slider.forms {
            for f2 in (0..forms.len()).filter(|&f| f != fd) {
                for f3 in (0..forms.len()).filter(|&f| f != fd && f != f2) {
                    find_for_families(&context, &mut meetings, slider.point, fd, f2, f3);
                }
            }
        }
    }
    meetings
}

struct Context<'a> {
    forms: &'a [LineForm],
    clustering: &'a Clustering,
    values: &'a [Vec<Option<Big>>],
    scaled: &'a [Vec<f64>],
    shift: u32,
    compact: &'a [bool],
    class_clusters: &'a [Vec<FxHashSet<usize>>],
    perceived: usize,
}

/// Which clusters are narrow enough that a crossing of two of their own lines can't meet a line
/// outside their classes.
fn find_compact_clusters(
    forms: &[LineForm],
    c: &Clustering,
    values: &[Vec<Option<Big>>],
) -> Vec<bool> {
    let n = c.clusters.len();
    let mut compact = vec![true; n];
    let width = Big::from(COMPACT_WIDTH);
    let axes = [
        forms.iter().position(|f| f.is_file()).unwrap(),
        forms.iter().position(|f| f.is_rank()).unwrap(),
    ];
    for axis in axes {
        let mut bounds: Vec<Option<(Big, Big)>> = vec![None; n];
        for (i, value) in values[axis].iter().enumerate() {
            let Some(value) = value else { continue };
            let slot = &mut bounds[c.cluster_of[i]];
            match slot {
                None => *slot = Some((value.clone(), value.clone())),
                Some((low, high)) => {
                    if value < low {
                        *low = value.clone();
                    }
                    if value > high {
                        *high = value.clone();
                    }
                }
            }
        }
        for (k, b) in bounds.iter().enumerate() {
            if let Some((low, high)) = b
                && high - low > width
            {
                compact[k] = false;
            }
        }
    }
    compact
}

/// Each value scaled down into double range, and the scale's exponent.
fn approximate(values: &[Vec<Option<Big>>]) -> (Vec<Vec<f64>>, u32) {
    let largest = values
        .iter()
        .flatten()
        .flatten()
        .map(Big::bit_length)
        .max()
        .unwrap_or(0);
    let shift = largest.saturating_sub(APPROX_BITS);
    let scaled = values
        .iter()
        .map(|v| {
            v.iter()
                .map(|x| x.as_ref().map_or(f64::NAN, |x| x.shr(shift).to_f64()))
                .collect()
        })
        .collect();
    (scaled, shift)
}

/// Records the meetings of slider `p1`'s `fd` line with every `f2` line, near every `f3` line. With
/// c1, c2, c3 the three lines' values, the crossing sits at `f3 = (A·c1 + B·c2) / det`, so for each
/// second line the third lines in reach are a binary search away.
fn find_for_families(
    cx: &Context,
    meetings: &mut Meetings,
    p1: usize,
    fd: usize,
    f2: usize,
    f3: usize,
) {
    let (form_d, form2, form3) = (cx.forms[fd], cx.forms[f2], cx.forms[f3]);
    let det = lineforms::determinant(form_d, form2);
    let a = lineforms::determinant(form3, form2);
    let b = lineforms::determinant(form_d, form3);
    let c = cx.clustering;
    let c1 = cx.values[fd][p1].as_ref().unwrap();
    let (scaled2, scaled3) = (&cx.scaled[f2], &cx.scaled[f3]);
    let order3 = &c.orders[f3];
    let (det_n, a_n, b_n, c1_n) = (det as f64, a as f64, b as f64, cx.scaled[fd][p1]);
    // The span, plus each of the three values' unit of scaling error carried through the formula.
    let fixed_window = Big::from(NEAR_LINE_SPAN).shr(cx.shift).to_f64()
        + 1.0
        + (a_n.abs() + b_n.abs()) / det_n.abs();
    let bound = Big::from(det.abs() * NEAR_LINE_SPAN);
    let clusters1 = &cx.class_clusters[fd][c.classes[fd][p1] as usize];
    let k1 = c.cluster_of[p1];

    for &p2 in &c.orders[f2] {
        if p2 == p1 || p2 >= cx.perceived || (c.cluster_of[p2] == k1 && cx.compact[k1]) {
            continue;
        }
        let (term1, term2) = (a_n * c1_n, b_n * scaled2[p2]);
        let target = (term1 + term2) / det_n;
        // Rounding grows with the terms the target is computed from, and with the third value's size.
        let window = fixed_window
            + RELATIVE_ERROR * ((term1.abs() + term2.abs()) / det_n.abs() + target.abs());
        let c2 = cx.values[f2][p2].as_ref().unwrap();
        let clusters2 = &cx.class_clusters[f2][c.classes[f2][p2] as usize];
        let start = order3.partition_point(|&q| scaled3[q] < target - window);
        for &q in &order3[start..] {
            if scaled3[q] > target + window {
                break;
            }
            if q >= cx.perceived {
                continue;
            }
            let clusters3 = &cx.class_clusters[f3][c.classes[f3][q] as usize];
            if clusters1
                .iter()
                .any(|k| clusters2.contains(k) && clusters3.contains(k))
            {
                continue;
            }
            let offset_times_det =
                &(&(c1 * a) + &(c2 * b)) - &(cx.values[f3][q].as_ref().unwrap() * det);
            if offset_times_det.abs() > bound
                || lineforms::crossing(form_d, c1, form2, c2).is_none()
            {
                continue;
            }
            meetings.insert((p1, fd, p2, f2, q, f3), offset_times_det.div_rem(det).0);
        }
    }
}

/// The distinct squares the meetings land on, in meeting order.
pub fn squares(
    forms: &[LineForm],
    values: &[Vec<Option<Big>>],
    meetings: &Meetings,
) -> Vec<Square> {
    let mut seen = FxHashSet::default();
    let mut landings = Vec::new();
    for &(p1, fd, p2, f2, _, _) in meetings.keys() {
        if seen.insert((p1, fd, p2, f2)) {
            let (c1, c2) = (
                values[fd][p1].as_ref().unwrap(),
                values[f2][p2].as_ref().unwrap(),
            );
            landings.push(lineforms::crossing(forms[fd], c1, forms[f2], c2).unwrap());
        }
    }
    landings
}
