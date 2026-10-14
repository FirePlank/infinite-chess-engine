//! Groups a position's points by what compression must keep exact. Under each line form, points
//! whose values chain together within [`EXACT_SPAN`] form a class, which only moves as one.
//! Points sharing classes in two forms can't move apart at all, so they weld into rigid clusters;
//! clusters sharing a class in one form are linked, free only to slide along it.

use super::bigint::Big;
use super::lineforms::LineForm;
use super::{EXACT_SPAN, Square};
use rustc_hash::FxHashMap;

/// A square or lone line whose relations compression keeps.
#[derive(Clone, Debug)]
pub struct CompressionPoint {
    pub coords: Square,
    /// Set for a lone line (a promotion rank, border edge or far-escape shell line) rather than a
    /// square: it takes part only in its own form, and its other coordinate means nothing.
    pub line: Option<LoneLine>,
}

#[derive(Clone, Copy, Debug)]
pub struct LoneLine {
    pub form: usize,
    /// A border edge, which also keeps which side of it every crossing lands on.
    pub boundary: bool,
}

/// Two clusters held to one class of a form: their translations agree under it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Link {
    pub form: usize,
    pub a: usize,
    pub b: usize,
}

/// How a position's points relate under each form.
pub struct Clustering {
    /// Per form, each point's value, or `None` where the point takes no part.
    pub values: Vec<Vec<Option<Big>>>,
    /// Per form, the taking-part points in ascending value.
    pub orders: Vec<Vec<usize>>,
    /// Per form, each taking-part point's class (-1 where it takes no part).
    pub classes: Vec<Vec<i32>>,
    pub cluster_of: Vec<usize>,
    /// Each cluster's points, its anchor first.
    pub clusters: Vec<Vec<usize>>,
    pub links: Vec<Link>,
}

/// Clusters `points` under `forms`.
pub fn build(points: &[CompressionPoint], forms: &[LineForm]) -> Clustering {
    let values: Vec<Vec<Option<Big>>> = forms
        .iter()
        .enumerate()
        .map(|(f, form)| {
            points
                .iter()
                .map(|p| {
                    p.line
                        .is_none_or(|l| l.form == f)
                        .then(|| form.value(&p.coords))
                })
                .collect()
        })
        .collect();
    let orders: Vec<Vec<usize>> = values.iter().map(|v| sort_by_value(v)).collect();
    let classes: Vec<Vec<i32>> = orders
        .iter()
        .zip(&values)
        .map(|(o, v)| classify(o, v, points.len()))
        .collect();
    let mut cluster_of = weld_clusters(points.len(), &orders, &classes);
    let clusters = group_clusters(&mut cluster_of);
    let links = find_links(&orders, &classes, &cluster_of);
    Clustering {
        values,
        orders,
        classes,
        cluster_of,
        clusters,
        links,
    }
}

/// The indices of the defined values, in ascending value (stably, as the site sorts).
fn sort_by_value(values: &[Option<Big>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).filter(|&i| values[i].is_some()).collect();
    order.sort_by(|&i, &j| values[i].as_ref().unwrap().cmp(values[j].as_ref().unwrap()));
    order
}

/// Numbers the classes along one form's order: a gap beyond the exact span starts the next.
fn classify(order: &[usize], values: &[Option<Big>], n: usize) -> Vec<i32> {
    let span = Big::from(EXACT_SPAN);
    let mut class_of = vec![-1; n];
    let mut current = -1;
    let mut previous: Option<&Big> = None;
    for &i in order {
        let value = values[i].as_ref().unwrap();
        if previous.is_none_or(|p| value - p > span) {
            current += 1;
        }
        class_of[i] = current;
        previous = Some(value);
    }
    class_of
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Each point's rigid cluster root: the closure of merging any two clusters that share classes in
/// two forms, since their translations must then agree in both coordinates.
fn weld_clusters(n: usize, orders: &[Vec<usize>], classes: &[Vec<i32>]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    // Points sharing two classes with each other weld at once, leaving few roots per class below.
    for (f, class_f) in classes.iter().enumerate() {
        for class_g in &classes[f + 1..] {
            let mut first: FxHashMap<(i32, i32), usize> = FxHashMap::default();
            for (i, (&cf, &cg)) in class_f.iter().zip(class_g).enumerate() {
                if cf == -1 || cg == -1 {
                    continue;
                }
                match first.get(&(cf, cg)) {
                    None => {
                        first.insert((cf, cg), i);
                    }
                    Some(&other) => {
                        let (ri, ro) = (find(&mut parent, i), find(&mut parent, other));
                        parent[ri] = ro;
                    }
                }
            }
        }
    }
    loop {
        // For each pair of cluster roots sharing a class, a bit per form they share one in.
        let mut shared: FxHashMap<(usize, usize), u32> = FxHashMap::default();
        for (f, order) in orders.iter().enumerate() {
            for roots in group_roots_by_class(order, &classes[f], |i| find(&mut parent, i)) {
                for i in 0..roots.len() {
                    for j in i + 1..roots.len() {
                        let pair = (roots[i].min(roots[j]), roots[i].max(roots[j]));
                        *shared.entry(pair).or_default() |= 1 << f;
                    }
                }
            }
        }
        let mut merged = false;
        for ((a, b), bits) in shared {
            if bits & (bits - 1) == 0 {
                continue; // Shares only one form: a link, not a weld.
            }
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
                merged = true;
            }
        }
        if !merged {
            return (0..n).map(|i| find(&mut parent, i)).collect();
        }
    }
}

/// The distinct cluster roots within each class of one form, in order.
fn group_roots_by_class(
    order: &[usize],
    class_of: &[i32],
    mut root: impl FnMut(usize) -> usize,
) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &i in order {
        let k = class_of[i] as usize;
        if groups.len() <= k {
            groups.push(Vec::new());
        }
        let r = root(i);
        if !groups[k].contains(&r) {
            groups[k].push(r);
        }
    }
    groups
}

/// Renumbers the cluster roots 0..n by first point and lists each cluster's points.
fn group_clusters(cluster_of: &mut [usize]) -> Vec<Vec<usize>> {
    let mut id_of_root: FxHashMap<usize, usize> = FxHashMap::default();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for (i, root) in cluster_of.iter_mut().enumerate() {
        let id = *id_of_root.entry(*root).or_insert_with(|| {
            clusters.push(Vec::new());
            clusters.len() - 1
        });
        clusters[id].push(i);
        *root = id;
    }
    clusters
}

/// Chains together the distinct clusters within each class of each form.
fn find_links(orders: &[Vec<usize>], classes: &[Vec<i32>], cluster_of: &[usize]) -> Vec<Link> {
    let mut links = Vec::new();
    for (form, order) in orders.iter().enumerate() {
        for members in group_roots_by_class(order, &classes[form], |i| cluster_of[i]) {
            for w in members.windows(2) {
                links.push(Link {
                    form,
                    a: w[0],
                    b: w[1],
                });
            }
        }
    }
    links
}

/// Whether `point`'s coordinate on `axis` (0 = x) means anything: always for a square, only its
/// own for a lone line.
pub fn uses_axis(forms: &[LineForm], point: &CompressionPoint, axis: usize) -> bool {
    point.line.is_none_or(|l| {
        (if axis == 0 {
            forms[l.form].a
        } else {
            forms[l.form].b
        }) != 0
    })
}
