//! Turns a cluster layout into exact compressed coordinates. The layout is approximate, so each
//! linked cluster is placed exactly on its parent's line along a spanning tree, every translation
//! kept a multiple of the residue modulus. A link closing a cycle is then repaired by sliding whole
//! subtrees along their tree links, solved as a small integer system.

use super::Square;
use super::bigint::Big;
use super::clustering::{Clustering, CompressionPoint, Link};
use super::lineforms::LineForm;
use super::simplex::js_round;
use super::skeleton::Layout;
use std::collections::VecDeque;

/// A tree link, from the cluster placed first to the one placed along it.
struct TreeEdge {
    form: usize,
    parent: usize,
    child: usize,
}

/// Every point's compressed coordinates, or `None` if a link cycle can't be closed exactly.
/// The home cluster keeps its own coordinates.
pub fn place(
    points: &[CompressionPoint],
    forms: &[LineForm],
    c: &Clustering,
    layout: &Layout,
    home: usize,
    unit: i64,
    modulus: i64,
) -> Option<Vec<Square>> {
    let anchors: Vec<&Square> = c
        .clusters
        .iter()
        .map(|members| &points[members[0]].coords)
        .collect();
    let home_at = anchors[home];
    let offset = |units: f64| Big::from_i128(js_round(units * unit as f64) as i128);
    let targets: Vec<Square> = (0..anchors.len())
        .map(|k| Square {
            x: &home_at.x + &offset(layout.x[k]),
            y: &home_at.y + &offset(layout.y[k]),
        })
        .collect();

    let (mut placed, tree, closures) =
        place_along_tree(forms, c, &anchors, &targets, home, modulus);
    if !close_cycles(forms, &anchors, &mut placed, &tree, &closures, modulus) {
        return None;
    }
    Some(
        points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let k = c.cluster_of[i];
                let (anchor, at) = (anchors[k], &placed[k]);
                Square {
                    x: &at.x + &(&point.coords.x - &anchor.x),
                    y: &at.y + &(&point.coords.y - &anchor.y),
                }
            })
            .collect(),
    )
}

/// Places home, then every cluster reachable along links breadth-first, each exactly on its
/// parent's line at the spot nearest its target. A cluster no link reaches roots a tree of its own.
fn place_along_tree(
    forms: &[LineForm],
    c: &Clustering,
    anchors: &[&Square],
    targets: &[Square],
    home: usize,
    modulus: i64,
) -> (Vec<Square>, Vec<TreeEdge>, Vec<Link>) {
    let n = anchors.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, link) in c.links.iter().enumerate() {
        adjacency[link.a].push(i);
        adjacency[link.b].push(i);
    }
    let mut placed: Vec<Option<Square>> = vec![None; n];
    let mut tree = Vec::new();
    let mut used = vec![false; c.links.len()];
    for root in std::iter::once(home).chain(0..n) {
        if placed[root].is_some() {
            continue;
        }
        placed[root] = Some(if root == home {
            anchors[root].clone()
        } else {
            Square {
                x: nearest_congruent(&targets[root].x, &anchors[root].x, modulus),
                y: nearest_congruent(&targets[root].y, &anchors[root].y, modulus),
            }
        });
        let mut queue = VecDeque::from([root]);
        while let Some(parent) = queue.pop_front() {
            for &i in &adjacency[parent] {
                let link = c.links[i];
                let child = if link.a == parent { link.b } else { link.a };
                if placed[child].is_some() {
                    continue;
                }
                used[i] = true;
                tree.push(TreeEdge {
                    form: link.form,
                    parent,
                    child,
                });
                let form = forms[link.form];
                let parent_at = placed[parent].as_ref().unwrap();
                let line = &(&form.value(parent_at) + &form.value(anchors[child]))
                    - &form.value(anchors[parent]);
                placed[child] = Some(nearest_on_line(
                    form,
                    &line,
                    &targets[child],
                    anchors[child],
                    modulus,
                ));
                queue.push_back(child);
            }
        }
    }
    let closures = c
        .links
        .iter()
        .zip(&used)
        .filter(|(_, u)| !**u)
        .map(|(l, _)| *l)
        .collect();
    (
        placed.into_iter().map(Option::unwrap).collect(),
        tree,
        closures,
    )
}

/// Makes every link outside the tree hold exactly, by sliding subtrees along their tree links in
/// multiples of the modulus. False when the integer system has no solution.
fn close_cycles(
    forms: &[LineForm],
    anchors: &[&Square],
    placed: &mut [Square],
    tree: &[TreeEdge],
    closures: &[Link],
    modulus: i64,
) -> bool {
    if closures.is_empty() {
        return true;
    }
    let residuals: Vec<Big> = closures
        .iter()
        .map(|l| link_residual(forms[l.form], anchors, placed, l))
        .collect();
    if residuals.iter().all(Big::is_zero) {
        return true;
    }
    if residuals.iter().any(|r| r.div_rem(modulus).1 != 0) {
        return false;
    }
    let Some(rhs) = residuals
        .iter()
        .map(|r| r.div_rem(modulus).0.to_i128().map(|v| -v))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let paths = tree_paths(anchors.len(), tree);
    let matrix: Vec<Vec<i128>> = closures
        .iter()
        .map(|l| {
            tree.iter()
                .enumerate()
                .map(|(e, edge)| {
                    let (dx, dy) = forms[edge.form].direction();
                    let along = forms[l.form].value_i(dx, dy) as i128;
                    along * (paths[l.a].contains(&e) as i128 - paths[l.b].contains(&e) as i128)
                })
                .collect()
        })
        .collect();
    let Some(slides) = solve_integer_system(matrix, &rhs) else {
        return false;
    };
    for (e, &slide) in slides.iter().enumerate() {
        if slide == 0 {
            continue;
        }
        let (dx, dy) = forms[tree[e].form].direction();
        let Some(step) = slide.checked_mul(modulus as i128) else {
            return false;
        };
        let (Some(sx), Some(sy)) = (step.checked_mul(dx as i128), step.checked_mul(dy as i128))
        else {
            return false;
        };
        let (sx, sy) = (Big::from_i128(sx), Big::from_i128(sy));
        for (k, path) in paths.iter().enumerate() {
            if path.contains(&e) {
                placed[k] = Square {
                    x: &placed[k].x + &sx,
                    y: &placed[k].y + &sy,
                };
            }
        }
    }
    closures
        .iter()
        .all(|l| link_residual(forms[l.form], anchors, placed, l).is_zero())
}

/// Each cluster's tree edges on its way up to its root.
fn tree_paths(n: usize, tree: &[TreeEdge]) -> Vec<Vec<usize>> {
    let mut edge_into = vec![None; n];
    for (e, edge) in tree.iter().enumerate() {
        edge_into[edge.child] = Some(e);
    }
    (0..n)
        .map(|k| {
            let mut path = Vec::new();
            let mut e = edge_into[k];
            while let Some(i) = e {
                path.push(i);
                e = edge_into[tree[i].parent];
            }
            path
        })
        .collect()
}

/// How far a link's placed clusters are from holding their original difference under its form.
fn link_residual(form: LineForm, anchors: &[&Square], placed: &[Square], l: &Link) -> Big {
    let placed_difference = &form.value(&placed[l.a]) - &form.value(&placed[l.b]);
    &placed_difference - &(&form.value(anchors[l.a]) - &form.value(anchors[l.b]))
}

/// Solves `matrix · x = rhs` over the integers, or `None` when no integer solution exists (or one
/// outgrows i128). The matrix is brought to column echelon form `matrix · U`, with `U` unimodular;
/// the triangular system then solves by forward substitution, and `x = U · y`.
fn solve_integer_system(matrix: Vec<Vec<i128>>, rhs: &[i128]) -> Option<Vec<i128>> {
    let (echelon, unimodular, pivots, rank) = column_echelon(matrix)?;
    let y = forward_substitute(&echelon, &pivots, rank, rhs)?;
    // The unimodular columns past the pivots span every solution's freedom.
    let freedoms: Vec<Vec<i128>> = (rank..unimodular.len())
        .map(|k| unimodular.iter().map(|row| row[k]).collect())
        .collect();
    let solution = unimodular
        .iter()
        .map(|row| dot(row, &y))
        .collect::<Option<Vec<_>>>()?;
    shrink(solution, &freedoms)
}

type Echelon = (Vec<Vec<i128>>, Vec<Vec<i128>>, Vec<Option<usize>>, usize);

/// `matrix · U` in column echelon form by Euclid's algorithm on columns, and each row's pivot column.
fn column_echelon(mut echelon: Vec<Vec<i128>>) -> Option<Echelon> {
    let cols = echelon[0].len();
    let mut unimodular: Vec<Vec<i128>> = (0..cols)
        .map(|i| (0..cols).map(|j| (i == j) as i128).collect())
        .collect();
    let mut pivots = Vec::with_capacity(echelon.len());
    let mut rank = 0;
    for r in 0..echelon.len() {
        loop {
            let row = &echelon[r];
            let smallest = (rank..cols)
                .filter(|&c| row[c] != 0)
                .min_by_key(|&c| row[c].unsigned_abs());
            let Some(smallest) = smallest else { break };
            for m in echelon.iter_mut().chain(unimodular.iter_mut()) {
                m.swap(rank, smallest);
            }
            let mut cleared = true;
            for c in rank + 1..cols {
                if echelon[r][c] == 0 {
                    continue;
                }
                let factor = echelon[r][c] / echelon[r][rank];
                for m in echelon.iter_mut().chain(unimodular.iter_mut()) {
                    m[c] = m[c].checked_sub(factor.checked_mul(m[rank])?)?;
                }
                if echelon[r][c] != 0 {
                    cleared = false;
                }
            }
            if cleared {
                break;
            }
        }
        let has_pivot = rank < cols && echelon[r][rank] != 0;
        pivots.push(has_pivot.then_some(rank));
        rank += has_pivot as usize;
    }
    Some((echelon, unimodular, pivots, rank))
}

/// The `y` solving the echelon system, or `None` when some row can't hold over the integers.
fn forward_substitute(
    echelon: &[Vec<i128>],
    pivots: &[Option<usize>],
    rank: usize,
    rhs: &[i128],
) -> Option<Vec<i128>> {
    let mut y = vec![0i128; echelon[0].len()];
    for (r, row) in echelon.iter().enumerate() {
        let mut remainder = rhs[r];
        for c in 0..pivots[r].unwrap_or(rank) {
            remainder = remainder.checked_sub(row[c].checked_mul(y[c])?)?;
        }
        match pivots[r] {
            None if remainder != 0 => return None,
            None => {}
            Some(p) => {
                if remainder % row[p] != 0 {
                    return None;
                }
                y[p] = remainder / row[p];
            }
        }
    }
    Some(y)
}

/// Moves `solution` along `freedoms` until no step shrinks it, as Euclid's steps can leave it huge.
fn shrink(mut solution: Vec<i128>, freedoms: &[Vec<i128>]) -> Option<Vec<i128>> {
    loop {
        let mut shrunk = false;
        for freedom in freedoms {
            let steps = nearest_quotient(dot(&solution, freedom)?, dot(freedom, freedom)?)?;
            if steps == 0 {
                continue;
            }
            for (v, f) in solution.iter_mut().zip(freedom) {
                *v = v.checked_sub(steps.checked_mul(*f)?)?;
            }
            shrunk = true;
        }
        if !shrunk {
            return Some(solution);
        }
    }
}

fn dot(a: &[i128], b: &[i128]) -> Option<i128> {
    a.iter()
        .zip(b)
        .try_fold(0i128, |sum, (x, y)| sum.checked_add(x.checked_mul(*y)?))
}

/// The integer nearest `numerator / denominator`, for a positive denominator.
fn nearest_quotient(numerator: i128, denominator: i128) -> Option<i128> {
    let doubled = numerator.checked_mul(2)?.checked_add(denominator)?;
    let twice = denominator.checked_mul(2)?;
    let quotient = doubled / twice;
    Some(if doubled < 0 && quotient * twice != doubled {
        quotient - 1
    } else {
        quotient
    })
}

/// The value congruent to `like` modulo `modulus` nearest `target`.
fn nearest_congruent(target: &Big, like: &Big, modulus: i64) -> Big {
    let below = target - &Big::from((target - like).rem_euclid(modulus));
    let above = &below + &Big::from(modulus);
    if target - &below <= &above - target {
        below
    } else {
        above
    }
}

/// The square on line `line` of `form`, congruent to `like`, nearest `target` along the line.
/// Every form has a unit coefficient, so its other coordinate is free and the division exact.
fn nearest_on_line(
    form: LineForm,
    line: &Big,
    target: &Square,
    like: &Square,
    modulus: i64,
) -> Square {
    if form.b.abs() == 1 {
        let x = nearest_congruent(&target.x, &like.x, modulus);
        let y = &(line - &(&x * form.a)) * form.b;
        Square { x, y }
    } else {
        let y = nearest_congruent(&target.y, &like.y, modulus);
        let x = &(line - &(&y * form.b)) * form.a;
        Square { x, y }
    }
}
