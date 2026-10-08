//! Classifies every insufficient-material "core" (White's army against Black's
//! royals only, a set the mating-set table says can mate) as forced, helpmate-only
//! or unknown, judged from spread starts: pieces 3+ apart, no shared lines.
//!
//! Forced: proof-number search in a box around the defender proves mate from every
//! sampled start; a defender royal leaving the box counts as escaping, so a proof
//! holds on the open board. Helpmate-only, only ever proved: the walk-back on a
//! frame that only gives the attacker extra power finds no spread start lost.
//! Unknown is everything else; the engine treats it as forced, the safe side.
//!
//! Usage: cargo run --release --bin forced_gen -- <out dir> [threads] [budget] [filter]
//! Writes forced.tsv, escape.txt (the helpmate-only list for insuffmat_build) and
//! unknown.tsv; progress.tsv lets a stopped run resume. FORCED_VERIFY=<list> instead
//! runs the forced prover on a helpmate-only list, which must find no mate.

use apeiron::evaluation::mating_sets::{self, KIND_CODES, KINDS};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// Geometry ---------------------------------------------------------------------

const K: u8 = 0;
const RC: u8 = 1;
const RQ: u8 = 2;
const P: u8 = 8;
const GU: u8 = 13;
const HU: u8 = 19;
const RO: u8 = 20;

/// Half-width of the proof box: the defender escapes by leaving it.
const BOX: i32 = 10;
/// Spread starts place attackers within this radius, at least `SPREAD` apart.
const START_RADIUS: i32 = 6;
const SPREAD: i32 = 3;
const STARTS: usize = 4;

const KING: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAG: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn leapers(a: i32, b: i32) -> Vec<(i32, i32)> {
    let mut v = Vec::new();
    for (x, y) in [(a, b), (b, a)] {
        for sx in [1, -1] {
            for sy in [1, -1] {
                if !v.contains(&(x * sx, y * sy)) {
                    v.push((x * sx, y * sy));
                }
            }
        }
    }
    v
}

type Offsets = Vec<(i32, i32)>;
/// A defender move target, with the square a two-step slide crosses.
type Target = ((i32, i32), Option<(i32, i32)>);

struct Def {
    leaps: Vec<(i32, i32)>,
    slides: Vec<(i32, i32)>,
    royal: bool,
    /// Moves at most one square per coordinate: what the escape lemma needs.
    slow: bool,
}

fn defs() -> Vec<Def> {
    let knight = leapers(1, 2);
    let mut v = Vec::new();
    for kind in 0..KINDS {
        let (leaps, slides): (Offsets, Offsets) = match kind {
            K | GU => (KING.to_vec(), vec![]),
            RC | 17 => ([&KING[..], &knight[..]].concat(), vec![]),
            RQ | 3 => (vec![], [&ORTHO[..], &DIAG[..]].concat()),
            4 => (vec![], ORTHO.to_vec()),
            5 | 6 => (vec![], DIAG.to_vec()),
            7 => (knight.clone(), vec![]),
            9 => (knight.clone(), [&ORTHO[..], &DIAG[..]].concat()),
            10 => {
                let mut h = leapers(2, 0);
                h.extend(leapers(2, 2));
                h.extend(leapers(3, 0));
                h.extend(leapers(3, 3));
                (h, vec![])
            }
            11 => (knight.clone(), ORTHO.to_vec()),
            12 => (knight.clone(), DIAG.to_vec()),
            14 => (leapers(1, 3), vec![]),
            15 => (leapers(1, 4), vec![]),
            16 => (leapers(2, 3), vec![]),
            18 => (vec![], knight.clone()),
            _ => (vec![], vec![]), // P, HU, RO have their own rules
        };
        v.push(Def {
            leaps,
            slides,
            royal: kind < 3,
            slow: matches!(kind, K | GU | P),
        });
    }
    v
}

fn is_prime(n: i32) -> bool {
    n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

/// The rose's 16 spirals as cumulative waypoints.
fn rose_spirals() -> Vec<Vec<(i32, i32)>> {
    let hops = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)];
    let mut out = Vec::new();
    for i in 0..8i32 {
        for dir in [1, -1] {
            let (mut x, mut y) = (0, 0);
            let mut spiral = Vec::new();
            for c in 0..7 {
                let (hx, hy) = hops[((i + dir * c).rem_euclid(8)) as usize];
                x += hx;
                y += hy;
                spiral.push((x, y));
            }
            out.push(spiral);
        }
    }
    out
}

// Positions --------------------------------------------------------------------

const A: u8 = 0; // attacker (White)
const D: u8 = 1; // defender (Black, royals only)

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Pc {
    kind: u8,
    side: u8,
    x: i32,
    y: i32,
}

#[derive(Clone, Copy, Debug)]
struct Pos {
    pcs: [Pc; 5],
    n: usize,
    d_to_move: bool,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Pos {
    fn hash(&self) -> u64 {
        let mut h = if self.d_to_move { 0x5555 } else { 0 };
        for p in &self.pcs[..self.n] {
            let k = p.kind as u64 | (p.side as u64) << 5 | ((p.x + 4096) as u64) << 6 | ((p.y + 4096) as u64) << 20;
            h ^= splitmix(k);
        }
        splitmix(h)
    }

    fn at(&self, x: i32, y: i32) -> Option<usize> {
        (0..self.n).find(|&i| self.pcs[i].x == x && self.pcs[i].y == y)
    }
}

struct World {
    defs: Vec<Def>,
    spirals: Vec<Vec<(i32, i32)>>,
}

impl World {
    /// Whether piece `i` attacks (tx, ty) along a clear path.
    fn attacks(&self, pos: &Pos, i: usize, tx: i32, ty: i32) -> bool {
        let p = pos.pcs[i];
        let (dx, dy) = (tx - p.x, ty - p.y);
        if dx == 0 && dy == 0 {
            return false;
        }
        let def = &self.defs[p.kind as usize];
        if def.leaps.contains(&(dx, dy)) {
            return true;
        }
        let pawn_dir = if p.side == A { 1 } else { -1 };
        if p.kind == P {
            return dy == pawn_dir && dx.abs() == 1;
        }
        for &(ax, ay) in &def.slides {
            let k = if ax != 0 { dx / ax } else { dy / ay };
            if k >= 1 && k * ax == dx && k * ay == dy && (1..k).all(|s| pos.at(p.x + ax * s, p.y + ay * s).is_none()) {
                return true;
            }
        }
        if p.kind == HU && (dx == 0) != (dy == 0) {
            let d = (dx + dy).abs();
            if is_prime(d) {
                let (sx, sy) = (dx.signum(), dy.signum());
                if (2..d).filter(|&s| is_prime(s)).all(|s| pos.at(p.x + sx * s, p.y + sy * s).is_none()) {
                    return true;
                }
            }
        }
        if p.kind == RO {
            for spiral in &self.spirals {
                if let Some(h) = spiral.iter().position(|&w| w == (dx, dy))
                    && spiral[..h].iter().all(|&(wx, wy)| pos.at(p.x + wx, p.y + wy).is_none())
                {
                    return true;
                }
            }
        }
        false
    }

    fn attacked_by(&self, pos: &Pos, side: u8, x: i32, y: i32) -> bool {
        (0..pos.n).any(|i| pos.pcs[i].side == side && self.attacks(pos, i, x, y))
    }

    fn royal_attacked(&self, pos: &Pos, side: u8) -> bool {
        (0..pos.n).any(|i| {
            let p = pos.pcs[i];
            p.side == side && self.defs[p.kind as usize].royal && self.attacked_by(pos, 1 - side, p.x, p.y)
        })
    }

    /// Every target square of piece `i`, ignoring legality.
    fn targets(&self, pos: &Pos, i: usize, out: &mut Vec<(i32, i32)>, reach: i32) {
        let p = pos.pcs[i];
        let def = &self.defs[p.kind as usize];
        out.clear();
        for &(ax, ay) in &def.leaps {
            out.push((p.x + ax, p.y + ay));
        }
        for &(ax, ay) in &def.slides {
            for s in 1..=reach {
                let (x, y) = (p.x + ax * s, p.y + ay * s);
                out.push((x, y));
                if pos.at(x, y).is_some() {
                    break;
                }
            }
        }
        if p.kind == P {
            let dir = if p.side == A { 1 } else { -1 };
            if pos.at(p.x, p.y + dir).is_none() {
                out.push((p.x, p.y + dir));
            }
            for sx in [-1, 1] {
                if pos.at(p.x + sx, p.y + dir).is_some() {
                    out.push((p.x + sx, p.y + dir));
                }
            }
        }
        if p.kind == HU {
            for (sx, sy) in ORTHO {
                for d in (2..=reach).filter(|&d| is_prime(d)) {
                    let (x, y) = (p.x + sx * d, p.y + sy * d);
                    out.push((x, y));
                    if pos.at(x, y).is_some() {
                        break;
                    }
                }
            }
        }
        if p.kind == RO {
            for spiral in &self.spirals {
                for &(wx, wy) in spiral {
                    out.push((p.x + wx, p.y + wy));
                    if pos.at(p.x + wx, p.y + wy).is_some() {
                        break;
                    }
                }
            }
        }
    }

    /// Legal moves of the side to move. `escape` is set when a defender royal can
    /// step out of the box, which the proof search counts as the defender getting away.
    fn children(&self, pos: &Pos, boxed: bool, out: &mut Vec<Pos>, escape: &mut bool) {
        out.clear();
        *escape = false;
        let side = if pos.d_to_move { D } else { A };
        let mut targets = Vec::new();
        let reach = if boxed { 2 * BOX + 1 } else { 64 };
        for i in 0..pos.n {
            if pos.pcs[i].side != side {
                continue;
            }
            self.targets(pos, i, &mut targets, reach);
            for &(x, y) in &targets {
                let outside = x.abs() > BOX || y.abs() > BOX;
                if boxed && outside {
                    if side == D {
                        *escape = true;
                    }
                    continue;
                }
                let mut next = *pos;
                if let Some(j) = pos.at(x, y) {
                    let q = pos.pcs[j];
                    if q.side == side || self.defs[q.kind as usize].royal {
                        continue;
                    }
                    next.pcs[j] = next.pcs[next.n - 1];
                    next.n -= 1;
                }
                let me = (0..next.n)
                    .find(|&k| next.pcs[k] == pos.pcs[i])
                    .unwrap();
                next.pcs[me].x = x;
                next.pcs[me].y = y;
                next.d_to_move = !pos.d_to_move;
                if !self.royal_attacked(&next, side) {
                    out.push(next);
                }
            }
        }
    }
}

// Proof-number search ------------------------------------------------------------

const INF: u32 = u32::MAX / 4;
/// Ply limits tried in turn, each with its share of the node budget (percent): short
/// mates are cheap to prove shallow, and a hard shallow stage cannot starve a deep one.
const PROOF_STAGES: [(u32, u64); 3] = [(21, 20), (41, 30), (81, 50)];
/// Largest number that is not a result: sums saturate here, never at `INF`.
const BIG: u32 = INF - 1;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    pn: u32,
    dn: u32,
}

struct Prover<'a> {
    world: &'a World,
    tt: Vec<Entry>,
    nodes: u64,
    budget: u64,
}

/// A position's key at a given remaining depth: depth is part of the node, so no
/// line can return to a node and the search has no cycles.
fn node_key(pos: &Pos, depth: u32) -> u64 {
    splitmix(pos.hash() ^ (depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

impl Prover<'_> {
    fn lookup(&self, h: u64) -> Option<(u32, u32)> {
        let e = self.tt[(h as usize) & (self.tt.len() - 1)];
        (e.key == h).then_some((e.pn, e.dn))
    }

    fn store(&mut self, h: u64, pn: u32, dn: u32) {
        let n = self.tt.len();
        self.tt[(h as usize) & (n - 1)] = Entry { key: h, pn, dn };
    }

    /// Proof and disproof numbers of a position not yet searched.
    fn init(&mut self, pos: &Pos, depth: u32) -> (u32, u32) {
        let mut kids = Vec::new();
        let mut escape = false;
        self.world.children(pos, true, &mut kids, &mut escape);
        if pos.d_to_move {
            if escape {
                return (INF, 0);
            }
            if kids.is_empty() {
                return if self.world.royal_attacked(pos, D) { (0, INF) } else { (INF, 0) };
            }
            if depth == 0 {
                return (INF, 0);
            }
            (kids.len() as u32, 1)
        } else {
            if kids.is_empty() || depth == 0 {
                return (INF, 0);
            }
            (1, (kids.len() as u32).min(8))
        }
    }

    fn numbers(&mut self, kid: &Pos, depth: u32) -> (u32, u32) {
        let h = node_key(kid, depth);
        match self.lookup(h) {
            Some(v) => v,
            None => {
                let v = self.init(kid, depth);
                self.store(h, v.0, v.1);
                v
            }
        }
    }

    fn mid(&mut self, pos: &Pos, depth: u32, th_pn: u32, th_dn: u32) -> (u32, u32) {
        let h = node_key(pos, depth);
        self.nodes += 1;
        let mut kids = Vec::new();
        let mut escape = false;
        self.world.children(pos, true, &mut kids, &mut escape);
        let or_node = !pos.d_to_move;
        if pos.d_to_move && escape {
            self.store(h, INF, 0);
            return (INF, 0);
        }
        if kids.is_empty() || depth == 0 {
            let mated = kids.is_empty() && pos.d_to_move && self.world.royal_attacked(pos, D);
            let v = if mated { (0, INF) } else { (INF, 0) };
            self.store(h, v.0, v.1);
            return v;
        }
        let result = loop {
            let nums: Vec<(u32, u32)> = kids.iter().map(|k| self.numbers(k, depth - 1)).collect();
            let sum = |f: fn(&(u32, u32)) -> u32| -> u32 {
                if nums.iter().any(|n| f(n) == INF) {
                    INF
                } else {
                    nums.iter().fold(0u32, |a, n| a.saturating_add(f(n))).min(BIG)
                }
            };
            let (pn, dn) = if or_node {
                (nums.iter().map(|n| n.0).min().unwrap(), sum(|n| n.1))
            } else {
                (sum(|n| n.0), nums.iter().map(|n| n.1).min().unwrap())
            };
            if pn >= th_pn || dn >= th_dn || self.nodes >= self.budget {
                break (pn, dn);
            }
            // Pick the child on the most proving (OR) or disproving (AND) side.
            let key = |n: &(u32, u32)| if or_node { n.0 } else { n.1 };
            let mut best = 0;
            let mut second = INF;
            for (i, n) in nums.iter().enumerate() {
                if key(n) < key(&nums[best]) {
                    second = key(&nums[best]);
                    best = i;
                } else if i != best && key(n) < second {
                    second = key(n);
                }
            }
            let (c_pn, c_dn) = nums[best];
            // 1+epsilon: let the child run past its sibling a little before switching.
            let widen = |x: u32| x.saturating_add(x / 4).saturating_add(1).min(BIG);
            let (t_pn, t_dn) = if or_node {
                (th_pn.min(widen(second)), (th_dn - dn).saturating_add(c_dn).min(BIG))
            } else {
                ((th_pn - pn).saturating_add(c_pn).min(BIG), th_dn.min(widen(second)))
            };
            let kid = kids[best];
            self.mid(&kid, depth - 1, t_pn, t_dn);
        };
        self.store(h, result.0, result.1);
        result
    }

    /// Whether the attacker forces mate from `pos` within the node budget, trying
    /// short mates first.
    fn proves(&mut self, pos: &Pos) -> bool {
        let total = self.budget;
        let mut used = 0;
        for (depth, share) in PROOF_STAGES {
            self.nodes = 0;
            self.budget = total * share / 100;
            let proved = loop {
                let (pn, dn) = self.mid(pos, depth, INF, INF);
                if pn == 0 || dn == 0 || self.nodes >= self.budget {
                    if std::env::var("FORCED_DEBUG").is_ok() {
                        eprintln!("  depth {depth}: pn {pn} dn {dn} nodes {} {:?}", self.nodes, &pos.pcs[..pos.n]);
                    }
                    break pn == 0;
                }
            };
            used += self.nodes;
            if proved {
                self.nodes = used;
                self.budget = total;
                return true;
            }
        }
        self.nodes = used;
        self.budget = total;
        false
    }
}

// Walk-back on a frame -------------------------------------------------------------
//
// The defender's royal sits at the centre of a frame that moves with it, and keeps
// to moves within radius 2. Attackers inside the frame are exact. Outside, a piece
// is only known to be far (a slow piece also keeps the side it is on), and every
// turn the attacker may place any far piece on the frame's outer ring for free:
// that covers a far piece the defender walks toward. Beyond the ring a leaper
// (reach <= 4) cannot touch the defender's 5x5 and a slider attacks it along at
// most one line, which a placement on that line covers, so far pieces themselves
// never attack. The attacker only gains power here, so a start it cannot win here
// is not won on the open board either.

const FR: i32 = 6;
const FSIDE: i32 = 2 * FR + 1;
const FSQ: usize = (FSIDE * FSIDE) as usize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Slider,
    Leaper,
    Slow,
}

/// One attacking piece's frame data.
struct FKind {
    kind: u8,
    class: Class,
    royal: bool,
    /// Frame squares a far leaper or slider can jump in to from the ring or beyond.
    leap_entry: Vec<(i32, i32)>,
    /// Moves of a slow piece (a pawn: forward one or two).
    steps: Vec<(i32, i32)>,
    /// For a slow piece, per side-set pair, the frame squares it can step in to.
    slow_entry: Vec<Vec<(i32, i32)>>,
}

/// Where a piece is: an exact frame square, or far (a slow piece with, per
/// coordinate, a set over {below -FR, within, above FR}, bits 1, 2, 4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cell {
    In(i32, i32),
    Far,
    FarSlow(u8, u8),
    Captured,
}

fn sq_index(x: i32, y: i32) -> usize {
    ((y + FR) * FSIDE + (x + FR)) as usize
}

fn in_frame(x: i32, y: i32) -> bool {
    x.abs() <= FR && y.abs() <= FR
}

fn cheb(x: i32, y: i32) -> i32 {
    x.abs().max(y.abs())
}

/// The side set of one coordinate after it moves by `s`.
fn shift_set(set: u8, s: i32) -> u8 {
    let mut out = 0;
    if set & 1 != 0 {
        out |= if s <= 0 { 1 } else { 1 | 2 };
    }
    if set & 2 != 0 {
        out |= match s.signum() {
            0 => 2,
            1 => 2 | 4,
            _ => 1 | 2,
        };
    }
    if set & 4 != 0 {
        out |= if s >= 0 { 4 } else { 4 | 2 };
    }
    out
}

fn side_of(v: i32) -> u8 {
    if v < -FR {
        1
    } else if v > FR {
        4
    } else {
        2
    }
}

/// Classes a coordinate value belongs to (the classes overlap at +-FR).
fn classes_of(v: i32) -> u8 {
    (if v <= -FR { 1 } else { 0 }) | (if v.abs() <= FR { 2 } else { 0 }) | (if v >= FR { 4 } else { 0 })
}

/// Whether `p`, at Chebyshev distance >= FR, fits the side sets.
fn in_region(sx: u8, sy: u8, p: (i32, i32)) -> bool {
    cheb(p.0, p.1) >= FR && classes_of(p.0) & sx != 0 && classes_of(p.1) & sy != 0
}

struct Frame<'a> {
    world: &'a World,
    pieces: Vec<FKind>,
    royal: u8,
    any_royal: bool,
    sizes: Vec<usize>,
    /// Defender targets within radius 2.
    targets: Vec<Target>,
}

impl<'a> Frame<'a> {
    fn new(world: &'a World, attackers: &[u8], royal: u8) -> Self {
        let mut pieces = Vec::new();
        for &kind in attackers {
            let def = &world.defs[kind as usize];
            let class = if def.slow {
                Class::Slow
            } else if !def.slides.is_empty() {
                Class::Slider
            } else {
                Class::Leaper
            };
            let mut leap_entry = Vec::new();
            for x in -FR..=FR {
                for y in -FR..=FR {
                    if def.leaps.iter().any(|&(ox, oy)| cheb(x - ox, y - oy) >= FR) {
                        leap_entry.push((x, y));
                    }
                }
            }
            let steps = if kind == P { vec![(0, 1), (0, 2)] } else { KING.to_vec() };
            let mut slow_entry = Vec::new();
            if class == Class::Slow {
                for sx in 1..8u8 {
                    for sy in 1..8u8 {
                        let mut cells = Vec::new();
                        for x in -FR..=FR {
                            for y in -FR..=FR {
                                if steps.iter().any(|&(ox, oy)| in_region(sx, sy, (x - ox, y - oy))) {
                                    cells.push((x, y));
                                }
                            }
                        }
                        slow_entry.push(cells);
                    }
                }
            }
            pieces.push(FKind { kind, class, royal: def.royal, leap_entry, steps, slow_entry });
        }
        let sizes = pieces
            .iter()
            .map(|p| FSQ + if p.class == Class::Slow { 49 } else { 1 } + 1)
            .collect();
        let mut targets: Vec<Target> = KING.iter().map(|&k| (k, None)).collect();
        match royal {
            RC => targets.extend(leapers(1, 2).into_iter().map(|k| (k, None))),
            RQ => targets.extend(KING.iter().map(|&(x, y)| ((2 * x, 2 * y), Some((x, y))))),
            _ => {}
        }
        let any_royal = pieces.iter().any(|p| p.royal);
        Frame { world, pieces, royal, any_royal, sizes, targets }
    }

    fn encode_cell(&self, i: usize, c: Cell) -> usize {
        match c {
            Cell::In(x, y) => sq_index(x, y),
            Cell::Far => FSQ,
            Cell::FarSlow(sx, sy) => FSQ + ((sx - 1) as usize) * 7 + (sy - 1) as usize,
            Cell::Captured => self.sizes[i] - 1,
        }
    }

    fn decode_cell(&self, i: usize, v: usize) -> Cell {
        if v == self.sizes[i] - 1 {
            Cell::Captured
        } else if v < FSQ {
            Cell::In((v as i32) % FSIDE - FR, (v as i32) / FSIDE - FR)
        } else if self.pieces[i].class == Class::Slow {
            let r = v - FSQ;
            Cell::FarSlow((r / 7 + 1) as u8, (r % 7 + 1) as u8)
        } else {
            Cell::Far
        }
    }

    fn encode(&self, cells: &[Cell]) -> usize {
        let mut idx = 0;
        for i in (0..cells.len()).rev() {
            idx = idx * self.sizes[i] + self.encode_cell(i, cells[i]);
        }
        idx
    }

    fn decode(&self, mut idx: usize, out: &mut [Cell]) {
        for (i, c) in out.iter_mut().enumerate() {
            *c = self.decode_cell(i, idx % self.sizes[i]);
            idx /= self.sizes[i];
        }
    }

    fn total(&self) -> usize {
        self.sizes.iter().product()
    }

    /// Whether an attacker at `from` attacks `to`; squares outside the frame count
    /// as empty, which only helps the attacker.
    fn piece_attacks(&self, kind: u8, from: (i32, i32), to: (i32, i32), occ: &dyn Fn(i32, i32) -> bool) -> bool {
        let (dx, dy) = (to.0 - from.0, to.1 - from.1);
        if dx == 0 && dy == 0 {
            return false;
        }
        let def = &self.world.defs[kind as usize];
        if def.leaps.contains(&(dx, dy)) {
            return true;
        }
        if kind == P {
            return dy == 1 && dx.abs() == 1;
        }
        def.slides.iter().any(|&(ax, ay)| {
            let k = if ax != 0 { dx / ax } else { dy / ay };
            k >= 1 && k * ax == dx && k * ay == dy && (1..k).all(|s| !occ(from.0 + ax * s, from.1 + ay * s))
        })
    }

    /// Whether square `t` is attacked by an attacker inside the frame.
    fn attacked(&self, cells: &[Cell], t: (i32, i32), occ: &dyn Fn(i32, i32) -> bool) -> bool {
        cells.iter().enumerate().any(|(i, c)| match *c {
            Cell::In(x, y) => self.piece_attacks(self.pieces[i].kind, (x, y), t, occ),
            _ => false,
        })
    }

    /// Whether the defender's royal at `r` attacks `t`.
    fn royal_attacks(&self, r: (i32, i32), t: (i32, i32), occ: &dyn Fn(i32, i32) -> bool) -> bool {
        let def = &self.world.defs[self.royal as usize];
        let (dx, dy) = (t.0 - r.0, t.1 - r.1);
        if def.leaps.contains(&(dx, dy)) {
            return true;
        }
        def.slides.iter().any(|&(ax, ay)| {
            let k = if ax != 0 { dx / ax } else { dy / ay };
            k >= 1 && k * ax == dx && k * ay == dy && (1..k).all(|s| !occ(r.0 + ax * s, r.1 + ay * s))
        })
    }

    fn occupancy(cells: &[Cell], royal: (i32, i32)) -> impl Fn(i32, i32) -> bool + '_ {
        move |x, y| (x, y) == royal || cells.contains(&Cell::In(x, y))
    }

    fn valid(cells: &[Cell]) -> bool {
        cells.iter().enumerate().all(|(i, a)| match *a {
            Cell::In(x, y) => (x, y) != (0, 0) && !cells[i + 1..].contains(a),
            _ => true,
        })
    }

    /// Whether an attacker royal inside the frame is attacked by the defender's royal.
    fn attacker_royal_hit(&self, cells: &[Cell]) -> bool {
        if !self.any_royal {
            return false;
        }
        let occ = Self::occupancy(cells, (0, 0));
        cells.iter().enumerate().any(|(i, c)| match *c {
            Cell::In(x, y) => self.pieces[i].royal && self.royal_attacks((0, 0), (x, y), &occ),
            _ => false,
        })
    }

    /// The attacker's position after the defender's royal moves by `m`.
    fn recenter(&self, cells: &[Cell], m: (i32, i32), out: &mut [Cell]) {
        for (i, c) in cells.iter().enumerate() {
            out[i] = match *c {
                Cell::In(x, y) => {
                    let (nx, ny) = (x - m.0, y - m.1);
                    if in_frame(nx, ny) {
                        Cell::In(nx, ny)
                    } else if self.pieces[i].class == Class::Slow {
                        Cell::FarSlow(side_of(nx), side_of(ny))
                    } else {
                        Cell::Far
                    }
                }
                Cell::FarSlow(sx, sy) => Cell::FarSlow(shift_set(sx, -m.0), shift_set(sy, -m.1)),
                other => other,
            };
        }
    }

    /// Every attacker move from an attacker-to-move position, as the position after it.
    fn attacker_moves(&self, cells: &[Cell], out: &mut Vec<usize>) {
        out.clear();
        let mut buf = [Cell::Captured; 4];
        let next = &mut buf[..cells.len()];
        next.copy_from_slice(cells);
        let occ_now = Self::occupancy(cells, (0, 0));
        let empty = |x: i32, y: i32| !occ_now(x, y);
        for i in 0..cells.len() {
            let p = &self.pieces[i];
            let def = &self.world.defs[p.kind as usize];
            let mut place = |c: Cell, out: &mut Vec<usize>| {
                next[i] = c;
                if Self::valid(next) && !self.attacker_royal_hit(next) {
                    out.push(self.encode(next));
                }
                next[i] = cells[i];
            };
            match cells[i] {
                Cell::In(x, y) => {
                    let mut land = |tx: i32, ty: i32, out: &mut Vec<usize>| {
                        if in_frame(tx, ty) {
                            if empty(tx, ty) {
                                place(Cell::In(tx, ty), out);
                            }
                        } else if p.class == Class::Slow {
                            place(Cell::FarSlow(side_of(tx), side_of(ty)), out);
                        } else {
                            place(Cell::Far, out);
                        }
                    };
                    for &(ax, ay) in &def.leaps {
                        land(x + ax, y + ay, out);
                    }
                    for &(ax, ay) in &def.slides {
                        let mut s = 1;
                        loop {
                            let (tx, ty) = (x + ax * s, y + ay * s);
                            if !in_frame(tx, ty) {
                                land(tx, ty, out);
                                break;
                            }
                            if !empty(tx, ty) {
                                break;
                            }
                            land(tx, ty, out);
                            s += 1;
                        }
                    }
                    if p.kind == P && (!in_frame(x, y + 1) || empty(x, y + 1)) {
                        land(x, y + 1, out);
                        if !in_frame(x, y + 2) || empty(x, y + 2) {
                            land(x, y + 2, out);
                        }
                    }
                }
                Cell::Far => {
                    // Staying far changes nothing a far piece is summarised by.
                    place(Cell::Far, out);
                    for &(x, y) in &p.leap_entry {
                        if empty(x, y) {
                            place(Cell::In(x, y), out);
                        }
                    }
                    // A far slider enters along a clear ray; allowing every free
                    // square of its colour only adds moves.
                    if p.class == Class::Slider {
                        for x in -FR..=FR {
                            for y in -FR..=FR {
                                let parity_ok = match p.kind {
                                    5 => (x + y).rem_euclid(2) == 0,
                                    6 => (x + y).rem_euclid(2) == 1,
                                    _ => true,
                                };
                                if parity_ok && empty(x, y) && !p.leap_entry.contains(&(x, y)) {
                                    place(Cell::In(x, y), out);
                                }
                            }
                        }
                    }
                }
                Cell::FarSlow(sx, sy) => {
                    for &(ox, oy) in &p.steps {
                        place(Cell::FarSlow(shift_set(sx, ox), shift_set(sy, oy)), out);
                    }
                    for &(x, y) in &p.slow_entry[((sx - 1) * 7 + (sy - 1)) as usize] {
                        if empty(x, y) {
                            place(Cell::In(x, y), out);
                        }
                    }
                }
                Cell::Captured => {}
            }
        }
    }

    /// The attacker-to-move positions a defender-to-move position can come from by
    /// the free ring placement: each piece on the ring may instead have been far.
    fn unplaced(&self, cells: &[Cell], out: &mut Vec<usize>) {
        out.clear();
        let mut options: Vec<Vec<Cell>> = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            let mut o = vec![*c];
            if let Cell::In(x, y) = *c
                && cheb(x, y) == FR
            {
                if self.pieces[i].class == Class::Slow {
                    for sx in 1..8u8 {
                        for sy in 1..8u8 {
                            if classes_of(x) & sx != 0 && classes_of(y) & sy != 0 {
                                o.push(Cell::FarSlow(sx, sy));
                            }
                        }
                    }
                } else {
                    o.push(Cell::Far);
                }
            }
            options.push(o);
        }
        let mut pick = vec![0usize; cells.len()];
        let mut cur: Vec<Cell> = cells.to_vec();
        loop {
            for i in 0..cells.len() {
                cur[i] = options[i][pick[i]];
            }
            out.push(self.encode(&cur));
            let mut i = 0;
            while i < cells.len() {
                pick[i] += 1;
                if pick[i] < options[i].len() {
                    break;
                }
                pick[i] = 0;
                i += 1;
            }
            if i == cells.len() {
                break;
            }
        }
    }

    /// Whether the defender is mated or every legal move reaches a won attacker
    /// position.
    fn defender_lost(&self, cells: &[Cell], win_a: &[u64], scratch: &mut [Cell]) -> bool {
        if !Self::valid(cells) || self.attacker_royal_hit(cells) {
            return false;
        }
        let in_check = {
            let occ = Self::occupancy(cells, (0, 0));
            self.attacked(cells, (0, 0), &occ)
        };
        let mut any_legal = false;
        let mut buf = [Cell::Captured; 4];
        let after = &mut buf[..cells.len()];
        for &(t, mid) in &self.targets {
            after.copy_from_slice(cells);
            if let Some(c) = cells.iter().position(|c| *c == Cell::In(t.0, t.1)) {
                if self.pieces[c].royal {
                    continue;
                }
                after[c] = Cell::Captured;
            }
            if let Some(m) = mid
                && after.contains(&Cell::In(m.0, m.1))
            {
                continue;
            }
            let occ = Self::occupancy(after, t);
            let occ_moved = |x: i32, y: i32| (x, y) != (0, 0) && occ(x, y);
            if self.attacked(after, t, &occ_moved) {
                continue;
            }
            any_legal = true;
            self.recenter(after, t, scratch);
            if !bit(win_a, self.encode(scratch)) {
                return false;
            }
        }
        // Every legal move loses; with none it is mate in check, stalemate otherwise.
        any_legal || in_check
    }
}

fn bit(v: &[u64], i: usize) -> bool {
    v[i >> 6] >> (i & 63) & 1 == 1
}

fn set_bit(v: &mut [u64], i: usize) {
    v[i >> 6] |= 1 << (i & 63);
}

/// Whether the defender escapes from every start in the frame game.
fn frame_escapes(world: &World, set: &[u8], starts: &[Pos]) -> bool {
    let royal = set.iter().find(|&&s| s / KINDS == 1).unwrap() % KINDS;
    let attackers: Vec<u8> = set.iter().filter(|&&s| s / KINDS == 0).map(|&s| s % KINDS).collect();
    let frame = Frame::new(world, &attackers, royal);
    let start_ids: Vec<usize> = starts
        .iter()
        .map(|s| {
            let mut cells = Vec::new();
            let mut used = vec![false; s.n];
            for &k in &attackers {
                let j = (0..s.n).find(|&j| !used[j] && s.pcs[j].side == A && s.pcs[j].kind == k).unwrap();
                used[j] = true;
                assert!(in_frame(s.pcs[j].x, s.pcs[j].y), "start outside the frame");
                cells.push(Cell::In(s.pcs[j].x, s.pcs[j].y));
            }
            frame.encode(&cells)
        })
        .collect();
    match walk_back(&frame, &start_ids) {
        Some(win_d) => every_spread_start_escapes(&frame, &win_d),
        None => false,
    }
}

/// Whether no spread position in the frame (defender to move, pieces 3+ apart, no
/// two on a line, nothing attacked) is lost: the sampled starts stand for all.
fn every_spread_start_escapes(frame: &Frame, win_d: &[u64]) -> bool {
    let n = frame.pieces.len();
    let mut cells = vec![Cell::Captured; n];
    for d in 0..frame.total() {
        if !bit(win_d, d) {
            continue;
        }
        frame.decode(d, &mut cells);
        let mut sq: Vec<(i32, i32)> = vec![(0, 0)];
        for c in &cells {
            match *c {
                Cell::In(x, y) => sq.push((x, y)),
                _ => break,
            }
        }
        if sq.len() != n + 1 {
            continue;
        }
        let apart = (0..sq.len()).all(|i| {
            (i + 1..sq.len()).all(|j| {
                let (dx, dy) = (sq[i].0 - sq[j].0, sq[i].1 - sq[j].1);
                cheb(dx, dy) >= SPREAD && dx != 0 && dy != 0 && dx.abs() != dy.abs()
            })
        });
        let parity_ok = (0..n).all(|i| match frame.pieces[i].kind {
            5 => (sq[i + 1].0 + sq[i + 1].1).rem_euclid(2) == 0,
            6 => (sq[i + 1].0 + sq[i + 1].1).rem_euclid(2) == 1,
            _ => true,
        });
        if !apart || !parity_ok {
            continue;
        }
        let occ = Frame::occupancy(&cells, (0, 0));
        let quiet = (0..n).all(|i| {
            !frame.piece_attacks(frame.pieces[i].kind, sq[i + 1], (0, 0), &occ)
                && !frame.royal_attacks((0, 0), sq[i + 1], &occ)
                && (0..n).all(|j| i == j || !frame.piece_attacks(frame.pieces[i].kind, sq[i + 1], sq[j + 1], &occ))
        });
        if quiet {
            return false;
        }
    }
    true
}

/// The fixed point of attacker wins: the lost defender positions, or `None` as
/// soon as a start is lost.
fn walk_back(frame: &Frame, start_ids: &[usize]) -> Option<Vec<u64>> {
    let total = frame.total();
    let words = total.div_ceil(64);
    // Defender to move lost; attacker to move won; positions the attacker can reach
    // a lost defender position from by placing far pieces on the ring.
    let mut win_d = vec![0u64; words];
    let mut win_any = vec![0u64; words];
    let mut win_a = vec![0u64; words];
    let n = frame.pieces.len();
    let mut cells = vec![Cell::Captured; n];
    let mut scratch = vec![Cell::Captured; n];
    let mut list = Vec::new();
    loop {
        let mut changed = false;
        for d in 0..total {
            if bit(&win_d, d) {
                continue;
            }
            frame.decode(d, &mut cells);
            if frame.defender_lost(&cells, &win_a, &mut scratch) {
                set_bit(&mut win_d, d);
                frame.unplaced(&cells, &mut list);
                for &b in &list {
                    set_bit(&mut win_any, b);
                }
                changed = true;
            }
        }
        if start_ids.iter().any(|&d| bit(&win_d, d)) {
            return None;
        }
        for a in 0..total {
            if bit(&win_a, a) {
                continue;
            }
            frame.decode(a, &mut cells);
            if !Frame::valid(&cells) {
                continue;
            }
            frame.attacker_moves(&cells, &mut list);
            if list.iter().any(|&b| bit(&win_any, b)) {
                set_bit(&mut win_a, a);
                changed = true;
            }
        }
        if !changed {
            return Some(win_d);
        }
    }
}

// Cores and starts -----------------------------------------------------------------

fn label(set: &[u8]) -> String {
    let side = |c: u8| -> String {
        let mut v: Vec<u8> = set.iter().filter(|&&s| s / KINDS == c).map(|&s| s % KINDS).collect();
        v.sort();
        v.iter()
            .map(|&k| {
                let code = KIND_CODES[k as usize];
                if c == 0 { code.to_string() } else { code.to_lowercase() }
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    format!("{} vs {}", side(0), side(1))
}

/// Spread starts: defender to move, pieces apart, no two on a shared line, nothing
/// attacked. `None` when the sampler cannot place the set.
fn spread_start(world: &World, set: &[u8], seed: u64) -> Option<Pos> {
    let mut rng = seed;
    let mut next = |n: i32| -> i32 {
        rng = splitmix(rng);
        (rng % (2 * n as u64 + 1)) as i32 - n
    };
    'attempt: for _ in 0..20_000 {
        let mut pos = Pos { pcs: [Pc::default(); 5], n: 0, d_to_move: true };
        let mut order: Vec<u8> = set.iter().copied().filter(|&s| s / KINDS == 1).collect();
        order.extend(set.iter().copied().filter(|&s| s / KINDS == 0));
        for (i, &s) in order.iter().enumerate() {
            let (kind, side) = (s % KINDS, s / KINDS);
            let (x, y) = if i == 0 { (0, 0) } else { (next(START_RADIUS), next(START_RADIUS)) };
            let parity_ok = match kind {
                5 => (x + y).rem_euclid(2) == 0,
                6 => (x + y).rem_euclid(2) == 1,
                _ => true,
            };
            let apart = pos.pcs[..pos.n].iter().all(|q| {
                let (dx, dy) = (q.x - x, q.y - y);
                dx.abs().max(dy.abs()) >= SPREAD && dx != 0 && dy != 0 && dx.abs() != dy.abs()
            });
            if !parity_ok || !apart {
                continue 'attempt;
            }
            pos.pcs[pos.n] = Pc { kind, side, x, y };
            pos.n += 1;
        }
        let quiet = (0..pos.n).all(|i| {
            (0..pos.n).all(|j| i == j || !world.attacks(&pos, i, pos.pcs[j].x, pos.pcs[j].y))
        });
        if quiet {
            return Some(pos);
        }
    }
    None
}

enum Verdict {
    Forced(u64),
    Escape,
    Unknown(&'static str),
}

/// The bishop-colour canonical form of a set, as cores are listed.
fn canonical(set: &[u8]) -> Vec<u8> {
    let mut a = set.to_vec();
    let mut b: Vec<u8> = set
        .iter()
        .map(|&s| match s % KINDS {
            5 => s + 1,
            6 => s - 1,
            _ => s,
        })
        .collect();
    a.sort();
    b.sort();
    a.min(b)
}

/// Whether some one-smaller attacking army against the same royals is a core not
/// proven helpmate-only: then this bigger army cannot be either.
fn smaller_army_wins(set: &[u8], proven: &std::collections::HashSet<Vec<u8>>) -> bool {
    (0..set.len()).filter(|&i| set[i] / KINDS == 0).any(|i| {
        let mut sub = set.to_vec();
        sub.remove(i);
        let is_core = sub.iter().any(|&s| s / KINDS == 0) && !mating_sets::is_dead(&sub);
        is_core && !proven.contains(&canonical(&sub))
    })
}

fn classify(
    world: &World,
    set: &[u8],
    budget: u64,
    tt_bits: u32,
    proven: &std::collections::HashSet<Vec<u8>>,
) -> Verdict {
    let starts: Vec<Pos> = (0..STARTS as u64)
        .filter_map(|i| spread_start(world, set, splitmix(i * 7919 + set.iter().map(|&s| s as u64).sum::<u64>())))
        .collect();
    if starts.len() < STARTS {
        return Verdict::Unknown("no spread start");
    }
    let royals = set.iter().filter(|&&s| s / KINDS == 1).count();
    let mut prover = Prover {
        world,
        tt: vec![Entry::default(); 1 << tt_bits],
        nodes: 0,
        budget,
    };
    let mut total = 0;
    // Only armies the walk-back may judge need the forced screen: the rest stay
    // unknown, which the engine treats as forced (FORCED_ALL proves them anyway).
    let attackers = set.iter().filter(|&&s| s / KINDS == 0).count();
    let screen = royals == 1 && (attackers <= 3 || std::env::var("FORCED_ALL").is_ok());
    let mut forced = screen;
    for s in starts.iter().filter(|_| screen) {
        if !prover.proves(s) {
            forced = false;
            break;
        }
        total += prover.nodes;
    }
    // The frame does not cover a far huygen's prime-distance pattern, the rose's
    // reach of 6 or knightrider lines.
    let walkable = royals == 1
        && attackers <= 3
        && !set.iter().any(|&s| matches!(s % KINDS, 18 | HU | RO))
        && !smaller_army_wins(set, proven);
    let check = std::env::var("FORCED_FRAME_CHECK").is_ok();
    if walkable && (!forced || check) {
        let t = std::time::Instant::now();
        let escaped = frame_escapes(world, set, &starts);
        if check {
            eprintln!("  walk-back {}: escape {escaped}, forced {forced}, {:.1}s", label(set), t.elapsed().as_secs_f64());
        }
        if escaped && forced {
            panic!("{}: proved forced yet the walk-back finds an escape", label(set));
        }
        if escaped {
            return Verdict::Escape;
        }
    }
    if forced { Verdict::Forced(total) } else { Verdict::Unknown("not proven") }
}

/// Runs the forced prover with `budget` on every core of a helpmate-only list; a
/// proof there contradicts the walk-back and fails loudly.
fn verify_escapes(path: &str, threads: usize, budget: u64) {
    let world = World { defs: defs(), spirals: rose_spirals() };
    let cores: Vec<Vec<u8>> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (w, b) = l.split_once(" vs ").unwrap();
            let side = |t: &str, c: u8| -> Vec<u8> {
                t.split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| c * KINDS + KIND_CODES.iter().position(|k| k.eq_ignore_ascii_case(x)).unwrap() as u8)
                    .collect()
            };
            let mut v = side(w, 0);
            v.extend(side(b, 1));
            v.sort();
            v
        })
        .collect();
    let next = AtomicUsize::new(0);
    let bad = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= cores.len() {
                    break;
                }
                let set = &cores[i];
                let seed = set.iter().map(|&s| s as u64).sum::<u64>();
                let starts: Vec<Pos> = (0..STARTS as u64)
                    .filter_map(|k| spread_start(&world, set, splitmix(k * 7919 + seed)))
                    .collect();
                let mut prover = Prover { world: &world, tt: vec![Entry::default(); 1 << 22], nodes: 0, budget };
                if let Some(s) = starts.iter().find(|s| prover.proves(s)) {
                    bad.lock().unwrap().push(format!("{}: forced from {:?}", label(set), &s.pcs[..s.n]));
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    for b in &bad {
        eprintln!("CONTRADICTION {b}");
    }
    eprintln!("{} helpmate-only cores checked with {budget} nodes per start: {} contradictions", cores.len(), bad.len());
    assert!(bad.is_empty());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().expect("usage: forced_gen <out dir> [threads] [budget] [filter]");
    let threads: usize = args.next().map_or(12, |s| s.parse().unwrap());
    let budget: u64 = args.next().map_or(300_000, |s| s.parse().unwrap());
    let filter = args.next();
    std::fs::create_dir_all(&out_dir).unwrap();
    // Cross-check: the forced prover must not prove any helpmate-only core.
    if let Ok(list) = std::env::var("FORCED_VERIFY") {
        verify_escapes(&list, threads, budget);
        return;
    }

    // White attacks; Black keeps only royals. One of each bishop-color mirror pair.
    let mut cores: Vec<Vec<u8>> = Vec::new();
    mating_sets::for_each_set(mating_sets::CAP, |set| {
        let black_royals_only = set.iter().filter(|&&s| s / KINDS == 1).all(|&s| mating_sets::is_royal_symbol(s));
        let has_black = set.iter().any(|&s| s / KINDS == 1);
        let has_white = set.iter().any(|&s| s / KINDS == 0);
        if !(black_royals_only && has_black && has_white) || mating_sets::is_dead(set) {
            return;
        }
        let swapped: Vec<u8> = set
            .iter()
            .map(|&s| match s % KINDS {
                5 => s + 1,
                6 => s - 1,
                _ => s,
            })
            .collect();
        let (mut a, mut b) = (set.to_vec(), swapped);
        a.sort();
        b.sort();
        if a <= b {
            cores.push(a);
        }
    });
    if let Some(f) = &filter {
        let wanted: Vec<&str> = f.split(';').collect();
        cores.retain(|c| wanted.contains(&label(c).as_str()));
    }
    // Phases by attacking army size: a bigger army consults the smaller armies'
    // escape proofs, which are complete by then.
    let attackers = |c: &Vec<u8>| c.iter().filter(|&&s| s / KINDS == 0).count();
    cores.sort_by_key(|c| (attackers(c), c.iter().filter(|&&s| s / KINDS == 1).count()));
    eprintln!("{} cores, {threads} threads, budget {budget} nodes per start", cores.len());

    let world = World { defs: defs(), spirals: rose_spirals() };
    // Each result is appended as it lands, so a stopped run resumes where it left off.
    let progress_path = format!("{out_dir}/progress.tsv");
    let mut done: HashMap<String, Verdict> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(&progress_path) {
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 3 {
                continue;
            }
            let v = match f[1] {
                "F" => Verdict::Forced(f[2].parse().unwrap_or(0)),
                "E" => Verdict::Escape,
                _ if f[2] == "no spread start" => Verdict::Unknown("no spread start"),
                _ => Verdict::Unknown("not proven"),
            };
            done.insert(f[0].to_string(), v);
        }
    }
    let progress = Mutex::new(
        std::fs::OpenOptions::new().create(true).append(true).open(&progress_path).unwrap(),
    );
    let results = Mutex::new(Vec::new());
    let mut escapes = std::collections::HashSet::new();
    for (i, c) in cores.iter().enumerate() {
        if let Some(v) = done.remove(&label(c)) {
            if matches!(v, Verdict::Escape) {
                escapes.insert(c.clone());
            }
            results.lock().unwrap().push((i, v));
        }
    }
    let resumed: std::collections::HashSet<usize> = results.lock().unwrap().iter().map(|r| r.0).collect();
    eprintln!("{} cores already classified", resumed.len());
    let started = std::time::Instant::now();
    for size in 1..=4 {
        let phase: Vec<usize> = (0..cores.len())
            .filter(|&i| attackers(&cores[i]) == size && !resumed.contains(&i))
            .collect();
        let next = AtomicUsize::new(0);
        let found = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= phase.len() {
                        break;
                    }
                    let i = phase[k];
                    let v = classify(&world, &cores[i], budget, 22, &escapes);
                    if matches!(v, Verdict::Escape) {
                        found.lock().unwrap().push(cores[i].clone());
                    }
                    let l = label(&cores[i]);
                    let line = match &v {
                        Verdict::Forced(n) => format!("{l}\tF\t{n}\n"),
                        Verdict::Escape => format!("{l}\tE\t-\n"),
                        Verdict::Unknown(why) => format!("{l}\tU\t{why}\n"),
                    };
                    progress.lock().unwrap().write_all(line.as_bytes()).unwrap();
                    let mut r = results.lock().unwrap();
                    r.push((i, v));
                    if r.len() % 200 == 0 {
                        eprintln!("{}/{} cores, {:.0}s", r.len(), cores.len(), started.elapsed().as_secs_f64());
                    }
                });
            }
        });
        escapes.extend(found.into_inner().unwrap());
        eprintln!("{size}-piece armies done: {} helpmate-only so far, {:.0}s", escapes.len(), started.elapsed().as_secs_f64());
    }

    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let mut forced = std::fs::File::create(format!("{out_dir}/forced.tsv")).unwrap();
    let mut escape = std::fs::File::create(format!("{out_dir}/escape.txt")).unwrap();
    let mut unknown = std::fs::File::create(format!("{out_dir}/unknown.tsv")).unwrap();
    let (mut nf, mut ne, mut nu) = (0, 0, 0);
    for (i, v) in &results {
        let l = label(&cores[*i]);
        match v {
            Verdict::Forced(nodes) => {
                nf += 1;
                writeln!(forced, "{l}\t{nodes}").unwrap();
            }
            Verdict::Escape => {
                ne += 1;
                writeln!(escape, "{l}").unwrap();
            }
            Verdict::Unknown(why) => {
                nu += 1;
                writeln!(unknown, "{l}\t{why}").unwrap();
            }
        }
    }
    eprintln!(
        "done in {:.0}s: {nf} forced, {ne} helpmate-only, {nu} unknown",
        started.elapsed().as_secs_f64()
    );
}
