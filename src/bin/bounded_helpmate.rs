//! Classifies every bounded-board core (White's army against Black's royals only, a
//! set of at most `BOUNDED_CAP` pieces the bounded table says can mate) as forced or
//! helpmate-only, solving it exactly on 8x8: a fixpoint over every placement, both
//! sides to move. Helpmate-only: White forces mate from no spread position (pieces
//! 3+ apart); the report gives how many positions it does win from and the longest
//! of those mates. An army holding a smaller army that is not helpmate-only counts
//! as forced unsolved; pawns are left unknown.
//!
//! Usage: cargo run --release --bin bounded_helpmate -- <out dir> [threads] [previous report.tsv]
//! Writes report.tsv and helpmate_only.txt (labels like `K,N,N vs k`). Given a previous
//! report, only the cores it lacks and the armies holding them are solved again, with
//! the helpmate-only armies their captures lead into; the rest is copied.

use apeiron::evaluation::mating_sets::{self, BOUNDED_CAP, KIND_CODES, KINDS};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const P: u8 = 8;
const HU: u8 = 19;
const RO: u8 = 20;
const SPREAD: i32 = 3;

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

fn is_prime(n: i32) -> bool {
    n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

fn on_board(x: i32, y: i32) -> bool {
    (0..8).contains(&x) && (0..8).contains(&y)
}

/// Move geometry per kind, as `forced_gen` has it, clipped to 8x8.
struct Geometry {
    leaps: Vec<[u64; 64]>,
    slides: Vec<Vec<(i32, i32)>>,
    spirals: Vec<Vec<(i32, i32)>>,
}

impl Geometry {
    fn new() -> Self {
        const KING: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
        const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        const DIAG: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let knight = leapers(1, 2);
        let (mut leaps, mut slides) = (Vec::new(), Vec::new());
        for kind in 0..KINDS {
            let (l, s): (Vec<(i32, i32)>, Vec<(i32, i32)>) = match kind {
                0 | 13 => (KING.to_vec(), vec![]),
                1 | 17 => ([&KING[..], &knight[..]].concat(), vec![]),
                2 | 3 => (vec![], [&ORTHO[..], &DIAG[..]].concat()),
                4 => (vec![], ORTHO.to_vec()),
                5 | 6 => (vec![], DIAG.to_vec()),
                7 => (knight.clone(), vec![]),
                9 => (knight.clone(), [&ORTHO[..], &DIAG[..]].concat()),
                10 => ([leapers(2, 0), leapers(2, 2), leapers(3, 0), leapers(3, 3)].concat(), vec![]),
                11 => (knight.clone(), ORTHO.to_vec()),
                12 => (knight.clone(), DIAG.to_vec()),
                14 => (leapers(1, 3), vec![]),
                15 => (leapers(1, 4), vec![]),
                16 => (leapers(2, 3), vec![]),
                18 => (vec![], knight.clone()),
                _ => (vec![], vec![]),
            };
            let mut table = [0u64; 64];
            for (sq, bits) in table.iter_mut().enumerate() {
                let (x, y) = ((sq % 8) as i32, (sq / 8) as i32);
                for &(dx, dy) in &l {
                    if on_board(x + dx, y + dy) {
                        *bits |= 1 << ((y + dy) * 8 + x + dx);
                    }
                }
            }
            leaps.push(table);
            slides.push(s);
        }
        let hops = [(-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1)];
        let mut spirals = Vec::new();
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
                spirals.push(spiral);
            }
        }
        Geometry { leaps, slides, spirals }
    }

    /// Squares a piece of `kind` on `sq` attacks given occupancy `occ`.
    #[inline]
    fn attacks(&self, kind: u8, sq: usize, occ: u64) -> u64 {
        let (x, y) = ((sq % 8) as i32, (sq / 8) as i32);
        let mut out = self.leaps[kind as usize][sq];
        for &(dx, dy) in &self.slides[kind as usize] {
            let (mut cx, mut cy) = (x + dx, y + dy);
            while on_board(cx, cy) {
                let b = 1u64 << (cy * 8 + cx);
                out |= b;
                if occ & b != 0 {
                    break;
                }
                cx += dx;
                cy += dy;
            }
        }
        if kind == HU {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                for d in (2..8).filter(|&d| is_prime(d)) {
                    let (cx, cy) = (x + dx * d, y + dy * d);
                    if !on_board(cx, cy) {
                        break;
                    }
                    let b = 1u64 << (cy * 8 + cx);
                    out |= b;
                    if occ & b != 0 {
                        break;
                    }
                }
            }
        }
        if kind == RO {
            for spiral in &self.spirals {
                for &(wx, wy) in spiral {
                    let (cx, cy) = (x + wx, y + wy);
                    if !on_board(cx, cy) {
                        break;
                    }
                    let b = 1u64 << (cy * 8 + cx);
                    out |= b;
                    if occ & b != 0 {
                        break;
                    }
                }
            }
        }
        out
    }
}

fn bitset(n: usize) -> Vec<AtomicU64> {
    (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect()
}

#[inline]
fn get(v: &[AtomicU64], i: usize) -> bool {
    v[i >> 6].load(Ordering::Relaxed) >> (i & 63) & 1 == 1
}

#[inline]
fn set(v: &[AtomicU64], i: usize) {
    v[i >> 6].fetch_or(1 << (i & 63), Ordering::Relaxed);
}

/// White-to-move positions White forces mate from, indexed like [`Army`].
type Won = Arc<Vec<AtomicU64>>;

/// One army's placements: piece `i` (symbols sorted, White's first) stands on
/// square `(idx >> 6i) & 63`.
struct Army<'a> {
    geo: &'a Geometry,
    syms: Vec<u8>,
    /// Per White piece, the army left after Black captures it: `None` when that army
    /// can never mate.
    subs: Vec<Option<Won>>,
}

struct Outcome {
    forced: bool,
    won: Won,
    won_count: usize,
    legal_count: usize,
    /// Longest forced mate, in plies, once every won position is found.
    longest: usize,
}

impl Army<'_> {
    fn kind(&self, i: usize) -> u8 {
        self.syms[i] % KINDS
    }

    fn white(&self, i: usize) -> bool {
        self.syms[i] < KINDS
    }

    fn royal(&self, i: usize) -> bool {
        mating_sets::is_royal_symbol(self.syms[i])
    }

    /// Distinct squares, bishops on their own color.
    fn placeable(&self, sq: &[usize]) -> bool {
        (0..sq.len()).all(|i| {
            let bishop_ok = match self.kind(i) {
                5 => (sq[i] % 8 + sq[i] / 8) % 2 == 0,
                6 => (sq[i] % 8 + sq[i] / 8) % 2 == 1,
                _ => true,
            };
            bishop_ok && (0..i).all(|j| sq[j] != sq[i])
        })
    }

    /// Whether a royal of the given side is attacked by a live enemy piece.
    fn royal_attacked(&self, sq: &[usize], alive: u32, white: bool, occ: u64) -> bool {
        let n = sq.len();
        let mut targets = 0u64;
        for i in (0..n).filter(|&i| alive >> i & 1 == 1 && self.white(i) == white && self.royal(i)) {
            targets |= 1 << sq[i];
        }
        (0..n).any(|i| {
            alive >> i & 1 == 1 && self.white(i) != white && self.geo.attacks(self.kind(i), sq[i], occ) & targets != 0
        })
    }

    fn spread(sq: &[usize]) -> bool {
        (0..sq.len()).all(|i| {
            (0..i).all(|j| {
                let (dx, dy) = ((sq[i] % 8) as i32 - (sq[j] % 8) as i32, (sq[i] / 8) as i32 - (sq[j] / 8) as i32);
                dx.abs().max(dy.abs()) >= SPREAD
            })
        })
    }

    fn solve(&self, threads: usize) -> Outcome {
        let n = self.syms.len();
        let size = 1usize << (6 * n);
        let won = Arc::new(bitset(size));
        let lost = bitset(size);
        let spread_won = AtomicBool::new(false);
        let mut rounds: usize = 0;
        loop {
            // Children are read as the last round left them, so round r finds the mates
            // in r - 1 plies and the round count gives the longest mate.
            let snapshot = |v: &[AtomicU64]| v.iter().map(|w| AtomicU64::new(w.load(Ordering::Relaxed))).collect::<Vec<_>>();
            let (won_before, lost_before) = (snapshot(&won), snapshot(&lost));
            let changed = AtomicUsize::new(0);
            let chunk = size.div_ceil(threads);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (won, lost, changed, spread_won) = (&won, &lost, &changed, &spread_won);
                    let (won_before, lost_before) = (&won_before, &lost_before);
                    s.spawn(move || {
                        let mut sq = [0usize; BOUNDED_CAP];
                        for idx in t * chunk..((t + 1) * chunk).min(size) {
                            for (i, s) in sq[..n].iter_mut().enumerate() {
                                *s = idx >> (6 * i) & 63;
                            }
                            let sq = &sq[..n];
                            if !self.placeable(sq) {
                                continue;
                            }
                            let occ = sq.iter().fold(0u64, |o, &s| o | 1 << s);
                            let all = (1u32 << n) - 1;
                            if !get(lost, idx) && !self.royal_attacked(sq, all, true, occ) && self.black_lost(sq, idx, occ, won_before)
                            {
                                set(lost, idx);
                                changed.fetch_add(1, Ordering::Relaxed);
                                if Self::spread(sq) {
                                    spread_won.store(true, Ordering::Relaxed);
                                }
                            }
                            if !get(won, idx) && !self.royal_attacked(sq, all, false, occ) && self.white_wins(sq, idx, occ, lost_before)
                            {
                                set(won, idx);
                                changed.fetch_add(1, Ordering::Relaxed);
                                if Self::spread(sq) {
                                    spread_won.store(true, Ordering::Relaxed);
                                }
                            }
                        }
                    });
                }
            });
            rounds += 1;
            if spread_won.load(Ordering::Relaxed) || changed.load(Ordering::Relaxed) == 0 {
                break;
            }
        }
        let forced_early = spread_won.load(Ordering::Relaxed);
        let (mut won_count, mut legal_count) = (0, 0);
        if !forced_early {
            let mut sq = [0usize; BOUNDED_CAP];
            for idx in 0..size {
                for (i, s) in sq[..n].iter_mut().enumerate() {
                    *s = idx >> (6 * i) & 63;
                }
                let sq = &sq[..n];
                if !self.placeable(sq) {
                    continue;
                }
                let occ = sq.iter().fold(0u64, |o, &s| o | 1 << s);
                if self.royal_attacked(sq, (1 << n) - 1, false, occ) {
                    continue;
                }
                legal_count += 1;
                won_count += get(&won, idx) as usize;
            }
        }
        // The last round found nothing new, and mate itself is found in round 1.
        Outcome { forced: forced_early, won, won_count, legal_count, longest: rounds.saturating_sub(2) }
    }

    /// Black to move is mated, or every legal move reaches a White win.
    fn black_lost(&self, sq: &[usize], idx: usize, occ: u64, won: &[AtomicU64]) -> bool {
        let n = sq.len();
        let all = (1u32 << n) - 1;
        let black_occ = (0..n).filter(|&i| !self.white(i)).fold(0u64, |o, i| o | 1 << sq[i]);
        let mut any_move = false;
        let mut next = [0usize; BOUNDED_CAP];
        for i in (0..n).filter(|&i| !self.white(i)) {
            let mut targets = self.geo.attacks(self.kind(i), sq[i], occ) & !black_occ;
            while targets != 0 {
                let t = targets.trailing_zeros() as usize;
                targets &= targets - 1;
                let captured = (0..n).find(|&j| sq[j] == t);
                if captured.is_some_and(|j| self.royal(j)) {
                    continue;
                }
                next[..n].copy_from_slice(sq);
                next[i] = t;
                let alive = captured.map_or(all, |j| all & !(1 << j));
                let new_occ = occ & !(1 << sq[i]) | 1 << t;
                if self.royal_attacked(&next[..n], alive, false, new_occ) {
                    continue;
                }
                any_move = true;
                let reaches_win = match captured {
                    None => get(won, idx - (sq[i] << (6 * i)) + (t << (6 * i))),
                    Some(j) => self.subs[j].as_ref().is_some_and(|sub| {
                        let sub_idx = (0..n)
                            .filter(|&p| p != j)
                            .enumerate()
                            .fold(0, |acc, (k, p)| acc | next[p] << (6 * k));
                        get(sub, sub_idx)
                    }),
                };
                if !reaches_win {
                    return false;
                }
            }
        }
        any_move || self.royal_attacked(sq, all, false, occ)
    }

    /// White to move has a legal move to a position where Black is lost.
    fn white_wins(&self, sq: &[usize], idx: usize, occ: u64, lost: &[AtomicU64]) -> bool {
        let n = sq.len();
        let all = (1u32 << n) - 1;
        let mut next = [0usize; BOUNDED_CAP];
        for i in (0..n).filter(|&i| self.white(i)) {
            // Black has only royals, which cannot be captured.
            let mut targets = self.geo.attacks(self.kind(i), sq[i], occ) & !occ;
            while targets != 0 {
                let t = targets.trailing_zeros() as usize;
                targets &= targets - 1;
                let to = idx - (sq[i] << (6 * i)) + (t << (6 * i));
                if !get(lost, to) {
                    continue;
                }
                next[..n].copy_from_slice(sq);
                next[i] = t;
                if !self.royal_attacked(&next[..n], all, true, occ & !(1 << sq[i]) | 1 << t) {
                    return true;
                }
            }
        }
        false
    }
}

fn label(set: &[u8]) -> String {
    let side = |white: bool| {
        set.iter()
            .filter(|&&s| (s < KINDS) == white)
            .map(|&s| {
                let code = KIND_CODES[(s % KINDS) as usize];
                if white { code.to_string() } else { code.to_lowercase() }
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    format!("{} vs {}", side(true), side(false))
}

#[derive(Clone)]
enum Verdict {
    /// Never mates: the bounded table calls it dead.
    Dead,
    Helpmate(Won),
    /// Helpmate-only, copied from a previous report without its table.
    Copied,
    Forced,
    Unknown,
}

/// White's armies one piece smaller that are cores themselves.
fn sub_cores<'a>(set: &'a [u8], cores: &'a HashSet<&[u8]>) -> impl Iterator<Item = Vec<u8>> + 'a {
    (0..set.len()).filter(|&j| set[j] < KINDS).filter_map(move |j| {
        let mut sub = set.to_vec();
        sub.remove(j);
        cores.contains(&sub[..]).then_some(sub)
    })
}

/// The cores to solve: all of them without a previous report; otherwise the new
/// ones and those holding one, plus every helpmate-only army their captures reach.
fn needed_cores(cores: &[Vec<u8>], previous: &HashMap<String, String>) -> HashSet<Vec<u8>> {
    if previous.is_empty() {
        return cores.iter().cloned().collect();
    }
    let all: HashSet<&[u8]> = cores.iter().map(|c| &c[..]).collect();
    let mut changed: HashSet<Vec<u8>> = HashSet::new();
    // Smallest first, so a core's smaller armies are decided before it.
    for set in cores {
        if !previous.contains_key(&label(set)) || sub_cores(set, &all).any(|sub| changed.contains(&sub)) {
            changed.insert(set.clone());
        }
    }
    let mut needed = changed.clone();
    let mut todo: Vec<Vec<u8>> = changed.into_iter().collect();
    while let Some(set) = todo.pop() {
        for sub in sub_cores(&set, &all) {
            let helpmate = previous.get(&label(&sub)).is_some_and(|l| l.split('\t').nth(1) == Some("helpmate-only"));
            if helpmate && needed.insert(sub.clone()) {
                todo.push(sub);
            }
        }
    }
    needed
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("usage: bounded_helpmate <out dir> [threads]");
    let threads: usize = args.next().map_or(12, |t| t.parse().unwrap());
    let previous: HashMap<String, String> = args.next().map_or_else(HashMap::new, |path| {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        text.lines()
            .filter_map(|l| l.split_once('\t').map(|(name, _)| (name.to_string(), l.to_string())))
            .collect()
    });
    std::fs::create_dir_all(&out).unwrap();
    let mut report = std::fs::File::create(format!("{out}/report.tsv")).unwrap();
    let mut helpmates = Vec::new();
    let geo = Geometry::new();
    let mut verdicts: HashMap<Vec<u8>, Verdict> = HashMap::new();
    let mut cores = Vec::new();
    mating_sets::for_each_set(BOUNDED_CAP, |set| {
        let black_royals_only = set.iter().filter(|&&s| s >= KINDS).all(|&s| mating_sets::is_royal_symbol(s));
        let has = |white: bool| set.iter().any(|&s| (s < KINDS) == white);
        if black_royals_only && has(true) && has(false) && !mating_sets::is_dead_bounded(set) {
            let mut v = set.to_vec();
            v.sort_unstable();
            cores.push(v);
        }
    });
    let needed = needed_cores(&cores, &previous);
    println!("{} bounded cores, {} to solve, {threads} threads", cores.len(), needed.len());
    let started = std::time::Instant::now();
    for (done, set) in cores.iter().enumerate() {
        let name = label(set);
        if !needed.contains(set) {
            let line = &previous[&name];
            writeln!(report, "{line}").unwrap();
            let verdict = match line.split('\t').nth(1) {
                Some("helpmate-only") => {
                    helpmates.push(name);
                    Verdict::Copied
                }
                Some("unknown") => Verdict::Unknown,
                _ => Verdict::Forced,
            };
            verdicts.insert(set.clone(), verdict);
            continue;
        }
        // Black capturing a White piece leaves the army without it.
        let mut subs = Vec::new();
        let mut inherited = None;
        for j in 0..set.len() {
            if set[j] >= KINDS {
                subs.push(None);
                continue;
            }
            let mut sub = set.clone();
            sub.remove(j);
            let verdict = if !sub.iter().any(|&s| s < KINDS) || mating_sets::is_dead_bounded(&sub) {
                Verdict::Dead
            } else {
                verdicts.get(&sub).cloned().unwrap_or(Verdict::Unknown)
            };
            match verdict {
                Verdict::Dead => subs.push(None),
                Verdict::Helpmate(won) => subs.push(Some(won)),
                Verdict::Copied => unreachable!("a needed army's helpmate-only parts are solved"),
                Verdict::Forced => inherited = Some(Verdict::Forced),
                Verdict::Unknown => inherited = inherited.or(Some(Verdict::Unknown)),
            }
        }
        if set.iter().any(|&s| s % KINDS == P) {
            inherited = Some(Verdict::Unknown);
        }
        let verdict = if let Some(v) = inherited {
            let tag = if matches!(v, Verdict::Forced) { "forced (smaller army)" } else { "unknown" };
            writeln!(report, "{name}\t{tag}").unwrap();
            v
        } else {
            let army = Army { geo: &geo, syms: set.clone(), subs };
            let o = army.solve(threads);
            let tag = if o.forced { "forced" } else { "helpmate-only" };
            writeln!(
                report,
                "{name}\t{tag}\t{} of {} won\tlongest mate {} plies",
                o.won_count, o.legal_count, o.longest
            )
            .unwrap();
            println!("{name}: {tag} ({} of {} won, longest mate {} plies, {:.0}s)", o.won_count, o.legal_count, o.longest, started.elapsed().as_secs_f64());
            if o.forced {
                Verdict::Forced
            } else {
                helpmates.push(name.clone());
                Verdict::Helpmate(o.won)
            }
        };
        report.flush().unwrap();
        verdicts.insert(set.clone(), verdict);
        if done % 100 == 99 {
            println!("{}/{} cores, {:.0}s", done + 1, cores.len(), started.elapsed().as_secs_f64());
        }
    }
    std::fs::write(format!("{out}/helpmate_only.txt"), helpmates.join("\n") + "\n").unwrap();
    println!("done in {:.0}s: {} helpmate-only", started.elapsed().as_secs_f64(), helpmates.len());
}
