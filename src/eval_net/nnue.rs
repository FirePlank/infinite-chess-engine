//! Board NNUE blended with the generic evaluator. Every coloured piece is a sparse feature
//! (own/enemy x type x its offset from a view's first royal, log-bucketed per axis), summed
//! into one accumulator per view with shared weights; a head reads both views' activations
//! after the Stage-A inputs and gives the side-to-move score. Integer and exact, so
//! `evalnet/train_nnue.py` reproduces every output.
//!
//! Accumulators are kept per thread on a stack indexed by the game's ply and checked by
//! hash: a position applies the board changes `make_move` logged since its nearest cached
//! ancestor (a view whose royal anchor moved is rebuilt instead). Integer sums are exact in
//! any order, so this always equals a rebuild.

use super::features::{FeatureCollector, TAC_NET_INPUTS, tac_net_schema_hash};
use super::weights::{EvalNetWeights, pad32};
use crate::board::{Piece, PlayerColor};
use crate::game::GameState;
use once_cell::sync::Lazy;

/// Piece types King (2) through Pawn (21).
pub const NT: usize = 20;
pub const AXIS: usize = 15;
pub const BUCKETS: usize = AXIS * AXIS;
pub const NUM_SPARSE: usize = 2 * NT * BUCKETS;
pub const ACC: usize = 128;
/// Accumulator units per 1.0 of activation: the head reads them as its integer inputs,
/// whose float value is x / 64 like every Stage-A input.
pub const QA: i16 = 64;
pub const DENSE: usize = TAC_NET_INPUTS;
pub const HEAD_IN: usize = DENSE + 2 * ACC;
/// Share of the NNUE in the blended score, in percent.
pub const BLEND: i32 = 50;

const MAGIC: &[u8; 8] = b"AINNUE01";
// Data run: both sides are this engine.

/// Exact offsets within 3 squares, then sign x (4-7, 8-15, 16-63, 64+).
#[inline]
pub fn axis_bucket(d: i64) -> usize {
    if (-3..=3).contains(&d) {
        return (d + 3) as usize;
    }
    let a = d.unsigned_abs();
    let band = if a < 8 {
        0
    } else if a < 16 {
        1
    } else if a < 64 {
        2
    } else {
        3
    };
    7 + 2 * band + usize::from(d > 0)
}

/// `axis_bucket` for offsets in -64..64.
static BUCKET_LUT: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        let d = i as i64 - 64;
        let a = d.unsigned_abs();
        t[i] = if d >= -3 && d <= 3 {
            (d + 3) as u8
        } else {
            let band = if a < 8 {
                0
            } else if a < 16 {
                1
            } else if a < 64 {
                2
            } else {
                3
            };
            7 + 2 * band + (d > 0) as u8
        };
        i += 1;
    }
    t
};

#[inline(always)]
fn bucket_fast(d: i64) -> usize {
    if (-64..64).contains(&d) {
        BUCKET_LUT[(d + 64) as usize] as usize
    } else {
        13 + usize::from(d > 0)
    }
}

/// An all-zero row after the real ones, for the piece that is a view's own anchor.
const ZERO_ROW: u16 = NUM_SPARSE as u16;

/// The feature of piece `p` on (x, y) in one view anchored on that view's royal (ranks
/// flipped for Black); None for pieces the net does not read.
#[inline(always)]
fn feature(white_view: bool, (ax, ay): (i64, i64), x: i64, y: i64, p: Piece) -> Option<u16> {
    let t = p.piece_type() as usize;
    if t.wrapping_sub(2) >= NT || p.color() == PlayerColor::Neutral {
        return None;
    }
    if x == ax && y == ay {
        return Some(ZERO_ROW);
    }
    let own = (p.color() == PlayerColor::White) == white_view;
    let dy = if white_view {
        y.saturating_sub(ay)
    } else {
        ay.saturating_sub(y)
    };
    Some(
        ((usize::from(!own) * NT + t - 2) * BUCKETS
            + bucket_fast(x.saturating_sub(ax)) * AXIS
            + bucket_fast(dy)) as u16,
    )
}

/// Calls `f` with every sparse feature of `view`; false when `view` has no royal.
pub fn for_each_feature(game: &GameState, view: PlayerColor, mut f: impl FnMut(usize)) -> bool {
    let royals = if view == PlayerColor::White {
        &game.white_royals
    } else {
        &game.black_royals
    };
    let Some(k) = royals.first() else {
        return false;
    };
    for (x, y, p) in game.board.iter_colored() {
        if let Some(i) = feature(view == PlayerColor::White, (k.x, k.y), x, y, p)
            && i != ZERO_ROW
        {
            f(i as usize);
        }
    }
    true
}

/// The ACC i16 lanes stay in registers (ACC / 16 of them) across every row added or
/// subtracted.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rows_avx2<const ADD: bool>(w: &[i16], acc: &mut [i16; ACC], feats: &[u16]) {
    use std::arch::x86_64::*;
    unsafe {
        let p = acc.as_mut_ptr() as *mut __m256i;
        let mut a = [_mm256_setzero_si256(); ACC / 16];
        for (j, aj) in a.iter_mut().enumerate() {
            *aj = _mm256_loadu_si256(p.add(j));
        }
        for &f in feats {
            debug_assert!((f as usize + 1) * ACC <= w.len());
            let r = w.as_ptr().add(f as usize * ACC) as *const __m256i;
            for (j, aj) in a.iter_mut().enumerate() {
                let row = _mm256_loadu_si256(r.add(j));
                *aj = if ADD {
                    _mm256_add_epi16(*aj, row)
                } else {
                    _mm256_sub_epi16(*aj, row)
                };
            }
        }
        for (j, aj) in a.iter().enumerate() {
            _mm256_storeu_si256(p.add(j), *aj);
        }
    }
}

/// Features gathered before each batch of row additions.
const CHUNK: usize = 64;
/// Plies a cached ancestor may lie back before the accumulators are rebuilt instead.
const MAX_BACK: usize = 8;

/// One ply of a thread's accumulator stack: raw sums of White's and Black's views.
#[derive(Clone, Copy)]
struct Entry {
    valid: bool,
    hash: u64,
    anchors: [(i64, i64); 2],
    acc: [[i16; ACC]; 2],
}

impl Entry {
    const EMPTY: Entry = Entry {
        valid: false,
        hash: 0,
        anchors: [(0, 0); 2],
        acc: [[0; ACC]; 2],
    };
}

thread_local! {
    static STACK: std::cell::RefCell<Vec<Entry>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub struct Nnue {
    /// `NUM_SPARSE` rows of `ACC` weights, then the zero row.
    ft_w: Box<[i16]>,
    ft_b: [i16; ACC],
    head: EvalNetWeights,
}

/// Schema of the head's inputs: the Stage-A layout, then the two accumulators.
pub fn schema_hash() -> u64 {
    tac_net_schema_hash() ^ 0x494e_4e55_4501_0000 ^ ((NUM_SPARSE as u64) << 20) ^ ((ACC as u64) << 8)
}

impl Nnue {
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        let rd = |at: usize| -> Result<u32, &'static str> {
            data.get(at..at + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or("short read")
        };
        if data.get(..8) != Some(&MAGIC[..]) {
            return Err("bad magic");
        }
        if rd(8)? as usize != NUM_SPARSE || rd(12)? as usize != ACC || rd(16)? as usize != DENSE {
            return Err("layout mismatch");
        }
        let ft = 20;
        let n = NUM_SPARSE * ACC;
        let i16s = |at: usize, len: usize| -> Result<Vec<i16>, &'static str> {
            let b = data.get(at..at + 2 * len).ok_or("short read (i16[])")?;
            Ok(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
        };
        let mut ft_w = i16s(ft, n)?;
        ft_w.resize(n + ACC, 0);
        let mut ft_b = [0i16; ACC];
        ft_b.copy_from_slice(&i16s(ft + 2 * n, ACC)?);
        let head = EvalNetWeights::from_bytes(&data[ft + 2 * (n + ACC)..], HEAD_IN, schema_hash())?;
        if !head.perspective {
            return Err("head must be a perspective net");
        }
        Ok(Nnue {
            ft_w: ft_w.into_boxed_slice(),
            ft_b,
            head,
        })
    }

    /// Adds (ADD) or subtracts the weight rows of `feats` into `acc`, wrapping like the
    /// trainer's i16 lanes.
    #[inline]
    fn rows<const ADD: bool>(&self, acc: &mut [i16; ACC], feats: &[u16]) {
        #[cfg(target_arch = "x86_64")]
        if super::inference::has_avx2() {
            unsafe { rows_avx2::<ADD>(&self.ft_w, acc, feats) };
            return;
        }
        for &f in feats {
            let row = &self.ft_w[f as usize * ACC..(f as usize + 1) * ACC];
            for (a, &w) in acc.iter_mut().zip(row) {
                *a = if ADD {
                    a.wrapping_add(w)
                } else {
                    a.wrapping_sub(w)
                };
            }
        }
    }

    /// Rebuilds the views not `done` from every piece, in one pass over the board.
    fn refresh(&self, game: &GameState, anchors: [(i64, i64); 2], done: [bool; 2], acc: &mut [[i16; ACC]; 2]) {
        let mut feats = [[0u16; CHUNK]; 2];
        let mut n = 0;
        for v in 0..2 {
            if !done[v] {
                acc[v] = self.ft_b;
            }
        }
        for (x, y, p) in game.board.iter_colored() {
            let Some(fw) = feature(true, anchors[0], x, y, p) else {
                continue;
            };
            feats[0][n] = fw;
            feats[1][n] = feature(false, anchors[1], x, y, p).unwrap_or(ZERO_ROW);
            n += 1;
            if n == CHUNK {
                for v in 0..2 {
                    if !done[v] {
                        self.rows::<true>(&mut acc[v], &feats[v]);
                    }
                }
                n = 0;
            }
        }
        for v in 0..2 {
            if !done[v] {
                self.rows::<true>(&mut acc[v], &feats[v][..n]);
            }
        }
    }

    /// Raw accumulators (White's view, Black's view) of `game`.
    fn accumulators(&self, game: &GameState, anchors: [(i64, i64); 2]) -> [[i16; ACC]; 2] {
        let ply = game.nnue_log.len();
        let hash = game.hash;
        STACK.with(|st| {
            let mut st = st.borrow_mut();
            if st.len() <= ply {
                st.resize(ply + 1, Entry::EMPTY);
            }
            let e = &st[ply];
            if e.valid && e.hash == hash && e.anchors == anchors {
                return e.acc;
            }
            let mut acc = [[0i16; ACC]; 2];
            let mut done = [false; 2];
            // The log describes this position only if its last entry ends on it.
            if ply > 0 && game.nnue_log[ply - 1].after == hash {
                let mut k = ply;
                while k > ply.saturating_sub(MAX_BACK) {
                    k -= 1;
                    let ek = &st[k];
                    if !(ek.valid && ek.hash == game.nnue_log[k].before) {
                        continue;
                    }
                    for v in 0..2 {
                        if ek.anchors[v] != anchors[v] {
                            continue;
                        }
                        acc[v] = ek.acc[v];
                        done[v] = true;
                        let mut ops = [[0u16; 6 * MAX_BACK]; 2];
                        let mut n = [0usize; 2];
                        for d in &game.nnue_log[k..ply] {
                            for &(x, y, p, added) in &d.ops[..d.n as usize] {
                                if let Some(f) = feature(v == 0, anchors[v], x, y, Piece::from_packed(p)) {
                                    let s = usize::from(!added);
                                    ops[s][n[s]] = f;
                                    n[s] += 1;
                                }
                            }
                        }
                        self.rows::<true>(&mut acc[v], &ops[0][..n[0]]);
                        self.rows::<false>(&mut acc[v], &ops[1][..n[1]]);
                    }
                    break;
                }
            }
            if !(done[0] && done[1]) {
                self.refresh(game, anchors, done, &mut acc);
            }
            st[ply] = Entry {
                valid: true,
                hash,
                anchors,
                acc,
            };
            acc
        })
    }

    /// Side-to-move score in centipawns, or None when a side has no royal.
    pub fn evaluate(&self, game: &GameState, fc: &FeatureCollector) -> Option<i32> {
        self.evaluate_dense(game, &super::net_inputs(game, fc, DENSE, true))
    }

    /// `evaluate` with the Stage-A perspective inputs already built.
    pub fn evaluate_dense(&self, game: &GameState, dense: &[i16]) -> Option<i32> {
        let (Some(wk), Some(bk)) = (game.white_royals.first(), game.black_royals.first()) else {
            return None;
        };
        let acc = self.accumulators(game, [(wk.x, wk.y), (bk.x, bk.y)]);
        let mut x = [0i16; pad32(HEAD_IN)];
        x[..DENSE].copy_from_slice(&dense[..DENSE]);
        // Side to move's view first.
        let us = usize::from(game.turn != PlayerColor::White);
        for (o, &a) in x[DENSE..DENSE + ACC].iter_mut().zip(&acc[us]) {
            *o = a.clamp(0, QA);
        }
        for (o, &a) in x[DENSE + ACC..HEAD_IN].iter_mut().zip(&acc[1 - us]) {
            *o = a.clamp(0, QA);
        }
        Some(super::inference::forward(&self.head, &x[..HEAD_IN]))
    }

    /// The side-to-move score `hce` blended with this net's, or `hce` when it has none.
    #[inline]
    pub fn blend(&self, game: &GameState, dense: &[i16], hce: i32) -> i32 {
        match self.evaluate_dense(game, dense) {
            Some(nn) => (hce * (100 - BLEND) + nn * BLEND) / 100,
            None => hce,
        }
    }
}

/// Trained weights; an empty file means no NNUE.
static NNUE_BYTES: &[u8] = include_bytes!("nnue.bin");

pub static NNUE: Lazy<Option<Nnue>> = Lazy::new(|| {
    if NNUE_BYTES.is_empty() || super::killed() {
        return None;
    }
    match Nnue::from_bytes(NNUE_BYTES) {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("nnue.bin rejected: {e}");
            None
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_the_axis() {
        let got: Vec<usize> = [-1000, -20, -9, -5, -3, 0, 3, 5, 9, 20, 1000]
            .iter()
            .map(|&d| axis_bucket(d))
            .collect();
        assert_eq!(got, vec![13, 11, 9, 7, 0, 3, 6, 8, 10, 12, 14]);
        assert!((-5000..5000).all(|d| axis_bucket(d) == bucket_fast(d)));
    }

    /// A net with random feature weights and a zero head, enough to compare accumulators.
    fn random_net() -> Nnue {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 41) as i16 - 20
        };
        let mut ft_w: Vec<i16> = (0..NUM_SPARSE * ACC).map(|_| next()).collect();
        ft_w.resize((NUM_SPARSE + 1) * ACC, 0);
        let mut blob = Vec::new();
        blob.extend_from_slice(b"AEVNET01");
        for v in [2u32, HEAD_IN as u32, 16, 16, 6, 6] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&schema_hash().to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.resize(blob.len() + 16 * HEAD_IN + 16 * 4 + 16 * 16 + 16 * 4 + 16 + 4, 0);
        Nnue {
            ft_w: ft_w.into_boxed_slice(),
            ft_b: [3; ACC],
            head: EvalNetWeights::from_bytes(&blob, HEAD_IN, schema_hash()).unwrap(),
        }
    }

    /// Plays pseudo-random legal lines with make/undo (null moves included) in several
    /// variants and checks the incremental accumulators against a rebuild at every node.
    #[test]
    fn incremental_equals_rebuild() {
        use crate::Variant;
        let net = random_net();
        let mut seed = 12345u64;
        let mut rnd = move |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as usize) % n.max(1)
        };
        let mut checked = 0;
        for variant in [Variant::Classical, Variant::CoaIP, Variant::Palace, Variant::Knightline, Variant::Space] {
            let mut g = GameState::default();
            g.setup_variant(variant);
            let mut path = Vec::new();
            for _ in 0..600 {
                let moves = g.get_pseudo_legal_moves();
                if (moves.is_empty() || rnd(4) == 0) && !path.is_empty() {
                    match path.pop().unwrap() {
                        Some((m, u)) => g.undo_move(&m, u),
                        None => g.unmake_null_move(),
                    }
                } else if rnd(10) == 0 {
                    g.make_null_move();
                    path.push(None);
                } else if !moves.is_empty() {
                    let m = moves[rnd(moves.len())];
                    let u = g.make_move(&m);
                    if g.is_move_illegal() {
                        g.undo_move(&m, u);
                    } else {
                        path.push(Some((m, u)));
                    }
                }
                let (Some(wk), Some(bk)) = (g.white_royals.first(), g.black_royals.first()) else {
                    continue;
                };
                let anchors = [(wk.x, wk.y), (bk.x, bk.y)];
                let inc = net.accumulators(&g, anchors);
                let mut full = [[0i16; ACC]; 2];
                net.refresh(&g, anchors, [false; 2], &mut full);
                assert_eq!(inc, full, "{variant:?} after {} plies", path.len());
                checked += 1;
            }
        }
        assert!(checked > 2000);
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    #[test]
    #[ignore]
    fn component_times() {
        let mut g = GameState::default();
        g.setup_variant(crate::Variant::Classical);
        let n = NNUE.as_ref().expect("embed a net to time it");
        let mut fc = FeatureCollector::default();
        crate::evaluation::base::evaluate_inner_traced(&g, &mut fc);
        let dense = super::super::net_inputs(&g, &fc, DENSE, true);
        let it = 1_000_000;
        let anchors = [
            (g.white_royals[0].x, g.white_royals[0].y),
            (g.black_royals[0].x, g.black_royals[0].y),
        ];
        let t = std::time::Instant::now();
        for _ in 0..it {
            let mut acc = [[0i16; ACC]; 2];
            n.refresh(std::hint::black_box(&g), anchors, [false; 2], &mut acc);
            std::hint::black_box(acc);
        }
        println!("rebuild both views {:?}/eval", t.elapsed() / it);
        let mut x = [0i16; pad32(HEAD_IN)];
        x[..DENSE].copy_from_slice(&dense);
        let t = std::time::Instant::now();
        for _ in 0..it {
            std::hint::black_box(super::super::inference::forward(&n.head, std::hint::black_box(&x[..HEAD_IN])));
        }
        println!("head {:?}/eval", t.elapsed() / it);
        let t = std::time::Instant::now();
        for _ in 0..it {
            std::hint::black_box(super::super::net_inputs(std::hint::black_box(&g), &fc, DENSE, true));
        }
        println!("net_inputs {:?}/eval", t.elapsed() / it);
        let t = std::time::Instant::now();
        for _ in 0..it {
            std::hint::black_box(n.evaluate_dense(std::hint::black_box(&g), &dense));
        }
        println!("evaluate_dense, cached accumulators {:?}/eval", t.elapsed() / it);
    }
}
