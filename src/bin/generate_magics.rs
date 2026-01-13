//! High-performance Magic Number Generator (Edge-Inclusive 8x8 chunk)
//!
//! Run:
//!   cargo run --release --bin generate_magics
//! Optional args:
//!   --threads N
//!   --refresh-ms 100
//!   --no-improve
//!
//! Notes for infinite board:
//! These magics are for an 8x8 CHUNK bitboard. For rays that pass the chunk edge
//! without a blocker, you continue into neighbor chunks with the same logic.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ------------------------------ CLI ------------------------------

#[derive(Clone, Copy)]
struct Args {
    threads: usize,
    refresh_ms: u64,
    improve: bool,
}

impl Args {
    fn parse() -> Self {
        let mut threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut refresh_ms = 100u64;
        let mut improve = true;

        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--threads" => {
                    if let Some(v) = it.next() {
                        threads = v.parse().unwrap_or(threads).max(1);
                    }
                }
                "--refresh-ms" => {
                    if let Some(v) = it.next() {
                        refresh_ms = v.parse().unwrap_or(refresh_ms).max(10);
                    }
                }
                "--no-improve" => improve = false,
                _ => {}
            }
        }

        Self {
            threads,
            refresh_ms,
            improve,
        }
    }
}

// ------------------------------ Shared state ------------------------------

struct Shared {
    running: AtomicBool,
    phase: AtomicU8,

    rook_magics: [AtomicU64; 64],
    rook_shifts: [AtomicU8; 64],
    rook_found: [AtomicBool; 64],

    bishop_magics: [AtomicU64; 64],
    bishop_shifts: [AtomicU8; 64],
    bishop_found: [AtomicBool; 64],

    improvements: AtomicU64,
}

impl Shared {
    fn new() -> Self {
        Self {
            running: AtomicBool::new(true),
            phase: AtomicU8::new(0),

            rook_magics: std::array::from_fn(|_| AtomicU64::new(0)),
            rook_shifts: std::array::from_fn(|_| AtomicU8::new(64)),
            rook_found: std::array::from_fn(|_| AtomicBool::new(false)),

            bishop_magics: std::array::from_fn(|_| AtomicU64::new(0)),
            bishop_shifts: std::array::from_fn(|_| AtomicU8::new(64)),
            bishop_found: std::array::from_fn(|_| AtomicBool::new(false)),

            improvements: AtomicU64::new(0),
        }
    }
}

// ------------------------------ Square data ------------------------------

#[derive(Clone)]
struct SquareData {
    // sq: usize,
    mask: u64,
    bits: u8,
    min_shift: u8,
    max_shift_possible: u8,
    occupancies: Vec<u64>,
    attacks_list: Vec<u64>,
}

fn compute_square_data(sq: usize, is_rook: bool) -> SquareData {
    let mask = if is_rook {
        gen_rook_mask_edge_inclusive(sq)
    } else {
        gen_bishop_mask_edge_inclusive(sq)
    };
    let bits = mask.count_ones() as u8;

    // enumerate all subsets via carry-rippler
    let n = 1usize << bits;
    let mut occupancies = vec![0u64; n];
    let mut attacks_list = vec![0u64; n];

    let mut occ = 0u64;
    for i in 0..n {
        occupancies[i] = occ;
        attacks_list[i] = if is_rook {
            gen_rook_attacks_edge_inclusive(sq, occ)
        } else {
            gen_bishop_attacks_edge_inclusive(sq, occ)
        };
        occ = occ.wrapping_sub(mask) & mask;
    }

    // Theoretical bound for max shift: table_size must be >= unique_attacks.
    let mut uniq = attacks_list.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let min_index_bits = ceil_log2_usize(uniq.len().max(1));

    // shift = 64 - index_bits
    // clamp to 63 to avoid shifting by 64 in (x >> shift)
    let mut max_shift_possible = 64u8.saturating_sub(min_index_bits);
    if max_shift_possible > 63 {
        max_shift_possible = 63;
    }

    let mut min_shift = 64u8.saturating_sub(bits);
    if min_shift > 63 {
        min_shift = 63;
    }

    // max_shift_possible should never be below min_shift, but clamp just in case
    if max_shift_possible < min_shift {
        max_shift_possible = min_shift;
    }

    SquareData {
        // sq,
        mask,
        bits,
        min_shift,
        max_shift_possible,
        occupancies,
        attacks_list,
    }
}

#[inline]
fn ceil_log2_usize(x: usize) -> u8 {
    // ceil(log2(x)), x>=1
    if x <= 1 {
        return 0;
    }
    let v = x - 1;
    (usize::BITS - v.leading_zeros()) as u8
}

// ------------------------------ Scratch (epoch-stamped) ------------------------------

struct Scratch {
    stamp: Vec<u32>,
    val: Vec<u64>,
    epoch: u32,
}

impl Scratch {
    fn new() -> Self {
        Self {
            stamp: Vec::new(),
            val: Vec::new(),
            epoch: 1,
        }
    }

    #[inline(always)]
    fn ensure(&mut self, size: usize) {
        if self.stamp.len() < size {
            self.stamp.resize(size, 0);
            self.val.resize(size, 0);
            self.epoch = 1;
        }
    }

    #[inline(always)]
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // overflow: reset stamps
            self.stamp.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }
}

// ------------------------------ RNG ------------------------------

#[inline]
fn mix64(mut x: u64) -> u64 {
    // splitmix64
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        self.state
    }

    #[inline(always)]
    fn sparse_rand(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

// ------------------------------ Magic search core ------------------------------

fn find_magic_for_shift_rng(
    data: &SquareData,
    shift: u8,
    max_attempts: u64,
    rng: &mut SimpleRng,
    scratch: &mut Scratch,
    running: &AtomicBool,
) -> Option<u64> {
    // shift in [0..=63]
    let index_bits = (64 - shift) as usize;
    let table_size = 1usize << index_bits;

    scratch.ensure(table_size);

    let occs = &data.occupancies;
    let atks = &data.attacks_list;

    for _ in 0..max_attempts {
        if !running.load(Ordering::Relaxed) {
            return None;
        }

        let magic = rng.sparse_rand();

        // quick reject (entropy-ish heuristic)
        if ((data.mask.wrapping_mul(magic)) >> 56).count_ones() < 6 {
            continue;
        }

        let epoch = scratch.next_epoch();
        let mut ok = true;

        for i in 0..occs.len() {
            let occ = unsafe { *occs.get_unchecked(i) };
            let atk = unsafe { *atks.get_unchecked(i) };
            let idx = ((occ.wrapping_mul(magic)) >> shift) as usize;

            unsafe {
                let s = scratch.stamp.get_unchecked_mut(idx);
                if *s != epoch {
                    *s = epoch;
                    *scratch.val.get_unchecked_mut(idx) = atk;
                } else if *scratch.val.get_unchecked(idx) != atk {
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            return Some(magic);
        }
    }

    None
}

fn find_best_magic_for_square(
    sd: &SquareData,
    rng: &mut SimpleRng,
    scratch: &mut Scratch,
    running: &AtomicBool,
) -> Option<(u8, u64)> {
    // Try tightest first (largest shift = smallest table).
    // If fails, relax (smaller shift) until success.
    for shift in (sd.min_shift..=sd.max_shift_possible).rev() {
        if !running.load(Ordering::Relaxed) {
            return None;
        }

        // more attempts for tighter tables
        let delta = (shift - sd.min_shift) as u64;
        let base = 120_000u64;
        let scale = (1u64 << (sd.bits.min(14) as u64)).saturating_mul(4);
        let attempts = base + scale + delta.saturating_mul(220_000);

        if let Some(magic) = find_magic_for_shift_rng(sd, shift, attempts, rng, scratch, running) {
            return Some((shift, magic));
        }
    }

    // Fallback: brute at min_shift with rising budgets
    let mut attempts = 500_000u64;
    while running.load(Ordering::Relaxed) {
        if let Some(magic) = find_magic_for_shift_rng(sd, sd.min_shift, attempts, rng, scratch, running) {
            return Some((sd.min_shift, magic));
        }
        attempts = attempts.saturating_mul(2).min(50_000_000);
    }

    None
}

// ------------------------------ Phase 1 & Phase 2 ------------------------------

#[derive(Clone, Copy)]
enum Kind {
    Rook,
    Bishop,
}

#[derive(Clone, Copy)]
struct Task {
    kind: Kind,
    sq: usize,
}

fn run_initial_search(shared: &Arc<Shared>, rook: &Arc<Vec<SquareData>>, bishop: &Arc<Vec<SquareData>>, threads: usize) {
    shared.phase.store(1, Ordering::Relaxed);

    let mut tasks = Vec::with_capacity(128);
    for sq in 0..64 {
        tasks.push(Task { kind: Kind::Rook, sq });
    }
    for sq in 0..64 {
        tasks.push(Task { kind: Kind::Bishop, sq });
    }
    let tasks = Arc::new(tasks);
    let next = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let sh = Arc::clone(shared);
        let rd = Arc::clone(rook);
        let bd = Arc::clone(bishop);
        let t = Arc::clone(&tasks);
        let n = Arc::clone(&next);

        handles.push(thread::spawn(move || {
            let mut rng = SimpleRng::new(mix64(0xA5A5_A5A5_1234_5678u64 ^ tid as u64));
            let mut scratch = Scratch::new();

            loop {
                if !sh.running.load(Ordering::Relaxed) {
                    break;
                }

                let i = n.fetch_add(1, Ordering::Relaxed);
                if i >= t.len() {
                    break;
                }

                let task = t[i];
                match task.kind {
                    Kind::Rook => {
                        let sd = &rd[task.sq];
                        if let Some((shift, magic)) =
                            find_best_magic_for_square(sd, &mut rng, &mut scratch, &sh.running)
                        {
                            sh.rook_magics[task.sq].store(magic, Ordering::Relaxed);
                            sh.rook_shifts[task.sq].store(shift, Ordering::Relaxed);
                            sh.rook_found[task.sq].store(true, Ordering::Relaxed);
                        }
                    }
                    Kind::Bishop => {
                        let sd = &bd[task.sq];
                        if let Some((shift, magic)) =
                            find_best_magic_for_square(sd, &mut rng, &mut scratch, &sh.running)
                        {
                            sh.bishop_magics[task.sq].store(magic, Ordering::Relaxed);
                            sh.bishop_shifts[task.sq].store(shift, Ordering::Relaxed);
                            sh.bishop_found[task.sq].store(true, Ordering::Relaxed);
                        }
                    }
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }
}

fn improve_worker(tid_seed: u64, shared: Arc<Shared>, rook: Arc<Vec<SquareData>>, bishop: Arc<Vec<SquareData>>) {
    let mut rng = SimpleRng::new(mix64(0xC3C3_C3C3_5A5A_5A5Au64 ^ tid_seed));
    let mut scratch = Scratch::new();

    // attempts per try (tune this)
    const ATTEMPTS: u64 = 250_000;

    while shared.running.load(Ordering::Relaxed) {
        let pick = rng.next();
        let is_rook = (pick & 1) == 0;
        let sq = ((pick >> 1) % 64) as usize;

        if is_rook {
            if !shared.rook_found[sq].load(Ordering::Relaxed) {
                continue;
            }
            let cur = shared.rook_shifts[sq].load(Ordering::Relaxed);
            let cap = rook[sq].max_shift_possible;
            if cur >= cap {
                continue;
            }
            let target = cur + 1;

            if let Some(magic) = find_magic_for_shift_rng(&rook[sq], target, ATTEMPTS, &mut rng, &mut scratch, &shared.running) {
                // only commit if nobody changed it since we read cur
                if shared.rook_shifts[sq]
                    .compare_exchange(cur, target, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    shared.rook_magics[sq].store(magic, Ordering::Release);
                    shared.improvements.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            if !shared.bishop_found[sq].load(Ordering::Relaxed) {
                continue;
            }
            let cur = shared.bishop_shifts[sq].load(Ordering::Relaxed);
            let cap = bishop[sq].max_shift_possible;
            if cur >= cap {
                continue;
            }
            let target = cur + 1;

            if let Some(magic) = find_magic_for_shift_rng(&bishop[sq], target, ATTEMPTS, &mut rng, &mut scratch, &shared.running) {
                if shared.bishop_shifts[sq]
                    .compare_exchange(cur, target, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    shared.bishop_magics[sq].store(magic, Ordering::Release);
                    shared.improvements.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

// ------------------------------ UI / stats ------------------------------

fn collect_stats(
    found: impl Fn(usize) -> bool,
    shift: impl Fn(usize) -> u8,
) -> (u32, f64, u8, u8, u64) {
    let mut cnt = 0u32;
    let mut total_entries = 0u64;
    let mut min_bits = 64u8;
    let mut max_bits = 0u8;

    for sq in 0..64 {
        if found(sq) {
            cnt += 1;
            let sh = shift(sq);
            let bits = 64 - sh;
            total_entries += 1u64 << bits;
            min_bits = min_bits.min(bits);
            max_bits = max_bits.max(bits);
        }
    }

    if cnt == 0 {
        min_bits = 0;
    }

    let kb = (total_entries as f64) * 8.0 / 1024.0;
    (cnt, kb, min_bits, max_bits, total_entries)
}

fn display_stats(shared: &Shared, rook: &[SquareData], bishop: &[SquareData], start: &Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let phase = shared.phase.load(Ordering::Relaxed);
    let imps = shared.improvements.load(Ordering::Relaxed);

    let (r_found, r_kb, r_min_bits, r_max_bits, _r_entries) = collect_stats(
        |sq| shared.rook_found[sq].load(Ordering::Relaxed),
        |sq| shared.rook_shifts[sq].load(Ordering::Relaxed),
    );

    let (b_found, b_kb, b_min_bits, b_max_bits, _b_entries) = collect_stats(
        |sq| shared.bishop_found[sq].load(Ordering::Relaxed),
        |sq| shared.bishop_shifts[sq].load(Ordering::Relaxed),
    );

    let mut r_cap_steps = 0u32;
    let mut b_cap_steps = 0u32;
    for sq in 0..64 {
        if shared.rook_found[sq].load(Ordering::Relaxed) {
            let cur = shared.rook_shifts[sq].load(Ordering::Relaxed);
            let cap = rook[sq].max_shift_possible;
            if cur < cap {
                r_cap_steps += (cap - cur) as u32;
            }
        }
        if shared.bishop_found[sq].load(Ordering::Relaxed) {
            let cur = shared.bishop_shifts[sq].load(Ordering::Relaxed);
            let cap = bishop[sq].max_shift_possible;
            if cur < cap {
                b_cap_steps += (cap - cur) as u32;
            }
        }
    }

    let total_kb = r_kb + b_kb;

    print!("\x1b[7;1H");
    print!("\x1b[0J");

    println!(
        "  Phase: {}    (1=initial, 2=improve)    Elapsed: {:.1}s    Improvements: {}",
        phase, elapsed, imps
    );

    println!("\n  \x1b[1;36mRook Magics (edge-inclusive chunk):\x1b[0m");
    println!("    Found: \x1b[32m{} / 64\x1b[0m", r_found);
    if r_found > 0 {
        println!("    Index bits range: \x1b[33m{}\x1b[0m - \x1b[33m{}\x1b[0m", r_min_bits, r_max_bits);
        println!("    Table size: \x1b[32m{:.2} KB\x1b[0m", r_kb);
        println!("    Remaining possible +shift steps (sum): {}", r_cap_steps);
    }

    println!("\n  \x1b[1;36mBishop Magics (edge-inclusive chunk):\x1b[0m");
    println!("    Found: \x1b[32m{} / 64\x1b[0m", b_found);
    if b_found > 0 {
        println!("    Index bits range: \x1b[33m{}\x1b[0m - \x1b[33m{}\x1b[0m", b_min_bits, b_max_bits);
        println!("    Table size: \x1b[32m{:.2} KB\x1b[0m", b_kb);
        println!("    Remaining possible +shift steps (sum): {}", b_cap_steps);
    }

    println!("\n  \x1b[1;37mCombined total: {:.2} KB\x1b[0m", total_kb);

    let _ = std::io::stdout().flush();
}

// ------------------------------ Output ------------------------------

fn total_entries(found: impl Fn(usize) -> bool, shift: impl Fn(usize) -> u8) -> u64 {
    let mut total = 0u64;
    for sq in 0..64 {
        if found(sq) {
            let bits = 64 - shift(sq);
            total += 1u64 << bits;
        }
    }
    total
}

fn output_rust_code(shared: &Shared) {
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("     READY-TO-PASTE RUST CODE");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    let rook_total = total_entries(
        |sq| shared.rook_found[sq].load(Ordering::Relaxed),
        |sq| shared.rook_shifts[sq].load(Ordering::Relaxed),
    );
    let bishop_total = total_entries(
        |sq| shared.bishop_found[sq].load(Ordering::Relaxed),
        |sq| shared.bishop_shifts[sq].load(Ordering::Relaxed),
    );

    println!("// Edge-inclusive chunk magics (8x8)");
    println!("// Rooks: {} KB ({} entries), Bishops: {} KB ({} entries)", rook_total * 8 / 1024, rook_total, bishop_total * 8 / 1024, bishop_total);
    println!();

    println!("pub const ROOK_MAGICS: [u64; 64] = [");
    for i in 0..64 {
        if i % 4 == 0 { print!("    "); }
        let magic = shared.rook_magics[i].load(Ordering::Relaxed);
        print!("0x{:016X},", magic);
        if (i + 1) % 4 == 0 { println!(); } else { print!(" "); }
    }
    println!("];\n");

    println!("pub const ROOK_SHIFTS: [u8; 64] = [");
    for i in 0..64 {
        if i % 8 == 0 { print!("    "); }
        let sh = shared.rook_shifts[i].load(Ordering::Relaxed);
        print!("{:2},", sh);
        if (i + 1) % 8 == 0 { println!(); } else { print!(" "); }
    }
    println!("];\n");

    println!("pub const BISHOP_MAGICS: [u64; 64] = [");
    for i in 0..64 {
        if i % 4 == 0 { print!("    "); }
        let magic = shared.bishop_magics[i].load(Ordering::Relaxed);
        print!("0x{:016X},", magic);
        if (i + 1) % 4 == 0 { println!(); } else { print!(" "); }
    }
    println!("];\n");

    println!("pub const BISHOP_SHIFTS: [u8; 64] = [");
    for i in 0..64 {
        if i % 8 == 0 { print!("    "); }
        let sh = shared.bishop_shifts[i].load(Ordering::Relaxed);
        print!("{:2},", sh);
        if (i + 1) % 8 == 0 { println!(); } else { print!(" "); }
    }
    println!("];");

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
}

// ------------------------------ Bitboard masks & attacks (edge-inclusive) ------------------------------

#[inline]
fn gen_rook_mask_edge_inclusive(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;

    for rr in 0..8 {
        if rr != r {
            m |= 1u64 << (rr * 8 + f);
        }
    }
    for ff in 0..8 {
        if ff != f {
            m |= 1u64 << (r * 8 + ff);
        }
    }
    m
}

#[inline]
fn gen_bishop_mask_edge_inclusive(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;

    for (dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let mut rr = r + dr;
        let mut ff = f + df;
        while rr >= 0 && rr < 8 && ff >= 0 && ff < 8 {
            m |= 1u64 << (rr * 8 + ff);
            rr += dr;
            ff += df;
        }
    }
    m
}

#[inline]
fn gen_rook_attacks_edge_inclusive(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;

    for rr in (r + 1)..8 {
        let s = (rr * 8 + f) as usize;
        a |= 1u64 << s;
        if (occ & (1u64 << s)) != 0 { break; }
    }
    for rr in (0..r).rev() {
        let s = (rr * 8 + f) as usize;
        a |= 1u64 << s;
        if (occ & (1u64 << s)) != 0 { break; }
    }
    for ff in (f + 1)..8 {
        let s = (r * 8 + ff) as usize;
        a |= 1u64 << s;
        if (occ & (1u64 << s)) != 0 { break; }
    }
    for ff in (0..f).rev() {
        let s = (r * 8 + ff) as usize;
        a |= 1u64 << s;
        if (occ & (1u64 << s)) != 0 { break; }
    }
    a
}

#[inline]
fn gen_bishop_attacks_edge_inclusive(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;

    for (dr, df) in &[(1i32, 1i32), (1, -1), (-1, 1), (-1, -1)] {
        let mut rr = r + dr;
        let mut ff = f + df;
        while rr >= 0 && rr < 8 && ff >= 0 && ff < 8 {
            let s = (rr * 8 + ff) as usize;
            a |= 1u64 << s;
            if (occ & (1u64 << s)) != 0 { break; }
            rr += dr;
            ff += df;
        }
    }
    a
}

// ------------------------------ main ------------------------------

fn main() {
    let args = Args::parse();

    let shared = Arc::new(Shared::new());
    // let shared_ctrlc = Arc::clone(&shared);

    // ctrlc::set_handler(move || {
    //     shared_ctrlc.running.store(false, Ordering::Relaxed);
    // })
    // .expect("Error setting Ctrl+C handler");

    print!("\x1b[2J\x1b[H");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("   HIGH-PERFORMANCE MAGIC GENERATOR (Edge-Inclusive, chunk-friendly)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Ctrl+C to stop and output code");
    println!(
        "  Threads: {}, Refresh: {} ms, Improve: {}",
        args.threads,
        args.refresh_ms,
        if args.improve { "ON" } else { "OFF" }
    );
    println!();

    let start = Instant::now();

    // Precompute per-square data
    let rook_data: Vec<SquareData> = (0..64).map(|sq| compute_square_data(sq, true)).collect();
    let bishop_data: Vec<SquareData> = (0..64).map(|sq| compute_square_data(sq, false)).collect();
    let rook_data = Arc::new(rook_data);
    let bishop_data = Arc::new(bishop_data);

    // Stats/UI thread
    let stats_shared = Arc::clone(&shared);
    let stats_rook = Arc::clone(&rook_data);
    let stats_bishop = Arc::clone(&bishop_data);
    let refresh = args.refresh_ms;

    let stats_thread = thread::spawn(move || {
        while stats_shared.running.load(Ordering::Relaxed) {
            display_stats(&stats_shared, &stats_rook, &stats_bishop, &start);
            thread::sleep(Duration::from_millis(refresh));
        }
        display_stats(&stats_shared, &stats_rook, &stats_bishop, &start);
    });

    // Phase 1
    run_initial_search(&shared, &rook_data, &bishop_data, args.threads);

    if !shared.running.load(Ordering::Relaxed) {
        let _ = stats_thread.join();
        println!("\nStopped during Phase 1 after {:.2}s", start.elapsed().as_secs_f64());
        output_rust_code(&shared);
        return;
    }

    // Phase 2
    if args.improve {
        shared.phase.store(2, Ordering::Relaxed);
        let mut workers = Vec::with_capacity(args.threads);
        for tid in 0..args.threads {
            let sh = Arc::clone(&shared);
            let rd = Arc::clone(&rook_data);
            let bd = Arc::clone(&bishop_data);
            workers.push(thread::spawn(move || improve_worker(tid as u64, sh, rd, bd)));
        }

        while shared.running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(50));
        }

        for w in workers {
            let _ = w.join();
        }
    }

    shared.running.store(false, Ordering::Relaxed);
    let _ = stats_thread.join();

    println!("\n\nStopped after {:.2}s", start.elapsed().as_secs_f64());
    output_rust_code(&shared);
}
