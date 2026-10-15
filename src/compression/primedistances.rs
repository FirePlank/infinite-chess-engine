//! Whether two squares on one rank or file lie a prime distance apart, which a Huygen reads at any
//! range and no compression keeps. Within a class of any line form every point shifts alike, so a
//! distance is taken on the original through a form both squares sit in a class's reach of. A
//! square in no class's reach (a far gap a piece slid into) has no original: no Huygen reaches it.

use super::bigint::Big;
use super::lineforms::LineForm;
use super::{EXACT_SPAN, NEAR_LINE_SPAN};
use crate::board::Coordinate;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::thread::LocalKey;

/// How far past a class's extent its shift still holds, and below which a distance is read as
/// compressed: classes lie more than [`EXACT_SPAN`] apart, so two reaches stay further apart than this.
pub(crate) const REACH: i64 = NEAR_LINE_SPAN;
const _: () = assert!(EXACT_SPAN > 3 * REACH);

/// Most distances a thread remembers before starting over.
const CACHE_CAP: usize = 1 << 14;

thread_local! {
    /// Each far distance's primality and gap down to the prime below it, per thread so no search
    /// thread waits on another. Both depend on the distance alone, so every position shares them.
    static PRIMES: RefCell<FxHashMap<Big, bool>> = RefCell::default();
    static GAPS: RefCell<FxHashMap<Big, i64>> = RefCell::default();
}

/// One class's compressed extent under a form, and what turns its values back into originals.
#[derive(Debug)]
struct Span {
    low: i64,
    high: i64,
    shift: Big,
}

/// Original distances between the compressed squares of one position.
#[derive(Debug)]
pub struct PrimeDistances {
    forms: Vec<LineForm>,
    /// Per form, its classes in ascending order.
    spans: Vec<Vec<Span>>,
}

impl PrimeDistances {
    /// From each form's `(compressed value, original minus compressed)` of every point.
    pub(super) fn new(forms: Vec<LineForm>, values: Vec<Vec<(i64, Big)>>) -> PrimeDistances {
        PrimeDistances {
            forms,
            spans: values.into_iter().map(spans).collect(),
        }
    }

    /// Whether two squares on one rank or file lie a prime distance apart.
    pub fn is_prime_apart(&self, a: Coordinate, b: Coordinate) -> bool {
        let d = distance_along(a, b);
        if d <= REACH as u64 {
            return crate::utils::is_prime_u64(d);
        }
        self.original_distance(a, b).is_some_and(|d| is_prime(&d))
    }

    /// How far below the original distance between two far squares on one rank or file, which
    /// [`Self::is_prime_apart`] reads as prime, the next prime lies; `None` for a near pair.
    pub fn prime_gap_below(&self, a: Coordinate, b: Coordinate) -> Option<i64> {
        if distance_along(a, b) <= REACH as u64 {
            return None;
        }
        let d = self.original_distance(a, b)?;
        Some(remembered(&GAPS, &d, || {
            let one = Big::from(1);
            let mut below = &d - &one;
            while !below.is_prime() {
                below = &below - &one;
            }
            (&d - &below).to_i64().unwrap()
        }))
    }

    /// The original distance between two squares on one rank or file, through the first form that
    /// runs across the line with both in a class's reach; `None` when there is none.
    fn original_distance(&self, a: Coordinate, b: Coordinate) -> Option<Big> {
        let rank = a.y == b.y;
        self.forms
            .iter()
            .zip(&self.spans)
            .find_map(|(form, spans)| {
                let across = if rank { form.a } else { form.b };
                if across == 0 {
                    return None;
                }
                let original = |s: Coordinate| {
                    let value = form.a as i128 * s.x as i128 + form.b as i128 * s.y as i128;
                    shift(spans, value).map(|shift| &Big::from_i128(value) + shift)
                };
                let (difference, remainder) = (&original(b)? - &original(a)?).div_rem(across);
                (remainder == 0).then(|| difference.abs())
            })
    }
}

/// Whether `|d|` is prime: a search and its lifted lines ask of the same few distances again and again.
pub(super) fn is_prime(d: &Big) -> bool {
    let d = d.abs();
    remembered(&PRIMES, &d, || d.is_prime())
}

/// `d`'s entry in `cache`, computed on a miss.
fn remembered<V: Copy>(
    cache: &'static LocalKey<RefCell<FxHashMap<Big, V>>>,
    d: &Big,
    compute: impl FnOnce() -> V,
) -> V {
    if let Some(known) = cache.with_borrow(|c| c.get(d).copied()) {
        return known;
    }
    let value = compute();
    cache.with_borrow_mut(|c| {
        if c.len() >= CACHE_CAP {
            c.clear();
        }
        c.insert(d.clone(), value);
    });
    value
}

/// How far apart two squares on one rank or file are.
pub(crate) fn distance_along(a: Coordinate, b: Coordinate) -> u64 {
    if a.y == b.y {
        a.x.abs_diff(b.x)
    } else {
        a.y.abs_diff(b.y)
    }
}

/// The shift of the class whose reach holds `value`.
fn shift(spans: &[Span], value: i128) -> Option<&Big> {
    let i = spans.partition_point(|s| (s.high as i128) + (REACH as i128) < value);
    spans
        .get(i)
        .filter(|s| s.low as i128 - REACH as i128 <= value)
        .map(|s| &s.shift)
}

/// Groups one form's values into classes: neighbours within the exact span share one.
fn spans(mut values: Vec<(i64, Big)>) -> Vec<Span> {
    values.sort_by_key(|v| v.0);
    let mut spans: Vec<Span> = Vec::new();
    for (value, shift) in values {
        match spans.last_mut() {
            Some(s) if value - s.high <= EXACT_SPAN => {
                debug_assert_eq!(s.shift, shift, "one class, one shift");
                s.high = value;
            }
            _ => spans.push(Span {
                low: value,
                high: value,
                shift,
            }),
        }
    }
    spans
}
