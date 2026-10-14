//! A signed big integer with exactly the operations compression needs. Every multiplier and
//! divisor it meets is a small form coefficient, determinant or modulus, so those take an `i64`.

use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Little-endian limbs without trailing zeros; zero is empty. 256 bits stay inline. Every operation
/// stays out of line: inlined, each would copy its limb loops into hundreds of callers.
type Limbs = SmallVec<[u64; 4]>;

#[derive(PartialEq, Eq, Hash, Default)]
pub struct Big {
    neg: bool,
    mag: Limbs,
}

impl Big {
    pub fn zero() -> Big {
        Big::default()
    }

    #[inline(never)]
    pub fn from_i128(v: i128) -> Big {
        let u = v.unsigned_abs();
        let mut mag: Limbs = [u as u64, (u >> 64) as u64].into_iter().collect();
        while mag.last() == Some(&0) {
            mag.pop();
        }
        Big {
            neg: v < 0 && !mag.is_empty(),
            mag,
        }
    }

    #[inline(never)]
    pub fn to_i128(&self) -> Option<i128> {
        let u = match self.mag.len() {
            0 => 0u128,
            1 => self.mag[0] as u128,
            2 => self.mag[0] as u128 | (self.mag[1] as u128) << 64,
            _ => return None,
        };
        if self.neg {
            (u <= 1 << 127).then(|| (u as i128).wrapping_neg())
        } else {
            i128::try_from(u).ok()
        }
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.to_i128().and_then(|v| i64::try_from(v).ok())
    }

    /// Parses an optionally `-`-signed decimal integer.
    #[inline(never)]
    pub fn parse(s: &str) -> Option<Big> {
        let (neg, digits) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut mag = Limbs::new();
        let head = digits.len() % 19;
        let chunks = std::iter::once(&digits[..head])
            .filter(|c| !c.is_empty())
            .chain(
                digits.as_bytes()[head..]
                    .chunks(19)
                    .map(|c| std::str::from_utf8(c).unwrap()),
            );
        for chunk in chunks {
            mul_add_small(
                &mut mag,
                10u64.pow(chunk.len() as u32),
                chunk.parse().unwrap(),
            );
        }
        let mut big = Big { neg, mag };
        big.normalize();
        Some(big)
    }

    fn normalize(&mut self) {
        while self.mag.last() == Some(&0) {
            self.mag.pop();
        }
        if self.mag.is_empty() {
            self.neg = false;
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub fn is_neg(&self) -> bool {
        self.neg
    }

    #[inline(never)]
    pub fn abs(&self) -> Big {
        Big {
            neg: false,
            mag: self.mag.clone(),
        }
    }

    /// Truncating division and remainder, like JS BigInt's `/` and `%`.
    #[inline(never)]
    pub fn div_rem(&self, k: i64) -> (Big, i64) {
        let mut mag = self.mag.clone();
        let r = divrem_small_mag(&mut mag, k.unsigned_abs()) as i64;
        let mut q = Big {
            neg: self.neg != (k < 0),
            mag,
        };
        q.normalize();
        (q, if self.neg { -r } else { r })
    }

    /// The remainder in `[0, |k|)`.
    #[inline(never)]
    pub fn rem_euclid(&self, k: i64) -> i64 {
        let r = self.div_rem(k).1;
        if r < 0 { r + k.abs() } else { r }
    }

    /// Floor division by `2^bits`, like JS BigInt's `>>`.
    #[inline(never)]
    pub fn shr(&self, bits: u32) -> Big {
        let (limbs, rem) = ((bits / 64) as usize, bits % 64);
        if limbs >= self.mag.len() {
            return if self.neg {
                Big::from_i128(-1)
            } else {
                Big::zero()
            };
        }
        let mut lost = self.mag[..limbs].iter().any(|&l| l != 0);
        let mut mag: Limbs = self.mag[limbs..].into();
        if rem > 0 {
            lost |= mag[0] & ((1 << rem) - 1) != 0;
            for i in 0..mag.len() {
                let hi = mag.get(i + 1).map_or(0, |&h| h << (64 - rem));
                mag[i] = (mag[i] >> rem) | hi;
            }
        }
        let mut b = Big { neg: self.neg, mag };
        b.normalize();
        if self.neg && lost {
            &b - &Big::from_i128(1)
        } else {
            b
        }
    }

    #[inline(never)]
    pub fn bit_length(&self) -> u32 {
        self.mag.last().map_or(0, |top| {
            (self.mag.len() as u32 - 1) * 64 + (64 - top.leading_zeros())
        })
    }

    /// The nearest double, rounding half to even like JS's `Number(bigint)`.
    #[inline(never)]
    pub fn to_f64(&self) -> f64 {
        let bits = self.bit_length();
        let v = if bits <= 64 {
            self.mag.first().map_or(0, |&l| l) as f64
        } else {
            // The top 64 bits, with every lower bit folded into a sticky last bit, round once.
            let shift = bits - 64;
            let top = self.abs().shr(shift).mag[0] | self.mag_low_bits_nonzero(shift) as u64;
            if shift > 1023 {
                f64::INFINITY
            } else {
                top as f64 * 2f64.powi(shift as i32)
            }
        };
        if self.neg { -v } else { v }
    }

    fn mag_low_bits_nonzero(&self, bits: u32) -> bool {
        let (limbs, rem) = ((bits / 64) as usize, bits % 64);
        self.mag[..limbs.min(self.mag.len())]
            .iter()
            .any(|&l| l != 0)
            || (rem > 0
                && self
                    .mag
                    .get(limbs)
                    .is_some_and(|&l| l & ((1 << rem) - 1) != 0))
    }
}

impl Clone for Big {
    #[inline(never)]
    fn clone(&self) -> Big {
        Big {
            neg: self.neg,
            mag: self.mag.clone(),
        }
    }
}

impl From<i64> for Big {
    fn from(v: i64) -> Big {
        Big::from_i128(v as i128)
    }
}

impl Ord for Big {
    #[inline(never)]
    fn cmp(&self, other: &Big) -> Ordering {
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.mag, &other.mag),
            (true, true) => cmp_mag(&other.mag, &self.mag),
        }
    }
}

impl PartialOrd for Big {
    fn partial_cmp(&self, other: &Big) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for &Big {
    type Output = Big;
    #[inline(never)]
    fn add(self, other: &Big) -> Big {
        if self.neg == other.neg {
            return Big {
                neg: self.neg,
                mag: add_mag(&self.mag, &other.mag),
            };
        }
        let mut b = match cmp_mag(&self.mag, &other.mag) {
            Ordering::Equal => return Big::zero(),
            Ordering::Greater => Big {
                neg: self.neg,
                mag: sub_mag(&self.mag, &other.mag),
            },
            Ordering::Less => Big {
                neg: other.neg,
                mag: sub_mag(&other.mag, &self.mag),
            },
        };
        b.normalize();
        b
    }
}

impl Sub for &Big {
    type Output = Big;
    #[inline(never)]
    fn sub(self, other: &Big) -> Big {
        self + &-other
    }
}

impl Neg for &Big {
    type Output = Big;
    #[inline(never)]
    fn neg(self) -> Big {
        Big {
            neg: !self.neg && !self.mag.is_empty(),
            mag: self.mag.clone(),
        }
    }
}

impl Mul<i64> for &Big {
    type Output = Big;
    #[inline(never)]
    fn mul(self, k: i64) -> Big {
        if k == 0 || self.mag.is_empty() {
            return Big::zero();
        }
        let mut mag = self.mag.clone();
        mul_add_small(&mut mag, k.unsigned_abs(), 0);
        Big {
            neg: self.neg != (k < 0),
            mag,
        }
    }
}

impl fmt::Display for Big {
    #[inline(never)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mag.is_empty() {
            return f.write_str("0");
        }
        let mut mag = self.mag.clone();
        let mut chunks = Vec::new();
        while !mag.is_empty() {
            chunks.push(divrem_small_mag(&mut mag, 10_000_000_000_000_000_000));
        }
        if self.neg {
            f.write_str("-")?;
        }
        write!(f, "{}", chunks.pop().unwrap())?;
        chunks.iter().rev().try_for_each(|c| write!(f, "{c:019}"))
    }
}

impl fmt::Debug for Big {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_mag(a: &[u64], b: &[u64]) -> Limbs {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Limbs::with_capacity(long.len() + 1);
    let mut carry = false;
    for (i, &l) in long.iter().enumerate() {
        let (s1, c1) = l.overflowing_add(short.get(i).copied().unwrap_or(0));
        let (s2, c2) = s1.overflowing_add(carry as u64);
        out.push(s2);
        carry = c1 | c2;
    }
    if carry {
        out.push(1);
    }
    out
}

/// `a - b` for `|a| >= |b|`.
fn sub_mag(a: &[u64], b: &[u64]) -> Limbs {
    let mut out = Limbs::with_capacity(a.len());
    let mut borrow = false;
    for (i, &l) in a.iter().enumerate() {
        let (d1, b1) = l.overflowing_sub(b.get(i).copied().unwrap_or(0));
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        out.push(d2);
        borrow = b1 | b2;
    }
    out
}

fn mul_add_small(mag: &mut Limbs, k: u64, add: u64) {
    let mut carry = add as u128;
    for limb in mag.iter_mut() {
        let p = *limb as u128 * k as u128 + carry;
        *limb = p as u64;
        carry = p >> 64;
    }
    if carry != 0 {
        mag.push(carry as u64);
    }
}

fn divrem_small_mag(mag: &mut Limbs, k: u64) -> u64 {
    let mut rem = 0u128;
    for limb in mag.iter_mut().rev() {
        let cur = (rem << 64) | *limb as u128;
        *limb = (cur / k as u128) as u64;
        rem = cur % k as u128;
    }
    while mag.last() == Some(&0) {
        mag.pop();
    }
    rem as u64
}

#[cfg(test)]
mod tests {
    use super::Big;

    fn b(s: &str) -> Big {
        Big::parse(s).unwrap()
    }

    #[test]
    fn round_trips_and_arithmetic() {
        for s in [
            "0",
            "-1",
            "18446744073709551616",
            "-340282366920938463463374607431768211457",
            "123456789012345678901234567890123456789012345678901234567890",
        ] {
            assert_eq!(b(s).to_string(), s);
        }
        assert_eq!(
            (&b("99999999999999999999") + &b("1")).to_string(),
            "100000000000000000000"
        );
        assert_eq!((&b("-5") - &b("-7")).to_string(), "2");
        assert_eq!(
            (&b("-123456789012345678901234567890") * -3).to_string(),
            "370370367037037036703703703670"
        );
        assert_eq!(b("-7").div_rem(2), (b("-3"), -1));
        assert_eq!(b("-7").rem_euclid(8), 1);
        assert_eq!(b("-7").shr(1).to_string(), "-4");
        assert_eq!(
            b("-340282366920938463463374607431768211456")
                .shr(64)
                .to_string(),
            "-18446744073709551616"
        );
        assert_eq!(b("9007199254740993").to_f64(), 9007199254740992.0);
        assert_eq!(b("36893488147419103233").to_f64(), 36893488147419103232.0);
        assert_eq!(b("36893488147419107328").to_f64(), 36893488147419103232.0); // A tie rounds to even.
        assert_eq!(b("36893488147419107329").to_f64(), 36893488147419111424.0);
    }
}
