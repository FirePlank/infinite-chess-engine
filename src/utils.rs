// Precomputed Prime Lookup Tables for O(1) Huygens distance checks

/// All primes under 128 - used for Huygens move generation iteration
pub static PRIMES_UNDER_128: [i64; 31] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127,
];

/// O(1) prime lookup for values 0-127. Index with distance to check primality instantly.
pub static IS_PRIME_LOOKUP: [bool; 128] = {
    let mut table = [false; 128];
    table[2] = true;
    table[3] = true;
    table[5] = true;
    table[7] = true;
    table[11] = true;
    table[13] = true;
    table[17] = true;
    table[19] = true;
    table[23] = true;
    table[29] = true;
    table[31] = true;
    table[37] = true;
    table[41] = true;
    table[43] = true;
    table[47] = true;
    table[53] = true;
    table[59] = true;
    table[61] = true;
    table[67] = true;
    table[71] = true;
    table[73] = true;
    table[79] = true;
    table[83] = true;
    table[89] = true;
    table[97] = true;
    table[101] = true;
    table[103] = true;
    table[107] = true;
    table[109] = true;
    table[113] = true;
    table[127] = true;
    table
};

/// Fast O(1) prime check for distances under 128, falls back to O(log^3 n) for larger values.
/// This is the hot path for Huygens piece logic where distances are typically < 100.
#[inline(always)]
pub fn is_prime_fast(n: i64) -> bool {
    let abs_n = n.abs();
    if abs_n < 128 {
        IS_PRIME_LOOKUP[abs_n as usize]
    } else {
        is_prime_i64(n)
    }
}

pub fn is_prime_i64(n: i64) -> bool {
    // i64::MIN cannot be negated; and it's even anyway, so not prime.
    if n == i64::MIN {
        return false;
    }

    let n = n.abs();

    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // For large numbers, use Miller-Rabin primality test (O(log^3 n))
    // rather than trial division (O(sqrt(n))).
    // 90 trillion takes ~10M iterations with trial division but < 100 with Miller-Rabin.
    is_prime_miller_rabin(n)
}

fn mod_pow(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    if modulus == 1 {
        return 0;
    }
    let mut res = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            res = (res * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    res
}

fn is_prime_miller_rabin(n: i64) -> bool {
    let n = n.abs();
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // Write n-1 as 2^s * d
    let mut d = (n - 1) as u128;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    // Witnesses for 64-bit integers
    // See: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases
    let witnesses: [u128; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    for &a in &witnesses {
        if a >= n as u128 {
            break;
        }
        let mut x = mod_pow(a, d, n as u128);
        if x == 1 || x == (n - 1) as u128 {
            continue;
        }
        let mut composite = true;
        for _ in 0..s - 1 {
            x = (x * x) % n as u128;
            if x == (n - 1) as u128 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_primes() {
        assert!(is_prime_i64(2));
        assert!(is_prime_i64(3));
        assert!(is_prime_i64(5));
        assert!(is_prime_i64(7));
        assert!(is_prime_i64(11));
        assert!(is_prime_i64(13));
        assert!(is_prime_i64(17));
        assert!(is_prime_i64(19));
        assert!(is_prime_i64(23));
    }

    #[test]
    fn test_small_composites() {
        assert!(!is_prime_i64(4));
        assert!(!is_prime_i64(6));
        assert!(!is_prime_i64(8));
        assert!(!is_prime_i64(9));
        assert!(!is_prime_i64(10));
        assert!(!is_prime_i64(12));
        assert!(!is_prime_i64(15));
        assert!(!is_prime_i64(21));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!is_prime_i64(0));
        assert!(!is_prime_i64(1));
        assert!(!is_prime_i64(-1));
        assert!(!is_prime_i64(i64::MIN));
    }

    #[test]
    fn test_negative_primes() {
        // abs() of negative primes should still return true
        assert!(is_prime_i64(-2));
        assert!(is_prime_i64(-3));
        assert!(is_prime_i64(-5));
        assert!(is_prime_i64(-7));
    }

    #[test]
    fn test_larger_primes() {
        assert!(is_prime_i64(97));
        assert!(is_prime_i64(101));
        assert!(is_prime_i64(1009));
    }

    #[test]
    fn test_larger_composites() {
        assert!(!is_prime_i64(100));
        assert!(!is_prime_i64(1000));
        assert!(!is_prime_i64(49)); // 7*7
        assert!(!is_prime_i64(121)); // 11*11
    }
}
