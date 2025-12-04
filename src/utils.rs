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

    // 6k ± 1 wheel: 5, 7, 11, 13, 17, 19, ...
    let mut i: i64 = 5;
    let mut step: i64 = 2;

    // Avoid overflow via division
    while i <= n / i {
        if n % i == 0 {
            return false;
        }
        i += step;
        step = 6 - step;
    }

    true
}
