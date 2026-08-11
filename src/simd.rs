//! Bitboard operations over WebAssembly 128-bit SIMD, auto-enabled for wasm targets
//! via .cargo/config.toml. Operations with no stable intrinsic, such as popcount,
//! fall back to scalar.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

/// SIMD-optimized population count for two 64-bit values.
/// Note: WASM SIMD doesn't have native i64x2_popcnt, so we use scalar.
#[inline(always)]
pub fn popcnt_pair(a: u64, b: u64) -> (u32, u32) {
    // Scalar implementation - WASM SIMD doesn't have vector popcount
    (a.count_ones(), b.count_ones())
}

/// SIMD-optimized check if both bitboards are zero (empty).
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn both_zero(a: u64, b: u64) -> bool {
    // Use v128_any_true on the OR of both values
    // If any bit is set, the result is non-zero, so we check !v128_any_true
    let vec = u64x2(a, b);
    !v128_any_true(vec)
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn both_zero(a: u64, b: u64) -> bool {
    a == 0 && b == 0
}

/// SIMD-optimized check if either bitboard is non-zero (has pieces).
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn either_nonzero(a: u64, b: u64) -> bool {
    let vec = u64x2(a, b);
    v128_any_true(vec)
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn either_nonzero(a: u64, b: u64) -> bool {
    a != 0 || b != 0
}

/// SIMD-optimized bitwise OR of two pairs:
/// Returns (a1 | b1, a2 | b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn or_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_or(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn or_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 | b1, a2 | b2)
}

/// SIMD-optimized bitwise AND of two pairs:
/// Returns (a1 & b1, a2 & b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn and_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_and(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn and_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 & b1, a2 & b2)
}

/// SIMD-optimized bitwise AND-NOT of two pairs:
/// Returns (a1 & !b1, a2 & !b2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn andnot_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    let vec_a = u64x2(a1, a2);
    let vec_b = u64x2(b1, b2);
    let result = v128_andnot(vec_a, vec_b);
    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn andnot_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 & !b1, a2 & !b2)
}

/// Sum two i32 accumulators in parallel.
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn add_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    let vec_a = i32x4(a1, a2, 0, 0);
    let vec_b = i32x4(b1, b2, 0, 0);
    let result = i32x4_add(vec_a, vec_b);
    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Fallback scalar implementation.
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn add_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    (a1 + b1, a2 + b2)
}

// Higher-Level SIMD Helpers for Chess

/// Count pieces for white and black simultaneously in a tile.
#[inline]
pub fn count_pieces_both_colors(occ_white: u64, occ_black: u64) -> (u32, u32) {
    popcnt_pair(occ_white, occ_black)
}

/// Check if a tile has any pieces (either color).
#[inline]
pub fn tile_is_empty(occ_white: u64, occ_black: u64) -> bool {
    both_zero(occ_white, occ_black)
}

/// Check if a tile has pieces of specified color.
#[inline]
pub fn has_pieces_of_either_type(occ_a: u64, occ_b: u64) -> bool {
    either_nonzero(occ_a, occ_b)
}

/// Combine slider bitboards: (bishops | queens, rooks | queens)
#[inline]
pub fn combined_sliders(occ_bishops: u64, occ_rooks: u64, occ_queens: u64) -> (u64, u64) {
    or_pairs(occ_bishops, occ_rooks, occ_queens, occ_queens)
}

// Advanced SIMD Operations for Evaluation

/// SIMD-optimized tapered evaluation.
/// Computes: (mg_score * phase + eg_score * (256 - phase)) / 256
/// Uses SIMD multiply-add when available.
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn tapered_eval_simd(mg_score: i32, eg_score: i32, phase: i32) -> i32 {
    use core::arch::wasm32::*;

    // Pack scores into i32x4: [mg, eg, 0, 0]
    let scores = i32x4(mg_score, eg_score, 0, 0);
    // Pack phases: [phase, 256-phase, 0, 0]
    let phases = i32x4(phase, 256 - phase, 0, 0);

    // Multiply: [mg * phase, eg * (256-phase), 0, 0]
    let products = i32x4_mul(scores, phases);

    // Extract and sum
    let mg_part = i32x4_extract_lane::<0>(products);
    let eg_part = i32x4_extract_lane::<1>(products);

    (mg_part + eg_part) >> 8
}

/// Scalar fallback for tapered eval
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn tapered_eval_simd(mg_score: i32, eg_score: i32, phase: i32) -> i32 {
    (mg_score * phase + eg_score * (256 - phase)) >> 8
}

/// SIMD-optimized material balance calculation.
/// Processes two piece values at a time.
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn material_diff_simd(white_vals: (i32, i32), black_vals: (i32, i32)) -> (i32, i32) {
    use core::arch::wasm32::*;

    let white = i32x4(white_vals.0, white_vals.1, 0, 0);
    let black = i32x4(black_vals.0, black_vals.1, 0, 0);
    let diff = i32x4_sub(white, black);

    (i32x4_extract_lane::<0>(diff), i32x4_extract_lane::<1>(diff))
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn material_diff_simd(white_vals: (i32, i32), black_vals: (i32, i32)) -> (i32, i32) {
    (white_vals.0 - black_vals.0, white_vals.1 - black_vals.1)
}

/// SIMD-optimized max of two i32 pairs: (max(a1,b1), max(a2,b2))
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn max_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let a = i32x4(a1, a2, 0, 0);
    let b = i32x4(b1, b2, 0, 0);
    let result = i32x4_max(a, b);

    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn max_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    (a1.max(b1), a2.max(b2))
}

/// SIMD-optimized min of two i32 pairs: (min(a1,b1), min(a2,b2))
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn min_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let a = i32x4(a1, a2, 0, 0);
    let b = i32x4(b1, b2, 0, 0);
    let result = i32x4_min(a, b);

    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn min_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (i32, i32) {
    (a1.min(b1), a2.min(b2))
}

/// SIMD-optimized clamp of two i32 values to range [lo, hi]
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn clamp_i32_pair(v1: i32, v2: i32, lo: i32, hi: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let v = i32x4(v1, v2, 0, 0);
    let lo_vec = i32x4(lo, lo, 0, 0);
    let hi_vec = i32x4(hi, hi, 0, 0);

    // clamp = min(max(v, lo), hi)
    let clamped = i32x4_min(i32x4_max(v, lo_vec), hi_vec);

    (
        i32x4_extract_lane::<0>(clamped),
        i32x4_extract_lane::<1>(clamped),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn clamp_i32_pair(v1: i32, v2: i32, lo: i32, hi: i32) -> (i32, i32) {
    (v1.clamp(lo, hi), v2.clamp(lo, hi))
}

/// SIMD-optimized absolute value of two i32 values
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn abs_i32_pair(a: i32, b: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let v = i32x4(a, b, 0, 0);
    let result = i32x4_abs(v);

    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn abs_i32_pair(a: i32, b: i32) -> (i32, i32) {
    (a.abs(), b.abs())
}

/// SIMD-optimized negation of two i32 values
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn neg_i32_pair(a: i32, b: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let v = i32x4(a, b, 0, 0);
    let result = i32x4_neg(v);

    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn neg_i32_pair(a: i32, b: i32) -> (i32, i32) {
    (-a, -b)
}

/// SIMD-optimized multiply-accumulate: (a1 + b1*c1, a2 + b2*c2)
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn madd_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32, c1: i32, c2: i32) -> (i32, i32) {
    use core::arch::wasm32::*;

    let a = i32x4(a1, a2, 0, 0);
    let b = i32x4(b1, b2, 0, 0);
    let c = i32x4(c1, c2, 0, 0);

    let product = i32x4_mul(b, c);
    let result = i32x4_add(a, product);

    (
        i32x4_extract_lane::<0>(result),
        i32x4_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn madd_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32, c1: i32, c2: i32) -> (i32, i32) {
    (a1 + b1 * c1, a2 + b2 * c2)
}

/// SIMD-optimized comparison: returns (a1 > b1, a2 > b2) as bools
#[inline(always)]
pub fn gt_i32_pairs(a1: i32, a2: i32, b1: i32, b2: i32) -> (bool, bool) {
    (a1 > b1, a2 > b2)
}

/// SIMD-optimized XOR of two u64 pairs
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn xor_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    use core::arch::wasm32::*;

    let a = u64x2(a1, a2);
    let b = u64x2(b1, b2);
    let result = v128_xor(a, b);

    (
        u64x2_extract_lane::<0>(result),
        u64x2_extract_lane::<1>(result),
    )
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn xor_pairs(a1: u64, a2: u64, b1: u64, b2: u64) -> (u64, u64) {
    (a1 ^ b1, a2 ^ b2)
}

/// Fast horizontal sum of 4 i32 values using SIMD shuffle
#[inline(always)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn hsum_i32x4(a: i32, b: i32, c: i32, d: i32) -> i32 {
    use core::arch::wasm32::*;

    let v = i32x4(a, b, c, d);
    // Shuffle and add: (a+c, b+d, a+c, b+d)
    let shuffled = i32x4_shuffle::<2, 3, 0, 1>(v, v);
    let sum1 = i32x4_add(v, shuffled);
    // Now (a+c, b+d, a+c, b+d), shuffle again
    let shuffled2 = i32x4_shuffle::<1, 0, 3, 2>(sum1, sum1);
    let sum2 = i32x4_add(sum1, shuffled2);

    i32x4_extract_lane::<0>(sum2)
}

/// Scalar fallback
#[inline(always)]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn hsum_i32x4(a: i32, b: i32, c: i32, d: i32) -> i32 {
    a + b + c + d
}

/// Process 4 piece values at once, returning their sum
#[inline(always)]
pub fn sum_piece_values_x4(v1: i32, v2: i32, v3: i32, v4: i32) -> i32 {
    hsum_i32x4(v1, v2, v3, v4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcnt_pair() {
        let (a, b) = popcnt_pair(0b1111, 0b11);
        assert_eq!(a, 4);
        assert_eq!(b, 2);

        let (a, b) = popcnt_pair(u64::MAX, 0);
        assert_eq!(a, 64);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_both_zero() {
        assert!(both_zero(0, 0));
        assert!(!both_zero(1, 0));
        assert!(!both_zero(0, 1));
        assert!(!both_zero(1, 1));
    }

    #[test]
    fn test_or_pairs() {
        let (a, b) = or_pairs(0b1100, 0b1010, 0b0011, 0b0101);
        assert_eq!(a, 0b1111);
        assert_eq!(b, 0b1111);
    }

    #[test]
    fn test_and_pairs() {
        let (a, b) = and_pairs(0b1100, 0b1010, 0b1111, 0b0011);
        assert_eq!(a, 0b1100);
        assert_eq!(b, 0b0010);
    }

    #[test]
    fn test_either_nonzero() {
        assert!(!either_nonzero(0, 0));
        assert!(either_nonzero(1, 0));
        assert!(either_nonzero(0, 1));
        assert!(either_nonzero(1, 1));
        assert!(either_nonzero(u64::MAX, u64::MAX));
    }

    #[test]
    fn test_andnot_pairs() {
        // a1 & !b1, a2 & !b2
        let (a, b) = andnot_pairs(0b1111, 0b1111, 0b1100, 0b0011);
        assert_eq!(a, 0b0011); // 1111 & !1100 = 1111 & 0011 = 0011
        assert_eq!(b, 0b1100); // 1111 & !0011 = 1111 & 1100 = 1100

        let (a, b) = andnot_pairs(0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(a, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_add_i32_pairs() {
        let (a, b) = add_i32_pairs(10, 20, 5, 15);
        assert_eq!(a, 15);
        assert_eq!(b, 35);

        let (a, b) = add_i32_pairs(-10, -20, 30, 40);
        assert_eq!(a, 20);
        assert_eq!(b, 20);
    }

    #[test]
    fn test_count_pieces_both_colors() {
        let (w, b) = count_pieces_both_colors(0b11111111, 0b1111);
        assert_eq!(w, 8);
        assert_eq!(b, 4);
    }

    #[test]
    fn test_tile_is_empty() {
        assert!(tile_is_empty(0, 0));
        assert!(!tile_is_empty(1, 0));
        assert!(!tile_is_empty(0, 1));
    }

    #[test]
    fn test_has_pieces_of_either_type() {
        assert!(!has_pieces_of_either_type(0, 0));
        assert!(has_pieces_of_either_type(1, 0));
        assert!(has_pieces_of_either_type(0, 1));
    }

    #[test]
    fn test_combined_sliders() {
        // bishops=0b0001, rooks=0b0010, queens=0b0100
        // Result: (bishops|queens, rooks|queens) = (0b0101, 0b0110)
        let (diag, ortho) = combined_sliders(0b0001, 0b0010, 0b0100);
        assert_eq!(diag, 0b0101);
        assert_eq!(ortho, 0b0110);
    }

    #[test]
    fn test_tapered_eval_simd() {
        // When phase = 256 (full middlegame), should return mg_score
        let result = tapered_eval_simd(100, 50, 256);
        assert_eq!(result, 100);

        // When phase = 0 (full endgame), should return eg_score
        let result = tapered_eval_simd(100, 50, 0);
        assert_eq!(result, 50);

        // When phase = 128 (midpoint), should return average
        let result = tapered_eval_simd(100, 50, 128);
        assert_eq!(result, 75);
    }

    #[test]
    fn test_material_diff_simd() {
        let (a, b) = material_diff_simd((1000, 500), (800, 600));
        assert_eq!(a, 200); // 1000 - 800
        assert_eq!(b, -100); // 500 - 600
    }

    #[test]
    fn test_max_i32_pairs() {
        let (a, b) = max_i32_pairs(10, 20, 15, 5);
        assert_eq!(a, 15);
        assert_eq!(b, 20);

        let (a, b) = max_i32_pairs(-10, -20, -5, -30);
        assert_eq!(a, -5);
        assert_eq!(b, -20);
    }

    #[test]
    fn test_min_i32_pairs() {
        let (a, b) = min_i32_pairs(10, 20, 15, 5);
        assert_eq!(a, 10);
        assert_eq!(b, 5);
    }

    #[test]
    fn test_clamp_i32_pair() {
        let (a, b) = clamp_i32_pair(50, 150, 0, 100);
        assert_eq!(a, 50);
        assert_eq!(b, 100);

        let (a, b) = clamp_i32_pair(-50, 50, 0, 100);
        assert_eq!(a, 0);
        assert_eq!(b, 50);
    }

    #[test]
    fn test_abs_i32_pair() {
        let (a, b) = abs_i32_pair(-10, 20);
        assert_eq!(a, 10);
        assert_eq!(b, 20);

        let (a, b) = abs_i32_pair(-100, -200);
        assert_eq!(a, 100);
        assert_eq!(b, 200);
    }

    #[test]
    fn test_neg_i32_pair() {
        let (a, b) = neg_i32_pair(10, -20);
        assert_eq!(a, -10);
        assert_eq!(b, 20);
    }

    #[test]
    fn test_madd_i32_pairs() {
        // (a1 + b1*c1, a2 + b2*c2)
        let (a, b) = madd_i32_pairs(10, 20, 3, 4, 5, 6);
        assert_eq!(a, 10 + 3 * 5); // 25
        assert_eq!(b, 20 + 4 * 6); // 44
    }

    #[test]
    fn test_xor_pairs() {
        let (a, b) = xor_pairs(0b1111, 0b1010, 0b0011, 0b1111);
        assert_eq!(a, 0b1100); // 1111 ^ 0011
        assert_eq!(b, 0b0101); // 1010 ^ 1111
    }

    #[test]
    fn test_hsum_i32x4() {
        let sum = hsum_i32x4(10, 20, 30, 40);
        assert_eq!(sum, 100);

        let sum = hsum_i32x4(-10, 20, -30, 40);
        assert_eq!(sum, 20);
    }

    #[test]
    fn test_sum_piece_values_x4() {
        let sum = sum_piece_values_x4(100, 450, 650, 1350);
        assert_eq!(sum, 2550);
    }
}
