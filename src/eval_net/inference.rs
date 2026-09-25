//! Integer forward pass for the Stage-A residual net. Mirrors the quantization
//! contract in `evalnet/export_eval_net.py`: any change here needs the same change
//! there, verified by its integer-simulation check.

use super::features::NUM_FEATURES;
use super::variant_features::MAX_VARIANT_FEATURES;
use super::weights::EvalNetWeights;

/// Hard cap on the residual so a bad net can misjudge, never dominate. Must
/// match the `--cap` the net was trained with.
pub const RESIDUAL_CAP: i32 = 500;

/// Widest hidden layer the stack buffers below allow.
pub const MAX_H: usize = 256;

/// i16 weights (widened i8) against i16 activations over a 32-padded row:
/// one `pmaddwd` per 8 MACs, no tail. `w` and `x` are at least `len` long.
#[inline(always)]
fn dot_i16(w: &[i16], x: &[i16], len: usize) -> i32 {
    debug_assert!(len.is_multiple_of(32) && w.len() >= len && x.len() >= len);
    dot_i16_chunks(&w[..len], &x[..len])
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dot_i16_chunks(w: &[i16], x: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm_setzero_si128();
        for i in (0..w.len()).step_by(8) {
            let wv = _mm_loadu_si128(w.as_ptr().add(i) as *const __m128i);
            let xv = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
            acc = _mm_add_epi32(acc, _mm_madd_epi16(wv, xv));
        }
        let hi = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        let hi = _mm_add_epi32(hi, _mm_shuffle_epi32(hi, 0b10_11_00_01));
        _mm_cvtsi128_si32(hi)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn dot_i16_chunks(w: &[i16], x: &[i16]) -> i32 {
    use std::arch::wasm32::*;
    unsafe {
        let mut acc = i32x4_splat(0);
        for i in (0..w.len()).step_by(8) {
            let wv = v128_load(w.as_ptr().add(i) as *const v128);
            let xv = v128_load(x.as_ptr().add(i) as *const v128);
            acc = i32x4_add(acc, i32x4_dot_i16x8(wv, xv));
        }
        i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
fn dot_i16_chunks(w: &[i16], x: &[i16]) -> i32 {
    let mut sum = 0i32;
    for (wc, xc) in w.chunks_exact(8).zip(x.chunks_exact(8)) {
        for k in 0..8 {
            sum += wc[k] as i32 * xc[k] as i32;
        }
    }
    sum
}

/// Eight rows per pass: one input load feeds eight madds, and a hadd tree turns
/// the eight accumulators into one vector of row sums instead of reducing each row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_layer_avx2(
    w: &[i16],
    b: &[i32],
    stride: usize,
    x: &[i16],
    shift: u32,
    out: &mut [i16],
) {
    use std::arch::x86_64::*;
    debug_assert!(out.len().is_multiple_of(8) && stride.is_multiple_of(16));
    debug_assert!(w.len() >= out.len() * stride && x.len() >= stride && b.len() >= out.len());
    unsafe {
        let count = _mm_cvtsi32_si128(shift as i32);
        let (zero, max) = (_mm256_setzero_si256(), _mm256_set1_epi32(127));
        for g in (0..out.len()).step_by(8) {
            let rows = w.as_ptr().add(g * stride);
            let mut acc = [_mm256_setzero_si256(); 8];
            for i in (0..stride).step_by(16) {
                let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
                for (r, a) in acc.iter_mut().enumerate() {
                    let wv = _mm256_loadu_si256(rows.add(r * stride + i) as *const __m256i);
                    *a = _mm256_add_epi32(*a, _mm256_madd_epi16(wv, xv));
                }
            }
            let s01 = _mm256_hadd_epi32(acc[0], acc[1]);
            let s23 = _mm256_hadd_epi32(acc[2], acc[3]);
            let s45 = _mm256_hadd_epi32(acc[4], acc[5]);
            let s67 = _mm256_hadd_epi32(acc[6], acc[7]);
            let t0 = _mm256_hadd_epi32(s01, s23);
            let t1 = _mm256_hadd_epi32(s45, s67);
            let sums = _mm256_add_epi32(
                _mm256_permute2x128_si256(t0, t1, 0x20),
                _mm256_permute2x128_si256(t0, t1, 0x31),
            );
            let v = _mm256_add_epi32(sums, _mm256_loadu_si256(b.as_ptr().add(g) as *const __m256i));
            let v = _mm256_min_epi32(_mm256_max_epi32(_mm256_sra_epi32(v, count), zero), max);
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
            for (o, &l) in out[g..g + 8].iter_mut().zip(&lanes) {
                *o = l as i16;
            }
        }
    }
}

/// `x` must be zero beyond the layer's real input count up to `stride`.
#[inline(always)]
fn dense_layer(
    w: &[i16],
    b: &[i32],
    stride: usize,
    x: &[i16],
    shift: u32,
    out: &mut [i16],
    avx2: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2 && out.len().is_multiple_of(8) {
        return unsafe { dense_layer_avx2(w, b, stride, x, shift, out) };
    }
    let _ = avx2;
    for (r, o) in out.iter_mut().enumerate() {
        let d = dot_i16(&w[r * stride..(r + 1) * stride], x, stride);
        *o = ((b[r] + d) >> shift).clamp(0, 127) as i16;
    }
}

#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use once_cell::sync::Lazy;
        static HAS_AVX2: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("avx2"));
        *HAS_AVX2
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// The SIMD dense layer checked against the scalar integer layer on pseudo-random
/// weights, run inside whatever build calls it (the wasm harness uses it).
#[cfg(any(test, feature = "bench_positions"))]
pub fn kernel_selftest() -> bool {
    let (n_out, stride, shift) = (64usize, 128usize, 6u32);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = |m: i64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i64 % (2 * m + 1) - m) as i32
    };
    for _ in 0..50 {
        let w: Vec<i16> = (0..n_out * stride).map(|_| next(128) as i16).collect();
        let b: Vec<i32> = (0..n_out).map(|_| next(20_000)).collect();
        let x: Vec<i16> = (0..stride).map(|_| next(600) as i16).collect();
        for avx2 in [false, has_avx2()] {
            let mut got = vec![0i16; n_out];
            dense_layer(&w, &b, stride, &x, shift, &mut got, avx2);
            for r in 0..n_out {
                let d: i32 = (0..stride).map(|i| w[r * stride + i] as i32 * x[i] as i32).sum();
                if got[r] != ((b[r] + d) >> shift).clamp(0, 127) as i16 {
                    return false;
                }
            }
        }
    }
    true
}

/// Input buffer wide enough for every layout, padded to the layer-1 stride.
pub const NUM_FEATURES_PAD: usize = super::weights::pad32(if NUM_FEATURES > MAX_VARIANT_FEATURES {
    NUM_FEATURES
} else {
    MAX_VARIANT_FEATURES
});

/// Raw net output in centipawns (White-ahead), before the residual cap. `x` holds at
/// least the net's `n_in` inputs.
pub fn forward(net: &EvalNetWeights, x: &[i16]) -> i32 {
    debug_assert!(net.h1 <= MAX_H && net.h2 <= MAX_H);
    debug_assert!(net.n_in <= x.len() && net.stride1 <= NUM_FEATURES_PAD);

    // CReLU outputs fit i16, so every layer reuses the same i16 x i16 kernel.
    // The zero padding of each buffer covers the padded weight columns.
    let avx2 = has_avx2();
    let mut xp = [0i16; NUM_FEATURES_PAD];
    xp[..net.n_in].copy_from_slice(&x[..net.n_in]);

    let mut h1 = [0i16; MAX_H];
    dense_layer(net.l1_w.as_slice(), &net.l1_b, net.stride1, &xp, net.s1, &mut h1[..net.h1], avx2);

    let mut h2 = [0i16; MAX_H];
    dense_layer(net.l2_w.as_slice(), &net.l2_b, net.stride2, &h1, net.s2, &mut h2[..net.h2], avx2);

    let raw = net.l3_b + dot_i16(net.l3_w.as_slice(), &h2, super::weights::pad32(net.h2));
    (raw as f32 * net.out_scale) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_net() -> EvalNetWeights {
        use super::super::weights::{AlignedI16, pad32};
        let n_in = NUM_FEATURES;
        EvalNetWeights {
            perspective: false,
            n_in,
            h1: 32,
            h2: 32,
            stride1: pad32(n_in),
            stride2: pad32(32),
            s1: 6,
            s2: 6,
            out_scale: 0.1,
            l1_w: AlignedI16::from_rows(&vec![1i16; 32 * n_in], n_in, pad32(n_in), 32),
            l1_b: vec![0i32; 32].into_boxed_slice(),
            l2_w: AlignedI16::from_rows(&vec![1i16; 32 * 32], 32, pad32(32), 32),
            l2_b: vec![0i32; 32].into_boxed_slice(),
            l3_w: AlignedI16::from_rows(&[1i16; 32], 32, pad32(32), 1),
            l3_b: 0,
        }
    }

    #[test]
    fn forward_matches_hand_computation() {
        let net = tiny_net();
        let mut x = [0i16; NUM_FEATURES];
        x[0] = 640; // acc1 = 640 -> h1 = min(640 >> 6, 127) = 10 for every neuron
        let h1 = (640 >> 6).clamp(0, 127);
        let h2 = ((h1 * 32) >> 6).clamp(0, 127);
        let raw = h2 * 32;
        assert_eq!(forward(&net, &x), (raw as f32 * 0.1) as i32);
    }

    /// Every kernel must reproduce the scalar integer layer exactly.
    #[test]
    fn dense_layer_matches_scalar_reference() {
        assert!(kernel_selftest());
    }

    #[test]
    fn forward_zero_input_is_bias_only() {
        let net = tiny_net();
        let x = [0i16; NUM_FEATURES];
        assert_eq!(forward(&net, &x), 0);
    }

    #[test]
    fn simd_dot_matches_scalar_with_signs() {
        let n = 256;
        let w: Vec<i16> = (0..n).map(|i| (i * 37 % 255 - 127) as i16).collect();
        let x: Vec<i16> = (0..n).map(|i| (i * 911 % 4001 - 2000) as i16).collect();
        for len in [32usize, 64, 96, 160, 256] {
            let s: i32 = w[..len].iter().zip(&x[..len]).map(|(a, b)| *a as i32 * *b as i32).sum();
            assert_eq!(dot_i16(&w, &x, len), s, "len {len}");
        }
    }

    #[test]
    fn aligned_buffer_is_64_byte_aligned() {
        let b = super::super::weights::AlignedI16::zeroed(1000);
        assert_eq!(b.as_slice().as_ptr() as usize % 64, 0);
        assert!(b.as_slice().len() >= 1000);
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// `cargo test --release --lib eval_net::inference::timing -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn forward_ns() {
        let Some(net) = super::super::weights::EVAL_NET.as_ref() else {
            eprintln!("no net embedded");
            return;
        };
        let mut xs: Vec<[i16; NUM_FEATURES]> = Vec::new();
        for s in 0..256u32 {
            let mut x = [0i16; NUM_FEATURES];
            for (i, v) in x.iter_mut().enumerate() {
                *v = (((i as u32 * 2654435761u32).wrapping_add(s * 97)) % 400) as i16 - 100;
            }
            xs.push(x);
        }
        let iters = 200_000;
        let t = std::time::Instant::now();
        let mut sink = 0i64;
        for k in 0..iters {
            sink += forward(net, &xs[k & 255]) as i64;
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        eprintln!(
            "forward: {ns:.0} ns ({}x{}x{}), avx2={} sink={sink}",
            net.n_in,
            net.h1,
            net.h2,
            is_x86_feature_detected!("avx2")
        );
    }
}
