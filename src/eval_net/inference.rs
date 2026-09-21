//! Integer forward pass for the Stage-A residual net. Mirrors the quantization
//! contract in `nnue/export_eval_net.py`: any change here needs the same change
//! there, verified by its integer-simulation check.

use super::features::NUM_FEATURES;
use super::weights::EvalNetWeights;

/// Hard cap on the residual so a bad net can misjudge, never dominate.
pub const RESIDUAL_CAP: i32 = 250;

#[inline(always)]
fn dot_i8_i16(w: &[i8], x: &[i16]) -> i32 {
    let mut sum = 0i32;
    for (wc, xc) in w.chunks_exact(16).zip(x.chunks_exact(16)) {
        for k in 0..16 {
            sum += wc[k] as i32 * xc[k] as i32;
        }
    }
    for (wr, xr) in w
        .chunks_exact(16)
        .remainder()
        .iter()
        .zip(x.chunks_exact(16).remainder())
    {
        sum += *wr as i32 * *xr as i32;
    }
    sum
}

#[inline(always)]
fn dot_i8_i32(w: &[i8], x: &[i32]) -> i32 {
    let mut sum = 0i32;
    for (wc, xc) in w.chunks_exact(16).zip(x.chunks_exact(16)) {
        for k in 0..16 {
            sum += wc[k] as i32 * xc[k];
        }
    }
    sum
}

/// Raw net output in centipawns (White-ahead), before the residual cap.
pub fn forward(net: &EvalNetWeights, x: &[i16; NUM_FEATURES]) -> i32 {
    const MAX_H: usize = 64;
    debug_assert!(net.h1 <= MAX_H && net.h2 <= MAX_H);

    let mut h1 = [0i32; MAX_H];
    for (i, h) in h1[..net.h1].iter_mut().enumerate() {
        let row = &net.l1_w[i * net.n_in..(i + 1) * net.n_in];
        let acc = net.l1_b[i] + dot_i8_i16(row, x);
        *h = (acc >> net.s1).clamp(0, 127);
    }

    let mut h2 = [0i32; MAX_H];
    for (i, h) in h2[..net.h2].iter_mut().enumerate() {
        let row = &net.l2_w[i * net.h1..(i + 1) * net.h1];
        let acc = net.l2_b[i] + dot_i8_i32(row, &h1[..net.h1]);
        *h = (acc >> net.s2).clamp(0, 127);
    }

    let raw = net.l3_b + dot_i8_i32(&net.l3_w, &h2[..net.h2]);
    (raw as f32 * net.out_scale) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_net() -> EvalNetWeights {
        let n_in = NUM_FEATURES;
        EvalNetWeights {
            n_in,
            h1: 32,
            h2: 32,
            s1: 6,
            s2: 6,
            out_scale: 0.1,
            l1_w: vec![1i8; 32 * n_in].into_boxed_slice(),
            l1_b: vec![0i32; 32].into_boxed_slice(),
            l2_w: vec![1i8; 32 * 32].into_boxed_slice(),
            l2_b: vec![0i32; 32].into_boxed_slice(),
            l3_w: vec![1i8; 32].into_boxed_slice(),
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

    #[test]
    fn forward_zero_input_is_bias_only() {
        let net = tiny_net();
        let x = [0i16; NUM_FEATURES];
        assert_eq!(forward(&net, &x), 0);
    }
}
