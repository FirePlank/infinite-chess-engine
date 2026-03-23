//! NNUE Quantized Inference
//!
//! Performs fast quantized inference using i16/i8 dot products.
//! The heavy computation is integer MACs; only final output conversion uses floats.

use super::features::build_threat_active_lists;
use super::state::{NnueState, RELKP_DIM};
use super::weights::{NNUE_WEIGHTS, NnueWeights};
use crate::board::PlayerColor;
use crate::game::GameState;

/// Threat stream dimension
const THREAT_DIM: usize = 64;
/// Head input dimension (2 * (256 + 64))
const HEAD_IN: usize = 640;
/// Hidden layer dimensions
const H1: usize = 32;
const H2: usize = 32;

/// Clipped ReLU activation for i16 values.
/// Clamps to [0, 127] (scaled for quantization).
#[inline]
fn crelu_i16(x: i16) -> i16 {
    x.clamp(0, 127)
}

/// Clipped ReLU activation for i32 values.
/// Clamps to [0, 127] after scaling.
#[inline]
fn crelu_i32(x: i32) -> i32 {
    x.clamp(0, 127)
}

/// Compute threat stream accumulator from active features.
fn compute_threat_stream(weights: &NnueWeights, features: &[u32]) -> [i16; THREAT_DIM] {
    let mut acc = [0i16; THREAT_DIM];

    // Start with bias
    for (i, v) in acc.iter_mut().enumerate() {
        *v = weights.thr_bias[i];
    }

    // Accumulate active features
    for &feat_id in features {
        let offset = (feat_id as usize) * THREAT_DIM;
        if offset + THREAT_DIM <= weights.thr_embed.len() {
            for (i, v) in acc.iter_mut().enumerate() {
                *v = v.saturating_add(weights.thr_embed[offset + i]);
            }
        }
    }

    acc
}

/// Apply ClippedReLU and concatenate to head input.
fn build_head_input(
    rel_friendly: &[i16; RELKP_DIM],
    thr_friendly: &[i16; THREAT_DIM],
    rel_enemy: &[i16; RELKP_DIM],
    thr_enemy: &[i16; THREAT_DIM],
) -> [i16; HEAD_IN] {
    let mut input = [0i16; HEAD_IN];
    let mut idx = 0;

    // [rel_friendly, thr_friendly, rel_enemy, thr_enemy]
    for &v in rel_friendly {
        input[idx] = crelu_i16(v);
        idx += 1;
    }
    for &v in thr_friendly {
        input[idx] = crelu_i16(v);
        idx += 1;
    }
    for &v in rel_enemy {
        input[idx] = crelu_i16(v);
        idx += 1;
    }
    for &v in thr_enemy {
        input[idx] = crelu_i16(v);
        idx += 1;
    }

    input
}

/// Compute dot product of i8 weights and i16 inputs using chunks of 16.
/// This manual unrolling helps the compiler generate SIMD instructions (e.g. AVX2/NEON).
#[inline(always)]
fn dot_product_i8_i16_chunked(weights: &[i8], input: &[i16]) -> i32 {
    let mut sum = 0;
    let mut w_iter = weights.chunks_exact(16);
    let mut i_iter = input.chunks_exact(16);

    for (w, i) in w_iter.by_ref().zip(i_iter.by_ref()) {
        sum += (w[0] as i32 * i[0] as i32)
            + (w[1] as i32 * i[1] as i32)
            + (w[2] as i32 * i[2] as i32)
            + (w[3] as i32 * i[3] as i32)
            + (w[4] as i32 * i[4] as i32)
            + (w[5] as i32 * i[5] as i32)
            + (w[6] as i32 * i[6] as i32)
            + (w[7] as i32 * i[7] as i32)
            + (w[8] as i32 * i[8] as i32)
            + (w[9] as i32 * i[9] as i32)
            + (w[10] as i32 * i[10] as i32)
            + (w[11] as i32 * i[11] as i32)
            + (w[12] as i32 * i[12] as i32)
            + (w[13] as i32 * i[13] as i32)
            + (w[14] as i32 * i[14] as i32)
            + (w[15] as i32 * i[15] as i32);
    }

    // Remainder should be zero for 640/32 dims, but handle just in case or for safety
    for (w, i) in w_iter.remainder().iter().zip(i_iter.remainder()) {
        sum += (*w as i32) * (*i as i32);
    }

    sum
}

/// Compute dot product of i8 weights and i32 inputs (where inputs are actually small positive integers).
#[inline(always)]
fn dot_product_i8_i32_chunked(weights: &[i8], input: &[i32]) -> i32 {
    let mut sum = 0;
    let mut w_iter = weights.chunks_exact(16);
    let mut i_iter = input.chunks_exact(16);

    for (w, i) in w_iter.by_ref().zip(i_iter.by_ref()) {
        sum += (w[0] as i32 * i[0])
            + (w[1] as i32 * i[1])
            + (w[2] as i32 * i[2])
            + (w[3] as i32 * i[3])
            + (w[4] as i32 * i[4])
            + (w[5] as i32 * i[5])
            + (w[6] as i32 * i[6])
            + (w[7] as i32 * i[7])
            + (w[8] as i32 * i[8])
            + (w[9] as i32 * i[9])
            + (w[10] as i32 * i[10])
            + (w[11] as i32 * i[11])
            + (w[12] as i32 * i[12])
            + (w[13] as i32 * i[13])
            + (w[14] as i32 * i[14])
            + (w[15] as i32 * i[15]);
    }

    for (w, i) in w_iter.remainder().iter().zip(i_iter.remainder()) {
        sum += (*w as i32) * (*i);
    }

    sum
}

/// Forward pass through the MLP head.
/// Uses i32 accumulation for precision.
fn forward_head(weights: &NnueWeights, input: &[i16; HEAD_IN]) -> i32 {
    // Layer 1: 640 -> 32
    let mut h1 = [0i32; H1];

    // Process rows of FC1
    // We assume HEAD_IN (640) is a multiple of 16 (it is).
    for (i, val) in h1.iter_mut().enumerate() {
        let row_offset = i * HEAD_IN;
        let row_weights = &weights.fc1_weight[row_offset..row_offset + HEAD_IN];

        let sum = weights.fc1_bias[i] + dot_product_i8_i16_chunked(row_weights, input);

        *val = crelu_i32(sum >> 6); // Scale down and apply activation
    }

    // Layer 2: 32 -> 32
    let mut h2 = [0i32; H2];
    for (i, val) in h2.iter_mut().enumerate() {
        let row_offset = i * H1;
        let row_weights = &weights.fc2_weight[row_offset..row_offset + H1];

        let sum = weights.fc2_bias[i] + dot_product_i8_i32_chunked(row_weights, &h1);

        *val = crelu_i32(sum >> 6);
    }

    // Layer 3: 32 -> 1
    // Flattened dot product
    weights.fc3_bias + dot_product_i8_i32_chunked(&weights.fc3_weight, &h2)
}

/// Main NNUE evaluation function.
///
/// Returns score in centipawns from side-to-move's perspective.
/// If NNUE weights are not available, returns 0.
pub fn evaluate(gs: &GameState) -> i32 {
    let weights = match NNUE_WEIGHTS.as_ref() {
        Some(w) => w,
        None => return 0,
    };

    // Build state from scratch (for now; incremental updates come later)
    let state = NnueState::from_position(gs);

    // Compute threat streams on the fly
    let (thr_white_feats, thr_black_feats) = build_threat_active_lists(gs);
    let thr_white = compute_threat_stream(weights, &thr_white_feats);
    let thr_black = compute_threat_stream(weights, &thr_black_feats);

    // Swap perspectives based on side to move
    let (rel_friendly, thr_friendly, rel_enemy, thr_enemy) = if gs.turn == PlayerColor::White {
        (
            &state.rel_acc_white,
            &thr_white,
            &state.rel_acc_black,
            &thr_black,
        )
    } else {
        (
            &state.rel_acc_black,
            &thr_black,
            &state.rel_acc_white,
            &thr_white,
        )
    };

    // Build head input and forward
    let head_input = build_head_input(rel_friendly, thr_friendly, rel_enemy, thr_enemy);
    let raw_output = forward_head(weights, &head_input);

    // Scale output to centipawns
    // The quantization scales are applied to convert back to float-like range
    let scale = weights.scales.s_out * weights.scales.s_h2 * weights.scales.s_h1;

    // Network output is already from STM perspective (friendly/enemy swap above)
    ((raw_output as f32) * scale) as i32
}

/// Evaluate with explicit state (for incremental updates).
#[allow(dead_code)]
pub fn evaluate_with_state(gs: &GameState, state: &NnueState) -> i32 {
    let weights = match NNUE_WEIGHTS.as_ref() {
        Some(w) => w,
        None => return 0,
    };

    let (thr_white_feats, thr_black_feats) = build_threat_active_lists(gs);
    let thr_white = compute_threat_stream(weights, &thr_white_feats);
    let thr_black = compute_threat_stream(weights, &thr_black_feats);

    let (rel_friendly, thr_friendly, rel_enemy, thr_enemy) = if gs.turn == PlayerColor::White {
        (
            &state.rel_acc_white,
            &thr_white,
            &state.rel_acc_black,
            &thr_black,
        )
    } else {
        (
            &state.rel_acc_black,
            &thr_black,
            &state.rel_acc_white,
            &thr_white,
        )
    };

    let head_input = build_head_input(rel_friendly, thr_friendly, rel_enemy, thr_enemy);
    let raw_output = forward_head(weights, &head_input);

    let scale = weights.scales.s_out * weights.scales.s_h2 * weights.scales.s_h1;

    // Network output is already from STM perspective (friendly/enemy swap above)
    ((raw_output as f32) * scale) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crelu_i16() {
        assert_eq!(crelu_i16(-100), 0);
        assert_eq!(crelu_i16(0), 0);
        assert_eq!(crelu_i16(50), 50);
        assert_eq!(crelu_i16(127), 127);
        assert_eq!(crelu_i16(200), 127);
    }

    #[test]
    fn test_crelu_i32() {
        assert_eq!(crelu_i32(-100), 0);
        assert_eq!(crelu_i32(0), 0);
        assert_eq!(crelu_i32(50), 50);
        assert_eq!(crelu_i32(127), 127);
        assert_eq!(crelu_i32(200), 127);
    }

    #[test]
    fn test_build_head_input() {
        let rel_friendly = [10; RELKP_DIM];
        let thr_friendly = [20; THREAT_DIM];
        let rel_enemy = [30; RELKP_DIM];
        let thr_enemy = [40; THREAT_DIM];

        let input = build_head_input(&rel_friendly, &thr_friendly, &rel_enemy, &thr_enemy);

        // Check a few indices
        assert_eq!(input[0], 10);
        assert_eq!(input[RELKP_DIM], 20); // Start of thr_friendly
        assert_eq!(input[RELKP_DIM + THREAT_DIM], 30); // Start of rel_enemy
        assert_eq!(input[RELKP_DIM * 2 + THREAT_DIM], 40); // Start of thr_enemy
    }

    #[test]
    fn test_dot_product_chunked() {
        // Test i8 * i16
        let weights_i8: Vec<i8> = (0..32).map(|i| (i % 10) as i8).collect();
        let input_i16: Vec<i16> = (0..32).map(|i| (i % 5) as i16).collect();

        let expected: i32 = weights_i8
            .iter()
            .zip(input_i16.iter())
            .map(|(&w, &i)| (w as i32) * (i as i32))
            .sum();

        let actual = dot_product_i8_i16_chunked(&weights_i8, &input_i16);
        assert_eq!(actual, expected, "i8*i16 chunked dot product failed");

        // Test i8 * i32
        let input_i32: Vec<i32> = (0..32).map(|i| i * 2).collect();
        let expected_i32: i32 = weights_i8
            .iter()
            .zip(input_i32.iter())
            .map(|(&w, &i)| (w as i32) * i)
            .sum();

        let actual_i32 = dot_product_i8_i32_chunked(&weights_i8, &input_i32);
        assert_eq!(
            actual_i32, expected_i32,
            "i8*i32 chunked dot product failed"
        );
    }
}
