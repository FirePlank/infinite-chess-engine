//! NNUE Weights Loading
//!
//! Loads quantized weights from the embedded binary file.
//! For WASM compatibility, weights are embedded at compile time.

use once_cell::sync::Lazy;
use std::io::{Cursor, Read};

/// Magic bytes for file identification
const MAGIC: &[u8; 8] = b"INNUEW1\0";

/// Quantization scales for inference
#[derive(Debug, Clone, Copy)]
pub struct NnueScales {
    pub s_rel: f32,
    pub s_thr: f32,
    pub s_h1: f32,
    pub s_h2: f32,
    pub s_out: f32,
}

/// NNUE weights structure holding all quantized parameters
pub struct NnueWeights {
    // Feature transformer dimensions
    pub rel_features: u32,
    pub rel_dim: u32,
    pub thr_features: u32,
    pub thr_dim: u32,
    pub head_in: u32,
    pub h1: u32,
    pub h2: u32,

    // Quantization scales
    pub scales: NnueScales,

    // RelKP transformer: i16 quantized
    pub rel_embed: Box<[i16]>, // [rel_features * rel_dim]
    pub rel_bias: Box<[i16]>,  // [rel_dim]

    // Threat transformer: i16 quantized
    pub thr_embed: Box<[i16]>, // [thr_features * thr_dim]
    pub thr_bias: Box<[i16]>,  // [thr_dim]

    // Head MLP: i8/i32 quantized
    pub fc1_weight: Box<[i8]>, // [h1 * head_in] - stored as [h1][head_in]
    pub fc1_bias: Box<[i32]>,  // [h1]
    pub fc2_weight: Box<[i8]>, // [h2 * h1]
    pub fc2_bias: Box<[i32]>,  // [h2]
    pub fc3_weight: Box<[i8]>, // [h2] - output is scalar
    pub fc3_bias: i32,
}

impl NnueWeights {
    /// Load weights from a byte slice (for embedded data).
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = Cursor::new(data);

        // Read magic
        let mut magic = [0u8; 8];
        cursor
            .read_exact(&mut magic)
            .map_err(|_| "Failed to read magic")?;
        if &magic != MAGIC {
            return Err("Invalid magic bytes");
        }

        // Read dimensions (7 × u32)
        let rel_features = read_u32(&mut cursor)?;
        let rel_dim = read_u32(&mut cursor)?;
        let thr_features = read_u32(&mut cursor)?;
        let thr_dim = read_u32(&mut cursor)?;
        let head_in = read_u32(&mut cursor)?;
        let h1 = read_u32(&mut cursor)?;
        let h2 = read_u32(&mut cursor)?;

        // Read scales (5 × f32)
        let scales = NnueScales {
            s_rel: read_f32(&mut cursor)?,
            s_thr: read_f32(&mut cursor)?,
            s_h1: read_f32(&mut cursor)?,
            s_h2: read_f32(&mut cursor)?,
            s_out: read_f32(&mut cursor)?,
        };

        // Read weight tensors
        let rel_embed = read_i16_array(&mut cursor, (rel_features * rel_dim) as usize)?;
        let rel_bias = read_i16_array(&mut cursor, rel_dim as usize)?;

        let thr_embed = read_i16_array(&mut cursor, (thr_features * thr_dim) as usize)?;
        let thr_bias = read_i16_array(&mut cursor, thr_dim as usize)?;

        let fc1_weight = read_i8_array(&mut cursor, (h1 * head_in) as usize)?;
        let fc1_bias = read_i32_array(&mut cursor, h1 as usize)?;

        let fc2_weight = read_i8_array(&mut cursor, (h2 * h1) as usize)?;
        let fc2_bias = read_i32_array(&mut cursor, h2 as usize)?;

        let fc3_weight = read_i8_array(&mut cursor, h2 as usize)?;
        let fc3_bias = read_i32_array(&mut cursor, 1)?[0];

        Ok(NnueWeights {
            rel_features,
            rel_dim,
            thr_features,
            thr_dim,
            head_in,
            h1,
            h2,
            scales,
            rel_embed,
            rel_bias,
            thr_embed,
            thr_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        })
    }
}

// Helper functions for reading binary data
fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32, &'static str> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| "Failed to read u32")?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32, &'static str> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| "Failed to read f32")?;
    Ok(f32::from_le_bytes(buf))
}

fn read_i16_array(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Box<[i16]>, &'static str> {
    let mut buf = vec![0u8; count * 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| "Failed to read i16 array")?;

    let mut result = Vec::with_capacity(count);
    for chunk in buf.chunks_exact(2) {
        result.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(result.into_boxed_slice())
}

fn read_i8_array(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Box<[i8]>, &'static str> {
    let mut buf = vec![0u8; count];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| "Failed to read i8 array")?;

    let result: Vec<i8> = buf.into_iter().map(|b| b as i8).collect();
    Ok(result.into_boxed_slice())
}

fn read_i32_array(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Box<[i32]>, &'static str> {
    let mut buf = vec![0u8; count * 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|_| "Failed to read i32 array")?;

    let mut result = Vec::with_capacity(count);
    for chunk in buf.chunks_exact(4) {
        result.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(result.into_boxed_slice())
}

// ============================================================================
// EMBEDDED WEIGHT
// ============================================================================

/// Embedded NNUE weights binary.
/// This file should be generated after training.
/// Place `innue.bin` in the `src/nnue/` directory.
#[cfg(feature = "nnue")]
static NNUE_BYTES: &[u8] = include_bytes!("innue.bin");

/// Placeholder for when NNUE feature is disabled or weights not available
#[cfg(not(feature = "nnue"))]
static NNUE_BYTES: &[u8] = &[];

/// Global NNUE weights, lazily loaded at first use.
pub static NNUE_WEIGHTS: Lazy<Option<NnueWeights>> = Lazy::new(|| {
    if NNUE_BYTES.is_empty() {
        None
    } else {
        match NnueWeights::from_bytes(NNUE_BYTES) {
            Ok(weights) => Some(weights),
            Err(e) => {
                let _ = e; // Suppress warning on WASM where e isn't used
                #[cfg(not(target_arch = "wasm32"))]
                eprintln!("Failed to load NNUE weights: {}", e);
                None
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_lazy_load() {
        // Just check it doesn't panic
        let _ = NNUE_WEIGHTS.is_some();
    }
}
