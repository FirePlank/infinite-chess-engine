//! Quantized eval-net weights, embedded at compile time. An empty or
//! schema-mismatched blob simply disables the net; the engine never fails open.

use once_cell::sync::Lazy;
use std::io::{Cursor, Read};

const MAGIC: &[u8; 8] = b"AEVNET01";

/// i16 buffer whose payload starts on a 64-byte boundary, so every 32-byte
/// weight load stays inside one cache line.
pub struct AlignedI16 {
    data: Box<[i16]>,
    off: usize,
}

impl AlignedI16 {
    pub fn zeroed(len: usize) -> Self {
        let data: Box<[i16]> = vec![0i16; len + 32].into_boxed_slice();
        let off = (64 - (data.as_ptr() as usize % 64)) % 64 / 2;
        AlignedI16 { data, off }
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[i16] {
        &self.data[self.off..]
    }
    pub fn as_mut_slice(&mut self) -> &mut [i16] {
        &mut self.data[self.off..]
    }
    pub fn from_rows(rows: &[i16], n_in: usize, stride: usize, n_rows: usize) -> Self {
        let mut out = Self::zeroed(stride * n_rows);
        for r in 0..n_rows {
            out.as_mut_slice()[r * stride..r * stride + n_in]
                .copy_from_slice(&rows[r * n_in..(r + 1) * n_in]);
        }
        out
    }
}

/// Row strides are padded to 32 i16 (64 bytes) so aligned rows stay aligned.
pub const fn pad32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

pub struct EvalNetWeights {
    /// Blob version 2+: inputs are re-encoded as (side to move, opponent) and the
    /// output is side-to-move relative.
    pub perspective: bool,
    pub n_in: usize,
    pub h1: usize,
    pub h2: usize,
    /// Padded row strides of `l1_w` (inputs) and `l2_w` (layer-1 outputs).
    pub stride1: usize,
    pub stride2: usize,
    /// Right-shifts applied to the layer-1/2 accumulators before the CReLU.
    pub s1: u32,
    pub s2: u32,
    /// Converts the raw integer output to centipawns.
    pub out_scale: f32,
    /// Weights are i8 on disk but widened to i16 at load: pmaddwd then needs no
    /// sign-extension step, and the 66 KB of layer 1 streams fine from L2.
    pub l1_w: AlignedI16,
    pub l1_b: Box<[i32]>,
    pub l2_w: AlignedI16,
    pub l2_b: Box<[i32]>,
    pub l3_w: AlignedI16,
    pub l3_b: i32,
}

fn read_u32(c: &mut Cursor<&[u8]>) -> Result<u32, &'static str> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|_| "short read (u32)")?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(c: &mut Cursor<&[u8]>) -> Result<u64, &'static str> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b).map_err(|_| "short read (u64)")?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32(c: &mut Cursor<&[u8]>) -> Result<f32, &'static str> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|_| "short read (f32)")?;
    Ok(f32::from_le_bytes(b))
}

fn read_i8s(c: &mut Cursor<&[u8]>, n: usize) -> Result<Box<[i16]>, &'static str> {
    let mut buf = vec![0u8; n];
    c.read_exact(&mut buf).map_err(|_| "short read (i8[])")?;
    Ok(buf.into_iter().map(|b| b as i8 as i16).collect())
}

fn read_i32s(c: &mut Cursor<&[u8]>, n: usize) -> Result<Box<[i32]>, &'static str> {
    let mut buf = vec![0u8; n * 4];
    c.read_exact(&mut buf).map_err(|_| "short read (i32[])")?;
    Ok(buf
        .chunks_exact(4)
        .map(|ch| i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect())
}

impl EvalNetWeights {
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        let mut c = Cursor::new(data);
        let mut magic = [0u8; 8];
        c.read_exact(&mut magic).map_err(|_| "short read (magic)")?;
        if &magic != MAGIC {
            return Err("bad magic");
        }
        let version = read_u32(&mut c)?;
        let n_in = read_u32(&mut c)? as usize;
        let h1 = read_u32(&mut c)? as usize;
        let h2 = read_u32(&mut c)? as usize;
        let s1 = read_u32(&mut c)?;
        let s2 = read_u32(&mut c)?;
        let schema = read_u64(&mut c)?;
        let out_scale = read_f32(&mut c)?;

        if n_in != super::features::NUM_FEATURES {
            return Err("feature count mismatch");
        }
        if schema != super::features::schema_hash() {
            return Err("schema hash mismatch");
        }
        if h1 == 0 || h2 == 0 || !h1.is_multiple_of(16) || !h2.is_multiple_of(16) {
            return Err("bad hidden dims");
        }

        let (stride1, stride2) = (pad32(n_in), pad32(h1));
        let l1 = read_i8s(&mut c, h1 * n_in)?;
        let l1_b = read_i32s(&mut c, h1)?;
        let l2 = read_i8s(&mut c, h2 * h1)?;
        let l2_b = read_i32s(&mut c, h2)?;
        let l3 = read_i8s(&mut c, h2)?;
        let l3_b = read_i32s(&mut c, 1)?[0];
        Ok(EvalNetWeights {
            perspective: version >= 2,
            n_in,
            h1,
            h2,
            stride1,
            stride2,
            s1,
            s2,
            out_scale,
            l1_w: AlignedI16::from_rows(&l1, n_in, stride1, h1),
            l1_b,
            l2_w: AlignedI16::from_rows(&l2, h1, stride2, h2),
            l2_b,
            l3_w: AlignedI16::from_rows(&l3, h2, pad32(h2), 1),
            l3_b,
        })
    }
}

/// Trained weights blob; regenerate with `nnue/export_eval_net.py`. An empty
/// file is a valid "no net yet" state.
static EVAL_NET_BYTES: &[u8] = include_bytes!("eval_net.bin");

pub static EVAL_NET: Lazy<Option<EvalNetWeights>> = Lazy::new(|| {
    if EVAL_NET_BYTES.is_empty() {
        return None;
    }
    match EvalNetWeights::from_bytes(EVAL_NET_BYTES) {
        Ok(w) => Some(w),
        Err(e) => {
            #[cfg(not(target_arch = "wasm32"))]
            eprintln!("eval_net: weights rejected ({e}), net disabled");
            let _ = e;
            None
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic_and_short_data() {
        assert!(EvalNetWeights::from_bytes(b"BADMAGIC").is_err());
        assert!(EvalNetWeights::from_bytes(b"AEVNET01").is_err());
    }

    #[test]
    fn lazy_load_does_not_panic() {
        let _ = EVAL_NET.is_some();
    }
}
