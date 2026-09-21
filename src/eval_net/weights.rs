//! Quantized eval-net weights, embedded at compile time. An empty or
//! schema-mismatched blob simply disables the net; the engine never fails open.

use once_cell::sync::Lazy;
use std::io::{Cursor, Read};

const MAGIC: &[u8; 8] = b"AEVNET01";

pub struct EvalNetWeights {
    pub n_in: usize,
    pub h1: usize,
    pub h2: usize,
    /// Right-shifts applied to the layer-1/2 accumulators before the CReLU.
    pub s1: u32,
    pub s2: u32,
    /// Converts the raw integer output to centipawns.
    pub out_scale: f32,
    pub l1_w: Box<[i8]>,
    pub l1_b: Box<[i32]>,
    pub l2_w: Box<[i8]>,
    pub l2_b: Box<[i32]>,
    pub l3_w: Box<[i8]>,
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

fn read_i8s(c: &mut Cursor<&[u8]>, n: usize) -> Result<Box<[i8]>, &'static str> {
    let mut buf = vec![0u8; n];
    c.read_exact(&mut buf).map_err(|_| "short read (i8[])")?;
    Ok(buf.into_iter().map(|b| b as i8).collect())
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
        let _version = read_u32(&mut c)?;
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

        Ok(EvalNetWeights {
            n_in,
            h1,
            h2,
            s1,
            s2,
            out_scale,
            l1_w: read_i8s(&mut c, h1 * n_in)?,
            l1_b: read_i32s(&mut c, h1)?,
            l2_w: read_i8s(&mut c, h2 * h1)?,
            l2_b: read_i32s(&mut c, h2)?,
            l3_w: read_i8s(&mut c, h2)?,
            l3_b: read_i32s(&mut c, 1)?[0],
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
