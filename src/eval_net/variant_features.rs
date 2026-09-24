//! Inputs for the specialized evaluators' nets. Each evaluator writes its own terms
//! into a `VariantSink` during its normal pass, so its net costs no second eval.

/// Widest variant layout; inference pads every input vector to this.
pub const MAX_VARIANT_FEATURES: usize = 64;

pub trait VariantSink {
    /// False for the plain eval, so every `set` call compiles away.
    const ON: bool;
    fn set(&mut self, col: usize, v: i16);
    fn set_phase(&mut self, phase: i32);
}

pub struct NoSink;

impl VariantSink for NoSink {
    const ON: bool = false;
    #[inline(always)]
    fn set(&mut self, _: usize, _: i16) {}
    #[inline(always)]
    fn set_phase(&mut self, _: i32) {}
}

pub struct VariantFeatures {
    pub x: [i16; MAX_VARIANT_FEATURES],
    /// The evaluator's own game phase, stored in each exported record.
    pub phase: i32,
}

impl Default for VariantFeatures {
    fn default() -> Self {
        VariantFeatures { x: [0; MAX_VARIANT_FEATURES], phase: 0 }
    }
}

impl VariantSink for VariantFeatures {
    const ON: bool = true;
    #[inline(always)]
    fn set(&mut self, col: usize, v: i16) {
        self.x[col] = v;
    }
    #[inline(always)]
    fn set_phase(&mut self, phase: i32) {
        self.phase = phase;
    }
}

/// Column layout: `fixed` columns that never change under the colour swap, then
/// `neg` White-ahead singles, then (White, Black) pairs to the end.
pub struct VariantLayout {
    pub names: &'static [&'static str],
    pub fixed: usize,
    pub neg: usize,
}

impl VariantLayout {
    pub const fn len(&self) -> usize {
        self.names.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Re-encodes a vector as (side to move, opponent), matching `to_perspective` in
    /// `nnue/train_eval_net.py` for a `--layout fixed,neg` net.
    pub fn to_perspective(&self, x: &mut [i16], black_to_move: bool) {
        if !black_to_move {
            return;
        }
        for v in &mut x[self.fixed..self.fixed + self.neg] {
            *v = -*v;
        }
        let mut c = self.fixed + self.neg;
        while c + 1 < self.len() {
            x.swap(c, c + 1);
            c += 2;
        }
    }

    /// FNV-1a over the column names and the split, sealed into the weights file.
    pub fn schema_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        };
        for name in self.names {
            eat(name.as_bytes());
        }
        eat(&(self.len() as u32).to_le_bytes());
        eat(&(self.fixed as u32).to_le_bytes());
        eat(&(self.neg as u32).to_le_bytes());
        h
    }
}

/// Centipawn term, quartered like the generic layout's.
#[inline(always)]
pub fn cp(v: i32) -> i16 {
    (v / 4).clamp(-2047, 2047) as i16
}

#[inline(always)]
pub fn ct(v: i32) -> i16 {
    v.clamp(0, 255) as i16
}
