//! Magic Bitboards for Sliders on 8×8 Tiles
//!
//! This module implements fast magic bitboards for O(1) slider attack
//! generation within 8×8 tiles for infinite chess.
//!
//! ## Key Difference from Standard Chess
//! Standard chess excludes edge squares from masks because edge blockers
//! can't affect attacks (board boundary stops the ray anyway).
//! For **infinite chess**, edges ARE included because rays continue to the next tile.

#![allow(static_mut_refs)]

use std::sync::Once;

// ============================================================================
// Public API - Maximum performance with inline hints
// ============================================================================

/// Get rook attacks for a square within a single tile.
/// Returns a bitboard of squares the rook can attack/move to within the tile.
#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    debug_assert!(sq < 64);
    unsafe {
        let mask = ROOK_MASKS[sq];
        let shift = ROOK_SHIFTS[sq] as u32;
        let idx = (((occ & mask).wrapping_mul(ROOK_MAGICS[sq])) >> shift) as usize;
        *ROOK_ATTACKS.get_unchecked(ROOK_OFFSETS[sq] + idx)
    }
}

/// Get bishop attacks for a square within a single tile.
/// Returns a bitboard of squares the bishop can attack/move to within the tile.
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    debug_assert!(sq < 64);
    unsafe {
        let mask = BISHOP_MASKS[sq];
        let shift = BISHOP_SHIFTS[sq] as u32;
        let idx = (((occ & mask).wrapping_mul(BISHOP_MAGICS[sq])) >> shift) as usize;
        *BISHOP_ATTACKS.get_unchecked(BISHOP_OFFSETS[sq] + idx)
    }
}

/// Get queen attacks (union of bishop and rook attacks)
#[inline(always)]
pub fn queen_attacks(sq: usize, occ: u64) -> u64 {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Initialize magic bitboards (thread-safe, called automatically or at startup)
#[inline]
pub fn init() {
    INIT.call_once(|| unsafe { init_tables() });
}

/// Legacy alias
pub fn init_magic_bitboards() {
    init();
}

// ============================================================================
// Pre-generated Edge-Inclusive Magic Numbers
// ============================================================================

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x2084001004080400, 0x82000A0040102080, 0x020004200A001040, 0x0200020010080420,
    0x0200010428B00200, 0x0200028200041908, 0x0200005400A10200, 0x0004082100408010,
    0x0005040080114000, 0x0002001020408200, 0x0002000820104200, 0x0000100010530620,
    0x0812000102840200, 0x0000040081004200, 0x000C000082011048, 0x0200080A21004080,
    0x0000280800308700, 0x0000420010220080, 0x0000420008201200, 0x8000160006201200,
    0x0000420001840E00, 0xC800020004008148, 0x00000400010200C8, 0x0000001110841440,
    0x0001004A04020800, 0x0000220200104080, 0x00001022000A0040, 0x0000201600061200,
    0x0000040200010890, 0x00004C0200010298, 0x2000010400008208, 0x0000002404001204,
    0x00000A1425040200, 0x0000204082001200, 0x0000102202000840, 0x8000522016000600,
    0x0000010A02000210, 0x0000040082000108, 0x8800008204000128, 0x4800002140080410,
    0x000A504023808000, 0x8000308040020400, 0x40001D3020000A00, 0xC008420100200080,
    0x4000082040100080, 0x2000040081020008, 0x000004B902040008, 0xA000004500003080,
    0x0002422200910600, 0x8000408010220200, 0x0001104108220200, 0x00000A000C200600,
    0x0000040200810A00, 0x3000408409080600, 0x1000220140880400, 0x0000072101804A00,
    0x0000620100415282, 0x8000402082000812, 0x400020400E06000A, 0x0000040312002002,
    0x000000810402000A, 0x00000104080040A2, 0x200000A042010804, 0x0000002080401102,
];

#[rustfmt::skip]
const ROOK_SHIFTS: [u8; 64] = [
    51, 52, 52, 52, 52, 52, 52, 51,
    51, 53, 53, 51, 51, 51, 52, 52,
    51, 53, 53, 53, 52, 52, 53, 51,
    51, 53, 53, 53, 52, 52, 53, 51,
    51, 53, 53, 53, 52, 52, 53, 51,
    52, 52, 52, 51, 52, 52, 53, 51,
    53, 53, 53, 53, 52, 52, 53, 52,
    52, 52, 52, 51, 51, 51, 52, 51,
];

#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x08581101080A1080, 0x0102100400808004, 0x012101310200000C, 0x00080A0020000013,
    0x0642021000009000, 0x8182020220004840, 0x0002010120100008, 0x484A04420084A000,
    0x0004041022120400, 0x0080020A0A0C0100, 0x0001100142042000, 0x0000080A00200008,
    0x0004442420000100, 0x1000220202200484, 0x0001040441080800, 0x3010090258040400,
    0x0040202104041080, 0x6002000404040404, 0x0001008800440080, 0x0008040082004000,
    0x0004001080A00000, 0x0000200202012000, 0x0002012C00820880, 0x000020C200840400,
    0x2010400808080140, 0x0008021004100200, 0x0000280004004404, 0x0000808018020002,
    0x0000848024002000, 0xC012008064100080, 0x200904020A022100, 0x2024022080420200,
    0x0010900500080800, 0x4008040204040800, 0x9004004400080020, 0x0000080800A20A00,
    0x0040002020020080, 0x0008020020041000, 0x01020A0201040080, 0x0004090454012400,
    0x2000820841102000, 0x8004020802000400, 0x0000220022005000, 0x0120004208004080,
    0x0000880100400400, 0x000AA00401202100, 0x00080200D4000200, 0x0002040420850420,
    0x1000451010B00400, 0x0400840108024100, 0x1000011401040000, 0x0002240042020000,
    0x0008001002088000, 0x0000C05002008A00, 0x0088028418120048, 0x80CD080600420400,
    0x2004804400A00800, 0x0600020500880400, 0x0100080884008814, 0x108800000084040A,
    0x0000030030021600, 0x040102C082040108, 0x0900624404088401, 0x0040080901020114,
];

#[rustfmt::skip]
const BISHOP_SHIFTS: [u8; 64] = [
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58,
];

// ============================================================================
// Static Tables
// ============================================================================

static INIT: Once = Once::new();

// Edge-inclusive magics need larger tables than standard chess
// Rook: max shift=51 -> 2^13=8192 per square, 64 squares -> 524288 max
// Bishop: max shift=55 -> 2^9=512 per square, 64 squares -> 32768 max
const ROOK_TABLE_SIZE: usize = 524288;
const BISHOP_TABLE_SIZE: usize = 32768;

static mut ROOK_ATTACKS: [u64; ROOK_TABLE_SIZE] = [0; ROOK_TABLE_SIZE];
static mut BISHOP_ATTACKS: [u64; BISHOP_TABLE_SIZE] = [0; BISHOP_TABLE_SIZE];
static mut ROOK_OFFSETS: [usize; 64] = [0; 64];
static mut BISHOP_OFFSETS: [usize; 64] = [0; 64];
static mut ROOK_MASKS: [u64; 64] = [0; 64];
static mut BISHOP_MASKS: [u64; 64] = [0; 64];

// ============================================================================
// Initialization
// ============================================================================

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn init_tables() {
    // Generate masks
    for sq in 0..64 {
        ROOK_MASKS[sq] = gen_rook_mask(sq);
        BISHOP_MASKS[sq] = gen_bishop_mask(sq);
    }

    // Initialize rook attack tables
    let mut rook_offset = 0usize;
    for sq in 0..64 {
        ROOK_OFFSETS[sq] = rook_offset;
        let mask = ROOK_MASKS[sq];
        let shift = ROOK_SHIFTS[sq] as u32;
        let magic = ROOK_MAGICS[sq];
        let table_size = 1usize << (64 - shift);

        let mut occ = 0u64;
        loop {
            let idx = (((occ & mask).wrapping_mul(magic)) >> shift) as usize;
            ROOK_ATTACKS[rook_offset + idx] = gen_rook_attacks(sq, occ);
            occ = occ.wrapping_sub(mask) & mask;
            if occ == 0 {
                break;
            }
        }
        rook_offset += table_size;
    }

    // Initialize bishop attack tables
    let mut bishop_offset = 0usize;
    for sq in 0..64 {
        BISHOP_OFFSETS[sq] = bishop_offset;
        let mask = BISHOP_MASKS[sq];
        let shift = BISHOP_SHIFTS[sq] as u32;
        let magic = BISHOP_MAGICS[sq];
        let table_size = 1usize << (64 - shift);

        let mut occ = 0u64;
        loop {
            let idx = (((occ & mask).wrapping_mul(magic)) >> shift) as usize;
            BISHOP_ATTACKS[bishop_offset + idx] = gen_bishop_attacks(sq, occ);
            occ = occ.wrapping_sub(mask) & mask;
            if occ == 0 {
                break;
            }
        }
        bishop_offset += table_size;
    }
}

#[inline(always)]
const fn bit(sq: usize) -> u64 {
    1u64 << sq
}

/// Generate rook mask (EDGE-INCLUSIVE for infinite chess)
fn gen_rook_mask(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;
    for rr in (r + 1)..=7 {
        m |= bit((rr as usize) * 8 + f as usize);
    }
    for rr in 0..r {
        m |= bit((rr as usize) * 8 + f as usize);
    }
    for ff in (f + 1)..=7 {
        m |= bit((r as usize) * 8 + ff as usize);
    }
    for ff in 0..f {
        m |= bit((r as usize) * 8 + ff as usize);
    }
    m
}

/// Generate bishop mask (EDGE-INCLUSIVE for infinite chess)
fn gen_bishop_mask(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;
    let mut rr = r + 1;
    let mut ff = f + 1;
    while rr <= 7 && ff <= 7 {
        m |= bit((rr as usize) * 8 + ff as usize);
        rr += 1;
        ff += 1;
    }
    rr = r + 1;
    ff = f - 1;
    while rr <= 7 && ff >= 0 {
        m |= bit((rr as usize) * 8 + ff as usize);
        rr += 1;
        ff -= 1;
    }
    rr = r - 1;
    ff = f + 1;
    while rr >= 0 && ff <= 7 {
        m |= bit((rr as usize) * 8 + ff as usize);
        rr -= 1;
        ff += 1;
    }
    rr = r - 1;
    ff = f - 1;
    while rr >= 0 && ff >= 0 {
        m |= bit((rr as usize) * 8 + ff as usize);
        rr -= 1;
        ff -= 1;
    }
    m
}

/// Generate rook attacks given occupancy
fn gen_rook_attacks(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;
    for rr in (r + 1)..=7 {
        let s = (rr as usize) * 8 + f as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
    }
    for rr in (0..r).rev() {
        let s = (rr as usize) * 8 + f as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
    }
    for ff in (f + 1)..=7 {
        let s = (r as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
    }
    for ff in (0..f).rev() {
        let s = (r as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
    }
    a
}

/// Generate bishop attacks given occupancy
fn gen_bishop_attacks(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;
    let mut rr = r + 1;
    let mut ff = f + 1;
    while rr <= 7 && ff <= 7 {
        let s = (rr as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
        rr += 1;
        ff += 1;
    }
    rr = r + 1;
    ff = f - 1;
    while rr <= 7 && ff >= 0 {
        let s = (rr as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
        rr += 1;
        ff -= 1;
    }
    rr = r - 1;
    ff = f + 1;
    while rr >= 0 && ff <= 7 {
        let s = (rr as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
        rr -= 1;
        ff += 1;
    }
    rr = r - 1;
    ff = f - 1;
    while rr >= 0 && ff >= 0 {
        let s = (rr as usize) * 8 + ff as usize;
        a |= bit(s);
        if occ & bit(s) != 0 {
            break;
        }
        rr -= 1;
        ff -= 1;
    }
    a
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
        init(); // Safe to call multiple times
    }

    #[test]
    fn test_rook_attacks_center() {
        init();
        let attacks = rook_attacks(28, 0); // e4 empty board
        assert_eq!(attacks.count_ones(), 14);
    }

    #[test]
    fn test_rook_attacks_with_blocker() {
        init();
        let occ = 1u64 << 44; // blocker at e6
        let attacks = rook_attacks(28, occ);
        assert!(attacks & (1u64 << 36) != 0); // e5
        assert!(attacks & (1u64 << 44) != 0); // e6 (blocker)
        assert!(attacks & (1u64 << 52) == 0); // e7 blocked
    }

    #[test]
    fn test_bishop_attacks_center() {
        init();
        let attacks = bishop_attacks(27, 0); // d4
        assert_eq!(attacks.count_ones(), 13);
    }

    #[test]
    fn test_edge_inclusive() {
        init();
        // Rook at a4 should attack h4 (edge)
        let attacks = rook_attacks(24, 0);
        assert!(attacks & (1u64 << 31) != 0, "Must attack h4");
        assert!(attacks & (1u64 << 0) != 0, "Must attack a1");
        assert!(attacks & (1u64 << 56) != 0, "Must attack a8");
    }

    #[test]
    fn test_queen_attacks() {
        init();
        let sq = 27;
        assert_eq!(
            queen_attacks(sq, 0),
            rook_attacks(sq, 0) | bishop_attacks(sq, 0)
        );
    }
}
