use crate::board::{PieceType, PlayerColor};

/// Number of piece types (used for indexing into piece keys)
const NUM_PIECE_TYPES: usize = 22;
const NUM_COLORS: usize = 3; // White, Black, Neutral

/// Pre-computed random keys for piece-type combinations (not position-dependent)
/// We mix these with position hashes at runtime
static PIECE_KEYS: [[u64; NUM_COLORS]; NUM_PIECE_TYPES] = {
    // Use a simple PRNG to generate constants at compile time
    const fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    let mut keys = [[0u64; NUM_COLORS]; NUM_PIECE_TYPES];
    let mut seed = 0x123456789ABCDEF0u64;

    let mut i = 0;
    while i < NUM_PIECE_TYPES {
        let mut j = 0;
        while j < NUM_COLORS {
            seed = splitmix64(seed);
            keys[i][j] = seed;
            j += 1;
        }
        i += 1;
    }
    keys
};

/// Key for side to move (XOR when black to move)
pub const SIDE_KEY: u64 = 0x9E3779B97F4A7C15;

/// Key for en passant file
const EN_PASSANT_KEY_MIXER: u64 = 0xCAFEBABE87654321;

/// Normalize coordinate for hashing (handle infinite board via bucketing)
///
/// This mirrors the old TT behaviour: coordinates within [-BOUND, BOUND]
/// are kept distinct; far-away squares are wrapped into BUCKETS-sized
/// buckets at the edges, preserving some translation invariance.
#[inline(always)]
fn normalize_coord(coord: i64) -> i32 {
    const BOUND: i64 = 150;
    const BUCKETS: i64 = 8;

    if coord.abs() <= BOUND {
        coord as i32
    } else {
        let sign = coord.signum();
        let delta = (coord - sign * BOUND) % BUCKETS;
        (sign * BOUND + delta) as i32
    }
}

/// Hash a coordinate into a u64
/// Uses a fast mixing function on *bucketed* coordinates, preserving
/// the infinite-board semantics while being efficient for incremental use.
#[inline(always)]
pub fn hash_coordinate(x: i64, y: i64) -> u64 {
    let nx = normalize_coord(x) as u64;
    let ny = normalize_coord(y) as u64;

    // Fast mixing - fewer operations, good enough distribution
    let h = nx.wrapping_mul(0x517cc1b727220a95) ^ ny.wrapping_mul(0x9e3779b97f4a7c15);
    h ^ (h >> 32)
}

/// Get the Zobrist key for a piece at a position
#[inline(always)]
pub fn piece_key(piece_type: PieceType, color: PlayerColor, x: i64, y: i64) -> u64 {
    let coord_hash = hash_coordinate(x, y);
    let pk = PIECE_KEYS[piece_type as usize][color as usize];
    coord_hash ^ pk
}

/// Pre-computed keys for effective castling rights
/// Indexed by: [0] = white kingside, [1] = white queenside, [2] = black kingside, [3] = black queenside
const CASTLING_RIGHTS_KEYS: [u64; 4] = [
    0x31D71DCE64B2C310, // White kingside
    0x1A8419B523E6D19D, // White queenside
    0x2E2B87D53B9A1C4F, // Black kingside
    0x7C8F5A0E6D3B2A1F, // Black queenside
];

/// Precomputed table for all 16 possible castling combinations
static CASTLING_COMBINATIONS: [u64; 16] = {
    let mut table = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let mut h = 0u64;
        if i & 1 != 0 {
            h ^= CASTLING_RIGHTS_KEYS[0];
        } // WKS
        if i & 2 != 0 {
            h ^= CASTLING_RIGHTS_KEYS[1];
        } // WQS
        if i & 4 != 0 {
            h ^= CASTLING_RIGHTS_KEYS[2];
        } // BKS
        if i & 8 != 0 {
            h ^= CASTLING_RIGHTS_KEYS[3];
        } // BQS
        table[i] = h;
        i += 1;
    }
    table
};

/// Get the Zobrist key for effective castling rights from a 4-bit bitfield.
/// Bit 0=WKS, 1=WQS, 2=BKS, 3=BQS.
#[inline(always)]
pub fn castling_rights_key_from_bitfield(bits: u8) -> u64 {
    CASTLING_COMBINATIONS[(bits & 0xF) as usize]
}

/// Get the Zobrist key for effective castling rights.
/// This hashes the ABILITY to castle (king + partner both have rights), not individual piece rights.
/// Much more efficient than hashing all individual special rights.
#[inline(always)]
pub fn castling_rights_key(
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
) -> u64 {
    let mut bits = 0u8;
    if white_kingside {
        bits |= 1;
    }
    if white_queenside {
        bits |= 2;
    }
    if black_kingside {
        bits |= 4;
    }
    if black_queenside {
        bits |= 8;
    }
    castling_rights_key_from_bitfield(bits)
}

/// Get the key for en passant square
#[inline(always)]
pub fn en_passant_key(x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ EN_PASSANT_KEY_MIXER
}

/// Key for pawn double-push special rights (not castling)
const PAWN_SPECIAL_RIGHT_MIXER: u64 = 0x5A5A5A5A3C3C3C3C;

/// Get the key for a pawn's double-push special right at a coordinate.
/// Used for hashing pawn special rights separately from castling rights.
#[inline(always)]
pub fn pawn_special_right_key(x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ PAWN_SPECIAL_RIGHT_MIXER
}

/// Key for pawn structure hash (used by correction history).
/// Includes only pawn positions, helps CoaIP variants.
const PAWN_KEY_MIXER: u64 = 0xABCDEF0123456789;

#[inline(always)]
pub fn pawn_key(color: PlayerColor, x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ PAWN_KEY_MIXER ^ (color as u64 * 0x9E3779B97F4A7C15)
}

/// Key for material configuration hash (used by correction history).
/// Based on piece type and color counts.
const MATERIAL_KEY_MIXER: u64 = 0xFEDCBA9876543210;

#[inline(always)]
pub fn material_key(piece_type: PieceType, color: PlayerColor) -> u64 {
    // Use piece type as a simple hash - no position dependency
    let pt = piece_type as u64;
    let c = color as u64;
    MATERIAL_KEY_MIXER.wrapping_mul(pt.wrapping_add(1)) ^ (c * 0x517CC1B727220A95)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_keys_unique() {
        // Verify piece keys are reasonably unique
        let mut keys = Vec::new();
        for i in 0..NUM_PIECE_TYPES {
            for j in 0..NUM_COLORS {
                keys.push(PIECE_KEYS[i][j]);
            }
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), NUM_PIECE_TYPES * NUM_COLORS);
    }

    #[test]
    fn test_coordinate_hash_different() {
        let h1 = hash_coordinate(1, 1);
        let h2 = hash_coordinate(1, 2);
        let h3 = hash_coordinate(2, 1);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }
}
