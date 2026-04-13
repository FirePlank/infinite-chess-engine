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

/// Normalize coordinate for hashing
#[inline(always)]
fn normalize_coord(coord: i64) -> u64 {
    ((coord << 1) ^ (coord >> 63)) as u64
}

/// Hash a coordinate into a u64.
/// The +1 offset prevents normalize_coord(0)==0 from causing piece_key to degenerate
/// to the bare PIECE_KEY constant at the origin, which eliminated coordinate separation.
#[inline(always)]
pub fn hash_coordinate(x: i64, y: i64) -> u64 {
    let nx = normalize_coord(x).wrapping_add(1);
    let ny = normalize_coord(y).wrapping_add(1);
    let hx = nx.wrapping_mul(0x9E3779B185EBCA87);
    let hy = ny.wrapping_mul(0xC2B2AE3D27D4EB4F);
    hx ^ hy.rotate_left(32) ^ hx.rotate_left(17).wrapping_add(hy)
}

/// Get the Zobrist key for a piece at a position
#[inline(always)]
pub fn piece_key(piece_type: PieceType, color: PlayerColor, x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ PIECE_KEYS[piece_type as usize][color as usize]
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
    hash_coordinate(x, y) ^ PAWN_KEY_MIXER ^ (color as u64).wrapping_mul(0x9E3779B97F4A7C15)
}

/// Key for material configuration hash (used by correction history).
/// Based on piece type and color counts.
const MATERIAL_KEY_MIXER: u64 = 0xFEDCBA9876543210;

#[inline(always)]
pub fn material_key(piece_type: PieceType, color: PlayerColor) -> u64 {
    let pt = piece_type as u64;
    let c = color as u64;
    let hp = pt.wrapping_add(1).wrapping_mul(0x9E3779B185EBCA87);
    let hc = c.wrapping_add(1).wrapping_mul(0xC2B2AE3D27D4EB4F);
    MATERIAL_KEY_MIXER ^ hp ^ hc.rotate_left(32)
}

// Secondary Zobrist keys for repetition detection (independent seed).
// When both hash_stack and rep_hash_stack match, false positive probability ~2^-128.
static REP_PIECE_KEYS: [[u64; NUM_COLORS]; NUM_PIECE_TYPES] = {
    const fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    let mut keys = [[0u64; NUM_COLORS]; NUM_PIECE_TYPES];
    let mut seed = 0xFEDCBA9876543210u64; // different seed from primary

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

pub const REP_SIDE_KEY: u64 = 0x517CC1B727220A95;

const REP_EN_PASSANT_KEY_MIXER: u64 = 0x3141592653589793;

const REP_PAWN_SPECIAL_RIGHT_MIXER: u64 = 0x2718281828459045;

const REP_CASTLING_RIGHTS_KEYS: [u64; 4] = [
    0xA0B1C2D3E4F50607,
    0x0817263544536271,
    0xF1E2D3C4B5A69788,
    0x89786756453423A1,
];

static REP_CASTLING_COMBINATIONS: [u64; 16] = {
    let mut table = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let mut h = 0u64;
        if i & 1 != 0 { h ^= REP_CASTLING_RIGHTS_KEYS[0]; }
        if i & 2 != 0 { h ^= REP_CASTLING_RIGHTS_KEYS[1]; }
        if i & 4 != 0 { h ^= REP_CASTLING_RIGHTS_KEYS[2]; }
        if i & 8 != 0 { h ^= REP_CASTLING_RIGHTS_KEYS[3]; }
        table[i] = h;
        i += 1;
    }
    table
};

/// Secondary hash for a piece at a position.
#[inline(always)]
pub fn rep_piece_key(piece_type: PieceType, color: PlayerColor, x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ REP_PIECE_KEYS[piece_type as usize][color as usize]
}

/// Secondary hash for en passant.
#[inline(always)]
pub fn rep_en_passant_key(x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ REP_EN_PASSANT_KEY_MIXER
}

/// Secondary hash for pawn double-push right.
#[inline(always)]
pub fn rep_pawn_special_right_key(x: i64, y: i64) -> u64 {
    hash_coordinate(x, y) ^ REP_PAWN_SPECIAL_RIGHT_MIXER
}

/// Secondary hash for castling rights from a 4-bit bitfield.
#[inline(always)]
pub fn rep_castling_rights_key_from_bitfield(bits: u8) -> u64 {
    REP_CASTLING_COMBINATIONS[(bits & 0xF) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_keys_unique() {
        // Verify piece keys are reasonably unique
        let mut keys = Vec::new();
        for row in PIECE_KEYS {
            for key in row {
                keys.push(key);
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

    #[test]
    fn test_hash_coordinate_deterministic() {
        let h1 = hash_coordinate(3, 5);
        let h2 = hash_coordinate(3, 5);
        assert_eq!(h1, h2);

        let h3 = hash_coordinate(-1, 8);
        let h4 = hash_coordinate(-1, 8);
        assert_eq!(h3, h4);
    }

    #[test]
    fn test_hash_coordinate_boundary_values() {
        let h0 = hash_coordinate(0, 0);
        let h_pos = hash_coordinate(7, 7);
        let h_neg = hash_coordinate(-1, -1);

        assert_ne!(h0, 0); // origin must produce a non-zero coordinate hash
        assert_ne!(h0, h_pos);
        assert_ne!(h0, h_neg);
        assert_ne!(h_pos, h_neg);
    }

    #[test]
    fn test_piece_key_different_colors() {
        use crate::board::PlayerColor;

        let pt = PieceType::Pawn;
        let white_key = piece_key(pt, PlayerColor::White, 1, 1);
        let black_key = piece_key(pt, PlayerColor::Black, 1, 1);
        assert_ne!(white_key, black_key);
    }

    #[test]
    fn test_piece_key_different_positions() {
        use crate::board::PlayerColor;

        let pt = PieceType::Knight;
        let key1 = piece_key(pt, PlayerColor::White, 0, 0);
        let key2 = piece_key(pt, PlayerColor::White, 1, 1);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_rep_piece_keys_unique() {
        let mut keys = Vec::new();
        for row in REP_PIECE_KEYS {
            for key in row {
                keys.push(key);
            }
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), NUM_PIECE_TYPES * NUM_COLORS);
    }

    #[test]
    fn test_rep_keys_independent_from_primary() {
        use crate::board::PlayerColor;
        // Verify secondary keys differ from primary keys for the same inputs
        let pt = PieceType::Pawn;
        assert_ne!(
            piece_key(pt, PlayerColor::White, 3, 4),
            rep_piece_key(pt, PlayerColor::White, 3, 4)
        );
        assert_ne!(en_passant_key(3, 5), rep_en_passant_key(3, 5));
        assert_ne!(
            castling_rights_key_from_bitfield(0b0011),
            rep_castling_rights_key_from_bitfield(0b0011)
        );
        assert_ne!(SIDE_KEY, REP_SIDE_KEY);
    }

    #[test]
    fn test_piece_key_different_types() {
        use crate::board::PlayerColor;

        let color = PlayerColor::White;
        let pawn_key = piece_key(PieceType::Pawn, color, 2, 2);
        let knight_key = piece_key(PieceType::Knight, color, 2, 2);
        assert_ne!(pawn_key, knight_key);
    }

    #[test]
    fn test_castling_rights_key_unique_combinations() {
        let mut hashes = vec![];
        for wks in [false, true] {
            for wqs in [false, true] {
                for bks in [false, true] {
                    for bqs in [false, true] {
                        hashes.push(castling_rights_key(wks, wqs, bks, bqs));
                    }
                }
            }
        }

        let mut unique_hashes = hashes.clone();
        unique_hashes.sort();
        unique_hashes.dedup();
        assert_eq!(unique_hashes.len(), 16);
    }

    #[test]
    fn test_castling_rights_key_consistency() {
        let h1 = castling_rights_key(true, false, true, false);
        let h2 = castling_rights_key(true, false, true, false);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_castling_rights_key_from_bitfield_all_combinations() {
        for bits in 0..=15u8 {
            let h = castling_rights_key_from_bitfield(bits);
            // bits=0 produces hash 0 (no castling rights), other combinations produce non-zero
            if bits != 0 {
                assert_ne!(h, 0);
            }
        }
    }

    #[test]
    fn test_castling_rights_key_bitfield_consistency() {
        let h_explicit = castling_rights_key(true, true, false, true);

        let mut bits = 0u8;
        bits |= 1; // white kingside
        bits |= 2; // white queenside
        bits |= 8; // black queenside
        let h_bitfield = castling_rights_key_from_bitfield(bits);

        assert_eq!(h_explicit, h_bitfield);
    }

    #[test]
    fn test_en_passant_key_different_squares() {
        let h1 = en_passant_key(2, 4);
        let h2 = en_passant_key(3, 4);
        let h3 = en_passant_key(2, 5);

        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }

    #[test]
    fn test_en_passant_key_deterministic() {
        let h1 = en_passant_key(4, 5);
        let h2 = en_passant_key(4, 5);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_pawn_special_right_key_different_squares() {
        let h1 = pawn_special_right_key(1, 3);
        let h2 = pawn_special_right_key(1, 4);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_pawn_key_different_colors() {
        use crate::board::PlayerColor;

        let white_key = pawn_key(PlayerColor::White, 3, 3);
        let black_key = pawn_key(PlayerColor::Black, 3, 3);
        assert_ne!(white_key, black_key);
    }

    #[test]
    fn test_pawn_key_different_positions() {
        use crate::board::PlayerColor;

        let color = PlayerColor::White;
        let k1 = pawn_key(color, 2, 3);
        let k2 = pawn_key(color, 2, 4);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_material_key_different_pieces() {
        use crate::board::PlayerColor;

        let color = PlayerColor::White;
        let pawn_key = material_key(PieceType::Pawn, color);
        let knight_key = material_key(PieceType::Knight, color);
        assert_ne!(pawn_key, knight_key);
    }

    #[test]
    fn test_material_key_different_colors() {
        let piece_type = PieceType::Rook;
        let white_key = material_key(piece_type, PlayerColor::White);
        let black_key = material_key(piece_type, PlayerColor::Black);
        assert_ne!(white_key, black_key);
    }

    #[test]
    fn test_side_key_nonzero() {
        assert_ne!(SIDE_KEY, 0);
    }

    #[test]
    fn test_normalize_coord_positive() {
        let n = normalize_coord(5);
        assert!(n > 0);
    }

    #[test]
    fn test_normalize_coord_zero() {
        let n = normalize_coord(0);
        // normalize_coord(0) produces 0 via the zigzag bitwise operation
        assert_eq!(n, 0);
        // hash_coordinate(0, 0) is non-zero because we apply wrapping_add(1) before multiplying
        assert_ne!(hash_coordinate(0, 0), 0);
    }

    #[test]
    fn test_normalize_coord_negative() {
        let n = normalize_coord(-3);
        assert!(n > 0);
    }

    #[test]
    fn test_normalize_coord_different_values() {
        let n1 = normalize_coord(1);
        let n2 = normalize_coord(2);
        let n3 = normalize_coord(3);
        assert_ne!(n1, n2);
        assert_ne!(n2, n3);
        assert_ne!(n1, n3);
    }

    #[test]
    fn test_castling_combinations_table_coverage() {
        for i in 0..16 {
            let h = CASTLING_COMBINATIONS[i];
            // When i=0 (no castling rights), hash is 0; otherwise should be non-zero
            if i != 0 {
                assert_ne!(h, 0);
            }
        }
    }
}
