//! Precomputed leaper attack masks for 8×8 tiles.
//!
//! Each mask table is indexed as [from_local_square][neighbor_tile_index] -> u64 bitmask.
//! The neighbor_tile_index encodes (dx, dy) in {-1, 0, 1}^2 as (dy+1)*3 + (dx+1).
//! Index 4 is the center tile (same tile as the piece).
//!
use crate::tiles::TILE_SIZE;

// ============================================================================
// Mask Generation Helpers
// ============================================================================

/// Compute target tile delta and local index for a move from local (lx, ly) by offset (dx, dy).
/// Returns (tile_dx, tile_dy, target_local_index) where tile_dx/dy are -1, 0, or 1.
#[inline]
const fn compute_target(lx: i64, ly: i64, dx: i64, dy: i64) -> Option<(i64, i64, usize)> {
    let target_x = lx + dx;
    let target_y = ly + dy;

    // Compute which tile the target lands in
    let tile_dx = if target_x < 0 {
        -1
    } else if target_x >= TILE_SIZE {
        1
    } else {
        0
    };
    let tile_dy = if target_y < 0 {
        -1
    } else if target_y >= TILE_SIZE {
        1
    } else {
        0
    };

    // Compute local coordinates in target tile (wrap around)
    let local_x = ((target_x % TILE_SIZE) + TILE_SIZE) % TILE_SIZE;
    let local_y = ((target_y % TILE_SIZE) + TILE_SIZE) % TILE_SIZE;
    let local_idx = (local_y * 8 + local_x) as usize;

    Some((tile_dx, tile_dy, local_idx))
}

/// Generate masks for a set of offsets.
const fn generate_masks<const N: usize>(offsets: &[(i64, i64); N]) -> [[u64; 9]; 64] {
    let mut masks = [[0u64; 9]; 64];

    let mut from_idx = 0;
    while from_idx < 64 {
        let lx = (from_idx % 8) as i64;
        let ly = (from_idx / 8) as i64;

        let mut offset_idx = 0;
        while offset_idx < N {
            let (dx, dy) = offsets[offset_idx];

            if let Some((tile_dx, tile_dy, local_idx)) = compute_target(lx, ly, dx, dy) {
                let neighbor_idx = ((tile_dy + 1) * 3 + (tile_dx + 1)) as usize;
                masks[from_idx][neighbor_idx] |= 1u64 << local_idx;
            }

            offset_idx += 1;
        }

        from_idx += 1;
    }

    masks
}

// ============================================================================
// Leaper Offset Arrays
// ============================================================================

const KNIGHT_OFFSETS: [(i64, i64); 8] = [
    (1, 2),
    (1, -2),
    (2, 1),
    (2, -1),
    (-1, 2),
    (-1, -2),
    (-2, 1),
    (-2, -1),
];

const KING_OFFSETS: [(i64, i64); 8] = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

const CAMEL_OFFSETS: [(i64, i64); 8] = [
    (1, 3),
    (1, -3),
    (3, 1),
    (3, -1),
    (-1, 3),
    (-1, -3),
    (-3, 1),
    (-3, -1),
];

const GIRAFFE_OFFSETS: [(i64, i64); 8] = [
    (1, 4),
    (1, -4),
    (4, 1),
    (4, -1),
    (-1, 4),
    (-1, -4),
    (-4, 1),
    (-4, -1),
];

const ZEBRA_OFFSETS: [(i64, i64); 8] = [
    (2, 3),
    (2, -3),
    (3, 2),
    (3, -2),
    (-2, 3),
    (-2, -3),
    (-3, 2),
    (-3, -2),
];

const HAWK_OFFSETS: [(i64, i64); 16] = [
    // Orthogonal distance 2 and 3
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (3, 0),
    (-3, 0),
    (0, 3),
    (0, -3),
    // Diagonal distance 2 and 3
    (2, 2),
    (2, -2),
    (-2, 2),
    (-2, -2),
    (3, 3),
    (3, -3),
    (-3, 3),
    (-3, -3),
];

// White pawn attacks (captures up-left and up-right from the pawn)
const WHITE_PAWN_ATTACK_OFFSETS: [(i64, i64); 2] = [(-1, 1), (1, 1)];

// Black pawn attacks (captures down-left and down-right from the pawn)
const BLACK_PAWN_ATTACK_OFFSETS: [(i64, i64); 2] = [(-1, -1), (1, -1)];

// ============================================================================
// Precomputed Mask Tables
// ============================================================================

/// Knight attack masks: [from_local_square][neighbor_tile] -> target squares bitmask
pub static KNIGHT_MASKS: [[u64; 9]; 64] = generate_masks(&KNIGHT_OFFSETS);

/// King attack masks (1 square in any direction)
pub static KING_MASKS: [[u64; 9]; 64] = generate_masks(&KING_OFFSETS);

/// Camel attack masks (1,3 leaper)
pub static CAMEL_MASKS: [[u64; 9]; 64] = generate_masks(&CAMEL_OFFSETS);

/// Giraffe attack masks (1,4 leaper)
pub static GIRAFFE_MASKS: [[u64; 9]; 64] = generate_masks(&GIRAFFE_OFFSETS);

/// Zebra attack masks (2,3 leaper)
pub static ZEBRA_MASKS: [[u64; 9]; 64] = generate_masks(&ZEBRA_OFFSETS);

/// Hawk attack masks (2-3 range compass)
pub static HAWK_MASKS: [[u64; 9]; 64] = generate_masks(&HAWK_OFFSETS);

/// White pawn attack masks (attacking squares, not movement)
pub static WHITE_PAWN_ATTACK_MASKS: [[u64; 9]; 64] = generate_masks(&WHITE_PAWN_ATTACK_OFFSETS);

/// Black pawn attack masks (attacking squares, not movement)
pub static BLACK_PAWN_ATTACK_MASKS: [[u64; 9]; 64] = generate_masks(&BLACK_PAWN_ATTACK_OFFSETS);

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the attack mask for a pawn of the given color from the perspective of the defender.
/// i.e., "which squares can attack this square with a pawn?"
/// For attack detection: we look for enemy pawns that could capture TO this square.
#[inline]
pub fn pawn_attacker_masks(attacker_is_white: bool) -> &'static [[u64; 9]; 64] {
    if attacker_is_white {
        &BLACK_PAWN_ATTACK_MASKS // Look at squares that white pawns attack FROM
    } else {
        &WHITE_PAWN_ATTACK_MASKS // Look at squares that black pawns attack FROM
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_masks_center() {
        // Knight at center of tile (e.g., local index 27 = (3, 3))
        let from_idx = 27; // (3, 3)

        // All 8 knight moves from center should land in the same tile (index 4)
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += KNIGHT_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(total_bits, 8, "Knight should have 8 moves from center");

        // Center tile should have all 8 moves since (3,3)±(1,2)/(2,1) stays in 0..7
        let center_mask = KNIGHT_MASKS[from_idx][4];
        assert_eq!(
            center_mask.count_ones(),
            8,
            "All knight moves from (3,3) stay in center tile"
        );
    }

    #[test]
    fn test_knight_masks_corner() {
        // Knight at corner (0, 0)
        let from_idx = 0;

        // From (0,0), knight moves are:
        // (1,2), (2,1) - stay in center tile
        // (1,-2), (2,-1), (-1,2), (-2,1), (-1,-2), (-2,-1) - go to neighbor tiles
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += KNIGHT_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(
            total_bits, 8,
            "Knight should have 8 move destinations from corner"
        );
    }

    #[test]
    fn test_king_masks() {
        // King at center (3, 3)
        let from_idx = 27;

        // All 8 king moves stay in center tile from (3,3)
        let center_mask = KING_MASKS[from_idx][4];
        assert_eq!(center_mask.count_ones(), 8, "King has 8 moves from center");

        // King at corner (0, 0)
        let from_idx_corner = 0;
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += KING_MASKS[from_idx_corner][n].count_ones();
        }
        // Only 3 moves from corner: (1,0), (0,1), (1,1) stay; others go to neighbor tiles
        assert_eq!(
            total_bits, 8,
            "King should have 8 move destinations even from corner"
        );
    }

    #[test]
    fn test_giraffe_masks() {
        // Giraffe (1,4 leaper) from center (3, 3)
        let from_idx = 27;

        // (3,3) + (1,4) = (4,7) - in center tile
        // (3,3) + (-1,4) = (2,7) - in center tile
        // (3,3) + (4,1) = (7,4) - in center tile
        // (3,3) + (-4,1) = (-1,4) - neighbor tile
        // etc.
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += GIRAFFE_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(total_bits, 8, "Giraffe has 8 moves");
    }

    #[test]
    fn test_pawn_masks() {
        // White pawn attack from (3, 3) attacks (2, 4) and (4, 4)
        let from_idx = 27; // (3, 3)

        let center_mask = WHITE_PAWN_ATTACK_MASKS[from_idx][4];
        assert_eq!(
            center_mask.count_ones(),
            2,
            "Pawn attacks 2 squares diagonally"
        );

        // Check specific bits: (2, 4) = 4*8+2 = 34, (4, 4) = 4*8+4 = 36
        assert!(center_mask & (1 << 34) != 0, "White pawn attacks (2, 4)");
        assert!(center_mask & (1 << 36) != 0, "White pawn attacks (4, 4)");
    }

    #[test]
    fn test_compute_target() {
        // From center (3, 3), move by (1, 2) -> (4, 5) stays in same tile
        let result = compute_target(3, 3, 1, 2);
        assert!(result.is_some());
        let (tile_dx, tile_dy, local_idx) = result.unwrap();
        assert_eq!(tile_dx, 0);
        assert_eq!(tile_dy, 0);
        assert_eq!(local_idx, 5 * 8 + 4); // local coords (4, 5)
    }

    #[test]
    fn test_compute_target_crosses_tile() {
        // From (7, 7), move by (1, 0) -> (8, 7) goes to tile (1, 0)
        let result = compute_target(7, 7, 1, 0);
        assert!(result.is_some());
        let (tile_dx, tile_dy, _) = result.unwrap();
        assert_eq!(tile_dx, 1);
        assert_eq!(tile_dy, 0);
    }

    #[test]
    fn test_compute_target_negative() {
        // From (0, 0), move by (-1, -1) -> (-1, -1) goes to tile (-1, -1)
        let result = compute_target(0, 0, -1, -1);
        assert!(result.is_some());
        let (tile_dx, tile_dy, local_idx) = result.unwrap();
        assert_eq!(tile_dx, -1);
        assert_eq!(tile_dy, -1);
        assert_eq!(local_idx, 7 * 8 + 7); // local coords (7, 7)
    }

    #[test]
    fn test_camel_masks() {
        // Camel from center (3, 3)
        let from_idx = 27;
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += CAMEL_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(total_bits, 8, "Camel has 8 moves");
    }

    #[test]
    fn test_zebra_masks() {
        // Zebra from center (3, 3)
        let from_idx = 27;
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += ZEBRA_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(total_bits, 8, "Zebra has 8 moves");
    }

    #[test]
    fn test_hawk_masks() {
        // Hawk from center (3, 3)
        let from_idx = 27;
        let mut total_bits = 0;
        for n in 0..9 {
            total_bits += HAWK_MASKS[from_idx][n].count_ones();
        }
        assert_eq!(total_bits, 16, "Hawk has 16 moves");
    }

    #[test]
    fn test_pawn_attacker_masks() {
        // White pawn attacker masks should be inverse of Black pawn attack masks
        let white_masks = pawn_attacker_masks(true);
        let black_masks = pawn_attacker_masks(false);

        // White attackers use BLACK pattern (looking down)
        assert_eq!(
            white_masks as *const _,
            &BLACK_PAWN_ATTACK_MASKS as *const _
        );
        // Black attackers use WHITE pattern (looking up)
        assert_eq!(
            black_masks as *const _,
            &WHITE_PAWN_ATTACK_MASKS as *const _
        );
    }

    #[test]
    fn test_all_masks_have_entries() {
        // Verify all mask lookup tables are properly initialized
        for from_idx in 0..64 {
            // Knight should have 8 total moves
            let mut knight_total = 0;
            for n in 0..9 {
                knight_total += KNIGHT_MASKS[from_idx][n].count_ones();
            }
            assert_eq!(
                knight_total, 8,
                "Knight at {} should have 8 moves",
                from_idx
            );

            // King should have 8 total moves
            let mut king_total = 0;
            for n in 0..9 {
                king_total += KING_MASKS[from_idx][n].count_ones();
            }
            assert_eq!(king_total, 8, "King at {} should have 8 moves", from_idx);
        }
    }
}
