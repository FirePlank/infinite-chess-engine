//! Precomputed attack patterns for fast attack detection.

use crate::board::PieceType;

// Leaper Offsets (Knights, Camels, etc.)

pub static KNIGHT_OFFSETS: [(i64, i64); 8] = [
    (1, 2),
    (1, -2),
    (2, 1),
    (2, -1),
    (-1, 2),
    (-1, -2),
    (-2, 1),
    (-2, -1),
];

pub static KING_OFFSETS: [(i64, i64); 8] = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

pub static CAMEL_OFFSETS: [(i64, i64); 8] = [
    (1, 3),
    (1, -3),
    (3, 1),
    (3, -1),
    (-1, 3),
    (-1, -3),
    (-3, 1),
    (-3, -1),
];

pub static GIRAFFE_OFFSETS: [(i64, i64); 8] = [
    (1, 4),
    (1, -4),
    (4, 1),
    (4, -1),
    (-1, 4),
    (-1, -4),
    (-4, 1),
    (-4, -1),
];

pub static ZEBRA_OFFSETS: [(i64, i64); 8] = [
    (2, 3),
    (2, -3),
    (3, 2),
    (3, -2),
    (-2, 3),
    (-2, -3),
    (-3, 2),
    (-3, -2),
];

/// Hawk movement offsets: 2-3 range in orthogonal and diagonal
pub static HAWK_OFFSETS: [(i64, i64); 16] = [
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

pub static KNIGHTRIDER_DIRS: [(i64, i64); 8] = [
    (1, 2),
    (1, -2),
    (2, 1),
    (2, -1),
    (-1, 2),
    (-1, -2),
    (-2, 1),
    (-2, -1),
];

// Slider Directions

pub static ORTHO_DIRS: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub static DIAG_DIRS: [(i64, i64); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

// Piece Type Categorization

#[inline]
pub const fn attacks_like_knight(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Knight
            | PieceType::Chancellor
            | PieceType::Archbishop
            | PieceType::Amazon
            | PieceType::Centaur
            | PieceType::RoyalCentaur
    )
}

#[inline]
pub const fn attacks_like_king(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::King | PieceType::Guard | PieceType::Centaur | PieceType::RoyalCentaur
    )
}

#[inline]
pub const fn is_ortho_slider(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Rook
            | PieceType::Queen
            | PieceType::Chancellor
            | PieceType::Amazon
            | PieceType::RoyalQueen
    )
}

#[inline]
pub const fn is_diag_slider(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Bishop
            | PieceType::Queen
            | PieceType::Archbishop
            | PieceType::Amazon
            | PieceType::RoyalQueen
    )
}

#[inline]
pub const fn is_slider(pt: PieceType) -> bool {
    is_ortho_slider(pt) || is_diag_slider(pt)
}

// Attack Type Lookup Table - for is_square_attacked optimization

pub static KNIGHT_ATTACKERS: [PieceType; 6] = [
    PieceType::Knight,
    PieceType::Chancellor,
    PieceType::Archbishop,
    PieceType::Amazon,
    PieceType::Centaur,
    PieceType::RoyalCentaur,
];

pub static KING_ATTACKERS: [PieceType; 4] = [
    PieceType::King,
    PieceType::Guard,
    PieceType::Centaur,
    PieceType::RoyalCentaur,
];

pub static ORTHO_ATTACKERS: [PieceType; 5] = [
    PieceType::Rook,
    PieceType::Queen,
    PieceType::Chancellor,
    PieceType::Amazon,
    PieceType::RoyalQueen,
];

pub static DIAG_ATTACKERS: [PieceType; 5] = [
    PieceType::Bishop,
    PieceType::Queen,
    PieceType::Archbishop,
    PieceType::Amazon,
    PieceType::RoyalQueen,
];

// Fast Piece Type Checking via Bitset

/// Bitset representation of piece types for membership testing.
/// Each bit corresponds to a PieceType's u8 value.
pub type PieceTypeMask = u32;

pub const fn make_mask(types: &[PieceType]) -> PieceTypeMask {
    let mut mask: u32 = 0;
    let mut i = 0;
    while i < types.len() {
        mask |= 1 << (types[i] as u8);
        i += 1;
    }
    mask
}

pub const KNIGHT_MASK: PieceTypeMask = make_mask(&[
    PieceType::Knight,
    PieceType::Chancellor,
    PieceType::Archbishop,
    PieceType::Amazon,
    PieceType::Centaur,
    PieceType::RoyalCentaur,
]);

pub const KING_MASK: PieceTypeMask = make_mask(&[
    PieceType::King,
    PieceType::Guard,
    PieceType::Centaur,
    PieceType::RoyalCentaur,
]);

pub const CAMEL_MASK: PieceTypeMask = make_mask(&[PieceType::Camel]);
pub const GIRAFFE_MASK: PieceTypeMask = make_mask(&[PieceType::Giraffe]);
pub const ZEBRA_MASK: PieceTypeMask = make_mask(&[PieceType::Zebra]);
pub const HAWK_MASK: PieceTypeMask = make_mask(&[PieceType::Hawk]);

pub const ORTHO_MASK: PieceTypeMask = make_mask(&[
    PieceType::Rook,
    PieceType::Queen,
    PieceType::Chancellor,
    PieceType::Amazon,
    PieceType::RoyalQueen,
]);

pub const DIAG_MASK: PieceTypeMask = make_mask(&[
    PieceType::Bishop,
    PieceType::Queen,
    PieceType::Archbishop,
    PieceType::Amazon,
    PieceType::RoyalQueen,
]);

pub const KNIGHTRIDER_MASK: PieceTypeMask = make_mask(&[PieceType::Knightrider]);

#[inline]
pub const fn matches_mask(pt: PieceType, mask: PieceTypeMask) -> bool {
    (mask >> (pt as u8)) & 1 != 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_mask() {
        assert!(matches_mask(PieceType::Knight, KNIGHT_MASK));
        assert!(matches_mask(PieceType::Chancellor, KNIGHT_MASK));
        assert!(matches_mask(PieceType::Amazon, KNIGHT_MASK));
        assert!(!matches_mask(PieceType::Bishop, KNIGHT_MASK));
        assert!(!matches_mask(PieceType::Rook, KNIGHT_MASK));
    }

    #[test]
    fn test_slider_masks() {
        assert!(matches_mask(PieceType::Rook, ORTHO_MASK));
        assert!(matches_mask(PieceType::Queen, ORTHO_MASK));
        assert!(!matches_mask(PieceType::Bishop, ORTHO_MASK));

        assert!(matches_mask(PieceType::Bishop, DIAG_MASK));
        assert!(matches_mask(PieceType::Queen, DIAG_MASK));
        assert!(!matches_mask(PieceType::Rook, DIAG_MASK));
    }

    #[test]
    fn test_offset_counts() {
        assert_eq!(KNIGHT_OFFSETS.len(), 8);
        assert_eq!(KING_OFFSETS.len(), 8);
        assert_eq!(CAMEL_OFFSETS.len(), 8);
        assert_eq!(HAWK_OFFSETS.len(), 16);
    }

    #[test]
    fn test_attacks_like_knight() {
        assert!(attacks_like_knight(PieceType::Knight));
        assert!(attacks_like_knight(PieceType::Chancellor));
        assert!(attacks_like_knight(PieceType::Archbishop));
        assert!(attacks_like_knight(PieceType::Amazon));
        assert!(attacks_like_knight(PieceType::Centaur));
        assert!(attacks_like_knight(PieceType::RoyalCentaur));

        assert!(!attacks_like_knight(PieceType::King));
        assert!(!attacks_like_knight(PieceType::Queen));
        assert!(!attacks_like_knight(PieceType::Rook));
        assert!(!attacks_like_knight(PieceType::Bishop));
        assert!(!attacks_like_knight(PieceType::Pawn));
    }

    #[test]
    fn test_attacks_like_king() {
        assert!(attacks_like_king(PieceType::King));
        assert!(attacks_like_king(PieceType::Guard));
        assert!(attacks_like_king(PieceType::Centaur));
        assert!(attacks_like_king(PieceType::RoyalCentaur));

        assert!(!attacks_like_king(PieceType::Knight));
        assert!(!attacks_like_king(PieceType::Queen));
        assert!(!attacks_like_king(PieceType::Rook));
    }

    #[test]
    fn test_is_ortho_slider() {
        assert!(is_ortho_slider(PieceType::Rook));
        assert!(is_ortho_slider(PieceType::Queen));
        assert!(is_ortho_slider(PieceType::Chancellor));
        assert!(is_ortho_slider(PieceType::Amazon));
        assert!(is_ortho_slider(PieceType::RoyalQueen));

        assert!(!is_ortho_slider(PieceType::Bishop));
        assert!(!is_ortho_slider(PieceType::Knight));
        assert!(!is_ortho_slider(PieceType::Pawn));
    }

    #[test]
    fn test_is_diag_slider() {
        assert!(is_diag_slider(PieceType::Bishop));
        assert!(is_diag_slider(PieceType::Queen));
        assert!(is_diag_slider(PieceType::Archbishop));
        assert!(is_diag_slider(PieceType::Amazon));
        assert!(is_diag_slider(PieceType::RoyalQueen));

        assert!(!is_diag_slider(PieceType::Rook));
        assert!(!is_diag_slider(PieceType::Knight));
        assert!(!is_diag_slider(PieceType::Pawn));
    }

    #[test]
    fn test_is_slider() {
        // Ortho only
        assert!(is_slider(PieceType::Rook));
        // Diag only
        assert!(is_slider(PieceType::Bishop));
        // Both
        assert!(is_slider(PieceType::Queen));
        assert!(is_slider(PieceType::Amazon));
        // Neither
        assert!(!is_slider(PieceType::Knight));
        assert!(!is_slider(PieceType::Pawn));
        assert!(!is_slider(PieceType::King));
    }

    #[test]
    fn test_make_mask() {
        let mask = make_mask(&[PieceType::Pawn, PieceType::King]);
        assert!(matches_mask(PieceType::Pawn, mask));
        assert!(matches_mask(PieceType::King, mask));
        assert!(!matches_mask(PieceType::Queen, mask));
    }

    #[test]
    fn test_king_mask() {
        assert!(matches_mask(PieceType::King, KING_MASK));
        assert!(matches_mask(PieceType::Guard, KING_MASK));
        assert!(!matches_mask(PieceType::Knight, KING_MASK));
    }

    #[test]
    fn test_leaper_masks() {
        assert!(matches_mask(PieceType::Camel, CAMEL_MASK));
        assert!(!matches_mask(PieceType::Knight, CAMEL_MASK));

        assert!(matches_mask(PieceType::Giraffe, GIRAFFE_MASK));
        assert!(!matches_mask(PieceType::Knight, GIRAFFE_MASK));

        assert!(matches_mask(PieceType::Zebra, ZEBRA_MASK));
        assert!(!matches_mask(PieceType::Knight, ZEBRA_MASK));

        assert!(matches_mask(PieceType::Hawk, HAWK_MASK));
        assert!(!matches_mask(PieceType::Knight, HAWK_MASK));
    }

    #[test]
    fn test_attacker_arrays() {
        assert_eq!(KNIGHT_ATTACKERS.len(), 6);
        assert_eq!(KING_ATTACKERS.len(), 4);
        assert_eq!(ORTHO_ATTACKERS.len(), 5);
        assert_eq!(DIAG_ATTACKERS.len(), 5);
    }

    #[test]
    fn test_direction_arrays() {
        assert_eq!(ORTHO_DIRS.len(), 4);
        assert_eq!(DIAG_DIRS.len(), 4);
        assert_eq!(KNIGHTRIDER_DIRS.len(), 8);
        assert_eq!(ZEBRA_OFFSETS.len(), 8);
        assert_eq!(GIRAFFE_OFFSETS.len(), 8);
    }
}
