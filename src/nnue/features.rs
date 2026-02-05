//! NNUE Feature Encoding
//!
//! Implements the same feature encoding as gen_nnue_data.rs for consistency.
//! This ensures training and inference use identical feature mappings.

use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;

// ============================================================================
// FEATURE ENCODING CONSTANTS
// ============================================================================

/// Number of RelKP buckets per piece code
const NUM_RELKP_BUCKETS: u32 = 1018;
/// Near zone size (squares within ±8 of king)
const NEAR_ZONE_SIZE: i64 = 8;
/// Near zone bucket count: (2*8+1)^2 = 289
const NEAR_ZONE_BUCKETS: u32 = 289;

// Piece codes for RelKP
const FRIENDLY_PAWN_BASE: u32 = 0; // 0-7: friendly pawn promo bins
const FRIENDLY_KNIGHT: u32 = 8;
const FRIENDLY_BISHOP: u32 = 9;
const FRIENDLY_ROOK: u32 = 10;
const FRIENDLY_QUEEN: u32 = 11;
const ENEMY_PAWN_BASE: u32 = 12; // 12-19: enemy pawn promo bins
const ENEMY_KNIGHT: u32 = 20;
const ENEMY_BISHOP: u32 = 21;
const ENEMY_ROOK: u32 = 22;
const ENEMY_QUEEN: u32 = 23;
const ENEMY_KING: u32 = 24;

// ThreatEdges feature offsets
const SLIDER_THREAT_FEATURES: u32 = 6336;
const KNIGHT_THREAT_FEATURES: u32 = 192;
const PAWN_THREAT_FEATURES: u32 = 48;

// ============================================================================
// RELKP ENCODING
// ============================================================================

/// Compute sign code for far zone: 0=negative, 1=zero, 2=positive
#[inline]
fn sign_code(d: i64) -> u32 {
    if d < 0 {
        0
    } else if d == 0 {
        1
    } else {
        2
    }
}

/// Compute log-distance bin for far zone
#[inline]
fn dist_bin(d: i64) -> u32 {
    let d = d.abs();
    match d {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        64..=127 => 7,
        _ => 8,
    }
}

/// Compute pawn promotion distance bin (0-7)
#[inline]
fn promo_dist_bin(pawn_y: i64, pawn_color: PlayerColor, promo_ranks: &[i64]) -> u32 {
    let dir = if pawn_color == PlayerColor::White {
        1
    } else {
        -1
    };

    let mut min_dist = i64::MAX;
    for &rank in promo_ranks {
        let steps = (rank - pawn_y) * dir;
        if steps >= 0 && steps < min_dist {
            min_dist = steps;
        }
    }

    if min_dist == i64::MAX {
        return 7;
    }

    match min_dist {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        _ => 7,
    }
}

/// Compute RelKP bucket from relative coordinates
#[inline]
fn relkp_bucket(dx: i64, dy: i64) -> u32 {
    if dx.abs() <= NEAR_ZONE_SIZE && dy.abs() <= NEAR_ZONE_SIZE {
        ((dx + NEAR_ZONE_SIZE) + 17 * (dy + NEAR_ZONE_SIZE)) as u32
    } else {
        let sx = sign_code(dx);
        let sy = sign_code(dy);
        let sign_pair = sx * 3 + sy;
        let bx = dist_bin(dx);
        let by = dist_bin(dy);
        NEAR_ZONE_BUCKETS + ((sign_pair * 9 + bx) * 9 + by)
    }
}

/// Get piece code for RelKP
fn get_piece_code(
    piece: Piece,
    is_friendly: bool,
    pawn_y: i64,
    pawn_color: PlayerColor,
    white_promo_ranks: &[i64],
    black_promo_ranks: &[i64],
) -> Option<u32> {
    let pt = piece.piece_type();

    match pt {
        PieceType::Pawn => {
            let promo_ranks = if pawn_color == PlayerColor::White {
                white_promo_ranks
            } else {
                black_promo_ranks
            };
            let promo_bin = promo_dist_bin(pawn_y, pawn_color, promo_ranks);
            if is_friendly {
                Some(FRIENDLY_PAWN_BASE + promo_bin)
            } else {
                Some(ENEMY_PAWN_BASE + promo_bin)
            }
        }
        PieceType::Knight => Some(if is_friendly {
            FRIENDLY_KNIGHT
        } else {
            ENEMY_KNIGHT
        }),
        PieceType::Bishop => Some(if is_friendly {
            FRIENDLY_BISHOP
        } else {
            ENEMY_BISHOP
        }),
        PieceType::Rook => Some(if is_friendly {
            FRIENDLY_ROOK
        } else {
            ENEMY_ROOK
        }),
        PieceType::Queen => Some(if is_friendly {
            FRIENDLY_QUEEN
        } else {
            ENEMY_QUEEN
        }),
        PieceType::King => {
            if is_friendly {
                None // Friendly king is always at origin
            } else {
                Some(ENEMY_KING)
            }
        }
        _ => None,
    }
}

/// Build RelKP feature list for a given perspective
fn build_relkp_list(gs: &GameState, perspective: PlayerColor) -> Vec<u32> {
    let king_pos = match perspective {
        PlayerColor::White => gs.white_king_pos,
        PlayerColor::Black => gs.black_king_pos,
        _ => None,
    };

    let Some(king) = king_pos else {
        return Vec::new();
    };

    let white_promo = &gs.game_rules.promotion_ranks.white;
    let black_promo = &gs.game_rules.promotion_ranks.black;

    let mut features = Vec::with_capacity(32);

    for (px, py, piece) in gs.board.iter_all_pieces() {
        let piece_color = piece.color();

        if piece_color == PlayerColor::Neutral {
            continue;
        }

        if piece.piece_type() == PieceType::King && piece_color == perspective {
            continue;
        }

        let is_friendly = piece_color == perspective;

        let (dx, dy) = if perspective == PlayerColor::White {
            (px - king.x, py - king.y)
        } else {
            (-(px - king.x), -(py - king.y))
        };

        let bucket = relkp_bucket(dx, dy);

        if let Some(code) = get_piece_code(
            piece,
            is_friendly,
            py,
            piece_color,
            white_promo,
            black_promo,
        ) {
            let feature_id = code * NUM_RELKP_BUCKETS + bucket;
            features.push(feature_id);
        }
    }

    features
}

/// Build RelKP active feature lists for both perspectives
pub fn build_relkp_active_lists(gs: &GameState) -> (Vec<u32>, Vec<u32>) {
    (
        build_relkp_list(gs, PlayerColor::White),
        build_relkp_list(gs, PlayerColor::Black),
    )
}

// ============================================================================
// THREATGEDGES ENCODING
// ============================================================================

/// Victim type encoding
#[inline]
fn victim_type(pt: PieceType) -> Option<u32> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Knight => Some(1),
        PieceType::Bishop => Some(2),
        PieceType::Rook => Some(3),
        PieceType::Queen => Some(4),
        PieceType::King => Some(5),
        _ => None,
    }
}

/// Direction index (0-7) from dx, dy
#[inline]
fn direction_index(dx: i64, dy: i64) -> u32 {
    match (dx.signum(), dy.signum()) {
        (0, 1) => 0,
        (1, 1) => 1,
        (1, 0) => 2,
        (1, -1) => 3,
        (0, -1) => 4,
        (-1, -1) => 5,
        (-1, 0) => 6,
        (-1, 1) => 7,
        _ => 0,
    }
}

/// Knight offset index (0-7)
#[allow(dead_code)]
#[inline]
fn knight_offset_index(dx: i64, dy: i64) -> u32 {
    match (dx, dy) {
        (-2, -1) => 0,
        (-2, 1) => 1,
        (-1, -2) => 2,
        (-1, 2) => 3,
        (1, -2) => 4,
        (1, 2) => 5,
        (2, -1) => 6,
        (2, 1) => 7,
        _ => 0,
    }
}

/// Pawn attack direction
#[inline]
fn pawn_attack_dir(dx: i64, pawn_color: PlayerColor) -> u32 {
    let effective_dx = if pawn_color == PlayerColor::White {
        dx
    } else {
        -dx
    };
    if effective_dx < 0 { 0 } else { 1 }
}

/// Slider distance bin
#[inline]
fn slider_dist_bin(dist: i64) -> u32 {
    match dist {
        1 => 0,
        2 => 1,
        3 => 2,
        4..=5 => 3,
        6..=7 => 4,
        8..=15 => 5,
        16..=31 => 6,
        32..=63 => 7,
        64..=127 => 8,
        128..=255 => 9,
        _ => 10,
    }
}

/// Slider subtype
#[inline]
fn slider_subtype(pt: PieceType) -> Option<u32> {
    match pt {
        PieceType::Bishop => Some(0),
        PieceType::Rook => Some(1),
        PieceType::Queen => Some(2),
        _ => None,
    }
}

/// Encode slider threat feature
fn encode_slider_threat(
    att_friendly: bool,
    vic_friendly: bool,
    subtype: u32,
    direction: u32,
    dist_bin: u32,
    vic_type: u32,
) -> u32 {
    let side_idx = (att_friendly as u32) * 2 + (vic_friendly as u32);
    side_idx * 1584 + subtype * 528 + direction * 66 + dist_bin * 6 + vic_type
}

/// Encode knight threat feature
fn encode_knight_threat(
    att_friendly: bool,
    vic_friendly: bool,
    offset_idx: u32,
    vic_type: u32,
) -> u32 {
    let side_idx = (att_friendly as u32) * 2 + (vic_friendly as u32);
    SLIDER_THREAT_FEATURES + side_idx * 48 + offset_idx * 6 + vic_type
}

/// Encode pawn threat feature
fn encode_pawn_threat(
    att_friendly: bool,
    vic_friendly: bool,
    attack_dir: u32,
    vic_type: u32,
) -> u32 {
    let side_idx = (att_friendly as u32) * 2 + (vic_friendly as u32);
    SLIDER_THREAT_FEATURES + KNIGHT_THREAT_FEATURES + side_idx * 12 + attack_dir * 6 + vic_type
}

/// Encode king threat feature
fn encode_king_threat(
    att_friendly: bool,
    vic_friendly: bool,
    direction: u32,
    vic_type: u32,
) -> u32 {
    let side_idx = (att_friendly as u32) * 2 + (vic_friendly as u32);
    SLIDER_THREAT_FEATURES
        + KNIGHT_THREAT_FEATURES
        + PAWN_THREAT_FEATURES
        + side_idx * 48
        + direction * 6
        + vic_type
}

/// Build ThreatEdges feature list for a given perspective
fn build_threat_list(gs: &GameState, perspective: PlayerColor) -> Vec<u32> {
    let mut features = Vec::with_capacity(64);
    let indices = &gs.spatial_indices;

    for (ax, ay, attacker) in gs.board.iter_all_pieces() {
        let att_color = attacker.color();
        if att_color == PlayerColor::Neutral {
            continue;
        }
        let att_pt = attacker.piece_type();
        let att_friendly = att_color == perspective;

        match att_pt {
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
                let subtype = match slider_subtype(att_pt) {
                    Some(s) => s,
                    None => continue,
                };

                let directions: &[(i64, i64)] = if att_pt == PieceType::Bishop {
                    &[(1, 1), (1, -1), (-1, 1), (-1, -1)]
                } else if att_pt == PieceType::Rook {
                    &[(0, 1), (0, -1), (1, 0), (-1, 0)]
                } else {
                    &[
                        (0, 1),
                        (0, -1),
                        (1, 0),
                        (-1, 0),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ]
                };

                for &(dx, dy) in directions {
                    if let Some((vx, vy, victim)) = indices.find_first_blocker(ax, ay, dx, dy) {
                        let vic_pt = victim.piece_type();
                        if let Some(_vic_idx) = victim_type(vic_pt) {
                            let vic_color = victim.color();
                            if vic_color == PlayerColor::Neutral {
                                continue;
                            }
                            let dist = (vx - ax).abs().max((vy - ay).abs());
                            let dir_idx = direction_index(dx, dy);
                            let db = slider_dist_bin(dist);
                            let vic_friendly = vic_color == perspective;
                            let feat = encode_slider_threat(
                                att_friendly,
                                vic_friendly,
                                subtype,
                                dir_idx,
                                db,
                                victim_type(vic_pt).unwrap(),
                            );
                            features.push(feat);
                        }
                    }
                }
            }
            PieceType::Knight => {
                const KNIGHT_OFFSETS: [(i64, i64); 8] = [
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ];
                for (i, &(dx, dy)) in KNIGHT_OFFSETS.iter().enumerate() {
                    let vx = ax + dx;
                    let vy = ay + dy;
                    if let Some(victim) = gs.board.get_piece(vx, vy) {
                        let vic_color = victim.color();
                        if vic_color == PlayerColor::Neutral {
                            continue;
                        }
                        if let Some(vt) = victim_type(victim.piece_type()) {
                            let vic_friendly = vic_color == perspective;
                            let feat =
                                encode_knight_threat(att_friendly, vic_friendly, i as u32, vt);
                            features.push(feat);
                        }
                    }
                }
            }
            PieceType::Pawn => {
                let forward = if att_color == PlayerColor::White {
                    1
                } else {
                    -1
                };
                for dx in [-1i64, 1] {
                    let vx = ax + dx;
                    let vy = ay + forward;
                    if let Some(victim) = gs.board.get_piece(vx, vy) {
                        let vic_color = victim.color();
                        if vic_color == PlayerColor::Neutral {
                            continue;
                        }
                        if let Some(vt) = victim_type(victim.piece_type()) {
                            let vic_friendly = vic_color == perspective;
                            let atk_dir = pawn_attack_dir(dx, att_color);
                            let feat = encode_pawn_threat(att_friendly, vic_friendly, atk_dir, vt);
                            features.push(feat);
                        }
                    }
                }
            }
            PieceType::King => {
                for dx in -1..=1i64 {
                    for dy in -1..=1i64 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let vx = ax + dx;
                        let vy = ay + dy;
                        if let Some(victim) = gs.board.get_piece(vx, vy) {
                            let vic_color = victim.color();
                            if vic_color == PlayerColor::Neutral {
                                continue;
                            }
                            if let Some(vt) = victim_type(victim.piece_type()) {
                                let vic_friendly = vic_color == perspective;
                                let dir_idx = direction_index(dx, dy);
                                let feat =
                                    encode_king_threat(att_friendly, vic_friendly, dir_idx, vt);
                                features.push(feat);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    features
}

/// Build ThreatEdges active feature lists for both perspectives
pub fn build_threat_active_lists(gs: &GameState) -> (Vec<u32>, Vec<u32>) {
    (
        build_threat_list(gs, PlayerColor::White),
        build_threat_list(gs, PlayerColor::Black),
    )
}

/// Compute a single RelKP feature ID for a piece at a given position.
/// Returns None if the piece type is not supported or is the friendly king.
#[allow(dead_code)]
pub fn relkp_feature_id(
    persp: PlayerColor,
    piece: Piece,
    piece_coord: Coordinate,
    king_coord: Coordinate,
    gs: &GameState,
) -> Option<u32> {
    let piece_color = piece.color();
    if piece_color == PlayerColor::Neutral {
        return None;
    }

    // Friendly king is at origin
    if piece.piece_type() == PieceType::King && piece_color == persp {
        return None;
    }

    let is_friendly = piece_color == persp;

    let (dx, dy) = if persp == PlayerColor::White {
        (piece_coord.x - king_coord.x, piece_coord.y - king_coord.y)
    } else {
        (
            -(piece_coord.x - king_coord.x),
            -(piece_coord.y - king_coord.y),
        )
    };

    let bucket = relkp_bucket(dx, dy);

    let white_promo = &gs.game_rules.promotion_ranks.white;
    let black_promo = &gs.game_rules.promotion_ranks.black;

    get_piece_code(
        piece,
        is_friendly,
        piece_coord.y,
        piece_color,
        white_promo,
        black_promo,
    )
    .map(|code| code * NUM_RELKP_BUCKETS + bucket)
}
