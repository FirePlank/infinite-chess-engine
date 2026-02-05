//! NNUE Dataset Generator for Infinite Chess
//!
//! Generates training data for InfNNUE-v1 by playing self-play games with variants
//! that only use standard chess pieces (K,Q,R,B,N,P).
//!
//! Features:
//! - RelKP: Translation-invariant piece positions relative to king (25 piece codes × 1018 buckets)
//! - ThreatEdges: Attack/defense relationships with distance binning (6768 features)
//! - Dynamic pawn promotion distance bins for shifted promotion lines
//! - Rayon-based parallel game generation
//! - Compact binary output format for efficient training

use hydrochess_wasm::{
    Variant,
    board::{Coordinate, Piece, PieceType, PlayerColor},
    game::GameState,
    moves::{Move, MoveGenContext},
    search::{SearchStats, get_best_move},
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Variants that only use standard chess pieces (suitable for NNUE training)
const NNUE_VARIANTS: &[Variant] = &[
    Variant::Classical,
    Variant::ClassicalPlus,
    Variant::Core,
    Variant::Pawndard,
    Variant::SpaceClassic,
    Variant::Knightline,
];

/// Target number of training samples
const DEFAULT_TARGET_SAMPLES: u64 = 10_000_000;
/// Probability of sampling a position after a move
const DEFAULT_SAMPLE_RATE: f64 = 0.05; // 5 samples per 100-ply game on average
/// Maximum coordinate bound for generated positions
const COORD_BOUND: i64 = 8192;
/// Search depth for best move during self-play
const DEFAULT_SELFPLAY_DEPTH: usize = 2;
/// Search depth for teacher scores
const DEFAULT_TEACHER_DEPTH: usize = 6;
/// Probability of playing best move (vs random)
const BEST_MOVE_PROB: f64 = 0.80;
/// Ply after which randomization kicks in
const RANDOM_PLY_START: u32 = 8;
/// Maximum game length in plies
const MAX_GAME_PLY: u32 = 300; // Reduced from 500 to prune super long games

// ============================================================================
// FEATURE ENCODING CONSTANTS
// ============================================================================

/// Number of RelKP buckets per piece code
const NUM_RELKP_BUCKETS: u32 = 1018;
/// Near zone size (squares within ±8 of king)  
const NEAR_ZONE_SIZE: i64 = 8;
/// Near zone bucket count: (2*8+1)^2 = 289
const NEAR_ZONE_BUCKETS: u32 = 289;

/// Piece codes for RelKP:
/// - Friendly: Pawn×8 (promo bins), Knight, Bishop, Rook, Queen = 12 codes
/// - Enemy: Pawn×8 (promo bins), Knight, Bishop, Rook, Queen, King = 13 codes
/// Total = 25 codes
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
const NUM_PIECE_CODES: u32 = 25;

/// Total RelKP features = 25 * 1018 = 25450
pub const TOTAL_RELKP_FEATURES: u32 = NUM_PIECE_CODES * NUM_RELKP_BUCKETS;

/// ThreatEdges feature counts per category
const SLIDER_THREAT_FEATURES: u32 = 6336; // 2*2*3*8*11*6
const KNIGHT_THREAT_FEATURES: u32 = 192; // 2*2*8*6
const PAWN_THREAT_FEATURES: u32 = 48; // 2*2*2*6
const KING_THREAT_FEATURES: u32 = 192; // 2*2*8*6

/// Total ThreatEdges features = 6768
pub const TOTAL_THREAT_FEATURES: u32 =
    SLIDER_THREAT_FEATURES + KNIGHT_THREAT_FEATURES + PAWN_THREAT_FEATURES + KING_THREAT_FEATURES;

// ============================================================================
// BINARY FORMAT
// ============================================================================

/// Magic bytes for file identification
const MAGIC: &[u8; 8] = b"INNUE1\0\0";
const VERSION: u32 = 1;

/// Header size in bytes (unused but kept for reference)
const _HEADER_SIZE: usize = 16;

// ============================================================================
// PRNG (from search.rs)
// ============================================================================

#[derive(Clone)]
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        let state = if seed == 0 { 0xDEADBEEFCAFE } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[allow(dead_code)]
    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

// ============================================================================
// RELKP FEATURE ENCODING
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

    // Find minimum distance to a valid promotion rank
    let mut min_dist = i64::MAX;
    for &rank in promo_ranks {
        let steps = (rank - pawn_y) * dir;
        if steps >= 0 && steps < min_dist {
            min_dist = steps;
        }
    }

    if min_dist == i64::MAX {
        return 7; // No valid promotion rank found
    }

    match min_dist {
        0 => 0, // Already on promotion rank
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
        // Near zone: exact geometry
        ((dx + NEAR_ZONE_SIZE) + 17 * (dy + NEAR_ZONE_SIZE)) as u32
    } else {
        // Far zone: sign + log-distance bins
        let sx = sign_code(dx);
        let sy = sign_code(dy);
        let sign_pair = sx * 3 + sy;
        let bx = dist_bin(dx);
        let by = dist_bin(dy);
        NEAR_ZONE_BUCKETS + ((sign_pair * 9 + bx) * 9 + by)
    }
}

/// Get piece code for RelKP, accounting for pawn promo bins
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
                None // Friendly king is always at origin, excluded
            } else {
                Some(ENEMY_KING)
            }
        }
        _ => None, // Fairy pieces, Void, Obstacle
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

        // Skip neutral pieces
        if piece_color == PlayerColor::Neutral {
            continue;
        }

        // Skip the friendly king (always at origin)
        if piece.piece_type() == PieceType::King && piece_color == perspective {
            continue;
        }

        let is_friendly = piece_color == perspective;

        // Compute relative coordinates
        let (dx, dy) = if perspective == PlayerColor::White {
            (px - king.x, py - king.y)
        } else {
            // Rotate 180° for Black perspective
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

// ============================================================================
// THREATGEDGES FEATURE ENCODING
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
        (0, 1) => 0,   // N
        (1, 1) => 1,   // NE
        (1, 0) => 2,   // E
        (1, -1) => 3,  // SE
        (0, -1) => 4,  // S
        (-1, -1) => 5, // SW
        (-1, 0) => 6,  // W
        (-1, 1) => 7,  // NW
        _ => 0,
    }
}

/// Knight offset index (0-7)
#[inline]
#[allow(dead_code)]
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

/// Pawn attack direction (0=forward-left, 1=forward-right from pawn's perspective)
#[inline]
#[allow(dead_code)]
fn pawn_attack_dir(dx: i64, pawn_color: PlayerColor) -> u32 {
    let effective_dx = if pawn_color == PlayerColor::White {
        dx
    } else {
        -dx
    };
    if effective_dx < 0 { 0 } else { 1 }
}

/// Slider dist bin (0-10 for distances 1, 2, 3, 4-5, 6-7, 8-15, 16-31, 32-63, 64-127, 128-255, 256+)
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
    // Layout: side_idx * (3*8*11*6) + subtype * (8*11*6) + dir * (11*6) + dist * 6 + vic
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
    // After sliders
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

    // Iterate through all pieces as potential attackers
    for (ax, ay, attacker) in gs.board.iter_all_pieces() {
        let att_color = attacker.color();
        if att_color == PlayerColor::Neutral {
            continue;
        }
        let att_pt = attacker.piece_type();
        let att_friendly = att_color == perspective;
        let _att_coord = Coordinate::new(ax, ay);

        match att_pt {
            // Sliders: find first blocker on each ray
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
                let subtype = slider_subtype(att_pt).unwrap();

                let directions: &[(i64, i64)] = match att_pt {
                    PieceType::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
                    PieceType::Rook => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
                    PieceType::Queen => &[
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ],
                    _ => continue,
                };

                for &(dx, dy) in directions {
                    if let Some((vx, vy, victim)) = indices.find_first_blocker(ax, ay, dx, dy) {
                        let vic_color = victim.color();
                        if vic_color == PlayerColor::Neutral {
                            continue;
                        }

                        let vic_friendly = vic_color == perspective;
                        let Some(vic_type) = victim_type(victim.piece_type()) else {
                            continue;
                        };

                        let dist = (vx - ax).abs().max((vy - ay).abs());
                        let db = slider_dist_bin(dist);
                        let dir_idx = direction_index(dx, dy);

                        features.push(encode_slider_threat(
                            att_friendly,
                            vic_friendly,
                            subtype,
                            dir_idx,
                            db,
                            vic_type,
                        ));
                    }
                }
            }

            // Knights
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

                for (idx, &(dx, dy)) in KNIGHT_OFFSETS.iter().enumerate() {
                    let vx = ax + dx;
                    let vy = ay + dy;

                    if let Some(victim) = gs.board.get_piece(vx, vy) {
                        let vic_color = victim.color();
                        if vic_color == PlayerColor::Neutral {
                            continue;
                        }

                        let vic_friendly = vic_color == perspective;
                        let Some(vic_type) = victim_type(victim.piece_type()) else {
                            continue;
                        };

                        features.push(encode_knight_threat(
                            att_friendly,
                            vic_friendly,
                            idx as u32,
                            vic_type,
                        ));
                    }
                }
            }

            // Pawns
            PieceType::Pawn => {
                let pawn_dir = if att_color == PlayerColor::White {
                    1
                } else {
                    -1
                };

                for (attack_idx, dx) in [(-1i64, 0u32), (1, 1)] {
                    let vx = ax + attack_idx;
                    let vy = ay + pawn_dir;

                    if let Some(victim) = gs.board.get_piece(vx, vy) {
                        let vic_color = victim.color();
                        if vic_color == PlayerColor::Neutral {
                            continue;
                        }

                        let vic_friendly = vic_color == perspective;
                        let Some(vic_type) = victim_type(victim.piece_type()) else {
                            continue;
                        };

                        features.push(encode_pawn_threat(att_friendly, vic_friendly, dx, vic_type));
                    }
                }
            }

            // King (adjacency threats)
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

                            let vic_friendly = vic_color == perspective;
                            let Some(vic_type) = victim_type(victim.piece_type()) else {
                                continue;
                            };

                            let dir_idx = direction_index(dx, dy);
                            features.push(encode_king_threat(
                                att_friendly,
                                vic_friendly,
                                dir_idx,
                                vic_type,
                            ));
                        }
                    }
                }
            }

            _ => {}
        }
    }

    features
}

// ============================================================================
// NNUE APPLICABILITY
// ============================================================================

/// Check if position is suitable for NNUE (standard pieces only)
fn is_nnue_applicable(gs: &GameState) -> bool {
    // Must have exactly one king per side
    if gs.white_king_pos.is_none() || gs.black_king_pos.is_none() {
        return false;
    }

    // All pieces must be standard chess pieces
    for (_, _, piece) in gs.board.iter_all_pieces() {
        match piece.piece_type() {
            PieceType::King
            | PieceType::Queen
            | PieceType::Rook
            | PieceType::Bishop
            | PieceType::Knight
            | PieceType::Pawn => {}
            PieceType::Void | PieceType::Obstacle => return false,
            _ => return false, // Fairy pieces
        }
    }

    true
}

// ============================================================================
// SAMPLE RECORD
// ============================================================================

/// Pending sample before game result is known
struct PendingSample {
    relkp_white: Vec<u32>,
    relkp_black: Vec<u32>,
    threat_white: Vec<u32>,
    threat_black: Vec<u32>,
    stm: PlayerColor,
    teacher_cp: i16,
}

/// Final sample record ready for output
struct SampleRecord {
    stm: u8,
    relkp_white: Vec<u32>,
    relkp_black: Vec<u32>,
    threat_white: Vec<u32>,
    threat_black: Vec<u32>,
    teacher_cp: i16,
    result_wdl: i8,
}

impl SampleRecord {
    /// Write record to binary output
    fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&[self.stm])?;
        writer.write_all(&[self.relkp_white.len() as u8])?;
        writer.write_all(&[self.relkp_black.len() as u8])?;
        writer.write_all(&(self.threat_white.len() as u16).to_le_bytes())?;
        writer.write_all(&(self.threat_black.len() as u16).to_le_bytes())?;

        for &f in &self.relkp_white {
            writer.write_all(&f.to_le_bytes())?;
        }
        for &f in &self.relkp_black {
            writer.write_all(&f.to_le_bytes())?;
        }
        for &f in &self.threat_white {
            writer.write_all(&f.to_le_bytes())?;
        }
        for &f in &self.threat_black {
            writer.write_all(&f.to_le_bytes())?;
        }

        writer.write_all(&self.teacher_cp.to_le_bytes())?;
        writer.write_all(&[self.result_wdl as u8])?;

        Ok(())
    }
}

// ============================================================================
// GAME PLAYING
// ============================================================================

/// Determine game result: +1 = White wins, 0 = Draw, -1 = Black wins
fn determine_game_result(gs: &GameState) -> i8 {
    let ctx = MoveGenContext {
        special_rights: &gs.special_rights,
        en_passant: &gs.en_passant,
        game_rules: &gs.game_rules,
        indices: &gs.spatial_indices,
        enemy_king_pos: None,
    };
    let moves = hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &ctx);

    // Filter for strictly legal moves (must not leave king in check)
    let mut has_moves = false;
    for m in moves {
        let mut test_gs = gs.clone();
        test_gs.make_move(&m);
        if !test_gs.is_move_illegal() {
            has_moves = true;
            break;
        }
    }

    // Check for checkmate
    if gs.is_in_check() && !has_moves {
        // Side to move is checkmated
        match gs.turn {
            PlayerColor::White => -1, // Black wins
            PlayerColor::Black => 1,  // White wins
            _ => 0,
        }
    } else if !has_moves {
        // Stalemate
        0
    } else if gs.is_repetition(0) || gs.is_fifty() {
        // Draw by repetition or 50-move rule
        0
    } else {
        // Game not over (shouldn't happen if called correctly)
        0
    }
}

/// Check if there is any strictly legal move
fn has_any_legal_move(gs: &GameState) -> bool {
    let ctx = MoveGenContext {
        special_rights: &gs.special_rights,
        en_passant: &gs.en_passant,
        game_rules: &gs.game_rules,
        indices: &gs.spatial_indices,
        enemy_king_pos: None,
    };
    let moves = hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &ctx);
    for m in moves {
        let mut test_gs = gs.clone();
        test_gs.make_move(&m);
        if !test_gs.is_move_illegal() {
            return true;
        }
    }
    false
}

/// Get weighted random move (higher weight for longer distance moves)
fn get_weighted_random_move(gs: &GameState, prng: &mut Prng) -> Option<Move> {
    let ctx = MoveGenContext {
        special_rights: &gs.special_rights,
        en_passant: &gs.en_passant,
        game_rules: &gs.game_rules,
        indices: &gs.spatial_indices,
        enemy_king_pos: None,
    };

    let moves = hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &ctx);

    if moves.is_empty() {
        return None;
    }

    // Filter moves and compute weights
    let mut valid_moves: Vec<(Move, f64)> = Vec::new();

    for m in moves.iter() {
        // Skip moves that go beyond coordinate bounds
        if m.to.x.abs() > COORD_BOUND || m.to.y.abs() > COORD_BOUND {
            continue;
        }

        // Make move and check legality
        let mut test_gs = gs.clone();
        let _undo = test_gs.make_move(m);
        if test_gs.is_move_illegal() {
            continue;
        }

        let dx = (m.to.x - m.from.x).abs();
        let dy = (m.to.y - m.from.y).abs();
        let dist = dx.max(dy);
        let weight = 1.0 + (dist * dist) as f64;

        valid_moves.push((*m, weight));
    }

    if valid_moves.is_empty() {
        return None;
    }

    // Weighted random selection
    let total_weight: f64 = valid_moves.iter().map(|(_, w)| w).sum();
    let mut r = prng.next_f64() * total_weight;

    for (m, w) in &valid_moves {
        r -= w;
        if r <= 0.0 {
            return Some(*m);
        }
    }

    // Fallback
    Some(valid_moves[0].0)
}

/// Play a single game and return samples
fn play_game(
    variant: Variant,
    thread_id: usize,
    game_idx: usize,
    selfplay_depth: usize,
    teacher_depth: usize,
    sample_rate: f64,
) -> (u64, Vec<SampleRecord>) {
    let mut gs = GameState::new();
    gs.setup_variant(variant);
    gs.recompute_piece_counts();
    gs.recompute_hash();

    let mut prng =
        Prng::new((thread_id as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (game_idx as u64));
    let mut pending_samples: Vec<PendingSample> = Vec::new();

    for ply in 0..MAX_GAME_PLY {
        // Check for game end
        if gs.is_repetition(0) || gs.is_fifty() || !has_any_legal_move(&gs) {
            break;
        }

        // Select move
        let chosen_move = if ply < RANDOM_PLY_START {
            // First 8 plies: always best move
            match get_best_move(&mut gs, selfplay_depth, u128::MAX, true, false) {
                Some((m, _, _)) => m,
                None => {
                    // Fallback to first legal move
                    let local_ctx = MoveGenContext {
                        special_rights: &gs.special_rights,
                        en_passant: &gs.en_passant,
                        game_rules: &gs.game_rules,
                        indices: &gs.spatial_indices,
                        enemy_king_pos: None,
                    };
                    let mut fallback = None;
                    let pseudo_moves =
                        hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &local_ctx);
                    for m in pseudo_moves {
                        let mut test_gs = gs.clone();
                        test_gs.make_move(&m);
                        if !test_gs.is_move_illegal() {
                            fallback = Some(m);
                            break;
                        }
                    }
                    fallback.expect("has_any_legal_move was true but no legal move found")
                }
            }
        } else {
            // After ply 8: 80% best, 20% weighted random
            if prng.next_f64() < BEST_MOVE_PROB {
                match get_best_move(&mut gs, selfplay_depth, u128::MAX, true, false) {
                    Some((m, _, _)) => m,
                    None => {
                        let local_ctx = MoveGenContext {
                            special_rights: &gs.special_rights,
                            en_passant: &gs.en_passant,
                            game_rules: &gs.game_rules,
                            indices: &gs.spatial_indices,
                            enemy_king_pos: None,
                        };
                        let mut fallback = None;
                        let pseudo_moves =
                            hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &local_ctx);
                        for m in pseudo_moves {
                            let mut test_gs = gs.clone();
                            test_gs.make_move(&m);
                            if !test_gs.is_move_illegal() {
                                fallback = Some(m);
                                break;
                            }
                        }
                        fallback.expect("has_any_legal_move was true but no legal move found")
                    }
                }
            } else {
                get_weighted_random_move(&gs, &mut prng).unwrap_or_else(|| {
                    // Fallback to first legal move
                    let local_ctx = MoveGenContext {
                        special_rights: &gs.special_rights,
                        en_passant: &gs.en_passant,
                        game_rules: &gs.game_rules,
                        indices: &gs.spatial_indices,
                        enemy_king_pos: None,
                    };
                    let pseudo_moves =
                        hydrochess_wasm::moves::get_legal_moves(&gs.board, gs.turn, &local_ctx);
                    for m in pseudo_moves {
                        let mut test_gs = gs.clone();
                        test_gs.make_move(&m);
                        if !test_gs.is_move_illegal() {
                            return m;
                        }
                    }
                    panic!("has_any_legal_move was true but no legal move found");
                })
            }
        };

        // Sample with probability SAMPLE_RATE if NNUE applicable
        if prng.next_f64() < sample_rate && is_nnue_applicable(&gs) {
            let (_, teacher_cp, _) = get_best_move(&mut gs, teacher_depth, u128::MAX, true, false)
                .unwrap_or((
                    Move::new(
                        Coordinate::new(0, 0),
                        Coordinate::new(0, 0),
                        Piece::new(PieceType::Void, PlayerColor::Neutral),
                    ),
                    0,
                    SearchStats {
                        nodes: 0,
                        tt_capacity: 0,
                        tt_used: 0,
                        tt_fill_permille: 0,
                    },
                ));

            let clamped_cp = teacher_cp.clamp(-31000, 31000);

            pending_samples.push(PendingSample {
                relkp_white: build_relkp_list(&gs, PlayerColor::White),
                relkp_black: build_relkp_list(&gs, PlayerColor::Black),
                threat_white: build_threat_list(&gs, PlayerColor::White),
                threat_black: build_threat_list(&gs, PlayerColor::Black),
                stm: gs.turn,
                teacher_cp: clamped_cp as i16,
            });
        }

        // Make move
        gs.make_move(&chosen_move);
    }

    // Determine game result
    let result = determine_game_result(&gs);

    // Convert pending samples to final records with WDL
    let final_samples: Vec<SampleRecord> = pending_samples
        .into_iter()
        .filter_map(|s| {
            // Convert result to side-to-move perspective
            let result_wdl = match s.stm {
                PlayerColor::White => result,
                PlayerColor::Black => -result,
                _ => 0,
            };

            // Filter inconsistent positions (Data Cleaning)
            // 1. High CP Draw (>1500cp)
            if result_wdl == 0 && s.teacher_cp.abs() > 1500 {
                return None;
            }
            // 2. Mismatch: Loss with high CP (>1500cp)
            if result_wdl == -1 && s.teacher_cp > 1500 {
                return None;
            }
            // 3. Mismatch: Win with low CP (<-1500cp)
            if result_wdl == 1 && s.teacher_cp < -1500 {
                return None;
            }

            Some(SampleRecord {
                stm: if s.stm == PlayerColor::White { 0 } else { 1 },
                relkp_white: s.relkp_white,
                relkp_black: s.relkp_black,
                threat_white: s.threat_white,
                threat_black: s.threat_black,
                teacher_cp: s.teacher_cp,
                result_wdl,
            })
        })
        .collect();

    (1, final_samples)
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    #[cfg(debug_assertions)]
    {
        println!("⚠️  WARNING: Running in DEBUG mode. Performance will be significantly reduced.");
        println!("   For production data generation, use: cargo run --bin gen_nnue_data --release");
        println!();
    }

    // Parse args
    let args: Vec<String> = std::env::args().collect();

    let mut target_samples = DEFAULT_TARGET_SAMPLES;
    let mut output_path = "nnue_data.bin".to_string();
    let mut verify_mode = false;
    let mut num_threads = num_cpus::get();
    let mut selfplay_depth = DEFAULT_SELFPLAY_DEPTH;
    let mut teacher_depth = DEFAULT_TEACHER_DEPTH;
    let mut sample_rate = DEFAULT_SAMPLE_RATE;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" | "-s" => {
                if i + 1 < args.len() {
                    target_samples = args[i + 1].parse().unwrap_or(DEFAULT_TARGET_SAMPLES);
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--verify" | "-v" => {
                verify_mode = true;
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--threads" | "-t" => {
                if i + 1 < args.len() {
                    num_threads = args[i + 1].parse().unwrap_or(num_cpus::get());
                    i += 1;
                }
            }
            "--selfplay-depth" => {
                if i + 1 < args.len() {
                    selfplay_depth = args[i + 1].parse().unwrap_or(DEFAULT_SELFPLAY_DEPTH);
                    i += 1;
                }
            }
            "--teacher-depth" => {
                if i + 1 < args.len() {
                    teacher_depth = args[i + 1].parse().unwrap_or(DEFAULT_TEACHER_DEPTH);
                    i += 1;
                }
            }
            "--sample-rate" => {
                if i + 1 < args.len() {
                    sample_rate = args[i + 1].parse().unwrap_or(DEFAULT_SAMPLE_RATE);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("NNUE Dataset Generator for Infinite Chess");
                println!();
                println!("Usage: gen_nnue_data [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -s, --samples <N>   Number of samples to generate (default: 10000000)");
                println!("  -o, --output <PATH> Output file path (default: nnue_data.bin)");
                println!("  -v, --verify <PATH> Verify an existing data file");
                println!("  -t, --threads <N>   Number of threads to use (default: num_cpus)");
                println!("  --selfplay-depth <N> Depth for self-play moves (default: 2)");
                println!("  --teacher-depth <N> Depth for teacher evaluation (default: 6)");
                println!(
                    "  --sample-rate <F>   Probability of sampling a position (default: 0.05)"
                );
                println!("  -h, --help          Show this help");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    if verify_mode {
        verify_data_file(&output_path);
        return;
    }

    // Initialize thread pool
    println!("[gen_nnue_data] Using {} threads", num_threads);
    println!("[gen_nnue_data] Target samples: {}", target_samples);
    println!("[gen_nnue_data] Self-play depth: {}", selfplay_depth);
    println!("[gen_nnue_data] Teacher depth: {}", teacher_depth);
    println!("[gen_nnue_data] Sample rate: {}", sample_rate);
    println!("[gen_nnue_data] Output: {}", output_path);
    println!("[gen_nnue_data] Variants: {}", NNUE_VARIANTS.len());

    // Open or create output file
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&output_path)
        .expect("Failed to open output file");

    let mut initial_samples = 0u64;
    let file_len = file.metadata().unwrap().len();

    if file_len == 0 {
        // Write fresh header
        file.write_all(MAGIC).unwrap();
        file.write_all(&VERSION.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // Placeholder for record count
        println!("[gen_nnue_data] Created new file: {}", output_path);
    } else {
        // Read existing header
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .expect("Failed to read magic from existing file");
        if &magic != MAGIC {
            panic!("Existing file has invalid magic!");
        }
        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)
            .expect("Failed to read version");
        if u32::from_le_bytes(version_buf) != VERSION {
            panic!("Existing file version mismatch!");
        }
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)
            .expect("Failed to read sample count");
        initial_samples = u64::from_le_bytes(count_buf);

        // Seek to end for appending
        file.seek(std::io::SeekFrom::End(0)).unwrap();
        println!(
            "[gen_nnue_data] Resuming from existing file with {} samples",
            initial_samples
        );
    }

    let writer = Mutex::new(BufWriter::new(file));
    let samples_written = AtomicU64::new(initial_samples);
    let games_done = AtomicU64::new(0);
    let start_time = Instant::now();
    let samples_at_start = initial_samples;
    let mut last_reported_pct =
        (initial_samples as f64 / target_samples as f64 * 20.0).floor() as i32;

    // Generate games in parallel until sample target is reached
    // Process in batches to allow clean termination and batch progress reporting
    const BATCH_SIZE: u64 = 500;

    while samples_written.load(Ordering::Relaxed) < target_samples {
        let game_base = games_done.load(Ordering::Relaxed);

        (0..BATCH_SIZE).into_par_iter().for_each(|i| {
            let game_id = game_base + i;

            // Second check inside parallel iterator to avoid starting expensive games
            if samples_written.load(Ordering::Relaxed) >= target_samples {
                return;
            }

            // Select variant round-robin
            let variant = NNUE_VARIANTS[(game_id % NNUE_VARIANTS.len() as u64) as usize];

            // Play game
            let (_games_run, samples) = play_game(
                variant,
                rayon::current_thread_index().unwrap_or(0), // thread_id
                game_id as usize,                           // game_idx
                selfplay_depth,
                teacher_depth,
                sample_rate,
            );

            // Write samples
            if !samples.is_empty() {
                let mut w = writer.lock().unwrap();

                // Re-check target within lock to avoid overshooting too much
                let current = samples_written.load(Ordering::Relaxed);
                if current >= target_samples {
                    // Still add to total samples for count accuracy, but don't write
                    return;
                }

                for sample in &samples {
                    sample.write_to(&mut *w).expect("Failed to write sample");
                }
                let total = samples_written.fetch_add(samples.len() as u64, Ordering::SeqCst)
                    + samples.len() as u64;

                // Crash-safe header update: patch the count every write
                if let Ok(pos) = w.stream_position() {
                    let _ = w.seek(SeekFrom::Start(12));
                    let _ = w.write_all(&total.to_le_bytes());
                    let _ = w.seek(SeekFrom::Start(pos));
                    let _ = w.flush(); // Ensure metadata is visible to other readers
                }
            }

            games_done.fetch_add(1, Ordering::Relaxed);
        });

        // Progress update every 5% increment
        let current_samples = samples_written.load(Ordering::SeqCst);
        let pct = current_samples as f64 / target_samples as f64 * 100.0;
        let pct_step = (pct / 5.0).floor() as i32;

        if pct_step > last_reported_pct {
            last_reported_pct = pct_step;
            let elapsed = start_time.elapsed().as_secs_f64();
            let session_samples = current_samples.saturating_sub(samples_at_start);
            let samples_per_sec = session_samples as f64 / elapsed;

            eprintln!(
                "[gen_nnue_data] Progress: {:.0}% | Samples: {}/{} | {:.1} samples/sec",
                (pct_step * 5) as f64,
                current_samples,
                target_samples,
                samples_per_sec
            );
        }
    }

    // Flush and close writer before patching header
    {
        let mut w = writer.lock().unwrap();
        w.flush().unwrap();
    }
    drop(writer);

    // Patch record count in header
    let total_samples = samples_written.load(Ordering::SeqCst);
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&output_path)
            .expect("Failed to reopen file");
        file.seek(SeekFrom::Start(12)).unwrap();
        file.write_all(&total_samples.to_le_bytes()).unwrap();
    }

    let elapsed = start_time.elapsed();
    println!();
    println!("[gen_nnue_data] Complete!");
    println!("[gen_nnue_data] Samples written: {}", total_samples);
    println!("[gen_nnue_data] Time: {:.1}s", elapsed.as_secs_f64());
    println!("[gen_nnue_data] Output: {}", output_path);
}

fn verify_data_file(path: &str) {
    use std::io::BufReader;

    let file = File::open(path).expect("Failed to open file");
    let mut reader = BufReader::new(file);

    // Read header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).expect("Failed to read magic");

    if &magic != MAGIC {
        eprintln!("Invalid magic: {:?}", magic);
        return;
    }

    let mut version_buf = [0u8; 4];
    reader.read_exact(&mut version_buf).unwrap();
    let version = u32::from_le_bytes(version_buf);

    let mut count_buf = [0u8; 8];
    reader.read_exact(&mut count_buf).unwrap();
    let record_count = u64::from_le_bytes(count_buf);

    println!("Header:");
    println!(
        "  Magic: {:?}",
        std::str::from_utf8(&magic[..6]).unwrap_or("???")
    );
    println!("  Version: {}", version);
    println!("  Records: {}", record_count);

    // Read and validate first few records
    let mut records_read = 0u64;
    let mut total_relkp_features = 0u64;
    let mut total_threat_features = 0u64;

    loop {
        let mut stm = [0u8; 1];
        if reader.read_exact(&mut stm).is_err() {
            break;
        }

        let mut header = [0u8; 6];
        if reader.read_exact(&mut header).is_err() {
            break;
        }

        let relkp_white_len = header[0] as usize;
        let relkp_black_len = header[1] as usize;
        let threat_white_len = u16::from_le_bytes([header[2], header[3]]) as usize;
        let threat_black_len = u16::from_le_bytes([header[4], header[5]]) as usize;

        // Skip feature data
        let features_size =
            (relkp_white_len + relkp_black_len + threat_white_len + threat_black_len) * 4;
        let mut features = vec![0u8; features_size];
        reader.read_exact(&mut features).ok();

        // Skip teacher_cp and result_wdl
        let mut footer = [0u8; 3];
        reader.read_exact(&mut footer).ok();

        total_relkp_features += (relkp_white_len + relkp_black_len) as u64;
        total_threat_features += (threat_white_len + threat_black_len) as u64;
        records_read += 1;

        if records_read >= 10 {
            break;
        }
    }

    println!();
    println!("Sample of first {} records:", records_read);
    println!(
        "  Avg RelKP features per record: {:.1}",
        total_relkp_features as f64 / records_read as f64
    );
    println!(
        "  Avg Threat features per record: {:.1}",
        total_threat_features as f64 / records_read as f64
    );
    println!();
    println!("File appears valid.");
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_code() {
        assert_eq!(sign_code(-5), 0);
        assert_eq!(sign_code(0), 1);
        assert_eq!(sign_code(10), 2);
    }

    #[test]
    fn test_dist_bin() {
        assert_eq!(dist_bin(0), 0);
        assert_eq!(dist_bin(1), 1);
        assert_eq!(dist_bin(-1), 1);
        assert_eq!(dist_bin(2), 2);
        assert_eq!(dist_bin(3), 2);
        assert_eq!(dist_bin(4), 3);
        assert_eq!(dist_bin(7), 3);
        assert_eq!(dist_bin(8), 4);
        assert_eq!(dist_bin(128), 8);
        assert_eq!(dist_bin(1000), 8);
    }

    #[test]
    fn test_near_zone_bucket() {
        // Center (0, 0) -> bucket = (0+8) + 17*(0+8) = 8 + 136 = 144
        assert_eq!(relkp_bucket(0, 0), 144);

        // (-8, -8) -> bucket = 0 + 0 = 0
        assert_eq!(relkp_bucket(-8, -8), 0);

        // (8, 8) -> bucket = 16 + 17*16 = 16 + 272 = 288
        assert_eq!(relkp_bucket(8, 8), 288);
    }

    #[test]
    fn test_far_zone_bucket() {
        // (100, 0) -> far zone
        let bucket = relkp_bucket(100, 0);
        assert!(bucket >= 289);
        assert!(bucket < 1018);
    }

    #[test]
    fn test_promo_dist_bin() {
        // White pawn at y=7, promo rank at y=8 -> 1 step
        assert_eq!(promo_dist_bin(7, PlayerColor::White, &[8]), 1);

        // White pawn at y=2, promo rank at y=8 -> 6 steps
        assert_eq!(promo_dist_bin(2, PlayerColor::White, &[8]), 3); // bin 3 is 4-7 steps

        // Black pawn at y=2, promo rank at y=1 -> 1 step
        assert_eq!(promo_dist_bin(2, PlayerColor::Black, &[1]), 1);

        // Already on promotion rank
        assert_eq!(promo_dist_bin(8, PlayerColor::White, &[8]), 0);
    }

    #[test]
    fn test_piece_codes() {
        let promo_white = vec![8];
        let promo_black = vec![1];

        // Friendly knight
        let knight = Piece::new(PieceType::Knight, PlayerColor::White);
        assert_eq!(
            get_piece_code(
                knight,
                true,
                1,
                PlayerColor::White,
                &promo_white,
                &promo_black
            ),
            Some(FRIENDLY_KNIGHT)
        );

        // Enemy king
        let king = Piece::new(PieceType::King, PlayerColor::Black);
        assert_eq!(
            get_piece_code(
                king,
                false,
                8,
                PlayerColor::Black,
                &promo_white,
                &promo_black
            ),
            Some(ENEMY_KING)
        );

        // Friendly king should be None
        let wking = Piece::new(PieceType::King, PlayerColor::White);
        assert_eq!(
            get_piece_code(
                wking,
                true,
                1,
                PlayerColor::White,
                &promo_white,
                &promo_black
            ),
            None
        );
    }

    #[test]
    fn test_slider_threat_encoding() {
        // Friendly rook attacks enemy pawn at distance 1
        let feat = encode_slider_threat(true, false, 1, 0, 0, 0);
        assert!(feat < SLIDER_THREAT_FEATURES);
    }

    #[test]
    fn test_nnue_applicable() {
        let mut gs = GameState::new();
        gs.setup_variant(Variant::Classical);
        gs.recompute_piece_counts();

        assert!(is_nnue_applicable(&gs));
    }
}
