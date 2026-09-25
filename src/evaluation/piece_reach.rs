//! Per-piece reach metrics that mirror each fairy piece's movegen exactly:
//! what its distinctive movement actually attacks, covers, or is blocked by.
use crate::board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::search::params::{huygen, knightrider, rose, slider_threat_cap, slider_threat_div};

use super::base::{MAX_PHASE, get_piece_value_base};
use crate::moves::knightrider_tile_mask;

const ROSE_DEFEND_BONUS: i32 = 4;

const COMPOUND_LEAP_DEFEND: i32 = 4;

const KNIGHTRIDER_DEFEND_BONUS: i32 = 4;
const KNIGHTRIDER_OPEN_RAY_EG: i32 = 2;

/// Distances beyond this are ignored, keeping the prime test on the O(1)
/// lookup path and the scan bounded on crowded lines.
const HUYGEN_SCAN_MAX: i64 = 120;
const HUYGEN_DEFEND_BONUS: i32 = 4;

/// A huygen jumps to PRIME distances along its orthogonals, hopping composite
/// gaps but stopped by a piece exactly at a prime distance -- so open lines say
/// nothing; what matters is the first prime-distance occupant per ray.
pub(crate) fn evaluate_huygen_reach(
    indices: &crate::moves::SpatialIndices,
    x: i64,
    y: i64,
    own: PlayerColor,
    enemy_royals: &[Coordinate],
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut attack = 0i32;
    let mut defend = 0i32;

    // First piece at a prime distance along one direction, or None if the ray
    // reaches nothing within the scan bound.
    let first_prime_hit = |line: Option<&crate::moves::SpatialLine>,
                           self_coord: i64,
                           forward: bool|
     -> Option<Piece> {
        let l = line?;
        let n = l.coords.len();
        let mut i = 0usize;
        while i < n {
            // Ascending for the forward ray, descending for the backward one, so
            // the first prime-distance occupant found is the nearest.
            let idx = if forward { i } else { n - 1 - i };
            i += 1;
            let c = l.coords[idx];
            let d = if forward { c - self_coord } else { self_coord - c };
            if d <= 0 {
                continue;
            }
            if d > HUYGEN_SCAN_MAX {
                break;
            }
            if crate::utils::is_prime_fast(d) {
                return Some(Piece::from_packed(l.pieces[idx]));
            }
        }
        None
    };


    let mut score_hit = |hit: Option<Piece>| {
        let Some(victim) = hit else { return };
        let vt = victim.piece_type();
        if vt.is_neutral_type() {
            return;
        }
        if victim.color() == own {
            defend += HUYGEN_DEFEND_BONUS;
            return;
        }
        let v = get_piece_value_base(vt);
        // A cheap attacker still generates a real threat, so a lesser victim
        // earns a floor rather than nothing.
        let raw = (v - huygen()).max(v / 4);
        attack += (raw / slider_threat_div()).min(slider_threat_cap());
    };

    let row = indices.rows.get(&y);
    let col = indices.cols.get(&x);
    score_hit(first_prime_hit(row, x, true));
    score_hit(first_prime_hit(row, x, false));
    score_hit(first_prime_hit(col, y, true));
    score_hit(first_prime_hit(col, y, false));

    let _ = enemy_royals;

    // Defence matters less as material leaves; the attacking half does not decay.
    attack + taper(defend, defend / 2)
}

/// A compound piece leaps like a knight as well as sliding, and no blocker can
/// stop the leap. The slider threat scan only walks lines, so that half of its
/// attacks is invisible: this prices the eight knight squares it really covers.
pub(crate) fn evaluate_leap_threats(
    game: &GameState,
    x: i64,
    y: i64,
    own: PlayerColor,
    attacker_value: i32,
    phase: i32,
    offsets: &[(i64, i64)],
    defend_unit: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut attack = 0i32;
    let mut defend = 0i32;

    for (ox, oy) in offsets.iter().copied() {
        let Some(target) = game.board.get_piece(x + ox, y + oy) else {
            continue;
        };
        let tt = target.piece_type();
        if tt.is_neutral_type() {
            continue;
        }
        if target.color() == own {
            defend += defend_unit;
            continue;
        }
        let v = get_piece_value_base(tt);
        // Unblockable, so even a lesser victim is a genuine threat and earns a floor.
        let raw = (v - attacker_value).max(v / 4);
        attack += (raw / slider_threat_div()).min(slider_threat_cap());
    }

    attack + taper(defend, defend / 2)
}

/// Mirrors rose movegen: sixteen spirals of up to seven hops, each stopping at
/// its first occupant, and squares recur across spirals so a target is counted
/// once. Bounded reach means a blocked spiral yields nothing past the blocker.
pub(crate) fn evaluate_rose_reach(
    game: &GameState,
    x: i64,
    y: i64,
    own: PlayerColor,
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut attack = 0i32;
    let mut defend = 0i32;
    // At most one target per spiral, so sixteen slots cover the dedup.
    let mut seen: [(i64, i64); 16] = [(i64::MIN, i64::MIN); 16];
    let mut seen_n = 0usize;

    for spirals_for_dir in &crate::moves::ROSE_SPIRALS {
        for spiral_path in spirals_for_dir {
            for &(cum_dx, cum_dy) in spiral_path.iter() {
                let (tx, ty) = (x + cum_dx, y + cum_dy);
                let Some(occupant) = game.board.get_piece(tx, ty) else {
                    continue;
                };
                if seen[..seen_n].contains(&(tx, ty)) {
                    break;
                }
                seen[seen_n] = (tx, ty);
                seen_n += 1;

                let ot = occupant.piece_type();
                if !ot.is_neutral_type() {
                    if occupant.color() == own {
                        defend += ROSE_DEFEND_BONUS;
                    } else {
                        let v = get_piece_value_base(ot);
                        let raw = (v - rose()).max(v / 4);
                        attack += (raw / slider_threat_div()).min(slider_threat_cap());
                    }
                }
                break;
            }
        }
    }

    attack + taper(defend, defend / 2)
}

/// The nearest occupant of each of a knightrider's eight rays.
pub(crate) type RiderRays = [Option<Piece>; 8];

/// Riders past this many fall back to one traversal each, keeping the batch
/// scratch on the stack.
pub(crate) const MAX_BATCHED_RIDERS: usize = 8;

/// Fills `out[i]` with the rays of `riders[i]` in one board traversal, so a
/// position with several riders does not re-walk every piece for each of them.
pub(crate) fn fill_knightrider_rays(riders: &[(i64, i64)], board: &Board, out: &mut [RiderRays]) {
    debug_assert_eq!(riders.len(), out.len());
    let mut best_k = [[i64::MAX; 8]; MAX_BATCHED_RIDERS];
    for rays in out.iter_mut() {
        *rays = [None; 8];
    }

    // Every occupant stops a ray, as in movegen; pawns and neutrals are scored
    // as blockers only, so the officer-calibrated credits below stay unchanged.
    for (cx, cy, tile) in board.tiles.iter() {
        for (i, &(x, y)) in riders.iter().enumerate() {
            let on_lines =
                knightrider_tile_mask((cx * 8).wrapping_sub(x), (cy * 8).wrapping_sub(y));
            let mut bits = tile.occ_all & on_lines.unwrap_or(!0);
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (px, py) = (cx * 8 + (idx % 8) as i64, cy * 8 + (idx / 8) as i64);
                let other = Piece::from_packed(tile.piece[idx]);
                let (rx, ry) = (px - x, py - y);
                let (dx, dy) = (rx.abs(), ry.abs());

                // Only check pieces that are on knightrider's rays.
                if dx == 0 || (dx * 2 != dy && dx != dy * 2) {
                    continue;
                }

                let slot = 4 * (rx > 0) as usize + 2 * (ry > 0) as usize + (dx > dy) as usize;

                // It only needs to check the minimum dx since it's always a multiple
                // of either 1 or 2 depending on the slot.
                if dx < best_k[i][slot] {
                    best_k[i][slot] = dx;
                    out[i][slot] = Some(other);
                }
            }
        }
    }
}

/// Mirrors knightrider movegen: each ray stops at its closest occupant, and a
/// capture is emitted at any distance while quiets cap out. One gcd places a
/// piece on its ray, since k = gcd(|rx|, |ry|) for a reduced knight step.
pub(crate) fn knightrider_rays(x: i64, y: i64, board: &Board) -> RiderRays {
    let mut out = [[None; 8]; 1];
    fill_knightrider_rays(&[(x, y)], board, &mut out);
    out[0]
}

/// Scores one rider from its precomputed rays.
pub(crate) fn score_knightrider_rays(rays: &RiderRays, own: PlayerColor, phase: i32) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };

    let mut attack = 0i32;
    let mut defend = 0i32;
    let mut open_rays = 0i32;

    for slot in rays.iter() {
        let Some(occupant) = slot else {
            open_rays += 1;
            continue;
        };
        let ot = occupant.piece_type();
        if ot.is_neutral_type() || ot == PieceType::Pawn {
            continue;
        }
        if occupant.color() == own {
            // The ray stops one leap short of it: cover, not a target.
            defend += KNIGHTRIDER_DEFEND_BONUS;
            continue;
        }
        let v = get_piece_value_base(ot);
        let raw = (v - knightrider()).max(v / 4);
        attack += (raw / slider_threat_div()).min(slider_threat_cap());
    }

    // Nearly every ray is empty on a sparse board, so an open one is priced as a
    // token, and only in the endgame where the rider has room to use it.
    attack + taper(defend, defend / 2) + taper(0, open_rays * KNIGHTRIDER_OPEN_RAY_EG)
}

/// Compound pieces threaten along the knight offsets they carry.
pub(crate) fn evaluate_compound_leap_threats(
    game: &GameState,
    x: i64,
    y: i64,
    own: PlayerColor,
    attacker_value: i32,
    phase: i32,
) -> i32 {
    evaluate_leap_threats(
        game,
        x,
        y,
        own,
        attacker_value,
        phase,
        &crate::attacks::KNIGHT_OFFSETS,
        COMPOUND_LEAP_DEFEND,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knightrider_tile_mask_is_exactly_the_lines() {
        for wx in -40i64..40 {
            for wy in -40i64..40 {
                let mut expect = 0u64;
                for idx in 0..64 {
                    let (dx, dy) = ((wx + idx % 8).abs(), (wy + idx / 8).abs());
                    if dx * 2 == dy || dx == dy * 2 {
                        expect |= 1 << idx;
                    }
                }
                assert_eq!(knightrider_tile_mask(wx, wy), Some(expect), "offset ({wx}, {wy})");
            }
        }
        assert_eq!(knightrider_tile_mask(1 << 60, 0), None);
        assert_eq!(knightrider_tile_mask(0, i64::MIN), None);
    }

    fn white_rider_reach(icn: &str) -> i32 {
        let mut g = GameState::new();
        g.setup_position_from_icn(icn);
        // Endgame phase so the open-ray token is live as well as the attack.
        let rays = knightrider_rays(0, 0, &g.board);
        score_knightrider_rays(&rays, PlayerColor::White, 0)
    }

    /// The rider's (2,1) ray meets the queen at (4,2) only via (2,1); any
    /// occupant there stops the ray exactly as movegen does.
    #[test]
    fn test_knightrider_ray_is_blocked_by_pawns_and_neutrals() {
        let open = white_rider_reach("w (8;q|1;q) K-10,-10|k20,20|NR0,0|q4,2");
        for blocker in ["P2,1", "p2,1", "OB2,1", "VO2,1"] {
            let icn = format!("w (8;q|1;q) K-10,-10|k20,20|NR0,0|q4,2|{blocker}");
            let blocked = white_rider_reach(&icn);
            assert!(blocked < open, "{blocker}: open {open}, blocked {blocked}");
        }
    }
}
