//! Per-piece reach metrics that mirror each fairy piece's movegen exactly:
//! what its distinctive movement actually attacks, covers, or is blocked by.
use crate::board::{Coordinate, Piece, PlayerColor};
use crate::game::GameState;
use crate::search::params::{huygen, knightrider, slider_threat_cap, slider_threat_div};

use super::base::{MAX_PHASE, get_piece_value_base};

const COMPOUND_LEAP_DEFEND: i32 = 4;
const COMPOUND_LEAP_ROYAL: i32 = 20;

const KNIGHTRIDER_ROYAL_ALIGN: i32 = 20;
const KNIGHTRIDER_DEFEND_BONUS: i32 = 4;
const KNIGHTRIDER_OPEN_RAY_EG: i32 = 2;

/// Distances beyond this are ignored, keeping the prime test on the O(1)
/// lookup path and the scan bounded on crowded lines.
const HUYGEN_SCAN_MAX: i64 = 120;
const HUYGEN_DEFEND_BONUS: i32 = 4;
const HUYGEN_ROYAL_ALIGN: i32 = 22;

/// A huygen jumps to PRIME distances along its orthogonals, hopping over anything
/// at a composite distance, but a piece sitting exactly at a prime distance stops
/// it for the rest of that direction. So open lines say nothing about it -- what
/// matters is the first prime-distance occupant of each of its four rays.
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

    let royal_at = |p: &Piece| -> bool { p.piece_type().is_royal() };

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
        if royal_at(&victim) {
            // Nothing can interpose at a composite distance, so this is a
            // standing check threat rather than a merely aligned one.
            attack += HUYGEN_ROYAL_ALIGN;
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
pub(crate) fn evaluate_compound_leap_threats(
    game: &GameState,
    x: i64,
    y: i64,
    own: PlayerColor,
    attacker_value: i32,
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };
    let mut attack = 0i32;
    let mut defend = 0i32;

    for (ox, oy) in crate::attacks::KNIGHT_OFFSETS {
        let Some(target) = game.board.get_piece(x + ox, y + oy) else {
            continue;
        };
        let tt = target.piece_type();
        if tt.is_neutral_type() {
            continue;
        }
        if target.color() == own {
            defend += COMPOUND_LEAP_DEFEND;
            continue;
        }
        if tt.is_royal() {
            attack += COMPOUND_LEAP_ROYAL;
            continue;
        }
        let v = get_piece_value_base(tt);
        // Unblockable, so even a lesser victim is a genuine threat and earns a floor.
        let raw = (v - attacker_value).max(v / 4);
        attack += (raw / slider_threat_div()).min(slider_threat_cap());
    }

    attack + taper(defend, defend / 2)
}

#[inline]
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Mirrors knightrider movegen: each ray stops at its closest occupant, and a
/// capture is emitted at any distance while quiets cap out. One gcd places a
/// piece on its ray, since k = gcd(|rx|, |ry|) for a reduced knight step.
pub(crate) fn evaluate_knightrider_reach(
    x: i64,
    y: i64,
    own: PlayerColor,
    piece_list: &[(i64, i64, Piece)],
    phase: i32,
) -> i32 {
    let taper =
        |mg: i32, eg: i32| -> i32 { ((mg * phase) + (eg * (MAX_PHASE - phase))) / MAX_PHASE };

    let mut best_k = [i64::MAX; 8];
    let mut best: [Option<Piece>; 8] = [None; 8];

    for &(px, py, other) in piece_list {
        let (rx, ry) = (px - x, py - y);
        if rx == 0 && ry == 0 {
            continue;
        }
        let g = gcd_i64(rx.abs(), ry.abs());
        if g == 0 {
            continue;
        }
        let (ux, uy) = (rx / g, ry / g);
        let (au, av) = (ux.abs(), uy.abs());
        if !((au == 1 && av == 2) || (au == 2 && av == 1)) {
            continue;
        }
        let slot = match (ux, uy) {
            (1, 2) => 0,
            (1, -2) => 1,
            (2, 1) => 2,
            (2, -1) => 3,
            (-1, 2) => 4,
            (-1, -2) => 5,
            (-2, 1) => 6,
            (-2, -1) => 7,
            _ => continue,
        };
        if g < best_k[slot] {
            best_k[slot] = g;
            best[slot] = Some(other);
        }
    }

    let mut attack = 0i32;
    let mut defend = 0i32;
    let mut open_rays = 0i32;

    for slot in best.iter() {
        let Some(occupant) = slot else {
            open_rays += 1;
            continue;
        };
        let ot = occupant.piece_type();
        if ot.is_neutral_type() {
            continue;
        }
        if occupant.color() == own {
            // The ray stops one leap short of it: cover, not a target.
            defend += KNIGHTRIDER_DEFEND_BONUS;
            continue;
        }
        if ot.is_royal() {
            attack += KNIGHTRIDER_ROYAL_ALIGN;
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

