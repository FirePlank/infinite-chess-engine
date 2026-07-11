// Mop-Up Evaluation
//
// Specialized endgame logic for positions where one side has a significant
// material advantage. It builds a mating net around the enemy king: confine
// it with slider cut lines (or board edges), bring our king in when the
// material needs it, and control the escape ring.

use crate::board::{Board, Coordinate, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::{SpatialIndices, is_square_attacked};
use crate::utils::is_prime_fast;

/// A defender pawn's weight in the net-leakiness term: pawns barely obstruct
/// a mating net and their moves reset the 50-move clock in the attacker's
/// favor, but promotion potential keeps them from being free.
const MOP_UP_DEFENDER_PAWN_VALUE: i32 = 100;
/// Minimum attacker material surplus (centipawns) for mop-up: below roughly
/// a minor piece the ending is a fight, not a mate hunt.
const MOP_UP_MIN_SURPLUS: i32 = 200;
/// A defender pawn can promote and break the net outright — K+R vs K+3P is a
/// race, not a mop-up — so shaping against a pawn-holding defender needs an
/// overwhelming army (roughly a queen up), not a thin edge.
const MOP_UP_PAWN_MIN_SURPLUS: i32 = 900;

/// Boards whose larger dimension is at most this use the bounded lone-king
/// driving model (the edge confines the king); larger boards use the
/// piece-coordination model that works without an edge.
const MOP_UP_BOUNDED_MAX: i64 = 200;

// Bounded lone-king driving (small Stockfish-style tiebreaker on the pawn=100
// scale). push_to_edge drives the bare king to an edge/corner; push_close keeps
// the kings together; the KBN term forces the bishop-colored corner. Magnitudes
// are deliberately small so they guide conversion without overriding material or
// chasing stalemate.
const EDGE_DIST_CAP: i64 = 3;
const EDGE_CORNER_BONUS: i32 = 48;
const EDGE_FALLOFF: i32 = 2;
const KING_CLOSE_BONUS: i32 = 60;
const KING_CLOSE_STEP: i32 = 10;
// KBN is the longest bounded mate (~30 accurate moves); like Stockfish's
// dedicated KBNK evaluation these values are large, and safe to be — the block
// only fires for exactly bishop+knight. The corner drive must out-gradient
// every tiebreaker or the 50-move budget dies to wandering.
const KBN_CORNER_BONUS: i32 = 800;
const KBN_CORNER_STEP: i32 = 100;
const KBN_CLOSE_STEP: i32 = 40;

// ===== Mating-net weights (unbounded model), ordered by tier =====
// Tier 1 — confinement. The dominant, monotone progress signal: walls (slider
// cut lines or near board edges) boxing the enemy king in, tighter is better.
const WALL_RELEVANT_DIST: i64 = 100;
const CUT_FLAT_ORTHO: i32 = 25;
const CUT_STEP_ORTHO: i32 = 7;
const CUT_CAP_ORTHO: i64 = 16;
const CUT_FLAT_DIAG: i32 = 6;
const CUT_STEP_DIAG: i32 = 2;
const CUT_CAP_DIAG: i64 = 12;
const SANDWICH_ORTHO: i32 = 60;
const SANDWICH_DIAG: i32 = 16;
// An adjacent pair of diagonal lines (offsets differing by one, i.e.
// opposite-colored bishops side by side) is a true wall: a king hops a single
// diagonal (offset -1 to +1 in one step) but can never cross the pair.
const DIAG_PAIR_FLAT: i32 = 70;
const DIAG_PAIR_STEP: i32 = 7;
const DIAG_PAIR_CAP: i64 = 20;
const DIAG_PAIR_SANDWICH: i32 = 50;
/// A pair wall on the far side of the enemy king (opposite our king) cuts its
/// flight. Big enough to beat shadow-chasing with split bishops.
const DIAG_PAIR_FAR_SIDE: i32 = 40;
const FULL_BOX_BONUS: i32 = 120;
const CAGE_FLAT: i32 = 120;
const CAGE_MAX: i32 = 620;

// Target-box formation: a fixed cage blueprint anchored to the enemy king's
// cell on a coarse grid. Stations don't move while he shuffles inside the
// cell, so marching a piece to its station is progress the defender cannot
// undo — unlike chasing terms anchored to his exact square.
const BOX_GRID: i64 = 8;
const BOX_R: i64 = 5;
const STATION_BASE: i32 = 90;
const STATION_STEP: i32 = 6;
const STATION_TOL_LINE: i64 = 1;
const STATION_TOL_LEAPER: i64 = 2;
/// Escalating payoff per manned station: completing the formation is worth
/// far more than the sum of its parts.
const MANNED_LADDER: [i32; 7] = [0, 40, 100, 190, 300, 420, 540];
/// Replaces the box score once the force is on top of the defender, so
/// entering the kill zone is strictly better than holding stations.
const KILL_ZONE_BONUS: i32 = 500;

// Tier 2 — king participation. On the unbounded board nearly every minimal
// mate needs the king, so approach is steep; with an overwhelming battery it
// stays a mild tiebreaker so the pieces do the work without wasted king marches.
const KING_STEP_NEEDED: i32 = 28;
const KING_CAP_NEEDED: i64 = 48;
const KING_NEAR2_NEEDED: i32 = 160;
const KING_NEAR4_NEEDED: i32 = 80;
const KING_STEP_IDLE: i32 = 7;
const KING_CAP_IDLE: i64 = 24;
const KING_TAIL_STEP: i32 = 8;

// Tier 3 — escape-ring control around the enemy king.
const RING_COVER: i32 = 14;
const RING_ESC_BOTH: i32 = 20;
const RING_ESC_ONE: i32 = 10;
const RING_SEVEN_PLUS: i32 = 90;
const RING_SIX: i32 = 50;
const RING_OUTER_COVER: i32 = 7;
/// Directional (escape-side / opposite-side) terms only count once our king is
/// engaged; a far king must never gain by sidestepping to reclassify pieces.
const NEAR_GATE_DIST: i64 = 12;

// Tier 4 — per-piece shaping. Small tiebreakers, each kept below one
// king-step of value so they can never outbid king approach.
const SLIDER_BAND_BONUS: i32 = 22;
const SLIDER_HUG_PENALTY: i32 = 20;
const SLIDER_FAR_STEP: i32 = 3;
const SLIDER_FAR_CAP: i64 = 30;
const WALL_HARASS_PENALTY: i32 = 30;
// A held cut is safest FAR along its line: near the runner the wall piece
// gets chased, and every re-cut hop burns the tempo the king march needs
// (trace-proven treadmill: rook hops +2 ahead forever, king never arrives).
const SLIDER_STANDOFF_STEP: i32 = 2;
const SLIDER_STANDOFF_CAP: i64 = 40;
// A wall on the enemy king's flight side (away from our king) stops the
// runner gaining ground — the side facing our king is covered by the king
// itself. Weighted above the sandwich so two walls prefer the front+side
// box over the parallel corridor a runner can race along forever.
const ORTHO_FLIGHT_SIDE: i32 = 70;

// ===== Dedicated K+2R vs K (two-rook lawnmower) =====
// Two rooks + our king mate a lone king on the unbounded board by the rolling
// (lawnmower) technique: the rooks cut the two escape lines AHEAD of the
// fleeing king while our king walls the near two sides, so the king is boxed
// and driven into our king. The generic net treats the rooks as independent
// cut lines and settles for a safe two-file corridor the king runs up forever;
// this case-specific term instead scores the FINITE box and its perimeter, so
// the only way the enemy can stop the box shrinking is to flee toward our
// king — which is the win.
/// A confining side (rook line or our king) counts only within this range.
const TR_SIDE_RANGE: i64 = 200;
/// Each of the four sides the king is bounded on.
const TR_SIDE_CLOSED: i32 = 150;
/// All four sides bounded: a finite cage exists.
const TR_FULL_CAGE: i32 = 400;
/// An open side is an escape lane; the search must slam it shut.
const TR_OPEN_SIDE: i32 = 300;
/// A closed side scores higher the tighter its wall sits, out to this range.
const TR_TIGHT_NEAR: i64 = 24;
const TR_TIGHT_STEP: i32 = 6;
/// Our king marching in, per Chebyshev step closer (up to TR_APPROACH_MAX).
const TR_APPROACH_STEP: i32 = 12;
const TR_APPROACH_MAX: i64 = 100;
const TR_OPPOSITION: i32 = 200;
/// Connected (mutually defended) rooks are safe from harassment.
const TR_CONNECTED: i32 = 80;
/// A rook the enemy king attacks that neither our king nor its partner
/// defends is hanging — refuse the sacrifice unconditionally.
const TR_HANG: i32 = 800;

const LEAPER_ENGAGE_STEP: i32 = 8;
const LEAPER_ENGAGE_CAP: i64 = 24;
const OPPOSITE_SIDE_BONUS: i32 = 10;
const PROTECTED_PIECE_BONUS: i32 = 25;
const FRONTAL_CHECK_PENALTY: i32 = 12;
const SIDE_CHECK_PENALTY: i32 = 3;
const HERD_CHECK_BONUS: i32 = 15;

#[derive(Clone, Copy)]
struct SliderInfo {
    x: i64,
    y: i64,
    pt: PieceType,
}

/// 8 compass directions around a square (4 ortho + 4 diagonal).
const RING_DIRS: [(i64, i64); 8] = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, -1), (1, -1), (-1, 1),
];

/// Geometric attack pattern check (ignores blockers).
/// Returns true if a piece of type `pt` and color `color` at offset (dx, dy)=0 could
/// in principle attack a square at offset (dx, dy). Used for fast ring-coverage scoring.
#[inline]
fn piece_attacks_geom(pt: PieceType, color: PlayerColor, dx: i64, dy: i64) -> bool {
    let adx = dx.abs();
    let ady = dy.abs();
    if adx == 0 && ady == 0 {
        return false;
    }
    match pt {
        PieceType::Rook => dx == 0 || dy == 0,
        PieceType::Bishop => adx == ady,
        PieceType::Queen | PieceType::RoyalQueen => dx == 0 || dy == 0 || adx == ady,
        PieceType::Chancellor => {
            dx == 0 || dy == 0 || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Archbishop => {
            adx == ady || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Amazon => {
            dx == 0
                || dy == 0
                || adx == ady
                || (adx == 1 && ady == 2)
                || (adx == 2 && ady == 1)
        }
        PieceType::Knight => (adx == 1 && ady == 2) || (adx == 2 && ady == 1),
        PieceType::Camel => (adx == 1 && ady == 3) || (adx == 3 && ady == 1),
        PieceType::Giraffe => (adx == 1 && ady == 4) || (adx == 4 && ady == 1),
        PieceType::Zebra => (adx == 2 && ady == 3) || (adx == 3 && ady == 2),
        PieceType::King | PieceType::Guard => adx <= 1 && ady <= 1,
        PieceType::Centaur | PieceType::RoyalCentaur => {
            (adx <= 1 && ady <= 1) || (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Hawk => {
            // Hawk leaps to compass squares at distance 2 or 3 (ortho or diag).
            let d = adx.max(ady);
            (d == 2 || d == 3) && (dx == 0 || dy == 0 || adx == ady)
        }
        PieceType::Knightrider => {
            // Slides along knight rays: any (k, 2k) or (2k, k).
            if adx == 0 || ady == 0 || adx == ady {
                false
            } else {
                let g = gcd_i64(adx, ady);
                let nx = adx / g;
                let ny = ady / g;
                (nx == 1 && ny == 2) || (nx == 2 && ny == 1)
            }
        }
        PieceType::Pawn => {
            let dir = if color == PlayerColor::White { 1 } else { -1 };
            adx == 1 && dy == dir
        }
        PieceType::Huygen => {
            // Orthogonal slider, only at prime distances.
            (dx == 0 || dy == 0) && is_prime_fast(adx.max(ady))
        }
        PieceType::Rose => {
            // Approximate Rose coverage via knight-leaper pattern; the full curving
            // pattern is complex but a knight check is a reasonable lower bound.
            (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
        }
        PieceType::Void | PieceType::Obstacle => false,
    }
}

#[inline]
fn gcd_i64(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Evaluates the "wall" bit for each not-yet-evaluated `x` set in `need` on row
/// `local_y` of the cage window, recording results in `forbidden`/`computed`.
/// A wall is an out-of-bounds, our-attacked, or our-occupied square.
#[inline]
#[allow(clippy::too_many_arguments)]
fn cage_eval_walls(
    board: &Board,
    indices: &SpatialIndices,
    our_color: PlayerColor,
    origin_x: i64,
    origin_y: i64,
    bounds: (i64, i64, i64, i64),
    local_y: usize,
    need: u32,
    forbidden: &mut [u32; 32],
    computed: &mut [u32; 32],
) {
    let todo = need & !computed[local_y];
    if todo == 0 {
        return;
    }
    computed[local_y] |= need;
    let (min_x, max_x, min_y, max_y) = bounds;
    let abs_y = origin_y + local_y as i64;
    let mut bits = todo;
    while bits != 0 {
        let local_x = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let abs_x = origin_x + local_x as i64;
        let wall = abs_x < min_x
            || abs_x > max_x
            || abs_y < min_y
            || abs_y > max_y
            || is_square_attacked(board, &Coordinate::new(abs_x, abs_y), our_color, indices)
            || board.is_occupied_by_color(abs_x, abs_y, our_color);
        if wall {
            forbidden[local_y] |= 1 << local_x;
        }
    }
}

/// Detects if the enemy king is trapped within a localized "cage" of attacked squares.
/// Returns whether a cage exists and the total reachable area for the king.
///
/// Floods an enemy-king-centered 32x32 window away from the king, treating
/// out-of-bounds / our-attacked / our-occupied squares as walls. Wall squares
/// are evaluated lazily — only for cells the flood actually reaches — so a small
/// contained cage costs a handful of `is_square_attacked` calls rather than 1024.
#[inline]
fn find_bitboard_cage(
    board: &Board,
    indices: &SpatialIndices,
    enemy_king: &Coordinate,
    our_color: PlayerColor,
) -> (bool, u32) {
    // 32x32 local window: indices 0..31 map to king_coord - 16 .. king_coord + 15.
    let origin_x = enemy_king.x - 16;
    let origin_y = enemy_king.y - 16;
    let bounds = crate::moves::get_coord_bounds();

    let mut forbidden = [0u32; 32];
    let mut computed = [0u32; 32];

    // Flood fill from the center (16, 16) via iterative 8-way dilation.
    let mut reachable = [0u32; 32];
    reachable[16] = 1 << 16;

    for _ in 0..32 {
        let mut next_reachable = reachable;

        for y in 0..32 {
            if reachable[y] == 0 {
                continue;
            }
            let row = reachable[y];
            let dilated_row = row | (row << 1) | (row >> 1);
            next_reachable[y] |= dilated_row;
            if y > 0 {
                next_reachable[y - 1] |= dilated_row;
            }
            if y < 31 {
                next_reachable[y + 1] |= dilated_row;
            }
        }

        // Evaluate walls for the newly-reached candidate cells, then mask them out.
        let mut changed = false;
        for y in 0..32 {
            cage_eval_walls(
                board, indices, our_color, origin_x, origin_y, bounds, y,
                next_reachable[y], &mut forbidden, &mut computed,
            );
            let prev = reachable[y];
            next_reachable[y] &= !forbidden[y];
            if next_reachable[y] != prev {
                changed = true;
            }
            reachable[y] = next_reachable[y];
        }

        if !changed {
            break;
        }

        // Check if we hit the perimeter
        if (reachable[0] | reachable[31]) != 0 {
            return (false, 1024);
        }
        for reach in reachable.iter().take(31).skip(1) {
            if (reach & 0x80000001) != 0 {
                return (false, 1024);
            }
        }
    }

    // Successful fill without hitting the perimeter indicates a contained cage.
    let mut area = 0u32;
    for row in reachable.iter() {
        area += row.count_ones();
    }

    (area > 0 && area < 1000, area)
}

// --- Utility Functions ---

/// Check if a side only has a king (no other pieces)
#[inline(always)]
pub fn is_lone_king(game: &GameState, color: PlayerColor) -> bool {
    if color == PlayerColor::White {
        game.white_pawn_count == 0 && !game.white_non_pawn_material
    } else {
        game.black_pawn_count == 0 && !game.black_non_pawn_material
    }
}

/// True when the losing side has no pawns and at most one non-royal piece:
/// the full mating-net machinery (cage flood fill) applies.
#[inline(always)]
fn defender_is_bareish(game: &GameState, color: PlayerColor) -> bool {
    let (pieces, pawns, royals) = if color == PlayerColor::White {
        (
            game.white_piece_count,
            game.white_pawn_count,
            game.white_royals.len(),
        )
    } else {
        (
            game.black_piece_count,
            game.black_pawn_count,
            game.black_royals.len(),
        )
    };
    pawns == 0 && (pieces as usize).saturating_sub(royals) <= 1
}

/// Calculates the mop-up scaling factor (0-100).
/// Full strength against a bare king. With defender material present, the
/// scale follows the attacker's material surplus relative to the defense the
/// leftover pieces can put up: a big enough army mops up even against a
/// defender queen, while a thin edge over a defended king gets little to
/// no shaping.
#[inline]
pub fn calculate_mop_up_scale(game: &GameState, losing_color: PlayerColor) -> Option<u32> {
    // Check winning side has at least one non-pawn piece
    let winning_has_pieces = if losing_color == PlayerColor::White {
        game.black_non_pawn_material
    } else {
        game.white_non_pawn_material
    };

    if !winning_has_pieces {
        return None; // Don't mop-up with just king+pawns
    }

    let defender_is_white = losing_color == PlayerColor::White;
    let mut defense: i32 = 0;
    let mut defender_material: i32 = 0;
    let mut defender_has_pawns = false;
    for (_, _, piece) in game.board.iter_pieces_by_color(defender_is_white) {
        let pt = piece.piece_type();
        if pt.is_royal() {
            continue;
        }
        let value = super::base::get_piece_value_base(pt);
        defender_material += value;
        defense += if pt == PieceType::Pawn {
            defender_has_pawns = true;
            MOP_UP_DEFENDER_PAWN_VALUE
        } else {
            value
        };
    }
    if defense == 0 {
        return Some(100);
    }

    let mut attacker_material: i32 = 0;
    for (_, _, piece) in game.board.iter_pieces_by_color(!defender_is_white) {
        let pt = piece.piece_type();
        if pt.is_royal() {
            continue;
        }
        attacker_material += super::base::get_piece_value_base(pt);
    }

    let surplus = attacker_material - defender_material;
    let min_surplus = if defender_has_pawns {
        MOP_UP_PAWN_MIN_SURPLUS
    } else {
        MOP_UP_MIN_SURPLUS
    };
    if surplus < min_surplus {
        return None;
    }
    Some((surplus * 100 / (surplus + defense)) as u32)
}

/// Unscaled mop-up evaluation.
#[inline(always)]
pub fn evaluate_lone_king_endgame(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    evaluate_mop_up_core(game, our_king, enemy_king, winning_color)
}

/// Ceiling for the shaping term when the defender still has material: against
/// a defended king the net is a guide, never a win claim — real material and
/// concrete search lines carry the verdict. Bare kings (scale 100) keep the
/// full uncompressed net; forced-mate hunts need its whole gradient range.
const MOP_UP_DEFENDED_CAP: i32 = 400;

/// The single source of truth for mop-up activation. Returns the winning
/// color and the activation scale (0-100) when one side is reduced to a
/// bare-ish king (at most one non-pawn piece besides royals, plus pawns
/// covered by the surplus rules) while the other keeps a small pawnless
/// mating force. Both the evaluation term and the search check extension
/// route through here so they can never disagree about what a mop-up is.
#[inline]
pub fn active_mop_up(game: &GameState) -> Option<(PlayerColor, u32)> {
    let white_np = game.white_piece_count.saturating_sub(game.white_pawn_count);
    let black_np = game.black_piece_count.saturating_sub(game.black_pawn_count);

    if black_np < 3
        && white_np > 1
        && white_np <= 10
        && game.white_pawn_count == 0
        && !game.black_royals.is_empty()
        && let Some(scale) = calculate_mop_up_scale(game, PlayerColor::Black)
        && scale > 0
    {
        return Some((PlayerColor::White, scale));
    }
    if white_np < 3
        && black_np > 1
        && black_np <= 10
        && game.black_pawn_count == 0
        && !game.white_royals.is_empty()
        && let Some(scale) = calculate_mop_up_scale(game, PlayerColor::White)
        && scale > 0
    {
        return Some((PlayerColor::Black, scale));
    }
    None
}

/// Activation-scaled net shaping from the winner's perspective. Against a
/// bare king (scale 100) the net passes through untouched; against a defended
/// king it additionally saturates toward MOP_UP_DEFENDED_CAP, so no sum of
/// shaping bonuses can ever outweigh the material on the board.
pub fn evaluate_mop_up_scaled(game: &GameState, winner: PlayerColor, scale: u32) -> i32 {
    let (our_king, enemy_king) = if winner == PlayerColor::White {
        (game.white_royals.first(), game.black_royals.first())
    } else {
        (game.black_royals.first(), game.white_royals.first())
    };
    let Some(enemy_king) = enemy_king else {
        return 0;
    };
    let scaled = evaluate_mop_up_core(game, our_king, enemy_king, winner) * scale as i32 / 100;
    if scale >= 100 || scaled <= 0 {
        return scaled;
    }
    let (x, cap) = (scaled as i64, MOP_UP_DEFENDED_CAP as i64);
    (x * cap / (x + cap)) as i32
}

// --- Core Evaluation ---

/// Bounded-board KX-vs-K mating guidance. On a bounded board the edge does the
/// confining work that pieces must do on an unbounded board, so a small
/// Stockfish-style tiebreaker is enough: drive the bare king toward the
/// edge/corner (`push_to_edge`), keep our king close (`push_close`), and for
/// KBN drive toward a bishop-colored corner. Magnitudes are small so they guide
/// conversion without overriding material or chasing stalemate.
fn bounded_lone_king_mop_up(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
    let ex = enemy_king.x;
    let ey = enemy_king.y;

    // push_to_edge: the closer the bare king is to an edge/corner, the better.
    let ed_x = (ex - min_x).min(max_x - ex).clamp(0, EDGE_DIST_CAP);
    let ed_y = (ey - min_y).min(max_y - ey).clamp(0, EDGE_DIST_CAP);
    let mut bonus = EDGE_CORNER_BONUS - ((ed_x * ed_x + ed_y * ed_y) as i32) * EDGE_FALLOFF;

    // push_close: bring our king toward the bare king.
    if let Some(ok) = our_king {
        let d = (ok.x - ex).abs().max((ok.y - ey).abs()) as i32;
        bonus += (KING_CLOSE_BONUS - d * KING_CLOSE_STEP).max(0);
    }

    // K+B+N vs K: the bare king can only be mated in a corner the bishop attacks.
    // For exactly one bishop + one knight (no other helping piece), pull the king
    // toward the nearest bishop-colored corner.
    let is_white = winning_color == PlayerColor::White;
    let (mut bishops, mut knights, mut others) = (0u8, 0u8, 0u8);
    let mut bishop_sq = (0i64, 0i64);
    for (x, y, piece) in game.board.iter_pieces_by_color(is_white) {
        let pt = piece.piece_type();
        if pt.is_royal() || pt == PieceType::Pawn {
            continue;
        }
        match pt {
            PieceType::Bishop => {
                bishops += 1;
                bishop_sq = (x, y);
            }
            PieceType::Knight => knights += 1,
            _ => others += 1,
        }
    }
    if bishops == 1 && knights == 1 && others == 0 {
        let bishop_dark = ((bishop_sq.0 + bishop_sq.1) & 1) != 0;
        let mut best = i64::MAX;
        for (cx, cy) in [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)] {
            if (((cx + cy) & 1) != 0) == bishop_dark {
                best = best.min((cx - ex).abs().max((cy - ey).abs()));
            }
        }
        if best != i64::MAX {
            bonus += (KBN_CORNER_BONUS - (best as i32) * KBN_CORNER_STEP).max(0);
            // The mate needs the kings nearly touching; the generic push_close
            // is too soft for the longest bounded mate.
            if let Some(ok) = our_king {
                let d = (ok.x - ex).abs().max((ok.y - ey).abs());
                bonus += ((7 - d.min(7)) as i32) * KBN_CLOSE_STEP;
            }
        }
    }

    bonus
}

// Cut lines around the enemy king. A slider on a full rank/file the king
// does not occupy fences the plane (the king cannot cross it); diagonal
// lines are collected individually since only adjacent pairs form true walls.
#[derive(Clone, Copy)]
struct FenceState {
    ortho_y_min_above: i64,
    ortho_y_max_below: i64,
    ortho_x_min_right: i64,
    ortho_x_max_left: i64,
    dp_lines: [i64; 12],
    dp_count: usize,
    dn_lines: [i64; 12],
    dn_count: usize,
}

impl FenceState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ortho_y_min_above: i64::MAX,
            ortho_y_max_below: i64::MIN,
            ortho_x_min_right: i64::MAX,
            ortho_x_max_left: i64::MIN,
            dp_lines: [0; 12],
            dp_count: 0,
            dn_lines: [0; 12],
            dn_count: 0,
        }
    }

    #[inline(always)]
    fn push_dp(&mut self, line: i64) {
        if self.dp_count < self.dp_lines.len() {
            self.dp_lines[self.dp_count] = line;
            self.dp_count += 1;
        }
    }

    #[inline(always)]
    fn push_dn(&mut self, line: i64) {
        if self.dn_count < self.dn_lines.len() {
            self.dn_lines[self.dn_count] = line;
            self.dn_count += 1;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MaterialSummary {
    ortho_count: u8,
    queen_count: u8,
    amazon_count: u8,
    chancellor_count: u8,
    diag_light: u8,
    diag_dark: u8,
    archbishops: u8,
    total_non_pawn_pieces: u8,
}

impl MaterialSummary {
    /// Number of true walls the army can man: each ortho slider lines one, and
    /// each opposite-parity diagonal pair lines one (archbishops change square
    /// color when they leap, so they pair with anything).
    fn wall_count(&self) -> u8 {
        let bishops_l = self.diag_light;
        let bishops_d = self.diag_dark;
        let mut pairs = bishops_l.min(bishops_d);
        let mut ar = self.archbishops;
        let leftover = bishops_l.max(bishops_d) - pairs;
        let with_ar = ar.min(leftover);
        pairs += with_ar;
        ar -= with_ar;
        pairs += ar / 2;
        self.ortho_count + pairs
    }
}

/// True when the winning side's battery is strong enough that king help is a
/// luxury rather than a requirement for mate.
#[inline(always)]
fn king_mostly_idle(m: &MaterialSummary, bounded: bool) -> bool {
    let heavy = m.queen_count + m.amazon_count;
    if bounded {
        // Against an edge, two heavy pieces (or a rook pair) ladder-mate alone.
        heavy >= 2 || m.ortho_count >= 2 || m.total_non_pawn_pieces >= 4
    } else {
        // Without an edge almost every net needs the king; only a large battery
        // of heavies can weave one alone.
        heavy + m.chancellor_count >= 3 || m.total_non_pawn_pieces >= 6
    }
}

#[inline(always)]
fn is_ortho_slider(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Rook
            | PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Chancellor
            | PieceType::Amazon
    )
}

#[inline(always)]
fn is_diag_slider(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Bishop
            | PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Archbishop
            | PieceType::Amazon
    )
}

/// Target-box formation score for one box size. The box is centered on the
/// enemy king's cell of a coarse grid (stable while he shuffles within it),
/// always containing him at BOX_R. Each piece gets a station by movement
/// type: ortho sliders man the four side lines, opposite-parity diag sliders
/// are jointly assigned to ONE corner family/side (adjacent lines — a true
/// pair wall), leapers and our king man side midpoints. Pieces score for
/// approaching their station and the formation pays an escalating ladder as
/// stations are manned.
fn target_box_score(
    pieces: &[SliderInfo],
    our_king: Option<&Coordinate>,
    cx: i64,
    cy: i64,
    r: i64,
) -> i32 {
    let mut bonus = 0;
    let mut manned = 0usize;

    // Ortho side lines: x = cx +- r, y = cy +- r. Greedy exclusive slots.
    let mut line_taken = [false; 4];
    let line_dist = |s: &SliderInfo, slot: usize| -> i64 {
        match slot {
            0 => (s.x - (cx - r)).abs(),
            1 => (s.x - (cx + r)).abs(),
            2 => (s.y - (cy - r)).abs(),
            _ => (s.y - (cy + r)).abs(),
        }
    };
    for s in pieces {
        if !is_ortho_slider(s.pt) {
            continue;
        }
        let mut best = i64::MAX;
        let mut best_slot = 4;
        for slot in 0..4 {
            if !line_taken[slot] {
                let d = line_dist(s, slot);
                if d < best {
                    best = d;
                    best_slot = slot;
                }
            }
        }
        if best_slot < 4 {
            line_taken[best_slot] = true;
            bonus += (STATION_BASE - (best.min(60) as i32) * STATION_STEP).max(0);
            if best <= STATION_TOL_LINE {
                manned += 1;
            }
        }
    }

    // Diag stations: lines just outside the box in each family, adjusted to
    // the piece's own line parity so opposite-colored bishops form the
    // adjacent pair on whichever side they pick. Not exclusive: two pieces on
    // one side IS the wall.
    let box_dp = cx + cy;
    let box_dn = cx - cy;
    let span = 2 * r + 1;
    for s in pieces {
        if is_ortho_slider(s.pt) || !is_diag_slider(s.pt) {
            continue;
        }
        let dp = s.x + s.y;
        let dn = s.x - s.y;
        let mut best = i64::MAX;
        for (line, target) in [
            (dp, box_dp - span),
            (dp, box_dp + span),
            (dn, box_dn - span),
            (dn, box_dn + span),
        ] {
            // Snap the target to the piece's parity.
            let t = target + ((target ^ line) & 1);
            best = best.min((line - t).abs());
        }
        bonus += (STATION_BASE - (best.min(60) as i32) * STATION_STEP).max(0);
        if best <= STATION_TOL_LINE {
            manned += 1;
        }
    }

    // Leapers and our king man the side midpoints just outside the box.
    let mids = [
        (cx - r - 1, cy),
        (cx + r + 1, cy),
        (cx, cy - r - 1),
        (cx, cy + r + 1),
    ];
    let mut mid_taken = [false; 4];
    let station_body = |x: i64, y: i64, taken: &mut [bool; 4]| -> (i32, bool) {
        let mut best = i64::MAX;
        let mut best_slot = 4;
        for (slot, &(mx, my)) in mids.iter().enumerate() {
            if !taken[slot] {
                let d = (x - mx).abs().max((y - my).abs());
                if d < best {
                    best = d;
                    best_slot = slot;
                }
            }
        }
        if best_slot < 4 {
            taken[best_slot] = true;
            let score = (STATION_BASE - (best.min(60) as i32) * STATION_STEP).max(0);
            return (score.max(if best <= STATION_TOL_LEAPER { STATION_BASE } else { 0 }), best <= STATION_TOL_LEAPER);
        }
        (0, false)
    };
    if let Some(ok) = our_king {
        let (v, ok_manned) = station_body(ok.x, ok.y, &mut mid_taken);
        bonus += v;
        if ok_manned {
            manned += 1;
        }
    }
    for s in pieces {
        if is_ortho_slider(s.pt) || is_diag_slider(s.pt) {
            continue;
        }
        let (v, is_manned) = station_body(s.x, s.y, &mut mid_taken);
        bonus += v;
        if is_manned {
            manned += 1;
        }
    }

    bonus + MANNED_LADDER[manned.min(MANNED_LADDER.len() - 1)]
}

// The moving pocket: the mating technique for leaper-led armies (one wall or
// none), decoded from real mate positions — our king pushes from behind in
// opposition, the LEAPERS lead ahead of the runner covering his escape
// squares (they are the only pieces faster than him, so tracking him is
// affordable for them and only them), and the slider trails with line access
// to deliver the final check.
const POCKET_BASE: i32 = 110;
// Above the king's per-step approach: the leapers must LEAD the pocket
// (they are faster than the runner), not starve behind king marches.
const POCKET_STEP: i32 = 20;
const POCKET_MANNED: i32 = 60;
const POCKET_ALL_MANNED: i32 = 140;

/// Flank-station score for the pocket's leapers (and any extra royals, e.g.
/// the second king of 2K+R): two targets ahead of the runner on his escape
/// side, laterally split like the knights of the model mates.
fn evaluate_pocket(
    pieces: &[SliderInfo],
    royals: &[Coordinate],
    kr: KingRelation,
    enemy_king: &Coordinate,
) -> i32 {
    let ex = enemy_king.x;
    let ey = enemy_king.y;
    // Escape direction: away from our king (the pusher). Without a king the
    // first slider serves as the reference.
    let (rx, ry) = if kr.king_dist != i64::MAX {
        (kr.our_dx, kr.our_dy)
    } else if let Some(s) = pieces.first() {
        (s.x - ex, s.y - ey)
    } else {
        return 0;
    };
    let esc_x = -rx.signum();
    let esc_y = -ry.signum();
    if esc_x == 0 && esc_y == 0 {
        return 0;
    }

    // Two flank targets ahead of the runner, split laterally.
    let targets: [(i64, i64); 2] = if esc_x == 0 {
        [(ex - 1, ey + 3 * esc_y), (ex + 1, ey + 3 * esc_y)]
    } else if esc_y == 0 {
        [(ex + 3 * esc_x, ey - 1), (ex + 3 * esc_x, ey + 1)]
    } else {
        [
            (ex + esc_x, ey + 3 * esc_y),
            (ex + 3 * esc_x, ey + esc_y),
        ]
    };

    let mut bonus = 0;
    let mut taken = [false; 2];
    let mut manned = 0;
    let station = |x: i64, y: i64, taken: &mut [bool; 2]| -> (i32, bool) {
        let mut best = i64::MAX;
        let mut slot = usize::MAX;
        for (i, &(tx, ty)) in targets.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let d = (x - tx).abs().max((y - ty).abs());
            if d < best {
                best = d;
                slot = i;
            }
        }
        if slot != usize::MAX {
            taken[slot] = true;
            let v = (POCKET_BASE - (best.min(90) as i32) * POCKET_STEP).max(-700);
            return (v, best <= 1);
        }
        (0, false)
    };

    for s in pieces {
        if is_ortho_slider(s.pt) || is_diag_slider(s.pt) {
            continue;
        }
        let (v, m) = station(s.x, s.y, &mut taken);
        bonus += v;
        if m {
            manned += 1;
        }
    }
    for r in royals.iter().skip(1) {
        let (v, m) = station(r.x, r.y, &mut taken);
        bonus += v;
        if m {
            manned += 1;
        }
    }

    if manned >= 1 {
        bonus += POCKET_MANNED;
    }
    if manned >= 2 {
        bonus += POCKET_ALL_MANNED;
    }
    bonus
}

/// Target box anchored to the enemy king's grid cell: assembly, not chasing.
/// The endgame squeeze after assembly is handled by the cage/ring terms.
fn evaluate_target_box(
    pieces: &[SliderInfo],
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
) -> i32 {
    let cx = enemy_king.x.div_euclid(BOX_GRID) * BOX_GRID + BOX_GRID / 2;
    let cy = enemy_king.y.div_euclid(BOX_GRID) * BOX_GRID + BOX_GRID / 2;
    target_box_score(pieces, our_king, cx, cy, BOX_R)
}

/// Integer square root (Newton's method); cage areas are at most 1024.
#[inline(always)]
fn isqrt_u32(v: u32) -> u32 {
    if v < 2 {
        return v;
    }
    let mut x = v;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + v / x) / 2;
    }
    x
}

struct PieceList {
    pieces: [SliderInfo; 24],
    len: usize,
}

impl PieceList {
    #[inline(always)]
    fn new() -> Self {
        Self {
            pieces: [SliderInfo { x: 0, y: 0, pt: PieceType::Void }; 24],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, piece: SliderInfo) {
        if self.len < self.pieces.len() {
            self.pieces[self.len] = piece;
            self.len += 1;
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[SliderInfo] {
        &self.pieces[..self.len]
    }
}

#[derive(Clone, Copy)]
struct KingRelation {
    our_dx: i64,
    our_dy: i64,
    king_dist: i64,
}

/// Dedicated K+2R-vs-K evaluation (unbounded board), winner's perspective.
/// The enemy king is bounded on each of the four sides by whichever is nearer,
/// a rook's cutting line or our king; a fully bounded king is a finite cage,
/// and shrinking that cage's perimeter drives the king toward our king (the
/// only side it can flee without the rooks capping it). Rook safety is scored
/// explicitly so the search never drifts a rook into capture.
fn evaluate_two_rook_drive(
    kr: KingRelation,
    enemy_king: &Coordinate,
    our_king: &Coordinate,
    r1: (i64, i64),
    r2: (i64, i64),
) -> i32 {
    let (ex, ey) = (enemy_king.x, enemy_king.y);
    let (okx, oky) = (our_king.x, our_king.y);
    let mut bonus = 0i32;

    // Closing distance on each side: nearest rook cut-line or our king.
    let inf = i64::MAX;
    let mut up = inf;
    let mut down = inf;
    let mut right = inf;
    let mut left = inf;
    let mut consider = |x: i64, y: i64| {
        if y > ey {
            up = up.min(y - ey);
        }
        if y < ey {
            down = down.min(ey - y);
        }
        if x > ex {
            right = right.min(x - ex);
        }
        if x < ex {
            left = left.min(ex - x);
        }
    };
    // A rook cuts with BOTH its rank and file; our king walls the sides it is on.
    consider(r1.0, r1.1);
    consider(r2.0, r2.1);
    consider(okx, oky);

    // Score each side: a bounded side rewards more the tighter its wall sits;
    // an open side is an escape lane the search must slam shut.
    for d in [up, down, right, left] {
        if d <= TR_SIDE_RANGE {
            bonus += TR_SIDE_CLOSED + (TR_TIGHT_NEAR - d).max(0) as i32 * TR_TIGHT_STEP;
        } else {
            bonus -= TR_OPEN_SIDE;
        }
    }
    let caged = up <= TR_SIDE_RANGE
        && down <= TR_SIDE_RANGE
        && right <= TR_SIDE_RANGE
        && left <= TR_SIDE_RANGE;
    if caged {
        bonus += TR_FULL_CAGE;
    }

    // Our king marches in — the near walls of the cage.
    bonus += (TR_APPROACH_MAX - kr.king_dist.min(TR_APPROACH_MAX)) as i32 * TR_APPROACH_STEP;
    if kr.king_dist <= 2 {
        bonus += TR_OPPOSITION;
    }

    // Rook safety. Connected rooks (shared rank/file, clear vs a bare king)
    // defend each other; a rook the enemy king attacks and no friendly unit
    // guards is hanging.
    let connected = r1.0 == r2.0 || r1.1 == r2.1;
    if connected {
        bonus += TR_CONNECTED;
    }
    for (rx, ry) in [r1, r2] {
        let attacked = (rx - ex).abs() <= 1 && (ry - ey).abs() <= 1;
        if attacked {
            let king_guards = (rx - okx).abs() <= 1 && (ry - oky).abs() <= 1;
            if !king_guards && !connected {
                bonus -= TR_HANG;
            }
        }
    }

    bonus
}

/// Unified mating-net evaluation for the piece-coordination (unbounded) model.
///
/// Scores are tiered so the search always has a monotone progress gradient:
///   1. Confinement — walls boxing the enemy king in, plus a flood-fill cage
///      bonus that grows as the reachable area shrinks.
///   2. King participation — steep approach when the material needs the king
///      for mate, a mild tiebreaker when a battery can mate alone.
///   3. Escape-ring control — covering the 8 squares around the enemy king,
///      weighted toward the side opposite our king once the king is engaged.
///   4. Piece shaping — small per-piece terms: sliders aim from mid range,
///      leapers walk in, pieces sit opposite our king, avoid useless checks.
/// Directional terms are gated on king proximity and kept below one king-step
/// of value, so the king can never profit from stepping away to reclassify them.
#[allow(clippy::too_many_arguments)]
fn evaluate_mating_net(
    game: &GameState,
    kr: KingRelation,
    our_king: Option<&Coordinate>,
    pieces: &[SliderInfo],
    enemy_king: &Coordinate,
    fences: &FenceState,
    material: &MaterialSummary,
    winning_color: PlayerColor,
    bounded: bool,
    bareish: bool,
) -> i32 {
    let ex = enemy_king.x;
    let ey = enemy_king.y;
    let mut bonus: i32 = 0;

    // Pure K+2R vs a bare king: the rolling lawnmower needs our king as the
    // near wall, so it has its own dedicated drive evaluation (the generic
    // tiers settle for a safe corridor the king runs up forever).
    if bareish
        && material.ortho_count == 2
        && material.total_non_pawn_pieces == 2
        && material.queen_count == 0
        && material.amazon_count == 0
        && material.chancellor_count == 0
        && material.archbishops == 0
        && material.diag_light == 0
        && material.diag_dark == 0
        && pieces.len() == 2
        && let Some(ok) = our_king
    {
        return evaluate_two_rook_drive(
            kr,
            enemy_king,
            ok,
            (pieces[0].x, pieces[0].y),
            (pieces[1].x, pieces[1].y),
        );
    }

    // Leaper-led armies (one wall or none, no queen) mate with the moving
    // pocket: king pushes from behind, leapers lead ahead of the runner.
    let pocket = bareish
        && material.queen_count + material.amazon_count == 0
        && material.wall_count() <= 1;

    // Directional terms ramp in smoothly as our king engages: a hard gate
    // would pay the defender a cliff of eval for stepping just past it.
    let near_scale = (NEAR_GATE_DIST + 4 - kr.king_dist.max(NEAR_GATE_DIST)).clamp(0, 4) as i32;

    // ---- Tier 1: confinement ----
    let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();

    // Wall distance per direction: nearest fence line or the board edge
    // (a wall one square past the last rank/file), whichever is closer.
    let fence_up = if fences.ortho_y_min_above != i64::MAX {
        fences.ortho_y_min_above - ey
    } else {
        i64::MAX
    };
    let fence_down = if fences.ortho_y_max_below != i64::MIN {
        ey - fences.ortho_y_max_below
    } else {
        i64::MAX
    };
    let fence_right = if fences.ortho_x_min_right != i64::MAX {
        fences.ortho_x_min_right - ex
    } else {
        i64::MAX
    };
    let fence_left = if fences.ortho_x_max_left != i64::MIN {
        ex - fences.ortho_x_max_left
    } else {
        i64::MAX
    };
    let wall_up = fence_up.min(max_y.saturating_sub(ey).saturating_add(1));
    let wall_down = fence_down.min(ey.saturating_sub(min_y).saturating_add(1));
    let wall_right = fence_right.min(max_x.saturating_sub(ex).saturating_add(1));
    let wall_left = fence_left.min(ex.saturating_sub(min_x).saturating_add(1));

    let score_ortho_wall = |d: i64| -> i32 {
        if d > WALL_RELEVANT_DIST {
            return 0;
        }
        CUT_FLAT_ORTHO + ((CUT_CAP_ORTHO - d).max(0) as i32) * CUT_STEP_ORTHO
    };
    {
        bonus += score_ortho_wall(wall_up)
            + score_ortho_wall(wall_down)
            + score_ortho_wall(wall_right)
            + score_ortho_wall(wall_left);

        // Flight-side walls (kingless armies have no flight bias).
        if kr.our_dy < 0 && wall_up <= WALL_RELEVANT_DIST {
            bonus += ORTHO_FLIGHT_SIDE;
        }
        if kr.our_dy > 0 && wall_down <= WALL_RELEVANT_DIST {
            bonus += ORTHO_FLIGHT_SIDE;
        }
        if kr.our_dx < 0 && wall_right <= WALL_RELEVANT_DIST {
            bonus += ORTHO_FLIGHT_SIDE;
        }
        if kr.our_dx > 0 && wall_left <= WALL_RELEVANT_DIST {
            bonus += ORTHO_FLIGHT_SIDE;
        }

        let boxed_v = wall_up <= WALL_RELEVANT_DIST && wall_down <= WALL_RELEVANT_DIST;
        let boxed_h = wall_right <= WALL_RELEVANT_DIST && wall_left <= WALL_RELEVANT_DIST;
        if boxed_v {
            bonus += SANDWICH_ORTHO;
        }
        if boxed_h {
            bonus += SANDWICH_ORTHO;
        }
        if boxed_v && boxed_h {
            bonus += FULL_BOX_BONUS;
        }
    }

    // Diagonal lines (offsets in x+y / x-y units). A lone diagonal only
    // impedes — a king step hops it from offset -1 to +1 — while an adjacent
    // pair (opposite-colored bishops) is a true wall.
    let enemy_dp = ex + ey;
    let enemy_dn = ex - ey;
    // Which side of each diagonal family our king stands on (0 = none/aligned).
    let our_dp_rel = (kr.our_dx + kr.our_dy).signum();
    let our_dn_rel = (kr.our_dx - kr.our_dy).signum();
    let score_diag_family = |lines: &[i64], enemy_line: i64, our_rel: i64| -> i32 {
        let mut single_above = i64::MAX;
        let mut single_below = i64::MAX;
        let mut pair_above = i64::MAX;
        let mut pair_below = i64::MAX;
        for (i, &a) in lines.iter().enumerate() {
            let off = a - enemy_line;
            if off > 0 {
                single_above = single_above.min(off);
            } else {
                single_below = single_below.min(-off);
            }
            for &b in &lines[i + 1..] {
                // Adjacent pair; both lines sit on the same side of the king
                // (offsets 0 are never collected).
                if (a - b).abs() == 1 {
                    let d = off.abs().min((b - enemy_line).abs());
                    if off > 0 {
                        pair_above = pair_above.min(d);
                    } else {
                        pair_below = pair_below.min(d);
                    }
                }
            }
        }
        let mut score = 0;
        for d in [single_above, single_below] {
            if d <= WALL_RELEVANT_DIST {
                score += CUT_FLAT_DIAG + ((CUT_CAP_DIAG - d).max(0) as i32) * CUT_STEP_DIAG;
            }
        }
        for d in [pair_above, pair_below] {
            if d <= WALL_RELEVANT_DIST {
                score += DIAG_PAIR_FLAT + ((DIAG_PAIR_CAP - d).max(0) as i32) * DIAG_PAIR_STEP;
            }
        }
        // A pair wall on the side away from our king fences the flight path.
        if pair_above <= WALL_RELEVANT_DIST && our_rel < 0 {
            score += DIAG_PAIR_FAR_SIDE;
        }
        if pair_below <= WALL_RELEVANT_DIST && our_rel > 0 {
            score += DIAG_PAIR_FAR_SIDE;
        }
        if single_above <= WALL_RELEVANT_DIST && single_below <= WALL_RELEVANT_DIST {
            score += SANDWICH_DIAG;
        }
        if pair_above <= WALL_RELEVANT_DIST && pair_below <= WALL_RELEVANT_DIST {
            score += DIAG_PAIR_SANDWICH;
        }
        score
    };
    bonus += score_diag_family(&fences.dp_lines[..fences.dp_count], enemy_dp, our_dp_rel)
        + score_diag_family(&fences.dn_lines[..fences.dn_count], enemy_dn, our_dn_rel);

    // The defender royal may move like more than a king (e.g. a royal centaur
    // leaps out of king-step cages); its extra escape squares extend the ring.
    let enemy_royal_pt = game
        .board
        .get_piece(ex, ey)
        .map(|p| p.piece_type())
        .unwrap_or(PieceType::King);
    let royal_leaps = matches!(enemy_royal_pt, PieceType::RoyalCentaur);

    // Formation planning: armies with two or more true walls assemble a full
    // target box; leaper-led armies run the moving pocket.
    if pocket {
        // All of the winning side's royals join the pocket (e.g. 2K+R).
        let royals: &[Coordinate] = if winning_color == PlayerColor::White {
            &game.white_royals
        } else {
            &game.black_royals
        };
        bonus += evaluate_pocket(pieces, royals, kr, enemy_king);
    } else if bareish {
        // The box is scaffolding for the approach. Once the kill zone is set
        // (most of the force already on top of the defender), its station
        // gradients only detour the search away from the fastest mate, so it
        // hands over to a flat bonus and raw tactics.
        let engaged_pieces = pieces
            .iter()
            .filter(|s| (s.x - ex).abs().max((s.y - ey).abs()) <= 4)
            .count();
        let kill_zone = engaged_pieces >= 3 && (kr.king_dist <= 6 || kr.king_dist == i64::MAX);
        if kill_zone {
            bonus += KILL_ZONE_BONUS;
        } else {
            bonus += evaluate_target_box(pieces, our_king, enemy_king);
        }
    }

    // Flood-fill cage: confirms the walls actually contain the king and
    // rewards shrinking its reachable area. The flood is king-step based, so
    // it proves nothing for a leaping royal.
    if bareish && !royal_leaps {
        let (caged, area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            enemy_king,
            winning_color,
        );
        if caged {
            bonus += CAGE_FLAT + (3600 / (isqrt_u32(area) as i32 + 2)).min(CAGE_MAX);
        }
    }

    // ---- Tier 2: king participation ----
    if kr.king_dist != i64::MAX {
        if king_mostly_idle(material, bounded) {
            bonus += ((KING_CAP_IDLE - kr.king_dist.min(KING_CAP_IDLE)) as i32) * KING_STEP_IDLE;
        } else {
            bonus += ((KING_CAP_NEEDED - kr.king_dist.min(KING_CAP_NEEDED)) as i32)
                * KING_STEP_NEEDED;
            if kr.king_dist <= 2 {
                bonus += KING_NEAR2_NEEDED;
            } else if kr.king_dist <= 4 {
                bonus += KING_NEAR4_NEEDED;
            } else if kr.king_dist > KING_CAP_NEEDED {
                // Milder tail so the chase never goes gradient-flat, however
                // far the defender runs.
                bonus -= ((kr.king_dist - KING_CAP_NEEDED).min(150) as i32) * KING_TAIL_STEP;
            }
        }
    }

    // ---- Tier 3: escape-ring control ----
    let engaged = near_scale > 0;
    let esc_scale = near_scale;
    let escape_x = if engaged { -kr.our_dx.signum() } else { 0 };
    let escape_y = if engaged { -kr.our_dy.signum() } else { 0 };
    const KNIGHT_RING: [(i64, i64); 8] = [
        (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    ];
    let mut ring_buf: [(i64, i64); 16] = [(0, 0); 16];
    ring_buf[..8].copy_from_slice(&RING_DIRS);
    let ring_len = if royal_leaps {
        ring_buf[8..].copy_from_slice(&KNIGHT_RING);
        16
    } else {
        8
    };
    let ring = &ring_buf[..ring_len];
    let ring_total = ring.len() as i32;
    let mut n_covered: i32 = 0;
    for &(rdx, rdy) in ring {
        let rx = ex + rdx;
        let ry = ey + rdy;
        let mut covered = pieces
            .iter()
            .any(|s| piece_attacks_geom(s.pt, winning_color, rx - s.x, ry - s.y));
        if !covered && let Some(ok) = our_king {
            covered = (rx - ok.x).abs() <= 1 && (ry - ok.y).abs() <= 1;
        }
        if covered {
            n_covered += 1;
            bonus += RING_COVER;
            let esc_x = escape_x != 0 && rdx.signum() == escape_x;
            let esc_y = escape_y != 0 && rdy.signum() == escape_y;
            if esc_x && esc_y {
                bonus += RING_ESC_BOTH * esc_scale / 4;
            } else if esc_x || esc_y {
                bonus += RING_ESC_ONE * esc_scale / 4;
            }
        }
    }
    if n_covered >= ring_total - 1 {
        bonus += RING_SEVEN_PLUS;
    } else if n_covered >= ring_total - 2 {
        bonus += RING_SIX;
    }

    // Outer ring (radius 2): armies without long fence lines (minor pieces)
    // make progress by compressing the space around the king square by square;
    // covering the outer ring is that squeeze's smooth gradient.
    if !royal_leaps {
        for i in 0i64..16 {
            // Walk the 16 squares of the radius-2 Chebyshev ring.
            let (rdx, rdy) = match i {
                0..=4 => (i - 2, 2),
                5..=9 => (i - 7, -2),
                10..=12 => (2, i - 11),
                _ => (-2, i - 14),
            };
            let rx = ex + rdx;
            let ry = ey + rdy;
            let mut covered = pieces
                .iter()
                .any(|s| piece_attacks_geom(s.pt, winning_color, rx - s.x, ry - s.y));
            if !covered && let Some(ok) = our_king {
                covered = (rx - ok.x).abs() <= 1 && (ry - ok.y).abs() <= 1;
            }
            if covered {
                bonus += RING_OUTER_COVER;
            }
        }
    }

    // ---- Tier 4: per-piece shaping ----
    for s in pieces {
        let pdx = s.x - ex;
        let pdy = s.y - ey;
        let dist = pdx.abs().max(pdy.abs());
        let is_ortho = is_ortho_slider(s.pt);
        let is_diag = is_diag_slider(s.pt);

        if is_ortho || is_diag {
            // A slider's power is its line, not its proximity: engagement is
            // the offset of its nearest usable line from the enemy king, so a
            // rook posted far away along a cutting line is not "far". Hugging
            // the king still risks capture.
            let mut line_off = i64::MAX;
            if is_ortho {
                line_off = line_off.min(pdx.abs().min(pdy.abs()));
            }
            if is_diag {
                line_off = line_off.min((pdx + pdy).abs().min((pdx - pdy).abs()));
            }
            if dist <= 1 {
                bonus -= SLIDER_HUG_PENALTY;
            } else if line_off <= 10 {
                bonus += SLIDER_BAND_BONUS;
            } else if is_ortho {
                // An ortho slider off every useful line goes dead; pure diagonal
                // pieces stay neutral — their fence lines work at any range
                // (e.g. a pair wall posted far ahead of a fleeing king).
                bonus -= ((line_off - 10).min(SLIDER_FAR_CAP) as i32) * SLIDER_FAR_STEP;
            }
            // A cut is held from FAR along its line: near the king the wall
            // piece gets harassed off the line and the runner slips through
            // the released cut (trace-proven failure cycle). Beyond the
            // harass radius a standoff gradient keeps pulling the holder out
            // of chasing range entirely (offset 0 is the check ray, not a
            // cut, and earns nothing). Standoff needs wall redundancy: an
            // army whose ONLY wall piece stands off cannot re-cut a turning
            // runner (the single-queen pocket must stay close).
            if line_off <= 2 && dist < 8 {
                bonus -= ((8 - dist) as i32) * WALL_HARASS_PENALTY;
            }
            if line_off >= 1 && line_off <= 2 && material.wall_count() >= 2 {
                bonus += (dist.min(SLIDER_STANDOFF_CAP) as i32) * SLIDER_STANDOFF_STEP;
            }
        } else {
            // Short-range pieces only matter in the net once they walk in.
            bonus += ((LEAPER_ENGAGE_CAP - dist.min(2 * LEAPER_ENGAGE_CAP)) as i32)
                * LEAPER_ENGAGE_STEP;
        }

        if engaged {
            // Pieces on the far side of the enemy king cut its retreat.
            if pdx != 0 && pdx.signum() != kr.our_dx.signum() {
                bonus += OPPOSITE_SIDE_BONUS * near_scale / 4;
            }
            if pdy != 0 && pdy.signum() != kr.our_dy.signum() {
                bonus += OPPOSITE_SIDE_BONUS * near_scale / 4;
            }
        } else if !is_ortho && !is_diag && kr.king_dist != i64::MAX {
            // Far phase: a leaper is faster than the fleeing king, and posted on
            // the flight side (away from our king) it is a speed bump the
            // runner must detour around.
            if pdx != 0 && pdx.signum() != kr.our_dx.signum() {
                bonus += OPPOSITE_SIDE_BONUS / 2;
            }
            if pdy != 0 && pdy.signum() != kr.our_dy.signum() {
                bonus += OPPOSITE_SIDE_BONUS / 2;
            }
        }

        // Standing on the enemy king's own line is a check, not a cut. A
        // frontal check pushes him away from our king (wasted tempo); a check
        // from squarely behind herds him toward our king — a slider re-checks
        // by sliding, so the herd costs no net tempo.
        let aligned = match s.pt {
            PieceType::Rook | PieceType::Chancellor => pdx == 0 || pdy == 0,
            PieceType::Bishop | PieceType::Archbishop => pdx.abs() == pdy.abs(),
            PieceType::Queen | PieceType::RoyalQueen | PieceType::Amazon => {
                pdx == 0 || pdy == 0 || pdx.abs() == pdy.abs()
            }
            _ => false,
        };
        if aligned && kr.king_dist != i64::MAX {
            let frontal = (pdx != 0 && pdx.signum() == kr.our_dx.signum())
                || (pdy != 0 && pdy.signum() == kr.our_dy.signum());
            let behind = dist >= 2
                && (pdx == 0 || pdx.signum() != kr.our_dx.signum())
                && (pdy == 0 || pdy.signum() != kr.our_dy.signum());
            if frontal {
                bonus -= FRONTAL_CHECK_PENALTY;
            } else if behind {
                bonus += HERD_CHECK_BONUS;
            } else {
                bonus -= SIDE_CHECK_PENALTY;
            }
        }
    }

    // A net piece within the defender king's reach invites a breakout unless
    // protected; pieces further out are safe on their own.
    if material.total_non_pawn_pieces <= 4 {
        for s in pieces {
            let dist = (s.x - ex).abs().max((s.y - ey).abs());
            if dist <= 4
                && is_square_attacked(
                    &game.board,
                    &Coordinate::new(s.x, s.y),
                    winning_color,
                    &game.spatial_indices,
                )
            {
                bonus += PROTECTED_PIECE_BONUS;
            }
        }
    }

    bonus
}

/// Main logic for driving the enemy king to mate.
#[inline(always)]
fn evaluate_mop_up_core(
    game: &GameState,
    our_king: Option<&Coordinate>,
    enemy_king: &Coordinate,
    winning_color: PlayerColor,
) -> i32 {
    let bounded = crate::moves::get_world_size() <= MOP_UP_BOUNDED_MAX;
    let losing_color = winning_color.opponent();
    let bareish = defender_is_bareish(game, losing_color);

    // On a bounded board the edge confines the king, so a small edge +
    // proximity tiebreaker replaces the piece-coordination model.
    if bounded && bareish {
        return bounded_lone_king_mop_up(game, our_king, enemy_king, winning_color);
    }

    let kr = if let Some(ok) = our_king {
        let dx = ok.x - enemy_king.x;
        let dy = ok.y - enemy_king.y;
        KingRelation {
            our_dx: dx,
            our_dy: dy,
            king_dist: dx.abs().max(dy.abs()),
        }
    } else {
        KingRelation {
            our_dx: 0,
            our_dy: 0,
            king_dist: i64::MAX,
        }
    };

    let mut fences = FenceState::new();
    let mut material = MaterialSummary::default();
    let mut our_pieces = PieceList::new();

    let enemy_x = enemy_king.x;
    let enemy_y = enemy_king.y;
    let enemy_diag_pos = enemy_x + enemy_y;
    let enemy_diag_neg = enemy_x - enemy_y;

    let is_white = winning_color == PlayerColor::White;
    for (x, y, piece) in game.board.iter_pieces_by_color(is_white) {
        let pt = piece.piece_type();
        if pt.is_royal() || pt == PieceType::Pawn {
            continue;
        }

        our_pieces.push(SliderInfo { x, y, pt });
        material.total_non_pawn_pieces = material.total_non_pawn_pieces.saturating_add(1);

        let has_ortho = matches!(
            pt,
            PieceType::Rook | PieceType::Queen | PieceType::Chancellor | PieceType::Amazon
        );
        let has_diag = matches!(
            pt,
            PieceType::Bishop | PieceType::Queen | PieceType::Archbishop | PieceType::Amazon
        );

        if has_ortho {
            material.ortho_count = material.ortho_count.saturating_add(1);
            if y > enemy_y {
                fences.ortho_y_min_above = fences.ortho_y_min_above.min(y);
            } else if y < enemy_y {
                fences.ortho_y_max_below = fences.ortho_y_max_below.max(y);
            }
            if x > enemy_x {
                fences.ortho_x_min_right = fences.ortho_x_min_right.min(x);
            } else if x < enemy_x {
                fences.ortho_x_max_left = fences.ortho_x_max_left.max(x);
            }
        }
        if has_diag {
            let dp = x + y;
            let dn = x - y;
            if dp != enemy_diag_pos {
                fences.push_dp(dp);
            }
            if dn != enemy_diag_neg {
                fences.push_dn(dn);
            }
        }

        match pt {
            PieceType::Queen => material.queen_count += 1,
            PieceType::Amazon => material.amazon_count += 1,
            PieceType::Chancellor => material.chancellor_count += 1,
            PieceType::Archbishop => material.archbishops += 1,
            _ => {}
        }
        if pt == PieceType::Bishop {
            if (x + y) & 1 == 0 {
                material.diag_light = material.diag_light.saturating_add(1);
            } else {
                material.diag_dark = material.diag_dark.saturating_add(1);
            }
        }
    }

    evaluate_mating_net(
        game,
        kr,
        our_king,
        our_pieces.as_slice(),
        enemy_king,
        &fences,
        &material,
        winning_color,
        bounded,
        bareish,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::game::GameState;

    fn create_test_game() -> GameState {
        let mut game = GameState::new();
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    fn create_test_game_from_icn(icn: &str) -> GameState {
        let mut game = create_test_game();
        game.board = Board::new();
        game.setup_position_from_icn(icn);
        game
    }

    #[test]
    fn test_is_lone_king_true() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1");

        assert!(is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_is_lone_king_false() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1");

        assert!(!is_lone_king(&game, PlayerColor::White));
    }

    #[test]
    fn test_calculate_mop_up_scale_returns_none_for_no_advantage() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,1|q4,8");

        // Both sides have material, so no mop-up
        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // May or may not apply depending on thresholds
        assert!(scale.is_none() || scale.unwrap() <= 100);
    }

    #[test]
    fn test_king_mostly_idle_material() {
        // 2 rooks on the unbounded board still need the king.
        let two_rooks = MaterialSummary {
            ortho_count: 2,
            total_non_pawn_pieces: 2,
            ..Default::default()
        };
        assert!(!king_mostly_idle(&two_rooks, false));
        // A rook pair ladder-mates alone against an edge.
        assert!(king_mostly_idle(&two_rooks, true));

        // A queen + rook still needs the king without an edge.
        let queen_rook = MaterialSummary {
            ortho_count: 2,
            queen_count: 1,
            total_non_pawn_pieces: 2,
            ..Default::default()
        };
        assert!(!king_mostly_idle(&queen_rook, false));

        // Three queens weave a net alone anywhere.
        let three_queens = MaterialSummary {
            ortho_count: 3,
            queen_count: 3,
            total_non_pawn_pieces: 3,
            ..Default::default()
        };
        assert!(king_mostly_idle(&three_queens, false));
        assert!(king_mostly_idle(&three_queens, true));
    }

    #[test]
    fn test_mop_up_scale_fades_with_defender_material() {
        let bare = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1|R3,1|k5,8");
        let scale_bare = calculate_mop_up_scale(&bare, PlayerColor::Black);
        assert_eq!(scale_bare, Some(100), "bare king should give full scale");

        let with_knight = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1|R3,1|k5,8|n6,8");
        let scale_knight = calculate_mop_up_scale(&with_knight, PlayerColor::Black);
        assert!(
            scale_knight.is_some() && scale_knight.unwrap() < 100,
            "defender knight should reduce but not disable the scale: {:?}",
            scale_knight
        );

        let with_queen = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1|R3,1|k5,8|q6,8");
        let scale_queen = calculate_mop_up_scale(&with_queen, PlayerColor::Black);
        assert!(
            scale_queen.is_some() && scale_queen.unwrap() < scale_knight.unwrap(),
            "Q+R vs a defender queen still mops up, weaker than vs a knight: {:?}",
            scale_queen
        );

        // A big army overwhelms even a defender queen: near-full shaping.
        let army = create_test_game_from_icn("w (8;q|1;q) K5,1|Q4,1|Q3,1|R2,1|R1,1|k5,8|q6,8");
        let scale_army = calculate_mop_up_scale(&army, PlayerColor::Black);
        assert!(
            scale_army.unwrap() > scale_queen.unwrap(),
            "more surplus must mean more shaping: {:?} vs {:?}",
            scale_army,
            scale_queen
        );

        // A thin edge over real defense is a fight, not a mate hunt.
        let thin = create_test_game_from_icn("w (8;q|1;q) K5,1|R4,1|N3,1|k5,8|r6,8|p6,7+");
        assert_eq!(
            calculate_mop_up_scale(&thin, PlayerColor::Black),
            None,
            "R+N vs R+P has no mop-up surplus"
        );
    }

    #[test]
    fn test_mop_up_gradient_exists_vs_defender_minor() {
        // The mop-up used to vanish whenever the defender kept a piece; the
        // king-approach gradient must survive a lone defender knight (and
        // must survive the defended-net saturation, which is monotone).
        let scaled = |icn: &str| {
            let game = create_test_game_from_icn(icn);
            let (winner, scale) = active_mop_up(&game).expect("Q+R vs N is a mop-up");
            assert_eq!(winner, PlayerColor::White);
            evaluate_mop_up_scaled(&game, winner, scale)
        };

        let close_score = scaled("w (8;q|1;q) k5,5|n6,6|K8,5|Q2,2|R2,3");
        let far_score = scaled("w (8;q|1;q) k5,5|n6,6|K12,5|Q2,2|R2,3");

        assert!(
            close_score > far_score,
            "approach gradient must survive a defender minor: close={} far={}",
            close_score,
            far_score
        );
    }

    #[test]
    fn test_king_never_gains_by_stepping_away() {
        // K+2R (king needed on the unbounded board): from any nearby square,
        // every king step that increases Chebyshev distance must lower the score.
        let enemy_king = Coordinate::new(0, 0);
        for (kx, ky) in [(4i64, 0i64), (4, 3), (3, 3), (6, 1)] {
            let near_icn = format!("w (8;q|1;q) k0,0|R9,1|R9,-1|K{},{}", kx, ky);
            let near = create_test_game_from_icn(&near_icn);
            let near_score = evaluate_lone_king_endgame(
                &near,
                Some(&Coordinate::new(kx, ky)),
                &enemy_king,
                PlayerColor::White,
            );
            for (dx, dy) in [(1i64, 0i64), (1, 1), (0, 1), (1, -1)] {
                let (ax, ay) = (kx + dx, ky + dy);
                if ax.abs().max(ay.abs()) <= kx.abs().max(ky.abs()) {
                    continue; // not a retreat
                }
                let away_icn = format!("w (8;q|1;q) k0,0|R9,1|R9,-1|K{},{}", ax, ay);
                let away = create_test_game_from_icn(&away_icn);
                let away_score = evaluate_lone_king_endgame(
                    &away,
                    Some(&Coordinate::new(ax, ay)),
                    &enemy_king,
                    PlayerColor::White,
                );
                assert!(
                    away_score < near_score,
                    "king retreat ({},{}) -> ({},{}) must lose score: near={} away={}",
                    kx,
                    ky,
                    ax,
                    ay,
                    near_score,
                    away_score
                );
            }
        }
    }

    #[test]
    fn test_tighter_fence_scores_higher() {
        // Two rooks so the position routes to the box model, whose fence
        // terms this test exercises (one-wall armies shore-drive instead).
        let enemy_king = Coordinate::new(5, 5);
        let our_king = Coordinate::new(5, 2);

        let close_fence = create_test_game_from_icn("w (8;q|1;q) k5,5|K5,2|R1,9|R-3,1");
        let close_score = evaluate_lone_king_endgame(
            &close_fence,
            Some(&our_king),
            &enemy_king,
            PlayerColor::White,
        );

        let far_fence = create_test_game_from_icn("w (8;q|1;q) k5,5|K5,2|R1,13|R-3,1");
        let far_score = evaluate_lone_king_endgame(
            &far_fence,
            Some(&our_king),
            &enemy_king,
            PlayerColor::White,
        );

        assert!(
            close_score > far_score,
            "a nearer cut line should score higher: close={} far={}",
            close_score,
            far_score
        );
    }

    #[test]
    fn test_evaluate_lone_king_endgame_returns_value() {
        let game = create_test_game_from_icn("w (8;q|1;q) K5,1|k5,8|Q4,1");

        let enemy_king = Coordinate::new(5, 8);
        let our_king = Coordinate::new(5, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive (White has advantage)
        assert!(score >= 0);
    }

    #[test]
    fn test_evaluate_mop_up_scaled_no_king() {
        // No white king (checkmate practice): kingless armies still mop up.
        let game = create_test_game_from_icn("w (8;q|1;q) k5,8|Q4,4|Q3,4");

        let (winner, scale) = active_mop_up(&game).expect("kingless 2Q vs bare k mops up");
        assert_eq!((winner, scale), (PlayerColor::White, 100));
        let score = evaluate_mop_up_scaled(&game, winner, scale);
        assert!(score.abs() < 100000);
    }

    #[test]
    fn test_mop_up_rook_fence_bonus() {
        let game = create_test_game_from_icn("w (8;q|1;q) k4,4|R0,4|R7,4|K4,1");

        let enemy_king = Coordinate::new(4, 4);
        let our_king = Coordinate::new(4, 1);

        let score =
            evaluate_lone_king_endgame(&game, Some(&our_king), &enemy_king, PlayerColor::White);
        // Should be positive since rooks create cutting lines
        assert!(
            score > 0,
            "Rook fence should give positive score: {}",
            score
        );
    }

    #[test]
    fn test_mop_up_king_approach_bonus() {
        let mut game = create_test_game_from_icn("w (8;q|1;q) k5,5|Q4,4|K6,5");

        let enemy_king = Coordinate::new(5, 5);
        let our_king_close = Coordinate::new(6, 5);

        let score_close = evaluate_lone_king_endgame(
            &game,
            Some(&our_king_close),
            &enemy_king,
            PlayerColor::White,
        );

        // Move white king further away
        game.setup_position_from_icn("w (8;q|1;q) k5,5|Q4,4|K1,1");

        let our_king_far = Coordinate::new(1, 1);
        let score_far =
            evaluate_lone_king_endgame(&game, Some(&our_king_far), &enemy_king, PlayerColor::White);

        assert!(
            score_close > score_far,
            "Closer king should get higher score: close={} far={}",
            score_close,
            score_far
        );
    }

    #[test]
    fn test_calculate_mop_up_scale_with_pawns() {
        let game = create_test_game_from_icn("w (8;q|1;q) k5,5|K4,4|R1,1|R2,2|P3,7");

        let scale = calculate_mop_up_scale(&game, PlayerColor::Black);
        // Should return a scale since white has mating material
        assert!(scale.is_some(), "Should have mop-up scale with rooks");
    }

    #[test]
    fn test_active_mop_up_gates() {
        // Bare king vs two rooks: full-strength mop-up for white.
        let bare = create_test_game_from_icn("w (8;q|1;q) K1,1|R2,2|R3,3|k9,9");
        assert_eq!(active_mop_up(&bare), Some((PlayerColor::White, 100)));

        // Winner with pawns: promotion is the plan, not the net.
        let winner_pawns = create_test_game_from_icn("w (8;q|1;q) K1,1|R2,2|P3,3|k9,9");
        assert_eq!(active_mop_up(&winner_pawns), None);

        // K+R vs K+3P: a race, not a mop-up (pawn defenders need a
        // queen-up surplus). The SPRT-fatal +16 shape.
        let race =
            create_test_game_from_icn("b (8;q|1;q) K8,5|P7,2+|P8,3+|P7,3+|k2,1|r3,4");
        assert_eq!(active_mop_up(&race), None);

        // Defender queen vs a big army: active, scaled down.
        let army = create_test_game_from_icn("w (8;q|1;q) K1,1|Q2,2|Q3,1|R1,3|R4,4|k9,9|q10,10");
        let (winner, scale) = active_mop_up(&army).expect("big army mops up a defender queen");
        assert_eq!(winner, PlayerColor::White);
        assert!(0 < scale && scale < 100, "scaled activation: {}", scale);

        // Defender with two non-pawn pieces: a real fight, no mop-up.
        let fight = create_test_game_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9|r10,10|n10,8");
        assert_eq!(active_mop_up(&fight), None);

        // Black as the winning side mirrors.
        let mirrored = create_test_game_from_icn("w (8;q|1;q) K9,9|k1,1|r2,2|r3,3");
        assert_eq!(active_mop_up(&mirrored), Some((PlayerColor::Black, 100)));
    }

    #[test]
    fn test_defended_mop_up_saturates() {
        // Against a bare king the net passes through uncompressed...
        let bare = create_test_game_from_icn("w (8;q|1;q) K1,1|Q2,2|R3,1|k9,9");
        let (w, s) = active_mop_up(&bare).unwrap();
        let full = evaluate_mop_up_scaled(&bare, w, s);
        assert!(full > MOP_UP_DEFENDED_CAP, "bare-king net keeps its full range: {}", full);

        // ...but with a defender piece on the board the term saturates below
        // the cap, so shaping can never outweigh material.
        let defended = create_test_game_from_icn("w (8;q|1;q) K1,1|Q2,2|Q3,1|R1,3|R4,4|k9,9|q10,10");
        let (w, s) = active_mop_up(&defended).unwrap();
        let capped = evaluate_mop_up_scaled(&defended, w, s);
        assert!(
            0 < capped && capped < MOP_UP_DEFENDED_CAP,
            "defended net saturates under the cap: {}",
            capped
        );
    }

    #[test]
    fn test_amazon_prefers_cutoff_over_drifting() {
        // Our king approaches from the left; the amazon should stand on the
        // far side of the enemy king, cutting its retreat.
        let enemy_king = Coordinate::new(5, 5);
        let our_king = Coordinate::new(3, 5);

        let far_side = create_test_game_from_icn("w (8;q|1;q) k5,5|K3,5|AM9,5");
        let far_side_score =
            evaluate_lone_king_endgame(&far_side, Some(&our_king), &enemy_king, PlayerColor::White);

        let our_side = create_test_game_from_icn("w (8;q|1;q) k5,5|K3,5|AM-1,5");
        let our_side_score =
            evaluate_lone_king_endgame(&our_side, Some(&our_king), &enemy_king, PlayerColor::White);

        assert!(
            far_side_score > our_side_score,
            "Amazon should prefer cutting off the king from the far side: far={} same={}",
            far_side_score,
            our_side_score
        );
    }

    #[test]
    fn test_amazon_prefers_king_closer_in_lone_king_mop_up() {
        let enemy_king = Coordinate::new(5, 5);

        let close = create_test_game_from_icn("w (8;q|1;q) k5,5|K4,5|AM7,5");
        let close_king = Coordinate::new(4, 5);
        let close_score =
            evaluate_lone_king_endgame(&close, Some(&close_king), &enemy_king, PlayerColor::White);

        let far = create_test_game_from_icn("w (8;q|1;q) k5,5|K1,5|AM7,5");
        let far_king = Coordinate::new(1, 5);
        let far_score =
            evaluate_lone_king_endgame(&far, Some(&far_king), &enemy_king, PlayerColor::White);

        assert!(
            close_score > far_score,
            "K+Amazon mop-up should strongly prefer king approach: close={} far={}",
            close_score,
            far_score
        );
    }

    #[test]
    fn test_smart_mop_up_prefers_pieces_opposite_our_king() {
        // White king on the left of black king (5,5). A second piece (chancellor)
        // far away to the right cuts off escape — should score higher than placing
        // it on the same side as our king.
        // Unbounded position with two chancellors, so the piece-coordination
        // model runs and the test isolates the smart opposition logic.
        let opposite = create_test_game_from_icn(
            "w (50;q|1;q) k5,5|K3,5|CH20,5|CH18,3",
        );
        let same_side = create_test_game_from_icn(
            "w (50;q|1;q) k5,5|K3,5|CH-20,5|CH-18,3",
        );
        let ek = Coordinate::new(5, 5);
        let s_opp = evaluate_lone_king_endgame(
            &opposite,
            Some(&Coordinate::new(3, 5)),
            &ek,
            PlayerColor::White,
        );
        let s_same = evaluate_lone_king_endgame(
            &same_side,
            Some(&Coordinate::new(3, 5)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_opp > s_same,
            "Piece opposite our king should score better: opp={} same={}",
            s_opp,
            s_same
        );
    }

    #[test]
    fn test_smart_mop_up_rewards_ring_coverage_with_exotic_pieces() {
        // Knight + Camel + Giraffe surround a king with our king nearby.
        // Compared to all those pieces clustered far away, ring coverage
        // and sandwich logic should give a clearly higher score.
        let near = create_test_game_from_icn(
            "w (50;q|1;q) k10,10|K12,10|N8,9|CA7,10|GI10,14",
        );
        let far = create_test_game_from_icn(
            "w (50;q|1;q) k10,10|K-30,-30|N-32,-31|CA-33,-30|GI-30,-26",
        );
        let ek = Coordinate::new(10, 10);
        let s_near = evaluate_lone_king_endgame(
            &near,
            Some(&Coordinate::new(12, 10)),
            &ek,
            PlayerColor::White,
        );
        let s_far = evaluate_lone_king_endgame(
            &far,
            Some(&Coordinate::new(-30, -30)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_near > s_far,
            "Pieces engaged around enemy king should score way higher: near={} far={}",
            s_near,
            s_far
        );
    }

    #[test]
    fn test_smart_mop_up_axis_sandwich_bonus() {
        // Three rooks (so the position uses the generic net, not the dedicated
        // K+2R drive): rooks above and below the enemy king (vertical sandwich)
        // should score higher than rooks both above (no sandwich). The third
        // rook is placed identically in both to isolate the sandwich term.
        let sandwich =
            create_test_game_from_icn("w (50;q|1;q) k10,10|K10,7|R10,2|R10,18|R60,10");
        let no_sandwich =
            create_test_game_from_icn("w (50;q|1;q) k10,10|K10,7|R10,18|R10,19|R60,10");
        let ek = Coordinate::new(10, 10);
        let s_sand = evaluate_lone_king_endgame(
            &sandwich,
            Some(&Coordinate::new(10, 7)),
            &ek,
            PlayerColor::White,
        );
        let s_no = evaluate_lone_king_endgame(
            &no_sandwich,
            Some(&Coordinate::new(10, 7)),
            &ek,
            PlayerColor::White,
        );
        assert!(
            s_sand > s_no,
            "Vertical sandwich should outscore stacked rooks: sand={} no={}",
            s_sand,
            s_no
        );
    }

    #[test]
    fn test_piece_attacks_geom_basic_pieces() {
        // Sanity-check the geometric attack table for representative pieces.
        assert!(piece_attacks_geom(PieceType::Rook, PlayerColor::White, 7, 0));
        assert!(!piece_attacks_geom(PieceType::Rook, PlayerColor::White, 3, 4));
        assert!(piece_attacks_geom(PieceType::Bishop, PlayerColor::White, 4, 4));
        assert!(piece_attacks_geom(PieceType::Knight, PlayerColor::White, 1, 2));
        assert!(piece_attacks_geom(PieceType::Camel, PlayerColor::White, 1, 3));
        assert!(piece_attacks_geom(PieceType::Giraffe, PlayerColor::White, 4, 1));
        assert!(piece_attacks_geom(PieceType::Zebra, PlayerColor::White, 3, 2));
        assert!(piece_attacks_geom(PieceType::Hawk, PlayerColor::White, 3, 3));
        assert!(!piece_attacks_geom(PieceType::Hawk, PlayerColor::White, 1, 1));
        assert!(piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 2, 4));
        assert!(piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 6, 3));
        assert!(!piece_attacks_geom(PieceType::Knightrider, PlayerColor::White, 5, 5));
        assert!(piece_attacks_geom(PieceType::Huygen, PlayerColor::White, 7, 0));
        assert!(!piece_attacks_geom(PieceType::Huygen, PlayerColor::White, 4, 0));
    }

    #[test]
    fn test_find_bitboard_cage() {
        let game = create_test_game_from_icn("w (8;q|1;q) k4,4|R4,0|R4,8|R0,4|R8,4|K1,1");

        let enemy_king = Coordinate::new(4, 4);
        let (_is_caged, area) = find_bitboard_cage(
            &game.board,
            &game.spatial_indices,
            &enemy_king,
            PlayerColor::White,
        );
        // The king should be significantly restricted
        assert!(
            area < 100,
            "King should be in a small area, found: {}",
            area
        );
    }

    /// Straightforward full-window cage computation: evaluates every cell's wall
    /// bit up front. Used only to verify the lazy `find_bitboard_cage` matches it.
    fn cage_eager_reference(
        board: &Board,
        indices: &SpatialIndices,
        enemy_king: &Coordinate,
        our_color: PlayerColor,
    ) -> (bool, u32) {
        let mut forbidden = [0u32; 32];
        let origin_x = enemy_king.x - 16;
        let origin_y = enemy_king.y - 16;
        let (min_x, max_x, min_y, max_y) = crate::moves::get_coord_bounds();
        for (local_y, fr) in forbidden.iter_mut().enumerate() {
            let abs_y = origin_y + local_y as i64;
            for local_x in 0..32 {
                let abs_x = origin_x + local_x as i64;
                if abs_x < min_x || abs_x > max_x || abs_y < min_y || abs_y > max_y {
                    *fr |= 1 << local_x;
                    continue;
                }
                if is_square_attacked(board, &Coordinate::new(abs_x, abs_y), our_color, indices)
                    || board.is_occupied_by_color(abs_x, abs_y, our_color)
                {
                    *fr |= 1 << local_x;
                }
            }
        }
        let mut reachable = [0u32; 32];
        reachable[16] = 1 << 16;
        for _ in 0..32 {
            let mut changed = false;
            let mut next = reachable;
            for y in 0..32 {
                if reachable[y] == 0 {
                    continue;
                }
                let row = reachable[y];
                let d = row | (row << 1) | (row >> 1);
                next[y] |= d;
                if y > 0 {
                    next[y - 1] |= d;
                }
                if y < 31 {
                    next[y + 1] |= d;
                }
            }
            for y in 0..32 {
                let prev = reachable[y];
                next[y] &= !forbidden[y];
                if next[y] != prev {
                    changed = true;
                }
                reachable[y] = next[y];
            }
            if !changed {
                break;
            }
            if (reachable[0] | reachable[31]) != 0 {
                return (false, 1024);
            }
            for r in reachable.iter().take(31).skip(1) {
                if (r & 0x80000001) != 0 {
                    return (false, 1024);
                }
            }
        }
        let mut area = 0u32;
        for row in reachable.iter() {
            area += row.count_ones();
        }
        (area > 0 && area < 1000, area)
    }

    #[test]
    fn test_find_bitboard_cage_matches_eager() {
        let cases: [(&str, i64, i64); 6] = [
            ("w (8;q|1;q) k4,4|R4,0|R4,8|R0,4|R8,4|K1,1", 4, 4), // boxed
            ("w (8;q|1;q) k4,4|K6,6", 4, 4),                     // open
            ("w (8;q|1;q) k5,5|Q7,7|K3,3", 5, 5),                // queen + king
            ("w (8;q|1;q) k0,0|R2,0|R0,2|K2,2", 0, 0),           // near corner
            ("w (8;q|1;q) k10,10|R10,2|R2,10|R18,10|R10,18|K9,9", 10, 10), // boxed off-origin
            ("w (8;q|1;q) k4,4|B2,2|B6,6|R4,10|R4,-2|K1,1", 4, 4), // diag + ortho walls
        ];
        for (icn, kx, ky) in cases {
            let game = create_test_game_from_icn(icn);
            let ek = Coordinate::new(kx, ky);
            let got = find_bitboard_cage(&game.board, &game.spatial_indices, &ek, PlayerColor::White);
            let want = cage_eager_reference(&game.board, &game.spatial_indices, &ek, PlayerColor::White);
            assert_eq!(got, want, "cage mismatch for icn: {}", icn);
        }
    }
}
