use crate::board::{PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;
use arrayvec::ArrayVec;

/// Maximum pieces we support for full SEE calculation.
/// 128 covers virtually all realistic positions while staying on the stack.
const SEE_MAX_PIECES: usize = 128;

/// Tests if SEE value of move is >= threshold.
/// Uses early cutoffs to avoid full SEE calculation when possible.
#[inline]
pub(crate) fn see_ge(game: &GameState, m: &Move, threshold: i32) -> bool {
    // BITBOARD: Fast piece check
    let captured = match game.board.get_piece(m.to.x, m.to.y) {
        Some(p) => p,
        None => return 0 >= threshold, // No capture: SEE = 0
    };

    let victim_val = get_piece_value(captured.piece_type());
    let attacker_val = get_piece_value(m.piece.piece_type());

    // Early cutoff 1: if capturing loses material even if undefended, fail
    let swap = victim_val - threshold;
    if swap < 0 {
        return false;
    }

    // Early cutoff 2: if capturing wins material even if we lose attacker, pass
    let swap = attacker_val - swap;
    if swap <= 0 {
        return true;
    }

    // Need full SEE for complex cases
    static_exchange_eval_impl(game, m) >= threshold
}

/// Static Exchange Evaluation implementation for a capture move on a single square.
///
/// Returns the net material gain (in centipawns) for the side to move if both
/// sides optimally capture/recapture on the destination square of `m`.
pub(crate) fn static_exchange_eval_impl(game: &GameState, m: &Move) -> i32 {
    // Only meaningful for captures; quiet moves (or moves to empty squares)
    // have no immediate material swing.
    // BITBOARD: Fast piece check
    let captured = match game.board.get_piece(m.to.x, m.to.y) {
        Some(p) => p,
        None => return 0,
    };

    // For very large boards (> SEE_MAX_PIECES), use approximate SEE
    // based on simple MVV-LVA rather than full exchange sequence
    if game.board.len() > SEE_MAX_PIECES {
        let victim_val = get_piece_value(captured.piece_type());
        let attacker_val = get_piece_value(m.piece.piece_type());
        // Simple approximation: gain if victim > attacker, otherwise assume even exchange
        return if victim_val >= attacker_val {
            victim_val - attacker_val
        } else {
            victim_val - attacker_val // Could be negative, which is correct
        };
    }

    #[derive(Clone, Copy)]
    struct PieceInfo {
        x: i64,
        y: i64,
        piece_type: PieceType,
        color: PlayerColor,
        alive: bool,
    }

    // Build piece list using tile bitboards - faster than HashMap iteration
    let mut pieces: ArrayVec<PieceInfo, SEE_MAX_PIECES> = ArrayVec::new();
    for (cx, cy, tile) in game.board.tiles.iter() {
        let mut bits = tile.occ_all;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let packed = tile.piece[idx];
            if packed == 0 {
                continue;
            }
            let piece = crate::board::Piece::from_packed(packed);
            let x = cx * 8 + (idx % 8) as i64;
            let y = cy * 8 + (idx / 8) as i64;
            pieces.push(PieceInfo {
                x,
                y,
                piece_type: piece.piece_type(),
                color: piece.color(),
                alive: true,
            });
        }
    }

    // Helper to find the index of a live piece at given coordinates.
    fn find_piece_index(pieces: &[PieceInfo], x: i64, y: i64) -> Option<usize> {
        for (i, p) in pieces.iter().enumerate() {
            if p.alive && p.x == x && p.y == y {
                return Some(i);
            }
        }
        None
    }

    // Locate the initial target piece in our local list.
    let to_idx = match find_piece_index(&pieces, m.to.x, m.to.y) {
        Some(i) => i,
        None => return 0,
    };

    let target_x = m.to.x;
    let target_y = m.to.y;

    // Current occupant on the target square: type and color.
    let mut occ_type = pieces[to_idx].piece_type;
    let mut _occ_color = pieces[to_idx].color;

    // Swap list of gains.
    let mut gain: [i32; 32] = [0; 32];
    let mut depth: usize = 1;

    // Check if a given live piece can (pseudo-legally) attack the target square,
    // using the local snapshot (pieces) for occupancy. This includes all fairy
    // pieces so that SEE works correctly on arbitrary variants.
    fn can_attack(p: &PieceInfo, tx: i64, ty: i64, pieces: &[PieceInfo]) -> bool {
        use crate::board::PieceType::*;

        if !p.alive {
            return false;
        }

        let dx = tx - p.x;
        let dy = ty - p.y;
        let adx = dx.abs();
        let ady = dy.abs();

        // Helper for sliding moves (rook/bishop/queen-like) over the local
        // snapshot, checking that the ray to (tx, ty) is not blocked.
        // IMPORTANT: We don't iterate step-by-step (would be O(distance)),
        // instead we check if any piece lies strictly between p and the target.
        fn is_clear_ray(p: &PieceInfo, dx: i64, dy: i64, pieces: &[PieceInfo]) -> bool {
            let adx = dx.abs();
            let ady = dy.abs();

            // Determine if this is a valid sliding direction
            let is_ortho = (dx == 0 && dy != 0) || (dy == 0 && dx != 0);
            let is_diag = adx == ady && adx != 0;

            if !is_ortho && !is_diag {
                return false;
            }

            let target_x = p.x + dx;
            let target_y = p.y + dy;

            // Check if any piece lies strictly between p and target
            for other in pieces.iter() {
                if !other.alive {
                    continue;
                }
                // Skip the target square itself
                if other.x == target_x && other.y == target_y {
                    continue;
                }
                // Skip the piece itself
                if other.x == p.x && other.y == p.y {
                    continue;
                }

                let odx = other.x - p.x;
                let ody = other.y - p.y;

                // Check if 'other' is on the same ray and strictly between p and target
                if is_ortho {
                    if dx == 0 {
                        // Vertical ray
                        if odx == 0 {
                            let ody_abs = ody.abs();
                            if ody.signum() == dy.signum() && ody_abs < ady {
                                return false; // Blocker found
                            }
                        }
                    } else {
                        // Horizontal ray
                        if ody == 0 {
                            let odx_abs = odx.abs();
                            if odx.signum() == dx.signum() && odx_abs < adx {
                                return false; // Blocker found
                            }
                        }
                    }
                } else {
                    // Diagonal ray
                    if odx.abs() == ody.abs() && odx.abs() > 0 {
                        if odx.signum() == dx.signum() && ody.signum() == dy.signum() {
                            if odx.abs() < adx {
                                return false; // Blocker found
                            }
                        }
                    }
                }
            }
            true
        }

        match p.piece_type {
            // Standard chess pieces
            Pawn => {
                let dir = match p.color {
                    PlayerColor::White => 1,
                    PlayerColor::Black => -1,
                    PlayerColor::Neutral => return false,
                };
                dy == dir && (dx == 1 || dx == -1)
            }
            Knight => (adx == 1 && ady == 2) || (adx == 2 && ady == 1),
            Bishop => adx == ady && adx != 0 && is_clear_ray(p, dx, dy, pieces),
            Rook => {
                ((dx == 0 && dy != 0) || (dy == 0 && dx != 0)) && is_clear_ray(p, dx, dy, pieces)
            }
            Queen | RoyalQueen => {
                if dx == 0 || dy == 0 || adx == ady {
                    is_clear_ray(p, dx, dy, pieces)
                } else {
                    false
                }
            }
            King | Guard => {
                // One-step king/guard move
                (adx <= 1 && ady <= 1) && (dx != 0 || dy != 0)
            }

            // Leaper fairies
            Giraffe => (adx == 1 && ady == 4) || (adx == 4 && ady == 1),
            Camel => (adx == 1 && ady == 3) || (adx == 3 && ady == 1),
            Zebra => (adx == 2 && ady == 3) || (adx == 3 && ady == 2),

            // Compound pieces
            Amazon => {
                // Queen + knight
                ((dx == 0 || dy == 0 || adx == ady) && is_clear_ray(p, dx, dy, pieces))
                    || ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Chancellor => {
                // Rook + knight
                (((dx == 0 && dy != 0) || (dy == 0 && dx != 0)) && is_clear_ray(p, dx, dy, pieces))
                    || ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Archbishop => {
                // Bishop + knight
                (adx == ady && adx != 0 && is_clear_ray(p, dx, dy, pieces))
                    || ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Centaur | RoyalCentaur => {
                // King + knight
                ((adx <= 1 && ady <= 1) && (dx != 0 || dy != 0))
                    || ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }

            // Hawk: fixed leaper offsets (see is_square_attacked)
            Hawk => {
                matches!(
                    (dx, dy),
                    (2, 0)
                        | (-2, 0)
                        | (0, 2)
                        | (0, -2)
                        | (3, 0)
                        | (-3, 0)
                        | (0, 3)
                        | (0, -3)
                        | (2, 2)
                        | (2, -2)
                        | (-2, 2)
                        | (-2, -2)
                        | (3, 3)
                        | (3, -3)
                        | (-3, 3)
                        | (-3, -3)
                )
            }

            // Knightrider: repeat knight vector in same direction; ignore blockers
            Knightrider => {
                const DIRS: &[(i64, i64)] = &[
                    (1, 2),
                    (2, 1),
                    (-1, 2),
                    (-2, 1),
                    (1, -2),
                    (2, -1),
                    (-1, -2),
                    (-2, -1),
                ];
                for (bx, by) in DIRS {
                    if dx == *bx && dy == *by {
                        return true;
                    }
                    if dx % bx == 0 && dy % by == 0 {
                        let kx = dx / bx;
                        let ky = dy / by;
                        if kx > 0 && kx == ky {
                            return true;
                        }
                    }
                }
                false
            }

            // Huygen: prime-distance orthogonal slider (approximate, ignore blockers)
            Huygen => {
                if (dx == 0 && dy != 0) || (dy == 0 && dx != 0) {
                    let d = if dx == 0 { ady } else { adx };
                    if d > 0 && crate::utils::is_prime_i64(d) {
                        return true;
                    }
                }
                false
            }

            // Rose: approximate as a knight-like leaper for SEE purposes.
            Rose => (adx == 1 && ady == 2) || (adx == 2 && ady == 1),

            // Neutral/blocking pieces do not attack in SEE
            Void | Obstacle => false,
        }
    }

    // Helper to find the least valuable attacker for a given side.
    fn least_valuable_attacker(
        pieces: &[PieceInfo],
        side: PlayerColor,
        tx: i64,
        ty: i64,
    ) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_val: i32 = i32::MAX;

        for (i, p) in pieces.iter().enumerate() {
            if !p.alive || p.color != side || p.piece_type.is_neutral_type() {
                continue;
            }
            if !can_attack(p, tx, ty, pieces) {
                continue;
            }
            let val = get_piece_value(p.piece_type);
            if val < best_val {
                best_val = val;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    // Initialize swap-list with value of the initially captured piece.
    gain[0] = get_piece_value(occ_type);

    // Side to move at the root.
    let mut side = game.turn;

    // First capture: moving piece `m` takes the target.
    // We conceptually move it to the target square and remove the original
    // occupant from play.
    pieces[to_idx].alive = false; // captured
    let attacker_idx_opt = find_piece_index(&pieces, m.from.x, m.from.y);
    let attacker_idx = match attacker_idx_opt {
        Some(i) => i,
        None => return gain[0],
    };

    occ_type = pieces[attacker_idx].piece_type;
    _occ_color = pieces[attacker_idx].color;
    pieces[attacker_idx].alive = false; // attacker now sits on target, but we model it abstractly

    // Alternating sequence of recaptures.
    loop {
        // Switch side to move.
        side = side.opponent();

        if depth >= gain.len() {
            break;
        }

        if let Some(att_idx) = least_valuable_attacker(&pieces, side, target_x, target_y) {
            // Next capture: side captures the current occupant on target.
            let captured_val = get_piece_value(occ_type);
            gain[depth] = captured_val - gain[depth - 1];

            // Update occupant to the capturing piece and remove it from its
            // original square for future x-ray style attacks.
            occ_type = pieces[att_idx].piece_type;
            _occ_color = pieces[att_idx].color;
            pieces[att_idx].alive = false;

            depth += 1;
        } else {
            break;
        }
    }

    // Negamax the swap list backwards to determine best achievable gain.
    while depth > 0 {
        let d = depth - 1;
        if d == 0 {
            break;
        }
        let v = gain[d];
        let prev = gain[d - 1];
        // gain[d-1] = -max(-gain[d-1], gain[d])
        gain[d - 1] = -std::cmp::max(-prev, v);
        depth -= 1;
    }

    gain[0]
}
