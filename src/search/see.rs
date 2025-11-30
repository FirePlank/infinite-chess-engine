use crate::board::{PieceType, PlayerColor};
use crate::evaluation::get_piece_value;
use crate::game::GameState;
use crate::moves::Move;

/// Static Exchange Evaluation implementation for a capture move on a single square.
///
/// Returns the net material gain (in centipawns) for the side to move if both
/// sides optimally capture/recapture on the destination square of `m`.
pub(crate) fn static_exchange_eval_impl(game: &GameState, m: &Move) -> i32 {
    // Only meaningful for captures; quiet moves (or moves to empty squares)
    // have no immediate material swing.
    if game.board.get_piece(&m.to.x, &m.to.y).is_none() {
        return 0;
    }

    #[derive(Clone, Copy)]
    struct PieceInfo {
        x: i64,
        y: i64,
        piece_type: PieceType,
        color: PlayerColor,
        alive: bool,
    }

    // Build piece list from the board HashMap.
    let mut pieces: Vec<PieceInfo> = Vec::with_capacity(game.board.pieces.len());
    for ((x, y), piece) in &game.board.pieces {
        pieces.push(PieceInfo {
            x: *x,
            y: *y,
            piece_type: piece.piece_type,
            color: piece.color,
            alive: true,
        });
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
        fn is_clear_ray(p: &PieceInfo, dx: i64, dy: i64, pieces: &[PieceInfo]) -> bool {
            let adx = dx.abs();
            let ady = dy.abs();

            let (step_x, step_y, steps) = if dx == 0 {
                (0, dy.signum(), ady)
            } else if dy == 0 {
                (dx.signum(), 0, adx)
            } else if adx == ady {
                (dx.signum(), dy.signum(), adx)
            } else {
                return false;
            };

            for s in 1..steps {
                let ix = p.x + step_x * s;
                let iy = p.y + step_y * s;
                if find_piece_index(pieces, ix, iy).is_some() {
                    return false;
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
            Knight => {
                (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
            }
            Bishop => {
                adx == ady && adx != 0 && is_clear_ray(p, dx, dy, pieces)
            }
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
            Giraffe => {
                (adx == 1 && ady == 4) || (adx == 4 && ady == 1)
            }
            Camel => {
                (adx == 1 && ady == 3) || (adx == 3 && ady == 1)
            }
            Zebra => {
                (adx == 2 && ady == 3) || (adx == 3 && ady == 2)
            }

            // Compound pieces
            Amazon => {
                // Queen + knight
                ((dx == 0 || dy == 0 || adx == ady) && is_clear_ray(p, dx, dy, pieces)) ||
                    ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Chancellor => {
                // Rook + knight
                (((dx == 0 && dy != 0) || (dy == 0 && dx != 0)) && is_clear_ray(p, dx, dy, pieces)) ||
                    ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Archbishop => {
                // Bishop + knight
                (adx == ady && adx != 0 && is_clear_ray(p, dx, dy, pieces)) ||
                    ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }
            Centaur | RoyalCentaur => {
                // King + knight
                ((adx <= 1 && ady <= 1) && (dx != 0 || dy != 0)) ||
                    ((adx == 1 && ady == 2) || (adx == 2 && ady == 1))
            }

            // Hawk: fixed leaper offsets (see is_square_attacked)
            Hawk => {
                matches!((dx, dy),
                    (2, 0) | (-2, 0) | (0, 2) | (0, -2) |
                    (3, 0) | (-3, 0) | (0, 3) | (0, -3) |
                    (2, 2) | (2, -2) | (-2, 2) | (-2, -2) |
                    (3, 3) | (3, -3) | (-3, 3) | (-3, -3)
                )
            }

            // Knightrider: repeat knight vector in same direction; ignore blockers
            Knightrider => {
                const DIRS: &[(i64, i64)] = &[
                    (1, 2), (2, 1), (-1, 2), (-2, 1),
                    (1, -2), (2, -1), (-1, -2), (-2, -1),
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
            Rose => {
                (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
            }

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
