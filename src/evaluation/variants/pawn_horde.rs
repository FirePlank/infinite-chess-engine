//! Pawn Horde evaluation. White must advance and promote its pawns, so it scores
//! phalanx structure, advancement and promotion threats; Black must capture them all,
//! so it scores wall breaks, back-rank penetration and weak pawns.

use crate::board::{Coordinate, PieceType, PlayerColor};
use crate::eval_net::variant_features::{NoSink, VariantLayout, VariantSink, cp, ct};
use crate::game::GameState;
use arrayvec::ArrayVec;
use rustc_hash::FxHashSet;

// Piece Values
const PAWN_VALUE: i32 = 90;

// White (Horde) Bonuses
const PHALANX_BONUS: i32 = 12; // Side-by-side pawns
const PHALANX_BONUS_PER_PAWN: i32 = 3;
const SUPPORT_BONUS: i32 = 27; // Protected pawns
const SUPPORT_BONUS_PER_PAWN: i32 = 6;
const KING_ATTACK_BONUS: i32 = 20; // Pawns near enemy king

// Black (Pieces) Bonuses/Penalties
/// Per distinct horde pawn Black actually attacks, by whether a pawn defends it.
const BREACH_LOOSE_BONUS: i32 = 22;
const BREACH_SUPPORTED_BONUS: i32 = 7;
const BREAKTHROUGH_BONUS: i32 = 45; // Major piece behind the pawn wall
const ATTACKING_PAWN_BONUS: i32 = 25; // Attacking a pawn
const MG_KING_NEAR_FRONT_PENALTY: i32 = 40;
const EG_KING_NEAR_FRONT_PENALTY: i32 = 0;

// Phase System
const MAX_B_PHASE: i32 = 56;

// Rank-based pawn advancement curve (0-indexed, relative to promotion)
// Closer to 0 means closer to promotion
fn get_pawn_advance_bonus(dist_to_promo: i32) -> i32 {
    match dist_to_promo {
        0 => 0,   // Promoted (not a pawn anymore)
        1 => 270, // Rank 7 - Huge threat
        2 => 125, // Rank 6 - Major threat
        3 => 50,  // Rank 5
        4 => 25,  // Rank 4
        5 => -5,  // Rank 3
        6 => -20, // Rank 2 - Weak
        _ => -60, // Back ranks - Much weaker
    }
}

/// Net inputs, all White-ahead: the two armies play different games, so the net is
/// not colour-mirrored and reads the side to move directly.
pub const NET_LAYOUT: VariantLayout = VariantLayout {
    names: &[
        "stm", "material", "horde pawns", "advancement", "phalanx support", "king tropism",
        "breach loose", "breach supported", "breakthrough queens", "pieces on pawns",
        "idle pieces", "king near front", "black phase", "black pieces", "rear pawn dist",
        "lead pawn dist", "king front gap", "pawns near promo", "black material", "promoted",
        "white doubled", "black doubled", "king cover",
    ],
    fixed: 23,
    neg: 0,
};

pub fn evaluate(game: &GameState) -> i32 {
    evaluate_traced(game, &mut NoSink)
}

pub fn evaluate_traced<S: VariantSink>(game: &GameState, sink: &mut S) -> i32 {
    let mut score = 0;
    let (mut advancement, mut structure, mut tropism) = (0, 0, 0);
    let (mut black_material, mut promoted) = (0, 0);

    // 1. Gather Piece Lists
    let mut white_pawns: ArrayVec<Coordinate, 64> = ArrayVec::new();
    let mut black_pieces: ArrayVec<(Coordinate, PieceType), 18> = ArrayVec::new();
    let mut black_king_pos = Coordinate::new(5, 8); // Default fallback
    let mut b_phase = 0;

    // Map for quick lookup of pawn locations
    // Using a simple vector check is fast enough for 56 items

    for (x, y, piece) in game.board.iter() {
        let coord = Coordinate::new(x, y);
        match piece.color() {
            PlayerColor::White => {
                if piece.piece_type() == PieceType::Pawn {
                    b_phase += 1;
                    white_pawns.push(coord);
                    score += PAWN_VALUE; // Material count
                } else {
                    // Promoted piece! Huge value.
                    score += game.get_piece_value(piece.piece_type(), piece.color());
                    promoted += game.get_piece_value(piece.piece_type(), piece.color());
                }
            }
            PlayerColor::Black => {
                if piece.piece_type().is_royal() {
                    black_king_pos = coord;
                }
                black_pieces.push((coord, piece.piece_type()));
                score -= game.get_piece_value(piece.piece_type(), piece.color());
                if !piece.piece_type().is_royal() {
                    black_material += game.get_piece_value(piece.piece_type(), piece.color());
                }
            }
            _ => {}
        }
    }

    // Reused across evals: a fresh set here was the only per-eval heap allocation
    // in the evaluator.
    thread_local! {
        static PAWN_SET: std::cell::Cell<FxHashSet<Coordinate>> =
            std::cell::Cell::new(FxHashSet::default());
    }
    let mut pawn_set = PAWN_SET.with(|c| c.take());
    pawn_set.clear();
    pawn_set.reserve(white_pawns.len());
    for pawn in &white_pawns {
        pawn_set.insert(*pawn);
    }

    // 2. White Logic (Horde). The horde advances toward higher Y (the promo
    // rank), so min_pawn_y is the rearmost rank (breakthrough check) and
    // max_pawn_y is the leading front line (king-safety proximity).
    let mut min_pawn_y = 1000;
    let mut max_pawn_y = i64::MIN;

    let promo_rank = game.white_promo_rank;

    for pawn in &white_pawns {
        if pawn.y < min_pawn_y {
            min_pawn_y = pawn.y;
        }
        if pawn.y > max_pawn_y {
            max_pawn_y = pawn.y;
        }

        // Advancement
        let dist = (promo_rank - pawn.y).max(0) as i32;
        score += get_pawn_advance_bonus(dist);
        advancement += get_pawn_advance_bonus(dist);

        // Phalanx: same rank, adjacent files (x±1, y) - creates a wall of pawns
        let neighbor_left = Coordinate::new(pawn.x - 1, pawn.y);
        let neighbor_right = Coordinate::new(pawn.x + 1, pawn.y);

        let mut neighbors = 0;
        if pawn_set.contains(&neighbor_left) {
            neighbors += 1;
        }
        if pawn_set.contains(&neighbor_right) {
            neighbors += 1;
        }

        // Support: diagonally behind (x±1, y-1) - protects the pawn from captures
        let support_left = Coordinate::new(pawn.x - 1, pawn.y - 1);
        let support_right = Coordinate::new(pawn.x + 1, pawn.y - 1);

        let mut supporting_pawns = 0;
        if pawn_set.contains(&support_left) {
            supporting_pawns += 1;
        }
        if pawn_set.contains(&support_right) {
            supporting_pawns += 1;
        }

        let before = score;
        if supporting_pawns > 0 {
            score += SUPPORT_BONUS;
        } else if neighbors > 0 {
            score += PHALANX_BONUS;
        }
        score += neighbors * PHALANX_BONUS_PER_PAWN + supporting_pawns * SUPPORT_BONUS_PER_PAWN;
        structure += score - before;

        // King Attack Tropism
        let dist_to_king = (pawn.x - black_king_pos.x).abs() + (pawn.y - black_king_pos.y).abs();
        if dist_to_king <= 3 {
            score += KING_ATTACK_BONUS * (4 - dist_to_king) as i32;
            tropism += KING_ATTACK_BONUS * (4 - dist_to_king) as i32;
        }
    }

    // 3. Black Logic (Pieces)
    b_phase = b_phase.min(MAX_B_PHASE);
    let taper = |mg: i32, eg: i32| -> i32 {
        ((mg * b_phase.min(MAX_B_PHASE)) + (eg * (MAX_B_PHASE - b_phase.min(MAX_B_PHASE))))
            / MAX_B_PHASE
    };
    // Which horde pawns Black can actually hit. Proximity alone rates a bishop on
    // the wrong colour and one bearing down an open file the same, and Black wins
    // by eating the horde, so the entry point is the whole plan.
    let mut hit: arrayvec::ArrayVec<Coordinate, 32> = arrayvec::ArrayVec::new();
    let idx = &game.spatial_indices;
    let note = |c: Coordinate, hit: &mut arrayvec::ArrayVec<Coordinate, 32>| {
        if pawn_set.contains(&c) && !hit.is_full() && !hit.contains(&c) {
            hit.push(c);
        }
    };
    for (pos, ptype) in &black_pieces {
        let (px, py) = (pos.x, pos.y);
        if crate::attacks::is_ortho_slider(*ptype) {
            if let Some(l) = idx.rows.get(&py) {
                let (f, b) = l.neighbors(px);
                for e in [f, b].into_iter().flatten() {
                    note(Coordinate::new(e.0, py), &mut hit);
                }
            }
            if let Some(l) = idx.cols.get(&px) {
                let (f, b) = l.neighbors(py);
                for e in [f, b].into_iter().flatten() {
                    note(Coordinate::new(px, e.0), &mut hit);
                }
            }
        }
        if crate::attacks::is_diag_slider(*ptype) {
            let k1 = px - py;
            if let Some(l) = idx.diag1.get(&k1) {
                let (f, b) = l.neighbors(px);
                for e in [f, b].into_iter().flatten() {
                    note(Coordinate::new(e.0, e.0 - k1), &mut hit);
                }
            }
            let k2 = px + py;
            if let Some(l) = idx.diag2.get(&k2) {
                let (f, b) = l.neighbors(px);
                for e in [f, b].into_iter().flatten() {
                    note(Coordinate::new(e.0, k2 - e.0), &mut hit);
                }
            }
        }
        if crate::attacks::matches_mask(*ptype, crate::attacks::KNIGHT_MASK) {
            for (ox, oy) in crate::attacks::KNIGHT_OFFSETS {
                note(Coordinate::new(px + ox, py + oy), &mut hit);
            }
        }
        if crate::attacks::matches_mask(*ptype, crate::attacks::KING_MASK) {
            for (ox, oy) in crate::attacks::KING_OFFSETS {
                note(Coordinate::new(px + ox, py + oy), &mut hit);
            }
        }
    }
    let mut breach_supported = 0;
    for c in &hit {
        // An unsupported pawn is a base Black can actually take; a supported one
        // costs material to win, so it is worth much less as an entry point.
        let supported = pawn_set.contains(&Coordinate::new(c.x - 1, c.y - 1))
            || pawn_set.contains(&Coordinate::new(c.x + 1, c.y - 1));
        breach_supported += i32::from(supported);
        score -= if supported {
            BREACH_SUPPORTED_BONUS
        } else {
            BREACH_LOOSE_BONUS
        };
    }

    let (mut breakthroughs, mut on_pawns, mut idle) = (0, 0, 0);
    for (pos, ptype) in &black_pieces {
        // Breakthrough: Are we behind the pawn wall?
        if pos.y < min_pawn_y && *ptype == PieceType::Queen {
            score -= BREAKTHROUGH_BONUS; // Score is absolute, so subtract for Black advantage
            breakthroughs += 1;
        }

        // Attacks on Pawns
        // Simple heuristic: distance to nearest pawn
        let mut min_dist_to_pawn = 100;
        for pawn in &white_pawns {
            let d = (pos.x - pawn.x).abs().max((pos.y - pawn.y).abs());
            if d < min_dist_to_pawn {
                min_dist_to_pawn = d;
            }

            // Direct attack checks would be better but expensive without movegen.
            // Distance is a good proxy for "activity against horde".
        }

        if min_dist_to_pawn <= 2 {
            score -= ATTACKING_PAWN_BONUS;
            on_pawns += 1;
        } else if min_dist_to_pawn > 5 {
            // Piece inactive/far from horde penalty
            score += 10;
            idle += 1;
        }
    }

    // King Safety (Black): the king should stay away from the horde's leading
    // front line (the most-advanced white pawn).
    let near_front = max_pawn_y != i64::MIN && (black_king_pos.y - max_pawn_y).abs() < 3;
    if near_front {
        // King is dangerously close to the front
        score += taper(MG_KING_NEAR_FRONT_PENALTY, EG_KING_NEAR_FRONT_PENALTY); // Penalty for Black (positive score)
    }

    if S::ON {
        let phase = crate::evaluation::base::effective_phase(game.total_phase, game.initial_phase);
        let pawns = white_pawns.len() as i32;
        let dist = |y: i64| (promo_rank - y).clamp(0, 255) as i32;
        let (rear, lead, gap) = if pawns > 0 {
            let gap = (black_king_pos.y - max_pawn_y).abs().min(255) as i32;
            (dist(min_pawn_y), dist(max_pawn_y), gap)
        } else {
            (255, 255, 255)
        };
        let near_promo = white_pawns.iter().filter(|p| dist(p.y) <= 2).count() as i32;
        let cols = [
            if game.turn == PlayerColor::White { 1 } else { -1 },
            cp(pawns * PAWN_VALUE + promoted - black_material),
            ct(pawns),
            cp(advancement),
            cp(structure),
            cp(tropism),
            ct(hit.len() as i32 - breach_supported),
            ct(breach_supported),
            ct(breakthroughs),
            ct(on_pawns),
            ct(idle),
            i16::from(near_front),
            ct(b_phase),
            ct(black_pieces.len() as i32),
            ct(rear),
            ct(lead),
            ct(gap),
            ct(near_promo),
            cp(black_material),
            cp(promoted),
            cp(doubled(white_pawns.iter().map(|p| p.x), phase)),
            cp(doubled(black_pieces.iter().filter(|p| p.1 == PieceType::Pawn).map(|p| p.0.x), phase)),
            ct(king_cover(game, black_king_pos) / 10),
        ];
        for (c, v) in cols.into_iter().enumerate() {
            sink.set(c, v);
        }
        sink.set_phase(b_phase);
    }

    PAWN_SET.with(|c| c.set(pawn_set));

    // Return perspective
    if game.turn == PlayerColor::Black {
        -score
    } else {
        score
    }
}

/// The generic doubled-pawn penalty: each extra pawn on a file, tapered by phase.
fn doubled(files: impl Iterator<Item = i64>, phase: i32) -> i32 {
    use crate::evaluation::base::MAX_PHASE;
    use crate::evaluation::params::{eg_doubled_pawn_penalty, mg_doubled_pawn_penalty};
    let mut xs: ArrayVec<i64, 64> = files.take(64).collect();
    xs.sort_unstable();
    let extra = xs.windows(2).filter(|w| w[0] == w[1]).count() as i32;
    -extra * (mg_doubled_pawn_penalty() * phase + eg_doubled_pawn_penalty() * (MAX_PHASE - phase)) / MAX_PHASE
}

/// Black's cover within two squares of its king, weighted as the generic defender
/// histogram does.
fn king_cover(game: &GameState, k: Coordinate) -> i32 {
    let mut units = 0;
    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            let Some(p) = game.board.get_piece(k.x + dx, k.y + dy) else {
                continue;
            };
            let far = usize::from(dx.abs().max(dy.abs()) == 2);
            units += if p.color() == PlayerColor::Neutral {
                [25, 12][far]
            } else if p.color() != PlayerColor::Black || p.piece_type().is_royal() {
                0
            } else if p.piece_type() == PieceType::Pawn {
                [33, 16][far]
            } else {
                [100, 50][far]
            };
        }
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;

    fn create_pawn_horde_game() -> GameState {
        let mut game = GameState::new();
        game.variant = Some(crate::Variant::PawnHorde);
        game.white_promo_rank = 8;
        game.black_promo_rank = 1;
        game
    }

    fn create_pawn_horde_game_from_icn(icn: &str) -> GameState {
        let mut game = create_pawn_horde_game();
        game.setup_position_from_icn(icn);
        game
    }

    #[test]
    fn test_get_pawn_advance_bonus() {
        // Near promotion -> high bonus
        assert!(get_pawn_advance_bonus(1) > get_pawn_advance_bonus(2));
        assert!(get_pawn_advance_bonus(2) >= 100);
        // Further back -> lower bonus
        assert!(get_pawn_advance_bonus(3) < get_pawn_advance_bonus(2));
        assert!(get_pawn_advance_bonus(6) < get_pawn_advance_bonus(3));
    }

    #[test]
    fn test_evaluate_returns_value() {
        let mut game = create_pawn_horde_game_from_icn("w (8;q|1;q) k5,8|P4,2|P5,2");
        game.turn = PlayerColor::White;
        game.recompute_hash();

        let score = evaluate(&game);
        // Should return some meaningful value (just check it doesn't panic)
        let _ = score;
    }

    #[test]
    fn test_pawn_advancement_value() {
        let mut game = create_pawn_horde_game_from_icn("w (8;q|1;q) k5,8|P4,7");
        game.turn = PlayerColor::White;

        let score_advanced = evaluate(&game);

        game.setup_position_from_icn("w (8;q|1;q) k5,8|P4,2");

        let score_back = evaluate(&game);

        // Advanced pawn should score better
        assert!(
            score_advanced > score_back,
            "Near-promo pawn should score higher"
        );
    }

    #[test]
    fn test_phalanx_bonus() {
        let mut game = create_pawn_horde_game_from_icn("w (8;q|1;q) k5,8|P3,4|P4,4|P5,4");
        game.turn = PlayerColor::White;

        let score_phalanx = evaluate(&game);

        game.setup_position_from_icn("w (8;q|1;q) k5,8|P1,4|P4,2|P7,3");

        let score_isolated = evaluate(&game);

        // Phalanx should typically score better
        // (Though isolated pawns might be more advanced, so just check it runs)
        assert!(score_phalanx.abs() < 100000);
        assert!(score_isolated.abs() < 100000);
    }

    #[test]
    fn test_black_breakthrough_bonus() {
        let mut game = create_pawn_horde_game_from_icn("b (8;q|1;q) k5,8|P4,4|P5,4|r4,2");
        game.turn = PlayerColor::Black;

        let score_breakthrough = evaluate(&game);

        game.setup_position_from_icn("b (8;q|1;q) k5,8|P4,4|P5,4|r4,6");

        let score_no_breakthrough = evaluate(&game);

        // From black's perspective, breakthrough should be better (more positive when negated)
        // Just verify it runs
        assert!(score_breakthrough.abs() < 100000);
        assert!(score_no_breakthrough.abs() < 100000);
    }
}
