use crate::board::{Board, PieceType, PlayerColor};
use rustc_hash::FxHashMap;
use std::cell::RefCell;

thread_local! {
    static MATERIAL_CACHE: RefCell<FxHashMap<u64, bool>> = RefCell::new(FxHashMap::default());
}

pub fn clear_material_cache() {
    MATERIAL_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[inline]
fn get_best_promotion_piece(game_rules: &crate::game::GameRules) -> Option<PieceType> {
    game_rules
        .promotion_types
        .as_ref()
        .filter(|t| !t.is_empty())
        .and_then(|types| {
            types
                .iter()
                .max_by_key(|pt| super::base::get_piece_value_base(**pt))
                .copied()
        })
}

#[inline]
fn can_pawn_promote(y: i64, color: PlayerColor, game_rules: &crate::game::GameRules) -> bool {
    let promo_ranks = match color {
        PlayerColor::White => &game_rules.promotion_ranks.white,
        PlayerColor::Black => &game_rules.promotion_ranks.black,
        PlayerColor::Neutral => return false,
    };
    match color {
        PlayerColor::White => promo_ranks.iter().any(|&rank| rank > y),
        PlayerColor::Black => promo_ranks.iter().any(|&rank| rank < y),
        PlayerColor::Neutral => false,
    }
}

// Material counts for one color. Bishops split by parity (majority, minority).
// u8 is sufficient: the caller bails out at >= 6 total pieces, so no count exceeds 5.
#[derive(Debug, Default)]
struct Mat {
    kings: u8,
    queens: u8,
    rooks: u8,
    knights: u8,
    bishops_maj: u8,
    bishops_min: u8,
    chancellors: u8,
    archbishops: u8,
    hawks: u8,
    guards: u8,
    pawns: u8,
    amazons: u8,
    knightriders: u8,
    huygens: u8,
    royal_centaurs: u8,
}

impl Mat {
    #[inline]
    fn non_royal(&self) -> u8 {
        self.queens
            + self.rooks
            + self.knights
            + self.bishops_maj
            + self.bishops_min
            + self.chancellors
            + self.archbishops
            + self.hawks
            + self.guards
            + self.pawns
            + self.amazons
            + self.knightriders
            + self.huygens
    }
}

/// Helper: checks if all "exotic" pieces are zero.
#[inline]
fn no_exotic_pieces(m: &Mat) -> bool {
    m.chancellors == 0
        && m.archbishops == 0
        && m.hawks == 0
        && m.guards == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
}

// Decision tree for insufficient material detection.
#[inline]
fn is_insufficient(m: &Mat) -> bool {
    // ===== INSUFFICIENT CASES (return true - cannot deliver mate) =====
    // Only royals - no other pieces
    if no_exotic_pieces(m)
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
    {
        return true;
    }

    // Single queen
    if m.queens == 1
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // Less than 4 knights
    if m.knights < 4
        && m.queens == 0
        && m.rooks == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // Less than 4 bishops
    if m.bishops_maj + m.bishops_min < 4
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // Single Chancellor alone
    if m.chancellors == 1
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && m.archbishops == 0
        && m.hawks == 0
        && m.guards == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
    {
        return true;
    }

    // Less than 3 guards
    if m.guards <= 2
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.chancellors == 0
        && m.archbishops == 0
        && m.hawks == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
    {
        return true;
    }

    // N with bishops
    if m.knights == 1
        && m.bishops_maj >= 1
        && m.bishops_min <= 1
        && m.queens == 0
        && m.rooks == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // H+B
    if m.hawks == 1
        && m.bishops_maj >= 1
        && m.bishops_min <= 1
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.chancellors == 0
        && m.archbishops == 0
        && m.guards == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
    {
        return true;
    }

    // AB with pieces
    if m.archbishops == 1
        && m.bishops_maj >= 1
        && m.bishops_min == 0
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.chancellors == 0
        && m.hawks == 0
        && m.guards == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
    {
        return true;
    }
    if m.archbishops == 1
        && m.knights <= 2
        && m.queens == 0
        && m.rooks == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.chancellors == 0
        && m.hawks == 0
        && m.guards == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
        && m.huygens == 0
    {
        return true;
    }

    // R alone
    if m.rooks == 1
        && m.queens == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // R+N alone
    if m.rooks == 1
        && m.knights == 1
        && m.queens == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // R+single bishop
    if m.rooks == 1
        && m.bishops_maj + m.bishops_min == 1
        && m.queens == 0
        && m.knights == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // Pawns 1-3 alone
    if m.pawns >= 1
        && m.pawns <= 3
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    // Huygens 1-4 alone
    if m.huygens >= 1
        && m.huygens <= 4
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.chancellors == 0
        && m.archbishops == 0
        && m.hawks == 0
        && m.guards == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
    {
        return true;
    }

    false
}

/// Bordered variant (smaller map).
#[inline]
fn is_insufficient_bordered(m: &Mat) -> bool {
    // ===== INSUFFICIENT CASES (return true) =====
    // Huygens 1-4 alone (less than 5 is insufficient)
    if m.huygens >= 1
        && m.huygens <= 4
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.chancellors == 0
        && m.archbishops == 0
        && m.hawks == 0
        && m.guards == 0
        && m.pawns == 0
        && m.amazons == 0
        && m.knightriders == 0
    {
        return true;
    }
    
    // Only royals
    if no_exotic_pieces(m)
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
    {
        return true;
    }

    // Bishops only (same color - insufficient; opposite color - sufficient)
    if (m.bishops_maj >= 1 || m.bishops_min >= 1)
        && m.queens == 0
        && m.rooks == 0
        && m.knights == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        // Opposite-color bishops (both maj and min) are sufficient
        if m.bishops_maj >= 1 && m.bishops_min >= 1 {
            return false;
        }
        // Same-color bishops only are insufficient
        return true;
    }

    // 2 knights
    if m.knights <= 2
        && m.queens == 0
        && m.rooks == 0
        && m.bishops_maj == 0
        && m.bishops_min == 0
        && m.pawns == 0
        && no_exotic_pieces(m)
    {
        return true;
    }

    false
}

/// Count material for both colors in a single board pass.
#[inline]
fn count_both(
    board: &Board,
    rules: &crate::game::GameRules,
) -> (Mat, Mat) {
    let mut w = Mat::default();
    let mut b = Mat::default();
    let mut w_lb: u8 = 0;
    let mut w_db: u8 = 0;
    let mut b_lb: u8 = 0;
    let mut b_db: u8 = 0;
    
    let best_promo = get_best_promotion_piece(rules);

    for (x, y, piece) in board.iter() {
        let color = piece.color();
        let m = match color {
            PlayerColor::White => &mut w,
            PlayerColor::Black => &mut b,
            _ => continue,
        };
        let (lb, db) = match color {
            PlayerColor::White => (&mut w_lb, &mut w_db),
            PlayerColor::Black => (&mut b_lb, &mut b_db),
            _ => unreachable!(),
        };

        let pt = piece.piece_type();
        let ept = if pt == PieceType::Pawn && can_pawn_promote(y, color, rules) {
            // Pawn can promote: count as best promotion piece
            best_promo.unwrap_or(PieceType::Queen)
        } else {
            pt
        };

        match ept {
            PieceType::King | PieceType::RoyalQueen => m.kings += 1,
            PieceType::RoyalCentaur => m.royal_centaurs += 1,
            PieceType::Queen => m.queens += 1,
            PieceType::Rook => m.rooks += 1,
            PieceType::Bishop => {
                if (x + y) % 2 == 0 {
                    *lb += 1;
                } else {
                    *db += 1;
                }
            }
            PieceType::Knight => m.knights += 1,
            PieceType::Chancellor => m.chancellors += 1,
            PieceType::Archbishop => m.archbishops += 1,
            PieceType::Hawk => m.hawks += 1,
            PieceType::Guard => m.guards += 1,
            PieceType::Pawn => m.pawns += 1,
            PieceType::Amazon => m.amazons += 1,
            PieceType::Knightrider => m.knightriders += 1,
            PieceType::Huygen => m.huygens += 1,
            _ => {}
        }
    }

    // Bishops: majority/minority parity
    if w_lb >= w_db {
        w.bishops_maj = w_lb;
        w.bishops_min = w_db;
    } else {
        w.bishops_maj = w_db;
        w.bishops_min = w_lb;
    }
    if b_lb >= b_db {
        b.bishops_maj = b_lb;
        b.bishops_min = b_db;
    } else {
        b.bishops_maj = b_db;
        b.bishops_min = b_lb;
    }

    (w, b)
}

/// Returns true if the position is a draw by insufficient material.
#[inline]
pub fn evaluate_insufficient_material(game: &crate::game::GameState) -> bool {
    if (game.white_piece_count + game.black_piece_count) >= 6 {
        return false;
    }

    // Only check insufficient material if both sides have checkmate as win condition
    if game.game_rules.white_win_condition != crate::game::WinCondition::Checkmate
        || game.game_rules.black_win_condition != crate::game::WinCondition::Checkmate
    {
        return false;
    }

    let hash = game.material_hash;
    let has_pawns = game.white_pawn_count > 0 || game.black_pawn_count > 0;

    if !has_pawns {
        let cached = MATERIAL_CACHE.with(|c| c.borrow().get(&hash).copied());
        if let Some(result) = cached {
            return result;
        }
    }

    let result = compute(game);

    if !has_pawns {
        MATERIAL_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 4096 {
                c.clear();
            }
            c.insert(hash, result);
        });
    }

    result
}

#[inline(always)]
fn compute(game: &crate::game::GameState) -> bool {
    let bordered = crate::moves::get_world_size() <= 200;
    let (w, b) = count_both(&game.board, &game.game_rules);

    let w_nr = w.non_royal();
    let b_nr = b.non_royal();

    // Special: only royals on both sides
    if w_nr == 0 && b_nr == 0 {
        if w.kings > 0 && b.kings > 0 && w.royal_centaurs == 0 && b.royal_centaurs == 0 {
            return true;
        }
        if w.royal_centaurs > 0 && b.royal_centaurs > 0 && w.kings == 0 && b.kings == 0 {
            return true;
        }
    }

    // Special: royal centaur vs amazon (unbounded)
    if !bordered {
        if b.royal_centaurs == 1
            && w.amazons == 1
            && w_nr == 1
            && b_nr == 0
            && w.kings == 0
            && w.royal_centaurs == 0
            && b.kings == 0
        {
            return true;
        }
        if w.royal_centaurs == 1
            && b.amazons == 1
            && b_nr == 1
            && w_nr == 0
            && b.kings == 0
            && b.royal_centaurs == 0
            && w.kings == 0
        {
            return true;
        }
    }

    // Cross-color: K+R vs K+R (unbounded)
    if !bordered
        && w.kings >= 1
        && b.kings >= 1
        && w.rooks == 1
        && b.rooks == 1
        && w_nr == 1
        && b_nr == 1
    {
        return true;
    }

    // Special: 2K + R vs K (one side has 2 kings with rook, other has just king)
    if w.kings >= 2 && w.rooks == 1 && w_nr == 1 && b.kings >= 1 && b_nr == 0 {
        return false;
    }
    if b.kings >= 2 && b.rooks == 1 && b_nr == 1 && w.kings >= 1 && w_nr == 0 {
        return false;
    }

    // Check both attack directions (decision tree directly)
    // Both sides must have a king to reach here (due to checkmate win condition requirement)
    let w_insuff = if bordered {
        is_insufficient_bordered(&w)
    } else {
        is_insufficient(&w)
    };
    if !w_insuff {
        return false;
    }
    let b_insuff = if bordered {
        is_insufficient_bordered(&b)
    } else {
        is_insufficient(&b)
    };

    w_insuff && b_insuff
}

/// Returns true if the combination of (attacker, defender) material represents
/// a helpmate-possible endgame that game handlers should NOT auto-declare as a draw.
/// Both `a` and `b` are already "individually insufficient" at this point.
/// Detects cross-board combinations where helpmate is theoretically possible
/// despite both sides being individually insufficient.
#[inline]
fn is_helpmate_only_combo(a: &Mat, b: &Mat, bordered: bool) -> bool {
    // R+B vs Q (either direction)
    let rb_vs_q = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && (x.bishops_maj + x.bishops_min) == 1
            && x.queens == 0
            && x.knights == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.queens == 1
            && y.rooks == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rb_vs_q(a, b) || rb_vs_q(b, a) {
        return true;
    }

    // R+N vs Q (either direction)
    let rn_vs_q = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && x.knights == 1
            && x.queens == 0
            && (x.bishops_maj + x.bishops_min) == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.queens == 1
            && y.rooks == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rn_vs_q(a, b) || rn_vs_q(b, a) {
        return true;
    }

    // R+N vs R (either direction)
    let rn_vs_r = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && x.knights == 1
            && x.queens == 0
            && (x.bishops_maj + x.bishops_min) == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.rooks == 1
            && y.queens == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rn_vs_r(a, b) || rn_vs_r(b, a) {
        return true;
    }

    // R+B vs R (either direction)
    let rb_vs_r = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && (x.bishops_maj + x.bishops_min) == 1
            && x.queens == 0
            && x.knights == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.rooks == 1
            && y.queens == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rb_vs_r(a, b) || rb_vs_r(b, a) {
        return true;
    }

    // R+B vs B (either direction)
    let rb_vs_b = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && (x.bishops_maj + x.bishops_min) == 1
            && x.queens == 0
            && x.knights == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && (y.bishops_maj + y.bishops_min) == 1
            && y.queens == 0
            && y.rooks == 0
            && y.knights == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rb_vs_b(a, b) || rb_vs_b(b, a) {
        return true;
    }

    // R+N vs B (either direction)
    let rn_vs_b = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && x.knights == 1
            && x.queens == 0
            && (x.bishops_maj + x.bishops_min) == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && (y.bishops_maj + y.bishops_min) == 1
            && y.queens == 0
            && y.rooks == 0
            && y.knights == 0
            && y.pawns == 0
            && no_exotic_pieces(y)
            && y.non_royal() == 1
    };
    if rn_vs_b(a, b) || rn_vs_b(b, a) {
        return true;
    }

    // Two pieces vs pawn (pawn not yet promoted)
    let rb_vs_p = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && (x.bishops_maj + x.bishops_min) == 1
            && x.queens == 0
            && x.knights == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.pawns >= 1
            && y.queens == 0
            && y.rooks == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && no_exotic_pieces(y)
            && y.non_royal() == y.pawns as u8
    };
    if rb_vs_p(a, b) || rb_vs_p(b, a) {
        return true;
    }

    // Two pieces vs pawn (pawn not yet promoted)
    let rn_vs_p = |x: &Mat, y: &Mat| {
        x.rooks == 1
            && x.knights == 1
            && x.queens == 0
            && (x.bishops_maj + x.bishops_min) == 0
            && x.pawns == 0
            && no_exotic_pieces(x)
            && x.non_royal() == 2
            && y.pawns >= 1
            && y.queens == 0
            && y.rooks == 0
            && y.knights == 0
            && (y.bishops_maj + y.bishops_min) == 0
            && no_exotic_pieces(y)
            && y.non_royal() == y.pawns as u8
    };
    if rn_vs_p(a, b) || rn_vs_p(b, a) {
        return true;
    }

    // Bounded-only helpmate combos
    if bordered {
        // B vs B opposite colors (either direction)
        let b_vs_b_opposite = |x: &Mat, y: &Mat| {
            x.queens == 0
                && x.rooks == 0
                && x.knights == 0
                && x.pawns == 0
                && no_exotic_pieces(x)
                && x.non_royal() == 1
                && y.queens == 0
                && y.rooks == 0
                && y.knights == 0
                && y.pawns == 0
                && no_exotic_pieces(y)
                && y.non_royal() == 1
                && ((x.bishops_maj == 1 && x.bishops_min == 0 && y.bishops_maj == 0 && y.bishops_min == 1)
                    || (x.bishops_maj == 0 && x.bishops_min == 1 && y.bishops_maj == 1 && y.bishops_min == 0))
        };
        if b_vs_b_opposite(a, b) || b_vs_b_opposite(b, a) {
            return true;
        }

        // N vs B (either direction)
        let n_vs_b = |x: &Mat, y: &Mat| {
            x.knights == 1
                && x.queens == 0
                && x.rooks == 0
                && (x.bishops_maj + x.bishops_min) == 0
                && x.pawns == 0
                && no_exotic_pieces(x)
                && x.non_royal() == 1
                && (y.bishops_maj + y.bishops_min) == 1
                && y.queens == 0
                && y.rooks == 0
                && y.knights == 0
                && y.pawns == 0
                && no_exotic_pieces(y)
                && y.non_royal() == 1
        };
        if n_vs_b(a, b) || n_vs_b(b, a) {
            return true;
        }

        // N vs N (either direction)
        let n_vs_n = |x: &Mat, y: &Mat| {
            x.knights == 1
                && x.queens == 0
                && x.rooks == 0
                && (x.bishops_maj + x.bishops_min) == 0
                && x.pawns == 0
                && no_exotic_pieces(x)
                && x.non_royal() == 1
                && y.knights == 1
                && y.queens == 0
                && y.rooks == 0
                && (y.bishops_maj + y.bishops_min) == 0
                && y.pawns == 0
                && no_exotic_pieces(y)
                && y.non_royal() == 1
        };
        if n_vs_n(a, b) || n_vs_n(b, a) {
            return true;
        }
    }

    false
}

/// Game-handler variant of `compute`.  Uses raw pawn counts (no pawn→promotion
/// substitution) and excludes cross-board combinations where helpmate is possible
/// despite both sides being individually insufficient.
#[inline(always)]
fn compute_game_handler(game: &crate::game::GameState) -> bool {
    let bordered = crate::moves::get_world_size() <= 200;
    let (w, b) = count_both(&game.board, &game.game_rules);

    let w_nr = w.non_royal();
    let b_nr = b.non_royal();

    // Special: only royals on both sides
    if w_nr == 0 && b_nr == 0 {
        if w.kings > 0 && b.kings > 0 && w.royal_centaurs == 0 && b.royal_centaurs == 0 {
            return true;
        }
        if w.royal_centaurs > 0 && b.royal_centaurs > 0 && w.kings == 0 && b.kings == 0 {
            return true;
        }
    }

    // Special: royal centaur vs amazon (unbounded)
    if !bordered {
        if b.royal_centaurs == 1
            && w.amazons == 1
            && w_nr == 1
            && b_nr == 0
            && w.kings == 0
            && w.royal_centaurs == 0
            && b.kings == 0
        {
            return true;
        }
        if w.royal_centaurs == 1
            && b.amazons == 1
            && b_nr == 1
            && w_nr == 0
            && b.kings == 0
            && b.royal_centaurs == 0
            && w.kings == 0
        {
            return true;
        }
    }

    // Cross-color: K+R vs K+R (unbounded)
    if !bordered
        && w.kings >= 1
        && b.kings >= 1
        && w.rooks == 1
        && b.rooks == 1
        && w_nr == 1
        && b_nr == 1
    {
        return true;
    }

    // Special: 2K + R vs K
    if w.kings >= 2 && w.rooks == 1 && w_nr == 1 && b.kings >= 1 && b_nr == 0 {
        return false;
    }
    if b.kings >= 2 && b.rooks == 1 && b_nr == 1 && w.kings >= 1 && w_nr == 0 {
        return false;
    }

    let w_insuff = if bordered {
        is_insufficient_bordered(&w)
    } else {
        is_insufficient(&w)
    };
    if !w_insuff {
        return false;
    }
    let b_insuff = if bordered {
        is_insufficient_bordered(&b)
    } else {
        is_insufficient(&b)
    };
    if !b_insuff {
        return false;
    }

    // Both sides are individually insufficient. Check for helpmate-only combos
    // that game handlers must not auto-declare as draws.
    if is_helpmate_only_combo(&w, &b, bordered) {
        return false;
    }

    true
}

/// Returns true if the position is a draw by insufficient material for game
/// handler purposes.  Unlike `evaluate_insufficient_material`, this function:
///   - does NOT substitute pawns with their best promotion piece, and
///   - does NOT classify helpmate-only endgames (R+B vs Q, R+N vs R,
///     R+B vs unpromotable/non-queen-promotable P, R+N vs P) as draws.
#[inline]
pub fn evaluate_insufficient_material_game_handler(game: &crate::game::GameState) -> bool {
    if (game.white_piece_count + game.black_piece_count) >= 6 {
        return false;
    }

    if game.game_rules.white_win_condition != crate::game::WinCondition::Checkmate
        || game.game_rules.black_win_condition != crate::game::WinCondition::Checkmate
    {
        return false;
    }
    
    compute_game_handler(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece, PieceType, PlayerColor};
    use crate::game::{GameRules, GameState, PromotionRanks};

    fn create_test_game_with_pieces(pieces: &[(i64, i64, PieceType, PlayerColor)]) -> GameState {
        let mut game = GameState::new();
        game.board = Board::new();

        for (x, y, pt, color) in pieces {
            game.board.set_piece(*x, *y, Piece::new(*pt, *color));
        }

        game.recompute_piece_counts();
        game.recompute_hash();
        game.recompute_correction_hashes();
        game
    }

    // ======================== Insufficient Material (dead draw) ========================

    #[test]
    fn test_king_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(evaluate_insufficient_material(&game), "K vs K");
    }

    #[test]
    fn test_king_queen_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Queen, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Q vs K insufficient on infinite board"
        );
    }

    #[test]
    fn test_king_rook_vs_king() {
        // K+R cannot deliver checkmate on unbounded board
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+R vs K insufficient on unbounded board"
        );
    }

    #[test]
    fn test_king_2rooks_vs_king_sufficient() {
        // K+2R is sufficient
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+2R vs K is sufficient"
        );
    }

    #[test]
    fn test_king_bishop_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+B vs K insufficient"
        );
    }

    #[test]
    fn test_king_knight_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+N vs K insufficient"
        );
    }

    #[test]
    fn test_king_2knights_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+2N vs K insufficient"
        );
    }

    #[test]
    fn test_king_3knights_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (3, 1, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+3N vs K insufficient"
        );
    }

    #[test]
    fn test_king_chancellor_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Chancellor, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Chancellor vs K insufficient"
        );
    }

    #[test]
    fn test_king_guard_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Guard, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Guard vs K insufficient"
        );
    }

    #[test]
    fn test_king_rook_knight_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+R+N vs K insufficient"
        );
    }

    #[test]
    fn test_king_bishop_knight_vs_king() {
        // K+B+N vs K is insufficient on infinite board
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(compute(&game), "K+B+N vs K insufficient");
    }

    // ======================== Sufficient Material ========================

    #[test]
    fn test_king_amazon_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Amazon, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+Amazon vs K should be sufficient"
        );
    }

    #[test]
    fn test_king_2queens_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (4, 4, PieceType::Queen, PlayerColor::White),
            (5, 5, PieceType::Queen, PlayerColor::White),
            (10, 10, PieceType::King, PlayerColor::Black),
        ]);
        assert!(!compute(&game), "K+Q+Q vs K sufficient");
    }

    // ======================== Both sides insufficient ========================

    #[test]
    fn test_kb_vs_kb_draw() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 6, PieceType::Bishop, PlayerColor::Black),
        ]);
        assert!(evaluate_insufficient_material(&game), "K+B vs K+B draw");
    }

    #[test]
    fn test_kn_vs_kn_draw() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 7, PieceType::Knight, PlayerColor::Black),
        ]);
        assert!(evaluate_insufficient_material(&game), "K+N vs K+N draw");
    }

    #[test]
    fn test_kr_vs_kr_draw() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 5, PieceType::Rook, PlayerColor::Black),
        ]);
        assert!(evaluate_insufficient_material(&game), "K+R vs K+R draw");
    }

    // ======================== Fast exit / misc ========================

    #[test]
    fn test_complex_position_fast_exit() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Queen, PlayerColor::White),
            (2, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 5, PieceType::Rook, PlayerColor::Black),
            (7, 5, PieceType::Bishop, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "6+ pieces fast exit"
        );
    }

    #[test]
    fn test_can_pawn_promote_basic() {
        let rules = GameRules {
            promotion_ranks: PromotionRanks {
                white: vec![8],
                black: vec![1],
            },
            promotion_types: None,
            promotions_allowed: None,
            move_rule_limit: None,
            white_win_condition: crate::game::WinCondition::Checkmate,
            black_win_condition: crate::game::WinCondition::Checkmate,
        };

        assert!(can_pawn_promote(5, PlayerColor::White, &rules));
        assert!(!can_pawn_promote(10, PlayerColor::White, &rules));
        assert!(can_pawn_promote(3, PlayerColor::Black, &rules));
        assert!(!can_pawn_promote(-5, PlayerColor::Black, &rules));
    }

    #[test]
    fn test_can_pawn_promote_no_ranks() {
        let mut rules = GameRules::default();
        rules.promotion_ranks = PromotionRanks {
            white: vec![],
            black: vec![],
        };
        assert!(!can_pawn_promote(5, PlayerColor::White, &rules));
    }

    #[test]
    fn test_pawn_past_promotion_insufficient() {
        let mut game = Box::new(create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (0, 10, PieceType::Pawn, PlayerColor::White), // Past rank 8
            (5, 5, PieceType::King, PlayerColor::Black),
        ]));
        game.game_rules.promotion_ranks = PromotionRanks {
            white: vec![8],
            black: vec![1],
        };
        assert!(compute(&game), "K + dead Pawn vs K should be insufficient");
    }

    #[test]
    fn test_king_chancellor_knight_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Chancellor, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+Chancellor+N vs K is sufficient"
        );
    }

    #[test]
    fn test_2chancellors_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Chancellor, PlayerColor::White),
            (2, 0, PieceType::Chancellor, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "2 Chancellors vs K is sufficient"
        );
    }

    #[test]
    fn test_3archbishops_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Archbishop, PlayerColor::White),
            (2, 0, PieceType::Archbishop, PlayerColor::White),
            (3, 0, PieceType::Archbishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "3 Archbishops vs K is sufficient"
        );
    }

    #[test]
    fn test_queen_2bishops_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Queen, PlayerColor::White),
            (2, 0, PieceType::Bishop, PlayerColor::White),
            (3, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "Q+2B vs K is sufficient"
        );
    }

    #[test]
    fn test_rook_2opposite_bishops_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Bishop, PlayerColor::White),
            (3, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+R+2 opposite bishops vs K is sufficient"
        );
    }

    #[test]
    fn test_rook_bishop_knight_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Bishop, PlayerColor::White),
            (3, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+R+B+N vs K is sufficient"
        );
    }

    #[test]
    fn test_rook_2knights_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (3, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+R+2N vs K is sufficient"
        );
    }

    #[test]
    fn test_2kings_rook_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::King, PlayerColor::White),
            (2, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "2K+R vs K is sufficient"
        );
    }

    #[test]
    fn test_2hawks_bishop_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (2, 0, PieceType::Hawk, PlayerColor::White),
            (3, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+2 Hawks+B vs K is sufficient"
        );
    }

    #[test]
    fn test_3hawks_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (2, 0, PieceType::Hawk, PlayerColor::White),
            (3, 0, PieceType::Hawk, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+3 Hawks vs K is sufficient"
        );
    }

    #[test]
    fn test_3knightriders_vs_king_sufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Knightrider, PlayerColor::White),
            (2, 0, PieceType::Knightrider, PlayerColor::White),
            (3, 0, PieceType::Knightrider, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+3 Knightriders vs K is sufficient"
        );
    }
}
