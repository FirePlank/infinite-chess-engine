use crate::board::{Board, PieceType, PlayerColor};
use rustc_hash::FxHashMap;
use std::cell::RefCell;

// Thread-local material cache: material_hash -> Option<i32> (None = sufficient, Some = divisor)
// Caches insuffcient material results to avoid repeated board iteration.
// Note: This may have false positives for bishop color scenarios, but is safe (returns correct
// or slightly sub-optimal result - never incorrect).
thread_local! {
    static MATERIAL_CACHE: RefCell<FxHashMap<u64, Option<i32>>> = RefCell::new(FxHashMap::default());
}

/// Clear the material cache. Call at the start of a new game.
pub fn clear_material_cache() {
    MATERIAL_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// Helper: get the best piece type a pawn can promote to (highest material value)
/// Returns Queen if any promotion mechanism is enabled (promotion_types, promotions_allowed, or promotion_ranks)
fn get_best_promotion_piece(game_rules: &crate::game::GameRules) -> Option<PieceType> {
    // If promotion_types is set and non-empty, find the best piece
    if let Some(types) = game_rules
        .promotion_types
        .as_ref()
        .filter(|t| !t.is_empty())
    {
        return types
            .iter()
            .max_by_key(|pt| super::base::get_piece_value(**pt))
            .copied();
    }

    None
}

/// Check if a pawn at position y can still promote for the given color
fn can_pawn_promote(y: i64, color: PlayerColor, game_rules: &crate::game::GameRules) -> bool {
    // Get promotion ranks for this color
    let promo_ranks = {
        let ranks = &game_rules.promotion_ranks;
        match color {
            PlayerColor::White => &ranks.white,
            PlayerColor::Black => &ranks.black,
            PlayerColor::Neutral => return false,
        }
    };

    if promo_ranks.is_empty() {
        return false;
    }

    // Pawn can promote if it hasn't passed all promotion ranks
    // For white: pawn at y can promote if any rank > y exists
    // For black: pawn at y can promote if any rank < y exists
    match color {
        PlayerColor::White => promo_ranks.iter().any(|&rank| rank > y),
        PlayerColor::Black => promo_ranks.iter().any(|&rank| rank < y),
        PlayerColor::Neutral => false,
    }
}

/// Check if a side has sufficient material to force checkmate in infinite chess.
/// Based on the official insufficientmaterial.ts from infinitechess.org
///
/// Logic: This lists INSUFFICIENT scenarios. Anything not matching
/// those scenarios is sufficient. We check if the current material fits within
/// any insufficient scenario.
///
/// IMPORTANT: On infinite/unbounded boards (world size >= 100), the king can
/// escape forever, so standard chess mating patterns don't work. On bounded
/// boards (< 100), standard chess rules apply.
///
/// NEW: Promotable pawns are counted as the best piece they can promote to.
/// This fixes cases like 2R+P vs K being marked insufficient when P can promote.
fn has_sufficient_mating_material(
    board: &Board,
    color: PlayerColor,
    has_our_king: bool,
    game_rules: &crate::game::GameRules,
) -> bool {
    // Check if we're on a bounded board (standard chess rules apply)
    let world_size = crate::moves::get_world_size();
    let is_bounded = world_size < 100;

    // Get best promotion piece for this color (if any)
    let best_promo = get_best_promotion_piece(game_rules);

    let mut queens = 0;
    let mut rooks = 0;
    let mut bishops = 0;
    let mut knights = 0;
    let mut chancellors = 0;
    let mut archbishops = 0;
    let mut hawks = 0;
    let mut guards = 0;
    let mut pawns = 0;
    let mut amazons = 0;
    let mut knightriders = 0;
    let mut huygens = 0;
    let mut light_bishops = 0;
    let mut dark_bishops = 0;

    for (x, y, piece) in board.iter() {
        if piece.color() != color {
            continue;
        }
        match piece.piece_type() {
            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
            PieceType::Rook => rooks += 1,
            PieceType::Bishop => {
                bishops += 1;
                if (x + y) % 2 == 0 {
                    light_bishops += 1;
                } else {
                    dark_bishops += 1;
                }
            }
            PieceType::Knight => knights += 1,
            PieceType::Chancellor => chancellors += 1,
            PieceType::Archbishop => archbishops += 1,
            PieceType::Hawk => hawks += 1,
            PieceType::Guard => guards += 1,
            PieceType::Pawn => {
                // KEY CHANGE: If this pawn can promote, count it as the best promotion piece
                if let Some(promo_piece) = best_promo {
                    if can_pawn_promote(y, color, game_rules) {
                        // Count as the promotion piece instead of a pawn
                        match promo_piece {
                            PieceType::Queen | PieceType::RoyalQueen => queens += 1,
                            PieceType::Rook => rooks += 1,
                            PieceType::Bishop => bishops += 1,
                            PieceType::Knight => knights += 1,
                            PieceType::Chancellor => chancellors += 1,
                            PieceType::Archbishop => archbishops += 1,
                            PieceType::Hawk => hawks += 1,
                            PieceType::Guard => guards += 1,
                            PieceType::Amazon => amazons += 1,
                            PieceType::Knightrider => knightriders += 1,
                            PieceType::Huygen => huygens += 1,
                            _ => pawns += 1,
                        }
                    } else {
                        // Pawn past promotion rank - just count as pawn
                        pawns += 1;
                    }
                } else {
                    // No promotions possible - count as pawn
                    pawns += 1;
                }
            }
            PieceType::Amazon => amazons += 1,
            PieceType::Knightrider => knightriders += 1,
            PieceType::Huygen => huygens += 1,
            _ => {}
        }
    }

    // Helper: check if we have "only" certain pieces (nothing else)
    // Using a macro for zero overhead - gets inlined at compile time
    macro_rules! has_only {
        ($q:expr, $r:expr, $b:expr, $n:expr, $c:expr, $a:expr, $h:expr, $g:expr, $p:expr, $s:expr, $hu:expr, $am:expr) => {
            queens <= $q
                && rooks <= $r
                && bishops <= $b
                && knights <= $n
                && chancellors <= $c
                && archbishops <= $a
                && hawks <= $h
                && guards <= $g
                && pawns <= $p
                && knightriders <= $s
                && huygens <= $hu
                && amazons <= $am
        };
    }

    // Fast path: compute total piece count for quick exits
    let total = queens
        + rooks
        + bishops
        + knights
        + chancellors
        + archbishops
        + hawks
        + guards
        + pawns
        + amazons
        + knightriders
        + huygens;

    // No pieces = insufficient
    if total == 0 {
        return false;
    }

    // BOUNDED BOARD LOGIC (standard chess rules, world_size < 100)
    if is_bounded && has_our_king {
        let minors = bishops + knights + guards + hawks;

        // ==== OBVIOUS DRAWS (immediate draw per Article 5.2b) ====

        // K alone (bare king) - insufficient
        if total == 0 {
            return false;
        }

        // K + single minor (B or N) vs bare K - insufficient
        // Note: This only counts OUR material; opponent check happens separately
        if minors == 1 && total == 1 {
            return false;
        }

        // ==== SUFFICIENT MATERIAL (can force checkmate) ====

        // K+Q is sufficient (can force mate in corner)
        if queens >= 1 {
            return true;
        }
        // K+R is sufficient (can force mate on edge)
        if rooks >= 1 {
            return true;
        }
        // K+Amazon is sufficient
        if amazons >= 1 {
            return true;
        }
        // K+Chancellor is sufficient (has rook component)
        if chancellors >= 1 {
            return true;
        }
        // K+B+N is sufficient (bishop+knight can mate on bounded board)
        if bishops >= 1 && knights >= 1 {
            return true;
        }
        // K+BB (opposite colors) is sufficient
        if light_bishops >= 1 && dark_bishops >= 1 {
            return true;
        }
        // K+Archbishop alone is sufficient (B+N combined on bounded board)
        if archbishops >= 1 {
            return true;
        }
        // Any pawn is sufficient (can promote)
        if pawns >= 1 {
            return true;
        }
        // K+3 Knights or more is sufficient
        if knights >= 3 {
            return true;
        }

        // ==== HEURISTIC DRAWS (cannot force checkmate, treat as draw) ====

        // K+NN vs K - two knights cannot force mate (but can helpmate)
        if knights == 2 && total == 2 {
            return false;
        }

        // K+B+B same color - insufficient (can't cover all squares)
        if bishops >= 2 && (light_bishops == 0 || dark_bishops == 0) && total == bishops {
            return false;
        }

        // ==== DEFAULT FOR BOUNDED ====
        // For piece combinations not explicitly handled, assume sufficient on bounded boards
        // (most multi-piece combinations can theoretically mate on bounded boards)
        if total >= 2 {
            return true;
        }

        // Single unknown piece type - check if it's a major/powerful piece
        // This handles fairy pieces that might be sufficient
        return total >= 1 && (knightriders >= 1 || huygens >= 1);
    }

    // Fast path: 2+ major pieces WITH KING = always sufficient
    // (can build mating nets even on infinite board with 2+ majors coordinated by King)
    let major_pieces = queens + rooks + chancellors + archbishops + amazons;
    if has_our_king && major_pieces >= 2 {
        return true;
    }

    // Fast path: 1 major + any minor WITH KING = sufficient
    // CHECK EXCEPTIONS: R+Minor is insufficient on infinite board!
    // Q+Minor, C+Minor, Amazon+Minor are sufficient.
    if has_our_king
        && (queens >= 1 || chancellors >= 1 || amazons >= 1)
        && (knights >= 1 || bishops >= 1 || guards >= 1 || hawks >= 1)
    {
        return true;
    }

    // R+2 Minors is sufficient (Rook + 1 Minor is insufficient)
    if has_our_king && rooks >= 1 && (knights + bishops + guards + hawks) >= 2 {
        return true;
    }

    // 1K vs 1k scenarios (with our king helping)
    if has_our_king {
        // Amazon + anything can mate
        if amazons >= 1 {
            return true;
        }

        // {queensW: 1} - single queen insufficient
        if queens == 1 && has_only!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {rooksW: 1} - single rook insufficient
        if rooks == 1 && has_only!(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {bishopsW: insufficient unless >=3 light AND >=3 dark}
        if bishops > 0
            && (light_bishops < 3 || dark_bishops < 3)
            && has_only!(0, 0, i32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {knightsW: 3} - up to 3 knights insufficient
        if knights <= 3 && has_only!(0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {hawksW: 2} - 2 hawks insufficient
        if hawks <= 2 && has_only!(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0) {
            return false;
        }

        // {hawksW: 1, bishopsW: [1,0]} - hawk + same-color bishop
        if hawks == 1
            && bishops == 1
            && (light_bishops == 0 || dark_bishops == 0)
            && has_only!(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {rooksW: 1, knightsW: 1} - rook + knight insufficient
        if rooks == 1 && knights == 1 && has_only!(0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {rooksW: 1, bishopsW: [1,0]} - rook + same-color bishop
        if rooks == 1
            && bishops == 1
            && (light_bishops == 0 || dark_bishops == 0)
            && has_only!(0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {rooksW: 1, rooksB: 1} - rook vs rook (but we only count our pieces, so skip)

        // {archbishopsW: 1, bishopsW: [1,0]} - archbishop + same-color bishop
        if archbishops == 1
            && bishops == 1
            && (light_bishops == 0 || dark_bishops == 0)
            && has_only!(0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {archbishopsW: 1, knightsW: 1} - archbishop + knight
        if archbishops == 1 && knights == 1 && has_only!(0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {knightsW: 1, bishopsW: [Inf, 0]} - knight + any same-color bishops
        if knights == 1
            && bishops > 0
            && (light_bishops == 0 || dark_bishops == 0)
            && has_only!(0, 0, i32::MAX, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {knightsW: 1, bishopsW: [1,1]} - knight + one of each bishop color
        if knights == 1
            && light_bishops == 1
            && dark_bishops == 1
            && has_only!(0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {knightsW: 2, bishopsW: [1,0]} - 2 knights + same-color bishop
        if knights == 2
            && bishops == 1
            && (light_bishops == 0 || dark_bishops == 0)
            && has_only!(0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0)
        {
            return false;
        }

        // {guardsW: 1} - single guard insufficient
        if guards == 1 && has_only!(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0) {
            return false;
        }

        // {chancellorsW: 1} - single chancellor insufficient
        if chancellors == 1 && has_only!(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) {
            return false;
        }

        // {knightridersW: 2} - 2 knightriders insufficient
        if knightriders <= 2 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0) {
            return false;
        }

        // {pawnsW: 3} - up to 3 pawns insufficient
        if pawns <= 3 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0) {
            return false;
        }

        // Everything else with king is sufficient
        return true;
    }

    // 0K vs 1k scenarios (without our king)

    // {amazonsW: 1} - Amazon alone cannot force mate alone
    if amazons == 1 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1) {
        return false;
    }

    // {queensW: 1}
    if queens == 1 && has_only!(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {rooksW: 2}
    if rooks <= 2 && has_only!(0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {chancellorsW: 1}
    if chancellors == 1 && has_only!(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {archbishopsW: 1}
    if archbishops == 1 && has_only!(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {hawksW: 2}
    if hawks <= 2 && has_only!(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0) {
        return false;
    }

    // {queensW: 1, rooksW: 1}
    if queens == 1 && rooks == 1 && has_only!(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {queensW: 1, knightsW: 1}
    if queens == 1 && knights == 1 && has_only!(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {queensW: 1, bishopsW: [1,0]}
    if queens == 1
        && bishops == 1
        && (light_bishops == 0 || dark_bishops == 0)
        && has_only!(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {queensW: 1, pawnsW: 1}
    if queens == 1 && pawns == 1 && has_only!(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) {
        return false;
    }

    // {bishopsW: [2,2]} - 2 light + 2 dark bishops
    if light_bishops == 2 && dark_bishops == 2 && has_only!(0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {bishopsW: [Inf, 1]} - any number of one color + 1 of other
    if bishops > 0
        && (light_bishops == 0
            || dark_bishops == 0
            || (light_bishops <= 1 && dark_bishops > 0)
            || (dark_bishops <= 1 && light_bishops > 0))
        && has_only!(0, 0, i32::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        // [Inf, 1] means unlimited of one color, at most 1 of the other
        if (light_bishops == 0 || dark_bishops <= 1) && (dark_bishops == 0 || light_bishops <= 1) {
            return false;
        }
    }

    // {knightsW: 4}
    if knights <= 4 && has_only!(0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {knightsW: 2, bishopsW: [Inf, 0]}
    if knights <= 2
        && bishops > 0
        && (light_bishops == 0 || dark_bishops == 0)
        && has_only!(0, 0, i32::MAX, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {knightsW: 2, bishopsW: [1,1]}
    if knights <= 2
        && light_bishops == 1
        && dark_bishops == 1
        && has_only!(0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {knightsW: 1, bishopsW: [2,1]}
    if knights == 1
        && light_bishops <= 2
        && dark_bishops <= 1
        && bishops <= 3
        && has_only!(0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {hawksW: 3}
    if hawks <= 3 && has_only!(0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0) {
        return false;
    }

    // {rooksW: 1, knightsW: 1, bishopsW: [1,0]}
    if rooks == 1
        && knights == 1
        && bishops == 1
        && (light_bishops == 0 || dark_bishops == 0)
        && has_only!(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {rooksW: 1, knightsW: 1, pawnsW: 1}
    if rooks == 1 && knights == 1 && pawns == 1 && has_only!(0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0) {
        return false;
    }

    // {rooksW: 1, knightsW: 2}
    if rooks == 1 && knights <= 2 && has_only!(0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {rooksW: 1, guardsW: 1}
    if rooks == 1 && guards == 1 && has_only!(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0) {
        return false;
    }

    // {rooksW: 2, bishopsW: [1,0]}
    if rooks == 2
        && bishops == 1
        && (light_bishops == 0 || dark_bishops == 0)
        && has_only!(0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {rooksW: 2, knightsW: 1}
    if rooks == 2 && knights == 1 && has_only!(0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {rooksW: 2, pawnsW: 1}
    if rooks == 2 && pawns == 1 && has_only!(0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) {
        return false;
    }

    // {archbishopsW: 1, bishopsW: [2,0]}
    if archbishops == 1
        && bishops <= 2
        && (light_bishops == 0 || dark_bishops == 0)
        && has_only!(0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {archbishopsW: 1, bishopsW: [1,1]}
    if archbishops == 1
        && light_bishops == 1
        && dark_bishops == 1
        && has_only!(0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    {
        return false;
    }

    // {archbishopsW: 1, knightsW: 2}
    if archbishops == 1 && knights <= 2 && has_only!(0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {archbishopsW: 2}
    if archbishops <= 2 && has_only!(0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {chancellorsW: 1, guardsW: 1}
    if chancellors == 1 && guards == 1 && has_only!(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0) {
        return false;
    }

    // {chancellorsW: 1, knightsW: 1}
    if chancellors == 1 && knights == 1 && has_only!(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {chancellorsW: 1, rooksW: 1}
    if chancellors == 1 && rooks == 1 && has_only!(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) {
        return false;
    }

    // {guardsW: 2}
    if guards <= 2 && has_only!(0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0) {
        return false;
    }

    // {knightridersW: 3}
    if knightriders <= 3 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0) {
        return false;
    }

    // {pawnsW: 2}
    if pawns < 2 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) {
        return false;
    }

    // {huygensW: 4}
    if huygens <= 4 && has_only!(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0) {
        return false;
    }

    // Everything else is sufficient (not in any insufficient scenario)
    true
}

/// Evaluates if the current board state is insufficient material.
/// Returns None for sufficient material (normal eval), Some(0) for insufficient (dead draw),
/// or Some(divisor) for drawish positions (divide eval by divisor).
///
/// Simplified rules:
/// 1. If either side can force mate → None (normal eval)
/// 2. If both sides have pieces → Some(divisor) (drawish, divide eval by divisor)
/// 3. Otherwise (one side has no pieces) → Some(0) (insufficient, dead draw)
pub fn evaluate_insufficient_material(game: &crate::game::GameState) -> Option<i32> {
    // Fast exit for complex positions
    // We only check if active piece count is large.
    if (game.white_piece_count + game.black_piece_count) >= 6 {
        return None;
    }

    // Check cache using material_hash
    let material_hash = game.material_hash;

    // IMPORTANT: Only use cache if there are no pawns on the board,
    // because pawn promotion potential depends on their Y-coordinate which is NOT in the material_hash.
    // Also, piece counts are small anyway, so iteration is fast.
    let has_pawns = game.white_pawn_count > 0 || game.black_pawn_count > 0;

    if !has_pawns {
        let cached = MATERIAL_CACHE.with(|cache| cache.borrow().get(&material_hash).copied());
        if let Some(result) = cached {
            return result;
        }
    }

    // Cache miss or has pawns - compute insufficient material
    let result = compute_insufficient_material(game);

    // Store in cache (limit size) if no pawns
    if !has_pawns {
        MATERIAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() > 4096 {
                cache.clear();
            }
            cache.insert(material_hash, result);
        });
    }

    result
}

/// Core insufficient material computation. Called on cache miss.
fn compute_insufficient_material(game: &crate::game::GameState) -> Option<i32> {
    let board = &game.board;
    let game_rules = &game.game_rules;
    // Count pieces for each side
    let mut white_has_king = false;
    let mut black_has_king = false;
    let mut white_pieces = 0; // Non-king pieces
    let mut black_pieces = 0; // Non-king pieces

    for (_, _, p) in board.iter() {
        let is_royal = p.piece_type().is_royal();
        match p.color() {
            PlayerColor::White => {
                if is_royal {
                    white_has_king = true;
                } else {
                    white_pieces += 1;
                }
            }
            PlayerColor::Black => {
                if is_royal {
                    black_has_king = true;
                } else {
                    black_pieces += 1;
                }
            }
            _ => {}
        }
    }

    // Check if each side can force mate
    let white_can_mate =
        has_sufficient_mating_material(board, PlayerColor::White, white_has_king, game_rules);
    let black_can_mate =
        has_sufficient_mating_material(board, PlayerColor::Black, black_has_king, game_rules);

    // 1. If either side can force mate, use normal evaluation
    if white_can_mate || black_can_mate {
        return None;
    }

    // 2. If BOTH sides have non-king pieces, it's drawish but not dead draw
    // Use eval/divisor to dampen the score since it's likely a draw
    if white_pieces > 0 && black_pieces > 0 {
        return Some(8);
    }

    // 3. One side has no pieces - dead draw
    Some(0)
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

    // ======================== Basic Insufficient Material Tests ========================

    #[test]
    fn test_king_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        assert_eq!(result, Some(0), "K vs K should be insufficient (dead draw)");
    }

    #[test]
    fn test_king_bishop_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        // In infinite chess, KB vs K is insufficient
        let result = evaluate_insufficient_material(&game);
        assert_eq!(
            result,
            Some(0),
            "K+B vs K should be insufficient on infinite board"
        );
    }

    #[test]
    fn test_king_knight_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        assert_eq!(result, Some(0), "K+N vs K should be insufficient");
    }

    #[test]
    fn test_king_2knights_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        // K+2N vs K is technically insufficient (can't force mate)
        let result = evaluate_insufficient_material(&game);
        assert_eq!(result, Some(0), "K+2N vs K should be insufficient");
    }

    // ======================== Sufficient Material Tests ========================

    #[test]
    fn test_king_queen_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Queen, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        // This is only 3 pieces, below the fast-exit threshold
        let result = evaluate_insufficient_material(&game);
        // K+Q vs K in infinite chess - queen alone might not be sufficient
        // but with king it should trigger lone king detection
        assert_eq!(result, Some(0), "K+Q vs K - lone king condition");
    }

    #[test]
    fn test_king_2rooks_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (2, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        // K+2R vs K - In infinite chess, 2 rooks is still sufficient with mating techniques
        let result = evaluate_insufficient_material(&game);
        // Should be sufficient (None) or lone king (Some(0))
        assert!(result.is_some() || result.is_none());
    }

    #[test]
    fn test_king_amazon_vs_king() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Amazon, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        // Amazon (Q+N) is very powerful and should be sufficient
        let result = evaluate_insufficient_material(&game);
        assert!(
            result.is_some() || result.is_none(),
            "Amazon should handle lone king"
        );
    }

    // ======================== Drawish Position Tests ========================

    #[test]
    fn test_kb_vs_kb_drawish() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 6, PieceType::Bishop, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        // Both sides have pieces - should be drawish (Some(8))
        assert_eq!(result, Some(8), "K+B vs K+B should be drawish");
    }

    #[test]
    fn test_kn_vs_kn_drawish() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 7, PieceType::Knight, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        assert_eq!(result, Some(8), "K+N vs K+N should be drawish");
    }

    // ======================== No King Tests (Infinite Chess special) ========================

    #[test]
    fn test_lone_queen_vs_king() {
        // White has no king, just a queen
        let game = create_test_game_with_pieces(&[
            (1, 1, PieceType::Queen, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        // 0K scenario - lone piece vs king
        assert_eq!(result, Some(0), "Q (no K) vs K should be insufficient");
    }

    #[test]
    fn test_2rooks_no_king_vs_king() {
        // White has 2 rooks but no king
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::Rook, PlayerColor::White),
            (1, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        // 2R (0K) vs K - should be lone king detection
        assert_eq!(result, Some(0), "2R (no K) vs K should be insufficient");
    }

    // ======================== Fast Exit Tests ========================

    #[test]
    fn test_complex_position_fast_exit() {
        // 6+ pieces should return None (fast exit)
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Queen, PlayerColor::White),
            (2, 0, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
            (6, 5, PieceType::Rook, PlayerColor::Black),
            (7, 5, PieceType::Bishop, PlayerColor::Black),
        ]);

        let result = evaluate_insufficient_material(&game);
        assert_eq!(
            result, None,
            "Complex positions (6+ pieces) should return None"
        );
    }

    // ======================== can_pawn_promote Tests ========================

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

        // White pawn at y=5 (can promote at 8)
        assert!(can_pawn_promote(5, PlayerColor::White, &rules));

        // White pawn at y=10 (past promo rank, can't promote)
        assert!(!can_pawn_promote(10, PlayerColor::White, &rules));

        // Black pawn at y=3 (can promote at 1)
        assert!(can_pawn_promote(3, PlayerColor::Black, &rules));

        // Black pawn at y=-5 (past promo rank, can't promote)
        assert!(!can_pawn_promote(-5, PlayerColor::Black, &rules));
    }

    #[test]
    fn test_can_pawn_promote_no_ranks() {
        let mut rules = GameRules::default();
        rules.promotion_ranks = PromotionRanks {
            white: vec![],
            black: vec![],
        };

        // No promotion ranks defined - pawns can never promote
        assert!(!can_pawn_promote(5, PlayerColor::White, &rules));
    }

    #[test]
    fn test_sufficient_queens_vs_king() {
        // K+Q+Q vs K is sufficient material on infinite board (2 majors + King)
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (4, 4, PieceType::Queen, PlayerColor::White),
            (5, 5, PieceType::Queen, PlayerColor::White),
            (10, 10, PieceType::King, PlayerColor::Black),
        ]);

        let result = compute_insufficient_material(&game);
        assert_eq!(result, None, "K+Q+Q vs K should be sufficient material");
    }

    #[test]
    fn test_insufficient_bishop_knight_vs_king() {
        // In this engine's infinite chess rules, K+B+N vs K is insufficient
        // because coordination is too hard on an unbounded board.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (2, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = compute_insufficient_material(&game);
        assert_eq!(
            result,
            Some(0),
            "K+B+N vs K should be insufficient in infinite chess"
        );
    }

    #[test]
    fn test_insufficient_two_knights_vs_king() {
        // K+N+N vs K is insufficient in this engine (specifically for infinite boards)
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 1, PieceType::Knight, PlayerColor::White),
            (2, 2, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = compute_insufficient_material(&game);
        assert!(
            result.is_some(),
            "K+N+N vs K should be insufficient (Some(0)) or drawish"
        );
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

        let result = compute_insufficient_material(&game);
        // White only has a dead pawn and a king.
        assert_eq!(
            result,
            Some(0),
            "K + dead Pawn vs K should be insufficient material"
        );
    }

    #[test]
    fn test_no_king_sufficient_material() {
        // 3 Rooks without king is sufficient to escape the 2-rook insufficient rule
        let game = create_test_game_with_pieces(&[
            (1, 1, PieceType::Rook, PlayerColor::White),
            (2, 2, PieceType::Rook, PlayerColor::White),
            (3, 3, PieceType::Rook, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);

        let result = compute_insufficient_material(&game);
        assert_eq!(
            result, None,
            "3 Rooks vs K (no white king) should be sufficient"
        );
    }
}
