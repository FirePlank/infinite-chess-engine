use crate::board::{Board, PieceType, PlayerColor};

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
fn has_sufficient_mating_material(board: &Board, color: PlayerColor, has_our_king: bool) -> bool {
    // Check if we're on a bounded board (standard chess rules apply)
    let world_size = crate::moves::get_world_size();
    let is_bounded = world_size < 100;

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

    for ((x, y), piece) in board.iter() {
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
            PieceType::Pawn => pawns += 1,
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

    // ==========================================================================
    // BOUNDED BOARD LOGIC (standard chess rules, world_size < 100)
    // On a bounded board, king can be cornered, so standard mating patterns work
    // ==========================================================================
    if is_bounded && has_our_king {
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
        // K+Archbishop + minor piece is sufficient
        if archbishops >= 1 && (bishops >= 1 || knights >= 1) {
            return true;
        }
        // K+2 Archbishops is sufficient
        if archbishops >= 2 {
            return true;
        }
        // K+BB (opposite colors) is sufficient
        if light_bishops >= 1 && dark_bishops >= 1 {
            return true;
        }
        // K+BN is sufficient
        if bishops >= 1 && knights >= 1 {
            return true;
        }
        // K+3 Knights is sufficient (but not 2)
        if knights >= 3 {
            return true;
        }
        // Any pawn is sufficient (can promote)
        if pawns >= 1 {
            return true;
        }
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

    // =====================================================================
    // 1K vs 1k scenarios (with our king helping)
    // These are INSUFFICIENT scenarios from insuffmatScenarios_1K1k
    // =====================================================================
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

    // =====================================================================
    // 0K vs 1k scenarios (without our king)
    // These are INSUFFICIENT scenarios from insuffmatScenarios_0K1k
    // Anything NOT in this list is sufficient
    // =====================================================================

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
pub fn evaluate_insufficient_material(board: &Board) -> Option<i32> {
    // Fast exit for complex positions
    if board.len() >= 6 {
        return None;
    }

    // Count pieces for each side
    let mut white_has_king = false;
    let mut black_has_king = false;
    let mut white_pieces = 0; // Non-king pieces
    let mut black_pieces = 0; // Non-king pieces

    for (_, p) in board.iter() {
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
    let white_can_mate = has_sufficient_mating_material(board, PlayerColor::White, white_has_king);
    let black_can_mate = has_sufficient_mating_material(board, PlayerColor::Black, black_has_king);

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
