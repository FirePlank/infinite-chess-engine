//! Evaluation dispatch keyed on the shape of the position (bounds, piece mix,
//! obstacle density) rather than the often-absent `[Variant]` tag. `detect` caches
//! into `GameState::eval_kind`, and can change mid-game as the board does.

use crate::board::{PieceType, PlayerColor};
use crate::game::{GameState, WinCondition};

/// Largest world dimension still treated as a "bounded" board for Obstocean.
/// The infinite default (±1e15) is far above this; real obstacle boards are
/// tens of squares across. Matches the bounded cutoff used elsewhere in eval.
const OBSTOCEAN_MAX_WORLD_SIZE: i64 = 200;

/// Fraction (percent) of the otherwise-empty squares that must be obstacles for
/// a bounded board to read as Obstocean. Obstacle boards start fully packed and
/// only slowly open up as pawns capture through the lanes, so this stays high.
const OBSTOCEAN_MIN_FILL_PCT: i128 = 60;

/// Most pawns a Pawn-Horde side may field (an 8-wide, 7-deep wall). Also keeps
/// the horde's pawn list within the evaluator's fixed-capacity buffer.
const PAWN_HORDE_MAX_PAWNS: i64 = 56;

/// Percent of White's pieces that must be pawns for the horde evaluator to
/// apply. The horde eval only understands a dominant pawn mass; once promotions
/// (or captures) leave White mostly officers, the base evaluator fits better.
const PAWN_HORDE_MIN_PAWN_PCT: i64 = 80;

/// Which evaluation function a position's characteristics select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvalKind {
    /// Standard 8×8 chess: board bounded to files/ranks 1..=8, orthodox pieces
    /// only, exactly one king per side, promotion on the far ranks.
    Chess,
    /// Pawn Horde: White is a kingless horde of only pawns (≤56), Black is an
    /// orthodox army with a king. White wins by checkmate, Black by capturing
    /// every pawn.
    PawnHorde,
    /// Obstocean: a not-too-large bounded board whose otherwise-empty squares
    /// are almost entirely obstacles, orthodox pieces only.
    Obstocean,
    /// Everything else, routed to NNUE when applicable and the base HCE otherwise.
    #[default]
    Generic,
}

/// Derive the evaluator from position characteristics. Called once per received
/// position; never on the search hot path. Reads the current world bounds.
pub fn detect(game: &GameState) -> EvalKind {
    detect_in_region(game, crate::moves::get_coord_bounds())
}

/// Core of [`detect`], taking the board region explicitly so it never reads the
/// world-bounds global and stays testable without mutating process-wide state.
fn detect_in_region(game: &GameState, region: (i64, i64, i64, i64)) -> EvalKind {
    let (min_x, max_x, min_y, max_y) = region;
    // Single board pass: obstacle count + per-side orthodox composition, plus a
    // flag for any fairy (non-orthodox, non-neutral) piece.
    let mut fairy_present = false;
    let mut obstacle_count: i64 = 0;
    // White's orthodox composition is all the Pawn-Horde test needs; Black is
    // only required to be an orthodox army with a king (fairy_present + royals).
    let mut w_pawns: i64 = 0;
    let mut w_officers: i64 = 0; // white Q/R/B/N

    for (_x, _y, piece) in game.board.iter_all_pieces() {
        let pt = piece.piece_type();
        match piece.color() {
            PlayerColor::Neutral => {
                if pt == PieceType::Obstacle {
                    obstacle_count += 1;
                }
            }
            color => match pt {
                PieceType::Pawn => {
                    if color == PlayerColor::White {
                        w_pawns += 1;
                    }
                }
                PieceType::Queen | PieceType::Rook | PieceType::Bishop | PieceType::Knight => {
                    if color == PlayerColor::White {
                        w_officers += 1;
                    }
                }
                PieceType::King => {} // orthodox royal; counted via royal lists
                // Anything else (Amazon, RoyalQueen, Centaur, …) is fairy.
                _ => fairy_present = true,
            },
        }
    }

    // All three specialized evaluators assume an orthodox army.
    if fairy_present {
        return EvalKind::Generic;
    }

    let w_royals = game.white_royals.len();
    let b_royals = game.black_royals.len();

    // Chess: an 8×8 board with a single king each and normal promotion.
    let is_8x8 = min_x == 1 && max_x == 8 && min_y == 1 && max_y == 8;
    if is_8x8
        && obstacle_count == 0
        && w_royals == 1
        && b_royals == 1
        && game.white_promo_rank == 8
        && game.black_promo_rank == 1
    {
        return EvalKind::Chess;
    }

    // Pawn Horde: White is a kingless, pawn-dominated army. The evaluator is
    // hard-wired to White-as-horde, and a mostly-promoted White no longer fits its
    // pawn-mass heuristics, so both cases fall back to the base evaluator.
    if w_royals == 0
        && (1..=PAWN_HORDE_MAX_PAWNS).contains(&w_pawns)
        && b_royals >= 1
        && obstacle_count == 0
        && game.game_rules.black_win_condition == WinCondition::AllPiecesCaptured
    {
        let white_total = w_pawns + w_officers;
        if w_pawns * 100 >= white_total * PAWN_HORDE_MIN_PAWN_PCT {
            return EvalKind::PawnHorde;
        }
    }

    // Obstocean: a bounded board packed with obstacles.
    let world_size = (max_x.saturating_sub(min_x)).max(max_y.saturating_sub(min_y));
    if obstacle_count > 0 && world_size <= OBSTOCEAN_MAX_WORLD_SIZE {
        let width = (max_x - min_x + 1) as i128;
        let height = (max_y - min_y + 1) as i128;
        let total = width * height;
        let real_pieces = (game.white_piece_count + game.black_piece_count) as i128;
        // Squares not holding a real piece: obstacles + empty (+ any voids).
        let non_piece = total - real_pieces;
        if non_piece > 0 && (obstacle_count as i128) * 100 >= non_piece * OBSTOCEAN_MIN_FILL_PCT {
            return EvalKind::Obstocean;
        }
    }

    EvalKind::Generic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Coordinate, Piece};

    // Boards are built by placing pieces directly and detection is driven with
    // an explicit region, so these tests never touch the world-bounds global and
    // are immune to the cross-test bounds races that plague ICN-based setup.
    struct Builder {
        game: GameState,
    }

    impl Builder {
        fn new() -> Self {
            let mut game = GameState::new();
            game.board = Board::new();
            // Standard promotion lines by default; overridden where needed.
            game.white_promo_rank = 8;
            game.black_promo_rank = 1;
            Builder { game }
        }

        fn put(mut self, x: i64, y: i64, pt: PieceType, color: PlayerColor) -> Self {
            self.game.board.set_piece(x, y, Piece::new(pt, color));
            self
        }

        fn win_conditions(mut self, white: WinCondition, black: WinCondition) -> Self {
            self.game.game_rules.white_win_condition = white;
            self.game.game_rules.black_win_condition = black;
            self
        }

        fn detect(mut self, region: (i64, i64, i64, i64)) -> EvalKind {
            self.game.recompute_piece_counts();
            detect_in_region(&self.game, region)
        }
    }

    /// Place an orthodox 8×8 army (both sides) onto a builder.
    fn standard_army(mut b: Builder) -> Builder {
        use PieceType::*;
        let back = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook];
        for (i, pt) in back.iter().enumerate() {
            let x = i as i64 + 1;
            b = b
                .put(x, 1, *pt, PlayerColor::White)
                .put(x, 8, *pt, PlayerColor::Black)
                .put(x, 2, Pawn, PlayerColor::White)
                .put(x, 7, Pawn, PlayerColor::Black);
        }
        b
    }

    #[test]
    fn standard_8x8_army_is_chess() {
        let kind = standard_army(Builder::new()).detect((1, 8, 1, 8));
        assert_eq!(kind, EvalKind::Chess);
    }

    #[test]
    fn same_army_unbounded_is_generic() {
        // Identical pieces on the infinite plane must NOT use the 8×8 evaluator.
        const INF: i64 = 1_000_000_000_000_000;
        let kind = standard_army(Builder::new()).detect((-INF, INF, -INF, INF));
        assert_eq!(kind, EvalKind::Generic);
    }

    #[test]
    fn fairy_piece_forces_generic() {
        // 8×8 bounds and kings, but an Amazon disqualifies every orthodox kind.
        let kind = Builder::new()
            .put(5, 1, PieceType::King, PlayerColor::White)
            .put(5, 8, PieceType::King, PlayerColor::Black)
            .put(4, 4, PieceType::Amazon, PlayerColor::White)
            .detect((1, 8, 1, 8));
        assert_eq!(kind, EvalKind::Generic);
    }

    #[test]
    fn kingless_pawn_wall_is_pawn_horde() {
        use PieceType::*;
        let mut b = Builder::new()
            .put(5, 8, King, PlayerColor::Black)
            .put(4, 8, Queen, PlayerColor::Black)
            .win_conditions(WinCondition::Checkmate, WinCondition::AllPiecesCaptured);
        // White: a kingless wall of pawns.
        for x in 1..=8 {
            b = b.put(x, 2, Pawn, PlayerColor::White);
        }
        const INF: i64 = 1_000_000_000_000_000;
        assert_eq!(b.detect((-INF, INF, -INF, INF)), EvalKind::PawnHorde);
    }

    #[test]
    fn lightly_promoted_horde_stays_pawn_horde() {
        // A handful of promotions still leaves White overwhelmingly pawns.
        use PieceType::*;
        const INF: i64 = 1_000_000_000_000_000;
        let mut b = Builder::new()
            .put(5, 8, King, PlayerColor::Black)
            .put(4, 8, Queen, PlayerColor::Black)
            .win_conditions(WinCondition::Checkmate, WinCondition::AllPiecesCaptured);
        // White: 16 pawns and 2 promoted queens → 16/18 ≈ 89% pawns.
        let mut placed = 0;
        for y in 2..=3 {
            for x in 1..=8 {
                b = b.put(x, y, Pawn, PlayerColor::White);
                placed += 1;
            }
        }
        assert_eq!(placed, 16);
        b = b
            .put(1, 5, Queen, PlayerColor::White)
            .put(8, 5, Queen, PlayerColor::White);
        assert_eq!(b.detect((-INF, INF, -INF, INF)), EvalKind::PawnHorde);
    }

    #[test]
    fn queen_heavy_kingless_white_is_generic() {
        // A White of mostly promoted officers no longer fits the pawn-mass eval, even
        // while still kingless with an AllPiecesCaptured win.
        use PieceType::*;
        const INF: i64 = 1_000_000_000_000_000;
        let mut b = Builder::new()
            .put(5, 8, King, PlayerColor::Black)
            .put(4, 8, Queen, PlayerColor::Black)
            .win_conditions(WinCondition::Checkmate, WinCondition::AllPiecesCaptured)
            .put(4, 2, Pawn, PlayerColor::White); // a single lone pawn
        // Ten White queens → 1/11 ≈ 9% pawns, far below the majority threshold.
        for x in 1..=10 {
            b = b.put(x, 5, Queen, PlayerColor::White);
        }
        assert_eq!(b.detect((-INF, INF, -INF, INF)), EvalKind::Generic);
    }

    #[test]
    fn black_pawn_wall_is_generic() {
        // The horde evaluator only handles a White horde; a Black one is Generic.
        use PieceType::*;
        let mut b = Builder::new()
            .put(5, 1, King, PlayerColor::White)
            .put(4, 1, Queen, PlayerColor::White)
            .win_conditions(WinCondition::AllPiecesCaptured, WinCondition::Checkmate);
        for x in 1..=8 {
            b = b.put(x, 7, Pawn, PlayerColor::Black);
        }
        const INF: i64 = 1_000_000_000_000_000;
        assert_eq!(b.detect((-INF, INF, -INF, INF)), EvalKind::Generic);
    }

    #[test]
    fn bounded_obstacle_board_is_obstocean() {
        use PieceType::*;
        let region = (1, 12, 1, 12);
        let mut b = Builder::new()
            .put(5, 1, King, PlayerColor::White)
            .put(5, 12, King, PlayerColor::Black)
            .put(1, 2, Pawn, PlayerColor::White)
            .put(1, 11, Pawn, PlayerColor::Black);
        // Fill (almost) every remaining square of the 12×12 board with obstacles.
        let occupied = [
            Coordinate::new(5, 1),
            Coordinate::new(5, 12),
            Coordinate::new(1, 2),
            Coordinate::new(1, 11),
        ];
        for x in 1..=12 {
            for y in 1..=12 {
                if !occupied.contains(&Coordinate::new(x, y)) {
                    b = b.put(x, y, Obstacle, PlayerColor::Neutral);
                }
            }
        }
        assert_eq!(b.detect(region), EvalKind::Obstocean);
    }

    #[test]
    fn bounded_board_without_obstacles_is_generic() {
        // A small bounded board that is NOT packed with obstacles stays Generic.
        let kind = Builder::new()
            .put(5, 1, PieceType::King, PlayerColor::White)
            .put(5, 12, PieceType::King, PlayerColor::Black)
            .put(4, 4, PieceType::Rook, PlayerColor::White)
            .detect((1, 12, 1, 12));
        assert_eq!(kind, EvalKind::Generic);
    }
}
