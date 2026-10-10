//! Insufficient material, read from infinitechess.org's generated mating-set tables
//! ([`mating_sets`]): one for unbounded boards, one for bounded boards.

use super::mating_sets::{self, BOUNDED_CAP, CAP, KINDS, ROYAL_KINDS};
use crate::board::{Piece, PieceType, PlayerColor};
use crate::game::{GameRules, GameState};
use rustc_hash::FxHashMap;
use std::cell::{Cell, RefCell};

const CACHE_SET_BITS: u32 = 11;

thread_local! {
    /// Two-way sets keyed by material and pawn hash, most recent first. An entry is
    /// the key with bit 0 set as a valid flag over the [`verdict`] bits.
    static MATERIAL_CACHE: [Cell<u64>; 2 << CACHE_SET_BITS] =
        const { [const { Cell::new(0) }; 2 << CACHE_SET_BITS] };
}

pub fn clear_material_cache() {
    MATERIAL_CACHE.with(|cache| cache.iter().for_each(|e| e.set(0)));
    NO_MATE_CACHE.with(|cache| cache.borrow_mut().clear());
}

thread_local! {
    /// Keyed by (material_hash, white?, border): whether that side's force alone
    /// could ever mate the enemy royals.
    static NO_MATE_CACHE: RefCell<FxHashMap<(u64, bool, Border), bool>> =
        RefCell::new(FxHashMap::default());
}

/// Worlds at most this wide use the bounded table.
const BOUNDED_MAX_SIZE: i64 = 200;
/// The bounded table was only generated down to 8x8, so narrower boards use neither.
const MIN_BOUNDED_WIDTH: i64 = 8;

/// The world border as the tables see it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Border {
    bounded: bool,
    narrow: bool,
}

#[inline]
fn border() -> Border {
    let bounded = crate::moves::get_world_size() <= BOUNDED_MAX_SIZE;
    let (left, right, bottom, top) = crate::moves::get_coord_bounds();
    Border {
        bounded,
        narrow: bounded && (right - left + 1 < MIN_BOUNDED_WIDTH || top - bottom + 1 < MIN_BOUNDED_WIDTH),
    }
}

impl Border {
    /// The table that decides this board, or `None` when it is too narrow for one.
    #[inline]
    fn table(self) -> Option<Table> {
        match (self.bounded, self.narrow) {
            (false, _) => Some(Table::Unbounded),
            (true, false) => Some(Table::Bounded),
            (true, true) => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Table {
    Unbounded,
    Bounded,
}

impl Table {
    #[inline]
    fn cap(self) -> usize {
        match self {
            Table::Unbounded => CAP,
            Table::Bounded => BOUNDED_CAP,
        }
    }

    /// Whether no checkmate is possible with exactly these pieces.
    #[inline]
    fn dead(self, syms: &[u8]) -> bool {
        match self {
            Table::Unbounded => mating_sets::is_dead(syms),
            Table::Bounded => mating_sets::is_dead_bounded(syms),
        }
    }

    /// Whether `white`'s army facing only the enemy royals can never force mate.
    #[inline]
    fn helpless(self, syms: &[u8], white: bool) -> bool {
        match self {
            Table::Unbounded => mating_sets::helpless(syms, white),
            Table::Bounded => mating_sets::helpless_bounded(syms, white),
        }
    }
}

/// True when `white`'s force could not mate the enemy royals even unopposed:
/// however far ahead it is, the game is a draw unless the defender helps.
pub fn side_cannot_mate(game: &GameState, white: bool) -> bool {
    let border = border();
    // Pawn verdicts depend on the pawns' ranks, which the material hash omits.
    let pawns = if white { game.white_pawn_count } else { game.black_pawn_count };
    let key = (game.material_hash, white, border);
    if pawns == 0
        && let Some(v) = NO_MATE_CACHE.with(|c| c.borrow().get(&key).copied())
    {
        return v;
    }
    let mut army = Army::default();
    let v = army.push_side(game, white, false)
        && army.push_side(game, !white, true)
        && border.table().is_some_and(|table| {
            army.helpless(&army.promotion_kinds(&game.game_rules), white, table)
        });
    if pawns > 0 {
        return v;
    }
    NO_MATE_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 4096 {
            c.clear();
        }
        c.insert(key, v);
    });
    v
}

#[inline]
fn get_best_promotion_piece(game_rules: &GameRules) -> Option<PieceType> {
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
fn can_pawn_promote(y: i64, color: PlayerColor, game_rules: &GameRules) -> bool {
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

/// Up to [`CAP`] pieces as table symbols; bit `i` of `promotes` marks a pawn
/// that can still promote.
#[derive(Default, Clone, Copy)]
struct Army {
    syms: [u8; CAP],
    promotes: u8,
    len: usize,
}

impl Army {
    /// Every colored piece, in one board pass; `None` past [`CAP`].
    #[inline]
    fn gather(game: &GameState) -> Option<Army> {
        let mut army = Army::default();
        let rules = &game.game_rules;
        game.board
            .iter_colored()
            .all(|(x, y, p)| army.push(x, y, p, rules))
            .then_some(army)
    }

    /// `white`'s pieces facing only the enemy royals.
    #[inline]
    fn alone(&self, white: bool) -> Army {
        let mut army = Army::default();
        for i in 0..self.len {
            let s = self.syms[i];
            if (s < KINDS) == white || mating_sets::is_royal_symbol(s) {
                army.promotes |= ((self.promotes >> i) & 1) << army.len;
                army.syms[army.len] = s;
                army.len += 1;
            }
        }
        army
    }

    /// Adds one side's pieces, or only its royals. False once past [`CAP`].
    #[inline]
    fn push_side(&mut self, game: &GameState, white: bool, royals_only: bool) -> bool {
        let rules = &game.game_rules;
        if royals_only {
            let royals = if white { &game.white_royals } else { &game.black_royals };
            return royals.iter().all(|c| match game.board.get_piece(c.x, c.y) {
                Some(p) => self.push(c.x, c.y, p, rules),
                None => true,
            });
        }
        game.board
            .iter_pieces_by_color(white)
            .all(|(x, y, p)| self.push(x, y, p, rules))
    }

    #[inline]
    fn push(&mut self, x: i64, y: i64, piece: Piece, rules: &GameRules) -> bool {
        let Some(s) = mating_sets::symbol(piece.piece_type(), piece.color(), x, y) else {
            return true;
        };
        if self.len == CAP {
            return false;
        }
        if piece.piece_type() == PieceType::Pawn && can_pawn_promote(y, piece.color(), rules) {
            self.promotes |= 1 << self.len;
        }
        self.syms[self.len] = s;
        self.len += 1;
        true
    }

    /// Whether no checkmate is possible with these pieces, whatever the pawns
    /// promote to among `kinds`.
    #[inline]
    fn dead(&self, kinds: &[u8], table: Table) -> bool {
        self.for_all_promotions(kinds, &|syms: &[u8]| table.dead(syms))
    }

    /// Whether one side's pieces facing only the enemy royals (as [`Army::alone`]
    /// builds them) can never force mate, whatever its pawns promote to.
    #[inline]
    fn helpless(&self, kinds: &[u8], white: bool, table: Table) -> bool {
        self.for_all_promotions(kinds, &|syms: &[u8]| table.helpless(syms, white))
    }

    #[inline]
    fn for_all_promotions(&self, kinds: &[u8], holds: &dyn Fn(&[u8]) -> bool) -> bool {
        if self.promotes == 0 {
            return holds(&self.syms[..self.len]);
        }
        let mut syms = self.syms;
        all_promotions(&mut syms, self.len, self.promotes, 0, kinds, None, holds)
    }

    /// The table kinds a pawn may promote to, a bishop as both square colors;
    /// empty when no pawn here can promote.
    fn promotion_kinds(&self, rules: &GameRules) -> smallvec::SmallVec<[u8; 8]> {
        const DEFAULT: [PieceType; 4] = [
            PieceType::Queen,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Knight,
        ];
        let mut kinds = smallvec::SmallVec::new();
        if self.promotes == 0 {
            return kinds;
        }
        for &pt in rules.promotion_types.as_deref().unwrap_or(&DEFAULT) {
            if pt == PieceType::Bishop {
                kinds.extend([5, 6]);
            } else if let Some(s) = mating_sets::symbol(pt, PlayerColor::White, 0, 0) {
                kinds.push(s);
            }
        }
        kinds
    }
}

/// Whether `holds` is true for every outcome of the promotable pawns from `from`
/// on: option 0 keeps the pawn, option `o` promotes to `kinds[o - 1]`. Like pawns
/// after one of the same color only take options from its own on, so each
/// multiset is tried once.
fn all_promotions(
    syms: &mut [u8; CAP],
    len: usize,
    promotes: u8,
    from: usize,
    kinds: &[u8],
    prev: Option<(u8, usize)>,
    holds: &dyn Fn(&[u8]) -> bool,
) -> bool {
    let Some(i) = (from..len).find(|&i| promotes & (1 << i) != 0) else {
        return holds(&syms[..len]);
    };
    let pawn = syms[i];
    let color_base = pawn - pawn % KINDS;
    let start = match prev {
        Some((p, o)) if p == pawn => o,
        _ => 0,
    };
    let dead = (start..=kinds.len()).all(|o| {
        syms[i] = if o == 0 { pawn } else { color_base + kinds[o - 1] };
        all_promotions(syms, len, promotes, i + 1, kinds, Some((pawn, o)), holds)
    });
    syms[i] = pawn;
    dead
}

/// Non-royal piece kinds, as [`material_kind`] numbers them.
pub(crate) const MATERIAL_KINDS: usize = (KINDS - ROYAL_KINDS) as usize;
/// The [`material_kind`] of a pawn that cannot promote.
pub(crate) const PAWN_MATERIAL_KIND: u8 = mating_sets::PAWN_KIND - ROYAL_KINDS;

/// The non-royal kind a piece counts as, for [`cheapest_mating_force`]: a pawn
/// that can still promote counts as the piece it would become.
pub(crate) fn material_kind(
    pt: PieceType,
    x: i64,
    y: i64,
    color: PlayerColor,
    rules: &GameRules,
) -> Option<u8> {
    let pt = if pt == PieceType::Pawn && can_pawn_promote(y, color, rules) {
        get_best_promotion_piece(rules).unwrap_or(PieceType::Queen)
    } else {
        pt
    };
    let s = mating_sets::symbol(pt, color, x, y)?;
    (!mating_sets::is_royal_symbol(s)).then(|| s % KINDS - ROYAL_KINDS)
}

/// The cheapest sub-force of `white`'s army that could force mate on the enemy
/// royals alone on an unbounded board, priced by `cost(kind, n)` for using `n`
/// pieces of a [`material_kind`]. `None` when no tabled sub-force can.
pub(crate) fn cheapest_mating_force(
    game: &GameState,
    white: bool,
    cost: impl Fn(u8, u8) -> i64,
) -> Option<i64> {
    let own_royals = if white { &game.white_royals } else { &game.black_royals };
    let mut base = Army::default();
    if own_royals.len() > 1
        || !base.push_side(game, white, true)
        || !base.push_side(game, !white, true)
        || base.len == own_royals.len()
    {
        return None;
    }
    let color = if white { PlayerColor::White } else { PlayerColor::Black };
    let mut have = [0u8; MATERIAL_KINDS];
    for (x, y, p) in game.board.iter_pieces_by_color(white) {
        if let Some(kind) = material_kind(p.piece_type(), x, y, color, &game.game_rules) {
            have[kind as usize] += 1;
        }
    }
    let first = (if white { 0 } else { KINDS }) + ROYAL_KINDS;
    let mut best: Option<i64> = None;
    // Depth-first over per-kind counts, as many non-royals as the table holds.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        kind: usize,
        sub: &mut Army,
        have: &[u8; MATERIAL_KINDS],
        first: u8,
        used: u8,
        spent: i64,
        cost: &dyn Fn(u8, u8) -> i64,
        best: &mut Option<i64>,
    ) {
        if best.is_some_and(|b| spent >= b) {
            return;
        }
        if kind == MATERIAL_KINDS {
            // `first` is a White symbol exactly when White attacks.
            if used > 0 && !mating_sets::helpless(&sub.syms[..sub.len], first < KINDS) {
                *best = Some(spent);
            }
            return;
        }
        let start = sub.len;
        for n in 0..=have[kind] {
            if n > 0 {
                if sub.len == CAP {
                    break;
                }
                sub.syms[sub.len] = first + kind as u8;
                sub.len += 1;
            }
            let extra = if n == 0 { 0 } else { cost(kind as u8, n) };
            walk(kind + 1, sub, have, first, used + n, spent + extra, cost, best);
        }
        sub.len = start;
    }
    walk(0, &mut base, &have, first, 0, 0, &cost, &mut best);
    best
}

/// [`verdict`] bit: the eval scores the position as 0.
const SCORED_ZERO: u64 = 2;
/// [`verdict`] bit: no checkmate is possible at all, so it is a certain draw.
const DEAD: u64 = 4;

/// Both verdicts for a position of at most [`CAP`] pieces: dead when no mate is
/// possible at all, scored zero as well when neither side can force mate against
/// the other's royals alone, so any mate needs the defender's help.
fn compute(game: &GameState) -> u64 {
    let Some(all) = Army::gather(game) else {
        return 0;
    };
    let Some(table) = border().table() else {
        return 0;
    };
    if all.len > table.cap() {
        return 0;
    }
    let kinds = all.promotion_kinds(&game.game_rules);
    if all.dead(&kinds, table) {
        let royal_queen = all.syms[..all.len]
            .iter()
            .any(|&s| s % KINDS == mating_sets::ROYAL_QUEEN_KIND);
        return dead_bits(game, table == Table::Unbounded && royal_queen);
    }
    let helpless = |white| all.alone(white).helpless(&kinds, white, table);
    if helpless(true) && helpless(false) { SCORED_ZERO } else { 0 }
}

/// The verdict of a position where no mate is possible. A neutral piece can wall a
/// royal in where the tables assume open board, and a royal queen can slide to a far
/// border, so the game goes on; the eval still scores it as dead. Obstacles never
/// made a mate possible with classical material and a king each.
fn dead_bits(game: &GameState, unbounded_royal_queen: bool) -> u64 {
    if unbounded_royal_queen {
        return SCORED_ZERO;
    }
    let classical = |pt| matches!(pt, PieceType::King | PieceType::Queen | PieceType::Rook | PieceType::Bishop | PieceType::Knight | PieceType::Pawn);
    let (mut obstacles, mut other_neutral, mut all_classical) = (false, false, true);
    let (mut white_king, mut black_king) = (false, false);
    for (_, _, p) in game.board.iter_all_pieces() {
        match (p.color(), p.piece_type()) {
            (PlayerColor::Neutral, PieceType::Obstacle) => obstacles = true,
            (PlayerColor::Neutral, _) => other_neutral = true,
            (PlayerColor::White, PieceType::King) => white_king = true,
            (PlayerColor::Black, PieceType::King) => black_king = true,
            (_, pt) => all_classical &= classical(pt),
        }
    }
    let promotions = game.game_rules.promotion_types.as_deref().unwrap_or_default();
    let harmless_walls = all_classical && white_king && black_king && promotions.iter().all(|&pt| classical(pt));
    if other_neutral || (obstacles && !harmless_walls) { SCORED_ZERO } else { SCORED_ZERO | DEAD }
}

/// Above [`CAP`], the one proven draw: a king, a knight and any bishops of one
/// square color against a lone king. A king's 3x3 holds at most 5 squares of one
/// color, so at most 5 bishops ever work in a mate, and none mates with 6.
fn compute_above_cap(game: &GameState) -> u64 {
    if border().bounded {
        return 0;
    }
    let white_attacks = game.white_piece_count > game.black_piece_count;
    let (mut kings, mut knights, mut bishop_color) = (0, 0, None);
    let fits = game.board.iter_colored().all(|(x, y, p)| {
        match ((p.color() == PlayerColor::White) == white_attacks, p.piece_type()) {
            (false, PieceType::King) => true,
            (true, PieceType::King) => {
                kings += 1;
                kings == 1
            }
            (true, PieceType::Knight) => {
                knights += 1;
                knights == 1
            }
            (true, PieceType::Bishop) => *bishop_color.get_or_insert((x ^ y) & 1) == (x ^ y) & 1,
            _ => false,
        }
    });
    if fits { dead_bits(game, false) } else { 0 }
}

/// Both insufficient-material verdicts as [`SCORED_ZERO`] and [`DEAD`] bits, cached.
#[inline]
fn verdict(game: &GameState) -> u64 {
    let (white, black) = (game.white_piece_count, game.black_piece_count);
    let above_cap = (white + black) as usize > CAP;
    if above_cap && white.min(black) > 1 {
        return 0;
    }
    if game.game_rules.white_win_condition != crate::game::WinCondition::Checkmate
        || game.game_rules.black_win_condition != crate::game::WinCondition::Checkmate
    {
        return 0;
    }
    // The pawn hash pins each pawn's square, which decides whether it can promote.
    let border = border();
    let salt = (border.bounded as u64 * 0x9E37_79B9_7F4A_7C15) ^ (border.narrow as u64 * 0x6A09_E667_F3BC_C909);
    let key = game.material_hash ^ game.pawn_hash.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let tag = (key ^ salt) & !7;
    let set = 2 * (tag >> (64 - CACHE_SET_BITS)) as usize;
    MATERIAL_CACHE.with(|cache| {
        let (first, second) = (cache[set].get(), cache[set + 1].get());
        if first & !6 == tag | 1 {
            return first;
        }
        if second & !6 == tag | 1 {
            cache[set].set(second);
            cache[set + 1].set(first);
            return second;
        }
        let bits = if above_cap { compute_above_cap(game) } else { compute(game) };
        cache[set + 1].set(first);
        cache[set].set(tag | 1 | bits);
        tag | 1 | bits
    })
}

/// Whether the position is scored as a dead draw: no mate is possible, or only
/// one the defender would have to help with. Helpmate-only games still go on.
#[inline]
pub fn evaluate_insufficient_material(game: &GameState) -> bool {
    verdict(game) & SCORED_ZERO != 0
}

/// Whether no checkmate is possible at all: a certain draw, which search may score
/// without looking further. A helpmate-only position is not one.
#[inline]
pub fn is_dead_draw(game: &GameState) -> bool {
    verdict(game) & DEAD != 0
}

/// Whether the position is a draw by insufficient material for game handlers:
/// no checkmate of either side is possible at all, not even a helpmate.
#[inline]
pub fn evaluate_insufficient_material_game_handler(game: &GameState) -> bool {
    is_dead_draw(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Piece, PieceType, PlayerColor};
    use crate::game::{GameRules, GameState, PromotionRanks, WinCondition};

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

    /// Two centaurs plus a king mate; two camels, giraffes, zebras or roses
    /// never can, not even as a helpmate.
    #[test]
    fn fairy_pairs_follow_the_table() {
        use PieceType as P;
        for (pt, dead) in [
            (P::Centaur, false),
            (P::Rose, true),
            (P::Camel, true),
            (P::Giraffe, true),
            (P::Zebra, true),
        ] {
            let game = create_test_game_with_pieces(&[
                (0, 0, P::King, PlayerColor::White),
                (1, 5, pt, PlayerColor::White),
                (0, 4, pt, PlayerColor::White),
                (0, 2, P::King, PlayerColor::Black),
            ]);
            assert_eq!(evaluate_insufficient_material(&game), dead, "K+2 {pt:?}");
            let handler = evaluate_insufficient_material_game_handler(&game);
            assert_eq!(handler, dead, "K+2 {pt:?}");
        }
    }

    /// The reported counterexample: CE1,5->0,3 is mate in one, yet the position
    /// before it was adjudicated a draw and the evaluation wrapper returned 0.
    #[test]
    fn centaur_premate_is_not_adjudicated_a_draw() {
        use PieceType as P;
        let game = create_test_game_with_pieces(&[
            (0, 0, P::King, PlayerColor::White),
            (1, 5, P::Centaur, PlayerColor::White),
            (0, 4, P::Centaur, PlayerColor::White),
            (0, 2, P::King, PlayerColor::Black),
        ]);
        assert!(!evaluate_insufficient_material(&game));
        assert!(!evaluate_insufficient_material_game_handler(&game));
    }

    /// Every config infinitechess.org's practice mode lists as matable (see
    /// validcheckmates.ts / tests/practice_mates.rs) must be judged WINNABLE —
    /// misjudging it draw silently zeroes the eval and the engine stops trying.
    #[test]
    fn test_site_matable_catalog_is_never_called_a_draw() {
        use PieceType as P;
        // Attacker squares spread near the origin, defender king far away, as in
        // the practice suite.
        const SPOTS: [(i64, i64); 9] = [
            (-2, 3),
            (3, -2),
            (0, 0),
            (-3, 1),
            (4, 2),
            (1, 4),
            (-1, -3),
            (2, 2),
            (-4, 0),
        ];
        let catalog: &[(&str, &[PieceType])] = &[
            ("2Q", &[P::Queen, P::Queen]),
            ("3R", &[P::Rook, P::Rook, P::Rook]),
            ("Q+R+B", &[P::Queen, P::Rook, P::Bishop]),
            ("Q+R+N", &[P::Queen, P::Rook, P::Knight]),
            ("K+2R", &[P::King, P::Rook, P::Rook]),
            ("Q+CH", &[P::Queen, P::Chancellor]),
            ("2CH", &[P::Chancellor, P::Chancellor]),
            (
                "K+4B",
                &[P::King, P::Bishop, P::Bishop, P::Bishop, P::Bishop],
            ),
            ("3AR", &[P::Archbishop, P::Archbishop, P::Archbishop]),
            ("K+AM", &[P::King, P::Amazon]),
            ("K+Q+B", &[P::King, P::Queen, P::Bishop]),
            ("K+Q+N", &[P::King, P::Queen, P::Knight]),
            ("Q+2B", &[P::Queen, P::Bishop, P::Bishop]),
            ("K+R+2B", &[P::King, P::Rook, P::Bishop, P::Bishop]),
            ("K+R+N+B", &[P::King, P::Rook, P::Knight, P::Bishop]),
            ("K+AR+R", &[P::King, P::Archbishop, P::Rook]),
            ("Q+N+B", &[P::Queen, P::Knight, P::Bishop]),
            ("Q+2N", &[P::Queen, P::Knight, P::Knight]),
            ("K+R+2N", &[P::King, P::Rook, P::Knight, P::Knight]),
            ("K+CH+N", &[P::King, P::Chancellor, P::Knight]),
            ("K+2AR", &[P::King, P::Archbishop, P::Archbishop]),
            ("K+2HA+B", &[P::King, P::Hawk, P::Hawk, P::Bishop]),
            (
                "5HU",
                &[P::Huygen, P::Huygen, P::Huygen, P::Huygen, P::Huygen],
            ),
        ];

        for (name, force) in catalog {
            let mut pieces: Vec<(i64, i64, PieceType, PlayerColor)> = force
                .iter()
                .enumerate()
                .map(|(i, pt)| {
                    let (x, y) = SPOTS[i % SPOTS.len()];
                    (x, y, *pt, PlayerColor::White)
                })
                .collect();
            pieces.push((13, 2, PieceType::King, PlayerColor::Black));
            let game = create_test_game_with_pieces(&pieces);
            assert!(
                !evaluate_insufficient_material(&game),
                "{name} vs bare king is matable per the site catalog but was judged a dead draw"
            );
        }
    }

    // Insufficient Material (dead draw)

    #[test]
    fn pawnless_scaling_preserves_large_minor_armies() {
        for army in ["K0,0|N1,2|B2,2|B4,4|B3,2", "K0,0|N1,2|N2,1|B2,2|B3,2"] {
            for white in [true, false] {
                let icn = if white {
                    format!("w (8;q|1;q) {army}|k13,7")
                } else {
                    format!("b (8;q|1;q) {}|K13,7", army.to_lowercase())
                };
                let mut game = GameState::new();
                game.setup_position_from_icn(&icn);
                assert!(!side_cannot_mate(&game, white), "{icn}");
            }
        }
        let mut game = GameState::new();
        game.setup_position_from_icn("w (8;q|1;q) K0,0|N1,2|B2,2|k13,7");
        assert!(side_cannot_mate(&game, true));
    }

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
    fn test_kingless_pairs_are_insufficient() {
        let army = |pieces: &[(i64, i64, PieceType)], king: bool| {
            let mut all: Vec<_> = pieces
                .iter()
                .map(|&(x, y, pt)| (x, y, pt, PlayerColor::White))
                .collect();
            if king {
                all.push((5, 5, PieceType::King, PlayerColor::White));
            }
            all.push((10, 10, PieceType::King, PlayerColor::Black));
            create_test_game_with_pieces(&all)
        };
        for second in [PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
            let pair = [(0, 0, PieceType::Queen), (3, 1, second)];
            let bare = army(&pair, false);
            assert!(evaluate_insufficient_material(&bare), "Q+{second:?}");
            assert!(
                evaluate_insufficient_material_game_handler(&bare),
                "Q+{second:?}"
            );
            // With its king the same pair mates.
            assert!(
                !evaluate_insufficient_material(&army(&pair, true)),
                "K+Q+{second:?}"
            );
        }
        assert!(evaluate_insufficient_material(&army(
            &[(0, 0, PieceType::Rook), (3, 1, PieceType::Rook)],
            false
        )));
        // Two queens, or an amazon with any partner, mate without a king.
        assert!(!evaluate_insufficient_material(&army(
            &[(0, 0, PieceType::Queen), (3, 1, PieceType::Queen)],
            false
        )));
        assert!(!evaluate_insufficient_material(&army(
            &[(0, 0, PieceType::Amazon), (3, 1, PieceType::Rook)],
            false
        )));
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
        assert!(compute(&game) & SCORED_ZERO != 0, "K+B+N vs K insufficient");
    }

    // Sufficient Material

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
        assert!(compute(&game) & SCORED_ZERO == 0, "K+Q+Q vs K sufficient");
    }

    // Both sides insufficient

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

    // Fast exit / misc

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
            variant: None,
        };

        assert!(can_pawn_promote(5, PlayerColor::White, &rules));
        assert!(!can_pawn_promote(10, PlayerColor::White, &rules));
        assert!(can_pawn_promote(3, PlayerColor::Black, &rules));
        assert!(!can_pawn_promote(-5, PlayerColor::Black, &rules));
    }

    #[test]
    fn test_can_pawn_promote_no_ranks() {
        let rules = GameRules {
            promotion_ranks: PromotionRanks {
                white: vec![],
                black: vec![],
            },
            ..Default::default()
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
        assert!(compute(&game) & SCORED_ZERO != 0, "K + dead Pawn vs K should be insufficient");
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

    // Two or three guards cannot force mate, though a bare king can walk into one: the
    // eval scores it dead, the game goes on. A defending knight changes neither.
    #[test]
    fn test_king_guards_vs_king_is_helpmate_only() {
        use PieceType as P;
        let (white, black) = (PlayerColor::White, PlayerColor::Black);
        for guards in [2, 3] {
            let mut pieces = vec![(0, 0, P::King, white), (5, 5, P::King, black)];
            pieces.extend((1..=guards).map(|x| (x, 0, P::Guard, white)));
            // With three guards the knight makes six pieces, past the table.
            let knight = (guards == 2).then_some((9, 3, P::Knight, black));
            for defender in [None, knight] {
                pieces.extend(defender);
                let game = create_test_game_with_pieces(&pieces);
                assert!(evaluate_insufficient_material(&game), "{pieces:?}");
                assert!(!evaluate_insufficient_material_game_handler(&game), "{pieces:?}");
                assert!(side_cannot_mate(&game, true), "{pieces:?}");
            }
        }
    }

    // [3] A lone archbishop (knight+bishop compound) cannot mate on the unbounded board.
    #[test]
    fn test_king_archbishop_vs_king_insufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Archbishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Archbishop vs K is insufficient"
        );
    }

    // [3] AB + two same-color bishops cannot cover both square colors -> draw.
    #[test]
    fn test_king_archbishop_2same_color_bishops_vs_king_insufficient() {
        // Two same-color bishops: (1,1) and (3,1) are both even parity.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Archbishop, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (3, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Archbishop+2 same-color Bishops vs K is a draw"
        );
    }

    // [3] AB + opposite-color bishops covers both colors -> win.
    #[test]
    fn test_king_archbishop_opposite_bishops_vs_king_sufficient() {
        // Opposite-color bishops: (1,1) even parity, (1,2) odd parity.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Archbishop, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (1, 2, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+Archbishop+opposite-color Bishops vs K is sufficient"
        );
    }

    #[test]
    fn test_king_archbishop_1knight_vs_king_insufficient() {
        // The exact spec draw {AB:1, N:1} is still insufficient.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Archbishop, PlayerColor::White),
            (2, 0, PieceType::Knight, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Archbishop+1 Knight vs K is insufficient"
        );
    }

    // [4] Hawk + two bishops is not a win (a single hawk is too weak).
    #[test]
    fn test_king_hawk_2same_color_bishops_vs_king_insufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (3, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Hawk+2 same-color Bishops vs K is a draw"
        );
    }

    #[test]
    fn test_king_hawk_opposite_bishops_vs_king_can_helpmate() {
        // Opposite-color bishops: (1,1) even parity, (1,2) odd parity.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (1, 2, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            !evaluate_insufficient_material(&game),
            "K+Hawk+opposite-color Bishops vs K"
        );
    }

    #[test]
    fn test_king_hawk_1bishop_vs_king_insufficient() {
        // The exact spec draw {HAWK:1, B:[1,0]} is still insufficient.
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (1, 1, PieceType::Bishop, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+Hawk+1 Bishop vs K is insufficient"
        );
    }

    // [13] Two lone hawks cannot force mate.
    #[test]
    fn test_king_2hawks_vs_king_insufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Hawk, PlayerColor::White),
            (2, 0, PieceType::Hawk, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+2 Hawks vs K is a draw"
        );
    }

    // [14] Two lone knightriders cannot force mate.
    #[test]
    fn test_king_2knightriders_vs_king_insufficient() {
        let game = create_test_game_with_pieces(&[
            (0, 0, PieceType::King, PlayerColor::White),
            (1, 0, PieceType::Knightrider, PlayerColor::White),
            (2, 0, PieceType::Knightrider, PlayerColor::White),
            (5, 5, PieceType::King, PlayerColor::Black),
        ]);
        assert!(
            evaluate_insufficient_material(&game),
            "K+2 Knightriders vs K is a draw"
        );
    }

    #[test]
    fn a_void_keeps_the_game_going_but_scores_dead() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K0,0|Q3,1|k10,10|vo20,20");
        assert!(evaluate_insufficient_material(&game));
        assert!(!evaluate_insufficient_material_game_handler(&game));
    }

    #[test]
    fn promotable_pawns_count_every_promotion() {
        let mut game = GameState::new();
        // Two queens mate a king, so two pawns that can still promote might.
        game.setup_position_from_icn("w (8;q,r,b,n|1;q,r,b,n) K0,0|P3,2|P5,2|k10,10");
        assert!(!evaluate_insufficient_material_game_handler(&game));
        // Past their promotion rank they stay pawns, which never mate.
        game.setup_position_from_icn("w (8;q,r,b,n|1;q,r,b,n) K0,0|P3,9|P5,9|k10,10");
        assert!(evaluate_insufficient_material_game_handler(&game));
        assert!(evaluate_insufficient_material(&game));
    }

    #[test]
    fn a_side_without_royals_is_tabled() {
        use PieceType as P;
        let black = PlayerColor::Black;
        let game = create_test_game_with_pieces(&[
            (0, 0, P::RoyalQueen, black),
            (3, 1, P::Queen, black),
            (5, 5, P::Knight, black),
        ]);
        // Off a bounded board a royal queen keeps the game going; the eval is still 0.
        assert!(!evaluate_insufficient_material_game_handler(&game));
        assert!(evaluate_insufficient_material(&game));
        let white = PlayerColor::White;
        let game = create_test_game_with_pieces(&[
            (0, 0, P::Queen, white),
            (3, 1, P::Queen, white),
            (10, 10, P::King, black),
        ]);
        assert!(!evaluate_insufficient_material_game_handler(&game));
        assert!(!evaluate_insufficient_material(&game));
    }

    /// A lone knight can never mate, so White is capped; Black wins by capturing
    /// every piece, so the position is no draw and Black stays uncapped.
    #[test]
    fn a_capture_all_opponent_keeps_the_game_alive() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w N0,0|k10,10");
        assert!(!evaluate_insufficient_material_game_handler(&game));
        assert!(!evaluate_insufficient_material(&game));
        assert!(side_cannot_mate(&game, true));
        assert_eq!(game.game_rules.black_win_condition, WinCondition::AllPiecesCaptured);
    }

    #[test]
    fn a_royal_queen_mates_like_a_queen() {
        let mut game = GameState::new();
        game.setup_position_from_icn("w K0,0|rq10,10|q12,13");
        assert!(!evaluate_insufficient_material(&game));
        assert!(!evaluate_insufficient_material_game_handler(&game));
    }

    /// K+Q vs K+2B is only drawn without help, so the game goes on; the eval still
    /// scores it as the dead draw it is.
    #[test]
    fn queen_vs_two_bishops_is_not_terminal() {
        let mut game = GameState::new();
        game.setup_position_from_icn("b 0/100 1 K4,-1|Q3,1|k3,4|b3,3|b1,1");
        assert!(!evaluate_insufficient_material_game_handler(&game));
        assert!(evaluate_insufficient_material(&game));
    }

    /// Mostly infinitechess.org's own insuffmat cases: which table a border selects,
    /// the bounded cap, and the proven draw above the unbounded cap.
    #[test]
    fn site_border_and_cap_cases() {
        // The border is process-global: put the unbounded one back even on failure.
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                let cap = crate::moves::PLAY_BORDER_CAP;
                crate::moves::set_world_bounds(-cap, cap, -cap, cap);
            }
        }
        let _reset = Reset;
        let draw = |icn: &str| {
            let mut game = GameState::new();
            game.setup_position_from_icn(icn);
            evaluate_insufficient_material_game_handler(&game)
        };
        let border = "-50,50,-50,50";
        let wide = "-1000,1000,-1000,1000";
        assert!(draw(&format!("w {border} K0,0|N5,0|k20,20")));
        assert!(!draw(&format!("w {border} K0,0|R5,0|k20,20")));
        assert!(!draw(&format!("w {border} K0,0|k20,20|rc25,20")));
        assert!(!draw(&format!("w {border} AM0,0|AM3,0|rq20,20")));
        assert!(!draw(&format!("w {border} K0,0|N5,0|N7,0|B9,0|k20,20")), "past the bounded cap");
        assert!(draw(&format!("w {wide} K0,0|R5,0|k20,20")), "wider than 200 is unbounded");
        assert!(!draw("w 1,7,1,8 K1,1|N3,1|k6,8"), "narrower than 8");
        assert!(draw("w K0,0|N1,0|B2,0|B4,0|B6,0|B8,0|B10,0|B12,0|k20,20"));
        assert!(!draw("w K0,0|N1,0|B2,0|B5,0|B6,0|B8,0|B10,0|B12,0|k20,20"));
        assert!(!draw("w K0,0|N3,0|N6,0|N9,0|N12,0|k20,20"));
        assert!(!draw("w K0,0|K5,0|K10,0|k20,20|k25,20|k30,20"));
        // Bounded helpmate-only: scored dead, but the game goes on.
        let scored = |icn: &str| {
            let mut game = GameState::new();
            game.setup_position_from_icn(icn);
            (evaluate_insufficient_material(&game), side_cannot_mate(&game, true))
        };
        assert!(!draw("w 1,8,1,8 K1,1|N2,1|N3,1|k8,8"));
        assert_eq!(scored("w 1,8,1,8 K1,1|N2,1|N3,1|k8,8"), (true, true));
        assert_eq!(scored("w 1,8,1,8 K1,1|B2,1|N3,1|k8,8"), (false, false));
        // A royal queen can slide to a far border, so off a bounded board it only zeroes the eval.
        let eval_only = |icn: &str| scored(icn).0 && !draw(icn);
        assert!(eval_only("w K0,0|N5,0|rq20,20"));
        assert!(draw("w 1,8,1,8 K1,1|N3,1|rq6,8"));
        assert!(!draw("w 1,8,1,8 K1,1|R3,1|rq6,8"));
        // Obstacles end the game only with classical material and a king each.
        assert!(draw("w K0,0|N5,0|k20,20|ob30,30"));
        assert!(eval_only("w K0,0|CA5,0|k20,20|ob30,30"));
    }
}
