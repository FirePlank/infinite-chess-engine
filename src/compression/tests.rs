//! Compression keeps the engine's legal moves: on random positions with far pieces, compressed as
//! if the engine's coordinates ended at 2^40, every legal move maps back onto the original's.

use super::{Big, EXACT_SPAN, prepare};
use crate::board::{Coordinate, PieceType};
use crate::game::GameState;
use crate::moves::{Move, MoveList, PLAY_BORDER_CAP, is_piece_attacking_square};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};

const FORCED_LIMIT: i64 = 1 << 40;
const HOME_TYPES: &[&str] = &[
    "Q", "q", "R", "r", "B", "b", "N", "n", "P", "p", "AM", "am", "CH", "ar", "GU", "ca",
];
const FAR_TYPES: &[&str] = &[
    "Q", "q", "R", "r", "B", "b", "AM", "am", "CH", "ch", "AR", "ar", "N", "n", "P", "p",
];

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(1 | self.0);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }

    fn int(&mut self, low: i64, high: i64) -> i64 {
        low + (self.next() * (high - low + 1) as f64) as i64
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() * items.len() as f64) as usize]
    }

    fn sign(&mut self) -> i64 {
        if self.next() < 0.5 { -1 } else { 1 }
    }

    /// From 5 up to `max_exponent` digits away.
    fn far(&mut self, max_exponent: i64) -> i64 {
        self.int(1, 9) * 10i64.pow(self.int(4, max_exponent) as u32)
            + self.int(-20, 20) * self.sign()
    }
}

struct Piece {
    abbr: String,
    x: i64,
    y: i64,
    rights: bool,
}

/// A home army around the origin plus far pieces, each aligned with a piece or scattered. `forks`
/// adds three far pieces whose lines meet at one far empty square.
fn random_position(r: &mut Rng, extra: &[&str], forks: bool) -> Vec<Piece> {
    let piece = |abbr: &str, x, y| Piece {
        abbr: abbr.to_string(),
        x,
        y,
        rights: false,
    };
    let mut pieces = vec![
        Piece {
            rights: true,
            ..piece("K", 5, 1)
        },
        piece("k", r.int(-6, 6), r.int(6, 9)),
    ];
    if r.next() < 0.5 {
        pieces.push(Piece {
            rights: true,
            ..piece("R", r.far(15), 1)
        }); // A far castling partner.
    }
    let home: Vec<&str> = HOME_TYPES.iter().chain(extra).copied().collect();
    let far_types: Vec<&str> = FAR_TYPES.iter().chain(extra).copied().collect();
    for _ in 0..r.int(2, 12) {
        let abbr = *r.pick(&home);
        pieces.push(piece(abbr, r.int(-8, 8), r.int(-8, 8)));
    }
    for _ in 0..r.int(1, 5) {
        let abbr = *r.pick(&far_types);
        if r.next() < 0.6 {
            let anchor = r.int(0, pieces.len() as i64 - 1) as usize;
            let (ax, ay) = (pieces[anchor].x, pieces[anchor].y);
            let (dx, dy) = *r.pick(&[(1, 0), (0, 1), (1, 1), (1, -1)]);
            let distance = r.far(15) * r.sign();
            let wiggle = if r.next() < 0.5 { 0 } else { r.int(-3, 3) };
            let (wx, wy) = if dy == 0 { (0, wiggle) } else { (wiggle, 0) };
            pieces.push(piece(
                abbr,
                ax + distance * dx + wx,
                ay + distance * dy + wy,
            ));
        } else {
            let (x, y) = (r.far(15) * r.sign(), r.far(15) * r.sign());
            pieces.push(piece(abbr, x, y));
        }
    }
    if forks {
        let (x, y, d1, d2) = (
            r.far(15) * r.sign(),
            r.far(15) * r.sign(),
            r.far(15),
            r.far(15),
        );
        let partner = *r.pick(&["r", "q"]);
        pieces.extend([
            piece("Q", x, y + d1),
            piece(partner, x + d2, y),
            piece("b", x - d1, y - d1),
        ]);
    }
    let mut seen = BTreeSet::new();
    pieces.retain(|p| seen.insert((p.x, p.y)));
    pieces
}

/// The rules section: turn, move rule, maybe promotion ranks and a real left border edge.
fn random_rules(r: &mut Rng, border: bool) -> String {
    let turn = if r.next() < 0.5 { "w" } else { "b" };
    let promotion = if r.next() < 0.5 {
        format!(" ({}|-{})", 8 + r.int(0, 3), 8 + r.int(0, 3))
    } else {
        String::new()
    };
    let left = if border && r.next() < 0.4 {
        format!("-{}", 9 + r.int(0, 3))
    } else {
        "_".to_string()
    };
    format!("{turn} 0/100 1{promotion} {left},_,_,_")
}

fn position_text(pieces: &[Piece]) -> String {
    let entries: Vec<String> = pieces
        .iter()
        .map(|p| {
            format!(
                "{}{},{}{}",
                p.abbr,
                p.x,
                p.y,
                if p.rights { "+" } else { "" }
            )
        })
        .collect();
    entries.join("|")
}

fn legal_moves(game: &mut GameState) -> Vec<Move> {
    let mut list = MoveList::new();
    game.get_pseudo_legal_moves_into(&mut list);
    list.into_iter()
        .filter(|m| {
            let undo = game.make_move(m);
            let illegal = game.is_move_illegal();
            game.undo_move(m, undo);
            !illegal
        })
        .collect()
}

fn load(icn: &str) -> GameState {
    let mut game = GameState::new();
    game.setup_position_from_icn(icn);
    game
}

/// Appends up to `plies` legal moves, so the position carries move history.
fn with_history(icn: &str, r: &mut Rng, plies: i64) -> String {
    let mut moves: Vec<String> = Vec::new();
    for _ in 0..plies {
        let current = if moves.is_empty() {
            icn.to_string()
        } else {
            format!("{icn} {}", moves.join("|"))
        };
        let legal = legal_moves(&mut load(&current));
        if legal.is_empty() {
            break;
        }
        let m = legal[(r.next() * legal.len() as f64) as usize];
        let promotion = if m.promotion.is_some() { "=Q" } else { "" };
        moves.push(format!(
            "{},{}>{},{}{promotion}",
            m.from.x, m.from.y, m.to.x, m.to.y
        ));
    }
    if moves.is_empty() {
        icn.to_string()
    } else {
        format!("{icn} {}", moves.join("|"))
    }
}

fn squares_only(token: &str) -> String {
    token.split('=').next().unwrap().to_string()
}

/// Compresses `icn` as if the engine's coordinates ended at the forced limit, and checks the
/// engine generates exactly the same legal moves there as on the original, once mapped back.
fn expect_same_legal_moves(icn: &str) {
    let original: BTreeSet<String> = legal_moves(&mut load(icn))
        .iter()
        .map(|m| format!("{},{}>{},{}", m.from.x, m.from.y, m.to.x, m.to.y))
        .collect();
    let Some(compressed) = prepare(icn, &Big::from(FORCED_LIMIT)).unwrap() else {
        return; // Nothing beyond the limit, or nothing far: the engine reads it as it is.
    };
    let mut game = compressed.game();
    let lifted: BTreeSet<String> = legal_moves(&mut game)
        .iter()
        .map(|m| {
            squares_only(
                compressed
                    .lifter
                    .lift_line(std::slice::from_ref(m))
                    .first()
                    .unwrap_or_else(|| panic!("unliftable move in {icn}")),
            )
        })
        .collect();
    assert_eq!(lifted, original, "{icn}");
}

#[test]
fn passes_a_position_within_the_limit_through() {
    let icn = "w _,_,_,_ K0,0|k5,5|Q900000000000000000,3";
    assert!(prepare(icn, &Big::from(PLAY_BORDER_CAP)).unwrap().is_none());
}

#[test]
fn keeps_legal_moves_with_far_pieces() {
    let mut r = Rng(7);
    for run in 0..150 {
        let icn = format!(
            "{} {}",
            random_rules(&mut r, true),
            position_text(&random_position(&mut r, &[], run % 3 == 2))
        );
        let icn = if r.next() < 0.5 {
            let plies = 1 + r.int(0, 29);
            with_history(&icn, &mut r, plies)
        } else {
            icn
        };
        expect_same_legal_moves(&icn);
    }
}

/// A random position with Huygens at home and far out, and far pieces on Huygens' ranks and files,
/// half a prime distance away, so far Huygens attack, check and are blocked.
fn huygen_position(r: &mut Rng) -> Vec<Piece> {
    let mut pieces = random_position(r, &["HU", "hu"], false);
    let on_line_of = |r: &mut Rng, anchor: &Piece, abbr: &str| {
        let mut distance = r.far(15);
        if r.next() < 0.5 {
            while !crate::utils::is_prime_u64(distance as u64) {
                distance += 1;
            }
        }
        let distance = distance * r.sign();
        let (x, y) = if r.next() < 0.5 {
            (anchor.x + distance, anchor.y)
        } else {
            (anchor.x, anchor.y + distance)
        };
        Piece {
            abbr: abbr.to_string(),
            x,
            y,
            rights: false,
        }
    };
    for _ in 0..r.int(1, 3) {
        let anchor = r.int(0, pieces.len() as i64 - 1) as usize;
        let abbr = *r.pick(&["HU", "hu"]);
        let far = on_line_of(r, &pieces[anchor], abbr);
        pieces.push(far);
    }
    let huygens: Vec<usize> = (0..pieces.len())
        .filter(|&i| pieces[i].abbr.eq_ignore_ascii_case("hu"))
        .collect();
    for _ in 0..r.int(0, 4) {
        let huygen = *r.pick(&huygens);
        let abbr = *r.pick(FAR_TYPES);
        let far = on_line_of(r, &pieces[huygen], abbr);
        pieces.push(far);
    }
    let mut seen = BTreeSet::new();
    pieces.retain(|p| seen.insert((p.x, p.y)));
    pieces
}

/// Runs `check` on every case number below `count`, spread over every core.
fn on_every_core(count: u32, check: impl Fn(u32) + Sync) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get()) as u32;
    std::thread::scope(|s| {
        for core in 0..cores {
            let check = &check;
            s.spawn(move || (core..count).step_by(cores as usize).for_each(check));
        }
    });
}

/// How often the original positions held each Huygen interaction beyond the exact span, so a
/// generator that stops producing them fails its test instead of passing it vacuously.
#[derive(Default, Debug)]
struct HuygenCoverage {
    captures: AtomicU32,
    checks: AtomicU32,
    blocks: AtomicU32,
}

impl HuygenCoverage {
    fn count(&self, icn: &str) {
        let mut game = load(icn);
        let moves = legal_moves(&mut game);
        let far = |a: Coordinate, b: Coordinate| {
            a.x.abs_diff(b.x).max(a.y.abs_diff(b.y)) > EXACT_SPAN as u64
        };
        let captures = moves.iter().filter(|m| {
            m.piece.piece_type() == PieceType::Huygen
                && far(m.from, m.to)
                && game.board.get_piece(m.to.x, m.to.y).is_some()
        });
        self.captures
            .fetch_add(captures.count() as u32, Ordering::Relaxed);
        let Some(king) = game
            .board
            .iter()
            .find(|(_, _, p)| p.color() == game.turn && p.piece_type() == PieceType::King)
            .map(|(x, y, _)| Coordinate::new(x, y))
        else {
            return;
        };
        for (x, y, huygen) in game.board.iter() {
            let at = Coordinate::new(x, y);
            if huygen.piece_type() != PieceType::Huygen
                || huygen.color() == game.turn
                || (x != king.x && y != king.y)
                || !far(at, king)
                || !is_piece_attacking_square(
                    &game.board,
                    &huygen,
                    &at,
                    &king,
                    &game.spatial_indices,
                    &game.game_rules,
                )
            {
                continue;
            }
            self.checks.fetch_add(1, Ordering::Relaxed);
            let between = |c: Coordinate| {
                let on_line = if x == king.x { c.x == x } else { c.y == y };
                let (low, high) = if x == king.x {
                    (y.min(king.y), y.max(king.y))
                } else {
                    (x.min(king.x), x.max(king.x))
                };
                let along = if x == king.x { c.y } else { c.x };
                on_line && low < along && along < high
            };
            let blocks = moves
                .iter()
                .filter(|m| m.from != king && between(m.to))
                .count();
            self.blocks.fetch_add(blocks as u32, Ordering::Relaxed);
        }
    }
}

#[test]
fn keeps_legal_moves_with_far_huygens() {
    let coverage = HuygenCoverage::default();
    // No border, which is process-global: every case sets the same unbounded one.
    on_every_core(1200, |run| {
        let mut r = Rng(1000 + run);
        let icn = format!(
            "{} {}",
            random_rules(&mut r, false),
            position_text(&huygen_position(&mut r))
        );
        let icn = if r.next() < 0.5 {
            let plies = 1 + r.int(0, 19);
            with_history(&icn, &mut r, plies)
        } else {
            icn
        };
        expect_same_legal_moves(&icn);
        coverage.count(&icn);
    });
    assert!(
        coverage.captures.load(Ordering::Relaxed) > 200,
        "{coverage:?}"
    );
}

/// White to move in check from a far black Huygen, with white pieces able to reach the check line
/// between them: half at a prime distance from the Huygen, where a block holds, half not.
fn huygen_check_position(r: &mut Rng) -> String {
    let piece = |abbr: &str, x, y| Piece {
        abbr: abbr.to_string(),
        x,
        y,
        rights: false,
    };
    let prime_from = |mut d: i64| {
        while !crate::utils::is_prime_u64(d as u64) {
            d += 1;
        }
        d
    };
    let distance = prime_from(r.far(15));
    let (sign, vertical) = (r.sign(), r.next() < 0.5);
    // A square on the check line, `along` from the king, and `across` beside it.
    let at = |along: i64, across: i64| {
        if vertical {
            (across, along * sign)
        } else {
            (along * sign, across)
        }
    };
    let (hx, hy) = at(distance, 0);
    let mut pieces = vec![
        piece("K", 0, 0),
        piece("hu", hx, hy),
        piece("k", r.int(-8, 8), r.int(20, 30)),
    ];
    for _ in 0..r.int(2, 7) {
        // Near the king, near the Huygen, or far from both.
        let along = match r.int(0, 2) {
            0 => r.int(2, 2000),
            1 => distance - r.int(2, 2000),
            _ => r.int(EXACT_SPAN, distance - EXACT_SPAN),
        };
        let along = if r.next() < 0.5 {
            distance - prime_from(distance - along)
        } else {
            along
        };
        if along <= 0 || along >= distance {
            continue;
        }
        let reach = r.far(15) * r.sign();
        let near = *r.pick(&[2, 3, 5, 7]) * r.sign();
        let ((x, y), abbr) = match r.int(0, 4) {
            0 => (at(along, reach), *r.pick(&["R", "Q", "CH", "AM"])),
            1 => (at(along + reach, reach), *r.pick(&["B", "Q", "AR", "AM"])),
            2 => (
                at(along + 1, 2 * r.sign()),
                *r.pick(&["N", "CH", "AR", "AM"]),
            ),
            3 => (at(along, near), "HU"),
            _ => (
                at(along, r.int(-3, 3)),
                *r.pick(&["r", "q", "n", "b", "P", "GU"]),
            ),
        };
        pieces.push(piece(abbr, x, y));
    }
    let mut seen = BTreeSet::new();
    pieces.retain(|p| seen.insert((p.x, p.y)));
    format!("w 0/100 1 _,_,_,_ {}", position_text(&pieces))
}

#[test]
fn keeps_every_block_of_a_far_huygen_check() {
    let coverage = HuygenCoverage::default();
    on_every_core(1500, |run| {
        let icn = huygen_check_position(&mut Rng(5000 + run));
        expect_same_legal_moves(&icn);
        coverage.count(&icn);
    });
    assert!(
        coverage.blocks.load(Ordering::Relaxed) > 1000,
        "{coverage:?}"
    );
}

#[test]
fn keeps_legal_moves_with_knightriders() {
    let mut r = Rng(11);
    for run in 0..60 {
        let icn = format!(
            "{} {}",
            random_rules(&mut r, true),
            position_text(&random_position(&mut r, &["NR", "nr"], run % 3 == 2))
        );
        expect_same_legal_moves(&icn);
    }
}

#[test]
fn keeps_legal_moves_with_the_army_far_from_the_origin() {
    let mut r = Rng(31);
    for _ in 0..40 {
        let shift = r.int(1, 9) * 10i64.pow(13) * r.sign();
        let mut pieces = random_position(&mut r, &[], false);
        pieces
            .iter_mut()
            .for_each(|p| (p.x, p.y) = (p.x + shift, p.y - shift));
        let icn = format!("{} {}", random_rules(&mut r, false), position_text(&pieces));
        let plies = 10 + r.int(0, 19);
        expect_same_legal_moves(&with_history(&icn, &mut r, plies));
    }
}

#[test]
fn keeps_a_border_edge_beyond_the_limit_beside_far_pieces() {
    expect_same_legal_moves(
        "w 0/100 1 -10,300000000000000,-10,10 K0,0|k5,5|R299999999999995,3|n299999999999999,9",
    );
    expect_same_legal_moves(
        "w 0/100 1 -10,300000000000000,-10,10 K0,0|k5,5|Q299999999999995,3|r299999999999990,-8",
    );
}

#[test]
fn maps_a_long_diagonal_slide_to_a_crossing_set_by_a_compressed_gap() {
    // The black bishop sits 20008 ranks above the army, so its diagonals cross the
    // archbishop's 10009 squares out: a short-looking move whose crossing is compressed.
    expect_same_legal_moves(
        "b 0/100 1 (9|-10) -10,_,_,_ K60000000000000,40000000000000|k59999999999994,40000000000003|B59999999999998,40000000000008|B59999999999994,40000000000004|R60000000000004,39999999999996|R60000000000001,39999999999993|P59999999999993,40000000000000|am60000000000000,39999999999997|ar59999999999997,39999999999993|P59999999999994,39999999999998|r60000000000001,40000000000000|r59999999999999,40000000000002|b59999999999994,40000000020008|Q59999999999997,5039999999999986",
    );
}

#[test]
fn cuts_a_line_at_a_slide_that_would_pass_a_piece_it_blocks() {
    // Black's queen checks along the king's diagonal, then white's far rook slides onto it: beyond
    // the queen on the original, so no block, but between her and the king compressed.
    let icn = "b 0/100 1 (8|1) _,_,_,_ K0,0|k-2,4|r-8,1|Q6,6|R7,-3|N-3,-3|R90000000000000000000000000000000000000000000000000000015,-999999999999999999999999999999999999999999999999981|q600000000000000000000000000000000007,-899999999999999999999999999999987";
    let compressed = prepare(icn, &Big::from(PLAY_BORDER_CAP)).unwrap().unwrap();
    let mut game = compressed.game();
    let squares: Vec<(i64, i64)> = compressed
        .icn
        .split_whitespace()
        .last()
        .unwrap()
        .split('|')
        .map(|p| {
            let (x, y) = p
                .trim_start_matches(|ch: char| ch.is_ascii_alphabetic())
                .trim_end_matches('+')
                .split_once(',')
                .unwrap();
            (x.parse().unwrap(), y.parse().unwrap())
        })
        .collect();
    let [.., rook, queen] = squares[..] else {
        panic!()
    };
    let find = |game: &mut GameState, from: (i64, i64), to: (i64, i64)| {
        *legal_moves(game)
            .iter()
            .find(|m| (m.from.x, m.from.y, m.to.x, m.to.y) == (from.0, from.1, to.0, to.1))
            .unwrap()
    };
    let check = find(&mut game, queen, (queen.0, -queen.0));
    let undo = game.make_move(&check);
    let block = find(&mut game, rook, (-rook.1, rook.1));
    game.undo_move(&check, undo);
    assert_eq!(compressed.lifter.lift_line(&[check, block]).len(), 1);
}

#[test]
fn reads_a_never_promoting_side_as_no_rank() {
    let ranks = format!("({};q|{};q)", i64::MIN, i64::MAX);
    let near = format!("w 0/100 1 {ranks} _,_,_,_ K0,0|k5,5|P3,3|p4,6");
    assert!(prepare(&near, &Big::from(1i64 << 14)).unwrap().is_none());
    let far = format!("w 0/100 1 {ranks} _,_,_,_ K0,0|k5,5|P3,3|q90000000000000000000000000,7");
    let compressed = prepare(&far, &Big::from(PLAY_BORDER_CAP)).unwrap().unwrap();
    assert!(compressed.icn.contains(&ranks), "{}", compressed.icn);
}

/// The first prime at or past `from`.
fn prime_from(from: &Big) -> Big {
    let mut p = from.clone();
    while !p.is_prime() {
        p = &p + &Big::from(1);
    }
    p
}

fn ten_to(exponent: usize) -> Big {
    Big::parse(&format!("1{}", "0".repeat(exponent))).unwrap()
}

/// The engine's best move at `depth`, as a site token on the original board.
fn best_move(icn: &str, depth: usize) -> String {
    let mut engine = crate::Engine::try_from_icn_native(icn, None).unwrap();
    let (m, _, _) = engine
        .search_native(0, Some(depth), true, None, Some(1))
        .unwrap();
    engine
        .move_token(&m)
        .unwrap_or_else(|| panic!("unliftable best move in {icn}"))
}

#[test]
fn takes_a_queen_a_far_prime_distance_away_past_a_pawn_at_a_composite_one() {
    let queen = prime_from(&ten_to(40));
    let pawn = &queen - &Big::from(1);
    let icn = format!("w 0/100 1 _,_,_,_ K-50,-50|k-50,50|HU0,0|p{pawn},0|q{queen},0");
    assert_eq!(best_move(&icn, 3), format!("0,0>{queen},0"));
}

#[test]
fn mates_from_a_far_prime_distance_and_not_a_composite_one() {
    // The black king is walled in by its own pawns; the one between is a composite distance away.
    let mate = prime_from(&ten_to(40));
    let decoy = &mate + &Big::from(1);
    let walls = "p-1,1|p0,1|p1,1|p-1,0|p1,0|p-1,-1|p0,-1|p1,-1";
    let icn = format!("w 0/100 1 _,_,_,_ K-100,-100|k0,0|{walls}|HU{mate},3|HU{decoy},2");
    assert_eq!(best_move(&icn, 2), format!("{mate},3>{mate},0"));
}

#[test]
fn blocks_a_far_check_only_a_prime_distance_from_the_huygen() {
    // The white king is walled in; of two rooks that can drop onto the check line, only one lands
    // a prime distance from the Huygen, which is the only legal move.
    let huygen = prime_from(&ten_to(40));
    let block = &huygen - &prime_from(&ten_to(39));
    let miss = &block - &Big::from(1);
    let walls = "P-1,1|P0,1|P1,1|P-1,0|P1,0|P-1,-1|P0,-1|P1,-1";
    let icn = format!("w 0/100 1 _,_,_,_ K0,0|{walls}|R{miss},50|R{block},-70|k5,100|hu{huygen},0");
    assert_eq!(best_move(&icn, 2), format!("{block},-70>{block},0"));
}

#[test]
fn skewers_a_king_to_a_queen_behind_it_from_far() {
    let king = prime_from(&ten_to(40));
    let queen = prime_from(&(&king + &ten_to(20)));
    // Knights for both sides, so that winning the queen wins the game rather than drawing it.
    let knights = "N-60,30|N-62,33|n-30,60|n-33,62";
    let icn = format!("w 0/100 1 _,_,_,_ K-50,-50|{knights}|HU0,5|k{king},0|q{queen},0");
    assert_eq!(best_move(&icn, 4), "0,5>0,0");
    let after = format!("{icn} 0,5>0,0|{king},0>{king},1");
    assert_eq!(best_move(&after, 2), format!("0,0>{queen},0"));
}

#[test]
fn reads_a_check_from_a_huygen_promoted_far_from_the_king() {
    // The knight beside the king blocks a promoted queen's rank, but is an even distance from the
    // new piece, so a Huygen checks past it.
    let king = prime_from(&ten_to(40));
    let knight = &king - &Big::from(1);
    let rules = "w 0/100 1 (8;q,hu|1;q,hu) _,_,_,_";
    let icn = format!("{rules} K-50,-50|P0,7|n{knight},8|k{king},8");
    for (promotion, check) in [("HU", true), ("Q", false)] {
        let promoted = format!("{icn} 0,7>0,8={promotion}");
        let compressed = prepare(&promoted, &Big::from(PLAY_BORDER_CAP))
            .unwrap()
            .unwrap();
        assert_eq!(compressed.game().is_in_check(), check, "{promotion}");
    }
}
