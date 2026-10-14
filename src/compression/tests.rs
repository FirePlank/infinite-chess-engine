//! Compression keeps the engine's legal moves: on random positions with far pieces, compressed as
//! if the engine's coordinates ended at 2^40, every legal move maps back onto the original's.

use super::{Big, Refusal, prepare};
use crate::game::GameState;
use crate::moves::{Move, MoveList, PLAY_BORDER_CAP};
use std::collections::BTreeSet;

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
        format!("-{PLAY_BORDER_CAP}")
    };
    format!(
        "{turn} 0/100 1{promotion} {left},{PLAY_BORDER_CAP},-{PLAY_BORDER_CAP},{PLAY_BORDER_CAP}"
    )
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
    let mut game = load(&compressed.icn);
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
    let icn = format!(
        "w -{c},{c},-{c},{c} K0,0|k5,5|Q900000000000000000,3",
        c = PLAY_BORDER_CAP
    );
    assert!(
        prepare(&icn, &Big::from(PLAY_BORDER_CAP))
            .unwrap()
            .is_none()
    );
}

#[test]
fn refuses_a_huygen_position_that_has_to_be_compressed() {
    let far = "1".to_string() + &"0".repeat(30);
    for pieces in [
        format!("K0,0|k5,5|HU3,0|q{far},7"),
        format!("K{far},{far}|k{far}5,{far}5|HU{far}3,{far}"),
    ] {
        let icn = format!("w -{c},{c},-{c},{c} {pieces}", c = PLAY_BORDER_CAP);
        assert_eq!(
            prepare(&icn, &Big::from(PLAY_BORDER_CAP)).err(),
            Some(Refusal::ReadsExactDistances),
            "{pieces}"
        );
    }
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
fn maps_a_long_diagonal_slide_to_a_crossing_set_by_a_compressed_gap() {
    // The black bishop sits 20008 ranks above the army, so its diagonals cross the
    // archbishop's 10009 squares out: a short-looking move whose crossing is compressed.
    let c = PLAY_BORDER_CAP;
    expect_same_legal_moves(&format!(
        "b 0/100 1 (9|-10) -10,{c},-{c},{c} K60000000000000,40000000000000|k59999999999994,40000000000003|B59999999999998,40000000000008|B59999999999994,40000000000004|R60000000000004,39999999999996|R60000000000001,39999999999993|P59999999999993,40000000000000|am60000000000000,39999999999997|ar59999999999997,39999999999993|P59999999999994,39999999999998|r60000000000001,40000000000000|r59999999999999,40000000000002|b59999999999994,40000000020008|Q59999999999997,5039999999999986"
    ));
}

#[test]
fn cuts_a_line_at_a_slide_that_would_pass_a_piece_it_blocks() {
    // Black's queen checks along the king's diagonal, then white's far rook slides onto it: beyond
    // the queen on the original, so no block, but between her and the king compressed.
    let c = PLAY_BORDER_CAP;
    let icn = format!(
        "b 0/100 1 (8|1) -{c},{c},-{c},{c} K0,0|k-2,4|r-8,1|Q6,6|R7,-3|N-3,-3|R90000000000000000000000000000000000000000000000000000015,-999999999999999999999999999999999999999999999999981|q600000000000000000000000000000000007,-899999999999999999999999999999987"
    );
    let compressed = prepare(&icn, &Big::from(c)).unwrap().unwrap();
    let mut game = load(&compressed.icn);
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
    let c = PLAY_BORDER_CAP;
    let ranks = format!("({};q|{};q)", i64::MIN, i64::MAX);
    let near = format!("w 0/100 1 {ranks} -{c},{c},-{c},{c} K0,0|k5,5|P3,3|p4,6");
    assert!(prepare(&near, &Big::from(1i64 << 14)).unwrap().is_none());
    let far =
        format!("w 0/100 1 {ranks} -{c},{c},-{c},{c} K0,0|k5,5|P3,3|q90000000000000000000000000,7");
    let compressed = prepare(&far, &Big::from(c)).unwrap().unwrap();
    assert!(compressed.icn.contains(&ranks), "{}", compressed.icn);
}
