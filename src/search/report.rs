//! Reported scores on Stockfish's scale: +100 wins the engine half its games outright.
//! The internal score for that is fitted per position on 49.2k self-play games (10+0.1).
//! Chess, Pawn Horde and Obstocean are not fitted and report the internal score.

use super::MATE_SCORE;
use crate::board::{PieceType, PlayerColor};
use crate::evaluation::eval_kind::{self, EvalKind};
use crate::game::GameState;

const INTERCEPT: f64 = 5.66303825378418;
/// Rider, leaper and pawn material (log, relative to Classical), piece count (log), obstacles,
/// material imbalance, king distance (log), king shelter, piece spread (log).
const BASE_COEF: [f64; 9] = [
    0.8336887359619141,
    -0.0033510178327560425,
    0.17853876948356628,
    -1.2726765871047974,
    0.1808243691921234,
    0.6011514663696289,
    -0.06707238405942917,
    0.09104565531015396,
    -0.3033381998538971,
];
/// Every product of two base features, in order (0,0), (0,1), ..., (8,8).
const QUAD_COEF: [f64; 45] = [
    0.1396501511335373,
    0.051229771226644516,
    0.052717071026563644,
    -0.21872350573539734,
    -0.19252155721187592,
    0.07721490412950516,
    -0.010765024460852146,
    0.1709117591381073,
    -0.04522569105029106,
    -0.026109805330634117,
    0.07036672532558441,
    -0.15992332994937897,
    0.05287276208400726,
    -0.04598768427968025,
    -0.053287651389837265,
    -0.0724925622344017,
    0.0026073954068124294,
    -0.09718146920204163,
    0.72502201795578,
    -0.49173399806022644,
    -0.03626950457692146,
    0.0386645644903183,
    -0.13248232007026672,
    0.04072271287441254,
    -0.44499021768569946,
    1.187364935874939,
    0.285158634185791,
    -0.0260500255972147,
    -0.14204823970794678,
    0.018107106909155846,
    -0.2774600684642792,
    -0.1579153835773468,
    0.10972176492214203,
    0.03023366816341877,
    -0.011583604849874973,
    -0.09459012746810913,
    0.09097235649824142,
    -0.17034785449504852,
    0.005962779745459557,
    0.11397676169872284,
    -0.05810220539569855,
    -0.00999677088111639,
    -0.013776328414678574,
    0.12270417809486389,
    0.04764712601900101,
];
/// Counts / 8 of pawn, knight, bishop, rook, queen, guard, centaur, camel, giraffe, zebra,
/// knightrider, hawk, archbishop, rose, huygen, chancellor, amazon, obstacle.
const COUNT_COEF: [f64; 18] = [
    0.25084829330444336,
    -0.2727685272693634,
    -0.42071616649627686,
    -0.6340559124946594,
    -1.167832851409912,
    -0.6690239906311035,
    -1.5405532121658325,
    -0.04238661751151085,
    -0.07000535726547241,
    -0.7326964139938354,
    -0.7027589678764343,
    0.43839791417121887,
    -1.6188023090362549,
    -0.37429139018058777,
    0.03063448891043663,
    -1.4439361095428467,
    -2.662442207336426,
    0.22604499757289886,
];
const BASE_RANGE: [(f64, f64); 9] = [
    (-2.995732273553991, 0.9125618396394037),
    (-2.995732273553991, 1.411374045516914),
    (-2.772588722239781, 1.4759065198095778),
    (-1.2685113254635072, 1.0775588794702773),
    (0.0, 2.8),
    (0.0, 3.0),
    (-0.980829253011726, 1.3862943611198908),
    (0.0, 2.9),
    (-0.3101549283038394, 1.4508328822574614),
];
const COUNT_RANGE: [(f64, f64); 18] = [
    (0.125, 8.75),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.25),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.5),
    (0.0, 0.25),
    (0.0, 3.5),
];
const WIN_HALF_EVAL_RANGE: (f64, f64) = (50.0, 1000.0);

/// Value, class (0 pawn, 1 rider, 2 leaper, 3 other) and count slot of a piece type, at the
/// piece values the fit used.
fn piece_info(pt: PieceType) -> (i64, u8, Option<usize>) {
    match pt {
        PieceType::Pawn => (100, 0, Some(0)),
        PieceType::Knight => (315, 2, Some(1)),
        PieceType::Bishop => (450, 1, Some(2)),
        PieceType::Rook => (618, 1, Some(3)),
        PieceType::Queen => (1518, 1, Some(4)),
        PieceType::Guard => (232, 2, Some(5)),
        PieceType::Centaur => (640, 2, Some(6)),
        PieceType::Camel => (175, 2, Some(7)),
        PieceType::Giraffe => (165, 2, Some(8)),
        PieceType::Zebra => (180, 2, Some(9)),
        PieceType::Knightrider => (900, 1, Some(10)),
        PieceType::Hawk => (540, 2, Some(11)),
        PieceType::Archbishop => (1080, 1, Some(12)),
        PieceType::Rose => (997, 1, Some(13)),
        PieceType::Huygen => (330, 1, Some(14)),
        PieceType::Chancellor => (1060, 1, Some(15)),
        PieceType::Amazon => (1793, 1, Some(16)),
        PieceType::Obstacle => (0, 3, Some(17)),
        _ => (0, 3, None),
    }
}

fn is_neutral_square(pt: PieceType) -> bool {
    matches!(pt, PieceType::Obstacle | PieceType::Void)
}

/// Internal score at which the engine wins half its games outright from this position, or
/// `None` where the scale was not measured: positions the generic evaluation does not score.
pub fn win_half_eval(game: &GameState) -> Option<f64> {
    (eval_kind::detect(game) == EvalKind::Generic).then(|| fitted_win_half_eval(game))
}

fn fitted_win_half_eval(game: &GameState) -> f64 {
    let (mut rider, mut leaper, mut pawn) = (0i64, 0i64, 0i64);
    let (mut pieces, mut neutral, mut imbalance) = (0usize, 0usize, 0i64);
    let mut counts = [0usize; 18];
    let mut kings: Vec<(i64, i64, PlayerColor)> = Vec::new();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (x, y, piece) in game.board.iter() {
        let pt = piece.piece_type();
        let (value, class, slot) = piece_info(pt);
        pieces += 1;
        match class {
            0 => pawn += value,
            1 => rider += value,
            2 => leaper += value,
            _ => {}
        }
        if let Some(slot) = slot {
            counts[slot] += 1;
        }
        if is_neutral_square(pt) {
            neutral += 1;
        } else {
            xs.push(x);
            ys.push(y);
        }
        match piece.color() {
            PlayerColor::White => imbalance += value,
            PlayerColor::Black => imbalance -= value,
            PlayerColor::Neutral => {}
        }
        if pt == PieceType::King {
            kings.push((x, y, piece.color()));
        }
    }
    let (mut king_distance, mut shelter) = (0u64, 0usize);
    if let [(ax, ay, _), (bx, by, _)] = kings[..] {
        king_distance = ax.abs_diff(bx).max(ay.abs_diff(by));
        for (x, y, piece) in game.board.iter() {
            let pt = piece.piece_type();
            if pt == PieceType::King || is_neutral_square(pt) {
                continue;
            }
            shelter += kings
                .iter()
                .filter(|&&(kx, ky, kc)| {
                    kc == piece.color() && kx.abs_diff(x).max(ky.abs_diff(y)) <= 2
                })
                .count();
        }
    }
    let spread = if xs.is_empty() {
        0
    } else {
        xs.sort_unstable();
        ys.sort_unstable();
        let (lo, hi) = (xs.len() / 10, 9 * xs.len() / 10);
        xs[hi].abs_diff(xs[lo]) + ys[hi].abs_diff(ys[lo])
    };

    let log_rel = |x: i64, reference: f64| (x as f64).max(reference * 0.05).ln() - reference.ln();
    let mut base = [
        log_rel(rider, 7308.0),
        log_rel(leaper, 1260.0),
        log_rel(pawn, 1600.0),
        (pieces.max(2) as f64 / 32.0).ln(),
        neutral as f64 / 10.0,
        imbalance.unsigned_abs().min(3000) as f64 / 1000.0,
        (king_distance.min(100_000) as f64).ln_1p() - 8f64.ln(),
        shelter as f64 / 10.0,
        (spread.min(100_000) as f64).ln_1p() - 15f64.ln(),
    ];
    for (f, &(lo, hi)) in base.iter_mut().zip(&BASE_RANGE) {
        *f = f.clamp(lo, hi);
    }
    let mut log_a = INTERCEPT;
    let mut q = 0;
    for i in 0..9 {
        log_a += BASE_COEF[i] * base[i];
        for j in i..9 {
            log_a += QUAD_COEF[q] * base[i] * base[j];
            q += 1;
        }
    }
    for ((&n, &coef), &(lo, hi)) in counts.iter().zip(&COUNT_COEF).zip(&COUNT_RANGE) {
        log_a += coef * (n as f64 / 8.0).clamp(lo, hi);
    }
    log_a
        .exp()
        .clamp(WIN_HALF_EVAL_RANGE.0, WIN_HALF_EVAL_RANGE.1)
}

/// An internal score on the reported scale, given [`win_half_eval`] of the root position.
/// Mate scores, and every score where the scale is unknown, pass through unchanged.
pub fn report_cp(score: i32, win_half_eval: Option<f64>) -> i32 {
    match win_half_eval {
        Some(a) if score.abs() <= MATE_SCORE => (score as f64 * 100.0 / a).round() as i32,
        _ => score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from the fit's own feature code, so the two cannot drift apart.
    #[test]
    fn test_win_half_eval_matches_the_fit() {
        let cases = [
            ("w K5,1|R1,1|P2,4|k6,8|r8,8|p3,6", REF_ROOK_ENDING),
            (
                "w P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+",
                REF_CLASSICAL,
            ),
        ];
        for (icn, expected) in cases {
            let mut game = GameState::new();
            game.setup_position_from_icn(icn);
            let a = win_half_eval(&game).expect("a generic position");
            assert!((a - expected).abs() < 1e-3, "{icn}: {a} vs {expected}");
        }
    }

    #[test]
    fn test_report_cp_scales_and_keeps_mates() {
        assert_eq!(report_cp(300, Some(300.0)), 100);
        assert_eq!(report_cp(-150, Some(300.0)), -50);
        assert_eq!(report_cp(-150, None), -150);
        let mate = super::super::MATE_VALUE - 3;
        assert_eq!(report_cp(mate, Some(300.0)), mate);
    }

    #[test]
    fn test_scale_is_off_where_it_was_not_measured() {
        let mut game = GameState::new();
        game.setup_variant(crate::Variant::Chess);
        assert_eq!(win_half_eval(&game), None);
    }

    const REF_CLASSICAL: f64 = 206.07768305335262;
    const REF_ROOK_ENDING: f64 = 479.5050980004054;
}
