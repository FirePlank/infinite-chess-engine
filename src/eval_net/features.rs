//! Stage-A eval-net features: scalars the HCE's single pass already computes,
//! collected via the tracer plus one raw-inputs handoff. The same code builds
//! the vector for training export and for inference, so they cannot disagree;
//! `schema_hash` seals the layout into the weights file.

use crate::board::{PieceType, PlayerColor};
use crate::evaluation::base::EvaluationTracer;
use crate::game::GameState;

/// Bump whenever `feature_vector`'s layout or scaling changes, so stale weight
/// files are rejected at load instead of silently misreading features.
pub const SCHEMA_VERSION: u32 = 5;

pub const NUM_ROWS: usize = 13;
pub const NUM_FEATURES: usize = 121;
/// Leading rows recorded as one White-ahead value; the rest are (White, Black) pairs.
const SINGLE_ROWS: usize = 2;

/// Eval-term rows captured from `tracer.record` calls, by exact name.
pub const ROW_NAMES: [&str; NUM_ROWS] = [
    "Material (net)",
    "Complexity scale",
    "Pawn Advancement",
    "Threats: Pawn",
    "Threats: Minor",
    "Threats: Slider",
    "Global Tropism",
    "King: Pawn Storm",
    "Piece: Activity",
    "Piece: Bishop Pair",
    "King: Shelter",
    "King: Attack",
    "Pawn: King Pawn Tropism",
];

/// Side-to-move column, set to 1 in the perspective encoding.
const STM_COL: usize = 2 * NUM_ROWS - SINGLE_ROWS;
/// Per-side blocks: White's starts at `SIDE_COL`, Black's right after it.
const SIDE_COL: usize = STM_COL + 21;
const SIDE_LEN: usize = 38;

/// Raw scalars handed out of the eval's main pass. Pair fields are indexed
/// [0]=White, [1]=Black explicitly, never via `PlayerColor as usize`.
#[derive(Clone, Copy, Default, Debug)]
pub struct EvalNetInputs {
    pub phase: i32,
    pub spread: i32,
    pub pawn_span: i32,
    pub wall_count: i32,
    pub void_count: i32,
    pub slider_geometry_ctx: i32,
    pub leaper_geometry_ctx: i32,
    pub cloud_avg_spread: i32,
    pub counterplay: [i32; 2],
    pub bishops: [i32; 2],
    pub bishop_pair: [i32; 2],
    pub diag_sliders: [i32; 2],
    pub ortho_sliders: [i32; 2],
    pub threat_points: [i32; 2],
    pub queen_threat: [i32; 2],
    pub sliders_in_zone: [i32; 2],
    pub extra_attack_units: [i32; 2],
    pub attacking_tropism: [i32; 2],
    pub defensive_tropism: [i32; 2],
    pub storm_count: [i32; 2],
    pub attack_ready: [i32; 2],
    pub urgency: [i32; 2],
    /// Diagonal rays around that side's royals with no piece on them at all.
    pub ray_open: [i32; 2],
    pub ray_enemy_min_dist: [i32; 2],
    pub ray_enemy_value: [i32; 2],
    pub ray_cover: [i32; 2],
    /// Same four summaries over the orthogonal rays.
    pub ortho_open: [i32; 2],
    pub ortho_enemy_min_dist: [i32; 2],
    pub ortho_enemy_value: [i32; 2],
    pub ortho_cover: [i32; 2],
    pub ring_covered: [i32; 2],
    /// First royal's defender units at distance 1-2, 3-4, 5-7.
    pub defender_hist: [[i32; 3]; 2],
    /// Chebyshev distance between the first royals (255 when a side has none).
    pub king_dist: i32,
    /// Each side's first royal's distance to the piece-cloud centre.
    pub king_cloud_dist: [i32; 2],
    /// Units attacking / defending that side's first royal.
    pub royal_attackers: [i32; 2],
    pub royal_defenders: [i32; 2],
    /// Most advanced pawn's distance to promotion (100 when the side has none).
    pub promo_dist: [i32; 2],
    pub non_pawn_non_royal: [i32; 2],
    /// Per side: open king rays, enemy owns a queen-like piece, own pieces near the king.
    pub king_exposure: [[i32; 3]; 2],
}

/// King-exposure inputs appended after the perspective vector: (own, opponent) x
/// (open rays, enemy queen-like, near), each x16 as `--extra-pairs 22,23,24` trains them.
pub const KEXP_INPUTS: usize = 6;
pub const NET_INPUTS: usize = NUM_FEATURES + KEXP_INPUTS;
const KEXP_RADIUS: i64 = 3;
const KEXP_NEAR: i64 = 2;

fn queen_like(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Queen
            | PieceType::RoyalQueen
            | PieceType::Amazon
            | PieceType::Chancellor
            | PieceType::Archbishop
    )
}

/// King-exposure inputs from what the eval already gathers: the first royal's nearest
/// piece per ray, own pieces within 2 of it, and a queen-like bit per colour.
#[derive(Default)]
pub struct KingExposure {
    queen_like_bits: u32,
}

/// One royal's nearest piece per ray: (distance, value, colour, type).
pub type KingRays = [(i32, i32, PlayerColor, PieceType); 8];

/// Bit per queen-like piece type (Queen, RoyalQueen, Amazon, Chancellor, Archbishop).
const QUEEN_LIKE_MASK: u32 = (1 << PieceType::Queen as u32)
    | (1 << PieceType::RoyalQueen as u32)
    | (1 << PieceType::Amazon as u32)
    | (1 << PieceType::Chancellor as u32)
    | (1 << PieceType::Archbishop as u32);

impl KingExposure {
    #[inline(always)]
    pub fn add(&mut self, color: PlayerColor, pt: PieceType) {
        self.queen_like_bits |= ((QUEEN_LIKE_MASK >> pt as u32) & 1) << color as u32;
    }

    /// `rays[side]` is that side's first royal's nearest piece per ray (distance,
    /// value, colour, type), `near[side]` its own pieces within 2 squares.
    pub fn finish(
        &self,
        rays: [Option<&KingRays>; 2],
        near: [i32; 2],
    ) -> [[i32; 3]; 2] {
        let mut out = [[0; 3]; 2];
        for (side, us) in [(0usize, PlayerColor::White), (1, PlayerColor::Black)] {
            let Some(r) = rays[side] else { continue };
            let open = r
                .iter()
                .filter(|&&(d, _, c, _)| !(d <= KEXP_RADIUS as i32 && (c == us || c == PlayerColor::Neutral)))
                .count() as i32;
            let enemy_ql = (self.queen_like_bits >> us.opponent() as u32) & 1;
            out[side] = [open, enemy_ql as i32, near[side]];
        }
        out
    }
}

/// Reference definition the training sidecar was built from (slow: board walks).
pub fn king_exposure_reference(g: &GameState) -> [[i32; 3]; 2] {
    let mut out = [[0; 3]; 2];
    let ql = |side: PlayerColor| g.board.iter().any(|(_, _, p)| p.color() == side && queen_like(p.piece_type()));
    for (side, us) in [(0usize, PlayerColor::White), (1, PlayerColor::Black)] {
        let kings = if side == 0 { &g.white_royals } else { &g.black_royals };
        let Some(k) = kings.first() else { continue };
        let mut open = 0;
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let shut = g.spatial_indices.find_first_blocker(k.x, k.y, dx, dy).is_some_and(|(x, y, p)| {
                (x - k.x).abs().max((y - k.y).abs()) <= KEXP_RADIUS
                    && (p.color() == us || p.color() == PlayerColor::Neutral)
            });
            open += i32::from(!shut);
        }
        let near = g
            .board
            .iter()
            .filter(|(x, y, p)| {
                p.color() == us && (x - k.x).abs().max((y - k.y).abs()) <= KEXP_NEAR && !(*x == k.x && *y == k.y)
            })
            .count()
            .min(255) as i32;
        out[side] = [open, i32::from(ql(us.opponent())), near];
    }
    out
}

/// Schema of a net that also reads the king-exposure inputs.
pub fn net_schema_hash() -> u64 {
    schema_hash() ^ 0x4b45_5850_3232_3234
}

/// Pawn-structure scalars handed out of `evaluate_pawn_structure_traced`.
/// Indexed [0]=White, [1]=Black explicitly.
#[derive(Clone, Copy, Default, Debug)]
pub struct PawnNetInputs {
    /// Tapered doubled, candidate, connected, isolated, backward terms.
    pub terms: [[i32; 5]; 2],
    pub passers: [i32; 2],
    /// Nearest passer's distance to promotion (100 when none).
    pub passer_min_dist: [i32; 2],
}

/// Summarize one ray class (4 rays) into (open rays, nearest enemy distance,
/// clamped enemy value sum on rays, rays covered by a friendly at dist <= 2).
pub fn summarize_rays(
    rays: &[(i32, i32, PlayerColor, PieceType)],
    own: PlayerColor,
) -> (i32, i32, i32, i32) {
    let mut open = 0;
    let mut enemy_min = 64;
    let mut enemy_val = 0i32;
    let mut cover = 0;
    for &(dist, value, color, _pt) in rays {
        if dist == i32::MAX {
            open += 1;
            continue;
        }
        if color == own {
            if dist <= 2 {
                cover += 1;
            }
        } else if color != PlayerColor::Neutral {
            enemy_min = enemy_min.min(dist.min(64));
            enemy_val += value;
        }
    }
    (open, enemy_min, enemy_val.min(8000), cover)
}

/// Collects the feature sources during one untraced evaluation. `is_active` is
/// false on purpose: the pawn cache must stay engaged, exactly as in search.
#[derive(Default)]
pub struct FeatureCollector {
    pub rows: [(i32, i32); NUM_ROWS],
    pub inputs: EvalNetInputs,
    pub pawn: PawnNetInputs,
}

impl EvaluationTracer for FeatureCollector {
    const WANTS_INPUTS: bool = true;

    // Always inlined so each call site's literal `term` folds the match to one
    // store instead of a runtime string comparison chain.
    #[inline(always)]
    fn record(&mut self, term: &str, white: i32, black: i32) {
        let idx = match term {
            "Material (net)" => 0,
            "Complexity scale" => 1,
            "Pawn Advancement" => 2,
            "Threats: Pawn" => 3,
            "Threats: Minor" => 4,
            "Threats: Slider" => 5,
            "Global Tropism" => 6,
            "King: Pawn Storm" => 7,
            "Piece: Activity" => 8,
            "Piece: Bishop Pair" => 9,
            "King: Shelter" => 10,
            "King: Attack" => 11,
            "Pawn: King Pawn Tropism" => 12,
            _ => return,
        };
        self.rows[idx] = (white, black);
    }

    #[inline]
    fn is_active(&self) -> bool {
        false
    }

    #[inline]
    fn record_inputs(&mut self, inputs: &EvalNetInputs) {
        self.inputs = *inputs;
    }

    #[inline]
    fn record_pawn_inputs(&mut self, pawn: &PawnNetInputs) {
        self.pawn = *pawn;
    }
}

/// Centipawn-scale value: quartered and clamped so the whole vector fits a
/// small integer range the quantized first layer can digest.
#[inline]
fn cp(v: i32) -> i16 {
    (v / 4).clamp(-2047, 2047) as i16
}

#[inline]
fn ct(v: i32) -> i16 {
    v.clamp(0, 255) as i16
}

#[inline]
fn sg(v: i32) -> i16 {
    v.clamp(-255, 255) as i16
}

fn win_condition_code(wc: crate::game::WinCondition) -> i16 {
    use crate::game::WinCondition;
    match wc {
        WinCondition::Checkmate => 0,
        WinCondition::AllPiecesCaptured => 1,
        WinCondition::AllRoyalsCaptured => 2,
        _ => 3,
    }
}

/// The full Stage-A feature vector. Order and scaling are part of the schema:
/// any change here must bump `SCHEMA_VERSION`.
pub fn feature_vector(game: &GameState, fc: &FeatureCollector) -> [i16; NUM_FEATURES] {
    let mut v = [0i16; NUM_FEATURES];
    let mut i = 0usize;
    macro_rules! push {
        ($x:expr) => {{
            v[i] = $x;
            i += 1;
        }};
    }

    // Eval-term rows, cp-scaled.
    for &(w, _) in &fc.rows[..SINGLE_ROWS] {
        push!(cp(w));
    }
    for &(w, b) in &fc.rows[SINGLE_ROWS..] {
        push!(cp(w));
        push!(cp(b));
    }
    debug_assert_eq!(i, STM_COL);

    // Game-level scalars.
    push!(if game.turn == PlayerColor::White { 1 } else { -1 });
    push!(cp(game.material_score));
    push!(game.initial_phase.clamp(0, 255) as i16);
    push!(ct(game.white_piece_count as i32));
    push!(ct(game.black_piece_count as i32));
    push!(ct(game.white_pawn_count as i32));
    push!(ct(game.black_pawn_count as i32));
    push!(ct(game.white_royals.len() as i32));
    push!(ct(game.black_royals.len() as i32));
    push!(win_condition_code(game.game_rules.white_win_condition));
    push!(win_condition_code(game.game_rules.black_win_condition));
    // Saturates at the 1e15 border every unbounded preset uses, so any larger
    // encoding of "unbounded" is the same input and cannot move the eval.
    push!(ct((64 - crate::moves::get_world_size().leading_zeros() as i32).min(50)));

    // Raw eval-pass scalars.
    let n = &fc.inputs;
    push!(ct(n.phase));
    push!(ct(n.spread));
    push!(ct(n.pawn_span));
    push!(ct(n.wall_count));
    push!(ct(n.void_count));
    push!(ct(n.slider_geometry_ctx));
    push!(ct(n.leaper_geometry_ctx));
    push!(ct(n.cloud_avg_spread));
    push!(ct(n.king_dist));
    debug_assert_eq!(i, SIDE_COL);
    let p = &fc.pawn;
    for side in 0..2 {
        push!(ct(n.counterplay[side]));
        push!(ct(n.bishops[side]));
        push!(ct(n.bishop_pair[side]));
        push!(ct(n.diag_sliders[side]));
        push!(ct(n.ortho_sliders[side]));
        push!(ct(n.threat_points[side]));
        push!(ct(n.queen_threat[side]));
        push!(ct(n.sliders_in_zone[side]));
        push!(ct(n.extra_attack_units[side] / 10));
        push!(sg(n.attacking_tropism[side] / 8));
        push!(sg(n.defensive_tropism[side] / 8));
        push!(ct(n.storm_count[side]));
        push!(ct(n.attack_ready[side]));
        push!(ct(n.urgency[side]));
        push!(ct(n.ray_open[side]));
        push!(ct(n.ray_enemy_min_dist[side]));
        push!(cp(n.ray_enemy_value[side]));
        push!(ct(n.ray_cover[side]));
        push!(ct(n.ortho_open[side]));
        push!(ct(n.ortho_enemy_min_dist[side]));
        push!(cp(n.ortho_enemy_value[side]));
        push!(ct(n.ortho_cover[side]));
        push!(ct(n.ring_covered[side]));
        for h in 0..3 {
            push!(ct(n.defender_hist[side][h] / 10));
        }
        push!(ct(n.king_cloud_dist[side]));
        for t in 0..5 {
            push!(cp(p.terms[side][t]));
        }
        push!(ct(p.passers[side]));
        push!(ct(p.passer_min_dist[side]));
        push!(ct(n.royal_attackers[side] / 10));
        push!(ct(n.royal_defenders[side] / 10));
        push!(ct(n.promo_dist[side]));
        push!(ct(n.non_pawn_non_royal[side]));
    }

    debug_assert_eq!(i, NUM_FEATURES);
    v
}

/// Column pairs a perspective net reads as (side to move, opponent): the two-sided
/// term rows, piece/pawn/royal counts, win conditions, and the two side blocks.
const PAIRED_ROWS: usize = NUM_ROWS - SINGLE_ROWS;
const PAIR_COLS: [(usize, usize); PAIRED_ROWS + 4 + SIDE_LEN] = {
    let mut out = [(0, 0); PAIRED_ROWS + 4 + SIDE_LEN];
    let mut i = 0;
    while i < PAIRED_ROWS {
        out[i] = (SINGLE_ROWS + 2 * i, SINGLE_ROWS + 1 + 2 * i);
        i += 1;
    }
    let mut k = 0;
    while k < 4 {
        out[PAIRED_ROWS + k] = (STM_COL + 3 + 2 * k, STM_COL + 4 + 2 * k);
        k += 1;
    }
    let mut j = 0;
    while j < SIDE_LEN {
        out[PAIRED_ROWS + 4 + j] = (SIDE_COL + j, SIDE_COL + SIDE_LEN + j);
        j += 1;
    }
    out
};

/// White-ahead single values (net material and complexity rows, material score).
const NEGATE_COLS: [usize; 3] = [0, 1, STM_COL + 1];

/// Re-encodes a vector as (side to move, opponent), matching `to_perspective` in
/// `evalnet/train_eval_net.py`: a position and its colour mirror then read identically.
pub fn to_perspective(v: &mut [i16], black_to_move: bool) {
    if black_to_move {
        for &(a, b) in &PAIR_COLS {
            v.swap(a, b);
        }
        for &c in &NEGATE_COLS {
            v[c] = -v[c];
        }
    }
    v[STM_COL] = 1;
}

/// FNV-1a over the row names, feature count and schema version. Weight files
/// carry this hash; a mismatch disables the net instead of misreading inputs.
pub fn schema_hash() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    for name in ROW_NAMES {
        eat(name.as_bytes());
    }
    eat(&(NUM_FEATURES as u32).to_le_bytes());
    eat(&SCHEMA_VERSION.to_le_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::base;

    /// The piece-loop collector must reproduce the definition the net was trained on.
    #[test]
    fn king_exposure_matches_reference() {
        use crate::Variant;
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut checked = 0;
        for v in [Variant::Classical, Variant::CoaIP, Variant::Palace, Variant::Space, Variant::Core, Variant::Obstocean] {
            for game_no in 0..6 {
                let mut g = GameState::new();
                g.setup_position_from_icn(&format!("[Variant \"{}\"] {}", v.to_str(), v.starting_icn()));
                for _ in 0..(20 + 15 * game_no) {
                    let moves = g.get_pseudo_legal_moves();
                    let legal: Vec<_> = moves
                        .iter()
                        .copied()
                        .filter(|m| {
                            let u = g.make_move(m);
                            let ok = !g.is_move_illegal();
                            g.undo_move(m, u);
                            ok
                        })
                        .collect();
                    if legal.is_empty() {
                        break;
                    }
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let m = legal[(seed >> 33) as usize % legal.len()];
                    g.make_move(&m);
                    let mut fc = FeatureCollector::default();
                    base::evaluate_inner_traced(&g, &mut fc);
                    assert_eq!(fc.inputs.king_exposure, king_exposure_reference(&g), "{} after {:?}", v.to_str(), m);
                    checked += 1;
                }
            }
        }
        assert!(checked > 500);
    }

    #[test]
    fn feature_vector_is_full_and_deterministic() {
        let mut game = GameState::new();
        game.setup_position_from_icn(
            "w (8;q|1;q) K5,1|k5,8|Q4,4|r1,8|P2,2|P3,2|p2,7|N7,7|b6,6",
        );
        let mut fc1 = FeatureCollector::default();
        let s1 = base::evaluate_inner_traced(&game, &mut fc1);
        let mut fc2 = FeatureCollector::default();
        let s2 = base::evaluate_inner_traced(&game, &mut fc2);
        assert_eq!(s1, s2);
        assert_eq!(feature_vector(&game, &fc1), feature_vector(&game, &fc2));
        // Material row must be filled: it is recorded unconditionally.
        assert_eq!(fc1.rows[0].0, game.material_score);
    }

    #[test]
    fn collector_does_not_change_eval() {
        let mut game = GameState::new();
        game.setup_position_from_icn(crate::Variant::Classical.starting_icn());
        let mut fc = FeatureCollector::default();
        let traced = base::evaluate_inner_traced(&game, &mut fc);
        let plain = base::evaluate_inner_traced(&game, &mut base::NoTrace);
        assert_eq!(traced, plain);
    }

    #[test]
    fn schema_hash_is_stable() {
        assert_eq!(schema_hash(), schema_hash());
        assert_ne!(schema_hash(), 0);
    }
}
