//! Stage-A eval-net features: scalars the HCE's single pass already computes,
//! collected via the tracer plus one raw-inputs handoff. The same code builds
//! the vector for training export and for inference, so they cannot disagree;
//! `schema_hash` seals the layout into the weights file.

use crate::board::{PieceType, PlayerColor};
use crate::evaluation::base::EvaluationTracer;
use crate::game::GameState;

/// Bump whenever `feature_vector`'s layout or scaling changes, so stale weight
/// files are rejected at load instead of silently misreading features.
pub const SCHEMA_VERSION: u32 = 1;

pub const NUM_ROWS: usize = 15;
pub const NUM_FEATURES: usize = 99;

/// Eval-term rows captured from `tracer.record` calls, by exact name.
pub const ROW_NAMES: [&str; NUM_ROWS] = [
    "Material (net)",
    "Pawn Advancement",
    "Threats: Pawn",
    "Threats: Minor",
    "Threats: Slider",
    "Global Tropism",
    "King: Pawn Storm",
    "Complexity scale",
    "Piece: Activity",
    "Piece: Bishop Pair",
    "King: Shelter",
    "King: Attack",
    "Pawn: King Pawn Tropism",
    "Pawn: Core",
    "Pawn: Passed",
];

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
    pub cloud_count: i32,
    pub counterplay: [i32; 2],
    pub undeveloped: [i32; 2],
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
    /// Rays around that side's own first royal with no piece on them at all.
    pub ray_open: [i32; 2],
    pub ray_enemy_min_dist: [i32; 2],
    pub ray_enemy_value: [i32; 2],
    pub ray_cover: [i32; 2],
    pub ring_covered: [i32; 2],
    /// Units attacking / defending that side's first royal.
    pub royal_attackers: [i32; 2],
    pub royal_defenders: [i32; 2],
    /// Most advanced pawn's distance to promotion (100 when the side has none).
    pub promo_dist: [i32; 2],
    pub non_pawn_non_royal: [i32; 2],
}

/// Summarize one royal's 8-ray array into (open rays, nearest enemy distance,
/// clamped enemy value sum on rays, rays covered by a friendly at dist <= 2).
pub fn summarize_rays(
    rays: &[(i32, i32, PlayerColor, PieceType); 8],
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
}

impl EvaluationTracer for FeatureCollector {
    const WANTS_INPUTS: bool = true;

    #[inline]
    fn record(&mut self, term: &str, white: i32, black: i32) {
        let idx = match term {
            "Material (net)" => 0,
            "Pawn Advancement" => 1,
            "Threats: Pawn" => 2,
            "Threats: Minor" => 3,
            "Threats: Slider" => 4,
            "Global Tropism" => 5,
            "King: Pawn Storm" => 6,
            "Complexity scale" => 7,
            "Piece: Activity" => 8,
            "Piece: Bishop Pair" => 9,
            "King: Shelter" => 10,
            "King: Attack" => 11,
            "Pawn: King Pawn Tropism" => 12,
            "Pawn: Core" => 13,
            "Pawn: Passed" => 14,
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

    // Eval-term rows (White, Black), cp-scaled.
    for &(w, b) in &fc.rows {
        push!(cp(w));
        push!(cp(b));
    }

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
    push!(ct(64 - crate::moves::get_world_size().leading_zeros() as i32));

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
    push!(ct(n.cloud_count));
    for side in 0..2 {
        push!(ct(n.counterplay[side]));
        push!(ct(n.undeveloped[side]));
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
        push!(ct(n.ring_covered[side]));
        push!(ct(n.royal_attackers[side] / 10));
        push!(ct(n.royal_defenders[side] / 10));
        push!(ct(n.promo_dist[side]));
        push!(ct(n.non_pawn_non_royal[side]));
    }

    debug_assert_eq!(i, NUM_FEATURES);
    v
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
