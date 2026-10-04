//! Stage-A hybrid evaluation net: a small quantized MLP over scalars the HCE
//! already computes, adding a capped residual to the generic eval.

pub mod features;
pub mod variant_features;

pub use features::{
    EvalNetInputs, FeatureCollector, NUM_FEATURES, PawnNetInputs, feature_vector, schema_hash,
    summarize_rays,
};

mod inference;
mod weights;

pub use inference::RESIDUAL_CAP;
#[cfg(any(test, feature = "bench_positions"))]
pub use inference::kernel_selftest;

/// True when trained weights are embedded and the runtime kill-switch
/// (`APEIRON_EVAL_NET=0`) is not set.
#[inline]
pub fn enabled() -> bool {
    !killed() && weights::EVAL_NET.is_some()
}

/// The runtime kill-switch, which turns off every net.
#[inline]
fn killed() -> bool {
    use once_cell::sync::Lazy;
    static KILLED: Lazy<bool> = Lazy::new(|| {
        std::env::var("APEIRON_EVAL_NET").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
    });
    *KILLED
}

/// Capped net residual in centipawns, White-ahead.
#[inline]
pub fn residual_white(game: &crate::game::GameState, fc: &FeatureCollector) -> i32 {
    let Some(net) = weights::EVAL_NET.as_ref() else {
        return 0;
    };
    residual_of(net, game, fc)
}

/// The net for a specialized evaluator, if one is embedded.
pub fn variant_net(kind: crate::evaluation::eval_kind::EvalKind) -> Option<&'static weights::EvalNetWeights> {
    use crate::evaluation::eval_kind::EvalKind;
    if killed() {
        return None;
    }
    match kind {
        EvalKind::Chess => weights::CHESS_NET.as_ref(),
        EvalKind::Obstocean => weights::OBSTOCEAN_NET.as_ref(),
        EvalKind::PawnHorde => weights::PAWN_HORDE_NET.as_ref(),
        EvalKind::Generic => None,
    }
}

/// Capped residual of a specialized evaluator's net over the features its own pass
/// wrote, White-ahead.
#[inline]
pub fn variant_residual(
    net: &weights::EvalNetWeights,
    layout: &variant_features::VariantLayout,
    feats: &variant_features::VariantFeatures,
    black_to_move: bool,
) -> i32 {
    let mut x = feats.x;
    if net.perspective {
        layout.to_perspective(&mut x[..layout.len()], black_to_move);
    }
    let r = inference::forward(net, &x[..layout.len()]).clamp(-RESIDUAL_CAP, RESIDUAL_CAP);
    if net.perspective && black_to_move { -r } else { r }
}

/// Capped residual of `net` for a position whose base-HCE features are in `fc`, White-ahead.
#[inline]
pub fn residual_of(net: &weights::EvalNetWeights, game: &crate::game::GameState, fc: &FeatureCollector) -> i32 {
    let base = feature_vector(game, fc);
    let black = game.turn == crate::board::PlayerColor::Black;
    let mut x = [0i16; features::TYPE_NET_INPUTS];
    x[..NUM_FEATURES].copy_from_slice(&base);
    if net.perspective {
        features::to_perspective(&mut x[..NUM_FEATURES], black);
    }
    // A type net reads the piece-type imbalances next, side-to-move relative.
    let mut off = NUM_FEATURES;
    if net.n_in == features::TYPE_NET_INPUTS {
        let sign = if net.perspective && black { -1 } else { 1 };
        for (k, &d) in features::type_count_diffs(game).iter().enumerate() {
            x[off + k] = (sign * d as i32 * 32).clamp(-2047, 2047) as i16;
        }
        off += features::TYPE_INPUTS;
    }
    // Extras run own side then opponent, each king exposure then slider rays, as far as
    // the net's width reaches (none for a 121-input net).
    let (ke, rays) = (&fc.inputs.king_exposure, &fc.inputs.slider_rays);
    let (own, opp) = if net.perspective && black { (1, 0) } else { (0, 1) };
    let per = net.n_in.saturating_sub(off) / 2;
    for j in 0..per {
        let side = |s: usize| if j < 3 { ke[s][j] } else { rays[s][j - 3].min(255) };
        x[off + j] = (side(own) * 16) as i16;
        x[off + per + j] = (side(opp) * 16) as i16;
    }
    let r = inference::forward(net, &x).clamp(-RESIDUAL_CAP, RESIDUAL_CAP);
    if net.perspective && black { -r } else { r }
}
