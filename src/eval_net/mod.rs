//! Stage-A hybrid evaluation net: a small quantized MLP over scalars the HCE
//! already computes, adding a capped residual to the generic eval.

pub mod features;
pub mod variant_features;

pub use features::{
    EvalNetInputs, FeatureCollector, NUM_FEATURES, PawnNetInputs, feature_vector, schema_hash,
    summarize_rays,
};

mod inference;
pub mod nnue;
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

/// The generic evaluator's side-to-move `score` plus the residual, blended with the NNUE
/// when one is embedded. Both nets read the same perspective inputs, built once.
#[inline]
pub fn generic_score(game: &crate::game::GameState, fc: &FeatureCollector, score: i32) -> i32 {
    let Some(net) = weights::EVAL_NET.as_ref() else {
        return score;
    };
    let sign = if game.turn == crate::board::PlayerColor::Black { -1 } else { 1 };
    let Some(nn) = nnue::NNUE.as_ref() else {
        return score + sign * residual_of(net, game, fc);
    };
    let x = net_inputs(game, fc, nnue::DENSE, true);
    let r = if net.n_in == nnue::DENSE && net.perspective {
        residual_from(net, game, &x)
    } else {
        residual_of(net, game, fc)
    };
    nn.blend(game, &x, score + sign * r)
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
pub fn variant_net(
    kind: crate::evaluation::eval_kind::EvalKind,
) -> Option<&'static weights::EvalNetWeights> {
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
    let r = inference::forward(net, &x[..layout.len()]).clamp(-layout.cap, layout.cap);
    if net.perspective && black_to_move {
        -r
    } else {
        r
    }
}

/// Capped residual of `net` for a position whose base-HCE features are in `fc`, White-ahead.
#[inline]
pub fn residual_of(
    net: &weights::EvalNetWeights,
    game: &crate::game::GameState,
    fc: &FeatureCollector,
) -> i32 {
    residual_from(net, game, &net_inputs(game, fc, net.n_in, net.perspective))
}

/// `residual_of` over inputs already built for `net`.
#[inline]
fn residual_from(net: &weights::EvalNetWeights, game: &crate::game::GameState, x: &[i16]) -> i32 {
    let black = game.turn == crate::board::PlayerColor::Black;
    // With many pieces on, the residual mostly cancels the HCE's over-reacting positional
    // swings and amplifying it overshoots; with few, it is independent signal worth sharpening.
    let pawns = (game.white_pawn_count + game.black_pawn_count) as i32;
    let nonpawn = (game.white_piece_count + game.black_piece_count) as i32 - pawns;
    let num = 20 - ((nonpawn - 12) / 3).clamp(0, 4);
    let r = (inference::forward(net, &x[..net.n_in]) * num / 16).clamp(-RESIDUAL_CAP, RESIDUAL_CAP);
    if net.perspective && black { -r } else { r }
}

/// A generic net's first `n_in` inputs (the widest layout is `TAC_NET_INPUTS`), side to
/// move first when `perspective`.
pub fn net_inputs(
    game: &crate::game::GameState,
    fc: &FeatureCollector,
    n_in: usize,
    perspective: bool,
) -> [i16; features::TAC_NET_INPUTS] {
    let base = feature_vector(game, fc);
    let black = game.turn == crate::board::PlayerColor::Black;
    let mut x = [0i16; features::TAC_NET_INPUTS];
    x[..NUM_FEATURES].copy_from_slice(&base);
    if perspective {
        features::to_perspective(&mut x[..NUM_FEATURES], black);
    }
    // A type net reads the piece-type imbalances next, side-to-move relative.
    let mut off = NUM_FEATURES;
    if n_in >= features::TYPE_NET_INPUTS {
        let sign = if perspective && black { -1 } else { 1 };
        for (k, &d) in features::type_count_diffs(game).iter().enumerate() {
            x[off + k] = (sign * d as i32 * 32).clamp(-2047, 2047) as i16;
        }
        off += features::TYPE_INPUTS;
    }
    // Extras run own side then opponent, each king exposure then slider rays, as far as
    // the net's width reaches (none for a 121-input net).
    let (ke, rays, tac) = (
        &fc.inputs.king_exposure,
        &fc.inputs.slider_rays,
        &fc.inputs.king_tactics,
    );
    let (own, opp) = if perspective && black {
        (1, 0)
    } else {
        (0, 1)
    };
    let per = n_in.saturating_sub(off) / 2;
    for j in 0..per {
        let side = |s: usize| match j {
            0..3 => ke[s][j],
            3..7 => rays[s][j - 3].min(255),
            _ => tac[s][j - 7].min(255),
        };
        x[off + j] = (side(own) * 16) as i16;
        x[off + per + j] = (side(opp) * 16) as i16;
    }
    x
}
