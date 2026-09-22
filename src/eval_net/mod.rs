//! Stage-A hybrid evaluation net: a small quantized MLP over scalars the HCE
//! already computes, adding a capped residual to the generic eval. Feature
//! extraction always compiles (the training exporter needs it); inference and
//! weights only exist under the `eval_net` cargo feature.

pub mod features;

pub use features::{
    EvalNetInputs, FeatureCollector, NUM_FEATURES, PawnNetInputs, feature_vector, schema_hash,
    summarize_rays,
};

#[cfg(feature = "eval_net")]
mod inference;
#[cfg(feature = "eval_net")]
mod weights;

#[cfg(feature = "eval_net")]
pub use inference::RESIDUAL_CAP;

/// True when trained weights are embedded and the runtime kill-switch
/// (`APEIRON_EVAL_NET=0`) is not set.
#[cfg(feature = "eval_net")]
#[inline]
pub fn enabled() -> bool {
    use once_cell::sync::Lazy;
    static KILLED: Lazy<bool> = Lazy::new(|| {
        std::env::var("APEIRON_EVAL_NET").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
    });
    !*KILLED && weights::EVAL_NET.is_some()
}

/// Capped net residual in centipawns, White-ahead.
#[cfg(feature = "eval_net")]
#[inline]
pub fn residual_white(game: &crate::game::GameState, fc: &FeatureCollector) -> i32 {
    let Some(net) = weights::EVAL_NET.as_ref() else {
        return 0;
    };
    let x = feature_vector(game, fc);
    inference::forward(net, &x).clamp(-RESIDUAL_CAP, RESIDUAL_CAP)
}
