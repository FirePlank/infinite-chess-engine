use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use hydrochess_wasm::Variant;
use hydrochess_wasm::game::GameState;
use hydrochess_wasm::search;

#[cfg(not(feature = "search_tuning"))]
fn main() {
    eprintln!("Error: Feature 'search_tuning' is required.");
}

#[cfg(feature = "search_tuning")]
use hydrochess_wasm::search::params::{get_search_params_as_json, set_search_params_from_json};
#[cfg(feature = "search_tuning")]
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct TunableParam {
    name: &'static str,
    min: f64,
    max: f64,
}

impl TunableParam {
    fn new(name: &'static str, min: f64, max: f64) -> Self {
        Self { name, min, max }
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================
const ITERATIONS: usize = 3000;
const BENCH_DEPTH: usize = 9;
// SPSA Constants (Standard Spall)
const SPSA_A: f64 = 0.1; // Smaller learning rate for stability
const SPSA_C: f64 = 0.02; // Perturbation (2% of range)
const SPSA_ALPHA: f64 = 0.602; // Standard decay
const SPSA_GAMMA: f64 = 0.101; // Standard decay
const SPSA_A_PARAM: f64 = 100.0; // Stability constant (about 10% of iterations)

// Loss Clipping to prevent explosions
const MAX_NODES_PER_VARIANT: u64 = 300_000; // Cap single-variant cost

// Helpers to list all variants
fn all_variants() -> Vec<Variant> {
    vec![
        Variant::Classical,
        Variant::ConfinedClassical,
        Variant::ClassicalPlus,
        Variant::CoaIP,
        Variant::CoaIPHO,
        Variant::CoaIPRO,
        Variant::CoaIPNO,
        Variant::Palace,
        Variant::Pawndard,
        Variant::Core,
        Variant::Standarch,
        Variant::SpaceClassic,
        Variant::Space,
        Variant::Abundance,
        Variant::PawnHorde,
        Variant::Knightline,
        Variant::Obstocean,
        Variant::Chess,
    ]
}

#[cfg(feature = "search_tuning")]
fn get_tunable_params() -> Vec<TunableParam> {
    vec![
        // Razoring
        TunableParam::new("razoring_linear", 200.0, 600.0),
        TunableParam::new("razoring_quad", 100.0, 400.0),
        // NMP
        TunableParam::new("nmp_base", 100.0, 500.0),
        TunableParam::new("nmp_depth_mult", 10.0, 40.0),
        TunableParam::new("nmp_reduction_base", 1.0, 10.0),
        TunableParam::new("nmp_reduction_div", 1.0, 5.0),
        TunableParam::new("nmp_min_depth", 1.0, 4.0),
        // RFP
        TunableParam::new("rfp_max_depth", 6.0, 16.0),
        TunableParam::new("rfp_mult_tt", 50.0, 150.0),
        TunableParam::new("rfp_mult_no_tt", 20.0, 100.0),
        TunableParam::new("rfp_improving_mult", 2000.0, 3500.0),
        TunableParam::new("rfp_worsening_mult", 500.0, 1500.0),
        // ProbCut
        TunableParam::new("probcut_margin", 50.0, 300.0),
        TunableParam::new("probcut_improving", 50.0, 200.0),
        TunableParam::new("probcut_min_depth", 3.0, 8.0),
        TunableParam::new("probcut_depth_sub", 2.0, 6.0),
        TunableParam::new("probcut_divisor", 200.0, 500.0),
        TunableParam::new("low_depth_probcut_margin", 800.0, 1500.0),
        // LMP
        TunableParam::new("lmp_base", 2.0, 8.0),
        TunableParam::new("lmp_depth_mult", 0.0, 3.0),
        // LMR
        TunableParam::new("lmr_min_depth", 2.0, 6.0),
        TunableParam::new("lmr_min_moves", 2.0, 12.0),
        TunableParam::new("lmr_divisor", 2.0, 6.0),
        TunableParam::new("lmr_history_thresh", 1000.0, 5000.0),
        TunableParam::new("lmr_cutoff_thresh", 1.0, 4.0),
        TunableParam::new("lmr_tt_history_thresh", -2000.0, -500.0),
        // SEE
        TunableParam::new("see_capture_linear", 100.0, 500.0),
        TunableParam::new("see_capture_hist_div", 20.0, 80.0),
        TunableParam::new("see_quiet_quad", 5.0, 50.0),
        // History
        TunableParam::new("history_bonus_base", 100.0, 500.0),
        TunableParam::new("history_bonus_sub", 100.0, 500.0),
        // Misc
        TunableParam::new("delta_margin", 100.0, 400.0),
        TunableParam::new("iir_min_depth", 2.0, 8.0),
    ]
}

#[cfg(feature = "search_tuning")]
fn build_params_json(theta_norm: &HashMap<String, f64>, tunables: &[TunableParam]) -> String {
    let mut entries = Vec::new();
    for t in tunables {
        let norm_val = theta_norm.get(t.name).unwrap_or(&0.5);
        let raw_val = t.min + norm_val * (t.max - t.min);
        let int_val = raw_val.round() as i64;
        entries.push(format!("\"{}\": {}", t.name, int_val));
    }
    format!("{{ {} }}", entries.join(", "))
}

#[cfg(feature = "search_tuning")]
fn run_bench() -> f64 {
    let variants = all_variants();

    // Parallel benchmarking with node capping
    let total_nodes: u64 = variants
        .par_iter()
        .map(|&variant| {
            let mut game = GameState::new();
            game.setup_variant(variant);

            let result = search::get_best_move(
                &mut game,
                BENCH_DEPTH,
                u128::MAX,
                true, // silent
                false,
            );

            match result {
                Some((_, _, stats)) => {
                    // Soft clip: if nodes > max, we still want a gradient (slope) to minimize it,
                    // but we don't want it to dominate the sum.
                    // However, for pure stability, hard capping is safer.
                    stats.nodes.min(MAX_NODES_PER_VARIANT)
                }
                None => MAX_NODES_PER_VARIANT,
            }
        })
        .sum();

    total_nodes as f64
}

#[cfg(feature = "search_tuning")]
fn main() {
    // Configure thread pool for deep recursion
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .unwrap();

    #[cfg(debug_assertions)]
    {
        println!("⚠️  WARNING: Running in DEBUG mode. SPSA optimization will be extremely slow.");
        println!(
            "   For production tuning, use: cargo run --bin spsa_tuner --release --features search_tuning"
        );
        println!();
    }

    let tunables = get_tunable_params();
    println!("SPSA Tuner (Stable Node Minimization) Initializing...");
    println!(
        "  - Benchmarking {} variants @ Depth {}",
        all_variants().len(),
        BENCH_DEPTH
    );
    println!(
        "  - Caps loss at {} nodes/variant to prevent explosions",
        MAX_NODES_PER_VARIANT
    );

    let mut theta_norm: HashMap<String, f64> = HashMap::new();

    // 1. Initialize Theta
    let current_json = get_search_params_as_json();
    let current_map: HashMap<String, serde_json::Value> =
        serde_json::from_str(&current_json).unwrap_or_default();

    for t in &tunables {
        let mut val = (t.min + t.max) / 2.0;
        if let Some(v) = current_map.get(t.name) {
            if let Some(f) = v.as_f64() {
                val = f;
            } else if let Some(i) = v.as_i64() {
                val = i as f64;
            }
        }
        let norm = ((val - t.min) / (t.max - t.min)).clamp(0.0, 1.0);
        theta_norm.insert(t.name.to_string(), norm);
    }

    println!("Starting optimization...");

    let mut best_loss = f64::MAX;
    let mut best_theta_norm = theta_norm.clone();
    // Use smoothed loss for display
    let mut smoothed_loss = 0.0;

    for k in 0..ITERATIONS {
        let ak = SPSA_A / (k as f64 + 1.0 + SPSA_A_PARAM).powf(SPSA_ALPHA);
        let ck = SPSA_C / (k as f64 + 1.0).powf(SPSA_GAMMA);

        let mut delta: HashMap<String, f64> = HashMap::new();
        let mut theta_plus = theta_norm.clone();
        let mut theta_minus = theta_norm.clone();

        for t in &tunables {
            // Rademacher distribution (+-1)
            let sign = if rand::random::<u8>().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            delta.insert(t.name.to_string(), sign);
            let p_val = ck * sign;
            let curr = theta_norm[t.name];
            theta_plus.insert(t.name.to_string(), (curr + p_val).clamp(0.0, 1.0));
            theta_minus.insert(t.name.to_string(), (curr - p_val).clamp(0.0, 1.0));
        }

        // Evaluate both sides
        set_search_params_from_json(&build_params_json(&theta_plus, &tunables));
        let y_plus = run_bench();

        set_search_params_from_json(&build_params_json(&theta_minus, &tunables));
        let y_minus = run_bench();

        let loss_diff = y_plus - y_minus;
        let avg_loss = (y_plus + y_minus) / 2.0;

        // Update smoothed loss
        if k == 0 {
            smoothed_loss = avg_loss;
        } else {
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * avg_loss;
        }

        for t in &tunables {
            let sign = delta[t.name];
            // Standard SPSA gradient estimation
            // g_k = (y_plus - y_minus) / (2 * ck * delta)
            // But since delta is +-1, we can simplify division or just multiply by sign
            let gh = loss_diff / (2.0 * ck) * sign;

            let mut step = ak * gh;
            // Clamp step magnitude to max 1% of range per iteration for max stability
            step = step.clamp(-0.01, 0.01);

            let new_val = (theta_norm[t.name] - step).clamp(0.0, 1.0);
            theta_norm.insert(t.name.to_string(), new_val);
        }

        if avg_loss < best_loss {
            best_loss = avg_loss;
            best_theta_norm = theta_norm.clone();
            let avg_per_var = best_loss / (all_variants().len() as f64);
            println!(
                "Iter {}: New Best Loss {:.0} (Avg {:.0}/var)",
                k, best_loss, avg_per_var
            );

            let json = build_params_json(&best_theta_norm, &tunables);
            if let Ok(mut file) = File::create("spsa_best_params.json") {
                let _ = file.write_all(json.as_bytes());
            }
        } else if k % 10 == 0 {
            let avg_per_var = smoothed_loss / (all_variants().len() as f64);
            println!(
                "Iter {}: Smoothed Loss {:.0} (Avg {:.0}/var)",
                k, smoothed_loss, avg_per_var
            );
        }
    }

    println!("Done.");
}
