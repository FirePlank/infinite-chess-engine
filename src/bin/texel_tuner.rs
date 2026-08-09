//! Texel-style tuner for evaluation parameters. The tuner reads a SPRT-generated
//! JSON file and optimizes the parameters in src/evaluation/base.rs to minimize
//! the errors.  
//! NOTE: The evaluation terms need to be expressed in linear combination of features
//! for this to work.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use apeiron::evaluation::{self, reset_eval_features, snapshot_eval_features};
use apeiron::game::GameState;
use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::Value;

const EVAL_BASE_RS_PATH: &str = "src/evaluation/base.rs";

#[derive(Parser, Debug)]
#[command(author, version, about = "Texel-style tuner for evaluation parameters")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run Texel tuner on a dataset of games to optimize evaluation parameters.
    Run {
        /// Path to the SPRT-generated JSON file.
        #[arg(long, default_value = "games.json")]
        games: PathBuf,

        /// Comma-separated parameter names or a preset to tune. Omit to tune all inferred params.
        #[arg(long)]
        params: Option<String>,

        /// Output JSON path for tuned parameter values.
        #[arg(long, default_value = "sprt/data/eval_params_tuned.json")]
        output: PathBuf,

        /// Number of tuning rounds.
        #[arg(long, default_value_t = 3)]
        rounds: usize,

        /// Scale factor for evaluation output before logistic activation.
        #[arg(long, default_value_t = 271.5856)]
        cp_scale: f64,

        /// Minimum step size fraction of the initial step.
        #[arg(long, default_value_t = 0.25)]
        min_step_fraction: f64,

        /// Print extra tuning information.
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Apply tuned values from JSON back into the Rust source constants.
    Apply {
        /// Optional result/checkpoint JSON path. Defaults to the latest checkpoint.
        #[arg(long, default_value = "sprt/data/eval_params_tuned.json")]
        input: PathBuf,
    },
    List {
        /// Parameter selection preset or comma-separated names: `all`, `piece-values`, or explicit names.
        #[arg(long, default_value = "all")]
        params: String,
    },
}


#[derive(Debug, Clone)]
struct Sample {
    result: f64,
    features: HashMap<String, f64>,
}

#[derive(Debug)]
struct ParamSpec {
    name: String,
    default_value: i64,
    step: i64,
    min: i64,
    max: i64,
}

#[derive(Debug, Serialize)]
struct Output {
    params: BTreeMap<String, i64>,
    neg_log_likelihood: f64,
    samples: usize,
    rounds: usize,
    timestamp: u64,
}

fn parse_samples_from_games(values: Vec<String>, path: &PathBuf) -> Result<Vec<Sample>, String> {
    let mut samples = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        samples.push(parse_sample_icn(&value, index + 1, path)?);
    }
    Ok(samples)
}

fn parse_sample_icn(icn: &str, index: usize, path: &PathBuf) -> Result<Sample, String> {
    let result = parse_result_from_icn(icn).ok_or_else(|| {
        format!("missing or invalid result tag in ICN line in {} at line {}", path.display(), index)
    })?;
    let features = compute_features_from_icn(icn)?;
    Ok(Sample { result, features })
}

fn parse_result_from_icn(icn: &str) -> Option<f64> {
    if let Some(tag) = parse_icn_tag(icn, "Result") {
        return match tag.as_str() {
            "1-0" => Some(1.0),
            "0-1" => Some(0.0),
            "1/2-1/2" => Some(0.5),
            _ => None,
        };
    }
    None
}

fn parse_icn_tag(icn: &str, tag: &str) -> Option<String> {
    let prefix = format!("[{} \"", tag);
    let start = icn.find(&prefix)? + prefix.len();
    let end = icn[start..].find('"')?;
    Some(icn[start..start + end].to_string())
}

fn compute_features_from_icn(icn: &str) -> Result<HashMap<String, f64>, String> {
    let mut game = GameState::new();
    game.setup_position_from_icn(icn);
    reset_eval_features();

    #[cfg(feature = "nnue")]
    {
        let _ = evaluation::evaluate(&game, None);
    }
    #[cfg(not(feature = "nnue"))]
    {
        let _ = evaluation::evaluate(&game);
    }

    let features_value = serde_json::to_value(snapshot_eval_features())
        .map_err(|e| format!("failed to serialize eval features: {}", e))?;
    extract_features(&features_value)
        .ok_or_else(|| "failed to extract features from ICN evaluation".to_string())
}

fn extract_features(value: &Value) -> Option<HashMap<String, f64>> {
    let obj = value.as_object()?;
    let mut features = HashMap::new();
    for (key, value) in obj {
        if let Some(v) = value_as_f64(value) {
            features.insert(key.clone(), v);
        }
    }
    Some(features)
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(num) => num.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

const PIECE_VALUE_NAMES: &[&str] = &[
    "pawn",
    "knight",
    "bishop",
    "rook",
    "guard",
    "centaur",
    "compound_bonus",
    "camel",
    "giraffe",
    "zebra",
    "knightrider",
    "hawk",
    "archbishop",
    "rose",
    "huygen",
    "chancellor_bonus",
];

fn parse_param_list(input: &str) -> HashSet<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn resolve_param_names(selector: &str, available: Option<&HashSet<String>>) -> Vec<String> {
    let available_names: HashSet<String> = available.cloned().unwrap_or_else(|| {
        PIECE_VALUE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    });
    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for token in selector
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        match token {
            "all" => {
                for name in available_names.iter() {
                    if seen.insert(name.clone()) {
                        selected.push(name.clone());
                    }
                }
            }
            "piece-values" | "material" => {
                for name in PIECE_VALUE_NAMES {
                    if available_names.contains(*name) && seen.insert((*name).to_string()) {
                        selected.push((*name).to_string());
                    }
                }
            }
            other => {
                if available_names.is_empty() || available_names.contains(other) || !available.is_some() {
                    if seen.insert(other.to_string()) {
                        selected.push(other.to_string());
                    }
                }
            }
        }
    }

    if selected.is_empty() {
        for name in available_names.iter() {
            if seen.insert(name.clone()) {
                selected.push(name.clone());
            }
        }
    }

    selected.sort();
    selected
}

fn build_tunable_specs(
    samples: &[Sample],
    requested: Option<&HashSet<String>>,
    base_text: &str,
) -> Vec<ParamSpec> {
    let mut feature_names: HashSet<String> = HashSet::new();
    for sample in samples {
        for key in sample.features.keys() {
            feature_names.insert(key.clone());
        }
    }

    let requested_names = if let Some(requested) = requested {
        requested.iter().cloned().collect::<Vec<_>>()
    } else {
        feature_names.iter().cloned().collect::<Vec<_>>()
    };

    let mut candidate_names: Vec<String> = if requested.is_some() {
        resolve_param_names(
            &requested_names.join(","),
            Some(&feature_names),
        )
    } else {
        feature_names.iter().cloned().collect::<Vec<_>>()
    };

    candidate_names.sort();

    let mut specs = Vec::new();
    for name in candidate_names {
        let default_value = infer_default_value(&name, base_text).unwrap_or(0);
        let (step, min, max) = guess_range(default_value);
        specs.push(ParamSpec {
            name,
            default_value,
            step,
            min,
            max,
        });
    }

    specs
}

fn infer_default_value(name: &str, base_text: &str) -> Option<i64> {
    for candidate in const_name_candidates(name) {
        if let Some(value) = extract_const_int(base_text, &candidate) {
            return Some(value);
        }
    }
    None
}

fn const_name_candidates(name: &str) -> Vec<String> {
    let normalized = to_const_name(name);
    let mut names = vec![
        format!("DEFAULT_{}", normalized),
        format!("DEFAULT_EVAL_{}", normalized),
        normalized.clone(),
    ];

    if !name.starts_with("mg_") && !name.starts_with("eg_") {
        names.push(format!("MG_{}", normalized));
        names.push(format!("EG_{}", normalized));
    }

    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for candidate in names {
        if seen.insert(candidate.clone()) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn to_const_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
}

fn extract_const_int(src: &str, const_name: &str) -> Option<i64> {
    for line in src.lines() {
        let trimmed = line.trim();
        if !(trimmed.starts_with("const ") || trimmed.starts_with("pub const ")) {
            continue;
        }
        if !trimmed.contains(const_name) {
            continue;
        }
        let left = if let Some(rest) = trimmed.strip_prefix("pub const ") {
            rest
        } else {
            trimmed.strip_prefix("const ")?
        };
        let left_name = left.split(':').next()?.trim();
        if left_name != const_name {
            continue;
        }

        let value_part = trimmed
            .split_once('=')?
            .1
            .split(';')
            .next()?
            .trim();
        if let Ok(parsed) = value_part.parse::<i64>() {
            return Some(parsed);
        }
    }
    None
}

fn parse_const_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let remainder = if let Some(rest) = trimmed.strip_prefix("pub const ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("const ") {
        rest
    } else {
        return None;
    };
    let (name_part, _) = remainder.split_once(':')?;
    Some(name_part.trim().to_string())
}

fn update_const_values_in_source(src: &str, updates: &HashMap<String, i64>) -> String {
    let mut output = String::new();
    for line in src.lines() {
        let mut replaced = false;
        let trimmed = line.trim();
        if let Some(const_name) = parse_const_name(trimmed) {
            if let Some(value) = updates.iter().find_map(|(param_name, value)| {
                const_name_candidates(param_name).iter().any(|candidate| *candidate == const_name).then_some(*value)
            }) {
                if let Some(eq_idx) = line.find('=') {
                    if let Some(semi_idx) = line.find(';') {
                        output.push_str(&line[..=eq_idx]);
                        output.push(' ');
                        output.push_str(&value.to_string());
                        output.push_str(&line[semi_idx..]);
                        output.push('\n');
                        replaced = true;
                    }
                }
            }
        }
        if !replaced {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}

fn read_json_params(path: &PathBuf) -> Result<HashMap<String, i64>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let root: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse JSON in {}: {}", path.display(), e))?;

    let obj = root
        .get("params")
        .and_then(Value::as_object)
        .or_else(|| root.as_object())
        .ok_or_else(|| format!("JSON in {} does not contain a parameter map", path.display()))?;

    let mut params = HashMap::new();
    for (name, value) in obj {
        let parsed = value.as_i64().or_else(|| value.as_u64().map(|n| n as i64)).or_else(|| {
            value.as_f64().map(|n| n.round() as i64)
        }).ok_or_else(|| format!("parameter '{}' in {} is not numeric", name, path.display()))?;
        params.insert(name.clone(), parsed);
    }
    Ok(params)
}

fn guess_range(default_value: i64) -> (i64, i64, i64) {
    let abs = default_value.abs().max(1);
    let step = (abs / 4).max(1);
    if default_value >= 0 {
        (step, default_value / 2 - 50, default_value * 2 + 50)
    } else {
        (step, default_value * 2 - 50, default_value / 2 + 50)
    }
}

fn evaluate_loss(params: &HashMap<String, i64>, samples: &[Sample], cp_scale: f64) -> f64 {
    let mut neg_ll = 0.0;
    for sample in samples {
        let mut cp_score = 0.0;
        for (name, value) in params {
            let feature_value = sample.features.get(name).copied().unwrap_or(0.0);
            cp_score += *value as f64 * feature_value;
        }
        let z = cp_score / cp_scale;
        let p = 1.0 / (1.0 + (-z).exp());
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        let r = sample.result;
        neg_ll += -(r * p.ln() + (1.0 - r) * (1.0 - p).ln());
    }
    neg_ll
}

struct TuneResult {
    improved: bool,
    value: i64,
    loss: f64,
}

fn tune_single_param(
    params: &HashMap<String, i64>,
    spec: &ParamSpec,
    samples: &[Sample],
    current_loss: f64,
    cp_scale: f64,
    min_step_fraction: f64,
    verbose: bool,
) -> TuneResult {
    let mut best_value = *params.get(&spec.name).unwrap_or(&spec.default_value);
    let mut best_loss = current_loss;
    let mut improved = false;

    let mut step = spec.step;
    let min_step = (spec.step as f64 * min_step_fraction).max(1.0).floor() as i64;
    let max_iterations = 8;

    if verbose {
        println!(
            "[texel_tuner] tuning {} (start={}, step={}, range=[{}, {}])",
            spec.name, best_value, step, spec.min, spec.max
        );
    }

    for _ in 0..max_iterations {
        if step < min_step {
            break;
        }

        let mut found_better = false;
        let mut candidates = Vec::new();
        let up = (best_value + step).min(spec.max);
        let down = (best_value - step).max(spec.min);
        if up != best_value {
            candidates.push(up);
        }
        if down != best_value && down != up {
            candidates.push(down);
        }

        for value in candidates {
            let mut test_params = params.clone();
            test_params.insert(spec.name.clone(), value);
            let loss = evaluate_loss(&test_params, samples, cp_scale);
            if loss + 1e-9 < best_loss {
                best_loss = loss;
                best_value = value;
                found_better = true;
            }
        }

        if found_better {
            improved = true;
            if verbose {
                println!("[texel_tuner]   improved {} -> {}", spec.name, best_value);
            }
        } else {
            step = (step / 2).max(1);
            if verbose {
                println!("[texel_tuner]   no improvement, halving step to {}", step);
            }
        }
    }

    if verbose && improved {
        let avg_loss = best_loss / samples.len() as f64;
        println!(
            "[texel_tuner]   {} final {} negLL={:.4} avg={:.6}",
            spec.name, best_value, best_loss, avg_loss
        );
    }

    TuneResult {
        improved,
        value: best_value,
        loss: best_loss,
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[texel_tuner] error: {}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Some(Commands::Run {
            games,
            params,
            output,
            rounds,
            cp_scale,
            min_step_fraction,
            verbose,
        }) => run_tuner(games, params, output, rounds, cp_scale, min_step_fraction, verbose),
        Some(Commands::Apply { input }) => apply_tuned_params(input),
        Some(Commands::List { params }) => list_tunable_params(&params),
        None => Err("no command specified; use --help for usage".to_string()),
    }
}

fn list_tunable_params(selector: &str) -> Result<(), String> {
    let base_text = fs::read_to_string(EVAL_BASE_RS_PATH)
        .map_err(|e| format!("failed to read {}: {}", EVAL_BASE_RS_PATH, e))?;
    let available = PIECE_VALUE_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<HashSet<_>>();
    let selected = resolve_param_names(selector, Some(&available));

    let mut specs = Vec::new();
    for name in selected {
        let default_value = infer_default_value(&name, &base_text).unwrap_or(0);
        let (step, min, max) = guess_range(default_value);
        specs.push((name, default_value, step, min, max));
    }

    for (name, default_value, step, min, max) in specs {
        println!("{:<28} default={:<6} step={:<6} range=[{}, {}]", name, default_value, step, min, max);
    }
    Ok(())
}

fn apply_tuned_params(input: PathBuf) -> Result<(), String> {
    let params = read_json_params(&input)?;
    if params.is_empty() {
        return Err(format!("no parameter values found in {}", input.display()));
    }

    let base_text = fs::read_to_string(EVAL_BASE_RS_PATH)
        .map_err(|e| format!("failed to read {}: {}", EVAL_BASE_RS_PATH, e))?;

    let mut updates = HashMap::new();
    for (name, value) in params {
        let candidates = const_name_candidates(&name);
        let matched = candidates.iter().find_map(|candidate| {
            extract_const_int(&base_text, candidate).map(|_| candidate.clone())
        });
        if let Some(target) = matched {
            updates.insert(target, value);
        }
    }

    if updates.is_empty() {
        return Err(format!("no matching eval constants found in {} for {}", EVAL_BASE_RS_PATH, input.display()));
    }

    let updated = update_const_values_in_source(&base_text, &updates);
    fs::write(EVAL_BASE_RS_PATH, updated)
        .map_err(|e| format!("failed to write {}: {}", EVAL_BASE_RS_PATH, e))?;
    println!("[texel_tuner] applied {} constants from {} to {}", updates.len(), input.display(), EVAL_BASE_RS_PATH);
    Ok(())
}

fn run_tuner(
    games_path: PathBuf,
    params: Option<String>,
    output_path: PathBuf,
    rounds: usize,
    cp_scale: f64,
    min_step_fraction: f64,
    verbose: bool,
) -> Result<(), String> {
    let content = std::fs::read_to_string(&games_path)
        .map_err(|e| format!("failed to read {}: {}", games_path.display(), e))?;
    let games: Vec<String> = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse JSON in {}: {}", games_path.display(), e))?;
    let samples = parse_samples_from_games(games, &games_path)?;

    if samples.is_empty() {
        return Err(format!(
            "no Texel samples found in {}",
            games_path.display()
        ));
    }

    let base_text = fs::read_to_string(EVAL_BASE_RS_PATH)
        .map_err(|e| format!("failed to read {}: {}", EVAL_BASE_RS_PATH, e))?;

    let requested_params = params
        .as_deref()
        .map(parse_param_list)
        .filter(|s| !s.is_empty());

    let tunable_specs = build_tunable_specs(&samples, requested_params.as_ref(), &base_text);
    if tunable_specs.is_empty() {
        return Err("no tunable parameters could be inferred".to_string());
    }

    let mut params: HashMap<String, i64> = tunable_specs
        .iter()
        .map(|spec| (spec.name.clone(), spec.default_value))
        .collect();

    let mut best_loss = evaluate_loss(&params, &samples, cp_scale);
    let baseline_avg = best_loss / samples.len() as f64;
    println!(
        "[texel_tuner] loaded {} samples from {}",
        samples.len(),
        games_path.display()
    );
    println!(
        "[texel_tuner] baseline neg-log-likelihood: {:.4} (avg={:.6})",
        best_loss, baseline_avg
    );

    for round in 0..rounds {
        println!("\n[texel_tuner] ===== round {}/{} =====", round + 1, rounds);
        let mut round_improved = false;

        for spec in &tunable_specs {
            let result = tune_single_param(&params, spec, &samples, best_loss, cp_scale, min_step_fraction, verbose);
            if result.improved {
                params.insert(spec.name.clone(), result.value);
                best_loss = result.loss;
                round_improved = true;
            }
        }

        let round_avg = best_loss / samples.len() as f64;
        println!(
            "[texel_tuner] round {} complete; negLL={:.4} (avg={:.6})",
            round + 1,
            best_loss,
            round_avg
        );

        if !round_improved {
            println!("[texel_tuner] no improvements in this round; stopping early.");
            break;
        }
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create output directory {}: {}", parent.display(), e))?;
    }

    let mut sorted_params = BTreeMap::new();
    for (key, value) in params.iter() {
        sorted_params.insert(key.clone(), *value);
    }

    let output = Output {
        params: sorted_params,
        neg_log_likelihood: best_loss,
        samples: samples.len(),
        rounds: rounds,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };

    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| format!("failed to serialize output JSON: {}", e))?;
    fs::write(&output_path, json)
        .map_err(|e| format!("failed to write {}: {}", output_path.display(), e))?;

    println!("[texel_tuner] tuning complete. wrote tuned parameters to {}", output_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_expands_presets_and_names() {
        let names = resolve_param_names("piece-values,knight", Some(&HashSet::from([
            "pawn".to_string(),
            "knight".to_string(),
            "bishop".to_string(),
            "rook".to_string(),
            "queen".to_string(),
        ])))
        .into_iter()
        .collect::<HashSet<_>>();

        assert!(names.contains("knight"));
        assert!(names.contains("pawn"));
        assert!(names.contains("bishop"));
        assert!(names.contains("rook"));
        assert!(!names.contains("queen"));
    }

    #[test]
    fn extracts_pub_const_values() {
        let src = r#"
            pub const DEFAULT_EVAL_PAWN: i32 = 100;
            const MG_BISHOP_PAIR_BONUS: i32 = 70;
        "#;

        assert_eq!(extract_const_int(src, "DEFAULT_EVAL_PAWN"), Some(100));
        assert_eq!(extract_const_int(src, "MG_BISHOP_PAIR_BONUS"), Some(70));
    }
}
