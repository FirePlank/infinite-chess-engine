/**
 * SPSA Configuration for HydroChess Search Parameter Tuning
 * 
 * This file defines all tunable search parameters with their:
 * - Default values (matching DEFAULT_* in params.rs)
 * - Min/max bounds for SPSA perturbation
 * - Step size (c_k coefficient) for SPSA gradient estimation
 * - Weight (importance) for parameter prioritization
 * 
 * Higher weight = more aggressive perturbation (use for important params).
 * Lower weight = more conservative tuning (for fragile/well-tuned params).
 */

export const SPSA_PARAMS = {
    // ============================================================================
    // NULL MOVE PRUNING - Critical for search speed
    // ============================================================================
    nmp_reduction: {
        default: 3,
        min: 2,
        max: 5,
        step: 1,
        weight: 0.8,
        description: "Null Move Pruning depth reduction"
    },
    nmp_min_depth: {
        default: 3,
        min: 2,
        max: 5,
        step: 1,
        weight: 0.6,
        description: "Minimum depth to apply NMP"
    },

    // ============================================================================
    // LATE MOVE REDUCTIONS - Major impact on tree shape
    // ============================================================================
    lmr_min_depth: {
        default: 3,
        min: 2,
        max: 5,
        step: 1,
        weight: 0.9,
        description: "Minimum depth for LMR"
    },
    lmr_min_moves: {
        default: 4,
        min: 2,
        max: 8,
        step: 1,
        weight: 0.8,
        description: "Minimum moves before LMR kicks in"
    },
    lmr_divisor: {
        default: 3,
        min: 2,
        max: 5,
        step: 1,
        weight: 0.7,
        description: "LMR formula divisor (higher = less reduction)"
    },

    // ============================================================================
    // HISTORY LEAF PRUNING - Aggressive pruning based on history
    // ============================================================================
    hlp_max_depth: {
        default: 3,
        min: 1,
        max: 5,
        step: 1,
        weight: 0.5,
        description: "Maximum depth for HLP"
    },
    hlp_min_moves: {
        default: 5,
        min: 3,
        max: 10,
        step: 1,
        weight: 0.5,
        description: "Minimum moves for HLP"
    },
    hlp_history_reduce: {
        default: 300,
        min: 100,
        max: 600,
        step: 50,
        weight: 0.6,
        description: "History threshold for extra reduction"
    },
    hlp_history_leaf: {
        default: 0,
        min: -200,
        max: 200,
        step: 50,
        weight: 0.5,
        description: "History threshold for leaf pruning"
    },

    // ============================================================================
    // LATE MOVE PRUNING - Per-depth move count thresholds
    // ============================================================================
    lmp_threshold_1: {
        default: 4,
        min: 2,
        max: 8,
        step: 1,
        weight: 0.7,
        description: "LMP threshold at depth 1"
    },
    lmp_threshold_2: {
        default: 8,
        min: 4,
        max: 16,
        step: 2,
        weight: 0.7,
        description: "LMP threshold at depth 2"
    },
    lmp_threshold_3: {
        default: 12,
        min: 6,
        max: 24,
        step: 2,
        weight: 0.7,
        description: "LMP threshold at depth 3"
    },
    lmp_threshold_4: {
        default: 16,
        min: 8,
        max: 32,
        step: 2,
        weight: 0.7,
        description: "LMP threshold at depth 4+"
    },

    // ============================================================================
    // ASPIRATION WINDOWS - Iterative deepening window size
    // ============================================================================
    aspiration_window: {
        default: 50,
        min: 20,
        max: 100,
        step: 10,
        weight: 0.8,
        description: "Initial aspiration window size"
    },
    aspiration_fail_mult: {
        default: 4,
        min: 2,
        max: 8,
        step: 1,
        weight: 0.5,
        description: "Window multiplier on fail high/low"
    },

    // ============================================================================
    // FUTILITY PRUNING - Per-depth margin thresholds
    // ============================================================================
    futility_margin_1: {
        default: 100,
        min: 50,
        max: 200,
        step: 20,
        weight: 0.7,
        description: "Futility margin at depth 1"
    },
    futility_margin_2: {
        default: 200,
        min: 100,
        max: 400,
        step: 30,
        weight: 0.7,
        description: "Futility margin at depth 2"
    },
    futility_margin_3: {
        default: 300,
        min: 150,
        max: 600,
        step: 40,
        weight: 0.7,
        description: "Futility margin at depth 3"
    },

    // ============================================================================
    // REVERSE FUTILITY PRUNING (Static Null Move Pruning)
    // ============================================================================
    rfp_max_depth: {
        default: 3,
        min: 2,
        max: 5,
        step: 1,
        weight: 0.6,
        description: "Max depth for reverse futility"
    },
    rfp_margin_per_depth: {
        default: 120,
        min: 60,
        max: 200,
        step: 20,
        weight: 0.7,
        description: "Reverse futility margin per depth"
    },

    // ============================================================================
    // MOVE ORDERING CONSTANTS - Critical for search efficiency
    // ============================================================================
    sort_hash: {
        default: 6000000,
        min: 4000000,
        max: 8000000,
        step: 500000,
        weight: 0.4,
        description: "Hash move ordering bonus"
    },
    sort_winning_capture: {
        default: 1000000,
        min: 500000,
        max: 2000000,
        step: 100000,
        weight: 0.5,
        description: "Winning capture ordering bonus"
    },
    sort_killer1: {
        default: 900000,
        min: 600000,
        max: 1500000,
        step: 100000,
        weight: 0.5,
        description: "First killer move bonus"
    },
    sort_killer2: {
        default: 800000,
        min: 500000,
        max: 1200000,
        step: 100000,
        weight: 0.5,
        description: "Second killer move bonus"
    },
    sort_countermove: {
        default: 600000,
        min: 300000,
        max: 900000,
        step: 100000,
        weight: 0.5,
        description: "Countermove bonus"
    },
    see_winning_threshold: {
        default: -90,
        min: -200,
        max: 0,
        step: 20,
        weight: 0.6,
        description: "SEE threshold for 'winning' captures"
    },

    // ============================================================================
    // HISTORY HEURISTIC TUNING
    // ============================================================================
    max_history: {
        default: 4000,
        min: 2000,
        max: 8000,
        step: 500,
        weight: 0.5,
        description: "Max absolute history score"
    },
    history_bonus_base: {
        default: 300,
        min: 100,
        max: 500,
        step: 50,
        weight: 0.4,
        description: "History bonus base (depth * base - sub)"
    },
    history_bonus_sub: {
        default: 250,
        min: 100,
        max: 400,
        step: 50,
        weight: 0.4,
        description: "History bonus subtraction"
    },
    history_bonus_cap: {
        default: 1536,
        min: 500,
        max: 3000,
        step: 200,
        weight: 0.4,
        description: "History bonus cap"
    },

    // ============================================================================
    // OTHER PARAMETERS
    // ============================================================================
    delta_margin: {
        default: 200,
        min: 100,
        max: 400,
        step: 30,
        weight: 0.5,
        description: "Quiescence search delta pruning margin"
    }
};

/**
 * SPSA hyperparameters (tuning the tuner!)
 * Classic SPSA uses:
 *   a_k = a / (A + k)^alpha   (learning rate)
 *   c_k = c / k^gamma         (perturbation size)
 */
export const SPSA_HYPERPARAMS = {
    // Learning rate schedule
    // Math: lossDiff ≈ 0.04 for 4% win rate diff
    // gradient = lossDiff / (2 * c_k) ≈ 0.04 / 2 = 0.02
    // update = a_k * gradient * step * weight
    // For step=1, weight=0.8, to get update ≈ 1.0:
    //   need a_k * 0.02 * 0.8 ≥ 0.5 → a_k ≥ 31
    // Using a=200 gives a_k(1) ≈ 80, which helps overcome noise
    a: 200.0,            // Learning rate scale (very high for integer params)
    A: 5,                // Stability constant
    alpha: 0.602,        // Learning rate decay exponent

    // Perturbation schedule  
    c: 1.0,              // Perturbation scale
    gamma: 0.101,        // Perturbation decay exponent

    // General tuning settings
    gamesPerIteration: 100,      // Games per side per SPSA iteration
    maxIterations: 10000,        // Stop after this many iterations
    checkpointEvery: 50,         // Save parameters every N iterations

    // Convergence detection
    minImprovement: 0.5,         // Stop if Elo gain < this for N iterations
    convergenceWindow: 100       // N iterations for convergence check
};

/**
 * Get all parameter names that are enabled for tuning
 */
export function getEnabledParams() {
    return Object.keys(SPSA_PARAMS);
}

/**
 * Get default parameter values as a flat object (for engine injection)
 */
export function getDefaultParams() {
    const params = {};
    for (const [name, config] of Object.entries(SPSA_PARAMS)) {
        params[name] = config.default;
    }
    return params;
}

/**
 * Validate a parameter object against bounds
 */
export function validateParams(params) {
    const validated = {};
    for (const [name, value] of Object.entries(params)) {
        if (SPSA_PARAMS[name]) {
            const config = SPSA_PARAMS[name];
            validated[name] = Math.max(config.min, Math.min(config.max, Math.round(value)));
        }
    }
    return validated;
}

/**
 * Generate a Bernoulli perturbation vector (±1 for each param)
 */
export function generatePerturbation() {
    const delta = {};
    for (const name of getEnabledParams()) {
        delta[name] = Math.random() < 0.5 ? -1 : 1;
    }
    return delta;
}

/**
 * Apply perturbation to parameters: θ + c_k * Δ * step * weight
 * Each param is perturbed by its own step size scaled by c_k
 */
export function applyPerturbation(params, delta, c_k, direction = 1) {
    const perturbed = {};
    for (const [name, value] of Object.entries(params)) {
        if (SPSA_PARAMS[name] && delta[name] !== undefined) {
            const config = SPSA_PARAMS[name];
            // Perturbation is c_k * delta * step * weight
            const perturbSize = config.step * config.weight;
            const newValue = value + direction * c_k * delta[name] * perturbSize;
            perturbed[name] = Math.max(config.min, Math.min(config.max, Math.round(newValue)));
        } else {
            perturbed[name] = value;
        }
    }
    return perturbed;
}

/**
 * Compute SPSA gradient estimate from win rate difference
 * 
 * Standard SPSA gradient: g_i = (L+ - L-) / (2 * c_k * Δ_i)
 * 
 * We divide by just (2 * c_k * Δ_i) - NOT by step.
 * This gives gradients in "normalized" units that can be multiplied
 * by step in the update phase.
 */
export function computeGradient(delta, c_k, lossPlusMinus) {
    const [lossPlus, lossMinus] = lossPlusMinus;
    const lossDiff = lossPlus - lossMinus;
    const gradient = {};

    for (const name of getEnabledParams()) {
        // Standard SPSA: divide by 2 * c_k * Δ_i only
        gradient[name] = lossDiff / (2 * c_k * delta[name]);
    }

    return gradient;
}

/**
 * Update parameters using gradient descent: θ - a_k * g_k * step * weight
 * 
 * The gradient is in normalized units, so we multiply by step * weight
 * to get the actual change in parameter units.
 */
export function updateParams(params, gradient, a_k) {
    const updated = {};
    for (const [name, value] of Object.entries(params)) {
        if (SPSA_PARAMS[name] && gradient[name] !== undefined) {
            const config = SPSA_PARAMS[name];
            // Scale update by step * weight to convert from normalized to param units
            const updateScale = config.step * config.weight;
            const newValue = value - a_k * gradient[name] * updateScale;
            updated[name] = Math.max(config.min, Math.min(config.max, Math.round(newValue)));
        } else {
            updated[name] = value;
        }
    }
    return updated;
}

/**
 * Compute learning rate a_k for iteration k
 */
export function getLearningRate(k, hyper = SPSA_HYPERPARAMS) {
    return hyper.a / Math.pow(hyper.A + k, hyper.alpha);
}

/**
 * Compute perturbation size c_k for iteration k
 */
export function getPerturbationSize(k, hyper = SPSA_HYPERPARAMS) {
    return hyper.c / Math.pow(k, hyper.gamma);
}
