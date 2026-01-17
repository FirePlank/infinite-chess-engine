#!/usr/bin/env node
/**
 * SPSA (Simultaneous Perturbation Stochastic Approximation) Tuner
 * 
 * This script tunes HydroChess search parameters by:
 * 1. Building the engine with search_tuning feature
 * 2. Running games between θ+ and θ- parameter configurations
 * 3. Estimating gradients from win rate differences
 * 4. Updating parameters and repeating
 * 
 * Usage:
 *   node spsa.js [options]
 * 
 * Options:
 *   --iterations <n>     Number of SPSA iterations (default: 100)
 *   --games <n>          Games per side per iteration (default: 100)
 *   --tc <ms>            Time control per move in ms (default: 100)
 *   --checkpoint <n>     Save every N iterations (default: 10)
 *   --resume <file>      Resume from checkpoint JSON file
 *   --verbose            Print detailed progress
 */

import { spawn, execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'fs';
import puppeteer from 'puppeteer';

// Try to use native Node.js WASM execution for better performance
import { NativeWasmRunner, canUseNativeWasm } from './native_runner.mjs';

/**
 * Check if iwasm is available in PATH
 */
function hasIwasm() {
    try {
        execSync('iwasm --version', { stdio: 'pipe' });
        return true;
    } catch {
        return false;
    }
}

/**
 * Check if WASI-targeted WASM binary exists
 */
function hasWasiBinary() {
    const wasmPath = join(ROOT_DIR, 'target', 'wasm32-wasip1', 'release', 'spsa_engine.wasm');
    return existsSync(wasmPath);
}

import {
    SPSA_PARAMS,
    SPSA_HYPERPARAMS,
    getDefaultParams,
    generatePerturbation,
    applyPerturbation,
    computeGradient,
    updateParams,
    getLearningRate,
    getPerturbationSize,
    validateParams
} from './spsa_config.mjs';

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

// ============================================================================
// Command Line Parsing with yargs
// ============================================================================

function parseArgs() {
    const argv = yargs(hideBin(process.argv))
        .scriptName('spsa')
        .usage('$0 [options] [checkpoint.json]')
        .option('iterations', {
            alias: 'i',
            type: 'number',
            default: 100,
            description: 'Number of SPSA iterations'
        })
        .option('games', {
            alias: 'g',
            type: 'number',
            default: 100,
            description: 'Games per side per iteration'
        })
        .option('tc', {
            alias: 't',
            type: 'number',
            default: 100,
            description: 'Time control per move in ms'
        })
        .option('checkpoint', {
            alias: 'c',
            type: 'number',
            description: 'Save checkpoint every N iterations (default: 5% of total)'
        })
        .option('resume', {
            alias: 'r',
            type: 'string',
            description: 'Resume from specific checkpoint JSON file'
        })
        .option('fresh', {
            alias: 'f',
            type: 'boolean',
            default: false,
            description: 'Start fresh, ignore any existing checkpoints'
        })
        .option('concurrency', {
            type: 'number',
            default: 20,
            description: 'Number of parallel workers'
        })
        .option('verbose', {
            alias: 'v',
            type: 'boolean',
            default: false,
            description: 'Print detailed progress'
        })
        .option('apply', {
            type: 'string',
            description: 'Apply tuned params to params.rs (use "latest" or path to JSON)'
        })
        .option('revert', {
            type: 'boolean',
            default: false,
            description: 'Revert params.rs to default values from spsa_config.mjs'
        })
        .option('native', {
            type: 'boolean',
            default: false,
            description: 'Force native Node.js WASM execution (faster)'
        })
        .option('browser', {
            type: 'boolean',
            default: false,
            description: 'Force browser-based execution (Puppeteer)'
        })
        .option('variant', {
            type: 'string',
            default: 'Classical',
            description: 'Chess variant to use for testing'
        })
        .help()
        .alias('help', 'h')
        .example('$0', 'Auto-resume or fresh start')
        .example('$0 --fresh', 'Force fresh start')
        .example('$0 --native', 'Use fast native WASM execution')
        .example('$0 --iterations 1000 --games 500 --tc 200', 'Custom settings')
        .example('$0 --apply latest', 'Apply latest tuned params to params.rs')
        .example('$0 --apply checkpoints/spsa_50.json', 'Apply specific checkpoint')
        .example('$0 --revert', 'Revert params.rs to defaults')
        .parseSync();

    // Handle positional argument as resume file
    if (argv._.length > 0 && String(argv._[0]).endsWith('.json')) {
        argv.resume = String(argv._[0]);
    }

    // Build options object
    const options = {
        iterations: argv.iterations,
        games: argv.games,
        tc: argv.tc,
        checkpoint: argv.checkpoint,
        resume: argv.resume || null,
        autoResume: !argv.fresh && !argv.resume,
        fresh: argv.fresh,
        verbose: argv.verbose,
        concurrency: argv.concurrency,
        apply: argv.apply,
        revert: argv.revert,
        native: argv.native,
        browser: argv.browser,
        variant: argv.variant
    };

    // Dynamic default for checkpoint: 5% of total iterations, min 1
    if (options.checkpoint === undefined || options.checkpoint === null) {
        options.checkpoint = Math.max(1, Math.floor(options.iterations * 0.05));
    }

    return options;
}

// ============================================================================
// Apply/Revert Parameters to params.rs
// ============================================================================

const PARAMS_RS_PATH = join(ROOT_DIR, 'src', 'search', 'params.rs');

// Mapping from SPSA param names to DEFAULT_* constant names in params.rs
const PARAM_TO_CONST = {
    nmp_reduction: 'DEFAULT_NMP_REDUCTION',
    nmp_min_depth: 'DEFAULT_NMP_MIN_DEPTH',
    lmr_min_depth: 'DEFAULT_LMR_MIN_DEPTH',
    lmr_min_moves: 'DEFAULT_LMR_MIN_MOVES',
    lmr_divisor: 'DEFAULT_LMR_DIVISOR',
    hlp_max_depth: 'DEFAULT_HLP_MAX_DEPTH',
    hlp_min_moves: 'DEFAULT_HLP_MIN_MOVES',
    hlp_history_reduce: 'DEFAULT_HLP_HISTORY_REDUCE',
    hlp_history_leaf: 'DEFAULT_HLP_HISTORY_LEAF',
    aspiration_window: 'DEFAULT_ASPIRATION_WINDOW',
    aspiration_fail_mult: 'DEFAULT_ASPIRATION_FAIL_MULT',
    rfp_max_depth: 'DEFAULT_RFP_MAX_DEPTH',
    rfp_margin_per_depth: 'DEFAULT_RFP_MARGIN_PER_DEPTH',
    sort_hash: 'DEFAULT_SORT_HASH',
    sort_winning_capture: 'DEFAULT_SORT_WINNING_CAPTURE',
    sort_killer1: 'DEFAULT_SORT_KILLER1',
    sort_killer2: 'DEFAULT_SORT_KILLER2',
    sort_countermove: 'DEFAULT_SORT_COUNTERMOVE',
    see_winning_threshold: 'DEFAULT_SEE_WINNING_THRESHOLD',
    max_history: 'DEFAULT_MAX_HISTORY',
    history_bonus_base: 'DEFAULT_HISTORY_BONUS_BASE',
    history_bonus_sub: 'DEFAULT_HISTORY_BONUS_SUB',
    history_bonus_cap: 'DEFAULT_HISTORY_BONUS_CAP',
    delta_margin: 'DEFAULT_DELTA_MARGIN',
    // Array-based params need special handling
    lmp_threshold_1: 'DEFAULT_LMP_THRESHOLD[1]',
    lmp_threshold_2: 'DEFAULT_LMP_THRESHOLD[2]',
    lmp_threshold_3: 'DEFAULT_LMP_THRESHOLD[3]',
    lmp_threshold_4: 'DEFAULT_LMP_THRESHOLD[4]',
    futility_margin_1: 'DEFAULT_FUTILITY_MARGIN[1]',
    futility_margin_2: 'DEFAULT_FUTILITY_MARGIN[2]',
    futility_margin_3: 'DEFAULT_FUTILITY_MARGIN[3]',
};

/**
 * Apply tuned parameters to params.rs by replacing constant values
 */
function applyParams(paramsFile) {
    // Find the params file if "latest" or empty
    let jsonPath;
    if (!paramsFile || paramsFile === 'latest' || paramsFile === '') {
        // Try spsa_final.json first, then latest checkpoint
        const finalPath = join(__dirname, 'spsa_final.json');
        if (existsSync(finalPath)) {
            jsonPath = finalPath;
        } else {
            jsonPath = findLatestCheckpoint();
            if (!jsonPath) {
                console.error('❌ No tuned parameters found. Run SPSA first or specify a JSON file.');
                process.exit(1);
            }
        }
    } else {
        jsonPath = paramsFile;
        if (!existsSync(jsonPath)) {
            // Try relative to sprt directory
            jsonPath = join(__dirname, paramsFile);
            if (!existsSync(jsonPath)) {
                console.error(`❌ File not found: ${paramsFile}`);
                process.exit(1);
            }
        }
    }

    console.log(`📂 Loading parameters from: ${jsonPath}`);
    const data = JSON.parse(readFileSync(jsonPath, 'utf-8'));

    // Extract theta from checkpoint or use directly if it's a final file
    const theta = data.theta || data;

    if (!theta || Object.keys(theta).length === 0) {
        console.error('❌ No parameters found in JSON file');
        process.exit(1);
    }

    // Read params.rs
    if (!existsSync(PARAMS_RS_PATH)) {
        console.error(`❌ params.rs not found at: ${PARAMS_RS_PATH}`);
        process.exit(1);
    }
    let content = readFileSync(PARAMS_RS_PATH, 'utf-8');

    let changedCount = 0;

    // Collect array parameter values
    const lmpThresholds = { 1: null, 2: null, 3: null, 4: null };
    const futilityMargins = { 1: null, 2: null, 3: null };

    // First pass: collect array values and apply scalar params
    for (const [paramName, value] of Object.entries(theta)) {
        const constName = PARAM_TO_CONST[paramName];
        if (!constName) {
            console.warn(`  ⚠️  Unknown parameter: ${paramName}`);
            continue;
        }

        // Handle array-based params - collect values
        if (constName.startsWith('DEFAULT_LMP_THRESHOLD[')) {
            const idx = parseInt(constName.match(/\[(\d+)\]/)[1], 10);
            lmpThresholds[idx] = Math.round(value);
            continue;
        }
        if (constName.startsWith('DEFAULT_FUTILITY_MARGIN[')) {
            const idx = parseInt(constName.match(/\[(\d+)\]/)[1], 10);
            futilityMargins[idx] = Math.round(value);
            continue;
        }

        // Match patterns like: pub const DEFAULT_NAME: usize = VALUE;
        const regex = new RegExp(
            `(pub const ${constName}:\\s*(?:usize|i32|f64)\\s*=\\s*)(-?[\\d_]+(?:\\.\\d+)?)(\\s*;)`,
            'g'
        );

        const newValue = Number.isInteger(value) ? String(Math.round(value)) : value.toFixed(2);
        const newContent = content.replace(regex, `$1${newValue}$3`);

        if (newContent !== content) {
            console.log(`  ✓ ${constName}: ${newValue}`);
            content = newContent;
            changedCount++;
        }
    }

    // Apply LMP thresholds array if we have any values
    if (Object.values(lmpThresholds).some(v => v !== null)) {
        // Get current array values first
        const lmpMatch = content.match(/pub const DEFAULT_LMP_THRESHOLD:\s*\[usize;\s*5\]\s*=\s*\[([^\]]+)\]/);
        if (lmpMatch) {
            const currentValues = lmpMatch[1].split(',').map(v => parseInt(v.trim(), 10));
            const newValues = [
                currentValues[0], // Keep index 0 as-is
                lmpThresholds[1] !== null ? lmpThresholds[1] : currentValues[1],
                lmpThresholds[2] !== null ? lmpThresholds[2] : currentValues[2],
                lmpThresholds[3] !== null ? lmpThresholds[3] : currentValues[3],
                lmpThresholds[4] !== null ? lmpThresholds[4] : currentValues[4],
            ];
            const newArrayStr = `pub const DEFAULT_LMP_THRESHOLD: [usize; 5] = [${newValues.join(', ')}]`;
            content = content.replace(/pub const DEFAULT_LMP_THRESHOLD:\s*\[usize;\s*5\]\s*=\s*\[[^\]]+\]/, newArrayStr);
            console.log(`  ✓ DEFAULT_LMP_THRESHOLD: [${newValues.join(', ')}]`);
            changedCount++;
        }
    }

    // Apply futility margins array if we have any values
    if (Object.values(futilityMargins).some(v => v !== null)) {
        const futMatch = content.match(/pub const DEFAULT_FUTILITY_MARGIN:\s*\[i32;\s*4\]\s*=\s*\[([^\]]+)\]/);
        if (futMatch) {
            const currentValues = futMatch[1].split(',').map(v => parseInt(v.trim(), 10));
            const newValues = [
                currentValues[0], // Keep index 0 as-is
                futilityMargins[1] !== null ? futilityMargins[1] : currentValues[1],
                futilityMargins[2] !== null ? futilityMargins[2] : currentValues[2],
                futilityMargins[3] !== null ? futilityMargins[3] : currentValues[3],
            ];
            const newArrayStr = `pub const DEFAULT_FUTILITY_MARGIN: [i32; 4] = [${newValues.join(', ')}]`;
            content = content.replace(/pub const DEFAULT_FUTILITY_MARGIN:\s*\[i32;\s*4\]\s*=\s*\[[^\]]+\]/, newArrayStr);
            console.log(`  ✓ DEFAULT_FUTILITY_MARGIN: [${newValues.join(', ')}]`);
            changedCount++;
        }
    }

    if (changedCount === 0) {
        console.log('⚠️  No parameters were changed');
        return;
    }

    // Write back
    writeFileSync(PARAMS_RS_PATH, content, 'utf-8');
    console.log(`\n✅ Applied ${changedCount} parameters to params.rs`);
    console.log('🔨 Rebuild the engine to use the new parameters');
}

/**
 * Revert params.rs to default values from spsa_config.mjs
 */
function revertParams() {
    console.log('🔄 Reverting params.rs to default values...');

    if (!existsSync(PARAMS_RS_PATH)) {
        console.error(`❌ params.rs not found at: ${PARAMS_RS_PATH}`);
        process.exit(1);
    }
    let content = readFileSync(PARAMS_RS_PATH, 'utf-8');

    let changedCount = 0;

    // Apply default values from SPSA_PARAMS
    for (const [paramName, config] of Object.entries(SPSA_PARAMS)) {
        const constName = PARAM_TO_CONST[paramName];
        if (!constName || constName.includes('[')) continue;

        const defaultValue = config.default;
        const regex = new RegExp(
            `(pub const ${constName}:\\s*(?:usize|i32|f64)\\s*=\\s*)(-?[\\d_]+(?:\\.\\d+)?)(\\s*;)`,
            'g'
        );

        const newValue = Number.isInteger(defaultValue) ? String(defaultValue) : defaultValue.toFixed(2);
        const newContent = content.replace(regex, `$1${newValue}$3`);

        if (newContent !== content) {
            console.log(`  ✓ ${constName}: ${newValue}`);
            content = newContent;
            changedCount++;
        }
    }

    if (changedCount === 0) {
        console.log('⚠️  No parameters were changed (already at defaults)');
        return;
    }

    writeFileSync(PARAMS_RS_PATH, content, 'utf-8');
    console.log(`\n✅ Reverted ${changedCount} parameters to defaults`);
    console.log('🔨 Rebuild the engine to use the default parameters');
}

/**
 * Find the latest checkpoint file in the checkpoints directory
 */
function findLatestCheckpoint() {
    const checkpointDir = join(__dirname, 'checkpoints');
    if (!existsSync(checkpointDir)) return null;

    const files = readdirSync(checkpointDir)
        .filter(f => f.startsWith('spsa_') && f.endsWith('.json'))
        .map(f => {
            const match = f.match(/spsa_(\d+)\.json/);
            return match ? { file: f, num: parseInt(match[1], 10) } : null;
        })
        .filter(x => x !== null)
        .sort((a, b) => b.num - a.num);

    if (files.length === 0) return null;
    return join(checkpointDir, files[0].file);
}

// ============================================================================
// Build Engine with search_tuning Feature
// ============================================================================

/**
 * Build engine for WASI target (for iwasm native execution)
 */
function buildEngineWasi() {
    console.log('🔨 Building engine with search_tuning feature (target: wasm32-wasip1 for iwasm)...');

    try {
        execSync(
            'cargo build --target wasm32-wasip1 --release --bin spsa_engine --features search_tuning',
            { cwd: ROOT_DIR, stdio: 'inherit' }
        );
        console.log('✅ WASI build complete');
        return true;
    } catch (e) {
        console.error('❌ WASI build failed:', e.message);
        return false;
    }
}

/**
 * Build engine for web/browser target (wasm-pack)
 */
function buildEngineWeb() {
    const target = 'web';
    const outDir = 'sprt/web/pkg-spsa';

    console.log(`🔨 Building engine with search_tuning feature (target: ${target})...`);

    try {
        execSync(
            `wasm-pack build --target ${target} --release --out-name hydrochess --out-dir ${outDir} -- --features search_tuning`,
            { cwd: ROOT_DIR, stdio: 'inherit' }
        );
        console.log('✅ Build complete');
        return true;
    } catch (e) {
        console.error('❌ Build failed:', e.message);
        return false;
    }
}

function buildEngine(useNative = false) {
    if (useNative) {
        return buildEngineWasi();
    } else {
        return buildEngineWeb();
    }
}

// ============================================================================
// Game Runner using Puppeteer
// ============================================================================

class SPSAGameRunner {
    constructor(options) {
        this.options = options;
        this.browser = null;
        this.page = null;
        this.consecutiveTimeouts = 0;
    }

    async init() {
        console.log('🌐 Starting browser...');
        this.browser = await puppeteer.launch({
            headless: 'new',
            protocolTimeout: 0,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--allow-file-access-from-files',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        });
        this.page = await this.browser.newPage();
        this.page.setDefaultTimeout(0);

        // Forward ALL browser console messages to Node console for debugging
        this.page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            if (type === 'error') {
                console.error('  [browser error]', text);
            } else if (type === 'warning') {
                console.warn('  [browser warn]', text);
            } else if (type === 'log' || type === 'info') {
                // console.log('  [browser]', text);
            }
        });

        // Log page errors
        this.page.on('pageerror', err => {
            console.error('  [page error]', err.message);
        });

        // Start local server
        const serverUrl = `file://${join(__dirname, 'web', 'spsa.html')}`;
        await this.page.goto(serverUrl, { waitUntil: 'networkidle0' });

        // Wait for the SPSA runner to be registered
        await this.page.waitForFunction(() => typeof window.runSPSAGames === 'function', { timeout: 30000 });

        console.log('✅ Browser ready');
    }

    async forceClose() {
        // Force kill the browser process
        if (this.browser) {
            try {
                const browserProcess = this.browser.process();
                if (browserProcess) {
                    browserProcess.kill('SIGKILL');
                }
            } catch (e) {
                // Ignore errors during force kill
            }
            try {
                await this.browser.close();
            } catch (e) {
                // Ignore errors during close
            }
            this.browser = null;
            this.page = null;
        }
    }

    async close() {
        if (this.browser) {
            try {
                await this.browser.close();
            } catch (e) {
                // Ignore
            }
            this.browser = null;
            this.page = null;
        }
    }

    /**
     * Run games between θ+ and θ- configurations
     * Browser-side stall detection will return partial results if stuck
     * Returns { plusWins, minusWins, draws }
     */
    async runGames(thetaPlus, thetaMinus, numGames) {
        try {
            const results = await this.page.evaluate(async (plus, minus, games, tc, concurrency) => {
                return await window.runSPSAGames(plus, minus, games, tc, concurrency);
            }, thetaPlus, thetaMinus, numGames, this.options.tc, this.options.concurrency);

            return results;

        } catch (error) {
            console.error(`  ❌ Error during games: ${error.message}`);
            console.error(`  Stack: ${error.stack}`);

            await this.forceClose();
            await new Promise(r => setTimeout(r, 1000));
            await this.init();

            return { plusWins: 0, minusWins: 0, draws: 0, stalled: true };
        }
    }
}

// ============================================================================
// SPSA Main Loop
// ============================================================================

async function runSPSA(options) {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║               SPSA TUNER FOR HYDROCHESS                      ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log('');

    // Initialize parameters
    let theta = getDefaultParams();
    let startIteration = 1;
    let history = [];

    // Auto-resume from latest checkpoint (default behavior) unless --fresh
    if (!options.fresh && options.autoResume && !options.resume) {
        const latest = findLatestCheckpoint();
        if (latest) {
            options.resume = latest;
            console.log(`🔍 Auto-resume: found ${latest}`);
        } else {
            console.log('🔍 No checkpoints found, starting fresh');
        }
    } else if (options.fresh) {
        console.log('🆕 Starting fresh (--fresh flag)');
    }

    // Resume from checkpoint if specified
    if (options.resume && existsSync(options.resume)) {
        console.log(`📂 Resuming from ${options.resume}`);
        const checkpoint = JSON.parse(readFileSync(options.resume, 'utf-8'));
        theta = checkpoint.theta;
        startIteration = checkpoint.iteration + 1;
        history = checkpoint.history || [];
        console.log(`   Starting from iteration ${startIteration}`);
        if (options.verbose) {
            console.log(`   Current params:`, JSON.stringify(theta, null, 2));
        }
    }

    // Determine execution mode before building
    let useNative = false;
    if (options.browser) {
        console.log('🌐 Using browser-based execution (--browser flag)');
        useNative = false;
    } else if (options.native) {
        console.log('🚀 Using native iwasm execution (--native flag)');
        useNative = true;
    } else {
        // Default: prefer iwasm if available for maximum performance
        if (hasIwasm()) {
            console.log('🚀 iwasm detected - using native execution for maximum performance');
            useNative = true;
        } else {
            console.log('🌐 iwasm not found - using browser-based execution');
            console.log('   Tip: Install iwasm for faster execution');
            useNative = false;
        }
    }

    // Build engine with appropriate target
    if (!buildEngine(useNative)) {
        process.exit(1);
    }

    // Initialize game runner
    let runner;
    if (useNative) {
        runner = new NativeWasmRunner(options);
    } else {
        runner = new SPSAGameRunner(options);
    }

    try {
        await runner.init();

        // SPSA main loop
        for (let k = startIteration; k <= options.iterations; k++) {
            const startTime = Date.now();

            // Compute learning rates for this iteration
            const a_k = getLearningRate(k);
            const c_k = getPerturbationSize(k);

            // Generate perturbation vector
            const delta = generatePerturbation();

            // Create θ+ and θ- configurations
            const thetaPlus = applyPerturbation(theta, delta, c_k, +1);
            const thetaMinus = applyPerturbation(theta, delta, c_k, -1);

            console.log(`\n📊 Iteration ${k}/${options.iterations}`);
            console.log(`   a_k = ${a_k.toFixed(4)}, c_k = ${c_k.toFixed(4)}`);

            // Run games
            console.log(`🎮 Running ${options.games} games per side...`);
            const results = await runner.runGames(thetaPlus, thetaMinus, options.games);

            // Compute win rates based on COMPLETED games only (from θ+'s perspective)
            const totalGames = results.plusWins + results.minusWins + results.draws;

            // Skip this iteration if we got no usable data
            if (totalGames === 0 || results.stalled) {
                console.log('   ⚠️  No games completed this iteration, skipping update');
                continue;
            }

            // Log if we got partial results
            if (totalGames < options.games * 2) {
                console.log(`   ⚠️  Partial results: ${totalGames}/${options.games * 2} games completed`);
            }

            const plusWinRate = (results.plusWins + 0.5 * results.draws) / totalGames;
            const minusWinRate = (results.minusWins + 0.5 * results.draws) / totalGames;

            // Loss = 1 - winRate (we minimize loss)
            const lossPlus = 1 - plusWinRate;
            const lossMinus = 1 - minusWinRate;

            // Compute gradient estimate
            const gradient = computeGradient(delta, c_k, [lossPlus, lossMinus]);

            // Update parameters
            const oldTheta = { ...theta };
            const newTheta = updateParams(theta, gradient, a_k);
            theta = validateParams(newTheta);

            // Count how many params changed
            let changedCount = 0;
            const changedParams = [];
            for (const key of Object.keys(theta)) {
                if (theta[key] !== oldTheta[key]) {
                    changedCount++;
                    changedParams.push(`${key}: ${oldTheta[key]} → ${theta[key]}`);
                }
            }
            console.log(`   📈 Params changed: ${changedCount}/${Object.keys(theta).length}`);

            // Record history
            const elapsed = Date.now() - startTime;
            const eloEstimate = 400 * Math.log10((plusWinRate + 0.001) / (minusWinRate + 0.001));

            history.push({
                iteration: k,
                plusWinRate,
                minusWinRate,
                eloEstimate,
                elapsed
            });

            // Print summary
            console.log(`   θ+ win rate: ${(plusWinRate * 100).toFixed(1)}%`);
            console.log(`   θ- win rate: ${(minusWinRate * 100).toFixed(1)}%`);
            console.log(`   Estimated Elo diff: ${eloEstimate.toFixed(1)}`);
            console.log(`   Time: ${(elapsed / 1000).toFixed(1)}s`);

            // Checkpoint
            if (k % options.checkpoint === 0) {
                const checkpointDir = join(__dirname, 'checkpoints');
                if (!existsSync(checkpointDir)) {
                    mkdirSync(checkpointDir, { recursive: true });
                }

                const checkpointPath = join(checkpointDir, `spsa_${k}.json`);
                writeFileSync(checkpointPath, JSON.stringify({
                    iteration: k,
                    theta,
                    history,
                    timestamp: new Date().toISOString()
                }, null, 2));

                console.log(`💾 Checkpoint saved: ${checkpointPath}`);
            }
        }

        // Final results
        console.log('');
        console.log('╔══════════════════════════════════════════════════════════════╗');
        console.log('║                    TUNING COMPLETE                          ║');
        console.log('╚══════════════════════════════════════════════════════════════╝');
        console.log('');
        console.log('📊 Final Parameters:');
        console.log(JSON.stringify(theta, null, 2));

        // Save final results
        const finalPath = join(__dirname, 'spsa_final.json');
        writeFileSync(finalPath, JSON.stringify({
            theta,
            history,
            options,
            timestamp: new Date().toISOString()
        }, null, 2));
        console.log(`\n💾 Final results saved: ${finalPath}`);

        // Generate Rust constants for copy-paste
        console.log('\n📝 Copy-paste for params.rs:\n');
        for (const [name, value] of Object.entries(theta)) {
            const constName = `DEFAULT_${name.toUpperCase()}`;
            if (SPSA_PARAMS[name]) {
                console.log(`pub const ${constName}: ${typeof value === 'number' ? (Number.isInteger(value) ? 'i32' : 'f64') : 'i32'} = ${value};`);
            }
        }

    } finally {
        await runner.close();
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

const options = parseArgs();

// Handle --apply and --revert commands (exit after)
if (options.revert) {
    revertParams();
    process.exit(0);
}

if (options.apply !== undefined) {
    applyParams(options.apply);
    process.exit(0);
}

// Normal SPSA tuning
runSPSA(options).catch(e => {
    console.error('❌ Fatal error:', e);
    process.exit(1);
});
