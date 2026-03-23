#!/usr/bin/env node
/**
 * Native SPRT Testing Script
 * 
 * Uses iwasm for high-performance self-play testing.
 * Compares the current engine (NEW) against a baseline (OLD) using
 * Sequential Probability Ratio Test.
 * 
 * Usage:
 *   node sprt-native.mjs [options]
 * 
 * Options:
 *   --games N        Number of game pairs (default: 100)
 *   --tc TIME        Time control in seconds (default: 0.1)
 *   --concurrency N  Number of parallel workers (default: 8)
 *   --variant NAME   Variant to test (default: All variants)
 *   --elo0 N         H0 Elo bound (default: -5)
 *   --elo1 N         H1 Elo bound (default: 5)
 *   --alpha N        Type I error rate (default: 0.05)
 *   --beta N         Type II error rate (default: 0.05)
 */

import { NativeWasmRunner, canUseNativeWasm, getAllVariants } from './native_runner.mjs';
import { execSync } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';
import os from 'os';


const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

// SPRT calculation functions
function eloToScore(eloDiff) {
    return 1 / (1 + Math.pow(10, -eloDiff / 400));
}

function calculateBounds(alpha, beta) {
    const lower = Math.log(beta / (1 - alpha));
    const upper = Math.log((1 - beta) / alpha);
    return { lower, upper };
}

function calculateLLR(wins, losses, draws, elo0, elo1) {
    const total = wins + losses + draws;
    if (total === 0) return 0;

    const score = (wins + draws * 0.5) / total;
    const s0 = eloToScore(elo0);
    const s1 = eloToScore(elo1);
    const clampedScore = Math.max(0.001, Math.min(0.999, score));

    const llr = total * (
        clampedScore * Math.log(s1 / s0) +
        (1 - clampedScore) * Math.log((1 - s1) / (1 - s0))
    );
    return llr;
}

function estimateElo(wins, losses, draws) {
    const total = wins + losses + draws;
    if (total === 0) return { elo: 0, error: 0 };

    const score = (wins + draws * 0.5) / total;
    if (score <= 0) return { elo: -999, error: 0 };
    if (score >= 1) return { elo: 999, error: 0 };

    const elo = -400 * Math.log10(1 / score - 1);

    const variance = (
        wins * Math.pow(1 - score, 2) +
        losses * Math.pow(0 - score, 2) +
        draws * Math.pow(0.5 - score, 2)
    ) / total;
    const stdDev = Math.sqrt(variance / total);
    const eloError = stdDev * 400 / (Math.log(10) * score * (1 - score));

    return { elo, error: Math.min(eloError, 200) };
}

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        games: 100,
        tc: '10+0.1', // Default time control
        concurrency: os.availableParallelism ? os.availableParallelism() : os.cpus().length,

        variant: null, // Default to all if null
        elo0: 0,
        elo1: 5,
        alpha: 0.05,
        beta: 0.05,
        maxMoves: 150,
        materialThreshold: 1500,
        build: true,
        oldPath: null,
        newPath: null,
    };

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        const next = args[i + 1];

        switch (arg) {
            case '--games':
                config.games = parseInt(next, 10) || config.games;
                i++;
                break;
            case '--tc':
                config.tc = next || config.tc;
                i++;
                break;
            case '--concurrency':
                config.concurrency = parseInt(next, 10) || config.concurrency;
                i++;
                break;
            case '--variant':
                config.variant = next || config.variant;
                i++;
                break;
            case '--elo0':
                config.elo0 = parseFloat(next) || config.elo0;
                i++;
                break;
            case '--elo1':
                config.elo1 = parseFloat(next) || config.elo1;
                i++;
                break;
            case '--alpha':
                config.alpha = parseFloat(next) || config.alpha;
                i++;
                break;
            case '--beta':
                config.beta = parseFloat(next) || config.beta;
                i++;
                break;
            case '--max-moves':
                config.maxMoves = parseInt(next, 10) || config.maxMoves;
                i++;
                break;
            case '--material':
                if (next === undefined || next === "" || next.startsWith('--')) {
                    config.materialThreshold = 0;
                } else {
                    const parsed = parseInt(next, 10);
                    config.materialThreshold = isNaN(parsed) ? 0 : parsed;
                    i++;
                }
                break;

            case '--old':
                config.oldPath = next;
                i++;
                break;
            case '--new':
                config.newPath = next;
                i++;
                break;
            case '--no-build':
                config.build = false;
                break;
            case '--help':
            case '-h':
                console.log(`
Native SPRT Testing Script

Usage: node sprt-native.mjs [options]

Options:
  --games N        Number of game pairs (default: 100)
  --tc TIME        Time control e.g. "10+0.1" (default: 10+0.1)
  --concurrency N  Number of parallel workers (default: physical threads)

  --variant NAME   Variant to test (default: All variants)
  --old PATH       Path to Old (Baseline) WASM binary
  --new PATH       Path to New (Test) WASM binary
  --elo0 N         H0 Elo bound (default: -5)
  --elo1 N         H1 Elo bound (default: 5)
  --alpha N        Type I error rate (default: 0.05)
  --beta N         Type II error rate (default: 0.05)
  --max-moves N    Maximum moves per game (default: 150)
  --material N     Material threshold for adjudication in cp (default: 1500)
  --no-build       Skip building the NEW WASM binary
  --help           Show this help message
`);
                process.exit(0);
        }
    }

    return config;
}

// Check if iwasm is available
function hasIwasm() {
    try {
        execSync('iwasm --version', { stdio: 'pipe' });
        return true;
    } catch {
        return false;
    }
}

// Build the WASI engine
function buildEngine() {
    console.log('🔨 Building NEW WASI engine...');
    try {
        execSync(
            'cargo build --target wasm32-wasip1 --release --bin spsa_engine --features search_tuning',
            { cwd: ROOT_DIR, stdio: 'inherit' }
        );
        console.log('✅ Build complete');
        return true;
    } catch (e) {
        console.error('❌ Build failed:', e.message);
        return false;
    }
}

async function main() {
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║           Native SPRT Testing (iwasm)                    ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log();

    const config = parseArgs();

    // Check for iwasm
    if (!hasIwasm()) {
        console.error('❌ Error: iwasm not found in PATH');
        console.error('   Please install iwasm from https://github.com/bytecodealliance/wasm-micro-runtime');
        process.exit(1);
    }

    console.log('✅ iwasm found');

    const defaultNewPath = join(ROOT_DIR, 'target', 'wasm32-wasip1', 'release', 'spsa_engine.wasm');
    const defaultOldPath = join(ROOT_DIR, 'target', 'wasm32-wasip1', 'release', 'spsa_engine_old.wasm');

    const newWasmPath = config.newPath || defaultNewPath;
    const oldWasmPath = config.oldPath || defaultOldPath;

    // Build NEW if needed (and if we are using the default path)
    if (config.build && newWasmPath === defaultNewPath) {
        if (!buildEngine()) {
            process.exit(1);
        }
    } else if (config.build) {
        console.warn(`⚠️  Build skipped because custom --new path provided: ${newWasmPath}`);
    }

    // Verify Output
    if (!existsSync(newWasmPath)) {
        console.error(`❌ Error: New WASM not found at ${newWasmPath}`);
        if (!config.build) console.error('   Run with build enabled or build manually.');
        process.exit(1);
    }

    // Verify Old (Baseline)
    if (!existsSync(oldWasmPath)) {
        console.error(`❌ Error: Old (Baseline) WASM not found at ${oldWasmPath}`);
        console.error('   To create a baseline:');
        console.error(`   1. Checkout your baseline commit`);
        console.error(`   2. Run: cargo build --target wasm32-wasip1 --release --bin spsa_engine`);
        console.error(`   3. Copy/Rename: spsa_engine.wasm -> spsa_engine_old.wasm`);
        console.error(`   4. Checkout your new code and run sprt-native.mjs`);
        process.exit(1);
    }

    // Initialize runner to access variants
    const runner = new NativeWasmRunner({
        concurrency: config.concurrency,
        tc: config.tc,
        variant: config.variant || 'Classical',
        newWasmPath: newWasmPath,
        oldWasmPath: oldWasmPath,
    });

    try {
        await runner.init();
    } catch (e) {
        console.error('❌ Failed to initialize runner:', e.message);
        process.exit(1);
    }

    // Determine variants to use
    let variantsToUse = [];
    if (!config.variant || config.variant.toLowerCase() === 'all') {
        variantsToUse = getAllVariants();
    } else {
        variantsToUse = config.variant.split(',').map(v => v.trim()).filter(Boolean);
    }

    console.log();
    console.log('📋 Configuration:');
    console.log(`   Games:       ${config.games} pairs (${config.games * 2} total)`);
    console.log(`   TC:          ${config.tc}`);
    console.log(`   Concurrency: ${config.concurrency} workers`);
    console.log(`   Variants:    ${variantsToUse.length === 1 ? variantsToUse[0] : 'All (' + variantsToUse.length + ' variants)'}`);
    console.log(`   SPRT bounds: [${config.elo0}, ${config.elo1}]`);
    console.log(`   Alpha/Beta:  ${config.alpha} / ${config.beta}`);
    console.log();

    const bounds = calculateBounds(config.alpha, config.beta);
    console.log(`   LLR bounds:  [${bounds.lower.toFixed(3)}, ${bounds.upper.toFixed(3)}]`);
    console.log();

    console.log('🏁 Starting SPRT test...');
    console.log();

    const startTime = Date.now();

    const results = await runner.runSprtGames(config.games, {
        timeControl: config.tc,
        maxMoves: config.maxMoves,
        materialThreshold: config.materialThreshold,
        variants: variantsToUse,
    });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    await runner.close();

    // Calculate final statistics
    const total = results.newWins + results.oldWins + results.draws;
    const llr = calculateLLR(results.newWins, results.oldWins, results.draws, config.elo0, config.elo1);
    const { elo, error } = estimateElo(results.newWins, results.oldWins, results.draws);
    const winRate = total > 0 ? ((results.newWins + results.draws * 0.5) / total * 100).toFixed(1) : '0.0';

    console.log();
    console.log('═══════════════════════════════════════════════════════════');
    console.log('                       SPRT RESULTS');
    console.log('═══════════════════════════════════════════════════════════');
    console.log();
    console.log(`   Total games:    ${total}`);
    console.log(`   New wins:       ${results.newWins}`);
    console.log(`   Old wins:       ${results.oldWins}`);
    console.log(`   Draws:          ${results.draws}`);
    console.log();
    console.log(`   Win rate:       ${winRate}%`);
    console.log(`   Elo:            ${elo >= 0 ? '+' : ''}${elo.toFixed(1)} ± ${error.toFixed(1)}`);
    console.log(`   LLR:            ${llr.toFixed(3)} [${bounds.lower.toFixed(3)}, ${bounds.upper.toFixed(3)}]`);
    console.log();

    // SPRT decision
    let decision = 'Inconclusive';
    if (llr >= bounds.upper) {
        decision = '✅ H1 ACCEPTED - New engine is likely stronger!';
    } else if (llr <= bounds.lower) {
        decision = '❌ H0 ACCEPTED - New engine is not stronger';
    } else {
        decision = '⏳ INCONCLUSIVE - Need more games';
    }

    console.log(`   Decision:      ${decision}`);
    console.log();
    console.log(`   Time elapsed:  ${elapsed}s`);
    console.log('═══════════════════════════════════════════════════════════');
}

main().catch(e => {
    console.error('Fatal error:', e);
    process.exit(1);
});
