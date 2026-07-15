#!/usr/bin/env node
/**
 * Apeiron SPRT Testing Helper
 *
 * Usage:
 *   node sprt.js          - Run web-based SPRT (opens browser UI)
 *   node sprt.js --native - Run native iwasm-based SPRT (CLI mode)
 * 
 * Web mode:
 * - Copies root/pkg-old (your baseline engine snapshot) to sprt/web/pkg-old
 * - Builds the NEW web WASM (MT by default) into sprt/web/pkg-new
 * - Starts `npx serve .` in sprt/web so the browser UI can import both
 * 
 * Native mode:
 * - Uses iwasm for high-performance testing
 * - Runs entirely from the command line
 * - Faster than browser-based testing
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

const SPRT_DIR = __dirname;
const PROJECT_ROOT = path.join(SPRT_DIR, '..');

// Root-level packages
const ROOT_PKG_OLD = path.join(PROJECT_ROOT, 'pkg-old');

// Web UI directories
const WEB_DIR = path.join(SPRT_DIR, 'web');
const WEB_PKG_OLD_DIR = path.join(WEB_DIR, 'pkg-old');
const WEB_PKG_NEW_DIR = path.join(WEB_DIR, 'pkg-new');

// Check flags
const isNativeMode = process.argv.includes('--native');

function hasIwasm() {
    try {
        execSync('iwasm --version', { stdio: 'pipe' });
        return true;
    } catch {
        return false;
    }
}

function copyDirectory(src, dest) {
    if (!fs.existsSync(src)) {
        throw new Error(`Source directory does not exist: ${src}`);
    }
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDirectory(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

function rmDirIfExists(dir) {
    if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

function snapshotOldFromRoot() {
    if (!fs.existsSync(ROOT_PKG_OLD)) {
        console.error('[web-sprt] Error: expected OLD pkg-old directory at', ROOT_PKG_OLD);
        console.error('[web-sprt] Build your old reference version with: wasm-pack build --target web --out-dir pkg-old');
        process.exit(1);
    }

    if (!fs.existsSync(WEB_DIR)) {
        fs.mkdirSync(WEB_DIR, { recursive: true });
    }

    console.log('[web-sprt] Copying OLD pkg (root/pkg-old) to sprt/web/pkg-old');
    rmDirIfExists(WEB_PKG_OLD_DIR);
    copyDirectory(ROOT_PKG_OLD, WEB_PKG_OLD_DIR);
}

function buildNewWebPkg() {
    // MT (Lazy SMP) is the default build now — .cargo/config.toml carries the atomics/
    // build-std flags persistently, so a plain wasm-pack build produces the threaded engine.
    console.log('\n[web-sprt] Building NEW WASM (Multi-threaded / Lazy SMP is default)...');
    try {
        rmDirIfExists(WEB_PKG_NEW_DIR);

        let features = 'debug';
        if (process.env.EVAL_TUNING === '1' || process.env.EVAL_TUNING === 'true') {
            features += ',eval_tuning';
        }

        execSync('wasm-pack build --target web --out-dir sprt/web/pkg-new --features ' + features, {
            cwd: PROJECT_ROOT,
            stdio: 'inherit',
            env: process.env,
        });
    } catch (e) {
        console.error('[web-sprt] wasm-pack build failed:', e.message);
        process.exit(1);
    }
    if (!fs.existsSync(WEB_PKG_NEW_DIR)) {
        console.error('[web-sprt] Build finished but pkg-new is missing:', WEB_PKG_NEW_DIR);
        process.exit(1);
    }
}

function startServer() {
    console.log('[web-sprt] Starting dev server in sprt/web (npx serve .)');

    const child = spawn('npx', ['serve', '.'], {
        cwd: WEB_DIR,
        stdio: ['ignore', 'pipe', 'inherit'],
        shell: true,
    });

    let sawUrl = false;
    let buffer = '';

    if (child.stdout) {
        child.stdout.on('data', (chunk) => {
            buffer += chunk.toString();
            let idx;
            while ((idx = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, idx).trim();
                buffer = buffer.slice(idx + 1);

                if (!sawUrl) {
                    const m = line.match(/https?:\/\/(localhost|127\.0\.0\.1):(\d+)/i);
                    if (m) {
                        sawUrl = true;
                        const url = `http://${m[1]}:${m[2]}`;
                        console.log('[web-sprt] Open this URL in your browser: ' + url);
                    }
                }
            }
        });
    }

    child.on('exit', (code) => {
        process.exit(code ?? 0);
    });
}

function runNativeSprt() {
    console.log('[native-sprt] Starting native SPRT testing...');

    // Build WASI binary if needed
    const wasmPath = path.join(PROJECT_ROOT, 'target', 'wasm32-wasip1', 'release', 'spsa_engine.wasm');
    if (!fs.existsSync(wasmPath)) {
        console.log('[native-sprt] Building WASI engine...');
        try {
            execSync('cargo build --target wasm32-wasip1 --release --bin spsa_engine', {
                cwd: PROJECT_ROOT,
                stdio: 'inherit',
            });
        } catch (e) {
            console.error('[native-sprt] Build failed:', e.message);
            process.exit(1);
        }
    }

    // Forward remaining args to sprt-native.mjs
    const forwardArgs = process.argv.slice(2).filter(a => a !== '--native');

    // Use spawn to run the native SPRT script
    const child = spawn('node', ['sprt-native.mjs', ...forwardArgs], {
        cwd: SPRT_DIR,
        stdio: 'inherit',
    });

    child.on('exit', (code) => {
        process.exit(code ?? 0);
    });
}

(function main() {
    // Show help if requested
    if (process.argv.includes('--help') || process.argv.includes('-h')) {
        console.log(`
Apeiron SPRT Testing Helper

Usage:
  node sprt.js              Run web-based SPRT (opens browser UI)
  node sprt.js --native     Run native iwasm-based SPRT (CLI mode)

Web Mode:
  Opens a browser UI for running SPRT tests with visual feedback.
  Both engines are Multi-threaded (Lazy SMP) builds by default; pick the
  thread count per engine in the UI (default 1 = single-threaded).
  Requires a pkg-old directory with the baseline engine build:
    wasm-pack build --target web --out-dir pkg-old

Native Mode Options:
  --games N        Number of game pairs (default: 100)
  --tc TIME        Time control in seconds per move (default: 0.1)
  --concurrency N  Number of parallel workers (default: physical threads)
  --variant NAME   Variant to test (default: Classical)
  --elo0 N         H0 Elo bound (default: -5)
  --elo1 N         H1 Elo bound (default: 5)
  --help           Show this help message

Examples:
  node sprt.js                           # Web mode
  node sprt.js --native                  # Native mode, default settings
  node sprt.js --native --games 500      # Native mode, 500 game pairs
  node sprt.js --native --variant Core   # Test Core variant
`);
        process.exit(0);
    }

    if (isNativeMode) {
        // Check for iwasm
        if (!hasIwasm()) {
            console.error('[native-sprt] Error: iwasm not found in PATH');
            console.error('[native-sprt] Please install iwasm from https://github.com/bytecodealliance/wasm-micro-runtime');
            process.exit(1);
        }
        runNativeSprt();
    } else {
        // Web mode
        try {
            snapshotOldFromRoot();
            buildNewWebPkg();
            startServer();
        } catch (e) {
            console.error('[web-sprt] Fatal error:', e.message);
            process.exit(1);
        }
    }
})();
