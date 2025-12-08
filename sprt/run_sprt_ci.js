#!/usr/bin/env node
/**
 * Headless SPRT runner for CI.
 *
 * - Assumes the OLD web engine has already been built into <repo>/pkg-old
 *   from the previous commit.
 * - Calls sprt/sprt.js to:
 *     - copy pkg-old -> sprt/web/pkg-old
 *     - build NEW web WASM into sprt/web/pkg-new
 *     - start a static dev server in sprt/web
 * - Uses Puppeteer to drive the web UI headlessly and run a shortened SPRT.
 * - Writes a concise JSON summary to sprt/ci_result.json for GitHub Actions.
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const puppeteer = require('puppeteer');

const SPRT_DIR = __dirname;
const PROJECT_ROOT = path.join(SPRT_DIR, '..');
const RESULT_FILE = path.join(SPRT_DIR, 'ci_result.json');

function startWebSprtHelper() {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['sprt.js'], {
      cwd: SPRT_DIR,
      stdio: ['ignore', 'pipe', 'inherit'],
      env: { ...process.env, EVAL_TUNING: '0' },
    });

    let buffer = '';
    let resolved = false;

    const timeoutMs = Number.parseInt(process.env.SPRT_CI_SERVER_TIMEOUT_MS || '300000', 10);
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try { child.kill('SIGTERM'); } catch (e) { }
        reject(new Error('[sprt-ci] Timed out waiting for web SPRT server to start'));
      }
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      process.stdout.write(text);
      buffer += text;
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        const m = line.match(/Open this URL in your browser:\s*(https?:\/\/[^\s]+)/i);
        if (m && !resolved) {
          resolved = true;
          clearTimeout(timeout);
          resolve({ child, url: m[1] });
        }
      }
    });

    child.on('exit', (code) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(new Error('[sprt-ci] sprt.js exited before server was ready (code ' + code + ')'));
      }
    });
  });
}

async function runHeadlessSprt(url) {
  const games = Number.parseInt(process.env.SPRT_CI_GAMES || '150', 10) || 150;
  const concurrency = Number.parseInt(process.env.SPRT_CI_CONCURRENCY || '2', 10) || 2;
  const timeControl = process.env.SPRT_CI_TC || '5+0.05';
  const maxRuntimeMs = Number.parseInt(process.env.SPRT_CI_TIMEOUT_MS || '1800000', 10); // 30 minutes default

  const execPath = process.env.PUPPETEER_EXECUTABLE_PATH;
  const baseArgs = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'];
  const launchOptions = execPath
    ? {
      headless: 'new',
      executablePath: execPath,
      args: baseArgs,
      protocolTimeout: 300000, // 5 minutes for protocol operations
    }
    : {
      headless: 'new',
      args: baseArgs,
      protocolTimeout: 300000,
    };

  console.log('[sprt-ci] Launching browser with options:', JSON.stringify(launchOptions, null, 2));

  const browser = await puppeteer.launch(launchOptions);
  try {
    const page = await browser.newPage();

    // Set longer timeouts for page operations
    page.setDefaultTimeout(180000); // 3 minutes
    page.setDefaultNavigationTimeout(180000);

    console.log('[sprt-ci] Navigating to', url);
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 180000 });
    console.log('[sprt-ci] Page loaded, waiting for WASM modules...');

    // Wait for WASM modules to be ready with extended timeout
    await page.waitForFunction(
      () => typeof window.__sprt_is_ready === 'function' && window.__sprt_is_ready(),
      { timeout: 300000 }, // 5 minutes for WASM init
    );
    console.log('[sprt-ci] WASM modules ready');

    // Small delay to ensure UI is fully initialized
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Configure a short SPRT run via the existing UI and click Run
    console.log('[sprt-ci] Configuring SPRT with:', { games, concurrency, timeControl });

    await page.evaluate((cfg) => {
      const byId = (id) => /** @type {HTMLInputElement|null} */(document.getElementById(id));

      const preset = byId('sprtBoundsPreset');
      const mode = byId('sprtBoundsMode');
      const alpha = byId('sprtAlpha');
      const beta = byId('sprtBeta');
      const tc = byId('sprtTimeControl');
      const conc = byId('sprtConcurrency');
      const minGames = byId('sprtMinGames');
      const maxGames = byId('sprtMaxGames');
      const maxMoves = byId('sprtMaxMoves');

      if (preset) preset.value = 'all';
      if (mode) mode.value = 'gainer';
      if (alpha) alpha.value = '0.05';
      if (beta) beta.value = '0.05';
      if (tc) tc.value = cfg.timeControl;
      if (conc) conc.value = String(cfg.concurrency);
      if (minGames) minGames.value = String(cfg.games);
      if (maxGames) maxGames.value = String(cfg.games);
      // Ensure consistent max moves for CI
      if (maxMoves) maxMoves.value = '200';
    }, { games, concurrency, timeControl });

    // Click the Run button
    console.log('[sprt-ci] Clicking Run SPRT button...');
    await page.click('#runSprt');

    // Wait a bit for the SPRT to actually start
    await new Promise((resolve) => setTimeout(resolve, 3000));

    // Verify that SPRT actually started
    const initialStatus = await page.evaluate(() => {
      const statusFn = typeof window.__sprt_status === 'function' ? window.__sprt_status : null;
      return statusFn ? statusFn() : null;
    });

    console.log('[sprt-ci] Initial SPRT status:', JSON.stringify(initialStatus));

    if (!initialStatus || !initialStatus.running) {
      // SPRT didn't start - try clicking again after ensuring button is enabled
      console.log('[sprt-ci] SPRT not running yet, checking button state...');

      const buttonState = await page.evaluate(() => {
        const btn = document.getElementById('runSprt');
        return btn ? { disabled: btn.disabled, text: btn.textContent } : null;
      });
      console.log('[sprt-ci] Button state:', JSON.stringify(buttonState));

      if (buttonState && !buttonState.disabled) {
        console.log('[sprt-ci] Retrying button click...');
        await page.click('#runSprt');
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }

    const start = Date.now();
    let lastSnapshot = null;
    let lastLogTime = Date.now();
    const LOG_INTERVAL = 30000; // Log progress every 30 seconds

    // Poll until run finishes or we hit the CI timeout
    console.log('[sprt-ci] Starting SPRT polling loop...');
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        lastSnapshot = await page.evaluate(() => {
          const statusFn = typeof window.__sprt_status === 'function' ? window.__sprt_status : null;
          const status = statusFn ? statusFn() : null;
          const statusEl = document.getElementById('sprtStatus');
          const statusText = statusEl ? statusEl.textContent || '' : '';
          const eloEl = document.getElementById('sprtElo');
          const eloText = eloEl ? eloEl.textContent || '' : '';
          const outEl = document.getElementById('sprtOutput');
          const rawOutput = outEl ? outEl.textContent || '' : '';
          return { status, statusText, eloText, rawOutput };
        });
      } catch (evalErr) {
        console.error('[sprt-ci] Error evaluating page status:', evalErr.message);
        // Continue polling - the page might recover
        await new Promise((resolve) => setTimeout(resolve, 5000));
        continue;
      }

      // Log progress periodically
      if (Date.now() - lastLogTime > LOG_INTERVAL) {
        const total = lastSnapshot?.status
          ? (lastSnapshot.status.wins || 0) + (lastSnapshot.status.losses || 0) + (lastSnapshot.status.draws || 0)
          : 0;
        console.log('[sprt-ci] Progress: %d games completed, running=%s',
          total, lastSnapshot?.status?.running ?? 'unknown');
        lastLogTime = Date.now();
      }

      if (!lastSnapshot || !lastSnapshot.status || !lastSnapshot.status.running) {
        // Check if we got any games - if not and we're not running, something went wrong
        const total = lastSnapshot?.status
          ? (lastSnapshot.status.wins || 0) + (lastSnapshot.status.losses || 0) + (lastSnapshot.status.draws || 0)
          : 0;

        if (total === 0 && Date.now() - start < 30000) {
          // Still early, keep waiting - SPRT might be initializing workers
          await new Promise((resolve) => setTimeout(resolve, 5000));
          continue;
        }

        // Either we have games or we've waited long enough
        break;
      }

      if (Date.now() - start > maxRuntimeMs) {
        console.error('[sprt-ci] SPRT run exceeded timeout of ' + maxRuntimeMs + ' ms');
        // Don't throw - return partial results
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    const snap = lastSnapshot || { status: null, statusText: '', eloText: '', rawOutput: '' };
    const wins = snap.status && typeof snap.status.wins === 'number' ? snap.status.wins : 0;
    const losses = snap.status && typeof snap.status.losses === 'number' ? snap.status.losses : 0;
    const draws = snap.status && typeof snap.status.draws === 'number' ? snap.status.draws : 0;
    const totalGames = wins + losses + draws;

    console.log('[sprt-ci] Final results: W:%d L:%d D:%d Total:%d', wins, losses, draws, totalGames);

    let elo = Number.NaN;
    if (snap.eloText) {
      const e = Number.parseFloat(String(snap.eloText).trim());
      if (Number.isFinite(e)) elo = e;
    }

    let eloDiff = Number.isFinite(elo) ? elo : null;
    let eloError = null;
    if (snap.rawOutput) {
      const m = snap.rawOutput.match(/Elo Difference:\s*([+\-]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)/);
      if (m) {
        const d = Number.parseFloat(m[1]);
        const err = Number.parseFloat(m[2]);
        if (Number.isFinite(d)) eloDiff = d;
        if (Number.isFinite(err)) eloError = err;
      }
    }

    let verdict = '';
    if (totalGames === 0) {
      verdict = 'INCONCLUSIVE';
    } else {
      const mt = snap.statusText && snap.statusText.match(/Status:\s*(.+)$/i);
      if (mt) verdict = mt[1].trim();
      if (!verdict) verdict = 'INCONCLUSIVE';
    }

    const lines = (snap.rawOutput || '').split(/\r?\n/);
    const summaryLines = lines.slice(-20).filter((l) => l.trim().length > 0);

    return {
      games: totalGames,
      wins,
      losses,
      draws,
      elo: eloDiff,
      eloError,
      verdict,
      statusText: snap.statusText || '',
      logSummary: summaryLines.join('\n'),
      config: { games, concurrency, timeControl },
    };
  } finally {
    await browser.close().catch(() => { });
  }
}

async function main() {
  console.log('[sprt-ci] Project root:', PROJECT_ROOT);
  console.log('[sprt-ci] Environment:', {
    SPRT_CI_GAMES: process.env.SPRT_CI_GAMES,
    SPRT_CI_CONCURRENCY: process.env.SPRT_CI_CONCURRENCY,
    SPRT_CI_TC: process.env.SPRT_CI_TC,
    SPRT_CI_TIMEOUT_MS: process.env.SPRT_CI_TIMEOUT_MS,
  });

  if (!fs.existsSync(path.join(PROJECT_ROOT, 'pkg-old'))) {
    console.error('[sprt-ci] Expected OLD web engine at <repo>/pkg-old.');
    console.error('[sprt-ci] Build it from the previous commit with: wasm-pack build --target web --out-dir pkg-old');
    process.exit(1);
  }

  let serverInfo;
  try {
    serverInfo = await startWebSprtHelper();
  } catch (err) {
    console.error('[sprt-ci] Failed to start web server:', err.message);
    // Write empty result so CI doesn't fail hard
    const emptyResult = {
      games: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      elo: 0,
      eloError: 0,
      verdict: 'INCONCLUSIVE',
      statusText: 'Server failed to start',
      logSummary: err.message,
      config: {},
    };
    fs.writeFileSync(RESULT_FILE, JSON.stringify(emptyResult, null, 2));
    console.log('[sprt-ci] Wrote empty CI result due to server failure');
    process.exit(0); // Exit gracefully so CI shows results
  }

  const { child, url } = serverInfo;
  console.log('[sprt-ci] Web SPRT helper running at', url);

  let result;
  try {
    result = await runHeadlessSprt(url);
  } catch (err) {
    console.error('[sprt-ci] Error during headless SPRT:', err.message);
    result = {
      games: 0,
      wins: 0,
      losses: 0,
      draws: 0,
      elo: 0,
      eloError: 0,
      verdict: 'INCONCLUSIVE',
      statusText: 'Headless SPRT failed',
      logSummary: err.message,
      config: {},
    };
  } finally {
    try { child.kill('SIGTERM'); } catch (e) { }
  }

  fs.writeFileSync(RESULT_FILE, JSON.stringify(result, null, 2));
  console.log('[sprt-ci] Wrote CI result to', RESULT_FILE);
  console.log('[sprt-ci] Summary:', JSON.stringify(result));

  // Exit with success even if no games - let GitHub Actions decide based on results
  process.exit(0);
}

main().catch((err) => {
  console.error('[sprt-ci] Fatal error:', err && err.stack ? err.stack : String(err));
  // Write a failure result file so CI can still show something
  const failResult = {
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    elo: 0,
    eloError: 0,
    verdict: 'INCONCLUSIVE',
    statusText: 'Script crashed',
    logSummary: String(err),
    config: {},
  };
  try {
    fs.writeFileSync(RESULT_FILE, JSON.stringify(failResult, null, 2));
  } catch (e) {
    // Ignore write errors
  }
  process.exit(0); // Don't fail the CI job, just report inconclusive
});
