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

    const timeoutMs = Number.parseInt(process.env.SPRT_CI_SERVER_TIMEOUT_MS || '180000', 10);
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try { child.kill('SIGTERM'); } catch (e) {}
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
  const games = Number.parseInt(process.env.SPRT_CI_GAMES || '200', 10) || 200;
  const concurrency = Number.parseInt(process.env.SPRT_CI_CONCURRENCY || '2', 10) || 2;
  const timeControl = process.env.SPRT_CI_TC || '5+0.05';
  const maxRuntimeMs = Number.parseInt(process.env.SPRT_CI_TIMEOUT_MS || '900000', 10); // 15 minutes

  const execPath = process.env.PUPPETEER_EXECUTABLE_PATH;
  const baseArgs = ['--no-sandbox', '--disable-setuid-sandbox'];
  const launchOptions = execPath
    ? { headless: 'new', executablePath: execPath, args: baseArgs }
    : { headless: 'new', args: baseArgs };

  const browser = await puppeteer.launch(launchOptions);
  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });

    // Wait for WASM modules to be ready
    await page.waitForFunction(
      () => typeof window.__sprt_is_ready === 'function' && window.__sprt_is_ready(),
      { timeout: 180000 },
    );

    // Configure a short SPRT run via the existing UI and click Run
    await page.evaluate((cfg) => {
      const byId = (id) => /** @type {HTMLInputElement|null} */ (document.getElementById(id));

      const preset = byId('sprtBoundsPreset');
      const mode = byId('sprtBoundsMode');
      const alpha = byId('sprtAlpha');
      const beta = byId('sprtBeta');
      const tc = byId('sprtTimeControl');
      const conc = byId('sprtConcurrency');
      const minGames = byId('sprtMinGames');
      const maxGames = byId('sprtMaxGames');
      const maxMoves = byId('sprtMaxMoves');
      const btn = document.getElementById('runSprt');

      if (preset) preset.value = 'all';
      if (mode) mode.value = 'gainer';
      if (alpha) alpha.value = '0.05';
      if (beta) beta.value = '0.05';
      if (tc) tc.value = cfg.timeControl;
      if (conc) conc.value = String(cfg.concurrency);
      if (minGames) minGames.value = String(cfg.games);
      if (maxGames) maxGames.value = String(cfg.games);
      if (maxMoves && !maxMoves.value) maxMoves.value = '200';

      if (btn && btn instanceof HTMLButtonElement) {
        btn.click();
      }
    }, { games, concurrency, timeControl });

    const start = Date.now();
    let lastSnapshot = null;

    // Poll until run finishes or we hit the CI timeout
    // eslint-disable-next-line no-constant-condition
    while (true) {
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

      if (!lastSnapshot || !lastSnapshot.status || !lastSnapshot.status.running) {
        break;
      }

      if (Date.now() - start > maxRuntimeMs) {
        throw new Error('[sprt-ci] SPRT run exceeded timeout of ' + maxRuntimeMs + ' ms');
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    const snap = lastSnapshot || { status: null, statusText: '', eloText: '', rawOutput: '' };
    const wins = snap.status && typeof snap.status.wins === 'number' ? snap.status.wins : 0;
    const losses = snap.status && typeof snap.status.losses === 'number' ? snap.status.losses : 0;
    const draws = snap.status && typeof snap.status.draws === 'number' ? snap.status.draws : 0;
    const totalGames = wins + losses + draws;

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
    const mt = snap.statusText && snap.statusText.match(/Status:\s*(.+)$/i);
    if (mt) verdict = mt[1].trim();

    const lines = (snap.rawOutput || '').split(/\r?\n/);
    const summaryLines = lines.slice(-15).filter((l) => l.trim().length > 0);

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
    await browser.close().catch(() => {});
  }
}

async function main() {
  console.log('[sprt-ci] Project root:', PROJECT_ROOT);

  if (!fs.existsSync(path.join(PROJECT_ROOT, 'pkg-old'))) {
    console.error('[sprt-ci] Expected OLD web engine at <repo>/pkg-old.');
    console.error('[sprt-ci] Build it from the previous commit with: wasm-pack build --target web --out-dir pkg-old');
    process.exit(1);
  }

  const { child, url } = await startWebSprtHelper();
  console.log('[sprt-ci] Web SPRT helper running at', url);

  let result;
  try {
    result = await runHeadlessSprt(url);
  } finally {
    try { child.kill('SIGTERM'); } catch (e) {}
  }

  fs.writeFileSync(RESULT_FILE, JSON.stringify(result, null, 2));
  console.log('[sprt-ci] Wrote CI result to', RESULT_FILE);
  console.log('[sprt-ci] Summary:', JSON.stringify(result));
}

main().catch((err) => {
  console.error('[sprt-ci] Fatal error:', err && err.stack ? err.stack : String(err));
  process.exit(1);
});
