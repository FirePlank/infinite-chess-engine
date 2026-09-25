// Interleaved A/B of harness builds inside one Node process.
//   node run.mjs search <depth> <pairs> <pkgA> [pkgB]   (NPS_ONLY=<variant> filters)
//   node run.mjs eval   <passes> <pairs> <pkgA> [pkgB]
// Prints each build's result set (must be one value, equal across builds for an
// identical change) and median wall time; the B/A ratio compares speed.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [mode, arg, pairsArg, ...pkgs] = process.argv.slice(2);
if (!['search', 'eval'].includes(mode) || pkgs.length === 0) {
  console.error('usage: node run.mjs <search|eval> <depth|passes> <pairs> <pkgA> [pkgB]');
  process.exit(2);
}
const only = process.env.NPS_ONLY ?? '';
// The MT build pulls in wasm-bindgen-rayon's worker helper, which reads the
// browser worker global `name` at import time. The pool itself is never started.
globalThis.name ??= 'main';

async function load(dir) {
  const mod = await import(pathToFileURL(path.resolve(dir, 'apeiron_wasm_harness.js')));
  await mod.default({ module_or_path: readFileSync(path.resolve(dir, 'apeiron_wasm_harness_bg.wasm')) });
  return mod;
}

const mods = [];
for (const p of pkgs) mods.push(await load(p));

const run = (m) => {
  const t0 = performance.now();
  const r = mode === 'search' ? m.search_nodes(Number(arg), only) : m.eval_sum(Number(arg));
  return [r, performance.now() - t0];
};

for (const [k, m] of mods.entries()) {
  if (!m.net_selftest()) throw new Error(`${pkgs[k]}: net kernel selftest failed`);
  run(m); // warm-up: tables, JIT tiers, corpus
}
const times = mods.map(() => []);
const results = mods.map(() => new Set());
for (let i = 0; i < Number(pairsArg); i++) {
  for (let k = 0; k < mods.length; k++) {
    const [r, t] = run(mods[k]);
    times[k].push(t);
    results[k].add(r);
  }
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
pkgs.forEach((p, k) => {
  console.log(`${p}: result=${[...results[k]].join(',')} median=${median(times[k]).toFixed(1)}ms`);
});
if (mods.length === 2) {
  const ratios = times[0].map((a, i) => times[1][i] / a);
  const wins = ratios.filter((r) => r < 1).length;
  const same = results[0].size === 1 && [...results[0]][0] === [...results[1]][0] && results[1].size === 1;
  console.log(`B/A median=${(median(times[1]) / median(times[0])).toFixed(4)} paired-median=${median(ratios).toFixed(4)} B faster ${wins}/${ratios.length} identical=${same}`);
}
