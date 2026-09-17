// Every suite, in one go. No dependencies — `npm test` works on a clean clone.
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';

const here = new URL('.', import.meta.url).pathname;
const suites = readdirSync(here)
  .filter(f => f.endsWith('.mjs') && !['run.mjs', 'harness.mjs', 'smoke.mjs'].includes(f))
  .sort();

let failed = 0;
for (const s of suites) {
  const r = spawnSync(process.execPath, [here + s], { stdio: 'inherit' });
  if (r.status !== 0) failed++;
}

// smoke.mjs is deliberately not in that list: it needs a browser and the
// vendored assets, and it takes half a minute. `npm run test:smoke`.

// A syntax gate on the browser-only module too. It has no unit tests of its
// own and a parse error there is a blank page, which on a phone is
// indistinguishable from the camera being refused.
for (const f of ['src/main.js', 'src/draw.js', 'src/tracker.js']) {
  const r = spawnSync(process.execPath, ['--check', new URL('../' + f, import.meta.url).pathname], { stdio: 'inherit' });
  if (r.status !== 0) { console.log(`x ${f} does not parse`); failed++; }
}

console.log(failed ? `\n${failed} suite(s) FAILED` : '\nall suites passed');
process.exit(failed ? 1 : 0);
