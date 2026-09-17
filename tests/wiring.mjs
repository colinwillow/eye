import { readFileSync } from 'node:fs';
import { check, report } from './harness.mjs';

// main.js is the one module with no unit tests, because it is all DOM and
// camera. What it CAN be checked for, cheaply and headlessly, is whether it
// and the markup still agree about what exists — a renamed id or a button
// wired to a case nobody handles is silently dead at runtime, and on a phone
// "that button does nothing" is indistinguishable from "the tracker is
// broken".
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const css = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');

const all = (src, re) => [...src.matchAll(re)].map(m => m[1]);
const uniq = a => [...new Set(a)];

const ids = uniq(all(html, /\bid="([^"]+)"/g));
const wanted = uniq([...all(main, /\$\('([^']+)'\)/g), ...all(main, /getElementById\('([^']+)'\)/g)]);

for (const id of wanted) check(`#${id} exists in index.html`, ids.includes(id));

// The other direction is a warning, not a failure — markup is allowed to carry
// ids for CSS alone. But every id main.js looks up has to be there.
check('main.js looks up a sensible number of elements', wanted.length >= 8, String(wanted.length));

// ── buttons ─────────────────────────────────────────────────────────────────
const modesInHtml = uniq(all(html, /data-mode="([^"]+)"/g));
const actsInHtml = uniq(all(html, /data-act="([^"]+)"/g));
const actsHandled = uniq(all(main, /case '([a-z]+)':/g));
const modesInCss = uniq(all(css, /\[data-mode="([^"]+)"\]/g));

for (const a of actsInHtml) check(`the "${a}" button is handled in main.js`, actsHandled.includes(a));
for (const a of actsHandled) check(`the "${a}" handler has a button`, actsInHtml.includes(a));
for (const m of modesInCss) check(`the CSS mode "${m}" is a real mode`, modesInHtml.includes(m));

check('there are three modes', modesInHtml.length === 3, modesInHtml.join(','));
check('the model toggle is on the bar', actsInHtml.includes('model'));

// The head pass asks for something different and looks identical. If nothing
// announces it, it gets done as another still pass and every head term ends
// up as dead as it was before the pass existed.
check('main.js announces the calibration pass', /showCalCard\(/.test(main));
check('  and the dot is drawn with its pass', /drawCalDot\([^)]*state\.cal\.pass/.test(main));
check('face is the default mode in the markup', /<body[^>]*data-mode="face"/.test(html));

// ── module graph ────────────────────────────────────────────────────────────
// index.html loads exactly one entry point as a module. A plain <script> here
// is a syntax error on the first import, and the page is simply blank.
check('index.html loads main.js as a module', /<script type="module" src="src\/main\.js">/.test(html));
// Every static import must be a relative path. A bare specifier ('three',
// '@mediapipe/...') needs either an import map or a bundler, and the whole
// point here is that index.html opens and runs. Anchored to lines that
// actually begin an import — matching the word "from" anywhere caught prose.
{
  const srcs = ['main', 'tracker', 'features', 'filter', 'calibrate', 'draw', 'solve']
    .map(n => [n, readFileSync(new URL(`../src/${n}.js`, import.meta.url), 'utf8')]);
  for (const [name, src] of srcs) {
    const specs = all(src, /^\s*import[^'"]*from\s*['"]([^'"]+)['"]/gm);
    for (const spec of specs) {
      check(`${name}.js imports "${spec}" by relative path`, spec.startsWith('./') || spec.startsWith('../'));
    }
  }
  // The one absolute URL is the tracker bundle, and it is loaded dynamically
  // at runtime rather than statically, so nothing needs to resolve it at parse
  // time.
  const tracker = srcs.find(s => s[0] === 'tracker')[1];
  check('the MediaPipe bundle is a dynamic import', /await import\(\s*MP\.bundle\s*\)/.test(tracker));
  check('its version is pinned, not floating', !/@mediapipe\/tasks-vision@(latest|\^|~)/.test(tracker));
}

// ── the mobile basics ───────────────────────────────────────────────────────
// Each of these is a specific way the page fails on a phone and nowhere else.
check('the viewport covers the notch', /viewport-fit=cover/.test(html));
check('zoom is pinned', /user-scalable=no/.test(html));
check('the video is playsinline', /<video[^>]*playsinline/.test(html));
check('the video is muted', /<video[^>]*muted/.test(html));   // iOS will not autoplay otherwise
check('scrolling and pinching are off', /touch-action:\s*none/.test(css));
check('the bar clears the home indicator', /safe-area-inset-bottom/.test(css));
check('touch targets are at least 44px', /min-height:\s*4[4-9]px|min-height:\s*[5-9]\dpx/.test(css));

// Nothing may be drawn over a calibration dot — see tests/smoke.mjs, which
// checks it for real. This is the cheap version: the rule has to exist.
for (const id of ['hud', 'bar', 'camwrap', 'note'])
  check(`#${id} is hidden while calibrating`, new RegExp(`body\\.calibrating #${id}[,\\s]`).test(css));
check('the calibration intro card exists', ids.includes('calintro'));
check('main.js drives the calibrating class', /classList\.(add|toggle)\('calibrating'/.test(main));

// The camera cannot start without a tap, so there has to be something to tap.
check('there is a start button', ids.includes('start'));
check('boot waits for a click', /el\.start\.addEventListener\('click'/.test(main));

report('wiring');
