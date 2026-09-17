// The real page, in a real browser, with the real tracker.
//
// Everything else in tests/ is pure maths. This is the only thing that checks
// the parts that only exist at runtime: that MediaPipe's options are accepted,
// that detectForVideo actually returns, that the canvases size themselves,
// that the calibration sequence completes and that a button does what its
// label says.
//
//   npm run vendor && npm run test:smoke
//
// It runs with ?local=1 so the tracker comes off disk — no network, and it is
// testing the exact files `npm run vendor` puts there.
//
// Chromium's fake camera is a spinning ball, not a face, so the REAL detector
// legitimately finds nothing. That half of the test is about the plumbing: the
// model loads, detect() runs every frame, nothing throws. The gaze half then
// stubs the detector with synthetic landmarks, because a face is the one thing
// this environment cannot supply.
import { spawn, execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { check, report } from './harness.mjs';

const root = new URL('..', import.meta.url).pathname;
const PORT = 8137;

if (!existsSync(root + 'vendor/tasks-vision/vision_bundle.mjs') || !existsSync(root + 'models/face_landmarker.task')) {
  console.log('smoke: SKIPPED — run `npm run vendor` first');
  process.exit(0);
}

// playwright may be a local dependency or only installed globally. Loaded
// through the global path it comes back as CommonJS behind a default export,
// so take whichever of the two shapes actually has a browser on it.
async function loadChromium() {
  for (const spec of ['playwright', null]) {
    try {
      const path = spec ?? `${execSync('npm root -g', { encoding: 'utf8' }).trim()}/playwright/index.js`;
      const m = await import(path);
      const c = m.chromium ?? m.default?.chromium;
      if (c) return c;
    } catch {}
  }
  return null;
}
const chromium = await loadChromium();
if (!chromium) {
  console.log('smoke: SKIPPED — playwright not installed');
  process.exit(0);
}

const server = spawn(process.execPath, ['-e', `
  const http = require('http'), fs = require('fs'), path = require('path');
  const types = { '.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript',
                  '.css':'text/css', '.json':'application/json', '.wasm':'application/wasm',
                  '.task':'application/octet-stream', '.webmanifest':'application/manifest+json' };
  http.createServer((req, res) => {
    const f = path.join(${JSON.stringify(root)}, decodeURIComponent(req.url.split('?')[0]).replace(/^\\/+/, '') || 'index.html');
    fs.readFile(f, (e, d) => e
      ? (res.writeHead(404), res.end('nope'))
      : (res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }), res.end(d)));
  }).listen(${PORT});
`], { stdio: 'ignore' });

const sleep = ms => new Promise(r => setTimeout(r, ms));
await sleep(500);

// The fake camera needs full Chromium, not the headless shell playwright
// reaches for by default — the shell has no media capture stack at all, and
// getUserMedia there fails with a NotFoundError that reads exactly like a
// phone refusing permission.
const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--no-sandbox'],
});

let ok = true;
try {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 780 }, permissions: ['camera'] });
  const page = await ctx.newPage();

  // MediaPipe logs its own INFO lines through console.error, and the fake
  // camera path makes Chromium grumble about software WebGL. Neither is a
  // failure, and treating them as one makes the check useless because it is
  // always red.
  const benign = /XNNPACK|TensorFlow Lite|GroupMarkerNotSet|swiftshader|WebGL|GL Driver|Automatic fallback/i;
  const errors = [];
  page.on('pageerror', e => errors.push('throw: ' + e));
  page.on('console', m => { if (m.type() === 'error' && !benign.test(m.text())) errors.push(m.text()); });
  page.on('requestfailed', r => errors.push('404: ' + r.url()));

  await page.goto(`http://localhost:${PORT}/?local=1&debug=1`);
  await page.click('#start');

  // ── the real tracker ──────────────────────────────────────────────────────
  await page.waitForFunction('window.__eye?.state.tracker.delegate', null, { timeout: 60000 });
  const delegate = await page.evaluate('window.__eye.state.tracker.delegate');
  check('the model loads and picks a delegate', !!delegate, String(delegate));
  check('the boot card gets out of the way', await page.locator('#boot.gone').count() === 1);

  await sleep(2500);

  const t = await page.evaluate(`({
    detFps: __eye.state.detFps,
    hasResult: !!__eye.state.tracker.result,
    landmarkArray: Array.isArray(__eye.state.tracker.result?.faceLandmarks),
    err: __eye.state.tracker.error ? String(__eye.state.tracker.error) : null,
    video: [document.getElementById('cam').videoWidth, document.getElementById('cam').videoHeight],
    hud: document.getElementById('hud').textContent,
  })`);
  check('the camera is delivering frames', t.video[0] > 0 && t.video[1] > 0, t.video.join('x'));
  check('detectForVideo is being called', t.detFps > 5, `${t.detFps} detections/s`);
  check('it returns a result of the expected shape', t.hasResult && t.landmarkArray);
  check('nothing threw inside the detector', t.err === null, t.err);
  check('no face in a spinning-ball feed is reported as such', /no face/.test(t.hud), t.hud);

  // ── canvases size themselves to the layout ────────────────────────────────
  const sizes = await page.evaluate(`
    ['overlay','stage','paint'].map(id => { const c = document.getElementById(id); return [id, c.width, c.height]; })
  `);
  for (const [id, w, h] of sizes) check(`#${id} has a backing store`, w > 0 && h > 0, `${w}x${h}`);

  // ── stub in a face ────────────────────────────────────────────────────────
  // The stub answers with whatever the calibration is currently asking for, so
  // the sequence sees a co-operative subject. Same eyeball model as
  // tests/calibrate.mjs, and the same reason for the sign: it is deliberately
  // the opposite of the raw mapping's assumption, so a fit that quietly leaned
  // on that assumption would come out mirrored here.
  await page.evaluate(`(async () => {
    const { makeFace } = await import('/tests/harness.mjs');
    window.__face = (sx, sy) => {
      const u = sx - 0.5, v = sy - 0.5;
      return makeFace({ gx: -(0.090 * u + 0.020 * u * v), gy: 0.055 * v + 0.012 * u * v });
    };
    window.__target = { x: 0.5, y: 0.5 };
    const tr = __eye.state.tracker;
    tr.detect = function () {
      const p = __eye.state.cal ? __eye.state.cal.point : window.__target;
      this.result = {
        faceLandmarks: [window.__face(p.x, p.y)],
        faceBlendshapes: [{ categories: [
          { categoryName: 'eyeBlinkLeft', score: 0.02 }, { categoryName: 'eyeBlinkRight', score: 0.02 } ] }],
      };
      this.lastVideoTime += 1 / 30;
      return this.result;
    };
  })()`);

  await sleep(600);
  check('a stubbed face is picked up', /(^|\\s)face/.test(await page.locator('#hud').textContent()));

  // The overlay has to actually put ink on the canvas. This is the cheapest
  // check that coverMap did not send every landmark off the edge of the box.
  const inked = await page.evaluate(`(() => {
    const c = document.getElementById('overlay');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 12) n++;
    return n;
  })()`);
  check('the overlay draws the face inside the canvas', inked > 200, `${inked} lit pixels`);

  // ── the calibration button runs a calibration ─────────────────────────────
  await page.click('[data-act="calibrate"]');
  check('calibrating switches to gaze mode', await page.evaluate('document.body.dataset.mode') === 'gaze');
  check('an intro card holds before the first dot', await page.locator('#calintro.show').count() === 1);

  // Nothing may sit on top of a calibration dot. The top-left dot is at 12%
  // of the screen, which is exactly where the HUD lives, and a dot you cannot
  // see records you looking at whatever is covering it.
  await sleep(2000);
  const clear = await page.evaluate(`(() => {
    const dot = __eye.state.cal ? __eye.state.cal.point : null;
    if (!dot) return { no: 'calibration is not running' };
    const x = dot.x * innerWidth, y = dot.y * innerHeight;
    const hit = document.elementsFromPoint(x, y).map(e => e.id).filter(Boolean);
    const vis = id => { const s = getComputedStyle(document.getElementById(id)); return +s.opacity > 0.01 && s.display !== 'none'; };
    return { hit, hud: vis('hud'), bar: vis('bar'), cam: vis('camwrap') };
  })()`);
  check('the HUD is out of the way while calibrating', clear.hud === false, JSON.stringify(clear));
  check('so is the bar', clear.bar === false);
  check('so is the camera preview', clear.cam === false);

  await page.waitForFunction('window.__eye.state.model && !window.__eye.state.cal', null, { timeout: 45000 });
  const meta = await page.evaluate('({ rmse: __eye.state.modelMeta.rmse, samples: __eye.state.modelMeta.samples })');
  check('the calibration produced a model', meta.samples > 100, JSON.stringify(meta));
  check('the residual is small', meta.rmse < 0.03, String(meta.rmse));
  check('it survives into localStorage', await page.evaluate(`!!localStorage.getItem('eye.calibration.v2')`));

  // ── and the dot goes where it is looking ──────────────────────────────────
  // Direction, not distance. A mirrored model would satisfy any check that only
  // asked whether the dot moved.
  const look = async (x, y) => {
    await page.evaluate(`window.__target = { x: ${x}, y: ${y} }`);
    await sleep(1400);                       // let One Euro settle
    return page.evaluate('({ ...__eye.state.gaze })');
  };
  const L = await look(0.15, 0.5), R = await look(0.85, 0.5);
  const U = await look(0.5, 0.15), D = await look(0.5, 0.85);
  check('looking left puts the dot left', L.x < 0.35, `x = ${L.x?.toFixed(3)}`);
  check('looking right puts the dot right', R.x > 0.65, `x = ${R.x?.toFixed(3)}`);
  check('looking up puts the dot up', U.y < 0.35, `y = ${U.y?.toFixed(3)}`);
  check('looking down puts the dot down', D.y > 0.65, `y = ${D.y?.toFixed(3)}`);

  // ── modes ─────────────────────────────────────────────────────────────────
  for (const m of ['face', 'paint', 'gaze']) {
    await page.click(`[data-mode="${m}"]`);
    check(`the ${m} button selects ${m} mode`, await page.evaluate('document.body.dataset.mode') === m);
  }
  await page.click('[data-act="reset"]');
  check('reset clears the calibration', await page.evaluate('!__eye.state.model'));
  check('  and forgets it on disk too', await page.evaluate(`!localStorage.getItem('eye.calibration.v2')`));

  check('no uncaught errors anywhere in that', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  check('smoke ran to completion', false, String(e));
  ok = false;
} finally {
  await browser.close();
  server.kill();
}

report('smoke');
