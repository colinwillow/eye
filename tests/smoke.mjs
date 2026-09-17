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
    const { makeFace, makeMatrix } = await import('/tests/harness.mjs');
    // Head pose follows the calibration's own pass: still through pass 1, a
    // slow wander through pass 2. That is what the second pass asks a person
    // to do, and without it the pose model has no head variation to fit and
    // the whole comparison below measures nothing.
    window.__head = { yaw: 0, pitch: 0, tx: 0, ty: 0 };
    window.__face = (sx, sy, h) => {
      const D = 5.5, SW = 1.2, SH = 2.5, K = 0.115, SCALE = 0.22;
      const ax = Math.atan2((sx - 0.5) * SW - h.tx, D);
      const ay = Math.atan2((sy - 0.5) * SH - h.ty, D);
      return makeFace({
        gx: -K * Math.sin(ax - h.yaw), gy: K * Math.sin(ay - h.pitch),
        yaw: h.yaw, pitch: h.pitch,
        cx: 0.5 + h.tx * SCALE, cy: 0.42 + h.ty * SCALE,
      });
    };
    window.__target = { x: 0.5, y: 0.5 };
    let t = 0;
    const tr = __eye.state.tracker;
    tr.detect = function () {
      t += 1 / 30;
      const cal = __eye.state.cal;
      const p = cal ? cal.point : window.__target;
      const h = (cal && cal.pass === 2)
        ? { yaw: 0.30 * Math.sin(t * 1.7), pitch: 0.18 * Math.sin(t * 1.1 + 2),
            tx: 0.30 * Math.sin(t * 0.9 + 1), ty: 0.22 * Math.sin(t * 1.4 + 0.5) }
        : window.__head;
      this.result = {
        faceLandmarks: [window.__face(p.x, p.y, h)],
        facialTransformationMatrixes: [makeMatrix({ yaw: h.yaw, pitch: h.pitch, dist: 35 })],
        faceBlendshapes: [{ categories: [
          { categoryName: 'eyeBlinkLeft', score: 0.02 }, { categoryName: 'eyeBlinkRight', score: 0.02 } ] }],
      };
      this.lastVideoTime += 1 / 30;
      return this.result;
    };
  })()`);

  await sleep(600);
  const hud = await page.locator('#hud').textContent();
  check('a stubbed face is picked up', /(^|\\s)face/.test(hud));
  // The matrix has to survive the trip from the detector into the features,
  // because the pose model is built entirely on it and a missing one is not an
  // error — it is a silent fall back to the flat model.
  check('head pose reaches the HUD', /yaw/.test(hud) && !/no head pose/.test(hud), hud);

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

  // Two passes now: nine still dots, then four with the head moving, with an
  // announcement card before each.
  // Wait on the CARD, not on the state machine reaching pass 2. The card is
  // raised by the render loop on the frame after the pass changes, so waiting
  // on the state and then asserting the card is a one-frame race that passes
  // on a fast machine and fails on a slow one.
  await page.waitForFunction(
    `window.__eye.state.cal && window.__eye.state.cal.pass === 2 &&
     document.querySelector('#calintro.show')`, null, { timeout: 60000 });
  check('the head pass announces itself', await page.locator('#calintro.show').count() === 1);
  check('  and says what is different about it',
    /move your head/i.test(await page.locator('#calintro').textContent()));

  await page.waitForFunction('!window.__eye.state.cal && Object.keys(window.__eye.state.models).length', null, { timeout: 120000 });
  const fitted = await page.evaluate('({ sets: Object.keys(__eye.state.models), active: __eye.state.active, meta: __eye.state.meta })');
  check('both models are fitted from the one calibration',
    fitted.sets.includes('flat') && fitted.sets.includes('pose'), JSON.stringify(fitted.sets));
  check('it lands on the head-aware one', fitted.active === 'pose', fitted.active);
  check('both have a residual', fitted.meta.flat?.rmse < 0.05 && fitted.meta.pose?.rmse < 0.05, JSON.stringify(fitted.meta));

  // The diagnostics that exist so "they feel the same" can be checked rather
  // than guessed at: how much the head actually moved, and how far apart the
  // two models are right now.
  const spread = await page.evaluate('({ ...__eye.state.spread })');
  check('the head pass spread is recorded', spread.yaw > 0.3, JSON.stringify(spread));
  check('  and judged sufficient', spread.enough === true);
  await page.evaluate(`window.__head = { yaw: 0.25, pitch: 0.12, tx: 0.2, ty: 0.12 }`);
  await sleep(1800);
  const split = await page.evaluate('__eye.state.split');
  check('the two models visibly disagree with the head turned', split > 0.02, `split ${split}`);
  check('  and the HUD shows it', /split/.test(await page.locator('#hud').textContent()));
  await page.evaluate(`window.__head = { yaw: 0, pitch: 0, tx: 0, ty: 0 }`);
  check('it survives into localStorage', await page.evaluate(`!!localStorage.getItem('eye.calibration.v3')`));

  // The toggle is the entire reason both models are kept.
  await page.click('[data-act="model"]');
  check('the model button flips the active set', await page.evaluate('__eye.state.active') === 'flat');
  check('  and the HUD says which is live', /flat/.test(await page.locator('#hud').textContent()));
  await page.click('[data-act="model"]');
  check('  and flips back', await page.evaluate('__eye.state.active') === 'pose');

  // ── and the dot goes where it is looking ──────────────────────────────────
  // Direction, not distance. A mirrored model would satisfy any check that only
  // asked whether the dot moved.
  const look = async (x, y, head = null) => {
    if (head) await page.evaluate(`window.__head = ${JSON.stringify(head)}`);
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

  // The same four, with the head turned and shifted. This is the claim the
  // second calibration pass exists to support, checked end to end in a browser
  // rather than only against the maths.
  const turned = { yaw: 0.22, pitch: 0.12, tx: 0.18, ty: 0.12 };
  const TL = await look(0.15, 0.5, turned), TR = await look(0.85, 0.5, turned);
  const TU = await look(0.5, 0.15, turned), TD = await look(0.5, 0.85, turned);
  check('with the head turned, left is still left', TL.x < 0.4, `x = ${TL.x?.toFixed(3)}`);
  check('with the head turned, right is still right', TR.x > 0.6, `x = ${TR.x?.toFixed(3)}`);
  check('with the head turned, up is still up', TU.y < 0.4, `y = ${TU.y?.toFixed(3)}`);
  check('with the head turned, down is still down', TD.y > 0.6, `y = ${TD.y?.toFixed(3)}`);
  await page.evaluate(`window.__head = { yaw: 0, pitch: 0, tx: 0, ty: 0 }`);

  // ── modes ─────────────────────────────────────────────────────────────────
  for (const m of ['face', 'paint', 'gaze']) {
    await page.click(`[data-mode="${m}"]`);
    check(`the ${m} button selects ${m} mode`, await page.evaluate('document.body.dataset.mode') === m);
  }
  await page.click('[data-act="reset"]');
  check('reset clears the calibration', await page.evaluate('!Object.keys(__eye.state.models).length'));
  check('  and forgets it on disk too', await page.evaluate(`!localStorage.getItem('eye.calibration.v3')`));

  check('no uncaught errors anywhere in that', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  check('smoke ran to completion', false, String(e));
  ok = false;
} finally {
  await browser.close();
  server.kill();
}

report('smoke');
