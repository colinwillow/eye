import { Tracker, BlinkDetector } from './tracker.js';
import { extract, rawMapping } from './features.js';
import { OneEuro2D } from './filter.js';
import { Calibration, predict, saveModel, loadModel, clearModel } from './calibrate.js';
import { coverMap, sizeCanvas, drawFace, drawGaze, drawCalDot } from './draw.js';

const TUNING = {
  // What the camera is asked for. Iris landmarks are five points inside a
  // region maybe thirty pixels across, so resolution is the one thing worth
  // spending on — at 480p the iris centre quantises hard and the whole gaze
  // signal gets a visible stair-step.
  video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' },

  // One Euro. Gaze is noisier than a mouse, so the floor is lower and the
  // speed coupling is stronger than the paper's defaults.
  smooth: { minCutoff: 0.9, beta: 0.06 },

  // The predicted point is allowed to run a little off-screen before it is
  // clamped, so looking past the edge pins the dot to the edge rather than
  // having it stop short and look stuck.
  overshoot: 0.25,

  trailLength: 26,
  paintFade: 0.006,   // per frame; a slow bleach so a drawing lasts but not forever
};

const $ = id => document.getElementById(id);
const el = {
  cam: $('cam'), wrap: $('camwrap'), overlay: $('overlay'), stage: $('stage'), paint: $('paint'),
  hud: $('hud'), boot: $('boot'), bootMsg: $('boot-msg'), start: $('start'), bar: $('bar'), note: $('note'),
  calintro: $('calintro'),
};

const state = {
  mode: 'face',
  mesh: false,
  tracker: new Tracker(),
  blink: new BlinkDetector(),
  filter: new OneEuro2D(TUNING.smooth),
  model: null,
  modelMeta: null,
  cal: null,
  rest: null,         // uncalibrated origin — where "straight ahead" is
  gaze: null,         // last smoothed point, 0..1 screen space
  trail: [],
  last: 0,
  fps: 0, detFps: 0, detCount: 0, fpsT: 0, frames: 0,
  faceSeen: 0,
  blinking: false,
  ripples: [],
};

// ── Boot ────────────────────────────────────────────────────────────────────
async function boot() {
  el.start.disabled = true;
  try {
    // getUserMedia needs a secure context AND a user gesture on iOS. The whole
    // reason there is a start button rather than an autostart is that second
    // one — Safari rejects a camera request that did not come from a tap.
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera API. Needs https:// — a plain http:// page cannot open the camera.');

    status('asking for the camera…');
    const stream = await navigator.mediaDevices.getUserMedia({ video: TUNING.video, audio: false });
    el.cam.srcObject = stream;

    // play() can reject on iOS even when everything is fine: the await on
    // getUserMedia has already ended the user-gesture window, and Safari counts
    // that against an explicit play(). The autoplay attribute still starts it,
    // so a rejection here is not a failure — what matters is whether frames
    // actually arrive, which is what the wait below measures.
    await el.cam.play().catch(() => {});
    status('waiting for the first frame…');
    await waitForFrames();

    await state.tracker.load(status);

    const saved = loadModel();
    if (saved) { state.model = saved.model; state.modelMeta = saved.meta; }

    el.boot.classList.add('gone');
    keepAwake();
    requestAnimationFrame(loop);
  } catch (e) {
    status(`${e.name || 'Error'}: ${e.message}`, true);
    el.start.disabled = false;
    el.start.textContent = 'try again';
  }
}

// The camera can be granted and still deliver nothing — another app holding
// it, a virtual device with no source. Without this the tracker loads fine and
// sits forever on a black frame reporting "no face", which looks like the
// tracking being bad rather than the camera being empty.
function waitForFrames(timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    (function poll() {
      if (el.cam.videoWidth > 0 && el.cam.readyState >= 2) return resolve();
      if (performance.now() - t0 > timeoutMs) {
        return reject(new Error('The camera opened but sent no frames. Something else may be using it — close other camera apps and try again.'));
      }
      requestAnimationFrame(poll);
    })();
  });
}

function status(msg, isError = false) {
  el.bootMsg.textContent = msg;
  el.bootMsg.classList.toggle('err', isError);
}

// A phone that sleeps mid-calibration is the single most annoying failure here,
// and the screen has every reason to sleep: nobody is touching it.
let wakeLock = null;
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
}
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') keepAwake(); });

// ── The loop ────────────────────────────────────────────────────────────────
function loop(now) {
  requestAnimationFrame(loop);
  const dt = state.last ? Math.min(0.1, (now - state.last) / 1000) : 1 / 60;
  state.last = now;

  const before = state.tracker.lastVideoTime;
  state.tracker.detect(el.cam, now);
  if (state.tracker.lastVideoTime !== before) state.detCount++;

  const lm = state.tracker.landmarks;
  const f = lm ? extract(lm) : null;
  if (f?.ok) state.faceSeen = now;

  // Blink. While the eyes are shut the iris centre is being inferred from an
  // iris that is not visible, so the gaze estimate goes wild — hold the last
  // good point instead of following it.
  const score = state.tracker.blinkScore;
  state.blinking = score != null ? score > 0.5 : (f?.ok && f.ear != null ? f.ear < 0.14 : false);
  if (state.blink.update(score, now) === 'click') onBlinkClick();

  if (state.cal) runCalibration(dt, f);
  else if (f?.ok && !state.blinking) updateGaze(f, dt);

  render(dt, lm, f);

  state.frames++;
  state.fpsT += dt;
  if (state.fpsT >= 0.5) {
    state.fps = state.frames / state.fpsT;
    state.detFps = state.detCount / state.fpsT;
    state.frames = 0; state.detCount = 0; state.fpsT = 0;
  }
  updateHud(f, now);
}

function updateGaze(f, dt) {
  // With no calibration, "straight ahead" is wherever the face was the first
  // time it was seen. It is a guess, it is labelled as one, and tapping the
  // screen re-takes it.
  if (!state.rest) state.rest = { gx: f.gx, gy: f.gy, hx: f.hx, hy: f.hy };

  const raw = state.model ? predict(state.model, f) : rawMapping(f, state.rest);
  const o = TUNING.overshoot;
  const p = state.filter.filter({
    x: Math.min(1 + o, Math.max(-o, raw.x)),
    y: Math.min(1 + o, Math.max(-o, raw.y)),
  }, dt);

  state.gaze = { x: Math.min(1, Math.max(0, p.x)), y: Math.min(1, Math.max(0, p.y)) };
}

// The intro card holds for a beat before the first dot. Two reasons: the eyes
// are still on the button that was just pressed, and a calibration wants you
// settled rather than caught mid-reach.
function startCalibration() {
  if (state.cal || state.calStarting) return;
  setMode('gaze');
  document.body.classList.add('calibrating');
  state.calStarting = true;
  el.calintro.classList.add('show');
  setTimeout(() => {
    el.calintro.classList.remove('show');
    state.calStarting = false;
    state.cal = new Calibration();
  }, 1500);
}

function runCalibration(dt, f) {
  const running = state.cal.step(dt, f);
  if (running) return;

  const r = state.cal.result;
  state.cal = null;
  document.body.classList.remove('calibrating');
  if (r?.ok) {
    state.model = r.model;
    state.modelMeta = { rmse: r.rmse.mean, samples: r.samples, at: Date.now() };
    saveModel(r.model, state.modelMeta);
    state.filter.reset();
    note(`calibrated · residual ${(r.rmse.mean * 100).toFixed(1)}% of screen · ${r.samples} samples`);
    setMode('gaze');
  } else {
    note(`calibration failed (${r?.reason || 'no samples'}) — keep your face in frame and try again`, true);
  }
}

// ── Rendering ───────────────────────────────────────────────────────────────
function render(dt, lm, f) {
  // Overlay — pinned to the camera picture.
  {
    const { ctx, w, h } = sizeCanvas(el.overlay);
    ctx.clearRect(0, 0, w, h);
    if (lm && el.cam.videoWidth) {
      const map = coverMap(el.cam.videoWidth, el.cam.videoHeight, w, h);
      drawFace(ctx, lm, map, { mesh: state.mesh, features: f });
    }
  }

  // Paint — never cleared, only bleached, so what you drew with your eyes stays.
  if (state.mode === 'paint') {
    const { ctx, w, h } = sizeCanvas(el.paint);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = `rgba(0,0,0,${TUNING.paintFade})`;
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
    if (state.gaze && !state.blinking && state.trail.length > 1) {
      const a = state.trail[state.trail.length - 2], b = state.trail[state.trail.length - 1];
      ctx.strokeStyle = 'rgba(124,224,255,0.5)';
      ctx.lineWidth = 6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
    }
  }

  // Stage — the dot, the calibration, the ripples.
  const { ctx, w, h } = sizeCanvas(el.stage);
  ctx.clearRect(0, 0, w, h);

  if (state.gaze) {
    state.trail.push({ ...state.gaze });
    if (state.trail.length > TUNING.trailLength) state.trail.shift();
  }

  if (state.cal) {
    const p = state.cal.point;
    drawCalDot(ctx, { x: p.x * w, y: p.y * h }, state.cal.phase, state.cal.phaseT);
  } else if (state.mode !== 'face' && state.gaze) {
    drawGaze(ctx, { x: state.gaze.x * w, y: state.gaze.y * h }, {
      w, h,
      confident: !!state.model,
      blink: state.blinking,
      trail: state.mode === 'paint' ? [] : state.trail.map(t => ({ x: t.x * w, y: t.y * h })),
    });
  }

  for (let i = state.ripples.length - 1; i >= 0; i--) {
    const r = state.ripples[i];
    r.t += dt;
    if (r.t > 0.6) { state.ripples.splice(i, 1); continue; }
    const k = r.t / 0.6;
    ctx.strokeStyle = `rgba(255,209,102,${1 - k})`;
    ctx.lineWidth = 3 * (1 - k);
    ctx.beginPath(); ctx.arc(r.x * w, r.y * h, 10 + 70 * k, 0, Math.PI * 2); ctx.stroke();
  }
}

function updateHud(f, now) {
  const live = now - state.faceSeen < 400;
  const cal = state.model
    ? `calibrated ${state.modelMeta?.rmse != null ? '±' + (state.modelMeta.rmse * 100).toFixed(1) + '%' : ''}`
    : 'UNCALIBRATED';
  el.hud.innerHTML = [
    `${live ? 'face' : '<b class="warn">no face</b>'} · ${state.tracker.delegate || '—'} · ${state.detFps.toFixed(0)}/${state.fps.toFixed(0)} fps`,
    `<span class="${state.model ? '' : 'warn'}">${cal}</span>${state.blinking ? ' · <b>blink</b>' : ''}`,
    f?.ok ? `gx ${f.gx.toFixed(3)}  gy ${f.gy.toFixed(3)}  hx ${f.hx.toFixed(3)}` : (f?.reason ? `— ${f.reason}` : '—'),
  ].join('<br>');
}

function note(msg, isError = false) {
  el.note.textContent = msg;
  el.note.classList.toggle('err', isError);
  el.note.classList.add('show');
  clearTimeout(note._t);
  note._t = setTimeout(() => el.note.classList.remove('show'), 4200);
}

function onBlinkClick() {
  if (!state.gaze || state.cal) return;
  state.ripples.push({ x: state.gaze.x, y: state.gaze.y, t: 0 });
}

// ── Controls ────────────────────────────────────────────────────────────────
function setMode(m) {
  state.mode = m;
  document.body.dataset.mode = m;
  for (const b of el.bar.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === m);
  state.trail.length = 0;
}

el.start.addEventListener('click', boot);

el.bar.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.mode) return setMode(b.dataset.mode);
  switch (b.dataset.act) {
    case 'calibrate':
      startCalibration();
      break;
    case 'mesh':
      state.mesh = !state.mesh;
      b.classList.toggle('on', state.mesh);
      break;
    case 'reset':
      clearModel(); state.model = null; state.modelMeta = null; state.rest = null; state.filter.reset();
      note('calibration cleared');
      break;
    case 'clear': {
      const c = el.paint.getContext('2d');
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, el.paint.width, el.paint.height);
      break;
    }
  }
});

// Tapping the picture re-takes the uncalibrated origin — look straight at the
// middle of the screen and tap, and the dot is centred again. Useless once
// calibrated, which is why it only says so then.
el.wrap.addEventListener('pointerdown', () => {
  if (state.model) return note('already calibrated — reset first if you want the rough mode back');
  state.rest = null; state.filter.reset();
  note('re-centred');
});

setMode('face');

// A handle for tests/smoke.mjs, which drives the real page in a real browser:
// it stubs the detector with synthetic landmarks so the calibration flow, the
// canvas sizing and the mode switching can be exercised without a face in
// front of the camera. Only present with ?debug=1, so nothing ships with a
// live handle on its own internals.
if (new URLSearchParams(location.search).get('debug') === '1') {
  window.__eye = { state, setMode, TUNING };
}
