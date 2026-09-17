import { Tracker, BlinkDetector } from './tracker.js';
import { extract, rawMapping, FEATURE_SETS } from './features.js';
import { OneEuro2D } from './filter.js';
import { Calibration, predict, saveCalibration, loadCalibration, clearCalibration } from './calibrate.js';
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

// state.models[state.active], or null. Everything downstream asks for this
// rather than reaching into the map, so flipping the active set is one write.
const activeModel = () => state.models[state.active] || null;

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
  // Two models fitted from one calibration, and a button that flips between
  // them. Keeping both is the point: the only way to answer "is head-pose
  // compensation actually better" on a real face is to switch between them
  // mid-session without recalibrating in between.
  models: {},
  meta: {},
  active: 'flat',
  spread: null,
  split: null,
  cal: null,
  shownPass: 0,
  calHoldUntil: 0,
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

    const saved = loadCalibration();
    if (saved) {
      state.models = saved.models; state.meta = saved.meta || {};
      state.active = saved.active; state.spread = saved.spread || null;
    }

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
  // The transformation matrix is a real 3D head pose solved against MediaPipe's
  // canonical face model. It was already being requested and then dropped on
  // the floor; the `pose` feature set is what finally reads it.
  const f = lm ? extract(lm, state.tracker.result?.facialTransformationMatrixes?.[0]) : null;
  if (f?.ok) state.faceSeen = now;

  // Blink. While the eyes are shut the iris centre is being inferred from an
  // iris that is not visible, so the gaze estimate goes wild — hold the last
  // good point instead of following it.
  const score = state.tracker.blinkScore;
  state.blinking = score != null ? score > 0.5 : (f?.ok && f.ear != null ? f.ear < 0.14 : false);
  if (state.blink.update(score, now) === 'click') onBlinkClick();

  if (state.cal) runCalibration(dt, f, now);
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

  // A pose model with no matrix this frame has nothing to predict from. Fall
  // back to the flat model rather than to the raw mapping: the flat one is a
  // real calibration and the raw one is a guess, and silently swapping a
  // calibrated estimate for a guess is a jump the user cannot account for.
  const m = activeModel();
  const set = m && FEATURE_SETS[m.set];
  const usableNow = m && (!set.needsPose || f.pose);
  const model = usableNow ? m : (state.models.flat || null);
  const raw = model ? predict(model, f) : rawMapping(f, state.rest);
  const o = TUNING.overshoot;
  const p = state.filter.filter({
    x: Math.min(1 + o, Math.max(-o, raw.x)),
    y: Math.min(1 + o, Math.max(-o, raw.y)),
  }, dt);

  state.gaze = { x: Math.min(1, Math.max(0, p.x)), y: Math.min(1, Math.max(0, p.y)) };

  // How far apart the two models think you are looking, right now, in percent
  // of the screen. This is the number that settles "they seem the same": if it
  // sits near zero they ARE the same, and the head pass is what to fix. It
  // should grow as you move your head and shrink toward zero when you hold
  // still, because that is the only thing they disagree about.
  if (state.models.flat && state.models.pose && f.pose) {
    const a = predict(state.models.flat, f), b = predict(state.models.pose, f);
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    state.split = state.split == null ? d : state.split + (d - state.split) * 0.08;
  }
}

// The intro card holds for a beat before the first dot. Two reasons: the eyes
// are still on the button that was just pressed, and a calibration wants you
// settled rather than caught mid-reach.
function startCalibration() {
  if (state.cal || state.calStarting) return;
  setMode('gaze');
  document.body.classList.add('calibrating');
  state.calStarting = true;
  state.shownPass = 1;
  showCalCard(1);
  setTimeout(() => {
    el.calintro.classList.remove('show');
    state.calStarting = false;
    state.cal = new Calibration();
  }, 1800);
}

function runCalibration(dt, f, now) {
  // The second pass asks for something different (keep looking at the dot,
  // move your head) and looks identical. Announce it and hold, or it gets done
  // as another still pass and the head terms end up as dead as they were.
  if (state.cal.pass !== state.shownPass) {
    state.shownPass = state.cal.pass;
    state.calHoldUntil = now + 2600;
    showCalCard(state.cal.pass);
  }
  if (now < state.calHoldUntil) return;
  el.calintro.classList.remove('show');

  const running = state.cal.step(dt, f);
  if (running) return;

  const r = state.cal.result;
  state.cal = null;
  document.body.classList.remove('calibrating');

  const fitted = Object.entries(r).filter(([k, v]) => k !== 'spread' && v?.ok);
  if (!fitted.length) {
    const why = Object.values(r)[0]?.reason || 'no samples';
    note(`calibration failed (${why}) — keep your face in frame and try again`, true);
    return;
  }

  state.models = {}; state.meta = {};
  for (const [set, v] of fitted) {
    state.models[set] = v.model;
    state.meta[set] = { rmse: v.rmse.mean, samples: v.samples };
  }
  // Land on the head-aware one when it fitted, since it is the one that was
  // just paid for with an extra pass. The button flips straight back.
  state.active = state.models.pose ? 'pose' : 'flat';
  state.spread = r.spread;
  saveCalibration({ models: state.models, meta: state.meta, active: state.active, spread: r.spread });
  state.filter.reset();
  setMode('gaze');

  // The one thing worth saying out loud. If the head barely moved during the
  // second pass then the head terms had nothing to learn from, the two models
  // are the same model, and flipping between them will feel like nothing is
  // happening — which is indistinguishable from the idea not working unless
  // somebody says so.
  if (!r.pose?.ok) {
    note(`calibrated (${(r.flat.rmse.mean * 100).toFixed(1)}%) — no head pose from the mesh, so only the basic model fitted`, true);
  } else if (r.spread && !r.spread.enough) {
    note(`your head barely moved on the last four dots (${(r.spread.yaw * 57).toFixed(0)}° of turn) — ` +
         `so both models came out the same. Recalibrate and move more on the amber dots.`, true);
  } else {
    const line = fitted.map(([set, v]) => `${set} ${(v.rmse.mean * 100).toFixed(1)}%`).join(' · ');
    note(`calibrated — ${line} · head pass ${(r.spread.yaw * 57).toFixed(0)}° · tap "model" to compare`);
  }
}

function showCalCard(pass) {
  el.calintro.innerHTML = pass === 2
    ? '<div>keep looking at the dot<br>and <b>slowly move your head</b></div>' +
      '<small>four dots · small circles, lean a little · this is what teaches it to ignore your head</small>'
    : '<div>look at each dot<br>until it fills</div>' +
      '<small>nine dots · hold still</small>';
  el.calintro.classList.add('show');
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
    drawCalDot(ctx, { x: p.x * w, y: p.y * h }, state.cal.phase, state.cal.phaseT, state.cal.pass);
  } else if (state.mode !== 'face' && state.gaze) {
    drawGaze(ctx, { x: state.gaze.x * w, y: state.gaze.y * h }, {
      w, h,
      confident: !!activeModel(),
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

const deg = r => (r * 180 / Math.PI).toFixed(0).padStart(3);

function updateHud(f, now) {
  const live = now - state.faceSeen < 400;
  const m = activeModel();
  const cal = m
    ? `${m.set}${state.meta[m.set]?.rmse != null ? ' ±' + (state.meta[m.set].rmse * 100).toFixed(1) + '%' : ''}`
    : 'UNCALIBRATED';
  const other = Object.keys(state.models).filter(k => k !== state.active)
    .map(k => `${k} ${(state.meta[k]?.rmse * 100).toFixed(1)}%`).join(' ');

  // Head pose is printed raw and in degrees because it is the one thing here
  // that cannot be checked without a face in front of the camera: turn your
  // head and watch whether yaw moves the way you would expect, and whether
  // dist matches a tape measure.
  const pose = f?.ok && f.pose
    ? `yaw${deg(f.pose.yaw)} pit${deg(f.pose.pitch)} rol${deg(f.pose.roll)} d${f.pose.dist.toFixed(1)}`
    : '<span class="warn">no head pose</span>';

  // "split" is how far apart the two models are RIGHT NOW. Near zero means
  // they are the same model and switching between them cannot do anything.
  const split = state.split != null
    ? ` · split <b class="${state.split < 0.01 ? 'warn' : ''}">${(state.split * 100).toFixed(1)}%</b>`
    : '';

  el.hud.innerHTML = [
    `${live ? 'face' : '<b class="warn">no face</b>'} · ${state.tracker.delegate || '—'} · ${state.detFps.toFixed(0)}/${state.fps.toFixed(0)} fps`,
    `<span class="${m ? '' : 'warn'}">${cal}</span>${other ? ` <span class="dim">(${other})</span>` : ''}${state.blinking ? ' · <b>blink</b>' : ''}`,
    `${state.spread ? `head pass ${(state.spread.yaw * 57).toFixed(0)}°${state.spread.enough ? '' : ' <b class="warn">TOO STILL</b>'}` : ''}${split}`,
    f?.ok ? `gx ${f.gx.toFixed(3)} gy ${f.gy.toFixed(3)} asy ${f.asym.toFixed(3)}` : (f?.reason ? `— ${f.reason}` : '—'),
    pose,
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
    case 'model': {
      const sets = Object.keys(state.models);
      if (sets.length < 2) { note(sets.length ? 'only one model fitted — recalibrate to get both' : 'calibrate first'); break; }
      state.active = sets[(sets.indexOf(state.active) + 1) % sets.length];
      saveCalibration({ models: state.models, meta: state.meta, active: state.active });
      state.filter.reset();
      note(`${state.active} — ${state.active === 'pose' ? 'head pose compensated' : 'the original, eyes only'}`);
      break;
    }
    case 'reset':
      clearCalibration(); state.models = {}; state.meta = {};
      state.spread = null; state.split = null; state.rest = null; state.filter.reset();
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
  if (activeModel()) return note('already calibrated — reset first if you want the rough mode back');
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
