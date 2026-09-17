import { designRow, FEATURE_SETS, FEATURE_TERMS, usable } from './features.js';
import { solveRidge, apply, rmse } from './solve.js';

// The calibration sequence and the model it produces.
//
// Shape of the thing: show a dot, wait for the eyes to actually get there,
// collect a burst of samples, move on. Nine dots, then one least-squares fit
// from features to screen position.

export const CAL = {
  // A 3x3 grid inset from the edges. Not the extreme corners: an eye at the
  // very corner of a phone held at arm's length is at the end of its comfortable
  // range and the samples there are both noisy and unrepresentative, which
  // drags the whole fit. 12% in still spans most of the screen.
  inset: 0.12,
  cols: 3,
  rows: 3,

  // Settle first, then collect. Collecting from the moment the dot appears
  // averages in the saccade that is still on its way there, which is what
  // pulls every calibration point toward the middle of the screen.
  settleMs: 650,
  collectMs: 750,

  // A sample every frame for 750ms is 20-45 of them. Below this many the dot
  // is dropped rather than fitted from noise.
  minSamples: 8,

  // Ridge strength. Picked so a nine-point calibration still extrapolates
  // sanely past its own corners instead of flinging the dot off-screen.
  lambda: 2e-4,

  // A blink mid-collection ruins a sample: the iris centre is being inferred
  // from a mostly-hidden iris. Anything below this eye-aspect-ratio is dropped.
  earFloor: 0.14,

  // ── The head pass ─────────────────────────────────────────────────────────
  // Four more dots, and this time you keep looking at the dot while slowly
  // moving your head around.
  //
  // This is the half of the head-compensation work that is not obvious, and
  // without it the other half measures as nothing. A calibration done with a
  // still head contains no head variation, so every head term and every
  // head-by-eye cross term is a column the data cannot distinguish from a
  // constant — ridge shrinks them to zero and the fit comes back saying the
  // new features did not help. They did not help because they were never
  // shown a single example of the thing they exist to correct.
  //
  // Corners rather than a grid: a corner is where eye rotation and head
  // rotation are most able to trade off against each other, which is exactly
  // the confusion the cross terms have to resolve.
  headPoints: [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]],
  headSettleMs: 500,
  headCollectMs: 2200,

  // How much the head has to actually MOVE during the second pass for the
  // head-aware model to be worth anything.
  //
  // This is the failure everyone will hit, because "move your head a little"
  // is a vague instruction and people under-do it. Move too little and the
  // head terms have nothing to explain, ridge shrinks them, and the pose model
  // comes out numerically identical to the flat one — at which point flipping
  // between them feels like nothing is happening, because nothing is. That is
  // indistinguishable from "the idea did not work", so it has to be MEASURED
  // and said out loud rather than left to be inferred.
  //
  // Ranges, not deviations: 0.35 rad is a 20-degree spread of yaw across the
  // pass, which is a small movement honestly done. The slide is a fraction of
  // the frame width.
  minSpread: { yaw: 0.35, pitch: 0.20, slide: 0.05 },
};

// What the head actually did during the second pass, as the total range each
// feature covered.
export function headSpread(samples) {
  const p2 = samples.filter(s => s.pass === 2);
  if (!p2.length) return null;
  const range = get => {
    let lo = Infinity, hi = -Infinity;
    for (const s of p2) { const v = get(s); if (v == null || !Number.isFinite(v)) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
    return hi > lo ? hi - lo : 0;
  };
  const spread = {
    yaw: range(s => s.f.pose?.yaw),
    pitch: range(s => s.f.pose?.pitch),
    slideX: range(s => s.f.faceX),
    slideY: range(s => s.f.faceY),
    n: p2.length,
  };
  spread.slide = Math.max(spread.slideX, spread.slideY);
  spread.enough = spread.yaw >= CAL.minSpread.yaw ||
                  spread.pitch >= CAL.minSpread.pitch ||
                  spread.slide >= CAL.minSpread.slide;
  return spread;
}

// Where the dots go, in 0..1 screen space, in the order they are shown.
// Centre first so the first dot is the easy one and the person is settled
// before the awkward corners; after that it walks the grid.
export function calPoints(cfg = CAL) {
  const pts = [];
  for (let r = 0; r < cfg.rows; r++) {
    for (let c = 0; c < cfg.cols; c++) {
      pts.push({
        x: cfg.inset + (c / (cfg.cols - 1)) * (1 - 2 * cfg.inset),
        y: cfg.inset + (r / (cfg.rows - 1)) * (1 - 2 * cfg.inset),
      });
    }
  }
  const mid = Math.floor(pts.length / 2);
  return [pts[mid], ...pts.slice(0, mid), ...pts.slice(mid + 1)];
}

export const MODEL_VERSION = 3;

// Fit features -> screen for one named feature set.
// samples is [{ f, target:{x,y} }].
export function fit(samples, set = 'flat', cfg = CAL) {
  const good = samples.filter(s => usable(s.f, set));
  const need = FEATURE_TERMS[set];
  if (good.length < need * 3) return { ok: false, set, reason: 'too-few-samples' };

  const X = good.map(s => designRow(s.f, set));
  const Y = good.map(s => [s.target.x, s.target.y]);

  const W = solveRidge(X, Y, cfg.lambda);
  if (!W) return { ok: false, set, reason: 'singular' };

  const err = rmse(W, X, Y);
  return {
    ok: true,
    set,
    model: { W, set, version: MODEL_VERSION },
    // In fractions of the screen. Residual on the training set, so it is the
    // optimistic number — treat it as a floor, not as accuracy.
    rmse: { x: err[0], y: err[1], mean: Math.hypot(err[0], err[1]) / Math.SQRT2 },
    samples: good.length,
  };
}

// Fit every set from ONE set of samples.
//
// This is the only honest way to compare them. Two separate calibrations
// differ in how you sat, where the light was and how steady you were that
// minute, and those differences are bigger than the difference between the
// models. Same samples, two fits, and a button that flips between them live.
//
// `flat` is deliberately fitted from the STILL pass only — it is the measured
// baseline and it stays exactly what it was. Feeding it the head-movement pass
// too would hand it variance it has no terms to explain, which would flatter
// the comparison by making the baseline worse rather than the new model better.
export function fitAll(samples, cfg = CAL) {
  const still = samples.filter(s => s.pass !== 2);
  return {
    flat: fit(still, 'flat', cfg),
    pose: fit(samples, 'pose', cfg),
    spread: headSpread(samples),
  };
}

export function predict(model, f) {
  const p = apply(model.W, designRow(f, model.set));
  return { x: p[0], y: p[1] };
}

// ── Running the sequence ────────────────────────────────────────────────────
// A tiny state machine the render loop pumps. It owns no DOM; main.js asks it
// where the dot is and how far through the current phase it is, and draws that.
export class Calibration {
  constructor(cfg = CAL) {
    this.cfg = cfg;
    this.still = calPoints(cfg);
    this.points = [
      ...this.still.map(p => ({ ...p, pass: 1 })),
      ...cfg.headPoints.map(([x, y]) => ({ x, y, pass: 2 })),
    ];
    this.index = 0;
    this.phase = 'settle';
    this.t = 0;
    this.samples = [];
    this.perPoint = 0;
    this.done = false;
    this.result = null;
  }

  get point() { return this.points[this.index]; }
  get progress() { return this.index / this.points.length; }

  // 1 = hold still, 2 = keep looking and move your head. The UI has to say
  // which, because the second pass looks identical and doing it wrong (holding
  // still through it) silently produces exactly the calibration this change
  // exists to avoid.
  get pass() { return this.point?.pass ?? 1; }

  get settleMs() { return this.pass === 2 ? this.cfg.headSettleMs : this.cfg.settleMs; }
  get collectMs() { return this.pass === 2 ? this.cfg.headCollectMs : this.cfg.collectMs; }

  // Fraction through the current phase, 0..1 — drives the dot's animation.
  get phaseT() {
    const dur = this.phase === 'settle' ? this.settleMs : this.collectMs;
    return Math.min(1, this.t / dur);
  }

  // dt in seconds. f is the current feature set, or null when there is no face.
  // Returns true while still running.
  step(dt, f) {
    if (this.done) return false;
    this.t += dt * 1000;

    if (this.phase === 'settle') {
      if (this.t >= this.settleMs) { this.phase = 'collect'; this.t = 0; this.perPoint = 0; }
      return true;
    }

    // Collecting. A blink or a lost face does not stall the sequence — it just
    // does not contribute. Stalling on a lost face means one bad dot holds the
    // whole calibration hostage, which is worse than nine dots and a warning.
    const ok = f && f.ok && (f.ear == null || f.ear >= this.cfg.earFloor);
    if (ok) {
      this.samples.push({
        // A flat copy of every raw feature, so a set that is added later can be
        // fitted from a calibration that was collected before it existed.
        //
        // `ok` is carried deliberately. usable() gates on it, and a copy that
        // dropped it made every stored sample fail that gate — which surfaces
        // as "too-few-samples" from a calibration that collected 415 of them,
        // i.e. as a calibration that silently refuses rather than one that
        // errors.
        f: {
          ok: true,
          gx: f.gx, gy: f.gy, hx: f.hx, hy: f.hy,
          asym: f.asym, widthRatio: f.widthRatio,
          faceX: f.faceX, faceY: f.faceY, span: f.span,
          pose: f.pose ? { yaw: f.pose.yaw, pitch: f.pose.pitch, roll: f.pose.roll, dist: f.pose.dist } : null,
        },
        target: { x: this.point.x, y: this.point.y },
        pass: this.point.pass,
      });
      this.perPoint++;
    }

    if (this.t >= this.collectMs) {
      this.dropped = (this.dropped || 0) + (this.perPoint < this.cfg.minSamples ? 1 : 0);
      if (this.perPoint < this.cfg.minSamples) {
        // Drop this dot's samples entirely rather than fitting to a handful.
        this.samples.length = this.samples.length - this.perPoint;
      }
      this.index++;
      this.t = 0;
      this.phase = 'settle';
      if (this.index >= this.points.length) {
        this.done = true;
        this.result = fitAll(this.samples, this.cfg);
        return false;
      }
    }
    return true;
  }
}

// ── Persistence ─────────────────────────────────────────────────────────────
// Keeping the model across reloads matters more than it sounds: the loop here
// is edit-push-look-at-phone, and re-calibrating on every reload would make
// the whole thing tedious enough not to get tested.
const KEY = 'eye.calibration.v3';

export function saveCalibration(data) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...data, at: Date.now() }));
    return true;
  } catch { return false; }
}

export function loadCalibration() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    if (!d?.models) return null;

    // A model fitted by an older or different feature set has the wrong number
    // of columns, and apply() would happily multiply what it was given and
    // return a confident wrong answer rather than throwing. Check the width
    // against the set it claims, and drop anything that does not line up.
    const models = {};
    for (const [set, m] of Object.entries(d.models)) {
      if (!m?.W || m.version !== MODEL_VERSION) continue;
      if (!FEATURE_SETS[set] || m.set !== set) continue;
      if (m.W.length !== FEATURE_TERMS[set]) continue;
      models[set] = m;
    }
    if (!Object.keys(models).length) return null;
    return { ...d, models, active: models[d.active] ? d.active : Object.keys(models)[0] };
  } catch { return null; }
}

export function clearCalibration() {
  // v2 too: an old key left behind is a calibration that comes back from the
  // dead the next time the loader is relaxed.
  try { localStorage.removeItem(KEY); localStorage.removeItem('eye.calibration.v2'); } catch {}
}
