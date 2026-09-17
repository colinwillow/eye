import { designRow, ORDER_TERMS } from './features.js';
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

  order: 2,

  // A blink mid-collection ruins a sample: the iris centre is being inferred
  // from a mostly-hidden iris. Anything below this eye-aspect-ratio is dropped.
  earFloor: 0.14,
};

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

// Fit features -> screen. samples is [{ f, target:{x,y} }].
export function fit(samples, cfg = CAL) {
  const order = cfg.order;
  const need = ORDER_TERMS[order];
  if (samples.length < need * 3) return { ok: false, reason: 'too-few-samples' };

  const X = samples.map(s => designRow(s.f, order));
  const Y = samples.map(s => [s.target.x, s.target.y]);

  const W = solveRidge(X, Y, cfg.lambda);
  if (!W) return { ok: false, reason: 'singular' };

  const err = rmse(W, X, Y);
  return {
    ok: true,
    model: { W, order, version: 2 },
    // In fractions of the screen. Residual on the training set, so it is the
    // optimistic number — treat it as a floor, not as accuracy.
    rmse: { x: err[0], y: err[1], mean: Math.hypot(err[0], err[1]) / Math.SQRT2 },
    samples: samples.length,
  };
}

export function predict(model, f) {
  const p = apply(model.W, designRow(f, model.order));
  return { x: p[0], y: p[1] };
}

// ── Running the sequence ────────────────────────────────────────────────────
// A tiny state machine the render loop pumps. It owns no DOM; main.js asks it
// where the dot is and how far through the current phase it is, and draws that.
export class Calibration {
  constructor(cfg = CAL) {
    this.cfg = cfg;
    this.points = calPoints(cfg);
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

  // Fraction through the current phase, 0..1 — drives the dot's animation.
  get phaseT() {
    const dur = this.phase === 'settle' ? this.cfg.settleMs : this.cfg.collectMs;
    return Math.min(1, this.t / dur);
  }

  // dt in seconds. f is the current feature set, or null when there is no face.
  // Returns true while still running.
  step(dt, f) {
    if (this.done) return false;
    this.t += dt * 1000;

    if (this.phase === 'settle') {
      if (this.t >= this.cfg.settleMs) { this.phase = 'collect'; this.t = 0; this.perPoint = 0; }
      return true;
    }

    // Collecting. A blink or a lost face does not stall the sequence — it just
    // does not contribute. Stalling on a lost face means one bad dot holds the
    // whole calibration hostage, which is worse than nine dots and a warning.
    const usable = f && f.ok && (f.ear == null || f.ear >= this.cfg.earFloor);
    if (usable) {
      this.samples.push({ f: { gx: f.gx, gy: f.gy, hx: f.hx, hy: f.hy }, target: this.point });
      this.perPoint++;
    }

    if (this.t >= this.cfg.collectMs) {
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
        this.result = fit(this.samples, this.cfg);
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
const KEY = 'eye.calibration.v2';

export function saveModel(model, meta) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ model, meta, at: Date.now() }));
    return true;
  } catch { return false; }
}

export function loadModel() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    // A model fitted by an older feature set has the wrong number of columns
    // and would silently produce nonsense rather than throwing.
    if (!d?.model?.W || d.model.version !== 2) return null;
    if (d.model.W.length !== ORDER_TERMS[d.model.order]) return null;
    return d;
  } catch { return null; }
}

export function clearModel() {
  try { localStorage.removeItem(KEY); } catch {}
}
