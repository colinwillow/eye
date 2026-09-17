import { Calibration, calPoints, fit, predict, CAL } from '../src/calibrate.js';
import { extract, designRow } from '../src/features.js';
import { check, near, report, makeFace } from './harness.mjs';

// Deterministic noise. A flaky accuracy check is worse than no accuracy check:
// it gets re-run until it passes and then nobody trusts it.
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const noise = a => (rnd() - 0.5) * 2 * a;

// ── A synthetic eyeball ─────────────────────────────────────────────────────
// Given a point on the screen, what would the iris offsets look like? Roughly
// linear, because the angles a phone subtends are small, plus a deliberate
// quadratic and a cross term so the second-order fit has something real to do.
//
// `flip` inverts the horizontal convention. That is the point of it: the fit
// must come out right EITHER WAY, because the one thing this design refuses to
// do is assume which way a positive iris offset points.
function eyeball(sx, sy, { flip = false, jitter = 0.0015 } = {}) {
  const u = sx - 0.5, v = sy - 0.5;
  const gx = (flip ? 1 : -1) * (0.090 * u + 0.020 * u * v + 0.015 * u * Math.abs(u)) + noise(jitter);
  const gy = 0.055 * v + 0.010 * v * v + 0.012 * u * v + noise(jitter);
  return extract(makeFace({ gx, gy, hx: noise(0.004) }));
}

// ── the dot grid ────────────────────────────────────────────────────────────
{
  const p = calPoints();
  check('nine calibration points', p.length === 9);
  check('the first one is the centre', Math.abs(p[0].x - 0.5) < 1e-9 && Math.abs(p[0].y - 0.5) < 1e-9);
  check('all inside the inset', p.every(q => q.x >= CAL.inset - 1e-9 && q.x <= 1 - CAL.inset + 1e-9 &&
                                             q.y >= CAL.inset - 1e-9 && q.y <= 1 - CAL.inset + 1e-9));
  check('they are all distinct', new Set(p.map(q => `${q.x},${q.y}`)).size === 9);
  check('the grid spans both axes', new Set(p.map(q => q.x.toFixed(4))).size === 3 &&
                                    new Set(p.map(q => q.y.toFixed(4))).size === 3);
}

// ── the sequence runs and produces a model ──────────────────────────────────
function runSequence(opts = {}) {
  const cal = new Calibration();
  const dt = 1 / 30;
  let guard = 0;
  while (cal.step(dt, opts.blind ? null : eyeball(cal.point.x, cal.point.y, opts)) && guard++ < 10000);
  return cal;
}

{
  const cal = runSequence();
  check('the sequence finishes', cal.done);
  check('it produced a model', cal.result?.ok, cal.result?.reason);
  check('it collected a sensible number of samples', cal.result.samples > 150 && cal.result.samples < 400,
    String(cal.result.samples));
  check('the design row width matches the model', cal.result.model.W.length === designRow({ gx: 0, gy: 0, hx: 0, hy: 0 }, 2).length);
  check('the training residual is small', cal.result.rmse.mean < 0.02, String(cal.result.rmse.mean));
}

// ── ACCURACY, on points it was never shown ──────────────────────────────────
// The residual a fit reports on its own training data is the optimistic
// number and it always looks good. This measures a fresh grid, deliberately
// offset from the calibration dots.
{
  const model = runSequence().result.model;
  let worst = 0, sum = 0, n = 0;
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const sx = 0.14 + i * 0.18, sy = 0.14 + j * 0.18;
    const p = predict(model, eyeball(sx, sy));
    const e = Math.hypot(p.x - sx, p.y - sy);
    worst = Math.max(worst, e); sum += e; n++;
  }
  check('mean error on unseen points is under 3% of the screen', sum / n < 0.03, `${(sum / n * 100).toFixed(2)}%`);
  check('worst-case error is under 6%', worst < 0.06, `${(worst * 100).toFixed(2)}%`);
}

// ── DIRECTION, after calibration ────────────────────────────────────────────
// Same reasoning as the feature suite: an accuracy number is a distance, and
// a distance is exactly what a mirrored model would also satisfy if the test
// grid were symmetric. These pin the sides.
{
  const model = runSequence().result.model;
  const at = (x, y) => predict(model, eyeball(x, y));
  check('the left of the screen predicts left', at(0.15, 0.5).x < 0.3, String(at(0.15, 0.5).x));
  check('the right of the screen predicts right', at(0.85, 0.5).x > 0.7, String(at(0.85, 0.5).x));
  check('the top of the screen predicts up', at(0.5, 0.15).y < 0.3, String(at(0.5, 0.15).y));
  check('the bottom of the screen predicts down', at(0.5, 0.85).y > 0.7, String(at(0.5, 0.85).y));
  check('the corners do not swap axes', at(0.15, 0.85).x < 0.35 && at(0.15, 0.85).y > 0.65);
}

// ── HANDEDNESS IS LEARNED, NOT ASSUMED ──────────────────────────────────────
// Feed the fit an eyeball wired backwards. If calibration were leaning on the
// raw mapping's sign convention anywhere, this would come out mirrored. It
// must not — that is the property that makes a wrong guess about the camera's
// mirroring survivable instead of fatal.
{
  const cal = new Calibration();
  const dt = 1 / 30;
  let guard = 0;
  while (cal.step(dt, eyeball(cal.point.x, cal.point.y, { flip: true })) && guard++ < 10000);
  const model = cal.result.model;
  const at = (x, y) => predict(model, eyeball(x, y, { flip: true }));
  check('an inverted eye still calibrates', cal.result.ok);
  check('  left is still left', at(0.15, 0.5).x < 0.3, String(at(0.15, 0.5).x));
  check('  right is still right', at(0.85, 0.5).x > 0.7, String(at(0.85, 0.5).x));
}

// ── second order earns its place ────────────────────────────────────────────
{
  const samples = [];
  for (const p of calPoints()) for (let i = 0; i < 25; i++) {
    const f = eyeball(p.x, p.y);
    samples.push({ f: { gx: f.gx, gy: f.gy, hx: f.hx, hy: f.hy }, target: p });
  }
  const o1 = fit(samples, { ...CAL, order: 1 });
  const o2 = fit(samples, { ...CAL, order: 2 });
  check('order 1 fits', o1.ok);
  check('order 2 fits at least as well', o2.rmse.mean <= o1.rmse.mean + 1e-9,
    `${o2.rmse.mean} vs ${o1.rmse.mean}`);
}

// ── it fails loudly rather than quietly ─────────────────────────────────────
{
  const cal = runSequence({ blind: true });
  check('a calibration with no face at all does not produce a model', !cal.result.ok);
  check('  and says why', cal.result.reason === 'too-few-samples', cal.result.reason);
}
{
  // Eyes shut through the whole thing. The landmarks are still there and still
  // plausible, so nothing throws — the iris is just being inferred from an iris
  // nobody can see. Dropping on the eye-aspect-ratio is what catches it.
  const cal = new Calibration();
  const dt = 1 / 30;
  let guard = 0;
  while (cal.step(dt, extract(makeFace({ ear: 0.05 }))) && guard++ < 10000);
  check('blinked-through samples are dropped', !cal.result.ok, JSON.stringify(cal.result));
}

// ── the model survives a round trip through JSON ────────────────────────────
// It is stored in localStorage between reloads, and a model that comes back
// as something predict() quietly returns NaN for is a dot that vanishes on
// the second visit and works on the first.
{
  const model = runSequence().result.model;
  const back = JSON.parse(JSON.stringify(model));
  const a = predict(model, eyeball(0.3, 0.7)), b = predict(back, eyeball(0.3, 0.7));
  check('a round-tripped model predicts finite numbers', Number.isFinite(b.x) && Number.isFinite(b.y));
  check('  and predicts the same thing', Math.abs(a.x - b.x) < 0.05 && Math.abs(a.y - b.y) < 0.05);
}

report('calibrate');
