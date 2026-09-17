import { Calibration, calPoints, fit, fitAll, predict, CAL, MODEL_VERSION,
         saveCalibration, loadCalibration, clearCalibration } from '../src/calibrate.js';
import { extract, designRow, FEATURE_TERMS } from '../src/features.js';
import { check, near, report, makeFace, makeMatrix } from './harness.mjs';

// Deterministic noise. A flaky accuracy check is worse than no accuracy check:
// it gets re-run until it passes and then nobody trusts it.
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const noise = a => (rnd() - 0.5) * 2 * a;
const reseed = s => { seed = s; };

// ── A synthetic head that obeys the actual geometry ─────────────────────────
// The eye has to POINT at the target, so what the iris does depends on where
// the head is and which way it is facing:
//
//   the angle from the eye to the target, in the world  =  head yaw
//                                                       +  eye rotation in the head
//
// and only that last term is what the iris offset measures. Rearranged, the
// screen position is  D * tan(yaw + eyeAngle), and the tangent of a sum is
// where the whole difficulty lives: expand it and the eye's contribution is
// multiplied by (1 + tan^2 yaw). A turned head does not shift the mapping, it
// RESCALES it — the same iris offset means a different distance across the
// screen depending on where your head is pointing. A model with head terms but
// no cross terms can only shift.
//
// Head TRANSLATION is separate and simpler: move sideways without turning and
// every target is at a new angle, while every rotation feature reads the same.
const D = 5.5;          // eye to screen, in inter-ocular units (~35cm)
const SW = 1.2, SH = 2.5;   // phone screen, same units (~7.5 x 15cm)
const K = 0.115;        // iris offset per unit sine of eye rotation
const SCALE = 0.22;     // projection scale, matching makeFace's default

function look(sx, sy, head = {}, { flip = false, jitter = 0.0015 } = {}) {
  const { yaw = 0, pitch = 0, tx = 0, ty = 0 } = head;
  const ax = Math.atan2((sx - 0.5) * SW - tx, D);
  const ay = Math.atan2((sy - 0.5) * SH - ty, D);
  const gx = (flip ? 1 : -1) * K * Math.sin(ax - yaw) + noise(jitter);
  const gy = K * Math.sin(ay - pitch) + noise(jitter);
  return extract(
    makeFace({ gx, gy, yaw, pitch, cx: 0.5 + tx * SCALE, cy: 0.42 + ty * SCALE }),
    makeMatrix({ yaw, pitch, dist: 35 }));
}

// What a person actually does when told to move their head a little: a slow
// wander, not a sweep to the extremes.
const wander = t => ({
  yaw: 0.30 * Math.sin(t * 1.7),
  pitch: 0.18 * Math.sin(t * 1.1 + 2),
  tx: 0.30 * Math.sin(t * 0.9 + 1),
  ty: 0.22 * Math.sin(t * 1.4 + 0.5),
});

// ── the dot grid ────────────────────────────────────────────────────────────
{
  const p = calPoints();
  check('nine still points', p.length === 9);
  check('the first one is the centre', Math.abs(p[0].x - 0.5) < 1e-9 && Math.abs(p[0].y - 0.5) < 1e-9);
  check('all inside the inset', p.every(q => q.x >= CAL.inset - 1e-9 && q.x <= 1 - CAL.inset + 1e-9 &&
                                             q.y >= CAL.inset - 1e-9 && q.y <= 1 - CAL.inset + 1e-9));
  check('they are all distinct', new Set(p.map(q => `${q.x},${q.y}`)).size === 9);
  check('the grid spans both axes', new Set(p.map(q => q.x.toFixed(4))).size === 3 &&
                                    new Set(p.map(q => q.y.toFixed(4))).size === 3);

  const cal = new Calibration();
  check('the head pass is appended, not mixed in', cal.points.slice(0, 9).every(q => q.pass === 1) &&
                                                   cal.points.slice(9).every(q => q.pass === 2));
  check('the head pass is four corners', cal.points.filter(q => q.pass === 2).length === 4);
}

// ── running the sequence ────────────────────────────────────────────────────
// `moveHead` is what the second pass asks of the person. Being able to turn it
// OFF is what lets the pairing check below exist at all.
function runSequence({ moveHead = true, still = false, blind = false, flip = false } = {}) {
  reseed(999);
  const cal = new Calibration();
  const dt = 1 / 30;
  let t = 0, guard = 0;
  for (;;) {
    t += dt;
    const head = (!still && moveHead && cal.pass === 2) ? wander(t) : {};
    const f = blind ? null : look(cal.point.x, cal.point.y, head, { flip });
    if (!cal.step(dt, f) || guard++ > 20000) break;
  }
  return cal;
}

{
  const cal = runSequence();
  check('the sequence finishes', cal.done);
  check('the flat model fits', cal.result.flat.ok, cal.result.flat.reason);
  check('the pose model fits', cal.result.pose.ok, cal.result.pose.reason);
  check('flat is fitted from the still pass only',
    cal.result.flat.samples < cal.result.pose.samples,
    `${cal.result.flat.samples} vs ${cal.result.pose.samples}`);
  check('the model carries its feature set', cal.result.pose.model?.set === 'pose');
  check('  and its width matches that set', cal.result.pose.model?.W.length === FEATURE_TERMS.pose);
  check('  and a version', cal.result.pose.model?.version === MODEL_VERSION);

  // A fit that drops every sample reports "too-few-samples" from a run that
  // collected hundreds, which reads as a refusal rather than a bug. Compare
  // what was kept against what was collected.
  check('the fits actually used the samples they were given',
    cal.result.pose.samples > cal.samples.length * 0.9,
    `kept ${cal.result.pose.samples} of ${cal.samples.length}`);
}

// ── ACCURACY WITH A STILL HEAD ──────────────────────────────────────────────
// The baseline case. The new model must not be WORSE here — extra terms fitted
// to a calibration that never exercised them is exactly how a "smarter" model
// ends up worse than the one it replaced.
function evaluate(model, { heads, n = 240 }) {
  reseed(4242);
  let sum = 0, worst = 0;
  for (let i = 0; i < n; i++) {
    const sx = 0.08 + rnd() * 0.84, sy = 0.08 + rnd() * 0.84;
    const p = predict(model, look(sx, sy, heads()));
    const e = Math.hypot(p.x - sx, p.y - sy);
    sum += e; worst = Math.max(worst, e);
  }
  return { mean: sum / n, worst };
}

const trained = runSequence();
const STILL = () => ({});
const MOVING = () => ({
  yaw: (rnd() - 0.5) * 0.6, pitch: (rnd() - 0.5) * 0.4,
  tx: (rnd() - 0.5) * 0.6, ty: (rnd() - 0.5) * 0.5,
});

{
  const flat = evaluate(trained.result.flat.model, { heads: STILL });
  const pose = evaluate(trained.result.pose.model, { heads: STILL });
  check('flat is accurate with a still head', flat.mean < 0.045, `${(flat.mean * 100).toFixed(2)}%`);
  check('pose is accurate with a still head', pose.mean < 0.045, `${(pose.mean * 100).toFixed(2)}%`);
  check('pose does not regress the still case', pose.mean < flat.mean * 1.15,
    `pose ${(pose.mean * 100).toFixed(2)}% vs flat ${(flat.mean * 100).toFixed(2)}%`);

  // The pose model's TRAINING residual is the worse of the two, because it is
  // fitted to a harder dataset — nine still dots plus four with the head
  // wandering. Reading that as "flat is the better model" is the trap: a
  // residual measured on a calibration where the head never moved is a score
  // for a test that left out the thing being tested.
  check('the residual on its own calibration flatters the flat model',
    trained.result.flat.rmse.mean < trained.result.pose.rmse.mean,
    `flat ${trained.result.flat.rmse.mean.toFixed(4)} vs pose ${trained.result.pose.rmse.mean.toFixed(4)}`);
}

// ── THE WHOLE POINT: A HEAD THAT MOVES ──────────────────────────────────────
// A phone at arm's length subtends about 12 degrees. A head turn of 17 — which
// is nothing, it is glancing at someone next to you — swings the point your
// eyes have to aim at by MORE THAN A SCREEN WIDTH. That is the whole reason
// propping the phone against something transforms this: not that the tracker
// needs a still head, but that a small head movement is enormous compared to
// the thing being measured, and the flat model has no way to see it.
{
  const gentle = () => ({ yaw: (rnd() - 0.5) * 0.24, pitch: (rnd() - 0.5) * 0.18,
                          tx: (rnd() - 0.5) * 0.24, ty: (rnd() - 0.5) * 0.2 });
  const flatG = evaluate(trained.result.flat.model, { heads: gentle });
  const poseG = evaluate(trained.result.pose.model, { heads: gentle });
  check('even a small head movement wrecks the flat model', flatG.mean > 0.15,
    `${(flatG.mean * 100).toFixed(1)}% — if this ever fails, the rig stopped moving the head`);
  check('the pose model shrugs off a small head movement', poseG.mean < 0.05,
    `${(poseG.mean * 100).toFixed(2)}%`);

  // SLIDING WITHOUT TURNING is the case a hand-held phone produces constantly,
  // and it is the one that only `faceX`, `faceY` and `span` can see — every
  // rotation feature reads identically through it. It is therefore the check
  // that fails if those three ever get regularised back into uselessness,
  // which is exactly what was happening before solveRidge centred its columns:
  // pose measured 13.7% here, barely better than the flat model's 13.7%.
  const slide = () => ({ tx: (rnd() - 0.5) * 0.6, ty: (rnd() - 0.5) * 0.5 });
  const flatS = evaluate(trained.result.flat.model, { heads: slide });
  const poseS = evaluate(trained.result.pose.model, { heads: slide });
  check('sliding the head defeats the flat model', flatS.mean > 0.09, `${(flatS.mean * 100).toFixed(2)}%`);
  check('the pose model handles a pure slide', poseS.mean < 0.055, `${(poseS.mean * 100).toFixed(2)}%`);
  check('  which is a 3x improvement, and it is the handheld case',
    poseS.mean * 3 < flatS.mean, `pose ${(poseS.mean * 100).toFixed(2)}% vs flat ${(flatS.mean * 100).toFixed(2)}%`);

  const flat = evaluate(trained.result.flat.model, { heads: MOVING });
  const pose = evaluate(trained.result.pose.model, { heads: MOVING });
  check('a freely moving head leaves the flat model useless', flat.mean > 0.2,
    `${(flat.mean * 100).toFixed(1)}%`);
  check('the pose model survives a freely moving head', pose.mean < 0.08, `${(pose.mean * 100).toFixed(2)}%`);
  check('pose beats flat by at least 4x with a moving head',
    pose.mean * 4 < flat.mean,
    `pose ${(pose.mean * 100).toFixed(2)}% vs flat ${(flat.mean * 100).toFixed(2)}%`);
  check('and the worst case improves too', pose.worst < flat.worst,
    `${(pose.worst * 100).toFixed(2)}% vs ${(flat.worst * 100).toFixed(2)}%`);
}

// ── THE HEAD PASS HAS TO ACTUALLY MOVE THE HEAD ─────────────────────────────
// The failure everybody will hit. "Move your head a little" is vague, people
// under-do it, and a head that stayed still produces a pose model numerically
// identical to the flat one — which feels exactly like the idea not working.
// So it is measured, and the app says so rather than leaving it to be guessed.
{
  const moved = runSequence();
  check('a real head pass is recognised as enough', moved.result.spread.enough,
    JSON.stringify(moved.result.spread));
  check('  and reports a sensible spread of yaw', moved.result.spread.yaw > 0.3,
    String(moved.result.spread.yaw));

  const stillPass = runSequence({ still: true });
  check('a head pass done without moving is caught', !stillPass.result.spread.enough,
    JSON.stringify(stillPass.result.spread));
  check('  and it is caught because the head really did not move',
    stillPass.result.spread.yaw < 0.01, String(stillPass.result.spread.yaw));

  // And the thing the warning is actually about: the two models come out the
  // same, so the button that switches between them does nothing.
  const at = m => predict(m, look(0.2, 0.3, { yaw: 0.2, tx: 0.15 }));
  const a = at(stillPass.result.flat.model), b = at(stillPass.result.pose.model);
  const moveA = at(moved.result.flat.model), moveB = at(moved.result.pose.model);
  check('without movement the two models barely disagree',
    Math.hypot(a.x - b.x, a.y - b.y) < 0.15, String(Math.hypot(a.x - b.x, a.y - b.y)));
  check('with movement they disagree a lot, which is the point',
    Math.hypot(moveA.x - moveB.x, moveA.y - moveB.y) > 0.3,
    String(Math.hypot(moveA.x - moveB.x, moveA.y - moveB.y)));
}

// ── THE TWO HALVES ONLY WORK AS A PAIR ──────────────────────────────────────
// The features and the head-movement pass were added together, and neither is
// worth anything alone. This is the check that says so, and it is the reason
// calibration got longer rather than just the feature list getting richer.
//
// Fit the SAME pose feature set from a calibration where the head never moved:
// every head term and every cross term is a column with no variation for the
// data to attribute anything to, ridge shrinks them toward zero, and the model
// comes back barely better than the flat one it was supposed to beat. Anyone
// measuring only the feature change would conclude the idea did not work.
{
  const stillOnly = runSequence({ still: true });
  const naive = stillOnly.result.pose.model;
  const naiveErr = evaluate(naive, { heads: MOVING });
  const goodErr = evaluate(trained.result.pose.model, { heads: MOVING });
  const flatErr = evaluate(trained.result.flat.model, { heads: MOVING });

  check('pose features WITHOUT the head pass barely help',
    naiveErr.mean > flatErr.mean * 0.6,
    `naive-pose ${(naiveErr.mean * 100).toFixed(2)}% vs flat ${(flatErr.mean * 100).toFixed(2)}%`);
  check('pose features WITH the head pass are transformative',
    goodErr.mean < naiveErr.mean * 0.5,
    `with ${(goodErr.mean * 100).toFixed(2)}% vs without ${(naiveErr.mean * 100).toFixed(2)}%`);
}

// ── DIRECTION, after calibration ────────────────────────────────────────────
// An accuracy number is a distance, and a mirrored model satisfies every
// distance a symmetric test grid can produce. These pin the sides, for both
// models, with the head both still and turned.
for (const set of ['flat', 'pose']) {
  const model = trained.result[set].model;
  for (const [name, head] of [['still', {}], ['turned', { yaw: 0.25, tx: 0.2 }]]) {
    const at = (x, y) => predict(model, look(x, y, head));
    const skip = set === 'flat' && name === 'turned';   // flat is not expected to hold up
    if (skip) continue;
    check(`${set}/${name}: the left of the screen predicts left`, at(0.15, 0.5).x < 0.35, String(at(0.15, 0.5).x));
    check(`${set}/${name}: the right predicts right`, at(0.85, 0.5).x > 0.65, String(at(0.85, 0.5).x));
    check(`${set}/${name}: the top predicts up`, at(0.5, 0.15).y < 0.35, String(at(0.5, 0.15).y));
    check(`${set}/${name}: the bottom predicts down`, at(0.5, 0.85).y > 0.65, String(at(0.5, 0.85).y));
    check(`${set}/${name}: the corners do not swap axes`, at(0.15, 0.85).x < 0.4 && at(0.15, 0.85).y > 0.6);
  }
}

// ── HANDEDNESS IS LEARNED, NOT ASSUMED ──────────────────────────────────────
// Feed the fit an eyeball wired backwards. If anything downstream of the
// features leaned on the raw mapping's sign convention, this comes out
// mirrored. It must not — that is the property that makes a wrong guess about
// the camera's mirroring survivable instead of fatal.
{
  const cal = runSequence({ flip: true });
  for (const set of ['flat', 'pose']) {
    const at = (x, y) => predict(cal.result[set].model, look(x, y, {}, { flip: true }));
    check(`${set}: an inverted eye still calibrates`, cal.result[set].ok);
    check(`${set}: left is still left`, at(0.15, 0.5).x < 0.35, String(at(0.15, 0.5).x));
    check(`${set}: right is still right`, at(0.85, 0.5).x > 0.65, String(at(0.85, 0.5).x));
  }
}

// ── it fails loudly rather than quietly ─────────────────────────────────────
{
  const cal = runSequence({ blind: true });
  check('no face at all produces no model', !cal.result.flat.ok && !cal.result.pose.ok);
  check('  and says why', cal.result.flat.reason === 'too-few-samples', cal.result.flat.reason);
}
{
  // Eyes shut throughout. The landmarks are still there and still plausible,
  // so nothing throws — the iris is just being inferred from an iris nobody
  // can see. The eye-aspect-ratio floor is what catches it.
  const cal = new Calibration();
  let guard = 0;
  while (cal.step(1 / 30, extract(makeFace({ ear: 0.05 }), makeMatrix({}))) && guard++ < 20000);
  check('blinked-through samples are dropped', !cal.result.flat.ok, JSON.stringify(cal.result.flat));
}
{
  // No transformation matrix at all — an older model, or the option failing.
  // The flat fit must still work and the pose fit must decline rather than
  // fitting to a column of NaN.
  const cal = new Calibration();
  let guard = 0, t = 0;
  while (cal.step(1 / 30, look(cal.point.x, cal.point.y, cal.pass === 2 ? wander(t += 1 / 30) : {})
    && (() => { const f = look(cal.point.x, cal.point.y); f.pose = null; return f; })()) && guard++ < 20000);
  check('without a matrix the flat model still fits', cal.result.flat.ok, cal.result.flat.reason);
  check('  and the pose model declines rather than fitting NaN', !cal.result.pose.ok, cal.result.pose.reason);
}

// ── persistence ─────────────────────────────────────────────────────────────
{
  // A minimal localStorage, since this runs in node.
  const store = new Map();
  globalThis.localStorage = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
  };

  const models = { flat: trained.result.flat.model, pose: trained.result.pose.model };
  const meta = { flat: { rmse: 0.01 }, pose: { rmse: 0.008 } };
  check('it saves', saveCalibration({ models, meta, active: 'pose' }));

  const back = loadCalibration();
  check('it comes back', !!back);
  check('  with both models', !!back.models.flat && !!back.models.pose);
  check('  and the active one', back.active === 'pose');
  const a = predict(models.pose, look(0.3, 0.7));
  const b = predict(back.models.pose, look(0.3, 0.7));
  check('  predicting finite numbers', Number.isFinite(b.x) && Number.isFinite(b.y));
  check('  predicting the same thing', Math.abs(a.x - b.x) < 0.06 && Math.abs(a.y - b.y) < 0.06);

  // A model whose width does not match the feature set it claims. apply()
  // would happily multiply what it was given and return a confident wrong
  // answer rather than throwing, so this has to be caught at load.
  saveCalibration({ models: { pose: { ...models.pose, W: models.pose.W.slice(0, 4) } }, meta, active: 'pose' });
  check('a model of the wrong width is rejected', loadCalibration() === null);

  saveCalibration({ models: { flat: { ...models.flat, version: 2 } }, meta, active: 'flat' });
  check('a model from an older version is rejected', loadCalibration() === null);

  saveCalibration({ models: { pose: models.pose }, meta, active: 'flat' });
  check('an active set that did not fit falls back to one that did', loadCalibration().active === 'pose');

  clearCalibration();
  check('clearing removes it', loadCalibration() === null);
  check('  and the old v2 key with it', store.size === 0);
}

// ── fitting a set from samples collected before it existed ──────────────────
// The samples keep every raw feature rather than a pre-built design row, which
// is what lets a new feature set be fitted from an old calibration instead of
// needing everyone to sit through another one.
{
  const both = fitAll(trained.samples);
  check('fitAll re-fits from stored samples', both.flat.ok && both.pose.ok);
  const one = fit(trained.samples, 'flat');
  check('a single set can be fitted on its own', one.ok && one.model.set === 'flat');
  check('designRow defaults to the baseline', designRow(trained.samples[0].f).length === FEATURE_TERMS.flat);
}

report('calibrate');
