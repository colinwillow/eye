// What this suite is for: every one of these checks exists because the
// corresponding mistake is INVISIBLE in a distance measurement. A gaze
// tracker with a flipped sign tracks beautifully and points the wrong way,
// and a suite that only asserted "the features moved" would pass the whole
// time. So each check below pins a DIRECTION or an INVARIANCE, never just a
// magnitude.
import { extract, rawMapping, eyeMetrics, EYES, LANDMARKS_WITH_IRIS } from '../src/features.js';
import { check, near, report, makeFace } from './harness.mjs';

// ── the signal exists and is zero when it should be ─────────────────────────
{
  const f = extract(makeFace({}));
  check('centred face extracts', f.ok, f.reason);
  near('centred gx is zero', f.gx, 0, 1e-9);
  near('centred gy is zero', f.gy, 0, 1e-9);
  near('centred hx is zero', f.hx, 0, 1e-9);
  near('ear is measured', f.ear, 0.30, 1e-6);
}

// ── HORIZONTAL DIRECTION ────────────────────────────────────────────────────
// The chain under test: iris toward +image x  ->  gx positive  ->  the subject
// is looking to their OWN left  ->  in a MIRRORED selfie view that is the LEFT
// of the screen  ->  a screen x BELOW 0.5.
//
// Three separate sign conventions have to agree for that to hold, and getting
// any one of them backwards produces a tracker that works perfectly in mirror
// image. That is exactly the bug that is impossible to spot by looking at it,
// because a mirrored gaze dot still follows your eyes.
{
  const rest = { gx: 0, gy: 0, hx: 0, hy: 0.55 };

  const right = extract(makeFace({ gx: +0.08 }));
  near('iris toward +image x gives gx > 0', Math.sign(right.gx), 1, 0);
  const mapR = rawMapping(right, rest);
  check('gx > 0 maps to the LEFT of a mirrored screen', mapR.x < 0.5, `x = ${mapR.x.toFixed(3)}`);

  const left = extract(makeFace({ gx: -0.08 }));
  const mapL = rawMapping(left, rest);
  check('gx < 0 maps to the RIGHT of a mirrored screen', mapL.x > 0.5, `x = ${mapL.x.toFixed(3)}`);

  check('the two horizontal extremes are on opposite sides', (mapL.x - 0.5) * (mapR.x - 0.5) < 0);
  near('the mapping is symmetric about the centre', mapL.x + mapR.x, 1.0, 1e-9);
}

// ── VERTICAL DIRECTION ──────────────────────────────────────────────────────
// No mirror on this axis: image y is down and screen y is down, so an iris
// that moved down is a screen y ABOVE 0.5. The check is here because the
// horizontal axis DOES flip, and "flip both for consistency" is the tempting
// wrong move.
{
  const rest = { gx: 0, gy: 0, hx: 0, hy: 0.55 };
  const down = extract(makeFace({ gy: +0.06 }));
  near('iris down gives gy > 0', Math.sign(down.gy), 1, 0);
  check('gy > 0 maps DOWN the screen', rawMapping(down, rest).y > 0.5);

  const up = extract(makeFace({ gy: -0.06 }));
  check('gy < 0 maps UP the screen', rawMapping(up, rest).y < 0.5);
}

// ── HEAD YAW SHARES A SIGN WITH GAZE ────────────────────────────────────────
// The head and the eyes have to push the estimate the same way. If they
// disagree, turning to look at the corner of the screen moves the dot back
// toward the middle — which reads as the tracker fighting you, not as a sign
// error.
{
  const rest = { gx: 0, gy: 0, hx: 0, hy: 0.55 };
  const turned = extract(makeFace({ hx: +0.10 }));
  check('nose toward +image x gives hx > 0', turned.hx > 0, `hx = ${turned.hx}`);
  check('hx > 0 maps to the same side as gx > 0', rawMapping(turned, rest).x < 0.5);

  const both = extract(makeFace({ gx: +0.06, hx: +0.10 }));
  const eyesOnly = extract(makeFace({ gx: +0.06 }));
  check('head and eyes ADD rather than cancel',
    rawMapping(both, rest).x < rawMapping(eyesOnly, rest).x);
}

// ── ROLL INVARIANCE ─────────────────────────────────────────────────────────
// Offsets are projected onto the eye's own axis, so tilting the head must not
// move the signal. Measuring in raw image x/y instead, a 20 degree tilt leaks
// about a third of the horizontal signal into the vertical one — which reads
// as "it drifts when I lean", and gets misdiagnosed as smoothing.
{
  const flat = extract(makeFace({ gx: 0.07, gy: 0.03 }));
  for (const deg of [-25, -12, 12, 25]) {
    const tilt = extract(makeFace({ gx: 0.07, gy: 0.03, roll: deg * Math.PI / 180 }));
    near(`gx survives a ${deg} degree head tilt`, tilt.gx, flat.gx, 2e-3);
    near(`gy survives a ${deg} degree head tilt`, tilt.gy, flat.gy, 2e-3);
  }
  const tilt = extract(makeFace({ roll: 0.3 }));
  near('roll is reported', tilt.roll, 0.3, 1e-6);
}

// ── SCALE INVARIANCE ────────────────────────────────────────────────────────
// Everything is divided by eye width, so holding the phone closer or further
// away must not change the gaze reading. Without this the dot creeps toward
// the centre as you lean back, because the raw pixel offsets shrink.
{
  const a = extract(makeFace({ gx: 0.07, gy: -0.04, scale: 0.15 }));
  const b = extract(makeFace({ gx: 0.07, gy: -0.04, scale: 0.34 }));
  near('gx is independent of distance', a.gx, b.gx, 1e-9);
  near('gy is independent of distance', a.gy, b.gy, 1e-9);
  check('span tracks distance', b.span > a.span * 2 - 1e-6);

  // And translation: the face moving across the frame is not a glance.
  const c = extract(makeFace({ gx: 0.07, gy: -0.04, cx: 0.22, cy: 0.7 }));
  near('gx is independent of face position', c.gx, a.gx, 1e-9);
  near('gy is independent of face position', c.gy, a.gy, 1e-9);
}

// ── CORNER ORDER IS DERIVED, NOT ASSUMED ────────────────────────────────────
// eyeMetrics sorts the two corners by image x instead of trusting which index
// is the outer one. Swapping the indices must therefore change nothing — if it
// flips the sign, the code is trusting MediaPipe's anatomical labelling, and
// that is a guess about handedness rather than a measurement of it.
{
  const lm = makeFace({ gx: 0.07, gy: 0.03 });
  const normal = eyeMetrics(lm, EYES.right);
  const swapped = eyeMetrics(lm, { ...EYES.right, cornerA: EYES.right.cornerB, cornerB: EYES.right.cornerA });
  near('swapping the corner indices does not flip offU', swapped.offU, normal.offU, 1e-12);
  near('swapping the corner indices does not flip offV', swapped.offV, normal.offV, 1e-12);
}

// ── A FACE MESH WITHOUT IRIS POINTS IS A HARD FAILURE ───────────────────────
// 468 landmarks is the un-refined mesh. It has eyelids and corners, so every
// eye-shaped thing still works and the gaze reads as a constant zero. Failing
// loudly beats a dot parked in the middle of the screen.
{
  const short = makeFace({}).slice(0, 468);
  const f = extract(short);
  check('a 468-point mesh is rejected', !f.ok);
  check('and says why', f.reason === 'no-iris', f.reason);
  check('no landmarks at all is rejected', !extract(null).ok);
  check('LANDMARKS_WITH_IRIS is 478', LANDMARKS_WITH_IRIS === 478);
}

// ── DEGENERATE GEOMETRY DOES NOT PRODUCE NaN ────────────────────────────────
// A face at the very edge of frame can collapse an eye to zero width. NaN
// propagates silently into the fit and poisons a calibration that then has to
// be redone; a null is caught on the frame it happens.
{
  const lm = makeFace({});
  lm[33] = { ...lm[133] };            // both corners in the same place
  const f = extract(lm);
  check('a zero-width eye is rejected rather than returning NaN', !f.ok, JSON.stringify(f.gx));
}

report('features');
