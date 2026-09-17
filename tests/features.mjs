// What this suite is for: every one of these checks exists because the
// corresponding mistake is INVISIBLE in a distance measurement. A gaze
// tracker with a flipped sign tracks beautifully and points the wrong way,
// and a suite that only asserted "the features moved" would pass the whole
// time. So each check below pins a DIRECTION or an INVARIANCE, never just a
// magnitude.
import { extract, rawMapping, eyeMetrics, matrixPose, designRow, usable,
         FEATURE_SETS, FEATURE_TERMS, EYES, LANDMARKS_WITH_IRIS } from '../src/features.js';
import { check, near, report, makeFace, makeMatrix } from './harness.mjs';

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


// ── HEAD POSE OUT OF THE TRANSFORMATION MATRIX ──────────────────────────────
// The three angles have to be DECOUPLED. Euler extraction with the wrong
// rotation order gives you angles that look plausible, move in roughly the
// right direction, and quietly contain each other — so a pure yaw also reads
// as some pitch, and the fit ends up with two columns that are partly the same
// column. Ridge does not fix that; it just shares the coefficient between them.
{
  const flat = matrixPose(makeMatrix({}));
  near('a head facing the camera reads zero yaw', flat.yaw, 0, 1e-9);
  near('  zero pitch', flat.pitch, 0, 1e-9);
  near('  zero roll', flat.roll, 0, 1e-9);
  near('  and the distance straight off the translation', flat.dist, 35, 1e-9);

  for (const d of [-30, -15, 15, 30]) {
    const r = d * Math.PI / 180;
    const y = matrixPose(makeMatrix({ yaw: r }));
    check(`yaw ${d} is monotonic`, Math.sign(y.yaw) === Math.sign(d) && Math.abs(y.yaw) > 0.1);
    near(`  and leaks no pitch`, y.pitch, 0, 1e-9);
    near(`  and leaks no roll`, y.roll, 0, 1e-9);

    const p = matrixPose(makeMatrix({ pitch: r }));
    check(`pitch ${d} is monotonic`, Math.abs(p.pitch) > 0.1);
    near(`  and leaks no yaw`, p.yaw, 0, 1e-9);
  }
  // Roll is measured off a different column on purpose, so it stays clean.
  const rolled = matrixPose(makeMatrix({ roll: 0.4 }));
  near('roll is recovered', Math.abs(rolled.roll), 0.4, 1e-9);
  near('  and leaks no yaw', rolled.yaw, 0, 1e-9);

  check('a missing matrix is null, not zeroes', matrixPose(null) === null);
  check('a short matrix is null', matrixPose({ data: [1, 0, 0] }) === null);
}

// ── HEAD POSE MUST NOT MOVE WHEN ONLY THE EYES DO ───────────────────────────
// This is the property that makes the cross terms mean anything. If the head
// features drifted with gaze, gx*yaw would be partly gx*gx and the fit would
// have no way to tell "you turned your head" from "you looked further across".
{
  const a = extract(makeFace({ gx: -0.09 }), makeMatrix({ yaw: 0 }));
  const b = extract(makeFace({ gx: +0.09 }), makeMatrix({ yaw: 0 }));
  near('yaw is unmoved by a glance', a.pose.yaw, b.pose.yaw, 1e-12);
  near('pitch is unmoved by a glance', a.pose.pitch, b.pose.pitch, 1e-12);
  near('asym is unmoved by a glance', a.asym, b.asym, 1e-9);
  near('widthRatio is unmoved by a glance', a.widthRatio, b.widthRatio, 1e-9);
}

// ── THE TWO EYES DISAGREE UNDER YAW, AND THAT IS THE SIGNAL ─────────────────
// Turning the head brings one eye nearer the camera, so it projects bigger.
// Averaging the eyes throws that away; `asym` and `widthRatio` keep it, and
// they are head-pose readings that need no matrix at all — which is what the
// pose model falls back on if MediaPipe ever stops handing one over.
{
  const straight = extract(makeFace({}));
  near('a straight-on face is symmetric', straight.widthRatio, 0, 1e-9);
  near('  and has no iris asymmetry', straight.asym, 0, 1e-9);

  const left = extract(makeFace({ yaw: -0.35 }));
  const right = extract(makeFace({ yaw: +0.35 }));
  check('yaw shows up in the eye widths', Math.abs(right.widthRatio) > 0.02, String(right.widthRatio));
  check('  and flips with the direction of the turn', left.widthRatio * right.widthRatio < 0);
  near('  symmetrically', left.widthRatio, -right.widthRatio, 1e-9);

  // Strictly monotonic, not just non-zero: a feature that saturates or turns
  // around is one the fit can only use over half its range. The DIRECTION is
  // not asserted — which eye the ratio is taken over is an arbitrary choice
  // and the fit derives the sign — but it has to be the same direction all the
  // way along, which is the part that is actually a property of the feature.
  const ws = [-0.4, -0.2, 0, 0.2, 0.4].map(y => extract(makeFace({ yaw: y })).widthRatio);
  const dir = Math.sign(ws[1] - ws[0]);
  check('widthRatio moves with yaw at all', dir !== 0);
  for (let i = 1; i < ws.length; i++) {
    check(`widthRatio is strictly monotonic across step ${i}`,
      Math.sign(ws[i] - ws[i - 1]) === dir, `${ws[i - 1]} -> ${ws[i]}`);
  }
  check('and it does not saturate at the ends',
    Math.abs(ws[4] - ws[3]) > Math.abs(ws[3] - ws[2]) * 0.5,
    ws.map(w => w.toFixed(4)).join(' '));
}

// ── THE FEATURE SETS ────────────────────────────────────────────────────────
{
  const f = extract(makeFace({ gx: 0.04, gy: 0.02 }), makeMatrix({ yaw: 0.2 }));
  for (const [name, set] of Object.entries(FEATURE_SETS)) {
    const row = designRow(f, name);
    check(`${name} row is the advertised width`, row.length === FEATURE_TERMS[name]);
    check(`${name} row is all finite`, row.every(Number.isFinite), JSON.stringify(row));
    check(`${name} starts with the intercept`, row[0] === 1);
    check(`${name} declares its need for pose correctly`, set.needsPose === (name === 'pose'));
  }

  // The flat set must not have quietly changed — it is the measured baseline
  // and a new idea does not get to move the thing it is being compared against.
  const flatRow = designRow(f, 'flat');
  check('flat is still the original eight terms', flatRow.length === 8);
  near('flat term 1 is still gx', flatRow[1], f.gx, 1e-12);
  near('flat term 6 is still hx', flatRow[6], f.hx, 1e-12);

  // A pose row without a matrix would be full of NaN. usable() is what keeps
  // those samples out of the fit, rather than letting them poison it.
  const noPose = extract(makeFace({}));
  check('a sample with no matrix is unusable for pose', !usable(noPose, 'pose'));
  check('  but still usable for flat', usable(noPose, 'flat'));
  check('pose is null rather than zeroed when there is no matrix', noPose.pose === null);
}

report('features');
