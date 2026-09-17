// Landmarks in, gaze features out. Pure maths, no DOM — so tests/features.mjs
// can feed it synthetic faces and pin down DIRECTION, which is the thing that
// silently goes wrong here.
//
// ── Coordinate space ────────────────────────────────────────────────────────
// MediaPipe hands back normalised IMAGE coordinates: x right, y DOWN, both
// 0..1 across the raw camera frame. The frame from a front camera is NOT
// mirrored — it is you as other people see you — so:
//
//     the subject's LEFT side of their face sits at HIGH image x.
//
// Everything in this file stays in image space. The mirror happens once, in
// the renderer, and the calibration fit learns whatever sign it needs. That
// ordering matters: the one thing you must never do here is guess a
// handedness, because a sign error in a gaze signal looks exactly like bad
// tracking rather than like a bug.

// MediaPipe's own labels are anatomical (its "left eye" is the subject's left
// eye, which appears on the RIGHT of an un-mirrored frame). Nothing below
// depends on that being true — which eye is which is re-derived from image x
// at runtime, and the two eyes are averaged anyway. The names are for the HUD.
export const EYES = {
  left:  { cornerA: 362, cornerB: 263, lidTop: 386, lidBottom: 374, iris: 473, irisRing: [474, 475, 476, 477] },
  right: { cornerA: 33,  cornerB: 133, lidTop: 159, lidBottom: 145, iris: 468, irisRing: [469, 470, 471, 472] },
};

export const NOSE_TIP = 1;

// A face mesh without the iris refinement has 468 points; with it, 478. The
// iris points are the last ten and they are the entire reason this works, so
// a 468-point result is a hard failure, not a degraded one.
export const LANDMARKS_WITH_IRIS = 478;

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const len = v => Math.hypot(v.x, v.y);
const dot = (a, b) => a.x * b.x + a.y * b.y;

// One eye, measured in its OWN frame rather than the image's.
//
// u runs along the eye slit from the low-x corner to the high-x corner, so u
// points roughly along +image x whichever corner index happens to be which.
// v is u turned a quarter turn, and since y is down that lands v pointing
// DOWN. Projecting onto (u, v) instead of onto (x, y) is what makes the
// signal survive a head tilt: roll the head 20 degrees and the raw image
// offsets swap axes, while offU/offV do not move at all.
//
// Both offsets are divided by the eye's width, which is what makes them
// independent of how far away the phone is being held.
export function eyeMetrics(lm, eye) {
  const a = lm[eye.cornerA], b = lm[eye.cornerB];
  if (!a || !b) return null;

  // Order the corners by image x rather than by which index is "outer".
  const [p0, p1] = a.x <= b.x ? [a, b] : [b, a];
  const w = len(sub(p1, p0));
  if (!(w > 1e-6)) return null;

  const u = { x: (p1.x - p0.x) / w, y: (p1.y - p0.y) / w };
  const v = { x: -u.y, y: u.x };

  const c = { x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2 };
  const iris = lm[eye.iris];
  if (!iris) return null;
  const d = sub(iris, c);

  // Lid gap over eye width — the classic eye-aspect-ratio blink signal. Also
  // projected onto v, so a tilted head does not read as a half-closed eye.
  const top = lm[eye.lidTop], bot = lm[eye.lidBottom];
  const ear = top && bot ? Math.abs(dot(sub(bot, top), v)) / w : null;

  return {
    offU: dot(d, u) / w,   // + = iris toward +image x = subject looking to THEIR left
    offV: dot(d, v) / w,   // + = iris down
    ear,
    width: w,
    centre: c,
    iris,
    u, v,
  };
}

// Head pose without touching the transformation matrix.
//
// The nose tip's position relative to the midpoint between the eyes, measured
// in the inter-ocular frame and scaled by the inter-ocular distance. Turn your
// head toward +image x and the nose swings that way ahead of the eye midpoint,
// so hx shares a sign with offU — the two add rather than fight, which is what
// you want feeding a linear fit.
//
// hy carries a large constant offset (the nose is always below the eyes). That
// is harmless: the fit's intercept absorbs it and only the variation matters.
export function headPose(lm, L, R) {
  const nose = lm[NOSE_TIP];
  if (!nose || !L || !R) return null;

  const [e0, e1] = L.centre.x <= R.centre.x ? [L.centre, R.centre] : [R.centre, L.centre];
  const span = len(sub(e1, e0));
  if (!(span > 1e-6)) return null;

  const u = { x: (e1.x - e0.x) / span, y: (e1.y - e0.y) / span };
  const v = { x: -u.y, y: u.x };
  const mid = { x: (e0.x + e1.x) / 2, y: (e0.y + e1.y) / 2 };
  const d = sub(nose, mid);

  return {
    hx: dot(d, u) / span,     // + = head turned toward +image x
    hy: dot(d, v) / span,     // + = nose lower in the eye frame (looking down)
    roll: Math.atan2(u.y, u.x),
    span,                     // inter-ocular, in image widths: a distance proxy
  };
}

// Everything the rest of the app needs out of one detection.
export function extract(lm) {
  if (!lm || lm.length < LANDMARKS_WITH_IRIS) return { ok: false, reason: lm ? 'no-iris' : 'no-face' };

  const L = eyeMetrics(lm, EYES.left);
  const R = eyeMetrics(lm, EYES.right);
  if (!L || !R) return { ok: false, reason: 'degenerate' };

  const head = headPose(lm, L, R);
  if (!head) return { ok: false, reason: 'degenerate' };

  // Average the two eyes. One eye alone works, but it is noticeably noisier
  // and it picks up the asymmetry of a face turned off-axis.
  return {
    ok: true,
    gx: (L.offU + R.offU) / 2,
    gy: (L.offV + R.offV) / 2,
    hx: head.hx,
    hy: head.hy,
    roll: head.roll,
    span: head.span,
    ear: (L.ear != null && R.ear != null) ? (L.ear + R.ear) / 2 : null,
    L, R,
  };
}

// ── The design row ──────────────────────────────────────────────────────────
// Order 1 is a plane through the features; order 2 adds the quadratic terms
// that soak up the fact that eye rotation maps to screen position through a
// tangent, not a line, and that the camera sits above the screen rather than
// behind it.
//
// Order 2 is eight terms against nine calibration points, which is why the fit
// is ridged. Ten terms (adding gx*hx, gy*hy) was tried and the head-cross
// columns are dead — a calibration is done with a still head, so there is no
// head variation for them to explain and ridge just shrinks them back to zero.
export const ORDER_TERMS = { 1: 5, 2: 8 };

export function designRow(f, order = 2) {
  const { gx, gy, hx, hy } = f;
  if (order === 1) return [1, gx, gy, hx, hy];
  return [1, gx, gy, gx * gy, gx * gx, gy * gy, hx, hy];
}

// ── The uncalibrated mapping ────────────────────────────────────────────────
// A rough guess so the dot does something the instant a face appears, before
// anybody has sat through a calibration. It is not accurate and the HUD says
// so; its job is to prove the signal is alive and pointing the right way.
//
// The signs are derived, not tuned:
//   +gx  = iris toward +image x = subject looking to their own LEFT
//        = the LEFT of a mirrored (selfie) view = a SMALLER screen x.  -> minus
//   +gy  = iris down = down the screen = a LARGER screen y (y is down). -> plus
//
// Gains are eyeballed from the range a comfortable glance actually covers:
// about +/-0.10 of eye width horizontally and a good deal less vertically,
// because eyes move less up and down than side to side and the eyelid hides
// most of what they do.
export const RAW_GAIN = { x: 4.6, y: 8.0, headX: 1.3, headY: 1.6 };

export function rawMapping(f, rest, gain = RAW_GAIN) {
  const dgx = f.gx - rest.gx, dgy = f.gy - rest.gy;
  const dhx = f.hx - rest.hx, dhy = f.hy - rest.hy;
  return {
    x: 0.5 - (gain.x * dgx + gain.headX * dhx),
    y: 0.5 + (gain.y * dgy + gain.headY * dhy),
  };
}
