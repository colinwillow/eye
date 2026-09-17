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

// ── Head pose, properly ─────────────────────────────────────────────────────
// MediaPipe will hand back a 4x4 that maps its canonical face model onto the
// face it found, which is a real 3D head pose solved against a known head
// shape — far better than inferring one from where the nose landed. We were
// already asking for it in the options and then ignoring it.
//
// The matrix is COLUMN-MAJOR, so data[0..2], data[4..6] and data[8..10] are
// the face's own x, y and z axes expressed in camera space, and data[12..14]
// is where the head is.
//
// What comes out of here is deliberately convention-light: the direction the
// face points, in camera coordinates, rather than Euler angles in somebody's
// particular order. Two reasons. Euler extraction needs you to know the axis
// convention and the rotation order, and getting either wrong gives you angles
// that look plausible and are coupled to each other. And a direction vector is
// testable against a synthetic rotation without knowing anything about
// MediaPipe at all.
//
// The SIGNS are not asserted anywhere. These go into a least-squares fit,
// which derives them — same discipline as the iris offsets. All that is
// required of them is that they move monotonically with head rotation and do
// not move at all when only the eyes move.
export function matrixPose(matrix) {
  const m = matrix?.data;
  if (!m || m.length < 16) return null;

  // Third column: the face's forward axis, in camera space.
  const f = { x: m[8], y: m[9], z: m[10] };
  const n = Math.hypot(f.x, f.y, f.z);
  if (!(n > 1e-6)) return null;
  f.x /= n; f.y /= n; f.z /= n;

  // First column: the face's own right axis. Its tilt in the image plane is
  // roll, and it is measured separately so a tilted head does not leak into
  // the other two.
  const r = { x: m[0], y: m[1] };

  return {
    yaw: Math.atan2(f.x, Math.abs(f.z) > 1e-6 ? f.z : 1e-6),
    pitch: Math.asin(Math.max(-1, Math.min(1, -f.y))),
    roll: Math.atan2(r.y, r.x),
    // Distance from the camera. The canonical model is metric, so this is a
    // real length — but which unit is MediaPipe's business, and nothing here
    // needs to know: solveRidge scales every column by its own RMS, so a
    // feature measured in centimetres and the same feature in metres produce
    // identical predictions. The HUD prints it raw so it can be checked
    // against a tape measure on an actual phone.
    dist: Math.hypot(m[12], m[13], m[14]),
  };
}

// Everything the rest of the app needs out of one detection.
export function extract(lm, matrix = null) {
  if (!lm || lm.length < LANDMARKS_WITH_IRIS) return { ok: false, reason: lm ? 'no-iris' : 'no-face' };

  const L = eyeMetrics(lm, EYES.left);
  const R = eyeMetrics(lm, EYES.right);
  if (!L || !R) return { ok: false, reason: 'degenerate' };

  const head = headPose(lm, L, R);
  if (!head) return { ok: false, reason: 'degenerate' };

  // Average the two eyes. One eye alone works, but it is noticeably noisier.
  //
  // What the average THROWS AWAY is the difference between them, and that
  // difference is not noise — turn your head and the near eye foreshortens
  // while the far one does not, so the two iris offsets stop agreeing in a way
  // that depends only on yaw. `asym` keeps it, and `widthRatio` is the same
  // information read off the eye widths instead, which does not involve the
  // irises at all. Both are head-pose signals that need no matrix.
  const pose = matrixPose(matrix);

  // Where the head IS in the frame, as opposed to which way it is pointing.
  //
  // Every other feature here is deliberately translation-invariant — a face
  // crossing the frame is not a glance, and tests/features.mjs pins that. But
  // invariance is exactly wrong for the mapping to a SCREEN: move your head
  // ten centimetres to the left without turning it and every point on the
  // screen is at a different angle from your eye, while every rotation-based
  // feature reads identically. That is most of why propping the phone on
  // something works so much better than holding it — not that the tracker
  // needs a still head, but that it could not see the head move.
  //
  // So the position is passed through separately and only the `pose` set uses
  // it. `span` (inter-ocular width in the frame) is the third axis of the same
  // thing: it is how far away you are, measured off the image rather than off
  // the matrix, and it works when the matrix is missing.
  const centre = {
    x: (L.centre.x + R.centre.x) / 2,
    y: (L.centre.y + R.centre.y) / 2,
  };

  return {
    ok: true,
    gx: (L.offU + R.offU) / 2,
    gy: (L.offV + R.offV) / 2,
    asym: L.offU - R.offU,
    widthRatio: Math.log((L.width + 1e-9) / (R.width + 1e-9)),
    hx: head.hx,
    hy: head.hy,
    roll: head.roll,
    span: head.span,
    faceX: centre.x,
    faceY: centre.y,
    centre,
    ear: (L.ear != null && R.ear != null) ? (L.ear + R.ear) / 2 : null,
    // Null whenever the matrix is missing. Nothing may quietly substitute a
    // zero: a constant column is indistinguishable from the intercept, and the
    // fit would report a healthy residual for a model that had silently
    // stopped using head pose at all.
    pose,
    L, R,
  };
}

// ── The design row ──────────────────────────────────────────────────────────
// Two feature sets, fitted from the SAME collected samples so they can be
// compared on one calibration rather than two.
//
// `flat` is the original. Eye offsets, their quadratic terms, and the two
// crude nose-against-eyes head proxies. It is kept bit-identical on purpose:
// it is the version that was measured to work, and a new idea does not get to
// quietly replace the baseline it is supposed to beat.
//
// `pose` adds real head pose and, more importantly, the CROSS TERMS. Those are
// the whole point. The head terms on their own only let the fit say "your head
// moved, so shift the estimate"; gx*yaw lets it say "your head is turned, so
// the SAME iris offset means something different now" — which is what actually
// happens, because the iris offset is measured in the head's frame and the
// screen is not.
//
// Why the original had no cross terms: they were tried and they were DEAD. A
// calibration performed with a still head contains no head variation for them
// to explain, so ridge correctly shrank them to nothing. The terms were never
// the problem; the calibration was. Hence the head-movement pass in
// calibrate.js — the two changes only work as a pair, and adding either one
// alone measures as no improvement at all.
export const FEATURE_SETS = {
  flat: {
    needsPose: false,
    row: f => [1, f.gx, f.gy, f.gx * f.gy, f.gx * f.gx, f.gy * f.gy, f.hx, f.hy],
  },
  pose: {
    needsPose: true,
    row: f => {
      const { gx, gy, asym, widthRatio, faceX, faceY, span } = f;
      const { yaw, pitch, dist } = f.pose;
      return [
        1,
        gx, gy, gx * gy, gx * gx, gy * gy,   // where the eyes point
        asym, widthRatio,                     // head yaw, straight off the two eyes
        yaw, pitch, dist,                     // which way the head points
        faceX, faceY, span,                   // and where it actually is
        gx * yaw, gy * pitch,                 // the compensation that matters
        gx * pitch, gy * yaw,                 // and the off-diagonal half of it
      ];
    },
  },
};

export const FEATURE_TERMS = Object.fromEntries(
  Object.entries(FEATURE_SETS).map(([k, v]) => [k, v.row({
    gx: 0, gy: 0, hx: 0, hy: 0, asym: 0, widthRatio: 0,
    faceX: 0, faceY: 0, span: 0, pose: { yaw: 0, pitch: 0, dist: 0 },
  }).length]));

export function designRow(f, set = 'flat') {
  return FEATURE_SETS[set].row(f);
}

// Whether a sample carries everything a given set needs.
export function usable(f, set) {
  if (!f || !f.ok) return false;
  return !FEATURE_SETS[set].needsPose || !!f.pose;
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
