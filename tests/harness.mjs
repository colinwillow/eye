// Bare-bones assertions. No framework — these run in node with nothing installed.
let pass = 0, fail = 0;
const fails = [];

export function check(name, cond, detail = '') {
  if (cond) { pass++; }
  else { fail++; fails.push(`${name}${detail ? ' — ' + detail : ''}`); }
}

export function near(name, got, want, tol, detail = '') {
  const ok = Number.isFinite(got) && Math.abs(got - want) <= tol;
  check(name, ok, detail || `got ${got}, want ${want} +/- ${tol}`);
}

export function report(suite) {
  const total = pass + fail;
  if (fail) {
    console.log(`\n${suite}: ${pass}/${total} passed, ${fail} FAILED`);
    for (const f of fails) console.log('  x ' + f);
    process.exitCode = 1;
  } else {
    console.log(`${suite}: ${total}/${total} passed`);
  }
  return fail === 0;
}

// ── A synthetic face ────────────────────────────────────────────────────────
// Only the landmarks this project actually reads are filled in; the rest are
// there so the array is the right length, because extract() treats a short
// array as "this model has no iris refinement" and bails.
//
// The face frame has +x toward the subject's LEFT (which is +image x in an
// un-mirrored front-camera frame, see features.js), +y DOWN and +z OUT of the
// face toward the camera. Distances are in inter-ocular units.
//
// It is projected with a real perspective divide rather than orthographically,
// and that matters for one specific reason: under orthographic projection a
// yawed head foreshortens BOTH eyes by the same cos(yaw) and the two eyes stay
// in perfect agreement. The asymmetry the `asym` and `widthRatio` features
// exist to read is a perspective effect — the near eye is closer, so it is
// bigger — and a test rig without the divide would report those features as
// identically zero and quietly pass a suite that proved nothing about them.
//
// Everything except the nose sits at z = 0 in the face plane, so at yaw and
// pitch of zero the projection is a uniform scale and the parameters still map
// one-to-one onto the extracted features: `gx` in, `gx` out.
const EW = 0.34;           // eye width, in inter-ocular units
const NOSE_OUT = 0.35;     // how far the nose tip stands off the face plane
const CAM_D = 5.5;         // camera distance, in inter-ocular units (~35cm)

export function makeFace({
  gx = 0, gy = 0, hx = 0, hy = 0.55,
  yaw = 0, pitch = 0, roll = 0,
  scale = 0.22, cx = 0.5, cy = 0.42, ear = 0.30,
} = {}) {
  const lm = new Array(478);
  const cy_ = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);

  // Rotate about Y (yaw), then X (pitch), then Z (roll, in the image plane).
  const rot = (x, y, z) => {
    let X = x * cy_ + z * sy, Y = y, Z = -x * sy + z * cy_;
    let Y2 = Y * cp - Z * sp, Z2 = Y * sp + Z * cp;
    return { x: X * cr - Y2 * sr, y: X * sr + Y2 * cr, z: Z2 };
  };

  const put = (i, x, y, z = 0) => {
    const p = rot(x, y, z);
    const k = CAM_D / (CAM_D - p.z);         // the perspective divide
    lm[i] = { x: cx + p.x * k * scale, y: cy + p.y * k * scale, z: p.z };
  };

  const eyes = [
    { c: -0.5, idx: { a: 33, b: 133, top: 159, bot: 145, iris: 468, ring: [469, 470, 471, 472] } },
    { c: +0.5, idx: { a: 362, b: 263, top: 386, bot: 374, iris: 473, ring: [474, 475, 476, 477] } },
  ];
  for (const e of eyes) {
    put(e.idx.a, e.c - EW / 2, 0);
    put(e.idx.b, e.c + EW / 2, 0);
    put(e.idx.top, e.c, -ear * EW / 2);
    put(e.idx.bot, e.c, +ear * EW / 2);
    put(e.idx.iris, e.c + gx * EW, gy * EW);
    const r = EW * 0.22;
    put(e.idx.ring[0], e.c + gx * EW + r, gy * EW);
    put(e.idx.ring[1], e.c + gx * EW, gy * EW - r);
    put(e.idx.ring[2], e.c + gx * EW - r, gy * EW);
    put(e.idx.ring[3], e.c + gx * EW, gy * EW + r);
  }
  put(1, hx, hy, NOSE_OUT);                  // nose tip, standing off the face

  for (let i = 0; i < 478; i++) if (!lm[i]) lm[i] = { x: cx, y: cy, z: 0 };
  return lm;
}

// The 4x4 MediaPipe would hand back for that head, COLUMN-MAJOR: each column
// is one of the face's own axes expressed in camera space.
//
// This pins the DECOMPOSITION, not MediaPipe's axis convention — which is
// MediaPipe's business and is not knowable from a headless box with no face in
// front of the camera. It does not need to be: the angles feed a least-squares
// fit that derives their signs, so all that is required of them is that they
// move monotonically with head rotation and stay put when only the eyes move.
// Both of those are properties of any rotation representation, and both are
// checked below. The real convention is read off the HUD on a real phone.
export function makeMatrix({ yaw = 0, pitch = 0, roll = 0, dist = 35 } = {}) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);

  // R = Rz(roll) . Rx(pitch) . Ry(yaw), columns = the face's x, y, z axes.
  const col = (x, y, z) => {
    let X = x * cy + z * sy, Y = y, Z = -x * sy + z * cy;
    let Y2 = Y * cp - Z * sp, Z2 = Y * sp + Z * cp;
    return [X * cr - Y2 * sr, X * sr + Y2 * cr, Z2, 0];
  };
  return {
    rows: 4, columns: 4,
    data: [...col(1, 0, 0), ...col(0, 1, 0), ...col(0, 0, 1), 0, 0, dist, 1],
  };
}
