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
// un-mirrored front-camera frame, see features.js) and +y DOWN. Distances are
// in inter-ocular units, so a whole face is about 1 unit wide, and `scale`
// converts that to the normalised image.
//
// The parameters map one-to-one onto the extracted features by construction —
// `gx` in, `gx` out — which is what makes the direction checks below readable.
export function makeFace({ gx = 0, gy = 0, hx = 0, hy = 0.55, roll = 0, scale = 0.22, cx = 0.5, cy = 0.42, ear = 0.30 } = {}) {
  const EW = 0.34;                       // eye width, in inter-ocular units
  const lm = new Array(478);
  const c = Math.cos(roll), s = Math.sin(roll);
  const put = (i, x, y) => { lm[i] = { x: cx + (x * c - y * s) * scale, y: cy + (x * s + y * c) * scale, z: 0 }; };

  // eye centres: subject's right at -x, subject's left at +x
  const eyes = [
    { cx: -0.5, idx: { a: 33, b: 133, top: 159, bot: 145, iris: 468, ring: [469, 470, 471, 472] } },
    { cx: +0.5, idx: { a: 362, b: 263, top: 386, bot: 374, iris: 473, ring: [474, 475, 476, 477] } },
  ];
  for (const e of eyes) {
    put(e.idx.a, e.cx - EW / 2, 0);
    put(e.idx.b, e.cx + EW / 2, 0);
    put(e.idx.top, e.cx, -ear * EW / 2);
    put(e.idx.bot, e.cx, +ear * EW / 2);
    put(e.idx.iris, e.cx + gx * EW, gy * EW);
    const r = EW * 0.22;
    put(e.idx.ring[0], e.cx + gx * EW + r, gy * EW);
    put(e.idx.ring[1], e.cx + gx * EW, gy * EW - r);
    put(e.idx.ring[2], e.cx + gx * EW - r, gy * EW);
    put(e.idx.ring[3], e.cx + gx * EW, gy * EW + r);
  }
  put(1, hx, hy);                        // nose tip

  for (let i = 0; i < 478; i++) if (!lm[i]) lm[i] = { x: cx, y: cy, z: 0 };
  return lm;
}
