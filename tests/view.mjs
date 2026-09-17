import { coverMap, CONTOUR, IRIS_RING, IRIS_CENTRE } from '../src/draw.js';
import { OneEuro, OneEuro2D } from '../src/filter.js';
import { check, near, report } from './harness.mjs';

// ── landmarks -> pixels, through object-fit: cover ──────────────────────────
// A phone screen is about 9:19 and a front camera frame is 16:9, so the video
// is cropped hard on one axis. Landmarks are normalised against the WHOLE
// frame including the cropped part. Drawing them as if the element were the
// frame puts the iris rings visibly beside the eyes — and it looks like bad
// tracking rather than like bad arithmetic.
{
  // 1280x720 into a 390x780 portrait box: scale by height is 1.083, by width
  // 0.305, so cover takes the larger and crops the sides.
  const m = coverMap(1280, 720, 390, 780, false);
  near('cover takes the larger scale', m.s, 780 / 720, 1e-9);
  check('the sides are cropped, not the top', m.ox < 0 && Math.abs(m.oy) < 1e-9);

  near('the frame centre lands at the box centre (x)', m.at({ x: 0.5, y: 0.5 }).x, 195, 1e-6);
  near('the frame centre lands at the box centre (y)', m.at({ x: 0.5, y: 0.5 }).y, 390, 1e-6);
  near('the frame top edge lands on the box top', m.at({ x: 0.5, y: 0 }).y, 0, 1e-6);
  near('the frame bottom edge lands on the box bottom', m.at({ x: 0.5, y: 1 }).y, 780, 1e-6);
  check('the frame left edge is cropped off the left', m.at({ x: 0, y: 0.5 }).x < 0);
}
{
  // Letterboxing the other way round: a tall frame in a wide box.
  const m = coverMap(720, 1280, 780, 390, false);
  check('the top and bottom are cropped instead', m.oy < 0 && Math.abs(m.ox) < 1e-9);
  near('the centre still lands centred', m.at({ x: 0.5, y: 0.5 }).x, 390, 1e-6);
}

// ── the mirror happens exactly once ─────────────────────────────────────────
// The video is flipped in CSS and the landmarks are flipped here. Flip neither
// and the overlay is on the wrong side of the face; flip both in one place and
// it is back where it started. The centre is the one point that cannot tell
// you which, so these check a point that is off-centre.
{
  const m = coverMap(640, 640, 400, 400, true);
  near('the centre is unmoved by mirroring', m.at({ x: 0.5, y: 0.5 }).x, 200, 1e-6);
  check('a point on the left of the frame draws on the right', m.at({ x: 0.2, y: 0.5 }).x > 200);
  check('a point on the right of the frame draws on the left', m.at({ x: 0.8, y: 0.5 }).x < 200);
  near('mirroring leaves y alone', m.at({ x: 0.2, y: 0.3 }).y, m.at({ x: 0.8, y: 0.3 }).y, 1e-9);

  const un = coverMap(640, 640, 400, 400, false);
  near('mirrored and un-mirrored are reflections', m.at({ x: 0.2, y: 0.5 }).x + un.at({ x: 0.2, y: 0.5 }).x, 400, 1e-9);
}

// ── the contour tables ──────────────────────────────────────────────────────
{
  for (const side of ['left', 'right']) {
    check(`${side} contour is a closed ring of 16`, CONTOUR[side].length === 16);
    check(`${side} contour has no repeats`, new Set(CONTOUR[side]).size === 16);
    check(`${side} iris ring is 4 points`, IRIS_RING[side].length === 4);
    check(`${side} iris indices are in the refined range`,
      [...IRIS_RING[side], IRIS_CENTRE[side]].every(i => i >= 468 && i < 478));
  }
  check('the two eyes share no landmarks', CONTOUR.left.every(i => !CONTOUR.right.includes(i)));
  check('the two irises share no landmarks', IRIS_RING.left.every(i => !IRIS_RING.right.includes(i)));
}

// ── One Euro ────────────────────────────────────────────────────────────────
{
  const f = new OneEuro();
  let y = 0;
  for (let i = 0; i < 200; i++) y = f.filter(7, 1 / 60);
  near('it converges to a held value', y, 7, 1e-3);
}
{
  // FRAME-RATE INDEPENDENCE. The same signal over the same wall-clock second
  // must land in the same place whether the phone managed 60fps or 24. A fixed
  // per-frame lerp does not do this, and the failure mode is that the tracker
  // feels sluggish exactly on the devices that are already struggling.
  const run = hz => {
    const f = new OneEuro();
    const dt = 1 / hz;
    let y = 0;
    for (let i = 0; i < hz; i++) y = f.filter(i < hz / 2 ? 0 : 1, dt);
    return y;
  };
  const a = run(60), b = run(24), c = run(120);
  near('60Hz and 24Hz agree after a second', a, b, 0.02, `${a} vs ${b}`);
  near('60Hz and 120Hz agree after a second', a, c, 0.02, `${a} vs ${c}`);
}
{
  // It has to let go when the signal moves. A smoother heavy enough to kill
  // gaze tremor is heavy enough to lag a glance by a third of a second, and
  // the whole reason for One Euro over an exponential is that it does not have
  // to pick one.
  const euro = new OneEuro({ minCutoff: 0.9, beta: 0.06 });
  const fixed = new OneEuro({ minCutoff: 0.9, beta: 0 });   // beta 0 == plain lowpass
  // Prime them both on a held zero first: the very first sample is taken as-is
  // (there is no history to blend with), so a step measured from cold is
  // instantaneous for every filter and compares nothing.
  let e = 0, x = 0;
  for (let i = 0; i < 60; i++) { euro.filter(0, 1 / 60); fixed.filter(0, 1 / 60); }
  for (let i = 0; i < 6; i++) { e = euro.filter(1, 1 / 60); x = fixed.filter(1, 1 / 60); }
  check('a fast move gets through faster than a plain lowpass', e > x, `${e} vs ${x}`);

  // ...and it is still quiet when nothing is happening.
  const still = new OneEuro({ minCutoff: 0.9, beta: 0.06 });
  let last = 0, maxJump = 0;
  for (let i = 0; i < 300; i++) {
    const v = still.filter(0.5 + Math.sin(i * 12.9898) * 0.004, 1 / 60);  // 0.4% jitter
    if (i > 60) maxJump = Math.max(maxJump, Math.abs(v - last));
    last = v;
  }
  check('held-still jitter is attenuated', maxJump < 0.0015, String(maxJump));
}
{
  const f = new OneEuro2D({ minCutoff: 1, beta: 0.05 });
  const p = f.filter({ x: 3, y: -2 }, 1 / 60);
  check('2D filters both axes independently', Number.isFinite(p.x) && Number.isFinite(p.y) && p.x !== p.y);
  f.reset();
  const q = f.filter({ x: 9, y: 9 }, 1 / 60);
  near('reset clears history', q.x, 9, 1e-9);
  // dt of zero happens on a doubled frame and must not divide by zero.
  check('a zero dt does not produce NaN', Number.isFinite(f.filter({ x: 1, y: 1 }, 0).x));
}

report('view');
