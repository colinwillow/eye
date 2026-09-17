import { solveLinear, solveRidge, apply, rmse } from '../src/solve.js';
import { check, near, report } from './harness.mjs';

// ── the elimination itself ──────────────────────────────────────────────────
{
  // 2x + y = 5 ; x - 3y = -8   ->  x = 1, y = 3
  const X = solveLinear([[2, 1], [1, -3]], [[5], [-8]]);
  near('solveLinear x', X[0][0], 1, 1e-12);
  near('solveLinear y', X[1][0], 3, 1e-12);
}
{
  // A zero on the diagonal. Without partial pivoting this divides by zero and
  // returns a matrix of NaN, which then flows all the way to the screen as a
  // dot that has simply disappeared.
  const X = solveLinear([[0, 2], [3, 0]], [[4], [9]]);
  check('a zero pivot is handled', X !== null && Number.isFinite(X[0][0]));
  near('  and still solves', X[0][0], 3, 1e-12);
  near('  both ways', X[1][0], 2, 1e-12);
}
{
  check('a singular system returns null rather than NaN', solveLinear([[1, 2], [2, 4]], [[1], [2]]) === null);
}

// ── ridge recovers a known linear model ─────────────────────────────────────
{
  // y0 = 1 + 2a - 3b ; y1 = -0.5 + 0.25a + 4b
  const truth = [[1, -0.5], [2, 0.25], [-3, 4]];
  const X = [], Y = [];
  for (let i = 0; i < 60; i++) {
    const a = Math.sin(i * 1.7), b = Math.cos(i * 0.9);
    const row = [1, a, b];
    X.push(row);
    Y.push([apply(truth, row)[0], apply(truth, row)[1]]);
  }
  const W = solveRidge(X, Y, 1e-9);
  for (let i = 0; i < 3; i++) for (let c = 0; c < 2; c++) near(`ridge recovers W[${i}][${c}]`, W[i][c], truth[i][c], 1e-4);
  const e = rmse(W, X, Y);
  check('residual on an exact fit is ~0', e[0] < 1e-5 && e[1] < 1e-5, e.join(','));
}

// ── the penalty actually shrinks, and leaves the intercept alone ────────────
{
  const X = [], Y = [];
  for (let i = 0; i < 40; i++) { const a = Math.sin(i); X.push([1, a]); Y.push([10 + 5 * a]); }
  const loose = solveRidge(X, Y, 1e-9);
  const tight = solveRidge(X, Y, 5);
  check('a big lambda shrinks the slope', Math.abs(tight[1][0]) < Math.abs(loose[1][0]) * 0.5,
    `${tight[1][0]} vs ${loose[1][0]}`);
  // Penalising the intercept pulls every prediction toward zero, which on a
  // screen means toward the top-left corner. It is a systematic offset that
  // looks exactly like a bad calibration.
  near('the intercept is NOT shrunk', tight[0][0], 10, 0.2, `intercept = ${tight[0][0]}`);
}

// ── a dead column does not take the whole fit down ──────────────────────────
{
  // Calibrating with a perfectly still head leaves the head columns constant,
  // which is collinear with the intercept. Unridged this is singular; the
  // whole reason for the penalty is that this is the NORMAL case, not an
  // edge case.
  const X = [], Y = [];
  for (let i = 0; i < 40; i++) { const a = Math.sin(i); X.push([1, a, 0.55]); Y.push([3 + 2 * a]); }
  check('an unridged dead column is singular', solveRidge(X, Y, 0) === null);
  const W = solveRidge(X, Y, 1e-4);
  check('a ridged dead column still fits', W !== null);
  check('  and the fit is still good', rmse(W, X, Y)[0] < 0.05, String(rmse(W, X, Y)[0]));
}


// ── COLUMN OFFSET INVARIANCE ────────────────────────────────────────────────
// The second half of the same lesson, and it hid behind the fix for the first.
// Dividing by RMS made the penalty independent of a feature's UNITS. It left
// it dependent on where the feature's ZERO happens to sit, because what a
// ridge penalty really costs a column is its ability to explain variation, and
// that is its standard deviation, not its RMS. Those are the same number only
// for a zero-mean column.
//
// Adding a constant to a feature changes nothing about the information in it,
// so it must change nothing about the prediction. Before centring it changed a
// great deal: measured on the real design, std/RMS was 0.015 for `span`, so it
// was penalised 67x harder than `gx` purely for being measured from a distant
// origin, and its fitted contribution came out at 0.0002. It was switched off,
// and the feature it switched off was the one that notices you sliding your
// head sideways.
//
// Note what this would still pass with if it only checked that the fit works:
// everything. The offset version fitted fine and reported a healthy residual —
// it had just stopped using several of its features.
{
  const X = [], Xoff = [], Y = [];
  for (let i = 0; i < 80; i++) {
    const a = Math.sin(i * 1.3), b = Math.cos(i * 0.7) * 0.01;
    X.push([1, a, b]);
    Xoff.push([1, a + 50, b + 1000]);        // same features, distant origins
    Y.push([2 + 0.5 * a + 30 * b]);
  }
  const W = solveRidge(X, Y, 1e-3);
  const Wo = solveRidge(Xoff, Y, 1e-3);
  for (let i = 0; i < X.length; i += 17) {
    near('offsetting a column does not change the prediction', apply(W, X[i])[0], apply(Wo, Xoff[i])[0], 1e-6);
  }
  check('an offset feature is still actually fitted', rmse(Wo, Xoff, Y)[0] < 0.02, String(rmse(Wo, Xoff, Y)[0]));

  // Both at once, which is the real case: every head feature has its own units
  // AND its own origin.
  const Wb = solveRidge(Xoff.map(r => [r[0], r[1] * 100, r[2] * 0.001]), Y, 1e-3);
  near('offset and rescaled together still agrees',
    apply(Wb, [1, (X[3][1] + 50) * 100, (X[3][2] + 1000) * 0.001])[0], apply(W, X[3])[0], 1e-6);
}

// ── A COLUMN THAT NEVER VARIES DROPS OUT ────────────────────────────────────
// Centring turns a constant column into exactly zero, so it contributes
// nothing and takes a zero coefficient. Uncentred it was a second intercept,
// collinear with the first, and the two split a coefficient between them —
// which is fine for the fit and confusing for anybody reading the weights to
// work out what the model is using.
{
  const X = [], Y = [];
  for (let i = 0; i < 40; i++) { const a = Math.sin(i); X.push([1, a, 7]); Y.push([3 + 2 * a]); }
  const W = solveRidge(X, Y, 1e-3);
  near('a constant column takes a zero coefficient', W[2][0], 0, 1e-9);
  near('  and the intercept keeps the whole offset', W[0][0], 3, 0.02);
  check('  and the fit is unharmed', rmse(W, X, Y)[0] < 0.02, String(rmse(W, X, Y)[0]));
}

// ── COLUMN SCALE INVARIANCE ─────────────────────────────────────────────────
// This is the check the first version did not have, and its absence cost the
// vertical axis. A ridge penalty is per-column, so a feature measured in small
// units gets shrunk harder than the identical feature measured in large ones —
// the fit stops being a property of the data and starts being a property of
// the units. Multiplying a column by a thousand must not move a prediction.
//
// Note what this would still pass with if it only asserted "the fit works":
// everything. The un-scaled version fitted fine, it just quietly ignored the
// smallest feature, which on screen read as "up and down does not track".
{
  const X = [], Xbig = [], Y = [];
  for (let i = 0; i < 80; i++) {
    const a = Math.sin(i * 1.3), b = Math.cos(i * 0.7) * 0.001;   // b is tiny
    X.push([1, a, b]);
    Xbig.push([1, a, b * 1000]);                                   // same feature, bigger units
    Y.push([2 + 0.5 * a + 300 * b]);
  }
  const W = solveRidge(X, Y, 1e-3);
  const Wb = solveRidge(Xbig, Y, 1e-3);
  for (let i = 0; i < X.length; i += 17) {
    near('rescaling a column does not change the prediction', apply(W, X[i])[0], apply(Wb, Xbig[i])[0], 1e-9);
  }

  // And the small column is genuinely used, rather than being shrunk away.
  const flat = solveRidge(X.map(r => [r[0], r[1], 0]), Y, 1e-3);
  check('the small-unit feature is actually fitted',
    rmse(W, X, Y)[0] < rmse(flat, X.map(r => [r[0], r[1], 0]), Y)[0] * 0.25,
    `${rmse(W, X, Y)[0]} vs ${rmse(flat, X.map(r => [r[0], r[1], 0]), Y)[0]}`);
}

report('solve');
