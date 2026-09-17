// Tiny dense linear algebra — enough to fit the gaze model and nothing more.
//
// The whole calibration is a least-squares fit from a handful of features
// (where the irises sit inside the eyes, plus where the head is pointing) to a
// screen coordinate. That is a 10x10 solve at the very worst, so there is no
// reason to pull in a matrix library.

// Gaussian elimination with partial pivoting. A is m x m, B is m x k.
// Both are arrays of rows and both are DESTROYED. Returns the m x k solution,
// or null if the system is singular.
export function solveLinear(A, B) {
  const m = A.length;
  const k = B[0].length;
  for (let col = 0; col < m; col++) {
    // Partial pivoting. Without it a zero on the diagonal — which happens the
    // moment a feature is constant across every sample, e.g. the head never
    // moved during calibration — divides by zero and poisons the whole fit
    // with NaN rather than failing loudly.
    let piv = col, best = Math.abs(A[col][col]);
    for (let r = col + 1; r < m; r++) {
      const v = Math.abs(A[r][col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-12) return null;
    if (piv !== col) { [A[col], A[piv]] = [A[piv], A[col]]; [B[col], B[piv]] = [B[piv], B[col]]; }

    const d = A[col][col];
    for (let c = col; c < m; c++) A[col][c] /= d;
    for (let c = 0; c < k; c++) B[col][c] /= d;

    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = A[r][col];
      if (f === 0) continue;
      for (let c = col; c < m; c++) A[r][c] -= f * A[col][c];
      for (let c = 0; c < k; c++) B[r][c] -= f * B[col][c];
    }
  }
  return B;
}

// Ridge regression. X is n x m (one design row per sample), Y is n x k
// (one target row per sample). Returns m x k coefficients, or null.
//
// Ridge rather than plain least squares because the design is nearly singular
// by construction: nine calibration points cannot independently excite eight
// polynomial terms, and a head that stayed still gives two dead columns. With
// lambda = 0 that comes back as a wild extrapolation that sends the dot off
// the screen the first time you look somewhere you did not calibrate. The
// penalty just shrinks the directions the data never pinned down.
//
// ── The columns are scaled first, and that is not a detail ──────────────────
// A ridge penalty is applied per column, so it only means the same thing to
// every column if the columns are the same size. They are nowhere near it
// here: the horizontal iris offset swings about +/-0.034 over a whole screen,
// the vertical one about +/-0.012, the quadratic terms about 1e-3, and the
// head-pose term sits near 0.55. One flat lambda across that lot barely
// touches the big columns and shrinks the small ones to almost nothing.
//
// Measured, un-scaled: the fit came out accurate horizontally and flattened
// vertically — a tracker that followed a glance left and right and would not
// follow one up and down. That reads as "vertical gaze does not work", which
// is a plausible enough sentence about eye tracking to be believed, and it
// sent the first version of this off hunting for a better vertical feature.
// It was the regulariser the whole time.
//
// So each column is divided by its own RMS before the solve and the
// coefficients are divided back out afterwards, which makes lambda a
// scale-free number. The intercept is left out of both the scaling and the
// penalty — shrinking it biases every prediction toward the origin, which on
// a screen is the top-left corner.
export function solveRidge(X, Y, lambda = 1e-4, intercept0 = true) {
  const n = X.length;
  if (n === 0) return null;
  const m = X[0].length, k = Y[0].length;

  const scale = new Array(m).fill(1);
  for (let j = 0; j < m; j++) {
    if (intercept0 && j === 0) continue;
    let ss = 0;
    for (let s = 0; s < n; s++) ss += X[s][j] * X[s][j];
    const rms = Math.sqrt(ss / n);
    scale[j] = rms > 1e-12 ? rms : 1;
  }

  const A = [], B = [];
  for (let i = 0; i < m; i++) { A.push(new Array(m).fill(0)); B.push(new Array(k).fill(0)); }

  for (let s = 0; s < n; s++) {
    const row = X[s], y = Y[s];
    for (let i = 0; i < m; i++) {
      const xi = row[i] / scale[i];
      for (let j = i; j < m; j++) A[i][j] += xi * (row[j] / scale[j]);
      for (let c = 0; c < k; c++) B[i][c] += xi * y[c];
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) A[i][j] = A[j][i];

  // Scale the penalty by n so lambda means the same thing whether you gave it
  // 100 samples or 1000.
  const lam = lambda * n;
  for (let i = 0; i < m; i++) if (!(intercept0 && i === 0)) A[i][i] += lam;

  const W = solveLinear(A, B);
  if (!W) return null;
  for (let i = 0; i < m; i++) for (let c = 0; c < k; c++) W[i][c] /= scale[i];
  return W;
}

// Y_hat = X . W for a single row.
export function apply(W, x) {
  const k = W[0].length, out = new Array(k).fill(0);
  for (let i = 0; i < x.length; i++) for (let c = 0; c < k; c++) out[c] += x[i] * W[i][c];
  return out;
}

// Root-mean-square error per output column, in the target's own units.
export function rmse(W, X, Y) {
  const k = Y[0].length, acc = new Array(k).fill(0);
  for (let s = 0; s < X.length; s++) {
    const p = apply(W, X[s]);
    for (let c = 0; c < k; c++) { const d = p[c] - Y[s][c]; acc[c] += d * d; }
  }
  return acc.map(v => Math.sqrt(v / X.length));
}
