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
// ── The columns are STANDARDISED first, and that is not a detail ────────────
// A ridge penalty is applied per column, so it only means the same thing to
// every column if the columns are comparable. They are nowhere near it here.
//
// This went wrong twice, in two different ways, and the second one hid behind
// the fix for the first.
//
// **Scale.** The features are wildly different sizes: horizontal iris offset
// swings about +/-0.02 over a whole screen, the quadratic terms about 1e-3,
// head distance sits near 35. One flat lambda barely touches the big columns
// and shrinks the small ones to nothing. Measured, the fit came out accurate
// horizontally and flat vertically — which reads as "vertical gaze tracking
// does not work", believable enough about eye tracking that the first version
// went hunting for a better vertical feature. It was the regulariser.
//
// **Offset.** Dividing by RMS fixed that and left a subtler version of it
// standing. What a ridge penalty actually costs is the column's ability to
// EXPLAIN VARIATION, and a column's variation is its standard deviation, not
// its RMS. For a zero-mean column like `gx` those are the same number. For a
// column with a big constant offset they are not, and the features that carry
// head POSITION are all offset: `faceX` sits near 0.5 and moves by 0.03,
// `span` sits near 0.22 and moves by 0.003.
//
// Measured, dividing by RMS: std/RMS was 0.069 for faceX, 0.065 for faceY and
// 0.015 for span — so those three were penalised 14x, 15x and 67x harder than
// gx, for no reason but where their zero happens to sit. span's fitted
// contribution came to 0.0002, which is nothing. They are precisely the
// features that answer "you moved your head sideways", which is what a phone
// held in a hand does constantly, and they were switched off.
//
// So: centre each column, divide by its standard deviation, solve, then fold
// the shift back into the intercept. Lambda is now free of both the units and
// the origin a feature happens to be measured from, and
// `tests/solve.mjs` pins both invariances.
//
// A bonus from centring: a column that never varies becomes exactly zero, so
// it drops out with a zero coefficient instead of quietly duplicating the
// intercept and splitting a coefficient with it.
export function solveRidge(X, Y, lambda = 1e-4, intercept0 = true) {
  const n = X.length;
  if (n === 0) return null;
  const m = X[0].length, k = Y[0].length;

  const mu = new Array(m).fill(0);
  const sd = new Array(m).fill(1);
  for (let j = 0; j < m; j++) {
    if (intercept0 && j === 0) continue;
    let sum = 0;
    for (let s = 0; s < n; s++) sum += X[s][j];
    mu[j] = sum / n;
    let ss = 0;
    for (let s = 0; s < n; s++) { const d = X[s][j] - mu[j]; ss += d * d; }
    const dev = Math.sqrt(ss / n);
    sd[j] = dev > 1e-12 ? dev : 1;
  }
  const z = (row, j) => (intercept0 && j === 0) ? row[0] : (row[j] - mu[j]) / sd[j];

  const A = [], B = [];
  for (let i = 0; i < m; i++) { A.push(new Array(m).fill(0)); B.push(new Array(k).fill(0)); }

  for (let s = 0; s < n; s++) {
    const row = X[s], y = Y[s];
    for (let i = 0; i < m; i++) {
      const zi = z(row, i);
      for (let j = i; j < m; j++) A[i][j] += zi * z(row, j);
      for (let c = 0; c < k; c++) B[i][c] += zi * y[c];
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) A[i][j] = A[j][i];

  // Scale the penalty by n so lambda means the same thing whether you gave it
  // 100 samples or 1000. The intercept is not penalised — shrinking it biases
  // every prediction toward the origin, which on a screen is the top-left
  // corner.
  const lam = lambda * n;
  for (let i = 0; i < m; i++) if (!(intercept0 && i === 0)) A[i][i] += lam;

  const W = solveLinear(A, B);
  if (!W) return null;

  // Undo the standardisation so the caller gets coefficients against the raw
  // features:  w0 + SUM wj (xj - muj)/sdj  ==  (w0 - SUM Wj muj) + SUM Wj xj
  for (let c = 0; c < k; c++) {
    let shift = 0;
    for (let j = 0; j < m; j++) {
      if (intercept0 && j === 0) continue;
      W[j][c] /= sd[j];
      shift += W[j][c] * mu[j];
    }
    if (intercept0) W[0][c] -= shift;
  }
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
