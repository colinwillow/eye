# Working on eye

Gaze tracking off a phone's front camera. MediaPipe's iris landmarks, a
nine-dot calibration, a dot that follows your eyes.

## Anything meant for testing goes to `main`

Pages serves `main`, and a phone is the only place this can actually be tested
— there is no camera on a laptop worth pointing at this, and no camera at all
in a headless box. So a change sitting on a branch cannot be looked at, which
means it is not done.

Branch as much as you like while working. What does not work is *ending* a
requested change on a branch. Merge to `main` and push before reporting back.
Don't open a pull request unless asked.

## Test before you push

```sh
npm test                              # ~171 checks, node only, ~2s
npm run vendor && npm run test:smoke   # the real page in real Chromium, ~40s
```

`npm test` needs nothing installed. `test:smoke` needs `npm run vendor` first
(it pulls ~40MB of MediaPipe into gitignored directories) and a Chromium.

Both have already earned their keep. `npm test` caught the ridge bug below;
`test:smoke` caught a relative import resolving against the wrong base, which
was a blank screen and an unhelpful error.

### When you add a check, ask what it would still pass with

A gaze tracker with a flipped sign tracks *beautifully* and points the wrong
way. Every accuracy number is a distance, and a mirrored model satisfies every
distance you can measure on a symmetric test grid. So most of the checks in
`tests/features.mjs` and `tests/calibrate.mjs` exist to pin a DIRECTION or an
INVARIANCE, and they say so at the top of each block. A suite that only
measured error would have passed the whole time.

## Handedness: derive it, never guess it

There are three sign conventions in play and they all have to agree:

* MediaPipe hands back **image** coordinates: x right, y **down**, against the
  raw frame. A front camera frame is **not** mirrored, so **the subject's left
  side of their face is at HIGH image x**.
* The view is mirrored (a selfie view — an un-mirrored one is disorienting
  enough that you cannot tell whether the tracker is wrong or you are). So
  looking at the LEFT of the screen means looking to your own left, means the
  irises move to HIGH image x, means a positive `gx`, means a screen x BELOW
  0.5. Horizontal flips; vertical does not.
* The **mirror happens exactly once**, in `draw.js`'s `coverMap`. The video is
  flipped in CSS and the landmarks are flipped there to match. Remove one and
  the overlay is on the wrong side of the face.

The design answer to all of this: **stay in image space everywhere, and let
the calibration fit learn whatever sign it needs.** It is least squares from
features to screen position, so it derives handedness from data. Nothing
downstream of a calibration can be mirrored, and `tests/calibrate.mjs` proves
it by fitting an eyeball wired deliberately backwards and checking the result
still points the right way.

Two smaller versions of the same rule, both already in the code:

* `eyeMetrics` sorts the two eye corners by image x rather than trusting which
  landmark index is the "outer" one.
* The two eyes are averaged, so which one MediaPipe calls "left" never matters.

The only place a sign is asserted rather than derived is `rawMapping`, the
uncalibrated preview, and it is labelled UNCALIBRATED on the HUD for that
reason. If it ever moves the wrong way on a real phone, flip `RAW_GAIN.x` —
it does not affect anything calibrated.

## A ridge penalty is per-column, so the columns must be scaled

This cost a whole debugging pass and it is the kind of bug that produces a
*plausible sentence*. The features are wildly different sizes: horizontal iris
offset swings about +/-0.034 across a whole screen, vertical about +/-0.012,
the quadratic terms about 1e-3, and head pose sits near 0.55. One flat lambda
across that barely touches the big columns and shrinks the small ones to
nothing.

Measured: the fit came out accurate horizontally and flat vertically. That
reads as "vertical gaze tracking does not work", which is believable enough
about eye tracking that the first version went looking for a better vertical
feature. It was the regulariser.

`solveRidge` now divides each column by its own RMS before solving and divides
it back out afterwards, which makes lambda scale-free. The intercept is left
out of both the scaling and the penalty — shrinking it biases every prediction
toward the origin, which on a screen is the top-left corner.
`tests/solve.mjs` pins this: rescaling a column must not change a prediction.

## Conventions

- **Everything is normalised.** Landmarks 0..1 against the frame, gaze 0..1
  against the screen, offsets in eye widths. Pixels appear only in `draw.js`.
- **No build step.** Native ES modules, no bundler, no TypeScript.
  `index.html` opens and runs. Don't introduce one without asking.
- **Frame-rate independent smoothing.** `OneEuro` recomputes alpha from the
  real dt every sample. Never a constant per-frame lerp.
- **Tuning lives in one object per system** — `TUNING` in `main.js`, `CAL` in
  `calibrate.js`, `MP` in `tracker.js`, `RAW_GAIN` in `features.js`. Put new
  numbers there, not inline.
- **MediaPipe's version is pinned.** A floating tag means the tracker can
  change under you between one look at the phone and the next, which is not
  something you want to debug alongside your own change.

## Traps that already bit

* **A relative path in a dynamic `import()` resolves against the MODULE, not
  the page** — but MediaPipe fetches its own wasm and model against the
  DOCUMENT base. Two resolvers, two answers, one of them a 404. `MP` builds
  absolute URLs off `document.baseURI`, which is also what makes it work under
  a Pages project subpath.
* **`object-fit: cover` crops the frame, and landmarks are against the whole
  frame including the cropped part.** A phone is 9:19 and a camera is 16:9, so
  this is always wrong if you ignore it, and it looks like bad tracking rather
  than bad arithmetic. `coverMap`.
* **Nothing may be drawn over a calibration dot.** The top-left dot is at 12%
  of the screen, which is exactly where the HUD was. A dot you cannot see
  records you looking at whatever is covering it. `body.calibrating` hides the
  HUD, the bar and the camera preview; `tests/smoke.mjs` checks it for real
  with `elementsFromPoint`.
* **A 468-point mesh is not a degraded 478-point one, it is useless.** Every
  eye-shaped measurement still works and the gaze reads as a constant zero.
  `extract` rejects it by length.
* **`play()` rejecting on iOS is not a failure.** The await on getUserMedia
  has already closed the user-gesture window and Safari counts that against an
  explicit `play()`; the autoplay attribute starts it anyway. What matters is
  whether frames arrive, which is what `waitForFrames` measures.

## Open seams

* **The blink click goes nowhere.** It fires a ripple. It is the obvious hook
  for picking something.
* **No dwell selection**, which is the other half of a usable gaze UI — hold
  the gaze inside a radius for N ms and commit.
* **No head-movement compensation beyond the two pose features.** Leaning a
  long way from where you calibrated drifts, and the honest fix is either
  recalibrating or a much better head model.
* **Nothing is wired to the games yet.** The output is a normalised 0..1 point
  and a click, which is the same shape as a pointer — `peggy`, `BigDon` and
  `robits` all take a stick, not a point, so something has to convert.
