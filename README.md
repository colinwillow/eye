# eye

Eye tracking on a phone, off the front camera. Open the page, tap start, look
at nine dots, and a glowing dot follows your gaze around the screen.

**https://colinwillow.github.io/eye/** — needs HTTPS, which Pages gives you.
Nothing is uploaded; the camera frames are decoded on-device and never leave it.

Three things to do with it:

* **face** — your face with the eye contours and both irises drawn on, plus a
  stub out of each iris showing the raw offset the tracker is measuring. No
  calibration needed. This is the "is it actually seeing my eyes" view.
* **gaze** — a dot where it thinks you are looking.
* **paint** — the same dot, leaving a trail. Draw with your eyes.

A deliberate blink (longer than a normal one, shorter than a rest) fires a
ripple at wherever you are looking. It is not wired to anything yet — it is
there because a gaze tracker needs a click, and the eyelid is the only thing
left that is not already busy.

## How it works

MediaPipe's **FaceLandmarker** with the refined mesh: 478 points, of which the
last ten are the two irises. Those ten points are the whole signal. The
un-refined 468-point mesh gives you eyelids and eye corners — where your eyes
*are* — and tells you nothing about where they are pointed.

From there:

1. **Per eye**, take the iris centre against the midpoint of the two eye
   corners, projected onto the eye's own axis and divided by the eye's width.
   Projecting onto the eye's axis is what makes it survive a head tilt;
   dividing by the width is what makes it survive you moving the phone closer.
   Average the two eyes.
2. **Head pose**, as the nose tip against the midpoint between the eyes in the
   same frame. Your head does a lot of the aiming and the fit needs to know.
3. **Head pose**, properly, out of MediaPipe's facial transformation matrix —
   a real 3D pose solved against its canonical face model. Plus where the head
   *is* in the frame, which is a different question from which way it points
   and matters just as much.
4. **Calibrate**: nine dots held still, then four more where you keep looking
   at the dot and slowly move your head. About 22 seconds, then a
   ridge-regularised least-squares fit from those features to a screen
   position.
5. **Smooth** with a One Euro filter, which is heavy while you hold a gaze and
   light while you move — a single exponential smoother has to pick one.

The calibration is kept in `localStorage`, so a reload does not cost you
another thirteen seconds.

### Two models, one calibration, and a button that flips between them

The calibration fits **both** a `flat` model (eyes only, plus two crude head
proxies — the original) and a `pose` model (real head pose, head position, and
the cross terms that let head and eyes be told apart) from the *same* samples.
The **model** button switches between them live. That is the only honest way to
compare: two separate calibrations differ by how you sat and where the light
was, and those differences are bigger than the difference between the models.

Why it matters, and it is bigger than it sounds: **a phone at arm's length
subtends about 12 degrees.** Turn your head 17 — which is nothing, it is
glancing at someone beside you — and the point your eyes must aim at swings by
more than a screen width. Head movement is not a small perturbation on eye
movement here, it is the dominant term. In simulation, the flat model goes from
3.6% error with a still head to unusable with a moving one; the pose model
holds around 5%.

That is also why propping the phone against something transforms it. Not that
the tracker needs a still head — that it could not *see* the head move.

### What accuracy to expect

Somewhere around **2–5 cm on a phone at arm's length**. Enough to tell which
quadrant of the screen you are looking at, comfortably enough for a 3x3 grid of
targets, and not enough to pick a word out of a paragraph.

For scale: Google's 2020 smartphone eye tracker reached **0.46 cm**, and the
commercial mobile SDKs land around 1.7 degrees (≈1 cm at arm's length). Both
get there the same way, and it is not a better fit — they run a CNN over the
**eye-crop pixels**, trained on a lot of faces.

That is the real ceiling here. A landmark is an information bottleneck: the
iris points are trained to *outline the iris*, and the outline throws away the
corneal reflection, the exact curvature of the limbus and the eyelid shape —
all of which carry gaze. No amount of cleverness downstream recovers what the
landmark did not encode. Careful engineering on top of landmarks gets you to
maybe 1.5–2.5 cm; past that needs a different model.

The rest of the limits are physical:

* The iris is about 30 px across in a 720p frame, so one pixel of landmark
  noise is a visible wobble.
* A front camera sits above the screen, so vertical gaze is measured off a
  smaller range than horizontal and the eyelid hides part of it.

Sources: [Google / Nature Communications 2020](https://www.nature.com/articles/s41467-020-18360-5) ·
[SeeSo mobile SDK](https://www.prnewswire.com/news-releases/visualcamp-launches-mobile-eye-tracking-software-seeso-2-0--301078495.html)

## Running it

No build step. `index.html` opens and runs — native ES modules, the tracker
loaded from a pinned CDN URL.

```sh
npm start     # static server on :8123
npm test      # 273 headless checks: maths, directions, wiring
```

`npm test` runs in node with nothing installed. It has already caught three
real bugs — a ridge penalty quietly flattening the vertical axis, a relative
import resolving against the wrong base, and a calibration that collected 415
samples and then silently discarded every one of them.

```sh
npm run vendor        # pull MediaPipe + the model local (~40MB, gitignored)
npm run test:smoke    # the real page in real Chromium, needs the above
```

`tests/smoke.mjs` boots the actual page, loads the actual tracker, and drives a
complete calibration with synthetic landmarks — Chromium's fake camera is a
spinning ball, and a face is the one thing a headless box cannot supply.

### getUserMedia needs a secure context

HTTPS or `localhost`, no exceptions. `npm start` on `localhost` is fine;
the same server reached over your LAN at `http://192.168.x.x:8123` is not, and
the camera request fails with no useful error. Either use Pages, or a tunnel.

### On the phone

Add it to your home screen. It gets you full screen with no browser chrome,
which matters because the calibration maps to the *visible* area and the URL
bar sliding in and out moves it. The manifest and the apple-touch icons are
there for that.

## Layout

```
index.html          markup: video, three canvases, the bar
styles.css
src/
  main.js           boot, the loop, modes, the HUD
  tracker.js        the MediaPipe FaceLandmarker, and blink-as-an-event
  features.js       landmarks -> gaze features. All the sign conventions.
  calibrate.js      the two-pass sequence, both fits, persistence
  solve.js          ridge regression and a Gaussian elimination
  filter.js         One Euro
  draw.js           the overlay, the dot, the calibration dots
tests/              headless; run.mjs runs all of them
tools/
  make-icons.py     every icon, from one drawing, no dependencies
  vendor.sh         pull MediaPipe local for offline / smoke
```
