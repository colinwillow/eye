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
3. **Calibrate**: nine dots, ~13 seconds, then one ridge-regularised
   least-squares fit from those features to a screen position. Second order,
   so it can cope with the fact that eye rotation maps to screen position
   through a tangent and the camera is above the screen rather than behind it.
4. **Smooth** with a One Euro filter, which is heavy while you hold a gaze and
   light while you move — a single exponential smoother has to pick one.

The calibration is kept in `localStorage`, so a reload does not cost you
another thirteen seconds.

### What accuracy to expect

Somewhere around **2–5 cm on a phone at arm's length**, degrading as you move
your head away from where you calibrated. It is enough to tell which quadrant
of the screen you are looking at, comfortably enough for a 3x3 grid of targets,
and not enough to pick a word out of a paragraph.

The limits are physical, not fixable by better code:

* The iris is about 30 px across in a 720p frame, so one pixel of landmark
  noise is a visible wobble.
* A front camera sits above the screen, so vertical gaze is measured off a
  smaller range than horizontal and the eyelid hides part of it.
* Everything is relative to the head, so leaning changes the mapping. Big
  moves want a recalibration.

If you want it dramatically better, the answer is not a smarter fit — it is
dedicated hardware pointed at the eye from close up, which is exactly what the
Quest Pro has and the Quest 3 does not.

## Running it

No build step. `index.html` opens and runs — native ES modules, the tracker
loaded from a pinned CDN URL.

```sh
npm start     # static server on :8123
npm test      # 171 headless checks: maths, directions, wiring
```

`npm test` runs in node with nothing installed. It has already caught two real
bugs — a ridge penalty that was quietly flattening the vertical axis, and a
relative import resolving against the wrong base.

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
  calibrate.js      the nine-dot sequence, the fit, persistence
  solve.js          ridge regression and a Gaussian elimination
  filter.js         One Euro
  draw.js           the overlay, the dot, the calibration dots
tests/              headless; run.mjs runs all of them
tools/
  make-icons.py     every icon, from one drawing, no dependencies
  vendor.sh         pull MediaPipe local for offline / smoke
```
