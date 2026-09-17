// The MediaPipe FaceLandmarker, wrapped so the rest of the app never has to
// know about filesets, wasm or delegates.
//
// Why this model: the plain face mesh is 468 points and has no iris at all —
// it gives you eyelids and eye corners, which tells you where someone's EYES
// are but nothing about where they are LOOKING. The refined model adds ten
// points, five per iris, and those ten points are the entire signal.

// Loaded from a CDN rather than vendored: ~12MB of wasm plus a 3.7MB model
// would be most of this repository, and a CDN edge is faster to a phone than
// GitHub Pages is. The version is PINNED — a floating tag means the tracker
// can change under you between one look at your phone and the next, which is
// not a thing you want to be debugging alongside your own changes.
//
// `npm run vendor` pulls the same files into vendor/ and models/, and
// `?local=1` on the URL makes the page use them. That is the offline path and
// it is also how tests/smoke.mjs runs the real tracker with no network.
const VERSION = '1.0.1';
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// The local paths are resolved against the PAGE, not against this file.
//
// Two different resolvers would otherwise disagree about what './vendor' means
// and only one of them would be right:
//   - a dynamic import() resolves relative to the MODULE doing the importing,
//     so './vendor/...' from src/tracker.js asks for /src/vendor/... and 404s;
//   - MediaPipe fetches the wasm and the model itself, against the DOCUMENT
//     base, so the same string would have resolved correctly for those two.
// Making them absolute up front settles it, and it is also what makes this
// work under a GitHub Pages project subpath rather than only at a domain root.
const here = typeof document !== 'undefined' ? document.baseURI : 'http://localhost/';
const abs = p => new URL(p, here).href;

const local = typeof location !== 'undefined' && new URLSearchParams(location.search).get('local') === '1';

export const MP = {
  version: VERSION,
  local,
  wasmBase: local ? abs('vendor/tasks-vision/wasm') : `${CDN}/wasm`,
  bundle:   local ? abs('vendor/tasks-vision/vision_bundle.mjs') : `${CDN}/vision_bundle.mjs`,
  model:    local ? abs('models/face_landmarker.task') : MODEL,
};

export class Tracker {
  constructor() {
    this.landmarker = null;
    this.delegate = null;
    this.lastVideoTime = -1;
    this.result = null;
  }

  async load(onStatus = () => {}) {
    onStatus('loading the tracker…');
    const { FilesetResolver, FaceLandmarker } = await import(MP.bundle);
    this.FaceLandmarker = FaceLandmarker;

    onStatus('loading wasm…');
    const fileset = await FilesetResolver.forVisionTasks(MP.wasmBase);

    const opts = delegate => ({
      baseOptions: { modelAssetPath: MP.model, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      // The blendshapes are what give a clean blink. A geometric eye-aspect
      // ratio works too and is the fallback, but it needs a per-face threshold
      // because eye shape varies more than you would expect; the blendshape is
      // already normalised for that.
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });

    // GPU first, CPU if the phone refuses. On a mid-range Android the CPU path
    // is roughly a third of the frame rate but it does still run, and a slow
    // tracker beats a blank screen.
    try {
      onStatus('loading the model (GPU)…');
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
      this.delegate = 'GPU';
    } catch (e) {
      onStatus('GPU refused, falling back to CPU…');
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
      this.delegate = 'CPU';
    }
    return this;
  }

  // Run detection, but only when the video has actually advanced. rAF fires
  // at 60Hz and a phone's front camera delivers 30 at best, so without this
  // gate half of every frame budget is spent re-detecting a frame that has
  // not changed.
  detect(video, nowMs) {
    if (!this.landmarker || video.readyState < 2) return this.result;
    if (video.currentTime === this.lastVideoTime) return this.result;
    this.lastVideoTime = video.currentTime;
    try {
      this.result = this.landmarker.detectForVideo(video, nowMs);
      this.fresh = true;
    } catch (e) {
      // A single bad frame (a resize mid-detect, a backgrounded tab) should not
      // take the whole loop down.
      this.error = e;
      this.fresh = false;
    }
    return this.result;
  }

  get landmarks() {
    const r = this.result;
    return r && r.faceLandmarks && r.faceLandmarks.length ? r.faceLandmarks[0] : null;
  }

  // 0..1, and it is the max of the two eyes rather than the mean: a real blink
  // closes both, and taking the max means a wink still registers instead of
  // being averaged down into nothing.
  get blinkScore() {
    const bs = this.result?.faceBlendshapes?.[0]?.categories;
    if (!bs) return null;
    let v = 0;
    for (const c of bs) if (c.categoryName === 'eyeBlinkLeft' || c.categoryName === 'eyeBlinkRight') v = Math.max(v, c.score);
    return v;
  }

  close() { try { this.landmarker?.close(); } catch {} this.landmarker = null; }
}

// ── Blink, as an event rather than a level ──────────────────────────────────
// An involuntary blink is 100-150ms; a deliberate one is longer. Gating on
// duration is what separates "he blinked" from "he clicked", and without the
// upper bound simply closing your eyes to rest fires a click a second.
export class BlinkDetector {
  constructor({ on = 0.5, off = 0.35, minMs = 180, maxMs = 800 } = {}) {
    Object.assign(this, { on, off, minMs, maxMs });
    this.closed = false;
    this.since = 0;
  }
  // Returns 'click' on the release of a deliberate blink, else null.
  update(score, nowMs) {
    if (score == null) return null;
    if (!this.closed && score > this.on) { this.closed = true; this.since = nowMs; return null; }
    if (this.closed && score < this.off) {
      this.closed = false;
      const held = nowMs - this.since;
      return (held >= this.minMs && held <= this.maxMs) ? 'click' : null;
    }
    return null;
  }
}
