// All the rendering. Two surfaces: an overlay pinned to the camera picture,
// and a full-screen stage the gaze dot and the paint live on.

// The eye contours, as MediaPipe orders them around the lid. These are the
// same rings FaceLandmarker exposes as FACE_LANDMARKS_*_EYE; they are inlined
// so this module does not need the landmarker instance to draw.
export const CONTOUR = {
  left:  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  right: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
};
export const IRIS_RING = { left: [474, 475, 476, 477], right: [469, 470, 471, 472] };
export const IRIS_CENTRE = { left: 473, right: 468 };

// ── Camera picture -> element pixels ────────────────────────────────────────
// The video is drawn with object-fit: cover, so it is scaled up until it fills
// the box and the overflow is cropped off the sides. Landmarks are normalised
// against the WHOLE frame including the cropped part, so drawing them straight
// onto the element puts them visibly off the face the moment the box is not
// the same aspect ratio as the camera — which on a phone it never is.
//
// Mirrored, because an un-mirrored selfie view is disorienting enough that you
// cannot tell whether the tracker is wrong or you are.
export function coverMap(vw, vh, bw, bh, mirror = true) {
  const s = Math.max(bw / vw, bh / vh);
  const ox = (bw - vw * s) / 2, oy = (bh - vh * s) / 2;
  return {
    s, ox, oy, mirror, bw,
    at(p) {
      const x = ox + p.x * vw * s;
      return { x: mirror ? bw - x : x, y: oy + p.y * vh * s };
    },
  };
}

export function sizeCanvas(canvas, dpr = Math.min(window.devicePixelRatio || 1, 2)) {
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height, dpr };
}

const path = (ctx, pts) => {
  ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
};

// Fit a circle to the four iris ring points. Drawing the quad itself reads as
// a diamond stuck on the eye; a circle reads as an iris.
function irisCircle(lm, map, side) {
  const c = map.at(lm[IRIS_CENTRE[side]]);
  let r = 0;
  for (const i of IRIS_RING[side]) { const p = map.at(lm[i]); r += Math.hypot(p.x - c.x, p.y - c.y); }
  return { c, r: r / IRIS_RING[side].length };
}

export function drawFace(ctx, lm, map, { mesh = false, features = null } = {}) {
  if (mesh) {
    ctx.fillStyle = 'rgba(120,200,255,0.16)';
    for (let i = 0; i < lm.length; i++) {
      const p = map.at(lm[i]);
      ctx.fillRect(p.x - 0.6, p.y - 0.6, 1.2, 1.2);
    }
  }

  for (const side of ['left', 'right']) {
    ctx.strokeStyle = 'rgba(120,220,255,0.85)';
    ctx.lineWidth = 1.6;
    path(ctx, CONTOUR[side].map(i => map.at(lm[i])));
    ctx.closePath();
    ctx.stroke();

    const { c, r } = irisCircle(lm, map, side);
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(r, 2), 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#ffd166';
    ctx.beginPath(); ctx.arc(c.x, c.y, 2.1, 0, Math.PI * 2); ctx.fill();
  }

  // A stub out of each iris along the measured offset — the raw signal, drawn.
  // If this points the wrong way the whole thing is wrong, and it is the
  // fastest way to see that without a calibration in the way.
  if (features?.ok) {
    ctx.strokeStyle = 'rgba(255,209,102,0.7)';
    ctx.lineWidth = 2;
    for (const [side, m] of [['left', features.L], ['right', features.R]]) {
      const { c, r } = irisCircle(lm, map, side);
      const k = r * 9;
      // offU/offV are in the eye's own frame; put them back on screen through
      // that same frame, then let the map's mirror flip x exactly once.
      const dx = (m.offU * m.u.x + m.offV * m.v.x) * k;
      const dy = (m.offU * m.u.y + m.offV * m.v.y) * k;
      ctx.beginPath();
      ctx.moveTo(c.x, c.y);
      ctx.lineTo(c.x + (map.mirror ? -dx : dx), c.y + dy);
      ctx.stroke();
    }
  }
}

// ── The gaze dot ────────────────────────────────────────────────────────────
export function drawGaze(ctx, p, { w, h, confident = true, blink = false, trail = [] }) {
  if (trail.length > 1) {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (let i = 1; i < trail.length; i++) {
      const a = i / trail.length;
      ctx.strokeStyle = `rgba(120,220,255,${0.32 * a * a})`;
      ctx.lineWidth = 2 + 10 * a;
      ctx.beginPath(); ctx.moveTo(trail[i - 1].x, trail[i - 1].y); ctx.lineTo(trail[i].x, trail[i].y); ctx.stroke();
    }
  }

  const R = blink ? 10 : 22;
  const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, R * 2.6);
  g.addColorStop(0, confident ? 'rgba(130,230,255,0.95)' : 'rgba(255,190,120,0.9)');
  g.addColorStop(0.35, confident ? 'rgba(90,180,255,0.35)' : 'rgba(255,160,90,0.3)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(p.x, p.y, R * 2.6, 0, Math.PI * 2); ctx.fill();

  ctx.strokeStyle = confident ? 'rgba(200,245,255,0.95)' : 'rgba(255,215,160,0.95)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = '#eaf9ff';
  ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill();
}

// A calibration dot: a ring that closes while the eyes settle, then a core
// that fills while samples are taken. Two visibly different phases, because
// "hold still, it is taking this one now" has to be readable without text.
export function drawCalDot(ctx, p, phase, t) {
  const R = 26;
  ctx.save();
  ctx.translate(p.x, p.y);

  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(0, 0, R, 0, Math.PI * 2); ctx.stroke();

  if (phase === 'settle') {
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(0, 0, R + 14 * (1 - t), 0, Math.PI * 2); ctx.stroke();
  } else {
    ctx.strokeStyle = '#7ce0ff';
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.arc(0, 0, R, -Math.PI / 2, -Math.PI / 2 + t * Math.PI * 2); ctx.stroke();
  }

  ctx.fillStyle = phase === 'collect' ? '#7ce0ff' : '#ffffff';
  ctx.beginPath(); ctx.arc(0, 0, phase === 'collect' ? 9 : 6, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}
