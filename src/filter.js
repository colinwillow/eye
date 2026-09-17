// One Euro filter (Casiez, Roussel, Vogel 2012).
//
// Gaze needs two contradictory things at once: it has to sit still when you
// stare, and it has to keep up when you flick across the screen. A plain
// exponential smoother can only trade one for the other — heavy enough to kill
// the tremor is heavy enough to lag a glance by a third of a second, which
// reads as the tracker being broken.
//
// One Euro adapts: the cutoff rises with the measured speed of the signal, so
// it is heavy while you hold and light while you move.
//
// Frame-rate independent by construction — alpha is recomputed from the real
// dt every sample, never a fixed per-frame lerp.
class LowPass {
  constructor() { this.y = null; }
  filter(x, alpha) {
    this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y;
    return this.y;
  }
  reset() { this.y = null; }
}

const alphaFor = (cutoff, dt) => {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
};

export class OneEuro {
  // minCutoff — how still it is when you hold still. Lower is steadier.
  // beta      — how fast it lets go when you move. Higher is more responsive.
  constructor({ minCutoff = 1.2, beta = 0.035, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = new LowPass(); this.dx = new LowPass();
    this.prev = null;
  }
  reset() { this.x.reset(); this.dx.reset(); this.prev = null; }
  filter(value, dt) {
    if (!(dt > 0)) dt = 1 / 60;
    const dv = this.prev === null ? 0 : (value - this.prev) / dt;
    this.prev = value;
    const edv = this.dx.filter(dv, alphaFor(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edv);
    return this.x.filter(value, alphaFor(cutoff, dt));
  }
}

export class OneEuro2D {
  constructor(opts) { this.fx = new OneEuro(opts); this.fy = new OneEuro(opts); }
  reset() { this.fx.reset(); this.fy.reset(); }
  filter(p, dt) { return { x: this.fx.filter(p.x, dt), y: this.fy.filter(p.y, dt) }; }
}
