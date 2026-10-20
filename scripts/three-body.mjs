#!/usr/bin/env node
// Writes a littlewood-fractals/1 settings file (docs/settings-format.md) whose three coefficients move
// like the three bodies of a periodic solution of the planar three-body problem.
//
// Physics: G = m1 = m2 = m3 = 1, Newtonian gravity, classical RK4 with fixed substeps. The initial conditions
// are the collinear family of Šuvakov & Dmitrašinović (PRL 110, 114301, 2013):
//   x1 = (-1, 0), x2 = (1, 0), x3 = (0, 0),  v1 = v2 = (p1, p2), v3 = -2 (p1, p2),  period T.
// They are published to 6 decimals, so the orbit does not close exactly after T. The small closure
// error is spread linearly over the orbit (x(t) -= (t/T)(x(T) - x(0))), and the last keyframe is a
// copy of the first, so the first and last keyframes are identical.
//
// Keyframes sample the orbit at equal TIME steps, so bodies speed up and slow down as they would
// physically. Body k becomes coefficient k: z = scale * (x + i y) / R + offset, R = the largest
// distance of any body from the centre of mass over the orbit.
//
// Usage: node scripts/three-body.mjs [--orbit figure8] [--keyframes 1440] [--degree 13] [--fps 60]
//          [--frames N] [--scale 1] [--offset re,im] [--phase 0] [--substeps auto] [--rings] [--out path]
//        node scripts/three-body.mjs --list
//   --degree  polynomial degree d (1..32). Render cost grows as 3^(d+1) per frame: d=13 is 4.8 M polynomials.
//   --frames  default = keyframes (linear, no loop: N = m(K-1)+1 hits every keyframe).
//   --scale / --offset  only the coefficient ratios shape the fractal; an offset breaks the sum-zero symmetry.
//   --phase   start this fraction of a period into the orbit (default 0). At phase 0 one body sits exactly at the
//             origin (a 0 coefficient), so frame 0 is atypical; 'locked' brightness is set by frame 0.

import { writeFileSync } from 'node:fs';

const ORBITS = {
  // name: [p1, p2, T]
  figure8: [0.347111, 0.532728, 6.324449], // Chenciner–Montgomery figure-eight
  butterfly1: [0.306893, 0.125507, 6.235641],
  butterfly2: [0.392955, 0.097579, 7.003707],
  butterfly3: [0.405916, 0.230163, 13.865763],
  moth1: [0.464445, 0.39606, 14.893911],
  moth2: [0.439166, 0.452968, 28.670765],
  moth3: [0.383444, 0.377364, 25.840545],
  goggles: [0.0833, 0.127889, 10.466818],
  bumblebee: [0.184279, 0.587188, 63.534541],
  dragonfly: [0.080584, 0.588836, 21.270975],
  yinyang1a: [0.513938, 0.304736, 17.328369],
  yinyang1b: [0.282699, 0.327209, 10.963252],
};

function parseArgs(argv) {
  const o = { orbit: 'figure8', keyframes: 1440, degree: 13, fps: 60, frames: null, scale: 1, offset: [0, 0],
    substeps: null, substepsUsed: null, phase: 0, rings: false, out: null, list: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    switch (a) {
      case '--orbit': o.orbit = next(); break;
      case '--keyframes': o.keyframes = int(next(), a, 2); break;
      case '--degree': o.degree = int(next(), a, 1, 32); break;
      case '--fps': o.fps = int(next(), a, 1, 120); break;
      case '--frames': o.frames = int(next(), a, 1, 100000); break;
      case '--scale': o.scale = num(next(), a); break;
      case '--offset': o.offset = next().split(',').map((s) => num(s, a)); if (o.offset.length !== 2) throw new Error('--offset wants re,im'); break;
      case '--substeps': o.substeps = int(next(), a, 1); break;
      case '--phase': o.phase = num(next(), a); if (o.phase < 0 || o.phase >= 1) throw new Error('--phase must be in [0, 1)'); break;
      case '--rings': o.rings = true; break;
      case '--out': o.out = next(); break;
      case '--list': o.list = true; break;
      default: throw new Error(`unknown argument ${a}`);
    }
  }
  if (!(o.orbit in ORBITS)) throw new Error(`unknown orbit ${o.orbit}; --list shows them`);
  return o;
}
function num(s, flag) {
  const v = Number(s);
  if (!Number.isFinite(v)) throw new Error(`${flag}: ${s} is not a number`);
  return v;
}
function int(s, flag, min, max = Infinity) {
  const v = num(s, flag);
  if (!Number.isInteger(v) || v < min || v > max) throw new Error(`${flag}: ${s} must be a whole number in ${min}..${max}`);
  return v;
}

/** state = [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3] */
function deriv(s, d) {
  for (let k = 0; k < 6; k++) d[k] = s[k + 6];
  for (let k = 6; k < 12; k++) d[k] = 0;
  for (let a = 0; a < 3; a++) {
    for (let b = a + 1; b < 3; b++) {
      const dx = s[2 * b] - s[2 * a], dy = s[2 * b + 1] - s[2 * a + 1];
      const r2 = dx * dx + dy * dy, inv = 1 / (r2 * Math.sqrt(r2));
      d[6 + 2 * a] += dx * inv; d[7 + 2 * a] += dy * inv;
      d[6 + 2 * b] -= dx * inv; d[7 + 2 * b] -= dy * inv;
    }
  }
}

function rk4Step(s, h, k1, k2, k3, k4, tmp) {
  deriv(s, k1);
  for (let i = 0; i < 12; i++) tmp[i] = s[i] + 0.5 * h * k1[i];
  deriv(tmp, k2);
  for (let i = 0; i < 12; i++) tmp[i] = s[i] + 0.5 * h * k2[i];
  deriv(tmp, k3);
  for (let i = 0; i < 12; i++) tmp[i] = s[i] + h * k3[i];
  deriv(tmp, k4);
  for (let i = 0; i < 12; i++) s[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

/** Positions [x1,y1,x2,y2,x3,y3] at t = i·T/(n-1), i = 0..n-1, plus the full-state closure error at T. */
function simulate([p1, p2, T], n, substeps) {
  const s = Float64Array.from([-1, 0, 1, 0, 0, 0, p1, p2, p1, p2, -2 * p1, -2 * p2]);
  const s0 = Float64Array.from(s);
  const k1 = new Float64Array(12), k2 = new Float64Array(12), k3 = new Float64Array(12), k4 = new Float64Array(12), tmp = new Float64Array(12);
  const h = T / (n - 1) / substeps;
  const out = [Array.from(s.subarray(0, 6))];
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < substeps; j++) rk4Step(s, h, k1, k2, k3, k4, tmp);
    out.push(Array.from(s.subarray(0, 6)));
  }
  const posErr = Math.hypot(...out[n - 1].map((v, k) => v - out[0][k]));
  const velErr = Math.hypot(...Array.from(s.subarray(6)).map((v, k) => v - s0[k + 6]));
  return { pos: out, posErr, velErr };
}

function main() {
  const o = parseArgs(process.argv.slice(2));
  if (o.list) {
    for (const [name, [p1, p2, T]] of Object.entries(ORBITS)) console.log(`${name.padEnd(11)} p1=${p1} p2=${p2} T=${T}`);
    return;
  }
  const K = o.keyframes;
  // Orbits with close encounters need finer steps. Without --substeps, double them until the closure error
  // stops changing: what is left is the 6-decimal rounding of the published initial conditions, not the integrator.
  let run = simulate(ORBITS[o.orbit], K, o.substeps ?? 100);
  if (o.substeps === null) {
    for (let s = 200; s <= 12800; s *= 2) {
      const finer = simulate(ORBITS[o.orbit], K, s);
      const settled = Math.abs(finer.posErr - run.posErr) < 0.1 * finer.posErr;
      run = finer;
      o.substepsUsed = s;
      if (settled) break;
    }
  }
  const { posErr, velErr } = run;
  let pos = run.pos;

  // Spread the closure error linearly over the orbit, then make the last keyframe exactly the first.
  const drift = pos[K - 1].map((v, k) => v - pos[0][k]);
  for (let i = 0; i < K; i++) for (let k = 0; k < 6; k++) pos[i][k] -= (i / (K - 1)) * drift[k];
  pos[K - 1] = pos[0].slice();

  // Start the orbit later by a whole number of keyframe steps (the samples are periodic now, so this is exact).
  const shift = Math.round(o.phase * (K - 1)) % (K - 1);
  if (shift !== 0) {
    pos = Array.from({ length: K - 1 }, (_, i) => pos[(i + shift) % (K - 1)]);
    pos.push(pos[0].slice());
  }

  let R = 0;
  for (const p of pos) for (let b = 0; b < 3; b++) R = Math.max(R, Math.hypot(p[2 * b], p[2 * b + 1]));
  const f = o.scale / R;
  const coeffsAt = (p) => [0, 1, 2].map((b) => ({ re: p[2 * b] * f + o.offset[0], im: p[2 * b + 1] * f + o.offset[1] }));
  const keyframes = pos.map((p) => ({ coeffs: coeffsAt(p) }));
  keyframes[K - 1] = { coeffs: keyframes[0].coeffs.map((c) => ({ ...c })) };

  // Linear easing, no loop: N = m(K-1)+1 frames put a frame exactly on every keyframe; m = 1 by default.
  const frames = o.frames ?? K;
  const now = new Date().toISOString();
  const tag = `three-body.mjs (${o.orbit})`;
  const doc = {
    schema: 'littlewood-fractals/1',
    kind: 'settings',
    build: { version: 'external', commit: tag, builtAt: now },
    environment: { gpu: { vendor: '', architecture: '', device: '', description: '' }, userAgent: tag, timestamp: now },
    fractal: { coeffs: keyframes[0].coeffs, degree: o.degree, maxIter: 32, tol: 1e-5, pinned: false, ordering: 'golden',
      kernel: 'perPoly', enumeration: 'reciprocal' },
    screen: { cx: 0, cy: 0, pxPerUnit: 270, width: 1920, height: 1080, dpr: 1 },
    output: { cx: 0, cy: 0, pxPerUnit: 270, width: 1920, height: 1080 },
    display: { colorMode: 1, exposure: 1, gamma: 0.6, countScale: 1, rings: o.rings },
    animation: { keyframes, frames, fps: o.fps, loop: false, easing: 'linear', brightness: 'locked', quality: 1,
      batchFrames: 8, output: 'mp4', codec: 'avc', bitrate: null },
  };

  const out = o.out ?? `three-body-${o.orbit}-d${o.degree}-${K}kf.settings.json`;
  writeFileSync(out, JSON.stringify(doc, null, 1) + '\n');
  console.log(`wrote ${out}`);
  console.log(`  orbit ${o.orbit}, ${K} keyframes, ${frames} frames at ${o.fps} fps (${(frames / o.fps).toFixed(1)} s), degree ${o.degree}`);
  console.log(`  RK4 substeps per keyframe: ${o.substeps ?? o.substepsUsed}`);
  console.log(`  closure error after one period (before correction): position ${posErr.toExponential(2)}, velocity ${velErr.toExponential(2)}`);
  if (posErr > 0.05) console.log('  warning: large closure error; the orbit is unstable at this precision, try more --substeps or another orbit');
}

try {
  main();
} catch (e) {
  console.error(`three-body: ${e.message}`);
  process.exit(1);
}
