import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine, type Enumeration } from '../../src/gpu/engine';
import { countSum, nodeDevice, readBuffer, runToEnd } from './gpuHelpers';

// Reciprocal enumeration (core/reciprocal.ts) against the plain path: every root deposited exactly once, so with
// every root inside the view the root counts and the colour sums are identical; pixels differ only where 1/z in f32
// lands one pixel away from the reversed polynomial's directly solved root. That is at most 1e-3 of the roots for
// sets without multiple roots. Multiple roots (Littlewood: +-1, roots of unity) are resolved by f32 only to ~1e-2,
// so there the reference is the noise between the two plain kernels on the very same polynomials (measured on
// Arc 140T: reciprocal-vs-plain 0.3-0.8x of perPoly-vs-coop, e.g. Littlewood d=10 1.8e-2 vs 2.2e-2).
const { device, errors } = await nodeDevice();
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => {
  engine.destroy();
  device.destroy();
});

const c = (re: number, im = 0): Complex => ({ re, im });
const rot = (k: number, j: number, c0: Complex): Complex => {
  const t = (2 * Math.PI * j) / k;
  return c(c0.re * Math.cos(t) - c0.im * Math.sin(t), c0.re * Math.sin(t) + c0.im * Math.cos(t));
};
const KERNELS: Kernel[] = ['perPoly', 'coop'];
const SIZE = 160;

// Every root of these sets has 1/2.3 < |z| < 2.3 (or is 0), so all land in a radius-3 view.
const CASES: { name: string; coeffs: Complex[]; degree: number; pinned: boolean }[] = [
  { name: 'Littlewood d=10 (pinned {±1})', coeffs: [c(1), c(-1)], degree: 10, pinned: true },
  { name: 'Littlewood d=9 (pinned, odd degree)', coeffs: [c(1), c(-1)], degree: 9, pinned: true },
  { name: '3 complex coefficients d=8 (unpinned)', coeffs: [c(1), c(-0.5, 0.8), c(0.3, -1.1)], degree: 8, pinned: false },
  { name: '{0, 1, -1} d=8 (contains 0: zero roots both ways)', coeffs: [c(0), c(1), c(-1)], degree: 8, pinned: false },
  { name: '{1, i, -1, -i} d=7 (pinned, twisted pairs)', coeffs: [c(1), c(0, 1), c(-1), c(0, -1)], degree: 7, pinned: true },
  { name: '(0.5+0.3i) x 6th roots, shuffled, d=6 (pinned)', coeffs: [0, 4, 1, 3, 5, 2].map((j) => rot(6, j, c(0.5, 0.3))), degree: 6, pinned: true },
];

async function render(kernel: Kernel, enumeration: Enumeration, coeffs: Complex[], degree: number, budget = 1 << 20) {
  engine.kernel = kernel;
  engine.enumeration = enumeration;
  engine.resize(SIZE, SIZE);
  engine.setParams({ coeffs, degree, view: fitView(SIZE, SIZE, 3), maxIter: 32 });
  const work = engine.work.total;
  await runToEnd(device, engine, budget);
  expect(engine.progress.done).toBe(engine.progress.total);
  return { hist: await engine.readHistogram(), unconverged: (await engine.readStats()).unconverged, work, polys: engine.progress.total, mode: engine.activeEnumeration };
}

/** Sums of count, r, g, b over all pixels. */
function channelSums(h: Uint32Array): number[] {
  const s = [0, 0, 0, 0];
  for (let i = 0; i < h.length; i++) s[i & 3] += h[i];
  return s;
}

function countL1(a: Uint32Array, b: Uint32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i += 4) d += Math.abs(a[i] - b[i]);
  return d;
}

describe.each(KERNELS)('reciprocal vs plain enumeration, %s kernel', (kernel) => {
  it.each(CASES)('$name: same root count and colour sums, same picture, about half the solves', async ({ coeffs, degree, pinned }) => {
    const plain = await render(kernel, 'plain', coeffs, degree);
    const recip = await render(kernel, 'reciprocal', coeffs, degree, 777); // many chunks: entries span regions
    expect(recip.mode).toBe('reciprocal');
    const k = BigInt(coeffs.length);
    expect(recip.polys).toBe(k ** BigInt(pinned ? degree : degree + 1));
    expect(plain.polys).toBe(recip.polys);
    expect(plain.work).toBe(plain.polys);
    expect(Number(recip.work)).toBeLessThan(Number(recip.polys) * 0.55); // low degrees have few layers
    const n = countSum(plain.hist);
    expect(n).toBeGreaterThan(Number(plain.polys) * (degree - 1)); // (nearly) every root landed
    // Exact: every root deposited once, with the partner's own colour (the hue of the reversed digits).
    expect(channelSums(recip.hist)).toEqual(channelSums(plain.hist));
    const other = await render(kernel === 'perPoly' ? 'coop' : 'perPoly', 'plain', coeffs, degree);
    expect(countL1(recip.hist, plain.hist)).toBeLessThanOrEqual(Math.max(n * 1e-3, 1.5 * countL1(other.hist, plain.hist)));
    expect(recip.unconverged).toBe(plain.unconverged);
    expect(errors).toEqual([]);
  });
});

describe('reciprocal splatFrames (animation batches)', () => {
  const lerp = (a: Complex, b: Complex, t: number): Complex => c(a.re + (b.re - a.re) * t, a.im + (b.im - a.im) * t);
  const MOVING: Complex[][] = Array.from({ length: 4 }, (_, f) => [c(1), lerp(c(-0.9, 0.15), c(-0.3, 0.8), f / 3), c(0)]);
  const PINNED: Complex[][] = Array.from({ length: 4 }, (_, f) => [0, 1, 2, 3].map((j) => rot(4, j, c(1 + 0.1 * f, 0.2 * f))));

  async function batch(frames: Complex[][], degree: number, pinned: boolean, enumeration: Enumeration) {
    const frameBytes = SIZE * SIZE * 16;
    const target = device.createBuffer({ size: frames.length * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    engine.enumeration = enumeration;
    engine.resize(SIZE, SIZE);
    engine.setFrames({ frames, degree, view: fitView(SIZE, SIZE, 3), maxIter: 32, pinned }, target);
    const mode = engine.activeEnumeration;
    await runToEnd(device, engine, 1000);
    const all = new Uint32Array(await readBuffer(device, target, frames.length * frameBytes));
    target.destroy();
    return { mode, unconverged: (await engine.readStats()).unconverged, hists: frames.map((_, f) => all.subarray((f * frameBytes) / 4, ((f + 1) * frameBytes) / 4)) };
  }

  it.each([
    { name: 'unpinned {1, c, 0}, c moving', frames: MOVING, degree: 8, pinned: false },
    { name: 'pinned c0 x {1, i, -1, -i}, c0 moving (shared group table)', frames: PINNED, degree: 7, pinned: true },
  ])('$name: every frame has the plain batch root count and colour sums', async ({ frames, degree, pinned }) => {
    const plain = await batch(frames, degree, pinned, 'plain');
    const recip = await batch(frames, degree, pinned, 'reciprocal');
    expect(recip.mode).toBe('reciprocal');
    expect(recip.unconverged).toBe(plain.unconverged);
    for (let f = 0; f < frames.length; f++) {
      const n = countSum(plain.hists[f]);
      expect(channelSums(recip.hists[f])).toEqual(channelSums(plain.hists[f]));
      // Warm starts add f32 edge noise; the reference noise is again plain perPoly vs plain coop on frame f.
      const floor = countL1((await render('perPoly', 'plain', frames[f], degree)).hist, (await render('coop', 'plain', frames[f], degree)).hist);
      expect(countL1(recip.hists[f], plain.hists[f])).toBeLessThanOrEqual(Math.max(n * 2e-3, 1.5 * floor));
    }
    expect(errors).toEqual([]);
  });

  it('pinned frames whose group tables differ run plain', async () => {
    const swapped = [PINNED[0], [0, 2, 1, 3].map((j) => PINNED[1][j])]; // same group, other digit order
    expect((await batch(swapped, 5, true, 'reciprocal')).mode).toBe('plain');
    expect(errors).toEqual([]);
  });
});
