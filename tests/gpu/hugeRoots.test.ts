import { afterAll, describe, expect, it } from 'vitest';
import { aberthRef } from '../../src/core/aberthRef';
import type { Complex } from '../../src/core/complex';
import { indexToDigits } from '../../src/core/enumeration';
import type { Kernel } from '../../src/core/schedule';
import { fitView, type View } from '../../src/core/view';
import { Engine, renderImage, type Enumeration } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// Big roots (|z| >> 1) of sets with a large coefficient ratio, e.g. {1, 0.01}: 0.01 z^12 + z^11 + ... has a root
// near -100. Evaluating p(z) there overflowed f32 (|p|^2 in the backward-error stop became inf <= inf), so such roots
// were "converged" wherever they were. Off-screen in the plain enumeration, but reciprocal symmetry deposits them as
// 1/z: a ring of wrong dots around the origin (user report). polish() now evaluates the reversed polynomial at 1/z
// for |z| > 1 (roots.wgsl, coop.wgsl). Every test here failed before that fix.
const { device, errors } = await nodeDevice();
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => {
  engine.destroy();
  device.destroy();
});

const SET: Complex[] = [{ re: 1, im: 0 }, { re: 0.01, im: 0 }];
const DEGREE = 12;
const POLYS = 2 ** (DEGREE + 1); // 8192: all of them, the f64 reference takes ~1 s
const PX = 250; // px per unit of the near-origin view: 1/z must be right to one pixel

let refCache: Complex[][] | null = null;
/** f64 roots of every polynomial, lexicographic (index n: digit i picks a_i). */
function reference(): Complex[][] {
  refCache ??= Array.from({ length: POLYS }, (_, n) => aberthRef(Array.from(indexToDigits(BigInt(n), SET.length, DEGREE), (g) => SET[g])));
  return refCache;
}

const inv = (z: Complex): Complex => {
  const d = z.re * z.re + z.im * z.im;
  return { re: z.re / d, im: -z.im / d };
};

describe('roots.wgsl big roots via Engine.solveDebug', () => {
  it(`{1, 0.01} d=${DEGREE}: every root with |z| > 4 matches the f64 reference, and so does its 1/z`, async () => {
    const out = await engine.solveDebug(SET, DEGREE, 0n, POLYS);
    const ref = reference();
    let big = 0;
    let worstRel = 0;
    let worstInv = 0;
    for (let n = 0; n < POLYS; n++) {
      const got: Complex[] = [];
      for (let j = 0; j < DEGREE; j++) {
        const re = out[(n * DEGREE + j) * 2];
        if (re < 1e37) got.push({ re, im: out[(n * DEGREE + j) * 2 + 1] }); // >= 1e37: unused slot
      }
      for (const r of ref[n]) {
        const mag = Math.hypot(r.re, r.im);
        if (mag <= 4) continue;
        big++;
        let best = Infinity;
        let nearest = got[0];
        for (const g of got) {
          const e = Math.hypot(g.re - r.re, g.im - r.im);
          if (e < best) [best, nearest] = [e, g];
        }
        worstRel = Math.max(worstRel, best / mag);
        const a = inv(nearest);
        const b = inv(r);
        worstInv = Math.max(worstInv, Math.hypot(a.re - b.re, a.im - b.im));
      }
    }
    expect(big).toBeGreaterThan(1000); // the case is really exercised
    expect(worstRel).toBeLessThan(1e-4);
    expect(worstInv).toBeLessThan(1 / PX);
    expect(errors).toEqual([]); // convergence: debugRoots has no stats buffer, the histogram tests below count it
  });
});

/** Count channel of a full render of SET at `degree` into `view` (w x h). */
async function histogram(kernel: Kernel, enumeration: Enumeration, degree: number, view: View, w: number, h: number) {
  engine.kernel = kernel;
  engine.enumeration = enumeration;
  engine.resize(w, h);
  engine.setParams({ coeffs: SET, degree, view, maxIter: 32 });
  await runToEnd(device, engine, 1 << 20);
  const hist = await engine.readHistogram();
  return { counts: Uint32Array.from({ length: w * h }, (_, i) => hist[i * 4]), unconverged: (await engine.readStats()).unconverged };
}

describe.each(['perPoly', 'coop'] as Kernel[])('big roots, %s kernel', (kernel) => {
  it.each(['reciprocal', 'plain'] as Enumeration[])(`{1, 0.01} d=${DEGREE}, %s: the histogram near 0 equals the f64 roots' (|z| < 0.25 at ${PX} px/unit)`, async (enumeration) => {
    const S = 126; // radius 0.25 = 62.5 px around the origin
    // Off-centre by a fraction of a pixel: exact roots (the real axis, -0.01, +-0.1i) must not sit on pixel edges.
    const view = { cx: 0.3 / PX, cy: 0.5 / PX, pxPerUnit: PX };
    const { counts, unconverged } = await histogram(kernel, enumeration, DEGREE, view, S, S);
    // The same picture from the f64 roots of every polynomial, deposited as deposit() in roots.wgsl does.
    const want = new Uint32Array(S * S);
    for (const roots of reference()) {
      for (const r of roots) {
        const x = Math.floor((r.re - view.cx) * PX + S / 2);
        const y = Math.floor(S / 2 - (r.im - view.cy) * PX);
        if (x >= 0 && y >= 0 && x < S && y < S) want[y * S + x]++;
      }
    }
    let total = 0;
    let off = 0;
    for (let i = 0; i < S * S; i++) {
      total += want[i];
      off += Math.abs(counts[i] - want[i]);
    }
    expect(total).toBeGreaterThan(5000);
    // Measured on Arc 140T: 0 for both kernels and enumerations. Before the fix: reciprocal 17-18 % of the roots in
    // view misplaced (1/z of big roots accepted unconverged); plain 1550 (perPoly) / 1765 (coop) unconverged.
    expect(off / total).toBeLessThan(1e-3);
    expect(unconverged).toBe(0);
    expect(errors).toEqual([]);
  });

  it('{1, 0.01} d=16 at 400x250: as many lit pixels within |z| < 0.25 with reciprocal symmetry as without', async () => {
    const W = 400;
    const H = 250;
    const view = fitView(W, H, 2); // 62.5 px per unit
    const lit = async (enumeration: Enumeration) => {
      const rgba = await renderImage(device, { kernel, enumeration, params: { coeffs: SET, degree: 16, view, maxIter: 32 }, display: { colorMode: 1, exposure: 1, gamma: 0.6 }, width: W, height: H });
      let n = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = (y * W + x) * 4;
          if (rgba[i] + rgba[i + 1] + rgba[i + 2] > 0 && Math.hypot(x + 0.5 - W / 2, y + 0.5 - H / 2) < 0.25 * view.pxPerUnit) n++;
        }
      }
      return n;
    };
    const plain = await lit('plain');
    const reciprocal = await lit('reciprocal');
    // Measured on Arc 140T: 14 and 14 for both kernels; before the fix 14 and 56 (the ring of dots).
    expect(plain).toBeGreaterThan(5);
    expect(Math.abs(reciprocal - plain)).toBeLessThanOrEqual(2);
    expect(errors).toEqual([]);
  });
});
