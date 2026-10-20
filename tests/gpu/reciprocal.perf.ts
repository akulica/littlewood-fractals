import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine, type Enumeration } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// Reciprocal enumeration (solve one polynomial of each reversal pair, deposit z and 1/z) vs the plain path, at a real
// output size: 1920x1080, both kernels, interleaved, best of 3 after a warm-up, one chunk in flight (as in the app).
// The spike (spike/costs.mjs) measured a 1.9-2.4x ceiling for {1, -0.4+0.75i}.
const { device, errors, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const W = 1920;
const H = 1080;
const engine = new Engine(device, 'rgba8unorm');
engine.resize(W, H);

async function time(coeffs: Complex[], degree: number, kernel: Kernel, enumeration: Enumeration): Promise<number> {
  engine.kernel = kernel;
  engine.enumeration = enumeration;
  engine.setParams({ coeffs, degree, view: fitView(W, H, 1.75), maxIter: 32 });
  const t0 = performance.now();
  await runToEnd(device, engine, 1 << 20, true);
  return performance.now() - t0;
}

const SPIKE: Complex[] = [{ re: 1, im: 0 }, { re: -0.4, im: 0.75 }];
const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];

it.each([
  { name: '{1, -0.4+0.75i} d=20', coeffs: SPIKE, degree: 20 },
  { name: '{1, -0.4+0.75i} d=22', coeffs: SPIKE, degree: 22 },
  { name: 'Littlewood d=20 (pinned)', coeffs: LW, degree: 20 },
  { name: 'Littlewood d=22 (pinned)', coeffs: LW, degree: 22 },
])('$name', async ({ name, coeffs, degree }) => {
  for (const kernel of ['perPoly', 'coop'] as Kernel[]) {
    const best: Record<Enumeration, number> = { plain: Infinity, reciprocal: Infinity };
    await time(coeffs, degree, kernel, 'plain'); // warm-up
    await time(coeffs, degree, kernel, 'reciprocal');
    for (let r = 0; r < 3; r++) {
      for (const e of ['plain', 'reciprocal'] as Enumeration[]) best[e] = Math.min(best[e], await time(coeffs, degree, kernel, e));
    }
    console.log(`${name} ${kernel}: plain ${best.plain.toFixed(0)} ms, reciprocal ${best.reciprocal.toFixed(0)} ms (speedup ${(best.plain / best.reciprocal).toFixed(2)}x)`);
    expect(best.reciprocal).toBeGreaterThan(0);
  }
  expect(errors).toEqual([]);
});
