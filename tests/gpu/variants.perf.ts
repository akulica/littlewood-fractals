import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine } from '../../src/gpu/engine';
import { countSum, nodeDevice, runToEnd } from './gpuHelpers';

// Kernel-variant shoot-out on the current GPU, through the real Engine (as the app runs it).
const { device, errors, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];

it.each([16, 20, 24])('Littlewood d=%i: per-polynomial vs cooperative', async (degree) => {
  const ms = { perPoly: 0, coop: 0 };
  const sum = { perPoly: 0, coop: 0 };
  for (const kernel of ['perPoly', 'coop'] as Kernel[]) {
    const e = new Engine(device, 'rgba8unorm');
    e.kernel = kernel;
    e.resize(1024, 1024);
    e.setParams({ coeffs: LW, degree, view: fitView(1024, 1024, 2), maxIter: 32 });
    const t0 = performance.now();
    await runToEnd(device, e, 1 << 20, true);
    ms[kernel] = performance.now() - t0;
    sum[kernel] = countSum(await e.readHistogram());
  }
  console.log(`d=${degree}: perPoly ${ms.perPoly.toFixed(0)} ms, coop ${ms.coop.toFixed(0)} ms (coop/perPoly = ${(ms.coop / ms.perPoly).toFixed(2)})`);
  // Jacobi vs Gauss-Seidel land a handful of roots on the other side of a pixel edge: allow 1e-6 relative.
  expect(Math.abs(sum.coop - sum.perPoly)).toBeLessThanOrEqual(Math.ceil(sum.perPoly * 1e-6));
  expect(errors).toEqual([]);
});

it('calibrate() choice on this GPU', async () => {
  const e = new Engine(device, 'rgba8unorm');
  e.resize(1024, 1024);
  const c = await e.calibrate();
  console.log(`calibrate: ${c.kernel} (perPoly ${c.perPolyMs.toFixed(1)} ms, coop ${c.coopMs.toFixed(1)} ms, coop/perPoly = ${(c.coopMs / c.perPolyMs).toFixed(2)})`);
  expect(e.kernel).toBe(c.kernel);
});
