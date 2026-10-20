import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { fitView } from '../../src/core/view';
import { Engine, type Enumeration } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';
import type { Kernel } from '../../src/core/schedule';

// Hue mode averages the per-root colour sums: a partner root (1/z) must carry the hue of the REVERSED polynomial.
// Compare the colour channels (not just counts) of plain vs reciprocal, per kernel.
const { device, errors } = await nodeDevice();
afterAll(() => device.destroy());

const sets: [string, Complex[]][] = [
  ['unpinned {1, -0.4+0.75i}', [{ re: 1, im: 0 }, { re: -0.4, im: 0.75 }]],
  ['pinned Littlewood', [{ re: 1, im: 0 }, { re: -1, im: 0 }]],
  ['C3 with 0', [{ re: 1, im: 0 }, { re: 0, im: 0 }, { re: -0.5, im: 0.8 }]],
];

async function hist(coeffs: Complex[], enumeration: Enumeration, kernel: Kernel): Promise<Uint32Array> {
  const e = new Engine(device, 'rgba8unorm');
  e.enumeration = enumeration;
  e.kernel = kernel;
  e.resize(256, 256);
  e.setParams({ coeffs, degree: 12, view: fitView(256, 256, 2.2), maxIter: 32 });
  await runToEnd(device, e, 1 << 20);
  const h = await e.readHistogram();
  e.destroy();
  return h;
}

it.each(sets.flatMap(([n, c]) => (['perPoly', 'coop'] as Kernel[]).map((k) => [n, c, k] as const)))('%s %s: colour sums match plain', async (_n, coeffs, kernel) => {
  const a = await hist(coeffs, 'plain', kernel);
  const b = await hist(coeffs, 'reciprocal', kernel);
  const sum = [0, 0, 0, 0], diff = [0, 0, 0, 0];
  for (let i = 0; i < a.length; i++) { sum[i % 4] += a[i]; diff[i % 4] += Math.abs(a[i] - b[i]); }
  const rel = diff.map((d, c) => d / Math.max(1, sum[c]));
  console.log(`${_n} ${kernel}: rel L1 count ${rel[0].toExponential(1)} r ${rel[1].toExponential(1)} g ${rel[2].toExponential(1)} b ${rel[3].toExponential(1)}`);
  for (let c = 1; c < 4; c++) expect(rel[c]).toBeLessThan(Math.max(0.05, 3 * rel[0]));
  expect(errors).toEqual([]);
});
