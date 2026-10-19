import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine, type Ordering } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// Golden order puts unrelated polynomials next to each other on the GPU (vs lexicographic neighbours that share
// their high-order coefficients), which can cost SIMD convergence coherence and atomic locality. This measures it:
// both orders, both kernels, interleaved, best of 3 after a warm-up, one chunk in flight (as in the app).
const { device, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const C3: Complex[] = [{ re: 1, im: 0 }, { re: -0.5, im: 0.8 }, { re: 0.3, im: -1.1 }];
const engine = new Engine(device, 'rgba8unorm');
engine.resize(1024, 1024);

async function time(coeffs: Complex[], degree: number, kernel: Kernel, ordering: Ordering): Promise<number> {
  engine.kernel = kernel;
  engine.ordering = ordering;
  engine.setParams({ coeffs, degree, view: fitView(1024, 1024, 2), maxIter: 32 });
  const t0 = performance.now();
  await runToEnd(device, engine, 1 << 20, true);
  return performance.now() - t0;
}

it.each([
  { name: 'Littlewood d=20', coeffs: LW, degree: 20 },
  { name: 'Littlewood d=22', coeffs: LW, degree: 22 },
  { name: '3 complex coeffs d=14', coeffs: C3, degree: 14 },
])('$name', async ({ name, coeffs, degree }) => {
  for (const kernel of ['perPoly', 'coop'] as Kernel[]) {
    const best: Record<Ordering, number> = { lexicographic: Infinity, golden: Infinity };
    await time(coeffs, degree, kernel, 'lexicographic'); // warm-up
    for (let r = 0; r < 3; r++) {
      for (const o of ['lexicographic', 'golden'] as Ordering[]) best[o] = Math.min(best[o], await time(coeffs, degree, kernel, o));
    }
    console.log(`${name} ${kernel}: lexicographic ${best.lexicographic.toFixed(0)} ms, golden ${best.golden.toFixed(0)} ms (golden/lex = ${(best.golden / best.lexicographic).toFixed(2)})`);
    expect(best.golden).toBeGreaterThan(0);
  }
});
