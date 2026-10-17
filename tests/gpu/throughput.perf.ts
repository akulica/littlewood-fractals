import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { fitView } from '../../src/core/view';
import { Engine } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

const { device, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const C3: Complex[] = [{ re: 1, im: 0 }, { re: -0.5, im: 0.8 }, { re: 0.3, im: -1.1 }];

it.each([
  { name: 'Littlewood d=16', coeffs: LW, degree: 16 },
  { name: 'Littlewood d=20', coeffs: LW, degree: 20 },
  { name: 'Littlewood d=24', coeffs: LW, degree: 24 },
  { name: '3 complex coeffs d=14', coeffs: C3, degree: 14 },
])('$name', async ({ name, coeffs, degree }) => {
  for (const waitEachChunk of [false, true]) {
    const e = new Engine(device, 'rgba8unorm');
    e.resize(1024, 1024);
    e.setParams({ coeffs, degree, view: fitView(1024, 1024, 2), maxIter: 32 });
    const t0 = performance.now();
    await runToEnd(device, e, 1 << 20, waitEachChunk);
    const ms = performance.now() - t0;
    const polys = Number(e.progress.total);
    const mode = waitEachChunk ? 'one chunk in flight' : 'all chunks queued';
    console.log(`${name} [${mode}]: ${polys.toLocaleString()} polys in ${ms.toFixed(0)} ms = ${((polys * degree) / ms / 1e3).toFixed(1)} M roots/s`);
    expect(ms).toBeGreaterThan(0);
  }
});
