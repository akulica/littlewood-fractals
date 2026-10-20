import { afterAll, expect, it } from 'vitest';
import { framesFor } from '../../src/core/animation';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { renderAnimation } from '../../src/gpu/animation';
import { Engine } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// Animation cost per frame at a real video size, for both kernels: every frame from scratch (Engine, as Save PNG does)
// vs. renderAnimation (batches of 8 frames, each warm-started from the previous frame's roots: splatFrames for perPoly,
// splatFramesCoop for coop). Both include tone-mapping; renderAnimation also includes the RGBA readback, so the
// comparison favours cold. Reciprocal enumeration (the default) throughout.
const { device, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const W = 1920;
const H = 1080;
const FRAMES = 16;
const view = fitView(W, H, 1.75);
const keys: Complex[][] = [
  [{ re: 1, im: 0 }, { re: -1, im: 0 }],
  [{ re: 1, im: 0 }, { re: -0.4, im: 0.75 }],
];
const display = { colorMode: 1 as const, exposure: 1, gamma: 0.6 };

const KERNELS: Kernel[] = ['perPoly', 'coop'];

it.each([20, 22])('Littlewood-type {1, c}, c moving, degree %i, 1920x1080', async (degree) => {
  const frames = framesFor(keys, FRAMES, { easing: 'linear', loop: false });
  const cold = { perPoly: 0, coop: 0 };
  const warm = { perPoly: 0, coop: 0 };
  for (const kernel of KERNELS) {
    const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const e = new Engine(device, 'rgba8unorm');
    e.kernel = kernel;
    e.resize(W, H);
    const coldFrames = 4; // cold frames cost the same each; 4 is enough for a stable mean
    // Frames 1..4: frame 0 is {1, -1}, which a lone render pins (half the work), but the animation as a whole
    // cannot be pinned, so the fair comparison is against unpinned frames.
    for (let f = 1; f <= coldFrames; f++) {
      e.setParams({ coeffs: frames[f], degree, view, maxIter: 32 });
      const t0 = performance.now();
      await runToEnd(device, e, 1 << 20, true);
      const enc = device.createCommandEncoder();
      e.encodeDisplay(enc, tex.createView(), display);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      cold[kernel] += performance.now() - t0;
    }
    cold[kernel] /= coldFrames;
    e.destroy();
    tex.destroy();
  }
  for (const kernel of KERNELS) {
    let n = 0;
    const t0 = performance.now();
    const r = await renderAnimation(device, { coeffFrames: frames, degree, view, width: W, height: H, display, brightness: 'locked', quality: 1, maxIter: 32, kernel }, () => {
      n++;
    });
    warm[kernel] = (performance.now() - t0) / FRAMES;
    expect(n).toBe(FRAMES);
    expect(r.unconverged).toBe(0);
  }
  const ms = (x: number) => `${x.toFixed(0)} ms/frame`;
  console.log(
    `degree ${degree}: cold perPoly ${ms(cold.perPoly)}, cold coop ${ms(cold.coop)}, ` +
      `warm perPoly (splatFrames) ${ms(warm.perPoly)}, warm coop (splatFramesCoop) ${ms(warm.coop)}; ` +
      `warm coop/perPoly = ${(warm.coop / warm.perPoly).toFixed(2)}, best warm vs best cold ${(Math.min(cold.perPoly, cold.coop) / Math.min(warm.perPoly, warm.coop)).toFixed(2)}x`,
  );
});
