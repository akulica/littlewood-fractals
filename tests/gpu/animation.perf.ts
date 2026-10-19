import { afterAll, expect, it } from 'vitest';
import { framesFor } from '../../src/core/animation';
import type { Complex } from '../../src/core/complex';
import { fitView } from '../../src/core/view';
import { renderAnimation } from '../../src/gpu/animation';
import { Engine } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// Animation cost per frame at a real video size: every frame from scratch (Engine, as Save PNG does) vs.
// renderAnimation (splatFrames: batches of 8 frames, each warm-started from the previous frame's roots).
// Both include tone-mapping; renderAnimation also includes the RGBA readback, so the comparison favours cold.
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

it.each([20, 22])('Littlewood-type {1, c}, c moving, degree %i, 1920x1080', async (degree) => {
  const frames = framesFor(keys, FRAMES, { easing: 'linear', loop: false });
  const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const e = new Engine(device, 'rgba8unorm');
  e.resize(W, H);
  const coldFrames = 4; // cold frames cost the same each; 4 is enough for a stable mean
  let cold = 0;
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
    cold += performance.now() - t0;
  }
  cold /= coldFrames;
  e.destroy();
  tex.destroy();
  let n = 0;
  const t0 = performance.now();
  const r = await renderAnimation(device, { coeffFrames: frames, degree, view, width: W, height: H, display, brightness: 'locked', quality: 1, maxIter: 32 }, () => {
    n++;
  });
  const warm = (performance.now() - t0) / FRAMES;
  console.log(`degree ${degree}: cold ${cold.toFixed(0)} ms/frame, renderAnimation ${warm.toFixed(0)} ms/frame (${(cold / warm).toFixed(2)}x), unconverged ${r.unconverged}`);
  expect(n).toBe(FRAMES);
});
