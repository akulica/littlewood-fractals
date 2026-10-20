import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { fitView, worldToPixel, type View } from '../../src/core/view';
import { renderAnimation, type AnimationRequest, type FrameInfo } from '../../src/gpu/animation';
import { Engine, renderImage, type DisplayParams } from '../../src/gpu/engine';
import { countSum, firstDifference, nodeDevice, readBuffer, runToEnd } from './gpuHelpers';

const { device, errors } = await nodeDevice();
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => {
  engine.destroy();
  device.destroy();
});

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const pixel = (rgba: Uint8ClampedArray, w: number, x: number, y: number) => {
  const i = (Math.floor(y) * w + Math.floor(x)) * 4;
  return [rgba[i], rgba[i + 1], rgba[i + 2]];
};
const isWhite = (rgba: Uint8ClampedArray, w: number, x: number, y: number) => pixel(rgba, w, x, y).every((v) => v === 255);

describe('coefficient rings (display.wgsl)', () => {
  const W = 300;
  const H = 200;
  const view = fitView(W, H, 2.2);
  const base = { colorMode: 0, exposure: 1, gamma: 0.6 } as const;
  const render = (display: DisplayParams) =>
    renderImage(device, { kernel: 'perPoly', params: { coeffs: LW, degree: 12, view, maxIter: 32 }, display, width: W, height: H });

  it('draws a white ring (radius 9..11 CSS px x scale) at each centre; nothing when disabled', async () => {
    const scale = 2; // ring spans radius 18..22 output px, halo out to ~30
    const centres = LW.map((c) => worldToPixel(view, W, H, c.re, c.im));
    const plain = await render(base);
    const ringed = await render({ ...base, rings: { centres, scale } });
    for (const p of centres) {
      for (const [dx, dy] of [[20, 0], [-20, 0], [0, 20], [0, -20], [14.1, 14.1]]) {
        expect(isWhite(ringed, W, p.x + dx, p.y + dy)).toBe(true);
      }
      // Inside the ring the picture is untouched: the CSS handle is transparent there.
      expect(pixel(ringed, W, p.x + 8, p.y)).toEqual(pixel(plain, W, p.x + 8, p.y));
    }
    let differ = 0;
    let white = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const near = centres.some((p) => Math.hypot(x + 0.5 - p.x, y + 0.5 - p.y) < 31); // 15 CSS px x 2 + 1
        if (!near && pixel(ringed, W, x, y).some((v, i) => v !== pixel(plain, W, x, y)[i])) differ++;
        if (isWhite(plain, W, x, y)) white++;
      }
    }
    expect(differ).toBe(0); // away from the rings both images are identical
    expect(white).toBe(0); // inferno never reaches pure white: without rings there is none
    expect(errors).toEqual([]);
  });

  it('a dark halo surrounds the ring, like the handle\'s box-shadow', async () => {
    const centre = { x: W / 2, y: H / 2 - 40 }; // over the dense upper half of the Littlewood picture
    const plain = await render(base);
    const ringed = await render({ ...base, rings: { centres: [centre], scale: 2 } });
    let lit = 0;
    let darker = 0;
    for (let a = 0; a < 64; a++) {
      const x = centre.x + 24 * Math.cos((a / 64) * 2 * Math.PI);
      const y = centre.y + 24 * Math.sin((a / 64) * 2 * Math.PI);
      const before = pixel(plain, W, x, y).reduce((s, v) => s + v, 0);
      if (before === 0) continue;
      lit++;
      if (pixel(ringed, W, x, y).reduce((s, v) => s + v, 0) < before) darker++;
    }
    expect(lit).toBeGreaterThan(10);
    expect(darker).toBe(lit);
  });
});

describe('per-frame views (splatFrames)', () => {
  const SIZE = 160;
  it('each frame of a batch is rendered at its own view, identical to a cold render there', async () => {
    const coeffs: Complex[] = [{ re: 1, im: 0 }, { re: -0.6, im: 0.5 }];
    const views: View[] = [fitView(SIZE, SIZE, 2.4), { cx: 0.3, cy: 0.4, pxPerUnit: 90 }, { cx: -0.5, cy: 0.1, pxPerUnit: 200 }];
    const frameBytes = SIZE * SIZE * 16;
    const target = device.createBuffer({ size: views.length * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    engine.resize(SIZE, SIZE);
    engine.setFrames({ frames: views.map(() => coeffs), views, degree: 11, view: views[0], maxIter: 32, pinned: false }, target);
    await runToEnd(device, engine, 1 << 20);
    const all = new Uint32Array(await readBuffer(device, target, views.length * frameBytes));
    target.destroy();
    const sums: number[] = [];
    for (let f = 0; f < views.length; f++) {
      engine.kernel = 'perPoly';
      engine.setParams({ coeffs, degree: 11, view: views[f], maxIter: 32 });
      await runToEnd(device, engine, 1 << 20);
      const cold = await engine.readHistogram();
      const batch = all.subarray((f * frameBytes) / 4, ((f + 1) * frameBytes) / 4);
      // Same coefficients in every frame: the warm start begins at converged roots, so later frames match exactly too.
      expect(firstDifference(batch, cold)).toBe(-1);
      sums.push(countSum(batch));
    }
    expect(sums[2]).toBeLessThan(sums[0]); // the zoomed frame crops most roots: the views really differ
    expect(errors).toEqual([]);
  });
});

describe('renderAnimation with view, exposure/gamma and rings per frame', () => {
  const W = 192;
  const H = 192;
  const coeffs: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }, { re: 0, im: 1 }, { re: 0, im: -1 }, { re: 0.5, im: 0.5 }];
  // Root density near 0.8+0.8i is smooth at this scale: probing four points, the 4x zoom's mean brightness under
  // the density lock was within 2.3 % of the same patch in frame 0.
  const v0: View = { cx: 0.8, cy: 0.8, pxPerUnit: W / 0.8 };
  const v1: View = { cx: 0.8, cy: 0.8, pxPerUnit: W / 0.2 }; // 4x zoom into the central quarter
  const req = (o: Partial<AnimationRequest>): AnimationRequest => ({
    coeffFrames: [coeffs, coeffs],
    viewFrames: [v0, v1],
    degree: 9,
    view: v0,
    width: W,
    height: H,
    display: { colorMode: 0, exposure: 1, gamma: 0.6 },
    brightness: 'locked',
    quality: 1,
    maxIter: 32,
    ...o,
  });
  async function collect(r: AnimationRequest) {
    const frames: Uint8ClampedArray[] = [];
    const infos: FrameInfo[] = [];
    await renderAnimation(device, r, (rgba, _i, info) => {
      frames.push(rgba);
      infos.push(info);
    });
    return { frames, infos };
  }
  const mean = (f: Uint8ClampedArray, from: number, to: number) => {
    let s = 0;
    let n = 0;
    for (let y = from; y < to; y++) {
      for (let x = from; x < to; x++) {
        s += pixel(f, W, x, y).reduce((a, v) => a + v, 0);
        n++;
      }
    }
    return s / n;
  };

  it('density lock keeps a zoomed constant-density region as bright as in frame 0; frame-0 lock darkens it', async () => {
    const locked = await collect(req({ brightness: 'locked' }));
    const frame0 = await collect(req({ brightness: 'frame0' }));
    const patch = mean(locked.frames[0], W * 0.375, W * 0.625); // what frame 1 shows, as seen in frame 0
    expect(mean(frame0.frames[0], W * 0.375, W * 0.625)).toBe(patch);
    const lockedRatio = mean(locked.frames[1], 0, W) / patch;
    const frame0Ratio = mean(frame0.frames[1], 0, W) / patch;
    console.log(`4x zoom, brightness vs frame 0: density lock ${lockedRatio.toFixed(3)}, frame-0 lock ${frame0Ratio.toFixed(3)}`);
    expect(Math.abs(lockedRatio - 1)).toBeLessThan(0.1);
    expect(frame0Ratio).toBeLessThan(0.9);
    expect(locked.infos[1].display.countScale).toBe(16);
    expect(locked.infos[1].view).toEqual(v1);
    expect(errors).toEqual([]);
  });

  it('per-frame exposure/gamma and rings: each frame equals renderImage with that frame\'s display', async () => {
    const { frames, infos } = await collect(
      req({ viewFrames: [v0, v0], brightness: 'perFrame', displayFrames: [{ exposure: 1, gamma: 0.6 }, { exposure: 2, gamma: 1 }], rings: { scale: 1.5 } }),
    );
    expect(infos[1].display.exposure).toBe(2);
    expect(infos[1].display.rings!.centres).toHaveLength(5);
    for (let f = 0; f < 2; f++) {
      const still = await renderImage(device, { kernel: 'perPoly', params: { coeffs, degree: 9, view: v0, maxIter: 32 }, display: infos[f].display, width: W, height: H });
      expect(firstDifference(new Uint32Array(frames[f].buffer), new Uint32Array(still.buffer))).toBe(-1);
    }
    const c = infos[0].display.rings!.centres[4]; // 0.5+0.5i: the only coefficient inside this view
    expect(isWhite(frames[0], W, c.x + 15, c.y)).toBe(true); // ring radius 9..11 x 1.5 = 13.5..16.5
    expect(errors).toEqual([]);
  });
});
