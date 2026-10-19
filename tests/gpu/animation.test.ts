import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { fitView } from '../../src/core/view';
import { renderAnimation, type AnimationProgress, type AnimationRequest } from '../../src/gpu/animation';
import { Engine, RgbaReadback, renderImage } from '../../src/gpu/engine';
import { countSum, firstDifference, nodeDevice, readBuffer, runToEnd } from './gpuHelpers';

const { device, errors } = await nodeDevice();
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => {
  engine.destroy();
  device.destroy();
});

const SIZE = 192;
const VIEW = fitView(SIZE, SIZE, 2.6); // every root of these sets has |z| < 2.2: all of them land in the picture
const lerp = (a: Complex, b: Complex, t: number): Complex => ({ re: a.re + (b.re - a.re) * t, im: a.im + (b.im - a.im) * t });

/** {1, c} with c moving -0.9+0.15i -> -0.3+0.8i: never c0 times a group, so nothing is pinned. */
const MOVING: Complex[][] = Array.from({ length: 6 }, (_, f) => [{ re: 1, im: 0 }, lerp({ re: -0.9, im: 0.15 }, { re: -0.3, im: 0.8 }, f / 5)]);
/** {c, -c} with c rotating and growing: every frame pins its leading digit. */
const PINNED: Complex[][] = Array.from({ length: 5 }, (_, f) => {
  const c = { re: (1 + 0.1 * f) * Math.cos(0.3 * f), im: (1 + 0.1 * f) * Math.sin(0.3 * f) };
  return [c, { re: -c.re, im: -c.im }];
});

/** Batch render through splatFrames: one histogram per frame, plus the unconverged count. */
async function batchRender(frames: Complex[][], degree: number, pinned: boolean, budget = 1 << 20) {
  const frameBytes = SIZE * SIZE * 16;
  const target = device.createBuffer({ size: frames.length * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  engine.resize(SIZE, SIZE);
  engine.setFrames({ frames, degree, view: VIEW, maxIter: 32, pinned }, target);
  await runToEnd(device, engine, budget);
  const all = new Uint32Array(await readBuffer(device, target, frames.length * frameBytes));
  target.destroy();
  const { unconverged } = await engine.readStats();
  return { hists: frames.map((_, f) => all.subarray((f * frameBytes) / 4, ((f + 1) * frameBytes) / 4)), unconverged };
}

/** Same frame rendered on its own by the ordinary (cold, splat) path. */
async function coldRender(coeffs: Complex[], degree: number): Promise<Uint32Array> {
  engine.kernel = 'perPoly';
  engine.resize(SIZE, SIZE);
  engine.setParams({ coeffs, degree, view: VIEW, maxIter: 32 });
  await runToEnd(device, engine, 1 << 20);
  return engine.readHistogram();
}

function countL1(a: Uint32Array, b: Uint32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i += 4) d += Math.abs(a[i] - b[i]);
  return d;
}

describe('splatFrames (warm-started batch) vs one cold render per frame', () => {
  // Measured on Arc 140T: L1 2-3.5e-4 of the roots per frame.
  it('unpinned {1, c}, c moving: same root counts, roots within a pixel, all converged', async () => {
    const batch = await batchRender(MOVING, 12, false, 1000); // many chunks: the warm state lives in each thread
    expect(batch.unconverged).toBe(0);
    for (let f = 0; f < MOVING.length; f++) {
      const cold = await coldRender(MOVING[f], 12);
      const n = countSum(cold);
      expect(n).toBe(2 ** 13 * 12);
      expect(Math.abs(countSum(batch.hists[f]) - n)).toBeLessThanOrEqual(n * 1e-4);
      expect(countL1(batch.hists[f], cold)).toBeLessThan(n * 2e-3); // f32 edge noise: a root landing one pixel over
    }
    expect(errors).toEqual([]);
  });

  // A pinnable set is c0 * G, whose roots do not depend on c0: the picture is the same in every frame. The warm
  // start begins at roots that already pass the stop test, so every frame keeps frame 0's roots exactly. Cold renders
  // of the rotated copies differ from each other by 3-5e-3 (Littlewood polynomials have many multiple roots, which
  // f32 resolves only to noise), so they are not the reference here.
  it('pinned {c, -c}: every frame is frame 0 again, all converged', async () => {
    const batch = await batchRender(PINNED, 12, true, 1000);
    expect(batch.unconverged).toBe(0);
    const cold0 = await coldRender(PINNED[0], 12);
    expect(countSum(cold0)).toBe(2 ** 12 * 12); // leading digit pinned: 2^12 polynomials, not 2^13
    for (const hist of batch.hists) {
      expect(countSum(hist)).toBe(2 ** 12 * 12);
      expect(countL1(hist, cold0)).toBeLessThan(2 ** 12 * 12 * 2e-3);
    }
    expect(errors).toEqual([]);
  });

  it('a one-frame batch equals the ordinary render (no warm start happens)', async () => {
    for (const frame of [MOVING[2], PINNED[3]]) {
      const pinned = frame === PINNED[3];
      const one = (await batchRender([frame], 10, pinned)).hists[0];
      const cold = await coldRender(frame, 10);
      expect(countSum(one)).toBe(countSum(cold));
      expect(firstDifference(one, cold)).toBe(-1);
    }
    expect(errors).toEqual([]);
  });

  it('frame 0 of a batch is cold, so it equals the ordinary render exactly', async () => {
    const batch = await batchRender(MOVING, 10, false);
    expect(firstDifference(batch.hists[0], await coldRender(MOVING[0], 10))).toBe(-1);
  });

  it('display honours fixedMax and readMaxCount reports the densest pixel', async () => {
    const hist = await coldRender(MOVING[1], 10);
    let max = 0;
    for (let i = 0; i < hist.length; i += 4) max = Math.max(max, hist[i]);
    const out = new RgbaReadback(device, SIZE, SIZE);
    const show = async (fixedMax?: number) => {
      const enc = device.createCommandEncoder();
      engine.encodeDisplay(enc, out.view, { colorMode: 0, exposure: 1, gamma: 0.6, fixedMax });
      return out.read(enc);
    };
    const auto = await show();
    expect(await engine.readMaxCount()).toBe(max);
    expect(firstDifference(new Uint32Array((await show(max)).buffer), new Uint32Array(auto.buffer))).toBe(-1);
    const dimmer = await show(max * 4); // a 4x larger scale: every lit pixel gets darker or stays
    let darker = 0;
    let brighter = 0;
    for (let i = 0; i < auto.length; i += 4) {
      const d = dimmer[i] + dimmer[i + 1] + dimmer[i + 2] - (auto[i] + auto[i + 1] + auto[i + 2]);
      if (d < 0) darker++;
      if (d > 0) brighter++;
    }
    expect(darker).toBeGreaterThan(100);
    expect(brighter).toBe(0);
    out.destroy();
    expect(errors).toEqual([]);
  });
});

describe('renderAnimation', () => {
  const W = 150; // 600 bytes per row, padded to 768 on readback
  const H = 100;
  const view = fitView(W, H, 2.6);
  const display = { colorMode: 0, exposure: 1, gamma: 0.6 } as const;
  const req = (o: Partial<AnimationRequest> = {}): AnimationRequest => ({
    coeffFrames: [...MOVING, ...MOVING.slice().reverse()], // 12 frames: a batch of 8, then 4
    degree: 10,
    view,
    width: W,
    height: H,
    display,
    brightness: 'perFrame',
    quality: 1,
    maxIter: 32,
    ...o,
  });
  async function collect(r: AnimationRequest, signal?: AbortSignal, onFrame?: (i: number) => void) {
    const frames: Uint8ClampedArray[] = [];
    const progress: AnimationProgress[] = [];
    const result = await renderAnimation(
      device,
      r,
      (rgba, i) => {
        expect(i).toBe(frames.length);
        frames.push(rgba);
        onFrame?.(i);
      },
      (p) => progress.push(p),
      signal,
    );
    return { frames, progress, result };
  }

  it('delivers every frame as opaque RGBA of the requested size; frame 0 matches renderImage', async () => {
    const { frames, progress, result } = await collect(req());
    expect(frames).toHaveLength(12);
    for (const f of frames) {
      expect(f.length).toBe(W * H * 4);
      let opaque = true;
      for (let i = 3; i < f.length; i += 4) if (f[i] !== 255) opaque = false;
      expect(opaque).toBe(true);
    }
    expect(result.unconverged).toBe(0);
    const last = progress.at(-1)!;
    expect(last).toEqual({ frame: 12, frames: 12, polysDone: 2 ** 11 * 12, polysTotal: 2 ** 11 * 12 });
    const still = await renderImage(device, { kernel: 'perPoly', params: { coeffs: MOVING[0], degree: 10, view, maxIter: 32 }, display, width: W, height: H });
    expect(firstDifference(new Uint32Array(frames[0].buffer), new Uint32Array(still.buffer))).toBe(-1);
    // The path runs back: frame 11 shows frame 0's coefficients again (warm-started, so equal up to edge noise).
    let differing = 0;
    for (let i = 0; i < still.length; i += 4) if (frames[11][i] !== still[i]) differing++;
    expect(differing).toBeLessThan(W * H * 0.01);
    expect(errors).toEqual([]);
  });

  it('locked brightness: one scale (frame 0 densest pixel) for all frames', async () => {
    const perFrame = await collect(req());
    const locked = await collect(req({ brightness: 'locked' }));
    engine.resize(W, H); // frame 0 on its own, for its densest pixel
    engine.kernel = 'perPoly';
    engine.setParams({ coeffs: MOVING[0], degree: 10, view, maxIter: 32 });
    await runToEnd(device, engine, 1 << 20);
    const out = new RgbaReadback(device, W, H);
    const enc = device.createCommandEncoder();
    engine.encodeDisplay(enc, out.view, display);
    await out.read(enc);
    out.destroy();
    expect(locked.result.fixedMax).toBe(await engine.readMaxCount());
    expect(firstDifference(new Uint32Array(locked.frames[0].buffer), new Uint32Array(perFrame.frames[0].buffer))).toBe(-1);
    // Later frames have other densest pixels, so the two modes must differ there.
    const differs = locked.frames.slice(1).filter((f, i) => firstDifference(new Uint32Array(f.buffer), new Uint32Array(perFrame.frames[i + 1].buffer)) !== -1);
    expect(differs.length).toBeGreaterThan(5);
    expect(errors).toEqual([]);
  });

  it('quality 25 % renders the first quarter of the golden order per frame', async () => {
    const { frames, progress } = await collect(req({ quality: 0.25, coeffFrames: MOVING }));
    expect(frames).toHaveLength(6);
    expect(progress.at(-1)!.polysTotal).toBe(2 ** 9 * 6);
    expect(progress.at(-1)!.polysDone).toBe(2 ** 9 * 6);
    let lit = 0;
    for (let i = 0; i < frames[3].length; i += 4) if (frames[3][i] > 0) lit++;
    expect(lit).toBeGreaterThan(200);
    expect(errors).toEqual([]);
  });

  it('stops when aborted (between frames)', async () => {
    const ctl = new AbortController();
    let delivered = 0;
    await expect(
      collect(req(), ctl.signal, () => {
        if (++delivered === 2) ctl.abort();
      }),
    ).rejects.toThrow(/abort/i);
    expect(delivered).toBe(2);
    expect(errors).toEqual([]);
  });
});
