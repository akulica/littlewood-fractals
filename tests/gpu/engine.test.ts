import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine } from '../../src/gpu/engine';
import { countSum, nodeDevice, runToEnd } from './gpuHelpers';

const { device, errors } = await nodeDevice();
afterAll(() => device.destroy());

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const C3: Complex[] = [{ re: 1, im: 0 }, { re: -0.5, im: 0.8 }, { re: 0.3, im: -1.1 }];
const KERNELS: Kernel[] = ['perPoly', 'coop'];

async function littlewood10(kernel: Kernel): Promise<Engine> {
  const e = new Engine(device, 'rgba8unorm');
  e.kernel = kernel;
  e.resize(256, 256);
  e.setParams({ coeffs: LW, degree: 10, view: fitView(256, 256, 2.5), maxIter: 32 });
  await runToEnd(device, e, 1 << 20);
  return e;
}

async function c3Histogram(kernel: Kernel, budget: number): Promise<Uint32Array> {
  const e = new Engine(device, 'rgba8unorm');
  e.kernel = kernel;
  e.resize(128, 128);
  e.setParams({ coeffs: C3, degree: 8, view: fitView(128, 128, 3), maxIter: 32 });
  expect(e.progress.total).toBe(3n ** 9n); // set is not c0*G: nothing pinned
  await runToEnd(device, e, budget);
  return e.readHistogram();
}

describe.each(KERNELS)('Engine with the %s kernel', (kernel) => {
  it('pins the leading coefficient and deposits every root (Littlewood degree 10)', async () => {
    const e = await littlewood10(kernel);
    expect(e.progress.total).toBe(1024n); // 2^10, not 2^11
    expect(countSum(await e.readHistogram())).toBe(1024 * 10); // all Littlewood roots satisfy 1/2 < |z| < 2
    expect((await e.readStats()).unconverged).toBe(0);
    expect(errors).toEqual([]);
  });

  it('produces an identical histogram whether the work is one chunk or many (carry logic)', async () => {
    const one = await c3Histogram(kernel, 1 << 20);
    const many = await c3Histogram(kernel, 1000);
    expect(countSum(one)).toBeGreaterThan(19683 * 8 * 0.9);
    expect(many).toEqual(one);
  });
});

describe('Engine', () => {
  it('both kernels draw the same picture (they differ only in roots landing across pixel edges)', async () => {
    const a = await c3Histogram('perPoly', 1 << 20);
    const b = await c3Histogram('coop', 1 << 20);
    let diff = 0;
    for (let i = 0; i < a.length; i += 4) diff += Math.abs(a[i] - b[i]);
    expect(countSum(b)).toBe(countSum(a));
    expect(diff).toBeLessThan(countSum(a) * 1e-3);
  });

  it('calibrate() times both kernels, picks one and leaves accumulation reset', async () => {
    const e = new Engine(device, 'rgba8unorm');
    e.resize(128, 128);
    const c = await e.calibrate();
    expect(KERNELS).toContain(c.kernel);
    expect(e.kernel).toBe(c.kernel);
    expect(c.perPolyMs).toBeGreaterThan(0);
    expect(c.coopMs).toBeGreaterThan(0);
    expect(e.progress.done).toBe(0n);
    expect(errors).toEqual([]);
  });

  it('renders roots on a black background', async () => {
    const e = await littlewood10('perPoly');
    const tex = device.createTexture({ size: [256, 256], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 256 * 4 * 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    e.encodeDisplay(enc, tex.createView(), { colorMode: 1, exposure: 1, gamma: 0.6 });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: 1024 }, [256, 256]);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(rb.getMappedRange());
    let lit = 0;
    for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] > 0) lit++;
    expect(lit).toBeGreaterThan(200);
    const centre = (128 * 256 + 128) * 4; // z = 0: no Littlewood root there
    expect([...px.slice(centre, centre + 4)]).toEqual([0, 0, 0, 255]);
    expect(errors).toEqual([]);
  });
});
