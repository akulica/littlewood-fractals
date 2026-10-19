import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { Engine, type Ordering } from '../../src/gpu/engine';
import { blockL1, countSum, firstDifference, nodeDevice, runToEnd } from './gpuHelpers';

const { device, errors } = await nodeDevice();
// One engine for every render below: building one compiles all its pipelines (~0.3-1 s), while setParams()
// restarts accumulation and kernel/ordering are plain fields.
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => {
  engine.destroy();
  device.destroy();
});

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const C3: Complex[] = [{ re: 1, im: 0 }, { re: -0.5, im: 0.8 }, { re: 0.3, im: -1.1 }];
const KERNELS: Kernel[] = ['perPoly', 'coop'];

function load(kernel: Kernel, ordering: Ordering, size: number, coeffs: Complex[], degree: number, radius: number): Engine {
  engine.kernel = kernel;
  engine.ordering = ordering;
  engine.resize(size, size);
  engine.setParams({ coeffs, degree, view: fitView(size, size, radius), maxIter: 32 });
  return engine;
}

async function littlewood10(kernel: Kernel, budget = 1 << 20, ordering: Ordering = 'golden'): Promise<Engine> {
  const e = load(kernel, ordering, 256, LW, 10, 2.5);
  await runToEnd(device, e, budget);
  return e;
}

async function c3Histogram(kernel: Kernel, budget: number, ordering: Ordering = 'golden'): Promise<Uint32Array> {
  const e = load(kernel, ordering, 128, C3, 8, 3);
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

  // Exact equality: the same multiset of polynomials, each solved identically, summed with integer atomics.
  it('golden order is a permutation: same full histogram as lexicographic (pinned Littlewood d=10)', async () => {
    const lex = await (await littlewood10(kernel, 1 << 20, 'lexicographic')).readHistogram();
    expect(countSum(lex)).toBe(1024 * 10);
    expect(firstDifference(await (await littlewood10(kernel, 1 << 20)).readHistogram(), lex)).toBe(-1);
    expect(firstDifference(await (await littlewood10(kernel, 100)).readHistogram(), lex)).toBe(-1); // chunk starts wrap mod N
    expect(errors).toEqual([]);
  });

  it('golden order is a permutation: same full histogram as lexicographic (3 coefficients, d=8, unpinned)', async () => {
    const lex = await c3Histogram(kernel, 1 << 20, 'lexicographic');
    expect(firstDifference(await c3Histogram(kernel, 1 << 20, 'golden'), lex)).toBe(-1);
    expect(firstDifference(await c3Histogram(kernel, 777, 'golden'), lex)).toBe(-1);
    expect(errors).toEqual([]);
  });

  // Measured 0.011 vs 0.150 on Arc 140T (the plan's spike: 0.017 vs 0.153 at degree 20, 512^2).
  it('a 1/16 golden prefix at full degree looks like the full picture, closer than a complete lower degree', async () => {
    const size = 256;
    const e = load(kernel, 'golden', size, LW, 18, 2.2); // 2^18 polynomials (leading digit pinned)
    const enc = device.createCommandEncoder();
    expect(e.encodeCompute(enc, 2 ** 18 / 16)).toBe(2 ** 14);
    device.queue.submit([enc.finish()]);
    const prefix = await e.readHistogram();
    await runToEnd(device, e, 1 << 16); // the full render continues where the preview stopped
    const full = await e.readHistogram();
    expect(countSum(full)).toBe(2 ** 18 * 18);
    load(kernel, 'golden', size, LW, 14, 2.2); // the old preview: every polynomial of degree 14, same 2^14 budget
    await runToEnd(device, e, 1 << 16);
    const lower = await e.readHistogram();
    const goldenL1 = blockL1(prefix, full, size, size, 8);
    expect(goldenL1).toBeLessThan(0.05);
    expect(goldenL1).toBeLessThan(blockL1(lower, full, size, size, 8));
    expect(errors).toEqual([]);
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
    e.destroy();
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
