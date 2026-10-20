import { mkdirSync, writeFileSync } from 'node:fs';
import { env, version as nodeVersion } from 'node:process';
import { afterAll, expect, it } from 'vitest';
import { batchFrames, frameBrightness, framesFor, pinnable } from '../../src/core/animation';
import type { Complex } from '../../src/core/complex';
import { encodePng } from '../../src/core/png';
import { nextBudget, type Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { renderAnimation, type AnimationRequest } from '../../src/gpu/animation';
import { Engine, renderImage, type DisplayParams, type Enumeration, type FractalParams } from '../../src/gpu/engine';
import { nodeDevice, readBuffer } from './gpuHelpers';
import {
  Profiler,
  TimedReadback,
  engineBuffers,
  now,
  spread,
  variantLayout,
  variantPipeline,
  type Layout,
  type Spread,
  type Variant,
} from './profileHelpers';

// Per-stage profile of the renders the app does: `npm run profile` (laptop/PC numbers in README "Where the time goes").
//
// What is timed, per case (set, degree, size, still | 16-frame warm animation, kernel, enumeration):
//   e2e       the shipped renderImage / renderAnimation, unmodified (animation: once with sink = noop, once with sink = PNG
//             encode of every frame, src/core/png.ts, the stand-in for a writer).
//   stages    the same work re-encoded in this file from the shipped Engine calls (encodeCompute / encodeDisplay /
//             loadHistogram / readMaxCount / readStats), each submit awaited, so every stage has
//             - GPU time: timestamp-query via timestampWrites on the Engine's own passes (solve+splat, reduce, tone-map), and
//               for copies/clears (the hist clear, the frame slice -> histogram copy, the texture -> buffer copy) the gap
//               between two empty timestamped passes around them;
//             - wall time: performance.now() around submit + queue.onSubmittedWorkDone() (the mapAsync and the CPU row-strip
//               loop of the RGBA readback separately). Wall stages are serialised, so their sum slightly exceeds a pipelined
//               render; it is printed against the e2e time ("sum wall/e2e", should be within ~15 %).
//   ceilings  test-only WGSL variants (tests/gpu/profileHelpers.ts), on a golden-order prefix sample of the solves
//             (chunks of CEIL_CHUNK solves, a fixed number per variant), GPU time of the dispatch passes, ratio to the full kernel:
//               noSplat    (a) solve, roots summed into a register stored under a never-true condition, no deposit
//               noSolve    (b) splat only, roots = the real roots of the same polynomials captured once into a buffer
//               uniform    (c) splat only, roots uniform over the view      conc  (c) splat only, roots on |z| = 1 +- 0.02
// Env: PROFILE_QUICK=1 (d = 16, 20 at 1024^2 and 1080p, one repetition), PROFILE_REPS (default 3), PROFILE_FILTER (substring of
// the case name, e.g. "LW d=20 1920x1080 anim coop"), PROFILE_OUT (JSON file), PROFILE_NO_CEIL=1 (skip the ceilings),
// PROFILE_NO_TIMESTAMPS=1 (do not request timestamp-query: the wall-clock fallback), PROFILE_RAW=1 (print every repetition).

const QUICK = env.PROFILE_QUICK === '1';
const REPS = Math.max(1, Number(env.PROFILE_REPS) || (QUICK ? 1 : 3));
const FILTER = (env.PROFILE_FILTER ?? '').toLowerCase();
const OUT = env.PROFILE_OUT ?? '';
const CEILINGS = env.PROFILE_NO_CEIL !== '1';
const RAW = env.PROFILE_RAW === '1'; // also print every repetition's numbers
const BURN_MS = 400; // GPU load before each repetition's first measurement: the iGPU clocks ramp up in the first few hundred ms
const FRAMES = 16;
const MAX_ITER = 32; // = MAX_ITER in src/ui/main.ts
const TARGET_MS = 100; // per chunk, as EXPORT_TARGET_MS in src/gpu/engine.ts and TARGET_MS in src/gpu/animation.ts
const CEIL_CHUNK = { still: 1 << 17, frames: 1 << 15 }; // solves per dispatch (x frames in a batch)
const CEIL_SAMPLE = { still: 1 << 19, frames: 1 << 17 }; // most solves a ceiling run covers (a prefix of the golden order)
const DISPLAY: DisplayParams = { colorMode: 1, exposure: 1, gamma: 0.6 }; // the app's defaults
const TIMEOUT = 24 * 3600 * 1000;

const { device, errors, adapterName, timestamps } = await nodeDevice({ timestamps: env.PROFILE_NO_TIMESTAMPS !== '1' });
const prof = new Profiler(device, timestamps);
afterAll(() => {
  prof.destroy();
  device.destroy();
});

const machine = `${adapterName} | ${env.COMPUTERNAME ?? 'unknown host'} | node ${nodeVersion} | Dawn via the webgpu npm package`;
const timing = timestamps
  ? 'GPU: timestamp-query (timestampWrites on the Engine passes; empty marker passes around copies/clears). wall: performance.now() around submit + onSubmittedWorkDone(), stages serialised.'
  : 'NO timestamp-query (the adapter lacks it, or PROFILE_NO_TIMESTAMPS=1): wall clock only (performance.now() around submit + onSubmittedWorkDone(), stages serialised); GPU columns are absent.';
console.log(`adapter: ${machine}`);
console.log(`timing: ${timing}`);

const c = (re: number, im = 0): Complex => ({ re, im });
interface SetDef {
  name: string;
  short: string;
  coeffs: Complex[];
  /** The set's coefficients at the last frame: the animation moves them linearly there (as tests/gpu/animation.perf.ts). */
  end: Complex[];
}
const LW: SetDef = { name: 'Littlewood {1,-1}', short: 'LW', coeffs: [c(1), c(-1)], end: [c(1), c(-0.4, 0.75)] };
const SP: SetDef = { name: '{1,-0.4+0.75i}', short: 'SP', coeffs: [c(1), c(-0.4, 0.75)], end: [c(1), c(-1)] };

interface Case {
  set: SetDef;
  degree: number;
  width: number;
  height: number;
  kernel: Kernel;
  enumeration: Enumeration;
  mode: 'still' | 'anim';
}
const caseName = (k: Case) => `${k.set.short} d=${k.degree} ${k.width}x${k.height} ${k.mode} ${k.kernel} ${k.enumeration}`;
const viewOf = (k: Case) => fitView(k.width, k.height, 2); // the app's Reset view
const paramsOf = (k: Case): FractalParams => ({ coeffs: k.set.coeffs, degree: k.degree, view: viewOf(k), maxIter: MAX_ITER });
const framesOf = (k: Case): Complex[][] => framesFor([k.set.coeffs, k.set.end], FRAMES, { easing: 'linear', loop: false });
const animRequest = (k: Case): AnimationRequest => ({
  coeffFrames: framesOf(k),
  degree: k.degree,
  view: viewOf(k),
  width: k.width,
  height: k.height,
  display: DISPLAY,
  brightness: 'locked',
  quality: 1,
  maxIter: MAX_ITER,
  enumeration: k.enumeration,
  kernel: k.kernel,
});

type Acc = Map<string, number[]>;
const push = (acc: Acc, key: string, v: number | null | undefined) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return;
  const a = acc.get(key);
  if (a) a.push(v);
  else acc.set(key, [v]);
};
const sp = (acc: Acc, key: string): Spread | null => {
  const a = acc.get(key);
  return a ? spread(a) : null;
};
const med = (acc: Acc, key: string): number => sp(acc, key)?.median ?? NaN;
const f = (x: number): string => (Number.isNaN(x) ? 'n/a' : x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toFixed(3));
const range = (s: Spread | null): string => (s ? `${f(s.median)} (${f(s.min)}-${f(s.max)})` : 'n/a');

// ---------------------------------------------------------------------------------------------------------------------
// Still: the renderImage path, re-encoded stage by stage
// ---------------------------------------------------------------------------------------------------------------------

interface StageRun {
  gpu: Map<string, number>; // ms per label (timestamps), empty without timestamp-query
  wall: Record<string, number>; // ms
  frames: Uint8ClampedArray<ArrayBuffer>[]; // kept only on request (equivalence check)
  unconverged: number;
}

async function stillStages(k: Case, keep = false): Promise<StageRun> {
  const wall = { setup: 0, solve: 0, display: 0, copy: 0, map: 0, strip: 0, chunks: 0 };
  let t = now();
  const engine = new Engine(device, 'rgba8unorm');
  const out = new TimedReadback(device, k.width, k.height);
  try {
    engine.kernel = k.kernel;
    engine.enumeration = k.enumeration;
    engine.resize(k.width, k.height);
    engine.setParams(paramsOf(k));
    wall.setup = now() - t;
    let budget = 1 << 16;
    for (;;) {
      const px = prof.encoder();
      const count = engine.encodeCompute(px.enc, budget);
      const t0 = now();
      device.queue.submit([px.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = now() - t0;
      wall.solve += ms;
      if (count === 0) break;
      wall.chunks++;
      budget = nextBudget(budget, ms, TARGET_MS);
      if (prof.nearFull) await prof.flush();
    }
    const px = prof.encoder();
    px.display(engine, out.view, DISPLAY);
    t = now();
    device.queue.submit([px.finish()]);
    await device.queue.onSubmittedWorkDone();
    wall.display = now() - t;
    const r = await out.read(prof);
    wall.copy = r.copyMs;
    wall.map = r.mapMs;
    wall.strip = r.stripMs;
    return { gpu: await prof.take(), wall, frames: keep ? [r.rgba] : [], unconverged: (await engine.readStats()).unconverged };
  } finally {
    out.destroy();
    engine.destroy();
  }
}

async function stillE2e(k: Case): Promise<number> {
  const t0 = now();
  await renderImage(device, { kernel: k.kernel, enumeration: k.enumeration, params: paramsOf(k), display: DISPLAY, width: k.width, height: k.height });
  return now() - t0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Animation: renderAnimation, re-encoded stage by stage (same batches, chunk budgets, locked brightness, readback)
// ---------------------------------------------------------------------------------------------------------------------

async function animStages(k: Case, keep = false): Promise<StageRun> {
  const wall = { setup: 0, setFrames: 0, solve: 0, stats: 0, f0: 0, display: 0, copy: 0, map: 0, strip: 0, png: 0, chunks: 0 };
  const { width, height } = k;
  const frames = framesOf(k);
  const view = viewOf(k);
  const views = frames.map(() => view);
  const baseScale = DISPLAY.countScale ?? 1;
  const displayOf = (i: number, max0: number): DisplayParams => ({
    ...DISPLAY,
    ...frameBrightness('locked', baseScale, views[i].pxPerUnit / views[0].pxPerUnit, max0),
    rings: undefined,
  });
  let t = now();
  const frameBytes = width * height * 16;
  const perBatch = Math.min(frames.length, batchFrames(frameBytes, device.limits));
  const pinned = pinnable(frames);
  const engine = new Engine(device, 'rgba8unorm');
  engine.enumeration = k.enumeration;
  engine.kernel = k.kernel;
  const out = new TimedReadback(device, width, height);
  let slices: GPUBuffer | null = null;
  const kept: Uint8ClampedArray<ArrayBuffer>[] = [];
  let fixedMax = 0;
  let unconverged = 0;
  let pngSamples = 0;
  try {
    engine.resize(width, height);
    slices = device.createBuffer({ size: perBatch * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    wall.setup = now() - t;
    let budget = 1 << 12;
    for (let first = 0; first < frames.length; first += perBatch) {
      const batch = frames.slice(first, first + perBatch);
      t = now();
      engine.setFrames({ frames: batch, views: views.slice(first, first + perBatch), degree: k.degree, view, maxIter: MAX_ITER, pinned }, slices);
      wall.setFrames += now() - t;
      const work = engine.work.total;
      while (engine.work.done < work) {
        const left = work - engine.work.done;
        const px = prof.encoder();
        engine.encodeCompute(px.enc, left < BigInt(budget) ? Number(left) : budget);
        const t0 = now();
        device.queue.submit([px.finish()]);
        await device.queue.onSubmittedWorkDone();
        const ms = now() - t0;
        wall.solve += ms;
        wall.chunks++;
        budget = nextBudget(budget, ms, TARGET_MS);
        if (prof.nearFull) await prof.flush();
      }
      t = now();
      unconverged += (await engine.readStats()).unconverged;
      wall.stats += now() - t;
      for (let i = 0; i < batch.length; i++) {
        if (fixedMax === 0) {
          // Frame 0 sets the scale: one extra tone-map pass just to find its densest pixel (renderAnimation does the same).
          t = now();
          const px = prof.encoder('f0.');
          engine.loadHistogram(px.enc, slices, i * frameBytes);
          px.display(engine, out.view, { ...displayOf(0, 0), fixedMax: 0, countScale: baseScale });
          device.queue.submit([px.finish()]);
          fixedMax = Math.max(1, await engine.readMaxCount());
          wall.f0 += now() - t;
        }
        const px = prof.encoder();
        engine.loadHistogram(px.enc, slices, i * frameBytes);
        px.display(engine, out.view, displayOf(first + i, fixedMax));
        t = now();
        device.queue.submit([px.finish()]);
        await device.queue.onSubmittedWorkDone();
        wall.display += now() - t;
        const r = await out.read(prof);
        wall.copy += r.copyMs;
        wall.map += r.mapMs;
        wall.strip += r.stripMs;
        if (keep) kept.push(r.rgba);
        if (i === 0) {
          // The PNG stand-in for a writer, timed on the first frame of each batch only: encoding every frame would leave the GPU
          // idle for most of the run (clocks drop) and is already measured by the e2e run with sink = PNG.
          t = now();
          await encodePng(r.rgba, width, height);
          wall.png += now() - t;
          pngSamples++;
        }
        if (prof.nearFull) await prof.flush();
      }
    }
    wall.png = (wall.png / Math.max(1, pngSamples)) * frames.length; // whole-animation equivalent, kept out of the stage sum
    return { gpu: await prof.take(), wall, frames: kept, unconverged };
  } finally {
    slices?.destroy();
    out.destroy();
    engine.destroy();
  }
}

async function animE2e(k: Case, sink: 'noop' | 'png'): Promise<{ ms: number; sinkMs: number; unconverged: number }> {
  let sinkMs = 0;
  const t0 = now();
  const r = await renderAnimation(device, animRequest(k), async (rgba) => {
    if (sink === 'noop') return;
    const t = now();
    await encodePng(rgba, k.width, k.height);
    sinkMs += now() - t;
  });
  return { ms: now() - t0, sinkMs, unconverged: r.unconverged };
}

// ---------------------------------------------------------------------------------------------------------------------
// Ceilings: the shipped kernel against solve-only and splat-only variants on the same polynomials
// ---------------------------------------------------------------------------------------------------------------------

type CeilName = 'full' | Exclude<Variant, 'capture'>;
const CEIL_NAMES: CeilName[] = ['full', 'solveOnly', 'real', 'uniform', 'conc'];
const layoutBGL = variantLayout(device);
const pipes = new Map<string, GPUComputePipeline>();
const pipeFor = (kernel: Kernel, layout: Layout, v: Variant) => {
  const key = `${kernel}|${layout}|${v}`;
  let p = pipes.get(key);
  if (!p) pipes.set(key, (p = variantPipeline(device, layoutBGL, kernel, layout, v)));
  return p;
};

interface CeilRun {
  solves: number; // solves of the sample (all frames of a batch each)
  frames: number; // frames solved per polynomial in the sample (1 for stills)
  gpu: Partial<Record<CeilName, number>>; // ms, sum of the dispatch passes
  wall: Partial<Record<CeilName, number>>; // ms, submit -> onSubmittedWorkDone per chunk, summed (includes the clear)
  sweeps: number; // Aberth sweeps per solve (perPoly: per thread) / per workgroup (coop), capture run
  unconverged: { capture: number; full: number; solveOnly: number };
  /** verify only */
  check?: { realEqualsFull: boolean; solveOnlyDeposits: number; fullDeposits: number };
}

async function ceilingRun(k: Case, rep: number, verify = false): Promise<CeilRun> {
  const layout: Layout = k.mode === 'still' ? 'still' : 'frames';
  const engine = new Engine(device, 'rgba8unorm');
  engine.kernel = k.kernel;
  engine.enumeration = k.enumeration;
  engine.resize(k.width, k.height);
  const frames = framesOf(k);
  const frameBytes = k.width * k.height * 16;
  const batch = layout === 'still' ? [k.set.coeffs] : frames.slice(0, Math.min(frames.length, batchFrames(frameBytes, device.limits)));
  const F = batch.length;
  const slices = layout === 'frames' ? device.createBuffer({ size: F * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }) : null;
  const begin = () => {
    if (slices) {
      const view = viewOf(k);
      engine.setFrames({ frames: batch, views: batch.map(() => view), degree: k.degree, view, maxIter: MAX_ITER, pinned: pinnable(frames) }, slices);
    } else engine.setParams(paramsOf(k));
  };
  begin();
  const bufs = engineBuffers(engine);
  const target = slices ?? bufs.hist;
  const targetBytes = F * frameBytes;
  const chunk = CEIL_CHUNK[layout];
  const maxSample = Math.floor(Math.min(device.limits.maxBufferSize, 1 << 30) / (F * k.degree * 8));
  const sample = Math.min(Number(engine.work.total), CEIL_SAMPLE[layout], maxSample);
  const counts: number[] = [];
  for (let left = sample; left > 0; left -= chunk) counts.push(Math.min(chunk, left));
  const roots = device.createBuffer({ size: Math.max(256, sample * F * k.degree * 8), usage: GPUBufferUsage.STORAGE });
  const hdr = device.createBuffer({ size: Math.max(256, sample * F * 16), usage: GPUBufferUsage.STORAGE });
  const groups = counts.map((n, i) => {
    const first = i * chunk * F;
    return device.createBindGroup({
      layout: layoutBGL,
      entries: [
        { binding: 0, resource: { buffer: bufs.params } },
        { binding: 1, resource: { buffer: bufs.coeffs } },
        { binding: 2, resource: { buffer: bufs.en } },
        { binding: 3, resource: { buffer: target } },
        { binding: 4, resource: { buffer: bufs.stats } },
        { binding: 5, resource: { buffer: roots, offset: first * k.degree * 8, size: n * F * k.degree * 8 } },
        { binding: 6, resource: { buffer: hdr, offset: first * 16, size: n * F * 16 } },
      ],
    });
  });
  const histOf = async () => new Uint32Array(await readBuffer(device, target, targetBytes));
  const depositsOf = (h: Uint32Array) => {
    let s = 0;
    for (let i = 0; i < h.length; i += 4) s += h[i];
    return s;
  };
  try {
    const run = async (name: CeilName | 'capture') => {
      begin(); // restarts the accumulation: clears the histogram and the stats
      let wallMs = 0;
      for (let i = 0; i < counts.length; i++) {
        const px = prof.encoder();
        if (name !== 'full') px.sub = { pipeline: pipeFor(k.kernel, layout, name), bindGroup: groups[i] };
        const n = engine.encodeCompute(px.enc, counts[i]);
        if (n !== counts[i]) throw new Error(`ceiling chunk ${i}: ${n} solves encoded, ${counts[i]} expected`);
        const t0 = now();
        device.queue.submit([px.finish()]);
        await device.queue.onSubmittedWorkDone();
        wallMs += now() - t0;
        if (prof.nearFull) await prof.flush();
      }
      const gpuMs = (await prof.take()).get('solve');
      const stats = new Uint32Array(await readBuffer(device, bufs.stats, 16));
      return { gpuMs, wallMs, stats };
    };
    const cap = await run('capture'); // real roots of the sample, untimed
    await run('full'); // warm-up
    const res: CeilRun = {
      solves: sample,
      frames: F,
      gpu: {},
      wall: {},
      sweeps: cap.stats[1] / (k.kernel === 'coop' ? Math.ceil(sample / Math.floor(256 / k.degree)) * F : sample * F),
      unconverged: { capture: cap.stats[0], full: 0, solveOnly: 0 },
    };
    let fullHist: Uint32Array | null = null;
    let realHist: Uint32Array | null = null;
    let solveOnlyDeposits = -1;
    const order = CEIL_NAMES.map((_, i) => CEIL_NAMES[(i + rep) % CEIL_NAMES.length]); // alternate the order across repetitions
    for (const name of order) {
      const r = await run(name);
      if (r.gpuMs !== undefined) res.gpu[name] = r.gpuMs;
      res.wall[name] = r.wallMs;
      if (name === 'full') res.unconverged.full = r.stats[0];
      if (name === 'solveOnly') res.unconverged.solveOnly = r.stats[0];
      if (verify && name === 'full') fullHist = await histOf();
      if (verify && name === 'real') realHist = await histOf();
      if (verify && name === 'solveOnly') solveOnlyDeposits = depositsOf(await histOf());
    }
    if (verify && fullHist && realHist) {
      let equal = realHist.length === fullHist.length;
      for (let i = 0; equal && i < realHist.length; i++) equal = realHist[i] === fullHist[i];
      res.check = { realEqualsFull: equal, solveOnlyDeposits, fullDeposits: depositsOf(fullHist) };
    }
    return res;
  } finally {
    roots.destroy();
    hdr.destroy();
    slices?.destroy();
    engine.destroy();
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// One repetition of one case
// ---------------------------------------------------------------------------------------------------------------------

interface Meta {
  ceil?: { solves: number; frames: number; sweeps: number; unconverged: CeilRun['unconverged']; chunk: number };
  unconverged?: number;
}

/** Keeps the GPU busy with the case's kernel for BURN_MS (small histogram, results discarded). */
async function burn(k: Case): Promise<void> {
  const engine = new Engine(device, 'rgba8unorm');
  try {
    engine.kernel = k.kernel;
    engine.enumeration = k.enumeration;
    engine.resize(512, 512);
    const t0 = now();
    while (now() - t0 < BURN_MS) {
      engine.setParams({ ...paramsOf(k), view: fitView(512, 512, 2) });
      for (let i = 0; i < 4 && now() - t0 < BURN_MS; i++) {
        const enc = device.createCommandEncoder();
        engine.encodeCompute(enc, 1 << 16);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
      }
    }
  } finally {
    engine.destroy();
  }
}

async function repetition(k: Case, acc: Acc, rep: number, meta: Meta): Promise<void> {
  await burn(k);
  const still = k.mode === 'still';
  const gpuKeys = still ? ['clear', 'solve', 'reduce', 'tonemap', 'copy'] : ['clear', 'solve', 'loadhist', 'reduce', 'tonemap', 'copy', 'f0.loadhist', 'f0.reduce', 'f0.tonemap'];
  let e2e: number;
  let e2ePng: { ms: number; sinkMs: number } | null = null;
  if (still) e2e = await stillE2e(k);
  else {
    const r = await animE2e(k, 'noop');
    e2e = r.ms;
    meta.unconverged = r.unconverged;
    e2ePng = await animE2e(k, 'png');
  }
  const s = still ? await stillStages(k) : await animStages(k);
  push(acc, 'e2e', e2e);
  if (e2ePng) {
    push(acc, 'e2e.png', e2ePng.ms);
    push(acc, 'e2e.png.sink', e2ePng.sinkMs);
  }
  let sumGpu = 0;
  for (const key of gpuKeys) {
    const v = s.gpu.get(key);
    push(acc, `gpu.${key}`, v);
    if (v !== undefined) sumGpu += v;
  }
  if (timestamps) push(acc, 'sum.gpu', sumGpu);
  let sumWall = 0;
  for (const [key, v] of Object.entries(s.wall)) {
    push(acc, `wall.${key}`, v);
    if (key !== 'png' && key !== 'chunks') sumWall += v;
  }
  push(acc, 'sum.wall', sumWall);
  if (timestamps) push(acc, 'gap.perChunk', (s.wall.solve - (s.gpu.get('solve') ?? 0) - (s.gpu.get('clear') ?? 0)) / Math.max(1, s.wall.chunks)); // CPU round trip + GPU idle between chunks
  push(acc, 'ratio.wall', sumWall / e2e);
  if (timestamps) push(acc, 'ratio.gpu', sumGpu / e2e);
  if (CEILINGS) {
    const cr = await ceilingRun(k, rep);
    meta.ceil = { solves: cr.solves, frames: cr.frames, sweeps: cr.sweeps, unconverged: cr.unconverged, chunk: CEIL_CHUNK[still ? 'still' : 'frames'] };
    for (const n of CEIL_NAMES) {
      push(acc, `ceil.gpu.${n}`, cr.gpu[n]);
      push(acc, `ceil.wall.${n}`, cr.wall[n]);
      if (n !== 'full') {
        if (cr.gpu[n] !== undefined && cr.gpu.full) push(acc, `ceil.ratio.gpu.${n}`, cr.gpu[n]! / cr.gpu.full);
        push(acc, `ceil.ratio.wall.${n}`, cr.wall[n]! / cr.wall.full!);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------------------------------------

interface CaseRecord {
  machine: string;
  timing: string;
  timestamps: boolean;
  timestampGranularity: string;
  case: { name: string; set: string; coeffs: Complex[]; degree: number; width: number; height: number; kernel: Kernel; enumeration: Enumeration; mode: string; frames: number; maxIter: number };
  reps: number;
  units: string;
  metrics: Record<string, Spread>;
  raw: Record<string, number[]>; // every repetition's value of every metric
  perFrameMs?: Record<string, number>; // animation: median stage ms / frames
  totals: Record<string, number>;
  ceilings?: Record<string, number>;
  meta: Meta;
}
const records: CaseRecord[] = [];

function report(k: Case, acc: Acc, meta: Meta): void {
  const anim = k.mode === 'anim';
  const n = anim ? FRAMES : 1;
  const per = (key: string) => med(acc, key) / n;
  const metrics: Record<string, Spread> = {};
  for (const key of acc.keys()) metrics[key] = sp(acc, key)!;
  const e2e = sp(acc, 'e2e')!;
  const unit = anim ? 'ms/frame' : 'ms';
  const parts: string[] = [`[${caseName(k)}]`, `e2e ${range({ median: e2e.median / n, min: e2e.min / n, max: e2e.max / n })} ${unit}`];
  if (anim) parts.push(`sink=PNG ${f(med(acc, 'e2e.png') / n)} (encode ${f(med(acc, 'e2e.png.sink') / n)})`);
  if (timestamps) {
    const g = (key: string) => f(per(`gpu.${key}`));
    parts.push(
      anim
        ? `GPU: solve ${g('solve')} clear ${g('clear')} load ${g('loadhist')} reduce ${g('reduce')} tone ${g('tonemap')} copy ${g('copy')} f0 ${f((med(acc, 'gpu.f0.loadhist') + med(acc, 'gpu.f0.reduce') + med(acc, 'gpu.f0.tonemap')) / n)}`
        : `GPU: solve ${g('solve')} clear ${g('clear')} reduce ${g('reduce')} tone ${g('tonemap')} copy ${g('copy')}`,
    );
  }
  const w = (key: string) => f(per(`wall.${key}`));
  parts.push(
    anim
      ? `wall: setup ${w('setup')} setFrames ${w('setFrames')} solve ${w('solve')} stats ${w('stats')} f0 ${w('f0')} display ${w('display')} copy ${w('copy')} map ${w('map')} strip ${w('strip')} png ${w('png')}`
      : `wall: setup ${w('setup')} solve ${w('solve')} display ${w('display')} copy ${w('copy')} map ${w('map')} strip ${w('strip')}`,
  );
  parts.push(`chunks ${f(med(acc, 'wall.chunks'))}${timestamps ? `, wall-gpu gap ${f(med(acc, 'gap.perChunk'))} ms/chunk` : ''}`);
  parts.push(`sum wall/e2e ${f(med(acc, 'ratio.wall'))}${timestamps ? ` sum gpu/e2e ${f(med(acc, 'ratio.gpu'))}` : ''}`);
  const ceilings: Record<string, number> = {};
  if (meta.ceil) {
    const which = timestamps ? 'gpu' : 'wall';
    const r = (nm: string) => med(acc, `ceil.ratio.${which}.${nm}`);
    for (const nm of CEIL_NAMES) {
      if (nm !== 'full') ceilings[`ratio.${nm}`] = r(nm);
      ceilings[`ms.${nm}`] = med(acc, `ceil.${which}.${nm}`);
    }
    const u = meta.ceil.unconverged;
    parts.push(
      `ceil(${which}, ${meta.ceil.solves} solves x ${meta.ceil.frames} fr: full ${f(ceilings['ms.full'])} ms): noSplat ${f(r('solveOnly'))}x noSolve ${f(r('real'))}x uniform ${f(r('uniform'))}x conc ${f(r('conc'))}x; ` +
        `sweeps ${f(meta.ceil.sweeps)}/${k.kernel === 'coop' ? 'workgroup' : 'solve'}, unconverged cap/full/noSplat ${u.capture}/${u.full}/${u.solveOnly}`,
    );
  }
  console.log(parts.join(' | '));
  if (RAW) for (const [key, v] of [...acc].sort()) console.log(`  raw ${key}: ${v.map(f).join(' ')}`);
  const perFrameMs = anim ? Object.fromEntries([...acc.keys()].filter((key) => (key.startsWith('gpu.') || key.startsWith('wall.') || key.startsWith('e2e')) && key !== 'wall.chunks').map((key) => [key, med(acc, key) / FRAMES])) : undefined;
  records.push({
    machine,
    timing,
    timestamps,
    timestampGranularity: prof.granularity,
    case: { name: caseName(k), set: k.set.name, coeffs: k.set.coeffs, degree: k.degree, width: k.width, height: k.height, kernel: k.kernel, enumeration: k.enumeration, mode: k.mode, frames: anim ? FRAMES : 1, maxIter: MAX_ITER },
    reps: REPS,
    units: 'ms for the whole render (animation: all 16 frames); ceilings: ms for the ceiling sample, ratios to the full kernel',
    metrics,
    raw: Object.fromEntries(acc),
    perFrameMs,
    totals: { e2e: e2e.median, sumWall: med(acc, 'sum.wall'), sumGpu: med(acc, 'sum.gpu'), ratioWall: med(acc, 'ratio.wall'), ratioGpu: med(acc, 'ratio.gpu') },
    ceilings: meta.ceil ? ceilings : undefined,
    meta,
  });
  if (OUT) {
    const dir = OUT.replace(/[\\/][^\\/]*$/, '');
    if (dir && dir !== OUT) mkdirSync(dir, { recursive: true });
    writeFileSync(OUT, new TextEncoder().encode(JSON.stringify({ machine, timing, timestamps, records }, null, 1)));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------------------------

const same = (a: Uint8ClampedArray, b: Uint8ClampedArray) => a.length === b.length && a.every((v, i) => v === b[i]);

it('the replicated paths equal the shipped ones (bit-identical) and the ceiling variants do what they claim', async () => {
  const small = (kernel: Kernel, mode: 'still' | 'anim', enumeration: Enumeration = 'reciprocal'): Case => ({ set: LW, degree: mode === 'still' ? 12 : 10, width: 320, height: 200, kernel, enumeration, mode });
  for (const kernel of ['perPoly', 'coop'] as Kernel[]) {
    for (const enumeration of ['reciprocal', 'plain'] as Enumeration[]) {
      // still: renderImage vs the stage-by-stage path (same Engine calls, TimedReadback for the readback)
      const st = small(kernel, 'still', enumeration);
      const shipped = await renderImage(device, { kernel, enumeration, params: paramsOf(st), display: DISPLAY, width: st.width, height: st.height });
      const mine = await stillStages(st, true);
      expect(same(shipped, mine.frames[0]), `still ${kernel} ${enumeration}: stage path differs from renderImage`).toBe(true);
    }
    // animation: renderAnimation vs the stage-by-stage path, every frame
    const an = small(kernel, 'anim');
    const got: Uint8ClampedArray[] = [];
    await renderAnimation(device, animRequest(an), (rgba) => {
      got.push(new Uint8ClampedArray(rgba));
    });
    const mine = await animStages(an, true);
    expect(mine.frames.length).toBe(FRAMES);
    mine.frames.forEach((fr, i) => expect(same(got[i], fr), `animation ${kernel} frame ${i}: stage path differs from renderAnimation`).toBe(true));
    // ceilings: noSolve (splat only, captured roots) reproduces the full kernel's histogram bit for bit; noSplat deposits nothing
    // yet its unconverged count equals the full kernel's, and it ran Aberth sweeps (the capture counted them)
    for (const mode of ['still', 'anim'] as const) {
      const r = await ceilingRun(small(kernel, mode), 0, true);
      const label = `${kernel} ${mode}`;
      expect(r.check?.realEqualsFull, `${label}: splat-only histogram differs from the full kernel's`).toBe(true);
      expect(r.check?.fullDeposits, `${label}: nothing deposited`).toBeGreaterThan(0);
      expect(r.check?.solveOnlyDeposits, `${label}: the solve-only variant deposited`).toBe(0);
      expect(r.unconverged.solveOnly, `${label}: solve-only unconverged count`).toBe(r.unconverged.full);
      expect(r.unconverged.capture, `${label}: capture unconverged count`).toBe(r.unconverged.full);
      expect(r.sweeps, `${label}: Aberth sweeps counted`).toBeGreaterThan(0.5);
      console.log(`equivalence ${label}: stage path == shipped, splat-only histogram == full (${r.check?.fullDeposits} roots), solve-only deposits ${r.check?.solveOnlyDeposits}, sweeps ${f(r.sweeps)}`);
    }
  }
  console.log(`timestamp granularity: ${prof.granularity}`);
  expect(errors).toEqual([]);
});

it('warm-up (GPU clocks, pipelines)', async () => {
  for (const kernel of ['perPoly', 'coop'] as Kernel[]) await stillStages({ set: LW, degree: 20, width: 1920, height: 1080, kernel, enumeration: 'reciprocal', mode: 'still' });
});

// The matrix: every group runs its cases' repetitions with the kernels alternating (the laptop iGPU is thermally noisy).
const SETS = [LW, SP];
const DEGREES = QUICK ? [16, 20] : [16, 20, 22, 24];
const SIZES: [number, number][] = QUICK ? [[1024, 1024], [1920, 1080]] : [[1024, 1024], [1920, 1080], [3840, 2160]];
for (const set of SETS) {
  for (const degree of DEGREES) {
    for (const [width, height] of SIZES) {
      for (const mode of ['still', 'anim'] as const) {
        if (mode === 'anim' && degree > 22) continue; // d = 24 stills only
        const plainRow = set === LW && degree === 20 && width === 1920;
        const cases: Case[] = [];
        for (const enumeration of plainRow ? (['reciprocal', 'plain'] as const) : (['reciprocal'] as const)) {
          for (const kernel of ['perPoly', 'coop'] as const) cases.push({ set, degree, width, height, kernel, enumeration, mode });
        }
        const chosen = cases.filter((k) => caseName(k).toLowerCase().includes(FILTER));
        if (chosen.length === 0) continue;
        it(`${set.short} d=${degree} ${width}x${height} ${mode}`, async () => {
          const accs = chosen.map(() => new Map<string, number[]>());
          const metas = chosen.map((): Meta => ({}));
          for (let rep = 0; rep < REPS; rep++) {
            for (let j = 0; j < chosen.length; j++) {
              const i = (j + rep) % chosen.length; // rotate the order of the cases each repetition
              await repetition(chosen[i], accs[i], rep, metas[i]);
            }
          }
          chosen.forEach((k, i) => report(k, accs[i], metas[i]));
          expect(errors).toEqual([]);
        }, TIMEOUT);
      }
    }
  }
}
