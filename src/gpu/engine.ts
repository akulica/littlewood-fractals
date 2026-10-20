import rootsSrc from './roots.wgsl?raw';
import coopSrc from './coop.wgsl?raw';
import displaySrc from './display.wgsl?raw';
import { MAX_BATCH_FRAMES } from '../core/animation';
import type { Complex } from '../core/complex';
import { MAX_COEFFS, MAX_DEGREE, leadingFixable } from '../core/enumeration';
import {
  ENUM_U32,
  EnumCursor,
  REGION_BASE,
  effectiveEnumeration,
  enumerationPlan,
  mixedDigits,
  packChunk,
  packPlan,
  type EnumPlan,
  type Enumeration,
} from '../core/reciprocal';
import { chooseKernel, coopGrid, dispatchGrid, nextBudget, type Kernel } from '../core/schedule';
import type { View } from '../core/view';

export { MAX_COEFFS, MAX_DEGREE };
export type { Enumeration };
export const TOL = 1e-5; // backward-error stop (measured: 1e-6..3e-5 all converge; 1e-5 is fastest without visible change)
const PARAMS_BYTES = 64;
const VIEW_BYTES = 176; // = struct View in display.wgsl: 12 scalars + 4 pad, then 8 vec4f of ring centres
const MAX_RINGS = 16; // = MAX_RINGS in display.wgsl
const FRAME_VIEWS = MAX_COEFFS * MAX_BATCH_FRAMES; // = FRAME_VIEWS in roots.wgsl and coop.wgsl: vec2f index of the per-frame views
const REDUCE_GROUPS = 256;
const CAL_DEGREE = 20; // calibration workload: 2^18 Littlewood polynomials of degree 20 (~15-70 ms per run)
const CAL_POLYS = 1 << 18;
const CAL_LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];

/** golden: sample m is polynomial m*G mod N (G ~ 0.618 N), so every prefix looks like the full picture.
 * lexicographic: m itself (tests/debugging). Both cover every polynomial exactly once. */
export type Ordering = 'golden' | 'lexicographic';

export interface Calibration {
  kernel: Kernel;
  perPolyMs: number;
  coopMs: number;
}

export interface FractalParams {
  coeffs: Complex[];
  degree: number;
  view: View;
  maxIter: number;
}

/** Coefficient rings drawn over an export, like the screen's handles. */
export interface RingOverlay {
  /** Ring centres in output pixels (core/view worldToPixel); at most 16 are drawn. */
  centres: { x: number; y: number }[];
  /** Output px per CSS px: rings, border and halo scale like a screenshot taken at the export's resolution. */
  scale: number;
}

export interface DisplayParams {
  colorMode: 0 | 1;
  exposure: number;
  gamma: number;
  /** Multiplies root counts before tone-mapping (default 1). An export at k times the screen's pixels per unit
   * gets about k^2 fewer roots per pixel; countScale = k^2 makes it look like the screen. */
  countScale?: number;
  /** > 0: tone-map against this root count instead of the frame's densest pixel (animation: locked brightness). */
  fixedMax?: number;
  /** Exports only (the live view has DOM handles). */
  rings?: RingOverlay;
}

/** Animation kernel of each splat kernel: roots.wgsl splatFrames (perPoly) or coop.wgsl splatFramesCoop (coop). */
export type FramesKernel = 'splatFrames' | 'splatFramesCoop';
export const framesKernel = (kernel: Kernel): FramesKernel => (kernel === 'coop' ? 'splatFramesCoop' : 'splatFrames');

/** One splatFrames / splatFramesCoop batch: up to MAX_BATCH_FRAMES coefficient sets of equal size k, solved per polynomial in turn. */
export interface FramesParams {
  frames: Complex[][];
  degree: number;
  view: View;
  maxIter: number;
  /** Pin the leading digit in every frame (core/animation pinnable(): decided once for the whole animation). */
  pinned: boolean;
  /** Per-frame views (animations that zoom or pan); default: `view` for every frame. */
  views?: View[];
}

/**
 * GPU resources + chunked dispatch. Device-injected so it runs in the browser and under Dawn in tests.
 * Contract: at most ONE compute dispatch per submitted command buffer (queue.writeBuffer happens at
 * submit time, so two dispatches in one encoder would both see the second chunk's parameters).
 */
export class Engine {
  readonly device: GPUDevice;
  width = 0;
  height = 0;
  /** Which splat kernel encodeCompute dispatches. calibrate() sets it to the faster one for this GPU. setFrames()
   * reads it too: perPoly batches run splatFrames, coop batches splatFramesCoop. */
  kernel: Kernel = 'perPoly';
  /** Enumeration order of the polynomials; read by setParams/setFrames (the next accumulation). */
  ordering: Ordering = 'golden';
  /** reciprocal: solve one polynomial of each reversal pair and deposit z and 1/z (about half the solves, the same
   * picture); plain: solve every polynomial (the original path). Read by setParams/setFrames. */
  enumeration: Enumeration = 'reciprocal';
  /** Unverified option (fails `npm run ab`, kept for comparison): every kernel's polish() evaluates p at z directly,
   * without the reversed evaluation for |z| > 1. Big roots then overflow f32 and are accepted unconverged:
   * off-screen with plain enumeration, a false ring near the origin with reciprocal. Read at every dispatch; callers
   * restart accumulation when they change it. */
  skipOverflowGuard = false;
  private readonly splatPipe: GPUComputePipeline;
  private readonly coopPipe: GPUComputePipeline;
  private readonly debugPipe: GPUComputePipeline;
  private readonly rootsModule: GPUShaderModule;
  private readonly coopModule: GPUShaderModule;
  // Built on first use: exports and the live view never need them.
  private readonly framesPipes: Partial<Record<FramesKernel, GPUComputePipeline>> = {};
  private readonly reducePipe: GPUComputePipeline;
  private readonly renderPipe: GPURenderPipeline;
  private readonly paramsBuf: GPUBuffer;
  private readonly coeffBuf: GPUBuffer;
  private readonly enumBuf: GPUBuffer;
  private readonly statsBuf: GPUBuffer;
  private readonly viewBuf: GPUBuffer;
  private readonly maxBuf: GPUBuffer;
  private hist: GPUBuffer | null = null;
  private splatBG: GPUBindGroup | null = null;
  private coopBG: GPUBindGroup | null = null;
  private reduceBG: GPUBindGroup | null = null;
  private renderBG: GPUBindGroup | null = null;
  private params: FractalParams | null = null;
  private frameCount = 0; // > 0: batch mode (setFrames), encodeCompute dispatches framesKernel into framesTarget
  private framesTarget: GPUBuffer | null = null;
  private framesBG: GPUBindGroup | null = null;
  private framesUsed: FramesKernel | null = null;
  private plan: EnumPlan | null = null;
  private cursor: EnumCursor | null = null;
  private needsClear = true;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    const roots = device.createShaderModule({ label: 'roots', code: rootsSrc });
    this.rootsModule = roots;
    const display = device.createShaderModule({ label: 'display', code: displaySrc });
    this.splatPipe = device.createComputePipeline({ layout: 'auto', compute: { module: roots, entryPoint: 'splat' } });
    this.debugPipe = device.createComputePipeline({ layout: 'auto', compute: { module: roots, entryPoint: 'debugRoots' } });
    const coop = device.createShaderModule({ label: 'coop', code: coopSrc });
    this.coopModule = coop;
    this.coopPipe = device.createComputePipeline({ layout: 'auto', compute: { module: coop, entryPoint: 'splatCoop' } });
    this.reducePipe = device.createComputePipeline({ layout: 'auto', compute: { module: display, entryPoint: 'reduceMax' } });
    this.renderPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: display, entryPoint: 'vs' },
      fragment: { module: display, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    const U = GPUBufferUsage;
    this.paramsBuf = device.createBuffer({ size: PARAMS_BYTES, usage: U.UNIFORM | U.COPY_DST });
    // Coefficients of up to MAX_BATCH_FRAMES frames, then two vec2f per frame for splatFrames' per-frame views.
    this.coeffBuf = device.createBuffer({ size: (FRAME_VIEWS + 2 * MAX_BATCH_FRAMES) * 8, usage: U.STORAGE | U.COPY_DST });
    this.enumBuf = device.createBuffer({ size: ENUM_U32 * 4, usage: U.STORAGE | U.COPY_DST });
    this.statsBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.viewBuf = device.createBuffer({ size: VIEW_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.maxBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  }

  /** Degree currently being accumulated. */
  get degree(): number {
    return this.params?.degree ?? 0;
  }

  /** Polynomials covered so far (a solve with a reciprocal partner covers two), of all polynomials. */
  get progress(): { done: bigint; total: bigint } {
    return { done: this.cursor?.covered ?? 0n, total: this.plan?.polys ?? 0n };
  }

  /** Solves so far, of all solves: the unit of encodeCompute's budget. */
  get work(): { done: bigint; total: bigint } {
    return { done: this.cursor?.solved ?? 0n, total: this.plan?.solves ?? 0n };
  }

  /** Animation kernel of the current setFrames batch, null outside batch mode. */
  get activeFramesKernel(): FramesKernel | null {
    return this.framesUsed;
  }

  /** Enumeration actually in use (a pinned set without a usable group table runs plain). */
  get activeEnumeration(): Enumeration | null {
    return this.plan?.mode ?? null;
  }

  /** (Re)allocates the W x H histogram (16 bytes per pixel) and restarts accumulation. */
  resize(width: number, height: number): void {
    if (this.hist && width === this.width && height === this.height) return;
    const bytes = width * height * 16;
    if (bytes > this.device.limits.maxStorageBufferBindingSize) {
      throw new RangeError(`histogram ${width}x${height} exceeds maxStorageBufferBindingSize`);
    }
    this.hist?.destroy();
    this.hist = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.width = width;
    this.height = height;
    const bind = (pipe: GPUComputePipeline | GPURenderPipeline, buffers: Record<number, GPUBuffer>) =>
      this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: Object.entries(buffers).map(([b, buffer]) => ({ binding: Number(b), resource: { buffer } })),
      });
    const splatBuffers = { 0: this.paramsBuf, 1: this.coeffBuf, 2: this.enumBuf, 3: this.hist, 4: this.statsBuf };
    this.splatBG = bind(this.splatPipe, splatBuffers);
    this.coopBG = bind(this.coopPipe, splatBuffers);
    this.reduceBG = bind(this.reducePipe, { 0: this.viewBuf, 1: this.hist, 2: this.maxBuf });
    this.renderBG = bind(this.renderPipe, { 0: this.viewBuf, 1: this.hist, 3: this.maxBuf });
    if (this.frameCount > 0) {
      // The batch's W x H slices no longer match: setFrames() again before the next encodeCompute.
      this.leaveFrames();
      this.params = null;
      this.plan = null;
    }
    this.restart();
  }

  /** New coefficients/degree/view: restart accumulation from polynomial 0. */
  setParams(p: FractalParams): void {
    if (!Number.isInteger(p.degree) || p.degree < 1 || p.degree > MAX_DEGREE) {
      throw new RangeError(`degree must be an integer in 1..${MAX_DEGREE}`);
    }
    if (p.coeffs.length < 1 || p.coeffs.length > MAX_COEFFS) throw new RangeError(`need 1..${MAX_COEFFS} coefficients`);
    this.leaveFrames();
    this.params = p;
    const pinned = leadingFixable(p.coeffs);
    const { group } = effectiveEnumeration([p.coeffs], pinned, this.enumeration);
    this.load(enumerationPlan(p.coeffs.length, p.degree, pinned, this.enumeration, group));
    this.device.queue.writeBuffer(this.coeffBuf, 0, new Float32Array(p.coeffs.flatMap((c) => [c.re, c.im])));
  }

  /**
   * Batch mode for animation: frame f of `p.frames` accumulates into bytes [f, f+1) * W*H*16 of `target`
   * (STORAGE | COPY_DST usage, at least frames * W*H*16 bytes) through the warm-starting splatFrames kernel; the
   * engine's own histogram is untouched (loadHistogram() copies a slice into it for display). `kernel` picks the
   * layout: perPoly runs splatFrames (a thread per polynomial), coop splatFramesCoop (a thread per root). Restarts accumulation;
   * every encodeCompute chunk then covers the same polynomials in all frames. Stays in effect until setParams() or
   * resize(). Pinned + reciprocal needs every frame to give the same group table; otherwise the batch runs plain.
   */
  setFrames(p: FramesParams, target: GPUBuffer): void {
    const n = p.frames.length;
    if (n < 1 || n > MAX_BATCH_FRAMES) throw new RangeError(`a batch holds 1..${MAX_BATCH_FRAMES} frames`);
    const k = p.frames[0].length;
    if (p.frames.some((f) => f.length !== k)) throw new RangeError('every frame needs the same number of coefficients');
    if (!this.hist) throw new Error('call resize() before setFrames()');
    if (target.size < n * this.width * this.height * 16) throw new RangeError(`target holds fewer than ${n} frames of ${this.width}x${this.height}`);
    this.setParams({ coeffs: p.frames[0], degree: p.degree, view: p.view, maxIter: p.maxIter }); // validates degree and k
    const { group } = effectiveEnumeration(p.frames, p.pinned, this.enumeration);
    this.load(enumerationPlan(k, p.degree, p.pinned, this.enumeration, group));
    if (p.views && p.views.length !== n) throw new RangeError('need one view per frame');
    this.device.queue.writeBuffer(this.coeffBuf, 0, new Float32Array(p.frames.flatMap((f) => f.flatMap((c) => [c.re, c.im]))));
    const views = p.views ?? p.frames.map(() => p.view);
    this.device.queue.writeBuffer(this.coeffBuf, FRAME_VIEWS * 8, new Float32Array(views.flatMap((v) => [v.cx, v.cy, v.pxPerUnit, 0])));
    const used = framesKernel(this.kernel);
    const pipe = (this.framesPipes[used] ??= this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: used === 'splatFramesCoop' ? this.coopModule : this.rootsModule, entryPoint: used },
    }));
    this.frameCount = n;
    this.framesTarget = target;
    this.framesUsed = used;
    this.framesBG = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [this.paramsBuf, this.coeffBuf, this.enumBuf, target, this.statsBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  /**
   * Encodes the next chunk (<= budget solves) as ONE dispatch. Returns how many solves it holds: each covers one
   * polynomial, or two with a reciprocal partner (progress counts polynomials, work counts solves). Every chunk
   * samples all regions of the enumeration in proportion, so any prefix is an even sample of the whole set.
   */
  encodeCompute(enc: GPUCommandEncoder, budget: number): number {
    if (!this.params || !this.plan || !this.cursor || !this.hist || !this.splatBG || !this.coopBG) return 0;
    const remaining = this.plan.solves - this.cursor.solved;
    const count = remaining <= 0n ? 0 : remaining < BigInt(budget) ? Number(remaining) : Math.floor(budget);
    // The shaders build slot digit i as start_i + G_i * local + carry in u32; that sum stays below radix * count.
    // MAX_BUDGET (2^22) keeps it below 2^32 for radix <= 256 (k^2 with k <= 16); checked before encoding anything.
    if (this.plan.maxRadix * count >= 2 ** 32) throw new RangeError(`chunk of ${count} polynomials overflows the u32 digit arithmetic`);
    if (this.needsClear) {
      if (this.framesTarget) enc.clearBuffer(this.framesTarget, 0, this.frameCount * this.width * this.height * 16);
      else enc.clearBuffer(this.hist);
      enc.clearBuffer(this.statsBuf);
      this.needsClear = false;
    }
    if (count === 0) return 0;
    const entries = this.cursor.next(count);
    this.device.queue.writeBuffer(this.enumBuf, 0, packChunk(entries));
    const pass = enc.beginComputePass();
    if (this.framesBG && this.framesUsed === 'splatFramesCoop') {
      // A workgroup keeps its polynomials' roots in registers and shared memory from frame to frame.
      const grid = coopGrid(count, this.params.degree);
      this.writeParams(count, 0, entries.length);
      pass.setPipeline(this.framesPipes.splatFramesCoop!);
      pass.setBindGroup(0, this.framesBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    } else if (this.framesBG) {
      // A thread keeps its roots in registers from frame to frame.
      const grid = dispatchGrid(count);
      this.writeParams(count, grid.strideX, entries.length);
      pass.setPipeline(this.framesPipes.splatFrames!);
      pass.setBindGroup(0, this.framesBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    } else if (this.kernel === 'coop') {
      const grid = coopGrid(count, this.params.degree);
      this.writeParams(count, 0, entries.length); // coop.wgsl derives the index from num_workgroups
      pass.setPipeline(this.coopPipe);
      pass.setBindGroup(0, this.coopBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    } else {
      const grid = dispatchGrid(count);
      this.writeParams(count, grid.strideX, entries.length);
      pass.setPipeline(this.splatPipe);
      pass.setBindGroup(0, this.splatBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    }
    pass.end();
    return count;
  }

  /**
   * Times both splat kernels on the same workload (interleaved, best of `rounds` after a warm-up) and keeps
   * the faster one in `this.kernel` (cooperative only on a >10 % win). Needs resize() first. Leaves the
   * calibration polynomials loaded with accumulation reset — call setParams() afterwards.
   */
  async calibrate(rounds = 3): Promise<Calibration> {
    if (!this.hist) throw new Error('call resize() before calibrate()');
    const time = async (kernel: Kernel): Promise<number> => {
      this.kernel = kernel;
      this.setParams({ coeffs: CAL_LW, degree: CAL_DEGREE, view: { cx: 0, cy: 0, pxPerUnit: Math.min(this.width, this.height) / 4 }, maxIter: 32 });
      const enc = this.device.createCommandEncoder();
      this.encodeCompute(enc, CAL_POLYS);
      const t0 = performance.now();
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await time('perPoly'); // warm-up: first dispatches pay for pipeline compilation and clock ramp-up
    await time('coop');
    let perPolyMs = Infinity;
    let coopMs = Infinity;
    for (let i = 0; i < rounds; i++) {
      perPolyMs = Math.min(perPolyMs, await time('perPoly'));
      coopMs = Math.min(coopMs, await time('coop'));
    }
    this.kernel = chooseKernel(perPolyMs, coopMs);
    this.restart();
    return { kernel: this.kernel, perPolyMs, coopMs };
  }

  /** Max-reduction + fullscreen tone-map into `target` (must be width x height). */
  encodeDisplay(enc: GPUCommandEncoder, target: GPUTextureView, d: DisplayParams): void {
    if (!this.reduceBG || !this.renderBG) return;
    const ab = new ArrayBuffer(VIEW_BYTES);
    const rings = d.rings?.centres.slice(0, MAX_RINGS) ?? [];
    new Uint32Array(ab).set([this.width, this.height, d.colorMode, rings.length]);
    new Float32Array(ab).set([d.exposure, d.gamma, d.countScale ?? 1, d.fixedMax ?? 0, d.rings?.scale ?? 1], 4);
    new Float32Array(ab).set(rings.flatMap((r) => [r.x, r.y]), 12);
    this.device.queue.writeBuffer(this.viewBuf, 0, ab);
    enc.clearBuffer(this.maxBuf);
    const c = enc.beginComputePass();
    c.setPipeline(this.reducePipe);
    c.setBindGroup(0, this.reduceBG);
    c.dispatchWorkgroups(REDUCE_GROUPS);
    c.end();
    const r = enc.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    r.setPipeline(this.renderPipe);
    r.setBindGroup(0, this.renderBG);
    r.draw(3);
    r.end();
  }

  /**
   * Copies one W x H histogram (e.g. a frame slice of a setFrames target; byteOffset a multiple of 4) into the
   * engine's own histogram, so encodeDisplay shows it. A copy rather than a bind-group offset, which would need
   * 256-byte alignment and a bind group per frame.
   */
  loadHistogram(enc: GPUCommandEncoder, src: GPUBuffer, byteOffset: number): void {
    if (!this.hist) throw new Error('call resize() before loadHistogram()');
    enc.copyBufferToBuffer(src, byteOffset, this.hist, 0, this.width * this.height * 16);
    if (this.frameCount === 0) this.needsClear = false; // the copy is the content now; a pending clear would wipe it
  }

  /** Densest pixel's root count from the last encodeDisplay (which must have been submitted). */
  async readMaxCount(): Promise<number> {
    return new Uint32Array(await this.readBuffer(this.maxBuf, 4))[0];
  }

  /** Test/debug: roots of polynomials first .. first+count-1 (lexicographic, wrapping mod N, plain enumeration);
   * `degree` vec2f slots each. Resets accumulation. */
  async solveDebug(coeffs: Complex[], degree: number, first: bigint, count: number): Promise<Float32Array> {
    this.setParams({ coeffs, degree, view: { cx: 0, cy: 0, pxPerUnit: 1 }, maxIter: 32 });
    const plan = enumerationPlan(coeffs.length, degree, leadingFixable(coeffs), 'plain', null);
    this.load(plan);
    const bytes = count * degree * 8;
    const out = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const grid = dispatchGrid(count);
    // Plain is one region whose index is the base-k polynomial index: polynomial t of the output is index first + t.
    const [region] = plan.regions;
    const radices = region.slots.map((s) => s.radix);
    const entry = { region: 0, threadStart: 0, count, start: mixedDigits(first % region.size, radices), stride: mixedDigits(1n, radices) };
    this.device.queue.writeBuffer(this.enumBuf, 0, packChunk([entry]));
    this.writeParams(count, grid.strideX, 1);
    const bg = this.device.createBindGroup({
      layout: this.debugPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.coeffBuf } },
        { binding: 2, resource: { buffer: this.enumBuf } },
        { binding: 5, resource: { buffer: out } },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.debugPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(grid.x, grid.y);
    pass.end();
    const result = new Float32Array(await this.readBuffer(out, bytes, enc));
    out.destroy();
    return result;
  }

  /** Frees every GPU buffer (the histogram can be hundreds of MB). The engine is unusable afterwards. */
  destroy(): void {
    this.hist?.destroy();
    this.hist = null;
    for (const b of [this.paramsBuf, this.coeffBuf, this.enumBuf, this.statsBuf, this.viewBuf, this.maxBuf]) b.destroy();
  }

  async readHistogram(): Promise<Uint32Array> {
    if (!this.hist) return new Uint32Array(0);
    return new Uint32Array(await this.readBuffer(this.hist, this.width * this.height * 16));
  }

  async readStats(): Promise<{ unconverged: number }> {
    return { unconverged: new Uint32Array(await this.readBuffer(this.statsBuf, 16))[0] };
  }

  private writeParams(count: number, strideX: number, entries: number): void {
    const p = this.params!;
    const ab = new ArrayBuffer(PARAMS_BYTES);
    new Uint32Array(ab).set([p.degree, p.coeffs.length, count, strideX, this.width, this.height, p.maxIter, entries]);
    new Float32Array(ab).set([p.view.cx, p.view.cy, p.view.pxPerUnit, TOL], 8);
    new Uint32Array(ab).set([Math.max(this.frameCount, 1), this.skipOverflowGuard ? 1 : 0], 12);
    this.device.queue.writeBuffer(this.paramsBuf, 0, ab);
  }

  /** Uses `plan` from now on (its regions and tables go to the GPU once) and restarts accumulation. */
  private load(plan: EnumPlan): void {
    this.plan = plan;
    this.device.queue.writeBuffer(this.enumBuf, REGION_BASE * 4, packPlan(plan));
    this.restart();
  }

  private restart(): void {
    this.cursor = this.plan && new EnumCursor(this.plan, this.ordering === 'golden');
    this.needsClear = true;
  }

  private leaveFrames(): void {
    this.frameCount = 0;
    this.framesTarget = null;
    this.framesBG = null;
    this.framesUsed = null;
  }

  private async readBuffer(src: GPUBuffer, bytes: number, enc = this.device.createCommandEncoder()): Promise<ArrayBuffer> {
    const rb = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const copy = rb.getMappedRange().slice(0);
    rb.unmap();
    rb.destroy();
    return copy;
  }
}

export interface ImageRequest {
  kernel: Kernel;
  /** Default reciprocal. */
  enumeration?: Enumeration;
  /** Engine.skipOverflowGuard (unverified option); default false. */
  skipOverflowGuard?: boolean;
  params: FractalParams;
  display: DisplayParams;
  width: number;
  height: number;
}

const EXPORT_TARGET_MS = 100; // per chunk while exporting: big chunks, still far below the ~2 s Windows TDR

/**
 * Off-screen render at an arbitrary size: a private Engine computes every polynomial into a width x height
 * histogram, tone-maps it into an rgba8unorm texture and reads it back as tightly packed RGBA (row padding
 * removed). Size limits are the caller's job (see core/export.ts).
 */
export async function renderImage(
  device: GPUDevice,
  req: ImageRequest,
  onProgress?: (done: bigint, total: bigint) => void,
): Promise<Uint8ClampedArray<ArrayBuffer>> {
  const { width, height } = req;
  const engine = new Engine(device, 'rgba8unorm');
  const out = new RgbaReadback(device, width, height);
  try {
    engine.kernel = req.kernel;
    engine.enumeration = req.enumeration ?? 'reciprocal';
    engine.skipOverflowGuard = req.skipOverflowGuard ?? false;
    engine.resize(width, height);
    engine.setParams(req.params);
    let budget = 1 << 16;
    for (;;) {
      const enc = device.createCommandEncoder();
      const count = engine.encodeCompute(enc, budget);
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (count === 0) break;
      budget = nextBudget(budget, performance.now() - t0, EXPORT_TARGET_MS);
      onProgress?.(engine.progress.done, engine.progress.total);
    }
    const enc = device.createCommandEncoder();
    engine.encodeDisplay(enc, out.view, req.display);
    return await out.read(enc);
  } finally {
    out.destroy();
    engine.destroy();
  }
}

/** Off-screen rgba8unorm target plus its readback buffer (exports and animation frames). */
export class RgbaReadback {
  readonly view: GPUTextureView;
  private readonly device: GPUDevice;
  private readonly texture: GPUTexture;
  private readonly buffer: GPUBuffer;
  private readonly bytesPerRow: number;

  constructor(device: GPUDevice, readonly width: number, readonly height: number) {
    this.device = device;
    this.texture = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.view = this.texture.createView();
    this.bytesPerRow = Math.ceil((width * 4) / 256) * 256; // copyTextureToBuffer requires 256-byte rows
    this.buffer = device.createBuffer({ size: this.bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  /** Appends the texture copy to `enc` (which should have drawn into `view`), submits it and returns tightly
   * packed RGBA, row padding removed. */
  async read(enc: GPUCommandEncoder): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const { width, height, bytesPerRow } = this;
    enc.copyTextureToBuffer({ texture: this.texture }, { buffer: this.buffer, bytesPerRow }, [width, height]);
    this.device.queue.submit([enc.finish()]);
    await this.buffer.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(this.buffer.getMappedRange());
    const out = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    this.buffer.unmap();
    return out;
  }

  destroy(): void {
    this.buffer.destroy();
    this.texture.destroy();
  }
}
