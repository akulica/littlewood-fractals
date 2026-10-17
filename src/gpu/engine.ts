import rootsSrc from './roots.wgsl?raw';
import coopSrc from './coop.wgsl?raw';
import displaySrc from './display.wgsl?raw';
import type { Complex } from '../core/complex';
import { indexToDigits, leadingFixable, totalPolys } from '../core/enumeration';
import { chooseKernel, coopGrid, dispatchGrid, nextBudget, type Kernel } from '../core/schedule';
import type { View } from '../core/view';

export const MAX_DEGREE = 32; // = MAX_DEG in roots.wgsl
export const MAX_COEFFS = 16;
const TOL = 1e-5; // backward-error stop (measured: 1e-6..3e-5 all converge; 1e-5 is fastest without visible change)
const PARAMS_BYTES = 48;
const VIEW_BYTES = 32;
const REDUCE_GROUPS = 256;
const CAL_DEGREE = 20; // calibration workload: 2^18 Littlewood polynomials of degree 20 (~15-70 ms per run)
const CAL_POLYS = 1 << 18;
const CAL_LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];

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

export interface DisplayParams {
  colorMode: 0 | 1;
  exposure: number;
  gamma: number;
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
  /** Which splat kernel encodeCompute dispatches. calibrate() sets it to the faster one for this GPU. */
  kernel: Kernel = 'perPoly';
  private readonly splatPipe: GPUComputePipeline;
  private readonly coopPipe: GPUComputePipeline;
  private readonly debugPipe: GPUComputePipeline;
  private readonly reducePipe: GPUComputePipeline;
  private readonly renderPipe: GPURenderPipeline;
  private readonly paramsBuf: GPUBuffer;
  private readonly coeffBuf: GPUBuffer;
  private readonly digitsBuf: GPUBuffer;
  private readonly statsBuf: GPUBuffer;
  private readonly viewBuf: GPUBuffer;
  private readonly maxBuf: GPUBuffer;
  private hist: GPUBuffer | null = null;
  private splatBG: GPUBindGroup | null = null;
  private coopBG: GPUBindGroup | null = null;
  private reduceBG: GPUBindGroup | null = null;
  private renderBG: GPUBindGroup | null = null;
  private params: FractalParams | null = null;
  private total = 0n;
  private next = 0n;
  private needsClear = true;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    const roots = device.createShaderModule({ label: 'roots', code: rootsSrc });
    const display = device.createShaderModule({ label: 'display', code: displaySrc });
    this.splatPipe = device.createComputePipeline({ layout: 'auto', compute: { module: roots, entryPoint: 'splat' } });
    this.debugPipe = device.createComputePipeline({ layout: 'auto', compute: { module: roots, entryPoint: 'debugRoots' } });
    const coop = device.createShaderModule({ label: 'coop', code: coopSrc });
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
    this.coeffBuf = device.createBuffer({ size: MAX_COEFFS * 8, usage: U.STORAGE | U.COPY_DST });
    this.digitsBuf = device.createBuffer({ size: (MAX_DEGREE + 1) * 4, usage: U.STORAGE | U.COPY_DST });
    this.statsBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.viewBuf = device.createBuffer({ size: VIEW_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.maxBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST });
  }

  /** Degree currently being accumulated (may be a preview degree). */
  get degree(): number {
    return this.params?.degree ?? 0;
  }

  get progress(): { done: bigint; total: bigint } {
    return { done: this.next, total: this.total };
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
    const splatBuffers = { 0: this.paramsBuf, 1: this.coeffBuf, 2: this.digitsBuf, 3: this.hist, 4: this.statsBuf };
    this.splatBG = bind(this.splatPipe, splatBuffers);
    this.coopBG = bind(this.coopPipe, splatBuffers);
    this.reduceBG = bind(this.reducePipe, { 0: this.viewBuf, 1: this.hist, 2: this.maxBuf });
    this.renderBG = bind(this.renderPipe, { 0: this.viewBuf, 1: this.hist, 3: this.maxBuf });
    this.next = 0n;
    this.needsClear = true;
  }

  /** New coefficients/degree/view: restart accumulation from polynomial 0. */
  setParams(p: FractalParams): void {
    if (!Number.isInteger(p.degree) || p.degree < 1 || p.degree > MAX_DEGREE) {
      throw new RangeError(`degree must be an integer in 1..${MAX_DEGREE}`);
    }
    if (p.coeffs.length < 1 || p.coeffs.length > MAX_COEFFS) throw new RangeError(`need 1..${MAX_COEFFS} coefficients`);
    this.params = p;
    this.total = totalPolys(p.coeffs.length, p.degree, leadingFixable(p.coeffs));
    this.next = 0n;
    this.needsClear = true;
    this.device.queue.writeBuffer(this.coeffBuf, 0, new Float32Array(p.coeffs.flatMap((c) => [c.re, c.im])));
  }

  /** Encodes the next chunk (<= budget polynomials) as ONE dispatch. Returns how many polynomials it covers. */
  encodeCompute(enc: GPUCommandEncoder, budget: number): number {
    if (!this.params || !this.hist || !this.splatBG || !this.coopBG) return 0;
    if (this.needsClear) {
      enc.clearBuffer(this.hist);
      enc.clearBuffer(this.statsBuf);
      this.needsClear = false;
    }
    const remaining = this.total - this.next;
    if (remaining <= 0n) return 0;
    const count = remaining < BigInt(budget) ? Number(remaining) : Math.floor(budget);
    this.device.queue.writeBuffer(this.digitsBuf, 0, indexToDigits(this.next, this.params.coeffs.length, this.params.degree));
    const pass = enc.beginComputePass();
    if (this.kernel === 'coop') {
      const grid = coopGrid(count, this.params.degree);
      this.writeParams(count, 0); // coop.wgsl derives the index from num_workgroups
      pass.setPipeline(this.coopPipe);
      pass.setBindGroup(0, this.coopBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    } else {
      const grid = dispatchGrid(count);
      this.writeParams(count, grid.strideX);
      pass.setPipeline(this.splatPipe);
      pass.setBindGroup(0, this.splatBG);
      pass.dispatchWorkgroups(grid.x, grid.y);
    }
    pass.end();
    this.next += BigInt(count);
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
    this.next = 0n;
    this.needsClear = true;
    return { kernel: this.kernel, perPolyMs, coopMs };
  }

  /** Max-reduction + fullscreen tone-map into `target` (must be width x height). */
  encodeDisplay(enc: GPUCommandEncoder, target: GPUTextureView, d: DisplayParams): void {
    if (!this.reduceBG || !this.renderBG) return;
    const ab = new ArrayBuffer(VIEW_BYTES);
    new Uint32Array(ab).set([this.width, this.height, d.colorMode, 0]);
    new Float32Array(ab).set([d.exposure, d.gamma], 4);
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

  /** Test/debug: roots of `count` polynomials starting at index `first`; `degree` vec2f slots each. Resets accumulation. */
  async solveDebug(coeffs: Complex[], degree: number, first: bigint, count: number): Promise<Float32Array> {
    this.setParams({ coeffs, degree, view: { cx: 0, cy: 0, pxPerUnit: 1 }, maxIter: 32 });
    const bytes = count * degree * 8;
    const out = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const grid = dispatchGrid(count);
    this.device.queue.writeBuffer(this.digitsBuf, 0, indexToDigits(first, coeffs.length, degree));
    this.writeParams(count, grid.strideX);
    const bg = this.device.createBindGroup({
      layout: this.debugPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.coeffBuf } },
        { binding: 2, resource: { buffer: this.digitsBuf } },
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
    for (const b of [this.paramsBuf, this.coeffBuf, this.digitsBuf, this.statsBuf, this.viewBuf, this.maxBuf]) b.destroy();
  }

  async readHistogram(): Promise<Uint32Array> {
    if (!this.hist) return new Uint32Array(0);
    return new Uint32Array(await this.readBuffer(this.hist, this.width * this.height * 16));
  }

  async readStats(): Promise<{ unconverged: number }> {
    return { unconverged: new Uint32Array(await this.readBuffer(this.statsBuf, 16))[0] };
  }

  private writeParams(count: number, strideX: number): void {
    const p = this.params!;
    const ab = new ArrayBuffer(PARAMS_BYTES);
    new Uint32Array(ab).set([p.degree, p.coeffs.length, count, strideX, this.width, this.height, p.maxIter, 0]);
    new Float32Array(ab).set([p.view.cx, p.view.cy, p.view.pxPerUnit, TOL], 8);
    this.device.queue.writeBuffer(this.paramsBuf, 0, ab);
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
): Promise<Uint8ClampedArray> {
  const { width, height } = req;
  const engine = new Engine(device, 'rgba8unorm');
  const tex = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256; // copyTextureToBuffer requires 256-byte rows
  const readback = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    engine.kernel = req.kernel;
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
    engine.encodeDisplay(enc, tex.createView(), req.display);
    enc.copyTextureToBuffer({ texture: tex }, { buffer: readback, bytesPerRow }, [width, height]);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(readback.getMappedRange());
    const out = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    readback.unmap();
    return out;
  } finally {
    readback.destroy();
    tex.destroy();
    engine.destroy();
  }
}
