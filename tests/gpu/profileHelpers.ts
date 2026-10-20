import coopSrc from '../../src/gpu/coop.wgsl?raw';
import rootsSrc from '../../src/gpu/roots.wgsl?raw';
import type { Kernel } from '../../src/core/schedule';
import type { Engine } from '../../src/gpu/engine';

// Helpers of tests/gpu/profile.perf.ts: GPU timestamps for the passes the Engine encodes (without touching src/), the
// readback path with timed phases, and the test-only WGSL variants (ceiling experiments). Nothing here is used by the app.

export const now = (): number => performance.now();

export interface Spread {
  median: number;
  min: number;
  max: number;
}
export function spread(xs: number[]): Spread {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return { median: s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2, min: s[0], max: s[s.length - 1] };
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU timestamps
// ---------------------------------------------------------------------------------------------------------------------

const BLOCK = 32; // queries per command buffer: 32 x 8 bytes = 256, the granularity of resolveQuerySet's destination offset
const BLOCKS = 128; // 128 x 32 = 4096 queries, the most a query set may hold
const BIG = 1 << 20; // copies and clears below this are not timed (stats, densest-pixel word)

interface Interval {
  label: string;
  a: number;
  b: number;
}

/**
 * Timestamp queries for profiling. encoder() returns a ProfEncoder: a command encoder the Engine can encode into as
 * usual, whose compute/render passes carry timestampWrites and whose big copies/clears are bracketed by two empty
 * timestamped passes (WebGPU has no timestamps outside passes). Without timestamp-query (`timestamps` false) the encoders
 * are plain and take() is empty: callers fall back to the wall-clock numbers.
 */
export class Profiler {
  readonly qs: GPUQuerySet | null;
  /** @internal (ProfEncoder.finish) */
  readonly resolveBuf: GPUBuffer | null = null;
  /** @internal */
  readonly readBuf: GPUBuffer | null = null;
  private used = 0; // blocks handed out since the last flush
  private pending: Interval[] = [];
  private totals = new Map<string, number>(); // label -> ns
  private mult1000 = true;
  private mult100000 = true;

  constructor(readonly device: GPUDevice, readonly timestamps: boolean) {
    if (!timestamps) {
      this.qs = null;
      return;
    }
    this.qs = device.createQuerySet({ type: 'timestamp', count: BLOCK * BLOCKS });
    this.resolveBuf = device.createBuffer({ size: BLOCK * BLOCKS * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: BLOCK * BLOCKS * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  /** True when flush() should run before encoding much more. */
  get nearFull(): boolean {
    return this.used >= BLOCKS - 4;
  }

  /** Timestamp granularity seen so far: 'ns' (any value), 'us' (all multiples of 1 us) or '100us' (quantised like a browser). */
  get granularity(): string {
    return this.mult100000 ? '100us' : this.mult1000 ? 'us' : 'ns';
  }

  allocBlock(): number {
    if (this.used >= BLOCKS) throw new Error('timestamp query set exhausted: call Profiler.flush() between submits');
    return this.used++;
  }

  /** Command encoder with timestamps. `prefix` is put before every label of the passes it encodes. */
  encoder(prefix = ''): ProfEncoder {
    return new ProfEncoder(this, prefix);
  }

  /** @internal */
  submitted(intervals: Interval[]): void {
    this.pending.push(...intervals);
  }

  /** Reads the timestamps of everything submitted (and finished) so far into the running totals. */
  async flush(): Promise<void> {
    if (!this.timestamps || this.used === 0) return;
    const rb = this.readBuf!;
    await rb.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rb.getMappedRange().slice(0));
    rb.unmap();
    for (const v of t.subarray(0, this.used * BLOCK)) {
      if (v === 0n) continue; // unwritten slot
      if (v % 1000n !== 0n) this.mult1000 = false;
      if (v % 100000n !== 0n) this.mult100000 = false;
    }
    for (const { label, a, b } of this.pending) {
      const ns = Number(t[b] - t[a]);
      this.totals.set(label, (this.totals.get(label) ?? 0) + Math.max(0, ns));
    }
    this.pending = [];
    this.used = 0;
  }

  /** Flushes and returns the milliseconds per label since the last take(). */
  async take(): Promise<Map<string, number>> {
    await this.flush();
    const out = new Map([...this.totals].map(([k, ns]) => [k, ns / 1e6]));
    this.totals = new Map();
    return out;
  }

  destroy(): void {
    this.qs?.destroy();
    this.resolveBuf?.destroy();
    this.readBuf?.destroy();
  }
}

/** Compute pipeline + bind group a ProfEncoder substitutes for whatever the Engine sets in its dispatch pass. */
export interface Substitute {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

export class ProfEncoder {
  /** Hand this to Engine.encodeCompute / encodeDisplay / loadHistogram and to copyTextureToBuffer. */
  readonly enc: GPUCommandEncoder;
  /** Label of the next compute passes: 'solve' (Engine.encodeCompute), 'reduce' (inside encodeDisplay, see display()). */
  computeLabel = 'solve';
  /** Replaces the pipeline and bind group of every compute pass opened through `enc` (ceiling variants). */
  sub: Substitute | null = null;
  private readonly real: GPUCommandEncoder;
  private readonly block: number;
  private n = 0;
  private readonly intervals: Interval[] = [];

  constructor(private readonly prof: Profiler, private readonly prefix: string) {
    this.real = prof.device.createCommandEncoder();
    this.block = prof.timestamps ? prof.allocBlock() : -1;
    const self = this;
    this.enc = new Proxy(this.real, {
      get(target, prop) {
        switch (prop) {
          case 'beginComputePass':
            return (d?: GPUComputePassDescriptor) => self.computePass(d);
          case 'beginRenderPass':
            return (d: GPURenderPassDescriptor) => self.renderPass(d);
          case 'clearBuffer':
            return (buffer: GPUBuffer, offset?: number, size?: number) => self.marked('clear', size ?? buffer.size, () => target.clearBuffer(buffer, offset, size));
          case 'copyBufferToBuffer':
            return (src: GPUBuffer, so: number, dst: GPUBuffer, dOff: number, size: number) =>
              self.marked('loadhist', size, () => target.copyBufferToBuffer(src, so, dst, dOff, size));
          case 'copyTextureToBuffer':
            return (src: GPUTexelCopyTextureInfo, dst: GPUTexelCopyBufferInfo, size: GPUExtent3DStrict) =>
              self.marked('copy', BIG, () => target.copyTextureToBuffer(src, dst, size));
          case 'finish':
            return (d?: GPUCommandBufferDescriptor) => self.finish(d);
        }
        const v = Reflect.get(target, prop);
        return typeof v === 'function' ? v.bind(target) : v;
      },
    }) as unknown as GPUCommandEncoder;
  }

  /** Engine.encodeDisplay with its compute pass labelled 'reduce' (the render pass is always 'tonemap'). */
  display(engine: Engine, view: GPUTextureView, params: Parameters<Engine['encodeDisplay']>[2]): void {
    this.computeLabel = 'reduce';
    try {
      engine.encodeDisplay(this.enc, view, params);
    } finally {
      this.computeLabel = 'solve';
    }
  }

  /** Closes the encoder (appending the timestamp resolve) and returns its command buffer. */
  finish(d?: GPUCommandBufferDescriptor): GPUCommandBuffer {
    if (this.block >= 0 && this.n > 0) {
      const first = this.block * BLOCK;
      const { qs, resolveBuf, readBuf } = this.prof;
      this.real.resolveQuerySet(qs!, first, this.n, resolveBuf!, this.block * 256);
      this.real.copyBufferToBuffer(resolveBuf!, this.block * 256, readBuf!, this.block * 256, this.n * 8);
      this.prof.submitted(this.intervals);
    }
    return this.real.finish(d);
  }

  private alloc(): number {
    if (this.n >= BLOCK) throw new Error('too many timestamps in one command encoder');
    return this.block * BLOCK + this.n++;
  }

  private stamps(): GPUComputePassTimestampWrites | undefined {
    if (this.block < 0) return undefined;
    return { querySet: this.prof.qs!, beginningOfPassWriteIndex: this.alloc(), endOfPassWriteIndex: this.alloc() };
  }

  private computePass(d?: GPUComputePassDescriptor): GPUComputePassEncoder {
    const tw = this.stamps();
    const pass = this.real.beginComputePass({ ...d, timestampWrites: tw });
    if (tw) this.intervals.push({ label: this.prefix + this.computeLabel, a: tw.beginningOfPassWriteIndex!, b: tw.endOfPassWriteIndex! });
    const sub = this.sub;
    if (!sub) return pass;
    return new Proxy(pass, {
      get(target, prop) {
        if (prop === 'setPipeline') return () => target.setPipeline(sub.pipeline);
        if (prop === 'setBindGroup') return (index: number) => target.setBindGroup(index, sub.bindGroup);
        const v = Reflect.get(target, prop);
        return typeof v === 'function' ? v.bind(target) : v;
      },
    });
  }

  private renderPass(d: GPURenderPassDescriptor): GPURenderPassEncoder {
    const tw = this.stamps();
    if (tw) this.intervals.push({ label: this.prefix + 'tonemap', a: tw.beginningOfPassWriteIndex!, b: tw.endOfPassWriteIndex! });
    return this.real.beginRenderPass({ ...d, timestampWrites: tw });
  }

  /** Times a copy/clear between two empty timestamped passes: from the end of the first to the beginning of the second. */
  private marked(label: string, bytes: number, op: () => void): void {
    if (this.block < 0 || bytes < BIG) {
      op();
      return;
    }
    const before = this.marker();
    op();
    const after = this.marker();
    this.intervals.push({ label: this.prefix + label, a: before + 1, b: after });
  }

  private marker(): number {
    const tw = this.stamps()!;
    this.real.beginComputePass({ timestampWrites: tw }).end();
    return tw.beginningOfPassWriteIndex!;
  }
}

/**
 * src/gpu/engine.ts RgbaReadback with its phases separated and timed: the texture copy (submitted and waited for), the
 * mapAsync, and the CPU loop that strips the row padding. Same texture, copy and strip as the shipped class (the test
 * asserts identical output for one small case).
 */
export class TimedReadback {
  readonly view: GPUTextureView;
  private readonly texture: GPUTexture;
  private readonly buffer: GPUBuffer;
  private readonly bytesPerRow: number;

  constructor(private readonly device: GPUDevice, readonly width: number, readonly height: number) {
    this.texture = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.view = this.texture.createView();
    this.bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    this.buffer = device.createBuffer({ size: this.bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  async read(prof: Profiler): Promise<{ rgba: Uint8ClampedArray<ArrayBuffer>; copyMs: number; mapMs: number; stripMs: number }> {
    const { width, height, bytesPerRow } = this;
    const px = prof.encoder();
    px.enc.copyTextureToBuffer({ texture: this.texture }, { buffer: this.buffer, bytesPerRow }, [width, height]);
    let t = now();
    this.device.queue.submit([px.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const copyMs = now() - t;
    t = now();
    await this.buffer.mapAsync(GPUMapMode.READ);
    const mapMs = now() - t;
    t = now();
    const src = new Uint8Array(this.buffer.getMappedRange());
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) rgba.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    this.buffer.unmap();
    return { rgba, copyMs, mapMs, stripMs: now() - t };
  }

  destroy(): void {
    this.buffer.destroy();
    this.texture.destroy();
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Ceiling variants (test-only WGSL, built by string-patching / extending the shipped sources)
// ---------------------------------------------------------------------------------------------------------------------

/** 'solveOnly': (a) roots computed, never deposited. 'capture': the real roots of every solve written to a buffer (untimed
 * set-up of (b)). 'real' / 'uniform' / 'conc': (b) splat only, roots from the capture / uniform over the view (c) / on the
 * circle |z| = 1 (c). */
export type Variant = 'solveOnly' | 'capture' | 'real' | 'uniform' | 'conc';
/** still: Engine.setParams path (splat / splatCoop); frames: setFrames path (splatFrames / splatFramesCoop). */
export type Layout = 'still' | 'frames';

/** Replaces `from` (which must occur exactly once in the shipped source, so the patch cannot silently miss) with `to`. */
export function patch(src: string, from: string, to: string): string {
  const i = src.indexOf(from);
  if (i < 0 || src.indexOf(from, i + 1) >= 0) throw new Error(`profile: ${JSON.stringify(from.slice(0, 70))} is not in the shipped WGSL exactly once`);
  return src.slice(0, i) + to + src.slice(i + from.length);
}

const lf = (s: string) => s.replace(/\r\n/g, '\n');

// (a): sum the roots into one private register; write it under a condition that is false at run time (P._pad2 is 0 and no
// root sum has the bit pattern 0xdeadbeef = -6e18) but unknowable to the compiler, so the solve cannot be eliminated and
// `hist` stays in the layout.
const GUARD = `
var<private> accRoots: vec2f;
fn guardRoots() {
  if (bitcast<u32>(accRoots.x) == P._pad2 + 0xdeadbeefu) { atomicStore(&hist[0], bitcast<u32>(accRoots.y)); }
}
`;

const COMMON = `
fn pack3(c: vec3u) -> u32 { return c.x | (c.y << 8u) | (c.z << 16u); }
fn unpack3(c: u32) -> vec3u { return vec3u(c & 255u, (c >> 8u) & 255u, (c >> 16u) & 255u); }
fn rnd(x: u32) -> f32 {   // PCG hash -> [0, 1)
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}
// MODE 0: the captured root; 1: uniform over the view; 2: on the unit circle (+-0.02), where Littlewood roots pile up.
fn rootAt(ci: u32, j: u32, c: vec2f, ppu: f32) -> vec2f {
  if (MODE == 0u) { return outRoots[ci * P.degree + j]; }
  let u1 = rnd((ci * 32u + j) * 2u);
  let u2 = rnd((ci * 32u + j) * 2u + 1u);
  if (MODE == 1u) { return c + vec2f((u1 - 0.5) * f32(P.width) / ppu, (u2 - 0.5) * f32(P.height) / ppu); }
  let ang = 6.2831853 * u1;
  return (1.0 + 0.04 * (u2 - 0.5)) * vec2f(cos(ang), sin(ang));
}
`;

// per poly (roots.wgsl): capHdr[ci] = (lo | hi << 8 | partner << 16, polyRgb, partnerRgb, 0); roots at outRoots[ci * degree + j]
const PP_CAPTURE_STORE = (ci: string) => `
    capHdr[${ci}] = vec4u(lh.x | (lh.y << 8u) | (select(0u, 1u, partner) << 16u), pack3(polyRgb), pack3(partnerRgb), 0u);
    for (var j = 0u; j < lh.y - lh.x; j++) { outRoots[${ci} * P.degree + j] = z[j]; }
`;
const PP_UNPACK = (ci: string) => `
    let h = capHdr[${ci}];
    let lh = vec2u(h.x & 255u, (h.x >> 8u) & 255u);
    partner = ((h.x >> 16u) & 1u) != 0u;
    polyRgb = unpack3(h.y);
    partnerRgb = unpack3(h.z);
`;
const PP_SPLAT_ONLY = `
@compute @workgroup_size(WG)
fn splatOnly(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x + gid.y * P.strideX;
  if (tid >= P.count) { return; }
  ${PP_UNPACK('tid')}
  for (var j = 0u; j < lh.y - lh.x; j++) { z[j] = rootAt(tid, j, P.center, P.pxPerUnit); }
  depositRoots(lh, 0u, P.center, P.pxPerUnit);
}
@compute @workgroup_size(WG)
fn splatOnlyFrames(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x + gid.y * P.strideX;
  if (tid >= P.count) { return; }
  for (var f = 0u; f < P.frames; f++) {
    ${PP_UNPACK('(tid * P.frames + f)')}
    let c = coeffs[FRAME_VIEWS + 2u * f];
    let ppu = coeffs[FRAME_VIEWS + 2u * f + 1u].x;
    for (var j = 0u; j < lh.y - lh.x; j++) { z[j] = rootAt(tid * P.frames + f, j, c, ppu); }
    depositRoots(lh, f * P.width * P.height * 4u, c, ppu);
  }
}
`;
// cooperative (coop.wgsl): capHdr[ci] = sinfo (lo, hi, packed rgb, packed partner rgb | partner << 24)
const COOP_SPLAT_ONLY = `
@compute @workgroup_size(WGS)
fn splatOnlyCoop(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3u) {
  let d = P.degree;
  let ppw = WGS / d;
  let slot = li / d;
  let j = li % d;
  let tid = (wid.x + wid.y * nwg.x) * ppw + slot;
  let valid = slot < ppw && tid < P.count;
  var info = vec4u(0u);
  if (valid) { info = capHdr[tid]; }
  let live = valid && j < info.y - info.x;
  var z = vec2f(0.0);
  if (live) { z = rootAt(tid, j, P.center, P.pxPerUnit); }
  depositRoots(j, valid, live, info.y, z, info, 0u, P.center, P.pxPerUnit);
}
@compute @workgroup_size(WGS)
fn splatOnlyFramesCoop(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3u) {
  let d = P.degree;
  let ppw = WGS / d;
  let slot = li / d;
  let j = li % d;
  let tid = (wid.x + wid.y * nwg.x) * ppw + slot;
  let valid = slot < ppw && tid < P.count;
  for (var f = 0u; f < P.frames; f++) {
    var info = vec4u(0u);
    if (valid) { info = capHdr[tid * P.frames + f]; }
    let live = valid && j < info.y - info.x;
    let c = coeffs[FRAME_VIEWS + 2u * f];
    let ppu = coeffs[FRAME_VIEWS + 2u * f + 1u].x;
    var z = vec2f(0.0);
    if (live) { z = rootAt(tid * P.frames + f, j, c, ppu); }
    depositRoots(j, valid, live, info.y, z, info, f * P.width * P.height * 4u, c, ppu);
  }
}
`;

const CAP_BINDINGS = `
@group(0) @binding(6) var<storage, read_write> capHdr: array<vec4u>;
`;
const COOP_ROOTS_BINDING = `
@group(0) @binding(5) var<storage, read_write> outRoots: array<vec2f>;
`;

const MODES: Record<'real' | 'uniform' | 'conc', number> = { real: 0, uniform: 1, conc: 2 };

/** WGSL source and entry point of a variant. Shipped entry points keep their name (their module is just patched). */
export function variantSource(kernel: Kernel, layout: Layout, v: Variant): { code: string; entry: string } {
  const coop = kernel === 'coop';
  const src = lf(coop ? coopSrc : rootsSrc);
  const entry = coop ? (layout === 'still' ? 'splatCoop' : 'splatFramesCoop') : layout === 'still' ? 'splat' : 'splatFrames';
  if (v === 'solveOnly') {
    let code = src + GUARD;
    if (!coop && layout === 'still') {
      code = patch(code, '  depositRoots(lh, 0u, P.center, P.pxPerUnit);\n', '  for (var j = 0u; j < lh.y - lh.x; j++) { accRoots += z[j]; }\n  guardRoots();\n');
    } else if (!coop) {
      code = patch(code, '    depositRoots(lh, slice, c, ppu);\n  }\n}', '    for (var j = 0u; j < lh.y - lh.x; j++) { accRoots += z[j]; }\n  }\n  guardRoots();\n}');
    } else if (layout === 'still') {
      code = patch(code, '  depositRoots(j, valid, live, hi, z, info, 0u, P.center, P.pxPerUnit);\n', '  if (live) { accRoots += z; }\n  guardRoots();\n');
    } else {
      code = patch(code, '    depositRoots(j, valid, live, hi, z, info, f * P.width * P.height * 4u, c, ppu);\n  }\n}', '    if (live) { accRoots += z; }\n  }\n  guardRoots();\n}');
    }
    return { code, entry };
  }
  if (v === 'capture') {
    // stats[1] counts Aberth sweeps (per thread; per workgroup in coop.wgsl) of the whole capture run.
    let code = coop
      ? patch(src, 'for (var it = 0u; it < P.maxIter; it++) {', 'for (var it = 0u; it < P.maxIter; it++) {\n    if (li == 0u) { atomicAdd(&stats[1], 1u); }') + COOP_ROOTS_BINDING
      : patch(src, 'for (var it = 0u; it < P.maxIter; it++) {', 'for (var it = 0u; it < P.maxIter; it++) {\n    atomicAdd(&stats[1], 1u);');
    code += COMMON.replace(/^/, 'const MODE: u32 = 0u;\n') + CAP_BINDINGS;
    if (!coop && layout === 'still') code = patch(code, '  depositRoots(lh, 0u, P.center, P.pxPerUnit);\n', PP_CAPTURE_STORE('tid'));
    else if (!coop) code = patch(code, '    depositRoots(lh, slice, c, ppu);\n', PP_CAPTURE_STORE('(tid * P.frames + f)'));
    else if (layout === 'still') {
      code = patch(code, '  depositRoots(j, valid, live, hi, z, info, 0u, P.center, P.pxPerUnit);\n', '  if (valid && j == 0u) { capHdr[tid] = info; }\n  if (live) { outRoots[tid * d + j] = z; }\n');
    } else {
      code = patch(
        code,
        '    depositRoots(j, valid, live, hi, z, info, f * P.width * P.height * 4u, c, ppu);\n',
        '    if (valid && j == 0u) { capHdr[tid * P.frames + f] = vec4u(lo, hi, info.z, info.w); }\n    if (live) { outRoots[(tid * P.frames + f) * d + j] = z; }\n',
      );
    }
    return { code, entry };
  }
  const code = src + (coop ? COOP_ROOTS_BINDING : '') + `const MODE: u32 = ${MODES[v]}u;\n` + COMMON + CAP_BINDINGS + (coop ? COOP_SPLAT_ONLY : PP_SPLAT_ONLY);
  return { code, entry: coop ? (layout === 'still' ? 'splatOnlyCoop' : 'splatOnlyFramesCoop') : layout === 'still' ? 'splatOnly' : 'splatOnlyFrames' };
}

/** One explicit layout for every variant (layout:'auto' would drop the bindings a variant happens not to read). */
export function variantLayout(device: GPUDevice): GPUBindGroupLayout {
  const C = GPUShaderStage.COMPUTE;
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: C, buffer: { type: 'storage' } },
      { binding: 4, visibility: C, buffer: { type: 'storage' } },
      { binding: 5, visibility: C, buffer: { type: 'storage' } },
      { binding: 6, visibility: C, buffer: { type: 'storage' } },
    ],
  });
}

export function variantPipeline(device: GPUDevice, layout: GPUBindGroupLayout, kernel: Kernel, frames: Layout, v: Variant): GPUComputePipeline {
  const { code, entry } = variantSource(kernel, frames, v);
  return device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({ label: `profile-${kernel}-${frames}-${v}`, code }), entryPoint: entry },
  });
}

/** The Engine's private buffers (TypeScript `private` is compile-time only): what a variant's bind group needs. */
export function engineBuffers(engine: Engine): { params: GPUBuffer; coeffs: GPUBuffer; en: GPUBuffer; hist: GPUBuffer; stats: GPUBuffer } {
  const e = engine as unknown as { paramsBuf: GPUBuffer; coeffBuf: GPUBuffer; enumBuf: GPUBuffer; hist: GPUBuffer; statsBuf: GPUBuffer };
  return { params: e.paramsBuf, coeffs: e.coeffBuf, en: e.enumBuf, hist: e.hist, stats: e.statsBuf };
}
