import { create, globals } from 'webgpu';
import type { GpuInfo } from '../../src/core/metadata';
import type { Engine } from '../../src/gpu/engine';

Object.assign(globalThis, globals); // GPUBufferUsage, GPUMapMode, ... as in a browser

// Keep the GPU instance referenced for the whole process. If it is garbage-collected, dawn.node tears the
// instance down under the live device and the process dies with 0xC0000005 (found while validating this plan:
// it only shows up once enough JS allocation triggers a GC between GPU calls).
const gpu = create([]);

/** Real GPU device through Dawn. `errors` collects uncaptured validation errors; tests assert it stays empty.
 * `gpu` is the adapter as export metadata records it (core/metadata GpuInfo). */
export async function nodeDevice(): Promise<{ device: GPUDevice; errors: string[]; adapterName: string; gpu: GpuInfo }> {
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('dawn.node found no GPU adapter');
  const adapterName = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`.trim();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  const { vendor, architecture, device: name, description } = adapter.info;
  return { device, errors, adapterName, gpu: { vendor, architecture, device: name, description } };
}

/** Submit compute chunks until the engine reports nothing left, then wait for the GPU.
 * waitEachChunk = true waits after every submit (like the interactive loop, one chunk in flight). */
export async function runToEnd(device: GPUDevice, engine: Engine, budget: number, waitEachChunk = false): Promise<void> {
  for (;;) {
    const enc = device.createCommandEncoder();
    const n = engine.encodeCompute(enc, budget);
    device.queue.submit([enc.finish()]);
    if (waitEachChunk) await device.queue.onSubmittedWorkDone();
    if (n === 0) break;
  }
  await device.queue.onSubmittedWorkDone();
}

/** Copy of the first `bytes` of a COPY_SRC buffer. */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const copy = rb.getMappedRange().slice(0);
  rb.unmap();
  rb.destroy();
  return copy;
}

/** Total root count in a histogram read back with Engine.readHistogram (4 u32 per pixel, count first). */
export function countSum(h: Uint32Array): number {
  let s = 0;
  for (let i = 0; i < h.length; i += 4) s += h[i];
  return s;
}

/** Index of the first differing element, or -1 when equal (toEqual is ~1 s per 256x256 histogram). */
export function firstDifference(a: Uint32Array, b: Uint32Array): number {
  if (a.length !== b.length) return Math.min(a.length, b.length);
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i;
  return -1;
}

/** L1 distance between two histograms' densities (each normalised to sum 1), summed over b x b pixel blocks.
 * 0 = identical, 2 = disjoint. */
export function blockL1(p: Uint32Array, q: Uint32Array, w: number, h: number, b: number): number {
  const sp = countSum(p);
  const sq = countSum(q);
  let d = 0;
  for (let by = 0; by < h; by += b) {
    for (let bx = 0; bx < w; bx += b) {
      let x = 0;
      let y = 0;
      for (let yy = by; yy < Math.min(by + b, h); yy++) {
        for (let xx = bx; xx < Math.min(bx + b, w); xx++) {
          x += p[(yy * w + xx) * 4];
          y += q[(yy * w + xx) * 4];
        }
      }
      d += Math.abs(x / sp - y / sq);
    }
  }
  return d;
}
