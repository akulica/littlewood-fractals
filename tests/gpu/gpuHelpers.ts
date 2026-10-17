import { create, globals } from 'webgpu';
import type { Engine } from '../../src/gpu/engine';

Object.assign(globalThis, globals); // GPUBufferUsage, GPUMapMode, ... as in a browser

// Keep the GPU instance referenced for the whole process. If it is garbage-collected, dawn.node tears the
// instance down under the live device and the process dies with 0xC0000005 (found while validating this plan:
// it only shows up once enough JS allocation triggers a GC between GPU calls).
const gpu = create([]);

/** Real GPU device through Dawn. `errors` collects uncaptured validation errors; tests assert it stays empty. */
export async function nodeDevice(): Promise<{ device: GPUDevice; errors: string[]; adapterName: string }> {
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
  return { device, errors, adapterName };
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

/** Total root count in a histogram read back with Engine.readHistogram (4 u32 per pixel, count first). */
export function countSum(h: Uint32Array): number {
  let s = 0;
  for (let i = 0; i < h.length; i += 4) s += h[i];
  return s;
}
