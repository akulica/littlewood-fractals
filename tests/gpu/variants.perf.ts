import { afterAll, expect, it } from 'vitest';
import rootsSrc from '../../src/gpu/roots.wgsl?raw';
import coopSrc from './coop.wgsl?raw';
import { indexToDigits } from '../../src/core/enumeration';
import { dispatchGrid } from '../../src/core/schedule';
import { nodeDevice } from './gpuHelpers';

// Kernel-variant shoot-out on the current GPU: thread-per-polynomial (shipping) vs workgroup-cooperative.
const { device, errors, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const W = 1024, H = 1024, CHUNK = 1 << 20;
const U = GPUBufferUsage;
const pipe = (code: string, entryPoint: string) =>
  device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint } });
const variants = { perPoly: pipe(rootsSrc, 'splat'), coop: pipe(coopSrc, 'splatCoop') };
const buf = (data: ArrayBuffer | ArrayBufferView<ArrayBuffer>, usage: number) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | U.COPY_DST });
  device.queue.writeBuffer(b, 0, data);
  return b;
};

async function run(kind: keyof typeof variants, degree: number): Promise<{ ms: number; sum: number }> {
  const p = variants[kind];
  const total = 2 ** degree; // Littlewood, leading coefficient pinned
  const hist = device.createBuffer({ size: W * H * 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  const stats = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_DST });
  const coeffs = buf(new Float32Array([1, 0, -1, 0]), U.STORAGE);
  const t0 = performance.now();
  for (let first = 0; first < total; first += CHUNK) {
    const count = Math.min(CHUNK, total - first);
    const perGroup = kind === 'coop' ? Math.floor(256 / degree) : 64;
    const g = dispatchGrid(Math.ceil(count / perGroup) * 64); // reuse the 2-D wrap; x*y >= groups needed
    const params = new ArrayBuffer(48);
    new Uint32Array(params).set([degree, 2, count, g.strideX, W, H, 32, 0]);
    new Float32Array(params).set([0, 0, W / 4, 1e-5], 8);
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf(params, U.UNIFORM) } },
        { binding: 1, resource: { buffer: coeffs } },
        { binding: 2, resource: { buffer: buf(indexToDigits(BigInt(first), 2, degree), U.STORAGE) } },
        { binding: 3, resource: { buffer: hist } },
        { binding: 4, resource: { buffer: stats } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(g.x, g.y);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const ms = performance.now() - t0;
  const rb = device.createBuffer({ size: W * H * 16, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(hist, 0, rb, 0, W * H * 16);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const h = new Uint32Array(rb.getMappedRange());
  let sum = 0;
  for (let i = 0; i < h.length; i += 4) sum += h[i];
  rb.unmap();
  return { ms, sum };
}

it.each([16, 20, 24])('Littlewood d=%i: per-polynomial vs cooperative', async (degree) => {
  const a = await run('perPoly', degree);
  const b = await run('coop', degree);
  console.log(`d=${degree}: perPoly ${a.ms.toFixed(0)} ms, coop ${b.ms.toFixed(0)} ms (coop/perPoly = ${(b.ms / a.ms).toFixed(2)})`);
  // Jacobi vs Gauss-Seidel land a handful of roots on the other side of a pixel edge: allow 1e-6 relative.
  expect(Math.abs(b.sum - a.sum)).toBeLessThanOrEqual(Math.ceil(a.sum * 1e-6));
  expect(errors).toEqual([]);
});
