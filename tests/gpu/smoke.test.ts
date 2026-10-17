import { afterAll, expect, it } from 'vitest';
import { nodeDevice } from './gpuHelpers';

const { device, errors } = await nodeDevice();
afterAll(() => device.destroy());

it('runs a compute shader on the real GPU', async () => {
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> h: array<atomic<u32>>;
           @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { atomicAdd(&h[g.x % 4u], 1u); }`,
  });
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] }));
  pass.dispatchWorkgroups(1024);
  pass.end();
  enc.copyBufferToBuffer(buf, 0, rb, 0, 16);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  expect([...new Uint32Array(rb.getMappedRange())]).toEqual([16384, 16384, 16384, 16384]);
  expect(errors).toEqual([]);
});
