/** Browser WebGPU device with the largest storage-buffer limits the adapter allows (big histograms). */
export async function getDevice(): Promise<GPUDevice> {
  if (!('gpu' in navigator)) {
    throw new Error(
      'WebGPU is not available. Use Chrome/Edge 113+ (Windows, macOS, ChromeOS, Android 12+, Linux on recent Intel/NVIDIA), ' +
        'Safari 26+, or Firefox 141+ on Windows. The page must be served over https:// or from localhost.',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter: the GPU may be blocklisted or hardware acceleration is off.');
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
}
