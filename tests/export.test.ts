import { describe, expect, it } from 'vitest';
import { MAX_EXPORT_PIXELS, exportSizeError, exportView } from '../src/core/export';

const limits = { maxTextureDimension2D: 16384, maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 };

describe('exportView', () => {
  it('keeps the centre and shows at least everything that is on screen', () => {
    const v = { cx: 0.3, cy: -0.2, pxPerUnit: 100 };
    expect(exportView(v, 1000, 500, 2000, 1000)).toEqual({ cx: 0.3, cy: -0.2, pxPerUnit: 200 }); // same aspect: pure 2x
    expect(exportView(v, 1000, 500, 1000, 1000).pxPerUnit).toBe(100); // taller: width limits, extra area above/below
    expect(exportView(v, 1000, 500, 4000, 1000).pxPerUnit).toBe(200); // wider: height limits
  });
});

describe('exportSizeError', () => {
  it('accepts sizes within the GPU limits', () => {
    expect(exportSizeError(3840, 2160, limits)).toBeNull();
    expect(exportSizeError(8192, 8192, limits)).toBeNull();
  });
  it('rejects non-integers and non-positive sizes', () => {
    expect(exportSizeError(0, 100, limits)).toMatch(/whole numbers/);
    expect(exportSizeError(100.5, 100, limits)).toMatch(/whole numbers/);
  });
  it('rejects a side above maxTextureDimension2D', () => {
    expect(exportSizeError(20000, 100, limits)).toMatch(/16384 px per side/);
  });
  it('rejects a histogram larger than maxStorageBufferBindingSize', () => {
    expect(exportSizeError(4096, 4096, { ...limits, maxStorageBufferBindingSize: 128 * 2 ** 20 })).toMatch(/256 MB histogram.*128 MB/);
  });
  it('caps total pixels to keep browser memory sane', () => {
    expect(MAX_EXPORT_PIXELS).toBe(8192 * 8192);
    expect(exportSizeError(8192, 8193, limits)).toMatch(/67,108,864 pixels/);
  });
});
