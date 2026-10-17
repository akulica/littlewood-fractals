import type { View } from './view';

/** 8192 x 8192: RGBA readback 256 MB + RGB scanlines 192 MB in browser memory — the sane upper bound. */
export const MAX_EXPORT_PIXELS = 8192 * 8192;

export interface ExportLimits {
  maxTextureDimension2D: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

/** View for a width x height export that keeps the centre and contains everything currently on screen. */
export function exportView(v: View, screenW: number, screenH: number, width: number, height: number): View {
  return { cx: v.cx, cy: v.cy, pxPerUnit: v.pxPerUnit * Math.min(width / screenW, height / screenH) };
}

const mb = (bytes: number) => `${Math.round(bytes / 2 ** 20).toLocaleString('en-US')} MB`;

/** Human-readable reason the size cannot be exported on this device, or null if it can. */
export function exportSizeError(width: number, height: number, limits: ExportLimits): string | null {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    return 'width and height must be whole numbers ≥ 1';
  }
  if (width > limits.maxTextureDimension2D || height > limits.maxTextureDimension2D) {
    return `this GPU allows at most ${limits.maxTextureDimension2D} px per side`;
  }
  if (width * height > MAX_EXPORT_PIXELS) {
    return `at most ${MAX_EXPORT_PIXELS.toLocaleString('en-US')} pixels (e.g. 8192×8192)`;
  }
  const hist = width * height * 16;
  if (hist > limits.maxStorageBufferBindingSize) {
    return `${width}×${height} needs a ${mb(hist)} histogram; this GPU allows ${mb(limits.maxStorageBufferBindingSize)}`;
  }
  if (Math.ceil((width * 4) / 256) * 256 * height > limits.maxBufferSize) {
    return `${width}×${height} exceeds this GPU's maximum buffer size`;
  }
  return null;
}
