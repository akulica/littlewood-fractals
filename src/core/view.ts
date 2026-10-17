import type { Complex } from './complex';

/** cx, cy: complex value at the canvas centre. pxPerUnit: device pixels per unit. */
export interface View {
  cx: number;
  cy: number;
  pxPerUnit: number;
}

export function worldToPixel(v: View, w: number, h: number, re: number, im: number): { x: number; y: number } {
  return { x: (re - v.cx) * v.pxPerUnit + w / 2, y: h / 2 - (im - v.cy) * v.pxPerUnit };
}

export function pixelToWorld(v: View, w: number, h: number, px: number, py: number): Complex {
  return { re: v.cx + (px - w / 2) / v.pxPerUnit, im: v.cy - (py - h / 2) / v.pxPerUnit };
}

/** Scales by `factor` while keeping the world point under pixel (px, py) fixed. */
export function zoomAt(v: View, w: number, h: number, px: number, py: number, factor: number): View {
  const p = pixelToWorld(v, w, h, px, py);
  const s = v.pxPerUnit * factor;
  return { pxPerUnit: s, cx: p.re - (px - w / 2) / s, cy: p.im + (py - h / 2) / s };
}

/** Pointer moved by (dx, dy) device pixels: content follows it. */
export function panBy(v: View, dx: number, dy: number): View {
  return { ...v, cx: v.cx - dx / v.pxPerUnit, cy: v.cy + dy / v.pxPerUnit };
}

export function fitView(w: number, h: number, radius: number): View {
  return { cx: 0, cy: 0, pxPerUnit: Math.min(w, h) / (2 * radius) };
}
