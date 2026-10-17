import { describe, expect, it } from 'vitest';
import { fitView, panBy, pixelToWorld, worldToPixel, zoomAt } from '../src/core/view';

const v = { cx: 0.3, cy: -0.2, pxPerUnit: 100 };

describe('view', () => {
  it('round-trips pixel -> world -> pixel', () => {
    const w = pixelToWorld(v, 800, 600, 123, 456);
    const p = worldToPixel(v, 800, 600, w.re, w.im);
    expect(p.x).toBeCloseTo(123, 9);
    expect(p.y).toBeCloseTo(456, 9);
  });
  it('puts the centre of the view at the canvas centre, imaginary axis up', () => {
    expect(worldToPixel(v, 800, 600, 0.3, -0.2)).toEqual({ x: 400, y: 300 });
    expect(worldToPixel(v, 800, 600, 0.3, 0.8).y).toBe(200);
  });
  it('zoomAt keeps the point under the cursor fixed', () => {
    const before = pixelToWorld(v, 800, 600, 123, 456);
    const z = zoomAt(v, 800, 600, 123, 456, 2.5);
    const after = pixelToWorld(z, 800, 600, 123, 456);
    expect(z.pxPerUnit).toBe(250);
    expect(after.re).toBeCloseTo(before.re, 12);
    expect(after.im).toBeCloseTo(before.im, 12);
  });
  it('panBy moves content with the pointer', () => {
    const p = panBy(v, 100, 50);
    expect(p.cx).toBeCloseTo(-0.7, 12);
    expect(p.cy).toBeCloseTo(0.3, 12);
  });
  it('fitView shows radius r in the shorter dimension', () => {
    expect(fitView(800, 600, 2)).toEqual({ cx: 0, cy: 0, pxPerUnit: 150 });
  });
});
