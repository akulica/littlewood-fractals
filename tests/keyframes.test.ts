import { describe, expect, it } from 'vitest';
import { frameBrightness, frameTime, interpolateView, stateAt, statesFor, type Keyframe, type PathFallback } from '../src/core/animation';
import { exportView } from '../src/core/export';
import { fitRect, viewRect, type ViewRect } from '../src/core/view';

const c = (re: number, im = 0) => ({ re, im });
const rect = (cx: number, cy: number, half: number, aspect = 1.5): ViewRect => ({ cx, cy, halfW: half * aspect, halfH: half });
const zoomOf = (r: ViewRect) => 1 / Math.sqrt(r.halfW * r.halfH);
const linear = { easing: 'linear', loop: false } as const;
const fallback: PathFallback = { view: rect(9, 9, 9), exposure: 1.5, gamma: 0.7 };

describe('viewRect / fitRect', () => {
  it('round-trip a view through a canvas of the same size', () => {
    const v = { cx: 0.3, cy: -0.2, pxPerUnit: 250 };
    const r = viewRect(v, 1000, 600);
    expect(r).toEqual({ cx: 0.3, cy: -0.2, halfW: 2, halfH: 1.2 });
    expect(fitRect(r, 1000, 600)).toEqual(v);
  });
  it('fitting into another size contains the rect, exactly like exportView', () => {
    const v = { cx: 0.3, cy: -0.2, pxPerUnit: 100 };
    for (const [w, h] of [[2000, 1000], [1000, 1000], [4000, 1000]]) {
      expect(fitRect(viewRect(v, 1000, 500), w, h).pxPerUnit).toBeCloseTo(exportView(v, 1000, 500, w, h).pxPerUnit, 9);
    }
  });
});

describe('interpolateView', () => {
  const a = rect(0, 0, 2);
  const b = rect(1, 0.5, 0.002); // 1000x zoom-in towards (1, 0.5)

  it('hits both ends exactly', () => {
    expect(interpolateView(a, b, 0)).toEqual(a);
    expect(interpolateView(a, b, 1)).toEqual(b);
  });
  it('zooms geometrically: log(zoom) is linear in f', () => {
    for (const f of [0.1, 0.25, 0.5, 0.9]) {
      const r = interpolateView(a, b, f);
      expect(Math.log(zoomOf(r))).toBeCloseTo(Math.log(zoomOf(a)) * (1 - f) + Math.log(zoomOf(b)) * f, 10);
      expect(r.halfW / r.halfH).toBeCloseTo(1.5, 12);
    }
  });
  it('zooming in: c = c2 + (c1 - c2) (s1 / s) (1 - f), so the target stays on screen and approaches the centre linearly', () => {
    for (const f of [0.1, 0.5, 0.9]) {
      const r = interpolateView(a, b, f);
      const k = (zoomOf(a) / zoomOf(r)) * (1 - f);
      expect(r.cx).toBeCloseTo(b.cx + (a.cx - b.cx) * k, 12);
      expect(r.cy).toBeCloseTo(b.cy + (a.cy - b.cy) * k, 12);
      // target offset in screen units (world offset x zoom) shrinks linearly: (1 - f) of the start offset
      const off0 = Math.hypot(b.cx - a.cx, b.cy - a.cy) * zoomOf(a);
      expect(Math.hypot(b.cx - r.cx, b.cy - r.cy) * zoomOf(r)).toBeCloseTo(off0 * (1 - f), 9);
      expect(Math.abs(b.cx - r.cx)).toBeLessThan(r.halfW); // never off-screen (linear centre would be ~16 widths out)
    }
  });
  it('zooming out is the mirror image, anchored at the start', () => {
    for (const f of [0.1, 0.5, 0.9]) {
      const fwd = interpolateView(a, b, f);
      const back = interpolateView(b, a, 1 - f);
      expect(back.cx).toBeCloseTo(fwd.cx, 12);
      expect(back.cy).toBeCloseTo(fwd.cy, 12);
      expect(back.halfW).toBeCloseTo(fwd.halfW, 12);
    }
  });
  it('equal zoom reduces to linear panning', () => {
    const p = rect(-1, 2, 0.5);
    const q = rect(3, -2, 0.5);
    const r = interpolateView(p, q, 0.25);
    expect(r.cx).toBeCloseTo(0, 12);
    expect(r.cy).toBeCloseTo(1, 12);
    expect(r.halfH).toBeCloseTo(0.5, 12);
  });
});

describe('stateAt', () => {
  const k0: Keyframe = { coeffs: [c(1), c(-1)], view: rect(0, 0, 2), exposure: 1, gamma: 0.6 };
  const k1: Keyframe = { coeffs: [c(1), c(0, 1)] }; // carries neither view nor display
  const k2: Keyframe = { coeffs: [c(1), c(1, 1)], view: rect(1, 0, 0.02), exposure: 4, gamma: 1 };

  it('keyframes are hit exactly; channels a keyframe lacks come from the nearest keyframes that have them', () => {
    const keys = [k0, k1, k2];
    expect(stateAt(keys, 0, linear, fallback)).toEqual({ coeffs: k0.coeffs, view: k0.view, exposure: 1, gamma: 0.6 });
    expect(stateAt(keys, 1, linear, fallback)).toEqual({ coeffs: k2.coeffs, view: k2.view, exposure: 4, gamma: 1 });
    const mid = stateAt(keys, 0.5, linear, fallback); // at k1: halfway between k0 and k2 for view and display
    expect(mid.coeffs).toEqual(k1.coeffs);
    const v = interpolateView(k0.view!, k2.view!, 0.5);
    expect(mid.view.cx).toBeCloseTo(v.cx, 12);
    expect(mid.view.halfH).toBeCloseTo(v.halfH, 12);
    expect(mid.exposure).toBeCloseTo(2, 12); // geometric: sqrt(1 * 4)
    expect(mid.gamma).toBeCloseTo(0.8, 12); // linear
  });
  it('no keyframe with a view / display: the fallback (current screen and sliders) applies', () => {
    const s = stateAt([{ coeffs: [c(1)] }, { coeffs: [c(2)] }], 0.3, linear, fallback);
    expect(s.view).toEqual(fallback.view);
    expect([s.exposure, s.gamma]).toEqual([1.5, 0.7]);
  });
  it('a single carrying keyframe holds its value for the whole path', () => {
    const keys: Keyframe[] = [{ coeffs: [c(1)] }, { coeffs: [c(2)], view: rect(5, 5, 1), exposure: 2, gamma: 1 }, { coeffs: [c(3)] }];
    for (const t of [0, 0.3, 1]) {
      const s = stateAt(keys, t, linear, fallback);
      expect(s.view).toEqual(rect(5, 5, 1));
      expect(s.exposure).toBe(2);
    }
  });
  it('without loop, values are held before the first and after the last carrying keyframe', () => {
    const keys: Keyframe[] = [{ coeffs: [c(1)] }, { coeffs: [c(2)], view: rect(0, 0, 1) }, { coeffs: [c(3)], view: rect(1, 1, 1) }, { coeffs: [c(4)] }];
    expect(stateAt(keys, 0.1, linear, fallback).view).toEqual(rect(0, 0, 1));
    expect(stateAt(keys, 0.95, linear, fallback).view).toEqual(rect(1, 1, 1));
  });
  it('with loop, the view wraps from the last carrying keyframe back to the first (smooth seam)', () => {
    const loop = { easing: 'linear', loop: true } as const;
    const keys: Keyframe[] = [{ coeffs: [c(1)], view: rect(0, 0, 1) }, { coeffs: [c(2)], view: rect(2, 0, 1) }, { coeffs: [c(3)] }];
    // path positions: k0 at 0, k1 at 1, k2 at 2, k0 again at 3. Between k1 (1) and k0 (3) the view pans 2 -> 0.
    expect(stateAt(keys, 2 / 3, loop, fallback).view.cx).toBeCloseTo(1, 12);
    expect(stateAt(keys, 1, loop, fallback).view).toEqual(rect(0, 0, 1));
    expect(stateAt(keys, 0.999, loop, fallback).view.cx).toBeCloseTo(0, 2);
  });
  it('easing applies to every channel: smooth starts slower', () => {
    const smooth = { easing: 'smooth', loop: false } as const;
    const keys = [k0, k2];
    const lin = stateAt(keys, 0.1, linear, fallback);
    const sm = stateAt(keys, 0.1, smooth, fallback);
    expect(sm.view.halfW).toBeGreaterThan(lin.view.halfW); // less zoomed in yet
    expect(sm.exposure).toBeLessThan(lin.exposure);
    expect(sm.exposure).toBeCloseTo(4 ** 0.028, 12); // smoothstep(0.1) = 0.028
  });
  it('statesFor uses the render timeline (frameTime)', () => {
    const keys = [k0, k1, k2];
    const states = statesFor(keys, 5, linear, fallback);
    expect(states).toHaveLength(5);
    states.forEach((s, i) => expect(s).toEqual(stateAt(keys, frameTime(i, 5, false), linear, fallback)));
  });
});

describe('frameBrightness', () => {
  it('perFrame: no fixed max; frame0: frame 0 max at a constant scale', () => {
    expect(frameBrightness('perFrame', 4, 3, 100)).toEqual({ countScale: 4, fixedMax: 0 });
    expect(frameBrightness('frame0', 4, 3, 100)).toEqual({ countScale: 4, fixedMax: 100 });
  });
  it('locked: countScale = base (s_f / s_0)^2 and the reference max x scale stays frame 0\'s', () => {
    expect(frameBrightness('locked', 4, 1, 100)).toEqual({ countScale: 4, fixedMax: 100 }); // frame 0 = frame0 mode
    const z = frameBrightness('locked', 4, 10, 100); // 10x zoomed in: a pixel covers 1/100 of the area
    expect(z.countScale).toBe(400);
    expect(z.fixedMax * z.countScale).toBeCloseTo(100 * 4, 9);
    const out = frameBrightness('locked', 2.25, 0.5, 80);
    expect(out.countScale).toBeCloseTo(2.25 / 4, 12);
    expect(out.fixedMax * out.countScale).toBeCloseTo(80 * 2.25, 9);
  });
});
