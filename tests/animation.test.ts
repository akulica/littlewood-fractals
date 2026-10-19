import { describe, expect, it } from 'vitest';
import {
  MAX_BATCH_FRAMES,
  batchFrames,
  frameTime,
  framesFor,
  interpolateAt,
  keyframeTime,
  pinnable,
  videoSizeError,
  type Keyframe,
} from '../src/core/animation';

const c = (re: number, im = 0) => ({ re, im });
const A: Keyframe = [c(1), c(-1)];
const B: Keyframe = [c(1), c(0, 1)];
const C: Keyframe = [c(1), c(1, 1)];
const linear = { easing: 'linear', loop: false } as const;

describe('interpolateAt', () => {
  it('hits the keyframes exactly and interpolates linearly between them, equal time per segment', () => {
    expect(interpolateAt([A, B, C], 0, linear)).toEqual(A);
    expect(interpolateAt([A, B, C], 0.5, linear)).toEqual(B);
    expect(interpolateAt([A, B, C], 1, linear)).toEqual(C);
    const q = interpolateAt([A, B, C], 0.25, linear); // halfway through the first segment
    expect(q[0]).toEqual(c(1));
    expect(q[1].re).toBeCloseTo(-0.5, 12);
    expect(q[1].im).toBeCloseTo(0.5, 12);
  });
  it('clamps t to [0, 1]', () => {
    expect(interpolateAt([A, B], -1, linear)).toEqual(A);
    expect(interpolateAt([A, B], 2, linear)).toEqual(B);
  });
  it('smooth easing applies smoothstep to global time (same endpoints, slower start)', () => {
    const smooth = { easing: 'smooth', loop: false } as const;
    expect(interpolateAt([A, B], 0.5, smooth)[1].re).toBeCloseTo(-0.5, 12); // smoothstep(0.5) = 0.5
    expect(interpolateAt([A, B], 0.1, smooth)[1].re).toBeCloseTo(-1 + 0.028, 12); // smoothstep(0.1) = 0.028
    expect(interpolateAt([A, B], 1, smooth)).toEqual(B);
  });
  it('loop appends the first keyframe: the path returns to it at t = 1', () => {
    const loop = { easing: 'linear', loop: true } as const;
    expect(interpolateAt([A, B], 0.5, loop)).toEqual(B);
    expect(interpolateAt([A, B], 1, loop)).toEqual(A);
  });
  it('needs at least two keyframes of equal length', () => {
    expect(() => interpolateAt([A], 0, linear)).toThrow(/two keyframes/);
    expect(() => interpolateAt([A, [c(1)]], 0, linear)).toThrow(/same number/);
  });
});

describe('keyframeTime', () => {
  it('is where interpolateAt reaches each keyframe, for both easings and with loop', () => {
    for (const easing of ['linear', 'smooth'] as const) {
      for (const loop of [false, true]) {
        const keys = [A, B, C];
        keys.forEach((key, i) => {
          const at = interpolateAt(keys, keyframeTime(i, 3, { easing, loop }), { easing, loop });
          at.forEach((z, j) => {
            expect(z.re).toBeCloseTo(key[j].re, 9);
            expect(z.im).toBeCloseTo(key[j].im, 9);
          });
        });
      }
    }
    expect(keyframeTime(1, 3, linear)).toBe(0.5);
  });
});

describe('framesFor', () => {
  it('without loop: first frame = first keyframe, last frame = last keyframe', () => {
    const f = framesFor([A, B, C], 5, linear);
    expect(f).toHaveLength(5);
    expect(f[0]).toEqual(A);
    expect(f[2]).toEqual(B);
    expect(f[4]).toEqual(C);
  });
  it('with loop: the duplicate of frame 0 is omitted so the video loops seamlessly', () => {
    const f = framesFor([A, B], 4, { easing: 'linear', loop: true });
    expect(f).toHaveLength(4);
    expect(f[0]).toEqual(A);
    expect(f[2]).toEqual(B); // t = 2/4: end of the first segment
    expect(f[3][1].re).toBeCloseTo(-0.5, 12); // halfway back; the next frame would be A again
  });
  it('matches interpolateAt at frameTime (the UI timeline uses the same math)', () => {
    const opts = { easing: 'smooth', loop: true } as const;
    const f = framesFor([A, B, C], 7, opts);
    f.forEach((frame, i) => expect(frame).toEqual(interpolateAt([A, B, C], frameTime(i, 7, true), opts)));
  });
  it('a single frame is the first keyframe', () => {
    expect(framesFor([A, B], 1, linear)).toEqual([A]);
  });
  it('rejects a frame count that is not a positive integer', () => {
    expect(() => framesFor([A, B], 0, linear)).toThrow(RangeError);
    expect(() => framesFor([A, B], 2.5, linear)).toThrow(RangeError);
  });
});

describe('pinnable', () => {
  it('is true only when every frame can pin its leading coefficient', () => {
    expect(pinnable([A, [c(2), c(-2)], [c(0, 1), c(0, -1)]])).toBe(true);
    expect(pinnable([A, B])).toBe(false); // {1, i} is not c0 times a group
    expect(pinnable([])).toBe(false);
  });
});

describe('batchFrames', () => {
  const big = { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 };
  it('is 8 at 1080p and 3 at 4K (512 MiB cap)', () => {
    expect(batchFrames(1920 * 1080 * 16, big)).toBe(MAX_BATCH_FRAMES);
    expect(MAX_BATCH_FRAMES).toBe(8);
    expect(batchFrames(3840 * 2160 * 16, big)).toBe(4); // 4 * 127 MiB = 506 MiB
    expect(batchFrames(4096 * 2160 * 16, big)).toBe(3);
  });
  it('respects the device limits and never drops below one frame', () => {
    expect(batchFrames(1920 * 1080 * 16, { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 2 ** 31 })).toBe(4);
    expect(batchFrames(1920 * 1080 * 16, { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 64 * 2 ** 20 })).toBe(2);
    expect(batchFrames(2 ** 30, big)).toBe(1);
  });
});

describe('videoSizeError', () => {
  it('requires an even width and height (H.264 4:2:0)', () => {
    expect(videoSizeError(1920, 1080)).toBeNull();
    expect(videoSizeError(1921, 1080)).toMatch(/even/);
    expect(videoSizeError(640, 361)).toMatch(/even/);
  });
});
