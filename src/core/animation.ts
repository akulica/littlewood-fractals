import type { Complex } from './complex';
import { leadingFixable } from './enumeration';

/** One coefficient set; every keyframe of an animation has the same length k. */
export type Keyframe = Complex[];
export type Easing = 'linear' | 'smooth';

export interface PathOptions {
  easing: Easing;
  /** Return to the first keyframe at the end, so the video loops. */
  loop: boolean;
}

/** Frames solved per thread in one splatFrames batch. Measured best on both GPUs (plan §1.2): fewer frames
 * leave warm-start gains unused, more lose histogram cache locality. = the coefficient buffer's frame capacity. */
export const MAX_BATCH_FRAMES = 8;
/** Upper bound for one batch's histograms (all frames of a batch live on the GPU at once). */
export const MAX_BATCH_BYTES = 512 * 2 ** 20;

const smoothstep = (t: number) => t * t * (3 - 2 * t);

function checkKeys(keys: Keyframe[]): void {
  if (keys.length < 2) throw new RangeError('an animation needs at least two keyframes');
  if (keys.some((key) => key.length !== keys[0].length || key.length === 0)) {
    throw new RangeError('every keyframe must have the same number of coefficients');
  }
}

/**
 * Coefficients at timeline position t in [0, 1]: piecewise linear through the keyframes (plus keys[0] again when
 * looping), equal time per segment. 'smooth' eases the whole timeline (smoothstep on t), not each segment, so the
 * motion only slows at the very start and end; with loop both ends have zero speed and the seam is smooth too.
 */
export function interpolateAt(keys: Keyframe[], t: number, opts: PathOptions): Complex[] {
  checkKeys(keys);
  const path = opts.loop ? [...keys, keys[0]] : keys;
  const u = Math.min(1, Math.max(0, t));
  const s = (opts.easing === 'smooth' ? smoothstep(u) : u) * (path.length - 1);
  const j = Math.min(Math.floor(s), path.length - 2);
  const f = s - j;
  // a(1-f) + bf rather than a + (b-a)f: exact at both ends, so keyframes are hit bit for bit.
  return path[j].map((a, i) => {
    const b = path[j + 1][i];
    return { re: a.re * (1 - f) + b.re * f, im: a.im * (1 - f) + b.im * f };
  });
}

/** Timeline position where keyframe `index` is reached: the inverse of the easing (smoothstep inverts in closed form). */
export function keyframeTime(index: number, keyCount: number, opts: PathOptions): number {
  const s = index / (opts.loop ? keyCount : keyCount - 1);
  return opts.easing === 'smooth' ? 0.5 - Math.sin(Math.asin(1 - 2 * s) / 3) : s;
}

/** Timeline position of frame i. With loop the last frame stops one step short of t = 1, which would repeat
 * frame 0: the video then loops without a stutter. */
export function frameTime(i: number, frameCount: number, loop: boolean): number {
  if (loop) return i / frameCount;
  return frameCount > 1 ? i / (frameCount - 1) : 0;
}

/** Coefficient set of every frame. */
export function framesFor(keys: Keyframe[], frameCount: number, opts: PathOptions): Complex[][] {
  if (!Number.isInteger(frameCount) || frameCount < 1) throw new RangeError('frame count must be a whole number ≥ 1');
  return Array.from({ length: frameCount }, (_, i) => interpolateAt(keys, frameTime(i, frameCount, opts.loop), opts));
}

/**
 * Pin the leading digit only if every frame allows it. Decided once per animation: pinning halves (k-folds)
 * the root multiplicity, so switching it between frames would make the brightness jump.
 */
export function pinnable(frames: Complex[][]): boolean {
  return frames.length > 0 && frames.every(leadingFixable);
}

/** Frames per splatFrames batch for frames of `frameBytes` histogram each (W * H * 16): 8 at 1080p, 4 at 3840x2160. */
export function batchFrames(frameBytes: number, limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }): number {
  const cap = Math.min(MAX_BATCH_BYTES, limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  return Math.max(1, Math.min(MAX_BATCH_FRAMES, Math.floor(cap / frameBytes)));
}

/** H.264 encodes 4:2:0 chroma (one sample per 2x2 pixels); encoders reject odd sizes. */
export function videoSizeError(width: number, height: number): string | null {
  return width % 2 === 0 && height % 2 === 0 ? null : 'MP4 (H.264) needs an even width and height';
}
