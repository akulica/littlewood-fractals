import type { Complex } from './complex';
import { leadingFixable } from './enumeration';
import type { ViewRect } from './view';

/** One coefficient set; every keyframe of an animation has the same length k. */
export type CoeffSet = Complex[];
export type Easing = 'linear' | 'smooth';

/**
 * One animation keyframe. The coefficients are always there; the view and the exposure/gamma pair are optional
 * ("Attach view", "Attach exposure/gamma"): keyframes without them take the value interpolated between the nearest
 * keyframes that have them.
 */
export interface Keyframe {
  coeffs: Complex[];
  /** Visible world rectangle, not pxPerUnit: it frames the same region whatever the window size or dpr. */
  view?: ViewRect;
  exposure?: number;
  gamma?: number;
}

/** Everything that varies along the timeline, at one moment. */
export interface PathState {
  coeffs: Complex[];
  view: ViewRect;
  exposure: number;
  gamma: number;
}

/** Used for a channel no keyframe carries: the current screen view / sliders. */
export interface PathFallback {
  view: ViewRect;
  exposure: number;
  gamma: number;
}

export interface PathOptions {
  easing: Easing;
  /** Return to the first keyframe at the end, so the video loops. */
  loop: boolean;
}

/**
 * Animation brightness. locked: density per unit area is locked to frame 0 (countScale grows with the zoom squared,
 * frame 0's densest pixel is the reference), so a zoom keeps regions of constant root density equally bright.
 * frame0: every frame is tone-mapped against frame 0's densest pixel with a fixed countScale (zooming in darkens).
 * perFrame: each frame against its own densest pixel (pulses).
 */
export type Brightness = 'locked' | 'frame0' | 'perFrame';

/** Frames solved per thread in one splatFrames batch. Measured best on both GPUs (plan §1.2): fewer frames
 * leave warm-start gains unused, more lose histogram cache locality. = the coefficient buffer's frame capacity. */
export const MAX_BATCH_FRAMES = 8;
/** Upper bound for one batch's histograms (all frames of a batch live on the GPU at once). */
export const MAX_BATCH_BYTES = 512 * 2 ** 20;

const smoothstep = (t: number) => t * t * (3 - 2 * t);

function checkKeys(keys: CoeffSet[]): void {
  if (keys.length < 2) throw new RangeError('an animation needs at least two keyframes');
  if (keys.some((key) => key.length !== keys[0].length || key.length === 0)) {
    throw new RangeError('every keyframe must have the same number of coefficients');
  }
}

/** Eased position along the path in keyframe units: keyframe i sits at i, the loop's return to keys[0] at keyCount. */
function pathPosition(t: number, keyCount: number, opts: PathOptions): number {
  const u = Math.min(1, Math.max(0, t));
  return (opts.easing === 'smooth' ? smoothstep(u) : u) * (opts.loop ? keyCount : keyCount - 1);
}

/**
 * Coefficients at timeline position t in [0, 1]: piecewise linear through the keyframes (plus keys[0] again when
 * looping), equal time per segment. 'smooth' eases the whole timeline (smoothstep on t), not each segment, so the
 * motion only slows at the very start and end; with loop both ends have zero speed and the seam is smooth too.
 */
export function interpolateAt(keys: CoeffSet[], t: number, opts: PathOptions): Complex[] {
  checkKeys(keys);
  const path = opts.loop ? [...keys, keys[0]] : keys;
  const s = pathPosition(t, keys.length, opts);
  const j = Math.min(Math.floor(s), path.length - 2);
  const f = s - j;
  // a(1-f) + bf rather than a + (b-a)f: exact at both ends, so keyframes are hit bit for bit.
  return path[j].map((a, i) => {
    const b = path[j + 1][i];
    return { re: a.re * (1 - f) + b.re * f, im: a.im * (1 - f) + b.im * f };
  });
}

/**
 * Zoom geometric (log-linear in the size), centre anchored to the more zoomed-in rectangle: that rectangle's centre
 * moves towards the frame centre linearly in screen units while the zoom changes geometrically, so the target never
 * leaves the screen. Zooming in towards b: c = b + (a - b) * (s_a / s) * (1 - f); zooming out: the mirror image,
 * anchored at a; equal zoom: plain linear panning. Rejected: a linear centre (zooming 1 -> 1000x towards a point
 * 1 unit away flies it ~16 screen widths off-screen mid-way) and van Wijk-Nuij (zooms out during equal-zoom pans).
 */
export function interpolateView(a: ViewRect, b: ViewRect, f: number): ViewRect {
  const halfW = a.halfW ** (1 - f) * b.halfW ** f;
  const halfH = a.halfH ** (1 - f) * b.halfH ** f;
  // Zoom = 1 / geometric-mean half extent: also log-linear, and defined when the aspect ratios differ.
  const sa = 1 / Math.sqrt(a.halfW * a.halfH);
  const sb = 1 / Math.sqrt(b.halfW * b.halfH);
  const s = 1 / Math.sqrt(halfW * halfH);
  // k = weight of a, applied as a k + b (1 - k): exact at k = 0 and k = 1.
  const k = sb >= sa ? (sa / s) * (1 - f) : 1 - (sb / s) * f;
  return { cx: a.cx * k + b.cx * (1 - k), cy: a.cy * k + b.cy * (1 - k), halfW, halfH };
}

/**
 * One optional channel at path position u: interpolated between the nearest keyframes that carry it (held before
 * the first / after the last; with loop it wraps around, so the seam is smooth). undefined if no keyframe carries it.
 */
function channelAt<T>(keys: Keyframe[], u: number, loop: boolean, get: (k: Keyframe) => T | undefined, mix: (a: T, b: T, f: number) => T): T | undefined {
  const marks: { p: number; v: T }[] = [];
  keys.forEach((key, i) => {
    const v = get(key);
    if (v !== undefined) marks.push({ p: i, v });
  });
  if (marks.length <= 1) return marks[0]?.v;
  if (loop) {
    const first = marks[0];
    const last = marks[marks.length - 1];
    marks.unshift({ p: last.p - keys.length, v: last.v });
    marks.push({ p: first.p + keys.length, v: first.v });
  }
  let j = 0;
  while (j + 1 < marks.length && marks[j + 1].p <= u) j++;
  const a = marks[j];
  if (u <= a.p || j + 1 === marks.length) return a.v; // on a keyframe (bit-exact), before the first or after the last
  const b = marks[j + 1];
  return mix(a.v, b.v, (u - a.p) / (b.p - a.p));
}

const displayOf = (k: Keyframe) => (k.exposure !== undefined && k.gamma !== undefined ? { exposure: k.exposure, gamma: k.gamma } : undefined);
const mixDisplay = (a: { exposure: number; gamma: number }, b: { exposure: number; gamma: number }, f: number) => ({
  exposure: a.exposure ** (1 - f) * b.exposure ** f, // exposure multiplies: geometric, like the zoom
  gamma: a.gamma * (1 - f) + b.gamma * f,
});

/**
 * Full state at timeline position t: coefficients as interpolateAt, the view by interpolateView, exposure
 * geometrically, gamma linearly. Every channel uses the same eased position, so loop and easing apply to all.
 */
export function stateAt(keys: Keyframe[], t: number, opts: PathOptions, fallback: PathFallback): PathState {
  const coeffs = interpolateAt(keys.map((k) => k.coeffs), t, opts);
  const u = pathPosition(t, keys.length, opts);
  const view = channelAt(keys, u, opts.loop, (k) => k.view, interpolateView) ?? fallback.view;
  const display = channelAt(keys, u, opts.loop, displayOf, mixDisplay) ?? fallback;
  return { coeffs, view, exposure: display.exposure, gamma: display.gamma };
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

function checkFrameCount(frameCount: number): void {
  if (!Number.isInteger(frameCount) || frameCount < 1) throw new RangeError('frame count must be a whole number ≥ 1');
}

/** Coefficient set of every frame. */
export function framesFor(keys: CoeffSet[], frameCount: number, opts: PathOptions): Complex[][] {
  checkFrameCount(frameCount);
  return Array.from({ length: frameCount }, (_, i) => interpolateAt(keys, frameTime(i, frameCount, opts.loop), opts));
}

/** Full state of every frame: the same timeline positions ▶ Preview steps through. */
export function statesFor(keys: Keyframe[], frameCount: number, opts: PathOptions, fallback: PathFallback): PathState[] {
  checkFrameCount(frameCount);
  return Array.from({ length: frameCount }, (_, i) => stateAt(keys, frameTime(i, frameCount, opts.loop), opts, fallback));
}

/**
 * countScale and fixedMax of one frame. baseScale: (export px per screen px)^2, as Save PNG uses. zoom: this frame's
 * pxPerUnit / frame 0's. max0: frame 0's densest pixel (raw count, found at baseScale).
 * locked: a pixel at zoom s covers 1/s^2 of the plane, so counts are scaled by zoom^2 (roots per unit area) and the
 * tone-map reference stays frame 0's: fixedMax * countScale = max0 * baseScale in every frame.
 */
export function frameBrightness(mode: Brightness, baseScale: number, zoom: number, max0: number): { countScale: number; fixedMax: number } {
  if (mode === 'perFrame') return { countScale: baseScale, fixedMax: 0 };
  if (mode === 'frame0') return { countScale: baseScale, fixedMax: max0 };
  const z2 = zoom * zoom;
  return { countScale: baseScale * z2, fixedMax: max0 / z2 };
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
