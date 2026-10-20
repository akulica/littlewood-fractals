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

/** Frames solved per polynomial in one splatFrames / splatFramesCoop batch. Measured best on both GPUs with
 * splatFrames (plan §1.2): fewer frames leave warm-start gains unused, more lose histogram cache locality.
 * = the coefficient buffer's frame capacity. */
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

/**
 * Keyframes no frame lands on or next to: keyframe i counts as shown when some frame's path position rounds to i
 * (the frame is within half a keyframe of it). With fewer frames than keyframes, or with smooth easing (it moves
 * 1.5x faster than linear mid-timeline), frames step over keyframes, and a keyframe-only detail (a sharp turn, a
 * one-keyframe exposure spike) can vanish from the video.
 */
export function uncoveredKeyframes(keyCount: number, frameCount: number, opts: PathOptions): number {
  if (keyCount < 2 || frameCount < 1) return 0;
  const shown = new Set<number>();
  for (let i = 0; i < frameCount; i++) shown.add(Math.round(pathPosition(frameTime(i, frameCount, opts.loop), keyCount, opts)) % keyCount);
  return keyCount - shown.size;
}

/**
 * The smallest frame count ≥ the keyframe count that shows every keyframe (uncoveredKeyframes = 0), at most
 * maxFrames. Linear: exactly the keyframe count, and every keyframe then falls exactly on a frame. Smooth: about
 * 1.5x the keyframe count, and frames land near (not on) the inner keyframes.
 */
export function framesToShowKeyframes(keyCount: number, opts: PathOptions, maxFrames = 100_000): number {
  for (let n = Math.max(2, keyCount); n < maxFrames; n++) if (uncoveredKeyframes(keyCount, n, opts) === 0) return n;
  return maxFrames;
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

/** Frames per splatFrames / splatFramesCoop batch for frames of `frameBytes` histogram each (W * H * 16): 8 at 1080p, 4 at 3840x2160. */
export function batchFrames(frameBytes: number, limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }): number {
  const cap = Math.min(MAX_BATCH_BYTES, limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  return Math.max(1, Math.min(MAX_BATCH_FRAMES, Math.floor(cap / frameBytes)));
}

/** Auto batch size: a candidate replaces the memory cap only when it solves a frame at least this much faster (noise guard). */
export const BATCH_SWITCH_GAIN = 0.05;

/** Batch sizes the auto calibration compares, largest first: the memory cap, half of it, a quarter (never below 2:
 * a one-frame batch has no warm start at all). One entry: nothing to compare. */
export function batchCandidates(cap: number): number[] {
  const sizes = [cap, Math.max(2, Math.floor(cap / 2)), Math.max(2, Math.floor(cap / 4))].filter((n) => n <= cap);
  return [...new Set(sizes)];
}

/**
 * Fastest candidate by solve time per frame (`ms`: best sample per batch size); the cap unless another size beats it by
 * BATCH_SWITCH_GAIN. Sizes without a sample are ignored.
 */
export function chooseBatchFrames(cap: number, ms: ReadonlyMap<number, number>): number {
  const capMs = ms.get(cap);
  if (capMs === undefined) return cap;
  let best = cap;
  let bestMs = capMs * (1 - BATCH_SWITCH_GAIN);
  for (const [size, t] of ms) {
    if (size < cap && t < bestMs) {
      best = size;
      bestMs = t;
    }
  }
  return best;
}

/**
 * Sizes of the batches of one animation. A fixed size just repeats. 'auto' calibrates online: the first batches are real
 * work rendered at the candidate sizes (cap, half, quarter, then the cap again, because a process's first batch pays
 * for clock ramp-up and first use of the buffers), each timed by its caller (record(): GPU solve time per frame, no
 * readback). The best size per frame, kept only if it beats the cap by BATCH_SWITCH_GAIN, then renders the rest.
 * Animations too short to profit (fewer frames than the trial plus one more cap batch) keep the cap. Batch sizes only
 * move where the warm-start chain restarts cold.
 */
export class BatchPlanner {
  /** The size the non-trial batches use: the forced size, the cap, or (auto, after the trial) the winner. */
  chosen: number;
  readonly auto: boolean;
  private readonly trial: number[];
  private started = 0;
  private recorded = 0;
  private readonly samples = new Map<number, number>();

  /** cap: batchFrames() for the output size; request: a forced size (capped) or 'auto'. */
  constructor(readonly cap: number, frameCount: number, request: number | 'auto' = 'auto') {
    if (!Number.isInteger(cap) || cap < 1) throw new RangeError('batch cap must be a whole number ≥ 1');
    if (request !== 'auto' && (!Number.isInteger(request) || request < 1)) throw new RangeError("batchFrames must be a whole number ≥ 1 or 'auto'");
    this.cap = Math.min(cap, Math.max(1, frameCount));
    const candidates = batchCandidates(this.cap);
    const trial = candidates.length > 1 ? [...candidates, this.cap] : [];
    const profitable = trial.reduce((a, b) => a + b, 0) + this.cap <= frameCount;
    this.auto = request === 'auto' && trial.length > 0 && profitable;
    this.trial = this.auto ? trial : [];
    this.chosen = request === 'auto' ? this.cap : Math.min(request, this.cap);
  }

  /** Frames of the next batch, `left` frames remaining. */
  next(left: number): number {
    const size = this.started < this.trial.length ? this.trial[this.started++] : this.chosen;
    return Math.min(size, left);
  }

  /** Solve time per frame of the batch just rendered; only full-size trial batches count. */
  record(frames: number, msPerFrame: number): void {
    if (this.recorded >= this.trial.length || frames !== this.trial[this.recorded]) return;
    this.samples.set(frames, Math.min(this.samples.get(frames) ?? Infinity, msPerFrame));
    if (++this.recorded === this.trial.length) this.chosen = chooseBatchFrames(this.cap, this.samples);
  }

  /** What the trial measured (best ms per frame by batch size). */
  get timings(): { frames: number; msPerFrame: number }[] {
    return [...this.samples].map(([frames, msPerFrame]) => ({ frames, msPerFrame }));
  }
}

/** H.264 encodes 4:2:0 chroma (one sample per 2x2 pixels); encoders reject odd sizes. */
export function videoSizeError(width: number, height: number): string | null {
  return width % 2 === 0 && height % 2 === 0 ? null : 'MP4 (H.264) needs an even width and height';
}
