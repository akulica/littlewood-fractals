import { batchFrames, frameBrightness, pinnable, type Brightness } from '../core/animation';
import type { Complex } from '../core/complex';
import { totalPolys } from '../core/enumeration';
import { nextBudget } from '../core/schedule';
import { worldToPixel, type View } from '../core/view';
import { Engine, RgbaReadback, type DisplayParams } from './engine';

/** Fraction of each frame's polynomials to render (a golden-order prefix: unbiased, just noisier). */
export type DraftQuality = 1 | 0.5 | 0.25 | 0.1;

export interface AnimationRequest {
  /** One coefficient set per frame (core/animation framesFor), all of the same size. */
  coeffFrames: Complex[][];
  degree: number;
  /** Output view of every frame unless viewFrames is given. */
  view: View;
  /** One output view per frame (view keyframes: zoom and pan), same length as coeffFrames. */
  viewFrames?: View[];
  width: number;
  height: number;
  /** countScale = (export px per screen px)^2, as for Save PNG, so frames are as bright as the screen. */
  display: DisplayParams;
  /** Per-frame exposure and gamma (exposure/gamma keyframes); default display's. */
  displayFrames?: { exposure: number; gamma: number }[];
  /** Coefficient rings in every frame, at that frame's coefficients; scale = output px per CSS px. */
  rings?: { scale: number };
  /** locked / frame0 tone-map against frame 0's densest pixel: it jumps 1.3-1.5x between adjacent frames (plan §1.3),
   * so per-frame normalisation makes the video pulse. locked also corrects for the zoom (core/animation frameBrightness). */
  brightness: Brightness;
  quality: DraftQuality;
  maxIter: number;
}

export interface AnimationProgress {
  /** Frames handed to the sink so far. */
  frame: number;
  frames: number;
  /** Polynomial-frames solved so far, of polysTotal. */
  polysDone: number;
  polysTotal: number;
}

/** What frame i was rendered with (PNG frame metadata). */
export interface FrameInfo {
  coeffs: Complex[];
  view: View;
  display: DisplayParams;
}

/** Receives each frame's tightly packed RGBA, in order; rendering waits for it (backpressure from the encoder). */
export type FrameSink = (rgba: Uint8ClampedArray<ArrayBuffer>, frame: number, info: FrameInfo) => Promise<void> | void;

const TARGET_MS = 100; // per chunk, as in exports: big chunks, far below the ~2 s Windows TDR

/**
 * Renders every frame at full size off-screen. Frames go to the GPU in batches of up to 8 (core/animation
 * batchFrames): the splatFrames kernel solves each polynomial for all frames of a batch in turn, warm-starting from
 * the previous frame's roots (2.2-2.6x faster than cold frames at 1080p). Then each frame's histogram slice is
 * tone-mapped, read back and passed to `sink`. Throws signal.reason when aborted (checked between chunks and frames).
 */
export async function renderAnimation(
  device: GPUDevice,
  req: AnimationRequest,
  sink: FrameSink,
  onProgress?: (p: AnimationProgress) => void,
  signal?: AbortSignal,
): Promise<{ fixedMax: number; unconverged: number }> {
  const { width, height, coeffFrames: frames } = req;
  if (frames.length === 0) throw new RangeError('no frames to render');
  const views = req.viewFrames ?? frames.map(() => req.view);
  if (views.length !== frames.length || (req.displayFrames && req.displayFrames.length !== frames.length)) {
    throw new RangeError('need one view and one exposure/gamma per frame');
  }
  const baseScale = req.display.countScale ?? 1;
  // Display of frame f once frame 0's densest pixel (max0) is known.
  const displayOf = (f: number, max0: number): DisplayParams => ({
    ...req.display,
    ...req.displayFrames?.[f],
    ...frameBrightness(req.brightness, baseScale, views[f].pxPerUnit / views[0].pxPerUnit, max0),
    rings: req.rings && { scale: req.rings.scale, centres: frames[f].map((c) => worldToPixel(views[f], width, height, c.re, c.im)) },
  });
  const frameBytes = width * height * 16;
  const perBatch = Math.min(frames.length, batchFrames(frameBytes, device.limits));
  const pinned = pinnable(frames);
  const total = totalPolys(frames[0].length, req.degree, pinned);
  // Golden order makes every prefix an even sample of the whole set, so a draft is simply a shorter prefix.
  const perFrame = req.quality >= 1 ? total : (total * BigInt(Math.round(req.quality * 1000)) + 999n) / 1000n;
  const progress: AnimationProgress = { frame: 0, frames: frames.length, polysDone: 0, polysTotal: Number(perFrame) * frames.length };
  const engine = new Engine(device, 'rgba8unorm');
  const out = new RgbaReadback(device, width, height);
  let slices: GPUBuffer | null = null;
  let fixedMax = 0;
  let unconverged = 0;
  try {
    engine.resize(width, height);
    slices = device.createBuffer({ size: perBatch * frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    let budget = 1 << 12; // polynomials per chunk, each solved for a whole batch of frames; adapts to TARGET_MS
    for (let first = 0; first < frames.length; first += perBatch) {
      const batch = frames.slice(first, first + perBatch);
      engine.setFrames({ frames: batch, views: views.slice(first, first + perBatch), degree: req.degree, view: req.view, maxIter: req.maxIter, pinned }, slices);
      while (engine.progress.done < perFrame) {
        signal?.throwIfAborted();
        const left = perFrame - engine.progress.done;
        const enc = device.createCommandEncoder();
        engine.encodeCompute(enc, left < BigInt(budget) ? Number(left) : budget);
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        budget = nextBudget(budget, performance.now() - t0, TARGET_MS);
        progress.polysDone = first * Number(perFrame) + Number(engine.progress.done) * batch.length;
        onProgress?.({ ...progress });
      }
      unconverged += (await engine.readStats()).unconverged;
      for (let i = 0; i < batch.length; i++) {
        signal?.throwIfAborted();
        if (req.brightness !== 'perFrame' && fixedMax === 0) {
          // Frame 0 sets the scale: one extra tone-map pass just to find its densest pixel.
          const enc = device.createCommandEncoder();
          engine.loadHistogram(enc, slices, i * frameBytes);
          engine.encodeDisplay(enc, out.view, { ...displayOf(0, 0), fixedMax: 0, countScale: baseScale });
          device.queue.submit([enc.finish()]);
          fixedMax = Math.max(1, await engine.readMaxCount());
        }
        const f = first + i;
        const display = displayOf(f, fixedMax);
        const enc = device.createCommandEncoder();
        engine.loadHistogram(enc, slices, i * frameBytes);
        engine.encodeDisplay(enc, out.view, display);
        await sink(await out.read(enc), f, { coeffs: frames[f], view: views[f], display });
        progress.frame = first + i + 1;
        onProgress?.({ ...progress });
      }
    }
    return { fixedMax, unconverged };
  } finally {
    slices?.destroy();
    out.destroy();
    engine.destroy();
  }
}
