// PNG encoding on worker threads, overlapped with rendering. The pool itself is environment-free: it is given a
// factory for encoders (one per worker; Node worker_threads in core/pngWorkerNode, Web Workers in ui/pngWorker) and
// adds the two guarantees the exports need: a bounded number of frames in flight (the producer awaits a slot, so
// 4K RGBA buffers cannot pile up) and delivery strictly in submission order, whichever worker finishes first.
import type { PngText } from './png';

export interface PngJob {
  /** Tightly packed RGBA, width * height * 4 bytes. The pool takes ownership: the buffer is transferred to a worker. */
  rgba: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  texts: PngText[];
}

/** One worker: encodes one frame at a time and resolves with the file's bytes. */
export interface PngEncoder {
  encode(job: PngJob): Promise<Uint8Array<ArrayBuffer>>;
  terminate(): void;
}

/** min(4, max(1, cores - 2)): leave the render loop and the GPU driver their own cores. */
export const defaultPngWorkers = (cores: number): number => Math.min(4, Math.max(1, (Number.isFinite(cores) ? Math.floor(cores) : 1) - 2));

/** The buffer to transfer for `rgba`: its own when it spans the whole ArrayBuffer, else a copy (never detaches a larger buffer). */
export function transferable(rgba: Uint8Array | Uint8ClampedArray): ArrayBuffer {
  const b = rgba.buffer;
  if (b instanceof ArrayBuffer && rgba.byteOffset === 0 && rgba.byteLength === b.byteLength) return b;
  return rgba.slice().buffer as ArrayBuffer;
}

export class PngPool<T = number> {
  /** At most this many frames are in flight (submitted, not yet delivered): twice the workers. */
  readonly maxInFlight: number;
  private readonly encoders: PngEncoder[];
  private readonly idle: PngEncoder[] = [];
  private readonly encoderWaiters: ((e: PngEncoder) => void)[] = [];
  private readonly slotWaiters: (() => void)[] = [];
  private inFlight = 0;
  private chain: Promise<void> = Promise.resolve();
  private failure: { error: unknown } | null = null;
  private closed = false;

  /**
   * @param spawn creates one encoder; called `workers` times up front (so a Worker that cannot be created throws here)
   * @param deliver gets each PNG with the tag it was submitted under, one at a time, in submission order. The next
   *   frame's slot is only freed once it returns, so a slow disk back-pressures the producer too.
   */
  constructor(
    spawn: () => PngEncoder,
    readonly workers: number,
    private readonly deliver: (png: Uint8Array<ArrayBuffer>, tag: T) => Promise<void> | void,
  ) {
    if (!Number.isInteger(workers) || workers < 1) throw new RangeError('a PNG pool needs at least one worker');
    this.maxInFlight = workers * 2;
    this.encoders = [];
    try {
      for (let i = 0; i < workers; i++) this.encoders.push(spawn());
    } catch (e) {
      this.terminate();
      throw e;
    }
    this.idle.push(...this.encoders);
  }

  /** Frames submitted and not yet delivered. */
  get pending(): number {
    return this.inFlight;
  }

  /**
   * Queues a frame, resolving once it has a slot (immediately while fewer than maxInFlight are pending). Throws the
   * first error any earlier frame hit (encoding or `deliver`).
   */
  async submit(job: PngJob, tag: T): Promise<void> {
    while (this.inFlight >= this.maxInFlight && !this.failure && !this.closed) await new Promise<void>((r) => this.slotWaiters.push(r));
    if (this.failure) throw this.failure.error;
    if (this.closed) throw new Error('the PNG pool is closed');
    this.inFlight++;
    const encoded = this.run(job);
    encoded.catch(() => {}); // surfaced by the chain below, not as an unhandled rejection in between
    this.chain = this.chain.then(async () => {
      try {
        if (!this.failure) await this.deliver(await encoded, tag);
      } catch (error) {
        this.fail(error);
      } finally {
        this.inFlight--;
        this.slotWaiters.shift()?.();
      }
    });
  }

  /** Encodes a 1x1 frame on one worker, rejecting if it cannot (script failed to load, ...). For a start-up probe. */
  async check(): Promise<void> {
    await this.encoders[0].encode({ rgba: new Uint8Array(4), width: 1, height: 1, texts: [] });
  }

  /** Waits for every submitted frame to be delivered, then stops the workers. Throws the first error. */
  async finish(): Promise<void> {
    await this.chain;
    this.terminate();
    if (this.failure) throw this.failure.error;
  }

  /** Stops the workers now; frames not yet delivered are dropped. */
  terminate(): void {
    this.closed = true;
    for (const e of this.encoders) e.terminate();
    for (const w of this.slotWaiters.splice(0)) w();
  }

  private fail(error: unknown): void {
    this.failure ??= { error };
    for (const w of this.slotWaiters.splice(0)) w();
  }

  private async run(job: PngJob): Promise<Uint8Array<ArrayBuffer>> {
    const enc = this.idle.pop() ?? (await new Promise<PngEncoder>((r) => this.encoderWaiters.push(r)));
    try {
      return await enc.encode(job);
    } finally {
      const next = this.encoderWaiters.shift();
      if (next) next(enc);
      else this.idle.push(enc);
    }
  }
}
