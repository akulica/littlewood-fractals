// The message protocol between PngPool encoders and their workers, shared by the Node (worker_threads) and browser
// (Web Worker) adapters. The worker runs core/png's own encodePng, so the files are byte-identical to the main-thread
// path by construction.
import { encodePng, type PngText } from './png';
import { transferable, type PngEncoder, type PngJob } from './pngPool';

export interface PngRequest {
  id: number;
  /** RGBA, transferred. */
  rgba: ArrayBuffer;
  width: number;
  height: number;
  texts: PngText[];
}

export type PngReply = { id: number; png: ArrayBuffer } | { id: number; error: string };

/** Worker side: one request in, one reply out. Never throws. */
export async function handlePngRequest(r: PngRequest): Promise<PngReply> {
  try {
    const blob = await encodePng(new Uint8Array(r.rgba), r.width, r.height, r.texts);
    return { id: r.id, png: await blob.arrayBuffer() };
  } catch (e) {
    return { id: r.id, error: e instanceof Error ? e.message : String(e) };
  }
}

/**
 * Main-thread side of one worker. `post` sends a request (listing the buffers to transfer); the adapter feeds the
 * worker's replies to `reply` and its crashes / load failures to `fail`, which rejects everything outstanding.
 */
export function rpcEncoder(post: (r: PngRequest, transfer: ArrayBuffer[]) => void, stop: () => void) {
  let nextId = 0;
  let dead: Error | null = null;
  const waiting = new Map<number, { ok: (png: Uint8Array<ArrayBuffer>) => void; err: (e: Error) => void }>();
  const fail = (e: unknown) => {
    dead ??= e instanceof Error ? e : new Error(String(e));
    for (const w of waiting.values()) w.err(dead);
    waiting.clear();
  };
  const encoder: PngEncoder = {
    encode(job: PngJob) {
      if (dead) return Promise.reject(dead);
      const id = nextId++;
      const rgba = transferable(job.rgba);
      return new Promise<Uint8Array<ArrayBuffer>>((ok, err) => {
        waiting.set(id, { ok, err });
        try {
          post({ id, rgba, width: job.width, height: job.height, texts: job.texts }, [rgba]);
        } catch (e) {
          waiting.delete(id);
          err(e instanceof Error ? e : new Error(String(e)));
        }
      });
    },
    terminate() {
      fail(new Error('PNG worker terminated'));
      stop();
    },
  };
  const reply = (r: PngReply) => {
    const w = waiting.get(r.id);
    if (!w) return;
    waiting.delete(r.id);
    if ('error' in r) w.err(new Error(r.error));
    else w.ok(new Uint8Array(r.png));
  };
  return { encoder, reply, fail };
}
