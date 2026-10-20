// Spawns the Web Worker PNG encoders of core/pngPool. Vite bundles ui/pngWorkerMain from the `new Worker(new URL(...))`.
import type { PngEncoder } from '../core/pngPool';
import { rpcEncoder, type PngReply } from '../core/pngWorkerCore';

/** One encoder in its own Web Worker. Throws when the browser cannot create the Worker. */
export function spawnWebPngEncoder(): PngEncoder {
  const worker = new Worker(new URL('./pngWorkerMain.ts', import.meta.url), { type: 'module' });
  const { encoder, reply, fail } = rpcEncoder(
    (r, transfer) => worker.postMessage(r, transfer),
    () => worker.terminate(),
  );
  worker.onmessage = (e: MessageEvent<PngReply>) => reply(e.data);
  worker.onerror = (e) => fail(new Error(`PNG worker failed: ${e.message || 'could not load'}`));
  worker.onmessageerror = () => fail(new Error('PNG worker sent an unreadable message'));
  return encoder;
}
