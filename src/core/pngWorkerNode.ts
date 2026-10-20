// Node side of the PNG worker pool (worker_threads). Not imported by browser code.
import { isMainThread, parentPort, Worker } from 'node:worker_threads';
import type { PngEncoder } from './pngPool';
import { handlePngRequest, rpcEncoder, type PngReply, type PngRequest } from './pngWorkerCore';

/** Main thread: one encoder running `entry` (a script that calls runNodePngWorker when it is not the main thread). */
export function spawnNodePngEncoder(entry: string | URL): PngEncoder {
  const worker = new Worker(entry);
  const { encoder, reply, fail } = rpcEncoder(
    (r, transfer) => worker.postMessage(r, transfer),
    () => void worker.terminate(),
  );
  worker.on('message', reply);
  worker.on('error', fail);
  worker.on('exit', (code) => fail(new Error(`PNG worker exited (code ${code})`)));
  return encoder;
}

/** Worker thread: serve encode requests until terminated. */
export function runNodePngWorker(): void {
  if (isMainThread || !parentPort) throw new Error('runNodePngWorker must run inside a worker thread');
  const port = parentPort;
  port.on('message', async (r: PngRequest) => {
    const reply: PngReply = await handlePngRequest(r);
    port.postMessage(reply, 'png' in reply ? [reply.png] : []);
  });
}
