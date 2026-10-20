// Web Worker entry: PNG frame encoder (spawned by ui/pngWorker; protocol in core/pngWorkerCore).
import { handlePngRequest, type PngReply, type PngRequest } from '../core/pngWorkerCore';

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<PngRequest>) => void) | null;
  postMessage(m: PngReply, transfer: Transferable[]): void;
};
scope.onmessage = async (e) => {
  const reply = await handlePngRequest(e.data);
  scope.postMessage(reply, 'png' in reply ? [reply.png] : []);
};
