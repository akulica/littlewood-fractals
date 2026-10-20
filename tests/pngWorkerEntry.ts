// Entry the pngPool test bundles into a real worker_threads script (the shipped one is scripts/render-headless.ts).
import { runNodePngWorker } from '../src/core/pngWorkerNode';

runNodePngWorker();
