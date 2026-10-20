// The Node APIs vite.config.ts uses, declared here instead of pulling @types/node into the whole project
// (its globals, e.g. setTimeout returning NodeJS.Timeout, would leak into the browser code).
declare module 'node:child_process' {
  export function execSync(command: string, options?: { stdio?: ('ignore' | 'pipe' | 'inherit')[] }): { toString(): string };
}
declare module 'node:fs' {
  export function readFileSync(path: URL | string, encoding: 'utf8'): string;
}
// tests/gpu/ab.perf.ts (the A/B image harness) writes and reads PNG files and reads its options from the environment.
declare module 'node:fs' {
  export function readFileSync(path: URL | string): Uint8Array<ArrayBuffer>;
  export function writeFileSync(path: string, data: Uint8Array | string): void;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function existsSync(path: string): boolean;
}
declare module 'node:process' {
  export const env: Record<string, string | undefined>;
  export const version: string;
}
// src/core/pngWorkerNode.ts (PNG encoding on worker threads) and its test.
declare module 'node:worker_threads' {
  export const isMainThread: boolean;
  export const parentPort: { on(event: 'message', cb: (message: any) => void): void; postMessage(message: unknown, transfer?: ArrayBuffer[]): void } | null;
  export class Worker {
    constructor(entry: string | URL, options?: { eval?: boolean });
    postMessage(message: unknown, transfer?: ArrayBuffer[]): void;
    on(event: 'message', cb: (message: any) => void): this;
    on(event: 'error', cb: (error: unknown) => void): this;
    on(event: 'exit', cb: (code: number) => void): this;
    terminate(): Promise<number>;
  }
}
declare module 'node:os' {
  export function tmpdir(): string;
}
declare module 'node:path' {
  export function join(...parts: string[]): string;
}
declare module 'node:fs' {
  export function mkdtempSync(prefix: string): string;
  export function rmSync(path: string, options?: { recursive?: boolean; force?: boolean }): void;
}
declare module 'node:url' {
  export function fileURLToPath(url: string | URL): string;
}
