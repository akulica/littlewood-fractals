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
  export function writeFileSync(path: string, data: Uint8Array): void;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function existsSync(path: string): boolean;
}
declare module 'node:process' {
  export const env: Record<string, string | undefined>;
  export const version: string;
}
