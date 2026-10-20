// The two Node APIs vite.config.ts uses, declared here instead of pulling @types/node into the whole project
// (its globals, e.g. setTimeout returning NodeJS.Timeout, would leak into the browser code).
declare module 'node:child_process' {
  export function execSync(command: string, options?: { stdio?: ('ignore' | 'pipe' | 'inherit')[] }): { toString(): string };
}
declare module 'node:fs' {
  export function readFileSync(path: URL | string, encoding: 'utf8'): string;
}
