import { defineConfig } from 'vitest/config';

// Separate config so `npm test` never runs the slow benchmark (env-var tricks break under cmd.exe).
// silent:false -> console.log lines are shown; fileParallelism:false -> benchmarks never share the GPU.
export default defineConfig({
  test: { pool: 'forks', testTimeout: 600_000, include: ['tests/**/*.perf.ts'], silent: false, fileParallelism: false },
});
