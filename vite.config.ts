import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

/** Recorded in every export's metadata (core/metadata BuildInfo). Computed once per build / dev-server start. */
function buildInfo(): { version: string; commit: string; builtAt: string } {
  let version = 'unknown';
  let commit = 'unknown';
  try {
    version = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')).version ?? 'unknown';
  } catch {
    // keep 'unknown'
  }
  try {
    commit = execSync('git describe --always --dirty', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() || 'unknown';
  } catch {
    // not a git checkout, or git is not installed
  }
  return { version, commit, builtAt: new Date().toISOString() };
}

export default defineConfig({
  base: './',
  build: { target: 'es2022' },
  define: { __BUILD_INFO__: JSON.stringify(buildInfo()) },
  test: {
    pool: 'forks', // dawn.node is a native addon: keep it out of worker threads
    testTimeout: 120_000,
    include: ['tests/**/*.test.ts'],
  },
});
