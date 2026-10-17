import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  build: { target: 'es2022' },
  test: {
    pool: 'forks', // dawn.node is a native addon: keep it out of worker threads
    testTimeout: 120_000,
    include: ['tests/**/*.test.ts'],
  },
});
