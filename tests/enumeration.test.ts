import { describe, expect, it } from 'vitest';
import { goldenStride, indexToDigits, leadingFixable, sampleIndex, totalPolys } from '../src/core/enumeration';

const c = (re: number, im = 0) => ({ re, im });
const w = (k: number, j: number) => c(Math.cos((2 * Math.PI * j) / k), Math.sin((2 * Math.PI * j) / k));

describe('leadingFixable', () => {
  it('accepts sets that are c0 times a finite group', () => {
    expect(leadingFixable([c(1), c(-1)])).toBe(true);
    expect(leadingFixable([c(2), c(-2)])).toBe(true);
    expect(leadingFixable([c(1), c(0, 1), c(-1), c(0, -1)])).toBe(true);
    expect(leadingFixable([w(3, 0), w(3, 1), w(3, 2)])).toBe(true);
  });
  it('rejects everything else', () => {
    expect(leadingFixable([c(1), c(-0.5)])).toBe(false);
    expect(leadingFixable([c(1), c(0)])).toBe(false); // 0 is not invertible
    expect(leadingFixable([c(1), c(-1), c(0.5)])).toBe(false);
    expect(leadingFixable([c(1)])).toBe(false);
  });
});

describe('totalPolys', () => {
  it('is k^(d+1), or k^d when the leading digit is pinned', () => {
    expect(totalPolys(2, 20, true)).toBe(2n ** 20n);
    expect(totalPolys(3, 4, false)).toBe(243n);
  });
});

describe('indexToDigits', () => {
  it('is little-endian base k with d+1 digits', () => {
    expect([...indexToDigits(5n, 2, 3)]).toEqual([1, 0, 1, 0]);
    expect([...indexToDigits(242n, 3, 4)]).toEqual([2, 2, 2, 2, 2]);
  });
  it('handles indices beyond 2^53', () => {
    expect([...indexToDigits(3n ** 30n - 1n, 3, 29)]).toEqual(new Array(30).fill(2));
  });
});

const gcd = (a: bigint, b: bigint): bigint => (b === 0n ? a : gcd(b, a % b));

describe('goldenStride', () => {
  it('is coprime to k, lies in (0, N) and is about 0.618 N', () => {
    for (let k = 2; k <= 16; k++) {
      for (const D of [1, 2, 3, 7, 12]) {
        const N = BigInt(k) ** BigInt(D);
        const G = goldenStride(N, k);
        expect(gcd(G, BigInt(k))).toBe(1n);
        expect(G > 0n && G < N).toBe(true);
        if (N > 1000n) expect(Math.abs(Number(G) / Number(N) - 0.6180339887498949)).toBeLessThan(0.02);
      }
    }
  });
  it('is 1 for N <= 2 (nothing to spread)', () => {
    expect(goldenStride(1n, 2)).toBe(1n);
    expect(goldenStride(2n, 2)).toBe(1n);
  });
  it('is exact BigInt arithmetic for N > 2^64', () => {
    const N = 3n ** 45n; // ~2^71
    const G = goldenStride(N, 3);
    expect(gcd(G, 3n)).toBe(1n);
    // floor(N * 2783565922149187 / 2^52), bumped by at most a couple of steps to reach gcd(G, 3) = 1
    const floor = (N * 2783565922149187n) >> 52n;
    expect(G - floor >= 0n && G - floor < 3n).toBe(true);
    expect(sampleIndex(1n, G, N)).toBe(G);
    expect(sampleIndex(N - 1n, G, N)).toBe(N - G);
    expect(sampleIndex(N, G, N)).toBe(0n);
  });
});

describe('sampleIndex with the golden stride', () => {
  // m -> m*G mod N is a bijection of [0, N): the full render is the continuation of the preview.
  it.each([
    [3, 5],
    [2, 10],
    [4, 4],
  ])('hits every index exactly once (k = %i, %i digits)', (k, D) => {
    const N = BigInt(k) ** BigInt(D);
    const G = goldenStride(N, k);
    const seen = new Set<bigint>();
    for (let m = 0n; m < N; m++) seen.add(sampleIndex(m, G, N));
    expect(seen.size).toBe(Number(N));
    expect([...seen].every((i) => i >= 0n && i < N)).toBe(true);
  });
  it('covers every pattern of the low j digits in each block of k^j consecutive samples', () => {
    const k = 3;
    const N = 3n ** 8n;
    const G = goldenStride(N, k);
    const block = 9n; // j = 2
    for (let m0 = 0n; m0 < 50n * block; m0 += block) {
      const low = new Set<bigint>();
      for (let m = m0; m < m0 + block; m++) low.add(sampleIndex(m, G, N) % block);
      expect(low.size).toBe(9);
    }
  });
});
