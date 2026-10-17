import { describe, expect, it } from 'vitest';
import { indexToDigits, leadingFixable, totalPolys } from '../src/core/enumeration';

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
