import { describe, expect, it } from 'vitest';
import { aberthRef } from '../src/core/aberthRef';
import type { Complex } from '../src/core/complex';
import { goldenStride, groupTable, indexToDigits, sampleIndex } from '../src/core/enumeration';
import {
  EnumCursor,
  chunkCounts,
  decodeThread,
  enumerationPlan,
  mixedDigits,
  partnerDigits,
  sharedGroupTable,
  type EnumPlan,
} from '../src/core/reciprocal';
import { maxRootError } from './helpers';

const c = (re: number, im = 0): Complex => ({ re, im });
const w = (k: number, j: number, c0: Complex = c(1)): Complex => {
  const t = (2 * Math.PI * j) / k;
  return { re: c0.re * Math.cos(t) - c0.im * Math.sin(t), im: c0.re * Math.sin(t) + c0.im * Math.cos(t) };
};
const roots = (k: number, c0?: Complex, order = Array.from({ length: k }, (_, j) => j)) => order.map((j) => w(k, j, c0));

/** Every string the plan's chunks cover (enumerated, plus partners), counted, after running it with `budget`. */
function coverage(plan: EnumPlan, budget: number, golden: boolean): Map<string, number> {
  const seen = new Map<string, number>();
  const add = (s: number[]) => seen.set(s.join(','), (seen.get(s.join(',')) ?? 0) + 1);
  const cursor = new EnumCursor(plan, golden);
  for (let entries = cursor.next(budget); entries.length > 0; entries = cursor.next(budget)) {
    expect(entries.reduce((n, e) => n + e.count, 0)).toBeLessThanOrEqual(budget);
    for (const e of entries) {
      for (let t = 0; t < e.count; t++) {
        const s = decodeThread(plan, e, t);
        add(s);
        if (plan.regions[e.region].partner) add(partnerDigits(plan, e.region, s));
      }
    }
  }
  expect(cursor.solved).toBe(plan.solves);
  expect(cursor.covered).toBe(plan.polys);
  return seen;
}

/** Asserts every polynomial of the plain enumeration (pinned: leading digit 0) is covered exactly once. */
function expectExactCover(plan: EnumPlan, budget: number, golden: boolean): void {
  const seen = coverage(plan, budget, golden);
  const { k, degree: d, pinned } = plan;
  const n = Number(plan.polys);
  expect(seen.size).toBe(n);
  for (let i = 0; i < n; i++) {
    const s = [...indexToDigits(BigInt(i), k, d)];
    if (pinned) {
      // index over k^d: digit d is the pinned 0
      s[d] = 0;
    }
    const key = s.join(',');
    if (seen.get(key) !== 1) throw new Error(`string ${key} covered ${seen.get(key) ?? 0} times (k=${k} d=${d} pinned=${pinned})`);
  }
}

describe('enumerationPlan: exact cover (every polynomial exactly once, enumerated or as a partner)', () => {
  const budgets = [1, 7, 1 << 20];
  it.each([2, 3, 4, 5])('unpinned, k = %i, degrees 1..7, golden and lexicographic, any chunking', (k) => {
    for (let d = 1; d <= (k <= 3 ? 7 : 5); d++) {
      const plan = enumerationPlan(k, d, false, 'reciprocal', null);
      expect(plan.mode).toBe('reciprocal');
      for (const budget of budgets) for (const golden of [true, false]) expectExactCover(plan, budget, golden);
    }
  });

  const pinnedSets: [string, Complex[]][] = [
    ['{±1}', [c(1), c(-1)]],
    ['{2, -2}', [c(2), c(-2)]],
    ['cube roots of unity', roots(3)],
    ['{1, i, -1, -i}', roots(4)],
    ['{1, -1, i, -i} (another digit order)', roots(4, c(1), [0, 2, 1, 3])],
    ['(0.5+0.3i) x sixth roots, shuffled', roots(6, c(0.5, 0.3), [0, 4, 1, 3, 5, 2])],
    ['fifth roots', roots(5)],
  ];
  it.each(pinnedSets)('pinned %s, degrees 1..7', (_, coeffs) => {
    const group = groupTable(coeffs);
    expect(group).not.toBeNull();
    const k = coeffs.length;
    for (let d = 1; d <= (k <= 3 ? 7 : k <= 4 ? 6 : 4); d++) {
      const plan = enumerationPlan(k, d, true, 'reciprocal', group);
      expect(plan.mode).toBe('reciprocal');
      for (const budget of budgets) for (const golden of [true, false]) expectExactCover(plan, budget, golden);
    }
  });

  it('plain plans cover every polynomial once too (pinned and unpinned)', () => {
    for (const pinned of [false, true]) expectExactCover(enumerationPlan(3, 5, pinned, 'plain', null), 10, true);
  });

  it('a pinned set without a group table falls back to plain', () => {
    expect(enumerationPlan(2, 6, true, 'reciprocal', null).mode).toBe('plain');
    expect(groupTable([c(1), c(1)])).toBeNull(); // duplicates: products are ambiguous
    expect(groupTable([c(1), c(-0.5)])).toBeNull();
  });
});

describe('partners are the reversed polynomials (same roots as the plain path solves)', () => {
  // Pinned: the partner T(s) must be a scalar multiple of reverse(s) with leading digit 0, so its roots are 1/z.
  it.each([
    ['{1, i, -1, -i}', roots(4)],
    ['(0.5+0.3i) x sixth roots, shuffled', roots(6, c(0.5, 0.3), [0, 4, 1, 3, 5, 2])],
  ])('pinned %s: partner coefficients = reversed coefficients x one constant', (_, coeffs) => {
    const d = 5;
    const plan = enumerationPlan(coeffs.length, d, true, 'reciprocal', groupTable(coeffs));
    const cursor = new EnumCursor(plan, true);
    for (const e of cursor.next(1 << 20)) {
      if (!plan.regions[e.region].partner) continue;
      for (let t = 0; t < e.count; t++) {
        const s = decodeThread(plan, e, t);
        const p = partnerDigits(plan, e.region, s);
        expect(p[d]).toBe(0);
        const ratio = (i: number) => {
          const a = coeffs[p[i]];
          const b = coeffs[s[d - i]];
          const m = b.re * b.re + b.im * b.im;
          return { re: (a.re * b.re + a.im * b.im) / m, im: (a.im * b.re - a.re * b.im) / m };
        };
        for (let i = 1; i <= d; i++) expect(Math.hypot(ratio(i).re - ratio(0).re, ratio(i).im - ratio(0).im)).toBeLessThan(1e-9);
      }
    }
  });

  // Unpinned with 0 in the set: the partner has d - hi(s) roots at 0 (s's trimmed leading zeros) and the
  // reciprocals of s's nonzero roots; s's roots at 0 become roots at infinity (dropped).
  it('a set containing 0: partner zero roots = d - hi, nonzero roots = 1/z', () => {
    const coeffs = [c(0), c(1), c(-1)];
    const d = 6;
    const plan = enumerationPlan(3, d, false, 'reciprocal', null);
    const cursor = new EnumCursor(plan, true);
    let checked = 0;
    for (const e of cursor.next(1 << 20)) {
      if (!plan.regions[e.region].partner) continue;
      for (let t = 0; t < e.count; t += 7) {
        const s = decodeThread(plan, e, t);
        const p = partnerDigits(plan, e.region, s);
        expect(p).toEqual([...s].reverse());
        const a = s.map((x) => coeffs[x]);
        let hi = d;
        while (hi > 0 && s[hi] === 0) hi--;
        const zs = aberthRef(a);
        const nonzero = zs.filter((z) => Math.hypot(z.re, z.im) > 1e-9);
        const expected = [...Array.from({ length: d - hi }, () => c(0)), ...nonzero.map((z) => { const m = z.re * z.re + z.im * z.im; return c(z.re / m, -z.im / m); })];
        expect(maxRootError(aberthRef(p.map((x) => coeffs[x])), expected)).toBeLessThan(1e-6);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(50);
  });
});

describe('enumerationPlan sizes', () => {
  it('reciprocal solves about half the polynomials (at most 1/32 solved alone)', () => {
    for (const [k, d, pinned, coeffs] of [
      [2, 20, false, null],
      [3, 14, false, null],
      [16, 6, false, null],
      [2, 20, true, [c(1), c(-1)]],
      [4, 12, true, roots(4)],
      [16, 6, true, roots(16)],
    ] as const) {
      const plan = enumerationPlan(k, d, pinned, 'reciprocal', coeffs && groupTable([...coeffs]));
      const ratio = Number(plan.solves) / Number(plan.polys);
      expect(ratio).toBeGreaterThan(0.5);
      expect(ratio).toBeLessThanOrEqual(0.5 * (1 + 1 / 32) + 1e-12);
      expect(plan.maxRadix).toBeLessThanOrEqual(256);
    }
  });

  it('plain is one region whose golden order is the original base-k index order', () => {
    const plan = enumerationPlan(3, 6, false, 'plain', null);
    expect(plan.regions).toHaveLength(1);
    const N = 3n ** 7n;
    const cursor = new EnumCursor(plan, true);
    cursor.next(100);
    const [e] = cursor.next(50);
    for (let t = 0; t < 50; t++) {
      expect(decodeThread(plan, e, t)).toEqual([...indexToDigits(sampleIndex(100n + BigInt(t), goldenStride(N, 3), N), 3, 6)]);
    }
  });
});

describe('chunkCounts', () => {
  it('splits each chunk across regions in proportion, so every prefix samples every region evenly', () => {
    const sizes = [524288n, 262144n, 131072n, 65536n, 32768n, 65536n, 3n, 1n];
    const total = sizes.reduce((a, b) => a + b, 0n);
    const done = sizes.map(() => 0n);
    let sum = 0n;
    for (const budget of [1, 5, 1000, 3, 77777, 2, 1 << 16, 1 << 20]) {
      const n = total - sum < BigInt(budget) ? Number(total - sum) : budget;
      const counts = chunkCounts(sizes, done, n);
      expect(counts.reduce((a, b) => a + b, 0)).toBe(n);
      counts.forEach((x, r) => {
        expect(x).toBeGreaterThanOrEqual(0);
        done[r] += BigInt(x);
        expect(done[r] <= sizes[r]).toBe(true);
      });
      sum += BigInt(n);
      // Within 1 solve of the exact proportional share (times the number of regions, for slack).
      sizes.forEach((s, r) => expect(Math.abs(Number(done[r]) - (Number(sum) * Number(s)) / Number(total))).toBeLessThan(sizes.length));
    }
    expect(done).toEqual(sizes);
  });
});

describe('helpers', () => {
  it('mixedDigits is little-endian mixed radix', () => {
    expect(mixedDigits(23n, [2, 3, 4])).toEqual([1, 2, 3]);
    expect(mixedDigits(0n, [1, 1, 5])).toEqual([0, 0, 0]);
  });
  it('sharedGroupTable needs every frame to give the same table', () => {
    const a = groupTable(roots(4));
    const b = groupTable(roots(4, c(1), [0, 2, 1, 3]));
    expect(sharedGroupTable([a, groupTable(roots(4, c(2, 1)))])).toEqual(a);
    expect(sharedGroupTable([a, b])).toBeNull();
    expect(sharedGroupTable([a, null])).toBeNull();
  });
});
