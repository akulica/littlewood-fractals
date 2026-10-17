import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { aberthRef } from '../../src/core/aberthRef';
import { indexToDigits } from '../../src/core/enumeration';
import { Engine } from '../../src/gpu/engine';
import { maxRootError } from '../helpers';
import { nodeDevice } from './gpuHelpers';

const { device, errors } = await nodeDevice();
const engine = new Engine(device, 'rgba8unorm');
afterAll(() => device.destroy());

const C3: Complex[] = [{ re: 1, im: 0 }, { re: -0.5, im: 0.8 }, { re: 0.3, im: -1.1 }];
const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];

/** Roots of polynomial t from solveDebug output; slots with re >= 1e37 are unused (dropped leading zeros). */
function gpuRoots(out: Float32Array, t: number, degree: number): Complex[] {
  const roots: Complex[] = [];
  for (let j = 0; j < degree; j++) {
    const re = out[(t * degree + j) * 2];
    if (re < 1e37) roots.push({ re, im: out[(t * degree + j) * 2 + 1] });
  }
  return roots;
}

async function worstError(coeffs: Complex[], degree: number, first: bigint, count: number): Promise<number> {
  const out = await engine.solveDebug(coeffs, degree, first, count);
  let worst = 0;
  for (let t = 0; t < count; t++) {
    const a = Array.from(indexToDigits(first + BigInt(t), coeffs.length, degree), (g) => coeffs[g]);
    const ref = aberthRef(a);
    const got = gpuRoots(out, t, degree);
    expect(got.length).toBe(ref.length);
    worst = Math.max(worst, maxRootError(got, ref));
  }
  return worst;
}

describe('roots.wgsl via Engine.solveDebug', () => {
  it('matches the f64 reference for a complex 3-coefficient set, degree 12', async () => {
    expect(await worstError(C3, 12, 123_456n, 500)).toBeLessThan(1e-4);
    expect(errors).toEqual([]);
  });
  it('matches the reference for Littlewood degree 20 (clustered and multiple roots)', async () => {
    expect(await worstError(LW, 20, 0n, 512)).toBeLessThan(1e-3);
  });
  it('trims zero coefficients: index 6 over {0, 1} at degree 10 is z + z^2', async () => {
    const out = await engine.solveDebug([{ re: 0, im: 0 }, { re: 1, im: 0 }], 10, 6n, 1);
    const got = gpuRoots(out, 0, 10);
    expect(got.length).toBe(2);
    expect(maxRootError(got, [{ re: 0, im: 0 }, { re: -1, im: 0 }])).toBeLessThan(1e-5);
  });
});
