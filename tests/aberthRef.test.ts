import { describe, expect, it } from 'vitest';
import { aberthRef } from '../src/core/aberthRef';
import { maxRootError } from './helpers';

const r = (...xs: number[]) => xs.map((re) => ({ re, im: 0 }));

describe('aberthRef', () => {
  it('z^2 - 1', () => {
    expect(maxRootError(aberthRef(r(-1, 0, 1)), r(1, -1))).toBeLessThan(1e-12);
  });
  it('(z-1)(z-2)(z-3)', () => {
    expect(maxRootError(aberthRef(r(-6, 11, -6, 1)), r(1, 2, 3))).toBeLessThan(1e-10);
  });
  it('z + z^2 has roots 0 and -1 (zero constant term)', () => {
    expect(maxRootError(aberthRef(r(0, 1, 1)), r(0, -1))).toBeLessThan(1e-12);
  });
  it('drops zero leading coefficients', () => {
    expect(aberthRef(r(0, 0, 1, 0))).toEqual([{ re: 0, im: 0 }, { re: 0, im: 0 }]);
  });
  it('z^2 + 1 (complex roots)', () => {
    expect(maxRootError(aberthRef(r(1, 0, 1)), [{ re: 0, im: 1 }, { re: 0, im: -1 }])).toBeLessThan(1e-12);
  });
});
