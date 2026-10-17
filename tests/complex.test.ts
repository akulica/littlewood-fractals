import { describe, expect, it } from 'vitest';
import { formatComplex, parseCoefficientList, parseComplex } from '../src/core/complex';

describe('parseComplex', () => {
  it.each([
    ['1', 1, 0],
    ['-1', -1, 0],
    ['0.5+0.3i', 0.5, 0.3],
    ['-i', 0, -1],
    ['2.5i', 0, 2.5],
    ['1e-3-2i', 0.001, -2],
    [' 1 - i ', 1, -1],
    ['.5', 0.5, 0],
  ])('%s', (text, re, im) => {
    expect(parseComplex(text)).toEqual({ re, im });
  });
  it.each(['abc', '1+', '', 'i2'])('rejects %j', (text) => {
    expect(parseComplex(text)).toBeNull();
  });
});

describe('parseCoefficientList', () => {
  it('splits on commas and semicolons', () => {
    expect(parseCoefficientList('1, -1; i')).toEqual([{ re: 1, im: 0 }, { re: -1, im: 0 }, { re: 0, im: 1 }]);
  });
  it('rejects the whole list if one entry is bad', () => {
    expect(parseCoefficientList('1, x')).toBeNull();
    expect(parseCoefficientList('  ')).toBeNull();
  });
});

describe('formatComplex', () => {
  it('round-trips through parseComplex at 3 decimals', () => {
    for (const c of [{ re: 1, im: 0 }, { re: 0, im: -1 }, { re: 0.25, im: 0.125 }, { re: -0.5, im: 2 }]) {
      expect(parseComplex(formatComplex(c))).toEqual({ re: +c.re.toFixed(3), im: +c.im.toFixed(3) });
    }
  });
  it('prints compact forms', () => {
    expect(formatComplex({ re: 1, im: 0 })).toBe('1');
    expect(formatComplex({ re: 0, im: -1 })).toBe('-1i');
    expect(formatComplex({ re: 0.5, im: -0.3 })).toBe('0.5-0.3i');
  });
});
