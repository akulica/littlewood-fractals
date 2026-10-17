import { describe, expect, it } from 'vitest';
import { chooseKernel, coopGrid, dispatchGrid, nextBudget, previewDegree } from '../src/core/schedule';

describe('dispatchGrid', () => {
  it('uses one row when it fits', () => {
    expect(dispatchGrid(1)).toEqual({ x: 1, y: 1, strideX: 64 });
    expect(dispatchGrid(640)).toEqual({ x: 10, y: 1, strideX: 640 });
  });
  it('wraps into a second row past 65535 workgroups', () => {
    expect(dispatchGrid(65535 * 64 + 1)).toEqual({ x: 65535, y: 2, strideX: 65535 * 64 });
  });
  it('handles zero', () => {
    expect(dispatchGrid(0)).toEqual({ x: 0, y: 0, strideX: 0 });
  });
});

describe('nextBudget', () => {
  it('grows at most 2x when the frame was fast, shrinks at most 2x when slow', () => {
    expect(nextBudget(10_000, 1, 12)).toBe(20_000);
    expect(nextBudget(10_000, 100, 12)).toBe(5_000);
    expect(nextBudget(10_000, 12, 12)).toBe(10_000);
  });
  it('clamps to [1024, 4M]', () => {
    expect(nextBudget(1024, 1000, 12)).toBe(1024);
    expect(nextBudget(1 << 22, 0.01, 12)).toBe(1 << 22);
  });
});

describe('previewDegree', () => {
  it('picks the largest degree whose polynomial count fits the budget', () => {
    expect(previewDegree(2, 24, true, 70_000)).toBe(16);
    expect(previewDegree(3, 10, false, 1000)).toBe(5);
    expect(previewDegree(2, 5, true, 1e6)).toBe(5);
    expect(previewDegree(16, 30, false, 10)).toBe(1);
  });
});

describe('coopGrid', () => {
  it('packs floor(256/d) polynomials per workgroup', () => {
    expect(coopGrid(100, 20)).toEqual({ x: 9, y: 1, polysPerGroup: 12 });
    expect(coopGrid(256, 1)).toEqual({ x: 1, y: 1, polysPerGroup: 256 });
    expect(coopGrid(0, 20)).toEqual({ x: 0, y: 0, polysPerGroup: 12 });
  });
  it('wraps into a second row past 65535 workgroups', () => {
    expect(coopGrid(65536 * 8, 32)).toEqual({ x: 65535, y: 2, polysPerGroup: 8 });
  });
});

describe('chooseKernel', () => {
  it('switches to the cooperative kernel only on a >10 % win', () => {
    expect(chooseKernel(100, 66)).toBe('coop');
    expect(chooseKernel(100, 95)).toBe('perPoly');
    expect(chooseKernel(100, 130)).toBe('perPoly');
  });
});
