import { totalPolys } from './enumeration';

export const WORKGROUP = 64; // must equal WG in roots.wgsl
export const COOP_WORKGROUP = 256; // must equal WGS in coop.wgsl
export const MAX_WG_PER_DIM = 65535;
export const MIN_BUDGET = 1024;
export const MAX_BUDGET = 1 << 22;

/** 2-D dispatch for `count` invocations; the kernel computes tid = gid.x + gid.y * strideX. */
export function dispatchGrid(count: number): { x: number; y: number; strideX: number } {
  if (count <= 0) return { x: 0, y: 0, strideX: 0 };
  const groups = Math.ceil(count / WORKGROUP);
  const x = Math.min(groups, MAX_WG_PER_DIM);
  return { x, y: Math.ceil(groups / x), strideX: x * WORKGROUP };
}

/** Scale the per-frame polynomial budget toward `targetMs` of GPU time (at most 2x per step). */
export function nextBudget(budget: number, elapsedMs: number, targetMs: number): number {
  const f = Math.min(2, Math.max(0.5, targetMs / Math.max(elapsedMs, 0.1)));
  return Math.round(Math.min(MAX_BUDGET, Math.max(MIN_BUDGET, budget * f)));
}

/** Highest degree <= `degree` (but >= 1) whose whole polynomial set fits in one frame's budget. */
export function previewDegree(k: number, degree: number, fixable: boolean, budget: number): number {
  let d = degree;
  while (d > 1 && totalPolys(k, d, fixable) > BigInt(Math.floor(budget))) d--;
  return d;
}

export type Kernel = 'perPoly' | 'coop';

/** Cooperative kernel: one thread per root, floor(256/d) polynomials per workgroup, 2-D wrap like dispatchGrid. */
export function coopGrid(count: number, degree: number): { x: number; y: number; polysPerGroup: number } {
  const polysPerGroup = Math.floor(COOP_WORKGROUP / degree);
  if (count <= 0) return { x: 0, y: 0, polysPerGroup };
  const groups = Math.ceil(count / polysPerGroup);
  const x = Math.min(groups, MAX_WG_PER_DIM);
  return { x, y: Math.ceil(groups / x), polysPerGroup };
}

/** Cooperative only on a clear (>10 %) win: per-polynomial is the default and timing noise must not flip it. */
export function chooseKernel(perPolyMs: number, coopMs: number): Kernel {
  return coopMs < 0.9 * perPolyMs ? 'coop' : 'perPoly';
}
