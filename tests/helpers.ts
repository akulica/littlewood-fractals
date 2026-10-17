import type { Complex } from '../src/core/complex';

/** Greedy nearest-neighbour pairing of two root lists; returns the worst paired distance. */
export function maxRootError(a: Complex[], b: Complex[]): number {
  if (a.length !== b.length) return Infinity;
  const used = new Array<boolean>(b.length).fill(false);
  let worst = 0;
  for (const x of a) {
    let best = Infinity;
    let bi = -1;
    b.forEach((y, i) => {
      if (used[i]) return;
      const e = Math.hypot(x.re - y.re, x.im - y.im);
      if (e < best) { best = e; bi = i; }
    });
    used[bi] = true;
    worst = Math.max(worst, best);
  }
  return worst;
}
