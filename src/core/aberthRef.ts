import type { Complex } from './complex';

const ONE: Complex = { re: 1, im: 0 };
const add = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
const sub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
const mul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const div = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};
const mag = (a: Complex) => Math.hypot(a.re, a.im);

/**
 * f64 Ehrlich–Aberth, tests only. a[i] is the coefficient of z^i. Zero leading coefficients are
 * dropped; k zero low coefficients give k roots at 0. Same initial guesses as roots.wgsl.
 */
export function aberthRef(a: Complex[], maxIter = 500): Complex[] {
  let hi = a.length - 1;
  while (hi > 0 && mag(a[hi]) < 1e-12) hi--;
  let lo = 0;
  while (lo < hi && mag(a[lo]) < 1e-12) lo++;
  const zeros: Complex[] = Array.from({ length: lo }, () => ({ re: 0, im: 0 }));
  const c = a.slice(lo, hi + 1);
  const n = c.length - 1;
  if (n === 0) return zeros;
  const r = Math.pow(mag(c[0]) / mag(c[n]), 1 / n);
  const z = Array.from({ length: n }, (_, j) => {
    const t = (2 * Math.PI * j) / n + 0.4;
    return { re: r * Math.cos(t), im: r * Math.sin(t) };
  });
  for (let it = 0; it < maxIter; it++) {
    let maxStep = 0;
    for (let j = 0; j < n; j++) {
      let p = c[n];
      let dp: Complex = { re: 0, im: 0 };
      for (let i = n; i > 0; i--) {
        dp = add(mul(dp, z[j]), p);
        p = add(mul(p, z[j]), c[i - 1]);
      }
      if (mag(p) === 0) continue;
      const ratio = div(p, dp);
      let s: Complex = { re: 0, im: 0 };
      for (let q = 0; q < n; q++) if (q !== j) s = add(s, div(ONE, sub(z[j], z[q])));
      const w = div(ratio, sub(ONE, mul(ratio, s)));
      z[j] = sub(z[j], w);
      maxStep = Math.max(maxStep, mag(w) / (1 + mag(z[j])));
    }
    if (maxStep < 1e-14) break;
  }
  return [...zeros, ...z];
}
