import type { Complex } from './complex';

const EPS = 1e-6;
const abs = (a: Complex) => Math.hypot(a.re, a.im);
const mul = (a: Complex, b: Complex): Complex => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const div = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};
const near = (a: Complex, b: Complex) => abs({ re: a.re - b.re, im: a.im - b.im }) <= EPS * (1 + abs(b));

/**
 * True when coeffs = c0 * G for a finite multiplicative group G (e.g. {±1}, {1, i, -1, -i}).
 * Then every polynomial is a unit multiple of one whose leading coefficient is coeffs[0],
 * so the leading digit can be pinned to 0: same root set, k times less work.
 */
export function leadingFixable(coeffs: Complex[]): boolean {
  if (coeffs.length < 2 || coeffs.some((c) => abs(c) < EPS)) return false;
  const c0 = coeffs[0];
  return coeffs.every((cj) => {
    const g = div(cj, c0);
    return coeffs.every((ci) => coeffs.some((cl) => near(mul(g, ci), cl)));
  });
}

/** Number of polynomials to enumerate: k^(d+1), or k^d with the leading digit pinned. */
export function totalPolys(k: number, degree: number, fixable: boolean): bigint {
  return BigInt(k) ** BigInt(fixable ? degree : degree + 1);
}

const GOLDEN_FIX52 = 2783565922149187n; // round(0.6180339887498949 * 2^52)
const gcd = (a: bigint, b: bigint): bigint => (b === 0n ? a : gcd(b, a % b));

/**
 * Enumeration stride for N = k^D polynomials: about 0.618 N, bumped until coprime to k (hence to N).
 * Sample m is index m*G mod N, a bijection of [0, N). Any prefix is spread evenly over the whole set (a Weyl
 * sequence in the high digits), and since G is invertible mod k^j, each block of k^j consecutive samples hits
 * every pattern of the low j digits exactly once. A small fixed stride instead freezes low digits when it
 * shares a factor with k, and leaves lattice artifacts even when it doesn't (measured in the plan, §1.1).
 */
export function goldenStride(total: bigint, k: number): bigint {
  if (total <= 2n) return 1n;
  const K = BigInt(k);
  let g = (total * GOLDEN_FIX52) >> 52n;
  while (gcd(g, K) !== 1n) g++;
  return g < total ? g : 1n;
}

/** Polynomial index of sample m: (m * G) mod N. */
export function sampleIndex(m: bigint, stride: bigint, total: bigint): bigint {
  return (m * stride) % total;
}

/** Little-endian base-k digits (d+1 of them) of a polynomial index; digit i selects coefficient a_i. */
export function indexToDigits(index: bigint, k: number, degree: number): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(degree + 1);
  const K = BigInt(k);
  let x = index;
  for (let i = 0; i <= degree; i++) {
    out[i] = Number(x % K);
    x /= K;
  }
  return out;
}
