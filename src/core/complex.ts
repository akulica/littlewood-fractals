export interface Complex {
  re: number;
  im: number;
}

const UNSIGNED = String.raw`(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?`;
const REAL = new RegExp(`^([+-]?${UNSIGNED})$`);
const IMAG = new RegExp(`^([+-]?)(${UNSIGNED})?i$`);
const BOTH = new RegExp(`^([+-]?${UNSIGNED})([+-])(${UNSIGNED})?i$`);

/** Parses "1", "-0.5", "2i", "-i", "0.5+0.3i", "1e-3-2i" (whitespace ignored). Returns null if invalid. */
export function parseComplex(text: string): Complex | null {
  const s = text.replace(/\s+/g, '').toLowerCase();
  let m = s.match(REAL);
  if (m) return { re: parseFloat(m[1]), im: 0 };
  m = s.match(IMAG);
  if (m) return { re: 0, im: (m[1] === '-' ? -1 : 1) * (m[2] ? parseFloat(m[2]) : 1) };
  m = s.match(BOTH);
  if (m) return { re: parseFloat(m[1]), im: (m[2] === '-' ? -1 : 1) * (m[3] ? parseFloat(m[3]) : 1) };
  return null;
}

/** "1, -1; 0.5+i" -> list, or null if any entry is invalid or the list is empty. */
export function parseCoefficientList(text: string): Complex[] | null {
  const parts = text.split(/[,;]/).map((p) => p.trim()).filter((p) => p.length > 0);
  if (parts.length === 0) return null;
  const out: Complex[] = [];
  for (const p of parts) {
    const c = parseComplex(p);
    if (!c) return null;
    out.push(c);
  }
  return out;
}

/** Compact 3-decimal form: "1", "-1i", "0.5-0.3i". */
export function formatComplex(c: Complex): string {
  const re = +c.re.toFixed(3);
  const im = +c.im.toFixed(3);
  if (im === 0) return `${re}`;
  if (re === 0) return `${im}i`;
  return `${re}${im < 0 ? '-' : '+'}${Math.abs(im)}i`;
}
