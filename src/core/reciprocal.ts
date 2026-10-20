// Reciprocal-symmetry enumeration: the reversed coefficient string a_d..a_0 is z^d p(1/z), whose roots are 1/z, so
// one solve can deposit the roots of both. Exact cover: every polynomial's roots are deposited exactly once.
//
// The set of strings is cut into regions, each a mixed-radix product of slots (a digit position, or a pair of
// positions i and d-i), so the GPU decodes a sample index slot by slot like the plain base-k index. Reversal T acts
// slot-wise by an involution P on each slot's values (plain: (x, y) -> (y, x); pinned sets: a twisted map, below).
// With layer slots 0..L-1 (ordered), region R_j (j < L) = layers 0..j-1 fixed by P, layer j a representative of a
// 2-cycle of P, the rest free: it deposits z and the partner T(s), which has layer j = P(rep), a non-representative,
// so T(s) lies in no region. Every string outside the regions has a first non-fixed layer holding a non-
// representative, so it is T of exactly one string of some R_j. R_L (all layers fixed, rest free) deposits z only.
//
// Pinned sets (c0 * G, leading digit pinned to the identity 0): T(s) = reverse(s) * r^-1 with r = s_0, so the class of
// strings with s_0 = r maps onto the class r^-1. Classes with r < r^-1 (digit index) are one free region that
// deposits the partner; classes with r = r^-1 are layered with the pair map (x, y) -> (y r, x r) (and x -> x r on the
// middle digit), an involution because r^2 = 1.
import type { Complex } from './complex';
import { MAX_DEGREE, goldenStrideMixed, groupTable } from './enumeration';

/** reciprocal: half the solves (deposit z and 1/z); plain: every polynomial solved (the original path). */
export type Enumeration = 'reciprocal' | 'plain';

export const MAX_SLOTS = MAX_DEGREE + 1; // = MAX_SLOTS in roots.wgsl / coop.wgsl
export const MAX_REGIONS = 32; // = MAX_REGIONS in the shaders
export const MAX_TABLE = 2048; // u32 table entries
/** Stop adding layers once the deposit-z-only region R_L holds at most this fraction of the strings. */
const LAYER_TARGET = 1 / 32;

/** One mixed-radix digit of a region: value v in [0, radix) writes table[table + v] = x | y << 8 to digit
 * positions p (x) and q (y); single positions have p = q and x = y. */
export interface Slot {
  radix: number;
  table: number;
  p: number;
  q: number;
}

export interface Region {
  /** Low slot first (it varies fastest in lexicographic order). */
  slots: Slot[];
  /** Product of the radices: the number of solves. */
  size: bigint;
  /** Also deposit the partner T(s): roots 1/z, plus d - hi roots at 0 (its trimmed leading zeros). */
  partner: boolean;
  /** Table offset of the partner digit map: partner digit i = table[pmap + s_(d-i)] (x -> x r^-1; identity unpinned). */
  pmap: number;
}

export interface EnumPlan {
  /** Effective mode: a pinned set without a valid group table falls back to plain. */
  mode: Enumeration;
  k: number;
  degree: number;
  pinned: boolean;
  regions: Region[];
  tables: number[];
  /** Polynomials covered: k^(d+1), or k^d pinned. */
  polys: bigint;
  /** Solves: the sum of the region sizes. */
  solves: bigint;
  maxRadix: number;
}

const pack = (x: number, y: number) => x | (y << 8);
const product = (slots: Slot[]) => slots.reduce((s, sl) => s * BigInt(sl.radix), 1n);

/** A slot position with the involution's value classes, as packed table entries. */
interface SlotSpec {
  p: number;
  q: number;
  free: number[];
  fixed: number[];
  rep: number[];
}

class Tables {
  readonly data: number[] = [];
  private readonly seen = new Map<string, number>();
  add(entries: number[]): number {
    const key = entries.join(',');
    let at = this.seen.get(key);
    if (at === undefined) {
      at = this.data.length;
      this.data.push(...entries);
      this.seen.set(key, at);
      if (this.data.length > MAX_TABLE) throw new RangeError('enumeration tables overflow MAX_TABLE');
    }
    return at;
  }
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Single digit position p under the digit involution f (identity: always fixed, never a layer). */
function singleSpec(p: number, k: number, f: (x: number) => number): SlotSpec {
  const all = range(k);
  return {
    p,
    q: p,
    free: all.map((x) => pack(x, x)),
    fixed: all.filter((x) => f(x) === x).map((x) => pack(x, x)),
    rep: all.filter((x) => x < f(x)).map((x) => pack(x, x)),
  };
}

/** Positions (p, q) under P(x, y) = (g(y), g(x)) for an involution g on digits: fixed iff y = g(x). */
function pairSpec(p: number, q: number, k: number, g: (x: number) => number): SlotSpec {
  const free: number[] = [];
  const fixed: number[] = [];
  const rep: number[] = [];
  for (let v = 0; v < k * k; v++) {
    const x = v % k;
    const y = Math.floor(v / k);
    free.push(pack(x, y));
    const image = g(y) + k * g(x);
    if (image === v) fixed.push(pack(x, y));
    else if (v < image) rep.push(pack(x, y));
  }
  return { p, q, free, fixed, rep };
}

/** Layered regions R_0..R_L over `specs` (plus constant slots in every region), partner digit map `pmap`. */
function layered(specs: SlotSpec[], constant: Slot[], pmap: number, tables: Tables): Region[] {
  const order = specs
    .map((s, i) => ({ s, i, frac: s.fixed.length / s.free.length }))
    .filter((o) => o.s.rep.length > 0)
    .sort((a, b) => a.frac - b.frac || a.i - b.i);
  const layers: number[] = [];
  let frac = 1;
  for (const o of order) {
    if (frac <= LAYER_TARGET) break;
    layers.push(o.i);
    frac *= o.frac;
  }
  const regions: Region[] = [];
  for (let j = 0; j <= layers.length; j++) {
    const slots: Slot[] = [...constant];
    specs.forEach((s, i) => {
      const l = layers.indexOf(i);
      const entries = l < 0 || l > j ? s.free : l < j || j === layers.length ? s.fixed : s.rep;
      slots.push({ radix: entries.length, table: tables.add(entries), p: s.p, q: s.q });
    });
    const size = product(slots);
    if (size > 0n) regions.push({ slots, size, partner: j < layers.length, pmap });
  }
  return regions;
}

/**
 * The regions that enumerate k-coefficient strings of degree `degree` (pinned: leading digit = 0, the group
 * identity). reciprocal with pinned needs `group` (enumeration groupTable), else the plan falls back to plain.
 */
export function enumerationPlan(k: number, degree: number, pinned: boolean, requested: Enumeration, group: number[][] | null): EnumPlan {
  const tables = new Tables();
  const d = degree;
  const identity = tables.add(range(k));
  const single = (p: number): SlotSpec => singleSpec(p, k, (x) => x);
  const freeSlot = (s: SlotSpec): Slot => ({ radix: s.free.length, table: tables.add(s.free), p: s.p, q: s.q });
  const mode: Enumeration = requested === 'reciprocal' && (!pinned || group !== null) ? 'reciprocal' : 'plain';
  let regions: Region[];
  if (mode === 'plain') {
    // Slot i = digit i, radix k: the region index is the base-k polynomial index of the original path.
    const slots = range(pinned ? d : d + 1).map((i) => freeSlot(single(i)));
    if (pinned) slots.push({ radix: 1, table: tables.add([pack(0, 0)]), p: d, q: d });
    regions = [{ slots, size: product(slots), partner: false, pmap: identity }];
  } else if (!pinned) {
    const specs: SlotSpec[] = [];
    for (let i = 0; 2 * i < d; i++) specs.push(pairSpec(i, d - i, k, (x) => x));
    if (d % 2 === 0) specs.push(single(d / 2));
    regions = layered(specs, [], identity, tables);
  } else {
    const mul = group!;
    const inv = range(k).map((x) => mul[x].indexOf(0));
    regions = [];
    for (let r = 0; r < k; r++) {
      const constant: Slot[] = [{ radix: 1, table: tables.add([pack(r, 0)]), p: 0, q: d }];
      const pmap = tables.add(range(k).map((x) => mul[x][inv[r]]));
      if (r < inv[r]) {
        const slots = [...constant, ...range(d - 1).map((i) => freeSlot(single(i + 1)))];
        regions.push({ slots, size: product(slots), partner: true, pmap });
      } else if (r === inv[r]) {
        const times = (x: number) => mul[x][r];
        const specs: SlotSpec[] = [];
        for (let i = 1; 2 * i < d; i++) specs.push(pairSpec(i, d - i, k, times));
        if (d % 2 === 0 && d >= 2) specs.push(singleSpec(d / 2, k, times));
        regions.push(...layered(specs, constant, pmap, tables));
      }
    }
  }
  if (regions.length > MAX_REGIONS) throw new RangeError(`${regions.length} enumeration regions exceed MAX_REGIONS`);
  return {
    mode,
    k,
    degree,
    pinned,
    regions,
    tables: tables.data,
    polys: BigInt(k) ** BigInt(pinned ? d : d + 1),
    solves: regions.reduce((s, r) => s + r.size, 0n),
    maxRadix: Math.max(...regions.flatMap((r) => r.slots.map((s) => s.radix))),
  };
}

/** The group table shared by every frame (animation: pinned + reciprocal only then), or null. */
export function sharedGroupTable(tables: (number[][] | null)[]): number[][] | null {
  if (tables.length === 0 || tables.some((t) => t === null)) return null;
  const key = JSON.stringify(tables[0]);
  return tables.every((t) => JSON.stringify(t) === key) ? tables[0] : null;
}

/** The enumeration a render of `frames` (one coefficient set, or an animation's) actually uses, and the group table
 * a pinned reciprocal plan needs. Decided once for all frames, as pinning is. */
export function effectiveEnumeration(frames: Complex[][], pinned: boolean, requested: Enumeration): { mode: Enumeration; group: number[][] | null } {
  if (requested === 'plain') return { mode: 'plain', group: null };
  if (!pinned) return { mode: 'reciprocal', group: null };
  const group = sharedGroupTable(frames.map(groupTable));
  return { mode: group ? 'reciprocal' : 'plain', group };
}

/** Little-endian mixed-radix digits of x (taken mod the product of the radices). */
export function mixedDigits(x: bigint, radices: number[]): number[] {
  return radices.map((r) => {
    const R = BigInt(r);
    const digit = Number(x % R);
    x /= R;
    return digit;
  });
}

/**
 * Solves per region for a chunk of `count` solves so that every region stays at about the same done/size fraction
 * (each chunk is then a proportional sample of all regions, so any prefix still looks like the full picture).
 * Targets floor(after * size_r / total); the remainder goes to the regions furthest behind, within capacity.
 */
export function chunkCounts(sizes: bigint[], done: bigint[], count: number): number[] {
  const total = sizes.reduce((a, b) => a + b, 0n);
  const after = done.reduce((a, b) => a + b, 0n) + BigInt(count);
  const cap = sizes.map((s, r) => s - done[r]);
  const counts = sizes.map((s, r) => {
    const want = (after * s) / total - done[r];
    return Number(want <= 0n ? 0n : want < cap[r] ? want : cap[r]);
  });
  // How far region r would trail (> 0) or lead its proportional share after this chunk, times total.
  const behind = (r: number) => after * sizes[r] - (done[r] + BigInt(counts[r])) * total;
  let left = count - counts.reduce((a, b) => a + b, 0);
  const byBehind = () => sizes.map((_, r) => r).sort((a, b) => { const x = behind(b) - behind(a); return x > 0n ? 1 : x < 0n ? -1 : a - b; });
  if (left < 0) {
    for (const r of byBehind().reverse()) {
      const take = Math.min(counts[r], -left);
      counts[r] -= take;
      left += take;
      if (left === 0) break;
    }
  }
  for (const r of byBehind()) {
    if (left === 0) break;
    const room = cap[r] - BigInt(counts[r]);
    const give = room < BigInt(left) ? Number(room) : left;
    counts[r] += give;
    left -= give;
  }
  return counts;
}

/** One region's share of a dispatch: threads [threadStart, threadStart + count) take samples done..done+count-1. */
export interface ChunkEntry {
  region: number;
  threadStart: number;
  count: number;
  /** Mixed-radix digits of the first sample's index ((done * G) mod size) and of the stride G. */
  start: number[];
  stride: number[];
}

/** Progress through a plan: which samples of each region the next chunk takes. */
export class EnumCursor {
  readonly done: bigint[];
  solved = 0n;
  covered = 0n;
  private readonly strides: bigint[];

  constructor(readonly plan: EnumPlan, golden: boolean) {
    this.done = plan.regions.map(() => 0n);
    this.strides = plan.regions.map((r) => (golden ? goldenStrideMixed(r.size) : 1n));
  }

  /** The next chunk of at most `budget` solves (empty when finished); advances the cursor. */
  next(budget: number): ChunkEntry[] {
    const left = this.plan.solves - this.solved;
    const count = left <= 0n ? 0 : left < BigInt(budget) ? Number(left) : Math.floor(budget);
    if (count <= 0) return [];
    const counts = chunkCounts(this.plan.regions.map((r) => r.size), this.done, count);
    const entries: ChunkEntry[] = [];
    let threadStart = 0;
    counts.forEach((c, i) => {
      if (c === 0) return;
      const region = this.plan.regions[i];
      const radices = region.slots.map((s) => s.radix);
      const G = this.strides[i];
      entries.push({ region: i, threadStart, count: c, start: mixedDigits((this.done[i] * G) % region.size, radices), stride: mixedDigits(G, radices) });
      threadStart += c;
      this.done[i] += BigInt(c);
      this.covered += BigInt(c) * (region.partner ? 2n : 1n);
    });
    this.solved += BigInt(count);
    return entries;
  }
}

/** CPU mirror of the shaders' decode: digit string (d + 1 digits) of thread `local` of a chunk entry. */
export function decodeThread(plan: EnumPlan, e: ChunkEntry, local: number): number[] {
  const out = new Array<number>(plan.degree + 1).fill(0);
  let carry = 0;
  plan.regions[e.region].slots.forEach((s, i) => {
    const v = e.start[i] + e.stride[i] * local + carry;
    const x = v % s.radix;
    carry = Math.floor(v / s.radix);
    const t = plan.tables[s.table + x];
    out[s.p] = t & 0xff;
    out[s.q] = t >> 8;
  });
  return out;
}

/** Digits of the partner T(s) of a string of region `region`: reversed, then each mapped by the region's pmap. */
export function partnerDigits(plan: EnumPlan, region: number, digits: number[]): number[] {
  const pmap = plan.regions[region].pmap;
  return digits.map((_, i) => plan.tables[pmap + digits[plan.degree - i]]);
}

// GPU layout of the enumeration buffer (u32), = the constants in roots.wgsl / coop.wgsl:
// [0, MAX_REGIONS * ENTRY_U32): chunk entries {threadStart, region, 0, 0, start[MAX_SLOTS], stride[MAX_SLOTS]};
// REGION_BASE: per region {slot count, partner, pmap, 0, then per slot radix, table, p, q};
// TABLE_BASE: the tables.
export const ENTRY_U32 = 4 + 2 * MAX_SLOTS;
export const REGION_U32 = 4 + 4 * MAX_SLOTS;
export const REGION_BASE = MAX_REGIONS * ENTRY_U32;
export const TABLE_BASE = REGION_BASE + MAX_REGIONS * REGION_U32;
export const ENUM_U32 = TABLE_BASE + MAX_TABLE;

/** The plan's static part (regions and tables), written at REGION_BASE. */
export function packPlan(plan: EnumPlan): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(ENUM_U32 - REGION_BASE);
  plan.regions.forEach((r, i) => {
    const b = i * REGION_U32;
    out.set([r.slots.length, r.partner ? 1 : 0, r.pmap, 0], b);
    r.slots.forEach((s, j) => out.set([s.radix, s.table, s.p, s.q], b + 4 + 4 * j));
  });
  out.set(plan.tables, TABLE_BASE - REGION_BASE);
  return out;
}

/** A chunk's entries, written at 0. */
export function packChunk(entries: ChunkEntry[]): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(Math.max(1, entries.length) * ENTRY_U32);
  entries.forEach((e, i) => {
    const b = i * ENTRY_U32;
    out.set([e.threadStart, e.region], b);
    out.set(e.start, b + 4);
    out.set(e.stride, b + 4 + MAX_SLOTS);
  });
  return out;
}
