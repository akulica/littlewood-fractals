import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { encodePng, readPngTexts, type PngText } from '../src/core/png';
import { PngPool, defaultPngWorkers, transferable, type PngEncoder, type PngJob } from '../src/core/pngPool';
import { spawnNodePngEncoder } from '../src/core/pngWorkerNode';

const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);
const bytes = async (b: Blob) => new Uint8Array(await b.arrayBuffer());

function pixels(w: number, h: number, seed: number): Uint8Array<ArrayBuffer> {
  const a = new Uint8Array(w * h * 4);
  let s = seed >>> 0 || 1;
  for (let i = 0; i < a.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    // smooth gradient plus noise on a third of the pixels: compressible, but not trivially
    a[i] = i % 4 === 3 ? 255 - (s >>> 28) : ((i >> 2) % 251) + (i % 3 === 0 ? s >>> 27 : 0);
  }
  return a;
}

const TEXTS: PngText[] = [
  { keyword: 'Title', text: 'Littlewood fractal d=16 · frame 7' },
  { keyword: 'Comment', text: JSON.stringify({ schema: 'littlewood-fractals/1', kind: 'frame', note: 'π ≈ 3.14159 — ünïcödé ✓' }), utf8: true },
  { keyword: 'Latin', text: 'café 中' }, // tEXt: the non-Latin-1 character becomes '?'
];

/** Encoders that run the job through a function the test controls, to observe the pool's scheduling. */
function fakePool<T = number>(workers: number, work: (job: PngJob, worker: number) => Promise<Uint8Array<ArrayBuffer>>, deliver: (png: Uint8Array<ArrayBuffer>, tag: T) => Promise<void> | void) {
  const stats = { running: 0, maxRunning: 0, terminated: 0 };
  let n = 0;
  const pool = new PngPool<T>(
    () => {
      const id = n++;
      const e: PngEncoder = {
        async encode(job) {
          stats.maxRunning = Math.max(stats.maxRunning, ++stats.running);
          try {
            return await work(job, id);
          } finally {
            stats.running--;
          }
        },
        terminate: () => void stats.terminated++,
      };
      return e;
    },
    workers,
    deliver,
  );
  return { pool, stats };
}

const job = (tag: number): PngJob => ({ rgba: new Uint8Array(4), width: 1, height: 1, texts: [{ keyword: 'n', text: String(tag) }] });
const tagOf = (j: PngJob) => Number(j.texts[0].text);
const png = (tag: number) => new Uint8Array([tag]);
const tick = () => new Promise<void>((r) => setTimeout(r, 0));
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

describe('defaultPngWorkers', () => {
  it('is min(4, max(1, cores - 2))', () => {
    expect([1, 2, 3, 4, 6, 8, 16, 64].map(defaultPngWorkers)).toEqual([1, 1, 1, 2, 4, 4, 4, 4]);
    expect(defaultPngWorkers(NaN)).toBe(1);
  });
});

describe('transferable', () => {
  it('gives up a buffer the array spans entirely and copies a view onto a larger one', () => {
    const whole = new Uint8Array(8);
    expect(transferable(whole)).toBe(whole.buffer);
    const big = new Uint8Array(16).fill(5);
    const view = big.subarray(4, 8);
    const t = transferable(view);
    expect(t).not.toBe(big.buffer);
    expect([...new Uint8Array(t)]).toEqual([5, 5, 5, 5]);
    expect(transferable(new Uint8ClampedArray(4))).toBeInstanceOf(ArrayBuffer);
  });
});

describe('PngPool scheduling (controlled encoders)', () => {
  it('delivers in submission order even when later frames finish first', async () => {
    const delivered: number[] = [];
    // frame k takes (8 - k) * 8 ms: with 4 workers the last of each group finishes first
    const { pool } = fakePool(4, async (j) => (await sleep((8 - tagOf(j)) * 8), png(tagOf(j))), (p, tag) => void delivered.push(p[0] * 100 + tag));
    for (let i = 0; i < 8; i++) await pool.submit(job(i), i);
    await pool.finish();
    expect(delivered).toEqual([0, 101, 202, 303, 404, 505, 606, 707]);
  });

  it('returns frame bytes to the right frame when completions are reversed', async () => {
    const order: number[] = [];
    const releases: (() => void)[] = [];
    const { pool } = fakePool(
      3,
      (j) => new Promise((r) => releases.push(() => r(png(tagOf(j))))),
      (p) => void order.push(p[0]),
    );
    for (let i = 0; i < 3; i++) await pool.submit(job(i), i);
    await tick();
    expect(releases).toHaveLength(3);
    releases[2](); // frame 2 first, then 1: nothing may be delivered before frame 0
    releases[1]();
    await tick();
    expect(order).toEqual([]);
    releases[0]();
    await pool.finish();
    expect(order).toEqual([0, 1, 2]);
  });

  it('bounds the frames in flight to 2x workers and makes the producer wait', async () => {
    const releases: (() => void)[] = [];
    const delivered: number[] = [];
    const { pool, stats } = fakePool(
      2,
      (j) => new Promise((r) => releases.push(() => r(png(tagOf(j))))),
      (p) => void delivered.push(p[0]),
    );
    expect(pool.maxInFlight).toBe(4);
    for (let i = 0; i < 4; i++) await pool.submit(job(i), i); // four slots: none waits
    let fifth = false;
    const blocked = pool.submit(job(4), 4).then(() => (fifth = true));
    await tick();
    expect(fifth).toBe(false);
    expect(pool.pending).toBe(4);
    expect(stats.running).toBe(2); // only two workers encode; two more frames queue behind them
    releases[0]();
    await blocked; // frame 0 delivered -> a slot frees
    expect(fifth).toBe(true);
    expect(delivered).toEqual([0]);
    // drain everything; pending never exceeded the bound
    let seen = 0;
    while (delivered.length < 5) {
      while (seen < releases.length) releases[seen++]();
      await tick();
    }
    await pool.finish();
    expect(delivered).toEqual([0, 1, 2, 3, 4]);
    expect(stats.maxRunning).toBe(2);
    expect(stats.terminated).toBe(2);
  });

  it('a slow deliver (disk) holds the slot and back-pressures the producer', async () => {
    let finishWrite!: () => void;
    const { pool } = fakePool(1, async (j) => png(tagOf(j)), () => new Promise<void>((r) => (finishWrite = r)));
    await pool.submit(job(0), 0);
    await pool.submit(job(1), 1); // 2 x 1 worker
    let third = false;
    const blocked = pool.submit(job(2), 2).then(() => (third = true));
    await sleep(20);
    expect(third).toBe(false);
    finishWrite();
    await blocked;
    for (let i = 0; i < 2; i++) {
      await tick();
      finishWrite();
    }
    await pool.finish();
  });

  it('propagates an encoding error from finish and to later submits, and stops delivering', async () => {
    const delivered: number[] = [];
    const { pool } = fakePool(
      2,
      async (j) => {
        if (tagOf(j) === 2) throw new Error('boom in frame 2');
        return png(tagOf(j));
      },
      (p) => void delivered.push(p[0]),
    );
    for (let i = 0; i < 4; i++) await pool.submit(job(i), i).catch(() => {});
    await expect(pool.finish()).rejects.toThrow('boom in frame 2');
    expect(delivered).toEqual([0, 1]);
    await expect(pool.submit(job(9), 9)).rejects.toThrow('boom in frame 2');
  });

  it('wakes a producer blocked on a slot when a frame fails', async () => {
    const { pool } = fakePool(
      1,
      async (j) => {
        await sleep(10);
        if (tagOf(j) === 0) throw new Error('first frame failed');
        return png(tagOf(j));
      },
      () => {},
    );
    await pool.submit(job(0), 0);
    await pool.submit(job(1), 1);
    await expect(pool.submit(job(2), 2)).rejects.toThrow('first frame failed');
    await expect(pool.finish()).rejects.toThrow('first frame failed');
  });

  it('propagates a deliver (disk write) error', async () => {
    const { pool } = fakePool(2, async (j) => png(tagOf(j)), (_p, tag) => {
      if (tag === 1) throw new Error('disk full');
    });
    await pool.submit(job(0), 0);
    await pool.submit(job(1), 1);
    await expect(pool.finish()).rejects.toThrow('disk full');
  });

  it('terminates already created encoders when one cannot be spawned, and rejects 0 workers', () => {
    let terminated = 0;
    let n = 0;
    expect(
      () =>
        new PngPool(
          () => {
            if (n++ === 2) throw new Error('no more workers');
            return { encode: async () => png(0), terminate: () => void terminated++ };
          },
          4,
          () => {},
        ),
    ).toThrow('no more workers');
    expect(terminated).toBe(2);
    expect(() => new PngPool(() => ({ encode: async () => png(0), terminate() {} }), 0, () => {})).toThrow(/at least one worker/);
  });
});

describe('PngPool on real worker threads', () => {
  let dir = '';
  let entry = '';
  beforeAll(async () => {
    // Bundle the worker the way the shipped script is bundled (vite SSR build), then run it as a worker_threads script.
    const out = await build({
      configFile: false,
      logLevel: 'silent',
      build: { ssr: fileURLToPath(new URL('./pngWorkerEntry.ts', import.meta.url)), write: false, minify: false, rollupOptions: { output: { format: 'es' } } },
    });
    const outputs = (Array.isArray(out) ? out : [out as { output: unknown[] }]).flatMap((r) => (r as { output: { type: string; code?: string }[] }).output);
    const code = outputs.filter((o) => o.type === 'chunk').map((o) => o.code).join(';\n');
    dir = mkdtempSync(join(tmpdir(), 'lwf-pngpool-'));
    entry = join(dir, 'worker.mjs');
    writeFileSync(entry, code);
  }, 60_000);
  afterAll(() => dir && rmSync(dir, { recursive: true, force: true }));

  it('produces files byte-identical to encodePng, in order, for several sizes with metadata', async () => {
    const sizes: [number, number][] = [[1, 1], [3, 2], [17, 9], [64, 48], [255, 3], [3, 255], [320, 180]];
    const frames = sizes.flatMap(([w, h], k) => [
      { w, h, rgba: pixels(w, h, k + 1), texts: TEXTS },
      { w, h, rgba: pixels(w, h, k + 100), texts: [] as PngText[] },
    ]);
    const expected = await Promise.all(frames.map(async (f) => bytes(await encodePng(f.rgba, f.w, f.h, f.texts))));
    const got: { tag: number; png: Uint8Array }[] = [];
    const pool = new PngPool<number>(() => spawnNodePngEncoder(entry), 3, (p, tag) => void got.push({ tag, png: p }));
    // give the pool its own copies: it takes the buffers over
    for (const [i, f] of frames.entries()) await pool.submit({ rgba: f.rgba.slice(), width: f.w, height: f.h, texts: f.texts }, i);
    await pool.finish();

    expect(got.map((g) => g.tag)).toEqual(frames.map((_, i) => i));
    for (const [i, g] of got.entries()) expect(same(g.png, expected[i]), `frame ${i} (${frames[i].w}x${frames[i].h})`).toBe(true);

    // metadata reads back with the independent parser
    const texts = readPngTexts(got[0].png);
    expect(texts).toEqual([
      { keyword: 'Title', text: TEXTS[0].text },
      { keyword: 'Comment', text: TEXTS[1].text, utf8: true },
      { keyword: 'Latin', text: 'café ?' },
    ]);
    expect(readPngTexts(got[1].png)).toEqual([]);
  });

  it('transfers the frame buffer (the producer cannot reuse it) but leaves a larger buffer alone', async () => {
    const got: Uint8Array[] = [];
    const pool = new PngPool<number>(() => spawnNodePngEncoder(entry), 1, (p) => void got.push(p));
    const owned = pixels(4, 4, 7);
    const big = pixels(8, 8, 8);
    const part = big.subarray(0, 64); // 4x4 RGBA out of a bigger buffer
    const expected = [await bytes(await encodePng(owned, 4, 4)), await bytes(await encodePng(part, 4, 4))];
    await pool.submit({ rgba: owned, width: 4, height: 4, texts: [] }, 0);
    await pool.submit({ rgba: part, width: 4, height: 4, texts: [] }, 1);
    await pool.finish();
    expect(owned.byteLength).toBe(0); // detached
    expect(big.byteLength).toBe(256); // untouched
    expect(same(got[0], expected[0])).toBe(true);
    expect(same(got[1], expected[1])).toBe(true);
  });

  it('propagates an error thrown in the worker (wrong buffer size) and keeps the pool usable for finish()', async () => {
    const pool = new PngPool<number>(() => spawnNodePngEncoder(entry), 2, () => {});
    await pool.submit({ rgba: new Uint8Array(10), width: 3, height: 2, texts: [] }, 0);
    await expect(pool.finish()).rejects.toThrow(/expected 24 bytes of RGBA, got 10/);
  });

  it('check() succeeds on a healthy worker and rejects when the script cannot run', async () => {
    const ok = new PngPool<number>(() => spawnNodePngEncoder(entry), 1, () => {});
    await ok.check();
    await ok.finish();
    const badEntry = join(dir, 'broken.mjs');
    writeFileSync(badEntry, 'throw new Error("worker script failed");');
    const bad = new PngPool<number>(() => spawnNodePngEncoder(badEntry), 1, () => {});
    await expect(bad.check()).rejects.toThrow(/worker script failed/);
    bad.terminate();
  });
});
