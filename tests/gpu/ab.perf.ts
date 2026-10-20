import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { env, version as nodeVersion } from 'node:process';
import { afterAll, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import { leadingFixable } from '../../src/core/enumeration';
import { PNG_KEYWORD, buildMetadata, metadataTexts, parseMetadata, type BuildInfo, type Metadata } from '../../src/core/metadata';
import { encodePng, readPngTexts } from '../../src/core/png';
import type { Kernel } from '../../src/core/schedule';
import { pixelToWorld, type View } from '../../src/core/view';
import { Engine, RgbaReadback, TOL, type DisplayParams, type Enumeration } from '../../src/gpu/engine';
import { nodeDevice, runToEnd } from './gpuHelpers';

// A/B image harness for optimisations: `npm run ab`. Renders the scenes of tests/scenes/scenes.json at 800x500 under
// every variant (enumeration plain/reciprocal x kernel perPoly/coop), writes ab-out/<variant>/<scene>.png with the
// full reproduction metadata, and compares each variant with the baseline plain-perPoly (the original path):
//   px %     pixels whose RGB differs at all (tone-mapping is relative to the densest pixel, so a changed maximum
//            shifts every pixel slightly: a coarse indicator)
//   mean     mean |difference| per RGB channel, 0..255
//   centre   |lit pixels within |z| < 0.25| difference (where 1/z of wrong big roots lands: the "ring of dots" bug)
//   block    largest |lit pixels| difference in any 32x32 block (catches any localized artifact)
// and saves a heatmap ab-out/diff/<variant>/<scene>.png: red = lit only in the variant, cyan = lit only in the
// baseline, grey = lit in both but different (brighter = bigger difference).
//
// Cross-build: AB_REF_DIR=<another build's ab-out/plain-perPoly> also compares this build's baseline with those
// images (row "ref <commit>"); their metadata must describe the same scenes.
//
// Thresholds, calibrated on Intel Arc 140T (Dawn/D3D12) with the big-root fix: the worst of the 8 scenes x 3 variants,
// then 2x headroom or more. The yardstick is the f32 noise between the two plain kernels on the same polynomials.
//   measured     px % up to 38.7 (littlewood-zoom4: tone-mapping follows the densest pixel, small shifts everywhere),
//                mean 0.203 (littlewood-zoom4), centre 0, block 6 (with-zero-d14 reciprocal-coop), unconverged 0
//   limits       px % reported only; mean 0.5, centre 4, block 12; unconverged <= 1e-4 of the polynomials (every
//                variant, the baseline too)
//   before the fix (same run with the old polish()): ratio-0.01-d20 reciprocal centre 533-535, block 417-418;
//                ratio-100-d16 reciprocal centre 265-267, block 232-235; unconverged up to 10 % of the polynomials
//                (mean only 0.29-0.38: mean alone would not catch it)

const W = 800;
const H = 500;
const MAX_ITER = 32;
const BUDGET = 1 << 18; // solves per chunk: ~15-70 ms at degree 20, far below the Windows TDR
const DISPLAY: DisplayParams = { colorMode: 1, exposure: 1, gamma: 0.6 };
const LIMITS = { mean: 0.5, centre: 4, block: 12, unconverged: 1e-4 };
const BLOCK = 32;
const OUT = 'ab-out';
const BASELINE = 'plain-perPoly';
const VARIANTS: { name: string; enumeration: Enumeration; kernel: Kernel }[] = (['plain', 'reciprocal'] as const).flatMap((enumeration) =>
  (['perPoly', 'coop'] as const).map((kernel) => ({ name: `${enumeration}-${kernel}`, enumeration, kernel })),
);

interface Scene {
  name: string;
  coeffs: Complex[];
  degree: number;
  /** zoom 1 = the app's Reset view (radius 2 across the shorter side). */
  view: { cx: number; cy: number; zoom: number };
}
const scenes: Scene[] = JSON.parse(new TextDecoder().decode(readFileSync(new URL('../scenes/scenes.json', import.meta.url)))).scenes;
const viewOf = (s: Scene): View => ({ cx: s.view.cx, cy: s.view.cy, pxPerUnit: (s.view.zoom * Math.min(W, H)) / 4 });

function buildInfo(): BuildInfo {
  const version = JSON.parse(new TextDecoder().decode(readFileSync(new URL('../../package.json', import.meta.url)))).version ?? 'unknown';
  let commit = 'unknown';
  try {
    commit = execSync('git describe --always --dirty', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() || 'unknown';
  } catch {
    // not a git checkout
  }
  return { version, commit, builtAt: new Date().toISOString() };
}

const { device, errors, adapterName, gpu } = await nodeDevice();
const build = buildInfo();
afterAll(() => device.destroy());

interface Render {
  rgba: Uint8ClampedArray<ArrayBuffer>;
  ms: number;
  unconverged: number;
  polys: number;
}

async function render(scene: Scene, v: (typeof VARIANTS)[number]): Promise<Render> {
  const engine = new Engine(device, 'rgba8unorm');
  const out = new RgbaReadback(device, W, H);
  try {
    const view = viewOf(scene);
    engine.kernel = v.kernel;
    engine.enumeration = v.enumeration;
    engine.resize(W, H);
    engine.setParams({ coeffs: scene.coeffs, degree: scene.degree, view, maxIter: MAX_ITER });
    const t0 = performance.now();
    await runToEnd(device, engine, BUDGET);
    const ms = performance.now() - t0;
    const unconverged = (await engine.readStats()).unconverged;
    const enc = device.createCommandEncoder();
    engine.encodeDisplay(enc, out.view, DISPLAY);
    const rgba = await out.read(enc);
    const pinned = leadingFixable(scene.coeffs);
    const meta = buildMetadata({
      kind: 'image',
      build,
      environment: { gpu, userAgent: `node ${nodeVersion}, Dawn (webgpu npm), tests/gpu/ab.perf.ts`, timestamp: new Date().toISOString() },
      fractal: { coeffs: scene.coeffs, degree: scene.degree, maxIter: MAX_ITER, tol: TOL, pinned, ordering: engine.ordering, kernel: v.kernel, enumeration: engine.activeEnumeration ?? v.enumeration },
      screen: { ...view, width: W, height: H, dpr: 1 },
      output: { ...view, width: W, height: H },
      display: { colorMode: DISPLAY.colorMode, exposure: DISPLAY.exposure, gamma: DISPLAY.gamma, countScale: 1, rings: false },
    });
    await writePng(`${OUT}/${v.name}/${scene.name}.png`, rgba, metadataTexts(meta));
    return { rgba, ms, unconverged, polys: Number(engine.progress.total) };
  } finally {
    out.destroy();
    engine.destroy();
  }
}

async function writePng(path: string, rgba: Uint8ClampedArray, texts: Parameters<typeof encodePng>[3] = []): Promise<void> {
  mkdirSync(path.slice(0, path.lastIndexOf('/')), { recursive: true });
  writeFileSync(path, new Uint8Array(await (await encodePng(rgba, W, H, texts)).arrayBuffer()));
}

interface Diff {
  pct: number;
  mean: number;
  centre: number;
  block: number;
  heat: Uint8ClampedArray<ArrayBuffer>;
}

const lit = (p: Uint8ClampedArray, i: number) => p[i] + p[i + 1] + p[i + 2] > 0;

/** `a` (the variant) against `b` (the baseline), both W x H RGBA of `view`. */
function compare(a: Uint8ClampedArray, b: Uint8ClampedArray, view: View): Diff {
  const heat = new Uint8ClampedArray(W * H * 4);
  const blocks = new Int32Array(Math.ceil(W / BLOCK) * Math.ceil(H / BLOCK));
  let differ = 0;
  let sum = 0;
  let centre = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const d = Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]));
      sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
      heat[i + 3] = 255;
      if (d === 0) continue;
      differ++;
      const la = lit(a, i);
      const lb = lit(b, i);
      const rgb = la && !lb ? [255, 40, 40] : !la && lb ? [40, 200, 255] : [64 + 4 * d, 64 + 4 * d, 64 + 4 * d];
      heat.set(rgb, i);
      if (la === lb) continue;
      const s = la ? 1 : -1;
      blocks[Math.floor(y / BLOCK) * Math.ceil(W / BLOCK) + Math.floor(x / BLOCK)] += s;
      const z = pixelToWorld(view, W, H, x + 0.5, y + 0.5);
      if (Math.hypot(z.re, z.im) < 0.25) centre += s;
    }
  }
  return { pct: (differ / (W * H)) * 100, mean: sum / (W * H * 3), centre: Math.abs(centre), block: Math.max(...blocks.map(Math.abs)), heat };
}

/** Minimal PNG decoder for 8-bit RGB/RGBA without interlacing (what core/png and most tools write); all 5 filters. */
async function decodePng(png: Uint8Array<ArrayBuffer>): Promise<{ width: number; height: number; rgba: Uint8ClampedArray<ArrayBuffer> }> {
  const dv = new DataView(png.buffer, png.byteOffset, png.byteLength);
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat: Uint8Array<ArrayBuffer>[] = [];
  for (let o = 8; o + 12 <= png.length; ) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(...png.subarray(o + 4, o + 8));
    const data = png.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(o + 8);
      height = dv.getUint32(o + 12);
      if (data[8] !== 8 || (data[9] !== 2 && data[9] !== 6) || data[12] !== 0) throw new Error('only 8-bit RGB/RGBA non-interlaced PNGs');
      channels = data[9] === 2 ? 3 : 4;
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  const raw = new Uint8Array(await new Response(new Blob(idat).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
  const stride = width * channels;
  const px = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = y * stride;
    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? px[row + i - channels] : 0;
      const b = y > 0 ? px[row - stride + i] : 0;
      const c = y > 0 && i >= channels ? px[row - stride + i - channels] : 0;
      const p = a + b - c;
      const paeth = Math.abs(p - a) <= Math.abs(p - b) && Math.abs(p - a) <= Math.abs(p - c) ? a : Math.abs(p - b) <= Math.abs(p - c) ? b : c;
      px[row + i] = src[i] + [0, a, b, (a + b) >> 1, paeth][f];
    }
  }
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = 0; i < width * height; i++, j += channels) rgba.set([px[j], px[j + 1], px[j + 2], channels === 4 ? px[j + 3] : 255], i * 4);
  return { width, height, rgba };
}

interface Row {
  scene: string;
  variant: string;
  ms: string;
  unconverged: string;
  diff?: Diff;
  note?: string;
}
const rows: Row[] = [];

const WIDTHS = [18, 30, 7, 7, 7, 7, 7, 6];
const fmtCells = (cells: string[]) => cells.map((c, i) => (i < 2 ? c.padEnd(WIDTHS[i]) : i < WIDTHS.length ? c.padStart(WIDTHS[i]) : c)).join(' ');
const fmtRow = (r: Row) => {
  const d = r.diff;
  return fmtCells([r.scene, r.variant, r.ms, r.unconverged, d ? d.pct.toFixed(2) : '', d ? d.mean.toFixed(3) : '', d ? String(d.centre) : '', d ? String(d.block) : '', r.note ?? '']);
};

afterAll(() => {
  console.log(
    [
      `A/B ${W}x${H} on ${adapterName}, build ${build.version} (${build.commit}); baseline ${BASELINE}`,
      `limits: mean <= ${LIMITS.mean}, centre <= ${LIMITS.centre}, block <= ${LIMITS.block}, unconverged <= ${LIMITS.unconverged * 100} % of the polynomials`,
      fmtCells(['scene', 'variant', 'ms', 'unconv', 'px %', 'mean', 'centre', 'block']),
      ...rows.map(fmtRow),
    ].join('\n'),
  );
});

const refDir = env.AB_REF_DIR;
const sameScene = (m: Metadata, s: Scene) => {
  const v = viewOf(s);
  return (
    m.fractal.degree === s.degree &&
    m.fractal.coeffs.length === s.coeffs.length &&
    m.fractal.coeffs.every((c, i) => c.re === s.coeffs[i].re && c.im === s.coeffs[i].im) &&
    m.output?.cx === v.cx && m.output.cy === v.cy && m.output.pxPerUnit === v.pxPerUnit && m.output.width === W && m.output.height === H
  );
};

it.each(scenes.map((s) => [s.name, s] as const))('%s', async (_, scene) => {
  const view = viewOf(scene);
  const results = new Map<string, Render>();
  for (const v of VARIANTS) results.set(v.name, await render(scene, v));
  const base = results.get(BASELINE)!;
  for (const v of VARIANTS) {
    const r = results.get(v.name)!;
    const row: Row = { scene: scene.name, variant: v.name, ms: r.ms.toFixed(0), unconverged: String(r.unconverged) };
    rows.push(row);
    expect.soft(r.unconverged, `${scene.name} ${v.name}: unconverged polynomials`).toBeLessThanOrEqual(LIMITS.unconverged * r.polys);
    if (v.name === BASELINE) continue;
    row.diff = compare(r.rgba, base.rgba, view);
    await writePng(`${OUT}/diff/${v.name}/${scene.name}.png`, row.diff.heat);
    expect.soft(row.diff.mean, `${scene.name} ${v.name}: mean |diff|`).toBeLessThanOrEqual(LIMITS.mean);
    expect.soft(row.diff.centre, `${scene.name} ${v.name}: lit-pixel difference within |z| < 0.25`).toBeLessThanOrEqual(LIMITS.centre);
    expect.soft(row.diff.block, `${scene.name} ${v.name}: lit-pixel difference in a 32x32 block`).toBeLessThanOrEqual(LIMITS.block);
  }
  if (refDir) {
    const file = `${refDir}/${scene.name}.png`;
    const row: Row = { scene: scene.name, variant: 'ref', ms: '', unconverged: '' };
    rows.push(row);
    if (!existsSync(file)) {
      row.note = `missing ${file}`;
      expect.soft(existsSync(file), `${scene.name}: reference image ${file}`).toBe(true);
      return;
    }
    const bytes = readFileSync(file);
    const text = readPngTexts(bytes).find((t) => t.keyword === PNG_KEYWORD);
    const meta = text && parseMetadata(text.text);
    row.variant = `ref ${meta?.build.commit ?? '?'} ${meta?.fractal.enumeration ?? 'plain'}-${meta?.fractal.kernel ?? '?'}`;
    expect.soft(meta !== undefined && sameScene(meta, scene), `${scene.name}: ${file} renders the same scene`).toBe(true);
    const ref = await decodePng(bytes);
    if (ref.width !== W || ref.height !== H) return void (row.note = `size ${ref.width}x${ref.height}`);
    // This build's baseline against the other build's: positive = lit only here.
    row.diff = compare(base.rgba, ref.rgba, view);
    await writePng(`${OUT}/diff/ref/${scene.name}.png`, row.diff.heat);
    expect.soft(row.diff.mean, `${scene.name} vs reference build: mean |diff|`).toBeLessThanOrEqual(LIMITS.mean);
    expect.soft(row.diff.centre, `${scene.name} vs reference build: lit-pixel difference within |z| < 0.25`).toBeLessThanOrEqual(LIMITS.centre);
    expect.soft(row.diff.block, `${scene.name} vs reference build: lit-pixel difference in a 32x32 block`).toBeLessThanOrEqual(LIMITS.block);
  }
  expect(errors).toEqual([]);
});
