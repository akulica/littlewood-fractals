// Headless animation renderer: renders a littlewood-fractals/1 settings file (docs/settings-format.md) on the GPU
// through Dawn (the `webgpu` package, as the GPU tests do), with the app's own renderAnimation. Writes PNG frames
// (with the same per-frame metadata the app writes) and/or an H.264 MP4 through ffmpeg (on PATH), whose comment tag
// carries the animation metadata, so Load accepts both.
//
// Build (needs vite for the ?raw WGSL imports and __BUILD_INFO__):
//   npx vite build --ssr scripts/render-headless.ts --outDir scripts/dist
// Run from the repo (so `webgpu` resolves from node_modules):
//   node scripts/dist/render-headless.js <settings.json> --out <dir> [--width 3840 --height 2160] [--degree d]
//     [--png] [--png-workers N] [--mp4] [--crf 15] [--preset slow] [--kernel auto|perPoly|coop] [--batch-frames N|auto] [--bench N]
//   --bench N renders only the first N frames, writes nothing and prints the time per frame.
//   --batch-frames N|auto frames per warm-start batch (default auto: the first batches are timed at 100/50/25 % of the
//   memory limit and the fastest size renders the rest; N is capped by the memory limit). The log line prints the choice.
//   --png-workers N encodes the PNG frames on N worker threads, overlapped with rendering (default
//   min(4, max(1, cores - 2))); 0 encodes on the main thread, one frame at a time (the old path, for benchmarking).
// The workers are this same file, started again as a worker_threads Worker (bottom of the file): the build stays one
// self-contained script, and the workers run the app's own encodePng, so the files are byte-identical either way.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isMainThread } from 'node:worker_threads';
import { BUILD_INFO } from '../src/buildInfo';
import { BatchPlanner, batchFrames, pinnable, statesFor } from '../src/core/animation';
import { effectiveEnumeration } from '../src/core/reciprocal';
import { buildMetadata, describe, metadataTexts, parseMetadata, stateFromMetadata, type Metadata } from '../src/core/metadata';
import { encodePng } from '../src/core/png';
import { PngPool, defaultPngWorkers } from '../src/core/pngPool';
import { runNodePngWorker, spawnNodePngEncoder } from '../src/core/pngWorkerNode';
import type { Kernel } from '../src/core/schedule';
import { fitRect } from '../src/core/view';
import { renderAnimation, type DraftQuality } from '../src/gpu/animation';
import { Engine, TOL, framesKernel, type DisplayParams } from '../src/gpu/engine';

const MAX_ITER = 32;
const PNG_IN_FLIGHT = 8;

interface Opts {
  settings: string;
  out: string;
  width?: number;
  height?: number;
  degree?: number;
  png: boolean;
  pngWorkers: number;
  mp4: boolean;
  crf: number;
  preset: string;
  kernel: 'auto' | Kernel;
  batchFrames: number | 'auto';
  bench: number;
}

function parseArgs(argv: string[]): Opts {
  const o: Opts = { settings: '', out: '', png: false, pngWorkers: defaultPngWorkers(availableParallelism()), mp4: false, crf: 15, preset: 'slow', kernel: 'auto', batchFrames: 'auto', bench: 0 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i] ?? '';
    const int = (min = 1) => {
      const v = Number(next());
      if (!Number.isInteger(v) || v < min) throw new Error(`${a} needs a whole number ≥ ${min}`);
      return v;
    };
    if (a === '--out') o.out = next();
    else if (a === '--width') o.width = int();
    else if (a === '--height') o.height = int();
    else if (a === '--degree') o.degree = int();
    else if (a === '--png') o.png = true;
    else if (a === '--png-workers') o.pngWorkers = int(0);
    else if (a === '--mp4') o.mp4 = true;
    else if (a === '--crf') o.crf = Number(next());
    else if (a === '--preset') o.preset = next();
    else if (a === '--kernel') o.kernel = next() as Opts['kernel'];
    else if (a === '--batch-frames') {
      if (argv[i + 1] === 'auto') i++;
      else o.batchFrames = int();
    } else if (a === '--bench') o.bench = int();
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.settings = a;
  }
  if (!o.settings) throw new Error('usage: render-headless <settings.json> --out <dir> [--png] [--mp4] ...');
  if (!o.bench && !o.out) throw new Error('--out is required');
  if (!o.bench && !o.png && !o.mp4) o.png = o.mp4 = true;
  return o;
}

const fmt = (s: number) => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s`;

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const { create, globals } = await import('webgpu'); // not at the top: the PNG workers run this file too and need no GPU
  Object.assign(globalThis, globals);
  const gpu = create([]); // keep referenced for the whole process (see tests/gpu/gpuHelpers.ts)
  const meta = parseMetadata(readFileSync(o.settings, 'utf8'));
  const st = stateFromMetadata(meta);
  const anim = st.animation;
  if (!anim || anim.keyframes.length < 2) throw new Error('the settings file has no animation with at least 2 keyframes');
  const degree = o.degree ?? st.degree;
  const width = o.width ?? st.size?.width ?? 1920;
  const height = o.height ?? st.size?.height ?? 1080;

  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no GPU adapter');
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  const { vendor, architecture, device: devName, description } = adapter.info;
  console.log(`adapter: ${vendor} ${architecture} ${description}`);

  let kernel: Kernel;
  if (o.kernel === 'auto') {
    const cal = new Engine(device, 'rgba8unorm');
    cal.resize(width, height);
    const c = await cal.calibrate();
    cal.destroy();
    kernel = c.kernel;
    console.log(`kernel: ${kernel} (perPoly ${c.perPolyMs.toFixed(1)} ms, coop ${c.coopMs.toFixed(1)} ms)`);
  } else kernel = o.kernel;

  // The recorded screen is the reference: an export at k x its pixels per unit gets countScale k^2 (as in the app).
  const screenView = fitRect(st.view, meta.screen.width, meta.screen.height);
  const outView = fitRect(st.view, width, height);
  const density = outView.pxPerUnit / screenView.pxPerUnit;
  const pathOpts = { easing: anim.easing, loop: anim.loop };
  let states = statesFor(anim.keyframes, anim.frames, pathOpts, { view: st.view, exposure: st.display.exposure, gamma: st.display.gamma });
  if (o.bench) states = states.slice(0, o.bench);
  const coeffFrames = states.map((s) => s.coeffs);
  const hasView = anim.keyframes.some((k) => k.view);
  const hasDisplay = anim.keyframes.some((k) => k.exposure !== undefined && k.gamma !== undefined);
  const viewFrames = hasView ? states.map((s) => fitRect(s.view, width, height)) : undefined;
  const base: DisplayParams = { colorMode: st.display.colorMode, exposure: st.display.exposure, gamma: st.display.gamma, countScale: density * density };
  const enumeration = st.enumeration ?? 'reciprocal';
  const pinned = pinnable(coeffFrames);
  const n = coeffFrames.length;
  const cap = batchFrames(width * height * 16, device.limits);
  const autoBatch = o.batchFrames === 'auto' && new BatchPlanner(cap, n, 'auto').auto;

  const animMeta: Metadata = buildMetadata({
    kind: 'animation',
    build: BUILD_INFO,
    environment: { gpu: { vendor, architecture, device: devName, description }, userAgent: `render-headless (node ${process.version})`, timestamp: new Date().toISOString() },
    fractal: { coeffs: meta.fractal.coeffs, degree, maxIter: MAX_ITER, tol: TOL, pinned, ordering: 'golden', kernel: framesKernel(kernel),
      enumeration: effectiveEnumeration(coeffFrames, pinned, enumeration).mode },
    unverified: st.unverified,
    screen: meta.screen,
    output: { ...(viewFrames?.[0] ?? outView), width, height },
    display: { colorMode: base.colorMode, exposure: base.exposure, gamma: base.gamma, countScale: base.countScale ?? 1, rings: st.display.rings },
    animation: { ...meta.animation!, frames: n, batchFrames: o.batchFrames === 'auto' ? cap : Math.min(o.batchFrames, cap), batchAuto: autoBatch || undefined, output: o.png && o.mp4 ? 'mp4+png' : o.png ? 'png' : 'mp4', codec: o.mp4 ? 'avc' : null, bitrate: null },
  });

  const name = `littlewood-d${degree}-k${meta.fractal.coeffs.length}-${width}x${height}-${n}f`;
  let ffmpeg: ReturnType<typeof spawn> | null = null;
  let ffmpegExit: Promise<unknown> | null = null;
  let pngDir = '';
  let log = (line: string) => console.log(line);
  if (!o.bench) {
    mkdirSync(o.out, { recursive: true });
    const logFile = join(o.out, `${name}.log`);
    log = (line: string) => {
      console.log(line);
      appendFileSync(logFile, line + '\n');
    };
    writeFileSync(join(o.out, `${name}.settings.json`), JSON.stringify(animMeta, null, 1));
    if (o.png) {
      pngDir = join(o.out, `${name}-png`);
      mkdirSync(pngDir, { recursive: true });
      writeFileSync(join(pngDir, 'animation.json'), JSON.stringify(animMeta));
    }
    if (o.mp4) {
      // ffmetadata file: the comment tag (©cmt) holds the JSON, as the app's mediabunny writer does.
      const esc = (s: string) => s.replace(/[\\=;#\n]/g, (c) => (c === '\n' ? '\\\n' : `\\${c}`));
      const metaFile = join(o.out, `${name}.ffmeta.txt`);
      writeFileSync(metaFile, `;FFMETADATA1\ntitle=${esc(describe(animMeta))}\ncomment=${esc(JSON.stringify(animMeta))}\n`);
      const args = [
        '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${width}x${height}`, '-r', String(anim.fps), '-i', '-',
        '-i', metaFile, '-map', '0:v', '-map_metadata', '1',
        '-c:v', 'libx264', '-preset', o.preset, '-crf', String(o.crf), '-pix_fmt', 'yuv420p',
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-movflags', '+faststart', join(o.out, `${name}.mp4`),
      ];
      ffmpeg = spawn('ffmpeg', args, { stdio: ['pipe', 'inherit', 'inherit'] });
      ffmpegExit = once(ffmpeg, 'exit');
    }
  }

  log(`${name}: degree ${degree}, ${n} frames, ${width}x${height}, kernel ${kernel}, batch ${o.batchFrames === 'auto' ? `auto (memory limit ${cap})` : Math.min(o.batchFrames, cap)}, brightness ${anim.brightness}, enumeration ${enumeration}${pngDir ? `, PNG ${o.pngWorkers ? `${Math.min(o.pngWorkers, n)} workers` : 'on the main thread'}` : ''}`);
  const frameFile = (i: number) => join(pngDir, `frame_${String(i).padStart(5, '0')}.png`);
  const pool = pngDir && o.pngWorkers
    ? new PngPool<number>(() => spawnNodePngEncoder(fileURLToPath(import.meta.url)), Math.min(o.pngWorkers, n), (png, i) => writeFile(frameFile(i), png))
    : null;
  const pending: Promise<void>[] = [];
  const t0 = performance.now();
  let lastLog = 0;
  const result = await renderAnimation(
    device,
    {
      coeffFrames, viewFrames, degree, view: outView, width, height, display: base,
      displayFrames: hasDisplay ? states.map((s) => ({ exposure: s.exposure, gamma: s.gamma })) : undefined,
      rings: st.display.rings ? { scale: density } : undefined,
      brightness: anim.brightness, quality: anim.quality as DraftQuality, maxIter: MAX_ITER, enumeration,
      skipOverflowGuard: st.unverified.skipOverflowGuard, kernel, batchFrames: o.batchFrames,
    },
    async (rgba, i, info) => {
      if (ffmpeg && !ffmpeg.stdin!.write(rgba)) await once(ffmpeg.stdin!, 'drain');
      if (pngDir) {
        const texts = metadataTexts(buildMetadata({
          ...animMeta, kind: 'frame',
          frame: { index: i, coeffs: info.coeffs, view: { cx: info.view.cx, cy: info.view.cy, pxPerUnit: info.view.pxPerUnit },
            exposure: info.display.exposure, gamma: info.display.gamma, countScale: info.display.countScale ?? 1, fixedMax: info.display.fixedMax ?? 0, batch: info.batch },
        }));
        if (pool) {
          // The pool takes the buffer over (transfers it); ffmpeg may still be reading it from its pipe queue.
          await pool.submit({ rgba: ffmpeg ? rgba.slice() : rgba, width, height, texts }, i);
        } else {
          pending.push(encodePng(rgba, width, height, texts).then(async (blob) => {
            await writeFile(frameFile(i), new Uint8Array(await blob.arrayBuffer()));
          }));
          if (pending.length >= PNG_IN_FLIGHT) await pending.shift();
        }
      }
      const s = (performance.now() - t0) / 1000;
      if (s - lastLog >= 30 || i === n - 1) {
        lastLog = s;
        const left = (s / (i + 1)) * (n - i - 1);
        log(`frame ${i + 1}/${n} · ${fmt(s)} elapsed · ${((s * 1000) / (i + 1)).toFixed(0)} ms/frame · ~${fmt(left)} left`);
      }
    },
  );
  await Promise.all(pending);
  await pool?.finish();
  if (ffmpeg) {
    ffmpeg.stdin!.end();
    await ffmpegExit;
    if (ffmpeg.exitCode !== 0) throw new Error(`ffmpeg exited with ${ffmpeg.exitCode}`);
  }
  const s = (performance.now() - t0) / 1000;
  log(`done: ${n} frames in ${fmt(s)} (${((s * 1000) / n).toFixed(0)} ms/frame), unconverged ${result.unconverged}, fixedMax ${result.fixedMax}, batch ${result.batch.frames}${result.batch.auto ? ` (auto: ${result.batch.timings.map((t) => `${t.frames} -> ${t.msPerFrame.toFixed(1)} ms/frame`).join(', ')})` : ''}`);
  if (errors.length) log(`GPU validation errors: ${errors.join(' | ')}`);
  device.destroy();
}

if (isMainThread) {
  main().catch((e) => {
    console.error(`render-headless: ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  });
} else runNodePngWorker();
