import { env } from 'node:process';
import { afterAll, expect, it } from 'vitest';
import { batchFrames, framesFor } from '../../src/core/animation';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { renderAnimation } from '../../src/gpu/animation';
import { Engine } from '../../src/gpu/engine';
import { nodeDevice } from './gpuHelpers';

// Frames per warm-start batch: end-to-end renderAnimation (tone-map + readback included, no-op sink) with the batch
// size forced to 2 / 4 / 8 (sizes above the memory cap are skipped: they would repeat the cap) and 'auto', all in this
// one process, interleaved round by round, the start rotating (the first round is a warm-up and is not counted). The best size depends on
// the GPU (PC RX 9060 XT: smaller is faster; Arc 140T: the cap), which is why renderAnimation times it online.
//
// Env (all optional):
//   BF_W, BF_H   output size, comma lists paired by position (default 3840x2160); e.g. BF_W=3840,1920 BF_H=2160,1080
//   BF_D         degrees, comma list, crossed with the sizes (default 20)
//   BF_FRAMES    frames per render (default 32; auto needs >= 30 to calibrate with a cap of 8, >= 14 with a cap of 4)
//   BF_ROUNDS    counted rounds (default 3)
//   BF_KERNEL    perPoly | coop (default: Engine.calibrate(), as the app and render-headless pick it)
// PC, ~3 minutes in total (PowerShell; cmd: `set BF_W=3840` and so on, bash: `BF_W=3840 npx ...`):
//   $env:BF_W='3840'; $env:BF_H='2160'; $env:BF_D='20,22'; npx vitest run --config vitest.perf.config.ts tests/gpu/batchframes.perf.ts
//   $env:BF_W='1920'; $env:BF_H='1080'; $env:BF_D='20'; npx vitest run --config vitest.perf.config.ts tests/gpu/batchframes.perf.ts
const { device, adapterName } = await nodeDevice();
afterAll(() => device.destroy());
console.log(`adapter: ${adapterName}`);

const list = (name: string, fallback: string) => (env[name] ?? fallback).split(',').map((x) => Number(x.trim()));
const widths = list('BF_W', '3840');
const heights = list('BF_H', '2160');
const degrees = list('BF_D', '20');
const FRAMES = Number(env.BF_FRAMES ?? 32);
const ROUNDS = Number(env.BF_ROUNDS ?? 3);
if (widths.length !== heights.length) throw new Error('BF_W and BF_H need the same number of entries');

const keys: Complex[][] = [
  [{ re: 1, im: 0 }, { re: -1, im: 0 }],
  [{ re: 1, im: 0 }, { re: -0.4, im: 0.75 }],
];
const display = { colorMode: 1 as const, exposure: 1, gamma: 0.6 };
const stats = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return { min: s[0], med: s[Math.floor(s.length / 2)], max: s[s.length - 1] };
};

let kernel: Kernel;
if (env.BF_KERNEL === 'perPoly' || env.BF_KERNEL === 'coop') kernel = env.BF_KERNEL;
else {
  const cal = new Engine(device, 'rgba8unorm');
  cal.resize(1920, 1080);
  kernel = (await cal.calibrate()).kernel;
  cal.destroy();
}
console.log(`kernel ${kernel}, ${FRAMES} frames, ${ROUNDS} rounds (+1 warm-up)`);

const cases = widths.flatMap((W, i) => degrees.map((d) => ({ W, H: heights[i], d })));
it.each(cases)('$W x $H degree $d', async ({ W, H, d }) => {
  const cap = Math.min(FRAMES, batchFrames(W * H * 16, device.limits));
  const sizes = [...new Set([...[2, 4, 8].filter((n) => n <= cap), cap])];
  const variants: (number | 'auto')[] = [...sizes, 'auto'];
  const frames = framesFor(keys, FRAMES, { easing: 'linear', loop: false });
  const view = fitView(W, H, 1.75);
  const ms = new Map<number | 'auto', number[]>(variants.map((v) => [v, []]));
  const chosen: string[] = [];
  for (let round = 0; round <= ROUNDS; round++) {
    for (let k = 0; k < variants.length; k++) {
      const v = variants[(k + round) % variants.length]; // the start rotates: no variant is always last (thermal drift)
      let n = 0;
      const t0 = performance.now();
      const r = await renderAnimation(device, { coeffFrames: frames, degree: d, view, width: W, height: H, display, brightness: 'locked', quality: 1, maxIter: 32, kernel, batchFrames: v }, () => void n++);
      const t = (performance.now() - t0) / FRAMES;
      expect(n).toBe(FRAMES);
      expect(r.unconverged).toBe(0);
      if (round === 0) continue;
      ms.get(v)!.push(t);
      if (v === 'auto') chosen.push(`${r.batch.frames}${r.batch.auto ? '' : ' (no calibration)'} [${r.batch.timings.map((x) => `${x.frames}:${x.msPerFrame.toFixed(1)}`).join(' ')}]`);
    }
  }
  const base = stats(ms.get(cap)!).med;
  console.log(`\n${W}x${H} d=${d}, kernel ${kernel}, memory cap ${cap}: e2e ms/frame, ratio of the median to forced ${cap}`);
  for (const v of variants) {
    const s = stats(ms.get(v)!);
    const label = v === 'auto' ? 'auto' : `forced ${v}`;
    console.log(`  ${label.padEnd(9)} min ${s.min.toFixed(1).padStart(7)}  med ${s.med.toFixed(1).padStart(7)}  max ${s.max.toFixed(1).padStart(7)}  ratio ${(s.med / base).toFixed(2)}  [${ms.get(v)!.map((x) => x.toFixed(0)).join(' ')}]`);
  }
  console.log(`  auto chose: ${chosen.join(' | ')}   (batch size: solve ms per frame of the trial batches)`);
});
