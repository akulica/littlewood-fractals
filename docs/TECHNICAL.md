# Littlewood fractals: technical reference

The full manual: every control, the benchmarks, and how the GPU solver works. For a quick overview see the [README](../README.md). File paths below (`src/…`, `tests/…`, `scripts/…`) are relative to the repository root.

Plots every root of every polynomial `a_0 + a_1 z + … + a_d z^d` whose coefficients `a_i` come from a set of complex numbers you choose. The default is `{1, −1}`, the Littlewood polynomials. All roots are computed on the GPU, fast enough to redraw while you drag the coefficients.

![Littlewood polynomials, degree 20](littlewood-d20.png)

## Requirements

- **Node.js 20+** (for the dev server, build and tests).
- **A WebGPU browser** to use the app (see *Browser support*). It must be opened via `localhost` or `https://`, because WebGPU is disabled on plain `http://` from other hosts.
- **A real GPU for `npm test`:** the GPU tests run the WGSL kernels through Dawn (the `webgpu` npm package, prebuilt for Windows/macOS/Linux x64). On a machine without a usable GPU, only the 13 pure-logic test files pass.
- **Windows PowerShell:** if `npm` fails with *"running scripts is disabled on this system"*, either use `npm.cmd` (e.g. `npm.cmd run dev`) or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Run

```bash
npm install
npm run dev       # http://localhost:5173  (WebGPU needs https:// or localhost)
npm test          # 226 tests: pure logic + real-GPU tests via Dawn (no browser needed)
npm run bench     # throughput + kernel-variant benchmark on this machine's GPU
npm run ab        # A/B image comparison of the enumeration/kernel variants (see *Checking an optimisation*)
npm run build     # static site in dist/
```

`scripts/pc-bench.cmd` runs tests and benchmarks on a Windows machine without Node installed. It downloads a portable Node 22 into `%LOCALAPPDATA%` (no admin rights needed). Run it from a copy of the repo **in a path without spaces**, e.g. `C:\lwf\scripts\pc-bench.cmd`.

`scripts/three-body.mjs` writes a settings file whose coefficients move like the three bodies of a periodic three-body orbit (figure-8 and 11 Šuvakov–Dmitrašinović orbits; `--list`, options in the file header). `scripts/render-headless.ts` renders any settings file's animation without a browser, through Dawn, to PNG frames and/or an MP4 (needs `ffmpeg` on PATH). Build it with `node node_modules/vite/bin/vite.js build --ssr scripts/render-headless.ts --outDir scripts/dist`, then run `node scripts/dist/render-headless.js <settings.json> --out <dir> [--width W --height H] [--degree d] [--png-workers N] [--batch-frames N|auto] [--bench N]`. PNG frames are encoded on worker threads overlapped with rendering (default `min(4, max(1, cores - 2))` workers, at most twice that many frames in flight, written in frame order; the workers are the built script itself, so it stays one file); `--png-workers 0` encodes on the main thread as before. The browser's PNG-folder export does the same with Web Workers. Both produce the exact bytes of `encodePng`.

## Controls

| Action | How |
|---|---|
| Move a coefficient | Drag its white ring. While dragging, every frame renders as many full-degree polynomials as fit in ~12 ms, spread evenly over the whole set, so the preview already looks like the final picture (the status line shows `preview · 3.1 % sampled`). On release it keeps refining the same sample to 100 % instead of starting over. |
| Set coefficients exactly | Type in **Coefficients**, e.g. `1, -1`, `1, i, -1, -i`, `0.5+0.3i, -1` (1–16 values), then press Enter. |
| Degree | 1–32 |
| Pan / zoom | Drag the background / mouse wheel (zooms around the cursor) / **Zoom ±** / **Reset view** |
| Colour | Hue by low-order coefficients (Egan's palette idea), or inferno density. Exposure and gamma apply instantly, without recomputing. |
| Kernel | **auto** (default) times both GPU kernels at startup and keeps the faster one. The choice and the measured ratio are shown in the status line. You can also force either kernel. Animations use the warm-starting variant of the same kernel (see *Animation*). |
| Enumeration | **reciprocal symmetry** (default, about 2× faster) solves one polynomial of each reversed pair and plots its roots `z` and the partner's `1/z`; **plain** solves every polynomial (the original path, for comparison). Same picture up to f32 rounding (see *Reciprocal symmetry*). Applies to the live view, Save PNG and animations; changing it restarts the accumulation. Recorded in every export and restored by Load. |
| Unverified options | A collapsed section at the bottom of the panel for options that fail `npm run ab` (see *Unverified options*). Off by default; changing one restarts the accumulation, and the status line shows `· unverified` while any is on. |
| Save PNG | Type the width × height, then **Save PNG**. The image is re-rendered at that size (never upscaled), shows everything currently on screen (extra margin if the aspect differs), and matches the screen's brightness. Up to 64 M pixels (e.g. 8192×8192 or 16384×4096) within the GPU's limits. Live rendering pauses while it exports. The PNG carries the full settings (see *Reproduction metadata*). |
| Coefficient rings | **Show coefficient rings** (off by default) draws the white handle rings into Save PNG and animation frames, sized like a screenshot of the screen at the export's resolution. In animations they follow each frame's coefficients. |
| Save settings / Load | **Save settings (.json)** downloads the current state (coefficients, degree, view, display, export size, keyframes and animation settings). **Load…**, or dropping a file anywhere on the page, restores it from such a JSON, from a PNG saved by the app, or from a rendered MP4. Other programs can generate settings files: format spec in [`docs/settings-format.md`](settings-format.md), JSON Schema in [`docs/settings.schema.json`](settings.schema.json). |
| Animation keyframes | Open **Animation**. Set the coefficients, **Add keyframe**, move them, add the next one; any number of keyframes (at least 2). Each coefficient moves in a straight line from keyframe to keyframe, with equal time per segment. With **Attach view (zoom + centre)** (on by default) a keyframe also stores what is on screen, so the animation can zoom and pan; with **Attach exposure/gamma** (on by default) it stores the two sliders. The list marks them 📷 (view) and ◐ (exposure/gamma). Ghost rings and dashed lines on the picture show the keyframes and the paths. Click a keyframe to jump to it (coefficients, view, exposure/gamma), × to remove it, **Clear** to start over. Changing the number of coefficients clears the keyframes. |
| Timeline | Drag the slider to scrub: the live view shows that moment, including the view and exposure/gamma (as a sampled preview, like dragging a handle). **▶ Preview** plays the frames the render will use, at the chosen FPS, looping. |
| Animation settings | **Loop back to start** adds the first keyframe at the end and drops the duplicate last frame, so the video loops seamlessly. **Frames**, **FPS**: frames sample the whole path, independent of the keyframe count. When frames would step over keyframes (fewer frames than keyframes, or smooth easing's faster middle), a warning under **Frames** says how many keyframes won't be shown exactly, and **Fit frames to keyframes** sets the smallest count that shows every one (linear: one frame per keyframe, exactly on each; smooth: about 1.45×). **Easing**: smooth (slow start and end) or linear. **Brightness**: *locked: density per area* (default; zoom-aware, see *View keyframes*), *locked to frame 0's densest pixel* (the old mode; zooming in darkens), or *per frame* (may pulse). **Quality**: 100 % for the final video; 50/25/10 % drafts render only that share of the polynomials per frame (an even sample, so the draft is noisier but otherwise the same picture). |
| Render animation | Size comes from the Save PNG W × H. **Output**: MP4 (H.264), MP4 + PNG frames, or PNG frames (`frame_00000.png`…; folder outputs need Chrome/Edge). You choose the file or folder first, then every frame is rendered at full size and written as it is finished. The status shows progress, ms/frame and the time left; **Cancel** stops it. Live rendering pauses meanwhile. |

## Browser support

WebGPU only (no WebGL fallback, by design):
- Chrome/Edge 113+ on Windows, macOS and ChromeOS
- Chrome on Android 12+
- Chrome on Linux with recent Intel or NVIDIA GPUs
- Safari 26+
- Firefox 141+ on Windows

Other browsers show an explanatory message.

## Performance

Benchmarks of record (`npm run bench`, 1024×1024 histogram, "one chunk in flight" mode as in the app):

| Case | Polynomials | RX 9060 XT (PC) | Arc 140T iGPU (laptop) |
|---|---|---|---|
| Littlewood d=20 | 1 048 576 | **67 ms** (313 M roots/s) | 110–250 ms |
| Littlewood d=24 | 16 777 216 | **0.96 s** (421 M roots/s) | 3.2–5 s |
| 3 complex coefficients, d=14 | 14 348 907 | **0.32 s** (627 M roots/s) | 0.9–1.7 s |

The laptop numbers vary run to run because the iGPU is power and thermal limited. In the browser on the laptop, degree 24 completes in about 2.4 s.

Kernel variants (`tests/gpu/variants.perf.ts`, ratio of cooperative time to per-polynomial time; below 1 means cooperative is faster):

| GPU | d=20 | d=24 |
|---|---|---|
| RX 9060 XT (RDNA 4) | 0.69–0.71 | 0.60–0.66 |
| Arc 140T | 1.19–1.33 | 1.15–1.30 |

The throughput table above uses the per-polynomial kernel. With auto-selection the PC uses the cooperative kernel, which brings degree 20 down to **50 ms** and degree 24 down to **0.71 s** (`calibrate: coop … coop/perPoly = 0.72` in `bench-pc.log`).

Since the winner depends on the GPU, both kernels ship. At startup `Engine.calibrate()` runs 2¹⁸ degree-20 Littlewood polynomials on each kernel: interleaved, best of 3 after a warm-up, about 0.1–0.5 s in total. It switches to the cooperative kernel only if that wins by more than 10 %, so timing noise can't flip the choice on GPUs where the two are close.

## Where the time goes

Per-stage profile of one still export (`renderImage`) and of a warm 16-frame animation (`renderAnimation`), measured
2026-10-20 with `npm run profile` (`tests/gpu/profile.perf.ts`; env `PROFILE_QUICK`, `PROFILE_REPS`, `PROFILE_FILTER`,
`PROFILE_OUT`). Measurement only: nothing shipped changed.

- **How it is timed.** GPU columns come from `timestamp-query` (requested only on the profiling device, never in the
  app) on the Engine's own passes, through a proxied command encoder. CPU columns are `performance.now()` around each
  submit + `onSubmittedWorkDone()` / `mapAsync`. A test at the top of the file proves the stage-by-stage path gives
  bit-identical images to the shipped `renderImage` / `renderAnimation`. Every case also runs the shipped function
  unmodified (the "e2e" column). Median of 3 repetitions, variants alternated.
- **Machines.** PC: RX 9060 XT (amd rdna-4, D3D12, Dawn via the `webgpu` npm package, node 24). Laptop: Arc 140T
  (intel xe-lpg, node 22). The laptop iGPU is power/thermal limited. Its e2e spread is up to 3.5× within one case,
  so read the PC as the numbers of record and the laptop as ranges.
- **Scene.** Littlewood {1,−1}, reciprocal enumeration, the app's Reset view, app display defaults. The second set
  {1, −0.4+0.75i} gives the same shares (raw data: `profile-*.json`, not committed). Animations move the coefficients
  linearly between the two sets over 16 frames (as `animation.perf.ts`), with locked brightness.
- **Column meanings.** "Engine setup" is `new Engine` + resize + setParams (for animations also `setFrames`, spread
  over the 16 frames). "rest" = e2e − all measured stages: CPU/driver latency between waited chunks (can be < 0
  through noise).

### Stills (ms per export, the faster kernel of each GPU)

| GPU | case | e2e ms (min–max) | Engine setup | solve+splat (GPU) | clear+reduce+tone-map+copy (GPU) | map+unpad (CPU) | rest |
|---|---|---|---|---|---|---|---|
| PC | d=16 1920x1080 coop | 123 (123–124) | 120 (98 %) | 1.8 (1 %) | 0.7 (1 %) | 1.2 (1 %) | -1 |
| PC | d=16 3840x2160 coop | 134 (133–138) | 127 (94 %) | 1.3 (1 %) | 2.5 (2 %) | 5.1 (4 %) | -1 |
| PC | d=20 1920x1080 coop | 160 (153–161) | 121 (76 %) | 29 (18 %) | 0.5 (0 %) | 1.2 (1 %) | 9 |
| PC | d=20 3840x2160 coop | 161 (161–169) | 127 (79 %) | 31 (19 %) | 2.4 (2 %) | 4.7 (3 %) | -4 |
| PC | d=22 1920x1080 coop | 218 (212–241) | 121 (55 %) | 88 (41 %) | 0.4 (0 %) | 1.2 (1 %) | 7 |
| PC | d=22 3840x2160 coop | 270 (236–300) | 126 (47 %) | 134 (50 %) | 2.4 (1 %) | 4.6 (2 %) | 2 |
| PC | d=24 1920x1080 coop | 532 (525–593) | 121 (23 %) | 404 (76 %) | 0.5 (0 %) | 1.2 (0 %) | 6 |
| PC | d=24 3840x2160 coop | 576 (553–585) | 128 (22 %) | 402 (70 %) | 2.5 (0 %) | 4.5 (1 %) | 40 |
| laptop | d=16 1920x1080 perPoly | 650 (179–658) | 195 (30 %) | 5.5 (1 %) | 3.3 (1 %) | 2.5 (0 %) | 444 |
| laptop | d=20 1920x1080 perPoly | 263 (230–312) | 179 (68 %) | 66 (25 %) | 2.4 (1 %) | 3.1 (1 %) | 14 |
| laptop | d=22 1920x1080 perPoly | 514 (446–522) | 189 (37 %) | 240 (47 %) | 2.5 (0 %) | 2.6 (1 %) | 80 |
| laptop | d=22 3840x2160 perPoly | 452 (451–568) | 230 (51 %) | 261 (58 %) | 8.5 (2 %) | 8.0 (2 %) | -56 |
| laptop | d=24 1920x1080 perPoly | 1446 (1346–1796) | 167 (12 %) | 1141 (79 %) | 2.2 (0 %) | 2.8 (0 %) | 134 |
| laptop | d=24 3840x2160 perPoly | 1491 (1412–2066) | 220 (15 %) | 1180 (79 %) | 9.2 (1 %) | 9.0 (1 %) | 73 |

**Engine setup is pipeline compilation.** A scratch split on the PC timed each `createComputePipeline`: `splat` 38–44 ms,
`splatCoop` 38 ms, `debugRoots` 29–31 ms (never used by exports), `display` render 10 ms, `reduceMax` 4 ms. Total
≈ 120 ms for every `new Engine`. A second Engine created while the first is alive costs the same, so Dawn does not
dedupe them. Animations compile `splatFrames` / `splatFramesCoop` lazily on top: +50–54 ms. Buffer allocation is 1.6 ms
(1024²) to 5.8 ms (4K). Below degree 22, a still export on the PC is mostly compile time. The live view doesn't pay
this cost (its Engine lives for the whole session), but every Save PNG, `renderAnimation` and test does.

### Warm animations (ms per frame, 16 frames, both kernels)

| GPU | case | e2e ms/frame (min–max) | setup (per frame) | solve+splat (GPU) | load+reduce+tone-map+copy (GPU) | map+unpad (CPU) | rest | PNG sink adds |
|---|---|---|---|---|---|---|---|---|
| PC | d=16 1920x1080 perPoly | 20 (18–20) | 11 (59 %) | 2.9 (15 %) | 0.7 (4 %) | 1.1 (6 %) | 3 | +38 |
| PC | d=16 1920x1080 coop | 15 (15–16) | 11 (76 %) | 1.3 (9 %) | 0.7 (5 %) | 1.1 (8 %) | 0 | +40 |
| PC | d=16 3840x2160 coop | 24 (23–25) | 12 (51 %) | 1.8 (8 %) | 3.4 (14 %) | 4.5 (19 %) | 2 | +149 |
| PC | d=20 1920x1080 perPoly | 70 (69–71) | 11 (16 %) | 50 (71 %) | 0.7 (1 %) | 1.2 (2 %) | 7 | +34 |
| PC | d=20 1920x1080 coop | 46 (44–48) | 11 (24 %) | 26 (55 %) | 0.7 (2 %) | 1.2 (3 %) | 7 | +33 |
| PC | d=20 3840x2160 coop | 65 (63–65) | 12 (19 %) | 35 (54 %) | 3.4 (5 %) | 4.6 (7 %) | 10 | +127 |
| PC | d=22 1920x1080 perPoly | 166 (166–168) | 11 (7 %) | 153 (92 %) | 0.7 (0 %) | 1.2 (1 %) | -0 | +40 |
| PC | d=22 1920x1080 coop | 114 (110–115) | 11 (10 %) | 99 (87 %) | 0.7 (1 %) | 1.2 (1 %) | 2 | +43 |
| PC | d=22 3840x2160 perPoly | 237 (237–242) | 12 (5 %) | 190 (80 %) | 3.5 (1 %) | 4.7 (2 %) | 27 | +125 |
| PC | d=22 3840x2160 coop | 173 (171–173) | 12 (7 %) | 135 (78 %) | 3.4 (2 %) | 4.7 (3 %) | 17 | +133 |
| laptop | d=16 1920x1080 perPoly | 29 (29–34) | 40 (137 %) | 5.0 (17 %) | 4.0 (14 %) | 2.6 (9 %) | -22 | +95 |
| laptop | d=20 1920x1080 coop | 101 (92–121) | 44 (44 %) | 61 (60 %) | 3.4 (3 %) | 3.0 (3 %) | -10 | +77 |
| laptop | d=20 3840x2160 perPoly | 149 (131–188) | 31 (21 %) | 73 (49 %) | 14 (10 %) | 9.5 (6 %) | 21 | +332 |
| laptop | d=22 1920x1080 perPoly | 317 (315–375) | 17 (6 %) | 275 (87 %) | 3.7 (1 %) | 2.9 (1 %) | 19 | +123 |
| laptop | d=22 1920x1080 coop | 331 (320–332) | 17 (5 %) | 279 (84 %) | 3.8 (1 %) | 2.8 (1 %) | 29 | +112 |
| laptop | d=22 3840x2160 coop | 413 (413–453) | 46 (11 %) | 342 (83 %) | 14 (3 %) | 9.2 (2 %) | 2 | +369 |

The e2e numbers agree with the earlier numbers of record:

- PC d=22 1080p: warm perPoly 166 vs 175 earlier, warm coop 114 vs 124.
- PC d=20 1080p warm coop: 46 vs 43–51.
- Laptop d=22 1080p: 317 / 331 vs 334 / 333.

The per-frame setup is the ~180 ms compile divided over 16 frames. It vanishes on long animations (1440 frames:
0.1 ms/frame). The earlier puzzle, "warm coop only ties cold coop on the PC at d=20", is most likely this fixed per-export cost: at
d=20 1080p the compile is 24 % of a 46 ms frame. It is not readback, which costs 1.2 ms (1080p) and 4.7 ms (4K) per
frame on the PC. The cold-vs-warm comparison itself was not re-run.

**Frame writers (per frame).**

- **PNG, PC:** `encodePng` (Node `CompressionStream`, serial in the sink) adds 33–43 ms at 1080p and 125–150 ms at 4K.
  At 4K and d ≤ 20 that is more than the render itself.
- **PNG, laptop:** 77–123 ms (1080p) and 280–475 ms (4K).
- **MP4, laptop:** WebCodecs H.264 + mediabunny muxing (headless Chrome 155, synthetic sparse frames, 96 frames × 2 runs,
  wall clock) is 7.4–8.1 ms/frame at 1080p and 27.7–29.1 ms/frame at 4K. MP4 export is therefore never encoder-bound
  at d ≥ 20. Not measured on the PC (no desktop session over SSH).

### Ceilings (test-only shader variants, GPU time relative to the shipped kernel)

| GPU | mode / kernel | no splat (solve only) | splat only, real roots | splat only, uniform | splat only, near \|z\|=1 |
|---|---|---|---|---|---|
| PC | still perPoly | 0.95–1.06 | 0.11–0.29 | 0.10–0.55 | 0.10–0.18 |
| PC | still coop | 0.86–0.99 | 0.14–0.38 | 0.16–0.78 | 0.15–0.32 |
| PC | anim perPoly | 0.77–0.97 | 0.38–0.54 | 0.53–0.82 | 0.14–0.27 |
| PC | anim coop | 0.59–0.80 | 1.02–1.39 | 1.28–2.02 | 0.32–0.74 |
| laptop | still perPoly | 0.84–0.98 | 0.18–0.30 | 0.23–0.41 | 0.16–0.24 |
| laptop | still coop | 0.92–0.99 | 0.13–0.32 | 0.17–0.45 | 0.13–0.19 |
| laptop | anim perPoly | 0.62–0.85 | 0.48–0.84 | 0.53–1.01 | 0.25–0.42 |
| laptop | anim coop | 0.63–0.91 | 0.54–0.96 | 0.57–0.97 | 0.29–0.40 |

How the ceilings were built:

- **No splat:** roots are summed into a register and written under a runtime-false uniform condition. Its unconverged
  count equals the full kernel's, so the solve still runs.
- **Splat only:** reads roots captured once from the same polynomials. Its histogram is bit-identical to the full
  kernel's. It includes reading the root buffer, which is why the coop animation layout's splat-only exceeds 1×. Read
  the "no splat" column for that layout.

What the ceilings show:

- **Stills:** if splatting cost nothing, a still would get at most 0–16 % faster. The atomics are not the problem, which
  confirms the earlier spike.
- **Warm animations:** splatting into 8 frame slices is 20–41 % of the coop kernel on the PC (3–23 % perPoly). The cost
  is memory locality, not contention on the same address: the same number of deposits spread uniformly is 1.3–4× slower
  than deposits concentrated near the unit circle. Eight slices × 16 B/px is 265 MB at 1080p and 1.06 GB at 4K, far
  beyond any cache.

### Next optimisations, ranked by measured payoff

1. **Compile the pipelines once per device, not per Engine.** A module-level cache keyed by device + entry point,
   shared by the live view, `renderImage` and `renderAnimation`. Compile `debugRoots` lazily (tests only), and optionally
   use `createComputePipelineAsync`. Expected gain on the PC, from the setup column:
   - Still exports: d=16–20 123–160 → ~3–40 ms, d=22 218 → ~100 ms (2.2×), d=24 532 → ~410 ms (1.3×).
   - Laptop stills: −170–230 ms each.
   - Animations: −180 ms per export.
   - Also: faster `npm test` / `npm run ab`.

   It touches no kernel, so images stay bit-identical. Cheapest and safest item on the list.
2. **Take PNG encoding off the render path.** A small worker pool (`worker_threads` headless, Web Workers in the
   browser's PNG-folder output) encodes frame i while the GPU renders the next batch; zlib level 1–3 is worth a test
   too. Expected:
   - PNG-sequence exports bounded by max(render, encode/workers) instead of render + encode.
   - PC 4K d=22 coop: 306 → ~175 ms/frame (1.75×). 4K d=20: 192 → ~65 (3×). 1080p d=20: 79 → ~46 (1.7×).
   - Headless renders with `--png` (e.g. the 4K three-body render) gain the same.

   MP4-only output gains nothing (the encoder is already async and 8–29 ms/frame).
3. **Shrink the animation histogram footprint (coop warm kernel).** Measured ceiling 1.25–1.7× on coop animations
   (no-splat 0.59–0.80). Realistic 1.1–1.25×. Candidates, each to be A/B-tested since some change colours:
   - 8 B/px records (u32 count + packed or lower-precision colour sums).
   - Fewer frames per batch at 4K (Fb = 4) to fit more of each slice in cache. This trades back some warm-start gain;
     Fb = 16 was already slower. Done as an online calibration (see *Batch size*): smaller wins on the RX 9060 XT and loses on the Arc.
   - Pixel-sorted deposits per workgroup.

   Stills gain little from this (no-splat 0.84–1.06×).

**Rejected, with the measurement:**

- **Faster reduce / tone-map / copy:** ≤ 3.5 ms GPU at 4K, ≤ 5 % of any still or d ≥ 20 frame.
- **Faster readback / row unpadding:** map ≈ 0.01 ms, unpad 1.2 ms (1080p) / 4.7 ms (4K) on the PC. That is only
  worth it for d=16 4K animations, where it is 19 %.
- **Histogram privatisation / atomics packing for stills:** no-splat ≥ 0.84×.
- **Keeping 2 chunks in flight:** "rest" ≤ 10 ms on the PC. It may help the laptop iGPU (rest up to ~30 ms/frame, wake
  latency after waits), but that is unmeasured.

The big remaining cost at d ≥ 22 is the Aberth solve itself (70–92 % of every frame). No candidate for it was measured
tonight.

## Reciprocal symmetry

- **Idea.** Reversing a coefficient string gives `z^d·p(1/z)`, again a polynomial of the set, with roots `1/z`. So each solve deposits its roots `z` and its partner's roots `1/z`, and only about half the polynomials are solved (`src/core/reciprocal.ts`, on by default; **Enumeration: plain** in the panel, or `Engine.enumeration = 'plain'`, is the original path). The partner gets its own colour (the hue of the reversed digits). Its trimmed leading zeros become roots at 0, and the polynomial's own roots at 0 go to infinity.
- **Exact cover.** Coefficient positions pair up from the outside, `(a_j, a_{d−j})`, and reversal swaps each pair. Region `R_j` fixes pairs `0..j−1` to `x = y`, takes one representative of each swapped couple at pair `j`, and leaves the rest free. The reversal of a string in `R_j` has the other member of the couple at pair `j`, so it lies in no region. Every string outside the regions is the reversal of exactly one string inside. The last region (all layer pairs fixed, at most 1/32 of the set) solves its strings alone.
- **Pinned sets** (`c0·G`) reverse and then renormalise the leading coefficient. This maps the class `a_0 = c0·r` onto `r⁻¹`: classes with `r ≠ r⁻¹` are enumerated once with partners, and self-inverse classes are layered with the twisted pair map `(x, y) → (y·r, x·r)`.
- **Even preview.** Each region has its own mixed-radix golden stride (coprime to every radix), and every chunk splits across all regions in proportion to their size. So any prefix is still an even sample of the whole set.
- **Tests.** `tests/reciprocal.test.ts` checks every string is covered exactly once, exhaustively, for small sets (pinned, unpinned, with 0, d = 1–7). `tests/gpu/reciprocal.test.ts` checks root counts and colour sums equal the plain path's exactly. Pixel differences stay below the noise between the two plain kernels.
- Measured (`tests/gpu/reciprocal.perf.ts`, 1920×1080, every polynomial, best of 3, plain → reciprocal):

  | Set | Degree | RX 9060 XT perPoly | RX 9060 XT coop | Arc 140T perPoly | Arc 140T coop |
  |---|---|---|---|---|---|
  | {1, −0.4+0.75i} | 20 | 82 → 43 ms (1.91×) | 68 → 34 ms (1.98×) | 219 → 119 ms (1.84×) | 288 → 150 ms (1.93×) |
  | {1, −0.4+0.75i} | 22 | 479 → 260 ms (1.84×) | 364 → 171 ms (2.12×) | 1065 → 561 ms (1.90×) | 1448 → 716 ms (2.02×) |
  | Littlewood (pinned) | 20 | 41 → 21 ms (1.93×) | 34 → 17 ms (2.04×) | 110 → 59 ms (1.86×) | 149 → 73 ms (2.05×) |
  | Littlewood (pinned) | 22 | 250 → 104 ms (2.42×) | 178 → 80 ms (2.23×) | 630 → 286 ms (2.20×) | 709 → 350 ms (2.02×) |

  Animation at 1920×1080, d = 22 (`animation.perf.ts`): 241 → 175 ms/frame on the RX 9060 XT, 780 → 380 ms/frame on the Arc 140T. Colour sums match the plain path to ~1e-4 (Littlewood: 1–2 %, the same multiple-root f32 noise as between the two plain kernels; `reciprocalColour.test.ts`).

## How roots are stored (they aren't)

- **While solving:** each root lives in GPU registers or workgroup memory as an f32 `vec2f` (8 bytes, about 7 significant digits).
- **Once converged:** it becomes a +1 on one pixel of a histogram with 4 × u32 per pixel: count, plus R, G, B sums with 0–63 added per root. After that its position is known only to one pixel.
- **Memory therefore scales with pixels, not roots:**
  - 1920×1080 is 33 MB, whether that's 1 root or 10 billion;
  - 8K export is about 531 MB.
- **For comparison, storing 100 M roots** would take 800 MB as f32 or 1.6 GB as f64. Littlewood degree 24 alone has 403 M roots.

**Export pipeline:**
1. A private `Engine` fills a W×H histogram.
2. It tone-maps into an `rgba8unorm` texture.
3. The texture is read back, with the 256-byte GPU row padding removed.
4. `src/core/png.ts` encodes the PNG: 8-bit RGB, deflate via `CompressionStream`, no canvas, so no browser canvas-size cap.

Root counts are multiplied by (export px ÷ screen px)² before tone-mapping, so a 4K export is as bright as the screen.

## Animation

- **One full render per frame.** Every frame is computed from scratch at its own coefficients. Nothing is interpolated between frames, because roots don't move in straight lines.
- **Warm start.** Consecutive frames move the roots only a little, so the previous frame's roots are a much better starting guess than the usual circle. `roots.wgsl`'s `splatFrames` kernel gives each GPU thread one polynomial for a batch of up to 8 frames: it solves frame 1, then starts frame 2 from those roots, and so on. If a warm start fails to converge, it retries from the circle.
  - Aberth sweeps drop from 6.4 to 2.3–2.7 per polynomial and frame. The 4 histogram atomics per root and the per-frame readback remain, so the end-to-end gain is smaller. Measured by `tests/gpu/animation.perf.ts`: 1920×1080, 16 frames, coefficient −1 → −0.4+0.75i, `renderAnimation` (including tone-map and RGBA readback) vs. every frame rendered from scratch:

    | GPU | Degree | From scratch | Animation | Speedup |
    |---|---|---|---|---|
    | RX 9060 XT | 22 | 470 ms/frame | **241 ms/frame** | **1.95×** |
    | RX 9060 XT | 20 | 83 ms/frame | 66 ms/frame | 1.25× (short frames: the fixed per-frame readback dominates) |
    | Arc 140T | 22 | 2131 ms/frame | **780 ms/frame** | **2.73×** |
    | Arc 140T | 20 | 337 ms/frame | 173 ms/frame | 1.95× |

    So 300 frames (10 s at 30 fps) at degree 22 take about 1.2 min on the PC and 3.9 min on the laptop.
  - Compared with cold renders: 0 unconverged, the same root counts, and 0.05–0.1 % of roots land one pixel over (f32 edge noise).
- **Cooperative warm start.** `coop.wgsl`'s `splatFramesCoop` does the same in the cooperative layout (one thread per root, floor(256/d) polynomials per workgroup). Each thread keeps its root in a register and in shared memory from frame to frame, and the digit of the one coefficient it loads per frame (thread 0 also loads the leading one). A polynomial whose warm start fails is solved again from the circle. That second pass runs for the whole workgroup (barriers need uniform control flow), but only when some polynomial in it failed. Unconverged polynomial-frames are counted once per polynomial, as in `splatFrames`. Frame 0 of a batch is bit-identical to `splatCoop`. Workgroup memory is 10.3 KB of the 16 KB WebGPU guarantees.
  - **Which one runs** follows the Kernel setting: auto uses the kernel `calibrate()` picked for still images, and a forced choice applies to animations too (perPoly → `splatFrames`, coop → `splatFramesCoop`). The metadata records the one that ran.
  - Measured by `animation.perf.ts` (1920×1080, 16 frames, reciprocal, ms/frame):

    | GPU | Degree | Cold perPoly | Cold coop | Warm perPoly | Warm coop |
    |---|---|---|---|---|---|
    | RX 9060 XT | 20 | 75 | 44 | 54–68 | **43–51** |
    | RX 9060 XT | 22 | 280 | 194 | 175 | **124** |
    | Arc 140T | 20 | 110 | 154 | 101 | 106 |
    | Arc 140T | 22 | 502 | 730 | 334 | 333 |

    On the Arc, the cooperative kernel's 1.45× cold handicap disappears with the warm start: Jacobi sweeps over a workgroup stop when its slowest polynomial converges, and a warm start leaves few slow ones. Auto still picks perPoly there (the calibration is a cold still render), which costs nothing given the tie. On the RX 9060 XT, where auto picks coop, the warm cooperative kernel is 1.41× faster than warm perPoly at d = 22 (175 → 124 ms/frame, two runs) and 1.57× faster than the best cold render; at d = 20 it only ties cold coop, because the per-frame readback dominates short frames. 300 frames at d = 22 now take about 37 s on the PC.
- **Batch size.** A batch keeps one histogram per frame on the GPU, capped at 512 MB (8 frames at 1080p, 4 at 4K). Measured with `splatFrames` 8 was fastest on both GPUs and 16 slower; later spikes with `splatFramesCoop` showed the best size depends on the GPU. RX 9060 XT, solve+splat ms/frame: 4K d = 20: Fb 2 / 4 / 8 = 30.5 / 35.2 / 45.0; 4K d = 22: 132 / 152 / 187; 1080p d = 20: Fb 4 / 8 = 24.4 / 26.2 (smaller is faster). Arc 140T: Fb 2 is 1.2–1.3× slower than 4 at 4K, 8 beats 4 at 1080p (64 vs 71–74 ms), and a histogram of 1 GiB or more (Fb 8 at 4K) falls off a cliff (5–20× slower). So the size is calibrated, not hard-coded. `renderAnimation`'s `batchFrames` option is `'auto'` (default) or a number (capped by the 512 MB rule). Auto renders the first batches at the memory cap, half, a quarter (at least 2), then the cap again, as real work, times the compute chunks of each (GPU solve time per frame; readback and the sink are not counted) and renders the rest at the fastest size by best time per size, leaving the cap only for a win of at least 5 %. The cap is measured twice because a render's first batch pays for clock ramp-up and first use of the buffer. Animations too short to profit (fewer frames than the trial plus one more cap batch: 30 with a cap of 8, 14 with a cap of 4) keep the cap. The choice only moves where the warm-start chain restarts cold; frame 0 is bit-identical whatever the size (tested). The trial costs a few percent of a 32-frame animation on a GPU where the cap wins (laptop: +2–6 %) and nothing on long ones. Pure logic: `BatchPlanner` in `core/animation.ts`; benchmark: `tests/gpu/batchframes.perf.ts`; `render-headless --batch-frames N|auto` prints the choice and the timings. Measured end to end (RX 9060 XT, coop, 32 frames, median of 3 interleaved rounds, ms/frame, auto vs the old fixed cap): 4K d = 20: 54.1 vs 54.3 (auto keeps 4); 4K d = 22: 145.9 vs 166.7 (0.88×, picks 2); 1080p d = 20: 26.5 vs 32.1 (0.83×, picks 4); 1080p d = 22: 101.6 vs 110.4 (0.92×, picks 4). At 1080p a forced 4 is faster still (21.8 / 88.6): on 32 frames the trial batches are a large share of the render, a cost that shrinks on longer animations. The animation settings record `batchFrames` (the largest batch) and `batchAuto: true`; each PNG frame records its batch as `frame.batch` (`first`, `frames`).
- **Locked brightness.** Without it the video pulses: the densest pixel changes 1.3–1.5× between adjacent frames, up and down. By default every frame is therefore tone-mapped against frame 0's densest pixel. Per-frame normalisation is still available.
- **Same look as the screen.** The animation uses the current colour, the same brightness scaling as Save PNG, and the current view and exposure/gamma unless keyframes carry their own.
- **Symmetry pinning** is decided once per animation: only if every frame allows it. Otherwise root multiplicity, and with it brightness, would jump between frames.
- **Outputs.**
  - MP4: H.264 via the browser's WebCodecs encoder, muxed by [mediabunny](https://mediabunny.dev) at about 0.25 bit per pixel per frame (1080p30 ≈ 15.6 Mbit/s). The size is checked up front: H.264 needs even width and height, and hardware encoders often stop around 4096 px.
  - Where the browser can write files (Chrome/Edge), the MP4 streams straight to the chosen file, so long videos never sit in memory. Elsewhere it is built in memory and downloaded.
  - PNG frames are written into the chosen folder as each frame finishes.

## View keyframes

- **Stored as the visible world rectangle** (centre plus half-width and half-height), not as `pxPerUnit`. `pxPerUnit` depends on the window size and devicePixelRatio when the keyframe was taken, so resizing the window between keyframes would silently change the framing. Each frame fits its rectangle into W×H (contain), exactly as Save PNG fits the screen.
- **Zoom is geometric** (linear in log scale). Rejected, linear in scale: a 1×→100× zoom would spend half the time between 1× and 50×, then crawl at the end.
- **The centre is anchored to the more zoomed-in keyframe.** Its on-screen offset from the frame centre shrinks (or grows) linearly while the zoom changes geometrically:
  - zooming in toward c₂: `c(t) = c₂ + (c₁ − c₂)·(s₁/s(t))·(1−t)`;
  - zooming out: the mirror image, anchored at c₁;
  - equal zoom: plain linear panning.
- Rejected, linear centre + geometric zoom: zooming 1→1000× toward a point 1 unit away puts the target about 16 screen widths off-screen mid-segment; it flies out, then snaps back.
- Rejected, van Wijk–Nuij "optimal" zoom-pan: it zooms *out* during long pans even when both keyframes have the same zoom, which is surprising for precise keyframing.
- **Keyframes without a view** take the view interpolated between the nearest keyframes that have one (with **Loop**, wrapping around). If no keyframe has a view, the current screen view is used. Exposure/gamma work the same way: exposure geometrically (it multiplies), gamma linearly. Easing and loop apply to every channel.
- **Brightness with zoom.** A pixel at zoom s covers 1/s² of the plane, so roots per pixel fall about s²-fold when zooming in, and a video locked to frame 0's brightest pixel fades to black. The default *density per area* lock scales each frame's counts by `(s_f / s_0)²` (on top of Save PNG's export factor) and keeps frame 0's reference, so regions of constant root density keep their brightness: in `tests/gpu/exportFeatures.test.ts` a 4× zoom stays within 0.5 % of frame 0, while the frame-0 lock drops it to 78 %. Caveats: at deep zoom, peaks that frame 0 averaged away can clip to full brightness (per-frame mode is still there), and deep zooms get **grainy**: most roots land off-screen, so a 30× zoom at degree 20 leaves few roots per pixel. Raising the degree is the remedy.
- **Rendering.** `splatFrames` reads a view per frame (stored after the coefficients in the same buffer), so a zooming batch still warm-starts.

## Coefficient rings

- Drawn by `display.wgsl`'s fragment shader over the tone-mapped colour: up to 16 ring centres (in output pixels) and the scale sit in the display uniform.
- The look copies the screen's `.handle`: a white ring spanning radius 9–11 CSS px with a soft 50 % black halo outside it (the CSS `box-shadow: 0 0 4px #000`), anti-aliased over one output pixel with `smoothstep`.
- Sizes are CSS px × (export px per CSS px), so an export looks like a screenshot of the screen at its resolution: big exports get big rings, small ones tiny rings.
- Rejected, compositing on the CPU after readback: it would duplicate work per frame, need its own anti-aliasing, and miss the video path's GPU timing. The live view keeps its DOM handles.

## Reproduction metadata

Every export records how to make it again, as JSON with schema `littlewood-fractals/1` (`src/core/metadata.ts`):
- **build:** package version, `git describe --always --dirty` and build time (injected by Vite `define` at build / dev-server start; `unknown` without git);
- **environment:** GPU adapter (vendor, architecture, device, description), user agent, timestamp;
- **fractal:** coefficients at full double precision, degree, `maxIter`, tolerance, pinned or not, enumeration order, kernel actually used (`perPoly` / `coop` for images; `splatFrames` / `splatFramesCoop` for animations), enumeration (`reciprocal` or `plain`; optional, files made before it existed load as before);
- **unverified:** only when an unverified option was on, e.g. `"unverified": { "skipOverflowGuard": true }` (absent = all off); the `Description` line then ends with `UNVERIFIED (…)`;
- **screen** (centre, `pxPerUnit`, canvas size, dpr) and the **output's own view** (centre, `pxPerUnit` in output pixels, W×H);
- **display:** colour mode, exposure, gamma, `countScale`, rings;
- **animation:** the keyframes exactly as stored, frames, fps, loop, easing, brightness, draft quality, batch size, codec and bitrate; each PNG frame adds its index and its interpolated coefficients, view, exposure/gamma, `countScale`, `fixedMax` and the batch it was solved in.

Where it goes:
- **PNG:** an `iTXt` chunk with keyword `littlewood-fractals` (uncompressed UTF-8 JSON), plus standard `tEXt` `Software` and `Description` (one human-readable line: degree, coefficients, centre, zoom). Any PNG tool shows them, e.g. `exiftool`.
- **MP4:** the JSON in the `comment` tag (the `©cmt` atom; `ffprobe -show_format` prints it), plus `title` and `date`.
- **PNG frame folder:** `animation.json` next to the frames, and each frame's own PNG chunks.
- **Save settings (.json):** the same record for the current state, with the export size and view Save PNG would use.

Reproducibility: the same parameters on the same GPU and driver give a bit-identical image. A different GPU family gives up to about 0.1 % of roots one pixel over (f32 rounding, measured between kernels and GPUs). That's why the adapter and the kernel are recorded.

## Load

**Load…** or a file dropped on the page reads the metadata from a PNG (`iTXt` chunk), an MP4 (comment tag, read with mediabunny's `Input`) or a settings JSON, validates it and restores coefficients, degree, view, colour/exposure/gamma, rings, export size, keyframes and animation settings. The recorded screen rectangle is fitted to the current window, so the same region is shown whatever the window size. Unknown or newer schemas and malformed fields are refused with a message naming the problem (e.g. `fractal.degree must be a whole number in 1..32`). If the file was made by another build or on another GPU, a note says so (results may differ slightly, see above). A file without an animation keeps the current keyframes if they have the same number of coefficients. The enumeration is restored; files made before it was recorded keep the current setting, and a note says so. Unverified options are restored too (a file without the record turns them off), with a note when one is on. The kernel choice is not restored: auto picks the fastest one for this GPU.

## Checking an optimisation

`npm run ab` (`tests/gpu/ab.perf.ts`) renders the scenes in `tests/scenes/scenes.json` at 800×500: Littlewood (full and zoomed 4×), `{1, 0.01}` and `{1, 100}` (huge coefficient ratios: roots near 0 and far away), `{1, −0.4+0.75i}`, three complex coefficients, a set containing 0, and the 4th roots of unity. It renders each scene with every variant (enumeration plain/reciprocal × kernel perPoly/coop), writes `ab-out/<variant>/<scene>.png` with the full reproduction metadata, and compares every variant with the baseline `plain-perPoly`:
- **px %:** pixels that differ at all (reported only; tone-mapping follows the densest pixel, so small shifts touch many pixels);
- **mean:** mean |difference| per channel;
- **centre:** lit-pixel difference within |z| < 0.25, where the old big-root bug drew its ring of dots;
- **block:** largest lit-pixel difference in any 32×32 block, which catches localized artifacts anywhere;
- unconverged polynomials.

It prints a table, saves a heatmap per scene in `ab-out/diff/<variant>/` (red = lit only in the variant, cyan = only in the baseline), and fails above limits calibrated on the Arc 140T (in the file header: mean 0.5, centre 4, block 12, unconverged ≤ 0.01 %; measured 0.20, 0, 6, 0). Without the big-root fix it failed with centre 533 and block 417. Rows named `unverified-*` (see *Unverified options*) are report-only: the note lists the limits they exceed, and they never fail the run.

**Animations** (`animations` in `scenes.json`): `{1, 0.01}` → `{1, 0.02+0.01i}` at d = 18, a Littlewood zoom 1× → 6× at d = 20, and a moving two-coefficient set zooming 1× → 3× at d = 18, 6 frames each. Every frame is rendered cold by the baseline path (`cold-plain-perPoly`, one `Engine.setParams` per frame) and by `renderAnimation` with each warm kernel (`warm-<enumeration>-<kernel>`: `splatFrames` / `splatFramesCoop`), with per-frame brightness so the tone-mapping matches. The same limits apply to every frame, and a row shows the worst frame. Measured on the Arc 140T: mean ≤ 0.225, centre ≤ 1, block ≤ 4, 0 unconverged. Frames go to `ab-out/<variant>/<scene>-f<i>.png` with frame metadata.

To check a change against the previous build:
1. Run `npm run ab` on the old build and copy its `ab-out/plain-perPoly` somewhere (e.g. `../ab-old`).
2. Run it on the new build with `AB_REF_DIR=../ab-old npm run ab` (PowerShell: `$env:AB_REF_DIR='../ab-old'; npm run ab`). This adds a `ref` row per scene that compares this build's baseline with the old images, under the same limits. The metadata of the old images must describe the same scenes.

## Unverified options

The panel's last section, **Unverified options (fail A/B tests)**, holds options and kernels that can draw visibly wrong pictures in some cases. They stay for speed experiments and comparison. The rule: anything that fails `npm run ab` goes here instead of being deleted. Each one is off by default and applies to the live view, Save PNG and animations. Exports record it (`unverified` in the metadata, `UNVERIFIED` in the PNG Description / MP4 title), and Load restores it. `npm run ab` renders these options as report-only `unverified-*` rows, which list the limits they exceed but never fail the run.

- **Skip big-root overflow guard** (`Engine.skipOverflowGuard`; the `direct` flag in the shaders' `Params`, a uniform, so no SIMD divergence). Every kernel's `polish()` (`splat`, `splatCoop`, `splatFrames`, `splatFramesCoop`, `debugRoots`) evaluates `p(z)` directly instead of the reversed polynomial at `1/z` for |z| > 1. That is the evaluation before the fix (see *How it works*): big roots overflow f32 in the stop test (`|p|² → inf ≤ inf`) and are accepted unconverged.
  - With **plain** enumeration they are off-screen, so the picture is the same.
  - With **reciprocal** symmetry their `1/z` draws a ring of false dots near the origin. The panel warns when both are on.
  - `tests/gpu/hugeRoots.test.ts`: `{1, 0.01}` at d = 16 has 14 lit pixels within |z| < 0.25 with the guard. Without it, plain gives 14 and reciprocal 56, on both kernels, still and animation; these are exactly the pre-fix numbers. In `npm run ab` the reciprocal row is expected to fail the centre limit on `ratio-0.01-d20`.
  - Measured in `npm run ab` (Arc 140T): the reciprocal row fails with centre 533 and block 417 on `ratio-0.01-d20` (267 and 235 on `ratio-100-d16`). Both rows have 1e4–2e5 unconverged polynomials on those two scenes. The other six scenes stay within every limit.
  - The guard costs nothing measurable: in a same-session A/B on the PC (RX 9060 XT) the guarded and unguarded shaders were within ±2 %. On scenes with big roots, skipping it is even slower (the unconverged polynomials run to `maxIter`: `ratio-0.01-d20` plain 215 → 394 ms in one Arc run). There is no speed reason to turn it off; it is here so the bug can be shown and compared.

## How it works

- **No QR / companion matrices.** Dense QR costs about 10n³ flops per polynomial, and the n×n complex matrix spills out of registers. Batched non-symmetric eigensolvers exist only in CUDA libraries.
- **Two kernels, picked per GPU.**
  - `roots.wgsl`: one polynomial per thread, Gauss–Seidel. Wins on Intel Arc.
  - `coop.wgsl`: one root per thread, floor(256/d) polynomials per workgroup, Jacobi sweeps through shared memory. Measured about 1.5× faster on AMD RDNA 4 (RX 9060 XT), and 1.2–1.3× slower on Intel Arc.
  - Each has an animation variant that warm-starts from the previous frame (`splatFrames`, `splatFramesCoop`); the same choice picks it.
- **Ehrlich–Aberth (`roots.wgsl`: one polynomial per GPU thread).** Cubic convergence, Gauss–Seidel updates, f32. Four guards make it robust:
  - a backward-error stop, so multiple roots don't burn iterations;
  - clamping iterates to the Cauchy root bound;
  - an integer inf/NaN check, because WGSL may assume floats are finite;
  - for |z| > 1, evaluating the reversed polynomial at `w = 1/z` (`p/p' = z·rev / (m·rev − w·rev')`), so no power of z overflows f32. Before this, big roots (e.g. `{1, 0.01}`: 0.01·z¹² + z¹¹ + … has a root near −100) made `|p|²` in the stop test `inf ≤ inf` and were accepted wherever they were. Off-screen in the plain path, but reciprocal symmetry drew their `1/z` as a ring of wrong dots around the origin. `tests/gpu/hugeRoots.test.ts` checks them against the f64 reference.
  - Accuracy against an f64 reference: better than 1e-4 for general sets, better than 1e-3 for Littlewood (clustered roots).
- **No roots stored.** Each root is `atomicAdd`ed into a per-pixel histogram (count + RGB). A reduction finds the densest pixel, and a fragment shader tone-maps with a log curve.
- **Enumeration.** Polynomial number n is decoded from its base-k digits (digit i picks `a_i`). The CPU sends each chunk's start digits as BigInt, so totals beyond 2⁵³ work. With reciprocal symmetry (the default) the same decoding runs per region, in that region's mixed radix (single digits and digit pairs, see *Reciprocal symmetry*).
- **Enumeration order: golden stride.** Sample m is polynomial m·G mod N, with G ≈ 0.618·N made coprime to k. The shader gets the digits of the chunk start and of G and adds `G·tid` digit by digit with carry.
  - It is a bijection of [0, N), so the full render is just the continuation of whatever prefix is on screen, with a bit-identical final histogram.
  - Any prefix is spread evenly over the whole set: a Weyl sequence in the high digits, and every block of k^j samples hits each pattern of the low j digits exactly once.
  - A small fixed stride ("skip 19, compute one") is biased. With k = 2 a stride of 20 freezes a₀ = a₁ = +1 and shows a different fractal; a coprime small stride still leaves lattice artifacts.
  - Measured at a 1/16 budget (Littlewood degree 20, 512², L1 distance from the full picture over 8×8 blocks; 0 = identical, 2 = disjoint): the old complete lower-degree preview (degree 16) scores **0.153**, stride 20 scores **0.282**, golden stride **0.017**.
- **Symmetry pinning.** If the set is `c0·G` for a finite group G (±1, roots of unity, …), every polynomial is a unit multiple of one with leading coefficient `c0`. The leading digit is pinned, giving k× less work.
- **Scheduling.**
  - While you interact: one adaptive (~12 ms) chunk per animation frame, at full degree, continuing the golden-order sample.
  - While idle: back-to-back ~40 ms chunks, with redraws on animation frames.

## Prior art examined

- Greg Egan's applet (`gregegan.net/SCIENCE/Littlewood`): CPU, per-pixel branch-and-bound + Halley step.
- 2swap's visualizer (`2swap.github.io/LittlewoodFractal`, `github.com/2swap/LittlewoodFractal`): CPU Durand–Kerner, 10 000-root cap.
- Simon Tatham's polyrootsfractal: WebGL per-pixel exclusion bound.

None of these compute roots on the GPU.

References:
- Aberth on GPU: Ghidouche, Couturier & Sider, JoCS 2017.
- MPSolve: Bini–Fiorentino; benchmarks in Fortune, JSC 2002.
- Fast companion QR: Aurentz–Mach–Vandebril–Watkins, SIMAX 2015.
