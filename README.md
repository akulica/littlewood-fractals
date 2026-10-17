# Littlewood fractals (WebGPU)

Plots every root of every polynomial `a_0 + a_1 z + … + a_d z^d` whose coefficients `a_i` come from a set of complex numbers you choose. The default is `{1, −1}`, the Littlewood polynomials. All roots are computed on the GPU, fast enough to redraw while you drag the coefficients.

![Littlewood polynomials, degree 20](docs/littlewood-d20.png)

## Run

```bash
npm install
npm run dev       # http://localhost:5173  (WebGPU needs https:// or localhost)
npm test          # 44 tests: pure logic + real-GPU tests via Dawn (no browser needed)
npm run bench     # throughput + kernel-variant benchmark on this machine's GPU
npm run build     # static site in dist/
```

`scripts/pc-bench.cmd` runs tests and benchmarks on a Windows machine without Node installed. It downloads a portable Node into `%LOCALAPPDATA%` and needs no admin rights.

## Controls

| Action | How |
|---|---|
| Move a coefficient | Drag its white ring. While dragging, a lower "preview" degree renders every frame; on release it refines to the full degree. |
| Set coefficients exactly | Type in **Coefficients**, e.g. `1, -1`, `1, i, -1, -i`, `0.5+0.3i, -1` (1–16 values), then press Enter. |
| Degree | 1–32 |
| Pan / zoom | Drag the background / mouse wheel (zooms around the cursor) / **Zoom ±** / **Reset view** |
| Colour | Hue by low-order coefficients (Egan's palette idea), or inferno density. Exposure and gamma apply instantly, without recomputing. |

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
| RX 9060 XT (RDNA 4) | 0.69 | 0.66 |
| Arc 140T | 1.19–1.33 | 1.15–1.30 |

The shipped kernel is per-polynomial. Auto-selecting the faster kernel per GPU is the obvious next step.

## How it works

- **No QR / companion matrices.** Dense QR costs about 10n³ flops per polynomial, and the n×n complex matrix spills out of registers. Batched non-symmetric eigensolvers exist only in CUDA libraries.
- **Ehrlich–Aberth, one polynomial per GPU thread.** Cubic convergence, Gauss–Seidel updates, f32. Three guards make it robust:
  - a backward-error stop, so multiple roots don't burn iterations;
  - clamping iterates to the Cauchy root bound;
  - an integer inf/NaN check, because WGSL may assume floats are finite.
  - Accuracy against an f64 reference: better than 1e-4 for general sets, better than 1e-3 for Littlewood (clustered roots).
- **No roots stored.** Each root is `atomicAdd`ed into a per-pixel histogram (count + RGB). A reduction finds the densest pixel, and a fragment shader tone-maps with a log curve.
- **Enumeration.** Polynomial number n is decoded from its base-k digits (digit i picks `a_i`). The CPU sends each chunk's start digits as BigInt, so totals beyond 2⁵³ work.
- **Symmetry pinning.** If the set is `c0·G` for a finite group G (±1, roots of unity, …), every polynomial is a unit multiple of one with leading coefficient `c0`. The leading digit is pinned, giving k× less work.
- **Scheduling.**
  - While you interact: one adaptive (~12 ms) chunk per animation frame.
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
