# Littlewood fractals

Pick a few complex numbers. This app takes every polynomial whose coefficients come from that set, finds all of their roots, and plots them. With the coefficients `{1, −1}` (the Littlewood polynomials) you get the picture below. Drag a coefficient around and the whole fractal reshapes as you move it.

![Littlewood polynomials, degree 20](docs/littlewood-d20.png)

Everything runs on your GPU through WebGPU. A degree-24 Littlewood plot has about 16.8 million polynomials and 403 million roots, and a desktop GPU solves all of them in under a second.

## Features

- **Live editing:** drag coefficients, type exact values (up to 16 coefficients), pick a degree from 1 to 32, pan and zoom.
- **Big still images:** save PNGs up to 64 megapixels, e.g. 8192×8192.
- **Animations:** set keyframes for the coefficients, zoom and brightness, then render an MP4 or a folder of PNG frames.
- **Reproducible output:** every PNG and MP4 has its full settings embedded in it. Drop one back onto the page and you're back where you made it.

## Quick start

You need Node.js 20 or newer and a WebGPU-capable browser: recent Chrome or Edge, Safari 26+, or Firefox 141+ on Windows.

```bash
npm install
npm run dev     # open http://localhost:5173
```

Other commands:

```bash
npm test        # unit tests, plus GPU tests if this machine has a GPU
npm run build   # static site in dist/
npm run bench   # benchmark your GPU
```

On Windows PowerShell, if `npm` complains that "running scripts is disabled", use `npm.cmd` instead.

## How it works, briefly

Each GPU thread runs the Ehrlich–Aberth method on its own polynomial (or, on some GPUs, a workgroup of threads shares the polynomials, one root per thread). The roots are never stored: each root adds one count to a pixel histogram, which is then tone-mapped into the image. Because of that, memory depends on the image size and not on how many roots there are.

The app saves work in two ways:

- **Reversed polynomials:** a polynomial written backwards has the reciprocal roots (`1/z`), so only about half of the polynomials actually have to be solved.
- **Warm starts in animations:** each frame starts from the roots of the previous one.

## More detail

[docs/TECHNICAL.md](docs/TECHNICAL.md) is the full reference: every control, benchmark tables for two GPUs, the solver design and its numerical safeguards, the export metadata format, and the A/B image tests that check every optimisation. The settings file format is specified in [docs/settings-format.md](docs/settings-format.md).

## Prior art

[Greg Egan](https://www.gregegan.net/SCIENCE/Littlewood/Littlewood.html), [2swap](https://github.com/2swap/LittlewoodFractal) and Simon Tatham have made Littlewood visualizers before. Theirs work on the CPU or per pixel; this one solves for the roots on the GPU.

## License

[MIT](LICENSE)
