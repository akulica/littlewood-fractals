import { describe, expect, it } from 'vitest';
import {
  MetadataError,
  PNG_KEYWORD,
  SCHEMA,
  buildMetadata,
  describe as describeMeta,
  environmentNotes,
  metadataTexts,
  parseMetadata,
  stateFromMetadata,
  type MetadataInput,
} from '../src/core/metadata';
import { encodePng, readPngTexts } from '../src/core/png';
import { BUILD_INFO } from '../src/buildInfo';

const gpu = { vendor: 'intel', architecture: 'xe-lpg', device: '', description: 'Intel(R) Arc(TM) 140T GPU' };
const build = { version: '0.1.0', commit: '2d04924-dirty', builtAt: '2026-10-20T01:00:00.000Z' };

/** An animation export with every optional part present. 0.1 + 0.2 and 1/3 check full double precision. */
const input: MetadataInput = {
  kind: 'frame',
  build,
  environment: { gpu, userAgent: 'Mozilla/5.0 test', timestamp: '2026-10-20T02:00:00.000Z' },
  fractal: { coeffs: [{ re: 1, im: 0 }, { re: 0.1 + 0.2, im: -1 / 3 }], degree: 20, maxIter: 32, tol: 1e-5, pinned: false, ordering: 'golden', kernel: 'perPoly' },
  screen: { cx: 0.25, cy: -1 / 7, pxPerUnit: 412.5, width: 2400, height: 1500, dpr: 1.5 },
  output: { cx: 0.25, cy: -1 / 7, pxPerUnit: 990, width: 3840, height: 2160 },
  display: { colorMode: 1, exposure: 1.25, gamma: 0.6, countScale: 5.76, rings: true },
  animation: {
    keyframes: [
      { coeffs: [{ re: 1, im: 0 }, { re: -1, im: 0 }], view: { cx: 0, cy: 0, halfW: 3.2, halfH: 2 }, exposure: 1, gamma: 0.6 },
      { coeffs: [{ re: 1, im: 0 }, { re: 0.3, im: -1 / 3 }] },
      { coeffs: [{ re: 1, im: 0 }, { re: 0, im: 1 }], view: { cx: 0.5, cy: 0.5, halfW: 0.032, halfH: 0.02 } },
    ],
    frames: 240,
    fps: 30,
    loop: true,
    easing: 'smooth',
    brightness: 'locked',
    quality: 0.5,
    batchFrames: 4,
    output: 'mp4+png',
    codec: 'avc',
    bitrate: 62_208_000,
  },
  frame: { index: 17, coeffs: [{ re: 1, im: 0 }, { re: -0.7, im: 0.1 }], view: { cx: 0.1, cy: 0.05, pxPerUnit: 1500 }, exposure: 1.1, gamma: 0.62, countScale: 9, fixedMax: 812 },
};

describe('BUILD_INFO (vite.config.ts define)', () => {
  it('carries the package version, git describe (or unknown) and an ISO build time', () => {
    expect(BUILD_INFO.version).toBe('0.1.0');
    expect(BUILD_INFO.commit).toMatch(/^([0-9a-f]{7,}|v?[\w.-]+-\d+-g[0-9a-f]{7,})(-dirty)?$|^unknown$/);
    expect(new Date(BUILD_INFO.builtAt).toISOString()).toBe(BUILD_INFO.builtAt);
  });
});

describe('buildMetadata', () => {
  it('stamps the schema, deep-copies, and leaves out absent optional parts', () => {
    const m = buildMetadata(input);
    expect(m.schema).toBe(SCHEMA);
    expect(m.fractal.coeffs).not.toBe(input.fractal.coeffs);
    expect(m.animation!.keyframes[0].view).not.toBe(input.animation!.keyframes[0].view);
    expect(m).toEqual({ schema: SCHEMA, ...input });
    const settings = buildMetadata({ ...input, kind: 'settings', output: undefined, animation: undefined, frame: undefined });
    expect(Object.keys(settings)).toEqual(['schema', 'kind', 'build', 'environment', 'fractal', 'screen', 'display']);
    expect(JSON.stringify(settings)).not.toMatch(/null|undefined/);
  });
});

describe('describe', () => {
  it('is one line with degree, coefficients, centre and zoom (1x = Reset view)', () => {
    const image = buildMetadata({ ...input, kind: 'image', animation: undefined, frame: undefined, output: { cx: 0.25, cy: -0.5, pxPerUnit: 540, width: 3840, height: 2160 } });
    expect(describeMeta(image)).toBe('Littlewood fractal · degree 20 · coefficients 1, 0.3-0.333i · centre 0.25-0.5i · zoom 1×');
    expect(describeMeta(buildMetadata(input))).toMatch(/^Littlewood fractal · frame 17 of 240 · degree 20 · coefficients 1, -0.7\+0.1i · centre 0.1\+0.05i · zoom 2.77778×$/);
  });
});

describe('parseMetadata', () => {
  it('round trip: metadata -> JSON -> parse gives the same record, at full precision', () => {
    const m = buildMetadata(input);
    const back = parseMetadata(JSON.stringify(m));
    expect(back).toEqual(m);
    expect(back.fractal.coeffs[1].re).toBe(0.1 + 0.2);
    expect(back.screen.cy).toBe(-1 / 7);
  });
  it('round trip: metadata -> JSON -> parse -> state equals the state that was recorded', () => {
    const state = stateFromMetadata(parseMetadata(JSON.stringify(buildMetadata(input))));
    expect(state).toEqual({
      coeffs: input.fractal.coeffs,
      degree: 20,
      view: { cx: 0.25, cy: -1 / 7, halfW: 2400 / 2 / 412.5, halfH: 1500 / 2 / 412.5 },
      display: { colorMode: 1, exposure: 1.25, gamma: 0.6, rings: true },
      size: { width: 3840, height: 2160 },
      animation: { keyframes: input.animation!.keyframes, frames: 240, fps: 30, loop: true, easing: 'smooth', brightness: 'locked', quality: 0.5, output: 'mp4+png' },
    });
    // And the state rebuilds the same metadata fields it came from.
    expect(stateFromMetadata(parseMetadata(JSON.stringify(buildMetadata({ ...input, kind: 'settings' }))))).toEqual(state);
  });
  it('explains what is wrong: not JSON, no schema, newer or unknown schema', () => {
    expect(() => parseMetadata('{oops')).toThrow(/not valid JSON/);
    expect(() => parseMetadata('{"degree": 3}')).toThrow(/no "schema"/);
    expect(() => parseMetadata('[1, 2]')).toThrow(MetadataError);
    expect(() => parseMetadata('{"schema": "littlewood-fractals/2"}')).toThrow(/newer version.*littlewood-fractals\/2.*reads littlewood-fractals\/1/);
    expect(() => parseMetadata('{"schema": "something-else"}')).toThrow(/unknown schema "something-else"/);
  });
  it('names the first malformed field', () => {
    const bad = (patch: (m: Record<string, any>) => void) => {
      const m = JSON.parse(JSON.stringify(buildMetadata(input)));
      patch(m);
      return () => parseMetadata(JSON.stringify(m));
    };
    expect(bad((m) => (m.fractal.coeffs[1].im = 'x'))).toThrow(/fractal\.coeffs\[1\]\.im must be a finite number/);
    expect(bad((m) => (m.fractal.degree = 33))).toThrow(/fractal\.degree must be a whole number in 1\.\.32/);
    expect(bad((m) => (m.fractal.coeffs = []))).toThrow(/1\.\.16 coefficients/);
    expect(bad((m) => (m.kind = 'movie'))).toThrow(/^invalid settings: kind must be one of/);
    expect(bad((m) => (m.screen.pxPerUnit = 0))).toThrow(/screen\.pxPerUnit must be a positive number/);
    expect(bad((m) => delete m.display)).toThrow(/display must be an object/);
    expect(bad((m) => (m.animation.easing = 'bounce'))).toThrow(/animation\.easing must be one of "smooth", "linear"/);
    expect(bad((m) => (m.animation.keyframes[1].coeffs = [{ re: 1, im: 0 }]))).toThrow(/keyframes\[1\]\.coeffs must have 2 coefficients/);
    expect(bad((m) => (m.animation.keyframes[2].view.halfW = -1))).toThrow(/keyframes\[2\]\.view\.halfW must be a positive number/);
    expect(bad((m) => delete m.animation.keyframes[0].gamma)).toThrow(/keyframes\[0\]\.gamma must be a positive number/);
  });
});

describe('environmentNotes', () => {
  it('notes a different build or GPU, nothing when both match', () => {
    const m = buildMetadata(input);
    expect(environmentNotes(m, { build, gpu })).toEqual([]);
    const notes = environmentNotes(m, { build: { ...build, commit: 'abc1234' }, gpu: { ...gpu, vendor: 'amd', description: 'RX 9060 XT' } });
    expect(notes).toHaveLength(2);
    expect(notes[0]).toMatch(/2d04924-dirty.*abc1234/);
    expect(notes[1]).toMatch(/Intel.*RX 9060 XT.*0\.1 %/);
  });
});

// ---- PNG text chunks: decoded here by an independent parser (own CRC-32, own chunk walk) ----
const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc = (b: Uint8Array) => (b.reduce((c, x) => CRC[(c ^ x) & 0xff] ^ (c >>> 8), 0xffffffff) ^ 0xffffffff) >>> 0;

function independentChunks(png: Uint8Array): { type: string; data: Uint8Array }[] {
  const out: { type: string; data: Uint8Array }[] = [];
  let o = 8;
  while (o < png.length) {
    const len = ((png[o] << 24) | (png[o + 1] << 16) | (png[o + 2] << 8) | png[o + 3]) >>> 0;
    const type = new TextDecoder('latin1').decode(png.subarray(o + 4, o + 8));
    const stored = ((png[o + 8 + len] << 24) | (png[o + 9 + len] << 16) | (png[o + 10 + len] << 8) | png[o + 11 + len]) >>> 0;
    expect(stored).toBe(crc(png.subarray(o + 4, o + 8 + len)));
    out.push({ type, data: png.subarray(o + 8, o + 8 + len) });
    o += 12 + len;
  }
  return out;
}

describe('PNG metadata chunks', () => {
  const rgba = new Uint8ClampedArray(2 * 2 * 4).fill(200);
  const json = JSON.stringify(buildMetadata(input));
  const description = describeMeta(buildMetadata(input));
  const texts = [
    { keyword: PNG_KEYWORD, text: json, utf8: true },
    { keyword: 'Software', text: 'Littlewood fractals 0.1.0 (2d04924-dirty)' },
    { keyword: 'Description', text: description },
  ];

  it('writes iTXt (uncompressed UTF-8) + tEXt (Latin-1) between IHDR and IDAT, with valid CRCs', async () => {
    const png = new Uint8Array(await (await encodePng(rgba, 2, 2, texts)).arrayBuffer());
    const cs = independentChunks(png);
    expect(cs.map((c) => c.type)).toEqual(['IHDR', 'iTXt', 'tEXt', 'tEXt', 'IDAT', 'IEND']);
    // iTXt: keyword NUL, compression flag 0, method 0, language NUL, translated keyword NUL, UTF-8 text
    const it = cs[1].data;
    const kw = it.indexOf(0);
    expect(new TextDecoder('latin1').decode(it.subarray(0, kw))).toBe(PNG_KEYWORD);
    expect([...it.subarray(kw + 1, kw + 5)]).toEqual([0, 0, 0, 0]);
    expect(JSON.parse(new TextDecoder().decode(it.subarray(kw + 5)))).toEqual(JSON.parse(json));
    const latin = (d: Uint8Array) => new TextDecoder('latin1').decode(d).split('\0');
    expect(latin(cs[2].data)).toEqual(['Software', 'Littlewood fractals 0.1.0 (2d04924-dirty)']);
    expect(latin(cs[3].data)).toEqual(['Description', description]); // '·' and '×' are Latin-1
  });

  it('readPngTexts returns what encodePng wrote; the JSON parses back to the same metadata', async () => {
    const png = new Uint8Array(await (await encodePng(rgba, 2, 2, [...texts, { keyword: 'Comment', text: 'ünïcödé ∞', utf8: true }])).arrayBuffer());
    const back = readPngTexts(png);
    expect(back).toEqual([...texts, { keyword: 'Comment', text: 'ünïcödé ∞', utf8: true }]);
    expect(parseMetadata(back[0].text)).toEqual(buildMetadata(input));
  });

  it('tEXt replaces characters outside Latin-1; readPngTexts rejects non-PNGs and corrupt text chunks', async () => {
    const png = new Uint8Array(await (await encodePng(rgba, 2, 2, [{ keyword: 'Title', text: 'z ∈ ℂ' }])).arrayBuffer());
    expect(readPngTexts(png)).toEqual([{ keyword: 'Title', text: 'z ? ?' }]);
    expect(() => readPngTexts(new TextEncoder().encode('{"schema": 1}'))).toThrow(/not a PNG/);
    const i = png.indexOf('T'.charCodeAt(0), 33); // inside the tEXt data
    png[i + 1] ^= 1;
    expect(() => readPngTexts(png)).toThrow(/CRC/);
  });

  it('metadataTexts: JSON iTXt + Software + Description, in that order', () => {
    const m = buildMetadata(input);
    expect(metadataTexts(m)).toEqual(texts);
  });

  it('without texts the file is unchanged (IHDR, IDAT, IEND)', async () => {
    const png = new Uint8Array(await (await encodePng(rgba, 2, 2)).arrayBuffer());
    expect(independentChunks(png).map((c) => c.type)).toEqual(['IHDR', 'IDAT', 'IEND']);
    expect(readPngTexts(png)).toEqual([]);
  });
});
