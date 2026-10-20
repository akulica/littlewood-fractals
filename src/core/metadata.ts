// Reproduction metadata, schema "littlewood-fractals/1": everything needed to render an export again (and to load it
// back into the app). Pure: the UI supplies build info, GPU and clock; this module only assembles, describes and
// validates. Stored as a PNG iTXt chunk, the MP4 comment tag, animation.json next to PNG frames, or a settings file.
import type { Brightness, Easing, Keyframe } from './animation';
import { formatComplex, type Complex } from './complex';
import { MAX_COEFFS, MAX_DEGREE } from './enumeration';
import type { PngText } from './png';
import type { Enumeration } from './reciprocal';
import type { ViewRect } from './view';

export const SCHEMA = 'littlewood-fractals/1';
/** PNG iTXt keyword of the JSON. */
export const PNG_KEYWORD = 'littlewood-fractals';

/** image: Save PNG. animation: a whole render (MP4, animation.json). frame: one PNG frame of a render.
 * settings: Save settings (.json), no output. */
export type MetadataKind = 'image' | 'animation' | 'frame' | 'settings';

export interface BuildInfo {
  version: string;
  /** git describe --always --dirty at build / dev-server start, or 'unknown'. */
  commit: string;
  builtAt: string;
}

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface Environment {
  gpu: GpuInfo;
  userAgent: string;
  /** ISO 8601 */
  timestamp: string;
}

export interface FractalRecord {
  /** Full double precision (JSON numbers round-trip exactly). */
  coeffs: Complex[];
  degree: number;
  maxIter: number;
  /** Backward-error stop of the Aberth iteration. */
  tol: number;
  /** Leading digit pinned (symmetric sets: same picture, k times less work). */
  pinned: boolean;
  ordering: string;
  /** Splat kernel actually used (f32 rounding differs between them): perPoly or coop; animations splatFrames or
   * splatFramesCoop (their warm-starting variants). */
  kernel: string;
  /** reciprocal (z and 1/z from one solve) or plain, as actually used (a pinned set without a shared group table
   * runs plain); f32 rounding differs slightly. Absent in files made before it existed (those were plain). */
  enumeration?: Enumeration;
}

/** A view on a width x height pixel grid (pxPerUnit in that grid's pixels). */
export interface ViewRecord {
  cx: number;
  cy: number;
  pxPerUnit: number;
  width: number;
  height: number;
}

export interface ScreenRecord extends ViewRecord {
  /** Device pixels per CSS pixel of the canvas (the handles' size is in CSS pixels). */
  dpr: number;
}

export interface DisplayRecord {
  colorMode: 0 | 1;
  exposure: number;
  gamma: number;
  /** Root counts are multiplied by this before tone-mapping (export: (export px per screen px)^2). */
  countScale: number;
  /** Coefficient rings drawn into the output. */
  rings: boolean;
  /** > 0: tone-mapped against this count (animation frames with locked brightness). */
  fixedMax?: number;
}

export interface AnimationRecord {
  /** Exactly as stored in the app: coefficients plus optional view rectangle and exposure/gamma. */
  keyframes: Keyframe[];
  frames: number;
  fps: number;
  loop: boolean;
  easing: Easing;
  brightness: Brightness;
  quality: number;
  /** Frames per splatFrames / splatFramesCoop batch (warm-start chains are this long, so it affects f32 rounding). */
  batchFrames: number;
  /** mp4 | mp4+png | png */
  output: string;
  codec: string | null;
  bitrate: number | null;
}

/** One PNG frame of an animation: what that frame was rendered with. */
export interface FrameRecord {
  index: number;
  coeffs: Complex[];
  view: { cx: number; cy: number; pxPerUnit: number };
  exposure: number;
  gamma: number;
  countScale: number;
  fixedMax: number;
}

export interface Metadata {
  schema: typeof SCHEMA;
  kind: MetadataKind;
  build: BuildInfo;
  environment: Environment;
  fractal: FractalRecord;
  /** The live view when the output was made. */
  screen: ScreenRecord;
  /** The output's own view, pxPerUnit in output pixels (animation and frame: frame 0's; `frame` holds this frame's).
   * settings: the export size and view Save PNG would use now. */
  output?: ViewRecord;
  display: DisplayRecord;
  animation?: AnimationRecord;
  frame?: FrameRecord;
}

export type MetadataInput = Omit<Metadata, 'schema'>;

const copyKey = (k: Keyframe): Keyframe => ({
  coeffs: k.coeffs.map((c) => ({ re: c.re, im: c.im })),
  ...(k.view && { view: { ...k.view } }),
  ...(k.exposure !== undefined && k.gamma !== undefined && { exposure: k.exposure, gamma: k.gamma }),
});

/** Assembles the record with a fixed key order and deep copies (later UI changes cannot leak in); optional parts
 * that are undefined are left out, so the JSON has no nulls standing for "not applicable". */
export function buildMetadata(m: MetadataInput): Metadata {
  return {
    schema: SCHEMA,
    kind: m.kind,
    build: { ...m.build },
    environment: { gpu: { ...m.environment.gpu }, userAgent: m.environment.userAgent, timestamp: m.environment.timestamp },
    fractal: { ...m.fractal, coeffs: m.fractal.coeffs.map((c) => ({ re: c.re, im: c.im })) },
    screen: { ...m.screen },
    ...(m.output && { output: { ...m.output } }),
    display: { ...m.display },
    ...(m.animation && { animation: { ...m.animation, keyframes: m.animation.keyframes.map(copyKey) } }),
    ...(m.frame && { frame: { ...m.frame, coeffs: m.frame.coeffs.map((c) => ({ re: c.re, im: c.im })), view: { ...m.frame.view } } }),
  };
}

const short = (x: number) => Number(x.toPrecision(6)).toString();

/** One human-readable line (PNG Description, MP4 title): degree, coefficients, centre and zoom. Zoom 1x is the
 * app's Reset view (radius 2 across the shorter side). */
export function describe(m: Metadata): string {
  const v = m.frame ? { ...m.frame.view, width: m.output?.width ?? 1, height: m.output?.height ?? 1 } : (m.output ?? m.screen);
  const zoom = (v.pxPerUnit * 4) / Math.min(v.width, v.height);
  const coeffs = (m.frame?.coeffs ?? m.fractal.coeffs).map(formatComplex).join(', ');
  // 8 significant digits: deep zooms need more than formatComplex's 3 decimals to locate the centre.
  const centre = `${Number(v.cx.toPrecision(8))}${v.cy < 0 ? '-' : '+'}${Number(Math.abs(v.cy).toPrecision(8))}i`;
  const what = m.frame ? `frame ${m.frame.index} of ${m.animation?.frames ?? '?'} · ` : m.animation ? `animation, ${m.animation.frames} frames · ` : '';
  return `Littlewood fractal · ${what}degree ${m.fractal.degree} · coefficients ${coeffs} · centre ${centre} · zoom ${short(zoom)}×`;
}

/** PNG chunks for `m`: the JSON as iTXt (UTF-8), plus the standard tEXt Software and Description. */
export function metadataTexts(m: Metadata): PngText[] {
  return [
    { keyword: PNG_KEYWORD, text: JSON.stringify(m), utf8: true },
    { keyword: 'Software', text: `Littlewood fractals ${m.build.version} (${m.build.commit})` },
    { keyword: 'Description', text: describe(m) },
  ];
}

export class MetadataError extends Error {}

type Obj = Record<string, unknown>;
const isObj = (x: unknown): x is Obj => typeof x === 'object' && x !== null && !Array.isArray(x);

const at = (path: string, key: string) => (path ? `${path}.${key}` : key);
function fail(path: string, what: string): never {
  throw new MetadataError(`invalid settings: ${path} ${what}`);
}
function obj(x: unknown, path: string): Obj {
  if (!isObj(x)) fail(path, 'must be an object');
  return x;
}
function num(o: Obj, key: string, path: string, check: (v: number) => boolean = () => true, what = 'a finite number'): number {
  const v = o[key];
  if (typeof v !== 'number' || !Number.isFinite(v) || !check(v)) fail(at(path, key), `must be ${what}`);
  return v;
}
const positive = (o: Obj, key: string, path: string) => num(o, key, path, (v) => v > 0, 'a positive number');
function int(o: Obj, key: string, path: string, min: number, max: number): number {
  return num(o, key, path, (v) => Number.isInteger(v) && v >= min && v <= max, `a whole number in ${min}..${max}`);
}
function str(o: Obj, key: string, path: string): string {
  const v = o[key];
  if (typeof v !== 'string') fail(at(path, key), 'must be a string');
  return v;
}
function bool(o: Obj, key: string, path: string): boolean {
  const v = o[key];
  if (typeof v !== 'boolean') fail(at(path, key), 'must be true or false');
  return v;
}
function oneOf<T extends string | number>(o: Obj, key: string, path: string, allowed: readonly T[]): T {
  const v = o[key] as T;
  if (!allowed.includes(v)) fail(at(path, key), `must be one of ${allowed.map((a) => JSON.stringify(a)).join(', ')}`);
  return v;
}
function coeffList(x: unknown, path: string): Complex[] {
  if (!Array.isArray(x) || x.length < 1 || x.length > MAX_COEFFS) fail(path, `must be a list of 1..${MAX_COEFFS} coefficients`);
  return x.map((c, i) => {
    const o = obj(c, `${path}[${i}]`);
    return { re: num(o, 're', `${path}[${i}]`), im: num(o, 'im', `${path}[${i}]`) };
  });
}
function viewRecord(x: unknown, path: string): ViewRecord {
  const o = obj(x, path);
  return { cx: num(o, 'cx', path), cy: num(o, 'cy', path), pxPerUnit: positive(o, 'pxPerUnit', path), width: int(o, 'width', path, 1, 1 << 16), height: int(o, 'height', path, 1, 1 << 16) };
}
function rect(x: unknown, path: string): ViewRect {
  const o = obj(x, path);
  return { cx: num(o, 'cx', path), cy: num(o, 'cy', path), halfW: positive(o, 'halfW', path), halfH: positive(o, 'halfH', path) };
}

const KINDS = ['image', 'animation', 'frame', 'settings'] as const;
const EASINGS = ['smooth', 'linear'] as const;
const BRIGHTNESS = ['locked', 'frame0', 'perFrame'] as const;
const QUALITIES = [1, 0.5, 0.25, 0.1] as const;
const ENUMERATIONS = ['reciprocal', 'plain'] as const;

/**
 * Parses and validates metadata JSON. Throws MetadataError with a message fit for the status line: not JSON,
 * no schema, a schema from a newer version, or the first malformed field (with its path).
 */
export function parseMetadata(json: string): Metadata {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new MetadataError('not a littlewood-fractals settings file (the metadata is not valid JSON)');
  }
  if (!isObj(raw) || typeof raw.schema !== 'string') throw new MetadataError('not a littlewood-fractals settings file (no "schema" field)');
  if (raw.schema !== SCHEMA) {
    const version = /^littlewood-fractals\/(\d+)$/.exec(raw.schema);
    throw new MetadataError(
      version
        ? `made by a newer version of the app (schema ${raw.schema}); this build reads ${SCHEMA} only`
        : `unknown schema "${raw.schema}" (expected ${SCHEMA})`,
    );
  }
  const build = obj(raw.build, 'build');
  const env = obj(raw.environment, 'environment');
  const gpu = obj(env.gpu, 'environment.gpu');
  const fractal = obj(raw.fractal, 'fractal');
  const screen = obj(raw.screen, 'screen');
  const display = obj(raw.display, 'display');
  const coeffs = coeffList(fractal.coeffs, 'fractal.coeffs');
  const m: Metadata = {
    schema: SCHEMA,
    kind: oneOf(raw, 'kind', '', KINDS),
    build: { version: str(build, 'version', 'build'), commit: str(build, 'commit', 'build'), builtAt: str(build, 'builtAt', 'build') },
    environment: {
      gpu: { vendor: str(gpu, 'vendor', 'environment.gpu'), architecture: str(gpu, 'architecture', 'environment.gpu'), device: str(gpu, 'device', 'environment.gpu'), description: str(gpu, 'description', 'environment.gpu') },
      userAgent: str(env, 'userAgent', 'environment'),
      timestamp: str(env, 'timestamp', 'environment'),
    },
    fractal: {
      coeffs,
      degree: int(fractal, 'degree', 'fractal', 1, MAX_DEGREE),
      maxIter: int(fractal, 'maxIter', 'fractal', 1, 1 << 16),
      tol: positive(fractal, 'tol', 'fractal'),
      pinned: bool(fractal, 'pinned', 'fractal'),
      ordering: str(fractal, 'ordering', 'fractal'),
      kernel: str(fractal, 'kernel', 'fractal'),
    },
    screen: { ...viewRecord(screen, 'screen'), dpr: positive(screen, 'dpr', 'screen') },
    display: {
      colorMode: oneOf(display, 'colorMode', 'display', [0, 1] as const),
      exposure: positive(display, 'exposure', 'display'),
      gamma: positive(display, 'gamma', 'display'),
      countScale: positive(display, 'countScale', 'display'),
      rings: bool(display, 'rings', 'display'),
    },
  };
  if (fractal.enumeration !== undefined) m.fractal.enumeration = oneOf(fractal, 'enumeration', 'fractal', ENUMERATIONS);
  if (display.fixedMax !== undefined) m.display.fixedMax = num(display, 'fixedMax', 'display', (v) => v >= 0, 'a number ≥ 0');
  if (raw.output !== undefined) m.output = viewRecord(raw.output, 'output');
  if (raw.animation !== undefined) {
    const a = obj(raw.animation, 'animation');
    if (!Array.isArray(a.keyframes)) fail('animation.keyframes', 'must be a list');
    const keyframes = a.keyframes.map((k, i): Keyframe => {
      const path = `animation.keyframes[${i}]`;
      const o = obj(k, path);
      const key: Keyframe = { coeffs: coeffList(o.coeffs, `${path}.coeffs`) };
      if (key.coeffs.length !== coeffs.length) fail(`${path}.coeffs`, `must have ${coeffs.length} coefficients, like fractal.coeffs`);
      if (o.view !== undefined) key.view = rect(o.view, `${path}.view`);
      if (o.exposure !== undefined || o.gamma !== undefined) {
        key.exposure = positive(o, 'exposure', path);
        key.gamma = positive(o, 'gamma', path);
      }
      return key;
    });
    m.animation = {
      keyframes,
      frames: int(a, 'frames', 'animation', 1, 100_000),
      fps: int(a, 'fps', 'animation', 1, 120),
      loop: bool(a, 'loop', 'animation'),
      easing: oneOf(a, 'easing', 'animation', EASINGS),
      brightness: oneOf(a, 'brightness', 'animation', BRIGHTNESS),
      quality: oneOf(a, 'quality', 'animation', QUALITIES),
      batchFrames: int(a, 'batchFrames', 'animation', 1, 1 << 10),
      output: str(a, 'output', 'animation'),
      codec: a.codec === null ? null : str(a, 'codec', 'animation'),
      bitrate: a.bitrate === null ? null : positive(a, 'bitrate', 'animation'),
    };
  }
  if (raw.frame !== undefined) {
    const f = obj(raw.frame, 'frame');
    const v = obj(f.view, 'frame.view');
    m.frame = {
      index: int(f, 'index', 'frame', 0, 100_000),
      coeffs: coeffList(f.coeffs, 'frame.coeffs'),
      view: { cx: num(v, 'cx', 'frame.view'), cy: num(v, 'cy', 'frame.view'), pxPerUnit: positive(v, 'pxPerUnit', 'frame.view') },
      exposure: positive(f, 'exposure', 'frame'),
      gamma: positive(f, 'gamma', 'frame'),
      countScale: positive(f, 'countScale', 'frame'),
      fixedMax: num(f, 'fixedMax', 'frame', (v) => v >= 0, 'a number ≥ 0'),
    };
  }
  return m;
}

/** What Load restores into the app. */
export interface LoadedState {
  coeffs: Complex[];
  degree: number;
  /** The recorded screen's visible rectangle: fitted to the current canvas, it frames the same region. */
  view: ViewRect;
  display: { colorMode: 0 | 1; exposure: number; gamma: number; rings: boolean };
  /** Absent in files made before the option existed: the app keeps its current setting. */
  enumeration?: Enumeration;
  /** Export / animation size (Save PNG W x H). */
  size?: { width: number; height: number };
  animation?: { keyframes: Keyframe[]; frames: number; fps: number; loop: boolean; easing: Easing; brightness: Brightness; quality: number; output: string };
}

/** The app state recorded in `m` (the screen at export time, not a single frame's interpolated state). */
export function stateFromMetadata(m: Metadata): LoadedState {
  const s = m.screen;
  const state: LoadedState = {
    coeffs: m.fractal.coeffs.map((c) => ({ ...c })),
    degree: m.fractal.degree,
    view: { cx: s.cx, cy: s.cy, halfW: s.width / 2 / s.pxPerUnit, halfH: s.height / 2 / s.pxPerUnit },
    display: { colorMode: m.display.colorMode, exposure: m.display.exposure, gamma: m.display.gamma, rings: m.display.rings },
  };
  if (m.fractal.enumeration) state.enumeration = m.fractal.enumeration;
  if (m.output) state.size = { width: m.output.width, height: m.output.height };
  if (m.animation) {
    const { keyframes, frames, fps, loop, easing, brightness, quality, output } = m.animation;
    state.animation = { keyframes: keyframes.map(copyKey), frames, fps, loop, easing, brightness, quality, output };
  }
  return state;
}

/** Non-blocking notes when the file was made by another build or on another GPU (results may differ slightly). */
export function environmentNotes(m: Metadata, current: { build: BuildInfo; gpu: GpuInfo }): string[] {
  const notes: string[] = [];
  if (m.build.commit !== current.build.commit || m.build.version !== current.build.version) {
    notes.push(`made by build ${m.build.version} (${m.build.commit}); this is ${current.build.version} (${current.build.commit})`);
  }
  const gpuName = (g: GpuInfo) => [g.vendor, g.architecture, g.device, g.description].filter(Boolean).join(' ') || 'unknown GPU';
  if (gpuName(m.environment.gpu) !== gpuName(current.gpu)) {
    notes.push(`rendered on ${gpuName(m.environment.gpu)}; this is ${gpuName(current.gpu)}: up to ~0.1 % of roots may land one pixel over`);
  }
  return notes;
}
