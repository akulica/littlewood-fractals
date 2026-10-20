// docs/settings-format.md + docs/settings.schema.json are the contract for programs that generate settings files;
// these tests keep them in agreement with parseMetadata (the authority).
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { framesToShowKeyframes, uncoveredKeyframes } from '../src/core/animation';
import { MAX_COEFFS, MAX_DEGREE } from '../src/core/enumeration';
import { MetadataError, SCHEMA, parseMetadata, stateFromMetadata } from '../src/core/metadata';

const read = (path: string) => readFileSync(new URL(`../docs/${path}`, import.meta.url), 'utf8');
const schema = JSON.parse(read('settings.schema.json'));
const exampleText = read('examples/zoom-animation.settings.json');
const example = () => JSON.parse(exampleText);
const parse = (doc: unknown) => parseMetadata(JSON.stringify(doc));

/** Sets `value` at a dotted path ('animation.easing', 'animation.keyframes.0.view'); undefined deletes it. */
function withValue(path: string, value: unknown): unknown {
  const doc = example();
  const keys = path.split('.');
  const parent = keys.slice(0, -1).reduce((o, k) => o[k], doc);
  if (value === undefined) delete parent[keys.at(-1)!];
  else parent[keys.at(-1)!] = value;
  return doc;
}

describe('docs/examples/zoom-animation.settings.json', () => {
  it('loads, with the documented meaning', () => {
    const st = stateFromMetadata(parseMetadata(exampleText));
    expect(st.degree).toBe(20);
    expect(st.enumeration).toBe('reciprocal');
    expect(st.size).toEqual({ width: 1920, height: 1080 });
    // screen 1920x1080 at 270 px/unit = the rectangle of keyframe 0 (zoom 1x: halfH 2)
    expect(st.view).toEqual(st.animation!.keyframes[0].view);
    expect(st.animation!.keyframes).toHaveLength(3);
    expect(st.animation!.keyframes[2].exposure).toBeUndefined();
  });
  it('its frame count puts a frame exactly on every keyframe (linear: N = m(K-1)+1)', () => {
    const a = example().animation;
    expect((a.frames - 1) % (a.keyframes.length - 1)).toBe(0);
    expect(uncoveredKeyframes(a.keyframes.length, a.frames, { easing: a.easing, loop: a.loop })).toBe(0);
  });
  it('the documented frame-count table holds', () => {
    expect(uncoveredKeyframes(120, 60, { easing: 'linear', loop: false })).toBe(60); // every other keyframe
    expect(uncoveredKeyframes(120, 239, { easing: 'linear', loop: false })).toBe(0); // 2(K-1)+1
    expect(uncoveredKeyframes(120, 240, { easing: 'linear', loop: true })).toBe(0); // 2K
    expect(framesToShowKeyframes(120, { easing: 'smooth', loop: false })).toBe(174);
  });
});

describe('docs/settings.schema.json agrees with parseMetadata', () => {
  it('schema string, limits', () => {
    expect(schema.properties.schema.const).toBe(SCHEMA);
    expect(schema.properties.fractal.properties.degree.maximum).toBe(MAX_DEGREE);
    expect(schema.$defs.coeffList.maxItems).toBe(MAX_COEFFS);
    expect(() => parse(withValue('fractal.degree', MAX_DEGREE))).not.toThrow();
    expect(() => parse(withValue('fractal.degree', MAX_DEGREE + 1))).toThrow(MetadataError);
    const many = Array.from({ length: MAX_COEFFS + 1 }, () => ({ re: 1, im: 0 }));
    expect(() => parse(withValue('fractal.coeffs', many))).toThrow(MetadataError);
  });

  // Every enum value in the schema loads; a value outside it is refused.
  const enums: [string, unknown[]][] = [
    ['kind', schema.properties.kind.enum],
    ['fractal.enumeration', schema.properties.fractal.properties.enumeration.enum],
    ['display.colorMode', schema.properties.display.properties.colorMode.enum],
    ['animation.easing', schema.properties.animation.properties.easing.enum],
    ['animation.brightness', schema.properties.animation.properties.brightness.enum],
    ['animation.quality', schema.properties.animation.properties.quality.enum],
  ];
  it.each(enums)('%s: exactly the schema enum', (path, values) => {
    for (const v of values) expect(() => parse(withValue(path, v))).not.toThrow();
    expect(() => parse(withValue(path, 'nope'))).toThrow(MetadataError);
    expect(() => parse(withValue(path, 0.3))).toThrow(MetadataError);
  });

  it('required fields: removing any one of them is refused (including the ones Load ignores)', () => {
    const required: string[] = [
      ...schema.required,
      ...schema.properties.fractal.required.map((k: string) => `fractal.${k}`),
      ...schema.properties.display.required.map((k: string) => `display.${k}`),
      ...schema.properties.animation.required.map((k: string) => `animation.${k}`),
      ...schema.$defs.viewRecord.required.map((k: string) => `screen.${k}`),
      'screen.dpr',
      ...schema.properties.build.required.map((k: string) => `build.${k}`),
    ];
    for (const path of required) expect(() => parse(withValue(path, undefined)), path).toThrow();
  });

  it('optional fields: removing any one of them still loads', () => {
    for (const path of ['fractal.enumeration', 'output', 'animation', 'animation.keyframes.0.view', 'animation.keyframes.1.exposure,animation.keyframes.1.gamma']) {
      let doc = example();
      for (const p of path.split(',')) {
        doc = JSON.parse(JSON.stringify(doc));
        const keys = p.split('.');
        delete keys.slice(0, -1).reduce((o: Record<string, any>, k) => o[k], doc)[keys.at(-1)!];
      }
      expect(() => parse(doc), path).not.toThrow();
    }
  });

  it('the rules JSON Schema cannot express (and dependentRequired) are enforced', () => {
    expect(() => parse(withValue('animation.keyframes.1.coeffs', [{ re: 1, im: 0 }]))).toThrow(/must have 2 coefficients/);
    expect(() => parse(withValue('animation.keyframes.1.gamma', undefined))).toThrow(/gamma/);
    expect(() => parse(withValue('fractal.coeffs.1', '1-2i'))).toThrow(MetadataError);
    expect(() => parse(withValue('schema', 'littlewood-fractals/2'))).toThrow(/newer version/);
    expect(() => parse(withValue('generator', { name: 'x' }))).not.toThrow(); // unknown fields are ignored
  });
});
