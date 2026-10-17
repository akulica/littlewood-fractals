import { describe, expect, it } from 'vitest';
import { crc32, encodePng } from '../src/core/png';

const u32 = (b: Uint8Array, o: number) => new DataView(b.buffer, b.byteOffset).getUint32(o);
const ascii = (b: Uint8Array, o: number) => String.fromCharCode(...b.subarray(o, o + 4));

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Splits a PNG into chunks, verifying every CRC. */
function chunks(png: Uint8Array): { type: string; data: Uint8Array }[] {
  const out: { type: string; data: Uint8Array }[] = [];
  for (let o = 8; o < png.length; ) {
    const len = u32(png, o);
    const type = ascii(png, o + 4);
    const data = png.subarray(o + 8, o + 8 + len);
    expect(u32(png, o + 8 + len)).toBe(crc32(png.subarray(o + 4, o + 8 + len)));
    out.push({ type, data });
    o += 12 + len;
  }
  return out;
}

describe('crc32', () => {
  it('matches the standard check value and the constant IEND chunk CRC', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
    expect(crc32(new TextEncoder().encode('IEND'))).toBe(0xae426082);
  });
});

describe('encodePng', () => {
  it('writes a valid 8-bit RGB PNG whose pixels round-trip', async () => {
    const w = 3;
    const h = 2;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) rgba.set([i * 40, 255 - i * 40, i * 7, 255], i * 4);
    const png = new Uint8Array(await (await encodePng(rgba, w, h)).arrayBuffer());

    expect([...png.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    const cs = chunks(png);
    expect(cs.map((c) => c.type)).toEqual(['IHDR', 'IDAT', 'IEND']);
    const ihdr = cs[0].data;
    expect([u32(ihdr, 0), u32(ihdr, 4), ...ihdr.subarray(8, 13)]).toEqual([w, h, 8, 2, 0, 0, 0]);

    const raw = await inflate(cs[1].data);
    expect(raw.length).toBe(h * (1 + w * 3));
    for (let y = 0; y < h; y++) {
      expect(raw[y * (1 + w * 3)]).toBe(0); // filter type None
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        expect([...raw.subarray(y * (1 + w * 3) + 1 + x * 3, y * (1 + w * 3) + 4 + x * 3)]).toEqual([...rgba.subarray(i * 4, i * 4 + 3)]);
      }
    }
  });

  it('rejects a buffer of the wrong size', async () => {
    await expect(encodePng(new Uint8ClampedArray(10), 3, 2)).rejects.toThrow(/expected 24 bytes/);
  });
});
