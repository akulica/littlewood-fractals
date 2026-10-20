// Minimal PNG encoder (8-bit RGB, one IDAT, filter None) plus text chunks (tEXt / iTXt) for reproduction metadata.
// zlib comes from the platform's CompressionStream, so there is no canvas involved: no browser canvas-size cap
// (Safari: ~16.7 M px) and it runs in Node tests.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** Standard CRC-32 (as used by PNG chunks and zlib). */
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** A PNG text chunk. utf8: iTXt (uncompressed UTF-8, any text); otherwise tEXt, which PNG restricts to Latin-1
 * (other characters become '?'). Keywords are 1-79 Latin-1 characters. */
export interface PngText {
  keyword: string;
  text: string;
  utf8?: boolean;
}

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

function latin1(s: string): Uint8Array {
  return Uint8Array.from(s, (ch) => (ch.charCodeAt(0) < 256 ? ch.charCodeAt(0) : 63));
}

function textChunk(t: PngText): Uint8Array<ArrayBuffer> {
  if (t.keyword.length < 1 || t.keyword.length > 79) throw new RangeError('PNG text keywords are 1-79 characters');
  const keyword = latin1(t.keyword);
  if (!t.utf8) return chunk('tEXt', concat([keyword, new Uint8Array([0]), latin1(t.text)]));
  // keyword NUL, compression flag 0 + method 0, empty language tag NUL, empty translated keyword NUL, text
  return chunk('iTXt', concat([keyword, new Uint8Array([0, 0, 0, 0, 0]), new TextEncoder().encode(t.text)]));
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/**
 * Encodes tightly packed RGBA (alpha dropped: the renders are opaque) as a PNG blob. `texts` go between IHDR and
 * IDAT, so a reader finds them without walking past the image data.
 */
export async function encodePng(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number, texts: PngText[] = []): Promise<Blob> {
  if (rgba.length !== width * height * 4) throw new RangeError(`expected ${width * height * 4} bytes of RGBA, got ${rgba.length}`);
  const stride = 1 + width * 3; // filter byte + RGB
  const raw = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    let o = y * stride + 1; // raw[y * stride] = 0: filter None
    let i = y * width * 4;
    for (let x = 0; x < width; x++, i += 4, o += 3) {
      raw[o] = rgba[i];
      raw[o + 1] = rgba[i + 1];
      raw[o + 2] = rgba[i + 2];
    }
  }
  const zlib = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr.set([8, 2, 0, 0, 0], 8); // bit depth 8, colour type 2 (RGB), deflate, no filter method, no interlace
  const signature = new Uint8Array(SIGNATURE);
  return new Blob([signature, chunk('IHDR', ihdr), ...texts.map(textChunk), chunk('IDAT', zlib), chunk('IEND', new Uint8Array(0))], { type: 'image/png' });
}

/**
 * Text chunks of a PNG file: tEXt and uncompressed iTXt (compressed iTXt and zTXt are skipped: this app never writes
 * them). Throws on a non-PNG, a truncated file or a text chunk whose CRC does not match.
 */
export function readPngTexts(png: Uint8Array): PngText[] {
  if (png.length < 8 || SIGNATURE.some((b, i) => png[i] !== b)) throw new Error('not a PNG file');
  const dv = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const out: PngText[] = [];
  for (let o = 8; o + 12 <= png.length; ) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(...png.subarray(o + 4, o + 8));
    if (o + 12 + len > png.length) throw new Error('truncated PNG file');
    if (type === 'tEXt' || type === 'iTXt') {
      if (dv.getUint32(o + 8 + len) !== crc32(png.subarray(o + 4, o + 8 + len))) throw new Error(`corrupt PNG ${type} chunk (CRC mismatch)`);
      const data = png.subarray(o + 8, o + 8 + len);
      const z = data.indexOf(0);
      if (z > 0) {
        const keyword = String.fromCharCode(...data.subarray(0, z));
        if (type === 'tEXt') {
          out.push({ keyword, text: String.fromCharCode(...data.subarray(z + 1)) });
        } else if (data[z + 1] === 0) {
          const lang = data.indexOf(0, z + 3); // after compression flag and method
          const translated = lang < 0 ? -1 : data.indexOf(0, lang + 1);
          if (translated >= 0) out.push({ keyword, text: new TextDecoder().decode(data.subarray(translated + 1)), utf8: true });
        }
      }
    }
    if (type === 'IEND') break;
    o += 12 + len;
  }
  return out;
}
