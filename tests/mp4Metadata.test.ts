import { describe, expect, it } from 'vitest';
import { BufferTarget, EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output } from 'mediabunny';
import { buildMetadata, parseMetadata, SCHEMA } from '../src/core/metadata';
import { mp4Tags, readMp4Comment } from '../src/ui/mp4Metadata';

// No video encoder in Node: one pre-encoded (dummy) packet is enough for mediabunny to write a complete MP4, which is
// all the metadata path needs. This checks the setMetadataTags -> MP4 -> Input.getMetadataTags round trip.
async function mp4With(tags: ReturnType<typeof mp4Tags> | null): Promise<Blob> {
  const target = new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat(), target });
  const source = new EncodedVideoPacketSource('vp9');
  output.addVideoTrack(source, { frameRate: 30 });
  if (tags) output.setMetadataTags(tags);
  await output.start();
  await source.add(new EncodedPacket(new Uint8Array([0x82, 0x49, 0x83, 0x42, 0x00, 0x00, 0xf0, 0x00, 0xf6]), 'key', 0, 1 / 30), {
    decoderConfig: { codec: 'vp09.00.10.08', codedWidth: 16, codedHeight: 16 },
  });
  await output.finalize();
  return new Blob([target.buffer!], { type: 'video/mp4' });
}

describe('MP4 metadata (mediabunny comment tag)', () => {
  it('the JSON written with setMetadataTags reads back unchanged, full precision and UTF-8 intact', async () => {
    const m = buildMetadata({
      kind: 'animation',
      build: { version: '0.1.0', commit: 'abc1234', builtAt: '2026-10-20T00:00:00.000Z' },
      environment: { gpu: { vendor: 'amd', architecture: 'rdna-4', device: '', description: 'Radeon™' }, userAgent: 'test', timestamp: '2026-10-20T00:00:00.000Z' },
      fractal: { coeffs: [{ re: 1, im: 0 }, { re: 0.1 + 0.2, im: -1 / 3 }], degree: 22, maxIter: 32, tol: 1e-5, pinned: false, ordering: 'golden', kernel: 'coop' },
      screen: { cx: 0, cy: 0, pxPerUnit: 300, width: 1200, height: 800, dpr: 2 },
      output: { cx: 0, cy: 0, pxPerUnit: 405, width: 1920, height: 1080 },
      display: { colorMode: 1, exposure: 1, gamma: 0.6, countScale: 1.8225, rings: false },
      animation: { keyframes: [{ coeffs: [{ re: 1, im: 0 }, { re: -1, im: 0 }] }], frames: 300, fps: 30, loop: false, easing: 'smooth', brightness: 'locked', quality: 1, batchFrames: 8, output: 'mp4', codec: 'avc', bitrate: 15_552_000 },
    });
    const json = JSON.stringify(m);
    const comment = await readMp4Comment(await mp4With(mp4Tags('Littlewood fractal · test', json, new Date('2026-10-20T00:00:00Z'))));
    expect(comment).toBe(json);
    expect(parseMetadata(comment!)).toEqual(m);
    expect(m.schema).toBe(SCHEMA);
  });

  it('null when the file has no comment; an error for a file mediabunny cannot read', async () => {
    expect(await readMp4Comment(await mp4With(null))).toBeNull();
    await expect(readMp4Comment(new Blob([new Uint8Array(64)]))).rejects.toThrow(/not a readable MP4/);
  });
});
