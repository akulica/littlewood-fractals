// Where rendered animation frames go: an H.264 MP4 (WebCodecs encoder, muxed by mediabunny) and/or a folder of
// PNG frames. Files are written as frames arrive, so a long video never sits in memory where the browser can
// stream to disk (File System Access API: Chrome/Edge); elsewhere the MP4 is built in memory and downloaded.
import { BufferTarget, Mp4OutputFormat, Output, Quality, StreamTarget, VideoSample, VideoSampleSource, canEncodeVideo } from 'mediabunny';
import { videoSizeError } from '../core/animation';
import { encodePng, type PngText } from '../core/png';
import { PngPool, defaultPngWorkers } from '../core/pngPool';
import { mp4Tags } from './mp4Metadata';
import { spawnWebPngEncoder } from './pngWorker';

export { readMp4Comment } from './mp4Metadata';

export type OutputKind = 'mp4' | 'mp4+png' | 'png';

// File System Access pickers are not in TypeScript's DOM lib yet.
interface PickerWindow {
  showSaveFilePicker?: (o: { suggestedName?: string; types?: { description: string; accept: Record<string, string[]> }[] }) => Promise<FileSystemFileHandle>;
  showDirectoryPicker?: (o?: { mode?: 'read' | 'readwrite' }) => Promise<FileSystemDirectoryHandle>;
}
const pickers = window as unknown as PickerWindow;

/** PNG frame folders need showDirectoryPicker (Chrome/Edge); the option is hidden elsewhere. */
export const canWriteFolders = (): boolean => typeof pickers.showDirectoryPicker === 'function';

/** High by default: ~0.25 bit per pixel per frame (1080p30: 15.6 Mbit/s). Fine root dust is hard on H.264. */
export const mp4Bitrate = (width: number, height: number, fps: number): number => Math.round(width * height * fps * 0.25);

/** Human-readable reason this browser cannot encode the MP4, or null. Checked before any file is created. */
export async function mp4Error(width: number, height: number, fps: number): Promise<string | null> {
  const odd = videoSizeError(width, height);
  if (odd) return odd;
  if (typeof VideoEncoder === 'undefined') return 'this browser has no WebCodecs video encoder: choose PNG frames, or use Chrome/Edge';
  const ok = await canEncodeVideo('avc', { width, height, frameRate: fps, quality: new Quality({ bitrate: mp4Bitrate(width, height, fps) }) });
  return ok
    ? null
    : `this browser's H.264 encoder rejects ${width}×${height} (hardware encoders often stop near 4096 px per side): try a smaller size or PNG frames`;
}

export interface AnimationWriter {
  /** Frames arrive in order; the returned promise applies the encoder's / disk's backpressure to the renderer.
   * `texts`: the PNG frame's metadata chunks (made only when PNG frames are written). */
  frame(rgba: Uint8ClampedArray<ArrayBuffer>, index: number, texts?: () => PngText[]): Promise<void>;
  /** Completes the files; returns a short summary for the status line. */
  finish(): Promise<string>;
  /** After an error or Cancel: drops the unfinished MP4 (PNG frames written so far stay). */
  cancel(): Promise<void>;
}

export interface WriterOptions {
  kind: OutputKind;
  width: number;
  height: number;
  fps: number;
  /** File name without extension. */
  name: string;
  /** Reproduction metadata (core/metadata): JSON for the MP4 comment tag and the PNG folder's animation.json. */
  metadata?: { title: string; json: string; date: Date };
}

/**
 * Asks for the destination and opens the outputs. Call it straight from the click handler, before any other await
 * that could outlast the click's user activation: the pickers refuse to open without it. Rejects with an
 * AbortError DOMException when the user dismisses a picker.
 */
export async function openAnimationWriter(o: WriterOptions): Promise<AnimationWriter> {
  // One picker per click: a picker consumes the activation, so MP4 + PNG goes into the one chosen folder.
  let dir: FileSystemDirectoryHandle | null = null;
  let mp4File: FileSystemFileHandle | null = null;
  if (o.kind !== 'mp4') {
    if (!pickers.showDirectoryPicker) throw new Error('PNG frames need a browser that can write folders (Chrome/Edge)');
    dir = await pickers.showDirectoryPicker({ mode: 'readwrite' });
    if (o.kind === 'mp4+png') mp4File = await dir.getFileHandle(`${o.name}.mp4`, { create: true });
  } else if (pickers.showSaveFilePicker) {
    mp4File = await pickers.showSaveFilePicker({ suggestedName: `${o.name}.mp4`, types: [{ description: 'MP4 video', accept: { 'video/mp4': ['.mp4'] } }] });
  }
  const video = o.kind === 'png' ? null : await openMp4(o, mp4File, dir);
  if (dir && o.metadata) {
    // Written first: a cancelled render still documents what its frames are.
    const file = await dir.getFileHandle('animation.json', { create: true });
    const w = await file.createWritable();
    await w.write(o.metadata.json);
    await w.close();
  }
  let pngs = 0;
  const writePng = async (png: Blob | Uint8Array<ArrayBuffer>, index: number) => {
    const file = await dir!.getFileHandle(`frame_${String(index).padStart(5, '0')}.png`, { create: true });
    const w = await file.createWritable();
    await w.write(png);
    await w.close();
    pngs++;
  };
  // PNG frames are encoded in Web Workers, overlapped with rendering and disk writes; without Workers (or if they
  // fail to start) the encoding stays on this thread, one frame at a time.
  let pool: PngPool<number> | null = null;
  if (dir) {
    try {
      pool = new PngPool<number>(spawnWebPngEncoder, defaultPngWorkers(navigator.hardwareConcurrency), (png, index) => writePng(png, index));
      await pool.check();
    } catch {
      pool?.terminate();
      pool = null;
    }
  }
  return {
    async frame(rgba, index, texts) {
      if (video) await video.add(rgba, index);
      if (!dir) return;
      // The pool transfers the buffer to a worker: give it a copy while the video encoder may still hold a reference.
      if (pool) await pool.submit({ rgba: video ? rgba.slice() : rgba, width: o.width, height: o.height, texts: texts?.() ?? [] }, index);
      else await writePng(await encodePng(rgba, o.width, o.height, texts?.()), index);
    },
    async finish() {
      await pool?.finish();
      const parts: string[] = [];
      if (video) parts.push(await video.finish());
      if (dir) parts.push(`${pngs} PNG frames in ${dir.name}/`);
      return parts.join(' + ');
    },
    async cancel() {
      pool?.terminate();
      await video?.cancel();
    },
  };
}

async function openMp4(o: WriterOptions, file: FileSystemFileHandle | null, dir: FileSystemDirectoryHandle | null) {
  const writable = file ? await file.createWritable() : null;
  // chunked: 16 MiB writes instead of one per encoded packet (each File System Access write has a fixed cost).
  const target = writable ? new StreamTarget(writable, { chunked: true }) : new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat(), target });
  const source = new VideoSampleSource({ codec: 'avc', quality: new Quality({ bitrate: mp4Bitrate(o.width, o.height, o.fps) }) });
  output.addVideoTrack(source, { frameRate: o.fps });
  if (o.metadata) output.setMetadataTags(mp4Tags(o.metadata.title, o.metadata.json, o.metadata.date));
  await output.start();
  return {
    async add(rgba: Uint8ClampedArray<ArrayBuffer>, index: number) {
      const sample = new VideoSample(rgba, { format: 'RGBA', codedWidth: o.width, codedHeight: o.height, timestamp: index / o.fps, duration: 1 / o.fps });
      try {
        await source.add(sample);
      } finally {
        sample.close();
      }
    },
    async finish(): Promise<string> {
      await output.finalize();
      if (target instanceof BufferTarget) {
        const blob = new Blob([target.buffer!], { type: 'video/mp4' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${o.name}.mp4`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
        return `downloaded ${a.download} (${(blob.size / 2 ** 20).toFixed(1)} MB)`;
      }
      return `saved ${file!.name}`;
    },
    async cancel() {
      // mediabunny closes the stream on cancel, which commits the partial file: delete it instead of leaving
      // an unplayable MP4 behind (best effort; FileSystemHandle.remove() is Chrome-only).
      if (output.state === 'started' || output.state === 'pending') await output.cancel().catch(() => {});
      if (!file) return;
      try {
        if (dir) await dir.removeEntry(file.name);
        else await (file as FileSystemFileHandle & { remove?: () => Promise<void> }).remove?.();
      } catch {
        // the partial file stays; nothing else to do
      }
    },
  };
}
