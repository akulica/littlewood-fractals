// MP4 metadata tags via mediabunny: the reproduction JSON goes into the `comment` tag (the ©cmt atom, shown by
// `ffprobe -show_format`), a one-line description into `title`. No DOM here, so it runs in Node tests too.
import { ALL_FORMATS, BlobSource, Input, type MetadataTags } from 'mediabunny';

/** Tags for Output.setMetadataTags (call before output.start()). */
export function mp4Tags(title: string, json: string, date: Date): MetadataTags {
  return { title, comment: json, date };
}

/** The comment tag of an MP4 (or any container mediabunny reads), or null if it has none. */
export async function readMp4Comment(file: Blob): Promise<string | null> {
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  try {
    if (!(await input.canRead())) throw new Error('not a readable MP4 file');
    return (await input.getMetadataTags()).comment ?? null;
  } finally {
    input.dispose();
  }
}
