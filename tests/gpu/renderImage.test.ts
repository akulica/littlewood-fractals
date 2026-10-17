import { afterAll, describe, expect, it } from 'vitest';
import type { Complex } from '../../src/core/complex';
import type { Kernel } from '../../src/core/schedule';
import { fitView } from '../../src/core/view';
import { renderImage } from '../../src/gpu/engine';
import { nodeDevice } from './gpuHelpers';

const { device, errors } = await nodeDevice();
afterAll(() => device.destroy());

const LW: Complex[] = [{ re: 1, im: 0 }, { re: -1, im: 0 }];
const W = 300; // 1200 bytes per row: GPU readback pads rows to 1280, so unpadding is exercised
const H = 200;

describe.each(['perPoly', 'coop'] as Kernel[])('renderImage (%s kernel)', (kernel) => {
  it('returns tightly packed, opaque RGBA that is mirror-symmetric for real coefficients', async () => {
    const progress: number[] = [];
    const rgba = await renderImage(
      device,
      {
        kernel,
        params: { coeffs: LW, degree: 12, view: fitView(W, H, 2.2), maxIter: 32 },
        display: { colorMode: 0, exposure: 1, gamma: 0.6 },
        width: W,
        height: H,
      },
      (done, total) => progress.push(Number((done * 100n) / total)),
    );
    expect(rgba.length).toBe(W * H * 4);
    expect(progress.at(-1)).toBe(100);

    let lit = 0;
    let opaque = true;
    let asym = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const j = ((H - 1 - y) * W + x) * 4; // conjugate root lands in the mirrored row
        if (rgba[i] + rgba[i + 1] + rgba[i + 2] > 0) lit++;
        if (rgba[i + 3] !== 255) opaque = false;
        if ((rgba[i] > 0) !== (rgba[j] > 0)) asym++;
      }
    }
    expect(opaque).toBe(true);
    expect(lit).toBeGreaterThan(1000);
    expect(asym / lit).toBeLessThan(0.01); // a sheared (badly unpadded) image breaks the symmetry everywhere
    const centre = ((H / 2) * W + W / 2) * 4; // z = 0: no Littlewood root
    expect([...rgba.slice(centre, centre + 4)]).toEqual([0, 0, 0, 255]);
    expect(errors).toEqual([]);
  });
});
