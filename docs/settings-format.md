# Settings file format (`littlewood-fractals/1`)

This is how another program can write a settings file that **Load…** (or drag-and-drop) accepts. It covers still settings and animations. The same JSON is embedded in every export: the PNG `iTXt` chunk `littlewood-fractals`, the MP4 `comment` tag and `animation.json`. Any export is therefore also a working example.

- Machine-readable schema: [`settings.schema.json`](settings.schema.json) (JSON Schema 2020-12).
- Complete example: [`examples/zoom-animation.settings.json`](examples/zoom-animation.settings.json). It is a 3-keyframe zoom into the Littlewood set followed by a coefficient move, 241 frames.
- The app itself is the authority: `parseMetadata` in `src/core/metadata.ts`. The test `tests/settingsSpec.test.ts` keeps this document, the schema and the parser in agreement.

## Rules for writers

1. **Encoding.** UTF-8 JSON, at most 16 MB. Numbers are plain JSON numbers (IEEE doubles; no `NaN` or `Infinity`, and no numbers written as strings). Doubles round-trip exactly, so write full precision.
2. **Complex numbers** are objects `{ "re": …, "im": … }`, never strings like `"1-2i"`.
3. **Every required field must be present**, even the ones Load ignores (marked *ignored* below). Use the suggested values for those.
4. **Unknown fields are ignored** and not kept. Put generator details in `build` or `environment`; they appear in the Load note.
5. **Validation is all-or-nothing.** The first bad field rejects the whole file with a message naming its path, e.g. `invalid settings: fractal.degree must be a whole number in 1..32`.
6. **Two rules the JSON Schema cannot express:**
   - Every `animation.keyframes[i].coeffs` must have **the same length as `fractal.coeffs`**. The parser enforces this.
   - An animation needs **at least 2 keyframes** before Render is possible. The parser accepts 0 or 1, and the user then has to add more.

## Top level

| Field | Required | Load uses it | Value |
|---|---|---|---|
| `schema` | yes | yes | `"littlewood-fractals/1"` exactly. Other `littlewood-fractals/N` values are refused as "made by a newer version". |
| `kind` | yes | no | `"settings"` for generated files (`image`, `animation` and `frame` are what exports write; all four load). |
| `build` | yes | note only | `{ "version", "commit", "builtAt" }`, all strings. |
| `environment` | yes | note only | `{ "gpu": { "vendor", "architecture", "device", "description" }, "userAgent", "timestamp" }`, all strings (they may be empty). |
| `fractal` | yes | yes | See below. |
| `unverified` | no | yes | `{ "skipOverflowGuard": true }` turns on an option that fails the A/B tests. Omit it to have every such option off. |
| `screen` | yes | yes | The live view, see [Views](#views). |
| `output` | no | width and height only | `{ cx, cy, pxPerUnit, width, height }`: the Save PNG / animation size. |
| `display` | yes | yes | See below. |
| `animation` | no | yes | See below. Without it, Load keeps the user's current keyframes if they have the same coefficient count, and clears them otherwise. |
| `frame` | no | no | Written into PNG frames by the app. Omit it. |

### `fractal`

| Field | Load uses it | Value |
|---|---|---|
| `coeffs` | yes | 1–16 complex numbers: the set *S*. The picture shows the roots of every polynomial a₀ + a₁z + … + a_d z^d with every aᵢ in *S*, so there are k^(d+1) polynomials for k = `coeffs.length`. |
| `degree` | yes | Integer d, 1–32. |
| `enumeration` | yes | `"reciprocal"` (≈2× faster) or `"plain"`. Optional; if absent the app keeps its current setting. |
| `maxIter` | *ignored* | Integer 1–65536. Suggested: `32`. |
| `tol` | *ignored* | Number > 0. Suggested: `1e-5`. |
| `pinned` | *ignored* | Boolean. Suggested: `false`. |
| `ordering` | *ignored* | String. Suggested: `"golden"`. |
| `kernel` | *ignored* | String. Suggested: `"perPoly"`. The app picks the kernel for its GPU. |

**Cost.** Render time grows with k^(d+1). Some reference points for one full render: k = 2 at d = 20 is 2 M polynomials, well under a second; k = 2 at d = 24 is 34 M; k = 4 at d = 12 is 67 M; k = 3 at d = 16 is 129 M. An animation multiplies the time by `frames`.

### `display`

| Field | Load uses it | Value |
|---|---|---|
| `colorMode` | yes | `1`: hue by low-order coefficients. `0`: density (inferno). |
| `exposure` | yes | Number > 0. The slider covers **0.2–3**, and values outside are clamped on Load. |
| `gamma` | yes | Number > 0. The slider covers **0.2–1.5**, and values outside are clamped on Load. |
| `rings` | yes | Boolean: draw coefficient rings into exports. |
| `countScale` | *ignored* | Number > 0. Suggested: `1`. |
| `fixedMax` | *ignored* | Optional number ≥ 0. Omit it. |

### `animation`

| Field | Load uses it | Value |
|---|---|---|
| `keyframes` | yes | A list of [keyframes](#keyframes). Each keyframe gets an equal share of the timeline. |
| `frames` | yes | Integer 1–100 000. The app renders at least 2. See [Choosing the frame count](#choosing-the-frame-count). |
| `fps` | yes | Integer 1–120. |
| `loop` | yes | Boolean. If `true` the path returns to keyframe 0, and the last frame stops one step before it so the video loops without a stutter. |
| `easing` | yes | `"linear"` or `"smooth"`. Smooth applies smoothstep to the **whole** timeline, not to each segment. |
| `brightness` | yes | `"locked"`: density per area is locked to frame 0, so a zoom keeps its brightness. `"frame0"`: every frame uses frame 0's densest pixel, so zooming in darkens. `"perFrame"`: each frame is scaled to its own densest pixel, so brightness may pulse. |
| `quality` | yes | `1`, `0.5`, `0.25` or `0.1`: the fraction of polynomials solved per frame. `1` is final quality. |
| `output` | yes | `"mp4"`, `"mp4+png"` or `"png"`. PNG output needs a browser that can write folders (Chrome/Edge). Otherwise, or for any other string, the current choice stays. |
| `batchFrames` | *ignored* | Integer 1–1024. Suggested: `8`. The app recomputes it. |
| `codec` | *ignored* | Must be present: `"avc"` or `null`. |
| `bitrate` | *ignored* | Must be present: a number > 0, or `null`. |

#### Keyframes

```json
{ "coeffs": [{ "re": 1, "im": 0 }, { "re": -1, "im": 0 }],
  "view": { "cx": 0.6, "cy": 0.8, "halfW": 0.3556, "halfH": 0.2 },
  "exposure": 1.4, "gamma": 0.6 }
```

- `coeffs` is required, with the same length as `fractal.coeffs`. Coefficient *i* moves in a straight line in the complex plane from one keyframe to the next.
- `view` is optional: a [view rectangle](#views) `{ cx, cy, halfW, halfH }`, with `halfW` and `halfH` > 0.
- `exposure` and `gamma` are optional, but they come as a pair: one without the other is rejected.
- **A keyframe without `view`, or without `exposure`/`gamma`, takes that value interpolated between the nearest keyframes that have it.** Before the first such keyframe and after the last one the value is held; with `loop` it wraps around. If no keyframe has a view, every frame uses `screen`. If none has exposure/gamma, every frame uses `display.exposure` and `display.gamma`.

## Views

- **Coordinates.** `cx` is the real part of the centre and `cy` the imaginary part; up on screen is +imaginary.
- **`screen`** is `{ cx, cy, pxPerUnit, width, height, dpr }`. Load shows the world rectangle with half extents `width / (2·pxPerUnit)` × `height / (2·pxPerUnit)` around (cx, cy). That rectangle is fitted *contain*-style into the user's window, so only the rectangle matters, not the pixel numbers. `dpr` is ignored; use `1`.
- **Keyframe `view`** is that rectangle written directly as `{ cx, cy, halfW, halfH }`. Each frame fits it into the output W×H the same way. Make `halfW / halfH` equal to W / H; otherwise the extra area along one axis also shows.
- **Zoom** as shown in the app: zoom = 2 / min(halfW, halfH). Zoom 1× is Reset view, where the shorter side spans −2 to 2.
- **Between keyframes**, zoom changes geometrically (constant speed in log scale). The centre is anchored to the more zoomed-in keyframe: its on-screen offset from the frame centre shrinks or grows linearly, so a zoom target never leaves the screen. Exposure is interpolated geometrically and gamma linearly.
- **`output.width` / `output.height`** set the export and video size. MP4 needs both to be even. The GPU limits each side, typically to 8192 px, and the total to 8192² pixels. Size problems are reported when the user renders, not on Load.

## Timeline

With K keyframes and N frames:

- Keyframe *i* sits at path position *u = i* (with `loop`, the return to keyframe 0 is at *u = K*).
- Frame *j* is rendered at timeline time *t = j / (N − 1)*, or *t = j / N* with `loop`.
- *u = e(t) · (K − 1)*, or *e(t) · K* with `loop`. Here *e(t) = t* for linear easing and *e(t) = t²(3 − 2t)* for smooth easing.
- The coefficients at *u* are mixed linearly between keyframes ⌊u⌋ and ⌊u⌋ + 1. A frame that lands exactly on a keyframe reproduces that keyframe's coefficients bit for bit.

### Choosing the frame count

`frames` and the keyframe count are independent: frames sample the path, and keyframes do not add frames. With fewer frames than keyframes, frames step over keyframes, and a detail that exists in only one keyframe (a sharp turn, an exposure spike) can vanish from the video. The app warns about this under **Frames** and offers **Fit frames to keyframes**. To put a frame **exactly** on every keyframe:

| Easing | Loop | Frame counts that hit every keyframe |
|---|---|---|
| linear | no | N = m·(K − 1) + 1, for example K, 2K − 1, … |
| linear | yes | N = m·K |
| smooth | either | only the first and last keyframes are hit exactly. About 1.45·K frames put one within half a keyframe of every keyframe (K = 120 needs 174). |

For example, 120 keyframes at 60 frames with linear easing shows only every other keyframe. 120 frames shows every one.

## Versioning

- The schema string changes, to `littlewood-fractals/2` and so on, only for changes that old readers would misread.
- New **optional** fields are added without a version change; `fractal.enumeration` and `unverified` were added this way. Their absence always means the old behaviour.
- Writers should emit only the fields listed here. Readers ignore anything else.

## Checking a generated file

- Validate it against [`settings.schema.json`](settings.schema.json) with any JSON Schema 2020-12 validator, then check the two rules from [Rules for writers](#rules-for-writers) item 6.
- Final check: drop the file on the app. The status under **Load…** either names the first bad field or confirms with a one-line description (degree, coefficients, centre, zoom).
