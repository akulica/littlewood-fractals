// display.wgsl — histogram -> screen. reduceMax finds the densest pixel; vs/fs tone-map with log scaling and, in
// exports, draw the coefficient rings on top.
const MAX_RINGS: u32 = 16u;  // = MAX_COEFFS in engine.ts

struct View {       // 176 bytes, written by Engine.encodeDisplay
  width: u32,
  height: u32,
  colorMode: u32,   // 0 = inferno density, 1 = average hue of the roots in the pixel
  ringCount: u32,   // coefficient rings to draw (0 on screen: the DOM handles are there)
  exposure: f32,    // multiplies the normalised log density
  gamma: f32,       // < 1 lifts faint structure
  countScale: f32,  // counts are multiplied by this before tone-mapping (exports: area ratio, so they match the screen)
  fixedMax: f32,    // > 0: tone-map against this count instead of the densest pixel (animation: locked brightness)
  ringScale: f32,   // output px per CSS px: rings look like the screen's handles, scaled like a screenshot
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  rings: array<vec4f, 8>,  // ring centres in output pixels, two per vec4 (uniform arrays need a 16-byte stride)
};

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> hist: array<u32>;          // 4 u32 per pixel: count, r, g, b
@group(0) @binding(2) var<storage, read_write> maxOut: atomic<u32>; // reduceMax only
@group(0) @binding(3) var<storage, read> maxIn: u32;                // fs only (same buffer as maxOut)

var<workgroup> wmax: atomic<u32>;

@compute @workgroup_size(256)
fn reduceMax(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32,
             @builtin(num_workgroups) nwg: vec3u) {
  let n = V.width * V.height;
  var m = 0u;
  for (var i = g.x; i < n; i += nwg.x * 256u) { m = max(m, hist[i * 4u]); }  // grid-stride loop
  atomicMax(&wmax, m);
  workgroupBarrier();
  if (li == 0u) { atomicMax(&maxOut, atomicLoad(&wmax)); }
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));   // fullscreen triangle
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// Polynomial fit of matplotlib's inferno (Matt Zucker, shadertoy WlfXRN).
fn inferno(t: f32) -> vec3f {
  let c0 = vec3f(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  let c1 = vec3f(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  let c2 = vec3f(11.60249308247187, -3.972853965665698, -15.9423941062914);
  let c3 = vec3f(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  let c4 = vec3f(77.162935699427, -33.40235894210092, -81.80730925738993);
  let c5 = vec3f(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  let c6 = vec3f(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}

fn tone(pos: vec4f) -> vec3f {
  let idx = (u32(pos.y) * V.width + u32(pos.x)) * 4u;
  let c = f32(hist[idx]);
  if (c == 0.0) { return vec3f(0.0); }
  let m = select(max(f32(maxIn), 1.0), V.fixedMax, V.fixedMax > 0.0);
  let t = clamp(pow(log(1.0 + c * V.countScale) / log(1.0 + m * V.countScale), V.gamma) * V.exposure, 0.0, 1.0);
  if (V.colorMode == 1u) {
    let avg = vec3f(f32(hist[idx + 1u]), f32(hist[idx + 2u]), f32(hist[idx + 3u])) / (63.0 * c);
    return avg * t;
  }
  return inferno(t);
}

// The screen's .handle (style.css): 18 px content + 2 px white border, so the ring spans radius 9..11 CSS px, and
// box-shadow 0 0 4px #000, a Gaussian (sigma 2 px) of the 11 px disc drawn only outside it: 50 % black at the
// border's outer edge, fading out ~4 px further. Edges are anti-aliased over one output pixel.
fn drawRing(col: vec3f, d: f32) -> vec3f {
  let k = V.ringScale;
  let inner = 9.0 * k;
  let outer = 11.0 * k;
  let halo = 0.5 * (1.0 - smoothstep(0.0, 4.0 * k, d - outer)) * smoothstep(inner - 0.5, inner + 0.5, d);
  let ring = smoothstep(inner - 0.5, inner + 0.5, d) * (1.0 - smoothstep(outer - 0.5, outer + 0.5, d));
  return mix(mix(col, vec3f(0.0), halo), vec3f(1.0), ring);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var col = tone(pos);
  let reach = 15.0 * V.ringScale + 1.0;  // beyond the halo a ring changes nothing
  for (var i = 0u; i < min(V.ringCount, MAX_RINGS); i++) {
    let pair = V.rings[i / 2u];
    let d = distance(pos.xy, select(pair.xy, pair.zw, (i & 1u) == 1u));
    if (d < reach) { col = drawRing(col, d); }
  }
  return vec4f(col, 1.0);
}
