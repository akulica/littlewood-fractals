// display.wgsl — histogram -> screen. reduceMax finds the densest pixel; vs/fs tone-map with log scaling.
struct View {
  width: u32,
  height: u32,
  colorMode: u32,   // 0 = inferno density, 1 = average hue of the roots in the pixel
  _pad: u32,
  exposure: f32,    // multiplies the normalised log density
  gamma: f32,       // < 1 lifts faint structure
  countScale: f32,  // counts are multiplied by this before tone-mapping (exports: area ratio, so they match the screen)
  fixedMax: f32,    // > 0: tone-map against this count instead of the densest pixel (animation: locked brightness)
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

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let idx = (u32(pos.y) * V.width + u32(pos.x)) * 4u;
  let c = f32(hist[idx]);
  if (c == 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let m = select(max(f32(maxIn), 1.0), V.fixedMax, V.fixedMax > 0.0);
  let t = clamp(pow(log(1.0 + c * V.countScale) / log(1.0 + m * V.countScale), V.gamma) * V.exposure, 0.0, 1.0);
  if (V.colorMode == 1u) {
    let avg = vec3f(f32(hist[idx + 1u]), f32(hist[idx + 2u]), f32(hist[idx + 3u])) / (63.0 * c);
    return vec4f(avg * t, 1.0);
  }
  return vec4f(inferno(t), 1.0);
}
