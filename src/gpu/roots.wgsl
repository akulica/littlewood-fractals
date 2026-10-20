// roots.wgsl — one invocation = one polynomial. Ehrlich–Aberth iteration (Gauss–Seidel order), f32.
// Measured on Intel Arc 140T: thread-per-polynomial beats thread-per-root (workgroup-cooperative) by 10-40%,
// and shrinking MAX_DEG below 32 does not speed it up, so there is exactly one kernel variant.
const MAX_DEG: u32 = 32u;
const WG: u32 = 64u;

struct Params {
  degree: u32,      // d: highest term index (polynomial has d+1 coefficient slots)
  numCoeffs: u32,   // k: size of the coefficient set
  count: u32,       // polynomials in this dispatch
  strideX: u32,     // invocations per dispatch row = workgroupsX * WG
  width: u32,       // histogram width in pixels
  height: u32,      // histogram height in pixels
  maxIter: u32,     // Aberth sweeps cap
  freeDigits: u32,  // D: digits that vary (d when the leading digit is pinned, else d+1); N = k^D
  center: vec2f,    // complex value at the canvas centre
  pxPerUnit: f32,   // pixels per unit of the complex plane
  tol: f32,         // backward-error stop: |p(z)| <= tol * sum|a_i||z|^i  (engine uses 1e-5)
  frames: u32,      // splatFrames only: coefficient sets in `coeffs` and histogram slices in `hist`
  _pad0: u32,       // 64 bytes, same layout as coop.wgsl
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
// k coefficient values. splatFrames: frames * k values, then from FRAME_VIEWS two vec2f per frame: centre, (pxPerUnit, 0).
@group(0) @binding(1) var<storage, read> coeffs: array<vec2f>;
// Base-k digits (little-endian, digit i picks a_i): [0..=32] = this dispatch's first index, [33..=65] = stride G.
@group(0) @binding(2) var<storage, read> digits: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;  // 4 u32 per pixel: count, r, g, b (splatFrames: one W*H slice per frame)
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>; // [0] = unconverged polynomials (splatFrames: polynomial-frames)
@group(0) @binding(5) var<storage, read_write> outRoots: array<vec2f>;    // debugRoots only: degree slots per polynomial

var<private> a: array<vec2f, MAX_DEG + 1u>;
var<private> z: array<vec2f, MAX_DEG>;
var<private> hue: f32;
var<private> dg: array<u32, MAX_DEG + 1u>;  // splatFrames: this thread's digits, decoded once for all frames

fn cmul(p: vec2f, q: vec2f) -> vec2f { return vec2f(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x); }
fn cdiv(p: vec2f, q: vec2f) -> vec2f {
  let d = max(dot(q, q), 1e-30);
  return vec2f(p.x * q.x + p.y * q.y, p.y * q.x - p.x * q.y) / d;
}
fn crecip(q: vec2f) -> vec2f { let d = max(dot(q, q), 1e-30); return vec2f(q.x, -q.y) / d; }
// Integer test for inf/NaN: WGSL compilers may assume floats are finite, so float compares cannot be trusted.
fn finite2(v: vec2f) -> bool {
  let e = bitcast<vec2u>(v) & vec2u(0x7f800000u);
  return all(e != vec2u(0x7f800000u));
}

const STEP: u32 = MAX_DEG + 1u;  // offset of the stride's digits in `digits`
const FRAME_VIEWS: u32 = 128u;   // = MAX_COEFFS * MAX_BATCH_FRAMES in engine.ts: per-frame views follow the coefficients

// Fills a[0..=degree] for polynomial number (first + tid * G) mod k^D, digit by digit with carry. Every digit
// value stays below 2 (k-1) count < 2^32 (Engine asserts it). The carry out of digit D-1 is dropped, which is
// exactly the mod N; digits >= D (the pinned leading one) are 0. Returns (lo, hi):
// lo = number of zero low coefficients (roots at 0), hi = effective degree after trimming zero leading coefficients.
fn loadPoly(tid: u32) -> vec2u {
  let k = P.numCoeffs;
  var carry = 0u;
  var w = 1.0;
  hue = 0.0;
  for (var i = 0u; i <= P.degree; i++) {
    var dg = 0u;
    if (i < P.freeDigits) {
      let v = digits[i] + digits[STEP + i] * tid + carry;
      dg = v % k;
      carry = v / k;
    }
    a[i] = coeffs[dg];
    w = w / f32(k);
    hue += f32(dg) * w;     // low-order terms dominate the hue (Egan's palette idea)
  }
  return trim();
}

fn trim() -> vec2u {
  var hi = P.degree;
  while (hi > 0u && dot(a[hi], a[hi]) < 1e-24) { hi--; }
  var lo = 0u;
  while (lo < hi && dot(a[lo], a[lo]) < 1e-24) { lo++; }
  return vec2u(lo, hi);
}

// splatFrames: loadPoly's digits and hue, kept in dg[] so every frame reuses them.
fn decodeDigits(tid: u32) {
  let k = P.numCoeffs;
  var carry = 0u;
  var w = 1.0;
  hue = 0.0;
  for (var i = 0u; i <= P.degree; i++) {
    var d = 0u;
    if (i < P.freeDigits) {
      let v = digits[i] + digits[STEP + i] * tid + carry;
      d = v % k;
      carry = v / k;
    }
    dg[i] = d;
    w = w / f32(k);
    hue += f32(d) * w;
  }
}

// Frame f's polynomial for the digits in dg[]; returns (lo, hi) as loadPoly does.
fn loadFrame(f: u32) -> vec2u {
  let base = f * P.numCoeffs;
  for (var i = 0u; i <= P.degree; i++) { a[i] = coeffs[base + dg[i]]; }
  return trim();
}

// Roots of sum_{i=lo..hi} a[i] z^(i-lo) into z[0..hi-lo). Returns false if maxIter was hit.
fn solve(lo: u32, hi: u32) -> bool {
  if (hi == lo) { return true; }
  coldStart(lo, hi);
  return polish(lo, hi);
}

// Initial guesses on a circle of radius = geometric mean of |roots|. Needs hi > lo.
fn coldStart(lo: u32, hi: u32) {
  let m = hi - lo;
  let r = pow(sqrt(dot(a[lo], a[lo]) / dot(a[hi], a[hi])), 1.0 / f32(m));
  for (var j = 0u; j < m; j++) {
    let t = 6.2831853 * f32(j) / f32(m) + 0.4;  // 0.4 rad offset breaks conjugate symmetry
    z[j] = r * vec2f(cos(t), sin(t));
  }
}

// Aberth sweeps from whatever z[0..hi-lo) holds (cold guesses or the previous frame's roots). Needs hi > lo.
fn polish(lo: u32, hi: u32) -> bool {
  let m = hi - lo;
  // Cauchy bound: every root has |z| <= 1 + max |a_i / a_hi|. Clamping iterates to it prevents f32 overflow.
  var cmax = 0.0;
  for (var i = lo; i < hi; i++) { cmax = max(cmax, dot(a[i], a[i])); }
  let rc = 1.0 + sqrt(cmax / dot(a[hi], a[hi]));
  let rc2 = rc * rc;
  let full = select((1u << m) - 1u, 0xffffffffu, m == 32u);
  var done = 0u;
  let tol2 = P.tol * P.tol;
  for (var it = 0u; it < P.maxIter; it++) {
    for (var j = 0u; j < m; j++) {
      if (((done >> j) & 1u) != 0u) { continue; }
      let zj = z[j];
      let az = length(zj);
      var p = a[hi];
      var dp = vec2f(0.0);
      var bound = abs(p.x) + abs(p.y);           // ~ sum |a_i| |z|^i, for the backward-error stop test
      for (var i = hi; i > lo; i--) {
        let ai = a[i - 1u];
        dp = cmul(dp, zj) + p;
        p = cmul(p, zj) + ai;
        bound = bound * az + abs(ai.x) + abs(ai.y);
      }
      // |p(z)| at f32 rounding level: z is as good as f32 can make it (also stops multiple roots,
      // which converge only linearly and would otherwise burn maxIter and stall the whole SIMD wave).
      if (dot(p, p) <= tol2 * bound * bound) { done |= 1u << j; continue; }
      let ratio = cdiv(p, dp);
      var s = vec2f(0.0);
      for (var q = 0u; q < m; q++) {
        if (q != j) { s += crecip(zj - z[q]); }
      }
      let wj = cdiv(ratio, vec2f(1.0, 0.0) - cmul(ratio, s));
      var nz = zj - wj;
      if (!finite2(nz)) { nz = cmul(zj, vec2f(0.5403, 0.8415)); }   // blew up: rotate the old guess by 1 rad instead
      let n2 = dot(nz, nz);
      if (n2 > rc2) { nz *= rc / sqrt(n2); }                       // pull back inside the Cauchy disc
      z[j] = nz;
      if (dot(wj, wj) <= 1e-12 * max(dot(zj, zj), 1e-6)) { done |= 1u << j; }
    }
    if (done == full) { return true; }
  }
  return false;
}

fn hue2rgb(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

// slice: u32 offset of the frame's histogram (0 except in splatFrames). c, ppu: the frame's view (animations zoom and pan).
fn deposit(r: vec2f, rgb: vec3u, slice: u32, c: vec2f, ppu: f32) {
  // Integer bounds check: WGSL may assume no NaN, so never rely on float comparisons to reject NaN.
  let ix = i32(floor((r.x - c.x) * ppu + f32(P.width) * 0.5));
  let iy = i32(floor(f32(P.height) * 0.5 - (r.y - c.y) * ppu));
  if (ix < 0 || iy < 0 || ix >= i32(P.width) || iy >= i32(P.height)) { return; }
  let idx = slice + (u32(iy) * P.width + u32(ix)) * 4u;
  atomicAdd(&hist[idx], 1u);
  // Colour sums are always accumulated (measured: no cost), so switching colour mode never needs a recompute.
  atomicAdd(&hist[idx + 1u], rgb.x);
  atomicAdd(&hist[idx + 2u], rgb.y);
  atomicAdd(&hist[idx + 3u], rgb.z);
}

@compute @workgroup_size(WG)
fn splat(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x + gid.y * P.strideX;
  if (tid >= P.count) { return; }
  let lh = loadPoly(tid);
  if (!solve(lh.x, lh.y)) { atomicAdd(&stats[0], 1u); }
  let rgb = vec3u(hue2rgb(hue) * 63.0 + 0.5);  // 6-bit so a pixel can take 68M roots before u32 overflow
  for (var j = 0u; j < lh.x; j++) { deposit(vec2f(0.0), rgb, 0u, P.center, P.pxPerUnit); }
  for (var j = 0u; j < lh.y - lh.x; j++) { deposit(z[j], rgb, 0u, P.center, P.pxPerUnit); }
}

// Animation: one thread follows its polynomial through P.frames frames (frame f uses coeffs[f*k ..] and histogram
// slice f). Consecutive frames move the roots only a little, so the previous frame's roots are a much better start
// than the circle: measured 6.4 -> 2.3-2.7 Aberth sweeps per polynomial-frame, 2.2-2.6x faster at 1080p (Fb = 8).
@compute @workgroup_size(WG)
fn splatFrames(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x + gid.y * P.strideX;
  if (tid >= P.count) { return; }
  decodeDigits(tid);
  let rgb = vec3u(hue2rgb(hue) * 63.0 + 0.5);
  var prev = vec2u(MAX_DEG + 1u);   // no previous frame: the first one starts cold
  for (var f = 0u; f < P.frames; f++) {
    let lh = loadFrame(f);
    var ok = true;
    if (lh.y > lh.x) {
      // Warm only if the trimmed shape is unchanged: otherwise z holds a different number of roots.
      let warm = all(lh == prev);
      if (!warm) { coldStart(lh.x, lh.y); }
      ok = polish(lh.x, lh.y);
      // A warm start can sit next to the wrong root cluster after a big step; the circle start is the robust one.
      if (!ok && warm) {
        coldStart(lh.x, lh.y);
        ok = polish(lh.x, lh.y);
      }
    }
    prev = lh;
    if (!ok) { atomicAdd(&stats[0], 1u); }
    let slice = f * P.width * P.height * 4u;
    let c = coeffs[FRAME_VIEWS + 2u * f];
    let ppu = coeffs[FRAME_VIEWS + 2u * f + 1u].x;
    for (var j = 0u; j < lh.x; j++) { deposit(vec2f(0.0), rgb, slice, c, ppu); }
    for (var j = 0u; j < lh.y - lh.x; j++) { deposit(z[j], rgb, slice, c, ppu); }
  }
}

@compute @workgroup_size(WG)
fn debugRoots(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x + gid.y * P.strideX;
  if (tid >= P.count) { return; }
  let lh = loadPoly(tid);
  let ok = solve(lh.x, lh.y);
  let base = tid * P.degree;
  let m = lh.y - lh.x;
  for (var j = 0u; j < P.degree; j++) {
    var v = vec2f(3.0e38, select(0.0, 1.0, ok));   // unused slot marker (dropped leading zeros)
    if (j < lh.x) { v = vec2f(0.0); } else if (j < lh.x + m) { v = z[j - lh.x]; }
    outRoots[base + j] = v;
  }
}
