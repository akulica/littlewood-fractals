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
  _pad0: u32,
  center: vec2f,    // complex value at the canvas centre
  pxPerUnit: f32,   // pixels per unit of the complex plane
  tol: f32,         // backward-error stop: |p(z)| <= tol * sum|a_i||z|^i  (engine uses 1e-5)
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<vec2f>;      // k coefficient values
@group(0) @binding(2) var<storage, read> baseDigits: array<u32>;    // d+1 base-k digits of this dispatch's first index; digit i picks a_i
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;  // 4 u32 per pixel: count, r, g, b
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>; // [0] = unconverged polynomials
@group(0) @binding(5) var<storage, read_write> outRoots: array<vec2f>;    // debugRoots only: degree slots per polynomial

var<private> a: array<vec2f, MAX_DEG + 1u>;
var<private> z: array<vec2f, MAX_DEG>;
var<private> hue: f32;

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

// Fills a[0..=degree] for polynomial number (base + tid). Returns (lo, hi):
// lo = number of zero low coefficients (roots at 0), hi = effective degree after trimming zero leading coefficients.
fn loadPoly(tid: u32) -> vec2u {
  let k = P.numCoeffs;
  var carry = tid;
  var w = 1.0;
  hue = 0.0;
  for (var i = 0u; i <= P.degree; i++) {
    let s = baseDigits[i] + carry % k;
    carry = carry / k + s / k;
    let dg = s % k;
    a[i] = coeffs[dg];
    w = w / f32(k);
    hue += f32(dg) * w;     // low-order terms dominate the hue (Egan's palette idea)
  }
  var hi = P.degree;
  while (hi > 0u && dot(a[hi], a[hi]) < 1e-24) { hi--; }
  var lo = 0u;
  while (lo < hi && dot(a[lo], a[lo]) < 1e-24) { lo++; }
  return vec2u(lo, hi);
}

// Roots of sum_{i=lo..hi} a[i] z^(i-lo) into z[0..hi-lo). Returns false if maxIter was hit.
fn solve(lo: u32, hi: u32) -> bool {
  let m = hi - lo;
  if (m == 0u) { return true; }
  let r = pow(sqrt(dot(a[lo], a[lo]) / dot(a[hi], a[hi])), 1.0 / f32(m)); // geometric mean of |roots|
  // Cauchy bound: every root has |z| <= 1 + max |a_i / a_hi|. Clamping iterates to it prevents f32 overflow.
  var cmax = 0.0;
  for (var i = lo; i < hi; i++) { cmax = max(cmax, dot(a[i], a[i])); }
  let rc = 1.0 + sqrt(cmax / dot(a[hi], a[hi]));
  let rc2 = rc * rc;
  for (var j = 0u; j < m; j++) {
    let t = 6.2831853 * f32(j) / f32(m) + 0.4;  // 0.4 rad offset breaks conjugate symmetry
    z[j] = r * vec2f(cos(t), sin(t));
  }
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

fn deposit(r: vec2f, rgb: vec3u) {
  // Integer bounds check: WGSL may assume no NaN, so never rely on float comparisons to reject NaN.
  let ix = i32(floor((r.x - P.center.x) * P.pxPerUnit + f32(P.width) * 0.5));
  let iy = i32(floor(f32(P.height) * 0.5 - (r.y - P.center.y) * P.pxPerUnit));
  if (ix < 0 || iy < 0 || ix >= i32(P.width) || iy >= i32(P.height)) { return; }
  let idx = (u32(iy) * P.width + u32(ix)) * 4u;
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
  for (var j = 0u; j < lh.x; j++) { deposit(vec2f(0.0), rgb); }
  for (var j = 0u; j < lh.y - lh.x; j++) { deposit(z[j], rgb); }
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
