// coop.wgsl — cooperative variant: one workgroup thread per ROOT, floor(256/d) polynomials per workgroup,
// Jacobi Aberth sweeps through shared memory. Same Params/bindings 0-4 as roots.wgsl (strideX is unused:
// the polynomial index comes from num_workgroups). Measured: ~1.5x faster than roots.wgsl on RX 9060 XT
// (RDNA 4), 1.2-1.3x slower on Intel Arc 140T — Engine.calibrate() picks per GPU at startup.

struct Params {
  degree: u32,      // d: highest term index (polynomial has d+1 coefficient slots)
  numCoeffs: u32,   // k: size of the coefficient set
  count: u32,       // solves in this dispatch
  strideX: u32,     // unused here (roots.wgsl only)
  width: u32,       // histogram width in pixels
  height: u32,      // histogram height in pixels
  maxIter: u32,     // Aberth sweeps cap
  entries: u32,     // chunk entries in `en` (roots.wgsl decode)
  center: vec2f,    // complex value at the canvas centre
  pxPerUnit: f32,   // pixels per unit of the complex plane
  tol: f32,
  frames: u32,      // roots.wgsl splatFrames only
  _pad0: u32,       // 64 bytes, same layout as roots.wgsl
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<vec2f>;      // k coefficient values
// Enumeration (core/reciprocal.ts packChunk / packPlan), same layout as roots.wgsl.
@group(0) @binding(2) var<storage, read> en: array<u32>;
const MAX_SLOTS: u32 = 33u;  // = MAX_DEG + 1 in roots.wgsl
const MAX_REGIONS: u32 = 32u;
const ENTRY: u32 = 4u + 2u * MAX_SLOTS;
const REGION: u32 = 4u + 4u * MAX_SLOTS;
const REGION_BASE: u32 = MAX_REGIONS * ENTRY;
const TABLE_BASE: u32 = REGION_BASE + MAX_REGIONS * REGION;
const MAX_TABLE: u32 = 2048u;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;  // 4 u32 per pixel: count, r, g, b
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>; // [0] = unconverged ROOTS (per-thread here)

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

fn hue2rgb(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}
fn tbl(i: u32) -> u32 { return en[TABLE_BASE + min(i, MAX_TABLE - 1u)]; }
fn packRgb(h: f32) -> u32 {
  let c = vec3u(hue2rgb(h) * 63.0 + 0.5);
  return c.x | (c.y << 8u) | (c.z << 16u);
}
fn unpackRgb(c: u32) -> vec3u { return vec3u(c & 255u, (c >> 8u) & 255u, (c >> 16u) & 255u); }
fn deposit(r: vec2f, rgb: vec3u) {
  let ix = i32(floor((r.x - P.center.x) * P.pxPerUnit + f32(P.width) * 0.5));
  let iy = i32(floor(f32(P.height) * 0.5 - (r.y - P.center.y) * P.pxPerUnit));
  if (ix < 0 || iy < 0 || ix >= i32(P.width) || iy >= i32(P.height)) { return; }
  let idx = (u32(iy) * P.width + u32(ix)) * 4u;
  atomicAdd(&hist[idx], 1u);
  atomicAdd(&hist[idx + 1u], rgb.x);
  atomicAdd(&hist[idx + 2u], rgb.y);
  atomicAdd(&hist[idx + 3u], rgb.z);
}

const WGS: u32 = 256u;
var<workgroup> sa: array<vec2f, 512>;    // coefficients of the polys in this workgroup, (d+1) per poly
var<workgroup> sz: array<vec2f, 256>;    // current root estimates, d per poly
var<workgroup> sinfo: array<vec4u, 256>; // per poly: lo, hi, packed rgb, packed partner rgb | partner << 24
var<workgroup> wbusy: atomic<u32>;
var<workgroup> sflag: u32;

@compute @workgroup_size(WGS)
fn splatCoop(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3u) {
  let d = P.degree;
  let k = P.numCoeffs;
  let ppw = WGS / d;
  let slot = li / d;
  let j = li % d;
  let tid = (wid.x + wid.y * nwg.x) * ppw + slot;
  let valid = slot < ppw && tid < P.count;
  let abase = slot * (d + 1u);
  let zbase = slot * d;
  if (valid && j == 0u) {
    // Digits as in roots.wgsl decode, staged as sa[abase + i].x = f32(digit), then replaced by the coefficients.
    var e = 0u;
    for (var i = 1u; i < min(P.entries, MAX_REGIONS); i++) {
      if (en[i * ENTRY] <= tid) { e = i; }
    }
    let eb = e * ENTRY;
    let local = tid - en[eb];
    let rb = REGION_BASE + min(en[eb + 1u], MAX_REGIONS - 1u) * REGION;
    let slots = min(en[rb], MAX_SLOTS);
    var carry = 0u;
    for (var s = 0u; s < slots; s++) {
      let sb = rb + 4u + 4u * s;
      let radix = max(en[sb], 1u);
      let v = en[eb + 4u + s] + en[eb + 4u + MAX_SLOTS + s] * local + carry;
      let t = tbl(en[sb + 1u] + v % radix);
      carry = v / radix;
      sa[abase + min(en[sb + 2u], d)] = vec2f(f32(t & 0xffu), 0.0);
      sa[abase + min(en[sb + 3u], d)] = vec2f(f32(t >> 8u), 0.0);
    }
    let pmap = en[rb + 2u];
    var w = 1.0; var h = 0.0; var ph = 0.0;
    for (var i = 0u; i <= d; i++) {
      w = w / f32(k); h += sa[abase + i].x * w;
      ph += f32(tbl(pmap + u32(sa[abase + d - i].x))) * w;
    }
    for (var i = 0u; i <= d; i++) { sa[abase + i] = coeffs[u32(sa[abase + i].x)]; }
    var hi = d;
    while (hi > 0u && dot(sa[abase + hi], sa[abase + hi]) < 1e-24) { hi--; }
    var lo = 0u;
    while (lo < hi && dot(sa[abase + lo], sa[abase + lo]) < 1e-24) { lo++; }
    sinfo[slot] = vec4u(lo, hi, packRgb(h), packRgb(ph) | (min(en[rb + 1u], 1u) << 24u));
  }
  workgroupBarrier();
  var info = vec4u(0u);
  if (valid) { info = sinfo[slot]; }
  let lo = info.x; let hi = info.y; let m = hi - lo;
  let live = valid && j < m;
  var z = vec2f(0.0);
  var rc = 1.0;
  if (live) {
    let alo = sa[abase + lo]; let ahi = sa[abase + hi];
    let r = pow(sqrt(dot(alo, alo) / dot(ahi, ahi)), 1.0 / f32(m));
    var cmax = 0.0;
    for (var i = lo; i < hi; i++) { cmax = max(cmax, dot(sa[abase + i], sa[abase + i])); }
    rc = 1.0 + sqrt(cmax / dot(ahi, ahi));
    let t = 6.2831853 * f32(j) / f32(m) + 0.4;
    z = r * vec2f(cos(t), sin(t));
    sz[zbase + j] = z;
  }
  var done = !live;
  let tol2 = P.tol * P.tol;
  workgroupBarrier();
  for (var it = 0u; it < P.maxIter; it++) {
    var nz = z;
    if (!done) {
      let az = length(z);
      var p = sa[abase + hi];
      var dp = vec2f(0.0);
      var bound = abs(p.x) + abs(p.y);
      for (var i = hi; i > lo; i--) {
        let ai = sa[abase + i - 1u];
        dp = cmul(dp, z) + p;
        p = cmul(p, z) + ai;
        bound = bound * az + abs(ai.x) + abs(ai.y);
      }
      if (dot(p, p) <= tol2 * bound * bound) {
        done = true;
      } else {
        let ratio = cdiv(p, dp);
        var s = vec2f(0.0);
        for (var q = 0u; q < m; q++) {
          if (q != j) { s += crecip(z - sz[zbase + q]); }
        }
        let wj = cdiv(ratio, vec2f(1.0, 0.0) - cmul(ratio, s));
        nz = z - wj;
        if (!finite2(nz)) { nz = cmul(z, vec2f(0.5403, 0.8415)); }
        let n2 = dot(nz, nz);
        if (n2 > rc * rc) { nz *= rc / sqrt(n2); }
        if (dot(wj, wj) <= 1e-12 * max(dot(z, z), 1e-6)) { done = true; }
      }
    }
    workgroupBarrier();
    if (live) { sz[zbase + j] = nz; }
    z = nz;
    if (!done) { atomicOr(&wbusy, 1u); }
    workgroupBarrier();
    if (li == 0u) { sflag = atomicLoad(&wbusy); atomicStore(&wbusy, 0u); }
    if (workgroupUniformLoad(&sflag) == 0u) { break; }
  }
  // Partner (the reversed polynomial, roots.wgsl depositRoots): root threads also deposit 1/z, threads hi..d-1 its
  // d - hi roots at the origin.
  let partner = (info.w >> 24u) != 0u;
  if (!done) { atomicAdd(&stats[0], select(1u, 2u, partner)); }
  let rgb = unpackRgb(info.z);
  if (live) {
    deposit(z, rgb);
    if (partner) { deposit(crecip(z), unpackRgb(info.w)); }
  }
  else if (valid && j < hi) { deposit(vec2f(0.0), rgb); }   // the lo roots at the origin
  else if (valid && partner) { deposit(vec2f(0.0), unpackRgb(info.w)); }
}
