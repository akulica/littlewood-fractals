// coop.wgsl — cooperative variant: one workgroup thread per ROOT, floor(256/d) polynomials per workgroup,
// Jacobi Aberth sweeps through shared memory. Same Params/bindings 0-4 as roots.wgsl (strideX is unused:
// the polynomial index comes from num_workgroups). Measured: ~1.5x faster than roots.wgsl on RX 9060 XT
// (RDNA 4), 1.2-1.3x slower on Intel Arc 140T — Engine.calibrate() picks per GPU at startup.
// splatFramesCoop is the animation kernel (roots.wgsl splatFrames' warm start) in this layout.

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
  frames: u32,      // splatFramesCoop: coefficient sets in `coeffs` and histogram slices in `hist`
  _pad0: u32,       // 64 bytes, same layout as roots.wgsl
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
// k coefficient values. splatFramesCoop: frames * k values, then from FRAME_VIEWS two vec2f per frame: centre, (pxPerUnit, 0).
@group(0) @binding(1) var<storage, read> coeffs: array<vec2f>;
// Enumeration (core/reciprocal.ts packChunk / packPlan), same layout as roots.wgsl.
@group(0) @binding(2) var<storage, read> en: array<u32>;
const FRAME_VIEWS: u32 = 128u;  // = FRAME_VIEWS in roots.wgsl and engine.ts
const MAX_SLOTS: u32 = 33u;  // = MAX_DEG + 1 in roots.wgsl
const MAX_REGIONS: u32 = 32u;
const ENTRY: u32 = 4u + 2u * MAX_SLOTS;
const REGION: u32 = 4u + 4u * MAX_SLOTS;
const REGION_BASE: u32 = MAX_REGIONS * ENTRY;
const TABLE_BASE: u32 = REGION_BASE + MAX_REGIONS * REGION;
const MAX_TABLE: u32 = 2048u;
// 4 u32 per pixel: count, r, g, b (splatFramesCoop: one W*H slice per frame)
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
// [0] = unconverged ROOTS (splatCoop, per-thread) / polynomial-frames (splatFramesCoop, as roots.wgsl splatFrames)
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

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
// slice: u32 offset of the frame's histogram (0 in splatCoop). c, ppu: the frame's view.
fn deposit(r: vec2f, rgb: vec3u, slice: u32, c: vec2f, ppu: f32) {
  let ix = i32(floor((r.x - c.x) * ppu + f32(P.width) * 0.5));
  let iy = i32(floor(f32(P.height) * 0.5 - (r.y - c.y) * ppu));
  if (ix < 0 || iy < 0 || ix >= i32(P.width) || iy >= i32(P.height)) { return; }
  let idx = slice + (u32(iy) * P.width + u32(ix)) * 4u;
  atomicAdd(&hist[idx], 1u);
  atomicAdd(&hist[idx + 1u], rgb.x);
  atomicAdd(&hist[idx + 2u], rgb.y);
  atomicAdd(&hist[idx + 3u], rgb.z);
}

// Workgroup memory: sa 4096 + sz 2048 + sinfo 4096 + wbusy/sflag 8 bytes (+ sfail/sbad 64 in splatFramesCoop)
// = 10.3 KB of the 16 KB WebGPU guarantees.
const WGS: u32 = 256u;
const WORDS: u32 = WGS / 32u;            // one bit per polynomial slot
var<workgroup> sa: array<vec2f, 512>;    // coefficients of the polys in this workgroup, (d+1) per poly
var<workgroup> sz: array<vec2f, 256>;    // current root estimates, d per poly
var<workgroup> sinfo: array<vec4u, 256>; // per poly: lo, hi, packed rgb, packed partner rgb | partner << 24
var<workgroup> wbusy: atomic<u32>;
var<workgroup> sflag: u32;
var<workgroup> sfail: array<atomic<u32>, WORDS>; // splatFramesCoop: polys whose first solve of this frame failed
var<workgroup> sbad: array<atomic<u32>, WORDS>;  // splatFramesCoop: polys whose cold retry failed too

// Digits of solve tid as in roots.wgsl decode, staged as sa[abase + i].x = f32(digit); colours and partner flag into
// sinfo[slot].zw (lo, hi are left 0).
fn decodeDigits(tid: u32, slot: u32, abase: u32) {
  let d = P.degree;
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
  let k = P.numCoeffs;
  var w = 1.0; var h = 0.0; var ph = 0.0;
  for (var i = 0u; i <= d; i++) {
    w = w / f32(k); h += sa[abase + i].x * w;
    ph += f32(tbl(pmap + u32(sa[abase + d - i].x))) * w;
  }
  sinfo[slot] = vec4u(0u, 0u, packRgb(h), packRgb(ph) | (min(en[rb + 1u], 1u) << 24u));
}

// (lo, hi) of the poly in sa[abase ..]: lo = zero low coefficients (roots at 0), hi = degree without zero leading terms.
fn trim(abase: u32) -> vec2u {
  var hi = P.degree;
  while (hi > 0u && dot(sa[abase + hi], sa[abase + hi]) < 1e-24) { hi--; }
  var lo = 0u;
  while (lo < hi && dot(sa[abase + lo], sa[abase + lo]) < 1e-24) { lo++; }
  return vec2u(lo, hi);
}

// Cauchy bound 1 + max |a_i / a_hi|: every root lies inside, iterates are clamped to it. Needs hi > lo.
fn cauchy(abase: u32, lo: u32, hi: u32) -> f32 {
  let ahi = sa[abase + hi];
  var cmax = 0.0;
  for (var i = lo; i < hi; i++) { cmax = max(cmax, dot(sa[abase + i], sa[abase + i])); }
  return 1.0 + sqrt(cmax / dot(ahi, ahi));
}

// Cold start of root j < hi - lo on the circle of radius = geometric mean of |roots| (roots.wgsl coldStart).
fn coldGuess(abase: u32, lo: u32, hi: u32, j: u32) -> vec2f {
  let m = hi - lo;
  let alo = sa[abase + lo]; let ahi = sa[abase + hi];
  let r = pow(sqrt(dot(alo, alo) / dot(ahi, ahi)), 1.0 / f32(m));
  let t = 6.2831853 * f32(j) / f32(m) + 0.4;
  return r * vec2f(cos(t), sin(t));
}

// Jacobi Aberth sweeps until no thread of the workgroup is busy or maxIter. Live thread j (j < hi - lo) owns root j
// in *z and sz[zbase + j], which must hold the same start value; threads that start *done only keep their root
// visible in sz. Barriers inside: call in workgroup-uniform control flow.
fn polish(li: u32, j: u32, abase: u32, zbase: u32, lo: u32, hi: u32, live: bool, rc: f32, z: ptr<function, vec2f>, done: ptr<function, bool>) {
  let m = hi - lo;
  let tol2 = P.tol * P.tol;
  workgroupBarrier();
  for (var it = 0u; it < P.maxIter; it++) {
    let zj = *z;
    var nz = zj;
    if (!*done) {
      let az = length(zj);
      // |z| > 1: evaluate the reversed polynomial at w = 1/z (no f32 overflow; see roots.wgsl polish).
      let flip = az > 1.0;
      let x = select(zj, crecip(zj), flip);
      let ax = select(az, 1.0 / az, flip);
      var p = sa[abase + select(hi, lo, flip)];
      var dp = vec2f(0.0);
      var bound = abs(p.x) + abs(p.y);
      for (var t = 1u; t <= m; t++) {
        let ai = sa[abase + select(hi - t, lo + t, flip)];
        dp = cmul(dp, x) + p;
        p = cmul(p, x) + ai;
        bound = bound * ax + abs(ai.x) + abs(ai.y);
      }
      let ratio = select(cdiv(p, dp), cdiv(cmul(zj, p), f32(m) * p - cmul(x, dp)), flip);
      if (dot(p, p) <= tol2 * bound * bound) {
        *done = true;
      } else {
        var s = vec2f(0.0);
        for (var q = 0u; q < m; q++) {
          if (q != j) { s += crecip(zj - sz[zbase + q]); }
        }
        let wj = cdiv(ratio, vec2f(1.0, 0.0) - cmul(ratio, s));
        nz = zj - wj;
        if (!finite2(nz)) { nz = cmul(zj, vec2f(0.5403, 0.8415)); }
        let n2 = dot(nz, nz);
        if (n2 > rc * rc) { nz *= rc / sqrt(n2); }
        if (dot(wj, wj) <= 1e-12 * max(dot(zj, zj), 1e-6)) { *done = true; }
      }
    }
    workgroupBarrier();
    if (live) { sz[zbase + j] = nz; }
    *z = nz;
    if (!*done) { atomicOr(&wbusy, 1u); }
    workgroupBarrier();
    if (li == 0u) { sflag = atomicLoad(&wbusy); atomicStore(&wbusy, 0u); }
    if (workgroupUniformLoad(&sflag) == 0u) { break; }
  }
}

// Thread j's share of its poly's roots, info = sinfo[slot] (roots.wgsl depositRoots): root threads deposit z and, with
// a partner (the reversed polynomial), 1/z; threads m..hi-1 the lo roots at the origin; threads hi..d-1 the partner's
// d - hi roots at the origin.
fn depositRoots(j: u32, valid: bool, live: bool, hi: u32, z: vec2f, info: vec4u, slice: u32, c: vec2f, ppu: f32) {
  let partner = (info.w >> 24u) != 0u;
  let rgb = unpackRgb(info.z);
  if (live) {
    deposit(z, rgb, slice, c, ppu);
    if (partner) { deposit(crecip(z), unpackRgb(info.w), slice, c, ppu); }
  }
  else if (valid && j < hi) { deposit(vec2f(0.0), rgb, slice, c, ppu); }
  else if (valid && partner) { deposit(vec2f(0.0), unpackRgb(info.w), slice, c, ppu); }
}

@compute @workgroup_size(WGS)
fn splatCoop(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3u) {
  let d = P.degree;
  let ppw = WGS / d;
  let slot = li / d;
  let j = li % d;
  let tid = (wid.x + wid.y * nwg.x) * ppw + slot;
  let valid = slot < ppw && tid < P.count;
  let abase = slot * (d + 1u);
  let zbase = slot * d;
  if (valid && j == 0u) {
    decodeDigits(tid, slot, abase);
    for (var i = 0u; i <= d; i++) { sa[abase + i] = coeffs[u32(sa[abase + i].x)]; }
    sinfo[slot] = vec4u(trim(abase), sinfo[slot].zw);
  }
  workgroupBarrier();
  var info = vec4u(0u);
  if (valid) { info = sinfo[slot]; }
  let lo = info.x; let hi = info.y;
  let live = valid && j < hi - lo;
  var z = vec2f(0.0);
  var rc = 1.0;
  if (live) {
    rc = cauchy(abase, lo, hi);
    z = coldGuess(abase, lo, hi, j);
    sz[zbase + j] = z;
  }
  var done = !live;
  polish(li, j, abase, zbase, lo, hi, live, rc, &z, &done);
  if (!done) { atomicAdd(&stats[0], select(1u, 2u, (info.w >> 24u) != 0u)); }
  depositRoots(j, valid, live, hi, z, info, 0u, P.center, P.pxPerUnit);
}

// Animation: roots.wgsl splatFrames in the cooperative layout. The workgroup follows its polynomials through P.frames
// frames (frame f uses coeffs[f*k ..], histogram slice f and view f). A poly whose trimmed shape is unchanged starts
// from its previous frame's roots, still in z and sz; if that warm start fails, the poly is solved again from the
// circle, in a second pass the whole workgroup runs only when some poly needs it. Each thread keeps the digit of the
// coefficient it loads per frame (a_j; thread 0 also a_d). Unconverged polynomial-frames are counted once per poly:
// the first of its root threads to set the poly's bit adds it.
@compute @workgroup_size(WGS)
fn splatFramesCoop(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3u) {
  let d = P.degree;
  let k = P.numCoeffs;
  let ppw = WGS / d;
  let slot = li / d;
  let j = li % d;
  let tid = (wid.x + wid.y * nwg.x) * ppw + slot;
  let valid = slot < ppw && tid < P.count;
  let abase = slot * (d + 1u);
  let zbase = slot * d;
  let word = slot / 32u;
  let bit = 1u << (slot % 32u);
  if (valid && j == 0u) { decodeDigits(tid, slot, abase); }
  workgroupBarrier();
  var info = vec4u(0u);
  var dj = 0u;
  var dd = 0u;
  if (valid) {
    info = sinfo[slot];
    dj = u32(sa[abase + j].x);
    dd = u32(sa[abase + d].x);
  }
  let unconv = select(1u, 2u, (info.w >> 24u) != 0u);
  var prev = vec2u(MAX_SLOTS + 1u);   // no previous frame: the first one starts cold
  var z = vec2f(0.0);
  for (var f = 0u; f < P.frames; f++) {
    workgroupBarrier();   // everyone is done with the previous frame's coefficients (and the digits) and fail bits
    if (li < WORDS) { atomicStore(&sfail[li], 0u); atomicStore(&sbad[li], 0u); }
    if (valid) {
      sa[abase + j] = coeffs[f * k + dj];
      if (j == 0u) { sa[abase + d] = coeffs[f * k + dd]; }
    }
    workgroupBarrier();
    var lh = vec2u(0u);
    if (valid) { lh = trim(abase); }
    let lo = lh.x; let hi = lh.y;
    let live = valid && j < hi - lo;
    // Warm only if the trimmed shape is unchanged: otherwise z holds a different number of roots.
    let warm = all(lh == prev);
    prev = lh;
    var rc = 1.0;
    if (live) {
      rc = cauchy(abase, lo, hi);
      if (!warm) { z = coldGuess(abase, lo, hi, j); sz[zbase + j] = z; }
    }
    var done = !live;
    polish(li, j, abase, zbase, lo, hi, live, rc, &z, &done);
    if (!done) {
      let old = atomicOr(&sfail[word], bit);
      if (warm) { atomicOr(&wbusy, 1u); } else if ((old & bit) == 0u) { atomicAdd(&stats[0], unconv); }
    }
    workgroupBarrier();
    if (li == 0u) { sflag = atomicLoad(&wbusy); atomicStore(&wbusy, 0u); }
    if (workgroupUniformLoad(&sflag) != 0u) {
      // A warm start can sit next to the wrong root cluster after a big step; the circle start is the robust one.
      done = true;
      if (live && warm && (atomicLoad(&sfail[word]) & bit) != 0u) {
        z = coldGuess(abase, lo, hi, j);
        sz[zbase + j] = z;
        done = false;
      }
      polish(li, j, abase, zbase, lo, hi, live, rc, &z, &done);
      if (!done && (atomicOr(&sbad[word], bit) & bit) == 0u) { atomicAdd(&stats[0], unconv); }
    }
    let c = coeffs[FRAME_VIEWS + 2u * f];
    let ppu = coeffs[FRAME_VIEWS + 2u * f + 1u].x;
    depositRoots(j, valid, live, hi, z, info, f * P.width * P.height * 4u, c, ppu);
  }
}
