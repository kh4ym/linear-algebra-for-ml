/* LA.ease / LA.seg / lerp helpers — the only animation vocabulary a lesson needs.
   Steps are pure functions of t ∈ [0,1]; these helpers carve t into eased sub-intervals. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;

  const ease = {
    linear: t => t,
    inOut: t => { t = clamp01(t); return t * t * (3 - 2 * t); },                 // smoothstep
    smooth: t => { t = clamp01(t); return t * t * t * (t * (t * 6 - 15) + 10); }, // smootherstep
    in: t => { t = clamp01(t); return t * t; },
    out: t => { t = clamp01(t); return 1 - (1 - t) * (1 - t); },
    outCubic: t => { t = clamp01(t); return 1 - Math.pow(1 - t, 3); },
    back: t => { t = clamp01(t); const c = 1.4; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); },
    bounce: t => { t = clamp01(t); return 1 - Math.abs(Math.cos(t * Math.PI * 2.5)) * (1 - t); },
  };

  /** Progress of t inside [a,b], eased (default inOut). 0 before a, 1 after b. */
  function seg(t, a, b, e) {
    if (a === undefined) { a = 0; b = 1; }
    if (b === undefined) { b = 1; }
    const u = b > a ? (t - a) / (b - a) : (t >= b ? 1 : 0);
    const f = typeof e === 'function' ? e : ease[e || 'inOut'];
    return f(clamp01(u));
  }
  /** 0 → 1 → 0 over [a,b] (for highlights / pulses). */
  seg.pulse = function (t, a, b) { const u = seg(t, a, b, 'linear'); return Math.sin(u * Math.PI); };
  /** Hard switch at time a. */
  seg.after = function (t, a) { return t >= a ? 1 : 0; };
  /** Repeating ramp for loops (n cycles over [0,1]). */
  seg.cycle = function (t, n) { return (clamp01(t) * (n || 1)) % 1; };

  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpV = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
  const lerpM = (A, B, t) => A.map((row, i) => row.map((x, j) => x + (B[i][j] - x) * t));
  /** Shortest-path angle interpolation. */
  const lerpAngle = (a, b, t) => { let d = (b - a) % (2 * Math.PI); if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return a + d * t; };
  /** Interpolate an entire camera / plain object field-by-field (numbers and arrays of numbers only). */
  const lerpObj = (a, b, t) => {
    const out = {};
    Object.keys(b).forEach(k => {
      const x = a[k], y = b[k];
      if (typeof y === 'number' && typeof x === 'number') out[k] = (k === 'yaw' || k === 'pitch') ? lerpAngle(x, y, t) : lerp(x, y, t);
      else if (Array.isArray(y) && Array.isArray(x)) out[k] = Array.isArray(y[0]) ? lerpM(x, y, t) : lerpV(x, y, t);
      else out[k] = t < 1 ? x : y;
    });
    return out;
  };

  LA.ease = ease;
  LA.seg = seg;
  LA.lerp = lerp; LA.lerpV = lerpV; LA.lerpM = lerpM; LA.lerpAngle = lerpAngle; LA.lerpObj = lerpObj;
  LA.clamp01 = clamp01;
  LA.clamp = (x, lo, hi) => x < lo ? lo : x > hi ? hi : x;

  /** Deep clone of plain JSON-like data (arrays, objects, numbers, strings, booleans, null). */
  LA.clone = function clone(x) {
    if (x === null || typeof x !== 'object') return x;
    if (Array.isArray(x)) { const a = new Array(x.length); for (let i = 0; i < x.length; i++) a[i] = clone(x[i]); return a; }
    if (x instanceof Float32Array || x instanceof Float64Array) return new x.constructor(x);
    const o = {}; for (const k in x) if (Object.prototype.hasOwnProperty.call(x, k)) o[k] = clone(x[k]); return o;
  };
  /** Reset `target` in place so it equals a deep copy of `source` (keeps object identity). */
  LA.assignInPlace = function (target, source) {
    for (const k in target) if (Object.prototype.hasOwnProperty.call(target, k)) delete target[k];
    const c = LA.clone(source);
    for (const k in c) target[k] = c[k];
    return target;
  };
  /** Path access: getPath(s, 'M[0][1]') / setPath(s, 'cam.dist', 5). */
  LA.getPath = function (obj, path) {
    const parts = String(path).replace(/\]/g, '').split(/[.\[]/).filter(Boolean);
    let cur = obj; for (const p of parts) { if (cur == null) return undefined; cur = cur[p]; } return cur;
  };
  LA.setPath = function (obj, path, value) {
    const parts = String(path).replace(/\]/g, '').split(/[.\[]/).filter(Boolean);
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) { if (cur[parts[i]] == null) cur[parts[i]] = /^\d+$/.test(parts[i + 1]) ? [] : {}; cur = cur[parts[i]]; }
    cur[parts[parts.length - 1]] = value; return obj;
  };
})();
