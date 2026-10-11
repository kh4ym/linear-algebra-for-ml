/*
 * js/core/rng.js — deterministic seeded RNG (mulberry32) and synthetic data generators for the LA course.
 * Classic script: LA.rng(seed) returns a function () => float in [0,1); helpers hang off LA.rng.
 * All generators are pure functions of their seed/arguments (no Date, no Math.random) and return
 * plain arrays. Depends on LA.num (looked up lazily at call time) for chol / eigSym / vector ops.
 */
window.LA = window.LA || {};
(function () {
  'use strict';
  const TAU = Math.PI * 2;
  const N = () => LA.num;

  /** Coerce any seed to a uint32 (NaN/Infinity → 0, negatives wrap). */
  const toSeed = (seed) => { const x = Number(seed); return isFinite(x) ? Math.floor(x) >>> 0 : 0; };

  /** LA.rng(seed) → mulberry32 generator returning floats in [0,1); same seed → same sequence. */
  const rng = function (seed) {
    let a = toSeed(seed);
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  // ===================== sampling helpers =====================
  /** Standard normal sample via Box–Muller from the given rand() function. */
  rng.gauss = (rand) => { const u1 = 1 - rand(), u2 = rand(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TAU * u2); };
  /** n independent standard normal samples. */
  rng.gaussV = (rand, n) => { const v = new Array(n); for (let i = 0; i < n; i++) v[i] = rng.gauss(rand); return v; };
  /** Fisher–Yates shuffle; returns a new array. */
  rng.shuffle = (rand, arr) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; };
  /** Uniformly pick one element of arr. */
  rng.pick = (rand, arr) => arr[Math.floor(rand() * arr.length)];
  /** Uniform integer in the INCLUSIVE range [lo, hi]. */
  rng.int = (rand, lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  /** Uniformly distributed unit vector in R^d (normalized Gaussian). */
  rng.randomUnit = (rand, d) => { const v = N().normalize(rng.gaussV(rand, d)); return N().norm(v) > 0 ? v : N().basis(0, d); };

  // ===================== data generators =====================
  /** Gaussian cloud: n points with the given mean (default [0,0]) and covariance (default I), via chol(cov) or eigSym square root if not PD. */
  rng.gaussianCloud = (seed, n, opts) => {
    const o = opts || {}; const mean = o.mean || [0, 0]; const d = mean.length; const cov = o.cov || N().identity(d);
    let L = N().chol(cov);
    if (!L) { const e = N().eigSym(cov); L = N().fromCols(e.vectors.map((q, i) => N().scale(q, Math.sqrt(Math.max(e.values[i], 0))))); }
    const rand = rng(seed); const pts = [];
    for (let i = 0; i < n; i++) pts.push(N().add(mean, N().matVec(L, rng.gaussV(rand, d))));
    return pts;
  };
  /** Two concentric noisy circles (radii r1, r2) → {X, y} with labels 0 (inner) / 1 (outer), alternating. */
  rng.circles = (seed, n, opts) => {
    const o = opts || {}; const r1 = o.r1 == null ? 1 : o.r1, r2 = o.r2 == null ? 2.2 : o.r2, noise = o.noise == null ? 0.08 : o.noise;
    const rand = rng(seed); const X = [], y = [];
    for (let i = 0; i < n; i++) { const label = i % 2; const r = label ? r2 : r1; const t = rand() * TAU; X.push([r * Math.cos(t) + noise * rng.gauss(rand), r * Math.sin(t) + noise * rng.gauss(rand)]); y.push(label); }
    return { X, y };
  };
  /** Two interleaved spirals centered at the origin (radius ≤ 2.5) → {X, y}. */
  rng.spirals = (seed, n, opts) => {
    const o = opts || {}; const turns = o.turns == null ? 1.5 : o.turns, noise = o.noise == null ? 0.05 : o.noise;
    const rand = rng(seed); const X = [], y = []; const rMin = 0.2, rMax = Math.max(rMin + 0.1, 2.5 - 4 * Math.abs(noise));
    for (let i = 0; i < n; i++) {
      const label = i % 2; const u = rand(); const r = rMin + (rMax - rMin) * u; const ang = turns * TAU * u + label * Math.PI;
      let p = [r * Math.cos(ang) + noise * rng.gauss(rand), r * Math.sin(ang) + noise * rng.gauss(rand)];
      const nr = Math.hypot(p[0], p[1]); if (nr > 2.5) p = [p[0] * 2.5 / nr, p[1] * 2.5 / nr];
      X.push(p); y.push(label);
    }
    return { X, y };
  };
  /** Two interleaved half-moons centered at the origin (scaled ×1.6) → {X, y}. */
  rng.moons = (seed, n, opts) => {
    const o = opts || {}; const noise = o.noise == null ? 0.08 : o.noise; const s = 1.6;
    const rand = rng(seed); const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const label = i % 2; const t = rand() * Math.PI;
      const p = label === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
      X.push([(p[0] - 0.5) * s + noise * rng.gauss(rand), (p[1] - 0.25) * s + noise * rng.gauss(rand)]); y.push(label);
    }
    return { X, y };
  };
  /** Gaussian blobs around the given centers (default three 2-D centers; d inferred from centers) → {X, y} with y = center index. */
  rng.blobs = (seed, n, opts) => {
    const o = opts || {}; const centers = o.centers || [[-1.5, -1], [1.5, -1], [0, 1.5]]; const spread = o.spread == null ? 0.4 : o.spread; const d = centers[0].length;
    const rand = rng(seed); const X = [], y = [];
    for (let i = 0; i < n; i++) { const label = i % centers.length; X.push(N().add(centers[label], N().scale(rng.gaussV(rand, d), spread))); y.push(label); }
    return { X, y };
  };
  /** 1-D regression data y = w·x + b + noise → {x (scalars when w has one entry, else vectors), y, w, b}. */
  rng.linearData = (seed, n, opts) => {
    const o = opts || {}; const w = o.w || [0.8]; const b = o.b == null ? 0.3 : o.b; const noise = o.noise == null ? 0.25 : o.noise; const xr = o.xRange || [-2, 2];
    const rand = rng(seed); const x = [], y = []; const p = w.length;
    for (let i = 0; i < n; i++) { const xi = []; for (let j = 0; j < p; j++) xi.push(xr[0] + rand() * (xr[1] - xr[0])); y.push(b + N().dot(w, xi) + noise * rng.gauss(rand)); x.push(p === 1 ? xi[0] : xi); }
    return { x, y, w: w.slice(), b };
  };
  /** 2-D regression data y = w·X + b + noise → {X (n×2), y, w, b}. */
  rng.planeData = (seed, n, opts) => {
    const o = opts || {}; const w = o.w || [0.6, -0.4]; const b = o.b == null ? 0.2 : o.b; const noise = o.noise == null ? 0.2 : o.noise; const xr = o.xRange || [-2, 2];
    const rand = rng(seed); const X = [], y = [];
    for (let i = 0; i < n; i++) { const xi = w.map(() => xr[0] + rand() * (xr[1] - xr[0])); X.push(xi); y.push(b + N().dot(w, xi) + noise * rng.gauss(rand)); }
    return { X, y, w: w.slice(), b };
  };

  // Procedural digit strokes in the unit square (x right, y down), as polylines.
  const L = 0.27, R = 0.73, T = 0.12, M = 0.5, B = 0.88, C = 0.5;
  const DIGIT_STROKES = [
    [[[L, T + 0.1], [L + 0.1, T], [R - 0.1, T], [R, T + 0.1], [R, B - 0.1], [R - 0.1, B], [L + 0.1, B], [L, B - 0.1], [L, T + 0.1]]],
    [[[C - 0.12, T + 0.14], [C, T], [C, B]]],
    [[[L, T + 0.1], [L + 0.1, T], [R - 0.1, T], [R, T + 0.12], [R, M - 0.05], [L, B], [R, B]]],
    [[[L, T + 0.08], [L + 0.1, T], [R - 0.1, T], [R, T + 0.1], [R, M - 0.08], [C - 0.05, M], [R, M + 0.08], [R, B - 0.1], [R - 0.1, B], [L + 0.1, B], [L, B - 0.08]]],
    [[[R - 0.12, T], [L, M + 0.05], [R, M + 0.05]], [[R - 0.12, T], [R - 0.12, B]]],
    [[[R, T], [L, T], [L, M - 0.02], [R - 0.1, M - 0.05], [R, M + 0.1], [R, B - 0.1], [R - 0.1, B], [L, B - 0.05]]],
    [[[R - 0.05, T], [L + 0.05, M - 0.1], [L, M + 0.15], [L, B - 0.1], [L + 0.1, B], [R - 0.1, B], [R, B - 0.1], [R, M + 0.12], [R - 0.1, M], [L, M + 0.05]]],
    [[[L, T], [R, T], [C - 0.05, B]]],
    [[[C, T], [R - 0.05, T + 0.08], [R - 0.05, M - 0.08], [C, M], [L + 0.05, M - 0.08], [L + 0.05, T + 0.08], [C, T]], [[C, M], [R, M + 0.08], [R, B - 0.08], [C, B], [L, B - 0.08], [L, M + 0.08], [C, M]]],
    [[[R, M - 0.05], [R - 0.1, M + 0.02], [L + 0.1, M + 0.02], [L, M - 0.1], [L, T + 0.1], [L + 0.1, T], [R - 0.1, T], [R, T + 0.1], [R, M - 0.05], [R, B - 0.1], [R - 0.1, B], [L + 0.1, B]]],
  ];
  /** Distance from point (px,py) to the segment (ax,ay)–(bx,by). */
  const segDist = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax, dy = by - ay; const len2 = dx * dx + dy * dy;
    let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  };
  /** Procedural digits: n samples {pixels (size² floats in [0,1], row-major), label 0–9} rendered from jittered stroke polylines with distance-based anti-aliasing. */
  rng.syntheticDigits = (seed, n, opts) => {
    const o = opts || {}; const size = o.size || 8; const rand = rng(seed); const out = [];
    const aa = 0.7 / size;
    for (let s = 0; s < n; s++) {
      const label = s % 10;
      const rot = (rand() - 0.5) * 0.24, sx = 0.9 + rand() * 0.2, sy = 0.9 + rand() * 0.2, tx = (rand() - 0.5) * 0.08, ty = (rand() - 0.5) * 0.08;
      const hw = 0.045 + rand() * 0.03; const cr = Math.cos(rot), sr = Math.sin(rot);
      const segs = [];
      for (const poly of DIGIT_STROKES[label]) {
        const pts = poly.map(([x, y]) => {
          const jx = x + 0.015 * rng.gauss(rand), jy = y + 0.015 * rng.gauss(rand);
          const cx = (jx - 0.5) * sx, cy = (jy - 0.5) * sy;
          return [0.5 + cr * cx - sr * cy + tx, 0.5 + sr * cx + cr * cy + ty];
        });
        for (let i = 1; i < pts.length; i++) segs.push([pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]]);
      }
      const pixels = new Array(size * size);
      for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
        const px = (c + 0.5) / size, py = (r + 0.5) / size; let d = Infinity;
        for (const g of segs) { const dd = segDist(px, py, g[0], g[1], g[2], g[3]); if (dd < d) d = dd; }
        let v = (hw + aa - d) / aa; v = v < 0 ? 0 : v > 1 ? 1 : v;
        v += 0.02 * rng.gauss(rand); pixels[r * size + c] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
      out.push({ pixels, label });
    }
    return out;
  };
  /** size×size grayscale image in [0,1]: smooth gradient + soft disc + dark rectangle + bright diagonal stripe (low effective rank). */
  rng.smallImage = (seed, opts) => {
    const o = opts || {}; const size = o.size || 32; const rand = rng(seed);
    const gx = 0.4 + 0.4 * rand(), gy = 0.2 + 0.4 * rand();
    const disc = { cx: 0.25 + 0.2 * rand(), cy: 0.25 + 0.2 * rand(), r: 0.12 + 0.08 * rand() };
    const rect = { x0: 0.55 + 0.1 * rand(), y0: 0.55 + 0.1 * rand(), w: 0.2 + 0.1 * rand(), h: 0.15 + 0.1 * rand() };
    const stripe = { off: -0.2 + 0.4 * rand(), w: 0.04 + 0.03 * rand() };
    const px = 1 / size; const img = [];
    for (let r = 0; r < size; r++) {
      const row = new Array(size); const y = (r + 0.5) / size;
      for (let c = 0; c < size; c++) {
        const x = (c + 0.5) / size; let v = 0.15 + 0.5 * (gx * x + gy * y) / (gx + gy);
        const e = Math.min(1, Math.max(0, (disc.r - Math.hypot(x - disc.cx, y - disc.cy)) / px + 0.5)); v += (0.95 - v) * e;
        if (x >= rect.x0 && x <= rect.x0 + rect.w && y >= rect.y0 && y <= rect.y0 + rect.h) v = 0.08;
        if (Math.abs(x - y - stripe.off) < stripe.w) v = Math.max(v, 0.9);
        row[c] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
      img.push(row);
    }
    return img;
  };
  /**
   * Deterministic toy word embeddings (3-D). Construction: axis x = gender (−1 male … +1 female), axis y = royalty
   * (0 common, 1 royal), axis z = generation (+0.5 adult, −0.5 child, +0.9 elder). People sit on a lattice so that
   * king − man + woman = queen and prince − boy + girl = princess exactly, then fixed small offsets (≤ 0.02) are added
   * for a natural look (analogy error stays < 0.15). Food and colour words live in a distant cluster (y ≈ −2.2, z ≈ −1.6)
   * with apple ↔ red and banana ↔ yellow related by the same +4.8 shift in x, so apple − red + yellow ≈ banana.
   */
  rng.wordVectors = () => [
    { word: 'king', v: [-1 + 0.02, 1 - 0.01, 0.5 + 0.01] },
    { word: 'queen', v: [1 - 0.01, 1 + 0.02, 0.5 - 0.01] },
    { word: 'man', v: [-1 - 0.01, 0 + 0.01, 0.5 + 0.02] },
    { word: 'woman', v: [1 + 0.02, 0 - 0.02, 0.5 - 0.01] },
    { word: 'boy', v: [-1 + 0.01, 0 - 0.01, -0.5 + 0.02] },
    { word: 'girl', v: [1 - 0.02, 0 + 0.01, -0.5 - 0.01] },
    { word: 'prince', v: [-1 - 0.02, 1 + 0.01, -0.5 - 0.01] },
    { word: 'princess', v: [1 + 0.01, 1 - 0.02, -0.5 + 0.02] },
    { word: 'uncle', v: [-1 + 0.01, 0.3 + 0.02, 0.9 - 0.01] },
    { word: 'aunt', v: [1 - 0.02, 0.3 - 0.01, 0.9 + 0.01] },
    { word: 'apple', v: [-2.4, -2.2, -1.5] },
    { word: 'banana', v: [-2.0, -2.5, -1.7] },
    { word: 'red', v: [2.4, -2.2, -1.5] },
    { word: 'yellow', v: [2.8, -2.5, -1.7] },
  ];
  /** n random unit vectors in R^d. */
  rng.tokenVectors = (seed, n, d) => { const rand = rng(seed); const out = []; for (let i = 0; i < n; i++) out.push(rng.randomUnit(rand, d)); return out; };
  /**
   * Histogram of cosines between random unit vectors in R^d → {bins (edges −1..1, length bins+1), counts, d, pairs, poolSize}.
   * Vectors are drawn from a pool (≤ ~2e5 Gaussian samples total, so d = 2000 uses a pool of 100 vectors) and
   * paired at random without repetition inside a pair; for small d the pool is 2·pairs and pairs are fully independent.
   */
  rng.cosineHistogram = (seed, d, pairs = 2000, bins = 40) => {
    const rand = rng(seed); d = Math.max(1, d | 0);
    const poolSize = Math.max(2, Math.min(2 * pairs, Math.max(Math.ceil(Math.sqrt(2 * pairs)) + 2, Math.floor(2e5 / d))));
    let spare = null;
    const g = () => { if (spare !== null) { const v = spare; spare = null; return v; } const u1 = 1 - rand(), u2 = rand(); const r = Math.sqrt(-2 * Math.log(u1)); spare = r * Math.sin(TAU * u2); return r * Math.cos(TAU * u2); };
    const pool = [], norms = [];
    for (let p = 0; p < poolSize; p++) { const v = new Float64Array(d); let s = 0; for (let k = 0; k < d; k++) { const x = g(); v[k] = x; s += x * x; } pool.push(v); norms.push(Math.sqrt(s) || 1); }
    const edges = []; for (let b = 0; b <= bins; b++) edges.push(-1 + (2 * b) / bins);
    const counts = new Array(bins).fill(0); const disjoint = poolSize === 2 * pairs;
    for (let q = 0; q < pairs; q++) {
      let i, j;
      if (disjoint) { i = 2 * q; j = 2 * q + 1; } else { i = Math.floor(rand() * poolSize); j = Math.floor(rand() * (poolSize - 1)); if (j >= i) j++; }
      const a = pool[i], b = pool[j]; let dp = 0; for (let k = 0; k < d; k++) dp += a[k] * b[k];
      const cosv = dp / (norms[i] * norms[j]); let idx = Math.floor(((cosv + 1) / 2) * bins); if (idx < 0) idx = 0; if (idx >= bins) idx = bins - 1; counts[idx]++;
    }
    return { bins: edges, counts, d, pairs, poolSize };
  };

  LA.rng = rng;
})();
