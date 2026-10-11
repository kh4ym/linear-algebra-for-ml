/*
 * js/core/num.js — pure, dependency-free numerical linear algebra for the LA course.
 * Classic script: attaches LA.num. Matrices are arrays of row arrays (M[i][j], row-major);
 * vectors are flat arrays. Every function is PURE (never mutates inputs, always returns new arrays).
 * Singular inverse/solve return null (never throw). Default tolerance eps = 1e-10 unless stated.
 * "sign-fixed" = each eigen/singular vector / principal component is scaled so that its
 * largest-magnitude component (first one on ties) is positive. Decompositions are sorted DESCENDING.
 * Deterministic: no Date, no Math.random anywhere.
 */
window.LA = window.LA || {};
(function () {
  'use strict';
  const num = {};
  const EPS = 1e-10;
  num.EPS = EPS;

  // ===================== internal helpers =====================
  /** sign(x) with sign(0) = +1 (used by Jacobi rotation formulas). */
  const sgn1 = (x) => (x < 0 ? -1 : 1);
  /** Deep copy of a matrix (array of rows). */
  const copyM = (M) => M.map((r) => r.slice());
  /** Largest absolute entry of a matrix (0 for empty). */
  const maxAbs = (M) => { let s = 0; for (const r of M) for (const x of r) { const a = Math.abs(x); if (a > s) s = a; } return s; };
  /** Return a copy of v with its sign flipped so the largest-magnitude component (first on ties) is positive. */
  const signFix = (v) => { let im = 0; for (let i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[im])) im = i; return v[im] < 0 ? v.map((x) => -x) : v.slice(); };
  /** Tiny deterministic LCG returning floats in [0,1) for internal start vectors (never Math.random). */
  const lcg = (seed) => { let s = (seed >>> 0) || 1; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; };
  /** Extend the orthonormal list `existing` (vectors in R^dim) by `count` new orthonormal vectors via Gram–Schmidt against the standard basis. */
  function completeBasis(existing, dim, count) {
    const basis = existing.slice();
    const out = [];
    for (let c = 0; c < count; c++) {
      let best = null, bestNorm = -1;
      for (let e = 0; e < dim; e++) {
        let r = num.basis(e, dim);
        for (let pass = 0; pass < 2; pass++) for (const q of basis) r = num.sub(r, num.scale(q, num.dot(r, q)));
        const nr = num.norm(r);
        if (nr > bestNorm) { bestNorm = nr; best = r; }
      }
      if (!best || bestNorm < 1e-8) break;
      const q = num.scale(best, 1 / bestNorm);
      basis.push(q);
      out.push(q);
    }
    return out;
  }

  // ===================== vectors =====================
  /** Elementwise a + b. */
  num.add = (a, b) => a.map((x, i) => x + b[i]);
  /** Elementwise a − b. */
  num.sub = (a, b) => a.map((x, i) => x - b[i]);
  /** Scalar multiple k·a. */
  num.scale = (a, k) => a.map((x) => x * k);
  /** Dot product a·b. */
  num.dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  /** Euclidean norm ‖a‖₂. */
  num.norm = (a) => Math.sqrt(num.dot(a, a));
  /** Manhattan norm ‖a‖₁. */
  num.norm1 = (a) => a.reduce((s, x) => s + Math.abs(x), 0);
  /** Max norm ‖a‖∞. */
  num.normInf = (a) => a.reduce((s, x) => Math.max(s, Math.abs(x)), 0);
  /** p-norm ‖a‖ₚ (p = Infinity gives the max norm). */
  num.normP = (a, p) => (p === Infinity ? num.normInf(a) : Math.pow(a.reduce((s, x) => s + Math.pow(Math.abs(x), p), 0), 1 / p));
  /** Unit vector a/‖a‖ (zero vector stays zero). */
  num.normalize = (a) => { const n = num.norm(a); return n > 0 ? a.map((x) => x / n) : a.map(() => 0); };
  /** Euclidean distance ‖a − b‖. */
  num.dist = (a, b) => num.norm(num.sub(a, b));
  /** Cross product: 3-vectors → 3-vector; 2-vectors → scalar z-component a₀b₁ − a₁b₀. */
  num.cross = (a, b) => (a.length === 2 ? a[0] * b[1] - a[1] * b[0] : [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
  /** Angle between a and b in radians via the stable form 2·atan2(‖‖b‖a − ‖a‖b‖, ‖‖b‖a + ‖a‖b‖) (exact near 0 and π; 0 if either vector is zero). */
  num.angle = (a, b) => { const na = num.norm(a), nb = num.norm(b); if (na === 0 || nb === 0) return 0; const u = num.scale(a, nb), w = num.scale(b, na); return 2 * Math.atan2(num.norm(num.sub(u, w)), num.norm(num.add(u, w))); };
  /** Vector projection of v onto `onto`: (v·onto / onto·onto)·onto (zero if onto = 0). */
  num.proj = (v, onto) => { const d = num.dot(onto, onto); if (d === 0) return onto.map(() => 0); return num.scale(onto, num.dot(v, onto) / d); };
  /** Linear interpolation a + t(b − a) elementwise. */
  num.lerpV = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
  /** Zero vector of length n. */
  num.zeros = (n) => new Array(n).fill(0);
  /** Standard basis vector e_i in R^n. */
  num.basis = (i, n) => { const v = num.zeros(n); v[i] = 1; return v; };
  /** Map fn(x, i) over the components of v. */
  num.mapV = (v, fn) => v.map((x, i) => fn(x, i));
  /** Sum of components. */
  num.sum = (v) => v.reduce((s, x) => s + x, 0);
  /** Mean of the components of a flat vector (0 for empty). */
  num.mean1 = (v) => (v.length ? num.sum(v) / v.length : 0);

  // ===================== matrices =====================
  /** n×n identity. */
  num.identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  /** r×c zero matrix. */
  num.zerosM = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  /** [rows, cols] of M. */
  num.shape = (M) => [M.length, M.length ? M[0].length : 0];
  /** Column j of M as a vector. */
  num.col = (M, j) => M.map((r) => r[j]);
  /** Row i of M as a (copied) vector. */
  num.row = (M, i) => M[i].slice();
  /** All columns of M as an array of column vectors. */
  num.cols = (M) => { const [r, c] = num.shape(M); const out = []; for (let j = 0; j < c; j++) { const v = new Array(r); for (let i = 0; i < r; i++) v[i] = M[i][j]; out.push(v); } return out; };
  /** Build a matrix (array of rows) from an array of column vectors. */
  num.fromCols = (cs) => { if (!cs.length) return []; const r = cs[0].length; const M = []; for (let i = 0; i < r; i++) { const row = new Array(cs.length); for (let j = 0; j < cs.length; j++) row[j] = cs[j][i]; M.push(row); } return M; };
  /** Transpose. */
  num.transpose = (M) => num.cols(M);
  /** Matrix product A·B. */
  num.matMul = (A, B) => {
    const [m, k] = num.shape(A); const n = B.length ? B[0].length : 0; const C = num.zerosM(m, n);
    for (let i = 0; i < m; i++) { const Ai = A[i], Ci = C[i]; for (let p = 0; p < k; p++) { const a = Ai[p]; if (a === 0) continue; const Bp = B[p]; for (let j = 0; j < n; j++) Ci[j] += a * Bp[j]; } }
    return C;
  };
  /** Matrix–vector product M·v. */
  num.matVec = (M, v) => M.map((r) => num.dot(r, v));
  /** Outer product u vᵀ (u.length × v.length). */
  num.outer = (u, v) => u.map((x) => v.map((y) => x * y));
  /** Elementwise A + B. */
  num.addM = (A, B) => A.map((r, i) => r.map((x, j) => x + B[i][j]));
  /** Elementwise A − B. */
  num.subM = (A, B) => A.map((r, i) => r.map((x, j) => x - B[i][j]));
  /** Scalar multiple k·A. */
  num.scaleM = (A, k) => A.map((r) => r.map((x) => x * k));
  /** Entrywise interpolation A + t(B − A). */
  num.lerpM = (A, B, t) => A.map((r, i) => r.map((x, j) => x + (B[i][j] - x) * t));
  /** Trace (sum of diagonal). */
  num.trace = (M) => { let s = 0; for (let i = 0; i < M.length; i++) s += M[i][i]; return s; };
  /** Frobenius norm. */
  num.frobenius = (M) => { let s = 0; for (const r of M) for (const x of r) s += x * x; return Math.sqrt(s); };
  /** Determinant via LU with partial pivoting (any n). */
  function detLU(M) {
    const n = M.length; const A = copyM(M); let d = 1;
    for (let k = 0; k < n; k++) {
      let p = k; for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      if (A[p][k] === 0) return 0;
      if (p !== k) { const t = A[p]; A[p] = A[k]; A[k] = t; d = -d; }
      d *= A[k][k];
      for (let i = k + 1; i < n; i++) { const f = A[i][k] / A[k][k]; if (f !== 0) for (let j = k; j < n; j++) A[i][j] -= f * A[k][j]; }
    }
    return d;
  }
  /** Determinant: closed form for n ≤ 3, LU with partial pivoting otherwise. */
  num.det = (M) => {
    const n = M.length;
    if (n === 0) return 1;
    if (n === 1) return M[0][0];
    if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (n === 3) {
      const [[a, b, c], [d, e, f], [g, h, i]] = M;
      return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }
    return detLU(M);
  };
  /** Inverse via Gauss–Jordan with partial pivoting; null if a pivot ≤ eps·max|M| (singular). */
  num.inv = (M, eps = EPS) => {
    const n = M.length; if (!n || M[0].length !== n) return null;
    const scale = maxAbs(M); if (scale === 0) return null;
    const A = M.map((r, i) => r.concat(num.basis(i, n)));
    for (let k = 0; k < n; k++) {
      let p = k; for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      if (Math.abs(A[p][k]) <= eps * scale) return null;
      if (p !== k) { const t = A[p]; A[p] = A[k]; A[k] = t; }
      const piv = A[k][k]; for (let j = 0; j < 2 * n; j++) A[k][j] /= piv;
      for (let i = 0; i < n; i++) if (i !== k) { const f = A[i][k]; if (f !== 0) for (let j = 0; j < 2 * n; j++) A[i][j] -= f * A[k][j]; }
    }
    return A.map((r) => r.slice(n));
  };
  /** Solve A x = b (Gaussian elimination, partial pivoting); null if singular. */
  num.solve = (A, b, eps = EPS) => {
    const n = A.length; if (!n || A[0].length !== n || b.length !== n) return null;
    const scale = maxAbs(A); if (scale === 0) return null;
    const M = A.map((r, i) => r.concat([b[i]]));
    for (let k = 0; k < n; k++) {
      let p = k; for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      if (Math.abs(M[p][k]) <= eps * scale) return null;
      if (p !== k) { const t = M[p]; M[p] = M[k]; M[k] = t; }
      for (let i = k + 1; i < n; i++) { const f = M[i][k] / M[k][k]; if (f !== 0) for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  };
  /** Numerical rank: number of singular values > eps·σmax. */
  num.rank = (A, eps = 1e-9) => { const S = num.svd(A).S; const smax = S[0] || 0; if (smax === 0) return 0; return S.filter((s) => s > eps * smax).length; };
  /** Orthonormal basis of the null space (right singular vectors with σ ≤ eps·σmax, completed to full dimension for wide A). */
  num.nullSpace = (A, eps = 1e-9) => {
    const [, n] = num.shape(A); const { S, V } = num.svd(A); const smax = S[0] || 0; const Vc = num.cols(V);
    const keep = [], nul = [];
    for (let k = 0; k < Vc.length; k++) { if (smax > 0 && S[k] > eps * smax) keep.push(Vc[k]); else nul.push(Vc[k]); }
    const extra = completeBasis(keep.concat(nul), n, n - Vc.length);
    return nul.concat(extra).map(signFix);
  };
  /** Orthonormal basis of the column space (left singular vectors with σ > eps·σmax). */
  num.colSpace = (A, eps = 1e-9) => { const { S, U } = num.svd(A); const smax = S[0] || 0; const Uc = num.cols(U); const out = []; for (let k = 0; k < Uc.length; k++) if (smax > 0 && S[k] > eps * smax) out.push(Uc[k]); return out; };
  /** True if A is square and |A[i][j] − A[j][i]| ≤ eps·max(1, max|A|). */
  num.isSymmetric = (A, eps = EPS) => { const n = A.length; if (!n || A[0].length !== n) return false; const tol = eps * Math.max(1, maxAbs(A)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(A[i][j] - A[j][i]) > tol) return false; return true; };
  /** Cholesky factor: lower-triangular L with M = L Lᵀ; null if M is not positive definite. */
  num.chol = (M) => {
    const n = M.length; const L = num.zerosM(n, n); let dmax = 0; for (let i = 0; i < n; i++) dmax = Math.max(dmax, Math.abs(M[i][i]));
    for (let j = 0; j < n; j++) {
      let s = M[j][j]; for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
      if (!(s > 1e-14 * dmax) || !(s > 0)) return null;
      L[j][j] = Math.sqrt(s);
      for (let i = j + 1; i < n; i++) { let t = M[i][j]; for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]; L[i][j] = t / L[j][j]; }
    }
    return L;
  };
  /** Square diagonal matrix from a vector of values. */
  num.diag = (values) => values.map((v, i) => values.map((_, j) => (i === j ? v : 0)));
  /** Diagonal of M as a vector. */
  num.diagOf = (M) => M.map((r, i) => r[i]);
  /** Horizontal concatenation [A | B]. */
  num.hstack = (A, B) => A.map((r, i) => r.concat(B[i]));
  /** Vertical concatenation [A ; B]. */
  num.vstack = (A, B) => A.map((r) => r.slice()).concat(B.map((r) => r.slice()));
  /** Submatrix picking the given row indices and column indices (either may be omitted for "all"). */
  num.submatrix = (M, rows, cols) => { const [r, c] = num.shape(M); const ri = rows || Array.from({ length: r }, (_, i) => i); const ci = cols || Array.from({ length: c }, (_, j) => j); return ri.map((i) => ci.map((j) => M[i][j])); };
  /** Integer matrix power A^k (k < 0 uses the inverse; null if singular). */
  num.matPow = (A, k) => {
    k = Math.round(k); let base = A; if (k < 0) { base = num.inv(A); if (!base) return null; k = -k; }
    let result = num.identity(A.length);
    while (k > 0) { if (k & 1) result = num.matMul(result, base); k >>= 1; if (k) base = num.matMul(base, base); }
    return result;
  };

  // ===================== constructions =====================
  /** 2×2 counter-clockwise rotation by theta radians. */
  num.rot2 = (t) => { const c = Math.cos(t), s = Math.sin(t); return [[c, -s], [s, c]]; };
  /** 3×3 rotation about `axis` (normalized internally) by theta radians (Rodrigues formula). */
  num.rot3 = (axis, t) => {
    const k = num.normalize(axis); if (num.norm(k) === 0) return num.identity(3);
    const c = Math.cos(t), s = Math.sin(t), v = 1 - c; const [x, y, z] = k;
    return [[c + x * x * v, x * y * v - z * s, x * z * v + y * s], [y * x * v + z * s, c + y * y * v, y * z * v - x * s], [z * x * v - y * s, z * y * v + x * s, c + z * z * v]];
  };
  /** Rotation about the x axis. */
  num.rotX = (t) => { const c = Math.cos(t), s = Math.sin(t); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
  /** Rotation about the y axis. */
  num.rotY = (t) => { const c = Math.cos(t), s = Math.sin(t); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; };
  /** Rotation about the z axis. */
  num.rotZ = (t) => { const c = Math.cos(t), s = Math.sin(t); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
  /** 2×2 axis scaling diag(sx, sy). */
  num.scale2 = (sx, sy) => [[sx, 0], [0, sy]];
  /** 3×3 axis scaling diag(sx, sy, sz). */
  num.scale3 = (sx, sy, sz) => [[sx, 0, 0], [0, sy, 0], [0, 0, sz]];
  /** 2×2 shear [[1, kx], [ky, 1]]. */
  num.shear2 = (kx, ky = 0) => [[1, kx], [ky, 1]];
  /** Reflection across the line through the origin at angle `a` (radians). */
  num.reflect2 = (a) => { const c = Math.cos(2 * a), s = Math.sin(2 * a); return [[c, s], [s, -c]]; };
  /** Orthogonal projection onto the line through the origin at angle `a`. */
  num.project2 = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c * c, c * s], [c * s, s * s]]; };
  /** Projection matrix onto span(vectors) (any dimension), via Gram–Schmidt orthonormalization. */
  num.projectOnto = (vectors) => { if (!vectors.length) return []; const d = vectors[0].length; const { basis } = num.gramSchmidt(vectors); let P = num.zerosM(d, d); for (const q of basis) P = num.addM(P, num.outer(q, q)); return P; };

  // ===================== decompositions =====================
  /** 2×2 eigen-decomposition: {real:true, values (desc), vectors (unit, sign-fixed), defective} or {real:false, re, im} for a complex pair. */
  num.eig2 = (A, eps = EPS) => {
    const [[a, b], [c, d]] = A;
    const tr = a + d, det = a * d - b * c; const disc = tr * tr / 4 - det;
    const scale = Math.max(1, Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
    if (disc < -eps * scale * scale) return { real: false, re: tr / 2, im: Math.sqrt(-disc) };
    const sq = Math.sqrt(Math.max(disc, 0)); const l1 = tr / 2 + sq, l2 = tr / 2 - sq;
    const vecFor = (l) => {
      let v;
      if (Math.abs(b) > eps * scale) v = [b, l - a];
      else if (Math.abs(c) > eps * scale) v = [l - d, c];
      else v = Math.abs(a - l) <= Math.abs(d - l) ? [1, 0] : [0, 1];
      return signFix(num.normalize(v));
    };
    if (Math.abs(l1 - l2) <= eps * scale) {
      if (Math.abs(b) <= eps * scale && Math.abs(c) <= eps * scale && Math.abs(a - d) <= eps * scale) return { real: true, values: [l1, l2], vectors: [[1, 0], [0, 1]], defective: false };
      return { real: true, values: [l1, l2], vectors: [vecFor(l1)], defective: true };
    }
    return { real: true, values: [l1, l2], vectors: [vecFor(l1), vecFor(l2)], defective: false };
  };
  /** Symmetric eigen-decomposition by cyclic Jacobi (n ≤ 64, tol 1e-12·‖A‖F on off-diagonal norm, ≤ 100 sweeps): {values desc, vectors orthonormal sign-fixed}. */
  num.eigSym = (A) => {
    const n = A.length; const M = copyM(A);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const avg = (M[i][j] + M[j][i]) / 2; M[i][j] = avg; M[j][i] = avg; }
    const V = num.identity(n); const normF = num.frobenius(M);
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] * M[i][j];
      if (Math.sqrt(2 * off) <= 1e-12 * normF) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = M[p][q]; if (apq === 0) continue;
        const theta = (M[q][q] - M[p][p]) / (2 * apq);
        const t = Math.abs(theta) > 1e150 ? 1 / (2 * theta) : sgn1(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const mkp = M[k][p], mkq = M[k][q]; M[k][p] = c * mkp - s * mkq; M[k][q] = s * mkp + c * mkq; }
        for (let k = 0; k < n; k++) { const mpk = M[p][k], mqk = M[q][k]; M[p][k] = c * mpk - s * mqk; M[q][k] = s * mpk + c * mqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
        M[p][q] = 0; M[q][p] = 0;
      }
    }
    const vals = num.diagOf(M); const vecs = num.cols(V);
    const order = vals.map((_, i) => i).sort((i, j) => vals[j] - vals[i]);
    return { values: order.map((i) => vals[i]), vectors: order.map((i) => signFix(vecs[i])) };
  };
  /** One-sided Jacobi (Hestenes) SVD for m ≥ n; returns column lists (no sign fixing). */
  function svdTall(A) {
    const [m, n] = num.shape(A); const U = num.cols(A); const V = num.cols(num.identity(n)); const tol = 1e-14;
    for (let sweep = 0; sweep < 60; sweep++) {
      let rotated = false;
      for (let i = 0; i < n - 1; i++) for (let j = i + 1; j < n; j++) {
        const ui = U[i], uj = U[j]; let alpha = 0, beta = 0, gamma = 0;
        for (let k = 0; k < m; k++) { alpha += ui[k] * ui[k]; beta += uj[k] * uj[k]; gamma += ui[k] * uj[k]; }
        if (alpha === 0 || beta === 0 || Math.abs(gamma) <= tol * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = sgn1(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let k = 0; k < m; k++) { const a = ui[k], b = uj[k]; ui[k] = c * a - s * b; uj[k] = s * a + c * b; }
        const vi = V[i], vj = V[j];
        for (let k = 0; k < n; k++) { const a = vi[k], b = vj[k]; vi[k] = c * a - s * b; vj[k] = s * a + c * b; }
      }
      if (!rotated) break;
    }
    const S = U.map((u) => num.norm(u)); const order = S.map((_, i) => i).sort((a, b) => S[b] - S[a]); const smax = S[order[0]] || 0;
    const Sout = [], Uc = [], Vc = [], zeroIdx = [];
    for (const idx of order) {
      const s = S[idx];
      if (smax > 0 && s > 1e-14 * smax) { Sout.push(s); Uc.push(U[idx].map((x) => x / s)); Vc.push(V[idx]); }
      else { Sout.push(0); Uc.push(null); Vc.push(V[idx]); zeroIdx.push(Sout.length - 1); }
    }
    if (zeroIdx.length) { const extra = completeBasis(Uc.filter((u) => u), m, zeroIdx.length); zeroIdx.forEach((k, i) => { Uc[k] = extra[i]; }); }
    return { Uc, S: Sout, Vc };
  }
  /** Thin SVD A = U·diag(S)·Vᵀ (U m×k, V n×k, k = min(m,n), S descending ≥ 0). Sign convention: each V column is sign-fixed and U follows; for σ = 0 both are sign-fixed independently. */
  num.svd = (A) => {
    const [m, n] = num.shape(A);
    let Uc, S, Vc;
    if (m >= n) ({ Uc, S, Vc } = svdTall(A));
    else { const r = svdTall(num.transpose(A)); Uc = r.Vc; S = r.S; Vc = r.Uc; }
    for (let k = 0; k < S.length; k++) {
      if (S[k] > 0) { const f = signFix(Vc[k]); if (num.dot(f, Vc[k]) < 0) Uc[k] = num.scale(Uc[k], -1); Vc[k] = f; }
      else { Vc[k] = signFix(Vc[k]); Uc[k] = signFix(Uc[k]); }
    }
    return { U: num.fromCols(Uc), S, V: num.fromCols(Vc) };
  };
  /** 2×2 rotate–stretch–rotate SVD: A = rot2(thetaU)·diag(S)·rot2(thetaV)ᵀ with both factors proper rotations; S[1] < 0 when det(A) < 0 (reflection folded into Σ). Closed form. */
  num.svd2rot = (A) => {
    const [[a, b], [c, d]] = A;
    const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
    const Q = Math.hypot(E, H), R = Math.hypot(F, G);
    const a1 = (F === 0 && G === 0) ? 0 : Math.atan2(G, F);
    const a2 = (E === 0 && H === 0) ? 0 : Math.atan2(H, E);
    return { thetaU: (a1 + a2) / 2, S: [Q + R, Q - R], thetaV: (a1 - a2) / 2 };
  };
  /** 2×2 polar decomposition A = rot2(theta)·P (P symmetric PSD) for det ≥ 0; for det < 0 returns {theta, P, reflect:true} with A = rot2(theta)·diag(1,−1)·P. */
  num.polar2 = (A) => {
    const { thetaU, S, thetaV } = num.svd2rot(A); const Rv = num.rot2(thetaV);
    const reflect = S[1] < 0;
    const P = num.matMul(num.matMul(Rv, num.diag([S[0], Math.abs(S[1])])), num.transpose(Rv));
    const off = (P[0][1] + P[1][0]) / 2; P[0][1] = off; P[1][0] = off;
    return reflect ? { theta: thetaU + thetaV, P, reflect: true } : { theta: thetaU - thetaV, P, reflect: false };
  };
  /** 2×2 matrix interpolation via polar decomposition (shortest-path angle lerp + entrywise lerp of P); entrywise lerp if either det ≤ 0. */
  num.matPolar = (A, B, t) => {
    if (!(num.det(A) > 0 && num.det(B) > 0)) return num.lerpM(A, B, t);
    const pa = num.polar2(A), pb = num.polar2(B);
    let dth = pb.theta - pa.theta; dth = Math.atan2(Math.sin(dth), Math.cos(dth));
    return num.matMul(num.rot2(pa.theta + dth * t), num.lerpM(pa.P, pb.P, t));
  };
  /** Thin QR by modified Gram–Schmidt: Q m×k orthonormal, R k×n upper triangular (k = min(m,n); dependent columns get R[j][j] = 0 and a completed Q column). */
  num.qr = (A) => {
    const [m, n] = num.shape(A); const k = Math.min(m, n); const a = num.cols(A); const Q = []; const R = num.zerosM(k, n);
    for (let j = 0; j < n; j++) {
      let v = a[j].slice();
      for (let i = 0; i < Q.length; i++) { const r = num.dot(Q[i], v); R[i][j] = r; v = num.sub(v, num.scale(Q[i], r)); }
      if (j < k) {
        const nv = num.norm(v);
        if (nv > 1e-12 * (num.norm(a[j]) || 1)) { R[j][j] = nv; Q.push(num.scale(v, 1 / nv)); }
        else { R[j][j] = 0; Q.push(completeBasis(Q, m, 1)[0]); }
      }
    }
    return { Q: num.fromCols(Q), R };
  };
  /** Gram–Schmidt with animation steps: {basis, steps[i] = {input, projections (one per previous basis vector), residual, normalized (null if dependent)}}. */
  num.gramSchmidt = (vs, eps = EPS) => {
    const basis = [], steps = [];
    for (const input of vs) {
      let r = input.slice(); const projections = [];
      for (const b of basis) { const p = num.scale(b, num.dot(r, b)); projections.push(p); r = num.sub(r, p); }
      const nr = num.norm(r); const normalized = nr > eps * (num.norm(input) || 1) ? num.scale(r, 1 / nr) : null;
      if (normalized) basis.push(normalized);
      steps.push({ input: input.slice(), projections, residual: r, normalized });
    }
    return { basis, steps };
  };
  /** Moore–Penrose pseudo-inverse via SVD (singular values ≤ rcond·σmax are dropped). */
  num.pinv = (A, rcond = 1e-10) => {
    const [m, n] = num.shape(A); const { U, S, V } = num.svd(A); const smax = S[0] || 0; const Uc = num.cols(U), Vc = num.cols(V); const P = num.zerosM(n, m);
    for (let k = 0; k < S.length; k++) { if (!(smax > 0 && S[k] > rcond * smax)) continue; const inv = 1 / S[k]; for (let i = 0; i < n; i++) { const vi = Vc[k][i] * inv; if (vi === 0) continue; for (let j = 0; j < m; j++) P[i][j] += vi * Uc[k][j]; } }
    return P;
  };
  /** Least squares weights w = pinv(X)·y (tall → least squares; wide → minimum-norm solution). */
  num.lstsq = (X, y) => num.matVec(num.pinv(X), y);
  /** Ridge regression: solve (XᵀX + λI) w = Xᵀy (falls back to lstsq if singular). */
  num.ridge = (X, y, lambda = 0) => {
    const Xt = num.transpose(X); const XtX = num.matMul(Xt, X); for (let i = 0; i < XtX.length; i++) XtX[i][i] += lambda;
    const w = num.solve(XtX, num.matVec(Xt, y)); return w || num.lstsq(X, y);
  };
  /** Normal equations: {XtX, Xty, w} with w = (XᵀX)⁻¹Xᵀy (w null if singular). */
  num.normalEq = (X, y) => { const Xt = num.transpose(X); const XtX = num.matMul(Xt, X); const Xty = num.matVec(Xt, y); return { XtX, Xty, w: num.solve(XtX, Xty) }; };
  /** Condition number σmax/σmin (Infinity if σmin < 1e-12·σmax or A = 0). */
  num.condNumber = (A) => { const S = num.svd(A).S; const smax = S[0] || 0, smin = S[S.length - 1] || 0; if (smax === 0 || smin < 1e-12 * smax) return Infinity; return smax / smin; };

  // ===================== statistics (X = array of sample rows) =====================
  /** Column means of X. */
  num.mean = (X) => { const n = X.length; if (!n) return []; const d = X[0].length; const mu = num.zeros(d); for (const r of X) for (let j = 0; j < d; j++) mu[j] += r[j]; return mu.map((x) => x / n); };
  /** Center rows: {Xc, mu}. */
  num.center = (X) => { const mu = num.mean(X); return { Xc: X.map((r) => num.sub(r, mu)), mu }; };
  /** Covariance matrix XcᵀXc/(n − ddof) (ddof default 1 = sample covariance). */
  num.cov = (X, opts) => {
    const ddof = opts && opts.ddof != null ? opts.ddof : 1; const n = X.length; const { Xc } = num.center(X); const d = Xc.length ? Xc[0].length : 0; const C = num.zerosM(d, d);
    for (const r of Xc) for (let i = 0; i < d; i++) { const ri = r[i]; if (ri === 0) continue; for (let j = i; j < d; j++) C[i][j] += ri * r[j]; }
    const denom = Math.max(n - ddof, 1);
    for (let i = 0; i < d; i++) for (let j = i; j < d; j++) { C[i][j] /= denom; C[j][i] = C[i][j]; }
    return C;
  };
  /** Correlation matrix (0 where a variable has zero variance). */
  num.corr = (X) => { const C = num.cov(X); const sd = C.map((r, i) => Math.sqrt(Math.max(r[i], 0))); return C.map((r, i) => r.map((x, j) => (sd[i] > 0 && sd[j] > 0 ? x / (sd[i] * sd[j]) : 0))); };
  /** Standardize columns to mean 0 / std 1 (ddof default 1; zero-variance columns use sigma = 1): {Xs, mu, sigma}. */
  num.standardize = (X, opts) => {
    const ddof = opts && opts.ddof != null ? opts.ddof : 1; const { Xc, mu } = num.center(X); const d = mu.length; const denom = Math.max(X.length - ddof, 1);
    const sigma = num.zeros(d); for (const r of Xc) for (let j = 0; j < d; j++) sigma[j] += r[j] * r[j];
    for (let j = 0; j < d; j++) { sigma[j] = Math.sqrt(sigma[j] / denom); if (sigma[j] === 0) sigma[j] = 1; }
    return { Xs: Xc.map((r) => r.map((x, j) => x / sigma[j])), mu, sigma };
  };
  /** PCA: {components (k unit sign-fixed vectors), values (variances desc), explained (fractions), mean, Z (n×k scores)}; eigSym of cov for d ≤ 64, else power iteration with deflation. */
  num.pca = (X, k) => {
    const n = X.length; const d = n ? X[0].length : 0; const { Xc, mu } = num.center(X); const denom = Math.max(n - 1, 1);
    const kk = k == null ? Math.min(d, n) : Math.max(0, Math.min(k, d));
    const total = Math.pow(num.frobenius(Xc), 2) / denom;
    let comps = [], vals = [];
    if (d <= 64) {
      const e = num.eigSym(num.cov(X)); comps = e.vectors.slice(0, kk); vals = e.values.slice(0, kk).map((v) => Math.max(v, 0));
    } else {
      let Y = copyM(Xc); const rnd = lcg(7919);
      const applyC = (v) => { const Yv = num.matVec(Y, v); const w = num.zeros(d); for (let i = 0; i < n; i++) { const yi = Yv[i]; if (yi === 0) continue; const row = Y[i]; for (let j = 0; j < d; j++) w[j] += row[j] * yi; } return num.scale(w, 1 / denom); };
      for (let c = 0; c < kk; c++) {
        let v = num.normalize(Array.from({ length: d }, () => rnd() - 0.5));
        for (let it = 0; it < 60; it++) { const w = applyC(v); const nw = num.norm(w); if (nw === 0) break; const vn = num.scale(w, 1 / nw); const delta = num.norm(num.sub(vn, v)); v = vn; if (delta < 1e-12) break; }
        const lam = num.dot(v, applyC(v)); comps.push(signFix(v)); vals.push(Math.max(lam, 0));
        const Yv = num.matVec(Y, v); Y = Y.map((row, i) => row.map((x, j) => x - Yv[i] * v[j]));
      }
    }
    const explained = vals.map((v) => (total > 0 ? v / total : 0));
    const Z = Xc.map((row) => comps.map((c) => num.dot(row, c)));
    return { components: comps, values: vals, explained, mean: mu, Z };
  };
  /** Mahalanobis distance sqrt((x−mu)ᵀ covInv (x−mu)). */
  num.mahalanobis = (x, mu, covInv) => { const dlt = num.sub(x, mu); return Math.sqrt(Math.max(num.dot(dlt, num.matVec(covInv, dlt)), 0)); };
  /** Whitening: {Xw, W, mean} with W = Λ^{-1/2}Qᵀ from eigSym(cov(X)) (eigenvalues ≤ 1e-12·λmax map to 0) and Xw = Xc Wᵀ. */
  num.whiten = (X) => {
    const { Xc, mu } = num.center(X); const { values, vectors } = num.eigSym(num.cov(X)); const d = values.length; const lmax = values[0] || 0; const W = num.zerosM(d, d);
    for (let i = 0; i < d; i++) { const inv = values[i] > 1e-12 * lmax && values[i] > 0 ? 1 / Math.sqrt(values[i]) : 0; for (let j = 0; j < d; j++) W[i][j] = vectors[i][j] * inv; }
    return { Xw: Xc.map((r) => num.matVec(W, r)), W, mean: mu };
  };

  // ===================== dynamics =====================
  /** Power iteration: {vector (unit iterate), value (Rayleigh quotient), history (normalized vector after each iteration)}. */
  num.powerIter = (A, iters = 50, v0) => {
    const n = A.length; const rnd = lcg(31337);
    let v = num.normalize(v0 ? v0.slice() : Array.from({ length: n }, (_, i) => 1 + 0.37 * rnd() + i / (2 * n))); const history = [];
    for (let it = 0; it < iters; it++) { const Av = num.matVec(A, v); const nv = num.norm(Av); if (nv === 0) break; v = num.scale(Av, 1 / nv); history.push(v); }
    const vv = num.dot(v, v); const value = vv > 0 ? num.dot(v, num.matVec(A, v)) / vv : 0;
    return { vector: v, value, history };
  };
  /** Unit rotation axis of a 3×3 rotation (from R − Rᵀ; null space of R − I for angle π; null for identity). */
  num.rotationAxis = (R) => {
    const ax = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]; const n = num.norm(ax);
    if (n > 1e-9) return num.scale(ax, 1 / n);
    const RmI = num.subM(R, num.identity(3)); if (num.frobenius(RmI) < 1e-9) return null;
    const ns = num.nullSpace(RmI, 1e-6); return ns.length ? signFix(ns[0]) : null;
  };
  /** Stationary distribution π of a stochastic matrix via the augmented linear system (column-stochastic P π = π by default; rows:true for row-stochastic π P = π); Σπ = 1. */
  num.stationary = (P, opts) => {
    const rows = !!(opts && opts.rows); const M = rows ? num.transpose(P) : copyM(P); const n = M.length;
    const A = M.map((r, i) => r.map((x, j) => x - (i === j ? 1 : 0))); const ones = new Array(n).fill(1);
    const Aaug = A.slice(0, n - 1).concat([ones]); const b = num.zeros(n); b[n - 1] = 1;
    let pi = num.solve(Aaug, b);
    if (!pi) pi = num.lstsq(A.concat([ones]), num.zeros(n).concat([1]));
    pi = pi.map((x) => (Math.abs(x) < 1e-14 ? 0 : x)); const s = num.sum(pi); return s !== 0 ? num.scale(pi, 1 / s) : pi;
  };

  // ===================== nonlinear helpers =====================
  /** Numerically stable softmax with temperature T. */
  num.softmax = (v, T = 1) => { const z = v.map((x) => x / T); const m = Math.max(...z); const e = z.map((x) => Math.exp(x - m)); const s = num.sum(e); return e.map((x) => x / s); };
  /** Logistic sigmoid 1/(1+e^−x). */
  num.sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
  /** Rectifier max(0, x). */
  num.relu = (x) => (x > 0 ? x : 0);
  /** Hyperbolic tangent. */
  num.tanh = (x) => Math.tanh(x);
  /** Elementwise sigmoid. */
  num.sigmoidV = (v) => v.map(num.sigmoid);
  /** Elementwise relu. */
  num.reluV = (v) => v.map(num.relu);
  /** Elementwise tanh. */
  num.tanhV = (v) => v.map(num.tanh);

  // ===================== stability / comparisons =====================
  /** Reorder and sign-flip `next` to best match `prev` greedily by max |dot| (unmatched extras appended in order). */
  num.stabilize = (prev, next) => {
    if (!prev || !prev.length || !next || !next.length) return next ? next.map((v) => v.slice()) : next;
    const pairs = [];
    for (let i = 0; i < prev.length; i++) for (let j = 0; j < next.length; j++) pairs.push({ i, j, d: num.dot(prev[i], next[j]) });
    pairs.sort((a, b) => Math.abs(b.d) - Math.abs(a.d) || a.i - b.i || a.j - b.j);
    const assigned = new Array(prev.length).fill(null); const used = new Array(next.length).fill(false);
    for (const p of pairs) { if (assigned[p.i] !== null || used[p.j]) continue; assigned[p.i] = p.d < 0 ? num.scale(next[p.j], -1) : next[p.j].slice(); used[p.j] = true; }
    const out = assigned.filter((v) => v !== null);
    for (let j = 0; j < next.length; j++) if (!used[j]) out.push(next[j].slice());
    return out;
  };
  /** |a − b| ≤ eps. */
  num.approxEq = (a, b, eps = EPS) => Math.abs(a - b) <= eps;
  /** Vectors equal within eps (same length). */
  num.approxEqV = (a, b, eps = EPS) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= eps);
  /** Matrices equal within eps (same shape). */
  num.approxEqM = (A, B, eps = EPS) => A.length === B.length && A.every((r, i) => num.approxEqV(r, B[i], eps));
  /** Clamp x into [lo, hi]. */
  num.clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  /** Scalar interpolation a + t(b − a). */
  num.lerp = (a, b, t) => a + (b - a) * t;

  // ===================== self test =====================
  /** Run built-in numerical checks; returns [{name, pass, detail}] (a thrown error becomes a failing entry). */
  num.selfTest = () => {
    const eps = 1e-8; const results = [];
    const check = (name, fn) => { let pass = false, detail = ''; try { const r = fn(); if (r === true) pass = true; else detail = String(r); } catch (e) { detail = 'threw: ' + (e && e.message ? e.message : e); } results.push({ name, pass, detail }); };
    const rnd = lcg(12345);
    const randV = (n) => Array.from({ length: n }, () => rnd() * 2 - 1);
    const randM = (r, c) => Array.from({ length: r }, () => randV(c));
    const randSym = (n) => { const A = randM(n, n); return num.addM(A, num.transpose(A)); };
    const errM = (A, B) => { let e = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) e = Math.max(e, Math.abs(A[i][j] - B[i][j])); return A.length === B.length ? e : Infinity; };
    const errV = (a, b) => (a.length !== b.length ? Infinity : a.reduce((e, x, i) => Math.max(e, Math.abs(x - b[i])), 0));
    const nearM = (A, B, tol = eps) => { const e = errM(A, B); return e <= tol ? true : 'max entry error ' + e; };
    const nearV = (a, b, tol = eps) => { const e = errV(a, b); return e <= tol ? true : 'max error ' + e + ' got [' + a.map((x) => +x.toFixed(6)).join(',') + ']'; };
    const near = (a, b, tol = eps) => (Math.abs(a - b) <= tol ? true : a + ' vs ' + b);
    const orthonormalCols = (M) => nearM(num.matMul(num.transpose(M), M), num.identity(M[0].length));
    const descending = (v) => { for (let i = 1; i < v.length; i++) if (v[i] > v[i - 1] + eps) return 'not descending: ' + v.join(','); return true; };
    const all = (...rs) => { for (const r of rs) if (r !== true) return r === false ? 'false' : r; return true; };
    const A3 = [[4, 1, 2], [1, 5, 3], [2, 3, 6]];
    const A4 = [[2, 1, 0, 3], [1, 3, 1, 0], [0, 1, 4, 1], [3, 0, 1, 5]];
    const A3copy = copyM(A3);

    check('inv 3x3: inv(A)·A = I', () => nearM(num.matMul(num.inv(A3), A3), num.identity(3)));
    check('inv 4x4: A·inv(A) = I', () => nearM(num.matMul(A4, num.inv(A4)), num.identity(4)));
    check('inv singular → null', () => (num.inv([[1, 2], [2, 4]]) === null ? true : 'not null'));
    check('solve reproduces b (3x3)', () => { const b = [1, -2, 3]; const x = num.solve(A3, b); return nearV(num.matVec(A3, x), b); });
    check('solve singular → null', () => (num.solve([[1, 2], [2, 4]], [1, 2]) === null ? true : 'not null'));
    check('det rot2 = 1', () => near(num.det(num.rot2(0.7)), 1));
    check('det reflect2 = −1', () => near(num.det(num.reflect2(0.4)), -1));
    check('det 4x4 (LU): triangular product and multiplicativity', () => { const T = [[2, 1, 3, 4], [0, -1, 5, 6], [0, 0, 3, 7], [0, 0, 0, 0.5]]; return all(near(num.det(T), -3), near(num.det(num.matMul(A4, A4)), num.det(A4) * num.det(A4), 1e-6), near(detLU(A3), num.det(A3))); });
    check('eig2 symmetric', () => { const A = [[2, 1], [1, 2]]; const e = num.eig2(A); return all(e.real === true || 'not real', nearV(e.values, [3, 1]), nearV(num.matVec(A, e.vectors[0]), num.scale(e.vectors[0], 3)), nearV(num.matVec(A, e.vectors[1]), num.scale(e.vectors[1], 1)), near(num.dot(e.vectors[0], e.vectors[1]), 0), e.vectors[0][0] > 0 || 'sign'); });
    check('eig2 shear (defective)', () => { const e = num.eig2([[1, 1], [0, 1]]); return all(e.real === true || 'not real', e.defective === true || 'not defective', e.vectors.length === 1 || 'vectors ' + e.vectors.length, nearV(e.vectors[0], [1, 0]), nearV(e.values, [1, 1])); });
    check('eig2 rotation (complex)', () => { const e = num.eig2(num.rot2(0.5)); return all(e.real === false || 'claimed real', near(e.re, Math.cos(0.5)), near(e.im, Math.sin(0.5))); });
    check('eig2 scaled identity (repeated, two vectors)', () => { const e = num.eig2([[2, 0], [0, 2]]); return all(e.vectors.length === 2 || 'vectors', !e.defective || 'defective', nearV(e.values, [2, 2])); });
    check('eigSym random symmetric 5x5: QΛQᵀ = A, Q orthonormal, descending', () => { const S = randSym(5); const e = num.eigSym(S); const Q = num.fromCols(e.vectors); return all(nearM(num.matMul(num.matMul(Q, num.diag(e.values)), num.transpose(Q)), S), orthonormalCols(Q), descending(e.values)); });
    check('eigSym repeated eigenvalue (2,2,1)', () => { const Q0 = num.rot3([1, 2, 3], 0.9); const S = num.matMul(num.matMul(Q0, num.diag([2, 2, 1])), num.transpose(Q0)); const e = num.eigSym(S); const Q = num.fromCols(e.vectors); return all(nearV(e.values, [2, 2, 1]), nearM(num.matMul(num.matMul(Q, num.diag(e.values)), num.transpose(Q)), S), orthonormalCols(Q)); });
    check('eigSym eigen-equation 6x6 (A v = λ v)', () => { const S = randSym(6); const e = num.eigSym(S); for (let i = 0; i < 6; i++) { const r = nearV(num.matVec(S, e.vectors[i]), num.scale(e.vectors[i], e.values[i])); if (r !== true) return r; } return true; });
    const svdCheck = (M) => { const { U, S, V } = num.svd(M); const [m, n] = num.shape(M); const k = Math.min(m, n); const recon = num.matMul(num.matMul(U, num.diag(S)), num.transpose(V)); return all(S.length === k || 'S length', U.length === m && U[0].length === k || 'U shape', V.length === n && V[0].length === k || 'V shape', nearM(recon, M), orthonormalCols(U), orthonormalCols(V), descending(S), S.every((s) => s >= 0) || 'negative σ'); };
    check('svd 3x2 reconstruction + orthonormal', () => svdCheck(randM(3, 2)));
    check('svd 2x3 reconstruction + orthonormal', () => svdCheck(randM(2, 3)));
    check('svd 4x4 reconstruction + orthonormal', () => svdCheck(randM(4, 4)));
    check('svd rank-1 (σ2 = 0)', () => { const M = [[1, 2], [2, 4]]; return all(svdCheck(M), num.svd(M).S[1] <= 1e-12 || 'σ2 ' + num.svd(M).S[1]); });
    check('svd zero 3x3', () => { const Z = num.zerosM(3, 3); return all(svdCheck(Z), num.svd(Z).S.every((s) => s === 0) || 'nonzero σ'); });
    check('svd 5x3 rank-2 completion', () => { const B = randM(5, 2); const M = num.hstack(B, [[0], [0], [0], [0], [0]].map((r, i) => [B[i][0] + B[i][1]])); return all(svdCheck(M), num.rank(M) === 2 || 'rank ' + num.rank(M)); });
    check('svd2rot det>0: A = R(θU)·Σ·R(θV)ᵀ, σ2 > 0', () => { const A = [[2, 1], [0.5, 1.5]]; const r = num.svd2rot(A); const recon = num.matMul(num.matMul(num.rot2(r.thetaU), num.diag(r.S)), num.transpose(num.rot2(r.thetaV))); return all(nearM(recon, A), r.S[0] >= Math.abs(r.S[1]) || 'order', r.S[1] > 0 || 'σ2 ≤ 0'); });
    check('svd2rot det<0: σ2 < 0, both factors proper rotations', () => { const A = [[1, 2], [2, 1]]; const r = num.svd2rot(A); const recon = num.matMul(num.matMul(num.rot2(r.thetaU), num.diag(r.S)), num.transpose(num.rot2(r.thetaV))); return all(nearM(recon, A), r.S[1] < 0 || 'σ2 ≥ 0', near(num.det(num.rot2(r.thetaU)), 1), near(num.det(num.rot2(r.thetaV)), 1)); });
    check('svd2rot agrees with svd singular values', () => { const A = randM(2, 2); const r = num.svd2rot(A); const S = num.svd(A).S; return nearV([r.S[0], Math.abs(r.S[1])], S); });
    check('polar2 det>0: A = R·P, P symmetric PSD', () => { const A = [[2, 1], [0.5, 1.5]]; const p = num.polar2(A); const e = num.eig2(p.P); return all(!p.reflect || 'reflect flag', nearM(num.matMul(num.rot2(p.theta), p.P), A), near(p.P[0][1], p.P[1][0]), e.values[1] >= -eps || 'P not PSD'); });
    check('polar2 det<0: A = R·diag(1,−1)·P', () => { const A = [[1, 2], [2, 1]]; const p = num.polar2(A); const e = num.eig2(p.P); return all(p.reflect === true || 'no reflect flag', nearM(num.matMul(num.matMul(num.rot2(p.theta), num.diag([1, -1])), p.P), A), e.values[1] >= -eps || 'P not PSD'); });
    check('matPolar(A,A,t) = A', () => { const A = [[2, 1], [0.5, 1.5]]; return all(nearM(num.matPolar(A, A, 0), A), nearM(num.matPolar(A, A, 0.3), A), nearM(num.matPolar(A, A, 1), A)); });
    check('matPolar endpoints and rotation midpoint', () => { const A = num.rot2(0.2), B = num.matMul(num.rot2(1.0), [[2, 0], [0, 1]]); return all(nearM(num.matPolar(A, B, 0), A), nearM(num.matPolar(A, B, 1), B), nearM(num.matPolar(num.rot2(0.2), num.rot2(1.0), 0.5), num.rot2(0.6))); });
    check('qr tall: A = QR, QᵀQ = I, R upper triangular', () => { const A = randM(5, 3); const { Q, R } = num.qr(A); let tri = true; for (let i = 0; i < R.length; i++) for (let j = 0; j < i; j++) if (Math.abs(R[i][j]) > eps) tri = false; return all(nearM(num.matMul(Q, R), A), orthonormalCols(Q), tri || 'R not triangular'); });
    check('qr wide 3x5', () => { const A = randM(3, 5); const { Q, R } = num.qr(A); return all(nearM(num.matMul(Q, R), A), orthonormalCols(Q), Q[0].length === 3 || 'Q cols'); });
    check('gramSchmidt orthonormal + steps consistent', () => { const vs = randM(4, 4); const { basis, steps } = num.gramSchmidt(vs); const Q = num.fromCols(basis); let ok = basis.length === 4 || 'basis size'; for (const s of steps) { let acc = s.input.slice(); for (const p of s.projections) acc = num.sub(acc, p); const r = nearV(acc, s.residual); if (r !== true) ok = r; } return all(ok, orthonormalCols(Q)); });
    check('gramSchmidt drops dependent vector', () => { const { basis, steps } = num.gramSchmidt([[1, 0, 0], [1, 1, 0], [2, 1, 0]]); return all(basis.length === 2 || 'basis ' + basis.length, steps[2].normalized === null || 'not null', steps.length === 3 || 'steps'); });
    check('pinv tall = least squares (X pinv X = X)', () => { const X = randM(8, 3); const y = randV(8); const P = num.pinv(X); return all(nearV(num.matVec(P, y), num.normalEq(X, y).w), nearM(num.matMul(num.matMul(X, P), X), X)); });
    check('pinv wide: min-norm result lies in row space', () => { const X = randM(3, 6); const y = randV(3); const w = num.matVec(num.pinv(X), y); const Prow = num.projectOnto(X); return all(nearV(num.matVec(X, w), y), nearV(num.matVec(Prow, w), w)); });
    check('lstsq matches normalEq (tall well-conditioned)', () => { const X = randM(10, 3); const y = randV(10); return nearV(num.lstsq(X, y), num.normalEq(X, y).w); });
    check('ridge λ=0 equals lstsq; λ>0 shrinks', () => { const X = randM(10, 3); const y = randV(10); const w0 = num.lstsq(X, y); return all(nearV(num.ridge(X, y, 0), w0), num.norm(num.ridge(X, y, 5)) < num.norm(w0) || 'no shrink'); });
    const Xd3 = Array.from({ length: 40 }, () => { const a = rnd() * 4 - 2, b = rnd() - 0.5, c = (rnd() - 0.5) * 0.2; return [a + 0.5, 0.6 * a + b, -0.3 * a + 0.5 * b + c + 1]; });
    check('pca d=3: values descending, explained sums to 1, components orthonormal', () => { const p = num.pca(Xd3); return all(descending(p.values), near(num.sum(p.explained), 1), orthonormalCols(num.fromCols(p.components)), p.components.length === 3 || 'k'); });
    check('pca d=3 agrees with eigSym(cov)', () => { const p = num.pca(Xd3); const e = num.eigSym(num.cov(Xd3)); return all(nearV(p.values, e.values), nearM(p.components, e.vectors), nearV(p.mean, num.mean(Xd3))); });
    check('pca Z = Xc·Cᵀ and reconstructs Xc', () => { const p = num.pca(Xd3); const { Xc } = num.center(Xd3); return all(nearM(p.Z, num.matMul(Xc, num.transpose(p.components))), nearM(num.matMul(p.Z, p.components), Xc)); });
    check('pca d>64 (power iteration) on rank-2 data', () => { const d = 70, n = 30; const gs = num.gramSchmidt([randV(d), randV(d)]).basis; const u = gs[0], w = gs[1]; const X = Array.from({ length: n }, () => num.add(num.scale(u, 3 * (rnd() - 0.5)), num.scale(w, rnd() - 0.5))); const p = num.pca(X, 2); const P = num.projectOnto(p.components); const { Xc } = num.center(X); return all(orthonormalCols(num.fromCols(p.components)), descending(p.values), nearV(num.matVec(P, u), u, 1e-6), nearV(num.matVec(P, w), w, 1e-6), nearM(num.matMul(p.Z, p.components), Xc, 1e-6), near(num.sum(p.explained), 1, 1e-6)); });
    check('powerIter converges to dominant eigenvector (SPD)', () => { const S = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]; const e = num.eigSym(S); const r = num.powerIter(S, 200); return all(near(Math.abs(num.dot(r.vector, e.vectors[0])), 1), near(r.value, e.values[0]), r.history.length === 200 || 'history'); });
    check('rotationAxis(rot3) returns axis up to sign', () => { const ax = num.normalize([1, 2, 3]); const r = num.rotationAxis(num.rot3([1, 2, 3], 0.8)); return near(Math.abs(num.dot(r, ax)), 1); });
    check('rotationAxis identity → null; angle π fallback', () => { const ax = num.normalize([1, -1, 2]); const r = num.rotationAxis(num.rot3(ax, Math.PI)); return all(num.rotationAxis(num.identity(3)) === null || 'identity not null', r !== null || 'π null', r && near(Math.abs(num.dot(r, ax)), 1, 1e-6)); });
    const Pc = [[0.9, 0.2, 0.1], [0.05, 0.7, 0.3], [0.05, 0.1, 0.6]];
    check('stationary (column-stochastic): Σπ = 1, Pπ = π', () => { const pi = num.stationary(Pc); return all(near(num.sum(pi), 1), nearV(num.matVec(Pc, pi), pi), pi.every((x) => x >= 0) || 'negative'); });
    check('stationary (row-stochastic rows:true)', () => { const Pr = num.transpose(Pc); const pi = num.stationary(Pr, { rows: true }); return all(near(num.sum(pi), 1), nearV(num.matVec(num.transpose(Pr), pi), pi), nearV(pi, num.stationary(Pc))); });
    check('softmax sums to 1 and is shift-invariant', () => { const v = [1, 2, 3, -1]; const s = num.softmax(v); return all(near(num.sum(s), 1), nearV(s, num.softmax(v.map((x) => x + 100))), s[2] > s[1] && s[1] > s[0] || 'order'); });
    check('chol: L Lᵀ = M, lower triangular; non-PD → null', () => { const L = num.chol(A3); return all(nearM(num.matMul(L, num.transpose(L)), A3), Math.abs(L[0][1]) + Math.abs(L[0][2]) + Math.abs(L[1][2]) === 0 || 'not lower', num.chol([[1, 2], [2, 1]]) === null || 'not null', num.chol([[1, 1], [1, 1]]) === null || 'PSD not null'); });
    check('rot3 orthonormal with det 1; rotX/Y/Z agree with rot3', () => { const R = num.rot3([0.3, -1, 2], 1.1); return all(orthonormalCols(R), near(num.det(R), 1), nearM(num.rotX(0.4), num.rot3([1, 0, 0], 0.4)), nearM(num.rotY(0.4), num.rot3([0, 1, 0], 0.4)), nearM(num.rotZ(0.4), num.rot3([0, 0, 1], 0.4))); });
    check('stabilize restores permuted/sign-flipped set', () => { const prev = num.eigSym(randSym(4)).vectors; const next = [num.scale(prev[2], -1), prev[0], num.scale(prev[3], -1), prev[1]]; return nearM(num.stabilize(prev, next), prev); });
    check('cross / angle / proj basics', () => all(nearV(num.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1]), near(num.cross([1, 0], [0, 1]), 1), near(num.angle([1, 0], [0, 1]), Math.PI / 2), near(num.angle([0.3, 0.4], [0.3, 0.4]), 0), near(num.angle([1, 1], [-1, -1]), Math.PI), nearV(num.proj([1, 1], [2, 0]), [1, 0])));
    check('rank / nullSpace / colSpace', () => { const A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]; const ns = num.nullSpace(A); const cs = num.colSpace(A); const Pc2 = num.projectOnto(cs); return all(num.rank(A) === 2 || 'rank ' + num.rank(A), ns.length === 1 || 'null dim ' + ns.length, nearV(num.matVec(A, ns[0]), [0, 0, 0]), near(num.norm(ns[0]), 1), cs.length === 2 || 'col dim', nearM(num.matMul(Pc2, A), A)); });
    check('nullSpace of wide 1x3 has 2 orthonormal vectors', () => { const A = [[1, 1, 1]]; const ns = num.nullSpace(A); return all(ns.length === 2 || 'dim ' + ns.length, orthonormalCols(num.fromCols(ns)), nearV(num.matVec(A, ns[0]), [0]), nearV(num.matVec(A, ns[1]), [0])); });
    check('condNumber', () => all(near(num.condNumber(num.identity(3)), 1), num.condNumber([[1, 2], [2, 4]]) === Infinity || 'not Inf', near(num.condNumber(num.diag([4, 1])), 4)));
    check('matPow: A³, A⁻¹, A⁰', () => all(nearM(num.matPow(A3, 3), num.matMul(num.matMul(A3, A3), A3)), nearM(num.matPow(A3, -1), num.inv(A3)), nearM(num.matPow(A3, 0), num.identity(3))));
    check('cov / corr / ddof', () => { const C = num.cov(Xd3); const C0 = num.cov(Xd3, { ddof: 0 }); const R = num.corr(Xd3); return all(nearM(C, num.transpose(C)), nearM(C0, num.scaleM(C, 39 / 40)), R.every((r, i) => Math.abs(r[i] - 1) < eps) || 'diag', R.every((r) => r.every((x) => Math.abs(x) <= 1 + eps)) || 'range'); });
    check('standardize: mean 0, var 1', () => { const { Xs } = num.standardize(Xd3); const C = num.cov(Xs); return all(nearV(num.mean(Xs), [0, 0, 0]), nearV(num.diagOf(C), [1, 1, 1])); });
    check('whiten → identity covariance', () => { const { Xw, W } = num.whiten(Xd3); return all(nearM(num.cov(Xw), num.identity(3), 1e-6), W.length === 3 || 'W'); });
    check('mahalanobis with identity = euclidean', () => near(num.mahalanobis([1, 2, 3], [0, 0, 1], num.identity(3)), Math.sqrt(1 + 4 + 4)));
    check('project2 idempotent, reflect2 involution', () => { const P = num.project2(0.6), R = num.reflect2(0.6); const v = [Math.cos(0.6), Math.sin(0.6)]; return all(nearM(num.matMul(P, P), P), nearM(num.matMul(R, R), num.identity(2)), nearV(num.matVec(P, v), v), nearV(num.matVec(R, v), v)); });
    check('projectOnto span', () => { const P = num.projectOnto([[1, 1, 0], [0, 0, 1]]); return all(nearM(num.matMul(P, P), P), nearM(P, num.transpose(P)), nearV(num.matVec(P, [1, 1, 0]), [1, 1, 0]), nearV(num.matVec(P, [1, -1, 0]), [0, 0, 0])); });
    check('norms / normalize / lerp / clamp', () => all(near(num.norm1([3, -4]), 7), near(num.normInf([3, -4]), 4), near(num.normP([3, -4], 2), 5), near(num.norm(num.normalize([3, 4])), 1), nearV(num.normalize([0, 0]), [0, 0]), near(num.lerp(1, 3, 0.25), 1.5), near(num.clamp(5, 0, 1), 1), nearV(num.lerpV([0, 0], [2, 4], 0.5), [1, 2])));
    check('transpose / hstack / vstack / submatrix / diag / trace', () => all(nearM(num.transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]), nearM(num.hstack([[1], [2]], [[3], [4]]), [[1, 3], [2, 4]]), nearM(num.vstack([[1, 2]], [[3, 4]]), [[1, 2], [3, 4]]), nearM(num.submatrix(A3, [0, 2], [1, 2]), [[1, 2], [3, 6]]), nearV(num.diagOf(num.diag([1, 2, 3])), [1, 2, 3]), near(num.trace(A3), 15), near(num.frobenius([[3, 0], [0, 4]]), 5)));
    check('inputs are not mutated', () => { num.inv(A3); num.eigSym(A3); num.svd(A3); num.det(A4); num.solve(A3, [1, 2, 3]); num.qr(A3); num.chol(A3); num.stationary(Pc); return nearM(A3, A3copy); });
    check('sigmoid / relu / tanh', () => all(near(num.sigmoid(0), 0.5), near(num.sigmoid(-800), 0), near(num.sigmoid(800), 1), nearV(num.reluV([-1, 2]), [0, 2]), near(num.tanh(0), 0), nearV(num.sigmoidV([0, 0]), [0.5, 0.5])));
    check('isSymmetric / approxEq helpers', () => all(num.isSymmetric(A3) || 'A3', !num.isSymmetric(A4.map((r, i) => r.map((x, j) => x + (i === 0 && j === 1 ? 1 : 0)))) || 'perturbed', num.approxEqM(A3, A3copy) || 'approxEqM', num.approxEqV([1, 2], [1, 2 + 1e-12]) || 'approxEqV'));
    return results;
  };

  LA.num = num;
})();
