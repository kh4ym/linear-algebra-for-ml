/* gfx3d — 3D-specific primitives added to LA.Gfx: planes with mutual clipping, unit box / parallelepiped,
   lattice, cage grids with view-dependent crossfade, surfaces, flat-quad heatmaps on a plane, ellipsoids. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const G = LA.Gfx.prototype;
  const role = c => LA.color.role(c);
  const n3 = v => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];

  /** Split a convex polygon by the plane (p0, n) into [front (n·(x−p0) ≥ 0), back]. */
  function splitPolygon(poly, p0, n) {
    const front = [], back = [];
    const side = p => dot(n, [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]);
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length], sa = side(a), sb = side(b);
      if (sa >= 0) front.push(a); if (sa <= 0) back.push(a);
      if ((sa > 0 && sb < 0) || (sa < 0 && sb > 0)) { const t = sa / (sa - sb); const m = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; front.push(m); back.push(m); }
    }
    return [front.length >= 3 ? front : null, back.length >= 3 ? back : null];
  }
  /** Rectangle polygon of a plane: through `point`, spanned by u,v (or from normal), half-extent e. */
  function planeQuad(pl) {
    let u, v;
    if (pl.span) { u = n3(pl.span[0]); v = n3(cross(cross(u, pl.span[1]), u)); if (!Number.isFinite(v[0]) || Math.hypot(v[0], v[1], v[2]) < 1e-9) v = n3(cross(u, Math.abs(u[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0])); }
    else { const n = n3(pl.normal); u = n3(cross(n, Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0])); v = cross(n, u); }
    const e = pl.extent || 3, p = pl.point || [0, 0, 0];
    const eu = pl.extentU || e, ev = pl.extentV || e;
    return [add(p, add(mul(u, -eu), mul(v, -ev))), add(p, add(mul(u, eu), mul(v, -ev))), add(p, add(mul(u, eu), mul(v, ev))), add(p, add(mul(u, -eu), mul(v, ev)))];
  }
  function planeNormal(pl) { return pl.normal ? n3(pl.normal) : n3(cross(pl.span[0], pl.span[1])); }

  /** Translucent planes [{point, normal | span:[u,v], extent, color, alpha, stroke}] mutually clipped so painter's sort is right. */
  G.planes = function (list, opts) {
    opts = opts || {};
    const polys = list.map(pl => ({ pl, pieces: [planeQuad(pl)] }));
    for (let i = 0; i < polys.length; i++) for (let j = 0; j < polys.length; j++) {
      if (i === j) continue;
      const p0 = polys[j].pl.point || [0, 0, 0], n = planeNormal(polys[j].pl);
      const next = [];
      for (const piece of polys[i].pieces) { const [f, b] = splitPolygon(piece, p0, n); if (f) next.push(f); if (b) next.push(b); }
      polys[i].pieces = next;
    }
    polys.forEach(({ pl, pieces }) => {
      const fill = role(pl.color || 'teal');
      pieces.forEach(piece => this.polygon(piece, { fill, alpha: pl.alpha === undefined ? 0.28 : pl.alpha, subdiv: pl.subdiv === undefined ? 3 : pl.subdiv }));
      if (pl.stroke !== false) this.polyline(planeQuad(pl), { color: pl.stroke || fill, width: 1, closed: true, alpha: 0.7, segmented: true });
      if (pl.label) { const q = planeQuad(pl); this.label(pl.label, q[2], { color: fill, size: 13 }); }
    });
  };
  /** One plane (convenience). */
  G.plane = function (pl) { this.planes([pl]); };

  /** Unit cube (or its image under 3×3 M) as 6 faces + 12 edges. opts: fill, alpha, stroke, origin, size, faces(bool) */
  G.box = function (M, opts) {
    opts = opts || {};
    const o = opts.origin || [0, 0, 0], sz = opts.size || 1;
    const corners = [];
    for (let k = 0; k < 8; k++) { const p = [(k & 1) ? sz : 0, (k & 2) ? sz : 0, (k & 4) ? sz : 0]; const q = LA.geom.applyM(M, p, true, 0); corners.push(add(q, o)); }
    const faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
    const fill = role(opts.fill || 'v');
    if (opts.faces !== false) faces.forEach(f => this.polygon(f.map(i => corners[i]), { fill, alpha: opts.alpha === undefined ? 0.22 : opts.alpha, subdiv: opts.subdiv || 0 }));
    const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    const st = { color: role(opts.stroke || opts.fill || 'v'), width: opts.width || 1.4, alpha: opts.strokeAlpha === undefined ? 0.95 : opts.strokeAlpha };
    edges.forEach(e => this._seg(corners[e[0]], corners[e[1]], st));
    return corners;
  };

  /** Full 3D lattice of lines (|coords| ≤ range) transformed by M. opts: range, step, color, alpha, width */
  G.lattice = function (M, opts) {
    opts = opts || {};
    const R = opts.range || 2, step = opts.step || 1;
    const st = { color: role(opts.color || 'grid'), width: opts.width || 1, alpha: opts.alpha === undefined ? 0.75 : opts.alpha };
    const ax = { color: role(opts.axisColor || 'gridAxis'), width: (opts.width || 1) * 1.6, alpha: 1 };
    const T = p => LA.geom.applyM(M, p, true, 0);
    for (let a = -R; a <= R + 1e-9; a += step) for (let b = -R; b <= R + 1e-9; b += step) {
      const isAx = Math.abs(a) < 1e-9 && Math.abs(b) < 1e-9;
      for (let c = -R; c < R - 1e-9; c += step) {
        this._seg(T([c, a, b]), T([c + step, a, b]), isAx ? ax : st);
        this._seg(T([a, c, b]), T([a, c + step, b]), isAx ? ax : st);
        this._seg(T([a, b, c]), T([a, b, c + step]), isAx ? ax : st);
      }
    }
  };

  /** Floor + back-wall grids whose alpha crossfades with the view angle so walls never pop. opts: range, walls(bool), M */
  G.cage = function (opts) {
    opts = opts || {};
    const R = opts.range || 4, M = opts.M || null;
    this.grid(M, { range: R, color: opts.color || 'grid', axisColor: opts.axisColor || 'gridAxis', alpha: opts.alpha === undefined ? 0.9 : opts.alpha, back: opts.back });
    if (opts.walls === false || !this.is3d) return;
    const f = this.P.forward;
    const walls = [{ n: [1, 0, 0], T: p => [-R * Math.sign(f[0] || 1), p[0], p[1]] }, { n: [0, 1, 0], T: p => [p[0], -R * Math.sign(f[1] || 1), p[1]] }];
    walls.forEach(w => {
      const a = Math.min(1, Math.abs(dot(f, w.n)) * 1.6) * 0.45;
      if (a < 0.03) return;
      const st = { color: role(opts.color || 'grid'), width: 0.8, alpha: a, back: true };
      for (let i = -R; i <= R; i++) { this._seg(w.T([i, -R]), w.T([i, R]), st); this._seg(w.T([-R, i]), w.T([R, i]), st); }
    });
  };

  /** Surface z = fn(x,y) over bounds. opts: res (≤40), mode 'wire'|'quads'|'both', color, alpha, ramp(z→color), zRange, zScale */
  G.surface = function (fn, bounds, opts) {
    opts = opts || {};
    const res = Math.min(opts.res || 24, 48), mode = opts.mode || 'both';
    const zs = opts.zScale || 1;
    const pts = [];
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i <= res; i++) { pts.push([]); for (let j = 0; j <= res; j++) { const x = bounds.x0 + (bounds.x1 - bounds.x0) * j / res, y = bounds.y0 + (bounds.y1 - bounds.y0) * i / res; let z = fn([x, y]); if (!Number.isFinite(z)) z = 0; z *= zs; pts[i].push([x, y, z]); if (z < zmin) zmin = z; if (z > zmax) zmax = z; } }
    const zr = opts.zRange || [zmin, zmax];
    const ramp = opts.ramp || (t => LA.color.mix(LA.color.heatLo, role(opts.color || 'teal'), 0.35 + 0.65 * t));
    if (mode !== 'wire') for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
      const q = [pts[i][j], pts[i][j + 1], pts[i + 1][j + 1], pts[i + 1][j]];
      const zc = (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4, t = (zc - zr[0]) / ((zr[1] - zr[0]) || 1);
      this._poly(q, { fill: ramp(t), alpha: opts.alpha === undefined ? 0.55 : opts.alpha, bias: 0.002 });
    }
    if (mode !== 'quads') {
      const st = { color: role(opts.wireColor || opts.color || 'teal'), width: 0.8, alpha: opts.wireAlpha === undefined ? 0.55 : opts.wireAlpha };
      const sk = opts.wireStep || 1;
      for (let i = 0; i <= res; i += sk) for (let j = 0; j < res; j++) { this._seg(pts[i][j], pts[i][j + 1], st); this._seg(pts[j][i], pts[j + 1][i], st); }
    }
    return { zmin, zmax };
  };

  /** Parametric surface: fn(u, v) → [x, y, z] over u ∈ [u0,u1], v ∈ [v0,v1]. opts: res (≤40), mode 'wire'|'quads'|'both', color, alpha, ramp(t) (t = v fraction), wireStep, colorAt(u,v)→color */
  G.paramSurface = function (fn, range, opts) {
    opts = opts || {};
    const res = Math.min(opts.res || 20, 40), mode = opts.mode || 'both';
    const u0 = range.u0 === undefined ? -1 : range.u0, u1 = range.u1 === undefined ? 1 : range.u1, v0 = range.v0 === undefined ? -1 : range.v0, v1 = range.v1 === undefined ? 1 : range.v1;
    const pts = [];
    for (let i = 0; i <= res; i++) { pts.push([]); for (let j = 0; j <= res; j++) { const u = u0 + (u1 - u0) * j / res, v = v0 + (v1 - v0) * i / res; let p = fn(u, v); if (!p || !Number.isFinite(p[0])) p = [0, 0, 0]; pts[i].push([p[0], p[1], p[2] || 0]); } }
    const color = role(opts.color || 'teal');
    if (mode !== 'wire') for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
      const q = [pts[i][j], pts[i][j + 1], pts[i + 1][j + 1], pts[i + 1][j]];
      const uu = u0 + (u1 - u0) * (j + 0.5) / res, vv = v0 + (v1 - v0) * (i + 0.5) / res;
      const fill = opts.colorAt ? role(opts.colorAt(uu, vv)) : (opts.ramp ? opts.ramp((i + 0.5) / res) : color);
      this._poly(q, { fill, alpha: opts.alpha === undefined ? 0.4 : opts.alpha, bias: 0.002 });
    }
    if (mode !== 'quads') {
      const st = { color: role(opts.wireColor || opts.color || 'teal'), width: opts.wireWidth || 0.8, alpha: opts.wireAlpha === undefined ? 0.6 : opts.wireAlpha };
      const sk = opts.wireStep || 1;
      for (let i = 0; i <= res; i += sk) for (let j = 0; j < res; j++) { this._seg(pts[i][j], pts[i][j + 1], st); this._seg(pts[j][i], pts[j + 1][i], st); }
    }
    return pts;
  };

  /** Flat-quad heatmap in 3D on a plane (default the floor z = opts.z). fnOrGrid as heat2d. opts: bounds, res (≤32), range, ramp, alpha, z, plane {origin,u,v} */
  G.heat3d = function (fnOrGrid, opts) {
    opts = opts || {};
    const b = opts.bounds || { x0: -3, x1: 3, y0: -3, y1: 3 };
    const res = Math.min(opts.res || 24, 32);
    const grid = typeof fnOrGrid === 'function' ? LA.geom.sampleGrid(fnOrGrid, b, res) : fnOrGrid;
    const n = grid.length, m = grid[0].length;
    let lo = opts.range ? opts.range[0] : Infinity, hi = opts.range ? opts.range[1] : -Infinity;
    if (!opts.range) for (const row of grid) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-12) hi = lo + 1;
    const ramp = opts.ramp || LA.color.heat, z = opts.z || 0;
    const pl = opts.plane;
    const at = (x, y) => pl ? add(pl.origin, add(mul(pl.u, x), mul(pl.v, y))) : [x, y, z];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const x0 = b.x0 + (b.x1 - b.x0) * j / m, x1 = b.x0 + (b.x1 - b.x0) * (j + 1) / m;
      const y1 = b.y1 - (b.y1 - b.y0) * i / n, y0 = b.y1 - (b.y1 - b.y0) * (i + 1) / n;
      const t = (grid[i][j] - lo) / (hi - lo);
      this._poly([at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1)], { fill: ramp(t), alpha: opts.alpha === undefined ? 0.8 : opts.alpha, bias: 0.01, back: opts.back });
    }
  };

  /** Ellipsoid = image of the unit sphere under 3×3 M (rings + meridians). opts: color, rings, alpha, fill */
  G.ellipsoid = function (M, opts) {
    opts = opts || {};
    const color = role(opts.color || 'teal'), c = opts.center || [0, 0, 0];
    const seg = 48;
    const ring = (fnp) => { const pts = []; for (let i = 0; i < seg; i++) { const a = i / seg * Math.PI * 2; pts.push(add(LA.geom.applyM(M, fnp(a), true, 0), c)); } return pts; };
    const rings = [ring(a => [Math.cos(a), Math.sin(a), 0]), ring(a => [0, Math.cos(a), Math.sin(a)]), ring(a => [Math.cos(a), 0, Math.sin(a)])];
    rings.forEach((r, i) => this.polyline(r, { color: opts.colors ? role(opts.colors[i]) : color, width: opts.width || 1.5, closed: true, alpha: opts.alpha === undefined ? 0.9 : opts.alpha, segmented: true }));
    const nm = opts.meridians === undefined ? 6 : opts.meridians;
    for (let k = 1; k < nm; k++) { const phi = k / nm * Math.PI; const r = ring(a => [Math.cos(a) * Math.sin(phi), Math.sin(a) * Math.sin(phi), Math.cos(phi)]); this.polyline(r, { color, width: 0.7, closed: true, alpha: 0.35, segmented: true }); }
    if (opts.fill) { // coarse translucent shell
      const N = 10;
      for (let i = 0; i < N; i++) for (let j = 0; j < 2 * N; j++) {
        const p = (ph, th) => add(LA.geom.applyM(M, [Math.cos(th) * Math.sin(ph), Math.sin(th) * Math.sin(ph), Math.cos(ph)], true, 0), c);
        const ph0 = i / N * Math.PI, ph1 = (i + 1) / N * Math.PI, th0 = j / (2 * N) * 2 * Math.PI, th1 = (j + 1) / (2 * N) * 2 * Math.PI;
        this._poly([p(ph0, th0), p(ph0, th1), p(ph1, th1), p(ph1, th0)], { fill: color, alpha: opts.fillAlpha || 0.08 });
      }
    }
  };

  /** Sphere of radius r (shorthand for ellipsoid). */
  G.sphere = function (c, r, opts) { this.ellipsoid([[r, 0, 0], [0, r, 0], [0, 0, r]], Object.assign({ center: c }, opts || {})); };

  /** Projection of a point onto a plane/line drawn as a dashed residual + foot point. opts: color */
  G.dropLine = function (p, foot, opts) {
    opts = opts || {};
    this.line(p, foot, { color: opts.color || 'residual', dashed: true, width: opts.width || 1.2, alpha: 0.9 });
    this.point(foot, { r: opts.r || 3.5, color: opts.footColor || 'mag' });
  };

  LA.geom.splitPolygon = splitPolygon;
  LA.geom.planeQuad = planeQuad;
})();
