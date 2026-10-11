/* p2-06 — Determinant: area, volume, orientation (+ cross product). */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg;
  const I2 = () => [[1, 0], [0, 1]];
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';
  const STRETCH = [[3, 0], [0, 2]], SHEAR = [[1, 1], [0, 1]], DISS = [[2, 1], [1, 3]], FLAT = [[1, 1.5], [0, 0]];
  const BOX3 = [[2, 0, 0], [0, 1, 0.5], [0, 0, 1.5]];
  const U0 = [2, 0, 1], V0 = [0, 1.5, 0.5], W0 = [0.5, 0.5, 2];
  const BLOB_C = [0.5, -0.9], BLOB_R = 0.82;
  // six inputs that all land on (2, 0) under FLAT = [[1, 1.5],[0, 0]]
  const SIX = [[2, 0], [0.5, 1], [-1, 2], [3.5, -1], [2.75, -0.5], [1.25, 0.5]];

  function makeCloud() { return LA.rng.gaussianCloud(66, 400, { mean: [0, 0], cov: [[1, 0], [0, 1]] }); }
  const cam2 = (cx, cy, hh) => Object.assign(LA.cam.defaults2d(), { center: [cx, cy], halfH: hh });
  const cam3 = (o) => Object.assign(LA.cam.defaults3d(), o);

  /* ---------- drawing helpers ---------- */
  // curved arrow from direction a to direction b around `center` (2D or 3D), radius r
  function curvedArrow(g, center, a, b, r, color) {
    const na = N.normalize(a), nb = N.normalize(b);
    const pts = [];
    for (let i = 0; i <= 16; i++) { const d = LA.geom.slerp(na, nb, i / 16); pts.push(center.map((x, k) => x + d[k] * r)); }
    g.polyline(pts.slice(0, 15), { color, width: 1.6, segmented: true });
    g.vector(pts[14], pts[16], { color, width: 1.6, head: 8 });
  }
  // unit-square tiles inside the axis-aligned rectangle [0,a]×[0,d] (first n of them), numbered
  function unitTiles(g, M, n) {
    const a = Math.round(M[0][0]), d = Math.round(M[1][1]);
    let k = 0;
    for (let j = 0; j < Math.abs(d); j++) for (let i = 0; i < Math.abs(a); i++) {
      if (k >= n) return;
      const sx = Math.sign(a), sy = Math.sign(d);
      const x0 = i * sx, y0 = j * sy, x1 = x0 + sx, y1 = y0 + sy;
      g.polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], { fill: (i + j) % 2 ? 'gold' : 'v', alpha: 0.28, stroke: 'gold', width: 1 });
      g.label(String(k + 1), [(x0 + x1) / 2, (y0 + y1) / 2], { color: 'text', size: 13, font: 'ui' });
      k++;
    }
  }
  // a round blob (centre c0, radius r) made of tiny grid squares, carried by M
  function blob(g, M, alpha) {
    const T = p => N.matVec(M, p);
    const ring = [];
    for (let i = 0; i < 64; i++) { const a = i / 64 * Math.PI * 2; ring.push(T([BLOB_C[0] + BLOB_R * Math.cos(a), BLOB_C[1] + BLOB_R * Math.sin(a)])); }
    g.polygon(ring, { fill: 'gold', alpha: 0.16 * alpha, stroke: 'gold', width: 1.4 });
    for (let k = -3; k <= 3; k++) {
      const c = k * 0.25; const h = Math.sqrt(Math.max(0, BLOB_R * BLOB_R - c * c));
      g.line(T([BLOB_C[0] + c, BLOB_C[1] - h]), T([BLOB_C[0] + c, BLOB_C[1] + h]), { color: 'gold', width: 0.8, alpha: 0.55 * alpha });
      g.line(T([BLOB_C[0] - h, BLOB_C[1] + c]), T([BLOB_C[0] + h, BLOB_C[1] + c]), { color: 'gold', width: 0.8, alpha: 0.55 * alpha });
    }
    g.label('area ' + LA.fmt(Math.PI * BLOB_R * BLOB_R * Math.abs(N.det(M))), T(BLOB_C), { color: 'gold', size: 13.5, font: 'ui' });
  }
  // the ad − bc dissection for M = [[a,b],[c,d]] with a,b,c,d ≥ 0; u ∈ [0,1] is the stage
  function dissection(g, M, u) {
    const a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
    const W = a + b, H = c + d;
    const rectA = seg(u, 0, 0.15);
    g.polyline([[0, 0], [W, 0], [W, H], [0, H]], { color: 'hud', width: 1.2, dashed: true, closed: true, alpha: rectA });
    if (rectA > 0.5) { g.dim([0, 0], [W, 0], { label: 'a + b = ' + LA.fmt(W), offset: 22 }); g.dim([W, 0], [W, H], { label: 'c + d = ' + LA.fmt(H), offset: 26 }); }
    const piece = (pts, fill, label, al) => { if (al <= 0) return; g.polygon(pts, { fill, alpha: 0.42 * al, stroke: fill, width: 1 }); const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length; g.label(label, [cx, cy], { color: fill, size: 13, alpha: al }); };
    const u1 = seg(u, 0.2, 0.45), u2 = seg(u, 0.5, 0.7), u3 = seg(u, 0.7, 0.88);
    piece([[a, 0], [W, 0], [W, c], [a, c]], 'b', 'bc', u1);
    piece([[0, d], [b, d], [b, H], [0, H]], 'b', 'bc', u1);
    piece([[0, 0], [a, 0], [a, c]], 'teal', 'ac/2', u2);
    piece([[b, d], [W, H], [b, H]], 'teal', 'ac/2', u2);
    piece([[a, c], [W, c], [W, H]], 'mag', 'bd/2', u3);
    piece([[0, 0], [b, d], [0, d]], 'mag', 'bd/2', u3);
  }

  LA.registerLesson({
    id: 'p2-06-determinant',
    part: 2,
    title: 'Determinant: area, volume, orientation',
    subtitle: 'How much a map scales area, whether it flips, and the cross product as an area arrow.',
    minutes: 12,
    prereqs: ['p2-04-composition', 'p2-05-3d-transformations'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'det of a 784×784 Jacobian in a normalizing flow; det of a 50×50 covariance in a Gaussian likelihood', transfers: '“volume factor, sign = orientation, zero = collapse” is exact for any n×n matrix; the cross product exists only in 3D' },
    cantShow: ['a 784-dimensional volume (it is computed; the picture is the 3D box)', 'a cross product in ℝ⁴ (there is none; the determinant does its job)'],
    terms: [
      { term: 'determinant', def: 'The factor by which a matrix multiplies every area (2D) or volume (3D); negative when orientation flips, zero when space is flattened.' },
      { term: 'orientation', def: 'Whether î-then-ĵ turns counter-clockwise (positive) or clockwise (negative) after the map.' },
      { term: 'singular', def: 'A matrix with det = 0: it flattens space onto a line or plane and loses information.' },
      { term: 'cross product', def: 'For two 3D vectors u and v, the vector u × v perpendicular to both whose length is the area of their parallelogram.' },
      { term: 'right-hand rule', def: 'The direction of u × v: curl your fingers from u toward v and your thumb points along u × v.' },
    ],
    atlas: ['gaussian', 'linear-regression'],
    thumbnail: 5,

    state() {
      return {
        view: '2d', cam: cam2(1.2, 0.8, 3.0),
        M: I2(), M3: I3(),
        showGhost: true, square: true, areaLabel: 'area', tiles: 0, blobA: 0, dims: 0, rotDim: false, dissect: 0,
        orientArrow: false, claim: null, collapse: 0,
        scene3: 'box', u: U0.slice(), v: V0.slice(), w: W0.slice(), crossT: 0, showW: 0, triple: 0, honesty: false,
        hud: [], ml: 0, showXtX: false, cloud: makeCloud(), recap: false, guess: [0, 0, 0], panel: 'area',
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.view === '3d') { render3d(g, s, ctx); return; }
      const M = s.M, det = N.det(M);
      const ci = N.col(M, 0), cj = N.col(M, 1);
      if (s.showGhost) g.ghostGrid();
      g.grid(M, { alpha: s.ml > 0 ? 0.5 : 1 });
      if (s.blobA > 0) blob(g, M, s.blobA);
      const showDiss = s.dissect > 0 || (s.pgDissect && ctx.mode === 'play');
      if (showDiss) dissection(g, M, s.dissect > 0 ? s.dissect : 1);
      if (s.square) g.parallelogram(ci, cj, { orient: true, alpha: 0.32 });
      if (s.tiles > 0) unitTiles(g, M, s.tiles);
      if (s.collapse > 0) {
        if (Math.abs(det) < 0.05) g.lineThrough([0, 0], ci, { color: 'teal', width: 2, alpha: 0.9 * s.collapse });
        SIX.forEach(p => { g.point(p, { r: 4, color: 'w', alpha: 0.6 }); g.line(p, N.matVec(M, p), { color: 'w', dashed: true, width: 1, alpha: 0.35 }); g.point(N.matVec(M, p), { r: 4.5, color: 'w' }); });
        if (Math.abs(det) < 0.05) g.label('singular: six inputs, one output', N.matVec(M, SIX[0]), { color: 'w', anchor: 'below', dy: 24, size: 13, font: 'ui', alpha: s.collapse });
      }
      if (s.ml > 0) {
        const T = p => N.matVec(M, p);
        const tp = s.cloud.map(T);
        g.points(tp, { r: 2.6, color: 'data', alpha: 0.75 });
        const R = 1.2; let inside = 0, inside0 = 0;
        for (const p of tp) if (Math.hypot(p[0], p[1]) <= R) inside++;
        for (const p of s.cloud) if (Math.hypot(p[0], p[1]) <= R) inside0++;
        g.circle([0, 0], R, { stroke: 'gold', width: 1.6, dashed: true });
        g.label('points inside: ' + (inside === inside0 ? inside : inside0 + ' → ' + inside), [0, -R], { color: 'gold', anchor: 'below', dy: 8, size: 13, font: 'ui' });
        g.hud('density × 1/|det A| = ' + LA.fmt(1 / Math.abs(det)), 14, 22, { size: 14, color: 'data' });
        g.hud('|det A| = ' + LA.fmt(Math.abs(det)), 14, 42, { size: 13 });
        if (s.showXtX) {
          let sxx = 0, sxy = 0, syy = 0; for (const p of s.cloud) { sxx += p[0] * p[0]; sxy += p[0] * p[1]; syy += p[1] * p[1]; }
          const n = s.cloud.length; sxx /= n; sxy /= n; syy /= n;
          const G = [[sxx, sxy, sxx], [sxy, syy, sxy], [sxx, sxy, sxx]];
          const bx = { x0: 1.9, x1: 3.7, y0: 1.9, y1: 3.7 };
          g.image(G, bx, { ramp: LA.color.heat, cells: true, range: [-0.2, Math.max(sxx, syy)] });
          g.label('XᵀX, features (x, y, x)', [(bx.x0 + bx.x1) / 2, bx.y1], { anchor: 'above', dy: -6, size: 12.5, font: 'ui', color: 'hud' });
          g.label('det(XᵀX) = 0: redundant feature', [(bx.x0 + bx.x1) / 2, bx.y0], { anchor: 'below', dy: 6, size: 12.5, font: 'ui', color: 'bad' });
        }
        g.hud('Part 5: Jacobian  ·  Part 4: Gaussians  ·  Part 3: det(XᵀX)', 14, g.size.h - 16, { size: 12.5, color: 'muted' });
      }
      g.vector(ci, { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(cj, { color: 'j', label: 'j', hat: true, width: 3 });
      if (s.dims > 0) {
        const perp = N.normalize([-ci[1], ci[0]]);
        g.dim([0, 0], ci, { label: 'base ' + LA.fmt(N.norm(ci)), offset: 18, color: 'i' });
        g.dim(ci, N.add(ci, perp), { label: 'height ' + LA.fmt(Math.abs(det) / (N.norm(ci) || 1)), offset: 18, color: 'j' });
      }
      if (s.orientArrow && Math.abs(det) > 0.03) curvedArrow(g, N.scale(N.add(ci, cj), 0.5), ci, cj, 0.3, det > 0 ? 'gold' : 'neg');
      if (s.square && s.areaLabel && !(s.tiles > 0)) {
        const c = N.scale(N.add(ci, cj), 0.5);
        const txt = s.areaLabel === 'area' ? 'area ' + LA.fmt(Math.abs(det)) : 'det A = ' + LA.fmt(det);
        g.label(txt, c, { color: det < -0.01 ? 'neg' : 'gold', size: 14, font: 'ui', dy: s.orientArrow ? 22 : (s.dims > 0 ? -22 : 0) });
      }
      if (s.claim) {
        const w = g.size.w;
        g.label(s.claim.text, [w / 2, 26], { screen: true, size: 15, color: 'hud', font: 'ui' });
        if (s.claim.cross) g.marker.cross([w / 2 + s.claim.text.length * 3.7 + 20, 26], { screen: true, size: 22 });
      }
      s.hud.forEach((h, i) => g.hud(h.text, 14, 22 + 20 * i, { color: h.color || 'hud', size: 14 }));
      if (ctx.mode === 'play' && showDiss && (M[0][0] < 0 || M[0][1] < 0 || M[1][0] < 0 || M[1][1] < 0)) g.hud('the dissection picture needs a, b, c, d ≥ 0', 14, g.size.h - 16, { color: 'muted', size: 12.5 });
      g.handle('i', ci, (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; }, { color: 'i' });
      g.handle('j', cj, (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
    },

    math(s, tex) {
      // local live number: tex.num spans after a tex.mat get misaligned by the panel's in-place update, so plain spans are used
      const num = (x, o) => '<span class="tx-n ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';
      const P = [], p = s.panel;
      const A = () => tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.M, { colors: 'cols' }));
      const det = N.det(s.M);
      if (p === 'area') { P.push(A()); P.push(tex.eq(tex.text('area of the square'), tex.op('='), num(Math.abs(det), { color: 'gold' }))); }
      else if (p === 'tiles') { P.push(A()); P.push(tex.eq(tex.text('det'), tex.sym('A'), tex.op('='), num(s.M[0][0], { color: 'i' }), tex.op('·'), num(s.M[1][1], { color: 'j' }), tex.op('='), num(det, { color: 'gold' }))); }
      else if (p === 'region') { P.push(tex.eq(tex.text('area(A·region)'), tex.op('='), tex.text('det'), tex.sym('A'), tex.op('·'), tex.text('area(region)'))); P.push(tex.eq(num(Math.abs(det), { color: 'gold' }), tex.op('·'), num(Math.PI * BLOB_R * BLOB_R), tex.op('='), num(Math.abs(det) * Math.PI * BLOB_R * BLOB_R, { color: 'gold' }))); }
      else if (p === 'shear') { P.push(A()); P.push(tex.eq(tex.text('det'), tex.sym('A'), tex.op('='), num(det, { color: 'gold' }), tex.text('  (base · height)'))); }
      else if (p === 'dissect') { const [a, b] = s.M[0], [c, d] = s.M[1]; P.push(tex.eq(tex.text('(a+b)(c+d) − 2bc − ac − bd = ad − bc'))); P.push(tex.eq(num(a, { color: 'i' }), tex.op('·'), num(d, { color: 'j' }), tex.op('−'), num(b, { color: 'j' }), tex.op('·'), num(c, { color: 'i' }), tex.op('='), num(a * d - b * c, { color: 'gold' }))); }
      else if (p === 'sign') { P.push(A()); P.push(tex.eq(tex.text('det'), tex.sym('A'), tex.op('='), num(det, { color: det < -0.01 ? 'neg' : 'gold' }), tex.text(det < -0.01 ? '  orientation flipped' : det < 0.03 ? '  flattened' : '  orientation kept', { color: det < -0.01 ? 'neg' : 'muted' }))); }
      else if (p === 'singular') { P.push(A()); P.push(tex.eq(tex.text('det = 0  ⇔  columns on one line  ⇔  singular', { color: 'bad' }))); }
      else if (p === 'vol') { P.push(tex.eq(tex.sym('M'), tex.op('='), tex.mat(s.M3, { colors: 'cols' }))); P.push(tex.eq(tex.text('volume'), tex.op('='), tex.text('det'), tex.sym('M'), tex.op('='), num(N.det(s.M3), { color: 'gold' }))); }
      else if (p === 'cross' || p === 'flip') {
        const c = N.cross(s.u, s.v), flip = p === 'flip';
        P.push(tex.eq(tex.sym('u', { color: 'v' }), tex.op('='), tex.vec(s.u, { color: 'v' }), tex.op(' '), tex.sym('v', { color: 'w' }), tex.op('='), tex.vec(s.v, { color: 'w' })));
        P.push(tex.eq(tex.sym('u', { color: 'v' }), tex.op('×'), tex.sym('v', { color: 'w' }), tex.op('='), tex.vec(['u₂v₃ − u₃v₂', 'u₃v₁ − u₁v₃', 'u₁v₂ − u₂v₁'], { color: 'gold' }), tex.op('='), tex.vec(c, { color: 'gold' })));
        if (flip) P.push(tex.eq(tex.sym('v', { color: 'w' }), tex.op('×'), tex.sym('u', { color: 'v' }), tex.op('='), tex.op('−('), tex.sym('u', { color: 'v' }), tex.op('×'), tex.sym('v', { color: 'w' }), tex.op(')'), tex.op('='), tex.vec(N.scale(c, -1), { color: 'gold' })));
        else P.push(tex.eq(tex.norm(tex.sym('u', { color: 'v' }) + tex.op('×') + tex.sym('v', { color: 'w' })), tex.op('='), num(N.norm(c), { color: 'gold' }), tex.op('='), tex.text('area')));
      }
      else if (p === 'triple') { const c = N.cross(s.u, s.v), vol = N.dot(c, s.w); P.push(tex.eq(tex.op('('), tex.sym('u', { color: 'v' }), tex.op('×'), tex.sym('v', { color: 'w' }), tex.op(')·'), tex.sym('w', { color: 'b' }), tex.op('='), tex.text('det'), tex.op('['), tex.sym('u', { color: 'v' }), tex.sym('v', { color: 'w' }), tex.sym('w', { color: 'b' }), tex.op(']'), tex.op('='), num(vol, { color: 'gold' }))); P.push(tex.eq(tex.text('area'), tex.op('·'), tex.text('height'), tex.op('='), num(N.norm(c)), tex.op('·'), num(vol / (N.norm(c) || 1), { color: 'mag' }), tex.op('='), num(vol, { color: 'gold' }))); }
      else if (p === 'mult') { P.push(tex.eq(tex.text('current map'), tex.op('='), tex.mat(s.M, { colors: 'cols' }))); P.push(tex.eq(tex.text('det(BA) = det B · det A = 3 · 2 ='), num(6, { color: 'gold' }))); }
      else if (p === 'ml') { P.push(tex.eq(tex.sym('ρ', { sub: 'out' }), tex.op('='), tex.sym('ρ', { sub: 'in' }), tex.op('/'), tex.abs(tex.text('det') + tex.sym('A')), tex.op('='), tex.sym('ρ', { sub: 'in' }), tex.op('/'), num(Math.abs(det), { color: 'gold' }), tex.text('(ρ = density)'))); P.push(tex.eq(tex.text('det(XᵀX) = 0  ⇒  redundant feature', { color: 'bad' }))); }
      else if (p === 'guess') { const c = N.cross(s.u, s.v); P.push(tex.eq(tex.sym('u', { color: 'v' }), tex.op('='), tex.vec(s.u, { color: 'v' }), tex.op(' '), tex.sym('v', { color: 'w' }), tex.op('='), tex.vec(s.v, { color: 'w' }))); P.push(tex.eq(tex.text('your guess'), tex.op('='), tex.vec(s.guess, { color: LA.judge.near(s.guess, c, LA.TOL.entry).ok ? 'gold' : 'b' }))); }
      else { P.push(A()); P.push(tex.eq(tex.text('det'), tex.sym('A'), tex.op('='), num(det, { color: det < -0.01 ? 'neg' : 'gold' }))); }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'The unit square under a stretch', duration: 4,
        narration: 'Every linear transformation stretches or squashes areas. Watch the unit square — the square with sides the green ' + ihat + ' and the red ' + jhat + ', area 1. Under A = diag(3, 2) it becomes a 3×2 rectangle, and its area counter climbs from 1 to 6.',
        setup(s) { s.view = '2d'; s.cam = cam2(1.2, 0.8, 3.0); s.M = I2(); s.square = true; s.areaLabel = 'area'; s.tiles = 0; s.blobA = 0; s.dims = 0; s.dissect = 0; s.orientArrow = false; s.claim = null; s.collapse = 0; s.hud = []; s.ml = 0; s.panel = 'area'; },
        at(s, t) { s.M = lerpM(I2(), STRETCH, seg(t, 0.1, 0.9)); } },

      { title: 'The determinant is that factor', duration: 3,
        narration: 'The number by which every area is multiplied is the <b>determinant</b>: det A = 6, a 3× stretch times a 2× stretch. Six copies of the unit square tile the rectangle exactly. One number summarizes how much the map inflates or deflates everything it touches.',
        setup(s) { s.M = LA.clone(STRETCH); s.areaLabel = null; s.panel = 'tiles'; },
        at(s, t) { s.tiles = Math.floor(seg(t, 0.1, 0.9, 'linear') * 6 + 1e-6); s.hud = [{ text: 'tiles: ' + s.tiles + ' of 6  ·  det A = 6', color: 'gold' }]; } },

      { title: 'Every region scales by the same factor', duration: 4,
        narration: 'Not only the unit square: any region is scaled by det A. The gold blob is made of tiny grid squares, and because the grid stays evenly spaced every tiny square scales identically. Its area 2.11 becomes 12.67 — six times more. The determinant is a property of the map, not of the shape.',
        setup(s) { s.cam = cam2(1.0, 0.0, 4.0); s.M = I2(); s.tiles = 0; s.hud = []; s.areaLabel = 'det'; s.panel = 'region'; },
        at(s, t) { s.blobA = seg(t, 0, 0.12); s.M = lerpM(I2(), STRETCH, seg(t, 0.15, 0.85)); } },

      { title: 'A shear changes no area', duration: 4,
        narration: 'The shear [[1, 1],[0, 1]] leans the square into a parallelogram but keeps its base and its height, so the area stays 1: det = 1. Leaning is free; only stretching costs area. A rotation also has det = 1 — it moves the square without changing it.',
        setup(s) { s.cam = cam2(0.6, 0.6, 2.6); s.blobA = 0; s.M = I2(); s.panel = 'shear'; },
        at(s, t) { const u1 = seg(t, 0.05, 0.45), u2 = seg(t, 0.6, 0.95); s.M = N.matMul(N.rot2(0.7 * u2), lerpM(I2(), SHEAR, u1)); s.dims = t > 0.05 ? 1 : 0; s.hud = [{ text: 'det A = ' + LA.fmt(N.det(s.M)) + (u2 > 0 ? '  (rotation: still 1)' : ''), color: 'gold' }]; } },

      { title: 'ad − bc is the picture', duration: 6,
        narration: 'For A = [[a, b],[c, d]] the formula is det A = ad − bc, and here is where it comes from. Enclose the parallelogram in a rectangle of sides a + b and c + d, then subtract two bc rectangles, two ac halves and two bd halves. What remains is ad − bc: with A = [[2, 1],[1, 3]], 6 − 1 = 5.',
        setup(s) { s.cam = cam2(1.3, 1.9, 3.0); s.dims = 0; s.hud = []; s.M = LA.clone(DISS); s.panel = 'dissect'; s.areaLabel = null; },
        at(s, t) { s.dissect = seg(t, 0, 0.9, 'linear'); s.areaLabel = t > 0.9 ? 'det' : null; s.hud = t > 0.9 ? [{ text: 'ad − bc = 2·3 − 1·1 = 5', color: 'gold' }] : []; } },

      { title: 'Negative determinant: the sheet turns over', duration: 5, misconception: true,
        narration: 'Belief: “area is never negative, so the determinant is never negative.” Swing the red ' + jhat + ' clockwise past the green ' + ihat + '. As it crosses, the area passes through zero and the fill turns violet: î-then-ĵ now turns clockwise, the <b>orientation</b> has flipped, and det = −1. The size is still an area; the sign records the flip.',
        setup(s) { s.cam = cam2(0.5, 0, 2.4); s.dissect = 0; s.hud = []; s.M = I2(); s.orientArrow = true; s.areaLabel = 'det'; s.panel = 'sign'; },
        at(s, t) { const phi = Math.PI / 2 - Math.PI * seg(t, 0.05, 0.8); s.M = [[1, Math.cos(phi)], [0, Math.sin(phi)]]; s.claim = { text: 'belief: “det ≥ 0 always”', cross: t > 0.88 }; } },

      { title: 'det = 0: flattened', duration: 4,
        narration: 'The special moment is det = 0: ' + jhat + ' lies on ' + ihat + '’s line and the square is squashed flat onto a line. The six purple inputs all land on one spot — information is lost, and no reverse map can exist; that is the next lesson. A matrix with det = 0 is called <b>singular</b>.',
        setup(s) { s.cam = cam2(0.8, 0.3, 3.0); s.claim = null; s.orientArrow = false; s.areaLabel = null; s._from = LA.clone(s.M); s.panel = 'singular'; },
        at(s, t) { s.M = lerpM(s._from, FLAT, seg(t, 0.05, 0.6)); s.collapse = seg(t, 0.1, 0.5); s.hud = t > 0.6 ? [{ text: 'det A = 0: singular', color: 'bad' }] : []; } },

      { title: 'In 3D the determinant is a volume', duration: 5,
        narration: 'In 3D the unit cube becomes a slanted box and det is its volume factor. Under M = [[2, 0, 0],[0, 1, 0.5],[0, 0, 1.5]] the volume is 3. Flatten the blue ' + khat + ' into the floor and the volume drops to 0: the box is a sheet. Positive, negative and zero mean the same three things as in 2D.',
        setup(s) { s.view = '3d'; s.cam = cam3({ target: [0.8, 0.5, 0.6], dist: 8.5, yaw: 0.5, pitch: 0.36 }); s.scene3 = 'box'; s.M3 = I3(); s.collapse = 0; s.hud = []; s.panel = 'vol'; },
        at(s, t) { const u1 = seg(t, 0.05, 0.45), u2 = seg(t, 0.6, 0.95); s.M3 = lerpM(I3(), BOX3, u1); s.M3[2][2] *= (1 - u2); LA.cam.tween(s, { yaw: 0.5 }, { yaw: 0.8 }, seg(t, 0, 1, 'linear')); } },

      { title: 'The cross product: an area as an arrow', duration: 5,
        narration: 'Two 3D vectors, the yellow <span class="c-v">u</span> = [2; 0; 1] and the purple <span class="c-w">v</span> = [0; 1.5; 0.5], span a parallelogram of area 3.5. The <b>cross product</b> u × v packages that area as an arrow: perpendicular to both, length 3.5. Here u × v = [−1.5; −1; 3]. It is how 3D code finds a normal to a surface.',
        setup(s) { s.scene3 = 'cross'; s.cam = cam3({ target: [0.3, 0.3, 0.4], dist: 12, yaw: 2.5, pitch: 0.25 }); s.u = U0.slice(); s.v = V0.slice(); s.crossT = 0; s.showW = 0; s.triple = 0; s.panel = 'cross'; },
        at(s, t) { s.crossT = seg(t, 0.4, 0.8); LA.cam.tween(s, { yaw: 2.5 }, { yaw: 2.95 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Order flips the cross product', duration: 3,
        narration: 'Swap the order and the gold arrow flips: v × u = −(u × v). The <b>right-hand rule</b> says which way: curl your fingers from the first vector to the second and your thumb points along the product. The parallelogram is the same; the arrow records which way around you went — an orientation, like the determinant’s sign.',
        setup(s) { s.crossT = 1; s.panel = 'flip'; },
        at(s, t) { s.crossT = 1 - 2 * seg(t, 0.2, 0.7); } },

      { title: 'Cross product meets determinant', duration: 4,
        narration: 'Add a third vector, the orange <span class="c-b">w</span>. The volume of the box spanned by u, v, w is (u × v)·w: the parallelogram’s area times w’s height above it. That is exactly det[u v w]. The cross product is the 3D determinant with one slot left open — which is why it exists only in 3D; elsewhere the determinant works alone.',
        setup(s) { s.scene3 = 'triple'; s.crossT = 1; s.w = W0.slice(); s.cam = cam3({ target: [0.5, 0.5, 1.4], dist: 11.5, yaw: 2.6, pitch: 0.3 }); s.panel = 'triple'; },
        at(s, t) { s.showW = seg(t, 0, 0.3); s.triple = seg(t, 0.35, 0.9, 'linear'); s.honesty = t > 0.5; } },

      { title: 'Determinants multiply', duration: 3.5,
        narration: 'Do one map, then another: areas scale by the first determinant, then by the second, so det(BA) = det B · det A. Stretch by 2, then by 3, and you have stretched by 6. Consequence: if any factor has det 0, the whole composition is singular.',
        setup(s) { s.view = '2d'; s.cam = cam2(1.0, 1.4, 3.0); s.M = I2(); s.square = true; s.areaLabel = 'area'; s.honesty = false; s.panel = 'mult'; },
        at(s, t) { const u1 = seg(t, 0.05, 0.4), u2 = seg(t, 0.55, 0.95); s.M = lerpM(lerpM(I2(), [[2, 0], [0, 1]], u1), [[2, 0], [0, 3]], u2); s.tiles = u2 >= 1 ? 6 : (u1 >= 1 ? 2 : 0); s.hud = [{ text: 'A = diag(2, 1): area 1 → ' + (u1 >= 1 ? '2' : '…'), color: 'gold' }]; if (t > 0.5) s.hud.push({ text: 'then B = diag(1, 3): area 2 → ' + (u2 >= 1 ? '6 = 3 · 2' : '…'), color: 'gold' }); } },

      { title: 'Where ML meets the determinant', duration: 5, mlTie: true,
        narration: 'A cloud of probability — many blue points — pushed through a map thins out where areas grow and bunches where they shrink; density changes by 1/|det|. Normalizing flows track exactly this. det(XᵀX) = 0 flags a redundant feature in regression, and the log-determinant of a covariance sits inside every Gaussian likelihood. Part 5’s Jacobian is a determinant per point.',
        setup(s) { s.cam = cam2(0.3, 0.3, 4.6); s.M = I2(); s.tiles = 0; s.square = false; s.hud = []; s.panel = 'ml'; },
        at(s, t) { s.ml = 1; s.M = lerpM(I2(), [[2, 0.5], [0, 0.75]], seg(t, 0.15, 0.6)); s.showXtX = t > 0.75; } },

      { title: 'Recap', duration: 1.5, recap: true,
        narration: 'The determinant is the area or volume factor of a linear map: negative means orientation flipped (the violet fill), zero means flattened (singular). ad − bc is a picture, not a formula to memorize. Determinants multiply under composition. The cross product is a 3D arrow whose length is a parallelogram’s area and whose direction is the normal.',
        cards: ['<b>det = area / volume factor.</b> The unit square becomes a parallelogram of area |det A|; any region scales the same way.', '<b>Sign and zero.</b> Negative = orientation flipped (violet fill); zero = flattened, singular, no way back. det(BA) = det B · det A.', '<b>u × v</b> is an arrow normal to u and v with length = the parallelogram’s area. 3D only; elsewhere the determinant works alone.'],
        setup(s) { s.recap = true; s.ml = 0; s.showXtX = false; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span> and watch det A: pale yellow for positive, violet for negative, a line at zero. In 3D, drag the column tips or the tips of <span class="c-v">u</span> and <span class="c-w">v</span>; drag empty space to orbit.</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(0.8, 0.8, 3.4); s.M = LA.clone(DISS); s.square = true; s.areaLabel = 'det'; s.orientArrow = true; s.panel = 'det'; s.M3 = LA.clone(BOX3); s.scene3 = 'box'; s.crossT = 1; s.pgDissect = false; },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'toggle', bind: 'pgDissect', label: 'Show the ad − bc dissection' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Stretch 3×2', apply(s) { to2d(s); s.M = LA.clone(STRETCH); } },
          { label: 'Shear', apply(s) { to2d(s); s.M = LA.clone(SHEAR); } },
          { label: 'Flip', apply(s) { to2d(s); s.M = [[0, 1], [1, 0]]; } },
          { label: 'Collapse', apply(s) { to2d(s); s.M = LA.clone(FLAT); } },
          { label: '3D box', apply(s) { s.view = '3d'; s.scene3 = 'box'; s.M3 = LA.clone(BOX3); s.cam = cam3({ target: [0.8, 0.5, 0.6], dist: 8.5, yaw: 0.6, pitch: 0.36 }); s.panel = 'vol'; } },
          { label: 'Cross product', apply(s) { s.view = '3d'; s.scene3 = 'cross'; s.crossT = 1; s.cam = cam3({ target: [0.3, 0.3, 0.6], dist: 12, yaw: 3.5, pitch: 0.28 }); s.panel = 'cross'; } },
        ] },
        { type: 'matrix', bind: 'u', label: 'u (3D cross product)', colors: 'v', step: 0.1 },
        { type: 'matrix', bind: 'v', label: 'v', colors: 'w', step: 0.1 },
        { type: 'button', label: 'Swap u and v', onClick(s) { const u = s.u; s.u = s.v; s.v = u; } },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + ihat + ' and ' + jhat + ' so that det A = −2.',
        setup(s) { s.view = '2d'; s.cam = cam2(0.6, 0.2, 3.0); s.M = I2(); s.square = true; s.areaLabel = 'det'; s.orientArrow = true; s.panel = 'sign'; },
        judge(s) { const d = N.det(s.M); const r = LA.judge.scalar(d, -2, LA.TOL.det); return { ok: r.ok, progress: Math.max(0, 1 - Math.abs(d + 2) / 3), hint: d > 0.05 ? 'The fill must turn violet first: swing ' + jhat + ' clockwise past ' + ihat + '.' : undefined }; },
        hint: 'Make the fill violet (swing ' + jhat + ' clockwise past ' + ihat + '), then enlarge the parallelogram until the label reads −2.',
        solution(s) { s.M = [[1, 0], [0, -2]]; } },
      { kind: 'manipulate',
        prompt: 'Make det A = 0 without moving ' + ihat + '.',
        setup(s) { s.view = '2d'; s.cam = cam2(0.6, 0.2, 3.0); s.M = I2(); s.square = true; s.areaLabel = 'det'; s.orientArrow = false; s.panel = 'det'; },
        judge(s) { const d = N.det(s.M), ci = N.col(s.M, 0), n1 = N.norm(N.col(s.M, 1)); const okI = LA.judge.near(ci, [1, 0], LA.TOL.pos).ok; const ok = Math.abs(d) <= LA.TOL.det && okI && n1 > 0.3; return { ok, progress: okI ? Math.max(0, 1 - Math.abs(d) / 1.2) : 0.1, hint: !okI ? 'Leave the green ' + ihat + ' at (1, 0).' : (n1 <= 0.3 ? 'Keep ' + jhat + ' longer than zero; just lay it along ' + ihat + '.' : undefined) }; },
        hint: 'Lay ' + jhat + ' along the green axis so both landing spots sit on one line.',
        solution(s) { s.M = [[1, 1.5], [0, 0]]; } },
      { kind: 'entry',
        prompt: 'Type u × v for <span class="c-v">u</span> = [2; 0; 1] and <span class="c-w">v</span> = [0; 1.5; 0.5]. Your guess is drawn orange and turns gold when it is right.',
        bind: 'guess', answer: [-1.5, -1, 3], tol: LA.TOL.entry, label: 'u × v', colors: 'b',
        setup(s) { s.view = '3d'; s.scene3 = 'guess'; s.u = U0.slice(); s.v = V0.slice(); s.guess = [0, 0, 0]; s.crossT = 0; s.cam = cam3({ target: [0.3, 0.3, 0.8], dist: 11, yaw: 3.5, pitch: 0.28 }); s.panel = 'guess'; } },
    ],
  });

  function to2d(s) { s.view = '2d'; s.cam = cam2(0.8, 0.8, 3.4); s.panel = 'det'; }

  function render3d(g, s, ctx) {
    g.cage({ range: 3, walls: true });
    g.axes({ length: 3.2, labels: true });
    const O = [0, 0, 0];
    if (s.scene3 === 'box') {
      const M = s.M3, d = N.det(M);
      g.lattice(M, { range: 2, alpha: 0.35 });
      g.box(M, { fill: d < -0.01 ? 'neg' : 'v', alpha: 0.22, stroke: d < -0.01 ? 'neg' : 'v' });
      const c = j => N.col(M, j);
      g.vector(O, c(0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(O, c(1), { color: 'j', label: 'j', hat: true, width: 3 });
      g.vector(O, c(2), { color: 'k', label: 'k', hat: true, width: 3 });
      g.hud('volume = det M = ' + LA.fmt(d), 14, 22, { size: 15, color: Math.abs(d) < 0.03 ? 'bad' : (d < 0 ? 'neg' : 'gold') });
      if (Math.abs(d) < 0.03) g.hud('flat: the box is a sheet', 14, 42, { size: 13, color: 'bad' });
      g.handle('i3', c(0), (p, st) => { st.M3[0][0] = p[0]; st.M3[1][0] = p[1]; st.M3[2][0] = p[2]; }, { color: 'i' });
      g.handle('j3', c(1), (p, st) => { st.M3[0][1] = p[0]; st.M3[1][1] = p[1]; st.M3[2][1] = p[2]; }, { color: 'j' });
      g.handle('k3', c(2), (p, st) => { st.M3[0][2] = p[0]; st.M3[1][2] = p[1]; st.M3[2][2] = p[2]; }, { color: 'k' });
      return;
    }
    const u = s.u, v = s.v, n = N.cross(u, v), area = N.norm(n);
    g.parallelogram(u, v, { fill: 'teal', alpha: 0.25, stroke: 'teal' });
    if (s.scene3 === 'triple' && s.showW > 0) {
      const w = N.scale(s.w, s.showW);
      if (s.triple > 0) g.box(N.fromCols([u, v, w]), { fill: 'b', alpha: 0.1 * seg(s.triple, 0, 0.3), stroke: 'b', strokeAlpha: 0.7 });
      g.vector(O, w, { color: 'b', label: 'w', width: 3 });
      if (s.triple > 0.35 && area > 1e-6) {
        const h = N.dot(w, n) / area, foot = N.scale(n, h / area);
        g.dropLine(w, foot, { color: 'residual', footColor: 'mag' });
        g.line(O, foot, { color: 'mag', width: 2, alpha: 0.8 });
        if (s.triple > 0.55) g.dim(O, foot, { label: 'height ' + LA.fmt(h), offset: -22, color: 'mag' });
        if (s.triple > 0.75) g.hud('volume = area · height = ' + LA.fmt(area) + ' · ' + LA.fmt(h) + ' = ' + LA.fmt(N.dot(n, w)), 14, 22, { size: 14.5, color: 'gold' });
      }
      if (s.honesty) g.hud('cross product: 3D only — the determinant works in every n', 14, g.size.h - 16, { size: 12.5, color: 'muted' });
    }
    g.vector(O, u, { color: 'v', label: 'u', width: 3 });
    g.vector(O, v, { color: 'w', label: 'v', width: 3 });
    if (s.scene3 === 'guess') {
      const ok = LA.judge.near(s.guess, n, LA.TOL.entry).ok;
      if (N.norm(s.guess) > 1e-6) g.vector(O, s.guess, { color: ok ? 'gold' : 'b', label: ok ? 'u × v ✓' : 'guess', width: 3 });
      if (ok) g.marker.check(s.guess, { dx: 22, dy: -10 });
    } else if (Math.abs(s.crossT) > 1e-3 && area > 1e-6) {
      const tip = N.scale(n, s.crossT), flipped = s.crossT < 0;
      g.vector(O, tip, { color: 'gold', label: flipped ? 'v × u' : 'u × v', width: 3.2 });
      if (Math.abs(s.crossT) > 0.95) {
        g.dim(O, tip, { label: 'length ' + LA.fmt(area) + ' = area', offset: 26, color: 'gold' });
        g.arc(O, u, tip, { r: 0.45, right: true, color: 'gold' });
        g.arc(O, v, tip, { r: 0.45, right: true, color: 'gold' });
        if (s.scene3 !== 'triple') curvedArrow(g, N.scale(N.add(u, v), 0.5), flipped ? v : u, flipped ? u : v, 0.5, 'gold');
      }
    }
    if (ctx.mode === 'play') {
      g.handle('u', u, (p, st) => { st.u = p.slice(); }, { color: 'v' });
      g.handle('v', v, (p, st) => { st.v = p.slice(); }, { color: 'w' });
    }
  }

  function renderRecap(g, s) {
    const card = (i, vopts, fn) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, vopts, fn);
    card(0, { view: '2d', cam: cam2(1.5, 1.9, 2.7) }, (g2, r) => {
      const M = DISS; g2.ghostGrid({ range: 6 }); g2.grid(M, { range: 6, alpha: 0.6 });
      g2.parallelogram(N.col(M, 0), N.col(M, 1), { orient: true, alpha: 0.32 });
      g2.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 2.6 }); g2.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 2.6 });
      g2.label('det A = 5 = area', [4.0, 2], { color: 'gold', size: 13, font: 'ui' });
      g2.hud('area factor', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(1, { view: '2d', cam: cam2(0.8, -0.3, 2.0) }, (g2, r) => {
      const M = [[1.2, 0.5], [0.3, -1.1]]; g2.ghostGrid({ range: 6 }); g2.grid(M, { range: 6, alpha: 0.6 });
      g2.parallelogram(N.col(M, 0), N.col(M, 1), { orient: true, alpha: 0.35 });
      g2.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 2.6 }); g2.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 2.6 });
      curvedArrow(g2, N.scale(N.add(N.col(M, 0), N.col(M, 1)), 0.5), N.col(M, 0), N.col(M, 1), 0.3, 'neg');
      g2.label('det A = ' + LA.fmt(N.det(M)) + ': flipped', [2.6, -0.6], { color: 'neg', size: 13, font: 'ui' });
      g2.hud('sign = orientation', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(2, { view: '3d', cam: cam3({ target: [0.3, 0.3, 1.3], dist: 8.5, yaw: 3.5, pitch: 0.25 }) }, (g2, r) => {
      g2.cage({ range: 3, walls: false }); g2.axes({ length: 3, labels: false });
      const n = N.cross(U0, V0);
      g2.parallelogram(U0, V0, { fill: 'teal', alpha: 0.25, stroke: 'teal' });
      g2.vector([0, 0, 0], U0, { color: 'v', label: 'u', width: 2.6 }); g2.vector([0, 0, 0], V0, { color: 'w', label: 'v', width: 2.6 });
      g2.vector([0, 0, 0], n, { color: 'gold', label: 'u × v', width: 3 });
      g2.hud('u × v: normal, length = area', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
  }
})();
