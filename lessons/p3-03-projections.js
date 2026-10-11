/* p3-03 — Projection onto a line and a plane: the perpendicular drop, P = aaᵀ/aᵀa, AᵀA x̂ = Aᵀb. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt;
  const I2 = () => [[1, 0], [0, 1]];
  const A_ = [3, 1], W_ = [1, 2];                                   // shared pair: a (yellow), w (purple)
  const A3_ = [[1, 0], [0, 1], [1, 1]], B3_ = [1, 2, 6];             // plane of a₁ = [1;0;1], a₂ = [0;1;1]; p = [2;3;5]
  const X_ = [[1, 1], [1, 2], [1, 3]], Y_ = [1, 2, 2];               // p3-05 data: ones, x, y
  const GHOSTS = [[-1, 2], [2, -1.2], [3, 2.6], [-1.2, -1.5], [0.4, 3.2]];
  const CAM2 = () => ({ center: [1.2, 0.9], halfH: 3.0 });
  const CAM3 = () => Object.assign(LA.cam.defaults3d(), { target: [1, 1.5, 2.4], dist: 16, yaw: -1.2, pitch: 0.45 });
  const CAM3ML = () => ({ target: [0.7, 1.2, 1.5], dist: 9, yaw: 1.2, pitch: 0.4 });
  const av = '<span class="c-v">a</span>', wv = '<span class="c-w">w</span>', pv = '<span class="c-mag">p</span>', bb = '<span class="c-b">b</span>';

  /* ---------- helpers ---------- */
  /* Engine workaround: LA.tex.reconcile maps tex.num spans onto matrix cells when only numbers change; emit live
     numbers as plain colored spans so any change rebuilds the panel cleanly. */
  function safeTex(tex) { return Object.assign({}, tex, { num: (x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; } }); }
  const projLine = (w, a) => { const aa = N.dot(a, a); return aa < 1e-9 ? [0, 0] : N.scale(a, N.dot(a, w) / aa); };
  const Pline = a => { const aa = N.dot(a, a) || 1; return [[a[0] * a[0] / aa, a[0] * a[1] / aa], [a[1] * a[0] / aa, a[1] * a[1] / aa]]; };
  /** projection of b onto the column space of the 3×2 matrix A: {p, xhat (null if dependent)} */
  function projPlane(A, b) {
    const At = N.transpose(A), G = N.matMul(At, A), r = N.matVec(At, b);
    const d = G[0][0] * G[1][1] - G[0][1] * G[1][0];
    if (Math.abs(d) > 1e-6 * Math.max(1, G[0][0] * G[1][1])) { const xh = N.solve(G, r); return { p: N.matVec(A, xh), xhat: xh }; }
    const c = N.norm(N.col(A, 0)) > 1e-6 ? N.col(A, 0) : N.col(A, 1);
    return { p: N.proj(b, c), xhat: null };
  }
  /** tip-to-tail chain of unit copies of the columns; prog ∈ [0,1]; returns the end point */
  function drawChain(g, cols, weights, colors, prog) {
    let pos = cols[0].map(() => 0); const total = weights.reduce((acc, w) => acc + Math.abs(w), 0); let left = prog * total;
    for (let j = 0; j < cols.length; j++) {
      const w = weights[j], sgn = w < 0 ? -1 : 1; let rem = Math.abs(w);
      while (rem > 1e-6 && left > 1e-6) { const part = Math.min(1, rem, left); const to = N.add(pos, N.scale(cols[j], sgn * part)); if (part > 0.02) g.vector(pos, to, { color: colors[j], width: 2.4, head: 9, alpha: 0.85 }); pos = to; rem -= part; left -= part; }
    }
    return pos;
  }
  function A3cols(s) { return [N.col(s.A3, 0), N.col(s.A3, 1)]; }

  /* ---------- 2D: projection onto a line ---------- */
  function draw2d(g, s, ctx) {
    g.ghostGrid();
    const a = s.a, w = s.w, p = projLine(w, a), P = Pline(a), gridU = ctx.mode === 'play' ? (s.showMatrixGrid ? 1 : 0) : s.gridU;
    if (s.entryGrid) { g.grid(s.Pe, { alpha: 0.85 }); }
    else if (gridU > 0) g.grid(lerpM(I2(), P, gridU), { alpha: 0.85 });
    g.axes();
    g.lineThrough([0, 0], a, { len: 12, color: 'v', width: 1.3, alpha: 0.55 });
    if (s.nullT > 0) {
      const n = N.normalize([-a[1], a[0]]);
      g.lineThrough([0, 0], n, { len: 12, color: 'mag', width: 2, alpha: 0.8 });
      g.label('null space', N.scale(n, 2.4), { color: 'mag', size: 13, dx: 44 });
      [-2.4, -1.4, 1.2, 2.2].forEach(k => { const v = N.scale(n, k * (1 - s.nullT)); if (Math.abs(k * (1 - s.nullT)) > 0.08) g.vector(v, { color: 'mag', width: 2, head: 9, alpha: 0.8 }); });
    }
    if (gridU > 0 || s.entryGrid) {
      const M = s.entryGrid ? s.Pe : lerpM(I2(), P, gridU);
      if (!s.entryGrid) GHOSTS.forEach(v => { const q = N.matVec(M, v); g.point(q, { r: 4, color: gridU > 0.98 ? 'mag' : 'w', alpha: 0.9 }); if (gridU < 0.98) g.point(v, { r: 3, color: 'w', alpha: 0.3 * (1 - gridU) }); });
      g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3, dx: -14, dy: -12 });
      if (s.entryGrid) { g.vector(a, { color: 'v', label: 'a', width: 2.4, alpha: 0.8 }); return; }
    }
    if (s.showA) g.vector(a, { color: 'v', label: 'a', width: 3 });
    if (s.showW) g.vector(w, { color: 'w', label: 'w', width: 3 });
    // misconception: vertical drop vs perpendicular drop
    if (s.vert > 0) {
      const q = Math.abs(a[0]) > 1e-6 ? [w[0], w[0] * a[1] / a[0]] : [0, w[1]];
      g.line(w, LA.lerpV(w, q, s.vert), { color: 'bad', dashed: true, width: 2 });
      if (s.vert > 0.98) {
        g.point(q, { r: 4.5, color: 'bad', label: 'q', labelColor: 'bad', dx: 12, dy: 12 });
        g.marker.cross(q, { dx: 26, dy: 22, size: 20 });
        g.label('|w − q| = ' + fmt(N.dist(w, q)), lerpV(w, q, 0.45), { color: 'bad', size: 13, anchor: 'right', dx: -10 });
      }
    }
    if (s.cand !== null && s.cand !== undefined) {
      const c = N.scale(a, s.cand);
      g.line(w, c, { color: 'hud', dashed: true, width: 1.4, alpha: 0.9 });
      g.point(c, { r: 6, color: 'hud', stroke: 'mag' });
      g.handle('cand', c, (q, st) => { const aa = N.dot(st.a, st.a) || 1; st.cand = LA.clamp(N.dot(q, st.a) / aa, -1.5, 1.8); }, { color: 'mag' });
      if (s.candReadout) {
        const v = N.dot(N.sub(w, c), a);
        g.bars([Math.abs(v)], { screen: true, x: 22, y: 48, w: 36, h: 70, max: 6, colors: ['residual'], title: '(w − point)·a = ' + fmt(v) });
      }
      if (s.minTick) {
        const n = N.normalize([-a[1], a[0]]);
        g.line(N.add(p, N.scale(n, -0.18)), N.add(p, N.scale(n, 0.18)), { color: 'gold', width: 3 });
        const pts = []; for (let k = 0; k <= 40; k++) { const kk = -0.1 + 1.2 * k / 40; pts.push([kk, N.dist(w, N.scale(a, kk))]); }
        const d = N.dist(w, c);
        g.lineChart([{ points: pts, color: 'hud' }, { points: [[s.cand, 1.4], [s.cand, d]], color: 'mag', width: 2.5 }], { x: g.size.w - 200, y: 48, w: 170, h: 90, xRange: [-0.1, 1.1], yRange: [1.4, 3.4], title: 'distance |w − point| = ' + fmt(d), xLabel: 'point = c·a', hline: N.dist(w, p), hlineColor: 'gold' });
      }
    }
    if (s.dropT > 0) {
      const foot = lerpV(w, p, s.dropT);
      g.dropLine(w, foot, { r: 4.5 });
      if (s.dropT > 0.98 && !s.showPvec) g.label('p', p, { color: 'mag', dx: 10, dy: 16 });
    }
    if (s.dimMode) {
      g.dim([0, 0], p, { label: s.dimMode === 'len' ? 'a·w/|a| = ' + fmt(N.dot(a, w) / N.norm(a)) : 'c = ' + fmt(N.dot(a, w) / N.dot(a, a)), color: s.dimMode === 'len' ? 'hud' : 'gold', offset: 20 });
    }
    if (s.showPvec) g.vector(p, { color: 'mag', label: s.pLabel || 'p', width: 3, dx: 8, dy: 18 });
    if (s.showRes) {
      g.vector(p, w, { color: 'residual', label: 'w − p', width: 2.4, labelAt: 0.55, labelOffset: -34 });
      g.arc(p, N.scale(a, 1), N.sub(w, p), { r: 0.32, color: 'hud', right: true });
    }
    if (s.resBar !== null && s.resBar !== undefined) {
      const foot = lerpV([0, 0], p, s.resBar), v = N.dot(N.sub(w, foot), a);
      if (s.resBar < 0.999) { g.line(w, foot, { color: 'residual', dashed: true, width: 1.3 }); g.point(foot, { r: 4, color: 'mag' }); }
      g.bars([Math.abs(v)], { screen: true, x: 22, y: 48, w: 36, h: 70, max: 6, colors: ['residual'], title: '(w − foot)·a = ' + fmt(v) });
    }
    if (s.misLabels) {
      g.label('|w − p| = ' + fmt(N.dist(w, p)), lerpV(w, p, 0.35), { color: 'gold', size: 13, anchor: 'left', dx: 14 });
    }
    if (s.okMark) { g.marker.check(s.okMark, { size: 26 }); g.label('P again: nothing moves', s.okMark, { color: 'ok', size: 13, dy: 24 }); }
    if (ctx.mode === 'play' && !s.entryGrid) {
      g.handle('a', a, (q, st) => { if (N.norm(q) > 0.2) st.a = [q[0], q[1]]; }, { color: 'v' });
      g.handle('w', w, (q, st) => { st.w = [q[0], q[1]]; }, { color: 'w' });
    }
  }

  /* ---------- 3D: projection onto a plane ---------- */
  function draw3d(g, s, ctx) {
    if (s.inset <= 0) { draw3dScene(g, s, ctx); return; }
    // the main 3D records flush at end(), after any viewport; so the scene goes in a full-canvas viewport first
    g.viewport({ x: 0, y: 0, w: 1, h: 1 }, { view: '3d', cam: s.cam, border: false }, g2 => draw3dScene(g2, s, ctx));
    drawInset(g, s, projPlane(s.A3, s.b3).p);
  }
  function draw3dScene(g, s, ctx) {
    g.cage({ range: 4, walls: false, alpha: 0.5 });
    g.axes({ length: s.mlLabels ? 2.8 : 4.4, names: s.mlLabels ? ['pt 1', 'pt 2', 'pt 3'] : ['x', 'y', 'z'] });
    const [c1, c2] = A3cols(s), b = s.b3, pr = projPlane(s.A3, b), p = pr.p;
    const L = s.mlLabels ? ['ones', 'x', 'y', 'ŷ'] : ['a₁', 'a₂', 'b', 'p'];
    const ext = s.mlLabels ? 3 : Math.max(4, 0.5 * N.norm(p) + 2.6);
    if (N.norm(N.cross(c1, c2)) > 1e-6) g.planes([{ point: N.scale(p, 0.5), span: [c1, c2], extent: ext, color: 'teal', alpha: 0.2 }]);
    g.vector([0, 0, 0], c1, { color: 'i', label: L[0], width: 3 });
    g.vector([0, 0, 0], c2, { color: 'j', label: L[1], width: 3 });
    g.vector([0, 0, 0], b, { color: 'b', label: L[2], width: 3 });
    if (s.chain3 > 0 && pr.xhat) drawChain(g, [c1, c2], pr.xhat, ['i', 'j'], s.chain3);
    if (s.drop3 > 0) {
      const foot = lerpV(b, p, s.drop3);
      g.dropLine(b, foot, { r: 4.5 });
    }
    if (s.slide3 !== null && s.slide3 !== undefined) {
      const f = lerpV([0, 0, 0], p, s.slide3), e = N.sub(b, f);
      if (s.slide3 < 0.999) g.dropLine(b, f, { r: 4.5 });
      g.bars([N.dot(e, c1), N.dot(e, c2)], { screen: true, x: 26, y: 52, w: 90, h: 70, max: 9, colors: ['i', 'j'], labels: ['e·a₁', 'e·a₂'], values: true, title: 'e = b − foot' });
    }
    if (s.showP3) g.vector([0, 0, 0], p, { color: 'mag', label: L[3], width: 3 });
    if (s.res3) {
      g.vector(p, b, { color: 'residual', label: s.mlLabels ? '' : 'b − p', width: 2.4, labelAt: 0.5 });
      if (!s.mlLabels) { g.arc(p, c1, N.sub(b, p), { r: 0.4, color: 'hud', right: true }); g.arc(p, c2, N.sub(b, p), { r: 0.4, color: 'hud', right: true }); }
    }
    if (ctx.mode !== 'learn') {
      g.handle('b', b, (q, st) => {
        let nb = [q[0], q[1], q[2]];
        const pp = projPlane(st.A3, nb).p; if (N.dist(nb, pp) < 0.2) nb = pp;          // snap onto the plane
        st.b3 = nb.map(x => LA.clamp(x, -6, 8));
      }, { color: 'b', plane: 'view' });
    }
  }
  function drawInset(g, s, yhat) {
    g.viewport({ x: 0.02, y: 0.02, w: 0.44, h: 0.34 }, { view: '2d', cam: { center: [2, 1.45], halfH: 1.45 }, bg: 'panel' }, (g2, r) => {
      g2.ghostGrid({ alpha: 0.7 });
      const xs = [1, 2, 3], ys = s.b3;
      const wh = projPlane(s.A3, s.b3).xhat || [0, 0];
      g2.line([0, wh[0]], [4, wh[0] + 4 * wh[1]], { color: 'gold', width: 2.4, alpha: s.inset });
      xs.forEach((x, i) => { g2.line([x, ys[i]], [x, yhat[i]], { color: 'residual', dashed: true, width: 1.5, alpha: s.inset }); g2.point([x, ys[i]], { r: 5, color: 'hud', alpha: s.inset }); });
      g2.label('the same fit, drawn as data (synthetic points)', [r.x + 10, r.y + 16], { screen: true, anchor: 'left', size: 11.5, color: 'muted', font: 'ui' });
    });
  }

  /* ---------- recap ---------- */
  function drawRecap(g) {
    const band = (i, vopt, text, fn) => g.viewport({ x: 0, y: i / 3, w: 1, h: 1 / 3 }, vopt, (g2, r) => { fn(g2); g2.label(text, [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', bold: true }); });
    band(0, { view: '2d', cam: { center: [1.4, 0.9], halfH: 1.45 } }, 'perpendicular drop', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      g2.lineThrough([0, 0], A_, { len: 9, color: 'v', width: 1.3, alpha: 0.6 });
      g2.vector(A_, { color: 'v', label: 'a', width: 2.6 }); g2.vector(W_, { color: 'w', label: 'w', width: 2.6 });
      g2.dropLine(W_, [1.5, 0.5], { r: 4.5 }); g2.vector([1.5, 0.5], { color: 'mag', label: 'p', width: 2.6, dx: 6, dy: 16 });
      g2.arc([1.5, 0.5], A_, [-0.5, 1.5], { r: 0.22, color: 'hud', right: true });
    });
    band(1, { view: '2d', cam: { center: [0.6, 0.3], halfH: 1.5 } }, 'P² = P, rank 1', g2 => {
      g2.ghostGrid({ alpha: 0.6 }); g2.grid(Pline(A_), { alpha: 0.85 });
      g2.lineThrough([0, 0], A_, { len: 9, color: 'v', width: 1.3, alpha: 0.6 });
      g2.vector([0.9, 0.3], { color: 'i', label: 'i', hat: true, width: 2.6 }); g2.vector([0.3, 0.1], { color: 'j', label: 'j', hat: true, width: 2.6, dx: -10, dy: -12 });
      g2.label('P² = P', [2.4, 1.2], { size: 16, color: 'gold', font: 'math' });
    });
    band(2, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [1, 1.6, 3.2], dist: 19, yaw: -1.2, pitch: 0.3 }) }, 'AᵀA x̂ = Aᵀb', g2 => {
      g2.planes([{ point: [1, 1.5, 2.5], span: [[1, 0, 1], [0, 1, 1]], extent: 4.2, color: 'teal', alpha: 0.22 }]);
      g2.vector([0, 0, 0], B3_, { color: 'b', label: 'b', width: 2.6 });
      g2.dropLine(B3_, [2, 3, 5], { r: 5 }); g2.vector([0, 0, 0], [2, 3, 5], { color: 'mag', label: 'p', width: 2.6 });
    });
  }

  function base2d(s) { s.scene = 'line'; s.view = '2d'; s.cam = CAM2(); s.a = A_.slice(); s.w = W_.slice(); s.showA = true; s.showW = true; s.dropT = 1; s.showPvec = true; s.pLabel = 'p'; s.showRes = true; s.dimMode = null; s.vert = 0; s.cand = null; s.candReadout = false; s.minTick = false; s.misLabels = false; s.gridU = 0; s.nullT = 0; s.okMark = null; s.resBar = null; s.entryGrid = false; }
  function base3d(s) { s.scene = 'plane'; s.view = '3d'; s.cam = CAM3(); s.A3 = LA.clone(A3_); s.b3 = B3_.slice(); s.drop3 = 1; s.showP3 = true; s.res3 = true; s.slide3 = null; s.chain3 = 0; s.mlLabels = false; s.inset = 0; }

  LA.registerLesson({
    id: 'p3-03-projections',
    part: 3,
    title: 'Projection onto a line and a plane',
    subtitle: 'The nearest point is a perpendicular drop: p = (a·w / a·a) a, and AᵀA x̂ = Aᵀb for a plane.',
    minutes: 10,
    prereqs: ['p1-06-dot-product', 'p2-08-rank-column-null', 'p2-10-transpose-outer-symmetric'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'least squares projects y ∈ ℝⁿ (n = training rows, often 10⁵) onto a d-dimensional column space', transfers: '“residual ⟂ every spanning vector” is the complete definition in any dimension; the formula is unchanged' },
    cantShow: ['a plane inside ℝⁿ for n > 3: the residual is still one perpendicular arrow, but the ambient space cannot be drawn', 'projection onto a 3-dimensional subspace of a 4-dimensional space (it would need a 4th axis)'],
    terms: [
      { term: 'projection', def: 'The shadow of a vector on a line or plane, dropped perpendicularly.' },
      { term: 'residual', def: 'What is left over: the vector minus its shadow, perpendicular to the line or plane.' },
      { term: 'projection matrix', def: 'The matrix P that computes the shadow of any vector; applying it twice changes nothing (P² = P).' },
      { term: 'subspace', def: 'A flat set through the origin (a line, a plane, …) that stays closed under adding and scaling.' },
      { term: 'orthogonal', def: 'Perpendicular: the dot product is zero.' },
    ],
    atlas: ['linear-regression', 'pca'],
    thumbnail: 6,

    state() {
      return {
        view: '2d', cam: CAM2(), scene: 'line', panel: 'ab',
        a: A_.slice(), w: W_.slice(), showA: true, showW: true, dropT: 0, showPvec: false, pLabel: 'p', showRes: false, dimMode: null,
        vert: 0, cand: null, candReadout: false, minTick: false, misLabels: false, gridU: 0, nullT: 0, okMark: null, resBar: null,
        entryGrid: false, Pe: I2(),
        A3: LA.clone(A3_), b3: B3_.slice(), drop3: 0, showP3: false, res3: false, slide3: null, chain3: 0, fill: 0, mlLabels: false, inset: 0,
        showMatrixGrid: false,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'recap') { drawRecap(g); return; }
      if (s.view === '3d') { draw3d(g, s, ctx); return; }
      draw2d(g, s, ctx);
    },

    math(s, tex0) {
      const tex = safeTex(tex0), P = s.panel;
      const a = s.a, w = s.w, p = projLine(w, a), aw = N.dot(a, w), aa = N.dot(a, a);
      if (P === 'recap') return tex.block(tex.eq(tex.sym('P'), tex.op(' = '), tex.sym('A'), tex.paren(tex.T(tex.sym('A')) + tex.sym('A')) + '<sup>−1</sup>', tex.T(tex.sym('A'))));
      if (P === 'ml') return tex.block(tex.eq(tex.T(tex.sym('X')), tex.sym('X'), tex.sym('ŵ'), tex.op(' = '), tex.T(tex.sym('X')), tex.sym('y', { color: 'b' })), tex.eq(tex.sym('ŷ', { color: 'mag' }), tex.op(' = '), tex.vec(projPlane(s.A3, s.b3).p, { color: 'mag' })));
      if (s.view === '3d') {
        const pr = projPlane(s.A3, s.b3), e = N.sub(s.b3, pr.p), [c1, c2] = A3cols(s);
        const parts = [tex.eq(tex.sym('A'), tex.op(' = '), tex.mat(s.A3, { colors: 'cols' })), tex.eq(tex.sym('b', { color: 'b' }), tex.op(' = '), tex.vec(s.b3, { color: 'b' }))];
        if (P === 'perp') {   // x̂ is introduced only in the next step, so state the two dot products directly
          const r0 = x => Math.abs(x) < 1e-9 ? 0 : x;
          parts.push(tex.eq(tex.paren(tex.sym('b', { color: 'b' }) + tex.op(' − ') + tex.sym('p', { color: 'mag' })), tex.op('·'), tex.sym('a', { sub: '1', color: 'i' }), tex.op(' = '), tex.num(r0(N.dot(e, c1)))));
          parts.push(tex.eq(tex.paren(tex.sym('b', { color: 'b' }) + tex.op(' − ') + tex.sym('p', { color: 'mag' })), tex.op('·'), tex.sym('a', { sub: '2', color: 'j' }), tex.op(' = '), tex.num(r0(N.dot(e, c2)))));
        }
        if (P === 'normal' || P === 'play3') {
          const At = N.transpose(s.A3), G = N.matMul(At, s.A3), r = N.matVec(At, s.b3), f = s.fill === undefined ? 4 : s.fill;
          const Gs = G.map((row, i) => row.map((x, j) => (i * 2 + j) < f ? x : '?'));
          parts.push(tex.eq(tex.T(tex.sym('A')), tex.sym('A'), tex.op(' = '), tex.mat(Gs)));
          parts.push(tex.eq(tex.T(tex.sym('A')), tex.sym('b', { color: 'b' }), tex.op(' = '), tex.vec(r, { color: 'b' })));
          if (pr.xhat) parts.push(tex.eq(tex.sym('x̂'), tex.op(' = '), tex.vec(pr.xhat, { color: 'gold' }), tex.op(', '), tex.sym('p', { color: 'mag' }), tex.op(' = '), tex.vec(pr.p, { color: 'mag' })));
          if (P === 'normal') parts.push(tex.eq(tex.sym('p', { color: 'mag' }), tex.op(' = '), tex.sym('A'), tex.paren(tex.T(tex.sym('A')) + tex.sym('A')) + '<sup>−1</sup>', tex.T(tex.sym('A')), tex.sym('b', { color: 'b' })));
          if (P === 'play3') parts.push(tex.eq(tex.text('e·a₁ = '), tex.num(N.dot(e, c1)), tex.text(', e·a₂ = '), tex.num(N.dot(e, c2))));
        }
        return tex.block.apply(null, parts);
      }
      if (P === 'ab') return tex.block(tex.eq(tex.sym('a', { color: 'v' }), tex.op(' = '), tex.vec(a, { color: 'v' })), tex.eq(tex.sym('w', { color: 'w' }), tex.op(' = '), tex.vec(w, { color: 'w' })));
      if (P === 'c') return tex.block(
        tex.eq(tex.sym('c'), tex.op(' = '), tex.frac(tex.sym('a') + tex.op('·') + tex.sym('w'), tex.sym('a') + tex.op('·') + tex.sym('a')), tex.op(' = '), tex.frac(tex.num(aw), tex.num(aa)), tex.op(' = '), tex.num(aw / aa, { color: 'gold' })),
        tex.eq(tex.sym('p', { color: 'mag' }), tex.op(' = '), tex.sym('c'), tex.op('·'), tex.sym('a', { color: 'v' }), tex.op(' = '), tex.vec(p, { color: 'mag' })));
      if (P === 'res') { const e = N.sub(w, p); return tex.block(tex.eq(tex.paren(tex.sym('w', { color: 'w' }) + tex.op(' − ') + tex.sym('p', { color: 'mag' })), tex.op('·'), tex.sym('a', { color: 'v' }), tex.op(' = '), tex.paren(tex.num(e[0])), tex.paren(tex.num(a[0], { color: 'v' })), tex.op(' + '), tex.paren(tex.num(e[1])), tex.paren(tex.num(a[1], { color: 'v' })), tex.op(' = '), tex.num(Math.abs(N.dot(e, a)) < 1e-9 ? 0 : N.dot(e, a), { color: 'gold' }))); }
      if (P === 'mis') return tex.block(tex.eq(tex.abs(tex.sym('w') + tex.op(' − ') + tex.sym('q')), tex.op(' = '), tex.num(1.67, { color: 'bad' })), tex.eq(tex.abs(tex.sym('w') + tex.op(' − ') + tex.sym('p')), tex.op(' = '), tex.num(N.dist(w, p), { color: 'gold' }), tex.text(' (the minimum)')));
      if (P === 'P') return tex.block(tex.eq(tex.sym('P'), tex.op(' = '), tex.frac(tex.sym('a') + tex.T(tex.sym('a')), tex.T(tex.sym('a')) + tex.sym('a')), tex.op(' = '), tex.mat(Pline(a), { colors: 'cols' })));
      if (P === 'P2') return tex.block(tex.eq(tex.sym('P'), tex.op('² = '), tex.sym('P')), tex.eq(tex.text('det '), tex.sym('P'), tex.op(' = '), tex.num(0)), tex.eq(tex.text('rank '), tex.sym('P'), tex.op(' = '), tex.num(1)));
      if (P === 'entry') return tex.block(tex.eq(tex.sym('a', { color: 'v' }), tex.op(' = '), tex.vec(a, { color: 'v' })), tex.eq(tex.sym('P'), tex.op(' = '), tex.mat(s.Pe, { colors: 'cols' })));
      // playground 2D
      return tex.block(tex.eq(tex.sym('p', { color: 'mag' }), tex.op(' = '), tex.frac(tex.num(aw), tex.num(aa)), tex.sym('a', { color: 'v' }), tex.op(' = '), tex.vec(p, { color: 'mag' })), tex.eq(tex.sym('P'), tex.op(' = '), tex.mat(Pline(a), { colors: 'cols' })), tex.eq(tex.paren(tex.sym('w') + tex.op(' − ') + tex.sym('p')), tex.op('·'), tex.sym('a'), tex.op(' = '), tex.num(Math.abs(N.dot(N.sub(w, p), a)) < 1e-9 ? 0 : N.dot(N.sub(w, p), a))));
    },

    steps: [
      { title: 'Recall: the shadow', duration: 3,
        narration: 'From the dot product lesson: drop a perpendicular from the purple ' + wv + ' = [1; 2] onto the line of the yellow ' + av + ' = [3; 1]. Where it lands is the <b>projection</b> of w onto a, its shadow. Today we write that shadow as a vector, and then as a matrix.',
        setup(s) { base2d(s); s.dropT = 0; s.showPvec = false; s.showRes = false; s.panel = 'ab'; },
        at(s, t) { s.dropT = seg(t, 0.15, 0.85); } },

      { title: 'How far along a? The dot product says', duration: 3.5,
        narration: 'The shadow lies on the line, so it is a multiple of ' + av + ': p = c·a. Its length along a is a·w = 5 divided by the length of a; dividing by |a| once more turns that length into multiples of a: c = a·w / a·a = 5/10 = ½.',
        setup(s) { s.dropT = 1; s.showPvec = false; s.dimMode = 'len'; s.panel = 'c'; s.pLabel = 'p = ½a'; },
        at(s, t) { s.dimMode = t < 0.5 ? 'len' : 'c'; s.showPvec = t > 0.6; } },

      { title: 'The residual is perpendicular', duration: 3.5,
        narration: 'What is left over, ' + wv + ' − ' + pv + ' = [−0.5; 1.5], is the <b>residual</b>. It is exactly perpendicular, <b>orthogonal</b>, to ' + av + ': (−0.5)(3) + (1.5)(1) = 0. Not a coincidence: “perpendicular residual” is what <i>defines</i> the shadow. Watch the gray bar reach zero only at p.',
        setup(s) { s.dimMode = null; s.showPvec = true; s.pLabel = 'p'; s.dropT = 0; s.showRes = false; s.resBar = 0; s.panel = 'res'; },
        at(s, t) { s.resBar = seg(t, 0.05, 0.75); s.showRes = t > 0.78; s.dropT = 0; } },

      { title: 'Misconception: the closest point is not straight below w', duration: 4, misconception: true,
        narration: '✕ “The closest point on the line is straight below w.” Dropping vertically lands at q = [1; 0.33], which is <i>farther</i> from ' + wv + ' than ' + pv + '. Perpendicular to the line, not to the axis, is what minimizes the distance. Watch the distance curve bottom out at p.',
        setup(s) { s.resBar = null; s.showRes = false; s.showPvec = false; s.dropT = 1; s.vert = 0; s.cand = null; s.minTick = false; s.misLabels = false; s.panel = 'mis'; },
        at(s, t) {
          s.vert = seg(t, 0, 0.25); s.misLabels = t > 0.25;
          if (t > 0.3) { s.cand = t < 0.8 ? LA.lerp(0.05, 0.95, seg(t, 0.3, 0.8)) : LA.lerp(0.95, 0.5, seg(t, 0.8, 1)); s.minTick = true; } else { s.cand = null; s.minTick = false; }
        } },

      { title: 'The projection matrix: a rank-1 map', duration: 4.5,
        narration: 'Projection is linear, so it has a matrix. From p = a(a·w)/(a·a) = (aaᵀ/aᵀa)w we read the <b>projection matrix</b> P = aaᵀ/aᵀa = [[0.9, 0.3], [0.3, 0.1]]. It is an outer product, so its rank is 1: the whole plane is flattened onto the yellow line.',
        setup(s) { s.vert = 0; s.cand = null; s.minTick = false; s.misLabels = false; s.dropT = 0; s.showPvec = false; s.showW = false; s.showA = true; s.gridU = 0; s.panel = 'P'; },
        at(s, t) { s.gridU = seg(t, 0.1, 0.9); } },

      { title: 'Projecting twice changes nothing: P² = P', duration: 3.5,
        narration: 'A shadow of a shadow is the same shadow. Apply P again and nothing moves: P² = P. Also det P = 0: a projection cannot be undone, because every vector along the magenta residual direction, the null space, was crushed to zero.',
        setup(s) { s.gridU = 1; s.nullT = 0; s.okMark = null; s.panel = 'P2'; },
        at(s, t) { s.nullT = seg(t, 0.3, 0.9); s.okMark = t > 0.2 ? [2.9, 1.9] : null; } },

      { title: 'Onto a plane in 3D', duration: 5,
        narration: 'Now the target is a plane, a <b>subspace</b>: the span of the green <span class="c-i">a₁</span> = [1; 0; 1] and the red <span class="c-j">a₂</span> = [0; 1; 1]. The orange ' + bb + ' = [1; 2; 6] sticks out of it. Its shadow ' + pv + ' is the point of the teal plane nearest to b: the perpendicular drop again.',
        setup(s) { s.nullT = 0; s.okMark = null; s.gridU = 0; base3d(s); s.drop3 = 0; s.showP3 = false; s.res3 = false; s.panel = 'A'; },
        at(s, t) { s.drop3 = seg(t, 0.3, 0.85); s.showP3 = t > 0.88; LA.cam.tween(s, { yaw: -1.2 }, { yaw: -0.95 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Perpendicular to the plane means perpendicular to each column', duration: 4,
        narration: 'The residual ' + bb + ' − ' + pv + ' = [−1; −1; 1] must be perpendicular to <i>every</i> direction in the plane, and it is enough to check the two spanning vectors: (b − p)·a₁ = 0 and (b − p)·a₂ = 0. Two dot products, two equations, two unknown weights.',
        setup(s) { s.drop3 = 0; s.showP3 = true; s.res3 = false; s.slide3 = 0; s.panel = 'perp'; },
        at(s, t) { s.slide3 = seg(t, 0.05, 0.75); s.res3 = t > 0.78; LA.cam.tween(s, { yaw: -0.95 }, { yaw: -0.8 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Written out: the normal equations', duration: 4.5,
        narration: 'p = Ax̂ for some weights x̂ = [x₁; x₂]. “Aᵀ(b − Ax̂) = 0” rearranges to <b>AᵀA x̂ = Aᵀb</b>, the <b>normal equations</b>. Here AᵀA = [[2, 1], [1, 2]] and Aᵀb = [7; 8], so x̂ = [2; 3] and p = 2a₁ + 3a₂ = [2; 3; 5]: two green steps, three red.',
        setup(s) { s.slide3 = null; s.res3 = true; s.chain3 = 0; s.fill = 0; s.panel = 'normal'; },
        at(s, t) { s.fill = Math.floor(seg(t, 0.05, 0.4, 'linear') * 4.999); s.chain3 = seg(t, 0.4, 0.95); } },

      { title: 'Why ML cares: fitting is projecting', duration: 4, mlTie: true,
        narration: 'Replace the plane by the column space of a data matrix X (a column of ones and the column x) and ' + bb + ' by the targets y: the shadow ŷ is the vector of fitted values and x̂, written ŵ there, holds the regression weights. The least-squares capstone draws exactly this picture. Projection <i>is</i> least squares.',
        setup(s) { s.chain3 = 0; s.fill = 4; s.res3 = true; s.mlLabels = false; s.inset = 0; s._A = LA.clone(s.A3); s._b = s.b3.slice(); s.panel = 'ml'; },
        at(s, t) {
          const u = seg(t, 0.05, 0.5); s.A3 = lerpM(s._A, X_, u); s.b3 = lerpV(s._b, Y_, u); s.mlLabels = t > 0.5; s.inset = seg(t, 0.55, 0.8);
          LA.cam.tween(s, { target: [1, 1.5, 2.4], dist: 16, yaw: -0.8, pitch: 0.45 }, CAM3ML(), seg(t, 0.05, 0.5));
        } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Shadow on a line: p = (a·w / a·a) a, with matrix P = aaᵀ/aᵀa, P² = P and rank 1. Shadow on a plane: the residual is perpendicular to every column, giving AᵀA x̂ = Aᵀb. Nearest point always means perpendicular drop.',
        cards: ['<b>Perpendicular drop.</b> The shadow p is where the residual w − p meets the line at a right angle.', '<b>P² = P, rank 1.</b> P = aaᵀ/aᵀa flattens the plane onto the line; projecting again changes nothing.', '<b>AᵀA x̂ = Aᵀb.</b> Residual ⟂ each column gives the normal equations; p = A x̂.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.cam = CAM2(); s.inset = 0; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>2D: drag the yellow <span class="c-v">a</span> and the purple <span class="c-w">w</span>. 3D: drag the orange <span class="c-b">b</span> (it snaps onto the plane when close) or edit the columns of A.</p>',
      setup(s) { base2d(s); s.panel = 'play'; },
      controls: [
        { type: 'toggle', bind: 'showMatrixGrid', label: 'Show P acting on the grid' },
        { type: 'matrix', bind: 'A3', label: 'A (3D plane)', colors: 'cols', step: 0.1 },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Line a = [3; 1]', apply(s) { base2d(s); s.panel = 'play'; } },
          { label: 'Unit a', apply(s) { base2d(s); s.a = N.normalize(A_); s.panel = 'play'; } },
          { label: 'Plane [1;0;1], [0;1;1]', apply(s) { base3d(s); s.panel = 'play3'; s.fill = 4; } },
          { label: 'b already on the plane', apply(s) { base3d(s); s.b3 = [2, 3, 5]; s.panel = 'play3'; s.fill = 4; } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D line', apply(s) { const a = s.a, w = s.w; base2d(s); s.a = a; s.w = w; s.panel = 'play'; } },
          { label: '3D plane', apply(s) { const A = s.A3, b = s.b3; base3d(s); s.A3 = A; s.b3 = b; s.panel = 'play3'; s.fill = 4; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the magenta-ringed point along the yellow line to where the shadow of ' + wv + ' really lands.',
        setup(s) { base2d(s); s.dropT = 0; s.showPvec = false; s.showRes = false; s.cand = 0.12; s.candReadout = true; s.panel = 'ab'; },
        judge(s) { return LA.judge.near(N.scale(s.a, s.cand), projLine(s.w, s.a), LA.TOL.pos); },
        hint: 'Watch the gray bar (w − point)·a: it should read 0.',
        solution(s) { s.cand = 0.5; } },
      { kind: 'entry',
        prompt: 'Type the projection matrix onto the line through [1; 1]. The grid flattens live under your matrix.',
        bind: 'Pe', answer: [[0.5, 0.5], [0.5, 0.5]], tol: LA.TOL.entry, label: 'P', colors: 'cols',
        setup(s) { base2d(s); s.a = [1, 1]; s.entryGrid = true; s.Pe = I2(); s.showW = false; s.dropT = 0; s.showPvec = false; s.showRes = false; s.panel = 'entry'; } },
      { kind: 'manipulate',
        prompt: 'Drag the orange ' + bb + ' into the teal plane so the residual vanishes.',
        setup(s) { base3d(s); s.panel = 'play3'; s.fill = 4; },
        judge(s) { const p = projPlane(s.A3, s.b3).p, d = N.dist(s.b3, p); return { ok: d < LA.TOL.pos && N.norm(s.b3) > 0.5, progress: Math.max(0, 1 - d / 1.7) }; },
        hint: 'On this plane, the third coordinate of b must equal the sum of the first two.',
        solution(s) { s.b3 = [2, 3, 5]; } },
    ],
  });

})();
