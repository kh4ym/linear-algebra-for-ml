/* p5-05 — The Jacobian: nonlinear maps are locally linear. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, lerpM = LA.lerpM, C = LA.color;
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const I2 = () => [[1, 0], [0, 1]];
  const P0 = [1, 0.5];
  // the neural layer of p2-12 used by the 'layer' map
  const W = [[1.5, 0.5], [-0.5, 1.2]], B = [0.2, -0.3];
  const BOUNDS = { x0: -3, x1: 3, y0: -3, y1: 3 };
  const divRamp = t => C.diverge((t - 0.5) * 2);
  const ZH = 0.35, ZSQ = 0.1;                       // zoomed camera half-height and tiny-square side
  /** static number for the math panel (tex.num mis-aligns when a panel also holds matrices; see report) */
  const nT = (x, color, digits) => '<span class="tx-nv ' + LA.tex.colorClass(color) + '">' + LA.fmt(x, digits) + '</span>';
  /** camera framing the image of the tiny square of side sq at F(p) */
  function zoomTarget(s, sq) { const fp = mapF(s, s.p), J = jac(s, s.p); return N.add(fp, N.scale(N.matVec(J, [1, 1]), sq / 2)); }

  /* ---------- the maps and their Jacobians (closed form) ---------- */
  function mapF(s, x) {
    if (s.map === 'layer') { const z = N.add(N.scale(N.matVec(W, x), s.c), B); return [Math.tanh(z[0]), Math.tanh(z[1])]; }
    if (s.map === 'swirl') { const r = Math.hypot(x[0], x[1]); const th = 0.6 * r; const c = Math.cos(th), sn = Math.sin(th); return [c * x[0] - sn * x[1], sn * x[0] + c * x[1]]; }
    return [x[0] + 0.4 * x[1] * x[1], x[1] + 0.4 * x[0] * x[0]];
  }
  function jac(s, x) {
    if (s.map === 'layer') { const z = N.add(N.scale(N.matVec(W, x), s.c), B); const d = [1 - Math.pow(Math.tanh(z[0]), 2), 1 - Math.pow(Math.tanh(z[1]), 2)]; return [[d[0] * s.c * W[0][0], d[0] * s.c * W[0][1]], [d[1] * s.c * W[1][0], d[1] * s.c * W[1][1]]]; }
    if (s.map === 'swirl') {
      const r = Math.hypot(x[0], x[1]); if (r < 1e-9) return I2();
      const th = 0.6 * r, c = Math.cos(th), sn = Math.sin(th);
      const R = [[c, -sn], [sn, c]], dR = [[-sn, -c], [c, -sn]];   // d/dθ of the rotation
      const u = N.matVec(dR, x), gth = [0.6 * x[0] / r, 0.6 * x[1] / r];   // ∂θ/∂x
      return N.addM(R, N.outer(u, gth));
    }
    return [[1, 0.8 * x[1]], [0.8 * x[0], 1]];
  }
  /** Jacobian of the partially warped map (1−w)·x + w·F(x) */
  function jacW(s, x) { return lerpM(I2(), jac(s, x), s.warp); }
  const detAt = (s, x) => N.det(jac(s, x));
  function layerD(s, x) { const z = N.add(N.scale(N.matVec(W, x), s.c), B); return { z, d: [1 - Math.pow(Math.tanh(z[0]), 2), 1 - Math.pow(Math.tanh(z[1]), 2)] }; }
  function dvec(s) { const n = N.norm(s.d); const u = n > 1e-9 ? N.scale(s.d, 1 / n) : [1, 0]; return N.scale(u, s.dlen); }
  function heatGrid() { const rand = LA.rng(505); const gr = []; for (let i = 0; i < 16; i++) { const row = []; for (let j = 0; j < 24; j++) row.push(LA.rng.gauss(rand) * 0.6); gr.push(row); } return gr; }

  /* ---------- drawing helpers ---------- */
  function drawWarp(g, s, alpha) {
    const Fw = x => lerpV(x, mapF(s, x), s.warp);
    const sp = s.map === 'layer' ? { lo: -3, hi: 3, step: 0.5 } : { lo: -6, hi: 6, step: 1 };
    for (let i = sp.lo; i <= sp.hi + 1e-9; i += sp.step) {
      const a = [], b = [];
      for (let k = 0; k <= 64; k++) { const u = sp.lo + (sp.hi - sp.lo) * k / 64; a.push(Fw([i, u])); b.push(Fw([u, i])); }
      const axis = Math.abs(i) < 1e-9;
      const st = { color: axis ? 'gridAxis' : 'grid', width: axis ? 1.6 : 1, alpha };
      g.polyline(a, st); g.polyline(b, st);
    }
  }
  /** the small square at p (outline) and its curved image under the current warp */
  function drawSquare(g, s, sq, alpha) {
    const p = s.p, Fw = x => lerpV(x, mapF(s, x), s.warp);
    const corners = [[p[0], p[1]], [p[0] + sq, p[1]], [p[0] + sq, p[1] + sq], [p[0], p[1] + sq]];
    const cols = ['i', 'hud', 'hud', 'j'];        // bottom green, right gray, top gray, left red
    for (let e = 0; e < 4; e++) {
      const A = corners[e], Bc = corners[(e + 1) % 4];
      g.line(A, Bc, { color: cols[e], width: cols[e] === 'hud' ? 1 : 2, alpha: 0.55 * alpha, dashed: cols[e] === 'hud' });
      const pts = []; for (let k = 0; k <= 12; k++) pts.push(Fw(lerpV(A, Bc, k / 12)));
      g.polyline(pts, { color: cols[e], width: cols[e] === 'hud' ? 1.2 : 2.6, alpha });
    }
  }
  /** parallelogram of J (side sq) at F(p) with colored column arrows */
  function drawJ(g, s, sq, opts) {
    opts = opts || {};
    const fp = lerpV(s.p, mapF(s, s.p), s.warp), J = opts.J || jacW(s, s.p);
    const u = N.scale(N.col(J, 0), sq), v = N.scale(N.col(J, 1), sq);
    g.parallelogram(u, v, { origin: fp, fill: opts.fill || 'v', alpha: opts.alpha === undefined ? 0.22 : opts.alpha, stroke: opts.stroke || 'gold', width: opts.width || 1.6, orient: true, dash: opts.dashed ? [6, 5] : null });
    if (opts.arrows !== false) {
      g.vector(fp, N.add(fp, u), { color: 'i', width: 2.6, head: 10, label: opts.labels ? '∂F/∂x' : undefined, labelAt: 1, labelOffset: -18, alpha: opts.arrowAlpha });
      g.vector(fp, N.add(fp, v), { color: 'j', width: 2.6, head: 10, label: opts.labels ? '∂F/∂y' : undefined, labelAt: 1, labelOffset: 18, alpha: opts.arrowAlpha });
    }
  }
  function drawZero(g, s, alpha) { g.contours(x => detAt(s, x), [0], BOUNDS, { res: 60, color: 'neg', width: 2, alpha: alpha === undefined ? 0.9 : alpha }); }

  LA.registerLesson({
    id: 'p5-05-jacobian',
    part: 5,
    title: 'The Jacobian: nonlinear maps are locally linear',
    subtitle: 'Zoom into any point of a curved warp and you see a parallelogram: its sides are the columns of J.',
    minutes: 10,
    prereqs: ['p2-01-linear-transformations', 'p2-06-determinant', 'p5-01-gradient', 'p2-12-neural-layer'],
    tags: ['core'],
    badge: { drawnIn: '2D → 2D map', realSize: 'a 784→128 layer has a 128×784 Jacobian; a whole digit network’s is 10×784', transfers: 'J = matrix of partials, rows = gradients of outputs, columns = where basis steps land, F(p + d) ≈ F(p) + Jd — exact for any input/output sizes; det J needs a square J' },
    cantShow: ['non-square Jacobians have no area factor; only singular values (p4-07) measure their stretch', 'ReLU’s Jacobian jumps between regions; we draw smooth tanh', 'the 128×784 picture — only its 2×2 slices look like this'],
    terms: [
      { term: 'nonlinear map', def: 'A rule sending each point to a point whose image of the grid bends, so no single matrix describes it.' },
      { term: 'Jacobian', def: 'The matrix of all partial derivatives ∂Fᵢ/∂xⱼ of a map at one point: its local linear approximation.' },
      { term: 'local linearization', def: 'The first-order rule F(p + d) ≈ F(p) + Jd for small displacements d.' },
      { term: 'area factor', def: '|det J|: how much the area of a tiny patch at p is scaled by the map.' },
      { term: 'orientation flip', def: 'det J < 0: the tiny patch at p is turned over by the map.' },
    ],
    atlas: ['backprop', 'mlp', 'tensors'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.4], halfH: 3.2 }),
        map: 'bend', p: P0.slice(), c: 1, warp: 1, warpAlpha: 1, sq: 0.2,
        d: [0.3, 0.2], dlen: 0.36, showPred: false, showSquare: false, showJ: false, jLabels: false,
        showRows: 0, hl: null, pulse: 0, heatA: 0, showDetHeat: false, showZero: false,
        ghostP: null, probes: false, cross: null, magnifier: false,
        guessJ: I2(), showGuessJ: false, areaA: 0, mlTie: false, recap: false, mathMode: 'F',
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.mlTie) { renderML(g, s); return; }
      const p = s.p, fp = lerpV(p, mapF(s, p), s.warp);
      const heatA = Math.max(s.heatA, s.showDetHeat ? 1 : 0);
      g.ghostGrid();
      if (heatA > 0) g.heat2d(x => detAt(s, x), BOUNDS, { res: 48, range: [-0.5, 2.5], ramp: divRamp, alpha: 0.8 * heatA });
      g.axes({ labels: s.cam.halfH > 0.8 });
      if (s.showZero || heatA > 0.5) drawZero(g, s, s.showZero ? 1 : heatA);
      drawWarp(g, s, s.warpAlpha * (heatA > 0 ? 0.45 : 1));
      if (s.showSquare) drawSquare(g, s, s.sq, 1);
      if (s.ghostP) { const J0 = jac(s, s.ghostP), f0 = mapF(s, s.ghostP); const u = N.scale(N.col(J0, 0), s.sq), v = N.scale(N.col(J0, 1), s.sq); g.parallelogram(u, v, { origin: f0, fill: 'hud', alpha: 0.12, stroke: 'hud', width: 1.2 }); g.label('J(' + LA.fmt(s.ghostP[0]) + ', ' + LA.fmt(s.ghostP[1]) + ')', f0, { color: 'muted', size: 12, anchor: 'below', dy: 8 }); }
      if (s.showJ) drawJ(g, s, s.sq, { labels: s.jLabels, width: 1.6 + 2 * s.pulse });
      if (s.areaA > 0 && s.showSquare && s.showJ) {
        const a0 = s.sq * s.sq, J = jacW(s, p), mid = N.add(fp, N.scale(N.matVec(J, [1, 1]), s.sq / 2));
        g.label('area ' + LA.fmt(a0), N.add(p, [s.sq / 2, s.sq / 2]), { color: 'text', size: 12 });
        g.label('area ' + LA.fmt(a0 * Math.abs(N.det(J))), mid, { color: 'gold', size: 12, dx: 44 });
      }
      if (s.showGuessJ) { const u = N.scale(N.col(s.guessJ, 0), s.sq), v = N.scale(N.col(s.guessJ, 1), s.sq); g.parallelogram(u, v, { origin: fp, fill: 'w', alpha: 0.15, stroke: 'w', width: 1.8, dash: [6, 5] }); g.label('your J', N.add(fp, N.scale(N.add(u, v), 0.5)), { color: 'w', size: 12 }); }
      if (s.showRows > 0) {
        const J = jacW(s, p);
        g.vector(p, N.add(p, N.scale(J[0], 0.5)), { color: 'v', width: 2.8, label: '∇F₁', labelOffset: -16 });
        if (s.showRows > 1) g.vector(p, N.add(p, N.scale(J[1], 0.5)), { color: 'v', width: 2.8, label: '∇F₂', labelOffset: 16 });
      }
      if (s.showPred) {
        const d = dvec(s), J = jacW(s, p), q = N.add(p, d);
        const pred = N.add(fp, N.matVec(J, d)), tru = lerpV(q, mapF(s, q), s.warp);
        g.vector(p, q, { color: 'w', width: 2.8, label: 'd', labelAt: 0.5, labelOffset: -14 });
        g.vector(fp, pred, { color: 'w', width: 2.2, dashed: true, label: 'Jd', labelAt: 0.5, labelOffset: -14 });
        g.line(pred, tru, { color: 'residual', width: 1.6, dashed: [3, 3] });
        g.point(pred, { r: 5, color: 'w', label: 'F(p) + Jd', dy: -6, dx: -52 });
        g.point(tru, { r: 5, color: 'v', label: 'F(p + d)', dy: 6, dx: 48 });
        g.label('error ' + LA.fmt.fixed(N.dist(pred, tru), 3), lerpV(pred, tru, 0.5), { color: 'muted', size: 12, dx: 44, dy: 24 });
      }
      if (s.probes) {
        const J = jac(s, p); const e = N.eigSym(N.matMul(N.transpose(J), J)); const v = e.vectors[1];
        [1, -1].forEach(sg => { const q = N.add(p, N.scale(v, 0.3 * sg)); g.point(q, { r: 4.5, color: 'w' }); g.line(q, mapF(s, q), { color: 'w', width: 1, dashed: [3, 4], alpha: 0.6 }); g.point(mapF(s, q), { r: 4.5, color: 'w' }); });
        g.label('two inputs, one output', mapF(s, p), { color: 'w', size: 12, anchor: 'below', dy: 14 });
      }
      if (s.cross) { g.label('“one global matrix”', s.cross, { color: 'muted', size: 15, anchor: 'left', dx: 24 }); g.marker.cross(s.cross, { size: 22, dx: 8 }); }
      // the probe and its image
      g.point(p, { r: 5, color: 'v', label: 'p', dy: -11, dx: -11 });
      if (s.warp > 0) { g.point(fp, { r: 5, color: 'v', stroke: 'white', label: 'F(p)', dy: -12, dx: 14 }); if (!s.showPred) g.line(p, fp, { color: 'v', width: 1, dashed: [3, 4], alpha: 0.5 }); }
      if (s.magnifier) renderMagnifier(g, s);
      // handles
      g.handle('p', p, (q, st) => { st.p = [LA.clamp(q[0], -3, 3), LA.clamp(q[1], -3, 3)]; }, { color: 'v' });
      if (s.showPred) g.handle('d', N.add(p, dvec(s)), (q, st) => { const dd = N.sub(q, st.p); const n = N.norm(dd); if (n > 1e-6) { st.d = dd; st.dlen = LA.clamp(n, 0.05, 1.2); } }, { color: 'w' });
    },

    math(s, tex) {
      const m = s.mathMode, J = jac(s, s.p);
      const Jsym = tex.sym('J');
      if (m === 'F') return tex.lines(tex.eq(tex.sym('F'), tex.paren(tex.sym('x') + tex.op(', ') + tex.sym('y')), tex.op('='), tex.paren(tex.sym('x') + tex.op('+') + nT(0.4) + tex.sym('y²') + tex.op(', ') + tex.sym('y') + tex.op('+') + nT(0.4) + tex.sym('x²'))), tex.eq(tex.sym('F'), tex.paren(tex.vec(s.p, { color: 'v' })), tex.op('='), tex.vec(mapF(s, s.p), { color: 'v' })));
      if (m === 'J') {
        const hl = s.hl === 'rows' ? (i, j) => i === s.hlIdx : s.hl === 'cols' ? (i, j) => j === s.hlIdx : undefined;
        const sym = tex.mat([[{ html: '∂F₁/∂x' }, { html: '∂F₁/∂y' }], [{ html: '∂F₂/∂x' }, { html: '∂F₂/∂y' }]], { colors: 'cols', highlight: hl });
        const line2 = s.hl === 'rows' ? tex.eq(tex.text('row i of J ='), tex.T(tex.sym('∇F', { sub: 'i', color: 'v' }))) : s.hl === 'cols' ? tex.eq(tex.text('column j of J = ∂F/∂xⱼ (where a basis step lands)')) : tex.eq(tex.text('at p ='), tex.vec(s.p, { color: 'v' }));
        return tex.lines(tex.eq(Jsym, tex.op('='), sym, tex.op('='), tex.mat(J, { colors: 'cols', highlight: hl })), line2);
      }
      if (m === 'pred') { const d = dvec(s), q = N.add(s.p, d); const pred = N.add(mapF(s, s.p), N.matVec(J, d)), tru = mapF(s, q); return tex.lines(tex.eq(tex.sym('F'), tex.paren(tex.sym('p') + tex.op(' + ') + tex.sym('d', { color: 'w' })), tex.op('≈'), tex.sym('F'), tex.paren(tex.sym('p')), tex.op('+'), Jsym, tex.sym('d', { color: 'w' })), tex.eq(tex.text('prediction'), tex.vec(pred, { color: 'w' }), tex.text('true'), tex.vec(tru, { color: 'v' }), tex.text('error'), nT(N.dist(pred, tru), null, 3))); }
      if (m === 'det') return tex.lines(tex.eq(tex.text('det'), Jsym, tex.op('='), nT(1), tex.op('−'), nT(0.64), tex.sym('x'), tex.sym('y'), tex.op('='), nT(detAt(s, s.p), 'gold')), tex.eq(tex.text('area of the patch ×'), nT(Math.abs(detAt(s, s.p)), 'gold')));
      if (m === 'Jat') return tex.lines(tex.eq(Jsym, tex.paren(nT(s.p[0]) + tex.op(', ') + nT(s.p[1])), tex.op('='), tex.mat(J, { colors: 'cols' })), tex.eq(tex.text('a matrix per point: ∂F/∂x = (1, 0.8x), ∂F/∂y = (0.8y, 1)')));
      if (m === 'detsign') { const dt = detAt(s, s.p); return tex.lines(tex.eq(tex.text('det'), Jsym, tex.op('='), nT(1), tex.op('−'), nT(0.64), tex.sym('x'), tex.sym('y'), tex.op('='), nT(dt, dt < -1e-3 ? 'neg' : dt < 1e-3 ? 'null' : 'gold')), tex.eq(tex.text(dt < -1e-3 ? 'negative: the patch is flipped' : dt < 1e-3 ? 'zero on the curve xy = 1.5625: a dimension is lost' : 'positive: orientation kept', { color: dt < -1e-3 ? 'neg' : dt < 1e-3 ? 'null' : 'gold' }))); }
      if (m === 'layer') { const ld = layerD(s, s.p); return tex.lines(tex.eq(Jsym, tex.op('='), tex.sym('D'), tex.sym('W'), tex.op('·'), tex.sym('D'), tex.op('='), tex.text('diag(1 − tanh²(zᵢ)) ='),tex.text('diag(', {}), nT(ld.d[0], 'i', 3), tex.op(', '), nT(ld.d[1], 'j', 3), tex.op(')')), tex.eq(tex.sym('z'), tex.op('='), tex.vec(ld.z, { color: 'v' }), tex.op('·'), Jsym, tex.op('='), tex.mat(J, { colors: 'cols' }))); }
      if (m === 'ml') return tex.eq(Jsym, tex.op('∈'), tex.sym('ℝ', { sup: '128×784' }), tex.text(' for a 784→128 layer; a 2×2 slice is this lesson’s picture'));
      if (m === 'guess') return tex.lines(tex.eq(tex.text('at p ='), tex.vec(s.p, { color: 'v' }), tex.text(':'), tex.sym('∂F₁/∂y = 0.8y, ∂F₂/∂x = 0.8x')), tex.eq(tex.text('yours:'), tex.mat(s.guessJ, { colors: 'w' })));
      return '';
    },

    steps: [
      { title: 'A warp that bends the grid', duration: 3,
        narration: 'Here is a <b>nonlinear map</b>: F(x, y) = (x + 0.4y², y + 0.4x²). Watch the grid lines bend — p2-01’s red ✕ rule — so there is no single matrix for the whole plane. The yellow point p rides along to F(p). Yet every linear idea you own is about to come back.',
        setup(s) { s.map = 'bend'; s.p = P0.slice(); s.c = 1; s.warp = 0; s.warpAlpha = 1; s.sq = 0.2; s.showPred = false; s.showSquare = false; s.showJ = false; s.jLabels = false; s.showRows = 0; s.hl = null; s.pulse = 0; s.heatA = 0; s.showDetHeat = false; s.showZero = false; s.ghostP = null; s.probes = false; s.cross = null; s.magnifier = false; s.showGuessJ = false; s.areaA = 0; s.mlTie = false; s.recap = false; s.mathMode = 'F'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.4], halfH: 3.2 }); },
        at(s, t) { s.warp = seg(t, 0.1, 0.9); } },

      { title: 'Zoom in', duration: 4,
        narration: 'Put a tiny square around the point, with a green bottom edge and a red left edge, and magnify. The curved image straightens: at small scale the warp looks like a plain linear transformation — a parallelogram with a green side and a red side. Nonlinear maps are locally linear.',
        setup(s) { s.warp = 1; s.showSquare = true; s.mathMode = 'F'; s._c1 = zoomTarget(s, ZSQ); },
        at(s, t) {
          // zoom geometrically 3.2 → ZH while the square shrinks with the view: same size on screen, ever straighter edges
          const u = seg(t, 0.1, 0.95, 'smooth'), h = 3.2 * Math.pow(ZH / 3.2, u);
          s.sq = ZSQ * h / ZH;
          if (s.cam.auto !== false) { s.cam.halfH = h; s.cam.center = lerpV([0.3, 0.4], s._c1, (3.2 - h) / (3.2 - ZH)); }
        } },

      { title: 'Read the parallelogram: the Jacobian', duration: 4,
        narration: 'The green side is where a tiny ' + ihat + '-step lands: ∂F/∂x = (1, 0.8x) = (1, 0.8). The red side is a tiny ' + jhat + '-step: ∂F/∂y = (0.8y, 1) = (0.4, 1). Put them in columns and you have the <b>Jacobian</b> J at p.',
        setup(s) { s.sq = ZSQ; s.cam = Object.assign(LA.cam.defaults2d(), { center: zoomTarget(s, ZSQ), halfH: ZH }); s.showJ = true; s.jLabels = true; s.mathMode = 'J'; s.hl = null; },
        at(s, t) { s.hl = t > 0.3 ? 'cols' : null; s.hlIdx = t > 0.65 ? 1 : 0; } },

      { title: 'Rows are gradients, columns are landing spots', duration: 3,
        narration: 'Two readings of the same box. Each row is the gradient (p5-01) of one output coordinate: row 1 is ∇F₁, drawn yellow at p. Each column is where a basis step lands (p2-01): the parallelogram’s sides. The Jacobian is the gradient’s big sibling: several outputs, one matrix.',
        setup(s) { s.jLabels = false; s.showSquare = false; s.showRows = 0; s.hl = 'rows'; s.hlIdx = 0; s._cam0 = LA.clone(s.cam); },
        at(s, t) {
          LA.cam.tween(s, { center: s._cam0.center, halfH: s._cam0.halfH }, { center: [1.05, 0.75], halfH: 1.3 }, seg(t, 0, 0.3));
          s.sq = LA.lerp(ZSQ, 0.3, seg(t, 0, 0.3));
          s.showRows = t > 0.1 ? (t > 0.35 ? 2 : 1) : 0;
          s.hl = t > 0.6 ? 'cols' : 'rows'; s.hlIdx = t > 0.6 ? (t > 0.8 ? 1 : 0) : (t > 0.35 ? 1 : 0);
          s.pulse = t > 0.6 ? seg.pulse(t, 0.6, 1) : 0;
        } },

      { title: 'The first-order rule, in 2D', duration: 4, interactive: true,
        narration: 'For a small displacement <span class="c-w">d</span> from p, the map moves it by about Jd: F(p + d) ≈ F(p) + Jd, the <b>local linearization</b>. The purple prediction and the true yellow landing point agree when d is small and drift apart as d grows. Drag the tip of d to see.',
        setup(s) { s.showRows = 0; s.hl = null; s.pulse = 0; s.showJ = false; s.showPred = true; s.d = [0.3, 0.2]; s.dlen = 0.6; s.mathMode = 'pred'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.35, 1.1], halfH: 1.0 }); },
        at(s, t) { s.dlen = LA.lerp(0.6, 0.2, seg(t, 0.15, 0.9)); } },

      { title: 'det J is the local area factor', duration: 4,
        narration: 'From p2-06, a matrix scales area by its determinant. Here det J = 1 − 0.64xy = 0.68, the local <b>area factor</b>: the square’s area 0.25 becomes 0.17. Color the whole plane by det J: teal above 1 where xy is negative, rose below 1, reaching zero on the violet curve xy = 1.56.',
        setup(s) { s.showPred = false; s.d = [0.3, 0.2]; s.dlen = 0.36; s.showJ = true; s.showSquare = true; s.sq = 0.5; s.areaA = 1; s.mathMode = 'det'; s.heatA = 0; s.showZero = false; s._cam0 = LA.clone(s.cam); },
        at(s, t) { LA.cam.tween(s, { center: s._cam0.center, halfH: s._cam0.halfH }, { center: [0.3, 0.4], halfH: 3.0 }, seg(t, 0, 0.35)); s.heatA = seg(t, 0.35, 0.75); s.showZero = t > 0.8; } },

      { title: '✕ One matrix for the whole map', duration: 3, misconception: true,
        narration: '“The derivative of a map is one matrix.” Not for a curved map. Move to (−1, 1): the parallelogram changes shape and J becomes [[1, 0.8],[−0.8, 1]]. The Jacobian is a matrix <i>per point</i> — a field of matrices, like the field of gradients in p5-01.',
        setup(s) { s.heatA = 0; s.areaA = 0; s.showZero = false; s.showSquare = false; s.sq = 0.45; s.ghostP = P0.slice(); s.mathMode = 'Jat'; s.magnifier = true; s._p0 = LA.clone(s.p); s._p1 = [-1, 1]; s.cross = null; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); },
        at(s, t) { s.p = lerpV(s._p0, s._p1, seg(t, 0.1, 0.6)); s.cross = t > 0.7 ? [-1.9, 2.75] : null; } },

      { title: 'Where the map flips', duration: 3,
        narration: 'At (1.5, 1.5) the determinant is 1 − 1.44 = −0.44. The local parallelogram has turned over — the fill goes violet as in p2-06, an <b>orientation flip</b> — and the grid folds over itself there. Orientation is a local property too.',
        setup(s) { s.cross = null; s.ghostP = null; s.mathMode = 'detsign'; s._p0 = LA.clone(s.p); s._p1 = [1.5, 1.5]; s.showZero = true; s.sq = 0.3; s._cam0 = LA.clone(s.cam); },
        at(s, t) { s.p = lerpV(s._p0, s._p1, seg(t, 0.1, 0.7)); LA.cam.tween(s, { center: s._cam0.center, halfH: s._cam0.halfH }, { center: [1.9, 2.1], halfH: 1.6 }, seg(t, 0.1, 0.7)); } },

      { title: 'Where the map squashes', duration: 3,
        narration: 'On the violet curve xy = 1.56 the determinant is zero: the parallelogram collapses to a segment, and two different nearby inputs, the purple probes, land on the same output. Locally the map loses a dimension (p2-08), even though globally it is a smooth warp.',
        setup(s) { s._p0 = LA.clone(s.p); s._p1 = [1.25, 1.25]; s.probes = false; },
        at(s, t) { s.p = lerpV(s._p0, s._p1, seg(t, 0.1, 0.6)); s.probes = t > 0.65; } },

      { title: 'A neural layer’s Jacobian', duration: 5,
        narration: 'The layer from p2-12, F(x) = tanh(Wx + b), is also a warp. Its Jacobian is J = DW: the matrix W (p2-01) multiplied by a diagonal D of tanh slopes 1 − tanh²(z). Where tanh is flat, a row of D is near zero and the parallelogram becomes a sliver.',
        setup(s) { s.probes = false; s.showZero = false; s.map = 'layer'; s.c = 1; s.sq = 0.2; s.p = [0.3, 0.2]; s._p0 = [0.3, 0.2]; s._p1 = [2, 1.5]; s.mathMode = 'layer'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.5, 0.3], halfH: 1.9 }); },
        at(s, t) { s.p = lerpV(s._p0, s._p1, seg(t, 0.45, 0.9)); } },

      { title: 'Why ML cares: every layer is a Jacobian', duration: 4, mlTie: true,
        narration: 'Backprop (next lesson) multiplies these matrices layer by layer. Normalizing flows track log|det J| to follow how a density stretches. A 784→128 layer has a 128×784 Jacobian — a heatmap, not a parallelogram. The picture stops at 3. The formula does not.',
        setup(s) { s.mlTie = true; s.magnifier = false; s.mathMode = 'ml'; s.mlA = 0; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0.85], halfH: 3.0 }); },
        at(s, t) { s.mlA = seg(t, 0.1, 0.8, 'linear'); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Zoom into any smooth warp and you see a matrix. Its columns are where tiny basis steps land, its rows are gradients, and F(p + d) ≈ F(p) + Jd. Its determinant is the local area factor; its sign tells you whether the patch flipped.',
        cards: ['<b>Locally, a warp is a matrix.</b> Magnify a tiny square and it becomes the parallelogram of J.', '<b>Columns land basis steps; rows are gradients.</b> J = [∂F/∂x ∂F/∂y], row i = ∇Fᵢᵀ.', '<b>det J = local area factor</b>, sign = flip; zero where the map loses a dimension.'],
        setup(s) { s.mlTie = false; s.recap = true; s.mathMode = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the yellow probe p and the tip of the purple <span class="c-w">d</span>. The parallelogram at F(p) is the Jacobian’s picture (violet when det J < 0); the magnifier shows the ×8 view around F(p).</p>',
      setup(s) { s.map = 'bend'; s.p = P0.slice(); s.warp = 1; s.sq = 0.4; s.showJ = true; s.jLabels = false; s.showPred = true; s.d = [0.3, 0.2]; s.dlen = 0.36; s.showSquare = true; s.magnifier = true; s.mathMode = 'J'; s.hl = null; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); },
      controls: [
        { type: 'preset', label: 'Map', items: [
          { label: 'Bend', apply(s) { s.map = 'bend'; s.c = 1; s.sq = 0.4; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); } },
          { label: 'Layer (healthy)', apply(s) { s.map = 'layer'; s.c = 1; s.sq = 0.2; s.p = [0.3, 0.2]; s.dlen = Math.min(s.dlen, 0.3); s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 1.45 }); } },
          { label: 'Layer (saturated, c = 3)', apply(s) { s.map = 'layer'; s.c = 3; s.sq = 0.2; s.p = [0.6, 0.4]; s.dlen = Math.min(s.dlen, 0.3); s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 1.45 }); } },
          { label: 'Swirl', apply(s) { s.map = 'swirl'; s.c = 1; s.sq = 0.4; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 3.2 }); } },
        ] },
        { type: 'toggle', bind: 'magnifier', label: 'Magnifier (×8 around F(p))' },
        { type: 'slider', bind: 'dlen', min: 0.05, max: 0.6, step: 0.01, label: 'Displacement size ‖d‖', color: 'w' },
        { type: 'toggle', bind: 'showDetHeat', label: 'det J heatmap' },
        { type: 'slider', bind: 'c', min: 0.2, max: 3, step: 0.05, label: 'Layer weight scale c' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow point to where the map flips orientation locally (det J < 0).',
        setup(s) { s.map = 'bend'; s.c = 1; s.warp = 1; s.p = [0, 0]; s.sq = 0.45; s.showJ = true; s.showSquare = false; s.showPred = false; s.magnifier = false; s.mathMode = 'detsign'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); },
        judge(s) { const d = detAt(s, s.p); return { ok: d < -LA.TOL.det, progress: Math.max(0, Math.min(1, (1 - d) / 1.1)) }; },
        hint: 'The fill turns violet where xy is large and positive — try the top-right corner.',
        solution(s) { s.p = [1.5, 1.5]; } },
      { kind: 'manipulate',
        prompt: 'Find a point where the local area factor det J is 0.5.',
        setup(s) { s.map = 'bend'; s.c = 1; s.warp = 1; s.p = [0, 0]; s.sq = 0.45; s.showJ = true; s.showSquare = false; s.showPred = false; s.magnifier = false; s.showDetHeat = true; s.mathMode = 'det'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); },
        judge(s) { return LA.judge.scalar(detAt(s, s.p), 0.5, LA.TOL.det); },
        hint: 'det J = 1 − 0.64xy; you need xy ≈ 0.78.',
        solution(s) { s.p = [1, 0.78]; } },
      { kind: 'entry',
        prompt: 'Type the Jacobian of F(x, y) = (x + 0.4y², y + 0.4x²) at p = (0.5, 1). Your J is drawn as a dashed purple parallelogram; in the magnifier a right answer hugs the curved image of the tiny square.',
        bind: 'guessJ', answer: [[1, 0.8], [0.4, 1]], tol: LA.TOL.entry, label: 'J', colors: 'cols',
        setup(s) { s.map = 'bend'; s.c = 1; s.warp = 1; s.p = [0.5, 1]; s.sq = 0.45; s.showJ = false; s.showSquare = false; s.showPred = false; s.magnifier = true; s.showGuessJ = true; s.guessJ = I2(); s.mathMode = 'guess'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.6], halfH: 2.8 }); },
        hint: '∂F₁/∂y = 0.8y, ∂F₂/∂x = 0.8x.' },
    ],
  });

  /* ---------- magnifier inset: ×8 around F(p) ---------- */
  function renderMagnifier(g, s) {
    const Wd = g.size.w, Hh = g.size.h; const w = Math.round(Math.min(Wd * 0.42, 300)), h = w;
    const fp = lerpV(s.p, mapF(s, s.p), s.warp);
    g.marker.ring(fp, { r: 10, color: 'gold' });
    g.viewport({ x: Wd - w - 10, y: Hh - h - 10, w, h }, { view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: fp, halfH: 0.46 }), bg: 'panel' }, (g2, r) => {
      g2.ghostGrid();
      drawWarp(g2, s, 0.8);
      const sv = LA.clone(s); sv.sq = 0.2;
      drawSquare(g2, sv, 0.2, 1);
      if (!s.showGuessJ) drawJ(g2, sv, 0.2, { arrows: true, fill: 'v', alpha: 0.3, dashed: true });   // check 3: never draw the true J
      if (s.showGuessJ) { const u = N.scale(N.col(s.guessJ, 0), 0.2), v = N.scale(N.col(s.guessJ, 1), 0.2); g2.parallelogram(u, v, { origin: fp, fill: 'w', alpha: 0.15, stroke: 'w', width: 1.8, dash: [6, 5] }); }
      g2.point(fp, { r: 4, color: 'v' });
      g2.label('×8 around F(p) · det J = ' + LA.fmt(detAt(s, s.p)), [r.x + r.w / 2, r.y + 14], { screen: true, size: 12, color: 'text', bold: true });
    });
  }

  /* ---------- ML tie-in: a 16×24 slice of a Jacobian + chained layers ---------- */
  function renderML(g, s) {
    const Wd = g.size.w, Hh = g.size.h;
    const grid = heatGrid();
    const bounds = { x0: -2.1, x1: 2.1, y0: 0.95, y1: 3.45 };
    g.heat2d(grid, bounds, { range: [-1.5, 1.5], ramp: divRamp, alpha: 0.95, smooth: false });
    g.polyline([[bounds.x0, bounds.y0], [bounds.x1, bounds.y0], [bounds.x1, bounds.y1], [bounds.x0, bounds.y1]], { color: 'line', closed: true, width: 1 });
    g.label('a 16×24 slice of a 128×784 Jacobian (synthetic, seed 505)', [0, bounds.y1], { anchor: 'above', dy: -6, size: 12.5, color: 'text' });
    g.label('J ∈ ℝ¹²⁸ˣ⁷⁸⁴: one row per output, one column per input pixel', [0, bounds.y0], { anchor: 'below', dy: 6, size: 11.5, color: 'muted' });
    // three tiny Jacobians chained
    const Js = [[[1, 0.4], [0.3, 1]], [[0.8, -0.3], [0.5, 0.9]], [[1.1, 0.2], [-0.4, 0.7]]];
    const xs = [-1.7, 0, 1.7], y0 = -1.1, sc = 0.62;
    const n = Math.min(3, Math.floor(s.mlA * 3.999));
    for (let k = 0; k < 3; k++) {
      const o = [xs[k] - 0.3, y0]; const a = k < n ? 1 : 0.25;
      g.parallelogram(N.scale(N.col(Js[k], 0), sc), N.scale(N.col(Js[k], 1), sc), { origin: o, fill: 'v', alpha: 0.2 * a, stroke: 'gold', width: 1.4 });
      g.vector(o, N.add(o, N.scale(N.col(Js[k], 0), sc)), { color: 'i', width: 2, head: 8, alpha: a });
      g.vector(o, N.add(o, N.scale(N.col(Js[k], 1), sc)), { color: 'j', width: 2, head: 8, alpha: a });
      g.label('layer ' + (k + 1) + ': J' + ['₁', '₂', '₃'][k], [xs[k], y0 - 0.35], { size: 12, color: 'hud' });
      if (k < 2) g.vector([xs[k] + 0.55, y0 + 0.3], [xs[k + 1] - 0.5, y0 + 0.3], { color: 'hud', width: 1.4, head: 8, alpha: k < n - 1 ? 0.9 : 0.3 });
    }
    g.label('chain rule: J₃ J₂ J₁ — multiply, right to left', [0, 0.3], { size: 12.5, color: 'gold' });
    g.hud('The picture stops at 3. The formula does not.', 18, Hh - 24, { size: 14, color: 'text' });
    g.hud('see p5-06', Wd - 16, Hh - 24, { size: 12, color: 'muted', anchor: 'right' });
  }

  /* ---------- recap: three mini-cards ---------- */
  function renderRecap(g, s) {
    const J0 = [[1, -0.5], [0.6, 1]];
    const bendS = { map: 'bend', c: 1, warp: 1, p: [0, 0] };
    const cards = [
      { title: 'square → parallelogram', cam: { center: [0.95, 0.6], halfH: 1.4 }, draw(g2) {
        g2.polygon([[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6]], { fill: 'hud', alpha: 0.08, stroke: null });
        g2.line([0, 0], [0.6, 0], { color: 'i', width: 2.4 }); g2.line([0, 0], [0, 0.6], { color: 'j', width: 2.4 }); g2.line([0.6, 0], [0.6, 0.6], { color: 'hud', width: 1, dashed: true }); g2.line([0, 0.6], [0.6, 0.6], { color: 'hud', width: 1, dashed: true });
        const o = [1.35, 0.1], u = N.scale(N.col(J0, 0), 0.55), v = N.scale(N.col(J0, 1), 0.55);
        g2.parallelogram(u, v, { origin: o, fill: 'v', alpha: 0.22, stroke: 'gold' });
        g2.vector(o, N.add(o, u), { color: 'i', width: 2.4, head: 9 }); g2.vector(o, N.add(o, v), { color: 'j', width: 2.4, head: 9 });
        g2.vector([0.45, 1.05], [1.05, 1.05], { color: 'hud', width: 1.4, head: 8, label: 'F', labelOffset: 10 });
      } },
      { title: 'columns & rows', cam: { center: [0.25, 0.4], halfH: 1.75 }, draw(g2) {
        g2.vector(N.col(J0, 0), { color: 'i', width: 2.6, label: '∂F/∂x', labelOffset: 14 }); g2.vector(N.col(J0, 1), { color: 'j', width: 2.6, label: '∂F/∂y', labelOffset: -14 });
        g2.vector(J0[0], { color: 'v', width: 1.8, dashed: true, label: '∇F₁', labelOffset: 14 }); g2.vector(J0[1], { color: 'v', width: 1.8, dashed: true, label: '∇F₂', labelOffset: -14 });
      } },
      { title: 'det J: sign = flip', cam: { center: [0, 0], halfH: 3.4 }, draw(g2) {
        g2.heat2d(x => detAt(bendS, x), BOUNDS, { res: 40, range: [-0.5, 2.5], ramp: divRamp, alpha: 0.85 });
        drawZero(g2, bendS, 1);
        g2.label('det J = 0', [1.7, 0.45], { color: 'neg', size: 11 });
      } },
    ];
    const Wd = g.size.w, Hh = g.size.h, pad = 10, cw = (Wd - pad * 4) / 3, ch = Math.min(Hh * 0.6, cw * 1.25), y = (Hh - ch) / 2;
    cards.forEach((cd, i) => {
      const r = { x: pad + i * (cw + pad), y, w: cw, h: ch };
      g.viewport(r, { view: '2d', cam: Object.assign(LA.cam.defaults2d(), cd.cam), bg: 'panel' }, (g2, rp) => { g2.ghostGrid(); g2.axes({ labels: false }); cd.draw(g2); g2.label(cd.title, [rp.x + rp.w / 2, rp.y + 16], { screen: true, size: 12.5, color: 'text', bold: true }); });
    });
  }
})();
