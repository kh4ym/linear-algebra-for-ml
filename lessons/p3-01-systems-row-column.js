/* p3-01 — Ax = b: row picture vs column picture. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt;
  const A0 = [[2, 1], [1, 3]], B0 = [4, 7];                       // the part's shared 2×2 system, x = [1; 2]
  const AF = [[1, 2], [2, 4]], B_NONE = [3, 1], B_MANY = [2, 4];  // the flattened system
  const A3 = [[1, 1, 0], [0, 1, 1], [1, 0, 1]], B3 = [3, 5, 4], X3 = [1, 2, 3];
  const RECIPES = [[2, 0], [0, 1], [1, 0.5]];                      // three recipes that all reach [2; 4]
  const CROSS_NONE = [-0.7, 1.25];                                 // between the two parallel lines
  const BOX = { x0: -3, x1: 5.6, y0: -1.6, y1: 8.2 };              // world box every 2×2 picture must show
  const CAM2 = () => ({ center: [1.3, 3.3], halfH: 5.2 });
  const CAM3 = () => Object.assign(LA.cam.defaults3d(), { target: [1, 2, 2.4], dist: 14.5, yaw: 0.44, pitch: 0.42 });
  const bb = '<span class="c-b">b</span>';
  const SUB = ['₁', '₂', '₃'];

  /* ---------- small geometry helpers ---------- */
  const det2 = A => A[0][0] * A[1][1] - A[0][1] * A[1][0];
  function rankDef(A) { const n1 = N.norm(N.col(A, 0)), n2 = N.norm(N.col(A, 1)); return Math.abs(det2(A)) < 0.1 * Math.max(1, n1 * n2); }
  function colDir(A) { const a1 = N.col(A, 0), a2 = N.col(A, 1); const d = N.norm(a1) >= N.norm(a2) ? a1 : a2; const n = N.norm(d); return n > 1e-9 ? N.scale(d, 1 / n) : [1, 0]; }
  function offLine(b, A) { const d = colDir(A); return N.dist(b, N.scale(d, N.dot(b, d))); }      // distance from b to the column line
  function inBox(p, m) { m = m || 0; return p[0] > BOX.x0 - m && p[0] < BOX.x1 + m && p[1] > BOX.y0 - m && p[1] < BOX.y1 + m; }
  function fitCam(w, h, box) { const bw = box.x1 - box.x0, bh = box.y1 - box.y0; return { center: [(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2], halfH: Math.max(bh / 2, bw / 2 * h / Math.max(1, w)) * 1.04 }; }
  /** the line {p : a·p = c}: closest point to the origin and a unit direction */
  function rowLine(a, c) { const nn = a[0] * a[0] + a[1] * a[1]; if (nn < 1e-9) return null; const n = Math.sqrt(nn); return { p0: [a[0] * c / nn, a[1] * c / nn], d: [-a[1] / n, a[0] / n] }; }
  /** the solution line of a rank-1 consistent system: {p (particular), d (null direction, integer-ish)} */
  function solLine(A, b) { const r = N.norm(A[0]) >= N.norm(A[1]) ? 0 : 1; const a = A[r]; if (N.norm(a) < 1e-6) return null; const p = Math.abs(a[0]) > 1e-6 ? [b[r] / a[0], 0] : [0, b[r] / a[1]]; return { p, d: simplest([-a[1], a[0]]) }; }
  /** divide an integer direction by the gcd of its entries ([−4; 2] → [−2; 1]); non-integer directions are kept */
  function simplest(d) {
    const ia = Math.round(d[0]), ib = Math.round(d[1]);
    if (Math.abs(ia - d[0]) > 1e-9 || Math.abs(ib - d[1]) > 1e-9) return d;
    let x = Math.abs(ia), y = Math.abs(ib); while (y) { const t = x % y; x = y; y = t; }
    return x > 1 ? [d[0] / x, d[1] / x] : d;
  }
  /** a spot on the line inside BOX, as far as possible from `avoid` */
  function labelSpot(L, avoid) {
    let best = null, bd = -1;
    for (let k = -4; k <= 4; k += 0.5) { const p = [L.p0[0] + L.d[0] * k, L.p0[1] + L.d[1] * k]; if (p[0] < BOX.x0 + 0.8 || p[0] > BOX.x1 - 0.8 || p[1] < BOX.y0 + 0.8 || p[1] > BOX.y1 - 0.8) continue; const dd = avoid ? N.dist(p, avoid) : Math.abs(k); if (dd > bd) { bd = dd; best = p; } }
    return best;
  }
  function drawRowLine(g, a, c, o) {
    const L = rowLine(a, c); if (!L) return null;
    const len = 16 * (o.prog === undefined ? 1 : o.prog); if (len <= 0.01) return L;   // grows from its nearest point outward
    g.line([L.p0[0] - L.d[0] * len, L.p0[1] - L.d[1] * len], [L.p0[0] + L.d[0] * len, L.p0[1] + L.d[1] * len], { color: o.color || 'hud', width: o.width || 2, alpha: o.alpha, dashed: o.dashed });
    return L;
  }
  /** tip-to-tail chain of unit copies of the columns (fractional last copy); prog ∈ [0,1] draws it progressively; returns the end */
  function drawChain(g, cols, weights, colors, prog, o) {
    o = o || {};
    let pos = cols[0].map(() => 0);
    const total = weights.reduce((acc, w) => acc + Math.abs(w), 0);
    let left = prog * total;
    for (let j = 0; j < cols.length; j++) {
      const w = weights[j], sgn = w < 0 ? -1 : 1; let rem = Math.abs(w);
      while (rem > 1e-6 && left > 1e-6) {
        const piece = Math.min(1, rem), part = Math.min(piece, left);
        const to = N.add(pos, N.scale(cols[j], sgn * part));
        if (part > 0.02) g.vector(pos, to, { color: colors[j], width: 2.4, head: 9, alpha: o.alpha === undefined ? 0.8 : o.alpha });
        pos = to; rem -= part; left -= part;
      }
    }
    return pos;
  }
  /* Engine workaround: LA.tex.reconcile updates tex.num spans in place by index, but its index list also contains
     matrix cells, so a changing number next to an unchanged matrix overwrites a matrix cell. Live numbers are
     therefore emitted as plain colored spans (part of the skeleton, so a change triggers a clean rebuild). */
  function safeTex(tex) { return Object.assign({}, tex, { num: (x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; } }); }
  function title(g, r, text) { g.label(text, [r.x + 10, r.y + 16], { screen: true, anchor: 'left', size: 13, color: 'muted', font: 'ui' }); }

  /* ---------- the 2×2 pictures ---------- */
  function draw2d(g, s, ctx, which) {          // which: 'both' | 'row' | 'col'
    const A = s.A, b = s.b, rd = rankDef(A);
    const x = rd ? null : N.solve(A, b);
    const consistent = !rd || offLine(b, A) < 0.12;
    g.ghostGrid(); g.axes();
    const rowAlpha = which === 'row' ? 1 : which === 'col' ? s.rowAlpha * (1 - s.split) : s.rowAlpha;
    if (rowAlpha > 0.02) {
      const merged = rd && consistent && s.showSolSet;
      const Ls = [0, 1].map(i => drawRowLine(g, A[i], b[i], { prog: s.lineDraw[i], alpha: rowAlpha * (merged ? 0.35 : 1), width: 2 }));
      if (rowAlpha > 0.3 && !merged) Ls.forEach((L, i) => { if (L && s.lineDraw[i] > 0.6) { const p = labelSpot(L, x); if (p) g.label('row ' + (i + 1), p, { color: 'hud', size: 12.5, dy: -11 }); } });
      if (merged) {
        const SL = solLine(A, b), L = rowLine(A[0], b[0]) || rowLine(A[1], b[1]);
        if (SL && L) {
          g.lineThrough(L.p0, L.d, { len: 16, color: 'gold', width: 3, alpha: 0.9 * rowAlpha, dashed: true });
          g.point(SL.p, { r: 4.5, color: 'gold' });
          g.vector(SL.p, N.add(SL.p, SL.d), { color: 'null', width: 2.2, label: '[' + fmt(SL.d[0]) + '; ' + fmt(SL.d[1]) + ']', labelAt: 0.55 });
          const sp = labelSpot(L, N.add(SL.p, N.scale(SL.d, 0.5))); if (sp) g.label('every point solves it', sp, { color: 'gold', size: 13, dy: -13 });
        }
      }
      if (x && s.showX && inBox(x, 0.5)) g.point(x, { r: 5 + 2 * (s.xPulse || 0), color: 'gold', label: 'x = [' + fmt(x[0]) + '; ' + fmt(x[1]) + ']', dx: 14, dy: -14 });
      if (s.cross && which !== 'col') { g.marker.cross(s.cross, { size: 24 }); g.label('no solution', s.cross, { color: 'bad', size: 13, dy: 24 }); }
      if (x && s.showX && !s.lockB && which !== 'col') g.handle('x', x, (p, st) => { st.b = N.matVec(st.A, p); if (st.autoW) st.weights = [p[0], p[1]]; }, { color: 'gold', r: 12 });
      if (which === 'row' && s.showStatus) g.hud(!rd ? 'one solution' : consistent ? 'infinitely many solutions' : 'no solution', 10, 36, { color: !rd ? 'gold' : consistent ? 'gold' : 'bad' });
    }
    if (which !== 'row' && s.showCols) {
      const a1 = N.col(A, 0), a2 = N.col(A, 1), ca = s.colsA;
      if (rd) {
        const d = colDir(A);
        g.lineThrough([0, 0], d, { len: 16, color: 'teal', width: 2.2, alpha: 0.85 * ca });
        g.label('column space', N.scale(d, 6.5), { color: 'teal', size: 12.5, dx: 16, dy: 10 });
        const foot = N.scale(d, N.dot(b, d));
        if (N.dist(b, foot) > 0.15) { g.line(b, foot, { color: 'residual', dashed: true, width: 1.4, alpha: ca }); g.point(foot, { r: 3, color: 'teal' }); g.label('out of reach', b, { color: 'bad', size: 12.5, dx: 12, dy: -18 }); }
      }
      g.vector(a1, { color: 'i', label: 'a' + SUB[0], width: 3, alpha: ca });
      g.vector(a2, { color: 'j', label: 'a' + SUB[1], width: 3, alpha: ca });
      g.vector(b, { color: 'b', label: 'b', width: 3, alpha: ca });
      if (s.showChain && ca > 0.5) {
        const end = drawChain(g, [a1, a2], s.weights, ['i', 'j'], s.chainT);
        const done = s.chainT >= 0.999, landed = done && N.dist(end, b) < 0.12;
        if (landed) g.marker.ring(b, { r: 13 });
        if (done) g.label(fmt(s.weights[0]) + '·a' + SUB[0] + ' + ' + fmt(s.weights[1]) + '·a' + SUB[1], end, { color: landed ? 'gold' : 'hud', size: 13, dx: 20, dy: 18 });
        if (s.chainHandles) {
          g.handle('w1', N.scale(a1, s.weights[0]), (p, st) => { const c = N.col(st.A, 0); const nn = N.dot(c, c) || 1; st.weights[0] = Math.round(N.dot(p, c) / nn * 20) / 20; }, { color: 'i', r: 12 });
          g.handle('w2', end, (p, st) => { const c1 = N.col(st.A, 0), c2 = N.col(st.A, 1); const q = N.sub(p, N.scale(c1, st.weights[0])); const nn = N.dot(c2, c2) || 1; st.weights[1] = Math.round(N.dot(q, c2) / nn * 20) / 20; }, { color: 'j', r: 12 });
        }
      }
      if (!s.lockB) g.handle('b', b, (p, st) => { st.b = [p[0], p[1]]; if (st.autoW && !rankDef(st.A)) { const w = N.solve(st.A, st.b); if (w) st.weights = w; } }, { color: 'b' });
      if (s.colHandles) {
        g.handle('a1', a1, (p, st) => { st.A[0][0] = p[0]; st.A[1][0] = p[1]; }, { color: 'i' });
        g.handle('a2', a2, (p, st) => { st.A[0][1] = p[0]; st.A[1][1] = p[1]; }, { color: 'j' });
      }
    }
  }
  function renderSplit(g, s, ctx) {
    const W = g.size.w, H = g.size.h, u = s.split, lw = 0.5 * u;
    const camHalf = fitCam(0.5 * W, H, BOX), camFull = fitCam(W, H, BOX);
    const camR = { center: camHalf.center.slice(), halfH: LA.lerp(camFull.halfH, camHalf.halfH, u) };
    if (lw > 0.002) g.viewport({ x: 0, y: 0, w: lw, h: 1 }, { view: '2d', cam: camHalf }, (g2, r) => { draw2d(g2, s, ctx, 'row'); title(g2, r, 'row picture: lines meet'); });
    g.viewport({ x: lw, y: 0, w: 1 - lw, h: 1 }, { view: '2d', cam: camR }, (g2, r) => { draw2d(g2, s, ctx, 'col'); title(g2, r, 'column picture: columns reach b'); });
  }

  /* ---------- 3D: three planes / three columns ---------- */
  function draw3d(g, s) {
    g.cage({ range: 4, walls: false, alpha: 0.6 });
    g.axes({ length: 4.6 });
    const A = s.A3, b = s.b3;
    if (s.planeA > 0.01) {
      const list = [];
      for (let r = 0; r < 3; r++) { const a = s.planeA * s.planeAlphas[r]; if (a < 0.02) continue; list.push({ point: X3, normal: A[r], extent: 3.2, color: 'hud', alpha: 0.16 * a, stroke: false, _a: a, _r: r }); }
      if (list.length) g.planes(list);
      list.forEach(pl => { const q = LA.geom.planeQuad(pl); g.polyline(q, { color: 'hud', width: 1, closed: true, alpha: 0.5 * pl._a, segmented: true }); g.label('row ' + (pl._r + 1), q[2], { color: 'hud', size: 12.5, alpha: pl._a }); });
      if (s.meetLine > 0.01) { const d = N.normalize(N.cross(A[0], A[1])); const L = 4 * s.meetLine; g.line(N.sub(X3, N.scale(d, L)), N.add(X3, N.scale(d, L)), { color: 'gold', dashed: true, width: 2, alpha: s.planeA }); }
    }
    if (s.showX3 > 0.01) g.point(X3, { r: 6 * Math.min(1, s.showX3), color: 'gold', label: 'x = [1; 2; 3]', dx: 14, dy: -12 });
    if (s.showCols3 > 0.01) {
      const ca = s.showCols3, cols = [0, 1, 2].map(j => N.col(A, j));
      cols.forEach((c, j) => g.vector([0, 0, 0], c, { color: ['i', 'j', 'k'][j], label: 'a' + SUB[j], width: 3, alpha: ca }));
      g.vector([0, 0, 0], b, { color: 'b', label: 'b', width: 3, alpha: ca });
      if (s.chain3T > 0) {
        const end = drawChain(g, cols, s.weights3, ['i', 'j', 'k'], s.chain3T);
        if (s.chain3T >= 0.999 && N.dist(end, b) < 0.12) { g.marker.ring(b, { r: 13 }); g.label('1·a₁ + 2·a₂ + 3·a₃', end, { color: 'gold', size: 13, dx: 22, dy: 18 }); }
      }
    }
  }

  /* ---------- ML tie-in: 12 equations in the (w₀, w₁) plane ---------- */
  function drawML(g, s) {
    g.ghostGrid({ range: 6 }); g.axes({ names: ['w₀', 'w₁'] });
    const n = s.mlx.length;
    for (let i = 0; i < n; i++) {
      const appear = seg(s.mlT, i / n * 0.6, i / n * 0.6 + 0.15);
      if (appear <= 0) continue;
      drawRowLine(g, [1, s.mlx[i]], s.mly[i], { prog: appear, alpha: 0.55, width: 1.3, color: 'hud' });
    }
    const w = s.mlw, blob = seg(s.mlT, 0.7, 1, 'back');
    if (blob > 0.01) { g.point(w, { r: 11 * blob, color: 'gold', alpha: 0.3 }); g.point(w, { r: 4.5 * blob, color: 'gold', label: 'least-squares w', dx: 16, dy: -14 }); }
    const bnd = g.P.bounds(), W = bnd.x1 - bnd.x0, H = bnd.y1 - bnd.y0;
    const ib = { x0: bnd.x1 - W * 0.21, x1: bnd.x1 - W * 0.11, y0: bnd.y1 - H * 0.5, y1: bnd.y1 - H * 0.09 };
    const yb = { x0: bnd.x1 - W * 0.095, x1: bnd.x1 - W * 0.045, y0: ib.y0, y1: ib.y1 };
    g.image(s.mlx.map(x => [1, x]), ib, { cells: true, range: [-2.2, 2.2], ramp: LA.color.heat });
    g.image(s.mly.map(y => [y]), yb, { cells: true, range: [-2.2, 2.2], ramp: LA.color.heat });
    g.label('X', [(ib.x0 + ib.x1) / 2, ib.y1], { anchor: 'above', dy: -4, size: 13, color: 'text' });
    g.label('y', [(yb.x0 + yb.x1) / 2, yb.y1], { anchor: 'above', dy: -4, size: 13, color: 'b' });
    g.label('12 equations, 2 unknowns', [(ib.x0 + yb.x1) / 2, ib.y0], { anchor: 'below', dy: 4, size: 11.5, color: 'muted' });
    g.label('synthetic data', [(ib.x0 + yb.x1) / 2, ib.y0], { anchor: 'below', dy: 20, size: 11, color: 'muted' });
  }

  /* ---------- recap: three mini-cards as horizontal bands ---------- */
  function drawRecap(g) {
    const band = (i, cam, text, fn) => g.viewport({ x: 0, y: i / 3, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => { fn(g2); g2.label(text, [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', bold: true }); });
    band(0, { center: [1.4, 2], halfH: 2.7 }, 'rows meet', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      drawRowLine(g2, A0[0], B0[0], { width: 2 }); drawRowLine(g2, A0[1], B0[1], { width: 2 });
      g2.point([1, 2], { r: 6, color: 'gold', label: 'x', dx: 12, dy: -10 });
    });
    band(1, { center: [2, 3.4], halfH: 4.5 }, 'columns reach', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      g2.vector([2, 1], { color: 'i', label: 'a' + SUB[0], width: 2.6, alpha: 0.5 }); g2.vector([1, 3], { color: 'j', label: 'a' + SUB[1], width: 2.6, alpha: 0.5 });
      drawChain(g2, [[2, 1], [1, 3]], [1, 2], ['i', 'j'], 1);
      g2.point([4, 7], { r: 6, color: 'b', label: 'b', dx: 12, dy: -10 });
    });
    band(2, { center: [0, 0.5], halfH: 2.3 }, 'one · none · many', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      g2.line([-7.1, -0.6], [-4.9, 1.6], { color: 'hud', width: 2 }); g2.line([-7.1, 1.6], [-4.9, -0.6], { color: 'hud', width: 2 });
      g2.point([-6, 0.5], { r: 5.5, color: 'gold' }); g2.label('one', [-6, -0.2], { color: 'gold', dy: 24, size: 12.5 });
      g2.line([-2.1, -1.2], [2.1, 0.9], { color: 'hud', width: 2 }); g2.line([-2.1, 0.2], [2.1, 2.3], { color: 'hud', width: 2 });
      g2.marker.cross([0, 0.55], { size: 22 }); g2.label('none', [0, 0.55], { color: 'bad', dy: 24, size: 12.5 });
      g2.line([3.6, -1.1], [7.4, 0.8], { color: 'hud', width: 5, alpha: 0.6 }); g2.line([3.6, -1.1], [7.4, 0.8], { color: 'gold', width: 2.5, dashed: true });
      g2.label('many', [5.5, -0.15], { color: 'gold', dy: 24, size: 12.5 });
    });
  }

  /** shared "both pictures live" configuration (playground and checks) */
  function base2d(s) {
    s.scene = 'main'; s.view = '2d'; s.cam = CAM2(); s.split = 1; s.showCols = true; s.colsA = 1; s.showX = true; s.lineDraw = [1, 1]; s.rowAlpha = 1;
    s.showChain = true; s.chainT = 1; s.weights = [1, 2]; s.showSolSet = true; s.panel = 'status'; s.panelCols = true; s.showStatus = true; s.cross = null;
    s.chainHandles = false; s.colHandles = false; s.autoW = false; s.lockB = false; s.planeA = 1; s.planeAlphas = [1, 1, 1]; s.meetLine = 1; s.showX3 = 1; s.showCols3 = 1; s.chain3T = 1;
  }

  LA.registerLesson({
    id: 'p3-01-systems-row-column',
    part: 3,
    title: 'Ax = b: row picture vs column picture',
    subtitle: 'Lines meeting, or columns combining to reach b: the same equation seen two ways.',
    minutes: 10,
    prereqs: ['p2-02-matrix-vector', 'p2-07-inverse-solving', 'p2-08-rank-column-null'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a linear layer with 4096 unknowns has 4096 “planes” in ℝ⁴⁰⁹⁶', transfers: 'each equation is one flat sheet, each unknown weights one column, and one / none / many are the only outcomes in any dimension' },
    cantShow: ['four or more equations as planes (only three planes fit in a 3D picture)', 'hyperplanes: in ℝⁿ an equation is an (n−1)-dimensional flat sheet, which has no picture past n = 3'],
    terms: [
      { term: 'system of linear equations', def: 'Several equations sharing the same unknowns, written as one matrix equation Ax = b.' },
      { term: 'row picture', def: 'Each equation drawn as a line (or plane) whose points satisfy it; the solution is where they all meet.' },
      { term: 'column picture', def: 'The right-hand side b drawn as a weighted sum of the columns of A, laid tip to tail.' },
      { term: 'consistent', def: 'A system that has at least one solution.' },
      { term: 'solution set', def: 'All x with Ax = b: a single point, a line, a plane, or nothing at all.' },
    ],
    atlas: ['linear-regression'],
    thumbnail: 3,

    state() {
      const d = LA.rng.linearData(31, 12, { w: [0.8], b: 0.3, noise: 0.35, xRange: [-2, 2] });
      const w = N.lstsq(d.x.map(x => [1, x]), d.y);
      return {
        view: '2d', cam: CAM2(),
        A: LA.clone(A0), b: B0.slice(), weights: [1, 2], A3: LA.clone(A3), b3: B3.slice(), weights3: [1, 2, 3],
        scene: 'main', split: 0, rowAlpha: 1, lineDraw: [1, 1], showX: true, xPulse: 0, showCols: true, colsA: 1, showChain: true, chainT: 1,
        cross: null, showSolSet: false, showStatus: false, chainHandles: false, colHandles: false, autoW: false, lockB: false,
        panel: 'system', panelCols: true, hlRow: null, eqForm: 1,
        planeA: 0, planeAlphas: [1, 1, 1], meetLine: 0, showX3: 0, showCols3: 0, chain3T: 0,
        mlT: 1, mlx: d.x, mly: d.y, mlw: w,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'recap') { drawRecap(g); return; }
      if (s.view === '3d') { draw3d(g, s); return; }
      if (s.scene === 'ml') { drawML(g, s); return; }
      if (s.split > 0.002) { renderSplit(g, s, ctx); return; }
      draw2d(g, s, ctx, 'both');
    },

    math(s, tex0) {
      const tex = safeTex(tex0), P = s.panel;
      if (P === 'eqs' && s.eqForm < 0.5) return tex.lines(
        tex.eq(tex.num(2), tex.sym('x'), tex.op(' + '), tex.sym('y'), tex.op(' = '), tex.num(4, { color: 'b' })),
        tex.eq(tex.sym('x'), tex.op(' + '), tex.num(3), tex.sym('y'), tex.op(' = '), tex.num(7, { color: 'b' })));
      if (s.view === '3d' || P === 'sys3' || P === 'cols3') {
        const parts = [tex.eq(tex.mat(s.A3, { colors: 'cols' }), tex.vec(['x', 'y', 'z']), tex.op('='), tex.vec(s.b3, { color: 'b' }))];
        if (P === 'cols3' || s.showCols3 > 0.5) parts.push(tex.eq(tex.num(1, { color: 'gold' }), tex.vec(N.col(s.A3, 0), { color: 'i' }), tex.op('+'), tex.num(2, { color: 'gold' }), tex.vec(N.col(s.A3, 1), { color: 'j' }), tex.op('+'), tex.num(3, { color: 'gold' }), tex.vec(N.col(s.A3, 2), { color: 'k' }), tex.op('='), tex.vec(s.b3, { color: 'b' })));
        else parts.push(tex.eq(tex.sym('x'), tex.op('='), tex.vec(X3, { color: 'gold' })));
        return tex.block.apply(null, parts);
      }
      if (P === 'ml') return tex.block(tex.eq(tex.sym('X'), tex.sym('w'), tex.op(' ≈ '), tex.sym('y', { color: 'b' })), tex.eq(tex.dim(12, 2), tex.op('·'), tex.dim(2, 1), tex.op(' ≈ '), tex.dim(12, 1)));
      if (P === 'recap') return tex.block(tex.eq(tex.sym('A'), tex.sym('x'), tex.op('='), tex.sym('b', { color: 'b' })));
      const A = s.A, b = s.b;
      const hl = s.hlRow === null || s.hlRow === undefined ? undefined : (i) => i === s.hlRow;
      const parts = [tex.eq(tex.mat(A, { colors: s.panelCols ? 'cols' : undefined, highlight: hl }), tex.vec(['x', 'y']), tex.op('='), tex.vec(b, { color: 'b', highlight: hl }))];
      if (P === 'recipe' || (P === 'status' && s.showChain)) parts.push(tex.eq(tex.num(s.weights[0], { color: 'gold' }), tex.vec(N.col(A, 0), { color: 'i' }), tex.op('+'), tex.num(s.weights[1], { color: 'gold' }), tex.vec(N.col(A, 1), { color: 'j' }), tex.op('='), tex.vec(N.matVec(A, s.weights), { color: 'b' })));
      if (P === 'status') {
        const rd = rankDef(A);
        if (!rd) { const x = N.solve(A, b) || [0, 0]; parts.push(tex.eq(tex.sym('x'), tex.op('='), tex.inv(tex.sym('A')), tex.sym('b', { color: 'b' }), tex.op('='), tex.vec(x, { color: 'gold' }))); }
        else if (offLine(b, A) >= 0.12) parts.push(tex.eq(tex.text('det A = 0, '), tex.sym('b', { color: 'b' }), tex.op(' ∉ '), tex.text('column space', { color: 'teal' }), tex.op(' → '), tex.text('no solution', { color: 'bad' })));
        else { const SL = solLine(A, b); if (SL) parts.push(tex.eq(tex.sym('x'), tex.op('='), tex.vec(SL.p, { color: 'gold' }), tex.op('+'), tex.sym('t'), tex.op('·'), tex.vec(SL.d, { color: 'null' }), tex.text(' infinitely many', { color: 'gold' }))); }
      }
      return tex.block.apply(null, parts);
    },

    steps: [
      { title: 'Two equations, two unknowns', duration: 3,
        narration: 'Here are two equations sharing two unknowns: 2x + y = 4 and x + 3y = 7. Together they form a <b>system of linear equations</b>. Stacked, they are one matrix equation <b>Ax = b</b>: the rows of A hold the coefficients, the orange ' + bb + ' holds the right-hand sides. Same facts, written once.',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = CAM2(); s.split = 0; s.A = LA.clone(A0); s.b = B0.slice(); s.showCols = false; s.colsA = 0; s.showX = false; s.xPulse = 0; s.lineDraw = [0, 0]; s.rowAlpha = 1; s.showChain = false; s.cross = null; s.showSolSet = false; s.showStatus = false; s.panel = 'eqs'; s.panelCols = false; s.hlRow = null; s.eqForm = 0; },
        at(s, t) { s.eqForm = t; s.hlRow = t < 0.5 ? null : t < 0.75 ? 0 : t < 0.95 ? 1 : null; } },

      { title: 'The row picture: each equation is a line', duration: 4,
        narration: 'Read A by <i>rows</i>: this is the <b>row picture</b>. Every point (x, y) with 2x + y = 4 lies on one straight line; every point with x + 3y = 7 lies on another. A solution must satisfy both, so it must sit on both lines: the one place they cross, the gold point.',
        setup(s) { s.panel = 'system'; s.eqForm = 1; s.hlRow = null; },
        at(s, t) { s.lineDraw = [seg(t, 0, 0.4), seg(t, 0.4, 0.8)]; s.hlRow = t < 0.4 ? 0 : t < 0.8 ? 1 : null; s.showX = t > 0.8; s.xPulse = seg.pulse(t, 0.8, 1); } },

      { title: 'The column picture: reach b with the columns', duration: 4.5,
        narration: 'Now read A by <i>columns</i>: the <b>column picture</b>. Column 1, the green <span class="c-i">a₁</span> = [2; 1], is where î lands; column 2, the red <span class="c-j">a₂</span> = [1; 3], is where ĵ lands. Ax = b asks: how much of each column, tip to tail, reaches the orange ' + bb + '? One of the first and two of the second.',
        setup(s) { s.lineDraw = [1, 1]; s.showX = true; s.xPulse = 0; s.hlRow = null; s.showCols = true; s.colsA = 0; s.weights = [1, 2]; s.showChain = true; s.chainT = 0; s.panel = 'recipe'; s.panelCols = true; },
        at(s, t) { s.rowAlpha = 1 - 0.75 * seg(t, 0, 0.3); s.colsA = seg(t, 0, 0.3); s.chainT = seg(t, 0.3, 0.9); } },

      { title: 'Same answer, two pictures', duration: 3, interactive: true,
        narration: 'The crossing point of the lines is [1; 2]. The weights on the columns are 1 and 2. They are the <i>same numbers</i>, because both pictures are the same equation. Drag the orange ' + bb + ' on the right: the gold crossing moves and the recipe changes together.',
        setup(s) { s.rowAlpha = 0.25; s.colsA = 1; s.chainT = 1; s.panel = 'status'; s.split = 0; s.showStatus = true; s.autoW = true; },
        at(s, t) { s.split = seg(t, 0.05, 0.6); } },

      { title: 'No solution: parallel lines, b off the column line', duration: 4, misconception: true,
        narration: '“If there is no solution, I made a mistake.” ✕ Some systems simply have none. Take A = [[1, 2], [2, 4]]: the rows give <i>parallel</i> lines that never meet, and both columns point along one teal line, so the orange ' + bb + ' = [3; 1] is out of reach. The system is not <b>consistent</b>.',
        setup(s) { s.split = 1; s.autoW = false; s._A = LA.clone(s.A); s._b = s.b.slice(); s.showChain = false; s.cross = null; },
        at(s, t) { const u = seg(t, 0, 0.5); s.A = lerpM(s._A, AF, u); s.b = lerpV(s._b, B_NONE, u); s.cross = t > 0.6 ? CROSS_NONE.slice() : null; } },

      { title: 'Infinitely many: coincident lines, b on the column line', duration: 4,
        narration: 'Keep that flattened A but move the orange ' + bb + ' onto the teal column line, to [2; 4]. Now both rows describe the <i>same</i> line, and every point on it solves the system: the <b>solution set</b> is a whole line. On the column side, many recipes reach b: 2a₁ + 0a₂, 0a₁ + 1a₂, and everything between.',
        setup(s) { s.A = LA.clone(AF); s.b = B_NONE.slice(); s._b = s.b.slice(); s.cross = null; s.showSolSet = false; s.showChain = false; },
        at(s, t) {
          s.b = lerpV(s._b, B_MANY, seg(t, 0, 0.4)); s.showSolSet = t > 0.35;
          if (t >= 0.4) { const u = (t - 0.4) / 0.6 * 3; const idx = Math.min(2, Math.floor(u)); s.showChain = true; s.weights = RECIPES[idx].slice(); s.chainT = seg(u - idx, 0, 0.8); }
          else s.showChain = false;
        } },

      { title: 'Three planes in 3D', duration: 5,
        narration: 'Three unknowns: each equation is now a flat <i>plane</i> in 3D. Two planes meet in a line, the gold dashed one; the third plane cuts that line at a single point. Here x + y = 3, y + z = 5 and x + z = 4 meet at the gold [1; 2; 3]. Drag to orbit.',
        setup(s) { s.view = '3d'; s.cam = CAM3(); s.planeA = 1; s.planeAlphas = [0, 0, 0]; s.meetLine = 0; s.showX3 = 0; s.showCols3 = 0; s.chain3T = 0; s.weights3 = [1, 2, 3]; s.panel = 'sys3'; },
        at(s, t) { s.planeAlphas = [seg(t, 0, 0.25), seg(t, 0.25, 0.5), seg(t, 0.5, 0.8)]; s.meetLine = seg(t, 0.3, 0.5); s.showX3 = seg(t, 0.8, 1, 'back'); LA.cam.tween(s, { yaw: 0.44 }, { yaw: 0.78 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Columns in 3D reach b', duration: 4,
        narration: 'Same system, column view: the three columns, green <span class="c-i">a₁</span> = [1; 0; 1], red <span class="c-j">a₂</span> = [1; 1; 0] and blue <span class="c-k">a₃</span> = [0; 1; 1], are stacked with weights 1, 2, 3 and arrive exactly at the orange ' + bb + ' = [3; 5; 4]. One picture is about where sheets cross; the other about reaching a point. Same x.',
        setup(s) { s.planeAlphas = [1, 1, 1]; s.meetLine = 1; s.showX3 = 1; s.panel = 'cols3'; },
        at(s, t) { s.planeA = 1 - seg(t, 0, 0.2); s.showX3 = 1 - seg(t, 0, 0.2); s.showCols3 = seg(t, 0.05, 0.3); s.chain3T = seg(t, 0.25, 0.95); LA.cam.tween(s, { yaw: 0.78, pitch: 0.42, dist: 14.5, target: [1, 2, 2.4] }, { yaw: -0.5, pitch: 0.32, dist: 17, target: [1.5, 2.6, 2] }, seg(t, 0, 0.6)); } },

      { title: 'Why ML cares: every “solve” is one of these', duration: 4, mlTie: true,
        narration: 'Training a linear model means finding weights w with Xw ≈ y. Every data row is one equation, one line in weight space. With far more rows than weights the lines rarely all cross: <i>no solution</i> is the normal case in ML. The least-squares capstone turns that into a best answer: the shadow of y on the column space.',
        setup(s) { s.view = '2d'; s.scene = 'ml'; s.cam = { center: [s.mlw[0] + 0.15, s.mlw[1] + 0.1], halfH: 2.3 }; s.panel = 'ml'; s.mlT = 0; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Rows: each equation is a line or plane; the solution is where they all meet. Columns: Ax = b asks for the recipe of columns that reaches the orange ' + bb + '. One answer, none, or a whole line of them — and the two pictures always agree.',
        cards: ['<b>Rows meet.</b> Each equation is a line (or plane); the solution is the one place they all cross.', '<b>Columns reach.</b> Ax = b asks which recipe of the columns, tip to tail, lands on b.', '<b>One, none or many.</b> Parallel rows: none (✕). Coincident rows with b on the column line: a whole line of answers.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.cam = CAM2(); s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit <b>A</b> or drag the orange <span class="c-b">b</span> (or the gold crossing); slide x₁, x₂ to build the column recipe. Presets show the three outcomes; “3 planes” switches to the 3×3 system in 3D (drag to orbit).</p>',
      setup(s) { base2d(s); s.colHandles = true; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'b', label: 'b', colors: 'b', step: 0.1 },
        { type: 'slider', bind: 'weights[0]', min: -3, max: 3, step: 0.1, label: 'x₁ (copies of a₁)', color: 'i' },
        { type: 'slider', bind: 'weights[1]', min: -3, max: 3, step: 0.1, label: 'x₂ (copies of a₂)', color: 'j' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Unique', apply(s) { s.view = '2d'; s.cam = CAM2(); s.A = LA.clone(A0); s.b = B0.slice(); s.weights = [1, 2]; } },
          { label: 'No solution', apply(s) { s.view = '2d'; s.cam = CAM2(); s.A = LA.clone(AF); s.b = B_NONE.slice(); s.weights = [1, 1]; } },
          { label: 'Infinitely many', apply(s) { s.view = '2d'; s.cam = CAM2(); s.A = LA.clone(AF); s.b = B_MANY.slice(); s.weights = [2, 0]; } },
          { label: '3 planes (3D)', apply(s) { s.view = '3d'; s.cam = CAM3(); } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D', apply(s) { s.view = '2d'; s.cam = CAM2(); } },
          { label: '3D', apply(s) { s.view = '3d'; s.cam = CAM3(); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the two recipe handles (the tip of the green stack and the tip of the red stack) until the column chain lands on the orange ' + bb + '.',
        setup(s) { base2d(s); s.A = LA.clone(A0); s.b = B0.slice(); s.weights = [0.5, 0.5]; s.chainHandles = true; s.lockB = true; s.showX = false; s.panel = 'recipe'; },
        judge(s) { const r = LA.judge.near(N.matVec(s.A, s.weights), s.b, LA.TOL.pos); return { ok: r.ok, progress: r.progress }; },
        hint: 'The chain should end on the orange tip. Which column is still short? Each handle slides along its own column direction.',
        solution(s) { s.weights = [1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Drag the orange ' + bb + ' so this flattened system (both columns on one teal line) has <i>no</i> solution.',
        setup(s) { base2d(s); s.A = LA.clone(AF); s.b = B_MANY.slice(); s.weights = [2, 0]; s.showChain = false; },
        judge(s) { const d = offLine(s.b, s.A); return { ok: d > 0.3, progress: Math.min(1, d / 0.3) }; },
        hint: bb + ' must leave the teal column line; then the two row-lines separate and never meet.',
        solution(s) { s.b = B_NONE.slice(); } },
      { kind: 'entry',
        prompt: 'Keep row 1 = [1, 2] and ' + bb + ' = [2; 4]. Type the second row of <b>A</b> so that its line coincides with the first one (infinitely many solutions).',
        bind: 'A', answer: [[1, 2], [2, 4]], tol: LA.TOL.entry, label: 'A', colors: 'text',
        setup(s) { base2d(s); s.A = [[1, 2], [1, 0]]; s.b = B_MANY.slice(); s.showChain = false; } },
    ],
  });
})();
