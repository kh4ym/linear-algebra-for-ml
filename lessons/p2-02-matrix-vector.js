/* p2-02 — Matrix × vector: column view and row view. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, fmt = LA.fmt;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[2, -1], [1, 1]], SING = [[1, 2], [2, 4]];
  const W3 = [[1, 0.5], [-0.5, 1], [0.8, -0.8]], X3 = [2, 1];       // the ML step: three neurons, two inputs
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', vv = '<span class="c-v">v</span>', xx = '<span class="c-v">x</span>';
  const SUB = ['₁', '₂', '₃'];
  const isNull = x => x === null || x === undefined;
  /* Live numbers for the math panel. Plain colored spans (not tex.num): the engine's in-place panel update pairs
     tex.num values with the wrong cells when a tex.mat precedes them, so these numbers are part of the skeleton and
     the panel simply rebuilds when they change. */
  const tn = (tex, x, color) => '<span class="' + tex.colorClass(color) + '">' + (typeof x === 'number' ? fmt(x) : x) + '</span>';
  const tnp = (tex, x, color) => (x < 0 ? tex.op('(') + tn(tex, x, color) + tex.op(')') : tn(tex, x, color));
  const tdot = (tex, a, b, cA, cB, cR) => tex.eq(a.map((x, i) => tnp(tex, x, cA) + tex.op('·') + tnp(tex, b[i], cB)).join(tex.op(' + ')), tex.op(' = '), tn(tex, N.dot(a, b), cR));

  function lattice() { const pts = []; for (let x = -3; x <= 3; x++) for (let y = -3; y <= 3; y++) pts.push([x, y]); return pts; }
  /** camera for a viewport that is `frac` of the canvas wide: keep halfW world units visible horizontally */
  function fitCam(g, frac, c) { const aspect = (g.size.w * frac) / g.size.h; return { center: c.center.slice(), halfH: Math.max(c.halfH, c.halfW / aspect) }; }
  /** screen-px line (2D only) */
  function sline(g, a, b, o) { g.line(g.P.toWorld(a[0], a[1]), g.P.toWorld(b[0], b[1]), o); }
  function bracket(g, x, y, h, dir, color) {
    const k = 6 * dir, st = { color, width: 1.5 };
    sline(g, [x + k, y], [x, y], st); sline(g, [x, y], [x, y + h], st); sline(g, [x, y + h], [x + k, y + h], st);
  }
  /** bracketed matrix at screen px; cells[i][j] = {t, c, hl, a}; returns its width */
  function hudMat(g, cells, x, y, o) {
    o = o || {}; const cw = o.cw || 46, ch = o.ch || 28, size = o.size || 15;
    const rows = cells.length, cols = cells[0].length, w = cols * cw, h = rows * ch, col = o.color || 'hud';
    bracket(g, x, y, h, 1, col); bracket(g, x + w + 8, y, h, -1, col);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const c = cells[i][j];
      g.label(c.t, [x + 4 + cw * (j + 0.5), y + ch * (i + 0.5)], { screen: true, color: c.c || col, size: c.hl ? size + 3 : size, bold: !!c.hl, font: 'math', alpha: c.a, bg: false });
    }
    return w + 8;
  }

  /* ---------------- the column view: columns mixed tip to tail ---------------- */
  function drawRecipe(g, c1, c2, v, r, hl) {
    const parts = [{ c: c1, x: v[0], color: 'i', dim: hl === 1 }, { c: c2, x: v[1], color: 'j', dim: hl === 0 }];
    let o = [0, 0];
    parts.forEach((p, k) => {
      const u = seg(r, k * 0.5, k * 0.5 + 0.5, 'linear');
      const n = Number.isInteger(p.x) && Math.abs(p.x) >= 1 && Math.abs(p.x) <= 4 ? Math.abs(p.x) : 1;
      const unit = N.scale(p.c, p.x / n), end = N.add(o, N.scale(p.c, p.x));
      const mid = Math.floor((n - 1) / 2);
      for (let i = 0; i < n; i++) {
        const f = LA.clamp01(u * n - i); if (f <= 0) break;
        const a = N.add(o, N.scale(unit, i)), b = N.add(a, N.scale(unit, f));
        g.vector(a, b, { color: p.color, width: 2.2, head: 9, alpha: p.dim ? 0.25 : 0.75, label: i === mid && f >= 1 && Math.abs(p.x) > 1e-9 ? fmt(p.x) + '·col' + SUB[k] : undefined, labelAt: 0.5, labelSize: 12.5 });
      }
      o = end;
    });
  }
  function drawWrong(g, s, Av) {
    const M = s.M, v = s.v, w = s.wrong, P = [M[0][0] * v[0], M[1][1] * v[1]];
    const u = seg(w, 0, 0.45), p = lerpV([0, 0], P, u);
    g.line([0, 0], p, { color: 'muted', dashed: true, width: 1.5 });
    g.point(p, { r: 6, color: 'muted', label: '[' + fmt(P[0]) + '; ' + fmt(P[1]) + ']?', dx: 0, dy: 20 });
    if (w > 0.6) { g.marker.cross(P, { size: 24, dx: 28 }); g.label('matching entries?', P, { color: 'bad', anchor: 'above', dy: -16, size: 13 }); }
    if (w > 0.75) g.dim(P, Av, { label: 'not the same point', color: 'bad', offset: -18 });
    const pulse = seg.pulse(w, 0.5, 1); if (pulse > 0) g.marker.ring(Av, { r: 9 + 10 * pulse, color: 'v' });
  }
  function renderColumn(g, s, ctx) {
    const M = s.M, v = s.v;
    const hl = !isNull(s.colHighlight) ? s.colHighlight : (ctx.hot === 'i' ? 0 : ctx.hot === 'j' ? 1 : null);
    if (s.showGhost) g.ghostGrid();
    const Mg = s.dots ? lerpM(I2(), M, s.morph) : M;
    g.grid(Mg, { alpha: s.dots ? 0.85 : 0.35 });
    const c1 = N.col(M, 0), c2 = N.col(M, 1), Av = N.matVec(M, v);
    if (s.dots) g.points(s.lattice.map(p => lerpV(p, N.matVec(M, p), s.morph)), { r: 3.5, color: 'teal' });
    if (s.showV) {
      if (s.showRecipe && s.recipeT > 0) drawRecipe(g, c1, c2, v, s.recipeT, hl);
      const a = s.showRecipe ? seg(s.recipeT, 0.85, 1) : 1;
      if (a > 0) {
        g.vector(Av, { color: 'v', label: 'Av', width: 3, alpha: a });
        if (s.pulse > 0) g.marker.ring(Av, { r: 9 + 10 * s.pulse, color: 'v' });
      }
    }
    g.vector(c1, { color: 'i', label: 'i', hat: true, width: 3, alpha: hl === 1 ? 0.3 : 1 });
    g.vector(c2, { color: 'j', label: 'j', hat: true, width: 3, alpha: hl === 0 ? 0.3 : 1 });
    if (s.target) { g.marker.ring(s.target, { r: 9, color: 'hud' }); g.label('target', s.target, { color: 'muted', anchor: 'below', dy: 12, size: 12 }); }
    if (s.wrong > 0) drawWrong(g, s, Av);
    g.handle('i', c1, (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; }, { color: 'i' });
    g.handle('j', c2, (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
    if (s.showV && ctx.mode === 'play') g.handle('av', Av, (p, st) => { const inv = N.inv(st.M); if (inv) st.v = N.matVec(inv, p); }, { color: 'v' });
  }

  /* ---------------- the row view: rows as gray arrows, shadows, bars ---------------- */
  function renderRow(g, s, ctx, rect) {
    const M = s.M, v = s.v, rows = [M[0].slice(), M[1].slice()];
    if (s.showGhost) g.ghostGrid();
    if (s.rightDim) { g.vector(v, { color: 'v', label: 'v', width: 3, alpha: 0.5 }); g.hud('row view — next', rect.x + 12, rect.y + 18, { color: 'muted' }); return; }
    const d = rows.map(r => N.dot(r, v));
    if (s.showOutLine) { g.lineThrough([0, 0], [1, 2], { color: 'teal', width: 2, alpha: 0.8 }); g.label('every output lands on y = 2x', [-0.6, -1.2], { color: 'teal', size: 12, anchor: 'right', dx: -10 }); }
    rows.forEach((r, i) => { const dimmed = !isNull(s.rowHighlight) && s.rowHighlight !== i; g.vector(r, { color: 'residual', label: 'r' + SUB[i], width: 2.4, alpha: dimmed ? 0.3 : 0.95, labelAt: i === 0 ? 0.6 : 1 }); });
    rows.forEach((r, i) => {
      const p = s.shadow[i]; if (!(p > 0)) return;
      const rn2 = N.dot(r, r); if (rn2 < 1e-9) return;
      const foot = N.scale(r, d[i] / rn2);
      g.line(v, lerpV(v, foot, seg(p, 0, 0.6)), { color: 'residual', dashed: true, width: 1.3 });
      if (p > 0.6) { g.line([0, 0], foot, { color: 'mag', width: 3.5, alpha: 0.85 }); g.point(foot, { r: 4, color: 'mag' }); }
    });
    g.vector(v, { color: 'v', label: 'v', width: 3 });
    const show = s.shadow.map(p => p >= 0.99);
    if (show.some(Boolean)) {
      const bw = Math.min(130, rect.w * 0.42);
      g.bars(d.map((x, i) => show[i] ? x : 0), { screen: true, x: rect.x + rect.w - bw - 14, y: rect.y + 34, w: bw, h: 58, colors: ['i', 'j'], labels: ['r₁·v', 'r₂·v'], values: true, max: 1.4 * Math.max(1, ...d.map(Math.abs)), title: 'entries of Av' });
    }
    if (s.assemble > 0) {
      const out = d.slice(), a = s.assemble;
      if (N.norm(out) < 0.05) g.point(out, { r: 6, color: 'v', label: 'Av = 0', alpha: a, dx: 12, dy: -14 });
      else g.vector(out, { color: 'v', label: 'Av', width: 3, alpha: a, components: a > 0.6 ? 'labels' : true });
      if (s.pulse > 0) g.marker.ring(out, { r: 9 + 10 * s.pulse, color: 'v' });
    }
    if (!s.lockV) g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'v' });
  }

  /* ---------------- the symbolic product across the top ---------------- */
  function renderSymbolic(g, s) {
    const col = s.sym.col, row = s.sym.row;
    const cx = g.size.w / 2, y = 20;
    const cellA = (t, j, i) => ({ t, c: j === 0 ? 'i' : 'j', hl: (col === j) || (row === i), a: (!isNull(col) && col !== j) || (!isNull(row) && row !== i) ? 0.35 : 1 });
    const A = [[cellA('a', 0, 0), cellA('b', 1, 0)], [cellA('c', 0, 1), cellA('d', 1, 1)]];
    const V = [[{ t: 'x', c: 'v', hl: col === 0, a: col === 1 ? 0.35 : 1 }], [{ t: 'y', c: 'v', hl: col === 1, a: col === 0 ? 0.35 : 1 }]];
    const R = [[{ t: 'ax + by', hl: row === 0, a: row === 1 ? 0.35 : 1 }], [{ t: 'cx + dy', hl: row === 1, a: row === 0 ? 0.35 : 1 }]];
    const total = 96 + 8 + 52 + 36 + 108; let x = cx - total / 2;
    x += hudMat(g, A, x, y, { cw: 44 }) + 8;
    x += hudMat(g, V, x, y, { cw: 44 }) + 4;
    g.label('=', [x + 14, y + 28], { screen: true, size: 18, color: 'hud', bg: false }); x += 32;
    hudMat(g, R, x, y, { cw: 100 });
    const cap = !isNull(col) ? (col === 0 ? 'x scales the green column' : 'y scales the red column') : !isNull(row) ? (row === 0 ? 'row 1 · v gives the top entry' : 'row 2 · v gives the bottom entry') : 'same formula, two highlights';
    g.hud(cap, cx, y + 76, { anchor: 'center', color: 'hud', size: 13 });
  }

  /* ---------------- shape tiles ---------------- */
  /** a shape tile centred at screen px (x, y); laid out in px so it looks the same in any viewport */
  function tile(g, x, y, parts, color, k) {
    const w = 108 * k, h = 56 * k, size = Math.max(11, 19 * k), W = (a, b) => g.P.toWorld(a, b);
    g.polygon([W(x - w / 2, y - h / 2), W(x + w / 2, y - h / 2), W(x + w / 2, y + h / 2), W(x - w / 2, y + h / 2)], { fill: color, alpha: 0.16, stroke: color, width: 1.4 });
    parts.forEach((p, i) => g.label(p.t, [x + (i - 1) * 28 * k, y], { screen: true, color: p.c || color, size, bold: true, font: 'ui', alpha: p.a, bg: false }));
  }
  function renderTiles(g, u, rect, k) {
    k = k || 1;
    const TW = 108 * k, cx = rect.x + rect.w / 2, pc = cx - 60 * k;
    const gapY = k >= 1 ? 150 : 72, cy = rect.y + rect.h / 2 - (k >= 1 ? 14 : -6);
    const row = (dy, L, R, res, t0, t1, mismatch) => {
      const y = cy + dy * gapY, a = seg(u, t0, t1, 'inOut');
      let xl = pc - TW / 2 - LA.lerp(90, 0, a) * k, xr = pc + TW / 2 + LA.lerp(90, 0, a) * k;
      if (mismatch) { const back = seg(u, t1, t1 + 0.1, 'out'); xl -= (10 + 50 * back) * k; xr += (10 + 50 * back) * k; }
      const innerA = mismatch ? 1 : 1 - seg(u, t1, t1 + 0.08, 'linear');
      const badC = mismatch && u > t1 ? 'bad' : undefined;
      tile(g, xl, y, [{ t: L[0] }, { t: '×' }, { t: L[1], a: innerA, c: badC }], 'teal', k);
      tile(g, xr, y, [{ t: R[0], a: innerA, c: badC }, { t: '×' }, { t: R[1] }], 'v', k);
      if (!mismatch && u > t1 + 0.08) { g.label('→', [pc + TW + 22 * k, y], { screen: true, size: 20 * k + 4, color: 'hud', bg: false }); tile(g, pc + 1.5 * TW + 44 * k, y, [{ t: res[0] }, { t: '×' }, { t: res[1] }], 'ok', k); }
      if (mismatch && u > t1 + 0.02) { g.marker.cross([pc, y], { screen: true, size: 26 * k }); g.label(L[1] + ' ≠ ' + R[0] + ': no product', [pc, y + 30 * k + 12], { screen: true, color: 'bad', size: Math.max(11, 13 * k) }); }
    };
    row(-1, ['2', '2'], ['2', '1'], ['2', '1'], 0.02, 0.3, false);
    row(0, ['2', '3'], ['3', '1'], ['2', '1'], 0.45, 0.68, false);
    row(1, ['2', '2'], ['3', '1'], null, 0.8, 0.9, true);
    if (k >= 1) g.hud('matrix (rows × columns) · vector (rows × 1): the inner numbers must match', cx, rect.y + rect.h - 24, { anchor: 'center', color: 'muted' });
  }

  /* ---------------- the ML step: a layer is many rows at once ---------------- */
  function renderLayer(g, s) {
    const u = s.layerT, h = W3.map(r => N.dot(r, X3)), ramp = t => LA.color.diverge(2 * t - 1);
    g.split((g2, rect) => {
      g2.ghostGrid();
      W3.forEach((r, i) => g2.vector(r, { color: 'residual', label: 'w' + SUB[i], width: 2.4, alpha: 0.95 }));
      W3.forEach((r, i) => {
        const p = seg(u, i * 0.25, i * 0.25 + 0.5); if (p <= 0) return;
        const foot = N.scale(r, h[i] / N.dot(r, r));
        g2.line(X3, lerpV(X3, foot, seg(p, 0, 0.6)), { color: 'residual', dashed: true, width: 1.3 });
        if (p > 0.6) { g2.line([0, 0], foot, { color: 'mag', width: 3.5, alpha: 0.85 }); g2.point(foot, { r: 4, color: 'mag' }); }
      });
      g2.vector(X3, { color: 'v', label: 'x', width: 3 });
      const bw = Math.min(150, rect.w * 0.5);
      g2.bars(h.map((x, i) => (seg(u, i * 0.25, i * 0.25 + 0.5) >= 1 ? x : 0)), { screen: true, x: rect.x + rect.w - bw - 12, y: rect.y + 36, w: bw, h: 56, color: 'gold', labels: ['w₁·x', 'w₂·x', 'w₃·x'], values: true, max: 3.6, title: 'neuron scores h' });
      g2.hud('row view: one neuron per row', rect.x + 12, rect.y + rect.h - 20, { color: 'muted' });
    }, (g2, rect) => {
      g2.hud('p2-12: a neural layer · p6-01: batches', rect.x + rect.w / 2, rect.y + 18, { anchor: 'center', color: 'teal' });
      const a = s.heatA;
      if (a <= 0) { g2.hud('all rows at once — next', rect.x + 12, rect.y + rect.h - 20, { color: 'muted' }); return; }
      const cellLabels = (G, b) => { const n = G.length, m = G[0].length; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) g2.label(fmt(G[i][j]), [b.x0 + (b.x1 - b.x0) * (j + 0.5) / m, b.y1 - (b.y1 - b.y0) * (i + 0.5) / n], { size: 13, color: 'text', bg: false, alpha: a }); };
      const bW = { x0: -2.3, x1: -0.9, y0: -1.05, y1: 1.05 }, bX = { x0: -0.2, x1: 0.5, y0: -0.7, y1: 0.7 }, bH = { x0: 1.25, x1: 1.95, y0: -1.05, y1: 1.05 };
      const xs = X3.map(x => [x]), hs = h.map(x => [x]);
      g2.image(W3, bW, { ramp, range: [-2.5, 2.5], cells: true, alpha: a }); cellLabels(W3, bW);
      g2.image(xs, bX, { ramp, range: [-2.5, 2.5], cells: true, alpha: a }); cellLabels(xs, bX);
      g2.image(hs, bH, { ramp, range: [-2.5, 2.5], cells: true, alpha: a }); cellLabels(hs, bH);
      g2.label('×', [-0.55, 0], { size: 18, color: 'hud', bg: false, alpha: a }); g2.label('=', [0.88, 0], { size: 18, color: 'hud', bg: false, alpha: a });
      g2.label('W  3×2', [-1.6, 1.4], { color: 'hud', size: 13, alpha: a }); g2.label('x', [0.15, 1.05], { color: 'v', size: 15, alpha: a }); g2.label('h', [1.6, 1.4], { color: 'gold', size: 15, alpha: a });
      g2.hud('all three rows at once: W (heatmap) × x = h', rect.x + 12, rect.y + rect.h - 20, { color: 'muted' });
    }, { left: { view: '2d', cam: fitCam(g, 0.5, { center: [0.9, 0.4], halfH: 3, halfW: 2.5 }) }, right: { view: '2d', cam: fitCam(g, 0.5, { center: [-0.2, 0], halfH: 2.4, halfW: 2.6 }) } });
  }

  function renderRecap(g, s, ctx) {
    const s1 = Object.assign({}, s, { showRecipe: true, recipeT: 1, showV: true, dots: false, wrong: 0, target: null, colHighlight: null, pulse: 0, M: LA.clone(A0), v: [2, 1] });
    const s2 = Object.assign({}, s1, { shadow: [1, 1], assemble: 1, rowHighlight: null, rightDim: false, showOutLine: false });
    const H = 0.6, fit = (c) => { const aspect = (g.size.w * 0.5) / (g.size.h * H); return { center: c.center, halfH: Math.max(c.halfH, c.halfW / aspect) }; };
    g.viewport({ x: 0, y: 0, w: 0.5, h: H }, { view: '2d', cam: fit({ center: [1.6, 1.3], halfH: 2.3, halfW: 3.1 }) }, (g2, rect) => { renderColumn(g2, s1, ctx); g2.hud('1 · column view', rect.x + 10, rect.y + 16, { color: 'muted' }); });
    g.viewport({ x: 0.5, y: 0, w: 0.5, h: H }, { view: '2d', cam: fit({ center: [1.3, 2.0], halfH: 3.2, halfW: 2.8 }) }, (g2, rect) => { renderRow(g2, s2, ctx, rect); g2.hud('2 · row view', rect.x + 10, rect.y + 16, { color: 'muted' }); });
    g.viewport({ x: 0, y: H, w: 1, h: 1 - H }, { view: '2d', cam: LA.cam.defaults2d() }, (g2, rect) => { renderTiles(g2, 1, rect, 0.55); g2.hud('3 · shapes chain', rect.x + 10, rect.y + 16, { color: 'muted' }); });
  }

  LA.registerLesson({
    id: 'p2-02-matrix-vector',
    part: 2,
    title: 'Matrix × vector: column view and row view',
    subtitle: 'Mix the columns, or dot each row: two faces of the same product.',
    minutes: 8,
    prereqs: ['p2-01-linear-transformations', 'p1-06-dot-product'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'h = Wx with W 128×784: 784 columns to mix, or 128 dot products to take', transfers: 'both views are exactly the same arithmetic in any shape; code uses whichever is faster, the picture is the same' },
    cantShow: ['128 rows casting 128 shadows at once (we draw two or three and say “one per output entry”)'],
    terms: [
      { term: 'column view', def: 'A v is x times column 1 plus y times column 2: the columns are mixed by the entries of v.' },
      { term: 'row view', def: 'Each entry of A v is the dot product of one row of A with v.' },
      { term: 'row', def: 'One horizontal strip of a matrix; read as a vector, its shadow of v gives one output entry.' },
      { term: 'shape rule', def: 'An m×n matrix times an n×1 vector gives an m×1 vector: the inner numbers must match.' },
      { term: 'elementwise product', def: 'Multiplying matching entries one by one; this is not what matrix-times-vector does.' },
    ],
    atlas: ['mlp', 'tensors'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: [1.2, 1.1] }),
        camL: { center: [1.6, 1.3], halfH: 3.4, halfW: 3.1 }, cam2: { center: [1.2, 1.3], halfH: 3.4, halfW: 2.7 },
        M: LA.clone(A0), v: [2, 1], viewMode: 'split',
        showGhost: true, showV: true, showRecipe: true, recipeT: 1, colHighlight: null, rowHighlight: null,
        shadow: [0, 0], assemble: 0, pulse: 0, rightDim: false,
        sym: null, wrong: 0, tiles: null, dots: false, morph: 1, lattice: lattice(),
        singular: false, showOutLine: false, layerT: 0, heatA: 0, target: null, lockV: false,
      };
    },

    render(g, s, ctx) {
      const mode = s.viewMode;
      if (mode === 'recap') { renderRecap(g, s, ctx); return; }
      if (mode === 'layer') { renderLayer(g, s); return; }
      if (s.tiles) { renderTiles(g, s.tiles.phase, { x: 0, y: 0, w: g.size.w, h: g.size.h }, 1); return; }
      if (mode === 'split') {
        g.split(g2 => renderColumn(g2, s, ctx), (g2, rect) => renderRow(g2, s, ctx, rect), { left: { view: '2d', cam: fitCam(g, 0.5, s.camL) }, right: { view: '2d', cam: fitCam(g, 0.5, s.cam2) } });
        return;
      }
      if (mode === 'row') renderRow(g, s, ctx, { x: 0, y: 0, w: g.size.w, h: g.size.h });
      else renderColumn(g, s, ctx);
      if (s.sym) renderSymbolic(g, s);
    },

    math(s, tex) {
      const M = s.M, v = s.v, Av = N.matVec(M, v);
      const V = tex.sym('v', { color: 'v' });
      if (s.tiles) return tex.block(
        tex.eq(tex.paren(tex.dim(2, 2)), tex.paren(tex.dim(2, 1)), tex.op('→'), tex.dim(2, 1)),
        tex.eq(tex.paren(tex.dim(2, 3)), tex.paren(tex.dim(3, 1)), tex.op('→'), tex.dim(2, 1)),
        tex.eq(tex.paren(tex.dim(2, 2)), tex.paren(tex.dim(3, 1)), tex.op('→'), tex.text('✕ no product', { color: 'bad' })));
      if (s.viewMode === 'layer') {
        return tex.block(tex.eq(tex.sym('h'), tex.op('='), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op(','), tex.sym('W'), tex.op(':'), tex.dim(3, 2)),
          ...W3.map((r, i) => tex.eq(tex.sym('h', { sub: i + 1, color: 'gold' }), tex.op('='), tex.sym('w', { sub: i + 1, color: 'muted' }), tex.op('·'), tex.sym('x', { color: 'v' }), tex.op('='), tdot(tex, r, X3, 'muted', 'v', 'gold'))));
      }
      if (s.sym) {
        const c = s.sym.col, r = s.sym.row;
        return tex.block(tex.eq(tex.mat([['a', 'b'], ['c', 'd']], { colors: 'cols', highlight: (i, j) => (!isNull(c) && j === c) || (!isNull(r) && i === r) }), tex.vec(['x', 'y'], { color: 'v', highlight: (i) => !isNull(c) && i === c }), tex.op('='), tex.vec(['ax + by', 'cx + dy'], { highlight: (i) => !isNull(r) && i === r })));
      }
      const parts = [];
      const hl = !isNull(s.colHighlight) ? (i, j) => j === s.colHighlight : !isNull(s.rowHighlight) ? (i) => i === s.rowHighlight : undefined;
      parts.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(M, { colors: 'cols', highlight: hl }), tex.op('  '), V, tex.op('='), tex.vec(v, { color: 'v' })));
      if (s.wrong > 0) { parts.push(tex.eq(tex.vec([M[0][0] * v[0], M[1][1] * v[1]], { color: 'muted' }), tex.op('≠'), tex.sym('A'), V, tex.op('='), tex.vec(Av, { color: 'v' }))); return tex.block.apply(null, parts); }
      const colView = s.viewMode !== 'row', rowView = s.viewMode !== 'column';
      if (colView && s.showV && (s.showRecipe || s.dots)) parts.push(tex.eq(tex.sym('A'), V, tex.op('='), tn(tex, v[0], 'v'), tex.vec(N.col(M, 0), { color: 'i' }), tex.op('+'), tn(tex, v[1], 'v'), tex.vec(N.col(M, 1), { color: 'j' }), tex.op('='), tex.vec(Av, { color: 'v' })));
      if (rowView && !s.rightDim) {
        for (let i = 0; i < 2; i++) if (s.shadow[i] > 0.6) parts.push(tex.eq(tex.paren(tex.sym('A') + V) + '<sub>' + (i + 1) + '</sub>', tex.op('='), tex.sym('r', { sub: i + 1, color: 'muted' }), tex.op('·'), V, tex.op('='), tdot(tex, M[i], v, 'muted', 'v', i === 0 ? 'i' : 'j')));
        if (s.assemble > 0) parts.push(tex.eq(tex.sym('A'), V, tex.op('='), tex.vec(Av, { colors: 'rows' }), tex.text(s.viewMode === 'split' ? 'both views agree' : '')));
      }
      return tex.block.apply(null, parts);
    },

    steps: [
      { title: 'The column view, once more', duration: 4,
        narration: 'Last lesson’s recipe: A' + vv + ' is x times the green column plus y times the red column. With A = [[2, −1],[1, 1]] and v = [2; 1]: two copies of [2; 1] and one copy of [−1; 1], tip to tail, reach [3; 3]. This is the <b>column view</b>. Keep it; now we add a second way of seeing the same product.',
        setup(s) { s.viewMode = 'split'; s.lockV = false; s.target = null; s.M = LA.clone(A0); s.v = [2, 1]; s.showV = true; s.showRecipe = true; s.recipeT = 0; s.shadow = [0, 0]; s.assemble = 0; s.pulse = 0; s.rowHighlight = null; s.rightDim = true; s.sym = null; s.wrong = 0; s.tiles = null; s.dots = false; s.target = null; s.showOutLine = false; },
        at(s, t) { const r = seg(t, 0.05, 0.8, 'linear'); s.recipeT = r; s.colHighlight = r <= 0 ? null : r < 0.5 ? 0 : r < 1 ? 1 : null; } },

      { title: 'The rows are arrows too', duration: 4,
        narration: 'Now read A sideways, one <b>row</b> at a time: [2, −1] and [1, 1] are vectors as well — drawn gray on the right as r₁ and r₂. Take the yellow ' + vv + '’s shadow on the first row arrow; shadow length times row length is the dot product r₁·v = 2·2 + (−1)·1 = 3. That number is the first entry of A v.',
        setup(s) { s.recipeT = 1; s.colHighlight = null; s.rightDim = false; s.rowHighlight = 0; s.shadow = [0, 0]; },
        at(s, t) { s.shadow = [seg(t, 0.15, 0.8), 0]; } },

      { title: 'The second row gives the second entry', duration: 4,
        narration: 'Row two does the same: r₂·v = 1·2 + 1·1 = 3, the second entry. Assemble [3; 3] from the two numbers — the same point the column view reached on the left. One product, two stories: mix the columns, or dot each row. This is the <b>row view</b>.',
        setup(s) { s.shadow = [1, 0]; s.rowHighlight = 1; s.assemble = 0; s.pulse = 0; },
        at(s, t) { s.shadow = [1, seg(t, 0, 0.45)]; s.assemble = seg(t, 0.5, 0.8); s.rowHighlight = t < 0.5 ? 1 : null; s.pulse = seg.pulse(t, 0.8, 1); } },

      { title: 'Written out, in general', duration: 4,
        narration: 'Written out for any 2×2: [[a, b],[c, d]] [x; y] = [ax + by; cx + dy]. Read down a column: x multiplies a and c, the green column; y multiplies b and d, the red column. Read across a row: ax + by is row one dotted with v. Same formula, two highlights.',
        setup(s) { s.viewMode = 'column'; s.cam.center = [1.2, 0.6]; s.cam.halfH = 3.8; s.pulse = 0; s.rowHighlight = null; s.sym = { col: null, row: null }; },
        at(s, t) { const c = t < 0.22 ? 0 : t < 0.45 ? 1 : null, r = t < 0.5 ? null : t < 0.72 ? 0 : t < 0.94 ? 1 : null; s.sym = { col: c, row: r }; s.colHighlight = c; } },

      { title: 'Not elementwise', duration: 3.5, misconception: true,
        narration: 'Belief: “multiply matching entries.” That <b>elementwise product</b> would give [2·2; 1·1] = [4; 1], a gray point far from the true [3; 3]. Matrix-times-vector always mixes: every input entry touches every output entry through the columns. Entries never pair off one to one.',
        setup(s) { s.sym = null; s.colHighlight = null; s.wrong = 0; s.cam.center = [1.9, 1.4]; s.cam.halfH = 3.6; },
        at(s, t) { s.wrong = Math.max(0.001, t); } },

      { title: 'Shapes must match', duration: 4.5,
        narration: 'A 2×2 matrix times a 2×1 vector gives a 2×1 vector. The inner 2s — the matrix’s columns and the vector’s rows — must be the same number, because each column needs an entry to scale it. Later, a 2×3 matrix eats a 3-vector and gives a 2-vector; the <b>shape rule</b> is the same.',
        setup(s) { s.wrong = 0; s.tiles = { phase: 0 }; s.cam.center = [0, 0]; s.cam.halfH = 3.8; },
        at(s, t) { s.tiles.phase = t; } },

      { title: 'A matrix is a recipe, not a picture', duration: 4.5,
        narration: 'Feed many vectors through A: every lattice point goes to its own A v, and together they are the moved grid of last lesson. The matrix itself is not the picture; it is the recipe that produces the picture when applied to every vector. Change one entry and every output changes.',
        setup(s) { s.tiles = null; s.viewMode = 'column'; s.showV = false; s.dots = true; s.morph = 0; s.M = LA.clone(A0); s._M0 = LA.clone(A0); s.cam.center = [0.6, 0.6]; s.cam.halfH = 4.2; },
        at(s, t) { s.morph = seg(t, 0.05, 0.6); const M = LA.clone(s._M0); M[0][1] = LA.lerp(-1, 0.5, seg(t, 0.72, 1)); s.M = M; } },

      { title: 'When the rows point the same way', duration: 4.5,
        narration: 'Switch to A = [[1, 2],[2, 4]]. Its two rows are multiples of each other, so both shadows measure the same direction and every output lands on the teal line y = 2x. The input [2; −1] is perpendicular to both rows: both shadows vanish and A v = 0. A non-zero input, squashed to nothing — a preview of the null space.',
        setup(s) { s.dots = false; s.morph = 1; s.viewMode = 'row'; s.M = LA.clone(SING); s.singular = true; s.v = [2, 1]; s.showV = true; s.shadow = [0, 0]; s.assemble = 0; s.pulse = 0; s.showOutLine = false; s.rowHighlight = null; s.cam.center = [1.2, 1.4]; s.cam.halfH = 3.8; },
        at(s, t) { s.v = lerpV([2, 1], [2, -1], seg(t, 0.05, 0.4)); s.shadow = [seg(t, 0.42, 0.68), seg(t, 0.5, 0.76)]; s.assemble = seg(t, 0.72, 0.86); s.showOutLine = t > 0.6; s.pulse = seg.pulse(t, 0.86, 1); } },

      { title: 'A layer is many rows at once', duration: 5, mlTie: true,
        narration: 'A layer computes h = W' + xx + '. Column view: mix the learned directions, one per input feature. Row view: each neuron is one row, taking its dot product with x — the neuron of Part 1, stacked. Three neurons, two inputs: a 3×2 matrix and three score bars. For a batch of inputs the rows become the matrix X, which Part 6 shows.',
        setup(s) { s.viewMode = 'layer'; s.layerT = 0; s.heatA = 0; },
        at(s, t) { s.layerT = seg(t, 0.02, 0.6, 'linear'); s.heatA = seg(t, 0.55, 0.75); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'A v has two faces. Column view: x times the green column plus y times the red column. Row view: each output entry is a row dotted with v. Both give the same point. Shapes chain inner-to-inner. It is never elementwise; every input touches every output.',
        cards: ['<b>Column view:</b> A v = x·col₁ + y·col₂, laid tip to tail.', '<b>Row view:</b> entry i of A v is rowᵢ · v, one shadow per entry.', '<b>Shapes chain</b> inner-to-inner; never elementwise.'],
        setup(s) { s.viewMode = 'recap'; s.M = LA.clone(A0); s.v = [2, 1]; s.singular = false; s.showOutLine = false; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit A and <span class="c-v">v</span>, or drag the landing spots of <span class="c-i">î</span> and <span class="c-j">ĵ</span> (left) and the tip of <span class="c-v">v</span> (right). Hover a landing spot to light up its column.</p>',
      setup(s) { s.viewMode = 'split'; s.lockV = false; s.target = null; s.M = LA.clone(A0); s.v = [2, 1]; s.showV = true; s.showRecipe = true; s.recipeT = 1; s.shadow = [1, 1]; s.assemble = 1; s.rightDim = false; s.colHighlight = null; s.rowHighlight = null; s.pulse = 0; },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A', colors: 'cols', step: 0.1, min: -3, max: 3 },
        { type: 'matrix', bind: 'v', label: 'v', colors: 'v', step: 0.1, min: -4, max: 4 },
        { type: 'preset', label: 'View', items: [
          { label: 'Column', apply(s) { s.viewMode = 'column'; s.cam.center = [1.2, 1.1]; } },
          { label: 'Row', apply(s) { s.viewMode = 'row'; s.cam.center = [1.2, 1.1]; } },
          { label: 'Split', apply(s) { s.viewMode = 'split'; } },
        ] },
        { type: 'preset', label: 'Examples', items: [
          { label: 'Default', apply(s) { s.M = LA.clone(A0); s.v = [2, 1]; s.singular = false; s.showOutLine = false; } },
          { label: 'Rotation 90°', apply(s) { s.M = [[0, -1], [1, 0]]; s.singular = false; s.showOutLine = false; } },
          { label: 'Singular', apply(s) { s.M = LA.clone(SING); s.v = [2, -1]; s.singular = true; s.showOutLine = true; } },
        ] },
        { type: 'toggle', bind: 'showGhost', label: 'Ghost grid' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the landing spots of ' + ihat + ' and ' + jhat + ' so that A·[1; 1] = [3; 0] (the gray target).',
        setup(s) { s.viewMode = 'split'; s.M = I2(); s.v = [1, 1]; s.showV = true; s.showRecipe = true; s.recipeT = 1; s.shadow = [1, 1]; s.assemble = 1; s.rightDim = false; s.target = [3, 0]; s.lockV = true; s.colHighlight = null; s.rowHighlight = null; s.pulse = 0; },
        judge(s) { return LA.judge.near(N.matVec(s.M, s.v), [3, 0], LA.TOL.pos); },
        hint: 'In the column view A v is column 1 plus column 2: make the two landed arrows add up to [3; 0], e.g. [2; −1] and [1; 1].',
        solution(s) { s.M = [[2, 1], [-1, 1]]; } },
      { kind: 'entry',
        prompt: 'Type the matrix that sends [1; 0] to [0; 1] and [0; 1] to [−1; 0].',
        bind: 'M', answer: [[0, -1], [1, 0]], tol: LA.TOL.entry, label: 'A', colors: 'cols',
        setup(s) { s.viewMode = 'column'; s.lockV = false; s.M = I2(); s.v = [2, 1]; s.showV = true; s.showRecipe = true; s.recipeT = 1; s.target = null; s.cam.center = [0.8, 0.8]; } },
      { kind: 'manipulate',
        prompt: 'With A = [[1, 2],[2, 4]], drag ' + vv + ' so that A v is the zero vector (v itself not zero).',
        setup(s) { s.viewMode = 'row'; s.lockV = false; s.target = null; s.M = LA.clone(SING); s.singular = true; s.v = [1, 1]; s.showV = true; s.shadow = [1, 1]; s.assemble = 1; s.showOutLine = true; s.rowHighlight = null; s.pulse = 0; s.cam.center = [1.0, 1.4]; s.cam.halfH = 3.8; },
        judge(s) { const n = N.norm(N.matVec(s.M, s.v)), ok = n < LA.TOL.pos && N.norm(s.v) > 0.5; return { ok, progress: Math.max(0, 1 - n / 4), hint: N.norm(s.v) <= 0.5 ? 'Keep v away from the origin; only its direction matters.' : undefined }; },
        hint: 'Make v perpendicular to the gray row arrows: both shadows must shrink to the origin.',
        solution(s) { s.v = [2, -1]; } },
    ],
  });
})();
