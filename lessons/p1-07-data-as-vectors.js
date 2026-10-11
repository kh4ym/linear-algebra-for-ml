/* p1-07 — Data as vectors; the n-D honesty lesson. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp, lerpV = LA.lerpV, C = LA.color;
  const FEAT = ['length', 'width', 'height', 'weight'];
  const FS = ['f₁', 'f₂', 'f₃', 'f₄'];
  const PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const HONEST = 'The picture stops at 3. The formula does not.';
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(ch => SUBS[+ch]).join('');
  const WORDS = ['dog', 'cat', 'bird', 'fish', 'car', 'tree'];
  const ONEHOT = [0, 1, 0, 0, 0, 0], BOW = [2, 0, 1, 0, 3, 1], RGB = [0.9, 0.5, 0.1];

  // ---------- seeded synthetic data ----------
  function makeFlowers() {                       // flowers60: LA.rng(7)
    const rand = LA.rng(7); const X = [], y = [];
    const grow = (mean, sd, label) => { for (let i = 0; i < 30; i++) { X.push(mean.map(m => Math.max(0.1, m + sd * LA.rng.gauss(rand)))); y.push(label); } };
    grow([2, 1, 3, 0.5], 0.4, 0); grow([3.5, 2.5, 1.5, 1.5], 0.5, 1);
    return { X, y };
  }
  function rowsOf(X, y, c) { return X.filter((_, i) => y[i] === c); }
  function classVar(rows) { const mu = N.mean(rows); return mu.map((m, k) => rows.reduce((a, r) => a + (r[k] - m) * (r[k] - m), 0) / (rows.length - 1)); }
  function fisherScores(X, y) {
    const A = rowsOf(X, y, 0), B = rowsOf(X, y, 1), mA = N.mean(A), mB = N.mean(B), vA = classVar(A), vB = classVar(B);
    return PAIRS.map(pr => pr.reduce((a, k) => a + (mA[k] - mB[k]) * (mA[k] - mB[k]) / (vA[k] + vB[k]), 0));
  }
  /** two flowers that (almost) coincide in (f₁, f₂) but differ a lot in f₃ */
  function overlapPair(X, y) {
    let best = [0, 30], bs = -Infinity;
    for (let i = 0; i < X.length; i++) for (let j = i + 1; j < X.length; j++) {
      const d12 = Math.hypot(X[i][0] - X[j][0], X[i][1] - X[j][1]), d3 = Math.abs(X[i][2] - X[j][2]);
      const score = Math.min(d3, 2.5) - 8 * d12;
      if (score > bs) { bs = score; best = [i, j]; }
    }
    return best;
  }
  /** axis names placed inside the canvas (the x name right-aligned at the right edge) */
  function axisNames(g, nx, ny) {
    const b = g.P.bounds();
    g.label(nx, [b.x1, 0], { anchor: 'right', dx: -8, dy: -13, color: 'hud', font: 'ui', size: 13 });
    g.label(ny, [0, b.y1], { anchor: 'left', dx: 8, dy: 13, color: 'hud', font: 'ui', size: 13 });
  }
  let DIG = null;
  const digits = () => { if (!DIG) DIG = LA.rng.syntheticDigits(2, 8, { size: 28 }); return DIG; };   // index 7 is the "7"
  const toGrid = (px, n) => { const g = []; for (let r = 0; r < n; r++) g.push(px.slice(r * n, r * n + n)); return g; };

  // ---------- cameras / layout ----------
  const CAM2 = () => ({ center: [2.4, 2.3], halfH: 3.8 });
  const CAM3 = () => Object.assign(LA.cam.defaults3d(), { target: [2.4, 2.4, 1.5], dist: 16.5, yaw: 0.6, pitch: 0.42 });
  const CAMD = () => ({ center: [0, 0.3], halfH: 3.6 });
  const CAMW = () => ({ center: [0, 0], halfH: 3.6 });
  const IMG = { x0: -2.75, x1: -0.35, y0: 1.0, y1: 3.4 };
  const STRIP = { x0: -2.65, x1: 2.65, y0: -0.25, y1: 0.07 };
  const GHOST = [{ x0: -2.65, x1: 2.65, y0: -0.95, y1: -0.7 }, { x0: -2.65, x1: 2.65, y0: -1.45, y1: -1.2 }];
  const MARK = [[0.0, -1.95], [-1.9, -2.45], [1.8, -2.55]];
  const lerpRect = (a, b, u) => ({ x0: lerp(a.x0, b.x0, u), x1: lerp(a.x1, b.x1, u), y0: lerp(a.y0, b.y0, u), y1: lerp(a.y1, b.y1, u) });
  const tiny = p => ({ x0: p[0] - 0.02, x1: p[0] + 0.02, y0: p[1] - 0.02, y1: p[1] + 0.02 });
  const rectPts = r => [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]];

  function base(s) {
    s.scene = 'flowers'; s.view = '2d'; s.cam = CAM2();
    s.axesFrom = [0, 1]; s.axesTo = [0, 1]; s.slide = 1; s.rise = 1; s.appear = 1;
    s.showTable = false; s.sizeBy4 = false; s.colorBy = true; s.pairT = 0; s.cross = null; s.dimT = 0;
    s.showStrip = 0; s.meanT = 0; s.showMeans = false; s.showLine = false; s.lineAngle = 0;
    s.flatten = 0; s.collapse = 0; s.ghostT = 0; s.ghostC = 0; s.pix = null; s.wordsT = 0;
    s.showMarker = false; s.showPick = false;
  }
  function pos2(s, i) { const x = s.X[i], a = s.axesFrom, b = s.axesTo; return [lerp(x[a[0]], x[b[0]], s.slide), lerp(x[a[1]], x[b[1]], s.slide)]; }
  function pos3(s, i) { const x = s.X[i]; return [x[0], x[1], x[2] * s.rise]; }
  const ptColor = (s, i) => s.colorBy ? (s.y[i] ? 'dataB' : 'data') : 'muted';
  const radius = (s, i) => s.sizeBy4 ? 2.5 + 4.5 * LA.clamp01((s.X[i][3] - 0.1) / 2.4) : 4.5;
  function lineTheta(s) { const ax = s.axesTo; const d = [s.meanB[ax[0]] - s.meanA[ax[0]], s.meanB[ax[1]] - s.meanA[ax[1]]]; return Math.atan2(d[1], d[0]) - Math.PI / 2; }
  function lineGeom(s, th) { const ax = s.axesTo; const m = [(s.meanA[ax[0]] + s.meanB[ax[0]]) / 2, (s.meanA[ax[1]] + s.meanB[ax[1]]) / 2]; return { m, d: [Math.cos(th), Math.sin(th)], n: [-Math.sin(th), Math.cos(th)] }; }
  function accuracy(s, th) { const L = lineGeom(s, th); let c = 0; for (let i = 0; i < s.X.length; i++) { const p = pos2(s, i); if ((N.dot(L.n, N.sub(p, L.m)) > 0 ? 1 : 0) === s.y[i]) c++; } return c; }
  const hotIndex = ctx => (ctx && typeof ctx.hot === 'string' && ctx.hot.indexOf('pt') === 0) ? +ctx.hot.slice(2) : null;

  /** a 1×n strip of cells in world coordinates with printed values and names under the cells */
  function cellsRow(g, vals, geo, o) {
    o = o || {}; const n = vals.length, x0 = geo.x0, y0 = geo.y0, cw = geo.cw, ch = geo.ch, al = o.alpha === undefined ? 1 : o.alpha;
    if (al <= 0) return;
    g.heat2d([vals], { x0, x1: x0 + n * cw, y0, y1: y0 + ch }, { range: o.range, ramp: o.ramp || C.heat, alpha: 0.95 * al, smooth: false });
    for (let k = 0; k < n; k++) {
      const cx = x0 + (k + 0.5) * cw;
      if (o.values !== false) g.label(o.fmt ? o.fmt(vals[k]) : LA.fmt(vals[k]), [cx, y0 + ch / 2], { size: o.size || 12, color: o.valueColor || 'text', bg: false, font: 'mono', alpha: al });
      if (o.labels) g.label(o.labels[k], [cx, y0], { anchor: 'below', dy: 3, size: 11, color: 'muted', alpha: al, font: 'ui' });
      if (k) g.line([x0 + k * cw, y0], [x0 + k * cw, y0 + ch], { color: 'bg', width: 1, alpha: al });
    }
    g.polyline(rectPts({ x0, x1: x0 + n * cw, y0, y1: y0 + ch }), { color: 'line', closed: true, width: 1, alpha: al });
    if (o.title) g.label(o.title, [x0, y0 + ch], { anchor: 'left', dy: -11, size: 12, color: o.titleColor || 'hud', bg: false, alpha: al, font: 'ui' });
  }

  // ---------- flowers, 2D ----------
  function drawTable(g, s, hi) {
    const W = g.size.w; const x = W - 212, y0 = 36, dy = 15;
    g.hud('row', x - 34, y0 - dy, { size: 11, color: 'muted' });
    FS.forEach((f, k) => g.hud(f, x + k * 38 + 8, y0 - dy, { size: 11, color: 'muted', anchor: 'center' }));
    g.hud('species', x + 164, y0 - dy, { size: 11, color: 'muted' });
    for (let r = 0; r < 8; r++) {
      const gold = r === hi;
      g.hud(String(r + 1), x - 34, y0 + r * dy, { size: 11, color: gold ? 'gold' : 'muted', font: 'mono' });
      for (let k = 0; k < 4; k++) g.hud(LA.fmt.fixed(s.X[r][k], 1), x + k * 38 + 8, y0 + r * dy, { size: 11.5, color: gold ? 'gold' : 'text', font: 'mono', anchor: 'center' });
      g.hud(s.y[r] ? 'B' : 'A', x + 176, y0 + r * dy, { size: 11.5, color: s.y[r] ? 'dataB' : 'data', font: 'ui', anchor: 'center' });
    }
    g.hud('… 60 rows, 4 features', x - 34, y0 + 8 * dy + 2, { size: 11, color: 'muted' });
    return { x, y0, dy };
  }
  function drawMean(g, s, c, u, col) {
    if (u <= 0) return;
    const idx = []; for (let i = 0; i < s.X.length; i++) if (s.y[i] === c) idx.push(i);
    const k = Math.max(1, Math.ceil(u * idx.length)); const used = idx.slice(0, k);
    const m = N.mean(used.map(i => pos2(s, i)));
    used.forEach(i => g.line(pos2(s, i), m, { color: col, width: 0.8, alpha: 0.3 }));
    g.point(m, { r: 7, color: 'gold', stroke: 'white' });
    g.label(c ? 'rose mean' : 'blue mean', m, { anchor: 'below', dy: 10, size: 12, color: 'gold', font: 'ui' });
    if (u >= 1) cellsRow(g, c ? s.meanB : s.meanA, { x0: 3.1, y0: c ? 0.5 : 1.15, cw: 0.5, ch: 0.4 }, { range: [0, 5], labels: c ? FS : null, title: (c ? 'rose mean' : 'blue mean') + ': a vector', titleColor: col, fmt: x => LA.fmt.fixed(x, 2), size: 11.5 });
  }
  function renderFlowers2d(g, s, ctx) {
    const H = g.size.h;
    const hot = hotIndex(ctx); const hi = hot !== null ? hot : s.hi;
    const ax = s.slide < 0.5 ? s.axesFrom : s.axesTo;
    g.ghostGrid({ range: 8 });
    g.axes({});
    axisNames(g, FEAT[ax[0]] + ' (' + FS[ax[0]] + ')', FEAT[ax[1]] + ' (' + FS[ax[1]] + ')');
    const tbl = s.showTable ? drawTable(g, s, hi) : null;
    for (let i = 0; i < s.X.length; i++) {
      let p = pos2(s, i), al = 0.95;
      if (s.appear < 1) {
        const u = seg(s.appear, i * 0.85 / 60, i * 0.85 / 60 + 0.15, 'out');
        if (u <= 0) continue;
        if (tbl) { const st = g.P.toWorld(tbl.x + 70, tbl.y0 + Math.min(i, 7) * tbl.dy); p = lerpV(st, p, u); }
        al = 0.25 + 0.7 * u;
      }
      g.point(p, { r: radius(s, i), color: ptColor(s, i), alpha: al });
      if (ctx.mode === 'play') g.handle('pt' + i, p, () => { }, { show: false, r: 9 });
    }
    if (s.showTable || s.showStrip > 0 || ctx.mode === 'play') g.marker.ring(pos2(s, hi), { r: 9, color: 'gold' });
    if (s.showStrip > 0) {
      const u = s.showStrip; const x0 = 2.45 + (1 - u) * 3.2, y0 = 4.65, cw = 0.55, ch = 0.5; const v = s.X[hi];
      cellsRow(g, v, { x0, y0, cw, ch }, { range: [0, 5], labels: FS, title: 'x' + sub(hi + 1) + ' — the whole row', fmt: x => LA.fmt.fixed(x, 1), alpha: u });
      g.line(pos2(s, hi), [x0, y0], { color: 'gold', width: 1, dashed: true, alpha: 0.8 * u });
    }
    if (s.showMeans) { drawMean(g, s, 0, seg(s.meanT, 0, 0.5, 'linear'), 'data'); drawMean(g, s, 1, seg(s.meanT, 0.5, 1, 'linear'), 'dataB'); }
    if (s.showLine) {
      const L = lineGeom(s, s.lineAngle);
      g.lineThrough(L.m, L.d, { color: 'gold', width: 2.5, len: 12 });
      let correct = 0;
      for (let i = 0; i < s.X.length; i++) { const p = pos2(s, i); if ((N.dot(L.n, N.sub(p, L.m)) > 0 ? 1 : 0) === s.y[i]) correct++; else g.marker.ring(p, { r: 8, color: 'residual' }); }
      g.vector(L.m, N.add(L.m, N.scale(L.n, 0.9)), { color: 'w', width: 2, head: 9, label: 'w', labelAt: 1 });
      g.hud('accuracy  ' + correct + ' / 60', 14, H - 58, { size: 15, color: 'gold', bold: true });
      g.hud('gray rings: misclassified · p1-09: your first model', 14, H - 38, { size: 11.5, color: 'muted' });
    }
    if (s.showMarker) {
      g.point(s.marker, { r: 7, color: 'gold', stroke: 'white', label: 'marker', labelColor: 'gold' });
      g.handle('marker', s.marker, (p, st) => { st.marker = [p[0], p[1]]; }, { color: 'gold' });
    }
    if (s.showPick) {
      const cy = 5.15, cw = 0.78, gap = 0.08, ch = 0.5, x0 = 0.15;
      const cxOf = k => x0 + k * (cw + gap);
      PAIRS.forEach((pr, k) => {
        const cx = cxOf(k), sel = k === s.pick;
        g.polygon([[cx, cy], [cx + cw, cy], [cx + cw, cy + ch], [cx, cy + ch]], { fill: sel ? 'gold' : 'panel2', alpha: sel ? 0.3 : 0.9, stroke: sel ? 'gold' : 'line', width: 1 });
        g.label(FS[pr[0]] + ', ' + FS[pr[1]], [cx + cw / 2, cy + ch / 2], { size: 12, color: sel ? 'gold' : 'hud', bg: false, font: 'ui' });
      });
      const selC = [cxOf(s.pick) + cw / 2, cy + ch / 2];
      g.marker.ring(selC, { r: 17, color: 'gold' });
      g.handle('pick', selC, (p, st) => {
        let bk = 0, bd = Infinity; PAIRS.forEach((pr, k) => { const d = Math.abs(p[0] - (cxOf(k) + cw / 2)); if (d < bd) { bd = d; bk = k; } });
        st.pick = bk; st.axesFrom = PAIRS[bk]; st.axesTo = PAIRS[bk]; st.slide = 1;
      }, { color: 'gold', r: 18 });
      g.hud('drag the gold ring onto an axis pair (top row)', 14, H - 60, { size: 11.5, color: 'muted' });
      g.hud('separation of this pair  ' + LA.fmt(s.pairScores[s.pick]) + '   (best pair  ' + LA.fmt(Math.max.apply(null, s.pairScores)) + ')', 14, H - 40, { size: 12.5, color: 'gold' });
    }
    if (s.pix === null && !s.showLine && !s.showPick && !s.showMarker) g.hud('2 of 4 features drawn: ' + FS[ax[0]] + ', ' + FS[ax[1]] + ' — the other two are hidden', 14, H - 38, { size: 11.5, color: 'muted' });
  }

  // ---------- flowers, 3D ----------
  function renderFlowers3d(g, s, ctx) {
    const H = g.size.h;
    for (let i = 0; i <= 5; i++) { g.line([i, 0, 0], [i, 5, 0], { color: 'grid', width: 1, alpha: 0.8, back: true }); g.line([0, i, 0], [5, i, 0], { color: 'grid', width: 1, alpha: 0.8, back: true }); }
    g.vector([0, 0, 0], [5.8, 0, 0], { color: 'gridAxis', width: 1.4, head: 8, label: 'length (f₁)', labelColor: 'hud', labelFont: 'ui', labelSize: 12 });
    g.vector([0, 0, 0], [0, 5.8, 0], { color: 'gridAxis', width: 1.4, head: 8, label: 'width (f₂)', labelColor: 'hud', labelFont: 'ui', labelSize: 12 });
    g.vector([0, 0, 0], [0, 0, 4.6], { color: 'gridAxis', width: 1.4, head: 8, label: 'height (f₃)', labelColor: 'hud', labelFont: 'ui', labelSize: 12 });
    const pair = s.pairT > 0 ? s.pairHi : null;
    for (let i = 0; i < s.X.length; i++) {
      const p = pos3(s, i); const isHi = pair && (i === pair[0] || i === pair[1]);
      if (s.rise > 0 && s.rise < 1) g.line([p[0], p[1], 0], p, { color: 'residual', width: 0.6, alpha: 0.35 });
      g.point(p, { r: radius(s, i), color: ptColor(s, i), alpha: pair && !isHi ? 0.4 : 0.95 });
    }
    if (pair) {
      const a = pos3(s, pair[0]), b = pos3(s, pair[1]);
      g.marker.ring(a, { r: 10, color: 'gold' }); g.marker.ring(b, { r: 10, color: 'gold' });
      if (s.cross) { g.label('“same point”', a, { anchor: 'left', dx: 18, dy: -24, color: 'bad', size: 13.5, font: 'ui' }); g.marker.cross(a, { dx: 112, dy: -24, size: 18 }); }
      if (s.dimT > 0) {
        g.line(a, b, { color: 'gold', width: 1.4, dashed: true, alpha: s.dimT });
        g.dim(a, b, { label: '3D distance ' + LA.fmt(N.dist(a, b)), color: 'gold', offset: 20 });
        g.hud('in the length–width view they are ' + LA.fmt(Math.hypot(a[0] - b[0], a[1] - b[1])) + ' apart · in 3D ' + LA.fmt(N.dist(a, b)), 14, H - 38, { size: 12, color: 'gold' });
      }
    }
    g.hud('3 of 4 features drawn' + (s.sizeBy4 ? ' · point size = weight (f₄), not a fourth axis' : ' · weight (f₄) hidden'), 14, H - 58, { size: 11.5, color: 'muted' });
    if (s.sizeBy4) {
      const x = g.size.w - 160, y = H - 84;
      g.viewport({ x, y, w: 150, h: 30 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, border: false }, (gg) => {
        [0.3, 0.9, 1.5, 2.1].forEach((w, k) => gg.point([-4.5 + k * 2.2, 0], { r: 2.5 + 4.5 * LA.clamp01((w - 0.1) / 2.4), color: 'hud', alpha: 0.9 }));
      });
      g.hud('size = weight (f₄): 0.3 … 2.1', x, y + 40, { size: 11, color: 'muted' });
    }
  }

  // ---------- digit scene ----------
  function renderDigit(g, s, ctx) {
    const D = digits(); const px7 = D[7].pixels, grid7 = toGrid(px7, 28);
    const cw = (STRIP.x1 - STRIP.x0) / 784, rowH = (IMG.y1 - IMG.y0) / 28, col = s.collapse;
    g.image(grid7, IMG, { range: [0, 1] });
    g.polyline(rectPts(IMG), { color: 'line', closed: true, width: 1 });
    g.label('synthetic digit · 28 × 28 = 784 pixels', [(IMG.x0 + IMG.x1) / 2, IMG.y1], { anchor: 'above', dy: -6, size: 12, color: 'muted', font: 'ui' });
    const kf = s.flatten * 28, done = Math.min(28, Math.floor(kf + 1e-9)), fly = s.flatten < 1 ? done : -1, u = kf - done;
    const rect = col > 0 ? lerpRect(STRIP, tiny(MARK[0]), col) : STRIP;
    if (col < 1) {
      g.polyline(rectPts(rect), { color: 'line', closed: true, width: 1, alpha: 1 - col });
      if (done > 0) g.image([px7.slice(0, done * 28)], { x0: rect.x0, x1: rect.x0 + (rect.x1 - rect.x0) * done / 28, y0: rect.y0, y1: rect.y1 }, { range: [0, 1] });
      if (done === 28 && col === 0) g.label('x — the same 784 numbers as one strip', [STRIP.x0, STRIP.y1], { anchor: 'left', dy: -11, size: 12, color: 'hud', bg: false, font: 'ui' });
      if (s.flatten > 0 && s.flatten < 1) g.label(String(done * 28) + ' of 784 cells', [STRIP.x1, STRIP.y1], { anchor: 'right', dy: -11, size: 12, color: 'hud', bg: false, font: 'ui' });
    }
    if (fly >= 0 && fly < 28) {
      const ry1 = IMG.y1 - fly * rowH, ry0 = ry1 - rowH;
      g.polygon([[IMG.x0, ry0], [IMG.x1, ry0], [IMG.x1, ry1], [IMG.x0, ry1]], { fill: 'gold', alpha: 0.22, stroke: 'gold', width: 1.2 });
      const from = { x0: IMG.x0, x1: IMG.x1, y0: ry0, y1: ry1 }, to = { x0: STRIP.x0 + fly * 28 * cw, x1: STRIP.x0 + (fly + 1) * 28 * cw, y0: STRIP.y0, y1: STRIP.y1 };
      const r = lerpRect(from, to, seg(u, 0, 1));
      g.image([px7.slice(fly * 28, fly * 28 + 28)], r, { range: [0, 1] });
      g.polyline(rectPts(r), { color: 'gold', closed: true, width: 1 });
      g.label('row ' + (fly + 1) + '  →  cells ' + (fly * 28 + 1) + ' … ' + (fly * 28 + 28), [1.25, 1.6], { size: 12.5, color: 'gold', bg: false, font: 'ui' });
    }
    g.label('read row by row', [1.25, 3.1], { size: 13, color: 'hud', bg: false, font: 'ui' });
    g.label('x = [x₁; x₂; … ; x₇₈₄]', [1.25, 2.6], { size: 14, color: 'text', bg: false, font: 'math' });
    g.label('784 numbers in, 784 numbers out: nothing is lost', [1.25, 2.15], { size: 11.5, color: 'muted', bg: false, font: 'ui' });
    // ghosts: two other synthetic digits as strips, then markers
    [1, 3].forEach((lab, k) => {
      const al = s.ghostT; if (al <= 0) return;
      const gr = s.ghostC > 0 ? lerpRect(GHOST[k], tiny(MARK[k + 1]), s.ghostC) : GHOST[k];
      if (s.ghostC < 1) { g.image([D[lab].pixels], gr, { range: [0, 1], alpha: 0.45 * al }); g.polyline(rectPts(gr), { color: 'line', closed: true, width: 1, alpha: al * (1 - s.ghostC) }); if (s.ghostC === 0) g.label('another synthetic digit (“' + lab + '”)', [GHOST[k].x0, GHOST[k].y1], { anchor: 'left', dy: -9, size: 11, color: 'muted', bg: false, font: 'ui', alpha: al }); }
      const mr = 6 * seg(s.ghostC, 0.6, 1);
      if (mr > 0) { g.point(MARK[k + 1], { r: mr, color: 'gold', alpha: 0.6 }); g.label('“' + lab + '”', MARK[k + 1], { anchor: 'below', dy: 8, size: 11.5, color: 'muted', font: 'ui' }); }
    });
    const mr = 8 * seg(col, 0.6, 1);
    if (mr > 0) {
      g.point(MARK[0], { r: mr, color: 'gold', stroke: 'white' });
      g.label('“7”: one point in ℝ⁷⁸⁴ (position not drawable)', MARK[0], { anchor: 'above', dy: -10, size: 12.5, color: 'gold', font: 'ui' });
    }
    if (col > 0.5) g.label(HONEST, [0, -3.15], { size: 13.5, color: 'hud', bold: true, font: 'ui' });
    if (s.pix) {
      const [pr, pc] = s.pix; const cx = IMG.x0 + (pc + 0.5) * rowH, cy = IMG.y1 - (pr + 0.5) * rowH;
      g.polyline(rectPts({ x0: IMG.x0 + pc * rowH, x1: IMG.x0 + (pc + 1) * rowH, y0: IMG.y1 - (pr + 1) * rowH, y1: IMG.y1 - pr * rowH }), { color: 'gold', closed: true, width: 2 });
      g.label('row ' + (pr + 1) + ', column ' + (pc + 1), [cx, IMG.y0], { anchor: 'below', dy: 5, size: 12, color: 'gold', font: 'ui' });
      // the band of this row in the strip
      const bx0 = STRIP.x0 + pr * 28 * cw, bx1 = bx0 + 28 * cw;
      g.polyline(rectPts({ x0: bx0, x1: bx1, y0: STRIP.y0 - 0.04, y1: STRIP.y1 + 0.04 }), { color: 'gold', closed: true, width: 1, alpha: 0.7 });
      g.label('row ' + (pr + 1) + '’s 28 cells', [(bx0 + bx1) / 2, STRIP.y0], { anchor: 'below', dy: 6, size: 11, color: 'gold', font: 'ui' });
      const idx = Math.round(s.pixIndex[0]);
      if (idx >= 1 && idx <= 784) {                    // cells counted from 1, like the subscripts x₁ … x₇₈₄
        const ix = STRIP.x0 + (idx - 0.5) * cw;
        g.line([ix, STRIP.y0 - 0.12], [ix, STRIP.y1 + 0.12], { color: 'b', width: 2 });
        g.label('typed cell ' + idx, [ix, STRIP.y0 - 0.12], { anchor: 'below', dy: 22, size: 12, color: 'b', font: 'ui' });
        g.line([cx, cy], [ix, STRIP.y1 + 0.12], { color: 'b', width: 1, dashed: true, alpha: 0.8 });
      }
    }
  }

  // ---------- words & colors ----------
  function renderWords(g, s, ctx) {
    const u1 = seg(s.wordsT, 0, 0.3), u2 = seg(s.wordsT, 0.3, 0.6), u3 = seg(s.wordsT, 0.6, 0.9);
    cellsRow(g, ONEHOT, { x0: -1.65, y0: 2.45, cw: 0.55, ch: 0.48 }, { range: [0, 1], ramp: t => C.mix(C.panel2 || '#1a1d26', C.gold, t), labels: WORDS, title: 'one-hot: “cat”', titleColor: 'gold', alpha: u1 });
    cellsRow(g, BOW, { x0: -1.65, y0: 0.95, cw: 0.55, ch: 0.48 }, { range: [0, 3], labels: WORDS, title: 'bag of words: counts', titleColor: 'teal', alpha: u2 });
    if (u2 > 0) g.label('“a dog, a bird, a dog, a tree, a car, a car, a car”', [0, 0.95], { dy: 34, size: 11, color: 'muted', bg: false, font: 'ui', alpha: u2 });
    if (u3 > 0) {
      g.viewport({ x: 0.1, y: 0.53, w: 0.8, h: 0.38 }, { view: '3d', cam: { target: [0.55, 0.55, 0.5], dist: 3.5, yaw: 0.75, pitch: 0.38, fov: 36 }, border: true }, (gg, r) => {
        gg.box(null, { fill: 'white', alpha: 0.05, stroke: 'muted', width: 1 });
        gg.vector([0, 0, 0], [1.3, 0, 0], { color: 'j', width: 1.6, head: 8, label: 'R', labelColor: 'j' });
        gg.vector([0, 0, 0], [0, 1.3, 0], { color: 'i', width: 1.6, head: 8, label: 'G', labelColor: 'i' });
        gg.vector([0, 0, 0], [0, 0, 1.3], { color: 'k', width: 1.6, head: 8, label: 'B', labelColor: 'k' });
        gg.vector([0, 0, 0], RGB, { color: 'b', width: 2.4, components: true, label: 'c', labelAt: 0.55 });
        gg.point(RGB, { r: 7, color: 'b' });
        gg.label('orange = [0.9; 0.5; 0.1]', RGB, { anchor: 'left', dx: 14, dy: 16, color: 'b', size: 13, font: 'ui' });
        gg.hud('a color is a point in the RGB cube', r.x + 10, r.y + 16, { size: 12, color: 'b' });
      });
    }
    g.label('listable as numbers  ⇒  a vector  ⇒  this course applies', [0, -3.35], { size: 13, color: 'hud', bg: false, font: 'ui', alpha: u3 });
  }

  // ---------- recap ----------
  function renderRecap(g, s, ctx) {
    const flowersMini = (gg, rr, title, means) => {
      gg.axes({ ticks: false, names: ['f₁', 'f₂'] });
      for (let i = 0; i < s.X.length; i++) gg.point([s.X[i][0], s.X[i][1]], { r: 3.2, color: s.y[i] ? 'dataB' : 'data' });
      if (means) [s.meanA, s.meanB].forEach((m, c) => { gg.point([m[0], m[1]], { r: 6, color: 'gold', stroke: 'white' }); gg.label(c ? 'rose mean' : 'blue mean', [m[0], m[1]], { anchor: 'below', dy: 9, size: 11, color: 'gold', font: 'ui' }); });
      gg.hud(title, rr.x + 10, rr.y + 16, { size: 12.5, color: 'gold' });
    };
    g.viewport({ x: 0, y: 0.02, w: 1, h: 0.31 }, { view: '2d', cam: { center: [1.0, 2.2], halfH: 3.1 } }, (gg, rr) => {
      flowersMini(gg, rr, 'row = feature vector · scatter = shadow on chosen axes (f₃, f₄ dropped)', false);
      for (let r = 0; r < 4; r++) gg.label('x' + sub(r + 1) + ' = [' + s.X[r].map(x => LA.fmt.fixed(x, 1)).join('; ') + ']', [-1.4, 3.3 - r * 0.75], { anchor: 'right', dx: -12, size: 11.5, color: s.y[r] ? 'dataB' : 'data', bg: false, font: 'mono' });
      gg.label('rows → points', [-1.4, 0.3], { anchor: 'right', dx: -12, size: 11.5, color: 'muted', bg: false, font: 'ui' });
    });
    g.viewport({ x: 0, y: 0.345, w: 1, h: 0.31 }, { view: '2d', cam: { center: [0, 0.9], halfH: 2.0 } }, (gg, rr) => {
      const D = digits(); const grid7 = toGrid(D[7].pixels, 28);
      const im = { x0: -4.4, x1: -2.4, y0: -0.1, y1: 1.9 }; gg.image(grid7, im, { range: [0, 1] }); gg.polyline(rectPts(im), { color: 'line', closed: true, width: 1 });
      gg.vector([-2.2, 0.9], [-1.6, 0.9], { color: 'hud', width: 1.4, head: 8 });
      const st = { x0: -1.4, x1: 2.2, y0: 0.72, y1: 1.08 }; gg.image([D[7].pixels], st, { range: [0, 1] }); gg.polyline(rectPts(st), { color: 'line', closed: true, width: 1 });
      gg.label('784 cells', [0.4, 0.72], { anchor: 'below', dy: 4, size: 11, color: 'muted', font: 'ui' });
      gg.vector([2.4, 0.9], [3.0, 0.9], { color: 'hud', width: 1.4, head: 8 });
      gg.point([3.6, 0.9], { r: 7, color: 'gold', stroke: 'white' }); gg.label('one point in ℝ⁷⁸⁴', [3.6, 0.9], { anchor: 'below', dy: 10, size: 11.5, color: 'gold', font: 'ui' });
      gg.hud('image → flatten → strip → one point (not drawable, computable)', rr.x + 10, rr.y + 16, { size: 12.5, color: 'gold' });
    });
    g.viewport({ x: 0, y: 0.67, w: 1, h: 0.31 }, { view: '2d', cam: { center: [1.0, 2.2], halfH: 3.1 } }, (gg, rr) => {
      flowersMini(gg, rr, 'the mean of vectors is a vector: the cloud’s center', true);
      gg.label('mean = (x₁ + … + x₃₀) / 30', [-1.4, 2.6], { anchor: 'right', dx: -12, size: 12.5, color: 'text', bg: false, font: 'math' });
      gg.label('component by component', [-1.4, 1.7], { anchor: 'right', dx: -12, size: 11.5, color: 'muted', bg: false, font: 'ui' });
    });
  }

  LA.registerLesson({
    id: 'p1-07-data-as-vectors',
    part: 1,
    title: 'Data as vectors; the n-D honesty lesson',
    subtitle: 'Every row is a point; every picture of it is a shadow on the axes you chose.',
    minutes: 9,
    prereqs: ['p1-01-what-is-a-vector', 'p1-02-add-scale'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D (3 of 4 features)', realSize: '784 pixels per digit image; 768–4096 per text embedding; 10⁴–10⁶ features in sparse text models', transfers: '“one feature per axis” and every formula on strips are exact in ℝⁿ; only which 2–3 axes we look at is a choice, and we always say so' },
    cantShow: ['all four flower features at once (the fourth is shown as point size, and labeled)', 'a point in ℝ⁷⁸⁴ (we draw a marker and say “not drawable”)'],
    terms: [
      { term: 'feature vector', def: 'One data row written as a vector, one component per feature.' },
      { term: 'feature space', def: 'The room whose axes are the features; every row is a point in it.' },
      { term: 'scatter plot', def: 'The points of a dataset drawn on two chosen feature axes.' },
      { term: 'flatten', def: 'Write a grid of pixels as one long strip, row by row.' },
      { term: 'mean vector', def: 'The average of many vectors, computed component by component; also called the centroid.' },
      { term: 'one-hot vector', def: 'A strip of zeros with a single 1 marking a category.' },
    ],
    atlas: ['logistic-regression', 'pca', 'embeddings'],
    thumbnail: 2,

    state() {
      const d = makeFlowers();
      const s = { view: '2d', cam: CAM2(), X: d.X, y: d.y, hi: 7, marker: [1, 4], pick: 0, pixIndex: [0] };
      s.meanA = N.mean(rowsOf(d.X, d.y, 0)); s.meanB = N.mean(rowsOf(d.X, d.y, 1));
      s.pairScores = fisherScores(d.X, d.y);
      s.pairHi = overlapPair(d.X, d.y);
      base(s);
      return s;
    },

    render(g, s, ctx) {
      if (s.scene === 'digit') return renderDigit(g, s, ctx);
      if (s.scene === 'words') return renderWords(g, s, ctx);
      if (s.scene === 'recap') return renderRecap(g, s, ctx);
      if (s.view === '3d') return renderFlowers3d(g, s, ctx);
      renderFlowers2d(g, s, ctx);
    },

    math(s, tex) {
      if (s.scene === 'digit') {
        const parts = [tex.eq(tex.sym('x'), tex.op('∈'), tex.text('ℝ⁷⁸⁴')), tex.eq(tex.sym('x', { sub: 'i' }), tex.op('='), tex.text('brightness of pixel i'))];
        if (s.collapse > 0) parts.push(tex.text('‖x‖, x·y, (x + y)/2 — all defined for x, y ∈ ℝ⁷⁸⁴'));
        if (s.pix) parts.push(tex.eq(tex.text('typed cell'), tex.op('='), tex.text(String(Math.round(s.pixIndex[0])), { color: 'b' })));
        return tex.block.apply(null, parts);
      }
      if (s.scene === 'words') return tex.block(
        tex.eq(tex.sym('cat'), tex.op('='), tex.rowvec(ONEHOT, { color: 'gold' })),
        tex.eq(tex.sym('counts'), tex.op('='), tex.rowvec(BOW, { color: 'teal' })),
        tex.eq(tex.sym('rgb'), tex.op('='), tex.rowvec(RGB, { colors: ['j', 'i', 'k'] })));
      if (s.scene === 'recap') return tex.block(tex.text('row = feature vector · strip = the whole vector · mean = a vector'));
      const ax = s.view === '3d' ? [0, 1, 2] : s.axesTo;
      const hi = s.hi, col = s.y[hi] ? 'dataB' : 'data';
      const parts = [tex.eq(tex.sym('x', { sub: String(hi + 1) }), tex.op('='), tex.vec(s.X[hi], { color: col, highlight: (i) => ax.indexOf(i) >= 0 }), tex.op('∈'), tex.text('ℝ⁴'))];
      parts.push(tex.text('drawn: ' + ax.map(k => FS[k] + ' ' + FEAT[k]).join(', ') + (s.sizeBy4 ? ' · f₄ → point size' : '')));
      // numbers next to matrices are written with tex.text (see the report: the panel's in-place update mixes up tex.num and matrix cells)
      if (s.pairT > 0) { const a = s.X[s.pairHi[0]], b = s.X[s.pairHi[1]]; parts.push(tex.eq(tex.norm(tex.sym('x', { sub: String(s.pairHi[0] + 1) }) + tex.op('−') + tex.sym('x', { sub: String(s.pairHi[1] + 1) })), tex.text('in (f₁, f₂) = ' + LA.fmt(Math.hypot(a[0] - b[0], a[1] - b[1])) + ' · in (f₁, f₂, f₃) ='), tex.text(LA.fmt(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])), { color: 'gold' }))); }
      if (s.showMeans) { parts.push(tex.eq(tex.sym('mean', { sub: 'blue' }), tex.op('='), tex.frac('1', '30'), tex.sum(tex.sym('x', { sub: 'i' }), 'i = 1', '30'), tex.op('='), tex.vec(s.meanA, { color: 'data' }))); parts.push(tex.eq(tex.sym('mean', { sub: 'rose' }), tex.op('='), tex.vec(s.meanB, { color: 'dataB' }))); }
      if (s.showLine) parts.push(tex.eq(tex.text('accuracy'), tex.op('='), tex.text(accuracy(s, s.lineAngle) + ' / 60', { color: 'gold' })));
      if (s.showPick) parts.push(tex.eq(tex.text('separation of (' + FS[PAIRS[s.pick][0]] + ', ' + FS[PAIRS[s.pick][1]] + ')'), tex.op('='), tex.text(LA.fmt(s.pairScores[s.pick]), { color: 'gold' })));
      if (s.showMarker) parts.push(tex.eq(tex.sym('marker'), tex.op('='), tex.vec(s.marker, { color: 'gold' })));
      return tex.block.apply(null, parts);
    },

    steps: [
      { title: 'Rows become points, again', duration: 4,
        narration: 'Sixty synthetic flowers, four measurements each: length, width, height, weight; two species, blue and rose. Each row is a <b>feature vector</b> in ℝ⁴. Draw the first two features as axes and each row becomes a point in a <b>scatter plot</b>. The species already begin to separate.',
        setup(s) { base(s); s.showTable = true; s.appear = 0; },
        at(s, t) { s.appear = seg(t, 0.05, 1, 'linear'); } },

      { title: 'Pick different axes, see a different shadow', duration: 4,
        narration: 'Nothing says the first two features are the right ones. Swap to width and height: every point slides to its new place. The data did not change; we are looking at a different shadow of the same sixty vectors, cast onto two other axes of the four-dimensional <b>feature space</b>.',
        setup(s) { base(s); s.showTable = true; s.axesFrom = [0, 1]; s.axesTo = [1, 2]; s.slide = 0; },
        at(s, t) { s.slide = seg(t, 0.1, 0.8); } },

      { title: 'Three features, three axes; the fourth as size', duration: 5,
        narration: 'Use three axes: length, width, height. Now the fourth feature, weight, has no axis left. We encode it as point size and say so on the picture — an honest trick, not a fourth dimension. Drag to orbit the cloud; three of the four features are in view.',
        setup(s) { base(s); s.view = '3d'; s.cam = CAM3(); s.rise = 0; },
        at(s, t) { s.rise = seg(t, 0.05, 0.5); s.sizeBy4 = t > 0.55; LA.cam.tween(s, { yaw: 0.6 }, { yaw: 1.3 }, seg(t, 0, 1, 'linear')); } },

      { title: 'A 2D plot is not the whole dataset', duration: 5, misconception: true,
        narration: 'Belief to test: “the scatter plot shows the data.” Seen from above, flattened to the length–width shadow, two flowers ringed in gold sit on top of each other. Lift the height back up: they are far apart, one about two units taller. Every drawing of more than three features hides something; always ask which features were dropped.',
        setup(s) { base(s); s.view = '3d'; s.cam = Object.assign(CAM3(), { yaw: 1.0 }); s.rise = 1; s.pairT = 1; s.cross = null; s.dimT = 0; },
        at(s, t) {
          const down = t < 0.5 ? seg(t, 0, 0.3) : 1 - seg(t, 0.55, 0.9);
          LA.cam.tween(s, { pitch: 0.42 }, { pitch: lerp(0.42, 1.5, down) }, 1);
          s.rise = 1 - down;
          s.cross = t > 0.3 && t < 0.55 ? 1 : null; s.dimT = seg(t, 0.9, 1);
        } },

      { title: 'The whole row at once: a strip', duration: 3,
        narration: 'To see all four features of one flower at once, draw its row as a strip of four colored cells. The strip is the honest full picture of one vector; the scatter plot is a shadow of many. In the playground, hover any point to see its strip.',
        setup(s) { base(s); s.showStrip = 0; },
        at(s, t) { s.showStrip = seg(t, 0.1, 0.7, 'out'); } },

      { title: 'An image is a vector too', duration: 6,
        narration: 'A 28 × 28 picture of a digit — synthetic, drawn by a program — has 784 pixel brightnesses. Read them row by row and lay them in one long strip: that is called to <b>flatten</b> the image. Nothing is lost; the strip is the image written as a vector with 784 components.',
        setup(s) { base(s); s.scene = 'digit'; s.cam = CAMD(); s.flatten = 0; },
        at(s, t) { s.flatten = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'One point in ℝ⁷⁸⁴', duration: 5,
        narration: 'That strip is one point in a space with 784 axes. We cannot draw the axes, so we draw a marker and label it honestly; two other digits become two other markers. Every formula from this part — length, distance, dot product, mean — works on the strip unchanged. The picture stops at 3. The formula does not.',
        setup(s) { base(s); s.scene = 'digit'; s.cam = CAMD(); s.flatten = 1; s.collapse = 0; s.ghostT = 0; s.ghostC = 0; },
        at(s, t) { s.ghostT = seg(t, 0, 0.25); s.collapse = seg(t, 0.3, 0.75); s.ghostC = seg(t, 0.4, 0.85); } },

      { title: 'The mean is a vector', duration: 4,
        narration: 'Average the blue flowers: add their thirty vectors component by component and divide by 30. The result is the <b>mean vector</b>, the blue cloud’s center. It is a point, not a number — a full four-cell strip, drawn beside the clouds; the rose mean follows. Centering, subtracting the mean from every row, is the first step of PCA in Part 4.',
        setup(s) { base(s); s.showMeans = true; s.meanT = 0; },
        at(s, t) { s.meanT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Words and colors as vectors', duration: 4,
        narration: 'Categories become vectors with a <b>one-hot vector</b>: a 1 in the slot for “cat”, zeros elsewhere. A bag of words counts words into a strip. A color is a point in a cube with axes red, green, blue. Whenever something can be listed as numbers, it is a vector, and this course applies.',
        setup(s) { base(s); s.scene = 'words'; s.cam = CAMW(); s.wordsT = 0; },
        at(s, t) { s.wordsT = seg(t, 0, 0.9, 'linear'); } },

      { title: 'A classifier is a map on these points', duration: 5, mlTie: true,
        narration: (s) => 'Once data are points, a classifier is a rule that splits the room. In the length–width view a single gold line, perpendicular to a purple arrow <span class="c-w">w</span>, already separates ' + (accuracy(s, lineTheta(s)) === 60 ? '' : 'most ') + 'blue from rose: ' + accuracy(s, lineTheta(s)) + ' of 60 correct. Two lessons from now you build exactly this line from a weight vector and a dot product.',
        setup(s) { base(s); s.showLine = true; s.lineAngle = 0; },
        at(s, t) { s.lineAngle = lerp(0, lineTheta(s), seg(t, 0.1, 0.8)); } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'Every data row is a feature vector; its scatter plot is a shadow on two or three chosen axes, and other axes may be hidden. A strip shows one vector whole. An image flattens to a 784-strip, one point in ℝ⁷⁸⁴. The mean of vectors is a vector. Categories, counts and colors are vectors too.',
        cards: ['<b>Row = feature vector.</b> A scatter plot is its shadow on the axes you chose; always name the ones you dropped.', '<b>Image → strip → point.</b> 784 pixels read row by row are one point in ℝ⁷⁸⁴: not drawable, fully computable.', '<b>The mean of vectors is a vector.</b> Add component by component and divide by the count: the cloud’s center.'],
        setup(s) { base(s); s.scene = 'recap'; s.cam = CAMW(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Choose which features are the axes, orbit the 3D view, size the points by weight, or switch to the digit strip. Hover any point to see its whole row as a strip.</p>',
      setup(s) { base(s); s.showMeans = true; s.meanT = 1; s.showStrip = 1; },
      controls: [
        { type: 'preset', label: 'Axes (2D)', items: PAIRS.map(pr => ({ label: FS[pr[0]] + ', ' + FS[pr[1]], apply(s) { s.scene = 'flowers'; s.view = '2d'; s.cam = CAM2(); s.axesFrom = pr; s.axesTo = pr; s.slide = 1; } })) },
        { type: 'preset', label: 'View', items: [
          { label: '2D', apply(s) { s.scene = 'flowers'; s.view = '2d'; s.cam = CAM2(); } },
          { label: '3D (f₁, f₂, f₃)', apply(s) { s.scene = 'flowers'; s.view = '3d'; s.cam = CAM3(); s.rise = 1; } },
        ] },
        { type: 'toggle', bind: 'sizeBy4', label: 'Point size = weight (f₄)' },
        { type: 'toggle', bind: 'colorBy', label: 'Color by species' },
        { type: 'slider', bind: 'hi', min: 0, max: 59, step: 1, label: 'Row shown as a strip', color: 'gold', fmt: x => 'x' + sub(x + 1) },
        { type: 'preset', label: 'Scene', items: [
          { label: 'Flowers', apply(s) { s.scene = 'flowers'; s.view = '2d'; s.cam = CAM2(); } },
          { label: 'Digit → strip', apply(s) { s.scene = 'digit'; s.view = '2d'; s.cam = CAMD(); s.flatten = 1; s.collapse = 0; s.ghostT = 0; s.ghostC = 0; s.pix = null; } },
          { label: 'Words & colors', apply(s) { s.scene = 'words'; s.view = '2d'; s.cam = CAMW(); s.wordsT = 1; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the gold marker to the mean of the rose points in the length–width view.',
        setup(s) { base(s); s.showMarker = true; s.marker = [1, 4]; },
        judge(s) { return LA.judge.near(s.marker, [s.meanB[0], s.meanB[1]], LA.TOL.pos); },
        hint: 'Aim for the middle of the rose cloud; the mean balances the points on every side.',
        solution(s) { s.marker = [s.meanB[0], s.meanB[1]]; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold ring onto the axis pair that separates the two species best.',
        setup(s) { base(s); s.showPick = true; let k = 0; s.pairScores.forEach((v, i) => { if (v < s.pairScores[k]) k = i; }); s.pick = k; s.axesFrom = PAIRS[k]; s.axesTo = PAIRS[k]; },
        judge(s) { const best = Math.max.apply(null, s.pairScores), v = s.pairScores[s.pick]; return { ok: v >= 0.9 * best, progress: Math.max(0, Math.min(1, v / best)) }; },
        hint: 'Try each pair and compare the separation readout with the best value shown beside it; you need a pair within 10% of the best.',
        solution(s) { let k = 0; s.pairScores.forEach((v, i) => { if (v > s.pairScores[k]) k = i; }); s.pick = k; s.axesFrom = PAIRS[k]; s.axesTo = PAIRS[k]; } },
      { kind: 'entry',
        prompt: 'The gold pixel sits in row 10, column 5 of the image. Which cell of the 784-strip holds it? Count from 1, like the subscripts in x₁ … x₇₈₄; a blue tick marks the cell you type.',
        bind: 'pixIndex', answer: [257], tol: LA.TOL.entry, step: 1, label: 'cell', colors: 'b',
        setup(s) { base(s); s.scene = 'digit'; s.cam = CAMD(); s.flatten = 1; s.pix = [9, 4]; s.pixIndex = [0]; },
        hint: 'Rows come first: nine full rows of 28 cells come before it, then it is the 5th cell of row 10: 9·28 + 5.' },
    ],
  });
})();
