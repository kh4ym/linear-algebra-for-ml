/* p0-01 — Why linear algebra? Rows are points, models are maps. (Part 0 primer) */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp;

  // The shared Part 0 table: size (×10 m²), rooms, price (×100 k). Synthetic, hard-coded, used again in p1-01 / p1-07.
  const ROWS = [[3.0, 1, 2.0], [4.5, 2, 3.1], [6.0, 3, 4.2], [2.0, 1, 1.5], [5.0, 2, 3.6], [7.0, 4, 5.0], [3.5, 2, 2.4], [5.5, 3, 4.0]];
  const COL = { name: ['size', 'rooms', 'price'], color: ['i', 'k', 'j'], range: [8, 5, 6] };
  const f1 = x => LA.fmt.fixed(x, 1);
  const ones = () => [1, 1, 1, 1, 1, 1, 1, 1];
  const cam2d = () => ({ center: [4.86, 4.5], halfH: 7 });                                   // plot bottom-left, table top-right
  const cam3d = o => Object.assign(LA.cam.defaults3d(), { target: [3.7, 2.4, 2.9], dist: 25, yaw: -0.9, pitch: 0.42 }, o || {});
  const camFit = () => ({ center: [4.2, 3.2], halfH: 6.2 });                                 // trailer vignette 0 (2D, no table)
  const xyz = r => [r[0], r[2], r[1]];                        // world: x = size, y = price, z = rooms (the 2D plot is the floor)
  const MEAN = N.mean(ROWS.map(xyz));
  const PC1 = N.pca(ROWS.map(xyz), 1).components[0];           // the cloud's long axis (trailer vignette 1)
  const RINGS = LA.rng.circles(12, 80, { r1: 0.5, r2: 1.5, noise: 0.05 });   // synthetic rings (trailer vignette 3)
  const fitOf = (rows, a, b) => N.lstsq(rows.map(r => [1, r[a]]), rows.map(r => r[b]));   // [intercept, slope]
  const bez = (p, c, q, u) => [0, 1].map(k => (1 - u) * (1 - u) * p[k] + 2 * (1 - u) * u * c[k] + u * u * q[k]);
  const PI = Math.PI;

  /** Assign camera fields; when the user has taken over (auto === false) only re-frame (target/dist/center/halfH). */
  function setCam(s, c) {
    s.cam = s.cam || {};
    if (s.cam.auto === false) { ['target', 'dist', 'center', 'halfH'].forEach(k => { if (c[k] !== undefined) s.cam[k] = c[k]; }); return; }
    Object.assign(s.cam, c);
  }
  /** Screen-space translucent rectangle (via the HUD bar primitive). */
  function srect(g, x, y, w, h, color, alpha) { g.bars([1], { screen: true, x, y, w, h, gap: 0, color, alpha, max: 1 }); }
  function tableLayout(W) { const x0 = W - 172; return { x0, y0: 64, rowH: 21, w: 166, cx: [x0 + 18, x0 + 62, x0 + 108, x0 + 156] }; }

  /* ---------------- drawing ---------------- */
  function drawAxes2d(g, a, b) {
    const Rx = COL.range[a], Ry = COL.range[b];
    for (let i = 1; i <= Rx; i++) g.line([i, 0], [i, Ry], { color: 'ghost', width: 1 });
    for (let j = 1; j <= Ry; j++) g.line([0, j], [Rx, j], { color: 'ghost', width: 1 });
    g.vector([0, 0], [Rx + 0.8, 0], { color: COL.color[a], width: 2, head: 10, label: COL.name[a], dx: 6, dy: 16, labelFont: 'ui' });
    g.vector([0, 0], [0, Ry + 0.8], { color: COL.color[b], width: 2, head: 10, label: COL.name[b], dx: 0, dy: -14, labelFont: 'ui' });
    for (let i = 1; i <= Rx; i++) { g.line([i, -0.08], [i, 0.08], { color: 'tick' }); g.label(String(i), [i, 0], { color: 'tick', size: 11, anchor: 'below', dy: 4, font: 'ui', bg: false }); }
    for (let j = 1; j <= Ry; j++) { g.line([-0.08, j], [0.08, j], { color: 'tick' }); g.label(String(j), [0, j], { color: 'tick', size: 11, anchor: 'right', dx: -7, font: 'ui', bg: false }); }
  }

  function draw2d(g, s, ctx) {
    const a = s.axes[0], b = s.axes[1], Rx = COL.range[a];
    drawAxes2d(g, a, b);
    const fitAlpha = s.showFit ? 0.9 : s.fit;
    if (fitAlpha > 0.01) {
      const w = fitOf(s.rows, a, b);
      g.line([0.3, w[0] + w[1] * 0.3], [Rx - 0.2, w[0] + w[1] * (Rx - 0.2)], { color: 'gold', dashed: true, width: 2, alpha: fitAlpha });
      if (s.showFit) g.label('fit', [1.0, w[0] + w[1] * 1.0], { color: 'gold', size: 12, anchor: 'above', dy: -5, font: 'ui' });
    }
    for (let i = 0; i < 8; i++) {
      if (i >= s.placed || i === s.hiddenRow) continue;
      const p = [s.rows[i][a], s.rows[i][b]], hl = i === s.highlight;
      g.point(p, { r: hl ? 6.5 : 5, color: 'v', alpha: s.rowAlpha[i] });
      if (hl) g.label(String(i + 1), p, { color: 'v', size: 12, dx: 11, dy: -11, font: 'ui' });
      if (ctx.mode === 'play') g.handle('p' + i, p, (q, st) => { st.rows[i][a] = LA.clamp(q[0], 0, COL.range[a] + 0.5); st.rows[i][b] = LA.clamp(q[1], 0, COL.range[b] + 0.5); }, { color: 'v', r: 12 });
    }
    if (s.highlight >= 0 && s.highlight < s.placed && s.dims > 0.01) {
      const r = s.rows[s.highlight], p = [r[a], r[b]];
      g.vector([0, 0], [p[0], 0], { color: COL.color[a], dashed: true, width: 1.6, head: 8, alpha: s.dims });
      g.vector([p[0], 0], p, { color: COL.color[b], dashed: true, width: 1.6, head: 8, alpha: s.dims });
      if (s.dims > 0.5) { g.dim([0, 0], [p[0], 0], { label: f1(p[0]), color: COL.color[a], offset: 28 }); g.dim([p[0], 0], p, { label: f1(p[1]), color: COL.color[b], offset: 22 }); }
    }
    if (s.flight && !g.is3d) {
      const i = s.flight.row, L = tableLayout(ctx.size.w);
      const st = g.P.toWorld(L.cx[1] - 12, L.y0 + 24 + i * L.rowH);
      const en = [s.rows[i][a], s.rows[i][b]];
      const c = [(st[0] + en[0]) / 2, Math.max(st[1], en[1]) + 1.2];
      g.point(bez(st, c, en, s.flight.u), { r: 5.5, color: 'v' });
    }
    if (s.marker) {
      g.point(s.marker, { r: 6, color: 'v', label: 'row 5?', dx: 14, dy: -12 });
      g.marker.ring(s.marker, { r: 12, color: 'v' });
      g.handle('marker', s.marker, (q, st) => { st.marker = [q[0], q[1]]; }, { color: 'v' });
    }
  }

  function drawAxes3d(g) {
    const Rx = 8, Ry = 6, Rz = 5;
    for (let i = 0; i <= Rx; i++) g.line([i, 0, 0], [i, Ry, 0], { color: 'grid', width: 1, alpha: 0.75 });
    for (let j = 0; j <= Ry; j++) g.line([0, j, 0], [Rx, j, 0], { color: 'grid', width: 1, alpha: 0.75 });
    g.vector([0, 0, 0], [Rx + 0.8, 0, 0], { color: 'i', width: 2, head: 10, label: 'size', labelFont: 'ui' });
    g.vector([0, 0, 0], [0, Ry + 0.8, 0], { color: 'j', width: 2, head: 10, label: 'price', labelFont: 'ui' });
    const topDown = g.P && g.P.forward && Math.abs(g.P.forward[2]) > 0.8;   // the blue axis is (nearly) a dot: no z ticks
    g.vector([0, 0, 0], [0, 0, Rz + 0.6], { color: 'k', width: 2, head: 10, label: topDown ? undefined : 'rooms', labelFont: 'ui' });
    if (topDown) g.label('rooms: toward you', [0, 0, 0], { color: 'k', size: 12, anchor: 'left', dx: 4, dy: 36, font: 'ui' });
    for (let i = 2; i <= Rx; i += 2) g.label(String(i), [i, -0.45, 0], { color: 'tick', size: 10.5, font: 'ui', bg: false });
    for (let j = 2; j <= Ry; j += 2) g.label(String(j), [-0.5, j, 0], { color: 'tick', size: 10.5, font: 'ui', bg: false });
    if (!topDown) for (let k = 1; k <= 4; k++) { g.line([0, 0, k], [0.15, 0, k], { color: 'tick' }); g.label(String(k), [0, 0, k], { color: 'tick', size: 10.5, font: 'ui', anchor: 'right', dx: -6, bg: false }); }
  }

  function draw3d(g, s, ctx) {
    drawAxes3d(g);
    for (let i = 0; i < 8; i++) {
      if (i >= s.placed || i === s.hiddenRow) continue;
      const r = s.rows[i], z = r[1] * s.lift, p = [r[0], r[2], z], al = s.rowAlpha[i];
      if (s.lift > 0 && s.drop === 0) g.line([r[0], r[2], 0], p, { color: 'k', dashed: true, width: 1.2, alpha: 0.8 * al });
      if (s.drop > 0) {
        const sh = [r[0], r[2], z * (1 - s.drop)];
        g.line(p, sh, { color: 'residual', dashed: true, width: 1.2, alpha: al });
        g.point(p, { r: 5, color: 'v', alpha: 0.3 * al });
        g.point(sh, { r: 4.5, color: 'mag', alpha: al });
      } else {
        g.point(p, { r: i === s.highlight ? 6.5 : 5, color: 'v', alpha: al });
        if (i === s.highlight) g.label(String(i + 1), p, { color: 'v', size: 12, dx: 11, dy: -11, font: 'ui' });
      }
      if (ctx.mode === 'play') g.handle('p' + i, p, (q, st) => { st.rows[i][0] = LA.clamp(q[0], 0, 8.5); st.rows[i][2] = LA.clamp(q[1], 0, 6.5); }, { color: 'v', r: 12, plane: 'xy' });
    }
    if (s.highlight >= 0 && s.highlight < s.placed && s.dims > 0.01 && s.drop === 0) {
      const r = s.rows[s.highlight], z = r[1] * s.lift, al = s.dims;
      g.line([0, 0, 0], [r[0], 0, 0], { color: 'i', dashed: true, width: 1.6, alpha: al });
      g.line([r[0], 0, 0], [r[0], r[2], 0], { color: 'j', dashed: true, width: 1.6, alpha: al });
      g.line([r[0], r[2], 0], [r[0], r[2], z], { color: 'k', dashed: true, width: 1.6, alpha: al });
      if (al > 0.5) {
        g.label(f1(r[0]), [r[0] / 2, 0, 0], { color: 'i', size: 12, anchor: 'below', dy: 6 });
        g.label(f1(r[2]), [r[0], r[2] / 2, 0], { color: 'j', size: 12, anchor: 'left', dx: 8 });
        g.label(String(r[1]), [r[0], r[2], z / 2], { color: 'k', size: 12, anchor: 'left', dx: 8 });
      }
    }
    if (s.shadowLabel) g.hud('shadow of 3 columns on 2 · lost: rooms', 16, ctx.size.h - 22, { color: 'mag', size: 13 });
    if (s.showProbe) {
      const q = [s.probe[0], s.probe[2], s.probe[1]];
      const near = N.dist(q, xyz(ROWS[2])) <= 0.1;
      g.line([0, 0, 0], [q[0], 0, 0], { color: 'i', dashed: true, width: 1.4 });
      g.line([q[0], 0, 0], [q[0], q[1], 0], { color: 'j', dashed: true, width: 1.4 });
      g.line([q[0], q[1], 0], q, { color: 'k', dashed: true, width: 1.4 });
      g.point(q, { r: 6, color: near ? 'gold' : 'b', label: 'probe', dx: 12, dy: -12 });
      if (near) { g.marker.ring(q, { r: 13, color: 'gold' }); g.marker.check(q, { dx: 26, dy: 10 }); }
    }
  }

  function drawTable(g, s, ctx) {
    const L = tableLayout(ctx.size.w);
    srect(g, L.x0 - 6, L.y0 - 16, L.w, 24 + 8 * L.rowH + 10, 'panel', 0.92);
    const heads = ['#', 'size', 'rooms', 'price'], hc = ['muted', 'i', 'k', 'j'];
    heads.forEach((h, k) => g.label(h, [L.cx[k], L.y0], { screen: true, anchor: 'right', color: hc[k], size: 12, font: 'ui', bold: true, bg: false }));
    const dimPlaced = s.placed < 8;
    for (let i = 0; i < 8; i++) {
      const y = L.y0 + 24 + i * L.rowH;
      const al = s.rowAlpha[i] * (dimPlaced && i < s.placed && i !== s.highlight ? 0.45 : 1);
      if (i === s.highlight) srect(g, L.x0 - 3, y - 10, L.w - 6, 20, 'v', 0.16);
      const r = s.rows[i], vals = [String(i + 1), f1(r[0]), String(r[1]), f1(r[2])];
      vals.forEach((v, k) => g.label(v, [L.cx[k], y], { screen: true, anchor: 'right', color: k === 0 ? 'muted' : 'text', size: 12.5, font: 'mono', bg: false, alpha: al }));
    }
    g.hud('synthetic data · 8 apartments', L.x0 - 4, L.y0 + 24 + 8 * L.rowH + 10, { size: 11, color: 'muted' });
  }

  function drawTrailer(g, s, ctx) {
    const v = s.trailer.v, u = s.trailer.u, W = ctx.size.w, H = ctx.size.h;
    if (v === 0) {                                   // fit: the gold line sweeps to the least-squares slope
      drawAxes2d(g, 0, 2);
      const w = fitOf(ROWS, 0, 2), m = w[1] * seg(u, 0.1, 0.75), mx = MEAN[0], my = MEAN[1];
      const line = x => my + m * (x - mx);
      const ra = seg(u, 0.3, 0.7);
      ROWS.forEach(r => { if (ra > 0) g.line([r[0], r[2]], [r[0], line(r[0])], { color: 'residual', dashed: true, width: 1.2, alpha: ra }); });
      g.line([0.4, line(0.4)], [8.4, line(8.4)], { color: 'gold', width: 2.5 });
      g.label('fit', [8.4, line(8.4)], { color: 'gold', size: 12, anchor: 'left', dx: 6, font: 'ui' });
      g.points(ROWS.map(r => [r[0], r[2]]), { r: 5, color: 'v' });
    } else if (v === 1) {                            // rotate: the long axis of the cloud
      drawAxes3d(g);
      ROWS.forEach(r => { const p = xyz(r); g.line([p[0], p[1], 0], p, { color: 'k', dashed: true, width: 1, alpha: 0.5 }); g.point(p, { r: 5, color: 'v' }); });
      const Lh = 3.2 * seg(u, 0.15, 0.7);
      if (Lh > 0.02) {
        const a = N.sub(MEAN, N.scale(PC1, Lh)), b = N.add(MEAN, N.scale(PC1, Lh));
        g.line(a, b, { color: 'gold', width: 3 });
        g.label('long axis', b, { color: 'gold', size: 12, dx: 10, dy: -10, font: 'ui' });
      }
      g.point(MEAN, { r: 3.5, color: 'gold' });
    } else if (v === 2) {                            // split: a teal plane between cheap and expensive
      drawAxes3d(g);
      const yP = lerp(1.0, 3.3, seg(u, 0.15, 0.7, 'back'));
      g.planes([{ point: [4.4, yP, 2.3], normal: [0, 1, 0], extentU: 4.6, extentV: 2.7, color: 'teal', alpha: 0.26, label: 'plane' }]);
      ROWS.forEach(r => { const p = xyz(r); g.line([p[0], p[1], 0], p, { color: 'k', dashed: true, width: 1, alpha: 0.5 }); g.point(p, { r: 5.5, color: r[2] < 3.3 ? 'data' : 'dataB' }); });
      g.hud('blue: cheap · rose: expensive', 16, H - 42, { size: 12, color: 'muted' });
    } else {                                         // bend: a sheet curves into a bowl; a flat gold cut separates the rings
      const bend = seg(u, 0.25, 0.7), cut = seg(u, 0.75, 0.95);
      g.grid(null, { range: 2, alpha: 0.5 });
      // a round sheet (radius 1.8) that bends into the bowl z = x² + y²
      g.paramSurface((r, a) => [r * Math.cos(a), r * Math.sin(a), bend * r * r], { u0: 0, u1: 1.8, v0: 0, v1: 2 * PI }, { res: 16, mode: 'wire', color: 'teal', alpha: 0.5 });
      RINGS.X.forEach((p, i) => g.point([p[0], p[1], bend * (p[0] * p[0] + p[1] * p[1])], { r: 4, color: RINGS.y[i] ? 'dataB' : 'data' }));
      if (cut > 0.02) { g.plane({ point: [0, 0, 1], normal: [0, 0, 1], extent: 2.1, color: 'gold', alpha: 0.3 * cut, stroke: 'gold' }); g.hud('gold: a flat cut at height 1', 16, H - 62, { size: 12, color: 'gold' }); }
      g.hud('synthetic rings', 16, H - 42, { size: 12, color: 'muted' });
    }
    const names = ['Part 3 · fit', 'Part 4 · PCA', 'Part 1 · classifier', 'Part 2 · neural layer'];
    g.hud(names[v], 16, H - 22, { size: 14, color: 'gold', bold: true });
    for (let k = 0; k < 4; k++) g.hud(k === v ? '●' : '○', W - 88 + k * 18, H - 22, { size: 12, color: k === v ? 'gold' : 'muted' });
  }

  function drawRecap(g, s, ctx) {
    const cards = [
      { title: 'Row = point, column = axis', draw(gg) {
        [[3.0, 2.0], [4.5, 3.1], [6.0, 4.2]].forEach((r, k) => { gg.label(f1(r[0]), [-4.3, 0.8 - 0.8 * k], { color: 'i', size: 13, font: 'mono', bg: false }); gg.label(f1(r[1]), [-3.4, 0.8 - 0.8 * k], { color: 'j', size: 13, font: 'mono', bg: false }); });
        gg.polyline([[-4.8, 1.3], [-4.8, -1.3]], { color: 'muted' }); gg.polyline([[-2.9, 1.3], [-2.9, -1.3]], { color: 'muted' });
        gg.vector([-2.3, 0], [-1.0, 0], { color: 'hud', width: 2, head: 9 });
        gg.vector([-0.2, -1.4], [4.6, -1.4], { color: 'i', width: 1.6, head: 8 }); gg.vector([-0.2, -1.4], [-0.2, 1.7], { color: 'j', width: 1.6, head: 8 });
        [[3.0, 2.0], [4.5, 3.1], [6.0, 4.2]].forEach(r => gg.point([-0.2 + r[0] * 0.6, -1.4 + r[1] * 0.6], { r: 5, color: 'v' }));
      } },
      { title: 'Dropping a column is a shadow; shadows lose something', draw(gg) {
        gg.line([-4.5, -1.2], [4.5, -1.2], { color: 'gridAxis', width: 1.6 });
        [[-2.6, 1.0], [-0.4, 0.3], [1.6, 1.3], [1.6, -0.2]].forEach(p => { gg.line(p, [p[0], -1.2], { color: 'residual', dashed: true, width: 1.2 }); gg.point(p, { r: 5, color: 'v', alpha: 0.45 }); gg.point([p[0], -1.2], { r: 4.5, color: 'mag' }); });
        gg.label('two points, one shadow', [3.4, -0.5], { color: 'mag', size: 11.5, font: 'ui', anchor: 'left', dx: -20 });
      } },
      { title: 'A model is a map on points', draw(gg) {
        gg.grid([[1, 0.6], [0, 1]], { range: 6, alpha: 0.55 });
        gg.lineThrough([0, -0.2], [1, 0.5], { color: 'gold', width: 2.5, len: 7 });
        [[-2.5, 0.6], [-1.2, 1.2], [0.4, 1.4], [-0.6, -0.9], [1.2, -0.6], [2.4, 0.1]].forEach((p, k) => gg.point(p, { r: 5, color: k < 3 ? 'dataB' : 'data' }));
      } },
    ];
    cards.forEach((c, k) => {
      g.viewport({ x: 0.04, y: 0.03 + k * 0.325, w: 0.92, h: 0.295 }, { view: '2d', cam: { center: [0, 0], halfH: 2.3 }, bg: 'panel' }, (gg, r) => {
        c.draw(gg);
        gg.hud(c.title, r.x + 12, r.y + 18, { size: 13, color: 'text', bold: true });
      });
    });
  }

  LA.registerLesson({
    id: 'p0-01-data-is-geometry',
    part: 0,
    title: 'Why linear algebra? Rows are points, models are maps',
    subtitle: 'A table becomes a cloud of points; every model moves points.',
    minutes: 7,
    prereqs: [],
    tags: ['primer'],
    badge: { drawnIn: '3D (8 rows × 3 columns)', realSize: 'a tabular dataset has 10–1000 columns; an image row has 784; a text embedding 768–4096', transfers: '“row = point, column = axis, model = map” is exactly true in any number of columns; only the number of axes we can draw stops at 3' },
    cantShow: ['a table with more than 3 columns as a picture (we drop columns and say so)', 'that real datasets have thousands of rows (we draw 8)'],
    terms: [
      { term: 'dataset', def: 'A table whose rows are the things we measured and whose columns are the measurements.' },
      { term: 'feature', def: 'One column of a dataset, one measurement.' },
      { term: 'point', def: 'A dot whose position is given by numbers along axes.' },
      { term: 'axis', def: 'A labeled direction along which one number is read.' },
      { term: 'model', def: 'A rule that moves or labels points.' },
      { term: 'projection', def: 'Flattening points onto fewer axes, like a shadow.' },
    ],
    atlas: ['linear-regression', 'pca', 'mlp'],
    thumbnail: 3,

    state() {
      return {
        view: '2d', cam: cam2d(),
        rows: LA.clone(ROWS), axes: [0, 2],
        placed: 0, flight: null, highlight: -1, dims: 0, rowAlpha: ones(), hiddenRow: -1,
        lift: 0, drop: 0, shadowLabel: false, fit: 0, showFit: false, showTable: true, cross: false,
        trailer: null, recap: false, marker: null, probe: [0, 0, 0], showProbe: false, panel: 'rows',
      };
    },

    render(g, s, ctx) {
      if (s.recap) { drawRecap(g, s, ctx); return; }
      if (s.trailer) { drawTrailer(g, s, ctx); return; }
      if (s.view === '3d') draw3d(g, s, ctx); else draw2d(g, s, ctx);
      if (s.showTable) drawTable(g, s, ctx);
      if (s.cross) { g.hud('“table ≠ picture”', 16, 32, { size: 14, color: 'text' }); g.marker.cross([152, 32], { screen: true, size: 20 }); }
    },

    math(s, tex) {
      const T = (t, o) => tex.text(t, o);
      const rowLine = (k) => { const r = s.rows[k]; return tex.eq(T('row ' + (k + 1) + ':'), T('size'), tex.num(f1(r[0]), { color: 'i' }), tex.op('·'), T('rooms'), tex.num(String(r[1]), { color: 'k' }), tex.op('·'), T('price'), tex.num(f1(r[2]), { color: 'j' })); };
      switch (s.panel) {
        case 'row1': { const r = s.rows[0]; return tex.block(tex.eq(T('row 1:'), T('size'), tex.num(f1(r[0]), { color: 'i' }), tex.op('·'), T('price'), tex.num(f1(r[2]), { color: 'j' }))); }
        case 'count': return tex.block(tex.eq(tex.num(8), T('rows →'), tex.num(s.placed), T(s.placed === 1 ? 'point' : 'points')));
        case 'row3': return tex.block(rowLine(2));
        case 'edit': return tex.block(tex.eq(T('row 5: price'), tex.num('3.6', { color: 'j' }), tex.op('→'), tex.num(f1(s.rows[4][2]), { color: 'j' })));
        case 'shadow': return tex.block(tex.eq(tex.op('('), T('size', { color: 'i' }), tex.op(','), T('<s>rooms</s>', { color: 'k', html: true }), tex.op(','), T('price', { color: 'j' }), tex.op(')'), tex.op('→'), tex.op('('), T('size', { color: 'i' }), tex.op(','), T('price', { color: 'j' }), tex.op(')')));
        case 'trailer': { const v = s.trailer ? s.trailer.v : -1; const parts = []; ['fit', 'rotate', 'split', 'bend'].forEach((w, k) => { if (k) parts.push(tex.op('·')); parts.push(tex.num(w, { color: k === v ? 'gold' : 'muted' })); }); return tex.block(tex.eq.apply(null, parts)); }
        case 'probe': return tex.block(tex.eq(T('probe = (size, rooms, price) ='), tex.rowvec(s.probe, { color: 'b' })));
        case 'row': return tex.block(s.highlight >= 0 ? rowLine(s.highlight) : tex.eq(T('dataset: 8 rows × 3 columns')));
        case 'none': return '';
        default: return tex.block(tex.eq(T('dataset: 8 rows × 3 columns')));
      }
    },

    steps: [
      { title: 'A table of eight apartments', duration: 3,
        narration: 'Here is a small <b>dataset</b>: eight apartments, three measurements each — size, number of rooms, price. Each column is a <b>feature</b>, one measurement. Every machine-learning problem starts as a table like this one. This course is about one idea: <b>a row of numbers is a point</b>, and a <b>model</b> is something that moves points around.',
        setup(s) { s.view = '2d'; s.cam = cam2d(); s.rows = LA.clone(ROWS); s.placed = 0; s.flight = null; s.highlight = -1; s.dims = 0; s.showTable = true; s.panel = 'rows'; },
        at(s, t) { for (let i = 0; i < 8; i++) s.rowAlpha[i] = seg(t, i * 0.09, i * 0.09 + 0.28); } },

      { title: 'One row becomes one point', duration: 3,
        narration: 'Take row 1: size 3.0, price 2.0. Walk 3.0 along the green <b>size</b> axis, then 2.0 up the red <b>price</b> axis, and put a dot there. An <b>axis</b> is a labeled direction along which one number is read. The row of the table and the dot are the same thing written two ways.',
        setup(s) { s.rowAlpha = ones(); s.highlight = 0; s.placed = 0; s.panel = 'row1'; },
        at(s, t) {
          if (t < 0.6) { s.placed = 0; s.flight = { row: 0, u: seg(t, 0.05, 0.6) }; } else { s.placed = 1; s.flight = null; }
          s.dims = seg(t, 0.62, 0.85);
        } },

      { title: 'Eight rows, eight points', duration: 4,
        narration: 'Do it for every row. The table turns into a cloud of yellow points. Look at the cloud and you already see something the table hides: bigger apartments cost more, and the points nearly line up along one gold line. A picture tells you what a column of numbers cannot.',
        setup(s) { s.highlight = -1; s.dims = 0; s.placed = 1; s.flight = null; s.panel = 'count'; },
        at(s, t) {
          const x = seg(t, 0, 0.84, 'linear') * 7, idx = Math.min(6, Math.floor(x));
          if (t < 0.84) { s.placed = 1 + idx; s.flight = { row: 1 + idx, u: seg(x - idx, 0, 1) }; } else { s.placed = 8; s.flight = null; }
          s.fit = 0.9 * seg(t, 0.84, 1);
        } },

      { title: 'A third column, a third axis', duration: 5,
        narration: 'There is a third column, rooms. Give it its own axis, the blue one pointing out of the screen toward you; then the view tilts so you can see it. Each point lifts to its rooms value along a dashed blue line. Three columns, three axes, one point per row. Drag to orbit and look around the cloud.',
        setup(s) { s.view = '3d'; s.cam = cam3d({ yaw: -PI / 2, pitch: 1.55 }); s.lift = 0; s.fit = 0; s.highlight = 2; s.dims = 0; s.showTable = false; s.panel = 'row3'; },
        at(s, t) {
          s.lift = seg(t, 0.3, 0.9);
          LA.cam.tween(s, { yaw: -PI / 2, pitch: 1.55 }, { yaw: -0.9, pitch: 0.42 }, seg(t, 0.05, 0.8));
          s.dims = seg(t, 0.85, 1);
        } },

      { title: 'Table and picture are one object', duration: 4, misconception: true,
        narration: 'A common belief: “the table is the data; the picture is just an illustration.” Test it. Change one number in the table — row 5’s price — and its point moves. Delete row 4 and a point disappears. Nothing is lost either way. The table <i>is</i> the cloud.',
        setup(s) { s.showTable = true; s._rows0 = LA.clone(s.rows); s.dims = 0; s.highlight = 4; s.panel = 'edit'; s.cross = false; },
        at(s, t) {
          s.rows[4][2] = s._rows0[4][2] + 1.0 * (seg(t, 0, 0.4) - seg(t, 0.8, 1));
          s.rowAlpha[3] = 1 - seg(t, 0.5, 0.7) + seg(t, 0.8, 1);
          s.cross = t >= 0.5;
        } },

      { title: 'Dropping a column is a shadow', duration: 4,
        narration: 'Suppose a model ignores rooms. Then it sees the cloud flattened onto the size–price floor: every point drops straight down, like a shadow — the magenta dots. Flattening is cheap and useful, and it loses something: two apartments with the same size and price but different rooms become one dot. We call this a <b>projection</b> and always say what it lost.',
        setup(s) { s.highlight = -1; s.showTable = false; s.cross = false; s.drop = 0; s.shadowLabel = false; s.panel = 'shadow'; },
        at(s, t) {
          s.drop = seg(t, 0.05, 0.5);
          LA.cam.tween(s, { yaw: -0.9, pitch: 0.42, dist: 25, target: [3.7, 2.4, 2.9] }, { yaw: -PI / 2, pitch: 1.52, dist: 19, target: [4.3, 3.0, 0] }, seg(t, 0.5, 0.95));
          s.shadowLabel = t > 0.9;
        } },

      { title: 'Four things a model can do', duration: 20, mlTie: true,
        narration: 'What will you be able to do by the end? Four previews, five seconds each. <b>Fit</b>: a gold line that predicts price from size. <b>Rotate</b>: find the direction the cloud is longest in. <b>Split</b>: a teal plane that separates cheap from expensive. <b>Bend</b>: a sheet that curves until a flat cut separates two rings. Each one is a map on points.',
        setup(s) { s.drop = 0; s.lift = 1; s.placed = 8; s.shadowLabel = false; s.view = '2d'; s.cam = camFit(); s.trailer = { v: 0, u: 0 }; s.panel = 'trailer'; },
        at(s, t) {
          const x = Math.min(3.999, t * 4), v = Math.floor(x), u = x - v;
          s.trailer = { v, u };
          if (v === 0) { s.view = '2d'; setCam(s, camFit()); }
          else if (v === 1) { s.view = '3d'; setCam(s, cam3d({ yaw: -1.15 + 1.05 * seg(u, 0, 1, 'linear'), pitch: 0.42 })); }
          else if (v === 2) { s.view = '3d'; setCam(s, cam3d({ yaw: -0.75, pitch: 0.3 })); }
          else { s.view = '3d'; setCam(s, Object.assign(LA.cam.defaults3d(), { target: [0, 0, 1.1], dist: 11, yaw: -0.9, pitch: lerp(1.5, 0.45, seg(u, 0.15, 0.55)) })); }
        } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Three things to keep. A row of a table is a point; the columns are axes. Dropping a column is a shadow, and a shadow loses something. A model is a map that moves, rotates, splits or bends the cloud of points. Everything else in this course is how to write such maps down.',
        cards: ['<b>Row = point, column = axis.</b> The table and the cloud are one object.', '<b>Dropping a column is a shadow</b> — and a shadow loses something; we always say what.', '<b>A model is a map on points:</b> fit, rotate, split or bend the cloud.'],
        setup(s) { s.trailer = null; s.recap = true; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag any yellow point and watch its row change; slide a price and watch the point move. Switch axes, or lift the cloud into 3D (drag to orbit).</p>',
      setup(s) { s.recap = false; s.trailer = null; s.placed = 8; s.lift = 1; s.drop = 0; s.showTable = true; s.showFit = true; s.view = '2d'; s.cam = cam2d(); s.highlight = -1; s.dims = 1; s.panel = 'row'; },
      controls: [
        { type: 'preset', label: 'View', items: [
          { label: 'size – price', apply(s) { s.view = '2d'; s.cam = cam2d(); s.axes = [0, 2]; s.drop = 0; s.shadowLabel = false; } },
          { label: 'size – rooms', apply(s) { s.view = '2d'; s.cam = cam2d(); s.axes = [0, 1]; s.drop = 0; s.shadowLabel = false; } },
          { label: 'rooms – price', apply(s) { s.view = '2d'; s.cam = cam2d(); s.axes = [1, 2]; s.drop = 0; s.shadowLabel = false; } },
          { label: 'all three (3D)', apply(s) { s.view = '3d'; s.cam = cam3d(); s.lift = 1; s.drop = 0; s.shadowLabel = false; } },
          { label: 'shadow (drop rooms)', apply(s) { s.view = '3d'; s.cam = cam3d({ pitch: 0.5 }); s.lift = 1; s.drop = 1; s.shadowLabel = true; } },
        ] },
        { type: 'toggle', bind: 'showTable', label: 'Show the table' },
        { type: 'toggle', bind: 'showFit', label: 'Show the fit line (2D)' },
        { type: 'preset', label: 'Highlight row', items: [1, 2, 3, 4, 5, 6, 7, 8].map(k => ({ label: String(k), apply(s) { s.highlight = k - 1; s.dims = 1; } })).concat([{ label: 'none', apply(s) { s.highlight = -1; } }]) },
        { type: 'slider', bind: 'rows[4][2]', min: 0, max: 6, step: 0.1, label: 'Price of row 5 (×100 k)', color: 'j', fmt: f1 },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow marker to where row 5 belongs in the size–price picture.',
        setup(s) { s.view = '2d'; s.cam = cam2d(); s.axes = [0, 2]; s.placed = 8; s.lift = 0; s.drop = 0; s.hiddenRow = 4; s.highlight = 4; s.dims = 0; s.showTable = true; s.marker = [1, 6]; s.panel = 'row'; },
        judge(s) { return LA.judge.near(s.marker, [5.0, 3.6], LA.TOL.pos); },
        hint: 'Read size along the green axis first, then price up the red one.',
        solution(s) { s.marker = [5.0, 3.6]; } },
      { kind: 'manipulate',
        prompt: 'Orbit (drag on the picture) until you look straight down the blue rooms axis, so the picture becomes the size–price plot.',
        setup(s) { s.view = '3d'; s.cam = cam3d({ yaw: 0.25, pitch: 0.5 }); s.placed = 8; s.lift = 1; s.drop = 0; s.hiddenRow = -1; s.highlight = -1; s.showTable = false; s.marker = null; s.panel = 'none'; },
        judge(s) { return LA.judge.lookAlong(s.cam, [0, 0, 1], 2 * LA.TOL.angleDeg); },
        hint: 'Drag the view until the blue axis shrinks to a dot.',
        solution(s) { s.cam = LA.cam.lookAlong(s.cam, 'z'); } },
      { kind: 'entry',
        prompt: 'Type the three numbers of row 3 as a point (size, rooms, price). The orange probe moves as you type.',
        bind: 'probe', answer: [6.0, 3, 4.2], tol: LA.TOL.entry, label: 'probe', colors: ['b'],
        setup(s) { s.view = '3d'; s.cam = cam3d(); s.placed = 8; s.lift = 1; s.drop = 0; s.hiddenRow = -1; s.highlight = -1; s.showTable = true; s.marker = null; s.probe = [0, 0, 0]; s.showProbe = true; s.panel = 'probe'; } },
    ],
  });
})();
