/* p6-01 — Batches, shapes, tensors, broadcasting; two faces of a matrix. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, C = LA.color, lerp = LA.lerp, lerpV = LA.lerpV;
  const clamp01 = LA.clamp01;
  const PX = 0.4;                                   // left 40 % of the canvas: tables; right: the plane / 3D scene
  const CW = 42, CH = 22;                           // table cell size (px)
  const W0 = [[1, 0.5], [-0.5, 1], [0.8, -0.6]];
  const B0 = [0.4, -0.2, 0.3];
  const I2 = () => [[1, 0], [0, 1]];
  const SUBS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
  const sub = k => String(k).split('').map(d => SUBS[+d]).join('');
  const xv = '<span class="c-v">x</span>', bv = '<span class="c-b">b</span>';

  // ---------- small screen-space helpers (tables, rectangles, shape tiles) ----------
  const rampDiv = t => C.diverge(2 * t - 1);
  function rect(g, x, y, w, h, fill) { g.strip([0], { x, y, w, h, range: [0, 1], ramp: () => fill }); }
  /** Matrix M as a heat table at screen (x, y). o: range, cw, ch, title, titleColor, colColors, rowLabels, hiRow, upto (rows shown filled), textColor */
  function table(g, M, x, y, o) {
    o = o || {};
    const cw = o.cw || CW, ch = o.ch || CH, R = o.range || 2.5, rows = M.length;
    for (let i = 0; i < rows; i++) {
      const row = M[i];
      const filled = o.upto === undefined || i < o.upto;
      if (filled) g.strip(row, { x, y: y + i * ch, w: cw * row.length, h: ch, range: [-R, R], ramp: rampDiv });
      else rect(g, x, y + i * ch, cw * row.length, ch, C.alpha(C.panel, 1));
      if (filled) row.forEach((v, j) => g.label(LA.fmt(v), [x + (j + 0.5) * cw, y + (i + 0.5) * ch], { screen: true, size: 11, bg: false, font: 'ui', color: o.colColors ? o.colColors[j] : (o.textColor || 'text') }));
      if (o.rowLabels) g.label(String(i + 1), [x - 8, y + (i + 0.5) * ch], { screen: true, size: 10, bg: false, font: 'ui', color: o.hiRow === i ? 'v' : 'muted', anchor: 'right' });
    }
    if (o.hiRow !== undefined && o.hiRow !== null && o.hiRow < rows) {
      const w = cw * M[0].length;
      g.strip([0, 0], { x: x - 2, y: y + o.hiRow * ch - 2, w: w + 4, h: 3, range: [0, 1], ramp: () => C.v });
      g.strip([0, 0], { x: x - 2, y: y + (o.hiRow + 1) * ch - 1, w: w + 4, h: 3, range: [0, 1], ramp: () => C.v });
    }
    if (o.title) g.label(o.title, [x, y - 12], { screen: true, size: 12.5, bg: false, font: 'ui', anchor: 'left', color: o.titleColor || 'hud' });
    return { w: cw * (M[0] ? M[0].length : 1), h: ch * rows };
  }
  function txt(g, s, x, y, o) { g.label(s, [x, y], Object.assign({ screen: true, size: 13, bg: false, font: 'ui', anchor: 'left', color: 'hud' }, o || {})); }
  /** A shape tile: an r×c rectangle (k px per unit, or a size function) with its shape written under it. */
  function tile(g, x, y, r, c, o) {
    o = o || {};
    const sz = o.size || (n => n * (o.k || 13));
    const w = o.w !== undefined ? o.w : sz(c), h = o.h !== undefined ? o.h : sz(r);
    rect(g, x, y - h / 2, w, h, C.alpha(C.role(o.color || 'teal'), o.alpha === undefined ? 0.35 : o.alpha));
    txt(g, o.label || (r + '×' + c), x + w / 2, y + h / 2 + 12, { anchor: 'center', size: 12, color: o.labelColor || 'text', font: 'mono' });
    return { x, y, w, h };
  }
  /** Lay out tiles left→right with a gap; returns their boxes. */
  function chain(g, x, y, dims, gap, o) {
    const out = []; let cx = x;
    dims.forEach((d, i) => { const b = tile(g, cx, y, d[0], d[1], Object.assign({}, o, d[2] || {})); out.push(b); cx += b.w + (i < dims.length - 1 ? gap : 0); });
    return out;
  }

  // ---------- data ----------
  function makeData() {
    const X12 = LA.rng.gaussianCloud(601, 12, { mean: [0.5, 0.2], cov: [[1, 0.4], [0.4, 0.6]] }).map(p => p.map(v => Math.round(v * 10) / 10));
    const imgs = [];
    for (let k = 0; k < 4; k++) { const chs = []; for (let c = 0; c < 3; c++) chs.push(LA.rng.smallImage(602 + 10 * k + c, { size: 6 }).map(r => r.map(v => Math.round(v * 100) / 100))); imgs.push(chs); }
    return { X12, imgs };
  }
  const A_EX = [[1, 0, 0.5], [0, 1, -0.5]], B_EX = [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 1]];   // check-1 example chain

  function Wuse(s) { return s.useGuess ? [s.guessW[0], s.guessW[1], [0, 0]] : s.W; }
  function Xn(s) { return s.X12.slice(0, s.n); }
  /** output of one sample under the current layer, as a 3-vector */
  function outOf(s, x) {
    const W = Wuse(s), o = [0, 0, 0];
    for (let i = 0; i < s.h; i++) o[i] = W[i][0] * x[0] + W[i][1] * x[1] + (s.useBias ? s.b[i] * s.biasU : 0);
    o[2] *= s.liftU;
    if (s.useBias && s.h === 3) o[2] += 0;   // bias already included above (scaled with lift on purpose)
    return o;
  }
  function progressOf(s, k) { return s.fill >= 0 ? clamp01(s.fill - k) : s.mapU; }
  function posOf(s, k) { const x = s.X12[k]; return lerpV([x[0], x[1], 0], outOf(s, x), progressOf(s, k)); }
  function Hmat(s, withBias) {
    const W = Wuse(s);
    return Xn(s).map(x => { const r = []; for (let i = 0; i < s.h; i++) r.push(W[i][0] * x[0] + W[i][1] * x[1] + (withBias ? s.b[i] : 0)); return r; });
  }
  function meanRows(M) { const m = M[0].map(() => 0); M.forEach(r => r.forEach((v, j) => { m[j] += v / M.length; })); return m; }
  const Wt = (s) => N.transpose(Wuse(s).slice(0, s.h));
  const camPlane = () => ({ center: [0.45, 0.25], halfH: 2.9 });
  const camLift = () => Object.assign(LA.cam.defaults3d(), { target: [0.35, 0.1, 0.25], dist: 8.6, yaw: -1.05, pitch: 0.5, fov: 36 });

  LA.registerLesson({
    id: 'p6-01-batches-shapes-tensors',
    part: 6,
    title: 'Batches, shapes, tensors, broadcasting; two faces of a matrix',
    subtitle: 'Rows are samples; one multiplication by Wᵀ moves the whole batch; shapes must chain.',
    minutes: 10,
    prereqs: ['p2-02-matrix-vector', 'p2-09-non-square', 'p2-10-transpose-outer-symmetric', 'p2-11-affine-bias', 'p1-07-data-as-vectors'],
    tags: ['core'],
    badge: { drawnIn: '2D data (d = 2) lifted to h ≤ 3 outputs in 3D', realSize: 'X is 60,000×784 for a digit set; a transformer activation tensor is (32, 2048, 4096)', transfers: 'XWᵀ row by row, broadcasting as the outer product 1bᵀ, matmul on the last axis only — exact at any size' },
    cantShow: ['the 784 feature columns of a real X; we draw 2', 'more than 3 output units per point (the 8×4 output in check 1 is a table, not a picture)', 'batch × time × feature at once beyond a few stacked 2D slices'],
    terms: [
      { term: 'batch', def: 'Several samples processed together as the rows of one matrix.' },
      { term: 'data matrix', def: 'The table X with n rows (samples) and d columns (features).' },
      { term: 'shape', def: 'The list of sizes along each axis of an array, e.g. (n, d).' },
      { term: 'broadcasting', def: 'Adding one row vector to every row of a matrix; the hidden array is the outer product 1bᵀ.' },
      { term: 'tensor', def: 'In ML, simply an array with more than two indices, e.g. (batch, height, width, channels).' },
      { term: 'reduction', def: 'Summing or averaging along one axis, which removes that axis from the shape.' },
      { term: 'row-major flatten', def: 'Reading an array row by row into one long strip.' },
    ],
    atlas: ['tensors', 'mlp', 'attention'],
    thumbnail: 3,

    state() {
      const d = makeData();
      return {
        view: '2d', cam: camPlane(),
        X12: d.X12, imgs: d.imgs, n: 8, W: LA.clone(W0), h: 2, b: B0.slice(), useBias: false, biasU: 0,
        mapU: 0, fill: -1, liftU: 0, gridU: 0, showGrid: false, ptsA: 0, hiRow: 2, showRaw: false,
        scene: 'points', panel: 'X', u: 0, mathMode: 'X', recipe: 0, showMean: 0,
        tiles: null, guessW: [[1, 0], [0, 1]], useGuess: false, checkMode: null,
      };
    },

    render(g, s, ctx) {
      const S = g.size;
      let info = {};
      if (s.scene === 'points') {
        g.viewport({ x: PX, y: 0, w: 1 - PX, h: 1 }, { view: s.view, cam: s.cam, border: false }, (gg) => { info = drawPoints(gg, s, ctx); });
        g.strip([0], { x: PX * S.w - 1, y: 0, w: 1, h: S.h, range: [0, 1], ramp: () => C.line });
      } else if (s.scene === 'tensor') {
        g.viewport({ x: 0, y: 0, w: 1, h: 0.78 }, { view: '3d', cam: s.cam, border: false }, (gg) => drawTensor(gg, s));
      } else if (s.scene === 'tiles') {
        drawTileCheck(g, s);
      }
      drawPanel(g, s, info);
    },

    math(s, tex) {
      const R = (r, c) => tex.text('ℝ' + sup(r) + 'ˣ' + sup(c));
      const W2 = s.W.slice(0, 2);
      switch (s.mathMode) {
        case 'X': return tex.block(tex.eq(tex.sym('X'), tex.op('∈'), R(s.n, 2)), tex.eq(tex.text('rows = samples, columns = features')));
        case 'W': return tex.block(tex.eq(tex.sym('W'), tex.op('='), tex.mat(W2, { colors: 'cols' })), tex.eq(tex.text('(a map)')));
        case 'Wx': { const x = s.X12[s.hiRow], h = N.matVec(W2, x); return tex.block(tex.eq(tex.sym('h'), tex.op('='), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op('='), tex.mat(W2, { colors: 'cols' }), tex.vec(x, { color: 'v' }), tex.op('='), tex.vec(h, { color: 'v' })), tex.eq(tex.text('(2×2)(2×1) → (2×1)'))); }
        case 'XWt': return tex.block(tex.eq(tex.sym('H'), tex.op('='), tex.sym('X'), tex.T(tex.sym('W'))), tex.eq(tex.text('(n×d)(d×h) → (n×h)')));
        case 'T': return tex.block(tex.eq(tex.paren(tex.sym('W') + tex.sym('x', { color: 'v' })), '<sup>ᵀ</sup>', tex.op('='), tex.T(tex.sym('x', { color: 'v' })), tex.T(tex.sym('W'))), tex.eq(tex.T(tex.sym('W')), tex.op('='), tex.mat(N.transpose(W2), { colors: 'rows' })));
        case 'lift': return tex.block(tex.eq(tex.sym('W'), tex.op('='), tex.mat(s.W, { colors: 'cols' }), tex.op('∈'), R(3, 2)), tex.eq(tex.sym('H'), tex.op('='), tex.sym('X'), tex.T(tex.sym('W')), tex.op('∈'), R(8, 3)));
        case 'bias': return tex.block(tex.eq(tex.sym('H'), tex.op('='), tex.sym('X'), tex.T(tex.sym('W')), tex.op('+'), tex.num(1), tex.T(tex.sym('b', { color: 'b' }))), tex.eq(tex.T(tex.sym('b', { color: 'b' })), tex.op('='), tex.rowvec(s.b, { color: 'b' })));
        case 'bad': return tex.block(tex.eq(tex.text('(8×2)(3×2)'), tex.text(' ✕', { color: 'bad' })), tex.eq(tex.text('(8×2)(2×3)'), tex.text(' ✓', { color: 'ok' })));
        case 'tensor': return tex.block(tex.eq(tex.text('shape (4, 6, 6, 3) → (4, 108)')));
        case 'reduce': { const X = Xn(s), m = meanRows(X); return tex.block(tex.eq(tex.text('mean(X, axis 0): shape (2,)'), tex.op('='), tex.rowvec(m, { color: 'gold' })), tex.eq(tex.text('sum(X, axis 1): shape (' + s.n + ',)'))); }
        case 'real': return tex.block(tex.eq(tex.text('(B, T, d)·(d×h) → (B, T, h)')));
        case 'guess': return tex.block(tex.eq(tex.sym('W'), tex.op('='), tex.mat(s.guessW, { colors: 'cols' })), tex.eq(tex.sym('H'), tex.op('='), tex.sym('X'), tex.T(tex.sym('W'))));
        case 'meanb': { const H = Hmat(s, true), m = meanRows(H); return tex.block(tex.eq(tex.sym('b', { color: 'b' }), tex.op('='), tex.vec(s.b.slice(0, 2), { color: 'b' })), tex.eq(tex.text('mean of output rows ='), tex.rowvec(m, { color: 'gold' }))); }
        case 'play': {
          const parts = [tex.eq(tex.sym('H'), tex.op('='), tex.sym('X'), tex.T(tex.sym('W')), s.useBias ? tex.op('+') + tex.num(1) + tex.T(tex.sym('b', { color: 'b' })) : '', tex.op('∈'), R(s.n, s.h))];
          parts.push(tex.eq(tex.text('(' + s.n + '×2)(2×' + s.h + ') → (' + s.n + '×' + s.h + ')')));
          return tex.block.apply(null, parts);
        }
        default: return '';
      }
    },

    steps: [
      { title: 'One array, two faces: a table', duration: 3,
        narration: 'Eight samples with two features: an 8×2 array X, the <b>data matrix</b>. Face one: a table, each row a sample. Face two: eight points in the plane (p1-07). Row 3 flies to its point. Same numbers, two pictures.',
        setup(s) { Object.assign(s, { view: '2d', cam: camPlane(), scene: 'points', panel: 'X', mathMode: 'X', n: 8, h: 2, W: LA.clone(W0), mapU: 0, fill: -1, liftU: 0, gridU: 0, showGrid: false, ptsA: 0, hiRow: 2, useBias: false, biasU: 0, recipe: 0, showMean: 0, u: 0 }); },
        at(s, t) { s.u = t; s.ptsA = seg(t, 0, 0.35); } },

      { title: 'The other kind of matrix: a map', duration: 3,
        narration: 'W is also a matrix, but it is a map (p2-01), not a table. Apply it and the grid moves, points riding along. A table holds data; a map moves it. Both are called “matrix” — keep asking which face you are looking at.',
        setup(s) { Object.assign(s, { panel: 'W', mathMode: 'W', showGrid: true, ptsA: 1, u: 0 }); },
        at(s, t) { s.u = t; const v = seg(t, 0.15, 0.9); s.gridU = v; s.mapU = v; } },

      { title: 'One sample: h = Wx', duration: 3,
        narration: 'For one sample, the layer computes h = W' + xv + ', a column (p2-02): shapes (2×2)(2×1) → (2×1). Row 3 of the table, standing up as a column, goes in; a column comes out.',
        setup(s) { Object.assign(s, { panel: 'Wx', mathMode: 'Wx', gridU: 1, mapU: 1, u: 0, recipe: 0 }); },
        at(s, t) { s.u = t; s.recipe = seg(t, 0.45, 0.95); } },

      { title: 'All samples at once: H = XWᵀ', duration: 4,
        narration: 'Lie every sample down as a row of a <b>batch</b> and multiply on the right by Wᵀ: H = XWᵀ, <b>shape</b> (8×2)(2×2) → (8×2). Each output row is one sample’s result. One multiplication moves all eight points. This is why code writes <b>x @ W.T</b>.',
        setup(s) { Object.assign(s, { panel: 'XWt', mathMode: 'XWt', recipe: 0, mapU: 0, gridU: 0, fill: 0, u: 0 }); },
        at(s, t) { s.u = t; s.fill = 8 * seg(t, 0.05, 0.85, 'linear'); s.gridU = s.fill / 8; } },

      { title: 'Why the transpose', duration: 3,
        narration: 'From p2-10: (W' + xv + ')ᵀ = ' + xv + 'ᵀWᵀ. Standing up, a sample is multiplied by W; lying down, by Wᵀ. The transpose flips the recipe so that rows stay rows. Nothing about the layer changed — only the orientation of the bookkeeping.',
        setup(s) { Object.assign(s, { panel: 'T', mathMode: 'T', fill: -1, mapU: 1, gridU: 1, u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Changing width', duration: 4,
        narration: 'Give W three rows: now H = XWᵀ is 8×3 and every point gets a third coordinate. The eight points lift into 3D onto a tilted plane (p2-09): a layer that changes width changes dimension, for the whole batch at once.',
        setup(s) {
          Object.assign(s, { panel: 'lift', mathMode: 'lift', h: 3, view: '3d', showGrid: false, liftU: 0, u: 0 });
          s.cam = Object.assign(camLift(), { yaw: -Math.PI / 2, pitch: 1.5 });
        },
        at(s, t) { s.u = t; s.liftU = seg(t, 0.25, 0.85); LA.cam.tween(s, { yaw: -Math.PI / 2, pitch: 1.5 }, { yaw: -1.05, pitch: 0.5 }, seg(t, 0, 0.8)); } },

      { title: 'Bias is broadcast', duration: 4,
        narration: 'Add the bias ' + bv + ' to every row: the whole cloud slides by b (p2-11). Code writes “+ b” once; the array behind it is 1bᵀ — a column of ones times the row bᵀ, the outer product from p2-10. <b>Broadcasting</b> is an outer product you never see.',
        setup(s) { Object.assign(s, { panel: 'bias', mathMode: 'bias', useBias: true, biasU: 0, liftU: 1, u: 0 }); s.cam = camLift(); },
        at(s, t) { s.u = t; s.biasU = seg(t, 0.5, 0.95); LA.cam.tween(s, { yaw: -1.05 }, { yaw: -0.85 }, seg(t, 0, 1, 'linear')); } },

      { title: '✕ Shapes that do not chain', duration: 3.5, misconception: true,
        narration: '“Any two matrices can be multiplied.” Try X(8×2) times W(3×2): the inner sizes 2 and 3 do not meet, and the tiles bounce apart. Transpose W and the 2s touch. Shape errors are the most common bug in ML code; this picture is the fix.',
        setup(s) { Object.assign(s, { panel: 'bad', mathMode: 'bad', u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Tensors: more indices, stacked slices', duration: 4,
        narration: 'An image is 6×6 pixels × 3 colors: three stacked 6×6 slices. Four images make a (4, 6, 6, 3) <b>tensor</b> (PyTorch orders it (4, 3, 6, 6)) — in ML the word just means “array with more indices”. A <b>row-major flatten</b> turns each image into a 108-cell row: a 4×108 matrix again. The picture stops at 3. The formula does not.',
        setup(s) { Object.assign(s, { scene: 'tensor', panel: 'tensor', mathMode: 'tensor', view: '3d', u: 0 }); s.cam = Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.4], dist: 9.4, yaw: -1.3, pitch: 0.62, fov: 36 }); },
        at(s, t) { s.u = t; LA.cam.tween(s, { yaw: -1.3 }, { yaw: -1.15 }, t); } },

      { title: 'Reductions: summing an axis away', duration: 3.5,
        narration: 'Average down the rows (axis 0) and the 8×2 table becomes one row of 2 numbers: the mean point, the centroid from p4-09. Sum across each row (axis 1) and you get 8 numbers, one per sample. A <b>reduction</b> removes its axis from the shape: (8, 2) becomes (2,) or (8,).',
        setup(s) { Object.assign(s, { scene: 'points', panel: 'reduce', mathMode: 'reduce', view: '2d', h: 2, useBias: false, biasU: 0, liftU: 0, mapU: 0, gridU: 0, showGrid: false, fill: -1, hiRow: null, u: 0, showMean: 0 }); s.cam = camPlane(); },
        at(s, t) { s.u = t; s.showMean = seg(t, 0.35, 0.5); } },

      { title: 'Why ML cares: reading the shapes in code', duration: 4, mlTie: true,
        narration: 'A digit batch is (32, 784); the layer weight is (128, 784); “x @ W.T” gives (32, 128). A transformer activation (32, 2048, 4096) times Wᵀ (4096, 1024) gives (32, 2048, 1024): only the last axis multiplies; the others ride along like our 8 rows. Attention (p6-03) starts with three of these.',
        setup(s) { Object.assign(s, { scene: 'none', panel: 'real', mathMode: 'real', u: 0, showMean: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'A matrix has two faces: a table of samples or a map that moves them. A batch is rows times Wᵀ — one multiplication for everyone — and shapes must chain (n×d)(d×h). Bias is broadcast as 1bᵀ; tensors are stacked slices; reducing an axis removes it.',
        cards: ['<b>Two faces.</b> The same 8×2 array is a table of samples (rows) or 8 points; a map W moves them.', '<b>Batch = rows × Wᵀ; shapes chain.</b> (n×d)(d×h) → (n×h): the inner sizes must touch.', '<b>Broadcast = 1bᵀ; tensors = stacked slices.</b> “+ b” hides an outer product; (4, 6, 6, 3) flattens to (4, 108).'],
        setup(s) { Object.assign(s, { scene: 'points', panel: 'recap', mathMode: 'none', view: '2d', h: 2, mapU: 1, gridU: 1, showGrid: true, ptsA: 1, hiRow: 2, fill: -1, u: 1 }); s.cam = camPlane(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Change the batch size, the output width and the weights; the tables X, Wᵀ and H and the picture update together. Select 3D to see h = 3 outputs as points in space.</p>',
      setup(s) { Object.assign(s, { scene: 'points', panel: 'play', mathMode: 'play', view: '2d', cam: camPlane(), n: 8, h: 2, mapU: 1, gridU: 1, showGrid: true, ptsA: 1, liftU: 1, biasU: 1, useBias: false, hiRow: null, showRaw: true, fill: -1 }); },
      controls: [
        { type: 'slider', bind: 'n', min: 2, max: 12, step: 1, label: 'Samples n', fmt: x => String(Math.round(x)) },
        { type: 'slider', bind: 'h', min: 1, max: 3, step: 1, label: 'Output width h', fmt: x => String(Math.round(x)) },
        { type: 'matrix', bind: 'W', label: 'W (rows used: h)', colors: 'cols', step: 0.1, min: -2, max: 2 },
        { type: 'toggle', bind: 'useBias', label: 'Add bias b' },
        { type: 'select', bind: 'view', label: 'View', options: [{ label: '2D (first two outputs)', value: '2d' }, { label: '3D (h = 3)', value: '3d' }] },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Rotate', apply(s) { const c = Math.cos(Math.PI / 6), sn = Math.sin(Math.PI / 6); s.W = [[c, -sn], [sn, c], [0, 0]]; s.h = 2; s.view = '2d'; s.cam = camPlane(); } },
          { label: 'Lift', apply(s) { s.W = LA.clone(W0); s.h = 3; s.view = '3d'; s.cam = camLift(); } },
          { label: 'Collapse', apply(s) { s.W = [[1, 0.5], [2, 1], [-1, -0.5]]; s.h = 2; s.view = '2d'; s.cam = camPlane(); } },
          { label: 'Reset', apply(s) { s.W = LA.clone(W0); s.h = 2; s.n = 8; s.useBias = false; s.view = '2d'; s.cam = camPlane(); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag tiles into the two slots so that (8×2) · A · B → (8×4) chains correctly.',
        setup(s) {
          Object.assign(s, { scene: 'tiles', panel: 'none', mathMode: 'none', view: '2d', cam: { center: [0, 0], halfH: 3.4 } });
          s.tiles = [{ r: 2, c: 3, p: [-2.2, -2.1] }, { r: 3, c: 4, p: [-0.75, -2.1] }, { r: 4, c: 3, p: [0.75, -2.1] }, { r: 3, c: 2, p: [2.2, -2.1] }];
        },
        judge(s) { const a = inSlot(s, 0), b = inSlot(s, 1); const okA = !!a && a.r === 2 && a.c === 3, okB = !!b && b.r === 3 && b.c === 4; return { ok: okA && okB, progress: (okA ? 0.5 : 0) + (okB ? 0.5 : 0) }; },
        hint: 'Inner sizes must touch: 2 meets 2, then 3 meets 3.',
        solution(s) { s.tiles[0].p = SLOTS[0].slice(); s.tiles[1].p = SLOTS[1].slice(); s.tiles[2].p = [0.75, -2.1]; s.tiles[3].p = [2.2, -2.1]; } },
      { kind: 'entry',
        prompt: 'Type W (2×2) so that H = XWᵀ swaps the two features of every sample — the points reflect across the line y = x onto the faint gold targets.',
        bind: 'guessW', answer: [[0, 1], [1, 0]], tol: LA.TOL.entry, label: 'W', colors: 'cols',
        setup(s) { Object.assign(s, { scene: 'points', panel: 'guess', mathMode: 'guess', view: '2d', cam: camPlane(), n: 8, h: 2, useGuess: true, guessW: [[1, 0], [0, 1]], useBias: false, biasU: 0, mapU: 1, gridU: 1, showGrid: true, ptsA: 1, hiRow: null, showRaw: true, checkMode: 'guess', fill: -1 }); } },
      { kind: 'manipulate',
        prompt: 'Drag the orange bias arrow ' + bv + ' so the mean of the output rows lands at the origin.',
        setup(s) { Object.assign(s, { scene: 'points', panel: 'meanb', mathMode: 'meanb', view: '2d', cam: camPlane(), n: 8, h: 2, W: [[1, 0], [0, 1], [0, 0]], useGuess: false, useBias: true, biasU: 1, b: [0.4, -0.2, 0], mapU: 1, gridU: 0, showGrid: false, ptsA: 1, hiRow: null, showRaw: true, checkMode: 'bias', fill: -1 }); },
        judge(s) { const m = meanRows(Hmat(s, true)); return LA.judge.near(m, [0, 0], LA.TOL.pos); },
        hint: 'You need b = −mean(X): slide the cloud back by its own center.',
        solution(s) { const m = meanRows(Xn(s)); s.b = [-m[0], -m[1], 0]; } },
    ],
  });

  const SLOTS = [[-1.2, 1.2], [0.4, 1.2]];
  function inSlot(s, k) { if (!s.tiles) return null; return s.tiles.find(t => Math.hypot(t.p[0] - SLOTS[k][0], t.p[1] - SLOTS[k][1]) < 0.35) || null; }
  function sup(n) { const m = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' }; return String(n).split('').map(d => m[d]).join(''); }

  // ---------- right-hand scene: the plane / 3D points ----------
  function drawPoints(g, s, ctx) {
    const info = {};
    const is3 = g.is3d;
    const W = Wuse(s);
    if (!is3) {
      g.ghostGrid({ range: 8 });
      if (s.showGrid && s.gridU > 0) {
        const M2 = [[W[0][0], W[0][1]], s.h >= 2 ? [W[1][0], W[1][1]] : [0, 0]];
        g.grid(LA.lerpM(I2(), M2, s.gridU), { range: 8, alpha: 0.75 });
      }
    } else {
      g.grid(null, { range: 2, alpha: 0.5 });
      g.axes({ length: 2.0, names: ['h₁', 'h₂', 'h₃'] });
      if (s.h === 3 && s.liftU > 0) {
        const c0 = [W[0][0], W[1][0], W[2][0] * s.liftU], c1 = [W[0][1], W[1][1], W[2][1] * s.liftU];
        const o = s.useBias ? N.scale(s.b, s.biasU) : [0, 0, 0];
        g.plane({ point: o, span: [c0, c1], extent: 1.7, color: 'teal', alpha: 0.18 * s.liftU });
      }
    }
    const n = s.n;
    if (s.showRaw && s.mapU > 0.5) g.points(Xn(s).map(x => is3 ? [x[0], x[1], 0] : x), { r: 3, color: 'muted', alpha: 0.45 });
    if (s.checkMode === 'guess') g.points(Xn(s).map(x => [x[1], x[0]]), { r: 7, color: 'gold', alpha: 0.3 });
    const pts = [];
    for (let k = 0; k < n; k++) {
      const p = posOf(s, k);
      if (is3 && s.liftU > 0 && s.h === 3) g.line([p[0], p[1], 0], p, { color: 'residual', dashed: true, width: 1, alpha: 0.7 });
      pts.push(is3 ? p : [p[0], p[1]]);
    }
    if (s.ptsA > 0) {
      const hi = (s.hiRow !== null && s.hiRow < n) ? s.hiRow : -1;
      g.points(pts, { r: 5, colors: (i) => i === hi ? 'v' : 'data', alpha: s.ptsA });
      if (hi >= 0 && s.panel !== 'Wx') g.label('row ' + (hi + 1), pts[hi], { dx: 30, dy: -12, size: 11.5, color: 'v', font: 'ui' });
      if (hi >= 0) info.hiScreen = g.P.toScreen(pts[hi]);
    }
    // step 3: x₃ and the column recipe h = x₁·(col 1) + x₂·(col 2)
    if (s.panel === 'Wx') {
      const x = s.X12[s.hiRow], c0 = [W[0][0], W[1][0]], c1 = [W[0][1], W[1][1]];
      g.vector(x, { color: 'v', alpha: 0.45, width: 2, label: 'x' + sub(3), labelColor: 'v' });
      if (s.recipe > 0) {
        const a = N.scale(c0, x[0] * seg(s.recipe, 0, 0.5)), bEnd = N.add(N.scale(c0, x[0]), N.scale(c1, x[1] * seg(s.recipe, 0.5, 1)));
        g.vector([0, 0], a, { color: 'i', width: 2.4, head: 10, label: s.recipe > 0.4 ? 'c₁' : undefined, labelAt: 0.5 });
        if (s.recipe > 0.5) g.vector(N.scale(c0, x[0]), bEnd, { color: 'j', width: 2.4, head: 10, label: s.recipe > 0.9 ? 'c₂' : undefined, labelAt: 0.5 });
        if (s.recipe > 0.98) g.vector(bEnd, { color: 'v', width: 3, label: 'h' + sub(3) });
      }
    }
    if (s.useBias && s.biasU > 0) {
      const bb = N.scale(s.b, s.checkMode === 'bias' ? 1 : s.biasU);
      g.vector(is3 ? bb : [bb[0], bb[1]], { color: 'b', width: 3, label: 'b' });
      if (s.checkMode === 'bias') {
        const m = meanRows(Hmat(s, true));
        g.point(m, { r: 6, color: 'gold', stroke: 'white', label: 'mean', labelColor: 'gold' });
        g.marker.ring([0, 0], { r: 12, color: 'gold' });
        g.handle('b', [s.b[0], s.b[1]], (p, st) => { st.b = [p[0], p[1], 0]; }, { color: 'b' });
      }
    }
    if (s.showMean > 0) {
      const m = meanRows(Xn(s));
      g.point(m, { r: 7, color: 'gold', stroke: 'white', alpha: s.showMean, label: 'mean', labelColor: 'gold' });
    }
    if (is3 && s.h === 3 && s.liftU > 0.5) g.hud('teal plane = span of W’s two columns', g.P.vp.x + 12, g.P.vp.y + g.P.vp.h - 20, { size: 11.5, color: 'teal' });
    if (!is3 && s.h === 3 && s.panel === 'play') g.hud('2D view: first two of three outputs', g.P.vp.x + 12, 20, { size: 12, color: 'muted' });
    if (!is3 && s.h === 1 && s.panel === 'play') g.hud('h = 1: every point lands on one line', g.P.vp.x + 12, 20, { size: 12, color: 'muted' });
    return info;
  }

  // ---------- tensor scene (step 9) ----------
  const CH_ROLE = ['j', 'i', 'k'], CH_NAME = ['R', 'G', 'B'];
  const chRamp = c => t => C.mix(C.bg, C[CH_ROLE[c]], 0.15 + 0.85 * t);
  function drawTensor(g, s) {
    const u = s.u, stackU = seg(u, 0.3, 0.6);
    g.grid(null, { range: 4, alpha: 0.25 });
    for (let k = 0; k < 4; k++) {
      const vis = k === 0 ? 1 : clamp01(stackU * 4 - (k - 1));
      if (vis <= 0) continue;
      const ox = k === 0 ? lerp(0, -2.4, stackU) : (k - 1.5) * 1.6;
      for (let c = 0; c < 3; c++) {
        g.heat3d(s.imgs[k][c], { bounds: { x0: ox - 0.6, x1: ox + 0.6, y0: -0.6, y1: 0.6 }, z: c * 0.4, range: [0, 1], ramp: chRamp(c), alpha: 0.85 * vis });
        if (k === 0) g.label(CH_NAME[c], [ox + 0.75, 0.6, c * 0.4], { color: CH_ROLE[c], size: 12, font: 'ui' });
      }
      g.label('image ' + (k + 1), [ox, -0.95, 0], { color: 'muted', size: 11.5, font: 'ui' });
    }
  }
  function flatRow(img) { const out = []; for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) for (let ch = 0; ch < 3; ch++) out.push([img[ch][r][c], ch]); return out; }

  // ---------- check 1: shape tiles in world coordinates ----------
  function drawTileCheck(g, s) {
    const k = 0.2;
    const box = (cx, cy, r, c, o) => {
      const w = c * k, h = r * k, pts = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]];
      g.polygon(pts, { fill: o.fill || 'teal', alpha: o.alpha === undefined ? 0.35 : o.alpha, stroke: o.stroke || o.fill || 'teal', dash: o.dash });
      if (o.label !== false) g.label(o.label || (r + '×' + c), [cx, cy - h / 2], { anchor: 'below', dy: 6, size: 13, color: o.labelColor || 'text', font: 'mono' });
    };
    g.hud('(8×2) · A · B → (8×4): drag two tiles into the slots', 16, 24, { size: 14, color: 'hud' });
    box(-2.4, 1.2, 8, 2, { fill: 'data', label: '8×2 (X)' });
    g.label('·', [-1.85, 1.2], { size: 30, color: 'hud', bg: false });
    SLOTS.forEach((p, i) => { box(p[0], p[1], 4, 4, { fill: 'ghost', alpha: 0.25, stroke: 'muted', dash: [5, 4], label: false }); g.label(i ? 'B' : 'A', [p[0], p[1] + 0.65], { size: 15, color: 'muted', font: 'math' }); });
    g.label('·', [-0.4, 1.2], { size: 30, color: 'hud', bg: false });
    g.label('→', [1.2, 1.2], { size: 22, color: 'hud' });
    const a = inSlot(s, 0), b = inSlot(s, 1);
    const okA = !!a && a.r === 2, okB = !!a && !!b && b.r === a.c;
    const valid = okA && okB && b.c === 4;
    box(2.2, 1.2, 8, 4, { fill: valid ? 'gold' : 'ghost', alpha: valid ? 0.35 : 0.15, stroke: valid ? 'gold' : 'muted', dash: valid ? null : [5, 4], label: '8×4' });
    if (a) g.label(okA ? '2 = 2 ✓' : '2 ≠ ' + a.r, [-1.6, 2.35], { size: 12.5, color: okA ? 'ok' : 'bad', font: 'ui' });
    if (a && b) g.label(okB ? a.c + ' = ' + b.r + ' ✓' : a.c + ' ≠ ' + b.r, [-0.4, 2.6], { size: 12.5, color: okB ? 'ok' : 'bad', font: 'ui' });
    if (valid) {
      const X = s.X12.slice(0, 8), H = N.matMul(N.matMul(X, A_EX), B_EX);
      g.heat2d(H, { x0: 1.8, x1: 2.6, y0: -0.75, y1: 0.15 }, { range: [-3, 3], ramp: rampDiv, alpha: 1, smooth: false });
      g.label('H: 8 rows × 4 numbers', [2.2, -0.75], { anchor: 'below', dy: 6, size: 11.5, color: 'gold', font: 'ui' });
      g.label('a table, not a picture (4 > 3)', [2.0, -1.1], { anchor: 'below', dy: 6, size: 11, color: 'muted', font: 'ui' });
    }
    s.tiles.forEach((t, i) => {
      box(t.p[0], t.p[1], t.r, t.c, { fill: 'w', alpha: 0.4 });
      g.handle('tile' + i, t.p, (p, st) => {
        let q = [p[0], p[1]];
        for (const sl of SLOTS) if (Math.hypot(q[0] - sl[0], q[1] - sl[1]) < 0.35) q = sl.slice();
        st.tiles[i].p = q;
      }, { color: 'w' });
    });
    g.hud('tiles snap into a slot when dropped near it', 16, g.size.h - 24, { size: 12, color: 'muted' });
  }

  // ---------- left panel ----------
  function drawPanel(g, s, info) {
    const S = g.size, x0 = 34, y0 = 58;
    const X = Xn(s), W2 = Wuse(s).slice(0, 2);
    const p = s.panel, u = s.u;
    if (p === 'none') return;
    if (p === 'X' || p === 'recap') {
      table(g, X, x0, y0, { rowLabels: true, hiRow: s.hiRow, title: 'X  (' + s.n + '×2): rows = samples' });
      txt(g, 'face 1: a table', x0, y0 + X.length * CH + 22, { color: 'muted' });
      txt(g, 'face 2: points  →', x0, y0 + X.length * CH + 42, { color: 'muted' });
      if (p === 'X' && info.hiScreen) {
        const f = seg(u, 0.3, 0.8);
        if (f > 0 && f < 1) {
          const sx = x0 + CW, sy = y0 + (s.hiRow + 0.5) * CH;
          const px = lerp(sx, info.hiScreen[0], f), py = lerp(sy, info.hiScreen[1], f) - Math.sin(Math.PI * f) * 60;
          const w = lerp(2 * CW, 16, f), h = lerp(CH, 12, f);
          rect(g, px - w / 2, py - h / 2, w, h, C.alpha(C.v, 0.55));
          if (f < 0.5) txt(g, LA.fmt(X[s.hiRow][0]) + '   ' + LA.fmt(X[s.hiRow][1]), px, py, { anchor: 'center', size: 11, color: 'bg' });
        }
      }
      if (p === 'recap') { txt(g, 'map W moves them', x0, y0 + X.length * CH + 62, { color: 'muted' }); }
      return;
    }
    if (p === 'W') {
      table(g, X, x0, y0, { rowLabels: true, hiRow: s.hiRow, title: 'X: a table (data)' });
      table(g, W2, x0 + 2 * CW + 24, y0, { title: 'W: a map', colColors: ['i', 'j'] });
      txt(g, 'columns = where', x0 + 2 * CW + 24, y0 + 2 * CH + 16, { size: 11.5, color: 'muted' }); txt(g, 'î and ĵ land', x0 + 2 * CW + 24, y0 + 2 * CH + 32, { size: 11.5, color: 'muted' });
      return;
    }
    if (p === 'Wx') {
      const x = s.X12[s.hiRow], h = N.matVec(W2, x);
      const stand = seg(u, 0.05, 0.4);
      const yy = y0 + 40;
      table(g, W2, x0, yy, { title: 'W', colColors: ['i', 'j'] });
      txt(g, '·', x0 + 2 * CW + 10, yy + CH, { size: 20 });
      // the row of X standing up: cell 1 swings from the right of cell 0 to below it
      const cx = x0 + 2 * CW + 30;
      rect(g, cx, yy, CW, CH, C.alpha(C.v, 0.25));
      txt(g, LA.fmt(x[0]), cx + CW / 2, yy + CH / 2, { anchor: 'center', size: 11, color: 'v' });
      const c1x = lerp(cx + CW, cx, stand), c1y = lerp(yy, yy + CH, stand);
      rect(g, c1x, c1y, CW, CH, C.alpha(C.v, 0.25));
      txt(g, LA.fmt(x[1]), c1x + CW / 2, c1y + CH / 2, { anchor: 'center', size: 11, color: 'v' });
      txt(g, stand < 0.5 ? 'row 3 of X, lying (1×2)' : 'x₃ standing up (2×1)', x0, yy + 2 * CH + 22, { size: 12, color: 'v' });
      if (u > 0.45) {
        const hx = cx + CW + 34;
        txt(g, '=', hx - 22, yy + CH, { size: 18 });
        table(g, h.map(v => [v]), hx, yy, { title: 'h₃', textColor: 'v' });
      }
      txt(g, '(2×2)(2×1) → (2×1)', x0, yy + 2 * CH + 50, { size: 13, color: 'hud', font: 'mono' });
      if (s.recipe > 0.98) {
        txt(g, 'h₃ = ' + LA.fmt(x[0]) + '·c₁ (green column of W)', x0, yy + 2 * CH + 84, { size: 12, color: 'hud' });
        txt(g, '     + ' + LA.fmt(x[1]) + '·c₂ (red column of W)', x0, yy + 2 * CH + 102, { size: 12, color: 'hud' });
      }
      return;
    }
    if (p === 'XWt' || p === 'lift' || p === 'play' || p === 'bias' || p === 'guess' || p === 'meanb') {
      // matmul laid out as a cross: X at left, Wᵀ above H, so every cell of H sits in X's row and Wᵀ's column
      const Wt2 = Wt(s), H = Hmat(s, false);
      const hcols = s.h;
      const cw = Math.max(30, Math.min(CW, (PX * S.w - x0 - 30) / (2 + hcols)));
      const xw = x0, hx = xw + 2 * cw + 18, yX = y0 + 2 * CH + 30;
      table(g, Wt2, hx, y0, { title: 'Wᵀ', cw });
      table(g, X, xw, yX, { rowLabels: true, title: 'X', cw });
      txt(g, '·', xw + 2 * cw + 4, yX - 12, { size: 16 });
      const Hshow = p === 'bias' ? H.map(r => r.map((v, j) => v + s.b[j] * seg(u, 0.5, 0.95))) : (s.useBias && (p === 'play' || p === 'meanb') ? Hmat(s, true) : H);
      const upto = p === 'XWt' ? Math.floor(s.fill + 1e-9) : undefined;
      table(g, Hshow, hx, yX, { title: p === 'bias' ? '= XWᵀ + 1bᵀ' : (s.useBias && p !== 'XWt' && p !== 'lift' ? '= XWᵀ + 1bᵀ' : '= H'), upto, range: 3, cw });
      const ty = yX + X.length * CH + 30;
      if (p === 'bias') {
        // the row b stacking into 8 copies = 1bᵀ, then added to H
        const bx = hx, by = ty + 10, stack = seg(u, 0, 0.5);
        txt(g, '1bᵀ', bx - 40, by + 6, { size: 13, color: 'b' });
        const copies = Math.max(1, Math.round(1 + 7 * stack));
        for (let i = 0; i < copies; i++) {
          const yy = by + i * 14 * stack;
          rect(g, bx, yy, 3 * cw, 13, C.alpha(C.b, i === 0 ? 0.55 : 0.25));
          if (i === 0) s.b.forEach((v, j) => txt(g, LA.fmt(v), bx + (j + 0.5) * cw, yy + 6.5, { anchor: 'center', size: 10.5, color: 'text' }));
        }
        txt(g, 'row b copied into', x0, by + 8 * 14 + 22, { size: 11.5, color: 'b' });
        txt(g, 'every row: 1bᵀ = (8×1)(1×3)', x0, by + 8 * 14 + 38, { size: 11.5, color: 'b' });
        return;
      }
      if (p === 'XWt' || p === 'lift') {
        const snap = seg(u, 0, 0.35);
        const dims = p === 'XWt' ? [[8, 2], [2, 2]] : [[8, 2], [2, 3]];
        const ty2 = ty + 60;
        const bxs = chain(g, x0, ty2, dims, lerp(50, 4, snap), { k: 10 });
        txt(g, '→', bxs[1].x + bxs[1].w + 10, ty2, { size: 16 });
        tile(g, bxs[1].x + bxs[1].w + 32, ty2, 8, dims[1][1], { k: 10, color: 'gold' });
        if (snap > 0.95) txt(g, 'inner 2s touch', x0, ty2 - 54, { size: 11.5, color: 'gold' });
        txt(g, '(n×d)(d×h) → (n×h)', x0, ty2 + 78, { size: 12.5, color: 'hud', font: 'mono' });
      }
      if (p === 'play') txt(g, '(' + s.n + '×2)(2×' + s.h + ') → (' + s.n + '×' + s.h + ')', x0, ty, { size: 13, color: 'hud', font: 'mono' });
      if (p === 'guess') txt(g, 'faint gold dots: where swapped points should land', x0, ty, { size: 12, color: 'gold' });
      if (p === 'meanb') { const m = meanRows(Hmat(s, true)); txt(g, 'mean of output rows: (' + LA.fmt(m[0]) + ', ' + LA.fmt(m[1]) + ')', x0, ty, { size: 12.5, color: 'gold' }); }
      return;
    }
    if (p === 'T') {
      const tr = seg(u, 0, 0.45), dt = seg(u, 0.5, 1, 'linear');
      const yy = y0 + 30, wx = x0 + 30;
      txt(g, tr < 0.5 ? 'W' : 'Wᵀ', wx, yy - 14, { size: 13 });
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const ci = lerp(i, j, tr), cj = lerp(j, i, tr);
        const px = wx + cj * CW, py = yy + ci * CH;
        const lit = dt > 0 && Math.floor(Math.min(1.999, dt * 2)) === i;
        rect(g, px, py, CW, CH, C.alpha(lit ? C.gold : C.panel2, lit ? 0.5 : 1));
        txt(g, LA.fmt(W2[i][j]), px + CW / 2, py + CH / 2, { anchor: 'center', size: 11, color: j ? 'j' : 'i' });
      }
      txt(g, 'entries slide across the diagonal', wx, yy + 2 * CH + 14, { size: 11.5, color: 'muted' });
      if (u >= 0.5) {
        const x = s.X12[s.hiRow], out = N.matVec(W2, x);
        const ry = yy + 2 * CH + 60;
        txt(g, 'xᵀ (row 3, lying down)', x0, ry - 14, { size: 12, color: 'v' });
        rect(g, x0, ry, 2 * CW, CH, C.alpha(C.v, 0.2));
        x.forEach((v, j) => txt(g, LA.fmt(v), x0 + (j + 0.5) * CW, ry + CH / 2, { anchor: 'center', size: 11, color: 'v' }));
        txt(g, '· Wᵀ =', x0 + 2 * CW + 8, ry + CH / 2, { size: 13 });
        const ox = x0 + 2 * CW + 64;
        for (let j = 0; j < 2; j++) {
          const on = dt * 2 > j + 0.3;
          rect(g, ox + j * CW, ry, CW, CH, C.alpha(on ? C.gold : C.panel2, on ? 0.45 : 1));
          if (on) txt(g, LA.fmt(out[j]), ox + (j + 0.5) * CW, ry + CH / 2, { anchor: 'center', size: 11, color: 'text' });
        }
        const jj = Math.min(1, Math.floor(dt * 2));
        txt(g, 'output ' + (jj + 1) + ' = xᵀ · (column ' + (jj + 1) + ' of Wᵀ)', x0, ry + CH + 18, { size: 11.5, color: 'gold' });
        txt(g, 'rows stay rows', x0, ry + CH + 40, { size: 12, color: 'muted' });
      }
      return;
    }
    if (p === 'bad') {
      const k = 19, yy = y0 + 170, xA = x0 + 10;
      txt(g, '“Any two matrices', x0 + 28, y0 + 4, { size: 13.5, color: 'text' });
      txt(g, 'can be multiplied.”', x0 + 28, y0 + 22, { size: 13.5, color: 'text' });
      const close = seg(u, 0, 0.3), bounce = seg(u, 0.3, 0.5, 'outCubic'), flip = seg(u, 0.58, 0.78), snap2 = seg(u, 0.8, 0.95);
      const A = tile(g, xA, yy, 8, 2, { k, color: 'data' });
      let gapPx;
      if (u < 0.3) gapPx = lerp(150, 0, close);
      else if (u < 0.58) gapPx = 46 * Math.sin(Math.PI * bounce) * (1 - 0.3 * bounce) + 40 * bounce;
      else gapPx = lerp(40, 4, snap2);
      const bw = lerp(2 * k, 3 * k, flip), bh = lerp(3 * k, 2 * k, flip);
      const lab = flip < 0.5 ? '3×2 (W)' : '2×3 (Wᵀ)';
      tile(g, A.x + A.w + gapPx, yy, 0, 0, { w: bw, h: bh, color: flip > 0.5 ? 'w' : 'b', label: lab });
      if (u > 0.28) {
        g.marker.cross([x0 + 10, y0 + 13], { screen: true, size: 22 });
        txt(g, 'inner sizes must match: 2 ≠ 3', x0, yy + 4 * k + 34, { size: 13, color: 'bad' });
      }
      if (u >= 0.95) {
        g.marker.check([A.x + A.w + 4 + bw + 26, yy], { screen: true, size: 22 });
        txt(g, '(8×2)(2×3) → (8×3) ✓', x0, yy + 4 * k + 58, { size: 13, color: 'ok' });
      }
      txt(g, 'the most common bug in ML code', x0, yy + 4 * k + 90, { size: 12, color: 'muted' });
      return;
    }
    if (p === 'tensor') {
      const y = S.h * 0.8;
      txt(g, 'shape (4, 6, 6, 3): 4 images × 6 rows × 6 columns × 3 colors', 20, 26, { size: 13.5, color: 'hud' });
      txt(g, 'channels-last (N, H, W, C), as in TensorFlow; PyTorch stores the same batch channels-first: (4, 3, 6, 6)', 20, 46, { size: 11.5, color: 'muted' });
      const unroll = seg(u, 0.6, 1);
      if (unroll > 0) {
        const cw = (S.w - 80) / 108, ch = 13;
        txt(g, 'row-major flatten → (4, 108): one image per row, read row by row (R, G, B per pixel)', 20, y - 14, { size: 12.5, color: 'hud' });
        for (let k = 0; k < 4; k++) {
          const fr = flatRow(s.imgs[k]);
          const nShow = k === 0 ? Math.round(108 * seg(unroll, 0, 0.7)) : (unroll > 0.75 ? 108 : 0);
          for (let i = 0; i < nShow; i++) rect(g, 50 + i * cw, y + k * (ch + 3), Math.max(1, cw - 0.5), ch, chRamp(fr[i][1])(fr[i][0]));
          if (nShow) txt(g, String(k + 1), 36, y + k * (ch + 3) + ch / 2, { size: 10.5, color: 'muted', anchor: 'center' });
        }
      }
      txt(g, 'The picture stops at 3. The formula does not.', S.w - 20, S.h - 18, { size: 13, color: 'gold', anchor: 'right' });
      return;
    }
    if (p === 'reduce') {
      const r1 = seg(u, 0, 0.45), r2 = seg(u, 0.55, 1);
      const m = meanRows(X), sums = X.map(r => r[0] + r[1]);
      txt(g, 'axis 0: average down the rows', x0, y0 - 16, { size: 12.5, color: 'gold' });
      const yMean = y0 + X.length * CH + 16;
      X.forEach((row, i) => { const yy = lerp(y0 + i * CH, yMean, r1); g.strip(row, { x: x0, y: yy, w: 2 * CW, h: CH, range: [-2.5, 2.5], ramp: rampDiv }); if (r1 < 0.6) row.forEach((v, j) => txt(g, LA.fmt(v), x0 + (j + 0.5) * CW, yy + CH / 2, { anchor: 'center', size: 11, color: 'text' })); });
      if (r1 > 0.6) { rect(g, x0, yMean, 2 * CW, CH, C.alpha(C.gold, 0.4)); m.forEach((v, j) => txt(g, LA.fmt(v), x0 + (j + 0.5) * CW, yMean + CH / 2, { anchor: 'center', size: 11, color: 'text' })); txt(g, '(' + X.length + ', 2) → (2,): the mean point', x0 + 2 * CW + 12, yMean + CH / 2, { size: 12, color: 'gold' }); }
      if (u >= 0.5) {
        const y2 = yMean + 70;
        txt(g, 'axis 1: sum across each row', x0, y2 - 16, { size: 12.5, color: 'teal' });
        X.forEach((row, i) => {
          const yy = y2 + i * CH;
          const c1x = lerp(x0 + CW, x0, r2);
          g.strip([row[0]], { x: x0, y: yy, w: CW, h: CH, range: [-2.5, 2.5], ramp: rampDiv });
          g.strip([row[1]], { x: c1x, y: yy, w: CW, h: CH, range: [-2.5, 2.5], ramp: rampDiv });
          if (r2 < 0.5) row.forEach((v, j) => txt(g, LA.fmt(v), (j ? c1x : x0) + CW / 2, yy + CH / 2, { anchor: 'center', size: 11, color: 'text' }));
          else { g.strip([sums[i]], { x: x0, y: yy, w: CW, h: CH, range: [-4, 4], ramp: rampDiv }); txt(g, LA.fmt(sums[i]), x0 + CW / 2, yy + CH / 2, { anchor: 'center', size: 11, color: 'text' }); }
        });
        if (r2 > 0.5) {
          txt(g, '(' + X.length + ', 2) → (' + X.length + ',): one number per sample', x0 + CW + 14, y2 + 4 * CH, { size: 12, color: 'teal' });
          txt(g, 'with keepdims=True the axis stays,', x0, y2 + X.length * CH + 22, { size: 11.5, color: 'muted' });
          txt(g, 'size 1: (1, 2) and (' + X.length + ', 1)', x0, y2 + X.length * CH + 40, { size: 11.5, color: 'muted' });
        }
      }
      return;
    }
    if (p === 'real') {
      const lg = n => 12 * Math.log2(n);
      const u1 = seg(u, 0, 0.4), u2 = seg(u, 0.5, 0.9);
      txt(g, 'tile sizes on a log scale', 24, 26, { size: 12, color: 'muted' });
      const y1 = 170;
      txt(g, 'digit batch: x @ W.T', 40, 70, { size: 14, color: 'hud' });
      const b1 = chain(g, 60, y1, [[32, 784, { label: '32×784', color: 'data' }], [784, 128, { label: '784×128 (Wᵀ)', color: 'w' }]], lerp(120, 4, u1), { size: lg });
      if (u1 > 0.95) { txt(g, '→', b1[1].x + b1[1].w + 18, y1, { size: 18 }); tile(g, b1[1].x + b1[1].w + 46, y1, 32, 128, { size: lg, color: 'gold', label: '32×128' }); }
      if (u >= 0.5) {
        const y2 = 470;
        txt(g, 'transformer: (B, T, d) @ W.T — only the last axis multiplies', 40, 330, { size: 14, color: 'hud' });
        const x2 = 60;
        for (let q = 3; q >= 1; q--) rect(g, x2 + q * 9, y2 - lg(2048) / 2 - q * 9, lg(4096), lg(2048), C.alpha(C.data, 0.12));
        const b2 = chain(g, x2, y2, [[2048, 4096, { label: '(32, 2048, 4096)', color: 'data' }], [4096, 1024, { label: '4096×1024 (Wᵀ)', color: 'w' }]], lerp(120, 30, u2), { size: lg });
        txt(g, '32 ×', x2 - 2, y2 - lg(2048) / 2 - 36, { size: 12, color: 'muted' });
        if (u2 > 0.95) {
          const ox = b2[1].x + b2[1].w + 46;
          txt(g, '→', ox - 28, y2, { size: 18 });
          for (let q = 3; q >= 1; q--) rect(g, ox + q * 9, y2 - lg(2048) / 2 - q * 9, lg(1024), lg(2048), C.alpha(C.gold, 0.12));
          tile(g, ox, y2, 2048, 1024, { size: lg, color: 'gold', label: '(32, 2048, 1024)' });
          txt(g, 'see p6-03: attention starts with three of these (Q, K, V)', S.w - 24, S.h - 30, { size: 13, color: 'gold', anchor: 'right' });
        }
      }
      return;
    }
  }
})();
