/* p3-07 — The Gram matrix: all pairwise dot products.
   Two rooms side by side (stacked when the canvas is tall): the six arrows in the plane, and the 6×6 table
   K = XXᵀ as a heatmap (blue negative · dark zero · orange positive). Cells are hoverable through invisible
   handles ('K:i,j' in ctx.hot). Full-canvas panels for the shape comparison and the 64-feature strip step. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, C = LA.color;
  const PTS = [[2, 0], [1.5, 1.5], [0, 2], [-1, 1], [-2, -1], [1, -2]];
  const SUB = ['₁', '₂', '₃', '₄', '₅', '₆'];
  const xn = i => 'x' + SUB[i];
  // 64-feature versions of six points (synthetic 8×8 digits, LA.rng seed 37), computed once
  const DIGITS = LA.rng.syntheticDigits(37, 6, { size: 8 }).map(d => d.pixels.map(v => +v.toFixed(2)));
  const K64 = DIGITS.map(a => DIGITS.map(b => N.dot(a, b)));

  function gram(pts) { return pts.map(a => pts.map(b => N.dot(a, b))); }
  function cosM(K) { return K.map((r, i) => r.map((v, j) => { const d = Math.sqrt(K[i][i] * K[j][j]); return d > 1e-12 ? v / d : 0; })); }
  function model(s) {
    const K = gram(s.pts), Kc = cosM(K);
    const Kd = K.map((r, i) => r.map((v, j) => LA.lerp(v, Kc[i][j], s.normT)));
    const X = s.pts, G = N.matMul(N.transpose(X), X);
    return { K, Kc, Kd, G, X };
  }
  const disp = (s, i) => { const p = s.pts[i], n = N.norm(p); return n > 1e-9 ? lerpV(p, N.scale(p, 1 / n), s.normT) : p; };
  const BASE = '#1a1d26';
  const kRamp = (v, range) => { const t = LA.clamp(v / (range || 1), -1, 1); return t < 0 ? C.mix(BASE, C.k, -t) : C.mix(BASE, C.b, t); };

  /* ---------------- layout ---------------- */
  function rects(size) {
    const W = size.w, H = size.h;
    if (W / H > 1.2) return { plot: { x: 0, y: 0, w: 0.52 * W, h: H }, panel: { x: 0.52 * W, y: 0, w: 0.48 * W, h: H } };
    return { plot: { x: 0, y: 0, w: W, h: 0.47 * H }, panel: { x: 0, y: 0.47 * H, w: W, h: 0.53 * H } };
  }
  function panel(g, r, opts, fn) {
    g.viewport({ x: r.x, y: r.y, w: Math.max(2, r.w), h: Math.max(2, r.h) }, Object.assign({ view: '2d', cam: { center: [r.w / 2, -r.h / 2], halfH: r.h / 2 }, border: false }, opts || {}), (g2, rr) => fn(g2, rr));
  }
  const P = (x, y) => [x, -y];
  function rect(g2, x, y, w, h, o) { g2.polygon([P(x, y), P(x + w, y), P(x + w, y + h), P(x, y + h)], Object.assign({ alpha: 1 }, o)); }
  function outline(g2, x, y, w, h, color, width, dashed) { g2.polyline([P(x, y), P(x + w, y), P(x + w, y + h), P(x, y + h)], { color, width: width || 1.4, closed: true, dashed }); }
  /** Heatmap geometry inside the panel rect. */
  function heatGeom(r) {
    const c = Math.round(LA.clamp(Math.min((r.w - 90) / 7.5, (r.h - 80) / 7), 26, 54));
    const hx = 52, hy = 44;
    const side = r.w - (hx + 6 * c) >= 175;
    const free = side ? { x: hx + 6 * c + 30, y: hy, w: r.w - (hx + 6 * c) - 40, h: 6 * c } : { x: hx - 30, y: hy + 6 * c + 36, w: r.w - hx + 20, h: r.h - (hy + 6 * c + 44) };
    return { c, hx, hy, side, free };
  }

  LA.registerLesson({
    id: 'p3-07-gram-matrix',
    part: 3,
    title: 'The Gram matrix: all pairwise dot products',
    subtitle: 'K = XXᵀ: one dot product per pair; symmetric, never negative on vᵀKv.',
    minutes: 9,
    prereqs: ['p1-06-dot-product', 'p2-10-transpose-outer-symmetric', 'p3-06-regression-matrix-form'],
    tags: ['core'],
    badge: { drawnIn: '2D (six points, d = 2)', realSize: 'attention builds a 4096×4096 Gram-like table per head; a kernel SVM on 10⁵ examples has a 10⁵×10⁵ K', transfers: 'K_ij = xᵢ·xⱼ is one number per pair in any d; symmetry and positive semi-definiteness hold for every size' },
    cantShow: ['points in d > 3, although K never needs them drawn: it is n×n whatever d is', 'kernel feature spaces (often infinite-dimensional): only their Gram matrix exists on screen'],
    terms: [
      { term: 'Gram matrix', def: 'The matrix of all pairwise dot products, K = XXᵀ with K_ij = xᵢ·xⱼ.' },
      { term: 'positive semi-definite', def: 'vᵀKv ≥ 0 for every v: K never turns a vector to point against itself (PSD for short).' },
      { term: 'cosine matrix', def: 'The Gram matrix of the unit-length versions of the vectors; its entries are cos θᵢⱼ.' },
      { term: 'XᵀX vs XXᵀ', def: 'Pairs of features (d×d) versus pairs of examples (n×n).' },
    ],
    atlas: ['kernels', 'attention', 'gaussian'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: { center: [0, 0], halfH: 2.7 },
        pts: LA.clone(PTS), layout: 'split', mathMode: 'pair',
        pair: [0, 1], showPair: true, showArc: true, flyT: 0, fillT: 36, filling: false, cellHover: false,
        normT: 0, cross: null, foldT: 0, shapeT: 0, v: [1, -0.5, 0.8, 0.3, -1, 0.6], vSeed: 0, showPsd: false,
        ndT: 0, mlT: 0, showFeatures: false, dragOnly: null, entry: [1], showEntryCell: false, rowTarget: false,
      };
    },

    render(g, s, ctx) {
      const m = model(s);
      if (s.layout === 'shapes') { panel(g, { x: 0, y: 0, w: g.size.w, h: g.size.h }, {}, (g2, r) => drawShapes(g2, s, m, r)); return; }
      if (s.layout === 'nd') { panel(g, { x: 0, y: 0, w: g.size.w, h: g.size.h }, {}, (g2, r) => drawND(g2, s, r)); return; }
      const R = rects(g.size);
      // which pair is lit: a hovered cell wins, else the step's pair
      let pair = s.showPair ? s.pair : null;
      if (ctx.hot && /^K:/.test(ctx.hot)) { const q = ctx.hot.slice(2).split(',').map(Number); pair = q; }
      if (s.filling && s.fillT < 36) { const k = Math.min(35, Math.floor(s.fillT)); pair = [Math.floor(k / 6), k % 6]; }
      let arcScreen = null, cellScreen = null;
      const across = s.showPsd ? 3.7 : 2.65;   // keep ±2.65 visible across in a tall plot (±3.7 for the long Xᵀv sums)
      const pcam = { center: s.cam.center, halfH: Math.max(s.showPsd ? 3.7 : s.cam.halfH, across * R.plot.h / R.plot.w) };
      g.viewport(R.plot, { view: '2d', cam: pcam }, (g2, r) => { arcScreen = drawPlot(g2, s, m, ctx, r, pair); });
      panel(g, R.panel, { border: true }, (g2, r) => { cellScreen = drawHeat(g2, s, m, ctx, r, pair); });
      if (s.mlT > 0) drawKernelInset(g, s, R.plot);
      // the first dot product flies into its cell
      if (s.flyT > 0 && s.flyT < 1 && arcScreen && cellScreen) g.label('3', lerpV(arcScreen, cellScreen, seg(s.flyT, 0, 1, 'inOut')), { screen: true, color: 'gold', size: 18, bold: true });
    },

    math(s, tex) {
      const m = model(s);
      const sym = tex.sym, op = tex.op, num = tex.num, eq = tex.eq, txt = tex.text;
      const X = sym('X'), K = sym('K');
      const xs = (i, color) => sym('x', { sub: String(i + 1), color });
      switch (s.mathMode) {
        case 'pair': { const a = s.pts[0], b = s.pts[1]; return tex.block(eq(xs(0, 'v'), op('·'), xs(1, 'w'), op('='), tex.dotExpanded(a, b, { colorA: 'v', colorB: 'w' }))); }
        case 'K': return tex.block(eq(K, op('='), X, tex.T(X)), eq(sym('K', { sub: 'ii' }), op('='), tex.norm(sym('x', { sub: 'i' })) + '<sup>2</sup>'), eq(op('('), tex.dim(6, 2), op(')('), tex.dim(2, 6), op(') ='), tex.dim(6, 6)));
        case 'hover': { const p = s.pair, a = s.pts[p[0]], b = s.pts[p[1]], na = N.norm(a), nb = N.norm(b); const c = na * nb > 1e-9 ? N.dot(a, b) / (na * nb) : 0;
          return tex.block(eq(sym('K', { sub: (p[0] + 1) + '' + (p[1] + 1) }), op('='), tex.abs(xs(p[0], 'v')), tex.abs(xs(p[1], 'w')), txt('cos θ'), op('='), num(na), op('·'), num(nb), op('·'), num(c), op('='), num(N.dot(a, b), { color: 'gold' }))); }
        case 'cos': return tex.block(eq(sym('K', { hat: true }), op('='), sym('D', { sup: '−1/2' }), K, sym('D', { sup: '−1/2' })), eq(sym('K', { hat: true, sub: 'ij' }), op('='), txt('cos θ'), op('ᵢⱼ')), eq(sym('K', { hat: true, sub: '25' }), op('='), num(m.Kc[1][4], { color: 'gold' })));
        case 'sym': return tex.block(eq(K, op('='), tex.T(K)), eq(sym('x', { sub: 'i' }), op('·'), sym('x', { sub: 'j' }), op('='), sym('x', { sub: 'j' }), op('·'), sym('x', { sub: 'i' })));
        case 'shapes': return tex.block(eq(X, tex.T(X), txt('(n×n: pairs of points)')), eq(tex.T(X), X, op('='), tex.mat(m.G, { colors: (i, j) => i === j ? (i ? 'j' : 'i') : 'text' }), txt('(d×d: pairs of features)')));
        case 'psd': { const u = N.matVec(N.transpose(m.X), s.v); return tex.block(eq(sym('v', { sup: 'T' }), K, sym('v'), op('='), sym('v', { sup: 'T' }), X, tex.T(X), sym('v'), op('='), tex.norm(tex.T(X) + sym('v')) + '<sup>2</sup>', op('='), num(N.dot(u, u), { color: 'gold' }), op('≥ 0')), eq(txt('rank'), K, op('='), num(N.rank(m.K, 1e-9)), op('≤'), sym('d'), op('='), num(2))); }
        case 'nd': return tex.block(eq(sym('K', { sub: 'ij' }), op('='), tex.sum(sym('X', { sub: 'ik' }) + sym('X', { sub: 'jk' }), 'k=1', '64')), eq(sym('K', { sub: '12' }), op('='), num(K64[0][1], { color: 'gold' })));
        case 'ml': return tex.block(eq(txt('scores'), op('='), sym('Q'), tex.T(sym('K'))), eq(sym('K', { sub: 'ij' }), op('='), sym('k'), tex.paren(sym('x', { sub: 'i' }) + op(',') + sym('x', { sub: 'j' }))));
        case 'entry': return tex.block(eq(sym('K', { sub: '24' }), op('='), xs(1, 'v'), op('·'), xs(3, 'w'), op('='), txt('?')), eq(xs(1, 'v'), op('='), tex.vec(s.pts[1], { color: 'v' }), xs(3, 'w'), op('='), tex.vec(s.pts[3], { color: 'w' })));
        case 'row': return tex.block(eq(txt('row 1:'), tex.rowvec(m.K[0], { digits: 1 })), eq(txt('row 6:'), tex.rowvec(m.K[5], { digits: 1, color: 'gold' })));
        default: return tex.block(eq(K, op('='), X, tex.T(X)), eq(sym('K', { sub: 'ij' }), op('='), sym('x', { sub: 'i' }), op('·'), sym('x', { sub: 'j' })));
      }
    },

    steps: [
      { title: 'Six arrows, one number per pair', duration: 3.5,
        narration: 'Six data points drawn as arrows from the origin. For any <i>pair</i>, the dot product is one number: how much they point the same way, times their lengths. x₁·x₂ = 2·1.5 + 0·1.5 = 3. Six arrows make 36 pairs. Let us collect all 36 numbers in one table.',
        setup(s) { s.layout = 'split'; s.view = '2d'; s.cam = { center: [0, 0], halfH: 2.7 }; s.pts = LA.clone(PTS); s.pair = [0, 1]; s.showPair = true; s.showArc = true; s.fillT = 0; s.flyT = 0; s.normT = 0; s.mathMode = 'pair'; s.cellHover = false; },
        at(s, t) { s.flyT = seg(t, 0.45, 0.9, 'linear'); s.fillT = t > 0.9 ? 0.0001 : 0; s.showArc = t > 0.15; } },

      { title: 'Filling K row by row', duration: 5,
        narration: 'Row i, column j holds xᵢ·xⱼ. Row 1 is x₁ against everyone: 4, 3, 0, −2, −4, 2. The diagonal is each arrow against itself, its squared length. This 6×6 table is the <b>Gram matrix</b> K = XXᵀ: the whole cloud’s geometry, with no coordinates in it.',
        setup(s) { s.flyT = 0; s.fillT = 0; s.showArc = false; s.mathMode = 'K'; s.filling = true; },
        at(s, t) { s.fillT = seg(t, 0.02, 0.92, 'linear') * 36; s.showPair = t < 0.95; s.pair = [0, 1]; } },

      { title: 'Hover a cell: the angle is in the sign', duration: 4, interactive: true,
        narration: 'Hover any cell. Positive: the two arrows lean the same way. Zero: perpendicular (x₁ and x₃). Negative: they lean apart; x₁ and x₅ point nearly opposite, −4. The table is a map of agreements and disagreements.',
        setup(s) { s.filling = false; s.fillT = 36; s.showPair = true; s.showArc = true; s.cellHover = true; s.pair = [0, 1]; s.mathMode = 'hover'; },
        at(s, t) { s.pair = t < 0.33 ? [0, 1] : (t < 0.66 ? [0, 2] : [0, 4]); } },

      { title: 'A big number does not mean similar', duration: 4, misconception: true,
        narration: '✕ “K₂₅ = −4.5 has the biggest size, so x₂ and x₅ are the most related.” Length inflates dot products. Divide each entry by both lengths and you get the <b>cosine matrix</b>: every entry lies between −1 and 1. Now x₂ and x₅ read −0.95: nearly opposite, not similar.',
        setup(s) { s.cellHover = false; s.pair = [1, 4]; s.showPair = true; s.showArc = true; s.normT = 0; s.cross = null; s.mathMode = 'cos'; },
        at(s, t) { s.cross = t > 0.08 ? 'k25' : null; s.normT = seg(t, 0.35, 0.85); } },

      { title: 'Symmetric, always', duration: 3,
        narration: 'xᵢ·xⱼ = xⱼ·xᵢ, so K_ij = K_ji: the matrix is a mirror image of itself across the diagonal. Every Gram matrix is symmetric, which, as Part 4 shows, gives it perpendicular eigenvectors and real eigenvalues.',
        setup(s) { s.normT = 0; s.cross = null; s.showPair = false; s.showArc = false; s.foldT = 0; s.mathMode = 'sym'; },
        at(s, t) { s.foldT = seg(t, 0.1, 0.9); } },

      { title: 'Two Gram matrices: pairs of points vs pairs of features', duration: 5,
        narration: 'X is 6×2. XXᵀ pairs <i>points</i>: 6×6. Flip the order and XᵀX pairs <i>features</i>: 2×2, the dot products of the two columns, 12.25 and 12.25 on the diagonal, 1.25 off it. That is <b>XᵀX vs XXᵀ</b>. Center and divide by n and XᵀX becomes the covariance matrix; the normal equations used it too.',
        setup(s) { s.foldT = 0; s.layout = 'shapes'; s.shapeT = 0; s.mathMode = 'shapes'; },
        at(s, t) { s.shapeT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Never negative: vᵀKv = |Xᵀv|²', duration: 4.5,
        narration: 'Take any six weights v and compute vᵀKv. It equals |Xᵀv|², the squared length of the weighted sum of the six arrows, so it can never drop below zero. That property is called <b>positive semi-definite</b>. K is also flat: six points in 2D give a 6×6 matrix of rank only 2.',
        setup(s) { s.layout = 'split'; s.showPsd = true; s.showPair = false; s.showArc = false; s.mathMode = 'psd'; s.vSeed = 0; s.v = randV(0); },
        at(s, t) { const k = Math.min(4, Math.floor(t * 5)); s.vSeed = k; s.v = randV(k); } },

      { title: 'The picture stops at 3; K does not', duration: 4,
        narration: 'Give each point 64 features instead of 2. We cannot draw the arrows, but K is still 6×6 with the same meaning: one dot product per pair. The picture stops at 3. The formula does not: it is a sum of products, just with more terms.',
        setup(s) { s.showPsd = false; s.layout = 'nd'; s.ndT = 0; s.mathMode = 'nd'; },
        at(s, t) { s.ndT = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Why ML cares: attention and kernels are Gram matrices', duration: 4, mlTie: true,
        narration: 'Attention scores are QKᵀ: every query against every key, a Gram-style table of dot products, one row per token. Kernel methods replace xᵢ·xⱼ by k(xᵢ, xⱼ) and never look at the points again: the Gram matrix is <i>all</i> they need. Both wait in Part 6.',
        setup(s) { s.layout = 'split'; s.ndT = 0; s.showPair = false; s.showArc = false; s.mlT = 0; s.mathMode = 'ml'; },
        at(s, t) { s.mlT = 0.001 + 0.999 * seg(t, 0.1, 0.8); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'K = XXᵀ holds every pairwise dot product: sign = angle, diagonal = squared lengths, normalize for cosines. It is symmetric and positive semi-definite because vᵀKv = |Xᵀv|². XXᵀ pairs points (n×n); XᵀX pairs features (d×d).',
        cards: ['<b>One dot product per pair.</b> K_ij = xᵢ·xⱼ = |xᵢ||xⱼ| cos θ; the sign is the angle.', '<b>Symmetric and PSD.</b> K = Kᵀ, and vᵀKv = |Xᵀv|² ≥ 0 for every v.', '<b>n×n vs d×d.</b> XXᵀ pairs points; XᵀX pairs features.'],
        setup(s) { s.mlT = 0; s.pair = [0, 1]; s.showPair = true; s.showArc = true; s.mathMode = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of the six arrows. Hover any cell of K to light up its pair. Switch to the cosine matrix, show XᵀX, or roll a random v to test vᵀKv ≥ 0.</p>',
      setup(s) { s.layout = 'split'; s.view = '2d'; s.fillT = 36; s.showPair = true; s.showArc = true; s.pair = [0, 1]; s.cellHover = true; s.mathMode = 'hover'; s.showPsd = false; s.v = randV(0); },
      controls: [
        { type: 'toggle', bind: 'normT', label: 'Cosine matrix (unit-length arrows)' },
        { type: 'toggle', bind: 'showFeatures', label: 'Show XᵀX (pairs of features)' },
        { type: 'button', label: 'Random v (test vᵀKv)', onClick(s) { s.vSeed = (s.vSeed || 0) + 1; s.v = randV(s.vSeed); s.showPsd = true; } },
        { type: 'preset', label: 'Points', items: [
          { label: 'Default', values: { pts: LA.clone(PTS) } },
          { label: 'Nearly perpendicular', values: { pts: [[2, 0], [0, 2], [-2, 0.1], [0.1, -2], [1.4, 1.4], [-1.4, 1.4]] } },
          { label: 'Collinear (rank 1)', values: { pts: [[2, 1], [1, 0.5], [-1, -0.5], [-2, -1], [0.6, 0.3], [-1.4, -0.7]] } },
          { label: 'Unit circle', values: { pts: PTS.map(p => N.normalize(p)) } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag x₃ so that K₁₃ = 0: make it perpendicular to x₁ (keep it longer than 0.5).',
        setup(s) { toPlay(s); s.pts = LA.clone(PTS); s.pts[2] = [1.2, 1.7]; s.dragOnly = 2; s.pair = [0, 2]; s.mathMode = 'hover'; },
        judge(s) { const a = s.pts[0], b = s.pts[2]; const r = LA.judge.angleDeg(a, b, 90, LA.TOL.angleDeg); const long = N.norm(b) > 0.5; return { ok: r.ok && long, progress: r.progress, hint: long ? undefined : 'Keep x₃ longer than 0.5.' }; },
        hint: 'x₁ points along the x-axis; what is perpendicular to it?',
        solution(s) { s.pts[2] = [0, 2]; } },
      { kind: 'entry',
        prompt: 'Type K₂₄ for the default points (x₂ = [1.5; 1.5], x₄ = [−1; 1]). Your number fills cell (2, 4); it turns green when it matches.',
        bind: 'entry', answer: [0], tol: LA.TOL.entry, label: 'K₂₄', colors: 'muted', step: 0.5,
        setup(s) { toPlay(s); s.pts = LA.clone(PTS); s.dragOnly = -1; s.cellHover = false; s.pair = [1, 3]; s.showArc = false; s.entry = [1]; s.showEntryCell = true; s.mathMode = 'entry'; } },
      { kind: 'manipulate',
        prompt: 'Drag x₆ until its row of K is the negative of row 1: each entry of row 6 equals the matching entry of row 1 with its sign flipped. Compare the two rows in the math panel.',
        setup(s) { toPlay(s); s.pts = LA.clone(PTS); s.dragOnly = 5; s.pair = [0, 5]; s.rowTarget = true; s.mathMode = 'row'; },
        judge(s) { const m = model(s); let worst = 0; for (let j = 0; j < 5; j++) worst = Math.max(worst, Math.abs(m.K[5][j] + m.K[0][j])); const r = LA.judge.near(s.pts[5], [-2, 0], LA.TOL.pos); return { ok: r.ok, progress: Math.max(0, 1 - worst / 8) }; },
        hint: 'Row 1 lists x₁ against everyone. Which arrow gives the opposite of each of those numbers? Point x₆ exactly against x₁, same length.',
        solution(s) { s.pts[5] = [-2, 0]; } },
    ],
  });

  function toPlay(s) { s.filling = false; s.layout = 'split'; s.view = '2d'; s.cam = { center: [0, 0], halfH: 2.7 }; s.fillT = 36; s.normT = 0; s.showPair = true; s.showArc = true; s.cellHover = true; s.showPsd = false; s.mlT = 0; s.foldT = 0; s.cross = null; s.showEntryCell = false; s.rowTarget = false; }
  function randV(k) { const r = LA.rng(700 + k); return LA.rng.gaussV(r, 6).map(x => +x.toFixed(2)); }

  /* ---------------- the plane of arrows ---------------- */
  function drawPlot(g, s, m, ctx, r, pair) {
    g.grid(null, { alpha: 0.35 });
    g.axes({ labels: false });
    if (s.normT > 0) g.circle([0, 0], 1, { stroke: 'muted', dashed: true, width: 1.2, strokeAlpha: 0.5 + 0.5 * s.normT });
    const lit = pair ? (pair[0] === pair[1] ? [pair[0]] : pair) : [];
    const isML = s.mlT > 0;
    let arcScreen = null;
    // PSD: the weighted arrows vᵢxᵢ laid tip to tail, and their sum Xᵀv
    if (s.showPsd) {
      let o = [0, 0];
      for (let i = 0; i < 6; i++) { const step = N.scale(s.pts[i], s.v[i]); g.vector(o, N.add(o, step), { color: 'w', width: 1.6, head: 7, alpha: 0.6 }); o = N.add(o, step); }
      g.vector([0, 0], o, { color: 'residual', width: 3, label: 'Xᵀv', labelFont: 'ui', labelSize: 13 });
      g.label('|Xᵀv|² = ' + LA.fmt(N.dot(o, o)), [r.x + 10, r.y + 16], { screen: true, anchor: 'left', color: 'hud', size: 13, font: 'ui' });
      g.label('purple: vᵢ·xᵢ tip to tail', [r.x + 10, r.y + 34], { screen: true, anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    }
    for (let i = 0; i < 6; i++) {
      const p = disp(s, i), k = lit.indexOf(i);
      const color = k === 0 ? 'v' : k === 1 ? 'w' : (s.rowTarget && i === 5 ? 'gold' : 'hud');
      const name = isML ? 't' + SUB[i] : xn(i);
      g.vector(p, { color, width: k >= 0 ? 3.2 : 2, alpha: k >= 0 || lit.length === 0 ? (s.showPsd ? 0.45 : 1) : 0.45, label: name, labelFont: 'math', labelSize: 15 });
    }
    if (pair && s.showArc && pair[0] !== pair[1]) {
      const a = disp(s, pair[0]), b = disp(s, pair[1]);
      const ang = N.angle ? N.angle(a, b) : Math.acos(LA.clamp(N.dot(a, b) / (N.norm(a) * N.norm(b)), -1, 1));
      g.arc([0, 0], a, b, { r: 0.45, color: 'gold', label: LA.fmt.deg(ang), right: Math.abs(ang - Math.PI / 2) < 0.01 });
      // the dashed shadow of the second arrow on the first
      const na = N.dot(a, a);
      if (na > 1e-9) { const foot = N.scale(a, N.dot(a, b) / na); g.line(b, foot, { color: 'residual', dashed: true, width: 1.3 }); g.line([0, 0], foot, { color: 'mag', width: 4, alpha: 0.55 }); }
      const mid = N.scale(N.normalize(N.add(N.normalize(a), N.normalize(b))), 0.45 * 1.45);
      const q = g.P.toScreen(Number.isFinite(mid[0]) ? mid : [0.4, 0.4]); arcScreen = [q[0], q[1]];
    }
    if (pair && pair[0] === pair[1]) g.label('|' + xn(pair[0]) + '|² = ' + LA.fmt(m.K[pair[0]][pair[0]]), disp(s, pair[0]), { color: 'v', size: 12.5, font: 'ui', dx: 14, dy: 16 });
    // drag the tips (play / check modes)
    if (ctx.mode !== 'learn') for (let i = 0; i < 6; i++) {
      if (s.dragOnly !== null && s.dragOnly !== undefined && s.dragOnly !== i) continue;
      g.handle('x' + i, disp(s, i), (p, st) => { st.pts[i] = [+LA.clamp(p[0], -2.6, 2.6).toFixed(2), +LA.clamp(p[1], -2.6, 2.6).toFixed(2)]; st.normT = 0; }, { color: i === 5 && s.rowTarget ? 'gold' : 'white' });
    }
    g.label(isML ? 'six tokens: here queries = keys = these arrows (toy)' : (s.normT > 0.5 ? 'unit-length arrows (cosine view)' : 'six data points as arrows'), [r.x + 10, r.y + r.h - 14], { screen: true, anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    return arcScreen;
  }

  /* ---------------- the heatmap of K ---------------- */
  function drawHeat(g2, s, m, ctx, r, pair) {
    const G = heatGeom(r), c = G.c, hx = G.hx, hy = G.hy;
    const isCos = s.normT > 0.5;
    const range = LA.lerp(Math.max(5, ...m.K.flat().map(Math.abs)), 1, s.normT);
    const title = s.mlT > 0 ? 'QKᵀ  (scores: every query against every key)' : (isCos ? 'cosine matrix K̂  (entries cos θ)' : 'K = XXᵀ  (6×6)');
    g2.label(title, P(hx - 40, 16), { anchor: 'left', color: s.mlT > 0 ? 'gold' : 'text', size: 12.5, font: 'ui' });
    const nf = Math.floor(s.fillT + 1e-9);
    for (let i = 0; i < 6; i++) {
      g2.label((s.mlT > 0 ? 't' : 'x') + SUB[i], P(hx - 14, hy + (i + 0.5) * c), { color: 'muted', size: 13, font: 'math', anchor: 'right' });
      g2.label((s.mlT > 0 ? 't' : 'x') + SUB[i], P(hx + (i + 0.5) * c, hy - 10), { color: 'muted', size: 13, font: 'math' });
      for (let j = 0; j < 6; j++) {
        const x = hx + j * c, y = hy + i * c, idx = i * 6 + j, filled = (s.filling ? idx <= nf && s.fillT > 0 : idx < nf) || (s.fillT >= 36);
        const hideVal = s.showEntryCell && ((i === 1 && j === 3) || (i === 3 && j === 1));
        if (!filled && !(idx === 1 && s.fillT > 0)) { rect(g2, x + 1, y + 1, c - 2, c - 2, { fill: 'panel', alpha: 1, stroke: 'line', width: 0.8 }); continue; }
        const v = m.Kd[i][j];
        rect(g2, x + 1, y + 1, c - 2, c - 2, { fill: hideVal ? C.panel2 : kRamp(v, range), alpha: 1, stroke: 'line', width: 0.8 });
        const txt = isCos ? LA.fmt.fixed(v, 2) : LA.fmt(Math.round(v * 100) / 100);
        if (!hideVal) g2.label(txt, P(x + c / 2, y + c / 2), { color: 'text', size: c < 34 ? 10.5 : 12, font: 'mono', bg: false });
        if (i === j && !isCos) outline(g2, x + 2, y + 2, c - 4, c - 4, 'gold', 1.2);
      }
    }
    // the typed entry (check 2)
    if (s.showEntryCell) {
      const ok = Math.abs(s.entry[0] - m.K[1][3]) <= LA.TOL.entry;
      [[1, 3], [3, 1]].forEach(q => { const x = hx + q[1] * c, y = hy + q[0] * c; rect(g2, x + 1, y + 1, c - 2, c - 2, { fill: ok ? C.mix(BASE, C.ok, 0.55) : kRamp(s.entry[0], range), alpha: 1 }); g2.label(LA.fmt(s.entry[0]), P(x + c / 2, y + c / 2), { color: ok ? 'white' : 'text', size: 12.5, font: 'mono', bold: true, bg: false }); outline(g2, x + 1, y + 1, c - 2, c - 2, ok ? 'ok' : 'gold', 2); });
    }
    // lit pair: white outline + its mirror
    if (pair && s.fillT > 0) {
      outline(g2, hx + pair[1] * c, hy + pair[0] * c, c, c, 'white', 2.2);
      if (pair[0] !== pair[1]) outline(g2, hx + pair[0] * c, hy + pair[1] * c, c, c, 'white', 1.2, [3, 3]);
    }
    if (s.cross === 'k25') outline(g2, hx + 4 * c, hy + c, c, c, 'bad', 2.4);
    // symmetry: the upper triangle folds onto the lower one
    if (s.foldT > 0) {
      // one copy at a time (row by row), so the moving cells never pile up on the diagonal
      let k = 0;
      for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++, k++) {
        const u = seg(s.foldT, 0.7 * k / 15, 0.7 * k / 15 + 0.3), lift = Math.sin(Math.PI * u) * 0.25;
        if (u <= 0) continue;
        const x = LA.lerp(hx + j * c, hx + i * c, u), y = LA.lerp(hy + i * c, hy + j * c, u), sz = c * (1 + lift);
        rect(g2, x + (c - sz) / 2 + 2, y + (c - sz) / 2 + 2, sz - 4, sz - 4, { fill: kRamp(m.K[i][j], range), alpha: 0.9, stroke: 'gold', width: 1.2 });
        if (u > 0.95 || u < 0.05) g2.label(LA.fmt(m.K[i][j]), P(x + c / 2, y + c / 2), { color: 'text', size: 11, font: 'mono', bg: false });
      }
      const u = s.foldT;
      g2.line(P(hx, hy), P(hx + 6 * c, hy + 6 * c), { color: 'gold', width: 2, alpha: 0.9 });
      if (u > 0.95) g2.label('K = Kᵀ: each copy lands on an equal number', P(hx, hy + 6 * c + 18), { anchor: 'left', color: 'gold', size: 12, font: 'ui' });
    }
    // hover handles on the cells
    if (s.cellHover || ctx.mode !== 'learn') for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) g2.handle('K:' + i + ',' + j, P(hx + (j + 0.5) * c, hy + (i + 0.5) * c), () => { }, { show: false, always: true, r: c / 2 - 2 });
    // side / bottom area: readouts, PSD bars, XᵀX tile, softmax preview
    const F = G.free;
    let fy = F.y + 6;
    if (pair && s.fillT >= 36 && s.foldT === 0 && !s.showPsd && s.mlT === 0 && !s.showEntryCell) {
      const v = m.Kd[pair[0]][pair[1]], sub = SUB[pair[0]] + SUB[pair[1]];
      g2.label(isCos ? 'K̂' + sub + ' = cos θ = ' + LA.fmt(v) : 'K' + sub + ' = ' + xn(pair[0]) + '·' + xn(pair[1]) + ' = ' + LA.fmt(v), P(F.x, fy + 6), { anchor: 'left', color: 'gold', size: 13, font: 'ui' });
      g2.label(v > 0.02 ? 'positive: lean the same way' : (v < -0.02 ? 'negative: lean apart' : 'zero: perpendicular'), P(F.x, fy + 26), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
      fy += 52;
    }
    if (s.cross === 'k25') {
      g2.marker.cross(P(F.x + 8, fy + 8), { size: 18 });
      g2.label(isCos ? 'big |K| ≠ related: −0.95 is direction only' : '“|K₂₅| = 4.5 is the largest, so most related”', P(F.x + 24, fy + 8), { anchor: 'left', color: 'bad', size: 11.5, font: 'ui' });
      fy += 30;
    }
    if (s.showPsd) {
      const bw = Math.min(F.w - 10, 200), bh = 46;
      g2.label('weights v', P(F.x, fy + 4), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
      signedBars(g2, s.v, F.x, fy + 16, bw, bh, 2.5, 'w');
      const u = N.matVec(N.transpose(m.X), s.v), q = N.dot(u, u);
      const by = fy + 16 + bh + 26;
      g2.label('vᵀKv = ' + LA.fmt(q) + '  (never below 0)', P(F.x, by), { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
      const bl = Math.min(bw, 200) * LA.clamp(q / 40, 0, 1);
      g2.line(P(F.x, by + 12), P(F.x, by + 28), { color: 'hud', width: 1.4 });
      rect(g2, F.x, by + 14, Math.max(1, bl), 12, { fill: 'gold', alpha: 0.9 });
      g2.label('0', P(F.x, by + 38), { color: 'muted', size: 10.5, font: 'ui' });
      g2.label('rank K = ' + N.rank(m.K, 1e-9) + '  (6 points in 2D)', P(F.x, by + 56), { anchor: 'left', color: 'hud', size: 12, font: 'ui' });
      fy = by + 70;
    }
    if (s.showFeatures) {
      g2.label('XᵀX (pairs of features) =', P(F.x, fy + 6), { anchor: 'left', color: 'text', size: 12, font: 'ui' });
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g2.label(LA.fmt(m.G[i][j]), P(F.x + 30 + j * 60, fy + 28 + i * 20), { color: i === j ? (i ? 'j' : 'i') : 'hud', size: 12.5, font: 'mono' });
      fy += 70;
    }
    if (s.mlT > 0) {
      // row 1 of the score table turns into softmax weights
      outline(g2, hx, hy, 6 * c, c, 'gold', 2.2);
      const sm = N.softmax(m.K[0]), bw = Math.min(F.w - 10, 210), bh = 60 * s.mlT;
      g2.label('softmax(row 1): attention weights of t₁', P(F.x, fy + 4), { anchor: 'left', color: 'gold', size: 11.5, font: 'ui' });
      signedBars(g2, sm.map(x => x * s.mlT), F.x, fy + 16, bw, 60, 1, 'gold', true);
      g2.label('kernel methods: same table, K_ij = k(xᵢ, xⱼ)', P(F.x, fy + 104), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    }
    return [r.x + hx + 1.5 * c, r.y + hy + 0.5 * c];
  }
  /** Bars around a zero line (or from the bottom when positiveOnly). */
  function signedBars(g2, vals, x, y, w, h, max, color, positiveOnly) {
    const n = vals.length, bw = w / n, zero = positiveOnly ? y + h : y + h / 2;
    g2.line(P(x, zero), P(x + w, zero), { color: 'line', width: 1 });
    vals.forEach((v, i) => {
      const hh = LA.clamp(v / max, -1, 1) * (positiveOnly ? h : h / 2);
      const top = hh >= 0 ? zero - hh : zero;
      rect(g2, x + i * bw + 3, top, bw - 6, Math.max(1, Math.abs(hh)), { fill: color, alpha: 0.85 });
      g2.label((positiveOnly ? 't' : 'v') + SUB[i], P(x + (i + 0.5) * bw, (positiveOnly ? y + h : y + h) + 10), { color: 'muted', size: 10, font: 'ui' });
    });
  }

  /* ---------------- pairs of points vs pairs of features ---------------- */
  function drawShapes(g2, s, m, r) {
    const X = m.X, cx = 34, a = seg(s.shapeT, 0, 0.45), b = seg(s.shapeT, 0.5, 0.95);
    const x0 = 40, y0 = 56;
    g2.label('X  (6×2)', P(x0, 22), { anchor: 'left', color: 'text', size: 12.5, font: 'ui' });
    g2.label('rows = points, columns = features', P(x0 - 26, y0 + 6 * cx + 16), { anchor: 'left', color: 'muted', size: 11, font: 'ui' });
    for (let i = 0; i < 6; i++) {
      g2.label(xn(i), P(x0 - 10, y0 + (i + 0.5) * cx), { anchor: 'right', color: 'muted', size: 13, font: 'math' });
      for (let j = 0; j < 2; j++) { rect(g2, x0 + j * cx, y0 + i * cx, cx, cx, { fill: C.mix(C.panel2, C[j ? 'j' : 'i'], 0.25 + 0.15 * Math.abs(X[i][j])), alpha: 1, stroke: 'line', width: 0.8 }); g2.label(LA.fmt(X[i][j]), P(x0 + (j + 0.5) * cx, y0 + (i + 0.5) * cx), { color: 'text', size: 12, font: 'mono', bg: false }); }
    }
    // rows → XXᵀ
    const row = Math.min(5, Math.floor(a * 6));
    if (a > 0 && a < 1) outline(g2, x0 - 2, y0 + row * cx, 2 * cx + 4, cx, 'v', 2);
    if (a >= 1) for (let i = 0; i < 6; i++) outline(g2, x0, y0 + i * cx + 1, 2 * cx, cx - 2, 'v', 1, true);
    const kx = x0 + 2 * cx + 70, kc = 26, range = 5;
    g2.label('XXᵀ  6×6: pairs of points', P(kx, 22), { anchor: 'left', color: 'v', size: 12.5, font: 'ui' });
    g2.line(P(x0 + 2 * cx + 12, y0 + 3 * cx), P(kx - 12, y0 + 3 * cx), { color: 'muted', width: 1.4 });
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) { const on = a >= 1 || i < row || (i === row && a > 0); if (!on) continue; rect(g2, kx + j * kc, y0 + i * kc, kc - 1, kc - 1, { fill: kRamp(m.K[i][j], range), alpha: 1 }); g2.label(LA.fmt(m.K[i][j]), P(kx + (j + 0.5) * kc, y0 + (i + 0.5) * kc), { color: 'text', size: 9.5, font: 'mono', bg: false }); }
    // columns → XᵀX
    if (b > 0) {
      const sy = y0 + 6 * cx + 50;
      outline(g2, x0, y0, cx, 6 * cx, 'i', 2.2); outline(g2, x0 + cx, y0, cx, 6 * cx, 'j', 2.2);
      g2.label('the two feature columns as strips (each is a vector in ℝ⁶)', P(x0, sy - 18), { anchor: 'left', color: 'muted', size: 12, font: 'ui', alpha: b });
      ['i', 'j'].forEach((role, j) => { for (let i = 0; i < 6; i++) { rect(g2, x0 + i * cx, sy + j * (cx + 8), cx, cx, { fill: C.mix(C.panel2, C[role], 0.25 + 0.15 * Math.abs(X[i][j])), alpha: b, stroke: role, width: 1 }); g2.label(LA.fmt(X[i][j]), P(x0 + (i + 0.5) * cx, sy + j * (cx + 8) + cx / 2), { color: 'text', size: 12, font: 'mono', bg: false, alpha: b }); } g2.label(j ? 'feature 2' : 'feature 1', P(x0 + 6 * cx + 10, sy + j * (cx + 8) + cx / 2), { anchor: 'left', color: role, size: 12, font: 'ui', alpha: b }); });
      const gx = x0 + 6 * cx + 110, gc = 52;
      g2.label('XᵀX  2×2: pairs of features', P(gx, sy - 18), { anchor: 'left', color: 'text', size: 12.5, font: 'ui', alpha: b });
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { rect(g2, gx + j * gc, sy + i * gc, gc - 2, gc - 2, { fill: kRamp(m.G[i][j], 13), alpha: b, stroke: i === j ? (i ? 'j' : 'i') : 'line', width: 1.4 }); g2.label(LA.fmt(m.G[i][j]), P(gx + (j + 0.5) * gc, sy + (i + 0.5) * gc), { color: 'text', size: 12.5, font: 'mono', bg: false, alpha: b }); }
      g2.label('centre and divide by n → covariance', P(gx, sy + 2 * gc + 16), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui', alpha: b });
    }
  }

  /* ---------------- 64 features: strips, not arrows ---------------- */
  function drawND(g2, s, r) {
    const n = 64, cw = Math.max(4, Math.min(7, Math.floor((r.w - 120) / n))), sh = 16, gap = 26, x0 = 60, y0 = 64;
    const sw = cw * n, kSide = r.w - (x0 + sw) > 230;
    g2.label('six points with 64 features each (synthetic 8×8 digits, LA.rng seed 37)', P(x0 - 40, 24), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    const ramp = t => C.mix(BASE, C.hud, LA.clamp01(t));
    for (let i = 0; i < 6; i++) {
      const y = y0 + i * gap, hl = i === 0 ? 'v' : i === 1 ? 'w' : null;
      g2.label(xn(i), P(x0 - 10, y + sh / 2), { anchor: 'right', color: hl || 'muted', size: 13, font: 'math' });
      for (let k = 0; k < n; k++) rect(g2, x0 + k * cw, y, cw, sh, { fill: ramp(DIGITS[i][k]), alpha: 1 });
      outline(g2, x0, y, sw, sh, hl || 'line', hl ? 2 : 0.8);
    }
    // the elementwise product of x₁ and x₂ and its running sum
    const py = y0 + 6 * gap + 20, kDone = Math.round(s.ndT * n);
    g2.label(xn(0) + ' ⊙ ' + xn(1) + '  (multiply matching cells)', P(x0, py - 10), { anchor: 'left', color: 'gold', size: 12, font: 'ui' });
    let sum = 0;
    for (let k = 0; k < n; k++) { const v = DIGITS[0][k] * DIGITS[1][k]; if (k < kDone) { sum += v; rect(g2, x0 + k * cw, py, cw, sh, { fill: C.mix(BASE, C.gold, LA.clamp01(v)), alpha: 1 }); } }
    outline(g2, x0, py, sw, sh, 'gold', 1.2);
    g2.label('running sum = ' + LA.fmt(sum), P(x0, py + sh + 16), { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
    // the 6×6 K for 64 features
    const kc = 30, kx = kSide ? x0 + sw + 60 : x0 + 30, ky = kSide ? y0 : py + sh + 60;
    const mx = Math.max(...K64.flat());
    g2.label('K = XXᵀ  still 6×6', P(kx, ky - 22), { anchor: 'left', color: 'text', size: 12.5, font: 'ui' });
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
      const target = (i === 0 && j === 1) || (i === 1 && j === 0);
      const v = target ? (s.ndT >= 1 ? K64[i][j] : sum) : K64[i][j];
      rect(g2, kx + j * kc, ky + i * kc, kc - 1, kc - 1, { fill: kRamp(v, mx), alpha: target ? 1 : 0.55 });
      g2.label(LA.fmt(Math.round(v * 10) / 10), P(kx + (j + 0.5) * kc, ky + (i + 0.5) * kc), { color: 'text', size: 9.5, font: 'mono', bg: false, alpha: target ? 1 : 0.7 });
      if (target && i === 0) outline(g2, kx + j * kc, ky + i * kc, kc - 1, kc - 1, 'white', 2);
    }
    const cy = kSide ? ky + 6 * kc + 30 : ky + 6 * kc + 26;
    g2.label('The picture stops at 3. The formula does not.', P(kSide ? kx - 20 : x0, cy), { anchor: 'left', color: 'text', size: 14, font: 'ui', bold: true });
    g2.label('n×n whatever d is: no arrow needed', P(kSide ? kx - 20 : x0, cy + 22), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
  }

  /* ---------------- kernel preview: the points lifted onto a paraboloid (tiny 3D inset) ---------------- */
  function drawKernelInset(g, s, R) {
    const w = Math.min(200, R.w * 0.4), h = 140, ir = { x: R.x + 8, y: R.y + 8, w, h };
    g.viewport(ir, { view: '3d', cam: { target: [0, 0, 1.6], dist: 11, yaw: 0.7, pitch: 0.42, fov: 36 }, bg: 'panel' }, (g2) => {
      const a = s.mlT;
      g2.paramSurface((u, v) => [u * Math.cos(v), u * Math.sin(v), u * u * 0.6], { u0: 0, u1: 2.3, v0: 0, v1: 2 * Math.PI }, { res: 10, mode: 'wire', color: 'teal', wireAlpha: 0.35 * a });
      s.pts.forEach(p => { const q = [p[0], p[1], 0.6 * N.dot(p, p) * a]; g2.line([p[0], p[1], 0], q, { color: 'residual', width: 1, dashed: [3, 3], alpha: 0.7 }); g2.point(q, { r: 3.5, color: 'teal' }); });
      g2.label('lift x → (x, |x|²): K_ij = k(xᵢ, xⱼ)', [ir.x + 6, ir.y + 12], { screen: true, anchor: 'left', color: 'teal', size: 11, font: 'ui' });
    });
  }
})();
