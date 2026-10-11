/* p2-03 — Rotation, scaling, shear, reflection, projection: the transformation zoo. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, fmt = LA.fmt;
  const I2 = () => [[1, 0], [0, 1]];
  const DEG = Math.PI / 180;
  const SCALE = [[2, 0], [0, 0.5]], ZOOM = [[1.5, 0], [0, 1.5]], SHEAR = [[1, 1], [0, 1]], FLIP = [[0, 1], [1, 0]];
  const PROJ = [[1, 0], [0, 0]], MYST = [[1, 0.5], [0.5, 1]], HALF = [[-1, 0], [0, -1]], STD = [[0.4, 0], [0, 2]];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const GUESSES = [[[0, -1], [1, 0]], [[1.5, 0], [0, 0.5]], [[1, -1], [0, 1]], [[-1, 0], [0, 1]], [[0.71, 0.71], [-0.71, 0.71]], [[1, 1], [0, 0]]];
  const ZOO = [
    { name: 'rotation 45°', M: N.rot2(Math.PI / 4), sym: 'R' },
    { name: 'scaling', M: SCALE, sym: 'S' },
    { name: 'shear', M: SHEAR, sym: 'H' },
    { name: 'reflection', M: FLIP, sym: 'F' },
    { name: 'projection', M: PROJ, sym: 'P' },
  ];
  // synthetic data (computed once): the standardization cloud and a 28×28 synthetic digit for augmentation
  const CLOUD = LA.rng.gaussianCloud(31, 60, { cov: [[6.25, 0], [0, 0.25]] });
  const DIGIT = LA.rng.syntheticDigits(5, 8, { size: 28 })[7].pixels;

  /* numbers in the math panel: plain colored spans (see p2-02: tex.num's in-place update can hit the wrong cell) */
  const tn = (tex, x, color) => '<span class="' + tex.colorClass(color) + '">' + (typeof x === 'number' ? fmt(x) : x) + '</span>';

  /** the matrix the picture shows: in the playground it is derived from the sliders unless the learner drags freely */
  function curM(s) {
    if (!s.live) return s.M;
    switch (s.kind) {
      case 'rotation': return N.rot2(s.theta);
      case 'scale': return [[s.sx, 0], [0, s.sy]];
      case 'shear': return [[1, s.k], [0, 1]];
      case 'reflection': return N.reflect2(s.theta);
      case 'projection': return N.project2(s.theta);
      default: return s.M;
    }
  }
  function fitCam(g, fw, fh, c) { const aspect = (g.size.w * fw) / (g.size.h * fh); return { center: c.center.slice(), halfH: Math.max(c.halfH, c.halfW / aspect) }; }
  function sline(g, a, b, o) { g.line(g.P.toWorld(a[0], a[1]), g.P.toWorld(b[0], b[1]), o); }
  function hudMat(g, cells, x, y, o) {
    o = o || {}; const cw = o.cw || 46, ch = o.ch || 28, size = o.size || 15, col = o.color || 'hud';
    const rows = cells.length, cols = cells[0].length, w = cols * cw, h = rows * ch, st = { color: col, width: 1.5, alpha: o.alpha };
    const br = (bx, d) => { sline(g, [bx + 6 * d, y], [bx, y], st); sline(g, [bx, y], [bx, y + h], st); sline(g, [bx, y + h], [bx + 6 * d, y + h], st); };
    br(x, 1); br(x + w + 8, -1);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const c = cells[i][j];
      g.label(c.t, [x + 4 + cw * (j + 0.5), y + ch * (i + 0.5)], { screen: true, color: c.c || col, size, font: 'math', bold: !!c.hl, alpha: o.alpha, bg: false });
    }
    return w + 8;
  }
  const cellsOf = (M, q) => M.map(r => r.map((x, j) => ({ t: q ? '?' : fmt(Math.round(x * 100) / 100), c: j === 0 ? 'i' : 'j' })));

  /** a small curved arrow inside the unit square, carried by M: counter-clockwise before, clockwise after a flip */
  function orientArrow(g, M, alpha) {
    if (Math.abs(N.det(M)) < 0.06) return;
    const pts = [];
    for (let i = 0; i <= 26; i++) { const a = -0.5 + (i / 26) * 4.3; pts.push(N.matVec(M, [0.5 + 0.24 * Math.cos(a), 0.5 + 0.24 * Math.sin(a)])); }
    g.polyline(pts.slice(0, -2), { color: 'hud', width: 1.6, alpha });
    g.vector(pts[pts.length - 3], pts[pts.length - 1], { color: 'hud', width: 1.6, head: 9, alpha });
  }

  /** the shared picture: ghost grid, moved grid, unit square (fill by orientation), circle → ellipse, î and ĵ */
  function drawPlane(g, M, o) {
    const a = o.alpha === undefined ? 1 : o.alpha;
    if (o.showGhost !== false) g.ghostGrid();
    g.grid(M, { alpha: 0.9 * a, range: o.range || 10 });
    if (o.showSquare !== false) g.parallelogram(N.col(M, 0), N.col(M, 1), { fill: 'v', orient: true, alpha: 0.28 * a });
    if (o.showCircle) g.ellipse(M, [0, 0], { stroke: 'hud', width: 1.6, strokeAlpha: 0.8 * a });
    if (o.showOrient !== false) orientArrow(g, M, 0.85 * a);
    const c1 = N.col(M, 0), c2 = N.col(M, 1);
    g.vector(c1, { color: 'i', label: 'i', hat: true, width: 3, alpha: a });
    g.vector(c2, { color: 'j', label: 'j', hat: true, width: 3, alpha: a });
  }

  function renderMain(g, s, ctx) {
    const M = curM(s), c1 = N.col(M, 0), c2 = N.col(M, 1);
    if (s.guess !== null && s.live && s.kind === 'guess') g.grid(GUESSES[s.guess % GUESSES.length], { color: 'gold', axisColor: 'gold', alpha: 0.4, range: 6 });
    if (s.showProjLine) { g.lineThrough([0, 0], [1, 0], { color: 'teal', width: 3, alpha: 0.75 }); }
    const lineKind = s.live && (s.kind === 'reflection' || s.kind === 'projection');
    if (s.live ? (lineKind || (s.kind === 'free' && s.mirror)) : s.mirror) {
      const ang = lineKind ? s.theta : Math.PI / 4;
      g.lineThrough([0, 0], [Math.cos(ang), Math.sin(ang)], { color: s.live && s.kind === 'projection' ? 'teal' : 'muted', dashed: true, width: 1.6 });
      g.label(s.live && s.kind === 'projection' ? 'onto this line' : (s.live && s.kind === 'reflection' ? 'mirror' : 'mirror y = x'), [2.0 * Math.cos(ang), 2.0 * Math.sin(ang)], { color: 'muted', size: 12, anchor: 'left', dx: 8 });
    }
    drawPlane(g, M, { showCircle: s.showCircle, showSquare: s.showSquare, showGhost: s.showGhost });
    // the fan of unit arrows that rotate with the plate (step 2)
    if (s.fan > 0) {
      for (let k = 0; k < 8; k++) {
        const u = N.matVec(M, [Math.cos((k + 0.5) * Math.PI / 4), Math.sin((k + 0.5) * Math.PI / 4)]);
        g.vector(u, { color: 'muted', width: 1.8, head: 8, alpha: 0.8 * s.fan, label: k === 1 ? 'u' : undefined });
        if (k === 1 && s.fan > 0.5) g.dim([0, 0], u, { label: '‖u‖ = ' + LA.fmt.fixed(N.norm(u), 2), offset: -16, color: 'hud' });
      }
    }
    if (s.rightMark) g.arc([0, 0], c1, c2, { r: 0.32, right: true, color: 'hud' });
    if (s.coordLabels) {
      g.label('[' + fmt(c1[0]) + '; ' + fmt(c1[1]) + ']', c1, { color: 'i', size: 12.5, dx: 30, dy: 16 });
      g.label('[' + fmt(c2[0]) + '; ' + fmt(c2[1]) + ']', c2, { color: 'j', size: 12.5, anchor: 'right', dx: -14, dy: -12 });
    }
    if (s.dims) {   // shear: base and height stay 1
      g.dim([0, 0], c1, { label: 'base ' + fmt(N.norm(c1)), offset: 16, color: 'hud' });
      const top = N.add(c1, c2);
      g.dim([top[0], 0], top, { label: 'height ' + fmt(top[1]), offset: -16, color: 'hud' });
      g.label('area ' + fmt(Math.abs(N.det(M))), N.scale(top, 0.5), { color: 'v', size: 13, dy: 18 });
    }
    if (s.projDots > 0) {   // a column of lattice points falls onto one point
      for (let y = -2; y <= 3; y++) {
        const p0 = [1, y], p = N.matVec(M, p0);
        g.line(p0, p, { color: 'residual', dashed: true, width: 1.2, alpha: s.projDots });
        g.point(p, { r: 4.5, color: 'mag', alpha: s.projDots });
        g.point(p0, { r: 3, color: 'ghost', alpha: s.projDots });
      }
      if (s.projDots > 0.9 && N.det(M) < 0.05) g.label('six points → one point', [1, 0], { color: 'mag', size: 12.5, anchor: 'below', dy: 14 });
    }
    if (s.rings > 0) { g.marker.ring(c1, { r: 10 + 8 * s.rings, color: 'i' }); g.marker.ring(c2, { r: 10 + 8 * s.rings, color: 'j' }); }
    if (s.gold > 0) {
      const d1 = [0.7071, 0.7071], d2 = [0.7071, -0.7071];
      g.line(N.scale(d1, -1.5), N.scale(d1, 1.5), { color: 'gold', width: 2.6, alpha: s.gold });
      g.line(N.scale(d2, -0.5), N.scale(d2, 0.5), { color: 'gold', width: 2.6, alpha: s.gold });
      g.label('stretch ×1.5', N.scale(d1, 1.5), { color: 'gold', size: 12.5, anchor: 'left', dx: 8, alpha: s.gold });
      g.label('squeeze ×0.5', N.scale(d2, 0.5), { color: 'gold', size: 12.5, anchor: 'left', dx: 8, dy: 8, alpha: s.gold });
    }
    if (s.hudQ !== null) {   // the on-canvas bracket of the mystery step
      const x = g.size.w - 132, y = 18;
      g.hud('A =', x - 40, y + 28, { size: 15, color: 'hud' });
      hudMat(g, cellsOf(M, s.hudQ), x, y, { cw: 50 });
    }
    if (s.tag) g.hud(s.tag, 14, 22, { color: s.tagColor || 'hud', size: 14 });
    if (s.live && s.kind === 'guess' && s.guess !== null) {
      const T = GUESSES[s.guess % GUESSES.length], ok = N.dist(c1, N.col(T, 0)) < LA.TOL.pos && N.dist(c2, N.col(T, 1)) < LA.TOL.pos;
      g.hud(ok ? 'match!' : 'drag î and ĵ until your grid lies on the gold one', 14, 22, { color: ok ? 'ok' : 'gold' });
      if (ok) g.marker.check([g.size.w - 40, 30], { screen: true, size: 26 });
    }
    const drag = (col) => (p, st) => { const A = LA.clone(curM(st)); A[0][col] = p[0]; A[1][col] = p[1]; st.M = A; if (st.kind !== 'guess') st.kind = 'free'; };
    g.handle('i', c1, drag(0), { color: 'i' });
    g.handle('j', c2, drag(1), { color: 'j' });
  }

  /* step 6: reflection on the left, a rotation sweeping on the right that never matches it */
  function renderMisc(g, s) {
    const camL = fitCam(g, 0.5, 1, { center: [0.1, 0.2], halfH: 1.9, halfW: 1.55 });
    g.split((g2, r) => {
      drawPlane(g2, FLIP, { range: 5 });
      g2.lineThrough([0, 0], [1, 1], { color: 'muted', dashed: true, width: 1.4 });
      g2.hud('reflection F: violet = flipped', r.x + 12, r.y + 20, { color: 'neg' });
    }, (g2, r) => {
      const R = N.rot2(s.theta);
      g2.vector([0, 1], { color: 'i', dashed: true, width: 2, alpha: 0.55, label: 'i', hat: true, dx: -14, dy: 0 });
      g2.vector([1, 0], { color: 'j', dashed: true, width: 2, alpha: 0.55, label: 'j', hat: true, dx: 0, dy: 16 });
      drawPlane(g2, R, { range: 5 });
      g2.hud('rotation by ' + (Math.round(s.theta / DEG) % 360) + '° (dashed: where F puts î, ĵ)', r.x + 12, r.y + 20, { color: 'hud' });
      if (s.cross) {
        g2.label('reflection = rotation 180°?', [r.x + r.w / 2, r.y + r.h - 70], { screen: true, color: 'bad', size: 15 });
        g2.marker.cross([r.x + r.w / 2 + 118, r.y + r.h - 70], { screen: true, size: 24 });
      }
    }, { left: { view: '2d', cam: camL }, right: { view: '2d', cam: camL } });
  }

  /* step 9: the five animals on one shelf */
  function renderShelf(g, s) {
    const t = s.shelfT, active = t >= 0.92 ? -1 : Math.min(4, Math.floor(t / 0.18));
    const rects = [{ x: 0, y: 0, w: 1 / 3, h: 0.5 }, { x: 1 / 3, y: 0, w: 1 / 3, h: 0.5 }, { x: 2 / 3, y: 0, w: 1 / 3, h: 0.5 }, { x: 0, y: 0.5, w: 0.5, h: 0.5 }, { x: 0.5, y: 0.5, w: 0.5, h: 0.5 }];
    ZOO.forEach((z, i) => {
      const R = rects[i], lit = active === -1 || active === i, a = lit ? 1 : 0.28;
      const cam = fitCam(g, R.w, R.h, { center: [0.5, 0.05], halfH: 1.75, halfW: 1.85 });
      g.viewport(R, { view: '2d', cam }, (g2, r) => {
        drawPlane(g2, z.M, { alpha: a, range: 5 });
        g2.hud(z.name, r.x + 10, r.y + 18, { color: lit ? 'gold' : 'muted', size: 14 });
        const w = 2 * 40 + 8;
        g2.label(z.sym + ' =', [r.x + r.w / 2 - w / 2 - 22, r.y + r.h - 50], { screen: true, color: 'hud', size: 14, alpha: a, bg: false });
        hudMat(g2, cellsOf(z.M, false), r.x + r.w / 2 - w / 2, r.y + r.h - 72, { cw: 40, ch: 22, size: 13, alpha: a });
        if (active === i) { const W = (x, y) => g2.P.toWorld(x, y); g2.polyline([W(r.x + 3, r.y + 3), W(r.x + r.w - 3, r.y + 3), W(r.x + r.w - 3, r.y + r.h - 3), W(r.x + 3, r.y + r.h - 3)], { color: 'gold', width: 2, closed: true }); }
      });
    });
  }

  /* step 10: standardizing a cloud is a diagonal matrix; augmentation rotates pixel positions */
  function renderML(g, s) {
    const u = s.mlT, rot = s.mlRot;
    g.split((g2, r) => {
      const D = lerpM(I2(), STD, u);
      g2.ghostGrid();
      g2.grid(D, { alpha: 0.35 });
      g2.points(CLOUD.map(p => N.matVec(D, p)), { r: 3.5, color: 'data' });
      g2.ellipse(N.matMul(D, [[2.5, 0], [0, 0.5]]), [0, 0], { stroke: 'teal', width: 2 });
      g2.hud('synthetic cloud: σ₁ = 2.5, σ₂ = 0.5', r.x + 12, r.y + 20, { color: 'muted' });
      g2.hud(u > 0.95 ? 'standardized: round, σ = 1 both ways' : 'standardize with D = diag(1/σ₁, 1/σ₂)', r.x + 12, r.y + 40, { color: 'teal' });
      g2.hud('Part 5: gradient descent', r.x + 12, r.y + r.h - 20, { color: 'muted' });
    }, (g2, r) => {
      const R = N.rot2(rot), n = 28, S = 2.4;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = DIGIT[i * n + j]; if (v < 0.08) continue;
        const x0 = (j / n - 0.5) * S, x1 = ((j + 1) / n - 0.5) * S, y1 = (0.5 - i / n) * S, y0 = (0.5 - (i + 1) / n) * S;
        g2.polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(p => N.matVec(R, p)), { fill: 'white', alpha: 0.9 * v });
      }
      const h = S / 2; g2.polyline([[-h, -h], [h, -h], [h, h], [-h, h]].map(p => N.matVec(R, p)), { color: 'muted', closed: true, width: 1.2 });
      g2.vector(N.matVec(R, [1.5, 0]), { color: 'i', label: 'i', hat: true, width: 2.4, alpha: 0.9 });
      g2.vector(N.matVec(R, [0, 1.5]), { color: 'j', label: 'j', hat: true, width: 2.4, alpha: 0.9 });
      g2.hud('synthetic 28×28 digit', r.x + 12, r.y + 20, { color: 'muted' });
      g2.hud('augmentation: R(' + Math.round(((rot / DEG) + 360) % 360) + '°) moves every pixel', r.x + 12, r.y + 40, { color: 'hud' });
    }, { left: { view: '2d', cam: fitCam(g, 0.5, 1, { center: [0, 0], halfH: 2, halfW: LA.lerp(6.6, 3.2, u) }) }, right: { view: '2d', cam: fitCam(g, 0.5, 1, { center: [0, 0], halfH: 2.2, halfW: 1.9 }) } });
  }

  function renderRecap(g, s) {
    const cards = [
      { title: '1 · rotation keeps the circle', M: N.rot2(Math.PI / 4), o: { showCircle: true } },
      { title: '2 · shear keeps base and height', M: SHEAR, o: {} },
      { title: '3 · projection loses a dimension', M: PROJ, o: {} },
    ];
    cards.forEach((c, i) => {
      const R = { x: 0, y: i / 3, w: 1, h: 1 / 3 };
      g.viewport(R, { view: '2d', cam: fitCam(g, 1, 1 / 3, { center: [0.55, 0.3], halfH: 1.5, halfW: 2 }) }, (g2, r) => {
        drawPlane(g2, c.M, Object.assign({ range: 6 }, c.o));
        if (i === 1) { g2.dim([0, 0], [1, 0], { label: 'base 1', offset: 16 }); g2.dim([2, 0], [2, 1], { label: 'height 1', offset: -16 }); }
        if (i === 2) {
          g2.lineThrough([0, 0], [1, 0], { color: 'teal', width: 3, alpha: 0.7 });
          for (let y = -1; y <= 1; y++) if (y) { g2.line([1, y], [1, 0], { color: 'residual', dashed: true, width: 1.2 }); g2.point([1, y], { r: 3, color: 'ghost' }); }
          g2.point([1, 0], { r: 5, color: 'mag' });
        }
        g2.hud(c.title, r.x + 10, r.y + 18, { color: 'muted' });
      });
    });
  }

  LA.registerLesson({
    id: 'p2-03-transformation-zoo',
    part: 2,
    title: 'Rotation, scaling, shear, reflection, projection',
    subtitle: 'Five animals, five matrices: read each one off where î and ĵ land.',
    minutes: 10,
    prereqs: ['p2-01-linear-transformations'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'standardizing 50 features is a 50×50 diagonal scaling; data augmentation rotates 28×28 images', transfers: 'diagonal = per-axis scaling, orthogonal = rotation or reflection, rank-deficient = projection, in every dimension; only the names “line” and “plane” change' },
    cantShow: ['a rotation in ℝⁿ as “about an axis” (n-D rotations turn planes; only 3D has an axis)'],
    terms: [
      { term: 'rotation matrix', def: 'The matrix with columns [cos θ; sin θ] and [−sin θ; cos θ]; it turns everything by θ and keeps lengths.' },
      { term: 'scaling matrix', def: 'A diagonal matrix: each axis is stretched by its own factor.' },
      { term: 'diagonal matrix', def: 'A matrix whose nonzero entries all sit on the diagonal.' },
      { term: 'shear', def: 'î stays put while ĵ leans sideways; squares become parallelograms of the same area.' },
      { term: 'reflection', def: 'A flip across a line; it reverses orientation.' },
      { term: 'orientation', def: 'Whether going from î to ĵ turns counter-clockwise (as usual) or clockwise (flipped).' },
      { term: 'projection matrix', def: 'A matrix that flattens the plane onto a line; a dimension is lost.' },
    ],
    atlas: ['normalization', 'svd', 'optimization'],
    thumbnail: 4,

    state() {
      return {
        view: '2d', cam: { center: [0.3, 0.35], halfH: 2.6 },
        M: I2(), kind: 'free', live: false, theta: 0, sx: 2, sy: 0.5, k: 1, guess: null,
        viewMode: 'main', mathMode: null,
        showGhost: true, showSquare: true, showCircle: true, coordLabels: false, fan: 0, rightMark: false, dims: false,
        mirror: false, showProjLine: false, projDots: 0, rings: 0, gold: 0, hudQ: null, tag: null, tagColor: null, cross: false,
        shelfT: 0, mlT: 0, mlRot: 0,
      };
    },

    render(g, s, ctx) {
      if (s.viewMode === 'misc') return renderMisc(g, s);
      if (s.viewMode === 'shelf') return renderShelf(g, s);
      if (s.viewMode === 'ml') return renderML(g, s);
      if (s.viewMode === 'recap') return renderRecap(g, s);
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const A = tex.sym('A'), eq = tex.op('='), M = curM(s);
      const mat = (X, o) => tex.mat(X, Object.assign({ colors: 'cols' }, o || {}));
      switch (s.mathMode) {
        case 'rot': return tex.block(
          tex.eq(tex.sym('R') + tex.paren(tex.sym('θ')), eq, mat([['cos θ', '−sin θ'], ['sin θ', 'cos θ']])),
          tex.eq(tex.sym('θ'), eq, tn(tex, Math.round(s.theta / DEG) + '°', 'gold'), tex.op(':'), tex.sym('R'), eq, mat(M)));
        case 'len': {
          const c1 = N.col(M, 0), c2 = N.col(M, 1);
          return tex.block(tex.eq(tex.norm(tex.sym('R') + tex.sym('v')), eq, tex.norm(tex.sym('v'))),
            tex.eq(tex.norm(tex.text('col₁')), eq, tn(tex, LA.fmt.fixed(N.norm(c1), 2), 'i'), tex.op(','), tex.text('col₁ · col₂'), eq, tn(tex, LA.fmt.fixed(Math.abs(N.dot(c1, c2)) < 5e-3 ? 0 : N.dot(c1, c2), 2), 'gold')),
            tex.eq(tex.sym('R'), eq, mat(M)));
        }
        case 'scale': return tex.block(tex.eq(tex.sym('S'), eq, mat(M)), tex.text('only the diagonal is nonzero'));
        case 'shear': return tex.block(tex.eq(tex.sym('H'), eq, mat(M, { highlight: [0, 1] })), tex.eq(tex.text('area factor'), eq, tn(tex, N.det(M), 'v')));
        case 'refl': return tex.block(tex.eq(tex.sym('F'), eq, mat(M)), tex.eq(tex.text('area factor'), eq, tn(tex, N.det(M), N.det(M) < 0 ? 'neg' : 'v')), N.det(M) < 0 ? tex.text('orientation flipped', { color: 'neg' }) : '');
        case 'misc': return tex.block(tex.eq(tex.sym('R') + tex.paren(tn(tex, '180°')), eq, mat(HALF), tex.op('≠'), tex.sym('F'), eq, mat(FLIP)),
          tex.eq(tex.text('now: R(' + (Math.round(s.theta / DEG) % 360) + '°)'), eq, mat(N.rot2(s.theta))));
        case 'proj': return tex.block(tex.eq(tex.sym('P'), eq, mat(M)), tex.text(N.det(M) < 0.02 ? 'a zero column: rank 1' : 'ĵ is being squashed to the origin'));
        case 'mystery': return tex.block(tex.eq(A, eq, s.hudQ ? mat([['?', '?'], ['?', '?']]) : mat(M)));
        case 'shelf': return tex.block.apply(null, ZOO.map(z => tex.eq(tex.sym(z.sym), eq, mat(z.M.map(r => r.map(x => Math.round(x * 100) / 100))))));
        case 'ml': return tex.block(tex.eq(tex.sym('D'), eq, mat([['1/σ₁', '0'], ['0', '1/σ₂']]), eq, mat(STD)), tex.eq(tex.sym('R') + tex.paren(tn(tex, Math.round(((s.mlRot / DEG) + 360) % 360) + '°')), eq, mat(N.rot2(s.mlRot))));
        case 'recap': return '';
        default: {
          const names = { rotation: 'rotation', scale: 'scaling', shear: 'shear', reflection: 'reflection', projection: 'projection', free: 'your matrix', guess: 'your matrix' };
          const d = N.det(M);
          return tex.block(tex.eq(A, eq, mat(M)), tex.eq(tex.text((names[s.kind] || '') + ' ·'), tex.text('area factor'), eq, tn(tex, d, d < 0 ? 'neg' : 'v')));
        }
      }
    },

    steps: [
      { title: 'Rotation: columns from the circle', duration: 5,
        narration: 'First animal in the zoo: the <b>rotation matrix</b> for an angle θ. The unit circle is drawn; the green ' + ihat + '’s tip slides along it to [cos θ; sin θ] and the red ' + jhat + '’s to [−sin θ; cos θ]. Those are the columns. At θ = 90° the matrix is [[0, −1],[1, 0]]; at 45°, all four entries are ±0.71.',
        setup(s) { s.viewMode = 'main'; s.mathMode = 'rot'; s.live = false; s.M = I2(); s.theta = 0; s.showCircle = true; s.coordLabels = true; s.fan = 0; s.rightMark = false; s.dims = false; s.mirror = false; s.showProjLine = false; s.projDots = 0; s.rings = 0; s.gold = 0; s.hudQ = null; s.tag = null; s.cross = false; },
        at(s, t) { const th = LA.lerp(0, 90, seg(t, 0, 0.5)) + LA.lerp(0, -45, seg(t, 0.6, 1)); s.theta = th * DEG; s.M = N.rot2(s.theta); } },

      { title: 'Rotations keep lengths', duration: 3.5,
        narration: 'Under a rotation the unit circle stays the unit circle. Every gray fan arrow keeps its length and every angle keeps its size: the grid turns as a rigid plate. The two columns stay perpendicular unit vectors — remember this; Part 3 names such matrices orthogonal.',
        setup(s) { s.mathMode = 'len'; s.coordLabels = false; s.rightMark = true; s.fan = 0; s.theta = 45 * DEG; s.M = N.rot2(s.theta); },
        at(s, t) { s.fan = seg(t, 0, 0.2); s.theta = LA.lerp(45, 165, seg(t, 0.15, 0.95)) * DEG; s.M = N.rot2(s.theta); } },

      { title: 'Scaling: a diagonal matrix', duration: 4,
        narration: 'A <b>scaling matrix</b> stretches each axis by its own factor: the green ' + ihat + ' ↦ [2; 0], the red ' + jhat + ' ↦ [0; 0.5]. Only the diagonal entries are nonzero — a <b>diagonal matrix</b>. The unit circle becomes an ellipse lined up with the axes; equal factors would simply zoom.',
        setup(s) { s.mathMode = 'scale'; s.fan = 0; s.rightMark = false; s.M = I2(); },
        at(s, t) { s.M = t < 0.72 ? lerpM(I2(), SCALE, seg(t, 0.05, 0.6)) : lerpM(SCALE, ZOOM, seg(t, 0.75, 0.98)); } },

      { title: 'Shear: î stays, ĵ leans', duration: 4,
        narration: 'A <b>shear</b>: the green ' + ihat + ' stays where it is while the red ' + jhat + ' leans over to [1; 1]. Squares become parallelograms, but each keeps its base and height, so area is unchanged — a fact the determinant lesson will measure. The matrix is [[1, 1],[0, 1]]: the lean is the off-diagonal 1.',
        setup(s) { s.mathMode = 'shear'; s.M = I2(); s.dims = true; s.showCircle = true; },
        at(s, t) { s.M = lerpM(I2(), SHEAR, seg(t, 0.05, 0.75)); } },

      { title: 'Reflection: orientation flips', duration: 4.5,
        narration: 'A <b>reflection</b> across the line y = x swaps ' + ihat + ' and ' + jhat + ': the matrix is [[0, 1],[1, 0]]. Watch the curved arrow in the square: î-then-ĵ used to turn counter-clockwise; now it turns clockwise. The sheet has been turned over, and the fill switches to violet to mark the flipped <b>orientation</b>.',
        setup(s) { s.mathMode = 'refl'; s.dims = false; s.M = I2(); s.mirror = true; s.tag = null; },
        at(s, t) { s.M = lerpM(I2(), FLIP, seg(t, 0.05, 0.8)); s.tag = t > 0.6 ? 'orientation flipped' : null; s.tagColor = 'neg'; } },

      { title: 'A reflection is not a half-turn', duration: 5, misconception: true,
        narration: 'Belief: “a reflection is just a rotation by 180°.” On the right a rotation sweeps a full turn, then stops at 180°. Its square stays yellow — orientation preserved — and at no angle do both arrows land on the dashed spots where the reflection sends them. You cannot flip the plate over without leaving the plane.',
        setup(s) { s.viewMode = 'misc'; s.mathMode = 'misc'; s.mirror = false; s.tag = null; s.theta = 0; s.cross = false; },
        at(s, t) { s.theta = LA.lerp(0, 540, seg(t, 0, 0.8, 'linear')) * DEG; s.cross = t > 0.84; } },   // a full turn, then stop at the half-turn

      { title: 'Projection: a dimension is lost', duration: 4.5,
        narration: 'A <b>projection matrix</b> onto the x-axis: [[1, 0],[0, 0]]. The green ' + ihat + ' stays, the red ' + jhat + ' is sent to the origin. The whole plane collapses onto the teal line; a column of points lands on one point. There is no way back — many inputs share one output. This is Part 1’s shadow as a matrix.',
        setup(s) { s.viewMode = 'main'; s.mathMode = 'proj'; s.cross = false; s.M = I2(); s.showCircle = false; s.showProjLine = false; s.projDots = 0; s.cam = { center: [0.4, 0.5], halfH: 3.2 }; },
        at(s, t) { s.projDots = seg(t, 0, 0.15); s.M = lerpM(I2(), PROJ, seg(t, 0.15, 0.85)); s.showProjLine = t > 0.6; s.tag = t > 0.85 ? 'rank 1: every output lies on one line' : null; s.tagColor = 'teal'; } },

      { title: 'Reading a mystery matrix', duration: 5,
        narration: 'A grid moves and you do not know its matrix. Read it off: the green ' + ihat + ' landed on [1; 0.5], the red ' + jhat + ' on [0.5; 1]. So the matrix is [[1, 0.5],[0.5, 1]] — a stretch along the diagonal direction and a squeeze across it. Any grid motion can be read this way, two arrows at a time.',
        setup(s) { s.mathMode = 'mystery'; s.projDots = 0; s.showProjLine = false; s.tag = null; s.M = I2(); s.showCircle = true; s.hudQ = true; s.rings = 0; s.gold = 0; s.cam = { center: [0.3, 0.35], halfH: 2.6 }; },
        at(s, t) { s.M = lerpM(I2(), MYST, seg(t, 0.03, 0.45)); s.rings = seg.pulse(t, 0.5, 0.9); s.hudQ = t < 0.7; s.coordLabels = t >= 0.5; s.gold = seg(t, 0.8, 0.95); } },

      { title: 'The zoo on one shelf', duration: 4,
        narration: 'Five animals, five matrices. Rotation: cos and sin in the columns, lengths kept. Scaling: a diagonal. Shear: a 1 off the diagonal. Reflection: columns swapped, orientation flipped. Projection: a zero column, a dimension lost. Every 2×2 matrix is a blend of these — Part 4 shows each one is a rotation, a scaling, and a rotation.',
        setup(s) { s.viewMode = 'shelf'; s.mathMode = 'shelf'; s.hudQ = null; s.gold = 0; s.rings = 0; s.shelfT = 0; },
        at(s, t) { s.shelfT = t; } },

      { title: 'Standardizing data is a diagonal matrix', duration: 6, mlTie: true,
        narration: 'A synthetic cloud stretched five times more along x than along y makes gradient descent zig-zag. Standardizing — dividing each feature by its spread — is the diagonal matrix diag(1/σ₁, 1/σ₂): a scaling that makes the cloud round. Rotating images for data augmentation is the rotation matrix applied to pixel positions.',
        setup(s) { s.viewMode = 'ml'; s.mathMode = 'ml'; s.mlT = 0; s.mlRot = 0; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.55); s.mlRot = (LA.lerp(0, 20, seg(t, 0.6, 0.76)) + LA.lerp(0, -35, seg(t, 0.82, 0.98))) * DEG; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Read every matrix from where ' + ihat + ' and ' + jhat + ' land. Rotation keeps lengths; scaling stretches the axes; shear leans ĵ and keeps area; reflection flips orientation and turns the fill violet; projection collapses a dimension. Standardizing data is a diagonal scaling.',
        cards: ['<b>Rotation:</b> columns [cos θ; sin θ] and [−sin θ; cos θ]; lengths and angles kept.', '<b>Scale</b> = diagonal; <b>shear</b> = one off-diagonal entry; <b>reflection</b> = swap, violet fill.', '<b>Projection:</b> a zero column; a dimension is lost and cannot come back.'],
        setup(s) { s.viewMode = 'recap'; s.mathMode = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Pick an animal and move its slider: θ drives rotation and the reflection / projection line, the scale sliders drive scaling, “shear” drives the shear. Drag ' + ihat + ' or ' + jhat + ' to go free. Guess mode: match the gold grid.</p>',
      setup(s) { s.viewMode = 'main'; s.mathMode = null; s.live = true; s.kind = 'rotation'; s.theta = Math.PI / 4; s.showCircle = true; s.mirror = false; s.guess = null; s.cam = { center: [0.3, 0.35], halfH: 2.8 }; },
      controls: [
        { type: 'preset', label: 'Transformation', items: [
          { label: 'Rotation', apply(s) { s.kind = 'rotation'; s.theta = Math.PI / 4; } },
          { label: 'Scaling', apply(s) { s.kind = 'scale'; s.sx = 2; s.sy = 0.5; } },
          { label: 'Shear', apply(s) { s.kind = 'shear'; s.k = 1; } },
          { label: 'Reflection', apply(s) { s.kind = 'reflection'; s.theta = Math.PI / 4; } },
          { label: 'Projection', apply(s) { s.kind = 'projection'; s.theta = 0; } },
          { label: 'Free', apply(s) { s.M = LA.clone(curM(s)); s.kind = 'free'; } },
          { label: 'Guess mode', apply(s) { s.kind = 'guess'; s.M = I2(); s.guess = s.guess === null ? 0 : s.guess; } },
        ] },
        { type: 'slider', bind: 'theta', min: 0, max: 6.283, step: 0.01, label: 'θ', color: 'gold', fmt: x => LA.fmt.deg(x) },
        { type: 'slider', bind: 'sx', min: -3, max: 3, step: 0.05, label: 'scale x', color: 'i' },
        { type: 'slider', bind: 'sy', min: -3, max: 3, step: 0.05, label: 'scale y', color: 'j' },
        { type: 'slider', bind: 'k', min: -2, max: 2, step: 0.05, label: 'shear', color: 'j' },
        { type: 'button', label: 'Next mystery', onClick(s) { s.kind = 'guess'; s.M = I2(); s.guess = s.guess === null ? 0 : s.guess + 1; } },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + ihat + ' and ' + jhat + ' to make a 90° counter-clockwise rotation.',
        setup(s) { s.viewMode = 'main'; s.mathMode = null; s.live = true; s.kind = 'free'; s.M = I2(); s.showCircle = true; s.mirror = false; },
        judge(s) { const M = curM(s); return LA.judge.all(LA.judge.near(N.col(M, 0), [0, 1]), LA.judge.near(N.col(M, 1), [-1, 0])); },
        hint: 'Both arrows turn a quarter turn the same way: ' + ihat + ' straight up, ' + jhat + ' straight left. The circle must stay a circle.',
        solution(s) { s.kind = 'free'; s.M = [[0, -1], [1, 0]]; } },
      { kind: 'manipulate',
        prompt: 'Make a reflection across the dashed line y = x.',
        setup(s) { s.viewMode = 'main'; s.mathMode = null; s.live = true; s.kind = 'free'; s.M = I2(); s.showCircle = false; s.mirror = true; },
        judge(s) { const M = curM(s); return LA.judge.all(LA.judge.near(N.col(M, 0), [0, 1]), LA.judge.near(N.col(M, 1), [1, 0])); },
        hint: 'Swap the two arrows; the square should turn violet.',
        solution(s) { s.kind = 'free'; s.M = [[0, 1], [1, 0]]; } },
      { kind: 'manipulate',
        prompt: 'Make a shear that keeps ' + ihat + ' fixed and sends ' + jhat + ' to [1; 1].',
        setup(s) { s.viewMode = 'main'; s.mathMode = null; s.live = true; s.kind = 'free'; s.M = I2(); s.showCircle = false; s.mirror = false; },
        judge(s) { const M = curM(s); return LA.judge.all(LA.judge.near(N.col(M, 0), [1, 0]), LA.judge.near(N.col(M, 1), [1, 1])); },
        hint: 'Leave the green arrow alone; lean the red one to the right by one unit.',
        solution(s) { s.kind = 'free'; s.M = [[1, 1], [0, 1]]; } },
    ],
  });
})();
