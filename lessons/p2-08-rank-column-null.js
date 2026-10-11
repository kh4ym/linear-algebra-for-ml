/* p2-08 — Rank, column space, null space. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg;
  const I2 = () => [[1, 0], [0, 1]];
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const SING = [[1, 2], [2, 4]], INV2 = [[2, 1], [1, 1]];
  const R2 = [[1, 0, 1], [0, 1, 1], [1, 1, 2]], R1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]], FULL = [[1, 2, 3], [2, 4, 6], [3, 6, 9]];
  const NULL2 = [2, -1], NULL3 = [1, 1, -1];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const bS = '<span class="c-b">b</span>', xS = '<span class="c-v">x</span>';
  const cam2 = (cx, cy, hh) => Object.assign(LA.cam.defaults2d(), { center: [cx, cy], halfH: hh });
  const cam3 = (o) => Object.assign(LA.cam.defaults3d(), o);
  const TOLR = 0.02; // relative singular-value cut for "rank" in the playground

  // 40 synthetic points lying exactly in the plane spanned by [1, 0, 0.5] and [0, 1, 0.5]
  function makeCloud() {
    const r = LA.rng(28), pts = [];
    for (let i = 0; i < 40; i++) { const a = LA.rng.gauss(r), b = LA.rng.gauss(r); pts.push([a, b, 0.5 * a + 0.5 * b].map(x => Math.round(x * 1000) / 1000)); }
    return pts;
  }
  // a 12×12 synthetic "weight" sample and a rank-1 update u vᵀ (for the LoRA picture)
  function makeW() {
    const r = LA.rng(81), W = [];
    for (let i = 0; i < 12; i++) { W.push([]); for (let j = 0; j < 12; j++) W[i].push(Math.round(LA.rng.gauss(r) * 0.8 * 100) / 100); }
    const u = [], v = [];
    for (let i = 0; i < 12; i++) { u.push(Math.round(Math.sin(i * 0.7) * 1.4 * 100) / 100); v.push(Math.round(Math.cos(i * 0.5) * 1.2 * 100) / 100); }
    return { W, u, v };
  }
  const angleLine = (a, d) => { const na = N.norm(a), nd = N.norm(d); if (na < 1e-9) return 90; return Math.acos(Math.min(1, Math.abs(N.dot(a, d)) / (na * nd))) * 180 / Math.PI; };
  // snap a dragged 2D point onto the line through 0 along d when it is within tol (world units)
  const snapLine = (p, d, tol) => { const u = N.normalize(d), k = N.dot(p, u), f = N.scale(u, k); return N.dist(p, f) < tol ? f : p.slice(); };
  const rankOf = (M) => { const S = N.svd(M).S; const smax = S[0] || 0; return smax < 1e-9 ? 0 : S.filter(x => x > TOLR * smax).length; };

  /* stacked rank/nullity bar (screen HUD): teal rank at the bottom, magenta nullity on top */
  function rankBars(g, rank, nullity, x, y, h) {
    const n = rank + nullity; h = h || 120;
    g.bars([n], { screen: true, x, y, w: 46, h, max: n, colors: ['mag'], alpha: 0.85 });
    if (rank > 0) g.bars([rank], { screen: true, x, y, w: 46, h, max: n, colors: ['teal'], alpha: 0.95 });
    g.hud('rank ' + rank, x + 54, y + h - (rank / n) * h / 2, { color: 'teal', size: 13.5 });
    if (nullity > 0) g.hud('nullity ' + nullity, x + 54, y + (nullity / n) * h / 2, { color: 'mag', size: 13.5 });
    g.hud(n + ' inputs = ' + rank + ' + ' + nullity, x - 4, y - 14, { color: 'hud', size: 14 });
  }

  LA.registerLesson({
    id: 'p2-08-rank-column-null',
    part: 2,
    title: 'Rank, column space, null space',
    subtitle: 'What survives the map (teal), what is crushed to zero (magenta), and how they add up.',
    minutes: 10,
    prereqs: ['p2-07-inverse-solving', 'p1-03-linear-combinations-span'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a 784×128 layer has rank ≤ 128; a LoRA update has rank 4–64; a data matrix with duplicated features has rank below its column count', transfers: 'rank + nullity = number of inputs for every matrix; column space = span of the columns = all reachable outputs, in any dimension' },
    cantShow: ['a 128-dimensional column space inside ℝ⁷⁸⁴ (we show a plane in 3D and count with bars)'],
    terms: [
      { term: 'column space', def: 'The span of the columns: every output the matrix can produce.' },
      { term: 'rank', def: 'The dimension of the column space: how many independent directions survive the map.' },
      { term: 'null space', def: 'All inputs that the matrix sends to the zero vector.' },
      { term: 'nullity', def: 'The dimension of the null space: how many input directions are lost.' },
      { term: 'rank–nullity', def: 'rank + nullity = the number of input dimensions.' },
    ],
    atlas: ['pca', 'svd', 'linear-regression'],
    thumbnail: 5,

    state() {
      const d = makeW();
      const cloud = makeCloud();
      return {
        view: '2d', cam: cam2(0.5, 0.8, 3.6), scene: 'grid',
        G: I2(), showGhost: true, showCol: 0, inset: 0, rankPanel: false, b: null, solLine: 0, xMark: null,
        showNull: 0, nullPts: -1, slide: -1, bars: null,
        M3t: LA.clone(R2), morph3: 0, showCol3: 0, showNull3: 0, claim: null,
        cloud, sv: N.svd(cloud).S, W: d.W, u: d.u, v: d.v, mlT: 0, panel: 'col',
        pg: false, chk: null, M: LA.clone(SING), M3: LA.clone(R2), pgCol: true, pgNull: true, bb: [1, 3], xIn: [1, 1],
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.pg && ctx.mode === 'play') { renderPlay(g, s); return; }
      if (s.chk) { renderCheck(g, s); return; }
      if (s.view === '3d') { render3d(g, s); return; }
      const G = s.G;
      if (s.showGhost) g.ghostGrid();
      g.grid(G, { alpha: 0.85 });
      if (s.showCol > 0) {
        g.lineThrough([0, 0], [1, 2], { color: 'teal', width: 3, alpha: s.showCol });
        g.label('column space: the line y = 2x', [1.6, 3.2], { color: 'teal', anchor: 'right', dx: -10, size: 13.5, font: 'ui', alpha: s.showCol });
      }
      if (s.showNull > 0) {
        g.lineThrough([0, 0], NULL2, { color: 'mag', width: 2.6, alpha: s.showNull });
        g.label('null space: multiples of [2; −1]', [2.6, -1.3], { color: 'mag', anchor: 'below', dy: 12, size: 13.5, font: 'ui', alpha: s.showNull });
      }
      if (s.nullPts >= 0) {
        const T = lerpM(I2(), SING, s.nullPts);
        [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2].forEach(k => { const p = N.scale(NULL2, k * 0.75); g.point(p, { r: 3, color: 'mag', alpha: 0.35 }); g.point(N.matVec(T, p), { r: 5, color: 'mag' }); });
        if (s.nullPts >= 1) g.label('all crushed to 0', [0, 0], { color: 'mag', anchor: 'right', dx: -16, dy: 20, size: 13, font: 'ui' });
      }
      if (s.solLine > 0) {
        g.lineThrough([0.5, 0], NULL2, { color: 'mag', width: 2.2, dashed: true, alpha: s.solLine });
        g.label('solutions x = [0.5; 0] + t·[2; −1]', [-1.1, 0.8], { color: 'mag', anchor: 'above', dy: -12, size: 13, font: 'ui', alpha: s.solLine });
      }
      const ci = N.col(G, 0), cj = N.col(G, 1);
      g.vector(ci, { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(cj, { color: 'j', label: 'j', hat: true, width: 3 });
      if (s.slide >= 0) {
        const x = N.add([1, 0], N.scale(NULL2, s.slide));
        g.vector(x, { color: 'v', label: 'x', width: 3 });
        g.point(N.matVec(SING, x), { r: 7, color: 'b', stroke: 'white' });
        g.label('Ax stays at (1, 2)', N.matVec(SING, x), { color: 'b', anchor: 'right', dx: -14, size: 13.5, font: 'ui' });
        g.line(x, N.matVec(SING, x), { color: 'v', dashed: true, width: 1.2, alpha: 0.6 });
      }
      if (s.b) {
        g.vector(s.b, { color: 'b', label: 'b', width: 3 });
        if (s.xMark) {
          if (s.xMark.ok) { g.marker.check(s.b, { dx: 22, dy: -12 }); g.label('on the teal line: solvable', s.b, { color: 'ok', anchor: 'left', dx: 36, dy: -12, size: 13, font: 'ui' }); }
          else { g.marker.cross(s.b, { dx: 22, dy: -12, size: 22 }); g.label('off the line: no solution', s.b, { color: 'bad', anchor: 'left', dx: 38, dy: -12, size: 13, font: 'ui' }); }
        }
      }
      if (s.inset > 0) {
        const W = g.size.w;
        g.viewport({ x: 0.58, y: 0.66, w: 0.4, h: 0.31 }, { view: '2d', cam: cam2(0.9, 0.7, 2.0), bg: 'bg' }, (g2, r) => {
          g2.ghostGrid({ range: 6 }); g2.grid(INV2, { range: 6, alpha: 0.6 });
          const b = g2.P.bounds();
          g2.polygon([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], { fill: 'teal', alpha: 0.16 * s.inset });
          g2.vector(N.col(INV2, 0), { color: 'i', label: 'i', hat: true, width: 2.4 });
          g2.vector(N.col(INV2, 1), { color: 'j', label: 'j', hat: true, width: 2.4 });
          g2.hud('invertible: Col = whole plane', r.x + 8, r.y + 16, { size: 12.5, color: 'teal' });
        });
      }
      if (s.rankPanel) {
        g.bars([1, 2], { screen: true, x: 24, y: 92, w: 150, h: 110, max: 2, colors: ['teal', 'teal'], labels: ['A: line', 'invertible: plane'], values: true });
        g.hud('rank = surviving directions', 24, 64, { size: 12.5, color: 'muted' });
      }
      if (s.bars) rankBars(g, s.bars.rank, s.bars.nullity, 26, 64, 110);
      if (s.claim) {
        const w = g.size.w;
        g.label(s.claim.text, [w / 2, 26], { screen: true, size: 15, color: 'hud', font: 'ui' });
        if (s.claim.cross) g.marker.cross([w / 2 + s.claim.text.length * 3.7 + 22, 26], { screen: true, size: 22 });
      }
    },

    math(s, tex) {
      const num = (x, o) => '<span class="tx-n ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';
      const P = [], p = s.panel;
      const Aeq = (M, name) => tex.eq(tex.sym(name || 'A'), tex.op('='), tex.mat(M, { colors: 'cols' }));
      const T = (w, c) => tex.text(w, { color: c });
      if (p === 'col') { P.push(Aeq(SING)); P.push(tex.eq(T('Col(A)'), tex.op('='), T('span'), tex.paren(tex.vec([1, 2], { color: 'i' }) + tex.op(',') + tex.vec([2, 4], { color: 'j' })), tex.op('='), T('line y = 2x', 'teal'))); }
      else if (p === 'rank') { P.push(Aeq(SING)); P.push(tex.eq(T('rank(A) = dim Col(A) ='), num(1, { color: 'teal' }))); }
      else if (p === 'solve') { P.push(Aeq(SING)); P.push(tex.eq(tex.sym('b', { color: 'b' }), tex.op('='), tex.vec(s.b || [0, 0], { color: 'b' }))); P.push(tex.eq(tex.sym('b', { color: 'b' }), tex.op('∈'), T('Col(A)', 'teal'), tex.op('⇔'), tex.sym('A'), tex.sym('x', { color: 'v' }), tex.op('='), tex.sym('b', { color: 'b' }), T('is solvable'))); }
      else if (p === 'null') { P.push(tex.eq(T('Null(A)'), tex.op('='), T('{ t ·'), tex.vec(NULL2, { color: 'null' }), T('}'))); P.push(tex.eq(tex.mat(SING, { colors: 'cols' }), tex.vec(NULL2, { color: 'null' }), tex.op('='), tex.vec([0, 0]))); }
      else if (p === 'rn') { P.push(tex.eq(T('rank', 'teal'), tex.op('+'), T('nullity', 'null'), tex.op('='), num(1, { color: 'teal' }), tex.op('+'), num(1, { color: 'null' }), tex.op('='), num(2), T('inputs'))); }
      else if (p === 'r2') { P.push(Aeq(s.M3t, 'M')); P.push(tex.eq(T('col₃ = col₁ + col₂ · rank', 'teal'), num(2, { color: 'teal' }), T(' · Null = span', 'null'), tex.vec(NULL3, { color: 'null' }))); }
      else if (p === 'r1') { P.push(Aeq(s.M3t, 'M')); P.push(tex.eq(T('rank', 'teal'), num(1, { color: 'teal' }), T('· nullity', 'null'), num(2, { color: 'null' }), T('· Null = plane ⟂', 'null'), tex.vec([1, 2, 3]))); }
      else if (p === 'mis') { P.push(tex.eq(tex.mat(FULL, { colors: 'cols' }), T('nine nonzero entries'))); P.push(tex.eq(T('rank ='), num(1, { color: 'teal' }), T('(every column is a multiple of [1; 2; 3])'))); }
      else if (p === 'cloud') { P.push(tex.eq(T('X: 40 × 3'), tex.op('·'), T('rank(X) ='), num(2, { color: 'teal' }), tex.op('<'), num(3), T('features'))); P.push(tex.eq(T('singular values'), tex.vec(s.sv, { color: 'gold' }))); }
      else if (p === 'ml') { P.push(tex.eq(T('rank(W + u vᵀ) ≤ rank(W) + 1'))); P.push(tex.eq(T('rank(XᵀX) = rank(X) = 2 < 3  ⇒  det(XᵀX) = 0', 'bad'))); }
      else if (p === 'play') {
        if (s.view === '3d') { const r = rankOf(s.M3); P.push(Aeq(s.M3, 'M')); P.push(tex.eq(T('rank', 'teal'), num(r, { color: 'teal' }), tex.op('+'), T('nullity', 'null'), num(3 - r, { color: 'null' }), tex.op('='), num(3), tex.op('·'), T('det'), num(N.det(s.M3)))); }
        else { const r = rankOf(s.M); P.push(Aeq(s.M)); P.push(tex.eq(T('rank', 'teal'), num(r, { color: 'teal' }), tex.op('+'), T('nullity', 'null'), num(2 - r, { color: 'null' }), tex.op('='), num(2), tex.op('·'), T('det'), num(N.det(s.M)))); }
      }
      else if (p === 'chk') { P.push(Aeq(s.M)); P.push(tex.eq(T('det'), num(N.det(s.M), { color: Math.abs(N.det(s.M)) < LA.TOL.det ? 'bad' : 'gold' }))); if (s.chk === 'b') P.push(tex.eq(tex.sym('b', { color: 'b' }), tex.op('='), tex.vec(s.bb, { color: 'b' }))); if (s.chk === 'x') P.push(tex.eq(tex.sym('A'), tex.sym('x', { color: 'v' }), tex.op('='), tex.vec(N.matVec(s.M, s.xIn)))); }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'The column space: everything the map can output', duration: 4,
        narration: 'A = [[1, 2],[2, 4]] sends the whole plane onto one line. That teal line is the set of every possible output — the <b>column space</b>, the span of the two columns. Both columns, the green [1; 2] and the red [2; 4], lie on it. Compare an invertible matrix, whose column space is the whole plane.',
        setup(s) { s.view = '2d'; s.scene = 'grid'; s.cam = cam2(0.5, 0.8, 3.6); s.G = I2(); s.showCol = 0; s.inset = 0; s.rankPanel = false; s.b = null; s.solLine = 0; s.xMark = null; s.showNull = 0; s.nullPts = -1; s.slide = -1; s.bars = null; s.claim = null; s.panel = 'col'; },
        at(s, t) { s.G = lerpM(I2(), SING, seg(t, 0.05, 0.55)); s.showCol = seg(t, 0.45, 0.6); s.inset = seg(t, 0.7, 0.85); } },

      { title: 'Rank counts surviving directions', duration: 3,
        narration: 'The <b>rank</b> is the dimension of the column space: 1 for this A (a line), 2 for the invertible matrix (the whole plane). Rank counts how many independent directions make it through the map. A rank-1 matrix remembers just one number about each input.',
        setup(s) { s.G = LA.clone(SING); s.showCol = 1; s.inset = 1; s.panel = 'rank'; },
        at(s, t) { s.rankPanel = t > 0.15; } },

      { title: 'Solvable exactly when b is in the column space', duration: 4,
        narration: 'A' + xS + ' = ' + bS + ' has a solution exactly when the orange b lies in the column space — when b is a reachable output. Off the teal line: no solution. Move it onto the line and a dashed magenta line of solutions x appears in the input picture. The column space is the map’s menu.',
        setup(s) { s.inset = 0; s.rankPanel = false; s.b = [0.5, 2.5]; s.xMark = null; s.solLine = 0; s.panel = 'solve'; },
        at(s, t) { s.b = lerpV([0.5, 2.5], [0.5, 1], seg(t, 0.35, 0.6)); s.xMark = t < 0.33 ? { ok: false } : (t > 0.62 ? { ok: true } : null); s.solLine = seg(t, 0.65, 0.9); } },

      { title: 'The null space: inputs sent to zero', duration: 4.5,
        narration: 'Some inputs are crushed to the origin. For this A every multiple of [2; −1] is: 1·2 + 2·(−1) = 0 in both rows. Those inputs form the <b>null space</b>, a magenta line in the input picture — the direction of lost information. Slide the yellow x along it and its output does not move.',
        setup(s) { s.b = null; s.xMark = null; s.solLine = 0; s.showCol = 0.35; s.G = I2(); s.showNull = 0; s.nullPts = 0; s.slide = -1; s.panel = 'null'; },
        at(s, t) {
          s.showNull = seg(t, 0, 0.1); const m = seg(t, 0.1, 0.65);
          s.G = lerpM(I2(), SING, m); s.nullPts = m;
          s.slide = t > 0.72 ? -1.2 + 1.8 * seg(t, 0.72, 1, 'inOut') : -1;
        } },

      { title: 'Rank + nullity = inputs', duration: 3,
        narration: 'One direction survives (rank 1), one is lost (<b>nullity</b> 1), and 1 + 1 = 2, the number of input dimensions. This is <b>rank–nullity</b>: every input direction either makes it through or is crushed. Nothing is in between.',
        setup(s) { s.G = LA.clone(SING); s.slide = -1; s.nullPts = -1; s.showNull = 1; s.showCol = 1; s.panel = 'rn'; s.bars = null; },
        at(s, t) { s.bars = t > 0.1 ? { rank: 1, nullity: 1 } : null; } },

      { title: '3D, rank 2: space onto a plane', duration: 6,
        narration: 'In 3D, M = [[1, 0, 1],[0, 1, 1],[1, 1, 2]] has a third column equal to the first plus the second. Space flattens onto a teal plane — rank 2 — and one input line, along [1; 1; −1], is crushed to the origin: the magenta null space. The bars read 3 = 2 + 1.',
        setup(s) { s.view = '3d'; s.scene = 'grid'; s.cam = cam3({ dist: 11.5, yaw: 0.55, pitch: 0.38 }); s.M3t = LA.clone(R2); s.morph3 = 0; s.showCol3 = 0; s.showNull3 = 0; s.bars = null; s.panel = 'r2'; },
        at(s, t) { s.morph3 = seg(t, 0.1, 0.65); s.showCol3 = seg(t, 0.55, 0.7); s.showNull3 = seg(t, 0, 0.1); s.bars = t > 0.7 ? { rank: 2, nullity: 1 } : null; LA.cam.tween(s, { yaw: 0.55 }, { yaw: 1.33 }, seg(t, 0, 1, 'linear')); } },

      { title: '3D, rank 1: space onto a line', duration: 5,
        narration: 'With all three columns multiples of each other, [[1, 2, 3],[1, 2, 3],[1, 2, 3]], everything lands on one teal line: rank 1. Now a whole plane of inputs is crushed: the null space is two-dimensional, drawn as a magenta plane. The bars read 3 = 1 + 2.',
        setup(s) { s.view = '3d'; s.scene = 'grid'; s.cam = cam3({ dist: 15.5, yaw: 1.0, pitch: 0.32, target: [0.4, 0.4, 0.9] }); s.M3t = LA.clone(R1); s.morph3 = 0; s.showCol3 = 0; s.showNull3 = 0; s.bars = null; s.panel = 'r1'; },
        at(s, t) { s.morph3 = seg(t, 0.1, 0.6); s.showCol3 = seg(t, 0.5, 0.65); s.showNull3 = seg(t, 0, 0.1); s.bars = t > 0.65 ? { rank: 1, nullity: 2 } : null; LA.cam.tween(s, { yaw: 1.0 }, { yaw: 1.4 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Rank is not “how many entries are nonzero”', duration: 4, misconception: true,
        narration: 'Belief: “a matrix full of nonzero numbers has full rank.” The matrix [[1, 2, 3],[2, 4, 6],[3, 6, 9]] has nine bright, nonzero entries and rank 1: its three columns, green, red and blue, all lie on one line through [1; 2; 3]. Rank is about directions, not about how many cells are filled.',
        setup(s) { s.view = '3d'; s.scene = 'mis'; s.cam = cam3({ dist: 27, yaw: 0.35, pitch: 0.3, target: [1.5, 2.5, 3.5] }); s.bars = null; s.claim = null; s.mlT = 0; s.panel = 'mis'; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.6); s.claim = t > 0.62 ? { text: 'belief: “all entries nonzero ⇒ rank 3”', cross: t > 0.7 } : null; s.bars = t > 0.7 ? { rank: 1, nullity: 2 } : null; } },

      { title: 'Rank of a data cloud', duration: 4,
        narration: 'Rank also describes data. Forty synthetic points in 3D that lie exactly in the teal plane form a 40×3 data matrix of rank 2: one feature is a recipe of the others, as in Part 1’s dependent columns. Real data is rarely exactly flat but often nearly so, which is what PCA measures in Part 4.',
        setup(s) { s.view = '3d'; s.scene = 'cloud'; s.cam = cam3({ dist: 11, yaw: 0.5, pitch: 0.12 }); s.claim = null; s.bars = null; s.mlT = 0; s.panel = 'cloud'; },
        at(s, t) { s.mlT = t; LA.cam.tween(s, { yaw: 0.5, pitch: 0.12 }, { yaw: 1.25, pitch: 0.05 }, seg(t, 0, 0.9)); } },

      { title: 'Low rank in ML: bottlenecks, LoRA, singular XᵀX', duration: 6, mlTie: true,
        narration: 'A layer that narrows 784 inputs to 128 has rank at most 128. A LoRA update adds a low-rank matrix to a huge weight — here a heatmap gains one rank-1 sheet. Redundant features make XᵀX singular, so infinitely many weights fit; Part 3 picks one. Rank is computed, never seen, beyond 3D. The picture stops at 3. The formula does not.',
        setup(s) { s.view = '2d'; s.scene = 'ml'; s.cam = cam2(0, 0, 3.6); s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 1.5, recap: true,
        narration: 'Column space = every reachable output = span of the columns; its dimension is the rank. Null space = inputs crushed to zero; its dimension is the nullity. Rank + nullity = number of inputs. In 3D, rank 2 flattens onto a plane and loses a line; rank 1 flattens onto a line and loses a plane.',
        cards: ['<b>Column space</b> = every reachable output = span of the columns (teal). Its dimension is the <b>rank</b>. Ax = b is solvable exactly when b is in it.', '<b>Null space</b> = the inputs crushed to zero (magenta). Its dimension is the <b>nullity</b>: how many input directions are lost.', '<b>rank + nullity = number of inputs.</b> 3D rank 2: 3 = 2 + 1 (plane kept, line lost); rank 1: 3 = 1 + 2.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the column tips (2D or 3D) and the orange <span class="c-b">b</span>. Teal shows the column space, magenta the null space, and the bars count rank + nullity. Presets switch between 2D and 3D.</p>',
      setup(s) { s.scene = 'grid'; s.pg = true; s.chk = null; s.view = '2d'; s.cam = cam2(0.5, 0.8, 3.6); s.M = LA.clone(SING); s.M3 = LA.clone(R2); s.bb = [1, 3]; s.pgCol = true; s.pgNull = true; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A (2D)', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'M3', label: 'M (3D)', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'bb', label: 'b (2D, orange)', colors: 'b', step: 0.1 },
        { type: 'toggle', bind: 'pgCol', label: 'Column space (teal)' },
        { type: 'toggle', bind: 'pgNull', label: 'Null space (magenta)' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Rank 2 (invertible)', apply(s) { s.view = '2d'; s.cam = cam2(0.5, 0.8, 3.6); s.M = LA.clone(INV2); } },
          { label: 'Rank 1', apply(s) { s.view = '2d'; s.cam = cam2(0.5, 0.8, 3.6); s.M = LA.clone(SING); } },
          { label: '3D rank 2', apply(s) { s.view = '3d'; s.cam = cam3({ dist: 11.5, yaw: 0.8, pitch: 0.38 }); s.M3 = LA.clone(R2); } },
          { label: '3D rank 1', apply(s) { s.view = '3d'; s.cam = cam3({ dist: 13, yaw: 1.1, pitch: 0.32 }); s.M3 = LA.clone(R1); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + jhat + ' to make A rank 1, with its column space along the teal line through [1; 2].',
        setup(s) { s.scene = 'grid'; s.view = '2d'; s.pg = false; s.chk = 'rank'; s.cam = cam2(0.5, 0.8, 3.2); s.M = [[1, 1], [2, 0]]; s.panel = 'chk'; },
        judge(s) { const c2 = N.col(s.M, 1), d = Math.abs(N.det(s.M)), ang = angleLine(c2, [1, 2]), n = N.norm(c2); const okI = LA.judge.near(N.col(s.M, 0), [1, 2], LA.TOL.pos).ok; return { ok: okI && d < LA.TOL.det && ang < LA.TOL.angleDeg && n > 0.3, progress: Math.max(0, 1 - ang / 60), hint: n <= 0.3 ? 'Keep ' + jhat + ' longer than zero.' : (!okI ? 'Leave the green ' + ihat + ' at [1; 2].' : undefined) }; },
        hint: 'Put ' + jhat + '’s landing spot on the teal line through [1; 2].',
        solution(s) { s.M = [[1, 0.5], [2, 1]]; } },
      { kind: 'manipulate',
        prompt: 'Drag the orange ' + bS + ' onto the column space so that A' + xS + ' = ' + bS + ' becomes solvable.',
        setup(s) { s.scene = 'grid'; s.view = '2d'; s.pg = false; s.chk = 'b'; s.cam = cam2(0.5, 0.8, 3.4); s.M = LA.clone(SING); s.bb = [1, 3]; s.panel = 'chk'; },
        judge(s) { const d = Math.abs(2 * s.bb[0] - s.bb[1]) / Math.sqrt(5); return { ok: d < LA.TOL.pos, progress: Math.max(0, 1 - d / 2) }; },
        hint: 'b must land on the teal line y = 2x.',
        solution(s) { s.bb = [1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Drag the yellow ' + xS + ' into the null space of A = [[1, 2],[2, 4]]: make its output A' + xS + ' (gray) shrink to the origin.',
        setup(s) { s.scene = 'grid'; s.view = '2d'; s.pg = false; s.chk = 'x'; s.cam = cam2(0.8, 1.8, 4.6); s.M = LA.clone(SING); s.xIn = [1, 1]; s.panel = 'chk'; },
        judge(s) { const ax = N.norm(N.matVec(s.M, s.xIn)), n = N.norm(s.xIn); return { ok: ax < LA.TOL.pos && n > 0.5, progress: Math.max(0, 1 - ax / 6), hint: n <= 0.5 ? 'The zero vector does not count: keep x longer than 0.5.' : undefined }; },
        hint: 'Find the direction whose output shrinks to the origin: along [2; −1].',
        solution(s) { s.xIn = [2, -1]; } },
    ],
  });

  /* ---------- 3D scenes ---------- */
  function render3d(g, s) {
    if (s.scene === 'mis') { renderMis(g, s); return; }
    if (s.scene === 'cloud') { renderCloud(g, s); return; }
    const M = lerpM(I3(), s.M3t, s.morph3);
    const rank = s.M3t === null ? 3 : rankOf(s.M3t);
    g.cage({ range: 3, walls: false, alpha: 0.5 });
    g.lattice(M, { range: 2, alpha: 0.4 * (rank === 1 ? 1 - 0.95 * s.showCol3 : 1) });
    drawSubspaces3(g, s.M3t, rank, s.showCol3, s.showNull3, M);
    const c = j => N.col(M, j);
    g.vector([0, 0, 0], c(0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector([0, 0, 0], c(1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.vector([0, 0, 0], c(2), { color: 'k', label: 'k', hat: true, width: 3 });
    if (s.bars) rankBars(g, s.bars.rank, s.bars.nullity, 26, g.size.h - 175, 120);
  }
  // teal column space (plane or line) and magenta null space (line or plane, with points carried by Mcur)
  function drawSubspaces3(g, Mt, rank, colA, nullA, Mcur) {
    const cs = N.colSpace(Mt, 1e-6), ns = N.nullSpace(Mt, 1e-6);
    const planes = [];
    if (colA > 0 && rank < 3) g.hud('teal: column space (' + (rank === 2 ? 'a plane' : rank === 1 ? 'a line' : 'the origin') + ', rank ' + rank + ')', 14, 22, { size: 13.5, color: 'teal', alpha: colA });
    if (nullA > 0 && ns.length) g.hud('magenta: null space (' + (ns.length === 1 ? 'a line' : ns.length === 2 ? 'a plane' : 'everything') + ', nullity ' + ns.length + ')', 14, 42, { size: 13.5, color: 'mag', alpha: nullA });
    if (colA > 0 && rank === 2) planes.push({ point: [0, 0, 0], span: [cs[0], cs[1]], extent: 2.4, color: 'teal', alpha: 0.22 * colA });
    if (nullA > 0 && ns.length === 2) planes.push({ point: [0, 0, 0], span: [ns[0], ns[1]], extent: 2.4, color: 'mag', alpha: 0.2 * nullA });
    if (planes.length) g.planes(planes);
    if (colA > 0 && rank === 1) { g.lineThrough([0, 0, 0], cs[0], { color: 'teal', width: 4, len: 10, alpha: colA }); }
    if (nullA > 0 && ns.length === 1) {
      const d = ns[0];
      g.lineThrough([0, 0, 0], d, { color: 'mag', width: 2.6, len: 3.5, alpha: nullA });
      if (Mcur) [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2].forEach(k => g.point(N.matVec(Mcur, N.scale(d, k * 1.2)), { r: 4.5, color: 'mag' }));
    }
    if (nullA > 0 && ns.length === 2 && Mcur) {
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { if (!a && !b) continue; g.point(N.matVec(Mcur, N.add(N.scale(ns[0], 1.5 * a), N.scale(ns[1], 1.5 * b))), { r: 4.5, color: 'mag' }); }
    }
  }
  function renderMis(g, s) {
    g.cage({ range: 4, walls: false, alpha: 0.4 });
    g.axes({ length: 4.5, labels: true });
    const u = seg(s.mlT, 0, 1);
    g.lineThrough([0, 0, 0], [1, 2, 3], { color: 'teal', width: 1.6, alpha: 0.5 * u, len: 13 });
    const cols = [0, 1, 2].map(j => N.col(FULL, j));
    const names = ['c₁', 'c₂', 'c₃'], roles = ['i', 'j', 'k'];
    [2, 1, 0].forEach(j => { if (u > j / 3) g.vector([0, 0, 0], N.scale(cols[j], seg(u, j / 3, j / 3 + 0.33)), { color: roles[j], label: names[j], width: 3.4 - j * 0.6 }); });
    // the 3×3 heatmap of entries (all bright)
    g.viewport({ x: 0.03, y: 0.1, w: 0.3, h: 0.28 }, { view: '2d', cam: cam2(0, 0, 1.25), bg: 'bg' }, (g2, r) => {
      g2.image(FULL, { x0: -0.9, x1: 0.9, y0: -0.9, y1: 0.9 }, { ramp: LA.color.heat, range: [0, 9], cells: true });
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) g2.label(String(FULL[i][j]), [-0.6 + 0.6 * j, 0.6 - 0.6 * i], { size: 13, color: FULL[i][j] > 4 ? 'bg' : 'text', bg: false });
      g2.hud('9 of 9 entries nonzero', r.x + 6, r.y + r.h - 8, { size: 11.5, color: 'hud' });
    });
    if (s.claim) {
      const w = g.size.w;
      g.label(s.claim.text, [w / 2, 26], { screen: true, size: 15, color: 'hud', font: 'ui' });
      if (s.claim.cross) g.marker.cross([w / 2 + s.claim.text.length * 3.7 + 22, 26], { screen: true, size: 22 });
    }
    if (u >= 1) g.label('all three columns on one line', [3, 6, 9], { color: 'teal', size: 13, font: 'ui', anchor: 'right', dx: -16, dy: 6 });
    if (s.bars) rankBars(g, s.bars.rank, s.bars.nullity, 26, g.size.h - 175, 120);
  }
  function renderCloud(g, s) {
    g.cage({ range: 3, walls: false, alpha: 0.4 });
    g.axes({ length: 3.2, labels: true });
    const a = seg(s.mlT, 0.1, 0.35);
    g.planes([{ point: [0, 0, 0], span: [[1, 0, 0.5], [0, 1, 0.5]], extent: 2.8, color: 'teal', alpha: 0.22 * a }]);
    g.points(s.cloud, { r: 3.6, color: 'data' });
    g.hud('40 synthetic points, 3 features', 14, 22, { size: 14 });
    if (s.mlT > 0.4) g.hud('teal plane: feature 3 = 0.5·feature 1 + 0.5·feature 2 → rank 2', 14, 42, { size: 13.5, color: 'teal' });
    if (s.mlT > 0.55) g.bars(s.sv, { screen: true, x: g.size.w - 190, y: g.size.h - 150, w: 150, h: 90, colors: ['gold', 'gold', 'gold'], labels: ['σ₁', 'σ₂', 'σ₃ = 0'], title: 'singular values (Part 4 teaser)' });
  }

  /* ---------- checks ---------- */
  function renderCheck(g, s) {
    const M = s.M;
    g.ghostGrid();
    g.grid(M, { alpha: 0.6 });
    if (s.chk === 'rank') {
      g.lineThrough([0, 0], [1, 2], { color: 'teal', width: 2, alpha: 0.45 });
      g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      g.hud('det A = ' + LA.fmt(N.det(M)) + ' · rank ' + rankOf(M), 14, 22, { size: 14, color: Math.abs(N.det(M)) < LA.TOL.det ? 'teal' : 'hud' });
      g.handle('j', N.col(M, 1), (p, st) => { const q = snapLine(p, [1, 2], 0.1); st.M[0][1] = q[0]; st.M[1][1] = q[1]; }, { color: 'j' });
      return;
    }
    g.lineThrough([0, 0], [1, 2], { color: 'teal', width: 2.6, alpha: 0.9 });
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 2.6, alpha: 0.7 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 2.6, alpha: 0.7 });
    if (s.chk === 'b') {
      g.vector(s.bb, { color: 'b', label: 'b', width: 3 });
      g.handle('b', s.bb, (p, st) => { st.bb = p.slice(); }, { color: 'b' });
      g.hud('teal: the column space of A', 14, 22, { size: 14, color: 'teal' });
      return;
    }
    const ax = N.matVec(M, s.xIn);
    g.vector(ax, { color: 'muted', label: 'Ax', width: 2.4, dashed: true });
    g.vector(s.xIn, { color: 'v', label: 'x', width: 3 });
    g.handle('x', s.xIn, (p, st) => { st.xIn = snapLine(p, NULL2, 0.12); }, { color: 'v' });
    g.hud('‖Ax‖ = ' + LA.fmt(N.norm(ax)), 14, 22, { size: 14, color: N.norm(ax) < LA.TOL.pos ? 'mag' : 'hud' });
  }

  /* ---------- playground ---------- */
  function renderPlay(g, s) {
    if (s.view === '3d') {
      const M = s.M3, r = rankOf(M);
      g.cage({ range: 3, walls: false, alpha: 0.5 });
      g.lattice(M, { range: 2, alpha: 0.35 });
      drawSubspaces3(g, M, r, s.pgCol ? 1 : 0, s.pgNull ? 1 : 0, M);
      const c = j => N.col(M, j);
      g.vector([0, 0, 0], c(0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector([0, 0, 0], c(1), { color: 'j', label: 'j', hat: true, width: 3 });
      g.vector([0, 0, 0], c(2), { color: 'k', label: 'k', hat: true, width: 3 });
      if (r === 3 && s.pgCol) g.hud('rank 3: the column space is all of space', 14, 22, { size: 13.5, color: 'teal' });
      rankBars(g, r, 3 - r, 26, g.size.h - 175, 120);
      ['i3', 'j3', 'k3'].forEach((id, j) => g.handle(id, c(j), (p, st) => { for (let k = 0; k < 3; k++) st.M3[k][j] = p[k]; }, { color: ['i', 'j', 'k'][j] }));
      return;
    }
    const M = s.M, r = rankOf(M), b = s.bb;
    g.ghostGrid(); g.grid(M, { alpha: 0.7 });
    const cs = N.colSpace(M, TOLR), ns = N.nullSpace(M, TOLR);
    if (s.pgCol) {
      if (r === 2) { const bd = g.P.bounds(); g.polygon([[bd.x0, bd.y0], [bd.x1, bd.y0], [bd.x1, bd.y1], [bd.x0, bd.y1]], { fill: 'teal', alpha: 0.08 }); g.hud('column space = the whole plane (rank 2)', 14, 22, { size: 13.5, color: 'teal' }); }
      else if (r === 1) g.lineThrough([0, 0], cs[0], { color: 'teal', width: 3 });
    }
    if (s.pgNull && r < 2) { if (r === 1) g.lineThrough([0, 0], ns[0], { color: 'mag', width: 2.4 }); }
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.vector(b, { color: 'b', label: 'b', width: 3 });
    if (r === 2) g.vector(N.solve(M, b), { color: 'v', label: 'x', width: 3 });
    else if (r === 1) {
      const x0 = N.matVec(N.pinv(M), b);
      if (N.dist(N.matVec(M, x0), b) < LA.TOL.pos) { g.lineThrough(x0, ns[0], { color: 'mag', width: 2, dashed: true }); g.hud('b reachable: a whole line of solutions (dashed)', 14, 22, { size: 13.5, color: 'mag' }); }
      else g.hud('b is off the teal line: no solution', 14, 22, { size: 13.5, color: 'bad' });
    }
    rankBars(g, r, 2 - r, 26, g.size.h - 175, 120);
    g.handle('i', N.col(M, 0), (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; }, { color: 'i' });
    g.handle('j', N.col(M, 1), (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
    g.handle('b', b, (p, st) => { st.bb = p.slice(); }, { color: 'b' });
  }

  /* ---------- ML tie ---------- */
  function renderML(g, s) {
    const t = s.mlT;
    const tile = (x0, x1, y0, y1, label, col) => { g.polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], { fill: col, alpha: 0.25, stroke: col, width: 1.4 }); g.label(label, [(x0 + x1) / 2, (y0 + y1) / 2], { color: 'text', size: 14, font: 'ui' }); };
    // 1. bottleneck shapes
    g.label('a classifier narrows 784 → 128 → 10', [0, 3.35], { color: 'hud', size: 14, font: 'ui' });
    tile(-2.9, -1.9, 2.0, 3.0, '784', 'data'); tile(-0.5, 0.5, 2.25, 2.75, '128', 'teal'); tile(1.9, 2.9, 2.35, 2.65, '10', 'gold');
    g.vector([-1.8, 2.5], [-0.6, 2.5], { color: 'w', label: 'W₁', width: 2.2, head: 9, dy: -14 });
    g.vector([0.6, 2.5], [1.8, 2.5], { color: 'w', label: 'W₂', width: 2.2, head: 9, dy: -14 });
    g.label('rank(W₁) ≤ 128', [-1.2, 1.85], { color: 'teal', size: 13, font: 'ui' });
    g.label('rank(W₂) ≤ 10', [1.2, 1.85], { color: 'teal', size: 13, font: 'ui' });
    // 2. LoRA: W + u vᵀ
    const a = seg(t, 0.25, 0.6);
    const uv = s.u.map(ui => s.v.map(vj => ui * vj));
    const Wp = s.W.map((row, i) => row.map((w, j) => w + a * uv[i][j]));
    const ramp = x => LA.color.diverge(2 * x - 1), rg = [-2.5, 2.5];
    const y0 = -0.7, y1 = 0.9;
    g.label('LoRA: add a rank-1 sheet to a weight (12×12 synthetic sample)', [0, 1.3], { color: 'hud', size: 13.5, font: 'ui' });
    g.image(s.W, { x0: -3.0, x1: -1.4, y0, y1 }, { ramp, range: rg });
    g.label('+', [-1.15, 0.1], { size: 22, color: 'hud', bg: false });
    g.image(uv, { x0: -0.9, x1: 0.7, y0, y1 }, { ramp, range: rg, alpha: 0.15 + 0.85 * a });
    g.label('=', [0.95, 0.1], { size: 22, color: 'hud', bg: false });
    g.image(Wp, { x0: 1.2, x1: 2.8, y0, y1 }, { ramp, range: rg });
    g.label('W', [-2.2, y0], { anchor: 'below', dy: 6, color: 'muted', size: 12.5, font: 'ui' });
    g.label('u vᵀ: column × row (rank 1)', [-0.1, y0], { anchor: 'below', dy: 6, color: 'gold', size: 12.5, font: 'ui' });
    g.label('W + u vᵀ', [2.0, y0], { anchor: 'below', dy: 6, color: 'muted', size: 12.5, font: 'ui' });
    // 3. singular XᵀX from the flat cloud
    const b3 = seg(t, 0.6, 0.8);
    if (b3 > 0) {
      const X = s.cloud; const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const p of X) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) G[i][j] += p[i] * p[j] / X.length;
      g.image(G, { x0: -3.0, x1: -1.8, y0: -3.0, y1: -1.8 }, { ramp: LA.color.heat, range: [0, 1.6], cells: true, alpha: b3 });
      g.label('XᵀX of the flat cloud (3×3)', [-1.6, -1.95], { anchor: 'left', color: 'hud', size: 13, font: 'ui', alpha: b3 });
      g.label('det(XᵀX) = ' + LA.fmt(N.det(G)) + ': singular', [-1.6, -2.4], { anchor: 'left', color: 'bad', size: 13.5, font: 'ui', alpha: b3 });
      g.label('→ infinitely many weights fit equally well', [-1.6, -2.85], { anchor: 'left', color: 'hud', size: 13, font: 'ui', alpha: b3 });
    }
    g.hud('The picture stops at 3. The formula does not.', 14, g.size.h - 36, { size: 13.5, color: 'gold' });
    g.hud('Part 4: low rank · Part 3: least squares', 14, g.size.h - 16, { size: 12.5, color: 'muted' });
  }

  /* ---------- recap ---------- */
  function renderRecap(g, s) {
    const card = (i, vopts, fn) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, vopts, fn);
    card(0, { view: '2d', cam: cam2(1.0, 1.9, 2.75) }, (g2, r) => {
      g2.ghostGrid({ range: 6 }); g2.grid(SING, { range: 6, alpha: 0.5 });
      g2.lineThrough([0, 0], [1, 2], { color: 'teal', width: 3 });
      g2.vector([1, 2], { color: 'i', label: 'i', hat: true, width: 2.6 }); g2.vector([2, 4], { color: 'j', label: 'j', hat: true, width: 2.6 });
      g2.hud('column space: every output (teal) · rank 1', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(1, { view: '2d', cam: cam2(0.6, 0, 2.0) }, (g2, r) => {
      g2.ghostGrid({ range: 6 });
      g2.lineThrough([0, 0], NULL2, { color: 'mag', width: 2.6 });
      [-1.5, -1, -0.5, 0.5, 1, 1.5].forEach(k => { const p = N.scale(NULL2, k); g2.point(p, { r: 4, color: 'mag' }); g2.vector(p, N.scale(p, 0.25), { color: 'mag', width: 1.2, head: 7, alpha: 0.6 }); });
      g2.point([0, 0], { r: 5, color: 'white' });
      g2.hud('null space: inputs crushed to 0 (magenta) · nullity 1', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(2, { view: '3d', cam: cam3({ dist: 12, yaw: 1.2, pitch: 0.35 }) }, (g2, r) => {
      g2.lattice(R2, { range: 1, alpha: 0.5 });
      g2.planes([{ point: [0, 0, 0], span: [[1, 0, 1], [0, 1, 1]], extent: 2.6, color: 'teal', alpha: 0.2 }]);
      g2.lineThrough([0, 0, 0], NULL3, { color: 'mag', width: 2.4, len: 3 });
      rankBars(g2, 2, 1, r.x + 24, r.y + 46, r.h - 70);
      g2.hud('3D rank 2: plane kept, line lost', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
  }
})();
