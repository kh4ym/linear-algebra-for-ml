/* p2-11 — Affine maps: Wx + b and the homogeneous trick. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const W0 = [[1, 0.5], [0, 1]], B0 = [2, -1];
  const bOr = '<span class="c-b">b</span>', xV = '<span class="c-v">x</span>';
  const DEG = Math.PI / 180;
  /** the augmented 3×3 [[W, b],[0, 1]] */
  const aug = (W, b) => [[W[0][0], W[0][1], b[0]], [W[1][0], W[1][1], b[1]], [0, 0, 1]];
  const aff = (W, b, p) => N.add(N.matVec(W, p), b);
  const f = x => LA.fmt(x);
  const pt2 = p => '(' + f(p[0]) + ', ' + f(p[1]) + ')';
  const pt3 = p => '(' + f(p[0]) + ', ' + f(p[1]) + ', ' + f(p[2]) + ')';
  const TARGET = [[1, 1], [2, 1], [3, 2], [2, 2]];          // check 3: the gray parallelogram
  const SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]];
  // chain of affine maps (step 8): rotate 30°, shift [1, 0], scale 1.3
  const chainAt = (a, sh, sc) => { const c = Math.cos(a), s = Math.sin(a); const R = [[c, -s, 0], [s, c, 0], [0, 0, 1]], T = [[1, 0, sh], [0, 1, 0], [0, 0, 1]], S = [[sc, 0, 0], [0, sc, 0], [0, 0, 1]]; return N.matMul(S, N.matMul(T, R)); };
  const CHAIN = chainAt(30 * DEG, 1, 1.3);

  /** a small synthetic design matrix: rows [1, x1, x2] */
  function design() {
    const r = LA.rng(9), X = [];
    for (let i = 0; i < 6; i++) X.push([1, Math.round(LA.rng.gauss(r) * 8) / 10, Math.round(LA.rng.gauss(r) * 8) / 10]);
    return X;
  }
  const WREG = [0.5, 1.2, -0.8];

  /** 2D camera that shows the world box [x0,x1]×[y0,y1] inside a viewport of fractional width fw / height fh. */
  function fit(g, fw, fh, x0, x1, y0, y1) {
    const W = Math.max(50, g.size.w * fw), H = Math.max(50, g.size.h * fh), m = 1.06;
    return { center: [(x0 + x1) / 2, (y0 + y1) / 2], halfH: m * Math.max((y1 - y0) / 2, (x1 - x0) / 2 * H / W) };
  }
  function cam3d() { return Object.assign(LA.cam.defaults3d(), { target: [1.3, -0.6, 0.5], dist: 15, yaw: 0.55, pitch: 0.45 }); }

  /* ---------- drawing ---------- */
  /** the grid under x ↦ Mx + t, drawn line by line (g.grid has no translation) */
  function affGrid(g, M, t, o) {
    o = o || {}; const R = o.range || 8, al = o.alpha === undefined ? 1 : o.alpha;
    for (let i = -R; i <= R; i++) {
      const ax = i === 0, st = { color: ax ? 'gridAxis' : 'grid', width: ax ? 1.6 : 1, alpha: al };
      g.line(aff(M, t, [i, -R]), aff(M, t, [i, R]), st);
      g.line(aff(M, t, [-R, i]), aff(M, t, [R, i]), st);
    }
  }
  function cells(g, M, o) {
    const rows = M.length, cols = M[0].length, cs = o.cs || 0.6, h = cs / 2 - 0.03;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = M[i][j], c = [o.x0 + (j + 0.5) * cs, o.y0 - (i + 0.5) * cs];
      const fill = o.fill ? o.fill(i, j, v) : 'panel2';
      g.polygon([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]], { fill, alpha: 0.92, stroke: o.strokeAt ? o.strokeAt(i, j) : 'line', width: 1.2 });
      g.label(typeof v === 'number' ? (v < 0 ? '−' : '') + Math.abs(v) : String(v), c, { color: o.textAt ? o.textAt(i, j, v) : 'text', size: o.size || 13, font: 'math', bg: false });
    }
    const p = 0.07, k = 0.12, st = { color: o.brColor || 'hud', width: 1.5 }, x0 = o.x0, y0 = o.y0, H = rows * cs, Wd = cols * cs;
    g.polyline([[x0 - p + k, y0 + p], [x0 - p, y0 + p], [x0 - p, y0 - H - p], [x0 - p + k, y0 - H - p]], st);
    g.polyline([[x0 + Wd + p - k, y0 + p], [x0 + Wd + p, y0 + p], [x0 + Wd + p, y0 - H - p], [x0 + Wd + p - k, y0 - H - p]], st);
  }
  function designScene(g, X, x0, y0, cs) {
    const heat = v => C.heat(0.5 + v * 0.35);
    ['1', 'x₁', 'x₂'].forEach((h, j) => g.label(h, [x0 + (j + 0.5) * cs, y0 + 0.22], { color: j === 0 ? 'gold' : 'muted', size: 13, font: 'math', bg: false }));
    cells(g, X, { x0, y0, cs, fill: (i, j, v) => (j === 0 ? 'gold' : heat(v)), textAt: (i, j, v) => (j === 0 ? 'bg' : (0.5 + v * 0.35 > 0.62 ? 'bg' : 'text')), strokeAt: (i, j) => (j === 0 ? 'gold' : 'line') });
    g.label('ones column', [x0 + cs / 2, y0 - X.length * cs - 0.25], { color: 'gold', size: 12, font: 'ui' });
    const yw = y0 - X.length * cs - 0.75;
    g.label('weights', [x0 - 0.15, yw - cs / 2], { color: 'muted', size: 12, font: 'ui', anchor: 'right' });
    cells(g, [WREG], { x0, y0: yw, cs, fill: () => 'panel2', textAt: () => 'w' });
    ['w₀', 'w₁', 'w₂'].forEach((h, j) => g.label(h, [x0 + (j + 0.5) * cs, yw - cs - 0.2], { color: 'w', size: 12, font: 'math', bg: false }));
  }

  function scene2d(g, s, ctx) {
    const Mt = lerpM(I2(), s.W, s.morph), bt = N.scale(s.b, s.slide), F = p => aff(Mt, bt, p);
    g.ghostGrid();
    affGrid(g, Mt, bt, { alpha: s.gridAlpha });
    if (s.targetPoly) { g.polygon(TARGET, { fill: 'muted', alpha: 0.22, stroke: 'muted', width: 1.4 }); g.label('target', [2.6, 2.25], { color: 'muted', size: 12, font: 'ui' }); }
    if (s.target) { g.point(s.target, { r: 6, color: 'muted' }); g.marker.ring(s.target, { r: 13, color: 'muted', width: 1.5 }); g.label('target ' + pt2(s.target), s.target, { color: 'muted', size: 12, font: 'ui', anchor: 'below', dy: 14 }); }
    if (s.showSquare) {
      const q = SQUARE.map(F);
      g.polygon(q, { fill: 'v', alpha: 0.2, stroke: 'v', width: 1.6 });
      if (s.corners > 0) {
        const names = ['b = ', '', '', ''], off = [[-10, 14], [14, 34], [10, -12], [-14, -12]], an = ['right', 'left', 'left', 'right'];
        q.forEach((p, k) => g.label(names[k] + pt2(p), p, { color: k === 0 ? 'b' : 'v', size: 12, font: 'ui', anchor: an[k], dx: off[k][0], dy: off[k][1], alpha: s.corners }));
      }
    }
    const ci = N.col(Mt, 0), cj = N.col(Mt, 1);
    g.vector(bt, N.add(bt, ci), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(bt, N.add(bt, cj), { color: 'j', label: 'j', hat: true, width: 3 });
    g.point(bt, { r: 4.5, color: 'white' });
    if (N.norm(bt) > 0.02) g.vector([0, 0], bt, { color: 'b', label: 'b', width: 3, labelAt: 0.55 });
    if (s.mis > 0) {
      g.dim([0, 0], bt, { label: 'f(0) = b ≠ 0', offset: 30, color: 'bad' });
      g.label('linear?', [0.2, -1.9], { color: 'bad', size: 15, font: 'ui' });
      g.marker.cross([0.2, -1.9], { dx: 46, size: 22 });
    }
    if (s.mis2 > 0) {
      const v = [1, 0.5], fv = aff(s.W, s.b, v), f2v = aff(s.W, s.b, N.scale(v, 2)), twofv = N.scale(fv, 2);
      g.vector([0, 0], twofv, { color: 'muted', label: '2·f(v)', width: 2, dashed: true, alpha: s.mis2 });
      g.vector([0, 0], fv, { color: 'v', label: 'f(v)', width: 3, alpha: s.mis2 });
      g.vector([0, 0], f2v, { color: 'v', label: 'f(2v)', width: 3, alpha: s.mis2 });
      if (s.mis2 >= 1) { g.line(f2v, twofv, { color: 'bad', dashed: true, width: 1.4 }); g.label('f(2v) ≠ 2·f(v)', lerpV(f2v, twofv, 0.5), { color: 'bad', size: 13, font: 'ui', anchor: 'above', dy: -8 }); g.marker.cross(lerpV(f2v, twofv, 0.5), { dy: 12, size: 20 }); }
    }
    if (s.probe > 0) {
      const x = [1, 1], Wx = N.matVec(s.W, x), fx = N.add(Wx, s.b), u1 = seg(s.probe, 0, 0.45), u2 = seg(s.probe, 0.55, 1);
      g.vector([0, 0], x, { color: 'v', label: 'x', width: 2, dashed: true, alpha: 0.55 });
      g.vector([0, 0], lerpV(x, Wx, u1), { color: 'v', label: u1 > 0.5 ? 'Wx' : 'x', width: u2 > 0 ? 2 : 3, alpha: u2 > 0 ? 0.6 : 1, dashed: u2 > 0 });
      if (u2 > 0) {
        g.vector(Wx, lerpV(Wx, fx, u2), { color: 'b', label: 'b', width: 3, components: u2 >= 1 ? 'labels' : false });
        if (u2 >= 1) g.vector([0, 0], fx, { color: 'v', label: 'f(x) = Wx + b', width: 3, labelAt: 0.45, dy: -14 });
      }
    }
    if (s.formula) g.hud('f(x) = W x + b', 14, 24, { size: 20, font: 'math', color: 'text' });
    if (s.lin) {
      const h = ctx.size.h, moved = N.norm(s.b) > 0.3;
      g.hud('f(0) = ' + pt2(s.b), 12, h - 34, { color: moved ? 'b' : 'hud', size: 14 });
      g.hud(moved ? 'the origin moved: not linear' : 'f(0) = 0 so far', 12, h - 14, { color: moved ? 'bad' : 'muted', size: 13 });
      if (moved) g.marker.cross([238, h - 14], { screen: true, size: 18 });
    }
    // handles: the moved origin (b) and the two landing spots measured from it
    // in checks, positions within 0.1 of a half-integer snap to it so the targets are reachable by hand
    const sn = (p, st) => (st.snap ? p.map(x => { const r = Math.round(x * 2) / 2; return Math.abs(x - r) < 0.1 ? r : x; }) : [p[0], p[1]]);
    g.handle('b', bt, (p, st) => { st.b = sn(p, st); st.slide = 1; }, { color: 'b' });
    if (!s.lockW) {
      g.handle('i', N.add(bt, ci), (p, st) => { const o = N.scale(st.b, st.slide), q = sn(p, st); st.W[0][0] = q[0] - o[0]; st.W[1][0] = q[1] - o[1]; st.morph = 1; }, { color: 'i' });
      g.handle('j', N.add(bt, cj), (p, st) => { const o = N.scale(st.b, st.slide), q = sn(p, st); st.W[0][1] = q[0] - o[0]; st.W[1][1] = q[1] - o[1]; st.morph = 1; }, { color: 'j' });
    }
  }

  /** 3D homogeneous picture: the slice z = lift under the 3×3 map M3, over the floor z = 0. */
  function scene3d(g, s, o) {
    o = o || {};
    const M = o.M3 || s.M3, lift = o.lift === undefined ? s.lift : o.lift, R = 2;
    const T = p => N.matVec(M, p);
    g.grid(M, { z: 0, range: R, color: 'ghost', axisColor: 'ghostAxis' });          // the floor (only W acts on it)
    g.axes({ length: 2.6 });
    const corners = [[-R, -R, lift], [R, -R, lift], [R, R, lift], [-R, R, lift]].map(T);
    if (lift > 0.02) g.polygon(corners, { fill: 'teal', alpha: 0.13, subdiv: 2 });
    g.grid(M, { z: lift, range: R, alpha: 0.95 });
    g.point([0, 0, 0], { r: 5, color: 'white', label: o.mini ? null : (s.pulse > 0 ? '3D origin: fixed' : '3D origin'), labelColor: 'hud' });
    if (s.pulse > 0 && !o.mini) g.marker.ring([0, 0, 0], { r: 12 + 10 * s.pulse, color: 'ok' });
    const top = T([0, 0, lift]);
    const moved = Math.hypot(top[0], top[1]) > 0.02;
    if (moved) {
      g.vector([0, 0, 0], top, { color: 'k', label: 'k', hat: true, width: 3 });
      g.vector([0, 0, lift], top, { color: 'b', label: 'b', width: 2.6, labelAt: 0.6 });
    } else if (lift > 0.02) g.line([0, 0, 0], top, { color: 'residual', dashed: true, width: 1.4 });
    if (lift > 0.02) { g.point(top, { r: 4, color: 'teal' }); if (!o.mini && !moved) g.label('slice origin ' + pt3(top), top, { color: 'teal', size: 13, anchor: 'left', dx: 10, dy: -26 }); }
    if (!o.mini && s.samples) {
      [[1, 0.5], [-1, 1.5], [0.5, -1.5]].forEach((p, k) => {
        const q = T([p[0], p[1], lift]);
        g.point(q, { r: 4.5, color: 'v', label: s.samples > 0.5 ? pt3(q) : pt2(p), labelColor: 'v' });
      });
    }
    if (s.probe3 > 0 && !o.mini) {
      const p = [1, 0.5, 1], q = T(p), cur = lerpV(p, q, s.probe3);
      g.point(p, { r: 4, color: 'v', alpha: 0.5 });
      g.line(p, cur, { color: 'v', dashed: true, width: 1.4 });
      g.point(cur, { r: 6, color: 'v', label: s.probe3 >= 1 ? 'W̃x̃ = ' + pt3(q) : 'x̃ = (1, 0.5, 1)', labelColor: 'v' });
    }
  }

  function chainTiles(g, s) {
    const names = ['R̃ · rotate 30°', 'T̃ · shift [1; 0]', 'S̃ · scale 1.3'];
    if (s.chainPhase < 3) {
      names.forEach((nm, k) => g.hud(nm, 14, 24 + 24 * k, { size: 14, color: k === s.chainPhase ? 'gold' : (k < s.chainPhase ? 'hud' : 'muted') }));
      g.hud('one map at a time', 14, 24 + 24 * 3, { size: 12, color: 'muted' });
    } else {
      g.hud('S̃ T̃ R̃ = one augmented matrix', 14, 24, { size: 15, color: 'gold' });
      g.hud('the same end state, in one step', 14, 48, { size: 12, color: 'muted' });
    }
  }

  function sceneML(g, s) {
    const X = s.X;
    const left = (g, r) => {
      designScene(g, X, -0.9, 2.2, 0.6);
      g.hud('regression (synthetic rows): append a 1 to each', r.x + 10, r.y + 18, { color: 'hud' });
      g.label('ŷ = w₀·1 + w₁x₁ + w₂x₂', [0, -3.2], { color: 'text', size: 13, font: 'math' });
    };
    const right = (g, r) => {
      g.ghostGrid(); affGrid(g, W0, B0, { range: 6 });
      g.vector([0, 0], B0, { color: 'b', label: 'b', width: 3 });
      g.vector(B0, N.add(B0, N.col(W0, 0)), { color: 'i', label: 'i', hat: true, width: 2.6 });
      g.vector(B0, N.add(B0, N.col(W0, 1)), { color: 'j', label: 'j', hat: true, width: 2.6 });
      // a small fold icon: the tanh curve in a box
      const o = [2.4, 1.9], w = 0.9, h = 0.55, pts = [];
      g.polygon([[o[0] - w, o[1] - h], [o[0] + w, o[1] - h], [o[0] + w, o[1] + h], [o[0] - w, o[1] + h]], { fill: 'panel2', alpha: 0.92, stroke: 'line' });
      for (let k = 0; k <= 30; k++) { const u = -1 + 2 * k / 30; pts.push([o[0] + u * w * 0.9, o[1] + Math.tanh(3 * u) * h * 0.8]); }
      g.polyline(pts, { color: 'gold', width: 2 });
      g.label('fold', [o[0], o[1] - h - 0.25], { color: 'gold', size: 12, font: 'ui' });
      g.hud('layer: Wx + b, then fold', r.x + 10, r.y + 18, { color: 'hud' });
      g.hud('p2-12: neural layer · Part 3: regression', r.x + 10, r.y + r.h - 14, { color: 'gold', size: 12 });
    };
    g.split(left, right, { ratio: 0.5, left: { view: '2d', cam: fit(g, 0.5, 1, -2.0, 1.4, -3.5, 2.6) }, right: { view: '2d', cam: fit(g, 0.5, 1, -1.0, 3.6, -2.4, 2.8) } });
  }

  function sceneRecap(g, s) {
    const h = 1 / 3;
    g.viewport({ x: 0, y: 0, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -1.6, 4.2, -2.0, 1.6) }, (g, r) => {
      g.ghostGrid(); affGrid(g, W0, B0, { range: 6 });
      g.vector([0, 0], B0, { color: 'b', label: 'b', width: 3 });
      g.label('linear?', [-0.6, 0.9], { color: 'bad', size: 14, font: 'ui' }); g.marker.cross([-0.6, 0.9], { dx: 40, size: 20 });
      g.hud('1 · a slide moves the origin: not linear', r.x + 8, r.y + 16, { size: 12 });
    });
    g.viewport({ x: 0, y: h, w: 1, h }, { view: '3d', cam: Object.assign(cam3d(), { dist: 12, pitch: 0.38 }) }, (g, r) => {
      scene3d(g, s, { M3: aug(W0, B0), lift: 1, mini: true });
      g.hud('2 · [[W, b],[0, 1]] slides the slice z = 1', r.x + 8, r.y + 16, { size: 12 });
    });
    g.viewport({ x: 0, y: 2 * h, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -2.4, 1.6, -1.35, 2.5) }, (g, r) => {
      designScene(g, s.X.slice(0, 3), -0.9, 2.0, 0.55);
      g.hud('3 · the ones column: an intercept as a weight', r.x + 8, r.y + 16, { size: 12 });
    });
  }

  function matW3(tex, M) {
    return tex.mat(M, { colors: (i, j) => (j === 2 ? (i < 2 ? 'k' : 'muted') : (i < 2 ? ['i', 'j'][j] : 'muted')) });
  }

  LA.registerLesson({
    id: 'p2-11-affine-bias',
    part: 2,
    title: 'Affine maps: Wx + b and the homogeneous trick',
    subtitle: 'A slide is not linear, until you add one more dimension.',
    minutes: 9,
    prereqs: ['p2-01-linear-transformations', 'p2-05-3d-transformations'],
    tags: ['core'],
    badge: { drawnIn: '2D, and 3D for the homogeneous view', realSize: 'every layer is Wx + b with b of size 128–4096; the ones column of a regression design matrix is this trick', transfers: '“linear + shift” and “append a 1 to make the shift linear one dimension up” are exact in any dimension' },
    cantShow: ['the homogeneous trick in ℝ⁷⁸⁵ (we show the z = 1 slice of ℝ³; the rule is the same)'],
    terms: [
      { term: 'translation', def: 'Sliding every point by the same vector; it moves the origin, so it is not linear.' },
      { term: 'affine map', def: 'A linear map followed by a translation: f(x) = Wx + b.' },
      { term: 'bias', def: 'The translation vector b of an affine map: the shift that lets the origin move.' },
      { term: 'homogeneous coordinates', def: 'Writing a 2D point as [x; y; 1] so that an affine map becomes a 3×3 linear map.' },
      { term: 'augmented matrix', def: 'The 3×3 matrix [[W, b],[0, 1]] that holds both W and b.' },
    ],
    atlas: ['mlp', 'linear-regression'],
    thumbnail: 5,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { center: [1.2, 0] }), scene: 'aff',
        W: LA.clone(W0), b: LA.clone(B0), morph: 0, slide: 0, gridAlpha: 1, showSquare: true, corners: 0,
        mis: 0, mis2: 0, probe: 0, formula: false, lin: false, lockW: false, target: null, targetPoly: false,
        M3: I3(), lift: 0, pulse: 0, samples: 0, probe3: 0, chainPhase: 0, chain: false, mathMode: '', snap: false,
        X: design(),
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'homog': scene3d(g, s); if (s.chainPhase > 0 || s.chain) chainTiles(g, s); break;
        case 'ml': sceneML(g, s); break;
        case 'recap': sceneRecap(g, s); break;
        case 'play': if (s.view === '3d') scene3d(g, Object.assign({}, s, { lift: 1, samples: 1, pulse: 0, probe3: 0 }), { M3: aug(s.W, s.b), lift: 1 }); else scene2d(g, s, ctx); break;
        default: scene2d(g, s, ctx);
      }
    },

    math(s, tex) {
      const p = [];
      const Wm = () => tex.mat(s.W, { colors: 'cols' }), bv = () => tex.vec(s.b, { color: 'b' });
      const W3 = aug(s.W, s.b);
      switch (s.scene) {
        case 'homog':
          if (s.chain) { p.push(tex.eq(tex.sym('S̃'), tex.sym('T̃'), tex.sym('R̃'), tex.op('='), matW3(tex, CHAIN.map(r => r.map(x => Math.round(x * 100) / 100))))); p.push(tex.text('a chain of affine maps is one affine map')); break; }
          if (s.mathMode === 'lift') { p.push(tex.eq(tex.vec(['x', 'y']), tex.op('↦'), tex.vec(['x', 'y', '1'], { colors: (i) => (i === 2 ? 'teal' : '') }))); break; }
          p.push(tex.eq(tex.sym('W̃'), tex.op('='), matW3(tex, W3)));
          p.push(tex.text('top-left block: W · blue column: b · bottom row: 0 0 1', { color: 'muted' }));
          if (s.mathMode === 'append') p.push(tex.eq(tex.sym('W̃'), tex.vec([1, 0.5, 1], { color: 'v' }), tex.op('='), tex.vec(N.matVec(W3, [1, 0.5, 1]), { color: 'v' }), tex.op('='), tex.vec(['Wx + b', '1'])));
          break;
        case 'ml':
          p.push(tex.eq(tex.sym('ŷ'), tex.op('='), tex.rowvec(['1', 'x₁', 'x₂']), tex.vec(['w₀', 'w₁', 'w₂'], { color: 'w' }), tex.op('='), tex.text('w₀ + w₁x₁ + w₂x₂')));
          p.push(tex.eq(tex.text('layer: '), tex.sym('h'), tex.op('='), tex.text('fold('), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { color: 'b' }), tex.text(')')));
          break;
        case 'recap': return '';
        default: {
          p.push(tex.eq(tex.sym('W'), tex.op('='), Wm(), tex.op('  '), tex.sym('b', { color: 'b' }), tex.op('='), bv()));
          if (s.mathMode === 'mis') { p.push(tex.eq(tex.sym('f'), tex.op('('), tex.text('0'), tex.op(')'), tex.op('='), tex.sym('b', { color: 'b' }), tex.op('≠'), tex.text('0', { color: 'bad' }))); p.push(tex.eq(tex.text('f(2v) = ' + pt2(aff(s.W, s.b, [2, 1])) + ' ≠ 2·f(v) = ' + pt2(N.scale(aff(s.W, s.b, [1, 0.5]), 2)), { color: 'bad' }))); }
          else if (s.mathMode === 'formula') p.push(tex.eq(tex.sym('f'), tex.op('('), tex.sym('x', { color: 'v' }), tex.op(')'), tex.op('='), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { color: 'b' }), tex.op('='), Wm(), tex.vec([1, 1], { color: 'v' }), tex.op('+'), bv(), tex.op('='), tex.vec(aff(s.W, s.b, [1, 1]), { color: 'v' })));
          else if (s.mathMode === 'corners') p.push(tex.text('corners: b, b + col₁, b + col₂, b + col₁ + col₂ = ' + SQUARE.map(q => pt2(aff(s.W, s.b, q))).join(', ')));
          else p.push(tex.eq(tex.sym('f'), tex.op('('), tex.text('0'), tex.op(')'), tex.op('='), tex.vec(s.b, { color: 'b' }), tex.text(N.norm(s.b) > 1e-6 ? '  origin moved' : '  origin fixed', { color: N.norm(s.b) > 1e-6 ? 'b' : 'ok' })));
          if (s.scene === 'play') p.push(tex.eq(tex.sym('W̃'), tex.op('='), matW3(tex, W3)));
        }
      }
      return tex.block.apply(null, p);
    },

    steps: [
      { title: 'Transform, then slide', duration: 5,
        narration: 'Apply the shear W = [[1, 0.5],[0, 1]], a linear map with the origin fixed. Now slide the whole picture by the orange ' + bOr + ' = [2; −1]. Every point moves by the same arrow, including the origin, which lands at b. The grid is still straight and evenly spaced, but something has changed.',
        setup(s) { s.scene = 'aff'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.2, 0] }); s.W = LA.clone(W0); s.b = LA.clone(B0); s.morph = 0; s.slide = 0; s.showSquare = true; s.corners = 0; s.mis = 0; s.mis2 = 0; s.probe = 0; s.formula = false; s.gridAlpha = 1; s.mathMode = ''; },
        at(s, t) { s.morph = seg(t, 0, 0.4); s.slide = seg(t, 0.5, 0.9); } },

      { title: 'A slide is not linear', duration: 5, misconception: true,
        narration: 'Belief: “lines stayed straight, so this is linear.” Linear maps keep the origin fixed, and this one moved it: f(0) = b ≠ 0. Also f(2<span class="c-v">v</span>) is not 2f(v): doubling the input does not double the output arrow. A <b>translation</b> breaks linearity even though it bends nothing.',
        setup(s) { s.scene = 'aff'; s.morph = 1; s.slide = 1; s.mis = 0; s.mis2 = 0; s.showSquare = false; s.mathMode = 'mis'; s.gridAlpha = 0.55; s._c0 = LA.clone(s.cam); s._c1 = { center: [2.8, 0.2], halfH: 5.0 }; },
        at(s, t) { s.mis = t > 0.12 ? 1 : 0; s.mis2 = seg(t, 0.5, 0.8); const u = seg(t, 0.4, 0.6); s.cam = { center: lerpV(s._c0.center, s._c1.center, u), halfH: LA.lerp(s._c0.halfH, s._c1.halfH, u) }; } },

      { title: 'The affine map f(x) = Wx + b', duration: 4,
        narration: 'Linear, then shift: that is an <b>affine map</b>, written f(' + xV + ') = W' + xV + ' + ' + bOr + '. The vector b is the <b>bias</b>. It is a vector, one entry per output, not a single number: here the shift is 2 along x (the green dashed piece) and −1 along y (the red one).',
        setup(s) { s.scene = 'aff'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.4, 0.4] }); s.mis = 0; s.mis2 = 0; s.morph = 1; s.slide = 1; s.gridAlpha = 0.35; s.showSquare = false; s.probe = 0; s.formula = true; s.mathMode = 'formula'; },
        at(s, t) { s.probe = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Where the square lands', duration: 4,
        narration: 'Under the affine map the pale yellow unit square becomes a parallelogram sitting at ' + bOr + ': its corner that was the origin is now at b, and its sides are the columns of W, the green ' + '<span class="c-i">î</span> and red <span class="c-j">ĵ</span>. Shape from W, position from b. Part 6 draws a layer exactly this way.',
        setup(s) { s.scene = 'aff'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [2.2, -0.2], halfH: 3.0 }); s.probe = 0; s.formula = false; s.gridAlpha = 0.8; s.showSquare = true; s.morph = 0; s.slide = 0; s.corners = 0; s.mathMode = 'corners'; },
        at(s, t) { s.morph = seg(t, 0.05, 0.4); s.slide = seg(t, 0.45, 0.8); s.corners = seg(t, 0.85, 0.95); } },

      { title: 'The plane as a slice of 3D', duration: 5,
        narration: 'Here is a trick that makes the shift linear. Lift the plane into 3D and place it at height z = 1: every point [x; y] becomes [x; y; 1]. These are <b>homogeneous coordinates</b>. The teal sheet is the lifted plane; the 3D origin sits below it, untouched, which is exactly what a linear map needs.',
        setup(s) { s.scene = 'homog'; s.view = '3d'; s.cam = cam3d(); s.M3 = I3(); s.lift = 0; s.samples = 0; s.pulse = 0; s.probe3 = 0; s.chain = false; s.chainPhase = 0; s.mathMode = 'lift'; },
        at(s, t) { s.lift = seg(t, 0.1, 0.6); s.samples = t > 0.7 ? 1 : 0.2; LA.cam.tween(s, { yaw: 0.55 }, { yaw: 0.75 }, t); } },

      { title: 'A 3D shear slides the slice', duration: 6,
        narration: 'Now apply the 3×3 linear map W̃ = [[1, 0.5, 2],[0, 1, −1],[0, 0, 1]]: W in the top-left, ' + bOr + ' in the blue third column, a 1 in the corner. It keeps the 3D origin fixed, so it is linear, yet it slides the z = 1 slice sideways by exactly b. Orbit: the floor z = 0 only shears in place.',
        setup(s) { s.scene = 'homog'; s.view = '3d'; s.lift = 1; s.samples = 1; s.M3 = I3(); s._to = aug(W0, B0); s.pulse = 0; s.probe3 = 0; s.mathMode = 'w3'; },
        at(s, t) { s.M3 = lerpM(I3(), s._to, seg(t, 0.1, 0.8)); s.pulse = seg.pulse(t, 0.8, 1); LA.cam.tween(s, { yaw: 0.75 }, { yaw: 0.75 + 0.7 }, t); } },

      { title: 'Append a 1', duration: 4,
        narration: 'Written out: W̃·[x; y; 1] = [Wx + b; 1]. The third coordinate stays 1 because the bottom row is [0, 0, 1]; the first two are exactly the affine map. So “append a 1 to the input” turns Wx + b into a single matrix multiplication. Code and regression both lean on this.',
        setup(s) { s.scene = 'homog'; s.view = '3d'; s.lift = 1; s.samples = 0; s.M3 = aug(W0, B0); s.pulse = 0; s.probe3 = 0; s.mathMode = 'append'; },
        at(s, t) { s.probe3 = seg(t, 0.15, 0.8); } },

      { title: 'Chains of affine maps', duration: 6,
        narration: 'Because each affine map is now one 3×3 matrix, a sequence of them is a product: rotate, then shift, then scale is one <b>augmented matrix</b>. Graphics engines compose every camera move this way. And as in p2-04, a chain of affine layers with nothing between them is again a single affine map.',
        setup(s) { s.scene = 'homog'; s.view = '3d'; s.lift = 1; s.samples = 0; s.probe3 = 0; s.pulse = 0; s.chain = true; s.chainPhase = 0; s.M3 = I3(); s.mathMode = 'chain'; },
        at(s, t) {
          if (t < 0.68) {
            const u1 = seg(t, 0.03, 0.22), u2 = seg(t, 0.26, 0.42), u3 = seg(t, 0.46, 0.62);
            s.M3 = chainAt(30 * DEG * u1, u2, LA.lerp(1, 1.3, u3));
            s.chainPhase = t < 0.24 ? 0 : (t < 0.44 ? 1 : 2);
          } else { s.M3 = lerpM(I3(), CHAIN, seg(t, 0.72, 0.95)); s.chainPhase = 3; }
        } },

      { title: 'The bias in layers and the ones column', duration: 5, mlTie: true,
        narration: 'A neural layer is W' + xV + ' + ' + bOr + ' followed by a fold, the next lesson. Linear regression appends a 1 to every data row, the ones column of the design matrix, so that the intercept becomes just another weight: the homogeneous trick, applied to a whole dataset at once. Part 3 fits with it.',
        setup(s) { s.scene = 'ml'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.chain = false; s.chainPhase = 0; s.mathMode = ''; },
        at(s, t) { } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'A translation moves the origin, so it is not linear. Linear then shift is affine: f(x) = Wx + b, with the bias b a vector. Append a 1 to the input and the affine map becomes the 3×3 linear map [[W, b],[0, 1]] acting on the slice z = 1. Layers and regression both use this trick.',
        cards: ['<b>A translation moves the origin:</b> it is not linear, even though no line bends.', '<b>Affine: f(x) = Wx + b.</b> Shape from W, position from the bias vector b.', '<b>Append a 1:</b> [[W, b],[0, 1]] acts linearly on the slice z = 1; the ones column in regression is the same trick.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.cam = LA.cam.defaults2d(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the white moved origin to set the orange <span class="c-b">b</span>, and the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span> to set W. Switch to the 3D view to see the same map as a 3×3 shear of the slice z = 1.</p>',
      setup(s) { s.scene = 'play'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.2, 0] }); s.W = LA.clone(W0); s.b = LA.clone(B0); s.morph = 1; s.slide = 1; s.showSquare = true; s.corners = 0; s.gridAlpha = 1; s.lin = false; s.mathMode = ''; },
      controls: [
        { type: 'matrix', bind: 'W', label: 'W', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'b', label: 'b (orange)', colors: 'b', step: 0.1 },
        { type: 'toggle', bind: 'showSquare', label: 'Unit square' },
        { type: 'preset', label: 'View', items: [
          { label: '2D plane', apply(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.2, 0] }); } },
          { label: '3D homogeneous view', apply(s) { s.view = '3d'; s.cam = cam3d(); } },
        ] },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Shear + shift', values: { W: W0, b: B0 } },
          { label: 'Rotate + shift', values: { W: [[0.87, -0.5], [0.5, 0.87]], b: [1, 1] } },
          { label: 'Pure translation', values: { W: [[1, 0], [0, 1]], b: [1.5, 0.5] } },
          { label: 'Linear (b = 0)', values: { W: W0, b: [0, 0] } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the moved origin (the white dot) so that it lands at (2, −1). The orange ' + bOr + ' follows it.',
        setup(s) { s.scene = 'aff'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.2, 0] }); s.W = LA.clone(W0); s.b = [0, 0]; s.morph = 1; s.slide = 1; s.showSquare = true; s.corners = 0; s.mis = 0; s.mis2 = 0; s.probe = 0; s.formula = false; s.gridAlpha = 1; s.lockW = true; s.target = [2, -1]; s.targetPoly = false; s.lin = false; s.mathMode = ''; s.snap = true; },
        judge(s) { return LA.judge.near(s.b, [2, -1], LA.TOL.pos); },
        hint: 'The origin goes exactly where ' + bOr + ' points: 2 to the right, 1 down.',
        solution(s) { s.b = [2, -1]; } },
      { kind: 'manipulate',
        prompt: 'Start from the identity with no shift. Make the map fail linearity by moving the origin.',
        setup(s) { s.scene = 'aff'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.W = I2(); s.b = [0, 0]; s.morph = 1; s.slide = 1; s.showSquare = true; s.corners = 0; s.mis = 0; s.mis2 = 0; s.probe = 0; s.formula = false; s.gridAlpha = 1; s.lockW = true; s.target = null; s.targetPoly = false; s.lin = true; s.mathMode = ''; },
        judge(s) { const d = N.norm(s.b); return { ok: d > 0.3, progress: Math.min(1, d / 0.3) }; },
        hint: 'Any shift at all breaks f(0) = 0: drag the white origin dot away from the center.',
        solution(s) { s.b = [1, 1]; } },
      { kind: 'manipulate',
        prompt: 'Set W and ' + bOr + ' so the unit square lands on the gray parallelogram: drag the white origin, then the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span>.',
        setup(s) { s.scene = 'aff'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.5, 1] }); s.W = I2(); s.b = [0, 0]; s.morph = 1; s.slide = 1; s.showSquare = true; s.corners = 0; s.mis = 0; s.mis2 = 0; s.probe = 0; s.formula = false; s.gridAlpha = 1; s.lockW = false; s.target = null; s.targetPoly = true; s.lin = false; s.mathMode = 'corners'; s.snap = true; },
        judge(s) {
          const d = Math.max.apply(null, SQUARE.map((q, k) => N.dist(aff(s.W, s.b, q), TARGET[k])));
          return { ok: d <= LA.TOL.pos, progress: Math.max(0, 1 - d / 3) };
        },
        hint: 'The bottom-left corner tells you ' + bOr + ' = (1, 1); the two edges leaving it are the columns of W: [1; 0] and [1; 1].',
        solution(s) { s.W = [[1, 1], [0, 1]]; s.b = [1, 1]; } },
    ],
  });
})();
