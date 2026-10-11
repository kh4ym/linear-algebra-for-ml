/* p3-06 — Regression as X, w: normal equations, pseudo-inverse, min-norm.
   Rooms: the design-matrix panel (heatmap of X, the strip y, the XᵀX tile), feature space in 3D (x₁, x₂ on the
   floor, y up; the model is a plane), and weight space (2D / 3D) for the wide, one-equation demos.
   Split pictures are two g.viewport calls (panel + 3D room), like g.split but stacked when the canvas is tall. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, C = LA.color;
  const W_START = [0.5, 0.8, 0.3, 0];        // the starting plane of step 2: [w₀, w₁, w₂, w₃]
  const EPS_NEAR = 0.005;                    // near-duplicate offset in the conditioning step
  const PERT_I = 5;                          // the house whose price is nudged by 1 %
  const WIDE = { row: [2, 1], y: 4, foot: [1.6, 0.8] };
  const W3 = { row: [1, 2, 2], y: 6, foot: [2 / 3, 4 / 3, 4 / 3] };
  const YV = '<span class="c-b">y</span>';
  const ROLE = { ones: 'i', x1: 'j', x2: 'k', x3: 'j' };
  const TITLE = { ones: 'ones', x1: 'x₁', x2: 'x₂', x3: 'x₃' };

  /** Twelve synthetic houses (LA.rng seed 36): two features and a price. */
  function makeData() {
    const d = LA.rng.planeData(36, 12, { w: [1, 0.5], b: 1, noise: 0.3, xRange: [0.2, 2.6] });
    return d.X.map((x, i) => ({ x1: +x[0].toFixed(2), x2: +x[1].toFixed(2), y: +d.y[i].toFixed(2) }));
  }
  function makeNoise() { const r = LA.rng(361); const out = []; for (let i = 0; i < 12; i++) out.push(+LA.rng.gauss(r).toFixed(3)); return out; }

  /** Design matrix + fit for the current flags. Full weights are always [w₀, w₁, w₂, w₃] (absent columns → 0). */
  function fit(s, o) {
    o = o || {};
    const hasOnes = o.hasOnes === undefined ? s.hasOnes : o.hasOnes, dup = o.dup === undefined ? s.dup : o.dup;
    const eps = o.eps === undefined ? s.eps : o.eps, pert = o.perturb === undefined ? s.perturb : o.perturb;
    const d = s.data;
    const y = d.map((r, i) => r.y * (i === PERT_I ? 1 + 0.01 * pert : 1));
    const keys = (hasOnes ? ['ones'] : []).concat(['x1', 'x2'], dup ? ['x3'] : []);
    const colVals = { ones: d.map(() => 1), x1: d.map(r => r.x1), x2: d.map(r => r.x2), x3: d.map((r, i) => +(r.x1 + eps * s.noise[i]).toFixed(4)) };
    const X = d.map((r, i) => keys.map(k => colVals[k][i]));
    const w = N.lstsq(X, y);
    const full = [0, 0, 0, 0]; keys.forEach((k, j) => { full[['ones', 'x1', 'x2', 'x3'].indexOf(k)] = w[j]; });
    return { keys, colVals, X, y, w, full };
  }
  function model(s) {
    const f = fit(s);
    const wf = s.wDisp || f.full;
    const x3 = f.colVals.x3;
    const yhat = s.data.map((r, i) => wf[0] + wf[1] * r.x1 + wf[2] * r.x2 + wf[3] * x3[i]);
    const e = N.sub(f.y, yhat);
    const Xt = N.transpose(f.X);
    f.wf = wf; f.yhat = yhat; f.e = e; f.sse = N.dot(e, e);
    f.XtX = N.matMul(Xt, f.X); f.Xty = N.matVec(Xt, f.y); f.det = N.det(f.XtX);
    return f;
  }
  const planeZ = (wf, x1, x2) => wf[0] + (wf[1] + wf[3]) * x1 + wf[2] * x2;   // x₃ ≈ x₁ on the floor picture

  /* ---------------- layout ---------------- */
  function rects(size) {
    const W = size.w, H = size.h;
    if (W / H > 1.2) return { panel: { x: 0.6 * W, y: 0, w: 0.4 * W, h: H }, space: { x: 0, y: 0, w: 0.6 * W, h: H } };
    return { panel: { x: 0, y: 0, w: W, h: 0.4 * H }, space: { x: 0, y: 0.4 * H, w: W, h: 0.6 * H } };
  }
  /** A 2D viewport in px: world (x, −y) ↔ px (rect.x + x, rect.y + y). */
  function panel(g, r, opts, fn) {
    g.viewport({ x: r.x, y: r.y, w: Math.max(2, r.w), h: Math.max(2, r.h) }, Object.assign({ view: '2d', cam: { center: [r.w / 2, -r.h / 2], halfH: r.h / 2 }, border: false }, opts || {}), (g2, rr) => fn(g2, rr));
  }
  const P = (x, y) => [x, -y];
  function rect(g2, x, y, w, h, o) { g2.polygon([P(x, y), P(x + w, y), P(x + w, y + h), P(x, y + h)], Object.assign({ alpha: 1 }, o)); }
  function outline(g2, x, y, w, h, color, width, dashed) { g2.polyline([P(x, y), P(x + w, y), P(x + w, y + h), P(x, y + h)], { color, width: width || 1.4, closed: true, dashed }); }
  function bracket(g2, x, y0, y1, open, color) {
    const d = open ? 6 : -6, o = { color: color || 'hud', width: 1.4, alpha: 0.9 };
    g2.line(P(x, y0), P(x, y1), o); g2.line(P(x, y0), P(x + d, y0), o); g2.line(P(x, y1), P(x + d, y1), o);
  }
  const tint = (role, t) => C.mix(C.panel2, C[role], 0.18 + 0.82 * LA.clamp01(t));
  const div = (v, mx) => C.diverge(LA.clamp(v / (mx || 1), -1, 1));
  const f1 = v => { const t = LA.fmt.fixed(v, 1); return t === LA.fmt.MINUS + '0.0' ? '0.0' : t; };

  /* ---------------- 3D cameras ---------------- */
  const CAM_FEAT = () => ({ target: [1.3, 1.3, 2.25], dist: 10.4, yaw: 0.61, pitch: 0.44, fov: 36 });
  const CAM_W3 = () => ({ target: [0.9, 1.1, 1.1], dist: 10.5, yaw: 0.45, pitch: 0.38, fov: 36 });
  const CAM_W2 = () => ({ center: [1.55, 1.75], halfH: 2.5 });

  LA.registerLesson({
    id: 'p3-06-regression-matrix-form',
    part: 3,
    title: 'Regression as X, w: normal equations, pseudo-inverse, min-norm',
    subtitle: 'Rows are examples, columns are features; X⁺ handles tall and wide X alike.',
    minutes: 11,
    prereqs: ['p3-05-least-squares', 'p2-09-non-square', 'p2-08-rank-column-null'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D (d = 2 features, n = 12 rows)', realSize: 'X is n×d with n = 10⁶ and d = 10³ (tall), or a 10⁹-parameter network trained on 10⁶ examples (wide)', transfers: 'the shapes (n×d)(d×1) = n×1, the normal equations and both pseudo-inverse formulas are size-independent' },
    cantShow: ['more than two features as a plane: with d = 3 the “plane” is a 3D flat inside 4D', 'sample space for n = 12 (we stay in feature space here; the previous lesson did the flip)', 'how (XᵀX)⁻¹ behaves numerically beyond a hint: the condition number and ridge come later'],
    terms: [
      { term: 'design matrix X', def: 'Rows are examples, columns are features; the first column is all ones for the intercept.' },
      { term: 'feature space', def: 'The space whose axes are the features (plus one for y), where each example is a point.' },
      { term: 'tall X', def: 'More rows than columns (n > d): usually no exact fit, so we use least squares.' },
      { term: 'wide X', def: 'More columns than rows (n < d): infinitely many exact fits.' },
      { term: 'pseudo-inverse X⁺', def: 'The matrix that gives the least-squares fit for a tall X and the minimum-norm fit for a wide X.' },
      { term: 'minimum-norm solution', def: 'Among all exact solutions, the one closest to the origin.' },
      { term: 'ill-conditioned', def: 'Small changes in the data cause large changes in the answer.' },
    ],
    atlas: ['linear-regression', 'regularization', 'optimization'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults3d(), CAM_FEAT(), { center: [0, 0], halfH: 3 }),
        data: makeData(), noise: makeNoise(),
        layout: 'table', mathMode: 'table',
        hasOnes: true, dup: false, eps: 0, perturb: 0,
        wDisp: null, onesA: 1, tableT: 1, shapeT: 1,
        showPlane: false, planeGold: false, showSticks: false, fillK: 9, glow: null,
        showTile: false, noOnesGhost: false, cross: null, check: false,
        pinvT: 1, wPoint: [0, 4], showFoot: false, footT: 0, showRow: false, wideReadout: true,
        w3T: 1, inset: null, gdK: 0, gd2: false, pinvEntry: [0, 0], showEntry: false,
        chipOff: [0, 0], chipMode: false, sseNoOnes: 0,
      };
    },

    render(g, s, ctx) {
      const m = model(s);
      const inPlay = ctx.mode === 'play';
      if (s.layout === 'wide2') { drawWide2(g, s, ctx); return; }
      if (s.layout === 'wide3') { drawWide3(g, s); return; }
      if (s.layout === 'table') { panel(g, { x: 0, y: 0, w: g.size.w, h: g.size.h }, {}, (g2, r) => drawTable(g2, s, m, r)); return; }
      if (s.layout === 'pinv') { panel(g, { x: 0, y: 0, w: g.size.w, h: g.size.h }, {}, (g2, r) => drawPinv(g2, s, m, r)); return; }
      const R = rects(g.size);
      g.viewport(R.space, { view: '3d', cam: s.cam }, (g2, r) => drawFeature(g2, s, m, ctx, r, inPlay));
      panel(g, R.panel, { border: true }, (g2, r) => drawDesign(g2, s, m, r, ctx));
      if (s.inset) { const iw = 190, ih = 160, ir = { x: R.space.x + R.space.w - iw - 8, y: R.space.y + 8, w: iw, h: ih }; drawInset(g, s, ir); }
    },

    math(s, tex) {
      const m = model(s);
      const sym = tex.sym, op = tex.op, num = tex.num, eq = tex.eq, mat = tex.mat, vec = tex.vec, txt = tex.text;
      const what = sym('w', { hat: true, color: 'gold' }), Xp = sym('X', { sup: '+' }), wv = sym('w');
      const colsRow = () => op('[') + m.keys.map(k => sym(TITLE[k], { color: ROLE[k] })).join(op(' ')) + op(']');
      const colColors = m.keys.map(k => ROLE[k]);
      switch (s.mathMode) {
        case 'table': return tex.block(eq(sym('X'), op('='), colsRow()), eq(sym('X'), wv, op('≈'), sym('y', { color: 'b' })), eq(op('('), tex.dim(12, 3), op(')('), tex.dim(3, 1), op(') ='), tex.dim(12, 1)));
        case 'plane': return tex.block(eq(sym('ŷ', { color: 'gold' }), op('='), num(m.wf[0]), op('+'), num(m.wf[1]), sym('x', { sub: '1', color: 'j' }), op('+'), num(m.wf[2]), sym('x', { sub: '2', color: 'k' })), eq(txt('SSE'), op('='), num(m.sse, { color: 'gold' })));
        case 'normal': return tex.block(eq(tex.T(sym('X')), sym('X'), op('='), mat(m.XtX, { colors: (i, j) => i === j ? colColors[i] : 'text', digits: 2 })), eq(what, op('='), txt('solve('), tex.T(sym('X')) + sym('X'), op(','), tex.T(sym('X')) + sym('y', { color: 'b' }), txt(')'), op('='), vec(m.w, { color: 'gold' })));
        case 'noOnes': return tex.block(eq(sym('X'), op('='), colsRow()), eq(txt('SSE'), op('='), num(m.sse, { color: m.keys[0] === 'ones' ? 'gold' : 'bad' }), txt(m.keys[0] === 'ones' ? '' : '  ↑ plane nailed to the origin', { color: 'bad' })));
        case 'pinv': return tex.block(eq(Xp, op('='), tex.inv(tex.paren(tex.T(sym('X')) + sym('X'))), tex.T(sym('X')), tex.dim(3, 12)), eq(Xp, sym('X'), op('='), sym('I', { sub: '3' })), eq(sym('X'), Xp, op('='), sym('P'), txt('  (P² = P)', { color: 'muted' })));
        case 'wide': return tex.block(eq(tex.rowvec([2, 1]), op('·'), wv, op('='), num(4)), eq(tex.norm(wv), op('='), num(N.norm(s.wPoint), { color: 'muted' })), eq(txt('residual'), op('='), num(Math.abs(2 * s.wPoint[0] + s.wPoint[1] - 4) < 0.005 ? 0 : 2 * s.wPoint[0] + s.wPoint[1] - 4)));
        case 'minnorm': return tex.block(eq(Xp, op('='), tex.T(sym('X')), tex.inv(tex.paren(sym('X') + tex.T(sym('X')))), op('='), vec([2, 1]), op('/'), num(5), op('='), vec([0.4, 0.2], { color: 'mag' })), eq(wv, op('='), Xp, sym('y', { color: 'b' }), op('='), vec(WIDE.foot, { color: 'mag' })));
        case 'entry': return tex.block(eq(sym('X'), op('='), tex.rowvec([2, 1]), op(','), sym('y', { color: 'b' }), op('='), num(4)), eq(Xp, op('='), txt('?'), op('  →  '), Xp, sym('y', { color: 'b' }), op('='), vec(N.scale(s.pinvEntry, 4), { color: 'muted' })));
        case 'wide3': return tex.block(eq(wv, op('='), tex.T(sym('X')), tex.inv(tex.paren(sym('X') + tex.T(sym('X')))), sym('y', { color: 'b' }), op('='), vec(W3.foot, { color: 'mag' })), eq(sym('X'), op('='), tex.rowvec(W3.row), op(','), sym('y', { color: 'b' }), op('='), num(6)));
        case 'dup': return tex.block(eq(txt('det'), tex.paren(tex.T(sym('X')) + sym('X')), op('='), num(Math.abs(m.det) < 1e-6 ? 0 : m.det, { color: 'bad' })), eq(sym('w', { sub: '1' }), op('+'), sym('w', { sub: '3' }), op('='), num(m.wf[1] + m.wf[3]), txt('  (any split fits)', { color: 'muted' })));
        case 'near': { const a = fit(s, { perturb: 0 }).full, b = fit(s, { perturb: 1 }).full; const dw = N.norm(N.sub(b, a)), dy = s.data[PERT_I].y * 0.01;
          return tex.block(eq(txt('det'), tex.paren(tex.T(sym('X')) + sym('X')), op('≈'), num(m.det, { digits: 3 })), eq(txt('Δy ='), num(dy), op('  →  '), txt('|Δw| ='), num(dw, { color: 'bad' }))); }
        case 'gd': return tex.block(eq(wv, op('←'), wv, op('+'), sym('η'), tex.T(sym('X')), tex.paren(sym('y', { color: 'b' }) + op('−') + sym('X') + wv)), eq(sym('w', { sub: '0' }), op('='), num(0), op('⇒'), wv, op('→'), Xp, sym('y', { color: 'b' })));
        default: return tex.block(eq(what, op('='), Xp, sym('y', { color: 'b' }), op('='), vec(m.w, { color: 'gold' })));
      }
    },

    steps: [
      { title: 'The table becomes X', duration: 3.5,
        narration: 'Twelve houses: two features each (x₁, x₂) and a price y. Stack them: one <i>row per house</i>, one <i>column per feature</i>, plus a column of ones for the intercept. That 12×3 block is the <b>design matrix X</b>; the prices form the column ' + YV + '. Rows are examples, columns are features, always.',
        setup(s) { s.layout = 'table'; s.view = '2d'; s.mathMode = 'table'; s.hasOnes = true; s.dup = false; s.eps = 0; s.perturb = 0; s.wDisp = null; s.tableT = 0; s.shapeT = 0; s.onesA = 1; },
        at(s, t) { s.tableT = seg(t, 0.25, 0.75); s.shapeT = seg(t, 0.75, 0.95); } },

      { title: 'Feature space: the model is a plane', duration: 4.5,
        narration: 'Now draw each house as a point: x₁ and x₂ on the floor, y as height. The model y = w₀ + w₁x₁ + w₂x₂ is a <i>plane</i> over the floor, and the dashed vertical sticks are the residuals. This room is <b>feature space</b>, different from the sample space of the last lesson, where the axes were houses.',
        setup(s) { s.layout = 'split'; s.view = '3d'; Object.assign(s.cam, CAM_FEAT()); s.tableT = 1; s.shapeT = 1; s.mathMode = 'plane'; s.wDisp = W_START.slice(); s.showPlane = true; s.planeGold = false; s.showSticks = false; s.showTile = false; s.fillK = 0; },
        at(s, t) { s.showSticks = t > 0.45; LA.cam.tween(s, { yaw: 0.35 }, { yaw: 0.61 }, seg(t, 0, 0.8)); } },

      { title: 'Normal equations, matrix form', duration: 4,
        narration: 'Same rule as before: the residual must be perpendicular to every column, Xᵀ(y − Xŵ) = 0, so XᵀX ŵ = Xᵀy. XᵀX is now 3×3, nine dot products of columns, and Xᵀy is three. Solve (never invert) and the plane snaps into place.',
        setup(s) { s.showSticks = true; s.showTile = true; s.fillK = 0; s.glow = null; s.mathMode = 'normal'; s._wa = W_START.slice(); s._wb = fit(s).full; s.wDisp = s._wa.slice(); s.planeGold = false; },
        at(s, t) {
          const k = seg(t, 0, 0.42, 'linear') * 9; s.fillK = t <= 0 ? 0 : Math.min(9, Math.floor(k) + 1);
          s.glow = t < 0.42 ? [Math.floor(Math.min(8, k) / 3), Math.floor(Math.min(8, k)) % 3] : null;
          const u = seg(t, 0.45, 0.95); s.wDisp = u >= 1 ? null : lerpV(s._wa, s._wb, u); s.planeGold = t > 0.95;
        } },

      { title: 'The intercept is not optional', duration: 4, misconception: true,
        narration: '✕ “Drop the ones column; the slopes will adjust.” Without it the plane is nailed to the origin: it must pass through (0, 0, 0), and the sticks grow. The intercept is not a separate formula; it is just one more column of X, and it is the column that lets the plane float.',
        setup(s) { s.wDisp = null; s.planeGold = true; s.fillK = 9; s.glow = null; s.mathMode = 'noOnes'; s._wa = fit(s, { hasOnes: true }).full; s._wb = fit(s, { hasOnes: false }).full; s.onesA = 1; s.cross = null; s.check = false; s.noOnesGhost = false; },
        at(s, t) {
          const down = seg(t, 0.08, 0.38), up = seg(t, 0.68, 0.92), u = down * (1 - up);
          s.wDisp = (u <= 0 && t > 0.92) ? null : lerpV(s._wa, s._wb, u);
          s.hasOnes = !(t > 0.2 && t < 0.8); s.onesA = 1 - 0.85 * u;
          s.cross = t > 0.3 ? 'origin' : null; s.noOnesGhost = t > 0.8; s.check = t > 0.92; s.planeGold = !(t > 0.2 && t < 0.8);
        } },

      { title: 'The pseudo-inverse of a tall X', duration: 4,
        narration: 'Pack the solve into one matrix: ŵ = (XᵀX)⁻¹Xᵀy = X⁺y, the <b>pseudo-inverse X⁺</b>. For this <b>tall X</b> it is 3×12, the shape of X turned on its side, and X⁺X = I₃: it undoes X. XX⁺ is the 12×12 projection onto the column space, the shadow machine from the projection lesson.',
        setup(s) { s.layout = 'pinv'; s.view = '2d'; s.hasOnes = true; s.wDisp = null; s.onesA = 1; s.cross = null; s.check = false; s.noOnesGhost = false; s.mathMode = 'pinv'; s.pinvT = 0; },
        at(s, t) { s.pinvT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Wide X: one equation, many perfect fits', duration: 4.5, interactive: true,
        narration: 'Flip the shape: one house, two weights. X = [2 1] (no intercept) and y = 4. Every w with 2w₁ + w₂ = 4 fits <i>perfectly</i>: a whole line of zero-residual solutions. This is a <b>wide X</b>. Drag w along the gold line: the error stays zero. Which one should we pick?',
        setup(s) { s.layout = 'wide2'; s.view = '2d'; Object.assign(s.cam, CAM_W2()); s.mathMode = 'wide'; s._p0 = [0, 4]; s._p1 = [2.4, -0.8]; s.wPoint = [0, 4]; s.showFoot = false; s.footT = 0; s.showRow = false; s.gdK = 0; s.gd2 = false; },
        at(s, t) { s.wPoint = t < 0.5 ? lerpV(s._p0, s._p1, seg(t, 0.08, 0.5)) : lerpV(s._p1, [0.6, 2.8], seg(t, 0.55, 0.95)); } },

      { title: 'Minimum norm: the foot of the perpendicular', duration: 3.5,
        narration: 'The pseudo-inverse picks the solution closest to the origin, the <b>minimum-norm solution</b>: the perpendicular foot from the origin onto the line, w = [1.6; 0.8]. For a wide X the formula flips to X⁺ = Xᵀ(XXᵀ)⁻¹: XᵀX (2×2) has no inverse now, but XXᵀ = 2·2 + 1·1 = 5 can be inverted, so X⁺ = [2; 1]/5.',
        setup(s) { s.mathMode = 'minnorm'; s._p0 = LA.clone(s.wPoint); s.showFoot = true; s.footT = 0; s.showRow = true; },
        at(s, t) { s.footT = seg(t, 0, 0.45); s.wPoint = lerpV(s._p0, WIDE.foot, seg(t, 0.5, 1)); } },

      { title: 'Wide in 3D: a plane of solutions', duration: 4,
        narration: 'With three weights and one equation, [1 2 2]·w = 6, the solutions form a whole plane in weight space. The minimum-norm solution is still the perpendicular foot from the origin, along the row direction. Label it honestly: wide X, n &lt; d, more parameters than data. Modern networks live here.',
        setup(s) { s.layout = 'wide3'; s.view = '3d'; Object.assign(s.cam, CAM_W3()); s.mathMode = 'wide3'; s.w3T = 0; },
        at(s, t) { s.w3T = seg(t, 0.1, 0.8); LA.cam.tween(s, { yaw: 0.25 }, { yaw: 0.45 }, seg(t, 0, 1, 'linear')); } },

      { title: 'XᵀX is not always invertible', duration: 3.5, misconception: true,
        narration: '✕ “XᵀX can always be inverted.” Add a feature x₃ that is an exact copy of x₁. The plane fits exactly as before, but w₁ and w₃ can now trade off freely, and det(XᵀX) = 0. Redundant features make a tall X behave like a wide one. The pseudo-inverse still answers, with the minimum-norm split.',
        setup(s) { s.layout = 'split'; s.view = '3d'; Object.assign(s.cam, CAM_FEAT()); s.hasOnes = true; s.dup = false; s.eps = 0; s.perturb = 0; s.wDisp = null; s.planeGold = true; s.showSticks = true; s.showTile = true; s.fillK = 16; s.mathMode = 'dup'; s.cross = null; s.inset = null; },
        at(s, t) { s.dup = t > 0.2; s.cross = t > 0.45 ? 'det' : null; s.inset = t > 0.6 ? 'dup' : null; } },

      { title: 'Nearly redundant: a steady plane, wobbly weights', duration: 4.5,
        narration: 'Make the copy <i>almost</i> exact: x₃ = x₁ + tiny noise. Now XᵀX is invertible, but barely. Nudge one price by 1 % and the plane hardly moves, yet w₁ and w₃ swing by about 1.5 each, in opposite directions. The predictions are stable; the weights are not. The problem is <b>ill-conditioned</b>.',
        setup(s) { s.dup = true; s.eps = EPS_NEAR; s.perturb = 0; s.cross = null; s.inset = 'near'; s.mathMode = 'near'; },
        at(s, t) { s.perturb = seg(t, 0.2, 0.6); } },

      { title: 'Why ML cares: gradient descent from zero finds the min-norm fit', duration: 4, mlTie: true,
        narration: 'Over-parameterized models have whole planes of perfect fits. Gradient descent started at w = 0 moves only along row directions, so it lands on the minimum-norm solution, the same point X⁺y picks. That hidden preference is called implicit regularization, and least-squares solvers in every library compute exactly X⁺y.',
        setup(s) { s.layout = 'wide2'; s.view = '2d'; Object.assign(s.cam, CAM_W2()); s.dup = false; s.eps = 0; s.perturb = 0; s.inset = null; s.mathMode = 'gd'; s.showFoot = true; s.footT = 1; s.showRow = true; s.wPoint = WIDE.foot.slice(); s.gdK = 0; s.gd2 = false; s.wideReadout = false; },
        at(s, t) { s.gdK = seg(t, 0.05, 0.6, 'linear') * 10; s.gd2 = t > 0.65; } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'X stacks examples as rows, features as columns, ones for the intercept. Tall X: least squares, ŵ = X⁺y = (XᵀX)⁻¹Xᵀy, a plane over feature space. Wide X: a line or plane of exact fits, and X⁺ = Xᵀ(XXᵀ)⁻¹ picks the minimum-norm one. Redundant columns make weights wobble.',
        cards: ['<b>Rows = examples, columns = features.</b> A 12×3 X with a ones column; y is 12×1.', '<b>Tall X: the shadow.</b> ŵ = X⁺y with X⁺ = (XᵀX)⁻¹Xᵀ; the residual is perpendicular to every column.', '<b>Wide X: the min-norm foot.</b> X⁺ = Xᵀ(XXᵀ)⁻¹ picks the exact fit closest to the origin.'],
        setup(s) { s.layout = 'split'; s.view = '3d'; Object.assign(s.cam, CAM_FEAT()); s.gdK = 0; s.gd2 = false; s.wideReadout = true; s.hasOnes = true; s.dup = false; s.wDisp = null; s.planeGold = true; s.showSticks = true; s.showTile = false; s.inset = 'wide'; s.mathMode = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag any house up or down (the white points), switch the intercept column off, add a duplicate feature and shrink its noise, or open the wide [2 1] demo and drag w along the line of perfect fits.</p>',
      setup(s) { s.layout = 'split'; s.view = '3d'; Object.assign(s.cam, CAM_FEAT()); s.hasOnes = true; s.dup = false; s.eps = 0.2; s.wDisp = null; s.planeGold = true; s.showPlane = true; s.showSticks = true; s.showTile = true; s.fillK = 16; s.tableT = 1; s.shapeT = 1; s.inset = null; s.mathMode = 'normal'; s.wPoint = [0, 4]; s.showFoot = true; s.footT = 1; s.showRow = true; },
      controls: [
        { type: 'toggle', bind: 'hasOnes', label: 'Intercept column (ones)' },
        { type: 'toggle', bind: 'dup', label: 'Add a duplicate feature x₃ ≈ x₁' },
        { type: 'slider', bind: 'eps', min: 0, max: 0.5, step: 0.005, label: 'duplicate noise ε', color: 'j' },
        { type: 'preset', label: 'Scenes', items: [
          { label: 'Tall, 12 houses', apply(s) { toTall(s); s.hasOnes = true; s.dup = false; s.inset = null; s.mathMode = 'normal'; } },
          { label: 'No intercept', apply(s) { toTall(s); s.hasOnes = false; s.dup = false; s.inset = null; s.mathMode = 'noOnes'; } },
          { label: 'Duplicate feature', apply(s) { toTall(s); s.hasOnes = true; s.dup = true; s.eps = 0; s.inset = 'dup'; s.mathMode = 'dup'; } },
          { label: 'Wide [2 1]', apply(s) { s.layout = 'wide2'; s.view = '2d'; Object.assign(s.cam, CAM_W2()); s.wPoint = [0, 4]; s.showFoot = true; s.footT = 1; s.showRow = true; s.mathMode = 'wide'; } },
        ] },
        { type: 'button', label: 'Reset the prices', onClick(s) { s.data = makeData(); } },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'The ones column has been pulled out of X and the plane is nailed to the origin. Drag the green <span class="c-i">ones</span> chip back into its slot so the plane can float.',
        setup(s) { toTall(s); s.hasOnes = false; s.chipMode = true; s.chipOff = [210, 0]; s.showTile = false; s.mathMode = 'noOnes'; s.sseNoOnes = sseOf(s, false); },
        judge(s) { const sse = sseOf(s, s.hasOnes); const d = Math.hypot(s.chipOff[0], s.chipOff[1]); return { ok: s.hasOnes === true && sse < 0.7 * s.sseNoOnes, progress: Math.max(0, 1 - d / 210) }; },
        hint: 'Which column lets the plane float? Drop the green chip onto the dashed slot at the left of X.',
        solution(s) { s.chipOff = [0, 0]; s.hasOnes = true; } },
      { kind: 'manipulate',
        prompt: 'Drag w along the gold line of perfect fits to the <b>minimum-norm solution</b>, the one closest to the origin.',
        setup(s) { s.layout = 'wide2'; s.view = '2d'; Object.assign(s.cam, CAM_W2()); s.wPoint = [0, 4]; s.showFoot = false; s.showRow = false; s.gdK = 0; s.gd2 = false; s.wideReadout = true; s.mathMode = 'wide'; },
        judge(s) { return LA.judge.near(s.wPoint, WIDE.foot, LA.TOL.pos); },
        hint: 'The shortest arrow from the origin meets the line at a right angle.',
        solution(s) { s.wPoint = WIDE.foot.slice(); } },
      { kind: 'entry',
        prompt: 'Type X⁺ for the wide X = [2 1] (a 2×1 column). The gray point shows your X⁺ times y = 4.',
        bind: 'pinvEntry', answer: [0.4, 0.2], tol: LA.TOL.entry, label: 'X⁺', colors: 'muted', step: 0.05,
        setup(s) { s.layout = 'wide2'; s.view = '2d'; Object.assign(s.cam, CAM_W2()); s.wPoint = [0, 4]; s.showFoot = false; s.showRow = true; s.gdK = 0; s.gd2 = false; s.wideReadout = false; s.showEntry = true; s.pinvEntry = [0, 0]; s.mathMode = 'entry'; } },
    ],
  });

  function toTall(s) { s.layout = 'split'; s.view = '3d'; Object.assign(s.cam, CAM_FEAT()); s.wDisp = null; s.planeGold = true; s.showPlane = true; s.showSticks = true; s.showTile = true; s.fillK = 16; s.tableT = 1; s.shapeT = 1; s.onesA = 1; s.perturb = 0; s.chipMode = false; s.chipOff = [0, 0]; }
  function sseOf(s, hasOnes) { const f = fit(s, { hasOnes }); const yh = N.matVec(f.X, f.w); const e = N.sub(f.y, yh); return N.dot(e, e); }

  /* ---------------- the design-matrix panel ---------------- */
  function drawDesign(g2, s, m, r, ctx) {
    const n = 12, compact = r.h < 330;
    const rh = LA.clamp((r.h - (compact ? 62 : 90)) / (compact ? n : n + 9), 13, 24);
    const cw = compact ? 38 : 42, hx = 52, hy = compact ? 36 : 48;
    const slots = ['ones', 'x1', 'x2'].concat(s.dup ? ['x3'] : []);
    const yx = hx + slots.length * cw + 16;
    const glowCols = s.glow ? [m.keys[s.glow[0]], m.keys[s.glow[1]]] : [];
    g2.label('design matrix X (synthetic houses)', P(hx - 40, compact ? 10 : 16), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    slots.forEach((k, j) => {
      const x = hx + j * cw, present = m.keys.indexOf(k) >= 0, vals = m.colVals[k];
      const a = k === 'ones' ? (s.chipMode ? 0 : (present ? s.onesA : Math.min(s.onesA, 0.15))) : 1;
      g2.label(TITLE[k], P(x + cw / 2, hy - 9), { color: ROLE[k], size: 12.5, font: k === 'ones' ? 'ui' : 'math', bold: true, alpha: Math.max(0.3, a) });
      if (k === 'ones' && (!present || s.chipMode) && a < 0.2) { outline(g2, x + 1, hy, cw - 2, n * rh, 'i', 1.2, true); }
      drawColumn(g2, x, hy, cw, rh, vals, ROLE[k], a, k === 'ones', k === 'x3');
      if (glowCols.indexOf(k) >= 0) outline(g2, x, hy - 1, cw, n * rh + 2, 'gold', 2.2);
    });
    // the chip for check 1: the pulled-out ones column
    if (s.chipMode) {
      const sx = hx + s.chipOff[0], sy = hy + s.chipOff[1];
      drawColumn(g2, sx, sy, cw, rh, m.colVals.ones, 'i', 1, true, false);
      outline(g2, sx, sy, cw, n * rh, 'i', 1.6);
      g2.handle('chip', P(sx + cw / 2, sy + n * rh / 2), (p, st) => {
        const ox = p[0] - cw / 2 - hx, oy = -p[1] - n * rh / 2 - hy;
        const near = Math.hypot(ox, oy) < 16;
        st.chipOff = near ? [0, 0] : [ox, oy]; st.hasOnes = near;
      }, { color: 'i', r: 16 });
    }
    // y strip
    g2.label('y', P(yx + cw / 2, hy - 9), { color: 'b', size: 13, font: 'math', bold: true });
    drawColumn(g2, yx, hy, cw, rh, m.y, 'b', 1, false, false);
    for (let i = 0; i < n; i++) g2.label(String(i + 1), P(hx - 8, hy + (i + 0.5) * rh), { anchor: 'right', color: 'muted', size: 10.5, font: 'ui' });
    if (s.shapeT > 0) {
      const yb = hy + n * rh + 13;
      bracket(g2, hx - 3, hy - 2, hy + n * rh + 2, true); bracket(g2, hx + slots.length * cw + 3, hy - 2, hy + n * rh + 2, false);
      g2.label('X  ' + n + '×' + m.keys.length, P(hx + slots.length * cw / 2, yb), { color: 'text', size: 12, font: 'ui', alpha: s.shapeT });
      g2.label('12×1', P(yx + cw / 2, yb), { color: 'b', size: 11.5, font: 'ui', alpha: s.shapeT });
    }
    // XᵀX tile
    if (s.showTile) {
      const k = m.keys.length, tcw = k > 3 ? 40 : 46, trh = 22;
      const right = r.w >= 470;
      const tx = right ? yx + cw + 50 : hx, ty = right ? hy + 6 : hy + n * rh + 44;
      g2.label('XᵀX =', P(tx, ty - 14), { anchor: 'left', color: 'text', size: 12.5, font: 'ui' });
      bracket(g2, tx, ty - 2, ty + k * trh + 2, true); bracket(g2, tx + k * tcw + 4, ty - 2, ty + k * trh + 2, false);
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        if (i * k + j >= s.fillK) continue;
        const hl = s.glow && s.glow[0] === i && s.glow[1] === j;
        g2.label(f1(m.XtX[i][j]), P(tx + 2 + (j + 0.5) * tcw, ty + (i + 0.5) * trh), { color: hl ? 'gold' : (i === j ? ROLE[m.keys[i]] : 'hud'), size: 12.5, font: 'mono', bold: hl });
      }
      if (k === m.keys.length && s.fillK >= k * k) {
        const det = Math.abs(m.det) < 1e-6 ? 0 : m.det;
        const bad = s.dup;
        g2.label('det = ' + (bad ? (Math.abs(det) < 1e-6 ? '0' : LA.fmt(det, 3)) : LA.fmt(det, 1)), P(tx, ty + k * trh + 18), { anchor: 'left', color: bad ? 'bad' : 'muted', size: 12.5, font: 'ui', bold: bad });
        if (s.cross === 'det') g2.marker.cross(P(tx + 110, ty + k * trh + 18), { size: 18 });
        if (s.dup && s.eps === 0) g2.label('x₃ = x₁: columns 2 and 4 are equal', P(tx, ty + k * trh + 38), { anchor: 'left', color: 'muted', size: 11, font: 'ui' });
      }
    }
    g2.label('SSE = ' + LA.fmt(m.sse), P(r.w - 10, compact ? 10 : 16), { anchor: 'right', color: 'hud', size: 12.5, font: 'ui' });
  }
  /** One heatmap column: tinted cells with printed values. */
  function drawColumn(g2, x, y0, cw, rh, vals, role, alpha, flat, dashed) {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (let i = 0; i < vals.length; i++) {
      const t = flat ? 0.55 : (vals[i] - lo) / ((hi - lo) || 1);
      rect(g2, x, y0 + i * rh, cw, rh, { fill: tint(role, t), alpha: alpha, stroke: 'line', width: 0.7 });
      if (alpha > 0.25 && rh >= 12) g2.label(flat ? '1' : f1(vals[i]), P(x + cw / 2, y0 + (i + 0.5) * rh), { color: 'text', size: rh < 16 ? 10 : 11, font: 'mono', bg: false, alpha: Math.min(1, alpha + 0.1) });
    }
    if (dashed) outline(g2, x, y0, cw, vals.length * rh, 'k', 1.6, true);
  }

  /* ---------------- step 1: the plain table morphing into the heatmap ---------------- */
  function drawTable(g2, s, m, r) {
    const n = 12, rh = LA.clamp((r.h - 140) / n, 18, 36), cw = Math.min(78, (r.w - 140) / 4.4);
    const x0 = Math.max(70, (r.w - 4.4 * cw) / 2), y0 = 70;
    const keys = ['ones', 'x1', 'x2'], u = s.tableT;
    g2.label('Twelve synthetic houses', P(x0 - 50, 24), { anchor: 'left', color: 'muted', size: 12.5, font: 'ui' });
    g2.label('house', P(x0 - 12, y0 - 14), { anchor: 'right', color: 'muted', size: 11.5, font: 'ui' });
    keys.forEach((k, j) => {
      const x = x0 + j * cw;
      g2.label(TITLE[k], P(x + cw / 2, y0 - 14), { color: u > 0.3 ? ROLE[k] : 'hud', size: 14, font: k === 'ones' ? 'ui' : 'math', bold: true, alpha: k === 'ones' ? Math.max(0.15, u) : 1 });
      const vals = m.colVals[k], lo = Math.min(...vals), hi = Math.max(...vals);
      for (let i = 0; i < n; i++) {
        const t = k === 'ones' ? 0.55 : (vals[i] - lo) / ((hi - lo) || 1);
        if (u > 0) rect(g2, x + 2, y0 + i * rh + 1, cw - 4, rh - 2, { fill: tint(ROLE[k], t), alpha: u, stroke: 'line', width: 0.7 });
        if (k !== 'ones' || u > 0.1) g2.label(k === 'ones' ? '1' : LA.fmt.fixed(vals[i], 2), P(x + cw / 2, y0 + (i + 0.5) * rh), { color: 'text', size: 13, font: 'mono', bg: false, alpha: k === 'ones' ? u : 1 });
      }
    });
    const yx = x0 + 3 * cw + 0.4 * cw;
    g2.label(u > 0.3 ? 'y' : 'price y', P(yx + cw / 2, y0 - 14), { color: 'b', size: 14, font: u > 0.3 ? 'math' : 'ui', bold: true });
    const ys = m.y, lo = Math.min(...ys), hi = Math.max(...ys);
    for (let i = 0; i < n; i++) {
      if (u > 0) rect(g2, yx + 2, y0 + i * rh + 1, cw - 4, rh - 2, { fill: tint('b', (ys[i] - lo) / (hi - lo)), alpha: u, stroke: 'line', width: 0.7 });
      g2.label(LA.fmt.fixed(ys[i], 2), P(yx + cw / 2, y0 + (i + 0.5) * rh), { color: 'text', size: 13, font: 'mono', bg: false });
      g2.label(String(i + 1), P(x0 - 12, y0 + (i + 0.5) * rh), { anchor: 'right', color: 'muted', size: 11.5, font: 'ui' });
    }
    if (s.shapeT > 0) {
      const yb = y0 + n * rh + 18, a = s.shapeT;
      bracket(g2, x0 - 2, y0 - 3, y0 + n * rh + 3, true); bracket(g2, x0 + 3 * cw + 2, y0 - 3, y0 + n * rh + 3, false);
      bracket(g2, yx - 2, y0 - 3, y0 + n * rh + 3, true, 'b'); bracket(g2, yx + cw + 2, y0 - 3, y0 + n * rh + 3, false, 'b');
      g2.label('X   12×3', P(x0 + 1.5 * cw, yb), { color: 'text', size: 14, font: 'ui', bold: true, alpha: a });
      g2.label('y   12×1', P(yx + cw / 2, yb), { color: 'b', size: 14, font: 'ui', bold: true, alpha: a });
      g2.label('rows = examples (houses), columns = features', P(x0 + 1.5 * cw, yb + 22), { color: 'muted', size: 12, font: 'ui', alpha: a });
    }
  }

  /* ---------------- step 5: X, X⁺ and the two products as tiles ---------------- */
  function drawPinv(g2, s, m, r) {
    const X = m.X, Xp = N.pinv(X), XpX = N.matMul(Xp, X), XXp = N.matMul(X, Xp);
    const c = LA.clamp(Math.min(r.w / 32, r.h / 36), 14, 22);
    const a1 = seg(s.pinvT, 0, 0.3), a2 = seg(s.pinvT, 0.3, 0.6), a3 = seg(s.pinvT, 0.6, 0.95);
    const x0 = 36, y0 = 48;
    // X (tall, column tints)
    g2.label('X  (12×3, tall)', P(x0, y0 - 18), { anchor: 'left', color: 'text', size: 12.5, font: 'ui', alpha: Math.max(0.2, a1) });
    ['ones', 'x1', 'x2'].forEach((k, j) => { const vals = m.colVals[k], lo = Math.min(...vals), hi = Math.max(...vals); for (let i = 0; i < 12; i++) rect(g2, x0 + j * c, y0 + i * c, c, c, { fill: tint(ROLE[k], k === 'ones' ? 0.55 : (vals[i] - lo) / ((hi - lo) || 1)), alpha: Math.max(0.15, a1), stroke: 'line', width: 0.6 }); });
    // X⁺ (wide, diverging)
    const px = x0 + 3 * c + 40, mxp = Math.max(...Xp.flat().map(Math.abs));
    g2.label('X⁺  (3×12, wide)', P(px, y0 - 18), { anchor: 'left', color: 'mag', size: 12.5, font: 'ui', alpha: Math.max(0.2, a1) });
    for (let i = 0; i < 3; i++) for (let j = 0; j < 12; j++) rect(g2, px + j * c, y0 + i * c, c, c, { fill: div(Xp[i][j], mxp), alpha: Math.max(0.15, a1), stroke: 'line', width: 0.6 });
    // X⁺X = I₃
    const iy = y0 + 3 * c + 46, ic = c * 1.5;
    if (a2 > 0) {
      g2.label('X⁺X = I₃', P(px, iy - 16), { anchor: 'left', color: 'text', size: 12.5, font: 'ui', alpha: a2 });
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { rect(g2, px + j * ic, iy + i * ic, ic, ic, { fill: div(XpX[i][j], 1), alpha: a2, stroke: 'line', width: 0.6 }); g2.label(f1(XpX[i][j]), P(px + (j + 0.5) * ic, iy + (i + 0.5) * ic), { color: 'text', size: 11, font: 'mono', bg: false, alpha: a2 }); }
      g2.label('undoes X', P(px, iy + 3 * ic + 14), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui', alpha: a2 });
    }
    // XX⁺ = P (12×12)
    const qx = px + 3 * ic + 40, qy = iy, mq = Math.max(...XXp.flat().map(Math.abs));
    if (a3 > 0) {
      g2.label('XX⁺ = P  (12×12)', P(qx, qy - 16), { anchor: 'left', color: 'text', size: 12.5, font: 'ui', alpha: a3 });
      for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) rect(g2, qx + j * c, qy + i * c, c, c, { fill: div(XXp[i][j], mq), alpha: a3, stroke: 'line', width: 0.4 });
      g2.label('projection onto the column space, P² = P', P(qx, qy + 12 * c + 16), { anchor: 'left', color: 'teal', size: 11.5, font: 'ui', alpha: a3 });
      g2.label('symmetric, soft diagonal', P(qx, qy + 12 * c + 34), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui', alpha: a3 });
    }
    // legend
    const ly = Math.max(y0 + 12 * c + 30, iy + 3 * ic + 40);
    g2.label('colour scale:', P(x0, ly), { anchor: 'left', color: 'muted', size: 11, font: 'ui' });
    for (let k = 0; k < 9; k++) rect(g2, x0 + 80 + k * 12, ly - 6, 12, 12, { fill: div((k - 4) / 4, 1), alpha: 1 });
    g2.label('−', P(x0 + 74, ly), { color: 'mag', size: 12, font: 'ui', anchor: 'right' });
    g2.label('+', P(x0 + 80 + 9 * 12 + 6, ly), { color: 'teal', size: 12, font: 'ui', anchor: 'left' });
  }

  /* ---------------- feature space (3D) ---------------- */
  function drawFeature(g, s, m, ctx, r, inPlay) {
    const wf = m.wf, L = [3.0, 3.0, 5.0];
    // floor grid + axes from the origin
    for (let i = 0; i <= 3; i++) { g.line([i, 0, 0], [i, 3, 0], { color: 'grid', width: 1, alpha: 0.5, back: true }); g.line([0, i, 0], [3, i, 0], { color: 'grid', width: 1, alpha: 0.5, back: true }); }
    const names = ['x₁', 'x₂', 'y'], cols = ['j', 'k', 'b'];
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((d, i) => { const tip = N.scale(d, L[i]); g.vector([0, 0, 0], tip, { color: 'gridAxis', width: 1.4, head: 8, alpha: 0.9 }); g.label(names[i], tip, { color: cols[i], size: 14, font: 'math', dx: 10, dy: -8 }); });
    // the model plane over the floor rectangle
    if (s.showPlane) {
      const c = [[0, 0], [2.8, 0], [2.8, 2.8], [0, 2.8]].map(q => [q[0], q[1], planeZ(wf, q[0], q[1])]);
      g.polygon(c, { fill: 'gold', alpha: s.planeGold ? 0.24 : 0.13, stroke: 'gold', width: s.planeGold ? 1.6 : 1, subdiv: 4 });
      g.label(s.planeGold ? 'ŷ = Xŵ (least squares)' : (s.hasOnes ? 'a candidate plane' : 'best plane through the origin'), c[3], { color: 'gold', size: 12, font: 'ui', anchor: 'right', dx: -8, dy: 18, alpha: s.planeGold ? 1 : 0.7 });
      if (s.noOnesGhost) {
        const w0 = fit(s, { hasOnes: false }).full;
        const c0 = [[0, 0], [2.8, 0], [2.8, 2.8], [0, 2.8]].map(q => [q[0], q[1], planeZ(w0, q[0], q[1])]);
        g.polyline(c0, { color: 'bad', width: 1.2, dashed: true, closed: true, alpha: 0.7 });
      }
    }
    // sticks and points
    s.data.forEach((d, i) => {
      const p = [d.x1, d.x2, m.y[i]], q = [d.x1, d.x2, m.yhat[i]];
      if (s.showSticks && Math.abs(m.e[i]) > 1e-4) g.line(p, q, { color: 'residual', width: 1.3, dashed: [4, 3], alpha: 0.95 });
      g.line([d.x1, d.x2, 0], [d.x1, d.x2, Math.min(m.y[i], m.yhat[i])], { color: 'ghost', width: 0.8, alpha: 0.5 });
    });
    g.points(s.data.map((d, i) => [d.x1, d.x2, m.y[i]]), { r: 4.2, color: 'white' });
    if (s.perturb > 0.01) g.marker.ring([s.data[PERT_I].x1, s.data[PERT_I].x2, m.y[PERT_I]], { r: 10, color: 'b' });
    // the origin claim (misconception)
    if (s.cross === 'origin') {
      g.point([0, 0, 0], { r: 5, color: 'bad' });
      g.marker.cross([0, 0, 0], { size: 18, dx: -22, dy: 6 });
      g.label('no intercept: plane through (0, 0, 0)', [0, 0, 0], { color: 'bad', size: 11.5, font: 'ui', anchor: 'right', dx: -36, dy: 22 });
    }
    if (s.check) { g.marker.check([0, 0, planeZ(wf, 0, 0)], { size: 18, dx: -18 }); g.label('w₀ lets the plane float', [0, 0, planeZ(wf, 0, 0)], { color: 'ok', size: 11.5, font: 'ui', anchor: 'right', dx: -30, dy: -12 }); }
    // drag the prices (playground)
    if (inPlay) {
      const f = LA.cam.forward(s.cam), nrm = [f[0], f[1], 0];
      s.data.forEach((d, i) => g.handle('pt' + i, [d.x1, d.x2, m.y[i]], (p, st) => { st.data[i].y = +LA.clamp(p[2], 0, 7).toFixed(2); }, { color: 'white', r: 10, plane: { point: [d.x1, d.x2, 0], normal: nrm } }));
    }
    g.label('feature space: axes = features + y', [r.x + 10, r.y + r.h - 14], { screen: true, anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
  }

  /* ---------------- weight space, 2D (wide X = [2 1], y = 4) ---------------- */
  function drawWide2(g, s, ctx) {
    g.grid(null, { alpha: 0.45 });
    g.axes({});
    const b = g.P.bounds();
    g.label('w₁', [b.x1, 0], { color: 'hud', size: 15, font: 'math', anchor: 'right', dx: -6, dy: -14 });
    g.label('w₂', [0, b.y1], { color: 'hud', size: 15, font: 'math', anchor: 'left', dx: 10, dy: 14 });
    // the line of perfect fits
    g.lineThrough([2, 0], [-1, 2], { color: 'gold', width: 2.6, len: 12 });
    g.label('2w₁ + w₂ = 4: every point fits perfectly', [0.25, 3.5], { color: 'gold', size: 12.5, font: 'ui', anchor: 'left', dx: 12 });
    if (s.showRow) { g.lineThrough([0, 0], WIDE.row, { color: 'muted', width: 1, dashed: [3, 5], alpha: 0.6, len: 12 }); g.label('row direction [2, 1]', [3.2, 1.6], { color: 'muted', size: 11.5, font: 'ui', anchor: 'right', dy: -14 }); }
    if (s.showFoot && s.footT > 0) {
      const f = lerpV([0, 0], WIDE.foot, s.footT);
      g.line([0, 0], f, { color: 'residual', width: 1.6, dashed: true });
      if (s.footT >= 1) {
        g.arc(WIDE.foot, [-2, -1], [-1, 2], { r: 0.22, right: true, color: 'hud' });
        g.point(WIDE.foot, { r: 6, color: 'mag' });
        g.label('X⁺y (min-norm)', WIDE.foot, { color: 'mag', size: 13, font: 'ui', anchor: 'left', dx: 12, dy: 18 });
      }
    }
    // gradient descent from zero (mlTie)
    if (s.gdK > 0) {
      let w = [0, 0]; const eta = 0.08, K = Math.floor(s.gdK);
      for (let k = 0; k < K; k++) { const r = 4 - (2 * w[0] + w[1]); const w2 = [w[0] + eta * r * 2, w[1] + eta * r]; g.vector(w, w2, { color: 'w', width: 2.4, head: 9, label: k === 0 ? 'GD from w = 0' : null, labelAt: 0.5, labelSize: 12, labelFont: 'ui', dx: -10, dy: -16 }); w = w2; }
      if (s.gd2) {
        const st = [-0.4, 0.7]; let v = st.slice();
        for (let k = 0; k < 8; k++) { const r = 4 - (2 * v[0] + v[1]); const v2 = [v[0] + eta * r * 2, v[1] + eta * r]; g.vector(v, v2, { color: 'residual', width: 2, head: 8 }); v = v2; }
        g.point(st, { r: 4, color: 'residual' });
        g.label('GD from a random start', st, { color: 'muted', size: 11.5, font: 'ui', anchor: 'left', dx: 2, dy: 16 });
        g.point(v, { r: 4.5, color: 'residual' });
        g.label('lands elsewhere: |w| = ' + LA.fmt(N.norm(v)), v, { color: 'muted', size: 11.5, font: 'ui', anchor: 'right', dx: -10, dy: -12 });
      }
    }
    // the draggable w (on the line)
    if (!s.gdK && !s.showEntry) {
      g.vector([0, 0], s.wPoint, { color: 'muted', width: 2, head: 9, label: 'w', labelAt: 0.55 });
      g.point(s.wPoint, { r: 5.5, color: 'v' });
      g.hud('|w| = ' + LA.fmt(N.norm(s.wPoint)), g.size.w - 12, 38, { anchor: 'right' });
      g.handle('w', s.wPoint, (p, st) => { const k = (2 * p[0] + p[1] - 4) / 5; st.wPoint = [+(p[0] - 2 * k).toFixed(3), +(p[1] - k).toFixed(3)]; }, { color: 'v' });
      // keep the magenta min-norm point visible when w sits on it
      if (s.showFoot && s.footT >= 1 && N.norm(N.sub(s.wPoint, WIDE.foot)) < 0.08) g.point(WIDE.foot, { r: 3.5, color: 'mag' });
    }
    if (s.showEntry) { const q = N.scale(s.pinvEntry, 4); g.point(q, { r: 5.5, color: 'muted', label: 'your X⁺·y', labelColor: 'muted', dx: 12, dy: 14 }); if (N.norm(N.sub(q, WIDE.foot)) < 0.25) g.marker.ring(q, { r: 11, color: 'ok' }); }
    if (s.wideReadout) g.hud('residual 2w₁ + w₂ − 4 = ' + LA.fmt(Math.abs(2 * s.wPoint[0] + s.wPoint[1] - 4) < 0.005 ? 0 : 2 * s.wPoint[0] + s.wPoint[1] - 4), g.size.w - 12, 18, { anchor: 'right' });
    g.hud('weight space (wide X = [2 1], y = 4)', 12, g.size.h - 16, { color: 'muted', size: 11.5 });
  }

  /* ---------------- weight space, 3D ([1 2 2]·w = 6) ---------------- */
  function drawWide3(g, s) {
    const L = 3;
    for (let i = 0; i <= 3; i++) { g.line([i, 0, 0], [i, 3, 0], { color: 'grid', width: 1, alpha: 0.45, back: true }); g.line([0, i, 0], [3, i, 0], { color: 'grid', width: 1, alpha: 0.45, back: true }); }
    ['w₁', 'w₂', 'w₃'].forEach((nm, i) => { const d = [0, 0, 0]; d[i] = L; g.vector([0, 0, 0], d, { color: 'gridAxis', width: 1.4, head: 8 }); g.label(nm, d, { color: 'hud', size: 14, font: 'math', dx: 8, dy: -8 }); });
    const a = s.w3T;
    g.planes([{ point: W3.foot, normal: W3.row, extent: 1.9, color: 'gold', alpha: 0.2, stroke: 'gold' }]);
    // a few exact solutions scattered on the plane
    const u = N.normalize([2, -1, 0]), v = N.normalize(N.cross(W3.row, u));
    [[-1.3, 0.6], [1.1, 1.0], [0.9, -1.2], [-1.0, -1.1], [1.6, -0.2]].forEach((q, i) => { const p = N.add(W3.foot, N.add(N.scale(u, q[0]), N.scale(v, q[1]))); g.point(p, { r: 4, color: 'residual' }); if (i === 1) g.label('all fit perfectly', p, { color: 'muted', size: 11.5, font: 'ui', dx: 10, dy: -12 }); });
    if (a > 0) {
      g.line([0, 0, 0], lerpV([0, 0, 0], W3.foot, a), { color: 'residual', width: 1.6, dashed: true });
      if (a >= 1) { g.point(W3.foot, { r: 6, color: 'mag', label: 'X⁺y (min-norm)', labelColor: 'mag', dx: 14, dy: 12 }); g.vector([0, 0, 0], N.scale(N.normalize(W3.row), 0.9), { color: 'muted', width: 1.6, head: 7, label: 'row', labelSize: 11.5, labelFont: 'ui' }); }
    }
    g.hud('wide X: n < d (1 equation, 3 weights)', 12, 18, { color: 'teal' });
    g.hud('weight space: plane [1 2 2]·w = 6', 12, g.size.h - 16, { color: 'muted', size: 11.5 });
  }

  /* ---------------- the (w₁, w₃) inset: redundant / near-redundant weights, or the wide recap ---------------- */
  function drawInset(g, s, ir) {
    const hud = (g2, text, x, y, o) => g2.label(text, [ir.x + x, ir.y + y], Object.assign({ screen: true, anchor: 'left', color: 'muted', size: 11, font: 'ui' }, o || {}));
    const names = (g2, a, b) => { hud(g2, a, ir.w - 8, ir.h - 12, { anchor: 'right', color: 'hud', font: 'math', size: 13 }); hud(g2, b, 8, 12, { color: 'hud', font: 'math', size: 13 }); };
    if (s.inset === 'wide') {
      g.viewport(ir, { view: '2d', cam: { center: [1.6, 1.7], halfH: 2.5 }, bg: 'panel' }, (g2) => {
        g2.axes({ ticks: false, labels: false });
        g2.lineThrough([2, 0], [-1, 2], { color: 'gold', width: 2 });
        g2.line([0, 0], WIDE.foot, { color: 'residual', dashed: true });
        g2.point(WIDE.foot, { r: 4.5, color: 'mag' });
        hud(g2, 'wide X: min-norm foot', ir.w - 8, 12, { anchor: 'right', color: 'mag' });
        names(g2, 'w₁', 'w₂');
      });
      return;
    }
    const a = fit(s, { perturb: 0 }).full, b = s.inset === 'near' ? fit(s, { perturb: 1 }).full : a, bNow = fit(s).full;
    const c = a[1] + a[3];
    const ctr = s.inset === 'dup' ? [c / 2, c / 2] : [(a[1] + b[1]) / 2, (a[3] + b[3]) / 2];
    g.viewport(ir, { view: '2d', cam: { center: ctr, halfH: s.inset === 'dup' ? 1.3 : 1.6 }, bg: 'panel' }, (g2) => {
      g2.axes({ ticks: false, labels: false });
      if (s.inset === 'dup') {
        g2.lineThrough([c / 2, c / 2], [1, -1], { color: 'gold', width: 2 });
        g2.point([c / 2, c / 2], { r: 4.5, color: 'mag' });
        hud(g2, 'w₁ + w₃ = ' + LA.fmt(c) + ': all equivalent', ir.w - 8, 12, { anchor: 'right', color: 'gold' });
        g2.label('min-norm split', [c / 2, c / 2], { color: 'mag', size: 11, font: 'ui', anchor: 'left', dx: 9, dy: 9 });
      } else {
        // a level set of the SSE in the (w₁, w₃) plane: a long thin ellipse along the near-null direction
        const f = fit(s, { perturb: 0 }); const H = N.matMul(N.transpose(f.X), f.X);
        const e = N.eigSym([[H[1][1], H[1][3]], [H[3][1], H[3][3]]]);
        const k = 1.9 * Math.sqrt(Math.max(e.values[1], 1e-12)), Q = N.fromCols(e.vectors), M = N.matMul(Q, N.diag(e.values.map(l => k / Math.sqrt(Math.max(l, 1e-12)))));
        g2.ellipse(M, [a[1], a[3]], { stroke: 'teal', width: 1.4, fill: 'teal', alpha: 0.12 });
        g2.point([a[1], a[3]], { r: 4, color: 'residual' });
        g2.point([bNow[1], bNow[3]], { r: 5, color: 'mag' });
        const dw = N.norm(N.sub(bNow, a));
        if (dw > 0.05) g2.dim([a[1], a[3]], [bNow[1], bNow[3]], { label: 'Δw = ' + LA.fmt(dw), color: 'bad', offset: 14 });
        hud(g2, 'near-null direction', ir.w - 8, 12, { anchor: 'right', color: 'teal' });
      }
      names(g2, 'w₁', 'w₃');
    });
    g.label('(w₁, w₃) weights', [ir.x + 6, ir.y + ir.h + 12], { screen: true, anchor: 'left', color: 'muted', size: 11, font: 'ui' });
  }
})();
