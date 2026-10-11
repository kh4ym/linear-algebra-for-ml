/* p3-05 — Least squares: fitting is projecting (capstone of Part 3).
   Two rooms: the 2D data plot (axes = features) and the 3D sample space (axes = data points), drawn side by side
   with g.viewport. The best line is the perpendicular shadow of y on the plane spanned by the columns of X. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, C = LA.color;
  const PTS3 = [[1, 1], [2, 2], [3, 2]];
  const ONES = '<span class="c-i">ones</span>', XCOL = '<span class="c-j">x</span>', YV = '<span class="c-b">y</span>', YHAT = '<span class="c-mag">ŷ</span>';

  // synthetic 12-point data for the n-D honesty step (seed 35, as the storyboard asks; w/b chosen to fit the frame)
  function makePts12() {
    const d = LA.rng.linearData(35, 12, { w: [0.55], b: 0.6, noise: 0.3, xRange: [0.3, 3.7] });
    return d.x.map((x, i) => [+x.toFixed(3), +d.y[i].toFixed(3)]);
  }
  // eye on the −n side of the column plane (n ∝ [1,−2,1]) so y sits in front of the translucent plane; the view
  // direction is roughly perpendicular to the cluster ones / x / y (all within ~25° of each other) so they fan out,
  // and ~45° off the plane so both the teal sheet and the perpendicular residual read clearly
  function cam3dDefault() { return { target: [0.85, 1.4, 1.6], dist: 6.9, yaw: 2.4, pitch: 0.28 }; }
  const AXLEN = [3.2, 2.25, 3.3];   // axis lengths per data point (point 2 points toward the eye, so it is drawn shorter and stays in frame)
  const CAM_FAR = { dist: 20, target: [1.0, 2.0, 4.3] };

  /** Everything derived from the state: design matrix, targets, fit, residual. */
  function model(s) {
    const pts = s.n12 ? s.pts12 : s.pts;
    const X = pts.map(p => [1, p[0]]), y = pts.map(p => p[1]);
    const what = N.lstsq(X, y);
    const w = s.autoFit ? what : s.w;
    const yhat = N.matVec(X, w), e = N.sub(y, yhat);
    const ones = X.map(r => r[0]), xs = X.map(r => r[1]);
    const ybar = N.mean1(y);
    const m = { pts, X, y, what, w, yhat, e, ones, xs, ybar, sse: N.dot(e, e) };
    m.sst = y.reduce((a, v) => a + (v - ybar) * (v - ybar), 0);
    m.XtX = N.matMul(N.transpose(X), X); m.Xty = N.matVec(N.transpose(X), y);
    return m;
  }
  /** Exact quadratic through the three default points (mlTie step): y = a + bx + cx². */
  function quadFit(pts) { const X = pts.map(p => [1, p[0], p[0] * p[0]]); return N.lstsq(X, pts.map(p => p[1])); }
  /** Total-least-squares line (perpendicular distances) for the misconception step: {p0, dir}. */
  function tlsLine(pts) { const e = N.eigSym(N.cov(pts)); return { p0: N.mean(pts), dir: e.vectors[0] }; }

  /** Layout rectangles (px) for the two rooms and the bracket panel. */
  function rects(size) {
    const W = size.w, H = size.h;
    if (W / H > 1.25) return { plot: { x: 0, y: 0, w: 0.42 * W, h: 0.70 * H }, tab: { x: 0, y: 0.70 * H, w: 0.42 * W, h: 0.30 * H }, space: { x: 0.42 * W, y: 0, w: 0.58 * W, h: H } };
    return { plot: { x: 0, y: 0, w: 0.64 * W, h: 0.40 * H }, tab: { x: 0.64 * W, y: 0, w: 0.36 * W, h: 0.40 * H }, space: { x: 0, y: 0.40 * H, w: W, h: 0.60 * H } };
  }
  /** A 2D viewport in px coordinates: world (x, −y) ↔ px (rect.x + x, rect.y + y). */
  function panel(g, r, opts, fn) {
    g.viewport({ x: r.x, y: r.y, w: Math.max(2, r.w), h: Math.max(2, r.h) }, Object.assign({ view: '2d', cam: { center: [r.w / 2, -r.h / 2], halfH: r.h / 2 }, border: false }, opts || {}), (g2, rr) => fn(g2, rr));
  }
  const P = (x, y) => [x, -y];
  function bracket(g2, x, y0, y1, open, color, alpha) {
    const d = open ? 6 : -6, o = { color: color || 'hud', width: 1.4, alpha: alpha === undefined ? 0.9 : alpha };
    g2.line(P(x, y0), P(x, y1), o); g2.line(P(x, y0), P(x + d, y0), o); g2.line(P(x, y1), P(x + d, y1), o);
  }
  /** The data plot's camera: s.cam2, widened vertically when dragged / preset points leave its frame. */
  function plotCam(s, m, rect) {
    const ys = m.pts.map(p => p[1]);
    const lo = Math.min(s.cam2.center[1] - s.cam2.halfH, Math.min(...ys) - 0.4), hi = Math.max(s.cam2.center[1] + s.cam2.halfH, Math.max(...ys) + 0.4);
    const halfH = Math.max((hi - lo) / 2, rect ? 2.25 * rect.h / rect.w : 0);   // x from about −0.25 to 4.25 even in a tall panel
    return { center: [s.cam2.center[0], (lo + hi) / 2], halfH };
  }
  /** Centre of the drawn patch of the column plane: covers the origin, both columns and the foot of y. */
  function planeCenter(ones, xs) {
    const u = N.normalize(ones), vr = N.sub(xs, N.scale(u, N.dot(xs, u))), nv = N.norm(vr);
    const v = nv > 1e-6 ? N.scale(vr, 1 / nv) : [0, 0, 0];
    return N.add(N.scale(u, 1.75), N.scale(v, 0.5));
  }
  /** Right-angle corner mark at `corner` between unit directions u and v (3D or 2D). */
  function rightAngle(g, corner, u, v, size, color) {
    const a = N.scale(N.normalize(u), size), b = N.scale(N.normalize(v), size);
    g.polyline([N.add(corner, a), N.add(N.add(corner, a), b), N.add(corner, b)], { color: color || 'hud', width: 1.3, alpha: 0.9, segmented: true });
  }

  LA.registerLesson({
    id: 'p3-05-least-squares',
    part: 3,
    title: 'Least squares: fitting is projecting',
    subtitle: 'The data’s y-values become one vector; the best line is its shadow on the column plane.',
    minutes: 13,
    prereqs: ['p3-03-projections', 'p3-01-systems-row-column', 'p1-07-data-as-vectors'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D (data) + 3D (sample space, n = 3)', realSize: 'n = 10⁵ rows means y ∈ ℝ¹⁰⁰⁰⁰⁰ and a d-dimensional plane inside it', transfers: '“residual ⟂ every column” and XᵀX ŵ = Xᵀy are exact for any n and d; only the ambient space becomes undrawable' },
    cantShow: ['sample space for n > 3: with 12 points y lives in ℝ¹², which has no picture (we switch to a strip and keep the formula)', 'both rooms as one picture: feature space (axes = features) and sample space (axes = data points) are different rooms; the lesson walks between them', 'statistical noise models (why squared error): this lesson is geometry only'],
    terms: [
      { term: 'least squares', def: 'Choose the line that makes the sum of squared vertical errors smallest.' },
      { term: 'residual', def: 'The vertical gap yᵢ − ŷᵢ between a data point and the line; collected into one vector, y − ŷ.' },
      { term: 'sample space', def: 'The space whose axes are the data points, where the whole target column y is one arrow.' },
      { term: 'design matrix X', def: 'One row per data point, one column per feature, plus a column of ones for the intercept.' },
      { term: 'fitted values ŷ', def: 'The predictions Xw, a point on the plane spanned by the columns of X.' },
      { term: 'normal equations', def: 'XᵀX ŵ = Xᵀy: “the residual is perpendicular to every column” written out.' },
    ],
    atlas: ['linear-regression'],
    thumbnail: 6,

    state() {
      return {
        view: '2d',
        cam: Object.assign(LA.cam.defaults3d(), { center: [1.75, 1.45], halfH: 2.3 }, cam3dDefault()),
        cam2: { center: [2, 1.28], halfH: 1.68 },
        pts: LA.clone(PTS3), pts12: makePts12(), n12: false,
        w: [0.5, 0.4], autoFit: false,
        layout: '2d', mathMode: 'sse',
        showSquares: true, sqT: 1, showCoords: true,
        perpAlpha: 0, vertAlpha: 1, cross: false, check: false, tls: false,
        flyY: 0, flyX: 0, tabY: 0, tabX: 0, axesT: 0, yT: 0, colsT: 0, yGuide: 0, guideA: 1, legHL: -1,
        showPlane: false, planeT: 0, showYhat: false, recipe: false, yhatT: 1,
        showE: false, eT: 1, compT: 0, pulse: -1,
        showBars: false, rightAngle: false, lineGold: false, fillT: 0, showWhat: false, lineLabel: false,
        showR2: false, r2T: 0, stripT: 1, showX2: false, x2T: 0, boxT: 0, exact: false,
        lineHandles: false, yhatHandle: false, ptHandles: 'none',
        XtXEntry: [[0, 0], [0, 0]], showEntry: false,
      };
    },

    render(g, s, ctx) {
      const m = model(s);
      const lay = s.layout === '2d' ? '2d' : (s.n12 ? 'strip' : s.layout);
      const inPlay = ctx.mode === 'play';
      if (lay === '2d') { drawPlot(g, s, m, ctx, null, inPlay); return; }
      const R = rects(g.size);
      let ptScreen = null, cellY = null, cellOnes = null, cellX = null;
      g.viewport({ x: R.plot.x, y: R.plot.y, w: R.plot.w, h: R.plot.h }, { view: '2d', cam: plotCam(s, m, R.plot) }, (g2, r) => { ptScreen = drawPlot(g2, s, m, ctx, r, inPlay); });
      panel(g, R.tab, {}, (g2, r) => { const c = drawTable(g2, s, m, r); cellY = c.y; cellOnes = c.ones; cellX = c.x; });
      if (lay === 'strip') panel(g, R.space, { border: true }, (g2, r) => drawCells(g2, s, m, r));
      else {
        // the default framing is tuned for a landscape room; back off in a portrait one so nothing leaves the sides
        const f = Math.max(1, Math.sqrt(1.45 * R.space.h / R.space.w));
        const cam = f > 1.001 ? Object.assign({}, s.cam, { dist: s.cam.dist * f }) : s.cam;
        g.viewport({ x: R.space.x, y: R.space.y, w: R.space.w, h: R.space.h }, { view: '3d', cam }, (g2, r) => drawSpace(g2, s, m, ctx, r, inPlay));
      }
      // flights of numbers from the data points into the brackets (drawn unclipped, on top of both rooms)
      if (ptScreen && s.flyY > 0 && s.flyY < 1) m.pts.forEach((p, i) => { const q = lerpV(ptScreen[i], cellY[i], s.flyY); g.label(LA.fmt(p[1]), q, { screen: true, color: 'b', size: 15, bold: true }); });
      if (ptScreen && s.flyX > 0 && s.flyX < 1) m.pts.forEach((p, i) => {
        g.label(LA.fmt(p[0]), lerpV([ptScreen[i][0], ptScreen[i][1] + 18], cellX[i], s.flyX), { screen: true, color: 'j', size: 15, bold: true });
        g.label('1', lerpV([ptScreen[i][0], ptScreen[i][1] - 18], cellOnes[i], s.flyX), { screen: true, color: 'i', size: 15, bold: true });
      });
    },

    math(s, tex) {
      const m = model(s), w = m.w;
      const sym = (n, o) => tex.sym(n, o), op = tex.op, num = tex.num, mat = tex.mat, vec = tex.vec, eq = tex.eq;
      const what = sym('w', { hat: true, color: 'gold' }), yv = sym('y', { color: 'b' }), yh = sym('y', { hat: true, color: 'mag' }), ev = sym('e', { color: 'muted' });
      switch (s.mathMode) {
        case 'perp': return tex.block(eq(tex.text('error'), op('='), sym('y', { sub: 'i', color: 'b' }), op('−'), sym('ŷ', { sub: 'i', color: 'mag' }), tex.text('  (vertical, never perpendicular to the line)', { color: 'muted' })));
        case 'y': return tex.block(eq(yv, op('='), vec(m.y, { color: 'b' })), tex.text('rows: point 1, point 2, point 3', { color: 'muted' }));
        case 'X': return tex.block(eq(sym('X'), op('='), mat(m.X, { colors: 'cols' })), eq(sym('X'), sym('w'), op('='), sym('w', { sub: '0' }), op('·'), sym('ones', { color: 'i' }), op('+'), sym('w', { sub: '1' }), op('·'), sym('x', { color: 'j' })));
        case 'yhat': return tex.block(eq(yh, op('='), sym('X'), sym('w'), op('='), num(w[0]), vec(m.ones, { color: 'i' }), op('+'), num(w[1]), vec(m.xs, { color: 'j' }), op('='), vec(m.yhat, { color: 'mag' })));
        case 'e': return tex.block(eq(tex.norm(ev) + '<sup>2</sup>', op('='), num(m.e[0] * m.e[0]), op('+'), num(m.e[1] * m.e[1]), op('+'), num(m.e[2] * m.e[2]), op('='), num(m.sse, { color: 'gold' })), eq(tex.text('= SSE: the same three gaps, squared and summed', { color: 'muted' })));
        case 'dots': return tex.block(eq(ev, op('·'), sym('ones', { color: 'i' }), op('='), num(N.dot(m.e, m.ones))), eq(ev, op('·'), sym('x', { color: 'j' }), op('='), num(N.dot(m.e, m.xs))), eq(tex.text('SSE'), op('='), num(m.sse, { color: 'gold' })));
        case 'normal': return tex.block(eq(tex.T(sym('X')), sym('X'), what, op('='), tex.T(sym('X')), yv), eq(mat(m.XtX, { colors: 'cols' }), what, op('='), vec(m.Xty, { color: 'b' })), eq(what, op('='), vec(m.what, { color: 'gold' })));
        case 'r2': return tex.block(eq(sym('R', { sup: '2' }), op('='), op('1 −'), tex.frac(tex.text('SSE'), tex.text('SST')), op('='), op('1 −'), tex.frac(num(m.sse), num(m.sst)), op('='), num(m.sst > 0 ? 1 - m.sse / m.sst : 0, { color: 'gold' })));
        case 'n12': return tex.block(eq(tex.T(sym('X')), sym('X'), op('='), mat(m.XtX, { colors: 'cols' }), tex.dim(2, 2)), eq(what, op('='), vec(m.what, { color: 'gold' })), eq(tex.text('X is 12×2, y has 12 cells', { color: 'muted' })));
        case 'x2': return tex.block(eq(sym('X'), op('='), op('['), sym('ones', { color: 'i' }), sym('x', { color: 'j' }), sym('x', { sup: '2', color: 'k' }), op(']')), eq(tex.text('SSE'), op('='), num(s.exact ? 0 : m.sse, { color: 'gold' }), tex.text(s.exact ? '  — exact fit, nothing left to explain' : '', { color: 'muted' })));
        case 'recap': return tex.block(eq(what, op('='), tex.inv(tex.paren(tex.T(sym('X')) + sym('X'))), tex.T(sym('X')), yv, op('='), vec(m.what, { color: 'gold' })));
        default: return tex.block(eq(sym('ŷ', { sub: 'i', color: 'mag' }), op('='), sym('w', { sub: '0' }), op('+'), sym('w', { sub: '1' }), sym('x', { sub: 'i' })), eq(tex.text('SSE'), op('='), tex.sum(tex.paren(sym('y', { sub: 'i', color: 'b' }) + op('−') + sym('ŷ', { sub: 'i', color: 'mag' })) + '<sup>2</sup>', 'i'), op('='), num(m.sse, { color: 'gold' })));
      }
    },

    steps: [
      { title: 'Three points, one candidate line', duration: 3.5, interactive: true,
        narration: 'Three measurements: (1, 1), (2, 2), (3, 2). We want a line y = w₀ + w₁x that comes closest to all three. Any line misses some points; the vertical gaps are the <b>residuals</b>. Drag the gold line by its two handles and watch the gaps and their squares.',
        setup(s) { s.layout = '2d'; s.view = '2d'; s.cam.center = [1.75, 1.45]; s.cam.halfH = 2.3; s.w = [0.5, 0.4]; s.showSquares = true; s.showCoords = true; s.lineHandles = true; s.mathMode = 'sse'; },
        at(s, t) { s.sqT = seg(t, 0.25, 0.85, 'back'); } },

      { title: 'Closest is not “perpendicular to the line”', duration: 3.5, misconception: true,
        narration: '✕ “Measure the error perpendicular to the line.” The error of a prediction is in y, the thing we predict, so the gap is vertical. Perpendicular distances pick a different, wrong line here, the dim gray one. Least squares always means vertical gaps, squared, summed.',
        setup(s) { s.showCoords = false; s.lineHandles = false; s.mathMode = 'perp'; s.sqT = 1; s.showSquares = false; },
        at(s, t) {
          s.perpAlpha = t < 0.55 ? seg(t, 0, 0.3) : 0.35;
          s.tls = t > 0.15; s.cross = t > 0.3;
          s.vertAlpha = t < 0.5 ? 0.12 : Math.max(0.12, seg(t, 0.55, 0.8));
          s.check = t > 0.7;
        } },

      { title: 'The flip: three numbers become one arrow', duration: 6.5,
        narration: 'Here is the move that makes everything geometric. Stack the three y-values, 1, 2, 2, into one tall bracket and read it as a single arrow ' + YV + ' in a new 3D room, <b>sample space</b>, with one axis per data point (not x and y). Walk 1 along the point 1 axis, 2 along point 2, 2 up point 3: each point’s height is one coordinate.',
        setup(s) { s.perpAlpha = 0; s.tls = false; s.cross = false; s.check = false; s.vertAlpha = 1; s.showSquares = true; s.layout = 'stacked'; s.view = '3d'; Object.assign(s.cam, cam3dDefault()); s.mathMode = 'y'; s.flyY = 0; s.tabY = 0; s.axesT = 0; s.yT = 0; s.yGuide = 0; s.guideA = 1; s.legHL = -1; },
        at(s, t) {
          s.flyY = seg(t, 0.03, 0.28); s.tabY = t >= 0.28 ? 1 : 0; s.axesT = seg(t, 0.28, 0.42);
          // walk the three legs one at a time, highlighting the matching data point and table row
          const legs = [[0.42, 0.56], [0.57, 0.71], [0.72, 0.86]];
          s.yGuide = legs.reduce((a, l) => a + seg(t, l[0], l[1]), 0);
          s.legHL = legs.findIndex(l => t >= l[0] - 0.01 && t < l[1] + 0.01);
          s.yT = seg(t, 0.84, 1, 'out');
          LA.cam.tween(s, { yaw: cam3dDefault().yaw - 0.25 }, { yaw: cam3dDefault().yaw }, seg(t, 0.3, 1));
        } },

      { title: 'The columns of X are two arrows in that same room', duration: 4,
        narration: 'Write the line’s ingredients the same way. The intercept adds w₀ to every point: a column of ones, ' + ONES + ' = [1; 1; 1]. The slope adds w₁xᵢ: the column ' + XCOL + ' = [1; 2; 3]. Side by side they form the <b>design matrix X</b>: one row per point, one column per feature.',
        setup(s) { s.flyY = 1; s.tabY = 1; s.axesT = 1; s.yT = 1; s.yGuide = 3; s.legHL = -1; s.mathMode = 'X'; s.flyX = 0; s.tabX = 0; s.colsT = 0; },
        at(s, t) { s.guideA = 1 - seg(t, 0, 0.3); s.flyX = seg(t, 0.05, 0.45); s.tabX = t >= 0.45 ? 1 : 0; s.colsT = seg(t, 0.45, 0.9, 'back'); } },

      { title: 'Every line is a point on a plane', duration: 4.5, interactive: true,
        narration: 'A line (w₀, w₁) has three heights at x = 1, 2, 3, the magenta dots. Stacked, they form ' + YHAT + ' = w₀·' + ONES + ' + w₁·' + XCOL + ': a weighted sum of the two arrows, so a point on the teal plane they span. Drag the line’s handles in the data plot, or ' + YHAT + ' in sample space: the line tilts while ' + YHAT + ' slides around the plane.',
        setup(s) { s.flyX = 1; s.tabX = 1; s.colsT = 1; s.guideA = 0; s.showPlane = true; s.planeT = 0; s.showYhat = true; s.recipe = true; s.yhatT = 0; s.lineHandles = true; s.yhatHandle = true; s.mathMode = 'yhat'; },
        at(s, t) { s.planeT = seg(t, 0, 0.4); s.yhatT = seg(t, 0.3, 0.7, 'back'); } },

      { title: 'y is off the plane, and the residual is the gap', duration: 4,
        narration: YV + ' does not lie on the plane: no line passes through all three points. The dashed arrow from ' + YHAT + ' to ' + YV + ' is the <b>residual</b> vector e = y − ŷ, and its squared length is exactly the sum of squares from the data plot: the three gaps are its three components.',
        setup(s) { s.planeT = 1; s.yhatT = 1; s.recipe = false; s.lineHandles = false; s.yhatHandle = false; s.showE = true; s.eT = 0; s.compT = 0; s.pulse = -1; s.mathMode = 'e'; },
        at(s, t) { s.eT = seg(t, 0, 0.3); s.compT = seg(t, 0.3, 0.6); s.pulse = t > 0.6 ? Math.min(2, Math.floor(seg(t, 0.6, 1, 'linear') * 3)) : -1; } },

      { title: 'Shortest residual = perpendicular drop', duration: 4.5,
        narration: 'The shortest arrow from a point to a plane is the perpendicular one: the shadow from the projection lesson. So the best line is the one whose ' + YHAT + ' is the foot of the perpendicular from ' + YV + '. Watch w slide toward it and two dot products, e·' + ONES + ' and e·' + XCOL + ', fall to zero together.',
        setup(s) { s.eT = 1; s.compT = 0; s.pulse = -1; s._w0 = LA.clone(s.w); s._w1 = model(s).what; s.showBars = true; s.rightAngle = true; s.mathMode = 'dots'; s.lineGold = false; },
        at(s, t) { s.w = lerpV(s._w0, s._w1, seg(t, 0.1, 0.85)); s.lineGold = t > 0.9; } },

      { title: 'Written out: the normal equations', duration: 4,
        narration: '“e is perpendicular to both columns” is Xᵀ(y − Xŵ) = 0, which rearranges to the <b>normal equations</b> XᵀX ŵ = Xᵀy. Here XᵀX = [[3, 6], [6, 14]] and Xᵀy = [5; 11]: a 2×2 system, with solution ŵ = [0.67; 0.5]. Regression does not search; it drops a perpendicular.',
        setup(s) { s.w = model(s).what; s.lineGold = true; s.showBars = false; s.recipe = false; s.fillT = 0; s.showWhat = false; s.lineLabel = false; s.mathMode = 'normal'; },
        at(s, t) { s.fillT = seg(t, 0.05, 0.8, 'linear') * 6; s.showWhat = t > 0.85; s.lineLabel = t > 0.85; } },

      { title: 'How much did the plane explain? R²', duration: 3,
        narration: 'Center ' + YV + ' by subtracting its mean 1.67. Of its squared length 0.67, the shadow keeps 0.5 and the residual holds 0.17: a right triangle in sample space. The kept fraction, 0.75, is <b>R²</b>, the share of the variation in y that the line accounts for.',
        setup(s) { s.fillT = 6; s.showWhat = true; s.showR2 = true; s.r2T = 0; s.rightAngle = false; s.mathMode = 'r2'; },
        at(s, t) { s.r2T = seg(t, 0.1, 0.9); } },

      { title: 'Twelve points: the picture stops, the formula does not', duration: 4,
        narration: 'Twelve data points put ' + YV + ' in ℝ¹², which we cannot draw. The picture stops at 3. The formula does not: X is 12×2, y has 12 cells, XᵀX is still 2×2 and the shadow is still perpendicular to both columns. The plane is real; only our view of it is gone.',
        setup(s) { s.showR2 = false; s.n12 = true; s.autoFit = true; s.lineGold = true; s.lineLabel = false; s.stripT = 0; s.mathMode = 'n12'; },
        at(s, t) { s.stripT = seg(t, 0.15, 0.9, 'linear'); } },

      { title: 'Why ML cares: more features widen the plane', duration: 4.5, mlTie: true,
        narration: 'Add a feature and X gets a column: here <span class="c-k">x²</span> = [1; 4; 9]. The plane grows into a bigger subspace, so the shadow of ' + YV + ' can only get closer: training error never rises when you add features. With three columns for three points the fit is exact: memorized, not learned. Part 4 asks which directions in that plane are worth keeping.',
        setup(s) { s.n12 = false; s.autoFit = false; s.w = model(s).what; s.showX2 = true; s.x2T = 0; s.boxT = 0; s.exact = false; s.mathMode = 'x2'; s.stripT = 1; s.fillT = 0; s.showWhat = false; },
        at(s, t) { s.x2T = seg(t, 0.1, 0.5, 'back'); s.boxT = seg(t, 0.45, 0.8); s.exact = t > 0.8; LA.cam.tween(s, { dist: cam3dDefault().dist, target: cam3dDefault().target }, CAM_FAR, seg(t, 0, 0.6)); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Flip: the targets become one vector ' + YV + ' in sample space, axes = data points. Each line is a point ' + YHAT + ' on the plane spanned by the columns of X. The best fit is the perpendicular shadow, which written out is the normal equations XᵀX ŵ = Xᵀy.',
        cards: ['<b>y-values → one arrow.</b> Stack the targets; in sample space (one axis per data point) they are a single vector <span class="c-b">y</span>.', '<b>Every line is a point.</b> ŷ = w₀·ones + w₁·x lives on the teal plane spanned by the columns of X.', '<b>Best fit = perpendicular shadow.</b> e ⟂ every column, i.e. XᵀX ŵ = Xᵀy.'],
        setup(s) { s.showX2 = false; s.exact = false; s.x2T = 0; s.boxT = 0; s.fillT = 6; s.showWhat = true; s.w = model(s).what; s.recipe = false; s.lineLabel = true; s.rightAngle = true; s.mathMode = 'recap'; },
        at(s, t) { LA.cam.tween(s, CAM_FAR, { dist: cam3dDefault().dist, target: cam3dDefault().target }, seg(t, 0, 0.7)); } },
    ],

    playground: {
      intro: '<p>Drag the data points (top), the gold line’s handles, or the magenta <span class="c-mag">ŷ</span> on the plane (bottom). The dashed residual, the two dot-product bars and SSE update live. Switch to 12 points to see the strip view.</p>',
      setup(s) {
        s.layout = 'stacked'; s.view = '3d'; Object.assign(s.cam, cam3dDefault()); s.w = [0.5, 0.4]; s.autoFit = false; s.n12 = false;
        s.showSquares = true; s.sqT = 1; s.showCoords = false; s.flyY = 1; s.flyX = 1; s.tabY = 1; s.tabX = 1; s.axesT = 1; s.yT = 1; s.colsT = 1;
        s.showPlane = true; s.planeT = 1; s.showYhat = true; s.recipe = true; s.yhatT = 1; s.showE = true; s.eT = 1; s.showBars = true; s.rightAngle = true;
        s.lineHandles = true; s.yhatHandle = true; s.ptHandles = 'free'; s.fillT = 6; s.showWhat = true; s.mathMode = 'dots';
      },
      controls: [
        { type: 'slider', bind: 'w[0]', min: -2, max: 3, step: 0.01, label: 'w₀ (intercept)', color: 'gold' },
        { type: 'slider', bind: 'w[1]', min: -2, max: 2, step: 0.01, label: 'w₁ (slope)', color: 'gold' },
        { type: 'button', label: 'Snap to least squares', primary: true, onClick(s) { s.w = model(s).what; } },
        { type: 'toggle', bind: 'n12', label: '12 points (strip view, synthetic data)' },
        { type: 'toggle', bind: 'showSquares', label: 'Show the squares' },
        { type: 'preset', label: 'Data', items: [
          { label: 'Default', values: { pts: LA.clone(PTS3), n12: false } },
          { label: 'Exactly collinear', values: { pts: [[1, 1], [2, 2], [3, 3]], n12: false } },
          { label: 'Outlier', values: { pts: [[1, 1], [2, 2], [3, 5]], n12: false } },
          { label: 'Flat', values: { pts: [[1, 2], [2, 2], [3, 2]], n12: false } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Move the gold line (its two handles, or the magenta <span class="c-mag">ŷ</span> on the plane) until the residual is perpendicular to the plane: both dot products e·<span class="c-i">ones</span> and e·<span class="c-j">x</span> within 0.1 of zero.',
        setup(s) { playSetup(s); s.w = [0, 0]; s.showSquares = false; s.ptHandles = 'none'; s.fillT = 0; s.showWhat = false; s.mathMode = 'dots'; },
        judge(s) { const m = model(s); const d1 = Math.abs(N.dot(m.e, m.ones)), d2 = Math.abs(N.dot(m.e, m.xs)); const tol = 2 * LA.TOL.scalar; const worst = Math.max(d1, d2); return { ok: worst <= tol, progress: Math.max(0, 1 - worst / 4), hint: worst < 0.6 && worst > tol ? (d1 > d2 ? 'The intercept bar is still nonzero: nudge the left handle (w₀).' : 'The slope bar is still nonzero: nudge the right handle (w₁).') : undefined }; },
        hint: 'Watch which bar is still nonzero: the left handle moves the intercept, the right handle tilts the slope.',
        solution(s) { s.w = model(s).what; } },
      { kind: 'manipulate',
        prompt: 'Drag the third data point up or down until <span class="c-b">y</span> lands exactly on the plane (zero residual; the line is refitted every frame).',
        setup(s) { playSetup(s); s.autoFit = true; s.showSquares = false; s.lineHandles = false; s.yhatHandle = false; s.ptHandles = 'third'; s.recipe = false; s.fillT = 0; s.showWhat = false; s.mathMode = 'e'; },
        judge(s) { const m = model(s); const d = N.norm(m.e); return { ok: d <= LA.TOL.pos, progress: Math.max(0, 1 - d / 1.2) }; },
        hint: 'Three points on one straight line: where must the third one go?',
        solution(s) { s.pts[2] = [3, 3]; } },
      { kind: 'entry',
        prompt: 'Type XᵀX for the default data (entries are dot products of the columns <span class="c-i">ones</span> and <span class="c-j">x</span>; a cell turns green when it matches).',
        bind: 'XtXEntry', answer: [[3, 6], [6, 14]], tol: 2 * LA.TOL.entry, label: 'XᵀX', colors: 'cols', step: 1,
        setup(s) { playSetup(s); s.w = model(s).what; s.showSquares = false; s.lineHandles = false; s.yhatHandle = false; s.ptHandles = 'none'; s.recipe = false; s.showBars = false; s.XtXEntry = [[0, 0], [0, 0]]; s.showEntry = true; s.fillT = 6; s.showWhat = false; s.mathMode = 'X'; } },
    ],
  });

  /** Shared check setup: the stacked two-room scene at its default state. */
  function playSetup(s) {
    s.layout = 'stacked'; s.view = '3d'; Object.assign(s.cam, cam3dDefault()); s.pts = LA.clone(PTS3); s.w = [0.5, 0.4]; s.autoFit = false; s.n12 = false;
    s.showSquares = true; s.sqT = 1; s.showCoords = false; s.flyY = 1; s.flyX = 1; s.tabY = 1; s.tabX = 1; s.axesT = 1; s.yT = 1; s.colsT = 1;
    s.showPlane = true; s.planeT = 1; s.showYhat = true; s.recipe = true; s.yhatT = 1; s.showE = true; s.eT = 1; s.showBars = true; s.rightAngle = true;
    s.yGuide = 0; s.legHL = -1;
    s.lineHandles = true; s.yhatHandle = true; s.ptHandles = 'none'; s.showEntry = false; s.showR2 = false; s.showX2 = false; s.exact = false; s.lineGold = false; s.lineLabel = false;
  }

  /* ---------------- the data plot (feature space) ---------------- */
  function drawPlot(g, s, m, ctx, r, inPlay) {
    const b = g.P.bounds();
    g.axes({ names: ['x', 'y'] });
    const w = m.w, lineY = x => w[0] + w[1] * x;
    const opt = N.norm(N.sub(w, m.what)) < 0.02;
    const gold = s.lineGold || opt || s.autoFit;
    // the candidate line (dim gold until it is the least-squares line)
    if (!(s.exact && s.showX2)) g.line([b.x0, lineY(b.x0)], [b.x1, lineY(b.x1)], { color: 'gold', width: gold ? 2.6 : 2, alpha: gold ? 1 : 0.55 });
    if (s.tls) { const L = tlsLine(m.pts); g.lineThrough(L.p0, L.dir, { color: 'residual', width: 1.4, alpha: 0.6, dashed: [3, 5] }); g.label('perpendicular fit (wrong)', [L.p0[0] + 1.2, L.p0[1] + 1.2 * L.dir[1] / L.dir[0]], { color: 'muted', size: 11.5, anchor: 'below', dy: 4 }); }
    // quadratic through the three points (mlTie)
    if (s.showX2 && s.x2T > 0) {
      const q = quadFit(m.pts), pts = [];
      for (let x = b.x0; x <= b.x1 + 1e-9; x += 0.08) pts.push([x, q[0] + q[1] * x + q[2] * x * x]);
      g.polyline(pts, { color: 'gold', width: 2.6, alpha: 0.4 + 0.6 * s.x2T });
    }
    // residuals: vertical dashed gaps (and, in the misconception step, red perpendicular drops)
    m.pts.forEach((p, i) => {
      const yh = s.exact && s.showX2 ? p[1] : lineY(p[0]);
      const e = p[1] - yh;
      if (s.perpAlpha > 0 && !s.n12) {
        const d = N.normalize([1, w[1]]), rel = [p[0], p[1] - w[0]], foot = N.add([0, w[0]], N.scale(d, N.dot(rel, d)));
        const fp = lerpV(p, foot, s.perpAlpha > 0.34 ? 1 : s.perpAlpha / 0.34);
        g.line(p, fp, { color: 'bad', width: 1.6, dashed: true, alpha: Math.min(1, s.perpAlpha * 1.5) });
      }
      const va = s.vertAlpha === undefined ? 1 : s.vertAlpha;
      if (va > 0.01 && Math.abs(e) > 1e-6) {
        const hl = s.pulse === i;
        g.line(p, [p[0], yh], { color: hl ? 'white' : 'residual', width: hl ? 2.4 : 1.4, dashed: true, alpha: va });
        if (s.showSquares && !s.n12) {
          const a = Math.abs(e) * s.sqT, y0 = Math.min(p[1], yh), y1 = y0 + a;
          g.polygon([[p[0], y0], [p[0] + a, y0], [p[0] + a, y1], [p[0], y1]], { fill: 'residual', alpha: 0.16 * va, stroke: 'residual', width: 0.8 });
        }
        if ((s.compT > 0 || s.pulse >= 0) && !s.n12) g.label('e' + '₁₂₃'[i], [p[0], (p[1] + yh) / 2], { color: hl ? 'white' : 'muted', size: 12, anchor: 'left', dx: 7 });
      }
    });
    // two-room steps: the line's own three heights (the entries of ŷ) as magenta dots, and each point's index
    if (r && !s.n12 && s.showYhat && s.yhatT > 0 && !(s.exact && s.showX2)) g.points(m.pts.map(p => [p[0], lineY(p[0])]), { r: 4.2 * Math.min(1, s.yhatT), color: 'mag' });
    g.points(m.pts, { r: s.n12 ? 3.8 : 5, color: 'white' });
    if (r && !s.n12) m.pts.forEach((p, i) => {
      const hl = s.legHL === i;
      if (hl) g.marker.ring(p, { r: 12 });
      const up = p[1] >= lineY(p[0]) - 1e-9;   // label on the side away from the line, its gap and its square
      g.label('pt ' + (i + 1), p, { color: hl ? 'white' : 'muted', size: hl ? 13 : 11.5, bold: hl, font: 'ui', anchor: up ? 'above' : 'below', dy: up ? -9 : 9 });
    });
    if (s.showCoords && !s.n12) m.pts.forEach(p => g.label('(' + LA.fmt(p[0]) + ', ' + LA.fmt(p[1]) + ')', p, { color: 'muted', size: 12, anchor: 'above', dy: -8 }));
    if (s.flyX > 0 && s.flyX < 1 && !s.n12) { /* flying numbers are drawn by render() unclipped */ }
    if (s.cross && !s.n12) { const q = [2.25, lineY(2.25) - 0.42]; g.marker.cross(q, { size: 18, dx: -4 }); g.label('perpendicular gaps', q, { color: 'bad', size: 12, anchor: 'left', dx: 10, font: 'ui' }); }
    if (s.check && !s.n12) { const q = [1.0, 1.72]; g.marker.check(q, { size: 18, dx: -4 }); g.label('vertical gaps', q, { color: 'ok', size: 12, anchor: 'left', dx: 10, font: 'ui' }); }
    if (s.lineLabel && !s.n12) g.label('y = ' + LA.fmt(w[0]) + ' + ' + LA.fmt(w[1]) + 'x', [1.5, lineY(1.5)], { color: 'gold', size: 13, anchor: 'left', dx: 10, dy: 16 });
    if (s.showX2 && s.exact) { const q = quadFit(m.pts); g.label('y = ' + LA.fmt(q[0]) + ' ' + LA.fmt.signed(q[1]) + 'x ' + LA.fmt.signed(q[2]) + 'x²', [3.2, 1.2], { color: 'gold', size: 12.5, anchor: 'below', dy: 10 }); }
    // readout
    const x0 = r ? r.x + r.w - 10 : g.size.w - 10, y0 = r ? r.y + 16 : 16;
    g.label('sum of squares = ' + LA.fmt(s.exact && s.showX2 ? 0 : m.sse), [x0, y0], { screen: true, anchor: 'right', color: 'hud', size: 13, font: 'ui' });
    if (s.n12) g.label('n = 12 · synthetic data', [x0, y0 + 18], { screen: true, anchor: 'right', color: 'muted', size: 11.5, font: 'ui' });
    // handles
    if (s.lineHandles || inPlay) {
      g.handle('w0', [0, w[0]], (p, st) => { st.w[0] = +p[1].toFixed(3); }, { color: 'gold' });
      g.handle('w1', [3.5, lineY(3.5)], (p, st) => { st.w[1] = +((p[1] - st.w[0]) / 3.5).toFixed(3); }, { color: 'gold' });
    }
    if ((s.ptHandles === 'free' || inPlay) && !s.n12) m.pts.forEach((p, i) => g.handle('pt' + i, p, (q, st) => { st.pts[i] = [LA.clamp(+q[0].toFixed(2), 0.2, 3.8), LA.clamp(+q[1].toFixed(2), -0.5, 3.5)]; }, { color: 'white' }));
    else if (s.ptHandles === 'third') g.handle('pt2', m.pts[2], (q, st) => { st.pts[2] = [3, LA.clamp(+q[1].toFixed(2), -0.5, 3.5)]; }, { color: 'white' });
    return m.pts.map(p => { const q = g.P.toScreen(p); return [q[0], q[1]]; });
  }

  /* ---------------- the bracket panel: one row per point ---------------- */
  function drawTable(g, s, m, r) {
    const wide = r.w > 440;   // room for the normal-equation tiles beside the brackets (else they go underneath)
    const x0 = 12, rowH = 24, yHead = 40, yRow = 64;
    const cx = { ones: 68, x: 108, y: 160 };
    g.label('one row per point', P(x0, 14), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    if (s.tabX > 0) { g.label('ones', P(cx.ones, yHead), { color: 'i', size: 12.5, font: 'ui', bold: true }); g.label('x', P(cx.x, yHead), { color: 'j', size: 14, font: 'math' }); }
    if (s.tabY > 0) g.label('y', P(cx.y, yHead), { color: 'b', size: 14, font: 'math' });
    const n = s.n12 ? 12 : 3, rh = s.n12 ? 14 : rowH;
    for (let i = 0; i < Math.min(n, 3); i++) {
      const yy = yRow + i * rowH;
      g.label('pt ' + (i + 1), P(x0, yy), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
      const dg = s.n12 ? 1 : 2;
      if (s.tabX > 0) { g.label(LA.fmt(m.ones[i]), P(cx.ones, yy), { color: 'i', size: 14, font: 'math' }); g.label(LA.fmt(m.xs[i], dg), P(cx.x, yy), { color: 'j', size: 14, font: 'math' }); }
      if (s.tabY > 0) {
        const hl = s.legHL === i;
        if (hl) g.marker.ring(P(cx.y, yy), { r: 11 });
        g.label(LA.fmt(m.y[i], dg), P(cx.y, yy), { color: 'b', size: hl ? 16 : 14, font: 'math', bold: hl });
      }
    }
    if (s.n12) g.label('⋮  (12 rows)', P(x0, yRow + 3 * rowH), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    const yb0 = yRow - 13, yb1 = yRow + 2 * rowH + 13;
    if (s.tabX > 0) { bracket(g, cx.ones - 20, yb0, yb1, true); bracket(g, cx.x + 20, yb0, yb1, false); g.label('X', P((cx.ones + cx.x) / 2, yb1 + 12), { color: 'text', size: 14, font: 'math' }); }
    if (s.tabY > 0) { bracket(g, cx.y - 16, yb0, yb1, true, 'b'); bracket(g, cx.y + 16, yb0, yb1, false, 'b'); g.label('y', P(cx.y, yb1 + 12), { color: 'b', size: 14, font: 'math' }); }
    // the line's three heights (the magenta dots in the plot), stacked: the vector ŷ, right beside y
    if (s.showYhat && s.yhatT > 0 && !s.n12) {
      const xh = cx.y + 50, al = Math.min(1, s.yhatT * 1.5);
      g.label('ŷ', P(xh, yHead), { color: 'mag', size: 14, font: 'math', alpha: al });
      for (let i = 0; i < 3; i++) g.label(LA.fmt(s.exact && s.showX2 ? m.y[i] : m.yhat[i]), P(xh, yRow + i * rowH), { color: 'mag', size: 14, font: 'math', alpha: al });
      bracket(g, xh - 20, yb0, yb1, true, 'mag', 0.9 * al); bracket(g, xh + 20, yb0, yb1, false, 'mag', 0.9 * al);
      g.label('ŷ', P(xh, yb1 + 12), { color: 'mag', size: 14, font: 'math', alpha: al });
    }
    // normal-equation tiles
    const tx = wide ? 250 : 12, ty = wide ? 30 : 160;
    if (s.fillT > 0 || s.showEntry) {
      const k = Math.floor(s.fillT);
      const cellCol = (i, j) => i === j ? (i === 0 ? 'i' : 'j') : 'hud';
      g.label('XᵀX =', P(tx, ty + 14), { anchor: 'left', color: 'text', size: 13, font: 'ui' });
      const mx = tx + 52, cw = s.n12 ? 44 : 28, vw = s.n12 ? 52 : 36;   // cell / column-vector widths (12-point sums are wider)
      bracket(g, mx, ty - 2, ty + 32, true); bracket(g, mx + 8 + 2 * cw, ty - 2, ty + 32, false);
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const idx = i * 2 + j; if (idx >= k && !s.showEntry) continue;
        g.label(s.showEntry ? '?' : LA.fmt(m.XtX[i][j], s.n12 ? 1 : 2), P(mx + 4 + cw / 2 + j * cw, ty + 4 + i * 24), { color: s.showEntry ? 'muted' : cellCol(i, j), size: 13.5, font: 'math' });
      }
      if (s.showEntry) { // the typed matrix under the unknown one: a cell turns green when it equals the column dot product
        const ey = ty + 46;
        g.label('yours', P(tx, ey + 14), { anchor: 'left', color: 'gold', size: 13, font: 'ui' });
        bracket(g, mx, ey - 2, ey + 32, true, 'gold'); bracket(g, mx + 8 + 2 * cw, ey - 2, ey + 32, false, 'gold');
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { const ok = Math.abs(s.XtXEntry[i][j] - m.XtX[i][j]) <= 2 * LA.TOL.entry; g.label(LA.fmt(s.XtXEntry[i][j]), P(mx + 4 + cw / 2 + j * cw, ey + 4 + i * 24), { color: ok ? 'ok' : 'text', size: 13.5, font: 'math', bold: ok }); }
      } else {
        g.label('Xᵀy =', P(tx, ty + 60), { anchor: 'left', color: 'text', size: 13, font: 'ui' });
        bracket(g, mx, ty + 44, ty + 78, true, 'b'); bracket(g, mx + vw, ty + 44, ty + 78, false, 'b');
        for (let i = 0; i < 2; i++) { if (4 + i >= k) continue; g.label(LA.fmt(m.Xty[i], s.n12 ? 1 : 2), P(mx + vw / 2, ty + 50 + i * 24), { color: 'b', size: 13.5, font: 'math' }); }
        if (s.showWhat) { const wx = mx + vw + 14; g.label('ŵ =', P(wx, ty + 60), { anchor: 'left', color: 'gold', size: 13, font: 'ui' }); bracket(g, wx + 30, ty + 44, ty + 78, true, 'gold'); bracket(g, wx + 72, ty + 44, ty + 78, false, 'gold'); for (let i = 0; i < 2; i++) g.label(LA.fmt(m.what[i]), P(wx + 51, ty + 50 + i * 24), { color: 'gold', size: 13.5, font: 'math' }); }
      }
    }
    const rows = i => [r.x + cx.y, r.y + yRow + i * rowH];
    return { y: [0, 1, 2].map(rows), ones: [0, 1, 2].map(i => [r.x + cx.ones, r.y + yRow + i * rowH]), x: [0, 1, 2].map(i => [r.x + cx.x, r.y + yRow + i * rowH]) };
  }

  /* ---------------- the sample space (one axis per data point) ---------------- */
  function drawSpace(g, s, m, ctx, r, inPlay) {
    const L = s.axesT;
    const yv = [m.y[0], m.y[1], m.y[2]], ones = m.ones.slice(0, 3), xs = m.xs.slice(0, 3);
    const yhat = s.exact ? yv.slice() : m.yhat.slice(0, 3), e = N.sub(yv, yhat);
    // floor grid in the (point 1, point 2) sheet + three axes from the origin
    if (s.axesT > 0) {
      for (let i = 0; i <= 2; i++) { g.line([i, 0, 0], [i, 2.3, 0], { color: 'grid', width: 1, alpha: 0.55 * s.axesT, back: true }); g.line([0, i, 0], [3, i, 0], { color: 'grid', width: 1, alpha: 0.55 * s.axesT, back: true }); }
      g.line([3, 0, 0], [3, 2.3, 0], { color: 'grid', width: 1, alpha: 0.55 * s.axesT, back: true });
      const names = ['point 1', 'point 2', 'point 3'];
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((d, i) => { const tip = N.scale(d, L * AXLEN[i]); g.vector([0, 0, 0], tip, { color: 'gridAxis', width: 1.4, head: 8, alpha: 0.9 }); if (s.axesT > 0.6) g.label(names[i], tip, { color: 'hud', size: 12.5, font: 'ui', dx: i === 2 ? 30 : (i === 1 ? -40 : 24), dy: i === 2 ? 4 : (i === 1 ? -6 : -14) }); });
    }
    // the flip: walk to y one coordinate at a time (point 1's height along axis 1, point 2's along axis 2, …)
    if (s.yGuide > 0 && s.guideA > 0.01) {
      const c = [[0, 0, 0], [yv[0], 0, 0], [yv[0], yv[1], 0], yv];
      const off = [{ dx: 4, dy: 14 }, { dx: -6, dy: 16 }, { dx: -40, dy: 0 }];
      for (let k = 0; k < 3; k++) {
        const u = LA.clamp01(s.yGuide - k); if (u <= 0) continue;
        const hl = s.legHL === k;
        g.line(c[k], lerpV(c[k], c[k + 1], u), { color: 'b', width: hl ? 2.8 : 1.8, dashed: [6, 4], alpha: s.guideA });
        if (u > 0.5 && s.guideA > 0.5) g.label('y' + '₁₂₃'[k] + ' = ' + LA.fmt(yv[k]), lerpV(c[k], c[k + 1], 0.5), { color: hl ? 'white' : 'b', size: hl ? 13.5 : 12.5, bold: hl, font: 'ui', dx: off[k].dx, dy: off[k].dy, alpha: s.guideA });
      }
    }
    // the box = all of ℝ³ (mlTie step)
    if (s.showX2 && s.boxT > 0) { const B = 4.5 * s.boxT; g.box([[B, 0, 0], [0, B, 0], [0, 0, B]], { fill: 'teal', alpha: 0.07, stroke: 'teal', strokeAlpha: 0.35, subdiv: 2 }); }
    // the teal column plane
    if (s.showPlane && s.planeT > 0) {
      const foot = N.matVec(m.X.slice(0, 3), m.what);
      const pa = 0.2 * s.planeT * (s.showX2 ? 1 - s.boxT : 1);
      const pc = planeCenter(ones, xs);
      if (pa > 0.005) {
        g.planes([{ point: pc, span: [ones, xs], extentU: 2.1 * s.planeT, extentV: 1.3 * s.planeT, color: 'teal', alpha: pa, stroke: 'teal', subdiv: 3 }]);
        if (s.planeT > 0.9) { const u = N.normalize(ones), v = N.normalize(N.sub(xs, N.scale(u, N.dot(xs, u)))); g.label('span(ones, x)', N.add(pc, N.add(N.scale(u, -0.9), N.scale(v, -1.05))), { color: 'teal', size: 12.5, font: 'ui' }); }
      }
    }
    // the columns of X
    const glow = s.fillT > 0 && s.fillT < 6 ? [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2]][Math.floor(s.fillT)] : null;
    const gw = k => glow && (glow[0] === k || glow[1] === k) ? 4.6 : 3;
    if (s.colsT > 0) {
      g.vector([0, 0, 0], N.scale(ones, s.colsT), { color: 'i', width: gw(0), label: 'ones', labelFont: 'ui', labelSize: 12.5 });
      g.vector([0, 0, 0], N.scale(xs, s.colsT), { color: 'j', width: gw(1), label: 'x' });
      if (glow && glow[0] === 0 && glow[1] === 0) g.marker.ring(ones, { r: 12 });
      if (glow && glow[0] === 1 && glow[1] === 1) g.marker.ring(xs, { r: 12 });
    }
    if (s.showX2 && s.x2T > 0) g.vector([0, 0, 0], N.scale([1, 4, 9], s.x2T), { color: 'k', width: 3, label: 'x²', labelFont: 'ui', labelSize: 12.5 });
    // y
    if (s.yT > 0) g.vector([0, 0, 0], N.scale(yv, s.yT), { color: 'b', width: gw(2), label: 'y', dx: -10 });
    // ŷ with its recipe, the residual, right angles
    if (s.showYhat && s.yhatT > 0) {
      if (s.recipe) {   // tip-to-tail recipe; labels only on legs long enough to carry them
        const a = N.scale(ones, m.w[0]), b2 = N.scale(xs, m.w[1]);
        if (N.norm(a) > 0.02) g.vector([0, 0, 0], a, { color: 'i', width: 2, dashed: true, alpha: 0.75, head: 8, label: N.norm(a) > 0.5 ? 'w₀·ones' : undefined, labelAt: 0.5, labelSize: 11.5, labelFont: 'ui' });
        if (N.norm(b2) > 0.02) g.vector(a, N.add(a, b2), { color: 'j', width: 2, dashed: true, alpha: 0.75, head: 8, label: N.norm(b2) > 0.5 ? 'w₁·x' : undefined, labelAt: 0.5, labelSize: 11.5, labelFont: 'ui' });
      }
      if (s.showE && s.eT > 0 && N.norm(e) > 1e-6) {
        g.line(yhat, lerpV(yhat, yv, s.eT), { color: 'residual', width: 1.6, dashed: true, alpha: 0.95 });
        if (s.eT > 0.9) g.label('e', lerpV(yhat, yv, 0.5), { color: 'hud', size: 14, dx: 11, dy: 9 });
        if (s.compT > 0) { // staircase: the three components of e, one per axis = one vertical gap each
          const c = [yhat, [yv[0], yhat[1], yhat[2]], [yv[0], yv[1], yhat[2]], yv];
          for (let k = 0; k < 3; k++) { const hl = s.pulse === k; const b2 = lerpV(c[k], c[k + 1], s.compT); g.line(c[k], b2, { color: hl ? 'white' : 'hud', width: hl ? 3 : 1.8, alpha: hl ? 1 : 0.7 }); if (s.compT > 0.9 && hl) g.label('e' + '₁₂₃'[k], lerpV(c[k], c[k + 1], 0.5), { color: hl ? 'white' : 'muted', size: 12, dx: 10, dy: 6 }); }
        }
        if (s.rightAngle && Math.abs(N.dot(N.normalize(e), N.normalize(ones))) < 0.05 && Math.abs(N.dot(N.normalize(e), N.normalize(xs))) < 0.05) { rightAngle(g, yhat, e, ones, 0.16); rightAngle(g, yhat, e, xs, 0.16); }
      }
      g.point(yhat, { r: 5.5 * Math.min(1, s.yhatT), color: 'mag', label: 'ŷ', labelColor: 'mag', dx: 13, dy: -9 });
      if (s.yhatHandle || inPlay) {
        const nrm = N.cross(ones, xs);
        g.handle('yhat', yhat, (p, st) => { const mm = model(st); st.w = N.lstsq(mm.X, p).map(v => +v.toFixed(3)); }, { color: 'mag', plane: { point: [0, 0, 0], normal: nrm } });
      }
    }
    // R²: the right triangle ȳ·ones → ŷ → y in sample space, plus a flat inset with the squared lengths
    if (s.showR2 && s.r2T > 0) {
      const yb = N.scale(ones, m.ybar);
      g.point(yb, { r: 4, color: 'hud', label: 'ȳ·ones', labelColor: 'muted', dx: -30, dy: 14, size: 11.5 });
      g.line(yb, lerpV(yb, yhat, s.r2T), { color: 'mag', width: 2.2 });
      g.line(yb, lerpV(yb, yv, s.r2T), { color: 'b', width: 2.2 });
      drawR2Inset(g, s, m, r);
    }
    if (s.showBars) {
      const bx = r.x + r.w - 140, by = r.y + r.h - 104;
      const d1 = N.dot(e, ones), d2 = N.dot(e, xs);
      g.bars([d1, d2], { screen: true, x: bx, y: by, w: 120, h: 56, max: Math.max(3, 1.3 * Math.abs(d1), 1.3 * Math.abs(d2)), colors: ['i', 'j'], labels: ['e·ones', 'e·x'], values: true, title: 'residual · column' });
    }
    if (s.showE) g.label('|e|² = ' + LA.fmt(N.dot(e, e)), [r.x + 10, r.y + 16], { screen: true, anchor: 'left', color: 'hud', size: 13, font: 'ui' });   // e is named only from step 6 on
    g.label('sample space: one axis per data point (not x and y)', [r.x + 10, r.y + r.h - 14], { screen: true, anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
    if (s.showX2 && s.exact) g.label('3 points, 3 columns: exact fit, no generalization', [r.x + r.w - 10, r.y + r.h - 14], { screen: true, anchor: 'right', color: 'teal', size: 12, font: 'ui' });
  }

  function drawR2Inset(g, s, m, r) {
    const W = 230, H = 192, ir = { x: r.x + r.w - W - 8, y: r.y + 8, w: W, h: H };
    const sst = Math.sqrt(m.sst), ssr = Math.sqrt(Math.max(0, m.sst - m.sse)), sse = Math.sqrt(m.sse);
    const k = 125 / Math.max(sst, 1e-6);
    panel(g, ir, { bg: 'panel' }, (g2) => {
      g2.label('centred y: a right triangle', P(12, 14), { anchor: 'left', color: 'muted', size: 11.5, font: 'ui' });
      const o = [20, 130], a = [20 + ssr * k * s.r2T, 130], b = [20 + ssr * k * s.r2T, 130 - sse * k * s.r2T];
      g2.line(P(o[0], o[1]), P(a[0], a[1]), { color: 'mag', width: 2.4 });
      g2.line(P(a[0], a[1]), P(b[0], b[1]), { color: 'residual', width: 2, dashed: true });
      g2.line(P(o[0], o[1]), P(b[0], b[1]), { color: 'b', width: 2.4 });
      if (s.r2T > 0.95) rightAngle(g2, P(a[0], a[1]), [-1, 0], [0, 1], 9);
      g2.label('|ŷ − ȳ|² = ' + LA.fmt(m.sst - m.sse), P((o[0] + a[0]) / 2, 146), { color: 'mag', size: 12, font: 'ui' });
      g2.label('|e|² = ' + LA.fmt(m.sse), P(a[0] + 8, (a[1] + b[1]) / 2), { color: 'muted', size: 12, font: 'ui', anchor: 'left' });
      g2.label('|y − ȳ|² = ' + LA.fmt(m.sst), P((o[0] + b[0]) / 2 - 10, (o[1] + b[1]) / 2 - 12), { color: 'b', size: 12, font: 'ui' });
      const r2 = m.sst > 0 ? 1 - m.sse / m.sst : 0;
      g2.label('R² = kept fraction = ' + LA.fmt(r2 * s.r2T), P(12, 170), { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
      g2.polygon([P(12, 178), P(12 + 200, 178), P(212, 184), P(12, 184)], { fill: 'line', alpha: 0.9 });
      g2.polygon([P(12, 178), P(12 + 200 * r2 * s.r2T, 178), P(12 + 200 * r2 * s.r2T, 184), P(12, 184)], { fill: 'gold', alpha: 0.9 });
    });
  }

  /* ---------------- n = 12: the strip view (no 12-D picture exists) ---------------- */
  function drawCells(g, s, m, r) {
    const n = 12, rowH = LA.clamp((r.h - 110) / n, 16, 30), cw = 40, gap = 18, top = 56, left = 58;
    const nShow = Math.round(s.stripT * n);
    const tint = (role, t) => C.mix(C.panel2, C[role], 0.25 + 0.75 * LA.clamp01(t));
    const fmt1 = v => { const t = LA.fmt.fixed(v, 1); return t === LA.fmt.MINUS + '0.0' ? '0.0' : t; };
    const cols = [
      { x: left, title: 'ones', role: 'i', vals: m.ones, max: 1 },
      { x: left + cw, title: 'x', role: 'j', vals: m.xs, max: 4 },
      { x: left + 2 * cw + gap, title: 'y', role: 'b', vals: m.y, max: 3.2 },
      { x: left + 3 * cw + 2 * gap, title: 'ŷ', role: 'mag', vals: m.yhat, max: 3.2 },
      { x: left + 4 * cw + 3 * gap, title: 'e', role: null, vals: m.e, max: 1 },
    ];
    cols.forEach(c => {
      g.label(c.title, P(c.x + cw / 2, top - 16), { color: c.role || 'hud', size: 13.5, font: c.title.length > 1 ? 'ui' : 'math', bold: true });
      for (let i = 0; i < nShow; i++) {
        const v = c.vals[i], y0 = top + i * rowH;
        const fill = c.role ? tint(c.role, v / c.max) : C.diverge(LA.clamp(v / 0.8, -1, 1));
        g.polygon([P(c.x, y0), P(c.x + cw, y0), P(c.x + cw, y0 + rowH), P(c.x, y0 + rowH)], { fill, alpha: 1, stroke: 'line', width: 0.8 });
        g.label(c.role === 'i' ? '1' : fmt1(v), P(c.x + cw / 2, y0 + rowH / 2), { color: 'text', size: 11.5, font: 'mono', bg: false });
      }
    });
    for (let i = 0; i < nShow; i++) g.label('pt ' + (i + 1), P(left - 8, top + i * rowH + rowH / 2), { anchor: 'right', color: 'muted', size: 11, font: 'ui' });
    bracket(g, left - 4, top - 4, top + n * rowH + 4, true); bracket(g, left + 2 * cw + 4, top - 4, top + n * rowH + 4, false);
    g.label('X  (12×2)', P(left + cw, top + n * rowH + 16), { color: 'text', size: 12.5, font: 'ui' });
    g.label('12×1', P(cols[2].x + cw / 2, top + n * rowH + 16), { color: 'muted', size: 11.5, font: 'ui' });
    g.label('= Xŵ', P(cols[3].x + cw / 2, top + n * rowH + 16), { color: 'muted', size: 11.5, font: 'ui' });
    g.label('= y − ŷ', P(cols[4].x + cw / 2, top + n * rowH + 16), { color: 'muted', size: 11.5, font: 'ui' });
    const tx = cols[4].x + cw + 22;
    g.label('n = 12: sample space is ℝ¹²', P(tx, top - 10), { anchor: 'left', color: 'hud', size: 12.5, font: 'ui' });
    g.label('(not drawable)', P(tx, top + 8), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    g.label('The picture stops at 3.', P(tx, top + 36), { anchor: 'left', color: 'text', size: 13.5, font: 'ui', bold: true });
    g.label('The formula does not.', P(tx, top + 56), { anchor: 'left', color: 'text', size: 13.5, font: 'ui', bold: true });
    g.label('XᵀX is still 2×2:', P(tx, top + 90), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    const mx = tx + 8, my = top + 104;
    bracket(g, mx, my - 2, my + 32, true); bracket(g, mx + 100, my - 2, my + 32, false);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g.label(LA.fmt(m.XtX[i][j]), P(mx + 27 + j * 48, my + 4 + i * 24), { color: i === j ? (i === 0 ? 'i' : 'j') : 'hud', size: 13, font: 'math' });
    g.label('ŵ = [' + LA.fmt(m.what[0]) + '; ' + LA.fmt(m.what[1]) + ']', P(tx, my + 56), { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
    g.label('e ⟂ ones and e ⟂ x, still', P(tx, my + 78), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    g.label('e·ones = ' + LA.fmt(N.dot(m.e, m.ones)) + '   e·x = ' + LA.fmt(N.dot(m.e, m.xs)), P(tx, my + 98), { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
  }
})();
