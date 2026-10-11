/* p4-06 — Symmetric matrices, quadratic forms, bowls and saddles. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM;
  const I2 = () => [[1, 0], [0, 1]];
  const SYM = [[2, 1], [1, 2]];
  const NONSYM = [[3, 1], [0, 2]];
  const HONEST = 'The picture stops at 3. The formula does not.';
  const gold = x => '<span class="c-gold">' + x + '</span>';
  const ZS = 0.15;                                // height scale for surfaces (z = ZS · xᵀAx)
  const Aeff = s => [[s.A[0][0], s.b], [s.b, s.A[1][1]]];
  const qf = (A, p) => A[0][0] * p[0] * p[0] + (A[0][1] + A[1][0]) * p[0] * p[1] + A[1][1] * p[1] * p[1];
  const eig = A => N.eigSym([[A[0][0], (A[0][1] + A[1][0]) / 2], [(A[0][1] + A[1][0]) / 2, A[1][1]]]);
  const surfRamp = t => t < 0.5 ? LA.color.mix(LA.color.neg, LA.color.heatLo, Math.max(0, t * 2 - 0.15) / 0.85) : LA.color.heat((t - 0.5) * 2);
  function shapeWord(l) {
    const eps = 0.05;
    if (l[0] > eps && l[1] > eps) return 'positive definite: bowl';
    if (l[0] < -eps && l[1] < -eps) return 'negative definite: dome';
    if (l[0] * l[1] < -eps * eps) return 'indefinite: saddle';
    return 'positive semi-definite: trough';
  }

  /* ---------- data computed once ---------- */
  function digitSpectrum() {
    const d = LA.rng.syntheticDigits(46, 90, { size: 8 }).map(x => x.pixels);
    const vals = N.eigSym(N.cov(d)).values.map(v => Math.max(0, v));
    return vals.map(v => Math.round(v * 10000) / 10000);
  }

  /* ---------- 2D pictures ---------- */
  function goldAxes(g, A, o) {
    o = o || {};
    const e = eig(A), q1 = e.vectors[0], q2 = e.vectors[1], l1 = e.values[0], l2 = e.values[1];
    if (o.lines) { g.lineThrough([0, 0], q1, { color: 'gold', alpha: 0.25, width: 1 }); g.lineThrough([0, 0], q2, { color: 'gold', alpha: 0.25, width: 1 }); }
    const s1 = o.unit ? 1 : Math.abs(l1), s2 = o.unit ? 1 : Math.abs(l2);
    g.vector(N.scale(q1, s1), { color: 'gold', width: 3, label: 'q₁', labelColor: 'gold' });
    g.vector(N.scale(q2, s2), { color: 'gold', width: 3, label: 'q₂', labelColor: 'gold' });
    g.arc([0, 0], q1, q2, { r: 0.32, color: 'gold', right: true });
    if (o.dims) { g.dim([0, 0], N.scale(q1, s1), { label: LA.fmt(l1), offset: 16, color: 'gold' }); g.dim([0, 0], N.scale(q2, s2), { label: LA.fmt(Math.abs(l2)), offset: -16, color: 'gold' }); }
  }
  function ellipseScene(g, A, o) {
    o = o || {};
    if (o.ghost !== false) g.ghostGrid();
    g.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.2, strokeAlpha: 0.6 });
    g.ellipse(A, [0, 0], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 2.2 });
  }
  function eigTicks(g, vals, y, o) {
    o = o || {};
    const lo = o.lo === undefined ? -2 : o.lo, hi = o.hi === undefined ? 5 : o.hi, x0 = o.x0 === undefined ? -2.6 : o.x0, x1 = o.x1 === undefined ? 2.6 : o.x1;
    const X = l => x0 + (LA.clamp(l, lo, hi) - lo) / (hi - lo) * (x1 - x0);
    g.line([x0, y], [x1, y], { color: 'gridAxis', width: 1.2 });
    for (let l = Math.ceil(lo); l <= hi; l++) { g.line([X(l), y - 0.05], [X(l), y + 0.05], { color: l === 0 ? 'text' : 'gridAxis', width: l === 0 ? 2 : 1 }); g.label(l < 0 ? '−' + (-l) : String(l), [X(l), y], { anchor: 'below', dy: 8, size: 10.5, color: l === 0 ? 'text' : 'muted', font: 'ui' }); }
    vals.forEach((l, i) => {
      const col = l < -0.05 ? 'bad' : 'gold';
      g.point([X(l), y], { r: 5, color: col });
      g.label('λ' + '₁₂'[i] + ' = ' + LA.fmt(l), [X(l), y], { anchor: 'above', dy: -8 - 14 * i, size: 11.5, color: col, font: 'ui' });
    });
    if (o.title) g.label(o.title, [x0, y], { anchor: 'left', dy: -32, size: 11.5, color: 'muted', font: 'ui' });
  }

  /* ---------- 3D surface picture ---------- */
  function surfaceScene(g, A, o) {
    o = o || {};
    const R = o.R || 1.5, grow = o.grow === undefined ? 1 : o.grow;
    const b = { x0: -R, x1: R, y0: -R, y1: R };
    const f = p => qf(A, p) * ZS * grow;
    g.grid(null, { range: 2, alpha: 0.35 });
    const lv = o.levels || [-4, -2, -1, 1, 2, 4, 8];
    if (lv.length) g.contours(p => qf(A, p), lv, b, { res: 40, colors: lv.map(l => l < 0 ? 'neg' : 'hud'), width: 1.1, alpha: 0.7, z: 0 });
    g.surface(f, b, { res: o.res || 26, mode: 'both', ramp: surfRamp, zRange: [-2.4, 2.4], alpha: 0.6, wireColor: 'teal', wireAlpha: 0.35 });
    const e = eig(A);
    if (o.parabolas !== false) e.vectors.forEach((q, i) => {
      const pts = []; for (let k = -24; k <= 24; k++) { const r = k / 24 * R; const p = N.scale(q, r); pts.push([p[0], p[1], f(p) + 0.01]); }
      const flat = Math.abs(e.values[i]) < 0.06;
      g.polyline(pts, { color: e.values[i] < -0.05 ? 'bad' : 'gold', width: flat ? 3.5 : 2.6 });
      if (o.labels !== false) g.label(flat ? 'λ' + '₁₂'[i] + ' = 0' : (Math.abs(e.values[i] - 1) < 0.01 ? '' : LA.fmt(e.values[i])) + 'r²', pts[pts.length - 1], { color: e.values[i] < -0.05 ? 'bad' : 'gold', size: 13, dx: 10, dy: -8 });
    });
  }
  const surfCam = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.45], dist: 10.5, yaw: 0.2, pitch: 0.5 });

  /* ---------- the b slider (2D inset with a handle) ---------- */
  function bSlider(g, s, rect) {
    g.viewport(rect, { view: '2d', cam: { center: [0, 0], halfH: 0.62 } }, (g2) => {
      g2.line([-3, -0.12], [3, -0.12], { color: 'gridAxis', width: 2 });
      [-3, -2, 0, 2, 3].forEach(x => { g2.line([x, -0.2], [x, -0.04], { color: Math.abs(x) === 2 ? 'gold' : 'gridAxis', width: 1.2 }); g2.label(x < 0 ? '−' + (-x) : String(x), [x, -0.2], { anchor: 'below', dy: 6, size: 10.5, color: Math.abs(x) === 2 ? 'gold' : 'muted', font: 'ui' }); });
      g2.label('b = ' + LA.fmt(s.b) + '   (drag)', [-3, 0.32], { anchor: 'left', size: 12, color: 'v', font: 'ui' });
      g2.label('|b| = 2: trough', [2, 0.32], { size: 11, color: 'gold', font: 'ui' });
      g2.point([s.b, -0.12], { r: 6, color: 'v' });
      g2.handle('b', [s.b, -0.12], (p, st) => { st.b = Math.round(LA.clamp(p[0], -3, 3) * 100) / 100; }, { color: 'v', r: 14 });
    });
  }

  /* ---------- scenes ---------- */
  function renderEllipse(g, s, ctx) {
    const M = s.morph < 1 ? lerpM(I2(), Aeff(s), s.morph) : Aeff(s);
    g.grid(M, { alpha: 0.35 });
    ellipseScene(g, M);
    if (s.morph >= 1) goldAxes(g, Aeff(s), { lines: true, dims: s.showDims });
    if (s.showCols) { g.lineThrough([0, 0], [1, 1], { color: 'muted', dashed: true, width: 1, len: 6 }); g.label('target direction [1; 1]', [2.6, 2.6], { color: 'muted', size: 12, font: 'ui' }); colHandles(g, s); }
  }
  function colHandles(g, s) {
    const A = Aeff(s), c1 = [A[0][0], A[1][0]], c2 = [A[0][1], A[1][1]];
    g.vector(c1, { color: 'i', width: 1.6, label: 'col 1', labelColor: 'i', alpha: 0.85, dashed: true, head: 9 });
    g.vector(c2, { color: 'j', width: 1.6, label: 'col 2', labelColor: 'j', alpha: 0.85, dashed: true, head: 9 });
    g.handle('c1', c1, (p, st) => { st.A[0][0] = round2(p[0]); st.b = round2(p[1]); }, { color: 'i' });
    g.handle('c2', c2, (p, st) => { st.b = round2(p[0]); st.A[1][1] = round2(p[1]); }, { color: 'j' });
  }
  const round2 = x => Math.round(LA.clamp(x, -4, 4) * 100) / 100;

  function renderCompare(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    const cam = { center: [0, 0], halfH: 2.5 };
    g.viewport({ x: 0, y: 0, w: 1, h: 0.5 }, { view: '2d', cam }, (g2, r) => {
      ellipseScene(g2, SYM); goldAxes(g2, SYM, { lines: true });
      g2.hud('symmetric [[2,1],[1,2]]', r.x + 10, r.y + 18, { color: 'text', size: 13 });
      g2.hud('eigenvectors = ellipse axes, 90°', r.x + 10, r.y + 38, { color: 'gold', size: 12.5 });
      if (s.chip > 0) g2.hud('Spectral theorem: A = Aᵀ ⇒ q₁ ⟂ q₂', r.x + r.w - 270, r.y + r.h - 20, { color: 'ok', size: 13, alpha: s.chip, bold: true });
    });
    g.viewport({ x: 0, y: 0.5, w: 1, h: 0.5 }, { view: '2d', cam: { center: [0, 0], halfH: 3.4 } }, (g2, r) => {
      ellipseScene(g2, NONSYM);
      const sv = N.svd(NONSYM), u1 = N.col(sv.U, 0), u2 = N.col(sv.U, 1);
      g2.line(N.scale(u1, -sv.S[0]), N.scale(u1, sv.S[0]), { color: 'residual', dashed: true, width: 1.3 });
      g2.line(N.scale(u2, -sv.S[1]), N.scale(u2, sv.S[1]), { color: 'residual', dashed: true, width: 1.3 });
      const e1 = [1, 0], e2 = N.normalize([1, -1]);
      g2.lineThrough([0, 0], e1, { color: 'gold', alpha: 0.6, width: 1.6 }); g2.lineThrough([0, 0], e2, { color: 'gold', alpha: 0.6, width: 1.6 });
      g2.vector(N.scale(e1, 3), { color: 'gold', width: 2.6, label: 'v₁', labelColor: 'gold' });
      g2.vector(N.scale(e2, -2), { color: 'gold', width: 2.6, label: 'v₂', labelColor: 'gold' });
      g2.arc([0, 0], e1, N.scale(e2, -1), { r: 0.6, color: 'gold', label: '135°' });
      g2.hud('not symmetric [[3,1],[0,2]]', r.x + 10, r.y + 18, { color: 'text', size: 13 });
      g2.hud('gold eigenlines meet at 135°', r.x + 10, r.y + 38, { color: 'gold', size: 12.5 });
      g2.hud('dashed: the ellipse’s own axes (different)', r.x + 10, r.y + 58, { color: 'residual', size: 12.5 });
    });
  }

  function stageMatrix(st) {
    const R = th => N.rot2(th), q = Math.PI / 4;
    if (st <= 1) return R(-q * st);
    if (st <= 2) { const u = st - 1; return N.matMul(N.diag([1 + 2 * u, 1]), R(-q)); }
    const u = st - 2; return N.matMul(R(q * u), N.matMul(N.diag([3, 1]), R(-q)));
  }
  function renderQLQ(g, s, ctx) {
    const W = ctx.size.w;
    const M = stageMatrix(s.stage);
    g.ghostGrid();
    g.grid(M, { alpha: 0.6 });
    g.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.2, strokeAlpha: 0.6 });
    g.ellipse(M, [0, 0], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 2.2 });
    const q1 = N.normalize([1, 1]), q2 = N.normalize([-1, 1]);
    g.vector(N.matVec(M, q1), { color: 'gold', width: 3, label: 'q₁', labelColor: 'gold' });
    g.vector(N.matVec(M, q2), { color: 'gold', width: 3, label: 'q₂', labelColor: 'gold' });
    const names = ['Qᵀ: turn −45°', 'Λ: stretch 3, 1', 'Q: turn back +45°'];
    const active = s.stage <= 0 ? -1 : s.stage < 1 ? 0 : s.stage < 2 ? 1 : s.stage < 3 ? 2 : 3;
    names.forEach((nm, i) => g.hud(nm, 14 + i * (W - 20) / 3, 22, { color: active === i ? 'gold' : (active > i ? 'hud' : 'muted'), size: 13.5, bold: active === i }));
    if (s.stage >= 3) g.hud('= A: the same ellipse as step 1', 14, 46, { color: 'teal', size: 13 });
  }

  function renderForm(g, s, ctx) {
    const A = Aeff(s), bnd = { x0: -2.2, x1: 2.2, y0: -2.2, y1: 2.2 };
    g.heat2d(p => qf(A, p), bnd, { res: 70, range: [0, 16], ramp: LA.color.heat, alpha: 0.85 * s.heatA });
    if (s.heatA > 0.5) g.contours(p => qf(A, p), [1, 2, 4, 8], bnd, { res: 60, color: 'white', width: 1.2, alpha: 0.75 });
    if (s.heatA > 0.5) [1, 2, 4, 8].forEach(l => { const e = eig(A); const p = N.scale(e.vectors[1], Math.sqrt(l / e.values[1])); if (Math.abs(p[0]) < 1.9 && Math.abs(p[1]) < 1.9) g.label(String(l), p, { size: 11, color: 'white', dx: 8, dy: 8 }); });
    goldAxes(g, A, { lines: true, unit: true });
    const v = qf(A, s.probe);
    g.point(s.probe, { r: 6, color: 'v', label: 'xᵀAx = ' + LA.fmt(v), labelColor: 'v' });
    g.handle('probe', s.probe, (p, st) => { st.probe = [LA.clamp(p[0], -2.2, 2.2), LA.clamp(p[1], -2.2, 2.2)]; }, { color: 'v' });
  }

  function renderSurface(g, s, ctx) {
    const A = Aeff(s);
    surfaceScene(g, A, { grow: s.grow, res: 26, labels: s.panel !== 'mis' });
    if (s.panel === 'mis') {
      g.hud('every entry of A is positive: 1, 2, 2, 1', 14, ctx.size.h - 52, { color: 'ok', size: 13.5 });
      if (s.cross) g.hud('yet xᵀAx = −1 at x = [1, −1]/√2: a saddle', 14, ctx.size.h - 28, { color: 'bad', size: 13.5 });
    }
    if (s.showProbe) {
      const p = s.probe, z = qf(A, p) * ZS * s.grow;
      g.line([p[0], p[1], 0], [p[0], p[1], z], { color: 'v', dashed: true, width: 1.4 });
      g.point([p[0], p[1], z], { r: 6, color: s.cross ? 'bad' : 'v', label: 'xᵀAx = ' + LA.fmt(qf(A, p)), labelColor: s.cross ? 'bad' : 'v' });
      if (s.cross) g.marker.cross([p[0], p[1], z], { dx: -40, dy: -22 });
      if (ctx.mode !== 'learn') g.handle('probe3', [p[0], p[1], 0], (q, st) => { st.probe = [LA.clamp(q[0], -1.5, 1.5), LA.clamp(q[1], -1.5, 1.5)]; }, { color: 'v', plane: 'xy' });
    }
    if (s.showTicks) g.viewport({ x: 0.02, y: 0.02, w: 0.5, h: 0.17 }, { view: '2d', cam: { center: [0, 0.15], halfH: 0.75 } }, (g2) => { eigTicks(g2, eig(A).values, 0, { lo: -2, hi: 5, x0: -1.7, x1: 1.7 }); });
    if (s.showWord) g.hud(shapeWord(eig(A).values), ctx.size.w * 0.56, ctx.size.h * 0.06, { color: eig(A).values[1] < -0.05 ? 'bad' : 'gold', size: 14, bold: true });
    if (s.showB) bSlider(g, s, { x: 0.04, y: 0.86, w: 0.92, h: 0.12 });
  }

  function miniSurface(g, rect, A, mark, dist) {
    g.viewport(rect, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.1], dist: dist || 7.2, yaw: 0.7, pitch: 0.5 }) }, (g2, r) => {
      surfaceScene(g2, A, { res: 14, parabolas: true, labels: false, levels: [-2, 1, 3] });
      if (mark === 'dot') g2.point([0, 0, 0], { r: 6, color: 'gold' });
      if (mark === 'line') g2.line([0, -1.5, 0.01], [0, 1.5, 0.01], { color: 'gold', width: 4 });
      if (mark === 'cross') g2.marker.cross([0, 0, 0], { size: 16 });
    });
  }
  const ROWS = [{ x: 0.03, y: 0.03, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.35, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.67, w: 0.55, h: 0.3 }];
  function rowCaptions(g, W, H, caps) {
    ROWS.forEach((b, i) => { const y = (b.y + b.h / 2) * H; const x = (b.x + b.w + 0.03) * W; g.hud(caps[i][0], x, y - 18, { color: caps[i][2] || 'text', size: 14, bold: true }); g.hud(caps[i][1], x, y + 4, { color: 'muted', size: 12.5 }); if (caps[i][3]) g.hud(caps[i][3], x, y + 24, { color: 'muted', size: 12.5 }); });
  }
  function renderThree(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    miniSurface(g, ROWS[0], [[3, 0], [0, 1]], 'dot');
    miniSurface(g, ROWS[1], [[3, 0], [0, 0]], 'line');
    miniSurface(g, ROWS[2], [[3, 0], [0, -1]], 'cross');
    rowCaptions(g, W, H, [['Positive definite', 'λ = 3, 1: all > 0', 'gold', 'a bowl, one lowest point'], ['Positive semi-definite', 'λ = 3, 0: none < 0', 'gold', 'a trough, a line of lowest points'], ['Indefinite', 'λ = 3, −1: mixed signs', 'bad', 'a saddle, no lowest point']]);
  }

  function renderN64(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h, v = s.eig64;
    const u = s.reveal;
    const shown = v.map((x, i) => i < Math.ceil(64 * u) ? x : 0);
    g.bars(shown, { screen: true, x: W * 0.06, y: H * 0.18, w: W * 0.88, h: H * 0.34, gap: 1, color: 'gold', max: v[0], title: '64 eigenvalues λ₁ ≥ … ≥ λ₆₄ ≥ 0 of a 64×64 covariance (sorted)' });
    g.strip(shown, { x: W * 0.06, y: H * 0.6, w: W * 0.88, h: 26, range: [0, v[0]], ramp: LA.color.heat, title: 'the same 64 numbers as a strip: all ≥ 0 → nothing curves down (a bowl, possibly with flat directions)' });
    g.hud('64 perpendicular directions', W * 0.06, H * 0.71, { color: 'gold', size: 14, bold: true });
    g.hud(HONEST, W * 0.06, H * 0.79, { color: 'text', size: 15 });
    g.hud('data: 90 synthetic 8×8 digit images (seed 46), 64 pixels each', W * 0.06, H * 0.87, { color: 'muted', size: 12 });
  }

  function renderML(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h, u = s.mlT;
    const cam = { center: [0, 0], halfH: 2.1 };
    g.viewport(ROWS[0], { view: '2d', cam: { center: [2, 1], halfH: 2.3 } }, (g2) => {
      g2.points(s.cloud, { r: 2.6, color: 'data', alpha: 0.85 });
      const mu = N.mean(s.cloud), C = N.cov(s.cloud), e = N.eigSym(C);
      const S = N.matMul(N.fromCols(e.vectors), N.diag(e.values.map(x => 2 * Math.sqrt(x))));
      g2.ellipse(S, mu, { stroke: 'teal', width: 2 });
      e.vectors.forEach((q, i) => g2.vector(mu, N.add(mu, N.scale(q, 2 * Math.sqrt(e.values[i]))), { color: 'gold', width: 2.4, head: 8 }));
    });
    g.viewport(ROWS[1], { view: '2d', cam }, (g2) => {
      const H2 = N.matMul(N.matMul(N.rot2(0.5), [[3, 0], [0, 0.35]]), N.transpose(N.rot2(0.5)));
      g2.contours(p => qf(H2, p), [0.2, 0.6, 1.2, 2.2, 3.4], { x0: -3.4, x1: 3.4, y0: -2.1, y1: 2.1 }, { res: 50, color: 'teal', width: 1.1 });
      let w = N.matVec(N.rot2(0.5), [0.9, -2.4]); const path = [w];
      for (let k = 0; k < 14; k++) { w = N.sub(w, N.scale(N.matVec(H2, w), 0.55)); path.push(w); }
      g2.point([0, 0], { r: 4, color: 'gold' });
      const K = Math.max(1, Math.round(u * 14));
      g2.polyline(path.slice(0, K + 1), { color: 'v', width: 2 });
      g2.points(path.slice(0, K + 1), { r: 3, color: 'v' });
    });
    g.viewport(ROWS[2], { view: '2d', cam: { center: [0, 0], halfH: 1.9 } }, (g2) => {
      const K = s.gram, b = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };
      g2.image(K, b, { ramp: LA.color.heat, range: [-0.2, 1], cells: true });
      g2.line([-1.5, 1.5], [1.5, -1.5], { color: 'gold', width: 2, dashed: true });
      const i = Math.min(5, Math.floor(u * 6)), j = (i + 2) % 6;
      const cell = (r, c) => [[-1.5 + c * 0.5, 1.5 - r * 0.5], [-1 + c * 0.5, 1.5 - r * 0.5], [-1 + c * 0.5, 1 - r * 0.5], [-1.5 + c * 0.5, 1 - r * 0.5]];
      g2.polygon(cell(i, j), { stroke: 'white', width: 2.2, alpha: 0 }); g2.polygon(cell(j, i), { stroke: 'white', width: 2.2, alpha: 0 });
    });
    rowCaptions(g, W, H, [['Covariance Σ (p4-09)', 'symmetric PSD: the data’s', 'teal', 'spread is its ellipse'], ['Loss Hessian H (p5-02)', 'bowl = convex; largest λ', 'teal', 'caps the learning rate'], ['Kernel / Gram K (p3-07)', 'symmetric PSD: mirror-equal', 'teal', 'across the gold diagonal']]);
  }

  function renderRecap(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    g.viewport(ROWS[0], { view: '2d', cam: { center: [0, 0], halfH: 2.4 } }, (g2) => { ellipseScene(g2, SYM, { ghost: false }); goldAxes(g2, SYM, { lines: false }); });
    g.viewport(ROWS[1], { view: '2d', cam: { center: [0, 0], halfH: 1.4 } }, (g2) => {
      [[-1.45, stageMatrix(1)], [0, stageMatrix(2)], [1.45, stageMatrix(3)]].forEach(([x, M], i) => {
        const S = N.scaleM(M, 0.23);
        g2.ellipse(S, [x, 0], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 1.8 });
        g2.vector([x, 0], N.add([x, 0], N.matVec(S, N.normalize([1, 1]))), { color: 'gold', width: 2, head: 7 });
        g2.label(['Qᵀ', 'Λ', 'Q'][i], [x, -1.1], { color: 'gold', size: 14 });
      });
      g2.label('→', [-0.74, 0], { color: 'hud', size: 18 }); g2.label('→', [0.74, 0], { color: 'hud', size: 18 });
    });
    const third = ROWS[2];
    [[[3, 0], [0, 1]], [[3, 0], [0, 0]], [[3, 0], [0, -1]]].forEach((A, i) => miniSurface(g, { x: third.x + i * third.w / 3, y: third.y, w: third.w / 3, h: third.h }, A, ['dot', 'line', 'cross'][i], 11));
    rowCaptions(g, W, H, [['Symmetric', 'perpendicular gold axes'], ['A = QΛQᵀ', 'turn, stretch, turn back'], ['Signs of λ', 'bowl · trough · saddle']]);
  }

  LA.registerLesson({
    id: 'p4-06-symmetric-quadratic-forms',
    part: 4,
    title: 'Symmetric matrices, quadratic forms, bowls and saddles',
    subtitle: 'A symmetric matrix is turn–stretch–turn back; xᵀAx is a surface whose curvatures are the eigenvalues.',
    minutes: 11,
    prereqs: ['p4-02-eigenvectors', 'p4-01-change-of-basis', 'p2-10-transpose-outer-symmetric'],
    tags: ['core'],
    badge: { drawnIn: '2D ellipse + 3D surface', realSize: 'a loss Hessian for a 10⁶-parameter model is a 10⁶×10⁶ symmetric matrix; a covariance of 768 features is 768×768', transfers: 'perpendicular eigenvectors, A = QΛQᵀ and “signs of λ decide bowl vs saddle” hold exactly in every dimension' },
    cantShow: ['a quadratic form in three or more variables as a surface (a bowl in ℝⁿ has n curvatures; we show two)', 'hyperbolic level sets in 3D (hyperboloids); the 2D hyperbola stands in'],
    terms: [
      { term: 'symmetric', def: 'A matrix equal to its transpose, A = Aᵀ: mirror-equal across the diagonal.' },
      { term: 'spectral theorem', def: 'Every symmetric matrix has perpendicular unit eigenvectors, so A = QΛQᵀ with Q orthonormal and Λ the diagonal of eigenvalues.' },
      { term: 'quadratic form', def: 'The number xᵀAx attached to each point x; drawn as height, it is a surface over the plane.' },
      { term: 'positive definite', def: 'xᵀAx > 0 for every x ≠ 0; all eigenvalues positive; a bowl.' },
      { term: 'positive semi-definite', def: 'xᵀAx ≥ 0 for every x; eigenvalues ≥ 0; a bowl with a flat direction (a trough).' },
      { term: 'indefinite', def: 'xᵀAx takes both signs; eigenvalues of mixed sign; a saddle.' },
    ],
    atlas: ['optimization', 'gaussian', 'kernels', 'pca'],
    thumbnail: 4,

    state() {
      const rand = LA.rng(61); const vecs = []; for (let i = 0; i < 6; i++) vecs.push(N.normalize(LA.rng.gaussV(rand, 4)));
      const gram = vecs.map(a => vecs.map(b => Math.round(N.dot(a, b) * 100) / 100));
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { halfH: 3.4 }),
        scene: 'ellipse', panel: 'sym',
        A: [[2, 0], [0, 2]], b: 1, morph: 1, showDims: true, showCols: false,
        stage: 0, chip: 0, heatA: 1, probe: [1.2, 0.4], grow: 1, showProbe: false, cross: false,
        showTicks: false, showWord: false, showB: false, reveal: 1, mlT: 1,
        eig64: digitSpectrum(), gram,
        cloud: LA.rng.gaussianCloud(49, 80, { mean: [2, 1], cov: [[2, 1.2], [1.2, 1]] }),
        Atyped: [[1, 0], [0, 1]], showContours: true,
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'compare': return renderCompare(g, s, ctx);
        case 'qlq': return renderQLQ(g, s, ctx);
        case 'form': return renderForm(g, s, ctx);
        case 'surface': return renderSurface(g, s, ctx);
        case 'three': return renderThree(g, s, ctx);
        case 'n64': return renderN64(g, s, ctx);
        case 'ml': return renderML(g, s, ctx);
        case 'recap': return renderRecap(g, s, ctx);
        case 'entry': {
          const A = s.Atyped; ellipseScene(g, A);
          const sym = Math.abs(A[0][1] - A[1][0]) < 0.05;
          if (sym) goldAxes(g, A, { lines: true }); else g.label('not symmetric yet: a₁₂ ≠ a₂₁', [0, -3], { color: 'bad', size: 13, font: 'ui' });
          return;
        }
        case 'play': return renderPlay(g, s, ctx);
        default: return renderEllipse(g, s, ctx);
      }
    },

    math(s, tex) {
      const A = s.scene === 'entry' ? s.Atyped : Aeff(s), e = eig(A);
      const T = (x, o) => tex.text(x, o);
      const lam = 'λ = ' + LA.fmt(e.values[0]) + ', ' + LA.fmt(e.values[1]);
      const qform = 'xᵀAx = ' + LA.fmt(A[0][0]) + 'x² ' + (2 * A[0][1] < 0 ? '− ' : '+ ') + LA.fmt(Math.abs(2 * A[0][1])) + 'xy ' + (A[1][1] < 0 ? '− ' : '+ ') + LA.fmt(Math.abs(A[1][1])) + 'y²';
      switch (s.panel) {
        case 'sym': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' }), tex.op('='), tex.sym('A', { sup: 'T' })), tex.eq(T(lam, { color: 'gold' })), tex.eq(tex.sym('q', { sub: '1', color: 'gold' }), tex.op('⟂'), tex.sym('q', { sub: '2', color: 'gold' })));
        case 'compare': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('A', { sup: 'T' }), tex.op('⇒'), tex.sym('q', { sub: '1', color: 'gold' }), tex.op('⟂'), tex.sym('q', { sub: '2', color: 'gold' })), tex.eq(T('[[3,1],[0,2]]: eigenvectors meet at 135°', { color: 'muted' })));
        case 'qlq': {
          const lit = k => (s.stage > k && s.stage <= k + 1) || (k === 2 && s.stage >= 3);
          const c = Math.SQRT1_2;
          return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('Q'), tex.op('·'), tex.sym('Λ'), tex.op('·'), tex.sym('Q', { sup: 'T' })),
            tex.eq(tex.sym('Q'), tex.op('='), tex.mat([[c, -c], [c, c]], { colors: 'gold', highlight: () => lit(2) }), tex.sym('Λ'), tex.op('='), tex.mat([[3, 0], [0, 1]], { highlight: () => lit(1) }), tex.sym('Q', { sup: 'T' }), tex.op('='), tex.mat([[c, c], [-c, c]], { colors: 'gold', highlight: () => lit(0) })));
        }
        case 'form': return tex.block(tex.eq(T(qform)), tex.eq(tex.sym('f'), T('(probe) = ' + LA.fmt(qf(A, s.probe)), { color: 'v' })));
        case 'bowl': return tex.block(tex.eq(T('z = xᵀAx = 3·(x·q₁)² + 1·(x·q₂)²', { color: 'gold' })));
        case 'slide': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' })), tex.eq(T('λ = 2 + b, 2 − b = ' + LA.fmt(e.values[0]) + ', ' + LA.fmt(e.values[1]), { color: e.values[1] < -0.05 ? 'bad' : 'gold' })), tex.eq(T(shapeWord(e.values))));
        case 'mis': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: () => 'ok' })), tex.eq(T('[1, −1]·A·[1; −1] = −2 < 0', { color: 'bad' })), tex.eq(T('λ = 3, −1', { color: 'bad' })));
        case 'words': return tex.block(tex.eq(T('PD: λ > 0', { color: 'gold' })), tex.eq(T('PSD: λ ≥ 0', { color: 'gold' })), tex.eq(T('indefinite: mixed', { color: 'bad' })));
        case 'n64': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('Q'), tex.sym('Λ'), tex.sym('Q', { sup: 'T' })), tex.eq(tex.sym('Λ'), tex.op('='), T('diag(λ₁ ≥ … ≥ λ₆₄ ≥ 0)')));
        case 'ml': return tex.block(tex.eq(tex.sym('Σ'), tex.op(','), tex.sym('H'), tex.op(','), tex.sym('K'), T(' all symmetric')));
        case 'entry': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' })), tex.eq(T(Math.abs(A[0][1] - A[1][0]) < 0.05 ? lam : 'not symmetric', { color: 'gold' })));
        case 'play': {
          const k = Math.abs(e.values[1]) > 1e-6 ? Math.abs(e.values[0] / e.values[1]) : Infinity;
          return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' })), tex.eq(T(lam, { color: 'gold' })), tex.eq(T(shapeWord(e.values))), tex.eq(T('λmax/λmin = ' + (Number.isFinite(k) ? LA.fmt(k) : '∞'))), tex.eq(T('xᵀAx at probe = ' + LA.fmt(qf(A, s.probe)), { color: 'v' })));
        }
        default: return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('Q'), tex.sym('Λ'), tex.sym('Q', { sup: 'T' })));
      }
    },

    steps: [
      { title: 'Recall: symmetric means perpendicular axes', duration: 3,
        narration: 'A = [[2,1],[1,2]] is <b>symmetric</b>: the 1s mirror across the diagonal. Its eigenvectors, the gold ' + gold('q₁') + ' = [1;1]/√2 (λ = 3) and ' + gold('q₂') + ' = [1;−1]/√2 (λ = 1), are perpendicular, and the unit circle becomes a teal ellipse whose axes are exactly those directions.',
        setup(s) { s.scene = 'ellipse'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.4 }); s.A = [[2, 0], [0, 2]]; s.b = 1; s.morph = 0; s.showDims = true; s.showCols = false; s.panel = 'sym'; },
        at(s, t) { s.morph = seg(t, 0.05, 0.85); } },

      { title: 'Not symmetric, not perpendicular', duration: 3,
        narration: 'Compare [[3,1],[0,2]] from p4-02: its gold eigenlines meet at 135°, and its ellipse’s own axes (dashed) are <i>not</i> its eigenvectors. Symmetry is what buys perpendicular eigenvectors, always, in any dimension. That fact has a name: the <b>spectral theorem</b>.',
        setup(s) { s.scene = 'compare'; s.chip = 0; s.panel = 'compare'; },
        at(s, t) { s.chip = seg(t, 0.55, 0.85); } },

      { title: 'A = QΛQᵀ: turn, stretch, turn back', duration: 5,
        narration: 'Because the eigenvectors are orthonormal, the change of basis is a rotation and its inverse is a transpose: <b>A = QΛQᵀ</b>. Three moves: Qᵀ turns the gold axes onto the grid axes, Λ stretches by 3 and 1 along them, Q turns back. The same PDP⁻¹ from p4-03, with no inverse to compute.',
        setup(s) { s.scene = 'qlq'; s.stage = 0; s.panel = 'qlq'; },
        at(s, t) { s.stage = 3 * seg(t, 0.04, 0.96, 'linear'); } },

      { title: 'The quadratic form: a number at every point', duration: 4, interactive: true,
        narration: 'Multiply x on both sides: the <b>quadratic form</b> xᵀAx = 2x² + 2xy + 2y². It is a single number for each point of the plane. Color the plane by it: the level curves, where the number is constant, are ellipses with the same gold axes. Drag the yellow probe and read the value.',
        setup(s) { s.scene = 'form'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 2.4 }); s.A = [[2, 0], [0, 2]]; s.b = 1; s.heatA = 0; s.probe = [1.2, 0.4]; s.panel = 'form'; },
        at(s, t) { s.heatA = seg(t, 0, 0.5); } },

      { title: 'The surface: a bowl', duration: 4.5,
        narration: 'Lift the number into height: z = xᵀAx is a <b>bowl</b>. Along the gold ' + gold('q₁') + ' it rises as 3r², steep; along ' + gold('q₂') + ' as 1r², gentle. The eigenvalues are the bowl’s two curvatures and the eigenvectors are the directions in which you feel them.',
        setup(s) { s.scene = 'surface'; s.view = '3d'; s.cam = surfCam(); s.grow = 0; s.showProbe = false; s.cross = false; s.showTicks = false; s.showWord = false; s.showB = false; s.panel = 'bowl'; },
        at(s, t) { s.grow = seg(t, 0, 0.6); LA.cam.tween(s, { yaw: 0.05 }, { yaw: 0.25 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Slide the off-diagonal: trough, then saddle', duration: 5, interactive: true,
        narration: 'Let A = [[2,b],[b,2]]; its eigenvalues are 2 + b and 2 − b. At b = 2 one eigenvalue hits 0: the bowl flattens into a trough with a level floor along q₂. Past that, b = 3 gives λ = −1: the surface drops along q₂ while rising along q₁, a saddle, and the level curves become hyperbolas.',
        setup(s) { s.grow = 1; s.b = 1; s.showTicks = true; s.showWord = true; s.showB = true; s.panel = 'slide'; },
        at(s, t) { s.b = 1 + 2 * seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Misconception: positive entries mean positive definite', duration: 3.5, misconception: true,
        narration: '✕ “All entries are positive, so xᵀAx is positive.” Take [[1,2],[2,1]]: every entry positive, yet along [1;−1] the form is 1 − 4 + 1 = −2 (−1 for the unit-length [1;−1]/√2 marked on the surface). Its eigenvalues are 3 and −1: a saddle. Definiteness lives in the eigenvalues, not in the entries.',
        setup(s) { s.A = [[1, 0], [0, 1]]; s.b = 2; s.showB = false; s.showTicks = true; s.showWord = true; s.showProbe = true; s.probe = [Math.SQRT1_2, -Math.SQRT1_2]; s.cross = false; s.panel = 'mis'; },
        at(s, t) { s.cross = t > 0.45; } },

      { title: 'Three words for three shapes', duration: 3,
        narration: 'All eigenvalues positive: <b>positive definite</b>, a bowl, one lowest point. Some zero, none negative: <b>positive semi-definite</b>, a trough, a whole line of lowest points. Mixed signs: <b>indefinite</b>, a saddle, no lowest point at all. The signs of λ are the entire story.',
        setup(s) { s.scene = 'three'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.showProbe = false; s.cross = false; s.panel = 'words'; },
        at(s, t) { } },

      { title: 'In 64 dimensions: 64 signs', duration: 3,
        narration: 'A symmetric 64×64 matrix has 64 perpendicular eigenvectors and 64 eigenvalues. We cannot draw the bowl, but we can draw the 64 numbers as bars and as a strip: no negative eigenvalue means nothing curves down: a bowl, with a flat trough direction wherever one is 0. The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'n64'; s.reveal = 0; s.panel = 'n64'; },
        at(s, t) { s.reveal = seg(t, 0.05, 0.8, 'linear'); } },

      { title: 'Why ML cares: three symmetric matrices', duration: 3.5, mlTie: true,
        narration: 'Covariance (p4-09) is symmetric PSD: its bowl is the data’s spread. The Hessian of a loss (p5-02) is symmetric: bowl means convex, saddle means a saddle point, and its largest λ caps the learning rate. Gram and kernel matrices (p3-07) are symmetric PSD. One picture, three objects.',
        setup(s) { s.scene = 'ml'; s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Symmetric means perpendicular eigenvectors and A = QΛQᵀ: turn, stretch, turn back. xᵀAx is a surface whose curvatures are the eigenvalues. Positive λ: bowl; a zero: trough; mixed: saddle. Entries do not decide; eigenvalues do.',
        cards: ['<b>Perpendicular axes.</b> A = Aᵀ ⇒ orthonormal eigenvectors, A = QΛQᵀ.', '<b>Curvatures = λ.</b> z = xᵀAx = Σ λᵢ (x·qᵢ)²: steep along big λ, gentle along small.', '<b>Signs decide.</b> All > 0 bowl, some 0 trough, mixed saddle — never the entries.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Set a, b, c (A = [[a, b], [b, c]] stays symmetric), drag the green and red column tips or the yellow probe, and switch between the flat heat map and the 3D surface.</p>',
      setup(s) { s.scene = 'play'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3 }); s.A = [[2, 0], [0, 2]]; s.b = 1; s.probe = [1, 0.5]; s.panel = 'play'; s.showContours = true; },
      controls: [
        { type: 'slider', bind: 'A[0][0]', min: -3, max: 4, step: 0.05, label: 'a', color: 'i' },
        { type: 'slider', bind: 'b', min: -3, max: 3, step: 0.05, label: 'b (both off-diagonals)', color: 'v' },
        { type: 'slider', bind: 'A[1][1]', min: -3, max: 4, step: 0.05, label: 'c', color: 'j' },
        { type: 'toggle', bind: 'showContours', label: 'Level curves' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Bowl [[2,1],[1,2]]', values: { A: [[2, 0], [0, 2]], b: 1 } },
          { label: 'Trough [[2,2],[2,2]]', values: { A: [[2, 0], [0, 2]], b: 2 } },
          { label: 'Saddle [[1,2],[2,1]]', values: { A: [[1, 0], [0, 1]], b: 2 } },
          { label: 'Round bowl I', values: { A: [[1, 0], [0, 1]], b: 0 } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: 'Flat (2D)', apply(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3 }); } },
          { label: '3D surface', apply(s) { s.view = '3d'; s.cam = surfCam(); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the b slider under the picture until the surface is a saddle.',
        setup(s) { s.scene = 'surface'; s.view = '3d'; s.cam = surfCam(); s.A = [[2, 0], [0, 2]]; s.b = 1; s.grow = 1; s.showB = true; s.showTicks = true; s.showWord = false; s.showProbe = false; s.panel = 'slide'; },
        judge(s) { const l = eig(Aeff(s)).values; const p = l[0] * l[1]; return { ok: p < -0.05, progress: LA.clamp01(1 - (p + 0.05) / 3) }; },
        hint: 'One eigenvalue must go negative: push |b| past 2.',
        solution(s) { s.b = 3; } },
      { kind: 'manipulate',
        prompt: 'Drag the green and red column tips (the matrix stays symmetric) so the ellipse’s long axis points along [1; 1].',
        setup(s) { s.scene = 'ellipse'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 4.3 }); s.A = [[3, 0], [0, 1]]; s.b = 0; s.morph = 1; s.showDims = false; s.showCols = true; s.panel = 'sym'; },
        judge(s) {
          const e = eig(Aeff(s)), q = e.vectors[0];
          const ang = Math.acos(Math.min(1, Math.abs(q[0] + q[1]) / Math.SQRT2)) * 180 / Math.PI;
          const gap = e.values[0] - e.values[1];
          return { ok: ang < LA.TOL.angleDeg && gap > 0.1, progress: Math.max(0, 1 - ang / 45) * (gap > 0.1 ? 1 : 0.6), hint: gap <= 0.1 ? 'Make one axis clearly longer than the other.' : undefined };
        },
        hint: 'Equal diagonal entries with a positive off-diagonal tilt the long axis to 45°.',
        solution(s) { s.A = [[2, 0], [0, 2]]; s.b = 1; } },
      { kind: 'entry',
        prompt: 'Type a symmetric matrix with eigenvalues 3 and 1 whose eigenvectors are [1; 1] and [1; −1].',
        bind: 'Atyped', answer: [[2, 1], [1, 2]], tol: LA.TOL.entry, label: 'A',
        setup(s) { s.scene = 'entry'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.4 }); s.Atyped = [[1, 0], [0, 1]]; s.panel = 'entry'; } },
    ],
  });

  function renderPlay(g, s, ctx) {
    const A = Aeff(s), e = eig(A);
    if (s.view === '3d') {
      surfaceScene(g, A, { res: 24, levels: s.showContours ? undefined : [] });
      const p = s.probe, z = qf(A, p) * ZS;
      g.line([p[0], p[1], 0], [p[0], p[1], z], { color: 'v', dashed: true, width: 1.4 });
      g.point([p[0], p[1], z], { r: 6, color: 'v', label: 'xᵀAx = ' + LA.fmt(qf(A, p)), labelColor: 'v' });
      g.handle('probe3', [p[0], p[1], 0], (q, st) => { st.probe = [LA.clamp(q[0], -1.5, 1.5), LA.clamp(q[1], -1.5, 1.5)]; }, { color: 'v', plane: 'xy' });
    } else {
      const bnd = { x0: -2.6, x1: 2.6, y0: -2.4, y1: 2.4 };
      const hi = Math.max(1, Math.abs(e.values[0]), Math.abs(e.values[1])) * 6;
      g.heat2d(p => qf(A, p), bnd, { res: 64, range: [-hi, hi], ramp: t => t < 0.5 ? LA.color.mix(LA.color.neg, LA.color.bg, t * 2) : LA.color.heat((t - 0.5) * 2), alpha: 0.8 });
      if (s.showContours) g.contours(p => qf(A, p), [-4, -2, -1, 1, 2, 4], bnd, { res: 56, colors: ['neg', 'neg', 'neg', 'white', 'white', 'white'], width: 1.2, alpha: 0.85 });
      goldAxes(g, A, { lines: true, unit: true });
      colHandles(g, s);
      g.point(s.probe, { r: 6, color: 'v', label: 'xᵀAx = ' + LA.fmt(qf(A, s.probe)), labelColor: 'v' });
      g.handle('probe', s.probe, (p, st) => { st.probe = [LA.clamp(p[0], -2.6, 2.6), LA.clamp(p[1], -2.4, 2.4)]; }, { color: 'v' });
    }
    g.hud(shapeWord(e.values), 12, 20, { color: e.values[1] < -0.05 ? 'bad' : 'gold', size: 14, bold: true });
  }
})();
